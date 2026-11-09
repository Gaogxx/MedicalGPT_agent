"""模型提供方：OpenAI 兼容（vLLM 本地 MedicalGPT / 百川 / DeepSeek / 通义 …）+ 离线 Mock + 故障注入。"""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..resilience.errors import classify_http_status
from ..types import AgentError, ErrorKind, LLMResponse, ToolCall
from .parser import parse_tool_calls


class OpenAICompatProvider:
    def __init__(self, name: str, base_url: str, model: str, api_key_env: str = "", timeout: float = 60.0,
                 price_in: float = 0.0, price_out: float = 0.0, extra_body: dict[str, Any] | None = None,
                 client: Any = None) -> None:
        import httpx
        self.name, self.model = name, model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        self.price_in, self.price_out = price_in, price_out   # 每百万 token 价格（按官网填写）
        self.extra_body = extra_body or {}
        self.http = client or httpx.AsyncClient(timeout=timeout)

    @property
    def configured(self) -> bool:
        return bool(self.api_key) or self.base_url.startswith("http://127.0.0.1") or "localhost" in self.base_url

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                   temperature: float = 0.3, max_tokens: int = 1024) -> LLMResponse:
        if not self.configured:
            raise AgentError(ErrorKind.AUTH, f"{self.name}: API key not configured", source=f"llm:{self.name}")
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature,
                                "max_tokens": max_tokens, **self.extra_body}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        t0 = time.perf_counter()
        r = await self.http.post(f"{self.base_url}/chat/completions", json=body,
                                 headers={"Authorization": f"Bearer {self.api_key or 'dummy'}"})
        if r.status_code >= 400:
            raise classify_http_status(r.status_code, r.text, source=f"llm:{self.name}", headers=dict(r.headers))
        data = r.json()
        try:
            choice = data["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError) as exc:
            raise AgentError(ErrorKind.MODEL, f"malformed response: {str(data)[:200]}", source=f"llm:{self.name}") from exc
        known = {t["function"]["name"] for t in tools or []}
        calls, rest = parse_tool_calls(msg.get("content") or "", msg.get("tool_calls"), known)
        finish = choice.get("finish_reason", "stop")
        if calls and not msg.get("tool_calls"):
            finish = "tool_calls_recovered"   # 工具调用是从正文里救回来的（vLLM hermes 解析失败的泄漏）
        usage = data.get("usage", {})
        cost = (usage.get("prompt_tokens", 0) * self.price_in + usage.get("completion_tokens", 0) * self.price_out) / 1e6
        return LLMResponse(rest, calls, self.name, finish, usage,
                           (time.perf_counter() - t0) * 1000, cost)

    async def complete_json(self, prompt: str) -> dict[str, Any]:
        resp = await self.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=128)
        try:
            return json.loads(resp.content[resp.content.find("{"): resp.content.rfind("}") + 1])
        except json.JSONDecodeError as exc:
            raise AgentError(ErrorKind.MODEL, "judge output not JSON", source=f"llm:{self.name}") from exc


class MockProvider:
    """离线脚本化模型：用于单测与混沌测试（验证编排/容错行为，不代表模型质量）。
    行为：首轮按关键词规划工具调用；拿到 observation 后用观察结果拼最终回答。"""

    def __init__(self, name: str = "mock", latency: float = 0.0, policy: Callable[..., LLMResponse] | None = None) -> None:
        self.name = name
        self.latency = latency
        self.policy = policy
        self.calls = 0

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                   temperature: float = 0.3, max_tokens: int = 1024) -> LLMResponse:
        self.calls += 1
        if self.latency:
            await asyncio.sleep(self.latency)
        if self.policy:
            return self.policy(messages, tools)
        return _default_mock_policy(self.name, messages, tools)


def _default_mock_policy(name: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> LLMResponse:
    from ..knowledge import load_kb
    kb = load_kb()
    user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    names = {t["function"]["name"] for t in tools or []}
    if not tool_msgs and names:
        ents = kb.link_entities(user)
        calls = []
        if ents["symptom"] and "check_symptom" in names:
            calls.append(ToolCall("check_symptom", {"symptom": user}, "c1"))
        for d in ents["drug"][:2]:
            if "search_drug" in names:
                calls.append(ToolCall("search_drug", {"name": d}, f"c{len(calls) + 1}"))
        if ents["disease"] and "search_guideline" in names:
            calls.append(ToolCall("search_guideline", {"topic": ents["disease"][0]}, f"c{len(calls) + 1}"))
        if calls:
            return LLMResponse("", calls, name)
    parts = []
    for m in tool_msgs:
        try:
            o = json.loads(m["content"])
        except json.JSONDecodeError:
            continue
        if o.get("possible_causes"):
            parts.append(f"可能原因：{'、'.join(o['possible_causes'])}。{o.get('recommendation', '')}。")
        elif o.get("found") and o.get("name"):
            parts.append(f"{o['name']}：适应症 {o['indications']}；不良反应 {o['side_effects']}；用法 {o['dosage']}。")
        elif o.get("found") and o.get("content"):
            parts.append(f"{o['topic']}：{o['content']}（{o['source']}）")
        elif o.get("error") or o.get("found") is False:
            parts.append(f"未找到相关信息（{o.get('error') or o.get('query', '')}）。")
    text = "".join(parts) or "这是一个一般性健康问题，建议保持规律作息、均衡饮食。"
    return LLMResponse(text + "以上仅供参考，请咨询医生。", [], name)


@dataclass
class FaultInjector:
    """按概率注入故障：network / timeout / rate_limit / external / auth / silent_empty / silent_truncated / tool_leak。
    seed 固定 → 混沌测试可复现。"""

    rates: dict[str, float] = field(default_factory=dict)
    seed: int = 0
    timeout_s: float = 0.05
    _rng: random.Random = field(init=False)
    injected: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def pick(self) -> str | None:
        r = self._rng.random()
        acc = 0.0
        for kind, p in self.rates.items():
            acc += p
            if r < acc:
                self.injected[kind] = self.injected.get(kind, 0) + 1
                return kind
        return None

    async def before(self, source: str) -> str | None:
        kind = self.pick()
        if kind is None:
            return None
        import httpx
        if kind == "network":
            raise httpx.ConnectError("injected connection reset")
        if kind == "timeout":
            await asyncio.sleep(self.timeout_s)
            raise asyncio.TimeoutError("injected timeout")
        if kind == "rate_limit":
            raise AgentError(ErrorKind.RATE_LIMIT, "HTTP 429 injected", source=source, status=429, retry_after=0.01)
        if kind == "external":
            raise AgentError(ErrorKind.EXTERNAL, "HTTP 503 injected", source=source, status=503)
        if kind == "auth":
            raise AgentError(ErrorKind.AUTH, "HTTP 401 injected", source=source, status=401)
        return kind  # silent_* / tool_leak：调用“成功”，由调用方篡改结果


class FaultyProvider:
    """给任意 provider 套故障注入。"""

    def __init__(self, inner: Any, injector: FaultInjector) -> None:
        self.inner, self.injector = inner, injector
        self.name = inner.name

    async def chat(self, messages, tools=None, temperature=0.3, max_tokens=1024) -> LLMResponse:
        kind = await self.injector.before(f"llm:{self.name}")
        resp = await self.inner.chat(messages, tools, temperature, max_tokens)
        if kind == "silent_empty":
            return LLMResponse("", [], self.name)
        if kind == "silent_truncated":
            return LLMResponse(resp.content[: max(4, len(resp.content) // 3)], [], self.name, "length")
        if kind == "tool_leak" and resp.tool_calls:
            c = resp.tool_calls[0]
            return LLMResponse("ptom\n" + json.dumps({"name": c.name, "arguments": c.arguments}, ensure_ascii=False) + "}",
                               [], self.name)
        return resp
