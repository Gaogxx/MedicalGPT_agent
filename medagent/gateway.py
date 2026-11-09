"""OpenAI 兼容的路由网关：让 DeepSeek Harness（DSH）作为主运行时，同时用上微调过的 MedicalGPT。

    DSH（agent 循环 / 会话 / 压缩 / Web·CLI）
      ├── 模型请求 → 本网关 http://127.0.0.1:9000/v1  ──→ local：微调 MedicalGPT（vLLM :8000）
      │                │ 分诊 → RouteLLM 路由 → 熔断/重试/跨层兜底   └→ strong：Baichuan-M3 / DeepSeek
      │                │ 泄漏的工具调用 → 修正为标准 tool_calls
      │                └ 最终回答 → 静默失败校验（必要时升级强模型重答）→ 输出安全关
      └── 工具 → dsh-mcp-client → medagent MCP Server（tools / resources）

为什么这样接：DSH 只认“一个 OpenAI 兼容的模型端点”，路由、容错、校验放在端点后面对 DSH 完全透明，
不需要改 DSH 源码（RouteLLM 官方也是以 OpenAI 兼容服务的形式提供路由）。

工具名适配：DSH 经 MCP 暴露的工具名是 mcp__medical__search_drug，而 MedicalGPT 训练时见到的是 search_drug。
网关把发给模型的工具 schema 与历史 tool_calls 去掉命名空间前缀，模型返回后再加回去——否则微调学到的工具调用能力
在 MCP 化之后会因为名字对不上而失效。

启动：python -m medagent.gateway --port 9000   （DSH 的 settings.yaml 指向这里，见 runtime/dsh-medical-plugins/）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from .router.policy import SafetyFirstRouter, TieredRouter
from .safety import EMERGENCY_BANNER, assess, guard_output
from .types import AgentError, LLMResponse
from .verify.attribution import attribute
from .verify.detectors import detect, multi_intent
from .verify.jev import JevClient, JevVerifier

GUARD_FIXABLE = {"emergency_missing", "prescription_in_high_risk"}
CONSULT_TOOL = "complex_consult"      # medagent.mcp_servers.consult_server：Python 编排器包成的 MCP 工具
CONSULT_MAX_CALLS = 2                 # 同一轮里最多发起 2 次（第 2 次 = 中断后用相同参数续跑）


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI 多段内容
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


@dataclass
class NameMap:
    """mcp__<server>__<tool> ⇄ <tool>"""
    to_full: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_tools(cls, tools: list[dict[str, Any]] | None) -> "NameMap":
        m = cls()
        for t in tools or []:
            full = t.get("function", {}).get("name", "")
            raw = full.split("__", 2)[-1] if full.startswith("mcp__") else full
            m.to_full.setdefault(raw, full)
        return m

    def raw(self, full: str) -> str:
        return full.split("__", 2)[-1] if full.startswith("mcp__") else full

    def full(self, raw: str) -> str:
        return self.to_full.get(raw, raw)

    def strip_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return tools
        out = []
        for t in tools:
            t = json.loads(json.dumps(t))
            t["function"]["name"] = self.raw(t["function"]["name"])
            out.append(t)
        return out

    def strip_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for m in messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                m = json.loads(json.dumps(m))
                for tc in m["tool_calls"]:
                    tc["function"]["name"] = self.raw(tc["function"]["name"])
            out.append(m)
        return out


class RoutingGateway:
    def __init__(self, pool: Any, router: SafetyFirstRouter, audit_path: str | None = None,
                 verifier: JevVerifier | None = None) -> None:
        self.pool, self.router = pool, router
        self.verifier = verifier or JevVerifier(JevClient())   # JEV_BASE_URL 未配置时只用规则
        self.audit_path = audit_path
        self.stats = {"requests": 0, "strong": 0, "local": 0, "regenerated": 0, "leak_fixed": 0, "fallback": 0,
                      "consult": 0}

    @staticmethod
    def _turn(messages: list[dict[str, Any]]) -> tuple[str, str, bool, list[Any]]:
        """返回（当前用户问题, 历史文本, 是否本轮首个助手回复, 本轮工具结果）。"""
        idx = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
        query = _text(messages[idx]["content"]) if idx >= 0 else ""
        history = " ".join(_text(m.get("content")) for m in messages[:max(idx, 0)] if m.get("role") in ("user", "assistant"))
        after = messages[idx + 1:] if idx >= 0 else []
        first = not any(m.get("role") == "assistant" for m in after)
        obs = []
        for m in after:
            if m.get("role") == "tool":
                try:
                    obs.append(json.loads(_text(m.get("content"))))
                except json.JSONDecodeError:
                    obs.append({"text": _text(m.get("content"))})
        return query, history, first, obs

    @staticmethod
    def _consult_state(messages: list[dict[str, Any]], names: "NameMap") -> tuple[int, dict[str, Any] | None, Any]:
        """本轮（最后一条用户消息之后）复杂问诊工具的情况：(已发起次数, 最近一次的参数, 最近一次的结果)。
        结果是解析后的 JSON；工具报错（DSH 传回的是错误文本）时是字符串。"""
        idx = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
        calls: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        results: dict[str, Any] = {}
        for m in messages[idx + 1:]:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    if names.raw(tc.get("function", {}).get("name", "")) == CONSULT_TOOL:
                        try:
                            args = json.loads(tc["function"].get("arguments") or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        calls[tc.get("id", "")] = args
                        order.append(tc.get("id", ""))
            elif m.get("role") == "tool" and m.get("tool_call_id") in calls:
                text = _text(m.get("content"))
                try:
                    results[m["tool_call_id"]] = json.loads(text)
                except json.JSONDecodeError:
                    results[m["tool_call_id"]] = text
        if not order:
            return 0, None, None
        last = order[-1]
        return len(order), calls[last], results.get(last)

    def _consult_call(self, body: dict[str, Any], names: "NameMap", args: dict[str, Any], meta: dict[str, Any],
                      query: str) -> tuple[dict[str, Any], dict[str, Any]]:
        self.stats["consult"] += 1
        call = {"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                "function": {"name": names.full(CONSULT_TOOL), "arguments": json.dumps(args, ensure_ascii=False)}}
        meta.update(served_tier="orchestrator", model="medagent-consult")
        self._audit(query, meta)
        return self._response(body, "", [call], meta), meta

    async def complete(self, body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        self.stats["requests"] += 1
        messages, tools = body.get("messages", []), body.get("tools")
        query, history, first, obs = self._turn(messages)
        a = assess(query)
        decision = self.router.route(query, history, a)
        names = NameMap.from_tools(tools)
        msgs, raw_tools = names.strip_messages(messages), names.strip_tools(tools)
        kw = {k: body[k] for k in ("temperature", "max_tokens") if k in body}
        events: list[dict[str, Any]] = []
        meta: dict[str, Any] = {"tier": decision.tier, "reason": decision.reason, "fastpath": decision.fastpath and first,
                                "repairs": []}
        tier = decision.tier
        if CONSULT_TOOL in names.to_full and not (a.level == "emergency" or a.crisis):
            n_calls, last_args, result = self._consult_state(messages, names)
            if first and multi_intent(query):
                # 多意图问题：不调模型，直接让 DSH 调“复杂问诊”工具——拆子问题、worker 隔离、逐步检查点都在工具里完成
                meta.update(tier="orchestrator", reason="consult:multi_intent")
                return self._consult_call(body, names, {"question": query, "history": history[-3000:]}, meta, query)
            if isinstance(result, dict) and result.get("status") == "done" and result.get("answer"):
                # 工具返回的回答已经在编排器里校验 / 修复过：原样交给用户（只过输出安全关），不再让模型转述一遍
                meta.update(tier="orchestrator", reason="consult:answer", served_tier="orchestrator",
                            model="medagent-consult", consult={k: result.get(k) for k in
                                                               ("run_id", "sub_questions", "repairs", "resumed", "replayed")})
                self._audit(query, meta)
                return self._response(body, guard_output(result["answer"], a), [], meta), meta
            if last_args is not None and not isinstance(result, dict) and n_calls < CONSULT_MAX_CALLS:
                # 工具报错 / 超时（MCP Server 重启、模型超时……）：用相同参数再发一次 → 同一个 run_id → 从检查点续跑
                meta.update(tier="orchestrator", reason="consult:resume_after_error")
                meta["repairs"].append("consult_retry_resume")
                return self._consult_call(body, names, last_args, meta, query)
            # 其余情况（暂停、重试后仍失败）：交给模型用普通工具继续
        if tier == "small" and not first:
            # 本轮已经进入工具循环（之前升级到了 local 并调用了工具）：不能再回落到不带工具的小模型
            tier = "local"
            meta["tier"], meta["reason"] = "local", decision.reason + "|in_tool_loop"
        tools_for_tier = None if tier == "small" else raw_tools
        try:
            resp = await self.pool.generate(tier, msgs, tools_for_tier, events=events, **kw)
        except AgentError as err:
            self.stats["fallback"] += 1
            meta.update(error=err.to_dict(), attribution=self.verifier.attribute(events))
            from .templates import kb_fallback_answer
            return self._response(body, guard_output(kb_fallback_answer(query, a, history), a), [], meta), meta
        if not resp.tool_calls:
            findings = detect(resp.content, query=query, assessment=a, observations=obs or None, history=history,
                              finish_reason=resp.finish_reason)
            meta["findings"] = [f.code for f in findings]
            verdict = await asyncio.to_thread(self.verifier.verify, query, resp.content,
                                              [f for f in findings if f.code not in GUARD_FIXABLE], obs, a)
            meta["verdict"] = {"blocked": verdict.blocked, "reason": verdict.reason, "confidence": round(verdict.confidence, 3),
                               "source": verdict.source, "needs_review": verdict.needs_review}
            if verdict.blocked and tier == "small":
                # 小模型答砸：先升到 local（带工具），local 再不行由下面一步升 strong
                self.stats["regenerated"] += 1
                meta["repairs"].append("escalate_local:" + verdict.reason)
                try:
                    resp, tier = await self.pool.generate("local", msgs, raw_tools, events=events, **kw), "local"
                    findings = detect(resp.content, query=query, assessment=a, observations=obs or None, history=history,
                                      finish_reason=resp.finish_reason) if not resp.tool_calls else []
                    verdict = await asyncio.to_thread(self.verifier.verify, query, resp.content,
                                                      [f for f in findings if f.code not in GUARD_FIXABLE], obs, a) \
                        if not resp.tool_calls else verdict.__class__(False, "tool_calls", 1.0, "rules")
                except AgentError as err:
                    meta["repairs"].append(f"local_unavailable:{err.kind.value}")
            if verdict.blocked and tier != "strong" and not resp.tool_calls:
                # 回答被拦截（规则或 Jev）：同一请求升级到强模型重答一次（强模型也可以选择先调工具）
                self.stats["regenerated"] += 1
                meta["repairs"].append("regenerate_strong:" + verdict.reason)
                try:
                    resp2 = await self.pool.generate("strong", msgs, raw_tools, events=events, allow_cross_tier=False, **kw)
                    resp, tier = resp2, "strong"
                except AgentError as err:
                    meta["repairs"].append(f"strong_unavailable:{err.kind.value}")
        if resp.finish_reason == "tool_calls_recovered":
            self.stats["leak_fixed"] += 1
            meta["repairs"].append("leaked_tool_call_converted")
        self.stats[tier] = self.stats.get(tier, 0) + 1
        meta.update(model=resp.model, served_tier=tier, attribution=self.verifier.attribute(events))
        calls = [{"id": c.id or f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                  "function": {"name": names.full(c.name), "arguments": json.dumps(c.arguments, ensure_ascii=False)}}
                 for c in resp.tool_calls]
        content = resp.content or ""
        if not calls:
            content = guard_output(content, a)
            if meta["fastpath"] and not body.get("stream") and EMERGENCY_BANNER not in content and not content.startswith("⚠️"):
                content = f"{EMERGENCY_BANNER}\n\n{content}"
        out = self._response(body, content, calls, meta, usage=resp.usage)
        self._audit(query, meta)
        return out, meta

    @staticmethod
    def _response(body: dict[str, Any], content: str, calls: list[dict[str, Any]], meta: dict[str, Any],
                  usage: dict[str, int] | None = None) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": content or None}
        if calls:
            msg["tool_calls"] = calls
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()),
                "model": body.get("model", "medical-router"),
                "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
                "usage": usage or {}, "medagent": meta}

    async def stream(self, body: dict[str, Any]) -> AsyncIterator[str]:
        """SSE。急症且是本轮首个回复时，先立刻推送急救横幅（不等模型），这就是 P0 快路径。
        其余内容在校验与输出关之后才推送——宁可晚一点，也不把未校验的内容流给用户。"""
        cid, created, model = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time()), body.get("model", "medical-router")

        def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
            return "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                                          "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]},
                                         ensure_ascii=False) + "\n\n"

        query, _, first, _ = self._turn(body.get("messages", []))
        a = assess(query)
        banner_sent = False
        yield chunk({"role": "assistant"})
        if first and (a.level == "emergency" or a.crisis):
            yield chunk({"content": EMERGENCY_BANNER + "\n\n"})
            banner_sent = True
        resp, _ = await self.complete({**body, "stream": False})
        msg = resp["choices"][0]["message"]
        content = msg.get("content") or ""
        if banner_sent:
            content = content.replace(EMERGENCY_BANNER, "").lstrip()
        for i in range(0, len(content), 24):
            yield chunk({"content": content[i:i + 24]})
        for i, tc in enumerate(msg.get("tool_calls") or []):
            yield chunk({"tool_calls": [{"index": i, **tc}]})
        yield chunk({}, resp["choices"][0]["finish_reason"])
        yield "data: [DONE]\n\n"

    def _audit(self, query: str, meta: dict[str, Any]) -> None:
        if self.audit_path:
            with open(self.audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.time(), "query": query[:200], **meta}, ensure_ascii=False, default=str) + "\n")


def create_app(gateway: RoutingGateway):
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Route

    async def models(_: Request) -> JSONResponse:
        return JSONResponse({"object": "list", "data": [{"id": "medical-router", "object": "model", "owned_by": "medagent"}]})

    async def chat(request: Request):
        body = await request.json()
        if body.get("stream"):
            return StreamingResponse(gateway.stream(body), media_type="text/event-stream")
        resp, meta = await gateway.complete(body)
        return JSONResponse(resp, headers={"x-medagent-tier": str(meta.get("served_tier", meta["tier"])),
                                           "x-medagent-route": meta["reason"]})

    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"stats": gateway.stats,
                             "breakers": {n: b.state() for n, b in gateway.pool.breakers.items()}})

    return Starlette(routes=[Route("/v1/models", models), Route("/v1/chat/completions", chat, methods=["POST"]),
                             Route("/healthz", health)])


def main() -> None:
    import uvicorn

    from .cli import _pool, _router
    from .triage import load_learned_triage
    ap = argparse.ArgumentParser("medagent-gateway")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--mock", action="store_true", help="用 Mock 模型（联调 DSH 时不需要 GPU / API）")
    ap.add_argument("--audit", default=".medagent_state/gateway_audit.jsonl")
    args = ap.parse_args()
    load_learned_triage()
    import os
    os.makedirs(os.path.dirname(args.audit) or ".", exist_ok=True)
    pool = _pool(args.mock)
    uvicorn.run(create_app(RoutingGateway(pool, _router(pool), args.audit)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
