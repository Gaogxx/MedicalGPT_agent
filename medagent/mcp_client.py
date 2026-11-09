"""MCP 客户端工具箱：取代原来的进程内 tool registry。用到 MCP Server 的三类能力——
  tools（模型决定调用）、resources（宿主按 URI 直接读知识条目）、prompts（服务端维护的提示词模板），
  外加 server instructions。注意：MCP Server 不“调度”任何东西，调度（选哪个工具、何时调用、失败怎么办）
  始终在宿主（本编排器或 DSH）里；Server 只负责暴露能力并执行被调用的那一个操作。

- 工具定义来自 MCP Server 的 tools/list（单一真源），转成 OpenAI function schema 给模型
- 每次调用：工具存在性检查（MODEL：模型编造了工具名）→ 必填参数预检（PARAM，不浪费一次网络往返）
  → 有副作用的工具自动补 idempotency_key 并做客户端幂等 → 熔断 → 超时 → 分类重试 → 结果归因
- 支持三种连接：进程内 MCPServer（测试）、stdio 子进程、streamable-http URL
"""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from mcp import Client, StdioServerParameters

from .resilience.breaker import BreakerConfig, CircuitBreaker, InMemoryBreakerStore
from .resilience.errors import classify_exception, classify_tool_result
from .resilience.idempotency import InMemoryIdempotencyStore, make_key, run_idempotent
from .resilience.retry import RetryPolicy, call_with_retry
from .types import AgentError, ErrorKind, ToolResult


@dataclass
class ToolInfo:
    name: str
    server: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool


@dataclass
class MCPToolbox:
    servers: dict[str, Any]                      # name -> MCPServer 实例 | StdioServerParameters | URL
    breaker_store: Any = field(default_factory=InMemoryBreakerStore)
    breaker_cfg: BreakerConfig = field(default_factory=lambda: BreakerConfig(min_calls=4, open_seconds=10))
    idem_store: Any = field(default_factory=InMemoryIdempotencyStore)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    call_timeout: float = 15.0
    fault_injector: Any = None                   # 混沌测试用（FaultInjector）
    tools: dict[str, ToolInfo] = field(default_factory=dict)
    resources: dict[str, list[str]] = field(default_factory=dict)
    prompts: dict[str, list[str]] = field(default_factory=dict)
    instructions: dict[str, str] = field(default_factory=dict)
    _clients: dict[str, Client] = field(default_factory=dict)
    _stack: AsyncExitStack | None = None
    _breakers: dict[str, CircuitBreaker] = field(default_factory=dict)

    async def __aenter__(self) -> "MCPToolbox":
        self._stack = AsyncExitStack()
        for name, target in self.servers.items():
            client = await self._stack.enter_async_context(Client(target))
            self._clients[name] = client
            listed = await client.list_tools()
            for t in listed.tools:
                ro = bool(t.annotations and t.annotations.read_only_hint)
                self.tools[t.name] = ToolInfo(t.name, name, t.description or "", t.input_schema, ro)
            caps = client.server_capabilities
            if caps.resources is not None:
                res = await client.list_resources()
                tpl = await client.list_resource_templates()
                self.resources[name] = [r.uri for r in res.resources] + [t.uri_template for t in tpl.resource_templates]
            if caps.prompts is not None:
                self.prompts[name] = [p.name for p in (await client.list_prompts()).prompts]
            self.instructions[name] = client.instructions or ""
        return self

    async def read_resource(self, uri: str, server: str | None = None) -> Any:
        """读 MCP 资源（kb://...）。JSON 内容自动解析；找不到返回 None（交给调用方兜底）。"""
        client = self._clients[server or next(iter(self._clients))]
        try:
            res = await asyncio.wait_for(client.read_resource(uri), timeout=self.call_timeout)
        except Exception:  # noqa: BLE001 —— 资源读取失败不应中断主流程
            return None
        text = "".join(getattr(c, "text", "") or "" for c in res.contents)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    async def get_prompt(self, name: str, arguments: dict[str, str] | None = None, server: str | None = None) -> str | None:
        """取 MCP 提示词模板并拼成文本；服务端没有该提示词时返回 None。"""
        srv = server or next((k for k, v in self.prompts.items() if name in v), None)
        if srv is None:
            return None
        res = await self._clients[srv].get_prompt(name, arguments or {})
        return "\n".join(getattr(m.content, "text", "") for m in res.messages)

    async def __aexit__(self, *exc: Any) -> None:
        if self._stack:
            await self._stack.aclose()

    @staticmethod
    def stdio(command: str, args: list[str], env: dict[str, str] | None = None) -> StdioServerParameters:
        return StdioServerParameters(command=command, args=args, env=env)

    def openai_tools(self, subset: list[str] | None = None) -> list[dict[str, Any]]:
        out = []
        for t in self.tools.values():
            if subset is not None and t.name not in subset:
                continue
            schema = {k: v for k, v in t.input_schema.items() if k in ("type", "properties", "required")}
            if not t.read_only:  # idempotency_key 由运行时注入，不让模型编
                schema = json.loads(json.dumps(schema))
                schema.get("properties", {}).pop("idempotency_key", None)
                schema["required"] = [r for r in schema.get("required", []) if r != "idempotency_key"]
            out.append({"type": "function", "function": {"name": t.name, "description": t.description,
                                                         "parameters": schema}})
        return out

    def _breaker(self, name: str) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(f"tool:{name}", self.breaker_store, self.breaker_cfg)
        return self._breakers[name]

    def _precheck(self, name: str, args: dict[str, Any]) -> ToolInfo:
        info = self.tools.get(name)
        if info is None:
            raise AgentError(ErrorKind.MODEL, f"unknown tool {name!r}; available: {sorted(self.tools)}",
                             source=f"tool:{name}")
        missing = [r for r in info.input_schema.get("required", []) if r not in args and r != "idempotency_key"]
        if missing:
            raise AgentError(ErrorKind.PARAM, f"missing required {missing}", source=f"tool:{name}")
        return info

    async def _raw_call(self, info: ToolInfo, args: dict[str, Any]) -> Any:
        client = self._clients[info.server]
        injected = None
        if self.fault_injector is not None:
            injected = await self.fault_injector.before(f"tool:{info.name}")
        res = await asyncio.wait_for(client.call_tool(info.name, args), timeout=self.call_timeout)
        if injected == "lost_response":   # 服务端已经执行了，但响应在网络上丢了 —— 幂等要解决的正是这种情况
            import httpx
            raise httpx.ReadError("injected: response lost after server-side execution")
        text = "".join(getattr(c, "text", "") for c in res.content)
        err = classify_tool_result(info.name, bool(res.is_error), text, res.structured_content)
        if err is not None:
            raise err
        data = res.structured_content
        if data is None:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = {"text": text}
        if isinstance(data, dict) and set(data) == {"result"}:
            data = data["result"]
        if injected == "silent_empty":
            data = {}
        if data in ({}, None, ""):
            raise AgentError(ErrorKind.SILENT, "tool returned empty result", source=f"tool:{info.name}")
        return data

    async def call(self, name: str, args: dict[str, Any], *, scope: str = "",
                   events: list[dict[str, Any]] | None = None) -> ToolResult:
        t0 = time.perf_counter()
        attempts: list = []
        from_cache = False
        try:
            info = self._precheck(name, args)
            args = dict(args)
            if not info.read_only:
                args.setdefault("idempotency_key", make_key(scope, name, args)[5:])
                key = "client:" + args["idempotency_key"]

                async def guarded() -> Any:
                    return await self._breaker(name).call(
                        lambda: call_with_retry(lambda: self._raw_call(info, args), policy=self.retry,
                                                source=f"tool:{name}", attempts_log=attempts))
                data, from_cache = await run_idempotent(self.idem_store, key, guarded)
            else:
                data = await self._breaker(name).call(
                    lambda: call_with_retry(lambda: self._raw_call(info, args), policy=self.retry,
                                            source=f"tool:{name}", attempts_log=attempts))
            result = ToolResult(name, True, data, from_cache=from_cache)
        except Exception as exc:  # noqa: BLE001
            err = classify_exception(exc, source=f"tool:{name}")
            result = ToolResult(name, False, None, err)
        if events is not None:
            events.append({"stage": "tool", "source": f"tool:{name}", "ok": result.ok, "args_from_model": True,
                           "retries": len(attempts), "from_cache": from_cache,
                           "error": result.error.to_dict() if result.error else None,
                           "ms": (time.perf_counter() - t0) * 1000})
        return result
