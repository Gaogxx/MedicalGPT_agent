"""复杂问诊：在路由网关进程内运行 Python 编排器（Router–Worker + 逐步检查点），由模型侧按需调用。

两种触发方式（都在网关里，DSH 看不到中间过程，只收到一条最终回答）：
  1. 规则：本轮第一次回复、问题是明显的多意图（“先…再…”、“药和饮食”），网关直接运行编排器；
  2. 模型：网关在发给 local / strong 模型的工具清单里加一个内部工具 complex_consult，模型判断问题复杂时调用它，
     网关拦下这个调用、运行编排器，不会转给 DSH。

编排器内部：拆子问题，每个 worker 只拿子问题 + 患者槽位 + 只读工具（经 MCP 调医疗工具 Server），用与网关相同的模型池，
每走完一步存检查点。run_id = hash(问题 + 历史)：网关进程崩溃后 DSH 重发同一请求，会命中同一个检查点续跑，
已完成的 worker 不重跑；同一进程里出错时自动再续跑一次。
"""
from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from typing import Any

READ_ONLY_TOOLS = ["check_symptom", "search_drug", "check_interaction", "search_guideline", "search_knowledge"]

TOOL_NAME = "complex_consult"
TOOL_SPEC = {"type": "function", "function": {
    "name": TOOL_NAME,
    "description": ("复杂问诊：一次咨询里有多个子问题、需要分别查证时调用（如“先说用药再说饮食”、“这几种药分别要注意什么”）。"
                    "会拆分子问题、分别查询知识工具并校验，返回完整回答。单一问题直接用普通工具，不要调用本工具。"),
    "parameters": {"type": "object", "properties": {
        "question": {"type": "string", "description": "用户本轮原话"},
        "history": {"type": "string", "description": "可选：之前对话里的关键信息（年龄、孕期、过敏、在用药等）"}},
        "required": ["question"]}}}


class ConsultError(Exception):
    pass


def default_run_id(question: str, history: str = "") -> str:
    return "cs-" + hashlib.sha1(f"{question.strip()}\n{history.strip()}".encode("utf-8")).hexdigest()[:12]


def state_path() -> str:
    d = Path(os.environ.get("MEDAGENT_STATE_DIR", ".medagent_state"))
    d.mkdir(parents=True, exist_ok=True)
    return str(d / "consult_runs.sqlite")


def summarize(state: dict[str, Any], status: str, answer: str | None = None) -> dict[str, Any]:
    workers = state.get("workers", {})
    return {
        "status": status,
        "run_id": state.get("run_id"),
        "node": state.get("node"),
        "answer": state.get("answer", "") if answer is None else answer,
        "tier": state.get("tier"),
        "sub_questions": [w.get("question") for wid, w in workers.items() if not wid.startswith(("repair", "w_repair"))],
        "workers": {wid: {"question": w.get("question"), "status": w.get("status"),
                          "tools": sorted({o[0] for o in w.get("observations", [])}),
                          "degraded": bool(w.get("degraded"))} for wid, w in workers.items()},
        "repairs": state.get("repairs", []),
        "findings": [f.get("code") for f in state.get("findings", []) if f.get("severity") == "block"],
    }


class ConsultRunner:
    """medical：医疗工具 MCP Server 的连接目标（进程内 MCPServer 实例 / streamable-http URL），默认进程内建一个。"""

    def __init__(self, pool: Any, router: Any, store: Any = None, medical: Any = None, max_attempts: int = 2) -> None:
        from .resilience.checkpoint import CheckpointStore
        self.pool, self.router = pool, router
        self.store = store or CheckpointStore(state_path())
        self._medical = medical or os.environ.get("MEDAGENT_MCP_URL")
        self.max_attempts = max_attempts
        self._active: dict[str, asyncio.Task] = {}

    @property
    def medical(self) -> Any:
        if self._medical is None:
            from .mcp_servers.medical_server import build_server
            self._medical = build_server()
        return self._medical

    async def _once(self, rid: str, question: str, history: str, resume: bool) -> dict[str, Any]:
        from .agent import AgentConfig, MedicalAgent
        from .mcp_client import MCPToolbox
        from .resilience.checkpoint import DONE, PAUSE_REQUESTED, PAUSED
        try:
            state, _, status = self.store.load(rid)
        except KeyError:
            state, status = None, None
        if status == DONE:
            return {**summarize(state, "done"), "resumed": False, "replayed": True}
        if status in (PAUSE_REQUESTED, PAUSED) and not resume:
            # 人为暂停（medagent pause <run_id> --consult）的问诊不自动续跑
            return {**summarize(state, "paused", ""), "resumed": False, "replayed": False}
        async with MCPToolbox({"medical": self.medical}) as tb:
            agent = MedicalAgent(self.pool, tb, self.router, self.store,
                                 cfg=AgentConfig(tool_subset=READ_ONLY_TOOLS, split_with_history=True))
            if state is None:
                hist = [{"role": "user", "content": history}] if history.strip() else []
                r, resumed = await agent.run(question, run_id=rid, history=hist), False
            else:
                r, resumed = await agent.resume(rid), True
        final, _, st = self.store.load(rid)
        return {**summarize(final, r.get("status", st), r.get("answer")), "resumed": resumed, "replayed": False}

    async def _with_retry(self, rid: str, question: str, history: str, resume: bool) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                out = await self._once(rid, question, history, resume)
                if attempt:
                    out["resumed"] = True
                return out
            except Exception as exc:  # noqa: BLE001 —— 任何中断都从检查点再续跑一次
                last = exc
        raise ConsultError(f"复杂问诊中断（{type(last).__name__}: {last}），run_id={rid}") from last

    async def run(self, question: str, history: str = "", run_id: str | None = None,
                  resume: bool = False) -> dict[str, Any]:
        q = (question or "").strip()
        if not q:
            raise ConsultError("question 不能为空")
        rid = run_id or default_run_id(q, history)
        task = self._active.get(rid)
        if task is None or task.done():
            task = asyncio.ensure_future(self._with_retry(rid, q, history, resume))
            self._active[rid] = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                self._active.pop(rid, None)

    def status(self, run_id: str) -> dict[str, Any] | None:
        try:
            state, version, status = self.store.load(run_id)
        except KeyError:
            return None
        return {**summarize(state, status), "version": version}
