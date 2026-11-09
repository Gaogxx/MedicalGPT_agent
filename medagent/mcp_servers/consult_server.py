"""“复杂问诊”MCP Server：把 Python 编排器（Router–Worker + 逐步检查点）包成一个工具，让 DSH 模式也用得上。

为什么单独一个 Server（而不是加进 medical_server）：
  - 编排器内部要调用 medical_server 的工具，放在同一个 Server 里会出现“工具调用自己”的递归；
  - 一次复杂问诊要跑多个 worker、多轮模型调用，耗时几十秒，DSH 里可以给它单独设更长的超时，
    不影响普通查药工具 15 秒的超时。

谁会调用它（在 DSH 模式下）：
  1. 路由网关：本轮第一次回复、问题是多意图（“先…再…”“药和饮食”）且 DSH 提供了这个工具时，网关不调模型，
     直接返回一个 complex_consult 工具调用——确定性触发，不依赖模型认不认识这个新工具；
  2. 模型：工具对 local（微调 MedicalGPT）和 strong（百川 / DeepSeek）两档都可见，规则没覆盖到的复杂问题，
     模型也可以自己选择调用（小模型档不带工具，不会调用）。

工具内部：
  - run_id 默认 = hash(问题 + 历史)。同一个调用重放（DSH 超时重试、MCP Server 重启后网关重发）会命中同一个检查点：
    已完成 → 直接返回存好的答案；中断 / 暂停 → 从上次的节点继续，已完成的 worker 不重跑；
  - worker 只拿只读工具（查症状 / 查药 / 相互作用 / 指南 / 检索），复杂问诊过程中不会产生预约之类的副作用；
  - 检查点落在 $MEDAGENT_STATE_DIR/consult_runs.sqlite；`medagent pause <run_id> --consult` 暂停正在跑的问诊，
    暂停后的问诊不会被重试自动续跑，要传 resume=true；进程崩溃 / 超时中断的问诊，重试即续跑。

运行：python -m medagent.mcp_servers.consult_server            # stdio（DSH cordis.yml 里配一个 mcp-medical-consult）
      MEDAGENT_CONSULT_MOCK=1 ...                                # 用 Mock 模型联调，不需要 GPU / API key
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
from pathlib import Path
from typing import Any, Callable

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

READ_ONLY_TOOLS = ["check_symptom", "search_drug", "check_interaction", "search_guideline", "search_knowledge"]
CONSULT = ToolAnnotations(read_only_hint=True, idempotent_hint=True, destructive_hint=False, open_world_hint=False)

INSTRUCTIONS = """复杂问诊编排器。complex_consult 适用于一次包含多个子问题的咨询（例如“先说说怎么用药，再说饮食注意什么”、
“这几种药和饮食分别要注意什么”）：工具内部拆分子问题、每个子问题由独立 worker 查工具作答，返回已校验的完整回答。
单一问题直接用普通工具即可，不要调用本工具。调用失败或超时可以用相同参数重试：会从断点继续，不会重复执行。"""

DESCRIPTION = ("复杂问诊：一次咨询里有多个子问题时调用（如“先…再…”、“药和饮食分别注意什么”）。"
               "内部拆分子问题、并行查询知识工具、校验后返回完整回答（answer）。"
               "question 传用户本轮原话；history 可选，传之前对话的要点（年龄、孕期、过敏、在用药等）。"
               "超时或失败时用相同参数重试会从断点继续。")


def default_run_id(question: str, history: str = "") -> str:
    return "cs-" + hashlib.sha1(f"{question.strip()}\n{history.strip()}".encode("utf-8")).hexdigest()[:12]


def _default_pool():
    from ..cli import _pool
    return _pool(os.environ.get("MEDAGENT_CONSULT_MOCK") == "1")


def _state_path() -> str:
    d = Path(os.environ.get("MEDAGENT_STATE_DIR", ".medagent_state"))
    d.mkdir(parents=True, exist_ok=True)
    return str(d / "consult_runs.sqlite")


def _summary(r: dict[str, Any], state: dict[str, Any], resumed: bool, replayed: bool) -> dict[str, Any]:
    workers = state.get("workers", {})
    return {
        "status": r.get("status", "done"),
        "run_id": state.get("run_id") or r.get("run_id"),
        "answer": r.get("answer", state.get("answer", "")),
        "tier": state.get("tier"),
        "sub_questions": [w.get("question") for wid, w in workers.items() if not wid.startswith(("repair", "w_repair"))],
        "workers": {wid: {"question": w.get("question"), "status": w.get("status"),
                          "tools": sorted({o[0] for o in w.get("observations", [])}),
                          "degraded": bool(w.get("degraded"))} for wid, w in workers.items()},
        "repairs": state.get("repairs", []),
        "findings": [f.get("code") for f in state.get("findings", []) if f.get("severity") == "block"],
        "resumed": resumed,
        "replayed": replayed,
    }


def build_consult_server(pool_factory: Callable[[], Any] | None = None, store: Any = None,
                         medical_server: Any = None) -> MCPServer:
    from ..agent import AgentConfig, MedicalAgent
    from ..cli import _router
    from ..mcp_client import MCPToolbox
    from ..resilience.checkpoint import DONE, PAUSE_REQUESTED, PAUSED, CheckpointStore
    from .medical_server import build_server

    server = MCPServer("medical_consult", instructions=INSTRUCTIONS, version="0.1.0")
    holder: dict[str, Any] = {}
    active: dict[str, asyncio.Task] = {}   # 同一进程内同一 run 只跑一份，重复调用等同一个任务

    def _deps() -> tuple[Any, Any, Any]:
        if "pool" not in holder:
            holder["pool"] = (pool_factory or _default_pool)()
            holder["store"] = store or CheckpointStore(_state_path())
            holder["medical"] = medical_server or build_server()
        return holder["pool"], holder["store"], holder["medical"]

    async def _execute(rid: str, question: str, history: str, resume: bool) -> dict[str, Any]:
        pool, st, med = _deps()
        try:
            state, _, status = st.load(rid)
        except KeyError:
            state, status = None, None
        if status == DONE:
            return _summary({"status": "done", "answer": state.get("answer", "")}, state, resumed=False, replayed=True)
        async with MCPToolbox({"medical": med}) as tb:
            agent = MedicalAgent(pool, tb, _router(pool), st,
                                 cfg=AgentConfig(tool_subset=READ_ONLY_TOOLS, split_with_history=True))
            if state is None:
                hist = [{"role": "user", "content": history}] if history.strip() else []
                r, resumed = await agent.run(question, run_id=rid, history=hist), False
            elif status in (PAUSE_REQUESTED, PAUSED) and not resume:
                # 人为暂停的问诊不自动续跑（尊重暂停）；崩溃 / 超时中断的（status 仍是 running）直接续跑
                return {**_summary({"status": "paused", "answer": ""}, state, False, False),
                        "hint": "该问诊已暂停，传 resume=true 继续"}
            else:
                r, resumed = await agent.resume(rid), True
        final, _, _ = st.load(rid)
        return _summary(r, final, resumed=resumed, replayed=False)

    @server.tool(annotations=CONSULT, description=DESCRIPTION)
    async def complex_consult(question: str, history: str = "", run_id: str = "",
                              resume: bool = False) -> dict[str, Any]:
        q = (question or "").strip()
        if not q:
            raise ToolError("PARAM: question 不能为空")
        if len(q) > 2000 or len(history) > 6000:
            raise ToolError("PARAM: question ≤ 2000 字，history ≤ 6000 字")
        rid = run_id.strip() or default_run_id(q, history)
        task = active.get(rid)
        if task is None or task.done():
            task = asyncio.ensure_future(_execute(rid, q, history, resume))
            active[rid] = task
        try:
            return await asyncio.shield(task)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001 —— 归因前缀让客户端 / 网关知道可以用同样参数重试续跑
            raise ToolError(f"EXTERNAL: 复杂问诊中断（{type(exc).__name__}: {exc}），用相同参数重试会从断点继续；"
                            f"run_id={rid}") from exc
        finally:
            if task.done():
                active.pop(rid, None)

    @server.resource("consult://run/{run_id}", description="一次复杂问诊的检查点：当前节点、子问题、各 worker 状态、修复记录",
                     mime_type="application/json")
    def consult_run(run_id: str) -> dict[str, Any]:
        _, st, _ = _deps()
        try:
            state, version, status = st.load(run_id)
        except KeyError:
            from mcp.server.mcpserver.exceptions import ResourceNotFoundError
            raise ResourceNotFoundError(f"consult://run/{run_id} not found") from None
        return {"run_id": run_id, "status": status, "version": version, "node": state.get("node"),
                **{k: v for k, v in _summary({"status": status}, state, False, False).items()
                   if k in ("sub_questions", "workers", "repairs", "findings", "tier")}}

    server._medagent_holder = holder   # 仅供测试
    return server


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    server = build_consult_server()
    if args.http:
        server.run("streamable-http", host=args.host, port=args.port)
    else:
        server.run("stdio")


if __name__ == "__main__":
    main()
