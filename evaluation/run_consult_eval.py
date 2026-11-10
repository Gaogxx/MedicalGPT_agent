#!/usr/bin/env python3
"""复杂问诊（网关进程内运行的编排器）的机制评测（Mock 模型，验证编排行为，不代表真实模型的回答质量）。

  1. 触发：80 条 dev 问题里，网关的确定性规则会把哪些交给复杂问诊；
  2. 完整性：同一个 Mock 模型，单 agent（不拆子问题）vs 复杂问诊（拆子问题 + 隔离 worker），多意图题的子问题覆盖率；
  3. 中断续跑：在每个节点注入一次崩溃（模拟网关进程退出），重启后同一请求再来，看能否续跑完成、多花了几次模型调用；
  4. 重放：已完成的问诊再调用一次的模型调用数。

输出：evaluation/results/consult_eval.json
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from metrics import must_cover_ok  # noqa: E402

from medagent.agent import AgentConfig, MedicalAgent  # noqa: E402
from medagent.llm.pool import ModelPool  # noqa: E402
from medagent.llm.providers import MockProvider  # noqa: E402
from medagent.mcp_client import MCPToolbox  # noqa: E402
from medagent.consult import ConsultError, ConsultRunner  # noqa: E402
from medagent.mcp_servers.medical_server import build_server  # noqa: E402
from medagent.resilience.checkpoint import CheckpointStore  # noqa: E402
from medagent.resilience.retry import RetryPolicy  # noqa: E402
from medagent.router.policy import SafetyFirstRouter  # noqa: E402
from medagent.verify.detectors import multi_intent  # noqa: E402



def runner(pl, store=None, attempts=1):
    """attempts=1：不在进程内自动重试，模拟进程崩溃后由新进程续跑。"""
    return ConsultRunner(pl, SafetyFirstRouter(), store or CheckpointStore(), max_attempts=attempts)


async def try_run(r, q):
    try:
        return await r.run(q)
    except ConsultError:
        return None


def pool():
    p = [MockProvider("local"), MockProvider("strong")]
    return ModelPool({"local": [p[0]], "strong": [p[1]]}, retry=RetryPolicy(base_delay=0)), p


def calls(ps) -> int:
    return sum(x.calls for x in ps)


async def completeness(cases: list[dict]) -> dict:
    rows = []
    for c in cases:
        pl, _ = pool()
        async with MCPToolbox({"medical": build_server()}) as tb:
            single = await MedicalAgent(pl, tb, SafetyFirstRouter(), CheckpointStore(),
                                        cfg=AgentConfig(max_workers=1)).run(c["query"])
        pl2, _ = pool()
        r = await try_run(runner(pl2, attempts=2), c["query"])
        rows.append({"query": c["query"], "single_ok": must_cover_ok(single["answer"], c["must_cover"]),
                     "consult_ok": bool(r) and must_cover_ok(r["answer"], c["must_cover"]),
                     "sub_questions": r["sub_questions"] if r else None})
    n = len(rows)
    return {"n": n, "single_agent": sum(r["single_ok"] for r in rows) / n,
            "complex_consult": sum(r["consult_ok"] for r in rows) / n, "rows": rows}


async def crash_resume(query: str) -> list[dict]:
    pl, ps = pool()
    await runner(pl).run(query)
    clean = calls(ps)
    out = []
    for node in ("route", "plan", "work", "aggregate", "verify", "respond"):
        pl, ps = pool()
        orig = getattr(MedicalAgent, f"_node_{node}")
        state = {"n": 0}

        async def crash(self, s, orig=orig, state=state):
            if not state["n"]:
                state["n"] += 1
                raise RuntimeError(f"injected crash before {orig.__name__}")
            return await orig(self, s)

        setattr(MedicalAgent, f"_node_{node}", crash)
        try:
            store = CheckpointStore()
            first = await try_run(runner(pl, store), query)            # 崩溃
            second = await try_run(runner(pl, store), query)           # 新进程，同一请求 → 续跑
            replay_before = calls(ps)
            third = await try_run(runner(pl, store), query)            # 已完成 → 直接返回检查点
        finally:
            setattr(MedicalAgent, f"_node_{node}", orig)
        out.append({"crash_at": node, "first_failed": first is None,
                    "resumed_done": bool(second) and second["status"] == "done",
                    "model_calls": replay_before, "clean_model_calls": clean,
                    "replay_model_calls": calls(ps) - replay_before, "replayed": bool(third) and third["replayed"]})
    # worker 级：w0 已完成落盘，w1 做完模型调用但在落盘前崩溃 → 续跑只重做 w1
    pl, ps = pool()
    orig_w = MedicalAgent._run_worker
    state = {"n": 0}

    async def crash_w1(self, s, wid, question, tier=None):
        res = await orig_w(self, s, wid, question, tier)
        if wid == "w1" and not state["n"]:
            state["n"] += 1
            raise RuntimeError("injected crash inside worker w1")
        return res

    MedicalAgent._run_worker = crash_w1
    try:
        store = CheckpointStore()
        first = await try_run(runner(pl, store), query)
        second = await try_run(runner(pl, store), query)
    finally:
        MedicalAgent._run_worker = orig_w
    out.append({"crash_at": "work:w1（w0 已完成）", "first_failed": first is None,
                "resumed_done": bool(second) and second["status"] == "done", "model_calls": calls(ps),
                "clean_model_calls": clean, "replay_model_calls": None, "replayed": None})
    return out


async def main() -> None:
    cases = [json.loads(line) for line in open(ROOT / "evaluation" / "datasets" / "agent_eval_dev.jsonl", encoding="utf-8")]
    trig = [c for c in cases if multi_intent(c["query"])]
    mc = [c for c in cases if c.get("must_cover")]
    multi_mc = [c for c in mc if len(c["must_cover"]) >= 2]
    res = {
        "note": "Mock 模型：验证编排机制（拆分 / 隔离 / 检查点 / 续跑），不代表真实模型回答质量",
        "trigger": {"dev_total": len(cases), "triggered": len(trig), "triggered_queries": [c["query"] for c in trig],
                    "multi_part_must_cover": len(multi_mc),
                    "multi_part_triggered": sum(1 for c in multi_mc if multi_intent(c["query"]))},
        "completeness": await completeness(multi_mc),
        "crash_resume": await crash_resume("先说说布洛芬的副作用，再说说高血压饮食注意什么"),
    }
    (ROOT / "evaluation" / "results" / "consult_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                                                     encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "completeness"}, ensure_ascii=False, indent=1))
    print("completeness", res["completeness"]["n"], res["completeness"]["single_agent"], res["completeness"]["complex_consult"])


if __name__ == "__main__":
    asyncio.run(main())
