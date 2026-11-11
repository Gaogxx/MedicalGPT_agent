#!/usr/bin/env python3
"""真实模型端到端评测：dev 问题走完整编排（MCP 工具 + 校验 + 输出安全关），全部固定在 local 档，
测的是“本地模型 + 运行时”的整体表现。

题目：--set dev80（2026-09 的 80 条，可与当时 MedicalGPT-7B 的回答同指标对比）| ext（扩充的 216 条）| all（默认，296 条）
指标：metrics.py 的自动检查（急症建议、高风险合规、格式、多意图覆盖、药品事实、查不到如实说明、不编造历史……）；
      --judge strong 时再请百川对照参考答案打质量等级（约 296 次调用），--judge verifier 用本地判别模型（不花钱，看不到参考答案）。

  python evaluation/run_live_eval.py --base-url http://127.0.0.1:8000/v1 --model medical-agent \
      --out evaluation/results/live_qwen3.5-9b.json
  python evaluation/run_live_eval.py --mock --set dev80     # 不需要模型，冒烟测试脚本本身
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from metrics import aggregate, score_case  # noqa: E402

from medagent.agent import MedicalAgent  # noqa: E402
from medagent.llm.pool import ModelPool  # noqa: E402
from medagent.llm.providers import MockProvider, OpenAICompatProvider  # noqa: E402
from medagent.mcp_client import MCPToolbox  # noqa: E402
from medagent.mcp_servers.medical_server import build_server  # noqa: E402
from medagent.resilience.checkpoint import CheckpointStore  # noqa: E402
from medagent.router.policy import RouteDecision, SafetyFirstRouter  # noqa: E402


async def judge_answers(args, cases: list[dict], rows: list[dict]) -> None:
    """可选：裁判对照参考答案（没有参考答案时按医学常识）给每个回答打 unsafe / poor / acceptable / good。"""
    sys.path.insert(0, str(ROOT / "evaluation" / "router"))
    from label_router_set import judge_strong, judge_verifier

    pool = ModelPool.from_yaml(str(ROOT / "configs" / "models.yaml")) if args.judge == "strong" else None
    sem = asyncio.Semaphore(8)

    async def one(c: dict, r: dict) -> None:
        async with sem:
            try:
                if args.judge == "strong":
                    r["judge"] = await judge_strong(pool, c["query"], c.get("reference") or "（无参考答案，按医学常识判断）", r["answer"])
                else:
                    r["judge"] = await asyncio.to_thread(judge_verifier, c["query"], r["answer"], [])
            except Exception as exc:  # noqa: BLE001
                r["judge"] = ("judge_error", str(exc)[:200])

    await asyncio.gather(*(one(c, r) for c, r in zip(cases, rows)))


def judge_summary(rows: list[dict]) -> dict:
    from collections import Counter
    levels = Counter(r["judge"][0] for r in rows if r.get("judge"))
    n = sum(v for k, v in levels.items() if k in ("unsafe", "poor", "acceptable", "good"))
    return {"levels": dict(levels), "pass_rate": round((levels["acceptable"] + levels["good"]) / n, 4) if n else None,
            "unsafe_rate": round(levels["unsafe"] / n, 4) if n else None}


class LocalOnly(SafetyFirstRouter):
    """所有问题都交给 local 档（保留急症快路径标记），评测的是本地模型本身。"""

    def route(self, query, history="", assessment=None, session_cost=0.0):
        d = super().route(query, history, assessment, session_cost)
        return RouteDecision("local", "eval:local_only", d.score, d.threshold, d.fastpath)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="medical-agent")
    ap.add_argument("--out", default=str(ROOT / "evaluation" / "results" / "live_eval.json"))
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--set", choices=("dev80", "ext", "all"), default="all")
    ap.add_argument("--judge", choices=("none", "strong", "verifier"), default="none")
    args = ap.parse_args()
    provider = MockProvider("mock") if args.mock else OpenAICompatProvider("local-eval", args.base_url, args.model,
                                                                          api_key_env="VLLM_API_KEY", timeout=120)
    pool = ModelPool({"local": [provider]})
    ds = ROOT / "evaluation" / "datasets"
    cases = []
    if args.set in ("dev80", "all"):
        cases += [{**json.loads(line), "subset": "dev80"} for line in open(ds / "agent_eval_dev.jsonl", encoding="utf-8")]
    if args.set in ("ext", "all") and (ds / "agent_eval_dev_ext.jsonl").exists():
        cases += [{**json.loads(line), "subset": "ext"} for line in open(ds / "agent_eval_dev_ext.jsonl", encoding="utf-8")]
    cases = cases[: args.limit] if args.limit else cases
    rows = []
    async with MCPToolbox({"medical": build_server()}) as tb:
        agent = MedicalAgent(pool, tb, LocalOnly(), CheckpointStore())
        for c in cases:
            t0 = time.perf_counter()
            r = await agent.run(c["query"])
            llm = [e for e in r["events"] if e.get("stage") == "llm"]
            tools = [e["source"].split(":", 1)[1] for e in r["events"] if e.get("stage") == "tool" and e.get("ok")]
            rows.append({"id": c["id"], "category": c["category"], "query": c["query"], "answer": r["answer"],
                         "repairs": r["repairs"], "tools": tools, "llm_calls": len(llm),
                         "seconds": round(time.perf_counter() - t0, 2)})
            print(c["id"], f"{rows[-1]['seconds']}s", r["repairs"], r["answer"][:50].replace("\n", " "), flush=True)
    if args.judge != "none":
        await judge_answers(args, cases, rows)
    new = [score_case(c, r["answer"]) for c, r in zip(cases, rows)]
    old = [(c, score_case(c, c["answer"])) for c in cases if c.get("answer")]     # 只有 dev 80 有 2026-09 的旧回答
    by = {}
    for name in ("dev80", "ext"):
        idx = [i for i, c in enumerate(cases) if c["subset"] == name]
        if idx:
            by[name] = aggregate([cases[i] for i in idx], [new[i] for i in idx], with_human=False)
    res = {"model": "mock" if args.mock else f"{args.model} @ {args.base_url}", "n": len(cases), "set": args.set,
           "this_model": aggregate(cases, new, with_human=False), "by_subset": by,
           "by_category": {cat: aggregate([c for c in cases if c["category"] == cat],
                                          [s for c, s in zip(cases, new) if c["category"] == cat], with_human=False)
                           for cat in sorted({c["category"] for c in cases})},
           "judge": judge_summary(rows) if args.judge != "none" else None,
           "medicalgpt_7b_2026_09": aggregate([c for c, _ in old], [s for _, s in old], with_human=False) if old else None,
           "leak_recovered": sum(any(x.startswith("recover_leaked_call") for x in r["repairs"]) for r in rows),
           "escalated_or_fallback": sum(bool(r["repairs"]) for r in rows),
           "answers": rows}
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    base = res["medicalgpt_7b_2026_09"] or {}
    for k in ("auto_pass_rate", "emergency_advice_rate", "high_risk_compliance", "format_clean_rate",
              "drug_fact_recall", "multi_intent_completion", "not_found_honesty", "history_faithfulness"):
        print(f"{k:28s} {res['this_model'][k]}  | dev80: {by.get('dev80', {}).get(k)}  ext: {by.get('ext', {}).get(k)}"
              f"  (2026-09 MedicalGPT-7B, dev80: {base.get(k)})")
    if res["judge"]:
        print("裁判", json.dumps(res["judge"], ensure_ascii=False))
    print("工具调用泄漏被救回的次数", res["leak_recovered"], "| 触发修复 / 兜底的题数", res["escalated_or_fallback"])


if __name__ == "__main__":
    asyncio.run(main())
