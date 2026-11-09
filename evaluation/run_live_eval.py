#!/usr/bin/env python3
"""真实模型端到端评测：80 条 dev 问题走完整编排（MCP 工具 + 校验 + 输出安全关），全部固定在 local 档，
测的是“本地模型 + 运行时”的整体表现；与 2026-09 线上 MedicalGPT-7B 的回答用同一套指标对比。

  python evaluation/run_live_eval.py --base-url http://127.0.0.1:8000/v1 --model medical-agent \
      --out evaluation/results/live_qwen3.5-9b.json
  python evaluation/run_live_eval.py --mock     # 不需要模型，冒烟测试脚本本身
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
    args = ap.parse_args()
    provider = MockProvider("mock") if args.mock else OpenAICompatProvider("local-eval", args.base_url, args.model,
                                                                          api_key_env="VLLM_API_KEY", timeout=120)
    pool = ModelPool({"local": [provider]})
    cases = [json.loads(line) for line in open(ROOT / "evaluation" / "datasets" / "agent_eval_dev.jsonl", encoding="utf-8")]
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
    new = [score_case(c, r["answer"]) for c, r in zip(cases, rows)]
    base = [score_case(c, c["answer"]) for c in cases]
    res = {"model": "mock" if args.mock else f"{args.model} @ {args.base_url}", "n": len(cases),
           "this_model": aggregate(cases, new, with_human=False),
           "medicalgpt_7b_2026_09": aggregate(cases, base, with_human=False),
           "leak_recovered": sum(any(x.startswith("recover_leaked_call") for x in r["repairs"]) for r in rows),
           "escalated_or_fallback": sum(bool(r["repairs"]) for r in rows),
           "answers": rows}
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k in ("auto_pass_rate", "emergency_advice_rate", "high_risk_compliance", "format_clean_rate",
              "drug_fact_recall", "multi_intent_completion"):
        print(f"{k:28s} {res['this_model'][k]}  (2026-09 MedicalGPT-7B: {res['medicalgpt_7b_2026_09'][k]})")
    print("工具调用泄漏被救回的次数", res["leak_recovered"], "| 触发修复 / 兜底的题数", res["escalated_or_fallback"])


if __name__ == "__main__":
    asyncio.run(main())
