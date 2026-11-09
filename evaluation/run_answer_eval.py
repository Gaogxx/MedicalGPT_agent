#!/usr/bin/env python3
"""回答层评测：在 80 条真实线上输出上逐版本回放运行时的“后处理”环节（不重新调用模型）。

  v0  基线：DSH + OPD 模型原始输出（2026-09）
  v1  + 安全输出关（急症横幅 / 处方式表述改写 / 免责声明）
  v2  + 校验器：工具调用泄漏→解析出调用→经 MCP 真实执行→模板化回答；乱码与内部工具名清理；
      其余静默失败（编造细节 / 多意图漏答 / 编造历史 …）标记为“需升级强模型重生成”
说明：v2 中被标记升级的样本，离线无法得到强模型的新回答，因此其回答文本保持不变（指标不虚增），
      另行报告“静默失败拦截率”——人工标注的失败里，被修复或被拦截升级的比例。
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from metrics import aggregate, score_case  # noqa: E402

from medagent.mcp_client import MCPToolbox  # noqa: E402
from medagent.mcp_servers.medical_server import build_server  # noqa: E402
from medagent.safety import assess, guard_output  # noqa: E402
from medagent.templates import render_fallback  # noqa: E402
from medagent.verify.detectors import detect, extract_leaked_tool_call  # noqa: E402

INTERNAL = re.compile(r"(通过)?调用?`?(check_symptom|search_drug|check_interaction|search_guideline|search_knowledge)`?(函数|工具)?")


def load_dev() -> list[dict]:
    return [json.loads(l) for l in open(ROOT / "evaluation" / "datasets" / "agent_eval_dev.jsonl", encoding="utf-8")]


async def v2_answer(case: dict, tb: MCPToolbox) -> tuple[str, dict]:
    a = assess(case["query"])
    ans = case["answer"]
    hist = case["query"].split("。 ")[0] if case["query"].startswith("我之前问过") else ""
    findings = detect(ans, query=case["query"], assessment=a, history=hist)
    codes = {f.code for f in findings if f.severity == "block"}
    info = {"findings": sorted({f.code for f in findings}), "action": "pass"}
    if "tool_leak" in codes:
        call = extract_leaked_tool_call(ans)
        if call and call.name in tb.tools:
            res = await tb.call(call.name, call.arguments, scope=f"replay:{case['id']}")
            if res.ok:
                ans = render_fallback([(call.name, res.data)])
                info["action"] = f"recovered_leaked_call:{call.name}"
    ans = ans.replace("�", "")
    ans = INTERNAL.sub("查询", ans)
    escalate = codes - {"tool_leak", "emergency_missing", "prescription_in_high_risk"}
    if escalate and info["action"] == "pass":
        info["action"] = "escalate_strong:" + ",".join(sorted(escalate))
    return guard_output(ans, a), info


async def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="迭代编号，如 it2；结果写入 results/answer_eval_<tag>.json")
    args = ap.parse_args()
    cases = load_dev()
    out_dir = ROOT / "evaluation" / "results"
    results = {}
    # v0
    s0 = [score_case(c, c["answer"]) for c in cases]
    results["v0_baseline"] = aggregate(cases, s0, with_human=True)
    # v1
    s1 = [score_case(c, guard_output(c["answer"], assess(c["query"]))) for c in cases]
    results["v1_safety_guard"] = aggregate(cases, s1, with_human=False)
    # v2
    async with MCPToolbox({"medical": build_server()}) as tb:
        pairs = [await v2_answer(c, tb) for c in cases]
    s2 = [score_case(c, a) for c, (a, _) in zip(cases, pairs)]
    agg2 = aggregate(cases, s2, with_human=False)
    failures = [c for c in cases if c["label"] != "ok"]
    handled = {c["id"]: info for c, (_, info) in zip(cases, pairs) if info["action"] != "pass"}
    intercepted = [c["id"] for c in failures if c["id"] in handled or s2[cases.index(c)]["auto_pass"]]
    false_alarms = [cid for cid, info in handled.items() if next(c for c in cases if c["id"] == cid)["label"] == "ok"]
    agg2["silent_failure_interception"] = round(len(intercepted) / len(failures), 4)
    agg2["false_escalation_rate"] = round(len(false_alarms) / (len(cases) - len(failures)), 4)
    agg2["missed_failures"] = sorted({c["id"]: c["label"] for c in failures if c["id"] not in intercepted}.items())
    agg2["actions"] = {cid: info["action"] for cid, info in handled.items()}
    results["v2_verifier_repair"] = agg2
    per_case = [{"id": c["id"], "label": c["label"], "v0": a, "v1": b, "v2": d, "v2_action": p[1]}
                for c, a, b, d, p in zip(cases, s0, s1, s2, pairs)]
    (out_dir / f"answer_eval_{args.tag}.json").write_text(json.dumps({"summary": results, "per_case": per_case}, ensure_ascii=False,
                                                         indent=1), encoding="utf-8")
    for k, v in results.items():
        print(k, json.dumps({kk: vv for kk, vv in v.items() if kk not in ("actions",)}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
