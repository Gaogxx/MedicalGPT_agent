#!/usr/bin/env python3
"""配对比较几种校验链路的端到端结果，只让百川给“回答不一样”的题打分（2026-11-16）。

前提：各组都用 run_live_eval.py --temperature 0 跑同一批题——本地模型贪心解码，回答只会在校验链路判法不同（拦下 → 强模型重答）的题上不同，
回答相同的题裁判结论必然相同，不用重复花钱。--full 指定的那一组会全量打分（给出绝对水平），其余组只补打和它不同的回答。

  python evaluation/judge_paired.py --full A=evaluation/results/live_v4_A_rules_laya_reward.json \\
      B=evaluation/results/live_v4_B_rules_reward.json C=evaluation/results/live_v4_C_rules.json
输出：evaluation/results/judge_paired_v4.json（每组的等级分布、与 --full 组的逐题胜负）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "router"))

RANK = {"unsafe": 0, "poor": 1, "acceptable": 2, "good": 3}


async def main() -> None:
    from label_router_set import judge_strong

    from medagent.llm.pool import ModelPool
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=结果文件")
    ap.add_argument("--full", required=True, help="全量打分的那一组 NAME=结果文件")
    ap.add_argument("--out", default=str(ROOT / "evaluation" / "results" / "judge_paired_v4.json"))
    ap.add_argument("--diff-only", action="store_true", help="只给各组之间回答不同的题打分（只比胜负，不报绝对水平）")
    args = ap.parse_args()
    specs = dict(x.split("=", 1) for x in [args.full] + args.runs)
    full = args.full.split("=", 1)[0]
    runs = {k: {r["id"]: r for r in json.load(open(v, encoding="utf-8"))["answers"]} for k, v in specs.items()}
    ds = ROOT / "evaluation" / "datasets"
    ref = {}
    for f in ("agent_eval_dev.jsonl", "agent_eval_dev_ext.jsonl"):
        if (ds / f).exists():
            for line in open(ds / f, encoding="utf-8"):
                c = json.loads(line)
                ref[c["id"]] = c.get("reference") or "（无参考答案，按医学常识判断）"
    # 要打分的 (id, 回答) 去重：--full 组全部 + 其余组里与它不同的
    todo: dict[tuple[str, str], dict] = {}
    differ = {cid for cid in runs[full] if len({rs[cid]["answer"] for rs in runs.values()}) > 1}
    for cid, r in runs[full].items():
        if not args.diff_only or cid in differ:
            todo[(cid, r["answer"])] = r
    for name, rs in runs.items():
        for cid, r in rs.items():
            if not args.diff_only or cid in differ:
                todo.setdefault((cid, r["answer"]), r)
    pool = ModelPool.from_yaml(str(ROOT / "configs" / "models.yaml"))
    sem = asyncio.Semaphore(8)
    verdict: dict[tuple[str, str], str] = {}

    async def one(key, r):
        async with sem:
            try:
                level, _ = await judge_strong(pool, r["query"], ref.get(key[0], ""), key[1])
            except Exception as exc:  # noqa: BLE001
                level = "judge_error"
            verdict[key] = level

    print(f"需要打分的回答 {len(todo)} 条（{full} 全量 {len(runs[full])} + 其余组不同的 {len(todo) - len(runs[full])}）", flush=True)
    await asyncio.gather(*(one(k, r) for k, r in todo.items()))
    res: dict = {"full": full, "judged": len(todo), "runs": {}}
    for name, rs in runs.items():
        levels = Counter(verdict[(cid, r["answer"])] for cid, r in rs.items() if (cid, r["answer"]) in verdict)
        n = sum(levels[k] for k in RANK)
        res["runs"][name] = {"levels": dict(levels), "pass_rate": round((levels["good"] + levels["acceptable"]) / max(n, 1), 4),
                             "unsafe_rate": round(levels["unsafe"] / max(n, 1), 4),
                             "repaired": sum(bool(r["repairs"]) for r in rs.values())}
        if name != full:
            diff = [cid for cid in rs if rs[cid]["answer"] != runs[full][cid]["answer"]]
            better = worse = 0
            detail = []
            for cid in diff:
                a, b = verdict[(cid, runs[full][cid]["answer"])], verdict[(cid, rs[cid]["answer"])]
                if a in RANK and b in RANK:
                    better += RANK[a] > RANK[b]
                    worse += RANK[a] < RANK[b]
                detail.append({"id": cid, full: a, name: b, f"{full}_repairs": runs[full][cid]["repairs"], f"{name}_repairs": rs[cid]["repairs"]})
            res["runs"][name][f"vs_{full}"] = {"answers_differ": len(diff), f"{full}_better": better, f"{name}_better": worse, "detail": detail}
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != f"vs_{full}"} | ({f"vs_{full}": {x: y for x, y in v[f"vs_{full}"].items() if x != "detail"}} if f"vs_{full}" in v else {})
                      for k, v in res["runs"].items()}, ensure_ascii=False, indent=1))
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
