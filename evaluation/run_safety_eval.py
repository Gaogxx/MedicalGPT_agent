#!/usr/bin/env python3
"""安全预检（紧急 / 危机 / 高风险意图识别）在 dev 与两个 held-out 集上的召回 / 误报。

协议：dev（80 条原用例）+ held-out A 可用于调规则；held-out B 只在最终版本上报告一次，不参与任何调参。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medagent.safety import assess  # noqa: E402


def load(name: str) -> list[dict]:
    return [json.loads(l) for l in open(ROOT / "evaluation" / "datasets" / name, encoding="utf-8")]


def evaluate(rows: list[dict]) -> dict:
    out: dict = {"n": len(rows)}
    for key, pred_fn in (("emergency", lambda a: a.level == "emergency" or a.crisis),
                         ("high_risk", lambda a: a.high_risk), ("crisis", lambda a: a.crisis)):
        gold_pos = [r for r in rows if r.get(key) or (key == "emergency" and r.get("crisis"))]
        if not gold_pos:
            continue
        preds = {r["id"]: pred_fn(assess(r["query"])) for r in rows}
        tp = sum(1 for r in gold_pos if preds[r["id"]])
        neg = [r for r in rows if r not in gold_pos]
        fp = sum(1 for r in neg if preds[r["id"]])
        out[key] = {"recall": round(tp / len(gold_pos), 4), "false_positive_rate": round(fp / max(len(neg), 1), 4),
                    "misses": [r["query"] for r in gold_pos if not preds[r["id"]]],
                    "false_positives": [r["query"] for r in neg if preds[r["id"]]]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--report-b", action="store_true", help="只在最终版本打开：报告 held-out B")
    args = ap.parse_args()
    dev = [{"id": r["id"], "query": r["query"], "emergency": r["emergency"], "high_risk": r["high_risk"], "crisis": False}
           for r in load("agent_eval_dev.jsonl")]
    res = {"dev": evaluate(dev), "heldout_a": evaluate(load("safety_heldout_a.jsonl"))}
    if args.report_b:
        res["heldout_b"] = evaluate(load("safety_heldout_b.jsonl"))
    (ROOT / "evaluation" / "results" / f"safety_eval_{args.tag}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for split, r in res.items():
        print(split, {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ("recall", "false_positive_rate")})
                      for k, v in r.items()})
        for k in ("emergency", "high_risk"):
            if k in r:
                print("   ", k, "misses:", r[k]["misses"][:12], "| FP:", r[k]["false_positives"][:6])


if __name__ == "__main__":
    main()
