#!/usr/bin/env python3
"""不需要标签的阈值校准：在路由数据集 val 部分（真实问诊提问）上，按“总体交给强模型的比例”重新定 MF 阈值。

原来冷启动 MF 的阈值是在 dev 80（它的训练集）上按 40% 定的；放到真实问诊上，MF 打分整体偏高，
加上安全规则强制交给强模型的部分，总体比例会远超预算。阈值只依赖分数分布，所以可以先在无标签的真实提问上校准，
等有了新模型的标签再用 train_router.py 重训。test 部分只用来报告校准后的实际比例。

  python evaluation/router/calibrate_router.py --budget 0.4
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from medagent.cli import _router
    from medagent.llm.pool import ModelPool
    from medagent.llm.providers import MockProvider
    from medagent.router.mf_router import calibrate_with_forced
    from medagent.safety import assess
    from medagent.triage import load_learned_triage

    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=0.4)
    args = ap.parse_args()
    load_learned_triage()
    pool = ModelPool({t: [MockProvider(t)] for t in ("small", "local", "strong")})
    router = _router(pool)
    if router.mf is None:
        raise SystemExit("没有可校准的 MF 路由器")
    rows = [json.loads(line) for line in open(ROOT / "evaluation" / "datasets" / "router_set.jsonl", encoding="utf-8")]

    def forced_and_scores(rs):
        forced = np.array([(lambda a: a.level == "emergency" or a.crisis or a.high_risk)(assess(r["query"])) for r in rs])
        return forced, np.array([router.score(r["query"]) for r in rs])

    val = [r for r in rows if r["split"] == "val"]
    test = [r for r in rows if r["split"] == "test"]
    fv, sv = forced_and_scores(val)
    thr = calibrate_with_forced(sv, fv, args.budget)
    art = ROOT / "medagent" / "router" / "artifacts"
    name = "mf_router.json" if router.source.startswith("mf_router") else "mf_hash_dev80.json"
    meta = json.loads((art / name).read_text(encoding="utf-8"))
    old = meta.get("threshold_router_val", meta.get("threshold", meta.get("threshold@40%")))
    meta.update(threshold_router_val=thr, calibration={"data": "router_set val（无标签真实提问）", "n": len(val),
                                                       "budget": args.budget, "safety_forced_rate": round(float(fv.mean()), 4)})
    if name == "mf_router.json":
        meta["threshold"] = thr
    (art / name).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    router.threshold = thr
    res = {"artifact": name, "threshold_before": old, "threshold_after": thr}
    for label, rs in (("val", val), ("test", test)):
        c = Counter(router.route(r["query"]).tier for r in rs)
        res[label] = {k: round(v / len(rs), 4) for k, v in sorted(c.items())}
    ft, _ = forced_and_scores(test)
    res["test_safety_forced_rate"] = round(float(ft.mean()), 4)
    out = ROOT / "evaluation" / "results" / "router_calibration.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
