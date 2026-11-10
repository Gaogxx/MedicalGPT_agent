#!/usr/bin/env python3
"""用独立的路由数据集训练 MF 路由器：train 训练、val 早停并定阈值、test 报告；dev 80（人工标注）全程不参与，作外部测试。

  python evaluation/router/train_router.py --labels evaluation/results/router_labels_qwen3.5-9b.jsonl --budget 0.4
产出：medagent/router/artifacts/mf_router.npz + mf_router.json（运行时优先加载它），
      evaluation/results/router_train_<tag>.json（各集合上的 AUROC、给定强模型调用比例下的失败召回）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

MIN_POSITIVES = 30


def featurize(queries: list[str]) -> np.ndarray:
    from medagent.router.features import QueryFeaturizer, split_history
    qs, hs = zip(*[split_history(q) for q in queries])
    return QueryFeaturizer().transform(list(qs), list(hs))


def forced_of(queries: list[str]) -> np.ndarray:
    from medagent.safety import assess
    return np.array([(lambda a: a.level == "emergency" or a.crisis or a.high_risk)(assess(q)) for q in queries])


def report(scores: np.ndarray, y: np.ndarray, thr: float, budget: float, forced: np.ndarray | None = None) -> dict:
    """at_threshold：线上实际行为（安全规则强制的 + MF 分数 ≥ 阈值的都去强模型）。"""
    from run_router_eval import at_budget, auroc
    forced = np.zeros(len(y), bool) if forced is None else forced
    routed = forced | (scores >= thr)
    return {"n": int(len(y)), "fail_rate": round(float(y.mean()), 4), "auroc": round(auroc(scores, y), 4),
            "safety_forced_rate": round(float(forced.mean()), 4),
            "at_threshold": {"strong_rate": round(float(routed.mean()), 4),
                             "failure_recall": round(float((routed & (y == 1)).sum() / max((y == 1).sum(), 1)), 4)},
            f"at_budget_{budget}": at_budget(scores, y, budget, forced)}


def train(labels: list[dict], budget: float = 0.4, dim: int = 32, out_dir: Path | None = None,
          meta_extra: dict | None = None) -> dict:
    from medagent.router.mf_router import MFRouter, calibrate_with_forced, save_meta
    by = {s: [r for r in labels if r["split"] == s] for s in ("train", "val", "test")}
    y = {s: np.array([r["label"] for r in rs], float) for s, rs in by.items()}
    if y["train"].sum() < MIN_POSITIVES or y["val"].sum() < 5:
        raise SystemExit(f"答砸样本太少（train {int(y['train'].sum())} / val {int(y['val'].sum())}），"
                         f"至少需要 train {MIN_POSITIVES} 条才训练，否则路由器学不到东西")
    X = {s: featurize([r["query"] for r in rs]) for s, rs in by.items()}
    m = MFRouter(dim=dim)
    fit = m.fit(X["train"], y["train"], X["val"], y["val"])
    s = {k: m.predict_strong_win(X[k]) for k in X}
    f = {k: forced_of([r["query"] for r in rs]) for k, rs in by.items()}
    thr = calibrate_with_forced(s["val"], f["val"], budget)        # 阈值只在 val 上定，按总体预算（含安全规则强制部分）
    res = {"budget": budget, "threshold": thr, "fit": fit,
           **{k: report(s[k], y[k], thr, budget, f[k]) for k in ("train", "val", "test")}}
    dev = [json.loads(line) for line in open(ROOT / "evaluation" / "datasets" / "agent_eval_dev.jsonl", encoding="utf-8")]
    yd = np.array([0 if r["label"] == "ok" else 1 for r in dev], float)
    dq = [r["query"] for r in dev]
    res["dev80_heldout"] = report(m.predict_strong_win(featurize(dq)), yd, thr, budget, forced_of(dq))
    res["dev80_heldout"]["note"] = "dev 80 是 2026-09 旧模型的人工标注；新模型重训后它的标签只能作参考"
    out_dir = out_dir or ROOT / "medagent" / "router" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    m.save(out_dir / "mf_router.npz")
    save_meta(out_dir / "mf_router.json", {"features": "mf_hash", "threshold": thr, "budget": budget,
                                           "trained_on": {k: len(v) for k, v in by.items()},
                                           "label": "本地模型在该题上答砸（修复前被校验器拦下，或裁判判为 unsafe/poor）",
                                           "test_auroc": res["test"]["auroc"], **(meta_extra or {})})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--budget", type=float, default=0.4, help="希望交给强模型的比例（在 val 上校准阈值）")
    ap.add_argument("--tag", default="local")
    args = ap.parse_args()
    labels = [json.loads(line) for line in open(args.labels, encoding="utf-8")]
    judge = sorted({r["judge"] for r in labels})
    res = train(labels, args.budget, meta_extra={"labels_file": args.labels, "judge": judge})
    out = ROOT / "evaluation" / "results" / f"router_train_{args.tag}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k in ("val", "test", "dev80_heldout"):
        print(k, "AUROC", res[k]["auroc"], "| 阈值下强模型比例", res[k]["at_threshold"]["strong_rate"],
              "失败召回", res[k]["at_threshold"]["failure_recall"])


if __name__ == "__main__":
    main()
