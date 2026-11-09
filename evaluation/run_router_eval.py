#!/usr/bin/env python3
"""路由评测：在 80 条 dev 上，以“本地 MedicalGPT 的真实回答是否达标（人工标注）”为标签，
比较不同路由器在相同强模型调用预算下，能把多少“本地会答砸的问题”提前送去强模型。

  failure_recall@budget   本地失败样本中被路由到强模型的比例（越高越好）
  auroc                   打分对“本地失败”的区分度（5 折交叉验证的折外分数）
  pass_upper_bound        假设强模型在被路由的样本上全部达标时的整体通过率（上界，需强模型实测验证）
小样本（80 条，20 个失败）结论只作方向参考；正式版需用两档模型各跑一遍 + 判别标注扩到上千条（需 API 权限）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medagent.router.features import QueryFeaturizer, split_history  # noqa: E402
from medagent.router.mf_router import MFRouter, calibrate_threshold, save_meta  # noqa: E402
from medagent.safety import assess  # noqa: E402


def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    pos, neg = scores[y == 1], scores[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def stratified_folds(y: np.ndarray, k: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            folds[i % k].append(j)
    return [np.array(f) for f in folds]


def oof_scores(X: np.ndarray, y: np.ndarray, k: int = 5, **kw) -> np.ndarray:
    s = np.zeros(len(y))
    folds = stratified_folds(y, k)
    for f in folds:
        tr = np.setdiff1d(np.arange(len(y)), f)
        inner = stratified_folds(y[tr], 4, seed=1)[0]
        fit_idx = np.setdiff1d(np.arange(len(tr)), inner)
        m = MFRouter(**kw)
        m.fit(X[tr][fit_idx], y[tr][fit_idx].astype(float), X[tr][inner], y[tr][inner].astype(float))
        s[f] = m.predict_strong_win(X[f])
    return s


def at_budget(scores: np.ndarray, y: np.ndarray, budget: float, forced: np.ndarray | None = None) -> dict:
    n = len(y)
    routed = np.zeros(n, bool) if forced is None else forced.copy()
    remaining = int(round(budget * n)) - routed.sum()
    if remaining > 0:
        order = [i for i in np.argsort(-scores) if not routed[i]]
        routed[order[:remaining]] = True
    rec = (routed & (y == 1)).sum() / max((y == 1).sum(), 1)
    return {"strong_rate": round(routed.mean(), 4), "failure_recall": round(float(rec), 4),
            "pass_upper_bound": round(1 - ((~routed) & (y == 1)).sum() / n, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bge", action="store_true", help="用 bge-small-zh 向量作为 e_q（否则用哈希 n-gram）")
    ap.add_argument("--tag", default="it1")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(ROOT / "evaluation" / "datasets" / "agent_eval_dev.jsonl", encoding="utf-8")]
    y = np.array([0 if r["label"] == "ok" else 1 for r in rows])
    qs, hs = zip(*[split_history(r["query"]) for r in rows])
    feats = {"hash": QueryFeaturizer()}
    if args.bge:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        feats["bge"] = QueryFeaturizer(encoder=lambda xs: m.encode(list(xs), normalize_embeddings=True, show_progress_bar=False))
    forced = np.array([(lambda a: a.level == "emergency" or a.crisis or a.high_risk)(assess(r["query"])) for r in rows])
    res: dict = {"n": len(y), "failures": int(y.sum()), "rules_forced_rate": round(float(forced.mean()), 4)}
    rng = np.random.default_rng(0)
    budgets = (0.2, 0.3, 0.4, 0.5)
    res["random"] = {str(b): {"failure_recall": b} for b in budgets}
    res["rules_only"] = at_budget(np.zeros(len(y)), y, forced.mean(), forced)
    hand_only = np.stack([feats["hash"].hand(q, h) for q, h in zip(qs, hs)])
    for name, X in [("mf_hand_features", hand_only)] + [(f"mf_{k}", f.transform(list(qs), list(hs))) for k, f in feats.items()]:
        s = oof_scores(X, y, dim=16 if X.shape[1] < 32 else 32)
        res[name] = {"auroc_oof": round(auroc(s, y), 4),
                     **{str(b): at_budget(s, y, b) for b in budgets},
                     **{f"safety_first+{b}": at_budget(s, y, max(b, forced.mean()), forced) for b in budgets}}
    out = ROOT / "evaluation" / "results" / f"router_eval_{args.tag}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))

    # 用全部 dev 训练最终路由器并按 40% 预算校准阈值（RouteLLM calibrate_threshold 做法），供运行时加载
    best = "mf_hash"   # 交叉验证里哈希特征（AUROC 0.78）优于 bge（0.71），运行时用哈希版，零模型依赖
    X = feats["hash"].transform(list(qs), list(hs))
    final = MFRouter(dim=32)
    final.fit(X, y.astype(float))
    scores = final.predict_strong_win(X)
    thr = calibrate_threshold(scores, 0.4)
    model_dir = ROOT / "medagent" / "router" / "artifacts"
    model_dir.mkdir(exist_ok=True)
    final.save(model_dir / f"{best}_dev80.npz")
    save_meta(model_dir / f"{best}_dev80.json", {"features": best, "threshold@40%": thr, "trained_on": "agent_eval_dev (80)",
                                                  "label": "local MedicalGPT answer fails (human label)",
                                                  "warning": "80 条样本训练，仅作冷启动；上线前需用两档模型实测标签重训"})


if __name__ == "__main__":
    main()
