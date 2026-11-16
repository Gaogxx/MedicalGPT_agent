#!/usr/bin/env python3
"""校验器 v4 的奖励模型（Skywork-Reward-V2）阈值校准与评测（2026-11-16）。

  1. 打分：Medical-GPT 在路由数据集 train / val 上规则放行的回答（answers_final9b.jsonl，百川标签）+ dev 80。
     零样本、不用这些数据训练，所以 train 部分可以直接当校准集。分数缓存在 evaluation/results/reward_scores_<模型>.jsonl。
  2. 定阈值（train）：block = 拦截比例 --block-rate（默认 3%）对应的分数，review = --review-rate（默认 10%）。
  3. 报告：train / val 上拦下多少真失败、多少误报；dev 80 上规则 / 规则→Laya / 规则→Laya→奖励模型 三种级联的 P / R / F1
     （Laya 的逐条判定取自 --laya-eval 指向的 discriminator_eval 结果）。
写入 medagent/verify/artifacts/reward_model.json（运行时读取）和 evaluation/results/reward_eval.json。

  python evaluation/run_reward_eval.py --url http://127.0.0.1:8003          # 用部署好的打分服务
  python evaluation/run_reward_eval.py --local                               # 本进程加载模型（GPU / MPS / CPU）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from medagent.safety import assess  # noqa: E402
from medagent.verify.detectors import detect  # noqa: E402
from medagent.verify.reward import CALIBRATION  # noqa: E402


def scorer(args):
    if not args.local:
        from medagent.verify.reward import RewardClient
        c = RewardClient(base_url=args.url, timeout=60)
        return c.score
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    m = AutoModelForSequenceClassification.from_pretrained(args.model, dtype=torch.bfloat16 if dev != "cpu" else torch.float32).to(dev).eval()

    def score(q: str, a: str) -> float:
        text = tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a}], tokenize=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=4096)
        with torch.no_grad():
            return float(m(**{k: v.to(dev) for k, v in ids.items()}).logits[0][0])
    return score


def prf(y: np.ndarray, pred: np.ndarray) -> dict:
    tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / max(p + r, 1e-9), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8003")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--model", default="Skywork/Skywork-Reward-V2-Qwen3-1.7B")
    ap.add_argument("--answers", default=str(ROOT / "training" / "data" / "verifier" / "answers_final9b.jsonl"))
    ap.add_argument("--laya-eval", default=str(ROOT / "evaluation" / "results" / "discriminator_eval.json"))
    ap.add_argument("--block-rate", type=float, default=0.03)
    ap.add_argument("--review-rate", type=float, default=0.10)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    cache_path = ROOT / "evaluation" / "results" / f"reward_scores_{args.model.split('/')[-1]}.jsonl"
    cache = {}
    if cache_path.exists():
        for line in open(cache_path, encoding="utf-8"):
            r = json.loads(line)
            cache[r["key"]] = r["score"]
    score_fn = None

    def score(key: str, q: str, a: str) -> float:
        nonlocal score_fn
        if key not in cache:
            score_fn = score_fn or scorer(args)
            cache[key] = score_fn(q, a)
            with open(cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "score": cache[key]}, ensure_ascii=False) + "\n")
        return cache[key]

    # ---- 1/2. Medical-GPT 的回答（规则放行部分）：train 定阈值、val 报告
    rows = [json.loads(line) for line in open(args.answers, encoding="utf-8")]
    rows = [r for r in rows if r["level"] in ("good", "acceptable", "poor", "unsafe") and not r["rule_block"]]
    s = np.array([score(f"final9b:{r['id']}", r["state"]["query"], r["state"]["answer"]) for r in rows])
    y = np.array([r["level"] in ("poor", "unsafe") for r in rows], int)
    split = np.array([r["split"] for r in rows])
    tr = split == "train"
    block = float(np.quantile(s[tr], args.block_rate))
    review = float(np.quantile(s[tr], args.review_rate))
    from sklearn.metrics import roc_auc_score
    res: dict = {"model": args.model, "block_reward": round(block, 4), "review_reward": round(review, 4),
                 "block_rate": args.block_rate, "review_rate": args.review_rate}
    for name in ("train", "val"):
        m = split == name
        res[name] = {"n": int(m.sum()), "failures": int(y[m].sum()), "auroc": round(float(roc_auc_score(y[m], -s[m])), 4),
                     "at_block": prf(y[m], s[m] <= block), "at_review": prf(y[m], s[m] <= review)}

    # ---- 3. dev 80：规则 / 规则→Laya / 规则→Laya→奖励模型
    dev = [json.loads(line) for line in open(ROOT / "evaluation" / "datasets" / "agent_eval_dev.jsonl", encoding="utf-8")]
    laya = json.load(open(args.laya_eval, encoding="utf-8"))["dev80"] if Path(args.laya_eval).exists() else None
    missed = {x.split(":")[0] for x in laya["missed"]} if laya else set()
    fa = set(laya["false_alarms"]) if laya else set()
    yd, rules, lay, sd = [], [], [], []
    for r in dev:
        q, a = r["query"], r["answer"]
        hist = q.split("。 ")[0] if q.startswith("我之前问过") else ""
        fail = r["label"] != "ok"
        yd.append(fail)
        rules.append(any(f.severity == "block" for f in detect(a, query=q, assessment=assess(q), history=hist)))
        lay.append((fail and r["id"] not in missed) or r["id"] in fa)
        sd.append(score(f"dev80:{r['id']}", q, a))
    yd, rules, lay, sd = (np.array(x) for x in (yd, rules, lay, sd))
    res["dev80"] = {"rules": prf(yd.astype(int), rules), "rules_to_reward": prf(yd.astype(int), rules | (sd <= block))}
    if laya:
        res["dev80"]["rules_to_laya"] = prf(yd.astype(int), lay)
        res["dev80"]["rules_to_laya_to_reward"] = prf(yd.astype(int), lay | (sd <= block))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if not args.no_write:
        (ROOT / "evaluation" / "results" / "reward_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        CALIBRATION.write_text(json.dumps({"model": args.model, "block_reward": res["block_reward"], "review_reward": res["review_reward"],
                                           "block_rate": args.block_rate, "review_rate": args.review_rate,
                                           "val": res["val"], "dev80": res["dev80"], "calibrated": "2026-11-16"},
                                          ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
