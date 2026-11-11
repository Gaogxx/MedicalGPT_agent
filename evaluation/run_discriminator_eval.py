#!/usr/bin/env python3
"""开源判别模型的校准与评测（部署到 vLLM 之后跑）。

  1. 校准（判别器验证集，真实回答、规则没拦下的那部分——线上判别模型看到的就是这些）：
       温度 T：在 0.3~5 上网格搜索，使四个等级的负对数似然最小；
       block_at：P(静默失败) 的最小阈值，使验证集上精确率 ≥ 0.8（误升级会多花强模型调用）；
       review_at：召回率 ≥ 0.8 的阈值（不超过 block_at），中间带进人工复核。
     写入 medagent/verify/artifacts/discriminator.json，运行时自动读取。
  2. 外部测试：dev 80 的真实线上回答 + 人工标注（20 条失败）。对比
       规则（it1：F1 0.69）、合成故障训练的 TF-IDF 分类器（it1：AUROC 0.56，级联 F1 0.41）、判别模型、规则 → 判别模型级联。
     dev 80 不参与校准，只报告。

  python evaluation/run_discriminator_eval.py --url http://127.0.0.1:8002/v1
  python evaluation/run_discriminator_eval.py --mock        # 用假 vLLM 冒烟（不代表判别模型的能力）
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from run_verifier_eval import auroc, prf  # noqa: E402

from medagent.safety import assess  # noqa: E402
from medagent.verify.detectors import detect  # noqa: E402
from medagent.verify.discriminator import CALIBRATION, DiscriminatorClient, distribution, fake_vllm_app, letter_logprobs  # noqa: E402
from medagent.verify.typed import answer_state  # noqa: E402

GUARD_FIXABLE = {"emergency_missing", "prescription_in_high_risk"}
LEVELS = ("good", "acceptable", "poor", "unsafe")


def raw_logprobs(c: DiscriminatorClient, state: dict) -> dict[str, float]:
    r = c.http.post(f"{c.base_url}/chat/completions", json=c.request_body(state),
                    headers={"Authorization": f"Bearer {c.api_key}"} if c.api_key else {})
    r.raise_for_status()
    content = r.json()["choices"][0]["logprobs"]["content"]
    return letter_logprobs(content[0]["top_logprobs"] if content else [])


def p_fail(lps: dict[str, float], t: float) -> float:
    d = distribution(lps, t)
    return d["poor"] + d["unsafe"]


def fit_temperature(lps: list[dict], y4: list[str]) -> float:
    best, best_t = math.inf, 1.0
    for t in np.round(np.arange(0.3, 5.01, 0.05), 2):
        nll = -sum(math.log(max(distribution(l, float(t))[lab], 1e-9)) for l, lab in zip(lps, y4))
        if nll < best:
            best, best_t = nll, float(t)
    return best_t


def pick_thresholds(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    grid = np.round(np.arange(0.30, 0.99, 0.01), 2)
    block = next((float(t) for t in grid if (p >= t).sum() and y[p >= t].mean() >= 0.8), 0.95)
    review = max((float(t) for t in grid if t <= block and y.sum() and (y[p >= t].sum() / y.sum()) >= 0.8), default=0.5)
    return block, min(review, block)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8002/v1")
    ap.add_argument("--model", default="medical-verifier")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--val", default=str(ROOT / "training" / "data" / "verifier" / "openai" / "val.jsonl"))
    ap.add_argument("--no-write", action="store_true", help="不写校准文件（只看效果）")
    args = ap.parse_args()
    if args.mock:
        from starlette.testclient import TestClient
        c = DiscriminatorClient(base_url="http://testserver/v1", model=args.model, client=TestClient(fake_vllm_app()))
    else:
        c = DiscriminatorClient(base_url=args.url, model=args.model, timeout=30)
    res: dict = {"model": "mock" if args.mock else f"{args.model} @ {args.url}"}

    # ---------------------------------------------------------------- 1. 在判别器验证集上校准
    t, block_at, review_at = 1.0, 0.8, 0.5
    val_path = Path(args.val)
    if val_path.exists():
        rows = [json.loads(line) for line in open(val_path, encoding="utf-8")]
        rows = [r for r in rows if not r["meta"].get("rule_block") and not r["meta"].get("synthetic")]
        t0 = time.perf_counter()
        lps = [raw_logprobs(c, r["meta"]["state"]) for r in rows]
        y4 = [r["meta"]["level"] for r in rows]
        yv = np.array([lab in ("poor", "unsafe") for lab in y4], int)
        p1 = np.array([p_fail(l, 1.0) for l in lps])
        t = fit_temperature(lps, y4)
        pv = np.array([p_fail(l, t) for l in lps])
        block_at, review_at = pick_thresholds(pv, yv)
        acc = np.mean([max(distribution(l, t), key=distribution(l, t).get) == lab for l, lab in zip(lps, y4)])
        res["val"] = {"n": len(rows), "fail_rate": round(float(yv.mean()), 4), "auroc": round(auroc(pv, yv), 4),
                      "auroc_T1": round(auroc(p1, yv), 4), "four_class_acc": round(float(acc), 4),
                      "temperature": t, "block_at": block_at, "review_at": review_at,
                      "at_block": prf(list(yv), list((pv >= block_at).astype(int))),
                      "ms_per_call": round((time.perf_counter() - t0) * 1000 / max(len(rows), 1), 1)}
    else:
        res["val"] = f"{val_path} 不存在：跳过校准，用默认 T=1、block_at=0.8、review_at=0.5"

    # ---------------------------------------------------------------- 2. dev 80（人工标注）外部测试
    dev = [json.loads(line) for line in open(ROOT / "evaluation" / "datasets" / "agent_eval_dev.jsonl", encoding="utf-8")]
    y = np.array([0 if r["label"] == "ok" else 1 for r in dev])
    rules_pred, disc_p = [], []
    for r in dev:
        q, a = r["query"], r["answer"]
        hist = q.split("。 ")[0] if q.startswith("我之前问过") else ""
        asm = assess(q)
        fs = detect(a, query=q, assessment=asm, history=hist)
        rules_pred.append(int(any(f.severity == "block" for f in fs)))
        state = answer_state(q, a, None, [f for f in fs if f.code not in GUARD_FIXABLE], asm)
        disc_p.append(p_fail(raw_logprobs(c, state), t))
    rules_pred, disc_p = np.array(rules_pred), np.array(disc_p)
    cascade = ((rules_pred == 1) | (disc_p >= block_at)).astype(int)
    score = np.where(rules_pred == 1, 1.0, disc_p)
    res["dev80"] = {
        "n": len(y), "failures": int(y.sum()),
        "rules": prf(list(y), list(rules_pred)),
        "discriminator_only@block_at": {**prf(list(y), list((disc_p >= block_at).astype(int))), "auroc": round(auroc(disc_p, y), 4)},
        "cascade(rules→discriminator)": {**prf(list(y), list(cascade)), "auroc": round(auroc(score, y), 4)},
        "review_band": int(((rules_pred == 0) & (disc_p >= review_at) & (disc_p < block_at)).sum()),
        "missed": [dev[i]["id"] + ":" + dev[i]["label"] for i in range(len(y)) if y[i] and not cascade[i]],
        "false_alarms": [dev[i]["id"] for i in range(len(y)) if not y[i] and cascade[i]],
        "baseline_it1": {"rules_f1": 0.69, "tfidf_classifier_auroc": 0.56, "tfidf_cascade_f1": 0.41},
        "note": "dev 80 的回答来自 2026-09 旧模型；不参与校准，只做外部测试",
    }
    out = ROOT / "evaluation" / "results" / ("discriminator_eval_mock.json" if args.mock else "discriminator_eval.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    if not args.mock and not args.no_write and isinstance(res["val"], dict):
        CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
        CALIBRATION.write_text(json.dumps({"temperature": t, "block_at": block_at, "review_at": review_at,
                                           "model": args.model, "val": res["val"], "calibrated": time.strftime("%Y-%m-%d")},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
