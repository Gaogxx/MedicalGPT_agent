#!/usr/bin/env python3
"""开源判别模型（微调后的 Laya）的阈值校准与评测，部署到 laya-serve 之后跑。

  1. 定阈值（判别器验证集：真实回答、规则没拦下的那部分——线上判别模型看到的就是这些）：
       Laya 微调时已经在留出数据上做了温度校准，这里只定两个阈值：
       block_at：P(答砸) 的最小阈值，使验证集上精确率 ≥ 0.8（误升级会多花强模型调用）；
       review_at：召回率 ≥ 0.8 的阈值（不超过 block_at），中间带进人工复核。
     写入 medagent/verify/artifacts/discriminator.json，运行时自动读取。
  2. 外部测试：dev 80 的真实线上回答 + 人工标注（20 条失败）。对比
       规则（it1：F1 0.69）、零样本选型（evaluation/results/judge_bakeoff.json：零样本 Laya 0.48、Skywork 奖励模型 0.85）、
       微调后的 Laya、规则 → Laya 级联；另报告失败类别（failure_type）在 dev 80 失败样本上的准确率。
     dev 80 不参与定阈值，只报告。

  python evaluation/run_discriminator_eval.py --url http://127.0.0.1:8002
  python evaluation/run_discriminator_eval.py --mock        # 用假 laya-serve 冒烟（不代表判别模型的能力）
"""
from __future__ import annotations

import argparse
import json
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
from medagent.verify.discriminator import CALIBRATION, DiscriminatorClient, fake_laya_app  # noqa: E402
from medagent.verify.typed import answer_state  # noqa: E402

GUARD_FIXABLE = {"emergency_missing", "prescription_in_high_risk"}


def pick_thresholds(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    grid = np.round(np.arange(0.30, 0.99, 0.01), 2)
    block = next((float(t) for t in grid if (p >= t).sum() and y[p >= t].mean() >= 0.8), 0.95)
    review = max((float(t) for t in grid if t <= block and y.sum() and (y[p >= t].sum() / y.sum()) >= 0.8), default=0.5)
    return block, min(review, block)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8002")
    ap.add_argument("--model", default="medical-verifier")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--val", default=str(ROOT / "training" / "data" / "verifier" / "laya" / "val.jsonl"))
    ap.add_argument("--no-write", action="store_true", help="不写阈值文件（只看效果）")
    args = ap.parse_args()
    if args.mock:
        from starlette.testclient import TestClient
        c = DiscriminatorClient(base_url="http://testserver", model=args.model, client=TestClient(fake_laya_app()))
    else:
        c = DiscriminatorClient(base_url=args.url, model=args.model, timeout=30)
    res: dict = {"model": "mock" if args.mock else f"{args.model} @ {args.url}"}

    def p_fail(state: dict) -> float:
        return float(c.answers(state)["silent_failure"]["noul"])

    # ---------------------------------------------------------------- 1. 判别器验证集上定阈值
    block_at, review_at = 0.8, 0.5
    val_path = Path(args.val)
    if val_path.exists():
        rows = [json.loads(line) for line in open(val_path, encoding="utf-8")]
        rows = [r for r in rows if not r["meta"].get("rule_block") and not r["meta"].get("synthetic")]
        t0 = time.perf_counter()
        pv = np.array([p_fail(r["meta"]["raw_state"]) for r in rows])
        ms = (time.perf_counter() - t0) * 1000 / max(len(rows), 1)
        yv = np.array([r["expected"]["silent_failure"] for r in rows], int)
        block_at, review_at = pick_thresholds(pv, yv)
        qual = [c.answers(r["meta"]["raw_state"])["answer_quality"]["choice"] == r["expected"]["answer_quality"] for r in rows]
        res["val"] = {"n": len(rows), "fail_rate": round(float(yv.mean()), 4), "auroc": round(auroc(pv, yv), 4),
                      "quality_acc": round(float(np.mean(qual)), 4), "block_at": block_at, "review_at": review_at,
                      "at_block": prf(list(yv), list((pv >= block_at).astype(int))), "ms_per_call": round(ms, 1)}
    else:
        res["val"] = f"{val_path} 不存在：不定阈值，用默认 block_at=0.8、review_at=0.5"

    # ---------------------------------------------------------------- 2. dev 80（人工标注）外部测试
    dev = [json.loads(line) for line in open(ROOT / "evaluation" / "datasets" / "agent_eval_dev.jsonl", encoding="utf-8")]
    y = np.array([0 if r["label"] == "ok" else 1 for r in dev])
    rules_pred, disc_p, types = [], [], []
    for r in dev:
        q, a = r["query"], r["answer"]
        hist = q.split("。 ")[0] if q.startswith("我之前问过") else ""
        asm = assess(q)
        fs = detect(a, query=q, assessment=asm, history=hist)
        rules_pred.append(int(any(f.severity == "block" for f in fs)))
        state = answer_state(q, a, None, [f for f in fs if f.code not in GUARD_FIXABLE], asm)
        disc_p.append(p_fail(state))
        types.append(c.answers(state)["failure_type"]["choice"])
    rules_pred, disc_p = np.array(rules_pred), np.array(disc_p)
    cascade = ((rules_pred == 1) | (disc_p >= block_at)).astype(int)
    score = np.where(rules_pred == 1, 1.0, disc_p)
    fails = [i for i in range(len(dev)) if y[i]]
    res["dev80"] = {
        "n": len(y), "failures": int(y.sum()), "rules_pass": int((rules_pred == 0).sum()),
        "rules": prf(list(y), list(rules_pred)),
        "laya_only@block_at": {**prf(list(y), list((disc_p >= block_at).astype(int))), "auroc": round(auroc(disc_p, y), 4)},
        "laya_auroc_rules_pass": round(auroc(disc_p[rules_pred == 0], y[rules_pred == 0]), 4),
        "cascade(rules→laya)": {**prf(list(y), list(cascade)), "auroc": round(auroc(score, y), 4)},
        "failure_type_acc_on_failures": round(float(np.mean([types[i] == dev[i]["label"] for i in fails])), 4),
        "failure_type_confusion": [f"{dev[i]['id']}:{dev[i]['label']}→{types[i]}" for i in fails if types[i] != dev[i]["label"]],
        "review_band": int(((rules_pred == 0) & (disc_p >= review_at) & (disc_p < block_at)).sum()),
        "missed": [dev[i]["id"] + ":" + dev[i]["label"] for i in fails if not cascade[i]],
        "false_alarms": [dev[i]["id"] for i in range(len(y)) if not y[i] and cascade[i]],
        "baselines": {"rules_f1_it1": 0.69, "zero_shot_auroc": {"laya_multilingual": 0.48, "skywork_reward_v2_0.6b": 0.85}},
        "note": "dev 80 的回答来自 2026-09 旧模型；不参与定阈值，只做外部测试",
    }
    out = ROOT / "evaluation" / "results" / ("discriminator_eval_mock.json" if args.mock else "discriminator_eval.json")
    if not args.mock:
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    if not args.mock and not args.no_write and isinstance(res["val"], dict):
        CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
        CALIBRATION.write_text(json.dumps({"block_at": block_at, "review_at": review_at, "model": args.model,
                                           "val": res["val"], "calibrated": time.strftime("%Y-%m-%d")},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
