#!/usr/bin/env python3
"""静默失败判别评测：RuleJudge / LocalClassifierJudge / CascadeJudge。

数据：
  real       80 条真实线上回答（20 条人工标注为失败）
  synthetic  由 real 中“ok”回答经故障注入生成（空回复 / 截断 / 工具 JSON 泄漏 / 乱码 / 编造持续时间 / 编造历史 /
             查不到却给剂量 / 重复循环），只用于训练，不进测试
协议：real 做 5 折分层交叉验证；每折用（其余 4 折 real + 全部 synthetic）训练分类器，在留出折上测。
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medagent.safety import assess  # noqa: E402
from medagent.verify.detectors import detect  # noqa: E402
from medagent.verify.judge import CascadeJudge, LocalClassifierJudge, RuleJudge  # noqa: E402


def state(q: str, a: str, obs=None, finish=None, history="") -> dict:
    asm = assess(q)
    fs = detect(a, query=q, assessment=asm, observations=obs, history=history, finish_reason=finish)
    return {"query": q, "answer": a, "observations": obs, "finish_reason": finish, "history": history,
            "assessment": asm, "findings": fs, "finding_codes": [f.code for f in fs]}


def synthesize(ok_rows: list[dict], rng: random.Random) -> list[dict]:
    out = []
    for r in ok_rows:
        q, a = r["query"], r["answer"]
        kind = rng.choice(["empty", "truncated", "leak", "garbled", "fake_duration", "fake_history", "miss_dose",
                           "repeat", "subtle_truncate"])
        obs, finish, a2 = None, None, a
        if kind == "empty":
            a2 = rng.choice(["", "。", "好的"])
        elif kind == "truncated":
            a2, finish = a[: max(5, len(a) // 3)], "length"
        elif kind == "subtle_truncate":
            a2 = a[: max(8, int(len(a) * 0.55))]
        elif kind == "leak":
            a2 = rng.choice(["ptom\n", "olith\n", ""]) + json.dumps({"name": "check_symptom", "arguments": {"symptom": q[:8]}},
                                                                   ensure_ascii=False)
        elif kind == "garbled":
            i = rng.randrange(0, max(1, len(a)))
            a2 = a[:i] + "��" + a[i:]
        elif kind == "fake_duration":
            a2 = f"您的症状已经持续了{rng.choice(['3', '两', '5'])}天，" + a
        elif kind == "fake_history":
            a2 = "根据您之前提到的头痛和用药情况，" + a
        elif kind == "miss_dose":
            obs = [{"found": False, "query": "某药"}]
            a2 = "该药适应症为缓解疼痛，用法用量：成人一次10mg，一日3次。" + a
        elif kind == "repeat":
            seg = a[:12] or "请注意休息。"
            a2 = seg * 5
        out.append({"query": q, "answer": a2, "observations": obs, "finish_reason": finish, "label": kind, "synthetic": True})
    return out


def prf(y: list[int], p: list[int]) -> dict:
    tp = sum(1 for a, b in zip(y, p) if a and b)
    fp = sum(1 for a, b in zip(y, p) if not a and b)
    fn = sum(1 for a, b in zip(y, p) if a and not b)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0}


def auroc(s: np.ndarray, y: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def main() -> None:
    rng = random.Random(7)
    real = [json.loads(l) for l in open(ROOT / "evaluation" / "datasets" / "agent_eval_dev.jsonl", encoding="utf-8")]
    synth_rows = synthesize([r for r in real if r["label"] == "ok"] * 3, rng)
    (ROOT / "evaluation" / "datasets" / "failure_cases_synthetic.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in synth_rows), encoding="utf-8")
    hist = lambda q: q.split("。 ")[0] if q.startswith("我之前问过") else ""  # noqa: E731
    real_states = [{**state(r["query"], r["answer"], history=hist(r["query"])), "label": r["label"]} for r in real]
    synth_states = [{**state(r["query"], r["answer"], r["observations"], r["finish_reason"]), "label": r["label"]}
                    for r in synth_rows]
    y = np.array([0 if s["label"] == "ok" else 1 for s in real_states])
    rules = RuleJudge()
    p_rules = [1 if rules.noul("", s).label == "yes" else 0 for s in real_states]
    folds = [[] for _ in range(5)]
    for cls in (0, 1):
        idx = [i for i in range(len(y)) if y[i] == cls]
        rng.shuffle(idx)
        for k, i in enumerate(idx):
            folds[k % 5].append(i)
    p_clf = np.zeros(len(y))
    p_cas = np.zeros(len(y), dtype=int)
    escalations = 0
    for f in folds:
        tr = [real_states[i] for i in range(len(y)) if i not in f] + synth_states
        clf = LocalClassifierJudge.train(tr)
        cas = CascadeJudge(classifier=clf)
        for i in f:
            p_clf[i] = clf.prob(real_states[i])
            d = cas.is_silent_failure(real_states[i])
            p_cas[i] = 1 if d.label == "yes" else 0
        escalations += cas.stats["escalated"]
    res = {
        "n_real": len(y), "n_real_failures": int(y.sum()), "n_synthetic_train": len(synth_states),
        "rules": prf(list(y), p_rules),
        "local_classifier@0.5": {**prf(list(y), list((p_clf >= 0.5).astype(int))), "auroc": round(auroc(p_clf, y), 4)},
        "cascade(rules→classifier)": prf(list(y), list(p_cas)),
        "missed_by_cascade": [real[i]["id"] + ":" + real[i]["label"] for i in range(len(y)) if y[i] and not p_cas[i]],
        "false_alarms_cascade": [real[i]["id"] for i in range(len(y)) if not y[i] and p_cas[i]],
        "note": "Jev / LLMJudge 升级层未启用（需 API）；cascade 中不确定带样本按分类器判定。",
    }
    (ROOT / "evaluation" / "results" / "verifier_eval_it1.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                                                         encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    # 结论：合成故障训练的分类器迁移不到真实失败（AUROC≈0.57），因此不导出运行时产物，默认只用规则层。


if __name__ == "__main__":
    main()
