#!/usr/bin/env python3
"""it4：可训练分诊分类器。训练 = dev + held-out A + held-out B + 红旗原型句；测试 = 独立集 C（由不了解规则的子代理盲写）。
产物：medagent/router/artifacts/triage_clf_tfidf.pkl（运行时用，无需加载向量模型）。"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medagent.safety import assess, current_turn  # noqa: E402
from medagent.triage import PROTOTYPES, LearnedTriage  # noqa: E402


def load(n: str) -> list[dict]:
    return [json.loads(l) for l in open(ROOT / "evaluation" / "datasets" / n, encoding="utf-8")]


def main() -> None:
    train = [(current_turn(r["query"]), int(r["emergency"]), int(r["high_risk"])) for r in load("agent_eval_dev.jsonl")]
    for n in ("safety_heldout_a.jsonl", "safety_heldout_b.jsonl"):
        train += [(r["query"], int(r["emergency"] or r["crisis"]), int(r["high_risk"])) for r in load(n)]
    train += [(t, 1, 0) for v in PROTOTYPES.values() for t in v]
    texts = [t for t, _, _ in train]
    labels = {"emergency": [a for _, a, _ in train], "high_risk": [b for _, _, b in train]}
    encoders = {"tfidf_only": None}
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        encoders["tfidf+bge"] = lambda xs: m.encode(list(xs), normalize_embeddings=True, show_progress_bar=False)
    except ImportError:
        pass
    C = load("safety_test_c.jsonl")
    res: dict = {"_protocol": "C 不参与训练/调参；阈值由训练集 5 折 CV 定在召回 0.9。"}
    for name, enc in encoders.items():
        clf = LearnedTriage(encoder=enc)
        out = {"train_n": len(train), "cv": clf.fit(texts, labels)}
        for task, key in (("emergency", lambda r: r["emergency"] or r["crisis"]), ("high_risk", lambda r: r["high_risk"])):
            y = [bool(key(r)) for r in C]
            p_clf = [clf.predict(r["query"])[task] >= clf.thresholds[task] for r in C]
            a = [assess(r["query"]) for r in C]
            p_rule = [(x.level == "emergency" or x.crisis) if task == "emergency" else x.high_risk for x in a]
            for lab, pred in (("rules_only", p_rule), ("clf_only", p_clf), ("rules_or_clf", [u or v for u, v in zip(p_clf, p_rule)])):
                tp = sum(yy and pp for yy, pp in zip(y, pred))
                fp = sum((not yy) and pp for yy, pp in zip(y, pred))
                out[f"{task}_{lab}"] = {"recall": round(tp / sum(y), 4), "fpr": round(fp / (len(y) - sum(y)), 4),
                                        "misses": [r["query"] for r, yy, pp in zip(C, y, pred) if yy and not pp]}
        res[name] = out
        if name == "tfidf_only":
            (ROOT / "medagent" / "router" / "artifacts").mkdir(exist_ok=True)
            pickle.dump(clf, open(ROOT / "medagent" / "router" / "artifacts" / "triage_clf_tfidf.pkl", "wb"))
        print(name, {k: (v["recall"], v["fpr"]) for k, v in out.items() if isinstance(v, dict) and "recall" in v})
    (ROOT / "evaluation" / "results" / "safety_eval_it4_learned_testC.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
