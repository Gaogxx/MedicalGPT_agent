#!/usr/bin/env python3
"""药品库补全试点的留出评测集（2026-11-19）。

Huatuo 百科问答的 validation 拆分里能对上试点药品的说明书题只有个位数，所以题目分两部分：
  · validation 拆分里的“用法用量 / 不良反应 / 禁忌”题（药品库只用 train 编译）
  · train 拆分里同一个药、但出自另一份说明书的问答：参考答案必须和药品库实际采用的那段文字不同（逐字相同或互相包含的不收）
两组评测（补全前 / 后）都能用本地全文索引检索到这些参考文本（索引收了 train 和 test），条件对等；
测的是“有了全文检索之后，结构化药品库还能多带来多少”。test 拆分不用。每个药最多 --per-drug 题。

  python evaluation/build_drug_label_eval.py --huatuo-dir ../huatuo --drug-kb medagent/knowledge/drug_kb.json
输出：evaluation/datasets/drug_label_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medagent.knowledge.drug_label_kb import _Q, _QFIELD, clean, form_of, strip_name  # noqa: E402
from medagent.knowledge.ext import NOT_A_DRUG  # noqa: E402

CORE = ("dosage", "side_effects", "contraindications")


def field_of(key: str) -> str | None:
    return _QFIELD.get(key) or ("contraindications" if key.startswith("禁忌症") else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--huatuo-dir", required=True)
    ap.add_argument("--drug-kb", default=str(ROOT / "medagent" / "knowledge" / "drug_kb.json"))
    ap.add_argument("--per-drug", type=int, default=2)
    ap.add_argument("--max", type=int, help="总题数上限（按字段分层抽样）")
    ap.add_argument("--out", default=str(ROOT / "evaluation" / "datasets" / "drug_label_eval.jsonl"))
    args = ap.parse_args()
    wd = json.loads((ROOT / "medagent" / "knowledge" / "wikidata_kb.json").read_text(encoding="utf-8"))["drugs"]
    kb = json.loads(Path(args.drug_kb).read_text(encoding="utf-8"))["drugs"]
    names = {}
    for q, e in wd.items():
        for n in [e["zh"], *e["aliases"]]:
            if len(n) >= 2 and n not in NOT_A_DRUG:
                names.setdefault(n, q)
    for q, e in kb.items():
        for n in e.get("new_aliases", {}):
            names.setdefault(n, q)
    train = set()
    for line in open(Path(args.huatuo_dir) / "train.jsonl", encoding="utf-8"):
        train.add(clean(json.loads(line)["answers"][0]))
    existing = {json.loads(line)["query"] for line in open(ROOT / "evaluation" / "datasets" / "agent_eval_dev_ext.jsonl", encoding="utf-8")}
    rows, per = [], Counter()
    import random
    random.seed(20261119)
    for split in ("validation", "train"):
        lines = open(Path(args.huatuo_dir) / f"{split}.jsonl", encoding="utf-8").readlines()
        if split == "train":
            random.shuffle(lines)
        for line in lines:
            o = json.loads(line)
            q = o["questions"][0][0].strip()
            m = _Q.match(q)
            if not m or field_of(m.group(2)) not in CORE or q in existing:
                continue
            qid = next((names[c] for c in strip_name(m.group(1)) if c in names), None)
            if not qid or qid not in kb or per[qid] >= args.per_drug:
                continue
            ref = clean(o["answers"][0])
            f = field_of(m.group(2))
            if len(ref) < 10:
                continue
            if split == "train":       # 只收药品库没用到的那份说明书（主条目和各剂型的文字都要不同）
                used = [re.sub(r"^（[^）]*）", "", kb[qid].get(f, "")).rstrip("…")] + \
                    [r.get(f, "").rstrip("…") for r in kb[qid].get("by_form", {}).values()]
                used = [u for u in used if u]
                if not used or kb[qid].get("kb_tier") == "curated" or any(u[:60] in ref or ref[:60] in u for u in used):
                    continue
            per[qid] += 1
            rows.append({"id": f"d{len(rows) + 1:03d}", "query": q, "reference": ref, "field": f, "qid": qid,
                         "drug": kb[qid]["name"], "product": m.group(1), "split": split, "exact_in_train": split == "validation" and ref in train,
                         "kb_has_field": f in kb[qid].get("source", {}), "asked_form": form_of(m.group(1)),
                         "kb_forms": sorted(kb[qid].get("by_form", {})), "category": "2-药品-说明书",
                         "subset": "druglabel"})
    if args.max and len(rows) > args.max:       # 全量库题太多：按字段分层抽样
        random.seed(7)
        keep = []
        for f in CORE:
            sub = [r for r in rows if r["field"] == f]
            keep += random.sample(sub, min(len(sub), round(args.max * len(sub) / len(rows))))
        rows = sorted(keep, key=lambda r: r["id"])
    with open(args.out, "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "题", Counter(r["field"] for r in rows), Counter(r["split"] for r in rows), "参考答案与 train 逐字相同", sum(r["exact_in_train"] for r in rows),
          "药品库有该字段", sum(r["kb_has_field"] for r in rows), "涉及药品", len(per), args.out)


if __name__ == "__main__":
    main()
