#!/usr/bin/env python3
"""下一轮微调（第 5 步：按工具结果作答，2026-11-20）的训练问题。和三套评测（药品说明书 400 / 某病用药与相互作用 300 / 原 296）
完全不重叠：评测里出现过的药（QID）、病、药对都不用，296 题的原问题不用。

  A 说明书题      Huatuo 百科问答 train 的“X 的用法用量 / 副作用 / 禁忌”，换成口语问法，可带剂型
  B 某病用什么药  医学知识图谱病名里 Wiki 有说明书适应症链接的病
  C 能不能一起吃  说明书相互作用段点名 / 点类的药对；外加两份说明书都没提到对方的药对（教“没提到≠没有相互作用”）
  D 查不到        Huatuo 里对不上药品库的商品名（中成药等）：教如实说查不到
  E 保持原有行为  sft_v3 里不带工具的一般医疗问题（答案用模型自己的，自蒸馏）

  python training/qwen35/grounding/g1_queries.py --huatuo .cache/corpus/FreedomIntelligence__huatuo_encyclopedia_qa/train_datasets.jsonl \
      --out training/data/grounding/queries.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from medagent.knowledge.drug_label_kb import FORM_CATS, load_huatuo  # noqa: E402
from medagent.knowledge.ext import load_ext  # noqa: E402
from medagent.knowledge.interactions import _GENERIC, _SENT, _class_hit, class_terms, label_pair  # noqa: E402
from medagent.knowledge.wiki import load_wiki  # noqa: E402

FIELD_T = {"dosage": ("{p}怎么吃？", "{p}一次用多少？", "{p}的用法用量是什么？", "{p}一天用几次？"),
           "side_effects": ("{p}有什么副作用？", "{p}的不良反应有哪些？", "吃{p}会有什么反应吗？"),
           "contraindications": ("{p}哪些人不能用？", "{p}有什么禁忌？", "孕妇能用{p}吗？", "肝不好能吃{p}吗？")}
CHOICE_T = ("{d}吃什么药好？", "{d}一般用哪些药？", "得了{d}可以用什么药？", "{d}有什么药能治？", "医生说我是{d}，常用药有哪些？")
INTER_T = ("{a}和{b}能一起吃吗？", "我在吃{a}，还能吃{b}吗？", "{a}跟{b}同时用有没有问题？", "{a}和{b}有相互作用吗？")
NOT_MED = {"鸦片", "乙醇", "酒精", "阿片", "烟草", "葡萄柚"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--huatuo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", default="A=1300,B=450,C=500,C0=150,D=120,E=800")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    n = {k: int(v) for k, v in (x.split("=") for x in args.n.split(","))}
    rng = random.Random(args.seed)
    ext, wiki = load_ext(), load_wiki()
    labels = ext.labels
    ds = ROOT / "evaluation" / "datasets"
    dl = [json.loads(x) for x in open(ds / "drug_label_eval.jsonl", encoding="utf-8")]
    we = [json.loads(x) for x in open(ds / "wiki_eval.jsonl", encoding="utf-8")]
    old = [json.loads(x) for f in ("agent_eval_dev.jsonl", "agent_eval_dev_ext.jsonl") for x in open(ds / f, encoding="utf-8")]
    eval_q = {c["qid"] for c in dl}
    for c in we:
        for k in ("drug_a", "drug_b"):
            if c.get(k):
                h = ext.find(c[k], "drug")
                if h:
                    eval_q.add(h[0])
    eval_dis = {c["disease"] for c in we if c["kind"] == "drug_choice"}
    eval_queries = {c["query"] for c in old}
    out: list[dict] = []

    # A
    huatuo = load_huatuo(args.huatuo)
    cand = []
    unmatched = []
    for prod, fields in huatuo.items():
        h = ext.find(prod, "drug")
        if not h:
            if not re.search(r"[A-Za-z0-9]", prod) and 3 <= len(prod) <= 10:
                unmatched.append(prod)
            continue
        if h[0] in eval_q or h[0] not in labels or labels[h[0]].get("kb_tier") not in ("label", "curated"):
            continue
        for f in FIELD_T:
            if fields.get(f):
                cand.append((prod, f))
    rng.shuffle(cand)
    for prod, f in cand[: n["A"]]:
        out.append({"kind": "label", "field": f, "query": rng.choice(FIELD_T[f]).format(p=prod)})

    # B
    kg = json.loads((ROOT / "medagent" / "knowledge" / "kg_disease_drugs.json").read_text(encoding="utf-8"))
    dis = [d for d in kg if d not in eval_dis and wiki.find_disease(d) and wiki.by_name.get(d) == wiki.find_disease(d)
           and len(wiki.dis_drugs.get(wiki.dis_of_path[wiki.find_disease(d)], [])) >= 1]
    rng.shuffle(dis)
    for d in dis[: n["B"]]:
        out.append({"kind": "drug_choice", "query": rng.choice(CHOICE_T).format(d=d)})

    # C / C0
    qs = [q for q, e in labels.items() if e.get("kb_tier") in ("label", "curated") and q not in eval_q
          and e["name"] not in NOT_MED and len(e["name"]) >= 2]
    with_text = [q for q in qs if labels[q].get("interactions_text")]
    names = {}
    for q in qs:
        for nm in [labels[q]["name"], *labels[q].get("aliases", [])]:
            if len(nm) >= 2:
                names.setdefault(nm, q)
    ordered = sorted(names, key=len, reverse=True)
    pairs, seen = [], set()
    for qa in with_text:
        got = 0
        for s in _SENT.split(labels[qa]["interactions_text"]):
            if got >= 3 or len(s.strip()) < 8 or _GENERIC.search(s):
                continue
            qb = next((names[x] for x in ordered if x in s and names[x] != qa and s[s.find(x) + len(x):s.find(x) + len(x) + 1] != "类"), None)
            if qb is None:
                members = [q for q in qs if q != qa and any(t in s and _class_hit(s, t, class_terms(labels[q].get("atc", [])))
                                                            for t in class_terms(labels[q].get("atc", [])))]
                qb = rng.choice(members) if members else None
            if qb and frozenset((qa, qb)) not in seen:
                seen.add(frozenset((qa, qb)))
                pairs.append((qa, qb))
                got += 1
    rng.shuffle(pairs)
    for qa, qb in pairs[: n["C"]]:
        a, b = labels[qa]["name"], labels[qb]["name"]
        if rng.random() < 0.5:
            a, b = b, a
        out.append({"kind": "interaction", "query": rng.choice(INTER_T).format(a=a, b=b)})
    k = 0
    while k < n["C0"]:
        qa, qb = rng.sample(with_text, 2)
        lp = label_pair(labels[qa]["name"], labels[qb]["name"])
        if lp and not lp["found"] and lp["labels_checked"]:
            out.append({"kind": "interaction_none", "query": rng.choice(INTER_T).format(a=labels[qa]["name"], b=labels[qb]["name"])})
            k += 1

    # D
    rng.shuffle(unmatched)
    for p in unmatched[: n["D"]]:
        out.append({"kind": "not_found", "query": rng.choice(FIELD_T["dosage"]).format(p=p)})

    # E
    pool = []
    for line in open(ROOT / "training" / "data" / "sft_v3" / "train.jsonl", encoding="utf-8"):
        conv = json.loads(line)["conversations"]
        if len(conv) == 2 and conv[0]["from"] == "human" and 6 <= len(conv[0]["value"]) <= 120 \
                and conv[0]["value"] not in eval_queries:
            pool.append(conv[0]["value"])
    rng.shuffle(pool)
    for q in pool[: n["E"]]:
        out.append({"kind": "retain", "query": q})

    rng.shuffle(out)
    for i, r in enumerate(out):
        r["id"] = f"g{i:05d}"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
    from collections import Counter
    print(len(out), Counter(r["kind"] for r in out))


if __name__ == "__main__":
    main()
