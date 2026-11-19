#!/usr/bin/env python3
"""医药 Wiki 用得上用不上：“某病用什么药”和“两个药能不能一起吃”两类题（2026-11-20）。

一、某病用什么药（drug_choice）
  病：医学知识图谱（huatuo_knowledge_graph_qa）里“X 的推荐药”是药品清单、且 Wiki 疾病页有说明书适应症链接的病。
  参考：知识图谱的推荐药清单（药品网站数据，和说明书是两个来源；有噪声，比如“感冒→阿莫西林”，所以只做次要指标）。
  打分时另用两个与问答无关的固定表（建题时从药品库算好写进题里，两种运行用同一份）：
    label_drugs      说明书适应症写明了这个病的药（经模型复核）——回答里点到的药有多少能在说明书里找到依据
    contra_drugs     说明书禁忌里写了这个病的药——回答把它们当成治这个病的药推荐就是安全问题
二、能不能一起吃（interaction）
  从药品库 450 份说明书的「药物相互作用」段落取药对：A 的段落点名提到 B，或提到 B 所属的药物类别（ATC 对得上）。
  开卷题：参考答案就是知识库里的说明书原句，考的是系统能不能查到并用上，不是模型自己知道多少。精选库手写的 10 对排除。

  python evaluation/build_wiki_eval.py --n-choice 150 --n-inter 150
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medagent.knowledge import _INTERACTION_NOTES  # noqa: E402
from medagent.knowledge.ext import load_ext  # noqa: E402
from medagent.knowledge.interactions import _GENERIC, _SENT, class_terms  # noqa: E402
from medagent.knowledge.wiki import load_wiki  # noqa: E402

CHOICE_T = ("{d}一般吃什么药？", "{d}用什么药治疗比较好？", "得了{d}，有哪些药可以用？", "我有{d}，平时可以用哪些药？")
NOT_MED = {"鸦片", "乙醇", "酒精", "阿片", "烟草", "葡萄柚"}
INTER_T = ("我在吃{a}，能同时吃{b}吗？", "{a}和{b}可以一起用吗？", "{a}跟{b}能一起吃吗？会不会有影响？")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-choice", type=int, default=150)
    ap.add_argument("--n-inter", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "evaluation" / "datasets" / "wiki_eval.jsonl"))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    ext, wiki = load_ext(), load_wiki()
    labels = ext.labels
    kg = json.loads((ROOT / "medagent" / "knowledge" / "kg_disease_drugs.json").read_text(encoding="utf-8"))
    cases = []

    # ---------------- 一、某病用什么药
    pool = []
    for d, prods in kg.items():
        p = wiki.find_disease(d)
        if not p or wiki.by_name.get(d) != p:          # 病名要正好是页面名或别名，避免“老年人糖尿病”落到“糖尿病”页
            continue
        rows = wiki.dis_drugs.get(wiki.dis_of_path[p], [])
        if len(rows) < 2:
            continue
        ref_generic = []
        for prod in prods:
            h = ext.find(prod, "drug") if "复方" not in prod else None
            if h and ext.drugs[h[0]]["zh"] not in ref_generic:
                ref_generic.append(ext.drugs[h[0]]["zh"])
        contra = [e["name"] for e in labels.values()
                  if e.get("kb_tier") in ("label", "curated") and re.search(f"(?<![非无]){re.escape(d)}", str(e.get("contraindications", "")))]
        pool.append({"disease": d, "kg_ref": prods, "kg_ref_generic": ref_generic,
                     "label_drugs": [n for _, n, _ in rows], "contra_drugs": contra[:40]})
    rng.shuffle(pool)
    for i, x in enumerate(pool[: args.n_choice]):
        cases.append({"id": f"w{i:03d}", "query": rng.choice(CHOICE_T).format(d=x["disease"]), "kind": "drug_choice",
                      **x, "category": "W-某病用药", "subset": "wiki"})

    # ---------------- 二、能不能一起吃
    names = {}
    for q, e in labels.items():
        if e.get("kb_tier") in ("label", "curated"):
            for n in [e["name"], *e.get("aliases", []), *e.get("new_aliases", {})]:
                if len(n) >= 2:
                    names.setdefault(n, q)
    ordered = sorted(names, key=len, reverse=True)
    by_class: dict[str, list[str]] = {}
    for q, e in labels.items():
        if e.get("kb_tier") in ("label", "curated"):
            for t in class_terms(e.get("atc", [])):
                by_class.setdefault(t, []).append(q)
    curated_pairs = {frozenset(p) for p in _INTERACTION_NOTES}
    named, classed = [], []
    for qa, ea in labels.items():
        text = ea.get("interactions_text") or ""
        if not text or ea.get("kb_tier") not in ("label", "curated"):
            continue
        for s in _SENT.split(text):
            s = s.strip()
            if len(s) < 8 or _GENERIC.search(s):
                continue
            for n in ordered:
                qb = names[n]
                i = s.find(n)
                if i >= 0 and qb != qa and s[i + len(n):i + len(n) + 1] != "类" and labels[qb]["name"] not in NOT_MED:
                    named.append((qa, qb, s, f"点名「{n}」"))
                    break
            for t, qs in by_class.items():
                if t in s:
                    qs2 = [q for q in qs if q != qa and labels[q]["name"] not in NOT_MED]
                    if qs2:
                        classed.append((qa, rng.choice(qs2), s, f"类别「{t}」"))
                    break
    seen, picked = set(), []
    for src in (named, classed):
        rng.shuffle(src)
    # 点名和点类大约 2:1；同一个 A 最多 2 题，同一对只出一次
    per_a: dict[str, int] = {}
    for src, quota in ((named, args.n_inter * 2 // 3), (classed, args.n_inter - args.n_inter * 2 // 3)):
        k = 0
        for qa, qb, s, how in src:
            a, b = labels[qa]["name"], labels[qb]["name"]
            key = frozenset({a, b})
            if key in seen or key in curated_pairs or per_a.get(qa, 0) >= 2 or a == b:
                continue
            seen.add(key)
            per_a[qa] = per_a.get(qa, 0) + 1
            x, y = (a, b) if rng.random() < 0.5 else (b, a)
            picked.append({"query": rng.choice(INTER_T).format(a=x, b=y), "kind": "interaction", "drug_a": a, "drug_b": b,
                           "label_of": a, "reference": s, "matched": how,
                           "source": labels[qa].get("source", {}).get("interactions_text", "")})
            k += 1
            if k >= quota:
                break
    for i, x in enumerate(picked):
        cases.append({"id": f"x{i:03d}", **x, "category": "W-相互作用", "subset": "wiki"})
    Path(args.out).write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
    print(f"某病用药 {sum(c['kind'] == 'drug_choice' for c in cases)}（候选病 {len(pool)}），"
          f"相互作用 {sum(c['kind'] == 'interaction' for c in cases)}（点名候选 {len(named)}、点类候选 {len(classed)}）→ {args.out}")


if __name__ == "__main__":
    main()
