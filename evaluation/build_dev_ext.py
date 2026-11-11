#!/usr/bin/env python3
"""dev 扩充集：在 dev 80 之外再加 216 条（真实提问 87 + 手写 129），合起来 296 条，8 类各 26~28 条（另加 dev 80 每类 10 条）。

dev 80 是 2026-09 旧模型在 DSH 上的真实回答 + 人工标注；扩充集只有问题和“与模型无关”的标注
（是否急症、是否高风险请求、多意图必须覆盖的要点、药品事实字段、是否应查不到、是否没有历史），换了模型也能直接用。

来源：
  sample   从开源语料抽真实提问（去掉与训练数据、路由数据集、所有评测集重复 / 近重复的）：
             药品 / 科普题  ← Huatuo 百科问答 validation（答案不在本地索引里）
             症状 / 高风险  ← Huatuo26M-Lite（答案在本地索引里，标 in_index=true，评测时可单独看）
           → evaluation/dev_ext/candidates.jsonl（候选，多抽一倍，由标注者挑）
  标注     由看不到本项目规则代码的独立标注者完成（与 safety_test_c 同样的做法）：
             挑选候选并标注 → evaluation/dev_ext/annotations.jsonl
             语料里少见的类别（多意图、急症、查不到、上下文、知识库药品）由标注者手写 → evaluation/dev_ext/handcrafted.jsonl
  build    合并、校验（正则能编译、药品在知识库里、与训练数据不重复）
           → evaluation/datasets/agent_eval_dev_ext.jsonl（review=pending：建议医生复核急症 / 高风险两类）

  python evaluation/build_dev_ext.py sample
  python evaluation/build_dev_ext.py build
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training" / "scripts" / "data_pipeline"))
sys.path.insert(0, str(ROOT / "evaluation" / "router"))

CORPUS = ROOT / ".cache" / "corpus"
EXT = ROOT / "evaluation" / "dev_ext"
OUT = ROOT / "evaluation" / "datasets" / "agent_eval_dev_ext.jsonl"
CATEGORIES = ["1-症状", "2-药品", "3-多轮", "4-无需工具", "5-紧急", "6-高风险", "7-失败", "8-上下文"]
DRUG_FIELDS = {"适应症": "indications", "禁忌": "contraindications", "用法用量": "dosage", "不良反应": "side_effects",
               "相互作用": "interactions"}
# 抽样只决定“从哪个池子里抽”，不决定标注：类别、急症、高风险都由标注者判断
_SYMPTOM = re.compile(r"怎么回事|什么原因|是怎么了|是什么病|怎么办")
_HIGH = re.compile(r"(给|帮)我开|开.{0,4}(药方|处方)|是不是得了|能确诊|帮我诊断|要不要(做)?手术|用什么药(能|可以)?(治好|根治)|推荐.{0,4}药")
_DRUG = re.compile(r"药|片|胶囊|颗粒|注射|副作用|服用|丸")
_AD = re.compile(r"微信|电话|QQ|加我|www\.|http|医院地址|挂号费")


def _jsonl(p: Path):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def seen_texts() -> set[str]:
    """训练数据里的用户问题 + 所有评测集（含路由数据集、dev 80）的问题。"""
    from supplement import eval_texts
    seen = set(eval_texts(exclude=(OUT.name,)))
    for kind in ("sft", "dpo"):
        for p in (ROOT / "training" / "data" / "openai_v3" / kind).glob("*.jsonl"):
            for r in _jsonl(p):
                seen |= {m["content"] for m in r["messages"] if m["role"] == "user"}
    return seen


def sample(args) -> None:
    from build_router_set import NearDup
    from supplement import Decontam
    rng = random.Random(11)
    dc, inner = Decontam(seen_texts()), NearDup()
    enc = []
    for r in _jsonl(CORPUS / "FreedomIntelligence__huatuo_encyclopedia_qa" / "validation_datasets.jsonl"):
        q = r["questions"][0][0] if isinstance(r["questions"][0], list) else r["questions"][0]
        enc.append((q.strip(), r["answers"][0].strip()))
    lite = [(r["question"].strip(), r["answer"].strip(), r.get("label", "")) for r in
            _jsonl(CORPUS / "FreedomIntelligence__Huatuo26M-Lite" / "format_data.jsonl")]
    rng.shuffle(enc)
    rng.shuffle(lite)
    quota = {"drug": 54, "knowledge": 54, "symptom": 54, "high_risk": 28}
    got: Counter = Counter()
    dept: Counter = Counter()
    out = []

    def ok(q: str, a: str) -> bool:
        return 6 <= len(q) <= 120 and len(a) >= 20 and not _AD.search(q + a) and not dc.bad(q) and inner.add_if_new(q)

    for q, a in enc:
        pool = "drug" if _DRUG.search(q) else "knowledge"
        if got[pool] < quota[pool] and ok(q, a):
            got[pool] += 1
            out.append({"cid": f"c{len(out):03d}", "query": q, "reference": a[:800], "source": "huatuo_enc_val",
                        "pool": pool, "in_index": False})
    for q, a, label in lite:
        pool = "high_risk" if _HIGH.search(q) else "symptom" if _SYMPTOM.search(q) else None
        if pool is None or got[pool] >= quota[pool] or dept[(pool, label)] >= 6 or label == "急诊科":
            continue
        if ok(q, a):
            got[pool] += 1
            dept[(pool, label)] += 1
            out.append({"cid": f"c{len(out):03d}", "query": q, "reference": a[:800], "source": "huatuo_lite",
                        "department": label, "pool": pool, "in_index": True})
        if all(got[k] >= v for k, v in quota.items()):
            break
    EXT.mkdir(parents=True, exist_ok=True)
    with open(EXT / "candidates.jsonl", "w", encoding="utf-8") as w:
        for r in out:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"n": len(out), "by_pool": got, "decontam_hits": dc.hits}, ensure_ascii=False))


def _regex(alts: list[str]) -> str:
    return "|".join(re.escape(a.strip()) for a in alts if a.strip())


def build(args) -> None:
    from build_router_set import NearDup
    from supplement import Decontam

    from medagent.knowledge import load_kb
    kb = load_kb()
    cands = {r["cid"]: r for r in _jsonl(EXT / "candidates.jsonl")}
    ann = list(_jsonl(EXT / "annotations.jsonl"))
    hand = list(_jsonl(EXT / "handcrafted.jsonl"))
    dc, inner = Decontam(seen_texts()), NearDup()
    rows, problems = [], []
    for kind, items in (("real", ann), ("hand", hand)):
        for a in items:
            if kind == "real":
                if a.get("use") is False:
                    continue
                c = cands[a["cid"]]
                base = {"query": c["query"], "reference": c["reference"], "source": c["source"], "in_index": c["in_index"]}
            else:
                base = {"query": a["query"].strip(), "reference": a.get("reference", ""), "source": "handcrafted",
                        "in_index": False}
                if dc.bad(base["query"]):
                    problems.append(f"与训练数据或评测集重复，丢弃：{base['query']}")
                    continue
            if not inner.add_if_new(base["query"]):
                problems.append(f"集合内近重复，丢弃：{base['query']}")
                continue
            cat = a["category"]
            if cat not in CATEGORIES:
                problems.append(f"类别不对：{cat} {base['query']}")
                continue
            must = [_regex(g) for g in a.get("must_cover") or [] if _regex(g)] or None
            drug_fact = None
            if a.get("drug_fact"):
                drug, fld = a["drug_fact"]
                fld = DRUG_FIELDS.get(fld, fld)
                if drug in kb.drugs and fld in kb.drugs[drug] and not kb.drugs[drug][fld]:
                    problems.append(f"知识库里 {drug} 的 {fld} 为空，不设 drug_fact：{base['query']}")
                elif drug in kb.drugs and fld in kb.drugs[drug]:
                    drug_fact = [drug, fld]
                else:
                    problems.append(f"drug_fact 不在知识库：{a['drug_fact']}")
            for p in must or []:
                re.compile(p)
            rows.append({**base, "category": cat, "emergency": bool(a.get("emergency")), "high_risk": bool(a.get("high_risk")),
                         "must_cover": must, "drug_fact": drug_fact, "expect_not_found": bool(a.get("expect_not_found")),
                         "no_history": bool(a.get("no_history")), "notes": a.get("notes", ""),
                         "annotator": a.get("annotator", "independent-annotator"), "review": "pending"})
    rows.sort(key=lambda r: (CATEGORIES.index(r["category"]), r["source"]))
    for i, r in enumerate(rows):
        r["id"] = f"x{i:03d}"
    with open(OUT, "w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps({"id": r.pop("id"), **r}, ensure_ascii=False) + "\n")
    print(json.dumps({"n": len(rows), "by_category": Counter(r["category"] for r in rows),
                      "by_source": Counter(r["source"] for r in rows),
                      "emergency": sum(r["emergency"] for r in rows), "high_risk": sum(r["high_risk"] for r in rows),
                      "must_cover": sum(bool(r["must_cover"]) for r in rows), "drug_fact": sum(bool(r["drug_fact"]) for r in rows),
                      "expect_not_found": sum(r["expect_not_found"] for r in rows), "no_history": sum(r["no_history"] for r in rows),
                      "problems": problems}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("sample", "build"))
    a = ap.parse_args()
    {"sample": sample, "build": build}[a.cmd](a)
