#!/usr/bin/env python3
"""路由器的独立数据集：和 dev 80（只做评测）、训练数据、所有评测集都不重叠。

为什么要它：原来的 MF 路由器用 dev 80 条训练，又在这 80 条上看路由效果，成绩是样本内的；而且 80 条太少。
这里从开源问诊语料里抽 2,000 条真实提问，去掉与训练数据 / 评测集重复或近重复的，再按问题哈希切成
train 1,400 / val 300 / test 300：train 训练，val 定阈值，test 报告效果；dev 80 保持完全不参与训练，作为人工标注的外部测试。

来源（都带参考答案，供裁判对照）：Huatuo26M-Lite（真实问诊，主体）、shibing624/medical（问诊）、
Huatuo 百科问答 validation（百科式提问）。test 集答案已进本地索引的那部分（huatuo test）不用。

  python evaluation/router/build_router_set.py      # → evaluation/datasets/router_set.jsonl
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training" / "scripts" / "data_pipeline"))

CORPUS = ROOT / ".cache" / "corpus"
QUOTA = {"huatuo_lite": 1300, "shibing": 450, "huatuo_enc_val": 250}
SPLIT = (("train", 0.70), ("val", 0.15), ("test", 0.15))
_AD = re.compile(r"微信|电话|QQ|加我|www\.|http|医院地址|挂号费")


def _jsonl(p: Path):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def candidates(rng: random.Random):
    lite = [(r["question"], r["answer"]) for r in _jsonl(CORPUS / "FreedomIntelligence__Huatuo26M-Lite" / "format_data.jsonl")]
    shib = [((r.get("instruction", "") + r.get("input", "")).strip(), r.get("output", ""))
            for r in _jsonl(CORPUS / "shibing624__medical" / "finetune" / "train_zh_0.json")]
    enc = [(r["questions"][0][0], r["answers"][0])
           for r in _jsonl(CORPUS / "FreedomIntelligence__huatuo_encyclopedia_qa" / "validation_datasets.jsonl")]
    for name, rows in (("huatuo_lite", lite), ("shibing", shib), ("huatuo_enc_val", enc)):
        rng.shuffle(rows)
        yield name, rows


class NearDup:
    """增量近重复检测：3-gram Jaccard ≥ thr 视为重复（倒排索引）。"""

    def __init__(self, thr: float = 0.6) -> None:
        from collections import defaultdict
        from supplement import ngram_set
        self.ngram_set, self.thr = ngram_set, thr
        self.grams: list[set] = []
        self.inv = defaultdict(list)

    def add_if_new(self, q: str) -> bool:
        g = self.ngram_set(q, 3)
        cnt = Counter(i for x in g for i in self.inv.get(x, ()))
        if any(inter / max(len(g | self.grams[i]), 1) >= self.thr for i, inter in cnt.items()):
            return False
        for x in g:
            self.inv[x].append(len(self.grams))
        self.grams.append(g)
        return True


def split_of(q: str) -> str:
    h = int(hashlib.sha1(q.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for name, frac in SPLIT:
        acc += frac
        if h < acc:
            return name
    return SPLIT[-1][0]


def main() -> None:
    from supplement import Decontam, eval_texts
    rng = random.Random(7)
    # 去污染：所有评测集 + 训练数据里出现过的用户问题
    seen = set(eval_texts(exclude=("router_set.jsonl",)))   # 不能拿上一次生成的自己去污染自己
    for p in (ROOT / "training" / "data" / "openai_v3" / "sft").glob("*.jsonl"):
        for r in _jsonl(p):
            seen |= {m["content"] for m in r["messages"] if m["role"] == "user"}
    for p in (ROOT / "training" / "data" / "openai_v3" / "dpo").glob("*.jsonl"):
        for r in _jsonl(p):
            seen |= {m["content"] for m in r["messages"] if m["role"] == "user"}
    dc = Decontam(seen)
    picked, inner, dups = [], NearDup(), 0
    for name, rows in candidates(rng):
        n = 0
        for q, a in rows:
            if n >= QUOTA[name]:
                break
            q, a = q.strip(), (a or "").strip()
            if not (6 <= len(q) <= 120) or len(a) < 20 or _AD.search(q + a) or dc.bad(q):
                continue
            if not inner.add_if_new(q):          # 集合内部近重复：避免同一问法同时出现在 train 和 test
                dups += 1
                continue
            picked.append({"id": f"rs{len(picked):05d}", "query": q, "reference": a[:1500], "source": name,
                           "split": split_of(q)})
            n += 1
    out = ROOT / "evaluation" / "datasets" / "router_set.jsonl"
    with open(out, "w", encoding="utf-8") as w:
        for r in picked:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"n": len(picked), "by_split": Counter(r["split"] for r in picked),
                      "by_source": Counter(r["source"] for r in picked), "decontam_hits": dc.hits, "near_dup_dropped": dups}, ensure_ascii=False))


if __name__ == "__main__":
    main()
