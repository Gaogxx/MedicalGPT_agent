#!/usr/bin/env python3
"""RAG 逐环节消融（每一行就是一次迭代）。两个评测集：
  kb   45 条口语化查询 → 内置 KB 条目（症状/药品/指南），考察查询改写与结构化路由
  qa   从 raw_medical_qa 随机抽 300 个真实问题 → 其原答案的切块，考察大语料下的混合检索与重排
指标：Recall@5、MRR@10、nDCG@10、平均延迟；压缩：证据 token 数与“压缩后仍保留金标文档”的比例；缓存：命中率与延迟。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medagent.rag.cache import RAGCache  # noqa: E402
from medagent.rag.corpus import docs_from_kb, docs_from_sharegpt_qa  # noqa: E402
from medagent.rag.pipeline import RAGConfig, RAGPipeline  # noqa: E402
from medagent.rag.text import estimate_tokens  # noqa: E402


def metrics(ranked: list[str], rel: set[str]) -> tuple[float, float, float]:
    r5 = len(set(ranked[:5]) & rel) / len(rel)
    mrr = next((1 / (i + 1) for i, d in enumerate(ranked[:10]) if d in rel), 0.0)
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if d in rel)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), 10)))
    return r5, mrr, dcg / idcg


def run(pipe: RAGPipeline, qs: list[dict]) -> dict:
    r5s, mrrs, ndcgs, lat, toks, kept = [], [], [], [], [], []
    for q in qs:
        pipe.cache = None
        t = time.perf_counter()
        res = pipe.search(q["query"], top_k=10)
        lat.append((time.perf_counter() - t) * 1000)
        ranked = [d.id for d in res.docs]
        rel = set(q["relevant"])
        a, b, c = metrics(ranked, rel)
        r5s.append(a); mrrs.append(b); ndcgs.append(c)
        if pipe.cfg.compress:
            ev5 = [e for e in res.evidence if e["rank"] < 5]
            toks.append(sum(estimate_tokens(e["text"]) for e in ev5))
            gold_top5 = rel & set(ranked[:5])
            if gold_top5:
                kept.append(bool(gold_top5 & {e["id"] for e in ev5}))
        else:
            toks.append(sum(estimate_tokens(d.text) for d in res.docs[:5]))
    out = {"recall@5": round(statistics.mean(r5s), 4), "mrr@10": round(statistics.mean(mrrs), 4),
           "ndcg@10": round(statistics.mean(ndcgs), 4), "latency_ms_mean": round(statistics.mean(lat), 1),
           "evidence_tokens_top5": round(statistics.mean(toks), 1)}
    if kept:
        out["gold_kept_after_compress"] = round(sum(kept) / len(kept), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--cross-encoder", action="store_true")
    ap.add_argument("--n-qa", type=int, default=300)
    ap.add_argument("--tag", default="it1")
    args = ap.parse_args()
    kb_docs = docs_from_kb()
    qa_path = ROOT / "training" / "data" / "raw_medical_qa.jsonl"
    qa_docs = docs_from_sharegpt_qa(qa_path, source="medical_qa")
    docs = kb_docs + qa_docs
    kb_q = [json.loads(l) for l in open(ROOT / "evaluation" / "datasets" / "rag_kb_qrels.jsonl", encoding="utf-8")]
    rows = [json.loads(l) for l in open(qa_path, encoding="utf-8")]
    by_qid: dict[int, list[str]] = {}
    for d in qa_docs:
        by_qid.setdefault(d.meta["qid"], []).append(d.id)
    rng = random.Random(0)
    qids = rng.sample(sorted(by_qid), args.n_qa)
    qa_q = [{"query": next(c["value"] for c in rows[i]["conversations"] if c["from"] == "human"), "relevant": by_qid[i]}
            for i in qids]

    dense = reranker = None
    if args.dense:
        from medagent.rag.retrievers import DenseRetriever
        dense = DenseRetriever(docs)
    if args.cross_encoder:
        from medagent.rag.rank import CrossEncoderReranker
        reranker = CrossEncoderReranker()

    base = dict(cache=False, compress=False)
    configs: list[tuple[str, dict, bool, bool]] = [
        ("R0 BM25", dict(rewrite=False, routes=("bm25",), rerank="none", **base), False, False),
        ("R1 +KG路由(RRF)", dict(rewrite=False, routes=("bm25", "kg"), rerank="none", **base), False, False),
        ("R2 +查询改写", dict(rewrite=True, routes=("bm25", "kg"), rerank="none", **base), False, False),
        ("R3 +词法重排(改写前查询,纯重排)", dict(rewrite=True, routes=("bm25", "kg"), rerank="lexical",
                                         rerank_query="cleaned", fusion_blend=0.0, pin_kg=False, **base), False, False),
        ("R4 词法重排修正(扩展查询+融合分)", dict(rewrite=True, routes=("bm25", "kg"), rerank="lexical",
                                          rerank_query="expanded", fusion_blend=0.5, pin_kg=False, **base), False, False),
        ("R4b +KG命中置顶", dict(rewrite=True, routes=("bm25", "kg"), rerank="lexical", pin_kg=True, **base), False, False),
    ]
    if dense is not None:
        configs += [
            ("R5 +向量召回(bge-small-zh, 三路RRF)", dict(rewrite=True, routes=("bm25", "kg", "dense"), rerank="none", **base), True, False),
        ]
    if reranker is not None:
        routes = ("bm25", "kg", "dense") if dense else ("bm25", "kg")
        configs += [
            ("R6 +交叉编码器重排(bge-reranker-base)", dict(rewrite=True, routes=routes, rerank="cross_encoder",
                                                     pin_kg=False, **base), dense is not None, True),
            ("R6b 交叉编码器+KG置顶", dict(rewrite=True, routes=routes, rerank="cross_encoder", pin_kg=True, **base),
             dense is not None, True),
        ]
    last = configs[-1]
    configs.append(("R7 +抽取式压缩(600 token)", {**last[1], "compress": True}, last[2], last[3]))

    results = {}
    for name, kw, use_dense, use_ce in configs:
        cfg = RAGConfig(**kw)
        pipe = RAGPipeline(docs, cfg, dense=dense if use_dense else None, reranker=reranker if use_ce else None)
        results[name] = {"kb": run(pipe, kb_q), "qa": run(pipe, qa_q)}
        print(name, json.dumps(results[name], ensure_ascii=False))

    # 缓存：模拟真实流量——200 条查询，其中 30% 精确重复、20% 仅标点/空格不同
    final = configs[-1]
    pipe = RAGPipeline(docs, RAGConfig(**{**final[1], "cache": True}), dense=dense if final[2] else None,
                       reranker=reranker if final[3] else None,
                       cache=RAGCache("kbv", encoder=dense.encode_query if dense else None))
    stream = [q["query"] for q in qa_q[:100]]
    stream += rng.choices(stream, k=60) + [s.replace("？", "?").replace("，", " ") + "？" for s in rng.choices(stream, k=40)]
    rng.shuffle(stream)
    hit_lat, miss_lat = [], []
    for q in stream:
        t = time.perf_counter()
        r = pipe.search(q, top_k=10)
        (hit_lat if r.from_cache else miss_lat).append((time.perf_counter() - t) * 1000)
    results["cache"] = {"queries": len(stream), "hit_rate": round(pipe.cache.hit_rate, 4), "stats": pipe.cache.stats,
                        "latency_ms_hit": round(statistics.mean(hit_lat), 2) if hit_lat else None,
                        "latency_ms_miss": round(statistics.mean(miss_lat), 1) if miss_lat else None}
    print("cache", results["cache"])
    out = ROOT / "evaluation" / "results" / f"rag_eval_{args.tag}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
