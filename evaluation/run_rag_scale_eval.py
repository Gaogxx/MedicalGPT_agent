#!/usr/bin/env python3
"""百万级检索评测：Huatuo 百科问答 1000 道测试题作查询；测试集答案（不带问题）混在百万篇文档里，看能不能找回来。

公平性：与标准答案文本完全相同的其他文档（百科里常见的说明书模板句）也算命中。
对比：FTS5 BM25（去高频词 vs 不去）/ + 交叉编码器重排（bge-reranker-base，抽样 300 题，因为 CPU/MPS 上较慢）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medagent.rag.scalable import CORPUS, FTS5Index, _jsonl  # noqa: E402
from medagent.rag.text import _NON_WORD  # noqa: E402


def h(text: str) -> str:
    return hashlib.md5(_NON_WORD.sub("", text).encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(ROOT / ".cache" / "index" / "medical.fts5"))
    ap.add_argument("--n-rerank", type=int, default=300)
    ap.add_argument("--tag", default="it1")
    args = ap.parse_args()
    idx = FTS5Index(args.index)
    N = len(idx)
    test = list(_jsonl(CORPUS / "FreedomIntelligence__huatuo_encyclopedia_qa" / "test_datasets.jsonl"))
    queries = []
    gold_hash = {}
    for i, r in enumerate(test):
        q = r["questions"][0][0] if isinstance(r["questions"][0], list) else r["questions"][0]
        queries.append((q, f"huatuo_enc_test:{i}"))
        gold_hash.setdefault(h(r["answers"][0]), set()).add(f"huatuo_enc_test:{i}")
    # 文本完全相同的文档 → 同一等价类
    equiv: dict[str, set[int]] = {}
    for rowid, doc_id, text in idx.db.execute("SELECT id, doc_id, text FROM docs"):
        k = h(text)
        if k in gold_hash:
            equiv.setdefault(k, set()).add(rowid)
    gold_rows = {}
    for k, ids in gold_hash.items():
        for gid in ids:
            gold_rows[gid] = equiv.get(k, set())
    res: dict = {"index_docs": N, "queries": len(queries), "index_size_gb": round(Path(args.index).stat().st_size / 1e9, 2)}

    def run(max_df: float, k: int = 50) -> tuple[dict, list]:
        lat, r5, r10, r50, mrr, cand = [], [], [], [], [], []
        for q, gid in queries:
            t = time.perf_counter()
            hits = idx.search(q, k=k, max_df_ratio=max_df)
            lat.append((time.perf_counter() - t) * 1000)
            rows = [x[0] for x in hits]
            g = gold_rows[gid]
            rank = next((i for i, x in enumerate(rows) if x in g), None)
            r5.append(rank is not None and rank < 5); r10.append(rank is not None and rank < 10); r50.append(rank is not None)
            mrr.append(1 / (rank + 1) if rank is not None and rank < 10 else 0.0)
            cand.append((q, gid, rows))
        lat.sort()
        return {"recall@5": round(sum(r5) / len(r5), 4), "recall@10": round(sum(r10) / len(r10), 4),
                "recall@50": round(sum(r50) / len(r50), 4), "mrr@10": round(statistics.mean(mrr), 4),
                "latency_ms_p50": round(lat[len(lat) // 2], 1), "latency_ms_p95": round(lat[int(len(lat) * 0.95)], 1)}, cand

    res["bm25_all_terms"], _ = run(max_df=1.0)
    print("bm25_all_terms", res["bm25_all_terms"], flush=True)
    res["bm25_df_pruned"], cand = run(max_df=0.05)
    print("bm25_df_pruned", res["bm25_df_pruned"], flush=True)

    from medagent.rag.rank import CrossEncoderReranker
    ce = CrossEncoderReranker()
    rng = random.Random(0)
    sample = rng.sample(cand, min(args.n_rerank, len(cand)))
    r5, mrr, lat, base5 = [], [], [], []
    for q, gid, rows in sample:
        docs = idx.get(rows)
        t = time.perf_counter()
        scores = ce.score(q, docs) if docs else []
        lat.append((time.perf_counter() - t) * 1000)
        order = [docs[i].meta["rowid"] for i in sorted(range(len(docs)), key=lambda i: -scores[i])]
        g = gold_rows[gid]
        rank = next((i for i, x in enumerate(order) if x in g), None)
        r5.append(rank is not None and rank < 5)
        mrr.append(1 / (rank + 1) if rank is not None and rank < 10 else 0.0)
        brank = next((i for i, x in enumerate(rows) if x in g), None)
        base5.append(brank is not None and brank < 5)
    res["rerank_sample"] = {"n": len(sample), "bm25_recall@5_same_sample": round(sum(base5) / len(base5), 4),
                            "rerank_recall@5": round(sum(r5) / len(r5), 4), "rerank_mrr@10": round(statistics.mean(mrr), 4),
                            "rerank_latency_ms_mean(50 docs)": round(statistics.mean(lat), 1)}
    print("rerank", res["rerank_sample"], flush=True)
    (ROOT / "evaluation" / "results" / f"rag_scale_eval_{args.tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                                                                  encoding="utf-8")


if __name__ == "__main__":
    main()
