"""RAG 主流程。每个环节都可开关，评测脚本（evaluation/run_rag_eval.py）逐个打开做消融，得到迭代记录。"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cache import RAGCache
from .corpus import Doc, docs_from_kb, docs_from_sharegpt_qa
from .rank import LexicalReranker, compress_evidence, rrf
from .retrievers import BM25Retriever, KGRetriever
from .rewrite import QueryRewriter

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RAGConfig:
    rewrite: bool = True
    routes: tuple[str, ...] = ("bm25", "kg")       # 可加 "dense" / "online"
    route_weights: dict[str, float] = field(default_factory=lambda: {"bm25": 1.0, "dense": 1.0, "kg": 1.2, "online": 0.6,
                                                                     "fts5": 0.9})
    recall_k: int = 50
    rerank: str = "lexical"                         # none | lexical | cross_encoder
    rerank_k: int = 20
    fusion_blend: float = 0.5                       # 重排分(归一化) + blend × 融合分(归一化)；0 = 纯重排
    rerank_query: str = "expanded"                  # expanded（含标准术语）| cleaned（仅去噪）
    pin_kg: bool = True                             # 重排时保住 KG 路由命中的 KB 条目
    compress: bool = True
    token_budget: int = 600
    cache: bool = True


@dataclass
class RAGResult:
    query: str
    rewritten: list[str]
    docs: list[Doc]
    evidence: list[dict[str, Any]]
    from_cache: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, docs: list[Doc], cfg: RAGConfig | None = None, *, dense: Any = None, reranker: Any = None,
                 online: list[Any] | None = None, cache: RAGCache | None = None, kb_version: str = "",
                 fts5: Any = None) -> None:
        self.cfg = cfg or RAGConfig()
        self.docs = docs
        self.by_id = {d.id: d for d in docs}
        self.rewriter = QueryRewriter()
        self.retrievers: dict[str, Any] = {"bm25": BM25Retriever(docs), "kg": KGRetriever(docs)}
        if dense is not None:
            self.retrievers["dense"] = dense
        self.online = online or []
        self.fts5 = fts5
        self.reranker = reranker or LexicalReranker()
        self.cache = cache if cache is not None else (RAGCache(kb_version) if self.cfg.cache else None)

    def _recall(self, queries: list[str]) -> tuple[dict[str, list[tuple[str, float]]], dict[str, Doc]]:
        lists: dict[str, list[tuple[str, float]]] = {}
        extra: dict[str, Doc] = {}
        for qi, q in enumerate(queries):
            for name in self.cfg.routes:
                if name == "fts5" and self.fts5 is not None:
                    found = self.fts5.search(q, k=self.cfg.recall_k)
                    for d in found:
                        extra[d.id] = d
                    lists[f"fts5:{qi}"] = [(d.id, 1.0) for d in found]
                    continue
                if name == "online":
                    for src in self.online:
                        found = src.search(q, k=3)
                        for d in found:
                            extra[d.id] = d
                        lists[f"online:{src.name}:{qi}"] = [(d.id, 1.0) for d in found]
                    continue
                r = self.retrievers.get(name)
                if r is None:
                    continue
                lists[f"{name}:{qi}"] = [(self.docs[i].id, s) for i, s in r.search(q, k=self.cfg.recall_k)]
        return lists, extra

    def search(self, query: str, top_k: int = 5) -> RAGResult:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        if self.cache is not None:
            hit = self.cache.get(query, top_k)
            if hit is not None:
                return RAGResult(query, hit["rewritten"], [self._doc(x) for x in hit["doc_ids"]], hit["evidence"],
                                 from_cache=True, timings_ms={"total": (time.perf_counter() - t0) * 1000})
        rq = self.rewriter.rewrite(query) if self.cfg.rewrite else None
        queries = rq.all_queries() if rq else [query]
        timings["rewrite"] = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        lists, extra = self._recall(queries)
        weights = {k: self.cfg.route_weights.get(k.split(":")[0], 1.0) for k in lists}
        fused = rrf(lists, weights=weights)
        timings["recall_fuse"] = (time.perf_counter() - t1) * 1000

        head = fused[: max(self.cfg.rerank_k, top_k)]
        pool = [self.by_id.get(i) or extra[i] for i, _ in head]
        t2 = time.perf_counter()
        if self.cfg.rerank != "none" and pool:
            # 用扩展后的查询重排（含标准术语），否则“美林”查不到写着“布洛芬”的说明书
            rq_text = (rq.expanded if self.cfg.rerank_query == "expanded" else rq.cleaned) if rq else query
            scores = self.reranker.score(rq_text, pool)
            lo, hi = min(scores), max(scores)
            norm = [(s - lo) / (hi - lo) if hi > lo else 0.0 for s in scores]
            top_fused = head[0][1] or 1.0
            blend = [n + self.cfg.fusion_blend * (f / top_fused) for n, (_, f) in zip(norm, head)]
            if self.cfg.pin_kg:
                # 实体链接命中的 KB 条目是高精度、可溯源的权威证据，重排不得把它挤出前列
                kg_ids = {i for k, hits in lists.items() if k.startswith("kg:") for i, _ in hits}
                blend = [b + (10.0 if d.id in kg_ids else 0.0) for b, d in zip(blend, pool)]
            order = sorted(range(len(pool)), key=lambda j: (-blend[j], j))
            pool = [pool[j] for j in order]
        timings["rerank"] = (time.perf_counter() - t2) * 1000
        top = pool[:top_k]

        t3 = time.perf_counter()
        if self.cfg.compress:
            evidence = compress_evidence(rq.cleaned if rq else query, top, token_budget=self.cfg.token_budget)
        else:
            evidence = [{"id": d.id, "source": d.source, "title": d.title, "text": d.text, "rank": r}
                        for r, d in enumerate(top)]
        timings["compress"] = (time.perf_counter() - t3) * 1000
        timings["total"] = (time.perf_counter() - t0) * 1000
        result = RAGResult(query, queries, top, evidence, timings_ms=timings)
        if self.cache is not None:
            for d in top:
                if d.id in extra:
                    self.by_id.setdefault(d.id, d)
            self.cache.put(query, top_k, {"rewritten": queries, "doc_ids": [d.id for d in top], "evidence": evidence})
        return result

    def _doc(self, doc_id: str) -> Doc:
        return self.by_id.get(doc_id) or Doc(doc_id, "", source="cache")


def default_corpus() -> list[Doc]:
    docs = docs_from_kb()
    qa = REPO_ROOT / "training" / "data" / "raw_medical_qa.jsonl"
    extra = os.environ.get("MEDAGENT_CORPUS")
    if qa.exists():
        docs += docs_from_sharegpt_qa(qa, source="medical_qa")
    if extra:
        from .corpus import docs_from_jsonl
        for p in extra.split(os.pathsep):
            docs += docs_from_jsonl(p, source=Path(p).stem)
    return docs


def build_default_pipeline(dense: bool = False, rerank: str = "lexical", online: bool | None = None) -> RAGPipeline:
    from ..knowledge import load_kb
    docs = default_corpus()
    routes = ["bm25", "kg"]
    dense_r = reranker = None
    if dense:
        from .retrievers import DenseRetriever
        dense_r = DenseRetriever(docs, model_name=os.environ.get("MEDAGENT_EMBED_MODEL", "BAAI/bge-small-zh-v1.5"))
        routes.append("dense")
    if rerank == "cross_encoder":
        from .rank import CrossEncoderReranker
        reranker = CrossEncoderReranker(os.environ.get("MEDAGENT_RERANK_MODEL", "BAAI/bge-reranker-base"))
    online = os.environ.get("MEDAGENT_ONLINE_KB") == "1" if online is None else online
    sources = []
    if online:
        from .sources import EuropePMCSearch, MedlinePlusTopics, OpenFDALabel, PubMedSearch
        sources = [OpenFDALabel(), PubMedSearch(), EuropePMCSearch(), MedlinePlusTopics()]
        routes.append("online")
    fts5 = None
    fts5_path = os.environ.get("MEDAGENT_FTS5_INDEX", str(REPO_ROOT / ".cache" / "index" / "medical.fts5"))
    if os.path.exists(fts5_path):
        from .sources import FTS5Source
        fts5 = FTS5Source(fts5_path)
        routes.append("fts5")
    redis_client = None
    if os.environ.get("MEDAGENT_REDIS_URL"):
        import redis
        redis_client = redis.Redis.from_url(os.environ["MEDAGENT_REDIS_URL"], socket_timeout=0.2)
    kbv = load_kb().version
    cache = RAGCache(kbv, redis_client=redis_client,
                     encoder=(dense_r.encode_query if dense_r is not None else None))
    return RAGPipeline(docs, RAGConfig(routes=tuple(routes), rerank=rerank), dense=dense_r, reranker=reranker,
                       online=sources, cache=cache, kb_version=kbv, fts5=fts5)
