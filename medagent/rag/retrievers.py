"""三路本地召回：BM25（稀疏）/ 向量（稠密）/ 结构化 KG（实体链接直达 KB 条目）。"""
from __future__ import annotations

import logging
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .corpus import Doc, corpus_fingerprint
from .text import tokenize

log = logging.getLogger(__name__)
Hit = tuple[int, float]  # (doc index, score)


class BM25Retriever:
    name = "bm25"

    def __init__(self, docs: list[Doc], k1: float = 1.5, b: float = 0.75, mode: str = "mixed") -> None:
        self.docs, self.k1, self.b, self.mode = docs, k1, b, mode
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len = np.zeros(len(docs), dtype=np.float32)
        for i, d in enumerate(docs):
            toks = tokenize(f"{d.title} {d.text}", mode)
            self.doc_len[i] = len(toks)
            for t, c in Counter(toks).items():
                self.postings[t].append((i, c))
        n = len(docs)
        self.avgdl = float(self.doc_len.mean()) if n else 0.0
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def search(self, query: str, k: int = 50) -> list[Hit]:
        scores = np.zeros(len(self.docs), dtype=np.float32)
        for t in set(tokenize(query, self.mode)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in self.postings[t]:
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                scores[i] += idf * tf * (self.k1 + 1) / denom
        if not scores.any():
            return []
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        return sorted(((int(i), float(scores[i])) for i in top if scores[i] > 0), key=lambda x: -x[1])


class DenseRetriever:
    """sentence-transformers 向量召回。默认 BAAI/bge-small-zh-v1.5（~100MB，CPU/MPS 可跑）；
    生产可换 Qwen3-Embedding-0.6B / bge-m3。向量按 (模型, 语料指纹) 缓存到磁盘。"""

    name = "dense"
    QUERY_PREFIX = {"BAAI/bge-small-zh-v1.5": "为这个句子生成表示以用于检索相关文章：",
                    "BAAI/bge-base-zh-v1.5": "为这个句子生成表示以用于检索相关文章："}

    def __init__(self, docs: list[Doc], model_name: str = "BAAI/bge-small-zh-v1.5",
                 cache_dir: str | None = None, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer
        self.docs = docs
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        cache_dir = cache_dir or os.environ.get("MEDAGENT_CACHE_DIR", ".cache/medagent")
        path = Path(cache_dir) / f"emb_{model_name.replace('/', '_')}_{corpus_fingerprint(docs)}.npy"
        if path.exists():
            self.emb = np.load(path)
        else:
            self.emb = self.model.encode([f"{d.title} {d.text}" for d in docs], batch_size=64,
                                         normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, self.emb)

    def encode_query(self, query: str) -> np.ndarray:
        prefix = self.QUERY_PREFIX.get(self.model_name, "")
        return self.model.encode([prefix + query], normalize_embeddings=True, show_progress_bar=False)[0]

    def search(self, query: str, k: int = 50) -> list[Hit]:
        q = self.encode_query(query)
        sims = self.emb @ q
        top = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        return sorted(((int(i), float(sims[i])) for i in top), key=lambda x: -x[1])


class KGRetriever:
    """结构化路由：实体链接命中 KB 的症状/药品/疾病时，直接返回对应 KB 文档（精确、可解释，带指南出处）。"""

    name = "kg"

    def __init__(self, docs: list[Doc]) -> None:
        from ..knowledge import load_kb
        self.kb = load_kb()
        self.index = {d.id: i for i, d in enumerate(docs)}

    def search(self, query: str, k: int = 10) -> list[Hit]:
        ents = self.kb.link_entities(query)
        hits = []
        for kind, names in ents.items():
            prefix = {"symptom": "kb:symptom:", "drug": "kb:drug:", "disease": "kb:guideline:"}[kind]
            for n in names:
                i = self.index.get(prefix + n)
                if i is not None:
                    hits.append((i, 1.0))
        return hits[:k]
