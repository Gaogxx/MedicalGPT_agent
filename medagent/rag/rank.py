"""融合（RRF）、重排（交叉编码器 / 词法回退）、证据压缩。"""
from __future__ import annotations

import logging
from collections import defaultdict

from .corpus import Doc
from .text import estimate_tokens, jaccard, ngram_set, split_sentences, tokenize

log = logging.getLogger(__name__)


def rrf(ranked_lists: dict[str, list[tuple[int, float]]], k: int = 60,
        weights: dict[str, float] | None = None) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion：只用名次不用分数，天然免去 BM25 分数与余弦相似度量纲不一致的问题。"""
    weights = weights or {}
    agg: dict[int, float] = defaultdict(float)
    for name, hits in ranked_lists.items():
        w = weights.get(name, 1.0)
        for rank, (i, _) in enumerate(hits):
            agg[i] += w / (k + rank + 1)
    return sorted(agg.items(), key=lambda x: -x[1])


class LexicalReranker:
    """无模型回退：查询词覆盖率（标准术语加权）× 位置先验。"""

    name = "lexical"

    def score(self, query: str, docs: list[Doc]) -> list[float]:
        q = set(tokenize(query, "word")) or set(tokenize(query, "bigram"))
        out = []
        for d in docs:
            t = set(tokenize(f"{d.title} {d.text}", "word"))
            out.append(len(q & t) / (len(q) or 1))
        return out


class CrossEncoderReranker:
    """交叉编码器重排。默认 BAAI/bge-reranker-base；可换 Qwen3-Reranker-0.6B / bge-reranker-v2-m3。"""

    name = "cross_encoder"

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str | None = None,
                 max_length: int = 384) -> None:
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)
        self.model_name = model_name

    def score(self, query: str, docs: list[Doc]) -> list[float]:
        if not docs:
            return []
        return [float(s) for s in self.model.predict([(query, f"{d.title} {d.text}") for d in docs],
                                                     batch_size=32, show_progress_bar=False)]


def compress_evidence(query: str, docs: list[Doc], token_budget: int = 600, min_sent_score: float = 0.0,
                      dedup_threshold: float = 0.8) -> list[dict]:
    """抽取式压缩：句子级打分（查询词覆盖），跨文档去近重复，按预算保留，原文顺序输出，每句保留出处。
    不用生成式摘要：医疗证据改写有引入错误的风险，抽取式保证每个字都来自原文。"""
    q = set(tokenize(query, "word")) | set(tokenize(query, "bigram"))
    cands = []
    for rank, d in enumerate(docs):
        for j, s in enumerate(split_sentences(d.text)):
            toks = set(tokenize(s, "word")) | set(tokenize(s, "bigram"))
            score = len(q & toks) / (len(q) or 1) + 0.15 / (rank + 1) + (0.05 if j == 0 else 0.0)
            cands.append((score, rank, j, s, d))
    cands.sort(key=lambda x: -x[0])
    kept, kept_grams, used = [], [], 0
    for score, rank, j, s, d in cands:
        if score < min_sent_score:
            break
        g = ngram_set(s)
        if any(jaccard(g, kg) >= dedup_threshold for kg in kept_grams):
            continue
        cost = estimate_tokens(s)
        if used + cost > token_budget:
            continue
        kept.append((rank, j, s, d, score))
        kept_grams.append(g)
        used += cost
    kept.sort(key=lambda x: (x[0], x[1]))
    grouped: dict[str, dict] = {}
    for rank, j, s, d, score in kept:
        e = grouped.setdefault(d.id, {"id": d.id, "source": d.source, "title": d.title, "text": "", "rank": rank})
        e["text"] += s
    return sorted(grouped.values(), key=lambda e: e["rank"])
