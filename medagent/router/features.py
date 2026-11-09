"""路由用的查询表示：稠密向量（bge-small-zh，可选）或 哈希字 n-gram（零依赖），外加少量可解释的手工特征。

手工特征都对应基线里本地 7B 模型真实翻车的模式：多意图（“先…再…”只做一半）、长历史（编造对话）、
KB 外实体（查不到时乱答）、急症/高风险词。"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Callable

import numpy as np

from ..knowledge import load_kb
from ..safety import assess
from ..verify.detectors import multi_intent

HAND_FEATURES = ["len_log", "multi_intent", "has_history", "n_entities", "kb_miss_hint", "emergency", "high_risk",
                 "question_concept", "bias"]


class QueryFeaturizer:
    def __init__(self, dim: int = 512, encoder: Callable[[list[str]], np.ndarray] | None = None) -> None:
        self.dim = dim
        self.encoder = encoder
        self.kb = load_kb()

    def _hash_vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        s = re.sub(r"\s+", "", text)
        for n in (1, 2, 3):
            for i in range(len(s) - n + 1):
                h = int(hashlib.md5(s[i:i + n].encode()).hexdigest()[:8], 16)
                v[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        nrm = np.linalg.norm(v)
        return v / nrm if nrm else v

    def hand(self, query: str, history: str = "") -> np.ndarray:
        a = assess(query)
        ents = self.kb.link_entities(query)
        n_ent = sum(len(v) for v in ents.values())
        miss = bool(re.search(r"查一下|这个|那个|特效|神奇|进口", query)) and n_ent == 0
        concept = bool(re.match(r"^(什么是|.{0,6}是什么|为什么)", query))
        return np.array([np.log1p(len(query) + len(history)) / 6, float(multi_intent(query)), float(bool(history)),
                         min(n_ent, 3) / 3, float(miss), float(a.level == "emergency"), float(a.high_risk),
                         float(concept), 1.0], dtype=np.float32)

    def transform(self, queries: list[str], histories: list[str] | None = None) -> np.ndarray:
        histories = histories or [""] * len(queries)
        texts = [f"{h[-200:]} {q}" if h else q for q, h in zip(queries, histories)]
        if self.encoder is not None:
            dense = np.asarray(self.encoder(texts), dtype=np.float32)
        else:
            dense = np.stack([self._hash_vec(t) for t in texts])
        hand = np.stack([self.hand(q, h) for q, h in zip(queries, histories)])
        return np.concatenate([dense, hand], axis=1)


def split_history(query: str) -> tuple[str, str]:
    """评测集里长历史被拼在 query 前面（“我之前问过……。 现在……”），拆开以便特征区分。"""
    m = re.match(r"^(我之前问过.+?。)\s*(.+)$", query)
    return (m.group(2), m.group(1)) if m else (query, "")


def featurizer_from_config(cfg: dict[str, Any] | None = None) -> QueryFeaturizer:
    cfg = cfg or {}
    if cfg.get("encoder") == "bge":
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(cfg.get("model", "BAAI/bge-small-zh-v1.5"))
        return QueryFeaturizer(encoder=lambda xs: m.encode(xs, normalize_embeddings=True, show_progress_bar=False))
    return QueryFeaturizer()
