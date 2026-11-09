"""检索缓存：精确缓存（规范化查询哈希）+ 语义缓存（查询向量余弦 ≥ 阈值）。
缓存键带 kb_version：知识库一更新，旧缓存自然失效，不会把过期的药品说明返回给用户。
后端：进程内 LRU（默认）或 Redis（多实例共享，TTL 由 Redis 负责）。Redis 不可用时降级为进程内，不影响检索。"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import OrderedDict
from typing import Any, Callable

import numpy as np

_NORM = re.compile(r"[\s，,。？?！!、]+")


def normalize_query(q: str) -> str:
    return _NORM.sub("", q.lower())


class RAGCache:
    def __init__(self, kb_version: str, *, max_items: int = 2048, ttl: float = 3600.0, redis_client: Any = None,
                 encoder: Callable[[str], np.ndarray] | None = None, semantic_threshold: float = 0.95) -> None:
        self.kb_version = kb_version
        self.max_items, self.ttl = max_items, ttl
        self.r = redis_client
        self.encoder = encoder
        self.threshold = semantic_threshold
        self._lru: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._sem_keys: list[str] = []
        self._sem_vecs: list[np.ndarray] = []
        self.stats = {"exact_hit": 0, "semantic_hit": 0, "miss": 0, "redis_errors": 0}

    def _key(self, q: str, top_k: int) -> str:
        return "rag:" + hashlib.sha256(f"{self.kb_version}|{top_k}|{normalize_query(q)}".encode()).hexdigest()[:24]

    def _get(self, key: str) -> Any:
        if self.r is not None:
            try:
                raw = self.r.get(key)
                return json.loads(raw) if raw else None
            except Exception:  # noqa: BLE001
                self.stats["redis_errors"] += 1
        item = self._lru.get(key)
        if item and item[0] > time.time():
            self._lru.move_to_end(key)
            return item[1]
        return None

    def _set(self, key: str, value: Any) -> None:
        if self.r is not None:
            try:
                self.r.set(key, json.dumps(value, ensure_ascii=False), ex=int(self.ttl))
            except Exception:  # noqa: BLE001
                self.stats["redis_errors"] += 1
        self._lru[key] = (time.time() + self.ttl, value)
        self._lru.move_to_end(key)
        while len(self._lru) > self.max_items:
            self._lru.popitem(last=False)

    def get(self, q: str, top_k: int) -> Any:
        key = self._key(q, top_k)
        v = self._get(key)
        if v is not None:
            self.stats["exact_hit"] += 1
            return v
        if self.encoder is not None and self._sem_vecs:
            vec = self.encoder(q)
            sims = np.stack(self._sem_vecs) @ vec
            j = int(np.argmax(sims))
            if sims[j] >= self.threshold:
                v = self._get(self._sem_keys[j])
                if v is not None:
                    self.stats["semantic_hit"] += 1
                    return v
        self.stats["miss"] += 1
        return None

    def put(self, q: str, top_k: int, value: Any) -> None:
        key = self._key(q, top_k)
        self._set(key, value)
        if self.encoder is not None:
            self._sem_keys.append(key)
            self._sem_vecs.append(self.encoder(q))
            if len(self._sem_keys) > self.max_items:
                self._sem_keys.pop(0)
                self._sem_vecs.pop(0)

    @property
    def hit_rate(self) -> float:
        total = sum(self.stats[k] for k in ("exact_hit", "semantic_hit", "miss"))
        return (self.stats["exact_hit"] + self.stats["semantic_hit"]) / total if total else 0.0
