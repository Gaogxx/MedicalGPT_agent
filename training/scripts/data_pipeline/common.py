"""数据管线公共工具：读写 jsonl、文本规范化、近重复检测（MinHash-LSH，纯 numpy）。"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

_NORM = re.compile(r"[\s\W_]+", re.U)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def norm(text: str) -> str:
    return _NORM.sub("", text.lower())


def human(row: dict[str, Any]) -> str:
    return next((c["value"] for c in row.get("conversations", []) if c.get("from") == "human"), "")


def turns(row: dict[str, Any], role: str) -> list[str]:
    return [c["value"] for c in row.get("conversations", []) if c.get("from") == role]


def row_key(row: dict[str, Any]) -> str:
    parts = [norm(c.get("value", "")) for c in row.get("conversations", [])]
    for k in ("chosen", "rejected"):
        if k in row:
            parts.append(norm(str(row[k])))
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def shingles(text: str, n: int = 5) -> set[str]:
    s = norm(text)
    return {s[i:i + n] for i in range(max(len(s) - n + 1, 1))}


class MinHashLSH:
    """近重复检测：128 个哈希 → 32 个 band × 4 行。Jaccard≈0.8 的对以高概率落进同一桶。"""

    def __init__(self, num_perm: int = 128, bands: int = 32, seed: int = 1) -> None:
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, 2**31 - 1, num_perm, dtype=np.int64)
        self.b = rng.integers(0, 2**31 - 1, num_perm, dtype=np.int64)
        self.bands, self.rows = bands, num_perm // bands
        self.buckets: dict[tuple, list[int]] = defaultdict(list)
        self.sigs: list[np.ndarray] = []

    def signature(self, sh: set[str]) -> np.ndarray:
        if not sh:
            return np.full(len(self.a), 2**31 - 1, dtype=np.int64)
        h = np.array([int(hashlib.md5(x.encode()).hexdigest()[:8], 16) for x in sh], dtype=np.int64)
        return ((np.outer(h, self.a) + self.b) % (2**31 - 1)).min(axis=0)

    def add(self, sig: np.ndarray) -> int:
        idx = len(self.sigs)
        self.sigs.append(sig)
        for bnd in range(self.bands):
            self.buckets[(bnd, tuple(sig[bnd * self.rows:(bnd + 1) * self.rows]))].append(idx)
        return idx

    def candidates(self, sig: np.ndarray) -> set[int]:
        out: set[int] = set()
        for bnd in range(self.bands):
            out.update(self.buckets.get((bnd, tuple(sig[bnd * self.rows:(bnd + 1) * self.rows])), []))
        return out

    def similarity(self, i: int, sig: np.ndarray) -> float:
        return float((self.sigs[i] == sig).mean())


def iter_function_calls(row: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for c in row.get("conversations", []):
        if c.get("from") == "function_call":
            try:
                yield json.loads(c["value"])
            except json.JSONDecodeError:
                yield {"_invalid": c["value"]}
    for k in ("chosen", "rejected"):
        v = row.get(k, "")
        m = re.search(r"Action:\s*(\w+)\s*\nAction Input:\s*(\{.*\})", str(v), re.S)
        if m:
            try:
                yield {"name": m.group(1), "arguments": json.loads(m.group(2)), "_from": k}
            except json.JSONDecodeError:
                yield {"_invalid": v, "_from": k}
