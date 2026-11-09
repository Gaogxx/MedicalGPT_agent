"""百万级本地检索：SQLite FTS5（磁盘倒排，BM25）+ FAISS（向量，IVF/HNSW）。

为什么不用之前的内存 BM25：它把倒排表放在 Python dict 里，几千篇文档没问题，几百万篇会吃光内存、启动要几分钟。
FTS5 是 SQLite 内置的全文索引：倒排在磁盘上、查询毫秒级、零额外服务；中文先用 jieba（带医学词典）切词再写入，
查询时去掉文档频率过高的词（“治疗”“患者”这类词会匹配上百万篇，既慢又没区分度）。
生产环境上千万文档时把同一接口换成 Elasticsearch/OpenSearch + Milvus/Qdrant 即可（见 docs/rag_sources.md）。

构建：python -m medagent.rag.scalable build --out .cache/index/medical.fts5   （语料见 iter_open_corpora）
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Iterator

from .corpus import Doc

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / ".cache" / "corpus"
_TOKENIZER = None


def _init_worker() -> None:
    global _TOKENIZER
    from .text import STOPWORDS, _jieba
    jb = _jieba()
    try:
        from ..knowledge.ext import load_ext
        ext = load_ext()
        for n in (ext.names if ext else {}):
            jb.add_word(n, freq=20000)
    except Exception:  # noqa: BLE001
        pass
    _TOKENIZER = (jb, STOPWORDS)


def _tok(text: str) -> str:
    jb, stop = _TOKENIZER
    return " ".join(w for w in jb.lcut(text.lower()) if len(w.strip()) > 0 and w not in stop and not w.isspace()
                    and not all(not ch.isalnum() for ch in w))


def _tok_batch(rows: list[tuple[str, str, str, str]]) -> list[tuple[str, str, str, str, str]]:
    return [(d, s, t, x, _tok(f"{t} {x[:600]}")) for d, s, t, x in rows]


class FTS5Index:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS docs (id INTEGER PRIMARY KEY, doc_id TEXT, source TEXT, title TEXT, text TEXT)")
        self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(tok, content='', tokenize='unicode61')")
        self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vocab USING fts5vocab(fts, 'row')")
        self._n: int | None = None
        self._df_cache: dict[str, int] = {}

    def __len__(self) -> int:
        if self._n is None:
            self._n = self.db.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        return self._n

    def build(self, rows: Iterable[tuple[str, str, str, str]], workers: int = 8, batch: int = 2000,
              log_every: int = 200_000) -> dict:
        t0, n = time.time(), 0

        def batches() -> Iterator[list]:
            buf = []
            for r in rows:
                buf.append(r)
                if len(buf) >= batch:
                    yield buf
                    buf = []
            if buf:
                yield buf

        with Pool(workers, initializer=_init_worker) as pool:
            for out in pool.imap(_tok_batch, batches(), chunksize=1):
                cur = self.db.executemany("INSERT INTO docs (doc_id, source, title, text) VALUES (?,?,?,?)",
                                          [(d, s, t, x) for d, s, t, x, _ in out])
                last = self.db.execute("SELECT MAX(id) FROM docs").fetchone()[0]
                first = last - len(out) + 1
                self.db.executemany("INSERT INTO fts (rowid, tok) VALUES (?,?)",
                                    [(first + i, o[4]) for i, o in enumerate(out)])
                n += len(out)
                if n // log_every != (n - len(out)) // log_every:
                    self.db.commit()
                    print(f"  indexed {n:,} docs in {time.time() - t0:.0f}s", flush=True)
        self.db.commit()
        self.db.execute("INSERT INTO fts(fts) VALUES('optimize')")
        self.db.commit()
        self._n = None
        return {"docs": n, "seconds": round(time.time() - t0, 1),
                "size_gb": round(os.path.getsize(self.path) / 1e9, 2)}

    def df(self, term: str) -> int:
        if term not in self._df_cache:
            r = self.db.execute("SELECT doc FROM vocab WHERE term = ?", (term,)).fetchone()
            self._df_cache[term] = r[0] if r else 0
        return self._df_cache[term]

    def search(self, query: str, k: int = 50, max_df_ratio: float = 0.05, max_terms: int = 12) -> list[tuple[int, float]]:
        global _TOKENIZER
        if _TOKENIZER is None:
            _init_worker()
        n = max(len(self), 1)
        terms = [t for t in dict.fromkeys(_tok(query).split()) if t]
        scored = [(t, self.df(t)) for t in terms]
        kept = [t for t, d in scored if 0 < d <= max_df_ratio * n] or [t for t, d in sorted(scored, key=lambda x: x[1]) if d > 0][:2]
        kept = sorted(kept, key=self.df)[:max_terms]
        if not kept:
            return []
        expr = " OR ".join('"' + t.replace('"', '') + '"' for t in kept)
        rows = self.db.execute("SELECT rowid, bm25(fts) AS s FROM fts WHERE fts MATCH ? ORDER BY s LIMIT ?", (expr, k)).fetchall()
        return [(int(r), -float(s)) for r, s in rows]

    def get(self, rowids: list[int]) -> list[Doc]:
        if not rowids:
            return []
        q = f"SELECT id, doc_id, source, title, text FROM docs WHERE id IN ({','.join('?' * len(rowids))})"
        by = {r[0]: Doc(r[1], r[4], r[3], r[2], {"rowid": r[0]}) for r in self.db.execute(q, rowids)}
        return [by[i] for i in rowids if i in by]


class FaissDense:
    """FAISS 向量索引。小于 50 万用 HNSW（无需训练），更大用 IVF-PQ（内存约 64B/向量）。"""

    def __init__(self, dim: int = 512, n_hint: int = 0) -> None:
        import faiss
        self.faiss = faiss
        if n_hint > 500_000:
            quant = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFPQ(quant, dim, min(65536, max(1024, n_hint // 256)), 64, 8, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        self.ids: list[int] = []

    def add(self, vecs, rowids: list[int]) -> None:
        import numpy as np
        v = np.asarray(vecs, dtype="float32")
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            self.index.train(v)
        self.index.add(v)
        self.ids.extend(rowids)

    def search(self, qvec, k: int = 50) -> list[tuple[int, float]]:
        import numpy as np
        D, I = self.index.search(np.asarray([qvec], dtype="float32"), k)
        return [(self.ids[i], float(d)) for d, i in zip(D[0], I[0]) if i >= 0]


# ---------------------------------------------------------------- 开源语料读取（Apache-2.0）
def _jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def iter_open_corpora(limit_per_source: int | None = None, include_test_answers: bool = True,
                      only: set[str] | None = None) -> Iterator[tuple[str, str, str, str]]:
    """产出 (doc_id, source, title, text)。title=问题（FAQ 式检索有用），text=答案。
    only ⊆ {"huatuo_enc", "huatuo_kg", "huatuo_lite", "shibing"}，用于分批建索引。"""
    def want(name: str) -> bool:
        return only is None or name in only

    def capped(it: Iterator) -> Iterator:
        for i, x in enumerate(it):
            if limit_per_source is not None and i >= limit_per_source:
                return
            yield x

    enc = CORPUS / "FreedomIntelligence__huatuo_encyclopedia_qa"
    if want("huatuo_enc"):
        for i, r in capped(enumerate(_jsonl(enc / "train_datasets.jsonl"))):
            q = r["questions"][0][0] if r["questions"] and isinstance(r["questions"][0], list) else str(r["questions"][0])
            yield f"huatuo_enc:{i}", "huatuo_encyclopedia_qa", q, r["answers"][0]
        if include_test_answers:   # 评测用：测试集答案进索引但不带问题，查询用测试问题（见 evaluation/run_rag_scale_eval.py）
            for i, r in enumerate(_jsonl(enc / "test_datasets.jsonl")):
                yield f"huatuo_enc_test:{i}", "huatuo_encyclopedia_qa(test)", "", r["answers"][0]
    if want("huatuo_kg"):
        kg = CORPUS / "FreedomIntelligence__huatuo_knowledge_graph_qa" / "train_datasets.jsonl"
        for i, r in capped(enumerate(_jsonl(kg))):
            yield f"huatuo_kg:{i}", "huatuo_knowledge_graph_qa", r["questions"][0], r["answers"][0]
    if want("huatuo_lite"):
        for r in capped(_jsonl(CORPUS / "FreedomIntelligence__Huatuo26M-Lite" / "format_data.jsonl")):
            yield f"huatuo_lite:{r['id']}", "Huatuo26M-Lite", r["question"], r["answer"]
    ft = CORPUS / "shibing624__medical" / "finetune" / "train_zh_0.json"
    if want("shibing") and ft.exists():
        for i, r in capped(enumerate(_jsonl(ft))):
            yield f"shibing_med:{i}", "shibing624/medical", (r.get("instruction", "") + r.get("input", ""))[:200], r.get("output", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default=str(ROOT / ".cache" / "index" / "medical.fts5"))
    b.add_argument("--limit-per-source", type=int)
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--only", help="逗号分隔：huatuo_enc,huatuo_kg,huatuo_lite,shibing")
    b.add_argument("--append", action="store_true", help="追加到已有索引")
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--index", default=str(ROOT / ".cache" / "index" / "medical.fts5"))
    args = ap.parse_args()
    if args.cmd == "build":
        if Path(args.out).exists() and not args.append:
            Path(args.out).unlink()
        idx = FTS5Index(args.out)
        only = set(args.only.split(",")) if args.only else None
        print(json.dumps(idx.build(iter_open_corpora(args.limit_per_source, include_test_answers=not args.append, only=only),
                                   workers=args.workers), ensure_ascii=False))
    else:
        idx = FTS5Index(args.index)
        t = time.perf_counter()
        hits = idx.search(args.query, k=5)
        for d in idx.get([h[0] for h in hits]):
            print(d.source, "|", d.title[:40], "|", d.text[:80].replace("\n", " "))
        print(f"{len(idx):,} docs, {(time.perf_counter() - t) * 1000:.1f}ms")


if __name__ == "__main__":
    main()
