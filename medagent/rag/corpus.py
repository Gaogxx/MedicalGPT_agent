"""语料加载与切块。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .text import split_sentences


@dataclass
class Doc:
    id: str
    text: str
    title: str = ""
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def chunk_text(text: str, max_chars: int = 320, overlap_sents: int = 1) -> list[str]:
    sents = split_sentences(text)
    chunks, cur = [], []
    for s in sents:
        if cur and sum(len(x) for x in cur) + len(s) > max_chars:
            chunks.append("".join(cur))
            cur = cur[-overlap_sents:] if overlap_sents else []
        cur.append(s)
    if cur:
        chunks.append("".join(cur))
    # 单句过长时硬切
    out = []
    for c in chunks:
        while len(c) > max_chars * 1.5:
            out.append(c[:max_chars])
            c = c[max_chars - 40:]
        out.append(c)
    return out


def docs_from_kb() -> list[Doc]:
    from ..knowledge import load_kb
    return [Doc(d["id"], d["text"], d["title"], d["source"]) for d in load_kb().documents()]


def docs_from_sharegpt_qa(path: str | Path, *, source: str, limit: int | None = None,
                          max_chars: int = 320) -> list[Doc]:
    """把 ShareGPT 问答的“答案”切块成文档；meta.qid 记录来源问题，用于构造检索评测 qrels。"""
    docs: list[Doc] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            row = json.loads(line)
            conv = row.get("conversations", [])
            q = next((c["value"] for c in conv if c.get("from") == "human"), "")
            a = next((c["value"] for c in conv if c.get("from") == "gpt"), "")
            if not a:
                continue
            # 注意：不能把问题放进 title —— 检索器会索引 title，评测时等于把查询本身塞进了语料（it1 的评测泄漏 bug）
            for j, ch in enumerate(chunk_text(a, max_chars=max_chars)):
                docs.append(Doc(f"{source}:{i}:{j}", ch, title="", source=source,
                                meta={"qid": i, "chunk": j, "question": q[:80]}))
    return docs


def docs_from_jsonl(path: str | Path, *, source: str, text_key: str = "text", title_key: str = "title",
                    max_chars: int = 320) -> list[Doc]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            for j, ch in enumerate(chunk_text(row[text_key], max_chars=max_chars)):
                docs.append(Doc(f"{source}:{row.get('id', i)}:{j}", ch, row.get(title_key, ""), source))
    return docs


def corpus_fingerprint(docs: Iterable[Doc]) -> str:
    h = hashlib.sha256()
    for d in docs:
        h.update(d.id.encode())
        h.update(hashlib.md5(d.text.encode()).digest())
    return h.hexdigest()[:16]
