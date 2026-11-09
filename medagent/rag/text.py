"""中文分词与文本工具。jieba 可用时用“词 + 字二元组”混合切分（词保证精度，二元组兜住未登录医学术语），
否则退化为纯字二元组。医学词典来自内置 KB（症状 / 药品 / 疾病 / 同义词），保证“对乙酰氨基酚”不被切碎。"""
from __future__ import annotations

import re
from functools import lru_cache

_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")
_NON_WORD = re.compile(r"[\s\W_]+", re.U)
STOPWORDS = set("的 了 吗 呢 吧 啊 呀 是 在 和 与 及 或 我 你 他 她 它 们 有 没 什么 怎么 怎么办 如何 请问 一下 可以 能 会 要 就 都 也 还 这 那 个 些 很 太 啥 么".split())


@lru_cache(maxsize=1)
def _jieba():
    try:
        import jieba  # type: ignore
        import logging
        jieba.setLogLevel(logging.WARNING)
        from ..knowledge import load_kb
        kb = load_kb()
        for w in list(kb.symptoms) + list(kb.drugs) + list(kb.guidelines):
            jieba.add_word(w, freq=50000)
        for table in ("symptom", "drug", "disease"):
            for alias, std in kb.synonyms.get(table, {}).items():
                jieba.add_word(alias, freq=30000)
                jieba.add_word(std, freq=50000)
        return jieba
    except ImportError:  # pragma: no cover
        return None


def char_bigrams(text: str) -> list[str]:
    s = _NON_WORD.sub("", text)
    return [s[i:i + 2] for i in range(len(s) - 1)] or ([s] if s else [])


def tokenize(text: str, mode: str = "mixed") -> list[str]:
    text = text.lower()
    if mode == "bigram":
        return char_bigrams(text)
    jb = _jieba()
    if jb is None:
        return char_bigrams(text)
    words = [w for w in jb.lcut(text) if w.strip() and not _NON_WORD.fullmatch(w) and w not in STOPWORDS]
    if mode == "word":
        return words
    return words + char_bigrams(text)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def ngram_set(text: str, n: int = 3) -> set[str]:
    s = _NON_WORD.sub("", text)
    return {s[i:i + n] for i in range(max(len(s) - n + 1, 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def estimate_tokens(text: str) -> int:
    """粗估 token：CJK 字≈1 token，其余按 4 字符 1 token。"""
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    return cjk + max(0, len(text) - cjk) // 4
