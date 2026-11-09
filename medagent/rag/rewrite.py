"""查询改写：
  1. 去口语噪声（“医生你好，我想问一下……谢谢”）
  2. 口语 → 标准术语扩展（拉肚子→腹泻、美林→布洛芬）——追加而非替换，保留原词以免丢召回
  3. 复合问题拆分（“先查症状再查布洛芬副作用” → 两个子查询，分别召回后合并）
  4. 可选：轻量 LLM 改写 / HyDE（llm_rewriter 回调；默认关闭，避免每次检索都多一次模型调用）
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

_FILLER = re.compile(r"^(医生|大夫)?(你好|您好)[，,。！!]?|请问(一下)?|我想(问|咨询)(一下)?|麻烦(问一下)?|谢谢(医生|大夫)?[！!。]?$|"
                     r"[啊呀呢吧嘛哈]+(?=[，,。？?！!]|$)")
_SPLIT = re.compile(r"[，,；;。]?(?:先|首先)(.+?)(?:再|然后|之后|接着)(.+)$")
_AND_SPLIT = re.compile(r"(.+?)(?:和|以及|还有|另外)(.+?(?:药|饮食|指南|副作用|注意|怎么办).*)$")


@dataclass
class RewrittenQuery:
    original: str
    cleaned: str
    expanded: str
    sub_queries: list[str] = field(default_factory=list)

    def all_queries(self) -> list[str]:
        qs = [self.expanded] + [q for q in self.sub_queries if q != self.expanded]
        return list(dict.fromkeys(qs))


class QueryRewriter:
    def __init__(self, llm_rewriter: Callable[[str], str] | None = None, decompose: bool = True) -> None:
        from ..knowledge import load_kb
        self.kb = load_kb()
        self.llm_rewriter = llm_rewriter
        self.decompose = decompose

    def expand_terms(self, q: str) -> str:
        add = []
        for table in ("symptom", "drug", "disease"):
            for alias in sorted(self.kb.synonyms.get(table, {}), key=len, reverse=True):
                std = self.kb.synonyms[table][alias]
                if alias in q and std not in q and std not in add:
                    add.append(std)
        for cat, members in self.kb.synonyms.get("category", {}).items():
            if cat in q:
                add.extend(m for m in members if m not in q and m not in add)
        return q + (" " + " ".join(add) if add else "")

    def rewrite(self, query: str) -> RewrittenQuery:
        cleaned = _FILLER.sub("", query).strip() or query.strip()
        expanded = self.expand_terms(cleaned)
        subs: list[str] = []
        if self.decompose:
            m = _SPLIT.search(cleaned) or _AND_SPLIT.search(cleaned)
            if m:
                head = cleaned[: m.start()].strip("，,。 ")
                for part in m.groups():
                    part = part.strip("，,。 ")
                    if part:
                        subs.append(self.expand_terms(f"{head}{part}" if head and len(part) < 6 else part))
        if self.llm_rewriter is not None:
            try:
                extra = self.llm_rewriter(cleaned).strip()
                if extra:
                    subs.append(extra)
            except Exception:  # noqa: BLE001 —— 改写失败不影响检索
                pass
        return RewrittenQuery(query, cleaned, expanded, subs)
