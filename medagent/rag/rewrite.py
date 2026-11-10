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
_AND_SPLIT = re.compile(r"(.+?)(?:和|以及|还有|另外)(.*?(?:药|饮食|指南|副作用|注意|怎么办).*)$")
# “糖尿病有什么药和饮食建议”：后半句“饮食建议”没有主语，从前半句取主语补上 → “糖尿病饮食建议”
_SUBJECT = re.compile(r"^(.+?)(?:有什么|有哪些|吃什么|用什么|怎么|如何|能不能|该|要)")
# 2026-11-10 扩充：按分句拆、“A和B分别…”拆。只有每一段都像一个问题（带问句线索）才拆，避免把单一问题拆碎
_Q_CUE = re.compile(r"什么|怎么|如何|吗|哪些|能不能|能否|可不可以|要不要|是否|建议|注意|讲究|原因|副作用|禁忌|用法|用量|适应症|不良反应")
_CLAUSE = re.compile(r"[，,；;？?。！!]|另外|还有|此外|同时|以及")
_TOPIC = re.compile(r"药|饮食|吃饭|运动|副作用|不良反应|禁忌|原因|症状|用法|用量|适应症|指南|检查|治疗|手术|预防|传染|保存|一起吃")
_RECAP = re.compile(r"^(我)?(之前|刚才|上次|以前)(问过|咨询过|说过|提到过?)|^你(之前|刚才|上次|分别)?(给了|给过|说了|说过|建议过)")
_RESPECTIVE = re.compile(r"^(?P<a>[^，,。？?]{1,16}?)(?:和|与|跟|及)(?P<b>[^，,。？?]{1,16}?)分别(?P<rest>.+)$")


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

    def _entities(self, text: str) -> list[str]:
        ents = self.kb.link_entities(text)
        return list(dict.fromkeys(ents["disease"] + ents["drug"] + ents["symptom"]))

    def _respective(self, clause: str) -> list[str]:
        """“二甲双胍和头孢分别有哪些禁忌” → 二甲双胍有哪些禁忌 / 头孢有哪些禁忌；
        “冠心病的饮食和运动分别有什么建议” → 冠心病的饮食有什么建议 / 冠心病的运动有什么建议；
        “请分别介绍一下A的用法和B的不良反应” → 介绍一下A的用法 / B的不良反应。"""
        m = _RESPECTIVE.match(clause)
        if m:
            a, b, rest = m.group("a"), m.group("b"), m.group("rest")
            pre = ""
            if "的" in a:                       # 公共前缀：“冠心病的饮食” → 前缀“冠心病的”
                cut = a.rindex("的") + 1
                pre, a = a[:cut], a[cut:]
            if a and b:
                return [f"{pre}{a}{rest}", f"{pre}{b}{rest}"]
        if "分别" in clause:
            body = clause.replace("请分别", "").replace("分别", "")
            parts = [x for x in re.split(r"和|与|跟|及", body) if len(x) >= 2]
            if len(parts) == 2 and all(self._entities(x) or _Q_CUE.search(x) for x in parts):
                return parts
        return []

    def _clauses(self, text: str) -> list[str]:
        """按分句拆，只保留带问句线索的分句；不带线索的分句（“我最近乏力”）并入后一个分句作为背景。
        缺主语的分句（“平时饮食要注意什么”）补上全句识别到的疾病 / 药名。"""
        raw = [c.strip() for c in _CLAUSE.split(text) if c and c.strip()]
        out, ctx = [], ""
        for c in raw:
            if _Q_CUE.search(c) and not _RECAP.search(c):      # 历史回顾（“我之前问过……”）只作背景
                out.append(f"{ctx}，{c}" if ctx else c)
                ctx = ""
            else:
                ctx = f"{ctx}，{c}" if ctx else c
        if ctx and out:
            out[-1] = f"{out[-1]}，{ctx}"
        expanded = []
        for c in out:
            expanded += self._respective(c) or [c]
        if len(expanded) < 2:
            return []
        if not any(self._respective(c) for c in out):
            # 纯分句拆分要求各段谈的对象 / 话题不同（“冠心病用什么药” vs “饮食注意什么”），
            # 同一件事的追问（“这个药怎么保存，会过期吗”）不拆
            sigs = [frozenset(self._entities(c)) | frozenset(_TOPIC.findall(c)) for c in expanded]
            if not all(sigs) or len(set(sigs)) < len(sigs):
                return []
        ents = self._entities(text)
        subject = "、".join(ents)
        return [c if (i == 0 or self._entities(c) or not subject) else f"{subject}{c}" for i, c in enumerate(expanded)]

    def rewrite(self, query: str) -> RewrittenQuery:
        cleaned = _FILLER.sub("", query).strip() or query.strip()
        expanded = self.expand_terms(cleaned)
        subs: list[str] = []
        if self.decompose:
            m = _SPLIT.search(cleaned)
            and_split = m is None
            if m is None:
                m = _AND_SPLIT.search(cleaned)
                if m and not _Q_CUE.search(m.group(1)):          # “体检甘油三酯和胆固醇都高……”不是两个问题
                    m = None
            if m:
                head = cleaned[: m.start()].strip("，,。 ")
                if and_split and not head:
                    sm = _SUBJECT.search(m.group(1))
                    head = sm.group(1) if sm else ""
                for i, part in enumerate(m.groups()):
                    part = part.strip("，,。 ")
                    if not part:
                        continue
                    short = len(part) < (8 if and_split else 6)
                    use_head = head and short and not (and_split and i == 0)
                    subs.append(self.expand_terms(f"{head}{part}" if use_head else part))
            if len(subs) < 2:
                subs = [self.expand_terms(c) for c in self._clauses(cleaned)]
        if self.llm_rewriter is not None:
            try:
                extra = self.llm_rewriter(cleaned).strip()
                if extra:
                    subs.append(extra)
            except Exception:  # noqa: BLE001 —— 改写失败不影响检索
                pass
        return RewrittenQuery(query, cleaned, expanded, subs)
