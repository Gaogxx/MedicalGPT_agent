"""上下文压缩。

医疗场景的原则：可以压缩“怎么说的”，不能压缩“病人是谁”。
  1. 临床关键事实槽位（年龄/性别/孕期/过敏史/在用药/慢病/已出现的危险信号）从全部历史里抽出来，常驻、永不丢弃；
  2. 最近 K 轮原文保留；
  3. 更早的轮次：有控制面小模型时做“保事实摘要”，否则抽取式（只留含实体/数字/建议的句子）；
  4. 本轮工具结果钉住不压缩（回答要以它为依据）。
基线里“长历史”类用例（8-上下文）本地 7B 会编造不存在的对话，结构化槽位 + 明确的“无历史”标记就是为它设计的。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .knowledge import load_kb
from .rag.text import estimate_tokens, split_sentences

_AGE = re.compile(r"(\d{1,3})\s*(岁|周岁)")
_SEX = re.compile(r"(男|女)(性|的|生)?[，,。\s]|我是(男|女)")
_PREG = re.compile(r"怀孕|孕\d+周|孕期|哺乳期|备孕")
_ALLERGY = re.compile(r"对(.{1,8}?)过敏|(.{1,8}?)过敏史")
_ON_MED = re.compile(r"(在吃|正在吃|在用|长期吃|一直吃|服用)(.{1,12}?)(药|片|胶囊)?[，,。\s]")
_CHRONIC = ("高血压", "糖尿病", "冠心病", "哮喘", "慢阻肺", "肾病", "肝病", "痛风", "高血脂", "甲亢", "甲减", "癫痫")
_KEEP_HINT = re.compile(r"建议|就医|禁用|慎用|注意|过敏|\d|120|急诊")


@dataclass
class PatientProfile:
    age: str | None = None
    sex: str | None = None
    pregnancy: bool = False
    allergies: list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)

    def update(self, text: str) -> None:
        from .safety import assess
        if m := _AGE.search(text):
            self.age = m.group(1)
        if m := _SEX.search(text):
            self.sex = m.group(1) or m.group(3)
        self.pregnancy = self.pregnancy or bool(_PREG.search(text))
        for m in _ALLERGY.finditer(text):
            a = (m.group(1) or m.group(2) or "").strip("我有")
            if a and a not in self.allergies:
                self.allergies.append(a)
        kb = load_kb()
        for d in kb.link_entities(text)["drug"]:
            if _ON_MED.search(text) and d not in self.medications:
                self.medications.append(d)
        for c in _CHRONIC:
            if c in text and re.search(rf"(有|患|得了|确诊).{{0,4}}{c}|{c}(病史|患者|多年)", text) and c not in self.conditions:
                self.conditions.append(c)
        for h in assess(text).emergency_hits:
            if h not in self.red_flags:
                self.red_flags.append(h)

    def render(self) -> str:
        items = [f"年龄:{self.age}" if self.age else "", f"性别:{self.sex}" if self.sex else "",
                 "孕期/哺乳:是" if self.pregnancy else "",
                 f"过敏:{'、'.join(self.allergies)}" if self.allergies else "",
                 f"在用药:{'、'.join(self.medications)}" if self.medications else "",
                 f"慢病:{'、'.join(self.conditions)}" if self.conditions else "",
                 f"既往危险信号:{'、'.join(self.red_flags)}" if self.red_flags else ""]
        s = "；".join(i for i in items if i)
        return f"【患者关键信息（不可忽略）】{s}" if s else ""


@dataclass
class CompressionStats:
    tokens_before: int
    tokens_after: int
    summarized_turns: int

    @property
    def ratio(self) -> float:
        return self.tokens_after / self.tokens_before if self.tokens_before else 1.0


class ContextCompressor:
    def __init__(self, budget_tokens: int = 3000, keep_recent: int = 4,
                 summarizer: Callable[[str], str] | None = None) -> None:
        self.budget = budget_tokens
        self.keep_recent = keep_recent
        self.summarizer = summarizer   # 控制面小模型（可选）

    @staticmethod
    def _tokens(msgs: list[dict[str, Any]]) -> int:
        return sum(estimate_tokens(str(m.get("content", ""))) for m in msgs)

    def _extractive(self, msgs: list[dict[str, Any]]) -> str:
        kb = load_kb()
        keep = []
        for m in msgs:
            for s in split_sentences(str(m.get("content", ""))):
                ents = kb.link_entities(s)
                if any(ents.values()) or _KEEP_HINT.search(s):
                    keep.append(("用户" if m["role"] == "user" else "助手") + "：" + s[:80])
        return "；".join(dict.fromkeys(keep))[:800]

    def compress(self, messages: list[dict[str, Any]], profile: PatientProfile | None = None
                 ) -> tuple[list[dict[str, Any]], CompressionStats]:
        before = self._tokens(messages)
        system = [m for m in messages if m["role"] == "system"]
        convo = [m for m in messages if m["role"] != "system"]
        profile = profile or PatientProfile()
        for m in convo:
            if m["role"] == "user":
                profile.update(str(m.get("content", "")))
        if before <= self.budget:
            out = system + ([{"role": "system", "content": profile.render()}] if profile.render() else []) + convo
            return out, CompressionStats(before, self._tokens(out), 0)
        # 切分点不能把 assistant(tool_calls) 与其后的 tool 结果拆开
        cut = max(0, len(convo) - self.keep_recent)
        while cut > 0 and convo[cut]["role"] == "tool":
            cut -= 1
        old, recent = convo[:cut], convo[cut:]
        summary = ""
        if old:
            text = "\n".join(f"{m['role']}: {m.get('content', '')}" for m in old)
            if self.summarizer is not None:
                try:
                    summary = self.summarizer(text)
                except Exception:  # noqa: BLE001 —— 小模型挂了退回抽取式
                    summary = self._extractive(old)
            else:
                summary = self._extractive(old)
        extra = []
        if profile.render():
            extra.append({"role": "system", "content": profile.render()})
        if summary:
            extra.append({"role": "system", "content": f"【更早对话摘要】{summary}"})
        out = system + extra + recent
        return out, CompressionStats(before, self._tokens(out), len(old))
