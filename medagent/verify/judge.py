"""类型化判别（typed judge）：输入程序状态 + 问题，输出预定义类型的值 + 置信度，而不是一段文字。

为什么是“类型化”：校验结果要直接驱动控制流（放行 / 重生成 / 升级强模型 / 转人工），
自由文本还得再解析一次，本身又是一个静默失败点。三种原语：Choice（多选一）、Score（有序等级）、Noul（是/否 + 单一概率）。
线上用的是 typed.TypedVerifier + 本地开源判别模型（discriminator.py）；本文件是离线评测用的级联实验（run_verifier_eval.py）。

级联：
  RuleJudge（确定性检测器，命中 block 即判失败，几乎零误报）
    → LocalClassifierJudge（字符 n-gram TF-IDF + LR，CPU 亚毫秒级；静默失败检测上这类轻量检测器
       往往优于让 LLM 读整段对话——见 docs/verifier.md 引用）
      → 置信度落在不确定带 [lo, hi] 时才升级到 LLMJudge（贵、慢，只处理少数难例）
注意：置信度不等于正确性。能用确定性检查（工具结果、证据比对）验证的，一律以确定性检查为准。
"""
from __future__ import annotations

import json

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .detectors import detect


@dataclass
class Decision:
    label: str
    confidence: float
    backend: str
    rationale: str = ""
    escalated: bool = False


class TypedJudge(Protocol):
    def noul(self, statement: str, state: dict[str, Any]) -> Decision: ...
    def choice(self, question: str, state: dict[str, Any], options: list[str]) -> Decision: ...


def state_text(state: dict[str, Any]) -> str:
    codes = " ".join(f"[F:{c}]" for c in state.get("finding_codes", []))
    obs = json.dumps(state.get("observations") or [], ensure_ascii=False)[:400]
    return f"Q:{state.get('query', '')} H:{state.get('history', '')[:200]} A:{state.get('answer', '')} O:{obs} {codes}"


class RuleJudge:
    name = "rules"

    def noul(self, statement: str, state: dict[str, Any]) -> Decision:
        findings = state.get("findings")
        if findings is None:
            findings = detect(state.get("answer", ""), query=state.get("query", ""), assessment=state.get("assessment"),
                              observations=state.get("observations"), history=state.get("history", ""),
                              finish_reason=state.get("finish_reason"))
        block = [x for x in findings if x.severity == "block"]
        if block:
            return Decision("yes", 0.97, self.name, ",".join(x.code for x in block))
        return Decision("no", 0.5, self.name, ",".join(x.code for x in findings))  # 0.5：规则没命中≠确认没问题

    def choice(self, question: str, state: dict[str, Any], options: list[str]) -> Decision:
        from .attribution import attribute
        a = attribute(state.get("events", []))
        label = a.kind if a.kind in options else options[-1]
        return Decision(label, 0.9 if a.kind in options else 0.3, self.name, a.reason)


class LocalClassifierJudge:
    """轻量判别模型：char 1-3gram TF-IDF + 逻辑回归（class_weight=balanced）。训练数据见
    evaluation/datasets/failure_cases.jsonl（真实失败样本 + 标注为 synthetic 的故障注入样本）。"""

    name = "local_clf"

    def __init__(self, model: Any = None) -> None:
        self.model = model

    @classmethod
    def train(cls, rows: list[dict[str, Any]], C: float = 4.0) -> "LocalClassifierJudge":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        x = [state_text(r) for r in rows]
        y = [1 if r["label"] != "ok" else 0 for r in rows]
        pipe = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=1, sublinear_tf=True),
                             LogisticRegression(C=C, class_weight="balanced", max_iter=2000))
        pipe.fit(x, y)
        return cls(pipe)

    def prob(self, state: dict[str, Any]) -> float:
        return float(self.model.predict_proba([state_text(state)])[0][1])

    def noul(self, statement: str, state: dict[str, Any]) -> Decision:
        p = self.prob(state)
        return Decision("yes" if p >= 0.5 else "no", p if p >= 0.5 else 1 - p, self.name, f"p_fail={p:.3f}")

    def choice(self, question: str, state: dict[str, Any], options: list[str]) -> Decision:  # pragma: no cover
        raise NotImplementedError("LocalClassifierJudge 只做二分类（是否静默失败）；归因用 RuleJudge")


class LLMJudge:
    """小 LLM 判别（如本地 Qwen3-1.7B / 云端便宜模型），强制 JSON 输出。provider 需实现 complete_json(prompt)->dict。"""

    name = "llm"

    def __init__(self, provider: Any) -> None:
        self.provider = provider

    def noul(self, statement: str, state: dict[str, Any]) -> Decision:
        prompt = (f"判断下述陈述是否成立，只输出 JSON：{{\"value\": true/false, \"confidence\": 0~1}}。\n"
                  f"陈述：{statement}\n状态：{state_text(state)}")
        out = self.provider.complete_json(prompt)
        v = out.get("value")
        return Decision("yes" if v in (True, "true", "yes") else "no", float(out.get("confidence", 0.5)), self.name)

    def choice(self, question: str, state: dict[str, Any], options: list[str]) -> Decision:
        prompt = (f"{question}\n只能从 {options} 中选一个，输出 JSON：{{\"value\": 选项, \"confidence\": 0~1}}。\n"
                  f"状态：{state_text(state)}")
        out = self.provider.complete_json(prompt)
        v = str(out.get("value"))
        return Decision(v if v in options else options[-1], float(out.get("confidence", 0.5)), self.name)


@dataclass
class CascadeJudge:
    rules: RuleJudge = field(default_factory=RuleJudge)
    classifier: LocalClassifierJudge | None = None
    escalate_to: Any = None              # LLMJudge
    uncertain_band: tuple[float, float] = (0.35, 0.65)
    stats: dict[str, int] = field(default_factory=lambda: {"rules": 0, "local_clf": 0, "escalated": 0})

    def is_silent_failure(self, state: dict[str, Any]) -> Decision:
        d = self.rules.noul("answer_is_failure", state)
        if d.label == "yes":
            self.stats["rules"] += 1
            return d
        if self.classifier is None:
            return d
        p = self.classifier.prob(state)
        lo, hi = self.uncertain_band
        if lo < p < hi and self.escalate_to is not None and getattr(self.escalate_to, "enabled", True):
            try:
                e = self.escalate_to.noul("answer_is_failure", state)
                self.stats["escalated"] += 1
                e.escalated = True
                return e
            except Exception:  # noqa: BLE001 —— 判别器自己挂了不能拖垮主流程
                pass
        self.stats["local_clf"] += 1
        return Decision("yes" if p >= 0.5 else "no", p if p >= 0.5 else 1 - p, "local_clf", f"p_fail={p:.3f}")


def has_cjk(text: str) -> bool:
    return bool(re.search(r"[一-鿿]", text))
