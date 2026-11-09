"""安全优先的路由策略。

两个路由器：
  - SafetyFirstRouter：两档（local = 微调 MedicalGPT-7B，strong = 云端医疗大模型）；
  - TieredRouter：三档，在两档基础上加 small（Qwen3.5-2B），只接通过门控的概念/科普题。
    2026-11-05 版本只有两档、小模型只做控制面；2026-11-09 实测 Qwen3.5-2B 在门控题上 10/10 通过后加了 small 档
    （evaluation/results/small_tier_eval.json）。配置里有 small 档模型时 cli/gateway 自动用 TieredRouter。

决策顺序（前面的规则一旦命中就不再看后面）：
  1. 急症 / 危机   → 确定性急救提示立即下发（P0 快路径），解释部分走 strong
  2. 高风险意图     → strong（拒绝开方/诊断的话术质量与边界把握更稳）
  3. RouteLLM MF 打分 P(本地模型不达标)：≥ 阈值 → strong，否则 local
     （MF 只在 local / strong 之间二选一；目前用 80 条人工标签冷启动训练）
  4. 预算闸门：会话成本超上限 → 只在 1/2 时用 strong
  5. 仅 TieredRouter：判为 local 且通过 is_simple_question 门控（规则，不是学出来的）→ small
  6. 可用性：目标层熔断打开时由 ModelPool 自动跨层兜底（small → local → strong，strong ⇄ local）
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..safety import SafetyAssessment, assess
from .features import QueryFeaturizer, split_history
from .mf_router import MFRouter


@dataclass
class RouteDecision:
    tier: str                 # strong | local | small
    reason: str
    score: float | None = None
    threshold: float | None = None
    fastpath: bool = False    # 是否先下发确定性急救提示


class SafetyFirstRouter:
    def __init__(self, mf: MFRouter | None = None, featurizer: QueryFeaturizer | None = None,
                 threshold: float = 0.5, session_budget: float | None = None) -> None:
        self.mf = mf
        self.featurizer = featurizer or QueryFeaturizer()
        self.threshold = threshold
        self.session_budget = session_budget

    def score(self, query: str, history: str = "") -> float:
        if self.mf is None:
            return 0.0
        q, h = (query, history) if history else split_history(query)
        X = self.featurizer.transform([q], [h])
        return float(self.mf.predict_strong_win(X)[0])

    def route(self, query: str, history: str = "", assessment: SafetyAssessment | None = None,
              session_cost: float = 0.0) -> RouteDecision:
        a = assessment or assess(query)
        if a.level == "emergency" or a.crisis:
            return RouteDecision("strong", f"emergency:{','.join(a.emergency_hits) or 'crisis'}", fastpath=True)
        if a.high_risk:
            return RouteDecision("strong", f"high_risk:{','.join(a.high_risk_hits)}")
        s = self.score(query, history)
        over_budget = self.session_budget is not None and session_cost >= self.session_budget
        if s >= self.threshold and not over_budget:
            return RouteDecision("strong", "router:weak_likely_fails", s, self.threshold)
        return RouteDecision("local", "router:weak_sufficient" if not over_budget else "budget_cap", s, self.threshold)

    def batch_scores(self, queries: list[str]) -> np.ndarray:
        return np.array([self.score(q) for q in queries])


_SIMPLE = re.compile(r"^(请问)?(什么是|.{1,10}是什么|.{1,10}指的是什么|为什么要?|.{1,10}有(什么|哪些)(危害|影响|好处|作用|区别)|"
                     r".{1,10}的(定义|意思|原理)|怎么(计算|算)|.{1,8}和.{1,8}有什么区别)")
_PERSONAL = re.compile(r"我|孩子|宝宝|老人|我妈|我爸|老公|老婆|媳妇|家人|病人|患者|他|她")
_DOSING = re.compile(r"吃多少|几片|剂量|用量|怎么吃|能不能吃|可以吃|一起吃|停药|换药")


def is_simple_question(query: str, history: str = "") -> bool:
    """小模型档的门控：只放行“不涉及具体个人、不涉及用药决策、单一意图、无历史”的概念/科普题。"""
    from ..knowledge import load_kb
    from ..knowledge.ext import load_ext
    from ..verify.detectors import multi_intent
    q = query.strip()
    if not (_SIMPLE.search(q) and not _PERSONAL.search(q) and not _DOSING.search(q)
            and not multi_intent(q) and not history and len(q) <= 40):
        return False
    # 涉及具体药品的问题必须查工具 / 说明书，不交给不带工具的小模型凭记忆回答
    ext = load_ext()
    return not load_kb().link_entities(q)["drug"] and not (ext and ext.link(q)["drug"])


class TieredRouter(SafetyFirstRouter):
    """三档：small（简单问答）/ local（微调 MedicalGPT）/ strong（云端医疗大模型）。
    相当于 RouteLLM 的两个二分类路由器级联：先用 MF 打分决定要不要上 strong，
    不上 strong 的再用门控决定能不能降到 small。安全规则永远优先。"""

    def __init__(self, *args, small_max_score: float | None = None, **kw) -> None:
        super().__init__(*args, **kw)
        self.small_max_score = small_max_score   # MF 分数低于它才允许走 small（None = 只看门控）

    def route(self, query: str, history: str = "", assessment: SafetyAssessment | None = None,
              session_cost: float = 0.0) -> RouteDecision:
        d = super().route(query, history, assessment, session_cost)
        if d.tier != "local":
            return d
        if is_simple_question(query, history) and (self.small_max_score is None or (d.score or 0) <= self.small_max_score):
            return RouteDecision("small", "router:simple_question", d.score, d.threshold)
        return d


class RouteLLMControllerAdapter:  # pragma: no cover - 需要 pip install routellm 与 OPENAI_API_KEY
    """直接用官方 RouteLLM（lm-sys/RouteLLM）的预训练路由器。注意：官方 MF 路由器推理时调用 OpenAI embedding，
    且权重来自通用对话偏好，不针对中文医疗；仅作对照基线。"""

    def __init__(self, router: str = "mf", strong_model: str = "gpt-4-1106-preview",
                 weak_model: str = "mixtral-8x7b-instruct-v0.1", threshold: float = 0.11593) -> None:
        from routellm.controller import Controller
        self.ctrl = Controller(routers=[router], strong_model=strong_model, weak_model=weak_model)
        self.router, self.threshold = router, threshold

    def strong_win_rate(self, prompt: str) -> float:
        return float(self.ctrl.routers[self.router].calculate_strong_win_rate(prompt))

    def route(self, prompt: str) -> str:
        return "strong" if self.strong_win_rate(prompt) >= self.threshold else "local"


def describe(decision: RouteDecision) -> dict[str, Any]:
    return {"tier": decision.tier, "reason": decision.reason, "score": decision.score,
            "threshold": decision.threshold, "fastpath": decision.fastpath}
