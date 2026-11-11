"""类型化判别：用“程序状态 + 一个问题 → 预先定义类型的值 + 置信度”做结果校验与出错归因。

判别后端是本地部署的开源判别模型（medagent/verify/discriminator.py：微调后的 Laya 多语言版，laya-serve 部署，不调外部 API）。
三个判定（DecisionSpec）：
  1. SILENT_FAILURE   是/否     “这个回答是否属于静默失败？”   → 校验：放行 / 升级重答 / 进人工复核
  2. ATTRIBUTION      多选一    “这次失败的根因属于哪一类？”   → 归因（判别模型不支持时只用规则）
  3. ANSWER_QUALITY   有序等级  “回答质量处于哪个等级？”       → 监控与抽检，不参与阻断
  （判别模型另有 failure_type：判为答砸时给出失败类别，写进校验结论）
在流水线里的位置（TypedVerifier）：
  规则检测器命中 block → 直接判失败（不问判别模型，确定性优先）
  规则没命中 → 问判别模型 SILENT_FAILURE：p ≥ block_at 视为静默失败（升级强模型重答）；review_at ≤ p < block_at 放行但记录复核
  归因：规则归因明确（异常类型 / HTTP 状态码 / 工具错误前缀）就用规则；规则说不清、且判别模型支持归因时才问它
判别模型调用失败（没部署 / 超时 / 报错）一律降级为“只用规则”，判别器自己挂了不能拖垮主流程。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .attribution import Attribution, attribute
from .detectors import Finding

ERROR_KINDS = ["network", "timeout", "rate_limit", "auth", "param", "model", "silent", "external", "policy"]
QUALITY_LEVELS = ["unsafe", "poor", "acceptable", "good"]


@dataclass(frozen=True)
class DecisionSpec:
    name: str
    primitive: str                 # noul（是/否）| choice（多选一）| score（有序等级）
    question: str
    options: tuple[str, ...] = ()


SILENT_FAILURE = DecisionSpec(
    "silent_failure", "noul",
    "这个医疗回答是否属于静默失败：内容为空或截断、编造了用户没给的信息或证据里没有的剂量、漏答了子问题、"
    "工具查不到却给出具体结论、或该给的急救/就医建议没给？")
ATTRIBUTION = DecisionSpec(
    "failure_attribution", "choice", "这次失败的根本原因属于哪一类？", tuple(ERROR_KINDS))
ANSWER_QUALITY = DecisionSpec(
    "answer_quality", "score", "这个回答在准确、安全、有依据三方面的综合等级？", tuple(QUALITY_LEVELS))


@dataclass
class TypedDecision:
    spec: str
    value: Any
    confidence: float
    backend: str = "discriminator"
    raw: dict[str, Any] = field(default_factory=dict)


def _evidence(o: Any, limit: int = 240) -> str:
    """工具结果压成一行：判别模型要能比对“回答里的剂量 / 禁忌是不是证据里的”，只给字段名不够。"""
    if not isinstance(o, dict):
        return str(o)[:limit]
    keep = {k: v for k, v in o.items() if k not in ("source", "provenance", "doc_id", "score", "rank")}
    return json.dumps(keep, ensure_ascii=False)[:limit]


def answer_state(query: str, answer: str, observations: list[Any] | None, findings: list[Finding] | list[dict],
                 assessment: Any = None) -> dict[str, Any]:
    """校验用的程序状态：结构化字段，不是一段话。训练数据和线上用同一个函数生成（格式一致）。"""
    obs = [{"found": o.get("found") if isinstance(o, dict) else None, "evidence": _evidence(o)}
           for o in (observations or [])[:4]]
    return {"query": query[:300], "answer": answer[:1500], "tool_results": obs,
            "rule_findings": [f.code if isinstance(f, Finding) else f.get("code") for f in findings],
            "safety": {"emergency": getattr(assessment, "level", "normal") == "emergency",
                       "high_risk": bool(getattr(assessment, "high_risk", False))}}


def trace_state(events: list[dict[str, Any]], rule: Attribution) -> dict[str, Any]:
    """归因用的程序状态：调用轨迹（阶段 / 来源 / 成败 / 错误类别 / HTTP 码 / 重试次数）+ 规则归因结论。"""
    ev = []
    for e in events[-20:]:
        err = e.get("error") or {}
        ev.append({"stage": e.get("stage"), "source": e.get("source"), "ok": e.get("ok"),
                   "error_kind": err.get("kind"), "status": err.get("status"), "message": str(err.get("message", ""))[:120],
                   "retries": e.get("retries", 0),
                   "findings": [f.get("code") for f in e.get("findings", []) if f.get("severity") == "block"]})
    return {"events": ev, "rule_attribution": {"kind": rule.kind, "responsible": rule.responsible, "component": rule.component}}


@dataclass
class Verdict:
    blocked: bool
    reason: str
    confidence: float
    source: str                        # rules | discriminator | rules(discriminator_unavailable)
    needs_review: bool = False
    quality: TypedDecision | None = None


def _supports(client: Any, spec: DecisionSpec) -> bool:
    return client is not None and client.enabled and spec.name in getattr(client, "supports", ())


@dataclass
class TypedVerifier:
    client: Any = None                # DiscriminatorClient（或任何实现 enabled / supports / decide 的对象）
    block_at: float = 0.8             # P(静默失败) ≥ block_at 才自动升级重答；部署后用 run_verifier_eval.py 在判别器验证集上校准
    review_at: float = 0.5            # review_at ~ block_at 放行，但进人工复核队列
    attribution_min: float = 0.7
    score_quality: bool = False
    stats: dict[str, int] = field(default_factory=lambda: {"rules_block": 0, "disc_block": 0, "disc_review": 0,
                                                           "disc_pass": 0, "disc_error": 0, "attr_disc": 0, "attr_rules": 0})

    def verify(self, query: str, answer: str, findings: list[Finding], observations: list[Any] | None = None,
               assessment: Any = None) -> Verdict:
        block = [f for f in findings if f.severity == "block"]
        if block:
            self.stats["rules_block"] += 1
            return Verdict(True, ",".join(f.code for f in block), 0.97, "rules")
        if not _supports(self.client, SILENT_FAILURE):
            return Verdict(False, "rules_pass", 0.5, "rules")
        state = answer_state(query, answer, observations, findings, assessment)
        try:
            d = self.client.decide(SILENT_FAILURE, state)
            q = self.client.decide(ANSWER_QUALITY, state) if self.score_quality and _supports(self.client, ANSWER_QUALITY) else None
        except Exception:  # noqa: BLE001
            self.stats["disc_error"] += 1
            return Verdict(False, "rules_pass", 0.5, "rules(discriminator_unavailable)")
        p = d.confidence if d.value else 1 - d.confidence     # 统一成 P(是静默失败)
        if p >= self.block_at:
            self.stats["disc_block"] += 1
            return Verdict(True, "discriminator:silent_failure" + self._failure_type(state), p, "discriminator", quality=q)
        if p >= self.review_at:
            self.stats["disc_review"] += 1
            return Verdict(False, "discriminator:review", p, "discriminator", needs_review=True, quality=q)
        self.stats["disc_pass"] += 1
        return Verdict(False, "discriminator:pass", p, "discriminator", quality=q)

    def _failure_type(self, state: dict[str, Any]) -> str:
        """判别模型给出的失败类别（同一次请求里已经算好，不额外调用）；拿不到就不写。"""
        if "failure_type" not in getattr(self.client, "supports", ()):
            return ""
        try:
            from .discriminator import FAILURE_TYPE
            d = self.client.decide(FAILURE_TYPE, state)
            return f":{d.value}" if d.value != "ok" else ""
        except Exception:  # noqa: BLE001
            return ""

    def attribute(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        rule = attribute(events)
        ambiguous = rule.kind in ("unknown",) or (
            rule.kind == "silent" and any(e.get("stage") == "tool" and e.get("ok") for e in events)
            and any(e.get("stage") == "llm" for e in events))
        out = {**rule.__dict__, "source": "rules", "ambiguous": ambiguous}
        if not ambiguous or not _supports(self.client, ATTRIBUTION):
            self.stats["attr_rules"] += 1
            return out
        try:
            d = self.client.decide(ATTRIBUTION, trace_state(events, rule))
        except Exception:  # noqa: BLE001
            self.stats["disc_error"] += 1
            return out
        if d.confidence >= self.attribution_min:
            self.stats["attr_disc"] += 1
            responsible = "upstream" if d.value in ("network", "timeout", "rate_limit", "auth", "external") else \
                "model" if d.value in ("model", "silent", "param") else "user"
            return {**out, "kind": d.value, "responsible": responsible, "source": d.backend, "confidence": d.confidence}
        return {**out, "low_confidence": {"value": d.value, "confidence": d.confidence}}


def default_verifier() -> TypedVerifier:
    """网关 / 编排器默认用的校验器：配置了 MEDAGENT_VERIFIER_URL 就接本地判别模型，否则只用规则。
    阈值优先取部署后校准出的 medagent/verify/artifacts/discriminator.json。"""
    from .discriminator import DiscriminatorClient, load_calibration
    cal = load_calibration()
    return TypedVerifier(DiscriminatorClient(), block_at=cal.get("block_at", 0.8), review_at=cal.get("review_at", 0.5))
