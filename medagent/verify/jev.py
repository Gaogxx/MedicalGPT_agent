"""Jev 适配层：用“类型化判别”做结果校验与出错归因。

Jev（TypeSafe AI 的 System-One 判别模型）公开资料只说明了：输入 = 程序状态 + 一个问题；输出 = 预先定义类型的值 + 校准置信度；
三种原语 Choice（多选一）/ Score（有序等级）/ Noul（是否 + 单一概率）；不输出推理文字。没有公开 HTTP 接口文档，
所以本模块自己定义一份**请求/响应契约**（见 docs/jev_adapter.md），拿到官方文档后只需改 build_request / parse_response。

本项目向 Jev 提三个问题（DecisionSpec）：
  1. SILENT_FAILURE   Noul    “这个回答是否属于静默失败？”         → 校验：决定放行 / 升级重答 / 进人工复核
  2. ATTRIBUTION      Choice  “这次失败的根因属于哪一类？”         → 归因：network/timeout/.../model/silent/param…
  3. ANSWER_QUALITY   Score   “回答质量处于哪个等级？”             → 监控与抽检，不参与阻断
在流水线里的位置（JevVerifier）：
  规则检测器命中 block → 直接判失败（不问 Jev，确定性优先）
  规则没命中 → 问 Jev SILENT_FAILURE：p ≥ block_at 视为静默失败（升级强模型重答）；review_at ≤ p < block_at 放行但记录复核
  归因：规则归因明确（异常类型 / HTTP 状态码 / 工具错误前缀）就用规则；规则给出 unknown，或“工具成功但回答被判失败”
        这种模型 vs 上游说不清的情况，才问 Jev ATTRIBUTION；置信度低于 attribution_min 时保留规则结论并标记 ambiguous
Jev 调用失败（网络 / 超时 / 未配置）一律降级为“只用规则”，判别器自己挂了不能拖垮主流程。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from .attribution import Attribution, attribute
from .detectors import Finding

ERROR_KINDS = ["network", "timeout", "rate_limit", "auth", "param", "model", "silent", "external", "policy"]
QUALITY_LEVELS = ["unsafe", "poor", "acceptable", "good"]


@dataclass(frozen=True)
class DecisionSpec:
    name: str
    primitive: str                 # noul | choice | score
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
    backend: str = "jev"
    raw: dict[str, Any] = field(default_factory=dict)


def answer_state(query: str, answer: str, observations: list[Any] | None, findings: list[Finding] | list[dict],
                 assessment: Any = None) -> dict[str, Any]:
    """校验用的程序状态：结构化 JSON，不是一段话——判别器读字段，不读散文。"""
    obs = []
    for o in observations or []:
        if isinstance(o, dict):
            obs.append({"found": o.get("found"), "keys": sorted(o)[:12],
                        "doses": [v for k, v in o.items() if k in ("dosage",)][:1]})
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


class JevClient:
    """HTTP 适配器。默认契约（本项目自定，待官方文档对齐）：
        POST {JEV_BASE_URL}/v1/decide
        {"primitive": "noul"|"choice"|"score", "question": str, "state": {...}, "options"|"levels": [...], "decision": name}
        → {"value": bool | str, "confidence": float, "model": str}
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: float = 1.5, client: Any = None,
                 build_request: Callable[[DecisionSpec, dict[str, Any]], dict[str, Any]] | None = None,
                 parse_response: Callable[[DecisionSpec, dict[str, Any]], tuple[Any, float]] | None = None) -> None:
        import httpx
        self.base_url = (base_url or os.environ.get("JEV_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("JEV_API_KEY", "")
        self.http = client or httpx.Client(timeout=timeout)
        self.build_request = build_request or self._build
        self.parse_response = parse_response or self._parse
        self.calls = 0

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    @staticmethod
    def _build(spec: DecisionSpec, state: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {"primitive": spec.primitive, "decision": spec.name, "question": spec.question, "state": state}
        if spec.primitive == "choice":
            body["options"] = list(spec.options)
        if spec.primitive == "score":
            body["levels"] = list(spec.options)
        return body

    @staticmethod
    def _parse(spec: DecisionSpec, data: dict[str, Any]) -> tuple[Any, float]:
        v, c = data.get("value"), float(data.get("confidence", 0.0))
        if spec.primitive == "noul":
            v = v in (True, "true", "yes", 1, "1")
        elif v not in spec.options:
            raise ValueError(f"jev returned {v!r} not in {spec.options}")   # 类型不对就当失败，不猜
        return v, max(0.0, min(1.0, c))

    def decide(self, spec: DecisionSpec, state: dict[str, Any]) -> TypedDecision:
        self.calls += 1
        r = self.http.post(f"{self.base_url}/v1/decide", json=self.build_request(spec, state),
                           headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})
        r.raise_for_status()
        data = r.json()
        v, c = self.parse_response(spec, data)
        return TypedDecision(spec.name, v, c, data.get("model", "jev"), data)


@dataclass
class Verdict:
    blocked: bool
    reason: str
    confidence: float
    source: str                        # rules | jev | rules(jev_unavailable)
    needs_review: bool = False
    quality: TypedDecision | None = None


@dataclass
class JevVerifier:
    client: JevClient | None = None
    block_at: float = 0.8             # Jev 判“是静默失败”的置信度 ≥ 0.8 才自动升级重答
    review_at: float = 0.5            # 0.5~0.8 放行，但进人工复核队列
    attribution_min: float = 0.7
    score_quality: bool = False
    stats: dict[str, int] = field(default_factory=lambda: {"rules_block": 0, "jev_block": 0, "jev_review": 0,
                                                           "jev_pass": 0, "jev_error": 0, "attr_jev": 0, "attr_rules": 0})

    def verify(self, query: str, answer: str, findings: list[Finding], observations: list[Any] | None = None,
               assessment: Any = None) -> Verdict:
        block = [f for f in findings if f.severity == "block"]
        if block:
            self.stats["rules_block"] += 1
            return Verdict(True, ",".join(f.code for f in block), 0.97, "rules")
        if self.client is None or not self.client.enabled:
            return Verdict(False, "rules_pass", 0.5, "rules")
        state = answer_state(query, answer, observations, findings, assessment)
        try:
            d = self.client.decide(SILENT_FAILURE, state)
            q = self.client.decide(ANSWER_QUALITY, state) if self.score_quality else None
        except Exception:  # noqa: BLE001
            self.stats["jev_error"] += 1
            return Verdict(False, "rules_pass", 0.5, "rules(jev_unavailable)")
        p = d.confidence if d.value else 1 - d.confidence     # 统一成 P(是静默失败)
        if p >= self.block_at:
            self.stats["jev_block"] += 1
            return Verdict(True, "jev:silent_failure", p, "jev", quality=q)
        if p >= self.review_at:
            self.stats["jev_review"] += 1
            return Verdict(False, "jev:review", p, "jev", needs_review=True, quality=q)
        self.stats["jev_pass"] += 1
        return Verdict(False, "jev:pass", p, "jev", quality=q)

    def attribute(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        rule = attribute(events)
        ambiguous = rule.kind in ("unknown",) or (
            rule.kind == "silent" and any(e.get("stage") == "tool" and e.get("ok") for e in events)
            and any(e.get("stage") == "llm" for e in events))
        out = {**rule.__dict__, "source": "rules", "ambiguous": ambiguous}
        if not ambiguous or self.client is None or not self.client.enabled:
            self.stats["attr_rules"] += 1
            return out
        try:
            d = self.client.decide(ATTRIBUTION, trace_state(events, rule))
        except Exception:  # noqa: BLE001
            self.stats["jev_error"] += 1
            return out
        if d.confidence >= self.attribution_min:
            self.stats["attr_jev"] += 1
            responsible = "upstream" if d.value in ("network", "timeout", "rate_limit", "auth", "external") else \
                "model" if d.value in ("model", "silent", "param") else "user"
            return {**out, "kind": d.value, "responsible": responsible, "source": "jev", "confidence": d.confidence}
        return {**out, "jev_low_confidence": {"value": d.value, "confidence": d.confidence}}


def fake_jev_app():
    """契约测试用的假 Jev 服务（Starlette）：按同一契约返回类型化结果，判定逻辑是简单启发式——只用来测管线，不代表 Jev 的能力。"""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def decide(request: Request) -> JSONResponse:
        b = await request.json()
        st = b["state"]
        if b["primitive"] == "noul":
            ans = st.get("answer", "")
            bad = (not ans.strip()) or "据我所知" in ans or ("一次" in ans and not st.get("tool_results"))
            return JSONResponse({"value": bad, "confidence": 0.9 if bad else 0.85, "model": "fake-jev"})
        if b["primitive"] == "choice":
            kinds = [e["error_kind"] for e in st.get("events", []) if e.get("error_kind")]
            tool_ok = any(e["stage"] == "tool" and e["ok"] for e in st.get("events", []))
            v = kinds[0] if kinds and kinds[0] in b["options"] else ("model" if tool_ok else "silent")
            return JSONResponse({"value": v, "confidence": 0.8, "model": "fake-jev"})
        return JSONResponse({"value": "acceptable", "confidence": 0.6, "model": "fake-jev"})

    return Starlette(routes=[Route("/v1/decide", decide, methods=["POST"])])


def dumps_contract() -> str:
    return json.dumps({"silent_failure": JevClient._build(SILENT_FAILURE, {"query": "…", "answer": "…"}),
                       "attribution": JevClient._build(ATTRIBUTION, {"events": []})}, ensure_ascii=False, indent=1)
