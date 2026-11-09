"""模型池：分层（strong 云端医疗大模型 / local 本地 MedicalGPT / small 控制面小模型）+ 层内降级链 + 层间兜底。

每次调用的保护顺序：熔断（快速失败）→ 超时 → 分类重试 → 记录事件（供归因）→ 失败则换同层下一个 provider → 换层。
small 层两种用途：控制面（查询改写、上下文压缩、判别），以及 v3 起的“简单问答”档——只回答经过门控的低风险科普/概念题，
回答同样过校验器，被拦截就升级到 local（见 router/policy.py 的 TieredRouter）。
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import yaml

from ..resilience.breaker import BreakerConfig, CircuitBreaker, InMemoryBreakerStore
from ..resilience.errors import classify_exception
from ..resilience.retry import RetryBudget, RetryPolicy, call_with_retry
from ..types import AgentError, ErrorKind, LLMResponse

# small 档失败（不可用）时向上兜底到 local，再到 strong；不会反向从 local 降到 small
TIER_FALLBACK = {"strong": ["strong", "local"], "local": ["local", "strong"], "small": ["small", "local", "strong"]}


@dataclass
class ProviderSpec:
    name: str
    tier: str
    base_url: str
    model: str
    api_key_env: str = ""
    timeout: float = 60.0
    price_in: float = 0.0
    price_out: float = 0.0


@dataclass
class ModelPool:
    tiers: dict[str, list[Any]]
    breaker_store: Any = field(default_factory=InMemoryBreakerStore)
    breaker_cfg: BreakerConfig = field(default_factory=BreakerConfig)
    retry: RetryPolicy = field(default_factory=lambda: RetryPolicy(budget=RetryBudget()))
    call_timeout: float = 60.0
    breakers: dict[str, CircuitBreaker] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str, **kw: Any) -> "ModelPool":
        from .providers import OpenAICompatProvider
        cfg = yaml.safe_load(open(path, encoding="utf-8"))
        tiers: dict[str, list[Any]] = {}
        for tier, items in cfg["tiers"].items():
            tiers[tier] = [OpenAICompatProvider(i["name"], i["base_url"], i["model"], i.get("api_key_env", ""),
                                                i.get("timeout", 60.0), i.get("price_in", 0.0), i.get("price_out", 0.0),
                                                i.get("extra_body"))
                           for i in items]
        return cls(tiers, **kw)

    def breaker(self, name: str) -> CircuitBreaker:
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(f"llm:{name}", self.breaker_store, self.breaker_cfg)
        return self.breakers[name]

    async def generate(self, tier: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                       *, events: list[dict[str, Any]] | None = None, allow_cross_tier: bool = True,
                       **kw: Any) -> LLMResponse:
        order = TIER_FALLBACK.get(tier, [tier]) if allow_cross_tier else [tier]
        last: AgentError | None = None
        for t in order:
            for p in self.tiers.get(t, []):
                br = self.breaker(p.name)
                t0 = time.perf_counter()

                async def once(p=p) -> LLMResponse:
                    return await asyncio.wait_for(p.chat(messages, tools, **kw), timeout=self.call_timeout)

                try:
                    attempts: list = []
                    resp = await br.call(lambda: call_with_retry(once, policy=self.retry, source=f"llm:{p.name}",
                                                                 attempts_log=attempts))
                    if events is not None:
                        events.append({"stage": "llm", "source": f"llm:{p.name}", "tier": t, "ok": True,
                                       "retries": len(attempts), "ms": (time.perf_counter() - t0) * 1000})
                    resp.model = p.name
                    return resp
                except Exception as exc:  # noqa: BLE001
                    last = classify_exception(exc, source=f"llm:{p.name}")
                    if events is not None:
                        events.append({"stage": "llm", "source": f"llm:{p.name}", "tier": t, "ok": False,
                                       "error": last.to_dict(), "ms": (time.perf_counter() - t0) * 1000})
                    if last.kind in (ErrorKind.PARAM, ErrorKind.POLICY):
                        raise last  # 请求本身有问题，换模型也没用
        raise last or AgentError(ErrorKind.EXTERNAL, f"no provider available for tier {tier}", source="pool")
