"""Saga 补偿：多步有副作用的操作（例：预约挂号 → 写随访提醒 → 发送通知），
中途某步失败时按逆序执行已完成步骤的补偿动作（取消预约 / 删除提醒），保证最终一致。

补偿本身也必须幂等、可重试；重试仍失败的补偿进入死信（dead letter），由人工 / 定时任务兜底，
并在审计日志中留痕——医疗场景下“悄悄留下一个没人知道的预约”比报错更糟。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..types import AgentError, ErrorKind
from .errors import classify_exception
from .retry import RetryPolicy, call_with_retry


@dataclass
class SagaStep:
    name: str
    action: Callable[[dict[str, Any]], Awaitable[Any]]
    compensate: Callable[[dict[str, Any], Any], Awaitable[None]] | None = None


@dataclass
class SagaLogEntry:
    step: str
    event: str            # done | failed | compensated | compensation_failed
    detail: str = ""
    at: float = field(default_factory=time.time)


@dataclass
class SagaResult:
    ok: bool
    results: dict[str, Any]
    log: list[SagaLogEntry]
    error: AgentError | None = None
    dead_letters: list[dict[str, Any]] = field(default_factory=list)


class Saga:
    def __init__(self, name: str, steps: list[SagaStep], *, compensation_policy: RetryPolicy | None = None,
                 on_dead_letter: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.name = name
        self.steps = steps
        self.policy = compensation_policy or RetryPolicy(max_attempts={k: 3 for k in ErrorKind}, base_delay=0.05)
        self.on_dead_letter = on_dead_letter

    async def run(self, ctx: dict[str, Any]) -> SagaResult:
        done: list[tuple[SagaStep, Any]] = []
        log: list[SagaLogEntry] = []
        results: dict[str, Any] = {}
        for step in self.steps:
            try:
                res = await step.action(ctx)
            except Exception as exc:  # noqa: BLE001
                err = classify_exception(exc, source=f"saga:{self.name}:{step.name}")
                log.append(SagaLogEntry(step.name, "failed", str(err)))
                dead = await self._compensate(done, ctx, log)
                return SagaResult(False, results, log, err, dead)
            done.append((step, res))
            results[step.name] = res
            ctx[step.name] = res
            log.append(SagaLogEntry(step.name, "done"))
        return SagaResult(True, results, log)

    async def _compensate(self, done: list[tuple[SagaStep, Any]], ctx: dict[str, Any],
                          log: list[SagaLogEntry]) -> list[dict[str, Any]]:
        dead: list[dict[str, Any]] = []
        for step, res in reversed(done):
            if step.compensate is None:
                continue
            comp = step.compensate
            try:
                await call_with_retry(lambda: comp(ctx, res), policy=self.policy,
                                      source=f"saga:{self.name}:{step.name}:compensate")
                log.append(SagaLogEntry(step.name, "compensated"))
            except AgentError as err:
                item = {"saga": self.name, "step": step.name, "result": res, "error": err.to_dict(), "at": time.time()}
                dead.append(item)
                log.append(SagaLogEntry(step.name, "compensation_failed", str(err)))
                if self.on_dead_letter:
                    self.on_dead_letter(item)
        return dead
