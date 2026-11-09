"""按错误类别区分的重试：只重试“换个时间就可能成功”的错误；参数错/鉴权错/策略拦截不盲目重试。

- NETWORK / TIMEOUT / EXTERNAL：指数退避 + full jitter
- RATE_LIMIT：优先遵守 Retry-After
- PARAM / MODEL：不在这里重试，交给上层“修复后重试一次”（让模型按错误信息改参数 / 改格式）
- AUTH / POLICY / CIRCUIT_OPEN：立即失败，交给降级链
另有全局重试预算（retry budget），防止下游故障时重试风暴把故障放大。
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, TypeVar

from ..types import AgentError, ErrorKind
from .errors import classify_exception

T = TypeVar("T")


@dataclass
class RetryBudget:
    """令牌桶：每次正常请求存入 ratio 个令牌，每次重试消耗 1 个。"""

    ratio: float = 0.2
    min_tokens: float = 10.0
    _tokens: float = 10.0

    def deposit(self) -> None:
        self._tokens = min(self._tokens + self.ratio, 1000.0)

    def withdraw(self) -> bool:
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


@dataclass
class RetryPolicy:
    max_attempts: dict[ErrorKind, int] = field(default_factory=lambda: {
        ErrorKind.NETWORK: 3, ErrorKind.TIMEOUT: 2, ErrorKind.RATE_LIMIT: 3, ErrorKind.EXTERNAL: 2,
    })
    base_delay: float = 0.2
    max_delay: float = 8.0
    budget: RetryBudget | None = None

    def attempts_for(self, kind: ErrorKind) -> int:
        return self.max_attempts.get(kind, 1)

    def delay(self, attempt: int, err: AgentError) -> float:
        if err.kind == ErrorKind.RATE_LIMIT and err.retry_after:
            return min(err.retry_after, self.max_delay)
        cap = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        return random.uniform(0, cap)  # full jitter


@dataclass
class AttemptRecord:
    attempt: int
    kind: str
    delay: float
    message: str
    at: float


async def call_with_retry(fn: Callable[[], Awaitable[T]], *, policy: RetryPolicy, source: str,
                          attempts_log: list[AttemptRecord] | None = None,
                          sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> T:
    attempt = 0
    if policy.budget:
        policy.budget.deposit()
    while True:
        attempt += 1
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 —— 统一归因
            err = classify_exception(exc, source=source)
            allowed = policy.attempts_for(err.kind)
            if attempt >= allowed or not err.retryable:
                if attempts_log is not None:
                    attempts_log.append(AttemptRecord(attempt, err.kind.value, 0.0, str(err), time.time()))
                raise err from exc
            if policy.budget and not policy.budget.withdraw():  # 预算耗尽：不再重试，原样上抛
                if attempts_log is not None:
                    attempts_log.append(AttemptRecord(attempt, err.kind.value, 0.0, "retry budget exhausted", time.time()))
                raise err from exc
            d = policy.delay(attempt, err)
            if attempts_log is not None:
                attempts_log.append(AttemptRecord(attempt, err.kind.value, d, str(err), time.time()))
            await sleep(d)
