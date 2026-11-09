"""熔断器：滑动窗口失败率 → OPEN → 冷却后 HALF_OPEN 单探针 → 成功即 CLOSED。

状态存储可放 Redis（多 worker / 多实例共享同一熔断状态，一个实例探测到上游挂了，其他实例立刻快速失败），
Redis 自身不可用时自动降级为进程内存储，并对 Redis 也做短路（避免每次请求都等 Redis 超时）。

只有“上游不健康”的错误计入失败（NETWORK/TIMEOUT/EXTERNAL/RATE_LIMIT/AUTH），
调用方自己的参数错误（PARAM）、策略拦截（POLICY）不会把上游熔断掉。
"""
from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol, TypeVar

from ..types import AgentError, ErrorKind

log = logging.getLogger(__name__)
T = TypeVar("T")

COUNTED = {ErrorKind.NETWORK, ErrorKind.TIMEOUT, ErrorKind.EXTERNAL, ErrorKind.RATE_LIMIT, ErrorKind.AUTH,
           ErrorKind.SILENT}


@dataclass
class BreakerConfig:
    window_seconds: float = 30.0
    min_calls: int = 5
    failure_rate: float = 0.5
    open_seconds: float = 15.0
    probe_ttl_seconds: float = 10.0
    half_open_grace_seconds: float = 120.0  # open_until 过后多久内仍视为 half-open 状态


class BreakerStore(Protocol):
    def record(self, name: str, ok: bool, now: float, window: float) -> tuple[int, int]: ...
    def get_open_until(self, name: str) -> float | None: ...
    def set_open(self, name: str, until: float, ttl: float) -> None: ...
    def try_acquire_probe(self, name: str, ttl: float) -> bool: ...
    def close(self, name: str) -> None: ...


class InMemoryBreakerStore:
    def __init__(self) -> None:
        self._events: dict[str, deque[tuple[float, bool]]] = {}
        self._open: dict[str, tuple[float, float]] = {}   # name -> (until, expire_at)
        self._probe: dict[str, float] = {}

    def record(self, name: str, ok: bool, now: float, window: float) -> tuple[int, int]:
        q = self._events.setdefault(name, deque())
        q.append((now, ok))
        while q and q[0][0] < now - window:
            q.popleft()
        return len(q), sum(1 for _, o in q if not o)

    def get_open_until(self, name: str) -> float | None:
        v = self._open.get(name)
        if v and time.time() > v[1]:
            self._open.pop(name, None)
            return None
        return v[0] if v else None

    def set_open(self, name: str, until: float, ttl: float) -> None:
        self._open[name] = (until, time.time() + ttl)
        self._probe.pop(name, None)

    def try_acquire_probe(self, name: str, ttl: float) -> bool:
        exp = self._probe.get(name)
        now = time.time()
        if exp and exp > now:
            return False
        self._probe[name] = now + ttl
        return True

    def close(self, name: str) -> None:
        self._open.pop(name, None)
        self._probe.pop(name, None)
        self._events.pop(name, None)


_RECORD_LUA = """
local calls, fails = KEYS[1], KEYS[2]
local now, window, member, ok = tonumber(ARGV[1]), tonumber(ARGV[2]), ARGV[3], ARGV[4]
redis.call('ZADD', calls, now, member)
if ok == '0' then redis.call('ZADD', fails, now, member) end
redis.call('ZREMRANGEBYSCORE', calls, '-inf', now - window)
redis.call('ZREMRANGEBYSCORE', fails, '-inf', now - window)
local ttl = math.ceil(window * 2)
redis.call('EXPIRE', calls, ttl)
redis.call('EXPIRE', fails, ttl)
return {redis.call('ZCARD', calls), redis.call('ZCARD', fails)}
"""


class RedisBreakerStore:
    """Redis 共享熔断状态。键：cb:{name}:calls / :fails（ZSET 滑窗）、:open（open_until）、:probe（NX 探针锁）。"""

    def __init__(self, client: Any, prefix: str = "cb", redis_cooldown: float = 5.0) -> None:
        self.r = client
        self.prefix = prefix
        self._fallback = InMemoryBreakerStore()
        self._redis_down_until = 0.0
        self._cooldown = redis_cooldown
        self._script = None
        self.degraded_events = 0

    def _k(self, name: str, part: str) -> str:
        return f"{self.prefix}:{name}:{part}"

    def _redis_ok(self) -> bool:
        return time.time() >= self._redis_down_until

    def _trip_redis(self, exc: Exception) -> None:
        self.degraded_events += 1
        self._redis_down_until = time.time() + self._cooldown
        log.warning("redis breaker store unavailable, degrade to in-memory for %.0fs: %r", self._cooldown, exc)

    def _guard(self, op: Callable[[], T], fallback: Callable[[], T]) -> T:
        if not self._redis_ok():
            return fallback()
        try:
            return op()
        except Exception as exc:  # noqa: BLE001
            self._trip_redis(exc)
            return fallback()

    def record(self, name: str, ok: bool, now: float, window: float) -> tuple[int, int]:
        def op() -> tuple[int, int]:
            if self._script is None:
                self._script = self.r.register_script(_RECORD_LUA)
            calls, fails = self._script(keys=[self._k(name, "calls"), self._k(name, "fails")],
                                        args=[now, window, uuid.uuid4().hex, "1" if ok else "0"])
            return int(calls), int(fails)
        return self._guard(op, lambda: self._fallback.record(name, ok, now, window))

    def get_open_until(self, name: str) -> float | None:
        def op() -> float | None:
            v = self.r.get(self._k(name, "open"))
            return float(v) if v is not None else None
        return self._guard(op, lambda: self._fallback.get_open_until(name))

    def set_open(self, name: str, until: float, ttl: float) -> None:
        def op() -> None:
            pipe = self.r.pipeline()
            pipe.set(self._k(name, "open"), str(until), px=int(ttl * 1000))
            pipe.delete(self._k(name, "probe"))
            pipe.execute()
        self._guard(op, lambda: self._fallback.set_open(name, until, ttl))

    def try_acquire_probe(self, name: str, ttl: float) -> bool:
        return self._guard(lambda: bool(self.r.set(self._k(name, "probe"), "1", nx=True, px=int(ttl * 1000))),
                           lambda: self._fallback.try_acquire_probe(name, ttl))

    def close(self, name: str) -> None:
        self._guard(lambda: self.r.delete(*(self._k(name, p) for p in ("open", "probe", "calls", "fails"))),
                    lambda: self._fallback.close(name))


class CircuitBreaker:
    def __init__(self, name: str, store: BreakerStore | None = None, cfg: BreakerConfig | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.name = name
        self.store = store or InMemoryBreakerStore()
        self.cfg = cfg or BreakerConfig()
        self.clock = clock

    def state(self) -> str:
        until = self.store.get_open_until(self.name)
        if until is None:
            return "closed"
        return "open" if self.clock() < until else "half_open"

    def acquire(self) -> bool:
        """返回是否是半开探针；不允许调用时抛 CIRCUIT_OPEN。"""
        until = self.store.get_open_until(self.name)
        if until is None:
            return False
        if self.clock() < until:
            raise AgentError(ErrorKind.CIRCUIT_OPEN, f"circuit {self.name} open", source=self.name)
        if self.store.try_acquire_probe(self.name, self.cfg.probe_ttl_seconds):
            return True
        raise AgentError(ErrorKind.CIRCUIT_OPEN, f"circuit {self.name} half-open, probe in flight", source=self.name)

    def on_success(self, probe: bool) -> None:
        if probe:
            self.store.close(self.name)
        else:
            self.store.record(self.name, True, self.clock(), self.cfg.window_seconds)

    def on_failure(self, err: AgentError, probe: bool) -> None:
        if err.kind not in COUNTED:
            return
        now = self.clock()
        if probe:
            self._open(now)
            return
        calls, fails = self.store.record(self.name, False, now, self.cfg.window_seconds)
        if calls >= self.cfg.min_calls and fails / calls >= self.cfg.failure_rate:
            self._open(now)

    def _open(self, now: float) -> None:
        self.store.set_open(self.name, now + self.cfg.open_seconds, self.cfg.open_seconds + self.cfg.half_open_grace_seconds)

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        from .errors import classify_exception
        probe = self.acquire()
        try:
            result = await fn()
        except Exception as exc:  # noqa: BLE001
            err = classify_exception(exc, source=self.name)
            self.on_failure(err, probe)
            raise err from exc
        self.on_success(probe)
        return result
