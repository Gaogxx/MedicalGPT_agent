import asyncio

import fakeredis
import httpx
import pytest

from medagent.resilience.breaker import BreakerConfig, CircuitBreaker, InMemoryBreakerStore, RedisBreakerStore
from medagent.resilience.checkpoint import CheckpointStore, ConcurrentModification, update_with_retry
from medagent.resilience.errors import classify_exception, classify_http_status, classify_tool_result
from medagent.resilience.idempotency import InMemoryIdempotencyStore, RedisIdempotencyStore, SQLiteIdempotencyStore, run_idempotent
from medagent.resilience.retry import RetryPolicy, call_with_retry
from medagent.resilience.saga import Saga, SagaStep
from medagent.types import AgentError, ErrorKind


def test_error_classification():
    assert classify_http_status(429, "").kind == ErrorKind.RATE_LIMIT
    assert classify_http_status(401, "").kind == ErrorKind.AUTH
    assert classify_http_status(422, "").kind == ErrorKind.PARAM
    assert classify_http_status(503, "").kind == ErrorKind.EXTERNAL
    assert classify_exception(httpx.ConnectError("x")).kind == ErrorKind.NETWORK
    assert classify_exception(asyncio.TimeoutError()).kind == ErrorKind.TIMEOUT
    assert classify_tool_result("t", True, "Error executing tool t: PARAM: name 不能为空").kind == ErrorKind.PARAM
    assert classify_tool_result("t", True, "Error executing tool t").kind == ErrorKind.EXTERNAL
    assert classify_tool_result("t", False, "") is None


async def test_retry_only_retryable():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("reset")
        return "ok"
    assert await call_with_retry(flaky, policy=RetryPolicy(base_delay=0), source="x") == "ok"
    calls["n"] = 0

    async def bad_param():
        calls["n"] += 1
        raise AgentError(ErrorKind.PARAM, "bad")
    with pytest.raises(AgentError):
        await call_with_retry(bad_param, policy=RetryPolicy(base_delay=0), source="x")
    assert calls["n"] == 1   # 参数错不重试


def _trip(br: CircuitBreaker, n: int = 5):
    for _ in range(n):
        br.on_failure(AgentError(ErrorKind.TIMEOUT, "t"), probe=False)


def test_breaker_open_halfopen_close():
    t = {"now": 1000.0}
    br = CircuitBreaker("x", InMemoryBreakerStore(), BreakerConfig(min_calls=3, open_seconds=10), clock=lambda: t["now"])
    _trip(br, 3)
    assert br.state() == "open"
    with pytest.raises(AgentError) as e:
        br.acquire()
    assert e.value.kind == ErrorKind.CIRCUIT_OPEN
    t["now"] += 11
    assert br.acquire() is True          # 半开：一个探针
    with pytest.raises(AgentError):
        br.acquire()                     # 第二个并发请求被拒
    br.on_success(probe=True)
    assert br.state() == "closed"


def test_param_errors_do_not_trip_breaker():
    br = CircuitBreaker("x", cfg=BreakerConfig(min_calls=2))
    for _ in range(5):
        br.on_failure(AgentError(ErrorKind.PARAM, "p"), probe=False)
    assert br.state() == "closed"


def test_redis_breaker_shared_and_degrades():
    server = fakeredis.FakeServer()
    cfg = BreakerConfig(min_calls=3)
    a = CircuitBreaker("llm:x", RedisBreakerStore(fakeredis.FakeRedis(server=server)), cfg)
    b = CircuitBreaker("llm:x", RedisBreakerStore(fakeredis.FakeRedis(server=server)), cfg)
    _trip(a, 3)
    assert b.state() == "open"

    class Down:
        def __getattr__(self, name):
            def f(*a, **k):
                raise ConnectionError("down")
            return f
    store = RedisBreakerStore(Down())
    c = CircuitBreaker("llm:y", store, cfg)
    _trip(c, 3)
    assert c.state() == "open" and store.degraded_events >= 1


@pytest.mark.parametrize("store", [InMemoryIdempotencyStore(), SQLiteIdempotencyStore(),
                                   RedisIdempotencyStore(fakeredis.FakeRedis())])
async def test_idempotency(store):
    n = {"calls": 0}

    async def side_effect():
        n["calls"] += 1
        return {"id": n["calls"]}
    r1, hit1 = await run_idempotent(store, "k1", side_effect)
    r2, hit2 = await run_idempotent(store, "k1", side_effect)
    assert r1 == r2 and not hit1 and hit2 and n["calls"] == 1


async def test_saga_compensates_in_reverse_and_dead_letters():
    log = []

    async def a1(ctx): log.append("a1"); return 1
    async def c1(ctx, r): log.append("c1")
    async def a2(ctx): log.append("a2"); return 2
    async def c2(ctx, r): raise AgentError(ErrorKind.EXTERNAL, "down")
    async def a3(ctx): raise AgentError(ErrorKind.EXTERNAL, "boom")
    dead = []
    res = await Saga("s", [SagaStep("1", a1, c1), SagaStep("2", a2, c2), SagaStep("3", a3)],
                     on_dead_letter=dead.append).run({})
    assert not res.ok and log == ["a1", "a2", "c1"] and len(dead) == 1 and dead[0]["step"] == "2"


def test_checkpoint_occ():
    s = CheckpointStore()
    s.create("r", {"x": 0})
    _, v, _ = s.load("r")
    s.save("r", {"x": 1}, v)
    with pytest.raises(ConcurrentModification):
        s.save("r", {"x": 2}, v)                     # 过期版本被拒，不会悄悄覆盖
    update_with_retry(s, "r", lambda st, status: ({"x": st["x"] + 10}, status))
    assert s.load("r")[0]["x"] == 11
    s.request_pause("r")
    assert s.status("r") == "pause_requested"
