"""容错层：错误归因 → 分类重试 → 熔断 → 降级；幂等 / 补偿 / 断点续跑 / 乐观并发。"""
from .breaker import BreakerConfig, CircuitBreaker, InMemoryBreakerStore, RedisBreakerStore
from .checkpoint import CheckpointStore, update_with_retry
from .errors import classify_exception, classify_http_status, classify_tool_result
from .idempotency import InMemoryIdempotencyStore, RedisIdempotencyStore, SQLiteIdempotencyStore, make_key, run_idempotent
from .retry import RetryPolicy, call_with_retry
from .saga import Saga, SagaStep

__all__ = [
    "BreakerConfig", "CircuitBreaker", "InMemoryBreakerStore", "RedisBreakerStore", "CheckpointStore",
    "update_with_retry", "classify_exception", "classify_http_status", "classify_tool_result",
    "InMemoryIdempotencyStore", "RedisIdempotencyStore", "SQLiteIdempotencyStore", "make_key", "run_idempotent",
    "RetryPolicy", "call_with_retry", "Saga", "SagaStep",
]
