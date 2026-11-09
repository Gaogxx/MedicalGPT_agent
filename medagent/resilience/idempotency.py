"""幂等：同一个业务动作（会话 + 步骤 + 工具 + 规范化参数）只执行一次。

用途：
  - 重试 / 断点续跑重放时，有副作用的工具（预约挂号、写随访提醒）不会重复执行
  - 同一请求被并发提交两次（用户连点），第二次拿到 CONFLICT 或第一次的结果
状态机：(无) --begin--> in_progress --complete--> done(带结果)
                               └--fail--> (删除，允许重试)
in_progress 带 TTL：执行者崩溃后锁自动过期，避免永久卡死。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from typing import Any, Awaitable, Callable, Protocol

from ..types import AgentError, ErrorKind


def make_key(scope: str, tool: str, args: dict[str, Any]) -> str:
    canon = json.dumps({k: v for k, v in args.items() if k != "idempotency_key"}, ensure_ascii=False, sort_keys=True)
    return "idem:" + hashlib.sha256(f"{scope}|{tool}|{canon}".encode()).hexdigest()[:32]


class IdempotencyStore(Protocol):
    def begin(self, key: str, ttl: float) -> tuple[str, Any]: ...
    def complete(self, key: str, result: Any, ttl: float) -> None: ...
    def fail(self, key: str) -> None: ...


class InMemoryIdempotencyStore:
    def __init__(self) -> None:
        self._d: dict[str, tuple[str, Any, float]] = {}
        self._lock = threading.Lock()

    def begin(self, key: str, ttl: float) -> tuple[str, Any]:
        now = time.time()
        with self._lock:
            cur = self._d.get(key)
            if cur and cur[2] > now:
                return cur[0], cur[1]
            self._d[key] = ("in_progress", None, now + ttl)
            return "new", None

    def complete(self, key: str, result: Any, ttl: float) -> None:
        with self._lock:
            self._d[key] = ("done", result, time.time() + ttl)

    def fail(self, key: str) -> None:
        with self._lock:
            self._d.pop(key, None)


class RedisIdempotencyStore:
    def __init__(self, client: Any) -> None:
        self.r = client

    def begin(self, key: str, ttl: float) -> tuple[str, Any]:
        if self.r.set(key, json.dumps({"s": "in_progress"}), nx=True, px=int(ttl * 1000)):
            return "new", None
        raw = self.r.get(key)
        if raw is None:  # 恰好过期，再抢一次
            return self.begin(key, ttl)
        v = json.loads(raw)
        return ("done", v.get("r")) if v["s"] == "done" else ("in_progress", None)

    def complete(self, key: str, result: Any, ttl: float) -> None:
        self.r.set(key, json.dumps({"s": "done", "r": result}, ensure_ascii=False), px=int(ttl * 1000))

    def fail(self, key: str) -> None:
        self.r.delete(key)


class SQLiteIdempotencyStore:
    """单机 / 无 Redis 时的持久化实现（进程重启后仍然幂等）。"""

    def __init__(self, path: str = ":memory:") -> None:
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.execute("CREATE TABLE IF NOT EXISTS idem (k TEXT PRIMARY KEY, s TEXT, r TEXT, exp REAL)")
        self._lock = threading.Lock()

    def begin(self, key: str, ttl: float) -> tuple[str, Any]:
        now = time.time()
        with self._lock:
            self.db.execute("DELETE FROM idem WHERE k=? AND exp<?", (key, now))
            cur = self.db.execute("INSERT OR IGNORE INTO idem VALUES (?,?,?,?)", (key, "in_progress", None, now + ttl))
            if cur.rowcount == 1:
                return "new", None
            s, r = self.db.execute("SELECT s, r FROM idem WHERE k=?", (key,)).fetchone()
            return (s, json.loads(r)) if s == "done" else (s, None)

    def complete(self, key: str, result: Any, ttl: float) -> None:
        with self._lock:
            self.db.execute("UPDATE idem SET s='done', r=?, exp=? WHERE k=?",
                            (json.dumps(result, ensure_ascii=False), time.time() + ttl, key))

    def fail(self, key: str) -> None:
        with self._lock:
            self.db.execute("DELETE FROM idem WHERE k=?", (key,))


async def run_idempotent(store: IdempotencyStore, key: str, fn: Callable[[], Awaitable[Any]], *,
                         lock_ttl: float = 60.0, result_ttl: float = 86400.0) -> tuple[Any, bool]:
    """返回 (结果, 是否命中幂等缓存)。in_progress 时抛 CONFLICT，由上层决定等待还是告知用户。"""
    status, cached = store.begin(key, lock_ttl)
    if status == "done":
        return cached, True
    if status == "in_progress":
        raise AgentError(ErrorKind.CONFLICT, f"operation {key} already in progress", source="idempotency")
    try:
        result = await fn()
    except BaseException:
        store.fail(key)
        raise
    store.complete(key, result, result_ttl)
    return result, False
