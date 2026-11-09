"""断点续跑 + 乐观并发（OCC）。

每个 Agent run 是一行：(run_id, version, status, state_json)。
  - 每走完一个节点就 save(expected_version) —— UPDATE ... WHERE version = ?；
    rowcount == 0 说明别人（另一个 worker / 用户的暂停请求 / 同会话的并发消息）先改过，抛 CONFLICT。
  - 暂停：request_pause 只是把 status 置为 pause_requested（同样走 CAS），
    运行中的 Agent 在节点边界看到后保存状态并退出（协作式暂停，不会在工具调用中途被腰斩）。
  - 恢复：load → 从 state["next_node"] 继续；已完成的有副作用步骤靠幂等键跳过，不会重复执行。
  - 崩溃恢复：status=running 但 heartbeat 过期的 run 可被接管（同样靠版本号保证只有一个接管者）。
为什么不用悲观锁：Agent 一步可能持续几十秒（LLM 调用），持锁会阻塞暂停请求和同会话消息；冲突概率低，OCC 更合适。
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Callable

from ..types import AgentError, ErrorKind

RUNNING, PAUSE_REQUESTED, PAUSED, DONE, FAILED = "running", "pause_requested", "paused", "done", "failed"


class ConcurrentModification(AgentError):
    def __init__(self, run_id: str, expected: int):
        super().__init__(ErrorKind.CONFLICT, f"run {run_id}: version {expected} is stale", source="checkpoint")


class CheckpointStore:
    def __init__(self, path: str = ":memory:") -> None:
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.execute("""CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, version INTEGER NOT NULL, status TEXT NOT NULL,
            state TEXT NOT NULL, heartbeat REAL NOT NULL, updated REAL NOT NULL)""")
        self._lock = threading.Lock()

    def create(self, run_id: str, state: dict[str, Any]) -> int:
        now = time.time()
        with self._lock:
            cur = self.db.execute("INSERT OR IGNORE INTO runs VALUES (?,?,?,?,?,?)",
                                  (run_id, 1, RUNNING, json.dumps(state, ensure_ascii=False), now, now))
        if cur.rowcount == 0:
            raise AgentError(ErrorKind.CONFLICT, f"run {run_id} exists", source="checkpoint")
        return 1

    def load(self, run_id: str) -> tuple[dict[str, Any], int, str]:
        row = self.db.execute("SELECT state, version, status FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return json.loads(row[0]), row[1], row[2]

    def save(self, run_id: str, state: dict[str, Any], expected_version: int, status: str | None = None) -> int:
        now = time.time()
        with self._lock:
            if status is None:
                cur = self.db.execute(
                    "UPDATE runs SET state=?, version=version+1, heartbeat=?, updated=? WHERE run_id=? AND version=?",
                    (json.dumps(state, ensure_ascii=False), now, now, run_id, expected_version))
            else:
                cur = self.db.execute(
                    "UPDATE runs SET state=?, status=?, version=version+1, heartbeat=?, updated=? "
                    "WHERE run_id=? AND version=?",
                    (json.dumps(state, ensure_ascii=False), status, now, now, run_id, expected_version))
        if cur.rowcount == 0:
            raise ConcurrentModification(run_id, expected_version)
        return expected_version + 1

    def status(self, run_id: str) -> str:
        return self.load(run_id)[2]

    def request_pause(self, run_id: str) -> int:
        def mutate(state: dict[str, Any], status: str) -> tuple[dict[str, Any], str]:
            return state, PAUSE_REQUESTED if status == RUNNING else status
        return update_with_retry(self, run_id, mutate)

    def mark_resumable(self, run_id: str) -> int:
        def mutate(state: dict[str, Any], status: str) -> tuple[dict[str, Any], str]:
            if status not in (PAUSED, FAILED, RUNNING, PAUSE_REQUESTED):
                raise AgentError(ErrorKind.CONFLICT, f"run {run_id} is {status}", source="checkpoint")
            return state, RUNNING
        return update_with_retry(self, run_id, mutate)

    def stale_runs(self, heartbeat_timeout: float) -> list[str]:
        cutoff = time.time() - heartbeat_timeout
        return [r[0] for r in self.db.execute("SELECT run_id FROM runs WHERE status=? AND heartbeat<?", (RUNNING, cutoff))]


def update_with_retry(store: CheckpointStore, run_id: str,
                      mutate: Callable[[dict[str, Any], str], tuple[dict[str, Any], str]],
                      max_retries: int = 8) -> int:
    """读-改-写 CAS 循环：冲突就重读重算，最多 max_retries 次。mutate 必须是纯函数（可能被调用多次）。"""
    for _ in range(max_retries):
        state, version, status = store.load(run_id)
        new_state, new_status = mutate(state, status)
        try:
            return store.save(run_id, new_state, version, status=new_status)
        except ConcurrentModification:
            time.sleep(0.001)
            continue
    raise ConcurrentModification(run_id, -1)
