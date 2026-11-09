"""公共类型：错误分类体系、消息、工具调用。"""
from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from typing import Any


class ErrorKind(str, enum.Enum):
    """出错归因的类别。决定重试 / 降级 / 补偿策略（见 resilience/retry.py）。"""

    NETWORK = "network"          # 连接失败、连接被重置、DNS
    TIMEOUT = "timeout"          # 读/写/整体超时
    RATE_LIMIT = "rate_limit"    # 429 / 配额
    AUTH = "auth"                # 401/403，密钥错误或过期
    PARAM = "param"              # 400/422、工具参数缺失或类型错误（调用方的错）
    MODEL = "model"              # 模型输出不可解析：工具名不存在、JSON 损坏、格式泄漏
    SILENT = "silent"            # 调用“成功”但结果错误/空/截断/无依据（静默失败）
    EXTERNAL = "external"        # 外部依赖 5xx、工具后端异常、知识库源不可用
    POLICY = "policy"            # 安全策略拦截（不重试）
    CIRCUIT_OPEN = "circuit_open"  # 熔断器打开，快速失败
    CONFLICT = "conflict"        # 乐观并发冲突 / 幂等键正在执行
    UNKNOWN = "unknown"


RETRYABLE = {ErrorKind.NETWORK, ErrorKind.TIMEOUT, ErrorKind.RATE_LIMIT, ErrorKind.EXTERNAL}


class AgentError(Exception):
    """带归因信息的统一异常。source 形如 'llm:baichuan' / 'tool:search_drug' / 'rag:openfda'。"""

    def __init__(self, kind: ErrorKind, message: str, *, source: str = "", status: int | None = None,
                 retry_after: float | None = None, cause: BaseException | None = None):
        super().__init__(message)
        self.kind = kind
        self.source = source
        self.status = status
        self.retry_after = retry_after
        self.__cause__ = cause

    @property
    def retryable(self) -> bool:
        return self.kind in RETRYABLE

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "source": self.source, "status": self.status, "message": str(self)}


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = ""

    def canonical(self) -> str:
        return json.dumps({"name": self.name, "arguments": self.arguments}, ensure_ascii=False, sort_keys=True)


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    cost: float = 0.0


@dataclass
class ToolResult:
    name: str
    ok: bool
    data: Any
    error: AgentError | None = None
    from_cache: bool = False

    def as_observation(self) -> str:
        if self.ok:
            return json.dumps(self.data, ensure_ascii=False)
        return json.dumps({"error": str(self.error), "kind": self.error.kind.value if self.error else "unknown"},
                          ensure_ascii=False)
