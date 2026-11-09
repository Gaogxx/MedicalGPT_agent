"""错误归因：把异常 / HTTP 状态 / MCP 工具结果统一映射为 ErrorKind。

归因顺序（越靠前越确定）：
  1. 本地异常类型（超时、连接错误）         → TIMEOUT / NETWORK
  2. HTTP 状态码（429/401/4xx/5xx）         → RATE_LIMIT / AUTH / PARAM / EXTERNAL
  3. MCP 工具 is_error 的错误前缀与文本     → PARAM / EXTERNAL / ...
  4. 结果“成功”但校验器判定异常             → SILENT（见 verify/）
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from ..types import AgentError, ErrorKind

_PARAM_HINT = re.compile(r"validation|missing|required|invalid|参数|必填|类型错误|unexpected keyword|field required", re.I)
_RATE_HINT = re.compile(r"rate.?limit|too many requests|quota|限流|配额", re.I)
_AUTH_HINT = re.compile(r"unauthori[sz]ed|forbidden|api.?key|鉴权|认证失败", re.I)
_TIMEOUT_HINT = re.compile(r"timed? ?out|超时", re.I)
_NETWORK_HINT = re.compile(r"connection (refused|reset|error)|name resolution|unreachable|broken pipe|连接", re.I)


def classify_http_status(status: int, body: str = "", *, source: str = "", headers: dict[str, str] | None = None) -> AgentError:
    headers = {k.lower(): v for k, v in (headers or {}).items()}
    retry_after = None
    if "retry-after" in headers:
        try:
            retry_after = float(headers["retry-after"])
        except ValueError:
            retry_after = None
    msg = f"HTTP {status}: {body[:300]}"
    if status == 429:
        return AgentError(ErrorKind.RATE_LIMIT, msg, source=source, status=status, retry_after=retry_after)
    if status in (401, 403):
        return AgentError(ErrorKind.AUTH, msg, source=source, status=status)
    if status in (408, 504):
        return AgentError(ErrorKind.TIMEOUT, msg, source=source, status=status)
    if status in (400, 404, 409, 413, 422):
        # 某些网关把上游过载也报 400，正文里带 rate limit 字样时以正文为准
        if _RATE_HINT.search(body):
            return AgentError(ErrorKind.RATE_LIMIT, msg, source=source, status=status, retry_after=retry_after)
        return AgentError(ErrorKind.PARAM, msg, source=source, status=status)
    if status >= 500:
        return AgentError(ErrorKind.EXTERNAL, msg, source=source, status=status, retry_after=retry_after)
    return AgentError(ErrorKind.UNKNOWN, msg, source=source, status=status)


def classify_exception(exc: BaseException, *, source: str = "") -> AgentError:
    if isinstance(exc, AgentError):
        if not exc.source:
            exc.source = source
        return exc
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return AgentError(ErrorKind.TIMEOUT, f"timeout: {exc!r}", source=source, cause=exc)
    try:
        import httpx
        if isinstance(exc, httpx.TimeoutException):
            return AgentError(ErrorKind.TIMEOUT, f"timeout: {exc!r}", source=source, cause=exc)
        if isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError,
                            httpx.NetworkError)):
            return AgentError(ErrorKind.NETWORK, f"network: {exc!r}", source=source, cause=exc)
        if isinstance(exc, httpx.HTTPStatusError):
            r = exc.response
            return classify_http_status(r.status_code, r.text, source=source, headers=dict(r.headers))
    except ImportError:  # pragma: no cover
        pass
    if isinstance(exc, (ConnectionError, OSError)):
        return AgentError(ErrorKind.NETWORK, f"network: {exc!r}", source=source, cause=exc)
    if isinstance(exc, (json.JSONDecodeError,)):
        return AgentError(ErrorKind.MODEL, f"unparseable output: {exc}", source=source, cause=exc)
    if isinstance(exc, (TypeError, ValueError, KeyError)):
        return AgentError(ErrorKind.PARAM, f"bad parameters: {exc!r}", source=source, cause=exc)
    return classify_text(str(exc), source=source, cause=exc)


def classify_text(text: str, *, source: str = "", cause: BaseException | None = None) -> AgentError:
    for pat, kind in ((_RATE_HINT, ErrorKind.RATE_LIMIT), (_AUTH_HINT, ErrorKind.AUTH),
                      (_TIMEOUT_HINT, ErrorKind.TIMEOUT), (_NETWORK_HINT, ErrorKind.NETWORK),
                      (_PARAM_HINT, ErrorKind.PARAM)):
        if pat.search(text):
            return AgentError(kind, text[:300], source=source, cause=cause)
    return AgentError(ErrorKind.UNKNOWN, text[:300], source=source, cause=cause)


# MCP Server 侧约定：工具失败时抛 ToolError("<KIND>: ...")（见 mcp_servers/medical_server.py），
# SDK 会包成 "Error executing tool <name>: <KIND>: ..."，客户端据此精确归因；
# 第三方 MCP Server 没有前缀时回退到文本启发式；SDK 吞掉细节的崩溃（只剩 "Error executing tool x"）按 EXTERNAL。
_PREFIX = re.compile(r"(?:^|:)\s*(NETWORK|TIMEOUT|RATE_LIMIT|AUTH|PARAM|EXTERNAL|CONFLICT|POLICY)\s*:")


def classify_tool_result(name: str, is_error: bool, text: str, structured: Any = None) -> AgentError | None:
    source = f"tool:{name}"
    if not is_error:
        return None
    m = _PREFIX.search(text or "")
    if m:
        return AgentError(ErrorKind(m.group(1).lower()), text, source=source)
    err = classify_text(text or "", source=source)
    if err.kind == ErrorKind.UNKNOWN:
        err.kind = ErrorKind.EXTERNAL  # 工具自身抛错且无法细分时，按外部依赖故障处理（可重试一次）
    return err
