"""工具调用解析：OpenAI tool_calls 优先；否则从文本里兜底解析三种格式——
  1. MedicalGPT 训练格式   Action: name \\n Action Input: {...}
  2. Qwen/hermes 格式       <tool_call>{"name":..., "arguments":...}</tool_call>
  3. 泄漏格式               前缀垃圾 token + {"name":..., "arguments":...}（vLLM hermes parser 流式出错时）
第 3 种正是基线里 “ptom\\n{\"name\": \"check_symptom\" ...}” 的成因：解析失败的工具调用被当成了最终回复。"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any

from ..types import ToolCall

_ACTION = re.compile(r"Action:\s*([a-zA-Z_][\w]*)\s*\n\s*Action Input:\s*(\{.*?\})\s*(?:\n|$)", re.S)
_HERMES = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_BARE = re.compile(r'\{\s*"name"\s*:\s*"([a-zA-Z_]\w*)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}+', re.S)


def _loads(s: str) -> dict[str, Any] | None:
    for cand in (s, s.rstrip("}") + "}"):
        try:
            v = json.loads(cand)
            return v if isinstance(v, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def parse_tool_calls(content: str, openai_tool_calls: list[dict[str, Any]] | None = None,
                     known_tools: set[str] | None = None) -> tuple[list[ToolCall], str]:
    """返回 (tool_calls, 去掉工具调用片段后的剩余文本)。"""
    calls: list[ToolCall] = []
    for tc in openai_tool_calls or []:
        fn = tc.get("function", {})
        args = fn.get("arguments") or "{}"
        parsed = _loads(args) if isinstance(args, str) else args
        calls.append(ToolCall(fn.get("name", ""), parsed or {}, tc.get("id") or uuid.uuid4().hex[:8]))
    if calls:
        return calls, content or ""
    text = content or ""
    for m in _ACTION.finditer(text):
        args = _loads(m.group(2))
        if args is not None:
            calls.append(ToolCall(m.group(1), args, uuid.uuid4().hex[:8]))
    if not calls:
        for m in _HERMES.finditer(text):
            obj = _loads(m.group(1))
            if obj and "name" in obj:
                calls.append(ToolCall(obj["name"], obj.get("arguments") or {}, uuid.uuid4().hex[:8]))
    if not calls:
        for m in _BARE.finditer(text):
            args = _loads(m.group(2))
            if args is not None:
                calls.append(ToolCall(m.group(1), args, uuid.uuid4().hex[:8]))
    if known_tools is not None:
        calls = [c for c in calls if c.name in known_tools] + [c for c in calls if c.name not in known_tools]
    rest = _BARE.sub("", _HERMES.sub("", _ACTION.sub("", text)))
    rest = re.sub(r"^\s*[a-z]{2,12}\s*\n", "", rest)  # 去掉泄漏时残留的半截 token（如 ptom / olith）
    for c in calls:  # 去掉模型自己编的占位参数
        c.arguments = {k: v for k, v in c.arguments.items() if v not in ("", "未提及", "无", None)}
    return calls, rest.strip()
