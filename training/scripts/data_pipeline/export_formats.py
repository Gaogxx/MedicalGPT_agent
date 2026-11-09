#!/usr/bin/env python3
"""把 MedicalGPT 的 ShareGPT 训练数据导出成两种格式，保证“训练时模型看到的文本”和“上线后 vLLM 渲染的文本”逐字一致。

  1. OpenAI 格式（openai/）：messages（system / user / assistant.tool_calls / tool）+ tools，
     与线上 DSH → 网关 → vLLM 的请求同构，也能直接给 TRL、LLaMA-Factory 等框架用；
  2. MedicalGPT 预渲染格式（medicalgpt_qwen35/）：仍是 ShareGPT，但工具说明、工具调用、工具结果已按
     runtime/vllm/medicalgpt_qwen35_tools.jinja 的写法渲染成文本，配合
     `--template_name qwen3_5_nothink`（不传 --tool_format）训练。

为什么要预渲染：MedicalGPT v2.7 对 Qwen3.5 的 tool_format 仍是 Qwen2.5 的 JSON 写法
（<tool_call>{"name": ..}</tool_call>），而 Qwen3.5 自带对话模板是 XML 写法（<function=..><parameter=..>）。
直接用 vLLM 默认模板 + qwen3_coder 解析器上线，模型输出和解析器对不上，就是 2026-09 那种工具调用泄漏。
现在训练与推理都用同一种写法（JSON 工具调用 + vLLM hermes 解析器），由同一个 Jinja 模板描述。

用法：
  python export_formats.py                                   # 默认导出 sft_v3 / reward_v3
  python export_formats.py --sft ../../data/sft_v3 --dpo ../../data/reward_v3 --out ../../data/export_v3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# 与 medagent/mcp_servers/medical_server.py 的 SYSTEM_PROMPT 保持一致：训练与线上用同一份系统提示
from medagent.mcp_servers.medical_server import SYSTEM_PROMPT  # noqa: E402

# 与 MedicalGPT training/tool_utils.py 的 QWEN_TOOL_PROMPT 逐字相同（v2.7.0 / main@ef9985a）
QWEN_TOOL_PROMPT = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>{tool_text}"
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
    """<tool_call></tool_call> XML tags:\n<tool_call>\n{{"name": <function-name>, """
    """"arguments": <args-json-object>}}\n</tool_call>"""
)


def _dumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False)


def normalize_tools(tools: Any) -> list[dict[str, Any]]:
    """tools 字段（JSON 字符串或列表）→ [{"name", "description", "parameters"}]，键顺序固定。"""
    if not tools:
        return []
    if isinstance(tools, str):
        tools = json.loads(tools)
    out = []
    for t in tools:
        t = t.get("function", t)
        out.append({"name": t["name"], "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}})})
    return out


def tools_prompt(tools: list[dict[str, Any]]) -> str:
    text = "".join("\n" + _dumps({"type": "function", "function": t}) for t in tools)
    return QWEN_TOOL_PROMPT.format(tool_text=text)


def render_call(name: str, arguments: Any) -> str:
    args = arguments if isinstance(arguments, str) else _dumps(arguments)
    return "<tool_call>\n" + f'{{"name": "{name}", "arguments": {args}}}' + "\n</tool_call>"


def render_response(content: str) -> str:
    return f"<tool_response>\n{content}\n</tool_response>"


def _parse_call(value: str) -> dict[str, Any] | None:
    try:
        d = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    return d if isinstance(d, dict) and "name" in d and "arguments" in d else None


# ---------------------------------------------------------------- ShareGPT → OpenAI
def to_openai_messages(conv: list[dict[str, str]], system: str = SYSTEM_PROMPT) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    sys_text = next((t["value"] for t in conv if t["from"] == "system"), system)
    msgs.append({"role": "system", "content": sys_text})
    pending: list[str] = []
    n = 0
    for t in conv:
        role, value = t["from"], t["value"]
        if role == "system":
            continue
        if role in ("human", "user"):
            msgs.append({"role": "user", "content": value})
        elif role == "function_call":
            fc = json.loads(value)
            cid = f"call_{n}"
            n += 1
            pending.append(cid)
            msgs.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": fc["name"], "arguments": _dumps(fc["arguments"])}}]})
        elif role == "observation":
            msgs.append({"role": "tool", "tool_call_id": pending.pop(0) if pending else f"call_{n}", "content": value})
        elif role in ("gpt", "assistant"):
            msgs.append({"role": "assistant", "content": value})
    return msgs


def answer_to_openai(value: Any) -> dict[str, Any]:
    """DPO 的 chosen / rejected：普通回答或一次工具调用。"""
    if isinstance(value, dict):
        value = _dumps(value)
    fc = _parse_call(value)
    if fc:
        return {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_r", "type": "function", "function": {"name": fc["name"], "arguments": _dumps(fc["arguments"])}}]}
    return {"role": "assistant", "content": value}


# ---------------------------------------------------------------- ShareGPT → MedicalGPT 预渲染
def wrap_system(system: str, tools: list[dict[str, Any]]) -> str:
    """--template_name 指定时 MedicalGPT 原样使用自定义 system（不加 <|im_start|>system 包装），所以这里自己包好。"""
    body = system + (tools_prompt(tools) if tools else "")
    return f"<|im_start|>system\n{body}<|im_end|>"


def to_medicalgpt_conv(conv: list[dict[str, str]], tools: list[dict[str, Any]],
                       system: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    sys_text = next((t["value"] for t in conv if t["from"] == "system"), system)
    out = [{"from": "system", "value": wrap_system(sys_text, tools)}]
    for t in conv:
        role, value = t["from"], t["value"]
        if role == "system":
            continue
        if role == "function_call":
            fc = json.loads(value)
            out.append({"from": "gpt", "value": render_call(fc["name"], fc["arguments"])})
        elif role == "observation":
            out.append({"from": "human", "value": render_response(value)})
        elif role in ("human", "user"):
            out.append({"from": "human", "value": value})
        else:
            out.append({"from": "gpt", "value": value})
    return out


def answer_to_medicalgpt(value: Any) -> str:
    if isinstance(value, dict):
        value = _dumps(value)
    fc = _parse_call(value)
    return render_call(fc["name"], fc["arguments"]) if fc else value


# ---------------------------------------------------------------- 批量导出
def export_sft(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    tools = normalize_tools(row.get("tools"))
    oa = {"messages": to_openai_messages(row["conversations"]), "meta": row.get("meta", {})}
    if tools:
        oa["tools"] = [{"type": "function", "function": t} for t in tools]
    mg = {"conversations": to_medicalgpt_conv(row["conversations"], tools), "meta": row.get("meta", {})}
    return oa, mg


def export_dpo(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    tools = normalize_tools(row.get("tools"))
    oa = {"messages": to_openai_messages(row["conversations"]), "chosen": answer_to_openai(row["chosen"]),
          "rejected": answer_to_openai(row["rejected"]), "meta": row.get("meta", {})}
    if tools:
        oa["tools"] = [{"type": "function", "function": t} for t in tools]
    mg = {"conversations": to_medicalgpt_conv(row["conversations"], tools),
          "chosen": answer_to_medicalgpt(row["chosen"]), "rejected": answer_to_medicalgpt(row["rejected"]),
          "meta": row.get("meta", {})}
    return oa, mg


def _run(src: Path, out: Path, kind: str, fn) -> dict[str, int]:
    stats = {}
    for f in sorted(src.glob("*.jsonl")):
        rows = [json.loads(line) for line in open(f, encoding="utf-8") if line.strip()]
        pairs = [fn(r) for r in rows]
        for sub, idx in (("openai", 0), ("medicalgpt_qwen35", 1)):
            d = out / sub / kind
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f.name, "w", encoding="utf-8") as w:
                for p in pairs:
                    w.write(_dumps(p[idx]) + "\n")
        stats[f"{kind}/{f.name}"] = len(rows)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default=str(REPO / "training" / "data" / "sft_v3"))
    ap.add_argument("--dpo", default=str(REPO / "training" / "data" / "reward_v3"))
    ap.add_argument("--out", default=str(REPO / "training" / "data" / "export_v3"))
    args = ap.parse_args()
    out = Path(args.out)
    stats = {**_run(Path(args.sft), out, "sft", export_sft), **_run(Path(args.dpo), out, "dpo", export_dpo)}
    (out / "manifest.json").write_text(_dumps({"source": {"sft": args.sft, "dpo": args.dpo}, "counts": stats,
                                               "system_prompt": SYSTEM_PROMPT,
                                               "chat_template": "runtime/vllm/medicalgpt_qwen35_tools.jinja"}),
                                       encoding="utf-8")
    print(_dumps(stats))


if __name__ == "__main__":
    main()
