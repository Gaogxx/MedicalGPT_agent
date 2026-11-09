#!/usr/bin/env python3
"""训练数据格式：以 OpenAI 格式为准，MedicalGPT 训练用的输入由它生成。

  ShareGPT v3（mix_v3.py 产出）
      └─▶ OpenAI 格式  training/data/openai_v3/{sft,dpo}/*.jsonl        ← 入库，训练数据的唯一标准格式
            messages（system / user / assistant.tool_calls / tool）+ tools，与线上 DSH → 网关 → vLLM 的请求同构，
            TRL、LLaMA-Factory 等框架也能直接读
            └─▶ MedicalGPT 输入  training/data/export_v3/medicalgpt_qwen35/{sft,dpo,opd}/   ← 不入库，几秒生成
                  仍是 MedicalGPT 读取的 conversations 结构，但工具说明 / 调用 / 结果已按
                  runtime/vllm/medicalgpt_qwen35_tools.jinja 的写法渲染成文本，配合 --template_name qwen3_5_nothink 训练

为什么 MedicalGPT 的输入要预渲染：MedicalGPT 读取的是 conversations 结构，且它对 Qwen3.5 的 tool_format 仍是 Qwen2.5 的
JSON 写法（<tool_call>{"name": ..}</tool_call>），而 Qwen3.5 自带模板是 XML 写法。预渲染 + 同一份 vLLM 模板，
保证训练文本与上线渲染文本逐字一致（tests/test_chat_template.py）。

用法：
  python export_formats.py                          # ShareGPT v3 → OpenAI（入库）→ MedicalGPT 输入
  python export_formats.py --from-openai DIR        # 已有 OpenAI 格式数据（自己补的数据也写成这个格式）→ MedicalGPT 输入
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


# ---------------------------------------------------------------- OpenAI → MedicalGPT 输入（训练时实际走这条）
def _assistant_text(m: dict[str, Any]) -> str:
    parts = [m["content"]] if m.get("content") else []
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function", tc)
        parts.append(render_call(fn["name"], fn["arguments"]))
    return "\n".join(parts)


def openai_to_medicalgpt_conv(messages: list[dict[str, Any]], tools: Any = None) -> list[dict[str, str]]:
    sys_text = messages[0]["content"] if messages and messages[0]["role"] == "system" else SYSTEM_PROMPT
    out = [{"from": "system", "value": wrap_system(sys_text, normalize_tools(tools))}]
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "user":
            out.append({"from": "human", "value": m["content"]})
        elif m["role"] == "tool":
            out.append({"from": "human", "value": render_response(m["content"])})
        elif m["role"] == "assistant":
            out.append({"from": "gpt", "value": _assistant_text(m)})
    return out


def openai_row_to_medicalgpt(row: dict[str, Any]) -> dict[str, Any]:
    out = {"conversations": openai_to_medicalgpt_conv(row["messages"], row.get("tools")), "meta": row.get("meta", {})}
    if "chosen" in row:
        out["chosen"], out["rejected"] = _assistant_text(row["chosen"]), _assistant_text(row["rejected"])
    return out


# OPD 只蒸馏问答类样本：工具调用轨迹的格式与工具结果用法由 SFT / DPO 学，不交给没见过本项目工具约定的通用教师
OPD_BUCKETS = ("real_qa", "safety", "clarify", "raft")


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


def _read(f: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(f, encoding="utf-8") if line.strip()]


def _write(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as w:
        for r in rows:
            w.write(_dumps(r) + "\n")
    return len(rows)


def sharegpt_to_openai_dir(sft: Path, dpo: Path, out: Path) -> dict[str, int]:
    stats = {}
    for kind, src, fn in (("sft", sft, export_sft), ("dpo", dpo, export_dpo)):
        for f in sorted(src.glob("*.jsonl")):
            stats[f"openai/{kind}/{f.name}"] = _write(out / kind / f.name, [fn(r)[0] for r in _read(f)])
    return stats


def openai_to_medicalgpt_dir(src: Path, out: Path, opd_train: int = 2000, opd_val: int = 100, seed: int = 42) -> dict[str, int]:
    import random
    stats = {}
    for kind in ("sft", "dpo"):
        for f in sorted((src / kind).glob("*.jsonl")):
            rows = [openai_row_to_medicalgpt(r) for r in _read(f)]
            stats[f"medicalgpt/{kind}/{f.name}"] = _write(out / kind / f.name, rows)
            if kind == "sft":
                # 只取单轮（system + 一问一答）：MedicalGPT 的 OPD 拼多轮 prompt 时轮间不加 eos，和 SFT 的拼法不同
                pool = [r for r in rows if r["meta"].get("bucket") in OPD_BUCKETS and len(r["conversations"]) == 3]
                random.Random(seed).shuffle(pool)
                n = opd_train if f.stem == "train" else opd_val
                stats[f"medicalgpt/opd/{f.name}"] = _write(out / "opd" / f.name, pool[:n])
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default=str(REPO / "training" / "data" / "sft_v3"))
    ap.add_argument("--dpo", default=str(REPO / "training" / "data" / "reward_v3"))
    ap.add_argument("--openai", default=str(REPO / "training" / "data" / "openai_v3"), help="OpenAI 格式数据目录（入库）")
    ap.add_argument("--out", default=str(REPO / "training" / "data" / "export_v3" / "medicalgpt_qwen35"))
    ap.add_argument("--from-openai", help="只做 OpenAI → MedicalGPT 输入，读这个目录（含 sft/ dpo/）")
    ap.add_argument("--opd-train", type=int, default=2000)
    args = ap.parse_args()
    stats: dict[str, int] = {}
    src = Path(args.from_openai) if args.from_openai else Path(args.openai)
    if not args.from_openai:
        stats.update(sharegpt_to_openai_dir(Path(args.sft), Path(args.dpo), src))
    stats.update(openai_to_medicalgpt_dir(src, Path(args.out), opd_train=args.opd_train))
    (Path(args.out) / "manifest.json").write_text(_dumps({"openai_source": str(src), "counts": stats,
                                                          "opd_buckets": OPD_BUCKETS, "system_prompt": SYSTEM_PROMPT,
                                                          "chat_template": "runtime/vllm/medicalgpt_qwen35_tools.jinja"}),
                                                  encoding="utf-8")
    print(_dumps(stats))


if __name__ == "__main__":
    main()
