"""训练格式 ⇄ 推理格式一致性：OpenAI 格式的消息经 vLLM 对话模板渲染后，必须与 MedicalGPT 训练时拼出的文本逐字相同。

MedicalGPT 侧按 v2.7（main@ef9985a）training/supervised_finetuning.py 的 get_dialog +
template.py 的 qwen3_5_nothink 模板复现：每段 target 后追加 eos（Qwen3.5 的 eos = <|im_end|>）。
"""
import json
import sys
from pathlib import Path

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training" / "scripts" / "data_pipeline"))

import export_formats as ef  # noqa: E402

from medagent.mcp_servers.medical_server import SYSTEM_PROMPT  # noqa: E402

EOS = "<|im_end|>"
TEMPLATE = (ROOT / "runtime" / "vllm" / "medicalgpt_qwen35_tools.jinja").read_text(encoding="utf-8")


def render(messages, tools=None, add_generation_prompt=False) -> str:
    # 与 transformers / vLLM 渲染对话模板的环境一致：trim_blocks + lstrip_blocks，tojson 不转义中文
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda x, indent=None, separators=None, sort_keys=False: json.dumps(
        x, ensure_ascii=False, indent=indent, separators=separators, sort_keys=sort_keys)
    return env.from_string(TEMPLATE).render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt)


def medicalgpt_dialog(conv):
    """复现 MedicalGPT get_dialog（template_name=qwen3_5_nothink，无 tools 字段）。"""
    system_prompt, messages = "", []
    for s in conv:
        if s["from"] == "system":
            system_prompt = s["value"]
        elif s["from"] in ("human", "user", "observation"):
            messages.append({"role": "user", "content": s["value"]})
        else:
            messages.append({"role": "assistant", "content": s["value"]})
    history, temp = [], []
    for m in messages:
        if not temp and m["role"] == "user":
            temp.append(m["content"])
        elif len(temp) == 1 and m["role"] == "assistant":
            temp.append(m["content"])
            history.append(temp)
            temp = []
        elif m["role"] == "user" and len(temp) == 1:
            temp[0] += "\n" + m["content"]
    sep, prompt = "\n", "<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    system_prompt = (system_prompt or "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n") + sep
    convs = []
    for i, (q, a) in enumerate(history):
        convs.append((system_prompt if i == 0 else sep) + prompt.format(query=q))
        convs.append(a)
    return convs, temp


def medicalgpt_text(conv) -> str:
    convs, _ = medicalgpt_dialog(conv)
    return "".join(convs[i] + (EOS if i % 2 else "") for i in range(len(convs)))


def rows(path: Path, limit: int | None = None):
    if not path.exists():
        pytest.skip(f"{path} 不存在")
    out = [json.loads(line) for line in open(path, encoding="utf-8")]
    return out[:limit] if limit else out


def test_template_default_system_matches_training():
    assert render([{"role": "user", "content": "你好"}]).startswith(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>")


def test_sft_openai_render_equals_medicalgpt_training_text():
    data = rows(ROOT / "training" / "data" / "sft_v3" / "val.jsonl")
    with_tools = 0
    for r in data:
        oa, mg = ef.export_sft(r)
        with_tools += bool(oa.get("tools"))
        assert render(oa["messages"], oa.get("tools")) == medicalgpt_text(mg["conversations"]), r["conversations"][0]
    assert with_tools > 50


def test_dpo_prompt_and_answers_match():
    data = rows(ROOT / "training" / "data" / "reward_v3" / "val.jsonl")
    for r in data:
        oa, mg = ef.export_dpo(r)
        convs, pending = medicalgpt_dialog(mg["conversations"])
        # DPO 的 prompt = 历史 + 最后一个未回答的 user 块 + 生成提示
        tail = pending[0] if pending else ""
        sep, prompt = "\n", "<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        expect = "".join(convs[i] + (EOS if i % 2 else "") for i in range(len(convs)))
        expect += ((mg["conversations"][0]["value"] + sep) if not convs else sep) + prompt.format(query=tail)
        assert render(oa["messages"], oa.get("tools"), add_generation_prompt=True) == expect
        full = render(oa["messages"] + [oa["chosen"]], oa.get("tools"))
        assert full == expect + mg["chosen"] + EOS


def test_tool_call_roundtrip_with_dict_or_string_arguments():
    tools = [{"type": "function", "function": {"name": "search_drug", "description": "查药",
                                               "parameters": {"type": "object", "properties": {"name": {"type": "string"}}}}}]
    base = [{"role": "system", "content": "S"}, {"role": "user", "content": "布洛芬"}]
    as_str = render(base + [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": "search_drug", "arguments": '{"name": "布洛芬"}'}}]}], tools)
    as_dict = render(base + [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": "search_drug", "arguments": {"name": "布洛芬"}}}]}], tools)
    assert as_str == as_dict
    assert '<tool_call>\n{"name": "search_drug", "arguments": {"name": "布洛芬"}}\n</tool_call><|im_end|>' in as_str
    assert ef.tools_prompt(ef.normalize_tools([t["function"] for t in tools])) in as_str


# ---------------------------------------------------------------- OpenAI 格式为准：训练实际走 OpenAI → MedicalGPT 输入
def test_openai_canonical_data_renders_like_training_text():
    files = [("sft", "val.jsonl"), ("sft", "val_consult.jsonl"), ("dpo", "val.jsonl")]
    for kind, name in files:
        for r in rows(ROOT / "training" / "data" / "openai_v3" / kind / name):
            mg = ef.openai_row_to_medicalgpt(r)
            msgs = r["messages"] + ([r["chosen"]] if kind == "dpo" else [])
            conv = mg["conversations"] + ([{"from": "gpt", "value": mg["chosen"]}] if kind == "dpo" else [])
            assert render(msgs, r.get("tools")) == medicalgpt_text(conv)


def test_openai_path_equals_sharegpt_path():
    for kind, fn in (("sft", ef.export_sft), ("dpo", ef.export_dpo)):
        src = "sft_v3" if kind == "sft" else "reward_v3"
        for r in rows(ROOT / "training" / "data" / src / "val.jsonl"):
            oa, mg = fn(r)
            again = ef.openai_row_to_medicalgpt(oa)
            assert again["conversations"] == mg["conversations"]
            if kind == "dpo":
                assert (again["chosen"], again["rejected"]) == (mg["chosen"], mg["rejected"])
