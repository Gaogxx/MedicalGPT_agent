"""DPO 数据（training/data/openai_v3/dpo，入库）的三条硬约束，2026-11-13/14 DPO 体检后加的：

1. 不能有 v0 的 ReAct 写法（"Action: …\\nAction Input: …"）——SFT 与线上都是 <tool_call>；
2. 对话或回答里有工具调用的样本必须带 tools（线上请求总带工具定义）；
3. 验证集不能含训练数据：用户问题不在 DPO 训练集 / SFT 训练集里，回答原文不在 DPO 训练集里。
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training" / "scripts" / "data_pipeline"))

from common import norm  # noqa: E402

DATA = ROOT / "training" / "data" / "openai_v3"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def _user(r: dict) -> str:
    return next((m["content"] for m in r["messages"] if m["role"] == "user"), "")


def _answer(m: dict) -> str:
    return (m.get("content") or "") + json.dumps(m.get("tool_calls") or "", ensure_ascii=False, sort_keys=True)


TRAIN, VAL = _rows(DATA / "dpo" / "train.jsonl"), _rows(DATA / "dpo" / "val.jsonl")


def test_no_legacy_react_calls():
    pat = re.compile(r"Action:\s*\w+\s*\nAction Input:")
    bad = [r for r in TRAIN + VAL
           if any(pat.search(m.get("content") or "") for m in r["messages"] + [r["chosen"], r["rejected"]])]
    assert not bad, f"{len(bad)} 条仍是 ReAct 写法"


def test_tool_rows_have_tools():
    bad = [r for r in TRAIN + VAL
           if any(m.get("tool_calls") for m in r["messages"] + [r["chosen"], r["rejected"]]) and not r.get("tools")]
    assert not bad, f"{len(bad)} 条有工具调用却没带 tools"


def test_val_disjoint_from_training_data():
    sft_q = {norm(_user(r)) for f in ("train.jsonl", "train_consult.jsonl") for r in _rows(DATA / "sft" / f)}
    dpo_q = {norm(_user(r)) for r in TRAIN}
    dpo_a = {_answer(r[k]) for r in TRAIN for k in ("chosen", "rejected")}
    q_leak = [_user(r) for r in VAL if norm(_user(r)) in sft_q | dpo_q]
    a_leak = [_user(r) for r in VAL if _answer(r["chosen"]) in dpo_a or _answer(r["rejected"]) in dpo_a]
    assert not q_leak, f"{len(q_leak)} 条验证问题出现在训练集：{q_leak[:3]}"
    assert not a_leak, f"{len(a_leak)} 条验证回答出现在 DPO 训练集：{a_leak[:3]}"
