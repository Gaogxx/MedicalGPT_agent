#!/usr/bin/env python3
"""“什么时候调用复杂问诊”的训练数据（OpenAI 格式）。

网关不再按规则触发复杂问诊，而是把 complex_consult 放进 local / strong 模型的工具清单，由模型决定要不要调用。
微调 MedicalGPT 训练时没见过这个工具，所以补两类样本：
  - 正例：一次问了多件事（症状原因 + 用药、用药 + 饮食、两种药分别的注意事项……）→ 调用 complex_consult；
  - 反例：只问一件事 → 照常调用普通工具（来自 v3 的工具调用样本），工具清单里同样有 complex_consult，教模型“别乱用”。
工具清单与线上一致：医疗工具 MCP Server 的 7 个工具在前，complex_consult 在最后（网关就是这样拼的）。
正例只到“发出调用”为止：调用之后由网关执行编排器并直接返回答案，模型不需要转述。

  python consult_samples.py          # → training/data/openai_v3/sft/{train,val}_consult.jsonl
"""
from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from medagent.consult import TOOL_NAME, TOOL_SPEC  # noqa: E402
from medagent.knowledge import load_kb  # noqa: E402
from medagent.mcp_servers.medical_server import SYSTEM_PROMPT  # noqa: E402
from medagent.rag.rewrite import QueryRewriter  # noqa: E402

# 训练用 6 种问法，验证集用另外 2 种（按问法分组切分，看模型能否泛化到没见过的说法）
TRAIN_TEMPLATES = [
    "先帮我分析一下{s}可能是什么原因，再说说{d}有什么副作用",
    "{g}一般用什么药，平时饮食要注意什么",
    "{d}和{d2}分别有哪些禁忌，两种能不能一起吃",
    "我最近{s}，想知道可能的原因，另外{d}该怎么吃",
    "{g}的饮食和运动分别有什么建议",
    "先讲讲{d}的适应症，再讲讲{g}患者要注意什么",
]
VAL_TEMPLATES = [
    "{s}是怎么回事？还有{g}在饮食上有什么讲究",
    "请分别介绍一下{d}的用法和{d2}的不良反应",
]


async def medical_tools() -> list[dict]:
    from medagent.mcp_client import MCPToolbox
    from medagent.mcp_servers.medical_server import build_server
    async with MCPToolbox({"medical": build_server()}) as tb:
        return tb.openai_tools()


def positives(templates: list[str], n: int, rng: random.Random, dc, tools: list[dict], split: str) -> list[dict]:
    kb, rw = load_kb(), QueryRewriter()
    syms, drugs, dis = list(kb.symptoms), list(kb.drugs), list(kb.guidelines)
    seen, out, tries = set(), [], 0
    while len(out) < n and tries < n * 50:
        tries += 1
        t = rng.choice(templates)
        d, d2 = rng.sample(drugs, 2)
        q = t.format(s=rng.choice(syms), d=d, d2=d2, g=rng.choice(dis))
        if q in seen or dc.bad(q) or len(rw.rewrite(q).sub_queries) < 2:   # 必须真能拆成多个子问题
            continue
        seen.add(q)
        out.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_0", "type": "function",
                 "function": {"name": TOOL_NAME, "arguments": json.dumps({"question": q}, ensure_ascii=False)}}]}],
            "tools": tools + [TOOL_SPEC], "meta": {"bucket": "consult_call", "split": split, "template": templates.index(t)}})
    return out


def negatives(src: Path, n: int, rng: random.Random, tools: list[dict], split: str) -> list[dict]:
    from medagent.verify.detectors import multi_intent
    rows = [json.loads(line) for line in open(src, encoding="utf-8")]
    pool = [r for r in rows if r.get("meta", {}).get("bucket") in ("toolcall", "toolcall_ext")
            and not multi_intent(next(m["content"] for m in r["messages"] if m["role"] == "user"))]
    rng.shuffle(pool)
    return [{**r, "tools": tools + [TOOL_SPEC], "meta": {**r.get("meta", {}), "bucket": "consult_negative", "split": split}}
            for r in pool[:n]]


def main() -> None:
    from supplement import Decontam, eval_texts
    rng = random.Random(42)
    dc = Decontam(eval_texts())
    tools = asyncio.run(medical_tools())
    out = REPO / "training" / "data" / "openai_v3" / "sft"
    train = positives(TRAIN_TEMPLATES, 400, rng, dc, tools, "train") + negatives(out / "train.jsonl", 400, rng, tools, "train")
    val = positives(VAL_TEMPLATES, 40, rng, dc, tools, "val") + negatives(out / "val.jsonl", 40, rng, tools, "val")
    for name, rows in (("train_consult.jsonl", train), ("val_consult.jsonl", val)):
        rng.shuffle(rows)
        with open(out / name, "w", encoding="utf-8") as w:
            for r in rows:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
    stat = {name: {b: sum(r["meta"]["bucket"] == b for r in rows) for b in ("consult_call", "consult_negative")}
            for name, rows in (("train", train), ("val", val))}
    print(json.dumps({**stat, "decontam_hits": dc.hits}, ensure_ascii=False))


if __name__ == "__main__":
    main()
