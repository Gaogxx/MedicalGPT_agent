#!/usr/bin/env python3
"""用现在的最终 9B 跑完整编排（真实 MCP 工具 + 药品库 / Wiki），记下每道题“最后一次作答”那次模型调用的完整输入
（system、用户问题、模型发出的工具调用、工具返回）和模型自己的回答。下一步由 27B 按同样的输入重写最终回答。

只记模型调用的原始输入，所以训练样本和线上推理时模型看到的内容逐字相同（同一个对话模板渲染）。

  python training/qwen35/grounding/g2_collect.py --queries training/data/grounding/queries.jsonl \
      --out training/data/grounding/trajectories.jsonl --conc 24
"""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from run_live_eval import LocalOnly  # noqa: E402

from medagent.agent import MedicalAgent  # noqa: E402
from medagent.llm.pool import ModelPool  # noqa: E402
from medagent.llm.providers import OpenAICompatProvider  # noqa: E402
from medagent.mcp_client import MCPToolbox  # noqa: E402
from medagent.mcp_servers.medical_server import build_server  # noqa: E402
from medagent.resilience.checkpoint import CheckpointStore  # noqa: E402

CALLS: contextvars.ContextVar[list] = contextvars.ContextVar("calls")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="medical-agent")
    ap.add_argument("--conc", type=int, default=24)
    args = ap.parse_args()
    provider = OpenAICompatProvider("local-collect", args.base_url, args.model, api_key_env="VLLM_API_KEY", timeout=180)
    chat = provider.chat

    async def recording(messages, tools=None, temperature=0.3, max_tokens=1024):
        resp = await chat(messages, tools, 0.0, max_tokens)
        rec = CALLS.get(None)
        if rec is not None:
            rec.append({"messages": copy.deepcopy(messages), "tools": tools, "content": resp.content,
                        "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in resp.tool_calls],
                        "finish": resp.finish_reason})
        return resp
    provider.chat = recording
    pool = ModelPool({"local": [provider]})
    qs = [json.loads(x) for x in open(args.queries, encoding="utf-8")]
    done = set()
    if Path(args.out).exists():
        done = {json.loads(x)["id"] for x in open(args.out, encoding="utf-8")}
    sem = asyncio.Semaphore(args.conc)
    fo = open(args.out, "a", encoding="utf-8")
    async with MCPToolbox({"medical": build_server()}) as tb:
        async def one(q: dict) -> None:
            async with sem:
                CALLS.set([])
                calls = CALLS.get()
                try:
                    r = await MedicalAgent(pool, tb, LocalOnly(), CheckpointStore()).run(q["query"])
                except Exception as exc:  # noqa: BLE001
                    print("ERR", q["id"], exc, flush=True)
                    return
                final = next((c for c in reversed(calls) if not c["tool_calls"] and c["content"].strip()
                              and any(m.get("role") == "tool" for m in c["messages"])), None)
                if final is None and q["kind"] == "retain":
                    final = next((c for c in reversed(calls) if not c["tool_calls"] and c["content"].strip()), None)
                fo.write(json.dumps({**q, "final_call": final, "agent_answer": r["answer"], "repairs": r["repairs"],
                                     "n_calls": len(calls)}, ensure_ascii=False) + "\n")
                fo.flush()
        await asyncio.gather(*(one(q) for q in qs if q["id"] not in done))
    fo.close()


if __name__ == "__main__":
    asyncio.run(main())
