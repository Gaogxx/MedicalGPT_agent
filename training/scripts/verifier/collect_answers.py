#!/usr/bin/env python3
"""判别模型训练数据第 1 步：让某个模型在路由数据集 train / val 的问题上走完整编排作答，记下“校验之前的那版回答”
（线上判别模型看到的就是它）和当时的程序状态，再请裁判对照参考答案打质量等级。

为什么用基座模型（Qwen3.5-2B / Qwen3.5-9B 未微调）而不是等 9B 微调完：判别模型要先训好，而且基座模型的回答
好坏都有，失败样本比微调后的模型多，判别模型更容易学到“答砸了长什么样”。微调完的 9B 也可以再跑一遍补充数据。
路由数据集的 test 部分不用：它留给路由器报告效果；dev 80 也不用：它是判别模型的外部测试。

  python training/scripts/verifier/collect_answers.py --base-url http://127.0.0.1:8011/v1 --model Qwen/Qwen3.5-2B \
      --tag base2b --judge strong --concurrency 8
  python training/scripts/verifier/collect_answers.py --mock --judge rules --limit 10 --tag smoke     # 联调
输出：training/data/verifier/answers_<tag>.jsonl（每行一题：状态 state、回答、裁判等级、理由），中断后重跑会接着做
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "evaluation" / "router"))

OUT_DIR = ROOT / "training" / "data" / "verifier"


async def main() -> None:
    from label_router_set import add_judge_args, judge_pool, judge_strong_typed
    from run_live_eval import LocalOnly

    from medagent.agent import GUARD_FIXABLE, MedicalAgent
    from medagent.llm.pool import ModelPool
    from medagent.llm.providers import MockProvider, OpenAICompatProvider
    from medagent.mcp_client import MCPToolbox
    from medagent.mcp_servers.medical_server import build_server
    from medagent.resilience.checkpoint import CheckpointStore
    from medagent.safety import SafetyAssessment
    from medagent.verify.typed import TypedVerifier, answer_state

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", required=True)
    add_judge_args(ap)
    args = ap.parse_args()
    splits = set(args.splits.split(","))
    if "test" in splits:
        raise SystemExit("路由数据集的 test 部分留给路由器报告效果，不能拿来训练判别模型")

    rows = [json.loads(line) for line in open(ROOT / "evaluation" / "datasets" / "router_set.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["split"] in splits]
    rows = rows[: args.limit] if args.limit else rows
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"answers_{args.tag}.jsonl"
    done = {json.loads(line)["id"] for line in open(out, encoding="utf-8")} if out.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    # 基座模型用 Qwen3.5 自带模板 + qwen3_coder 解析器部署，关掉思考（和线上小模型档一致，也更快）
    local = MockProvider("mock") if args.mock else OpenAICompatProvider(
        f"collect-{args.tag}", args.base_url, args.model, api_key_env="VLLM_API_KEY", timeout=180,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    jpool = judge_pool(args) if args.judge == "strong" else None
    drafts: dict[str, dict] = {}

    class Recording(MedicalAgent):
        """在校验节点之前记下回答与状态：这正是线上判别模型的输入。"""

        async def _node_verify(self, s):
            # 与 _node_verify 交给判别模型的一致：去掉输出安全关能确定性修好的那几类
            findings = [f for f in self._detect(s, s["answer"]) if f["code"] not in GUARD_FIXABLE]
            drafts[s["run_id"]] = answer_state(s["query"], s["answer"], [o for _, o in self._observations(s)], findings,
                                               SafetyAssessment(**s["assessment"]))
            drafts[s["run_id"]]["_rule_block"] = [f["code"] for f in findings if f["severity"] == "block"]
            return await super()._node_verify(s)

    sem, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()
    print(f"[{args.tag}] 待作答 {len(todo)} 题（已完成 {len(done)}），裁判：{args.judge}", flush=True)
    async with MCPToolbox({"medical": build_server(consult=None)}) as tb:
        agent = Recording(ModelPool({"local": [local]}), tb, LocalOnly(), CheckpointStore(), verifier=TypedVerifier(None))

        async def one(r: dict) -> None:
            async with sem:
                t0 = time.perf_counter()
                rid = f"vd-{args.tag}-{r['id']}"
                try:
                    await agent.run(r["query"], run_id=rid)
                except Exception as exc:  # noqa: BLE001
                    print(r["id"], "作答失败", exc, flush=True)
                    return
                state = drafts.pop(rid, None)
                if state is None:          # 急症快路径等没有经过校验节点的，判别模型线上也看不到
                    return
                rule_block = state.pop("_rule_block")
                level, reason, ftype = "skipped", "", "unknown"
                if args.judge == "strong":
                    try:
                        level, reason, ftype = await judge_strong_typed(jpool, r["query"], r["reference"], state["answer"])
                    except Exception as exc:  # noqa: BLE001
                        level, reason = "judge_error", str(exc)[:200]
                rec = {"id": r["id"], "split": r["split"], "source": r["source"], "model": args.tag, "state": state,
                       "rule_block": rule_block, "judge": args.judge, "level": level, "failure_type": ftype, "reason": reason,
                       "seconds": round(time.perf_counter() - t0, 2)}
                async with lock:
                    with open(out, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        await asyncio.gather(*(one(r) for r in todo))
    recs = [json.loads(line) for line in open(out, encoding="utf-8")] if out.exists() else []
    from collections import Counter
    print(json.dumps({"n": len(recs), "levels": Counter(r["level"] for r in recs),
                      "rule_blocked": sum(bool(r["rule_block"]) for r in recs), "out": str(out.relative_to(ROOT))},
                     ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
