#!/usr/bin/env python3
"""27B 按同一份工具结果重写最终回答 → 确定性核对 → SFT 样本（2026-11-20）。

为什么：评测里 9B 查到了也常答得含糊（药品说明书题覆盖率 0.196）、某病用药约三成药名不是工具给的、
相互作用要分清“说明书没提到”和“没有相互作用”。这些都是“按工具结果作答”的问题，不是调不调工具的问题。

核对（不过就让 27B 带着问题重写一次，再不过就退回模型自己的回答——它也得过核对——都不过就丢掉）：
  剂量数字    回答里的每个剂量 / 频次数字都要在工具返回或问题里出现
  药名        回答点到的每个药都要在工具返回或问题里出现（不能凭记忆加药）
  相互作用    工具给了说明书原文时不能说“可以放心 / 没有相互作用”；说明书都没提到时要说“没提到、不代表绝对安全”
  查不到      工具查不到时要如实说查不到
  格式        不带工具调用片段，40～700 字
“保持原有行为”那一类直接用模型自己的回答（过格式和剂量核对），不让 27B 改，免得把别的能力带偏。

  python training/qwen35/grounding/g3_teacher.py --traj training/data/grounding/trajectories.jsonl \
      --out training/data/grounding/sft.jsonl --llm http://127.0.0.1:8100/v1 --model compiler
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from score_drug_label_eval import DOSE  # noqa: E402
from score_wiki_eval import SAFE, drug_mentions, names_of  # noqa: E402

from medagent.knowledge.drug_label_kb import LLM  # noqa: E402
from medagent.knowledge.ext import load_ext  # noqa: E402

RULES = """你是医疗助手的示范答题者。下面是用户问题、助手已经查到的资料（工具调用和返回），请直接写给用户的最终回答。要求：
1. 只依据查到的资料作答：药名、剂量、频次、人群、后果都要出自资料，不能凭记忆补充任何药名或数字。
2. 只回答用户问的那一项（问副作用就说副作用，问用法就说用法）；资料里有的关键信息要具体写出来（剂量和频次、主要禁忌、常见不良反应、相互作用的后果和处理），不要只说“请遵医嘱”；不要罗列资料里缺了哪些别的项目。
3. 对用户说“说明书写明 / 说明书显示”，不要提“工具”“知识库”“检索”“查询结果”这类词。
4. 剂型：资料明确说“没有某剂型的用法用量”时，如实告诉用户没有该剂型的资料，这时不要给出任何剂量数字；没有这句话时，同一个药的片剂 / 胶囊 / 颗粒 / 口服液等可以直接按资料回答，不要说“没找到该剂型”。
5. 相互作用：资料给了说明书原文就说清后果和建议；资料说说明书没提到时，要说“说明书没有提到两者合用的问题，但这不代表绝对安全”，不能说没有相互作用或可以放心合用。
6. 问某病用什么药：只列资料里的药，说明这些是说明书适应症写明的药、不是治疗推荐，具体选药由医生决定。
7. 资料显示查不到这个药时如实说没有查到，建议看药品说明书或问药师，不编造。
8. 只有用户提到孕妇、哺乳、儿童、老人、肝肾功能不全或正在合用其他药时，才提醒需要医生或药师评估；用户没提就不要写这类提醒。
9. 中文，简洁，不超过 250 字，不用标题和表格；最后一句写“以上信息仅供参考，不能替代专业医疗建议。”只输出回答正文。"""
NOT_FOUND = re.compile(r"查不到|未找到|没有查到|没有找到|暂无|没有.{0,6}(资料|信息|记录)")
NONE_OK = re.compile(r"(没有|未|没)提(到|及)")
_COMMON = {"1", "2", "3", "4", "5", "6", "7", "8", "12", "24", "120"}     # 次数、小时、急救电话这类不算剂量


def obs_text(call: dict) -> str:
    return "\n".join(str(m.get("content", "")) for m in call["messages"] if m.get("role") == "tool")


def lint(ext, kind: str, query: str, obs: str, ans: str) -> str | None:
    a = ans.strip()
    if not 40 <= len(a) <= 700:
        return f"长度 {len(a)}"
    if "<tool" in a or re.search(r"\{\s*\"", a):
        return "带工具调用片段"
    allowed = set(re.findall(r"\d+(?:\.\d+)?", obs + query)) | _COMMON
    for m in DOSE.findall(a):
        for n in re.findall(r"\d+(?:\.\d+)?", m):
            if n not in allowed:
                return f"剂量数字 {n} 不在工具返回里"
    if kind != "retain" and re.search(r"工具|知识库|检索|查询结果", a):
        return "提到了工具 / 知识库"
    if kind != "retain" and not re.search(r"孕|哺乳|儿童|孩子|宝宝|小孩|老人|老年|肝|肾|在吃|合用|同时", query) \
            and re.search(r"(孕妇|哺乳|儿童|老人|肝肾功能不全)[^。]{0,20}(评估|咨询|遵医嘱|在医生)", a):
        return "用户没提特殊人群却写了特殊人群提醒"
    missing_form = re.search(r"没有.{0,6}剂型的用法用量", obs)
    if kind == "label" and missing_form and DOSE.findall(a):
        return "所问剂型没有资料时给了剂量"
    if kind == "label" and not missing_form and "found\": false" not in obs.replace(" ", "") \
            and re.search(r"(未|没有?)(找到|查到|收录)", a):
        return "资料里有这个药却说没找到"
    if kind != "retain":
        src = obs + query
        for q, _ in drug_mentions(ext, a):
            if not any(nm in src for nm in names_of(ext, q)):
                return f"药名「{ext.drugs[q]['zh']}」不在工具返回里"
    if kind == "interaction" and SAFE.search(a):
        return "说明书给了相互作用却说可以放心"
    if kind == "interaction_none" and (SAFE.search(a) or not NONE_OK.search(a)):
        return "说明书没提到时要说没提到、不代表安全"
    if kind == "not_found" and "found\": false" in obs.replace(" ", "") and not NOT_FOUND.search(a):
        return "查不到时没有如实说"
    return None


def transcript(call: dict) -> str:
    parts = []
    for m in call["messages"]:
        r = m.get("role")
        if r == "user":
            parts.append(f"【用户】{m['content']}")
        elif r == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", tc)
                parts.append(f"【助手调用工具】{fn.get('name')} {fn.get('arguments')}")
        elif r == "tool":
            parts.append(f"【工具返回】{str(m.get('content'))[:3500]}")
    return "\n".join(parts)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--model", default="compiler")
    args = ap.parse_args()
    ext = load_ext()
    llm = LLM(args.llm, args.model, 48)
    rows = [json.loads(x) for x in open(args.traj, encoding="utf-8")]
    stats: Counter = Counter()
    out = []

    async def one(r: dict) -> None:
        call = r.get("final_call")
        if not call:
            stats["no_final_call"] += 1
            return
        obs = obs_text(call)
        student = call["content"].strip()
        if r["kind"] == "retain":
            bad = lint(ext, "retain", r["query"], obs, student)
            stats["retain_ok" if not bad else "retain_drop"] += 1
            if not bad:
                out.append({**r, "answer": student, "source": "self"})
            return
        feedback = ""
        for attempt in range(2):
            ans = (await llm(RULES + feedback + "\n\n" + transcript(call), 700)).strip()
            bad = lint(ext, r["kind"], r["query"], obs, ans)
            if not bad:
                stats[f"teacher_ok_{r['kind']}"] += 1
                out.append({**r, "answer": ans, "source": "teacher", "student": student})
                return
            stats[f"teacher_retry_{bad.split(' ')[0][:6]}"] += 1
            feedback = f"\n（上一稿的问题：{bad}。请改正。）"
        if not lint(ext, r["kind"], r["query"], obs, student):
            stats[f"fallback_self_{r['kind']}"] += 1
            out.append({**r, "answer": student, "source": "self"})
        else:
            stats[f"drop_{r['kind']}"] += 1

    await asyncio.gather(*(one(r) for r in rows))
    with open(args.out, "w", encoding="utf-8") as f:
        for r in sorted(out, key=lambda x: x["id"]):
            msgs = r["final_call"]["messages"] + [{"role": "assistant", "content": r["answer"]}]
            f.write(json.dumps({"id": r["id"], "kind": r["kind"], "source": r["source"], "messages": msgs,
                                "tools": r["final_call"]["tools"]}, ensure_ascii=False) + "\n")
    print(json.dumps({"samples": len(out), **dict(sorted(stats.items()))}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
