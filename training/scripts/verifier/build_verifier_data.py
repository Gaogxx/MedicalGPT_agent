#!/usr/bin/env python3
"""判别模型训练数据第 2 步：真实回答 + 裁判标签 + 语义故障注入 → OpenAI 格式（为准）→ MedicalGPT 输入。

样本 = build_messages(state) + 一个字母（A good / B acceptable / C poor / D unsafe）。
  1. 真实样本：collect_answers.py 的输出，标签 = 裁判等级；裁判出错 / 解析失败的丢掉。
     规则已经拦下的（rule_block 非空）线上不会交给判别模型，但仍保留少量（≤10%）作为“明显失败”的锚点。
  2. 故障注入（只进 train，不进 val）：从裁判判为 good / acceptable 的回答出发，做规则抓不到的语义改动，
     改完再跑一遍规则检测，被规则拦下的丢掉（线上判别模型看不到这类样本）：
       off_topic      换成另一道题的回答                          → C
       half           在句号处截掉后半                            → C（只用于多句回答）
       dose_x4        剂量数字放大 4 倍                           → D
       antibiotic     追加“可以自行服用阿莫西林 / 头孢”           → D
       prescribe      追加具体用药 + “不需要去医院”               → D
       diagnose       开头替医生下诊断“可以确诊为…”               → D
       drop_emergency 急症题删掉 120 / 急诊 / 立即就医的句子      → D
     注入样本数不超过真实 train 样本的 35%。
  val 只用真实样本：温度缩放和阈值要在真实分布上校准。

  python training/scripts/verifier/build_verifier_data.py            # 读 training/data/verifier/answers_*.jsonl
输出：training/data/verifier/openai/{train,val}.jsonl（OpenAI 格式，meta 里带 state 和等级）
      training/data/verifier/medicalgpt/{train,val}/data.jsonl（MedicalGPT 的 conversations 输入）
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training" / "scripts" / "data_pipeline"))

from medagent.safety import assess  # noqa: E402
from medagent.verify.detectors import detect  # noqa: E402
from medagent.verify.discriminator import LETTER, build_messages  # noqa: E402

D = ROOT / "training" / "data" / "verifier"
LEVELS = ("good", "acceptable", "poor", "unsafe")
GUARD_FIXABLE = {"emergency_missing", "prescription_in_high_risk"}
_DOSE = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|毫克|g|克|片|粒|袋|ml|毫升)")
_EMERG = re.compile(r"[^。！!\n]*(120|急诊|立即就医|马上就医|尽快就医|立即就诊)[^。！!\n]*[。！!]?")
_SENT = re.compile(r"(?<=[。！？!?])")


def _findings(state: dict, answer: str) -> list:
    return [f for f in detect(answer, query=state["query"], assessment=assess(state["query"]),
                              observations=[{"found": t.get("found")} for t in state.get("tool_results", [])] or None)
            if f.code not in GUARD_FIXABLE]


def _rules_block(state: dict, answer: str) -> bool:
    return any(f.severity == "block" for f in _findings(state, answer))


def _with_answer(state: dict, answer: str) -> dict:
    return {**state, "answer": answer[:1500], "rule_findings": [f.code for f in _findings(state, answer)]}


def perturb(rec: dict, others: list[dict], rng: random.Random) -> tuple[str, str, str] | None:
    """返回（注入类型, 改后的回答, 等级）；这条回答不适合任何注入时返回 None。"""
    st, ans = rec["state"], rec["state"]["answer"]
    kinds = ["off_topic", "antibiotic", "prescribe", "diagnose"]
    if _DOSE.search(ans):
        kinds += ["dose_x4", "dose_x4"]
    if st["safety"]["emergency"] and _EMERG.search(ans):
        kinds += ["drop_emergency"] * 3
    if len(_SENT.split(ans)) >= 4:
        kinds.append("half")
    k = rng.choice(kinds)
    if k == "off_topic":
        o = rng.choice(others)["state"]["answer"]
        return (k, o, "poor") if o != ans else None
    if k == "half":
        sents = [x for x in _SENT.split(ans) if x]
        return k, "".join(sents[: max(1, len(sents) // 2)]), "poor"
    if k == "dose_x4":
        return k, _DOSE.sub(lambda m: f"{float(m.group(1)) * 4:g}{m.group(2)}", ans, count=2), "unsafe"
    if k == "antibiotic":
        return k, ans.rstrip() + "如果几天还没好，可以自行服用阿莫西林或头孢类抗生素。", "unsafe"
    if k == "prescribe":
        return k, ans.rstrip() + "建议直接服用布洛芬，每次2片，每天3次，连续一周，不需要去医院。", "unsafe"
    if k == "diagnose":
        return k, "根据您的描述，可以确诊为" + rng.choice(["胃炎", "偏头痛", "支气管炎", "颈椎病", "焦虑症"]) + "。" + ans, "unsafe"
    if k == "drop_emergency":
        return k, _EMERG.sub("", ans).strip() or "注意休息，多喝水。", "unsafe"
    return None


def row(state: dict, level: str, meta: dict) -> dict:
    return {"messages": build_messages(state) + [{"role": "assistant", "content": LETTER[level]}],
            "meta": {**meta, "level": level, "state": state}}


def main() -> None:
    import export_formats as ef

    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", help="默认 training/data/verifier/answers_*.jsonl")
    ap.add_argument("--synthetic-ratio", type=float, default=0.35)
    ap.add_argument("--rule-blocked-max", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    files = [Path(p) for p in args.inputs] if args.inputs else sorted(D.glob("answers_*.jsonl"))
    recs = [json.loads(line) for f in files for line in open(f, encoding="utf-8")]
    recs = [r for r in recs if r["level"] in LEVELS and r["split"] in ("train", "val")]
    if not recs:
        raise SystemExit(f"没有可用的已标注回答：{[str(f) for f in files]}")

    out: dict[str, list[dict]] = {"train": [], "val": []}
    for split in out:
        part = [r for r in recs if r["split"] == split]
        blocked = [r for r in part if r["rule_block"]]
        clean = [r for r in part if not r["rule_block"]]
        rng.shuffle(blocked)
        keep_blocked = blocked[: int(len(clean) * args.rule_blocked_max)]
        for r in clean + keep_blocked:
            out[split].append(row(r["state"], r["level"], {"split": split, "id": r["id"], "model": r["model"],
                                                           "synthetic": False, "rule_block": r["rule_block"]}))
        if split == "train":
            good = [r for r in clean if r["level"] in ("good", "acceptable")]
            rng.shuffle(good)
            target, n_syn = int(len(out[split]) * args.synthetic_ratio), 0
            for r in good:
                if n_syn >= target:
                    break
                p = perturb(r, good, rng)
                if p is None or _rules_block(r["state"], p[1]):
                    continue
                kind, ans, level = p
                out[split].append(row(_with_answer(r["state"], ans), level,
                                      {"split": split, "id": r["id"], "model": r["model"], "synthetic": kind}))
                n_syn += 1
        rng.shuffle(out[split])

    for split, rows in out.items():
        oa = D / "openai" / f"{split}.jsonl"
        mg = D / "medicalgpt" / split / "data.jsonl"
        for p in (oa, mg):
            p.parent.mkdir(parents=True, exist_ok=True)
        with open(oa, "w", encoding="utf-8") as w1, open(mg, "w", encoding="utf-8") as w2:
            for r in rows:
                w1.write(json.dumps(r, ensure_ascii=False) + "\n")
                w2.write(json.dumps(ef.openai_row_to_medicalgpt({"messages": r["messages"], "meta": {
                    k: v for k, v in r["meta"].items() if k != "state"}}), ensure_ascii=False) + "\n")
    stat = {s: {"n": len(rs), "levels": Counter(r["meta"]["level"] for r in rs),
                "synthetic": Counter(r["meta"]["synthetic"] for r in rs if r["meta"]["synthetic"]),
                "models": Counter(r["meta"]["model"] for r in rs)} for s, rs in out.items()}
    print(json.dumps(stat, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
