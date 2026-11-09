#!/usr/bin/env python3
"""训练数据审计：重复 / 近重复 / 训练-评测泄漏 / 工具参数是否有依据 / 非医疗污染 / DPO 偏差。

用法：
  python training/scripts/data_pipeline/audit_data.py --out evaluation/results/data_audit_v0.json
  python training/scripts/data_pipeline/audit_data.py --sft training/data/sft_v2/train.jsonl ... --out ..._v1.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from common import REPO, MinHashLSH, human, iter_function_calls, norm, read_jsonl, row_key, shingles, turns

from medagent.knowledge import load_kb

NON_MEDICAL = re.compile(r"React|JavaScript|Python|Java\b|SQL|HTML|CSS|Linux|内核|编程|代码|股票|基金|期货|电影|小说|"
                         r"诗人|诗歌|写一首|歌词|rap|足球|篮球|房价|作文|翻译成|数学题|方程|组件|函数式", re.I)
DUR = re.compile(r"([0-9]+|[一二两三四五六七八九十半几多]+)\s*(个)?(多)?\s*(天|周|星期|礼拜|月|小时|年)|最近|近期|这几天|好久|长期|很久")


def _template(q: str, kb) -> str:
    t = q
    for vocab in (kb.symptoms, kb.drugs, kb.guidelines):
        for k in sorted(vocab, key=len, reverse=True):
            t = t.replace(k, "<E>")
    return norm(re.sub(r"\d+", "<N>", t))


def dup_stats(rows: list[dict], name: str) -> dict:
    keys = Counter(row_key(r) for r in rows)
    exact_dups = sum(c - 1 for c in keys.values() if c > 1)
    lsh = MinHashLSH()
    near = 0
    for r in rows:
        sig = lsh.signature(shingles(human(r) + "".join(turns(r, "gpt"))[:300]))
        if any(lsh.similarity(i, sig) >= 0.8 for i in lsh.candidates(sig)):
            near += 1
        lsh.add(sig)
    return {"set": name, "n": len(rows), "exact_dup": exact_dups, "exact_dup_rate": round(exact_dups / max(len(rows), 1), 4),
            "near_dup": near, "near_dup_rate": round(near / max(len(rows), 1), 4)}


def overlap(train: list[dict], evals: list[dict]) -> dict:
    tk = {row_key(r) for r in train}
    tq = {norm(human(r)) for r in train}
    lsh = MinHashLSH()
    for r in train:
        lsh.add(lsh.signature(shingles(human(r))))
    exact = sum(1 for r in evals if row_key(r) in tk)
    same_q = sum(1 for r in evals if norm(human(r)) in tq)
    near = 0
    for r in evals:
        sig = lsh.signature(shingles(human(r)))
        if any(lsh.similarity(i, sig) >= 0.8 for i in lsh.candidates(sig)):
            near += 1
    n = max(len(evals), 1)
    return {"n_eval": len(evals), "exact_overlap": exact, "same_query": same_q, "near_query": near,
            "same_query_rate": round(same_q / n, 4), "near_query_rate": round(near / n, 4)}


def tool_arg_grounding(rows: list[dict], kb) -> dict:
    total = ungrounded_duration = drug_mismatch = invalid = 0
    examples: list[dict] = []
    for r in rows:
        q = human(r)
        for call in iter_function_calls(r):
            if "_invalid" in call:
                invalid += 1
                continue
            total += 1
            args = call.get("arguments", {})
            dur = str(args.get("duration", "") or "")
            if dur and dur not in ("未提及", "") and not DUR.search(q) and dur not in q:
                ungrounded_duration += 1
                if len(examples) < 5:
                    examples.append({"type": "ungrounded_duration", "query": q[:60], "arg": dur})
            if call.get("name") == "search_drug":
                name = str(args.get("name", ""))
                ctx = q + "".join(turns(r, "gpt"))
                std = kb.normalize(name, "drug")
                if name and name not in ctx and std not in ctx and not any(
                        a in ctx for a, s in kb.synonyms["drug"].items() if s == std):
                    drug_mismatch += 1
                    if len(examples) < 10:
                        examples.append({"type": "drug_not_in_context", "query": q[:60], "arg": name})
    return {"tool_calls": total, "invalid_json": invalid,
            "ungrounded_duration": ungrounded_duration,
            "ungrounded_duration_rate": round(ungrounded_duration / max(total, 1), 4),
            "drug_arg_not_in_context": drug_mismatch,
            "drug_arg_not_in_context_rate": round(drug_mismatch / max(total, 1), 4), "examples": examples}


MEDICAL = re.compile(r"病|症|药|医|痛|疼|治|疗|检查|血|炎|孕|月经|胃|肝|肾|心脏|肺|烧|咳|感冒|过敏|皮肤|健康|身体|"
                     r"手术|患者|诊断|激素|维生素|营养|睡眠|体重|饮食|婴|孩子|宝宝|性功能|排卵|脱发|视力|眼")


def is_non_medical(r: dict) -> bool:
    text = human(r) + " " + " ".join(turns(r, "gpt"))[:300]
    return bool(NON_MEDICAL.search(text)) and len(MEDICAL.findall(text)) <= 1


def contamination(rows: list[dict]) -> dict:
    # 有意加入的通用回放（meta.bucket/source = general）不算污染
    rows = [r for r in rows if (r.get("meta") or {}).get("bucket") != "general" and (r.get("meta") or {}).get("source") != "general"]
    hits = [human(r)[:50] for r in rows if is_non_medical(r)]
    return {"non_medical": len(hits), "rate": round(len(hits) / max(len(rows), 1), 4), "examples": hits[:10]}


def template_concentration(rows: list[dict], kb) -> dict:
    c = Counter(_template(human(r), kb) for r in rows)
    top = c.most_common(5)
    return {"unique_templates": len(c), "n": len(rows), "unique_ratio": round(len(c) / max(len(rows), 1), 4),
            "top5_share": round(sum(v for _, v in top) / max(len(rows), 1), 4)}


def dpo_bias(rows: list[dict]) -> dict:
    if not rows:
        return {}
    same = sum(1 for r in rows if norm(str(r.get("chosen"))) == norm(str(r.get("rejected"))))
    lc = [len(str(r.get("chosen", ""))) for r in rows]
    lr = [len(str(r.get("rejected", ""))) for r in rows]
    chosen_longer = sum(1 for a, b in zip(lc, lr) if a > b)
    uniq_chosen = len({norm(str(r.get("chosen"))) for r in rows})
    tool_chosen = sum(1 for r in rows if "Action:" in str(r.get("chosen", "")))
    return {"n": len(rows), "chosen_eq_rejected": same, "chosen_longer_rate": round(chosen_longer / len(rows), 4),
            "median_len_chosen": statistics.median(lc), "median_len_rejected": statistics.median(lr),
            "unique_chosen_ratio": round(uniq_chosen / len(rows), 4),
            "chosen_is_tool_call_rate": round(tool_chosen / len(rows), 4)}


def main() -> None:
    d = REPO / "training" / "data"
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", nargs="*", default=[str(d / "sft" / "all_sft.jsonl")])
    ap.add_argument("--sft-eval", default=str(d / "sft" / "eval_sft.jsonl"))
    ap.add_argument("--dpo", nargs="*", default=[str(d / "reward" / "medical_dpo.jsonl"), str(d / "reward" / "toolcall_dpo.jsonl")])
    ap.add_argument("--dpo-eval", default=str(d / "reward" / "eval_dpo.jsonl"))
    ap.add_argument("--opd", default=str(d / "opd" / "opd_data.jsonl"))
    ap.add_argument("--opd-eval", default=str(d / "opd" / "eval_opd.jsonl"))
    ap.add_argument("--raw-qa", default=str(d / "raw_medical_qa.jsonl"))
    ap.add_argument("--sft-validation-is-train", action="store_true", default=True,
                    help="sft_train.sh 里 --validation_file_dir 与 --train_file_dir 相同（v0 事实）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    kb = load_kb()
    sft = [r for p in args.sft for r in read_jsonl(p)]
    sft_eval = read_jsonl(args.sft_eval)
    dpo = [r for p in args.dpo for r in read_jsonl(p)]
    dpo_eval = read_jsonl(args.dpo_eval)
    opd = read_jsonl(args.opd) if Path(args.opd).exists() else []
    opd_eval = read_jsonl(args.opd_eval) if Path(args.opd_eval).exists() else []
    raw = read_jsonl(args.raw_qa) if Path(args.raw_qa).exists() else []
    report = {
        "duplicates": [dup_stats(sft, "sft"), dup_stats(dpo, "dpo"), dup_stats(opd, "opd")],
        "leakage": {"sft_eval_vs_train": overlap(sft, sft_eval), "dpo_eval_vs_train": overlap(dpo, dpo_eval),
                    "opd_eval_vs_train": overlap(opd, opd_eval) if opd else None,
                    "opd_vs_sft": overlap(sft, opd) if opd else None},
        "tool_args": tool_arg_grounding(sft + dpo, kb),
        "contamination": {"raw_medical_qa": contamination(raw), "sft": contamination(sft)},
        "templates": {"sft_toolcall": template_concentration([r for r in sft if turns(r, "function_call")], kb),
                      "dpo": template_concentration(dpo, kb)},
        "dpo_bias": dpo_bias(dpo),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "contamination"}, ensure_ascii=False, indent=1)[:3000])
    print("contamination:", report["contamination"]["raw_medical_qa"]["non_medical"], "rows")


if __name__ == "__main__":
    main()
