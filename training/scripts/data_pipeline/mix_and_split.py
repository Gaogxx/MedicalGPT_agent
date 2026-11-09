#!/usr/bin/env python3
"""清洗 → 去重 → 去污染 → 修复 → 按配比混合 → 按“模板组”切分 train/val，产出 v2 数据与 manifest。

防过拟合的关键动作：
  1. 评测集与训练集互斥：v0 的 eval_sft/eval_dpo/eval_opd 150/150 全部原样出现在训练集里
  2. 按问法模板分组切分：同一模板（实体替换后相同）的样本只会落在 train 或 val 一侧，val_loss 才有意义
  3. 模板上限：同一模板最多 N 条（DPO v0 只有 12 种模板）
  4. 近重复只留一条（MinHash，Jaccard ≥ 0.8）
  5. 增强样本 ≤ 30%，每个原始样本最多 2 个变体
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from audit_data import _template, is_non_medical
from common import REPO, MinHashLSH, human, read_jsonl, row_key, shingles, turns, write_jsonl

from medagent.knowledge import load_kb

DUR = re.compile(r"([0-9]+|[一二两三四五六七八九十半几多]+)\s*(个)?(多)?\s*(天|周|星期|礼拜|月|小时|年)|最近|近期|这几天|好久|长期|很久")


def dedup(rows: list[dict], thr: float, protected_keys: set[str] | None = None) -> list[dict]:
    seen: set[str] = set()
    lsh = MinHashLSH()
    out = []
    for r in rows:
        k = row_key(r)
        if k in seen:
            continue
        sig = lsh.signature(shingles(human(r) + "".join(turns(r, "gpt"))[:300] + str(r.get("chosen", ""))[:200]))
        if any(lsh.similarity(i, sig) >= thr for i in lsh.candidates(sig)):
            continue
        seen.add(k)
        lsh.add(sig)
        out.append(r)
    return out


def decontaminate(rows: list[dict], eval_queries: set[str]) -> list[dict]:
    from common import norm
    return [r for r in rows if norm(human(r)) not in eval_queries]


def cap_templates(rows: list[dict], kb, cap: int, rng: random.Random) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[_template(human(r), kb)].append(r)
    out = []
    for g in groups.values():
        rng.shuffle(g)
        out.extend(g[:cap])
    return out


def group_split(rows: list[dict], kb, val_ratio: float) -> tuple[list[dict], list[dict]]:
    train, val = [], []
    for r in rows:
        anchor = r.get("meta", {}).get("orig_query") or human(r)   # 增强样本跟原句同组，避免变体跨 train/val
        h = int(hashlib.md5(_template(anchor, kb).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        (val if h < val_ratio else train).append(r)
    return train, val


def repair_dpo(row: dict) -> dict:
    q = human(row)
    ch = str(row.get("chosen", ""))
    m = re.search(r'("duration"\s*:\s*"([^"]*)")', ch)
    if m and m.group(2) and not DUR.search(q) and m.group(2) not in q:
        ch = re.sub(r',\s*"duration"\s*:\s*"[^"]*"|"duration"\s*:\s*"[^"]*",?\s*', "", ch)
        row = {**row, "chosen": ch, "meta": {**row.get("meta", {}), "repaired": "ungrounded_duration"}}
    return row


def take(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    rows = list(rows)
    rng.shuffle(rows)
    return rows[: max(0, n)]


def main() -> None:
    d = REPO / "training" / "data"
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "training" / "configs" / "data_mix_v2.yaml"))
    ap.add_argument("--out-sft", default=str(d / "sft_v2"))
    ap.add_argument("--out-dpo", default=str(d / "reward_v2"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    rng = random.Random(cfg["seed"])
    kb = load_kb()
    aug = d / "augmented"

    # 评测集（含新评测集 evaluation/datasets/*）的查询全部从训练侧剔除
    from common import norm
    eval_q = {norm(human(r)) for p in (d / "sft" / "eval_sft.jsonl", d / "reward" / "eval_dpo.jsonl",
                                       d / "opd" / "eval_opd.jsonl") for r in read_jsonl(p)}
    for p in (REPO / "evaluation" / "datasets").glob("agent_eval*.jsonl"):
        eval_q |= {norm(json.loads(l)["query"]) for l in open(p, encoding="utf-8")}

    raw = read_jsonl(d / "sft" / "medical_qa.jsonl")
    general = [r for r in raw if is_non_medical(r)]
    medical_qa = [r for r in raw if not is_non_medical(r)]
    toolcall = [r for f in ("toolcall_symptom", "toolcall_drug", "toolcall_guideline") for r in read_jsonl(d / "sft" / f"{f}.jsonl")]
    repaired = {row_key(r): r for r in read_jsonl(aug / "sft_repaired.jsonl")} if (aug / "sft_repaired.jsonl").exists() else {}
    toolcall = [repaired.get(row_key(r), r) for r in toolcall]
    sources = {"medical_qa": medical_qa, "toolcall": toolcall}
    for k in ("slot_mask", "ask_clarify", "obs_failure", "unknown_entity", "colloquial", "raft"):
        p = aug / f"sft_{k}.jsonl"
        sources[k] = read_jsonl(p) if p.exists() else []
    # 安全样本：从 v0 DPO 里取急症/高风险类的 chosen 改造为 SFT（去重后）
    dpo_v0 = read_jsonl(d / "reward" / "medical_dpo.jsonl") + read_jsonl(d / "reward" / "toolcall_dpo.jsonl")
    safety = [{"conversations": [{"from": "human", "value": human(r)}, {"from": "gpt", "value": r["chosen"]}],
               "meta": {"src": "dpo_v0_safety"}} for r in dpo_v0
              if re.search(r"处方|诊断|120|急诊|医生确认", str(r.get("chosen", ""))) and "Action:" not in str(r.get("chosen", ""))]
    sources["safety"] = safety
    gp = cfg["sft"].get("general_replay_path")
    sources["general_replay"] = read_jsonl(gp) if gp else general

    report: dict = {"sft": {}, "dpo": {}}
    pool: list[dict] = []
    total = cfg["sft"]["target_total"]
    for name, ratio in cfg["sft"]["mix"].items():
        rows = sources.get(name, [])
        before = len(rows)
        rows = decontaminate(rows, eval_q)
        rows = dedup(rows, cfg["near_dup_threshold"])
        rows = cap_templates(rows, kb, cfg["max_per_template"], rng)
        picked = take(rows, int(round(total * ratio)), rng)
        for r in picked:
            r.setdefault("meta", {})["source"] = name
        report["sft"][name] = {"raw": before, "after_clean": len(rows), "picked": len(picked),
                               "target": int(round(total * ratio))}
        pool.extend(picked)
    train, val = group_split(pool, kb, cfg["val_ratio"])
    aug_names = {"slot_mask", "ask_clarify", "obs_failure", "unknown_entity", "colloquial", "raft"}
    n_aug = sum(1 for r in pool if r["meta"]["source"] in aug_names)
    report["sft"]["_summary"] = {"train": len(train), "val": len(val), "aug_share": round(n_aug / max(len(pool), 1), 4),
                                 "by_source": Counter(r["meta"]["source"] for r in pool)}
    write_jsonl(Path(args.out_sft) / "train.jsonl", train)
    write_jsonl(Path(args.out_sft) / "val.jsonl", val)

    # ---------- DPO ----------
    dpo_orig = [repair_dpo(r) for r in decontaminate(dpo_v0, eval_q)]
    dpo_orig = dedup(dpo_orig, cfg["near_dup_threshold"])
    dpo_orig = cap_templates(dpo_orig, kb, cfg["dpo"]["max_per_template"], rng)
    cf = read_jsonl(aug / "dpo_counterfactual.jsonl") if (aug / "dpo_counterfactual.jsonl").exists() else []
    cf = cap_templates(dedup(decontaminate(cf, eval_q), cfg["near_dup_threshold"]), kb, cfg["dpo"]["max_per_template"], rng)
    n_orig = len(dpo_orig)
    n_cf = min(len(cf), max(int(n_orig / cfg["dpo"]["mix"]["original_dedup"] * cfg["dpo"]["mix"]["counterfactual"]),
                            cfg["dpo"].get("min_counterfactual", 0)))
    dpo = dpo_orig + take(cf, n_cf, rng)
    dtrain, dval = group_split(dpo, kb, cfg["val_ratio"] * 1.5)
    report["dpo"] = {"original_raw": len(dpo_v0), "original_after_clean": n_orig, "counterfactual_available": len(cf),
                     "counterfactual_picked": n_cf, "train": len(dtrain), "val": len(dval),
                     "repaired_ungrounded_duration": sum(1 for r in dpo_orig if r.get("meta", {}).get("repaired"))}
    write_jsonl(Path(args.out_dpo) / "train.jsonl", dtrain)
    write_jsonl(Path(args.out_dpo) / "val.jsonl", dval)
    man = Path(args.out_sft).parent / "data_v2_manifest.json"
    man.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=dict), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1, default=dict))


if __name__ == "__main__":
    main()
