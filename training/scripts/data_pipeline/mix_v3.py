#!/usr/bin/env python3
"""v3 混合与切分：读 data_mix_v3.yaml，逐来源 清洗 → 修复 → 去污染 → 去近重复 → 模板上限 → 按配比取样 → 按模板分组切分。
产物：training/data/sft_v3/{train,val}.jsonl、training/data/reward_v3/{train,val}.jsonl、training/data/data_v3_manifest.json"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import yaml
from audit_data import is_non_medical
from common import REPO, human, read_jsonl, row_key, write_jsonl
from mix_and_split import cap_same_chosen, cap_templates, dedup, fix_legacy_calls, group_split, repair_dpo
from supplement import Decontam, eval_texts

from medagent.knowledge import load_kb

D = REPO / "training" / "data"


def load(files: list[str]) -> list[dict]:
    rows = []
    for f in files:
        p = D / f
        if p.exists():
            rows += read_jsonl(p)
    return rows


def balance_length(rows: list[dict], max_share: float) -> list[dict]:
    """DPO 长度偏差平衡：chosen 比 rejected 长的对超过 max_share 时，先丢长度比最悬殊的那些（问题 N57）。"""
    longer = [r for r in rows if len(str(r["chosen"])) > len(str(r["rejected"]))]
    other = [r for r in rows if len(str(r["chosen"])) <= len(str(r["rejected"]))]
    keep_n = int(max_share / (1 - max_share) * len(other)) if max_share < 1 else len(longer)
    if len(longer) <= keep_n:
        return rows
    longer.sort(key=lambda r: len(str(r["chosen"])) / max(len(str(r["rejected"])), 1))
    return other + longer[:keep_n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "training" / "configs" / "data_mix_v3.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    rng = random.Random(cfg["seed"])
    kb = load_kb()
    dc = Decontam(eval_texts())
    repaired = {row_key(r): r for r in read_jsonl(D / "augmented" / "sft_repaired.jsonl")} if (D / "augmented" / "sft_repaired.jsonl").exists() else {}
    report: dict = {"sft": {}, "dpo": {}}
    pool: list[dict] = []
    total = cfg["sft"]["target_total"]
    for name, spec in cfg["sft"]["sources"].items():
        rows = load(spec["files"])
        raw = len(rows)
        if name == "real_qa":
            rows = [r for r in rows if not is_non_medical(r)]
        if name == "toolcall":
            rows = [repaired.get(row_key(r), r) for r in rows]
        rows = [r for r in rows if not dc.bad(human(r))]
        rows = dedup(rows, cfg["near_dup_threshold"])
        rows = cap_templates(rows, kb, cfg["max_per_template"], rng)
        rng.shuffle(rows)
        target = int(round(total * spec["ratio"]))
        picked = rows[:target]
        for r in picked:
            r.setdefault("meta", {})["bucket"] = name
        report["sft"][name] = {"raw": raw, "after_clean": len(rows), "picked": len(picked), "target": target}
        pool += picked
    # 跨来源全局去重：同一个问题可能同时出现在 Huatuo 百科与 Lite 里，只在来源内去重会让它落到 train/val 两侧（问题 N56）
    from common import norm
    seen_q: set[str] = set()
    uniq = []
    for r in pool:
        k = norm(human(r))
        if k in seen_q:
            continue
        seen_q.add(k)
        uniq.append(r)
    report["sft"]["_cross_source_dups_removed"] = len(pool) - len(uniq)
    pool = uniq
    train, val = group_split(pool, kb, cfg["val_ratio"])
    synthetic = {"toolcall_ext", "masked", "raft", "safety", "clarify"}
    n_syn = sum(1 for r in pool if r["meta"]["bucket"] in synthetic)
    report["sft"]["_summary"] = {"train": len(train), "val": len(val), "total": len(pool),
                                 "synthetic_share": round(n_syn / max(len(pool), 1), 4),
                                 "real_share": round(1 - n_syn / max(len(pool), 1), 4),
                                 "by_bucket": dict(Counter(r["meta"]["bucket"] for r in pool))}
    write_jsonl(D / "sft_v3" / "train.jsonl", train)
    write_jsonl(D / "sft_v3" / "val.jsonl", val)

    dpool: list[dict] = []
    for name, spec in cfg["dpo"]["sources"].items():
        rows = load(spec["files"])
        raw = len(rows)
        if name == "original":
            rows = [repair_dpo(r) for r in rows]
        rows = [fix_legacy_calls(r) for r in rows]
        if name == "real":
            rows = balance_length(rows, cfg["dpo"].get("max_chosen_longer_share", 0.6))
        rows = [r for r in rows if not dc.bad(human(r))]
        rows = dedup(rows, cfg["near_dup_threshold"])
        rows = cap_templates(rows, kb, cfg["dpo"]["max_per_template"], rng)
        if cfg["dpo"].get("max_same_chosen"):
            rows = cap_same_chosen(rows, cfg["dpo"]["max_same_chosen"], rng)
        rng.shuffle(rows)
        picked = rows[: spec["cap"]]
        for r in picked:
            r.setdefault("meta", {})["bucket"] = name
        report["dpo"][name] = {"raw": raw, "after_clean": len(rows), "picked": len(picked)}
        dpool += picked
    dtrain, dval = group_split(dpool, kb, cfg["val_ratio"] * 1.5)
    report["dpo"]["_summary"] = {"train": len(dtrain), "val": len(dval), "total": len(dpool),
                                 "by_bucket": dict(Counter(r["meta"]["bucket"] for r in dpool))}
    write_jsonl(D / "reward_v3" / "train.jsonl", dtrain)
    write_jsonl(D / "reward_v3" / "val.jsonl", dval)
    report["decontam_rejected"] = dc.hits
    (D / "data_v3_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
