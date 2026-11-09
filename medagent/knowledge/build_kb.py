#!/usr/bin/env python3
"""可复现地构建统一知识库 kb.json，并记录每个条目的来源与冲突处理。

背景（为什么要“统一”）：原仓库里同一套医疗知识表有两份拷贝——
  A. runtime/dsh-medical-plugins/src/index.ts       线上运行时工具查的表：症状 10 / 药品 5 / 指南 4
  B. training/scripts/build_toolcall_data.py        造训练数据时生成 observation 的表：症状 20 / 药品 10 / 指南 9
模型是用 B 生成的轨迹训练的（会去查“咽喉痛”“奥美拉唑”“冠心病”），上线后工具却查 A——
训练与服务不一致（train/serve skew）：B 有而 A 没有的 11 个症状、5 种药、5 条指南线上一律“查不到”，
“头晕”的病因两边还不一样。基线里“我嗓子疼，先查症状再查阿莫西林”答成“无法确定病因”就是这个原因。

合并规则：取 A ∪ B；同名条目冲突时以 B 为准（与训练数据一致），并在 provenance 里记录冲突内容。
注意：A、B 都是原作者手写的规则表，不是从权威数据库导入的，指南条目只是对指南名称的一两句概括，
所有条目 reviewed=false，上线前需要医生 / 药师逐条审核（见 docs/rag_sources.md 的权威来源建议）。

用法：python -m medagent.knowledge.build_kb [--version 2026.11.05-2]
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

HERE = Path(__file__).parent
SOURCES = {"A": HERE / "sources" / "dsh_index_ts_v0.json", "B": HERE / "sources" / "build_toolcall_data_v0.json"}
SECTIONS = ("symptoms", "drugs", "guidelines")


def build(version: str) -> tuple[dict, dict]:
    src = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in SOURCES.items()}
    kb: dict = {"version": version}
    provenance: dict = {}
    report: dict = {"sources": {k: v["source"] for k, v in src.items()}, "counts": {}, "only_in_A": {}, "only_in_B": {},
                    "conflicts": []}
    for sec in SECTIONS:
        a, b = src["A"][sec], src["B"][sec]
        merged = {}
        for key in sorted(set(a) | set(b), key=lambda k: (k not in b, k)):
            if key in b:
                merged[key] = b[key]
                origin = ["B"] + (["A"] if key in a else [])
                if key in a and a[key] != b[key]:
                    diff = {f: {"A": a[key].get(f), "B": b[key].get(f)} for f in set(a[key]) | set(b[key])
                            if a[key].get(f) != b[key].get(f)}
                    report["conflicts"].append({"section": sec, "key": key, "resolved_with": "B", "diff": diff})
            else:
                merged[key] = a[key]
                origin = ["A"]
            provenance[f"{sec}:{key}"] = {"from": origin, "reviewed": False,
                                          "note": "原仓库手写规则表，未经医学审核"}
        kb[sec] = merged
        report["counts"][sec] = {"A": len(a), "B": len(b), "merged": len(merged)}
        report["only_in_A"][sec] = sorted(set(a) - set(b))
        report["only_in_B"][sec] = sorted(set(b) - set(a))
    kb["meta"] = {"built": date.today().isoformat(), "builder": "medagent/knowledge/build_kb.py",
                  "merge_rule": "A ∪ B，冲突以 B（训练数据所用）为准", "reviewed_entries": 0,
                  "warning": "规则表来自原仓库手写内容，非权威数据库导入；上线前需医生/药师审核"}
    kb["provenance"] = provenance
    return kb, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="2026.11.05-2")
    ap.add_argument("--report", default=str(HERE.parents[1] / "evaluation" / "results" / "kb_unification.json"))
    args = ap.parse_args()
    kb, report = build(args.version)
    (HERE / "kb.json").write_text(json.dumps(kb, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("counts", "only_in_A", "only_in_B")}, ensure_ascii=False))
    print("conflicts:", [(c["section"], c["key"], list(c["diff"])) for c in report["conflicts"]])


if __name__ == "__main__":
    main()
