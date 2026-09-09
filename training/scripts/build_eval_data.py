#!/usr/bin/env python3
"""
生成 DPO 和 OPD 的 eval 数据（各 150 条，用于训练后的端到端评估）

DPO eval: 从 toolcall_dpo(105) + medical_dpo(45) 采样，保持 7:3
OPD eval: 从 opd_data.jsonl 随机采样 150 条（数据已打乱，保持比例）
"""
import json
import random

random.seed(2026)

# ============ DPO eval ============
dpo_tool = [json.loads(l) for l in open("/root/autodl-tmp/MedicalGPT/data/reward/toolcall_dpo.jsonl", encoding="utf-8") if l.strip()]
dpo_med = [json.loads(l) for l in open("/root/autodl-tmp/MedicalGPT/data/reward/medical_dpo.jsonl", encoding="utf-8") if l.strip()]

sel_tool = random.sample(dpo_tool, min(105, len(dpo_tool)))
sel_med = random.sample(dpo_med, min(45, len(dpo_med)))
dpo_eval = sel_tool + sel_med
random.shuffle(dpo_eval)

with open("/root/autodl-tmp/MedicalGPT/data/reward/eval_dpo.jsonl", "w", encoding="utf-8") as f:
    for d in dpo_eval:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
print(f"DPO eval 数据: {len(dpo_eval)} 条 (工具调用 {len(sel_tool)} + 普通问答 {len(sel_med)}) -> data/reward/eval_dpo.jsonl")

# ============ OPD eval ============
opd_data = [json.loads(l) for l in open("/root/autodl-tmp/MedicalGPT/data/opd/opd_data.jsonl", encoding="utf-8") if l.strip()]
opd_eval = random.sample(opd_data, min(150, len(opd_data)))

with open("/root/autodl-tmp/MedicalGPT/data/opd/eval_opd.jsonl", "w", encoding="utf-8") as f:
    for d in opd_eval:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
print(f"OPD eval 数据: {len(opd_eval)} 条 -> data/opd/eval_opd.jsonl")

# 统计 OPD eval 的类别分布（多轮/追问/单轮/普通/失败/模糊）
from collections import Counter
cat = Counter()
for d in opd_eval:
    n_func = sum(1 for c in d["conversations"] if c["from"] == "function_call")
    has_err = any("error" in c["value"] or "未找到" in c["value"] for c in d["conversations"] if c["from"] == "observation")
    has_clarify = len([c for c in d["conversations"] if c["from"] == "human"]) >= 2
    if n_func >= 2:
        cat["多轮"] += 1
    elif has_err:
        cat["失败恢复"] += 1
    elif has_clarify:
        cat["追问澄清"] += 1
    elif n_func == 1:
        cat["单轮"] += 1
    else:
        cat["普通问答"] += 1
print(f"OPD eval 类别分布: {dict(cat)}")
