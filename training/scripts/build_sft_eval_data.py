#!/usr/bin/env python3
"""生成 SFT 的 eval 数据（150 条，工具调用 105 + 普通问答 45，保持 7:3）"""
import json
import random

random.seed(2026)

src = "/root/autodl-tmp/MedicalGPT/data/sft/all_sft.jsonl"
tool_calls, plain_qa = [], []
with open(src, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        has_fc = any(c["from"] == "function_call" for c in d["conversations"])
        if has_fc:
            tool_calls.append(d)
        else:
            plain_qa.append(d)

sel_tool = random.sample(tool_calls, min(105, len(tool_calls)))
sel_qa = random.sample(plain_qa, min(45, len(plain_qa)))
sft_eval = sel_tool + sel_qa
random.shuffle(sft_eval)

out = "/root/autodl-tmp/MedicalGPT/data/sft/eval_sft.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for d in sft_eval:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

print(f"SFT eval 数据: {len(sft_eval)} 条 (工具调用 {len(sel_tool)} + 普通问答 {len(sel_qa)})")
print(f"保存到 {out}")
