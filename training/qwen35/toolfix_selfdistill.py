"""给工具修补 SFT 准备“直接回答”样本：答案由 OPD 模型自己生成（自蒸馏），保住 OPD 的回答风格。

为什么（2026-11-14）：第一版工具修补只用调工具的样本，结果模型“见题就调工具”——没有工具定义的问答题 120/120 也去调。
SFT 数据里，系统提示带 <tools> 的样本 100% 调工具、不带的 100% 直接答，模型靠这一点区分；修补数据必须两种都有。
直接答样本的答案不能用 SFT 原始参考答案（就是 OPD 纠正掉的百科式、复读、荐药风格），所以用 OPD 自己的贪心输出，
并过滤掉：以 <tool_call> 开头、原地复读、没写完（到上限）的。

用法：python toolfix_selfdistill.py <OPD 合并模型> <sft_train 目录> <输出 jsonl> [条数=1600] [batch=16]
"""
import glob
import json
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/root/autodl-tmp/MedicalGPT/training")
from template import get_conv_template  # noqa: E402

MODEL, SRC, OUT = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 1600
BS = int(sys.argv[5]) if len(sys.argv) > 5 else 16
tpl = get_conv_template("qwen3_5_nothink")


def repetitive(t):  # 任一 12 字片段出现 ≥3 次
    seen = {}
    for i in range(0, max(0, len(t) - 12)):
        s = t[i:i + 12]
        seen[s] = seen.get(s, 0) + 1
        if seen[s] >= 3:
            return True
    return False


pool = []
for f in sorted(glob.glob(f"{SRC}/*.jsonl")):
    for line in open(f, encoding="utf-8"):
        conv = json.loads(line)["conversations"]
        if len(conv) == 3 and conv[0]["from"] == "system" and conv[1]["from"] == "human" \
                and "<tools>" not in conv[0]["value"] and not conv[2]["value"].lstrip().startswith("<tool_call>"):
            pool.append(conv[:2])
random.seed(0)
pick = random.sample(pool, min(int(N * 1.3), len(pool)))  # 多取 30%，过滤后再截到 N
print(f"无工具直接答候选 {len(pool)} 条，生成 {len(pick)} 条")

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
EOS = tok.convert_tokens_to_ids("<|im_end|>")
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0").eval()
kept, dropped = [], {"tool_call": 0, "复读": 0, "没写完": 0}
# 左填充批量生成与单条生成是否一致（Qwen3.5 有线性注意力层，填充处理不对会污染状态）
_ps = sorted((tpl.get_prompt(messages=[[c[1]["value"], ""]], system_prompt=c[0]["value"]) for c in pick[:20]), key=len)
p0, p1 = _ps[0], _ps[-1]  # p0 最短，和最长的 p1 同批时一定被左填充
with torch.no_grad():
    single = model.generate(**tok([p0], return_tensors="pt", add_special_tokens=False).to("cuda:0"), max_new_tokens=64,
                            do_sample=False, eos_token_id=EOS, pad_token_id=tok.pad_token_id)
    enc = tok([p1, p0], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
    batched = model.generate(**enc, max_new_tokens=64, do_sample=False, eos_token_id=EOS, pad_token_id=tok.pad_token_id)
a = tok.decode(single[0, len(tok(p0, add_special_tokens=False)["input_ids"]):], skip_special_tokens=True)
b = tok.decode(batched[1, enc["input_ids"].shape[1]:], skip_special_tokens=True)
print("批量 vs 单条（前 64 token）一致：", a == b)
if a != b:
    print("单条:", a[:120].replace("\n", " "), "\n批量:", b[:120].replace("\n", " "))
for i in range(0, len(pick), BS):
    batch = pick[i:i + BS]
    prompts = [tpl.get_prompt(messages=[[c[1]["value"], ""]], system_prompt=c[0]["value"]) for c in batch]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=384, do_sample=False, eos_token_id=EOS, pad_token_id=tok.pad_token_id)
    for conv, seq in zip(batch, out[:, enc["input_ids"].shape[1]:]):
        done = bool((seq == EOS).any())
        text = tok.decode(seq, skip_special_tokens=True).strip()
        if text.startswith("<tool_call>"):
            dropped["tool_call"] += 1
        elif repetitive(text):
            dropped["复读"] += 1
        elif not done:
            dropped["没写完"] += 1
        else:
            kept.append({"conversations": conv + [{"from": "gpt", "value": text}]})
    if (i // BS) % 10 == 0:
        print(f"  {i + len(batch)}/{len(pick)} 保留 {len(kept)} 丢弃 {dropped}", flush=True)
kept = kept[:N]
with open(OUT, "w", encoding="utf-8") as fo:
    for r in kept:
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"写出 {len(kept)} 条 → {OUT}；丢弃 {dropped}")
