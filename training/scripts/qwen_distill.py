#!/usr/bin/env python3
"""
Qwen2.5-32B → Qwen2.5-7B token 级 logprob 蒸馏（软标签蒸馏）。

原理：
  Qwen2.5-32B(教师) 与 Qwen2.5-7B(学生) 共享同一套 tokenizer(vocab 151936)，
  因此教师 API 返回的 top-k logprob 可以逐 token 对齐到学生模型，
  用教师的 top-k 概率分布做软标签，训练学生对齐（token 级 KL 蒸馏）。

前提：
  Qwen2.5-32B 的 API 需支持返回 logprobs（OpenAI 兼容接口 + logprobs=true + top_logprobs=k）。

用法：
  export QWEN_API_KEY=sk-xxx
  export QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
  python qwen_distill.py \
      --student /root/autodl-tmp/MedicalGPT/outputs/dpo-qwen2.5-7b-agent \
      --teacher-model qwen2.5-32b-instruct \
      --train-file /root/autodl-tmp/MedicalGPT/data/sft/all_sft.jsonl \
      --n 1000 --top-k 10 --epochs 1 \
      --output /root/autodl-tmp/MedicalGPT/outputs/distill-qwen2.5-7b-agent

依赖：torch, transformers（纯标准库调 API，无 openai 依赖）。
"""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

API_KEY = os.environ.get("QWEN_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")


def teacher_logprobs(messages: list[dict], model: str, top_k: int) -> dict:
    """调教师 API，返回每个 token 的 top-k logprob（OpenAI 兼容格式）。"""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512,
        "logprobs": True,
        "top_logprobs": top_k,
    }
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    resp = urllib.request.urlopen(req, timeout=180)
    data = json.loads(resp.read().decode("utf-8"))
    choice = data["choices"][0]
    content = choice["message"]["content"]
    logprobs = choice.get("logprobs", {}).get("content", [])
    return {"content": content, "token_logprobs": logprobs}


def main():
    import torch
    from torch.nn import functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher-model", default="qwen2.5-32b-instruct")
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if not API_KEY:
        sys.exit("缺少 QWEN_API_KEY")

    # 加载学生模型 + tokenizer（Qwen2.5-7B + LoRA）
    print("[蒸馏] 加载学生模型 ...")
    tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.student, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.train()

    # 提取 human 问题
    questions = []
    with open(args.train_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            for c in d.get("conversations", []):
                if c.get("from") == "human":
                    questions.append(c["value"])
                    break
    questions = questions[: args.n]
    print(f"[蒸馏] 样本数: {len(questions)}")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for i, q in enumerate(questions):
            try:
                # 1) 教师生成 + top-k logprob
                t = teacher_logprobs([{"role": "user", "content": q}], args.teacher_model, args.top_k)
                teacher_tokens = t["token_logprobs"]  # [{token, logprob, top_logprobs:[...]}]
                if not teacher_tokens:
                    continue
                # 教师生成的 token id 序列（tokenizer 一致，直接 encode）
                teacher_ids = tokenizer.convert_tokens_to_ids(
                    [tk["token"] for tk in teacher_tokens if tk.get("token")]
                )
                # 2) 学生 forward：用教师 token 序列作为 target
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
                )
                prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
                full_ids = torch.cat([prompt_ids, torch.tensor([teacher_ids], device=model.device)], dim=1)

                logits = model(full_ids).logits  # [1, L, V]
                # 3) 逐 token 计算软标签蒸馏损失（KL）
                loss = 0.0
                for j, tk in enumerate(teacher_tokens):
                    top = tk.get("top_logprobs", [])
                    if not top:
                        continue
                    # 教师 top-k 分布（token -> logprob -> prob）
                    tok_ids = []
                    tok_probs = []
                    for entry in top:
                        tid = tokenizer.convert_tokens_to_ids(entry["token"])
                        if tid != tokenizer.unk_token_id:
                            tok_ids.append(tid)
                            tok_probs.append(torch.exp(torch.tensor(entry["logprob"])))
                    if not tok_ids:
                        continue
                    tok_probs = torch.tensor(tok_probs, device=model.device)
                    tok_probs = tok_probs / tok_probs.sum()  # 归一化到 top-k
                    # 学生 logits 取对应位置
                    pos = prompt_ids.shape[1] + j  # 学生预测第 j 个 token 的位置
                    student_logits = logits[0, pos, tok_ids]
                    student_probs = F.softmax(student_logits, dim=-1)
                    # KL(教师 || 学生)
                    loss = loss + F.kl_div(
                        student_probs.log(), tok_probs, reduction="sum"
                    )
                loss = loss / len(teacher_tokens)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                if (i + 1) % 20 == 0:
                    print(f"  epoch {epoch} | 样本 {i+1}/{len(questions)} | loss {loss.item():.4f}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  [跳过] 样本 {i}: {e}", file=sys.stderr)
                continue

        print(f"[蒸馏] epoch {epoch} 完成，平均 loss {total_loss/max(len(questions),1):.4f}")

    # 保存
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"[蒸馏] 完成，模型保存到 {args.output}")


if __name__ == "__main__":
    main()
