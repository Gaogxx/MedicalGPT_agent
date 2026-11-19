#!/usr/bin/env python3
"""第 5 步 LoRA：在最终 9B（SFT → DPO v3 → OPD → 工具修补 v2 合并后）上训“按工具结果作答”（2026-11-20）。

- 样本用线上推理同一个对话模板（runtime/vllm/medicalgpt_qwen35_tools.jinja）渲染，和 vLLM 服务时模型看到的文本逐字相同；
- 损失只算最后一段回答（工具调用、工具结果、系统提示都不训练），调不调工具、调哪个的习惯不动；
- 按 Qwen3_5ForCausalLM 加载（只有文本部分），adapter 参数名与前几步一致，合并前用 remap_lora_for_vl.py 改名。

  python training/qwen35/grounding/g4_sft.py --model models/qwen3.5-9b-medical --data training/data/grounding/sft.jsonl \
      --out outputs/grounding-qwen3.5-9b-medical
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
import time

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments


class TimeBudget(TrainerCallback):
    """训练时间上限（分钟）：到点就停，按验证损失保留最好的检查点。"""
    def __init__(self, minutes: float) -> None:
        self.limit, self.t0 = minutes * 60, time.time()

    def on_step_end(self, args, state, control, **kw):
        if self.limit and time.time() - self.t0 > self.limit:
            control.should_training_stop = True
            control.should_evaluate = control.should_save = True
        return control

ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = (ROOT / "runtime" / "vllm" / "medicalgpt_qwen35_tools.jinja").read_text(encoding="utf-8")


def encode(tok, rec: dict, max_len: int) -> dict | None:
    msgs, tools = rec["messages"], rec.get("tools") or None
    prompt = tok.apply_chat_template(msgs[:-1], tools=tools, chat_template=TEMPLATE, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs, tools=tools, chat_template=TEMPLATE, tokenize=False)
    if not full.startswith(prompt):
        raise SystemExit(f"模板渲染不一致：{rec['id']}")
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
    if len(p) + len(a) > max_len:
        return None
    return {"input_ids": p + a, "labels": [-100] * len(p) + a}


def collate(batch: list[dict], pad: int) -> dict:
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad)
    lab = torch.full((len(batch), n), -100)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k] = torch.tensor(b["input_ids"])
        lab[i, :k] = torch.tensor(b["labels"])
        att[i, :k] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=16, help="每张卡的梯度累积步数；双卡（torchrun --nproc_per_node 2）时用 8，等效批 16")
    ap.add_argument("--max-samples", type=int, default=0, help="最多用多少条（按类别比例截取，0 = 全用）")
    ap.add_argument("--time-budget", type=float, default=0, help="训练最多多少分钟（0 = 不限）")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(x) for x in open(args.data, encoding="utf-8")]
    random.Random(0).shuffle(rows)
    if args.max_samples and len(rows) > args.max_samples:
        rows = rows[: args.max_samples]                     # 已打乱，各类别大致按原比例保留
    enc = [e for e in (encode(tok, r, args.max_len) for r in rows) if e]
    n_val = max(20, len(enc) // 33)
    train, val = enc[n_val:], enc[:n_val]
    if int(__import__("os").environ.get("LOCAL_RANK", 0)) == 0:
        print(f"样本 {len(rows)}，可用 {len(enc)}（超长丢弃 {len(rows) - len(enc)}），训练 {len(train)}、验证 {len(val)}", flush=True)
    import os
    rank = int(os.environ.get("LOCAL_RANK", 0))           # torchrun 多卡数据并行时每个进程一张卡
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": rank})
    names = sorted({n.split(".")[-1] for n, m in model.named_modules()
                    if isinstance(m, torch.nn.Linear) and ".layers." in n})
    print("LoRA 目标模块", names, flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
                                             target_modules=names, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    targs = TrainingArguments(output_dir=args.out, per_device_train_batch_size=1, per_device_eval_batch_size=1,
                              gradient_accumulation_steps=args.accum, num_train_epochs=args.epochs, learning_rate=args.lr,
                              lr_scheduler_type="cosine", warmup_steps=0.05, weight_decay=0.0, bf16=True,
                              logging_steps=10, eval_strategy="steps", eval_steps=20, save_strategy="steps", save_steps=20,
                              save_total_limit=2, report_to=[], remove_unused_columns=False, dataloader_num_workers=2,
                              load_best_model_at_end=True, metric_for_best_model="eval_loss", ddp_find_unused_parameters=False)
    tr = Trainer(model=model, args=targs, train_dataset=train, eval_dataset=val,
                 data_collator=lambda b: collate(b, tok.pad_token_id or 0), callbacks=[TimeBudget(args.time_budget)])
    tr.train()
    tr.save_model(args.out)
    tok.save_pretrained(args.out)
    json.dump(tr.state.log_history, open(Path(args.out) / "log_history.json", "w"), ensure_ascii=False, indent=1)
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
