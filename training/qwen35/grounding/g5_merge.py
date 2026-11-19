#!/usr/bin/env python3
"""把第 5 步 LoRA 合并进最终 9B（和 rebuild_from_release.sh 同一套做法：remap 参数名 → peft 合并 → 逐层核对增量）。

  python training/qwen35/grounding/g5_merge.py <base 合并模型> <adapter 目录> <输出目录>
"""
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

REMAP = Path(__file__).resolve().parents[1] / "remap_lora_for_vl.py"
base, ad, out = sys.argv[1:4]
subprocess.run([sys.executable, str(REMAP), ad, ad + "-vl"], check=True)
m = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
m = PeftModel.from_pretrained(m, ad + "-vl").merge_and_unload()
m.save_pretrained(out, max_shard_size="10GB")
AutoProcessor.from_pretrained(base).save_pretrained(out)
del m
torch.cuda.empty_cache()
subprocess.run([sys.executable, str(REMAP), "--check", base, out, ad], check=True)
print("MERGED", out)
