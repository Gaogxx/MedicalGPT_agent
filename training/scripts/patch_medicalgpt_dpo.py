#!/usr/bin/env python3
"""给 MedicalGPT 的 training/dpo_training.py 增加 --dpo_beta / --dpo_loss_type / --dpo_label_smoothing / --dpo_rpo_alpha。

原脚本（2026-09 版本）构造 DPOConfig 时没有透传这些参数，beta 固定为 TRL 默认 0.1，也无法开 RPO / label smoothing。
用法：python patch_medicalgpt_dpo.py /path/to/MedicalGPT/training/dpo_training.py   （幂等，可重复执行）
"""
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
src = path.read_text(encoding="utf-8")
if "dpo_beta" in src:
    print("already patched")
    sys.exit(0)
fields = '''    dpo_beta: Optional[float] = field(default=0.1, metadata={"help": "DPO beta"})
    dpo_loss_type: Optional[str] = field(default="sigmoid", metadata={"help": "sigmoid | ipo | ..."})
    dpo_label_smoothing: Optional[float] = field(default=0.0, metadata={"help": "cDPO label smoothing"})
    dpo_rpo_alpha: Optional[float] = field(default=None, metadata={"help": "RPO: add SFT NLL on chosen"})
'''
anchor = '    eval_strategy: Optional[str] = field(default="steps"'
assert anchor in src, "ScriptArguments 结构已变化，请手动修改"
src = src.replace(anchor, fields + anchor, 1)
call = "    training_args = DPOConfig(\n"
assert call in src, "DPOConfig 调用结构已变化，请手动修改"
src = src.replace(call, call + "        beta=args.dpo_beta,\n        loss_type=args.dpo_loss_type,\n"
                  "        label_smoothing=args.dpo_label_smoothing,\n        rpo_alpha=args.dpo_rpo_alpha,\n", 1)
path.write_text(src, encoding="utf-8")
print("patched", path)
