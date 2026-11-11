#!/usr/bin/env python3
"""判别模型训练第 3 步：微调 Laya 多语言版（mmBERT-base 编码器 + 决策头，3.2 亿参数）。

用 Laya 自带的 laya.train.finetune：读 {state, questions, expected} 行 → 训练（RLCD 损失）→ 留出一小份做温度校准 → 保存。
保存的目录可以直接 laya.load(<目录>)，也可以注册给 laya-serve（04_serve.sh）。

  python training/scripts/verifier/train_laya.py --out /root/autodl-tmp/models/laya-medical-verifier
  python training/scripts/verifier/train_laya.py --train <小文件> --epochs 1 --out <目录>   # 本机冒烟（Mac 用 MPS）
需要单独的环境：pip install laya（会装 torch / transformers，不要装进网关的运行环境）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
D = ROOT / "training" / "data" / "verifier" / "laya"


def main() -> None:
    from laya.train import TrainConfig, finetune

    from medagent.verify.discriminator import MAX_LEN

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="multilingual", help="Laya 基础检查点：别名 multilingual 或本地目录")
    ap.add_argument("--train", default=str(D / "train.jsonl"))
    ap.add_argument("--val", default=str(D / "val.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    cfg = TrainConfig(epochs=args.epochs, micro_batch=args.micro_batch, grad_accum=args.grad_accum,
                      max_len=MAX_LEN, eval_data=args.val if Path(args.val).exists() else None)
    summary = finetune(args.train, args.base, args.out, cfg, device=args.device)
    Path(args.out, "medagent_train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str),
                                                             encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k in ("train_items", "calibration_items", "eval_items", "skipped",
                                                                 "evaluation")}, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
