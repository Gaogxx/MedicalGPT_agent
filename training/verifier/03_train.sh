#!/bin/bash
# 判别模型 第 3 步：微调 Laya 多语言版（3.2 亿参数，mmBERT-base + 决策头）。Laya 自带训练：RLCD 损失 + 留出数据温度校准
#   单张 24GB 卡足够（作者报告 2×T4 约 30k 题 4 epoch 4~5 小时；本项目约 4k 条 × 3 个问题）；Mac 也能跑（--micro-batch 2）
#   环境：laya（单独的 conda 环境，Laya 会装自己需要的 torch / transformers）
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate laya
WORK=${WORK:-/root/autodl-tmp}; REPO=${REPO:-$WORK/MedicalGPT_agent}
cd $REPO
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
python training/scripts/verifier/train_laya.py --out $WORK/models/laya-medical-verifier \
    --epochs 4 --micro-batch 8 --grad-accum 8 2>&1 | tee $WORK/outputs/laya-medical-verifier.log
# 检查：日志末尾 “Evaluation (before vs after fine-tuning on N held-out items …)” 里 Accuracy 上升、Loss 下降；
#       skipped 里不应有 options_beyond_max_len（状态太长时会出现）
