#!/bin/bash
# 判别模型 第 2 步：真实回答 + 裁判标签（等级 + 失败类别）+ 语义故障注入 → Laya 训练格式；跑一致性测试
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate serve
WORK=${WORK:-/root/autodl-tmp}; REPO=${REPO:-$WORK/MedicalGPT_agent}
cd $REPO
python training/scripts/verifier/build_verifier_data.py
python -m pytest -q tests/test_discriminator.py
wc -l training/data/verifier/laya/*.jsonl
# 检查：train 里答砸（poor / unsafe）应占 25%~50%；太少就加 --synthetic-ratio，太多说明裁判偏严
