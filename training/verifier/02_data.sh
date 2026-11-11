#!/bin/bash
# 判别模型 第 2 步：真实回答 + 裁判标签 + 语义故障注入 → OpenAI 格式 → MedicalGPT 输入；跑格式一致性测试
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate serve
WORK=${WORK:-/root/autodl-tmp}; REPO=${REPO:-$WORK/MedicalGPT_agent}
cd $REPO
python training/scripts/verifier/build_verifier_data.py
python -m pytest -q tests/test_discriminator.py
rm -rf $WORK/data_verifier && mkdir -p $WORK/data_verifier
cp -r training/data/verifier/medicalgpt/train $WORK/data_verifier/train
cp -r training/data/verifier/medicalgpt/val $WORK/data_verifier/val
wc -l $WORK/data_verifier/*/data.jsonl
# 检查：train 里 C / D（失败）应占 25%~50%；太少就加 --synthetic-ratio，太多说明裁判偏严
