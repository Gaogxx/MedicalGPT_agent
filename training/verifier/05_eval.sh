#!/bin/bash
# 判别模型 第 5 步：在判别器验证集上校准（温度 + 两个阈值，写入 medagent/verify/artifacts/discriminator.json），
# 再在 dev 80（人工标注的真实线上回答）上与规则、旧 TF-IDF 分类器对比
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate serve
WORK=${WORK:-/root/autodl-tmp}; REPO=${REPO:-$WORK/MedicalGPT_agent}
cd $REPO
python evaluation/run_discriminator_eval.py --url http://127.0.0.1:8002/v1
# 看三个数：dev80 的 cascade F1 要高于规则（it1：0.69）；false_alarms 不能明显增加；val 的 auroc
# 达标后把 MEDAGENT_VERIFIER_URL 写进 .env，再做 9B 的训练（qwen35/01~07）和路由（qwen35/08）
