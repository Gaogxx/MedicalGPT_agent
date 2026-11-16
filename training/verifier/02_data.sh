#!/bin/bash
# 判别模型 第 2 步：真实回答 + 裁判标签（等级 + 失败类别）+ 语义故障注入 → Laya 训练格式；跑一致性测试
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate serve
WORK=${WORK:-/root/autodl-tmp}; REPO=${REPO:-$WORK/MedicalGPT_agent}
cd $REPO
# 有最终 9B 的采集结果（answers_final9b.jsonl，08_router.sh 第 1 步）时用 v3 配方：线上判的是它的回答，
# 它的真实样本 ×2、其中答砸的再 ×4，故障注入比例 0.35 → 0.05（2026-11-16：dev 80 级联 F1 0.49 → 0.76）
if [ -s training/data/verifier/answers_final9b.jsonl ]; then
  python training/scripts/verifier/build_verifier_data.py --oversample final9b:2 --oversample-fail final9b:4 \
      --synthetic-ratio 0.05 --out-dir training/data/verifier/laya_v3
else
  python training/scripts/verifier/build_verifier_data.py
fi
python -m pytest -q tests/test_discriminator.py
wc -l training/data/verifier/laya*/*.jsonl
# 检查：train 里答砸（poor / unsafe）应占 25%~50%；太少就加 --synthetic-ratio，太多说明裁判偏严
