#!/bin/bash
# 判别模型 第 5 步：在判别器验证集上定阈值（写入 medagent/verify/artifacts/discriminator.json），
# 再在 dev 80（人工标注的真实线上回答）上与规则、零样本选型结果对比，并看失败类别判得准不准
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate serve
WORK=${WORK:-/root/autodl-tmp}; REPO=${REPO:-$WORK/MedicalGPT_agent}
cd $REPO
# 阈值只在最终 9B 的验证回答上定（--val-model final9b）：线上判的就是它的回答，混合验证集以基座模型为主，阈值会偏低、误报多
if [ -d training/data/verifier/laya_v3 ]; then
  python evaluation/run_discriminator_eval.py --url http://127.0.0.1:8002 --val training/data/verifier/laya_v3/val.jsonl --val-model final9b
else
  python evaluation/run_discriminator_eval.py --url http://127.0.0.1:8002
fi
# 看：dev80 的 laya_auroc_rules_pass 要明显高于零样本 Laya（0.53），最好不低于零样本 Skywork（0.91）；
#     cascade F1 高于纯规则（0.73）；false_alarms 不明显增加
# 校验器默认接本机 :8002（MEDAGENT_VERIFIER_URL 不配置即可；设成空值 = 只用规则），再做 9B（qwen35/01~07）和路由器（qwen35/08）
