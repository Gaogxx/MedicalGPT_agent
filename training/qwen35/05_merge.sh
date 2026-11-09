#!/bin/bash
# 第 5 步：合并 OPD adapter → 最终权重 qwen3.5-9b-medical（merge_peft_adapter.py 保留 Qwen3.5 原始 config，vLLM 可直接加载）
#   SKIP_OPD=1：没做 OPD 时直接用 DPO 合并后的权重作为最终模型
set -e
WORK=${WORK:-/root/autodl-tmp}
source /root/miniconda3/etc/profile.d/conda.sh && conda activate medgpt35
cd $WORK/MedicalGPT
if [ "${SKIP_OPD:-0}" = "1" ]; then
  ln -sfn $WORK/models/qwen3.5-9b-medical-dpo $WORK/models/qwen3.5-9b-medical
else
  python tools/merge_peft_adapter.py --base_model $WORK/models/qwen3.5-9b-medical-dpo \
      --lora_model $WORK/outputs/opd-qwen3.5-9b-medical --output_dir $WORK/models/qwen3.5-9b-medical
fi
ls -lh $WORK/models/qwen3.5-9b-medical/
