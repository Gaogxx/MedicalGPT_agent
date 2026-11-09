#!/bin/bash
# 第 4 步：合并 DPO adapter → 最终权重（merge_peft_adapter.py 会按多模态架构保留 Qwen3.5 原始 config，vLLM 可直接加载）
set -e
WORK=${WORK:-/root/autodl-tmp}
source /root/miniconda3/etc/profile.d/conda.sh && conda activate medgpt35
cd $WORK/MedicalGPT
python tools/merge_peft_adapter.py --base_model $WORK/models/qwen3.5-9b-medical-sft \
    --lora_model $WORK/outputs/dpo-qwen3.5-9b-medical --output_dir $WORK/models/qwen3.5-9b-medical
ls -lh $WORK/models/qwen3.5-9b-medical
