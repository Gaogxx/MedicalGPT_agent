#!/usr/bin/env bash
# 合并 LoRA 权重到基础模型，导出可推理的完整权重
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

MODEL_TYPE="auto"
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
PEFT_MODEL="./outputs/dpo-qwen2.5-7b-agent"     # 最终选用的 LoRA checkpoint
OUTPUT_DIR="./models/medical-agent-qwen2.5-7b"

python ../MedicalGPT/tools/merge_peft_adapter.py \
    --model_type ${MODEL_TYPE} \
    --base_model_name_or_path ${BASE_MODEL} \
    --peft_model_path ${PEFT_MODEL} \
    --output_dir ${OUTPUT_DIR}

echo "合并完成，模型已导出到 ${OUTPUT_DIR}"
echo "验证：python ../MedicalGPT/demo/inference.py --base_model ${OUTPUT_DIR} --template_name qwen --interactive"
