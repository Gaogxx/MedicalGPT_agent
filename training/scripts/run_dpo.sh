#!/usr/bin/env bash
# DPO 偏好优化 —— 对齐"主动调用工具/安全准确回答"的偏好 (核心)
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

MODEL_TYPE="auto"
MODEL_PATH="./outputs/sft-qwen2.5-7b-agent"   # 基于 SFT 输出
TRAIN_DIR="./data/reward"
OUTPUT_DIR="./outputs/dpo-qwen2.5-7b-agent"

python ../MedicalGPT/training/dpo_training.py \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_PATH} \
    --train_file_dir ${TRAIN_DIR} \
    --per_device_train_batch_size 2 \
    --max_source_length 2048 \
    --max_target_length 1024 \
    --do_train \
    --use_peft True \
    --lora_rank 16 \
    --lora_alpha 32 \
    --dpo_beta 0.1 \
    --num_train_epochs 2 \
    --learning_rate 5e-5 \
    --output_dir ${OUTPUT_DIR} \
    --bf16 \
    --gradient_checkpointing True \
    --torch_dtype bfloat16 \
    --tool_format default \
    --template_name qwen
