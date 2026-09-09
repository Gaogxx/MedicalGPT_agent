#!/usr/bin/env bash
# OPD 蒸馏 (可选，v2.7+) —— 用更强的教师模型蒸馏能力到 7B 学生模型
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

MODEL_TYPE="auto"
MODEL_PATH="./outputs/dpo-qwen2.5-7b-agent"        # 学生模型
TEACHER_MODEL="Qwen/Qwen2.5-72B-Instruct"          # 教师模型
TRAIN_DIR="./data/sft"
OUTPUT_DIR="./outputs/opd-qwen2.5-7b-agent"

python ../MedicalGPT/training/opd_training.py \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_PATH} \
    --teacher_model_name_or_path ${TEACHER_MODEL} \
    --train_file_dir ${TRAIN_DIR} \
    --per_device_train_batch_size 1 \
    --max_source_length 2048 \
    --max_target_length 1024 \
    --do_train \
    --use_peft True \
    --lora_rank 16 \
    --lora_alpha 32 \
    --num_train_epochs 1 \
    --learning_rate 5e-5 \
    --output_dir ${OUTPUT_DIR} \
    --bf16 \
    --gradient_checkpointing True \
    --torch_dtype bfloat16 \
    --tool_format default \
    --template_name qwen
