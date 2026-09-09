#!/usr/bin/env bash
# 增量预训练 (PT) —— 医疗知识注入
# 运行环境：AutoDL / 带 GPU 的 Linux (CUDA 11.8+, PyTorch 2.x)
# 依赖：MedicalGPT 已克隆至 ../MedicalGPT 并完成 pip install -r requirements.txt
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

MODEL_TYPE="auto"
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
TRAIN_DIR="./data/pretrain"
OUTPUT_DIR="./outputs/pt-qwen2.5-7b"

python ../MedicalGPT/training/pretraining.py \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_PATH} \
    --train_file_dir ${TRAIN_DIR} \
    --validation_file_dir ${TRAIN_DIR} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --use_peft True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --num_train_epochs 1 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --block_size 1024 \
    --output_dir ${OUTPUT_DIR} \
    --bf16 \
    --gradient_checkpointing True \
    --torch_dtype bfloat16
