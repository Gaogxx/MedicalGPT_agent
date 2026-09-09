#!/usr/bin/env bash
# 有监督微调 (SFT) —— 医疗问答 + 工具调用能力 (核心)
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

MODEL_TYPE="auto"
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"      # 若做过 PT，可改为 ./outputs/pt-qwen2.5-7b
TRAIN_DIR="./data/sft"
OUTPUT_DIR="./outputs/sft-qwen2.5-7b-agent"

python ../MedicalGPT/training/supervised_finetuning.py \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_PATH} \
    --train_file_dir ${TRAIN_DIR} \
    --validation_file_dir ${TRAIN_DIR} \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --max_source_length 2048 \
    --max_target_length 1024 \
    --do_train \
    --do_eval \
    --use_peft True \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.03 \
    --output_dir ${OUTPUT_DIR} \
    --bf16 \
    --gradient_checkpointing True \
    --torch_dtype bfloat16 \
    --tool_format default \
    --template_name qwen
