#!/bin/bash
# DPO 训练：基于 SFT 输出做偏好对齐
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
cd /root/autodl-tmp/MedicalGPT

python training/dpo_training.py \
    --model_name_or_path /root/autodl-tmp/MedicalGPT/models/sft-merged \
    --train_file_dir /root/autodl-tmp/MedicalGPT/data/reward \
    --per_device_train_batch_size 1 \
    --do_train \
    --use_peft True \
    --lora_rank 16 \
    --lora_alpha 32 \
    --max_steps 1500 \
    --learning_rate 5e-5 \
    --max_source_length 1024 \
    --max_target_length 512 \
    --output_dir /root/autodl-tmp/MedicalGPT/outputs/dpo-qwen2.5-7b-agent \
    --bf16 \
    --fp16 False \
    --eval_strategy no \
    --gradient_checkpointing True \
    --torch_dtype bfloat16 \
    --tool_format default \
    --template_name qwen \
    --logging_steps 10 \
    --save_steps 500
echo "=== DPO 训练完成 ==="
