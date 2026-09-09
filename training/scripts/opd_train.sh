#!/bin/bash
# OPD 蒸馏：Qwen2.5-32B 教师(bf16 + CPU offload) → Qwen2.5-7B 学生(4bit QLoRA)，完整 logits 蒸馏(GKDTrainer)
# 学生 = sft-merged(4bit) + DPO LoRA adapter(在 DPO 基础上继续蒸馏)
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export TRL_EXPERIMENTAL_SILENCE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/MedicalGPT

python training/opd_training.py \
    --model_name_or_path /root/autodl-tmp/MedicalGPT/models/sft-merged \
    --peft_path /root/autodl-tmp/MedicalGPT/outputs/dpo-qwen2.5-7b-agent \
    --teacher_model_name_or_path /root/autodl-tmp/models/Qwen2.5-32B-Instruct \
    --teacher_device_map auto \
    --teacher_load_in_8bit False \
    --load_in_4bit True \
    --qlora True \
    --train_file_dir /root/autodl-tmp/MedicalGPT/data/opd \
    --per_device_train_batch_size 1 \
    --do_train \
    --use_peft True \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --max_prompt_length 1024 \
    --opd_beta 0.5 \
    --opd_lambda 0.5 \
    --num_train_epochs 1 \
    --max_steps 1000 \
    --learning_rate 5e-5 \
    --output_dir /root/autodl-tmp/MedicalGPT/outputs/opd-qwen2.5-7b-agent \
    --bf16 True \
    --gradient_checkpointing True \
    --torch_dtype bfloat16 \
    --tool_format default \
    --template_name qwen \
    --logging_steps 5 \
    --save_steps 500
echo "=== OPD 蒸馏完成 ==="
