#!/bin/bash
# SFT 训练：Qwen2.5-7B-Instruct + LoRA + 工具调用数据
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
cd /root/autodl-tmp/MedicalGPT

python training/supervised_finetuning.py \
    --model_name_or_path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
    --train_file_dir /root/autodl-tmp/MedicalGPT/data/sft \
    --validation_file_dir /root/autodl-tmp/MedicalGPT/data/sft \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --use_peft True \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.03 \
    --model_max_length 1024 \
    --output_dir /root/autodl-tmp/MedicalGPT/outputs/sft-qwen2.5-7b-agent \
    --bf16 \
    --gradient_checkpointing True \
    --torch_dtype bfloat16 \
    --tool_format default \
    --template_name qwen \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500
echo "=== SFT 训练完成 ==="
