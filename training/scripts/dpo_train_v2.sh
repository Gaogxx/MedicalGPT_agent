#!/bin/bash
# DPO v2：防过拟合 / 防 reward hacking 版本（对照 v0 的 dpo_train.sh）
#   v0 问题：--eval_strategy no；1500 对里去重后只有 65 对、12 种模板，跑了约 3 个 epoch，
#            rewards/accuracies 0.975→1.0、margins≈10、loss 0.0069 —— 典型的记住训练对
#   v2 改动：去重 + 反事实对（330 对）、独立 val、约 1 epoch、beta 0.1→0.2、RPO（叠加 SFT NLL，防止 chosen 似然一起掉）
#   需要先执行一次：python $REPO/training/scripts/patch_medicalgpt_dpo.py /root/autodl-tmp/MedicalGPT/training/dpo_training.py
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
REPO=${REPO:-/root/autodl-tmp/MedicalGPT_agent}
cd /root/autodl-tmp/MedicalGPT

mkdir -p /root/autodl-tmp/data_v2/dpo_train /root/autodl-tmp/data_v2/dpo_val
cp $REPO/training/data/reward_${DATA_VER:-v3}/train.jsonl /root/autodl-tmp/data_v2/dpo_train/
cp $REPO/training/data/reward_${DATA_VER:-v3}/val.jsonl   /root/autodl-tmp/data_v2/dpo_val/
N=$(wc -l < /root/autodl-tmp/data_v2/dpo_train/train.jsonl)
STEPS=$(( (N + 3) / 4 ))   # batch 1 × grad_accum 4 → 约 1 个 epoch

python training/dpo_training.py \
    --model_name_or_path /root/autodl-tmp/MedicalGPT/models/sft-v2-merged \
    --train_file_dir /root/autodl-tmp/data_v2/dpo_train \
    --validation_file_dir /root/autodl-tmp/data_v2/dpo_val \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 4 \
    --do_train --do_eval \
    --use_peft True --lora_rank 16 --lora_alpha 32 \
    --max_steps $STEPS \
    --learning_rate 5e-6 \
    --dpo_beta 0.2 --dpo_rpo_alpha 0.5 \
    --max_source_length 1024 --max_target_length 512 \
    --eval_strategy steps --eval_steps 20 --save_steps 20 \
    --output_dir /root/autodl-tmp/MedicalGPT/outputs/dpo-v2-qwen2.5-7b-agent \
    --bf16 --fp16 False --gradient_checkpointing True --torch_dtype bfloat16 \
    --tool_format default --template_name qwen --logging_steps 5
echo "=== DPO v2 完成：eval rewards/accuracies 若 >0.95 且 eval loss 回升，说明仍在过拟合，减 steps 或加大 beta ==="
