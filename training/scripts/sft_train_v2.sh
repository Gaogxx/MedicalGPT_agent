#!/bin/bash
# SFT v2：防过拟合版本（对照 v0 的 sft_train.sh）
#   v0 问题：--validation_file_dir 与训练目录相同 → eval_loss 是在训练集上算的（eval_samples=4945=train_samples）；
#            3 个 epoch，loss 在 2.25 epoch 已降到 0.41，存在记忆化风险
#   v2 改动：独立 val（按问法模板分组切分）、2 epoch + 按 eval_loss 早停/取最优、NEFTune 噪声、更高 dropout、weight decay
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
REPO=${REPO:-/root/autodl-tmp/MedicalGPT_agent}
cd /root/autodl-tmp/MedicalGPT

mkdir -p /root/autodl-tmp/data_v2/sft_train /root/autodl-tmp/data_v2/sft_val
cp $REPO/training/data/sft_${DATA_VER:-v3}/train.jsonl /root/autodl-tmp/data_v2/sft_train/
cp $REPO/training/data/sft_${DATA_VER:-v3}/val.jsonl   /root/autodl-tmp/data_v2/sft_val/

python training/supervised_finetuning.py \
    --model_name_or_path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
    --train_file_dir /root/autodl-tmp/data_v2/sft_train \
    --validation_file_dir /root/autodl-tmp/data_v2/sft_val \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --do_train --do_eval \
    --use_peft True --lora_rank 16 --lora_alpha 32 --lora_dropout 0.1 \
    --num_train_epochs 2 \
    --learning_rate 1e-4 --weight_decay 0.01 --warmup_ratio 0.03 \
    --neft_alpha 5 \
    --model_max_length 1024 \
    --eval_strategy steps --eval_steps 100 \
    --save_strategy steps --save_steps 100 --save_total_limit 3 \
    --load_best_model_at_end True --metric_for_best_model eval_loss --greater_is_better False \
    --output_dir /root/autodl-tmp/MedicalGPT/outputs/sft-v2-qwen2.5-7b-agent \
    --bf16 --gradient_checkpointing True --torch_dtype bfloat16 \
    --tool_format default --template_name qwen --logging_steps 10
echo "=== SFT v2 完成：请检查 trainer_state.json 里 best_metric 对应的 step，以及 train/eval loss 曲线是否分叉 ==="
