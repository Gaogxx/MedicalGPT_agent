#!/bin/bash
# 判别模型 第 3 步：Qwen3.5-2B + LoRA SFT（bf16，2B 很小，不用 4-bit）。输出只有一个字母，训练很快
# 与 9B 同一套格式约定：--template_name qwen3_5_nothink、system 已在导出时包好、不传 --tool_format
set -e
WORK=${WORK:-/root/autodl-tmp}; D=$WORK/data_verifier
source /root/miniconda3/etc/profile.d/conda.sh && conda activate medgpt35
cd $WORK/MedicalGPT
CUDA_VISIBLE_DEVICES=0 python training/supervised_finetuning.py \
    --model_name_or_path $WORK/models/Qwen3.5-2B \
    --train_file_dir $D/train --validation_file_dir $D/val \
    --template_name qwen3_5_nothink \
    --model_max_length 3072 \
    --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 2 \
    --do_train --do_eval \
    --use_peft True --target_modules all --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
    --num_train_epochs 3 --learning_rate 1e-4 --weight_decay 0.01 --warmup_ratio 0.05 \
    --eval_strategy steps --eval_steps 50 --save_strategy steps --save_steps 50 --save_total_limit 3 \
    --load_best_model_at_end True --metric_for_best_model eval_loss --greater_is_better False \
    --bf16 --torch_dtype bfloat16 --gradient_checkpointing True \
    --logging_steps 10 --report_to tensorboard \
    --output_dir $WORK/outputs/verifier-qwen3.5-2b 2>&1 | tee $WORK/outputs/verifier-qwen3.5-2b.log
# 检查：开头日志 “Decode input_ids[0]” 以 <|im_start|>system\n你是医疗问答质检员 开头，label 部分只有一个字母 + <|im_end|>
python tools/merge_peft_adapter.py --base_model $WORK/models/Qwen3.5-2B \
    --lora_model $WORK/outputs/verifier-qwen3.5-2b --output_dir $WORK/models/qwen3.5-2b-verifier
