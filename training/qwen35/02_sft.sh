#!/bin/bash
# 第 2 步：SFT。关键：--template_name qwen3_5_nothink、不传 --tool_format（数据已预渲染，和 vLLM 模板逐字一致）
# QLORA=1（默认）：4-bit QLoRA，batch 2 × 累积 8，24GB 卡可跑；
# QLORA=0：bf16 LoRA，batch 1 × 累积 16（每次更新同样 16 条）。2026-11-13 在 32GB vGPU（RTX 4080 SUPER）实测：
#   bf16 峰值 23.5GB、0.54 条/秒；4-bit 峰值 28GB、0.56 条/秒——一样快，32GB 及以上的卡用 bf16（不量化，精度更高）
# 验证：每 250 步、400 条（每 100 步全量 1,142 条，单验证就要多花约 4 小时）
# MedicalGPT@ef9985a + transformers 5.22：fsdp 默认值变成 None，需要先打补丁（00_env.sh 已处理）
set -e
WORK=${WORK:-/root/autodl-tmp}; D=$WORK/data_qwen35
if [ "${QLORA:-1}" = "1" ]; then
  QLORA_ARGS="--load_in_4bit True --qlora True"; BATCH="--per_device_train_batch_size 2 --gradient_accumulation_steps 8"
else
  QLORA_ARGS=""; BATCH="--per_device_train_batch_size 1 --gradient_accumulation_steps 16"
fi
source /root/miniconda3/etc/profile.d/conda.sh && conda activate medgpt35
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $WORK/outputs
cd $WORK/MedicalGPT
CUDA_VISIBLE_DEVICES=0 python training/supervised_finetuning.py \
    --model_name_or_path $WORK/models/Qwen3.5-9B \
    --train_file_dir $D/sft_train --validation_file_dir $D/sft_val \
    --template_name qwen3_5_nothink \
    --model_max_length 2048 \
    $BATCH --per_device_eval_batch_size 2 \
    --do_train --do_eval --max_eval_samples 400 \
    --use_peft True $QLORA_ARGS \
    --target_modules all --lora_rank 16 --lora_alpha 32 --lora_dropout 0.1 \
    --num_train_epochs 2 --learning_rate 1e-4 --weight_decay 0.01 --warmup_steps 0.03 \
    --neft_alpha 5 \
    --eval_strategy steps --eval_steps 250 --save_strategy steps --save_steps 250 --save_total_limit 3 \
    --load_best_model_at_end True --metric_for_best_model eval_loss --greater_is_better False \
    --bf16 --torch_dtype bfloat16 --gradient_checkpointing True \
    --logging_steps 10 --report_to tensorboard \
    --output_dir $WORK/outputs/sft-qwen3.5-9b-medical 2>&1 | tee $WORK/outputs/sft-qwen3.5-9b-medical.log
# 训练开头日志里的 “Decode input_ids[0]” 应以 <|im_start|>system\n你是一名专业、谨慎的医疗健康助手 开头，
# 工具样本里应出现 <tool_call>\n{"name": ...} 与 <tool_response>
