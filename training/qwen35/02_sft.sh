#!/bin/bash
# 第 2 步：SFT（QLoRA，单卡 24GB）。48GB 卡可去掉 --load_in_4bit/--qlora 改 bf16 LoRA。
# 关键：--template_name qwen3_5_nothink、不传 --tool_format（数据已预渲染，和 vLLM 模板逐字一致）
set -e
WORK=${WORK:-/root/autodl-tmp}; D=$WORK/data_qwen35
# QLORA=1（默认）：4-bit QLoRA，24GB 卡可跑；48GB / 80GB 卡设 QLORA=0 改 bf16 LoRA（更快，精度更高）
if [ "${QLORA:-1}" = "1" ]; then QLORA_ARGS="--load_in_4bit True --qlora True"; else QLORA_ARGS=""; fi
source /root/miniconda3/etc/profile.d/conda.sh && conda activate medgpt35
cd $WORK/MedicalGPT
CUDA_VISIBLE_DEVICES=0 python training/supervised_finetuning.py \
    --model_name_or_path $WORK/models/Qwen3.5-9B \
    --train_file_dir $D/sft_train --validation_file_dir $D/sft_val \
    --template_name qwen3_5_nothink \
    --model_max_length 2048 \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 8 \
    --do_train --do_eval \
    --use_peft True $QLORA_ARGS \
    --target_modules all --lora_rank 16 --lora_alpha 32 --lora_dropout 0.1 \
    --num_train_epochs 2 --learning_rate 1e-4 --weight_decay 0.01 --warmup_ratio 0.03 \
    --neft_alpha 5 \
    --eval_strategy steps --eval_steps 100 --save_strategy steps --save_steps 100 --save_total_limit 3 \
    --load_best_model_at_end True --metric_for_best_model eval_loss --greater_is_better False \
    --bf16 --torch_dtype bfloat16 --gradient_checkpointing True \
    --logging_steps 10 --report_to tensorboard \
    --output_dir $WORK/outputs/sft-qwen3.5-9b-medical 2>&1 | tee $WORK/outputs/sft-qwen3.5-9b-medical.log
# 训练开头日志里的 “Decode input_ids[0]” 应以 <|im_start|>system\n你是一名专业、谨慎的医疗健康助手 开头，
# 工具样本里应出现 <tool_call>\n{"name": ...} 与 <tool_response>
