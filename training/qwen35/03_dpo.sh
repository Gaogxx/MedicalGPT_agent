#!/bin/bash
# 第 3 步：DPO（在 SFT 合并权重上训新 adapter；参考模型 = 关掉 adapter 的同一模型，不额外占显存）
# MedicalGPT 的 dpo_training.py 只认 max_steps（没有 num_train_epochs）：2,765 对 / 每步 16 对 ≈ 173 步 = 1 个 epoch
set -e
WORK=${WORK:-/root/autodl-tmp}; D=$WORK/data_qwen35
source /root/miniconda3/etc/profile.d/conda.sh && conda activate medgpt35
cd $WORK/MedicalGPT
# 先把 SFT adapter 合并成完整权重，DPO 在其上训新 adapter
python tools/merge_peft_adapter.py --base_model $WORK/models/Qwen3.5-9B \
    --lora_model $WORK/outputs/sft-qwen3.5-9b-medical --output_dir $WORK/models/qwen3.5-9b-medical-sft
CUDA_VISIBLE_DEVICES=0 python training/dpo_training.py \
    --model_name_or_path $WORK/models/qwen3.5-9b-medical-sft \
    --train_file_dir $D/dpo_train --validation_file_dir $D/dpo_val \
    --template_name qwen3_5_nothink \
    --max_source_length 1024 --max_target_length 512 \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 16 \
    --do_train --do_eval \
    --use_peft True --load_in_4bit True --qlora True \
    --target_modules all --lora_rank 8 --lora_alpha 16 --lora_dropout 0.05 \
    --max_steps 170 --learning_rate 5e-6 --warmup_steps 15 \
    --eval_strategy steps --eval_steps 25 --save_steps 25 --logging_steps 5 \
    --bf16 True --fp16 False --torch_dtype bfloat16 --gradient_checkpointing True \
    --remove_unused_columns False --report_to tensorboard \
    --output_dir $WORK/outputs/dpo-qwen3.5-9b-medical 2>&1 | tee $WORK/outputs/dpo-qwen3.5-9b-medical.log
# 盯住 eval 的 rewards/accuracies 与 margins：v0 曾出现 accuracy 1.0、margin≈10（过拟合），出现就提前停在更早的 checkpoint
# 合并 DPO adapter → OPD 的学生模型（跳过 OPD 时它就是最终模型，见 05_merge.sh）
python tools/merge_peft_adapter.py --base_model $WORK/models/qwen3.5-9b-medical-sft \
    --lora_model $WORK/outputs/dpo-qwen3.5-9b-medical --output_dir $WORK/models/qwen3.5-9b-medical-dpo
