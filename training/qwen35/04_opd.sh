#!/bin/bash
# 第 4 步：OPD 在线蒸馏（MedicalGPT training/opd_training.py，基于 TRL GKDTrainer）
#   学生：DPO 合并后的 Qwen3.5-9B（4-bit QLoRA）；教师：Qwen3.5-27B（4-bit，冻结）。两者词表相同（248,320），满足 logits 蒸馏要求。
#   数据：只用问答类样本（real_qa / safety / clarify / raft 共 2,000 条），工具调用轨迹不蒸馏——
#         通用教师没见过本项目的工具约定，蒸馏工具轨迹会把 SFT/DPO 学到的调用格式往回拉。
#   opd_lambda=0.5：一半样本用学生自己当场生成的回答（on-policy），一半用数据里的参考回答；opd_beta=0.5：JSD。
#   显存：学生约 6GB + 教师约 15GB + 生成与激活，单卡 24GB 放不下，建议 48GB（A6000 / L20）。
#         只有 24GB 时：设 TEACHER_OFFLOAD=1，教师按 device_map=auto 部分放 CPU（2026-09 原项目 32B 教师就是这样跑的，很慢）。
# 不做 OPD：直接跑 05_merge.sh SKIP_OPD=1。
set -e
WORK=${WORK:-/root/autodl-tmp}; D=$WORK/data_qwen35
source /root/miniconda3/etc/profile.d/conda.sh && conda activate medgpt35
export TRL_EXPERIMENTAL_SILENCE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -d $WORK/models/Qwen3.5-27B ] || huggingface-cli download Qwen/Qwen3.5-27B --local-dir $WORK/models/Qwen3.5-27B
if [ "${TEACHER_OFFLOAD:-0}" = "1" ]; then TEACHER="--teacher_load_in_4bit False --teacher_device_map auto"; else TEACHER="--teacher_load_in_4bit True"; fi
cd $WORK/MedicalGPT
CUDA_VISIBLE_DEVICES=0 python training/opd_training.py \
    --model_name_or_path $WORK/models/qwen3.5-9b-medical-dpo \
    --teacher_model_name_or_path $WORK/models/Qwen3.5-27B $TEACHER \
    --train_file_dir $D/opd_train --validation_file_dir $D/opd_val \
    --template_name qwen3_5_nothink \
    --max_prompt_length 1024 --max_new_tokens 384 --temperature 0.9 \
    --opd_lambda 0.5 --opd_beta 0.5 \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 4 \
    --do_train --do_eval \
    --use_peft True --load_in_4bit True --qlora True \
    --target_modules all --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
    --max_steps 500 --learning_rate 5e-5 --warmup_steps 20 \
    --eval_strategy steps --eval_steps 100 --save_steps 100 --save_total_limit 3 --logging_steps 5 \
    --bf16 True --torch_dtype bfloat16 --gradient_checkpointing True --report_to tensorboard \
    --output_dir $WORK/outputs/opd-qwen3.5-9b-medical 2>&1 | tee $WORK/outputs/opd-qwen3.5-9b-medical.log
# 500 步 × 4 = 2,000 条，正好过一遍 OPD 数据。看 eval loss（JSD）是否下降；
# 跑完后用 07_eval.sh 和 DPO 版本对比，工具类指标（药品事实覆盖、格式干净率）不能降，降了就不用 OPD 版本
