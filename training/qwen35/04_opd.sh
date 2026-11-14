#!/bin/bash
# 第 4 步：OPD 在线蒸馏（MedicalGPT training/opd_training.py，基于 TRL GKDTrainer，经 opd_prefix_teacher.py 启动）
#   学生：DPO 合并后的 Qwen3.5-9B（默认 bf16 LoRA；24GB 卡设 QLORA=1）；教师：Qwen3.5-27B（4-bit，冻结）。词表相同（248,320）。
#   数据：只用问答类样本（real_qa / safety / clarify 共 2,000 条），工具调用轨迹不蒸馏——
#         通用教师没见过本项目的工具约定，蒸馏工具轨迹会把 SFT/DPO 学到的调用格式往回拉。
#   教师提示与学生不同（2026-11-14 教师检查后改）：
#     - 学生模板 qwen3_5_nothink 没有空思考块，27B 看到会先“Thinking Process”，打分整体错位 → 教师侧补 <think>\n\n</think>\n\n；
#     - 关掉思考后教师很少点名药物（药名片段上教师 logp 比学生低 0.83/token，其余 0.57），但会写长篇 Markdown、
#       还会说“我需要先调用工具查询”（系统提示要求先查工具，OPD 数据却不带工具）→ 用 TEACHER_SYSTEM_EXTRA 只给教师加规则。
#   opd_lambda=0.8：80% 样本用学生当场生成的回答（教师直接纠正学生自己的说法，如“可以口服某某丸”），
#     20% 用数据参考回答（参考回答也只是上下文，损失仍是向教师分布对齐）；opd_beta=0.5：JSD。
#   2026-11-14 RTX PRO 6000（96GB）试跑：BS=4 GA=1（4 条一批生成）约 25 秒/步、峰值 66GB，500 步约 3.5 小时；
#   BS=1 GA=4 约 67 秒/步、峰值 45GB（显存不够 70GB 时用这组）。生成必须开 KV 缓存，见 opd_prefix_teacher.py。
# 不做 OPD：直接跑 05_merge.sh SKIP_OPD=1。
set -eo pipefail  # 训练命令后面接了 tee，不加 pipefail 时训练报错整个脚本仍返回 0
WORK=${WORK:-/root/autodl-tmp}; D=$WORK/data_qwen35; REPO=${REPO:-$WORK/MedicalGPT_agent}
source /root/miniconda3/etc/profile.d/conda.sh && conda activate medgpt35
export TRL_EXPERIMENTAL_SILENCE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TEACHER_SYSTEM_EXTRA=${TEACHER_SYSTEM_EXTRA-"补充要求（本轮）：本轮没有可用工具，不要说要调用或查询工具，直接根据医学常识回答。回答控制在 300 字以内，用简短段落或短列表，不用 Markdown 标题和加粗。不点名推荐具体药品（包括中成药、汤剂）和剂量，需要用药时说明请医生或药师根据检查决定。信息不足时最多追问 2 个最关键的问题。"}
[ -d $WORK/models/Qwen3.5-27B ] || modelscope download --model Qwen/Qwen3.5-27B --local_dir $WORK/models/Qwen3.5-27B
if [ "${TEACHER_OFFLOAD:-0}" = "1" ]; then TEACHER="--teacher_load_in_4bit False --teacher_device_map auto"; else TEACHER="--teacher_load_in_4bit True"; fi
if [ "${QLORA:-0}" = "1" ]; then STUDENT_Q="--load_in_4bit True --qlora True"; else STUDENT_Q=""; fi
OUT=${OUT:-$WORK/outputs/opd-qwen3.5-9b-medical}
# 断点续训：RESUME=$OUT/checkpoint-100 之类（日志追加写，不覆盖上一段）
if [ -n "${RESUME:-}" ]; then RESUME_ARGS="--resume_from_checkpoint $RESUME"; TEE="tee -a"; else RESUME_ARGS=""; TEE="tee"; fi
cd $WORK/MedicalGPT
CUDA_VISIBLE_DEVICES=0 python $REPO/training/qwen35/opd_prefix_teacher.py \
    --model_name_or_path $WORK/models/qwen3.5-9b-medical-dpo \
    --teacher_model_name_or_path $WORK/models/Qwen3.5-27B $TEACHER \
    --train_file_dir $D/opd_train --validation_file_dir $D/opd_val \
    --template_name qwen3_5_nothink \
    --max_prompt_length 1024 --max_new_tokens 384 --temperature 0.9 \
    --opd_lambda 0.8 --opd_beta 0.5 \
    --per_device_train_batch_size ${BS:-4} --per_device_eval_batch_size 1 --gradient_accumulation_steps ${GA:-1} \
    --do_train --do_eval \
    --use_peft True $STUDENT_Q \
    --target_modules all --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
    --max_steps ${MAX_STEPS:-500} --learning_rate 5e-5 --warmup_steps 20 \
    --eval_strategy steps --eval_steps ${EVAL_STEPS:-100} --save_steps ${EVAL_STEPS:-100} --save_total_limit 3 --logging_steps ${LOG_STEPS:-5} \
    --bf16 True --torch_dtype bfloat16 --gradient_checkpointing True --report_to tensorboard \
    --output_dir $OUT $RESUME_ARGS 2>&1 | $TEE $OUT.log
# 500 步 × 4 = 2,000 条，正好过一遍 OPD 数据。看 eval loss（JSD）是否下降；
# 跑完后用 07_eval.sh 和 DPO 版本对比，工具类指标（药品事实覆盖、格式干净率）不能降，降了就不用 OPD 版本
