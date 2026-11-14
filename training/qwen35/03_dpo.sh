#!/bin/bash
# 第 3 步：DPO（在 SFT 合并权重上训新 adapter；参考模型 = 关掉 adapter 的同一模型，不额外占显存）
# MedicalGPT 的 dpo_training.py 只认 max_steps（没有 num_train_epochs）：2,605 对 / 每步 16 对 ≈ 163 步 = 1 个 epoch
# （2026-11-13/14 修过数据，见 fix_dpo_v3.py：旧 ReAct 写法转 <tool_call>、补回 tools、固定话术分训练/验证两组且同句限 20 条、
#   验证集与 DPO / SFT 训练集零重叠、去掉 chosen 直接推荐具体药物的真实偏好对）
# 注意 dpo_training.py 按“字符数”过滤 prompt+回答 ≤ max_source_length+max_target_length：带 <tools> 的样本最长 2,256 字符，
# 输入上限必须 ≥ 2048，否则 325 对会被悄悄丢掉
# 2026-11-13 在 32GB vGPU（RTX 4080 SUPER）实测 bf16 LoRA：峰值 18.6GB、12.6 秒/步（数据最长 942 token）
# 磁盘：两份合并权重各约 18GB，加上原版 9B 共约 55GB，数据盘至少 100GB
set -e
WORK=${WORK:-/root/autodl-tmp}; D=$WORK/data_qwen35; REPO=${REPO:-$WORK/MedicalGPT_agent}
# QLORA=0（默认）：bf16 LoRA，32GB 及以上；24GB 卡设 QLORA=1 改 4-bit
if [ "${QLORA:-0}" = "1" ]; then QLORA_ARGS="--load_in_4bit True --qlora True"; else QLORA_ARGS=""; fi
source /root/miniconda3/etc/profile.d/conda.sh && conda activate medgpt35
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $WORK/MedicalGPT
# SFT / DPO 按纯文本结构（Qwen3_5ForCausalLM，参数名 model.layers.*）训练，合并脚本按多模态结构加载
# （model.language_model.layers.*）。不改名的话 LoRA 权重会被静默丢弃、合出来还是原版，所以先改名、合并后核对。
merge() {  # merge <base> <adapter> <out>
  python $REPO/training/qwen35/remap_lora_for_vl.py $2 $2-vl
  python tools/merge_peft_adapter.py --base_model $1 --lora_model $2-vl --output_dir $3
  python $REPO/training/qwen35/remap_lora_for_vl.py --check $1 $3 $2
}
# 先把 SFT adapter（load_best_model_at_end 留下的最优那个）合并成完整权重，DPO 在其上训新 adapter
[ -f $WORK/models/qwen3.5-9b-medical-sft/config.json ] || \
  merge $WORK/models/Qwen3.5-9B $WORK/outputs/sft-qwen3.5-9b-medical $WORK/models/qwen3.5-9b-medical-sft
CUDA_VISIBLE_DEVICES=0 python training/dpo_training.py \
    --model_name_or_path $WORK/models/qwen3.5-9b-medical-sft \
    --train_file_dir $D/dpo_train --validation_file_dir $D/dpo_val \
    --template_name qwen3_5_nothink \
    --max_source_length 2048 --max_target_length 512 \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 16 \
    --do_train --do_eval \
    --use_peft True $QLORA_ARGS \
    --target_modules all --lora_rank 8 --lora_alpha 16 --lora_dropout 0.05 \
    --max_steps 163 --learning_rate 5e-6 --warmup_steps 13 \
    --eval_strategy steps --eval_steps 25 --save_steps 25 --logging_steps 5 \
    --bf16 True --fp16 False --torch_dtype bfloat16 --gradient_checkpointing True \
    --remove_unused_columns False --report_to tensorboard \
    --output_dir $WORK/outputs/dpo-qwen3.5-9b-medical 2>&1 | tee $WORK/outputs/dpo-qwen3.5-9b-medical.log
# 盯住 eval 的 rewards/accuracies 与 margins：v0 曾出现 accuracy 1.0、margin≈10（过拟合），出现就提前停在更早的 checkpoint
# 合并 DPO adapter → OPD 的学生模型（跳过 OPD 时它就是最终模型，见 05_merge.sh）
merge $WORK/models/qwen3.5-9b-medical-sft $WORK/outputs/dpo-qwen3.5-9b-medical $WORK/models/qwen3.5-9b-medical-dpo
