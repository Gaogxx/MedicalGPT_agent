#!/bin/bash
# 第 4b 步：OPD 之后的工具调用小步 SFT（2026-11-14）
# 为什么：OPD 只用问答数据，学生在 </tool_call> 之后“该停”的习惯被冲淡——150 条带工具的验证题里，
#   OPD 第 300 / 500 步都有约 21% 调完工具还接着写正文（DPO v3 为 0），工具名一致率 150 → 146。
# 做法：SFT 训练集的工具轨迹（系统 → 用户 → <tool_call> → 工具结果 → 最终回答）只保留到第一次 <tool_call> 为止，
#   损失只落在“调哪个工具、参数、调完结束（<|im_end|>）”上；最终回答不训练，避免把 SFT 数据的回答风格带回来。
#   在 OPD 合并模型上训新的小 LoRA，步数少、学习率低；训完重跑工具组和问答组检查，两边都不能退。
# 第一版（只用 2,400 条调工具样本）矫枉过正：调完不停 32/150 → 0、工具名 150/150，但没有工具定义的问答题也 120/120 去调工具。
#   SFT 数据里带 <tools> 的样本 100% 调工具、不带的 100% 直接答，修补数据只有一种就把这个区分冲掉了。
#   第二版：1,600 条调工具（同上）+ 1,600 条不带工具的直接答，后者答案由 OPD 自己生成（toolfix_selfdistill.py），不用 SFT 原答案。
set -eo pipefail
WORK=${WORK:-/root/autodl-tmp}; D=$WORK/data_qwen35
N_TRAIN=${N_TRAIN:-1600}; N_VAL=${N_VAL:-200}; N_DIRECT=${N_DIRECT:-1600}
REPO=${REPO:-$WORK/MedicalGPT_agent}
source /root/miniconda3/etc/profile.d/conda.sh && conda activate medgpt35
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python - "$D" "$N_TRAIN" "$N_VAL" <<'EOF'
import glob, json, os, random, sys
D, n_train, n_val = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

def cut(split):
    out = []
    for f in sorted(glob.glob(f"{D}/sft_{split}/*.jsonl")):
        for line in open(f, encoding="utf-8"):
            conv = json.loads(line)["conversations"]
            if len(conv) >= 3 and conv[0]["from"] == "system" and conv[1]["from"] == "human" \
                    and conv[2]["from"] == "gpt" and conv[2]["value"].lstrip().startswith("<tool_call>"):
                out.append({"conversations": conv[:3]})
    return out

random.seed(0)
for split, n in (("train", n_train), ("val", n_val)):
    rows = cut(split)
    pick = random.sample(rows, min(n, len(rows)))
    os.makedirs(f"{D}/toolfix_{split}", exist_ok=True)
    with open(f"{D}/toolfix_{split}/{split}.jsonl", "w", encoding="utf-8") as fo:
        for r in pick:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"toolfix_{split}: 候选 {len(rows)} 条，取 {len(pick)} 条")
EOF
# 直接答样本：OPD 自己生成答案（约 1,600 条 × 384 token 上限，batch 16，RTX PRO 6000 上约半小时）
DIRECT=$D/toolfix_direct.jsonl
[ -s $DIRECT ] || python $REPO/training/qwen35/toolfix_selfdistill.py $WORK/models/qwen3.5-9b-medical-opd $D/sft_train $DIRECT $N_DIRECT 16
cat $DIRECT >> $D/toolfix_train/train.jsonl
echo "toolfix_train 合计 $(wc -l < $D/toolfix_train/train.jsonl) 条（调工具 + 直接答）"
cd $WORK/MedicalGPT
OUT=${OUT:-$WORK/outputs/toolfix-qwen3.5-9b-medical}
CUDA_VISIBLE_DEVICES=0 python training/supervised_finetuning.py \
    --model_name_or_path $WORK/models/qwen3.5-9b-medical-opd \
    --train_file_dir $D/toolfix_train --validation_file_dir $D/toolfix_val \
    --template_name qwen3_5_nothink \
    --model_max_length 2048 \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --per_device_eval_batch_size 4 \
    --do_train --do_eval \
    --use_peft True \
    --target_modules all --lora_rank 8 --lora_alpha 16 --lora_dropout 0.05 \
    --num_train_epochs 1 --learning_rate ${LR:-5e-5} --weight_decay 0.0 --warmup_steps 0.05 \
    --eval_strategy steps --eval_steps 50 --save_strategy steps --save_steps 50 --save_total_limit 3 \
    --bf16 --torch_dtype bfloat16 --gradient_checkpointing True \
    --logging_steps 10 --report_to tensorboard \
    --output_dir $OUT 2>&1 | tee $OUT.log
# 3,200 条 / 每步 16 条 = 200 步。之后用 opd_toolfix_check 比较 OPD 与 OPD+toolfix（工具组 150 条 + 问答组 120 条）
