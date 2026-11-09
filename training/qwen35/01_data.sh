#!/bin/bash
# 第 1 步：数据。v3（SFT 20,663 / DPO 2,971）→ MedicalGPT 预渲染格式（工具说明 / 调用 / 结果已按上线模板渲染）
set -e
REPO=${REPO:-/root/autodl-tmp/MedicalGPT_agent}
WORK=${WORK:-/root/autodl-tmp}
cd $REPO
python training/scripts/data_pipeline/export_formats.py            # → training/data/export_v3/{openai,medicalgpt_qwen35}
python -m pytest -q tests/test_chat_template.py                    # 训练文本 == 上线渲染文本，不一致就别开训
D=$WORK/data_qwen35
mkdir -p $D/sft_train $D/sft_val $D/dpo_train $D/dpo_val
cp training/data/export_v3/medicalgpt_qwen35/sft/train.jsonl $D/sft_train/
cp training/data/export_v3/medicalgpt_qwen35/sft/val.jsonl   $D/sft_val/
cp training/data/export_v3/medicalgpt_qwen35/dpo/train.jsonl $D/dpo_train/
cp training/data/export_v3/medicalgpt_qwen35/dpo/val.jsonl   $D/dpo_val/
python $WORK/MedicalGPT/tools/validate_jsonl.py --file_path $D/sft_train/train.jsonl || true
wc -l $D/*/*.jsonl
