#!/bin/bash
# 第 1 步：数据。训练数据以 OpenAI 格式为准（training/data/openai_v3，入库）；
#   MedicalGPT 的输入由它生成：工具说明 / 调用 / 结果按上线用的 vLLM 模板预渲染，另抽 2,000 条问答类样本给 OPD
set -e
REPO=${REPO:-/root/autodl-tmp/MedicalGPT_agent}
WORK=${WORK:-/root/autodl-tmp}
cd $REPO
python training/scripts/data_pipeline/export_formats.py            # openai_v3 → export_v3/medicalgpt_qwen35/{sft,dpo,opd}
python -m pytest -q tests/test_chat_template.py                    # 训练文本 == 上线渲染文本，不一致就别开训
D=$WORK/data_qwen35
E=training/data/export_v3/medicalgpt_qwen35
for k in sft dpo opd; do
  mkdir -p $D/${k}_train $D/${k}_val
  cp $E/$k/train.jsonl $D/${k}_train/ && cp $E/$k/val.jsonl $D/${k}_val/
done
python $WORK/MedicalGPT/tools/validate_jsonl.py --file_path $D/sft_train/train.jsonl || true
wc -l $D/*/*.jsonl
