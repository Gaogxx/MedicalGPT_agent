#!/bin/bash
# 第 1 步：数据。训练数据以 OpenAI 格式为准（training/data/openai_v3，入库）；
#   MedicalGPT 的输入由它生成：工具说明 / 调用 / 结果按上线用的 vLLM 模板预渲染，另抽 2,000 条问答类样本给 OPD
set -e
REPO=${REPO:-/root/autodl-tmp/MedicalGPT_agent}
WORK=${WORK:-/root/autodl-tmp}
cd $REPO
pip install -e ".[dev]" -q                                           # 数据导出要 import medagent（装进 medgpt35 环境）
python training/scripts/data_pipeline/export_formats.py --from-openai training/data/openai_v3   # 入库的 OpenAI 格式 → export_v3/medicalgpt_qwen35/{sft,dpo,opd}
# （openai_v3/sft/{train,val}_consult.jsonl 是“何时调用复杂问诊”的正反例，由 consult_samples.py 生成，已入库）
python -m pytest -q tests/test_chat_template.py                    # 训练文本 == 上线渲染文本，不一致就别开训
D=$WORK/data_qwen35
E=training/data/export_v3/medicalgpt_qwen35
for k in sft dpo opd; do
  mkdir -p $D/${k}_train $D/${k}_val
  cp $E/$k/train*.jsonl $D/${k}_train/ && cp $E/$k/val*.jsonl $D/${k}_val/
done
for f in $D/sft_train/*.jsonl; do python $WORK/MedicalGPT/tools/validate_jsonl.py --file_path $f || true; done
wc -l $D/*/*.jsonl
