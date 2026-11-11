#!/bin/bash
# 第 7 步：评测（与 2026-09 的 MedicalGPT-7B 用同一套指标对比；做了 OPD 时，DPO 版和 OPD 版各跑一次）
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate serve
REPO=${REPO:-/root/autodl-tmp/MedicalGPT_agent}
cd $REPO
# 1) 让 configs/models.yaml 的 local 档指向新模型（端口 / model 名不变即可：127.0.0.1:8000 / medical-agent）
# 2) dev 80 + 扩充集（共 296 条）走完整编排（MCP 工具 + 校验 + 安全关），固定 local 档；dev 80 部分与 2026-09 MedicalGPT-7B 同指标对比
#    加 --judge strong 再请百川按参考答案打质量等级（约 296 次调用）
python evaluation/run_live_eval.py --base-url http://127.0.0.1:8000/v1 --model medical-agent \
    --out evaluation/results/live_qwen3.5-9b.json
# 3) 工具调用格式：泄漏率应为 0（content 里不应出现 <tool_call> / {"name": ）
python -m pytest -q tests/test_chat_template.py
