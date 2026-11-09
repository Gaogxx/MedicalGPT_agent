#!/bin/bash
# 第 6 步：评测（与 2026-09 的 MedicalGPT-7B、未微调 Qwen3.5-2B 用同一套指标对比）
set -e
REPO=${REPO:-/root/autodl-tmp/MedicalGPT_agent}
cd $REPO
# 1) 让 configs/models.yaml 的 local 档指向新模型（端口 / model 名不变即可：127.0.0.1:8000 / medical-agent）
# 2) 80 条 dev 问题走完整编排（MCP 工具 + 校验 + 安全关），固定 local 档，与 2026-09 MedicalGPT-7B 同指标对比
python evaluation/run_live_eval.py --base-url http://127.0.0.1:8000/v1 --model medical-agent \
    --out evaluation/results/live_qwen3.5-9b.json
# 3) 工具调用格式：泄漏率应为 0（content 里不应出现 <tool_call> / {"name": ）
python -m pytest -q tests/test_chat_template.py
