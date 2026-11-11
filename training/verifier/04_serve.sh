#!/bin/bash
# 判别模型 第 4 步：vLLM 部署在 :8002，模型名 medical-verifier。对话模板与 9B 相同（训练 / 推理文本逐字一致）
# 2B 的 bf16 权重约 4GB，和 9B（:8000）、小模型档（:8001）放在同一张卡上
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate serve
WORK=${WORK:-/root/autodl-tmp}; REPO=${REPO:-$WORK/MedicalGPT_agent}
vllm serve $WORK/models/qwen3.5-2b-verifier \
    --served-model-name medical-verifier --port 8002 \
    --chat-template $REPO/runtime/vllm/medicalgpt_qwen35_tools.jinja \
    --max-model-len 4096 --max-logprobs 20 --gpu-memory-utilization 0.12 \
    --language-model-only
# 网关 / 编排器 / 标注脚本通过环境变量接入（写进 .env）：
#   MEDAGENT_VERIFIER_URL=http://127.0.0.1:8002/v1
# 冒烟：
# curl -s localhost:8002/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"medical-verifier",
#   "messages":[{"role":"user","content":"用户问题：布洛芬怎么吃\n\n助手回答：据我所知一次吃4片。"}],
#   "max_tokens":1,"logprobs":true,"top_logprobs":5}'
# 期望：choices[0].logprobs.content[0].top_logprobs 里是 A/B/C/D
