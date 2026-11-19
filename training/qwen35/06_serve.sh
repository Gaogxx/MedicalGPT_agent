#!/bin/bash
# 第 6 步：vLLM 部署（≥0.17）。必须用本仓库的对话模板 + hermes 解析器，和训练格式对齐。
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate serve
WORK=${WORK:-/root/autodl-tmp}; REPO=${REPO:-$WORK/MedicalGPT_agent}
# 默认部署 qwen3.5-9b-medical-v5（Medical-GPT + 第 5 步“按工具结果作答”LoRA，见 09_grounding.sh；还原见 rebuild_from_release.sh）。
# 第 4 步后的 Medical-GPT：MODEL=$WORK/models/qwen3.5-9b-medical
vllm serve ${MODEL:-$WORK/models/qwen3.5-9b-medical-v5} \
    --served-model-name medical-agent --port 8000 \
    --chat-template $REPO/runtime/vllm/medicalgpt_qwen35_tools.jinja \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --max-model-len 8192 --gpu-memory-utilization 0.85 \
    --language-model-only
# --language-model-only：只加载文本部分，省掉视觉编码器显存（vLLM 不认这个参数时去掉即可）
# 冒烟测试：
# curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"medical-agent",
#   "messages":[{"role":"user","content":"布洛芬有什么副作用"}],
#   "tools":[{"type":"function","function":{"name":"search_drug","description":"查药","parameters":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}}}]}'
# 期望：choices[0].message.tool_calls[0].function.name == "search_drug"，content 里没有 <tool_call> 泄漏
