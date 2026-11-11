#!/bin/bash
# 判别模型 第 1 步：两个基座模型在路由数据集 train / val（1,737 题）上作答，百川裁判对照参考答案打质量等级
#   回答来源：Qwen3.5-2B（失败多）+ Qwen3.5-9B（失败少），都是未微调的基座 → 好坏样本都有
#   裁判：默认 strong 档（百川，.env 里的 BAICHUAN_API_KEY），约 3,500 次调用、一次性；
#         不想花 API 费：JUDGE="--judge-url http://127.0.0.1:8013/v1 --judge-model Qwen/Qwen3.5-27B"（本机 vLLM 部署 27B，质量不如百川）
#   中断后重跑会接着做
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate serve
WORK=${WORK:-/root/autodl-tmp}; REPO=${REPO:-$WORK/MedicalGPT_agent}
cd $REPO && mkdir -p $WORK/outputs
# 基座模型用 Qwen3.5 自带模板 + qwen3_coder 解析器（XML 写法的工具调用）
vllm serve $WORK/models/Qwen3.5-2B --served-model-name Qwen/Qwen3.5-2B --port 8011 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --max-model-len 8192 --gpu-memory-utilization 0.15 --language-model-only > $WORK/outputs/vllm_base2b.log 2>&1 &
P1=$!
vllm serve $WORK/models/Qwen3.5-9B --served-model-name Qwen/Qwen3.5-9B --port 8012 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --max-model-len 8192 --gpu-memory-utilization 0.45 --language-model-only > $WORK/outputs/vllm_base9b.log 2>&1 &
P2=$!
trap "kill $P1 $P2 2>/dev/null" EXIT
for port in 8011 8012; do until curl -sf localhost:$port/v1/models > /dev/null; do sleep 5; done; done
python training/scripts/verifier/collect_answers.py --base-url http://127.0.0.1:8011/v1 --model Qwen/Qwen3.5-2B \
    --tag base2b --judge strong --concurrency 8 $JUDGE &
C1=$!
python training/scripts/verifier/collect_answers.py --base-url http://127.0.0.1:8012/v1 --model Qwen/Qwen3.5-9B \
    --tag base9b --judge strong --concurrency 8 $JUDGE &
C2=$!
wait $C1 $C2
# 检查：两个文件各约 1,700 行，levels 里 judge_error / unparsed 应接近 0
