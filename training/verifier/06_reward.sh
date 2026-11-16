#!/bin/bash
# 判别模型 第 6 步（校验器 v4，2026-11-16）：奖励模型打分服务 + 阈值校准
#   Skywork-Reward-V2-Qwen3-1.7B 零样本给（问题, 回答）打分，部署在 :8003；校验顺序 规则 → Laya（:8002）→ 奖励模型
#   为什么：Laya v3 对 Medical-GPT 自己的失败几乎无效（AUROC 0.543），Skywork 1.7B 零样本 0.818（漏答 0.846、不当用药 0.805）
#   显存：bf16 约 4GB，可与 vLLM、laya-serve 同卡；Mac 上用 MPS 也能跑
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate laya     # 有 torch / transformers / starlette / uvicorn
WORK=${WORK:-/root/autodl-tmp}; REPO=${REPO:-$WORK/MedicalGPT_agent}
cd $REPO
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
python training/verifier/reward_serve.py --port 8003 > $WORK/logs/reward_serve.log 2>&1 &
until curl -sf localhost:8003/health > /dev/null; do sleep 5; done
# 阈值：在 Medical-GPT 的路由数据集 train 回答上按拦截比例定（block 3% / review 10%），写 medagent/verify/artifacts/reward_model.json
# （需要 answers_final9b.jsonl，见 08_router.sh 第 1 步；在 serve 环境里跑）
conda activate serve
python evaluation/run_reward_eval.py --url http://127.0.0.1:8003
# 网关 / 编排器默认连 :8003（MEDAGENT_REWARD_URL，设成空值 = 不用奖励模型）
