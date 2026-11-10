#!/bin/bash
# 第 8 步：用新模型的实测结果重训 MF 路由器（路由器学的是“本地模型哪些题会答砸”，换了模型就要重训）
#   数据：evaluation/datasets/router_set.jsonl —— 2,000 条真实问诊提问，与训练数据、所有评测集都不重叠，
#         已切好 train 1,432 / val 305 / test 263；dev 80 不参与训练，只做外部测试
#   裁判：strong 档（百川 → DeepSeek），需要 BAICHUAN_API_KEY 或 DEEPSEEK_API_KEY；中断后重跑会接着标
set -e
REPO=${REPO:-/root/autodl-tmp/MedicalGPT_agent}
cd $REPO
# 1) 本地模型逐题作答 + 校验器 + 裁判 → 标签（2,000 题，并发 8）
python evaluation/router/label_router_set.py --base-url http://127.0.0.1:8000/v1 --model medical-agent \
    --judge strong --concurrency 8 --tag qwen3.5-9b
# 2) train 训练、val 定阈值（总体 40% 交给强模型，含安全规则强制部分）、test 和 dev 80 报告
python evaluation/router/train_router.py --labels evaluation/results/router_labels_qwen3.5-9b.jsonl --budget 0.4 --tag qwen3.5-9b
# 3) 重启网关后 /healthz 里 router 字段应显示 “mf_router（独立路由数据集训练）”
