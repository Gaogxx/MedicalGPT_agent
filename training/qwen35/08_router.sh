#!/bin/bash
# 第 8 步：用新模型的实测结果重训 MF 路由器（路由器学的是“本地模型哪些题会答砸”，换了模型就要重训）
#   数据：evaluation/datasets/router_set.jsonl —— 2,000 条真实问诊提问，与训练数据、所有评测集都不重叠，
#         已切好 train 1,432 / val 305 / test 263；dev 80 不参与训练，只做外部测试
#   裁判：strong 档（百川 → DeepSeek），需要 BAICHUAN_API_KEY 或 DEEPSEEK_API_KEY（写在仓库根目录的 .env 里，或先 export）；
#         中断后重跑会接着标。省钱时可改 --judge verifier（本地判别模型打等级，看不到参考答案，比百川弱）
#   校验器：.env 里配了 MEDAGENT_VERIFIER_URL（判别模型 Laya :8002）时，“校验器拦下”这一条也包含判别模型，与线上一致
#   省一遍裁判调用（2026-11-16 实际做法，约 2,000 次 → 约 280 次）：train / val 的标签与判别模型的训练数据用同一次采集，
#     collect_answers.py --consult auto（与线上一致）→ labels_from_answers.py 转成路由器标签 → label_router_set.py 只补 test 和裁判出错的题
#   2026-11-16 结果（最终 9B）：答砸 9.0%；test AUROC 0.584（冷启动版 0.514），40% 预算下失败召回 0.50——
#     最终 9B 答砸得少、从问题文本很难预判，路由器只比随机好一点；安全规则强制（6%）和事后校验仍是主要防线
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate serve
REPO=${REPO:-/root/autodl-tmp/MedicalGPT_agent}
cd $REPO
TAG=${TAG:-qwen3.5-9b}
# 1) train / val：复用判别模型的采集结果（没有就先采集，约 1,737 次裁判调用，结果同时进 Laya 训练数据）
A=training/data/verifier/answers_final9b.jsonl
[ -s $A ] || python training/scripts/verifier/collect_answers.py --base-url http://127.0.0.1:8000/v1 --model medical-agent \
    --tag final9b --consult auto --judge strong --concurrency 8
python evaluation/router/labels_from_answers.py --answers $A --tag $TAG
# 2) 其余（test 263 题 + 裁判出错的）：本地模型逐题作答 + 校验器 + 裁判 → 标签（跳过已有 id）
python evaluation/router/label_router_set.py --base-url http://127.0.0.1:8000/v1 --model medical-agent \
    --judge strong --concurrency 8 --tag $TAG
# 3) train 训练、val 定阈值（总体 40% 交给强模型，含安全规则强制部分）、test 和 dev 80 报告
python evaluation/router/train_router.py --labels evaluation/results/router_labels_$TAG.jsonl --budget ${BUDGET:-0.4} --tag $TAG
# 4) 重启网关后 /healthz 里 router 字段应显示 “mf_router（独立路由数据集训练）”
