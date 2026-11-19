#!/bin/bash
# 医药 Wiki 每月更新（GPU 机器上跑；建议 cron：0 3 1 * * $HOME/medagent_kb/repo/scripts/wiki_cycle.sh >> $HOME/medagent_kb/logs/cycle.log 2>&1）
#
#   1. 拉 openFDA 新导出（没更新且线上缺口不多就只做复核和体检）
#   2. 在候选目录全量重编译：说明书字段（按剂型）→ 要点 → 疾病—药物链接复核 → 全部字段逐段独立复核 → 渲染 Wiki → 体检
#   3. 用最终 9B 端到端评测“现行版”和“候选版”（药品说明书题 + 某病用药 / 相互作用题 + 原 296 题），过门槛才替换
#   4. 替换后在仓库里留下改动，由人提交（不自动推送）
#
# 环境变量（都有默认值）：W 工作目录、HUATUO Huatuo 百科问答 train、OPENFDA 说明书导出目录、COMPILER 编译模型目录、
# MEDICAL 最终 9B 目录、GPU 用哪张卡
set -eo pipefail
W=${W:-$HOME/medagent_kb}; R=${R:-$W/repo}; C=$W/cand; L=$W/logs/cycle_$(date +%Y%m%d); mkdir -p $L
HUATUO=${HUATUO:-$W/huatuo/train.jsonl}; OPENFDA=${OPENFDA:-$W/openfda}
COMPILER=${COMPILER:-$W/models/Qwen3.5-27B-FP8}; MEDICAL=${MEDICAL:-$W/models/qwen3.5-9b-medical-v5}
PY=${PY:-$W/venv/bin/python}; export PATH=$(dirname $PY):$PATH CUDA_VISIBLE_DEVICES=${GPU:-1}
wait_up() { for i in $(seq 1 180); do curl -s -m 5 localhost:$1/v1/models | grep -q "$2" && return 0; sleep 10; done; return 1; }
stop_vllm() { for p in $(pgrep -f "[v]llm serve"); do kill $p; done; for i in $(seq 1 60); do pgrep -f "[v]llm serve" >/dev/null || break; sleep 2; done; sleep 5; }
K=medagent/knowledge

# ---------- 1. 原始资料
set +e; $PY $R/scripts/fetch_openfda.py --dir $OPENFDA > $L/fetch.log 2>&1; FDA=$?; set -e
cd $R && $PY -m $K.wiki gaps --names-out $L/gap_names.json > $L/gaps.log
NGAP=$($PY -c "import json;print(len(json.load(open('$L/gap_names.json'))))")
echo "openFDA 更新=$([ $FDA = 0 ] && echo 是 || echo 否)，线上缺口药名 $NGAP 个" | tee $L/summary.txt

# ---------- 2. 候选版
rsync -a --delete --exclude .git --exclude .venv --exclude .cache $R/ $C/
ln -sfn $R/.cache $C/.cache
stop_vllm
setsid nohup vllm serve $COMPILER --served-model-name compiler --port 8100 --max-model-len 12288 \
    --gpu-memory-utilization 0.85 --max-num-seqs 64 --language-model-only > $L/compiler.log 2>&1 < /dev/null &
wait_up 8100 compiler
cd $C
if [ $FDA = 0 ] || [ "$NGAP" -gt 20 ]; then
  $PY -m $K.drug_label_kb --huatuo $HUATUO --openfda-dir $OPENFDA --llm http://127.0.0.1:8100/v1 --model compiler \
      --pilot-names $L/gap_names.json --out $C/$K/drug_kb.json --review-out $L/review.json > $L/compile.log 2>&1
  $PY -m $K.drug_label_kb --summarize --llm http://127.0.0.1:8100/v1 --model compiler --out $C/$K/drug_kb.json > $L/summarize.log 2>&1
fi
$PY -m $K.wiki_check verify-links --llm http://127.0.0.1:8100/v1 --model compiler > $L/verify_links.log 2>&1
$PY -m $K.wiki_check audit --llm http://127.0.0.1:8100/v1 --model compiler --n 100000 --huatuo $HUATUO \
    --translations $([ -f $L/review.json ] && echo $L/review.json || echo $W/review_full.json) > $L/audit.log 2>&1
$PY -m $K.wiki build > $L/build.log 2>&1
set +e; $PY -m $K.wiki_check check > $L/check.log 2>&1; HARD=$?; set -e
stop_vllm
if [ $HARD != 0 ]; then echo "候选版体检有硬性问题，不替换（见 $L/check.log）" | tee -a $L/summary.txt; exit 1; fi

# ---------- 3. 端到端评测：现行版 vs 候选版
setsid nohup vllm serve $MEDICAL --served-model-name medical-agent --port 8000 \
    --chat-template $R/runtime/vllm/medicalgpt_qwen35_tools.jinja --enable-auto-tool-choice --tool-call-parser hermes \
    --max-model-len 8192 --gpu-memory-utilization 0.9 --max-num-seqs 64 --language-model-only > $L/serve9b.log 2>&1 < /dev/null &
wait_up 8000 medical-agent
export MEDAGENT_VERIFIER_URL= MEDAGENT_REWARD_URL= MEDAGENT_WIKI_GAPS=0
pids=()
for side in base cand; do
  D=$([ $side = base ] && echo $R || echo $C)
  for set in druglabel wiki all; do
    (cd $D && $PY evaluation/run_live_eval.py --set $set --temperature 0 --out $L/live_${set}_$side.json > $L/live_${set}_$side.log 2>&1) &
    pids+=($!)
  done
done
wait "${pids[@]}"
stop_vllm
cd $C
$PY evaluation/score_drug_label_eval.py base=$L/live_druglabel_base.json cand=$L/live_druglabel_cand.json --out=$L/druglabel_score.json > /dev/null
$PY evaluation/score_wiki_eval.py base=$L/live_wiki_base.json cand=$L/live_wiki_cand.json --out=$L/wiki_score.json > /dev/null
set +e
$PY -m $K.wiki_check gate --druglabel $L/druglabel_score.json --wiki $L/wiki_score.json \
    --kb-base $L/live_all_base.json --kb-cand $L/live_all_cand.json | tee $L/gate.txt
GATE=${PIPESTATUS[0]}; set -e

# ---------- 4. 替换（只换知识库文件，代码不动）
if [ $GATE = 0 ]; then
  for f in drug_kb.json wiki_link_checks.json wiki_quarantine.json; do [ -f $C/$K/$f ] && cp $C/$K/$f $R/$K/$f; done
  rsync -a --delete --exclude reviews.jsonl $C/$K/wiki/ $R/$K/wiki/
  echo "已替换为候选版；请在 $R 检查 git diff 后提交" | tee -a $L/summary.txt
else
  echo "候选版没过门槛，现行版不变（见 $L/gate.txt）" | tee -a $L/summary.txt
fi
