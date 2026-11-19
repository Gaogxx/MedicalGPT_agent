#!/bin/bash
# 第 5 步：按工具结果作答（2026-11-20）。在第 4 步的最终 9B 上再训一个小 LoRA，只训“拿到工具结果之后怎么答”。
# 为什么：药品库 / 医药 Wiki 接上以后，9B 查到了也常答得含糊（药品说明书题覆盖率 0.196），某病用药约三成药名不是工具给的，
#   相互作用要分清“说明书没提到”和“没有相互作用”。调不调工具、调哪个已经很稳（第 4 步），这一步不动它。
# 流程（两张 L20 48G，约 1 小时）：
#   g1 出题（和三套评测的药 / 病 / 药对零重叠）→ g2 现 9B 跑完整编排收集轨迹 → g3 两个 27B 各写一半示范回答，
#   数字 / 药名 / 相互作用措辞做确定性核对 → g4 双卡数据并行 LoRA（只训最后一段回答，限时 28 分钟）→ g5 合并
# 之后用 run_live_eval.py 把新旧模型各放一张卡并排评测（--set wiki / druglabel / all）。
set -eo pipefail
W=${W:-$HOME/medagent_kb}; R=${R:-$W/repo}; M=$W/models; D=$R/training/data/grounding; L=$W/logs/g5; mkdir -p $L $D
OLD=$M/qwen3.5-9b-medical; NEW=$M/qwen3.5-9b-medical-v5; AD=$W/outputs/grounding-qwen3.5-9b-medical
PY=${PY:-$W/venv/bin/python}; export PATH=$(dirname $PY):$PATH MEDAGENT_WIKI_GAPS=0 MEDAGENT_VERIFIER_URL= MEDAGENT_REWARD_URL=
wait_up() { for i in $(seq 1 180); do curl -s -m 5 localhost:$1/v1/models | grep -q "$2" && return 0; sleep 10; done; return 1; }
stop_vllm() { for p in $(pgrep -f "[v]llm serve"); do kill $p; done; for i in $(seq 1 60); do pgrep -f "[v]llm serve" >/dev/null || break; sleep 2; done; sleep 5; }
cd $R
[ -s $D/queries.jsonl ] || $PY training/qwen35/grounding/g1_queries.py --huatuo ${HUATUO:-$W/huatuo/train.jsonl} --out $D/queries.jsonl
# g2：两张卡各起一个 9B，各收一半
$PY - <<PYEOF
rows=open("$D/queries.jsonl",encoding="utf-8").read().splitlines()
open("$D/q_0.jsonl","w").write("\n".join(rows[0::2])+"\n"); open("$D/q_1.jsonl","w").write("\n".join(rows[1::2])+"\n")
PYEOF
stop_vllm
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g setsid nohup vllm serve $OLD --served-model-name medical-agent --port 800$g \
      --chat-template $R/runtime/vllm/medicalgpt_qwen35_tools.jinja --enable-auto-tool-choice --tool-call-parser hermes \
      --max-model-len 8192 --gpu-memory-utilization 0.9 --max-num-seqs 64 --language-model-only > $L/serve$g.log 2>&1 < /dev/null &
done
wait_up 8000 medical-agent; wait_up 8001 medical-agent
for g in 0 1; do
  $PY training/qwen35/grounding/g2_collect.py --base-url http://127.0.0.1:800$g/v1 --queries $D/q_$g.jsonl --out $D/traj_$g.jsonl --conc 32 \
      > $L/collect$g.log 2>&1 & eval p$g=\$!
done
wait $p0 $p1; stop_vllm
# g3：两个 27B 各写一半
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g setsid nohup vllm serve $M/Qwen3.5-27B-FP8 --served-model-name compiler --port 810$g --max-model-len 12288 \
      --gpu-memory-utilization 0.85 --max-num-seqs 64 --language-model-only > $L/compiler$g.log 2>&1 < /dev/null &
done
wait_up 8100 compiler; wait_up 8101 compiler
for g in 0 1; do
  $PY training/qwen35/grounding/g3_teacher.py --traj $D/traj_$g.jsonl --out $D/sft_$g.jsonl --llm http://127.0.0.1:810$g/v1 \
      > $L/teacher$g.log 2>&1 & eval p$g=\$!
done
wait $p0 $p1; cat $D/sft_0.jsonl $D/sft_1.jsonl > $D/sft.jsonl; stop_vllm
# g4：双卡数据并行（每卡累积 8 步，等效批 16），最多 2,200 条、3,072 token、28 分钟
CUDA_VISIBLE_DEVICES=0,1 $(dirname $PY)/torchrun --nproc_per_node 2 training/qwen35/grounding/g4_sft.py --model $OLD --data $D/sft.jsonl \
    --out $AD --accum 8 --max-samples 2200 --max-len 3072 --time-budget 28 > $L/train.log 2>&1
# g5：合并 + 逐层核对
CUDA_VISIBLE_DEVICES=1 $PY training/qwen35/grounding/g5_merge.py $OLD $AD $NEW > $L/merge.log 2>&1
echo "完成：$NEW"
