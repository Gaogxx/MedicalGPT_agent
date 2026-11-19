#!/bin/bash
# 从 GitHub Release（models-2026-11-20）的 LoRA 附件还原最终的 Qwen3.5-9B 医疗模型（约 18GB，不直接放附件）。
#   链路：Qwen3.5-9B 基座 + SFT → + DPO v3 → + OPD → + 工具修补 v2 → + 按工具结果作答（第 5 步，2026-11-20）= qwen3.5-9b-medical-v5
#   每一步都是 remap_lora_for_vl.py 改参数名 → peft 合并 → 逐层核对增量（合并是确定性的）；
#   中间结果 m-sft / m-dpo / m-opd / qwen3.5-9b-medical（第 4 步后的上一版最终模型）也会留在 $M 里，用不到可以删。
#   最后与训练机上的最终模型比对权重分片的 md5（SHA_md5.txt；同一 transformers 版本下逐字节一致）。
# 需要：Python 环境里有 torch、transformers（≥5.20）、peft、safetensors；约 110GB 磁盘（基座 + 5 份合并结果；CLEAN=1 时约 40GB）；gh 或 curl。
# 每一步合并完都按 SHA_md5.txt 核对该步结果的分片 md5，对不上就停。
# 用法：WORK=/path/to/work bash training/qwen35/rebuild_from_release.sh
set -eo pipefail
WORK=${WORK:-$HOME/medagent_rebuild}; REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}; M=$WORK/models; R=$WORK/release
TAG=${TAG:-models-2026-11-20}; PY=${PY:-python}
BASE=${BASE:-$M/Qwen3.5-9B}
mkdir -p $R $M && cd $R
for a in lora-1-sft lora-2-dpo-v3 lora-3-opd lora-4-toolfix-v2 lora-5-grounding SHA_md5; do
  f=$a.tar; [ $a = SHA_md5 ] && f=SHA_md5.txt
  [ -f $f ] || gh release download $TAG --repo Gaogxx/MedicalGPT_agent --pattern "$f" \
    || curl -fL -o $f https://github.com/Gaogxx/MedicalGPT_agent/releases/download/$TAG/$f
  [ $a = SHA_md5 ] || tar xf $f
done
[ -f $BASE/config.json ] || $PY -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-9B', local_dir='$BASE')"
CLEAN=${CLEAN:-0}     # CLEAN=1：每合完一步就核对并删掉上一步的中间结果，峰值磁盘约 40GB
merge() {  # merge <base> <adapter> <out>
  [ -f $3/config.json ] && { echo "已有 $3，跳过"; return; }
  $PY $REPO/training/qwen35/grounding/g5_merge.py $1 $2 $3
  (cd $M && grep " $(basename $3)/" $R/SHA_md5.txt | md5sum -c)
  [ $CLEAN = 1 ] && [ $1 != $BASE ] && rm -rf $1 || true
}
merge $BASE                          $R/sft-qwen3.5-9b-medical          $M/m-sft
merge $M/m-sft                       $R/dpo-qwen3.5-9b-medical-v3       $M/m-dpo
merge $M/m-dpo                       $R/opd-qwen3.5-9b-medical          $M/m-opd
merge $M/m-opd                       $R/toolfix-qwen3.5-9b-medical-v2   $M/qwen3.5-9b-medical
merge $M/qwen3.5-9b-medical          $R/grounding-qwen3.5-9b-medical    $M/qwen3.5-9b-medical-v5
echo "还原完成：$M/qwen3.5-9b-medical-v5（部署见 06_serve.sh，--chat-template runtime/vllm/medicalgpt_qwen35_tools.jinja）"
