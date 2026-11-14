#!/bin/bash
# 从 GitHub Release（models-2026-11-14）的 LoRA 附件还原最终的 Qwen3.5-9B 医疗模型（约 18GB，不直接放附件）。
#   链路：Qwen3.5-9B 基座 + SFT → + DPO v3 → + OPD → + 工具修补 v2 = qwen3.5-9b-medical
#   每一步都是 MedicalGPT 的 merge_peft_adapter.py（先 remap_lora_for_vl.py 改参数名，合并后逐元素核对）。
#   合并是确定性的，最后与训练机上的最终模型比对两个权重分片的 md5。
# 需要：medgpt35 环境与 MedicalGPT 代码（00_env.sh）、约 75GB 磁盘（基座 + 4 份中间 / 最终权重，可边合边删）、gh 或 curl。
set -eo pipefail
WORK=${WORK:-/root/autodl-tmp}; REPO=${REPO:-$WORK/MedicalGPT_agent}; M=$WORK/models; R=$WORK/release
TAG=${TAG:-models-2026-11-14}
source /root/miniconda3/etc/profile.d/conda.sh && conda activate medgpt35
mkdir -p $R && cd $R
for a in lora-1-sft lora-2-dpo-v3 lora-3-opd lora-4-toolfix-v2; do
  [ -f $a.tar ] || gh release download $TAG --repo Gaogxx/MedicalGPT_agent --pattern "$a.tar" \
    || curl -fL -o $a.tar https://github.com/Gaogxx/MedicalGPT_agent/releases/download/$TAG/$a.tar
  tar xf $a.tar
done
[ -f $M/Qwen3.5-9B/config.json ] || modelscope download --model Qwen/Qwen3.5-9B --local_dir $M/Qwen3.5-9B
cd $WORK/MedicalGPT
merge() {  # merge <base> <adapter> <out>
  python $REPO/training/qwen35/remap_lora_for_vl.py $2 $2-vl
  python tools/merge_peft_adapter.py --base_model $1 --lora_model $2-vl --output_dir $3
  python $REPO/training/qwen35/remap_lora_for_vl.py --check $1 $3 $2
}
merge $M/Qwen3.5-9B               $R/sft-qwen3.5-9b-medical          $M/qwen3.5-9b-medical-sft
merge $M/qwen3.5-9b-medical-sft   $R/dpo-qwen3.5-9b-medical-v3       $M/qwen3.5-9b-medical-dpo
merge $M/qwen3.5-9b-medical-dpo   $R/opd-qwen3.5-9b-medical          $M/qwen3.5-9b-medical-opd
merge $M/qwen3.5-9b-medical-opd   $R/toolfix-qwen3.5-9b-medical-v2   $M/qwen3.5-9b-medical
cd $M/qwen3.5-9b-medical
md5sum -c <<'EOF'
f61baabd53b3b9ab0b5856110df0b20d  model-00001-of-00002.safetensors
2c63c7832fc2151cd2bdfcc9b9b2503c  model-00002-of-00002.safetensors
EOF
echo "还原完成：$M/qwen3.5-9b-medical（部署见 06_serve.sh）"
