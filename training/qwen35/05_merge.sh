#!/bin/bash
# 第 5 步：合并 → 最终权重 qwen3.5-9b-medical（merge_peft_adapter.py 保留 Qwen3.5 原始 config，vLLM 可直接加载）
#   链路：DPO 合并权重 + OPD adapter → qwen3.5-9b-medical-opd；再 + 工具修补 adapter（04b_toolfix.sh）→ qwen3.5-9b-medical
#   SKIP_OPD=1：没做 OPD 时直接用 DPO 合并后的权重作为最终模型
#   SKIP_TOOLFIX=1：只用 OPD（不建议：2026-11-14 实测 OPD 约 21% 调完工具还接着写正文，修补后为 0）
#   每次合并都先把参数名改成多模态结构，合并后逐元素核对（原因见 03_dpo.sh / remap_lora_for_vl.py）
set -eo pipefail
WORK=${WORK:-/root/autodl-tmp}; REPO=${REPO:-$WORK/MedicalGPT_agent}; M=$WORK/models
source /root/miniconda3/etc/profile.d/conda.sh && conda activate medgpt35
cd $WORK/MedicalGPT
merge() {  # merge <base> <adapter> <out>
  python $REPO/training/qwen35/remap_lora_for_vl.py $2 $2-vl
  python tools/merge_peft_adapter.py --base_model $1 --lora_model $2-vl --output_dir $3
  python $REPO/training/qwen35/remap_lora_for_vl.py --check $1 $3 $2
}
if [ "${SKIP_OPD:-0}" = "1" ]; then
  ln -sfn $M/qwen3.5-9b-medical-dpo $M/qwen3.5-9b-medical
else
  [ -f $M/qwen3.5-9b-medical-opd/config.json ] || merge $M/qwen3.5-9b-medical-dpo $WORK/outputs/opd-qwen3.5-9b-medical $M/qwen3.5-9b-medical-opd
  if [ "${SKIP_TOOLFIX:-0}" = "1" ]; then
    ln -sfn $M/qwen3.5-9b-medical-opd $M/qwen3.5-9b-medical
  else
    merge $M/qwen3.5-9b-medical-opd ${TOOLFIX:-$WORK/outputs/toolfix-qwen3.5-9b-medical} $M/qwen3.5-9b-medical
  fi
fi
ls -lh $M/qwen3.5-9b-medical/
