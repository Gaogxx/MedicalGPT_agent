#!/bin/bash
# 第 0 步：环境（AutoDL 单卡 24GB，CUDA 12.x）。MedicalGPT 用最新 main（v2.7.0 之后，已适配 Qwen3.5 / Qwen3.6）。
set -e
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
WORK=${WORK:-/root/autodl-tmp}
cd $WORK
[ -d MedicalGPT ] || git clone https://github.com/shibing624/MedicalGPT.git
cd MedicalGPT && git pull && git log -1 --format='MedicalGPT %h %cd'
conda create -n medgpt35 python=3.11 -y || true
source /root/miniconda3/etc/profile.d/conda.sh && conda activate medgpt35
pip install -r requirements.txt                       # transformers>=5.6、trl>=0.29、peft>=0.19.1
pip install bitsandbytes                              # QLoRA（24GB 卡训 9B 必需）
# Qwen3.5 的 Gated DeltaNet 线性注意力层：装上这两个才走 CUDA 快路径，否则退回 torch 实现，训练慢数倍
pip install flash-linear-attention causal-conv1d || echo "快路径内核安装失败，可继续，但会慢"
python -c "import transformers, trl, peft; print(transformers.__version__, trl.__version__, peft.__version__)"
# 基座：Qwen3.5 没有 7B，最接近的是 9B（95 亿参数）；显存更紧可换 Qwen3.5-4B
huggingface-cli download Qwen/Qwen3.5-9B --local-dir $WORK/models/Qwen3.5-9B
