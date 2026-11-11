#!/bin/bash
# 第 0 步：环境（AutoDL 单卡，推荐 80GB；24GB 也能跑 SFT / DPO，见各脚本的 QLORA 开关；CUDA 12.x）。MedicalGPT 用最新 main（v2.7.0 之后，已适配 Qwen3.5 / Qwen3.6）。
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
# Qwen3.5-2B：线上小模型档，也用来给判别模型产出待判的回答（training/verifier/01）
huggingface-cli download Qwen/Qwen3.5-2B --local-dir $WORK/models/Qwen3.5-2B
# 部署与评测用单独的环境 serve：vLLM 会固定自己的 torch / transformers 版本，和训练环境装在一起容易冲突
REPO=${REPO:-$WORK/MedicalGPT_agent}
conda create -n serve python=3.11 -y || true
conda activate serve
pip install "vllm>=0.17"
pip install -e "$REPO[dev]" || pip install -e "$REPO"   # 本项目（网关 / 编排器 / 评测脚本）
# 之后：训练（02_sft / 03_dpo / 04_opd / 05_merge）在 medgpt35 环境；
#       部署与评测（06_serve / 07_eval / 08_router、verifier/01 02 05）在 serve 环境
# 判别模型 Laya 单独一个环境：它固定自己的 torch / transformers 版本
conda create -n laya python=3.11 -y || true
conda activate laya
pip install "laya[serve]"
pip install -e "$REPO"
python -c "from laya.train import resolve_checkpoint_dir; print(resolve_checkpoint_dir('multilingual'))"   # 预下载 Laya 多语言版
#       判别模型的训练与部署（verifier/03 04）在 laya 环境
