#!/bin/bash
# 第 0 步：环境（AutoDL 单卡，推荐 80GB；24GB 也能跑 SFT / DPO，见各脚本的 QLORA 开关；CUDA 12.x）。MedicalGPT 用最新 main（v2.7.0 之后，已适配 Qwen3.5 / Qwen3.6）。
set -e
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
WORK=${WORK:-/root/autodl-tmp}
export PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}   # 2026-11-13 实测：清华 14MB/s，AutoDL 默认的阿里云镜像 0.7MB/s
export PIP_CACHE_DIR=$WORK/.cache/pip HF_HOME=$WORK/.cache/huggingface        # 缓存放数据盘，系统盘只有 30GB
cd $WORK
[ -d MedicalGPT ] || git clone https://github.com/shibing624/MedicalGPT.git
cd MedicalGPT && git log -1 --format='MedicalGPT %h %cd'
# MedicalGPT@ef9985a + transformers 5.x：TrainingArguments.fsdp 默认值变成 None，len(None) 报错
sed -i "s/len(training_args.fsdp) > 0/len(training_args.fsdp or []) > 0/" training/supervised_finetuning.py training/pretraining.py
cd $WORK
conda create -n medgpt35 python=3.11 -y || true
source /root/miniconda3/etc/profile.d/conda.sh && conda activate medgpt35
# dpo_training.py 会把所有 requires_grad 参数转 fp32（为 fp16 报错准备的）；bf16 LoRA 时整个 9B 会变成约 36GB，32GB 卡直接爆，改成只在 fp16 时转
python - <<'EOF'
p = "MedicalGPT/training/dpo_training.py"; s = open(p).read()
old = "    for param in filter(lambda p: p.requires_grad, model.parameters()):\n        param.data = param.data.to(torch.float32)"
new = "    if args.fp16:  # bf16 时不转：全量转 fp32 会让 9B 占约 36GB\n        for param in filter(lambda p: p.requires_grad, model.parameters()):\n            param.data = param.data.to(torch.float32)"
if old in s and "if args.fp16:  # bf16" not in s:
    open(p, "w").write(s.replace(old, new)); print("patched dpo_training.py")
EOF
pip install torch                                     # 2026-11-13 装到 2.15.0+cu132；驱动要支持 CUDA 13（RTX 50 系必须 CUDA ≥ 12.8）
pip install -r MedicalGPT/requirements.txt            # transformers>=5.6、trl>=0.29、peft>=0.19.1（实测 5.22.2 / 1.17.1 / 0.21.2）
pip install bitsandbytes                              # QLoRA（24GB 卡训 9B 必需）
# Qwen3.5 的 Gated DeltaNet 线性注意力层：装上 fla 才走 Triton 快路径，否则退回 torch 实现，训练慢数倍
pip install flash-linear-attention || echo "fla 安装失败，可继续，但会慢"
# causal-conv1d 要用和 torch 同版本的 CUDA 编译（镜像自带 12.8、torch 是 13.2 时会失败）；不装只是退回 F.conv1d，影响很小
pip install causal-conv1d --no-build-isolation || echo "causal-conv1d 安装失败，可继续"
python -c "import transformers, trl, peft; print(transformers.__version__, trl.__version__, peft.__version__)"
# 基座：Qwen3.5 没有 7B，最接近的是 9B（95 亿参数）；显存更紧可换 Qwen3.5-4B
# 用魔搭下载（2026-11-13 实测 hf-mirror 卡住不动，魔搭约 16MB/s）
pip install -U modelscope
modelscope download --model Qwen/Qwen3.5-9B --local_dir $WORK/models/Qwen3.5-9B
# Qwen3.5-2B：线上小模型档，也用来给判别模型产出待判的回答（training/verifier/01）
modelscope download --model Qwen/Qwen3.5-2B --local_dir $WORK/models/Qwen3.5-2B
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
