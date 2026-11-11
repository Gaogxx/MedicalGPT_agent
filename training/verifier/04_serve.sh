#!/bin/bash
# 判别模型 第 4 步：laya-serve 部署在 :8002，模型名 medical-verifier（接口 POST /v1/systemone，与 Jev 的接口格式兼容）
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate laya
WORK=${WORK:-/root/autodl-tmp}
pip install -q "laya[serve]"
LAYA_HOST=127.0.0.1 LAYA_PORT=8002 \
LAYA_EXTRA_MODELS="{\"medical-verifier\": \"$WORK/models/laya-medical-verifier\"}" \
LAYA_MODELS=medical-verifier LAYA_DEFAULT_MODEL=medical-verifier \
    laya-serve
# 网关 / 编排器 / 标注脚本通过 .env 接入：MEDAGENT_VERIFIER_URL=http://127.0.0.1:8002
# 冒烟：curl -s localhost:8002/health   → "loaded": ["medical-verifier"]
