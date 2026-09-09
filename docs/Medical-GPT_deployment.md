# 部署文档

本文档描述 Medical-GPT 医疗大模型 Agent 的环境要求、启动步骤与 API 文档。

## 1. 环境要求

### 1.1 硬件

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| GPU | RTX 3090 (24GB) × 1 | A100 (40/80GB) × 1 或 RTX 4090 × 2 |
| 内存 | 32GB | 64GB |
| 磁盘 | 100GB SSD | 200GB NVMe SSD |
| 网络 | 能访问 HuggingFace | 能访问 HuggingFace + GitHub |

> 无 GPU 时：训练用 AutoDL 租卡（3090/4090 约 1.5~3 元/小时）；推理运行时可先用 `--mock` 模式（免 GPU）联调。

### 1.2 软件

| 组件 | 版本 |
|------|------|
| Python | 3.10 |
| PyTorch | 2.x（按 CUDA 版本选择） |
| CUDA | 11.8 / 12.x |
| Node.js | 20+（仅 DeepSeek Harness 路径需要） |
| pnpm | 最新（仅 dsh 路径需要） |

## 2. 训练侧部署

```bash
# 1. 克隆项目
git clone <本仓库> && cd Medical
git clone https://github.com/shibing624/MedicalGPT.git

# 2. 建环境
conda create -n medgpt python=3.10 -y && conda activate medgpt
cd MedicalGPT && pip install -r requirements.txt && cd ..
pip install -r training/requirements.txt   # 本项目附加依赖(openai 等)

# 3. 验证
python -c "import torch; print(torch.cuda.is_available())"
python training/scripts/validate_data.py --dir training/data/sft

# 4. 训练
cd training/scripts
bash run_sft.sh        # SFT（核心）
bash run_dpo.sh        # DPO（核心）
bash merge_lora.sh     # 合并导出
cd ../..
```

训练完成后，模型位于 `models/medical-agent-qwen2.5-7b/`。

## 3. 运行时侧部署

### 3.1 方式一：Python FastAPI 运行时（推荐联调）

```bash
cd runtime/server
pip install -r requirements.txt

# 免 GPU 联调（mock 后端，不接真实模型）
python main.py --backend mock --port 8000

# 接 vLLM（需先启动 vLLM，见 3.3）
python main.py --backend vllm --base-url http://localhost:8000/v1 --model medical-agent
```

### 3.2 方式二：DeepSeek Harness 插件

```bash
# 安装 Node 20+ 与 pnpm
cd runtime/dsh-medical-plugins
npm install && npm run build

# 启动 dsh（需 vLLM 已在运行，配置见 3.3 / config）
npx @deepseek-ai/dsh web --no-open
# 访问 http://127.0.0.1:3080
```

### 3.3 模型服务（vLLM）

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./models/medical-agent-qwen2.5-7b \
    --served-model-name medical-agent \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --port 8000
```

验证：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "medical-agent", "messages": [{"role": "user", "content": "我头痛3天，有点恶心"}]}'
```

## 4. API 文档（Python FastAPI 运行时）

### 4.1 健康检查

```
GET /health
```

响应：

```json
{"status": "ok", "backend": "mock"}
```

### 4.2 对话（非流式）

```
POST /chat
Content-Type: application/json
```

请求体：

```json
{
  "messages": [
    {"role": "user", "content": "我头痛3天，有点恶心"}
  ],
  "session_id": "optional-session-id",
  "temperature": 0.3
}
```

响应：

```json
{
  "session_id": "abc123",
  "answer": "根据症状分析……建议就医。",
  "tool_calls": [
    {"name": "check_symptom", "arguments": {"symptom": "头痛伴恶心", "duration": "3天"}}
  ],
  "policy_notes": []
}
```

字段说明：

| 字段 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | 会话 ID，不传则服务端生成 |
| `answer` | string | 最终回答 |
| `tool_calls` | array | 本次触发的工具调用轨迹 |
| `policy_notes` | array | 安全策略触发的提示（如紧急就医建议） |

### 4.3 对话（流式 SSE）

```
POST /chat/stream
Content-Type: application/json
Accept: text/event-stream
```

请求体同 `/chat`。响应为 SSE 事件流：

```
event: tool_call
data: {"name": "check_symptom", "arguments": {...}}

event: token
data: {"delta": "根据症状"}

event: done
data: {"session_id": "abc123"}
```

### 4.4 工具列表

```
GET /tools
```

响应：

```json
[
  {"name": "check_symptom", "description": "根据症状分析可能原因并提供初步建议"},
  {"name": "search_drug", "description": "查询药品信息"},
  {"name": "check_interaction", "description": "检查药品相互作用"},
  {"name": "search_guideline", "description": "检索临床指南"}
]
```

## 5. 审计日志

运行时会将完整会话轨迹写入 `logs/audit.jsonl`，每行一条事件：

```json
{"ts": "2026-09-07T10:00:00Z", "session_id": "abc123", "type": "tool_call", "tool": "check_symptom", "arguments": {"symptom": "头痛"}}
{"ts": "2026-09-07T10:00:01Z", "session_id": "abc123", "type": "tool_result", "tool": "check_symptom", "result": {"possible_causes": ["偏头痛"]}}
{"ts": "2026-09-07T10:00:02Z", "session_id": "abc123", "type": "policy_intervention", "reason": "high_risk_requires_approval"}
{"ts": "2026-09-07T10:00:03Z", "session_id": "abc123", "type": "final_answer", "content": "……"}
```

## 6. 常见问题

| 问题 | 排查 |
|------|------|
| vLLM 启动报 OOM | 降低 `--max-model-len`，或加 `--gpu-memory-utilization 0.85` |
| dsh 插件不加载 | 检查 `package.json` 的 `dsh.plugin` 字段与 Node 版本 |
| mock 模式不返回工具调用 | 确认使用 `--backend mock`，mock 后端内置规则模拟工具调用 |
| 训练 loss 不下降 | 先跑 `validate_data.py` 校验数据格式 |
| 工具调用解析失败 | 检查 LLM 输出是否为 `Action: name\nAction Input: {...}` 格式 |
