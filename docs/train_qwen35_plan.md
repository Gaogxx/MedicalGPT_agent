# 本地模型换 Qwen3.5 基座：重训方案（MedicalGPT 最新版）

脚本在 [`training/qwen35/`](../training/qwen35/)，按编号顺序执行即可。

## 1. 选型

| 项 | 选择 | 原因 |
|---|---|---|
| 基座 | **Qwen/Qwen3.5-9B**（95 亿参数，2026-02-27 发布） | Qwen3.5 没有 7B（稠密款只有 0.8B / 2B / 4B / 9B / 27B），9B 是最接近原 7B 的一档；Qwen3.6 目前只有 27B 和 35B-A3B，单卡 24GB 训不动 |
| 备选 | Qwen/Qwen3.5-4B | 显存或推理延迟更紧时用；未微调的 Qwen3.5-2B 在 80 条真实问题上已达到 81.3% 自动通过率 |
| 训练框架 | MedicalGPT main（v2.7.0 之后，2026-09-15 提交 ef9985a） | v2.5 起全流程适配 Qwen3.5，v2.6 支持工具调用 SFT / DPO，v2.7 加了 OPD；依赖 transformers ≥ 5.6、trl ≥ 0.29、peft ≥ 0.19.1 |
| 训练方式 | QLoRA（4-bit），LoRA rank 16 | 9B 的 bf16 权重约 19GB，单卡 24GB 放不下 LoRA 训练的激活；48GB 卡可改 bf16 LoRA |
| 推理 | vLLM ≥ 0.17 + 本仓库对话模板 + hermes 解析器 | 见第 2 节 |
| 数据 | v3：SFT 20,663 条 / DPO 2,971 对 | 最长样本 1,529 token，`model_max_length` 取 2048 |

## 2. 先解决的问题：训练格式和推理格式要一致

2026-09 那版模型上线后出现过工具调用 JSON 泄漏到正文：训练时的工具调用写法和 vLLM 解析器期望的写法不一致。换 Qwen3.5 时这个问题会更明显：

- MedicalGPT 对 Qwen3.5 的 `tool_format` 仍是 Qwen2.5 的 JSON 写法：`<tool_call>{"name": …, "arguments": …}</tool_call>`；
- Qwen3.5 自带的对话模板是 XML 写法：`<tool_call><function=search_drug><parameter=name>…`，vLLM 官方推荐的 `qwen3_coder` 解析器解析的也是这种。

如果用 MedicalGPT 默认方式训练、用 vLLM 默认模板上线，两边对不上。本方案的做法：

1. `export_formats.py` 把 v3 数据导出成两份：OpenAI 格式（messages + tools，和线上请求同构）和 MedicalGPT 预渲染格式（工具说明、调用、结果已经渲染成文本）；
2. MedicalGPT 用 `--template_name qwen3_5_nothink` 训练，**不传 `--tool_format`**；
3. vLLM 用 [`runtime/vllm/medicalgpt_qwen35_tools.jinja`](../runtime/vllm/medicalgpt_qwen35_tools.jinja) 作对话模板、`hermes` 作工具解析器；
4. `tests/test_chat_template.py` 检查：OpenAI 格式经模板渲染后，与 MedicalGPT 训练时拼出的文本逐字相同。已用 MedicalGPT 源码和 Qwen3.5 分词器核对，1,062 条 SFT 验证样本的 token 序列全部一致。

## 3. 执行顺序

| 步骤 | 脚本 | 做什么 | 检查点 |
|---|---|---|---|
| 0 | `00_env.sh` | 拉 MedicalGPT 最新代码、建 Python 3.11 环境、装 bitsandbytes 和 Gated DeltaNet 的 CUDA 内核、下载 Qwen3.5-9B | 打印出 transformers ≥ 5.6、trl ≥ 0.29 |
| 1 | `01_data.sh` | 导出两种格式、跑一致性测试、拷到训练目录 | 测试全部通过；SFT 19,601 / 1,062，DPO 2,765 / 206 |
| 2 | `02_sft.sh` | QLoRA SFT，2 epoch，按 eval_loss 取最优 | 开头日志里解码出的样本以医疗系统提示开头；工具样本里有 `<tool_call>` 和 `<tool_response>`；train / eval loss 不分叉 |
| 3 | `03_dpo.sh` | 合并 SFT 权重 → 在其上训 DPO adapter（170 步 ≈ 1 epoch） | eval 的 rewards/accuracies 不应很快到 1.0、margin 不应冲到 10 左右（v0 的过拟合信号） |
| 4 | `04_merge.sh` | 合并 DPO adapter 得到最终权重 | 输出目录里保留 Qwen3.5 原始 config |
| 5 | `05_serve.sh` | vLLM 部署在 :8000，模型名 `medical-agent` | 冒烟请求返回标准 `tool_calls`，正文里没有 `<tool_call>` |
| 6 | `06_eval.sh` | `evaluation/run_live_eval.py`：80 条 dev 问题走完整编排，与 2026-09 MedicalGPT-7B 同指标对比 | 结果写入 `evaluation/results/live_qwen3.5-9b.json` |

可选第 7 步：OPD 蒸馏（`MedicalGPT/scripts/run_opd.sh`），学生用本模型，教师用 Qwen3.5-27B（4-bit 加载）；单卡 24GB 放不下 9B 学生 + 27B 教师，需要 48GB 以上。

## 4. 上线切换

`configs/models.yaml` 的 local 档不用改：端口仍是 8000、模型名仍是 `medical-agent`，网关与 DSH 配置都不变。
网关会把 DSH 的 `mcp__medical__search_drug` 去掉前缀再发给模型，与训练数据里的工具名一致。

## 5. 和旧方案的差异

| | 2026-09（旧） | 本方案 |
|---|---|---|
| 基座 | Qwen2.5-7B-Instruct | Qwen3.5-9B |
| MedicalGPT | v2.7 之前 | main@ef9985a |
| 工具调用格式 | `--tool_format default --template_name qwen`，vLLM 默认模板 + hermes | 预渲染数据 + 同一份 Jinja 模板，训练 / 推理逐字一致（有测试） |
| 系统提示 | 训练时无（默认 “You are a helpful assistant.”） | 训练与线上用同一份医疗系统提示 |
| 验证集 | 与训练集相同 | 按问法模板分组切出的独立验证集 |
| 数据 | SFT 4,945 / DPO 1,500 对（去重后 65 对） | SFT 20,663 / DPO 2,971 对 |
