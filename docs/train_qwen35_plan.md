# 本地模型换 Qwen3.5 基座：重训方案（MedicalGPT 最新版）

脚本在 [`training/qwen35/`](../training/qwen35/) 和 [`training/verifier/`](../training/verifier/)。整体顺序：

1. `qwen35/00_env.sh`：环境（训练用 `medgpt35`，部署 / 评测用 `serve`，判别模型用 `laya`）与模型下载；
2. `verifier/01~05`：**先训开源判别模型**（微调 Laya，结果校验用，替代 Jev，见 [verifier_model.md](verifier_model.md)）；
3. `qwen35/01~07`：9B 的 SFT → DPO → OPD → 合并 → 部署 → 评测；
4. `qwen35/08_router.sh`：用新模型重训路由器（校验器里已包含判别模型）。

**显卡**：推荐单张 80GB（A800 / A100 80GB）：SFT / DPO 可设 `QLORA=0` 用 bf16 LoRA，OPD 的 27B 教师（4-bit）+ 9B 学生放得下，
判别模型采集回答时两个基座模型也能同时部署。48GB 能跑完但 OPD 较紧；24GB 只适合 SFT / DPO（QLoRA），OPD 要 `TEACHER_OFFLOAD=1`，很慢。
数据盘约 200GB（9B / 2B / 27B 原始权重、三次合并的权重、本地全文索引）。

## 1. 选型

| 项 | 选择 | 原因 |
|---|---|---|
| 基座 | **Qwen/Qwen3.5-9B**（95 亿参数，2026-02-27 发布） | Qwen3.5 没有 7B（稠密款只有 0.8B / 2B / 4B / 9B / 27B），9B 是最接近原 7B 的一档；Qwen3.6 目前只有 27B 和 35B-A3B，单卡 24GB 训不动 |
| 备选 | Qwen/Qwen3.5-4B | 显存或推理延迟更紧时用；未微调的 Qwen3.5-2B 在 80 条真实问题上已达到 81.3% 自动通过率 |
| 训练框架 | MedicalGPT main（v2.7.0 之后，2026-09-15 提交 ef9985a） | v2.5 起全流程适配 Qwen3.5，v2.6 支持工具调用 SFT / DPO，v2.7 加了 OPD；依赖 transformers ≥ 5.6、trl ≥ 0.29、peft ≥ 0.19.1 |
| 训练方式 | LoRA rank 16；`QLORA=1`（默认，4-bit）或 `QLORA=0`（bf16） | 9B 的 bf16 权重约 19GB，单卡 24GB 放不下 bf16 LoRA 训练的激活，只能 QLoRA；48GB / 80GB 卡用 bf16 LoRA |
| 推理 | vLLM ≥ 0.17 + 本仓库对话模板 + hermes 解析器 | 见第 2 节 |
| 数据格式 | **OpenAI 格式为准**（`training/data/openai_v3/`） | messages + tools，与线上请求同构；MedicalGPT 的输入由它自动生成 |
| 数据量 | SFT 20,663 条 + 复杂问诊调用 880 条 / DPO 2,971 对 / OPD 2,100 条 | 最长样本 1,529 token，`model_max_length` 取 2048 |
| 复杂问诊 | `consult_samples.py`：多意图 → 调用 `complex_consult`（训练 400 / 验证 40），单一问题 → 调普通工具（400 / 40），验证集用训练没见过的两种问法 | 网关不再按规则触发，要靠模型自己判断 |
| 流程 | SFT → DPO → OPD | 与 2026-09 原项目同顺序；OPD 教师 Qwen3.5-27B（与 9B 词表相同，248,320） |

## 2. 先解决的问题：训练格式和推理格式要一致

2026-09 那版模型上线后出现过工具调用 JSON 泄漏到正文：训练时的工具调用写法和 vLLM 解析器期望的写法不一致。换 Qwen3.5 时这个问题会更明显：

- MedicalGPT 对 Qwen3.5 的 `tool_format` 仍是 Qwen2.5 的 JSON 写法：`<tool_call>{"name": …, "arguments": …}</tool_call>`；
- Qwen3.5 自带的对话模板是 XML 写法：`<tool_call><function=search_drug><parameter=name>…`，vLLM 官方推荐的 `qwen3_coder` 解析器解析的也是这种。

如果用 MedicalGPT 默认方式训练、用 vLLM 默认模板上线，两边对不上。本方案的做法：

1. 训练数据以 OpenAI 格式为准（`training/data/openai_v3/`，入库）。`export_formats.py` 从它生成 MedicalGPT 的输入：MedicalGPT 只读 `conversations` 结构，所以工具说明、调用、结果提前渲染成文本；
2. MedicalGPT 用 `--template_name qwen3_5_nothink` 训练，**不传 `--tool_format`**；
3. vLLM 用 [`runtime/vllm/medicalgpt_qwen35_tools.jinja`](../runtime/vllm/medicalgpt_qwen35_tools.jinja) 作对话模板、`hermes` 作工具解析器；
4. `tests/test_chat_template.py` 检查：OpenAI 格式经模板渲染后，与 MedicalGPT 训练时拼出的文本逐字相同。已用 MedicalGPT 源码和 Qwen3.5 分词器核对，1,062 条 SFT 验证样本的 token 序列全部一致。

以后补充数据，直接按 OpenAI 格式写进 `openai_v3/`（或另一个目录），再运行 `export_formats.py --from-openai <目录>`。

## 3. 执行顺序

| 步骤 | 脚本 | 做什么 | 检查点 |
|---|---|---|---|
| 0 | `00_env.sh` | 拉 MedicalGPT 最新代码、建 Python 3.11 环境、装 bitsandbytes 和 Gated DeltaNet 的 CUDA 内核、下载 Qwen3.5-9B | 打印出 transformers ≥ 5.6、trl ≥ 0.29 |
| 1 | `01_data.sh` | OpenAI 格式 → MedicalGPT 输入（SFT / DPO / OPD）、跑一致性测试、拷到训练目录 | 测试全部通过；SFT 19,601 + 800 / 1,062 + 80（后者是复杂问诊调用样本），DPO 2,765 / 206，OPD 2,000 / 100 |
| 2 | `02_sft.sh` | QLoRA SFT，2 epoch，按 eval_loss 取最优 | 开头日志里解码出的样本以医疗系统提示开头；工具样本里有 `<tool_call>` 和 `<tool_response>`；train / eval loss 不分叉 |
| 3 | `03_dpo.sh` | 合并 SFT 权重 → 训 DPO adapter（163 步 ≈ 1 epoch，2,605 对）→ 合并成 OPD 的学生模型 | eval 的 rewards/accuracies 不应很快到 1.0、margin 不应冲到 10 左右（v0 的过拟合信号）；合并后核对逐元素一致 |
| 4 | `04_opd.sh` | OPD 在线蒸馏（`opd_prefix_teacher.py` 启动）：学生 = DPO 后的 9B（bf16 LoRA），教师 = Qwen3.5-27B（4-bit，冻结，带空思考块和教师专属规则），500 步 | eval 的 JSD 下降；96GB 卡峰值 54GB，24GB 卡设 `QLORA=1 BS=1 GA=4` |
| 4b | `04b_toolfix.sh` | OPD 之后的工具调用小步 SFT：工具样本只训到 `<tool_call><|im_end|>`，配同样多的 OPD 自生成直接答，200 步 | 调完工具不再写正文、不误调工具，问答指标不退 |
| 5 | `05_merge.sh` | 合并 OPD adapter → `qwen3.5-9b-medical-opd`，再合并工具修补 adapter → 最终权重（`SKIP_OPD=1` / `SKIP_TOOLFIX=1` 可跳过） | 输出目录里保留 Qwen3.5 原始 config；每次合并逐元素核对 |
| 6 | `06_serve.sh` | vLLM 部署在 :8000，模型名 `medical-agent` | 冒烟请求返回标准 `tool_calls`，正文里没有 `<tool_call>` |
| 7 | `07_eval.sh` | `evaluation/run_live_eval.py`：dev 80 + 扩充集（共 296 条）走完整编排；dev 80 部分与 2026-09 MedicalGPT-7B 同指标对比 | DPO 版与 OPD 版各跑一次；OPD 版的药品事实覆盖、格式干净率不能比 DPO 版低 |
| 8 | `08_router.sh` | 用新模型重训 MF 路由器：在 2,000 条独立路由数据集上逐题作答 → 校验器 + 百川裁判打“答砸”标签 → train 训练、val 定阈值、test 与 dev 80 报告 | test 的 AUROC；阈值下总体交给强模型约 40%；网关 `/healthz` 显示加载的是 `mf_router` |

### OPD 的设计

- **为什么放在最后**：SFT 学格式和工具用法，DPO 学“先查后答、不乱开药”的偏好，OPD 用更强的教师在学生自己生成的回答上逐 token 纠正，提升问答质量。
- **教师**：Qwen3.5-27B（278 亿参数）。MedicalGPT 示例脚本里写的教师 `Qwen/Qwen3.5-7B-Instruct` 在 Hugging Face 上不存在，不能照抄。
- **只蒸馏问答类样本**（real_qa / safety / clarify / raft，抽 2,000 条）：通用教师没见过本项目的工具约定，蒸馏工具调用轨迹会把 SFT / DPO 学到的调用格式往回拉。
- **参数**：`opd_lambda 0.8`（80% 样本用学生当场生成的回答，教师直接纠正学生自己的说法），`opd_beta 0.5`（JSD），`max_new_tokens 384`。
- **教师提示与学生不同**（2026-11-14）：学生模板没有空思考块，27B 会先思考，所以只给教师补 `<think>\n\n</think>\n\n`；
  另给教师一条学生看不到的规则（本轮无工具、300 字内、不点名药品），学生学到的是教师在这条规则下的说法。
- **工具调用要补回来**：只蒸馏问答会冲淡“调完工具就停”，OPD 后用 `04b_toolfix.sh` 小步修补。
- **怎么判断有没有用**：OPD 版和 DPO 版用 `07_eval.sh` 各评一次，问答类指标升、工具类指标不降才采用 OPD 版。

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
| OPD | Qwen2.5-32B 教师（bf16 + CPU offload）蒸馏全部数据 | Qwen3.5-27B 教师（4-bit），只蒸馏问答类样本 |

## 6. 准备方案时发现的问题与解决

下面是写这套方案时对照 MedicalGPT 源码、Hugging Face 模型库和 Qwen3.5 分词器查出来的问题，每条都已在脚本或数据里处理。
真正上 GPU 训练时遇到的问题，记在第 7 节的表里。

| # | 现象 | 原因 | 解决 | 验证 |
|---|---|---|---|---|
| 1 | 找不到 Qwen3.5-7B | Qwen3.5 稠密款只有 0.8B / 2B / 4B / 9B / 27B；MedicalGPT 模板注释里列的 3B / 8B / 32B / 72B 和示例里的教师 `Qwen3.5-7B-Instruct` 都不存在 | 学生用 9B，教师用 27B | Hugging Face 接口：7B 返回 401，9B / 27B 返回 200 |
| 2 | 工具调用写法两套 | MedicalGPT 对 Qwen3.5 的 `tool_format` 是 Qwen2.5 的 JSON 写法；Qwen3.5 自带模板和 `qwen3_coder` 解析器是 XML 写法。照默认方式训练再上线，会重现 2026-09 的工具 JSON 泄漏 | 数据预渲染成 JSON 写法 + vLLM 用同一份自定义模板 + hermes 解析器 | 1,062 条验证样本 token 序列逐一比对，全部一致 |
| 3 | 指定 `--template_name` 后系统提示没有 `<|im_start|>system` 包装 | MedicalGPT 的 `_format_example` 用了自定义系统提示就原样拼接，不加包装 | 导出数据时把系统提示（含工具说明）包好 | 一致性测试 |
| 4 | 不指定模板时会混进 `<think>` 块 | Qwen3.5 自带模板在 assistant 段插入思考块，和 MedicalGPT 分段拼接、逐段加 eos 的方式对不上 | 用 `qwen3_5_nothink` 模板，训练和推理都不带思考块 | 渲染比对 |
| 5 | 旧数据没有系统提示 | 训练时用的是默认 “You are a helpful assistant.”，上线时却是医疗系统提示 | 每条样本都加上与线上相同的医疗系统提示（`medical_server.SYSTEM_PROMPT`），模板里的默认值也是它 | 测试检查两者一致 |
| 6 | DPO 的 `--num_train_epochs` 不生效 | `dpo_training.py` 的参数类里没有这个字段，且解析时忽略未知参数，不报错 | 改用 `--max_steps 170`（2,765 对 ÷ 每步 16 对 ≈ 1 epoch） | 对照源码字段列表 |
| 7 | 连续的 assistant 消息会被丢掉 | MedicalGPT 把消息配成“问—答”对，已配对后的第二条 assistant 直接跳过 | 数据里一个回合的多个工具调用合并到同一条 assistant | v3 数据里连续工具调用 0 条 |
| 8 | OPD 读 `messages` 格式时工具调用丢失 | OPD 规范化 `messages` 时只保留 role 和 content，`tool_calls` 被去掉 | OPD 用预渲染的 `conversations` 输入，且只蒸馏问答类样本 | 对照源码 |
| 9 | 多轮样本在 OPD 里拼接方式不同 | OPD 拼 prompt 时轮与轮之间不加 eos，SFT 会加 | OPD 数据只取单轮问答 | 原先抽出的 2,000 条里有 30 条两轮对话，导出时已过滤，现在全是单轮 |
| 10 | 24GB 显存不够 | 9B 的 bf16 权重约 19GB；OPD 还要同时放 27B 教师 | SFT / DPO 用 4-bit QLoRA；OPD 建议 48GB 卡，或教师部分放 CPU | 按参数量估算 |
| 11 | 自定义模板里工具调用没被渲染出来 | Jinja 的 `set` 在 for 循环里不会带到循环外 | 改用 namespace 累积 | 单元测试先失败、修复后通过 |

## 7. 训练记录（训练时填写）

按步骤跑的时候，每遇到一个问题就记一行；跑完后可以把关键几条挪到 README 的迭代时间线。

| 日期 | 步骤 | 现象（报错 / 指标） | 原因 | 解决 | 解决后的指标 |
|---|---|---|---|---|---|
| 2026-11-13 | 2 SFT | 32GB vGPU 上 bf16 LoRA 与 4-bit 一样快（0.54 vs 0.56 条/秒） | — | 32GB 及以上用 bf16（`QLORA=0`） | 最优 checkpoint-1250，eval_loss 2.333；第 2 个 epoch 不再下降 |
| 2026-11-13 | 3 DPO 合并 | 合并后的模型和原版一模一样 | 训练按纯文本结构（`model.layers.*`），`merge_peft_adapter.py` 按多模态结构加载（`model.language_model.layers.*`），PEFT 找不到对应层就静默丢掉 LoRA | `remap_lora_for_vl.py` 先改参数名再合并，合并后抽几层核对 | 增量与 B@A·α/r 的余弦 0.995～1.004 |
| 2026-11-13 | 3 DPO | bf16 LoRA 时显存直接翻倍 | `dpo_training.py` 把所有可训练参数转 fp32（为 fp16 准备的） | 补丁：只在 fp16 时转（`00_env.sh`） | 峰值 18.6GB |
| 2026-11-13 | 3 DPO | 加回 `<tools>` 后 325 对被悄悄丢掉 | `dpo_training.py` 按字符数过滤 prompt+回答 ≤ max_source+max_target | `max_source_length` 提到 2048 | 2,605 对全部保留 |
| 2026-11-13 | 3 DPO v1 / v2 | 验证准确率第一次评估就是 1.0 | 验证集里固定话术（工具失败、查无此药、急症）与训练集同句各约 20 条；262 条仍是旧 ReAct 写法、965 条工具相关样本没带 tools；76 对真实偏好的 chosen 直接推荐具体药物 | `fix_dpo_v3.py`：旧写法转 `<tool_call>`、补回 tools、话术分训练 / 验证两组（`dpo_phrases.py`）且同句限 20 条、验证集与 DPO / SFT 训练集零重叠、去掉推荐药物的偏好对 | v3：train 2,605 / val 234；在同一份新验证集上 v3 准确率 1.000（急症组 margin 中位 0.59），v2 只有 0.906（急症组 0.545） |
| 2026-11-14 | 3 DPO 合并核对 | 余弦只有 0.42～0.67，核对报失败 | DPO 学习率 5e-6，ΔW 中位只有 W 的千分之一，bf16 存盘后约 84% 元素不变 | 核对改为“与 fp32 算好再转 bf16 的理想合并逐元素一致率 > 0.99” | 一致率 1.0000；合并后 DPO margin 约保留运行时 adapter 的 73%，准确率仍 1.0 |
| 2026-11-14 | 4 OPD | 27B 教师输出 “Thinking Process…” | `qwen3_5_nothink` 模板没有空思考块，27B 看到裸的 assistant 开头会先思考；GKDTrainer 让教师和学生吃同一串 token，打分整体错位 | `opd_prefix_teacher.py`：只给教师补 `<think>\n\n</think>\n\n`，回答 token 按位置对齐后算 JSD | 补前抽看的教师回答几乎都以思考开头；补后 40 题里思考 0 条 |
| 2026-11-14 | 4 OPD | 关掉思考后教师写长篇 Markdown、说“我需要先调用工具查询” | 系统提示要求先查工具，OPD 数据却不带工具 | 只给教师加一条规则（`TEACHER_SYSTEM_EXTRA`）：本轮无工具、300 字内、不用标题加粗、不点名药品、最多追问 2 个 | 教师回答简洁，药名片段上教师 logp 比学生低 0.83 / token |
| 2026-11-14 | 4 OPD | 一启动就报 `trust_remote_code` 参数冲突 | MedicalGPT 的 `ModelArguments` 与 TRL 1.17 的 `SFTConfig` 都定义了它 | 启动器里同名参数合并为一个值 | — |
| 2026-11-14 | 4 OPD | 每步 80 秒，500 步要 11 小时 | GKDTrainer 开梯度检查点时把生成的 `use_cache` 设成 False，每生成一个 token 都重算整段 | 生成时打开 KV 缓存 + 4 条一批 | 每步约 25 秒 |
| 2026-11-14 | 4 OPD | 第 ~150 步 OOM（要 10.6GB） | 数据里的参考回答没截断，4 条一起算 JSD | 参考回答也截到 384 token，JSD 逐条算 | 峰值 54GB；从 checkpoint-100 续训到 500 步，eval JSD 0.0675 → 0.0646 |
| 2026-11-14 | 4 OPD 效果 | — | — | — | 120 条没训练过的问答：原地复读 55 → 0，写到上限没说完 47 → 0，主动点名推荐药物（人工核对）约 10 → 0 |
| 2026-11-14 | 4 OPD 副作用 | 150 条该调工具的题里 32 条调完还接着写正文（DPO 为 0）；工具名一致 150 → 146 | OPD 只用问答数据，`</tool_call>` 之后该停的习惯被冲淡 | `04b_toolfix.sh`：1,600 条只保留到第一次 `<tool_call>` 的工具样本 + 1,600 条不带工具的直接答（答案由 OPD 自己生成，`toolfix_selfdistill.py`），200 步小 LoRA | 调完还写正文 0/150，工具名 149/150，参数完全一致 132/150（与 DPO 持平），误调工具 0/25；问答组不变 |
| 2026-11-14 | 4b 工具修补第一版 | 只用工具样本时，没有工具定义的问答题也 120/120 去调工具 | SFT 数据里带 `<tools>` 的样本 100% 调工具、不带的 100% 直接答，修补数据只有一种就把区分冲掉了 | 见上一行（第二版加直接答样本） | — |
| 2026-11-14 | 6 部署 | vLLM 0.33 起 2B 报 `No available memory for the cache blocks` / `max_num_seqs exceeds available Mamba cache blocks` | 两个服务同时启动互相干扰显存估算；默认 `max_num_seqs=1024` 超过 Qwen3.5 线性注意力层的状态块数 | 先起 2B 再起 9B；`--max-num-seqs 64` | — |
| 2026-11-14 | 7 评测 | 见 README 迭代时间线 2026-11-14 两行 | | | |

常见情况的处理：

| 现象 | 先查 | 处理 |
|---|---|---|
| CUDA OOM | batch、序列长度、是否真的 4-bit 加载 | `per_device_train_batch_size 1` + 加大 `gradient_accumulation_steps`；`model_max_length` 降到 1536 |
| 训练特别慢、日志提示 fast path 不可用 | `flash-linear-attention`、`causal-conv1d` 是否装上 | 按 `00_env.sh` 重装；CUDA 版本要和 torch 匹配 |
| eval_loss 先降后升 | 是否过拟合 | 已设 `load_best_model_at_end`，取最优 checkpoint；必要时降到 1 epoch |
| DPO 的 rewards/accuracies 很快到 1.0 | 偏好对太容易区分 | 降学习率或步数，取更早的 checkpoint |
| vLLM 返回的 content 里有 `<tool_call>` | 是否带了 `--chat-template` 和 `--tool-call-parser hermes` | 两个参数都要加；`tests/test_chat_template.py` 必须通过 |
| 回答里出现 `<think>` | 模型偶尔仍按基座习惯思考 | vLLM 加 `--reasoning-parser qwen3` 把思考内容分离出去 |
| vLLM 加载合并后的模型报架构错误 | 输出目录的 config.json 是否还是 `Qwen3_5ForConditionalGeneration` | 用 MedicalGPT 的 `tools/merge_peft_adapter.py` 合并（它会保留原始架构），不要自己 `save_pretrained` |

