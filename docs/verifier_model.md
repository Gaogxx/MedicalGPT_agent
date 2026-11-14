# 开源判别模型：结果校验（微调 Laya，替代 Jev）

代码：`medagent/verify/typed.py`（判定定义、校验流程）、`medagent/verify/discriminator.py`（Laya 客户端、问题定义、状态格式）；
训练：`training/verifier/01~05`；选型：`evaluation/run_judge_bakeoff.py`；测试：`tests/test_discriminator.py`。

## 1. 用在哪一块

**结果校验**：模型答完、回答交给用户之前，判断这个回答是不是答砸了；答砸了就升级给更强的模型重答（2B → 9B → 百川）。
它不参与路由（路由在作答之前、只看问题）。

```
回答 ──► 规则检测器（工具调用泄漏、编造细节、漏答、剂量无依据、急症漏报……）
          │ 命中 block ──► 直接判失败（不问判别模型，确定性优先）
          ▼ 没命中
        Laya：一次请求回答三个问题
          │ P(答砸) ≥ block_at              ──► 判失败，升级重答；校验结论里带上失败类别
          │ review_at ≤ P(答砸) < block_at  ──► 放行，标记 needs_review 进人工复核
          ▼ P(答砸) < review_at             ──► 放行
        Laya 没部署 / 超时 / 报错 ──► 只用规则，记 disc_error，不影响主流程
```

调用出错的归因（超时、限流、鉴权、工具报错……）仍只用规则：受控测试 66/66 判对，不需要模型。

## 2. 为什么是 Laya，为什么要微调

Jev（TypeSafe AI）是闭源外部 API、早期访问，一直没有接入地址。Laya（Convai Innovations，Apache-2.0）是它的开源对应物：
同样是“类型化决策模型”——输入程序状态 + 预先定义的问题，一次前向给出每个问题的类型化答案和概率，不生成文字；
接口与 Jev 兼容（`POST /v1/systemone`）。用多语言版（mmBERT-base 编码器 + 决策头，3.2 亿参数），本机部署，不按次计费。

零样本选型（dev 80，2026-11-11，`evaluation/results/judge_bakeoff.json`；AUROC，0.5 = 随机）：

| 模型 | 类型 | 全部 80 条 | 规则没拦下的 67 条（其中 8 条失败） |
|---|---|---|---|
| Skywork-Reward-V2-Qwen3-0.6B | 奖励模型（只给一个质量分） | 0.85 | 0.91 |
| Skywork-Reward-V2-Qwen3-1.7B | 奖励模型 | 0.84 | 0.87 |
| Granite Guardian 3.2 3B | 风险检测（是 / 否 + 概率） | 0.77 | 0.81 |
| **Laya 多语言版** | 类型化决策模型 | 0.48 | 0.53 |
| LettuceDetect 中文版 | 幻觉检测（需要检索证据，dev 80 没有） | 0.51 | 0.42 |
| Qwen3Guard-Gen-0.6B | 安全审核 | 0.50 | 0.50 |

零样本的 Laya 接近随机，这与作者的说法一致（基础检查点零样本接近随机，是“用来微调的底座”）。
选 Laya 而不是零样本最好的 Skywork，是因为 Skywork 只能打一个分，做不了“失败属于哪一类”这样的多选判定；
Laya 能自定义任意是 / 否、多选题。微调后要在 dev 80 上与 Skywork 的 0.91 对比（见 §5），赢不了就该重新考虑。

## 3. 问 Laya 的三个问题

| 问题 | 类型 | 选项 | 用途 |
|---|---|---|---|
| `silent_failure` | 是 / 否 | 回答合格 / 答砸了 | **结果校验**，P(答砸) 和阈值比较 |
| `answer_quality` | 多选一 | good / acceptable / poor / unsafe | 监控与抽检 |
| `failure_type` | 多选一 | ok + dev 标注规范的 9 类：tool_leak、emergency_missing、incomplete、unsupported_detail、unsafe_recommendation、weak_refusal、poor_failure_handling、internal_name_leak、fabricated_history | 判为答砸时写进校验结论（`discriminator:silent_failure:<类别>`），便于排查 |

- 等级用多选而不是 `score`：作者报告 `score` 是最弱的题型，多语言版在 `score` 上有位置偏差；
- 是 / 否题自定义了选项标签（A / B）：默认的 true / false 字样会让模型跟着字样走（作者 issue #156）；
- 状态（`laya_state()`）：用户问题、助手回答（校验前的那版）、工具结果证据（每条截 240 字，最多 4 条）、规则检测结论、急症 / 高风险标记；
  最长 2,048 token。问题定义和状态格式训练与线上共用一份代码（有测试）。

## 4. 训练数据

| 来源 | 内容 | 标签 |
|---|---|---|
| 真实回答 | Qwen3.5-2B、Qwen3.5-9B 两个**未微调的基座**在路由数据集 train / val（1,737 题）上走完整编排，记下**校验前**的回答和程序状态 | 百川对照参考答案打的质量等级 + 失败类别（一次性约 3,500 次调用）；也可用本机部署的 Qwen3.5-27B 当裁判，不花 API 费 |
| 语义故障注入（只进 train，≤ 真实样本的 35%） | 从判为 good / acceptable 的回答出发：换成别题的回答、截掉后半、剂量放大 4 倍、追加“自行服用抗生素”、追加具体用药 +“不用去医院”、替医生下诊断、急症题删掉 120 / 急诊建议；改完被规则拦下的丢掉 | 按构造给等级和失败类别 |

- 不用等 9B 训完：基座模型的回答好坏都有，失败样本更多；
- 9B 训完后，第 8 步路由标注会让微调后的 9B 在 2,000 题上作答并由百川打标签，其中 train / val 部分可以直接再微调一次 Laya，不多花调用；
- 路由数据集的 test 部分和 dev 80 都不进训练：前者留给路由器报告，后者是判别模型的外部测试；
- val 只用真实样本：阈值要在真实分布上定。

## 5. 训练与部署（GPU 上，先于 9B 训练）

| 步骤 | 脚本 | 环境 | 做什么 |
|---|---|---|---|
| 1 | `training/verifier/01_answers.sh` | serve | 部署两个基座模型，采集回答 + 百川打等级和失败类别 |
| 2 | `02_data.sh` | serve | 生成 Laya 训练格式（`{state, questions, expected}`），跑一致性测试 |
| 3 | `03_train.sh` | laya | `laya.train.finetune`：RLCD 损失 + 留出数据温度校准；单张 24GB 卡足够，Mac 也能跑 |
| 4 | `04_serve.sh` | laya | `laya-serve`，:8002，模型名 `medical-verifier` |
| 5 | `05_eval.sh` | serve | 在验证集上定 block_at / review_at；dev 80 上对比规则、零样本选型结果，并看失败类别准确率 |

达标后在 `.env` 里加 `MEDAGENT_VERIFIER_URL=http://127.0.0.1:8002`，再训练 9B（`training/qwen35/01~07`）和路由器（`08`）。
2026-11-11 已在本机（Mac，MPS）用一小份假标签数据跑通 微调 → laya-serve 部署 → 客户端调用 → 评测脚本 全流程。

### 2026-11-14 真实标签训练结果（RTX PRO 6000）

- **数据**：2B / 9B 基座各作答 1,737 题，共 3,474 条；百川标注成功约 2,200 条，其余因百川额度用完（`insufficient_quota`）、
  备用 DeepSeek key 无效而记为 `judge_error`，构造数据时丢掉，没有改用别的裁判（避免两种标准混在一起）。
  加故障注入后 train 2,451 条（答砸占 47%，含注入 635 条）、val 378 条（只有真实样本）。
- **训练**：4 epoch，约 8 分钟。1,134 个留出判定上：准确率 0.282 → 0.831，损失 2.274 → 0.617，ECE 0.348 → 0.098。
- **定阈值**（判别器验证集 347 条，规则没拦下的部分）：AUROC 0.894，质量等级准确率 0.781；block_at 0.88（P 0.80 / R 0.74），review_at 0.50。
  结果写在 `medagent/verify/artifacts/discriminator.json`、`evaluation/results/discriminator_eval.json`。
- **dev 80 外部测试**（2026-09 旧模型的回答 + 人工标注）：规则放行部分 AUROC 0.59（零样本 Laya 0.48、零样本 Skywork 0.85）；
  规则→Laya 级联 F1 0.50（P 0.36 / R 0.80），低于纯规则 0.73，误报 28 条；失败类别在 20 条失败上全部判错。
- **端到端**（新 9B，296 条）：接入后触发修复 / 兜底 19 → 60 题，整体自动通过 0.909 → 0.892，高风险合规 0.972 → 0.861。
  这次强模型档（百川）没有额度，判别模型要求的“升级强模型重答”都落到了模板兜底，误报的代价被放大。
- **结论**：没达到 `05_eval.sh` 的上线标准，`.env` 默认不配 `MEDAGENT_VERIFIER_URL`，只用规则。
  同分布验证集上效果好、换到不同来源的回答就掉，说明训练数据的来源太单一（只有两个基座模型的回答）；
  下一步可以把微调后 9B 的回答、dev 扩充集的回答加进训练数据，并在强模型可用时重测端到端。

## 6. 已知限制

- 训练标签来自百川，判别模型的上限受百川判断质量限制；dev 80 的人工标注是唯一的人工对照，只有 20 条失败；
- Laya 多语言版的基准里没有明确列出中文，中文能力要看微调后的 dev 80 结果；
- 状态超过 2,048 token 时会被截断（Laya 返回的 usage.truncated 会标出来）。
