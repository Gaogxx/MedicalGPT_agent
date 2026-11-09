# Jev 适配层：结果校验与出错归因

代码：`medagent/verify/jev.py`；测试：`tests/test_jev.py`（用按同一契约实现的假 Jev 服务跑通全部路径）。

## 1. Jev 是什么、公开信息到哪为止

Jev 是 TypeSafe AI 的“System-One”判别模型（早期访问）。公开资料能确认的只有：
- 输入：**程序状态 + 一个问题**；输出：**预先定义类型的值 + 校准置信度**，不输出推理文字；
- 三种原语：**Choice**（从给定选项里选一个）、**Score**（有序等级）、**Noul**（是/否 + 单一概率）；
- 定位是快速判断（资料称单次 70~500ms），用于路由、工具调用前的策略闸门、分类等。
没有公开 HTTP 接口文档，所以下面的**请求/响应格式是本项目自定的契约**。拿到官方文档后，只需替换
`JevClient.build_request` / `JevClient.parse_response` 两个函数，调用方不用改。

## 2. 契约

```
POST {JEV_BASE_URL}/v1/decide
Authorization: Bearer {JEV_API_KEY}
{
  "primitive": "noul" | "choice" | "score",
  "decision":  "silent_failure" | "failure_attribution" | "answer_quality",
  "question":  "…（固定问题文本，见 jev.py 的 DecisionSpec）",
  "state":     { …结构化 JSON，见 §3… },
  "options":   ["network", "timeout", …]        // choice 才有
  "levels":    ["unsafe", "poor", "acceptable", "good"]   // score 才有
}
→ { "value": true | "model" | "acceptable", "confidence": 0.0~1.0, "model": "jev-…" }
```
返回值类型不对（choice 返回了选项外的值）一律当调用失败处理，不猜。

## 3. 本项目问 Jev 的三个问题

| 判定 | 原语 | 输入状态（state） | 用途 |
|---|---|---|---|
| `silent_failure` | Noul | `query`、`answer`、`tool_results`（每个工具结果的 found / 字段名 / 剂量字段）、`rule_findings`（规则检测器的结论代码）、`safety`（是否急症/高风险） | **结果校验**：规则没拦住的回答，问“这是不是静默失败” |
| `failure_attribution` | Choice（network / timeout / rate_limit / auth / param / model / silent / external / policy） | 调用轨迹：每步的 stage、source、ok、error_kind、HTTP 码、错误信息前 120 字、重试次数、被拦截的规则代码；外加规则归因的结论 | **出错归因**：规则说不清是谁的错时，问“根因属于哪一类” |
| `answer_quality` | Score（unsafe / poor / acceptable / good） | 同 `silent_failure` | 监控与抽检，**不参与阻断** |

状态用结构化 JSON，不用自然语言段落：判别器读字段，字段名就是语义，避免“把上下文写成散文”带来的歧义。

## 4. 结果校验是怎么做的（`JevVerifier.verify`）

```
最终回答 ──► 规则检测器（来自真实失败样本：工具调用泄漏、编造细节、漏答、剂量无依据、急症漏报…）
               │ 命中 block ──► 直接判失败（不调 Jev，确定性优先，置信度 0.97）
               ▼ 没命中
             Jev silent_failure（Noul）→ 统一换算成 p = P(是静默失败)
               │ p ≥ 0.8        ──► 判失败：网关/编排器升级强模型重答（小模型档先升本地模型）
               │ 0.5 ≤ p < 0.8  ──► 放行，但标记 needs_review 进人工复核队列
               ▼ p < 0.5        ──► 放行
             Jev 不可用（未配置 / 超时 / 报错）──► 只用规则结论，记 jev_error，不影响主流程
```
设计取舍：
- **置信度不等于正确性**。能确定性验证的（剂量与工具结果比对、急救建议有无、泄漏格式）一律以规则为准，Jev 只补规则覆盖不到的语义类失败；
- **阈值分两档**：高置信自动处理，中间带交给人——Jev 资料本身也建议把置信度当“快路径 / 升级”的路由信号；
- 本轮离线评测里，用合成故障训练的本地轻量判别器在真实失败上 AUROC 只有 0.56，这正是需要 Jev（或用线上真实失败日志训练的判别器）来补的部分。

## 5. 出错归因是怎么做的（`JevVerifier.attribute`）

```
调用轨迹 events ──► 规则归因（verify/attribution.py）
                     · 本地异常类型：超时 / 连接错误
                     · HTTP 状态码：429 限流、401/403 鉴权、4xx 参数、5xx 外部
                     · MCP 工具错误前缀：PARAM / EXTERNAL / CONFLICT
                     · 取“第一个根因事件”，熔断打开这类派生错误不覆盖根因
                     · PARAM 再细分：参数是模型生成的 → 责任在模型；是代码拼的 → 系统 bug
                   │ 规则结论明确 ──► 用规则（source=rules）
                   ▼ 规则说不清（kind=unknown；或“工具调用成功、模型也返回了，但回答被判失败”——是模型没用好结果，还是工具结果本身不对？）
                  Jev failure_attribution（Choice）
                   │ confidence ≥ 0.7 ──► 采用 Jev 结论（source=jev，附 responsible：upstream / model / user）
                   ▼ 否则           ──► 保留规则结论，附 jev_low_confidence 供复核
```
受控测试（11 类故障 × 6 次）中规则归因 66/66 判对——明确的故障不需要 Jev；Jev 的价值在模糊的静默失败归因上。

## 6. 接入点与配置

- 网关（DSH 模式）：`RoutingGateway(verifier=JevVerifier(JevClient()))`，最终回答校验与 `meta.attribution`；
- 编排器（headless 模式）：`MedicalAgent._node_verify` 与 `_node_respond`；
- 环境变量：`JEV_BASE_URL`、`JEV_API_KEY`；不配置时自动只用规则；
- 阈值：`block_at=0.8`、`review_at=0.5`、`attribution_min=0.7`（拿到真实 Jev 后应在 dev 集上重新校准）。

## 7. 还需要你提供的

Jev 早期访问的接口地址、key 与官方原语文档。拿到后：改两个映射函数 → 在 80 条 dev + 线上抽样上校准阈值 → 与规则层对比误报/漏报。
