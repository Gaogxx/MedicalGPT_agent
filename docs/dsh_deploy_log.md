# DSH（DeepSeek Harness）运行时部署日志

> 记录 Medical-GPT × DeepSeek Harness 医疗 Agent 运行时搭建过程中遇到的问题与解决方法。
> 目标：把训练好的模型（opd-merged，Qwen2.5-7B）通过 DSH 作为运行时跑起来。

---

## 一、重大认知修正：DSH 是真实存在的

**结论**：DeepSeek Harness（`dsh`）是**真实存在**的开源项目，不是虚构。

验证结果（2026-09-09）：
- GitHub 仓库 `deepseek-ai/deepseek-harness` 存在，描述 "DeepSeek Harness: Everything is a Plugin."，TypeScript，master 分支
- npm 包 `@deepseek-ai/dsh` 存在（latest 0.1.2-rc.1），bin 是 `dsh` CLI
- 文档站：https://deepseek-harness.github.io/deepseek-harness/
- 基于 Cordis（DeepSeek 自 fork 为 `@deepseek-ai/cordis`）

⚠️ **之前的错误认知**：早期把 DSH 当成虚构项目，用 Python FastAPI 写了 `runtime/server/` 作为替代。这是错的，DSH 真实存在。`runtime/server/` 已删除（规则数据库已移植进 DSH 插件 index.ts）。

---

## 二、参考文档的 DSH 代码与真实 API 完全对不上

参考文档里给出的 DSH 代码是**虚构 API**，需要按真实 API 全部重写：

| 参考文档（虚构） | 真实 DSH |
|---|---|
| `dsh.config.yaml` | `cordis.yml`（patch overlay） |
| `ctx.tools.register('name', {description, parameters, handler})` | `ctx.tools.register(defineTool({name, description, parameters, output, execute}))` |
| `ctx.on('agent/before-tool-call', ...)` | 不存在；用 `tools/pre-execute` 扩展点 |
| `npm install cordis @deepseek-ai/dsh-core` | `@deepseek-ai/dsh-core` 不存在；用 `@deepseek-ai/cordis` + `@deepseek-ai/dsh-tools` |
| `llm: {provider: openai-compatible, base_url}` | `settings.yaml` 里 `llm-pi-ai.providers.<id>.{apiKeyEnv, api, baseURL, models}` |

### 真实工具注册 API（defineTool）

```ts
import type { Context } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'

export const name = 'dsh-medical-tools'
export const inject = ['tools']

export function apply(ctx: Context) {
  ctx.tools.register(defineTool({
    name: 'check_symptom',
    description: '...',               // 模型看到的内容
    parameters: {
      symptom: { type: 'string', required: true, description: '...' },
      duration: { type: 'string' },   // 可选默认
    },
    output: {
      schema: { type: 'json' },       // 注意：object 需 additionalProperties，返回任意 JSON 用 'json'
      render: (_args, value) => [{ type: 'text', text: JSON.stringify(value) }],
    },
    async execute(args) {             // args 已按 schema 校验
      return analyzeSymptom(args.symptom, args.duration)
    },
  }))
}
```

### 真实 LLM 配置（连 OpenAI 兼容 API / vllm）

`$DSH_HOME/settings.yaml`（默认 `~/.dsh/settings.yaml`）：

```yaml
llm-pi-ai:
  providers:
    vllm-medical:
      apiKeyEnv: VLLM_API_KEY        # 环境变量名，不是 apiKey
      api: openai-completions        # OpenAI Chat Completions 协议
      baseURL: http://127.0.0.1:8000/v1
      models:
        - id: medical-agent
          name: Medical Agent (Qwen2.5-7B OPD)
          contextWindow: 16384
          maxTokens: 2048
```

### 真实 cordis.yml（patch overlay）

```yaml
# 用 `pnpm dsh web --patch ./cordis.yml` 应用
- id: agent-default-model
  config: { provider: vllm-medical, model: medical-agent }
- insert:
    - id: dsh-medical-tools
      name: '/absolute/path/to/index.ts'   # 本地插件用绝对路径 + .ts 直接加载
```

---

## 三、环境坑与解决方法

| 症状 | 原因 | 解决 |
|---|---|---|
| `pnpm install` 报 `ERR_UNKNOWN_BUILTIN_MODULE` | DSH 要 Node 22.19+（不是 20），node 20 缺 node:sqlite | 下载 node 22.19.0 官方二进制 |
| vllm 新版下 torch 2.13 且给 CPU 版 | vllm 0.12+ 要 torch 2.13，清华源给 CPU 版 | **降级 vllm 0.8.x**（兼容 torch 2.6.0+cu124，复用 GPU torch） |
| `Qwen2Tokenizer has no attribute all_special_tokens_extended` | transformers 5.6（OPD 训练）与 vllm 不兼容 | 降级 `transformers==4.51.3` |
| `"auto" tool choice requires --enable-auto-tool-choice` | vllm 需要工具调用解析器 | 加 `--enable-auto-tool-choice --tool-call-parser hermes` |
| `CONTEXT_WINDOW_EXCEEDED 400` | system prompt + 工具描述超 8192 | vllm `--max-model-len 16384` |
| `Cannot find package '@deepseek-ai/dsh-tools'` | scratch-plugin 不在 pnpm workspace | 相对路径 import：`../../packages/core/tools/src/index.ts` + `../../vendor/cordis/src/index.ts` |
| `UNSUPPORTED_SCHEMA` | output.schema object 缺 additionalProperties | 改用 `{ type: 'json' }` |
| `PI_AI_ERROR: No API key` | settings 用错字段名 | 用 `apiKeyEnv: VLLM_API_KEY` + 启动 `export VLLM_API_KEY=dummy` |
| GitHub 直连慢 | 服务器直连 github 超时 | `source /etc/network_turbo` |

### tool_call 泄漏 bug（vllm hermes parser）

模型输出 `<tool_call>` 时 vllm 0.8.4 的 hermes parser 流式解析出错（`IndexError: pop from empty list`），导致 JSON + 垃圾字符泄漏到最终回复。**升级 vllm 0.8.5.post1 后大幅改善**（"恶心""心悸"两例已无泄漏）。注：0.8.5.post1 仍会报该错误（高频），但最终输出已基本不泄漏。

---

## 四、关键认知：DSH 默认是编码 agent

DSH headless/web 默认加载 **30+ 编码工具**（bash/fs/web/subagent/workflow/todo/goal/skill）+ 注入仓库 AGENTS.md。要变成纯医疗 agent，必须 cordis.yml patch：

```yaml
- id: tool-bash
  disabled: true
# ... tool-fs, tool-web, tool-subagent, tool-workflow, tool-todo, tool-goal, tool-skill 等
- id: agent-instructions   # 禁用 AGENTS.md 注入
  disabled: true
- id: system-prompt
  config:
    includeHarnessIdentity: false
    includeRuntimeContext: false
    personaPrefix: |
      你是一位专业的医疗助手...
```

完整架构说明见 `docs/Dsh_architecture.md`（分层：模型权重 → vLLM → DSH 框架 55 子包 → 医疗定制 6 文件）。

---

## 五、最终部署命令

```bash
export VLLM_API_KEY=dummy
export PATH=/root/autodl-tmp/node-v22.19.0-linux-x64/bin:/root/miniconda3/bin:$PATH

# 1. vllm 模型服务（端口 8000）
/root/miniconda3/bin/python -m vllm.entrypoints.openai.api_server \
  --model /root/autodl-tmp/MedicalGPT/models/opd-merged \
  --served-model-name medical-agent --port 8000 \
  --max-model-len 16384 --dtype bfloat16 \
  --enable-auto-tool-choice --tool-call-parser hermes

# 2. DSH Web UI（端口 3080）
cd /root/autodl-tmp/deepseek-harness
pnpm dsh web --patch ./scratch-plugin/cordis.yml

# 3. DSH headless（单任务测试）
node apps/cli/lib/bin.js --profile headless --patch ./scratch-plugin/cordis.yml "我头痛3天"
```

---

## 六、扩展测试（8 类 × 10 个用例）与改进

### 6.1 基线结果（80 用例）

| 类别 | 通过 | 问题 |
|---|---|---|
| 1-症状查询 | 10/10 | 1 处 tool_call 泄漏 |
| 2-药品查询 | 10/10 | 无 |
| 3-多轮工具调用 | 6/10 | 链式调用只做第一步；"咳嗽"误推抗生素 |
| 4-无需工具 | 9/10 | "睡眠不足"误调 check_symptom |
| 5-紧急症状 | 10/10 | "剧烈头痛+呕吐"未判紧急 |
| 6-高风险 | 10/10 | 无 |
| 7-工具失败 | 10/10 | 无 |
| 8-上下文 | 8/10 | headless 无跨会话记忆 |

### 6.2 三个真实问题与修复

1. **tool_call 泄漏**：vllm 0.8.4 hermes parser 流式 bug → 升级 0.8.5.post1 修复。
2. **多轮链式调用**：模型"先查症状再查药"只做第一步 → search_drug 增加**症状/适应症关键词反查**（三级匹配：精确药名 → 药名子串 → 症状反查）+ description 明确"不知道药名先 check_symptom"。验证：头痛→对乙酰氨基酚、关节痛→布洛芬、嗓子疼→对乙酰氨基酚。
3. **误调工具**："睡眠不足"被当症状查 → 属工具选择边界 case，需补训练数据（SFT 加"概念问答 ≠ 症状查询"样本），非运行时能改。

### 6.3 上下文溢出结论

headless 模式源码定位是 **"one-shot... creates one Agent... and exits"**——每次新会话是设计使然（用于 CI/单任务）。真正多轮对话应走 web 模式或 Python SDK（`--session-id`）。模型长上下文理解/总结能力已用「长历史注入」验证良好。

---

## 七、关键文件

本地 `runtime/dsh-medical-plugins/`：
- `src/index.ts` — 医疗工具插件（defineTool + 3 工具 + 规则库 + 症状反查）
- `cordis.yml` — 精简 patch overlay（禁编码工具 + 医疗 persona）
- `settings.yaml` — provider 配置（连 vllm）
- `test_medical.sh` / `test_medical_extended.sh` — 8 项 / 80 用例测试脚本
