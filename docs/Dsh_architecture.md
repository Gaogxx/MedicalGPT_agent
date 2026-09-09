# DSH 医疗 Agent 架构说明

> 回答一个问题：桌面 `runtime/dsh-medical-plugins/` 只有 6 个文件，是怎么支撑起"模型变成 agent"的？

---

## 一、核心认知：桌面文件是「增量定制」，不是「框架本体」

`runtime/dsh-medical-plugins/` 里的 6 个文件只是「医疗业务的定制部分」。真正支撑 agent 的主体是 **DeepSeek Harness（DSH）框架本身**——一个 55 个子包的庞大代码库，位于服务器 `/root/autodl-tmp/deepseek-harness/`（`pnpm install` 后 node_modules 有几千个包）。

桌面那几份文件的价值是：**换一台机器，只要有 DSH 框架，把这几份文件放进去就能复现医疗 agent**——它们是"配方"，不是"食材"。

## 二、完整分层架构

```
┌──────────────────────────────────────────────────────┐
│  ① 模型权重  opd-merged（15GB，训练产物）               │
│     Qwen2.5-7B 经过 SFT→DPO→OPD 训练，学会了            │
│     医疗知识 + "该调工具时输出工具调用指令"               │
├──────────────────────────────────────────────────────┤
│  ② vLLM（模型推理服务，端口 8000）                      │
│     加载模型，提供 OpenAI 兼容 API                      │
├──────────────────────────────────────────────────────┤
│  ③ DeepSeek Harness 框架（55 子包，← 主体！）           │
│     agent-loop   ：agent 主循环                        │
│     tools        ：工具注册表 + defineTool + 执行管道    │
│     llm          ：LLM 抽象 + 适配器（连 vllm）          │
│     system-prompt：系统提示词装配                      │
│     session      ：会话管理 + 持久化                   │
│     compaction   ：上下文压缩                          │
│     cordis       ：插件框架（"一切皆插件"）             │
│     ... 其余 50 个包                                   │
├──────────────────────────────────────────────────────┤
│  ④ 我的医疗定制（桌面 6 个文件，← 增量）                 │
│     index.ts     ：3 个医疗工具（check_symptom 等）     │
│     cordis.yml   ：医疗 persona + 禁掉编码工具          │
│     settings.yaml：连 vllm 的 provider 配置            │
└──────────────────────────────────────────────────────┘
```

## 三、「模型 → agent」到底发生了什么

模型本身只会「输入文本 → 输出文本」，它**没有**工具执行、多轮记忆、上下文管理这些能力。是 DSH 框架的 `agent-loop` + `tools` 给它套上了这些：

```
用户问"我头痛3天"
        ↓
DSH 组装 system prompt（医疗 persona + 工具定义）→ 发给模型
        ↓
模型输出工具调用指令 <tool_call>{"name":"check_symptom",...}</tool_call>
        ↓
DSH 的 tools 服务解析这个指令 → 执行我写的 check_symptom 工具
        （index.ts 里的规则逻辑：症状→病因→建议→紧急程度）
        ↓
工具返回 {"possible_causes":["偏头痛",...], "urgency":"medium", ...}
        ↓
DSH 把结果作为 observation 回填 → 再次发给模型
        ↓
模型输出最终回复（基于工具结果 + 建议就医）
```

这个循环（**LLM → 工具调用 → 执行 → 观察回填 → 再生成**）就是 agent 的本质，由 DSH 的 `agent-loop` + `tools` 两个包完成。

- 我的 `index.ts` 只是往这个循环里塞了 3 个医疗工具（规则逻辑）
- 我的 `cordis.yml` 只是告诉它"用医疗 persona、别用编码工具"
- 我的 `settings.yaml` 只是告诉它"连 vllm 里的 medical-agent 模型"

## 四、一个类比

就像用 **Django** 写网站：

| Django 生态 | DSH 生态 |
|---|---|
| Django 框架（路由、ORM、模板） | DSH 框架（agent-loop、tools、session） |
| 你写的 models.py / views.py | 我写的 index.ts / cordis.yml |
| 数据库 | vLLM 里的模型权重 |

你只写了自己业务的几个文件，但支撑网站的是整个 Django。同样的，桌面那 6 个文件是"医疗业务代码"，支撑 agent 的是整个 DSH 框架。

## 五、三者合力

真正的 agent 能力来自三者合力，缺一不可：

1. **训练好的模型**（opd-merged，15GB）——医疗知识 + 工具调用的"内在能力"
2. **DSH 框架**（55 子包）——agent 循环、工具执行、会话管理、上下文压缩的"运行时"
3. **我的医疗定制**（6 个文件）——医疗工具 + 医疗 persona + 模型连接的"业务配置"

模型提供"智能"，框架提供"机制"，定制提供"领域知识"。
