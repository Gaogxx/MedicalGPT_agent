# 医药 Wiki 约定（SCHEMA）

这份 Wiki 按 LLM Wiki 的方式维护：**原始资料只读 → 模型编译成带出处的页面 → 自动体检 → 线上缺口回流再编译**。
页面全部由程序生成，不要手改页面；要改内容，改原始资料或编译规则后重新编译。

## 目录结构

| 路径 | 内容 | 生成方式 |
|---|---|---|
| `drugs/<药名>.md` | 每个药一页 | `drug_label_kb.py`（模型抽取 / 核对 / 译写）→ `drug_kb.json` → `wiki.py build` 渲染 |
| `diseases/<病名>.md` | 与 Wiki 药品相关的疾病 | Wikidata 疾病—症状、疾病—治疗药物关系 |
| `index.md` | 目录：药品按 ATC 大类，疾病按 ICD-10 章 | `wiki.py build` |
| `log.md` | 每次编译与体检的记录（只追加） | `wiki.py build` |
| `gaps.md` | 缺口：缺字段的药、线上 `search_drug` 查不到 / 缺字段的药名 | `wiki.py build`（读 `.cache/medagent/wiki_gaps.jsonl`） |
| `_pages.json` | QID → 页面路径（运行时查找用） | `wiki.py build` |

## 药品页结构

顶部 YAML：`qid`、`name`、`generic_name`、`atc`、`aliases`（含编译时对齐出的新别名）、`kb_tier`（curated / label）、`reviewed: false`、`forms`（有哪些剂型）、`updated`。
正文按顺序：要点 → 适应症（链到疾病页）→ 禁忌 → 用法用量（多剂型时每个剂型一小节）→ 不良反应 → 相互作用（链到其他药品页）→
黑框警告 → 其他剂型的禁忌 / 不良反应 → 同类药物（ATC 前 5 位相同）→ 来源。每一段下面都有 `来源：`。

## 来源优先级与模型能做什么

1. **精选库**（kb.json，10 个药）：原字段一律保留，训练数据和评测按它。
2. **中文说明书原文**（Huatuo 百科问答 train 拆分）：只允许**原样摘录**或**挑句子编号**（抽取式），不改写。
   每段先过正则（缺数字 / 乱码），再由模型判断是否答非所问、有没有单位或数字损坏；不过就换同药的下一个商品，都不过就留空。
3. **美国 FDA 说明书**（openFDA，优先口服）：只补中文缺的字段。模型译写要点，译文里每个数字必须在英文原文出现，
   再核对有没有原文没有的内容；用法用量注明给药途径和“国内说明书可能不同”。
4. **Wikidata**：药名、别名、ATC、疾病—治疗药物关系。不提供剂量。
5. **要点**：模型只根据本页已有字段概括，不写剂量数字，数字与忠实度核对不过就不写。

模型**不能**：凭记忆补剂量 / 禁忌；在不同剂型之间换算剂量；把类别词（“疫苗”“雄激素”）当具体药。

## 剂型

同一个药的不同剂型（口服 / 注射 / 外用 / 栓剂 / 眼用 / 耳鼻 / 吸入 / 阴道用）分别存用法用量、禁忌、不良反应。
`search_drug` 认出问的剂型时返回该剂型；问到的剂型没有时明确说“知识库没有该剂型的用法用量”，**不给别的剂型的剂量**；
没看出剂型时附上各剂型的用法用量，并提示按所问剂型选用。

## 维护流程

```bash
python -m medagent.knowledge.wiki gaps --names-out gap_names.json          # 线上缺口
python -m medagent.knowledge.drug_label_kb --huatuo huatuo/train.jsonl --openfda-dir openfda \
    --llm http://127.0.0.1:8100/v1 --model compiler --pilot-names gap_names.json   # 重新编译（GPU）
python -m medagent.knowledge.drug_label_kb --summarize --llm http://127.0.0.1:8100/v1 --model compiler
python -m medagent.knowledge.wiki build                                    # 渲染 + 体检，结果追加到 log.md
```

体检项目：断链、孤立页、缺出处的段落、精选库与说明书的剂量数字矛盾、同一别名指向不同药。

## 运行时怎么用

- `search_drug`：读 `drug_kb.json`（与页面同源），按剂型返回，每个字段带来源；`MEDAGENT_DRUG_LABELS=0` 关闭。
- `search_knowledge`：问题里提到有页面的药 / 病时，把页面放在证据最前面，其余仍是全文检索；`MEDAGENT_WIKI=0` 关闭。
- MCP 资源：`wiki://index`、`wiki://page/{药名或病名}`。
- 缺口：`search_drug` 查不到或缺字段时记药名（不记用户原文）到 `.cache/medagent/wiki_gaps.jsonl`；`MEDAGENT_WIKI_GAPS=0` 关闭。

全部内容未经医生 / 药师审核。
