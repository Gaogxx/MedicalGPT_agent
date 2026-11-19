# 知识库与 RAG 知识源（三层）

## 总览

| 层 | 内容 | 规模 | 许可 | 用途 | 代码 |
|---|---|---|---|---|---|
| ① 精选 KB | 症状 / 药品 / 指南，含用法、禁忌、指南概括 | 21 / 10 / 9 条 | 原仓库手写，未审核 | 工具直接回答（症状→病因、药品说明、相互作用）；降级模式唯一来源 | `medagent/knowledge/kb.json`、`build_kb.py` |
| ② 实体层 | Wikidata 疾病（ICD-10）、药物（ATC）、中文别名、疾病—治疗药物关系 | 2,395 疾病 / 2,483 药物 / 4,817 别名 | CC0 | 认出几千个疾病和药名、给出编码与基础信息；**不做症状→病因** | `medagent/knowledge/wikidata_kb.py`、`ext.py` |
| ②′ 药品说明书层 + 医药 Wiki | 给实体层药品补适应症 / 禁忌 / 用法用量（按剂型）/ 不良反应 / 相互作用，并入精选库 10 个药；渲染成互相链接、带出处的 Markdown 页面 | 2,483 个药中 650 个三项齐全；Wiki 药品 1,323 页、疾病 344 页 | Huatuo（Apache-2.0）+ openFDA（公有领域）+ Wikidata（CC0） | `search_drug` 按剂型返回；`search_knowledge` 证据优先放 Wiki 页；`MEDAGENT_DRUG_LABELS=0` / `MEDAGENT_WIKI=0` 关掉 | `drug_label_kb.py` → `drug_kb.json` → `wiki.py` → `wiki/` |
| ③ 全文检索 | 本地：Huatuo 百科问答 / 知识图谱问答 / Huatuo26M-Lite / shibing624/medical | **3,287,539 篇**（3.58GB，SQLite FTS5） | Apache-2.0 | `search_knowledge` 工具查资料、RAG 证据 | `medagent/rag/scalable.py` |
| ③ 在线文献 | PubMed E-utilities / Europe PMC / openFDA 药品说明书 / MedlinePlus | **41,410,190 / 49,193,995 条**（2026-11-08 实时查询） | 免费公开接口 | 最新文献、英文说明书；各自熔断 + 超时 | `medagent/rag/sources.py` |

“千万级”指的是在线文献层：PubMed 与 Europe PMC 都是实时检索，不需要下载。本地层目前是三百万级，更大规模见文末。

## 为什么要分层、为什么不能把所有数据都当“知识”

- 越往下规模越大、权威性越低、越不能直接当结论：精选 KB 用来答，实体层用来认，全文检索用来查资料，最后由模型带出处综合；
- 实测踩到的坑：Wikidata 的“疾病—症状”关系没有发病率排序，“过度充血”会给出“鼠疫”、“焦虑”会给出“番木鳖碱中毒”，
  所以实体层不接入症状分析工具，也不用来造训练数据（问题 N52）；
- 开源问答语料来自网络问诊和百科，质量参差，进索引可以，进训练要过滤（`supplement.py` 的长度 / 广告 / 去重 / 质量分过滤）。

## 药品说明书层与医药 Wiki（2026-11-19）

Wikidata 实体层的 2,483 个药原来没有剂量、禁忌、不良反应，Medical-GPT 查到这些药时只能答“知识库未包含具体剂量”。
这一层按 **LLM Wiki** 的方式建：原始资料只读 → 模型“编译”成带出处的条目和 Markdown 页面 → 自动体检 → 线上缺口回流再编译；
医疗场景下把模型能自由发挥的地方压到最少。页面约定见 [medagent/knowledge/wiki/SCHEMA.md](../medagent/knowledge/wiki/SCHEMA.md)。

```
原始资料（只读）                         编译（GPU，Qwen3.5-27B-FP8）            Wiki（仓库里）                    运行时
Wikidata 实体层 ─┐                       drug_label_kb.py                         wiki/drugs/*.md   1,323 页     search_drug：按剂型返回、字段带来源
Huatuo 百科问答 ─┼─ 名称对齐 → 抽取/体检 → drug_kb.json ── wiki.py build ──→   wiki/diseases/*.md  344 页     search_knowledge：页面排在证据最前
openFDA 说明书 ──┤   （按剂型）  → 译写/体检   + 要点（模型概括，已核对）     index / log / gaps / SCHEMA     MCP 资源 wiki://index、wiki://page/{名}
精选库 kb.json ──┘                                                                                                查不到 → .cache/medagent/wiki_gaps.jsonl → gaps.md → 下次编译
```

**编译（全部 2,483 个药，模型调用 11,910 次）**

| 步骤 | 做法 | 结果 |
|---|---|---|
| 名称对齐 | 商品名去括号 / 剂型 / 盐 → 等于 Wikidata 中文名或别名才算；对不上的请模型给英文 INN，反译回中文一致才收；剥成化学碎片的（“硝酸甘油”→“甘油”、“葡萄糖酸钙”→“葡萄糖酸”）不收，指向两个药的别名删掉 | 7,855 个 Huatuo 药品对上 1,004 个 Wikidata 药；新增别名 49 个 |
| 中文字段（按剂型） | 口服 / 注射 / 外用 / 栓剂 / 眼用 / 耳鼻 / 吸入 / 阴道用分别取商品，同一剂型的用法、禁忌、不良反应尽量出自同一商品；只摘录原文，过长时模型只挑句子编号 | 279 个药有多个剂型 |
| 体检 | 正则筛缺数字 / 乱码；模型判断是否答非所问、有没有单位或数字损坏（如注射剂量“100～200μg”被抓成“100～200g”），不过就换下一个商品，都不过就留空 | 拒掉 747 段（正则）+ 688 段（损坏）+ 98 段（答非所问） |
| openFDA 补缺 | 中文缺的字段才译写，优先口服说明书；译文每个数字都必须在英文原文出现，再核对有没有原文没有的内容；用法用量注明给药途径 | 通过 2,163 段，拒掉 297 段（数字）+ 237 段（不忠实） |
| 要点 | 模型只根据本条目已有字段写 ≤100 字概括，不写剂量数字，数字与忠实度核对不过就不写 | 写出 1,191 段，拒掉 90 段 |
| 精选库并入 | 10 个药原字段不动（训练数据和评测都按它），补相互作用说明 / 黑框警告；问到具体剂型时附该剂型说明书；“头孢”是类别，单独成条 | — |
| 显示名 | Wikidata 中文标签偶有错误（格鲁米特被标成“Vologases六世”）或是化学名，换成中文别名 | 6 个 |

结果：2,483 个药里 **650 个三项（用法、禁忌、不良反应）齐全**、416 个两项、256 个一项、1,161 个一项都没有
（Huatuo 与 openFDA 都没有，或是“疫苗”“雄激素”这类类别词 / 非药实体）。
Wiki 体检：断链 0、孤立页 0、缺出处段落 0、精选库与说明书剂量矛盾 0、别名冲突 0（每次编译追加到 `wiki/log.md`）。

**端到端评测**（最终 9B，GPU1 单卡 L20，温度 0，只用规则校验，不经强模型；补全前 = `MEDAGENT_DRUG_LABELS=0 MEDAGENT_WIKI=0`）

药品留出题 400 道（`evaluation/build_drug_label_eval.py --max 400`，470 个药）：同一个药、但出自药品库没用到的另一份说明书的问答，
参考答案与药品库采用的文字不同；两组都能检索本地全文索引（参考文本也在索引里），测的是“有了全文检索之后，药品库和 Wiki 还能多带来多少”。

| 指标（400 道药品留出题） | 补全前 | 补全后（最终版） |
|---|---|---|
| 参考答案要点覆盖率（自动） | 0.053 | **0.200** |
| 　其中 用法用量 / 不良反应 / 禁忌 | 0.006 / 0.088 / 0.099 | 0.121 / 0.259 / 0.276 |
| 回答“查不到 / 知识库没有”而无实质内容 | 22.5% | **5.3%** |
| 　其中 用法用量 / 不良反应 / 禁忌 | 28.2% / 23.6% / 6.7% | 11.1% / 0.7% / 0.0% |
| 给出的剂量数字不在药品库或参考答案中（编造剂量代理） | 0 | 0 |
| 跨剂型剂量（明确问了剂型的 160 道用法题） | 0 | 0 |
| 逐题更好 / 更差（覆盖率差 > 0.1） | — | 164 / 8 |

| 指标（原 296 题回归） | 补全前 | 补全后（最终版） |
|---|---|---|
| 自动通过率 | 0.919 | 0.922 |
| 急症建议 / 格式 / 查不到如实说明 / 不编造历史 | 0.974 / 1.0 / 0.941 / 1.0 | 0.974 / 1.0 / 0.941 / 1.0 |
| 高风险合规 | 0.917 | 0.944 |
| 多意图覆盖 | 0.865 | 0.904 |
| 药品事实覆盖率（精选库 10 个药，18 题） | 0.667 | 0.667 |

覆盖率是按字面比对的（“150mg”和“0.15g”算不上），绝对值偏低，主要看前后差。“跨剂型剂量”只统计明确问了剂型的 160 道用法题：
回答给出的剂量只出现在药品库另一个已知剂型的说明书里、且没点明那个剂型。

消融（同 400 题，和“全开”比）：关掉按剂型返回，用法题“查不到”11.1% → 15.5%，覆盖率 0.122 → 0.119；关掉 Wiki 进检索证据，覆盖率 0.200 → 0.197、逐题 5 好 2 差。
药品题主要靠 `search_drug`，Wiki 的贡献在这类题上很小，主要用在 `search_knowledge`（疾病用药、相互作用）和人工查阅、维护上。

评测过程中发现并修掉的问题：
- 用户问“尼古丁贴剂”、模型只传“尼古丁”，药品库只有 openFDA 的口服含片说明书，回答把含片用法当贴剂答了——
  现在 agent 把原话里紧跟药名的剂型补进 `search_drug` 参数，openFDA 条目按给药途径定剂型，问到没有的剂型就明确说没有；
- 精选库的药附上各剂型用法后，模型不敢给剂量（“二甲双胍怎么吃”），没问剂型时不再附。
四组评测并发跑，温度 0 下 vLLM 批处理仍有不确定性（不涉及药品的题也有约一半回答不同），296 题只看总体指标。

**已知问题**：全部未经医生 / 药师审核（`reviewed=false`，字段带来源）；1,161 个药仍无说明书字段；剂型只分大类（“缓释片”和“片”同属口服）；
openFDA 是美国说明书，与国内可能不同（已注明）。

```bash
# 编译（GPU 机，需要 OpenAI 兼容的模型服务）
python -m medagent.knowledge.drug_label_kb --huatuo huatuo/train.jsonl --openfda-dir openfda \
    --llm http://127.0.0.1:8100/v1 --model compiler --out medagent/knowledge/drug_kb.json
python -m medagent.knowledge.drug_label_kb --summarize --llm http://127.0.0.1:8100/v1 --model compiler
python -m medagent.knowledge.wiki build          # 渲染 + 体检
python -m medagent.knowledge.wiki gaps --names-out gap_names.json   # 线上缺口 → 下次编译 --pilot-names
# 评测
python evaluation/build_drug_label_eval.py --huatuo-dir huatuo --max 400
MEDAGENT_DRUG_LABELS=0 MEDAGENT_WIKI=0 python evaluation/run_live_eval.py --set druglabel --temperature 0 --out evaluation/results/live_druglabel_full_off.json
python evaluation/run_live_eval.py --set druglabel --temperature 0 --out evaluation/results/live_druglabel_final2.json
python evaluation/score_drug_label_eval.py off=evaluation/results/live_druglabel_full_off.json final=evaluation/results/live_druglabel_final2.json
```

## 本地索引怎么建

```bash
# 一步完成：下载 5 个语料文件（约 2.2GB，已下载的跳过）+ 建索引（8 进程 jieba 切词，329 万篇约 2.5 分钟，约 3.3GB）
python -m medagent.rag.scalable setup            # 国内网络：HF_ENDPOINT=https://hf-mirror.com python -m ...
# 也可以分开做
python -m medagent.rag.scalable download         # --with-training 额外下载训练数据补充用的语料
python -m medagent.rag.scalable build            # --only huatuo_enc,huatuo_kg 只建部分来源；--limit-per-source 1000 先试小规模
# 检查
python -m medagent.rag.scalable search "甲状腺结节需要手术吗"
```

运行时自动发现 `.cache/index/medical.fts5`（或用 `MEDAGENT_FTS5_INDEX` 指定路径），`search_knowledge` 多出 fts5 一路召回；
MCP 资源 `kb://sources` 会显示 `local_fulltext.available: true`。

索引和语料不放进 GitHub：单文件超过 GitHub 100MB 上限，而且能用上面的命令在几分钟内重建。语料许可均为 Apache-2.0，
如需分发现成索引，建议放 Hugging Face 数据集仓库并附上许可与出处。

## 在线文献怎么用

不需要下载或建索引，PubMed / Europe PMC / openFDA / MedlinePlus 都是检索时实时调用对方公开接口，不需要 API key：

```bash
export MEDAGENT_ONLINE_KB=1      # search_knowledge 增加 online 一路召回
```

需要能访问 eutils.ncbi.nlm.nih.gov、www.ebi.ac.uk、api.fda.gov、wsearch.nlm.nih.gov。每个源各自熔断、超时（Europe PMC 10 秒，其余 4 秒），
任一源不可用只少一路召回，不影响回答。PubMed 未带 API key 时官方限速约 3 次/秒，高并发上线需申请 NCBI API key。
在线文献多为英文摘要，中文问题主要靠本地索引。

## 规模评测（`evaluation/run_rag_scale_eval.py`）

Huatuo 百科问答 1000 道测试题作查询，标准答案混在全量文档里：

| 规模 | BM25 Recall@5 | 去高频词后 p50 延迟 | + 交叉编码器重排（抽样 300 题）Recall@5 |
|---|---|---|---|
| 1,337,567 篇 | 0.153 | 12ms（原 45ms） | 0.167 → 0.293 |
| 3,287,539 篇 | 0.120 | 32ms（原 127ms） | 0.123 → 0.223 |

结论：规模扩大后纯词法检索的区分度下降，必须靠重排与向量召回补；交叉编码器在 CPU/MPS 上约 1 秒 / 50 篇，生产需 GPU 或换更小的重排模型。

## 再往上扩到千万级本地索引

同一接口换后端即可：
- 稀疏：Elasticsearch / OpenSearch（IK 中文分词 + 医学词典），分片部署；
- 稠密：Milvus / Qdrant，或 FAISS IVF-PQ（`FaissDense` 已实现，约 64B / 向量），向量在 GPU 上批量计算；
- 数据：Huatuo-26M 的问诊部分只公开了 URL（版权原因），需自行合规获取；PubMed 全量摘要可用 NCBI 年度基线包（baseline）离线建索引。

## 中文权威来源的现状

| 来源 | 现状 |
|---|---|
| 国家药监局药品说明书 | 无公开 API，需合规渠道获取 |
| CMeKG（北大） | 不提供批量开放下载 |
| 中文临床指南原文 | 多为期刊出版物，需版权授权 |
| UMLS / DrugBank | 需单独申请许可 |
