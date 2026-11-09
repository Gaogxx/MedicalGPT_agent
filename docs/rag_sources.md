# 知识库与 RAG 知识源（三层）

## 总览

| 层 | 内容 | 规模 | 许可 | 用途 | 代码 |
|---|---|---|---|---|---|
| ① 精选 KB | 症状 / 药品 / 指南，含用法、禁忌、指南概括 | 21 / 10 / 9 条 | 原仓库手写，未审核 | 工具直接回答（症状→病因、药品说明、相互作用）；降级模式唯一来源 | `medagent/knowledge/kb.json`、`build_kb.py` |
| ② 实体层 | Wikidata 疾病（ICD-10）、药物（ATC）、中文别名、疾病—治疗药物关系 | 2,395 疾病 / 2,483 药物 / 4,817 别名 | CC0 | 认出几千个疾病和药名、给出编码与基础信息；**没有剂量，不做症状→病因** | `medagent/knowledge/wikidata_kb.py`、`ext.py` |
| ③ 全文检索 | 本地：Huatuo 百科问答 / 知识图谱问答 / Huatuo26M-Lite / shibing624/medical | **3,287,539 篇**（3.58GB，SQLite FTS5） | Apache-2.0 | `search_knowledge` 工具查资料、RAG 证据 | `medagent/rag/scalable.py` |
| ③ 在线文献 | PubMed E-utilities / Europe PMC / openFDA 药品说明书 / MedlinePlus | **41,410,190 / 49,193,995 条**（2026-11-08 实时查询） | 免费公开接口 | 最新文献、英文说明书；各自熔断 + 超时 | `medagent/rag/sources.py` |

“千万级”指的是在线文献层：PubMed 与 Europe PMC 都是实时检索，不需要下载。本地层目前是三百万级，更大规模见文末。

## 为什么要分层、为什么不能把所有数据都当“知识”

- 越往下规模越大、权威性越低、越不能直接当结论：精选 KB 用来答，实体层用来认，全文检索用来查资料，最后由模型带出处综合；
- 实测踩到的坑：Wikidata 的“疾病—症状”关系没有发病率排序，“过度充血”会给出“鼠疫”、“焦虑”会给出“番木鳖碱中毒”，
  所以实体层不接入症状分析工具，也不用来造训练数据（问题 N52）；
- 开源问答语料来自网络问诊和百科，质量参差，进索引可以，进训练要过滤（`supplement.py` 的长度 / 广告 / 去重 / 质量分过滤）。

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
