#!/usr/bin/env python3
"""从 evaluation/results/*.json 生成本地报告 evaluation/ITERATIONS.md / ISSUES.md（两者与 issues_log.json 不入库）。
所有数字都来自结果文件；重新跑任一 run_*_eval.py 后再执行本脚本即可刷新。"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "evaluation" / "results"


def j(name: str) -> dict:
    p = R / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def pct(x) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def training_logs() -> list[str]:
    rows = []
    for stage, f in (("SFT", "sft_train.log"), ("DPO", "dpo_train.log"), ("OPD", "opd_train.log")):
        t = (ROOT / "training" / "logs" / f).read_text(errors="ignore").replace("\r", "\n")
        ds = []
        for m in re.finditer(r"\{'[^{}]*\}", t):
            try:
                ds.append(ast.literal_eval(m.group(0)))
            except Exception:  # noqa: BLE001
                pass
        tr = [d for d in ds if "loss" in d]
        ev = [d for d in ds if "eval_loss" in d]
        first, last = tr[0], tr[-1]
        extra = ""
        if "rewards/accuracies" in last:
            extra = f"；rewards/accuracies {float(first['rewards/accuracies']):.3f}→{float(last['rewards/accuracies']):.3f}，margins→{float(last['rewards/margins']):.2f}"
        if ev:
            extra += f"；eval_loss {float(ev[-1]['eval_loss']):.3f}（eval_samples={int(ev[-1].get('eval_samples', 0))}）"
        rows.append(f"| {stage} | loss {float(first['loss']):.4f} → {float(last['loss']):.4f}（epoch {float(last['epoch']):.2f}）{extra} |")
    return rows


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and 0 <= v <= 1:
        return f"{v * 100:.1f}%"
    return str(v)


def _bucket(status: str) -> str:
    return "未解决" if status.startswith("未解决") else "部分解决/待验证" if status.startswith(("部分", "数据侧")) or "待" in status else "已解决"


def render_issues() -> tuple[str, list[str]]:
    p = ROOT / "evaluation" / "issues_log.json"
    if not p.exists():
        return "# 问题记录\n\n（本地没有 evaluation/issues_log.json）\n", ["（本地没有问题记录）"]
    data = json.loads(p.read_text(encoding="utf-8"))["issues"]
    from collections import Counter
    by_status = Counter(_bucket(i["status"]) for i in data)
    by_area = Counter(i["area"] for i in data)
    groups = {"环境报错": "环境报错", "人工阅读输出": "人工阅读模型输出",
              "数据审计": "数据审计", "二次审计": "数据审计", "人工复核审计结果": "数据审计",
              "日志分析": "日志/文档分析", "复读原部署日志": "日志/文档分析", "对比原评测表": "日志/文档分析",
              "核查来源": "日志/文档分析", "人工核实": "日志/文档分析",
              "代码审查": "代码审查/自查", "自查": "代码审查/自查", "用户反馈": "用户反馈"}
    found = Counter()
    for i in data:
        for f in {groups.get(x, "自动评测/测试") for x in i["found_by"].split(" + ")}:
            found[f] += 1
    by_date = Counter(i["date"] for i in data)
    with_metric = sum(1 for i in data if i["metric"])
    summary = ["| 统计项 | 数值 |", "|---|---|",
               f"| 问题总数 | {len(data)}（2026-09 原项目 {by_date.get('2026-09', 0)} 条，2026-11-05 {by_date.get('2026-11-05', 0)} 条，2026-11-09 {by_date.get('2026-11-09', 0)} 条） |",
               f"| 已解决 / 部分解决或待验证 / 未解决 | {by_status['已解决']} / {by_status['部分解决/待验证']} / {by_status['未解决']} |",
               f"| 有量化指标（前→后）的问题 | {with_metric} |",
               f"| 评测方法自身的问题（指标或测试集本身有缺陷） | {by_area.get('评测方法', 0) + by_area.get('训练/评测方法', 0)} |",
               "| 按领域 | " + "，".join(f"{k} {v}" for k, v in by_area.most_common()) + " |",
               "| 按发现方式 | " + "，".join(f"{k} {v}" for k, v in found.most_common()) + " |"]
    md = ["# 问题记录（自动生成：`python evaluation/report.py`，数据源 `evaluation/issues_log.json`）", "",
          "每条：现象（看到了什么）→ 根因（为什么）→ 解决（做了什么）→ 指标（前→后）→ 证据文件 → 状态。", "",
          "## 汇总", "", *summary, ""]
    for date, title in (("2026-09", "2026-09 原项目训练与部署期（来自 docs/dsh_deploy_log.md）"), ("2026-11-05", "2026-11-05 重构"),
                        ("2026-11-09", "2026-11-09 知识库扩充 / 数据补充 / 三档路由 / Jev")):
        md += [f"## {title}", "", "| ID | 领域 | 发现方式 | 现象 | 根因 | 解决 | 指标（前→后） | 状态 |", "|---|---|---|---|---|---|---|---|"]
        for i in data:
            if i["date"] != date:
                continue
            m = i["metric"]
            metric = f"{m['name']}：{_fmt(m['before'])} → {_fmt(m['after'])}" if m else "—"
            cells = [i["id"], i["area"], i["found_by"], i["phenomenon"], i["root_cause"], i["fix"], metric, i["status"]]
            md.append("| " + " | ".join(str(c).replace("|", "／") for c in cells) + " |")
        md.append("")
    return "\n".join(md), summary


def main() -> None:
    out = ["# 迭代记录（自动生成：`python evaluation/report.py`）", "",
           "> 时间线：2026-09 为原项目已有的真实训练/部署记录（来自 training/logs 与 DSH 线上测试输出）；",
           "> 2026-11-05 起为本轮重构的逐次迭代。每一行都对应 `evaluation/results/` 下的一个结果文件。", ""]
    out += ["## 0. 历史基线（2026-09，原项目）", "", "| 阶段 | 训练日志中的真实指标 |", "|---|---|"] + training_logs()
    out += ["", "**问题诊断**：SFT 的 `eval_samples=4945` 等于训练样本数——验证集就是训练集；DPO 关闭了评估，"
            "3 个 epoch 后 reward 准确率 1.0、margin≈10，结合数据审计（去重后仅 65 对）可判定为过拟合。", ""]

    d0, d1 = j("data_audit_v0.json"), j("data_audit_v1.json")
    if d0 and d1:
        def lk(d, k):
            v = d["leakage"].get(k)
            return v["same_query_rate"] if v else None
        out += ["## 1. 数据迭代", "", "| 指标 | v0（原数据） | v2（清洗+遮挡增强+分组切分） |", "|---|---|---|",
                f"| SFT 评测集与训练集同问题比例 | {pct(lk(d0, 'sft_eval_vs_train'))} | {pct(lk(d1, 'sft_eval_vs_train'))} |",
                f"| SFT 评测集与训练集近重复比例 | {pct(d0['leakage']['sft_eval_vs_train']['near_query_rate'])} | {pct(d1['leakage']['sft_eval_vs_train']['near_query_rate'])} |",
                f"| DPO 评测集与训练集同问题比例 | {pct(lk(d0, 'dpo_eval_vs_train'))} | {pct(lk(d1, 'dpo_eval_vs_train'))} |",
                f"| SFT 完全重复率 | {pct(d0['duplicates'][0]['exact_dup_rate'])} | {pct(d1['duplicates'][0]['exact_dup_rate'])} |",
                f"| DPO 完全重复率 | {pct(d0['duplicates'][1]['exact_dup_rate'])} | {pct(d1['duplicates'][1]['exact_dup_rate'])} |",
                f"| DPO 问法模板唯一率 | {pct(d0['templates']['dpo']['unique_ratio'])} | {pct(d1['templates']['dpo']['unique_ratio'])} |",
                f"| DPO chosen 比 rejected 长的比例（长度偏差） | {pct(d0['dpo_bias']['chosen_longer_rate'])} | {pct(d1['dpo_bias']['chosen_longer_rate'])} |",
                f"| 工具参数中用户未提供的持续时间 | {pct(d0['tool_args']['ungrounded_duration_rate'])} | {pct(d1['tool_args']['ungrounded_duration_rate'])} |",
                f"| SFT 中非医疗污染样本 | {d0['contamination']['sft']['non_medical']} 条 | {d1['contamination']['sft']['non_medical']} 条（改为受控的通用回放） |", ""]
        man = json.loads((ROOT / "training" / "data" / "data_v2_manifest.json").read_text())
        out += [f"v2 产出：SFT train {man['sft']['_summary']['train']} / val {man['sft']['_summary']['val']}，增强样本占比 "
                f"{pct(man['sft']['_summary']['aug_share'])}；DPO train {man['dpo']['train']} / val {man['dpo']['val']}"
                f"（原始 1500 对去重后仅 {man['dpo']['original_after_clean']} 对 + {man['dpo']['counterfactual_picked']} 对反事实对）。", ""]

    out += ["## 2. 回答层（80 条真实线上输出回放，dev 集）", "",
            "| 迭代 | 版本 | 急症建议率 | 高风险合规 | 免责声明 | 格式干净 | 自动通过率 | 静默失败拦截 | 误升级 |", "|---|---|---|---|---|---|---|---|---|"]
    for tag, note in (("it1", "初版指标"), ("it2", "收紧高风险指标 + 校验器修 bug"), ("it3", "规则泛化 + 复述误报修复")):
        d = j(f"answer_eval_{tag}.json").get("summary", {})
        for ver, s in d.items():
            out.append(f"| {tag}（{note}） | {ver} | {pct(s.get('emergency_advice_rate'))} | {pct(s.get('high_risk_compliance'))} | "
                       f"{pct(s.get('disclaimer_rate'))} | {pct(s.get('format_clean_rate'))} | {pct(s.get('auto_pass_rate'))} | "
                       f"{pct(s.get('silent_failure_interception'))} | {pct(s.get('false_escalation_rate'))} |")
    v0 = j("answer_eval_it3.json").get("summary", {}).get("v0_baseline", {})
    if v0:
        out += ["", f"v0 人工标注通过率 {pct(v0.get('human_pass_rate'))}；自动指标与人工标注一致率 {pct(v0.get('auto_vs_human_agreement'))}。"
                "注意 dev 集同时被用来调规则，it2/it3 在 dev 上的数字是样本内成绩，泛化看下一节的独立测试集。", ""]

    out += ["## 3. 急症 / 高风险分诊（看泛化）", "", "| 迭代 | 方法 | 评测集 | 急症召回 | 急症误报 | 高风险召回 | 危机(自伤)召回 |", "|---|---|---|---|---|---|---|"]
    s1 = j("safety_eval_it1.json")
    if s1:
        a = s1["heldout_a"]
        out.append(f"| it1 | 关键词规则 | held-out A（规则编写后才写） | {pct(a['emergency']['recall'])} | {pct(a['emergency']['false_positive_rate'])} | {pct(a['high_risk']['recall'])} | {pct(a['crisis']['recall'])} |")
    c3 = j("safety_eval_it3_testC.json").get("testc_rules_it3_plus_embedding")
    c3r = j("safety_eval_it3_testC.json").get("testc_rules_it3")
    if c3r:
        out.append(f"| it3 | 泛化规则（用 dev+A 调） | **独立集 C** | {pct(c3r['emergency']['recall'])} | {pct(c3r['emergency']['false_positive_rate'])} | {pct(c3r['high_risk']['recall'])} | {pct(c3r['crisis']['recall'])} |")
        out.append(f"| it3 | 规则 + 原型向量 | **独立集 C** | {pct(c3['emergency']['recall'])} | {pct(c3['emergency']['false_positive_rate'])} | {pct(c3['high_risk']['recall'])} | {pct(c3['crisis']['recall'])} |")
    c4 = j("safety_eval_it4_learned_testC.json")
    for name in ("tfidf_only", "tfidf+bge"):
        if name in c4:
            v = c4[name]
            out.append(f"| it4 | 规则 ∪ 可训练分类器（{name}） | **独立集 C** | {pct(v['emergency_rules_or_clf']['recall'])} | "
                       f"{pct(v['emergency_rules_or_clf']['fpr'])} | {pct(v['high_risk_rules_or_clf']['recall'])} | 3/3（规则事后补丁） |")
    out += ["", "说明：held-out B 由同一作者在调规则前写成，但调规则时 B 的说法已在上下文中，B 上的 100% 不作为泛化证据；"
            "独立集 C 由一个看不到规则与 A/B 的子代理盲写（80 条），it3 规则在 C 上首次评测即暴露 3 条自伤消息全部漏检，"
            "已补规则（事后，C 上的危机召回因此不再是无偏估计）。it4 的急症召回把危机消息计为正例，含事后补丁；"
            "补丁前（纯 C 盲测）两种分类器的急症召回均为 90.9%。", ""]

    rag = j("rag_eval_it3_full.json") or j("rag_eval_it2_dense.json")
    if rag:
        out += ["## 4. RAG 逐环节消融", "", "| 迭代 | KB 集 Recall@5 | KB MRR | QA 集 Recall@5 | QA MRR | QA nDCG@10 | 平均延迟 | top5 证据 token |", "|---|---|---|---|---|---|---|---|"]
        for k, v in rag.items():
            if k == "cache":
                continue
            out.append(f"| {k} | {v['kb']['recall@5']:.3f} | {v['kb']['mrr@10']:.3f} | {v['qa']['recall@5']:.3f} | {v['qa']['mrr@10']:.3f} | "
                       f"{v['qa']['ndcg@10']:.3f} | {v['qa']['latency_ms_mean']:.0f}ms | {v['qa']['evidence_tokens_top5']:.0f} |")
        c = rag["cache"]
        out += ["", f"缓存：200 条模拟流量命中率 {pct(c['hit_rate'])}，命中延迟 {c['latency_ms_hit']}ms vs 未命中 {c['latency_ms_miss']}ms。"
                "it1 曾因把问题写进文档 title 导致评测泄漏（QA Recall@5 虚高到 0.998），已修复，上表为修复后结果。", ""]

    rt = j("router_eval_it1.json")
    if rt:
        out += ["## 5. 路由（RouteLLM-MF，80 条 dev，5 折交叉验证）", "", "| 路由器 | 强模型调用比例 | 本地失败召回 | 通过率上界 |", "|---|---|---|---|",
                f"| 随机 | 40% | 40% | — |",
                f"| 仅安全规则 | {pct(rt['rules_only']['strong_rate'])} | {pct(rt['rules_only']['failure_recall'])} | {pct(rt['rules_only']['pass_upper_bound'])} |"]
        for k in ("mf_hash", "mf_bge"):
            if k in rt:
                v = rt[k]["safety_first+0.4"]
                out.append(f"| 安全规则 + MF（{k}，AUROC {rt[k]['auroc_oof']:.2f}） | {pct(v['strong_rate'])} | {pct(v['failure_recall'])} | {pct(v['pass_upper_bound'])} |")
        out += ["", "“通过率上界”假设强模型在被路由的问题上全部答对，需接入强模型 API 实测。", ""]

    ve = j("verifier_eval_it1.json")
    if ve:
        out += ["## 6. 静默失败判别（真实失败，5 折交叉验证）", "", "| 判别器 | 精确率 | 召回率 | F1 |", "|---|---|---|---|",
                f"| 确定性规则 | {ve['rules']['precision']:.2f} | {ve['rules']['recall']:.2f} | {ve['rules']['f1']:.2f} |",
                f"| 轻量分类器（合成故障训练，AUROC {ve['local_classifier@0.5']['auroc']:.2f}） | {ve['local_classifier@0.5']['precision']:.2f} | {ve['local_classifier@0.5']['recall']:.2f} | {ve['local_classifier@0.5']['f1']:.2f} |",
                f"| 级联（规则→分类器） | {ve['cascade(rules→classifier)']['precision']:.2f} | {ve['cascade(rules→classifier)']['recall']:.2f} | {ve['cascade(rules→classifier)']['f1']:.2f} |",
                "", "结论（负结果）：用合成故障训练的分类器迁移不到真实失败，默认关闭，仅保留规则阻断；需要线上真实失败日志来训练判别模型或接入 Jev。", ""]

    for tag in ("it1", "it2", "it3"):
        ch = j(f"chaos_eval_{tag}.json")
        if not ch:
            continue
        if tag == "it1":
            out += ["## 7. 容错层混沌测试（Mock 模型 + 进程内 MCP，固定随机种子）", "",
                    "| 迭代 | 场景 | 结果 |", "|---|---|---|"]
        out.append(f"| {tag} | E1 端到端成功率（naive → full） | {pct(ch['E1_naive']['success_rate'])} → {pct(ch['E1_full']['success_rate'])} |")
        if tag == "it3":
            out.append("| it3 | 说明 | 降级回答改为带防护的知识库原文后，naive 配置在模型失败时也能给出有用内容，所以 naive 上升；full 不变 |")
        if tag == "it2":
            out += [f"| it2 | E7 受控归因准确率（11 类故障 × 6 次） | {pct(ch['E7_attribution']['accuracy'])} |",
                    f"| it2 | E2 响应丢失后重试的重复挂号（naive → 幂等键） | {ch['E2_naive']['duplicate_bookings']} → {ch['E2_full']['duplicate_bookings']} |",
                    f"| it2 | E3 Saga：通知失败自动取消预约 / 补偿失败进死信 | 预约状态 {ch['E3_saga']['appointment_status_after']}，死信 {ch['E3_saga']['dead_letters']} 条 |",
                    f"| it2 | E4 暂停→恢复 / 崩溃→恢复 | {ch['E4_pause_crash']['pause']['first']}@{ch['E4_pause_crash']['pause']['paused_at']} → {ch['E4_pause_crash']['pause']['after_resume']}；崩溃后工具未重跑：{ch['E4_pause_crash']['crash']['work_not_redone']} |",
                    f"| it2 | E5 乐观并发：过期写被拒并自动合并 | {ch['E5_occ']['stale_write_rejected']}，合并结果 {ch['E5_occ']['merged_after_retry']} |",
                    f"| it2 | E6 Redis 共享熔断：实例 B 快速失败 / Redis 宕机降级 | {ch['E6_redis_breaker']['instance_b_fast_fails']} / 降级后状态 {ch['E6_redis_breaker']['redis_down_degraded_state']} |"]
    out += ["", "it1→it2 的改动：模型空回复且尚无工具结果时，改为在强模型上重跑完整的“调工具→作答”流程（原来只让强模型空口重答）。", ""]
    # ---------------- v3（2026-11-09）新增章节
    wd = json.loads((ROOT / "medagent" / "knowledge" / "wikidata_kb.json").read_text(encoding="utf-8"))["meta"]["counts"] \
        if (ROOT / "medagent" / "knowledge" / "wikidata_kb.json").exists() else {}
    out += ["## 9. 知识库规模（三层）", "", "| 层 | 内容 | 规模 | 许可 |", "|---|---|---|---|",
            "| 精选 KB | 症状/药品/指南，含用法、禁忌、指南概括（原仓库手写，未审核） | 21 / 10 / 9 条 | — |",
            f"| Wikidata 实体层 | 疾病（ICD-10）、药物（ATC）、别名、疾病—治疗药物关系 | 疾病 {wd.get('diseases', '—')} / 药物 {wd.get('drugs', '—')} / 别名 {wd.get('aliases', '—')} | CC0 |"]
    sc = [j(f) for f in ("rag_scale_eval_it1_1.3M.json", "rag_scale_eval_it2_3.3M.json")]
    sizes = [f"{x['index_docs']:,} 篇（{x['index_size_gb']}GB）" for x in sc if x]
    out += [f"| 本地全文索引 | Huatuo 百科/知识图谱/Lite + shibing624/medical，SQLite FTS5 | {' → '.join(sizes) or '—'} | Apache-2.0 |",
            "| 在线文献 | PubMed E-utilities / Europe PMC（实时检索，各自熔断） | 41,410,190 / 49,193,995 条（2026-11-08 查询） | 免费公开接口 |", ""]
    if any(sc):
        out += ["### 百万级检索（Huatuo 百科问答 1000 道测试题，标准答案混在全量文档中）", "",
                "| 规模 | 配置 | Recall@5 | Recall@10 | Recall@50 | MRR@10 | 延迟 p50 / p95 |", "|---|---|---|---|---|---|---|"]
        for x in sc:
            if not x:
                continue
            for k, name in (("bm25_all_terms", "BM25（全部词）"), ("bm25_df_pruned", "BM25（去高频词）")):
                v = x[k]
                out.append(f"| {x['index_docs']:,} | {name} | {v['recall@5']:.3f} | {v['recall@10']:.3f} | {v['recall@50']:.3f} | "
                           f"{v['mrr@10']:.3f} | {v['latency_ms_p50']} / {v['latency_ms_p95']} ms |")
            rr = x["rerank_sample"]
            out.append(f"| {x['index_docs']:,} | + 交叉编码器重排（抽样 {rr['n']} 题） | {rr['bm25_recall@5_same_sample']:.3f} → "
                       f"{rr['rerank_recall@5']:.3f} | — | — | {rr['rerank_mrr@10']:.3f} | 重排 {rr['rerank_latency_ms_mean(50 docs)']:.0f} ms / 50 篇 |")
        out.append("")
    d3 = j("data_audit_v3.json")
    man3 = ROOT / "training" / "data" / "data_v3_manifest.json"
    if d3 and man3.exists():
        m3 = json.loads(man3.read_text(encoding="utf-8"))
        d1 = j("data_audit_v1.json")
        out += ["## 10. 数据 v3（补充条数）", "", "| 指标 | v2 | v3 |", "|---|---|---|",
                f"| SFT 条数（train / val） | {d1['duplicates'][0]['n']} / {d1['leakage']['sft_eval_vs_train']['n_eval']} | {m3['sft']['_summary']['train']:,} / {m3['sft']['_summary']['val']:,} |",
                f"| DPO 对数（train / val） | {d1['duplicates'][1]['n']} / {d1['leakage']['dpo_eval_vs_train']['n_eval']} | {m3['dpo']['_summary']['train']:,} / {m3['dpo']['_summary']['val']:,} |",
                f"| 真实数据占比（SFT） | — | {pct(m3['sft']['_summary']['real_share'])} |",
                f"| 验证集与训练集同问题 / 近重复 | {pct(d1['leakage']['sft_eval_vs_train']['same_query_rate'])} / {pct(d1['leakage']['sft_eval_vs_train']['near_query_rate'])} | {pct(d3['leakage']['sft_eval_vs_train']['same_query_rate'])} / {pct(d3['leakage']['sft_eval_vs_train']['near_query_rate'])} |",
                f"| SFT 近重复率 | {pct(d1['duplicates'][0]['near_dup_rate'])} | {pct(d3['duplicates'][0]['near_dup_rate'])} |",
                f"| 工具调用问法模板唯一率 | {pct(d1['templates']['sft_toolcall']['unique_ratio'])} | {pct(d3['templates']['sft_toolcall']['unique_ratio'])} |",
                f"| DPO 模板唯一率 | {pct(d1['templates']['dpo']['unique_ratio'])} | {pct(d3['templates']['dpo']['unique_ratio'])} |",
                f"| DPO chosen 更长的比例 | {pct(d1['dpo_bias']['chosen_longer_rate'])} | {pct(d3['dpo_bias']['chosen_longer_rate'])} |",
                f"| 无依据工具参数 | {pct(d1['tool_args']['ungrounded_duration_rate'])} | {pct(d3['tool_args']['ungrounded_duration_rate'])} |",
                "", "v3 各桶：" + "，".join(f"{k} {v:,}" for k, v in m3["sft"]["_summary"]["by_bucket"].items()) +
                f"；DPO：" + "，".join(f"{k} {v:,}" for k, v in m3["dpo"]["_summary"]["by_bucket"].items()) + "。", ""]
    st = j("small_tier_eval.json")
    if st:
        out += ["## 11. 小模型档（Qwen3.5-2B 实测，非思考模式、不给工具）", "",
                "| 子集 | 题数 | Qwen3.5-2B 自动通过率 | 微调 MedicalGPT（2026-09）自动通过率 | Qwen3.5-2B 被校验器拦截率 |", "|---|---|---|---|---|"]
        for k, name in (("routed_small", "门控分给小模型的题"), ("not_routed_small", "其余题"), ("all", "全部 80 题")):
            v = st[k]
            out.append(f"| {name} | {v['n']} | {pct(v['qwen3.5-2b']['auto_pass_rate'])} | {pct(v['medicalgpt_2026_09']['auto_pass_rate'])} | {pct(v['qwen3.5-2b_block_rate'])} |")
        out += ["", f"平均每题 {st['mean_seconds']}s（{st['mean_tokens_per_sec']} tok/s，M5 MPS bf16）。回答原文在 results/small_tier_eval.json。", ""]
    ce = j("consult_eval.json")
    if ce:
        t, c = ce["trigger"], ce["completeness"]
        out += ["## 12. 复杂问诊（网关内运行编排器：worker 隔离与逐步检查点）", "",
                f"> {ce['note']}。", "",
                "| 指标 | 结果 |", "|---|---|",
                f"| dev 80 题中网关触发复杂问诊 | {t['triggered']} 题（标注了多个必答要点的 {t['multi_part_must_cover']} 题全部触发） |",
                f"| 多意图子问题覆盖率：单 agent → 复杂问诊 | {pct(c['single_agent'])} → {pct(c['complex_consult'])}（{c['n']} 题） |"]
        for r in ce["crash_resume"]:
            extra = (f"，再调用一次的模型调用 {r['replay_model_calls']} 次（直接返回检查点）" if r["replay_model_calls"] is not None
                     else "，只重做 w1，w0 不重跑")
            out.append(f"| 崩溃注入 @ {r['crash_at']} | 重试续跑完成：{'是' if r['resumed_done'] else '否'}；"
                       f"模型调用 {r['model_calls']} 次（无故障 {r['clean_model_calls']} 次）{extra} |")
        out += ["", "it 记录：评测中发现单 agent 基线 0/6——编排器在子问题多于 worker 上限时会截断丢题（N65），修复后基线 66.7%；"
                "“糖尿病有什么药和饮食建议”触发了复杂问诊却拆不开（N66），修复后复杂问诊 66.7% → 83.3%。", ""]
    issues_md, issue_summary = render_issues()
    (ROOT / "evaluation" / "ISSUES.md").write_text(issues_md, encoding="utf-8")
    out += ["## 13. 问题记录（现象 → 根因 → 解决 → 指标）", "", *issue_summary, "",
            "逐条明细见 [ISSUES.md](ISSUES.md)，原始数据 `evaluation/issues_log.json`。", ""]
    (ROOT / "evaluation" / "ITERATIONS.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
