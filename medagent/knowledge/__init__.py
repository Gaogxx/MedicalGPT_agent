"""内置结构化医疗知识库（症状 / 药品 / 指南 / 同义词）。

原先分散在 runtime/dsh-medical-plugins/src/index.ts 与 training/scripts/build_toolcall_data.py
两处、且内容不一致（训练与服务不一致，见 build_kb.py 文件头）；现由 build_kb.py 合并为 kb.json 一份真源，
训练造数与运行时工具读同一份数据，每个条目的来源记录在 kb.json["provenance"]。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any

_DIR = Path(__file__).parent

# 公认的高风险药物相互作用（仅收录 kb 中出现的药物；表述保守，结论均为“需医生/药师评估”）
_INTERACTION_NOTES: dict[frozenset[str], tuple[str, str]] = {
    frozenset({"布洛芬", "阿司匹林"}): ("moderate", "布洛芬可能减弱低剂量阿司匹林的抗血小板作用，并叠加胃肠道出血风险"),
    frozenset({"布洛芬", "华法林"}): ("major", "非甾体抗炎药与华法林合用可增加出血风险"),
    frozenset({"阿司匹林", "华法林"}): ("major", "合用显著增加出血风险"),
    frozenset({"布洛芬", "甲氨蝶呤"}): ("major", "可能降低甲氨蝶呤清除、增加其毒性"),
    frozenset({"阿司匹林", "甲氨蝶呤"}): ("major", "可能降低甲氨蝶呤清除、增加其毒性"),
    frozenset({"对乙酰氨基酚", "华法林"}): ("moderate", "长期规律使用可能升高 INR，需监测"),
    frozenset({"阿莫西林", "华法林"}): ("moderate", "抗菌药可能影响 INR，需监测凝血"),
    frozenset({"头孢", "华法林"}): ("moderate", "部分头孢菌素可能增强抗凝作用，需监测凝血"),
    frozenset({"甲硝唑", "华法林"}): ("major", "甲硝唑可明显增强华法林抗凝作用"),
    frozenset({"奥美拉唑", "华法林"}): ("minor", "可能轻度影响华法林代谢，需监测 INR"),
}


@dataclass
class KnowledgeBase:
    version: str
    symptoms: dict[str, dict[str, Any]]
    drugs: dict[str, dict[str, Any]]
    guidelines: dict[str, dict[str, Any]]
    synonyms: dict[str, dict[str, Any]]
    meta: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None

    def source_of(self, section: str, key: str) -> dict[str, Any]:
        return (self.provenance or {}).get(f"{section}:{key}", {"from": [], "reviewed": False})

    # ---------- 规范化 ----------
    def normalize(self, text: str, kind: str) -> str:
        """口语 → 标准术语（最长匹配优先），找不到返回原文。"""
        table = self.synonyms.get(kind, {})
        for alias in sorted(table, key=len, reverse=True):
            if alias in text:
                return table[alias]
        return text

    def link_entities(self, text: str) -> dict[str, list[str]]:
        """简单实体链接：在文本中找出 kb 中的症状 / 药品 / 疾病（含同义词）。"""
        found: dict[str, list[str]] = {"symptom": [], "drug": [], "disease": []}
        for kind, vocab in (("symptom", self.symptoms), ("drug", self.drugs), ("disease", self.guidelines)):
            for key in vocab:
                if key in text and key not in found[kind]:
                    found[kind].append(key)
            for alias, std in self.synonyms.get(kind, {}).items():
                if alias in text and std in vocab and std not in found[kind]:
                    found[kind].append(std)
        return found

    # ---------- 工具语义 ----------
    def check_symptom(self, symptom: str, duration: str | None = None) -> dict[str, Any]:
        query = symptom.strip()
        hits = [k for k in self.symptoms if k in query]
        if not hits:
            std = self.normalize(query, "symptom")
            hits = [std] if std in self.symptoms else []
        if not hits:
            # 不回退到 Wikidata 的“疾病—症状”关系：没有发病率排序，“焦虑→番木鳖碱中毒”“充血→鼠疫”这类结果比查不到更危险
            return {"found": False, "symptom": symptom, "possible_causes": [],
                    "recommendation": "症状信息不足或不在知识库覆盖范围，建议线下就诊详细检查", "urgency": "unknown",
                    "confidence": 0.0}
        # 多症状时取最紧急的那条为主
        order = {"emergency": 3, "medium": 2, "low": 1}
        hits.sort(key=lambda k: order.get(self.symptoms[k]["urgency"], 0), reverse=True)
        main = dict(self.symptoms[hits[0]])
        main.update({"found": True, "symptom": symptom, "matched": hits})
        if duration:
            main["duration"] = duration
        return main

    def _label_extra(self, name: str, query: str = "") -> dict[str, Any]:
        """精选库已并入 drug_kb.json：精选字段原样返回，只补上精选库没有的说明书信息
        （所问剂型的用法用量、相互作用说明、黑框警告）。精选库的用法用量是常用口服剂型。"""
        from .drug_label_kb import form_of
        from .ext import form_view, load_ext
        ext = load_ext()
        lab = None
        if ext and ext.labels:
            hit = ext.lookup.get(name)
            lab = ext.labels.get(hit[1]) if hit and hit[0] == "drug" else ext.labels.get(f"curated:{name}")
        if not lab:
            return {}
        out: dict[str, Any] = {}
        asked = form_of(query) if query else None
        fv = form_view(lab, query)
        if fv.get("dosage_form"):
            out.update({"dosage_form": fv["dosage_form"], "dosage_for_form": fv.get("dosage"),
                        "form_note": "dosage 为精选库的常用口服剂型用法；所问剂型的说明书用法见 dosage_for_form"})
        elif asked and asked != "口服":
            out["form_note"] = f"dosage 为常用口服剂型用法；知识库没有{asked}剂型的用法用量，不同剂型剂量不能互换，请以所用药品说明书或医嘱为准"
        # 没问剂型时不附各剂型用法：精选库的用法就是常用口服剂型，附上别的剂型反而让模型不敢给剂量（“二甲双胍怎么吃”实测退步）
        if lab.get("interactions_text"):
            out["interaction_notes"] = lab["interactions_text"]
        if lab.get("boxed_warning"):
            out["boxed_warning"] = lab["boxed_warning"]
        return out

    def search_drug(self, name: str) -> dict[str, Any]:
        q = name.strip()
        if not q:
            return {"found": False, "error": "药品名为空"}
        if q in self.drugs:
            return {"found": True, **self.drugs[q], **self._label_extra(q, q)}
        std = self.normalize(q, "drug")
        if std in self.drugs:
            return {"found": True, "matched": f"别名「{q}」→「{std}」", **self.drugs[std], **self._label_extra(std, q)}
        from .ext import load_ext
        ext = load_ext()
        # 问题里有比精选库药名更具体的药名时用具体的（“头孢曲松”不能落到精选库的类别条目“头孢”）
        longer = next((n for n in ext._sorted if n in q and ext.lookup[n][0] == "drug"), "") if ext else ""
        for dn, info in self.drugs.items():
            if dn in q and not (len(longer) > len(dn) and dn in longer):
                return {"found": True, "matched": f"已匹配药品「{dn}」", **info, **self._label_extra(dn, q)}
        ext_hit = ext.find(q, "drug") if ext else None
        if ext_hit:
            entry = {**ext.drug_entry(ext_hit[0], q), "query": name}
            missing = [f for f in ("contraindications", "dosage", "side_effects") if "请查阅" in str(entry.get(f, ""))
                       or "不含" in str(entry.get(f, "")) or "未记录" in str(entry.get(f, ""))]
            if missing:
                from .wiki import record_gap
                record_gap(entry["name"], missing)
            return entry
        for cat, members in self.synonyms.get("category", {}).items():
            if cat in q:
                return {"found": False, "is_category": True, "query": name, "related_drugs": members,
                        "suggestion": f"「{name}」是药物类别而非具体药名，相关药品：{'、'.join(members)}；请用具体药名查询。"}
        # 症状/适应症关键词反查（迁移自原 index.ts：“头痛”“退烧”“止痛药” → 相关药品），
        # 用于“先查症状再查药”但模型把症状当药名传进来的情况
        hit = [kw for kw in self.synonyms.get("symptom_keywords", []) if kw in q]
        if hit:
            related = [dn for dn, info in self.drugs.items() if any(kw in info["indications"] for kw in hit)]
            if related:
                return {"found": False, "is_category": True, "query": name, "matched_by": hit[0], "related_drugs": related,
                        "suggestion": f"「{name}」不是具体药名。针对「{hit[0]}」可参考的药品：{'、'.join(related)}；"
                                      "请用具体药名查询用法与注意事项，具体用药请遵医嘱。"}
        from .wiki import record_gap
        record_gap(q, ["not_found"])
        return {"found": False, "query": name, "error": f"知识库中未找到药品「{name}」"}

    def check_interaction(self, drugs: list[str]) -> dict[str, Any]:
        names = []
        for d in drugs:
            std = self.normalize(d.strip(), "drug")
            names.append(std if std else d)
        from .interactions import _resolve, enabled, label_pair
        pairs, checked = [], []
        for a, b in combinations(dict.fromkeys(names), 2):
            note = _INTERACTION_NOTES.get(frozenset({a, b}))
            listed = b in self.drugs.get(a, {}).get("interactions", []) or a in self.drugs.get(b, {}).get("interactions", [])
            if note or listed:
                sev, desc = note or ("unknown", "说明书提示存在相互作用")
                pairs.append({"drugs": [a, b], "severity": sev, "description": desc})
                continue
            lp = label_pair(a, b) if enabled() else None     # 精选库没有 → 查两份说明书的相互作用段落里有没有提到对方
            if lp and lp["found"]:
                pairs.append({"drugs": [a, b], "severity": "label",
                              "description": "说明书原文：" + lp["found"][0]["text"],
                              "label_evidence": lp["found"]})
            elif lp:
                checked += [n for n in lp["labels_checked"] if n not in checked]
        known = set(self.drugs) | {"华法林", "甲氨蝶呤"}
        unknown = [n for n in names if n not in known and not (enabled() and _resolve(n))]
        out = {"drugs": names, "interactions": pairs, "unknown_drugs": unknown,
               "advice": "存在相互作用时请勿自行合用，咨询医生或药师" if pairs else "知识库未发现已知相互作用，但不代表绝对安全"}
        if checked and not pairs and not unknown:
            out["labels_checked"] = checked
            out["advice"] = (f"已查{'、'.join(checked)}说明书的相互作用部分，没有提到与另一药合用的问题；"
                             "这不代表绝对安全，合用前请咨询医生或药师")
        if unknown and not pairs:      # 有药查不到：先说查不到，不能让人以为“说明书没提到”就等于这个药存在
            out["advice"] = (f"知识库中没有查到「{'、'.join(unknown)}」，无法判断相互作用；请核对药名（警惕来路不明的“特效药”），"
                             "合用前咨询医生或药师")
        return out

    def _wiki_drugs(self, q: str, aspect: str | None, always: bool) -> dict[str, Any]:
        """医药 Wiki 疾病页：说明书适应症写明了这个病的药（经模型复核）。精选指南只在问用药时附上，
        免得把药名清单塞进饮食 / 运动类问题。MEDAGENT_WIKI_TOOLS=0 或 MEDAGENT_WIKI=0 关闭。"""
        from .interactions import enabled
        from .wiki import DRUG_CHOICE_INTENT, load_wiki
        if not enabled() or not (always or DRUG_CHOICE_INTENT.search(f"{q} {aspect or ''}") or "药" in (aspect or "")):
            return {}
        w = load_wiki()
        dd = w.disease_drugs(q) if w else None
        return {"label_drugs": dd} if dd else {}

    def search_guideline(self, topic: str, aspect: str | None = None) -> dict[str, Any]:
        q = topic.strip()
        asp = {"aspect": aspect} if aspect else {}
        for key, val in self.guidelines.items():
            if key in q:
                return {"found": True, **val, **asp, **self._wiki_drugs(key, aspect, False)}
        std = self.normalize(q, "disease")
        if std in self.guidelines:
            return {"found": True, **self.guidelines[std], **asp, **self._wiki_drugs(std, aspect, False)}
        from .ext import load_ext
        ext = load_ext()
        ext_hit = ext.find(q, "disease") if ext else None
        if ext_hit:
            return {**ext.disease_entry(ext_hit[0]), **asp, **self._wiki_drugs(q, aspect, True)}
        wd = self._wiki_drugs(q, aspect, True)       # Wikidata 没有、医学知识图谱有的病名（糖尿病之外的常见病多在这里）
        if wd:
            dd = wd["label_drugs"]
            return {"found": True, "topic": dd["disease"], "content": f"知识库没有「{dd['disease']}」的临床指南；"
                    "下面是说明书适应症写明了本病的药，选药须由医生根据病情决定。", **asp, **wd,
                    "source": "医药 Wiki（说明书适应症，未经审核）", "kb_tier": "wiki", "reviewed": False}
        return {"found": False, "topic": topic, "error": f"知识库中未找到「{topic}」相关指南"}

    def documents(self, include_ext: bool = True) -> list[dict[str, str]]:
        """把结构化 KB 展平为 RAG 文档（含 Wikidata 扩展实体）。"""
        docs = []
        if include_ext:
            from .ext import load_ext
            ext = load_ext()
            if ext:
                docs += ext.documents()
        for k, v in self.symptoms.items():
            docs.append({"id": f"kb:symptom:{k}", "title": k, "source": "kb/symptom",
                         "text": f"{k}：可能原因包括{'、'.join(v['possible_causes'])}。{v['recommendation']}。紧急程度：{v['urgency']}。"})
        for k, v in self.drugs.items():
            docs.append({"id": f"kb:drug:{k}", "title": k, "source": "kb/drug",
                         "text": f"{v['name']}（{v['generic_name']}）。适应症：{v['indications']}。禁忌：{v['contraindications']}。"
                                 f"用法用量：{v['dosage']}。不良反应（副作用）：{v['side_effects']}。"
                                 f"相互作用：{'、'.join(v['interactions']) or '无已知记录'}。"})
        for k, v in self.guidelines.items():
            docs.append({"id": f"kb:guideline:{k}", "title": v["topic"], "source": f"kb/guideline {v['source']}",
                         "text": f"{v['topic']}：{v['content']}（来源：{v['source']} {v['year']}）"})
        return docs


@lru_cache(maxsize=1)
def load_kb(path: str | None = None) -> KnowledgeBase:
    data = json.loads(Path(path or _DIR / "kb.json").read_text(encoding="utf-8"))
    syn = json.loads((_DIR / "synonyms.json").read_text(encoding="utf-8"))
    return KnowledgeBase(version=data["version"], symptoms=data["symptoms"], drugs=data["drugs"],
                         guidelines=data["guidelines"], synonyms=syn, meta=data.get("meta"),
                         provenance=data.get("provenance"))
