"""扩展结构化知识（Wikidata 实体层，约 2.4k 疾病 + 2.5k 药物）。

分层原则（越靠前越权威，工具按此顺序查）：
  1. 内置精选 KB（kb.json，40 条，含用法/禁忌/指南概括，但同样未经审核）
  2. Wikidata 实体层（本模块）：名称、别名、ICD-10 / ATC 编码、疾病—症状、疾病—治疗药物关系；**没有剂量、没有指南**
     2026-11-19 起药品条目可叠加 drug_kb.json（drug_label_kb.py 编译：说明书摘录 + openFDA 译写，带字段来源，试点 201 个药）
  3. 百万级本地全文索引 + 千万级在线文献（rag/），用于“查资料”而不是“给结论”
扩展条目返回时一律带 source / reviewed=false，剂量与禁忌字段明确写“知识库无此信息，请查说明书”，绝不留空让模型自由发挥。
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

PATH = Path(__file__).parent / "wikidata_kb.json"
DRUG_PATH = Path(__file__).parent / "drug_kb.json"     # 2026-11-18：补了说明书字段、并入精选库的药品条目（drug_label_kb.py 编译）
STOP = {"药物", "疾病", "病毒", "细菌", "感染", "炎症", "疼痛", "肿瘤", "癌症", "综合征", "氧气", "水", "维生素"}
NO_DOSE = "扩展知识库（Wikidata）不含剂量信息，请以药品说明书或医嘱为准"
NO_CONTRA = "扩展知识库（Wikidata）不含禁忌信息，请查阅药品说明书"
NO_SIDE = "扩展知识库未记录，请查阅说明书"
# 只在药名链接时排除（不改 self.names，本地全文索引的分词词典仍按原样）：盐、溶剂、类别词被当成药名会链错
NOT_A_DRUG = {"盐酸", "硫酸", "柠檬酸", "枸橼酸", "乙酸", "乳酸", "磷酸", "醋酸", "消毒", "疫苗", "化妆品", "血小板", "红血球",
              "尿液分析", "子宫环", "羽毛", "创可贴", "葡萄糖", "氯化钠", "乙醇", "甘油", "酚丁", "消毒药水", "止痒剂", "解毒剂"}
_SHORT_SRC = (("kb.json", "精选库"), ("Huatuo", "说明书摘录（Huatuo 百科问答）"), ("openFDA", "美国 FDA 说明书（模型译写）"),
              ("Wikidata", "Wikidata"))


class ExtKB:
    def __init__(self, data: dict[str, Any], labels: dict[str, Any] | None = None) -> None:
        self.meta = data["meta"]
        self.diseases: dict[str, dict] = data["diseases"]
        self.drugs: dict[str, dict] = data["drugs"]
        self.labels: dict[str, dict] = (labels or {}).get("drugs", {})     # QID → 补全后的条目（可能只覆盖试点药）
        self.names: dict[str, tuple[str, str]] = {}
        for kind, table in (("disease", self.diseases), ("drug", self.drugs)):
            for q, e in table.items():
                for n in [e["zh"], *e["aliases"]]:
                    if len(n) >= 2 and n not in STOP:
                        self.names.setdefault(n, (kind, q))
        # 链接用的名字表 = Wikidata 名字 + 补全时对齐出的新别名（如“多巴酚丁胺”），去掉盐 / 类别词
        self.lookup = {n: v for n, v in self.names.items() if n not in NOT_A_DRUG}
        for q, e in self.labels.items():
            if q in self.drugs:
                for n in e.get("new_aliases", {}):
                    if len(n) >= 2 and n not in NOT_A_DRUG:
                        self.lookup.setdefault(n, ("drug", q))
        self.by_symptom: dict[str, list[str]] = {}
        for q, e in self.diseases.items():
            for s in e["symptoms"]:
                self.by_symptom.setdefault(s, []).append(q)
        self._sorted = sorted(self.lookup, key=len, reverse=True)

    def find(self, text: str, kind: str) -> tuple[str, dict] | None:
        hit = self.lookup.get(text.strip())
        if hit and hit[0] == kind:
            return hit[1], (self.diseases if kind == "disease" else self.drugs)[hit[1]]
        for n in self._sorted:                      # 最长匹配：问题里包含的最长实体名
            if n in text:
                k, q = self.lookup[n]
                if k == kind:
                    return q, (self.diseases if kind == "disease" else self.drugs)[q]
        return None

    def link(self, text: str) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {"disease": [], "drug": []}
        covered: list[tuple[int, int]] = []
        for n in self._sorted:
            i = text.find(n)
            if i < 0 or any(a <= i < b for a, b in covered):
                continue
            kind, q = self.lookup[n]
            covered.append((i, i + len(n)))
            if q not in out[kind]:
                out[kind].append(q)
        return out

    def drug_entry(self, q: str, query: str = "") -> dict[str, Any]:
        """query = 调用方传进来的药名原文（如“注射用盐酸多巴酚丁胺”），用来认出问的是哪个剂型。"""
        e = self.drugs[q]
        lab = self.labels.get(q)
        if lab and lab.get("kb_tier") in ("label", "curated"):
            lab = {**lab, **form_view(lab, query)}
            src = lab.get("source", {})

            def short(f: str) -> str:
                s = src.get(f, "")
                return next((v for k, v in _SHORT_SRC if s.startswith(k)), s)
            out = {"found": True, "name": lab["name"], "generic_name": lab["generic_name"], "atc": lab["atc"],
                   "indications": lab.get("indications") or e["desc"] or "扩展知识库未记录适应症",
                   "contraindications": lab.get("contraindications") or NO_CONTRA.replace("扩展知识库（Wikidata）", "知识库"),
                   "dosage": lab.get("dosage") or NO_DOSE.replace("扩展知识库（Wikidata）", "知识库"),
                   "side_effects": lab.get("side_effects") or NO_SIDE,
                   "interactions": lab.get("interactions", [])}
            if lab.get("interactions_text"):
                out["interaction_notes"] = lab["interactions_text"]
            if lab.get("boxed_warning"):
                out["boxed_warning"] = lab["boxed_warning"]
            for k in ("dosage_form", "dosage_by_form", "form_note"):
                if lab.get(k):
                    out[k] = lab[k]
            out["field_sources"] = {f: short(f) for f in ("indications", "contraindications", "dosage", "side_effects") if f in src}
            out.update({"source": f"Wikidata {q} + 说明书摘录（未经审核）", "kb_tier": lab["kb_tier"], "reviewed": False})
            return out
        return {"found": True, "name": e["zh"], "generic_name": e["en"], "atc": e["atc"],
                "indications": ("用于：" + "、".join(e["treats"][:8])) if e["treats"] else (e["desc"] or "扩展知识库未记录适应症"),
                "contraindications": NO_CONTRA, "dosage": NO_DOSE, "side_effects": "扩展知识库未记录，请查阅说明书",
                "interactions": [], "source": f"Wikidata {q}", "kb_tier": "wikidata", "reviewed": False}

    def disease_entry(self, q: str) -> dict[str, Any]:
        e = self.diseases[q]
        parts = [f"{e['zh']}（ICD-10：{'、'.join(e['icd10'])}）" + (f"：{e['desc']}" if e["desc"] else "")]
        if e["symptoms"]:
            parts.append("常见症状：" + "、".join(e["symptoms"][:10]))
        if e["drugs"]:
            parts.append("文献记载的治疗药物：" + "、".join(e["drugs"][:10]) + "（用药须遵医嘱）")
        return {"found": True, "topic": e["zh"], "content": "；".join(parts) + "。", "icd10": e["icd10"],
                "source": f"Wikidata {q}（实体信息，非临床指南）", "year": self.meta.get("retrieved", "")[:4],
                "kb_tier": "wikidata", "reviewed": False}

    def symptom_causes(self, text: str) -> dict[str, Any] | None:
        """仅供人工浏览 / 研究：不接入 check_symptom 工具，也不用于造训练数据（无发病率排序，罕见病会排在前面）。"""
        hits = [s for s in self.by_symptom if s in text and len(s) >= 2]
        if not hits:
            return None
        hits.sort(key=len, reverse=True)
        causes = [self.diseases[q]["zh"] for q in self.by_symptom[hits[0]]][:8]
        return {"found": True, "symptom": text, "matched": hits[:3], "possible_causes": causes,
                "recommendation": "以上为 Wikidata“疾病—症状”关系中与该症状相关的疾病，仅供了解，不是诊断；症状持续或加重请就医",
                "urgency": "unknown", "confidence": 0.2, "source": "Wikidata", "kb_tier": "wikidata", "reviewed": False}

    def documents(self) -> list[dict[str, str]]:
        docs = []
        for q in self.diseases:
            d = self.disease_entry(q)
            docs.append({"id": f"wd:disease:{q}", "title": d["topic"], "source": d["source"], "text": d["content"]})
        for q, e in self.drugs.items():
            d = self.drug_entry(q)
            docs.append({"id": f"wd:drug:{q}", "title": e["zh"], "source": d["source"],
                         "text": f"{e['zh']}（{e['en']}，ATC {'、'.join(e['atc'])}）。{d['indications']}。{e['desc']}"})
        return docs


_ROUTE = {"ORAL": "口服", "INTRAVENOUS": "注射", "INTRAMUSCULAR": "注射", "SUBCUTANEOUS": "注射", "TOPICAL": "外用",
          "TRANSDERMAL": "外用", "OPHTHALMIC": "眼用", "RESPIRATORY (INHALATION)": "吸入", "NASAL": "耳鼻", "RECTAL": "栓剂",
          "VAGINAL": "阴道用", "BUCCAL": "口服", "SUBLINGUAL": "口服"}


def _product(src: str) -> str:
    return src.split("「")[1].split("」")[0] if "「" in src else ""


def form_view(lab: dict[str, Any], query: str) -> dict[str, Any]:
    """按问到的剂型改写条目（不同剂型的用法用量不能互换，注射和口服差很多）：
    · 问到的剂型药品库里有 → 用法 / 禁忌 / 不良反应换成该剂型的说明书原文；
    · 问到的剂型药品库里没有 → 用法用量改成明确说“没有这个剂型的”，不给别的剂型的剂量；
    · 没看出问的剂型、但药品库有多个剂型 → 附上各剂型的用法用量，并提示按所问剂型选用。"""
    import os
    if os.environ.get("MEDAGENT_DRUG_FORMS", "1") == "0":      # 消融用：关掉按剂型返回
        return {}
    from .drug_label_kb import form_of
    forms: dict[str, dict] = lab.get("by_form") or {}
    asked = form_of(query) if query else None
    main_prod = _product(lab.get("source", {}).get("dosage", ""))
    main_form = form_of(main_prod) if main_prod else None
    route = re.search(r"给药途径 ([A-Z ()]+)", lab.get("dosage", ""))
    if not main_prod and route:      # 主条目来自 openFDA：按给药途径定剂型（尼古丁取到的是口服含片说明书，不能拿来答贴剂）
        main_form = _ROUTE.get(route.group(1).strip())
        main_prod = f"美国 FDA 说明书（{route.group(1).strip()}）"
    if asked and asked in forms:
        r = forms[asked]
        out: dict[str, Any] = {"dosage_form": f"{asked}（{r['product']}）", "source": {**lab.get("source", {})}}
        for f in ("dosage", "contraindications", "side_effects"):
            if f in r:
                out[f] = f"（{_product(r['source'][f])}）{r[f]}"
                out["source"][f] = r["source"][f]
        return out
    if asked and main_prod and asked != (main_form or "未注明剂型") and lab.get("dosage"):
        return {"dosage": f"知识库没有{asked}剂型的用法用量（已有的是「{main_prod}」），不同剂型的剂量不能互换，"
                          "请以所用药品的说明书或医嘱为准",
                "form_note": f"问的是{asked}剂型；禁忌与不良反应取自「{main_prod}」，仅供参考"}
    if not asked and len(forms) > 1:
        return {"dosage_by_form": {f"{c}（{r['product']}）": r["dosage"][:240] for c, r in forms.items() if "dosage" in r},
                "form_note": "不同剂型的用法用量不同，回答时请与用户所问的剂型对应"}
    return {}


@lru_cache(maxsize=1)
def load_ext() -> ExtKB | None:
    if not PATH.exists():
        return None
    import os
    labels = None
    if DRUG_PATH.exists() and os.environ.get("MEDAGENT_DRUG_LABELS", "1") != "0":   # MEDAGENT_DRUG_LABELS=0 → 回到纯 Wikidata
        labels = json.loads(DRUG_PATH.read_text(encoding="utf-8"))
        from .wiki_check import apply_quarantine
        apply_quarantine(labels["drugs"])          # 独立复核不通过的字段（wiki_quarantine.json）运行时不再使用
    return ExtKB(json.loads(PATH.read_text(encoding="utf-8")), labels)
