"""扩展结构化知识（Wikidata 实体层，约 2.4k 疾病 + 2.5k 药物）。

分层原则（越靠前越权威，工具按此顺序查）：
  1. 内置精选 KB（kb.json，40 条，含用法/禁忌/指南概括，但同样未经审核）
  2. Wikidata 实体层（本模块）：名称、别名、ICD-10 / ATC 编码、疾病—症状、疾病—治疗药物关系；**没有剂量、没有指南**
  3. 百万级本地全文索引 + 千万级在线文献（rag/），用于“查资料”而不是“给结论”
扩展条目返回时一律带 source / reviewed=false，剂量与禁忌字段明确写“知识库无此信息，请查说明书”，绝不留空让模型自由发挥。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

PATH = Path(__file__).parent / "wikidata_kb.json"
STOP = {"药物", "疾病", "病毒", "细菌", "感染", "炎症", "疼痛", "肿瘤", "癌症", "综合征", "氧气", "水", "维生素"}
NO_DOSE = "扩展知识库（Wikidata）不含剂量信息，请以药品说明书或医嘱为准"
NO_CONTRA = "扩展知识库（Wikidata）不含禁忌信息，请查阅药品说明书"


class ExtKB:
    def __init__(self, data: dict[str, Any]) -> None:
        self.meta = data["meta"]
        self.diseases: dict[str, dict] = data["diseases"]
        self.drugs: dict[str, dict] = data["drugs"]
        self.names: dict[str, tuple[str, str]] = {}
        for kind, table in (("disease", self.diseases), ("drug", self.drugs)):
            for q, e in table.items():
                for n in [e["zh"], *e["aliases"]]:
                    if len(n) >= 2 and n not in STOP:
                        self.names.setdefault(n, (kind, q))
        self.by_symptom: dict[str, list[str]] = {}
        for q, e in self.diseases.items():
            for s in e["symptoms"]:
                self.by_symptom.setdefault(s, []).append(q)
        self._sorted = sorted(self.names, key=len, reverse=True)

    def find(self, text: str, kind: str) -> tuple[str, dict] | None:
        hit = self.names.get(text.strip())
        if hit and hit[0] == kind:
            return hit[1], (self.diseases if kind == "disease" else self.drugs)[hit[1]]
        for n in self._sorted:                      # 最长匹配：问题里包含的最长实体名
            if n in text:
                k, q = self.names[n]
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
            kind, q = self.names[n]
            covered.append((i, i + len(n)))
            if q not in out[kind]:
                out[kind].append(q)
        return out

    def drug_entry(self, q: str) -> dict[str, Any]:
        e = self.drugs[q]
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


@lru_cache(maxsize=1)
def load_ext() -> ExtKB | None:
    if not PATH.exists():
        return None
    return ExtKB(json.loads(PATH.read_text(encoding="utf-8")))
