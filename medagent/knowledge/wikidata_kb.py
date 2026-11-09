#!/usr/bin/env python3
"""从 Wikidata（CC0）构建扩展结构化知识库：疾病（有 ICD-10 编码）、药物（有 ATC 编码）及“疾病—症状”“疾病—治疗药物”关系。

为什么用 Wikidata：中文侧没有开放的权威药典 / 指南 API；Wikidata 是 CC0、可程序化获取、带 ICD/ATC 等标准编码，
适合做“实体层”（名字、别名、编码、关系），让工具能认出几千个疾病和药物、给出带编码的结构化信息。
它不是临床指南也没有剂量：扩展条目一律标注 source=Wikidata、reviewed=false，剂量与禁忌只引导去查说明书 / openFDA。

用法：python -m medagent.knowledge.wikidata_kb   → medagent/knowledge/wikidata_kb.json
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import httpx

ENDPOINT = "https://query.wikidata.org/sparql"
UA = "medagent-kb-builder/0.2 (research; https://github.com/Gaogxx/MedicalGPT_agent)"
OUT = Path(__file__).parent / "wikidata_kb.json"
ZH = 'FILTER(LANG(?{v}) IN ("zh", "zh-hans", "zh-cn"))'

QUERIES = {
    "diseases": f"""SELECT ?d ?zh ?en ?icd ?desc WHERE {{
        ?d wdt:P494 ?icd . ?d rdfs:label ?zh . {ZH.format(v='zh')}
        OPTIONAL {{ ?d rdfs:label ?en . FILTER(LANG(?en) = "en") }}
        OPTIONAL {{ ?d schema:description ?desc . {ZH.format(v='desc')} }} }}""",
    "drugs": f"""SELECT ?d ?zh ?en ?atc ?desc WHERE {{
        ?d wdt:P267 ?atc . ?d rdfs:label ?zh . {ZH.format(v='zh')}
        OPTIONAL {{ ?d rdfs:label ?en . FILTER(LANG(?en) = "en") }}
        OPTIONAL {{ ?d schema:description ?desc . {ZH.format(v='desc')} }} }}""",
    "symptoms": f"""SELECT ?d ?szh WHERE {{ ?d wdt:P494 ?icd . ?d wdt:P780 ?s . ?s rdfs:label ?szh . {ZH.format(v='szh')} }}""",
    "treatments": f"""SELECT ?d ?m ?mzh WHERE {{ ?d wdt:P494 ?icd . ?d wdt:P2176 ?m . ?m rdfs:label ?mzh . {ZH.format(v='mzh')} }}""",
    "aliases": f"""SELECT ?d ?alias WHERE {{ {{ ?d wdt:P494 ?c }} UNION {{ ?d wdt:P267 ?c }}
        ?d skos:altLabel ?alias . {ZH.format(v='alias')} }}""",
}


def run(query: str, tries: int = 3) -> list[dict[str, str]]:
    for i in range(tries):
        try:
            r = httpx.get(ENDPOINT, params={"query": query}, timeout=120,
                          headers={"Accept": "application/sparql-results+json", "User-Agent": UA})
            r.raise_for_status()
            return [{k: v["value"] for k, v in b.items()} for b in r.json()["results"]["bindings"]]
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))
    return []


def qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def _clean(res: dict[str, list[dict[str, str]]]) -> dict[str, int]:
    """繁→简（原写法保留为别名）；丢弃明显的脏数据：人名（含“·”）、纯拉丁字母、单字。"""
    try:
        from opencc import OpenCC
        t2s = OpenCC("t2s").convert
    except ImportError:  # pragma: no cover
        t2s = lambda x: x  # noqa: E731
    dropped = 0
    for rows in res.values():
        keep = []
        for r in rows:
            for k in ("zh", "szh", "mzh", "alias", "desc"):
                if k in r:
                    if k != "desc" and ("·" in r[k] or r[k].isascii() or len(r[k]) < 2):
                        r["_bad"] = True
                    simp = t2s(r[k])
                    if k in ("zh", "szh", "mzh") and simp != r[k]:
                        r.setdefault("_orig", []).append(r[k])
                    r[k] = simp
            if r.pop("_bad", False):
                dropped += 1
                continue
            keep.append(r)
        rows[:] = keep
    return {"dropped_rows": dropped}


def build() -> dict:
    res = {k: run(q) for k, q in QUERIES.items()}
    cleaning = _clean(res)
    diseases: dict[str, dict] = {}
    for r in res["diseases"]:
        e = diseases.setdefault(qid(r["d"]), {"zh": r["zh"], "en": r.get("en", ""), "icd10": [], "desc": r.get("desc", ""),
                                             "aliases": list(r.get("_orig", [])), "symptoms": [], "drugs": []})
        if r["icd"] not in e["icd10"]:
            e["icd10"].append(r["icd"])
    drugs: dict[str, dict] = {}
    for r in res["drugs"]:
        e = drugs.setdefault(qid(r["d"]), {"zh": r["zh"], "en": r.get("en", ""), "atc": [], "desc": r.get("desc", ""),
                                          "aliases": list(r.get("_orig", [])), "treats": []})
        if r["atc"] not in e["atc"]:
            e["atc"].append(r["atc"])
    for r in res["symptoms"]:
        d = diseases.get(qid(r["d"]))
        if d is not None and r["szh"] not in d["symptoms"]:
            d["symptoms"].append(r["szh"])
    for r in res["treatments"]:
        d = diseases.get(qid(r["d"]))
        if d is not None and r["mzh"] not in d["drugs"]:
            d["drugs"].append(r["mzh"])
            m = drugs.get(qid(r["m"]))
            if m is not None and d["zh"] not in m["treats"]:
                m["treats"].append(d["zh"])
    for r in res["aliases"]:
        e = diseases.get(qid(r["d"])) or drugs.get(qid(r["d"]))
        if e is not None and r["alias"] != e["zh"] and r["alias"] not in e["aliases"] and len(r["alias"]) >= 2:
            e["aliases"].append(r["alias"])
    meta = {"source": "Wikidata (https://www.wikidata.org)", "license": "CC0 1.0", "retrieved": date.today().isoformat(),
            "endpoint": ENDPOINT, "queries": QUERIES,
            "counts": {"diseases": len(diseases), "drugs": len(drugs),
                       "disease_symptom_links": sum(len(d["symptoms"]) for d in diseases.values()),
                       "disease_drug_links": sum(len(d["drugs"]) for d in diseases.values()),
                       "aliases": sum(len(e["aliases"]) for e in list(diseases.values()) + list(drugs.values()))},
            "cleaning": cleaning,
            "warning": "实体层数据（名称/别名/编码/关系），不是临床指南，没有剂量；reviewed=false"}
    return {"meta": meta, "diseases": diseases, "drugs": drugs}


def main() -> None:
    kb = build()
    OUT.write_text(json.dumps(kb, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(kb["meta"]["counts"], ensure_ascii=False), OUT, f"{OUT.stat().st_size / 1e6:.1f}MB")


if __name__ == "__main__":
    main()
