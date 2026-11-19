#!/usr/bin/env python3
"""医药 Wiki（LLM Wiki 模式，2026-11-19）。

三层（与 Karpathy 的 LLM Wiki 一致）：
  1. 原始资料（只读）：Wikidata 实体层 wikidata_kb.json、Huatuo 百科问答 train、openFDA 说明书、精选库 kb.json
  2. Wiki：由模型编译、带出处的 Markdown 页面 —— medagent/knowledge/wiki/
       drugs/<药名>.md      每个药一页：要点 / 适应症 / 禁忌 / 用法用量（按剂型）/ 不良反应 / 相互作用 / 黑框警告 / 同类药 / 来源
       diseases/<病名>.md   与 Wiki 药品相关的疾病：ICD-10、症状、治疗药物（链到药品页）
       index.md             目录（按 ATC 大类 / ICD 章）
       log.md               编译与体检日志（只追加）
       gaps.md              缺口：缺字段的药 + 线上查询没查到的药名（下次编译优先补）
  3. 约定：wiki/SCHEMA.md（页面结构、来源优先级、哪些内容模型不能自由发挥）

编译分两步：drug_label_kb.py（GPU，模型抽取 / 核对 / 译写）→ drug_kb.json；本模块把 drug_kb.json 渲染成页面、建交叉链接、体检。
运行时：search_knowledge 命中 Wiki 实体时把对应页面放在证据最前面；MCP 资源 wiki://index、wiki://page/{name}；
search_drug 查不到 / 缺字段时把药名记到 .cache/medagent/wiki_gaps.jsonl（gaps.md 汇总，编译时用 --pilot-names 优先补）。

  python -m medagent.knowledge.wiki build     # drug_kb.json → wiki/（含体检，结果追加到 log.md）
  python -m medagent.knowledge.wiki lint
  python -m medagent.knowledge.wiki gaps --names-out gap_names.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
WIKI = HERE / "wiki"
ROOT = HERE.parents[1]
GAPS_PATH = ROOT / ".cache" / "medagent" / "wiki_gaps.jsonl"
KG_PATH = HERE / "kg_disease_drugs.json"       # 医学知识图谱“某病的推荐药”（只取药品清单式条目），wiki.py kg 生成
ATC1 = {"A": "消化道及代谢", "B": "血液和造血器官", "C": "心血管系统", "D": "皮肤病用药", "G": "泌尿生殖系统和性激素",
        "H": "全身用激素类制剂（不含性激素和胰岛素）", "J": "全身用抗感染药", "L": "抗肿瘤药和免疫调节剂", "M": "肌肉-骨骼系统",
        "N": "神经系统", "P": "抗寄生虫药、杀虫药和驱虫药", "R": "呼吸系统", "S": "感觉器官", "V": "其他"}
SECTIONS = (("indications", "适应症"), ("contraindications", "禁忌"), ("dosage", "用法用量"), ("side_effects", "不良反应"),
            ("interactions_text", "相互作用"), ("boxed_warning", "黑框警告"))
DISCLAIMER = "> 未经医生 / 药师审核，仅供参考；剂量以所用药品的说明书或医嘱为准。"
_LINK = re.compile(r"\]\((\.\./(?:drugs|diseases)/[^)]+\.md)\)")


def _h(text: str) -> str:
    import hashlib
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


REVIEWED = "> 已由{reviewer}于 {at} 审核本页内容（审核后页面一旦重新编译改动，签字自动失效）；剂量以所用药品说明书或医嘱为准。"


def page_body_hash(text: str) -> str:
    """页面内容哈希：去掉会随审核 / 编译日期变化的行（YAML 的 reviewed / reviewer / reviewed_at / updated、免责声明行）。"""
    t = re.sub(r"^(reviewed|reviewer|reviewed_at|updated): .*$\n?", "", text, flags=re.M)
    return _h(re.sub(r"^> (未经医生|已由).*$", "", t, flags=re.M))


def valid_reviews(wiki: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    path = wiki / "reviews.jsonl"
    if path.exists():
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            latest[r["page"]] = r
    return latest


def slug(name: str) -> str:
    """文件名：去掉路径符、空白和括号逗号引号（Markdown 链接里的括号会把链接截断）。"""
    return re.sub(r"[\\/:*?\"<>|\s()（）,，'’]+", "_", name).strip("_")[:60] or "unnamed"


def _src_short(s: str) -> str:
    if s.startswith("Huatuo"):
        return "说明书摘录（Huatuo 百科问答" + (s.split("Huatuo 百科问答")[1] if "Huatuo 百科问答" in s else "") + "）"
    if s.startswith("openFDA"):
        return "美国 FDA 说明书（模型译写，已核对）" + s[len("openFDA 说明书"):].split("（")[0]
    if s.startswith("kb.json"):
        return "精选库（原仓库手写）"
    return s


DIS_STOP = {"疾病", "不适", "妊娠", "肿瘤", "阵发性", "冠状病毒", "感染", "炎症", "中毒", "综合征", "病毒", "细菌"}
_NEG = re.compile(r"不用于|不适用|不能用于|不宜用于|不可用于|禁用|禁止|不推荐|尚未确立|未确定|不应用于|无效")
_CLAUSE = re.compile(r"[^，,。；;：:\n]+")


def indication_links(kb: dict[str, dict], wd_dis: dict[str, dict]) -> dict[str, list[dict[str, str]]]:
    """说明书「适应症」里提到的疾病 → 药（LLM Wiki 的“一份资料更新多页”：药品页的适应症反向写进疾病页）。
    规则：疾病名 / 别名最长匹配；被更长的病名覆盖时，只有更长的是它的亚型（“2型糖尿病”⊃“糖尿病”）才同时记上；
    所在分句有“不用于 / 禁用 / 尚未确立”等否定的不记；病名后面紧跟“病毒”（“人类免疫缺陷病毒”）的不记。"""
    names: dict[str, str] = {}
    for d, x in wd_dis.items():
        for n in [x["zh"], *x.get("aliases", [])]:
            if len(n) >= 2 and n not in DIS_STOP:
                names.setdefault(n, d)
    ordered = sorted(names, key=len, reverse=True)
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for q, e in kb.items():
        if e.get("kb_tier") not in ("label", "curated"):
            continue
        text, src = e.get("indications") or "", e.get("source", {}).get("indications") or e.get("source", {}).get("all", "")
        if not text:
            r = next((r for r in e.get("by_form", {}).values() if r.get("indications")), None)
            if r:
                text, src = r["indications"], r.get("source", {}).get("indications", "")
        if not text:
            continue
        spans: list[tuple[int, int, str]] = []
        for n in ordered:
            if n not in text:
                continue
            for m in re.finditer(re.escape(n), text):
                i, j = m.start(), m.end()
                if text[j:j + 2] == "病毒":
                    continue
                cover = [(a, b, nn) for a, b, nn in spans if a <= i and j <= b]
                if cover and not all(nn.endswith(n) for _, _, nn in cover):
                    continue
                if any(a <= i < b or a < j <= b for a, b, nn in spans if not (a <= i and j <= b)):
                    continue
                spans.append((i, j, n))
        seen: set[str] = set()
        for i, j, n in sorted(spans):
            d = names[n]
            if d in seen:
                continue
            clause = next((c for c in _CLAUSE.finditer(text) if c.start() <= i < c.end()), None)
            sent = clause.group(0).strip() if clause else text[max(0, i - 30):j + 30]
            if _NEG.search(sent):
                continue
            seen.add(d)
            out[d].append({"q": q, "snippet": sent[:90] + ("……" if len(sent) > 90 else ""), "source": src})
    return out


# ---------------------------------------------------------------- 渲染
class Builder:
    def __init__(self, drug_kb: dict, wd: dict, kg: dict[str, list[str]] | None = None,
                 link_checks: dict[str, dict] | None = None) -> None:
        self.kb = drug_kb["drugs"]
        self.meta = drug_kb.get("meta", {})
        self.wd_drugs, self.wd_dis = wd["drugs"], dict(wd["diseases"])
        # 疾病词表补充：医学知识图谱里“某病的推荐药”是药品清单的病名（Wikidata 没有糖尿病、哮喘、感冒、抑郁症这类常见病条目）
        self.kg: dict[str, list[str]] = kg or {}
        wd_names = {n: d for d, x in self.wd_dis.items() for n in [x["zh"], *x.get("aliases", [])]}
        self.kg_of: dict[str, list[str]] = {}
        for n, prods in self.kg.items():
            d = wd_names.get(n)
            if not d:
                d = f"kg:{n}"
                self.wd_dis[d] = {"zh": n, "en": "", "icd10": [], "desc": "", "aliases": [], "symptoms": [], "drugs": []}
            self.kg_of[d] = prods
        qs = [q for q, e in self.kb.items() if e.get("kb_tier") in ("label", "curated")]
        dup = Counter(slug(self.kb[q]["name"]) for q in qs)
        self.drug_page = {q: f"drugs/{slug(self.kb[q]['name'])}" + (f"_{q.split(':')[-1]}" if dup[slug(self.kb[q]['name'])] > 1 else "")
                          + ".md" for q in qs}                 # 重名（不同 QID 同中文名）加 QID 区分
        self.dis_by_zh = {d["zh"]: q for q, d in self.wd_dis.items()}
        # 与 Wiki 药品相关的疾病：Wikidata 疾病—治疗药物关系
        self.dis_drugs: dict[str, list[str]] = defaultdict(list)
        for q in self.drug_page:
            wdq = q if q in self.wd_drugs else None
            for dz in (self.wd_drugs[wdq]["treats"] if wdq else []):
                if dz in self.dis_by_zh:
                    self.dis_drugs[self.dis_by_zh[dz]].append(q)
        # 说明书适应症提到的疾病（反向链接；比 Wikidata 的疾病—治疗药物关系全得多）
        self.dis_label_raw = {d: [x for x in v if x["q"] in self.drug_page] for d, v in indication_links(self.kb, self.wd_dis).items()}
        self.dis_label_raw = {d: v for d, v in self.dis_label_raw.items() if v}
        # 模型复核（wiki_check verify-links）判为“只是提到病名、不是用于这个病”的链接不进疾病页
        checks = link_checks or {}
        self.links_dropped = 0
        self.dis_label = {}
        for d, xs in self.dis_label_raw.items():
            keep = [x for x in xs if checks.get(f"{x['q']}|{d}|{_h(self.indication_text(x['q']))}", {}).get("ok", True)]
            self.links_dropped += len(xs) - len(keep)
            if keep:
                self.dis_label[d] = keep
        dis_all = list(dict.fromkeys([*self.dis_drugs, *self.dis_label]))
        dd = Counter(slug(self.wd_dis[d]["zh"]) for d in dis_all)
        self.dis_page = {d: f"diseases/{slug(self.wd_dis[d]['zh'])}" + (f"_{d}" if dd[slug(self.wd_dis[d]['zh'])] > 1 else "") + ".md"
                         for d in dis_all}
        self.name_to_drug = {}
        for q in self.drug_page:
            e = self.kb[q]
            for n in [e["name"], *e.get("aliases", []), *e.get("new_aliases", {})]:
                if len(n) >= 2:
                    self.name_to_drug.setdefault(n, q)
        self._names_sorted = sorted(self.name_to_drug, key=len, reverse=True)
        # 知识图谱的推荐药只当候选：该药说明书适应症写明了这个病才算核实（已在 dis_label 里）；
        # 没写明的列为“待核对”，不提供给模型；该药禁忌里写了这个病的记为矛盾（体检项）
        self.kg_pending: dict[str, list[str]] = {}
        self.kg_conflicts: list[dict[str, str]] = []
        for d, prods in self.kg_of.items():
            verified = {x["q"] for x in self.dis_label.get(d, [])}
            pend = []
            for prod in prods:
                q = next((self.name_to_drug[n] for n in self._names_sorted if n in prod), None) if "复方" not in prod else None
                if q is None or q in verified:
                    continue
                names = [self.wd_dis[d]["zh"], *self.wd_dis[d].get("aliases", [])]
                contra = str(self.kb[q].get("contraindications", ""))
                if any(len(n) >= 2 and re.search(f"(?<![非无])" + re.escape(n), contra) for n in names):
                    self.kg_conflicts.append({"disease": self.wd_dis[d]["zh"], "drug": self.kb[q]["name"], "product": prod})
                elif q not in pend:
                    pend.append(q)
            if pend and d in self.dis_page:
                self.kg_pending[d] = pend

    def link_drugs(self, text: str, self_q: str) -> list[str]:
        """正文里提到的其他 Wiki 药品（最长匹配，不重叠）。"""
        hits, covered = [], []
        for n in self._names_sorted:
            i = text.find(n)
            if i < 0 or any(a <= i < b for a, b in covered):
                continue
            q = self.name_to_drug[n]
            covered.append((i, i + len(n)))
            if q != self_q and q not in hits:
                hits.append(q)
        return hits

    def _a(self, q: str, frm: str = "drugs") -> str:
        e = self.kb[q]
        return f"[{e['name']}](../{self.drug_page[q]})"

    def drug(self, q: str) -> str:
        e = self.kb[q]
        src = dict(e.get("source", {}))
        if "all" in src:                                     # 精选库独有条目（“头孢”）：一个来源覆盖全部字段
            src.update({f: src["all"] for f, _ in SECTIONS if e.get(f)})
        aliases = list(dict.fromkeys([*e.get("aliases", []), *e.get("new_aliases", {})]))
        fm = {"qid": q if not q.startswith("curated:") else None, "name": e["name"], "generic_name": e.get("generic_name"),
              "wikidata_label": e.get("wikidata_label"), "atc": e.get("atc", []), "aliases": aliases, "kb_tier": e.get("kb_tier"), "reviewed": False,
              "forms": sorted(e.get("by_form", {})), "updated": self.meta.get("built")}
        out = ["---", *[f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in fm.items() if v not in (None, [], "")], "---",
               f"# {e['name']}" + (f"（{e['generic_name']}）" if e.get("generic_name") else ""), "", DISCLAIMER, ""]
        if aliases:
            out += [f"别名：{'、'.join(aliases[:12])}", ""]
        if e.get("summary"):
            out += ["## 要点", "", e["summary"], f"<sub>来源：{_src_short(src.get('summary', '模型据本页内容概括'))}</sub>", ""]
        for f, title in SECTIONS:
            has_form_dose = f == "dosage" and any(r.get("dosage") for r in e.get("by_form", {}).values())
            if not e.get(f) and not has_form_dose:
                continue
            out += [f"## {title}", ""]
            if has_form_dose:
                if e.get("dosage"):
                    main_dose = re.sub(r"^（[^）]*）", "", e["dosage"])
                    out += [f"**常用（{src.get('dosage', '').split('「')[1].split('」')[0] if '「' in src.get('dosage', '') else '主条目'}）**：{main_dose}",
                            f"<sub>来源：{_src_short(src.get('dosage', ''))}</sub>", ""]
                for cat, r in e["by_form"].items():
                    if r.get("dosage"):
                        out += [f"### {cat}（{r['product']}）", "", r["dosage"], f"<sub>来源：{_src_short(r['source']['dosage'])}</sub>", ""]
                out += ["不同剂型的用法用量不能互换。", ""]
                continue
            out += [str(e[f]), ""]
            if f in src:
                out += [f"<sub>来源：{_src_short(src[f])}</sub>", ""]
            if f == "interactions_text":
                rel = self.link_drugs(e[f], q)
                if rel:
                    out += ["涉及的药品：" + "、".join(self._a(x) for x in rel[:15]), ""]
            if f == "indications":
                ds = list(dict.fromkeys([d for d, qs in self.dis_drugs.items() if q in qs] +
                                        [d for d, xs in self.dis_label.items() if any(x["q"] == q for x in xs)]))
                if ds:
                    out += ["相关疾病：" + "、".join(f"[{self.wd_dis[d]['zh']}](../{self.dis_page[d]})" for d in ds[:12]), ""]
        if e.get("by_form"):
            for cat, r in e["by_form"].items():
                extra = [(f, t) for f, t in (("contraindications", "禁忌"), ("side_effects", "不良反应")) if r.get(f) and r.get(f) not in str(e.get(f, ""))]
                if extra:
                    out += [f"## {cat if cat == '未注明剂型' else cat + '剂型'}的其他说明（{r['product']}）", ""]
                    for f, t in extra:
                        out += [f"**{t}**：{r[f]}", f"<sub>来源：{_src_short(r['source'][f])}</sub>", ""]
        atc4 = {a[:5] for a in e.get("atc", []) if len(a) >= 5}
        peers = [x for x in self.drug_page if x != q and atc4 & {a[:5] for a in self.kb[x].get("atc", [])}]
        if peers:
            out += ["## 同类药物（ATC 同一化学亚组）", "", "、".join(self._a(x) for x in peers[:15]), ""]
        missing = [t for f, t in (("contraindications", "禁忌"), ("dosage", "用法用量"), ("side_effects", "不良反应"))
                   if not e.get(f) and not any(r.get(f) for r in e.get("by_form", {}).values())]
        if missing:
            out += ["## 缺口", "", f"知识库暂无该药的{'、'.join(missing)}（原始资料没有，或没通过体检），请查阅药品说明书。", ""]
        out += ["## 来源", ""] + [f"- {t}：{_src_short(src[f])}" for f, t in SECTIONS if f in src] + \
               ([f"- 实体信息：Wikidata {q}（CC0）"] if not q.startswith("curated:") else []) + [""]
        return "\n".join(out)

    def disease(self, d: str) -> str:
        x = self.wd_dis[d]
        drugs = self.dis_drugs.get(d, [])
        out = ["---", f"qid: \"{d}\"", f"name: {json.dumps(x['zh'], ensure_ascii=False)}",
               *([f"aliases: {json.dumps(x['aliases'][:12], ensure_ascii=False)}"] if x.get("aliases") else []),
               f"icd10: {json.dumps(x['icd10'], ensure_ascii=False)}", "reviewed: false", "---",
               f"# {x['zh']}" + (f"（ICD-10：{'、'.join(x['icd10'])}）" if x["icd10"] else ""), "", DISCLAIMER, ""]
        if x.get("desc"):
            out += [x["desc"], ""]
        if x.get("symptoms"):
            out += ["## 常见症状（Wikidata，不是诊断依据）", "", "、".join(x["symptoms"][:15]), ""]
        lab = self.label_drugs(d)
        if lab:
            out += ["## 说明书适应症提到本病的药物（不是治疗推荐或排序，用药须遵医嘱）", ""]
            out += [f"- {self._a(x['q'])}：{x['snippet']}" for x in lab[:30]]
            if len(lab) > 30:
                out += [f"- ……另有 {len(lab) - 30} 个药，见各药品页"]
            out += ["", "<sub>来源：各药品页「适应症」（说明书摘录 / 美国 FDA 说明书译写），按病名匹配</sub>", ""]
        if drugs or not d.startswith("kg:"):
            out += ["## 文献记载的治疗药物（Wikidata，用药须遵医嘱）", "",
                    "、".join(self._a(q) for q in drugs) or "（无）", ""]
        if self.kg_pending.get(d):
            out += ["## 待核对（医学知识图谱记载、但该药说明书适应症没写明本病；不提供给问答模型）", "",
                    "、".join(self._a(q) for q in self.kg_pending[d][:20]), "",
                    "<sub>来源：Huatuo 医学知识图谱问答（huatuo_knowledge_graph_qa，“推荐药”关系，Apache-2.0）</sub>", ""]
        others = [m for m in x.get("drugs", []) if m not in {self.kb[q]["name"] for q in drugs}]
        if others:
            out += ["其他（Wiki 暂无页面）：" + "、".join(others[:15]), ""]
        out += ["## 来源", ""] + ([f"- Wikidata {d}（疾病—症状、疾病—治疗药物关系，CC0）"] if not d.startswith("kg:") else
                                 ["- 病名：Huatuo 医学知识图谱问答（Apache-2.0）"]) + \
               (["- 药品说明书「适应症」段落（经各药品页）"] if lab else []) + [""]
        return "\n".join(out)

    def indication_text(self, q: str) -> str:
        e = self.kb[q]
        if e.get("indications"):
            return e["indications"]
        return next((r["indications"] for r in e.get("by_form", {}).values() if r.get("indications")), "")

    def label_drugs(self, d: str) -> list[dict[str, str]]:
        """精选库的药在前，其余按“适应症里提到的病越少越靠前”（专用药优先于广谱说明）。"""
        n_dis = Counter(x["q"] for v in self.dis_label.values() for x in v)
        return sorted(self.dis_label.get(d, []), key=lambda x: (self.kb[x["q"]].get("kb_tier") != "curated", n_dis[x["q"]],
                                                                 self.kb[x["q"]]["name"]))

    def index(self) -> str:
        by = defaultdict(list)
        for q in self.drug_page:
            atc = (self.kb[q].get("atc") or ["?"])[0][:1]
            by[atc].append(q)
        n_full = sum(all(self.kb[q].get(f) for f in ("contraindications", "dosage", "side_effects")) for q in self.drug_page)
        out = ["# 医药 Wiki 目录", "", DISCLAIMER, "",
               f"药品 {len(self.drug_page)} 页（三项齐全 {n_full}）、疾病 {len(self.dis_page)} 页；编译于 {self.meta.get('built')}。"
               "页面约定见 [SCHEMA.md](SCHEMA.md)，更新记录见 [log.md](log.md)，缺口见 [gaps.md](gaps.md)。", "", "## 药品（按 ATC 大类）", ""]
        for k in sorted(by, key=lambda c: (c not in ATC1, c)):
            out += [f"### {k} {ATC1.get(k, '未分类')}（{len(by[k])}）", "",
                    "、".join(f"[{self.kb[q]['name']}]({self.drug_page[q]})" for q in sorted(by[k], key=lambda q: self.kb[q]["name"])), ""]
        out += ["## 疾病（按 ICD-10 章）", ""]
        ch = defaultdict(list)
        for d in self.dis_page:
            ch[(self.wd_dis[d]["icd10"] or ["未分类"])[0][:1] if self.wd_dis[d]["icd10"] else "未分类"].append(d)
        for k in sorted(ch):
            out += [f"### {k}（{len(ch[k])}）", "", "、".join(f"[{self.wd_dis[d]['zh']}]({self.dis_page[d]})"
                                                          for d in sorted(ch[k], key=lambda d: self.wd_dis[d]["zh"])), ""]
        return "\n".join(out)

    def gaps(self, runtime: Counter) -> str:
        miss = [(self.kb[q]["name"], [t for f, t in SECTIONS[1:4] if not self.kb[q].get(f)]) for q in self.drug_page]
        miss = [(n, m) for n, m in miss if m]
        out = ["# 缺口", "", "下次编译（`drug_label_kb.py --pilot-names`）优先补这些。", "",
               f"## 线上查询没查到 / 缺字段的药名（{len(runtime)}）", ""]
        out += [f"- {n}（{c} 次）" for n, c in runtime.most_common(200)] or ["（暂无记录）"]
        out += ["", f"## Wiki 页面缺字段（{len(miss)}）", ""] + [f"- {n}：缺 {'、'.join(m)}" for n, m in sorted(miss)[:500]] + [""]
        return "\n".join(out)


_PROD = re.compile(r"(片|胶囊|颗粒|丸|注射液|口服液|糖浆|软膏|乳膏|滴眼液|气雾剂|喷雾剂|栓|散|合剂|酊|贴|凝胶|混悬液|溶液|洗剂|搽剂|滴鼻液|滴耳液|粉|注射剂)$")
_KG_BAD = re.compile(r"大鼠|小鼠|兔|模型|术后|联合|组$|细胞|[A-Za-z0-9]|实验|所致|诱导")


def extract_kg(path: Path) -> dict[str, list[str]]:
    """从 huatuo_knowledge_graph_qa 取“X的推荐药有哪些”。只要答案是药品清单的条目（≥5 个、七成以上带剂型后缀）——
    那是药品网站的“疾病—常用药”数据；文献挖掘出来的（“TBI大鼠肺损伤→丙泊酚”）和病名带实验 / 联合用药字样的丢掉。"""
    out: dict[str, list[str]] = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        m = re.match(r"^(.+)的推荐药(有些什么|有哪些|是什么)", r["questions"][0])
        if not m or not 2 <= len(m.group(1)) <= 10 or _KG_BAD.search(m.group(1)):
            continue
        items = [x.strip() for x in r["answers"][0].split("；") if x.strip()]
        prods = [x for x in items if _PROD.search(x)]
        if len(prods) >= 5 and len(prods) >= 0.7 * len(items):
            out[m.group(1)] = prods[:30]
    return out


def runtime_gaps() -> Counter:
    c: Counter = Counter()
    if GAPS_PATH.exists():
        for line in open(GAPS_PATH, encoding="utf-8"):
            try:
                c[json.loads(line)["name"]] += 1
            except (json.JSONDecodeError, KeyError):
                continue
    return c


def lint(wiki: Path = WIKI, drug_kb: dict | None = None) -> dict[str, Any]:
    """体检：断链、孤立页、缺出处的段落、精选库与说明书剂量数字矛盾、同一别名指向不同页。"""
    pages = {p.relative_to(wiki).as_posix(): p.read_text(encoding="utf-8") for p in wiki.glob("*/*.md")}
    inbound: Counter = Counter()
    broken, unsourced = [], []
    for name, text in pages.items():
        for target in _LINK.findall(text):
            t = target[3:]
            if t not in pages:
                broken.append((name, t))
            inbound[t] += 1
        for sec in re.split(r"\n(?=## )", text)[1:]:
            title = sec.split("\n", 1)[0][3:]
            if title in ("适应症", "禁忌", "用法用量", "不良反应", "相互作用", "黑框警告") and "来源：" not in sec:
                unsourced.append((name, title))
    idx = (wiki / "index.md").read_text(encoding="utf-8") if (wiki / "index.md").exists() else ""
    for m in re.findall(r"\]\(((?:drugs|diseases)/[^)]+\.md)\)", idx):
        inbound[m] += 1
    orphans = [n for n in pages if inbound[n] == 0]
    conflicts, alias_clash = [], []
    if drug_kb:
        seen: dict[str, str] = {}
        for q, e in drug_kb["drugs"].items():
            if e.get("kb_tier") == "curated" and e.get("by_form", {}).get("口服", {}).get("dosage"):
                a = set(re.findall(r"\d+(?:\.\d+)?", e["dosage"]))
                b = set(re.findall(r"\d+(?:\.\d+)?", e["by_form"]["口服"]["dosage"]))
                if a and not (a & b):
                    conflicts.append({"drug": e["name"], "curated": e["dosage"][:80], "label": e["by_form"]["口服"]["dosage"][:80]})
            for n in e.get("new_aliases", {}):
                if n in seen and seen[n] != q:
                    alias_clash.append((n, seen[n], q))
                seen[n] = q
    return {"pages": len(pages), "broken_links": broken[:50], "n_broken": len(broken), "orphans": orphans[:50],
            "n_orphans": len(orphans), "unsourced_sections": unsourced[:50], "n_unsourced": len(unsourced),
            "dose_conflicts_curated_vs_label": conflicts, "alias_clash": alias_clash[:20]}


def _builder(drug_kb_path: Path = HERE / "drug_kb.json") -> tuple[dict, Builder]:
    from .wiki_check import CHECKS_PATH, apply_quarantine
    kb = json.loads(drug_kb_path.read_text(encoding="utf-8"))
    apply_quarantine(kb["drugs"])                 # 独立复核不通过、进了隔离区的字段不上页面
    wd = json.loads((HERE / "wikidata_kb.json").read_text(encoding="utf-8"))
    kg = json.loads(KG_PATH.read_text(encoding="utf-8")) if KG_PATH.exists() else None
    checks = json.loads(CHECKS_PATH.read_text(encoding="utf-8")) if CHECKS_PATH.exists() else None
    return kb, Builder(kb, wd, kg, checks)


def render_pages(drug_kb_path: Path = HERE / "drug_kb.json", wiki: Path = WIKI) -> dict[str, str]:
    """drug_kb.json → {相对路径: 页面文本}（不写盘；wiki_check check 用它比对页面有没有被手改）。
    审核签字仍然有效（页面内容哈希没变）的页面标成已审核。"""
    kb, b = _builder(drug_kb_path)
    pages = {p: b.drug(q) for q, p in b.drug_page.items()}
    pages.update({p: b.disease(d) for d, p in b.dis_page.items()})
    for p, r in valid_reviews(wiki).items():
        if p in pages and page_body_hash(pages[p]) == r["hash"]:
            t = pages[p].replace("reviewed: false", f"reviewed: true\nreviewer: {json.dumps(r['reviewer'], ensure_ascii=False)}\n"
                                                     f"reviewed_at: \"{r['at'][:10]}\"", 1)
            pages[p] = t.replace(DISCLAIMER, REVIEWED.format(reviewer=r["reviewer"], at=r["at"][:10]), 1)
    return pages


def build(drug_kb_path: Path = HERE / "drug_kb.json", wiki: Path = WIKI) -> dict[str, Any]:
    kb, b = _builder(drug_kb_path)
    pages = render_pages(drug_kb_path, wiki)
    for sub in ("drugs", "diseases"):
        d = wiki / sub
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.md"):           # 页面全部由编译生成，重建前清掉旧页
            old.unlink()
    for p, text in pages.items():
        (wiki / p).write_text(text, encoding="utf-8")
    (wiki / "index.md").write_text(b.index(), encoding="utf-8")
    (wiki / "gaps.md").write_text(b.gaps(runtime_gaps()), encoding="utf-8")
    meta = {"drugs": {q: p for q, p in b.drug_page.items()}, "diseases": {d: p for d, p in b.dis_page.items()},
            "disease_drugs": {d: [[x["q"], b.kb[x["q"]]["name"], x["snippet"]] for x in b.label_drugs(d)[:30]]
                              for d in b.dis_page if b.dis_label.get(d)}}
    (wiki / "_pages.json").write_text(json.dumps(meta, ensure_ascii=False, indent=0), encoding="utf-8")
    rep = lint(wiki, kb)
    st = kb.get("meta", {}).get("stats", {})
    entry = (f"\n## {datetime.now():%Y-%m-%d %H:%M} 编译\n\n- 来源：drug_kb.json（{kb.get('meta', {}).get('built')}，"
             f"编译模型 {kb.get('meta', {}).get('compiler')}，模型调用 {kb.get('meta', {}).get('llm_calls')} 次）\n"
             f"- 页面：药品 {len(b.drug_page)}、疾病 {len(b.dis_page)}；三项齐全 {st.get('filled_core_3')}；多剂型 {st.get('multi_form', 0)}\n"
             f"- 体检：断链 {rep['n_broken']}、孤立页 {rep['n_orphans']}、缺出处段落 {rep['n_unsourced']}、"
             f"精选库与说明书剂量矛盾 {len(rep['dose_conflicts_curated_vs_label'])}、别名冲突 {len(rep['alias_clash'])}\n"
             f"- 疾病—药物：说明书适应症链接 {sum(map(len, b.dis_label.values()))} 条；知识图谱推荐药待核对 "
             f"{sum(map(len, b.kg_pending.values()))} 条；与该药禁忌矛盾而剔除 {len(b.kg_conflicts)} 条；"
             f"模型复核判为“只是提到病名”而剔除 {b.links_dropped} 条\n")
    log = wiki / "log.md"
    if not log.exists():
        log.write_text("# 更新日志（只追加）\n", encoding="utf-8")
    with open(log, "a", encoding="utf-8") as f:
        f.write(entry)
    return {"drug_pages": len(b.drug_page), "disease_pages": len(b.dis_page), "kg_conflicts": b.kg_conflicts[:30],
            "disease_label_links": sum(map(len, b.dis_label.values())), **{k: v for k, v in rep.items() if k.startswith("n_")},
            "dose_conflicts": rep["dose_conflicts_curated_vs_label"], "alias_clash": rep["alias_clash"]}


# ---------------------------------------------------------------- 运行时
INTERACT_INTENT = re.compile(r"一起|同时|合用|同服|相互作用|搭配|联用|混着|配着|能和|可以和|跟.{1,8}吃|和.{1,8}吃")
DRUG_CHOICE_INTENT = re.compile(r"什么药|哪些药|哪种药|啥药|药物治疗|推荐.{0,4}药|用药|吃药|治疗药")


class Wiki:
    def __init__(self, wiki: Path = WIKI) -> None:
        self.dir = wiki
        meta = json.loads((wiki / "_pages.json").read_text(encoding="utf-8"))
        self.drugs, self.diseases = meta["drugs"], meta["diseases"]
        self.dis_drugs: dict[str, list[list[str]]] = meta.get("disease_drugs", {})
        self.dis_of_path = {p: d for d, p in self.diseases.items()}
        self.by_name: dict[str, str] = {}
        for p in [*self.drugs.values(), *self.diseases.values()]:
            text = (wiki / p).read_text(encoding="utf-8")
            m = re.search(r"^name: (.+)$", text, re.M)
            if m:
                self.by_name.setdefault(json.loads(m.group(1)), p)
            m = re.search(r"^aliases: (.+)$", text, re.M)
            for a in (json.loads(m.group(1)) if m else []):
                self.by_name.setdefault(a, p)
        from .ext import STOP
        self._dis_names = sorted((n for n, p in self.by_name.items() if p.startswith("diseases/") and len(n) >= 2
                                  and n not in STOP and n not in DIS_STOP), key=len, reverse=True)

    def page(self, name_or_path: str) -> str | None:
        p = self.by_name.get(name_or_path) or (name_or_path if (self.dir / name_or_path).exists() else None)
        return (self.dir / p).read_text(encoding="utf-8") if p else None

    def find_disease(self, text: str) -> str | None:
        """问题 / 病名 → 疾病页路径（精确名优先，再最长匹配）。词表 = Wikidata 疾病 + 医学知识图谱病名。"""
        t = text.strip()
        if t in self.by_name and self.by_name[t].startswith("diseases/"):
            return self.by_name[t]
        return next((self.by_name[n] for n in self._dis_names if n in t), None)

    def disease_drugs(self, text: str, k: int = 12) -> dict[str, Any] | None:
        """病名 → 说明书适应症写明了这个病的药（每个带适应症原句）。没有页面或没有链接时返回 None。"""
        p = self.find_disease(text)
        if not p:
            return None
        rows = self.dis_drugs.get(self.dis_of_path[p], [])
        if not rows:
            return None
        name = re.search(r"^name: (.+)$", (self.dir / p).read_text(encoding="utf-8"), re.M)
        return {"disease": json.loads(name.group(1)) if name else p, "page": f"wiki:{p}",
                "drugs": [{"name": n, "indication": snip} for _, n, snip in rows[:k]],
                "n_total": len(rows),
                "note": "以上是说明书适应症写明本病的药（按专用程度排列，不是治疗推荐）；具体选药须由医生根据病情决定"}

    def pages_for(self, text: str, k: int = 2) -> list[str]:
        """问题里提到的 Wiki 实体 → 页面路径（药品在前）。药名用 ExtKB 的链接逻辑（和 search_drug 同口径），病名用 Wiki 自己的词表。"""
        from .ext import load_ext
        ext = load_ext()
        out = []
        if ext:
            out = [self.drugs[q] for q in ext.link(text)["drug"] if q in self.drugs]
        from .interactions import enabled
        d = self.find_disease(text) if enabled() else next((self.diseases[q] for q in (ext.link(text)["disease"] if ext else [])
                                                           if q in self.diseases), None)
        if d and d not in out:
            out.append(d) if not (enabled() and DRUG_CHOICE_INTENT.search(text)) else out.insert(0, d)
        return out[:k]

    def _sections(self, p: str, intent: str) -> str:
        body = (self.dir / p).read_text(encoding="utf-8").split("---", 2)[-1]
        body = re.sub(r"<sub>来源：([^<]*)</sub>", r"（来源：\1）", body)
        body = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", body)
        body = body.replace(DISCLAIMER, "")
        head, *secs = re.split(r"\n(?=## )", body)
        first = {"interaction": ("## 相互作用", "## 黑框警告"),
                 "drug_choice": ("## 说明书适应症提到本病的药物",)}.get(intent, ())
        secs = [x for x in secs if x.startswith(first)] + [x for x in secs if not x.startswith(first)]
        secs = [x for x in secs if not x.startswith(("## 来源", "## 待核对", "## 同类药物"))]
        return re.sub(r"\n{2,}", "\n", "\n".join([head, *secs])).strip()

    def evidence(self, text: str, k: int = 2, chars: int = 900) -> list[dict[str, Any]]:
        """按问题意图挑页面段落：问能不能一起吃 → 相互作用段在前，并先给两份说明书互查的结果；
        问某病用什么药 → 疾病页“说明书适应症提到本病的药物”在前；其余按页面顺序。"""
        from .interactions import enabled
        intent = "" if not enabled() else \
            "interaction" if INTERACT_INTENT.search(text) else "drug_choice" if DRUG_CHOICE_INTENT.search(text) else ""
        out: list[dict[str, Any]] = []
        pages = self.pages_for(text, k)
        if intent == "interaction":
            from .ext import load_ext
            from .interactions import enabled, label_pair
            ext = load_ext()
            drugs = [ext.drugs[q]["zh"] for q in ext.link(text)["drug"]][:3] if ext else []
            for a, b in ([(drugs[0], drugs[1])] if len(drugs) >= 2 and enabled() else []):
                lp = label_pair(a, b)
                if lp and lp["labels_checked"]:
                    txt = "；".join(f"{f['label_of']}说明书：{f['text']}" for f in lp["found"]) or \
                        f"已查{'、'.join(lp['labels_checked'])}说明书的相互作用部分，没有提到与{b if lp['labels_checked'][0] == a else a}合用的问题（不代表绝对安全）"
                    out.append({"id": f"wiki:interaction:{a}|{b}", "source": "医药 Wiki · 说明书相互作用互查（未经审核）",
                                "title": f"{a} 与 {b}", "text": txt[:chars], "rank": 0})
        for p in pages:
            body = self._sections(p, intent if (intent != "drug_choice" or p.startswith("diseases/")) else "")
            title = body.split("\n", 1)[0].lstrip("# ")
            out.append({"id": f"wiki:{p}", "source": "医药 Wiki（说明书编译，未经审核）", "title": title,
                        "text": body[:chars] + ("……" if len(body) > chars else ""), "rank": 0})
        return out[: max(k, 1) + (1 if out and out[0]["id"].startswith("wiki:interaction") else 0)]


@lru_cache(maxsize=1)
def load_wiki() -> Wiki | None:
    if os.environ.get("MEDAGENT_WIKI", "1") == "0" or not (WIKI / "_pages.json").exists():
        return None
    return Wiki()


def record_gap(name: str, missing: list[str]) -> None:
    """search_drug 查不到 / 缺字段时记一笔（MEDAGENT_WIKI_GAPS=0 关闭）。只记药名和缺的字段，不记用户问题原文。"""
    if os.environ.get("MEDAGENT_WIKI_GAPS", "1") == "0":
        return
    try:
        GAPS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(GAPS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"name": name[:40], "missing": missing, "t": datetime.now().isoformat(timespec="seconds")},
                               ensure_ascii=False) + "\n")
    except OSError:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("lint")
    k = sub.add_parser("kg", help="从医学知识图谱问答抽“某病的推荐药”→ kg_disease_drugs.json（build 时用作疾病词表和待核对候选）")
    k.add_argument("--kg-path", default=str(ROOT / ".cache" / "corpus" / "FreedomIntelligence__huatuo_knowledge_graph_qa" / "train_datasets.jsonl"))
    g = sub.add_parser("gaps")
    g.add_argument("--names-out", help="把线上缺口药名写成 JSON 列表，给 drug_label_kb.py --pilot-names 用")
    args = ap.parse_args()
    if args.cmd == "build":
        print(json.dumps(build(), ensure_ascii=False, indent=1))
    elif args.cmd == "kg":
        kg = extract_kg(Path(args.kg_path))
        KG_PATH.write_text(json.dumps(kg, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(len(kg), "个病名 →", KG_PATH)
    elif args.cmd == "lint":
        kb = json.loads((HERE / "drug_kb.json").read_text(encoding="utf-8"))
        print(json.dumps(lint(WIKI, kb), ensure_ascii=False, indent=1))
    else:
        c = runtime_gaps()
        print(json.dumps(c.most_common(50), ensure_ascii=False))
        if args.names_out:
            Path(args.names_out).write_text(json.dumps([n for n, _ in c.most_common()], ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
