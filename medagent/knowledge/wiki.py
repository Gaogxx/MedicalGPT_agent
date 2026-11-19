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
ATC1 = {"A": "消化道及代谢", "B": "血液和造血器官", "C": "心血管系统", "D": "皮肤病用药", "G": "泌尿生殖系统和性激素",
        "H": "全身用激素类制剂（不含性激素和胰岛素）", "J": "全身用抗感染药", "L": "抗肿瘤药和免疫调节剂", "M": "肌肉-骨骼系统",
        "N": "神经系统", "P": "抗寄生虫药、杀虫药和驱虫药", "R": "呼吸系统", "S": "感觉器官", "V": "其他"}
SECTIONS = (("indications", "适应症"), ("contraindications", "禁忌"), ("dosage", "用法用量"), ("side_effects", "不良反应"),
            ("interactions_text", "相互作用"), ("boxed_warning", "黑框警告"))
DISCLAIMER = "> 未经医生 / 药师审核，仅供参考；剂量以所用药品的说明书或医嘱为准。"
_LINK = re.compile(r"\]\((\.\./(?:drugs|diseases)/[^)]+\.md)\)")


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


# ---------------------------------------------------------------- 渲染
class Builder:
    def __init__(self, drug_kb: dict, wd: dict) -> None:
        self.kb = drug_kb["drugs"]
        self.meta = drug_kb.get("meta", {})
        self.wd_drugs, self.wd_dis = wd["drugs"], wd["diseases"]
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
        dd = Counter(slug(self.wd_dis[d]["zh"]) for d in self.dis_drugs)
        self.dis_page = {d: f"diseases/{slug(self.wd_dis[d]['zh'])}" + (f"_{d}" if dd[slug(self.wd_dis[d]['zh'])] > 1 else "") + ".md"
                         for d in self.dis_drugs}
        self.name_to_drug = {}
        for q in self.drug_page:
            e = self.kb[q]
            for n in [e["name"], *e.get("aliases", []), *e.get("new_aliases", {})]:
                if len(n) >= 2:
                    self.name_to_drug.setdefault(n, q)
        self._names_sorted = sorted(self.name_to_drug, key=len, reverse=True)

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
                ds = [d for d, qs in self.dis_drugs.items() if q in qs]
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
        drugs = self.dis_drugs[d]
        out = ["---", f"qid: \"{d}\"", f"name: {json.dumps(x['zh'], ensure_ascii=False)}",
               f"icd10: {json.dumps(x['icd10'], ensure_ascii=False)}", "reviewed: false", "---",
               f"# {x['zh']}（ICD-10：{'、'.join(x['icd10'])}）", "", DISCLAIMER, ""]
        if x.get("desc"):
            out += [x["desc"], ""]
        if x.get("symptoms"):
            out += ["## 常见症状（Wikidata，不是诊断依据）", "", "、".join(x["symptoms"][:15]), ""]
        out += ["## 文献记载的治疗药物（用药须遵医嘱）", "",
                "、".join(self._a(q) for q in drugs) or "（无）", ""]
        others = [m for m in x.get("drugs", []) if m not in {self.kb[q]["name"] for q in drugs}]
        if others:
            out += ["其他（Wiki 暂无页面）：" + "、".join(others[:15]), ""]
        out += ["## 来源", "", f"- Wikidata {d}（疾病—症状、疾病—治疗药物关系，CC0）", ""]
        return "\n".join(out)

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
            ch[(self.wd_dis[d]["icd10"] or ["?"])[0][:1]].append(d)
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


def build(drug_kb_path: Path = HERE / "drug_kb.json", wiki: Path = WIKI) -> dict[str, Any]:
    kb = json.loads(drug_kb_path.read_text(encoding="utf-8"))
    wd = json.loads((HERE / "wikidata_kb.json").read_text(encoding="utf-8"))
    b = Builder(kb, wd)
    for sub in ("drugs", "diseases"):
        d = wiki / sub
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.md"):           # 页面全部由编译生成，重建前清掉旧页
            old.unlink()
    for q, p in b.drug_page.items():
        (wiki / p).write_text(b.drug(q), encoding="utf-8")
    for d, p in b.dis_page.items():
        (wiki / p).write_text(b.disease(d), encoding="utf-8")
    (wiki / "index.md").write_text(b.index(), encoding="utf-8")
    (wiki / "gaps.md").write_text(b.gaps(runtime_gaps()), encoding="utf-8")
    pages = {"drugs": {q: p for q, p in b.drug_page.items()}, "diseases": {d: p for d, p in b.dis_page.items()}}
    (wiki / "_pages.json").write_text(json.dumps(pages, ensure_ascii=False, indent=0), encoding="utf-8")
    rep = lint(wiki, kb)
    st = kb.get("meta", {}).get("stats", {})
    entry = (f"\n## {datetime.now():%Y-%m-%d %H:%M} 编译\n\n- 来源：drug_kb.json（{kb.get('meta', {}).get('built')}，"
             f"编译模型 {kb.get('meta', {}).get('compiler')}，模型调用 {kb.get('meta', {}).get('llm_calls')} 次）\n"
             f"- 页面：药品 {len(b.drug_page)}、疾病 {len(b.dis_page)}；三项齐全 {st.get('filled_core_3')}；多剂型 {st.get('multi_form', 0)}\n"
             f"- 体检：断链 {rep['n_broken']}、孤立页 {rep['n_orphans']}、缺出处段落 {rep['n_unsourced']}、"
             f"精选库与说明书剂量矛盾 {len(rep['dose_conflicts_curated_vs_label'])}、别名冲突 {len(rep['alias_clash'])}\n")
    log = wiki / "log.md"
    if not log.exists():
        log.write_text("# 更新日志（只追加）\n", encoding="utf-8")
    with open(log, "a", encoding="utf-8") as f:
        f.write(entry)
    return {"drug_pages": len(b.drug_page), "disease_pages": len(b.dis_page), **{k: v for k, v in rep.items() if k.startswith("n_")},
            "dose_conflicts": rep["dose_conflicts_curated_vs_label"], "alias_clash": rep["alias_clash"]}


# ---------------------------------------------------------------- 运行时
class Wiki:
    def __init__(self, wiki: Path = WIKI) -> None:
        self.dir = wiki
        meta = json.loads((wiki / "_pages.json").read_text(encoding="utf-8"))
        self.drugs, self.diseases = meta["drugs"], meta["diseases"]
        self.by_name: dict[str, str] = {}
        for p in [*self.drugs.values(), *self.diseases.values()]:
            text = (wiki / p).read_text(encoding="utf-8")
            m = re.search(r"^name: (.+)$", text, re.M)
            if m:
                self.by_name.setdefault(json.loads(m.group(1)), p)
            m = re.search(r"^aliases: (.+)$", text, re.M)
            for a in (json.loads(m.group(1)) if m else []):
                self.by_name.setdefault(a, p)

    def page(self, name_or_path: str) -> str | None:
        p = self.by_name.get(name_or_path) or (name_or_path if (self.dir / name_or_path).exists() else None)
        return (self.dir / p).read_text(encoding="utf-8") if p else None

    def pages_for(self, text: str, k: int = 2) -> list[str]:
        """问题里提到的 Wiki 实体 → 页面路径（药品在前）。用 ExtKB 的链接逻辑，和 search_drug 认药名的口径一致。"""
        from .ext import load_ext
        ext = load_ext()
        if not ext:
            return []
        ents = ext.link(text)
        out = [self.drugs[q] for q in ents["drug"] if q in self.drugs] + \
              [self.diseases[q] for q in ents["disease"] if q in self.diseases]
        return out[:k]

    def evidence(self, text: str, k: int = 2, chars: int = 900) -> list[dict[str, Any]]:
        out = []
        for p in self.pages_for(text, k):
            body = (self.dir / p).read_text(encoding="utf-8").split("---", 2)[-1]
            body = re.sub(r"<sub>来源：([^<]*)</sub>", r"（来源：\1）", body)
            body = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", body)
            body = re.sub(r"\n{2,}", "\n", body.replace(DISCLAIMER, "")).strip()
            title = body.split("\n", 1)[0].lstrip("# ")
            out.append({"id": f"wiki:{p}", "source": "医药 Wiki（说明书编译，未经审核）", "title": title,
                        "text": body[:chars] + ("……" if len(body) > chars else ""), "rank": 0})
        return out


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
    g = sub.add_parser("gaps")
    g.add_argument("--names-out", help="把线上缺口药名写成 JSON 列表，给 drug_label_kb.py --pilot-names 用")
    args = ap.parse_args()
    if args.cmd == "build":
        print(json.dumps(build(), ensure_ascii=False, indent=1))
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
