#!/usr/bin/env python3
"""给 Wikidata 扩展库的药品补上适应症 / 禁忌 / 用法用量 / 不良反应 / 相互作用，并把精选库并进来 → drug_kb.json。

2026-11-18 试点。思路借鉴 LLM Wiki：原始资料只读，由模型“编译”成带来源的结构化条目，再自动体检；
但医疗场景下把模型能自由发挥的地方压到最少：
  原始资料（只读，不改）
    · Wikidata 实体层 wikidata_kb.json（CC0）：药名、别名、ATC、英文名
    · Huatuo 百科问答训练集（FreedomIntelligence/huatuo_encyclopedia_qa，Apache-2.0）：大量“X片的用法用量 / 副作用（不良反应）/
      禁忌症是什么”问答，答案基本是说明书原文。只用 train 拆分（评测题取自它的 validation 拆分）
    · openFDA 药品说明书（公有领域，英文）
  编译
    1. 名称对齐：Huatuo 的商品名去括号、去剂型、去盐 → 候选通用名，等于 Wikidata 中文名 / 别名才算对上；
       对不上的请模型给英文 INN，INN 等于 Wikidata 英文名、且模型把 INN 反译回中文时与原名一致，才收为别名（source=llm_inn）
    2. 中文字段只用 Huatuo 原文（去掉“本文只供参考”之类套话）；过长时让模型挑句子编号（抽取式，一个字不改）
    3. 中文没有的字段才用 openFDA：模型把英文段落译成中文要点，体检两道——译文里的每个数字都必须在英文原文出现过；
       再让模型逐条核对“译文有没有原文没有的内容”，有就整段丢弃
    4. 精选库（kb.json）10 个药原字段全部保留（训练数据和评测都按它），只记来源；“头孢”是类别，单独成条
  所有条目 reviewed=false，每个字段都带 source；没有来源的字段保持“请查阅说明书”，不留空让模型自由发挥。

在 GPU 机上跑（需要一个 OpenAI 兼容的模型服务，默认 Qwen3.5-27B-FP8）：
  python -m medagent.knowledge.drug_label_kb --huatuo huatuo/train.jsonl --openfda-dir openfda \\
      --llm http://127.0.0.1:8100/v1 --pilot pilot_qids.json --out medagent/knowledge/drug_kb.json
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import html
import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
FIELDS = ("indications", "contraindications", "dosage", "side_effects", "interactions_text")

# Huatuo 问题模式 → 字段
_Q = re.compile(r"^(?:请描述)?(.{2,40}?)(?:的)?(用法用量|副作用（不良反应）|不良反应|禁忌症是什么？?|禁忌|适应症|功能主治|药物相互作用)$")
_QFIELD = {"用法用量": "dosage", "副作用（不良反应）": "side_effects", "不良反应": "side_effects", "禁忌": "contraindications",
           "适应症": "indications", "功能主治": "indications", "药物相互作用": "interactions_text"}
_BOILER = re.compile(r"注意：[^。]*?(包装规格|企业生产)[^。]*。(本文[^。]*。)?(如果不确定[^。]*。)?|本文(用法用量)?只供参考。|"
                     r"如果不确定，请参看药品随带的说明书或向医生询问。")
FORMS = sorted(["缓释片", "控释片", "肠溶片", "分散片", "咀嚼片", "泡腾片", "口腔崩解片", "口崩片", "含片", "薄膜衣片", "糖衣片", "片",
                "缓释胶囊", "肠溶胶囊", "软胶囊", "硬胶囊", "胶囊", "颗粒", "干混悬剂", "混悬液", "混悬剂", "口服溶液", "口服液",
                "糖浆", "滴剂", "注射液", "粉针剂", "粉针", "注射剂", "乳膏", "软膏", "凝胶", "栓", "滴眼液", "眼膏", "滴耳液", "滴鼻液",
                "喷雾剂", "气雾剂", "吸入剂", "吸入粉雾剂", "贴剂", "贴片", "散", "酊", "搽剂", "洗剂", "溶液", "冻干粉", "输液",
                "片剂", "胶丸", "微丸", "缓释微丸胶囊", "乳剂", "膜", "合剂"], key=len, reverse=True)
SALT_PRE = ("注射用", "盐酸", "硫酸", "枸橼酸", "马来酸", "甲磺酸", "苯磺酸", "磷酸", "醋酸", "富马酸", "琥珀酸", "酒石酸", "乳酸",
            "葡萄糖酸", "氢溴酸", "门冬氨酸", "苯甲酸", "甲硫酸", "萘磺酸", "硝酸", "草酸", "丙酸", "水杨酸", "二盐酸", "重酒石酸", "精氨酸")
SALT_POST = ("钠", "钾", "钙", "镁", "锌", "盐", "水合物", "一水合物", "二水合物")
NOT_A_DRUG_NAME = {"盐酸", "硫酸", "柠檬酸", "枸橼酸", "乙酸", "乳酸", "磷酸", "醋酸", "消毒", "疫苗", "化妆品", "血小板", "红血球",
                   "尿液分析", "子宫环", "羽毛", "创可贴", "葡萄糖", "氯化钠", "乙醇", "甘油", "酚丁", "消毒药水", "止痒剂", "解毒剂"}


def strip_name(name: str) -> list[str]:
    """商品名 → 由松到紧的候选通用名（复方 / 多成分直接放弃）。"""
    n = re.sub(r"[（(][^）)]*[）)]|[ⅠⅡⅢⅣ]+型?|[ABⅠⅡ?？]$|\s", "", name)
    if n.startswith("复方") or "/" in n or len(n) < 2:
        return []
    full = n                                   # 原名本身也是候选（“蒙脱石散”“葡萄糖注射液”在 Wikidata 里就叫这个）
    for _ in range(2):
        for f in FORMS:
            if n.endswith(f) and len(n) > len(f) + 1:
                n = n[: -len(f)]
                break
    out = [full, n]
    m = n
    for p in SALT_PRE:
        if m.startswith(p) and len(m) > len(p) + 1:
            m = m[len(p):]
            out.append(m)
    for s in SALT_POST:
        if m.endswith(s) and len(m) > len(s) + 2:
            out.append(m[: -len(s)])
    return list(dict.fromkeys(out))


def clean(text: str) -> str:
    t = _BOILER.sub("", html.unescape(text)).strip()
    return re.sub(r"\s+", " ", t)


# 抓取损坏（原始资料里的问题，不是我们引入的）：数字丢了（“一次片”“一日-3次”）、符号变问号（“1? 心肌梗死”）
_BROKEN = re.compile(r"\?|一次[片粒袋支丸]|每次[片粒袋支丸]|一日-|一次-|[（(]\s*[)）]|[，,]\s*[片粒]\b")
ORAL = ("片", "胶囊", "颗粒", "口服液", "口服溶液", "糖浆", "丸", "混悬液", "干混悬剂", "滴剂", "散")


# 剂型大类：同一个药不同剂型的用法用量不能互换（注射 vs 口服），不良反应 / 禁忌也常不同（栓剂的局部反应等）
FORM_CATS = (("注射", ("注射", "针剂", "粉针", "输液")), ("眼用", ("滴眼", "眼膏", "眼用")), ("耳鼻", ("滴耳", "滴鼻", "鼻喷", "鼻用")),
             ("吸入", ("吸入", "气雾", "雾化")), ("栓剂", ("栓",)), ("阴道用", ("阴道", "泡腾片（阴道）")),
             ("外用", ("乳膏", "软膏", "凝胶", "酊", "搽剂", "洗剂", "贴", "涂剂", "霜", "喷雾", "外用", "擦剂", "洗液")),
             ("口服", ORAL + ("口服", "缓释", "肠溶", "含片", "咀嚼", "分散", "泡腾", "冲剂", "糖浆", "口含")))


def form_of(text: str) -> str | None:
    """商品名 / 问题 → 剂型大类；看不出来返回 None（如只写了通用名）。"""
    for cat, keys in FORM_CATS:
        if any(k in text for k in keys):
            return cat
    return None


def looks_broken(text: str) -> bool:
    return bool(_BROKEN.search(text))


# ---------------------------------------------------------------- 原始资料
def load_huatuo(path: str) -> dict[str, dict[str, list[str]]]:
    prod: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            ans = clean(o["answers"][0])
            if len(ans) < 4:
                continue
            for q in o["questions"][0]:
                m = _Q.match(q.strip())
                if not m:
                    continue
                key = m.group(2)
                field = _QFIELD.get(key) or ("contraindications" if key.startswith("禁忌症") else None)
                name = m.group(1)
                if not field or "饮食" in name or name.endswith(("病", "症", "炎")):
                    continue
                if ans not in prod[name][field]:
                    prod[name][field].append(ans)
    return prod


def load_openfda(d: str, wanted: set[str]) -> dict[str, dict]:
    """每个单成分通用名留一份说明书：优先处方药、有禁忌段、生效日期最新。"""
    best: dict[str, dict] = {}
    secs = {"indications": "indications_and_usage", "contraindications": "contraindications",
            "dosage": "dosage_and_administration", "side_effects": "adverse_reactions", "interactions_text": "drug_interactions",
            "boxed_warning": "boxed_warning"}
    for fp in sorted(glob.glob(f"{d}/drug-label-*.json")):
        for r in json.load(open(fp, encoding="utf-8"))["results"]:
            of = r.get("openfda") or {}
            names = {x.lower() for x in of.get("generic_name", []) + of.get("substance_name", [])}
            if len(of.get("substance_name", [])) > 1:
                continue
            for n in names & wanted:
                rx = "PRESCRIPTION" in " ".join(of.get("product_type", []))
                oral = "ORAL" in of.get("route", [])          # 优先口服说明书（奥氮平曾取到肌注说明书）
                rank = (oral, rx, "contraindications" in r, r.get("effective_time", ""))
                if n not in best or rank > best[n]["_rank"]:
                    best[n] = {"_rank": rank, "set_id": r.get("set_id"), "effective_time": r.get("effective_time"),
                               "route": of.get("route", []),
                               "brand": (of.get("brand_name") or [""])[0],
                               **{k: " ".join(r[v])[:4000] for k, v in secs.items() if r.get(v)}}
    for v in best.values():
        v.pop("_rank")
    return best


# ---------------------------------------------------------------- 模型
class LLM:
    def __init__(self, base: str, model: str, conc: int = 64) -> None:
        import httpx
        self.c = httpx.AsyncClient(base_url=base, timeout=600)
        self.model, self.sem, self.calls = model, asyncio.Semaphore(conc), 0

    async def __call__(self, prompt: str, max_tokens: int = 800) -> str:
        async with self.sem:
            r = await self.c.post("/chat/completions", json={
                "model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}})
            r.raise_for_status()
            self.calls += 1
            return r.json()["choices"][0]["message"]["content"].strip()

    async def json(self, prompt: str, max_tokens: int = 800) -> Any:
        t = await self(prompt, max_tokens)
        m = re.search(r"[\[{].*[\]}]", t, re.S)
        try:
            return json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            return None


async def inn_aliases(llm: LLM, names: list[str], by_en: dict[str, str]) -> dict[str, tuple[str, str]]:
    """中文通用名 → (QID, INN)。正向译 INN 命中 Wikidata 英文名，且反向译回的中文与原名一致才收。"""
    out: dict[str, tuple[str, str]] = {}
    batches = [names[i:i + 40] for i in range(0, len(names), 40)]

    async def fwd(b):
        r = await llm.json("下面是中文药品通用名（每行一个）。给出每个药对应的英文国际非专利药名（INN，小写）；中成药、复方、"
                           "不是药品或你不确定的写 unknown。只输出一个 JSON 对象，键是中文名，值是 INN。\n" + "\n".join(b), 2000)
        return r if isinstance(r, dict) else {}
    cand = {}
    for r in await asyncio.gather(*(fwd(b) for b in batches)):
        for zh, inn in r.items():
            inn = str(inn).strip().lower()
            if zh in names and inn in by_en:
                cand[zh] = inn
    items = list(cand.items())
    rb = [items[i:i + 40] for i in range(0, len(items), 40)]

    async def back(b):
        r = await llm.json("下面是英文药品 INN（每行一个）。给出每个药在中国大陆通用的中文通用名（不带剂型、不带盐）。"
                           "只输出一个 JSON 对象，键是 INN，值是中文名。\n" + "\n".join(inn for _, inn in b), 2000)
        return b, (r if isinstance(r, dict) else {})
    for b, r in await asyncio.gather(*(back(b) for b in rb)):
        for zh, inn in b:
            if str(r.get(inn, "")).strip() == zh:
                out[zh] = (by_en[inn], inn)
    return out


_SENT = re.compile(r"(?<=[。；;！!])|(?=\s*(?:\d{1,2}[)）、.．]|[（(]\d{1,2}[)）]|[①-⑳]))")


CUT_NOTE = "（节选，完整内容见说明书）"


def cut_at_sentence(text: str, n: int) -> str:
    """超长时在 n 字以内最后一个句号 / 分号处截断，不切半句（“当开始用吡格列酮治疗时，胰岛……”会丢掉关键的减量说明）。"""
    if len(text) <= n:
        return text
    head = text[:n]
    i = max(head.rfind(c) for c in "。；;！!")
    return (head[: i + 1] if i >= n // 3 else head[: max(head.rfind("，"), n // 3)] + "。") + CUT_NOTE


async def condense(llm: LLM, field: str, text: str, limit: int = 320) -> tuple[str, str]:
    """过长的说明书原文：让模型挑句子编号，按原顺序拼回原句（抽取式，不改写）。"""
    if len(text) <= limit:
        return text, "verbatim"
    sents = [s.strip() for s in _SENT.split(text) if s.strip()]
    if len(sents) < 2:
        return cut_at_sentence(text, limit), "verbatim_truncated"
    label = {"dosage": "成人常用用法用量", "side_effects": "最常见和最严重的不良反应", "contraindications": "禁用人群和禁用情形",
             "indications": "主要适应症", "interactions_text": "最重要的药物相互作用"}[field]
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sents))
    r = await llm.json(f"下面是药品说明书某一段，已按句编号。请挑出最能概括“{label}”的句子编号，总字数不超过 {limit} 字，"
                       f"不要改写。只输出 JSON 数组，例如 [0, 2, 5]。\n{numbered}", 200)
    idx = sorted({i for i in (r or []) if isinstance(i, int) and 0 <= i < len(sents)})
    if not idx:
        return cut_at_sentence(text, limit), "verbatim_truncated"
    picked = "".join(sents[i] for i in idx)
    return cut_at_sentence(picked, int(limit * 1.3)), "extract"


_NUM = re.compile(r"\d+(?:\.\d+)?")
REVIEW: list[dict] = []          # 译写的每一段（英文原文 + 译文 + 体检结论），写到 --review-out 供人工抽查


async def translate(llm: LLM, field: str, drug: str, en: str) -> tuple[str | None, str]:
    """openFDA 英文段 → 中文要点；两道体检不过就返回 None。"""
    label = {"dosage": "成人常用用法用量（主要适应症）", "side_effects": "常见及严重不良反应", "contraindications": "禁忌（禁用情形）",
             "indications": "适应症", "interactions_text": "重要药物相互作用", "boxed_warning": "黑框警告"}[field]
    zh = await llm(f"下面是美国 FDA 药品说明书中「{drug}」的一段英文原文。请只根据这段原文，用中文写出“{label}”的要点，"
                   "不超过 200 字。要求：不加入原文没有的任何信息；剂量、频次、单位、百分比照原文数字写；"
                   "原文没有相关内容就只输出“无”。\n原文：\n" + en, 500)
    zh = zh.strip().strip("`")
    if not zh or zh in ("无", "无。") or len(zh) < 4:
        return None, "empty"
    src_nums = set(_NUM.findall(en))
    bad = [n for n in _NUM.findall(zh) if n not in src_nums]
    if bad:
        REVIEW.append({"drug": drug, "field": field, "en": en, "zh": zh, "verdict": f"lint_number:{bad}"})
        return None, "lint_number"
    v = await llm.json("核对一段中文要点是否忠实于英文原文。要点是概括，省略原文内容是允许的，不算不忠实；"
                       "只有当要点里出现了原文没有的信息（药名、剂量、频次、人群、不良反应、结论等），或把原文意思说反、说错，才判为不忠实。"
                       f"只输出 JSON：{{\"faithful\": true/false, \"problem\": \"…\"}}\n原文：\n{en}\n译文：\n{zh}", 200)
    if not isinstance(v, dict) or v.get("faithful") is not True:
        REVIEW.append({"drug": drug, "field": field, "en": en, "zh": zh, "verdict": f"lint_faithful:{(v or {}).get('problem')}"})
        return None, "lint_faithful"
    REVIEW.append({"drug": drug, "field": field, "en": en, "zh": zh, "verdict": "ok"})
    return zh, "translated"


# ---------------------------------------------------------------- 编译
_FRAGMENT_END = ("酸", "酯", "钠", "钾", "钙", "镁", "锌", "铁", "亚铁", "氢", "盐", "碱")


def ok_alias(a: str) -> bool:
    """剥盐得到的别名能不能用：“硝酸甘油”剥成“甘油”、“葡萄糖酸钙”剥成“葡萄糖酸”、“硝酸异山梨酯”剥成“异山梨酯”都会串药。"""
    return len(a) >= 2 and not a.endswith(_FRAGMENT_END) and a not in NOT_A_DRUG_NAME


_CJK = re.compile(r"^[\u3400-\u9fff·]{2,12}$")
_BAD_LABEL = re.compile(r"[A-Za-z]{3,}|[,'()（）]|\d-|-\d")


def display_name(zh: str, aliases: list[str]) -> str:
    """Wikidata 中文标签偶尔是错的（格鲁米特被标成“Vologases六世”）或是化学名（“4-甲基吡唑”）；这时用第一个纯中文别名。
    “维生素C”“两性霉素B”“地尔硫䓬”这类正常名字不动。"""
    if not _BAD_LABEL.search(zh):
        return zh
    return next((a for a in aliases if _CJK.match(a)), zh)


def drop_alias_clashes(entries: dict[str, dict]) -> list[str]:
    """同一个新别名指向两个以上的药 → 全部删掉（宁可认不出，不可认错）。"""
    owners: dict[str, set] = defaultdict(set)
    for q, e in entries.items():
        for a in e.get("new_aliases", {}):
            owners[a].add(q)
    names = {n for e in entries.values() for n in [e["name"], *e.get("aliases", [])]}
    bad = [a for a, qs in owners.items() if len(qs) > 1 or a in names]       # 与别的药同名的也删
    bad += [a for e in entries.values() for a, src in e.get("new_aliases", {}).items() if src == "huatuo_name" and not ok_alias(a)]
    for e in entries.values():
        for a in bad:
            e.get("new_aliases", {}).pop(a, None)
    return bad


def pick_product(prods: list[str], huatuo: dict, prefer: str | None = None) -> list[str]:
    """同一个药有多个商品时的取用顺序：先口服剂型（注射剂 / 栓剂 / 外用的不良反应与禁忌和口服不同，混用会误导），
    再优先和已选用法用量同一个商品（prefer），再按字段最全、名字短。"""
    core = ("contraindications", "dosage", "side_effects")
    return sorted(prods, key=lambda p: (not p.endswith(ORAL), p != prefer, -sum(f in huatuo[p] for f in core), len(p), p))


async def check_field(llm: LLM, drug: str, field: str, text: str) -> tuple[bool, str]:
    """两道检查：正则筛掉明显缺数字 / 乱码；模型判断是否答非所问、有没有单位或数字损坏。"""
    if looks_broken(text):
        return False, "regex_broken"
    if not llm:
        return True, "unchecked"
    label = {"dosage": "用法用量", "side_effects": "不良反应", "contraindications": "禁忌", "indications": "适应症",
             "interactions_text": "药物相互作用"}[field]
    v = await llm.json(f"下面是一段据称是「{drug}」药品说明书里“{label}”部分的文字（网上抓取，可能有损坏）。请判断：\n"
                       f"1) on_topic：它是否确实在讲{label}（而不是药理作用、成分、性状、别的药、疾病介绍）；\n"
                       "2) corrupted：有没有明显的抓取损坏——数字缺失、单位错误或缺失（例如注射剂量“100～200g”实为微克）、"
                       "乱码、句子被截断到无法理解。\n"
                       '只输出 JSON：{"on_topic": true/false, "corrupted": true/false, "why": "不超过20字"}\n文字：\n' + text[:1500], 120)
    if not isinstance(v, dict):
        return False, "llm_unparsable"
    if v.get("on_topic") is not True:
        return False, f"off_topic:{v.get('why', '')}"
    if v.get("corrupted") is not False:
        return False, f"corrupted:{v.get('why', '')}"
    return True, "checked"


async def compile_kb(args) -> None:
    wd = json.loads((HERE / "wikidata_kb.json").read_text(encoding="utf-8"))
    kb = json.loads((HERE / "kb.json").read_text(encoding="utf-8"))
    drugs = wd["drugs"]
    names: dict[str, str] = {}
    for q, e in drugs.items():
        for n in [e["zh"], *e["aliases"]]:
            if len(n) >= 2 and n not in NOT_A_DRUG_NAME:
                names.setdefault(n, q)
    by_en = {e["en"].lower(): q for q, e in drugs.items() if e["en"]}

    huatuo = load_huatuo(args.huatuo)
    match: dict[str, list[str]] = defaultdict(list)           # QID → Huatuo 商品名
    alias_src: dict[str, dict[str, str]] = defaultdict(dict)  # QID → {新别名: 来源}
    unmatched: Counter = Counter()
    for p in huatuo:
        cands = strip_name(p)
        hit = next((c for c in cands if c in names), None)
        q = names[hit] if hit else None
        if q:
            match[q].append(p)
            # 去盐后的名字收为别名（“盐酸异丙嗪”→“异丙嗪”），但剥成化学碎片的不收（见 ok_alias）
            if cands[-1] not in names and ok_alias(cands[-1]):
                alias_src[q][cands[-1]] = "huatuo_name"
        elif cands:
            unmatched[cands[-1]] += 1

    llm = LLM(args.llm, args.model) if args.llm else None
    inn_hits: dict[str, tuple[str, str]] = {}
    if llm:
        inn_hits = await inn_aliases(llm, [n for n, _ in unmatched.most_common(args.inn_limit)], by_en)
        for p in huatuo:
            cands = strip_name(p)
            if cands and cands[-1] in inn_hits and not any(c in names for c in cands):
                q = inn_hits[cands[-1]][0]
                match[q].append(p)
                alias_src[q][cands[-1]] = f"llm_inn:{inn_hits[cands[-1]][1]}"

    pilot = set(json.loads(Path(args.pilot).read_text())) if args.pilot else set(drugs)
    for p in (json.loads(Path(args.pilot_names).read_text()) if args.pilot_names else []):
        # 试点里点名要覆盖的商品名（如评测题里的冷门药）：经名称对齐 / INN 别名后能对上 Wikidata 的就加进来
        cands = strip_name(p)
        q = next((names[c] for c in cands if c in names), None) or next((inn_hits[c][0] for c in cands if c in inn_hits), None)
        if q:
            pilot.add(q)
    wanted = {drugs[q]["en"].lower() for q in pilot if drugs[q]["en"]}
    fda = load_openfda(args.openfda_dir, wanted) if args.openfda_dir else {}

    curated_by_q = {}
    for cname, c in kb["drugs"].items():
        q = names.get(cname)
        if q:
            curated_by_q[q] = cname

    out: dict[str, dict] = {}
    stats: Counter = Counter()
    lint: list[dict] = []

    async def build_one(q: str) -> None:
        e = drugs[q]
        entry = {"name": display_name(e["zh"], e["aliases"]), "generic_name": e["en"], "atc": e["atc"], "aliases": e["aliases"],
                 "new_aliases": alias_src.get(q, {}), "kb_tier": "wikidata", "reviewed": False, "source": {}}
        entry["name"] = display_name(e["zh"], e["aliases"] + list(alias_src.get(q, {})))
        if entry["name"] != e["zh"]:
            entry["wikidata_label"] = e["zh"]
        if q in curated_by_q:
            c = kb["drugs"][curated_by_q[q]]
            entry.update({k: c[k] for k in ("indications", "contraindications", "dosage", "side_effects", "interactions") if k in c})
            if "pregnancy_category" in c:
                entry["pregnancy_category"] = c["pregnancy_category"]
            entry["kb_tier"] = "curated"
            entry["source"] = {k: "kb.json（精选库，原仓库手写，未经医学审核）" for k in ("indications", "contraindications", "dosage",
                                                                                     "side_effects", "interactions")}
            stats["curated"] += 1
        async def huatuo_field(prods: list[str], field: str, prefer: str | None) -> tuple[str, str, str] | None:
            """在给定商品里按顺序找一段通过体检的原文 → (文字, 来源, 商品)。"""
            for p in pick_product(prods, huatuo, prefer)[:3]:
                for txt in sorted(huatuo[p].get(field, []), key=len, reverse=True)[:2]:
                    ok, why = await check_field(llm, e["zh"], field, txt)
                    stats[f"check_{why.split(':')[0]}"] += 1
                    if not ok:
                        lint.append({"qid": q, "drug": e["zh"], "field": field, "product": p, "reason": why, "text": txt[:300]})
                        continue
                    txt, how = (await condense(llm, field, txt)) if llm else (txt, "verbatim")
                    return txt, f"Huatuo 百科问答「{p}」（{how}）", p
            return None

        # 1) 中文说明书原文，按剂型大类分别编译：每类取一个商品的 用法用量 / 禁忌 / 不良反应（尽量同一商品）
        cats: dict[str | None, list[str]] = defaultdict(list)
        for p in match.get(q, []):
            cats[form_of(p)].append(p)
        by_form: dict[str, dict] = {}
        for cat, prods in cats.items():
            rec: dict[str, Any] = {"source": {}}
            chosen = None
            for field in ("dosage", "contraindications", "side_effects"):
                r = await huatuo_field(prods, field, chosen)
                if r:
                    rec[field], rec["source"][field], prod = r
                    if field == "dosage":
                        chosen = rec["product"] = prod
            if rec["source"]:
                rec.setdefault("product", next(iter(rec["source"].values())).split("「")[1].split("」")[0])
                by_form[cat or "未注明剂型"] = rec
        main = by_form.get("口服") or by_form.get("未注明剂型") or \
            max(by_form.values(), key=lambda r: len(r["source"]), default=None)
        if len(by_form) > 1:
            entry["by_form"] = by_form
            stats["multi_form"] += 1
        chosen = main["product"] if main else None
        for field in FIELDS:
            if entry.get(field) or field in entry["source"]:
                continue
            got = None
            if main and field in main:
                p = main["product"] if field == "dosage" else main["source"][field].split("「")[1].split("」")[0]
                pre = f"（{p}）" if p != e["zh"] else ""
                got = (pre + main[field], main["source"][field])
            elif field in ("indications", "interactions_text") or not main:
                r = await huatuo_field(match.get(q, []), field, chosen)
                if r:
                    txt, src, p = r
                    pre = f"（{p}）" if p != e["zh"] and (field == "dosage" or form_of(p) not in (None, "口服")) else ""
                    got = (pre + txt, src)
            if not got and llm and fda.get(e["en"].lower(), {}).get(field):   # 2) openFDA 翻译
                lab = fda[e["en"].lower()]
                zh, how = await translate(llm, field, e["en"], lab[field])
                stats[f"fda_{how}"] += 1
                if zh:
                    route = "、".join(lab.get("route") or []) or "未注明途径"
                    note = f"（依据美国 FDA 说明书，给药途径 {route}，国内说明书可能不同）" if field == "dosage" else ""
                    got = (zh + note, f"openFDA 说明书 set_id={lab['set_id']} effective={lab['effective_time']}（模型译写，已过数字与忠实度核对）")
                else:
                    lint.append({"qid": q, "drug": e["zh"], "field": field, "reason": how})
            if not got and field in ("dosage", "contraindications", "side_effects"):   # 3) 只有其他剂型有：注明剂型
                other = next((r for c, r in by_form.items() if field in r and r is not main), None)
                if other:
                    p = other["source"][field].split("「")[1].split("」")[0]
                    got = (f"（{p}）" + other[field], other["source"][field])
            if not got and field == "indications" and e["treats"]:            # 4) Wikidata 关系
                got = ("用于：" + "、".join(e["treats"][:8]), f"Wikidata {q}（疾病—治疗药物关系）")
            if got:
                entry[field], entry["source"][field] = got
                stats[f"{field}:{got[1].split('「')[0].split(' ')[0]}"] += 1
        lab = fda.get(e["en"].lower())
        if lab and lab.get("boxed_warning") and llm and q in pilot:
            zh, how = await translate(llm, "boxed_warning", e["en"], lab["boxed_warning"])
            if zh:
                entry["boxed_warning"] = zh
                entry["source"]["boxed_warning"] = f"openFDA 说明书黑框警告 set_id={lab['set_id']}（模型译写，已核对）"
        filled = [f for f in ("contraindications", "dosage", "side_effects") if f in entry["source"]]
        stats[f"filled_core_{len(filled)}"] += 1
        if entry["kb_tier"] == "wikidata" and filled:
            entry["kb_tier"] = "label"
        out[q] = entry

    await asyncio.gather(*(build_one(q) for q in sorted(pilot)))
    drop_alias_clashes(out)
    out = {q: v for q, v in out.items() if v["kb_tier"] != "wikidata" or v["new_aliases"]}   # 只留补到内容或有新别名的
    cur = {}
    for cname, c in kb["drugs"].items():                      # 精选库里 Wikidata 没有的（“头孢”是类别）
        if cname not in curated_by_q.values():
            cur[f"curated:{cname}"] = {"name": c["name"], "generic_name": c["generic_name"], "atc": [], "aliases": [],
                                       "new_aliases": {}, "kb_tier": "curated", "reviewed": False,
                                       **{k: c[k] for k in ("indications", "contraindications", "dosage", "side_effects",
                                                            "interactions") if k in c},
                                       "source": {"all": "kb.json（精选库，原仓库手写，未经医学审核）"}}
    meta = {"built": date.today().isoformat(), "scope": "pilot" if args.pilot else "all", "n_pilot": len(pilot),
            "sources": {"wikidata": "wikidata_kb.json（CC0）",
                        "huatuo": "FreedomIntelligence/huatuo_encyclopedia_qa train 拆分（Apache-2.0）",
                        "openfda": "openFDA drug label bulk export（公有领域）",
                        "curated": "kb.json"},
            "compiler": args.model if llm else None, "llm_calls": llm.calls if llm else 0,
            "huatuo_products": len(huatuo), "huatuo_matched_drugs": len(match), "inn_aliases": len(inn_hits),
            "openfda_labels_for_pilot": len(fda), "stats": dict(stats), "lint_dropped": lint,
            "n_entries": len(out),
            "warning": "未经医生 / 药师审核；剂量以药品说明书或医嘱为准"}
    Path(args.out).write_text(json.dumps({"meta": meta, "drugs": {**out, **cur}}, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.review_out:
        Path(args.review_out).write_text(json.dumps({"translations": REVIEW,
                                                     "inn_aliases": {zh: list(v) for zh, v in inn_hits.items()},
                                                     "matched_products": {q: match[q] for q in sorted(pilot) if q in match}},
                                                    ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "lint_dropped"}, ensure_ascii=False, indent=1))


MODAL = ("禁用", "禁忌", "禁止", "慎用", "不推荐", "不宜", "避免", "忌用", "不建议", "不应", "停用", "勿用")
FREQ = ("十分常见", "非常常见", "较常见", "常见", "多见", "偶见", "偶有", "较少见", "少见", "罕见", "极少数", "极少", "很少", "个别")
_SOFT = ("慎用", "不推荐", "不宜", "不建议", "避免", "不应")
SUMMARY_SECTIONS = (("用途", "indications"), ("禁忌", "contraindications"), ("不良反应", "side_effects"))


def lexical_lint(section: str, text: str, src: str) -> str | None:
    """程度词 / 频率词 / “严重”必须照原文：原文没有就不能出现（“不推荐”写成“禁用”、“极少数”写成“偶见”是 2026-11-20 复核里最常见的错）。
    例外：禁忌段写“禁用 / 禁忌”，而原文是纯禁忌清单（没有慎用 / 不推荐这类软性说法）时允许。返回问题描述，没问题返回 None。"""
    for n in _NUM.findall(text):
        if n not in set(_NUM.findall(src)):
            return f"数字 {n} 原文没有"
    for w in MODAL:
        if w in text and w not in src:
            if section == "禁忌" and w in ("禁用", "禁忌") and not any(x in src for x in _SOFT):
                continue
            return f"程度词「{w}」原文没有"
    for w in FREQ:
        if w in text and w not in src and not any(w in x and x in src for x in FREQ):
            return f"频率词「{w}」原文没有"
    if "严重" in text and "严重" not in src:
        return "原文没说“严重”"
    return None


async def summarize(args) -> None:
    """Wiki 页的“要点”（2026-11-20 重写）：用途 / 禁忌 / 不良反应分段，每段只用对应字段、用原文词语、程度词和频率词照抄；
    逐段过数字 + 程度词 / 频率词的确定性检查和药师复核（不通过的带着问题重写一次，再不过就不写这段）。
    旧版整段概括的复核出错率约 21%（把“不推荐”写成“禁用”、频率改写、把黑框警告当成不良反应、跨字段混写）。
    进了隔离区的字段不参与概括。"""
    from .wiki_check import _h, apply_quarantine
    path = Path(args.out)
    kb = json.loads(path.read_text(encoding="utf-8"))
    view = json.loads(json.dumps(kb["drugs"]))
    apply_quarantine(view)
    llm = LLM(args.llm, args.model)
    stats: Counter = Counter()

    async def section(name: str, sec: str, src: str) -> str | None:
        feedback = ""
        for attempt in range(2):
            t = (await llm(f"下面是药品「{name}」说明书的「{sec}」段。写一句不超过 40 字的要点，供医生快速浏览。规则：\n"
                           "1. 只用这一段里的内容，用原文的词语，不加原文没有的信息；\n"
                           "2. 程度词照抄原文（禁用 / 慎用 / 不推荐 / 不宜 / 避免 等），不能把“慎用”“不推荐”写成“禁用”；\n"
                           "3. 频率词照抄原文（常见 / 偶见 / 罕见 / 极少数 等），原文没写频率就不写；原文没说“严重”就不写“严重”；\n"
                           "4. 不写剂量和任何数字；只输出这一句。" + feedback + f"\n原文：\n{src[:3000]}", 160)).strip().strip("。") + "。"
            bad = lexical_lint(sec, t, src) if len(t) <= 70 else "太长"
            if not bad:
                v = await llm.json(f"你是药师。下面一句要点概括自药品「{name}」说明书的「{sec}」段。概括省略原文内容是允许的；"
                                   "检查是否有实质性错误：原文没有的信息、意思说反或程度改变（如慎用 / 不推荐写成禁用）、频率改变、"
                                   "把别的字段（如黑框警告、用法）的内容当成本段内容。\n"
                                   f"原文：\n{src[:3000]}\n要点：{t}\n只输出 JSON：{{\"ok\": true/false, \"problem\": \"20字内\"}}", 100)
                if isinstance(v, dict) and v.get("ok") is True:
                    stats[f"ok_{sec}"] += 1
                    return t
                bad = (v or {}).get("problem", "复核未通过")
            stats[f"retry_{sec}" if attempt == 0 else f"drop_{sec}"] += 1
            feedback = f"\n（上一稿的问题：{bad}，请改正）"
        return None

    async def one(q: str, e: dict, v: dict) -> None:
        secs = []
        for sec, f in SUMMARY_SECTIONS:
            src = v.get(f) or ""
            if len(src) >= 8:
                t = await section(e["name"], sec, src)
                if t:
                    secs.append(f"{sec}：{t}")
        if secs:
            e["summary"] = "".join(secs)
            e.setdefault("source", {})["summary"] = (f"模型据本条目适应症 / 禁忌 / 不良反应分段概括（{args.model}，"
                                                     "已过数字、程度词与频率词、药师复核）")
            stats["drugs_with_summary"] += 1
        else:
            e.pop("summary", None)
            e.get("source", {}).pop("summary", None)
            stats["drugs_without_summary"] += 1

    await asyncio.gather(*(one(q, e, view[q]) for q, e in kb["drugs"].items() if e.get("kb_tier") in ("label", "curated")))
    kb["meta"]["summaries"] = {"version": 2, **dict(stats)}
    kb["meta"]["llm_calls"] = kb["meta"].get("llm_calls", 0) + llm.calls
    path.write_text(json.dumps(kb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(dict(stats), ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--huatuo")
    ap.add_argument("--openfda-dir")
    ap.add_argument("--llm", help="OpenAI 兼容服务地址；不给就只做确定性部分（不做 INN 对齐、不抽取、不翻译）")
    ap.add_argument("--model", default="Qwen3.5-27B-FP8")
    ap.add_argument("--pilot", help="试点 QID 列表（JSON）；不给就编译全部药品")
    ap.add_argument("--pilot-names", help="额外要覆盖的商品名列表（JSON），名称对齐之后再解析成 QID")
    ap.add_argument("--inn-limit", type=int, default=3000)
    ap.add_argument("--out", default=str(HERE / "drug_kb.json"))
    ap.add_argument("--review-out", help="译写段落、INN 别名、商品名对齐结果（人工抽查用，不进仓库）")
    ap.add_argument("--summarize", action="store_true", help="只给已编译的 --out 补 Wiki 要点（需要 --llm）")
    args = ap.parse_args()
    asyncio.run(summarize(args) if args.summarize else compile_kb(args))


if __name__ == "__main__":
    main()
