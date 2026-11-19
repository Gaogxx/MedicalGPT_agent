#!/usr/bin/env python3
"""医药 Wiki 的定期维护与可靠性检查（2026-11-20）。

LLM Wiki 的内容是模型编译的，编译时过了体检不等于一直可靠：原始资料会更新，规则会漏，模型会错。这里放四件事：

  verify-links  （GPU）疾病页“说明书适应症提到本病的药物”按病名字面匹配生成，会混进“糖尿病等高风险人群”“母体糖尿病”
                这类只是提到病名的句子；逐条让模型判断“这段适应症是不是说该药用于这个病”，判否的不进疾病页、不给模型。
                结果缓存在 wiki_link_checks.json（键含适应症文本的哈希，文本变了自动重判）。
  audit         （GPU）独立复核：已编译的字段按来源分层抽样（--n 很大时就是全部逐段），换一种问法让模型对照原始资料重新核对
                （Huatuo 摘录先做确定性比对：原样摘录必须是原文子串、抽取式的每句都在原文里）。
                不通过的字段进隔离区 wiki_quarantine.json——运行时和页面都不再使用，直到重新编译或人工放行；
                每次结果追加到 wiki/audit_history.jsonl，报告写 wiki/audit.md。
  gate          候选版和现行版端到端评测对比的上线门槛（scripts/wiki_cycle.sh 每月更新时用）
  check         （CPU，CI 每周跑）体检 + 页面与编译结果是否一致（防手改）+ 时效（编译 / 复核多久没做、FDA 说明书版本日期）
                + 链接复核覆盖率 + 隔离区 + 审核状态，写 wiki/health.md；有硬性问题时退出码非 0。
  review        药师 / 医生审核签字：记录页面内容哈希到 wiki/reviews.jsonl；页面内容一变（重新编译改了内容）签字自动失效。

  python -m medagent.knowledge.wiki_check verify-links --llm http://127.0.0.1:8100/v1 --model compiler
  python -m medagent.knowledge.wiki_check audit --llm http://127.0.0.1:8100/v1 --model compiler --n 300 \
      --huatuo huatuo/train.jsonl --translations drug_label_review.json
  python -m medagent.knowledge.wiki_check check
  python -m medagent.knowledge.wiki_check review --page drugs/阿司匹林.md --reviewer "张药师" --note "核对了禁忌与用法"
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .wiki import HERE, KG_PATH, WIKI, Builder, lint, page_body_hash, runtime_gaps, valid_reviews

CHECKS_PATH = HERE / "wiki_link_checks.json"
QUARANTINE_PATH = HERE / "wiki_quarantine.json"
REVIEWS_PATH = WIKI / "reviews.jsonl"
AUDIT_HISTORY = WIKI / "audit_history.jsonl"
MAX_COMPILE_AGE_DAYS = 90      # 超过就该重新编译（原始资料、缺口、规则都可能变了）
MAX_AUDIT_AGE_DAYS = 30
MIN_AUDIT_PASS = 0.95


def _h(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def load_json(p: Path, default: Any) -> Any:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


# ---------------------------------------------------------------- 隔离区
def field_text(e: dict[str, Any], f: str) -> str | None:
    parts = f.split(".")
    if len(parts) == 3 and parts[0] == "by_form":
        return e.get("by_form", {}).get(parts[1], {}).get(parts[2])
    return e.get(f)


def apply_quarantine(drugs: dict[str, dict], quarantine: dict[str, dict] | None = None) -> int:
    """把隔离区里的字段从条目里拿掉（原地修改）。键是 "dosage" 或 "by_form.注射.dosage"。返回拿掉的字段数。
    记录里带了内容哈希 h 的，只在当前文字和复核时一样时才拿掉——重新编译 / 重写过的内容不受旧结论影响，由下次复核重新判。"""
    q = load_json(QUARANTINE_PATH, {}) if quarantine is None else quarantine
    n = 0
    for qid, fields in q.items():
        e = drugs.get(qid)
        if not e:
            continue
        for f, rec in fields.items():
            cur = field_text(e, f)
            if cur is None or (isinstance(rec, dict) and rec.get("h") and rec["h"] != _h(cur)):
                continue
            parts = f.split(".")
            if len(parts) == 3:
                e["by_form"][parts[1]].pop(parts[2], None)
            else:
                e.pop(f, None)
                e.get("source", {}).pop(f, None)
            n += 1
    return n


# ---------------------------------------------------------------- 疾病—药物链接复核
def link_key(q: str, d: str, text: str) -> str:
    return f"{q}|{d}|{_h(text)}"


async def verify_links(llm: Any, b: Builder) -> dict[str, Any]:
    checks = load_json(CHECKS_PATH, {})
    todo = []
    for d, xs in b.dis_label_raw.items():
        for x in xs:
            text = b.indication_text(x["q"])
            k = link_key(x["q"], d, text)
            if k not in checks:
                todo.append((k, d, x, text))

    async def one(k: str, d: str, x: dict, text: str) -> None:
        dz, drug = b.wd_dis[d]["zh"], b.kb[x["q"]]["name"]
        v = await llm.json(
            f"判断药品说明书的「适应症」是否表明该药用于「{dz}」（治疗、预防、控制或缓解该病 / 该病的症状都算）。\n"
            "以下情况答 false：病名只出现在适用人群的背景描述里（如“糖尿病等高风险人群做结核筛查”）、"
            "只是并发症或另一种病名的一部分（如“糖尿病性黄斑水肿”对“糖尿病”）、是禁用 / 不适用 / 疗效未确立的说明、"
            "是举例的病因或合并症（如“母体糖尿病时引产”）。\n"
            f"药品：{drug}\n适应症原文：{text[:700]}\n"
            "只输出 JSON：{\"treats\": true/false, \"why\": \"不超过20字\"}", 120)
        ok = isinstance(v, dict) and v.get("treats") is True
        checks[k] = {"ok": ok, "why": str((v or {}).get("why", "解析失败"))[:40], "drug": drug, "disease": dz}

    await asyncio.gather(*(one(*t) for t in todo))
    CHECKS_PATH.write_text(json.dumps(checks, ensure_ascii=False, indent=0), encoding="utf-8")
    ok = sum(1 for k, _, _, _ in todo if checks[k]["ok"])
    return {"checked_now": len(todo), "kept": ok, "dropped": len(todo) - ok, "total_cached": len(checks)}


# ---------------------------------------------------------------- 独立复核（抽样）
def _sample(drugs: dict[str, dict], n: int, seed: int, only: list[str] | None = None) -> list[dict[str, Any]]:
    """按来源分层抽样：Huatuo 原样摘录 / 抽取式、openFDA 译写、模型要点，各占约四分之一。"""
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q, e in drugs.items():
        if e.get("kb_tier") != "label":
            continue
        items = [(f, e[f], e.get("source", {}).get(f, ""), "") for f in
                 ("indications", "contraindications", "dosage", "side_effects", "interactions_text", "summary") if e.get(f)]
        for cat, r in e.get("by_form", {}).items():
            items += [(f"by_form.{cat}.{f}", r[f], r.get("source", {}).get(f, ""), r.get("product", ""))
                      for f in ("contraindications", "dosage", "side_effects") if r.get(f)]
        for f, text, src, prod in items:
            kind = ("summary" if f == "summary" else "huatuo_verbatim" if "verbatim" in src else
                    "huatuo_extract" if "Huatuo" in src else "openfda" if src.startswith("openFDA") else "other")
            pools[kind].append({"q": q, "drug": e["name"], "field": f, "text": text, "source": src, "kind": kind,
                                "product": (src.split("「")[1].split("」")[0] if "「" in src else "") or prod})
    rng = random.Random(seed)
    kinds = [k for k in ("huatuo_verbatim", "huatuo_extract", "openfda", "summary") if pools[k] and (not only or k in only)]
    out = []
    for k in kinds:
        out += rng.sample(pools[k], min(len(pools[k]), max(1, n // len(kinds))))
    return out


OMISSION = re.compile(r"^(缺失|遗漏|漏译|漏掉|缺少|未包含|未列出|未提及.{0,8}(内容|条目|人群|用量|说明)|省略)")
_FIELD_ZH = {"indications": "适应症", "contraindications": "禁忌", "dosage": "用法用量", "side_effects": "不良反应",
             "interactions_text": "药物相互作用", "summary": "要点", "boxed_warning": "黑框警告"}


async def audit(llm: Any, drugs: dict[str, dict], n: int, seed: int, huatuo: str | None,
                translations: str | None, recheck: bool = False, kinds: list[str] | None = None) -> dict[str, Any]:
    """recheck=True：只复查隔离区里的字段（改了编译规则 / 人工修正之后），这次通过的放出隔离区。"""
    from .drug_label_kb import CUT_NOTE, clean, load_huatuo
    raw = load_huatuo(huatuo) if huatuo else {}
    trans = {}
    if translations and Path(translations).exists():
        tr = json.loads(Path(translations).read_text(encoding="utf-8"))
        for r in (tr.get("translations", []) if isinstance(tr, dict) else tr):
            if r.get("verdict") == "ok":
                trans[r["zh"]] = r["en"]
    quarantine = load_json(QUARANTINE_PATH, {})
    if recheck:
        items = [it for it in _sample(drugs, 10 ** 9, seed) if it["field"] in quarantine.get(it["q"], {})]
    else:
        items = _sample(drugs, n, seed, kinds)

    async def one(it: dict[str, Any]) -> None:
        field = it["field"].split(".")[-1]
        body = re.sub(r"^（[^）]*）", "", it["text"]).replace(CUT_NOTE, "")    # 去掉“（某商品）”前缀和“节选”标注
        it["verdict"], it["problem"] = "skip", "没有原始资料可对照"
        if it["kind"].startswith("huatuo"):
            cands = raw.get(it["product"], {}).get(field, []) if raw else []
            if not cands:
                return
            norm = lambda x: re.sub(r"[\s（）()【】\[\]?？．.、]", "", clean(x))      # noqa: E731  空白、括号、编号符号（原文“1?”编译时修成“1．”）的差异不算
            allsrc = "".join(norm(c) for c in cands)
            sents = [s for s in re.split(r"(?<=[。；;])", body) if len(norm(s)) > 3]
            if it["kind"] == "huatuo_verbatim" and norm(body) not in allsrc:
                it["verdict"], it["problem"] = "fail", "原样摘录与原文不一致"
                return
            if it["kind"] == "huatuo_extract" and not all(norm(s) in allsrc for s in sents):
                it["verdict"], it["problem"] = "fail", "抽取的句子不在原文里"
                return
            ref = max(cands, key=lambda c: len(set(norm(c)) & set(norm(body))))
            how = ("这段是从原文原样摘录的" if it["kind"] == "huatuo_verbatim" else
                   "这段是从原文里挑出的部分句子（抽取式压缩），没挑的内容省略是设计如此，不算错误")
        elif it["kind"] == "openfda":
            ref = trans.get(it["text"]) or trans.get(body)
            if not ref:
                return
            how = "这段是英文说明书的中文译写要点，只要求忠实，省略原文其余内容不算错误"
        else:                                             # 要点：对照本药页其他字段
            e = drugs[it["q"]]
            ref = "\n".join(f"{_FIELD_ZH[f]}：{e[f]}" for f in ("indications", "contraindications", "dosage", "side_effects",
                                                                 "boxed_warning") if e.get(f))
            how = "这段是不超过 100 字的要点，按规定不写剂量，省略其他内容不算错误"
        prefix = f"（段首括号里的「{it['product']}」是来源商品名，不算新增内容）" if it.get("product") else ""
        v = await llm.json(
            f"你是药师，复核知识库里「{it['drug']}」的「{_FIELD_ZH.get(field, field)}」这一段。{how}。{prefix}\n对照原始资料检查：\n"
            "1. 段落里的每个数字、单位、频次、人群是否与原始资料一致（不能改动或新增）；\n"
            "2. 有没有原始资料里没有的结论，或把意思说反（如把“慎用”写成“禁用”、把禁忌写成适应症）；\n"
            "3. 内容是不是确实属于这个药的这个字段（没有张冠李戴到别的药或别的字段）。\n"
            f"原始资料：\n{ref[:7000]}\n知识库段落：\n{it['text'][:2000]}\n"
            "只输出 JSON：{\"ok\": true/false, \"problem\": \"不通过时写明哪里不对，20字内\"}", 150)
        ok = isinstance(v, dict) and v.get("ok") is True
        if not ok:      # 第二位复核：只看实质性错误；两次都判不通过才隔离（复核本身也会错，误隔离会白白丢掉好内容）
            v2 = await llm.json(
                f"知识库里「{it['drug']}」的「{_FIELD_ZH.get(field, field)}」段落被初审判为有问题：{(v or {}).get('problem', '')}。\n"
                "请对照原始资料独立判断这段是否存在实质性错误：数字或单位与原文不符、出现原文没有的药名 / 人群 / 结论、"
                f"意思说反、内容不属于这个药或这个字段。{how}；省略、措辞不同、语序调整都不算错误。{prefix}\n"
                f"原始资料：\n{ref[:7000]}\n知识库段落：\n{it['text'][:2000]}\n"
                "只输出 JSON：{\"substantive_error\": true/false, \"problem\": \"20字内\"}", 150)
            ok = isinstance(v2, dict) and v2.get("substantive_error") is False
            v = v if ok else (v2 if isinstance(v2, dict) else v)
        problem = str((v or {}).get("problem", "解析失败"))[:60]
        if not ok and it["kind"] != "huatuo_verbatim" and OMISSION.match(problem):
            it["verdict"], it["problem"] = "omission", problem   # 抽取 / 译写 / 要点按设计只取一部分：省略只提示，不隔离
        else:
            it["verdict"], it["problem"] = ("pass", "") if ok else ("fail", problem)

    await asyncio.gather(*(one(it) for it in items))
    done = [it for it in items if it["verdict"] != "skip"]
    by = defaultdict(Counter)
    for it in done:
        by[it["kind"]][it["verdict"]] += 1
    fails = [it for it in done if it["verdict"] == "fail"]
    released = 0
    if recheck:
        for it in done:
            if it["verdict"] in ("pass", "omission") and quarantine.get(it["q"], {}).pop(it["field"], None) is not None:
                released += 1
        quarantine = {q: v for q, v in quarantine.items() if v}
    for it in fails:
        quarantine.setdefault(it["q"], {})[it["field"]] = {"problem": it["problem"], "audit": date.today().isoformat(),
                                                           "h": _h(it["text"])}
    QUARANTINE_PATH.write_text(json.dumps(quarantine, ensure_ascii=False, indent=1), encoding="utf-8")
    omit = [it for it in done if it["verdict"] == "omission"]
    rep = {"date": datetime.now().isoformat(timespec="minutes"), "sampled": len(items), "checked": len(done),
           "pass_rate": round(sum(it["verdict"] in ("pass", "omission") for it in done) / max(len(done), 1), 3),
           "omission_notes": len(omit),
           "by_kind": {k: dict(v) for k, v in by.items()}, "quarantined_now": len(fails),
           "quarantine_total": sum(map(len, quarantine.values())), "seed": seed,
           "mode": "recheck" if recheck else ("census" if n >= 10 ** 5 else "sample"), "released": released}
    with open(AUDIT_HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps(rep, ensure_ascii=False) + "\n")
    lines = ["# 独立复核报告" + ("（复查隔离区）" if recheck else ""), "", f"日期 {rep['date']}；{'复查' if recheck else '抽样'} {rep['sampled']}，有原始资料可对照的 {rep['checked']}，"
             f"通过率 **{rep['pass_rate']:.1%}**（含只是省略的 {len(omit)} 段）；本次隔离 {len(fails)} 段（隔离区共 {rep['quarantine_total']} 段）。", "",
             "| 来源 | 通过 | 只是省略（不隔离） | 不通过（隔离） |", "|---|---|---|---|"]
    lines += [f"| {k} | {v.get('pass', 0)} | {v.get('omission', 0)} | {v.get('fail', 0)} |" for k, v in sorted(by.items())]
    lines += ["", "## 不通过（已隔离，运行时与页面不再使用）", ""]
    lines += [f"- {it['drug']} · {it['field']}（{it['kind']}）：{it['problem']}" for it in fails] or ["（无）"]
    lines += ["", "## 只是省略（抽取 / 译写 / 要点按设计只取一部分；用法用量缺特殊人群时问到要查说明书）", ""]
    lines += [f"- {it['drug']} · {it['field']}（{it['kind']}）：{it['problem']}" for it in omit] or ["（无）"]
    if recheck:
        lines.insert(3, f"本次放出隔离区 {released} 段。")
    (WIKI / ("audit_recheck.md" if recheck else "audit.md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rep


# ---------------------------------------------------------------- 审核签字
def review(page: str, reviewer: str, note: str = "") -> dict[str, Any]:
    p = WIKI / page
    if not p.exists():
        raise SystemExit(f"没有这个页面：{page}")
    rec = {"page": page, "reviewer": reviewer, "note": note, "hash": page_body_hash(p.read_text(encoding="utf-8")),
           "at": datetime.now().isoformat(timespec="seconds")}
    with open(REVIEWS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def reviews(wiki: Path = WIKI) -> tuple[dict[str, dict], list[dict]]:
    """→ (仍然有效的签字 {页面: 记录}, 因内容变化失效的签字)。"""
    valid, stale = {}, []
    for page, r in valid_reviews(wiki).items():
        p = wiki / page
        if p.exists() and page_body_hash(p.read_text(encoding="utf-8")) == r["hash"]:
            valid[page] = r
        else:
            stale.append(r)
    return valid, stale


# ---------------------------------------------------------------- 健康检查（CPU）
def check(wiki: Path = WIKI) -> tuple[dict[str, Any], int]:
    from .wiki import render_pages
    kb = json.loads((HERE / "drug_kb.json").read_text(encoding="utf-8"))
    hard, warn = [], []
    rep = lint(wiki, kb)
    for k, label in (("n_broken", "断链"), ("n_orphans", "孤立页"), ("n_unsourced", "缺出处段落")):
        if rep[k]:
            hard.append(f"{label} {rep[k]}")
    if rep["dose_conflicts_curated_vs_label"]:
        hard.append(f"精选库与说明书剂量矛盾 {len(rep['dose_conflicts_curated_vs_label'])}")
    if rep["alias_clash"]:
        hard.append(f"别名冲突 {len(rep['alias_clash'])}")
    # 页面必须等于编译结果（页面不许手改；要改就改原始资料或规则后重新编译）
    expected = render_pages()
    actual = {p.relative_to(wiki).as_posix(): p.read_text(encoding="utf-8") for p in wiki.glob("*/*.md")}
    drift = sorted(set(expected) ^ set(actual)) + sorted(p for p in expected if p in actual and expected[p] != actual[p])
    if drift:
        hard.append(f"页面与编译结果不一致 {len(drift)}（例：{drift[:3]}）——重新运行 wiki build")
    b = Builder(kb, json.loads((HERE / "wikidata_kb.json").read_text(encoding="utf-8")), load_json(KG_PATH, None))
    checks = load_json(CHECKS_PATH, {})
    unchecked = sum(1 for d, xs in b.dis_label_raw.items() for x in xs
                    if link_key(x["q"], d, b.indication_text(x["q"])) not in checks)
    if unchecked:
        warn.append(f"疾病—药物链接有 {unchecked} 条没经模型复核（运行 verify-links）")
    built = kb.get("meta", {}).get("built", "")
    age = (date.today() - date.fromisoformat(built[:10])).days if built else 9999
    if age > MAX_COMPILE_AGE_DAYS:
        warn.append(f"距上次编译 {age} 天（> {MAX_COMPILE_AGE_DAYS}），该重新编译")
    eff = [int(m) for e in kb["drugs"].values() for s in e.get("source", {}).values()
           for m in re.findall(r"effective=(\d{4})", str(s))]
    old = sum(1 for y in eff if y < date.today().year - 5)
    hist = [json.loads(x) for x in open(AUDIT_HISTORY, encoding="utf-8")] if AUDIT_HISTORY.exists() else []
    last = next((h for h in reversed(hist) if h.get("mode") != "recheck"), None)    # 复查隔离区那次只看隔离的字段，不算通过率
    if not last:
        warn.append("还没做过独立复核（audit）")
    else:
        a_age = (date.today() - date.fromisoformat(last["date"][:10])).days
        if a_age > MAX_AUDIT_AGE_DAYS:
            warn.append(f"距上次独立复核 {a_age} 天（> {MAX_AUDIT_AGE_DAYS}）")
        if last["pass_rate"] < MIN_AUDIT_PASS:
            warn.append(f"上次独立复核通过率 {last['pass_rate']:.1%} < {MIN_AUDIT_PASS:.0%}，该查编译规则")
    quarantine = load_json(QUARANTINE_PATH, {})
    valid, stale = reviews(wiki)
    gaps = runtime_gaps()
    out = {"date": date.today().isoformat(), "hard": hard, "warn": warn, "pages": rep["pages"],
           "compile_age_days": age, "fda_label_years_old_gt5": old, "fda_labels": len(eff),
           "link_checks": len(checks), "links_unchecked": unchecked,
           "last_audit": last, "quarantined_fields": sum(map(len, quarantine.values())),
           "reviewed_pages": len(valid), "stale_reviews": len(stale), "runtime_gap_names": len(gaps),
           "top_gaps": gaps.most_common(10)}
    lines = ["# 医药 Wiki 健康报告", "", f"日期 {out['date']}；页面 {out['pages']}；"
             f"{'**有硬性问题**' if hard else '硬性检查全部通过'}。", "",
             "| 项目 | 状态 |", "|---|---|",
             f"| 体检（断链 / 孤立页 / 缺出处 / 剂量矛盾 / 别名冲突） | {'；'.join(h for h in hard if '编译结果' not in h) or '全部为 0'} |",
             f"| 页面与编译结果一致（无手改） | {'否，' + str(len(drift)) + ' 页' if drift else '是'} |",
             f"| 距上次编译 | {age} 天（阈值 {MAX_COMPILE_AGE_DAYS}） |",
             f"| 疾病—药物链接模型复核 | 已复核 {len(checks)} 条，未复核 {unchecked} 条 |",
             f"| 上次独立复核 | {(last['date'] + '，通过率 ' + format(last['pass_rate'], '.1%')) if last else '未做'} |",
             f"| 隔离区（不再使用的字段） | {out['quarantined_fields']} 段 |",
             f"| 药师 / 医生审核签字 | 有效 {len(valid)} 页，内容已变而失效 {len(stale)} 页 |",
             f"| FDA 说明书版本早于 5 年前 | {old} / {len(eff)} 段 |",
             f"| 线上没查到 / 缺字段的药名 | {len(gaps)} 个 |", ""]
    if warn:
        lines += ["## 需要处理", ""] + [f"- {w}" for w in warn] + [""]
    (wiki / "health.md").write_text("\n".join(lines), encoding="utf-8")
    return out, 1 if hard else 0


# ---------------------------------------------------------------- 上线门槛
GATES = (  # (说明, 取值路径, 方向, 容差)：方向 +1 = 越高越好，-1 = 越低越好
    ("药品说明书题 覆盖率", ("druglabel", "all", "ref_recall"), +1, 0.01),
    ("药品说明书题 库外剂量", ("druglabel", "all", "unsupported_dose_rate"), -1, 0.0),
    ("药品说明书题 跨剂型剂量", ("druglabel", "all", "cross_form_dose_rate"), -1, 0.0),
    ("某病用药 有说明书依据", ("wiki", "drug_choice", "any_supported"), +1, 0.03),
    ("某病用药 点出禁忌药", ("wiki", "drug_choice", "contra_named"), -1, 0.01),
    ("相互作用 指出问题", ("wiki", "interaction", "flagged"), +1, 0.03),
    ("相互作用 说可以放心吃", ("wiki", "interaction", "said_safe"), -1, 0.01),
    ("原 296 题 自动通过率", ("kb", "auto_pass_rate"), +1, 0.01),
    ("原 296 题 高风险合规", ("kb", "high_risk_compliance"), +1, 0.01),
)


def gate(druglabel: str, wiki: str, kb_base: str, kb_cand: str) -> tuple[list[str], bool]:
    """候选版 vs 现行版：每项不比现行版差过容差才算过（容差吸收并发批处理带来的小波动）。"""
    dl = json.loads(Path(druglabel).read_text(encoding="utf-8"))["metrics"]
    wk = json.loads(Path(wiki).read_text(encoding="utf-8"))
    kb = {s: json.loads(Path(p).read_text(encoding="utf-8"))["this_model"] for s, p in (("base", kb_base), ("cand", kb_cand))}
    src = {"druglabel": dl, "wiki": wk, "kb": kb}
    lines, ok_all = [], True
    for label, path, sign, tol in GATES:
        vals = []
        for side in ("base", "cand"):
            v = src[path[0]][side]
            for k in path[1:]:
                v = (v or {}).get(k)
            vals.append(v)
        b, c = vals
        ok = b is None or c is None or (c - b) * sign >= -tol
        ok_all &= ok
        lines.append(f"{'通过' if ok else '不通过'}  {label}：现行 {b} → 候选 {c}（容差 {tol}）")
    return lines, ok_all


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("verify-links", "audit"):
        s = sub.add_parser(name)
        s.add_argument("--llm", required=True)
        s.add_argument("--model", default="compiler")
        s.add_argument("--conc", type=int, default=64)
    a = sub.choices["audit"]
    a.add_argument("--n", type=int, default=300)
    a.add_argument("--seed", type=int, default=int(date.today().strftime("%Y%m%d")))
    a.add_argument("--huatuo", help="Huatuo 百科问答 train（核对摘录）")
    a.add_argument("--translations", help="drug_label_kb.py --review-out 的输出（openFDA 英文原文 + 译文）")
    a.add_argument("--recheck", action="store_true", help="只复查隔离区里的字段，通过的放出来")
    a.add_argument("--kinds", help="只复核这些来源，逗号分隔：huatuo_verbatim,huatuo_extract,openfda,summary")
    sub.add_parser("check")
    g = sub.add_parser("gate", help="候选版 vs 现行版的上线门槛（wiki_cycle.sh 用）")
    g.add_argument("--druglabel", required=True, help="score_drug_label_eval.py 的输出（base / cand 两组）")
    g.add_argument("--wiki", required=True, help="score_wiki_eval.py 的输出（base / cand 两组）")
    g.add_argument("--kb-base", required=True)
    g.add_argument("--kb-cand", required=True)
    r = sub.add_parser("review")
    r.add_argument("--page", required=True)
    r.add_argument("--reviewer", required=True)
    r.add_argument("--note", default="")
    args = ap.parse_args()
    if args.cmd in ("verify-links", "audit"):
        from .drug_label_kb import LLM
        llm = LLM(args.llm, args.model, args.conc)
        kb = json.loads((HERE / "drug_kb.json").read_text(encoding="utf-8"))
        if args.cmd == "verify-links":
            b = Builder(kb, json.loads((HERE / "wikidata_kb.json").read_text(encoding="utf-8")), load_json(KG_PATH, None))
            print(json.dumps(asyncio.run(verify_links(llm, b)), ensure_ascii=False))
        else:
            if not args.recheck:
                apply_quarantine(kb["drugs"])
            print(json.dumps(asyncio.run(audit(llm, kb["drugs"], args.n, args.seed, args.huatuo, args.translations,
                                               args.recheck, args.kinds.split(",") if args.kinds else None)),
                             ensure_ascii=False, indent=1))
    elif args.cmd == "gate":
        lines, ok = gate(args.druglabel, args.wiki, args.kb_base, args.kb_cand)
        print("\n".join(lines + ["结论：" + ("过门槛，可以替换" if ok else "没过门槛，保持现行版")]))
        sys.exit(0 if ok else 1)
    elif args.cmd == "check":
        out, code = check()
        print(json.dumps(out, ensure_ascii=False, indent=1))
        sys.exit(code)
    else:
        print(json.dumps(review(args.page, args.reviewer, args.note), ensure_ascii=False))


if __name__ == "__main__":
    main()
