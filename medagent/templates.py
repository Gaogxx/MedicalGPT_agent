"""确定性兜底回答：当模型输出无法使用（工具调用泄漏 / 空 / 截断 / 两档模型都不可用）时，
直接用工具结果渲染一段保守的回答。宁可朴素，不可编造。"""
from __future__ import annotations

import re
from typing import Any

URGENCY = {"emergency": "可能为急症", "medium": "建议近期就医评估", "low": "可先观察", "unknown": "信息不足"}


def render_observation(tool: str, o: Any) -> str:
    if not isinstance(o, dict):
        return ""
    if tool == "check_symptom" or "possible_causes" in o:
        if not o.get("found", True) or not o.get("possible_causes"):
            return f"根据您描述的「{o.get('symptom', '')}」，目前信息不足以判断原因，建议线下就诊做进一步检查。"
        dur = f"（持续 {o['duration']}）" if o.get("duration") else ""
        return (f"您描述的「{o.get('symptom', '')}」{dur}可能与{'、'.join(o['possible_causes'])}有关"
                f"（{URGENCY.get(o.get('urgency', ''), '')}）。{o.get('recommendation', '')}。")
    if tool == "search_drug" or "indications" in o or o.get("is_category"):
        if o.get("is_category"):
            return o.get("suggestion", "")
        if not o.get("found", True):
            return f"知识库中没有查到「{o.get('query', '')}」的药品信息，请确认药名，或携带药盒咨询药师。"
        return (f"{o['name']}（{o.get('generic_name', '')}）：适应症为{o['indications']}；禁忌：{o['contraindications']}；"
                f"用法用量：{o['dosage']}；常见不良反应：{o['side_effects']}。具体用药请遵医嘱或说明书。")
    if tool == "check_interaction" or "interactions" in o and "drugs" in o:
        if not o.get("interactions"):
            return f"知识库未发现{'、'.join(o.get('drugs', []))}之间的已知相互作用，但不代表绝对安全，请咨询药师。"
        sev = {"label": "说明书", "major": "严重", "moderate": "中等", "minor": "轻微", "unknown": "程度未知"}
        lines = [f"{'+'.join(i['drugs'])}：{i['description']}（{sev.get(i['severity'], i['severity'])}）" for i in o["interactions"]]
        return "发现以下相互作用：" + "；".join(lines) + "。请勿自行合用，咨询医生或药师。"
    if tool == "search_guideline" or "content" in o and "source" in o:
        if not o.get("found", True):
            return f"暂未检索到「{o.get('topic', '')}」相关的指南条目。"
        drugs = (o.get("label_drugs") or {}).get("drugs", [])
        extra = ("说明书适应症写明了本病的药：" + "、".join(d["name"] for d in drugs[:8]) + "（选药须遵医嘱）。") if drugs else ""
        return f"{o['topic']}：{o['content']}（出处：{o['source']} {o.get('year', '')}）。{extra}"
    if tool == "search_knowledge" and o.get("evidence"):
        ev = o["evidence"][0]
        return f"相关资料：{ev['text'][:200]}（来源：{ev.get('source', '')}）"
    return ""


def render_fallback(observations: list[tuple[str, Any]]) -> str:
    parts = [render_observation(t, o) for t, o in observations]
    parts = list(dict.fromkeys(p for p in parts if p))   # 多个 worker 查到同一条目时只输出一次
    if not parts:
        return "抱歉，系统暂时无法给出可靠的回答。如有不适请及时就医；如为急症请立即拨打 120。"
    return "\n".join(parts)


# ---------------------------------------------------------------- 降级模式（两档 / 三档模型全部不可用）
# 这里没有任何生成模型参与：输出 = 知识库字段原文 + 固定话术，所以不会有“生成式幻觉”。
# 2026-11-20 起用医药 Wiki 库（精选库已并入；独立复核不通过、进了隔离区的字段不用）。风险与防护：
#   1. 知识库本身错误或未审核   → 只摘说明书原文 / 精选库字段，不用模型写的“要点”；每段注明来源，标注“未经审核”；
#                                  不用 Wikidata 的“可能病因”（关系噪声大）
#   2. 实体链接认错 / 有歧义     → 命中多个药名时全部列出、不替用户选；药名按 search_drug 同一口径（精确名 / 别名 / 最长匹配）
#   3. 缺少个体化判断（孕期、儿童、老人、过敏、合并用药）→ 出现这些情况时不输出任何剂量，改为“需医生个体化判断”
#   4. 剂型 → 问到的剂型没有用法时明说，不给别的剂型的剂量
#   另外：急症只给急救提示、高风险请求只给“需面诊”，都不附知识库内容。
DEGRADED_HEADER = "【降级模式】医疗模型暂时不可用，以下仅为知识库条目原文摘录，未经模型理解与个体化判断，不能作为用药依据："
_RISK_PROFILE = ("孕", "怀孕", "哺乳", "宝宝", "婴儿", "孩子", "儿童", "小孩", "岁的娃", "老人", "老年", "过敏", "肝", "肾", "在吃", "正在服用")
_MISSING = ("请查阅", "未记录", "不含")
_ASK_SIDE = re.compile(r"副作用|不良反应|反应")
_ASK_CONTRA = re.compile(r"禁忌|能不能吃|能吃吗|可以吃吗|能用吗|可以用吗|孕|哺乳|过敏")
_ASK_DOSE = re.compile(r"怎么吃|怎么用|用法|用量|剂量|吃几|吃多少|一次|一天|多久")


def _short(text: str, n: int = 160) -> str:
    from .knowledge.drug_label_kb import cut_at_sentence
    return cut_at_sentence(str(text), n)


def _drug_names(query: str) -> list[str]:
    """问题里的药名：精选库同义词 + Wiki 药品（Wikidata 名 / 编译对齐的别名），最长匹配，去重。"""
    from .knowledge import load_kb
    from .knowledge.ext import load_ext
    names = list(load_kb().link_entities(query)["drug"])
    ext = load_ext()
    if ext:
        for q in ext.link(query)["drug"]:
            n = (ext.labels.get(q) or {}).get("name") or ext.drugs[q]["zh"]
            if n in names or any(n in m for m in names):
                continue
            names = [m for m in names if m not in n] + [n]      # 更具体的药名（“头孢曲松”）替换类别词（“头孢”）
    return names


def kb_fallback_answer(query: str, assessment=None, history: str = "") -> str:
    from .agent import drug_name_with_form
    from .knowledge import load_kb
    from .knowledge.interactions import enabled
    from .knowledge.wiki import load_wiki
    from .safety import EMERGENCY_BANNER, HIGH_RISK_DEFERRAL
    if assessment is not None and (assessment.level == "emergency" or assessment.crisis):
        return EMERGENCY_BANNER
    if assessment is not None and assessment.high_risk:
        return HIGH_RISK_DEFERRAL
    kb = load_kb()
    ents = kb.link_entities(query)
    risky = any(w in f"{history} {query}" for w in _RISK_PROFILE)
    parts = []
    drugs = _drug_names(query)[:3]
    from .knowledge.wiki import DRUG_CHOICE_INTENT, INTERACT_INTENT
    inter = len(drugs) > 1 and bool(INTERACT_INTENT.search(query))
    for name in drugs:
        v = kb.search_drug(drug_name_with_form(name, query))
        if not v.get("found") or v.get("kb_tier") == "wikidata":
            continue
        src = v.get("field_sources", {})
        tag = lambda f: f"（{src[f]}）" if src.get(f) else ""                 # noqa: E731
        if risky:
            dose = "用法用量：您的情况可能涉及孕期/儿童/老人/过敏/肝肾功能/合并用药，剂量需医生个体化判断，此处不提供。"
        elif v.get("dosage_for_form") or (v.get("dosage_form") and v.get("dosage")):
            dose = f"{v['dosage_form']}用法：{_short(v.get('dosage_for_form') or v['dosage'])}"
        elif v.get("form_note") and "没有" in v["form_note"]:
            dose = v["form_note"].split("；", 1)[-1] if v["form_note"].startswith("dosage") else v["form_note"]
        else:
            dose = f"说明书用法：{_short(v['dosage'])}{tag('dosage')}"
        # 按问的是什么取字段：问能不能一起吃只给互查结果和禁忌，问副作用给不良反应，问怎么吃给用法；没看出来给适应症 / 禁忌 / 用法
        want = (["contraindications"] if inter else
                [f for f, pat in (("side_effects", _ASK_SIDE), ("contraindications", _ASK_CONTRA), ("dosage", _ASK_DOSE))
                 if pat.search(query)] or ["indications", "contraindications", "dosage"])
        line = []
        for f, t in (("indications", "适应症"), ("contraindications", "禁忌"), ("side_effects", "不良反应")):
            if f in want and v.get(f) and not any(m in str(v[f]) for m in _MISSING):
                line.append(f"{t} {_short(v[f])}{tag(f)}")
        if "dosage" in want:
            line.append(dose)
        if v.get("boxed_warning") and not inter:
            line.append(f"黑框警告：{_short(v['boxed_warning'], 120)}")
        if not line:
            if inter:
                continue
            line.append("知识库没有这一项的说明书内容，请查阅药品说明书或咨询药师")
        parts.append(f"· {v['name']}" + (f"（{v['generic_name']}）" if v.get("generic_name") else "") + "：" + "；".join(line))
    if len(drugs) > 1:
        r = kb.check_interaction(drugs)
        for it in r["interactions"][:3]:
            parts.append(f"· {'+'.join(it['drugs'])}：{_short(it['description'], 140)}")
        if not r["interactions"]:
            parts.append(f"· {r['advice']}")
    for s in ents["symptom"][:2]:
        v = kb.symptoms[s]
        parts.append(f"· {s}：常见相关原因包括{'、'.join(v['possible_causes'])}（并非诊断）。{v['recommendation']}。")
    if ents["disease"]:
        v = kb.guidelines[ents["disease"][0]]
        parts.append(f"· {v['topic']}：{v['content']}（出处：{v['source']} {v['year']}）")
    if not drugs and enabled() and (DRUG_CHOICE_INTENT.search(query) or not ents["disease"]):
        w = load_wiki()
        dd = w.disease_drugs(query, k=8) if w else None
        if dd:
            parts.append(f"· {dd['disease']}：说明书适应症写明了本病的药有 {'、'.join(d['name'] for d in dd['drugs'])}"
                         "（不是治疗推荐或排序，选药须由医生根据病情决定）")
    if not parts:
        return "抱歉，医疗模型暂时不可用，且知识库中没有与您问题直接对应的条目，我无法给出可靠回答。如有不适请及时就医；如为急症请立即拨打 120。"
    if len(drugs) > 1:
        parts.append("· 涉及多种药物时请勿自行合用，请咨询医生或药师。")
    return DEGRADED_HEADER + "\n" + "\n".join(parts) + "\n（知识库条目摘自药品说明书与精选库，未经医生审核，请以说明书与医嘱为准。）"
