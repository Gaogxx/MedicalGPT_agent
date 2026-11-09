"""确定性兜底回答：当模型输出无法使用（工具调用泄漏 / 空 / 截断 / 两档模型都不可用）时，
直接用工具结果渲染一段保守的回答。宁可朴素，不可编造。"""
from __future__ import annotations

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
        lines = [f"{'+'.join(i['drugs'])}：{i['description']}（{i['severity']}）" for i in o["interactions"]]
        return "发现以下相互作用：" + "；".join(lines) + "。请勿自行合用，咨询医生或药师。"
    if tool == "search_guideline" or "content" in o and "source" in o:
        if not o.get("found", True):
            return f"暂未检索到「{o.get('topic', '')}」相关的指南条目。"
        return f"{o['topic']}：{o['content']}（出处：{o['source']} {o.get('year', '')}）。"
    if tool == "search_knowledge" and o.get("evidence"):
        ev = o["evidence"][0]
        return f"相关资料：{ev['text'][:200]}（来源：{ev.get('source', '')}）"
    return ""


def render_fallback(observations: list[tuple[str, Any]]) -> str:
    parts = [render_observation(t, o) for t, o in observations]
    parts = [p for p in parts if p]
    if not parts:
        return "抱歉，系统暂时无法给出可靠的回答。如有不适请及时就医；如为急症请立即拨打 120。"
    return "\n".join(parts)


# ---------------------------------------------------------------- 降级模式（两档 / 三档模型全部不可用）
# 这里没有任何生成模型参与：输出 = 知识库字段原文 + 固定话术，所以不会有“生成式幻觉”。
# 但它有三类风险，下面逐条防护：
#   1. 知识库本身错误或未审核   → 只用精选 KB（kb.json），不用 Wikidata 扩展层的“可能病因”（关系噪声大），并标注“未经审核”
#   2. 实体链接认错 / 有歧义     → 命中多个药名时全部列出、不替用户选；只认精确命中的条目
#   3. 缺少个体化判断（孕期、儿童、老人、过敏、合并用药）→ 出现这些情况时不输出任何剂量，改为“需医生个体化判断”
#   另外：急症只给急救提示、高风险请求只给“需面诊”，都不附知识库内容。
DEGRADED_HEADER = "【降级模式】医疗模型暂时不可用，以下仅为知识库条目原文摘录，未经模型理解与个体化判断，不能作为用药依据："
_RISK_PROFILE = ("孕", "怀孕", "哺乳", "宝宝", "婴儿", "孩子", "儿童", "小孩", "岁的娃", "老人", "老年", "过敏", "肝", "肾", "在吃", "正在服用")


def kb_fallback_answer(query: str, assessment=None, history: str = "") -> str:
    from .knowledge import load_kb
    from .safety import EMERGENCY_BANNER, HIGH_RISK_DEFERRAL
    if assessment is not None and (assessment.level == "emergency" or assessment.crisis):
        return EMERGENCY_BANNER
    if assessment is not None and assessment.high_risk:
        return HIGH_RISK_DEFERRAL
    kb = load_kb()
    ents = kb.link_entities(query)
    risky = any(w in f"{history} {query}" for w in _RISK_PROFILE)
    parts = []
    for d in ents["drug"][:3]:
        v = kb.drugs[d]
        dose = "（您的情况可能涉及孕期/儿童/老人/过敏/肝肾功能/合并用药，剂量需医生个体化判断，此处不提供）" if risky \
            else f"说明书用法：{v['dosage']}。"
        parts.append(f"· {v['name']}（{v['generic_name']}）：适应症 {v['indications']}；禁忌 {v['contraindications']}；{dose}")
    for s in ents["symptom"][:2]:
        v = kb.symptoms[s]
        parts.append(f"· {s}：常见相关原因包括{'、'.join(v['possible_causes'])}（并非诊断）。{v['recommendation']}。")
    for g in ents["disease"][:1]:
        v = kb.guidelines[g]
        parts.append(f"· {v['topic']}：{v['content']}（出处：{v['source']} {v['year']}）")
    if not parts:
        return "抱歉，医疗模型暂时不可用，且知识库中没有与您问题直接对应的条目，我无法给出可靠回答。如有不适请及时就医；如为急症请立即拨打 120。"
    if len(ents["drug"]) > 1:
        parts.append("· 涉及多种药物时请勿自行合用，请咨询医生或药师。")
    return DEGRADED_HEADER + "\n" + "\n".join(parts) + "\n（知识库条目未经医生审核，请以说明书与医嘱为准。）"
