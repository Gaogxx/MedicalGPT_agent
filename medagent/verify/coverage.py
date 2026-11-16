"""漏答检查（2026-11-16）：问题拆成子问题后，逐个确认回答里谈到了。

为什么：最终 9B 在路由数据集上的失败里，“漏答子问题 / 答非所问”（incomplete）占 81 条中的 54 条。
原有的 incomplete_multi 只在“先…再…”“和…药 / 饮食”这类句式出现时检查点名的药是否都提到。这里改用
复杂问诊（complex_consult）同一套子问题拆分（QueryRewriter：先/再、A 和 B 分别、按分句拆且各段话题不同），
每个子问题取它“独有”的锚点（疾病 / 药 / 症状实体 + 话题词，去掉各子问题共有的主语），回答里一个锚点及其同义说法都没有 → 漏答。

偏保守：子问题没有独有锚点就不判；话题词带同义说法（“饮食”也认“吃 / 食物 / 忌口”），宁可放过也不误拦
（误拦的代价是多一次强模型重答）。
"""
from __future__ import annotations

import re
from functools import lru_cache

# 话题词 → 回答里可接受的说法
TOPICS: dict[str, tuple[str, ...]] = {
    "饮食": ("饮食", "吃", "食物", "食品", "忌口", "营养", "喝", "膳食"),
    "吃饭": ("饮食", "吃", "食物", "进食"),
    "运动": ("运动", "锻炼", "活动", "散步", "跑步", "健身"),
    "副作用": ("副作用", "不良反应", "反应"),
    "不良反应": ("副作用", "不良反应", "反应"),
    "禁忌": ("禁忌", "禁用", "慎用", "不能", "不宜", "避免"),
    "原因": ("原因", "引起", "导致", "病因", "因为", "由于", "可能是", "可能与"),
    "症状": ("症状", "表现", "出现", "感觉"),
    "用法": ("用法", "用量", "剂量", "服用", "每次", "一次", "一日", "说明书", "遵医嘱"),
    "用量": ("用法", "用量", "剂量", "服用", "每次", "一次", "一日", "说明书", "遵医嘱"),
    "适应症": ("适应症", "用于", "治疗", "主治", "适用"),
    "指南": ("指南", "建议", "推荐"),
    "检查": ("检查", "化验", "检测", "筛查", "B超", "超声", "CT", "核磁", "活检"),
    "治疗": ("治疗", "疗法", "用药", "手术", "治", "药物"),
    "手术": ("手术", "术", "切除", "微创"),
    "预防": ("预防", "避免", "防止", "降低风险"),
    "传染": ("传染", "传播", "感染", "隔离"),
    "保存": ("保存", "储存", "存放", "冷藏", "避光"),
    "一起吃": ("一起", "同服", "合用", "同时", "间隔", "相互作用"),
    "药": ("药", "片", "胶囊", "颗粒", "用药", "服用"),
}
_TOPIC_RE = re.compile("|".join(sorted(TOPICS, key=len, reverse=True)))


@lru_cache(maxsize=1)
def _rewriter():
    from ..rag.rewrite import QueryRewriter
    return QueryRewriter()


def _anchors(sub: str) -> set[str]:
    rw = _rewriter()
    return set(rw._entities(sub)) | set(_TOPIC_RE.findall(sub))


def _mentioned(anchor: str, answer: str) -> bool:
    if anchor in TOPICS:
        return any(w in answer for w in TOPICS[anchor])
    if anchor in answer:
        return True
    syn = _rewriter().kb.synonyms
    return any(alias in answer for table in ("drug", "disease", "symptom")
               for alias, std in syn.get(table, {}).items() if std == anchor)


def missed_subquestions(query: str, answer: str) -> list[str]:
    """返回没被回答谈到的子问题（只看它独有的锚点）。拆不出 2 个以上子问题时返回空。"""
    subs = _rewriter().rewrite(query).sub_queries
    if len(subs) < 2:
        return []
    anchors = [_anchors(s) for s in subs]
    shared = set.intersection(*anchors) if anchors else set()
    missed = []
    for sub, a in zip(subs, anchors):
        own = a - shared
        if own and not any(_mentioned(x, answer) for x in own):
            missed.append(sub)
    return missed
