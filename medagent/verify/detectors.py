"""静默失败检测器。每条规则都来自真实失败样本（runtime/dsh-medical-plugins/test_results_extended.txt）：

  tool_leak          “ptom\\n{"name": "check_symptom", ...}” —— vLLM hermes 解析失败，工具调用 JSON 漏给用户
  garbled            “��好，突然出现的抽搐……” —— 流式解码截断多字节字符
  emergency_missing  “剧烈头痛还呕吐” 被当普通头痛，建议“休息观察 3 天”
  unsupported_detail “您的症状是嗓子疼，持续了3天”（用户没说 3 天）、“您提到的症状（口渴）”（用户没提）
  fabricated_history “根据您之前提到的头痛、阿司匹林……”（根本没有历史对话）
  incomplete_multi   “先查症状再查布洛芬副作用” 只做了第一步，结尾“请问是否继续？”
  ungrounded_dose    回答里的剂量在工具结果 / 证据里找不到
  fabricated_on_miss 工具返回 found=false，回答却给出了具体用法用量
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..safety import SafetyAssessment, contains_prescription, has_emergency_advice
from ..types import ErrorKind, ToolCall


@dataclass
class Finding:
    code: str
    kind: ErrorKind
    severity: str          # block：必须修复/重生成；warn：记录并由输出关修补
    evidence: str = ""


_LEAK = re.compile(r'\{\s*"name"\s*:\s*"[a-z_]+"\s*,\s*"arguments"\s*:|</?tool_call>|^\s*Action( Input)?\s*:', re.M)
_INTERNAL_TOOL_NAME = re.compile(r"`?(check_symptom|search_drug|check_interaction|search_guideline|search_knowledge)`?")
_GARBLED = re.compile(r"�")
_DOSE = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|g|μg|毫克|克|ml|毫升)", re.I)
_DURATION = re.compile(r"(持续|已经|有)(了)?\s*([0-9一二两三四五六七八九十半]+)\s*(个)?(天|周|星期|个月|月|小时)")
_CLAIMED = re.compile(r"您(提到|说|描述)的(症状)?[（(“\"]([^）)”\"]{1,12})[）)”\"]")
_HISTORY_CLAIM = re.compile(r"(您|你)(之前|此前|曾经?|刚才)(提到|问过|询问|说过)|我之前(给出|提供)的")
_DEFER_ONLY = re.compile(r"(我会|我将|我可以)(为您|帮您)?(查询|调用|检索)[\s\S]{0,20}(请问|是否)(是否)?(继续|需要)")
_GENERIC_DRUG_ASK = re.compile(r"(查|推荐|吃什么|什么)药|药(和|与|以及)")
_MISS_ACK = re.compile(r"未(能)?找到|没有找到|无法找到|查不到|未查到|没查到|暂无|无法确认|不在.{0,6}(范围|知识库)")
_MULTI = re.compile(r"(先|首先).{1,20}(再|然后)|和.{0,6}(饮食|药)|(药|饮食).{0,3}(和|以及|及).{0,4}(药|饮食|建议)")
_UNIT = {"克": "g", "毫克": "mg", "毫升": "ml"}


def _norm_dose(v: str, unit: str) -> str:
    u = _UNIT.get(unit, unit.lower())
    num = float(v)
    if u == "g" and num < 10:
        return f"{num * 1000:g}mg"
    return f"{num:g}{u}"


def _cn_num(s: str) -> str:
    table = {"一": "1", "二": "2", "两": "2", "三": "3", "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9",
             "十": "10", "半": "0.5"}
    return table.get(s, s)


def extract_leaked_tool_call(answer: str) -> ToolCall | None:
    """从泄漏的文本里把工具调用救回来：{"name": ..., "arguments": {...}}（容忍多余的右括号）。"""
    m = re.search(r'\{\s*"name"\s*:\s*"([a-z_]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})', answer, re.S)
    if not m:
        return None
    try:
        args = json.loads(m.group(2))
    except json.JSONDecodeError:
        return None
    return ToolCall(m.group(1), {k: v for k, v in args.items() if v not in ("", "未提及", None)})


def detect(answer: str, *, query: str, assessment: SafetyAssessment | None = None,
           observations: list[dict[str, Any]] | None = None, history: str = "",
           finish_reason: str | None = None) -> list[Finding]:
    f: list[Finding] = []
    text = answer or ""
    stripped = text.strip()
    ctx_text = f"{history} {query}"
    if len(stripped) < 8:
        f.append(Finding("empty", ErrorKind.SILENT, "block", stripped))
    if _LEAK.search(text):
        f.append(Finding("tool_leak", ErrorKind.MODEL, "block", _LEAK.search(text).group(0)))
    elif _INTERNAL_TOOL_NAME.search(text):
        f.append(Finding("internal_name_leak", ErrorKind.MODEL, "warn", _INTERNAL_TOOL_NAME.search(text).group(0)))
    if _GARBLED.search(text):
        f.append(Finding("garbled", ErrorKind.MODEL, "warn", "U+FFFD"))
    if finish_reason == "length":
        f.append(Finding("truncated", ErrorKind.SILENT, "block", "finish_reason=length"))
    if re.search(r"(.{6,}?)\1{3,}", text):
        f.append(Finding("repetition", ErrorKind.MODEL, "block", "repeated span"))

    if assessment is not None:
        if assessment.level == "emergency" and not _unconditional_emergency_advice(text):
            f.append(Finding("emergency_missing", ErrorKind.SILENT, "block", ",".join(assessment.emergency_hits)))
        if assessment.high_risk and contains_prescription(text):
            f.append(Finding("prescription_in_high_risk", ErrorKind.POLICY, "block", ""))

    for m in _DURATION.finditer(text):
        n = _cn_num(m.group(3))
        if n not in ctx_text and m.group(3) not in ctx_text and "超过" not in text[max(0, m.start() - 4):m.start()]:
            f.append(Finding("unsupported_detail", ErrorKind.SILENT, "block", m.group(0)))
            break
    for m in _CLAIMED.finditer(text):
        claim = re.sub(r"[^一-鿿]", "", m.group(3))
        overlap = sum(1 for ch in claim if ch in ctx_text) / max(len(claim), 1)
        if m.group(3) not in ctx_text and overlap < 0.6:   # 允许复述/同义改写（“早上”→“早晨”）
            f.append(Finding("unsupported_detail", ErrorKind.SILENT, "block", m.group(0)))
            break
    if not history.strip() and _HISTORY_CLAIM.search(text) and not re.search(r"没有|未提供|无法(直接)?(总结|回顾)", text):
        f.append(Finding("fabricated_history", ErrorKind.SILENT, "block", _HISTORY_CLAIM.search(text).group(0)))
    if _DEFER_ONLY.search(text):
        f.append(Finding("incomplete_multi", ErrorKind.SILENT, "block", _DEFER_ONLY.search(text).group(0)))
    elif multi_intent(query):
        # 多意图覆盖检查：问题里点名的药必须在回答里出现；泛问“查药/吃什么药”时回答里至少要有一个药名
        from ..knowledge import load_kb
        kb = load_kb()
        named = kb.link_entities(query)["drug"]
        missing = [d for d in named if d not in text and not any(a in text for a, s in kb.synonyms["drug"].items() if s == d)]
        generic_miss = not named and _GENERIC_DRUG_ASK.search(query) and not kb.link_entities(text)["drug"]
        if missing or generic_miss:
            f.append(Finding("incomplete_multi", ErrorKind.SILENT, "block", ",".join(missing) or "drug"))

    if observations is not None:
        evid = json.dumps(observations, ensure_ascii=False)
        ev_doses = {_norm_dose(v, u) for v, u in _DOSE.findall(evid)}
        for v, u in _DOSE.findall(text):
            if _norm_dose(v, u) not in ev_doses and f"{v}{u}" not in ctx_text:
                f.append(Finding("ungrounded_dose", ErrorKind.SILENT, "block", f"{v}{u}"))
                break
        misses = [o for o in observations if isinstance(o, dict) and o.get("found") is False and not o.get("is_category")]
        if misses and not _MISS_ACK.search(text) and (_DOSE.search(text) or re.search(r"适应症|用法用量|禁忌", text)):
            f.append(Finding("fabricated_on_miss", ErrorKind.SILENT, "block", str(misses[0].get("query") or "")))
    return f


def _unconditional_emergency_advice(text: str) -> bool:
    """“如果出现胸痛请拨打120”是条件句，不算对当前急症给出了急救建议。"""
    for sent in re.split(r"[。！!\n]", text):
        if has_emergency_advice(sent) and not re.search(r"如果|若|如出现|一旦出现|假如", sent):
            return True
    return False


def multi_intent(query: str) -> bool:
    return bool(_MULTI.search(query))
