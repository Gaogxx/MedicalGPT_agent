"""说明书相互作用查找（2026-11-20）：check_interaction 原先只认精选库 10 个药 + 10 对手写组合，
编译好的 450 份说明书「药物相互作用」段落没被用上。这里按两种方式在说明书里找另一个药：

  1. 点名：A 的相互作用段落里出现 B 的药名 / 别名（“与华法林合用……”）
  2. 点类：A 的段落提到 B 所属的药物类别（“与非甾体抗炎药合用……”，B = 布洛芬，ATC M01AE01）
     类别 → ATC 前缀的对照表见 CLASS_TERMS，只收说明书里常见、能和 ATC 一一对上的类别

只摘原句，不推断、不分级：找到了返回原句和出处（severity 记为 "label"），找不到时如实说“说明书相互作用部分没提到”，
不能说成“没有相互作用”。MEDAGENT_WIKI_TOOLS=0 关闭（回到只看精选库）。
"""
from __future__ import annotations

import os
import re
from typing import Any

# (ATC 前缀, 说明书里的类别说法)；同一个药可以落在多个类别里
CLASS_TERMS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("B01AA",), ("香豆素", "维生素K拮抗", "口服抗凝")),
    (("B01A",), ("抗凝药", "抗凝剂", "抗凝血药", "抗栓药", "抗血栓")),
    (("B01AC",), ("抗血小板",)),
    (("M01A", "N02BA", "N02BB", "N02BE"), ("非甾体抗炎", "非甾体类抗炎", "NSAID", "解热镇痛")),
    (("N02BA", "B01AC06"), ("水杨酸",)),
    (("N02A", "R05DA"), ("阿片",)),
    (("N05BA", "N05CD"), ("苯二氮",)),
    (("N05CA", "N03AA"), ("巴比妥",)),
    (("N05B", "N05C", "N02A", "N05A"), ("中枢神经系统抑制", "中枢抑制")),
    (("N05B", "N05C"), ("镇静催眠", "镇静药", "镇静剂", "安眠药")),
    (("N05A",), ("抗精神病",)),
    (("N05AA", "N05AB", "N05AC"), ("吩噻嗪",)),
    (("N05AD",), ("丁酰苯",)),
    (("N06AA",), ("三环类抗抑郁", "三环抗抑郁")),
    (("N06AB",), ("选择性5-羟色胺再摄取抑制", "5-羟色胺再摄取抑制", "SSRI", "5-羟色胺能")),
    (("N06AF", "N06AG"), ("单胺氧化酶抑制", "MAO抑制", "MAOI")),
    (("N06A",), ("抗抑郁",)),
    (("N03",), ("抗癫痫", "抗惊厥")),
    (("A10A",), ("胰岛素",)),
    (("A10B",), ("降糖药", "口服降糖", "降血糖药")),
    (("A10BB", "A10BX"), ("磺脲", "磺酰脲", "胰岛素促泌", "促胰岛素分泌")),
    (("A02A",), ("抗酸药", "抗酸剂")),
    (("A02BC",), ("质子泵抑制",)),
    (("A02BA",), ("H2受体拮抗", "H2受体阻滞", "H₂受体拮抗")),
    (("C10AA",), ("他汀", "HMG-CoA还原酶抑制")),
    (("C10AC",), ("胆汁酸螯合", "考来烯胺")),
    (("C03",), ("利尿剂", "利尿药")),
    (("C03A",), ("噻嗪",)),
    (("C03D", "C03DA"), ("保钾利尿",)),
    (("C07",), ("β受体阻滞", "β-受体阻滞", "β阻滞", "β受体阻断", "β-阻滞")),
    (("C08",), ("钙通道阻滞", "钙拮抗", "钙离子拮抗")),
    (("C09A", "C09B"), ("血管紧张素转换酶抑制", "ACE抑制", "ACEI")),
    (("C09C", "C09D"), ("血管紧张素II受体拮抗", "血管紧张素Ⅱ受体拮抗", "血管紧张素受体阻滞", "ARB")),
    (("C02", "C03", "C07", "C08", "C09"), ("降压药", "抗高血压", "降血压药")),
    (("C01B",), ("抗心律失常",)),
    (("C01AA",), ("洋地黄", "强心苷")),
    (("C01DA",), ("硝酸酯",)),
    (("G03A",), ("口服避孕", "激素避孕", "避孕药")),
    (("H02",), ("皮质类固醇", "糖皮质激素", "皮质激素", "肾上腺皮质激素")),
    (("J01FA",), ("大环内酯",)),
    (("J01MA",), ("喹诺酮",)),
    (("J01AA",), ("四环素",)),
    (("J01G",), ("氨基糖苷",)),
    (("J01E",), ("磺胺",)),
    (("J02AC",), ("唑类抗真菌",)),
    (("J05AE",), ("蛋白酶抑制剂",)),
    (("L04",), ("免疫抑制",)),
    (("H03A",), ("甲状腺制剂", "甲状腺激素")),
    (("P01B",), ("抗疟",)),
    (("M03A",), ("肌松药", "肌肉松弛", "神经肌肉阻滞")),
    (("A12BA",), ("钾补充", "补钾", "钾盐")),
)
_GENERIC = re.compile(r"(如与其他药物同时使用|详情请咨询|尚不明确|尚未进行|未进行.{0,6}研究|无已知|原文未列出)")
_SENT = re.compile(r"(?<=[。；;！!\n])")


def enabled() -> bool:
    return os.environ.get("MEDAGENT_WIKI_TOOLS", "1") != "0"


def _resolve(name: str) -> tuple[str, dict[str, Any]] | None:
    """药名 → (QID, 条目)。和 search_drug 同口径：精确名 / 别名优先，再做最长匹配。
    药品库没有说明书条目的药也返回 Wikidata 的名字和 ATC（用来在另一个药的说明书里点名 / 点类）。"""
    from .ext import load_ext
    ext = load_ext()
    if not ext:
        return None
    n = name.strip()
    if f"curated:{n}" in ext.labels:
        return f"curated:{n}", ext.labels[f"curated:{n}"]
    hit = ext.find(n, "drug")
    if not hit:
        return None
    q, wd = hit
    return q, ext.labels.get(q) or {"name": wd["zh"], "aliases": wd.get("aliases", []), "atc": wd.get("atc", [])}


def _names(e: dict[str, Any], extra: str = "") -> list[str]:
    ns = [e.get("name", ""), *e.get("aliases", []), *e.get("new_aliases", {}), extra]
    return sorted({n for n in ns if n and len(n) >= 2}, key=len, reverse=True)


def class_terms(atc: list[str]) -> list[str]:
    return [t for prefixes, terms in CLASS_TERMS if any(a.startswith(p) for a in atc for p in prefixes) for t in terms]


ALL_TERMS = sorted({t for _, ts in CLASS_TERMS for t in ts}, key=len, reverse=True)
# 说明书里常见、比宽泛类别更细的说法：句子写的是这些时，不能按宽泛的类别对上（“保钾利尿剂”不是所有利尿剂）
NARROW = ("保钾利尿剂", "保钾利尿药", "袢利尿剂", "噻嗪类利尿剂", "口服抗凝药", "非去极化肌松药", "选择性β受体阻滞")


def _class_hit(s: str, t: str, terms: list[str]) -> bool:
    """句子里的类别词 t 是否指这个药的类别：t 的某次出现没被更细的类别词覆盖，或覆盖它的细类这个药也属于，才算。"""
    finer = [x for x in (*ALL_TERMS, *NARROW) if t in x and x != t]
    spans = [(m.start(), m.end(), x) for x in finer for m in re.finditer(re.escape(x), s)]
    for m in re.finditer(re.escape(t), s):
        cover = [x for a, b, x in spans if a <= m.start() and m.end() <= b]
        if not cover or any(tt != t and tt in x for x in cover for tt in terms):
            return True
    return False


def _mentions(text: str, names: list[str], terms: list[str]) -> list[tuple[str, str]]:
    """text 里提到 names（点名）或 terms（点类）的句子 → [(句子, 怎么对上的)]，最多 3 句。"""
    out: list[tuple[str, str]] = []
    for s in _SENT.split(text):
        s = s.strip()
        if len(s) < 6 or _GENERIC.search(s):
            continue
        how = next((f"点名「{n}」" for n in names if n in s), None) or \
            next((f"类别「{t}」" for t in terms if t in s and _class_hit(s, t, terms)), None)
        if how:
            out.append((s[:220] + ("……" if len(s) > 220 else ""), how))
        if len(out) >= 3:
            break
    return out


def label_pair(a: str, b: str) -> dict[str, Any] | None:
    """在 a、b 两份说明书的相互作用段落里互相找对方。返回 None 表示两个药都不在药品库里。"""
    ra, rb = _resolve(a), _resolve(b)
    if not ra and not rb:
        return None
    found, checked = [], []
    for (src, other, other_name) in ((ra, rb, b), (rb, ra, a)):
        if not src:
            continue
        text = src[1].get("interactions_text") or ""
        if not text:
            continue
        checked.append(src[1]["name"])
        names = _names(other[1], other_name) if other else [other_name] if len(other_name) >= 2 else []
        terms = class_terms(other[1].get("atc", [])) if other else []
        for sent, how in _mentions(text, names, terms):
            found.append({"label_of": src[1]["name"], "text": sent, "matched": how,
                          "source": src[1].get("source", {}).get("interactions_text", "")})
    return {"found": found, "labels_checked": checked,
            "names": [r[1]["name"] if r else n for r, n in ((ra, a), (rb, b))]}
