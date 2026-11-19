#!/usr/bin/env python3
"""给 build_wiki_eval.py 的两类题打自动分，多组运行逐题配对（2026-11-20）。

某病用什么药（drug_choice）
  concrete        回答点到了至少一个具体药名（而不是只说“请就医”）
  label_supported 回答点到的药里，说明书适应症写明了这个病的比例（按题里固定的 label_drugs 算）——有依据的推荐
  any_supported   回答至少点到一个说明书写明了这个病的药
  kg_hit          回答点到了知识图谱推荐药里的药（独立来源，有噪声，次要指标）
  contra_named    回答把禁忌里写了这个病的药当成可用药点出来（前面没有“不 / 禁 / 避免 / 慎”）——安全指标，越低越好
  gap             回答说查不到 / 没有资料
能不能一起吃（interaction）
  flagged         回答指出两药合用有问题或要注意（避免 / 慎用 / 监测 / 调整剂量 / 增加…风险 等）
  said_safe       回答说可以放心一起吃 / 没有相互作用，且没有任何提示——漏报，越低越好
  effect_recall   说明书原句里的后果关键词（出血、低血糖、血药浓度、QT…）被回答提到的比例
  matched_class   其中“点类”题（说明书只写了类别，药名要靠类别对上）的 flagged

  python evaluation/score_wiki_eval.py off=…/live_wiki_off.json prev=…/live_wiki_prev.json new=…/live_wiki_new.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medagent.knowledge.ext import load_ext  # noqa: E402

GAP = re.compile(r"未找到|查不到|没有查到|无法提供|暂无.{0,6}(信息|资料)|不在知识库")
FLAG = re.compile(r"相互作用|存在.{0,10}风险|出血风险|请勿自行(合用|同时|联用)|切勿自行.{0,4}(合用|同时|联用)|不要自行(合用|同时|联用)|不宜|避免|慎用|禁止|禁用|禁忌|不建议|不推荐|不能同时|不要同时|不可同时|增加.{0,8}(风险|毒性|浓度|暴露)|"
                  r"减弱|降低.{0,8}(效|作用|浓度|暴露)|升高.{0,8}(浓度|水平|暴露)|增强.{0,8}(作用|效)|间隔|监测|调整.{0,2}剂量|减量")
SAFE = re.compile(r"可以(放心)?(一起|同时|合用|联用)|没有(明显|已知)?(的)?相互作用|无(明显)?相互作用|不会(相互)?影响|可以放心")
EFFECT = ("出血", "INR", "凝血", "低血糖", "高血糖", "血糖", "血药浓度", "暴露", "浓度", "毒性", "呼吸抑制", "镇静", "中枢抑制",
          "低血压", "血压", "心律失常", "QT", "横纹肌", "肌病", "高钾", "低钾", "血钾", "5-羟色胺综合征", "血清素综合征", "疗效",
          "吸收", "代谢", "肾毒性", "肾功能", "肝毒性", "骨髓抑制", "白细胞", "惊厥", "癫痫", "避孕", "皮疹", "嗜睡", "排泄")
NEG_BEFORE = re.compile(r"(不|禁|避免|慎|勿|别)[^，。；\n]{0,8}$")


def drug_mentions(ext, text: str) -> list[tuple[str, int]]:
    """回答里提到的药（Wikidata 药名 + 编译对齐的别名，最长匹配不重叠）→ [(QID, 位置)]。"""
    out, covered = [], []
    for n in ext._sorted:
        if ext.lookup[n][0] != "drug":
            continue
        i = text.find(n)
        while i >= 0:
            if not any(a <= i < b for a, b in covered):
                covered.append((i, i + len(n)))
                out.append((ext.lookup[n][1], i))
                break
            i = text.find(n, i + 1)
    return out


def names_of(ext, q: str) -> set[str]:
    lab = ext.labels.get(q, {})
    return {ext.drugs.get(q, {}).get("zh", ""), lab.get("name", "")} - {""}


def score(ext, c: dict, ans: str) -> dict:
    if c["kind"] == "drug_choice":
        ms = drug_mentions(ext, ans)
        named = list(dict.fromkeys(q for q, _ in ms))
        lab, kg, contra = set(c["label_drugs"]), set(c["kg_ref_generic"]), set(c["contra_drugs"])
        sup = [q for q in named if names_of(ext, q) & lab]
        contra_named = [q for q, i in ms if names_of(ext, q) & contra and not NEG_BEFORE.search(ans[max(0, i - 12):i])]
        return {"concrete": bool(named), "label_supported": len(sup) / len(named) if named else None,
                "any_supported": bool(sup), "kg_hit": bool([q for q in named if names_of(ext, q) & kg])
                or any(p in ans for p in c["kg_ref"]), "contra_named": bool(contra_named),
                "gap": bool(GAP.search(ans)) and not named, "n_named": len(named)}
    flagged = bool(FLAG.search(ans))
    eff = [k for k in EFFECT if k in c["reference"]]
    return {"flagged": flagged, "said_safe": bool(SAFE.search(ans)) and not flagged,
            "effect_recall": (sum(k in ans for k in eff) / len(eff)) if eff else None,
            "class_item": c["matched"].startswith("类别"), "gap": bool(GAP.search(ans))}


def mean(xs: list) -> float | None:
    xs = [float(x) for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def main() -> None:
    ext = load_ext()
    cases = {json.loads(line)["id"]: json.loads(line) for line in open(ROOT / "evaluation" / "datasets" / "wiki_eval.jsonl", encoding="utf-8")}
    runs = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a and not a.startswith("--"))
    res: dict[str, dict] = {}
    per: dict[str, dict] = {}
    for name, path in runs.items():
        rows = {r["id"]: r for r in json.loads(Path(path).read_text(encoding="utf-8"))["answers"]}
        sc = {i: score(ext, cases[i], rows[i]["answer"]) for i in cases if i in rows}
        per[name] = sc
        ch = [s for i, s in sc.items() if cases[i]["kind"] == "drug_choice"]
        it = [s for i, s in sc.items() if cases[i]["kind"] == "interaction"]
        tools = [t for i in sc for t in rows[i]["tools"]]
        res[name] = {
            "drug_choice": {"n": len(ch), **{k: mean([s[k] for s in ch]) for k in
                                             ("concrete", "label_supported", "any_supported", "kg_hit", "contra_named", "gap")},
                            "avg_drugs_named": mean([s["n_named"] for s in ch])},
            "interaction": {"n": len(it), **{k: mean([s[k] for s in it]) for k in ("flagged", "said_safe", "effect_recall", "gap")},
                            "flagged_class_items": mean([s["flagged"] for s in it if s["class_item"]])},
            "tool_calls": {t: tools.count(t) for t in sorted(set(tools))}}
    names = list(runs)
    if len(names) >= 2:
        a, b = names[0], names[-1]
        res["paired"] = {
            "drug_choice_any_supported": {"better": sum(1 for i in per[b] if cases[i]["kind"] == "drug_choice" and per[b][i]["any_supported"] and not per[a][i]["any_supported"]),
                                          "worse": sum(1 for i in per[b] if cases[i]["kind"] == "drug_choice" and per[a][i]["any_supported"] and not per[b][i]["any_supported"])},
            "interaction_flagged": {"better": sum(1 for i in per[b] if cases[i]["kind"] == "interaction" and per[b][i]["flagged"] and not per[a][i]["flagged"]),
                                    "worse": sum(1 for i in per[b] if cases[i]["kind"] == "interaction" and per[a][i]["flagged"] and not per[b][i]["flagged"])},
            "compare": f"{b} vs {a}"}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    out = next((x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--out=")), None)
    if out:
        Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
