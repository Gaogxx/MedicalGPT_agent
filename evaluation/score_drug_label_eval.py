#!/usr/bin/env python3
"""给药品库试点的两次端到端运行（补全前 / 补全后）打自动分，逐题配对（2026-11-18）。

指标（都不花钱，不需要裁判模型）：
  ref_recall      参考答案（说明书原文）要点被回答覆盖的比例：用法用量看“数字+单位 / 频次”，其余看分句关键词
  gap_rate        回答说“查不到 / 知识库没有 / 请查说明书”而没有给出实质内容的比例
  unsupported_dose 用法用量题里，回答给出的剂量数字既不在参考答案、也不在药品库条目里的比例（编造剂量的代理指标）
  cross_form_dose 明确问了剂型的用法用量题里，回答给出的剂量数字只出现在药品库“另一个已知剂型”的说明书里
                  （不在参考答案、不在所问剂型和未注明剂型的条目里），且回答没点明那个剂型——把口服剂量当成注射剂量答的代理指标

  python evaluation/score_drug_label_eval.py off=evaluation/results/live_druglabel_off.json on=evaluation/results/live_druglabel_on.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medagent.knowledge.drug_label_kb import form_of  # noqa: E402
DOSE = re.compile(r"\d+(?:\.\d+)?\s*(?:～|~|-|—|至)?\s*\d*(?:\.\d+)?\s*(?:mg|g|ml|mL|μg|ug|µg|万单位|万U|单位|IU|U|片|粒|丸|袋|支|滴|喷|次|小时|h)")
SPLIT = re.compile(r"[，,。；;、：:（）()\s]|或|及|和|等")
GAP = re.compile(r"未找到|查不到|没有查到|不含(剂量|禁忌)|未记录|无法提供|暂无.{0,6}(信息|资料)|请(查阅|参看|以).{0,8}说明书")


def _nums(text: str) -> set[str]:
    return {re.sub(r"\s", "", m) for m in DOSE.findall(text)} | set(re.findall(r"\d+(?:\.\d+)?", text))


ROUTE = {"ORAL": "口服", "INTRAVENOUS": "注射", "INTRAMUSCULAR": "注射", "SUBCUTANEOUS": "注射", "TOPICAL": "外用",
         "OPHTHALMIC": "眼用", "RESPIRATORY (INHALATION)": "吸入", "NASAL": "耳鼻", "RECTAL": "栓剂", "VAGINAL": "阴道用"}
FORM_WORDS = {"口服": ("口服", "片", "胶囊", "颗粒", "糖浆", "口服液"), "注射": ("注射", "静脉", "肌内", "肌注", "静滴", "皮下"),
              "外用": ("外用", "涂", "乳膏", "软膏"), "栓剂": ("栓", "直肠", "阴道"), "眼用": ("滴眼", "眼膏", "眼"),
              "吸入": ("吸入", "气雾", "雾化"), "耳鼻": ("滴耳", "滴鼻", "鼻"), "阴道用": ("阴道",)}


def kb_dose_by_form(e: dict) -> dict:
    """药品库条目 → {剂型: 用法用量}；剂型看不出来的记在 None 下。主条目的剂型看来源商品名或 openFDA 给药途径。"""
    from medagent.knowledge.drug_label_kb import form_of
    out: dict = {}
    for cat, r in e.get("by_form", {}).items():
        if r.get("dosage"):
            out[None if cat == "未注明剂型" else cat] = out.get(None if cat == "未注明剂型" else cat, "") + r["dosage"]
    if e.get("dosage"):
        src = e.get("source", {}).get("dosage", "")
        prod = re.search(r"「([^」]*)」", src)
        route = re.search(r"给药途径 ([A-Z ()、]+)", e["dosage"])
        cat = form_of(prod.group(1)) if prod else (ROUTE.get(route.group(1).split("、")[0].strip()) if route else None)
        out[cat] = out.get(cat, "") + e["dosage"]
    return out


def ref_recall(ans: str, ref: str, field: str) -> float:
    if field == "dosage":
        terms = list({re.sub(r"\s", "", m) for m in DOSE.findall(ref)})
        a = re.sub(r"\s", "", ans).replace("～", "~").replace("—", "-")
        if terms:
            return sum(t.replace("～", "~").replace("—", "-") in a for t in terms) / len(terms)
    terms = [t.strip() for t in SPLIT.split(ref) if len(t.strip()) >= 2]
    terms = [re.sub(r"^(偶见|可见|少见|罕见|常见|可有|有|出现|患者|者|本品|对本品)", "", t) for t in terms]
    terms = [t for t in terms if len(t) >= 2][:30]
    if not terms:
        return 1.0
    return sum(t in ans or t[:3] in ans for t in terms) / len(terms)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))     # --cases=… --kb=… --out=…（复算旧版本用）
    runs = dict(a.split("=", 1) for a in args)
    cases = {json.loads(line)["id"]: json.loads(line) for line in
             open(opt.get("cases", ROOT / "evaluation" / "datasets" / "drug_label_eval.jsonl"), encoding="utf-8")}
    kb = json.loads(Path(opt.get("kb", ROOT / "medagent" / "knowledge" / "drug_kb.json")).read_text(encoding="utf-8"))["drugs"]
    res = {}
    per_case = {}
    for name, path in runs.items():
        ans = {r["id"]: r for r in json.load(open(path, encoding="utf-8"))["answers"]}
        rows = []
        for cid, c in cases.items():
            if cid not in ans:
                continue
            a = ans[cid]["answer"]
            rec = ref_recall(a, c["reference"], c["field"])
            gap = bool(GAP.search(a)) and rec < 0.5
            uns = cross = None
            if c["field"] == "dosage":
                e = kb.get(c["qid"], {})
                typed = kb_dose_by_form(e)                      # 剂型 → 药品库里该剂型的用法用量
                got = {re.sub(r"\s", "", m) for m in DOSE.findall(a) if not re.search(r"次|小时|h$", m)}
                allowed = _nums(c["reference"]) | _nums(" ".join(typed.values()))
                uns = any(not (re.findall(r"\d+(?:\.\d+)?", g) and all(n in allowed for n in re.findall(r"\d+(?:\.\d+)?", g)))
                          for g in got)
                af = c.get("asked_form") or form_of(c["product"])
                if af:          # 只看明确问了剂型的题：给出的剂量只出现在“另一个已知剂型”的说明书里，且回答没点明那个剂型
                    same = _nums(c["reference"] + typed.get(af, "") + typed.get(None, ""))
                    nums = {n for g in got for n in re.findall(r"\d+(?:\.\d+)?", g)}
                    cross = False
                    for cat, txt in typed.items():
                        if cat in (None, af):
                            continue
                        leaked = [n for n in nums if n in _nums(txt) and n not in same]
                        if leaked and not any(k in a for k in FORM_WORDS.get(cat, (cat,))):
                            cross = True
            rows.append({"id": cid, "recall": rec, "gap": gap, "unsupported_dose": uns, "cross_form_dose": cross, "exact": c["exact_in_train"],
                         "kb_has_field": c["kb_has_field"], "field": c["field"]})
            per_case.setdefault(cid, {})[name] = round(rec, 3)

        def agg(sub):
            d = [r for r in sub if r["unsupported_dose"] is not None]
            return {"n": len(sub), "ref_recall": round(sum(r["recall"] for r in sub) / max(len(sub), 1), 4),
                    "gap_rate": round(sum(r["gap"] for r in sub) / max(len(sub), 1), 4),
                    "unsupported_dose_rate": round(sum(r["unsupported_dose"] for r in d) / max(len(d), 1), 4) if d else None,
                    "cross_form_dose_rate": (round(sum(r["cross_form_dose"] for r in x) / len(x), 4)
                                             if (x := [r for r in sub if r["cross_form_dose"] is not None]) else None),
                    "n_form_asked": sum(r["cross_form_dose"] is not None for r in sub)}
        res[name] = {"all": agg(rows), "not_exact_in_train": agg([r for r in rows if not r["exact"]]),
                     "kb_has_field": agg([r for r in rows if r["kb_has_field"]]),
                     "by_field": {f: agg([r for r in rows if r["field"] == f]) for f in ("dosage", "side_effects", "contraindications")}}
    names = list(runs)
    if len(names) == 2:
        a, b = names
        res["paired"] = {f"{b}_better": sum(v[b] > v[a] + 0.1 for v in per_case.values() if len(v) == 2),
                         f"{a}_better": sum(v[a] > v[b] + 0.1 for v in per_case.values() if len(v) == 2)}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    out = Path(opt.get("out", ROOT / "evaluation" / "results" / "drug_label_eval.json"))
    out.write_text(json.dumps({"runs": runs, "metrics": res, "per_case": per_case}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
