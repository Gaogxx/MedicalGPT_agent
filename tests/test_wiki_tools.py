"""医药 Wiki 接入工具（check_interaction / search_guideline / search_knowledge）与维护机制（隔离区、审核签字、门槛）。"""
from __future__ import annotations

import json

import pytest

from medagent.knowledge import load_kb
from medagent.knowledge.interactions import class_terms, label_pair
from medagent.knowledge.wiki import (DISCLAIMER, Builder, indication_links, load_wiki, page_body_hash,
                                     render_pages)
from medagent.knowledge.wiki_check import GATES, apply_quarantine, gate


def test_label_pair_by_name_and_class():
    lp = label_pair("氨氯地平", "辛伐他汀")                 # 氨氯地平说明书点名辛伐他汀
    assert any("辛伐他汀" in f["text"] for f in lp["found"])
    lp = label_pair("西地那非", "硝酸甘油")                 # 只写了“硝酸酯类”，靠 ATC C01DA 对上
    assert lp["found"] and lp["found"][0]["matched"].startswith("类别")


def test_class_terms_do_not_overreach():
    assert "镇静催眠" not in class_terms(["N02AX02"])      # 曲马多是阿片类，不是镇静催眠药
    assert "阿片" in class_terms(["N02AX02"])


def test_check_interaction_uses_labels_and_stays_honest():
    kb = load_kb()
    r = kb.check_interaction(["西地那非", "硝酸甘油"])
    assert r["interactions"] and r["interactions"][0]["severity"] == "label"
    r = kb.check_interaction(["二甲双胍", "氯雷他定"])
    assert not r["interactions"] and "不代表绝对安全" in r["advice"]
    r = kb.check_interaction(["布洛芬", "华法林"])          # 精选库手写的组合仍然优先
    assert r["interactions"][0]["severity"] == "major"


def test_check_interaction_switch(monkeypatch):
    monkeypatch.setenv("MEDAGENT_WIKI_TOOLS", "0")
    assert not load_kb().check_interaction(["西地那非", "硝酸甘油"])["interactions"]


def test_search_guideline_label_drugs():
    kb = load_kb()
    r = kb.search_guideline("哮喘")                          # Wikidata 没有“哮喘”，来自知识图谱病名
    assert r["found"] and r["label_drugs"]["drugs"]
    assert "不是治疗推荐" in r["label_drugs"]["note"]
    r = kb.search_guideline("高血压")                        # 精选指南：没问用药就不附药名清单
    assert "label_drugs" not in r
    assert "label_drugs" in kb.search_guideline("高血压", "用药")


def test_indication_links_rules():
    kb = {"Q1": {"name": "甲药", "kb_tier": "label", "indications": "用于2型糖尿病患者的血糖控制。", "source": {}},
          "Q2": {"name": "乙药", "kb_tier": "label", "indications": "本品不用于糖尿病酮症酸中毒。", "source": {}},
          "Q3": {"name": "丙药", "kb_tier": "label", "indications": "人类免疫缺陷病毒感染。", "source": {}}}
    dis = {"D1": {"zh": "2型糖尿病", "aliases": []}, "D2": {"zh": "糖尿病", "aliases": []},
           "D3": {"zh": "免疫缺陷", "aliases": []}}
    links = indication_links(kb, dis)
    assert [x["q"] for x in links["D1"]] == ["Q1"]
    assert [x["q"] for x in links["D2"]] == ["Q1"]          # 亚型同时记到上位病；否定句（乙药）不记
    assert "D3" not in links                                 # “免疫缺陷病毒”不是“免疫缺陷”


def test_link_checks_drop_rejected_links():
    kb = json.loads(open("medagent/knowledge/drug_kb.json", encoding="utf-8").read())
    wd = json.loads(open("medagent/knowledge/wikidata_kb.json", encoding="utf-8").read())
    b = Builder(kb, wd)
    d, xs = next((d, xs) for d, xs in b.dis_label_raw.items() if xs)
    from medagent.knowledge.wiki import _h
    bad = {f"{xs[0]['q']}|{d}|{_h(b.indication_text(xs[0]['q']))}": {"ok": False}}
    b2 = Builder(kb, wd, None, bad)
    assert b2.links_dropped == 1
    assert xs[0]["q"] not in {x["q"] for x in b2.dis_label.get(d, [])}


def test_wiki_evidence_by_intent():
    w = load_wiki()
    ev = w.evidence("西地那非和硝酸甘油能一起用吗")
    assert ev[0]["id"].startswith("wiki:interaction") and "硝酸酯" in ev[0]["text"]
    ev = w.evidence("哮喘一般用什么药")
    assert ev[0]["id"].startswith("wiki:diseases/") and "说明书适应症提到本病的药物" in ev[0]["text"]


def test_apply_quarantine():
    drugs = {"Q1": {"dosage": "x", "source": {"dosage": "s"}, "by_form": {"注射": {"dosage": "y"}}}}
    n = apply_quarantine(drugs, {"Q1": {"dosage": {}, "by_form.注射.dosage": {}}})
    assert n == 2 and "dosage" not in drugs["Q1"] and "dosage" not in drugs["Q1"]["by_form"]["注射"]


def test_review_hash_ignores_review_marks_but_not_content():
    page = render_pages()["drugs/阿司匹林.md"]
    h = page_body_hash(page)
    signed = page.replace("reviewed: false", "reviewed: true\nreviewer: \"x\"").replace(DISCLAIMER, "> 已由x于 2026-01-01 审核")
    assert page_body_hash(signed) == h
    assert page_body_hash(page.replace("禁", "慎", 1)) != h


def test_gate(tmp_path):
    def w(name, obj):
        p = tmp_path / name
        p.write_text(json.dumps(obj), encoding="utf-8")
        return str(p)
    dl = {"metrics": {s: {"all": {"ref_recall": r, "unsupported_dose_rate": 0.0, "cross_form_dose_rate": 0.0}}
                      for s, r in (("base", 0.20), ("cand", 0.21))}}
    wk = {s: {"drug_choice": {"any_supported": 0.5, "contra_named": 0.0},
              "interaction": {"flagged": f, "said_safe": 0.0}} for s, f in (("base", 0.8), ("cand", 0.7))}
    kb = {"this_model": {"auto_pass_rate": 0.92, "high_risk_compliance": 0.94}}
    lines, ok = gate(w("dl.json", dl), w("wk.json", wk), w("b.json", kb), w("c.json", kb))
    assert not ok and any("相互作用 指出问题" in x and x.startswith("不通过") for x in lines)
    assert len(lines) == len(GATES)


@pytest.mark.parametrize("q", ["我在吃华法林，能同时吃阿莫西林吗？"])
def test_search_knowledge_puts_interaction_first(q):
    ev = load_wiki().evidence(q)
    assert "华法林" in ev[0]["title"] and "阿莫西林" in ev[0]["title"]


def test_narrow_class_not_widened():
    """缬沙坦说明书写的是“保钾利尿剂”：不能按“利尿剂”对上布美他尼（排钾的袢利尿剂）。"""
    assert not label_pair("缬沙坦", "布美他尼")["found"]
    assert label_pair("缬沙坦", "螺内酯")["found"]


def test_specific_drug_beats_curated_category():
    kb = load_kb()
    assert kb.search_drug("注射用头孢曲松钠")["name"] == "头孢曲松"
    assert kb.search_drug("头孢")["name"] == "头孢类抗生素"
    assert kb.search_drug("布洛芬缓释胶囊")["name"] == "布洛芬"


def test_degraded_mode_uses_wiki():
    from medagent.templates import DEGRADED_HEADER, kb_fallback_answer
    a = kb_fallback_answer("西地那非和硝酸甘油能一起用吗")
    assert a.startswith(DEGRADED_HEADER) and "硝酸酯" in a
    a = kb_fallback_answer("哮喘一般用什么药")
    assert "说明书适应症写明了本病的药" in a and "不是治疗推荐" in a
    a = kb_fallback_answer("尼古丁贴剂怎么用")
    assert "没有外用剂型" in a and "mg" not in a                     # 问到的剂型没有：不给别的剂型的剂量
    a = kb_fallback_answer("我怀孕了，头孢克肟一次吃多少")
    assert "此处不提供" in a and "0.1g" not in a                     # 孕期：不给剂量


def test_degraded_mode_skips_quarantined(monkeypatch):
    import json
    from medagent.knowledge import ext as ext_mod
    from medagent.templates import kb_fallback_answer
    ext_mod.load_ext.cache_clear()
    q = next(q for q, e in json.load(open("medagent/knowledge/drug_kb.json", encoding="utf-8"))["drugs"].items()
             if e["name"] == "头孢克肟")
    monkeypatch.setattr("medagent.knowledge.wiki_check.QUARANTINE_PATH",
                        type("P", (), {"exists": lambda s: True, "read_text": lambda s, encoding=None:
                                       json.dumps({q: {"dosage": {}, "by_form.口服.dosage": {}}})})())
    try:
        assert "0.1g" not in kb_fallback_answer("头孢克肟一次吃多少")
    finally:
        ext_mod.load_ext.cache_clear()
