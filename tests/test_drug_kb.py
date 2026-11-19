"""药品库补全（drug_label_kb.py）与合并后查询（ExtKB + search_drug）的单元测试。"""
from medagent.knowledge.drug_label_kb import _BOILER, clean, strip_name
from medagent.knowledge.ext import ExtKB

WD = {"meta": {"retrieved": "2026-11-09"}, "diseases": {},
      "drugs": {"Q1": {"zh": "坦度螺酮", "en": "tandospirone", "atc": ["N05BE03"], "desc": "药物", "aliases": [], "treats": []},
                "Q2": {"zh": "杜丁胺", "en": "dobutamine", "atc": ["C01CA07"], "desc": "药物", "aliases": ["酚丁"], "treats": []},
                "Q3": {"zh": "柠檬酸", "en": "citric acid", "atc": ["A09AB04"], "desc": "化合物", "aliases": ["枸橼酸"], "treats": []}}}
LABELS = {"drugs": {
    "Q1": {"name": "坦度螺酮", "generic_name": "tandospirone", "atc": ["N05BE03"], "kb_tier": "label",
           "dosage": "通常成人每次口服10mg，1日3次。", "side_effects": "嗜睡、头晕。",
           "new_aliases": {}, "source": {"dosage": "Huatuo 百科问答「枸橼酸坦度螺酮胶囊」（verbatim）",
                                         "side_effects": "Huatuo 百科问答「枸橼酸坦度螺酮胶囊」（verbatim）"}},
    "Q2": {"name": "杜丁胺", "generic_name": "dobutamine", "atc": ["C01CA07"], "kb_tier": "wikidata",
           "new_aliases": {"多巴酚丁胺": "llm_inn:dobutamine"}, "source": {}}}}


def test_strip_name_forms_and_salts():
    assert strip_name("枸橼酸坦度螺酮胶囊")[-1] == "坦度螺酮"
    assert strip_name("注射用盐酸多巴酚丁胺")[-1] == "多巴酚丁胺"
    assert strip_name("布洛芬缓释胶囊")[-1] == "布洛芬"
    assert strip_name("复方甘草片") == []                 # 复方不对到单一成分
    assert "氯化" not in strip_name("氯化钾缓释片")       # 不能把“氯化钾”剥成“氯化”


def test_clean_removes_boilerplate_only():
    t = "注意：同种药品可由于不同的包装规格有不同的用法或用量。本文只供参考。如果不确定，请参看药品随带的说明书或向医生询问。口服。一次50mg"
    assert clean(t) == "口服。一次50mg"
    assert _BOILER.sub("", "口服，一次1片。") == "口服，一次1片。"


def test_filled_entry_and_salt_not_linked():
    ext = ExtKB(WD, LABELS)
    q, _ = ext.find("枸橼酸坦度螺酮胶囊的用法用量", "drug")
    assert q == "Q1"                                      # 不再链到“枸橼酸 / 柠檬酸”
    e = ext.drug_entry(q)
    assert e["dosage"].startswith("通常成人每次口服10mg") and e["kb_tier"] == "label"
    assert "请查阅" in e["contraindications"]              # 没有来源的字段仍然明确说没有
    assert e["field_sources"]["dosage"].startswith("说明书摘录")


def test_new_alias_links_but_unfilled_entry_stays_wikidata():
    ext = ExtKB(WD, LABELS)
    q, _ = ext.find("注射用盐酸多巴酚丁胺的用法用量", "drug")
    assert q == "Q2"                                      # 新别名优先于“酚丁”这种短别名
    assert ext.drug_entry(q)["kb_tier"] == "wikidata"
    assert "多巴酚丁胺" not in ext.names                   # 分词词典（本地全文索引用）不变


def test_without_labels_behaves_as_before():
    ext = ExtKB(WD)
    assert ext.drug_entry("Q1")["dosage"].startswith("扩展知识库（Wikidata）不含剂量")


FORMS = {"drugs": {"Q1": {**LABELS["drugs"]["Q1"],
                          "dosage": "（枸橼酸坦度螺酮胶囊）通常成人每次口服10mg，1日3次。",
                          "source": {"dosage": "Huatuo 百科问答「枸橼酸坦度螺酮胶囊」（verbatim）"},
                          "by_form": {"口服": {"product": "枸橼酸坦度螺酮胶囊", "dosage": "每次口服10mg，1日3次。",
                                               "source": {"dosage": "Huatuo 百科问答「枸橼酸坦度螺酮胶囊」（verbatim）"}},
                                      "注射": {"product": "注射用坦度螺酮", "dosage": "静滴5mg。",
                                               "source": {"dosage": "Huatuo 百科问答「注射用坦度螺酮」（verbatim）"}}}}}}


def test_form_of():
    from medagent.knowledge.drug_label_kb import form_of
    assert form_of("注射用盐酸多巴酚丁胺") == "注射"
    assert form_of("布洛芬缓释胶囊") == "口服"
    assert form_of("奥硝唑栓") == "栓剂"
    assert form_of("坦度螺酮") is None


def test_asked_form_is_served():
    ext = ExtKB(WD, FORMS)
    e = ext.drug_entry("Q1", "注射用坦度螺酮")
    assert "静滴5mg" in e["dosage"] and e["dosage_form"].startswith("注射")


def test_missing_form_gives_no_dose():
    ext = ExtKB(WD, {"drugs": {"Q1": {k: v for k, v in FORMS["drugs"]["Q1"].items() if k != "by_form"}}})
    e = ext.drug_entry("Q1", "坦度螺酮注射液")
    assert "知识库没有注射剂型" in e["dosage"] and "10mg" not in e["dosage"]


def test_unspecified_form_lists_all_forms():
    ext = ExtKB(WD, FORMS)
    e = ext.drug_entry("Q1", "坦度螺酮")
    assert len(e["dosage_by_form"]) == 2 and "剂型" in e["form_note"]


def test_wiki_build_links_and_lint(tmp_path):
    import json

    from medagent.knowledge import wiki
    wd = {"meta": {}, "drugs": {**WD["drugs"], "Q9": {"zh": "西咪替丁", "en": "cimetidine", "atc": ["A02BA01"], "desc": "",
                                                     "aliases": [], "treats": ["胃溃疡"]}},
          "diseases": {"D1": {"zh": "胃溃疡", "icd10": ["K25"], "desc": "", "aliases": [], "symptoms": [], "drugs": ["西咪替丁"]}}}
    kb = {"meta": {"built": "2026-11-19"}, "drugs": {
        "Q1": {**FORMS["drugs"]["Q1"], "interactions_text": "与西咪替丁合用需谨慎。",
               "source": {**FORMS["drugs"]["Q1"]["source"], "interactions_text": "openFDA 说明书 set_id=x"}},
        "Q9": {"name": "西咪替丁", "generic_name": "cimetidine", "atc": ["A02BA01"], "kb_tier": "label", "aliases": [],
               "new_aliases": {}, "indications": "用于胃溃疡", "source": {"indications": "Huatuo 百科问答「西咪替丁片」（verbatim）"}}}}
    (tmp_path / "kb.json").write_text(json.dumps(kb, ensure_ascii=False), encoding="utf-8")
    b = wiki.Builder(kb, wd)
    page = b.drug("Q1")
    assert "### 注射（注射用坦度螺酮）" in page and "](../drugs/西咪替丁.md)" in page and "来源：" in page
    assert b.dis_page == {"D1": "diseases/胃溃疡.md"}


def test_drug_name_with_form():
    from medagent.agent import drug_name_with_form as f
    assert f("尼古丁", "尼古丁贴剂的用法用量") == "尼古丁贴剂"
    assert f("多巴酚丁胺", "注射用盐酸多巴酚丁胺的用法用量") == "注射用盐酸多巴酚丁胺"
    assert f("阿莫西林", "阿莫西林和布洛芬能一起吃吗") == "阿莫西林"          # 原话没有剂型：不动
    assert f("布洛芬缓释胶囊", "布洛芬缓释胶囊一次吃几粒") == "布洛芬缓释胶囊"   # 已带剂型：不动
