#!/usr/bin/env python3
"""
构建 Agent 工具调用训练数据（ShareGPT 格式，含 function_call/observation 角色）。

两部分（混合生成）：
  A. 真实数据改写（50%，~1500 条）：从真实医疗问答中匹配症状/药品/疾病主题，
     用规则 + 内置知识库生成工具调用轨迹，最终回答保留真实答案。
  B. DeepSeek 生成（50%，~1500 条）：调用 DeepSeek API 批量生成（需 OPENAI_API_KEY）。

用法：
  python3 build_toolcall_data.py --raw ../data/sft/_raw_medical_qa.jsonl --mode rewrite
  OPENAI_API_KEY=sk-xxx OPENAI_BASE_URL=https://api.deepseek.com \
  python3 build_toolcall_data.py --raw ../data/sft/_raw_medical_qa.jsonl --mode both --model deepseek-chat
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

# ============================================================
# 内置知识库（供生成 observation）
# ============================================================

SYMPTOM_DB = {
    "头痛": {"possible_causes": ["偏头痛", "紧张性头痛", "高血压"], "recommendation": "建议休息观察，如持续超过3天或加重请就医", "urgency": "medium", "confidence": 0.65},
    "发烧": {"possible_causes": ["流行性感冒", "病毒性感染", "细菌性感染"], "recommendation": "多饮水休息，体温超过38.5℃可遵医嘱退烧，持续高热需就医", "urgency": "medium", "confidence": 0.7},
    "咳嗽": {"possible_causes": ["上呼吸道感染", "支气管炎", "过敏性咳嗽"], "recommendation": "注意休息多饮水，持续超过2周请就医行胸片检查", "urgency": "low", "confidence": 0.55},
    "腹泻": {"possible_causes": ["急性胃肠炎", "消化不良", "食物不耐受"], "recommendation": "注意补充水分和电解质，频繁腹泻或便血请及时就医", "urgency": "medium", "confidence": 0.6},
    "胃痛": {"possible_causes": ["胃炎", "消化性溃疡", "功能性消化不良"], "recommendation": "避免刺激性食物，规律饮食，持续疼痛或黑便请就医", "urgency": "medium", "confidence": 0.55},
    "失眠": {"possible_causes": ["焦虑", "压力过大", "睡眠习惯不良"], "recommendation": "保持规律作息，睡前放松，长期失眠请就医", "urgency": "low", "confidence": 0.5},
    "头晕": {"possible_causes": ["贫血", "低血压", "颈椎病", "耳石症"], "recommendation": "注意休息避免突然起身，频繁头晕请就医", "urgency": "medium", "confidence": 0.55},
    "皮疹": {"possible_causes": ["过敏", "湿疹", "接触性皮炎"], "recommendation": "避免抓挠和接触可疑过敏原，加重请就医", "urgency": "medium", "confidence": 0.55},
    "关节痛": {"possible_causes": ["骨关节炎", "类风湿关节炎", "痛风"], "recommendation": "避免过度活动，可就医行风湿免疫相关检查", "urgency": "low", "confidence": 0.5},
    "胸痛": {"possible_causes": ["心绞痛", "心肌梗死", "胸膜炎"], "recommendation": "胸痛可能为急症，请立即就医或拨打120", "urgency": "emergency", "confidence": 0.75},
    "呼吸困难": {"possible_causes": ["哮喘发作", "心力衰竭", "肺栓塞"], "recommendation": "呼吸困难为急症，请立即拨打120", "urgency": "emergency", "confidence": 0.8},
    "咽喉痛": {"possible_causes": ["急性咽炎", "扁桃体炎", "上呼吸道感染"], "recommendation": "多饮水，避免辛辣刺激，持续加重请就医", "urgency": "low", "confidence": 0.6},
    "鼻塞": {"possible_causes": ["感冒", "过敏性鼻炎", "鼻窦炎"], "recommendation": "注意保暖休息，可用生理盐水清洗鼻腔，持续不缓解请就医", "urgency": "low", "confidence": 0.6},
    "乏力": {"possible_causes": ["贫血", "甲状腺功能减退", "慢性疲劳"], "recommendation": "保证睡眠和营养，持续乏力请就医检查", "urgency": "low", "confidence": 0.5},
    "呕吐": {"possible_causes": ["急性胃肠炎", "食物中毒", "妊娠反应"], "recommendation": "注意补液，频繁呕吐或脱水请及时就医", "urgency": "medium", "confidence": 0.55},
    "腹痛": {"possible_causes": ["急性胃肠炎", "肠痉挛", "消化性溃疡"], "recommendation": "避免刺激性食物，持续剧痛或伴发热请及时就医", "urgency": "medium", "confidence": 0.55},
    "腰痛": {"possible_causes": ["腰肌劳损", "腰椎间盘突出", "肾结石"], "recommendation": "注意休息避免久坐弯腰，持续疼痛请就医", "urgency": "low", "confidence": 0.5},
    "耳鸣": {"possible_causes": ["神经性耳鸣", "中耳炎", "梅尼埃病"], "recommendation": "避免噪音刺激，持续耳鸣请就医耳鼻喉科", "urgency": "low", "confidence": 0.5},
    "便秘": {"possible_causes": ["饮食纤维不足", "肠道功能紊乱", "药物影响"], "recommendation": "增加蔬果和饮水，长期便秘请就医", "urgency": "low", "confidence": 0.55},
    "脱发": {"possible_causes": ["脂溢性脱发", "精神压力", "营养缺乏"], "recommendation": "规律作息减轻压力，严重脱发请就医皮肤科", "urgency": "low", "confidence": 0.5},
}

DRUG_DB = {
    "布洛芬": {"name": "布洛芬", "generic_name": "Ibuprofen", "indications": "缓解轻至中度疼痛，如头痛、关节痛、牙痛、痛经", "contraindications": "对本品过敏者禁用，孕妇禁用，严重肝肾功能不全者禁用", "dosage": "成人一次1片（0.2g），一日3次，饭后服用", "side_effects": "胃肠道不适、恶心、头晕、皮疹", "interactions": ["阿司匹林", "华法林", "甲氨蝶呤"], "pregnancy_category": "C"},
    "对乙酰氨基酚": {"name": "对乙酰氨基酚", "generic_name": "Paracetamol", "indications": "解热镇痛，用于感冒发热、头痛、关节痛等", "contraindications": "严重肝肾功能不全者禁用，对本品过敏者禁用", "dosage": "成人一次0.5g，一日不超过2g", "side_effects": "偶见皮疹、恶心，过量可致肝损伤", "interactions": ["华法林"], "pregnancy_category": "B"},
    "阿司匹林": {"name": "阿司匹林", "generic_name": "Aspirin", "indications": "解热镇痛、抗血小板聚集", "contraindications": "活动性消化道溃疡出血者禁用，对本品过敏者禁用，儿童病毒感染时禁用", "dosage": "遵医嘱，抗血小板常用75-100mg/日", "side_effects": "胃肠道刺激、出血风险、过敏反应", "interactions": ["布洛芬", "华法林", "甲氨蝶呤"], "pregnancy_category": "D"},
    "阿莫西林": {"name": "阿莫西林", "generic_name": "Amoxicillin", "indications": "敏感菌所致的呼吸道、泌尿道等感染", "contraindications": "青霉素过敏者禁用", "dosage": "遵医嘱，成人常用0.5g，一日3次", "side_effects": "过敏反应、胃肠道不适、皮疹", "interactions": ["华法林"], "pregnancy_category": "B"},
    "二甲双胍": {"name": "二甲双胍", "generic_name": "Metformin", "indications": "2型糖尿病一线降糖药", "contraindications": "严重肾功能不全、酮症酸中毒、缺氧性疾病禁用", "dosage": "遵医嘱，成人起始0.5g，一日2次", "side_effects": "胃肠道反应、乳酸酸中毒（罕见）", "interactions": [], "pregnancy_category": "B"},
    "奥美拉唑": {"name": "奥美拉唑", "generic_name": "Omeprazole", "indications": "胃酸相关性疾病，如胃溃疡、反流性食管炎", "contraindications": "对本品过敏者禁用", "dosage": "遵医嘱，成人常用20mg，一日1次", "side_effects": "头痛、腹泻、恶心", "interactions": ["华法林"], "pregnancy_category": "C"},
    "氯雷他定": {"name": "氯雷他定", "generic_name": "Loratadine", "indications": "过敏性鼻炎、慢性荨麻疹等过敏性疾病", "contraindications": "对本品过敏者禁用", "dosage": "成人一次10mg，一日1次", "side_effects": "嗜睡、口干、乏力", "interactions": [], "pregnancy_category": "B"},
    "氨氯地平": {"name": "氨氯地平", "generic_name": "Amlodipine", "indications": "高血压、心绞痛", "contraindications": "对本品过敏者禁用，严重低血压者禁用", "dosage": "遵医嘱，成人常用5mg，一日1次", "side_effects": "踝部水肿、头痛、面部潮红", "interactions": [], "pregnancy_category": "C"},
    "头孢": {"name": "头孢类抗生素", "generic_name": "Cephalosporin", "indications": "敏感菌所致的呼吸道、泌尿道等感染", "contraindications": "头孢菌素过敏者禁用", "dosage": "遵医嘱", "side_effects": "过敏反应、胃肠道反应", "interactions": [], "pregnancy_category": "B"},
    "甲硝唑": {"name": "甲硝唑", "generic_name": "Metronidazole", "indications": "厌氧菌感染、滴虫病、阿米巴病", "contraindications": "对本品过敏者、妊娠早期禁用", "dosage": "遵医嘱", "side_effects": "恶心、口腔金属味", "interactions": [], "pregnancy_category": "B"},
}

GUIDELINE_DB = {
    "高血压": {"topic": "高血压", "content": "低盐饮食（每日食盐<5g）、控制体重、戒烟限酒、增加蔬果和钾摄入、减少饱和脂肪；规律服药并监测血压。", "source": "《中国高血压防治指南》", "year": 2023},
    "糖尿病": {"topic": "2型糖尿病", "content": "二甲双胍为一线首选降糖药；合并动脉粥样硬化性心血管疾病或高风险者可优先选择SGLT2抑制剂或GLP-1受体激动剂。", "source": "《中国2型糖尿病防治指南》", "year": 2020},
    "高血脂": {"topic": "高血脂", "content": "生活方式干预（低脂饮食、运动、控制体重）为基础；他汀类药物为降脂首选。", "source": "《中国血脂管理指南》", "year": 2023},
    "痛风": {"topic": "痛风", "content": "限制高嘌呤食物（动物内脏、海鲜、啤酒），多饮水，急性期使用非甾体抗炎药或秋水仙碱，缓解期降尿酸治疗。", "source": "《中国高尿酸血症与痛风诊疗指南》", "year": 2019},
    "感冒": {"topic": "普通感冒", "content": "多为病毒感染，具有自限性，一般1-2周自愈；多饮水休息，对症处理，无需抗生素。", "source": "《普通感冒规范诊治专家共识》", "year": 2012},
    "湿疹": {"topic": "湿疹", "content": "避免接触刺激物和过敏原，注意皮肤保湿，可外用糖皮质激素；严重或反复发作请就医皮肤科。", "source": "《湿疹诊疗指南》", "year": 2020},
    "脂肪肝": {"topic": "脂肪肝", "content": "控制饮食、增加运动、减轻体重是基础；戒酒，控制血脂血糖。", "source": "《非酒精性脂肪性肝病防治指南》", "year": 2018},
    "胃炎": {"topic": "慢性胃炎", "content": "规律饮食，避免辛辣刺激和酒精，必要时使用抑酸药和胃黏膜保护剂。", "source": "《慢性胃炎诊疗指南》", "year": 2022},
    "冠心病": {"topic": "冠心病", "content": "控制血压血脂血糖、戒烟、抗血小板和他汀治疗为基础，胸痛发作及时就医。", "source": "《冠心病防治指南》", "year": 2023},
}

# 工具 schema
TOOL_SCHEMAS = {
    "check_symptom": {"name": "check_symptom", "description": "根据患者描述的症状分析可能原因，提供初步建议", "parameters": {"type": "object", "properties": {"symptom": {"type": "string", "description": "患者描述的症状"}, "duration": {"type": "string", "description": "症状持续时间"}}, "required": ["symptom"]}},
    "search_drug": {"name": "search_drug", "description": "查询药品的适应症、禁忌症、用法用量、副作用等信息", "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "药品名称"}}, "required": ["name"]}},
    "check_interaction": {"name": "check_interaction", "description": "检查多种药品之间是否存在相互作用", "parameters": {"type": "object", "properties": {"drugs": {"type": "array", "items": {"type": "string"}, "description": "药品名称列表"}}, "required": ["drugs"]}},
    "search_guideline": {"name": "search_guideline", "description": "检索临床指南中关于某疾病或主题的推荐意见", "parameters": {"type": "object", "properties": {"topic": {"type": "string", "description": "疾病或主题"}, "aspect": {"type": "string", "description": "关注的方面"}}, "required": ["topic"]}},
}

# 疾病主题词表（用于 search_guideline 匹配，覆盖真实数据中的高频疾病）
DISEASE_TOPICS = [
    "高血压", "糖尿病", "高血脂", "痛风", "冠心病", "哮喘", "胃炎", "胃溃疡",
    "乙肝", "脂肪肝", "肾病", "贫血", "甲亢", "甲减", "骨质疏松", "抑郁症",
    "焦虑症", "鼻炎", "咽炎", "前列腺", "月经不调", "更年期", "肺炎", "支气管炎",
    "肝炎", "胆囊炎", "胰腺炎", "阑尾炎", "肾炎", "膀胱炎", "尿道炎", "盆腔炎",
    "乳腺增生", "子宫肌瘤", "卵巢囊肿", "脑梗", "心梗", "心绞痛", "心律失常",
    "肺结核", "带状疱疹", "牛皮癣", "白癜风", "荨麻疹", "痤疮", "白内障", "青光眼",
    "鼻窦炎", "扁桃体炎", "癫痫", "手足口病", "肺癌", "肠粘连", "痔疮", "中耳炎",
    "湿疹", "感冒", "流感", "尖锐湿疣", "紫癜", "肿瘤", "癌症", "颈椎病",
    "腰椎间盘突出", "类风湿", "强直性脊柱炎", "失眠", "慢性咽炎", "口腔溃疡",
]

# 症状触发词表（用于 check_symptom 匹配）
SYMPTOM_KEYWORDS = [
    "头痛", "头疼", "头晕", "头昏", "发烧", "发热", "咳嗽", "咳痰", "咽喉痛",
    "嗓子疼", "鼻塞", "流涕", "耳鸣", "耳聋", "心悸", "心慌", "胸闷", "胸痛",
    "气短", "呼吸困难", "腹痛", "肚子疼", "胃痛", "胃胀", "恶心", "呕吐",
    "腹泻", "便秘", "腹胀", "便血", "尿频", "尿急", "尿痛", "血尿", "腰痛",
    "腰疼", "关节痛", "腿疼", "脚疼", "乏力", "疲劳", "失眠", "多梦", "脱发",
    "皮疹", "瘙痒", "水肿", "消瘦", "肥胖", "多汗", "盗汗", "口渴", "多尿",
    "耳鸣", "健忘", "射精快", "早泄", "阳痿", "月经不调", "痛经", "白带",
    # 单字症状触发词（兜底）
    "疼", "痛", "痒", "肿", "晕", "吐", "泻", "咳", "烧",
]


def load_raw(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def make_toolcall_data(human: str, final_answer: str, tool_name: str, arguments: dict, observation: dict) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": human},
            {"from": "function_call", "value": json.dumps({"name": tool_name, "arguments": arguments}, ensure_ascii=False)},
            {"from": "observation", "value": json.dumps(observation, ensure_ascii=False)},
            {"from": "gpt", "value": final_answer},
        ],
        "tools": json.dumps([TOOL_SCHEMAS[tool_name]], ensure_ascii=False),
    }


def generic_guideline_obs(topic: str) -> dict:
    return {
        "topic": topic,
        "content": "建议前往正规医疗机构明确诊断并制定个体化治疗方案，遵医嘱规范治疗和随访。",
        "source": "临床诊疗共识",
        "year": 2023,
    }


def generic_symptom_obs(symptom: str) -> dict:
    return {
        "symptom": symptom,
        "possible_causes": ["需结合详细病史和检查进一步明确"],
        "recommendation": "建议前往医院就诊，明确病因后进行针对性治疗",
        "urgency": "medium",
        "confidence": 0.4,
    }


def rewrite_one(human: str, answer: str) -> dict | None:
    """把一条真实问答改写成工具调用数据；无法匹配则返回 None。"""
    # 1) 药品：问题/答案提到具体药名（优先）
    for drug_name in DRUG_DB:
        if drug_name in human or drug_name in answer:
            return make_toolcall_data(human, answer, "search_drug", {"name": drug_name}, DRUG_DB[drug_name])

    # 2) 疾病/指南：问题含疾病名（词表 + 正则兜底提取 "X病/X炎/X症/X瘤/X癌"）
    matched_topic = None
    for topic in DISEASE_TOPICS:
        if topic in human:
            matched_topic = topic
            break
    if matched_topic is None:
        m = re.search(r"[\u4e00-\u9fff]{1,6}(?:病|炎|症|瘤|癌)", human)
        if m and m.group(0) not in ("没病", "毛病"):
            matched_topic = m.group(0)
    if matched_topic:
        obs = GUIDELINE_DB.get(matched_topic)
        if obs is None:
            obs = generic_guideline_obs(matched_topic)
        return make_toolcall_data(human, answer, "search_guideline", {"topic": matched_topic}, dict(obs))

    # 3) 症状：问题含症状词
    for sym in SYMPTOM_KEYWORDS:
        if sym in human:
            obs = SYMPTOM_DB.get(sym)
            if obs is None:
                obs = generic_symptom_obs(sym)
            else:
                obs = dict(obs); obs["symptom"] = sym
            return make_toolcall_data(human, answer, "check_symptom", {"symptom": sym}, obs)

    return None


# ============================================================
# 规则生成器（补充改写不足，也用于无 key 兜底）
# ============================================================

SYMPTOM_QUESTION_TPL = [
    "我最近{0}，该怎么办？", "{0}好几天了，一直没好转，是什么原因？",
    "请问{0}严重吗，需要去医院吗？", "最近总是{0}，有点担心，怎么回事？",
]
DRUG_QUESTION_TPL = [
    "{0}有什么副作用？", "{0}的用法用量是多少？", "吃{0}有什么禁忌吗？",
    "{0}和别的药能一起吃吗？", "想了解一下{0}的作用。",
]
GUIDELINE_QUESTION_TPL = [
    "{0}患者平时要注意什么？", "{0}饮食上有什么禁忌？", "{0}怎么预防和调理？",
    "{0}的治疗原则是什么？", "{0}平时该怎么管理？",
]


def classify_tool(name: str) -> str:
    if name == "check_symptom":
        return "symptom"
    if name == "search_drug":
        return "drug"
    return "guideline"


def rule_generate_type(kind: str, n: int) -> list[dict]:
    """按指定类型程序化生成工具调用数据。"""
    out = []
    rng = random.Random(42)
    syms = list(SYMPTOM_DB)
    drugs = list(DRUG_DB)
    topics = list(GUIDELINE_DB)
    while len(out) < n:
        if kind == "symptom":
            sym = rng.choice(syms)
            q = rng.choice(SYMPTOM_QUESTION_TPL).format(sym)
            info = dict(SYMPTOM_DB[sym]); info["symptom"] = sym
            causes = "、".join(info["possible_causes"])
            ans = f"根据症状分析，可能的原因包括：{causes}。{info['recommendation']}。请结合自身情况，如有加重请及时就医。"
            out.append(make_toolcall_data(q, ans, "check_symptom", {"symptom": sym}, info))
        elif kind == "drug":
            name = rng.choice(drugs)
            q = rng.choice(DRUG_QUESTION_TPL).format(name)
            info = DRUG_DB[name]
            ans = f"关于{name}：{info['indications']}。用法用量：{info['dosage']}。注意事项：{info['contraindications']}。用药前请咨询医生或药师。"
            out.append(make_toolcall_data(q, ans, "search_drug", {"name": name}, info))
        else:
            topic = rng.choice(topics)
            q = rng.choice(GUIDELINE_QUESTION_TPL).format(topic)
            info = GUIDELINE_DB[topic]
            ans = f"根据{info['source']}：{info['content']}。具体方案请遵医嘱。"
            out.append(make_toolcall_data(q, ans, "search_guideline", {"topic": topic}, info))
    return out


def rewrite_mode(raw: list[dict], target: int) -> list[dict]:
    """真实改写全部保留，规则生成补充 drug/symptom 到目标总量。"""
    buckets: dict[str, list[dict]] = {"symptom": [], "drug": [], "guideline": []}
    for it in raw:
        convs = it["conversations"]
        human = next((c["value"] for c in convs if c["from"] == "human"), "")
        gpt = next((c["value"] for c in convs if c["from"] == "gpt"), "")
        r = rewrite_one(human, gpt)
        if r:
            fc = json.loads(r["conversations"][1]["value"])
            buckets[classify_tool(fc["name"])].append(r)

    print(f"[改写] 真实改写：symptom={len(buckets['symptom'])} drug={len(buckets['drug'])} guideline={len(buckets['guideline'])}", file=sys.stderr)

    # 真实改写全部保留
    result = list(buckets["guideline"]) + list(buckets["symptom"]) + list(buckets["drug"])

    # 规则补充：优先补齐 drug 与 symptom（真实数据中这两类较少）
    need = target - len(result)
    if need > 0:
        add_drug = need * 2 // 3
        add_symptom = need - add_drug
        print(f"[规则补充] drug +{add_drug} 条, symptom +{add_symptom} 条", file=sys.stderr)
        result.extend(rule_generate_type("drug", add_drug))
        result.extend(rule_generate_type("symptom", add_symptom))
    return result[:target]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../data/raw_medical_qa.jsonl")
    ap.add_argument("--mode", choices=["rewrite", "both"], default="rewrite")
    ap.add_argument("--rewrite-n", type=int, default=1500)
    ap.add_argument("--gen-n", type=int, default=1500)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    root = scripts_dir.parent
    out_dir = root / "data" / "sft"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = Path(args.raw)
    if not raw_path.is_absolute():
        raw_path = scripts_dir / args.raw
    raw = load_raw(raw_path)
    print(f"[加载] 真实问答 {len(raw)} 条", file=sys.stderr)

    rewritten = rewrite_mode(raw, args.rewrite_n)
    buckets: dict[str, list[dict]] = {"symptom": [], "drug": [], "guideline": []}
    for it in rewritten:
        fc = json.loads(it["conversations"][1]["value"])
        name = fc["name"]
        key = "symptom" if name == "check_symptom" else ("drug" if name == "search_drug" else "guideline")
        buckets[key].append(it)

    for key, fname in [("symptom", "toolcall_symptom.jsonl"), ("drug", "toolcall_drug.jsonl"), ("guideline", "toolcall_guideline.jsonl")]:
        with open(out_dir / fname, "w", encoding="utf-8") as f:
            for it in buckets[key]:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"[写入] {fname}: {len(buckets[key])} 条", file=sys.stderr)

    if args.mode == "both":
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from gen_seed_data import API_KEY, TOOL_SPECS, generate_one
        except ImportError as e:
            sys.exit(f"导入失败：{e}")
        if not API_KEY:
            sys.exit("缺少 API key：export DEEPSEEK_API_KEY=sk-xxx")

        # 分配：真实改写中 drug/symptom 少，LLM 生成重点补这两类
        alloc = {"search_drug": 700, "check_symptom": 500, "search_guideline": 300}
        for tool, n in alloc.items():
            seeds = TOOL_SPECS[tool]["seeds"]
            tasks = [(tool, seeds[i % len(seeds)], args.model) for i in range(n)]
            gen: list[dict] = []
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs = [pool.submit(generate_one, t, s, m) for t, s, m in tasks]
                for fut in as_completed(futs):
                    r = fut.result()
                    if r:
                        gen.append(r)
            key = classify_tool(tool)
            out_file = out_dir / f"toolcall_{key}.jsonl"
            with open(out_file, "a", encoding="utf-8") as f:
                for it in gen:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
            print(f"[DeepSeek] {tool} 生成 {len(gen)}/{n} 条，追加到 {out_file.name}", file=sys.stderr, flush=True)

    print("\n完成。请用 validate_data.py 校验：python3 validate_data.py --dir ../data/sft")


if __name__ == "__main__":
    main()
