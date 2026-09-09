#!/usr/bin/env python3
"""
构建 OPD 蒸馏训练数据（1500 条，按比例采样 + 生成高价值样本）

比例（参考 SFT 分布）：
  - Agent 工具调用 70% = 1050 条（多轮 350 + 追问 250 + 单轮 450，多轮+追问 > 单轮）
  - 普通医疗问答 20% = 300 条
  - 边界/复杂场景 10% = 150 条（工具失败恢复 80 + 模糊症状 70）

高价值样本（生成）：
  多轮工具调用 350 条 / 追问澄清 250 条 / 工具失败恢复 80 条 / 模糊症状 70 条
单轮工具调用 450 条 / 普通问答 300 条（从现有数据采样）

关键：每条样本都保留 tools 字段（OPD 会把它组装进 prompt）
"""
import json
import random

random.seed(42)

# ============ 工具定义（复用现有数据格式） ============
TOOLS = {
    "check_symptom": {
        "name": "check_symptom",
        "description": "根据患者描述的症状分析可能原因，提供初步建议",
        "parameters": {
            "type": "object",
            "properties": {
                "symptom": {"type": "string", "description": "患者描述的症状"},
                "duration": {"type": "string", "description": "症状持续时间"},
            },
            "required": ["symptom"],
        },
    },
    "search_drug": {
        "name": "search_drug",
        "description": "查询药品的适应症、禁忌症、用法用量、副作用等信息",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "药品名称"},
            },
            "required": ["name"],
        },
    },
    "search_guideline": {
        "name": "search_guideline",
        "description": "检索临床指南中关于某疾病或主题的推荐意见",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "疾病或主题"},
                "aspect": {"type": "string", "description": "关注的方面"},
            },
            "required": ["topic"],
        },
    },
}


def tools_json(*names):
    return json.dumps([TOOLS[n] for n in names], ensure_ascii=False)


def fc(tool_name, **args):
    return {"from": "function_call", "value": json.dumps({"name": tool_name, "arguments": args}, ensure_ascii=False)}


def obs(value):
    return {"from": "observation", "value": json.dumps(value, ensure_ascii=False)}


# ============ 预定义数据（扩充保证多样性） ============
SYMPTOM_CASES = [
    ("头痛", "3天", ["偏头痛", "紧张性头痛"], "建议休息观察，如持续超过3天请就医", 0.65),
    ("发烧", "1天", ["上呼吸道感染", "流感"], "建议多饮水，监测体温，超过38.5℃可就医", 0.7),
    ("咳嗽", "1周", ["急性支气管炎", "感冒后咳嗽"], "建议就诊，明确病因", 0.6),
    ("腹痛", "2天", ["急性胃炎", "消化不良"], "建议清淡饮食，如疼痛加剧请及时就医", 0.55),
    ("头晕", "3天", ["体位性低血压", "贫血"], "建议测量血压，如持续请就医", 0.5),
    ("关节痛", "1个月", ["骨关节炎", "类风湿关节炎"], "建议风湿免疫科就诊", 0.55),
    ("皮疹", "2天", ["过敏性皮炎", "湿疹"], "避免抓挠，建议皮肤科就诊", 0.6),
    ("腹泻", "1天", ["急性肠炎", "食物不耐受"], "注意补水，如脱水请就医", 0.65),
    ("胸闷", "2天", ["心血管问题", "焦虑"], "建议尽快就医排除心脏问题", 0.6),
    ("乏力", "1周", ["贫血", "甲状腺功能减退"], "建议查血常规及甲状腺功能", 0.5),
    ("咽喉痛", "2天", ["急性咽炎", "扁桃体炎"], "建议多饮水，避免辛辣，如伴高热请就医", 0.6),
    ("失眠", "2周", ["焦虑", "睡眠环境不佳"], "建议规律作息，如持续请就医", 0.5),
    ("胃胀", "3天", ["功能性消化不良", "慢性胃炎"], "建议少食多餐，避免产气食物", 0.55),
    ("耳鸣", "1周", ["噪声性耳鸣", "内耳疾病"], "建议耳鼻喉科就诊", 0.5),
    ("视力模糊", "3天", ["视疲劳", "屈光不正"], "建议眼科检查，排除眼底问题", 0.55),
]

DRUG_CASES = [
    ("布洛芬", "缓解轻至中度疼痛，如头痛、关节痛", "成人一次1片，一日3次，饭后服用", "对本品过敏者禁用，孕妇禁用", "胃肠道不适、头晕"),
    ("对乙酰氨基酚", "缓解发热、头痛", "成人一次1片，一日3次", "肝肾功能不全者慎用", "偶见皮疹"),
    ("阿莫西林", "敏感菌引起的呼吸道、泌尿道感染", "遵医嘱，通常一次0.5g，一日3次", "青霉素过敏者禁用", "过敏反应、胃肠道反应"),
    ("奥美拉唑", "胃酸过多相关疾病，如胃溃疡", "一次20mg，一日1次，晨起服用", "对本品过敏者禁用", "头痛、腹泻"),
    ("氯雷他定", "过敏性鼻炎、荨麻疹", "一次10mg，一日1次", "对本品过敏者禁用", "嗜睡、口干"),
    ("蒙脱石散", "急慢性腹泻", "成人一次1袋，一日3次", "肠梗阻者禁用", "偶见便秘"),
    ("左氧氟沙星", "敏感菌引起的呼吸道、泌尿道感染", "遵医嘱，通常一次0.5g，一日1次", "18岁以下禁用，孕妇禁用", "胃肠道反应、头晕"),
    ("二甲双胍", "2型糖尿病，尤其肥胖者", "遵医嘱，通常一次0.5g，一日2-3次", "严重肾功能不全禁用", "胃肠道反应、乳酸酸中毒"),
    ("阿托伐他汀", "高胆固醇血症、冠心病", "一次10-20mg，一日1次", "活动性肝病禁用，孕妇禁用", "肌痛、肝功能异常"),
    ("布地奈德", "支气管哮喘、过敏性鼻炎", "遵医嘱吸入，剂量个体化", "对本品过敏者禁用", "口咽部念珠菌感染、声音嘶哑"),
]

GUIDELINE_CASES = [
    ("高血压", "生活方式干预", "建议低盐饮食、规律运动、监测血压，必要时药物治疗"),
    ("糖尿病", "血糖管理", "建议控制饮食、规律运动，定期监测血糖，遵医嘱用药"),
    ("哮喘", "长期管理", "建议避免过敏原、规律使用吸入药物、定期随访"),
    ("胃炎", "饮食调理", "建议规律饮食、避免辛辣刺激、必要时药物治疗"),
    ("高血脂", "降脂治疗", "建议低脂饮食、增加运动，必要时他汀类药物治疗"),
    ("冠心病", "二级预防", "建议抗血小板、他汀、控制血压血糖，戒烟限酒"),
    ("痛风", "降尿酸治疗", "建议低嘌呤饮食、多饮水、必要时降尿酸药物"),
    ("慢性阻塞性肺病", "稳定期管理", "建议戒烟、规律吸入支气管扩张剂、肺康复训练"),
]

VAGUE_SYMPTOMS = ["不舒服", "难受", "没精神", "总是累", "感觉不对劲", "浑身不得劲", "有点疼",
                  "说不上来", "老毛病", "时好时坏", "隐隐约约", "不太对劲"]

# ============ 生成函数 ============
def gen_multi_tool(n):
    """多轮工具调用：连续调用 2-3 个工具"""
    out = []
    combos = [
        ("check_symptom", "search_drug"),
        ("check_symptom", "search_guideline"),
        ("search_drug", "search_guideline"),
        ("check_symptom", "search_drug", "search_guideline"),  # 三轮：症状→药品→指南
    ]
    for i in range(n):
        combo = combos[i % len(combos)]
        s_name, s_dur, causes, rec, conf = random.choice(SYMPTOM_CASES)
        d_name, d_ind, d_dos, d_contra, d_side = random.choice(DRUG_CASES)
        g_topic, g_aspect, g_rec = random.choice(GUIDELINE_CASES)

        if combo == ("check_symptom", "search_drug"):
            convs = [
                {"from": "human", "value": f"我{s_name}{s_dur}了，需要吃药吗？"},
                fc("check_symptom", symptom=s_name, duration=s_dur),
                obs({"possible_causes": causes, "recommendation": rec, "confidence": conf}),
                {"from": "gpt", "value": f"根据症状分析，可能是{'、'.join(causes)}。{rec}。如果你想缓解症状，我可以帮你查一下常用药品。"},
                {"from": "human", "value": f"那帮我查一下{d_name}的信息"},
                fc("search_drug", name=d_name),
                obs({"name": d_name, "indications": d_ind, "dosage": d_dos, "contraindications": d_contra, "side_effects": d_side}),
                {"from": "gpt", "value": f"{d_name}可用于{d_ind}，{d_dos}。注意：{d_contra}，可能出现{d_side}。建议在医生指导下使用，如症状未缓解请及时就医。"},
            ]
        elif combo == ("check_symptom", "search_guideline"):
            convs = [
                {"from": "human", "value": f"我最近{s_name}，已经{s_dur}了，想了解一下该怎么处理"},
                fc("check_symptom", symptom=s_name, duration=s_dur),
                obs({"possible_causes": causes, "recommendation": rec, "confidence": conf}),
                {"from": "gpt", "value": f"可能是{'、'.join(causes)}。{rec}。我可以帮你查一下相关的临床指南。"},
                {"from": "human", "value": f"那查一下{g_topic}的指南吧"},
                fc("search_guideline", topic=g_topic, aspect=g_aspect),
                obs({"topic": g_topic, "aspect": g_aspect, "recommendation": g_rec}),
                {"from": "gpt", "value": f"根据{g_topic}的{g_aspect}指南：{g_rec}。请结合自身情况，如有不适及时就医。"},
            ]
        elif combo == ("search_drug", "search_guideline"):
            convs = [
                {"from": "human", "value": f"{d_name}这个药怎么用？我主要是{s_name}的症状"},
                fc("search_drug", name=d_name),
                obs({"name": d_name, "indications": d_ind, "dosage": d_dos, "contraindications": d_contra, "side_effects": d_side}),
                {"from": "gpt", "value": f"{d_name}用于{d_ind}，{d_dos}。注意{d_contra}。另外我帮你查一下相关疾病的治疗指南，更全面。"},
                {"from": "human", "value": "好，那看看相关指南"},
                fc("search_guideline", topic=g_topic, aspect=g_aspect),
                obs({"topic": g_topic, "aspect": g_aspect, "recommendation": g_rec}),
                {"from": "gpt", "value": f"根据{g_topic}指南：{g_rec}。用药建议在医生指导下进行。"},
            ]
        else:  # 三轮：症状 → 药品 → 指南
            convs = [
                {"from": "human", "value": f"我{s_name}{s_dur}了，该吃什么药？"},
                fc("check_symptom", symptom=s_name, duration=s_dur),
                obs({"possible_causes": causes, "recommendation": rec, "confidence": conf}),
                {"from": "gpt", "value": f"可能是{'、'.join(causes)}。{rec}。我帮你查一下常用药{d_name}。"},
                {"from": "human", "value": "好"},
                fc("search_drug", name=d_name),
                obs({"name": d_name, "indications": d_ind, "dosage": d_dos, "contraindications": d_contra, "side_effects": d_side}),
                {"from": "gpt", "value": f"{d_name}可用于{d_ind}，{d_dos}。我再查一下相关疾病的诊疗指南，给你更全面的建议。"},
                {"from": "human", "value": "那看看指南怎么说"},
                fc("search_guideline", topic=g_topic, aspect=g_aspect),
                obs({"topic": g_topic, "aspect": g_aspect, "recommendation": g_rec}),
                {"from": "gpt", "value": f"综合来看，{g_topic}的{g_aspect}建议：{g_rec}。结合你的症状，用药请在医生指导下进行，如未缓解请就医。"},
            ]
        out.append({"conversations": convs, "tools": tools_json(*combo)})
    return out


def gen_clarify(n):
    """追问澄清：症状模糊，模型先问再决定调不调工具"""
    out = []
    ask_templates = [
        "抱歉听到你不太舒服。为了更准确帮你分析，能具体说一下哪里不舒服吗？持续多久了？有没有其他伴随症状？",
        "能具体描述一下吗？比如疼痛的部位、持续时间，有没有伴随发热等其他症状？",
        "我先了解一下情况：具体是哪里不舒服？什么时候开始的？有没有加重或缓解的因素？",
    ]
    for i in range(n):
        s_name, s_dur, causes, rec, conf = random.choice(SYMPTOM_CASES)
        vague = random.choice(VAGUE_SYMPTOMS)
        ask = ask_templates[i % len(ask_templates)]
        convs = [
            {"from": "human", "value": f"我最近有点{vague}"},
            {"from": "gpt", "value": ask},
            {"from": "human", "value": f"主要是{s_name}，大概{s_dur}了"},
            fc("check_symptom", symptom=s_name, duration=s_dur),
            obs({"possible_causes": causes, "recommendation": rec, "confidence": conf}),
            {"from": "gpt", "value": f"根据你的描述（{s_name}{s_dur}），可能是{'、'.join(causes)}。{rec}。如果症状持续或加重，请及时就医。"},
        ]
        out.append({"conversations": convs, "tools": tools_json("check_symptom")})
    return out


def gen_tool_fail(n):
    """工具失败恢复：工具返回错误/空，模型优雅降级"""
    out = []
    for i in range(n):
        if i % 2 == 0:
            bad_drug = random.choice(["不存在的药XYZ", "某种偏方", "名字记不清的药"])
            convs = [
                {"from": "human", "value": f"帮我查一下{bad_drug}的信息"},
                fc("search_drug", name=bad_drug),
                obs({"error": "未找到该药品，请确认药品名称是否正确"}),
                {"from": "gpt", "value": "抱歉，我这边没有查到该药品的信息。建议你确认一下药品的准确名称，或者咨询药师/医生。如果你是想缓解某种症状，可以告诉我具体症状，我帮你查一下常用药。"},
            ]
            out.append({"conversations": convs, "tools": tools_json("search_drug")})
        else:
            bad_topic = random.choice(["很罕见的不明疾病", "不存在的病名"])
            convs = [
                {"from": "human", "value": f"帮我查一下{bad_topic}的诊疗指南"},
                fc("search_guideline", topic=bad_topic),
                obs({"error": "未检索到相关指南"}),
                {"from": "gpt", "value": "抱歉，没有检索到相关临床指南。建议你提供更准确的疾病名称，或直接前往医院由医生评估。如果是常见症状，我也可以帮你分析。"},
            ]
            out.append({"conversations": convs, "tools": tools_json("search_guideline")})
    return out


def gen_vague(n):
    """模糊症状边界：症状描述不清，教师展示谨慎处理"""
    out = []
    for i in range(n):
        vague = random.choice(VAGUE_SYMPTOMS)
        convs = [
            {"from": "human", "value": f"我就是{vague}，怎么办？"},
            {"from": "gpt", "value": "你的描述比较笼统，我暂时无法判断具体原因。建议你留意一下具体症状（比如是否有发热、疼痛部位、持续时间等），如果症状明显或持续不缓解，请及时就医。你可以告诉我更具体的情况，我再帮你分析。"},
        ]
        out.append({"conversations": convs, "tools": tools_json("check_symptom")})
    return out


# ============ 采样函数 ============
def load_and_classify(path):
    single_tool, plain_qa = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            has_fc = any(c["from"] == "function_call" for c in d["conversations"])
            if has_fc:
                single_tool.append(d)
            else:
                plain_qa.append(d)
    return single_tool, plain_qa


def sample_plain_qa(plain_qa, n):
    """采样普通问答，优先回答较长的（信息量大，剔除过简问题）"""
    scored = []
    for d in plain_qa:
        gpt_val = next((c["value"] for c in d["conversations"] if c["from"] == "gpt"), "")
        scored.append((len(gpt_val), d))
    scored.sort(key=lambda x: -x[0])
    selected = [d for _, d in scored[:n]]
    random.shuffle(selected)
    return selected


def sample_single_tool(single_tool, n):
    """采样单轮工具调用"""
    return random.sample(single_tool, n)


def main():
    src = "/root/autodl-tmp/MedicalGPT/data/sft/all_sft.jsonl"
    single_tool, plain_qa = load_and_classify(src)
    print(f"现有数据: 单轮工具调用 {len(single_tool)}, 普通问答 {len(plain_qa)}")

    # 采样（多轮+追问 > 单轮）
    sel_plain = sample_plain_qa(plain_qa, 300)
    sel_single = sample_single_tool(single_tool, 450)

    # 生成高价值样本
    gen_multi = gen_multi_tool(350)
    gen_clar = gen_clarify(250)
    gen_fail = gen_tool_fail(80)
    gen_vag = gen_vague(70)

    # 组装
    data = []
    data.extend(gen_multi)     # 350 多轮
    data.extend(gen_clar)      # 250 追问
    data.extend(sel_single)    # 450 单轮
    data.extend(sel_plain)     # 300 普通问答
    data.extend(gen_fail)      # 80 工具失败
    data.extend(gen_vag)       # 70 模糊症状

    random.shuffle(data)

    n_high = len(gen_multi) + len(gen_clar)
    print(f"生成结果: 总计 {len(data)} 条")
    print(f"  多轮工具调用: {len(gen_multi)}")
    print(f"  追问澄清: {len(gen_clar)}")
    print(f"  单轮工具调用: {len(sel_single)}")
    print(f"  普通问答: {len(sel_plain)}")
    print(f"  工具失败恢复: {len(gen_fail)}")
    print(f"  模糊症状边界: {len(gen_vag)}")
    print(f"  多轮+追问({n_high}) vs 单轮({len(sel_single)}): {'多轮+追问 多于 单轮' if n_high > len(sel_single) else '单轮 多于 多轮+追问'}")
    print(f"  工具调用类占比: {(n_high+len(sel_single))/len(data):.1%}")
    print(f"  普通问答占比: {len(sel_plain)/len(data):.1%}")
    print(f"  边界场景占比: {(len(gen_fail)+len(gen_vag))/len(data):.1%}")

    import os
    os.makedirs("/root/autodl-tmp/MedicalGPT/data/opd", exist_ok=True)
    out_path = "/root/autodl-tmp/MedicalGPT/data/opd/opd_data.jsonl"
    n_with_tools = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for d in data:
            if d.get("tools"):
                n_with_tools += 1
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\n保存到 {out_path}")
    print(f"带 tools 字段的样本: {n_with_tools}/{len(data)}")


if __name__ == "__main__":
    main()
