#!/usr/bin/env python3
"""
构建 DPO 偏好训练数据（MedicalGPT DPO 格式，chosen vs rejected）。

5 类场景与比例（总计 1500 条）：
  工具调用场景 70%（1050 条）：
    1. 正确调用工具 vs 直接凭知识回答   —— 600 条
    2. 正确调用工具 vs 调用错误工具      —— 300 条
    3. 正确调用工具 vs 参数错误          —— 150 条
  普通问答场景 30%（450 条）：
    4. 直接准确回答 vs 画蛇添足调用工具   —— 300 条
    5. 礼貌拒绝 vs 胡说八道              —— 150 条

输出（MedicalGPT DPO 格式，chosen/rejected 为模型输出文本）：
  {"conversations": [{"from": "human", "value": "..."}], "tools": "[...]",
   "chosen": "Action: check_symptom\nAction Input: {...}", "rejected": "..."}

用法：
  python3 build_dpo_data.py            # 全量生成 1500 条
  python3 build_dpo_data.py --total 1500
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# 复用 build_toolcall_data 的知识库与工具 schema
import build_toolcall_data as btd

SYMPTOM_DB = btd.SYMPTOM_DB
DRUG_DB = btd.DRUG_DB
GUIDELINE_DB = btd.GUIDELINE_DB
TOOL_SCHEMAS = btd.TOOL_SCHEMAS

SYMPTOMS = list(SYMPTOM_DB)
DRUGS = list(DRUG_DB)
TOPICS = list(GUIDELINE_DB)


def action_text(tool: str, args: dict) -> str:
    return f"Action: {tool}\nAction Input: {json.dumps(args, ensure_ascii=False)}"


def make_dpo(human: str, tool_schemas: list[dict], chosen: str, rejected: str) -> dict:
    return {
        "conversations": [{"from": "human", "value": human}],
        "tools": json.dumps(tool_schemas, ensure_ascii=False),
        "chosen": chosen,
        "rejected": rejected,
    }


def symptom_tools() -> list[dict]:
    return [TOOL_SCHEMAS["check_symptom"]]


def drug_tools() -> list[dict]:
    return [TOOL_SCHEMAS["search_drug"]]


def symptom_drug_tools() -> list[dict]:
    return [TOOL_SCHEMAS["check_symptom"], TOOL_SCHEMAS["search_drug"]]


# ------------------------------------------------------------
# 类型 1：正确调用工具 vs 直接凭知识回答（600）
# ------------------------------------------------------------
SYMPTOM_Q = [
    "我{0}，该怎么办？",
    "最近一直{0}，好几天了，吃什么药？",
    "{0}持续3天了，有点担心，怎么办？",
    "请问{0}是怎么回事，需要吃药吗？",
]
# 症状 → 对症药（让 rejected 的"直接回答"看似合理）
SYMPTOM_TO_DRUG = {
    "头痛": "布洛芬", "发烧": "对乙酰氨基酚", "关节痛": "布洛芬",
    "胃痛": "奥美拉唑", "腹痛": "奥美拉唑", "皮疹": "氯雷他定",
    "腰痛": "布洛芬",
}
GENERIC_DIRECT = [
    "多喝水多休息，应该能自己缓解。",
    "注意休息，别太劳累，观察几天看看。",
    "问题不大，休息两天应该就好了。",
]
DRUG_DIRECT = [
    "你可以吃{1}缓解一下，多喝水多休息。",
    "建议吃点{1}，一般一天2-3次，应该能缓解。",
]


def gen_type1(n: int) -> list[dict]:
    rng = random.Random(1)
    out = []
    while len(out) < n:
        sym = rng.choice(SYMPTOMS)
        human = rng.choice(SYMPTOM_Q).format(sym)
        chosen = action_text("check_symptom", {"symptom": sym, "duration": "3天"})
        drug = SYMPTOM_TO_DRUG.get(sym)
        if drug:
            rejected = rng.choice(DRUG_DIRECT).format(sym, drug)
        else:
            rejected = rng.choice(GENERIC_DIRECT)
        out.append(make_dpo(human, symptom_drug_tools(), chosen, rejected))
    return out


# ------------------------------------------------------------
# 类型 2：正确调用工具 vs 调用错误工具（300）
# ------------------------------------------------------------
def gen_type2(n: int) -> list[dict]:
    rng = random.Random(2)
    out = []
    while len(out) < n:
        sym = rng.choice(SYMPTOMS)
        human = rng.choice(SYMPTOM_Q).format(sym)
        chosen = action_text("check_symptom", {"symptom": sym, "duration": "3天"})
        # 错误工具：把症状当药名去查药，或该查症状却查指南
        if rng.random() < 0.5:
            rejected = action_text("search_drug", {"name": sym})
        else:
            rejected = action_text("search_guideline", {"topic": sym})
        out.append(make_dpo(human, [TOOL_SCHEMAS["check_symptom"], TOOL_SCHEMAS["search_drug"], TOOL_SCHEMAS["search_guideline"]], chosen, rejected))
    return out


# ------------------------------------------------------------
# 类型 3：正确调用工具 vs 参数错误（150）
# ------------------------------------------------------------
def gen_type3(n: int) -> list[dict]:
    rng = random.Random(3)
    out = []
    while len(out) < n:
        sym = rng.choice(SYMPTOMS)
        human = rng.choice(SYMPTOM_Q).format(sym)
        # 正确：完整参数
        chosen = action_text("check_symptom", {"symptom": sym, "duration": "3天"})
        # 参数错误：缺 duration，或 symptom 提取过粗
        if rng.random() < 0.6:
            rejected = action_text("check_symptom", {"symptom": sym})
        else:
            rejected = action_text("check_symptom", {"symptom": "不舒服"})
        out.append(make_dpo(human, symptom_tools(), chosen, rejected))
    return out


# ------------------------------------------------------------
# 类型 4：直接准确回答 vs 画蛇添足调用工具（300）
# ------------------------------------------------------------
CONCEPT_Q = [
    "什么是{0}？",
    "{0}是什么意思？",
    "能解释一下{0}吗？",
    "{0}的定义是什么？",
]


def gen_type4(n: int) -> list[dict]:
    rng = random.Random(4)
    out = []
    concept_answers = {
        "高血压": "高血压是指血液在血管中流动时对血管壁造成的压力持续高于正常值的状态，通常指诊室血压收缩压≥140mmHg和/或舒张压≥90mmHg。长期高血压会损害心、脑、肾等靶器官。",
        "糖尿病": "糖尿病是一种以慢性高血糖为特征的代谢性疾病，主要分为1型、2型等类型。2型糖尿病最常见，与胰岛素抵抗和分泌不足有关。",
        "感冒": "普通感冒是由病毒引起的上呼吸道感染，具有自限性，主要表现为鼻塞、流涕、咳嗽、咽痛等，一般1-2周可自愈。",
        "痛风": "痛风是尿酸盐结晶沉积在关节引起的炎症性疾病，与高尿酸血症相关，常表现为关节红肿热痛，尤其是大脚趾。",
        "高血脂": "高血脂是指血液中胆固醇或甘油三酯水平升高，是动脉粥样硬化和心血管疾病的重要危险因素。",
    }
    while len(out) < n:
        topic = rng.choice(TOPICS)
        human = rng.choice(CONCEPT_Q).format(topic)
        chosen = concept_answers.get(topic, f"{topic}是一种常见的健康问题，具体定义需要结合医学标准，建议咨询专业医生获取准确解释。")
        # 画蛇添足：概念问题却调用工具
        rejected = action_text("check_symptom", {"symptom": topic})
        out.append(make_dpo(human, symptom_drug_tools(), chosen, rejected))
    return out


# ------------------------------------------------------------
# 类型 5：礼貌拒绝 vs 胡说八道（150）
# ------------------------------------------------------------
REQUEST_Q = [
    "帮我开个处方，我最近{0}。",
    "我是不是得了{1}？帮我诊断一下。",
    "给我开点药，我{0}好几天了。",
    "能不能帮我确诊一下，我怀疑是{1}。",
]
POLITE_REFUSE = [
    "抱歉，我不能为您开具处方或做出诊断。处方和诊断需要执业医师结合您的具体检查结果判断，建议您前往正规医院就诊。",
    "很抱歉，作为AI助手我无法提供诊断或处方。您描述的情况需要专业医生评估，请尽快就医。",
    "抱歉，涉及具体诊断和用药需要执业医师面诊后决定，我不能代替医生。建议您到医院相关科室就诊。",
]
NONSENSE = [
    "根据你的描述，你这明显是{1}，吃{2}就行，一天三次，不用去医院。",
    "你这是{1}没跑了，直接吃{2}，很快就能好。",
    "我确定你是{1}，赶紧吃{2}，别的都不用管。",
]


def gen_type5(n: int) -> list[dict]:
    rng = random.Random(5)
    out = []
    while len(out) < n:
        sym = rng.choice(SYMPTOMS)
        topic = rng.choice(TOPICS)
        drug = rng.choice(DRUGS)
        human = rng.choice(REQUEST_Q).format(sym, topic)
        chosen = rng.choice(POLITE_REFUSE)
        rejected = rng.choice(NONSENSE).format(sym, topic, drug)
        out.append(make_dpo(human, symptom_drug_tools(), chosen, rejected))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=1500)
    args = ap.parse_args()

    # 按比例分配
    n1 = round(args.total * 0.40)  # 600 / 1500
    n2 = round(args.total * 0.20)  # 300
    n3 = round(args.total * 0.10)  # 150
    n4 = round(args.total * 0.20)  # 300
    n5 = round(args.total * 0.10)  # 150

    scripts_dir = Path(__file__).resolve().parent
    reward_dir = scripts_dir.parent / "data" / "reward"
    reward_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DPO] 生成：类型1={n1} 类型2={n2} 类型3={n3} 类型4={n4} 类型5={n5}", file=sys.stderr)

    toolcall_dpo = gen_type1(n1) + gen_type2(n2) + gen_type3(n3)
    medical_dpo = gen_type4(n4) + gen_type5(n5)

    with open(reward_dir / "toolcall_dpo.jsonl", "w", encoding="utf-8") as f:
        for it in toolcall_dpo:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(reward_dir / "medical_dpo.jsonl", "w", encoding="utf-8") as f:
        for it in medical_dpo:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"[写入] toolcall_dpo.jsonl: {len(toolcall_dpo)} 条（工具调用场景）", file=sys.stderr)
    print(f"[写入] medical_dpo.jsonl: {len(medical_dpo)} 条（普通问答场景）", file=sys.stderr)
    print(f"[合计] {len(toolcall_dpo) + len(medical_dpo)} 条 DPO 数据", file=sys.stderr)


if __name__ == "__main__":
    main()
