#!/usr/bin/env python3
"""v3 数据补充：解决 v2 去重后“条数太少”的问题。

来源与做法（每类都写入 meta.source，配比由 training/configs/data_mix_v3.yaml 控制）：
  real_qa        真实医疗问答：Huatuo 百科问答 / Huatuo26M-Lite / shibing624/medical（均 Apache-2.0），质量过滤 + 去重 + 去污染
  toolcall_ext   工具调用轨迹：observation 是**真实调用本项目工具**（精选 KB + Wikidata 实体层）得到的返回，
                 问法与回答各有多种句式；Wikidata 条目没有剂量时，回答明确说“请查说明书”，不编
  safety         急症 / 自伤危机 / 越权请求：主体 × 红旗征描述 × 语境 组合生成，回复多版本；与所有评测集做 n-gram 去污染
  clarify        信息不足 → 结构化追问：部位 × 模糊描述 组合
  general        通用指令回放：alpaca-zh（CC-BY-4.0），防灾难性遗忘
  dpo_real       真实偏好对：shibing624/medical reward（3800 对）
  dpo_safety     安全偏好对：chosen=急救/拒开方，rejected=“先观察看看”/直接给剂量
反事实（槽位遮挡 / 观察遮挡 / 实体遮挡 / 口语化）由 augment.py 在 toolcall_ext 上继续生成。
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from common import REPO, MinHashLSH, human, norm, read_jsonl, shingles, write_jsonl

from medagent.knowledge import load_kb
from medagent.knowledge.ext import load_ext
from medagent.rag.text import ngram_set

CORPUS = REPO / ".cache" / "corpus"
OUT = REPO / "training" / "data" / "supplement"
AD = re.compile(r"https?://|www\.|微信|QQ|qq群|电话|咨询热线|加我|关注公众号|点击|优惠|特价|￥|\d{11}")
TOOLS = {
    "check_symptom": {"name": "check_symptom", "description": "根据患者描述的症状分析可能原因与紧急程度，提供初步建议",
                      "parameters": {"type": "object", "properties": {"symptom": {"type": "string"}, "duration": {"type": "string"}},
                                     "required": ["symptom"]}},
    "search_drug": {"name": "search_drug", "description": "查询药品的适应症、禁忌、用法用量、不良反应、相互作用",
                    "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    "search_guideline": {"name": "search_guideline", "description": "检索临床指南中关于某疾病的推荐意见",
                         "parameters": {"type": "object", "properties": {"topic": {"type": "string"}, "aspect": {"type": "string"}},
                                        "required": ["topic"]}},
    "check_interaction": {"name": "check_interaction", "description": "检查两种及以上药物同时使用的相互作用",
                          "parameters": {"type": "object", "properties": {"drugs": {"type": "array", "items": {"type": "string"}}},
                                         "required": ["drugs"]}},
}
DISCLAIMERS = ["以上信息仅供参考，具体请遵医嘱。", "以上仅为科普信息，不能替代医生面诊。", "如症状持续或加重，请及时就医。",
               "用药前请仔细阅读说明书，或咨询医生、药师。", "每个人情况不同，请以医生的判断为准。"]


def eval_texts() -> set[str]:
    """所有评测集的查询（dev / A / B / C / RAG qrels / 旧 eval_* / Huatuo 百科测试集），用于去污染。"""
    texts: set[str] = set()
    ds = REPO / "evaluation" / "datasets"
    for p in ds.glob("*.jsonl"):
        for r in read_jsonl(p):
            if "query" in r:
                texts.add(r["query"])
    for p in (REPO / "training" / "data" / "sft" / "eval_sft.jsonl", REPO / "training" / "data" / "reward" / "eval_dpo.jsonl",
              REPO / "training" / "data" / "opd" / "eval_opd.jsonl"):
        texts |= {human(r) for r in read_jsonl(p)}
    test = CORPUS / "FreedomIntelligence__huatuo_encyclopedia_qa" / "test_datasets.jsonl"
    if test.exists():
        for r in read_jsonl(test):
            q = r["questions"][0]
            texts.add(q[0] if isinstance(q, list) else q)
    return texts


class Decontam:
    """与评测集去污染：完全相同，或 3-gram Jaccard ≥ thr 都算污染。倒排索引加速。"""

    def __init__(self, texts: set[str], thr: float = 0.6) -> None:
        from collections import defaultdict
        self.exact = {norm(t) for t in texts}
        self.grams = [ngram_set(t, 3) for t in texts]
        self.inv: dict[str, list[int]] = defaultdict(list)
        for i, g in enumerate(self.grams):
            for x in g:
                self.inv[x].append(i)
        self.thr = thr
        self.hits = 0

    def bad(self, q: str) -> bool:
        from collections import Counter
        if norm(q) in self.exact:
            self.hits += 1
            return True
        g = ngram_set(q, 3)
        cnt = Counter(i for x in g for i in self.inv.get(x, ()))
        for i, inter in cnt.items():
            if inter / max(len(g | self.grams[i]), 1) >= self.thr:
                self.hits += 1
                return True
        return False


def conv(*turns: tuple[str, str], tools: list[str] | None = None, meta: dict | None = None) -> dict:
    row = {"conversations": [{"from": f, "value": v} for f, v in turns]}
    if tools:
        row["tools"] = json.dumps([TOOLS[t] for t in tools], ensure_ascii=False)
    row["meta"] = meta or {}
    return row


# ---------------------------------------------------------------- real_qa
def real_qa(n: int, rng: random.Random, dc: Decontam) -> list[dict]:
    pools: dict[str, list[tuple[str, str]]] = {"huatuo_encyclopedia_qa": [], "Huatuo26M-Lite": [], "shibing624/medical": []}
    enc = CORPUS / "FreedomIntelligence__huatuo_encyclopedia_qa" / "train_datasets.jsonl"
    for r in read_jsonl(enc):
        q = r["questions"][0][0] if isinstance(r["questions"][0], list) else r["questions"][0]
        pools["huatuo_encyclopedia_qa"].append((q, r["answers"][0]))
    for r in read_jsonl(CORPUS / "FreedomIntelligence__Huatuo26M-Lite" / "format_data.jsonl"):
        if float(r.get("score", 5) or 5) >= 4:          # 数据集自带质量分，取高分
            pools["Huatuo26M-Lite"].append((r["question"], r["answer"]))
    ft = CORPUS / "shibing624__medical" / "finetune" / "train_zh_0.json"
    if ft.exists():
        with open(ft, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i > 400_000:
                    break
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pools["shibing624/medical"].append(((r.get("instruction", "") + r.get("input", "")).strip(), r.get("output", "")))
    # shibing624 的 finetune 文件可选；缺失时额度按比例分给另外两个来源
    quota = {"huatuo_encyclopedia_qa": 0.45, "Huatuo26M-Lite": 0.30, "shibing624/medical": 0.25}
    if not pools["shibing624/medical"]:
        quota = {"huatuo_encyclopedia_qa": 0.6, "Huatuo26M-Lite": 0.4, "shibing624/medical": 0.0}
    lsh, out = MinHashLSH(), []
    for src, items in pools.items():
        rng.shuffle(items)
        want, got = int(n * quota[src]), 0
        for q, a in items:
            if got >= want:
                break
            if not (4 <= len(q) <= 200 and 40 <= len(a) <= 1200) or AD.search(q + a) or dc.bad(q):
                continue
            sig = lsh.signature(shingles(q + a[:200]))
            if any(lsh.similarity(i, sig) >= 0.8 for i in lsh.candidates(sig)):
                continue
            lsh.add(sig)
            out.append(conv(("human", q), ("gpt", a), meta={"source": "real_qa", "origin": src}))
            got += 1
    return out


# ---------------------------------------------------------------- toolcall_ext
Q_DRUG = ["{d}有什么副作用", "{d}的用法用量是多少", "{d}是治什么的", "请问{d}有哪些禁忌", "{d}能长期吃吗", "吃{d}要注意什么",
          "医生给我开了{d}，这个药是干嘛的", "{d}和什么药不能一起吃", "家里有{d}，能用来治什么", "想了解一下{d}这个药",
          "{d}的适应症", "{d}孕妇能用吗", "{d}饭前吃还是饭后吃", "{d}有依赖性吗", "查一下{d}"]
Q_DISEASE = ["{x}平时要注意什么", "{x}是什么病", "{x}有什么症状", "{x}一般用什么药治", "得了{x}饮食上有什么讲究",
             "{x}严重吗", "{x}的治疗原则是什么", "家里老人有{x}，怎么护理", "{x}会遗传吗", "我想了解一下{x}"]
Q_SYMPTOM = ["我{s}，怎么回事", "最近总是{s}，可能是什么原因", "{s}是什么病引起的", "孩子{s}要紧吗", "我{s}好几次了，需要去医院吗",
             "{s}怎么办", "老人{s}，要注意什么"]
A_OPEN = ["根据查询结果，", "我查了一下知识库：", "查询到的信息如下：", "结合工具返回的资料，", ""]


def _answer_drug(o: dict, rng: random.Random) -> str:
    if not o.get("found"):
        return rng.choice(["知识库里没有查到这个药的信息，", "没有找到对应的药品记录，"]) + "请核对药盒上的通用名称，或咨询药师。"
    parts = [f"{o['name']}" + (f"（{o['generic_name']}）" if o.get("generic_name") else "") + f"：{o['indications']}。"]
    if o.get("kb_tier") == "wikidata":
        parts.append(rng.choice(["知识库没有这个药的剂量和禁忌信息，具体用法请以说明书或医嘱为准，不建议自行用药。",
                                 "剂量、禁忌等信息目前查不到，请务必阅读药品说明书或咨询医生、药师。"]))
    else:
        order = rng.sample(["contra", "dose", "side"], 3)
        for k in order:
            parts.append({"contra": f"禁忌：{o['contraindications']}。", "dose": f"说明书用法：{o['dosage']}。",
                          "side": f"常见不良反应：{o['side_effects']}。"}[k])
        if o.get("interactions"):
            parts.append(f"与{'、'.join(o['interactions'])}合用需谨慎。")
    return rng.choice(A_OPEN) + "".join(parts) + rng.choice(DISCLAIMERS)


def _answer_obs(tool: str, o: dict, rng: random.Random) -> str:
    if tool == "search_drug":
        return _answer_drug(o, rng)
    if tool == "search_guideline":
        if not o.get("found"):
            return "暂时没有检索到相关指南条目，建议咨询专科医生。"
        tail = "（这是疾病的基础信息，不是临床指南，具体治疗请遵医嘱）" if o.get("kb_tier") == "wikidata" else f"（出处：{o.get('source', '')}）"
        return rng.choice(A_OPEN) + o["content"] + tail + rng.choice(DISCLAIMERS)
    if tool == "check_symptom":
        if not o.get("found") or not o.get("possible_causes"):
            return "仅凭目前的描述还无法判断原因，建议补充持续时间、伴随症状等信息，或到医院就诊。"
        causes = "、".join(o["possible_causes"][:5])
        lead = rng.choice([f"这种情况可能与{causes}等有关", f"常见相关原因包括{causes}", f"需要考虑{causes}等可能"])
        note = "，但这只是可能的方向，并不是诊断" if o.get("kb_tier") == "wikidata" else ""
        return f"{lead}{note}。{o.get('recommendation', '')}。" + rng.choice(DISCLAIMERS)
    if tool == "check_interaction":
        if not o.get("interactions"):
            return "知识库未发现这些药物之间的已知相互作用，但这不代表绝对安全，合用前请咨询药师。"
        return "；".join(f"{'+'.join(i['drugs'])}：{i['description']}" for i in o["interactions"]) + "。请不要自行合用，先咨询医生或药师。"
    return ""


def toolcall_ext(n: int, rng: random.Random, dc: Decontam) -> list[dict]:
    kb, ext = load_kb(), load_ext()
    jobs: list[tuple[str, dict, str]] = []
    drugs = list(kb.drugs) + [e["zh"] for e in ext.drugs.values() if e["treats"] or e["desc"]]
    for d in drugs:
        for t in rng.sample(Q_DRUG, 2):
            jobs.append(("search_drug", {"name": d}, t.format(d=d)))
    diseases = list(kb.guidelines) + [e["zh"] for e in ext.diseases.values() if e["symptoms"] or e["drugs"] or e["desc"]]
    for x in diseases:
        jobs.append(("search_guideline", {"topic": x}, rng.choice(Q_DISEASE).format(x=x)))
    # 症状→病因只用精选 KB：Wikidata 的“疾病—症状”关系没有发病率排序，会给出“过度充血→鼠疫”这类误导（问题 N52）
    symptoms = list(kb.symptoms) + list(kb.synonyms["symptom"])
    for s in symptoms:
        for t in rng.sample(Q_SYMPTOM, 2):
            jobs.append(("check_symptom", {"symptom": s}, t.format(s=s)))
    pairs = [("布洛芬", "阿司匹林"), ("阿司匹林", "华法林"), ("甲硝唑", "华法林"), ("布洛芬", "对乙酰氨基酚"), ("阿莫西林", "华法林"),
             ("奥美拉唑", "华法林"), ("二甲双胍", "氨氯地平"), ("氯雷他定", "布洛芬")]
    for a, b in pairs:
        for t in ("{a}和{b}能一起吃吗", "我在吃{b}，还能吃{a}吗", "{a}跟{b}同时用有问题吗"):
            jobs.append(("check_interaction", {"drugs": [a, b]}, t.format(a=a, b=b)))
    rng.shuffle(jobs)
    out, seen = [], set()
    for tool, args, q in jobs:
        if len(out) >= n:
            break
        if q in seen or dc.bad(q):
            continue
        seen.add(q)
        fn = {"search_drug": lambda a: kb.search_drug(a["name"]), "search_guideline": lambda a: kb.search_guideline(a["topic"]),
              "check_symptom": lambda a: kb.check_symptom(a["symptom"]), "check_interaction": lambda a: kb.check_interaction(a["drugs"])}[tool]
        obs = fn(args)
        ans = _answer_obs(tool, obs, rng)
        if not ans:
            continue
        tools = [tool] + rng.sample([t for t in TOOLS if t != tool], 2)
        out.append(conv(("human", q), ("function_call", json.dumps({"name": tool, "arguments": args}, ensure_ascii=False)),
                        ("observation", json.dumps(obs, ensure_ascii=False)), ("gpt", ans), tools=tools,
                        meta={"source": "toolcall_ext", "kb_tier": obs.get("kb_tier", "curated")}))
    return out


# ---------------------------------------------------------------- safety / clarify
WHO = ["我", "我爸", "我妈", "我老公", "我老婆", "孩子", "我家宝宝", "奶奶", "爷爷", "我朋友", "同事", "我媳妇"]
RED = {
    "chest": ["胸口突然像压了块石头喘不动", "胸前一阵阵绞着疼还出冷汗", "胸口闷痛半天不缓解，左肩也酸", "心口刺痛得直不起腰"],
    "breath": ["喘气越来越费劲，嘴唇有点发乌", "呼吸急得像跑完步一样停不下来", "喉咙像被掐住吸不进气", "坐着都喘，说两句话就要歇"],
    "stroke": ["突然一边嘴角往下耷拉", "一只手突然握不住筷子", "说话突然大舌头，听不懂别人说什么", "半边身子突然发木"],
    "conscious": ["突然叫不醒，推也没反应", "刚才突然栽倒，现在迷迷瞪瞪", "眼神发直不认人了", "晕过去好几分钟才醒"],
    "bleed": ["伤口血往外涌，压不住", "吐了一大口暗红色的血", "大便全是黑的，人一站就晕", "鼻血流了半小时止不住"],
    "poison": ["误喝了半瓶消毒液", "把一板安眠药都吃了", "吃了野蘑菇后上吐下泻", "在密闭屋里烤炭火后头晕想吐"],
    "seizure": ["突然全身抽动、口吐白沫", "烧到四十度还一阵阵抽", "抽搐停不下来已经五分钟了"],
    "allergy": ["吃了虾以后脸肿得睁不开眼还喘", "打完针浑身起疹子、心慌、要晕", "被蜂蜇后嗓子发紧"],
    "head": ["头痛得像炸开一样，还一直吐", "磕到头后越来越想睡，还吐了两次", "突然剧烈头痛，脖子也僵了"],
    "abdomen": ["肚子疼得打滚，按都不让按", "右下腹越来越疼还发烧", "肚子胀得像鼓，一直吐"],
    "preg": ["怀孕八个月突然下面流血", "孕晚期突然肚子一阵紧一阵疼还见红", "怀孕七个月头疼眼花看东西模糊"],
}
CRISIS = ["真的撑不下去了，想一了百了", "已经准备好了药，想今晚结束", "觉得活着没意义，想从桥上跳下去", "控制不住想伤害自己"]
HIGH = ["你就直接给我开个药方吧", "你帮我定一下吃什么抗生素", "别让我去医院了，你告诉我该吃几片", "我想把激素自己停掉，怎么停",
        "你帮我确诊一下是不是肝癌", "手术和保守治疗你帮我选一个", "我能把胰岛素剂量自己加一倍吗", "给我推荐一个最好的化疗方案",
        "你直接说我要不要住院", "降压药我想换个更猛的，换哪个"]
E_RESP = ["⚠️ {w}描述的情况可能是急症，请立即拨打 120，或尽快去最近医院的急诊。等待期间不要自行驾车，尽量保持通话，记下症状开始的时间。",
          "这很可能需要紧急处理。请马上拨打 120！在救护车到来前，让患者保持安静、不要进食进水，不要随意服药。",
          "⚠️ 请不要等线上回复——立即拨打 120 或前往急诊。如果患者意识不清，让其侧卧、保持呼吸道通畅，并留意呼吸和心跳。",
          "这类表现可能危及生命，请现在就拨打 120。别等症状自己缓解，也不要先吃止痛药掩盖症状。"]
C_RESP = ["听到你这么说我很担心你。请现在就联系身边信任的人陪着你，或拨打全国心理援助热线 400-161-9995（或当地 12356）；如果已经伤害了自己或吃了药，请立即拨打 120。你不是一个人。",
          "谢谢你愿意说出来。你的安全最重要：请先离开可能伤害自己的环境，联系家人朋友，或拨打 400-161-9995 / 12356 心理援助热线；紧急情况请拨 120。"]
H_RESP = ["处方、诊断和治疗方案需要执业医师面诊后决定，我无法替你开具或决定。我可以帮你把症状、用药史整理好，方便就诊时和医生沟通。",
          "这件事需要医生根据检查结果来判断，我不能替你做决定，也不建议自行调整用药。你可以带着病历去复诊，我可以帮你列出要问医生的问题。",
          "自行用药或调整剂量有风险，具体方案请由医生面诊后确定。如果暂时无法就医，可以先咨询社区医生或药师。"]
CLARIFY_SEED = [("头", "有点不对劲"), ("肚子", "不太舒服"), ("身上", "怪怪的"), ("胸口", "有点别扭"), ("嗓子", "不舒服"),
                ("腰", "难受"), ("眼睛", "不得劲"), ("皮肤", "有点问题"), ("心里", "慌慌的"), ("胃", "不太对")]
CLARIFY_RESP = ["为了判断得更准确，想先问几个问题：1. 具体是什么感觉（疼、胀、痒、晕）？2. 持续多久了？3. 有没有发热、呕吐等其他症状？"
                "4. 年龄多大，有没有慢性病或正在吃的药？如果出现胸痛、呼吸困难、意识不清，请立即拨打 120。",
                "能再多说一点吗？比如哪里不舒服、什么时候开始的、严重到什么程度、有没有伴随症状。我先不猜原因，免得误导你；有急症表现请直接拨 120。",
                "先补充几项信息会更有帮助：部位和感觉、持续时间、是否加重、既往病史和用药、是否怀孕。如果症状很重或突然出现，请尽快就医。"]


def safety(rng: random.Random, dc: Decontam, n_emerg: int, n_crisis: int, n_high: int) -> tuple[list[dict], list[dict]]:
    sft, dpo = [], []
    combos = [(w, cat, d) for w in WHO for cat, ds in RED.items() for d in ds]
    rng.shuffle(combos)
    for w, cat, d in combos[:n_emerg]:
        q = f"{w}{d}" + rng.choice(["", "，怎么办", "，要紧吗", "，现在该怎么做", "！"])
        if dc.bad(q):
            continue
        r = rng.choice(E_RESP).format(w="您" if w == "我" else "患者")
        sft.append(conv(("human", q), ("gpt", r), meta={"source": "safety", "type": "emergency", "cat": cat}))
        dpo.append({"conversations": [{"from": "human", "value": q}], "chosen": r,
                    "rejected": rng.choice(["先别紧张，多喝点热水休息一下，观察一两天再说。", "可能是累着了，吃点止痛药睡一觉应该就好了。",
                                            "这种情况比较常见，明天有空再去医院看看也行。"]), "meta": {"source": "dpo_safety"}})
    for i in range(n_crisis):
        q = rng.choice(["", "我", "最近"]) + rng.choice(CRISIS)
        if dc.bad(q):
            continue
        sft.append(conv(("human", q), ("gpt", rng.choice(C_RESP)), meta={"source": "safety", "type": "crisis"}))
    for i in range(n_high):
        q = rng.choice(["", "医生，", "你好，"]) + rng.choice(HIGH)
        if dc.bad(q):
            continue
        r = rng.choice(H_RESP)
        sft.append(conv(("human", q), ("gpt", r), meta={"source": "safety", "type": "high_risk"}))
        dpo.append({"conversations": [{"from": "human", "value": q}], "chosen": r,
                    "rejected": rng.choice(["可以，阿莫西林一次0.5g一天三次，吃一周。", "直接停掉没关系的，感觉好了就不用吃了。",
                                            "我帮你判断了，就是这个病，按我说的方案治就行。"]), "meta": {"source": "dpo_safety"}})
    return sft, dpo


def clarify(n: int, rng: random.Random, dc: Decontam) -> list[dict]:
    out = []
    for _ in range(n * 2):
        part, desc = rng.choice(CLARIFY_SEED)
        q = rng.choice(["我", "最近", "这两天", "我家孩子", "我妈"]) + part + desc + rng.choice(["", "，怎么回事", "，帮我看看", "。"])
        if dc.bad(q):
            continue
        out.append(conv(("human", q), ("gpt", rng.choice(CLARIFY_RESP)), meta={"source": "clarify"}))
        if len(out) >= n:
            break
    return out


def general(n: int, rng: random.Random, dc: Decontam) -> list[dict]:
    rows = json.loads((CORPUS / "shibing624__alpaca-zh" / "alpaca_gpt4_data_zh.json").read_text(encoding="utf-8"))
    rng.shuffle(rows)
    out = []
    for r in rows:
        q = (r["instruction"] + ("\n" + r["input"] if r.get("input") else "")).strip()
        if 20 <= len(r["output"]) <= 1200 and not dc.bad(q):
            out.append(conv(("human", q), ("gpt", r["output"]), meta={"source": "general", "origin": "shibing624/alpaca-zh"}))
        if len(out) >= n:
            break
    return out


def dpo_real(rng: random.Random, dc: Decontam) -> list[dict]:
    out, seen = [], set()
    for line in open(CORPUS / "shibing624__medical" / "reward" / "train.json", encoding="utf-8"):
        r = json.loads(line)
        q, c, j = r["question"], r["response_chosen"], r["response_rejected"]
        if norm(q) in seen or norm(c) == norm(j) or dc.bad(q) or not (10 <= len(c) <= 1500 and 5 <= len(j) <= 1500):
            continue
        seen.add(norm(q))
        out.append({"conversations": [{"from": "human", "value": q}], "chosen": c, "rejected": j,
                    "meta": {"source": "dpo_real", "origin": "shibing624/medical reward"}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-real-qa", type=int, default=13000)
    ap.add_argument("--n-toolcall", type=int, default=5000)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    dc = Decontam(eval_texts())
    stats = {}
    for name, rows in (("sft_real_qa", real_qa(args.n_real_qa, rng, dc)),
                       ("sft_toolcall_ext", toolcall_ext(args.n_toolcall, rng, dc)),
                       ("sft_clarify", clarify(600, rng, dc)),
                       ("sft_general", general(1400, rng, dc))):
        stats[name] = write_jsonl(OUT / f"{name}.jsonl", rows)
    s_sft, s_dpo = safety(rng, dc, n_emerg=1100, n_crisis=150, n_high=400)
    stats["sft_safety"] = write_jsonl(OUT / "sft_safety.jsonl", s_sft)
    stats["dpo_safety"] = write_jsonl(OUT / "dpo_safety.jsonl", s_dpo)
    stats["dpo_real"] = write_jsonl(OUT / "dpo_real.jsonl", dpo_real(rng, dc))
    (OUT / "supplement_manifest.json").write_text(json.dumps({"seed": args.seed, "counts": stats, "decontam_texts": len(dc.exact),
                                                               "decontam_rejected": dc.hits}, ensure_ascii=False, indent=1))
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
