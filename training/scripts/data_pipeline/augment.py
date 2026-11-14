#!/usr/bin/env python3
"""遮挡 / 反事实造数。每一种增强都对准基线里真实出现过的失败模式（见 evaluation/results/baseline_v0.json）：

  slot_mask        槽位遮挡：删掉问题里的持续时间/年龄，同时删掉 function_call 里对应参数
                   → 学会“没给的信息不编”（对准：“您的嗓子疼持续了3天”，用户根本没说）
  arg_repair       修复：原数据里 11% 的工具调用带着用户没说过的 duration（如“3天”），直接删掉该参数
  ask_clarify      信息遮挡→追问：把具体症状遮成“我不舒服”，目标回答是结构化追问（参考 Baichuan-M3 的问诊建模思路）
  obs_failure      观察遮挡：工具结果换成超时/空/报错，目标回答如实说明并给安全的通用建议，不编具体信息
  unknown_entity   实体遮挡：药名换成知识库外的名字，observation=found:false，目标回答“查不到”
  colloquial       口语/别名噪声：标准术语换成口语说法（腹泻→拉肚子、布洛芬→美林），目标不变 → 提升鲁棒性
  raft             RAG 干扰文档：1 篇真证据 + 3 篇难负例（BM25 召回的其他答案）打乱；20% 去掉真证据，
                   目标回答“资料不足以回答”（RAFT 思路：训练模型在噪声证据中选证据、没证据时不硬答）
同时产出反事实 DPO 对：chosen=不编造 / 追问 / 承认查不到；rejected=编造参数 / 编造药品信息 / 硬答。
所有增强样本带 meta.aug 标记，配比与上限由 mix_and_split.py 控制。
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
from pathlib import Path

from common import REPO, human, read_jsonl, turns, write_jsonl

from medagent.knowledge import load_kb
from medagent.rag.corpus import docs_from_sharegpt_qa
from medagent.rag.retrievers import BM25Retriever

DUR_PHRASE = re.compile(r"[，,]?(已经|有|持续|快)?(了)?\s*([0-9]+|[一二两三四五六七八九十半几]+)\s*(个)?(多)?\s*(天|周|星期|礼拜|月|小时)(了|左右|多)?")
AGE_PHRASE = re.compile(r"[，,]?(我|孩子|宝宝)?(今年)?\s*\d{1,3}\s*岁(了)?")
FAKE_DRUGS = ["康复灵胶囊", "速效清片", "XX特效丸", "新安宁颗粒", "复方奇效片", "进口神经修复剂", "XYZ-12注射液"]
CLARIFY = ("为了更准确地帮助您，请补充几点信息：\n1. 具体哪里不舒服？（如头痛、腹痛、发热等）\n2. 持续多久了？\n"
           "3. 严重程度如何，是否影响日常活动？\n4. 年龄、是否怀孕、有无慢性病或正在服用的药物？\n"
           "如果出现胸痛、呼吸困难、意识不清、肢体无力或大出血，请立即拨打 120。")
OBS_FAIL = [{"error": "TIMEOUT: 工具服务超时"}, {"error": "EXTERNAL: 知识库服务暂不可用"}, {}]
FAIL_ANSWERS = [
    "抱歉，刚才查询{what}时服务暂时不可用，我无法给出经过核实的具体信息，也不会凭记忆编造剂量或用法。建议稍后再试，或直接咨询医生/药师；如症状明显加重，请及时就医。",
    "{what}暂时没有查到结果（系统超时）。为避免给出不准确的信息，我先不提供具体用药建议；您可以稍后再问一次，或线下咨询药师。若出现明显加重，请尽快就医。",
    "这次没能获取到{what}，我没法核实相关内容。比起凭印象回答，更稳妥的是咨询医生或药师；如有急症表现（胸痛、呼吸困难、意识改变等）请立即拨打120。",
    "系统查询{what}时出错了，我不能确认具体信息。请您稍后重试，或带着病历/药盒去看医生，让医生根据您的情况判断。",
]
UNKNOWN_ANSWERS = [
    "知识库中没有查到「{name}」的药品信息。我不能提供未经核实的用法用量。请核对药盒上的通用名，或携带药品咨询医生或药师。",
    "没有找到名为「{name}」的药品。可能是商品名不常见或名称有误，建议提供药盒上的“通用名称”，或直接咨询药师。",
    "「{name}」不在我能查到的药品范围内，我无法确认它的成分和用法，请不要依据网络传言自行服用，可以咨询医生或药师。",
]
NO_EVIDENCE = "检索到的资料与您的问题不直接相关，我无法据此给出可靠回答。建议咨询专科医生或提供更多细节。"


def _conv(row: dict) -> list[dict]:
    return row["conversations"]


def _set_args(fc_value: str, drop: tuple[str, ...]) -> str:
    obj = json.loads(fc_value)
    obj["arguments"] = {k: v for k, v in obj.get("arguments", {}).items() if k not in drop}
    return json.dumps(obj, ensure_ascii=False)


def arg_repair(row: dict) -> dict | None:
    """原数据里 duration 参数没有依据时删掉；返回修复后的样本（不是新增样本）。"""
    q = human(row)
    out, changed = copy.deepcopy(row), False
    for c in _conv(out):
        if c["from"] == "function_call":
            obj = json.loads(c["value"])
            dur = str(obj.get("arguments", {}).get("duration", "") or "")
            if dur and not DUR_PHRASE.search(q) and dur not in q:
                c["value"] = _set_args(c["value"], ("duration",))
                changed = True
    if changed:
        out.setdefault("meta", {})["repaired"] = "ungrounded_duration"
        return out
    return None


def slot_mask(row: dict) -> tuple[dict, dict] | None:
    q = human(row)
    m = DUR_PHRASE.search(q)
    fcs = [c for c in _conv(row) if c["from"] == "function_call"]
    if not m or not fcs or "duration" not in json.loads(fcs[0]["value"]).get("arguments", {}):
        return None
    new_q = (q[: m.start()] + q[m.end():]).strip("，, ")
    out = copy.deepcopy(row)
    for c in _conv(out):
        if c["from"] == "human":
            c["value"] = new_q
        elif c["from"] == "function_call":
            c["value"] = _set_args(c["value"], ("duration",))
        elif c["from"] == "observation":
            try:
                o = json.loads(c["value"])
                o.pop("duration", None)
                c["value"] = json.dumps(o, ensure_ascii=False)
            except json.JSONDecodeError:
                pass
        elif c["from"] == "gpt":
            c["value"] = DUR_PHRASE.sub("", c["value"])
    out["meta"] = {"aug": "slot_mask", "masked": m.group(0), "orig_query": q}
    # 反事实 DPO：rejected 编一个持续时间
    fc_obj = json.loads([c for c in _conv(out) if c["from"] == "function_call"][0]["value"])
    # 写成 {"name", "arguments"} JSON，export_formats 才会渲染成 <tool_call>（原先的 "Action: …" 写法会被当成普通回答）
    chosen = json.dumps({"name": fc_obj["name"], "arguments": fc_obj["arguments"]}, ensure_ascii=False)
    bad = dict(fc_obj["arguments"], duration="3天")
    rejected = json.dumps({"name": fc_obj["name"], "arguments": bad}, ensure_ascii=False)
    dpo = {"conversations": [{"from": "human", "value": new_q}], "tools": row.get("tools", ""),
           "chosen": chosen, "rejected": rejected, "meta": {"aug": "slot_mask_dpo"}}
    return out, dpo


def ask_clarify(rng: random.Random) -> dict:
    vague = rng.choice(["我不舒服", "我最近身体不太好", "我感觉怪怪的，怎么办", "医生，我难受", "我好像生病了"])
    return {"conversations": [{"from": "human", "value": vague}, {"from": "gpt", "value": CLARIFY}],
            "meta": {"aug": "ask_clarify"}}


def obs_failure(row: dict, rng: random.Random) -> tuple[dict, dict] | None:
    conv = _conv(row)
    idx = [i for i, c in enumerate(conv) if c["from"] == "observation"]
    if not idx:
        return None
    out = copy.deepcopy(row)
    fail = rng.choice(OBS_FAIL)
    out["conversations"][idx[0]]["value"] = json.dumps(fail, ensure_ascii=False)
    fc = json.loads(conv[idx[0] - 1]["value"]) if conv[idx[0] - 1]["from"] == "function_call" else {"name": "工具"}
    what = {"search_drug": "药品信息", "check_symptom": "症状分析", "search_guideline": "指南"}.get(fc.get("name"), "资料")
    out["conversations"] = out["conversations"][: idx[0] + 1] + [{"from": "gpt", "value": rng.choice(FAIL_ANSWERS).format(what=what)}]
    out["meta"] = {"aug": "obs_failure", "fail": fail, "orig_query": human(row)}
    original_answer = next((c["value"] for c in reversed(conv) if c["from"] == "gpt"), "")
    # 上下文里有工具调用，tools 必须带上（线上请求总带工具定义；原先漏了，system 里没有 <tools>）
    dpo = {"conversations": out["conversations"][:-1], "chosen": rng.choice(FAIL_ANSWERS).format(what=what),
           "rejected": original_answer, "tools": row.get("tools", ""), "meta": {"aug": "obs_failure_dpo"}}
    return out, dpo


def unknown_entity(row: dict, kb, rng: random.Random) -> tuple[dict, dict] | None:
    fcs = [c for c in _conv(row) if c["from"] == "function_call"]
    if not fcs or json.loads(fcs[0]["value"]).get("name") != "search_drug":
        return None
    orig = json.loads(fcs[0]["value"])["arguments"].get("name", "")
    q = human(row)
    if not orig or orig not in q:
        return None
    fake = rng.choice(FAKE_DRUGS)
    new_q = q.replace(orig, fake)
    obs = {"found": False, "query": fake, "error": f"知识库中未找到药品「{fake}」"}
    out = {"conversations": [{"from": "human", "value": new_q},
                             {"from": "function_call", "value": json.dumps({"name": "search_drug", "arguments": {"name": fake}}, ensure_ascii=False)},
                             {"from": "observation", "value": json.dumps(obs, ensure_ascii=False)},
                             {"from": "gpt", "value": rng.choice(UNKNOWN_ANSWERS).format(name=fake)}],
           "tools": row.get("tools", ""), "meta": {"aug": "unknown_entity", "orig": orig, "orig_query": q}}
    info = kb.drugs.get(orig) or next(iter(kb.drugs.values()))
    fabricated = f"{fake}主要用于{info['indications']}，成人一次{rng.choice(['1片', '2粒', '10mg'])}，一日{rng.choice([2, 3])}次。"
    dpo = {"conversations": out["conversations"][:3], "chosen": out["conversations"][3]["value"], "rejected": fabricated,
           "tools": out["tools"], "meta": {"aug": "unknown_entity_dpo"}}
    return out, dpo


def colloquial(row: dict, kb, rng: random.Random) -> dict | None:
    q = human(row)
    reverse: dict[str, list[str]] = {}
    for table in ("symptom", "drug", "disease"):
        for alias, std in kb.synonyms[table].items():
            reverse.setdefault(std, []).append(alias)
    hits = [s for s in reverse if s in q]
    if not hits:
        return None
    s = rng.choice(hits)
    new_q = q.replace(s, rng.choice(reverse[s]), 1)
    if rng.random() < 0.5:
        new_q = rng.choice(["医生你好，", "请问一下，", "大夫，"]) + new_q + rng.choice(["", "谢谢", "急！"])
    out = copy.deepcopy(row)
    for c in _conv(out):
        if c["from"] == "human":
            c["value"] = new_q
            break
    out["meta"] = {"aug": "colloquial", "from": s, "orig_query": q}
    return out


def raft_samples(raw_path: Path, n: int, rng: random.Random, p_no_oracle: float = 0.2) -> list[dict]:
    docs = docs_from_sharegpt_qa(raw_path, source="qa")
    first_chunk = {d.meta["qid"]: d for d in docs if d.meta["chunk"] == 0}
    bm25 = BM25Retriever(docs)
    rows = read_jsonl(raw_path)
    qids = [i for i in first_chunk if i < len(rows)]
    rng.shuffle(qids)
    out = []
    for qid in qids[:n]:
        q = human(rows[qid])
        ans = next((c["value"] for c in rows[qid]["conversations"] if c["from"] == "gpt"), "")
        if not q or not ans:
            continue
        negs = [docs[i] for i, _ in bm25.search(q, k=20) if docs[i].meta["qid"] != qid][:3]
        no_oracle = rng.random() < p_no_oracle
        ev = ([] if no_oracle else [first_chunk[qid]]) + negs
        rng.shuffle(ev)
        obs = {"query": q, "evidence": [{"id": d.id, "text": d.text[:300], "source": "知识库"} for d in ev]}
        out.append({"conversations": [
            {"from": "human", "value": q},
            {"from": "function_call", "value": json.dumps({"name": "search_knowledge", "arguments": {"query": q}}, ensure_ascii=False)},
            {"from": "observation", "value": json.dumps(obs, ensure_ascii=False)},
            {"from": "gpt", "value": NO_EVIDENCE if no_oracle else ans}],
            "meta": {"aug": "raft", "oracle": not no_oracle, "qid": qid}})
    return out


def main() -> None:
    d = REPO / "training" / "data"
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", nargs="*", default=[str(d / "sft" / f) for f in
                                                 ("toolcall_symptom.jsonl", "toolcall_drug.jsonl", "toolcall_guideline.jsonl")])
    ap.add_argument("--raw-qa", default=str(d / "raw_medical_qa.jsonl"))
    ap.add_argument("--out-dir", default=str(d / "augmented"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-clarify", type=int, default=120)
    ap.add_argument("--n-raft", type=int, default=300)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    kb = load_kb()
    rows = [r for p in args.sft for r in read_jsonl(p)]
    sft_out: dict[str, list[dict]] = {k: [] for k in ("repaired", "slot_mask", "ask_clarify", "obs_failure",
                                                     "unknown_entity", "colloquial", "raft")}
    dpo_out: list[dict] = []
    for r in rows:
        fixed = arg_repair(r)
        if fixed:
            sft_out["repaired"].append(fixed)
        src = fixed or r
        if (x := slot_mask(src)) and rng.random() < 0.5:
            sft_out["slot_mask"].append(x[0]); dpo_out.append(x[1])
        if rng.random() < 0.12 and (x := obs_failure(src, rng)):
            sft_out["obs_failure"].append(x[0]); dpo_out.append(x[1])
        if rng.random() < 0.3 and (x := unknown_entity(src, kb, rng)):
            sft_out["unknown_entity"].append(x[0]); dpo_out.append(x[1])
        if rng.random() < 0.15 and (x := colloquial(src, kb, rng)):
            sft_out["colloquial"].append(x)
    sft_out["ask_clarify"] = [ask_clarify(rng) for _ in range(args.n_clarify)]
    sft_out["raft"] = raft_samples(Path(args.raw_qa), args.n_raft, rng)
    out = Path(args.out_dir)
    stats = {}
    for k, v in sft_out.items():
        stats[k] = write_jsonl(out / f"sft_{k}.jsonl", v)
    stats["dpo_counterfactual"] = write_jsonl(out / "dpo_counterfactual.jsonl", dpo_out)
    (out / "augment_manifest.json").write_text(json.dumps({"seed": args.seed, "counts": stats}, ensure_ascii=False, indent=2))
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
