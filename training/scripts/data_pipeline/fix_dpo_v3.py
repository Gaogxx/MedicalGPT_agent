#!/usr/bin/env python3
"""DPO v3 修复：reward_v3（mix_v3.py 产出，保持原样）→ reward_v3_fixed（export_formats.py 默认读这里）。SFT 数据一字不动。

2026-11-13 DPO 体检发现、2026-11-14 修完的问题：
  1. 243 对的 chosen / rejected 是 v0 的 ReAct 写法（"Action: …\\nAction Input: {…}"），导出时被当成普通回答，
     和 SFT / 线上的 <tool_call> 写法不一致 → 转成 {"name", "arguments"} JSON，导出时渲染成 <tool_call>；
  2. 891 对（obs_failure_dpo / unknown_entity_dpo）上下文里有工具调用却没带 tools，system 里没有 <tools> →
     从 augmented_v3 里同源的 SFT 样本（对话前缀相同）找回原来的工具定义；同一前缀有多个来源时按对话哈希固定挑一个；
  3. 固定话术：工具失败 / 查无此药 / 急症 / 高风险请求的 chosen（和急症、高风险、查无此药的 rejected）只有几句模板，
     验证集的原句训练集里各有约 20 条，验证准确率从一开始就是 1.0 →
       - 话术池分 TRAIN / VAL 两组（dpo_phrases.py），验证集只用 VAL 话术，查无此药的验证样本还换成只给验证集用的虚构药名；
       - 这几组和 slot_mask_dpo 的验证比例从 7.5% 提到 15%，按问法模板 / 症状描述整组切分，同一问法不会跨 train/val；
       - 训练集的 chosen 也从 TRAIN 话术池重新抽，同一句 chosen 最多 20 条（验证集 10 条）。
  4. 验证集不能含训练数据：派生组的验证样本几乎都来自 SFT 训练集的题 →
       - 与 DPO 训练集或 SFT 训练集有重叠（问题原文 / 近似重复 / 派生组问法模板 / 回答原文）的验证候选全部挪回训练集；
       - 反事实三组改从 SFT 验证集（SFT 没训过）的工具调用样本重新生成，急症 / 高风险用 dpo_phrases.py 里新写的问题；
       - 最后自检，验证集与训练数据零重叠、VAL 话术不进训练集，否则报错。
  5. dpo_real（shibing624/medical reward）里有约 5% 的 chosen 直接推荐具体药物 / 剂量（如“可以使用甲硝唑和左氧氟沙星”），
     等于奖励“凭记忆开药”，和 SYSTEM_PROMPT 第 1、4 条冲突；DPO v2 在直接回答题上确实变得更爱报药名 →
     chosen 推荐具体药物的真实偏好对整对去掉（rejected 推荐药、chosen 不推荐的保留，那是正确信号）。
  dpo_real / original 两组不重写话术（chosen 本来就各不相同），只做第 4 条的重叠检查。

augment.py / mix_v3.py 已修，以后重新生成不会再有 1、2；重新生成后仍要再跑一次本脚本做 3。
用法：python fix_dpo_v3.py   然后   python export_formats.py
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter

import dpo_phrases as P
from audit_data import _template
from common import REPO, MinHashLSH, human, norm, read_jsonl, row_key, shingles, write_jsonl
from mix_and_split import cap_same_chosen, fix_legacy_calls
from supplement import HIGH, RED, WHO

from medagent.knowledge import load_kb

D = REPO / "training" / "data"
SRC, OUT = D / "reward_v3", D / "reward_v3_fixed"
RESPLIT = {"obs_failure_dpo", "unknown_entity_dpo", "dpo_safety", "slot_mask_dpo"}
VAL_RATIO = 0.15
CAPS = {"train": 20, "val": 10}   # 验证话术训练没见过，同句可以多留几条
WHATS = ("药品信息", "症状分析", "指南", "资料")
FAB = re.compile(r"^(?P<name>.+?)主要用于(?P<ind>.+)，成人一次(?P<dose>[^，]+)，一日(?P<n>\d+)次。$", re.S)


def group(r: dict) -> str:
    m = r.get("meta", {})
    return m.get("aug") or m.get("source") or m.get("bucket") or "-"


def _h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def _prefix_key(conv: list[dict]) -> str:
    return json.dumps([c for c in conv if c["from"] != "system"], ensure_ascii=False)


def tools_index() -> dict[str, list[str]]:
    idx: dict[str, set[str]] = {}
    for f in ("sft_obs_failure.jsonl", "sft_unknown_entity.jsonl"):
        for r in read_jsonl(D / "augmented_v3" / f):
            if r.get("tools"):
                idx.setdefault(_prefix_key(r["conversations"][:-1]), set()).add(r["tools"])
    return {k: sorted(v) for k, v in idx.items()}


def fake_in(text: str) -> str | None:
    return next((n for n in sorted(P.FAKE_DRUGS_TRAIN, key=len, reverse=True) if n in text), None)


def split_key(r: dict, kb) -> str:
    """切分用的组键：同一问法（实体、虚构药名、称呼替换后相同）只落在一侧。"""
    q = human(r)
    if group(r) == "dpo_safety":
        d = next((d for ds in RED.values() for d in ds if d in q), None)
        if d:
            return "E:" + d
        h = next((h for h in HIGH if h in q), None)
        if h:
            return "H:" + h
    name = fake_in(q)
    if name:
        q = q.replace(name, "<FAKE>")
    return group(r) + ":" + _template(q, kb)


def _replace_name(r: dict, old: str, new: str) -> dict:
    conv = [{**c, "value": c["value"].replace(old, new)} for c in r["conversations"]]
    return {**r, "conversations": conv, "rejected": str(r["rejected"]).replace(old, new)}


def rephrase(r: dict, is_val: bool, rng: random.Random) -> dict:
    g, meta = group(r), {**r.get("meta", {}), "phrase_pool": "val" if is_val else "train"}
    if g == "obs_failure_dpo":
        what = next((w for w in WHATS if w in r["chosen"]), "资料")
        pool = P.FAIL_VAL if is_val else P.FAIL_TRAIN
        return {**r, "chosen": rng.choice(pool).format(what=what), "meta": meta}
    if g == "unknown_entity_dpo":
        name = fake_in(human(r))
        if name is None:
            return r
        if is_val:
            new = rng.choice(P.FAKE_DRUGS_VAL)
            r = _replace_name(r, name, new)
            m = FAB.match(str(r["rejected"]))
            if m:
                r = {**r, "rejected": rng.choice(P.FABRICATED_VAL).format(name=new, ind=m["ind"], dose=m["dose"], n=m["n"])}
            name = new
        pool = P.UNKNOWN_VAL if is_val else P.UNKNOWN_TRAIN
        return {**r, "chosen": rng.choice(pool).format(name=name), "meta": meta}
    if g == "dpo_safety":
        q = human(r)
        if r["rejected"] in P.E_REJ_TRAIN:      # 急症
            who = max((w for w in WHO if q.startswith(w)), key=len, default="")
            w = "您" if who == "我" else "患者"
            ch = rng.choice(P.E_VAL if is_val else P.E_TRAIN).format(w=w)
            rj = rng.choice(P.E_REJ_VAL) if is_val else r["rejected"]
        elif r["rejected"] in P.H_REJ_TRAIN:    # 高风险请求
            ch = rng.choice(P.H_VAL if is_val else P.H_TRAIN)
            rj = rng.choice(P.H_REJ_VAL) if is_val else r["rejected"]
        else:
            return r
        return {**r, "chosen": ch, "rejected": rj, "meta": meta}
    return r


# ---------------------------------------------------------------- 验证集与训练数据互斥
DERIVED = {"obs_failure_dpo", "unknown_entity_dpo", "slot_mask_dpo", "dpo_safety"}   # 由 SFT 样本 / 同一批问题派生的组


def _tmpl(q: str, kb) -> str:
    for n in sorted(P.FAKE_DRUGS_TRAIN + P.FAKE_DRUGS_VAL, key=len, reverse=True):
        q = q.replace(n, "<E>")
    who = max((w for w in WHO if q.startswith(w)), key=len, default="")
    return _template(q[len(who):], kb)


class Seen:
    """训练数据（DPO 训练集 + SFT 训练集）里出现过的问题与回答。命中任意一条就不能进验证集：
    问题原文相同 / 问题近似重复（MinHash Jaccard ≥ 0.8）/ 派生组的问法模板相同 / 回答原文相同。"""

    def __init__(self, kb, questions: list[str], responses: list[str]):
        self.kb = kb
        self.q = {norm(x) for x in questions}
        self.t = {_tmpl(x, kb) for x in questions}
        self.resp = {x.strip() for x in responses if x.strip()}
        self.lsh = MinHashLSH()
        for x in questions:
            self.lsh.add(self.lsh.signature(shingles(x)))

    def hit(self, r: dict) -> str | None:
        q = human(r)
        if norm(q) in self.q:
            return "问题原文"
        # 模板规则抓“从 SFT 训练题换个实体改出来”的样本；从 SFT 验证集新造的样本来源本身没训过，
        # 和训练集只是碰巧同一种通用问法（如“<E>是治什么的”），不算重叠
        if group(r) in DERIVED and not r.get("meta", {}).get("val_source") and _tmpl(q, self.kb) in self.t:
            return "问法模板"
        sig = self.lsh.signature(shingles(q))
        if any(self.lsh.similarity(i, sig) >= 0.8 for i in self.lsh.candidates(sig)):
            return "近似重复"
        if str(r["chosen"]).strip() in self.resp or str(r["rejected"]).strip() in self.resp:
            return "回答原文"
        return None


def sft_train_texts() -> tuple[list[str], list[str]]:
    qs, rs = [], []
    for r in read_jsonl(D / "sft_v3" / "train.jsonl"):
        qs.append(human(r))
        rs += [c["value"] for c in r["conversations"] if c["from"] == "gpt"]
    for r in read_jsonl(D / "openai_v3" / "sft" / "train_consult.jsonl"):   # 复杂问诊调用样本也进了 SFT 训练
        qs.append(next((m["content"] for m in r["messages"] if m["role"] == "user"), ""))
    return qs, rs


def new_val_rows(kb) -> list[dict]:
    """补充验证样本：反事实三组从 SFT 验证集（SFT 没训过）的工具调用样本生成，急症 / 高风险用只给验证集写的新问题。"""
    import augment
    rng = random.Random(20261114)
    src = [r for r in read_jsonl(D / "sft_v3" / "val.jsonl")
           if r.get("meta", {}).get("bucket") in ("toolcall", "toolcall_ext") and r.get("tools")]
    made: dict[str, list[dict]] = {"obs_failure_dpo": [], "unknown_entity_dpo": [], "slot_mask_dpo": []}
    for r in src:
        for g, x in (("slot_mask_dpo", augment.slot_mask(r)), ("obs_failure_dpo", augment.obs_failure(r, rng)),
                     ("unknown_entity_dpo", augment.unknown_entity(r, kb, rng))):
            if x:
                made[g].append({**x[1], "meta": {**x[1]["meta"], "val_source": "sft_val"}})
    out = []
    for g, rows in made.items():
        rng.shuffle(rows)
        out += rows[:NEW_VAL_CAP[g]]
    for d in P.RED_VAL:
        for w in rng.sample(WHO, 3):
            q = w + d + rng.choice(P.SAFETY_SUFFIX)
            out.append({"conversations": [{"from": "human", "value": q}], "chosen": "", "rejected": P.E_REJ_TRAIN[0],
                        "meta": {"source": "dpo_safety", "val_source": "new_question"}})
    for h in P.HIGH_VAL:
        for pre in rng.sample(["", "医生，", "你好，"], 2):
            out.append({"conversations": [{"from": "human", "value": pre + h}], "chosen": "", "rejected": P.H_REJ_TRAIN[0],
                        "meta": {"source": "dpo_safety", "val_source": "new_question"}})
    return [rephrase(r, True, random.Random(_h(row_key(r)))) for r in out]


NEW_VAL_CAP = {"obs_failure_dpo": 60, "unknown_entity_dpo": 50, "slot_mask_dpo": 50}   # 可用的 SFT 验证样本有限，取到上限为止


DRUG_REC = re.compile(r"(口服|服用|吃点|可以吃|可用|使用)[^。；\n]{0,12}(片|胶囊|颗粒|丸|口服液|注射液|软膏|滴眼液|冲剂|霉素|西林|沙星|洛芬|地平|普利)"
                      r"|[0-9]+\s*(mg|毫克|g|克|ml|毫升)")


def recommends_drug(text: str, kb) -> bool:
    return bool(DRUG_REC.search(text)) or any(n in text for n in kb.drugs if len(n) >= 3)


def main() -> None:
    kb = load_kb()
    idx = tools_index()
    st: Counter = Counter()
    # 1) 修格式、补 tools，按组重新切分（暂定）
    tent: list[tuple[dict, str]] = []
    for r, split in [(r, "train") for r in read_jsonl(SRC / "train.jsonl")] + [(r, "val") for r in read_jsonl(SRC / "val.jsonl")]:
        r2 = fix_legacy_calls(r)
        st["legacy_call_fixed"] += r2 is not r
        if any(c["from"] == "function_call" for c in r2["conversations"]) and not r2.get("tools"):
            k = _prefix_key(r2["conversations"])
            cands = idx.get(k)
            if not cands:
                raise SystemExit(f"找不到工具定义：{k[:120]}")
            r2 = {**r2, "tools": cands[_h(k) % len(cands)], "meta": {**r2.get("meta", {}), "tools_restored": True}}
            st["tools_restored"] += 1
        if group(r2) == "dpo_real" and recommends_drug(str(r2["chosen"]), kb):
            st["real_chosen_recommends_drug_dropped"] += 1
            continue
        if group(r2) in RESPLIT:
            split = "val" if _h(split_key(r2, kb)) / 0xFFFFFFFF < VAL_RATIO else "train"
        tent.append((r2, split))
    # 2) 验证候选只要和训练数据（DPO 训练集 + SFT 训练集）有任何重叠，就挪回训练集；挪动会扩大训练集，所以迭代到不再变化
    sft_q, sft_r = sft_train_texts()
    train = [r for r, s in tent if s == "train"]
    val = [r for r, s in tent if s == "val"]
    moved: Counter = Counter()
    while True:
        seen = Seen(kb, sft_q + [human(r) for r in train], sft_r + [str(x) for r in train for x in (r["chosen"], r["rejected"])])
        keep, back = [], []
        for r in val:
            why = seen.hit(r)
            (back if why else keep).append(r)
            if why:
                moved[f"{group(r)}/{why}"] += 1
        if not back:
            break
        train += back
        val = keep
    # 3) 按最终归属换话术：训练集用 TRAIN 话术，验证集用 VAL 话术（查无此药的验证样本同时换成验证专用药名）
    train = [rephrase(r, False, random.Random(_h(row_key(r)))) if group(r) in RESPLIT else r for r in train]
    val = [rephrase(r, True, random.Random(_h(row_key(r)))) if group(r) in RESPLIT else r for r in val]
    # 4) 补充验证样本（同样过一遍重叠检查，命中的直接丢掉：它们带着 VAL 话术，不能进训练集）
    seen = Seen(kb, sft_q + [human(r) for r in train], sft_r + [str(x) for r in train for x in (r["chosen"], r["rejected"])])
    added: Counter = Counter()
    have = {norm(human(r)) for r in val}
    for r in new_val_rows(kb):
        if seen.hit(r) or norm(human(r)) in have:
            added[f"{group(r)}/丢弃"] += 1
            continue
        have.add(norm(human(r)))
        val.append(r)
        added[f"{group(r)}/加入"] += 1
    out = {"train": cap_same_chosen(train, CAPS["train"], random.Random(42)),
           "val": cap_same_chosen(val, CAPS["val"], random.Random(42))}
    # 5) 自检：最终验证集与最终训练数据零重叠、VAL 话术不进训练集
    seen = Seen(kb, sft_q + [human(r) for r in out["train"]],
                sft_r + [str(x) for r in out["train"] for x in (r["chosen"], r["rejected"])])
    leaks = Counter(f"{group(r)}/{seen.hit(r)}" for r in out["val"] if seen.hit(r))
    val_phrases = set(P.FAIL_VAL + P.UNKNOWN_VAL + P.E_VAL + P.H_VAL + P.E_REJ_VAL + P.H_REJ_VAL)
    tr_text = "\n".join(str(r["chosen"]) + "\n" + str(r["rejected"]) for r in out["train"])
    phrase_leak = sum(any(p.split("{")[0][:12] in tr_text for p in [v]) for v in val_phrases if len(v.split("{")[0]) >= 12)
    report = {"fixes": dict(st), "val_moved_to_train": dict(moved), "val_added": dict(added),
              "leaks": dict(leaks), "val_phrase_prefix_in_train": phrase_leak}
    for split, rs in out.items():
        report[split] = {"n": len(rs), "by_group": dict(Counter(group(r) for r in rs)),
                         "distinct_chosen": len({str(r["chosen"]) for r in rs})}
        write_jsonl(OUT / f"{split}.jsonl", rs)
    (OUT / "manifest.json").write_text(json.dumps({"source": "training/data/reward_v3", "script": "fix_dpo_v3.py",
                                                   "val_ratio_resplit": VAL_RATIO, "same_chosen_cap": CAPS, **report},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if leaks or phrase_leak:
        raise SystemExit("验证集仍与训练数据重叠")


if __name__ == "__main__":
    main()
