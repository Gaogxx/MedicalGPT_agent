#!/usr/bin/env python3
"""
从 HuggingFace 数据集下载并采样 2000 条真实医疗问答数据。

数据源（走国内镜像 hf-mirror.com）：
  1. shibing624/medical finetune/train_zh_0.json    (instruction/input/output)  ~1200 条
  2. FreedomIntelligence/Huatuo26M-Lite format_data.jsonl (question/answer/score) ~400 条
  3. shibing624/medical reward/train.json            (question/response_chosen)  ~200 条
  4. shibing624/sharegpt_gpt4 sharegpt_zh_38K_format.jsonl (conversations)       ~200 条

输出：统一 ShareGPT 格式，每行一个 JSON：
  {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}

用法：
  HF_ENDPOINT=https://hf-mirror.com python3 download_real_data.py --out ../data/sft/_raw_medical_qa.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

HF_ENDPOINT = "https://hf-mirror.com"

# 医疗关键词：用于过滤通用对话数据中的医疗内容
MEDICAL_KEYWORDS = [
    "病", "症", "药", "医", "治疗", "诊断", "检查", "手术", "疫苗", "发烧", "发热",
    "头痛", "咳嗽", "高血压", "糖尿病", "心脏", "肝", "肾", "胃", "肺", "血液", "骨骼",
    "神经", "皮肤", "怀孕", "月经", "失眠", "焦虑", "抑郁", "过敏", "感染", "发炎",
    "疼痛", "肿胀", "出血", "呕吐", "腹泻", "便秘", "感冒", "流感", "肿瘤", "癌",
    "挂号", "门诊", "住院", "体检", "处方", "剂量", "副作用", "禁忌", "康复", "养生",
]


def stream_lines(url: str):
    """流式读取远程文件，逐行 yield（支持 hf-mirror 重定向）。"""
    req = urllib.request.Request(url, headers={"User-Agent": "medical-gpt-data-prep"})
    resp = urllib.request.urlopen(req, timeout=60)
    buffer = b""
    try:
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                yield line.decode("utf-8", errors="ignore")
    finally:
        resp.close()
    if buffer:
        yield buffer.decode("utf-8", errors="ignore")


def is_valid_qa(q: str, a: str) -> bool:
    """基础清洗：非空、长度合理、非纯数字/符号、不含明显噪声。"""
    q, a = q.strip(), a.strip()
    if len(q) < 4 or len(a) < 8:
        return False
    if len(q) > 200 or len(a) > 2000:
        return False
    if not re.search(r"[\u4e00-\u9fff]", q):  # 必须含中文
        return False
    if re.search(r"https?://|www\.|<.*?>", a):  # 去掉含 URL/HTML 的
        return False
    return True


def sample_medical_zh(train_zh_url: str, n: int) -> list[dict]:
    """shibing624/medical finetune/train_zh_0.json：instruction/output 格式。"""
    out = []
    for line in stream_lines(train_zh_url):
        if len(out) >= n:
            break
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        q = obj.get("instruction", "") or ""
        a = obj.get("output", "") or ""
        if obj.get("input"):
            q = q + ("\n" + obj["input"] if q else obj["input"])
        if is_valid_qa(q, a):
            out.append({"conversations": [{"from": "human", "value": q}, {"from": "gpt", "value": a}]})
    return out


def sample_huatuo(url: str, n: int) -> list[dict]:
    """Huatuo26M-Lite：question/answer/score，筛选高分中文医疗问答。"""
    out = []
    for line in stream_lines(url):
        if len(out) >= n:
            break
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        score = obj.get("score", 0)
        q = obj.get("question", "") or ""
        a = obj.get("answer", "") or ""
        if score >= 3 and is_valid_qa(q, a):
            out.append({"conversations": [{"from": "human", "value": q}, {"from": "gpt", "value": a}]})
    return out


def sample_reward(url: str, n: int) -> list[dict]:
    """shibing624/medical reward/train.json：question + response_chosen。"""
    out = []
    for line in stream_lines(url):
        if len(out) >= n:
            break
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        q = obj.get("question", "") or ""
        a = obj.get("response_chosen", "") or ""
        if is_valid_qa(q, a):
            out.append({"conversations": [{"from": "human", "value": q}, {"from": "gpt", "value": a}]})
    return out


def sample_sharegpt_medical(url: str, n: int) -> list[dict]:
    """sharegpt_zh_38K：已是 conversations 格式，过滤医疗内容。"""
    out = []
    for line in stream_lines(url):
        if len(out) >= n:
            break
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        convs = obj.get("conversations", [])
        if not convs:
            continue
        # 提取 human/gpt 角色；仅保留含医疗关键词的
        human = next((c["value"] for c in convs if c.get("from") == "human"), "")
        gpt = next((c["value"] for c in convs if c.get("from") == "gpt"), "")
        if not human or not gpt:
            continue
        if not any(k in human or k in gpt for k in MEDICAL_KEYWORDS):
            continue
        if is_valid_qa(human, gpt):
            out.append({"conversations": [{"from": "human", "value": human}, {"from": "gpt", "value": gpt}]})
    return out


def dedup(items: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for it in items:
        q = it["conversations"][0]["value"]
        if q in seen:
            continue
        seen.add(q)
        out.append(it)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/sft/_raw_medical_qa.jsonl")
    ap.add_argument("--train-zh", type=int, default=1200)
    ap.add_argument("--huatuo", type=int, default=400)
    ap.add_argument("--reward", type=int, default=200)
    ap.add_argument("--sharegpt", type=int, default=200)
    args = ap.parse_args()

    base = f"{HF_ENDPOINT}/datasets"
    sources = [
        ("shibing624/medical train_zh_0", sample_medical_zh,
         f"{base}/shibing624/medical/resolve/main/finetune/train_zh_0.json", args.train_zh),
        ("Huatuo26M-Lite", sample_huatuo,
         f"{base}/FreedomIntelligence/Huatuo26M-Lite/resolve/main/format_data.jsonl", args.huatuo),
        ("medical reward", sample_reward,
         f"{base}/shibing624/medical/resolve/main/reward/train.json", args.reward),
        ("sharegpt_zh_38K", sample_sharegpt_medical,
         f"{base}/shibing624/sharegpt_gpt4/resolve/main/sharegpt_zh_38K_format.jsonl", args.sharegpt),
    ]

    all_items: list[dict] = []
    for name, fn, url, n in sources:
        print(f"[下载] {name} 采样 {n} 条 ...", file=sys.stderr)
        try:
            items = fn(url, n)
            all_items.extend(items)
            print(f"  -> 实际获取 {len(items)} 条", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"  -> 失败：{e}", file=sys.stderr)

    all_items = dedup(all_items)
    # 相对路径基于脚本所在目录 (scripts/) 解析
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in all_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"\n完成：共 {len(all_items)} 条去重后数据，写入 {out_path}")


if __name__ == "__main__":
    main()
