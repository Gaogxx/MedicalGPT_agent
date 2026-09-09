#!/usr/bin/env python3
"""
DeepSeek 离线蒸馏：用 DeepSeek（教师模型）对 SFT 问题生成高质量回答，
形成蒸馏 SFT 数据，再用 MedicalGPT 跑一轮 SFT 完成"从教师模型蒸馏能力"。

背景：MedicalGPT 的 OPD 训练(opd_training.py)用 AutoModelForCausalLM 加载
本地教师模型，不支持 API 教师(GPT-4/DeepSeek)。故采用"离线蒸馏"方案替代：
用 DeepSeek API 生成教师标注，再蒸馏。

用法：
    export DEEPSEEK_API_KEY=sk-xxx
    python3 distil_deepseek.py --input ../data/sft --n 1000 --out ../data/distil/distil_sft.jsonl

纯标准库实现(urllib)，无第三方依赖。
"""
import argparse
import json
import os
import random
import sys
import urllib.request
from pathlib import Path

API_KEY = os.environ.get("DEEPSEEK_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# 与 SFT 数据一致的工具 schema
TOOLS = [
    {"name": "check_symptom", "description": "根据患者描述的症状分析可能原因，提供初步建议",
     "parameters": {"type": "object", "properties": {"symptom": {"type": "string", "description": "患者描述的症状"}, "duration": {"type": "string", "description": "症状持续时间"}}, "required": ["symptom"]}},
    {"name": "search_drug", "description": "查询药品的适应症、禁忌症、用法用量、副作用等信息",
     "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "药品名称"}}, "required": ["name"]}},
    {"name": "search_guideline", "description": "检索临床指南中关于某疾病或主题的推荐意见",
     "parameters": {"type": "object", "properties": {"topic": {"type": "string", "description": "疾病或主题"}, "aspect": {"type": "string", "description": "关注的方面"}}, "required": ["topic"]}},
]

SYSTEM = """你是一位专业的医疗助手。请根据患者问题，生成一条完整的对话训练样本。

要求：
1. 输出严格的 JSON 对象，不要任何额外文字或 markdown 标记。
2. 若患者问题涉及症状/用药/疾病管理，需调用工具查询后再回答，结构：
   {"conversations": [
     {"from": "human", "value": "<问题>"},
     {"from": "function_call", "value": "{\"name\": \"工具名\", \"arguments\": {...}}"},
     {"from": "observation", "value": "<工具返回的 JSON 字符串>"},
     {"from": "gpt", "value": "<结合观察结果的最终回答>"}
   ], "tools": "<工具 schema JSON 数组字符串>"}
3. 若问题是概念性/定义性问题（如"什么是XX"），直接回答，不调用工具，结构：
   {"conversations": [{"from": "human", "value": "<问题>"}, {"from": "gpt", "value": "<直接回答>"}]}
4. 最终回答必须专业、客观，并包含适当的就医提示或免责声明，不得直接下诊断或处方。"""


def deepseek_chat(messages, temperature=0.7):
    payload = {"model": "deepseek-chat", "messages": messages, "temperature": temperature, "max_tokens": 2048}
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    resp = urllib.request.urlopen(req, timeout=120)
    return json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]


def load_questions(input_dir: Path) -> list[str]:
    """从 SFT 数据提取所有 human 问题。"""
    questions = []
    for f in input_dir.glob("*.jsonl"):
        if "medical_qa" in f.name or "toolcall" in f.name:
            for line in open(f, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    for c in d.get("conversations", []):
                        if c.get("from") == "human":
                            questions.append(c["value"])
                            break
                except json.JSONDecodeError:
                    continue
    return questions


def gen_one(question: str) -> dict | None:
    try:
        content = deepseek_chat([
            {"role": "system", "content": SYSTEM + f"\n\n可用工具：{json.dumps(TOOLS, ensure_ascii=False)}"},
            {"role": "user", "content": question},
        ])
        content = content.strip().strip("`")
        if content.startswith("json"):
            content = content[4:]
        data = json.loads(content)
        assert "conversations" in data
        return data
    except Exception as e:  # noqa: BLE001
        print(f"[失败] {question[:30]}: {e}", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../data/sft")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="../data/distil/distil_sft.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not API_KEY:
        sys.exit("缺少 DEEPSEEK_API_KEY")

    scripts_dir = Path(__file__).resolve().parent
    input_dir = Path(args.input)
    if not input_dir.is_absolute():
        input_dir = scripts_dir / args.input
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = scripts_dir / args.out

    questions = load_questions(input_dir)
    rng = random.Random(7)
    sample = rng.sample(questions, min(args.n, len(questions)))
    print(f"[蒸馏] 从 {len(questions)} 个问题中采样 {len(sample)} 个", file=sys.stderr)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    results, ok = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(gen_one, q): q for q in sample}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                results.append(r)
                ok += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[蒸馏] 完成：成功 {ok}/{len(sample)} 条，写入 {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
