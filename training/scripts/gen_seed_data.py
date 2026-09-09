#!/usr/bin/env python3
"""
用 DeepSeek 批量生成 Agent 工具调用的种子训练数据。

用法：
    export DEEPSEEK_API_KEY=sk-xxx
    python gen_seed_data.py --model deepseek-chat --tool check_symptom --n 200 --out ./data/sft/toolcall_symptom_gen.jsonl

说明：
    纯标准库实现（urllib），无第三方依赖，Python 3.8+ 均可运行。
    生成的是"用户提问 + 工具调用 + 观察结果 + 最终回答"的完整对话链，
    落盘为 MedicalGPT ShareGPT 格式（含 function_call / observation 角色）。
"""
import argparse
import json
import os
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = os.environ.get("DEEPSEEK_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"))

# 各工具的 schema 与生成种子
TOOL_SPECS = {
    "check_symptom": {
        "description": "根据症状分析可能原因并提供初步建议",
        "parameters": {
            "type": "object",
            "properties": {
                "symptom": {"type": "string", "description": "患者描述的症状"},
                "duration": {"type": "string", "description": "症状持续时间"},
                "temperature": {"type": "string", "description": "体温（如有发热）"},
            },
            "required": ["symptom"],
        },
        "seeds": [
            "头痛", "发烧", "咳嗽", "腹泻", "胃痛", "失眠", "皮疹", "关节痛",
            "头晕", "胸闷", "咽喉痛", "鼻塞流涕", "乏力", "呕吐", "心悸",
            "腰痛", "耳鸣", "便秘", "尿频尿急", "视力模糊",
        ],
    },
    "search_drug": {
        "description": "查询药品的适应症、禁忌症、用法用量、副作用等信息",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "药品名称"}},
            "required": ["name"],
        },
        "seeds": [
            "布洛芬", "对乙酰氨基酚", "阿莫西林", "阿司匹林", "二甲双胍",
            "氨氯地平", "奥美拉唑", "氯雷他定", "头孢克肟", "蒙脱石散",
            "左氧氟沙星", "辛伐他汀", "硝苯地平", "甲硝唑", "罗红霉素",
        ],
    },
    "search_guideline": {
        "description": "检索临床指南中关于某疾病或主题的推荐意见",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "疾病或主题"},
                "aspect": {"type": "string", "description": "关注的方面"},
            },
            "required": ["topic"],
        },
        "seeds": [
            "高血压", "糖尿病", "高血脂", "冠心病", "哮喘", "痛风",
            "骨质疏松", "抑郁症", "慢性胃炎", "脑卒中", "脂肪肝",
            "慢性支气管炎", "甲亢", "贫血", "过敏性鼻炎",
        ],
    },
}

SYSTEM_PROMPT = """你是一名资深医疗 AI 训练数据标注专家。请为指定的医疗工具生成一条高质量的训练样本。

要求：
1. 输出严格的 JSON 对象，不要包含任何额外文字或 markdown 代码块标记。
2. JSON 结构必须为：
{
  "conversations": [
    {"from": "human", "value": "<患者提问>"},
    {"from": "function_call", "value": "<工具调用的 JSON 字符串，含 name 和 arguments>"},
    {"from": "observation", "value": "<工具返回结果的 JSON 字符串>"},
    {"from": "gpt", "value": "<基于观察结果的、专业且包含就医建议的最终回答>"}
  ]
}
3. function_call.value 必须是形如 {"name": "...", "arguments": {...}} 的字符串。
4. observation.value 必须是合法的 JSON 字符串，内容要符合工具返回的字段含义。
5. gpt.value 要结合观察结果回答，且包含适当的免责/就医提示，不得直接下诊断或处方。
6. 回答语气专业、客观、有温度。"""


def deepseek_chat(messages: list[dict], model: str = "deepseek-chat", temperature: float = 0.9) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2048,
    }
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    resp = urllib.request.urlopen(req, timeout=120)
    data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def build_user_prompt(tool: str, seed: str) -> str:
    spec = TOOL_SPECS[tool]
    return (
        f"工具名：{tool}\n"
        f"工具描述：{spec['description']}\n"
        f"工具参数 schema：{json.dumps(spec['parameters'], ensure_ascii=False)}\n"
        f"请围绕主题「{seed}」构造一条真实的患者提问。"
    )


def generate_one(tool: str, seed: str, model: str = "deepseek-chat") -> dict | None:
    try:
        content = deepseek_chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(tool, seed)},
            ],
            model=model,
        )
        content = content.strip()
        # 去掉可能的 ```json ... ``` 包裹
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:]
        data = json.loads(content)
        convs = data["conversations"]
        assert any(c["from"] == "function_call" for c in convs), "缺少 function_call"
        assert any(c["from"] == "observation" for c in convs), "缺少 observation"
        spec = TOOL_SPECS[tool]
        data["tools"] = json.dumps(
            [{"name": tool, "description": spec["description"], "parameters": spec["parameters"]}],
            ensure_ascii=False,
        )
        return data
    except Exception as e:  # noqa: BLE001
        print(f"[失败] {tool}/{seed}: {e}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--tool", choices=list(TOOL_SPECS), required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not API_KEY:
        sys.exit("缺少 API key：export DEEPSEEK_API_KEY=sk-xxx")

    seeds = TOOL_SPECS[args.tool]["seeds"]
    tasks = [(random.choice(seeds), args.model) for _ in range(args.n)]

    results, ok = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(generate_one, t, s, m): (t, s) for t, m in [(x[0], x[1]) for x in tasks]}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                results.append(r)
                ok += 1

    import pathlib
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"完成：成功 {ok}/{args.n} 条，写入 {out_path}")


if __name__ == "__main__":
    main()
