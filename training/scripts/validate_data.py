#!/usr/bin/env python3
"""
校验训练数据格式是否符合 MedicalGPT ShareGPT 规范。

用法：
    python validate_data.py --dir ./data/sft      # 校验 SFT 数据
    python validate_data.py --dir ./data/reward   # 校验 DPO 数据

规则：
    - 每行必须是合法 JSON
    - SFT：conversations 为非空列表，每个元素含 from/value；工具调用样本须同时含 function_call 与 observation
    - DPO：必须含 chosen 与 rejected
"""
import argparse
import json
import sys
from pathlib import Path

VALID_ROLES = {"human", "gpt", "function_call", "observation", "system", "tool"}


def check_sft(obj: dict, path: Path, ln: int) -> list[str]:
    errs = []
    convs = obj.get("conversations")
    if not isinstance(convs, list) or not convs:
        errs.append(f"{path}:{ln} conversations 缺失或为空")
        return errs
    roles = []
    for i, c in enumerate(convs):
        if not isinstance(c, dict) or "from" not in c or "value" not in c:
            errs.append(f"{path}:{ln} conversations[{i}] 缺少 from/value")
            continue
        if c["from"] not in VALID_ROLES:
            errs.append(f"{path}:{ln} 非法角色 {c['from']}")
        roles.append(c["from"])
    # 工具调用样本的完整性
    if "function_call" in roles:
        if "observation" not in roles:
            errs.append(f"{path}:{ln} 含 function_call 但缺少 observation")
        if "tools" not in obj:
            errs.append(f"{path}:{ln} 工具调用样本缺少 tools 字段")
    return errs


def check_dpo(obj: dict, path: Path, ln: int) -> list[str]:
    errs = []
    if "chosen" not in obj or "rejected" not in obj:
        errs.append(f"{path}:{ln} DPO 样本缺少 chosen/rejected")
    if "conversations" not in obj:
        errs.append(f"{path}:{ln} DPO 样本缺少 conversations")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--mode", choices=["auto", "sft", "dpo"], default="auto")
    args = ap.parse_args()

    root = Path(args.dir)
    files = sorted(root.glob("*.jsonl"))
    if not files:
        sys.exit(f"目录 {root} 下没有 .jsonl 文件")

    total, bad = 0, 0
    for path in files:
        mode = args.mode
        if mode == "auto":
            mode = "dpo" if "dpo" in path.name else "sft"
        with open(path, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"{path}:{ln} JSON 解析失败: {e}")
                    bad += 1
                    continue
                errs = check_dpo(obj, path, ln) if mode == "dpo" else check_sft(obj, path, ln)
                for e in errs:
                    print(e)
                bad += len(errs)

    print(f"\n共 {total} 条样本，{bad} 条异常" + ("，全部通过 ✅" if bad == 0 else "，请修复 ❌"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
