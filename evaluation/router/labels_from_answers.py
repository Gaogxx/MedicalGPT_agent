#!/usr/bin/env python3
"""把判别模型的采集结果（collect_answers.py，路由数据集 train / val）转成路由器标签，省一遍裁判调用。

为什么可以直接转：两边用同一个裁判函数（judge_strong_typed，百川对照参考答案），“答砸”的定义也相同——
  label_router_set.py：修复前那次校验的 block 项（去掉输出安全关能修好的）非空，或裁判判 unsafe / poor；
  collect_answers.py ：同一个校验节点里同样口径的 rule_block，加同一个裁判对“校验前那版回答”的等级。
差别：这里的 answer 是校验前的回答（路由器标签记的是修复后的最终回答），只用于查看，不参与路由器训练（训练只用 query 和 label）。
采集时要加 --consult auto，否则复杂问诊工具没注册、和线上不一致。

裁判出错 / 解析失败的题不转，留给 label_router_set.py 按原流程补标（它会跳过输出文件里已有的 id）；test 部分本来就不在采集范围内。

  python evaluation/router/labels_from_answers.py --answers training/data/verifier/answers_final9b.jsonl --tag qwen3.5-9b
  python evaluation/router/label_router_set.py --base-url http://127.0.0.1:8000/v1 --model medical-agent --judge strong --tag qwen3.5-9b
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    queries = {}
    for line in open(ROOT / "evaluation" / "datasets" / "router_set.jsonl", encoding="utf-8"):
        r = json.loads(line)
        queries[r["id"]] = r["query"]
    out = ROOT / "evaluation" / "results" / f"router_labels_{args.tag}.jsonl"
    done = {json.loads(line)["id"] for line in open(out, encoding="utf-8")} if out.exists() else set()
    stats = Counter()
    with open(out, "a", encoding="utf-8") as fo:
        for line in open(args.answers, encoding="utf-8"):
            a = json.loads(line)
            if a["id"] in done:
                stats["已在输出里"] += 1
                continue
            if a["level"] in ("judge_error", "unparsed", "skipped"):
                stats["裁判无结果，留给 label_router_set 补标"] += 1
                continue
            label = int(bool(a["rule_block"]) or a["level"] in ("unsafe", "poor"))
            rec = {"id": a["id"], "split": a["split"], "source": a["source"], "query": queries[a["id"]],
                   "answer": a["state"]["answer"], "blocked": a["rule_block"], "judge": a["judge"], "level": a["level"],
                   "reason": a["reason"], "label": label, "seconds": a["seconds"], "from": "collect_answers"}
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["转换"] += 1
            stats[f"label={label}"] += 1
    print(json.dumps({"out": str(out.relative_to(ROOT)), **stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
