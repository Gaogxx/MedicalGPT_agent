#!/usr/bin/env python3
"""小模型档评测：用真实的 Qwen3.5-2B（非思考模式、不给工具——与线上 small 档条件一致）回答 80 条 dev 问题，
与 2026-09 线上微调 MedicalGPT 的回答用同一套指标对比，并检查 TieredRouter 的门控是否只把它“答得住”的题分给它。

输出：evaluation/results/small_tier_eval.json（含全部回答原文，供人工复核）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from metrics import aggregate, score_case  # noqa: E402

from medagent.router.policy import TieredRouter  # noqa: E402
from medagent.safety import assess  # noqa: E402
from medagent.verify.detectors import detect  # noqa: E402

SYSTEM = "你是一名谨慎的医疗科普助手，不能替代医生。回答简洁准确；不确定时说明；涉及用药、诊断请建议就医；急症请建议拨打120。"


def load_model(name: str = "Qwen/Qwen3.5-2B"):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16).to("mps").eval()
    return tok, model


def generate(tok, model, query: str, max_new_tokens: int = 320) -> tuple[str, float, int]:
    import torch
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": query}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt").to("mps")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.05)
    new = out[0][ids["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip(), time.perf_counter() - t0, len(new)


def main() -> None:
    cases = [json.loads(l) for l in open(ROOT / "evaluation" / "datasets" / "agent_eval_dev.jsonl", encoding="utf-8")]
    router = TieredRouter()
    tok, model = load_model()
    rows = []
    for c in cases:
        ans, sec, ntok = generate(tok, model, c["query"])
        d = router.route(c["query"])
        f = [x.code for x in detect(ans, query=c["query"], assessment=assess(c["query"])) if x.severity == "block"]
        rows.append({"id": c["id"], "category": c["category"], "query": c["query"], "routed_tier": d.tier,
                     "small_answer": ans, "seconds": round(sec, 2), "tokens": ntok, "block_findings": f,
                     "medicalgpt_label": c["label"]})
        print(c["id"], d.tier, f"{sec:.1f}s", ans[:60].replace("\n", " "), flush=True)
    by = {"all": cases, "routed_small": [c for c, r in zip(cases, rows) if r["routed_tier"] == "small"],
          "not_routed_small": [c for c, r in zip(cases, rows) if r["routed_tier"] != "small"]}
    res = {"model": "Qwen/Qwen3.5-2B (bf16, MPS, non-thinking, no tools)", "router": "TieredRouter",
           "mean_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 2),
           "mean_tokens_per_sec": round(sum(r["tokens"] for r in rows) / sum(r["seconds"] for r in rows), 1)}
    for name, subset in by.items():
        ids = {c["id"] for c in subset}
        small_scores = [score_case(c, r["small_answer"]) for c, r in zip(cases, rows) if c["id"] in ids]
        base_scores = [score_case(c, c["answer"]) for c in subset]
        res[name] = {"n": len(subset), "qwen3.5-2b": aggregate(subset, small_scores, with_human=False),
                     "medicalgpt_2026_09": aggregate(subset, base_scores, with_human=False),
                     "qwen3.5-2b_block_rate": round(sum(1 for r in rows if r["id"] in ids and r["block_findings"]) / max(len(ids), 1), 4)}
    res["answers"] = rows
    (ROOT / "evaluation" / "results" / "small_tier_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                                                        encoding="utf-8")
    for k in ("all", "routed_small", "not_routed_small"):
        v = res[k]
        print(k, v["n"], "small auto_pass", v["qwen3.5-2b"]["auto_pass_rate"], "| medicalgpt", v["medicalgpt_2026_09"]["auto_pass_rate"],
              "| small block", v["qwen3.5-2b_block_rate"])


if __name__ == "__main__":
    main()
