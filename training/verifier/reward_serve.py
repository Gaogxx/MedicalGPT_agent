#!/usr/bin/env python3
"""奖励模型打分服务（校验器 v4）：Skywork-Reward-V2-Qwen3-1.7B，POST /v1/reward {"query", "answer"} → {"score"}。

  python training/verifier/reward_serve.py --port 8003                      # GPU / Apple MPS / CPU 自动选
  python training/verifier/reward_serve.py --model Skywork/Skywork-Reward-V2-Qwen3-0.6B   # 显存紧时用 0.6B（AUROC 低约 0.03）
需要 torch、transformers、starlette、uvicorn（laya 或 medgpt35 环境都有）。模型从 Hugging Face 下载（国内可设 HF_ENDPOINT=https://hf-mirror.com）。
"""
from __future__ import annotations

import argparse
import asyncio

import torch
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Skywork/Skywork-Reward-V2-Qwen3-1.7B")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--max-len", type=int, default=4096)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, dtype=torch.bfloat16 if dev != "cpu" else torch.float32).to(dev).eval()
    lock = asyncio.Lock()

    def score(query: str, answer: str) -> float:
        text = tok.apply_chat_template([{"role": "user", "content": query}, {"role": "assistant", "content": answer}],
                                       tokenize=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=args.max_len)
        with torch.no_grad():
            return float(model(**{k: v.to(dev) for k, v in ids.items()}).logits[0][0])

    async def reward(request: Request) -> JSONResponse:
        body = await request.json()
        async with lock:   # 单卡串行，避免并发请求互相抢显存
            s = await asyncio.to_thread(score, str(body.get("query", "")), str(body.get("answer", "")))
        return JSONResponse({"score": s, "model": args.model})

    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "model": args.model, "device": dev})

    import uvicorn
    uvicorn.run(Starlette(routes=[Route("/v1/reward", reward, methods=["POST"]), Route("/health", health)]),
                host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
