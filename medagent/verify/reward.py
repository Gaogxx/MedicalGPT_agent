"""奖励模型打分（校验器 v4，2026-11-16）：Skywork-Reward-V2-Qwen3-1.7B 零样本给（问题, 回答）打质量分，分数越低越可能答砸。

为什么加它：Laya v3 对 Medical-GPT 自己的失败几乎无效（验证集 15 条失败抓到 1 条，AUROC 0.543）——Laya 只有 3.2 亿参数、
没有大规模偏好预训练，而 Medical-GPT 的失败多是漏答要点、事实性用药错误。Skywork 奖励模型不用本项目数据训练，
在 Medical-GPT 规则放行的回答上 AUROC 0.818（验证集）/ 0.787（全部 1,656 条），漏答 0.846、不当用药 0.805。

校验顺序：规则 → Laya（类型化判定 + 失败类别）→ 奖励模型。奖励分 ≤ block_reward 升级强模型重答，≤ review_reward 放行但进人工复核。
阈值在 Medical-GPT 路由数据集 train 部分（零样本，不参与训练）的回答上按拦截比例定：block 3%（-2.02）、review 10%（0.65），
写在 artifacts/reward_model.json（evaluation/run_reward_eval.py）。

部署：python training/verifier/reward_serve.py --port 8003（training/verifier/06_reward.sh）
接口：POST /v1/reward {"query", "answer"} → {"score"}；GET /health
配置：MEDAGENT_REWARD_URL（默认 http://127.0.0.1:8003；设成空值 = 不用奖励模型）。没部署时连接立即被拒，校验器跳过这一步。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CALIBRATION = Path(__file__).resolve().parent / "artifacts" / "reward_model.json"
DEFAULT_URL = "http://127.0.0.1:8003"


def load_reward_calibration(path: Path = CALIBRATION) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


class RewardClient:
    def __init__(self, base_url: str | None = None, timeout: float = 5.0, client: Any = None) -> None:
        import httpx
        self.base_url = (base_url if base_url is not None else os.environ.get("MEDAGENT_REWARD_URL", DEFAULT_URL)).rstrip("/")
        self.http = client or httpx.Client(timeout=timeout)
        self.calls = 0

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    def score(self, query: str, answer: str) -> float:
        self.calls += 1
        r = self.http.post(f"{self.base_url}/v1/reward", json={"query": query, "answer": answer})
        r.raise_for_status()
        return float(r.json()["score"])


def fake_reward_app(score_fn=None):
    """测试用的假打分服务：score_fn(query, answer) -> float；默认回答越短分越低（只测管线）。"""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    fn = score_fn or (lambda q, a: len(a) / 10 - 3)

    async def reward(request: Request) -> JSONResponse:
        body = await request.json()
        return JSONResponse({"score": float(fn(body["query"], body["answer"]))})

    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    return Starlette(routes=[Route("/v1/reward", reward, methods=["POST"]), Route("/health", health)])
