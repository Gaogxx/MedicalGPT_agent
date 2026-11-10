"""命令行入口。

  medagent ask "布洛芬有什么副作用"            # 用 configs/models.yaml 的真实模型（需 API key / vLLM）
  medagent ask --mock "我胸口剧痛冒冷汗"       # 离线 Mock 模型，验证编排与容错
  medagent pause <run_id> / medagent resume <run_id>
  medagent pause <cs-run_id> --consult          # 暂停正在跑的复杂问诊（MCP 工具 complex_consult）
  medagent-mcp [--http --port 8765]             # 启动医疗工具 MCP Server
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from .agent import JsonlAudit, MedicalAgent
from .mcp_client import MCPToolbox
from .resilience.checkpoint import CheckpointStore
from .router.policy import SafetyFirstRouter

ROOT = Path(__file__).resolve().parents[1]


def _pool(mock: bool):
    from .llm.pool import ModelPool
    from .llm.providers import MockProvider
    if mock:
        return ModelPool({"strong": [MockProvider("strong-mock")], "local": [MockProvider("local-mock")]})
    kw = {}
    if os.environ.get("MEDAGENT_REDIS_URL"):
        import redis
        from .resilience.breaker import RedisBreakerStore
        kw["breaker_store"] = RedisBreakerStore(redis.Redis.from_url(os.environ["MEDAGENT_REDIS_URL"], socket_timeout=0.2))
    return ModelPool.from_yaml(str(ROOT / "configs" / "models.yaml"), **kw)


def _router(pool=None) -> SafetyFirstRouter:
    """有 small 档模型时用三档 TieredRouter，否则两档 SafetyFirstRouter。

    MF 打分器的加载顺序：
      1. mf_router.npz —— evaluation/router/train_router.py 用独立路由数据集（train / val / test）训练，dev 80 不参与；
      2. mf_hash_dev80.npz —— 冷启动版：只用 dev 80 条训练，在 dev 80 上的路由表现是样本内成绩，不能当效果报告。"""
    import logging

    from .router.mf_router import MFRouter
    from .router.policy import TieredRouter
    art = Path(os.environ.get("MEDAGENT_ROUTER_DIR", ROOT / "medagent" / "router" / "artifacts"))
    mf, thr, source = None, 0.5, "none"
    if (art / "mf_router.npz").exists():
        mf = MFRouter.load(art / "mf_router.npz")
        thr = json.loads((art / "mf_router.json").read_text())["threshold"]
        source = "mf_router（独立路由数据集训练）"
    elif (art / "mf_hash_dev80.npz").exists():
        mf = MFRouter.load(art / "mf_hash_dev80.npz")
        meta = json.loads((art / "mf_hash_dev80.json").read_text())
        thr = meta.get("threshold_router_val", meta["threshold@40%"])   # 优先用在独立数据上重新校准的阈值
        source = "mf_hash_dev80（冷启动，dev 80 训练）"
        logging.getLogger("medagent").warning("路由器用的是冷启动版 MF（dev 80 训练）；重训本地模型后请运行 "
                                              "evaluation/router/label_router_set.py + train_router.py")
    cls = TieredRouter if pool is not None and pool.tiers.get("small") else SafetyFirstRouter
    r = cls(mf=mf, threshold=thr)
    r.source = source
    return r


async def _ask(args: argparse.Namespace) -> None:
    from .mcp_servers.medical_server import build_server
    from .triage import load_learned_triage
    load_learned_triage()
    state = Path(os.environ.get("MEDAGENT_STATE_DIR", ".medagent_state"))
    state.mkdir(exist_ok=True)
    async with MCPToolbox({"medical": build_server()}) as tb:
        pool = _pool(args.mock)
        agent = MedicalAgent(pool, tb, _router(pool), CheckpointStore(str(state / "runs.sqlite")),
                             audit=JsonlAudit(str(state / "audit.jsonl")))
        if args.cmd == "ask":
            r = await agent.run(args.query, run_id=args.run_id)
        else:
            r = await agent.resume(args.run_id)
    print(r.get("answer") or r)
    print(json.dumps({k: r.get(k) for k in ("status", "run_id", "tier", "route", "repairs", "attribution")},
                     ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser("medagent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("query")
    a.add_argument("--mock", action="store_true")
    a.add_argument("--run-id")
    r = sub.add_parser("resume")
    r.add_argument("run_id")
    r.add_argument("--mock", action="store_true")
    p = sub.add_parser("pause")
    p.add_argument("run_id")
    p.add_argument("--consult", action="store_true", help="暂停的是复杂问诊工具的 run（cs-…，进度读 MCP 资源 consult://run/<run_id>）")
    args = ap.parse_args()
    if args.cmd == "pause":
        state = Path(os.environ.get("MEDAGENT_STATE_DIR", ".medagent_state"))
        CheckpointStore(str(state / ("consult_runs.sqlite" if args.consult else "runs.sqlite"))).request_pause(args.run_id)
        print("pause requested")
        return
    asyncio.run(_ask(args))


if __name__ == "__main__":
    main()
