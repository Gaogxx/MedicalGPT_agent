"""命令行入口。

  medagent ask "布洛芬有什么副作用"            # 用 configs/models.yaml 的真实模型（需 API key / vLLM）
  medagent ask --mock "我胸口剧痛冒冷汗"       # 离线 Mock 模型，验证编排与容错
  medagent pause <run_id> / medagent resume <run_id>
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
    """有 small 档模型时用三档 TieredRouter，否则两档 SafetyFirstRouter。"""
    from .router.mf_router import MFRouter
    from .router.policy import TieredRouter
    art = ROOT / "medagent" / "router" / "artifacts"
    mf, thr = None, 0.5
    if (art / "mf_hash_dev80.npz").exists():
        mf = MFRouter.load(art / "mf_hash_dev80.npz")
        thr = json.loads((art / "mf_hash_dev80.json").read_text())["threshold@40%"]
    cls = TieredRouter if pool is not None and pool.tiers.get("small") else SafetyFirstRouter
    return cls(mf=mf, threshold=thr)


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
    args = ap.parse_args()
    if args.cmd == "pause":
        state = Path(os.environ.get("MEDAGENT_STATE_DIR", ".medagent_state"))
        CheckpointStore(str(state / "runs.sqlite")).request_pause(args.run_id)
        print("pause requested")
        return
    asyncio.run(_ask(args))


if __name__ == "__main__":
    main()
