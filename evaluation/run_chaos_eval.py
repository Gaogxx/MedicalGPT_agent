#!/usr/bin/env python3
"""容错层混沌评测（离线可复现：Mock 模型 + 进程内 MCP Server + 固定随机种子的故障注入）。

衡量的是“系统在故障下的行为”，不是模型回答质量：
  E1 端到端：注入 LLM/工具 故障，比较 naive（无重试/无熔断/无跨层兜底/无校验修复）与 full 配置
       success_rate        回答非空、无泄漏、且包含知识库事实（不是只有一句“系统暂时无法回答”）
       attribution_acc     失败或被修复的 run 中，归因结果与首个注入故障类别一致的比例
  E2 幂等：预约挂号时注入“服务端已执行但响应丢失”，统计重复挂号数（naive=每次重试新 key / full=同 key）
  E3 Saga：预约成功、发通知失败 → 自动取消预约；补偿也失败 → 进死信
  E4 暂停/崩溃恢复：执行到一半暂停 / 在有副作用之后崩溃，resume 后结果正确且副作用不重复
  E5 乐观并发：两个执行者同时写同一个 run，只有一个成功，另一个拿到 CONFLICT；update_with_retry 合并
  E6 Redis 共享熔断：实例 A 连续失败打开熔断，实例 B 立刻快速失败；Redis 挂掉自动降级为本地熔断
"""
from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medagent.agent import MedicalAgent  # noqa: E402
from medagent.knowledge import load_kb  # noqa: E402
from medagent.llm.pool import ModelPool  # noqa: E402
from medagent.llm.providers import FaultInjector, FaultyProvider, MockProvider  # noqa: E402
from medagent.mcp_client import MCPToolbox  # noqa: E402
from medagent.mcp_servers.medical_server import build_server  # noqa: E402
from medagent.resilience.breaker import BreakerConfig, CircuitBreaker, RedisBreakerStore  # noqa: E402
from medagent.resilience.checkpoint import CheckpointStore, ConcurrentModification, update_with_retry  # noqa: E402
from medagent.resilience.retry import RetryPolicy  # noqa: E402
from medagent.resilience.saga import Saga, SagaStep  # noqa: E402
from medagent.router.policy import SafetyFirstRouter  # noqa: E402
from medagent.types import AgentError, ErrorKind  # noqa: E402

LLM_FAULTS = {"network": 0.10, "timeout": 0.05, "rate_limit": 0.05, "external": 0.05, "silent_empty": 0.05, "tool_leak": 0.08}
TOOL_FAULTS = {"network": 0.08, "external": 0.05, "silent_empty": 0.05}


class LoggingInjector(FaultInjector):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.log: list[str] = []

    def pick(self):
        k = super().pick()
        if k:
            self.log.append(k)
        return k


def queries(n: int, rng: random.Random) -> list[str]:
    kb = load_kb()
    tpl = [("drug", "{}有什么副作用"), ("drug", "{}怎么吃"), ("symptom", "我{}好几天了怎么办"), ("guideline", "{}饮食要注意什么")]
    out = []
    for _ in range(n):
        kind, t = rng.choice(tpl)
        vocab = {"drug": list(kb.drugs), "symptom": [s for s, v in kb.symptoms.items() if v["urgency"] != "emergency"],
                 "guideline": list(kb.guidelines)}[kind]
        out.append(t.format(rng.choice(vocab)))
    return out


def useful(answer: str, query: str) -> bool:
    kb = load_kb()
    ents = kb.link_entities(query)
    if '{"name"' in answer or "ptom" in answer or len(answer.strip()) < 20:
        return False
    facts = []
    for d in ents["drug"]:
        facts += [kb.drugs[d]["side_effects"][:4], kb.drugs[d]["indications"][:4], kb.drugs[d]["dosage"][:4]]
    for s in ents["symptom"]:
        facts += [c[:3] for c in kb.symptoms[s]["possible_causes"]]
    for g in ents["disease"]:
        facts.append(kb.guidelines[g]["content"][:4])
    return any(f and f in answer for f in facts)


async def e1(config: str, n: int = 60, seed: int = 11) -> dict:
    rng = random.Random(seed)
    qs = queries(n, rng)
    full = config == "full"
    inj_s = LoggingInjector(dict(LLM_FAULTS), seed=seed)
    inj_l = LoggingInjector(dict(LLM_FAULTS), seed=seed + 1)
    inj_t = LoggingInjector(dict(TOOL_FAULTS), seed=seed + 2)
    pool = ModelPool({"strong": [FaultyProvider(MockProvider("strong"), inj_s)],
                      "local": [FaultyProvider(MockProvider("local"), inj_l)]},
                     retry=RetryPolicy(base_delay=0.001) if full else RetryPolicy(max_attempts={}),
                     breaker_cfg=BreakerConfig(min_calls=5, open_seconds=0.5) if full else BreakerConfig(min_calls=10**9),
                     call_timeout=2.0)
    if not full:
        orig = pool.generate

        async def no_cross(tier, messages, tools=None, *, events=None, allow_cross_tier=True, **kw):
            return await orig(tier, messages, tools, events=events, allow_cross_tier=False, **kw)
        pool.generate = no_cross  # type: ignore[assignment]
    ok = 0
    attr_total = attr_hit = 0
    lat = []
    async with MCPToolbox({"medical": build_server()}, fault_injector=inj_t,
                          retry=RetryPolicy(base_delay=0.001) if full else RetryPolicy(max_attempts={})) as tb:
        agent = MedicalAgent(pool, tb, SafetyFirstRouter(), CheckpointStore())
        if not full:
            agent.cfg.repair_attempts = 0

            async def passthrough(s):  # naive：不做校验修复与模板兜底
                s["findings"], s["node"] = [], "respond"
                return s
            agent._node_verify = passthrough  # type: ignore[assignment]
        for q in qs:
            before = len(inj_s.log) + len(inj_l.log) + len(inj_t.log)
            t0 = time.perf_counter()
            try:
                r = await agent.run(q)
                answer = r["answer"]
                attribution = r["attribution"]
            except AgentError as err:
                answer, attribution = "", {"kind": err.kind.value}
            lat.append((time.perf_counter() - t0) * 1000)
            if useful(answer, q):
                ok += 1
            injected = (inj_s.log + inj_l.log + inj_t.log)[before:]
            if injected and attribution and attribution.get("kind") not in (None, "ok"):
                attr_total += 1
                first = injected[0]
                expect = {"silent_empty": "silent", "tool_leak": "model"}.get(first, first)
                attr_hit += attribution["kind"] == expect
    return {"config": config, "n": n, "success_rate": round(ok / n, 4),
            "attribution_acc": round(attr_hit / attr_total, 4) if attr_total else None, "attributed_runs": attr_total,
            "latency_ms_mean": round(sum(lat) / len(lat), 1),
            "injected": {"llm": dict(inj_s.injected), "llm_local": dict(inj_l.injected), "tool": dict(inj_t.injected)}}


async def e2(full: bool, n: int = 20) -> dict:
    server = build_server()
    inj = FaultInjector({"lost_response": 0.4}, seed=5)
    async with MCPToolbox({"medical": server}, fault_injector=inj, retry=RetryPolicy(base_delay=0.001)) as tb:
        ok = 0
        for i in range(n):
            args = {"department": "内科", "date": "2026-11-20", "patient_ref": f"p{i}"}
            if full:
                r = await tb.call("book_appointment", args, scope=f"run{i}")       # 运行时注入稳定的幂等键
            else:
                r = None
                for attempt in range(3):  # naive：每次重试生成新 key（等于没有幂等）
                    r = await tb.call("book_appointment", {**args, "idempotency_key": f"naive-{i}-{attempt}-{random.random()}"},
                                      scope=f"run{i}:{attempt}")
                    if r.ok:
                        break
            ok += bool(r and r.ok)
    appts = list(server._medagent_state["appointments"].values())
    dup = len(appts) - len({a["patient_ref"] for a in appts})
    return {"config": "full(idempotency_key)" if full else "naive(new key per retry)", "requests": n,
            "succeeded": ok, "appointments_created": len(appts), "duplicate_bookings": dup,
            "lost_responses_injected": inj.injected.get("lost_response", 0)}


async def e3() -> dict:
    server = build_server()
    dead: list = []
    async with MCPToolbox({"medical": server}, retry=RetryPolicy(base_delay=0.001)) as tb:
        async def book(ctx):
            r = await tb.call("book_appointment", {"department": "心内科", "date": "2026-11-21", "patient_ref": "p-saga"},
                              scope="saga-1")
            if not r.ok:
                raise r.error
            return r.data

        async def cancel(ctx, appt):
            r = await tb.call("cancel_appointment", {"appointment_id": appt["appointment_id"]}, scope="saga-1-comp")
            if not r.ok:
                raise r.error

        async def notify(ctx):
            raise AgentError(ErrorKind.EXTERNAL, "SMS gateway 503", source="notify")

        res = await Saga("booking", [SagaStep("book", book, cancel), SagaStep("notify", notify)],
                         on_dead_letter=dead.append).run({})
        status = server._medagent_state["appointments"][res.log and res.results["book"]["appointment_id"]]["status"]

        async def broken_cancel(ctx, appt):
            raise AgentError(ErrorKind.EXTERNAL, "cancel API down", source="cancel")
        res2 = await Saga("booking2", [SagaStep("book", book, broken_cancel), SagaStep("notify", notify)],
                          on_dead_letter=dead.append).run({})
    return {"saga_ok": res.ok, "log": [f"{e.step}:{e.event}" for e in res.log], "appointment_status_after": status,
            "compensation_failure_log": [f"{e.step}:{e.event}" for e in res2.log], "dead_letters": len(dead)}


async def e4() -> dict:
    store = CheckpointStore()
    pool = ModelPool({"strong": [MockProvider("s")], "local": [MockProvider("l")]})
    out = {}
    async with MCPToolbox({"medical": build_server()}) as tb:
        agent = MedicalAgent(pool, tb, SafetyFirstRouter(), store)
        t = asyncio.create_task(agent.run("布洛芬和阿司匹林有什么副作用", run_id="pause-1"))
        await asyncio.sleep(0)
        agent.pause("pause-1")
        r1 = await t
        r2 = await agent.resume("pause-1")
        out["pause"] = {"first": r1["status"], "paused_at": r1.get("node"), "after_resume": r2["status"],
                        "answer_has_facts": "胃肠道" in r2["answer"]}
        # 崩溃：verify 节点第一次执行时抛异常（此前 work 节点已完成并落盘）
        crashed = {"n": 0}
        orig = agent._node_verify

        async def flaky(s):
            if crashed["n"] == 0:
                crashed["n"] += 1
                raise RuntimeError("simulated process crash")
            return await orig(s)
        agent._node_verify = flaky  # type: ignore[assignment]
        try:
            await agent.run("对乙酰氨基酚怎么吃", run_id="crash-1")
        except RuntimeError:
            pass
        st, _, status = store.load("crash-1")
        calls_before = sum(1 for e in st["events"] if e["stage"] == "tool")
        r3 = await agent.resume("crash-1")
        st2, _, _ = store.load("crash-1")
        calls_after = sum(1 for e in st2["events"] if e["stage"] == "tool")
        out["crash"] = {"status_at_crash": status, "node_at_crash": st["node"], "resumed": r3["status"],
                        "tool_calls_before": calls_before, "tool_calls_after_resume": calls_after,
                        "work_not_redone": calls_after == calls_before}
    return out


def e5() -> dict:
    store = CheckpointStore()
    store.create("occ", {"notes": []})
    s1, v1, _ = store.load("occ")
    s2, v2, _ = store.load("occ")
    store.save("occ", {"notes": ["A"]}, v1)
    try:
        store.save("occ", {"notes": ["B"]}, v2)
        lost_update = True
    except ConcurrentModification:
        lost_update = False
    update_with_retry(store, "occ", lambda st, status: ({"notes": st["notes"] + ["B"]}, status))
    final = store.load("occ")[0]
    return {"stale_write_rejected": not lost_update, "merged_after_retry": final["notes"]}


def e6() -> dict:
    import fakeredis
    server = fakeredis.FakeServer()
    r1, r2 = fakeredis.FakeRedis(server=server), fakeredis.FakeRedis(server=server)
    cfg = BreakerConfig(min_calls=4, failure_rate=0.5, open_seconds=30)
    a, b = CircuitBreaker("llm:baichuan", RedisBreakerStore(r1), cfg), CircuitBreaker("llm:baichuan", RedisBreakerStore(r2), cfg)
    for _ in range(4):
        a.on_failure(AgentError(ErrorKind.TIMEOUT, "t"), probe=False)
    shared = b.state()
    try:
        b.acquire()
        b_fast_fail = False
    except AgentError as e:
        b_fast_fail = e.kind == ErrorKind.CIRCUIT_OPEN

    class Down:
        def __getattr__(self, name):
            def f(*a, **k):
                raise ConnectionError("redis down")
            return f
    store = RedisBreakerStore(Down())
    c = CircuitBreaker("llm:deepseek", store, cfg)
    for _ in range(4):
        c.on_failure(AgentError(ErrorKind.NETWORK, "n"), probe=False)
    return {"instance_b_sees_open": shared, "instance_b_fast_fails": b_fast_fail,
            "redis_down_degraded_state": c.state(), "degraded_events": store.degraded_events}


async def e7(reps: int = 6) -> dict:
    """受控归因：每个 run 只在一个组件上注入一种故障（概率 1、不重试），检查归因是否判对“是谁的错”。"""
    cases = [("llm", k) for k in ("network", "timeout", "rate_limit", "external", "auth", "silent_empty", "tool_leak")] + \
            [("tool", k) for k in ("network", "external", "silent_empty")] + [("tool", "bad_args")]
    expect = {"silent_empty": "silent", "tool_leak": "model", "bad_args": "param"}
    confusion: dict = {}
    hit = total = 0
    for comp, kind in cases:
        for rep in range(reps):
            llm_inj = FaultInjector({kind: 1.0} if comp == "llm" and kind != "bad_args" else {}, seed=rep)
            tool_inj = FaultInjector({kind: 1.0} if comp == "tool" and kind != "bad_args" else {}, seed=rep)

            def bad_policy(messages, tools):
                from medagent.types import LLMResponse, ToolCall
                if not any(m["role"] == "tool" for m in messages):
                    return LLMResponse("", [ToolCall("check_interaction", {"drugs": ["布洛芬"]}, "x")], "local")
                return LLMResponse("查询失败，请咨询药师。", [], "local")
            local = MockProvider("local", policy=bad_policy) if kind == "bad_args" else MockProvider("local")
            pool = ModelPool({"local": [FaultyProvider(local, llm_inj)]}, retry=RetryPolicy(max_attempts={}))
            async with MCPToolbox({"medical": build_server()}, fault_injector=tool_inj, retry=RetryPolicy(max_attempts={})) as tb:
                agent = MedicalAgent(pool, tb, SafetyFirstRouter(), CheckpointStore())
                agent.cfg.repair_attempts = 0
                try:
                    r = await agent.run("布洛芬有什么副作用")
                    got = (r["attribution"] or {}).get("kind")
                    resp = (r["attribution"] or {}).get("responsible")
                except AgentError as err:
                    got, resp = err.kind.value, "?"
            want = expect.get(kind, kind)
            total += 1
            hit += got == want
            confusion.setdefault(f"{comp}:{kind}", {}).setdefault(f"{got}/{resp}", 0)
            confusion[f"{comp}:{kind}"][f"{got}/{resp}"] += 1
    return {"accuracy": round(hit / total, 4), "runs": total, "confusion(kind/responsible)": confusion}


async def main() -> None:
    res = {"E7_attribution": await e7(), "E1_naive": await e1("naive"), "E1_full": await e1("full"), "E2_naive": await e2(False), "E2_full": await e2(True),
           "E3_saga": await e3(), "E4_pause_crash": await e4(), "E5_occ": e5(), "E6_redis_breaker": e6()}
    tag = sys.argv[1] if len(sys.argv) > 1 else "it1"
    (ROOT / "evaluation" / "results" / f"chaos_eval_{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                                                      encoding="utf-8")
    for k, v in res.items():
        print(k, json.dumps(v, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
