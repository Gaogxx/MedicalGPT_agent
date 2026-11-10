"""复杂问诊：编排器在网关进程内运行，由规则或模型按需触发；worker 隔离、逐步检查点、中断续跑。"""
import json

import httpx

from medagent.agent import MedicalAgent
from medagent.consult import TOOL_NAME, ConsultRunner, default_run_id
from medagent.gateway import RoutingGateway, create_app
from medagent.llm.pool import ModelPool
from medagent.llm.providers import MockProvider, _default_mock_policy
from medagent.mcp_servers.medical_server import build_server
from medagent.resilience.checkpoint import CheckpointStore
from medagent.resilience.retry import RetryPolicy
from medagent.router.policy import SafetyFirstRouter
from medagent.types import LLMResponse, ToolCall

Q = "先说说布洛芬的副作用，再说说高血压饮食注意什么"


def mock_pool(policy=None):
    local, strong = MockProvider("local", policy=policy), MockProvider("strong", policy=policy)
    return ModelPool({"local": [local], "strong": [strong]}, retry=RetryPolicy(base_delay=0)), local, strong


def runner(pool, store=None, **kw):
    return ConsultRunner(pool, SafetyFirstRouter(), store or CheckpointStore(), **kw)


async def test_consult_splits_into_isolated_workers_and_replays():
    pool, local, strong = mock_pool()
    r = runner(pool)
    d = await r.run(Q)
    assert d["status"] == "done" and d["run_id"] == default_run_id(Q)
    assert len(d["sub_questions"]) == 2
    assert d["workers"]["w0"]["tools"] == ["search_drug"]          # 每个 worker 只查自己那一个子问题
    assert d["workers"]["w1"]["tools"] == ["search_guideline"]
    assert "胃肠道" in d["answer"] and "高血压" in d["answer"]
    calls = local.calls + strong.calls
    again = await r.run(Q)                                           # 重放：直接返回检查点里的答案，不再调模型
    assert again["replayed"] and again["answer"] == d["answer"] and local.calls + strong.calls == calls
    assert r.status(d["run_id"])["node"] == "done"


async def test_consult_resumes_from_checkpoint_after_crash(monkeypatch):
    pool, local, strong = mock_pool()
    store = CheckpointStore()
    original = MedicalAgent._node_aggregate
    crashed = {"n": 0}

    async def crash_once(self, s):
        if not crashed["n"]:
            crashed["n"] += 1
            raise RuntimeError("gateway process died")
        return await original(self, s)

    monkeypatch.setattr(MedicalAgent, "_node_aggregate", crash_once)
    first = runner(pool, store, max_attempts=1)                      # 模拟网关进程在这一步崩溃
    try:
        await first.run(Q)
        raise AssertionError("should have failed")
    except Exception as exc:  # noqa: BLE001
        assert "中断" in str(exc)
    state, _, _ = store.load(default_run_id(Q))
    assert state["node"] == "aggregate" and all(w["status"] == "done" for w in state["workers"].values())
    calls = local.calls + strong.calls
    second = await runner(pool, store).run(Q)                        # 重启后 DSH 重发同一请求 → 同一 run_id → 续跑
    assert second["resumed"] and second["status"] == "done"
    assert local.calls + strong.calls == calls                       # 两个 worker 都没有重跑


async def test_consult_retries_in_process_once(monkeypatch):
    pool, _, _ = mock_pool()
    original = MedicalAgent._node_verify
    crashed = {"n": 0}

    async def crash_once(self, s):
        if not crashed["n"]:
            crashed["n"] += 1
            raise RuntimeError("transient")
        return await original(self, s)

    monkeypatch.setattr(MedicalAgent, "_node_verify", crash_once)
    d = await runner(pool).run(Q)                                    # 默认 max_attempts=2：同一进程里自动续跑一次
    assert d["status"] == "done" and d["resumed"]


async def test_paused_consult_waits_for_explicit_resume():
    pool, _, _ = mock_pool()
    store = CheckpointStore()
    rid = default_run_id(Q)
    store.create(rid, {"run_id": rid, "query": Q, "history": [], "node": "intake", "events": [], "workers": {},
                       "answer": "", "tier": None, "route": None, "fastpath": False, "findings": [], "repairs": [],
                       "attribution": None, "started": 0})
    store.request_pause(rid)
    r = runner(pool, store)
    assert (await r.run(Q))["status"] == "paused"
    d = await r.run(Q, resume=True)
    assert d["status"] == "done" and d["resumed"]


async def test_consult_worker_cannot_call_side_effect_tools():
    def booker(messages, tools):
        names = {t["function"]["name"] for t in tools or []}
        assert "book_appointment" not in names
        if not any(m["role"] == "tool" for m in messages):
            return LLMResponse("", [ToolCall("book_appointment", {"department": "内科", "date": "2026-12-01",
                                                                  "patient_ref": "x"}, "c1")], "local")
        return LLMResponse("已为您查询。仅供参考。", [], "local")

    pool, _, _ = mock_pool(booker)
    med = build_server()
    d = await runner(pool, medical=med).run(Q)
    assert d["status"] == "done"
    assert med._medagent_state["appointments"] == {}


# ---------------------------------------------------------------- 网关：规则触发、模型按需调用、失败退回
DSH_TOOLS = [{"type": "function", "function": {"name": "mcp__medical__search_drug", "description": "",
                                                "parameters": {"type": "object", "properties": {}}}}]


def gateway(policy=None):
    pool, local, strong = mock_pool(policy)
    gw = RoutingGateway(pool, SafetyFirstRouter(), consult=runner(pool))
    return gw, local, strong


async def test_rule_trigger_runs_consult_inside_gateway():
    gw, local, strong = gateway()
    resp, meta = await gw.complete({"messages": [{"role": "user", "content": Q}], "tools": DSH_TOOLS})
    msg = resp["choices"][0]["message"]
    assert meta["reason"] == "consult:multi_intent" and "tool_calls" not in msg   # DSH 收到的就是最终回答
    assert "胃肠道" in msg["content"] and "高血压" in msg["content"]
    assert len(meta["consult"]["sub_questions"]) == 2
    assert local.calls + strong.calls >= 2                            # 模型在编排器的 worker 里被调用


async def test_model_can_call_internal_consult_tool_and_dsh_never_sees_it():
    seen = {}

    def policy(messages, tools):
        names = {t["function"]["name"] for t in tools or []}
        user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        if TOOL_NAME in names:
            seen["tools"] = sorted(names)
            return LLMResponse("", [ToolCall(TOOL_NAME, {"question": user}, "c1")], "local")
        return _default_mock_policy("local", messages, tools)

    gw, _, _ = gateway(policy)
    query = "布洛芬和对乙酰氨基酚有什么区别，各自要注意什么"
    resp, meta = await gw.complete({"messages": [{"role": "user", "content": query}], "tools": DSH_TOOLS})
    assert seen["tools"] == ["complex_consult", "search_drug"]       # 模型看到内部工具 + DSH 的工具（已去前缀）
    assert meta["reason"] == "consult:model_call"
    assert "tool_calls" not in resp["choices"][0]["message"]


async def test_consult_failure_falls_back_to_normal_path(monkeypatch):
    async def boom(self, *a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(MedicalAgent, "_node_plan", boom)
    gw, _, _ = gateway()
    resp, meta = await gw.complete({"messages": [{"role": "user", "content": Q}], "tools": DSH_TOOLS})
    assert any(r.startswith("consult_failed") for r in meta["repairs"])
    assert meta["reason"].startswith("router:")                      # 退回普通单 agent 路径继续回答


async def test_emergency_skips_consult():
    gw, _, _ = gateway()
    _, meta = await gw.complete({"messages": [{"role": "user", "content": "我胸口剧痛冒冷汗，先说怎么办再说吃什么药"}],
                                 "tools": DSH_TOOLS})
    assert meta["reason"].startswith("emergency")


async def test_consult_progress_endpoint_and_stream():
    gw, _, _ = gateway()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(gw)), base_url="http://gw") as c:
        r = await c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": Q}],
                                                       "tools": DSH_TOOLS, "stream": True})
        content = "".join(json.loads(line[6:])["choices"][0]["delta"].get("content", "")
                          for line in r.text.splitlines() if line.startswith("data: {"))
        assert "胃肠道" in content
        st = await c.get(f"/consult/{default_run_id(Q)}")
        assert st.status_code == 200 and st.json()["status"] == "done"
        assert (await c.get("/consult/cs-none")).status_code == 404
