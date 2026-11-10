"""复杂问诊：医疗工具 MCP Server 的第 8 个工具 complex_consult；worker 隔离、逐步检查点、中断续跑。"""
import json

import httpx

from medagent.agent import MedicalAgent
from medagent.consult import TOOL_NAME, ConsultRunner, default_run_id
from medagent.gateway import RoutingGateway, create_app
from medagent.mcp_client import MCPToolbox
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


# ---------------------------------------------------------------- 作为 MCP 工具：第 8 个工具
async def test_consult_is_eighth_mcp_tool_and_runs_orchestrator():
    pool, _, _ = mock_pool()
    async with MCPToolbox({"medical": build_server(consult=runner(pool))}) as tb:
        assert list(tb.tools)[-1] == "complex_consult" and len(tb.tools) == 8
        assert tb.tools["complex_consult"].read_only
        r = await tb.call("complex_consult", {"question": Q})
        assert r.ok and r.data["status"] == "done" and len(r.data["sub_questions"]) == 2
        st = await tb.read_resource(f"consult://run/{r.data['run_id']}")
        assert st["status"] == "done"


async def test_consult_disabled_server_has_seven_tools():
    async with MCPToolbox({"medical": build_server(consult=None)}) as tb:
        assert "complex_consult" not in tb.tools and len(tb.tools) == 7


async def test_orchestrator_workers_never_get_consult_tool():
    """编排器自己的 worker 连的是带 8 个工具的 Server 时，也看不到、调不了 complex_consult（防递归）。"""
    seen = []

    def policy(messages, tools):
        names = {t["function"]["name"] for t in tools or []}
        seen.append(names)
        if not any(m["role"] == "tool" for m in messages):
            return LLMResponse("", [ToolCall(TOOL_NAME, {"question": "x"}, "c1")], "local")
        return LLMResponse("已查询。仅供参考。", [], "local")

    pool, _, _ = mock_pool(policy)
    async with MCPToolbox({"medical": build_server(consult=runner(pool))}) as tb:
        r = await MedicalAgent(pool, tb, SafetyFirstRouter(), CheckpointStore()).run("布洛芬有什么副作用")
    assert all(TOOL_NAME not in names for names in seen if names)
    assert any(e.get("source") == f"tool:{TOOL_NAME}" and not e["ok"] for e in r["events"])   # 调用被拒绝


# ---------------------------------------------------------------- 网关：工具由 DSH 执行，网关只负责直接交付结果
DSH_TOOLS = [{"type": "function", "function": {"name": f"mcp__medical__{n}", "description": "",
                                                "parameters": {"type": "object", "properties": {}}}}
             for n in ("search_drug", "check_symptom", TOOL_NAME)]


def consult_caller(seen=None):
    """模拟会用复杂问诊的模型：问题是多意图时调用 complex_consult，否则按默认行为查普通工具。"""
    from medagent.verify.detectors import multi_intent

    def policy(messages, tools):
        names = {t["function"]["name"] for t in tools or []}
        user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        if seen is not None:
            seen.append(sorted(names))
        if TOOL_NAME in names and multi_intent(user) and not any(m["role"] == "tool" for m in messages):
            return LLMResponse("", [ToolCall(TOOL_NAME, {"question": user}, "c1")], "local")
        return _default_mock_policy("local", messages, tools)
    return policy


def gateway(policy=None):
    pool, local, strong = mock_pool(policy)
    return RoutingGateway(pool, SafetyFirstRouter()), local, strong


async def test_model_call_to_consult_is_returned_to_dsh():
    seen = []
    gw, _, _ = gateway(consult_caller(seen))
    resp, meta = await gw.complete({"messages": [{"role": "user", "content": Q}], "tools": DSH_TOOLS})
    calls = resp["choices"][0]["message"]["tool_calls"]
    assert calls[0]["function"]["name"] == "mcp__medical__complex_consult"   # 和其它工具一样交回 DSH 执行
    assert seen[0] == ["check_symptom", "complex_consult", "search_drug"]   # 模型看到的就是 DSH 给的工具（去前缀）
    assert meta["tier"] in ("local", "strong")


async def test_gateway_delivers_consult_result_directly():
    gw, local, strong = gateway(consult_caller())
    pool, _, _ = mock_pool()
    result = await runner(pool).run(Q)                                 # DSH 经 MCP 执行后的工具结果
    messages = [{"role": "user", "content": Q},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "mcp__medical__complex_consult", "arguments": json.dumps({"question": Q})}}]},
                {"role": "tool", "tool_call_id": "c1", "content": json.dumps(result, ensure_ascii=False)}]
    calls_before = local.calls + strong.calls
    resp, meta = await gw.complete({"messages": messages, "tools": DSH_TOOLS})
    assert meta["reason"] == "consult:answer" and local.calls + strong.calls == calls_before   # 不再调模型转述
    assert "胃肠道" in resp["choices"][0]["message"]["content"]


async def test_consult_error_result_lets_model_continue():
    gw, _, _ = gateway()
    messages = [{"role": "user", "content": Q},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "mcp__medical__complex_consult", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "Error executing tool complex_consult: EXTERNAL: 中断"}]
    resp, meta = await gw.complete({"messages": messages, "tools": DSH_TOOLS})
    assert meta["reason"] != "consult:answer"                          # 交给模型用普通工具继续


async def test_small_tier_never_gets_tools():
    from medagent.router.policy import TieredRouter
    seen = []
    pool, _, _ = mock_pool(consult_caller(seen))
    pool.tiers["small"] = [MockProvider("small", policy=consult_caller(seen))]
    gw = RoutingGateway(pool, TieredRouter())
    _, meta = await gw.complete({"messages": [{"role": "user", "content": "什么是BMI"}], "tools": DSH_TOOLS})
    assert meta["tier"] == "small" and seen == [[]]                    # 小模型档不带任何工具


async def test_stream_still_works():
    gw, _, _ = gateway()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(gw)), base_url="http://gw") as c:
        r = await c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "布洛芬有什么副作用"}],
                                                       "tools": DSH_TOOLS, "stream": True})
        assert "data: [DONE]" in r.text
