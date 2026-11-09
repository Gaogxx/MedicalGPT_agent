"""“复杂问诊”MCP 工具：编排器包成工具后，DSH 模式也有 worker 隔离、逐步检查点、中断续跑。"""
import json

import pytest

from medagent.agent import MedicalAgent
from medagent.gateway import RoutingGateway
from medagent.llm.pool import ModelPool
from medagent.llm.providers import MockProvider
from medagent.mcp_client import MCPToolbox
from medagent.mcp_servers.consult_server import build_consult_server, default_run_id
from medagent.mcp_servers.medical_server import build_server
from medagent.resilience.checkpoint import CheckpointStore
from medagent.resilience.retry import RetryPolicy
from medagent.router.policy import SafetyFirstRouter

Q = "先说说布洛芬的副作用，再说说高血压饮食注意什么"


def mock_pool():
    local, strong = MockProvider("local"), MockProvider("strong")
    return ModelPool({"local": [local], "strong": [strong]}, retry=RetryPolicy(base_delay=0)), local, strong


async def test_consult_splits_into_isolated_workers_and_replays():
    pool, local, strong = mock_pool()
    store = CheckpointStore()
    server = build_consult_server(lambda: pool, store)
    async with MCPToolbox({"consult": server}) as tb:
        assert tb.tools["complex_consult"].read_only
        r = await tb.call("complex_consult", {"question": Q})
        assert r.ok, r.error
        d = r.data
        assert d["status"] == "done" and d["run_id"] == default_run_id(Q)
        assert len(d["sub_questions"]) == 2
        assert d["workers"]["w0"]["tools"] == ["search_drug"]          # 每个 worker 只查自己那一个子问题
        assert d["workers"]["w1"]["tools"] == ["search_guideline"]
        assert "胃肠道" in d["answer"] and "高血压" in d["answer"]
        calls = local.calls + strong.calls
        again = await tb.call("complex_consult", {"question": Q})      # 重放：直接返回检查点里的答案，不再调模型
        assert again.data["replayed"] and again.data["answer"] == d["answer"]
        assert local.calls + strong.calls == calls
        state = await tb.read_resource(f"consult://run/{d['run_id']}")
        assert state["status"] == "done" and state["node"] == "done"


async def test_consult_resumes_from_checkpoint_after_crash(monkeypatch):
    pool, local, strong = mock_pool()
    store = CheckpointStore()
    server = build_consult_server(lambda: pool, store)
    original = MedicalAgent._node_aggregate
    crashed = {"n": 0}

    async def crash_once(self, s):
        if not crashed["n"]:
            crashed["n"] += 1
            raise RuntimeError("worker host died")
        return await original(self, s)

    monkeypatch.setattr(MedicalAgent, "_node_aggregate", crash_once)
    # 模拟 DSH 的 MCP 客户端：不自动重试（本仓库 MCPToolbox 默认会对 EXTERNAL 重试一次，那样第一次调用就已经续跑完成）
    async with MCPToolbox({"consult": server}, retry=RetryPolicy(max_attempts={}, base_delay=0)) as tb:
        first = await tb.call("complex_consult", {"question": Q})
        assert not first.ok and first.error.kind.value == "external"
        state, _, _ = store.load(default_run_id(Q))
        assert state["node"] == "aggregate" and all(w["status"] == "done" for w in state["workers"].values())
        calls = local.calls + strong.calls
        second = await tb.call("complex_consult", {"question": Q})     # 相同参数重试 → 同一 run_id → 续跑
        assert second.ok and second.data["resumed"] and second.data["status"] == "done"
        assert local.calls + strong.calls == calls                      # 两个 worker 都没有重跑


async def test_paused_consult_waits_for_explicit_resume():
    pool, _, _ = mock_pool()
    store = CheckpointStore()
    rid = default_run_id(Q)
    async with MCPToolbox({"medical": build_server()}) as med:
        agent = MedicalAgent(pool, med, SafetyFirstRouter(), store)
        store.create(rid, {"run_id": rid, "query": Q, "history": [], "node": "intake", "events": [], "workers": {},
                           "answer": "", "tier": None, "route": None, "fastpath": False, "findings": [], "repairs": [],
                           "attribution": None, "started": 0})
        store.request_pause(rid)
        del agent
    server = build_consult_server(lambda: pool, store)
    async with MCPToolbox({"consult": server}) as tb:
        r = await tb.call("complex_consult", {"question": Q})
        assert r.data["status"] == "paused" and "resume" in r.data["hint"]
        r = await tb.call("complex_consult", {"question": Q, "resume": True})
        assert r.data["status"] == "done" and r.data["resumed"]


async def test_consult_worker_cannot_call_side_effect_tools():
    from medagent.types import LLMResponse, ToolCall

    def booker(messages, tools):
        names = {t["function"]["name"] for t in tools or []}
        assert "book_appointment" not in names
        if not any(m["role"] == "tool" for m in messages):
            return LLMResponse("", [ToolCall("book_appointment", {"department": "内科", "date": "2026-12-01",
                                                                  "patient_ref": "x"}, "c1")], "local")
        return LLMResponse("已为您查询。仅供参考。", [], "local")

    pool = ModelPool({"local": [MockProvider("local", policy=booker)], "strong": [MockProvider("local", policy=booker)]},
                     retry=RetryPolicy(base_delay=0))
    med = build_server()
    server = build_consult_server(lambda: pool, CheckpointStore(), medical_server=med)
    async with MCPToolbox({"consult": server}) as tb:
        r = await tb.call("complex_consult", {"question": Q})
        assert r.ok
    assert med._medagent_state["appointments"] == {}


# ---------------------------------------------------------------- 网关：确定性触发、直接交付、出错续跑
DSH_TOOLS = [{"type": "function", "function": {"name": n, "description": "", "parameters": {"type": "object", "properties": {}}}}
             for n in ("mcp__medical__search_drug", "mcp__medical_consult__complex_consult")]


def gateway():
    pool, local, strong = mock_pool()
    return RoutingGateway(pool, SafetyFirstRouter()), local, strong


def consult_turn(result_content: str, n: int = 1):
    msgs = [{"role": "user", "content": Q}]
    for i in range(n):
        msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {
                     "name": "mcp__medical_consult__complex_consult", "arguments": json.dumps({"question": Q, "history": ""})}}]},
                 {"role": "tool", "tool_call_id": f"c{i}", "content": result_content}]
    return msgs


async def test_gateway_triggers_consult_for_multi_intent_without_model_call():
    gw, local, strong = gateway()
    resp, meta = await gw.complete({"messages": [{"role": "user", "content": Q}], "tools": DSH_TOOLS})
    call = resp["choices"][0]["message"]["tool_calls"][0]["function"]
    assert call["name"] == "mcp__medical_consult__complex_consult"
    assert json.loads(call["arguments"])["question"] == Q
    assert meta["reason"] == "consult:multi_intent" and local.calls + strong.calls == 0
    single, meta = await gw.complete({"messages": [{"role": "user", "content": "布洛芬有什么副作用"}], "tools": DSH_TOOLS})
    assert meta["reason"] != "consult:multi_intent"                  # 单一问题照常路由到模型


async def test_gateway_delivers_consult_answer_and_retries_once_on_error():
    gw, local, strong = gateway()
    done = json.dumps({"status": "done", "answer": "【布洛芬】胃肠道不适。\n\n【高血压饮食】低盐。仅供参考。", "run_id": "cs-x"})
    resp, meta = await gw.complete({"messages": consult_turn(done), "tools": DSH_TOOLS})
    assert "低盐" in resp["choices"][0]["message"]["content"] and meta["reason"] == "consult:answer"
    assert local.calls + strong.calls == 0                            # 不让模型再转述一遍
    err = "Error executing tool complex_consult: EXTERNAL: 复杂问诊中断"
    resp, meta = await gw.complete({"messages": consult_turn(err), "tools": DSH_TOOLS})
    assert meta["reason"] == "consult:resume_after_error"
    assert json.loads(resp["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])["question"] == Q
    resp, meta = await gw.complete({"messages": consult_turn(err, n=2), "tools": DSH_TOOLS})
    assert meta["reason"].startswith("router:") and local.calls + strong.calls >= 1   # 两次都失败：交给模型用普通工具


async def test_emergency_skips_consult():
    gw, _, _ = gateway()
    _, meta = await gw.complete({"messages": [{"role": "user", "content": "我胸口剧痛冒冷汗，先说怎么办再说吃什么药"}],
                                 "tools": DSH_TOOLS})
    assert meta["reason"].startswith("emergency")


# ---------------------------------------------------------------- 资源：知识源状态、检索命中文档全文
async def test_kb_sources_and_doc_resources(tmp_path, monkeypatch):
    from medagent.rag.scalable import FTS5Index
    path = tmp_path / "mini.fts5"
    filler = [(f"f:{i}", "测试来源", f"条目{i}", f"第{i}号无关条目，记录天气与交通。") for i in range(60)]  # 检索会忽略 df>5% 的词
    FTS5Index(path).build([("a:1", "测试来源", "布洛芬是什么", "布洛芬是一种非甾体抗炎药，可缓解轻中度疼痛。"),
                           ("a:2", "测试来源", "高血压饮食", "高血压患者应限制钠盐摄入。")] + filler, workers=1)
    monkeypatch.setenv("MEDAGENT_FTS5_INDEX", str(path))
    async with MCPToolbox({"medical": build_server()}) as tb:
        src = await tb.read_resource("kb://sources")
        assert src["local_fulltext"]["available"] and src["curated"]["drugs"] >= 10
        r = await tb.call("search_knowledge", {"query": "非甾体抗炎药缓解疼痛", "top_k": 10})
        ids = [e["id"] for e in r.data["evidence"] if e["id"].startswith("fts5:")]
        assert ids, r.data
        doc = await tb.read_resource(f"kb://doc/{ids[0]}")
        assert doc["text"].startswith("布洛芬是一种") and doc["origin_id"] == "a:1"
        assert await tb.read_resource("kb://doc/fts5:999999") is None
