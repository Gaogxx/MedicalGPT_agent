import asyncio

import numpy as np
import pytest

from medagent.agent import MedicalAgent
from medagent.llm.parser import parse_tool_calls
from medagent.llm.pool import ModelPool
from medagent.llm.providers import FaultInjector, FaultyProvider, MockProvider
from medagent.mcp_client import MCPToolbox
from medagent.mcp_servers.medical_server import build_server
from medagent.rag.pipeline import build_default_pipeline
from medagent.resilience.checkpoint import CheckpointStore
from medagent.resilience.retry import RetryPolicy
from medagent.router.mf_router import MFRouter, calibrate_threshold
from medagent.router.policy import SafetyFirstRouter
from medagent.safety import assess, guard_output
from medagent.types import ErrorKind
from medagent.verify.attribution import attribute
from medagent.verify.detectors import detect


def test_parser_recovers_leaked_call():
    calls, rest = parse_tool_calls('ptom\n{"name": "check_symptom", "arguments": {"symptom": "心悸", "duration": ""}}}\n根据...')
    assert calls[0].name == "check_symptom" and calls[0].arguments == {"symptom": "心悸"}
    assert "ptom" not in rest and "{" not in rest
    calls, _ = parse_tool_calls('Action: search_drug\nAction Input: {"name": "布洛芬"}\n')
    assert calls[0].arguments["name"] == "布洛芬"


def test_safety_and_guard():
    a = assess("我胸口剧痛冒冷汗")
    assert a.level == "emergency" and a.requires_strong_model
    assert not assess("开心，吃了药好多了").high_risk
    hr = assess("帮我开个处方")
    out = guard_output("可以吃布洛芬 0.2g 一日3次", hr)
    assert "0.2g" not in out and "执业医师" in out
    assert assess("我之前问过呼吸困难等问题，你建议立即就医。 现在头痛怎么办").level == "normal"  # 只看当前句


def test_detectors_on_real_failure_patterns():
    codes = lambda ans, q, **kw: {f.code for f in detect(ans, query=q, assessment=assess(q), **kw)}  # noqa: E731
    assert "tool_leak" in codes('olith\n{"name": "check_symptom", "arguments": {}}', "我恶心")
    assert "unsupported_detail" in codes("您的症状是嗓子疼，持续了3天。", "我嗓子疼")
    assert "unsupported_detail" not in codes("您描述的症状（关节痛，早晨起来更明显）", "我关节痛，早上起来更明显")
    assert "emergency_missing" in codes("建议休息观察，持续超过3天请就医。", "我剧烈头痛还呕吐")
    assert "incomplete_multi" in codes("可能是关节炎，建议就医。", "我关节痛，先分析原因再查布洛芬")
    assert "fabricated_on_miss" in codes("该药一次10mg，一日3次。", "查一下神奇药丸", observations=[{"found": False, "query": "神奇药丸"}])


def test_attribution_param_blames_model():
    a = attribute([{"stage": "tool", "source": "tool:check_interaction", "ok": False, "args_from_model": True,
                    "error": {"kind": "param", "message": "drugs 至少两个"}}])
    assert a.kind == "param" and a.responsible == "model"
    a = attribute([{"stage": "llm", "source": "llm:x", "ok": False, "error": {"kind": "circuit_open"}},
                   {"stage": "llm", "source": "llm:y", "ok": False, "error": {"kind": "timeout"}}])
    assert a.kind == "timeout"   # 熔断是结果不是根因


def test_mf_router_learns_and_calibrates():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8))
    y = (X[:, 0] + 0.3 * rng.normal(size=200) > 0).astype(float)
    m = MFRouter(dim=8, epochs=300)
    m.fit(X[:150], y[:150], X[150:], y[150:])
    s = m.predict_strong_win(X[150:])
    acc = ((s > 0.5) == y[150:]).mean()
    assert acc > 0.8
    thr = calibrate_threshold(s, 0.3)
    assert abs((s >= thr).mean() - 0.3) < 0.05


def test_rag_pipeline_kg_and_cache():
    p = build_default_pipeline()
    r = p.search("美林有什么副作用", top_k=3)
    assert r.docs[0].id == "kb:drug:布洛芬"
    assert p.search("美林有什么副作用？", top_k=3).from_cache


async def test_mcp_tools_and_idempotent_booking():
    server = build_server()
    async with MCPToolbox({"medical": server}) as tb:
        assert {"check_symptom", "search_drug", "check_interaction", "book_appointment"} <= set(tb.tools)
        r = await tb.call("check_interaction", {"drugs": ["布洛芬"]})
        assert not r.ok and r.error.kind == ErrorKind.PARAM
        r = await tb.call("nonexistent_tool", {})
        assert r.error.kind == ErrorKind.MODEL
        args = {"department": "内科", "date": "2026-12-01", "patient_ref": "p1"}
        a = await tb.call("book_appointment", args, scope="run-1")
        b = await tb.call("book_appointment", args, scope="run-1")
        assert a.ok and b.ok and a.data["appointment_id"] == b.data["appointment_id"]
    assert len(server._medagent_state["appointments"]) == 1


async def test_lost_response_does_not_double_book():
    server = build_server()
    inj = FaultInjector({"lost_response": 1.0}, seed=0)
    async with MCPToolbox({"medical": server}, fault_injector=inj, retry=RetryPolicy(base_delay=0)) as tb:
        tb.fault_injector.rates = {"lost_response": 0.5}
        for i in range(6):
            await tb.call("book_appointment", {"department": "内科", "date": "2026-12-01", "patient_ref": f"p{i}"},
                          scope=f"r{i}")
    refs = [a["patient_ref"] for a in server._medagent_state["appointments"].values()]
    assert len(refs) == len(set(refs))


async def test_agent_pause_resume_and_failover():
    store = CheckpointStore()
    dead = FaultyProvider(MockProvider("local"), FaultInjector({"auth": 1.0}))
    pool = ModelPool({"strong": [MockProvider("strong")], "local": [dead]}, retry=RetryPolicy(base_delay=0))
    async with MCPToolbox({"medical": build_server()}) as tb:
        agent = MedicalAgent(pool, tb, SafetyFirstRouter(), store)
        r = await agent.run("布洛芬有什么副作用")          # local 鉴权失败 → 自动跨层到 strong
        assert r["status"] == "done" and "胃肠道" in r["answer"]
        assert any(e.get("error", {}) and e["error"]["kind"] == "auth" for e in r["events"] if e.get("error"))
        t = asyncio.create_task(agent.run("对乙酰氨基酚怎么吃", run_id="p"))
        await asyncio.sleep(0)
        agent.pause("p")
        first = await t
        assert first["status"] == "paused"
        done = await agent.resume("p")
        assert done["status"] == "done" and "0.5g" in done["answer"]


async def test_mcp_resources_prompts_and_symptom_lookup():
    async with MCPToolbox({"medical": build_server()}) as tb:
        assert "kb://drug/{name}" in tb.resources["medical"]
        assert {"medical_system_prompt", "triage_clarify", "answer_with_evidence"} <= set(tb.prompts["medical"])
        drug = await tb.read_resource("kb://drug/美林")
        assert drug["key"] == "布洛芬" and drug["provenance"]["reviewed"] is False
        assert await tb.read_resource("kb://drug/不存在的药") is None
        assert "急症" in (await tb.get_prompt("medical_system_prompt"))
        r = await tb.call("search_drug", {"name": "头痛"})          # 原 index.ts 的症状反查（迁移时曾丢失，已恢复）
        assert r.ok and "对乙酰氨基酚" in r.data["related_drugs"]


def test_kb_build_is_reproducible():
    from medagent.knowledge.build_kb import build
    kb, report = build("test")
    assert report["counts"]["symptoms"] == {"A": 10, "B": 20, "merged": 21}
    assert [c["key"] for c in report["conflicts"]] == ["头晕"]
