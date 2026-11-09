"""网关测试：用一个模拟 vLLM（OpenAI 兼容 HTTP）复现线上 MedicalGPT 的行为，走真实的 HTTP 调用路径。"""
import json

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from medagent.gateway import RoutingGateway, create_app
from medagent.llm.pool import ModelPool
from medagent.llm.providers import FaultInjector, FaultyProvider, MockProvider, OpenAICompatProvider
from medagent.resilience.retry import RetryPolicy
from medagent.router.policy import SafetyFirstRouter
from medagent.safety import EMERGENCY_BANNER

SEEN: dict = {}


def fake_vllm(behavior: str) -> Starlette:
    async def chat(request: Request) -> JSONResponse:
        body = await request.json()
        SEEN["tool_names"] = [t["function"]["name"] for t in body.get("tools") or []]
        has_tool_result = any(m["role"] == "tool" for m in body["messages"])
        msg: dict = {"role": "assistant", "content": ""}
        if behavior == "tool" and not has_tool_result:
            msg["tool_calls"] = [{"id": "c1", "type": "function",
                                  "function": {"name": "search_drug", "arguments": json.dumps({"name": "布洛芬"})}}]
        elif behavior == "leak" and not has_tool_result:
            msg["content"] = 'ptom\n{"name": "check_symptom", "arguments": {"symptom": "心悸", "duration": ""}}}'
        elif behavior == "empty":
            msg["content"] = ""
        else:
            msg["content"] = "布洛芬常见不良反应有胃肠道不适、恶心、头晕。"
        return JSONResponse({"choices": [{"message": msg, "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop"}],
                             "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])


def local_provider(behavior: str) -> OpenAICompatProvider:
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=fake_vllm(behavior)), base_url="http://vllm")
    return OpenAICompatProvider("medicalgpt-7b", "http://127.0.0.1:8000/v1", "medical-agent", client=client)


DSH_TOOLS = [{"type": "function", "function": {"name": f"mcp__medical__{n}", "description": "",
                                                "parameters": {"type": "object", "properties": {}}}}
             for n in ("check_symptom", "search_drug", "search_guideline")]


async def call(gw: RoutingGateway, body: dict) -> httpx.Response:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(gw)), base_url="http://gw") as c:
        return await c.post("/v1/chat/completions", json=body)


def gw(behavior: str, strong=None) -> RoutingGateway:
    pool = ModelPool({"local": [local_provider(behavior)], "strong": [strong or MockProvider("strong")]},
                     retry=RetryPolicy(base_delay=0))
    return RoutingGateway(pool, SafetyFirstRouter())


async def test_namespace_mapping_for_finetuned_model():
    r = await call(gw("tool"), {"model": "medical-router", "messages": [{"role": "user", "content": "布洛芬有什么副作用"}],
                                "tools": DSH_TOOLS})
    assert SEEN["tool_names"] == ["check_symptom", "search_drug", "search_guideline"]   # 发给 MedicalGPT 的是训练时的名字
    tc = r.json()["choices"][0]["message"]["tool_calls"][0]
    assert tc["function"]["name"] == "mcp__medical__search_drug"                          # 还给 DSH 的是 MCP 名字
    assert r.headers["x-medagent-tier"] == "local"


async def test_leaked_tool_call_is_converted():
    r = await call(gw("leak"), {"messages": [{"role": "user", "content": "我心跳有点快"}], "tools": DSH_TOOLS})
    body = r.json()
    msg = body["choices"][0]["message"]
    assert msg["tool_calls"][0]["function"]["name"] == "mcp__medical__check_symptom"
    assert "ptom" not in (msg["content"] or "") and "leaked_tool_call_converted" in body["medagent"]["repairs"]


async def test_empty_local_answer_regenerated_on_strong():
    r = await call(gw("empty"), {"messages": [{"role": "user", "content": "布洛芬有什么副作用"}]})
    body = r.json()
    assert body["medagent"]["served_tier"] == "strong" and body["choices"][0]["message"]["content"]
    assert any(x.startswith("regenerate_strong") for x in body["medagent"]["repairs"])


async def test_emergency_stream_sends_banner_first():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(gw("text"))), base_url="http://gw") as c:
        r = await c.post("/v1/chat/completions", json={"stream": True, "messages": [{"role": "user", "content": "我胸口剧痛冒冷汗"}]})
    chunks = [json.loads(l[6:]) for l in r.text.splitlines() if l.startswith("data: {")]
    first_content = next(c["choices"][0]["delta"]["content"] for c in chunks if c["choices"][0]["delta"].get("content"))
    assert first_content.startswith(EMERGENCY_BANNER)
    assert r.text.strip().endswith("data: [DONE]")


async def test_all_models_down_returns_safe_fallback():
    dead = FaultyProvider(MockProvider("strong"), FaultInjector({"auth": 1.0}))
    g = gw("empty", strong=dead)
    g.pool.tiers["local"] = [FaultyProvider(MockProvider("local"), FaultInjector({"network": 1.0}))]
    r = await call(g, {"messages": [{"role": "user", "content": "布洛芬有什么副作用"}]})
    content = r.json()["choices"][0]["message"]["content"]
    assert r.status_code == 200 and content.startswith("【降级模式】") and "布洛芬" in content   # 降级：知识库原文 + 标注
    assert r.json()["medagent"]["error"]["kind"] in ("auth", "network")


async def test_degraded_mode_withholds_dose_for_pregnancy_and_emergency():
    dead = FaultyProvider(MockProvider("x"), FaultInjector({"network": 1.0}))
    g = gw("empty", strong=dead)
    g.pool.tiers["local"] = [dead]
    r = await call(g, {"messages": [{"role": "user", "content": "孕妇可以吃布洛芬吗"}]})
    c = r.json()["choices"][0]["message"]["content"]
    assert "0.2g" not in c and "个体化判断" in c
    r = await call(g, {"messages": [{"role": "user", "content": "我胸口剧痛冒冷汗"}]})
    assert r.json()["choices"][0]["message"]["content"].startswith(EMERGENCY_BANNER)


async def test_small_tier_routing_and_escalation():
    from medagent.router.policy import TieredRouter
    small = MockProvider("small", policy=lambda m, t: __import__("medagent.types", fromlist=["LLMResponse"]).LLMResponse("", [], "small"))
    pool = ModelPool({"small": [small], "local": [local_provider("text")], "strong": [MockProvider("strong")]},
                     retry=RetryPolicy(base_delay=0))
    g = RoutingGateway(pool, TieredRouter())
    r = await call(g, {"messages": [{"role": "user", "content": "什么是BMI"}], "tools": DSH_TOOLS})
    meta = r.json()["medagent"]
    assert meta["tier"] == "small" and meta["served_tier"] == "local"          # 小模型空回复 → 升级 local
    assert any(x.startswith("escalate_local") for x in meta["repairs"])
    r = await call(g, {"messages": [{"role": "user", "content": "布洛芬的适应症是什么"}], "tools": DSH_TOOLS})
    assert r.json()["medagent"]["tier"] == "local"                               # 涉及药品不给小模型
