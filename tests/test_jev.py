"""Jev 适配层契约测试：用按同一契约实现的假 Jev 服务跑通 校验 / 归因 / 降级 三条路径。"""
from starlette.testclient import TestClient

from medagent.safety import assess
from medagent.verify.detectors import detect
from medagent.verify.jev import ATTRIBUTION, SILENT_FAILURE, JevClient, JevVerifier, fake_jev_app


def verifier(**kw) -> JevVerifier:
    return JevVerifier(JevClient(base_url="http://testserver", client=TestClient(fake_jev_app())), **kw)


def test_contract_shapes():
    body = JevClient._build(ATTRIBUTION, {"events": []})
    assert body["primitive"] == "choice" and "network" in body["options"] and body["decision"] == "failure_attribution"
    assert JevClient._build(SILENT_FAILURE, {})["primitive"] == "noul"


def test_rules_block_without_calling_jev():
    v = verifier()
    q, a = "我恶心", 'olith\n{"name": "check_symptom", "arguments": {}}'
    verdict = v.verify(q, a, detect(a, query=q, assessment=assess(q)))
    assert verdict.blocked and verdict.source == "rules" and v.client.calls == 0


def test_jev_catches_what_rules_miss():
    v = verifier()
    q, a = "对乙酰氨基酚怎么吃", "据我所知，一次吃两片就行。"   # 规则没命中：没有工具结果可比对剂量
    verdict = v.verify(q, a, detect(a, query=q, assessment=assess(q)), observations=None)
    assert verdict.blocked and verdict.source == "jev" and verdict.confidence >= 0.8


def test_jev_unavailable_degrades_to_rules():
    v = JevVerifier(JevClient(base_url="http://127.0.0.1:9", timeout=0.2))
    verdict = v.verify("布洛芬有什么副作用", "胃肠道不适。", [])
    assert not verdict.blocked and verdict.source == "rules(jev_unavailable)" and v.stats["jev_error"] == 1


def test_attribution_uses_jev_only_when_ambiguous():
    v = verifier()
    clear = [{"stage": "llm", "source": "llm:x", "ok": False, "error": {"kind": "timeout"}}]
    assert v.attribute(clear)["source"] == "rules"
    ambiguous = [{"stage": "tool", "source": "tool:search_drug", "ok": True},
                 {"stage": "llm", "source": "llm:local", "ok": True},
                 {"stage": "verify", "source": "verifier", "ok": True,
                  "findings": [{"code": "ungrounded_dose", "kind": "silent", "severity": "block"}]}]
    out = v.attribute(ambiguous)
    assert out["source"] == "jev" and out["kind"] == "model" and out["responsible"] == "model"
