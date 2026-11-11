"""开源判别模型（微调 Laya，替代 Jev）：用按 laya-serve 接口实现的假服务跑通 校验 / 降级 / 归因 / 失败类别，
并检查训练数据与线上请求用的是同一份问题定义和状态格式。"""
import sys
from pathlib import Path

from starlette.testclient import TestClient

from medagent.safety import assess
from medagent.verify.detectors import detect
from medagent.verify.discriminator import (FAILURE_TYPE, FAILURE_TYPES, QUESTIONS, DiscriminatorClient, fake_laya_app,
                                           laya_state)
from medagent.verify.typed import ANSWER_QUALITY, SILENT_FAILURE, TypedVerifier, answer_state

ROOT = Path(__file__).resolve().parents[1]


def client(judge=None, **kw) -> DiscriminatorClient:
    return DiscriminatorClient(base_url="http://testserver", client=TestClient(fake_laya_app(judge)), **kw)


def verifier(judge=None, **kw) -> TypedVerifier:
    return TypedVerifier(client(judge), **kw)


def test_rules_block_without_calling_discriminator():
    v = verifier()
    q, a = "我恶心", 'olith\n{"name": "check_symptom", "arguments": {}}'
    verdict = v.verify(q, a, detect(a, query=q, assessment=assess(q)))
    assert verdict.blocked and verdict.source == "rules" and v.client.calls == 0


def test_discriminator_catches_what_rules_miss_with_failure_type():
    v = verifier()
    q, a = "对乙酰氨基酚怎么吃", "据我所知，一次吃两片就行。"   # 规则没命中：没有工具结果可比对剂量
    verdict = v.verify(q, a, detect(a, query=q, assessment=assess(q)), observations=None)
    assert verdict.blocked and verdict.source == "discriminator" and verdict.confidence >= 0.8
    assert verdict.reason == "discriminator:silent_failure:unsafe_recommendation"
    assert v.client.calls == 1                                   # 三个问题一次请求


def test_good_answer_passes_and_quality_shares_one_request():
    v = verifier(score_quality=True)
    verdict = v.verify("布洛芬有什么副作用", "常见胃肠道不适，建议饭后服用。仅供参考。", [])
    assert not verdict.blocked and verdict.source == "discriminator"
    assert verdict.quality.value == "good" and v.client.calls == 1


def test_unavailable_degrades_to_rules():
    v = TypedVerifier(DiscriminatorClient(base_url="http://127.0.0.1:9", timeout=0.2))
    verdict = v.verify("布洛芬有什么副作用", "胃肠道不适。", [])
    assert not verdict.blocked and verdict.source == "rules(discriminator_unavailable)" and v.stats["disc_error"] == 1


def test_not_configured_means_rules_only(monkeypatch):
    monkeypatch.delenv("MEDAGENT_VERIFIER_URL", raising=False)
    v = TypedVerifier(DiscriminatorClient())
    assert v.verify("布洛芬有什么副作用", "胃肠道不适。", []).source == "rules"


def test_url_with_v1_suffix_is_accepted():
    assert DiscriminatorClient(base_url="http://127.0.0.1:8002/v1").base_url == "http://127.0.0.1:8002"


def test_attribution_stays_rules():
    v = verifier()
    ambiguous = [{"stage": "tool", "source": "tool:search_drug", "ok": True},
                 {"stage": "llm", "source": "llm:local", "ok": True},
                 {"stage": "verify", "source": "verifier", "ok": True,
                  "findings": [{"code": "ungrounded_dose", "kind": "silent", "severity": "block"}]}]
    out = v.attribute(ambiguous)
    assert out["source"] == "rules" and out["ambiguous"] and v.client.calls == 0   # 调用出错的归因仍用规则


def test_decisions_map_laya_answers():
    c = client(lambda st: (0.62, "poor", "incomplete"))
    st = answer_state("q", "a", None, [])
    d = c.decide(SILENT_FAILURE, st)
    assert d.value is True and abs(d.confidence - 0.62) < 1e-9
    assert c.decide(ANSWER_QUALITY, st).value == "poor"
    assert c.decide(FAILURE_TYPE, st).value == "incomplete" and c.calls == 1


def test_out_of_range_choice_is_an_error():
    c = client(lambda st: (0.9, "terrible", "ok"))
    try:
        c.decide(ANSWER_QUALITY, answer_state("q", "a", None, []))
        raise AssertionError("should have raised")
    except ValueError:
        pass


def test_request_carries_state_questions_and_budget():
    st = answer_state("布洛芬怎么吃", "一次0.2g。", [{"found": True, "name": "布洛芬", "dosage": "成人一次0.2g，一日3次"}], [],
                      assess("布洛芬怎么吃"))
    body = DiscriminatorClient(base_url="http://x").request_body(st)
    assert body["questions"] is QUESTIONS and body["max_len"] >= 1024 and body["model"] == "medical-verifier"
    assert body["state"]["助手回答"] == "一次0.2g。" and "成人一次0.2g" in body["state"]["工具结果"][0]
    assert QUESTIONS["failure_type"]["criteria"] is FAILURE_TYPES and "ok" in FAILURE_TYPES


def test_training_rows_use_runtime_questions_and_state():
    """训练数据的问题定义与状态格式必须和线上请求完全一致，否则模型学到的选项文字线上看不到。"""
    sys.path.insert(0, str(ROOT / "training" / "scripts" / "verifier"))
    from build_verifier_data import row
    st = answer_state("我胸口痛，喘不上气", "多休息，喝点热水。", None, [], assess("我胸口痛，喘不上气"))
    r = row(st, "unsafe", "emergency_missing", {"split": "train"})
    assert r["state"] == laya_state(st)
    assert all(r["questions"][k] is QUESTIONS[k] for k in r["questions"])
    assert r["expected"] == {"silent_failure": True, "answer_quality": "unsafe", "failure_type": "emergency_missing"}
    ok = row(st, "good", "unknown", {"split": "train"})
    assert ok["expected"]["failure_type"] == "ok"
    unk = row(st, "poor", "unknown", {"split": "train"})
    assert "failure_type" not in unk["expected"] and "failure_type" not in unk["questions"]   # 没有有效类别就不训练这一项
