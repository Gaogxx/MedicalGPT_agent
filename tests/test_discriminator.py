"""开源判别模型（替代 Jev）：用返回 logprobs 的假 vLLM 跑通 校验 / 降级 / 归因 / 置信度换算，并检查训练与推理格式一致。"""
import math
import sys
from pathlib import Path

from starlette.testclient import TestClient

from medagent.safety import assess
from medagent.verify.detectors import detect
from medagent.verify.discriminator import (DiscriminatorClient, build_messages, distribution, fake_vllm_app,
                                           letter_logprobs)
from medagent.verify.typed import ANSWER_QUALITY, SILENT_FAILURE, TypedVerifier, answer_state

ROOT = Path(__file__).resolve().parents[1]


def client(judge=None, **kw) -> DiscriminatorClient:
    return DiscriminatorClient(base_url="http://testserver/v1", client=TestClient(fake_vllm_app(judge)), **kw)


def verifier(judge=None, **kw) -> TypedVerifier:
    return TypedVerifier(client(judge), **kw)


def test_rules_block_without_calling_discriminator():
    v = verifier()
    q, a = "我恶心", 'olith\n{"name": "check_symptom", "arguments": {}}'
    verdict = v.verify(q, a, detect(a, query=q, assessment=assess(q)))
    assert verdict.blocked and verdict.source == "rules" and v.client.calls == 0


def test_discriminator_catches_what_rules_miss():
    v = verifier()
    q, a = "对乙酰氨基酚怎么吃", "据我所知，一次吃两片就行。"   # 规则没命中：没有工具结果可比对剂量
    verdict = v.verify(q, a, detect(a, query=q, assessment=assess(q)), observations=None)
    assert verdict.blocked and verdict.source == "discriminator" and verdict.confidence >= 0.8


def test_good_answer_passes_and_quality_shares_one_request():
    v = verifier(score_quality=True)
    verdict = v.verify("布洛芬有什么副作用", "常见胃肠道不适，建议饭后服用。仅供参考。", [])
    assert not verdict.blocked and verdict.source == "discriminator"
    assert verdict.quality.value == "good" and v.client.calls == 1          # 静默失败 + 质量等级 = 一次请求


def test_unavailable_degrades_to_rules():
    v = TypedVerifier(DiscriminatorClient(base_url="http://127.0.0.1:9/v1", timeout=0.2))
    verdict = v.verify("布洛芬有什么副作用", "胃肠道不适。", [])
    assert not verdict.blocked and verdict.source == "rules(discriminator_unavailable)" and v.stats["disc_error"] == 1


def test_not_configured_means_rules_only(monkeypatch):
    monkeypatch.delenv("MEDAGENT_VERIFIER_URL", raising=False)
    v = TypedVerifier(DiscriminatorClient())
    assert v.verify("布洛芬有什么副作用", "胃肠道不适。", []).source == "rules"


def test_attribution_stays_rules():
    v = verifier()
    ambiguous = [{"stage": "tool", "source": "tool:search_drug", "ok": True},
                 {"stage": "llm", "source": "llm:local", "ok": True},
                 {"stage": "verify", "source": "verifier", "ok": True,
                  "findings": [{"code": "ungrounded_dose", "kind": "silent", "severity": "block"}]}]
    out = v.attribute(ambiguous)
    assert out["source"] == "rules" and out["ambiguous"] and v.client.calls == 0   # 判别模型不做归因


def test_letter_logprobs_merge_and_temperature():
    top = [{"token": "A", "logprob": math.log(0.3)}, {"token": " A", "logprob": math.log(0.3)},
           {"token": "C", "logprob": math.log(0.3)}, {"token": "的", "logprob": math.log(0.1)}]
    lps = letter_logprobs(top)
    assert set(lps) == {"A", "C"} and abs(math.exp(lps["A"]) - 0.6) < 1e-9
    d1 = distribution(lps, 1.0)
    assert abs(d1["good"] - 2 / 3) < 1e-9 and d1["unsafe"] == 0.0
    d2 = distribution(lps, 2.0)                                       # 温度越高分布越平
    assert d2["good"] < d1["good"] and abs(sum(d2.values()) - 1) < 1e-9


def test_silent_failure_is_poor_plus_unsafe():
    c = client(lambda text: {"A": 0.2, "B": 0.2, "C": 0.35, "D": 0.25})
    st = answer_state("q", "a", None, [])
    d = c.decide(SILENT_FAILURE, st)
    assert d.value is True and abs(d.confidence - 0.6) < 1e-6
    assert c.decide(ANSWER_QUALITY, st).value == "poor" and c.calls == 1


def test_state_carries_tool_evidence():
    st = answer_state("布洛芬怎么吃", "一次0.2g。", [{"found": True, "name": "布洛芬", "dosage": "成人一次0.2g，一日3次"}], [])
    text = build_messages(st)[1]["content"]
    assert "成人一次0.2g" in text and "found=True" in text and "助手回答：一次0.2g。" in text


def test_training_text_equals_inference_text():
    """判别模型训练样本（OpenAI 格式 → MedicalGPT 输入）与 vLLM 推理时渲染的文本逐字一致。"""
    sys.path.insert(0, str(ROOT / "training" / "scripts" / "data_pipeline"))
    sys.path.insert(0, str(ROOT / "tests"))
    import export_formats as ef
    from test_chat_template import EOS, medicalgpt_text, render

    st = answer_state("我胸口痛，喘不上气", "多休息，喝点热水。", None, detect("多休息", query="我胸口痛", assessment=assess("我胸口痛")),
                      assess("我胸口痛，喘不上气"))
    msgs = build_messages(st)
    mg = ef.openai_row_to_medicalgpt({"messages": msgs + [{"role": "assistant", "content": "D"}]})
    assert render(msgs + [{"role": "assistant", "content": "D"}]) == medicalgpt_text(mg["conversations"])
    prompt = render(msgs, add_generation_prompt=True)
    assert prompt.endswith("<|im_start|>assistant\n") and (prompt + "D" + EOS) == medicalgpt_text(mg["conversations"]) + ""
