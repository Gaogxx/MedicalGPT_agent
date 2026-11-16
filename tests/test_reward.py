"""校验器 v4：规则 → Laya → 奖励模型（Skywork-Reward-V2）。用假打分服务测管线，不代表奖励模型的能力。"""
from starlette.testclient import TestClient

from medagent.verify.reward import RewardClient, fake_reward_app, load_reward_calibration
from medagent.verify.typed import TypedVerifier, default_verifier


def client(fn):
    return RewardClient(base_url="http://testserver", client=TestClient(fake_reward_app(fn)))


def test_low_reward_blocks_and_mid_reward_goes_to_review():
    v = TypedVerifier(None, reward=client(lambda q, a: -3.0 if "乱说" in a else 0.0 if "一般" in a else 5.0),
                      reward_block=-2.0, reward_review=0.65)
    bad = v.verify("布洛芬有什么副作用", "乱说的回答", [])
    assert bad.blocked and bad.source == "reward_model"
    mid = v.verify("布洛芬有什么副作用", "一般的回答", [])
    assert not mid.blocked and mid.needs_review
    good = v.verify("布洛芬有什么副作用", "常见胃肠道不适，少数过敏。", [])
    assert not good.blocked and not good.needs_review
    assert v.stats["reward_block"] == 1 and v.stats["reward_review"] == 1


def test_reward_unavailable_keeps_previous_verdict():
    v = TypedVerifier(None, reward=RewardClient(base_url="http://127.0.0.1:9"), reward_block=-2.0, reward_review=0.65)
    verdict = v.verify("布洛芬有什么副作用", "胃肠道不适。", [])
    assert not verdict.blocked and verdict.source == "rules" and v.stats["reward_error"] == 1


def test_empty_url_disables_reward(monkeypatch):
    monkeypatch.setenv("MEDAGENT_REWARD_URL", "")
    assert not RewardClient().enabled
    monkeypatch.delenv("MEDAGENT_REWARD_URL")
    assert RewardClient().base_url == "http://127.0.0.1:8003"


def test_calibration_file_loaded_by_default_verifier():
    cal = load_reward_calibration()
    assert cal["block_reward"] < cal["review_reward"]
    v = default_verifier()
    assert v.reward_block == cal["block_reward"] and v.reward_review == cal["review_reward"]
