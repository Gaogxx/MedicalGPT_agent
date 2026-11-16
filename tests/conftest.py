"""测试默认不连判别模型和奖励模型：结果不受本机有没有部署 laya-serve / 打分服务影响。要测的用例自己传 base_url 或 client。"""
import pytest


@pytest.fixture(autouse=True)
def _no_local_verifier(monkeypatch):
    monkeypatch.setenv("MEDAGENT_VERIFIER_URL", "")
    monkeypatch.setenv("MEDAGENT_REWARD_URL", "")
