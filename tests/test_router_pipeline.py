"""路由器训练流程：独立数据集 train / val / test，dev 80 不参与训练；运行时优先加载独立训练的路由器。"""
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "evaluation" / "router"))


def synthetic_labels(n=900, seed=0):
    """“本地模型在长问题、含用药决定的问题上更容易答砸”——让标签和问题内容相关，检验路由器能学到。"""
    rng = random.Random(seed)
    easy = ["{}是什么", "{}有什么作用", "{}的原因是什么", "怎么预防{}"]
    hard = ["我{}三天了，吃了两种药还没好，能不能加量，孩子也有点咳嗽", "{}合并高血压和糖尿病，降压药和二甲双胍怎么调整剂量"]
    words = ["头痛", "胃炎", "湿疹", "失眠", "咽炎", "贫血", "痛风", "便秘", "鼻炎", "腰痛"]
    rows = []
    for i in range(n):
        hard_q = rng.random() < 0.35
        q = rng.choice(hard if hard_q else easy).format(rng.choice(words)) + f"（{i}）"
        label = int(hard_q if rng.random() > 0.1 else not hard_q)
        split = "train" if i % 10 < 7 else ("val" if i % 10 < 9 else "test")
        rows.append({"id": f"s{i}", "split": split, "query": q, "label": label, "judge": "synthetic"})
    return rows


def test_train_router_uses_independent_splits(tmp_path):
    from train_router import train
    res = train(synthetic_labels(), budget=0.4, out_dir=tmp_path)
    assert res["test"]["auroc"] > 0.8                                  # 在没见过的 test 上评
    assert 0.25 <= res["val"]["at_threshold"]["strong_rate"] <= 0.55   # 阈值在 val 上按 40% 预算校准
    assert "dev80_heldout" in res and res["dev80_heldout"]["n"] == 80   # dev 80 只用来评
    meta = json.loads((tmp_path / "mf_router.json").read_text())
    assert meta["trained_on"]["train"] > 0 and meta["threshold"] == res["threshold"]


def test_train_router_refuses_too_few_failures(tmp_path):
    from train_router import train
    rows = [{**r, "label": 0} for r in synthetic_labels(200)]
    try:
        train(rows, out_dir=tmp_path)
        raise AssertionError("should refuse")
    except SystemExit as e:
        assert "太少" in str(e)


def test_runtime_prefers_independently_trained_router(tmp_path, monkeypatch):
    from train_router import train

    from medagent.cli import _router
    train(synthetic_labels(), out_dir=tmp_path)
    monkeypatch.setenv("MEDAGENT_ROUTER_DIR", str(tmp_path))
    r = _router()
    assert r.source.startswith("mf_router") and r.threshold == json.loads((tmp_path / "mf_router.json").read_text())["threshold"]
    monkeypatch.setenv("MEDAGENT_ROUTER_DIR", str(tmp_path / "empty"))
    assert _router().source == "none"


def test_router_set_is_disjoint_from_training_and_eval_data():
    sys.path.insert(0, str(ROOT / "training" / "scripts" / "data_pipeline"))
    rs = [json.loads(line) for line in open(ROOT / "evaluation" / "datasets" / "router_set.jsonl", encoding="utf-8")]
    dev = {json.loads(line)["query"] for line in open(ROOT / "evaluation" / "datasets" / "agent_eval_dev.jsonl", encoding="utf-8")}
    train_q = set()
    for p in (ROOT / "training" / "data" / "openai_v3").glob("*/*.jsonl"):
        for line in open(p, encoding="utf-8"):
            train_q |= {m["content"] for m in json.loads(line)["messages"] if m["role"] == "user"}
    qs = [r["query"] for r in rs]
    assert len(qs) == len(set(qs)) == 2000
    assert not set(qs) & dev and not set(qs) & train_q
    assert {r["split"] for r in rs} == {"train", "val", "test"}


async def test_label_script_judge_parsing():
    from label_router_set import judge_strong

    from medagent.llm.pool import ModelPool
    from medagent.llm.providers import MockProvider
    from medagent.types import LLMResponse
    judge = MockProvider("judge", policy=lambda m, t: LLMResponse('判定：{"level": "poor", "reason": "漏了禁忌"}', [], "j"))
    level, reason = await judge_strong(ModelPool({"strong": [judge]}), "问", "参考", "答")
    assert (level, reason) == ("poor", "漏了禁忌")


def test_calibration_counts_safety_forced_share():
    from medagent.router.mf_router import calibrate_with_forced
    rng = np.random.default_rng(0)
    scores, forced = rng.random(1000), rng.random(1000) < 0.3
    thr = calibrate_with_forced(scores, forced, 0.4)
    total = (forced | (scores >= thr)).mean()
    assert abs(total - 0.4) < 0.02                                   # 总体约 40%，不是 MF 单独 40%
    assert calibrate_with_forced(scores, rng.random(1000) < 0.5, 0.4) == float("inf")   # 强制部分已超预算
