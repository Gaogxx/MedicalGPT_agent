"""RouteLLM 矩阵分解路由（Ong et al., 2024, "RouteLLM: Learning to Route LLMs with Preference Data"）。

与官方实现（routellm/routers/matrix_factorization/model.py）同构：
    s(m, q) = wᵀ ( normalize(P_m) ⊙ (W e_q) )
    P(strong 胜 | q) = σ( s(strong, q) − s(weak, q) )
差别只在 e_q：官方用 OpenAI text-embedding-3-small（推理时必须调 OpenAI），这里换成本地 bge-small-zh 或哈希 n-gram，
并用本项目自己的偏好标签训练（官方权重是在 Chatbot Arena 的 GPT-4 vs Mixtral 对上训的，对中文医疗不适用）。

标签：一条查询上 “本地 MedicalGPT 的回答是否达标”。不达标 → strong 胜（y=1）。
阈值：RouteLLM 的做法——在校准集上取 P(strong 胜) 的 (1 − 目标强模型调用比例) 分位数。
纯 numpy 实现（Adam + L2 + 早停），无 torch 依赖，可在 CPU 毫秒级推理。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class MFRouter:
    dim: int = 32
    num_models: int = 2           # 0 = weak(local MedicalGPT), 1 = strong(cloud)
    l2: float = 1e-3
    lr: float = 0.03
    epochs: int = 400
    seed: int = 0
    W: np.ndarray | None = None   # (text_dim, dim)
    P: np.ndarray | None = None   # (num_models, dim)
    w: np.ndarray | None = None   # (dim,)

    STRONG, WEAK = 1, 0

    def _init(self, text_dim: int) -> None:
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(0, 1 / np.sqrt(text_dim), (text_dim, self.dim))
        self.P = rng.normal(0, 1, (self.num_models, self.dim))
        self.w = rng.normal(0, 1 / np.sqrt(self.dim), self.dim)

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        Pn = self.P / np.linalg.norm(self.P, axis=1, keepdims=True)
        H = X @ self.W                                          # (n, dim)
        diff = Pn[self.STRONG] - Pn[self.WEAK]                  # s(strong)-s(weak) = wᵀ((Ps-Pw) ⊙ H)
        logit = (H * diff) @ self.w
        return logit, {"Pn": Pn, "H": H, "diff": diff}

    def predict_strong_win(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(self._forward(X)[0])

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
            sample_weight: np.ndarray | None = None) -> dict:
        self._init(X.shape[1])
        params = {"W": self.W, "P": self.P, "w": self.w}
        m = {k: np.zeros_like(v) for k, v in params.items()}
        v = {k: np.zeros_like(p) for k, p in params.items()}
        sw = sample_weight if sample_weight is not None else np.ones(len(y))
        pos = max(y.mean(), 1e-6)
        cw = np.where(y == 1, 0.5 / pos, 0.5 / max(1 - pos, 1e-6)) * sw   # 类别平衡
        best = (np.inf, None, 0)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for ep in range(1, self.epochs + 1):
            logit, c = self._forward(X)
            p = _sigmoid(logit)
            g = (p - y) * cw / len(y)                           # dL/dlogit
            gw = (c["H"] * c["diff"]).T @ g + self.l2 * self.w
            gH = np.outer(g, self.w * c["diff"])
            gW = X.T @ gH + self.l2 * self.W
            # 经 normalize(P) 的梯度
            gdiff = (c["H"] * self.w).T @ g                      # (dim,)
            grads_P = np.zeros_like(self.P)
            for idx, sign in ((self.STRONG, 1.0), (self.WEAK, -1.0)):
                pn = c["Pn"][idx]
                nrm = np.linalg.norm(self.P[idx])
                gp = sign * gdiff
                grads_P[idx] = (gp - pn * (pn @ gp)) / nrm
            grads = {"W": gW, "P": grads_P + self.l2 * self.P, "w": gw}
            for k in params:
                m[k] = b1 * m[k] + (1 - b1) * grads[k]
                v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
                params[k] -= self.lr * (m[k] / (1 - b1 ** ep)) / (np.sqrt(v[k] / (1 - b2 ** ep)) + eps)
            if X_val is not None and ep % 10 == 0:
                pv = np.clip(self.predict_strong_win(X_val), 1e-6, 1 - 1e-6)
                loss = -np.mean(y_val * np.log(pv) + (1 - y_val) * np.log(1 - pv))
                if loss < best[0] - 1e-4:
                    best = (loss, {k: p.copy() for k, p in params.items()}, ep)
                elif ep - best[2] > 60:
                    break
        if best[1] is not None:
            self.W, self.P, self.w = best[1]["W"], best[1]["P"], best[1]["w"]
        return {"best_val_loss": float(best[0]) if best[1] is not None else None, "best_epoch": best[2]}

    def save(self, path: str | Path) -> None:
        np.savez(path, W=self.W, P=self.P, w=self.w, dim=self.dim)

    @classmethod
    def load(cls, path: str | Path) -> "MFRouter":
        d = np.load(path)
        r = cls(dim=int(d["dim"]))
        r.W, r.P, r.w = d["W"], d["P"], d["w"]
        return r


def calibrate_threshold(scores: np.ndarray, strong_pct: float) -> float:
    """RouteLLM calibrate_threshold：让校准集上约 strong_pct 的查询被送去强模型。"""
    return float(np.quantile(scores, 1 - strong_pct))


def calibrate_with_forced(scores: np.ndarray, forced: np.ndarray, budget: float) -> float:
    """整体预算版：安全规则已经强制交给强模型的（急症 / 危机 / 开药诊断）先占预算，
    MF 只在剩下的问题里挑，使总体约 budget 的问题去强模型。强制部分已超预算时，MF 不再送任何问题。"""
    forced = np.asarray(forced, bool)
    rest = scores[~forced]
    left = budget - forced.mean()
    if left <= 0 or not len(rest):
        return float("inf")
    share = min(left / (1 - forced.mean()), 1.0)
    return float(np.quantile(rest, 1 - share))


def apgr(weak_quality: np.ndarray, strong_quality: np.ndarray, routed_strong: np.ndarray) -> float:
    """Average Performance Gap Recovered（RouteLLM 论文指标）：(路由质量 − 弱模型) / (强模型 − 弱模型)。"""
    routed = np.where(routed_strong, strong_quality, weak_quality).mean()
    gap = strong_quality.mean() - weak_quality.mean()
    return float((routed - weak_quality.mean()) / gap) if gap > 0 else 1.0


def save_meta(path: str | Path, meta: dict) -> None:
    Path(path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
