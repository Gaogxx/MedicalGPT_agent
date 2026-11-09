"""出错归因：从一次 run 的事件轨迹判断“到底是谁的错”。

输出 kind ∈ {network, timeout, rate_limit, auth, param, model, silent, external, policy, circuit_open, conflict, ok}
以及 responsible ∈ {model, system, upstream, user}：
  - 工具报 PARAM，但参数是模型生成的 → responsible=model（该让模型改参数重试，而不是修代码）
  - 工具报 PARAM，参数是代码拼的     → responsible=system（这是 bug，要告警）
  - NETWORK/TIMEOUT/EXTERNAL/RATE_LIMIT/AUTH → upstream
  - SILENT（调用全成功但结果被校验器判错） → model（或 upstream，若最后一个成功事件来自工具且结果为空）
规则：取轨迹中“第一个根因事件”，后续的连锁失败（如熔断打开）不覆盖根因。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_UPSTREAM = {"network", "timeout", "rate_limit", "auth", "external"}
_DERIVED = {"circuit_open"}   # 熔断是结果不是原因：有更早的根因时以根因为准


@dataclass
class Attribution:
    kind: str
    responsible: str
    component: str
    reason: str
    chain: list[str] = field(default_factory=list)


def attribute(events: list[dict[str, Any]]) -> Attribution:
    chain: list[str] = []
    root: dict[str, Any] | None = None
    for ev in events:
        err = ev.get("error")
        if err:
            chain.append(f"{ev.get('source', '?')}:{err.get('kind')}")
            if root is None or (root["error"]["kind"] in _DERIVED and err.get("kind") not in _DERIVED):
                root = ev
        elif ev.get("findings"):
            blocks = [x for x in ev["findings"] if x.get("severity") == "block"]
            if blocks:
                chain.append(f"{ev.get('source', '?')}:silent[{blocks[0]['code']}]")
                if root is None:
                    root = {**ev, "error": {"kind": blocks[0]["kind"], "message": blocks[0]["code"]}}
    if root is None:
        return Attribution("ok", "none", "", "no failure", chain)
    kind = root["error"]["kind"]
    src = root.get("source", "")
    if kind in _UPSTREAM:
        resp = "upstream"
    elif kind == "param":
        resp = "model" if root.get("args_from_model", src.startswith("tool:")) else "system"
    elif kind in ("model", "silent"):
        resp = "model" if not src.startswith("tool:") else "upstream"
    elif kind == "policy":
        resp = "user"
    else:
        resp = "system"
    return Attribution(kind, resp, src, str(root["error"].get("message", ""))[:200], chain)
