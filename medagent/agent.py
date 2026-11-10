"""Router–Worker 编排（参考 DeepSeek Super-agent：常驻轻量 Router 负责交互与分派，
上下文隔离的 Worker 负责干活并通过 Router 汇报），落到医疗问答上：

  intake → route → plan → work(并行 worker) → aggregate → verify(→repair) → respond → done
    │        │       │        │                                │
  安全预检  RouteLLM  拆子问题  每个 worker 只拿“子问题+患者槽位”，  静默失败校验：泄漏救回 / 升级强模型重生成 /
           +安全规则           自带 MCP 工具子集，LLM↔工具循环       模板兜底；输出关加急救横幅与免责声明

每走完一个节点写一次检查点（乐观并发），节点边界检查暂停请求；崩溃/暂停后 resume 从 state["node"] 继续，
已完成的 worker 不重跑，有副作用的工具靠幂等键保证重放不重复执行。
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from .context import ContextCompressor, PatientProfile
from .rag.rewrite import QueryRewriter
from .resilience.checkpoint import DONE, PAUSE_REQUESTED, PAUSED, CheckpointStore, ConcurrentModification
from .router.policy import SafetyFirstRouter
from .safety import EMERGENCY_BANNER, SafetyAssessment, assess, guard_output
from .templates import render_fallback
from .types import AgentError, ErrorKind
from .verify.attribution import attribute
from .verify.detectors import detect, extract_leaked_tool_call
from .verify.jev import JevClient, JevVerifier
from .verify.judge import CascadeJudge

SYSTEM_PROMPT = """你是一名专业、谨慎的医疗健康助手，不能替代执业医师。
1. 药品、剂量、相互作用必须先调用工具获取，不得凭记忆给出剂量；症状先 check_symptom。
2. 工具查不到（found=false）时如实说明查不到，不得编造。
3. 用户没有提供的信息（持续时间、年龄、既往史等）不要假设；需要时主动追问。
4. 不开处方、不下诊断、不制定治疗方案，涉及时明确建议面诊。
5. 急症（胸痛、呼吸困难、意识障碍、卒中表现、大出血等）立即建议拨打 120。
6. 回答使用简洁中文，结尾提示仅供参考。"""

GUARD_FIXABLE = {"emergency_missing", "prescription_in_high_risk"}   # 输出关可确定性修复，无需重生成


@dataclass
class AgentConfig:
    max_tool_rounds: int = 3
    max_workers: int = 3
    repair_attempts: int = 1
    llm_aggregate: bool = False
    tool_subset: list[str] | None = None   # worker 可见的工具（None = 全部）；作为“复杂问诊”工具运行时只给只读工具
    split_with_history: bool = False       # 有历史时是否仍拆子问题（复杂问诊工具开启：历史只进 w0 与患者槽位）


class MedicalAgent:
    def __init__(self, pool: Any, toolbox: Any, router: SafetyFirstRouter, store: CheckpointStore,
                 judge: CascadeJudge | None = None, compressor: ContextCompressor | None = None,
                 audit: Any = None, cfg: AgentConfig | None = None, verifier: JevVerifier | None = None) -> None:
        self.pool, self.toolbox, self.router, self.store = pool, toolbox, router, store
        self.judge = judge or CascadeJudge()
        self.compressor = compressor or ContextCompressor()
        self.audit = audit
        self.cfg = cfg or AgentConfig()
        self.rewriter = QueryRewriter()
        self.verifier = verifier or JevVerifier(JevClient())
        self._versions: dict[str, int] = {}

    # ------------------------------------------------------------------ 生命周期
    async def run(self, query: str, *, run_id: str | None = None, history: list[dict[str, Any]] | None = None,
                  session_id: str = "default") -> dict[str, Any]:
        run_id = run_id or uuid.uuid4().hex[:12]
        state = {"run_id": run_id, "session_id": session_id, "query": query, "history": history or [],
                 "node": "intake", "events": [], "workers": {}, "answer": "", "tier": None, "route": None,
                 "fastpath": False, "findings": [], "repairs": [], "attribution": None, "started": time.time()}
        self._versions[run_id] = self.store.create(run_id, state)
        return await self._drive(state)

    async def resume(self, run_id: str) -> dict[str, Any]:
        self._versions[run_id] = self.store.mark_resumable(run_id)
        state, version, _ = self.store.load(run_id)
        self._versions[run_id] = version
        state["events"].append({"stage": "lifecycle", "source": "agent", "ok": True, "note": f"resumed at {state['node']}"})
        return await self._drive(state)

    def pause(self, run_id: str) -> None:
        self.store.request_pause(run_id)

    def _checkpoint(self, state: dict[str, Any]) -> bool:
        """保存检查点；若期间有人提了暂停请求（版本号被改），转为 PAUSED 并返回 False。"""
        rid = state["run_id"]
        try:
            self._versions[rid] = self.store.save(rid, state, self._versions[rid])
            return True
        except ConcurrentModification:
            _, v, status = self.store.load(rid)
            if status == PAUSE_REQUESTED:
                self._versions[rid] = self.store.save(rid, state, v, status=PAUSED)
                return False
            raise  # 另一个执行者接管了这个 run：本执行者必须停止，不能覆盖别人的进度

    async def _drive(self, state: dict[str, Any]) -> dict[str, Any]:
        rid = state["run_id"]
        while state["node"] != "done":
            _, v, status = self.store.load(rid)
            if status in (PAUSE_REQUESTED, PAUSED):   # 协作式暂停：只在节点边界停
                self._versions[rid] = self.store.save(rid, state, v, status=PAUSED)
                return {"status": "paused", "run_id": rid, "node": state["node"]}
            if v != self._versions[rid]:
                raise ConcurrentModification(rid, self._versions[rid])
            state = await getattr(self, f"_node_{state['node']}")(state)
            if not self._checkpoint(state):
                return {"status": "paused", "run_id": rid, "node": state["node"]}
        self._versions[rid] = self.store.save(rid, state, self._versions[rid], status=DONE)
        if self.audit:
            self.audit.write(state)
        return {"status": "done", "run_id": rid, "answer": state["answer"], "tier": state["tier"],
                "route": state["route"], "findings": state["findings"], "repairs": state["repairs"],
                "attribution": state["attribution"], "events": state["events"]}

    # ------------------------------------------------------------------ 节点
    async def _system_prompt(self) -> str:
        if not hasattr(self, "_sp"):
            try:
                self._sp = await self.toolbox.get_prompt("medical_system_prompt") or SYSTEM_PROMPT
            except Exception:  # noqa: BLE001
                self._sp = SYSTEM_PROMPT
        return self._sp

    async def _node_intake(self, s: dict[str, Any]) -> dict[str, Any]:
        a = assess(s["query"])
        prof = PatientProfile()
        for m in s["history"]:
            if m.get("role") == "user":
                prof.update(str(m.get("content", "")))
        prof.update(s["query"])
        s["assessment"] = asdict(a)
        s["profile"] = asdict(prof)
        s["node"] = "route"
        return s

    async def _node_route(self, s: dict[str, Any]) -> dict[str, Any]:
        hist = " ".join(str(m.get("content", "")) for m in s["history"][-6:])
        d = self.router.route(s["query"], hist, SafetyAssessment(**s["assessment"]))
        s["tier"], s["fastpath"] = d.tier, d.fastpath
        s["route"] = {"tier": d.tier, "reason": d.reason, "score": d.score, "threshold": d.threshold}
        s["node"] = "plan"
        return s

    async def _node_plan(self, s: dict[str, Any]) -> dict[str, Any]:
        rq = self.rewriter.rewrite(s["query"])
        allow = not s["history"] or self.cfg.split_with_history
        subs = rq.sub_queries if len(rq.sub_queries) >= 2 and allow and self.cfg.max_workers >= 2 else []
        if len(subs) > self.cfg.max_workers:
            # 子问题多于 worker 上限：多出来的并进最后一个 worker，不能悄悄丢掉（2026-11-09 之前会截断丢题）
            k = self.cfg.max_workers - 1
            subs = subs[:k] + ["；".join(subs[k:])]
        questions = subs or [s["query"]]
        s["workers"] = {f"w{i}": {"question": q, "status": "pending"} for i, q in enumerate(questions)}
        s["node"] = "work"
        return s

    async def _node_work(self, s: dict[str, Any]) -> dict[str, Any]:
        pending = [(wid, w) for wid, w in s["workers"].items() if w["status"] != "done"]
        lock = asyncio.Lock()

        async def one(wid: str, w: dict[str, Any]) -> None:
            res = await self._run_worker(s, wid, w["question"])
            async with lock:
                s["workers"][wid] = {**w, **res, "status": "done"}
                self._checkpoint(s)   # 每个 worker 完成即落盘：中途暂停/崩溃只重跑没完成的 worker

        await asyncio.gather(*(one(wid, w) for wid, w in pending))
        s["node"] = "aggregate"
        return s

    async def _node_aggregate(self, s: dict[str, Any]) -> dict[str, Any]:
        parts = [w for w in s["workers"].values() if w.get("answer")]
        if len(parts) <= 1:
            s["answer"] = parts[0]["answer"] if parts else ""
        elif not self.cfg.llm_aggregate:
            # 默认确定性拼接：LLM 合并可能悄悄丢掉某个子问题的内容（本身就是一种静默失败）
            s["answer"] = "\n\n".join(f"【{w['question']}】\n{w['answer']}" for w in parts)
        else:
            merged = "\n\n".join(f"【{w['question']}】\n{w['answer']}" for w in parts)
            try:
                resp = await self.pool.generate(s["tier"], [
                    {"role": "system", "content": "把以下分问题的回答合并成一段连贯回复，不得新增任何事实或剂量。"},
                    {"role": "user", "content": merged}], events=s["events"])
                s["answer"] = resp.content or merged
            except AgentError:
                s["answer"] = merged
        s["node"] = "verify"
        return s

    def _observations(self, s: dict[str, Any]) -> list[tuple[str, Any]]:
        return [tuple(o) for w in s["workers"].values() for o in w.get("observations", [])]

    def _detect(self, s: dict[str, Any], answer: str, finish: str | None = None) -> list[dict[str, Any]]:
        hist = " ".join(str(m.get("content", "")) for m in s["history"])
        obs = [o for _, o in self._observations(s)]
        fs = detect(answer, query=s["query"], assessment=SafetyAssessment(**s["assessment"]),
                    observations=obs, history=hist, finish_reason=finish)
        return [{"code": f.code, "kind": f.kind.value, "severity": f.severity, "evidence": f.evidence} for f in fs]

    async def _node_verify(self, s: dict[str, Any]) -> dict[str, Any]:
        findings = self._detect(s, s["answer"])
        if not any(f["severity"] == "block" and f["code"] not in GUARD_FIXABLE for f in findings):
            from .verify.detectors import Finding as _F
            from .types import ErrorKind as _K
            verdict = await asyncio.to_thread(
                self.verifier.verify, s["query"], s["answer"],
                [_F(f["code"], _K(f["kind"]), f["severity"], f["evidence"]) for f in findings if f["code"] not in GUARD_FIXABLE],
                [o for _, o in self._observations(s)], SafetyAssessment(**s["assessment"]))
            s["verdict"] = {"blocked": verdict.blocked, "reason": verdict.reason, "confidence": verdict.confidence,
                            "source": verdict.source, "needs_review": verdict.needs_review}
            if verdict.blocked:   # Jev 判为静默失败：按 SILENT 处理，走同一条修复路径
                findings.append({"code": "jev_silent_failure", "kind": "silent", "severity": "block",
                                 "evidence": f"p={verdict.confidence:.2f}"})
        s["events"].append({"stage": "verify", "source": "verifier", "ok": True, "findings": findings})
        for _ in range(self.cfg.repair_attempts):
            blocking = [f for f in findings if f["severity"] == "block" and f["code"] not in GUARD_FIXABLE]
            if not blocking:
                break
            s["answer"] = await self._repair(s, blocking)
            findings = self._detect(s, s["answer"])
            s["events"].append({"stage": "verify", "source": "verifier:after_repair", "ok": True, "findings": findings})
        if any(f["severity"] == "block" and f["code"] not in GUARD_FIXABLE for f in findings):
            s["repairs"].append("template_fallback")
            s["answer"] = render_fallback(self._observations(s))
        s["findings"] = findings
        s["node"] = "respond"
        return s

    async def _repair(self, s: dict[str, Any], blocking: list[dict[str, Any]]) -> str:
        codes = {f["code"] for f in blocking}
        if "tool_leak" in codes:
            call = extract_leaked_tool_call(s["answer"])
            if call is not None and call.name in self.toolbox.tools:
                s["repairs"].append(f"recover_leaked_call:{call.name}")
                res = await self.toolbox.call(call.name, call.arguments, scope=f"{s['run_id']}:repair",
                                              events=s["events"])
                if res.ok:
                    w = s["workers"].setdefault("repair", {"question": s["query"], "status": "done", "observations": []})
                    w.setdefault("observations", []).append([call.name, res.data])
                    return await self._answer_from_obs(s, s["tier"], [(call.name, res.data)])
        obs = self._observations(s)
        if not obs:
            # 空回复 / 截断且还没拿到任何工具结果：在强模型上把 worker 完整重跑一遍（含工具调用）
            s["repairs"].append(f"rerun_worker_strong:{','.join(sorted(codes))}")
            res = await self._run_worker(s, "w_repair", s["query"], tier="strong")
            s["workers"]["w_repair"] = {"question": s["query"], "status": "done", **res}
            return res.get("answer") or render_fallback([tuple(o) for o in res.get("observations", [])])
        # 其它静默失败：升级到强模型、带着工具结果与校验意见重生成一次
        s["repairs"].append(f"regenerate_strong:{','.join(sorted(codes))}")
        return await self._answer_from_obs(s, "strong", obs, feedback=sorted(codes))

    async def _answer_from_obs(self, s: dict[str, Any], tier: str, obs: list[tuple[str, Any]],
                               feedback: list[str] | None = None) -> str:
        note = f"上一版回答被校验器拦截（{','.join(feedback)}），请避免同类问题。" if feedback else ""
        msgs = [{"role": "system", "content": await self._system_prompt() + "\n" + note},
                {"role": "user", "content": s["query"]},
                {"role": "user", "content": "工具结果：" + json.dumps([{"tool": t, "result": o} for t, o in obs],
                                                                   ensure_ascii=False)[:3000]}]
        try:
            resp = await self.pool.generate(tier, msgs, events=s["events"])
            return resp.content
        except AgentError:
            return render_fallback(obs)

    async def _node_respond(self, s: dict[str, Any]) -> dict[str, Any]:
        a = SafetyAssessment(**s["assessment"])
        ans = guard_output(s["answer"] or render_fallback(self._observations(s)), a)
        if s["fastpath"] and EMERGENCY_BANNER not in ans and not ans.startswith("⚠️"):
            ans = f"{EMERGENCY_BANNER}\n\n{ans}"
        s["answer"] = ans
        s["attribution"] = self.verifier.attribute(s["events"])
        s["node"] = "done"
        return s

    # ------------------------------------------------------------------ Worker
    async def _run_worker(self, s: dict[str, Any], wid: str, question: str, tier: str | None = None) -> dict[str, Any]:
        tier = tier or s["tier"]
        prof = PatientProfile(**s["profile"])
        msgs: list[dict[str, Any]] = [{"role": "system", "content": await self._system_prompt()}]
        if s["history"] and wid == "w0":
            msgs += s["history"]
        msgs.append({"role": "user", "content": question})
        msgs, _ = self.compressor.compress(msgs, prof)
        from .consult import TOOL_NAME
        # worker 不能调用复杂问诊（它自己就在编排器里，调用会递归）
        allowed = self.cfg.tool_subset or [t for t in self.toolbox.tools if t != TOOL_NAME]
        tools = None if tier == "small" else self.toolbox.openai_tools(allowed)   # 小模型档不给工具
        observations: list[list[Any]] = []
        answer, finish = "", None
        for rnd in range(self.cfg.max_tool_rounds):
            try:
                resp = await self.pool.generate(tier, msgs, tools, events=s["events"])
            except AgentError as err:
                # 两档模型都不可用：确定性兜底——按实体链接直接查工具并模板化
                s["repairs"].append(f"model_unavailable:{err.kind.value}")
                observations += await self._deterministic_tools(s, wid, question)
                from .templates import kb_fallback_answer
                hist = " ".join(str(m.get("content", "")) for m in s["history"])
                return {"answer": kb_fallback_answer(question, SafetyAssessment(**s["assessment"]), hist),
                        "observations": observations, "degraded": True}
            if not resp.tool_calls:
                answer, finish = resp.content, resp.finish_reason
                break
            msgs.append({"role": "assistant", "content": resp.content or "", "tool_calls": [
                {"id": c.id or f"{rnd}{i}", "type": "function",
                 "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)}}
                for i, c in enumerate(resp.tool_calls)]})
            for i, c in enumerate(resp.tool_calls):
                if c.name not in allowed:
                    # 模型调了不在白名单里的工具（例如作为复杂问诊运行时去调预约）：不执行，按 MODEL 错误回给模型
                    s["events"].append({"stage": "tool", "source": f"tool:{c.name}", "ok": False,
                                        "error": AgentError(ErrorKind.MODEL, f"tool {c.name} not allowed here",
                                                            source=f"tool:{c.name}").to_dict()})
                    msgs.append({"role": "tool", "tool_call_id": c.id or f"{rnd}{i}",
                                 "content": json.dumps({"error": f"此场景不可用工具 {c.name}"}, ensure_ascii=False)})
                    continue
                res = await self.toolbox.call(c.name, c.arguments, scope=f"{s['run_id']}:{wid}:{rnd}:{i}",
                                              events=s["events"])
                if not res.ok and res.error and res.error.kind == ErrorKind.SILENT:
                    res = await self.toolbox.call(c.name, c.arguments, scope=f"{s['run_id']}:{wid}:{rnd}:{i}:r",
                                                  events=s["events"])
                if res.ok:
                    observations.append([c.name, res.data])
                content = res.as_observation()
                if not res.ok and res.error and res.error.kind == ErrorKind.PARAM:
                    content = json.dumps({"error": str(res.error), "hint": "参数有误，请按工具 schema 修正后重试"},
                                         ensure_ascii=False)
                msgs.append({"role": "tool", "tool_call_id": c.id or f"{rnd}{i}", "content": content})
        return {"answer": answer, "observations": observations, "finish_reason": finish}

    async def _deterministic_tools(self, s: dict[str, Any], wid: str, question: str) -> list[list[Any]]:
        """两档模型都不可用时的确定性兜底：实体链接 → 通过 MCP resources 直接读知识条目（只读、不经模型）。
        症状分析需要匹配逻辑，仍走 check_symptom 工具；资源读取失败时退回对应工具。"""
        from .knowledge import load_kb
        ents = load_kb().link_entities(question)
        out: list[list[Any]] = []
        if ents["symptom"]:
            res = await self.toolbox.call("check_symptom", {"symptom": question}, scope=f"{s['run_id']}:{wid}:det",
                                          events=s["events"])
            if res.ok:
                out.append(["check_symptom", res.data])
        for kind, uri, tool, arg in (("drug", "kb://drug/{}", "search_drug", "name"),
                                     ("disease", "kb://guideline/{}", "search_guideline", "topic")):
            for name in ents[kind][:2]:
                data = await self.toolbox.read_resource(uri.format(name))
                s["events"].append({"stage": "resource", "source": f"resource:{uri.format(name)}", "ok": data is not None})
                if isinstance(data, dict):
                    out.append([tool, {"found": True, **data}])
                else:
                    res = await self.toolbox.call(tool, {arg: name}, scope=f"{s['run_id']}:{wid}:det:{name}",
                                                  events=s["events"])
                    if res.ok:
                        out.append([tool, res.data])
        return out


class JsonlAudit:
    def __init__(self, path: str) -> None:
        self.path = path

    def write(self, state: dict[str, Any]) -> None:
        rec = {k: state.get(k) for k in ("run_id", "session_id", "query", "tier", "route", "findings", "repairs",
                                          "attribution", "started")}
        rec["events"] = state.get("events", [])
        rec["finished"] = time.time()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
