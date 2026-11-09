"""医疗工具 MCP Server —— 替代原先 DSH 插件里的 ctx.tools.register（tool registry）。

同一个 Server 同时服务两个客户端：
  1. DeepSeek Harness：cordis.yml 里配 @deepseek-ai/dsh-mcp-client（stdio），工具名 mcp__medical__<tool>
  2. 本仓库 Python 编排器：medagent.mcp_client.MCPToolbox（stdio / streamable-http / 进程内）

约定：
  - 预期内的失败抛 ToolError("<KIND>: ...")，KIND ∈ PARAM/EXTERNAL/CONFLICT/POLICY，客户端据此归因
  - 只读工具标 read_only_hint / idempotent_hint；有副作用的工具（预约）必须带 idempotency_key，服务端幂等
  - 资源 kb://version 暴露知识库版本，RAG 缓存键带上它，知识库更新后缓存自动失效

运行：
  python -m medagent.mcp_servers.medical_server                         # stdio
  python -m medagent.mcp_servers.medical_server --http --port 8765      # streamable-http，/mcp
"""
from __future__ import annotations

import argparse
import os
import re
import time
import uuid
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

from ..knowledge import load_kb
from ..resilience.idempotency import InMemoryIdempotencyStore, SQLiteIdempotencyStore

SYSTEM_PROMPT = """你是一名专业、谨慎的医疗健康助手，不能替代执业医师。
1. 药品、剂量、相互作用必须先调用工具获取，不得凭记忆给出剂量；症状先 check_symptom。
2. 工具查不到（found=false）时如实说明查不到，不得编造。
3. 用户没有提供的信息（持续时间、年龄、既往史等）不要假设；需要时主动追问。
4. 不开处方、不下诊断、不制定治疗方案，涉及时明确建议面诊。
5. 急症（胸痛、呼吸困难、意识障碍、卒中表现、大出血等）立即建议拨打 120。
6. 回答使用简洁中文，结尾提示仅供参考。"""

INSTRUCTIONS = """医疗工具服务器（tools + resources + prompts）。规则：
1. 药品信息必须通过 search_drug 获取，不要凭记忆给出剂量；多药同服先调 check_interaction。
2. 症状先 check_symptom；疾病管理建议用 search_guideline；以上都查不到时用 search_knowledge 检索知识库。
3. 工具返回 found=false 时如实告知查不到，不得编造。
4. book_appointment 有副作用，必须传 idempotency_key；失败重试时复用同一个 key。
5. 知识库条目也可按 URI 直接读取：kb://drug/{name}、kb://symptom/{name}、kb://guideline/{topic}、kb://catalog；
   search_knowledge 命中的本地文档可用 kb://doc/{id} 读全文；kb://sources 是各知识源的规模与可用状态。"""

READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, destructive_hint=False, open_world_hint=False)
SIDE_EFFECT = ToolAnnotations(read_only_hint=False, idempotent_hint=True, destructive_hint=False, open_world_hint=True)
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DEPARTMENTS = {"内科", "外科", "儿科", "妇产科", "皮肤科", "神经内科", "心内科", "消化内科", "呼吸内科", "内分泌科",
               "风湿免疫科", "耳鼻喉科", "急诊科", "全科"}


def _state_store():
    d = os.environ.get("MEDAGENT_STATE_DIR")
    if d:
        os.makedirs(d, exist_ok=True)
        return SQLiteIdempotencyStore(os.path.join(d, "server_idem.sqlite"))
    return InMemoryIdempotencyStore()


def build_server() -> MCPServer:
    kb = load_kb()
    server = MCPServer("medical", instructions=INSTRUCTIONS, version="0.2.0")
    idem = _state_store()
    appointments: dict[str, dict[str, Any]] = {}
    rag_holder: dict[str, Any] = {}

    def _require(value: str, field: str) -> str:
        if not value or not value.strip():
            raise ToolError(f"PARAM: {field} 不能为空")
        return value.strip()

    @server.tool(annotations=READ_ONLY, description="根据患者描述的症状分析可能原因与紧急程度，提供初步建议")
    def check_symptom(symptom: str, duration: str | None = None) -> dict[str, Any]:
        return kb.check_symptom(_require(symptom, "symptom"), duration)

    @server.tool(annotations=READ_ONLY,
                 description="查询具体药品的适应症、禁忌、用法用量、不良反应、相互作用。name 必须是药品名（支持常见商品名），"
                             "不要传症状；传药物类别（如“止痛药”）会返回候选药品列表")
    def search_drug(name: str) -> dict[str, Any]:
        return kb.search_drug(_require(name, "name"))

    @server.tool(annotations=READ_ONLY, description="检查两种及以上药物同时使用的相互作用")
    def check_interaction(drugs: list[str]) -> dict[str, Any]:
        cleaned = [d.strip() for d in drugs if d and d.strip()]
        if len(cleaned) < 2:
            raise ToolError("PARAM: drugs 至少需要两个药品名")
        return kb.check_interaction(cleaned)

    @server.tool(annotations=READ_ONLY, description="检索临床指南中关于某疾病的推荐意见（饮食、运动、用药原则等）")
    def search_guideline(topic: str, aspect: str | None = None) -> dict[str, Any]:
        return kb.search_guideline(_require(topic, "topic"), aspect)

    @server.tool(annotations=READ_ONLY,
                 description="在医疗知识库中做混合检索（查询改写 + BM25/向量/结构化多路召回 + 重排 + 压缩），返回带出处的证据片段")
    def search_knowledge(query: str, top_k: int = 5) -> dict[str, Any]:
        q = _require(query, "query")
        if not 1 <= top_k <= 20:
            raise ToolError("PARAM: top_k 取值 1~20")
        if "pipeline" not in rag_holder:
            from ..rag.pipeline import build_default_pipeline
            rag_holder["pipeline"] = build_default_pipeline(dense=os.environ.get("MEDAGENT_RAG_DENSE") == "1")
        try:
            res = rag_holder["pipeline"].search(q, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"EXTERNAL: 知识库检索失败: {exc}") from exc
        return {"query": q, "rewritten": res.rewritten, "evidence": res.evidence, "kb_version": kb.version,
                "from_cache": res.from_cache}

    @server.tool(annotations=SIDE_EFFECT,
                 description="为用户预约挂号（有副作用）。必须提供 idempotency_key，重试时复用同一 key 保证不会重复挂号")
    def book_appointment(department: str, date: str, patient_ref: str, idempotency_key: str) -> dict[str, Any]:
        dep = _require(department, "department")
        if dep not in DEPARTMENTS:
            raise ToolError(f"PARAM: 未知科室「{dep}」，可选：{'、'.join(sorted(DEPARTMENTS))}")
        if not _DATE.match(date or ""):
            raise ToolError("PARAM: date 需为 YYYY-MM-DD")
        key = "book:" + _require(idempotency_key, "idempotency_key")
        status, cached = idem.begin(key, ttl=60)
        if status == "done":
            return {**cached, "replayed": True}
        if status == "in_progress":
            raise ToolError("CONFLICT: 同一 idempotency_key 的预约正在处理中")
        appt = {"appointment_id": "A" + uuid.uuid4().hex[:10], "department": dep, "date": date,
                "patient_ref": patient_ref, "status": "booked", "created_at": time.time()}
        appointments[appt["appointment_id"]] = appt
        idem.complete(key, appt, ttl=86400)
        return {**appt, "replayed": False}

    @server.tool(annotations=SIDE_EFFECT, description="取消预约（Saga 补偿动作，幂等：重复取消返回同一结果）")
    def cancel_appointment(appointment_id: str, idempotency_key: str) -> dict[str, Any]:
        aid = _require(appointment_id, "appointment_id")
        key = "cancel:" + _require(idempotency_key, "idempotency_key")
        status, cached = idem.begin(key, ttl=60)
        if status == "done":
            return {**cached, "replayed": True}
        appt = appointments.get(aid)
        if appt is None:
            idem.fail(key)
            raise ToolError(f"PARAM: 预约 {aid} 不存在")
        appt["status"] = "cancelled"
        out = {"appointment_id": aid, "status": "cancelled"}
        idem.complete(key, out, ttl=86400)
        return {**out, "replayed": False}

    # ------------------------------------------------------------------ Resources：只读知识，按 URI 寻址
    # 工具是“让模型做一件事”，资源是“让宿主读一份数据”。知识库条目做成资源后，宿主可以不经模型直接读取
    # （确定性兜底、上下文注入、审核界面），也能在 DSH 里通过 mcp-resources 的共享工具读到。
    def _entry(section: str, key: str) -> dict[str, Any]:
        table = {"symptoms": kb.symptoms, "drugs": kb.drugs, "guidelines": kb.guidelines}[section]
        kind = {"symptoms": "symptom", "drugs": "drug", "guidelines": "disease"}[section]
        std = key if key in table else kb.normalize(key, kind)
        if std not in table:
            from mcp.server.mcpserver.exceptions import ResourceNotFoundError
            uri_kind = {"symptoms": "symptom", "drugs": "drug", "guidelines": "guideline"}[section]
            raise ResourceNotFoundError(f"kb://{uri_kind}/{key} not found")
        return {"key": std, **table[std], "provenance": kb.source_of(section, std), "kb_version": kb.version}

    @server.resource("kb://version", description="知识库版本号（RAG 缓存键的一部分）")
    def kb_version() -> str:
        return kb.version

    @server.resource("kb://catalog", description="知识库目录：全部症状 / 药品 / 指南条目名", mime_type="application/json")
    def kb_catalog() -> dict[str, Any]:
        return {"version": kb.version, "symptoms": list(kb.symptoms), "drugs": list(kb.drugs),
                "guidelines": list(kb.guidelines), "meta": kb.meta}

    @server.resource("kb://provenance", description="每个条目的来源与审核状态", mime_type="application/json")
    def kb_provenance() -> dict[str, Any]:
        return {"meta": kb.meta, "provenance": kb.provenance}

    @server.resource("kb://drug/{name}", description="单个药品条目（支持商品名）", mime_type="application/json")
    def kb_drug(name: str) -> dict[str, Any]:
        return _entry("drugs", name)

    @server.resource("kb://symptom/{name}", description="单个症状条目（支持口语说法）", mime_type="application/json")
    def kb_symptom(name: str) -> dict[str, Any]:
        return _entry("symptoms", name)

    @server.resource("kb://guideline/{topic}", description="单个指南条目", mime_type="application/json")
    def kb_guideline(topic: str) -> dict[str, Any]:
        return _entry("guidelines", topic)

    # 知识源概况与检索命中的全文。注意分工：
    #   - “查”（按问题检索本地 329 万篇索引 / 在线文献）是工具 search_knowledge——要模型决定查什么、带参数、结果随查询变；
    #   - “读”（按 ID 取某一篇的全文、看知识源状态）是资源——地址固定、只读、宿主不经模型就能读。
    #   在线文献（PubMed / Europe PMC）存在对方服务器上，只能实时检索，不做成资源。
    def _fts5_index():
        if "fts5" not in rag_holder:
            from ..rag.pipeline import REPO_ROOT
            path = os.environ.get("MEDAGENT_FTS5_INDEX", str(REPO_ROOT / ".cache" / "index" / "medical.fts5"))
            if not os.path.exists(path):
                rag_holder["fts5"] = None
            else:
                from ..rag.scalable import FTS5Index
                rag_holder["fts5"] = FTS5Index(path)
        return rag_holder["fts5"]

    @server.resource("kb://sources", description="知识源三层的规模与可用状态（精选 KB / Wikidata 实体层 / 本地全文索引 / 在线文献）",
                     mime_type="application/json")
    def kb_sources() -> dict[str, Any]:
        from ..knowledge.ext import load_ext
        from ..rag.pipeline import REPO_ROOT
        ext = load_ext()
        idx = _fts5_index()
        online = os.environ.get("MEDAGENT_ONLINE_KB") == "1"
        return {
            "curated": {"symptoms": len(kb.symptoms), "drugs": len(kb.drugs), "guidelines": len(kb.guidelines),
                        "version": kb.version},
            "wikidata": {"available": ext is not None,
                         **({"diseases": len(ext.diseases), "drugs": len(ext.drugs)} if ext is not None else {})},
            "local_fulltext": {"available": idx is not None,
                               "path": os.path.relpath(idx.path, REPO_ROOT) if idx is not None else None,
                               "build": "python -m medagent.rag.scalable setup"},
            "online": {"enabled": online, "sources": ["openFDA", "PubMed", "Europe PMC", "MedlinePlus"],
                       "enable": "MEDAGENT_ONLINE_KB=1"},
        }

    @server.resource("kb://doc/{doc_id}", description="search_knowledge 命中的本地全文索引文档全文（doc_id 形如 fts5:123）",
                     mime_type="application/json")
    def kb_doc(doc_id: str) -> dict[str, Any]:
        from mcp.server.mcpserver.exceptions import ResourceNotFoundError
        m = re.fullmatch(r"(?:fts5:)?(\d+)", doc_id)
        idx = _fts5_index()
        docs = idx.get([int(m.group(1))]) if (m and idx is not None) else []
        if not docs:
            raise ResourceNotFoundError(f"kb://doc/{doc_id} not found")
        d = docs[0]
        return {"id": f"fts5:{m.group(1)}", "source": d.source, "origin_id": d.id, "title": d.title, "text": d.text}

    # ------------------------------------------------------------------ Prompts：可复用的提示词模板
    # 提示词放在 Server 端，和工具定义同一处维护；Python 编排器通过 prompts/get 取用。
    # （DSH 的 MCP 客户端目前不支持 prompts，DSH 侧用 server instructions + cordis.yml 的 persona。）
    @server.prompt(description="医疗助手系统提示词（Python 编排器与评测共用的唯一来源）")
    def medical_system_prompt() -> str:
        return SYSTEM_PROMPT

    @server.prompt(description="信息不足时的结构化追问模板")
    def triage_clarify(symptom: str = "") -> str:
        head = f"关于「{symptom}」，" if symptom else ""
        return (f"{head}请向患者追问且只追问缺失的关键信息：部位与性质、持续时间、严重程度、伴随症状、年龄/孕期、"
                "慢病与在用药、过敏史；同时检查是否存在急症红旗征（胸痛、呼吸困难、意识改变、肢体无力、大出血），"
                "如有立即建议拨打 120。不要假设患者没有提供的信息。")

    @server.prompt(description="基于检索证据作答：只能使用证据中的事实并标注出处")
    def answer_with_evidence(question: str, evidence: str) -> str:
        return (f"问题：{question}\n证据：\n{evidence}\n\n要求：只使用证据中的事实作答，剂量必须与证据一致，"
                "每条结论后用（来源：…）标注出处；证据不足以回答时直接说明“资料不足”，不要补充证据外的内容。")

    server._medagent_state = {"appointments": appointments}  # 仅供测试 / 混沌评测统计重复副作用
    return server


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true", help="以 streamable-http 方式启动（默认 stdio）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    server = build_server()
    if args.http:
        server.run("streamable-http", host=args.host, port=args.port)
    else:
        server.run("stdio")


if __name__ == "__main__":
    main()
