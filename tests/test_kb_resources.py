"""MCP 资源：知识源状态 kb://sources、检索命中文档全文 kb://doc/{id}。"""
from medagent.mcp_client import MCPToolbox
from medagent.mcp_servers.medical_server import build_server


async def test_kb_sources_and_doc_resources(tmp_path, monkeypatch):
    from medagent.rag.scalable import FTS5Index
    path = tmp_path / "mini.fts5"
    filler = [(f"f:{i}", "测试来源", f"条目{i}", f"第{i}号无关条目，记录天气与交通。") for i in range(60)]  # 检索会忽略 df>5% 的词
    FTS5Index(path).build([("a:1", "测试来源", "布洛芬是什么", "布洛芬是一种非甾体抗炎药，可缓解轻中度疼痛。"),
                           ("a:2", "测试来源", "高血压饮食", "高血压患者应限制钠盐摄入。")] + filler, workers=1)
    monkeypatch.setenv("MEDAGENT_FTS5_INDEX", str(path))
    async with MCPToolbox({"medical": build_server()}) as tb:
        src = await tb.read_resource("kb://sources")
        assert src["local_fulltext"]["available"] and src["curated"]["drugs"] >= 10
        r = await tb.call("search_knowledge", {"query": "非甾体抗炎药缓解疼痛", "top_k": 10})
        ids = [e["id"] for e in r.data["evidence"] if e["id"].startswith("fts5:")]
        assert ids, r.data
        doc = await tb.read_resource(f"kb://doc/{ids[0]}")
        assert doc["text"].startswith("布洛芬是一种") and doc["origin_id"] == "a:1"
        assert await tb.read_resource("kb://doc/fts5:999999") is None
