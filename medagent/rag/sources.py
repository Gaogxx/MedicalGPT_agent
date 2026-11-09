"""在线开源医疗知识源（可选路由，默认关闭：MEDAGENT_ONLINE_KB=1 开启）。

选型（均为免费、官方、可程序化访问）：
  - openFDA Drug Label API     药品说明书（适应症 / 禁忌 / 警告 / 用法），美国 FDA，公共领域
  - NCBI PubMed E-utilities    医学文献检索（标题 / PMID），用于“最新证据”类问题
  - MedlinePlus Web Service    NLM 面向大众的健康主题科普
中文侧没有同等开放的官方 API（国家药监局说明书库无公开接口；CMeKG 不提供批量开放），
因此中文知识走本地索引：内置 KB + 可下载的开源数据（见 docs/rag_sources.md 的 fetch 脚本）。
每个在线源都包在独立熔断器 + 超时里：在线源挂了只影响这一路召回，不拖垮整次检索。
"""
from __future__ import annotations

import logging
import re
from typing import Any

from ..resilience.breaker import CircuitBreaker
from ..types import AgentError
from .corpus import Doc

log = logging.getLogger(__name__)


class OnlineSource:
    name = "online"

    def __init__(self, breaker: CircuitBreaker | None = None, timeout: float = 4.0, client: Any = None) -> None:
        import httpx
        self.http = client or httpx.Client(timeout=timeout, headers={"User-Agent": "medagent-rag/0.2"})
        self.breaker = breaker or CircuitBreaker(f"rag:{self.name}")

    def fetch(self, query: str, k: int) -> list[Doc]:  # pragma: no cover - 子类实现
        raise NotImplementedError

    def search(self, query: str, k: int = 5) -> list[Doc]:
        try:
            probe = self.breaker.acquire()
        except AgentError:
            return []
        try:
            docs = self.fetch(query, k)
        except Exception as exc:  # noqa: BLE001
            from ..resilience.errors import classify_exception
            err = classify_exception(exc, source=f"rag:{self.name}")
            self.breaker.on_failure(err, probe)
            log.warning("online source %s failed: %s", self.name, err)
            return []
        self.breaker.on_success(probe)
        return docs


class OpenFDALabel(OnlineSource):
    name = "openfda"
    URL = "https://api.fda.gov/drug/label.json"

    def fetch(self, query: str, k: int) -> list[Doc]:
        from ..knowledge import load_kb
        kb = load_kb()
        generics = [v["generic_name"] for n, v in kb.drugs.items() if n in query or kb.normalize(query, "drug") == n]
        generics += re.findall(r"[A-Za-z][A-Za-z\-]{3,}", query)
        docs = []
        for g in dict.fromkeys(generics):
            r = self.http.get(self.URL, params={"search": f'openfda.generic_name:"{g.lower()}"', "limit": 1})
            if r.status_code == 404:
                continue
            r.raise_for_status()
            for res in r.json().get("results", [])[:1]:
                parts = []
                for field in ("indications_and_usage", "contraindications", "warnings", "dosage_and_administration"):
                    if res.get(field):
                        parts.append(f"{field}: {res[field][0][:600]}")
                docs.append(Doc(f"openfda:{res.get('id', g)}", " ".join(parts), title=f"FDA label: {g}",
                                source="openFDA drug label"))
        return docs[:k]


class PubMedSearch(OnlineSource):
    name = "pubmed"
    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def fetch(self, query: str, k: int) -> list[Doc]:
        term = " ".join(re.findall(r"[A-Za-z][A-Za-z\-]+", query)) or query
        r = self.http.get(f"{self.BASE}/esearch.fcgi", params={"db": "pubmed", "term": term, "retmode": "json",
                                                               "retmax": k, "sort": "relevance"})
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        s = self.http.get(f"{self.BASE}/esummary.fcgi", params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"})
        s.raise_for_status()
        res = s.json().get("result", {})
        return [Doc(f"pubmed:{i}", res[i].get("title", ""), title=res[i].get("source", ""),
                    source=f"PubMed PMID {i} ({res[i].get('pubdate', '')})") for i in ids if i in res]


class MedlinePlusTopics(OnlineSource):
    name = "medlineplus"
    URL = "https://wsearch.nlm.nih.gov/ws/query"

    def fetch(self, query: str, k: int) -> list[Doc]:
        term = " ".join(re.findall(r"[A-Za-z][A-Za-z\-]+", query)) or query
        r = self.http.get(self.URL, params={"db": "healthTopics", "term": term, "retmax": k})
        r.raise_for_status()
        docs = []
        for m in re.finditer(r'<document[^>]*url="([^"]+)".*?<content name="title">(.*?)</content>.*?'
                             r'<content name="FullSummary">(.*?)</content>', r.text, re.S):
            text = re.sub(r"<[^>]+>|&lt;[^&]*&gt;", "", m.group(3))[:800]
            title = re.sub(r"<[^>]+>|&lt;[^&]*&gt;", "", m.group(2))
            docs.append(Doc(f"medlineplus:{m.group(1)}", text, title=title, source=f"MedlinePlus {m.group(1)}"))
        return docs[:k]


class EuropePMCSearch(OnlineSource):
    """Europe PMC REST（约 4,900 万条文献，含 PubMed 与预印本、部分开放全文），免费、无需 key。"""
    name = "europepmc"
    URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def __init__(self, breaker=None, timeout: float = 10.0, client=None) -> None:   # Europe PMC 响应较慢，单独放宽超时
        super().__init__(breaker, timeout, client)

    def fetch(self, query: str, k: int) -> list[Doc]:
        term = " ".join(re.findall(r"[A-Za-z][A-Za-z\-]+", query)) or query
        r = self.http.get(self.URL, params={"query": term, "format": "json", "pageSize": k, "resultType": "lite"})
        r.raise_for_status()
        out = []
        for it in r.json().get("resultList", {}).get("result", [])[:k]:
            out.append(Doc(f"europepmc:{it.get('source')}:{it.get('id')}", it.get("title", ""),
                           title=it.get("journalTitle", ""),
                           source=f"Europe PMC {it.get('source')} {it.get('id')} ({it.get('pubYear', '')})"))
        return out


class FTS5Source:
    """百万级本地全文索引作为一路召回（medagent/rag/scalable.py 构建）。"""
    name = "fts5"

    def __init__(self, path: str) -> None:
        from .scalable import FTS5Index
        self.index = FTS5Index(path)

    def search(self, query: str, k: int = 20) -> list[Doc]:
        hits = self.index.search(query, k=k)
        docs = self.index.get([h[0] for h in hits])
        for d in docs:
            d.id = f"fts5:{d.id}"
        return docs
