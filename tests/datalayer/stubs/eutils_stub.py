"""A local NCBI E-utilities stub: canned esearch/efetch/esummary answers and a request log.

``start(log_path)`` serves on 127.0.0.1 in a daemon thread and returns an :class:`EutilsStub`
whose ``base`` is what ``VBT_EUTILS_BASE`` should be set to. Every request is appended to the
log as one JSON line ``{"endpoint", "params"}``, so a test can assert that a refused call never
reached the service.

Like the live API, efetch treats a non-numeric id by its digits: ``id=PMC1234`` returns the
article with PMID 1234 (a 1975 paper unrelated to PMC1234; VERIFIED live 2026-10-06).
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from xml.sax.saxutils import escape

PATH = "/entrez/eutils"

ARTICLES: dict[str, dict[str, Any]] = {
    "1234": {"title": "Biochemical studies of the red cell in 1975", "journal": "Clin Chim Acta", "year": 1975,
             "month": "Jun", "abstract": "A 1975 study of erythrocyte enzymes."},
    "30595370": {"title": "PCSK9 inhibition and LDL cholesterol", "journal": "Lancet", "year": 2019,
                 "month": "Jan", "abstract": "Evolocumab lowered LDL cholesterol (NCT01764633)."},
}
SEARCHES: dict[str, list[str]] = {"pcsk9": ["30595370"]}


def article(pmid: str) -> dict[str, Any]:
    return ARTICLES.get(pmid) or {"title": f"Fixture article {pmid}", "journal": "Fixture J", "year": 2001,
                                  "month": "Feb", "abstract": f"Fixture abstract {pmid}."}


def efetch_xml(ids: list[str]) -> str:
    parts = ['<?xml version="1.0" ?>', "<PubmedArticleSet>"]
    for raw in ids:
        pmid = re.sub(r"\D", "", raw)
        if not pmid:
            continue
        a = article(pmid)
        parts.append(
            "<PubmedArticle><MedlineCitation><PMID Version=\"1\">{pmid}</PMID><Article><Journal>"
            "<JournalIssue><PubDate><Year>{year}</Year><Month>{month}</Month></PubDate></JournalIssue>"
            "<Title>{journal}</Title></Journal><ArticleTitle>{title}</ArticleTitle>"
            "<Abstract><AbstractText>{abstract}</AbstractText></Abstract>"
            "<PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList>"
            "</Article></MedlineCitation></PubmedArticle>".format(
                pmid=pmid, year=a["year"], month=a["month"], journal=escape(a["journal"]),
                title=escape(a["title"]), abstract=escape(a["abstract"])))
    parts.append("</PubmedArticleSet>")
    return "".join(parts)


def esummary_json(ids: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"uids": ids}
    for pmid in ids:
        a = article(pmid)
        result[pmid] = {"uid": pmid, "title": a["title"], "fulljournalname": a["journal"],
                        "pubdate": f"{a['year']} {a['month']}", "epubdate": "",
                        "authors": [{"name": "Fixture A"}], "articleids": [{"idtype": "pubmed", "value": pmid}]}
    return {"result": result}


def esearch_json(term: str) -> dict[str, Any]:
    ids = SEARCHES.get(term.strip().casefold(), [])
    return {"esearchresult": {"count": str(len(ids)), "retmax": str(len(ids)), "idlist": ids}}


class _Handler(BaseHTTPRequestHandler):
    server: "_Server"

    def log_message(self, format: str, *args: Any) -> None:   # noqa: A002 - BaseHTTPRequestHandler API
        return

    def do_GET(self) -> None:   # noqa: N802 - BaseHTTPRequestHandler API
        url = urlparse(self.path)
        params = {k: v[-1] for k, v in parse_qs(url.query).items()}
        endpoint = url.path.rsplit("/", 1)[-1]
        self.server.record({"endpoint": endpoint, "params": params})
        ids = [i for i in (params.get("id") or "").split(",") if i]
        if endpoint == "efetch.fcgi":
            self._send(efetch_xml(ids), "text/xml")
        elif endpoint == "esummary.fcgi":
            self._send(json.dumps(esummary_json(ids)), "application/json")
        elif endpoint == "esearch.fcgi":
            self._send(json.dumps(esearch_json(params.get("term", ""))), "application/json")
        else:
            self.send_error(404)

    def _send(self, body: str, ctype: str) -> None:
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, log_path: Path) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.log_path = log_path
        self._lock = threading.Lock()

    def record(self, entry: dict[str, Any]) -> None:
        with self._lock, self.log_path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")


@dataclass
class EutilsStub:
    server: _Server
    thread: threading.Thread
    log_path: Path

    @property
    def base(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}{PATH}"

    def requests(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        return [json.loads(line) for line in self.log_path.read_text().splitlines() if line.strip()]

    def clear(self) -> None:
        self.log_path.write_text("")

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def start(log_path: str | Path) -> EutilsStub:
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.touch()
    server = _Server(log)
    thread = threading.Thread(target=server.serve_forever, name="eutils-stub", daemon=True)
    thread.start()
    return EutilsStub(server, thread, log)
