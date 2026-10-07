"""Third-party servers (phase 4, F21): an HTTP-style fixture service behind an overlay, the generic guard,
HTTP failure classification and the ``vbt ds overlay init`` scaffold.

The fixture is a real HTTP server on 127.0.0.1 (``http.server``) with a UniProt-like API: entries by
accession (404 with a JSON message for unknown ones, 503 for a broken one), a search with a
``totalCount``, and an unreviewed listing endpoint. The "MCP server" in front of it is a function that
calls the API and returns what typical third-party wrappers return on failure
(``{"error": "Failed to retrieve data: status code: 404", ...}``), driven through the gateway with the
fake data child of ``test_dl_gateway_flow``.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import yaml

from test_dl_gateway_flow import REGISTRY, call, hdr, make_gateway, raw_of
from vbt.datalayer.catalog import Catalog
from vbt.datalayer.cli import THIRD_PARTY_HOWTO, cmd_overlay_init, lint_scaffold, scaffold_overlay
from vbt.datalayer.descriptor.overlay import Overlay
from vbt.datalayer.errors import ErrorKind, GatewayError
from vbt.datalayer.gateway.classify import classify

ENTRIES = {"P04637": {"primaryAccession": "P04637", "gene": "TP53", "length": 393},
           "Q8NBP7": {"primaryAccession": "Q8NBP7", "gene": "PCSK9", "length": 692}}
BROKEN = "A0A024RBG1"


class _Api(BaseHTTPRequestHandler):
    def _send(self, status: int, body: Any) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        url = urllib.parse.urlparse(self.path)
        q = dict(urllib.parse.parse_qsl(url.query))
        if url.path.startswith("/entry/"):
            acc = url.path.rsplit("/", 1)[-1]
            if acc == BROKEN:
                return self._send(503, {"status": 503, "message": "Service Unavailable"})
            if acc not in ENTRIES:
                return self._send(404, {"status": 404, "message": f"Entry {acc} not found"})
            return self._send(200, ENTRIES[acc])
        if url.path == "/search":
            size = int(q.get("size", "10"))
            hits = [ENTRIES[a] for a in sorted(ENTRIES)][:size]
            return self._send(200, {"totalCount": 42, "results": hits})
        if url.path == "/things":
            return self._send(200, {"count": 0, "results": []})
        return self._send(404, {"status": 404, "message": "no such endpoint"})

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture(scope="module")
def api():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Api)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def http_tool(base: str):
    """The third-party MCP server: one function per tool over the HTTP API (no proxy for localhost)."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def get(path: str, **params: Any) -> Any:
        url = f"{base}{path}" + (f"?{urllib.parse.urlencode(params)}" if params else "")
        try:
            with opener.open(url, timeout=10) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = json.loads(exc.read() or b"{}")
            return {"error": f"Failed to retrieve data: status code: {exc.code}", "message": body.get("message")}

    def tool(name: str, args: dict[str, Any]) -> Any:
        if name == "get_entry":
            return get(f"/entry/{args['accession']}")
        if name == "search":
            return get("/search", query=args.get("query", ""), size=args.get("size", 10))
        return get("/things")
    return tool


SOURCE: dict[str, Any] = {
    "schema": "vbt.datasource/1", "source": "uniprot_rest", "title": "UniProt REST (fixture)", "kind": "remote",
    "release": {"from": "as_of"}, "defaults": {"format": "none", "layout": "upstream_only"},
    "id_types": {"uniprot_accession": {"plugin": "uniprot_accession", "resolvable": False}},
    "tables": {"entry": {"kind": "entity", "grain": "one UniProtKB entry", "key": {"columns": ["primaryAccession"],
                                                                                    "check": "none"},
                         "columns": {"primaryAccession": {"role": "identifier", "id_type": "uniprot_accession",
                                                          "self": True},
                                     "gene": {"role": "label", "of": "primaryAccession"},
                                     "length": {"role": "count", "counts": "residues"}}}},
}

OVERLAY: dict[str, Any] = {
    "schema": "vbt.overlay/1", "server": "uniprot", "sources": ["uniprot_rest"],
    "tools": {
        "get_entry": {
            "reads": {"uniprot_rest.entry": {"access": "upstream"}},
            "args": {"accession": {"binds": "uniprot_rest.entry.primaryAccession", "accepts": ["uniprot_accession"],
                                   "existence": "upstream"}},
            "result": {"kind": "record", "rows": "$", "echo": {"accession": {"path": "$.primaryAccession"}},
                       "not_found_when": ["$.message =~ '(?i)not found'"]}},
        "search": {
            "reads": {"uniprot_rest.entry": {"access": "upstream"}},
            "args": {"query": {"role": "free_text", "interpreted_as": "engine"},
                     "size": {"role": "limit", "min": 1, "max": 500}},
            "result": {"rows": "$.results", "total": {"path": "$.totalCount", "method": "upstream"},
                       "row_key": ["primaryAccession"], "order_source": "source_server_side"}},
    },
}

GENERIC: dict[str, Any] = {"schema": "vbt.overlay/1", "server": "*",
                           "generic": {"not_found_when": ["$.error =~ '(?i)status code: 404'"],
                                       "empty_when": ["$.count == 0"]}}


@pytest.fixture
def gw(tmp_path: Path):
    return make_gateway(tmp_path, [SOURCE], [OVERLAY], {}, generic=[GENERIC])


async def test_not_found_when_from_an_http_404(gw, api) -> None:
    tool = http_tool(api)
    plan, res = await call(gw, "uniprot", "get_entry", {"accession": "P04637"}, lambda a: tool("get_entry", a))
    assert json.loads(res.text)["gene"] == "TP53" and hdr(res)["status"] == "ok"
    with pytest.raises(GatewayError) as e:
        await call(gw, "uniprot", "get_entry", {"accession": "Q9Y6Y9"}, lambda a: tool("get_entry", a))
    assert e.value.kind == ErrorKind.not_found and e.value.argument == "accession"


async def test_http_5xx_is_a_source_error_never_a_negative(gw, api) -> None:
    tool = http_tool(api)
    with pytest.raises(GatewayError) as e:
        await call(gw, "uniprot", "get_entry", {"accession": BROKEN}, lambda a: tool("get_entry", a))
    assert e.value.kind == ErrorKind.source_error and e.value.subkind == "http_status"
    assert e.value.payload["http_status"] == 503 and e.value.retryable == "later"


async def test_upstream_totals_and_rows(gw, api) -> None:
    tool = http_tool(api)
    plan, res = await call(gw, "uniprot", "search", {"query": "kinase", "size": 1}, lambda a: tool("search", a))
    h = hdr(res)
    assert h["status"] in ("ok", "partial") and h["total"] == 42
    assert len(json.loads(res.text)["results"]) == 1


async def test_unreviewed_tool_gets_the_generic_guard(gw, api) -> None:
    tool = http_tool(api)
    plan, res = await call(gw, "uniprot", "list_things", {}, lambda a: tool("list_things", a))
    assert hdr(res)["status"] == "empty_unverified"
    with pytest.raises(GatewayError) as e:
        await call(gw, "uniprot", "unknown_endpoint", {}, lambda a: {"error": "Failed: status code: 404"})
    assert e.value.kind == ErrorKind.not_found


def test_404_needs_a_declared_meaning() -> None:
    """A 404 on a reviewed binding without ``not_found_when`` is an error naming the status, not 'no record'."""
    ov = json.loads(json.dumps(OVERLAY))
    ov["tools"]["get_entry"]["result"].pop("not_found_when")
    from vbt.datalayer.descriptor.models import SourceDescriptor

    cat = Catalog({"uniprot_rest": SourceDescriptor.model_validate(SOURCE)}, {"uniprot": Overlay.model_validate(ov)},
                  [], registry=REGISTRY)
    c = cat.contract("uniprot", "get_entry")
    out = classify(raw_of({"error": "Failed to retrieve data: status code: 404"}), c, None)
    assert out.outcome == "source_error" and out.error.payload["http_status"] == 404
    out = classify(raw_of("Server error '502 Bad Gateway' for url 'https://x'", is_error=True), c, None)
    assert out.outcome == "source_error" and out.error.subkind == "http_status"


# --------------------------------------------------------------------------- overlay init

LISTING = [
    {"name": "get_entry", "description": "Get one UniProtKB entry.\n\nArgs:\n    accession: e.g. P04637 or Q8NBP7",
     "inputSchema": {"type": "object", "properties": {"accession": {"type": "string"}}, "required": ["accession"]}},
    {"name": "search", "description": "Search entries by text.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"},
                                                      "size": {"type": "integer", "default": 10},
                                                      "gene_id": {"type": "string", "description":
                                                                  "Ensembl gene, e.g. ENSG00000141510"}}}},
    {"name": "export", "description": "Write results to a file.",
     "inputSchema": {"type": "object", "properties": {"output_path": {"type": "string"}}}},
]


def test_scaffold_is_unreviewed_lints_and_guesses_kinds() -> None:
    text = scaffold_overlay("uniprot", LISTING, REGISTRY, url="http://127.0.0.1:1/mcp")
    assert text.startswith("# Overlay scaffold for the MCP server 'uniprot'")
    assert THIRD_PARTY_HOWTO.splitlines()[0] in text
    ov = Overlay.model_validate(yaml.safe_load(text))
    assert ov.server == "uniprot" and ov.match == {"url_prefix": "http://127.0.0.1:1/mcp"}
    assert sorted(ov.tools) == ["export", "get_entry", "search"]
    assert all(b.status == "unreviewed" for b in ov.tools.values())
    assert ov.tools["get_entry"].args["accession"].role == "unbound"
    assert ov.tools["search"].args["size"].role == "limit"
    assert ov.tools["search"].args["query"].role == "free_text"
    assert ov.tools["export"].args["output_path"].role == "output_path"
    acc_line = next(ln for ln in text.splitlines() if '"accession"' in ln)
    assert "guess: uniprot_accession" in acc_line
    gene_line = next(ln for ln in text.splitlines() if '"gene_id"' in ln)
    assert "ensembl_gene" in gene_line
    findings = lint_scaffold(text)
    assert not [f for f in findings if f.level == "error"], findings
    # loaded next to the shipped catalog, every scaffolded tool is served by the generic guard
    from vbt.datalayer.descriptor.models import SourceDescriptor

    cat = Catalog({"uniprot_rest": SourceDescriptor.model_validate(SOURCE)}, {"uniprot": ov},
                  [Overlay.model_validate(GENERIC)], registry=REGISTRY)
    assert cat.contract("uniprot", "get_entry").generic


def test_overlay_init_cli_from_json(tmp_path: Path, capsys) -> None:
    import argparse

    from vbt.config import load_config

    listing = tmp_path / "tools.json"
    listing.write_text(json.dumps({"tools": LISTING, "url": "http://127.0.0.1:1/mcp"}))
    out = tmp_path / "uniprot.yaml"
    config = load_config(["mock"])
    ns = argparse.Namespace(server="uniprot", from_json=str(listing), out=str(out), force=False)
    assert cmd_overlay_init(ns, config) == 0
    assert "written:" in capsys.readouterr().out
    assert Overlay.model_validate(yaml.safe_load(out.read_text())).tools["search"].status == "unreviewed"
    assert cmd_overlay_init(ns, config) == 2                    # never overwrites without --force
    assert "exists" in capsys.readouterr().err
    ns.force = True
    assert cmd_overlay_init(ns, config) == 0
