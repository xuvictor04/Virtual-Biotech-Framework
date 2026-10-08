"""Live and remote sources (phase 4, F20) against stubs: no network.

* A local HTTP stub plays ClinicalTrials.gov v2, cBioPortal and E-utilities; the shipped descriptors are
  loaded with their base URLs pointed at it (``${VBT_CTGOV_BASE}``, ``${VBT_CBIOPORTAL_BASE}``,
  ``${VBT_EUTILS_BASE}``).
* A stub ``cellxgene_census`` module plays the Census (``open_soma``, ``get_census_version_description``).

Covered: ``as_of`` read from the payload (and the ``Date`` header), the remote witness as one count request
and its contradiction becoming ``tool_defect`` through the gateway, Census release resolution with drift,
quoting of SOMA, Essie and E-utilities literals, count-first Census admission with donor-balanced sampling
keyed by ``(dataset_id, donor_id)``, cBioPortal patient-level find over the pivot, record-version changes
flagged ``source_updated``, and the PubMed server's reconciliation of requested and returned PMIDs.
"""

from __future__ import annotations

import copy
import json
import sys
import threading
import types
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from test_dl_gateway_flow import REGISTRY, call, make_gateway, raw_of
from vbt.datalayer.catalog import Catalog
from vbt.datalayer.descriptor.load import load_yaml
from vbt.datalayer.descriptor.models import SourceDescriptor
from vbt.datalayer.descriptor.overlay import Overlay
from vbt.datalayer.errors import ErrorKind, GatewayError
from vbt.datalayer.gateway.classify import classify
from vbt.datalayer.gateway.soma_filter import parse as parse_soma
from vbt.datalayer.ipc import WitnessRequest, WitnessResponse
from vbt.datalayer.plugins.base import LayoutSpec
from vbt.datalayer.plugins.conformance.golden import F9_LITERALS
from vbt.datalayer.plugins.formats.rest_json import RestJsonFormat, essie_quote, eutils_quote
from vbt.datalayer.plugins.formats.soma import SomaFormat
from vbt.datalayer.plugins.layouts import live_api as live
from vbt.datalayer.plugins.layouts import soma as soma_layout
from vbt.datalayer.plugins.layouts.live_api import Budget, RecordVersions, RemoteError, RemoteVocab
from vbt.datalayer.predicate import And, Cmp, Contains, Eq, In, IsNull, Not, Or, Range, TextMatch, to_json
from vbt.datalayer.service import ServiceContext
from vbt.datalayer.service.verbs import load_verbs
from vbt.datalayer.service.verbs import witness as witness_verbs
from vbt.datalayer.service.verbs.census_count import donor_balanced
from vbt.datalayer.settings import DataSettings

REPO = Path(__file__).resolve().parents[2]
SOURCES = REPO / "configs" / "data" / "sources"
OVERLAYS = REPO / "configs" / "data" / "overlays"

STUDY = {"protocolSection": {"identificationModule": {"nctId": "NCT01234567"},
                             "statusModule": {"overallStatus": "RECRUITING",
                                              "lastUpdatePostDateStruct": {"date": "2024-05-01"}}}}


class Stub:
    """Mutable answers of the stub API and the requests it saw."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, str]]] = []
        self.count = 17
        self.study_version = "2024-05-01"
        self.import_date = "2026-06-05 15:19:54"
        self.clinical: list[dict[str, Any]] = [
            {"studyId": "s1", "patientId": "p1", "clinicalAttributeId": "OS_MONTHS", "value": "12.5"},
            {"studyId": "s1", "patientId": "p1", "clinicalAttributeId": "OS_STATUS", "value": "1:DECEASED"},
            {"studyId": "s1", "patientId": "p2", "clinicalAttributeId": "OS_MONTHS", "value": "30"},
            {"studyId": "s1", "patientId": "p2", "clinicalAttributeId": "OS_STATUS", "value": "0:LIVING"},
        ]


def _handler(stub: Stub) -> type:
    class H(BaseHTTPRequestHandler):
        def _send(self, status: int, body: Any, headers: dict[str, str] | None = None) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            url = urllib.parse.urlparse(self.path)
            q = dict(urllib.parse.parse_qsl(url.query))
            stub.requests.append((url.path, q))
            if url.path == "/ctgov/version":
                return self._send(200, {"apiVersion": "2.0.3", "dataTimestamp": "2026-10-01T09:30:00"})
            if url.path == "/ctgov/studies":
                if q.get("countTotal") == "true":
                    return self._send(200, {"totalCount": stub.count, "studies": []})
                study = copy.deepcopy(STUDY)
                study["protocolSection"]["statusModule"]["lastUpdatePostDateStruct"]["date"] = stub.study_version
                if q.get("pageToken") == "p2":
                    return self._send(200, {"studies": [study], "totalCount": 2})
                return self._send(200, {"studies": [study], "totalCount": 2, "nextPageToken": "p2"})
            if url.path == "/cbio/studies/s1/clinical-data":
                return self._send(200, stub.clinical)
            if url.path == "/cbio/studies/s1":
                return self._send(200, {"studyId": "s1", "name": "Study one", "importDate": stub.import_date})
            if url.path == "/cbio/studies/s1/samples":
                if q.get("projection") == "META":
                    return self._send(200, {}, {"total-count": "5"})
                return self._send(200, [{"studyId": "s1", "sampleId": "x"}])
            if url.path == "/eutils/esearch.fcgi":
                return self._send(200, {"esearchresult": {"count": "23", "idlist": []}})
            return self._send(404, {"message": "no such endpoint"})

        def log_message(self, *args: Any) -> None:
            pass
    return H


@pytest.fixture(scope="module")
def stub():
    state = Stub()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state.base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


def _variables(stub: Stub) -> dict[str, str]:
    return {"env.VBT_CTGOV_BASE": f"{stub.base}/ctgov", "env.VBT_CBIOPORTAL_BASE": f"{stub.base}/cbio",
            "env.VBT_EUTILS_BASE": f"{stub.base}/eutils"}


def _source(name: str, stub: Stub) -> dict[str, Any]:
    return load_yaml(SOURCES / f"{name}.yaml", _variables(stub))


@pytest.fixture
def ctx(stub: Stub, tmp_path: Path) -> ServiceContext:
    descs = {d["source"]: SourceDescriptor.model_validate(d)
             for d in (_source(n, stub) for n in ("clinicaltrials", "cbioportal", "pubmed", "census"))}
    settings = DataSettings.from_dict({"cache_dir": str(tmp_path / "cache")}, project_root=tmp_path)
    return ServiceContext(settings, catalog=Catalog(descs, {}, [], registry=REGISTRY), registry=REGISTRY)


def _spec(ctx: ServiceContext, ref: str) -> LayoutSpec:
    from vbt.datalayer.service import layout_spec

    return layout_spec(ctx.table(ref))


# --------------------------------------------------------------------------- as_of, pages, budgets


def test_as_of_comes_from_the_payload(ctx: ServiceContext, stub: Stub) -> None:
    layout = ctx.plugin("layout", "live_api")
    page = layout.request(_spec(ctx, "clinicaltrials_gov.version"), predicate=None, projection=[], page_token=None,
                          budget=None)
    assert page.as_of == "2026-10-01T09:30:00" and page.rows[0]["dataTimestamp"] == "2026-10-01T09:30:00"
    assert layout.release(_spec(ctx, "clinicaltrials_gov.studies")) == "2026-10-01T09:30:00"
    studies = layout.request(_spec(ctx, "clinicaltrials_gov.studies"), predicate=None, projection=[],
                             page_token=None, budget=None)
    assert studies.as_of and studies.as_of.endswith("Z")            # the Date header, as ISO UTC
    assert studies.total == 2 and studies.next == "p2"
    assert layout.as_of("", _spec(ctx, "clinicaltrials_gov.studies")) == studies.as_of


def test_pages_stop_at_the_budget_and_say_truncated(ctx: ServiceContext, stub: Stub) -> None:
    layout = ctx.plugin("layout", "live_api")
    spec = _spec(ctx, "clinicaltrials_gov.studies")
    got = live.fetch_all(layout, spec, None, {"max_pages": 1})
    assert got["truncated"] and got["pages"] == 1 and len(got["rows"]) == 1
    got = live.fetch_all(layout, spec, None, {"max_pages": 10})
    assert not got["truncated"] and got["pages"] == 2 and len(got["rows"]) == 2
    with pytest.raises(RemoteError, match="request budget"):
        live.fetch_all(layout, spec, None, Budget(max_requests_per_call=1))


# --------------------------------------------------------------------------- remote witness


def test_remote_witness_is_one_count_request(ctx: ServiceContext, stub: Stub) -> None:
    stub.requests.clear()
    pred = And((In("protocolSection.statusModule.overallStatus", ("RECRUITING", "COMPLETED")),
                Contains("protocolSection.designModule.phases[]", "PHASE2")))
    out = load_verbs()["_witness"](ctx, {"table": "clinicaltrials_gov.studies", "predicate": to_json(pred)})
    assert out["total"] == 17 and out["total_method"] == "scan" and out["reason"] == "remote count request"
    assert len(stub.requests) == 1
    path, q = stub.requests[0]
    assert path == "/ctgov/studies" and q["countTotal"] == "true" and q["pageSize"] == "0"
    assert q["filter.overallStatus"] == "RECRUITING,COMPLETED" and q["filter.advanced"] == "AREA[Phase]PHASE2"
    # a predicate the source cannot express is unknown, never a guess
    out = load_verbs()["_witness"](ctx, {"table": "clinicaltrials_gov.studies",
                                         "predicate": to_json(IsNull("protocolSection.statusModule.whyStopped"))})
    assert out["total"] is None and out["total_method"] == "unknown"


def test_pubmed_witness_is_an_esearch_count(ctx: ServiceContext, stub: Stub) -> None:
    stub.requests.clear()
    pred = And((TextMatch("title", "PCSK9 AND evolocumab", "word"), Range("pubdate", None, "2025-01-31")))
    out = load_verbs()["_witness"](ctx, {"table": "pubmed.records", "predicate": to_json(pred)})
    assert out["total"] == 23
    path, q = stub.requests[-1]
    assert path == "/eutils/esearch.fcgi" and q["rettype"] == "count" and q["term"] == "PCSK9 AND evolocumab"
    assert q["maxdate"] == "2025/01/31" and q["mindate"] == "1800/01/01" and q["datetype"] == "pdat"
    assert q["db"] == "pubmed"


def test_cbioportal_count_from_the_total_count_header(ctx: ServiceContext, stub: Stub) -> None:
    out = load_verbs()["_witness"](ctx, {"table": "cbioportal.sample", "predicate": to_json(Eq("studyId", "s1"))})
    assert out["total"] == 5
    assert stub.requests[-1] == ("/cbio/studies/s1/samples", {"projection": "META"})
    out = load_verbs()["_witness"](ctx, {"table": "cbioportal.sample", "predicate": None})
    assert out["total_method"] == "unknown" and "studyId" in out["reason"]


CT_COUNT_OVERLAY = {"schema": "vbt.overlay/1", "server": "clinicaltrials", "sources": ["clinicaltrials_gov"],
                    "tools": {"count_clinical_trials": {
                        "reads": {"clinicaltrials_gov.studies": {"access": "upstream"}},
                        "args": {"status": {"binds": "clinicaltrials_gov.studies.protocolSection.statusModule."
                                                     "overallStatus", "op": "in", "each": True}},
                        "result": {"kind": "count", "rows": None, "total": {"path": "$.total_count",
                                                                             "method": "upstream"}},
                        "serve": "pass"}}}


async def test_remote_witness_contradiction_is_tool_defect(stub: Stub, tmp_path: Path) -> None:
    """count_clinical_trials answers 0 (a reply without totalCount, CT-GOV-004); the independent count
    request counts 17: the gateway withholds the count as tool_defect."""
    desc = _source("clinicaltrials", stub)
    gw = make_gateway(tmp_path, [desc], [CT_COUNT_OVERLAY], {})
    ctx = ServiceContext(gw.settings, catalog=gw.catalog, registry=REGISTRY)
    gw.service.witness_hook = lambda req, resp: WitnessResponse.model_validate(
        load_verbs()["_witness"](ctx, WitnessRequest.model_validate(req.model_dump()).model_dump(mode="json")))
    stub.count = 17
    with pytest.raises(GatewayError) as e:
        await call(gw, "clinicaltrials", "count_clinical_trials", {"status": ["RECRUITING"]},
                   lambda a: {"total_count": 0})
    assert e.value.kind == ErrorKind.tool_defect and "independent count request counted 17" in e.value.message
    assert e.value.payload["witness"]["total"] == 17
    assert [p for p, _q in stub.requests if p == "/ctgov/studies"][-1:] == ["/ctgov/studies"]


async def test_shipped_count_tool_gets_the_remote_witness(stub: Stub, tmp_path: Path) -> None:
    """F20: the shipped clinicaltrials overlay (``access: upstream`` on a ``kind: remote`` table) runs through
    the unmodified gateway; the live_api layout's ``count`` capability sends the independent count request
    and a 0 against 17 (CT-GOV-004, totalCount missing) is a tool_defect."""
    desc = _source("clinicaltrials", stub)
    ov = load_yaml(SOURCES.parent / "overlays" / "clinicaltrials.yaml", _variables(stub))
    gw = make_gateway(tmp_path, [desc], [ov], {})
    ctx = ServiceContext(gw.settings, catalog=gw.catalog, registry=REGISTRY)
    gw.service.witness_hook = lambda req, resp: WitnessResponse.model_validate(
        load_verbs()["_witness"](ctx, WitnessRequest.model_validate(req.model_dump()).model_dump(mode="json")))
    contract = gw.catalog.contract("clinicaltrials", "count_clinical_trials")
    assert not gw._scannable(contract, "clinicaltrials_gov.studies")       # upstream access on a remote table
    assert gw._remote_countable(contract, "clinicaltrials_gov.studies")     # ... with a count capability
    stub.count = 17
    with pytest.raises(GatewayError) as e:
        await call(gw, "clinicaltrials", "count_clinical_trials", {"status": ["RECRUITING"], "country": None},
                   lambda a: {"total_count": 0})
    assert e.value.kind == ErrorKind.tool_defect and "independent count request counted 17" in e.value.message
    plan, res = await call(gw, "clinicaltrials", "count_clinical_trials", {"status": ["RECRUITING"], "country": None},
                           lambda a: {"total_count": 17})
    assert res.header.get("total") == 17


async def test_engine_text_goes_to_its_request_parameter(stub: Stub, tmp_path: Path) -> None:
    """Free text the source's engine matches (``condition``, ``advanced_filter``) is sent to the request parameter
    it fills (``engine_param``), so the realistic CT.gov count gets an independent count of the same search; the
    advanced filter keeps the caller's text, and the evidence ceiling is the witness's own conjunct. Engine text bound
    to a column (``eligibility_text``) is counted as an exact phrase on that column's field."""
    desc = _source("clinicaltrials", stub)
    ov = load_yaml(SOURCES.parent / "overlays" / "clinicaltrials.yaml", _variables(stub))
    gw = make_gateway(tmp_path, [desc], [ov], {})
    ctx = ServiceContext(gw.settings, catalog=gw.catalog, registry=REGISTRY)
    gw.service.witness_hook = lambda req, resp: WitnessResponse.model_validate(
        load_verbs()["_witness"](ctx, WitnessRequest.model_validate(req.model_dump()).model_dump(mode="json")))
    stub.count = 17
    args = {"condition": "glioblastoma", "status": ["RECRUITING"], "country": None,
            "advanced_filter": "AREA[Phase]PHASE3"}
    seen = len(stub.requests)
    with pytest.raises(GatewayError) as e:
        await call(gw, "clinicaltrials", "count_clinical_trials", args, lambda a: {"total_count": 0})
    assert e.value.kind == ErrorKind.tool_defect and "independent count request counted 17" in e.value.message
    counts = [q for p, q in stub.requests[seen:] if p == "/ctgov/studies" and q.get("countTotal") == "true"]
    assert len(counts) == 1, stub.requests[seen:]
    assert counts[0]["query.cond"] == "(glioblastoma)" and counts[0]["filter.overallStatus"] == "RECRUITING"
    assert counts[0]["filter.advanced"] == "(AREA[Phase]PHASE3)", counts[0]
    plan, res = await call(gw, "clinicaltrials", "count_clinical_trials", args, lambda a: {"total_count": 17})
    assert res.header.get("total") == 17 and plan.witness["total"] == 17
    # bound to the criteria column without an engine_param: each phrase is counted as the registry matches it on
    # that field, quoted as upstream quotes it (AREA[EligibilityCriteria]"..."), so this count is witnessed too
    seen = len(stub.requests)
    plan, res = await call(gw, "clinicaltrials", "count_clinical_trials",
                           {"eligibility_text": ["glioblastoma", "MGMT"], "country": None},
                           lambda a: {"total_count": 17})
    counts = [q for p, q in stub.requests[seen:] if p == "/ctgov/studies" and q.get("countTotal") == "true"]
    assert len(counts) == 1, stub.requests[seen:]
    advanced = counts[0]["filter.advanced"]
    assert 'AREA[EligibilityCriteria]"glioblastoma"' in advanced and 'AREA[EligibilityCriteria]"MGMT"' in advanced
    assert plan.witness["total"] == 17 and res.header.get("total") == 17
    with pytest.raises(GatewayError) as e:                      # and a count that differs is a contradiction
        await call(gw, "clinicaltrials", "count_clinical_trials",
                   {"eligibility_text": ["glioblastoma", "MGMT"], "country": None}, lambda a: {"total_count": 3})
    assert e.value.kind == ErrorKind.tool_defect


async def test_pubmed_query_is_counted_by_esearch(stub: Stub, tmp_path: Path) -> None:
    """``search_pubmed(query=...)``: the query goes to esearch's ``term`` in the count request (``rettype=count``)."""
    desc = _source("pubmed", stub)
    ov = load_yaml(SOURCES.parent / "overlays" / "pubmed.yaml", _variables(stub))
    gw = make_gateway(tmp_path, [desc], [ov], {})
    ctx = ServiceContext(gw.settings, catalog=gw.catalog, registry=REGISTRY)
    gw.service.witness_hook = lambda req, resp: WitnessResponse.model_validate(
        load_verbs()["_witness"](ctx, WitnessRequest.model_validate(req.model_dump()).model_dump(mode="json")))
    seen = len(stub.requests)
    rows = [{"pmid": str(30000000 + i), "title": f"t{i}"} for i in range(5)]
    plan, res = await call(gw, "pubmed", "search_pubmed", {"query": "pcsk9[tiab]", "max_results": 5},
                           lambda a: {"count": 23, "results": rows})
    counts = [q for p, q in stub.requests[seen:] if p == "/eutils/esearch.fcgi" and q.get("rettype") == "count"]
    assert len(counts) == 1 and counts[0]["term"] == "pcsk9[tiab]", stub.requests[seen:]
    assert plan.witness["total"] == 23 and res.header.get("total") == 23 and res.header.get("truncated")
    assert plan.args_sent["max_results"] == 5                  # a remote count never inflates the page


def test_classify_compares_remote_totals_only() -> None:
    src = {"schema": "vbt.datasource/1", "source": "reg", "title": "r", "kind": "remote", "release": {"from": "as_of"},
           "defaults": {"format": "none", "layout": "upstream_only"},
           "tables": {"t": {"kind": "records", "grain": "one", "key": {"columns": ["id"], "check": "none"},
                            "columns": {"id": {"role": "label"}}}}}
    ov = {"schema": "vbt.overlay/1", "server": "s", "tools": {"n": {
        "reads": {"reg.t": {"access": "upstream"}}, "args": {"q": {"binds": "reg.t.id"}},
        "result": {"kind": "count", "rows": None, "total": "$.n"}}}}
    for kind, want in (("remote", "contradiction"), ("local", "ok")):
        s = copy.deepcopy(src)
        s["kind"] = kind
        cat = Catalog({"reg": SourceDescriptor.model_validate(s)}, {"s": Overlay.model_validate(ov)}, [],
                      registry=REGISTRY)
        c = cat.contract("s", "n")
        assert classify(raw_of({"n": 0}), c, None, witness_total=4).outcome == want, kind
        assert classify(raw_of({"n": 4}), c, None, witness_total=4).outcome == "ok"


# --------------------------------------------------------------------------- live find, versions


def test_patient_level_find_pivots_clinical_data(ctx: ServiceContext, stub: Stub, monkeypatch) -> None:
    monkeypatch.setattr(witness_verbs, "_STUDY_RELEASES", {})
    out = load_verbs()["_live_find"](ctx, {"table": "cbioportal.patient_clinical",
                                          "predicate": to_json(Eq("studyId", "s1"))})
    rows = {r["patientId"]: r for r in out["rows"]}
    assert set(rows) == {"p1", "p2"} and rows["p1"]["OS_MONTHS"] == "12.5" and rows["p2"]["OS_STATUS"] == "0:LIVING"
    assert ("/cbio/studies/s1/clinical-data", {"clinicalDataType": "PATIENT"}) in stub.requests
    out = load_verbs()["_live_find"](ctx, {"table": "cbioportal.patient_clinical",
                                          "predicate": to_json(And((Eq("studyId", "s1"),
                                                                    Eq("OS_STATUS", "1:DECEASED"))))})
    assert [r["patientId"] for r in out["rows"]] == ["p1"]


def test_cbioportal_rows_name_their_study_import_as_the_release(ctx: ServiceContext, stub: Stub, monkeypatch) -> None:
    """LIVE-11: rows of a study (patients, samples, clinical data) depend on that study's import: its importDate
    (``release.per``) is the answer's as_of and the study's record version, not the HTTP fetch time, and a
    re-import is reported as source_updated."""
    monkeypatch.setattr(witness_verbs, "_STUDY_RELEASES", {})
    monkeypatch.setattr(stub, "import_date", "2026-06-05 15:19:54")
    out = load_verbs()["_live_find"](ctx, {"table": "cbioportal.patient_clinical",
                                          "predicate": to_json(Eq("studyId", "s1"))})
    assert out["as_of"] == "2026-06-05 15:19:54" and out["fetched_at"] != out["as_of"]
    assert out["record_versions"] == {"cbioportal.study": {"s1": "2026-06-05 15:19:54"}}
    assert ("/cbio/studies/s1", {"projection": "DETAILED"}) in stub.requests
    found = load_verbs()["find"](ctx, {"table": "cbioportal.sample", "where": {"studyId": "s1"}})
    assert found["_vbt"]["source"] == "cbioportal@2026-06-05 15:19:54"
    assert found["record_versions"] == {"cbioportal.study": {"s1": "2026-06-05 15:19:54"}}
    monkeypatch.setattr(witness_verbs, "_STUDY_RELEASES", {})
    monkeypatch.setattr(stub, "import_date", "2026-09-01 08:00:00")               # the study was re-imported
    again = load_verbs()["_live_find"](ctx, {"table": "cbioportal.sample", "predicate": to_json(Eq("studyId", "s1"))})
    assert again["as_of"] == "2026-09-01 08:00:00" and again["source_updated"] == ["cbioportal.study:s1"]


def test_public_find_and_lookup_serve_live_tables(ctx: ServiceContext, stub: Stub) -> None:
    """F20: mcp__data__find/lookup on a live table go to _live_find (never 'served upstream only'): the
    cBioPortal patient-level pivot counts each patient once, and the header names the source's as_of."""
    def split_header(out: dict) -> tuple[dict, dict]:
        return out, out["_vbt"]

    verbs = load_verbs()
    out = verbs["find"](ctx, {"table": "cbioportal.patient_clinical", "where": {"studyId": "s1"}})
    body, hdr = split_header(out)
    assert sorted(r["patientId"] for r in body["rows"]) == ["p1", "p2"] and hdr["status"] == "ok"
    assert hdr["served_by"] == "derived" and hdr["tables"] == ["cbioportal.patient_clinical"]
    one = verbs["find"](ctx, {"table": "cbioportal.patient_clinical",
                              "where": {"studyId": "s1", "OS_STATUS": "1:DECEASED"}})
    assert [r["patientId"] for r in split_header(one)[0]["rows"]] == ["p1"]
    studies = verbs["find"](ctx, {"table": "clinicaltrials_gov.studies", "limit": 1})
    assert split_header(studies)[0]["rows"]
    rec = verbs["lookup"](ctx, {"table": "cbioportal.patient_clinical", "key": {"studyId": "s1", "patientId": "p2"}})
    assert [r["patientId"] for r in split_header(rec)[0]["rows"]] == ["p2"]
    refused = verbs["find"](ctx, {"table": "cbioportal.patient_clinical", "where": {"studyId": "s1"}, "rank_by": "x"})
    assert refused["kind"] == "unsupported_combination"


def test_serve_answers_find_on_live_tables(ctx: ServiceContext, stub: Stub) -> None:
    """``_serve`` lookup/find on a live table goes through ``_live_find`` (served derived, with the source's
    release), orders complete reads only, and refuses what a paged source cannot answer."""
    from vbt.datalayer.ipc import RankKeyModel, ServeRequest

    serve = load_verbs()["_serve"]
    req = ServeRequest(table="cbioportal.patient_clinical", verb="find", predicate=to_json(Eq("studyId", "s1")),
                       order=[RankKeyModel(column="OS_MONTHS", direction="desc")], limit=1)
    out = serve(ctx, req.model_dump(mode="json"))
    assert [r["patientId"] for r in out["rows"]] == ["p2"] and out["truncated"] and out["total"] == 2
    assert out["served_by"] == "derived" and out["key_columns"] == ["studyId", "patientId"]
    one = serve(ctx, ServeRequest(table="cbioportal.patient_clinical", verb="lookup",
                                  predicate=to_json(And((Eq("studyId", "s1"), Eq("patientId", "p1"))))
                                  ).model_dump(mode="json"))
    assert [r["OS_STATUS"] for r in one["rows"]] == ["1:DECEASED"] and not one["truncated"]
    trials = serve(ctx, ServeRequest(table="clinicaltrials_gov.studies", verb="find", limit=5).model_dump(mode="json"))
    assert trials["as_of"] == "2026-10-01T09:30:00" and trials["rows"]
    # an ordered cut of a truncated read (one page of two within a one-page budget) is refused, never guessed
    d = _source("clinicaltrials", stub)
    d["budget"]["max_pages"] = 1
    one_page = ServiceContext(ctx.settings, catalog=Catalog({d["source"]: SourceDescriptor.model_validate(d)}, {}, [],
                                                            registry=REGISTRY), registry=REGISTRY)
    nct = RankKeyModel(column="protocolSection.identificationModule.nctId", direction="asc")
    cut = serve(one_page, ServeRequest(table="clinicaltrials_gov.studies", verb="find", limit=1,
                                       order=[nct]).model_dump(mode="json"))
    assert cut["rows"] == [] and cut["truncated"] and cut["reason"].startswith("too_large:"), cut
    with pytest.raises(Exception, match="live table"):
        serve(ctx, ServeRequest(table="cbioportal.patient_clinical", verb="find", group_by=["studyId"],
                                predicate=to_json(Eq("studyId", "s1"))).model_dump(mode="json"))


async def test_derived_binding_reads_a_live_table(stub: Stub, tmp_path: Path) -> None:
    """A ``serve: derived`` binding over a live table (patient-level clinical data, one row per patient) is answered
    by the data child's ``_serve`` through ``_live_find``; the result names the source's release."""
    from vbt.datalayer.ipc import ServeResponse

    desc = _source("cbioportal", stub)
    ov = {"schema": "vbt.overlay/1", "server": "cbio_derived", "sources": ["cbioportal"], "tools": {"patients": {
        "reads": {"cbioportal.patient_clinical": {"access": "remote"}},
        "args": {"study_id": {"binds": "cbioportal.patient_clinical.studyId", "existence": "off"}},
        "result": {"rows": "$.rows", "row_key": ["studyId", "patientId"]},
        "serve": "derived", "derived": {"verb": "find", "table": "cbioportal.patient_clinical"}}}}
    gw = make_gateway(tmp_path, [desc], [ov], {})
    ctx = ServiceContext(gw.settings, catalog=gw.catalog, registry=REGISTRY)
    gw.service._serve = lambda req: ServeResponse.model_validate(
        load_verbs()["_serve"](ctx, req.model_dump(mode="json")))
    plan, res = await call(gw, "cbio_derived", "patients", {"study_id": "s1"})
    assert plan.route == "derived" and res.header["served_by"] == "derived", res.header
    assert sorted(r["patientId"] for r in res.obj["rows"]) == ["p1", "p2"] and res.header["total"] == 2
    assert res.header["source"].startswith("cbioportal@"), res.header      # the page's time: no release endpoint


def test_record_version_change_is_flagged_source_updated(ctx: ServiceContext, stub: Stub) -> None:
    stub.study_version = "2024-05-01"
    find = load_verbs()["_live_find"]
    first = find(ctx, {"table": "clinicaltrials_gov.studies"})
    assert first["source_updated"] == [] and first["rows"]
    stub.study_version = "2026-09-30"
    second = find(ctx, {"table": "clinicaltrials_gov.studies"})
    assert second["source_updated"] == ["NCT01234567"]
    store = RecordVersions(Path(ctx.settings.cache_dir) / "clinicaltrials_gov" / "record_versions.json")
    assert store.observe("clinicaltrials_gov.studies", "NCT01234567", "2026-09-30") == "same"


def test_remote_vocab_ttl_and_drift() -> None:
    now = [0.0]
    vocab = RemoteVocab(ttl_s=60, clock=lambda: now[0])
    values = iter([["acc", "brca"], ["acc", "brca", "luad"]])
    assert vocab.get("cbio_study", lambda: next(values)) == ("acc", "brca")
    assert vocab.get("cbio_study", lambda: pytest.fail("refetched inside the TTL")) == ("acc", "brca")
    now[0] = 61.0
    assert vocab.get("cbio_study", lambda: next(values)) == ("acc", "brca", "luad")
    assert vocab.drifted("cbio_study") == {"added": ["luad"], "removed": []}


def test_release_from_payload() -> None:
    assert live.release_from_payload({"result": "$.census_version"}, {"census_version": "2025-01-30"}) == "2025-01-30"
    assert live.release_from_payload({"result": "$.census_version"}, {}) is None


# --------------------------------------------------------------------------- quoting


@pytest.mark.parametrize("lit", F9_LITERALS)
def test_soma_filter_quoting_round_trips(lit: str) -> None:
    pred = And((Eq("disease", lit), In("tissue", (lit, "lung")), Cmp("assay", "!=", lit),
                Not(In("sex", (lit,)))))
    text, residual = SomaFormat().compile(pred)
    assert residual is None and parse_soma(text) == pred
    assert SomaFormat().compile(Not(Eq("assay", lit)))[0] == SomaFormat().compile(Cmp("assay", "!=", lit))[0]


def test_soma_residuals_and_injection() -> None:
    fmt = SomaFormat()
    text, residual = fmt.compile(And((Eq("tissue", "lung"), IsNull("disease"), Eq("bad name') or (1", "x"))))
    assert text == "tissue == 'lung'" and residual == And((IsNull("disease"), Eq("bad name') or (1", "x")))
    text, residual = fmt.compile(Or((Eq("a", 1), TextMatch("b", "x", "word"))))
    assert text is None and residual is not None             # an Or with an inexpressible branch stays whole
    text, _ = fmt.compile(Range("n", 1, 5, hi_inclusive=False))
    assert text == "(n >= 1 and n < 5)"


@pytest.mark.parametrize("lit", F9_LITERALS)
def test_essie_and_eutils_literals_stay_literals(lit: str) -> None:
    q = essie_quote(lit)
    assert q.startswith('"') and q.endswith('"') and '"' not in q[1:-1].replace('\\"', "")
    params, residual = RestJsonFormat().compile(Eq("status", lit), {"remote_names": {"status": "AREA[OverallStatus]"}})
    assert residual is None and params["filter.advanced"] == f"AREA[OverallStatus]{q}"
    e = eutils_quote(lit)
    assert e.count('"') == 2


def test_rest_json_compile_partial_and_ranges() -> None:
    fmt = RestJsonFormat()
    params, residual = fmt.compile(
        And((In("s", ("A", "B")), Cmp("d", ">=", "2020-01-01"), Not(Eq("x", 1)), TextMatch("@query.cond", "lung"))),
        {"filters": {"s": {"param": "filter.overallStatus"}}, "remote_names": {"d": "AREA[StartDate]"}})
    assert params == {"filter.overallStatus": "A,B", "query.cond": "lung",
                      "filter.advanced": "AREA[StartDate]RANGE[2020-01-01,MAX]"}
    assert residual == Not(Eq("x", 1))
    assert essie_quote("OR") == '"OR"' and essie_quote("PHASE2") == "PHASE2"
    # an Or over one list column (the phase overlap of count_clinical_trials) is one Essie fragment; an Or
    # across columns stays a residual (a count request cannot express it)
    phases = Or((Contains("phases[]", "PHASE2"), Contains("phases[]", "PHASE3")))
    params, residual = fmt.compile(phases, {"remote_names": {"phases": "AREA[Phase]"}})
    assert params == {"filter.advanced": "AREA[Phase](PHASE2 OR PHASE3)"} and residual is None
    mixed = Or((Contains("phases[]", "PHASE2"), Eq("s", "A")))
    assert fmt.compile(mixed, {"remote_names": {"phases": "AREA[Phase]", "s": "AREA[OverallStatus]"}})[1] == mixed


# --------------------------------------------------------------------------- Census


class FakeFrame:
    def __init__(self, rows: list[dict[str, Any]], log: list[Any]) -> None:
        self.rows, self.log = rows, log

    def read(self, value_filter: str | None = None, column_names: list[str] | None = None) -> Any:
        self.log.append(value_filter)
        rows = [r for r in self.rows if value_filter is None or eval_filter(value_filter, r)]
        rows = [{k: v for k, v in r.items() if not column_names or k in column_names} for r in rows]
        return types.SimpleNamespace(concat=lambda: _Table(rows))


class _Table(list):
    def to_pylist(self) -> list[dict[str, Any]]:
        return list(self)


def eval_filter(text: str, row: dict[str, Any]) -> bool:
    from vbt.datalayer.predicate import evaluate

    return evaluate(parse_soma(text), row) is True


def fake_census(rows: list[dict[str, Any]], builds: list[str]) -> Any:
    log: list[Any] = []
    calls = iter(builds)
    frame = FakeFrame(rows, log)
    census = {"census_data": {"homo_sapiens": types.SimpleNamespace(obs=frame)}}
    mod = types.SimpleNamespace(get_census_version_description=lambda v: {"release_build": next(calls), "alias": v},
                                log=log, opened=[])

    class Census(dict):
        def close(self) -> None:
            pass

    def open_soma(census_version: str | None = None) -> Any:
        mod.opened.append(census_version)
        return Census(census)

    mod.open_soma = open_soma
    return mod


CELLS = ([{"soma_joinid": i, "dataset_id": "d1", "donor_id": "D1", "tissue": "lung", "is_primary_data": True}
          for i in range(0, 6)]
         + [{"soma_joinid": i, "dataset_id": "d1", "donor_id": "D2", "tissue": "lung", "is_primary_data": True}
            for i in range(6, 8)]
         + [{"soma_joinid": i, "dataset_id": "d2", "donor_id": "D1", "tissue": "lung", "is_primary_data": True}
            for i in range(8, 20)]
         + [{"soma_joinid": 99, "dataset_id": "d2", "donor_id": "D1", "tissue": "lung", "is_primary_data": False}])


def test_census_release_is_resolved_and_drift_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = fake_census(CELLS, ["2025-01-30", "2025-01-30", "2025-07-01"])
    monkeypatch.setattr(soma_layout, "_RESOLVED", {})
    record = tmp_path / "census_versions.json"
    first = soma_layout.resolve_version("stable", module=mod, record=record)
    assert first["resolved"] == "2025-01-30" and first["release_confidence"] == "inferred" and "drift" not in first
    assert soma_layout.resolve_version("stable", module=mod, record=record).get("drift") is None
    moved = soma_layout.resolve_version("stable", module=mod, record=record)
    assert moved["drift"] == {"before": "2025-01-30", "now": "2025-07-01"}
    assert soma_layout.resolve_version("2024-07-01", module=mod)["release_confidence"] == "exact"
    broken = types.SimpleNamespace(get_census_version_description=lambda v: (_ for _ in ()).throw(OSError("offline")))
    assert soma_layout.resolve_version("latest", module=broken)["release_confidence"] == "unknown"
    # a new process remembers the release through the record
    monkeypatch.setattr(soma_layout, "_RESOLVED", {})
    again = soma_layout.resolve_version("stable", module=fake_census(CELLS, ["2025-09-01"]), record=record)
    assert again["drift"] == {"before": "2025-07-01", "now": "2025-09-01"}


def test_census_count_first_and_donor_balanced_sample(ctx: ServiceContext, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = fake_census(CELLS, ["2025-01-30"] * 5)
    monkeypatch.setattr(soma_layout, "_RESOLVED", {})
    monkeypatch.setitem(sys.modules, "cellxgene_census", mod)
    verb = load_verbs()["_census_count"]
    out = verb(ctx, {"value_filter": "tissue == 'lung' and is_primary_data == True", "n_genes": 1000,
                     "cap_bytes": 10 * (200 + 1000 * 4)})
    assert out["n_cells"] == 20 and out["admissible"] is False and out["release"]["resolved"] == "2025-01-30"
    assert mod.log[-1] == "tissue == 'lung' and is_primary_data == True" and mod.opened[-1] == "2025-01-30"
    out = verb(ctx, {"predicate": to_json(And((Eq("dataset_id", "d1"), Eq("is_primary_data", True)))),
                     "n_genes": 10, "cap_bytes": 10 ** 9, "max_cells": 5})
    assert out["n_cells"] == 8 and out["n_cells_pulled"] == 5 and out["admissible"] and out["truncated_by_max_cells"]
    out = verb(ctx, {"value_filter": "is_primary_data == True", "sample": {"max_cells": 8, "seed": 1}})
    sample = out["sample"]
    assert sample["donor_key"] == ["dataset_id", "donor_id"] and sample["n_sampled"] == 8
    assert sample["per_dataset"] == {"d1": 4, "d2": 4}
    # donor D1 of d1 and donor D1 of d2 are different donors
    assert sample["per_donor"] == {"d1/D1": 2, "d1/D2": 2, "d2/D1": 4}
    assert 99 not in sample["soma_joinids"]


def test_donor_balanced_gives_unused_shares_back() -> None:
    rows = [{"soma_joinid": i, "dataset_id": "a", "donor_id": "small" if i < 1 else "big"} for i in range(10)]
    out = donor_balanced(rows, 6, seed=0)
    assert out["per_donor"] == {"a/small": 1, "a/big": 5}
    assert donor_balanced(rows, 6, seed=0) == out                   # seeded: reproducible


# --------------------------------------------------------------------------- PubMed server


def test_pubmed_server_reconciles_and_surfaces_query_problems(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastmcp")
    pytest.importorskip("httpx")
    from vbt.mcp_servers import pubmed_server as pm
    from stubs.eutils_stub import efetch_xml

    monkeypatch.delenv(pm.MAXDATE_ENV, raising=False)
    sent: list[dict[str, Any]] = []

    class Resp:
        def __init__(self, payload: Any = None, text: str = "") -> None:
            self.payload, self.text = payload, text

        def json(self) -> Any:
            return self.payload

    def fake_get(endpoint: str, params: dict[str, Any]) -> Resp:
        sent.append({"endpoint": endpoint, **params})
        if endpoint == "efetch.fcgi":
            ids = [i for i in params["id"].split(",") if i not in ("30595370",)]       # 30595370: no record
            return Resp(text=efetch_xml(ids))
        if endpoint == "esearch.fcgi":
            return Resp({"esearchresult": {"count": "0", "idlist": [],
                                           "querytranslation": "pcsk9[All Fields]",
                                           "errorlist": {"phrasesnotfound": ["zzzqqq"], "fieldsnotfound": []},
                                           "warninglist": {"outputmessages": ["No items found."]}}})
        return Resp({"result": {}})

    monkeypatch.setattr(pm, "_get", fake_get)
    ids = ["1234", "PMC1234", "30595370"] + [str(100000 + i) for i in range(57)]
    out = pm.fetch(ids)
    assert sent[0]["id"].split(",") == ids[:50]
    assert out["not_fetched"] == ids[50:] and "beyond the 50-ID cap" in out["summary"]
    assert out["not_returned"] == ["PMC1234", "30595370"]
    assert out["invalid_pmids"] == ["PMC1234"]
    assert "unrequested" not in out                       # 1234 was requested too
    alone = pm.fetch(["PMC1234"])
    assert alone["unrequested"] == ["1234"] and alone["articles"][0]["pmid"] == "1234"
    res = pm.search("pcsk9 zzzqqq")
    assert res["query_translation"] == "pcsk9[All Fields]"
    assert res["errors"] == ["phrasesnotfound: zzzqqq"] and res["warnings"] == ["outputmessages: No items found."]
    assert "ignored part of the query" in res["summary"]


def test_pubmed_overlay_reads_the_new_fields(tmp_path: Path) -> None:
    ov = Overlay.model_validate(load_yaml(OVERLAYS / "pubmed.yaml"))
    desc = SourceDescriptor.model_validate(load_yaml(SOURCES / "pubmed.yaml"))
    cat = Catalog({"pubmed": desc}, {"pubmed": ov}, [], registry=REGISTRY)
    c = cat.contract("pubmed", "search_pubmed")
    out = classify(raw_of({"count": 3, "results": [], "errors": ["phrasesnotfound: zzz"]}), c, None)
    assert out.outcome == "partial"
    c = cat.contract("pubmed", "fetch_abstracts")
    out = classify(raw_of({"articles": [{"pmid": "1234"}], "unrequested": ["1234"]}), c, None)
    assert out.outcome == "source_error"


def test_live_find_returns_the_record_versions(ctx: ServiceContext, stub: Stub) -> None:
    stub.study_version = "2024-05-01"
    out = load_verbs()["_live_find"](ctx, {"table": "clinicaltrials_gov.studies"})
    assert out["record_versions"] == {"clinicaltrials_gov.studies": {"NCT01234567": "2024-05-01"}}


def _census_gateway(stub: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, limit_mb: int) -> Any:
    from vbt.datalayer.ipc import CensusCountResponse, ReleaseResponse

    mod = fake_census(CELLS, ["2025-01-30"] * 20)
    monkeypatch.setattr(soma_layout, "_RESOLVED", {})
    monkeypatch.setitem(sys.modules, "cellxgene_census", mod)
    ov = load_yaml(SOURCES.parent / "overlays" / "single_cell.yaml", _variables(stub))
    # gene symbols resolve through Open Targets (not loaded here): bind them as plain values for this test
    ov["tools"]["get_anndata"]["args"]["gene_symbols"] = {"role": "projection"}
    gw = make_gateway(tmp_path, [_source("census", stub)], [ov], {},
                      data={"witness": {"enabled": False}, "memory": {"default_server_mb": limit_mb}})
    ctx = ServiceContext(gw.settings, catalog=gw.catalog, registry=REGISTRY)
    gw.service._census_count = lambda req: CensusCountResponse.model_validate(   # type: ignore[attr-defined]
        load_verbs()["_census_count"](ctx, req.model_dump(exclude_none=True)))
    gw.service._release = lambda req: ReleaseResponse.model_validate(            # type: ignore[attr-defined]
        load_verbs()["_release"](ctx, req.model_dump(exclude_none=True)))
    return gw


async def test_census_pulls_are_admitted_count_first(stub: Stub, tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """F20: the shipped single_cell overlay counts the cells a value_filter selects before upstream fetches
    anything: over the server's limit the pull is too_large (upstream never called); within it the call
    goes on with the count and the resolved release disclosed."""
    calls: list[Any] = []

    def upstream(tool: str, args: dict[str, Any]) -> Any:
        if tool == "list_metadata_values":                # the value_filter's vocabulary check
            col = args["column_name"]
            return {"value_counts": [{"value": v} for v in sorted({str(c[col]) for c in CELLS if col in c})]}
        calls.append((tool, args))
        out = tmp_path / "ok" / "out" / "x.h5ad"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return {"success": True, "output_path": str(out), "n_cells": 20, "n_genes": 2}

    big = ["G%d" % i for i in range(20000)]
    gw = _census_gateway(stub, tmp_path / "small", monkeypatch, limit_mb=1)
    gw.bridge.upstream = upstream
    with pytest.raises(GatewayError) as e:
        await call(gw, "single_cell", "get_anndata",
                   {"value_filter": "tissue == 'lung'", "gene_symbols": big, "output_path": "x.h5ad"}, upstream)
    assert e.value.kind == ErrorKind.too_large, e.value.message
    assert e.value.payload["n_cells"] == 20 and not calls
    assert e.value.payload["release"]["resolved"] == "2025-01-30"
    gw = _census_gateway(stub, tmp_path / "ok", monkeypatch, limit_mb=4000)
    gw.bridge.upstream = upstream
    plan = await gw.prepare("single_cell", "get_anndata",
                            {"value_filter": "tissue == 'lung'", "gene_symbols": ["G1", "G2"],
                             "output_path": "x.h5ad"}, None)
    st = plan._vbt_state  # type: ignore[attr-defined]
    assert st.count_first["n_cells"] == 20 and st.count_first["admissible"] is True
    assert any("count-first: the filter selects 20 cells" in n for n in st.notes)


def test_genes_found_are_recomputed_from_feature_name(ctx: ServiceContext, tmp_path: Path) -> None:
    """F20: _census_count with genes_file reads var.feature_name of the written h5ad."""
    pytest.importorskip("anndata")
    import anndata as ad
    import numpy as np
    import pandas as pd

    var = pd.DataFrame({"feature_name": ["CD276", "OSMR"]}, index=["0", "1"])     # positional var_names
    path = tmp_path / "x.h5ad"
    ad.AnnData(X=np.zeros((2, 2), dtype="float32"), var=var).write_h5ad(path)
    out = load_verbs()["_census_count"](ctx, {"genes_file": str(path), "genes": ["CD276", "TP53"]})
    assert out["genes_found"] == ["CD276"] and out["genes_not_found"] == ["TP53"]
