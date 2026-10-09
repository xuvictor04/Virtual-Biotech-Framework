"""Regressions for the live-source findings of the review (ClinicalTrials.gov, PubMed, cBioPortal, the Census);
offline: the shipped descriptors and overlays against fake transports, recorded replays and a stub Census.

* LIVE-2: native find/lookup on a live table honour the evidence ceiling (request, count, rows, header, provenance).
* LIVE-3: the PubMed witness counts under the literature ceiling the server applies (VBT_LITERATURE_MAXDATE).
* LIVE-5: strict bounds are sent as the next whole number or day (Essie RANGE and mindate/maxdate are inclusive).
* LIVE-6: an engine-matched text column takes ``search`` only; equality is refused (never a fuzzy total).
* LIVE-7: a SOMA row read names its columns (every column read reserves a buffer: std::bad_alloc under the limit).
* LIVE-8: ``is_primary_data == True`` (default_filter) is applied to native Census finds and disclosed.
* LIVE-9: ``columns`` keep dotted paths and the key; an unknown column is invalid_argument.
* LIVE-10: keys a live find names that the source does not hold are ``not_found_items``.
* LIVE-12: count_cells is witnessed from the parsed SOMA filter; single_cell results record the Census release.
* RR-4: a failed release (``/version``) request keeps the rows of a CT.gov find.
"""

from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from test_dl_gateway_flow import REGISTRY, raw_of
from test_dl_real_live import NCT, SOURCES, Replay, _ctx, _run, fixture
from vbt.datalayer.catalog import Catalog
from vbt.datalayer.descriptor.load import load_yaml
from vbt.datalayer.descriptor.models import SourceDescriptor
from vbt.datalayer.descriptor.overlay import Overlay
from vbt.datalayer.errors import ErrorKind, GatewayError
from vbt.datalayer.plugins.formats.rest_json import RestJsonFormat, inclusive_bounds
from vbt.datalayer.plugins.layouts import live_api as live
from vbt.datalayer.plugins.layouts import soma as soma_layout
from vbt.datalayer.predicate import And, Cmp, Eq, Range, TextMatch, evaluate, to_json
from vbt.datalayer.service import ServiceContext
from vbt.datalayer.settings import DataSettings

OVERLAYS = SOURCES.parent / "overlays"
CT = "clinicaltrials_gov.studies"
COUNT = "protocolSection.designModule.enrollmentInfo.count"
TITLE = "protocolSection.identificationModule.briefTitle"
STATUS = "protocolSection.statusModule.overallStatus"
FIRST = "AREA[StudyFirstPostDate]RANGE[MIN,2017-12-31]"
LAST = "AREA[LastUpdatePostDate]RANGE[MIN,2017-12-31]"


def study(nct: str, first: str, updated: str, *, title: str = "A trial", count: int = 100) -> dict[str, Any]:
    return {"protocolSection": {
        "identificationModule": {"nctId": nct, "briefTitle": title},
        "statusModule": {"overallStatus": "RECRUITING", "studyFirstPostDateStruct": {"date": first},
                         "lastUpdatePostDateStruct": {"date": updated}},
        "designModule": {"enrollmentInfo": {"count": count, "type": "ACTUAL"}}},
        "hasResults": False}


class CtGov:
    """ClinicalTrials.gov v2 as a transport: pages of ``studies`` (``filter.ids`` honoured, nothing else), the
    count and the version; every request is recorded."""

    def __init__(self, studies: list[dict[str, Any]], *, count: int = 4321, version_status: int = 200) -> None:
        self.studies, self.count, self.version_status = studies, count, version_status
        self.sent: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, params: Any = None, *, timeout: float = 30.0, headers: Any = None
                 ) -> tuple[int, dict[str, str], bytes]:
        sent = {str(k): str(v) for k, v in (params or {}).items() if v is not None}
        self.sent.append((url, sent))
        date = {"date": "Thu, 08 Oct 2026 02:45:10 GMT"}
        if url.endswith("/version"):
            if self.version_status != 200:
                return self.version_status, date, b'{"message": "internal error"}'
            return 200, date, json.dumps({"apiVersion": "2.0.5", "dataTimestamp": "2026-10-07T09:00:06"}).encode()
        if sent.get("countTotal") == "true":
            return 200, date, json.dumps({"totalCount": self.count, "studies": []}).encode()
        ids = sent.get("filter.ids")
        rows = [s for s in self.studies
                if not ids or s["protocolSection"]["identificationModule"]["nctId"] in ids.split(",")]
        size = int(sent.get("pageSize", 100))
        body: dict[str, Any] = {"studies": rows[:size]}
        if len(rows) > size:
            body["nextPageToken"] = "page2"
        return 200, date, json.dumps(body).encode()

    def pages(self) -> list[dict[str, str]]:
        return [s for u, s in self.sent if u.endswith("/studies") and s.get("countTotal") != "true"]

    def counts(self) -> list[dict[str, str]]:
        return [s for u, s in self.sent if s.get("countTotal") == "true"]


@pytest.fixture
def ctgov(monkeypatch: pytest.MonkeyPatch) -> Callable[..., CtGov]:
    monkeypatch.setattr(live, "_wait_turn", lambda base, rpm: None)
    monkeypatch.setattr(live, "_RELEASES", {})

    def install(studies: list[dict[str, Any]], **kw: Any) -> CtGov:
        t = CtGov(studies, **kw)
        monkeypatch.setattr(live.LiveApiLayout, "transport", staticmethod(t))
        return t
    return install


OLD = study("NCT00000001", "2016-01-05", "2016-03-01")
OLD_UPDATED = study("NCT00000002", "2016-02-01", "2020-05-01")        # first posted before, updated after
NEW = study("NCT04368728", "2020-04-30", "2026-03-25")
MORE = [study(f"NCT0000001{i}", "2015-06-01", "2015-07-01") for i in range(3)]


# --------------------------------------------------------------------------- LIVE-2


def test_live_find_under_the_ceiling_bounds_the_request_and_the_count(tmp_path, ctgov):
    """LIVE-2: under a 2017-12-31 ceiling a recruiting find returned trials first posted in 2025 and the uncapped
    total (64,639 against 2,323). The ceiling is a conjunct of the page request and of the count, on the first
    posting and (rows: withhold) the last update, so the page holds rows that can be returned and the total counts
    them; a row the source still returns past the ceiling is withheld, and the header says so."""
    t = ctgov([OLD, OLD_UPDATED, *MORE])
    out = _run(_ctx(tmp_path, ceiling="2017-12-31"), "find", {"table": CT, "where": {STATUS: "RECRUITING"}, "limit": 2})
    page, = t.pages()
    assert FIRST in page["filter.advanced"] and LAST in page["filter.advanced"]
    assert page["filter.overallStatus"] == "RECRUITING"
    count, = t.counts()
    assert FIRST in count["filter.advanced"] and LAST in count["filter.advanced"]
    h = out["_vbt"]
    ids = [r["protocolSection"]["identificationModule"]["nctId"] for r in out["rows"]]
    assert ids == ["NCT00000001"] and h["withheld"] == {"leakage": 1}      # the fake ignores the bounds it was sent
    assert h["leakage"] == {"ceiling": "2017-12-31", "withheld": 1, "risk": False}
    assert h["total"] == 4321 and any("evidence ceiling 2017-12-31" in n for n in h["notes"])


def test_live_lookup_under_the_ceiling_withholds_the_record_it_names(tmp_path, ctgov):
    """LIVE-2: lookup of NCT04368728 (first posted 2020-04-30) answered the record under a 2017 ceiling. A named key is
    read as named and withheld visibly: empty with withheld 1, never not_found."""
    t = ctgov([NEW])
    out = _run(_ctx(tmp_path, ceiling="2017-12-31"), "lookup", {"table": CT, "key": {"nctId": "NCT04368728"}})
    page, = t.pages()
    assert page["filter.ids"] == "NCT04368728" and FIRST not in page.get("filter.advanced", "")
    h = out["_vbt"]
    assert out["rows"] == [] and h["status"] == "empty" and h["withheld"] == {"leakage": 1}
    assert h.get("not_found_items") is None


def _gateway(ctx: ServiceContext, tmp_path: Path, upstream: dict[str, Callable[[str, dict[str, Any]], Any]] | None = None
             ) -> Any:
    from test_dl_native_tools import InProcessBridge
    from vbt.datalayer.gateway import DataGateway

    class Bridge(InProcessBridge):
        async def call_raw(self, server: str, tool: str, args: dict[str, Any]) -> Any:
            if server == "data" and "request" in args:
                return await super().call_raw(server, tool, args)       # the gateway's own verb requests
            self.calls.append((server, tool))
            if server == "data":                                          # a native tool: the agent's arguments
                return json.dumps(self.verbs[tool](self.ctx, args), default=str)
            return json.dumps((upstream or {})[server](tool, args), default=str)

    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    gw = DataGateway(ctx.settings, ctx.catalog, ctx.registry, run={"mcp_output_dir": str(out)})
    gw.bind_bridge(Bridge(ctx))
    return gw


def _ctx_with(tmp_path: Path, overlays: list[str], *, data: dict[str, Any] | None = None,
              sources: tuple[str, ...] = ("clinicaltrials", "cbioportal", "pubmed", "census")) -> ServiceContext:
    descs = {}
    for name in sources:
        d = load_yaml(SOURCES / f"{name}.yaml")
        descs[d["source"]] = SourceDescriptor.model_validate(d)
    ovs = {}
    for name in overlays:
        o = Overlay.model_validate(yaml.safe_load((OVERLAYS / f"{name}.yaml").read_text()))
        ovs[o.server] = o
    settings = DataSettings.from_dict({"cache_dir": str(tmp_path / "cache"), **(data or {})}, project_root=tmp_path)
    return ServiceContext(settings, catalog=Catalog(descs, ovs, [], registry=REGISTRY), registry=REGISTRY)


async def _call(gw: Any, server: str, tool: str, args: dict[str, Any]) -> Any:
    """prepare -> the bridge (upstream or the native verb) -> finish, like ``MCPBridge.call``."""
    plan = await gw.prepare(server, tool, args, None)
    raw = None
    if plan.route == "upstream":
        raw = raw_of(await gw.bridge.call_raw(server, tool, dict(plan.args_sent)))
    return await gw.finish(plan, raw)


async def test_native_find_under_the_ceiling_records_the_leakage(tmp_path, ctgov):
    """LIVE-2 through the gateway: data.find's provenance had leakage null while the upstream tool withheld the
    same trial in the same session."""
    ctgov([OLD, OLD_UPDATED])
    ctx = _ctx_with(tmp_path, ["data"], data={"leakage": {"ceiling": "2017-12-31"}})
    res = await _call(_gateway(ctx, tmp_path), "data", "find", {"table": CT, "where": {STATUS: "RECRUITING"}})
    assert res.header["withheld"] == {"leakage": 1} and res.header["returned"] == 1
    assert res.provenance.leakage == {"ceiling": "2017-12-31", "withheld": 1, "risk": False}
    assert "NCT00000002" not in res.text
    assert any("evidence ceiling 2017-12-31" in n for n in res.header["notes"])     # the child's disclosure


# --------------------------------------------------------------------------- LIVE-3


def test_pubmed_witness_counts_under_the_literature_ceiling(tmp_path, monkeypatch):
    """LIVE-3: the server bounds every search by VBT_LITERATURE_MAXDATE; the witness counted the same term without it
    (1,212 against 304 by 2017/12/31), so every search under a ceiling was a W1 tool_defect. The recorded count
    request (maxdate 2017/12/31) is the one the witness now sends for the bare engine text."""
    monkeypatch.setattr(live, "_wait_turn", lambda base, rpm: None)
    rec = fixture("replay/pubmed_term_count_ceiling.json")
    r = Replay(rec["exchanges"])
    monkeypatch.setattr(live.LiveApiLayout, "transport", staticmethod(r))
    monkeypatch.setenv("VBT_LITERATURE_MAXDATE", "2017/12/31")
    out = _run(_ctx(tmp_path), "_witness", {"table": "pubmed.records",
                                            "predicate": to_json(TextMatch("@term", "PCSK9 AND evolocumab"))})
    assert out["total"] == 304 and r.left == [] and "evidence ceiling 2017-12-31" in out["reason"]


async def test_search_pubmed_under_the_literature_ceiling_is_not_a_tool_defect(tmp_path, monkeypatch):
    """LIVE-3 through the gateway (the no-web profile sets 2025/01/31): upstream counts 304 by the ceiling, so does
    the witness; the data ceiling (data.leakage.ceiling) does not bound PubMed, whose server applies its own."""
    monkeypatch.setattr(live, "_wait_turn", lambda base, rpm: None)
    r = Replay(fixture("replay/pubmed_term_count_ceiling.json")["exchanges"])
    monkeypatch.setattr(live.LiveApiLayout, "transport", staticmethod(r))
    monkeypatch.setenv("VBT_LITERATURE_MAXDATE", "2017/12/31")
    ctx = _ctx_with(tmp_path, ["pubmed"], data={"leakage": {"ceiling": "2010-01-01"}})

    def pubmed(tool: str, args: dict[str, Any]) -> Any:
        assert tool == "search_pubmed"
        return {"query": args["query"], "count": 304, "returned": 2,
                "results": [{"pmid": "29224444", "title": "t1", "pubdate": "2017 Dec"},
                            {"pmid": "28304224", "title": "t2", "pubdate": "2017 May"}]}

    res = await _call(_gateway(ctx, tmp_path, {"pubmed": pubmed}), "pubmed", "search_pubmed",
                      {"query": "PCSK9 AND evolocumab", "max_results": 2})
    assert res.header["status"] == "partial" and res.header["total"] == 304
    # the record names the server's own (literature) ceiling, not data.leakage.ceiling (LIVE3-09)
    assert r.left == [] and res.provenance.leakage == {"ceiling": "2017-12-31", "withheld": 0, "risk": False}
    assert ("witness_count", True, "total=304 (remote count request)") in [
        (c.name, c.ok, c.detail) for c in res.provenance.checks]


def test_only_data_ceiling_sources_are_the_gateways_to_bound() -> None:
    from vbt.datalayer.gateway.leakage import ceiling_for
    from vbt.datalayer.descriptor.models import LeakageSpec

    spec = LeakageSpec(ceiling_from="web.literature_max_date", available_at="pubdate", counts="inject_filter")
    assert str(ceiling_for(spec, None, {"VBT_LITERATURE_MAXDATE": "2025/01/31"})) == "2025-01-31"
    assert ceiling_for(spec, None, {}) is None
    from vbt.datalayer.descriptor.lint import lint_descriptor

    d = load_yaml(SOURCES / "pubmed.yaml")
    d["leakage"]["ceiling_from"] = "web.literature_maxdate"
    found = [f for f in lint_descriptor(SourceDescriptor.model_validate(d), registry=REGISTRY)
             if f.where.endswith("leakage.ceiling_from")]
    assert found and found[0].level == "error"


# --------------------------------------------------------------------------- LIVE-5


def test_strict_bounds_are_sent_as_the_next_value() -> None:
    """LIVE-5: count {gt: 100, lt: 101} was sent as RANGE[100,MAX] AND RANGE[MIN,101] and answered 18,953 trials
    (all enrolling 100 or 101); the answer is none."""
    fmt = RestJsonFormat()
    req = {"remote_names": {COUNT: "AREA[EnrollmentCount]", "d": "AREA[StartDate]"}, "kinds": {COUNT: "integer"}}
    params, rest = fmt.compile(And((Cmp(COUNT, ">", 100), Cmp(COUNT, "<", 101))), req)
    assert params == {"filter.advanced": "AREA[EnrollmentCount]RANGE[101,MAX] AND AREA[EnrollmentCount]RANGE[MIN,100]"}
    assert rest is None
    params, rest = fmt.compile(Range(COUNT, 100, 101, lo_inclusive=False, hi_inclusive=False), req)
    assert params["filter.advanced"] == "AREA[EnrollmentCount]RANGE[101,100]" and rest is None
    params, rest = fmt.compile(Cmp("d", ">", "2020-01-31"), req)
    assert params["filter.advanced"] == "AREA[StartDate]RANGE[2020-02-01,MAX]"
    # no next value (a fraction, a partial date, an undeclared number): the bound is a residual, never widened
    for p in (Cmp(COUNT, ">", 100.5), Cmp("d", "<", "2020-01"), Cmp("x", ">", 3)):
        params, rest = fmt.compile(p, {**req, "remote_names": {**req["remote_names"], "x": "AREA[X]"}})
        assert rest == p and not params, p
    assert inclusive_bounds(Cmp(COUNT, ">=", 100), "integer") == (100, None)
    pub = {"filters": {"pubdate": {"range": {"lo": "mindate", "hi": "maxdate", "lo_default": "1800/01/01",
                                             "hi_default": "3000/12/31", "format": "%Y/%m/%d"}}}}
    params, rest = fmt.compile(Cmp("pubdate", ">", "2020-01-01"), pub)
    assert params == {"mindate": "2020/01/02", "maxdate": "3000/12/31"} and rest is None


def test_a_strict_bound_on_the_live_table_is_exact(tmp_path, ctgov):
    t = ctgov([OLD])
    _run(_ctx(tmp_path), "find", {"table": CT, "where": {COUNT: {"gt": 100, "lt": 101}}, "limit": 2})
    assert t.pages()[0]["filter.advanced"] == ("AREA[EnrollmentCount]RANGE[101,MAX] AND "
                                               "AREA[EnrollmentCount]RANGE[MIN,100]")


# --------------------------------------------------------------------------- LIVE-6


def test_equality_on_an_engine_matched_text_column_is_refused(tmp_path, ctgov):
    """LIVE-6: briefTitle == 'Cancer' was sent as AREA[BriefTitle]Cancer (a word and synonym search: 65,307 trials,
    titles without the word among them) and its total reported as exact."""
    t = ctgov([study("NCT00391092", "2006-10-20", "2016-01-01", title="A Study of Avastin in Breast Cancer")])
    for where in ({TITLE: "Cancer"}, {TITLE: ["Cancer"]}, {TITLE: {"eq": "Cancer"}}, {TITLE: {"contains": "Cancer"}}):
        bad = _run(_ctx(tmp_path), "find", {"table": CT, "where": where})
        assert bad["kind"] == "invalid_argument" and "search" in bad["message"], where
    assert t.sent == []
    out = _run(_ctx(tmp_path), "find", {"table": CT, "where": {TITLE: {"search": "Cancer"}}, "limit": 5})
    assert t.pages()[0]["filter.advanced"] == "AREA[BriefTitle]Cancer"
    assert any("search engine" in n for n in out["_vbt"]["notes"])
    # the compiler never sends an equality on such a column (any other caller gets a residual)
    params, rest = RestJsonFormat().compile(Eq(TITLE, "Cancer"), {"remote_names": {TITLE: "AREA[BriefTitle]"},
                                                                  "kinds": {TITLE: "text"}})
    assert params == {} and rest == Eq(TITLE, "Cancer")
    status = _run(_ctx(tmp_path), "find", {"table": CT, "where": {STATUS: "RECRUITING", "hasResults": False}})
    assert status["_vbt"]["status"] in ("ok", "partial")       # categories and flags compare as before


# --------------------------------------------------------------------------- LIVE-9


def test_columns_keep_dotted_paths_and_the_key(tmp_path, ctgov):
    """LIVE-9: columns [protocolSection.identificationModule.nctId] gave rows [{}, {}] cited as 'top 2 of 1779';
    an unknown column was accepted."""
    ctgov([OLD, NEW])
    out = _run(_ctx(tmp_path), "find", {"table": CT, "where": {STATUS: "RECRUITING"}, "columns": [COUNT]})
    first = out["rows"][0]
    assert first == {"protocolSection": {"identificationModule": {"nctId": "NCT00000001"},
                                         "statusModule": {"lastUpdatePostDateStruct": {"date": "2016-03-01"}},
                                         "designModule": {"enrollmentInfo": {"count": 100}}}}
    assert out["record_versions"] == {CT: {"NCT00000001": "2016-03-01", "NCT04368728": "2026-03-25"}}
    bad = _run(_ctx(tmp_path), "find", {"table": CT, "where": {STATUS: "RECRUITING"}, "columns": ["no_such_column"]})
    assert bad["kind"] == "invalid_argument" and "no_such_column" in bad["message"]
    inside = _run(_ctx(tmp_path), "find", {"table": CT, "columns": ["protocolSection.descriptionModule.briefSummary"]})
    assert inside["_vbt"]["status"] in ("ok", "partial")       # a path inside a payload column is the source's


def test_project_row_follows_paths_through_lists() -> None:
    row = {"a": {"b": [{"c": 1, "d": 2}, {"c": 3, "d": 4}], "e": 5}, "f": 6}
    assert live.project_row(row, ["a.b[].c", "f"]) == {"a": {"b": [{"c": 1}, {"c": 3}]}, "f": 6}
    assert live.project_row(row, ["a", "a.e"]) == {"a": row["a"]}
    assert live.project_row(row, ["x.y"]) == {}


# --------------------------------------------------------------------------- LIVE-10


def test_keys_the_source_does_not_hold_are_not_found_items(tmp_path, ctgov):
    """LIVE-10: nctId in [NCT04368728, NCT99999999] was ok with nothing about NCT99999999."""
    ctgov([NEW])
    out = _run(_ctx(tmp_path), "find", {"table": CT, "where": {NCT: ["NCT04368728", "NCT99999999"]}})
    h = out["_vbt"]
    assert h["status"] == "partial" and h["not_found_items"] == ["NCT99999999"]
    alone = _run(_ctx(tmp_path), "find", {"table": CT, "where": {NCT: "NCT99999999"}})
    assert alone["_vbt"]["status"] == "empty" and alone["_vbt"]["not_found_items"] == ["NCT99999999"]
    looked = _run(_ctx(tmp_path), "lookup", {"table": CT, "key": {"nctId": "NCT99999999"}})
    assert looked["kind"] == "not_found"
    # a key the ceiling withheld is accounted for, not missing
    held = _run(_ctx(tmp_path, ceiling="2017-12-31"), "find",
                {"table": CT, "where": {NCT: ["NCT04368728", "NCT99999999"]}})
    assert held["_vbt"]["not_found_items"] == ["NCT99999999"] and held["_vbt"]["withheld"] == {"leakage": 1}


async def test_native_not_found_items_make_the_answer_partial(tmp_path, ctgov):
    ctgov([NEW, OLD])
    ctx = _ctx_with(tmp_path, ["data"])
    res = await _call(_gateway(ctx, tmp_path), "data", "find",
                      {"table": CT, "where": {NCT: ["NCT04368728", "NCT00000001", "NCT99999999"]}})
    assert res.header["status"] == "partial" and res.header["not_found_items"] == ["NCT99999999"]


# --------------------------------------------------------------------------- RR-4


def test_a_failed_release_request_keeps_the_rows(tmp_path, monkeypatch):
    """RR-4: /version answering 500 turned the recorded 2-trial find into source_error and dropped its rows."""
    monkeypatch.setattr(live, "_wait_turn", lambda base, rpm: None)
    monkeypatch.setattr(live, "_RELEASES", {})
    rec = fixture("replay/ctgov_find_limit.json")
    replay = Replay([e for e in rec["exchanges"] if not e["url"].endswith("/version")])
    asked: list[str] = []

    def transport(url: str, params: Any = None, **kw: Any) -> tuple[int, dict[str, str], bytes]:
        if url.endswith("/version"):
            asked.append(url)
            return 500, {}, b"internal error"
        return replay(url, params, **kw)

    monkeypatch.setattr(live.LiveApiLayout, "transport", staticmethod(transport))
    out = _run(_ctx(tmp_path), "find", {"table": CT, "where": {STATUS: "SUSPENDED"}, "limit": 2})
    assert out.get("status") != "tool_error", out
    page = next(e for e in rec["exchanges"] if e["url"].endswith("/studies"))
    assert len(out["rows"]) == 2 and asked and replay.left == []
    assert out["_vbt"]["source"] == "clinicaltrials_gov@2026-10-08T02:45:10Z" == \
        f"clinicaltrials_gov@{live.RestJsonFormat().decode_page(page['body'], {}, page['headers']).as_of}"


# --------------------------------------------------------------------------- LIVE-7, LIVE-8, LIVE-12: the Census


CELLS = [
    {"soma_joinid": 1, "dataset_id": "d1", "donor_id": "D1", "cell_type": "B cell", "tissue": "adrenal gland",
     "tissue_general": "adrenal gland", "is_primary_data": True, "assay": "10x", "sex": "female"},
    {"soma_joinid": 2, "dataset_id": "d1", "donor_id": "D1", "cell_type": "B cell", "tissue": "adrenal gland",
     "tissue_general": "adrenal gland", "is_primary_data": False, "assay": "10x", "sex": "female"},
    {"soma_joinid": 3, "dataset_id": "d2", "donor_id": "D7", "cell_type": "T cell", "tissue": "lung",
     "tissue_general": "lung", "is_primary_data": True, "assay": "10x", "sex": "male"},
]


class Frame:
    """A Census obs frame: every read is logged with its column names; 28 columns like the real one."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows, self.reads = rows, []
        self.all = list(rows[0]) + [f"extra_{i}" for i in range(28 - len(rows[0]))]

    def read(self, value_filter: str | None = None, column_names: list[str] | None = None) -> Any:
        from vbt.datalayer.gateway.soma_filter import parse

        self.reads.append({"value_filter": value_filter, "column_names": column_names})
        rows = [r for r in self.rows if value_filter is None or evaluate(parse(value_filter), r) is True]
        cols = column_names or self.all
        out = [{c: r.get(c) for c in cols} for r in rows]

        class T(list):
            num_rows = len(out)

            def to_pylist(self) -> list[dict[str, Any]]:
                return list(self)
        return types.SimpleNamespace(concat=lambda: T(out))


def census(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]] = CELLS, release: str = "2025-11-08") -> Frame:
    frame = Frame(rows)

    class Census(dict):
        def close(self) -> None:
            pass

    mod = types.ModuleType("cellxgene_census")
    mod.open_soma = lambda census_version=None, tiledb_config=None: Census(  # type: ignore[attr-defined]
        {"census_data": {"homo_sapiens": {"obs": frame}}})
    mod.get_census_version_description = lambda v: {"release_build": release, "alias": v}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cellxgene_census", mod)
    monkeypatch.setattr(soma_layout, "_RESOLVED", {})
    monkeypatch.setattr(soma_layout, "_RELEASES", {})
    return frame


OBS = "cellxgene_census.obs"
GROUP = {"dataset_id": "d1", "cell_type": "B cell", "tissue_general": "adrenal gland"}


def test_a_census_find_reads_named_columns(tmp_path, monkeypatch):
    """LIVE-7: without columns every obs column was read (28 x 128 MiB of buffers: std::bad_alloc under the data
    child's 3000 MB limit, and so was every lookup). The declared columns are read, plus the key and the filter's."""
    frame = census(monkeypatch)
    out = _run(_ctx(tmp_path), "find", {"table": OBS, "where": GROUP, "limit": 3})
    assert out.get("status") != "tool_error", out
    rows_read = [r for r in frame.reads if r["column_names"] != ["soma_joinid"]]
    declared = list(load_yaml(SOURCES / "census.yaml")["tables"]["obs"]["columns"])
    assert rows_read and set(rows_read[-1]["column_names"]) == set(declared) and len(declared) < len(frame.all)
    named = _run(_ctx(tmp_path), "find", {"table": OBS, "where": GROUP, "columns": ["cell_type"]})
    last = [r for r in frame.reads if r["column_names"] != ["soma_joinid"]][-1]["column_names"]
    assert set(last) == {"cell_type", "soma_joinid", "dataset_id", "tissue_general", "is_primary_data"}
    assert set(named["rows"][0]) == {"cell_type", "soma_joinid"}
    looked = _run(_ctx(tmp_path), "lookup", {"table": OBS, "key": {"soma_joinid": 999999999999}})
    assert looked["kind"] == "not_found"
    assert set([r for r in frame.reads if r["column_names"] != ["soma_joinid"]][-1]["column_names"]) <= set(declared)


def test_a_census_find_excludes_duplicate_cells_and_says_so(tmp_path, monkeypatch):
    """LIVE-8: the group above held 158 cells, all is_primary_data false: native find said 158, count_cells 0."""
    frame = census(monkeypatch)
    out = _run(_ctx(tmp_path), "find", {"table": OBS, "where": GROUP})
    h = out["_vbt"]
    assert [r["soma_joinid"] for r in out["rows"]] == [1] and h["total"] == 1
    assert all("is_primary_data == True" in (r["value_filter"] or "") for r in frame.reads)
    assert any("is_primary_data == true added" in n for n in h["notes"])
    both = _run(_ctx(tmp_path), "find", {"table": OBS, "where": {**GROUP, "is_primary_data": [True, False]}})
    assert sorted(r["soma_joinid"] for r in both["rows"]) == [1, 2]
    assert not any("added" in n for n in both["_vbt"].get("notes", []))


def _single_cell(frame_rows: list[dict[str, Any]]) -> Callable[[str, dict[str, Any]], Any]:
    from vbt.datalayer.gateway.soma_filter import parse

    def server(tool: str, args: dict[str, Any]) -> Any:
        if tool == "list_metadata_values":
            col = args["column_name"]
            vals = sorted({str(r[col]) for r in frame_rows})
            return {"column": col, "value_counts": [{"value": v, "count": 1} for v in vals]}
        if tool == "count_cells":
            pred = parse(args["value_filter"])
            return {"n_cells": sum(evaluate(pred, r) is True for r in frame_rows), "value_filter": args["value_filter"]}
        if tool == "get_census_info":
            return {"census_version": "stable", "organisms": ["homo_sapiens"]}
        raise AssertionError(tool)
    return server


async def test_count_cells_is_witnessed_and_records_the_release(tmp_path, monkeypatch):
    """LIVE-12: count_cells had checks [] (value_filter counted as opaque engine text) and source.release null; the
    gateway parsed the filter already, so the remote witness counts the same predicate."""
    frame = census(monkeypatch)
    ctx = _ctx_with(tmp_path, ["single_cell"])
    gw = _gateway(ctx, tmp_path, {"single_cell": _single_cell(CELLS)})
    res = await _call(gw, "single_cell", "count_cells", {"value_filter": "tissue_general == 'adrenal gland'"})
    checks = {c.name: (c.ok, c.detail) for c in res.provenance.checks}
    assert checks["witness_count"] == (True, "total=1 (remote count request)")
    assert any("is_primary_data == True" in (r["value_filter"] or "") for r in frame.reads)
    assert res.provenance.source.release == "2025-11-08" and res.header["source"] == "cellxgene_census@2025-11-08"
    info = await _call(gw, "single_cell", "get_census_info", {})
    assert info.provenance.source.release == "2025-11-08"
    # the body names the dated release too (result.release_alias), not the alias the server opened
    assert info.obj["census_version"] == "2025-11-08"
    assert any("opened 'stable'" in n for n in info.header.get("notes", [])), info.header


async def test_a_count_that_disagrees_with_the_parsed_filter_is_caught(tmp_path, monkeypatch):
    census(monkeypatch)
    ctx = _ctx_with(tmp_path, ["single_cell"])
    rows = copy.deepcopy(CELLS) + [dict(CELLS[0], soma_joinid=9)]          # upstream counts a cell the store lacks
    gw = _gateway(ctx, tmp_path, {"single_cell": _single_cell(rows)})
    with pytest.raises(GatewayError) as e:
        await _call(gw, "single_cell", "count_cells", {"value_filter": "tissue_general == 'adrenal gland'"})
    assert e.value.kind == ErrorKind.tool_defect and "W1" in e.value.message


def test_soma_release_is_the_resolved_alias(monkeypatch):
    census(monkeypatch, release="2025-11-08")
    from vbt.datalayer.plugins.base import LayoutSpec

    layout = soma_layout.SomaLayout()
    spec = LayoutSpec(table=OBS, path=None, options={"uri": "census_data/homo_sapiens/obs", "census_version": "stable"})
    assert layout.release(spec, max_age_s=600) == "2025-11-08"
    census(monkeypatch, release="2026-01-01")
    monkeypatch.setattr(soma_layout, "_RELEASES", {"stable": (soma_layout.time.monotonic(), "2025-11-08")})
    assert layout.release(spec, max_age_s=600) == "2025-11-08"          # reused within max_age_s
    assert layout.release(spec) == "2026-01-01"


def test_boolean_in_lists_compile_as_equalities() -> None:
    """LIVE-8 override on the real Census: ``is_primary_data in [True, False]`` failed in tiledbsoma
    ("PyQueryCondition.create_uint8() not found"); a boolean list is sent as one == per value."""
    from vbt.datalayer.gateway.soma_filter import parse
    from vbt.datalayer.plugins.formats.soma import SomaFormat
    from vbt.datalayer.predicate import In, Not

    fmt = SomaFormat()
    text, rest = fmt.compile(In("is_primary_data", (True, False)))
    assert text == "(is_primary_data == True or is_primary_data == False)" and rest is None
    assert all(evaluate(parse(text), {"is_primary_data": v}) is True for v in (True, False))
    assert fmt.compile(In("is_primary_data", (False,)))[0] == "is_primary_data == False"
    assert fmt.compile(Not(In("is_primary_data", (True,))))[0] == "is_primary_data != True"
    assert fmt.compile(In("tissue", ("lung", "liver")))[0] == "tissue in ['lung', 'liver']"


def test_soma_row_reads_use_small_buffers(monkeypatch) -> None:
    """LIVE-7: the 13 declared obs columns still failed with 128 MiB buffers under 3000 MB (std::bad_alloc) and took
    5.6 s at 1,246 MB with 16 MiB ones (real Census, 158 cells); counts keep the larger buffers."""
    from vbt.datalayer.plugins.base import LayoutSpec

    seen: list[dict[str, Any]] = []
    frame = census(monkeypatch)
    mod = sys.modules["cellxgene_census"]
    opener = mod.open_soma

    def open_soma(census_version=None, tiledb_config=None):  # noqa: ANN001, ANN202
        seen.append(dict(tiledb_config or {}))
        return opener(census_version=census_version, tiledb_config=tiledb_config)

    monkeypatch.setattr(mod, "open_soma", open_soma)
    layout = soma_layout.SomaLayout()
    spec = LayoutSpec(table=OBS, path=None, options={"uri": "census_data/homo_sapiens/obs"})
    layout.request(spec, predicate=Eq("dataset_id", "d1"), projection=["soma_joinid", "cell_type"], page_token=None,
                   budget=None)
    assert seen[-1]["soma.init_buffer_bytes"] == seen[-1]["py.init_buffer_bytes"] == soma_layout.ROW_BUFFER_BYTES
    layout.count(spec, predicate=Eq("dataset_id", "d1"))
    assert seen[-1] == soma_layout.DEFAULT_TILEDB_CONFIG and frame.reads[-1]["column_names"] == ["soma_joinid"]
