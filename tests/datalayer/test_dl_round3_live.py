"""Gateway contract requests for live and remote sources (round 3, S1): recorded responses replayed offline, the
live APIs on demand.

What is covered (the requests are numbered as in the real-data reports R2/R3):

* **CR3** the derived ``(dataset_id, donor_id)`` sample of ``single_cell.get_anndata_donor_balanced``: the data child
  draws upstream's own cell-type-stratified, donor-balanced sample (``census_count.stratified_sample``, the same
  generator and order as ``single_cell_mcp/tools.py``) with donors keyed by dataset, and the gateway asks the
  unmodified server for exactly those cells (``soma_joinid in [...]``, ``max_cells`` = the sample's size), completes
  ``obs_columns`` so the file shows its cells, restores the counted total and checks that the file holds the sample.
  The shipped overlay declares the facet (``count_first.sample``); tests that need another facet set it on the parsed
  binding. With ``VBT_DL_NETWORK=1`` the 200,000-cell path (a 2.1 MB ``soma_joinid`` filter) is drawn and counted
  again on the real Census.
* **CR4** ``clinicaltrials.get_clinical_data`` served derived from the live cBioPortal tables (``derived.compose``:
  the samples, their sample attributes, their patients' attributes; ``clinical_attributes`` from a live section):
  the rows equal upstream's, patient attributes go to the ``patients`` section, a sample the study does not hold is
  ``not_found``. The shipped binding is this derived route; the ``pass`` variant (upstream answering) is built here
  for the comparison.
* **CR7** the release a live call observed, in its provenance: the data release (CT.gov ``dataTimestamp``; one
  cBioPortal study's ``importDate``), the API and software versions (``apiVersion``; ``portalVersion``, ``dbVersion``;
  the PubMed build) and the per-record releases (the study), asked of the data child's ``_release`` verb.
* **CR8** the evidence ceiling's two totals (an upstream count bounds the first posting only; a native find also the
  last update) are both in the header, and a Census read cut to ``limit`` says it read every matching row.
* a ``soma_joinid`` filter never asks the server for a vocabulary of the join id (217 M values on the real Census),
  and "No cells found" with a count-first count of 0 is an empty answer, not a missing gene.

``real/live/round3/`` holds the exchanges each scenario made with the real APIs on 2026-10-08 (URL, parameters,
status, body; trimmed where noted). Offline, the scenarios run again on the shipped descriptors through the gateway
and the data child's verbs in this process, with a transport that answers from the recording and fails on any
request it did not record. With ``VBT_DL_NETWORK=1`` they run against the live APIs, and the Census tests read the
real Census (``cellxgene_census`` installed).
"""

from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path
from typing import Any, Callable

import pytest

from netgate import network_enabled
from test_dl_gateway_flow import REGISTRY, FakeBridge, raw_of
from test_dl_real_live import Replay
from vbt.datalayer.catalog import Catalog
from vbt.datalayer.descriptor.load import load_yaml
from vbt.datalayer.descriptor.models import SourceDescriptor
from vbt.datalayer.descriptor.overlay import Overlay, SampleSpec
from vbt.datalayer.errors import ErrorKind, GatewayError
from vbt.datalayer.gateway import DataGateway
from vbt.datalayer.gateway.service_client import ServiceClient, ServiceError
from vbt.datalayer.ipc import CensusCountRequest, parse_response
from vbt.datalayer.plugins.layouts import live_api as live
from vbt.datalayer.plugins.layouts import soma as soma_layout
from vbt.datalayer.resolve import IndexStore
from vbt.datalayer.service import ServiceContext
from vbt.datalayer.service.verbs import load_verbs
from vbt.datalayer.service.verbs import witness as witness_verbs
from vbt.datalayer.service.verbs.census_count import sample_filter, stratified_sample
from vbt.datalayer.settings import DataSettings

REPO = Path(__file__).resolve().parents[2]
SOURCES = REPO / "configs" / "data" / "sources"
OVERLAYS = REPO / "configs" / "data" / "overlays"
FIX = Path(__file__).resolve().parent / "real" / "live"
ROUND3 = FIX / "round3"
NETWORK = network_enabled()
needs_network = pytest.mark.skipif(not NETWORK, reason="set VBT_DL_NETWORK=1 to call the live APIs")

ACC = "acc_tcga_pan_can_atlas_2018"
SAMPLES = ["TCGA-OR-A5J1-01", "TCGA-OR-A5J2-01"]
PHANTOM = "TCGA-OR-A5ZZ-01"
ST = "protocolSection.statusModule.overallStatus"
CT = "clinicaltrials_gov.studies"
CEILING = "2017-12-31"

#: The derived binding of get_clinical_data (the shipped overlay's, pinned by test_shipped_bindings_are_the_derived_routes).
DERIVED_CLINICAL = {
    "verb": "find", "table": "cbioportal.sample", "columns": ["studyId", "sampleId", "patientId"],
    "envelope": {"success": True, "study_id": "{study_id}", "sample_count": None},
    "compose": [{"verb": "find", "table": "cbioportal.sample_clinical"},
                {"verb": "find", "table": "cbioportal.patient_clinical"}],
    "sections": {"clinical_attributes": {"path": "$.clinical_attributes", "table": "cbioportal.clinical_attribute",
                                         "verb": "find", "key_from_args": {"studyId": "study_id"},
                                         "value": "clinicalAttributeId"}},
}
#: The sample facet of get_anndata_donor_balanced (the shipped overlay's).
SAMPLE_FACET = {"grain": "donor", "stratify": "cell_type", "seed": 42, "columns_arg": "obs_columns",
                "total_path": "$.n_cells_total", "max_cells_default": 20000}


def fixture(name: str) -> dict[str, Any]:
    return json.loads((FIX / name).read_text())


# --------------------------------------------------------------------------- the gateway on in-process verbs


class InProcess(ServiceClient):
    """The data child's verbs run in this process on a real :class:`ServiceContext` (shipped descriptors), as the
    child answers them: a verb that raises is a refused request (``ServiceError``)."""

    def __init__(self, ctx: ServiceContext) -> None:
        super().__init__(None)
        self.ctx = ctx
        self.log: list[tuple[str, Any]] = []

    @property
    def available(self) -> bool:
        return True

    async def call(self, verb: str, request: Any) -> Any:
        self.log.append((verb, request))
        self.calls += 1
        try:
            out = load_verbs()[verb](self.ctx, request.model_dump(mode="json", by_alias=True, exclude_none=True))
        except Exception as exc:  # noqa: BLE001 - what the child's tool call reports
            raise ServiceError(f"data child failed on {verb}: Error calling tool: {exc}", verb=verb,
                               subkind="rejected") from exc
        return parse_response(verb, out)

    def verbs(self, verb: str) -> list[Any]:
        return [r for v, r in self.log if v == verb]


def _settings(tmp_path: Path, data: dict[str, Any] | None = None) -> DataSettings:
    return DataSettings.from_dict({"cache_dir": str(tmp_path / "cache"), **dict(data or {})}, project_root=tmp_path)


def _descriptors() -> dict[str, SourceDescriptor]:
    return {d.source: d for d in (SourceDescriptor.model_validate(load_yaml(SOURCES / f"{n}.yaml"))
                                  for n in ("clinicaltrials", "cbioportal", "pubmed", "census"))}


def _ctx(tmp_path: Path, data: dict[str, Any] | None = None) -> ServiceContext:
    settings = _settings(tmp_path, data)
    return ServiceContext(settings, catalog=Catalog(_descriptors(), {}, [], registry=REGISTRY), registry=REGISTRY)


def clinical_overlay(*, derived: bool) -> dict[str, Any]:
    """The shipped clinicaltrials overlay (get_clinical_data served derived); ``derived=False``: served upstream."""
    ov = load_yaml(OVERLAYS / "clinicaltrials.yaml")
    b = ov["tools"]["get_clinical_data"]
    if not derived:
        b["serve"] = "pass"
        b.pop("derived", None)
    return ov


def gateway(tmp_path: Path, overlays: list[dict[str, Any]], *, data: dict[str, Any] | None = None,
            upstream: Callable[[str, dict[str, Any]], Any] | None = None) -> DataGateway:
    """The gateway on the shipped descriptors and ``overlays``, its data child the verbs in this process."""
    settings = _settings(tmp_path, data)
    ovs = {o["server"]: Overlay.model_validate(copy.deepcopy(o)) for o in overlays}
    catalog = Catalog(_descriptors(), ovs, [], registry=REGISTRY)
    service = InProcess(ServiceContext(settings, catalog=catalog, registry=REGISTRY))
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    gw = DataGateway(settings, catalog, REGISTRY, service=service, index_store=IndexStore(settings.cache_dir),
                     run={"mcp_output_dir": str(out)})
    gw.bind_bridge(FakeBridge(upstream))
    return gw


async def call(gw: DataGateway, server: str, tool: str, args: dict[str, Any],
               upstream: Callable[[Any, dict[str, Any]], Any] | None = None) -> Any:
    """prepare -> (upstream, given the plan, when routed there) -> finish."""
    plan = await gw.prepare(server, tool, args, None)
    raw = raw_of(upstream(plan, dict(plan.args_sent))) if plan.route == "upstream" and upstream else None
    return await gw.finish(plan, raw)


def _fresh_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live, "_wait_turn", lambda base, rpm: None)
    monkeypatch.setattr(live, "_RELEASES", {})
    monkeypatch.setattr(witness_verbs, "_STUDY_RELEASES", {})


# --------------------------------------------------------------------------- scenarios (recorded, replayed, live)


async def s_clinical_derived(tmp_path: Path) -> dict[str, Any]:
    """CR4: get_clinical_data served derived from the live cBioPortal tables."""
    gw = gateway(tmp_path, [clinical_overlay(derived=True)])
    res = await call(gw, "clinicaltrials", "get_clinical_data", {"study_id": ACC, "sample_ids": list(SAMPLES)})
    return {"header": res.header, "obj": res.obj, "prov": res.provenance.to_dict()}


async def s_clinical_phantom(tmp_path: Path) -> dict[str, Any]:
    """CR4: a sample the study does not hold is not_found, before the attributes are read."""
    gw = gateway(tmp_path, [clinical_overlay(derived=True)])
    try:
        await call(gw, "clinicaltrials", "get_clinical_data", {"study_id": ACC, "sample_ids": [SAMPLES[0], PHANTOM]})
    except GatewayError as exc:
        return {"error": exc.envelope()}
    return {"error": None}


async def s_clinical_unknown_study(tmp_path: Path) -> dict[str, Any]:
    """CR4: a study the portal does not hold is not_found naming study_id (the source's 404 on the sample list)."""
    gw = gateway(tmp_path, [clinical_overlay(derived=True)])
    try:
        await call(gw, "clinicaltrials", "get_clinical_data", {"study_id": "no_such_study_xyz", "sample_ids": ["X-01"]})
    except GatewayError as exc:
        return {"error": exc.envelope()}
    return {"error": None}


async def s_study_details_release(tmp_path: Path) -> dict[str, Any]:
    """CR7: an upstream-served cBioPortal call names the study's importDate as its release and records the portal's
    versions (it named no release before: source "cbioportal")."""
    study = next(e["body"] for e in recorded("release_cbio")["exchanges"] if e["url"].endswith(f"/studies/{ACC}"))
    gw = gateway(tmp_path, [load_yaml(OVERLAYS / "clinicaltrials.yaml")])
    res = await call(gw, "clinicaltrials", "get_study_details", {"study_id": ACC}, lambda plan, a: dict(study))
    return {"header": res.header, "prov": res.provenance.to_dict()}


def s_release_ctgov(tmp_path: Path) -> dict[str, Any]:
    """CR7: what CT.gov says about the data a call reads (the version endpoint)."""
    return load_verbs()["_release"](_ctx(tmp_path), {"table": CT})


def s_release_pubmed(tmp_path: Path) -> dict[str, Any]:
    """CR7: the PubMed build (einfo)."""
    return load_verbs()["_release"](_ctx(tmp_path), {"table": "pubmed.records"})


def s_release_cbio(tmp_path: Path) -> dict[str, Any]:
    """CR7: cBioPortal's software versions and the importDate of the study the call names."""
    from vbt.datalayer.predicate import Eq, to_json

    return load_verbs()["_release"](_ctx(tmp_path), {"table": "cbioportal.sample",
                                                     "predicate": to_json(Eq("studyId", ACC))})


async def s_ceiling_count(tmp_path: Path) -> dict[str, Any]:
    """CR8: count_clinical_trials under a 2017-12-31 ceiling, upstream answering the registry's own count."""
    gw = gateway(tmp_path, [load_yaml(OVERLAYS / "clinicaltrials.yaml")], data={"leakage": {"ceiling": CEILING}})
    args = {"status": ["RECRUITING"], "country": None}
    res = await call(gw, "clinicaltrials", "count_clinical_trials", args,
                     lambda plan, a: {"total_count": plan.witness["total"], "query_params": a})
    return {"header": res.header, "prov": res.provenance.to_dict()}


def s_find_ceiling(tmp_path: Path) -> dict[str, Any]:
    """CR8: a native find of recruiting trials under the same ceiling."""
    return load_verbs()["find"](_ctx(tmp_path, {"leakage": {"ceiling": CEILING}}),
                                {"table": CT, "where": {ST: "RECRUITING"}, "limit": 2})


SCENARIOS: dict[str, Callable[[Path], Any]] = {
    "clinical_derived": s_clinical_derived, "clinical_phantom": s_clinical_phantom,
    "release_ctgov": s_release_ctgov, "release_pubmed": s_release_pubmed, "release_cbio": s_release_cbio,
    "ceiling_count": s_ceiling_count, "find_ceiling": s_find_ceiling,
    "clinical_unknown_study": s_clinical_unknown_study, "study_details_release": s_study_details_release,
}


async def run_scenario(name: str, tmp_path: Path) -> Any:
    out = SCENARIOS[name](tmp_path)
    return await out if hasattr(out, "__await__") else out


@pytest.fixture
def replay(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], Replay]:
    """``replay(name)`` installs the recorded exchanges of ``round3/<name>.json`` as the HTTP transport."""
    _fresh_caches(monkeypatch)

    def install(name: str) -> Replay:
        r = Replay(json.loads((ROUND3 / f"{name}.json").read_text())["exchanges"])
        monkeypatch.setattr(live.LiveApiLayout, "transport", staticmethod(r))
        return r
    return install


def recorded(name: str) -> dict[str, Any]:
    return json.loads((ROUND3 / f"{name}.json").read_text())


# --------------------------------------------------------------------------- CR4: get_clinical_data derived


async def test_get_clinical_data_derived_equals_upstream(tmp_path: Path, replay: Callable[[str], Replay]) -> None:
    """The derived rows hold upstream's values for every attribute the recorded upstream answer kept (sample and
    patient level), the patient attributes go to the ``patients`` section (one row per patient), and the result is
    served derived with the study's importDate as its release and the portal's versions in its provenance."""
    r = replay("clinical_derived")
    out = await run_scenario("clinical_derived", tmp_path)
    assert r.left == [], r.left                                     # every recorded request was made
    hdr, obj, prov = out["header"], out["obj"], out["prov"]
    assert hdr["served_by"] == "derived" and hdr["status"] == "ok" and hdr["returned"] == 2
    up = {row["sampleId"]: row for row in fixture("cbioportal/upstream_clinical_data_phantom.json")["result"]["data"]
          if row.get("patientId")}
    rows = {row["sampleId"]: row for row in obj["data"]}
    patients = {p["patientId"]: p for p in obj["patients"]}
    assert set(rows) == set(up) == set(SAMPLES) and len(patients) == 2
    levels = set(load_yaml(OVERLAYS / "clinicaltrials.yaml")["tools"]["get_clinical_data"]["result"]["levels"]["patient"])
    for sid, want in up.items():
        got = {**rows[sid], **patients[want["patientId"]]}
        for attr, value in want.items():
            assert got.get(attr) == value, (sid, attr, got.get(attr), value)
        assert not (set(rows[sid]) & levels)                        # patient values are not repeated on the rows
    assert obj["sample_count"] == 2 and obj["study_id"] == ACC and obj["success"] is True
    assert len(obj["clinical_attributes"]) == recorded("clinical_derived")["clinical_attribute_count"]
    assert hdr["grains"]["patient"] == {"returned": 2, "total": 2}
    study = recorded("clinical_derived")["study_import_date"]
    assert hdr["source"] == f"cbioportal@{study}"
    assert prov["source"]["versions"].keys() == {"portalVersion", "dbVersion"}
    assert prov["result"]["record_versions"]["cbioportal.study"] == {ACC: study}
    assert any(t.startswith("composed cbioportal.patient_clinical") for t in prov["result"]["transforms"])


async def test_get_clinical_data_derived_refuses_a_phantom_sample(tmp_path: Path,
                                                                  replay: Callable[[str], Replay]) -> None:
    """Upstream answers a sample the study does not hold with a row whose patientId is null; derived, the sample is
    absent from the study's sample list, so the call is not_found naming it before any attribute is read."""
    r = replay("clinical_phantom")
    out = await run_scenario("clinical_phantom", tmp_path)
    err = out["error"]
    assert err["kind"] == "not_found" and err["items"] == [PHANTOM] and err["argument"] == "sample_ids"
    assert not [u for u, _ in r.sent if "clinical-data" in u]      # refused before the attribute reads


async def test_get_clinical_data_derived_of_an_unknown_study_is_not_found(tmp_path: Path,
                                                                        replay: Callable[[str], Replay]) -> None:
    """The portal answers 404 for the study's sample list: not_found naming study_id (the caller's argument, not the
    column). Through MCPBridge this came back service_unavailable while the error rode in ServeResponse.error."""
    replay("clinical_unknown_study")
    err = (await run_scenario("clinical_unknown_study", tmp_path))["error"]
    assert err["kind"] == "not_found" and err["argument"] == "study_id" and err["value"] == "no_such_study_xyz"


async def test_upstream_served_cbioportal_call_names_its_release(tmp_path: Path,
                                                                replay: Callable[[str], Replay]) -> None:
    replay("study_details_release")
    out = await run_scenario("study_details_release", tmp_path)
    study = next(e["body"] for e in recorded("release_cbio")["exchanges"] if e["url"].endswith(f"/studies/{ACC}"))
    assert out["header"]["source"] == f"cbioportal@{study['importDate']}"
    assert out["prov"]["source"]["release"] == study["importDate"]
    assert set(out["prov"]["source"]["versions"]) == {"portalVersion", "dbVersion"}
    assert out["prov"]["result"]["record_versions"]["cbioportal.study"] == {ACC: study["importDate"]}


async def test_derived_compose_merges_on_each_sub_tables_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """compose merges each sub-read on that table's key, a later sub-read winning a shared name (upstream's
    data.update order), and a sub-read the source cut marks the answer partial."""
    from vbt.datalayer.ipc import ServeResponse

    gw = gateway(tmp_path, [clinical_overlay(derived=True)])
    tables = {
        "cbioportal.sample": [{"studyId": ACC, "sampleId": "S1", "patientId": "P1"},
                              {"studyId": ACC, "sampleId": "S2", "patientId": "P1"}],
        "cbioportal.sample_clinical": [{"studyId": ACC, "sampleId": "S1", "SAMPLE_TYPE": "Primary", "AGE": "1"},
                                       {"studyId": ACC, "sampleId": "S2", "SAMPLE_TYPE": "Metastasis"}],
        "cbioportal.patient_clinical": [{"studyId": ACC, "patientId": "P1", "AGE": "58", "OS_STATUS": "1:DECEASED"}],
        "cbioportal.clinical_attribute": [{"studyId": ACC, "clinicalAttributeId": "AGE"},
                                          {"studyId": ACC, "clinicalAttributeId": "SAMPLE_TYPE"}],
    }
    sent: list[Any] = []

    async def serve(req: Any) -> Any:
        sent.append(req)
        from vbt.datalayer.predicate import evaluate, from_json

        pred = from_json(req.predicate) if req.predicate else None
        rows = [r for r in tables[req.table] if pred is None or evaluate(pred, r) is True]
        return ServeResponse(rows=rows, total=len(rows), truncated=req.table == "cbioportal.patient_clinical",
                             key_columns=[], as_of="2026-06-04 18:48:28")

    monkeypatch.setattr(gw.service, "serve", serve)
    # the release provenance the gateway reads for the live source goes nowhere (RR-3: it reached cBioPortal) and
    # its failure costs the call nothing
    released: list[str] = []

    def refuse(url: str, params: Any = None, **_kw: Any) -> Any:
        released.append(url)
        raise live.RemoteError(f"no network here: {url}", url=url)

    monkeypatch.setattr(live.LiveApiLayout, "transport", staticmethod(refuse))
    monkeypatch.setattr(live, "_RELEASES", {})
    res = await call(gw, "clinicaltrials", "get_clinical_data", {"study_id": ACC, "sample_ids": ["S1", "S2"]})
    assert released and all("cbioportal" in u for u in released)
    rows = {r["sampleId"]: r for r in res.obj["data"]}
    assert rows["S1"]["SAMPLE_TYPE"] == "Primary" and rows["S2"]["SAMPLE_TYPE"] == "Metastasis"
    assert res.obj["patients"] == [{"studyId": ACC, "patientId": "P1", "AGE": "58", "OS_STATUS": "1:DECEASED"}]
    assert res.header["status"] == "partial"                        # the patient read was cut
    assert res.obj["clinical_attributes"] == ["AGE", "SAMPLE_TYPE"]
    # the patient read names the patients of the rows read, the sample attributes their samples
    by_table = {r.table: r.predicate for r in sent}
    assert "P1" in json.dumps(by_table["cbioportal.patient_clinical"])
    assert "S2" in json.dumps(by_table["cbioportal.sample_clinical"])


def test_live_serve_answers_an_unknown_study_as_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A derived read of a live table whose source answers 404 for the study is the typed not_found (the derived
    get_clinical_data of an unknown study), never a failed request read as an outage."""
    _fresh_caches(monkeypatch)
    monkeypatch.setattr(live.LiveApiLayout, "transport", staticmethod(Replay(
        fixture("replay/cbio_unknown_study.json")["exchanges"])))
    from vbt.datalayer.predicate import Eq, to_json

    out = load_verbs()["_serve"](_ctx(tmp_path), {"table": "cbioportal.patient_clinical", "verb": "find",
                                                  "predicate": to_json(Eq("studyId", "no_such_study_xyz"))})
    err = out["refusal"]                                   # ipc.SERVE_ERROR: MCPBridge reads a top-level error as a failure
    assert "error" not in out and err["kind"] == "not_found" and err["value"] == "no_such_study_xyz"
    from vbt.tools.mcp_bridge import tool_result_error

    assert tool_result_error(out) is None                  # through the bridge: a typed answer, not a failed call
    assert parse_response("_serve", out).error == err


# --------------------------------------------------------------------------- CR7: releases and versions


def test_ctgov_release_and_api_version(tmp_path: Path, replay: Callable[[str], Replay]) -> None:
    replay("release_ctgov")
    rel = s_release_ctgov(tmp_path)["release"]
    body = recorded("release_ctgov")["exchanges"][0]["body"]
    assert rel["resolved"] == body["dataTimestamp"] and rel["versions"] == {"apiVersion": body["apiVersion"]}


def test_pubmed_build_is_recorded(tmp_path: Path, replay: Callable[[str], Replay]) -> None:
    replay("release_pubmed")
    rel = s_release_pubmed(tmp_path)["release"]
    info = recorded("release_pubmed")["exchanges"][0]["body"]["einforesult"]["dbinfo"][0]
    assert rel == {"versions": {"dbbuild": info["dbbuild"], "lastupdate": info["lastupdate"]}}


def test_cbioportal_versions_and_the_studys_import_date(tmp_path: Path, replay: Callable[[str], Replay]) -> None:
    replay("release_cbio")
    rel = s_release_cbio(tmp_path)["release"]
    ex = recorded("release_cbio")["exchanges"]
    info = next(e["body"] for e in ex if e["url"].endswith("/info"))
    study = next(e["body"] for e in ex if e["url"].endswith(f"/studies/{ACC}"))
    assert rel["versions"] == {"portalVersion": info["portalVersion"], "dbVersion": info["dbVersion"]}
    assert rel["per"] == {"cbioportal.study": {ACC: study["importDate"]}}
    assert rel["resolved"] == study["importDate"] and rel["release_confidence"] == "per_record"


def test_the_witness_reads_no_release_a_source_does_not_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A release block with versions only (cBioPortal, PubMed) adds no request to the remote witness or a native
    find: only the gateway's release request reads it (the recorded scenarios of earlier rounds stay exact)."""
    _fresh_caches(monkeypatch)
    sent: list[str] = []

    def transport(url: str, params: Any = None, **kw: Any) -> tuple[int, dict[str, str], bytes]:
        sent.append(url)
        return 404, {}, b"{}"

    monkeypatch.setattr(live.LiveApiLayout, "transport", staticmethod(transport))
    ctx = _ctx(tmp_path)
    for ref in ("pubmed.records", "cbioportal.study"):
        t = ctx.table(ref)
        layout = ctx.plugin("layout", t.layout)
        from vbt.datalayer.service import layout_spec

        assert layout.release(layout_spec(t)) is None
    assert sent == []


# --------------------------------------------------------------------------- CR8: the ceiling's two totals


async def test_upstream_count_under_the_ceiling_reports_both_totals(tmp_path: Path,
                                                                    replay: Callable[[str], Replay]) -> None:
    """The upstream count bounds StudyFirstPostDate only; one more remote count with LastUpdatePostDate bounded too
    gives the rows a find may return under rows: withhold. Both are in the header, with a note."""
    r = replay("ceiling_count")
    out = await run_scenario("ceiling_count", tmp_path)
    counts = [s for u, s in r.sent if s.get("countTotal") == "true"]
    assert len(counts) == 2 and all("AREA[StudyFirstPostDate]RANGE[MIN,2017-12-31]" in s["filter.advanced"]
                                    for s in counts)
    assert "AREA[LastUpdatePostDate]RANGE[MIN,2017-12-31]" in counts[1]["filter.advanced"]
    tot = out["header"]["ceiling_totals"]
    rec = recorded("ceiling_count")
    assert tot == {"available": rec["available"], "available_and_unchanged": rec["available_and_unchanged"],
                   "ceiling": CEILING}
    assert tot["available"] > tot["available_and_unchanged"]
    # LIVE3-01: status selects on the record as it is today, which a first-posted bound does not bound: the total is
    # the records also unchanged since the ceiling (their status then is their status now), a lower bound
    h = out["header"]
    assert h["total"] == tot["available_and_unchanged"] and h["status"] == "partial"
    assert h["total_method"] == "ceiling_unchanged" and any("selects on status (overallStatus)" in n for n in h["notes"])
    assert any("also last changed" in n for n in out["prov"]["result"].get("notes", []) + out["header"]["notes"])
    assert out["prov"]["source"]["versions"] == {"apiVersion": "2.0.5"}


def test_native_find_under_the_ceiling_says_what_its_total_counts(tmp_path: Path,
                                                                 replay: Callable[[str], Replay]) -> None:
    replay("find_ceiling")
    out = s_find_ceiling(tmp_path)
    notes = out["_vbt"]["notes"]
    assert out["_vbt"]["total"] == recorded("ceiling_count")["available_and_unchanged"]
    assert any("last changed (lastUpdatePostDateStruct)" in n and "upstream count" in n for n in notes), notes


# --------------------------------------------------------------------------- CR3: the derived donor sample


def _obs(spec: list[tuple[str, str, str, int]], start: int = 0) -> list[dict[str, Any]]:
    """``(dataset_id, donor_id, cell_type, n)`` blocks of primary cells, numbered by soma_joinid."""
    out: list[dict[str, Any]] = []
    for dataset, donor, cell_type, n in spec:
        for _ in range(n):
            out.append({"soma_joinid": start + len(out), "dataset_id": dataset, "donor_id": donor,
                        "cell_type": cell_type, "tissue": "lung", "disease": "normal", "assay": "10x",
                        "sex": "female", "development_stage": "adult", "is_primary_data": True})
    return out


#: One dataset, no ties in any count (upstream's value_counts order is then unambiguous).
ONE_DATASET = _obs([("d1", "A", "T cell", 41), ("d1", "B", "T cell", 23), ("d1", "C", "T cell", 7),
                    ("d1", "A", "B cell", 19), ("d1", "B", "B cell", 5), ("d1", "C", "NK cell", 3)])


def test_stratified_sample_is_upstreams_own_draw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With one dataset the derived sample picks exactly the cells the unmodified upstream function picks (its
    code, run on the network-free Census stub): same allocation over cell types, same donor order, same generator."""
    from dl_upstream import upstream_missing

    if upstream_missing() is not None:
        pytest.skip(str(upstream_missing()))
    pytest.importorskip("anndata")
    import pandas as pd
    from test_dl_defect_detectors import GENES, census_call

    out, log = census_call(tmp_path, monkeypatch, {"obs": ONE_DATASET, "var": GENES}, "get_anndata_donor_balanced",
                           output_path=str(tmp_path / "b.h5ad"), value_filter="tissue == 'lung'", max_cells=30)
    upstream_ids = sorted(next(e for e in log if e["kind"] == "get_anndata")["obs_coords"])
    frame = pd.DataFrame(ONE_DATASET)[["soma_joinid", "dataset_id", "donor_id", "cell_type"]]
    mine = stratified_sample(frame, 30)
    assert mine["soma_joinids"] == upstream_ids and out["n_cells_sampled"] == len(upstream_ids)
    assert sum(mine["per_stratum"].values()) == len(upstream_ids) and set(mine["per_stratum"]) == {
        "T cell", "B cell", "NK cell"}


def test_stratified_sample_keeps_donors_of_two_datasets_apart() -> None:
    """Two datasets share the donor label D1: upstream balances them as one donor (SC-001); the derived sample keys
    donors by (dataset_id, donor_id) and samples both."""
    import pandas as pd

    rows = _obs([("d1", "D1", "T cell", 50), ("d2", "D1", "T cell", 50), ("d2", "D2", "T cell", 10)])
    out = stratified_sample(pd.DataFrame(rows), 30)
    assert set(out["per_donor"]) == {"d1/D1", "d2/D1", "d2/D2"} and out["n_donors"] == 3
    assert out["n_sampled"] == 30 and out["per_dataset"]["d1"] == out["per_donor"]["d1/D1"]
    assert stratified_sample(pd.DataFrame(rows), 30) == out                      # seeded: reproducible
    assert stratified_sample(pd.DataFrame(rows), 500)["n_sampled"] == len(rows)   # under max_cells: every cell


class _Frame:
    """A SOMA dataframe over rows: ``read(column_names, value_filter)`` as tiledbsoma answers it (batches with
    ``num_rows``; ``concat().to_pandas()`` with categorical columns)."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.filters: list[str | None] = []

    def read(self, column_names: list[str] | None = None, value_filter: str | None = None, **_: Any) -> Any:
        import pandas as pd

        from vbt.datalayer.gateway import soma_filter
        from vbt.datalayer.predicate import evaluate

        self.filters.append(value_filter)
        pred = soma_filter.parse(value_filter) if value_filter else None
        hit = [r for r in self.rows if pred is None or evaluate(pred, r) is True]
        df = pd.DataFrame(hit, columns=list(self.rows[0]))[list(column_names or self.rows[0])]
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype("category")
        return _Read(df)


class _Read:
    def __init__(self, df: Any) -> None:
        self.df = df

    def __iter__(self) -> Any:
        return iter([types.SimpleNamespace(num_rows=len(self.df))])

    def concat(self) -> Any:
        df = self.df
        return types.SimpleNamespace(num_rows=len(df), to_pandas=lambda: df.copy(),
                                     column=lambda c: types.SimpleNamespace(to_pylist=lambda: df[c].tolist()),
                                     to_pylist=lambda: df.to_dict("records"))


def fake_census(rows: list[dict[str, Any]]) -> Any:
    frame = _Frame(rows)
    census = types.SimpleNamespace(close=lambda: None,
                                   census_data={"homo_sapiens": types.SimpleNamespace(obs=frame)})
    return types.SimpleNamespace(open_soma=lambda census_version=None, tiledb_config=None: census, frame=frame,
                                 get_census_version_description=lambda v: {"release_build": "2025-11-08"})


def census_gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]], *,
                   facet: dict[str, Any] | None = SAMPLE_FACET) -> tuple[DataGateway, Any, dict[str, Any]]:
    """The shipped single_cell overlay over a fake Census (``facet``: the sample facet of get_anndata_donor_balanced,
    set on its parsed binding when it differs from the shipped one; None removes it); upstream is played by a
    function that writes the cells its filter selects into the h5ad it reports."""
    mod = fake_census(rows)
    monkeypatch.setitem(sys.modules, "cellxgene_census", mod)
    monkeypatch.setattr(soma_layout, "_RESOLVED", {})
    monkeypatch.setattr(soma_layout, "_RELEASES", {})
    seen: dict[str, Any] = {}

    def upstream(tool: str, args: dict[str, Any]) -> Any:
        if tool == "list_metadata_values":
            col = args["column_name"]
            seen.setdefault("vocab", []).append(col)
            return {"value_counts": [{"value": v, "count": 1} for v in sorted({str(r[col]) for r in rows})]}
        return {}

    gw = gateway(tmp_path, [load_yaml(OVERLAYS / "single_cell.yaml")], upstream=upstream)
    b = gw.catalog.overlays["single_cell"].tools["get_anndata_donor_balanced"]
    if facet != SAMPLE_FACET:
        b.count_first.sample = SampleSpec.model_validate(facet) if facet is not None else None
    return gw, mod, seen


def test_shipped_bindings_are_the_derived_routes() -> None:
    """The shipped overlays serve both routes this module checks: get_anndata_donor_balanced as the derived sample
    (no dataset_id has to be fixed) and get_clinical_data derived from the live cBioPortal tables."""
    sc = Overlay.model_validate(load_yaml(OVERLAYS / "single_cell.yaml")).tools["get_anndata_donor_balanced"]
    assert sc.count_first.sample.model_dump(exclude_none=True) == SAMPLE_FACET and not sc.requires_fixed
    assert sc.count_first.genes_args == ["gene_symbols", "ensembl_ids"]
    ct = Overlay.model_validate(load_yaml(OVERLAYS / "clinicaltrials.yaml")).tools["get_clinical_data"]
    assert ct.serve == "derived" and ct.derived.model_dump(exclude_none=True, exclude_defaults=True) == \
        Overlay.model_validate({"schema": "vbt.overlay/1", "server": "x", "tools": {"t": {
            "serve": "derived", "derived": DERIVED_CLINICAL}}}).tools["t"].derived.model_dump(
            exclude_none=True, exclude_defaults=True)


def _pull(tmp_path: Path, rows: list[dict[str, Any]], *, drop: int = 0) -> Callable[[Any, dict[str, Any]], Any]:
    """Upstream get_anndata_donor_balanced on the fake Census: every cell when its filter selects at most max_cells
    (the derived route), with ``drop`` cells missing from the file (a defective server)."""
    import anndata
    import numpy as np
    import pandas as pd

    from vbt.datalayer.gateway import soma_filter
    from vbt.datalayer.predicate import evaluate

    def run(plan: Any, args: dict[str, Any]) -> Any:
        pred = soma_filter.parse(args["value_filter"])
        hit = [r for r in rows if evaluate(pred, r) is True]
        assert len(hit) <= int(args["max_cells"])
        keep = hit[drop:]
        cols = list(args.get("obs_columns") or ["cell_type", "donor_id", "dataset_id"])
        obs = pd.DataFrame([{c: (r[c] if c == "soma_joinid" else str(r.get(c, ""))) for c in cols} for r in keep],
                           index=[str(i) for i in range(len(keep))])
        # as the real Census files: var holds a soma_joinid column too (the gene's), the cells are obs.soma_joinid
        var = pd.DataFrame({"soma_joinid": [7], "feature_id": ["ENSG00000198851"], "feature_name": ["CD3E"]},
                           index=["0"])
        path = tmp_path / "out" / args["output_path"]
        anndata.AnnData(X=np.zeros((len(keep), 1), dtype=np.float32), obs=obs, var=var).write_h5ad(path)
        return {"output_path": args["output_path"], "n_cells_total": len(hit), "n_cells_sampled": len(keep),
                "n_donors": len({r["donor_id"] for r in keep}), "donor_summary": {}, "n_genes": 1}
    return run


TWO_DATASETS = _obs([("d1", "D1", "T cell", 30), ("d1", "D2", "B cell", 12), ("d2", "D1", "T cell", 25),
                     ("d2", "D3", "NK cell", 9)])


async def test_donor_balanced_is_served_as_the_derived_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("anndata")
    gw, mod, seen = census_gateway(tmp_path, monkeypatch, TWO_DATASETS)
    args = {"value_filter": "tissue == 'lung'", "max_cells": 20, "ensembl_ids": ["ENSG00000198851"],
            "output_path": "s.h5ad"}
    plan = await gw.prepare("single_cell", "get_anndata_donor_balanced", dict(args), None)
    sent = plan.args_sent
    assert sent["value_filter"].startswith("soma_joinid in [")
    assert {"soma_joinid", "dataset_id", "donor_id", "cell_type"} <= set(sent["obs_columns"])
    ids = sorted(int(x) for x in sent["value_filter"][len("soma_joinid in ["):-1].split(","))
    assert ids == sorted(stratified_sample(_frame(TWO_DATASETS), 20)["soma_joinids"])
    # upstream keeps every cell a filter selects when they are at most max_cells: max_cells is the sample's size
    # (upstream's allocation can draw fewer than max_cells: 19 of 20 here)
    assert sent["max_cells"] == len(ids) <= 20
    res = await gw.finish(plan, raw_of(_pull(tmp_path, TWO_DATASETS)(plan, dict(sent))))
    hdr, obj = res.header, res.obj
    assert hdr["served_by"] == "derived" and obj["n_cells_total"] == len(TWO_DATASETS)
    drawn = obj["derived_sample"]
    assert drawn["donor_key"] == ["dataset_id", "donor_id"] and "d2/D1" in drawn["per_donor"]
    assert {c.name: c.ok for c in res.provenance.checks}["sample_cells"] is True
    assert any(n.startswith("served as the derived sample") for n in hdr["notes"] + res.provenance.result.transforms
               ) or res.provenance.result.transforms[0].startswith("derived sample")
    assert "soma_joinid" not in seen.get("vocab", [])                # the join id's "vocabulary" is never listed


async def test_a_file_without_the_sampled_cells_is_a_tool_defect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("anndata")
    gw, _mod, _seen = census_gateway(tmp_path, monkeypatch, TWO_DATASETS)
    plan = await gw.prepare("single_cell", "get_anndata_donor_balanced",
                            {"value_filter": "tissue == 'lung'", "max_cells": 20, "output_path": "s.h5ad"}, None)
    with pytest.raises(GatewayError) as e:
        await gw.finish(plan, raw_of(_pull(tmp_path, TWO_DATASETS, drop=1)(plan, dict(plan.args_sent))))
    assert e.value.kind == ErrorKind.tool_defect and "sampled cells" in e.value.message


async def test_a_filter_under_max_cells_is_passed_as_it_is(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gw, _mod, _seen = census_gateway(tmp_path, monkeypatch, TWO_DATASETS)
    plan = await gw.prepare("single_cell", "get_anndata_donor_balanced",
                            {"value_filter": "dataset_id == 'd2'", "max_cells": 100, "output_path": "s.h5ad"}, None)
    assert plan.args_sent["value_filter"] == "dataset_id == 'd2' and is_primary_data == True"
    assert plan.args_sent["max_cells"] == 100
    assert any("nothing is sampled" in n for n in _state_notes(plan))


async def test_no_sample_across_datasets_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a sample (more cells than a sample may read) upstream would balance donor labels across datasets:
    refused, unless the filter fixes one dataset_id (then upstream's own balancing is right and is served)."""
    gw, _mod, _seen = census_gateway(tmp_path, monkeypatch, TWO_DATASETS,
                                     facet={**SAMPLE_FACET, "max_read": 10})
    with pytest.raises(GatewayError) as e:
        await gw.prepare("single_cell", "get_anndata_donor_balanced",
                         {"value_filter": "tissue == 'lung'", "max_cells": 20, "output_path": "s.h5ad"}, None)
    assert e.value.kind == ErrorKind.unsupported_combination and "dataset_id" in e.value.message
    plan = await gw.prepare("single_cell", "get_anndata_donor_balanced",
                            {"value_filter": "dataset_id == 'd1'", "max_cells": 20, "output_path": "s.h5ad"}, None)
    assert plan.args_sent["value_filter"] == "dataset_id == 'd1' and is_primary_data == True"
    assert any("upstream's own donor balancing is served" in n for n in _state_notes(plan))


async def test_no_cells_found_with_a_zero_count_is_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Upstream's "No cells found for filter" named ensembl_ids as not found (a real call, 2026-10-08): the count-first
    count of the same filter is 0, so the answer is empty, with that count as its total."""
    gw, _mod, _seen = census_gateway(tmp_path, monkeypatch, TWO_DATASETS)
    args = {"value_filter": "dataset_id == 'd1' and cell_type == 'NK cell'", "max_cells": 20,
            "ensembl_ids": ["ENSG00000198851"], "output_path": "s.h5ad"}
    plan = await gw.prepare("single_cell", "get_anndata_donor_balanced", args, None)
    answer = {"output_path": "s.h5ad", "n_cells_total": 0, "n_cells_sampled": 0, "n_donors": 0, "n_cell_types": 0,
              "error": f"No cells found for filter: {plan.args_sent['value_filter']}"}
    res = await gw.finish(plan, raw_of(answer))
    assert res.header["status"] == "empty" and res.header["total"] == 0 and res.header["total_method"] == "count_first"


async def test_a_soma_joinid_filter_lists_no_vocabulary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gw, _mod, seen = census_gateway(tmp_path, monkeypatch, TWO_DATASETS, facet=None)
    plan = await gw.prepare("single_cell", "count_cells", {"value_filter": "soma_joinid in [1, 2, 3] and "
                                                                          "dataset_id == 'd1'"}, None)
    assert plan.args_sent["value_filter"].startswith("soma_joinid in [1, 2, 3]")
    assert seen.get("vocab") == ["dataset_id"]


def _frame(rows: list[dict[str, Any]]) -> Any:
    import pandas as pd

    df = pd.DataFrame(rows)[["soma_joinid", "dataset_id", "donor_id", "cell_type"]]
    for c in ("dataset_id", "donor_id", "cell_type"):
        df[c] = df[c].astype("category")
    return df


def _state_notes(plan: Any) -> list[str]:
    return list(getattr(plan, "_vbt_state").notes)


# --------------------------------------------------------------------------- CR8: the Census read cut to limit


def test_census_find_cut_to_limit_says_every_row_was_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Census read is one unpaged request: the whole match is read and cut to limit here. The note said "1 page(s)
    read within the source's budget: the rows are a prefix", which describes a paged source."""
    mod = fake_census(TWO_DATASETS)
    monkeypatch.setitem(sys.modules, "cellxgene_census", mod)
    monkeypatch.setattr(soma_layout, "_RESOLVED", {})
    monkeypatch.setattr(soma_layout, "_RELEASES", {})
    out = load_verbs()["find"](_ctx(tmp_path), {"table": "cellxgene_census.obs", "where": {"dataset_id": "d1"},
                                                "columns": ["soma_joinid", "dataset_id"], "limit": 5})
    hdr = out["_vbt"]
    assert hdr["returned"] == 5 and hdr["total"] == 42 and hdr["truncated"] is True and hdr["status"] == "partial"
    assert any("every matching row was read (42)" in n for n in hdr["notes"]), hdr["notes"]
    assert not any("page(s) read" in n for n in hdr["notes"])


# --------------------------------------------------------------------------- live (VBT_DL_NETWORK=1)


@needs_network
@pytest.mark.parametrize("name", sorted(SCENARIOS))
async def test_live_scenarios(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The recorded scenarios against the live APIs: the answers have the recorded structure."""
    _fresh_caches(monkeypatch)
    out = await run_scenario(name, tmp_path)
    if name == "clinical_derived":
        assert out["header"]["served_by"] == "derived" and len(out["obj"]["patients"]) == 2
        assert out["prov"]["source"]["versions"]["portalVersion"]
    elif name == "clinical_phantom":
        assert out["error"]["kind"] == "not_found" and out["error"]["items"] == [PHANTOM]
    elif name == "clinical_unknown_study":
        assert out["error"]["kind"] == "not_found" and out["error"]["argument"] == "study_id"
    elif name == "study_details_release":
        assert out["header"]["source"].startswith("cbioportal@") and out["prov"]["source"]["versions"]
    elif name.startswith("release_"):
        assert out["release"]["versions"]
    elif name == "ceiling_count":
        tot = out["header"]["ceiling_totals"]
        assert tot["available"] >= tot["available_and_unchanged"] >= 0
    else:
        assert out["_vbt"]["total"] is not None


@needs_network
def test_live_census_stratified_sample_counts_what_the_filter_selects(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """On the real Census the sample reads the same cells the count counts and draws at most max_cells of them,
    balanced over (dataset_id, donor_id) donors; its value_filter selects exactly the sample."""
    pytest.importorskip("cellxgene_census")
    monkeypatch.setattr(soma_layout, "_RESOLVED", {})
    filt = "dataset_id == 'f7c1c579-2dc0-47e2-ba19-8165c5a0e353' and tissue_general == 'spleen' and " \
           "is_primary_data == True"
    out = load_verbs()["_census_count"](_ctx(tmp_path), {"value_filter": filt, "sample": {
        "max_cells": 300, "stratify": "cell_type", "key": ["dataset_id", "donor_id"]}})
    s = out["sample"]
    assert out["n_cells"] > 300 and s["n_total"] == out["n_cells"] and 0 < s["n_sampled"] <= 300
    assert s["value_filter"] == sample_filter(s["soma_joinids"]) and s["donor_key"] == ["dataset_id", "donor_id"]
    again = load_verbs()["_census_count"](_ctx(tmp_path), {"value_filter": s["value_filter"]})
    assert again["n_cells"] == s["n_sampled"]


@needs_network
def test_live_census_200k_id_filter_counts_the_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The 200,000-cell path: a spleen sample of at most 200,000 cells is a value_filter of about 2.1 MB that the soma
    layout counts as exactly the sample (on 2025-11-08: 577,677 primary spleen cells, 199,744 drawn over the 64
    donors of 7 datasets). In-process: 24.8 s and a 1,430 MB process tree on 2026-10-08."""
    pytest.importorskip("cellxgene_census")
    monkeypatch.setattr(soma_layout, "_RESOLVED", {})
    filt = "tissue_general == 'spleen' and is_primary_data == True"
    out = load_verbs()["_census_count"](_ctx(tmp_path), {"value_filter": filt, "sample": {
        "max_cells": 200_000, "stratify": "cell_type", "key": ["dataset_id", "donor_id"]}})
    s = out["sample"]
    assert s["n_total"] == out["n_cells"] > 200_000 and 100_000 < s["n_sampled"] <= 200_000
    assert len(s["value_filter"]) > 1_000_000 and s["n_donors_sampled"] == s["n_donors"]
    again = load_verbs()["_census_count"](_ctx(tmp_path), {"value_filter": s["value_filter"]})
    assert again["n_cells"] == s["n_sampled"]


@needs_network
async def test_live_spleen_pulls_are_sized_by_the_calibrated_estimate(tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """Through the gateway on the real Census with the single_cell server at 4,500 MB: the 200,000-cell spleen pull
    that was admitted and killed in S1 is refused too_large before upstream is called, and a 100,000-cell pull is
    admitted as the derived sample (upstream is never called here)."""
    pytest.importorskip("cellxgene_census")
    monkeypatch.setattr(soma_layout, "_RESOLVED", {})
    def listing(tool: str, args: dict[str, Any]) -> Any:      # the value_filter's vocabulary check, nothing else
        assert tool == "list_metadata_values", tool
        values = {"tissue_general": ["spleen"], "is_primary_data": ["True", "False"]}[args["column_name"]]
        return {"value_counts": [{"value": v, "count": 1} for v in values]}

    gw = gateway(tmp_path, [load_yaml(OVERLAYS / "single_cell.yaml")], upstream=listing,
                 data={"memory": {"default_server_mb": 4500}})
    args = {"value_filter": "tissue_general == 'spleen'", "ensembl_ids": ["ENSG00000198851", "ENSG00000156738"],
            "output_path": "spleen.h5ad"}
    with pytest.raises(GatewayError) as e:
        await gw.prepare("single_cell", "get_anndata_donor_balanced", {**args, "max_cells": 200_000}, None)
    assert e.value.kind == ErrorKind.too_large and e.value.payload["n_cells"] > 500_000
    plan = await gw.prepare("single_cell", "get_anndata_donor_balanced", {**args, "max_cells": 100_000}, None)
    assert plan.args_sent["value_filter"].startswith("soma_joinid in [") and plan.args_sent["max_cells"] <= 100_000


#: Peak RSS of real pulls (2026-10-08, Census 2025-11-08, the unmodified upstream function in a fresh process):
#: (tool, cells, genes named, MB); None: killed at the single_cell server's 4,500 MB limit (S1, through the server).
REAL_PULLS = [("get_anndata", 7_750, 2, 1_805), ("get_anndata", 7_750, None, 2_878),
              ("get_anndata_donor_balanced", 1_842, 2, 3_821), ("get_anndata_donor_balanced", 19_782, 2, 3_199),
              ("get_anndata_donor_balanced", 49_783, 2, 3_190), ("get_anndata_donor_balanced", 99_771, 2, 3_136),
              ("get_anndata_donor_balanced", 149_759, 2, 4_489), ("get_anndata_donor_balanced", 199_744, 2, None)]


@pytest.mark.parametrize("tool,cells,genes,peak_mb", REAL_PULLS)
def test_count_first_estimates_cover_the_real_pulls(tool: str, cells: int, genes: int | None,
                                                     peak_mb: int | None) -> None:
    """The shipped estimates (count_first.estimate) admit the real pulls that fit the server's 4,500 MB and refuse
    those that did not (the 199,744-cell pull was admitted at 40,000,000 bytes before and killed); every admitted estimate is
    at least the measured peak. A pull naming no gene reads every gene of each cell."""
    from vbt.datalayer.gateway.gateway import _estimate_fields
    from vbt.datalayer.service.verbs.census_count import estimate_bytes

    cf = Overlay.model_validate(load_yaml(OVERLAYS / "single_cell.yaml")).tools[tool].count_first
    req = CensusCountRequest(table=cf.table, n_genes=genes, **_estimate_fields(cf.estimate))
    need_mb = estimate_bytes(req, cells, cells) / (1024 * 1024)
    limit_mb = 4_500
    if peak_mb is None or peak_mb + 150 > limit_mb:          # the server adds its own footprint to the function's
        assert need_mb > limit_mb, (tool, cells, need_mb)
    else:
        assert peak_mb <= need_mb <= limit_mb, (tool, cells, need_mb)


def test_every_gene_argument_counts_toward_the_estimate() -> None:
    """Upstream ORs gene_symbols and ensembl_ids into one var filter: both count (S1's 200,000-cell call named only
    Ensembl IDs and was estimated with 0 genes)."""
    for tool in ("get_anndata", "get_anndata_donor_balanced"):
        cf = Overlay.model_validate(load_yaml(OVERLAYS / "single_cell.yaml")).tools[tool].count_first
        assert set(cf.genes_args) == {"gene_symbols", "ensembl_ids"} and cf.estimate.base_mb > 0


async def test_count_first_counts_ensembl_ids_and_sends_the_estimate(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """A call naming only Ensembl IDs (as S1's 200,000-cell call did) is estimated with those genes, not 0."""
    gw, _mod, _seen = census_gateway(tmp_path, monkeypatch, TWO_DATASETS)
    await gw.prepare("single_cell", "get_anndata_donor_balanced",
                     {"value_filter": "tissue == 'lung'", "max_cells": 20, "output_path": "s.h5ad",
                      "ensembl_ids": ["ENSG00000198851", "ENSG00000156738"]}, None)
    req = next(r for r in gw.service.verbs("_census_count") if r.sample is not None)
    assert req.n_genes == 2 and req.base_bytes == 3900 * 1024 * 1024 and req.read_all_bytes == 100


def test_census_count_request_model_carries_the_sample() -> None:
    """The gateway's sample request travels in CensusCountRequest.sample; the drawn sample and a written file's cells
    come back in typed response fields, and a release lookup is its own verb (``_release``)."""
    req = CensusCountRequest(table="cellxgene_census.obs", value_filter="x == 'y'",
                             sample={"max_cells": 5, "seed": 42, "key": ["dataset_id", "donor_id"],
                                     "stratify": "cell_type"})
    assert req.sample["stratify"] == "cell_type"
    resp = parse_response("_census_count", {"table": "cellxgene_census.obs", "n_cells": 9,
                                            "sample": {"soma_joinids": [1, 2]}, "file_cells": [1, 2]})
    assert resp.sample["soma_joinids"] == [1, 2] and resp.file_cells == [1, 2] and not resp.model_extra
    with pytest.raises(ValueError):
        CensusCountRequest(table="cellxgene_census.obs", release_only=True)
    rel = parse_response("_release", {"table": CT, "release": {"resolved": "2026-10-08T09:00:05"}})
    assert rel.release["resolved"] == "2026-10-08T09:00:05"
