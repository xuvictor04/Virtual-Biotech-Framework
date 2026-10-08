"""Live sources against the real APIs (F20): recorded responses replayed offline, and the live APIs on demand.

``tests/datalayer/real/live/`` holds small real responses of ClinicalTrials.gov v2, cBioPortal, NCBI
E-utilities and the CELLxGENE Census, retrieved on 2026-10-08 (each file names its URL, parameters, status,
retrieval time and how it was trimmed; ``MANIFEST.json`` lists them):

* ``replay/<scenario>.json``: every request one data-child scenario made to the real API (URL, parameters,
  status, headers, trimmed body). Offline, the scenario runs again on the shipped descriptors with a transport
  that answers from the recording and fails on any request it did not record, so the request shape is checked
  as well as the answer.
* ``<source>/*.json``: single responses (counts, unknown identifiers, the upstream servers' own answers) that
  pin facts the descriptors and overlays rely on.

With ``VBT_DL_NETWORK=1`` the same scenarios run against the live APIs (structure only: the registries move),
and the Census release and one count are resolved for real (``cellxgene_census`` installed).
"""

from __future__ import annotations

import copy
import json
import os
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import pytest

from test_dl_gateway_flow import REGISTRY, make_gateway, raw_of
from vbt.datalayer.catalog import Catalog
from vbt.datalayer.descriptor.load import load_yaml
from vbt.datalayer.descriptor.models import SourceDescriptor
from vbt.datalayer.errors import ErrorKind, GatewayError
from vbt.datalayer.ipc import VERB_VOCAB
from vbt.datalayer.plugins.layouts import live_api as live
from vbt.datalayer.plugins.layouts import soma as soma_layout
from vbt.datalayer.predicate import And, Contains, Eq, In, Range, TextMatch, to_json
from vbt.datalayer.service import ServiceContext
from vbt.datalayer.service.verbs import load_verbs
from vbt.datalayer.service.verbs.census_count import donor_balanced_columns, sample_filter
from vbt.datalayer.settings import DataSettings

REPO = Path(__file__).resolve().parents[2]
SOURCES = REPO / "configs" / "data" / "sources"
OVERLAYS = REPO / "configs" / "data" / "overlays"
FIX = Path(__file__).resolve().parent / "real" / "live"
NETWORK = os.environ.get("VBT_DL_NETWORK") == "1"
needs_network = pytest.mark.skipif(not NETWORK, reason="set VBT_DL_NETWORK=1 to call the live APIs")

ST = "protocolSection.statusModule.overallStatus"
PH = "protocolSection.designModule.phases[]"
NCT = "protocolSection.identificationModule.nctId"
COUNTRY = "protocolSection.contactsLocationsModule.locations[].country"
ACC = "acc_tcga_pan_can_atlas_2018"


def fixture(name: str) -> dict[str, Any]:
    return json.loads((FIX / name).read_text())


# --------------------------------------------------------------------------- scenarios


def _ctx(tmp_path: Path, *, budgets: dict[str, dict[str, Any]] | None = None, ceiling: str | None = None
         ) -> ServiceContext:
    """The data child's context on the shipped descriptors (real base URLs), optionally with other budgets."""
    descs = {}
    for name in ("clinicaltrials", "cbioportal", "pubmed", "census"):
        d = load_yaml(SOURCES / f"{name}.yaml")
        if budgets and d["source"] in budgets:
            d["budget"] = {**d.get("budget", {}), **budgets[d["source"]]}
        descs[d["source"]] = SourceDescriptor.model_validate(d)
    data: dict[str, Any] = {"cache_dir": str(tmp_path / "cache")}
    if ceiling:
        data["leakage"] = {"ceiling": ceiling}
    settings = DataSettings.from_dict(data, project_root=tmp_path)
    return ServiceContext(settings, catalog=Catalog(descs, {}, [], registry=REGISTRY), registry=REGISTRY)


def _run(ctx: ServiceContext, verb: str, payload: dict[str, Any]) -> dict[str, Any]:
    return load_verbs()[verb](ctx, payload)


def s_ctgov_count(tmp_path: Path) -> dict[str, Any]:
    pred = And((In(ST, ("RECRUITING",)), Contains(PH, "PHASE2")))
    return _run(_ctx(tmp_path), "_witness", {"table": "clinicaltrials_gov.studies", "predicate": to_json(pred)})


def s_ctgov_count_ceiling(tmp_path: Path) -> dict[str, Any]:
    pred = And((In(ST, ("COMPLETED",)), Contains(PH, "PHASE3"), Contains(COUNTRY, "Germany")))
    return _run(_ctx(tmp_path, ceiling="2017-12-31"), "_witness",
                {"table": "clinicaltrials_gov.studies", "predicate": to_json(pred)})


def s_ctgov_find_limit(tmp_path: Path) -> dict[str, Any]:
    return _run(_ctx(tmp_path), "find", {"table": "clinicaltrials_gov.studies", "where": {ST: "SUSPENDED"}, "limit": 2})


def s_ctgov_lookup_unknown(tmp_path: Path) -> dict[str, Any]:
    return _run(_ctx(tmp_path), "lookup", {"table": "clinicaltrials_gov.studies", "key": {NCT: "NCT99999999"}})


def s_cbio_patients_paged(tmp_path: Path) -> dict[str, Any]:
    ctx = _ctx(tmp_path, budgets={"cbioportal": {"page_size": 3, "max_pages": 2}})
    return _run(ctx, "_live_find", {"table": "cbioportal.patient",
                                    "predicate": to_json(Eq("studyId", "msk_impact_2017"))})


def s_cbio_unknown_study(tmp_path: Path) -> dict[str, Any]:
    return _run(_ctx(tmp_path), "find", {"table": "cbioportal.patient_clinical", "where": {"studyId": "no_such_study_xyz"}})


def s_cbio_clinical(tmp_path: Path) -> dict[str, Any]:
    return _run(_ctx(tmp_path), "find", {"table": "cbioportal.patient_clinical", "where": {"studyId": ACC},
                                         "limit": 1000})


def s_cbio_study(tmp_path: Path) -> dict[str, Any]:
    return _run(_ctx(tmp_path), "find", {"table": "cbioportal.study", "where": {"studyId": ACC}})


def s_cbio_molecular(tmp_path: Path) -> dict[str, Any]:
    where = {"molecularProfileId": f"{ACC}_rna_seq_v2_mrna", "sampleListId": f"{ACC}_all", "entrezGeneId": 80381}
    ctx = _ctx(tmp_path)
    found = _run(ctx, "find", {"table": "cbioportal.molecular_data", "where": where, "limit": 1000})
    pred = And(tuple(Eq(k, v) for k, v in where.items()))
    count = _run(ctx, "_witness", {"table": "cbioportal.molecular_data", "predicate": to_json(pred)})
    return {"find": found, "witness": count}


def s_pubmed_count_ceiling(tmp_path: Path) -> dict[str, Any]:
    pred = And((TextMatch("title", "PCSK9 AND evolocumab", "word"), Range("pubdate", None, "2017-12-31")))
    return _run(_ctx(tmp_path), "_witness", {"table": "pubmed.records", "predicate": to_json(pred)})


SCENARIOS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "ctgov_count": s_ctgov_count, "ctgov_count_ceiling": s_ctgov_count_ceiling, "ctgov_find_limit": s_ctgov_find_limit,
    "ctgov_lookup_unknown": s_ctgov_lookup_unknown, "cbio_patients_paged": s_cbio_patients_paged,
    "cbio_unknown_study": s_cbio_unknown_study, "cbio_clinical": s_cbio_clinical, "cbio_study": s_cbio_study,
    "cbio_molecular": s_cbio_molecular, "pubmed_count_ceiling": s_pubmed_count_ceiling,
}


# --------------------------------------------------------------------------- transports


class Replay:
    """Answers each request from a recorded exchange with the same URL and parameters (each used once)."""

    def __init__(self, exchanges: list[dict[str, Any]]) -> None:
        self.left = [dict(e) for e in exchanges]
        self.sent: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, params: Any = None, *, timeout: float = 30.0, headers: Any = None
                 ) -> tuple[int, dict[str, str], bytes]:
        sent = {str(k): str(v) for k, v in (params or {}).items() if v is not None}
        self.sent.append((url, sent))
        for i, e in enumerate(self.left):
            if e["url"] == url and e["params"] == sent:
                self.left.pop(i)
                body = e.get("text") if "text" in e else json.dumps(e.get("body"))
                return int(e["status"]), dict(e.get("headers") or {}), (body or "").encode()
        raise AssertionError(f"unrecorded request {url} {sent}")


@pytest.fixture
def replay(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], Replay]:
    """``replay(name)`` installs the recorded exchanges of ``replay/<name>.json`` as the HTTP transport."""
    monkeypatch.setattr(live, "_wait_turn", lambda base, rpm: None)
    monkeypatch.setattr(live, "_RELEASES", {})

    def install(name: str) -> Replay:
        rec = fixture(f"replay/{name}.json")
        r = Replay(rec["exchanges"])
        monkeypatch.setattr(live.LiveApiLayout, "transport", staticmethod(r))
        return r
    return install


# --------------------------------------------------------------------------- ClinicalTrials.gov


def test_ctgov_count_request_asks_for_one_field(tmp_path: Path, replay: Callable[[str], Replay]) -> None:
    """The registry ignores pageSize=0 (10 full records, ~140 KB, came back with the count): the remote
    witness asks for the NCT ID only."""
    r = replay("ctgov_count")
    out = s_ctgov_count(tmp_path)
    assert out["total"] == fixture("replay/ctgov_count.json")["exchanges"][0]["body"]["totalCount"]
    assert out["total_method"] == "scan" and out["reason"] == "remote count request"
    # the witness also names the registry's data release (the version endpoint, read at most hourly), which
    # an upstream-served call records as its source release
    assert out["as_of"] == "2026-10-07T09:00:06"
    (url, sent), = [(u, s) for u, s in r.sent if not u.endswith("/version")]
    assert sent["fields"] == "NCTId" and sent["countTotal"] == "true" and sent["filter.advanced"] == "AREA[Phase]PHASE2"
    shape = fixture("ctgov/count_page_size_0_without_fields.json")["body"]
    assert shape["studies_returned"] == 10 and shape["body_bytes"] > 100_000      # what pageSize=0 alone costs


def test_ctgov_count_under_the_evidence_ceiling(tmp_path: Path, replay: Callable[[str], Replay]) -> None:
    """Under data.leakage.ceiling the tool's count is asked with AREA[StudyFirstPostDate]RANGE[MIN,ceiling]
    (the overlay's leakage_filter); the remote witness counts the same thing (3,846 against 4,700 without it,
    which made every bounded count a tool_defect)."""
    r = replay("ctgov_count_ceiling")
    out = s_ctgov_count_ceiling(tmp_path)
    (url, sent), = [(u, s) for u, s in r.sent if not u.endswith("/version")]
    assert "AREA[StudyFirstPostDate]RANGE[MIN,2017-12-31]" in sent["filter.advanced"]
    assert out["total"] == fixture("ctgov/count_germany_phase3_completed_ceiling_2017.json")["body"]["totalCount"]
    assert out["total"] < fixture("ctgov/count_germany_phase3_completed.json")["body"]["totalCount"]
    assert "evidence ceiling 2017-12-31" in out["reason"]


def test_ctgov_find_reads_what_the_limit_asks_for(tmp_path: Path, replay: Callable[[str], Replay]) -> None:
    """A find of 2 trials reads one page of 2 (it read ten pages of 100, 11.8 MB, before), takes the total
    from one count request and names the registry's data timestamp as its as_of."""
    r = replay("ctgov_find_limit")
    out = s_ctgov_find_limit(tmp_path)
    hdr = out["_vbt"]
    assert len(out["rows"]) == 2 and hdr["status"] == "partial" and hdr["truncated"] is True
    version = next(e for e in fixture("replay/ctgov_find_limit.json")["exchanges"] if e["url"].endswith("/version"))
    assert hdr["source"] == f"clinicaltrials_gov@{version['body']['dataTimestamp']}"
    assert isinstance(hdr["total"], int) and hdr["total"] > 2 and hdr["total_method"] == "remote"
    pages = [s for u, s in r.sent if u.endswith("/studies") and "countTotal" not in s]
    assert len(pages) == 1 and pages[0]["pageSize"] == "2"
    assert len(r.sent) == 3                                        # one page, the version, one count
    assert set(out["record_versions"]["clinicaltrials_gov.studies"]) == {
        row["protocolSection"]["identificationModule"]["nctId"] for row in out["rows"]}


def test_ctgov_unknown_nct(tmp_path: Path, replay: Callable[[str], Replay]) -> None:
    """filter.ids of an unknown NCT ID is an empty page (the record endpoint answers 404 text): the registry is
    the authority on its IDs, so the lookup is not_found naming the key (it was empty_unverified), as the
    upstream get_clinical_trial_details answers."""
    replay("ctgov_lookup_unknown")
    out = s_ctgov_lookup_unknown(tmp_path)
    assert out["status"] == "tool_error" and out["kind"] == "not_found" and "NCT99999999" in out["message"]
    rec = fixture("ctgov/study_unknown_nct.json")
    assert rec["status"] == 404 and "NCT99999999 not found" in rec["text"]


def test_live_lookup_takes_the_short_key_name(tmp_path: Path, replay: Callable[[str], Replay]) -> None:
    """``nctId`` names protocolSection.identificationModule.nctId, as a local lookup's short names do: the
    same request goes out (the replay fails on any other) and a key without it is incomplete_key."""
    replay("ctgov_lookup_unknown")
    out = _run(_ctx(tmp_path), "lookup", {"table": "clinicaltrials_gov.studies", "key": {"nctId": "NCT99999999"}})
    assert out["kind"] == "not_found" and "NCT99999999" in out["message"]
    bad = _run(_ctx(tmp_path), "lookup", {"table": "clinicaltrials_gov.studies", "key": {"briefTitle": "x"}})
    assert "incomplete_key" in json.dumps(bad), bad


async def test_country_filter_is_evaluated_on_upstream_rows(tmp_path: Path) -> None:
    """Upstream flattens contactsLocationsModule.locations to `locations`: unmapped, the country predicate
    excluded every row as unknown (a real Iceland search answered total_count 3 with 0 trials)."""
    rec = fixture("ctgov/upstream_search_iceland.json")
    gw = make_gateway(tmp_path, [load_yaml(SOURCES / "clinicaltrials.yaml"), load_yaml(SOURCES / "cbioportal.yaml")],
                      [load_yaml(OVERLAYS / "clinicaltrials.yaml")], {}, data={"witness": {"enabled": False}},
                      fail={VERB_VOCAB})              # as live: "country: vocabulary not available; value not checked"
    res = await _call(gw, "clinicaltrials", "search_clinical_trials", rec["args"], lambda tool, a: rec["result"])
    hdr = res.header
    assert hdr["returned"] == 3 and hdr["status"] == "ok" and not hdr.get("excluded_unknown")
    assert sorted(t["nctId"] for t in res.obj["trials"]) == sorted(t["nctId"] for t in rec["result"]["trials"])


# --------------------------------------------------------------------------- cBioPortal


def test_cbioportal_pages_by_page_number(tmp_path: Path, replay: Callable[[str], Replay]) -> None:
    """pageNumber is a page index: page 1 follows page 0 (an offset sent there skipped every later page,
    and 500 of msk_impact_2017's 10,336 patients came back as complete); a cut read takes its total from
    the total-count header."""
    r = replay("cbio_patients_paged")
    out = s_cbio_patients_paged(tmp_path)
    numbers = [s.get("pageNumber") for u, s in r.sent if u.endswith("/patients") and s.get("projection") != "META"]
    assert numbers == [None, "1"]
    assert len(out["rows"]) == 6 and len({p["patientId"] for p in out["rows"]}) == 6
    assert out["truncated"] is True and out["total"] == 10336 and out["total_method"] == "remote"


def test_cbioportal_unknown_study_is_not_found(tmp_path: Path, replay: Callable[[str], Replay]) -> None:
    replay("cbio_unknown_study")
    out = s_cbio_unknown_study(tmp_path)
    assert out["status"] == "tool_error" and out["kind"] == "not_found"
    assert out["argument"] == "studyId" and out["value"] == "no_such_study_xyz"


def test_cbioportal_patient_clinical_pivot(tmp_path: Path, replay: Callable[[str], Replay]) -> None:
    """One row per patient with one column per attribute (the recording keeps 6 patients)."""
    replay("cbio_clinical")
    out = s_cbio_clinical(tmp_path)
    rows = out["rows"]
    assert len(rows) == 6 and len({r["patientId"] for r in rows}) == 6 and not any("_conflicts" in r for r in rows)
    assert all(r["studyId"] == ACC for r in rows) and all("OS_STATUS" in r for r in rows)


def test_cbioportal_status_encodings_cover_the_real_values() -> None:
    """Every survival status value of a TCGA PanCancer study is in the descriptor's encoding (PFS and DSS
    were missing) or its missing values."""
    table = SourceDescriptor.model_validate(load_yaml(SOURCES / "cbioportal.yaml")).tables["patient_clinical"]
    seen = fixture("cbioportal/acc_patient_clinical.json")["body"]["status_values"]
    endpoints = {"OS_STATUS", "DFS_STATUS", "PFS_STATUS", "DSS_STATUS"}
    assert endpoints < set(seen)
    for attr in endpoints:
        col = table.columns[attr]
        assert col.verified is not False and col.event_of == attr.replace("_STATUS", "_MONTHS")
        for v in seen[attr]:
            assert col.encoding.get(v) is v.startswith("1:"), (attr, v)
    # a *_STATUS attribute that is not a survival event is no event flag (it matched "{stem}_STATUS" before)
    assert set(seen["PERSON_NEOPLASM_CANCER_STATUS"]) == {"With Tumor", "Tumor Free"}
    assert "PERSON_NEOPLASM_CANCER_STATUS" not in table.columns
    assert table.pattern_column("PERSON_NEOPLASM_CANCER_STATUS") is None
    assert table.pattern_column("OS_MONTHS") is not None


def test_cbioportal_study_by_key(tmp_path: Path, replay: Callable[[str], Replay]) -> None:
    """A study fixed by its key is read from studies/{studyId} (1.4 KB), not from the 550-study listing."""
    r = replay("cbio_study")
    out = s_cbio_study(tmp_path)
    assert [u for u, _ in r.sent] == ["https://www.cbioportal.org/api/studies/" + ACC]
    (row,) = out["rows"]
    assert row["studyId"] == ACC and row["importDate"][4] == "-" and row["importDate"][10] == " "


def test_cbioportal_molecular_profile_table(tmp_path: Path, replay: Callable[[str], Replay]) -> None:
    """The expression download of vbt.analysis.survival: one gene of an RNA-seq profile, and its count."""
    replay("cbio_molecular")
    out = s_cbio_molecular(tmp_path)
    rows = out["find"]["rows"]
    assert rows and all(r["entrezGeneId"] == 80381 and isinstance(r["value"], float) for r in rows)
    assert out["witness"]["total"] == 78 and out["witness"]["total_method"] == "scan"


async def test_get_clinical_data_guards_on_real_rows(tmp_path: Path) -> None:
    """The upstream answer for two real samples and one that does not exist: the phantom (patientId null)
    is not_found, and patient attributes are counted once per patient in the `patients` section."""
    rec = fixture("cbioportal/upstream_clinical_data_phantom.json")
    descs = [load_yaml(SOURCES / "clinicaltrials.yaml"), load_yaml(SOURCES / "cbioportal.yaml")]
    gw = make_gateway(tmp_path / "a", descs, [load_yaml(OVERLAYS / "clinicaltrials.yaml")], {},
                      data={"witness": {"enabled": False}})
    with pytest.raises(GatewayError) as e:
        await _call(gw, "clinicaltrials", "get_clinical_data", rec["args"], lambda tool, a: rec["result"])
    assert e.value.kind == ErrorKind.not_found and "TCGA-OR-A5ZZ-01" in e.value.message
    real = copy.deepcopy(rec["result"])
    real["data"] = [r for r in real["data"] if r.get("patientId")]
    real["sample_count"] = len(real["data"])
    gw = make_gateway(tmp_path / "b", descs, [load_yaml(OVERLAYS / "clinicaltrials.yaml")], {},
                      data={"witness": {"enabled": False}})
    args = {"study_id": ACC, "sample_ids": [r["sampleId"] for r in real["data"]]}
    res = await _call(gw, "clinicaltrials", "get_clinical_data", args, lambda tool, a: real)
    patients = res.obj["patients"]
    assert len(patients) == 2 and all("OS_STATUS" in p and "RACE" in p for p in patients)
    assert all("OS_STATUS" not in r and "RACE" not in r for r in res.obj["data"])
    assert res.header["grains"]["patient"]["total"] == 2


# --------------------------------------------------------------------------- E-utilities


def test_pubmed_count_under_the_literature_ceiling(tmp_path: Path, replay: Callable[[str], Replay]) -> None:
    r = replay("pubmed_count_ceiling")
    out = s_pubmed_count_ceiling(tmp_path)
    (url, sent), = r.sent
    assert sent["rettype"] == "count" and sent["maxdate"] == "2017/12/31" and sent["datetype"] == "pdat"
    assert out["total"] == int(fixture("eutils/esearch_count_ceiling_2017.json")["body"]["esearchresult"]["count"])
    assert out["total"] < int(fixture("eutils/esearch_count.json")["body"]["esearchresult"]["count"])


def test_pmcid_read_as_pmid_names_another_paper() -> None:
    """PMC6309485 sent to efetch as a PMID returns PMID 6309485; the ID converter maps it to PMID 30009548.
    The pmid plugin rejects it and says it looks like a PMCID."""
    fetched = fixture("eutils/efetch_pmcid_as_pmid.json")["body"]
    converted = fixture("eutils/pmc_idconv.json")["body"]["records"][0]
    assert fetched["pmids"] == ["6309485"] and str(converted["pmid"]) == "30009548"
    plugin = REGISTRY.find("identifier", "pmid").configure({}, None)
    out = plugin.normalize("PMC6309485")
    assert getattr(out, "looks_like", None) and "pmcid" in out.looks_like
    summary = fixture("eutils/esummary_known_and_unknown.json")["body"]["result"]
    assert summary["99999999"].get("error") and summary["28304224"]["title"].startswith("Evolocumab")


def test_esearch_reports_ignored_phrases() -> None:
    body = fixture("eutils/esearch_ignored_phrase.json")["body"]["esearchresult"]
    assert body["errorlist"]["phrasesnotfound"] == ["zzzqqqxyzzy"] and int(body["count"]) > 0


# --------------------------------------------------------------------------- Census


def test_census_stable_alias_resolves_to_the_recorded_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rel = fixture("census/release.json")["body"]
    stable = rel["stable"]
    mod = types.SimpleNamespace(get_census_version_description=lambda v: dict(rel[rel.get(v, v)]))
    monkeypatch.setattr(soma_layout, "_RESOLVED", {})
    out = soma_layout.resolve_version("stable", module=mod, record=tmp_path / "v.json")
    assert out["resolved"] == stable == rel[stable]["release_build"] and out["release_confidence"] == "inferred"
    probe = fixture("census/probe.json")
    assert probe["description"]["release_build"] == stable


def test_census_soma_reads_use_small_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    """open_soma reserves 1 GiB per column by default, which fails under RLIMIT_DATA (std::bad_alloc on the
    real Census at 260 MB resident): the soma layout passes 32 MiB buffers (128 MiB failed once a session had counted),
    and counts stream the batches."""
    seen: dict[str, Any] = {}

    class Frame:
        def read(self, **kw: Any) -> Any:
            seen["read"] = kw
            return iter([types.SimpleNamespace(num_rows=3), types.SimpleNamespace(num_rows=4)])

    def open_soma(census_version: str, tiledb_config: dict[str, Any] | None = None) -> Any:
        seen["config"] = tiledb_config
        return {"census_data": {"homo_sapiens": {"obs": Frame()}}}

    mod = types.SimpleNamespace(open_soma=open_soma,
                                get_census_version_description=lambda v: {"release_build": "2025-11-08"})
    monkeypatch.setattr(soma_layout, "_RESOLVED", {})
    layout = soma_layout.SomaLayout()
    layout.module = mod
    from vbt.datalayer.plugins.base import LayoutSpec

    spec = LayoutSpec(table="cellxgene_census.obs", path=None, options={"uri": "census_data/homo_sapiens/obs"})
    assert layout.count(spec, predicate=Eq("dataset_id", "d1")) == 7
    assert seen["config"] == soma_layout.DEFAULT_TILEDB_CONFIG
    assert seen["config"]["soma.init_buffer_bytes"] == 32 * 1024 ** 2
    assert seen["read"] == {"column_names": ["soma_joinid"], "value_filter": "dataset_id == 'd1'"}


def test_donor_balanced_sample_on_real_donor_counts() -> None:
    """Spleen in the 2025-11-08 Census: six donor_id labels occur in two datasets. Keyed by (dataset_id,
    donor_id) they are separate donors (34, not 28); the sample is split evenly over the datasets."""
    rec = fixture("census/spleen_donors.json")
    ids, datasets, donors = [], [], []
    nxt = 0
    for pair, n in sorted(rec["cells_per_pair"].items()):
        ds, donor = pair.split("/", 1)
        ids.extend(range(nxt, nxt + n))
        datasets.extend([ds] * n)
        donors.extend([donor] * n)
        nxt += n
    out = donor_balanced_columns(ids, datasets, donors, 600, seed=0)
    live_sample = fixture("census/probe.json")["sample"]
    assert out["n_sampled"] == live_sample["n_sampled"] == 600
    assert out["per_dataset"] == live_sample["per_dataset"]
    assert out["n_donors"] == live_sample["n_donors"] == len(rec["cells_per_pair"]) == 34
    assert len({d for d in donors}) == 28 and set(rec["shared_labels"]) <= set(donors)
    assert {k for k in out["per_donor"] if k.endswith("/D496")} == {
        "1b350d0a-4535-4879-beb6-1142f3f94947/D496", "1b9d8702-5af8-4142-85ed-020eb06ec4f6/D496"}
    vf = sample_filter(out["soma_joinids"][:3])
    assert vf == "soma_joinid in [" + ", ".join(str(i) for i in out["soma_joinids"][:3]) + "]"


def test_census_x_values_are_counts() -> None:
    """census.yaml anndata_outputs X: whole-number counts with implicit zeros (no longer verified: false)."""
    x = fixture("census/probe.json")["x_raw_slice"]
    assert x["stored_values"] > 0 and x["non_integer"] == 0 and x["stored_zeros"] == 0 and x["min"] >= 1
    desc = SourceDescriptor.model_validate(load_yaml(SOURCES / "census.yaml"))
    values = desc.tables["anndata_outputs"].matrix.values["X"]
    assert values.verified is not False and values.missing == "zero"


# --------------------------------------------------------------------------- identifiers


@pytest.mark.parametrize("id_type,values", [
    ("cbio_patient", ["P-0000004", "TCGA-OR-A5J1"]),
    ("cbio_sample", ["TCGA-OR-A5J1-01", "P-0000004-T01-IM3"]),
    ("cbio_study", [ACC, "msk_impact_2017"]),
    ("nct_id", ["NCT01295827", "NCT04368728"]),
    ("pmid", ["28304224", "30009548"]),
    ("pmcid", ["PMC6309485"]),
    ("doi", ["10.1056/NEJMoa1615664", "10.1111/pcmr.12724"]),
    ("census_feature", ["ENSG00000103855", "ENSG00000145623"]),
])
def test_identifier_plugins_accept_real_ids(id_type: str, values: list[str]) -> None:
    """IDs seen in the recorded responses normalize to themselves."""
    plugin = REGISTRY.find("identifier", id_type).configure({}, None)
    assert [getattr(plugin.normalize(v), "value", None) for v in values] == values


# --------------------------------------------------------------------------- the user agent


def test_requests_name_their_user_agent() -> None:
    """cBioPortal refuses the default Python-urllib agent (403, recorded in MANIFEST.json)."""
    agents: list[str] = []

    class H(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            agents.append(self.headers.get("User-Agent", ""))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"[]")

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        live.http_get(f"http://127.0.0.1:{server.server_address[1]}/x")
    finally:
        server.shutdown()
        server.server_close()
    assert agents == ["vbt-datalayer"]
    refused = [f for f in fixture("MANIFEST.json")["files"] if f.get("user_agent")]
    assert {f["user_agent"]: f["status"] for f in refused} == {"Python-urllib/3.11": 403, "vbt-datalayer": 200}


# --------------------------------------------------------------------------- live (VBT_DL_NETWORK=1)


@needs_network
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_live_scenarios(name: str, tmp_path: Path) -> None:
    """The recorded scenarios against the live APIs: answers have the recorded structure."""
    out = SCENARIOS[name](tmp_path)
    if name == "cbio_unknown_study":
        assert out["kind"] == "not_found"
    elif name == "cbio_molecular":
        assert out["find"]["rows"] and isinstance(out["witness"]["total"], int)
    elif name in ("ctgov_count", "ctgov_count_ceiling", "pubmed_count_ceiling"):
        assert isinstance(out["total"], int) and out["total"] > 0 and out["total_method"] == "scan"
    elif name == "cbio_patients_paged":
        assert len(out["rows"]) == 6 and out["truncated"] and out["total"] > 6
    elif name == "ctgov_lookup_unknown":
        assert out["rows"] == []
    else:
        assert out["rows"], out


@needs_network
def test_live_census_release_and_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("cellxgene_census")
    monkeypatch.setattr(soma_layout, "_RESOLVED", {})
    ctx = _ctx(tmp_path)
    probe = fixture("census/probe.json")["count"]
    out = load_verbs()["_census_count"](ctx, {"value_filter": probe["filter"], "n_genes": 2, "cap_bytes": 10 ** 9})
    assert out["release"]["release_confidence"] == "inferred" and len(out["release"]["resolved"]) == 10
    assert isinstance(out["n_cells"], int) and out["n_cells"] > 0 and out["admissible"] is True


# --------------------------------------------------------------------------- helpers


async def _call(gw: Any, server: str, tool: str, args: dict[str, Any], upstream: Callable[[str, dict[str, Any]], Any]
                ) -> Any:
    plan = await gw.prepare(server, tool, args, None)
    gw.bridge.upstream = upstream
    raw = raw_of(upstream(tool, dict(plan.args_sent))) if plan.route == "upstream" else None
    return await gw.finish(plan, raw)
