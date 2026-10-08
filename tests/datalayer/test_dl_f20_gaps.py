"""F20 gaps closed in this pass (offline, scripted transports):

* live_api: page-index paging (cBioPortal ``pageNumber``), record endpoints (``key_endpoint``; 404 = no
  record), bounded 429/503 retries with ``Retry-After``, pages sized by the caller's limit and reading that
  stops at the limit, typed errors for failed requests (404 on a filled path = ``not_found``);
* the remote witness under the evidence ceiling (``leakage.counts: inject_filter``);
* the soma layout: small TileDB buffers and a columnar read; the derived donor-balanced sample's upstream
  arguments and its read cap;
* ``vbt.analysis.survival``: the expression download goes through ``vbt.datalayer.client``.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from test_dl_gateway_flow import REGISTRY
from vbt.datalayer.catalog import Catalog
from vbt.datalayer.descriptor.load import load_yaml
from vbt.datalayer.descriptor.models import SourceDescriptor
from vbt.datalayer.errors import ErrorKind
from vbt.datalayer.plugins.base import LayoutSpec
from vbt.datalayer.plugins.layouts import live_api as live
from vbt.datalayer.plugins.layouts import soma as soma_layout
from vbt.datalayer.plugins.layouts.live_api import Budget, LiveApiLayout, RemoteError
from vbt.datalayer.predicate import And, Cmp, Eq, In, to_json
from vbt.datalayer.service import ServiceContext
from vbt.datalayer.service.verbs import load_verbs
from vbt.datalayer.service.verbs.witness import leakage_conjunct, remote_failure
from vbt.datalayer.settings import DataSettings

REPO = Path(__file__).resolve().parents[2]
SOURCES = REPO / "configs" / "data" / "sources"


class Script:
    """A transport answering requests in order from ``(status, headers, body)`` triples; records requests."""

    def __init__(self, *answers: tuple[int, dict[str, str], Any]) -> None:
        self.answers = list(answers)
        self.sent: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, params: Any = None, *, timeout: float = 30.0, headers: Any = None
                 ) -> tuple[int, dict[str, str], bytes]:
        self.sent.append((url, {str(k): str(v) for k, v in (params or {}).items()}))
        status, hdrs, body = self.answers.pop(0)
        return status, dict(hdrs), (body if isinstance(body, bytes) else json.dumps(body).encode())


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(live, "_wait_turn", lambda base, rpm: None)
    monkeypatch.setattr(live, "_RELEASES", {})
    slept: list[float] = []
    monkeypatch.setattr(live.time, "sleep", slept.append)

    def install(*answers: tuple[int, dict[str, str], Any]) -> Script:
        s = Script(*answers)
        s.slept = slept                                    # type: ignore[attr-defined]
        monkeypatch.setattr(LiveApiLayout, "transport", staticmethod(s))
        return s
    return install


def _spec(**options: Any) -> LayoutSpec:
    return LayoutSpec(table="s.t", path=None, options={"base_url": "https://api.test", **options})


def _ctx(tmp_path: Path, *, ceiling: str | None = None) -> ServiceContext:
    descs = {d["source"]: SourceDescriptor.model_validate(d)
             for d in (load_yaml(SOURCES / f"{n}.yaml") for n in ("clinicaltrials", "cbioportal", "census"))}
    data: dict[str, Any] = {"cache_dir": str(tmp_path / "cache")}
    if ceiling:
        data["leakage"] = {"ceiling": ceiling}
    return ServiceContext(DataSettings.from_dict(data, project_root=tmp_path),
                          catalog=Catalog(descs, {}, [], registry=REGISTRY), registry=REGISTRY)


# --------------------------------------------------------------------------- live_api


def test_page_numbers_are_page_indexes(script: Any) -> None:
    s = script((200, {}, [{"id": 1}, {"id": 2}]), (200, {}, [{"id": 3}, {"id": 4}]), (200, {}, [{"id": 5}]))
    spec = _spec(endpoint="items", page_size_param="pageSize", page_number_param="pageNumber")
    got = live.fetch_all(LiveApiLayout(), spec, None, Budget(page_size=2))
    assert [r["id"] for r in got["rows"]] == [1, 2, 3, 4, 5] and not got["truncated"] and got["pages"] == 3
    assert [q.get("pageNumber") for _, q in s.sent] == [None, "1", "2"]
    assert all(q["pageSize"] == "2" for _, q in s.sent)


def test_reading_stops_at_max_rows_and_says_truncated(script: Any) -> None:
    s = script((200, {}, {"rows": [1, 2], "next": "t2"}), (200, {}, {"rows": [3, 4], "next": "t3"}))
    spec = _spec(endpoint="items", rows_path="$.rows", next_path="$.next", page_token_param="token")
    got = live.fetch_all(LiveApiLayout(), spec, None, Budget(max_pages=10), max_rows=3)
    assert got["rows"] == [1, 2, 3, 4] and got["truncated"] and got["pages"] == 2 and len(s.sent) == 2


def test_budget_pages_of_the_limit() -> None:
    b = Budget(page_size=500, max_pages=20, requests_per_min=60)
    assert b.pages_of(2).page_size == 2 and b.pages_of(2).max_pages == 20 and b.pages_of(None) is b
    assert b.pages_of(1000).page_size == 500 and Budget().pages_of(7).page_size == 7


def test_key_endpoint_reads_one_record_and_404_is_no_record(script: Any) -> None:
    s = script((200, {}, {"studyId": "acc", "name": "ACC"}), (404, {}, {"message": "Study not found: zz"}))
    spec = _spec(endpoint="studies", params={"projection": "DETAILED"}, key_endpoint="studies/{studyId}")
    layout = LiveApiLayout()
    page = layout.request(spec, predicate=Eq("studyId", "acc"), projection=[], page_token=None, budget=None)
    assert page.rows == [{"studyId": "acc", "name": "ACC"}] and page.total == 1
    assert s.sent[0] == ("https://api.test/studies/acc", {"projection": "DETAILED"})
    page = layout.request(spec, predicate=Eq("studyId", "zz"), projection=[], page_token=None, budget=None)
    assert page.rows == [] and page.total == 0
    # without the key fixed, the listing is read (the rest of the filter applied to its rows)
    s = script((200, {}, [{"studyId": "a", "cancerTypeId": "x"}, {"studyId": "b", "cancerTypeId": "y"}]))
    page = layout.request(spec, predicate=Eq("cancerTypeId", "y"), projection=[], page_token=None, budget=None)
    assert [r["studyId"] for r in page.rows] == ["b"] and s.sent[0][0] == "https://api.test/studies"


def test_429_is_retried_after_retry_after(script: Any) -> None:
    s = script((429, {"Retry-After": "3"}, {"error": "slow down"}), (503, {}, b""), (200, {}, [{"id": 1}]))
    page = LiveApiLayout().request(_spec(endpoint="items"), predicate=None, projection=[], page_token=None,
                                   budget=Budget(max_requests_per_call=5))
    assert page.rows == [{"id": 1}] and len(s.sent) == 3 and s.slept == [3.0, 2.0]
    s = script(*[(429, {}, b"")] * (live.RETRIES + 1))
    with pytest.raises(RemoteError) as e:
        LiveApiLayout().request(_spec(endpoint="items"), predicate=None, projection=[], page_token=None, budget=None)
    assert e.value.status == 429 and len(s.sent) == live.RETRIES + 1


def test_failed_requests_are_typed_errors(script: Any) -> None:
    script((404, {}, {"message": "Study not found: zz"}))
    with pytest.raises(RemoteError) as e:
        LiveApiLayout().request(_spec(endpoint="studies/{studyId}/patients"), predicate=Eq("studyId", "zz"),
                                projection=[], page_token=None, budget=None)
    assert e.value.filled == {"studyId": "zz"}
    err = remote_failure("cbioportal.patient", e.value)
    assert err.kind == ErrorKind.not_found and err.argument == "studyId" and err.value == "zz"
    assert remote_failure("t", RemoteError("bad", status=400)).kind == ErrorKind.invalid_argument
    down = remote_failure("t", RemoteError("x", status=503))
    assert down.kind == ErrorKind.source_error and down.retryable == "later"


def test_release_is_cached(script: Any) -> None:
    s = script((200, {}, {"dataTimestamp": "2026-10-07T09:00:06"}))
    spec = _spec(endpoint="studies", release={"endpoint": "version", "path": "$.dataTimestamp"})
    layout = LiveApiLayout()
    assert layout.release(spec, max_age_s=600) == "2026-10-07T09:00:06"
    assert layout.release(spec, max_age_s=600) == "2026-10-07T09:00:06" and len(s.sent) == 1


# --------------------------------------------------------------------------- remote witness and ceiling


def test_witness_counts_under_the_evidence_ceiling(tmp_path: Path, script: Any) -> None:
    s = script((200, {}, {"totalCount": 3846, "studies": []}))
    ctx = _ctx(tmp_path, ceiling="2017-12-31")
    t = ctx.table("clinicaltrials_gov.studies")
    assert leakage_conjunct(ctx, t) == Cmp("protocolSection.statusModule.studyFirstPostDateStruct.date", "<=",
                                           "2017-12-31")
    assert leakage_conjunct(_ctx(tmp_path / "x"), t) is None
    assert leakage_conjunct(ctx, ctx.table("cbioportal.sample")) is None          # no leakage block
    pred = And((In("protocolSection.statusModule.overallStatus", ("COMPLETED",)),
                Eq("protocolSection.designModule.studyType", "INTERVENTIONAL")))
    out = load_verbs()["_witness"](ctx, {"table": "clinicaltrials_gov.studies", "predicate": to_json(pred)})
    assert out["total"] == 3846 and "evidence ceiling 2017-12-31" in out["reason"]
    q = s.sent[0][1]
    assert q["filter.advanced"] == "AREA[StudyType]INTERVENTIONAL AND AREA[StudyFirstPostDate]RANGE[MIN,2017-12-31]"
    assert q["fields"] == "NCTId" and q["filter.overallStatus"] == "COMPLETED"


# --------------------------------------------------------------------------- Census


class _Rows(list):
    def to_pylist(self) -> list[dict[str, Any]]:
        return list(self)


class _Frame:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def read(self, value_filter: str | None = None, column_names: list[str] | None = None) -> Any:
        from vbt.datalayer.gateway.soma_filter import parse
        from vbt.datalayer.predicate import evaluate

        rows = [r for r in self.rows if value_filter is None or evaluate(parse(value_filter), r) is True]
        rows = [{k: r.get(k) for k in (column_names or r)} for r in rows]
        return types.SimpleNamespace(concat=lambda: _Rows(rows))


def _census(rows: list[dict[str, Any]]) -> Any:
    frame = _Frame(rows)
    return types.SimpleNamespace(
        open_soma=lambda census_version=None, tiledb_config=None: types.SimpleNamespace(
            close=lambda: None, census_data={"homo_sapiens": types.SimpleNamespace(obs=frame)}),
        get_census_version_description=lambda v: {"release_build": "2025-11-08"})


CELLS = [{"soma_joinid": i, "dataset_id": "dA" if i < 10 else "dB", "donor_id": "D1" if i % 2 else "D2",
          "is_primary_data": True} for i in range(20)]


def test_donor_balanced_sample_returns_the_upstream_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The sample comes with the value_filter and max_cells that make upstream fetch exactly its cells
    (upstream keeps every cell when the filter selects at most max_cells)."""
    monkeypatch.setattr(soma_layout, "_RESOLVED", {})
    monkeypatch.setitem(sys.modules, "cellxgene_census", _census(CELLS))
    out = load_verbs()["_census_count"](_ctx(tmp_path), {"value_filter": "is_primary_data == True",
                                                         "sample": {"max_cells": 8, "seed": 3}})
    sample = out["sample"]
    assert sample["n_sampled"] == 8 and sample["max_cells"] == 8 and sample["seed"] == 3
    assert sample["per_dataset"] == {"dA": 4, "dB": 4} and sample["n_donors"] == 4
    assert sample["value_filter"] == "soma_joinid in [" + ", ".join(map(str, sample["soma_joinids"])) + "]"
    assert out["n_cells_pulled"] == 8                                  # the pull is sized by the sample
    capped = load_verbs()["_census_count"](_ctx(tmp_path), {"value_filter": "is_primary_data == True",
                                                            "sample": {"max_cells": 8, "max_read": 5}})
    assert capped["sample"]["n_sampled"] is None and "at most 5" in capped["sample"]["reason"]


def test_soma_columns_read_and_tiledb_config_option(monkeypatch: pytest.MonkeyPatch) -> None:
    configs: list[Any] = []
    frame = _Frame(CELLS)

    def open_soma(census_version: str, tiledb_config: Any = None) -> Any:
        configs.append(tiledb_config)
        return {"census_data": {"homo_sapiens": {"obs": frame}}}

    monkeypatch.setattr(soma_layout, "_RESOLVED", {})
    layout = soma_layout.SomaLayout()
    layout.module = types.SimpleNamespace(open_soma=open_soma,
                                          get_census_version_description=lambda v: {"release_build": "2025-11-08"})
    spec = LayoutSpec(table="c.obs", path=None, options={"uri": "census_data/homo_sapiens/obs",
                                                          "tiledb_config": {"soma.init_buffer_bytes": 1}})
    got = layout.columns(spec, predicate=Eq("dataset_id", "dB"), columns=["soma_joinid", "donor_id"])
    assert got["soma_joinid"] == list(range(10, 20)) and len(got["donor_id"]) == 10
    assert configs == [{"soma.init_buffer_bytes": 1}]
    assert soma_layout.count_rows(iter([types.SimpleNamespace(num_rows=2), [1, 2, 3]])) == 5


# --------------------------------------------------------------------------- survival through the client


def test_survival_expression_download_goes_through_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """fetch_cbioportal_expression_and_clinical reads cbioportal.molecular_data, patient_clinical and
    sample_clinical through vbt.datalayer.client (three provenance ids, no REST fallback)."""
    pytest.importorskip("httpx")
    from vbt.analysis import survival
    from vbt.datalayer import client

    calls: list[tuple[str, dict[str, Any]]] = []
    tables = {
        "cbioportal.molecular_data": [{"sampleId": "TCGA-AA-0001-01", "patientId": "TCGA-AA-0001", "value": 15.0},
                                      {"sampleId": "TCGA-AA-0002-01", "patientId": "TCGA-AA-0002", "value": 3.0}],
        "cbioportal.patient_clinical": [{"studyId": "s", "patientId": "TCGA-AA-0001", "OS_MONTHS": "10"},
                                        {"studyId": "s", "patientId": "TCGA-AA-0002", "OS_MONTHS": "20"}],
        "cbioportal.sample_clinical": [{"studyId": "s", "sampleId": "TCGA-AA-0001-01", "patientId": "TCGA-AA-0001",
                                        "SAMPLE_TYPE": "Primary"}],
    }

    def find(table: str, where: dict[str, Any] | None = None, **kw: Any) -> Any:
        calls.append((table, dict(where or {})))
        return client.Result(rows=tables[table], header={"status": "ok", "truncated": False},
                             prov=f"dp_{len(calls)}")

    monkeypatch.setattr(client, "find", find)
    out = survival.fetch_cbioportal_expression_and_clinical("s", "CD276")
    assert calls[0] == ("cbioportal.molecular_data", {"molecularProfileId": "s_rna_seq_v2_mrna",
                                                      "sampleListId": "s_all", "entrezGeneId": 80381})
    assert [c[0] for c in calls[1:]] == ["cbioportal.patient_clinical", "cbioportal.sample_clinical"]
    assert out.attrs["vbt_prov"] == ["dp_1", "dp_2", "dp_3"] and "vbt_prov_fallback" not in out.attrs
    assert sorted(out["patientId"]) == ["TCGA-AA-0001", "TCGA-AA-0002"]
    assert out.set_index("patientId").loc["TCGA-AA-0001", "expr"] == pytest.approx(4.0)   # log2(15 + 1)
    assert out.set_index("patientId").loc["TCGA-AA-0001", "OS_MONTHS"] == "10"
    # a truncated read is not used: the expression falls back (here: refused, no network)
    monkeypatch.setattr(client, "find", lambda *a, **k: client.Result(rows=[], header={"status": "partial",
                                                                                      "truncated": True}, prov="dp"))
    assert survival._expression_via_data_client("p", "l", 1) is None
