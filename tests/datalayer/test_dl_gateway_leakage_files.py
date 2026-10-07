"""Leakage ceilings (§6.1 LeakageSpec, §11.3 step 8, T1), returned files (FileCheckSpec, write-once,
``materialized_by``) and SOMA value filters (§11.3 step 4)."""

from __future__ import annotations

import copy
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from test_dl_gateway_flow import call, hdr, make_gateway
from vbt.datalayer.descriptor.overlay import FileCheckSpec
from vbt.datalayer.errors import ErrorKind, GatewayError
from vbt.datalayer.gateway import soma_filter as sf
from vbt.datalayer.gateway.files import (
    MaterializedRegistry,
    positional_index,
    read_h5ad_header,
    reconcile,
    uniquified_names,
    write_once,
)
from vbt.datalayer.gateway.leakage import parse_date, prepare_leakage
from vbt.datalayer.predicate import And, Cmp, Eq, In, Not, Or, evaluate

CT: dict[str, Any] = {
    "schema": "vbt.datasource/1", "source": "ctgov", "title": "ClinicalTrials.gov", "kind": "remote",
    "release": {"from": "as_of"},
    "leakage": {"available_at": "posted", "changed_at": "updated", "rows": "withhold"},
    "tables": {"studies": {"kind": "records", "grain": "one registered study", "format": "none",
                           "key": {"columns": ["nctId"]},
                           "columns": {"nctId": {"role": "identifier"}, "posted": {"role": "time"},
                                       "updated": {"role": "time"}, "status": {"role": "category"}}}},
}
CT_OVERLAY: dict[str, Any] = {
    "schema": "vbt.overlay/1", "server": "clinicaltrials",
    "tools": {
        "search": {"reads": {"ctgov.studies": {"access": "upstream"}},
                   "args": {"condition": {"role": "free_text", "interpreted_as": "engine"},
                            "advanced_filter": {"role": "free_text"}},
                   "leakage_filter": {"arg": "advanced_filter", "template": "AREA[StudyFirstPostDate]RANGE[MIN,{ceiling}]"},
                   "result": {"rows": "$.trials", "row_key": ["nctId"], "total": {"path": "$.total_count"},
                              "order_source": "source_server_side"}},
        "count": {"reads": {"ctgov.studies": {"access": "upstream"}},
                  "args": {"condition": {"role": "free_text", "interpreted_as": "engine"}},
                  "result": {"kind": "count", "rows": None, "total": "$.count"}},
        "get": {"reads": {"ctgov.studies": {"access": "upstream"}},
                "args": {"nct": {"binds": "ctgov.studies.nctId"}},
                "result": {"rows": "$.trials", "row_key": ["nctId"]}},
    },
}
TRIALS = [{"nctId": "NCT01", "posted": "2018-03", "updated": "2019-01-02", "status": "COMPLETED"},
          {"nctId": "NCT02", "posted": "2020-01", "updated": "2020-01-20", "status": "RECRUITING"},
          {"nctId": "NCT03", "posted": "2019", "updated": "2021-05-01", "status": "TERMINATED"},
          {"nctId": "NCT04", "posted": None, "updated": None, "status": "UNKNOWN"}]


def ct(tmp_path, ceiling="2020-01-15", overlay=CT_OVERLAY):
    return make_gateway(tmp_path, [CT], [overlay], {"ctgov.studies": TRIALS},
                        data={"leakage": {"ceiling": ceiling}})


async def test_filter_injected_into_search_and_rows_withheld(tmp_path):
    gw = ct(tmp_path)
    sent: dict[str, Any] = {}
    plan, res = await call(gw, "clinicaltrials", "search", {"condition": "asthma", "advanced_filter": "AREA[Phase]PHASE3"},
                           lambda a: sent.update(a) or {"trials": TRIALS, "total_count": 4})
    assert sent["advanced_filter"] == "AREA[Phase]PHASE3 AND AREA[StudyFirstPostDate]RANGE[MIN,2020-01-15]"
    ids = [t["nctId"] for t in res.obj["trials"]]
    assert ids == ["NCT01", "NCT04"]                 # NCT02 posted Jan 2020 (= Jan 31), NCT03 changed after
    h = hdr(res)
    assert h["withheld"] == {"leakage": 2}
    # NCT04 has no dates: kept, but the record says the ceiling could not be checked
    assert res.provenance.leakage == {"ceiling": "2020-01-15", "withheld": 2, "risk": True}
    assert h["total_method"] == "upstream_upper_bound" and h.get("order") is None


async def test_counting_tool_without_filter_is_quarantined(tmp_path):
    gw = ct(tmp_path)
    with pytest.raises(GatewayError) as e:
        await call(gw, "clinicaltrials", "count", {"condition": "asthma"}, lambda a: {"count": 3})
    assert e.value.kind == ErrorKind.quarantined and e.value.subkind == "leakage"
    gw = ct(tmp_path / "noceiling", ceiling=None)
    plan, res = await call(gw, "clinicaltrials", "count", {"condition": "asthma"}, lambda a: {"count": 3})
    assert res.provenance.leakage is None
    assert hdr(res)["status"] == "ok" and hdr(res)["total"] == 3 and hdr(res)["total_method"] == "upstream"


def test_prepare_leakage_stamp_and_partial_dates():
    assert parse_date("2004-01") == date(2004, 1, 31) and parse_date("2004-01", "earliest") == date(2004, 1, 1)
    assert parse_date("2004") == date(2004, 12, 31) and parse_date("March 2010") == date(2010, 3, 31)
    assert parse_date("March 5, 2010") == date(2010, 3, 5) and parse_date("2021-02-03T10:00:00Z") == date(2021, 2, 3)
    assert parse_date("n/a") is None and parse_date(None) is None


# --------------------------------------------------------------------------- files


def _h5ad(path: Path, obs_index: list[str], var_index: list[str], obs_cols: list[str]) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as f:
        obs = f.create_group("obs")
        obs.attrs["_index"] = "_index"
        obs.attrs["column-order"] = obs_cols
        obs.create_dataset("_index", data=[s.encode() for s in obs_index])
        for c in obs_cols:
            obs.create_dataset(c, data=[b"x"] * len(obs_index))
        var = f.create_group("var")
        var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=[s.encode() for s in var_index])


def test_file_checks_echo_key_columns_and_var_index(tmp_path):
    _h5ad(tmp_path / "ok.h5ad", ["c1", "c2", "c3"], ["PCSK9", "TP53"], ["dataset_id", "donor_id"])
    spec = FileCheckSpec(path_from="$.output_path", echo_checks={"n_obs": "$.n_cells"},
                         key_columns=["dataset_id", "donor_id"], forbid_positional_index=True)
    checks = reconcile([spec], {"output_path": "ok.h5ad", "n_cells": 3}, output_dir=tmp_path)
    assert all(c.ok for c in checks), checks
    checks = {c.name: c for c in reconcile([spec], {"output_path": "ok.h5ad", "n_cells": 5}, output_dir=tmp_path)}
    assert checks["file_echo:n_obs"].ok is False
    _h5ad(tmp_path / "bad.h5ad", ["c1"], ["0", "1", "2"], ["dataset_id"])
    checks = {c.name: c for c in reconcile([spec], {"output_path": "bad.h5ad", "n_cells": 1}, output_dir=tmp_path)}
    assert checks["file_key_columns"].ok is False and checks["file_var_index"].ok is False
    checks = reconcile([spec], {"output_path": "missing.h5ad"}, output_dir=tmp_path)
    assert checks[0].name == "file_exists" and checks[0].ok is False
    assert read_h5ad_header(tmp_path / "ok.h5ad")["n_vars"] == 2
    assert uniquified_names(["A", "A-1", "B-2"]) == ["A-1"] and positional_index(["0", "1"])
    checks = reconcile([spec], {"output_path": "x.bin"}, output_dir=tmp_path,
                       header_reader=lambda p: None)
    assert checks[0].ok is False                                   # does not exist
    (tmp_path / "x.bin").write_bytes(b"?")
    checks = {c.name: c for c in reconcile([spec], {"output_path": "x.bin"}, output_dir=tmp_path,
                                           header_reader=lambda p: None)}
    assert checks["file_header"].ok is None                        # an unmade check is never a pass


def test_write_once_and_registry(tmp_path):
    p = tmp_path / "out.h5ad"
    assert write_once(p) is None
    p.write_bytes(b"one")
    moved = write_once(p, token="dp_abc")
    assert moved.name == "out.dp_abc.h5ad" and not p.exists()
    p.write_bytes(b"two")
    assert write_once(p, token="dp_abc").name == "out.dp_abc-1.h5ad"
    reg = MaterializedRegistry()
    e = reg.register("census.cells", moved, prov="dp_1", tool="mcp__single_cell__get_anndata")
    assert e["sha256"] and reg.lookup(moved)["prov"] == "dp_1" and reg.for_table("census.cells") == [e]


CENSUS: dict[str, Any] = {
    "schema": "vbt.datasource/1", "source": "census", "title": "CELLxGENE Census", "kind": "remote",
    "release": {"from": "as_of"},
    "tables": {
        "obs": {"kind": "records", "grain": "one cell", "format": "none", "key": {"columns": ["soma_joinid"]},
                "columns": {"soma_joinid": {"role": "identifier"},
                            "donor_id": {"role": "category", "unique_within": ["dataset_id"]},
                            "dataset_id": {"role": "category"},
                            "is_primary_data": {"role": "qualifier", "effect": "duplicate", "default_filter": True}}},
        "cells_file": {"kind": "records", "grain": "one cell in a written file", "format": "none",
                       "key": {"columns": ["soma_joinid"]}, "columns": {"soma_joinid": {"role": "identifier"}},
                       "materialized_by": {"tool": "single_cell.get_anndata", "path_from": "$.output_path"}},
    },
}
SC_OVERLAY: dict[str, Any] = {
    "schema": "vbt.overlay/1", "server": "single_cell",
    "tools": {
        "get_anndata": {"reads": {"census.obs": {"access": "remote"}},
                        "args": {"value_filter": {"role": "free_text", "interpreted_as": "engine", "escape": "soma"},
                                 "output_path": {"role": "output_path"}},
                        "result": {"kind": "file", "rows": None,
                                   "files": [{"path_from": "$.output_path", "echo_checks": {"n_obs": "$.n_cells"}}]}},
        "donor_balanced": {"reads": {"census.obs": {"access": "remote", "columns": ["donor_id"]}},
                           "args": {"value_filter": {"role": "free_text", "interpreted_as": "engine",
                                                     "escape": "soma"}},
                           "result": {"rows": "$.cells"}},
    },
}
VOCAB = {"tissue_general": ["lung", "liver"], "disease": ["normal", "COVID-19"], "dataset_id": ["d1", "d2"],
         "is_primary_data": [True, False]}


def census(tmp_path):
    gw = make_gateway(tmp_path, [CENSUS], [SC_OVERLAY], {})
    gw.bridge.upstream = lambda tool, args: {"value_counts": [{"value": v, "count": 1}
                                                              for v in VOCAB.get(args.get("column_name"), [])]}
    return gw


async def test_soma_filter_resolved_and_primary_data_enforced(tmp_path):
    gw = census(tmp_path)
    sent: dict[str, Any] = {}

    def upstream(args: dict[str, Any], n: int = 2) -> dict[str, Any]:     # writes the file it reports
        sent.update(args)
        _h5ad(tmp_path / "out" / "cells.h5ad", ["a", "b"], ["G1"], ["dataset_id"])
        return {"output_path": "cells.h5ad", "n_cells": n}

    plan, res = await call(gw, "single_cell", "get_anndata",
                           {"value_filter": "tissue_general == 'Lung' and disease in ['covid-19']",
                            "output_path": "cells.h5ad"}, upstream)
    assert sent["value_filter"] == ("tissue_general == 'lung' and disease in ['COVID-19'] and "
                                    "is_primary_data == True")
    assert any("casefold" in n for n in hdr(res)["notes"]) and hdr(res)["status"] == "ok"
    assert {c.name: c.ok for c in res.provenance.checks}["file_echo:n_obs"] is True
    entry = gw.materialized.for_table("census.cells_file")[0]
    assert entry["path"].endswith("cells.h5ad") and entry["prov"] == res.provenance.id and entry["sha256"]
    with pytest.raises(GatewayError) as e:                       # n_obs disagrees with the payload
        await call(gw, "single_cell", "get_anndata", {"value_filter": "disease == 'normal'"},
                   lambda a: upstream(a, 7))
    assert e.value.kind == ErrorKind.tool_defect


async def test_soma_filter_errors_and_donor_needs_dataset(tmp_path):
    gw = census(tmp_path)
    for bad in ("tissue_general = 'lung'", "tissue_general == 'lung' and", "disease in ['x'"):
        with pytest.raises(GatewayError) as e:
            await gw.prepare("single_cell", "get_anndata", {"value_filter": bad}, None)
        assert e.value.kind == ErrorKind.invalid_argument
    with pytest.raises(GatewayError) as e:
        await gw.prepare("single_cell", "get_anndata", {"value_filter": "tissue_general == 'brain'"}, None)
    assert e.value.envelope()["valid_values"] == ["lung", "liver"]
    with pytest.raises(GatewayError) as e:
        await gw.prepare("single_cell", "donor_balanced", {"value_filter": "disease == 'normal'"}, None)
    assert e.value.kind == ErrorKind.incomplete_key and e.value.envelope()["dimension"] == "dataset_id"
    plan = await gw.prepare("single_cell", "donor_balanced", {"value_filter": "dataset_id == 'd1'"}, None)
    assert plan.args_sent["value_filter"] == "dataset_id == 'd1' and is_primary_data == True"
    plan = await gw.prepare("single_cell", "donor_balanced",
                            {"value_filter": "dataset_id == 'd1'", "include_duplicates": True}, None)
    assert plan.args_sent["value_filter"] == "dataset_id == 'd1'"


def test_soma_parser_round_trips():
    cases = {
        "a == 'x'": Eq("a", "x"),
        "a in ['x', 'y'] and b != 3": And((In("a", ("x", "y")), Cmp("b", "!=", 3))),
        "not (a == 'x' or b < 2.5)": Not(Or((Eq("a", "x"), Cmp("b", "<", 2.5)))),
        "a not in ['x']": Not(In("a", ("x",))),
        "is_primary_data == True": Eq("is_primary_data", True),
        "name == 'O\\'Brien'": Eq("name", "O'Brien"),
    }
    for text, want in cases.items():
        got = sf.parse(text)
        assert got == want, text
        again = sf.parse(sf.compile_filter(got))
        assert again == got, text
    p = sf.parse("tissue == 'lung' and (disease == 'a' or disease == 'b')")
    assert evaluate(p, {"tissue": "lung", "disease": "b"}) is True
    assert sf.compile_filter(p) == "tissue == 'lung' and (disease == 'a' or disease == 'b')"
    assert sf.fixes_single(sf.parse("dataset_id in ['d1'] and x == 1"), "dataset_id")
    assert not sf.fixes_single(sf.parse("dataset_id in ['d1', 'd2']"), "dataset_id")
    assert not sf.fixes_single(sf.parse("dataset_id == 'd1' or x == 1"), "dataset_id")
    pred, added = sf.enforce_primary(sf.parse("a == 1"))
    assert added and sf.compile_filter(pred) == "a == 1 and is_primary_data == True"
    pred, added = sf.enforce_primary(sf.parse("is_primary_data == False"))
    assert not added
    with pytest.raises(sf.SomaFilterError):           # two casefold matches: no guess
        sf.resolve_filter(sf.parse("disease == 'Normal'"), {"disease": ["normal", "NORMAL"]})
    resolved, notes = sf.resolve_filter(sf.parse("disease == 'Normal' and n > 2"), {"disease": ["normal"], "n": None})
    assert resolved == And((Eq("disease", "normal"), Cmp("n", ">", 2))) and notes


def test_prepare_leakage_contract_object():
    from types import SimpleNamespace
    contract = SimpleNamespace(tables={}, binding=None)
    plan = prepare_leakage(contract, {}, date(2020, 1, 1))
    assert not plan.active and plan.injected is None
    gw_copy = copy.deepcopy(CT_OVERLAY)
    assert gw_copy["tools"]["search"]["leakage_filter"]["arg"] == "advanced_filter"
