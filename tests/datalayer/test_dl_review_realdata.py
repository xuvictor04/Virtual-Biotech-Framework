"""Regressions for the findings of the review on real data (Open Targets 25.09, the live APIs); each test is
offline and names its finding.

* RV-OT-01: a derived serve over its scan budget is ``too_large``, never an empty answer.
* RV-OT-02: a tool that writes its rows to a file is never re-called or inflated (the re-call overwrote the agent's
  file with every match); a declared ``preview`` is no short page; a re-call records the limit it sent.
* RV-OT-03: a list-form count field counts the rows upstream listed (checked statically against the upstream
  source, and at run time: a field that counts something else is left as returned); ``num_high_quality`` is
  recomputed as the probes with ``isHighQuality``.
* RV-OT-05: a record's list is trimmed once, by the overlay's cap (``$.descendants`` and ``descendants`` are one
  column), with the stored length as the total and ``order: key`` applied to a list of ids.
* RV-OT-07: grains count list elements, never null, and a derived result takes the data child's returned count.
* LIVE-1: a record withheld by the evidence ceiling leaves nothing of the upstream payload in the text.
* LIVE-4: a native data tool's provenance names its table, key, source, release and record versions.
* LIVE-10: a set argument's unknown members are ``not_found_items`` (status partial), all of them ``not_found``.
"""

from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from dl_upstream import REPO, upstream_root
from test_dl_gateway_flow import (
    DRUG_OVERLAY,
    KNOWN_DRUG,
    OT,
    FakeService,
    call,
    hdr,
    make_gateway,
    world,
)
from vbt.datalayer.descriptor.overlay import TrimSpec
from vbt.datalayer.errors import ErrorKind, GatewayError
from vbt.datalayer.gateway.fields import grain_values
from vbt.datalayer.gateway.transforms import Counters, t11_counts, t12_trim
from vbt.datalayer.ipc import ServeResponse

OVERLAYS = REPO / "configs" / "data" / "overlays"


# --------------------------------------------------------------------------- RV-OT-01


async def test_a_serve_over_its_scan_budget_is_too_large_not_empty(tmp_path, monkeypatch):
    """RV-OT-01: the child answers ``rows: [], total: null, reason: too_large: ...``; the gateway used to place
    the empty list and finish it as status ``empty``."""
    gw = world(tmp_path)

    def over(self: FakeService, req: Any) -> ServeResponse:
        return ServeResponse(rows=[], total=None, truncated=True,
                             reason="too_large: reading open_targets.known_drug needs ~759037123 decoded bytes")

    monkeypatch.setattr(FakeService, "_serve", over)
    with pytest.raises(GatewayError) as e:
        await call(gw, "drug", "search_known_drugs_derived", {"target_id": "PCSK9", "limit": 5})
    assert e.value.kind == ErrorKind.too_large and e.value.subkind == "scan_budget"
    assert "759037123" in e.value.message and e.value.tool == "mcp__drug__search_known_drugs_derived"


async def test_a_real_serve_over_a_one_byte_budget_is_too_large(ot_root, tmp_path, monkeypatch):
    """RV-OT-01 through the real data child on the OT fixture: interaction_evidence under a 1-byte repair budget."""
    from test_dl_native_tools import InProcessBridge
    from vbt.datalayer.gateway import DataGateway
    from vbt.datalayer.service import ServiceContext
    from vbt.datalayer.settings import DataSettings

    monkeypatch.setenv("OPEN_TARGETS_DATA_PATH", str(ot_root))

    async def evidence(budget: int) -> Any:
        settings = DataSettings.from_dict({"descriptors_dir": str(REPO / "configs" / "data" / "sources"),
                                           "overlays_dir": str(OVERLAYS), "cache_dir": str(tmp_path / f"c{budget}"),
                                           "witness": {"repair_max_bytes": budget}}, project_root=REPO)
        ctx = ServiceContext(settings)
        gw = DataGateway(ctx.settings, ctx.catalog, ctx.registry, run={"mcp_output_dir": str(tmp_path)})
        gw.bind_bridge(InProcessBridge(ctx))
        plan = await gw.prepare("interaction", "get_interaction_evidence", {"target_id": "PCSK9", "limit": 5}, None)
        assert plan.route == "derived"
        return await gw.finish(plan, None)

    res = await evidence(500_000_000)
    assert res.header["status"] in ("ok", "partial") and res.header["total"] > 0
    with pytest.raises(GatewayError) as e:
        await evidence(1)
    assert e.value.kind == ErrorKind.too_large and "scan budget" in e.value.message


# --------------------------------------------------------------------------- RV-OT-02


def _writer_overlay(preview: int | None = None) -> dict[str, Any]:
    ov = copy.deepcopy(DRUG_OVERLAY)
    tool = copy.deepcopy(ov["tools"]["search_known_drugs"])
    tool["args"]["output_path"] = {"role": "output_path"}
    tool["result"] = {"rows": "$.drugs", "order": [{"column": "phase", "direction": "desc"}],
                      "order_source": "upstream_full_sort", "arg_echo": {"limit": "$.limit"}}
    if preview is not None:
        tool["result"]["preview"] = preview
    ov["tools"]["write_drugs"] = tool
    return ov


def _writer(written: list[int]) -> Any:
    """Upstream: sorts every match, writes ``limit`` rows to the file and lists head(3) (association_mcp)."""
    def tool(args: dict[str, Any]) -> Any:
        rows = sorted([r for r in KNOWN_DRUG if r["targetId"] == args.get("target_id")],
                      key=lambda r: -(r["phase"] if r["phase"] is not None else -9))[: int(args.get("limit", 20))]
        written.append(len(rows))
        return {"success": True, "output_path": args.get("output_path") or "auto.parquet", "count": len(rows),
                "limit": args.get("limit"), "drugs": rows[:3]}
    return tool


@pytest.mark.parametrize("output_path", ["drugs.parquet", None])
async def test_a_tool_that_writes_its_rows_is_never_recalled(tmp_path, output_path):
    """RV-OT-02: 3 listed of 6 is a W6 short page; the re-call overwrote the agent's file with every match (992
    rows for PCSK9 limit 100). Upstream names a file itself when the agent gives none, so neither is re-called."""
    gw = world(tmp_path, overlay=_writer_overlay())
    written: list[int] = []
    args = {"target_id": "PCSK9", "limit": 6, **({"output_path": output_path} if output_path else {})}
    plan, res = await call(gw, "drug", "write_drugs", args, _writer(written))
    assert written == [6] and gw.bridge.calls == []
    assert plan.args_sent["limit"] == 6 and "short_page_recall" not in " ".join(res.provenance.result.transforms)
    assert hdr(res)["status"] == "partial" and "short page: 3 of 6" in " ".join(hdr(res)["notes"])


async def test_a_declared_preview_is_no_short_page(tmp_path):
    """RV-OT-02: ``preview: 3`` (head(3) of what the tool wrote) is the whole listing, not a short page."""
    gw = world(tmp_path, overlay=_writer_overlay(preview=3))
    written: list[int] = []
    plan, res = await call(gw, "drug", "write_drugs", {"target_id": "PCSK9", "limit": 6, "output_path": "d.parquet"},
                           _writer(written))
    h = hdr(res)
    assert written == [6] and gw.bridge.calls == []
    assert not any("short page" in n for n in h["notes"])
    assert h["returned"] == 3 and h["total"] == 8 and h["status"] == "partial"
    assert res.obj["count"] == 6 and res.obj["limit"] == 6        # the rows written, as upstream reported them


async def test_a_recall_records_the_limit_it_sent(tmp_path):
    """RV-OT-02: the provenance's args_sent named the agent's limit after a re-call sent another one."""
    from test_dl_gateway_flow import sorted_first

    gw = world(tmp_path)
    calls: list[dict[str, Any]] = []

    def tool(args: dict[str, Any]) -> Any:     # half a page the first time
        calls.append(dict(args))
        out = sorted_first(args)
        if len(calls) == 1:
            out["drugs"] = out["drugs"][:2]
        return out

    plan, res = await call(gw, "drug", "search_known_drugs", {"target_id": "PCSK9", "limit": 20}, tool)
    assert len(calls) == 2 and calls[1]["limit"] != 20
    assert res.provenance.request.args_sent["limit"] == calls[1]["limit"]
    assert res.provenance.request.args_raw["limit"] == 20 and res.obj["limit"] == 20


# --------------------------------------------------------------------------- RV-OT-03


def test_a_count_field_that_counts_something_else_is_left_as_returned():
    """RV-OT-03: T11 set every list-form count field to the rows listed: num_high_quality 9 became 228 and an
    association count of 50 written rows became the 10 previewed."""
    c = Counters()
    obj = {"num_high_quality": 9, "count": 4, "rows": [{"x": i} for i in range(3)]}
    t11_counts(obj, obj["rows"], count_fields=["$.num_high_quality", "$.count"], counters=c, upstream_returned=4)
    assert obj["num_high_quality"] == 9 and obj["count"] == 3
    assert any("num_high_quality (9) is not the number of rows upstream listed (4)" in n for n in c.notes)
    # recomputed as a summary over the full list instead
    probes = [{"isHighQuality": v} for v in (True, False, None, True)]
    obj = {"num_high_quality": 228}
    from vbt.datalayer.descriptor.overlay import RecomputeSpec
    t11_counts(obj, probes, summary_fields={"$.num_high_quality": RecomputeSpec(agg="count_true", of="isHighQuality")},
               full_rows=probes, counters=Counters())
    assert obj["num_high_quality"] == 2


def _upstream_functions(server: str) -> dict[str, ast.FunctionDef]:
    out: dict[str, ast.FunctionDef] = {}
    for f in sorted((upstream_root() / "src" / "mcp_servers" / f"{server}_mcp").glob("*.py")):
        for n in ast.walk(ast.parse(f.read_text())):
            if isinstance(n, ast.FunctionDef):
                out.setdefault(n.name, n)
    return out


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def test_list_count_fields_count_the_rows_upstream_lists():
    """RV-OT-03: in every dict an upstream pass tool returns with a list-form count field and its rows, the count is
    ``len(x)`` of the variable the rows are built from (a preview's ``top_10`` built from ``matches`` with
    ``len(matches)`` is refused unless the binding declares the preview and leaves the count to upstream)."""
    if not (upstream_root() / "src" / "mcp_servers").is_dir():
        pytest.skip("the upstream checkout is not here")
    bad: list[str] = []
    checked = 0
    for f in sorted(OVERLAYS.glob("*.yaml")):
        ov = yaml.safe_load(f.read_text())
        server = ov["server"]
        if not (upstream_root() / "src" / "mcp_servers" / f"{server}_mcp").is_dir():
            continue
        fns = _upstream_functions(server)
        for tool, b in (ov.get("tools") or {}).items():
            r = b.get("result") or {}
            fields = r.get("count_fields")
            if not isinstance(fields, list) or b.get("serve", "pass") != "pass" or tool not in fns:
                continue
            rows = r.get("rows")
            rows = [rows] if isinstance(rows, str) else list(rows or [])
            preview = r.get("preview")
            for path in rows:
                if isinstance(preview, int) or (isinstance(preview, dict) and path in preview):
                    continue
                rkey = path[2:].split(".")[0].split("[")[0]
                for field in fields:
                    ckey = field[2:]
                    for d in (n for n in ast.walk(fns[tool]) if isinstance(n, ast.Dict)):
                        keys = [k.value if isinstance(k, ast.Constant) else None for k in d.keys]
                        if ckey not in keys or rkey not in keys:
                            continue
                        cv, rv = d.values[keys.index(ckey)], d.values[keys.index(rkey)]
                        if isinstance(cv, ast.Constant):
                            continue                   # an empty answer's literal 0
                        checked += 1
                        ok = (isinstance(cv, ast.Call) and getattr(cv.func, "id", None) == "len" and cv.args
                              and _names(cv.args[0]) and _names(cv.args[0]) <= _names(rv)
                              and ".head(" not in ast.unparse(rv))
                        if not ok:
                            bad.append(f"{server}.{tool}: {ckey} = {ast.unparse(cv)} but {rkey} = {ast.unparse(rv)}")
    assert checked > 15, checked
    assert not bad, "\n".join(bad)


def test_shipped_preview_tools_leave_the_written_count_to_upstream():
    ov = yaml.safe_load((OVERLAYS / "association.yaml").read_text())
    for tool in ("query_associations", "get_associations_for_disease", "get_associations_for_target",
                 "filter_by_datatype", "filter_by_datasource"):
        r = ov["tools"][tool]["result"]
        assert r["preview"] == 10 and "count_fields" not in r, tool
    probes = yaml.safe_load((OVERLAYS / "target.yaml").read_text())["tools"]["get_chemical_probes"]["result"]
    assert "count_fields" not in probes
    assert probes["summary_fields"]["$.num_high_quality"] == {"recompute": {"agg": "count_true", "of": "isHighQuality"}}


# --------------------------------------------------------------------------- RV-OT-05


def test_a_list_is_trimmed_once_with_its_stored_length_as_total():
    """RV-OT-05: ``$.descendants`` (overlay, 100, key) and ``descendants`` (relation_list_max, 50) were both applied:
    ``{returned 50, total 100}`` for 18,649 stored, in storage order."""
    stored = [f"EFO_{i:07d}" for i in range(1000, 0, -1)]
    row = {"descendants": list(stored)}
    c = Counters()
    t12_trim([row], {"descendants": TrimSpec(max=100, order="key")}, c)
    assert c.trimmed == {"descendants": {"returned": 100, "total": 1000}}
    assert row["descendants"] == sorted(stored)[:100]
    # the same column under two spellings is trimmed by the first entry only
    row = {"descendants": list(stored)}
    c = Counters()
    t12_trim([row], {"$.descendants": TrimSpec(max=100, order="key"), "descendants": 50}, c)
    assert c.trimmed == {"descendants": {"returned": 100, "total": 1000}} and len(row["descendants"]) == 100


async def test_a_record_list_is_trimmed_by_the_overlay_cap_through_the_gateway(tmp_path):
    desc = copy.deepcopy(OT)
    desc["tables"]["disease"] = {
        "kind": "entity", "grain": "one disease", "key": {"columns": ["id"]},
        "columns": {"id": {"role": "identifier"},
                    "descendants": {"role": "hierarchy", "relation": "descendant", "of": "id"},
                    "children": {"role": "hierarchy", "relation": "child", "of": "id"}}}
    ov = {"schema": "vbt.overlay/1", "server": "disease", "sources": ["open_targets"], "tools": {"info": {
        "reads": {"open_targets.disease": {"access": "full_table"}},
        "args": {"disease_id": {"binds": "open_targets.disease.id"}},
        "result": {"kind": "record", "rows": "$", "trim": {"$.descendants": {"max": 60, "order": "key"}}}}}}
    stored = [f"D{i:04d}" for i in range(200, 0, -1)]
    gw = make_gateway(tmp_path, [desc], [ov], {"open_targets.disease": [{"id": "X1"}]},
                      data={"results": {"relation_list_max": 50}})
    plan, res = await call(gw, "disease", "info", {"disease_id": "X1"},
                           lambda a: {"id": "X1", "descendants": list(stored), "children": list(stored[:70])})
    h = hdr(res)
    assert h["trimmed"] == {"descendants": {"returned": 60, "total": 200}, "children": {"returned": 50, "total": 70}}
    assert res.obj["descendants"] == sorted(stored)[:60]


def _shipped_catalog() -> Any:
    from vbt.datalayer.catalog import Catalog
    from vbt.datalayer.descriptor.load import load_descriptors, load_overlays
    from vbt.datalayer.plugins.registry import discover

    variables = {"project_root": str(REPO)}
    overlays, generic = load_overlays(OVERLAYS, variables)
    return Catalog(load_descriptors(REPO / "configs" / "data" / "sources", variables), overlays, generic,
                   registry=discover())


def test_depth_order_on_a_list_of_ids_is_a_lint_error():
    from vbt.datalayer.descriptor.lint import lint_overlay
    from vbt.datalayer.descriptor.overlay import Overlay

    cat = _shipped_catalog()
    ov = yaml.safe_load((OVERLAYS / "disease.yaml").read_text())
    ov["tools"]["get_disease_info"]["result"]["trim"]["$.ancestors"] = {"max": 100, "order": "depth"}
    found = [f for f in lint_overlay(Overlay.model_validate(ov), cat) if f.level == "error"]
    assert any("order: depth" in f.message and "ancestors" in f.where for f in found), found


# --------------------------------------------------------------------------- RV-OT-07


def test_grain_values_count_list_elements_and_never_null():
    """RV-OT-07: ``drugs[].drugId`` counted distinct lists (null lists and null elements included)."""
    rows = [{"drugs": [{"drugId": "CHEMBL1"}, {"drugId": None}]}, {"drugs": [{"drugId": "CHEMBL1"}]},
            {"drugs": [{"drugId": "CHEMBL2"}]}, {"drugs": []}, {"drugs": None}]
    assert len({v for r in rows for v in grain_values(r, ["drugs[].drugId"])}) == 2
    assert grain_values({"variantRsId": None}, ["variantRsId"]) == []
    assert grain_values({"drugId": "CHEMBL9"}, ["drugs[].drugId"]) != []       # an item row carries the field


def test_the_reader_counts_list_grains_by_element_without_null(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from test_dl_service_reader import make_ctx, source

    rows = [{"id": "a", "rs": "rs1", "drugs": [{"drugId": "C1"}, {"drugId": None}]},
            {"id": "b", "rs": None, "drugs": [{"drugId": "C1"}]},
            {"id": "c", "rs": "rs2", "drugs": [{"drugId": "C2"}, {"drugId": "C3"}]},
            {"id": "d", "rs": None, "drugs": None}]
    d = tmp_path / "s" / "pgx"
    d.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), d / "part-0.parquet")
    spec = {"kind": "fact", "path": "pgx", "grain": "one annotation", "key": {"columns": ["id"]},
            "grains": {"drug": ["drugs[].drugId"], "variant": ["rs"]},
            "columns": {"id": {"role": "identifier"}, "rs": {"role": "identifier"},
                        "drugs": {"role": "nested", "item_key": {"identity": "position"},
                                  "fields": {"drugId": {"role": "identifier"}}}}}
    ctx = make_ctx(tmp_path, source("s", tmp_path / "s", {"pgx": spec}))
    reader = ctx.reader("s.pgx")
    assert reader.distinct_counts(None, {"drug": ["drugs[].drugId"], "variant": ["rs"]}) == {"drug": 3, "variant": 2}
    from vbt.datalayer.service.reader import ScanStats
    st = ScanStats()
    reader.rows(None, [], [], 2, None, stats=st, grains={"drug": ["drugs[].drugId"], "variant": ["rs"]})
    assert st.grains_returned == {"drug": 1, "variant": 1}


async def test_derived_grains_take_the_childs_returned_count(tmp_path, monkeypatch):
    """RV-OT-07: the gateway recounted output rows (2 Lipitor rows of one parent family read as 2 parents)."""
    gw = world(tmp_path)
    real = FakeService._serve

    def counted(self: FakeService, req: Any) -> ServeResponse:
        resp = real(self, req)
        return resp.model_copy(update={"grains": {"drug": {"returned": 1, "total": 7}}})

    monkeypatch.setattr(FakeService, "_serve", counted)
    plan, res = await call(gw, "drug", "search_known_drugs_derived", {"target_id": "PCSK9", "limit": 3})
    assert hdr(res)["grains"]["drug"] == {"returned": 1, "total": 7}


# --------------------------------------------------------------------------- LIVE-1


async def test_a_withheld_record_leaves_nothing_of_the_payload(tmp_path):
    """LIVE-1: the header said withheld {leakage: 1} and status empty while the text held the whole post-ceiling
    record (overallStatus, completion date, results)."""
    from test_dl_gateway_leakage_files import CT, CT_OVERLAY, TRIALS

    ov = copy.deepcopy(CT_OVERLAY)
    ov["tools"]["detail"] = {"reads": {"ctgov.studies": {"access": "upstream"}},
                             "args": {"nct": {"binds": "ctgov.studies.nctId"}},
                             "result": {"kind": "record", "rows": "$", "row_key": ["nctId"]}}
    gw = make_gateway(tmp_path, [CT], [ov], {"ctgov.studies": TRIALS}, data={"leakage": {"ceiling": "2020-01-15"}})
    secret = {**TRIALS[1], "secret": "POST_CEILING_FACT"}
    plan, res = await call(gw, "clinicaltrials", "detail", {"nct": "NCT02"}, lambda a: secret)
    h = hdr(res)
    assert h["withheld"] == {"leakage": 1} and h["returned"] == 0
    assert "POST_CEILING_FACT" not in res.text and "RECRUITING" not in res.text
    assert res.obj in ({}, None) or "secret" not in json.dumps(res.obj)


# --------------------------------------------------------------------------- LIVE-4


async def test_native_tool_provenance_names_its_table_key_source_and_record_versions(tmp_path):
    """LIVE-4: the record of a native live lookup had source, release, tables, key and record versions all empty,
    so a moved record version was never ``source_updated`` on replay."""
    from test_dl_gateway_leakage_files import CT

    data_ov = yaml.safe_load((OVERLAYS / "data.yaml").read_text())
    gw = make_gateway(tmp_path, [CT], [data_ov], {})
    body = {"_vbt": {"status": "ok", "source": "ctgov@2026-10-08T09:00:05", "tables": ["ctgov.studies"],
                     "key": ["nctId"], "returned": 1, "total": 1, "total_method": "remote", "served_by": "derived"},
            "rows": [{"nctId": "NCT04368728", "updated": "2026-03-25"}],
            "record_versions": {"ctgov.studies": {"NCT04368728": "2026-03-25"}}}
    plan, res = await call(gw, "data", "lookup", {"table": "ctgov.studies", "key": {"nctId": "NCT04368728"}},
                           lambda a: body)
    rec = res.provenance
    assert rec.source.name == "ctgov" and rec.source.release == "2026-10-08T09:00:05"
    assert [t.name for t in rec.tables] == ["studies"] and rec.served_by == "derived"
    assert rec.result.key_columns == ["nctId"] and rec.result.row_keys == [["NCT04368728"]]
    assert rec.result.total_method == "remote"
    assert rec.result.record_versions == {"ctgov.studies": {"NCT04368728": "2026-03-25"}}
    from vbt.datalayer.replay import record_versions
    assert record_versions(rec.to_dict()) == {"ctgov.studies": {"NCT04368728": "2026-03-25"}}
    assert hdr(res)["tables"] == ["ctgov.studies"] and hdr(res)["key"] == ["nctId"]


# --------------------------------------------------------------------------- LIVE-10


PUBMED_DESC = {"schema": "vbt.datasource/1", "source": "pubmed", "title": "PubMed", "kind": "remote",
               "release": {"from": "as_of"},
               "id_types": {"pmid": {"plugin": "pmid", "index": "remote"}},
               "tables": {"records": {"kind": "records", "grain": "one PubMed record", "format": "none",
                                      "key": {"columns": ["pmid"]},
                                      "columns": {"pmid": {"role": "identifier", "id_type": "pmid"}}}}}


def _pubmed(tmp_path: Path) -> Any:
    ov = yaml.safe_load((OVERLAYS / "pubmed.yaml").read_text())
    ov["tools"] = {"fetch_abstracts": ov["tools"]["fetch_abstracts"]}
    return make_gateway(tmp_path, [PUBMED_DESC], [ov], {})


async def test_unknown_members_of_a_set_are_not_found_items(tmp_path):
    """LIVE-10: fetch_abstracts(['28304224', '99999999']) was ``ok`` with the unknown PMID only in ``not_returned``."""
    gw = _pubmed(tmp_path)
    plan, res = await call(gw, "pubmed", "fetch_abstracts", {"pmids": ["28304224", "99999999"]},
                           lambda a: {"articles": [{"pmid": "28304224", "title": "t"}], "not_returned": ["99999999"]})
    h = hdr(res)
    assert h["status"] == "partial" and h["not_found_items"] == ["99999999"]
    assert any("99999999" in n for n in h["notes"])
    with pytest.raises(GatewayError) as e:
        await call(gw, "pubmed", "fetch_abstracts", {"pmids": ["99999998", "99999999"]},
                   lambda a: {"articles": [], "not_returned": ["99999998", "99999999"]})
    assert e.value.kind == ErrorKind.not_found and e.value.payload["items"] == ["99999998", "99999999"]
    # a withheld record is accounted for, not missing
    plan, res = await call(gw, "pubmed", "fetch_abstracts", {"pmids": ["28304224", "11111111"]},
                           lambda a: {"articles": [{"pmid": "28304224"}], "withheld": [{"pmid": "11111111"}]})
    assert hdr(res).get("not_found_items") is None


# --------------------------------------------------------------------------- RV-OT-10


def test_search_breaks_class_ties_by_the_declared_order(tmp_path):
    """RV-OT-10: search_targets_by_name declares [match_class asc, approvedSymbol asc]; ties within a class went by
    Ensembl id, so the limit cut MAD1L1, RRM2B, TP53BP1 ... instead of the first symbols."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from test_dl_service_reader import make_ctx, source
    from vbt.datalayer.service.verbs.serve import search

    rows = [{"id": "ENSG1", "approvedSymbol": "TPZ"}, {"id": "ENSG2", "approvedSymbol": "TP53BP1"},
            {"id": "ENSG3", "approvedSymbol": "TP"}, {"id": "ENSG4", "approvedSymbol": "TPA"}]
    d = tmp_path / "s" / "target"
    d.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), d / "part-0.parquet")
    spec = {"kind": "entity", "path": "target", "grain": "one gene", "key": {"columns": ["id"]},
            "grains": {"gene": ["id"]},
            "columns": {"id": {"role": "identifier"}, "approvedSymbol": {"role": "label", "of": "id"}}}
    ctx = make_ctx(tmp_path, source("s", tmp_path / "s", {"target": spec}))
    order = [{"column": "match_class", "direction": "asc"}, {"column": "approvedSymbol", "direction": "asc"}]
    grains: dict[str, Any] = {}
    out, _keys, total = search(ctx.reader("s.target"), "TP", None, 3, {}, None, order=order, grains_out=grains)
    assert total == 4 and [r["approvedSymbol"] for r in out] == ["TP", "TP53BP1", "TPA"]
    assert grains == {"gene": {"returned": 3, "total": 4}}
    out, _keys, _total = search(ctx.reader("s.target"), "TP", None, 3, {}, None)
    assert [r["id"] for r in out] == ["ENSG3", "ENSG1", "ENSG2"]       # without an order: the key breaks ties


async def test_the_header_states_every_order_key(tmp_path):
    ov = copy.deepcopy(DRUG_OVERLAY)
    ov["tools"]["search_known_drugs_derived"]["result"]["order"] = [{"column": "phase", "direction": "desc"},
                                                                    {"column": "drugId", "direction": "asc"}]
    gw = world(tmp_path / "b", overlay=ov)
    plan, res = await call(gw, "drug", "search_known_drugs_derived", {"target_id": "PCSK9", "limit": 3})
    assert hdr(res)["order"] == "phase desc, then drugId asc (verified)"


# --------------------------------------------------------------------------- RV-OT-04


async def test_rows_stored_under_another_family_member_are_counted(tmp_path):
    """RV-OT-04: amifampridine resolved to the parent CHEMBL354077, whose 61 adverse-event rows are stored under the
    salt CHEMBL3301611 only: status empty, total 0. ``family: exact`` now counts them (``_vbt.family_rows``) and
    the answer is partial."""
    from test_dl_gateway_flow import KEYS, REGISTRY, TABLES, index_rows
    from vbt.datalayer.resolve import Entry

    c = REGISTRY.get("identifier", "chembl_molecule").label_key
    idx = index_rows()
    idx["open_targets:chembl_molecule"] = [e for e in idx["open_targets:chembl_molecule"]
                                          if e.canonical != "CHEMBL2"] + [
        Entry(c("CHEMBL50"), "CHEMBL50", "exact", family="CHEMBL50"),
        Entry(c("CHEMBL2"), "CHEMBL2", "exact", family="CHEMBL50")]
    ov = copy.deepcopy(DRUG_OVERLAY)
    ov["tools"]["drug_rows"] = {
        "reads": {"open_targets.known_drug": {"access": "full_table"}},
        "args": {"drug_id": {"binds": "open_targets.known_drug.drugId", "accepts": ["chembl_molecule"]},
                 "limit": {"role": "limit", "min": 1, "max": 50}},
        "result": {"rows": "$.rows", "order": [{"column": "phase", "direction": "desc"}]},
        "serve": "derived", "derived": {"verb": "find", "table": "open_targets.known_drug", "envelope": {"ok": True}}}
    tables = {**TABLES, "open_targets.drug_molecule": TABLES["open_targets.drug_molecule"] + [{"id": "CHEMBL50"}]}
    desc = copy.deepcopy(OT)
    desc["id_types"]["chembl_molecule"]["canonicalize"] = {"parent": "drug_molecule.parentId"}
    desc["tables"]["drug_molecule"]["columns"]["parentId"] = {"role": "identifier", "id_type": "chembl_molecule"}
    gw = make_gateway(tmp_path, [desc], [ov], tables, index_rows=idx, keys=KEYS)
    plan, res = await call(gw, "drug", "drug_rows", {"drug_id": "CHEMBL50", "limit": 5})
    h = hdr(res)
    assert h["returned"] == 0 and h["family_rows"] == {"CHEMBL2": 1} and h["status"] == "partial"
    assert any("CHEMBL2: 1" in n for n in h["notes"])
    # the member itself: its own row, and the parent has none to add
    plan, res = await call(gw, "drug", "drug_rows", {"drug_id": "CHEMBL2", "limit": 5})
    assert hdr(res)["returned"] == 1 and hdr(res).get("family_rows") is None
    # family: include reads every member and counts nothing apart
    ov["tools"]["drug_rows"]["args"]["drug_id"]["family"] = "include"
    gw = make_gateway(tmp_path / "inc", [desc], [ov], tables, index_rows=idx, keys=KEYS)
    plan, res = await call(gw, "drug", "drug_rows", {"drug_id": "CHEMBL50", "limit": 5})
    assert hdr(res)["returned"] == 1 and hdr(res).get("family_rows") is None


# --------------------------------------------------------------------------- RV-OT-06


def test_an_arrow_allocation_failure_is_out_of_memory_not_an_unreadable_file():
    """RV-OT-06: ArrowMemoryError (a MemoryError) was reported as 'not a readable Parquet file', so a data child at its
    memory limit failed every later read as a broken file."""
    pa = pytest.importorskip("pyarrow")
    from vbt.datalayer.plugins.base import Fragment, FormatError
    from vbt.datalayer.plugins.formats.parquet import _unreadable

    frag = Fragment(uri="/data/expression/part-00000.parquet", size=1, mtime_ns=None)
    err = _unreadable(frag, pa.ArrowMemoryError("malloc of size 16777216 failed"))
    assert isinstance(err, MemoryError) and not isinstance(err, FormatError) and "out of memory" in str(err)
    assert isinstance(_unreadable(frag, pa.ArrowInvalid("bad magic")), FormatError)


def test_a_memory_error_ends_the_data_child_with_a_memory_exit(tmp_path):
    """RV-OT-06: the child stayed alive at its RLIMIT_DATA after one over-memory scan. A MemoryError in a verb now
    ends it (exit 70, a last MemoryError line), and the reaper labels the exit memory_error."""
    import subprocess
    import sys as _sys

    from vbt.datalayer.launch import REAPER
    from vbt.datalayer.memory import crash

    code = ("import sys, contextlib\nsys.path.insert(0, sys.argv[1])\n"
            "from vbt.datalayer.service.server import tool_function\n"
            "class Ctx:\n    slots = contextlib.nullcontext()\n"
            "def verb(ctx, payload):\n    raise MemoryError('malloc of size 33554432 failed')\n"
            "tool_function(lambda: Ctx(), '_serve', verb)({})\nprint('unreachable')\n")
    proc = subprocess.run([_sys.executable, "-E", str(REAPER), "--limit-mb", "0", "--status", str(tmp_path / "s.json"),
                           "--server", "data", "--", _sys.executable, "-c", code, str(REPO / "src")],
                          capture_output=True, text=True, timeout=120)
    marker = crash.parse_exit_marker(proc.stderr)
    assert proc.returncode == 70 and "unreachable" not in proc.stdout
    assert marker["code"] == 70 and marker["cause"] == "memory_error" and marker["reason"] == "memory_limit"
    assert "MemoryError: data child out of memory in _serve" in proc.stderr


async def test_a_data_child_memory_exit_is_too_large_for_the_call():
    from vbt.datalayer.gateway.service_client import ServiceClient, ServiceError, ServiceMemoryError
    from vbt.datalayer.ipc import ServeRequest

    class Bridge:
        async def call_raw(self, server, tool, args):
            raise GatewayError(ErrorKind.oom, "data._serve: the server ran out of memory (an allocation failed)")

    client = ServiceClient(Bridge())
    with pytest.raises(ServiceMemoryError) as e:
        await client.serve(ServeRequest(table="open_targets.expression_tissues", verb="find"))
    assert isinstance(e.value, ServiceError) and e.value.kind == ErrorKind.too_large
    assert e.value.subkind == "data_child_memory" and e.value.retryable == "no"


def test_scan_chunks_are_sized_by_the_values_a_row_holds():
    """RV-OT-06: 65,536 rows a chunk converted an expression row group whole (11,082 genes, 1,622 values a row:
    1.8 GB in Python); the chunk is sized by the footer's value counts."""
    from vbt.datalayer.service.reader import SCAN_CHUNK_ROWS, chunk_rows
    from vbt.datalayer.service.sidecar import ChunkInfo, RowGroupInfo

    wide = RowGroupInfo(rows=11082, chunks={"id": ChunkInfo(1, 11082, 0),
                                            "tissues.list.element.label": ChunkInfo(1, 11082 * 1621, 0)})
    n = chunk_rows(wide, ["id", "tissues"])
    assert 400 <= n <= 600 and n * 1622 * 160 <= 128 * 1024 * 1024
    narrow = RowGroupInfo(rows=1_300_000, chunks={"a": ChunkInfo(1, 1_300_000, 0), "b": ChunkInfo(1, 1_300_000, 0)})
    assert chunk_rows(narrow, ["a", "b"]) == SCAN_CHUNK_ROWS and chunk_rows(None, ["a"]) == SCAN_CHUNK_ROWS


# --------------------------------------------------------------------------- RV-OT-08


async def test_an_argument_can_set_the_default_order_upstream_uses(tmp_path):
    """RV-OT-08: prioritize_targets(min_genetic_constraint=-0.5) ranks most constrained first upstream; the derived
    tool cut the 4,496 matches by targetId. ``result.order_when`` gives that argument its order."""
    ov = copy.deepcopy(DRUG_OVERLAY)
    ov["tools"]["ranked"] = {
        "reads": {"open_targets.known_drug": {"access": "full_table"}},
        "args": {"max_phase": {"binds": "open_targets.known_drug.phase", "op": "le"},
                 "sort_by": {"role": "order_by", "values": {"phase": {"column": "phase", "direction": "desc"}}},
                 "limit": {"role": "limit", "min": 1, "max": 50}},
        "result": {"rows": "$.rows", "order_from_arg": "sort_by",
                   "order_when": {"max_phase": [{"column": "phase", "direction": "asc", "nulls": "last"},
                                                {"column": "drugId", "direction": "asc"}]},
                   "order": [{"column": "drugId", "direction": "asc"}]},
        "serve": "derived", "derived": {"verb": "find", "table": "open_targets.known_drug", "envelope": {"ok": True}}}
    gw = world(tmp_path, overlay=ov)
    plan, res = await call(gw, "drug", "ranked", {"max_phase": 3, "limit": 3})
    assert [r["phase"] for r in res.obj["rows"]] == [1, 2, 2] and hdr(res)["order"].startswith("phase asc")
    plan, res = await call(gw, "drug", "ranked", {"limit": 3})
    assert [r["drugId"] for r in res.obj["rows"]] == ["CHEMBL1", "CHEMBL2", "CHEMBL3"]
    plan, res = await call(gw, "drug", "ranked", {"max_phase": 3, "sort_by": "phase", "limit": 2})
    assert [r["drugId"] for r in res.obj["rows"]] == ["CHEMBL3", "CHEMBL8"]     # sort_by decides


def test_prioritize_targets_lists_the_phase_scale_as_a_number():
    golden = json.loads((REPO / "tests" / "datalayer" / "golden" / "target.prioritize_targets.json").read_text())
    prop = golden["schema"]["properties"]["min_clinical_phase"]
    assert prop["type"] == "number" and prop["maximum"] == 1.0 and "0-1 scale" in golden["description"]
    assert "with min_genetic_constraint: by geneticConstraint asc" in golden["description"]


# --------------------------------------------------------------------------- RV-OT-09


async def test_a_pooled_search_argument_is_never_a_substring_collision(tmp_path):
    """RV-OT-09: search_drugs('statin'), search_pathways('cholesterol') and get_drug_mechanisms(mechanism='inhibitor')
    were refused 'matches several values as a substring; pass the exact value' (the exact 'ATORVASTATIN' too, inside
    'ATORVASTATIN CALCIUM'). A ``pooled`` argument searches: every match is in the answer and the total."""
    ov = copy.deepcopy(DRUG_OVERLAY)
    ov["tools"]["find_genes"] = {
        "reads": {"open_targets.target": {"access": "full_table"}},
        "args": {"query": {"role": "free_text", "interpreted_as": "casefold_substring",
                           "binds": "open_targets.target.approvedSymbol"},
                 "limit": {"role": "limit", "min": 1, "max": 50}},
        "result": {"rows": "$.rows", "order": [{"column": "id", "direction": "asc"}]},
        "serve": "derived", "derived": {"verb": "find", "table": "open_targets.target", "envelope": {"ok": True}}}
    desc = copy.deepcopy(OT)
    desc["tables"]["target"]["columns"]["approvedSymbol"] = {"role": "category", "vocab": "data"}
    from test_dl_gateway_flow import KEYS, TABLES, index_rows
    gw = make_gateway(tmp_path, [desc], [ov], TABLES, index_rows=index_rows(), keys=KEYS)
    with pytest.raises(GatewayError) as e:
        await call(gw, "drug", "find_genes", {"query": "p", "limit": 5})
    assert e.value.kind == ErrorKind.invalid_argument and e.value.payload["reason"] == "substring_collision"
    ov["tools"]["find_genes"]["args"]["query"]["pooled"] = True
    gw = make_gateway(tmp_path / "pooled", [desc], [ov], TABLES, index_rows=index_rows(), keys=KEYS)
    plan, res = await call(gw, "drug", "find_genes", {"query": "p", "limit": 5})
    assert sorted(r["approvedSymbol"] for r in res.obj["rows"]) == ["PCSK9", "TP53"] and hdr(res)["total"] == 2


def test_the_shipped_search_arguments_are_pooled():
    drug = yaml.safe_load((OVERLAYS / "drug.yaml").read_text())["tools"]
    pathway = yaml.safe_load((OVERLAYS / "pathway.yaml").read_text())["tools"]
    assert drug["search_drugs"]["args"]["query"]["pooled"] and drug["get_drug_mechanisms"]["args"]["mechanism"]["pooled"]
    assert pathway["search_pathways"]["args"]["query"]["pooled"] and pathway["search_go_terms"]["args"]["query"]["pooled"]


# --------------------------------------------------------------------------- RV-OT-11


async def test_a_par_y_id_whose_copy_has_its_own_id_is_ambiguous(tmp_path):
    """RV-OT-11: ENSG00000182484_PAR_Y was stripped to the X-chromosome WASH6P while 25.09 stores the Y copy as
    ENSG00000292372 (same symbol)."""
    from test_dl_gateway_flow import KEYS, TABLES, gene_key, index_rows
    from vbt.datalayer.resolve import Entry

    x, y = "ENSG00000182484", "ENSG00000292372"
    idx = index_rows()
    idx["open_targets:ensembl_gene"] += [Entry(gene_key(x), x, "exact", "WASH6P"),
                                         Entry(gene_key("WASH6P"), x, "label_exact:approvedSymbol", "WASH6P"),
                                         Entry(gene_key(y), y, "exact", "WASH6P"),
                                         Entry(gene_key("WASH6P"), y, "label_exact:approvedSymbol", "WASH6P")]
    tables = {**TABLES, "open_targets.target": TABLES["open_targets.target"] + [
        {"id": x, "approvedSymbol": "WASH6P"}, {"id": y, "approvedSymbol": "WASH6P"}]}
    gw = make_gateway(tmp_path, [OT], [DRUG_OVERLAY], tables, index_rows=idx, keys=KEYS)
    with pytest.raises(GatewayError) as e:
        await call(gw, "drug", "get_target", {"target_id": f"{x}_PAR_Y"}, lambda a: {"id": a["target_id"]})
    assert e.value.kind == ErrorKind.ambiguous
    assert {c["id"] for c in e.value.payload["candidates"]} == {x, y}
    # the plain ID and a PAR_Y suffix on a gene without a separate copy still resolve
    plan, res = await call(gw, "drug", "get_target", {"target_id": x}, lambda a: {"id": a["target_id"]})
    assert plan.args_sent["target_id"] == x
    plan, res = await call(gw, "drug", "get_target", {"target_id": f"{PCSK9_ID}_PAR_Y"},
                           lambda a: {"id": a["target_id"]})
    assert plan.args_sent["target_id"] == PCSK9_ID


PCSK9_ID = "ENSG00000169174"


# --------------------------------------------------------------------------- ACC-1


def test_versioned_endpoint_ids_are_read_under_their_stored_spelling(tmp_path):
    """ACC-1: 61 target genes (MIRLET7E, SNORA57, ...) are stored in interaction only as ENSG00000198972.3: the
    canonical ENSG00000198972 matched 0 rows. A declared stored form maps the canonical key to the stored spelling."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from test_dl_service_reader import make_ctx, source
    from vbt.datalayer.resolve import IndexStore
    from vbt.datalayer.service.verbs import load_verbs

    root = tmp_path / "s"
    for name, rows in {"target": [{"id": "ENSG00000198972", "approvedSymbol": "MIRLET7E"},
                                  {"id": "ENSG00000141510", "approvedSymbol": "TP53"}],
                       "interaction": [{"targetA": "ENSG00000198972.3", "targetB": "ENSG00000141510"},
                                       {"targetA": "ENSG00000141510", "targetB": "ENSG00000198972.3"}]}.items():
        (root / name).mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), root / name / "part-0.parquet")
    tables = {"target": {"kind": "entity", "path": "target", "grain": "one gene", "key": {"columns": ["id"]},
                         "columns": {"id": {"role": "identifier", "id_type": "ensembl_gene", "self": True},
                                     "approvedSymbol": {"role": "label", "of": "id"}}},
              "interaction": {"kind": "edges", "path": "interaction", "grain": "one edge",
                              "key": {"columns": ["targetA", "targetB"]},
                              "columns": {"targetA": {"role": "endpoint", "side": "a", "id_type": "ensembl_gene"},
                                          "targetB": {"role": "endpoint", "side": "b", "id_type": "ensembl_gene"}}}}
    id_types = {"ensembl_gene": {"plugin": "ensembl_gene", "universe": "target.id",
                                 "resolve_via": ["target.approvedSymbol"],
                                 "stored_forms": {"interaction.targetA": "as_stored",
                                                  "interaction.targetB": "as_stored"}}}
    ctx = make_ctx(tmp_path, source("s", root, tables, id_types=id_types))
    out = load_verbs()["_build_index"](ctx, {"source": "s", "id_type": "ensembl_gene"})
    index = IndexStore(ctx.settings.cache_dir).load("s", out["fingerprint"], "ensembl_gene")
    assert index.stored_value("ENSG00000198972", "s.interaction") == "ENSG00000198972.3"
    assert index.stored_value("ENSG00000141510", "s.interaction") is None      # stored as is


def test_other_species_ids_beside_the_stored_forms_leave_the_table_ready(tmp_path):
    """ACC-1, found on the real 25.09 interaction_evidence: declaring the version-suffixed endpoints as stored forms
    made R9:stored_form a key violation for the other species' versioned IDs on the same columns
    (ENSAMEG00000026011.1, ...), so the table was not_ready and get_interaction_evidence refused every call. The
    columns are integrity: partial: the unmatched stored values are a warning, as their R9 dangling references are."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from test_dl_service_reader import make_ctx, source
    from vbt.datalayer.service.checks import check_table

    root = tmp_path / "s"
    edges = [{"targetA": "ENSG00000198972.3", "targetB": "ENSG00000141510"},
             {"targetA": "ENSAMEG00000026011.1", "targetB": "ENSG00000141510"}]
    for name, rows in {"target": [{"id": "ENSG00000198972"}, {"id": "ENSG00000141510"}], "interaction": edges}.items():
        (root / name).mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), root / name / "part-0.parquet")

    def ctx_for(integrity: str) -> Any:
        tables = {"target": {"kind": "entity", "path": "target", "grain": "one gene", "key": {"columns": ["id"]},
                             "columns": {"id": {"role": "identifier", "id_type": "ensembl_gene", "self": True}}},
                  "interaction": {"kind": "edges", "path": "interaction", "grain": "one edge",
                                  "key": {"columns": ["targetA", "targetB"]},
                                  "columns": {"targetA": {"role": "endpoint", "side": "a", "id_type": "ensembl_gene",
                                                          "ref": "target.id", "integrity": integrity},
                                              "targetB": {"role": "endpoint", "side": "b", "id_type": "ensembl_gene",
                                                          "ref": "target.id", "integrity": integrity}}}}
        id_types = {"ensembl_gene": {"plugin": "ensembl_gene", "universe": "target.id",
                                     "stored_forms": {"interaction.targetA": "as_stored",
                                                      "interaction.targetB": "as_stored"}}}
        (tmp_path / integrity).mkdir()
        return make_ctx(tmp_path / integrity, source("s", root, tables, id_types=id_types))

    model = check_table(ctx_for("partial"), "s.interaction")
    hits = [c for c in model.checks if c.name == "R9:stored_form" and c.column == "targetA"]
    assert model.status == "ready" and hits and hits[0].level == "warning" and "ENSAMEG00000026011.1" in hits[0].detail
    assert check_table(ctx_for("full"), "s.interaction").status == "key_violation"


def test_the_shipped_gene_type_declares_the_versioned_endpoints():
    from vbt.datalayer.descriptor.load import load_descriptors

    ot = load_descriptors(REPO / "configs" / "data" / "sources", {"project_root": str(REPO)})["open_targets"]
    assert set(ot.id_types["ensembl_gene"].stored_forms) == {
        "interaction.targetA", "interaction.targetB", "interaction_evidence.targetA", "interaction_evidence.targetB"}


# --------------------------------------------------------------------------- ACC-2


HYPHENATED_UKB_PPP = ("UKB_PPP_EUR_HLA-DRA_P01903_OID20520_v1", "UKB_PPP_EUR_HLA-A_P04439_OID31048_v1",
                      "UKB_PPP_EUR_HLA-E_P13747_OID20532_v1", "UKB_PPP_EUR_ERVV-1_B6SEH8_OID30094_v1")


def _gwas_study_plugin():
    from vbt.datalayer.descriptor.load import load_descriptors
    from vbt.datalayer.plugins.identifiers.study_locus import GwasStudy

    ot = load_descriptors(REPO / "configs" / "data" / "sources", {"project_root": str(REPO)})["open_targets"]
    return GwasStudy().configure(ot.id_types["gwas_study"].options, None)


def test_ukb_ppp_study_ids_with_a_hyphen_normalize():
    """ACC-2: the four 25.09 UKB-PPP pQTL studies whose gene part holds a hyphen were rejected."""
    from vbt.datalayer.plugins.base import Normalized

    p = _gwas_study_plugin()
    for sid in (*HYPHENATED_UKB_PPP, "GCST004988", "GCST000337_7", "FINNGEN_R12_I9_HYPTENS",
                "gtex_ge_brain_cerebellar_hemisphere_ensg00000067445"):
        n = p.normalize(sid)
        assert isinstance(n, Normalized) and n.value == sid, sid
        assert isinstance(p.normalize(f" {sid} "), Normalized)


@pytest.mark.skipif(not os.environ.get("VBT_DL_REAL_DATA"), reason="VBT_DL_REAL_DATA=<dir> with OT 25.09 study")
def test_every_real_study_id_normalizes():
    pq = pytest.importorskip("pyarrow.parquet")
    from dl_upstream import real_ot_dir
    from vbt.datalayer.plugins.base import Normalized

    d = real_ot_dir() / "study"
    if not d.is_dir():
        pytest.skip(f"{d} is not there")
    ids = pq.read_table(d, columns=["studyId"])["studyId"].to_pylist()
    p = _gwas_study_plugin()
    bad = [i for i in ids if not isinstance(p.normalize(i), Normalized)]
    assert len(ids) == 1_964_234 and bad == []
