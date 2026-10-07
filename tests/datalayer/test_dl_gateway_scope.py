"""The scope-completeness rule (§11.3 step 7, I7) in pass and derived mode, per-group cuts and
incomparable orders, on a Tahoe-shaped composite key (drug x cell line x concentration x plate x gene)."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest

from test_dl_gateway_flow import REGISTRY, call, f32, hdr, make_gateway
from vbt.datalayer.catalog import Catalog
from vbt.datalayer.descriptor.models import SourceDescriptor
from vbt.datalayer.descriptor.overlay import Overlay
from vbt.datalayer.errors import ErrorKind, GatewayError
from vbt.datalayer.gateway.scope import per_group_cut, scope_completeness
from vbt.datalayer.ipc import WitnessResponse
from vbt.datalayer.resolve import Entry

DRUG, CELL = "Erdafitinib", "ACH-000001"

TAHOE: dict[str, Any] = {
    "schema": "vbt.datasource/1", "source": "tahoe_100m", "title": "Tahoe-100M",
    "release": {"from": "literal"}, "defaults": {"format": "parquet", "layout": "single_file"},
    "id_types": {
        "tahoe_drug": {"plugin": "tahoe_drug", "universe": "drug_metadata.drug", "resolve_via": ["drug_metadata.drug"],
                       "stored_forms": {"de_permissive.drug": "as_stored"}},
        "depmap_cell_line": {"plugin": "depmap_cell_line", "universe": "cell_line_metadata.Cell_ID_DepMap"},
    },
    "tables": {
        "de_permissive": {
            "kind": "fact", "grain": "one gene in one contrast",
            "key": {"columns": ["drug", "Cell_ID_DepMap", "concentration", "plate", "gene_name"]},
            "grains": {"gene": ["gene_name"]},
            "columns": {
                "drug": {"role": "identifier", "id_type": "tahoe_drug"},
                "Cell_ID_DepMap": {"role": "identifier", "id_type": "depmap_cell_line"},
                "concentration": {"role": "scope", "unit_from": "concentration_unit"},
                "concentration_unit": {"role": "category"},
                "plate": {"role": "scope", "scope": {"kind": "replicate"}},
                "gene_name": {"role": "label"},
                "log2FoldChange": {"role": "measure", "statistic": "numeric", "direction": "signed"},
                "padj": {"role": "measure", "statistic": "numeric"}}},
        "drug_metadata": {"kind": "entity", "grain": "one drug", "key": {"columns": ["drug"]},
                          "columns": {"drug": {"role": "identifier", "id_type": "tahoe_drug", "self": True}}},
        "cell_line_metadata": {"kind": "entity", "grain": "one cell line", "key": {"columns": ["Cell_ID_DepMap"]},
                               "columns": {"Cell_ID_DepMap": {"role": "identifier", "id_type": "depmap_cell_line",
                                                              "self": True}}},
    },
}

QDP: dict[str, Any] = {
    "reads": {"tahoe_100m.de_permissive": {"access": "bounded_scan"},
              "tahoe_100m.drug_metadata": {"access": "full_table"}},
    "args": {
        "drug_name": {"binds": "tahoe_100m.de_permissive.drug", "accepts": ["tahoe_drug"], "send_as": "stored"},
        "cell_line_id": {"binds": "tahoe_100m.de_permissive.Cell_ID_DepMap", "accepts": ["depmap_cell_line"]},
        "concentration": {"binds": "tahoe_100m.de_permissive.concentration", "gateway_only": True},
        "plate": {"binds": "tahoe_100m.de_permissive.plate", "gateway_only": True},
        "max_padj": {"binds": "tahoe_100m.de_permissive.padj", "op": "le", "max": 0.10},
        "top_n": {"role": "limit", "min": 1, "max": 500, "limit_grain": "gene"}},
    "result": {"rows": ["$.top_upregulated", "$.top_downregulated"], "grain": "gene",
               "summary_fields": {"$.num_total_significant": {"recompute": {"agg": "count_distinct", "of": "gene_name"}}}},
    "serve": "derived",
    "derived": {"verb": "find", "table": "tahoe_100m.de_permissive",
                "split": {"by_sign": "log2FoldChange", "into": {"+": "$.top_upregulated", "-": "$.top_downregulated"}},
                "rename": {"gene_name": "gene"},
                "sections": {"drug_info": {"path": "$.drug_info", "table": "tahoe_100m.drug_metadata", "verb": "lookup",
                                           "single": True, "key_from_args": {"drug": "drug_name"}}}},
}
OVERLAY: dict[str, Any] = {"schema": "vbt.overlay/1", "server": "functional_genomics", "sources": ["tahoe_100m"],
                           "tools": {"query_drug_perturbation": QDP}}

DE = []
for conc in (0.05, 0.5, 5.0):
    for plate in (["plate3", "plate9"] if conc == 0.05 else ["plate3"]):
        for gene, lfc in (("PCSK9", 1.5), ("TP53", -2.0), ("BRCA1", 0.7)):
            DE.append({"drug": "Erdafitinib ", "Cell_ID_DepMap": CELL, "concentration": f32(conc),
                       "concentration_unit": "uM", "plate": plate, "gene_name": gene,
                       "log2FoldChange": lfc * conc, "padj": 0.01})
TABLES = {"tahoe_100m.de_permissive": DE, "tahoe_100m.drug_metadata": [{"drug": DRUG, "moa": "FGFR"}],
          "tahoe_100m.cell_line_metadata": [{"Cell_ID_DepMap": CELL}]}
KEYS = {"tahoe_100m.de_permissive": ["drug", "Cell_ID_DepMap", "concentration", "plate", "gene_name"]}


def index_rows() -> dict[str, list[Entry]]:
    td = REGISTRY.get("identifier", "tahoe_drug").label_key
    dc = REGISTRY.get("identifier", "depmap_cell_line").label_key
    return {"tahoe_100m:tahoe_drug": [Entry(td(DRUG), DRUG, "exact"),
                                      Entry(td(DRUG), DRUG, "label_exact:drug", DRUG),
                                      Entry(td("Erdafitinib "), DRUG, "stored_form",
                                            stored_table="tahoe_100m.de_permissive", stored_value="Erdafitinib ")],
            "tahoe_100m:depmap_cell_line": [Entry(dc(CELL), CELL, "exact")]}


def tahoe(tmp_path, overlay=OVERLAY):
    return make_gateway(tmp_path, [TAHOE], [overlay], TABLES, index_rows=index_rows(), keys=KEYS,
                        storage={"tahoe_100m.de_permissive": {"concentration": "float"}})


async def test_derived_query_without_concentration_is_incomplete_key_in_storage_type(tmp_path):
    gw = tahoe(tmp_path)
    with pytest.raises(GatewayError) as e:
        await call(gw, "functional_genomics", "query_drug_perturbation", {"drug_name": DRUG, "cell_line_id": CELL})
    err = e.value
    assert err.kind == ErrorKind.incomplete_key
    env = err.envelope()
    assert env["dimension"] == "concentration" and env["argument"] == "concentration"
    assert env["values"] == [0.05, 0.5, 5.0] and env["unit"] == "uM"
    assert {"concentration": 0.05} in env["retry_with"]
    assert not gw.service.verbs("_serve")            # refused before any rows were served


async def test_pass_mode_same_rule(tmp_path):
    ov = copy.deepcopy(OVERLAY)
    tool = ov["tools"]["query_drug_perturbation"]
    tool["serve"] = "pass"
    tool.pop("derived")
    gw = tahoe(tmp_path, ov)
    with pytest.raises(GatewayError) as e:
        await call(gw, "functional_genomics", "query_drug_perturbation", {"drug_name": DRUG, "cell_line_id": CELL},
                   lambda a: {"top_upregulated": [], "top_downregulated": []})
    assert e.value.kind == ErrorKind.incomplete_key


async def test_fixed_concentration_snaps_to_stored_value_and_lists_plates(tmp_path):
    gw = tahoe(tmp_path)
    plan, res = await call(gw, "functional_genomics", "query_drug_perturbation",
                           {"drug_name": DRUG, "cell_line_id": CELL, "concentration": 0.05, "top_n": 10})
    h = hdr(res)
    assert h["scope"]["concentration"] == 0.05
    assert h["scope"]["plate"] == {"listed": ["plate3", "plate9"]}
    assert "concentration" not in plan.args_sent and plan.gateway_args["concentration"] == f32(0.05)
    assert plan.args_sent["drug_name"] == "Erdafitinib "          # send_as: stored
    up = res.obj["top_upregulated"]
    assert up and all("gene" in r for r in up)
    assert res.obj["drug_info"]["drug"] == DRUG
    assert h["served_by"] == "derived"


async def test_unknown_concentration_value_is_invalid_argument(tmp_path):
    gw = tahoe(tmp_path)
    with pytest.raises(GatewayError) as e:
        await call(gw, "functional_genomics", "query_drug_perturbation",
                   {"drug_name": DRUG, "cell_line_id": CELL, "concentration": 0.07})
    assert e.value.kind == ErrorKind.invalid_argument
    assert e.value.envelope()["valid_values"] == [0.05, 0.5, 5.0]


# --------------------------------------------------------------------------- unit level


def _contract(dims_pooling: str = "forbid", grain: str | None = "gene", order=(), limit_mode="per_group"):
    desc = copy.deepcopy(TAHOE)
    desc["tables"]["de_permissive"]["columns"]["log2FoldChange"]["comparable_within"] = ["concentration"]
    desc["tables"]["de_permissive"]["columns"]["concentration"]["scope"] = {"kind": "condition",
                                                                            "pooling": dims_pooling}
    ov = copy.deepcopy(OVERLAY)
    tool = ov["tools"]["query_drug_perturbation"]
    tool["serve"] = "pass"
    tool.pop("derived")
    tool["result"]["summary_fields"] = {}
    tool["result"]["grain"] = grain
    tool["result"]["order"] = list(order)
    tool["args"]["top_n"]["limit_grain"] = None
    tool["args"]["top_n"]["limit_mode"] = limit_mode
    cat = Catalog({"tahoe_100m": SourceDescriptor.model_validate(desc)},
                  {"functional_genomics": Overlay.model_validate(ov)})
    return cat.contract("functional_genomics", "query_drug_perturbation")


def _w(**distinct):
    return WitnessResponse(total=10, total_method="scan", distinct=distinct)


def test_pooling_group_and_list():
    c = _contract("group")
    d = scope_completeness(c, SimpleNamespace(bound_table=None), _w(concentration=[0.5, 0.05], plate=["p1"]))
    assert d.grouped == {"concentration": [0.05, 0.5]}
    assert d.disclosure()["plate"] == "p1"
    c = _contract("list")
    d = scope_completeness(c, SimpleNamespace(bound_table=None), _w(concentration=[0.5, 0.05]))
    assert d.listed == {"concentration": [0.05, 0.5]}


def test_fixed_dimension_and_single_value_pass():
    c = _contract("forbid")
    d = scope_completeness(c, SimpleNamespace(bound_table=None), _w(concentration=[0.05, 0.5]),
                           fixed={"concentration": 0.5})
    assert d.fixed == {"concentration": 0.5}
    d = scope_completeness(c, SimpleNamespace(bound_table=None), _w(concentration=[0.05]))
    assert d.listed == {"concentration": [0.05]}


def test_rows_carrying_the_dimension_are_listed_not_refused():
    c = _contract("forbid", grain=None)
    d = scope_completeness(c, SimpleNamespace(bound_table=None), _w(concentration=[0.05, 0.5]))
    assert d.listed["concentration"] == [0.05, 0.5]


def test_incomparable_order_per_group_or_refused():
    order = [{"column": "log2FoldChange", "direction": "desc_abs"}]
    c = _contract("forbid", grain=None, order=order)
    d = scope_completeness(c, SimpleNamespace(bound_table=None), _w(concentration=[0.05, 0.5]), inflated=True)
    assert d.per_group == ["concentration"]
    with pytest.raises(GatewayError) as e:           # pass mode, limit not inflated: cannot cut per group
        scope_completeness(c, SimpleNamespace(bound_table=None), _w(concentration=[0.05, 0.5]), inflated=False)
    assert e.value.subkind == "incomparable_order"
    c = _contract("forbid", grain=None, order=order, limit_mode="refuse")
    with pytest.raises(GatewayError) as e:
        scope_completeness(c, SimpleNamespace(bound_table=None), _w(concentration=[0.05, 0.5]), inflated=True)
    assert e.value.kind == ErrorKind.incomplete_key and e.value.subkind == "incomparable_order"


def test_pool_ok_for_verbs():
    desc = copy.deepcopy(TAHOE)
    desc["tables"]["de_permissive"]["columns"]["concentration"]["scope"] = {"kind": "condition",
                                                                            "pool_ok_for": ["count"]}
    ov = copy.deepcopy(OVERLAY)
    ov["tools"]["query_drug_perturbation"]["serve"] = "pass"
    ov["tools"]["query_drug_perturbation"].pop("derived")
    cat = Catalog({"tahoe_100m": SourceDescriptor.model_validate(desc)},
                  {"functional_genomics": Overlay.model_validate(ov)})
    c = cat.contract("functional_genomics", "query_drug_perturbation")
    d = scope_completeness(c, SimpleNamespace(bound_table=None), _w(concentration=[0.05, 0.5]), verb="count")
    assert "concentration" not in d.listed and any("pool" in n for n in d.notes)


def test_per_group_cut():
    rows = [{"g": "a", "v": 3}, {"g": "a", "v": 2}, {"g": "b", "v": 9}, {"g": "a", "v": 1}, {"g": "b", "v": 1}]
    kept, totals = per_group_cut(rows, ["g"], 1)
    assert [r["v"] for r in kept] == [3, 9] and sorted(totals.values()) == [2, 3]
    kept, _ = per_group_cut(rows, [], 2)
    assert len(kept) == 2
