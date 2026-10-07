"""Derived argument schemas and agent-facing text (§10.1, §10.2) for representative contracts,
snapshot-tested; generic tools stay untouched; the gateway and derive modules import without
pyarrow or pandas (I12)."""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from test_dl_gateway_flow import DRUG_OVERLAY, OT, REGISTRY, f32
from test_dl_gateway_scope import OVERLAY as TAHOE_OVERLAY
from test_dl_gateway_scope import TAHOE
from vbt.datalayer.catalog import Catalog
from vbt.datalayer.descriptor.models import SourceDescriptor
from vbt.datalayer.descriptor.overlay import Overlay
from vbt.datalayer.derive import annotate_schema, auto_scope_args, describe_tool
from vbt.datalayer.gateway.contracts import VocabSnapshot

SRC = Path(__file__).resolve().parents[2] / "src"

EXTRA_OT = copy.deepcopy(OT)
EXTRA_OT["tables"]["target"]["columns"]["go"] = {
    "role": "nested", "item_key": ["id"], "grain": "one GO annotation of the gene",
    "fields": {"id": {"role": "label"}, "aspect": {"role": "category", "vocab": ["P", "F", "C"]}}}
EXTRA_OT["tables"]["target_go"] = {"kind": "fact", "grain": "one GO annotation of the gene",
                                   "items_of": {"table": "target", "path": "go[]"}, "key": {"columns": []},
                                   "coverage": {"statement": "GO annotations from the release.",
                                                "absence_means": "unknown"}}
EXTRA_OT["tables"]["coloc"] = {"kind": "fact", "grain": "one colocalisation", "key": {"columns": ["l"]},
                               "columns": {"l": {"role": "label"}, "h4": {"role": "measure", "scale": [0, 1]},
                                           "clpp": {"role": "measure"}}}
EXTRA_OT["tables"]["ecaviar"] = copy.deepcopy(EXTRA_OT["tables"]["coloc"])
EXTRA_OT["tables"]["known_drug"]["evidence_nature"] = {"kind": "curation", "caveat": "curated trial records"}

OV = copy.deepcopy(DRUG_OVERLAY)
OV["tools"]["go_terms"] = {
    "reads": {"open_targets.target": {"access": "full_table", "columns": ["go"]}},
    "args": {"target_id": {"binds": "open_targets.target.id", "accepts": ["ensembl_gene", "hgnc_symbol"]},
             "aspect": {"binds": "open_targets.target.go[].aspect", "send_map": {"P": "biological_process"}}},
    "result": {"rows": "$.go", "rows_of": "open_targets.target_go", "parent_key": {"id": "$.target_id"}},
    "text": {"drop_promises": ["cancer_hallmarks"]}}
OV["tools"]["colocalisation"] = {
    "reads": {"open_targets.coloc": {"access": "full_table"}, "open_targets.ecaviar": {"access": "full_table"}},
    "args": {"method": {"role": "selector", "values": {"coloc": "open_targets.coloc", "ecaviar": "open_targets.ecaviar"}},
             "sort_by": {"role": "order_by", "binds": "open_targets.coloc.h4"},
             "min_h4": {"binds": "open_targets.coloc.h4", "op": "ge"}},
    "result": {"rows": "$.rows"}}
OV["tools"]["search_trials"] = {
    "reads": {"open_targets.known_drug": {"access": "upstream"}},
    "args": {"condition": {"role": "free_text", "interpreted_as": "engine",
                           "engine_doc": "registry search with synonym expansion"},
             "pattern": {"role": "free_text", "interpreted_as": "regex"},
             "status": {"binds": "open_targets.known_drug.status"}},
    "exclusive": [["condition", "pattern"]],
    "result": {"rows": "$.trials", "order_source": "source_server_side",
               "order": [{"column": "phase", "direction": "desc"}]}}

TAHOE_AUTO = copy.deepcopy(TAHOE_OVERLAY)
del TAHOE_AUTO["tools"]["query_drug_perturbation"]["args"]["concentration"]


@pytest.fixture(scope="module")
def catalog():
    return Catalog({"open_targets": SourceDescriptor.model_validate(copy.deepcopy(EXTRA_OT)),
                    "tahoe_100m": SourceDescriptor.model_validate(copy.deepcopy(TAHOE))},
                   {"drug": Overlay.model_validate(copy.deepcopy(OV)),
                    "functional_genomics": Overlay.model_validate(copy.deepcopy(TAHOE_AUTO))}, registry=REGISTRY)


SCHEMAS: dict[str, dict[str, Any]] = {
    "search_known_drugs": {"type": "object", "properties": {"target_id": {"type": "string", "pattern": "^ENSG"},
                                                            "min_phase": {"type": "number"},
                                                            "limit": {"type": "integer", "default": 20}},
                           "additionalProperties": False},
    "get_target": {"type": "object", "properties": {"target_id": {"type": "string"}}},
    "go_terms": {"type": "object", "properties": {"target_id": {"type": "string"}, "aspect": {"type": "string"}}},
    "colocalisation": {"type": "object", "properties": {"method": {"type": "string"},
                                                        "sort_by": {"type": "string"}, "min_h4": {"type": "number"}}},
    "search_trials": {"type": "object", "properties": {"condition": {"type": "string"}, "pattern": {"type": "string"},
                                                       "status": {"type": "string", "default": "Completed"}}},
    "old_tool": {"type": "object", "properties": {}},
    "query_drug_perturbation": {"type": "object", "properties": {"drug_name": {"type": "string"},
                                                                 "cell_line_id": {"type": "string"},
                                                                 "max_padj": {"type": "number"},
                                                                 "top_n": {"type": "integer"}}},
}
DESCRIPTIONS = {
    "search_known_drugs": "Search known drugs for a target.\n\nReturns:\n    drugs with mechanismOfAction",
    "get_target": "Get target info. Includes chromosome.",
    "go_terms": "List GO terms and cancer_hallmarks for a gene. Returns aspects.",
    "colocalisation": "Colocalisation evidence (coloc or eCAVIAR).",
    "search_trials": "Search trials by condition.",
    "old_tool": "An old tool.",
    "query_drug_perturbation": "Query DE results for a drug in a cell line.",
}
SERVER = {"query_drug_perturbation": "functional_genomics"}
VOCAB = {"tahoe_100m.de_permissive.concentration": VocabSnapshot(values=[f32(0.05), f32(0.5), f32(5.0)],
                                                                 storage_type="float"),
         "tahoe_100m.de_permissive.plate": VocabSnapshot(values=["plate3", "plate9"])}

def snapshot(catalog: Catalog, tool: str) -> dict[str, Any]:
    c = catalog.contract(SERVER.get(tool, "drug"), tool)
    schema = annotate_schema(c, SCHEMAS[tool], catalog=catalog, registry=REGISTRY, vocab=VOCAB)
    text = describe_tool(c, DESCRIPTIONS[tool], catalog=catalog)
    return {"schema": schema, "text": text}


def test_search_known_drugs(catalog):
    s = snapshot(catalog, "search_known_drugs")
    p = s["schema"]["properties"]
    assert p["target_id"] == {
        "type": "string", "x-vbt-id-type": "open_targets:ensembl_gene",
        "x-vbt-accepts": ["open_targets:ensembl_gene", "open_targets:hgnc_symbol"],
        "examples": ["ENSG00000169174", "PCSK9"],
        "description": "Accepts ensembl_gene, hgnc_symbol; resolved to one ensembl_gene (unknown values are errors, "
                       "never empty results)."}
    assert p["min_phase"] == {"type": "number", "minimum": 0, "maximum": 4,
                              "description": "Keeps rows with phase >= this value; unknown values never pass."}
    assert p["limit"] == {"type": "integer", "default": 20, "minimum": 1, "maximum": 200}
    assert s["schema"]["additionalProperties"] is False
    assert s["schema"]["x-vbt-require-any"] == [["target_id"]]
    assert s["text"] == (
        "Search known drugs for a target.\n"
        "Data: Open Targets 25.09 · table known_drug — one record = one (drug, target) record.\n"
        "Key: drugId, targetId.\n"
        "Arguments: target_id accepts ensembl gene or hgnc symbol (resolved; unknown -> error).\n"
        "Results: ranked by phase desc (verified by the harness); `_vbt.total` counts all matches; "
        "`_vbt.grains.drug` counts drugs.\n"
        "Rows with unknown phase are excluded and counted in `_vbt.excluded_unknown`.\n"
        "Evidence: curated trial records.\n"
        "Empty vs not found: unknown identifiers are errors; an empty result is citable only as an absence. "
        "ChEMBL-curated drugs only.")


def test_record_tool_drops_returns_section(catalog):
    s = snapshot(catalog, "get_target")
    assert s["text"].startswith("Get target info.\nData: Open Targets 25.09 · table target — one record = one target.")
    assert "chromosome" not in s["text"]


def test_item_table_grain_and_drop_promises(catalog):
    s = snapshot(catalog, "go_terms")
    assert "cancer_hallmarks" not in s["text"]                    # the first sentence promised it
    assert "table target_go — one record = one GO annotation of the gene." in s["text"]
    assert "one row = one item (one GO annotation of the gene); `_vbt.total` counts items" in s["text"]
    assert "Key: id, go[].id." in s["text"]
    assert s["schema"]["properties"]["aspect"]["enum"] == ["P", "F", "C"]


def test_selector_and_order_by_enums(catalog):
    p = snapshot(catalog, "colocalisation")["schema"]["properties"]
    assert p["method"]["enum"] == ["coloc", "ecaviar"]
    assert p["sort_by"]["enum"] == ["clpp", "h4"]
    assert p["min_h4"]["minimum"] == 0 and p["min_h4"]["maximum"] == 1


def test_engine_regex_defaults_and_exclusive(catalog):
    s = snapshot(catalog, "search_trials")
    p = s["schema"]["properties"]
    assert p["condition"]["description"] == "Matched by the source's search engine: registry search with synonym expansion."
    assert p["pattern"]["description"] == "Literal text, not a pattern."
    assert p["status"]["enum"] == ["Completed", "Recruiting"]
    assert p["status"]["description"] == "Defaults to 'Completed' (null is not accepted)."   # GW-NULL
    assert s["schema"]["x-vbt-exclusive"] == [["condition", "pattern"]]
    assert "ranked by phase desc (ranked by the source, not verified)" in s["text"]


def test_blocked_tool_text(catalog):
    s = snapshot(catalog, "old_tool")
    assert s["text"] == ("UNAVAILABLE: wrong answers on this data; use mcp__drug__search_known_drugs. An old tool.")


def test_auto_derived_gateway_only_scope_argument(catalog):
    c = catalog.contract("functional_genomics", "query_drug_perturbation")
    assert auto_scope_args(c, SCHEMAS["query_drug_perturbation"]) == {"concentration": "concentration"}
    s = snapshot(catalog, "query_drug_perturbation")
    p = s["schema"]["properties"]
    assert p["concentration"] == {"x-gateway": True, "enum": [0.05, 0.5, 5.0],
                                  "description": "Results are per concentration; pass one value, or the call fails "
                                                 "with incomplete_key when several apply."}
    assert p["plate"]["x-gateway"] is True and p["plate"]["enum"] == ["plate3", "plate9"]
    assert p["max_padj"]["maximum"] == 0.1
    assert p["top_n"] == {"type": "integer", "minimum": 1, "maximum": 500, "description": "Counts genes, not rows."}
    assert "top_n counts genes, not rows" in s["text"]


def test_generic_tools_untouched(catalog):
    c = catalog.contract("unknown", "tool")
    schema = {"type": "object", "properties": {"q": {"type": "string", "pattern": "x"}}}
    assert annotate_schema(c, schema, catalog=catalog, registry=REGISTRY) == schema
    assert describe_tool(c, "Upstream text. Returns: x", catalog=catalog) == "Upstream text. Returns: x"


def test_description_cap(catalog):
    c = catalog.contract("drug", "search_known_drugs")
    text = describe_tool(c, DESCRIPTIONS["search_known_drugs"], catalog=catalog, max_chars=200)
    assert len(text) <= 200 and text.endswith("…")


def test_modules_import_without_pyarrow_or_pandas():
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('pyarrow', 'pandas'):\n"
        "            raise ImportError('blocked ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import vbt.datalayer.gateway, vbt.datalayer.derive\n"
        "from vbt.datalayer.gateway import classify, contracts, fields, files, leakage, readiness, scope\n"
        "from vbt.datalayer.gateway import service_client, soma_filter, transforms\n"
        "from vbt.datalayer import build_gateway\n"
        "build_gateway({}, None)\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('pyarrow', 'pandas'))\n"
        "assert not bad, bad\n"
        "print('ok')\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(SRC.parent),
                         env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
