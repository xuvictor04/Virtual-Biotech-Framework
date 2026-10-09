"""Enforce-mode positive controls of the phase-1 review findings, on the unmodified upstream servers.

Every call here has a correct upstream answer on the fixtures (or a correct empty one); the gateway must
pass it through, never withhold it as ``tool_defect``, drop its rows, or report a false empty. The
calls run through the shared live bridges of ``conftest`` (see :mod:`test_dl_correctness_six`).
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

import dl_fixtures as F
from dl_upstream import CallResult, gateway_missing, needs_arrow, needs_fastmcp

pytestmark = [needs_arrow, needs_fastmcp, pytest.mark.correctness]

PCSK9, TP53, TP53BP1, NDE1 = F.PCSK9, F.TP53, F.TP53BP1, F.G_L2G


def enforce(live: Callable[[str], Any], fixture_ready: Callable[[Any], None]) -> Any:
    if gateway_missing():
        pytest.skip(gateway_missing())
    bridge = live("enforce")
    fixture_ready(bridge)
    return bridge


def ok(r: CallResult, status: str = "ok") -> CallResult:
    assert not r.is_error, r.text[:800]
    assert (r.status or r.header.get("status")) == status, r.header
    return r


def ids(rows: list[Any], col: str = "id") -> list[Any]:
    return [r.get(col) for r in rows]


# ---------------------------------------------------------------------------- items of a parent record


@pytest.mark.parametrize("server", ["target", "drug"])
@pytest.mark.parametrize("target", [PCSK9, "pcsk9", "ENSG00000169174.12"])
def test_tractability_buckets_survive_the_parent_key(live, fixture_ready, server: str, target: str) -> None:
    """Each bucket has its own `id`: the parent's target_id is checked on the parent key, never on it."""
    r = ok(enforce(live, fixture_ready).call(server, "get_target_tractability", {"target_id": target}))
    assert [(b["modality"], b["id"]) for b in r.rows("tractability")] == [("SM", "Approved Drug"),
                                                                           ("AB", "Approved Drug")]
    assert not r.header.get("excluded") and r.header.get("returned") == r.header.get("total") == 2


def test_target_with_probes_is_never_an_absence(live, fixture_ready) -> None:
    r = ok(enforce(live, fixture_ready).call("target", "get_chemical_probes", {"target_id": PCSK9}))
    assert ids(r.rows("chemical_probes")) == ["PROBE-1"] and not r.header.get("excluded")
    assert "citable only as an absence" not in str(r.header.get("cite"))


def test_nested_annotation_lists_are_counted_as_items(live, fixture_ready) -> None:
    """The witness counts constraint/class/location items, not the one parent record (no W1/W2)."""
    bridge = enforce(live, fixture_ready)
    off = live("off")
    r = ok(bridge.call("target", "get_genetic_constraint", {"target_id": NDE1}))
    want = [c["constraintType"] for c in off.call("target", "get_genetic_constraint", {"target_id": NDE1}).obj["constraint"]]
    assert ids(r.rows("constraint"), "constraintType") == want and len(want) == 3
    for tool, path in [("get_genetic_constraint", "constraint"), ("get_target_class", "target_class"),
                       ("get_subcellular_locations", "subcellular_locations")]:
        for gene in (PCSK9, TP53):
            r = bridge.call("target", tool, {"target_id": gene})
            assert not r.is_error, (tool, gene, r.text[:600])
            assert r.rows(path) == off.call("target", tool, {"target_id": gene}).obj.get(path), (tool, gene)
    r = ok(bridge.call("target", "get_target_tep", {"target_id": TP53}), "empty")
    assert r.obj.get("tep") is None


def test_study_by_id_is_a_record(live, fixture_ready) -> None:
    r = ok(enforce(live, fixture_ready).call("genetics", "get_study_metadata", {"study_id": F.STUDY_BIG}))
    assert r.obj.get("studyId") == F.STUDY_BIG and r.obj.get("traitFromSource") == "LDL cholesterol"


# ---------------------------------------------------------------------------- field maps and item filters


def test_mapped_fields_are_returned_as_upstream_gave_them(live, fixture_ready) -> None:
    """List-crossing and dotted field maps never null an upstream value or add a key (F3)."""
    bridge, off = enforce(live, fixture_ready), live("off")
    calls = [("functional_genomics", "compare_essentiality_across_diseases", {"gene_id": TP53}),
             ("functional_genomics", "query_gene_essentiality", {"gene_id": TP53}),
             ("expression", "query_expression_by_gene", {"gene_id": PCSK9}),
             ("expression", "query_expression_by_gene", {"gene_id": PCSK9, "tissue": "liver"}),
             ("expression", "compare_expression_across_tissues", {"gene_id": PCSK9}),
             ("disease", "get_disease_hierarchy", {"disease_id": F.T2D}),
             ("expression", "list_available_tissues", {})]
    for server, tool, args in calls:
        r, o = bridge.call(server, tool, args), off.call(server, tool, args)
        assert not r.is_error, (tool, r.text[:600])
        assert set(r.obj) - {"_vbt"} <= set(o.obj), (tool, sorted(set(r.obj) - set(o.obj)))
    r = bridge.call("functional_genomics", "compare_essentiality_across_diseases", {"gene_id": TP53})
    assert [d["disease"] for d in r.obj["disease_essentiality"]] == ["Lung Cancer"]
    r = bridge.call("disease", "get_disease_hierarchy", {"disease_id": F.T2D})
    assert "children[].name" not in r.obj


@pytest.mark.parametrize("tissue", ["liver", "UBERON_0002107"])
def test_expression_by_tissue_finds_the_tissue(live, fixture_ready, tissue: str) -> None:
    r = ok(enforce(live, fixture_ready).call("expression", "query_expression_by_tissue",
                                             {"tissue": tissue, "output_path": f"byt_{tissue}"}))
    assert ids(r.rows("top_genes"), "gene_id") == [PCSK9]


@pytest.mark.parametrize("aspect,term", [("P", "GO:0006629"), ("F", "GO:1000001"),
                                         ("molecular_function", "GO:1000001")])
def test_gene_ontology_aspect_filters_items(live, fixture_ready, aspect: str, term: str) -> None:
    r = ok(enforce(live, fixture_ready).call("pathway", "get_gene_ontology", {"target_id": PCSK9, "aspect": aspect}))
    assert ids(r.rows("go_terms")) == [term]


def test_phenotype_min_evidence_and_limit_follow_negation(live, fixture_ready) -> None:
    """having (min_evidence) and the limit apply after negated items are removed (F4)."""
    bridge = enforce(live, fixture_ready)
    r = ok(bridge.call("disease", "find_diseases_by_phenotype", {"phenotype_id": F.SEIZURE, "min_evidence": 2}))
    assert [(d["disease_id"], d["evidence_count"]) for d in r.rows("diseases")] == [(F.PHENO["Z"], 2)]
    assert r.obj.get("phenotype_name") == "Seizure"
    r = ok(bridge.call("disease", "find_diseases_by_phenotype", {"phenotype_id": F.SEIZURE, "limit": 1}), "partial")
    assert len(r.rows("diseases")) == 1 and r.header.get("total") == 2


def test_phenotype_limit_keeps_the_best_supported_diseases(live, fixture_ready) -> None:
    """OT-RV3-01: upstream sorts the diseases by evidence count (most first) before its limit, so a limited call
    keeps the best-supported diseases, not the lowest disease ids; the envelope keeps upstream's `count` and
    each row's `disease_name`."""
    bridge, off = enforce(live, fixture_ready), live("off")
    r = ok(bridge.call("disease", "find_diseases_by_phenotype", {"phenotype_id": F.SEIZURE, "limit": 1}), "partial")
    # Y (MONDO_0100002) has one item, Z (MONDO_0100003) two after its negated item is removed
    assert [(d["disease_id"], d["evidence_count"]) for d in r.rows("diseases")] == [(F.PHENO["Z"], 2)]
    assert r.rows("diseases")[0]["disease_name"] == "seizure disorder Z"
    assert r.obj["count"] == 1 and r.header["total"] == 2
    assert r.header["order"].startswith("evidence_count desc, then disease asc")
    full = ok(bridge.call("disease", "find_diseases_by_phenotype", {"phenotype_id": F.SEIZURE}))
    assert [d["disease_id"] for d in full.rows("diseases")] == [F.PHENO["Z"], F.PHENO["Y"]]
    assert full.obj["count"] == 2
    assert {"count", "diseases", "phenotype_name"} <= set(full.obj) and \
        set(full.obj) - {"_vbt"} <= set(off.call("disease", "find_diseases_by_phenotype",
                                                 {"phenotype_id": F.SEIZURE}).obj)


def test_therapeutic_area_list_comes_back_under_upstreams_key(live, fixture_ready) -> None:
    """OT-RV3-06: list_therapeutic_areas=True lists the areas under `therapeutic_areas` with `count`, as upstream
    does; a therapeutic_area call keeps `diseases`."""
    bridge, off = enforce(live, fixture_ready), live("off")
    want = off.call("disease", "find_diseases_by_therapeutic_area", {"list_therapeutic_areas": True}).obj
    r = ok(bridge.call("disease", "find_diseases_by_therapeutic_area", {"list_therapeutic_areas": True}))
    assert "diseases" not in r.obj and sorted(ids(r.rows("therapeutic_areas"))) == \
        sorted(ids(want["therapeutic_areas"])) and want["therapeutic_areas"]
    assert r.obj["count"] == want["count"] == len(want["therapeutic_areas"])


@pytest.mark.parametrize("args", [{"target_a": TP53BP1, "target_b": TP53}, {"target_a": TP53, "target_b": TP53BP1},
                                  {"target_a": TP53BP1}, {"target_b": TP53}])
def test_interaction_pair_is_undirected(live, fixture_ready, args: dict[str, Any]) -> None:
    r = ok(enforce(live, fixture_ready).call("interaction", "search_interactions", args))
    assert [(i["targetA"], i["targetB"]) for i in r.rows("interactions")] == [(TP53, TP53BP1)]


# ---------------------------------------------------------------------------- text arguments and coverage


@pytest.mark.parametrize("server,tool,args,path,col,want", [
    ("genetics", "get_study_metadata", {"trait": "LDL"}, "studies", "studyId", [F.STUDY_BIG]),
    ("genetics", "get_study_metadata", {"trait": "ldl cholesterol"}, "studies", "studyId", [F.STUDY_BIG]),
    ("target", "get_homologues", {"target_id": PCSK9, "species_filter": "mou"}, "homologues", "speciesName",
     ["Mouse"]),
    ("drug", "search_drugs", {"query": "acetylsalicylic"}, "drugs", "id", ["CHEMBL25"]),
])
def test_text_arguments_match_as_text(live, fixture_ready, server, tool, args, path, col, want) -> None:
    """interpreted_as substring is a text match, never an exact filter or a vocabulary value (F2, F10)."""
    r = ok(enforce(live, fixture_ready).call(server, tool, args))
    assert ids(r.rows(path), col) == want


def test_text_miss_is_not_a_covered_absence(live, fixture_ready) -> None:
    r = ok(enforce(live, fixture_ready).call("drug", "search_drugs", {"query": "ASPIRIN tablets"}), "empty")
    assert r.header.get("coverage") == "unknown" and "citable only as an absence" not in str(r.header.get("cite"))


# ---------------------------------------------------------------------------- identifiers and grains


def test_cell_line_name_resolves_to_its_depmap_id(live, fixture_ready) -> None:
    r = ok(enforce(live, fixture_ready).call("functional_genomics", "query_cell_line_dependency",
                                             {"cell_line_name": "A549"}))
    assert {d.get("depmapId") for d in r.rows("dependencies")} == {F.A549}


@pytest.mark.parametrize("so_id", ["SO:0001583", "SO_0001583"])
def test_so_terms_are_ready_in_either_spelling(live, fixture_ready, so_id: str) -> None:
    """so.id stores SO:0001583 (25.09); the SO_ spelling variant consequences use resolves to it."""
    r = ok(enforce(live, fixture_ready).call("pathway", "get_sequence_ontology_term", {"so_id": so_id}))
    assert r.obj.get("label") == "missense_variant"


def test_biosample_lookup_by_id(live, fixture_ready) -> None:
    r = ok(enforce(live, fixture_ready).call("expression", "search_biosample_ontology",
                                             {"biosample_id": "UBERON_0002048"}))
    assert "lung" in str(r.obj)


def test_tahoe_row_grain_is_counted(live, fixture_ready) -> None:
    """result.grain row is the reserved row grain, not a descriptor grain (no service_unavailable)."""
    bridge = enforce(live, fixture_ready)
    r = bridge.call("functional_genomics", "find_drugs_affecting_gene", {"gene_name": F.SELECTIVE})
    assert r.kind != "service_unavailable", r.text[:600]
    r = ok(bridge.call("functional_genomics", "find_drugs_affecting_gene",
                       {"gene_name": F.SELECTIVE, "concentration": F.CONCENTRATIONS[0]}))
    assert {d.get("drug_name") for d in r.rows("top_upregulators") + r.rows("top_downregulators")} == {F.BORTEZOMIB}
