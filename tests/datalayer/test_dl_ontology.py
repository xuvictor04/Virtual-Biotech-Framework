"""Ontology semantics: hierarchy expansion, include_descendants and propagated membership (§11.5; phase 3, F18).

A tiny Open Targets-shaped release: a disease chain MONDO_0000001 > MONDO_0000002 > MONDO_0000003 (with
the descriptor's MONDO_0005148 sentinel beside it), known_drug rows whose ``ancestors`` lists exclude
the term itself, disease-phenotype annotations at every level, and a Reactome hierarchy R-HSA-0000010 >
R-HSA-0000020 > R-HSA-0000030 whose top level has no gene of its own.

* ``include_descendants`` keeps the term itself: ``Eq OR Contains(ancestors)`` on known_drug, and
  ``In(expand(X) ∪ {X})`` through the disease hierarchy elsewhere; a column already propagated over the
  hierarchy (indirect associations) refuses it;
* the closure is built once per hierarchy fingerprint and kept in a sidecar; cycles are detected and
  a term in a cycle is ``not_ready`` (``hierarchy_cycle``); more than ``max_expand`` terms is
  ``too_large`` (``expansion``); depths are returned;
* ``find_genes_in_pathway`` on a top-level Reactome pathway lists the genes of its sub-pathways
  (propagated, disclosed) and the ``mixed`` membership is measured;
* Cell Ontology subtrees follow ``is_a`` only and agree with ``vbt.analysis.ontology``.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from dl_upstream import REPO, needs_arrow

pytestmark = needs_arrow

PCSK9 = "ENSG00000169174"
G2 = "ENSG00000000002"
D = {0: "MONDO_0005148", 1: "MONDO_0000001", 2: "MONDO_0000002", 3: "MONDO_0000003"}
R = {1: "R-HSA-0000010", 2: "R-HSA-0000020", 3: "R-HSA-0000030", 4: "R-HSA-0000040"}


def build(root: Path, *, cycle: bool = False) -> Path:
    import dl_fixtures as F

    ot = root / "ot" / "25.09"
    disease_parents = {D[0]: [], D[1]: [], D[2]: [D[1]], D[3]: [D[2]]}
    F.write_table(ot, "disease", F.table("disease", [
        {"id": d, "name": f"disease {d[-1]}", "parents": ps, "children": [c for c, p in disease_parents.items() if d in p],
         "ancestors": [], "descendants": []} for d, ps in disease_parents.items()]))
    F.write_table(ot, "disease_phenotype", F.table("disease_phenotype", [
        {"disease": D[i], "phenotype": f"HP_000000{i}", "evidence": [{"evidenceType": "IEA", "qualifierNot": False}]}
        for i in (1, 2, 3)]))
    F.write_table(ot, "known_drug", F.table("known_drug", [
        {"drugId": "CHEMBL1", "targetId": PCSK9, "diseaseId": D[1], "phase": 4.0, "ancestors": []},
        {"drugId": "CHEMBL2", "targetId": PCSK9, "diseaseId": D[3], "phase": 2.0, "ancestors": [D[2], D[1]]},
        {"drugId": "CHEMBL3", "targetId": PCSK9, "diseaseId": D[0], "phase": 1.0, "ancestors": []}]))
    reactome_parents = {R[1]: [], R[2]: [R[1]], R[3]: [R[2]], R[4]: [R[1]]}
    if cycle:
        reactome_parents[R[1]] = [R[3]]
    F.write_table(ot, "reactome", F.table("reactome", [
        {"id": r, "label": f"pathway {r[-2]}", "parents": ps, "children": [c for c, p in reactome_parents.items() if r in p],
         "ancestors": [], "descendants": [], "path": []} for r, ps in reactome_parents.items()]))
    pw = lambda *rs: [{"pathwayId": r, "pathway": f"pathway {r[-2]}", "topLevelTerm": "top"} for r in rs]  # noqa: E731
    F.write_table(ot, "target", F.table("target", [
        {"id": PCSK9, "approvedSymbol": "PCSK9", "biotype": "protein_coding", "approvedName": "p",
         "pathways": pw(R[3], R[2]),                 # mixed: the lowest level and its parent
         "go": [{"id": "GO:0000001", "source": "x", "evidence": "IDA", "aspect": "P", "geneProduct": "P1",
                 "ecoId": "ECO"}]},
        {"id": G2, "approvedSymbol": "GENEB", "biotype": "protein_coding", "approvedName": "b", "pathways": pw(R[4])}]))
    return ot


def make_ctx(ot: Path, cache: Path) -> Any:
    from vbt.datalayer.service import ServiceContext
    from vbt.datalayer.settings import DataSettings

    settings = DataSettings.from_dict({"descriptors_dir": str(REPO / "configs" / "data" / "sources"),
                                       "overlays_dir": str(REPO / "configs" / "data" / "overlays"),
                                       "cache_dir": str(cache)}, project_root=REPO)
    return ServiceContext(settings)


@pytest.fixture(scope="module")
def ctx(tmp_path_factory: pytest.TempPathFactory) -> Any:
    tmp = tmp_path_factory.mktemp("ontology")
    ot = build(tmp)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OPEN_TARGETS_DATA_PATH", str(ot))
        yield make_ctx(ot, tmp / "cache")


def call(ctx: Any, name: str, /, **payload: Any) -> dict[str, Any]:
    from vbt.datalayer.service.verbs import load_verbs

    out = load_verbs()[name](ctx, payload)
    json.dumps(out)
    return out


def ok(out: dict[str, Any]) -> dict[str, Any]:
    assert out.get("status") != "tool_error", json.dumps(out)[:1500]
    return out


# ---------------------------------------------------------------------------- include_descendants


def test_include_descendants_keeps_the_term_itself(ctx) -> None:
    from vbt.datalayer.predicate import Contains, Eq, In, Or, evaluate
    from vbt.datalayer.service.verbs.hierarchy import descendant_predicate

    pred, record = descendant_predicate(ctx, "open_targets.known_drug", "diseaseId", D[1])
    assert pred == Or((Eq("diseaseId", D[1]), Contains("ancestors", D[1])))
    assert record["form"] == "eq_or_contains_ancestor" and record["reflexive"] is False
    rows = [{"diseaseId": D[1], "ancestors": []}, {"diseaseId": D[3], "ancestors": [D[2], D[1]]},
            {"diseaseId": D[0], "ancestors": []}]
    assert [evaluate(pred, r) for r in rows] == [True, True, False], "ancestors exclude the term: Eq keeps it"
    pred, record = descendant_predicate(ctx, "open_targets.disease_phenotype", "disease", D[1])
    assert pred == In("disease", (D[1], D[2], D[3]))
    assert record["form"] == "in_expansion" and record["n"] == 2 and record["max_depth"] == 2
    assert record["hierarchy"] == "open_targets.disease"


def test_include_descendants_through_a_derived_tool(ctx, tmp_path) -> None:
    import asyncio

    from test_dl_native_tools import _derived, _gateway

    gw = _gateway(ctx, tmp_path)
    plain = asyncio.run(_derived(gw, "disease", "get_disease_phenotypes", {"disease_id": D[1]}))
    wide = asyncio.run(_derived(gw, "disease", "get_disease_phenotypes", {"disease_id": D[1],
                                                                        "include_descendants": True}))
    assert {r["disease"] for r in plain.obj["phenotypes"]} == {D[1]}
    assert {(r["disease"], r["phenotype"]) for r in wide.obj["phenotypes"]} == {
        (D[1], "HP_0000001"), (D[2], "HP_0000002"), (D[3], "HP_0000003")}


def test_include_descendants_on_search_known_drugs(ctx, tmp_path) -> None:
    """F18: drug.search_known_drugs offers include_descendants; such a call is served derived through the
    known_drug ancestors (Eq OR Contains), keeps the term's own rows, adds the descendants' rows, and records
    the expansion; without it the tool still goes upstream."""
    import asyncio
    import os

    from test_dl_native_tools import _derived, _gateway
    from vbt.datalayer.service import ServiceContext
    from vbt.datalayer.settings import DataSettings

    # the tiny release has no known_drug sentinel rows: readiness sentinels are not this test's subject
    settings = DataSettings.from_dict({"descriptors_dir": str(REPO / "configs" / "data" / "sources"),
                                       "overlays_dir": str(REPO / "configs" / "data" / "overlays"),
                                       "cache_dir": str(tmp_path / "cache"), "readiness": {"sentinels": False}},
                                      project_root=REPO)
    assert os.environ.get("OPEN_TARGETS_DATA_PATH")
    gw = _gateway(ServiceContext(settings), tmp_path)
    wide = asyncio.run(_derived(gw, "drug", "search_known_drugs", {"disease_id": D[1], "include_descendants": True}))
    assert {(r["drugId"], r["diseaseId"]) for r in wide.obj["drugs"]} == {("CHEMBL1", D[1]), ("CHEMBL2", D[3])}
    assert wide.header["served_by"] == "derived"
    records = wide.provenance.to_dict().get("derived") or {}
    assert "_expansion" in records or "expansion" in json.dumps(records), records
    b = gw.catalog.contract("drug", "search_known_drugs").binding
    assert b.serve == "pass" and b.derived_when == ["include_descendants"]   # without it: upstream as before


def test_propagated_over_columns_refuse_include_descendants(ctx) -> None:
    from vbt.datalayer.errors import GatewayError
    from vbt.datalayer.service.verbs.hierarchy import descendant_predicate

    with pytest.raises(GatewayError) as e:
        descendant_predicate(ctx, "open_targets.association_by_overall_indirect", "diseaseId", D[1])
    assert e.value.kind.value == "unsupported_combination" and e.value.payload["reason"] == "propagated_over"


# ---------------------------------------------------------------------------- the closure


def test_expand_returns_depths_and_keeps_a_sidecar(ctx) -> None:
    out = ok(call(ctx, "_expand", id_type="open_targets:reactome_pathway", values=[R[1]]))
    assert [(r["term"], r["depth"]) for r in out["rows"]] == [(R[2], 1), (R[4], 1), (R[3], 2)]
    rec = out["expansion"][0]
    assert out["_vbt"]["expansion"][0]["n"] == 3
    assert rec["n"] == 3 and rec["max_depth"] == 2 and rec["predicates"] == ["is_a"]
    side = Path(rec["closure"])
    assert side.exists() and rec["fingerprint"].replace(":", "_") in str(side)
    data = json.loads(gzip.decompress(side.read_bytes()))
    assert data["parents"][R[3]] == [R[2]] and data["cycles"] == []
    up = ok(call(ctx, "_expand", id_type="open_targets:reactome_pathway", values=["pathway 3"],
                 direction="ancestors", include_self=True))
    assert [(r["term"], r["depth"]) for r in up["rows"]] == [(R[3], 0), (R[2], 1), (R[1], 2)]
    # a fresh data child reads the sidecar instead of rebuilding
    from vbt.datalayer.service.verbs.hierarchy import closure_for

    fresh = make_ctx(Path(ctx.catalog.source("open_targets").root), Path(ctx.settings.cache_dir))
    assert closure_for(fresh, "open_targets:reactome_pathway").sidecar == str(side)


def test_max_expand_is_too_large(ctx) -> None:
    out = call(ctx, "_expand", id_type="open_targets:reactome_pathway", values=[R[1]], max_expand=2)
    assert out["kind"] == "too_large" and out["subkind"] == "expansion", out


def test_cycles_are_detected(tmp_path) -> None:
    ot = build(tmp_path, cycle=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OPEN_TARGETS_DATA_PATH", str(ot))
        cctx = make_ctx(ot, tmp_path / "cache")
        from vbt.datalayer.service.verbs.hierarchy import closure_for

        assert closure_for(cctx, "open_targets:reactome_pathway").cycles == [[R[1], R[2], R[3]]]
        out = call(cctx, "_expand", id_type="open_targets:reactome_pathway", values=[R[2]])
        assert out["kind"] == "not_ready" and out["subkind"] == "hierarchy_cycle", out
        # a term outside the cycle still expands
        assert ok(call(cctx, "_expand", id_type="open_targets:reactome_pathway", values=[R[4]]))["rows"] == []


# ---------------------------------------------------------------------------- propagated membership


def test_reactome_parent_membership(ctx, tmp_path) -> None:
    import asyncio

    from test_dl_native_tools import _derived, _gateway

    direct = call(ctx, "members", table="open_targets.target_pathways", set_id=R[1])
    assert direct["status"] == "tool_error" or direct["rows"] == []
    out = ok(call(ctx, "members", table="open_targets.target_pathways", set_id=R[1], propagate=True))
    rows = {r["member"]: r for r in out["rows"]}
    assert rows[PCSK9]["via"] == R[2] and rows[G2]["via"] == R[4], "the shallowest annotated descendant"
    assert all(r["propagated"] for r in rows.values())
    mixed = out["mixed"]
    assert mixed == {"annotations": 3, "ancestor_annotations": 1, "fraction": 1 / 3,
                     "fingerprint": mixed["fingerprint"]}
    assert any("mixed" in n for n in out["_vbt"]["notes"]), json.dumps(out["_vbt"])
    gw = _gateway(ctx, tmp_path)
    res = asyncio.run(_derived(gw, "pathway", "find_genes_in_pathway", {"pathway_id": R[1]}))
    assert sorted(g["id"] for g in res.obj["genes"]) == sorted([PCSK9, G2]) and res.obj["gene_count"] == 2


# ---------------------------------------------------------------------------- Cell Ontology


def test_cell_ontology_subtree_matches_the_analysis_module(ctx) -> None:
    from vbt.analysis.ontology import load_cell_ontology

    path = REPO / "tests" / "fixtures" / "mini_cl.obo"
    onto = load_cell_ontology(path)
    out = ok(call(ctx, "_expand", id_type="cell_ontology:cell_ontology", values=["CL:0000000"]))
    got = {r["term"] for r in out["rows"]}
    want = {t for t in onto.parents if t != "CL:0000000" and "CL:0000000" in onto.ancestors(t, include_self=False)}
    assert got == want and got
    assert out["expansion"][0]["predicates"] == ["is_a"]
