"""Native enrichment from the member role's enrichment contract (§6.8 S9, §9.4; phase 3, F16).

A tiny Open Targets-shaped release (twelve annotated genes, eight unannotated), a four-pathway Reactome
hierarchy, a go-basic.obo with ``is_a``, ``part_of`` and ``regulates`` edges and a two-set MSigDB GMT,
read through the shipped descriptors. Every p-value and q-value is recomputed here by hand from binomial
coefficients (an oracle independent of the plugins):

* the BH family is every set within the size bounds **counted within the universe**, the zero-overlap
  set included (upstream's ``m`` counted only overlapping sets, so its q-values were too small);
* GO uses an aspect-specific background (genes with an annotation in that aspect);
* annotations propagate to ancestors over ``is_a`` and ``part_of``, never over ``regulates``;
* a gene list that mostly fails to resolve is ``insufficient_resolution``; symbols resolve to genes;
* the statistics block records the test, the correction, the family size and the universe;
* the derived ``get_pathway_enrichment`` and ``get_go_enrichment`` (pathway overlay) answer through the
  real gateway with upstream's field names, ``enrichment_ratio`` recomputed as fold enrichment.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from dl_upstream import REPO, needs_arrow

pytestmark = needs_arrow

PCSK9 = "ENSG00000169174"                                 # G12: the descriptor's readiness sentinel
GENES = [PCSK9 if i == 12 else f"ENSG{i:011d}" for i in range(1, 21)]      # G1..G20
SYMBOL = {g: ("PCSK9" if g == PCSK9 else f"GENE{chr(64 + i)}") for i, g in enumerate(GENES, start=1)}
G = {i: GENES[i - 1] for i in range(1, 21)}

# Reactome: R1 (top) > R2, R3; R4 (top) > R5 (no genes)
PATHWAYS = {"R-HSA-0000002": [1, 2, 3], "R-HSA-0000003": [4, 5], "R-HSA-0000004": [6, 7, 8, 9, 10, 11, 12]}
REACTOME = [("R-HSA-0000001", "Top one", []), ("R-HSA-0000002", "Child two", ["R-HSA-0000001"]),
            ("R-HSA-0000003", "Child three", ["R-HSA-0000001"]), ("R-HSA-0000004", "Top four", []),
            ("R-HSA-0000005", "Child five", ["R-HSA-0000004"])]
# GO annotations: (gene, term, aspect)
GO_ANN = [(1, "GO:0000002", "P"), (2, "GO:0000003", "P"), (3, "GO:0000004", "P"), (7, "GO:0000002", "P"),
          (4, "GO:0000011", "F"), (5, "GO:0000011", "F"), (6, "GO:0000010", "F"), (12, "GO:0000020", "C")]
OBO = """format-version: 1.2
data-version: releases/2026-01-01
ontology: go

[Term]
id: GO:0008150
name: biological_process
namespace: biological_process

[Term]
id: GO:0000001
name: root process
namespace: biological_process

[Term]
id: GO:0000002
name: child by is_a
namespace: biological_process
is_a: GO:0000001 ! root process

[Term]
id: GO:0000003
name: child by part_of
namespace: biological_process
relationship: part_of GO:0000001 ! root process

[Term]
id: GO:0000004
name: regulator
namespace: biological_process
relationship: regulates GO:0000001 ! root process

[Term]
id: GO:0000010
name: root function
namespace: molecular_function

[Term]
id: GO:0000011
name: child function
namespace: molecular_function
is_a: GO:0000010 ! root function
"""
HALLMARK = {"HALLMARK_ONE": ["GENEA", "GENEB", "GENEC", "GENED", "NOTAGENE"],
            "HALLMARK_TWO": ["GENEK", "GENEM", "PCSK9", "GENEN", "GENEO", "GENEP"]}


def build(root: Path) -> dict[str, Path]:
    import dl_fixtures as F

    ot = root / "ot" / "25.09"
    rows = []
    for i, g in enumerate(GENES, start=1):
        pw = [{"pathwayId": p, "pathway": dict((r[0], r[1]) for r in REACTOME)[p], "topLevelTerm": "Top"}
              for p, members in PATHWAYS.items() if i in members]
        go = [{"id": t, "source": "GO_Central", "evidence": "IDA", "aspect": a, "geneProduct": f"P{i:05d}",
               "ecoId": "ECO_0000314"} for gi, t, a in GO_ANN if gi == i]
        rows.append({"id": g, "approvedSymbol": SYMBOL[g], "biotype": "protein_coding", "approvedName": f"gene {i}",
                     "pathways": pw or None, "go": go or None})
    F.write_table(ot, "target", F.table("target", rows))
    F.write_table(ot, "reactome", F.table("reactome", [
        {"id": i, "label": label, "parents": parents, "children": [c for c, _l, ps in REACTOME if i in ps],
         "ancestors": parents, "descendants": [c for c, _l, ps in REACTOME if i in ps], "path": []}
        for i, label, parents in REACTOME]))
    terms = sorted({t for _g, t, _a in GO_ANN} | {"GO:0000001", "GO:0000010", "GO:0008150"})
    F.write_table(ot, "go", F.table("go", [{"id": t, "name": f"term {t[-2:]}"} for t in terms]))
    go = root / "go"
    go.mkdir(parents=True)
    (go / "go-basic.obo").write_text(OBO)
    ms = root / "msigdb"
    ms.mkdir(parents=True)
    (ms / "h.all.v2024.1.Hs.symbols.gmt").write_text(
        "".join(f"{k}\thttps://example.org/{k}\t" + "\t".join(v) + "\n" for k, v in HALLMARK.items()))
    return {"ot": ot, "go": go, "msigdb": ms}


@pytest.fixture(scope="module")
def ctx(tmp_path_factory: pytest.TempPathFactory) -> Any:
    from vbt.datalayer.service import ServiceContext
    from vbt.datalayer.settings import DataSettings

    tmp = tmp_path_factory.mktemp("enrich")
    roots = build(tmp)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OPEN_TARGETS_DATA_PATH", str(roots["ot"]))
        mp.setenv("GO_DATA_PATH", str(roots["go"]))
        mp.setenv("MSIGDB_DATA_PATH", str(roots["msigdb"]))
        settings = DataSettings.from_dict({"descriptors_dir": str(REPO / "configs" / "data" / "sources"),
                                           "overlays_dir": str(REPO / "configs" / "data" / "overlays"),
                                           "cache_dir": str(tmp / "cache")}, project_root=REPO)
        yield ServiceContext(settings)


def call(ctx: Any, name: str, /, **payload: Any) -> dict[str, Any]:
    from vbt.datalayer.service.verbs import load_verbs

    out = load_verbs()[name](ctx, payload)
    json.dumps(out)
    return out


def ok(out: dict[str, Any]) -> dict[str, Any]:
    assert out.get("status") != "tool_error", json.dumps(out)[:1500]
    return out


# ---------------------------------------------------------------------------- the oracle


def sf(k: int, N: int, K: int, n: int) -> float:
    """P(X >= k), X ~ Hypergeom(N, K, n), from binomial coefficients."""
    total = math.comb(N, n)
    return sum(math.comb(K, i) * math.comb(N - K, n - i) for i in range(k, min(K, n) + 1)) / total


def bh(ps: list[float]) -> list[float]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    q = [0.0] * m
    run = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        run = min(run, ps[i] * m / rank)
        q[i] = min(1.0, run)
    return q


def by_set(rows: list[dict[str, Any]], key: str = "set_id") -> dict[str, dict[str, Any]]:
    return {r[key]: r for r in rows}


# ---------------------------------------------------------------------------- Reactome: family and propagation


def test_reactome_full_family_with_zero_overlap_and_propagation(ctx) -> None:
    query = [G[1], G[2], G[4], G[13]]                 # G13 has no pathway: outside the universe
    out = ok(call(ctx, "_enrich", table="open_targets.target", column="pathways", genes=query, min_size=2))
    rows = by_set(out["rows"])
    N, n = 12, 3                                       # annotated genes; query genes among them
    sizes = {"R-HSA-0000001": (5, 3), "R-HSA-0000002": (3, 2), "R-HSA-0000003": (2, 1), "R-HSA-0000004": (7, 0)}
    assert set(rows) == set(sizes), "R-HSA-0000005 has no member and R1 gets its children's genes"
    ps = {s: sf(k, N, K, n) for s, (K, k) in sizes.items()}
    qs = dict(zip(ps, bh(list(ps.values()))))
    for s, (K, k) in sizes.items():
        r = rows[s]
        assert (r["set_size"], r["overlap"]) == (K, k), s
        assert r["pvalue"] == pytest.approx(ps[s], rel=1e-12)
        assert r["fdr"] == pytest.approx(qs[s], rel=1e-12)
        assert r["fold_enrichment"] == (pytest.approx((k / n) / (K / N)) if k else 0.0)
    assert rows["R-HSA-0000004"]["pvalue"] == 1.0 and rows["R-HSA-0000004"]["overlap"] == 0
    # the family counts the zero-overlap set: upstream's m = 3 would understate every q-value
    wrong = dict(zip([s for s in ps if sizes[s][1]], bh([ps[s] for s in ps if sizes[s][1]])))
    assert all(rows[s]["fdr"] >= wrong[s] for s in wrong) and any(rows[s]["fdr"] > wrong[s] for s in wrong)
    assert rows["R-HSA-0000001"]["overlap_members"] == sorted([G[1], G[2], G[4]])
    assert rows["R-HSA-0000002"]["top_level"] == ["R-HSA-0000001"]
    stats = out["statistics"]
    assert stats["test"] == "hypergeom_enrichment" and stats["correction"] == "fdr_bh"
    assert stats["family"]["m"] == {"None": 4} and stats["family"]["include_zero_overlap"] is True
    assert stats["family"]["size_bounds"] == {"min": 2, "max": None, "counted_within": "universe"}
    assert stats["universe"]["sizes"] == {"None": 12} and stats["propagation"]["propagated"] is True
    assert [r["set_id"] for r in out["rows"]] == sorted(rows, key=lambda s: (rows[s]["pvalue"], s))
    assert out["_vbt"]["coverage"] == "covered"


def test_size_bounds_are_counted_within_the_universe(ctx) -> None:
    out = ok(call(ctx, "_enrich", table="open_targets.target", column="pathways", genes=[G[1], G[4]],
                  min_size=3, max_size=5))
    assert set(by_set(out["rows"])) == {"R-HSA-0000001", "R-HSA-0000002"}
    assert out["statistics"]["family"]["m"] == {"None": 2}
    # a caller universe that drops G3 shrinks R2 to two members, below min_size
    uni = [G[i] for i in (1, 2, 4, 5, 6, 7, 8)]
    out = ok(call(ctx, "_enrich", table="open_targets.target", column="pathways", genes=[G[1], G[4]],
                  universe=uni, min_size=3))
    rows = by_set(out["rows"])
    assert "R-HSA-0000002" not in rows and rows["R-HSA-0000001"]["set_size"] == 4
    assert rows["R-HSA-0000001"]["pvalue"] == pytest.approx(sf(2, 7, 4, 2), rel=1e-12)
    assert out["statistics"]["universe"]["override"] is True


def test_direct_only_when_propagation_is_off(ctx) -> None:
    out = ok(call(ctx, "_enrich", table="open_targets.target", column="pathways", genes=[G[1], G[4]], min_size=1,
                  propagate=False))
    rows = by_set(out["rows"])
    assert "R-HSA-0000001" not in rows and out["statistics"]["propagation"] == {"propagated": False}


# ---------------------------------------------------------------------------- GO: aspect backgrounds and predicates


def test_go_background_is_aspect_specific_and_regulates_does_not_propagate(ctx) -> None:
    out = ok(call(ctx, "_enrich", table="open_targets.target", column="go", genes=[G[1], G[3]],
                  scope={"aspect": "biological_process"}, min_size=1))
    rows = by_set(out["rows"])
    # P universe: genes with a P annotation (G1, G2, G3, G7); GO:1 gets G1 (is_a), G7 (is_a), G2 (part_of), not G3
    assert out["statistics"]["universe"]["sizes"] == {"None": 4}
    root = rows["GO:0000001"]
    assert root["set_size"] == 3 and root["overlap_members"] == [G[1]]
    assert G[3] in rows["GO:0000004"]["overlap_members"] and rows["GO:0000004"]["set_size"] == 1
    assert root["pvalue"] == pytest.approx(sf(1, 4, 3, 2), rel=1e-12)
    assert not any(r["set_id"] in ("GO:0000010", "GO:0000011") for r in out["rows"]), "F terms are another family"
    prop = out["statistics"]["propagation"]
    assert prop["hierarchy"] == "gene_ontology.term" and prop["predicates"] == ["is_a", "part_of"]
    assert prop["edges_not_followed"] == {"regulates": 1}
    ps = [r["pvalue"] for r in out["rows"]]
    assert [r["fdr"] for r in out["rows"]] == pytest.approx(bh(ps))
    # without an aspect: one universe and one family per aspect
    both = ok(call(ctx, "_enrich", table="open_targets.target", column="go", genes=[G[1], G[4]], min_size=1))
    assert both["statistics"]["universe"]["sizes"] == {"C": 1, "P": 4, "F": 3}
    assert both["statistics"]["family"]["m"] == {"C": 1, "F": 2, "P": 4}
    f_root = by_set([r for r in both["rows"] if r["scope"] == "F"])["GO:0000010"]
    assert f_root["set_size"] == 3 and f_root["pvalue"] == pytest.approx(sf(1, 3, 3, 1))


# ---------------------------------------------------------------------------- resolution


def test_symbols_resolve_and_mostly_unresolved_lists_are_refused(ctx) -> None:
    out = ok(call(ctx, "_enrich", table="open_targets.target", column="pathways",
                  genes=["GENEA", G[1], "geneb", "ENSG00000000004.3"], min_size=2))
    summary = out["_vbt"]["resolution_summary"]
    assert summary["requested"] == 4 and summary["duplicates"] == [G[1]] and not summary["unresolved"]
    assert by_set(out["rows"])["R-HSA-0000001"]["overlap"] == 3
    bad = call(ctx, "_enrich", table="open_targets.target", column="pathways",
               genes=["GENEA", "NOPE1", "NOPE2", "ENSG00000999999"], min_resolved_fraction=0.9)
    assert bad["status"] == "tool_error" and bad["kind"] == "insufficient_resolution", bad
    assert bad["requested"] == 4 and bad["resolved"] == 1 and set(bad["unresolved"]) >= {"ENSG00000999999"}


def test_msigdb_members_map_to_genes_within_the_protein_coding_universe(ctx) -> None:
    out = ok(call(ctx, "_enrich", table="msigdb.hallmark", genes=["GENEA", "GENEB", "GENEK"], min_size=1))
    rows = by_set(out["rows"])
    N = 20                                             # protein-coding genes of the release
    assert rows["HALLMARK_ONE"]["set_size"] == 4, "NOTAGENE maps to no gene and is not counted"
    assert rows["HALLMARK_ONE"]["pvalue"] == pytest.approx(sf(2, N, 4, 3), rel=1e-12)
    assert rows["HALLMARK_TWO"]["pvalue"] == pytest.approx(sf(1, N, 6, 3), rel=1e-12)
    assert out["statistics"]["members_unmapped"] == 1
    assert out["statistics"]["universe"]["table"] == "open_targets.target"


# ---------------------------------------------------------------------------- derived tools through the gateway


async def test_pathway_tools_are_derived_with_upstream_fields(ctx, tmp_path) -> None:
    from test_dl_native_tools import _derived, _gateway

    gw = _gateway(ctx, tmp_path)
    res = await _derived(gw, "pathway", "get_pathway_enrichment",
                         {"gene_list": ["GENEA", G[2], G[4]], "min_pathway_size": 2, "pvalue_threshold": 1.0})
    obj = res.obj
    rows = {r["pathwayId"]: r for r in obj["enriched_pathways"]}
    assert set(rows) == {"R-HSA-0000001", "R-HSA-0000002", "R-HSA-0000003", "R-HSA-0000004"}
    r1 = rows["R-HSA-0000001"]
    assert r1["pathway_size"] == 5 and r1["overlap_count"] == 3 and "enrichment_ratio" not in r1
    assert r1["pvalue"] == pytest.approx(sf(3, 12, 5, 3), rel=1e-12)
    assert res.header["served_by"] == "derived"
    go = await _derived(gw, "pathway", "get_go_enrichment",
                        {"gene_list": [G[1], G[3]], "go_type": "P", "pvalue_threshold": 1.0})
    terms = {r["go_id"]: r for r in go.obj["enriched_terms"]}
    assert terms["GO:0000001"]["background_count"] == 3
    assert terms["GO:0000001"]["enrichment_ratio"] == pytest.approx((1 / 2) / (3 / 4))
    assert all(r["go_aspect"] == "P" for r in terms.values())
    assert terms["GO:0000001"]["go_name"] == "term 01", "names come from the OT go table"
    assert go.obj["count"] == len(terms)


# ---------------------------------------------------------------------------- DepMap essentiality aggregation

LUNG, BREAST = ("UBERON_0002048", "lung"), ("UBERON_0000310", "breast")
SCREENS = {  # gene -> [(tissue, line, disease, effect)]
    1: [(LUNG, "A1", "Lung Cancer", -1.0), (LUNG, "A2", "Lung Cancer", -0.5), (LUNG, "A3", "Lung Cancer", None),
        (LUNG, "C1", "Lung Adenocarcinoma", -2.0), (BREAST, "B1", "Breast Cancer", 0.1),
        (BREAST, "B2", "Breast Cancer", -0.2)],
    2: [(LUNG, "A1", "Lung Cancer", -0.1), (LUNG, "A2", "Lung Cancer", -0.2), (BREAST, "B1", "Breast Cancer", -0.9),
        (BREAST, "B2", "Breast Cancer", -1.0)],
}


@pytest.fixture(scope="module")
def depmap(ctx: Any) -> Any:
    import dl_fixtures as F

    ot = Path(ctx.catalog.source("open_targets").root)
    rows = []
    for gi, screens in SCREENS.items():
        tissues: dict[str, dict[str, Any]] = {}
        for (tid, tname), line, disease, effect in screens:
            t = tissues.setdefault(tid, {"tissueId": tid, "tissueName": tname, "screens": []})
            t["screens"].append({"depmapId": f"ACH-{line}", "cellLineName": line, "diseaseFromSource": disease,
                                 "geneEffect": effect, "expression": 1.0})
        rows.append({"id": G[gi], "geneEssentiality": [{"isEssential": gi == 1,
                                                        "depMapEssentiality": list(tissues.values())}]})
    F.write_table(ot, "target_essentiality", F.table("target_essentiality", rows))
    return ctx


async def test_essentiality_tools_use_exact_groups_and_the_declared_cutoff(depmap, tmp_path) -> None:
    from test_dl_native_tools import _derived, _gateway

    from vbt.datalayer.errors import GatewayError

    gw = _gateway(depmap, tmp_path)
    res = await _derived(gw, "functional_genomics", "query_gene_essentiality", {"gene_id": "GENEA"})
    tissues = {t["tissue"]["id"]: t for t in res.obj["essentiality_by_tissue"]}
    lung = tissues[LUNG[0]]
    assert lung["num_cell_lines"] == 4 and lung["num_with_effect"] == 3
    assert lung["mean_gene_effect"] == pytest.approx((-1.0 - 0.5 - 2.0) / 3)
    assert lung["essential_fraction"] == 1.0, "-0.5 meets the inclusive cutoff (upstream's strict < missed it)"
    assert tissues[BREAST[0]]["essential_fraction"] == 0.0 and res.obj["is_essential"] is True
    only = await _derived(gw, "functional_genomics", "query_gene_essentiality",
                          {"gene_id": G[1], "disease": "Lung Cancer"})
    (t,) = only.obj["essentiality_by_tissue"]
    assert t["num_cell_lines"] == 3 and t["mean_gene_effect"] == pytest.approx(-0.75), "no Lung Adenocarcinoma lines"
    with pytest.raises(GatewayError) as e:
        await _derived(gw, "functional_genomics", "query_gene_essentiality", {"gene_id": G[1], "disease": "lung"})
    assert e.value.kind.value == "invalid_argument"
    cmp = await _derived(gw, "functional_genomics", "compare_essentiality_across_diseases",
                         {"gene_id": G[1], "diseases": ["Lung Cancer", "Breast Cancer"]})
    rows = cmp.obj["disease_essentiality"]
    assert [(r["disease"], r["num_cell_lines"], r["rank"]) for r in rows] == [("Lung Cancer", 2, 1),
                                                                            ("Breast Cancer", 2, 2)]
    assert cmp.obj["num_diseases_compared"] == 2
    ess = await _derived(gw, "functional_genomics", "find_essential_genes", {"disease": "Lung Cancer",
                                                                            "min_cell_lines": 2})
    assert [g["gene_id"] for g in ess.obj["genes"]] == [G[1]] and ess.obj["genes"][0]["essential_fraction"] == 1.0
    assert ess.obj["num_results"] == 1
    sel = await _derived(gw, "functional_genomics", "find_selective_dependencies",
                         {"target_disease": "Breast Cancer", "comparison_disease": "Lung Cancer", "min_cell_lines": 2})
    (g,) = sel.obj["genes"]
    assert g["gene_id"] == G[2] and g["effect_difference"] == pytest.approx(-0.95 + 0.15)


# ---------------------------------------------------------------------------- tissue specificity (unit guard)


async def test_tissue_specific_genes_compare_within_one_unit(ctx, tmp_path) -> None:
    import dl_fixtures as F
    from test_dl_native_tools import _derived, _gateway

    def tissue(code: str, label: str, value: float | None, unit: str) -> dict[str, Any]:
        return {"efo_code": code, "label": label, "rna": {"value": value, "zscore": 1, "level": 1, "unit": unit}}

    liver, lung, brain, gut = ("UBERON_0002107", "liver"), ("UBERON_0002048", "lung"), ("UBERON_0000955", "brain"), \
        ("UBERON_0000160", "intestine")
    rows = [
        {"id": G[1], "tissues": [tissue(*liver, 100.0, "TPM"), tissue(*lung, 10.0, "TPM"), tissue(*brain, 20.0, "TPM"),
                                 tissue(*gut, 5000.0, "")]},            # the blank-unit value is never pooled
        {"id": G[2], "tissues": [tissue(*liver, 8.0, "TPM"), tissue(*lung, 0.0, "TPM"), tissue(*brain, 0.0, "TPM")]},
        {"id": G[3], "tissues": [tissue(*liver, 12.0, "TPM"), tissue(*lung, 10.0, "TPM"), tissue(*brain, 11.0, "TPM")]},
        {"id": G[4], "tissues": [tissue(*liver, 50.0, ""), tissue(*lung, 1.0, "")]},
    ]
    F.write_table(Path(ctx.catalog.source("open_targets").root), "expression", F.table("expression", rows))
    gw = _gateway(ctx, tmp_path)
    res = await _derived(gw, "expression", "find_tissue_specific_genes",
                         {"output_path": "liver.csv", "tissue": "Liver", "fold_change_threshold": 2.0})
    genes = res.obj["top_genes"]
    assert [g["gene_id"] for g in genes] == [G[2], G[1]], "only-in-target first, then fold change; G3 is not specific"
    assert genes[0]["fold_change"] is None and genes[0]["expressed_only_in_target"] is True, "no 999.9 sentinel"
    assert genes[1]["fold_change"] == pytest.approx(100.0 / 15.0) and genes[1]["median_other_tissues"] == 15.0
    assert all(g["unit"] == "TPM" for g in genes), "G4 (blank unit) is excluded, not compared"
    assert res.obj["num_results"] == 2


# ---------------------------------------------------------------------------- the DepMap matrix through aggregate


def test_depmap_matrix_essential_fraction_uses_the_gene_effect_cutoff(tmp_path, monkeypatch) -> None:
    """The P12 ``aggregate`` verb over the DepMap long view: with ``gene_effect`` registered the measure's
    declared cutoff (<= -0.5, inclusive) defines ``essential_fraction``; a NaN cell is excluded and counted;
    groups below ``min_n`` are dropped."""
    import yaml
    from test_dl_native_tools import _ctx, _depmap_descriptor, _write_depmap

    (tmp_path / "data").mkdir()
    (tmp_path / "sources").mkdir()
    _write_depmap(tmp_path / "data")
    (tmp_path / "sources" / "depmap.yaml").write_text(yaml.safe_dump(_depmap_descriptor(), sort_keys=False))
    monkeypatch.setenv("DEPMAP_DATA_PATH", str(tmp_path / "data"))
    dctx = _ctx(tmp_path, tmp_path / "sources")
    assert dctx.statistic("gene_effect") is not None
    out = ok(call(dctx, "aggregate", table="depmap.gene_effect", where={"entrez_id": "TP53"},
                  group_by=["OncotreeLineage"], measure="gene_effect", how="essential_fraction"))
    rows = {r["OncotreeLineage"]: r for r in out["rows"]}
    assert rows["Skin"]["essential_fraction_gene_effect"] == 1.0 and rows["Skin"]["n"] == 1
    assert rows["Lung"]["essential_fraction_gene_effect"] == 0.0 and rows["Lung"]["n"] == 2
    assert rows["Lung"]["n_unknown"] == 1
    rpl3 = ok(call(dctx, "aggregate", table="depmap.gene_effect", where={"entrez_id": "RPL3"},
                   group_by=["OncotreeLineage"], measure="gene_effect", how="essential_fraction", min_n=2))
    assert [(r["OncotreeLineage"], r["essential_fraction_gene_effect"]) for r in rpl3["rows"]] == [("Lung", 1.0)]
    assert rpl3["_vbt"]["groups_below_min_n"] == 1
