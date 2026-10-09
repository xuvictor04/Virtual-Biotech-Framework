"""Native role-derived tools of the data child (§10.6, F14) on fixtures.

* Every public verb (``resolve``, ``describe``, ``lookup``, ``find``, ``search``, ``vocab``, ``members``,
  ``aggregate``, ``similar``, ``neighbors``) is registered under its own name (``mcp__data__<verb>``) and
  answers on the OT fixture through the shipped descriptors; ``where`` and ``key`` entries resolve
  like overlay arguments (symbol -> Ensembl, unknown -> ``not_found``, wrong vocabulary value ->
  ``invalid_argument`` with valid values).
* ``find`` on a DepMap-shaped gene-effect matrix: a symbol and an Ensembl ID (through a declared
  crosswalk) find the gene's column, an unknown gene is ``not_found`` and a known but unscreened gene
  is ``empty`` with ``coverage: not_covered``; the shipped DepMap descriptor resolves symbols from the
  matrix header itself.
* ``aggregate`` by lineage (row attributes joined through ``attributes_from``) with ``min_n``, nulls
  excluded and counted, per-group ``n``.
* ``similar``: the anchor is excluded from candidates and total; a zero-norm candidate is excluded and
  counted; a zero-norm anchor has no defined similarity; ties by key.
* The target-profile view reports a status per section: with ``known_drug`` deleted that section is
  ``not_ready`` (no count), the others ``ok`` or ``empty``; ``serve_sections`` marks it unavailable.
* ``compare_direct_indirect`` derived through the real gateway on the CT-5 fixture:
  ``unique_to_direct_count == 0`` on the complete key sets, ``limit`` bounding only the rows.
* ``derive.tools.native_tools`` lists ``mcp__data__*`` with table enums of ready tables, without
  ``expose.native: false`` tables and without tables withheld from the named agent.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import pytest

from dl_upstream import REPO, needs_arrow

pytestmark = needs_arrow

PUBLIC = ("resolve", "describe", "lookup", "find", "search", "vocab", "members", "aggregate", "similar", "neighbors",
          "expand", "enrich")


def _ctx(tmp: Path, sources: Path, *, overlays: Path | None = None) -> Any:
    from vbt.datalayer.service import ServiceContext
    from vbt.datalayer.settings import DataSettings

    ov = overlays or (tmp / "no-overlays")
    ov.mkdir(exist_ok=True)
    settings = DataSettings.from_dict({"descriptors_dir": str(sources), "overlays_dir": str(ov),
                                       "cache_dir": str(tmp / "cache")}, project_root=REPO)
    return ServiceContext(settings)


def call(ctx: Any, name: str, /, **payload: Any) -> dict[str, Any]:
    from vbt.datalayer.service.verbs import load_verbs

    out = load_verbs()[name](ctx, payload)
    json.dumps(out)                                    # every answer is JSON on the wire
    return out


def err(out: dict[str, Any], kind: str) -> dict[str, Any]:
    assert out.get("status") == "tool_error" and out.get("kind") == kind, out
    return out


def hdr(out: dict[str, Any]) -> dict[str, Any]:
    assert out.get("status") != "tool_error", out
    return out["_vbt"]


# ---------------------------------------------------------------------------- the OT fixture


@pytest.fixture(scope="module")
def ot(ot_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Any:
    tmp = tmp_path_factory.mktemp("native-ot")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OPEN_TARGETS_DATA_PATH", str(ot_root))
        ctx = _ctx(tmp, REPO / "configs" / "data" / "sources", overlays=REPO / "configs" / "data" / "overlays")
        yield ctx


def test_public_verbs_are_registered_under_their_names() -> None:
    from vbt.datalayer.ipc import VERB_SERVE
    from vbt.datalayer.service.verbs import load_verbs
    from vbt.datalayer.service.verbs.public import PUBLIC_VERBS

    verbs = load_verbs()
    assert PUBLIC_VERBS == PUBLIC and set(PUBLIC) <= set(verbs)
    assert {"_aggregate", "_similar", "_compare", "_view"} <= set(verbs)
    assert verbs[VERB_SERVE].__module__.endswith(".serve"), "the phase-2 verbs are routed by serve.py itself"


def test_find_resolves_where_like_a_bound_argument(ot) -> None:
    import dl_fixtures as F

    out = call(ot, "find", table="open_targets.known_drug", where={"targetId": "PCSK9"}, limit=5)
    h = hdr(out)
    assert h["resolved"] == {"where.targetId": f"PCSK9 -> {F.PCSK9} (label_exact:approvedSymbol)"}
    oracle = F.oracle_known_drug_topk(F.Path(ot.catalog.source("open_targets").root), F.PCSK9, 5)
    assert [tuple(r[k] for k in F.KNOWN_DRUG_KEY) for r in out["rows"]] == oracle
    assert h["total"] == 37 and h["truncated"] is True and h["status"] == "ok"
    by_id = call(ot, "find", table="open_targets.known_drug", where={"targetId": F.PCSK9}, limit=5)
    assert by_id["rows"] == out["rows"]
    unknown = err(call(ot, "find", table="open_targets.known_drug", where={"targetId": F.UNKNOWN_GENE}), "not_found")
    assert unknown["argument"] == "where.targetId" and unknown["tried"]
    err(call(ot, "find", table="open_targets.known_drug", where={"nope": 1}), "invalid_argument")
    bad = err(call(ot, "find", table="open_targets.association_by_datasource_direct",
                   where={"datasourceId": "gwas_catalog"}), "invalid_argument")
    assert bad["valid_values"] == ["eva", "gwas_credible_sets"]
    err(call(ot, "find", table="open_targets.known_drug", rank_by="drugId"), "invalid_argument")
    err(call(ot, "find", table="open_targets.known_drug", limit=0), "invalid_argument")
    err(call(ot, "find", table="nowhere.table"), "invalid_argument")


def test_find_empty_carries_coverage(ot) -> None:
    import dl_fixtures as F

    out = call(ot, "find", table="open_targets.mouse_phenotype", where={"targetFromSourceId": F.TP53})
    h = hdr(out)
    assert out["rows"] == [] and h["status"] == "empty" and h["coverage"] == "unknown" and h["coverage_statement"]


def test_lookup_search_vocab_resolve_describe(ot) -> None:
    import dl_fixtures as F

    rec = call(ot, "lookup", table="open_targets.target", key={"id": "pcsk9"})
    assert rec["rows"][0]["id"] == F.PCSK9 and "label_casefold" in hdr(rec)["resolved"]["where.id"]
    found = call(ot, "search", table="open_targets.target", text="TP53", limit=3)
    assert found["rows"][0]["id"] == F.TP53 and found["rows"][0]["match"] == "exact"
    v = call(ot, "vocab", table="open_targets.association_by_datasource_direct", column="datasourceId")
    assert v["rows"] == ["eva", "gwas_credible_sets"] and v["complete"] is True
    r = call(ot, "resolve", id_type="open_targets:ensembl_gene", values=["NARC1", F.UNKNOWN_GENE])
    assert [x["status"] for x in r["rows"]] == ["resolved", "not_found"]
    assert r["rows"][0]["canonical"] == F.PCSK9 and r["rows"][0]["rule"] == "synonym:alias"
    d = call(ot, "describe", source="open_targets", table="interaction")
    assert d["kind"] == "edges" and "neighbors" in d["verbs"] and d["columns"]["targetA"]["role"] == "endpoint"
    # the operators a `where` entry takes per column: the listing names describe instead of carrying the map
    assert d["columns"]["targetA"]["ops"] == ["eq", "in", "ne"] and "ge" in d["columns"]["scoring"]["ops"]
    listing = call(ot, "describe", source="open_targets")
    assert any(t["table"] == "open_targets.target_go" and t["item_table_of"] == "open_targets.target"
               for t in listing["tables"])


def test_neighbors_one_hop_on_either_side(ot) -> None:
    import dl_fixtures as F

    out = call(ot, "neighbors", table="open_targets.interaction", node="PCSK9", limit=100)
    rows = out["rows"]
    oracle = F.oracle_interactions(F.Path(ot.catalog.source("open_targets").root))
    assert len(rows) == len(oracle) == hdr(out)["total"]
    assert all(F.PCSK9 in (r["targetA"], r["targetB"]) and r["partner"] != F.PCSK9 or r["targetA"] == r["targetB"]
               for r in rows)
    assert any(r["targetB"] == F.PCSK9 for r in rows), "matched on side B too"
    two = call(ot, "neighbors", table="open_targets.interaction", node="PCSK9", hops=2, limit=100)
    nodes = {n["node"]: n for n in two["nodes"]}                       # phase 3: a deterministic network
    assert hdr(two)["network"]["hops"] == 2 and nodes[F.PCSK9]["is_seed"]
    assert {r["partner"] for r in rows if r["partner"] != F.PCSK9} <= set(nodes)
    err(call(ot, "neighbors", table="open_targets.interaction", node="PCSK9", hops=5), "invalid_argument")


def test_withheld_and_unexposed_tables_are_refused(ot) -> None:
    out = err(call(ot, "find", table="zenodo_vbt.clinical_trial_labels", agent="trial-annotator"), "invalid_argument")
    assert out["subkind"] in ("withheld", "not_exposed")


# ---------------------------------------------------------------------------- DepMap matrix

GENES = (("RPL3", "6122"), ("TP53", "7157"), ("A1BG", "1"), ("SEPTIN9", "10801"))
MODELS = (("ACH-000001", "Lung"), ("ACH-000002", "Lung"), ("ACH-000003", "Skin"), ("ACH-000004", "Lung"))
EFFECT = ((-1.8, 0.1, None, -0.2), (-1.2, float("nan"), 0.05, -0.4), (-2.1, -0.9, 0.0, None),
          (-0.6, -0.3, 0.2, -0.1))
#: the gene universe (an HGNC-like crosswalk): BRCA1 is a gene but not a column of the matrix
HGNC = (("6122", "RPL3", "ENSG00000174748"), ("7157", "TP53", "ENSG00000141510"), ("1", "A1BG", "ENSG00000121410"),
        ("10801", "SEPTIN9", "ENSG00000184640"), ("672", "BRCA1", "ENSG00000012048"))


def _write_depmap(root: Path) -> None:
    lines = [",".join(["ModelID", *(f"{s} ({e})" for s, e in GENES)])]
    for (model, _l), vals in zip(MODELS, EFFECT):
        lines.append(",".join([model, *("" if v is None else ("nan" if v != v else repr(v)) for v in vals)]))
    (root / "CRISPRGeneEffect.csv").write_text("\n".join(lines) + "\n")
    model_cols = ["ModelID", "CellLineName", "StrippedCellLineName", "OncotreeLineage", "OncotreePrimaryDisease"]
    rows = [",".join(model_cols)] + [f"{m},L{m[-1]},L{m[-1]},{lin},D" for m, lin in MODELS]
    (root / "Model.csv").write_text("\n".join(rows) + "\n")
    (root / "Genes.csv").write_text("entrez_id,symbol,ensembl_id\n" + "".join(f"{e},{s},{g}\n" for e, s, g in HGNC))


def _depmap_descriptor() -> dict[str, Any]:
    """The shipped DepMap gene-effect matrix and models, with the gene universe of an HGNC-like table (the
    shipped file names the matrix header as the universe until an HGNC source arrives)."""
    import yaml

    d = yaml.safe_load((REPO / "configs" / "data" / "sources" / "depmap.yaml").read_text())
    d["root"] = "${DEPMAP_DATA_PATH}"
    d["strict"] = False
    d["tables"] = {k: v for k, v in d["tables"].items() if k in ("gene_effect", "model")}
    d["tables"]["gene_effect"].pop("sentinels", None)
    model = d["tables"]["model"]
    model["columns"] = {k: v for k, v in model["columns"].items()
                        if k in ("ModelID", "CellLineName", "StrippedCellLineName", "OncotreeLineage",
                                 "OncotreePrimaryDisease")}
    model.pop("sentinels", None)
    d["tables"]["genes"] = {"kind": "crosswalk", "path": "Genes.csv", "grain": "one gene",
                            "key": {"columns": ["entrez_id"]},
                            "coverage": {"statement": "HGNC protein-coding genes", "absence_means": "absent"},
                            "columns": {"entrez_id": {"role": "identifier", "id_type": "ncbi_gene", "self": True},
                                        "symbol": {"role": "label", "of": "entrez_id"},
                                        "ensembl_id": {"role": "identifier", "id_type": "ensembl"}}}
    d["id_types"]["ncbi_gene"] = {"plugin": "ncbi_gene", "options": {"input_requires_prefix": True},
                                  "universe": "genes.entrez_id", "resolve_via": ["genes.symbol"],
                                  "crosswalks": [{"name": "hgnc_ensembl", "table": "genes", "from": "ensembl_id",
                                                  "to": "entrez_id"}],
                                  "maps_to": [{"id_type": "depmap:ensembl", "via": "hgnc_ensembl"}]}
    d["id_types"]["ensembl"] = {"plugin": "ensembl_gene", "universe": "genes.ensembl_id"}
    return d


@pytest.fixture(scope="module")
def depmap(tmp_path_factory: pytest.TempPathFactory) -> Any:
    import yaml

    tmp = tmp_path_factory.mktemp("native-depmap")
    (tmp / "data").mkdir()
    (tmp / "sources").mkdir()
    _write_depmap(tmp / "data")
    (tmp / "sources" / "depmap.yaml").write_text(yaml.safe_dump(_depmap_descriptor(), sort_keys=False))
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DEPMAP_DATA_PATH", str(tmp / "data"))
        yield _ctx(tmp, tmp / "sources")


def _cells(out: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [(r["ModelID"], r["entrez_id"], r["gene_effect"]) for r in out["rows"]]


def test_find_on_the_depmap_matrix_by_symbol_ensembl_unknown_and_unscreened(depmap) -> None:
    by_symbol = call(depmap, "find", table="depmap.gene_effect", where={"entrez_id": "TP53"}, limit=10)
    h = hdr(by_symbol)
    assert h["resolved"]["where.entrez_id"].startswith("TP53 -> 7157")
    # ranked by gene_effect asc (the descriptor's rank), the NaN cell last (not measured: null, never 0)
    assert _cells(by_symbol) == [("ACH-000003", "7157", -0.9), ("ACH-000004", "7157", -0.3),
                                 ("ACH-000001", "7157", 0.1), ("ACH-000002", "7157", None)]
    assert by_symbol["rows"][0]["symbol"] == "TP53" and by_symbol["rows"][0]["OncotreeLineage"] == "Skin"
    by_ensembl = call(depmap, "find", table="depmap.gene_effect", where={"entrez_id": "ENSG00000141510"}, limit=10)
    assert _cells(by_ensembl) == _cells(by_symbol)
    assert "crosswalk:hgnc_ensembl" in hdr(by_ensembl)["resolved"]["where.entrez_id"]
    assert _cells(call(depmap, "find", table="depmap.gene_effect", where={"entrez_id": "NCBIGene:7157"},
                       limit=10)) == _cells(by_symbol)
    for gene in ("NCBIGene:999999", "ENSG00000999999"):
        unknown = err(call(depmap, "find", table="depmap.gene_effect", where={"entrez_id": gene}), "not_found")
        assert unknown["argument"] == "where.entrez_id" and unknown["tried"]
    # no accepted kind reads a bare word that is no symbol: rejected, never an empty answer
    err(call(depmap, "find", table="depmap.gene_effect", where={"entrez_id": "NOTAGENE1"}), "invalid_argument")
    unscreened = call(depmap, "find", table="depmap.gene_effect", where={"entrez_id": "BRCA1"})
    h = hdr(unscreened)
    assert unscreened["rows"] == [] and h["status"] == "empty" and h["coverage"] == "not_covered"
    assert any("672" in n for n in h["notes"])
    cell = call(depmap, "find", table="depmap.gene_effect", where={"entrez_id": "RPL3", "ModelID": "ACH-000001"})
    assert _cells(cell) == [("ACH-000001", "6122", -1.8)]
    deps = call(depmap, "find", table="depmap.gene_effect", where={"gene_effect": {"le": -0.5}}, limit=100)
    assert {(m, e) for m, e, _v in _cells(deps)} == {("ACH-000001", "6122"), ("ACH-000002", "6122"),
                                                     ("ACH-000003", "6122"), ("ACH-000003", "7157"),
                                                     ("ACH-000004", "6122")}
    assert hdr(deps)["total"] == 5


def test_shipped_depmap_descriptor_resolves_symbols_from_the_header(tmp_path, monkeypatch) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "sources").mkdir()
    _write_depmap(tmp_path / "data")
    shutil.copy(REPO / "configs" / "data" / "sources" / "depmap.yaml", tmp_path / "sources" / "depmap.yaml")
    monkeypatch.setenv("DEPMAP_DATA_PATH", str(tmp_path / "data"))
    ctx = _ctx(tmp_path, tmp_path / "sources")
    out = call(ctx, "find", table="depmap.gene_effect", where={"entrez_id": "SEPTIN9"}, limit=10)
    assert hdr(out)["resolved"]["where.entrez_id"] == "SEPTIN9 -> 10801 (label_exact:symbol)"
    assert len(out["rows"]) == 4
    # the header is the universe here: a gene that is not a column is not a gene of this source
    err(call(ctx, "find", table="depmap.gene_effect", where={"entrez_id": "NCBIGene:672"}), "not_found")


def test_aggregate_by_lineage_with_min_n(depmap) -> None:
    out = call(depmap, "aggregate", table="depmap.gene_effect", where={"entrez_id": "TP53"},
               group_by=["OncotreeLineage"], measure="gene_effect", how="mean", min_n=2)
    h = hdr(out)
    # Lung: ACH-000001 0.1, ACH-000002 NaN (excluded and counted), ACH-000004 -0.3 -> n 2; Skin: n 1 < min_n
    assert out["rows"] == [{"OncotreeLineage": "Lung", "mean_gene_effect": pytest.approx(-0.1), "n": 2,
                            "n_unknown": 1}]
    assert h["excluded_unknown"] == {"gene_effect": 1} and h["groups_below_min_n"] == 1
    assert any("fewer than 2" in n for n in h["notes"])
    counts = call(depmap, "aggregate", table="depmap.gene_effect", group_by=["OncotreeLineage"], how="count")
    assert {r["OncotreeLineage"]: r["count"] for r in counts["rows"]} == {"Lung": 12, "Skin": 4}
    err(call(depmap, "aggregate", table="depmap.gene_effect", group_by=["nope"]), "invalid_argument")
    err(call(depmap, "aggregate", table="depmap.gene_effect", group_by=["OncotreeLineage"], how="mean"),
        "invalid_argument")


def test_aggregate_over_facts(ot) -> None:
    out = call(ot, "aggregate", table="open_targets.known_drug", where={"targetId": "PCSK9"}, group_by=["phase"],
               how="count")
    rows = {r["phase"]: r["count"] for r in out["rows"]}
    assert sum(rows.values()) == 37 and rows[4.0] == 5


# ---------------------------------------------------------------------------- similar

VECTORS = {"A": [1.0, 0.0], "B": [0.9, 0.1], "C": [0.0, 1.0], "D": [0.0, 0.0], "E": [0.9, 0.1], "Z": [0.0, 0.0]}


@pytest.fixture(scope="module")
def vectors(tmp_path_factory: pytest.TempPathFactory) -> Any:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml

    tmp = tmp_path_factory.mktemp("native-vectors")
    (tmp / "data").mkdir()
    (tmp / "sources").mkdir()
    words = sorted(VECTORS)
    pq.write_table(pa.table({"word": words, "category": ["x"] * len(words),
                             "vector": pa.array([VECTORS[w] for w in words], pa.list_(pa.float64()))}),
                   tmp / "data" / "vectors.parquet")
    desc = {"schema": "vbt.datasource/1", "source": "emb", "title": "Embeddings", "root": str(tmp / "data"),
            "release": {"expect": "1", "from": "literal"},
            "defaults": {"format": "parquet", "layout": "single_file"},
            "id_types": {"word": {"plugin": "local_key", "options": {"canonical": "^[A-Z]$"},
                                  "universe": "vec.word"}},
            "tables": {"vec": {"kind": "vectors", "path": "vectors.parquet", "grain": "one word vector",
                               "key": {"columns": ["word"]},
                               "coverage": {"statement": "words of the model", "absence_means": "absent"},
                               "columns": {"word": {"role": "identifier", "id_type": "word", "self": True},
                                           "category": {"role": "category", "vocab": ["x"]},
                                           "vector": {"role": "vector"}}}}}
    (tmp / "sources" / "emb.yaml").write_text(yaml.safe_dump(desc, sort_keys=False))
    return _ctx(tmp, tmp / "sources")


def test_similar_excludes_the_anchor_and_non_finite_scores(vectors) -> None:
    out = call(vectors, "similar", table="emb.vec", anchor="A", top_k=2)
    h = hdr(out)
    # B and E tie: ordered by key; the anchor A is neither a candidate nor counted; D and Z have norm 0
    assert [r["word"] for r in out["rows"]] == ["B", "E"]
    assert out["rows"][0]["similarity"] == pytest.approx(0.9 / math.sqrt(0.82))
    assert h["total"] == 3 and h["excluded_unknown"] == {"similarity": 2} and h["truncated"] is True
    err(call(vectors, "similar", table="emb.vec", anchor="Q"), "not_found")
    undefined = err(call(vectors, "similar", table="emb.vec", anchor="D"), "invalid_argument")
    assert undefined["subkind"] == "undefined"


def test_serve_similar_answers_derived_requests(vectors) -> None:
    out = call(vectors, "_serve", table="emb.vec", verb="similar", anchor={"column": "word", "value": "C"}, limit=5)
    assert [r["word"] for r in out["rows"]] == ["B", "E", "A"]
    assert out["excluded_unknown"] == {"similarity": 2} and out["total"] == 3 and out["row_keys"][0] == ["B"]


def test_serve_similar_scores_a_pair_of_anchors(vectors) -> None:
    """Every anchor of the call reaches ``_serve``: two anchors are the cosine of the pair (the repair of
    compute_entity_similarity), with each anchor's columns under its argument name."""
    pair = [{"name": "entity_a", "column": "word", "value": "A"}, {"name": "entity_b", "column": "word", "value": "C"}]
    out = call(vectors, "_serve", table="emb.vec", verb="similar", anchor=pair[0], anchors=pair, columns=["category"])
    a, c = VECTORS["A"], VECTORS["C"]
    cos = sum(x * y for x, y in zip(a, c)) / (math.hypot(*a) * math.hypot(*c))
    assert out["rows"] == [{"entity_a": "A", "entity_a_category": "x", "entity_b": "C", "entity_b_category": "x",
                            "similarity": pytest.approx(cos)}] and out["total"] == 1
    missing = call(vectors, "_serve", table="emb.vec", verb="similar", anchors=[pair[0], {**pair[1], "value": "Q"}])
    assert missing["rows"] == [] and missing["total"] == 0
    from vbt.datalayer.errors import GatewayError

    with pytest.raises(GatewayError) as zero:                # a zero-norm anchor has no defined cosine
        call(vectors, "_serve", table="emb.vec", verb="similar", anchors=[pair[0], {**pair[1], "value": "D"}])
    assert zero.value.kind.value == "invalid_argument" and zero.value.envelope()["subkind"] == "undefined"


# ---------------------------------------------------------------------------- views


def test_view_sections_have_their_own_status(ot_root: Path, tmp_path) -> None:
    import dl_fixtures as F

    root = F.copy_fixture(ot_root, tmp_path / "ot" / "25.09")
    F.delete_table(root, "known_drug")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OPEN_TARGETS_DATA_PATH", str(root))
        ctx = _ctx(tmp_path, REPO / "configs" / "data" / "sources", overlays=REPO / "configs" / "data" / "overlays")
        out = call(ctx, "_view", server="target", tool="get_comprehensive_target_profile",
                   args={"target_id": "PCSK9"})
        secs = out["sections"]
        assert out["record"]["id"] == F.PCSK9 and out["_vbt"]["status"] == "partial"
        kd = secs["known_drugs"]
        assert kd["status"] == "not_ready" and "known_drug" in kd["reason"]
        assert "total" not in kd and "rows" not in kd, "no fabricated zero count"
        assert secs["pathways"]["status"] == "ok" and secs["pathways"]["total"] >= 1
        assert secs["mouse_phenotypes"]["status"] == "ok"
        assert secs["adverse_events"]["status"] == "ok"
        assert out["_vbt"]["sections"]["known_drugs"] == "not_ready"
        tp53 = call(ctx, "_view", server="target", tool="get_comprehensive_target_profile",
                    args={"target_id": "TP53"})
        assert tp53["sections"]["adverse_events"]["status"] == "empty"
        assert tp53["sections"]["adverse_events"]["coverage"] == "unknown"
        err(call(ctx, "_view", server="target", tool="get_comprehensive_target_profile",
                 args={"target_id": F.UNKNOWN_GENE}), "not_found")
        from vbt.datalayer.service.verbs.views import serve_sections

        sec = {"table": "open_targets.known_drug", "key": {"targetId": F.PCSK9}}
        got = serve_sections(ctx, {"known_drugs": sec}, {})
        assert got["known_drugs"]["status"] == "not_ready" and got["known_drugs"]["_vbt_unavailable"]
        # the same view declared in the descriptor (views.target_profile): the argument resolves like the
        # identifier column it keys, and the sections carry the same statuses
        desc = call(ctx, "_view", view="open_targets.target_profile", args={"target_id": "PCSK9"})
        assert desc["_vbt"]["status"] == "partial" and desc["sections"]["target"]["record"]["id"] == F.PCSK9
        assert {n: s["status"] for n, s in desc["sections"].items() if n != "target"} == \
            {n: s["status"] for n, s in secs.items()}
        err(call(ctx, "_view", view="open_targets.target_profile", args={"target_id": F.UNKNOWN_GENE}), "not_found")


# ---------------------------------------------------------------------------- through the real gateway


class InProcessBridge:
    """What the gateway needs from ``MCPBridge``, with the data child's verbs run in this process."""

    def __init__(self, ctx: Any) -> None:
        from vbt.datalayer.service.verbs import load_verbs

        self.ctx = ctx
        self.verbs = load_verbs()
        self.calls: list[tuple[str, str]] = []

    async def call_raw(self, server: str, tool: str, args: dict[str, Any]) -> Any:
        assert server == "data", f"no upstream server in these tests ({server}.{tool})"
        self.calls.append((server, tool))
        return json.dumps(self.verbs[tool](self.ctx, args.get("request") or {}), default=str)

    def status(self) -> dict[str, Any]:
        return {}

    async def recycle(self, server: str, wait_s: float = 30.0) -> bool:
        return True

    def _emit(self, kind: str, **data: Any) -> None:
        pass


def _gateway(ctx: Any, tmp: Path) -> Any:
    from vbt.datalayer.gateway import DataGateway

    out = tmp / "out"
    out.mkdir(parents=True, exist_ok=True)
    gw = DataGateway(ctx.settings, ctx.catalog, ctx.registry, run={"mcp_output_dir": str(out)})
    gw.bind_bridge(InProcessBridge(ctx))
    return gw


async def _derived(gw: Any, server: str, tool: str, args: dict[str, Any]) -> Any:
    plan = await gw.prepare(server, tool, args, None)
    assert plan.route == "derived", plan.route
    return await gw.finish(plan, None)


async def test_compare_direct_indirect_derived_on_the_ct5_fixture(ot, tmp_path) -> None:
    import dl_fixtures as F

    gw = _gateway(ot, tmp_path)
    res = await _derived(gw, "association", "compare_direct_indirect",
                         {"target_id": F.T, "limit": 3, "output_path": "y"})
    obj = res.obj
    # direct is a subset of indirect: nothing is direct-only, whatever the limit
    assert obj["counts"]["unique_to_direct_count"] == 0
    assert obj["counts"] == {"direct_count": 10, "indirect_count": 17, "unique_to_direct_count": 0,
                             "unique_to_indirect_count": 7, "shared_count": 10}
    assert obj["direct_only"] == [] and len(obj["indirect_only"]) == 3 and len(obj["shared"]) == 3
    assert [r["score"] for r in obj["indirect_only"]] == [0.95, 0.94, 0.93]
    assert res.header["served_by"] == "derived"


async def test_target_profile_derived_through_the_gateway(ot_root: Path, tmp_path) -> None:
    import dl_fixtures as F

    root = F.copy_fixture(ot_root, tmp_path / "ot" / "25.09")
    F.delete_table(root, "known_drug")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OPEN_TARGETS_DATA_PATH", str(root))
        ctx = _ctx(tmp_path, REPO / "configs" / "data" / "sources", overlays=REPO / "configs" / "data" / "overlays")
        gw = _gateway(ctx, tmp_path)
        res = await _derived(gw, "target", "get_comprehensive_target_profile", {"target_id": "PCSK9"})
        obj = res.obj
        assert obj["id"] == F.PCSK9
        assert "_vbt_unavailable" in obj["known_drugs"] and "num_drugs" not in obj
        assert isinstance(obj["pathways"], list) and obj["pathways"]
        assert res.header["status"] == "partial"


async def test_data_verbs_through_the_gateway_carry_the_calling_agent(ot, tmp_path) -> None:
    """The gateway passes a public verb's arguments through as the payload and sets ``agent`` from the
    calling agent's context (never the agent's own value), so the child applies ``withhold_from``."""
    from types import SimpleNamespace

    from vbt.datalayer.api import RawResult
    from vbt.datalayer.service.verbs import load_verbs

    gw = _gateway(ot, tmp_path)
    args = {"table": "open_targets.known_drug", "where": {"targetId": "PCSK9"}, "limit": 3, "agent": "spoofed"}
    plan = await gw.prepare("data", "find", args, SimpleNamespace(agent="target-biologist"))
    assert plan.route == "upstream" and plan.args_sent == {**args, "agent": "target-biologist"}
    out = load_verbs()["find"](ot, plan.args_sent)
    res = await gw.finish(plan, RawResult(json.dumps(out), out, None, "ok"))
    assert res.obj["_vbt"]["total"] > len(res.obj["rows"]) == 3 and res.status == "partial", res.text
    # the child's own header survives the gateway: its source and served_by, never "upstream"
    assert res.header["served_by"] == out["_vbt"]["served_by"] != "upstream"
    assert res.header["source"] == out["_vbt"]["source"] and res.header["total"] == out["_vbt"]["total"]
    # a typed error of the child is that error, not an empty_unverified success around it
    from vbt.datalayer.errors import GatewayError

    unknown = await gw.prepare("data", "lookup", {"table": "open_targets.target", "key": {"id": "ENSG00000999999"}},
                               SimpleNamespace(agent="a"))
    env = load_verbs()["lookup"](ot, unknown.args_sent)
    if env.get("status") == "tool_error":
        with pytest.raises(GatewayError) as err:
            await gw.finish(unknown, RawResult(json.dumps(env), env, None, "ok"))
        assert err.value.kind.value == env["kind"]
    bad = await gw.prepare("data", "find", {"table": "open_targets.known_drug", "where": {"nope": 1}}, None)
    env = load_verbs()["find"](ot, bad.args_sent)
    assert env["status"] == "tool_error"
    with pytest.raises(GatewayError) as err:
        await gw.finish(bad, RawResult(json.dumps(env), env, None, "ok"))
    assert err.value.kind.value == env["kind"] == "invalid_argument"
    wrapped = await gw.prepare("data", "describe", {"request": {"source": "open_targets"}}, SimpleNamespace(agent="a"))
    assert wrapped.args_sent == {"request": {"source": "open_targets", "agent": "a"}}
    bare = await gw.prepare("data", "describe", {"source": "open_targets"}, None)
    assert bare.args_sent == {"source": "open_targets"}


async def test_compute_entity_similarity_repairs_from_both_anchors(ot_root: Path, tmp_path) -> None:
    """Both anchors reach ``_serve``: a contradicted (or wrong) pair similarity is repaired with the
    cosine of the two stored vectors, in upstream's record shape without the vetoed interpretation."""
    import dl_fixtures as F
    from test_dl_gateway_flow import raw_of

    def unit(sim: float) -> list[float]:
        return [sim, math.sqrt(1 - sim * sim)] + [0.0] * 98

    root = F.copy_fixture(ot_root, tmp_path / "ot" / "25.09")
    F.delete_table(root, "literature_vector")
    F.write_table(root, "literature_vector", F.table("literature_vector", [
        {"category": "target", "word": F.PCSK9, "norm": 1.0, "vector": unit(1.0)},
        {"category": "disease", "word": "EFO_0020000", "norm": 1.0, "vector": unit(0.6)}]))
    F.write_manifest(root)
    a, b = F.PCSK9, "EFO_0020000"
    wrong = {"success": True, "entity_a": a, "entity_b": b, "entity_a_category": "target",
             "entity_b_category": "disease", "similarity": 0.123, "interpretation": "Very low/no similarity"}
    missing = {"success": False, "error": f"Entity '{a}' not found in literature vector dataset"}
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OPEN_TARGETS_DATA_PATH", str(root))
        ctx = _ctx(tmp_path, REPO / "configs" / "data" / "sources", overlays=REPO / "configs" / "data" / "overlays")
        gw = _gateway(ctx, tmp_path)
        for raw in (wrong, missing):
            plan = await gw.prepare("association", "compute_entity_similarity", {"entity_a": a, "entity_b": b}, None)
            res = await gw.finish(plan, raw_of(raw))
            assert res.header["served_by"] == "repaired", res.header
            assert {k: v for k, v in res.obj.items() if k != "_vbt"} == {
                "entity_a": a, "entity_a_category": "target", "entity_b": b, "entity_b_category": "disease",
                "similarity": pytest.approx(0.6)}


def test_listed_schemas_offer_the_phase3_verbs(ot) -> None:
    """F17/F18: the listings advertise what the runtime verbs accept: multi-hop neighbors with nodes and
    max_nodes, propagated members, and the public expand and enrich verbs."""
    from vbt.datalayer.derive.tools import MAX_HOPS, native_tools
    from vbt.datalayer.service.verbs.network import MAX_HOPS as CHILD_MAX_HOPS

    assert MAX_HOPS == CHILD_MAX_HOPS
    tools = {t.verb: t for t in native_tools(ot.catalog)}
    nb = tools["neighbors"].input_schema
    assert nb["properties"]["hops"]["maximum"] == MAX_HOPS and "enum" not in nb["properties"]["hops"]
    assert {"nodes", "max_nodes", "score_order"} <= set(nb["properties"])
    assert "phase 3" not in json.dumps(tools["members"].input_schema)
    assert "one hop" not in tools["neighbors"].description
    ex = tools["expand"].input_schema
    assert "open_targets:ot_disease" in ex["properties"]["id_type"]["enum"]
    assert ex["required"] == ["id_type", "values"]
    assert "genes" in tools["enrich"].input_schema["properties"] and tools["enrich"].tables


def test_expand_is_a_public_verb(ot) -> None:
    import dl_fixtures as F

    out = call(ot, "expand", id_type="open_targets:ot_disease", values=[F.T2D], direction="ancestors")
    assert "rows" in out and out["_vbt"]["status"] in ("ok", "empty"), out
