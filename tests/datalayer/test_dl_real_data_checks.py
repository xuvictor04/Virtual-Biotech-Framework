"""Readiness and data-child behaviour the real releases called for (Wave A contract requests).

Each test builds the smallest table showing one fact the real data had: content-identified rows whose
grouping key repeats (OT 25.09 pharmacogenomics), a universe whose column also holds other ids
(disease_hpo, target.proteinIds), declared missing values in a universe column (Tahoe ``NA``), a rare
prefix at the end of a file (cl-basic ``CP:``), lists nested in lists (drug_indication references),
nullable item-key parts (target chemicalProbes ``drugId``), an optional nested field the data lacks
(expression ``protein.cell_type``), a placeholder-like word with a meaning (DepMap ``PlateCoating``),
SIGNOR rows stored in both orientations with their roles swapped, and the composite reference check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

import yaml  # noqa: E402

from vbt.datalayer.descriptor.columns import ItemKey  # noqa: E402
from vbt.datalayer.descriptor.models import EdgeSpec  # noqa: E402
from vbt.datalayer.service import ServiceContext  # noqa: E402
from vbt.datalayer.service import items as _items  # noqa: E402
from vbt.datalayer.service.checks import _spread, check_table  # noqa: E402
from vbt.datalayer.service.verbs.network import _direction_fields, _directed, _identity  # noqa: E402
from vbt.datalayer.settings import DataSettings  # noqa: E402


def make_ctx(tmp: Path, tables: dict[str, Any], *, id_types: dict[str, Any] | None = None) -> ServiceContext:
    desc = {"schema": "vbt.datasource/1", "source": "s", "title": "s", "root": str(tmp / "data"),
            "release": {"from": "literal"}, "defaults": {"format": "parquet", "layout": "sharded_dir"},
            "id_types": id_types or {}, "tables": tables}
    (tmp / "sources").mkdir(parents=True, exist_ok=True)
    (tmp / "overlays").mkdir(parents=True, exist_ok=True)
    (tmp / "sources" / "s.yaml").write_text(yaml.safe_dump(desc, sort_keys=False))
    settings = DataSettings.from_dict({"descriptors_dir": str(tmp / "sources"), "overlays_dir": str(tmp / "overlays"),
                                       "cache_dir": str(tmp / "cache")}, project_root=tmp)
    return ServiceContext(settings)


def write(tmp: Path, name: str, rows: list[dict[str, Any]], schema: Any = None, part: str = "part-00000") -> None:
    d = tmp / "data" / name
    d.mkdir(parents=True, exist_ok=True)
    tbl = pa.Table.from_pylist(rows, schema=schema) if schema is not None else pa.Table.from_pylist(rows)
    pq.write_table(tbl, d / f"{part}.parquet")


def checks(model: Any, name: str) -> list[Any]:
    return [c for c in model.checks if c.name == name]


def failing(model: Any, name: str) -> list[Any]:
    return [c for c in checks(model, name) if not c.ok and c.level == "error"]


# ---------------------------------------------------------------------------- R5b: content identity


def _pgx(tmp: Path, rows: list[dict[str, Any]]) -> Any:
    write(tmp, "pgx", rows)
    t = {"kind": "fact", "path": "pgx", "grain": "annotation",
         "key": {"columns": ["study", "variant"], "nullable": ["variant"], "row_identity": "content_hash",
                 "check": "sampled"},
         "columns": {"study": {"role": "identifier"}, "variant": {"role": "identifier"},
                     "drug": {"role": "identifier"}}}
    return check_table(make_ctx(tmp, {"pgx": t}), "s.pgx", "deep")


def test_content_identity_tests_equal_rows_not_the_grouping_key(tmp_path):
    """25.09 pharmacogenomics: the grouping key repeats 9,703 times while no two rows are equal."""
    rows = [{"study": "S1", "variant": None, "drug": "D1"}, {"study": "S1", "variant": None, "drug": "D2"},
            {"study": "S2", "variant": "rs1", "drug": "D1"}]
    model = _pgx(tmp_path / "ok", rows)
    assert model.status == "ready", [c.detail for c in model.checks if not c.ok]
    assert "no two rows equal" in checks(model, "R5b")[0].detail
    bad = _pgx(tmp_path / "dup", rows + [dict(rows[1])])
    assert bad.status == "key_violation" and "equal to another row" in failing(bad, "R5b")[0].detail


def test_rows_without_identity_keep_their_copies_and_still_check_key_nulls(tmp_path, monkeypatch):
    """25.09 interaction_evidence holds 24,280 exact copies of rows (interval 67 in two shards): content identity
    is refuted, so R5b failed as key_violation in every session. ``row_identity: none`` declares that copies occur:
    R5b has no uniqueness to test (and reads no row), and R5 still catches nulls in a non-nullable part."""
    from vbt.datalayer.service.reader import TableReader

    rows = [{"study": "S1", "variant": None, "drug": "D1"}, {"study": "S1", "variant": None, "drug": "D1"},
            {"study": "S2", "variant": "rs1", "drug": "D2"}]

    def table(tmp: Path, data: list[dict[str, Any]]) -> Any:
        write(tmp, "ev", data)
        t = {"kind": "fact", "path": "ev", "grain": "evidence record",
             "key": {"columns": ["study", "variant"], "nullable": ["variant"], "row_identity": "none",
                     "check": "sampled"},
             "columns": {"study": {"role": "identifier"}, "variant": {"role": "identifier"},
                         "drug": {"role": "identifier"}}}
        return make_ctx(tmp, {"ev": t})

    def no_scan(*a: Any, **k: Any) -> Any:
        raise AssertionError("a table without row identity has no uniqueness to scan for")

    monkeypatch.setattr(TableReader, "scan", no_scan)
    for depth in ("standard", "deep"):
        model = check_table(table(tmp_path / depth, rows), "s.ev", depth)
        assert model.status == "ready", [c.detail for c in model.checks if not c.ok]
        (r5b,) = checks(model, "R5b")
        assert r5b.ok and "row_identity: none" in r5b.detail and model.key_check.duplicates == 0
        assert "footer null counts" in checks(model, "R5")[0].detail
    bad = check_table(table(tmp_path / "nulls", rows + [{"study": None, "variant": "rs2", "drug": "D3"}]), "s.ev")
    assert bad.status == "key_violation" and "study=1" in failing(bad, "R5")[0].detail


def test_the_check_reports_the_storage_types_stats_would(tmp_path):
    """The gateway types witness keys from the session check (a first call on the 25.09 target tables otherwise
    waited 14.5 s more for ``_stats``): the check's types are the footers' and agree with ``_stats``, for a table
    and for an item table over it (its physical table's leaves)."""
    from vbt.datalayer.service.verbs.stats import table_stats

    item = pa.struct([("id", pa.string()), ("score", pa.float32())])
    schema = pa.schema([("gene", pa.large_string()), ("n", pa.int32()), ("probes", pa.list_(item))])
    write(tmp_path, "target", [{"gene": "g1", "n": 1, "probes": [{"id": "P1", "score": 0.5}]},
                               {"gene": "g2", "n": 2, "probes": []}], schema)
    tables = {"target": {"kind": "entity", "path": "target", "grain": "gene", "key": {"columns": ["gene"]},
                         "columns": {"gene": {"role": "identifier", "self": True}, "n": {"role": "count"},
                                     "probes": {"role": "nested", "item_key": ["id"],
                                                "fields": {"id": {"role": "identifier"},
                                                           "score": {"role": "measure"}}}}},
              "target_probes": {"kind": "fact", "items_of": {"table": "target", "path": "probes[]"},
                                "grain": "probe", "key": {"columns": [], "check": "sampled"}}}
    ctx = make_ctx(tmp_path, tables)
    for ref in ("s.target", "s.target_probes"):
        model = check_table(ctx, ref)
        stats = {c: cs.storage_type for c, cs in table_stats(ctx, ref).columns.items()}
        assert model.storage_types == stats and {"gene", "n"} <= set(stats), (ref, model.storage_types, stats)
    assert check_table(ctx, "s.target", "shallow").storage_types == {}       # no footers read


# ---------------------------------------------------------------------------- R4b: universe samples


def test_universe_where_filters_rows_and_list_items(tmp_path):
    """disease_hpo.id holds non-HP terms and target.proteinIds[] ENSP ids: R4b checks what ``where`` keeps."""
    write(tmp_path, "hpo", [{"id": i} for i in ("HP_0000001", "HP_0001250", "UBERON_0002048", "GO_0005737")])
    item = pa.struct([("id", pa.string()), ("source", pa.string())])
    write(tmp_path, "target", [{"gene": "g1", "ids": [{"id": "P04637", "source": "uniprot_swissprot"},
                                                      {"id": "ENSP00000269305", "source": "ensembl_PRO"}]},
                               {"gene": "g2", "ids": [{"id": "Q8NBP7", "source": "uniprot_trembl"}]}],
          pa.schema([("gene", pa.string()), ("ids", pa.list_(item))]))
    tables = {"hpo": {"kind": "ontology", "path": "hpo", "grain": "term", "key": {"columns": ["id"]},
                      "columns": {"id": {"role": "identifier", "id_type": "hpo", "self": True}}},
              "target": {"kind": "entity", "path": "target", "grain": "gene", "key": {"columns": ["gene"]},
                         "columns": {"gene": {"role": "identifier", "self": True},
                                     "ids": {"role": "nested", "item_key": ["id", "source"],
                                             "fields": {"id": {"role": "identifier", "id_type": "uniprot"},
                                                        "source": {"role": "category"}}}}}}
    where_hpo = {"text": ["id", "HP_", "substring"]}
    where_uni = {"in": ["ids[].source", ["uniprot_swissprot", "uniprot_trembl"]]}
    filtered = make_ctx(tmp_path, tables, id_types={
        "hpo": {"plugin": "hpo", "universe": {"table": "hpo", "keys": ["id"], "where": where_hpo}},
        "uniprot": {"plugin": "uniprot_accession", "universe": {"table": "target", "keys": ["ids[].id"],
                                                                "where": where_uni}}})
    for ref in ("s.hpo", "s.target"):
        model = check_table(filtered, ref)
        r4b = checks(model, "R4b")
        assert r4b and all(c.ok for c in r4b) and "where" in r4b[0].detail, [c.detail for c in r4b]
    whole = make_ctx(tmp_path / "whole", {k: {**v, "path": str(tmp_path / "data" / v["path"])} for k, v in tables.items()},
                     id_types={"hpo": {"plugin": "hpo", "universe": "hpo.id"},
                               "uniprot": {"plugin": "uniprot_accession",
                                           "universe": {"table": "target", "keys": ["ids[].id"]}}})
    assert check_table(whole, "s.hpo").status == "encoding_drift"
    assert "ENSP" in failing(check_table(whole, "s.target"), "R4b")[0].detail


def test_declared_missing_values_are_not_universe_keys(tmp_path):
    """Tahoe stores hTERT-HPNE's DepMap ID as the string 'NA', declared ``missing_values: [NA]``."""
    write(tmp_path, "de", [{"line": "ACH-000681", "v": 1.0}, {"line": "NA", "v": 2.0}])
    t = {"kind": "fact", "path": "de", "grain": "row", "key": {"columns": ["line", "v"]},
         "columns": {"line": {"role": "identifier", "id_type": "line", "missing_values": ["NA"]},
                     "v": {"role": "measure"}}}
    model = check_table(make_ctx(tmp_path, {"de": t},
                                 id_types={"line": {"plugin": "depmap_cell_line", "universe": "de.line"}}), "s.de")
    assert model.status == "ready" and "1 universe keys canonical" in checks(model, "R4b")[0].detail


def test_universe_sample_keeps_every_prefix():
    """cl-basic.obo ends with 9 obsolete CP: terms: a sample of the first values in file order missed them."""
    values = [f"CL:{i:07d}" for i in range(5000)] + [f"CP:{i:07d}" for i in range(9)]
    picked = _spread(values, 2000)
    assert len(picked) == 2000 and sum(v.startswith("CP:") for v in picked) == 9
    assert _spread(["b", "a", "a"], 10) == ["a", "b"]


# ---------------------------------------------------------------------------- item keys


def test_item_keys_of_a_list_in_a_list_are_unique_per_parent_item():
    """drug_indication: one source cited under two indications of a drug is not a repeat."""
    row = {"indications": [{"disease": "D1", "references": [{"source": "ClinicalTrials"}, {"source": "DailyMed"}]},
                           {"disease": "D2", "references": [{"source": "ClinicalTrials"}]}]}
    assert _items.item_key_duplicates(row, "indications[].references", ["source"]) == []
    row["indications"][1]["references"].append({"source": "ClinicalTrials"})
    assert _items.item_key_duplicates(row, "indications[].references", ["source"]) == ['["ClinicalTrials"]']
    assert _items.item_key_duplicates({"refs": [{"s": 1}, {"s": 1}]}, "refs", ["s"]) == ["[1]"]


def test_nullable_item_key_parts(tmp_path):
    """target chemicalProbes: (id, origin, drugId) is unique with NULLS NOT DISTINCT; drugId is null in 141 items."""
    with pytest.raises(ValueError, match="not item-key columns"):
        ItemKey(columns=["id"], nullable=["drugId"])
    item = pa.struct([("id", pa.string()), ("drugId", pa.string())])
    schema = pa.schema([("gene", pa.string()), ("probes", pa.list_(item))])
    write(tmp_path, "target", [{"gene": "g1", "probes": [{"id": "P1", "drugId": None}, {"id": "P1", "drugId": "C1"}]},
                               {"gene": "g2", "probes": [{"id": "P1", "drugId": None}]}], schema)

    def tables(nullable: list[str], path: str = "target") -> dict[str, Any]:
        return {"target": {"kind": "entity", "path": path, "grain": "gene", "key": {"columns": ["gene"]},
                           "columns": {"gene": {"role": "identifier", "self": True},
                                       "probes": {"role": "nested",
                                                  "item_key": {"columns": ["id", "drugId"], "nullable": nullable},
                                                  "fields": {"id": {"role": "identifier"},
                                                             "drugId": {"role": "identifier"}}}}},
                "target_probes": {"kind": "fact", "items_of": {"table": "target", "path": "probes[]"},
                                  "grain": "probe", "key": {"columns": [], "check": "sampled"}}}

    ctx = make_ctx(tmp_path, tables(["drugId"]))
    assert ctx.table("s.target_probes").nullable_key == ("probes[].drugId",)
    model = check_table(ctx, "s.target_probes", "deep")
    assert model.status == "ready", [c.detail for c in model.checks if not c.ok]
    strict = check_table(make_ctx(tmp_path / "strict", tables([], str(tmp_path / "data" / "target"))),
                         "s.target_probes", "deep")
    assert strict.status == "key_violation" and "probes[].drugId=2" in failing(strict, "R5")[0].detail
    write(tmp_path, "target", [{"gene": "g3", "probes": [{"id": "P2", "drugId": None}, {"id": "P2", "drugId": None}]}],
          schema, part="part-00001")
    dup = check_table(make_ctx(tmp_path / "dup", tables(["drugId"], str(tmp_path / "data" / "target"))),
                      "s.target_probes", "deep")
    assert dup.status == "key_violation" and failing(dup, "R5b"), "nulls are not distinct"


# ---------------------------------------------------------------------------- R4: optional nested fields


def test_an_optional_container_the_data_lacks_is_a_warning_with_its_fields(tmp_path):
    """expression tissues[].protein.cell_type (25.09) is absent from older files: a warning, never schema drift."""
    prot = pa.struct([("level", pa.int32())])
    schema = pa.schema([("id", pa.string()), ("tissues", pa.list_(pa.struct([("code", pa.string()),
                                                                              ("protein", prot)])))])
    write(tmp_path, "expr", [{"id": "g1", "tissues": [{"code": "UBERON_1", "protein": {"level": 2}}]}], schema)
    cell_type = {"role": "nested", "optional": True, "item_key": ["name", "level"],
                 "fields": {"name": {"role": "category"}, "level": {"role": "measure", "statistic": "ordinal"}}}
    t = {"kind": "entity", "path": "expr", "grain": "gene", "key": {"columns": ["id"]},
         "columns": {"id": {"role": "identifier", "self": True},
                     "tissues": {"role": "nested", "item_key": ["code"],
                                 "fields": {"code": {"role": "category"},
                                            "protein": {"role": "nested",
                                                        "fields": {"level": {"role": "measure", "statistic": "ordinal"},
                                                                   "cell_type": cell_type}}}}}}
    model = check_table(make_ctx(tmp_path, {"expr": t}), "s.expr")
    assert model.status == "ready", [c.detail for c in model.checks if not c.ok and c.level == "error"]
    assert model.columns["tissues.protein.cell_type"] == "missing"
    assert model.columns["tissues.protein.cell_type.name"] == "missing"
    warned = [c for c in checks(model, "R4") if not c.ok]
    assert len(warned) == 1 and warned[0].level == "warning" and warned[0].column == "tissues.protein.cell_type"


def test_an_absent_optional_field_blocks_only_its_readers():
    from vbt.datalayer.gateway.readiness import _within

    assert _within("tissues.protein.cell_type.name", "tissues.protein.cell_type")
    assert _within("tissues.protein.cell_type", "tissues.protein.cell_type")
    assert not _within("tissues", "tissues.protein.cell_type")


# ---------------------------------------------------------------------------- R7, R9


def test_a_placeholder_like_word_with_a_meaning_is_a_value(tmp_path):
    """DepMap PlateCoating 'None' means no coating (2,104 of 2,105 models)."""
    write(tmp_path, "m", [{"id": "a", "coat": "None"}, {"id": "b", "coat": "Laminin"}])

    def model(values: dict[str, str]) -> Any:
        coat = {"role": "category", "vocab": "data", **({"values": values} if values else {})}
        t = {"kind": "entity", "path": str(tmp_path / "data" / "m"), "grain": "model", "key": {"columns": ["id"]},
             "columns": {"id": {"role": "identifier", "self": True}, "coat": coat}}
        return check_table(make_ctx(tmp_path / str(bool(values)), {"m": t}), "s.m")

    assert any(not c.ok and "placeholder-like" in c.detail for c in checks(model({}), "R7"))
    assert all(c.ok for c in checks(model({"None": "no coating"}), "R7"))


def test_composite_references_are_checked_on_the_sampled_tuples(tmp_path):
    write(tmp_path, "edges", [{"a": f"g{i}", "b": f"g{i + 1}"} for i in range(50)])
    write(tmp_path, "ev", [{"id": "e1", "a": "g1", "b": "g2"}, {"id": "e2", "a": "g7", "b": "g8"}])
    tables = {"edges": {"kind": "fact", "path": "edges", "grain": "edge", "key": {"columns": ["a", "b"]},
                        "columns": {"a": {"role": "identifier"}, "b": {"role": "identifier"}}},
              "ev": {"kind": "fact", "path": "ev", "grain": "evidence", "key": {"columns": ["id"]},
                     "columns": {"id": {"role": "identifier"}, "b": {"role": "identifier"},
                                 "a": {"role": "identifier", "ref": {"table": "edges", "on": {"a": "a", "b": "b"}}}}}}
    model = check_table(make_ctx(tmp_path, tables), "s.ev")
    assert any(c.ok and "0 of 2 sampled" in c.detail for c in checks(model, "R9")), [c.detail for c in model.checks]
    write(tmp_path, "ev", [{"id": "e3", "a": "g9", "b": "g1"}], part="part-00001")
    again = check_table(make_ctx(tmp_path / "again", {**tables, "ev": {**tables["ev"],
                                                                         "path": str(tmp_path / "data" / "ev")},
                                                      "edges": {**tables["edges"],
                                                                "path": str(tmp_path / "data" / "edges")}}), "s.ev")
    assert again.status == "key_violation" and "1 of 3 sampled" in failing(again, "R9")[0].detail


# ---------------------------------------------------------------------------- edges


def test_signor_direction_comes_from_the_roles():
    """25.09 stores every SIGNOR relation in both orientations with the roles swapped: one directed edge."""
    edge = EdgeSpec(a="targetA", b="targetB", orientation="both",
                    sides={"a": ["intA", "roleA"], "b": ["intB", "roleB"]},
                    direction_from_roles={"columns": {"a": "roleA", "b": "roleB"}, "source": "regulator",
                                          "target": "regulator target", "when": {"src": ["signor"]}})
    key = ["src", "intA", "intB", "targetA", "targetB", "roleA", "roleB"]
    fwd = {"src": "signor", "intA": "P1", "intB": "P2", "targetA": "G1", "targetB": "G2",
           "roleA": "regulator", "roleB": "regulator target"}
    rev = {"src": "signor", "intA": "P2", "intB": "P1", "targetA": "G2", "targetB": "G1",
           "roleA": "regulator target", "roleB": "regulator"}
    assert _directed(fwd, edge) and _directed(rev, edge)
    assert _identity(fwd, key, edge) == _identity(rev, key, edge)
    assert _direction_fields(fwd, edge) == _direction_fields(rev, edge) == {"source_node": "G1", "target_node": "G2"}
    opposite = {**fwd, "roleA": "regulator target", "roleB": "regulator"}     # G2 regulates G1: another edge
    assert _identity(opposite, key, edge) != _identity(fwd, key, edge)
    intact = {**fwd, "src": "intact", "roleA": "unspecified role", "roleB": "unspecified role"}
    assert not _directed(intact, edge) and _direction_fields(intact, edge) == {}
    swapped = {**intact, "intA": "P2", "intB": "P1", "targetA": "G2", "targetB": "G1"}
    assert _identity(intact, key, edge) == _identity(swapped, key, edge)


# ---------------------------------------------------------------------------- R8 through _check


def test_check_names_the_quarantined_files(tmp_path):
    """The data child loads the catalog tolerantly: ``_check`` reports the files it left out."""
    from vbt.datalayer.service.verbs.check import check

    write(tmp_path, "t", [{"id": "a"}])
    ctx = make_ctx(tmp_path, {"t": {"kind": "entity", "path": "t", "grain": "row", "key": {"columns": ["id"]},
                                    "columns": {"id": {"role": "identifier", "self": True}}}})
    (tmp_path / "sources" / "broken.yaml").write_text("schema: vbt.datasource/1\nsource: broken\ntables: [\n")
    ctx = ServiceContext(ctx.settings)
    out = check(ctx, {"tables": ["s.t"], "depth": "shallow"})
    assert out["tables"]["s.t"]["status"] == "ready"
    assert [(q["file"], q["kind"]) for q in out["quarantined"]] == [("broken.yaml", "descriptor")]


# ---------------------------------------------------------------------------- matrices


def test_a_strict_matrix_reports_undeclared_axis_columns(tmp_path):
    """The Zenodo ibd_cohorts obs held response_remission, mayo_score, cohort and platform undeclared, and
    readiness never said so: a strict matrix axis that declares columns must declare every one."""
    anndata = pytest.importorskip("anndata")
    import warnings

    import numpy as np
    import pandas as pd

    d = tmp_path / "data"
    d.mkdir(parents=True)
    obs = pd.DataFrame({"patient_id": ["p1", "p2"], "platform": ["GPL570", "GPL570"]},
                       index=pd.Index(["GSM1", "GSM2"], name="sample_id"))
    var = pd.DataFrame(index=pd.Index(["OSMR", "IL6"], name="gene_symbol"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        anndata.AnnData(X=np.array([[1.0, 2.0], [3.0, 4.0]]), obs=obs, var=var).write_h5ad(d / "GSE1.h5ad")

    def table(obs_columns: dict[str, Any]) -> dict[str, Any]:
        return {"kind": "matrix", "path": str(d / "GSE1.h5ad"), "layout": "single_file", "format": "h5ad",
                "grain": "cell", "key": {"columns": ["@row.sample_id", "@col.gene_symbol"]},
                "matrix": {"axes": {
                    "obs": {"name": "sample", "from": "index", "index_name": "sample_id",
                            "key": {"columns": ["sample_id"]},
                            "columns": {"sample_id": {"role": "identifier", "self": True}, **obs_columns}},
                    "var": {"name": "gene", "from": "index", "index_name": "gene_symbol",
                            "key": {"columns": ["gene_symbol"]},
                            "columns": {"gene_symbol": {"role": "identifier", "self": True}}}},
                    "values": {"X": {"role": "measure", "statistic": "numeric"}}}}

    def model(name: str, obs_columns: dict[str, Any]) -> Any:
        ctx = make_ctx(tmp_path / name, {"m": table(obs_columns)})
        desc = yaml.safe_load((tmp_path / name / "sources" / "s.yaml").read_text())
        desc["strict"] = True
        (tmp_path / name / "sources" / "s.yaml").write_text(yaml.safe_dump(desc, sort_keys=False))
        return check_table(ServiceContext(ctx.settings), "s.m")

    lax = model("lax", {"patient_id": {"role": "category"}})
    assert lax.status == "schema_drift" and "platform" in failing(lax, "R4:undeclared")[0].detail
    full = model("full", {"patient_id": {"role": "category"}, "platform": {"role": "category"}})
    assert full.status == "ready", [c.detail for c in full.checks if not c.ok]


# ---------------------------------------------------------------------------- provenance of live sources


def test_live_sources_name_the_release_the_call_observed():
    """CT.gov has no pinned release: the remote witness's as_of (the registry's dataTimestamp) names it; a
    Census pull names the dated release 'stable' resolved to, never the alias."""
    from types import SimpleNamespace

    from vbt.datalayer.gateway.gateway import DataGateway, _CallState
    from vbt.datalayer.ipc import WitnessResponse

    live = SimpleNamespace(release=SimpleNamespace(expect=None))
    st = _CallState(name="clinicaltrials.count_clinical_trials", t0=0.0)
    assert DataGateway._release_of(live, st) is None
    st.witness = WitnessResponse(total=705, total_method="scan", as_of="2026-10-07T09:00:06")
    assert DataGateway._release_of(live, st) == "2026-10-07T09:00:06"
    census = _CallState(name="single_cell.get_anndata", t0=0.0)
    census.count_first = {"n_cells": 4771, "release": {"requested": "stable", "resolved": "2025-11-08"}}
    assert DataGateway._release_of(live, census) == "2025-11-08"
    pinned = SimpleNamespace(release=SimpleNamespace(expect="25.09"))
    assert DataGateway._release_of(pinned, st) == "25.09"


# ---------------------------------------------------------------------------- the check's own memory


def test_the_data_check_runs_under_the_reaper(monkeypatch: pytest.MonkeyPatch) -> None:
    """``vbt ds check`` and preflight run the data child as the bridge launches it: under the reaper with the data
    child's limit (a deep check of the interaction_evidence table, 27.3 M rows, reached 10.3 GiB resident when it ran
    uncontained)."""
    import subprocess
    import sys

    from vbt import preflight
    from vbt.config import load_config
    from vbt.datalayer.settings import DataSettings

    seen: dict[str, Any] = {}

    def fake_run(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        seen["argv"] = list(argv)
        return subprocess.CompletedProcess(argv, 0, stdout='{"tables": {}}\n', stderr="")

    monkeypatch.setattr(preflight.subprocess, "run", fake_run)
    cfg = load_config()
    assert preflight.run_data_check(cfg, tables=["open_targets.so"], depth="shallow") == {"tables": {}}
    argv = seen["argv"]
    assert "--check" in argv and argv[argv.index("--table") + 1] == "open_targets.so"
    if sys.platform.startswith("linux"):
        assert argv[2].endswith("reaper.py") and argv.index("--limit-mb") < argv.index("--"), argv
        assert int(argv[argv.index("--limit-mb") + 1]) == int(DataSettings.from_config(cfg).service.mem_limit_mb)


# ---------------------------------------------------------------------------- R10 edge orientation


def _edges_table(tmp: Path, rows: list[dict[str, Any]], rows_per_group: int) -> Any:
    d = tmp / "data" / "edges"
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), d / "part-00000.parquet", row_group_size=rows_per_group)
    t = {"kind": "fact", "path": "edges", "grain": "edge", "key": {"columns": ["a", "b"], "check": "full"},
         "edge": {"a": "a", "b": "b", "orientation": "both"},
         "columns": {"a": {"role": "endpoint", "side": "a"}, "b": {"role": "endpoint", "side": "b"}}}
    return check_table(make_ctx(tmp, {"edges": t}), "s.edges", "standard")


def test_edge_reverses_are_looked_up_beyond_the_sample(tmp_path):
    """OT 25.09 interaction stores every pair in both orientations, yet R10 sampled whole row groups and
    reported 18,827 of the sampled edges 'without their reverse': the reverse sat in another row group. The
    reverses are now looked up in the table, so only a truly missing one is reported."""
    rows = [{"a": f"G{i:05d}", "b": f"G{i + 1:05d}"} for i in range(0, 6000, 2)]
    rows += [{"a": r["b"], "b": r["a"]} for r in rows]        # the reverses in the second half of the file
    model = _edges_table(tmp_path / "both", rows, 500)
    got = checks(model, "R10:edges")
    assert got and got[0].ok, [c.detail for c in got]
    one_way = rows[:3000] + rows[3001:]                    # G00000 -> G00001 has no reverse now
    model = _edges_table(tmp_path / "one_way", one_way, 100_000)
    got = checks(model, "R10:edges")
    assert got and not got[0].ok and got[0].detail.startswith("1 of ") and "G00001" in got[0].detail, \
        [c.detail for c in got]
