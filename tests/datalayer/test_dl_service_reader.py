"""The data child's bounded reader (service/reader.py, service/sidecar.py; §11.6, §14.5).

Fixtures are written here with pyarrow (known_drug rows from ``dl_fixtures``, small hive and
single-file tables); every expectation is computed from the rows directly (an oracle), never from
the reader under test.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

import dl_fixtures as F  # noqa: E402
import yaml  # noqa: E402

from vbt.datalayer.plugins.base import FormatError  # noqa: E402
from vbt.datalayer.predicate import And, Any as AnyP, Cmp, Eq, In, IsNull  # noqa: E402
from vbt.datalayer.rowkey import canonical  # noqa: E402
from vbt.datalayer.service import ServiceContext  # noqa: E402
from vbt.datalayer.service.reader import BudgetExceeded, ScanStats, logical_leaves  # noqa: E402
from vbt.datalayer.service.verbs import load_verbs  # noqa: E402
from vbt.datalayer.settings import DataSettings  # noqa: E402


def make_ctx(tmp: Path, *descriptors: dict[str, Any], **data: Any) -> ServiceContext:
    (tmp / "sources").mkdir(exist_ok=True)
    (tmp / "overlays").mkdir(exist_ok=True)
    for d in descriptors:
        (tmp / "sources" / f"{d['source']}.yaml").write_text(yaml.safe_dump(d, sort_keys=False))
    settings = DataSettings.from_dict({"descriptors_dir": str(tmp / "sources"), "overlays_dir": str(tmp / "overlays"),
                                       "cache_dir": str(tmp / "cache"), **data}, project_root=tmp)
    return ServiceContext(settings)


def source(name: str, root: Path, tables: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"schema": "vbt.datasource/1", "source": name, "title": name, "root": str(root),
            "release": {"from": "literal"}, "defaults": {"format": "parquet", "layout": "sharded_dir"},
            "tables": tables, **extra}


KNOWN_DRUG = {
    "kind": "fact", "path": "known_drug", "grain": "one clinical precedence record",
    "key": {"columns": list(F.KNOWN_DRUG_KEY), "nullable": ["phase", "status"], "check": "full"},
    "grains": {"drug": ["drugId"], "indication": ["drugId", "diseaseId"]},
    "rank": [{"column": "phase", "direction": "desc"}],
    "columns": {"drugId": {"role": "identifier"}, "targetId": {"role": "identifier"},
                "diseaseId": {"role": "identifier"},
                "phase": {"role": "measure", "statistic": "clinical_phase", "scale": [0, 4], "missing": "unknown"},
                "status": {"role": "category"}, "drugType": {"role": "category", "placeholders": ["Unknown"]}},
}


@pytest.fixture
def kd(tmp_path: Path) -> tuple[ServiceContext, list[dict[str, Any]]]:
    rows = F.ot_rows()["known_drug"]
    tbl = F.table("known_drug", rows)
    d = tmp_path / "ot" / "known_drug"
    d.mkdir(parents=True)
    pq.write_table(tbl.slice(0, 20), d / "part-00000.parquet", row_group_size=7)
    pq.write_table(tbl.slice(20), d / "part-00001.parquet", row_group_size=7)
    (d / "part-00002.parquet.part").write_bytes(b"PAR1 partial download")
    ctx = make_ctx(tmp_path, source("ot", tmp_path / "ot", {"known_drug": KNOWN_DRUG}))
    return ctx, rows


def _known(v: Any) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def oracle_topk(rows: list[dict[str, Any]], k: int) -> list[list[Any]]:
    """Phase descending, nulls last, ties by the canonical key (storage types as stored)."""
    types = ["string", "string", "string", "double", "string"]
    ranked = sorted(rows, key=lambda r: ((0, -r["phase"]) if _known(r["phase"]) else (1, 0.0),
                                         canonical([r[c] for c in F.KNOWN_DRUG_KEY], types)))
    return [json.loads(canonical([r[c] for c in F.KNOWN_DRUG_KEY], types)) for r in ranked[:k]]


def test_topk_is_global_with_ties_nulls_and_nullable_key_parts(kd):
    ctx, rows = kd
    reader = ctx.reader("ot.known_drug")
    mine = [r for r in rows if r["targetId"] == F.T]
    for k in (1, 5, 12, 40):
        got = reader.top_k(Eq("targetId", F.T), [{"column": "phase", "direction": "desc"}], k)
        assert got == oracle_topk(mine, k)
    got = reader.top_k(Eq("targetId", F.T), [{"column": "phase", "direction": "desc"}], 40)
    assert got[-2:] == [k for k in oracle_topk(mine, 40)[-2:]] and all(k[3] is None for k in got[-2:])
    assert any(k[4] is None for k in got[:5]), "a null status is a key part, rendered null"
    first_in_file = [[r[c] for c in F.KNOWN_DRUG_KEY] for r in mine[:5]]
    assert got[:5] != first_in_file, "the fixture must discriminate file order from the global top-k"


def test_totals_and_unknown_attribution(kd):
    ctx, rows = kd
    reader = ctx.reader("ot.known_drug")
    mine = [r for r in rows if r["targetId"] == F.T]
    pred = And((Eq("targetId", F.T), Cmp("phase", ">=", 3)))
    total, unknown, not_applicable, unknown_total = reader.count(pred)
    assert total == sum(1 for r in mine if _known(r["phase"]) and r["phase"] >= 3)
    n_unknown = sum(1 for r in mine if not _known(r["phase"]))
    assert unknown == {"phase": n_unknown, "_rows": n_unknown} and unknown_total == n_unknown == 2
    assert not_applicable == {}
    # rows of another target are false, not unknown: they never count
    assert reader.count(Cmp("phase", ">=", 3))[3] == n_unknown
    # grains count distinct values over all matching rows
    total_drugs, *_ = reader.count(Eq("targetId", F.T), "drug")
    assert total_drugs == len({r["drugId"] for r in mine})


def test_key_set_rows_and_placeholders(kd):
    ctx, rows = kd
    reader = ctx.reader("ot.known_drug")
    keys = reader.key_set(Eq("drugId", "CHEMBL6000"))
    assert keys == [["CHEMBL6000", F.T, "EFO_0030000", None, None]]
    assert reader.key_set(None, max_n=3) is None, "over the cap: no key set"
    out, row_keys, stats = reader.rows(Eq("targetId", F.T), ["drugId", "phase"], limit=3)
    assert len(out) == 3 and stats.total == sum(1 for r in rows if r["targetId"] == F.T)
    assert [r["phase"] for r in out] == [4.0, 4.0, 4.0] and len(row_keys) == 3
    # limit_grain keeps each drug's best row
    best, _, _ = reader.rows(Eq("targetId", F.T), ["drugId", "phase"], limit=4, limit_grain="drug")
    assert len({r["drugId"] for r in best}) == 4


def test_partial_files_are_never_read(kd):
    ctx, rows = kd
    reader = ctx.reader("ot.known_drug")
    assert [Path(f.uri).name for f in reader.fragments()] == ["part-00000.parquet", "part-00001.parquet"]
    assert reader.count()[0] == len(rows)


def test_float32_equality_and_storage_typed_distinct(tmp_path):
    root = F.build_tahoe_fixture(tmp_path / "tahoe")
    de = {"kind": "fact", "path": F.TAHOE_DE, "layout": "single_file", "grain": "one contrast",
          "key": {"columns": list(F.TAHOE_KEY)},
          "columns": {"drug": {"role": "identifier"}, "concentration": {"role": "scope", "statistic": "numeric"},
                      "concentration_unit": {"role": "scope"}, "Cell_ID_DepMap": {"role": "identifier"},
                      "plate": {"role": "scope", "scope": {"kind": "replicate"}},
                      "gene_name": {"role": "identifier"}, "padj": {"role": "measure"}}}
    ctx = make_ctx(tmp_path, source("tahoe", root, {"de": de}))
    reader = ctx.reader("tahoe.de")
    total, *_ = reader.count(And((Eq("drug", F.BORTEZOMIB), Eq("concentration", 0.05))))
    assert total == len(F.oracle_tahoe(root, F.BORTEZOMIB, concentration=0.05)) > 0
    assert reader.distinct(Eq("drug", F.BORTEZOMIB), ["concentration"])["concentration"] == [0.05, 0.5, 5.0]
    keys = reader.key_set(And((Eq("drug", F.BORTEZOMIB), Eq("concentration", 0.05))))
    assert keys and all(k[1] == 0.05 for k in keys), "float32 key parts render in their storage type"


def _target_ctx(tmp_path: Path) -> ServiceContext:
    d = tmp_path / "ot" / "target"
    d.mkdir(parents=True)
    pq.write_table(F.table("target", F.ot_rows()["target"]), d / "part-00000.parquet")
    target = {"kind": "entity", "path": "target", "grain": "gene", "key": {"columns": ["id"]},
              "columns": {"id": {"role": "identifier", "self": True}, "approvedSymbol": {"role": "label", "of": "id"},
                          "go": {"role": "nested", "item_key": ["id", "aspect", "evidence", "source", "geneProduct"],
                                 "fields": {"id": {"role": "identifier"}, "aspect": {"role": "category"},
                                            "evidence": {"role": "category"}, "source": {"role": "category"},
                                            "geneProduct": {"role": "identifier"}}}}}
    return make_ctx(tmp_path, source("ot", tmp_path / "ot", {"target": target}))


def test_leaf_projection_inside_lists_reads_only_key_and_filter_leaves(tmp_path):
    ctx = _target_ctx(tmp_path)
    reader = ctx.reader("ot.target")
    calls: list[list[str]] = []
    fmt = reader.fmt

    class Recorder:
        def __getattr__(self, name: str) -> Any:
            return getattr(fmt, name)

        def read_leaves(self, frag, leaves, row_groups):
            calls.append(list(leaves))
            return fmt.read_leaves(frag, leaves, row_groups)

    reader.fmt = Recorder()
    total, *_ = reader.count(AnyP("go[]", Eq("aspect", "F")))
    assert total == 1
    assert calls and all(set(c) <= {"id", "go.list.element.aspect"} for c in calls), calls
    calls.clear()
    out, _, _ = reader.rows(AnyP("go[]", Eq("aspect", "F")), ["approvedSymbol"])
    assert out == [{"approvedSymbol": "PCSK9"}]
    assert calls[0] == ["id", "go.list.element.aspect"] and "approvedSymbol" in calls[1], \
        "pass 2 reads the output leaves only for the matching row group"


def _hive(tmp_path: Path, *, corrupt: bool = False) -> ServiceContext:
    base = tmp_path / "ot" / "evidence"
    schema = pa.schema([("id", pa.string()), ("datasourceId", pa.string()), ("targetId", pa.string()),
                        ("score", pa.float64())])
    for src, n in (("chembl", 3), ("europepmc", 4)):
        p = base / f"sourceId={src}"
        p.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([{"id": f"{src}-{i}", "datasourceId": src, "targetId": F.T,
                                               "score": i / 10} for i in range(n)], schema=schema),
                       p / "part-00000.parquet")
    if corrupt:
        (base / "sourceId=europepmc" / "part-00001.parquet").write_bytes(b"not a parquet file at all")
    ev = {"kind": "fact", "path": "evidence", "layout": "hive", "grain": "one evidence string",
          "key": {"columns": ["id"]},
          "partitions": {"sourceId": {"column": {"role": "category", "vocab": ["chembl", "europepmc"]},
                                      "mirrored_by": ["datasourceId"]}},
          "columns": {"id": {"role": "identifier"}, "datasourceId": {"role": "category"},
                      "targetId": {"role": "identifier"}, "score": {"role": "measure", "statistic": "score_0_1"}}}
    return make_ctx(tmp_path, source("ot", tmp_path / "ot", {"evidence": ev}))


def test_hive_partitions_restored_and_pruned(tmp_path):
    ctx = _hive(tmp_path)
    reader = ctx.reader("ot.evidence")
    rows, _, stats = reader.rows(Eq("sourceId", "chembl"), [])
    assert {r["sourceId"] for r in rows} == {"chembl"} and len(rows) == 3
    assert stats.pruned_fragments == 1
    st = ScanStats()
    total = sum(1 for _ in reader.scan(Eq("datasourceId", "europepmc"), stats=st))
    assert total == 4 and st.pruned_fragments == 1, "a mirrored column prunes through its partition"
    assert reader.count(IsNull("sourceId"))[0] == 0


def test_corrupt_shard_is_an_error_naming_it_and_its_partition(tmp_path):
    ctx = _hive(tmp_path, corrupt=True)
    reader = ctx.reader("ot.evidence")
    with pytest.raises(FormatError) as err:
        reader.count()
    assert "part-00001.parquet" in str(err.value) and err.value.partition == {"sourceId": "europepmc"}
    assert reader.count(Eq("sourceId", "chembl"))[0] == 3, "a pruned partition is never opened"


def test_sidecar_index_prunes_row_groups_that_all_span_the_gene_range(tmp_path):
    root = tmp_path / "t"
    root.mkdir()
    genes = [f"G{i:03d}" for i in range(50)]
    rows = []
    for rg in range(8):
        for g in genes:
            if g == "G025" and rg != 5:
                continue
            rows.append({"drug": f"D{rg}", "gene_name": g, "padj": 0.01 * (rg + 1)})
    rows.append({"drug": "D9", "gene_name": "A000", "padj": 0.5})       # widen every range a little
    pq.write_table(pa.Table.from_pylist(rows), root / "de.parquet", row_group_size=49)
    de = {"kind": "fact", "path": "de.parquet", "layout": "single_file", "grain": "g",
          "key": {"columns": ["drug", "gene_name"]},
          "access_paths": [{"columns": ["drug"], "via": "row_group_stats"},
                           {"columns": ["gene_name"], "via": "sidecar_index"}],
          "columns": {"drug": {"role": "identifier"}, "gene_name": {"role": "identifier"},
                      "padj": {"role": "measure"}}}
    ctx = make_ctx(tmp_path, source("x", root, {"de": de}))
    reader = ctx.reader("x.de")
    info = reader.footer(reader.fragments()[0])
    spans = [g.chunks["gene_name"] for g in info.row_groups]
    assert sum(1 for c in spans if c.min <= "G025" <= c.max) >= len(spans) - 1, "statistics cannot prune genes"
    st = ScanStats()
    hits = list(reader.scan(Eq("gene_name", "G025"), stats=st))
    assert len(hits) == 1 and hits[0].key == ("D5", "G025")
    assert st.used_sidecar and st.pruned_row_groups == len(spans) - 1
    sidecars = list((tmp_path / "cache" / "x").rglob("access/de.gene_name.idx"))
    assert len(sidecars) == 1
    st2 = ScanStats()
    assert sum(1 for _ in reader.scan(Eq("drug", "D3"), stats=st2)) == 49
    assert st2.pruned_row_groups >= len(spans) - 2, "row_group_stats prune the drug column"


def test_budget_exceeded_before_reading(kd):
    ctx, _rows = kd
    reader = ctx.reader("ot.known_drug")
    assert reader.estimate_scan_bytes(reader.leaves(["drugId"]), None) > 100
    with pytest.raises(BudgetExceeded):
        reader.count(Eq("targetId", F.T), budget_bytes=100)
    out = load_verbs()["_witness"](ctx, {"table": "ot.known_drug", "predicate": {"eq": ["targetId", F.T]},
                                         "budget_bytes": 100})
    assert out["total_method"] == "unknown" and "budget" in out["reason"] and out["total"] is None


def test_logical_leaves_and_in_predicate(kd):
    ctx, rows = kd
    reader = ctx.reader("ot.known_drug")
    paths = dict(logical_leaves(reader.schema()))
    assert "urls[].url" in paths and "phase" in paths
    total, *_ = reader.count(In("drugId", ("CHEMBL5000", "CHEMBL5001")))
    assert total == sum(1 for r in rows if r["drugId"] in ("CHEMBL5000", "CHEMBL5001"))


def test_sidecar_build_over_budget_names_the_offline_command(tmp_path):
    root = tmp_path / "t"
    root.mkdir()
    pq.write_table(pa.Table.from_pylist([{"drug": f"D{i % 3}", "gene_name": f"G{i:04d}"} for i in range(3000)]),
                   root / "de.parquet", row_group_size=500)
    de = {"kind": "fact", "path": "de.parquet", "layout": "single_file", "grain": "g",
          "key": {"columns": ["drug", "gene_name"]},
          "access_paths": [{"columns": ["gene_name"], "via": "sidecar_index", "build": "readiness"}],
          "columns": {"drug": {"role": "identifier"}, "gene_name": {"role": "identifier"}}}
    ctx = make_ctx(tmp_path, source("x", root, {"de": de}), witness={"max_scan_bytes": 2000})
    with pytest.raises(BudgetExceeded, match="vbt ds index build --access-paths"):
        ctx.reader("x.de").count(Eq("gene_name", "G0007"))


def test_not_applicable_is_counted_apart_from_unknown(tmp_path):
    d = tmp_path / "ot" / "ev"
    d.mkdir(parents=True)
    rows = [{"id": "e1", "datatypeId": "literature", "year": 2001},
            {"id": "e2", "datatypeId": "literature", "year": None},
            {"id": "e3", "datatypeId": "genetic", "year": None}, {"id": "e4", "datatypeId": "genetic", "year": None}]
    pq.write_table(pa.Table.from_pylist(rows), d / "part-00000.parquet")
    ev = {"kind": "fact", "path": "ev", "grain": "evidence", "key": {"columns": ["id"]},
          "columns": {"id": {"role": "identifier"}, "datatypeId": {"role": "category"},
                      "year": {"role": "time", "applies_when": {"datatypeId": ["literature"]}}}}
    ctx = make_ctx(tmp_path, source("ot", tmp_path / "ot", {"ev": ev}))
    total, unknown, not_applicable, unknown_total = ctx.reader("ot.ev").count(Cmp("year", ">=", 2000))
    assert (total, unknown, not_applicable, unknown_total) == (1, {"year": 1, "_rows": 1}, {"year": 2}, 3)


def test_in_band_unknowns_read_as_null(tmp_path):
    d = tmp_path / "ot" / "expr"
    d.mkdir(parents=True)
    rows = [{"id": "a", "level": 3, "unit": "TPM", "kind": "x"}, {"id": "b", "level": -1, "unit": "TPM", "kind": "x"},
            {"id": "c", "level": 2, "unit": "", "kind": "Unknown"}, {"id": "d", "level": 0, "unit": "TPM", "kind": "y"}]
    pq.write_table(pa.Table.from_pylist(rows), d / "part-00000.parquet")
    t = {"kind": "fact", "path": "expr", "grain": "row", "key": {"columns": ["id"]},
         "columns": {"id": {"role": "identifier"},
                     "level": {"role": "measure", "statistic": "ordinal", "missing_values": [-1],
                               "unknown_when": [{"column": "unit", "eq": ""}]},
                     "unit": {"role": "category"}, "kind": {"role": "category", "placeholders": ["Unknown"]}}}
    reader = make_ctx(tmp_path, source("ot", tmp_path / "ot", {"expr": t})).reader("ot.expr")
    total, unknown, _na, unknown_total = reader.count(Cmp("level", ">=", 0))
    assert (total, unknown, unknown_total) == (2, {"level": 2, "_rows": 2}, 2)
    assert reader.count(Cmp("level", "!=", 3))[0] == 1, "an unknown level never passes, not even !="
    rows_out, _, _ = reader.rows(None, ["id", "level", "kind"], order=[{"column": "id", "direction": "asc"}])
    assert [(r["level"], r["kind"]) for r in rows_out] == [(3, "x"), (None, "x"), (None, None), (0, "y")]
    assert reader.distinct(None, ["kind"]) == {"kind": ["x", "y"]}


def test_arrow_pushdown_keeps_unknown_rows_for_attribution(tmp_path):
    d = tmp_path / "ot" / "s"
    d.mkdir(parents=True)
    scores = [0.9, 0.1, None, float("nan"), 0.7] * 60            # odd period: group "a" sees null and NaN
    pq.write_table(pa.table({"id": [f"r{i}" for i in range(len(scores))], "g": ["a", "b"] * (len(scores) // 2),
                             "score": scores}), d / "part-00000.parquet", row_group_size=64)
    t = {"kind": "fact", "path": "s", "grain": "row", "key": {"columns": ["id"]},
         "columns": {"id": {"role": "identifier"}, "g": {"role": "category"},
                     "score": {"role": "measure", "statistic": "score_0_1"}}}
    reader = make_ctx(tmp_path, source("ot", tmp_path / "ot", {"s": t})).reader("ot.s")
    pred = And((Eq("g", "a"), Cmp("score", ">", 0.5)))
    assert reader._pushable(list(pred.preds), [["g"], ["score"]]), "both conjuncts go to Arrow"
    rows = [{"id": f"r{i}", "g": ["a", "b"][i % 2], "score": s} for i, s in enumerate(scores)]
    mine = [r for r in rows if r["g"] == "a"]
    known = [r for r in mine if _known(r["score"])]
    total, unknown, _na, unknown_total = reader.count(pred)
    assert total == sum(1 for r in known if r["score"] > 0.5)
    assert unknown == {"score": len(mine) - len(known), "_rows": len(mine) - len(known)}
    assert unknown_total == len(mine) - len(known) == 60
    st = ScanStats()
    assert len(list(reader.scan(pred, stats=st))) == total and st.total == total
