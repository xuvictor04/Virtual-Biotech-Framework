"""Real Open Targets 25.09 through the unmodified upstream servers, the gateway and the data child.

Offline (always): regression tests for what the real tables showed, each on the smallest fixture with the
real shape:

* R9 looked references up with a ``scan`` whose ``In`` predicate is tested row by row against every value of
  the chunk in Python: the deep check of target_prioritisation (78,726 ``targetId`` -> ``target.id``) took
  51 s for a 1 MB table. Flat string and integer target columns are now matched with Arrow ``is_in``.

``VBT_DL_REAL_DATA=<dir>`` (the ``open_targets/25.09`` output directory downloaded from
https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/25.09/output/, or a ``data/real`` root holding it):
the same checks on the real tables.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

import yaml  # noqa: E402

from vbt.datalayer.service import ServiceContext  # noqa: E402
from vbt.datalayer.service import checks as _checks  # noqa: E402
from vbt.datalayer.service.checks import check_table  # noqa: E402
from vbt.datalayer.service.reader import TableReader  # noqa: E402
from vbt.datalayer.settings import DataSettings  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def real_ot_dir() -> Path | None:
    """The OT 25.09 directory named by ``VBT_DL_REAL_DATA`` (itself, or ``<dir>/open_targets/25.09``)."""
    root = os.environ.get("VBT_DL_REAL_DATA", "").strip()
    if not root:
        return None
    for cand in (Path(root), Path(root) / "open_targets" / "25.09"):
        if (cand / "target").is_dir():
            return cand
    return None


REAL = real_ot_dir()
needs_real = pytest.mark.skipif(REAL is None, reason="VBT_DL_REAL_DATA=<dir> with the Open Targets 25.09 tables needed")


# ---------------------------------------------------------------------------- small fixtures


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


def write(tmp: Path, name: str, rows: list[dict[str, Any]], schema: Any = None, part: str = "part-00000",
          row_group_size: int | None = None) -> None:
    d = tmp / "data" / name
    d.mkdir(parents=True, exist_ok=True)
    tbl = pa.Table.from_pylist(rows, schema=schema) if schema is not None else pa.Table.from_pylist(rows)
    pq.write_table(tbl, d / f"{part}.parquet", row_group_size=row_group_size)


def checks(model: Any, name: str) -> list[Any]:
    return [c for c in model.checks if c.name == name]


# ---------------------------------------------------------------------------- R9: references by Arrow is_in


def _prioritisation(tmp: Path, n: int = 3000, dangling: int = 7) -> ServiceContext:
    """target (ids over two shards, several row groups each) and target_prioritisation whose targetId refers to
    it (one row per target, as in 25.09, plus ``dangling`` ids the target table does not hold)."""
    ids = [f"ENSG{i:011d}" for i in range(n)]
    write(tmp, "target", [{"id": i, "approvedSymbol": f"G{k}"} for k, i in enumerate(ids[: n // 2])],
          part="part-00000", row_group_size=400)
    write(tmp, "target", [{"id": i, "approvedSymbol": f"G{k}"} for k, i in enumerate(ids[n // 2:])],
          part="part-00001", row_group_size=400)
    extra = [f"ENSG9{i:010d}" for i in range(dangling)]
    write(tmp, "prio", [{"targetId": i, "isInMembrane": k % 2} for k, i in enumerate(ids + extra)],
          row_group_size=1000)
    tables = {"target": {"kind": "entity", "path": "target", "grain": "gene", "key": {"columns": ["id"]},
                         "columns": {"id": {"role": "identifier", "self": True},
                                     "approvedSymbol": {"role": "label"}}},
              "prio": {"kind": "fact", "path": "prio", "grain": "gene", "key": {"columns": ["targetId"]},
                       "columns": {"targetId": {"role": "identifier", "ref": "target.id"},
                                   "isInMembrane": {"role": "category"}}}}
    return make_ctx(tmp, tables)


def test_r9_matches_flat_references_with_arrow_not_row_by_row(tmp_path, monkeypatch):
    """The deep R9 of a reference to a flat string key never scans the target with an ``In`` predicate, and finds
    exactly the dangling values the row-by-row scan finds."""
    ctx = _prioritisation(tmp_path)
    scans: list[Any] = []
    real_scan = TableReader.scan

    def counting_scan(self, predicate=None, **kw):
        if self.ref == "s.target":
            scans.append(predicate)
        return real_scan(self, predicate, **kw)

    monkeypatch.setattr(TableReader, "scan", counting_scan)
    model = check_table(ctx, "s.prio", "deep")
    (r9,) = checks(model, "R9")
    assert not r9.ok and "7 of 3007 sampled value(s) dangle" in r9.detail, r9.detail
    assert scans == [], "R9 scanned the target table row by row"

    # the scan fallback (what R9 did before) reports the same values
    monkeypatch.setattr(_checks, "_found_in_column", lambda *a, **k: None)
    (slow,) = checks(check_table(_prioritisation(tmp_path / "slow"), "s.prio", "deep"), "R9")
    assert slow.detail == r9.detail
    assert scans, "the fallback scans the target"


def test_found_in_column_handles_types_and_falls_back(tmp_path):
    """String and integer keys are matched in Arrow (nulls never match); a nested column, an item table's column
    or mixed value types fall back to the scan (None)."""
    item = pa.struct([("id", pa.string())])
    write(tmp_path, "t", [{"k": "a", "n": 1, "items": [{"id": "x"}]}, {"k": None, "n": None, "items": []},
                          {"k": "b", "n": 2, "items": None}],
          pa.schema([("k", pa.string()), ("n", pa.int32()), ("items", pa.list_(item))]))
    ctx = make_ctx(tmp_path, {"t": {"kind": "entity", "path": "t", "grain": "row", "key": {"columns": ["k"]},
                                    "columns": {"k": {"role": "identifier", "self": True},
                                                "n": {"role": "identifier"},
                                                "items": {"role": "nested", "item_key": ["id"],
                                                          "fields": {"id": {"role": "identifier"}}}}}})
    reader = ctx.reader("s.t")
    assert _checks._found_in_column(reader, "k", ["a", "zz", "b"]) == {'"a"', '"b"'}
    assert _checks._found_in_column(reader, "n", [2, 5]) == {"2"}
    assert _checks._found_in_column(reader, "items[].id", ["x"]) is None
    assert _checks._found_in_column(reader, "k", ["a", 1]) is None


# ---------------------------------------------------------------------------- samples of one-row-group shards


def _count_native(monkeypatch: Any, ctx: ServiceContext, ref: str) -> tuple[Any, list[int]]:
    reader = ctx.reader(ref)
    converted: list[int] = []
    real = reader.fmt.to_native

    def counting(table: Any) -> list[dict[str, Any]]:
        converted.append(table.num_rows)
        return real(table)

    monkeypatch.setattr(reader.fmt, "to_native", counting)
    return reader, converted


def test_a_sample_converts_only_the_sampled_rows_of_a_large_row_group(tmp_path, monkeypatch):
    """Every 25.09 shard is one row group (study: 1,964,234 rows); a 500-row ``_stats`` sample converted the whole
    group to Python rows (4.6 GB). Only the sampled rows are converted now, and they are the rows drawn before."""
    n = 20000
    write(tmp_path, "study", [{"studyId": f"GCST{i:08d}", "nSamples": i} for i in range(n)])
    ctx = make_ctx(tmp_path, {"study": {"kind": "entity", "path": "study", "grain": "study",
                                        "key": {"columns": ["studyId"]},
                                        "columns": {"studyId": {"role": "identifier", "self": True},
                                                    "nSamples": {"role": "category"}}}})
    reader, converted = _count_native(monkeypatch, ctx, "s.study")
    rows = reader.sample_rows(50, seed=7)
    assert len(rows) == 50 and sum(converted) == 50, converted
    import random
    expected = random.Random(7).sample([f"GCST{i:08d}" for i in range(n)], 50)   # what the whole-group draw chose
    assert [r["studyId"] for r in rows] == expected


def test_an_item_table_sample_converts_rows_in_chunks(tmp_path, monkeypatch):
    """target_essentiality and expression groups hold thousands of genes with nested items (2.3 GB and 2.4 GB for
    one ``_stats`` sample). An item table's large group is converted in chunks until the sample has its items."""
    item = pa.struct([("tissueId", pa.string()), ("score", pa.float64())])
    schema = pa.schema([("gene", pa.string()), ("tissues", pa.list_(item))])
    write(tmp_path, "expr", [{"gene": f"g{i}", "tissues": [{"tissueId": "UBERON_1", "score": 1.0},
                                                           {"tissueId": "UBERON_2", "score": float(i)}]}
                             for i in range(5000)], schema)
    ctx = make_ctx(tmp_path, {"expr": {"kind": "entity", "path": "expr", "grain": "gene", "key": {"columns": ["gene"]},
                                       "columns": {"gene": {"role": "identifier", "self": True},
                                                   "tissues": {"role": "nested", "item_key": ["tissueId"],
                                                               "fields": {"tissueId": {"role": "identifier"},
                                                                          "score": {"role": "category"}}}}},
                              "expr_tissues": {"kind": "fact", "items_of": {"table": "expr", "path": "tissues[]"},
                                               "grain": "tissue", "key": {"columns": [], "check": "sampled"}}})
    reader, converted = _count_native(monkeypatch, ctx, "s.expr_tissues")
    rows = reader.sample_rows(100, seed=3)
    assert len(rows) == 100 and sum(converted) <= 256, converted
    assert all(r["gene"].startswith("g") and r["tissueId"] in ("UBERON_1", "UBERON_2") for r in rows), rows[:3]


# ---------------------------------------------------------------------------- R5 of item tables: sampled by parent row


def _go(tmp: Path, *, dup_every: int | None = None) -> ServiceContext:
    """target with a go[] list of 30 annotations per gene (25.09: 821,377 items over 78,726 genes); with
    ``dup_every`` every that many genes repeat one annotation."""
    item = pa.struct([("id", pa.string()), ("aspect", pa.string())])
    rows = []
    for g in range(400):
        go = [{"id": f"GO:{g * 100 + i:07d}", "aspect": "P"} for i in range(30)]
        if dup_every and g % dup_every == 0:
            go.append(dict(go[0]))
        rows.append({"id": f"ENSG{g:011d}", "go": go})
    write(tmp, "target", rows, pa.schema([("id", pa.string()), ("go", pa.list_(item))]))
    tables = {"target": {"kind": "entity", "path": "target", "grain": "gene", "key": {"columns": ["id"]},
                         "columns": {"id": {"role": "identifier", "self": True},
                                     "go": {"role": "nested", "item_key": ["id", "aspect"],
                                            "fields": {"id": {"role": "identifier"}, "aspect": {"role": "category"}}}}},
              "target_go": {"kind": "fact", "items_of": {"table": "target", "path": "go[]"}, "grain": "annotation",
                            "key": {"columns": [], "check": "sampled"}}}
    return make_ctx(tmp, tables)


def test_a_sampled_item_key_check_samples_parent_rows(tmp_path):
    """An item table has no row count, so its ``check: sampled`` key check expanded and rendered every item: the
    25.09 target item tables (5.7 M items) took 12 minutes of a standard check. Whole parent rows are sampled now,
    and a duplicate inside a sampled row is still found."""
    model = check_table(_go(tmp_path / "ok"), "s.target", "standard", sample_rows=3000)
    (r5b,) = [c for c in checks(model, "R5b") if "target_go" in c.detail]
    assert r5b.ok and "of parent rows" in r5b.detail, r5b.detail
    (r5,) = [c for c in checks(model, "R5") if "target_go" in c.detail]
    scanned = int(r5.detail.rsplit("(", 1)[1].split()[0])
    assert 0 < scanned < 12000 and scanned % 30 == 0, r5.detail      # whole rows of 30 items, not all 12,000
    bad = check_table(_go(tmp_path / "dup", dup_every=1), "s.target", "standard", sample_rows=3000)
    assert any(not c.ok and "duplicate key" in c.detail for c in checks(bad, "R5b")), \
        [c.detail for c in checks(bad, "R5b")]
    full = check_table(_go(tmp_path / "deep"), "s.target", "deep", sample_rows=3000)
    assert any("target_go" in c.detail and "every key" in c.detail for c in checks(full, "R5b"))


# ---------------------------------------------------------------------------- R5 of flat keys on Arrow arrays


def _assoc(tmp: Path, rows: list[dict[str, Any]]) -> ServiceContext:
    write(tmp, "assoc", rows[: len(rows) // 2], part="part-00000", row_group_size=7)
    write(tmp, "assoc", rows[len(rows) // 2:], part="part-00001", row_group_size=5)
    return make_ctx(tmp, {"assoc": {"kind": "fact", "path": "assoc", "grain": "pair",
                                    "key": {"columns": ["diseaseId", "targetId", "datasourceId"],
                                            "nullable": ["datasourceId"], "check": "full"},
                                    "columns": {"diseaseId": {"role": "identifier"}, "targetId": {"role": "identifier"},
                                                "datasourceId": {"role": "category"}, "score": {"role": "category"}}}})


def test_flat_key_uniqueness_is_counted_on_arrow_arrays(tmp_path, monkeypatch):
    """The full key check rendered every row's key in Python: 3.4 minutes for the 4.0 M rows of 25.09
    association_overall_direct, and the indirect tables hold 13 M. Flat keys are counted on dictionary codes now,
    with the same duplicates (nulls not distinct, across files and row groups) as the row scan."""
    rows = [{"diseaseId": f"EFO_{i % 9:07d}", "targetId": f"ENSG{i % 13:011d}", "datasourceId": None if i % 5 else "eva",
             "score": i} for i in range(60)]
    rows += [dict(rows[3]), dict(rows[40]), dict(rows[40])]        # three duplicates, one with a null part
    scans: list[Any] = []
    real_scan = TableReader.scan

    def counting_scan(self, predicate=None, **kw):
        if self.ref == "s.assoc" and kw.get("columns") == []:
            scans.append(1)
        return real_scan(self, predicate, **kw)

    monkeypatch.setattr(TableReader, "scan", counting_scan)
    fast = check_table(_assoc(tmp_path / "fast", rows), "s.assoc", "standard")
    assert not scans, "the key was rendered row by row"
    monkeypatch.setattr(_checks, "_arrow_key_duplicates", lambda *a, **k: None)
    slow = check_table(_assoc(tmp_path / "slow", rows), "s.assoc", "standard")
    assert scans
    for model in (fast, slow):
        assert model.key_check.duplicates == 3 and model.status == "key_violation", model.key_check
    assert sorted(checks(fast, "R5b")[0].detail.split(": ", 1)[1].split(", ")) == \
        sorted(checks(slow, "R5b")[0].detail.split(": ", 1)[1].split(", "))
    assert checks(fast, "R5")[0].detail == checks(slow, "R5")[0].detail


# ---------------------------------------------------------------------------- search: only the row's own names


def test_search_matches_a_rows_own_labels_not_those_of_the_entities_it_lists(tmp_path):
    """25.09 target lists each gene's paralogues: TP63 has ``homologues[].targetGeneSymbol == "TP53"``. The derived
    search_targets_by_name("TP53", limit=1) matched that label exactly and returned TP63 (it sorts before TP53 by
    id). A label counts only when its ``of`` is the row's key."""
    from vbt.datalayer.service.verbs.serve import search

    hom = pa.struct([("targetGeneId", pa.string()), ("targetGeneSymbol", pa.string())])
    syn = pa.struct([("label", pa.string())])
    schema = pa.schema([("id", pa.string()), ("approvedSymbol", pa.string()), ("symbolSynonyms", pa.list_(syn)),
                        ("homologues", pa.list_(hom))])
    write(tmp_path, "target", [
        {"id": "ENSG00000073282", "approvedSymbol": "TP63", "symbolSynonyms": [{"label": "TP53L"}],
         "homologues": [{"targetGeneId": "ENSG00000141510", "targetGeneSymbol": "TP53"}]},
        {"id": "ENSG00000141510", "approvedSymbol": "TP53", "symbolSynonyms": [{"label": "p53"}],
         "homologues": [{"targetGeneId": "ENSG00000073282", "targetGeneSymbol": "TP63"}]}], schema)
    ctx = make_ctx(tmp_path, {"target": {
        "kind": "entity", "path": "target", "grain": "gene", "key": {"columns": ["id"]},
        "columns": {"id": {"role": "identifier", "self": True},
                    "approvedSymbol": {"role": "label", "of": "id"},
                    "symbolSynonyms": {"role": "synonym", "of": "id", "synonym_kind": "alias", "path": "[].label"},
                    "homologues": {"role": "nested", "item_key": ["targetGeneId"], "fields": {
                        "targetGeneId": {"role": "identifier"},
                        "targetGeneSymbol": {"role": "label", "of": "targetGeneId"}}}}}})
    rows, _keys, total = search(ctx.reader("s.target"), "TP53", None, 1, {}, None)
    assert rows[0]["id"] == "ENSG00000141510" and rows[0]["match"] == "exact", rows
    assert total == 2                                   # TP63 still matches, by its own TP53L synonym (a prefix)
    rows, _keys, _total = search(ctx.reader("s.target"), "TP53", None, None, {}, None)
    assert [r["_match"]["column"] for r in rows] == ["approvedSymbol", "symbolSynonyms[].label"], rows


# ---------------------------------------------------------------------------- _stats of a positional item table


def test_stats_count_an_item_table_keyed_by_position(tmp_path):
    """25.09 target hallmarks.cancerHallmarks[] has ``item_key: {identity: position}``: counting its item table read
    the key part ``hallmarks.cancerHallmarks[]#`` as a column path and raised PathError, so the data child's
    ``_stats`` failed for it and the gateway had no storage types for get_target_hallmarks."""
    from vbt.datalayer.service.verbs.stats import table_stats

    item = pa.struct([("label", pa.string()), ("impact", pa.string())])
    write(tmp_path, "target", [{"id": "ENSG00000141510", "hallmarks": {"cancerHallmarks": [
        {"label": "genome instability", "impact": "promotes"}, {"label": "genome instability", "impact": "promotes"}]}},
        {"id": "ENSG00000169174", "hallmarks": None}],
        pa.schema([("id", pa.string()), ("hallmarks", pa.struct([("cancerHallmarks", pa.list_(item))]))]))
    ctx = make_ctx(tmp_path, {
        "target": {"kind": "entity", "path": "target", "grain": "gene", "key": {"columns": ["id"]},
                   "columns": {"id": {"role": "identifier", "self": True},
                               "hallmarks": {"role": "nested", "fields": {
                                   "cancerHallmarks": {"role": "nested", "item_key": {"identity": "position"},
                                                       "fields": {"label": {"role": "category"},
                                                                  "impact": {"role": "category"}}}}}}},
        "target_cancer_hallmarks": {"kind": "fact", "items_of": {"table": "target", "path": "hallmarks.cancerHallmarks[]"},
                                    "grain": "hallmark", "key": {"columns": [], "check": "sampled"}}})
    assert ctx.reader("s.target_cancer_hallmarks").key[-1].endswith("[]#")
    stats = table_stats(ctx, "s.target_cancer_hallmarks")
    assert stats.rows == 2 and stats.row_bytes_p99.get("row")


# ---------------------------------------------------------------------------- R6 container counts on Arrow arrays


def test_container_counts_are_read_from_arrow_arrays(tmp_path, monkeypatch):
    """R6:containers converted every row to count null, empty and non-empty lists: 150 s of the standard check of the
    25.09 expression table (43,804 genes, 4,940,421 tissues). The counts now come from the list arrays and equal the
    row-based ones, for a list of structs under a struct, with null and empty lists and null items, two levels deep."""
    from vbt.datalayer.service import items as _items

    cell = pa.struct([("name", pa.string()), ("level", pa.int32())])
    tissue = pa.struct([("efo_code", pa.string()), ("protein", pa.struct([("cell_type", pa.list_(cell))]))])
    rows = [{"id": "g1", "tissues": [{"efo_code": "UBERON_1", "protein": {"cell_type": [{"name": "a", "level": 1}]}},
                                     None,
                                     {"efo_code": "UBERON_2", "protein": {"cell_type": []}},
                                     {"efo_code": "UBERON_3", "protein": None}]},
            {"id": "g2", "tissues": []},
            {"id": "g3", "tissues": None},
            {"id": "g4", "tissues": [{"efo_code": "UBERON_4", "protein": {"cell_type": [None, {"name": "b", "level": 2}]}}]}]
    write(tmp_path, "expr", rows, pa.schema([("id", pa.string()), ("tissues", pa.list_(tissue))]))
    ctx = make_ctx(tmp_path, {"expr": {"kind": "entity", "path": "expr", "grain": "gene", "key": {"columns": ["id"]},
                                       "columns": {"id": {"role": "identifier", "self": True},
                                                   "tissues": {"role": "nested", "item_key": ["efo_code"], "fields": {
                                                       "efo_code": {"role": "identifier"},
                                                       "protein": {"role": "nested", "fields": {
                                                           "cell_type": {"role": "nested", "item_key": ["name", "level"],
                                                                         "fields": {"name": {"role": "category"},
                                                                                    "level": {"role": "category"}}}}}}}}}})
    reader = ctx.reader("s.expr")
    keys = ("rows", "null", "empty", "nonempty", "items", "null_items")
    for container in ("tissues[]", "tissues[].protein.cell_type[]"):
        lvls = _items.levels(container)
        leaf = reader.leaf(container)
        fast = _checks._arrow_container_counts(reader, lvls, leaf)
        slow = _items.ContainerCounts()
        for frag, rgs in reader.plan(None, [leaf], use_sidecars=False)[0]:
            for rg in (rgs if rgs is not None else [None]):
                for row in reader._read(frag, rg, [leaf], reader.footer(frag)):
                    for _ in _items.explode(row, lvls, slow):
                        pass
        assert fast is not None, container
        assert {k: getattr(fast, k) for k in keys} == {k: getattr(slow, k) for k in keys}, container
    tissues = _checks._arrow_container_counts(reader, _items.levels("tissues[]"), reader.leaf("tissues[]"))
    assert (tissues.null, tissues.empty, tissues.nonempty, tissues.items, tissues.null_items) == (1, 1, 2, 4, 1)
    monkeypatch.setattr(reader.fmt, "to_native", lambda *_a, **_k: pytest.fail("rows were converted"))
    _checks._arrow_container_counts(reader, _items.levels("tissues[]"), reader.leaf("tissues[]"))


# ---------------------------------------------------------------------------- memory estimates and calibration


def test_flat_values_cost_their_slot_even_when_dictionary_encoded(tmp_path):
    """25.09 target_prioritisation stores 17 int32 factors with few distinct values: dictionary encoding leaves
    almost no footer bytes, so the bytes-only seed estimated 11 MB for a 51 MB load. ``flat_value`` charges each
    flat value; it is off in the in-code seeds and set in configs/default.yaml."""
    from vbt.datalayer.memory.calibrate import stats_of_fragments
    from vbt.datalayer.memory.estimate import MemoryEstimator
    from vbt.datalayer.plugins.base import Fragment
    from vbt.datalayer.plugins.formats.parquet import ParquetFormat

    n = 50000
    path = tmp_path / "prio.parquet"
    pq.write_table(pa.table({"f": pa.array([i % 3 - 1 for i in range(n)], pa.int32())}), path)
    stats = stats_of_fragments(ParquetFormat(), [Fragment(uri=str(path), size=path.stat().st_size,
                                                          mtime_ns=path.stat().st_mtime_ns)])
    assert stats["columns"]["f"]["uncompressed_bytes"] < n          # far less than 4 bytes per value
    seed = MemoryEstimator(fragmentation=1.0)
    assert seed.peak_upstream(stats) < n
    calibrated = MemoryEstimator(fragmentation=1.0, object_overhead_bytes={"flat_value": 20})
    assert calibrated.peak_upstream(stats) >= 20 * n


def test_the_shipped_memory_factors_estimate_a_real_load():
    """configs/default.yaml estimates the 25.09 target load (3,031 MB measured over the loader's baseline) within
    10%; the original seeds said 12,297 MB, which no server limit on this host admits."""
    import json as _json

    from vbt.config import load_config
    from vbt.datalayer.memory.estimate import MB, MemoryEstimator
    from vbt.datalayer.settings import DataSettings

    stats = _json.loads((REPO / "tests" / "datalayer" / "real" / "ot_25_09_servers" / "target_stats.json").read_text())
    est = MemoryEstimator.from_settings(DataSettings.from_config(load_config(["mock"])))
    assert abs(est.peak_upstream(stats) / MB - 3031) / 3031 < 0.10
    assert MemoryEstimator().peak_upstream(stats) / MB > 12000


def test_calibration_measures_at_most_the_row_cap_of_a_one_group_shard(tmp_path):
    """The 25.09 study table is one row group of 1,964,234 rows: calibrating it read and walked every row and the
    data child ran out of memory. The sample keeps at most ``max_rows`` evenly spaced rows of the groups it picks."""
    from vbt.datalayer.memory.calibrate import calibrate_fragments, stats_of_fragments
    from vbt.datalayer.plugins.base import Fragment
    from vbt.datalayer.plugins.formats.parquet import ParquetFormat

    n = 40000
    path = tmp_path / "study.parquet"
    pq.write_table(pa.table({"studyId": [f"GCST{i:08d}" for i in range(n)], "nSamples": list(range(n))}), path)
    frag = Fragment(uri=str(path), size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
    fmt = ParquetFormat()
    cal = calibrate_fragments(fmt, [frag], table_stats=stats_of_fragments(fmt, [frag]), max_rows=2000)
    assert cal["rows_sampled"] == 2000 and cal["rows"] == n and cal["scale"] == n / 2000
    assert cal["sampled"] == [{"fragment": str(path), "row_group": 0, "rows": [n, 2000]}]
    assert cal["bytes_per_row"] > 50


def test_calibration_of_nested_rows_stays_within_the_seed_budget(tmp_path, monkeypatch):
    """25.09 expression and target_essentiality rows hold hundreds of nested items: 10,000 rows of each of three
    shards exhausted the data child. The sample keeps the rows the seed model expects to fit in the budget."""
    from vbt.datalayer.memory import calibrate as cal_mod
    from vbt.datalayer.memory.calibrate import calibrate_fragments, stats_of_fragments
    from vbt.datalayer.memory.estimate import MemoryEstimator
    from vbt.datalayer.plugins.base import Fragment
    from vbt.datalayer.plugins.formats.parquet import ParquetFormat

    screen = pa.struct([("depmapId", pa.string()), ("geneEffect", pa.float64())])
    tissue = pa.struct([("tissueId", pa.string()), ("screens", pa.list_(screen))])
    rows = [{"id": f"ENSG{i:011d}", "tissues": [{"tissueId": f"UBERON_{t}", "screens": [
        {"depmapId": f"ACH-{s:06d}", "geneEffect": -0.1 * s} for s in range(20)]} for t in range(5)]}
            for i in range(3000)]
    path = tmp_path / "ess.parquet"
    pq.write_table(pa.Table.from_pylist(rows, pa.schema([("id", pa.string()), ("tissues", pa.list_(tissue))])), path)
    frag = Fragment(uri=str(path), size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
    fmt = ParquetFormat()
    stats = stats_of_fragments(fmt, [frag])
    seed = MemoryEstimator()
    per_row = seed.peak_upstream(stats) / stats["rows"]
    monkeypatch.setattr(cal_mod, "SAMPLE_BUDGET_BYTES", int(per_row * 300) + 1)
    cal = calibrate_fragments(fmt, [frag], table_stats=stats, seed=seed)
    assert cal["rows_sampled"] == 300, cal["rows_sampled"]
    monkeypatch.setattr(cal_mod, "SAMPLE_BUDGET_BYTES", 1)
    assert calibrate_fragments(fmt, [frag], table_stats=stats, seed=seed)["rows_sampled"] == cal_mod.MIN_SAMPLE_ROWS


@needs_real
def test_real_target_prioritisation_r9_is_fast(tmp_path):
    """25.09 target_prioritisation -> target: all 78,726 targetIds resolve, in seconds (51 s before)."""
    ctx = _real_ctx(tmp_path / "cache")
    t0 = time.monotonic()
    model = check_table(ctx, "open_targets.target_prioritisation", "deep")
    seconds = time.monotonic() - t0
    (r9,) = checks(model, "R9")
    assert r9.ok and "78726 sampled value(s) resolve" in r9.detail, r9.detail
    assert seconds < 20, seconds


# ---------------------------------------------------------------------------- real data helpers


def _real_ctx(cache: Path) -> ServiceContext:
    from vbt.datalayer.catalog import build_catalog
    from vbt.datalayer.plugins.registry import discover

    assert REAL is not None
    registry = discover(entry_points=False)
    settings = DataSettings.from_dict({"descriptors_dir": str(REPO / "configs/data/sources"),
                                       "overlays_dir": str(REPO / "configs/data/overlays"),
                                       "cache_dir": str(cache)},
                                      project_root=REPO)
    variables = {"project_root": str(REPO), "env.OPEN_TARGETS_DATA_PATH": str(REAL)}
    return ServiceContext(settings, catalog=build_catalog(settings, registry, variables=variables), registry=registry)
