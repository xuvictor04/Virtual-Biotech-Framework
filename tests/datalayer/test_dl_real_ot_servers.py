"""Real Open Targets 25.09 through the unmodified upstream servers, the gateway and the data child.

Offline (always): regression tests for what the real tables showed, each on the smallest fixture with the
real shape (every 25.09 shard is a single row group; target and expression nest lists of structs):

* readiness checks: R9 references matched with Arrow ``is_in`` (target_prioritisation deep 51 s -> 2 s); samples
  that convert only the sampled rows of a one-group shard (study ``_stats`` > 4.6 GB -> 1.2 GB); item-table key
  checks sampled by parent row (target standard 184 s -> 52 s); flat key uniqueness and container counts on Arrow
  arrays; content-identity samples drawn in Arrow (interaction_evidence > 15 min -> 97 s); positional item keys;
* serving: scans convert a row group in slices (MemoryError on interaction), predicates on columns that only say
  what null means are pushed to Arrow, search ranks a row's own labels only (TP63 before TP53), resolver indexes
  are reused across sessions;
* memory: the shipped factors against a real load (target, 3.0 GB), flat values cost their slot, calibration
  samples bounded by rows and by the seed budget.

``VBT_DL_REAL_DATA=<dir>`` (the ``open_targets/25.09`` output directory downloaded from
https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/25.09/output/, or a ``data/real`` root holding it):
the checks on the real tables, and the six correctness tests (DATA_LAYER.md §19) through the unmodified target
and drug servers with the gateway enforcing and without it (those also need the upstream checkout: the submodule
or ``VBT_UPSTREAM``). Each server stays under 5 GB and the data child under 3 GB.
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


def test_a_wide_key_is_counted_on_arrow_arrays_too(tmp_path, monkeypatch):
    """25.09 interaction's key has seven parts (source, both interactors, both genes, both roles): the product of
    their cardinalities (about 5e21) did not fit 63 bits, so its 14.5 M keys were rendered in Python, minutes of
    the genetics and interaction session's first check. Codes combined so far are encoded again when the next part
    would overflow; the duplicates and examples are the scan's."""
    n = 60_000
    parts = ["sourceDatabase", "intA", "intB", "targetA", "targetB"]
    rows = [{"sourceDatabase": ("intact", "string")[i % 2], "intA": f"P{i:06d}", "intB": f"Q{(i * 7) % n:06d}",
             "targetA": f"ENSG{i:011d}", "targetB": None if i % 9 == 0 else f"ENSG{(i * 3) % n:011d}"} for i in range(n)]
    rows += [dict(rows[5]), dict(rows[9]), dict(rows[9]), dict(rows[n - 1])]
    assert 2 * n ** 4 > 2 ** 63                     # source x four near-unique parts

    def build(tmp: Path) -> ServiceContext:
        write(tmp, "edges", rows[: n // 2], part="part-00000")
        write(tmp, "edges", rows[n // 2:], part="part-00001", row_group_size=4096)
        return make_ctx(tmp, {"edges": {"kind": "fact", "path": "edges", "grain": "pair",
                                        "key": {"columns": parts, "nullable": ["targetB"], "check": "full"},
                                        "columns": {p: {"role": "identifier"} for p in parts}}})

    scans: list[Any] = []
    real_scan = TableReader.scan

    def counting_scan(self, predicate=None, **kw):
        if self.ref == "s.edges" and kw.get("columns") == []:
            scans.append(1)
        return real_scan(self, predicate, **kw)

    monkeypatch.setattr(TableReader, "scan", counting_scan)
    fast = check_table(build(tmp_path / "fast"), "s.edges", "standard")
    assert not scans, "the key was rendered row by row"
    monkeypatch.setattr(_checks, "_arrow_key_duplicates", lambda *a, **k: None)
    slow = check_table(build(tmp_path / "slow"), "s.edges", "standard")
    assert scans
    assert fast.key_check.duplicates == slow.key_check.duplicates == 4
    assert sorted(checks(fast, "R5b")[0].detail.split(": ", 1)[1].split(", ")) == \
        sorted(checks(slow, "R5b")[0].detail.split(": ", 1)[1].split(", "))


def _evidence(tmp: Path, n: int) -> ServiceContext:
    """interaction_evidence's shape: a content-identity table whose grouping key has a nullable identifier, every
    record stored twice; the copies in another file and row groups of another size."""
    rows = [{"interactionIdentifier": None if i % 3 else f"EBI-{i // 2}", "intA": f"P{i % 97:05d}",
             "intB": f"Q{i % 89:05d}", "targetA": f"ENSG{i % 97:011d}", "targetB": f"ENSG{i % 89:011d}",
             "hostOrganismTaxId": 9606 + i % 2, "evidenceScore": None if i % 4 else i / 7,
             "participantDetectionMethodA": [{"miIdentifier": f"MI:{i % 5:04d}", "shortName": "x"}]}
            for i in range(n)]
    write(tmp, "evidence", rows, part="part-00000", row_group_size=97)
    write(tmp, "evidence", rows[::-1], part="part-00001", row_group_size=61)
    key = ["interactionIdentifier", "intA", "intB", "targetA", "targetB"]
    return make_ctx(tmp, {"evidence": {
        "kind": "edges", "path": "evidence", "grain": "evidence",
        "key": {"columns": key, "nullable": ["interactionIdentifier"], "row_identity": "content_hash", "check": "sampled"},
        "columns": {**{k: {"role": "identifier"} for k in key[:3]},
                    # cleaned as in 25.09 (non-entity endpoints): the sample is drawn on the other parts
                    **{k: {"role": "identifier", "missing_values": ["-"]} for k in key[3:]},
                    "hostOrganismTaxId": {"role": "category"},
                    "evidenceScore": {"role": "category"}, "participantDetectionMethodA": {"role": "category"}}}})


def test_a_sampled_content_check_samples_prefix_blocks_before_converting_rows(tmp_path, monkeypatch):
    """The content check of 25.09 interaction_evidence (27.3 M rows, 24,280 exact copies) converted every column of
    every row to Python and then kept the 0.7% whose key prefix was sampled: more than 15 minutes of a session's
    first check. The prefix blocks are sampled on the Arrow arrays now, alike in every file and row group, so the
    copies of a sampled record are still found."""
    converted: list[int] = []
    real_take = TableReader._read_take

    def counting_take(self, frag, rg, leaves, indices, **kw):
        converted.append(len(indices))
        return real_take(self, frag, rg, leaves, indices, **kw)

    monkeypatch.setattr(TableReader, "_read_take", counting_take)
    n = 3000
    ctx = _evidence(tmp_path / "s", n)
    assert ctx.reader("s.evidence")._unclean("targetA")
    model = check_table(ctx, "s.evidence", "standard", sample_rows=300)
    (r5b,) = checks(model, "R5b")
    assert not r5b.ok and "equal to another row" in r5b.detail and "prefix blocks" in r5b.detail, r5b.detail
    scanned = int(checks(model, "R5")[0].detail.rsplit("(", 1)[1].split()[0])
    dups = model.key_check.duplicates
    assert scanned == 2 * dups and 0 < scanned < 2 * n // 4, (scanned, dups)    # both copies of each sampled record
    assert sum(converted) == scanned, "rows outside the sampled blocks were converted"
    deep = check_table(_evidence(tmp_path / "d", n), "s.evidence", "deep")
    assert deep.key_check.duplicates == n and "every key" in checks(deep, "R5b")[0].detail


# ---------------------------------------------------------------------------- scans convert a row group in slices


def _interaction(tmp: Path, n: int) -> ServiceContext:
    """interaction's shape: each shard one row group (1.3 M rows in 25.09), a struct column beside the key."""
    sp = pa.struct([("mnemonic", pa.string()), ("taxon_id", pa.int64())])
    schema = pa.schema([("sourceDatabase", pa.string()), ("targetA", pa.string()), ("intA", pa.string()),
                        ("targetB", pa.string()), ("intB", pa.string()), ("speciesB", sp), ("count", pa.int64()),
                        ("scoring", pa.float64())])
    rows = [{"sourceDatabase": ("intact", "string", "signor")[i % 3], "targetA": f"ENSG{i % 11:011d}",
             "intA": f"P{i:05d}", "targetB": None if i % 13 == 0 else f"ENSG{i % 7:011d}", "intB": f"Q{i:05d}",
             "speciesB": {"mnemonic": "human", "taxon_id": 9606}, "count": i, "scoring": None if i % 5 else i / 9}
            for i in range(n)]
    write(tmp, "interaction", rows, schema, part="part-00000")
    write(tmp, "interaction", rows[: n // 3], schema, part="part-00001")
    key = ["sourceDatabase", "targetA", "intA", "targetB", "intB"]
    return make_ctx(tmp, {"interaction": {
        "kind": "edges", "path": "interaction", "grain": "pair", "key": {"columns": key, "nullable": ["targetB"]},
        "columns": {**{k: {"role": "identifier"} for k in key},
                    # as in 25.09: a null endpoint is an interactor that is not a gene
                    **{k: {"role": "identifier", "missing": "non_entity"} for k in ("targetA", "targetB")},
                    "count": {"role": "category"},
                    "scoring": {"role": "category"}, "speciesB.mnemonic": {"role": "category"}}}})


def test_a_scan_converts_a_large_row_group_in_slices(tmp_path, monkeypatch):
    """A 25.09 interaction shard is one row group of 1.3 M rows. The scan behind a witness or a serve converted the
    key and filter columns of the whole group to Python at once, and the data child ran out of memory (MemoryError
    in ``to_pylist`` on get_interactions and get_interaction_network). The group is converted SCAN_CHUNK_ROWS at a
    time; the matches, their order and the totals are the same."""
    from vbt.datalayer.predicate import Eq
    from vbt.datalayer.service import reader as _reader

    def run(chunk: int) -> tuple[list[Any], Any, list[int]]:
        monkeypatch.setattr(_reader, "SCAN_CHUNK_ROWS", chunk)
        reader = ictx.reader("s.interaction")
        sizes: list[int] = []
        real = reader.fmt.to_native

        def to_native(tbl):
            sizes.append(tbl.num_rows)
            return real(tbl)

        monkeypatch.setattr(reader.fmt, "to_native", to_native)
        st = _reader.ScanStats()
        out = [(m.key, m.row.get("count"), m.row.get("speciesB")) for m in
               reader.scan(Eq("sourceDatabase", "string"), columns=None, stats=st)]
        agg = reader.aggregate(Eq("targetA", "ENSG00000000003"), key=["intA"], key_set_max=10_000)
        monkeypatch.setattr(reader.fmt, "to_native", real)
        return out, (st.total, st.scanned_bytes, agg.stats.total, agg.key_set), sizes

    ictx = _interaction(tmp_path, 500)
    whole, whole_totals, whole_sizes = run(1_000_000)
    sliced, sliced_totals, sliced_sizes = run(16)
    assert max(whole_sizes) > 100 and max(sliced_sizes) <= 16, (max(whole_sizes), max(sliced_sizes))
    expected = [i for i in range(500) if i % 3 == 1] + [i for i in range(500 // 3) if i % 3 == 1]
    assert sliced == whole and sorted(c for _, c, _ in whole) == sorted(expected)
    assert all(sp == {"mnemonic": "human", "taxon_id": 9606} for _, _, sp in whole)   # second-pass columns
    assert sliced_totals == whole_totals and whole_totals[0] == len(expected) and whole_totals[2] > 0


def test_a_predicate_on_a_column_that_only_says_what_null_means_is_pushed_to_arrow(tmp_path, monkeypatch):
    """``missing: non_entity`` (25.09 interaction targetA/targetB) says what a null endpoint means and rewrites no
    value, but it counted as cleaning: every get_interactions predicate stayed out of Arrow and each call converted
    all 14.5 M interaction rows. Only the matching rows are converted now, with the same matches and totals."""
    from vbt.datalayer.predicate import Eq
    from vbt.datalayer.service import reader as _reader

    ictx = _interaction(tmp_path, 500)
    reader = ictx.reader("s.interaction")
    assert not reader._unclean("targetA") and not reader._unclean("targetB")

    def run(pushed: bool) -> tuple[list[Any], Any, int]:
        if not pushed:
            monkeypatch.setattr(TableReader, "_unclean", lambda self, path: True)
        sizes: list[int] = []
        real = reader.fmt.to_native
        monkeypatch.setattr(reader.fmt, "to_native", lambda tbl: (sizes.append(tbl.num_rows), real(tbl))[1])
        st = _reader.ScanStats()
        out = sorted((m.key, m.row.get("count")) for m in reader.scan(Eq("targetB", "ENSG00000000003"), stats=st))
        total, *_ = reader.count(Eq("targetB", "ENSG00000000003"))
        monkeypatch.undo()
        return out, (st.total, total), sum(sizes)

    pushed, pushed_totals, pushed_converted = run(True)
    scanned, scanned_totals, scanned_converted = run(False)
    assert pushed == scanned and pushed_totals == scanned_totals and pushed_totals[0] == len(pushed) > 0
    assert pushed_converted < scanned_converted / 3, (pushed_converted, scanned_converted)


# ---------------------------------------------------------------------------- resolver sidecars are reused


def test_a_built_resolver_index_is_reused_for_the_same_data(tmp_path, monkeypatch):
    """Each new session's first call asked the data child to build the resolver index again, and it did: 30 s for
    the 25.09 ensembl_gene index (465,164 rows) although the same file was on disk. An index already built for the
    universe table's fingerprint is returned as it is; ``force`` rebuilds it."""
    from vbt.datalayer.service.verbs import index_build

    write(tmp_path, "target", [{"id": f"ENSG{i:011d}", "approvedSymbol": f"G{i}"} for i in range(50)])
    ctx = make_ctx(tmp_path, {"target": {"kind": "entity", "path": "target", "grain": "gene", "key": {"columns": ["id"]},
                                         "columns": {"id": {"role": "identifier", "id_type": "gene", "self": True},
                                                     "approvedSymbol": {"role": "label", "of": "id"}}}},
                   id_types={"gene": {"plugin": "ensembl_gene", "universe": "target.id"}})
    first = index_build.build_resolver_index(ctx, "s", "gene")
    assert first.rows > 0 and Path(first.path).is_file()
    monkeypatch.setattr(index_build, "resolver_rows", lambda *a, **k: pytest.fail("the index was built again"))
    again = index_build.build_resolver_index(ctx, "s", "gene")
    assert (again.path, again.rows, again.fingerprint) == (first.path, first.rows, first.fingerprint)
    with pytest.raises(pytest.fail.Exception):
        index_build.build_resolver_index(ctx, "s", "gene", force=True)
    monkeypatch.undo()
    Path(first.path).write_bytes(b"not a sidecar")  # an unreadable file is rebuilt, not returned
    rebuilt = index_build.build_resolver_index(ctx, "s", "gene")
    assert (rebuilt.rows, rebuilt.fingerprint) == (first.rows, first.fingerprint)


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


@needs_real
def test_real_standard_checks_of_the_target_server_tables_are_ready_in_minutes(tmp_path):
    """The session check of the target server's tables (standard depth) is what its first call waits for: 612 s on
    25.09 before the Arrow key, container and reference checks, 178 s after (table by table in separate processes).
    In one context here: every table ready, expression under 90 s, target under 120 s."""
    ctx = _real_ctx(tmp_path / "cache")
    for ref, limit in (("open_targets.expression", 90), ("open_targets.target", 120),
                       ("open_targets.association_overall_direct", 60)):
        t0 = time.monotonic()
        model = check_table(ctx, ref, "standard")
        seconds = time.monotonic() - t0
        assert model.status == "ready", [c.detail for c in model.checks if not c.ok and c.level == "error"]
        assert seconds < limit, (ref, seconds)


# ---------------------------------------------------------------------------- real data: the six tests, both modes

PCSK9 = "ENSG00000169174"
TP53 = "ENSG00000141510"


def _oracle_known_drug_top(k: int) -> tuple[list[list[Any]], int]:
    """Independent pyarrow answer: PCSK9's known_drug rows by phase desc (nulls last), ties by the key ascending."""
    import math

    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    assert REAL is not None
    rows = ds.dataset(str(REAL / "known_drug"), format="parquet").to_table(
        columns=["drugId", "targetId", "diseaseId", "phase", "status"], filter=pc.field("targetId") == PCSK9).to_pylist()

    def order(r: dict[str, Any]) -> tuple[Any, ...]:
        p = r["phase"]
        unknown = p is None or (isinstance(p, float) and math.isnan(p))
        return (unknown, 0 if unknown else -p, *((v is None, "" if v is None else str(v))
                                                for v in (r["drugId"], r["targetId"], r["diseaseId"], r["status"])))
    ranked = sorted(rows, key=order)
    return [[r["drugId"], r["diseaseId"], r["phase"], r["status"]] for r in ranked[:k]], len(rows)


def _oracle_pgx(drug: str, target: str | None = None) -> int:
    import pyarrow.dataset as ds

    assert REAL is not None
    rows = ds.dataset(str(REAL / "pharmacogenomics"), format="parquet").to_table(
        columns=["targetFromSourceId", "drugs"]).to_pylist()
    return sum(1 for r in rows if (target is None or r["targetFromSourceId"] == target)
               and any((d or {}).get("drugId") == drug for d in r["drugs"] or []))


def _live(mode: str, servers: tuple[str, ...], tmp_path: Path) -> Any:
    from dl_upstream import DataEnv, LiveBridge, upstream_missing

    if upstream_missing():
        pytest.skip(upstream_missing())
    overrides = {"data": {"memory": {"default_server_mb": 5000, "limit_kind": "rlimit_data", "host_budget_mb": "off"},
                          "service": {"mem_limit_mb": 3000, "timeout_s": 1800}}}
    bridge = LiveBridge(servers, env=DataEnv(ot_root=REAL, output_dir=tmp_path / "out"), gateway=mode == "enforce",
                        tmp_path=tmp_path / mode, overrides=overrides)
    if bridge.gateway is not None:
        bridge.run(bridge.gateway.wait_readiness(1500), timeout=1600)    # the session check, before the first call
    return bridge


def _keys(r: Any) -> list[Any]:
    prov = r.provenance.to_dict() if hasattr(r.provenance, "to_dict") else (r.provenance or {})
    return (prov.get("result") or {}).get("row_keys") or []


@needs_real
@pytest.mark.correctness
def test_real_six_tests_through_the_unmodified_servers_with_the_gateway(tmp_path):
    """CT-1..CT-6 on the real 25.09 tables, gateway enforcing: identifiers of the wrong form are resolved or
    rejected, the unknown ENSG is not_found, the derived name search ranks the exact symbol first, the known-drug
    top-k equals the pyarrow answer with the honest total, wrong argument values are refused, the unconfirmable
    safety filter is unsupported, and pharmacogenomics is not a false empty."""
    bridge = _live("enforce", ("target", "drug"), tmp_path)
    try:
        for tid, rule in (("PCSK9", "label_exact:approvedSymbol"), ("ENSG00000169174.12", "normalized:strip_version"),
                          ("ensg00000169174", "normalized:upper"), ("NARC1", "synonym:alias")):
            r = bridge.call("target", "get_target_info", {"target_id": tid})
            assert not r.is_error and _keys(r) == [[PCSK9]] and rule in str(r.header.get("resolved")), r.text[:400]
        r = bridge.call("target", "get_target_info", {"target_id": "ENSG00000999999"})
        assert r.is_error and r.kind == "not_found" and r.payload.get("citable") is False and r.payload.get("tried")
        r = bridge.call("target", "search_targets_by_name", {"query": "TP53", "limit": 1})
        first = r.rows("results")[0]
        assert first["id"] == TP53 and first["match"] == "exact", first
        top, total = _oracle_known_drug_top(5)
        r = bridge.call("drug", "search_known_drugs", {"target_id": "PCSK9", "limit": 5})
        got = [[d["drugId"], d["diseaseId"], d["phase"], d["status"]] for d in r.rows("drugs")]
        assert got == top and r.header.get("total") == total, (got, r.header)
        for limit in (0, -1):
            r = bridge.call("drug", "search_known_drugs", {"target_id": PCSK9, "limit": limit})
            assert r.is_error and r.kind == "invalid_argument", r.text[:300]
        r = bridge.call("target", "prioritize_targets", {"sort_by": "nonexistent"})
        assert r.is_error and r.kind == "invalid_argument", r.text[:300]
        r = bridge.call("target", "prioritize_targets", {"no_safety_events": True})
        assert r.is_error and r.kind == "unsupported_filter", r.text[:300]
        r = bridge.call("drug", "get_pharmacogenomics", {"drug_id": "CHEMBL3"})
        assert r.header.get("total") == _oracle_pgx("CHEMBL3") and r.rows("pgx_relationships"), r.header
        r = bridge.call("drug", "get_pharmacogenomics", {"target_id": "ENSG00000112038", "drug_id": "CHEMBL3"})
        assert r.header.get("total") == _oracle_pgx("CHEMBL3", "ENSG00000112038")
        assert all(any(d.get("drugId") == "CHEMBL3" for d in row["drugs"]) for row in r.rows("pgx_relationships"))
        r = bridge.call("target", "get_mouse_phenotype", {"target_id": PCSK9})
        assert r.header.get("total") == 18 and len(r.rows("phenotypes")) == 18, r.header
        r = bridge.call("target", "get_target_safety_profile", {"target_id": TP53})
        assert r.header.get("status") == "empty" and r.header.get("coverage") == "unknown" and "message" not in r.obj
    finally:
        bridge.close()


@needs_real
@pytest.mark.correctness
def test_real_wrong_answers_without_the_gateway(tmp_path):
    """The same calls on the unmodified servers with no gateway: today's real wrong answers, pinned so an upstream
    fix shows up here."""
    bridge = _live("off", ("target", "drug"), tmp_path)
    try:
        r = bridge.call("target", "get_target_info", {"target_id": "PCSK9"})
        assert not r.is_error and r.obj.get("error") == "Target PCSK9 not found"
        r = bridge.call("target", "search_targets_by_name", {"query": "TP53", "limit": 1})
        assert r.rows("results")[0]["id"] == "ENSG00000120471"            # TP53AIP1, first in file order
        r = bridge.call("drug", "search_known_drugs", {"target_id": PCSK9, "limit": 5})
        assert max(d["phase"] for d in r.rows("drugs")) == 3.0            # 23 phase-4 rows exist
        r = bridge.call("drug", "search_known_drugs", {"target_id": PCSK9, "limit": -1})
        assert not r.is_error and r.obj.get("count") == _oracle_known_drug_top(0)[1] - 1
        r = bridge.call("drug", "get_pharmacogenomics", {"drug_id": "CHEMBL3"})
        assert not r.is_error and r.obj.get("count") == 0                 # 508 rows hold CHEMBL3
        r = bridge.call("target", "get_mouse_phenotype", {"target_id": PCSK9})
        assert not r.is_error and r.obj.get("count") == 0                 # 18 rows for PCSK9
        r = bridge.call("target", "prioritize_targets", {"no_safety_events": True, "limit": 20})
        assert any(t.get("hasSafetyEvent") == -1 for t in r.rows("targets"))   # an event recorded counts as none
    finally:
        bridge.close()
