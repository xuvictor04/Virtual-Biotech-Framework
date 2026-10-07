"""Layout conformance suite (§9.3 L-1..L-7) over the registered layout plugins, plus explicit cases:
hive partitions restored and pruned to one partition, truncated shards, stat-only signatures,
probe findings (R1/R2) and fingerprints of large files."""

from __future__ import annotations

import builtins
import io
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

from vbt.datalayer.plugins.base import FormatError, Fragment, LayoutSpec, Manifest  # noqa: E402
from vbt.datalayer.plugins.conformance import golden as g  # noqa: E402
from vbt.datalayer.plugins.conformance.format import scan_rows  # noqa: E402
from vbt.datalayer.plugins.conformance.layout import *  # noqa: E402,F401,F403  (collects L-1..L-7)
from vbt.datalayer.plugins.formats.parquet import ParquetFormat  # noqa: E402
from vbt.datalayer.plugins.layouts import (  # noqa: E402
    PROBE_STATUS,
    is_data_name,
    manifest_entry,
    partition_label,
    prune_fragments,
)
from vbt.datalayer.plugins.layouts.hive import HiveLayout, convert_partition  # noqa: E402
from vbt.datalayer.plugins.layouts.sharded_dir import ShardedDirLayout  # noqa: E402
from vbt.datalayer.plugins.layouts.single_file import SingleFileLayout  # noqa: E402
from vbt.datalayer.plugins.layouts.upstream_only import UpstreamOnlyLayout  # noqa: E402
from vbt.datalayer.predicate import And, Eq, Not  # noqa: E402

SRC = Path(__file__).resolve().parents[2] / "src"
PARQUET = ParquetFormat()
FMT = PARQUET.conformance_cases()
HIVE = HiveLayout()
SHARDED = ShardedDirLayout()
SINGLE = SingleFileLayout()
HIVE_SPEC = LayoutSpec(table="ot.evidence", path="evidence", partitions={"sourceId": "string", "year": "int64"})


def _items(items, name):
    return [i for i in items if i.name == name]


# ---------------------------------------------------------------------------
# Hive: partitions restored with declared types; pruning reads one partition
# ---------------------------------------------------------------------------

def test_hive_partition_restored_and_pruning_reads_one_partition(tmp_path):
    tree = g.write_tree("hive", str(tmp_path), FMT)
    frags = HIVE.fragments(tree.root, HIVE_SPEC)
    target = [f for f in frags if f.partition == {"sourceId": "chembl", "year": 2020}]
    assert len(target) == 1
    for f in frags:                                         # every other partition is unreadable
        if f is not target[0]:
            g.truncate(f.uri)
    p = And((Eq("sourceId", "chembl"), Eq("year", 2020)))
    assert prune_fragments(frags, p) == target
    rows = scan_rows(PARQUET, frags, predicate=p, partitions=HIVE.partition_columns(HIVE_SPEC))
    assert [r["id"] for r in rows] == ["ENSG00000169174", "Straße"]
    assert {(r["sourceId"], r["year"]) for r in rows} == {("chembl", 2020)}
    assert all(type(r["year"]) is int for r in rows)
    with pytest.raises(FormatError):
        scan_rows(PARQUET, frags, predicate=Eq("sourceId", "europepmc"), partitions={"sourceId": "string"})


def test_hive_values_unquoted_typed_and_null(tmp_path):
    assert convert_partition("__HIVE_DEFAULT_PARTITION__", "string") is None
    assert convert_partition("2020", "int64") == 2020 and convert_partition("2020-01-31", "date") == date(2020, 1, 31)
    assert convert_partition("a%2Fb%20c", "string") == "a/b c" and convert_partition("x", "int64") == "x"
    root = tmp_path / "ev"
    for rel in ("sourceId=a%2Fb/year=2020/part-0.parquet", "sourceId=c/year=oops/part-0.parquet"):
        g.write_parquet(g.build_scalars().slice(0, 1), str(root / rel))
    spec = LayoutSpec(table="t", path="ev", partitions={"sourceId": "string", "year": "int64"})
    frags = HIVE.fragments(str(tmp_path), spec)
    assert [f.partition for f in frags] == [{"sourceId": "a/b", "year": 2020}, {"sourceId": "c", "year": "oops"}]
    assert [i.detail for i in _items(HIVE.probe(str(tmp_path), spec, None), "partition_type")] == \
        ["year values not int64: oops"]
    assert partition_label({"sourceId": None, "year": 2021}) == "sourceId=__HIVE_DEFAULT_PARTITION__/year=2021"


def test_hive_probe_partition_vocabulary_partial_files_and_empty_partitions(tmp_path):
    tree = g.write_tree("hive", str(tmp_path), FMT)
    os.makedirs(os.path.join(tree.root, "evidence", "sourceId=ot_genetics", "year=2020"))
    spec = LayoutSpec(table="ot.evidence", path="evidence", partitions={"sourceId": "string", "year": "int64"},
                      partition_expect={"sourceId": ["chembl", "europepmc", "gwas_credible_sets", None]})
    items = HIVE.probe(tree.root, spec, None)
    assert [(i.name, i.partition) for i in items if not i.ok] == [
        ("partial_files", "sourceId=europepmc/year=2020"),
        ("partition_missing", "sourceId=gwas_credible_sets"),
        ("partition_empty", "sourceId=ot_genetics/year=2020"),
    ]
    assert all(i.name in PROBE_STATUS for i in items)
    spec2 = LayoutSpec(table="ot.evidence", path="evidence", partitions={"sourceId": "string"},
                       partition_expect={"sourceId": ["chembl", None]})
    extra = [i.partition for i in HIVE.probe(tree.root, spec2, None) if i.name == "partition_extra"]
    assert extra == ["sourceId=europepmc"]


# ---------------------------------------------------------------------------
# Sharded directories: listing by name, truncated shards
# ---------------------------------------------------------------------------

def test_truncated_shard_raises_format_error_not_fewer_rows(tmp_path):
    tree = g.write_tree("sharded", str(tmp_path), FMT)
    spec = LayoutSpec(table="ot.target", path="target")
    frags = SHARDED.fragments(tree.root, spec)
    assert len(scan_rows(PARQUET, frags)) == 10
    g.truncate(frags[1].uri)
    listed = SHARDED.fragments(tree.root, spec)
    assert [f.uri for f in listed] == [f.uri for f in frags]
    batches = PARQUET.scan(listed, columns=["id"], predicate=None, partitions={})
    seen = 0
    with pytest.raises(FormatError) as err:
        for b in batches:
            seen += b.num_rows
    assert err.value.fragment == frags[1].uri and seen < 10      # the error, never a shorter success


def test_listing_excludes_junk_by_name(tmp_path):
    assert is_data_name("part-00000-1a2b.snappy.parquet") and not is_data_name("part-1.parquet.part")
    assert not is_data_name("x.part.json", "*") and not is_data_name("_SUCCESS", "*") and not is_data_name(".x.parquet")
    tree = g.write_tree("sharded", str(tmp_path), FMT)
    spec = LayoutSpec(table="t", path="target")
    items = SHARDED.probe(tree.root, spec, None)
    partial = _items(items, "partial_files")
    assert len(partial) == 1 and "part-00003.parquet.part" in partial[0].detail and "_temporary/" in partial[0].detail
    one = LayoutSpec(table="t", path="target/part-00000.parquet")
    assert [os.path.basename(f.uri) for f in SHARDED.fragments(tree.root, one)] == ["part-00000.parquet"]
    csv = LayoutSpec(table="t", path="target", format="csv")
    assert SHARDED.fragments(tree.root, csv) == []


def test_single_file_and_fragment_keys(tmp_path):
    tree = g.write_tree("single", str(tmp_path), FMT)
    spec = LayoutSpec(table="z.cohorts", path="cohorts/GSE*.parquet",
                      fragment_key={"name": "cohort", "from": "filename_regex", "pattern": r"(GSE\d+)"})
    frags = SINGLE.fragments(tree.root, spec)
    assert [f.fragment_key for f in frags] == ["GSE1001", "GSE2002"]
    one = LayoutSpec(table="z.one", path="cohorts/GSE1001.parquet")
    assert [f.fragment_key for f in SINGLE.fragments(tree.root, one)] == [None]
    assert SINGLE.fragments(tree.root, LayoutSpec(table="z", path="cohorts/missing.parquet")) == []
    assert SINGLE.fragments(str(tmp_path / "cohorts" / "GSE1001.parquet"), LayoutSpec(table="z", path=None))
    unmatched = LayoutSpec(table="z.cohorts", path="cohorts/GSE*.parquet",
                           fragment_key={"name": "cohort", "from": "filename_regex", "pattern": r"(GSE9\d+)"})
    items = SINGLE.probe(tree.root, unmatched, None)
    assert [i.ok for i in _items(items, "fragment_key")] == [False]
    assert [i.ok for i in _items(items, "partial_files")] == [False]
    directory = LayoutSpec(table="z", path="cohorts/*.parquet", fragment_key={"name": "d", "from": "directory"})
    assert {f.fragment_key for f in SINGLE.fragments(tree.root, directory)} == {"cohorts"}


# ---------------------------------------------------------------------------
# Manifests (R2), signatures and fingerprints
# ---------------------------------------------------------------------------

def test_probe_manifest_attribution(tmp_path):
    tree = g.write_tree("sharded", str(tmp_path), FMT)
    spec = LayoutSpec(table="ot.target", path="target")
    sizes = {rel: os.path.getsize(os.path.join(tree.root, rel)) for rel in tree.data_files}
    entries = {rel: {"bytes": size} for rel, size in sizes.items()}
    entries["target/part-00009.parquet"] = {"bytes": 10}           # listed, never downloaded
    entries["target/part-00000.parquet"] = {"bytes": sizes["target/part-00000.parquet"] + 1}
    entries["disease/part-00000.parquet"] = {"bytes": 1}           # another table: not attributed here
    items = SHARDED.probe(tree.root, spec, Manifest(path="manifest.json", entries=entries, complete=False))
    failed = {i.name: i.detail for i in items if not i.ok}
    assert set(failed) == {"partial_files", "manifest_complete", "manifest_missing", "manifest_bytes"}
    assert "part-00009" in failed["manifest_missing"] and "disease" not in failed["manifest_missing"]
    assert "part-00000.parquet" in failed["manifest_bytes"]
    del entries["target/sub/part-00002.parquet"]
    unlisted = _items(SHARDED.probe(tree.root, spec, Manifest(path=None, entries=entries)), "manifest_unlisted")
    assert unlisted and unlisted[0].level == "warning"
    assert manifest_entry(os.path.join(tree.root, "target/part-00001.parquet"), Manifest(None, entries)) == \
        entries["target/part-00001.parquet"]


def test_signature_never_opens_files(tmp_path, monkeypatch):
    g.write_tree("hive", str(tmp_path), FMT)
    g.write_tree("single", str(tmp_path), FMT)

    def refuse(*a, **k):
        raise AssertionError("opened a file")

    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(io, "open", refuse)
    monkeypatch.setattr(os, "open", refuse)
    sigs = {HIVE.signature(str(tmp_path), HIVE_SPEC),
            SINGLE.signature(str(tmp_path), LayoutSpec(table="t", path="cohorts/GSE*.parquet")),
            SINGLE.signature(str(tmp_path), LayoutSpec(table="t", path="cohorts/GSE1001.parquet")),
            SHARDED.signature(str(tmp_path), LayoutSpec(table="t", path="nowhere")),
            UpstreamOnlyLayout().signature(str(tmp_path), LayoutSpec(table="t", path=None))}
    assert len(sigs) == 5 and all(s.startswith("sig1:") for s in sigs)


def test_signature_sees_partial_files_next_to_a_single_file(tmp_path):
    tree = g.write_tree("single", str(tmp_path), FMT)
    spec = LayoutSpec(table="t", path="cohorts/GSE1001.parquet")
    before = SINGLE.signature(tree.root, spec)
    (tmp_path / "cohorts" / "GSE1001.parquet.part").write_bytes(b"x")
    assert SINGLE.signature(tree.root, spec) != before
    assert [i.ok for i in _items(SINGLE.probe(tree.root, spec, None), "partial_files")] == [False]


def test_fingerprint_large_files_use_stat_and_footer(tmp_path, monkeypatch):
    tree = g.write_tree("sharded", str(tmp_path), FMT)
    spec = LayoutSpec(table="t", path="target")
    frags = SHARDED.fragments(tree.root, spec)
    assert SHARDED.fingerprint(frags, None).startswith("fp1:sha256:")
    monkeypatch.setattr(ShardedDirLayout, "content_hash_max_bytes", 10)
    fp = SHARDED.fingerprint(frags, None)
    assert fp.startswith("fp1:stat:") and SHARDED.fingerprint(frags, None) == fp
    path = frags[0].uri
    data = bytearray(open(path, "rb").read())
    data[-12] ^= 0xFF                                       # inside the footer; same size
    st = os.stat(path)
    with open(path, "wb") as fh:
        fh.write(bytes(data))
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))     # same mtime: only the footer hash can tell
    assert SHARDED.fingerprint(SHARDED.fragments(tree.root, spec), None) != fp
    manifest = Manifest(None, {rel: {"sha256": "ab" * 32} for rel in tree.data_files})
    assert SHARDED.fingerprint(frags, manifest).startswith("fp1:manifest:")
    moved = tmp_path / "moved"
    os.rename(tree.root + "/target", str(moved))
    frags_moved = SHARDED.fragments(str(tmp_path), LayoutSpec(table="t", path="moved"))
    monkeypatch.setattr(ShardedDirLayout, "content_hash_max_bytes", 1 << 30)
    assert SHARDED.fingerprint(frags_moved, None).startswith("fp1:sha256:")


def test_upstream_only(tmp_path):
    layout = UpstreamOnlyLayout()
    spec = LayoutSpec(table="clinicaltrials.studies", path=None)
    assert layout.fragments(str(tmp_path), spec) == [] and layout.partition_columns(spec) == {}
    [item] = layout.probe(str(tmp_path), spec, None)
    assert item.ok and item.detail == "served by upstream tools only" and PROBE_STATUS[item.name] == "ready"
    as_of = layout.as_of(str(tmp_path), spec)
    assert as_of.endswith("Z") and len(as_of) == 20
    assert layout.signature("a", spec) == layout.signature("b", spec)
    assert layout.fingerprint([], None) == "fp1:upstream_only"


def test_prune_fragments_is_three_valued():
    frags = [Fragment(f"/x/{i}", 1, 1, partition={"s": s}) for i, s in enumerate(["a", "b", None])]
    assert [f.partition["s"] for f in prune_fragments(frags, Not(Eq("s", "a")))] == ["b"]
    assert prune_fragments(frags, None) == frags
    assert prune_fragments(frags, Eq("other", 1)) == frags            # not a partition-only conjunct
    from vbt.datalayer.predicate import Param

    assert prune_fragments(frags, Eq("s", Param("source"))) == frags  # decided at call time
    assert [f.partition["s"] for f in prune_fragments(frags, Eq("s", Param("source")), {"source": "b"})] == ["b"]


def test_layout_modules_import_without_pyarrow(tmp_path):
    code = (
        "import sys\n"
        "sys.modules['pyarrow'] = None\n"
        "sys.modules['pandas'] = None\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "from vbt.datalayer.plugins.base import LayoutSpec\n"
        "from vbt.datalayer.plugins.layouts.hive import HiveLayout\n"
        "from vbt.datalayer.plugins.layouts.single_file import SingleFileLayout\n"
        "from vbt.datalayer.plugins.layouts.upstream_only import UpstreamOnlyLayout\n"
        f"print(HiveLayout().signature({str(tmp_path)!r}, LayoutSpec(table='t', path='x'))[:5])\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "sig1:"
