"""Layout conformance suite (§9.3, L-1..L-7), parametrized over the registered layout plugins.

Each plugin's ``conformance_cases()`` returns :class:`~.golden.LayoutCases`: which golden tree
(``single``, ``sharded``, ``hive``, or ``none`` for ``upstream_only``) and the ``LayoutSpec``
to read it with; a custom ``build`` writes trees the standard ones do not fit. Trees are
written with the format plugin the case names (``parquet``), which L-2 and L-7 also scan with.
L-3 exempts ``upstream_only`` layouts, which report "served by upstream tools only"; L-6 runs
for layouts with the ``live`` or ``upstream_only`` capability.
"""

from __future__ import annotations

import builtins
import functools
import io
import os
import shutil
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")

from ...predicate import Eq, In, IsNull, Not  # noqa: E402
from ..base import FormatError, Fragment, LayoutSpec, Manifest  # noqa: E402
from ..layouts import prune_fragments  # noqa: E402
from . import has_capability, selected_plugins  # noqa: E402
from .golden import FormatCases, GoldenTree, LayoutCases, truncate, write_tree  # noqa: E402

LAYOUT_PLUGINS = selected_plugins("layout")


@functools.lru_cache(maxsize=1)
def _registry() -> Any:
    from ...settings import DataSettings
    from ..registry import discover

    return discover(DataSettings.from_env())


def layout_cases(plugin: Any) -> LayoutCases:
    cases = plugin.conformance_cases()
    if not isinstance(cases, LayoutCases):
        pytest.fail(f"{plugin.name}: conformance_cases() must return LayoutCases")
    return cases


def format_for(cases: LayoutCases) -> tuple[Any, FormatCases]:
    plugin = _registry().find("format", cases.format)
    if plugin is None:
        pytest.skip(f"format plugin {cases.format!r} is not registered")
    fmt = plugin.conformance_cases()
    if not isinstance(fmt, FormatCases):
        pytest.skip(f"format {cases.format!r} cannot write goldens")
    return plugin, fmt


def build_tree(plugin: Any, root: Any) -> tuple[LayoutCases, GoldenTree, LayoutSpec, Any]:
    """Write the plugin's golden tree under ``root``; returns ``(cases, tree, spec, format plugin)``."""
    cases = layout_cases(plugin)
    if cases.tree == "none":
        pytest.skip(f"{plugin.name} reads no files")
    fmt_plugin, fmt = format_for(cases)
    root = str(root)
    tree = cases.build(root, fmt) if cases.build is not None else write_tree(cases.tree, root, fmt)
    return cases, tree, cases.spec(), fmt_plugin


def relpaths(frags: list[Fragment], root: str) -> list[str]:
    return [os.path.relpath(f.uri, root).replace(os.sep, "/") for f in frags]


def scan_all(fmt_plugin: Any, frags: list[Fragment], partitions: Any, predicate: Any = None) -> list[dict[str, Any]]:
    from .format import scan_rows

    return scan_rows(fmt_plugin, frags, predicate=predicate, partitions=partitions)


def _params(*, files: bool = False, caps: tuple[str, ...] = ()) -> list[Any]:
    out = []
    for p in LAYOUT_PLUGINS:
        if files and has_capability(p, "upstream_only"):
            continue
        if caps and not any(has_capability(p, c) for c in caps):
            continue
        out.append(pytest.param(p, id=p.name))
    return out


# ---------------------------------------------------------------------------
# L-1 listing by name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("plugin", _params(files=True))
def test_l1_fragments_by_name(plugin: Any, tmp_path: Any) -> None:
    cases, tree, spec, _fmt = build_tree(plugin, tmp_path)
    frags = plugin.fragments(tree.root, spec)
    assert relpaths(frags, tree.root) == list(tree.data_files)
    listed = set(relpaths(frags, tree.root))
    assert not listed & set(tree.junk), "junk listed as data"
    for f in frags:
        assert f.size and f.mtime_ns, f.uri
    if cases.fragment_key:
        keys = [f.fragment_key for f in frags]
        assert all(keys) and len(set(keys)) == len(keys), keys
    assert plugin.fragments(tree.root, spec) == frags, "listing is deterministic"


# ---------------------------------------------------------------------------
# L-2 hive partitions: restored with their declared types, pruned three-valued
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("plugin", _params(files=True))
def test_l2_partitions_restored_and_pruned(plugin: Any, tmp_path: Any) -> None:
    cases, tree, spec, fmt_plugin = build_tree(plugin, tmp_path)
    declared = plugin.partition_columns(spec)
    if not cases.partitions:
        assert declared == {}
        pytest.skip(f"{plugin.name} has no partitions")
    assert declared == dict(cases.partitions)
    frags = plugin.fragments(tree.root, spec)
    for f in frags:
        want = tree.partitions[os.path.relpath(f.uri, tree.root).replace(os.sep, "/")]
        assert dict(f.partition) == dict(want)
        assert all(type(f.partition[k]) is type(v) for k, v in want.items())
    rows = scan_all(fmt_plugin, frags, declared)
    key = lambda r: r["id"]  # noqa: E731
    assert sorted(rows, key=key) == sorted(fmt_plugin.to_native(tree.table), key=key)
    by_source: dict[Any, list[str]] = {}
    for r in fmt_plugin.to_native(tree.table):
        by_source.setdefault(r["sourceId"], []).append(r["id"])
    # three-valued pruning: the null partition never satisfies Eq or Not(Eq)
    assert len(prune_fragments(frags, Eq("sourceId", "chembl"))) == 2
    assert {f.partition["sourceId"] for f in prune_fragments(frags, Not(Eq("sourceId", "chembl")))} == {"europepmc"}
    assert [f.partition["sourceId"] for f in prune_fragments(frags, IsNull("sourceId"))] == [None]
    assert len(prune_fragments(frags, In("year", (2021,)))) == 2
    # a damaged partition that the predicate excludes is never read
    damaged = [f for f in frags if f.partition.get("sourceId") == "europepmc"]
    truncate(damaged[0].uri)
    got = scan_all(fmt_plugin, frags, declared, Eq("sourceId", "chembl"))
    assert sorted(r["id"] for r in got) == sorted(by_source["chembl"])
    assert {(r["sourceId"], type(r["year"])) for r in got} == {("chembl", int)}
    got = scan_all(fmt_plugin, frags, declared, IsNull("sourceId"))
    assert sorted(r["id"] for r in got) == sorted(by_source[None])
    with pytest.raises(FormatError):
        scan_all(fmt_plugin, frags, declared, Not(Eq("sourceId", "chembl")))


# ---------------------------------------------------------------------------
# L-3 empty or missing location -> not ready, never an empty table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("plugin", _params())
def test_l3_missing_or_empty_location(plugin: Any, tmp_path: Any) -> None:
    cases = layout_cases(plugin)
    spec = cases.spec()
    if has_capability(plugin, "upstream_only"):
        items = plugin.probe(str(tmp_path), spec, None)
        assert items and all(i.ok for i in items)
        assert any("served by upstream tools only" in i.detail for i in items)
        assert plugin.fragments(str(tmp_path), spec) == []
        return
    missing = plugin.probe(str(tmp_path / "nowhere"), spec, None)
    assert any(not i.ok and i.level == "error" for i in missing), missing
    assert plugin.fragments(str(tmp_path / "nowhere"), spec) == []
    empty_root = tmp_path / "empty"
    location = empty_root / (spec.path or "")
    if any(ch in (spec.path or "") for ch in "*?["):
        location = empty_root / os.path.dirname(spec.path or "")
    location.mkdir(parents=True)
    items = plugin.probe(str(empty_root), spec, None)
    assert any(not i.ok and i.level == "error" for i in items), items
    assert plugin.fragments(str(empty_root), spec) == []


# ---------------------------------------------------------------------------
# L-4 fingerprints, L-5 stat-only signatures, L-6 as_of
# ---------------------------------------------------------------------------

def _flip_byte(path: str) -> None:
    with open(path, "rb+") as fh:
        fh.seek(8)
        b = fh.read(1)
        fh.seek(8)
        fh.write(bytes([b[0] ^ 0xFF]))


@pytest.mark.parametrize("plugin", _params())
def test_l4_fingerprints(plugin: Any, tmp_path: Any) -> None:
    if has_capability(plugin, "upstream_only"):
        spec = layout_cases(plugin).spec()
        frags = plugin.fragments(str(tmp_path), spec)
        assert plugin.fingerprint(frags, None) == plugin.fingerprint(frags, None)
        assert plugin.partition_fingerprints(frags, None) == {}
        return
    cases, tree, spec, _fmt = build_tree(plugin, tmp_path)

    def fp() -> tuple[str, dict[str, str]]:
        frags = plugin.fragments(tree.root, spec)
        return plugin.fingerprint(frags, None), plugin.partition_fingerprints(frags, None)

    fp0, parts0 = fp()
    assert fp0.startswith("fp1:") and fp() == (fp0, parts0), "stable"
    first = os.path.join(tree.root, tree.data_files[0])
    _flip_byte(first)
    fp1, parts1 = fp()
    assert fp1 != fp0, "a byte change changes the fingerprint"
    if parts0:
        label = "/".join(f"{k}={'__HIVE_DEFAULT_PARTITION__' if v is None else v}"
                         for k, v in tree.partitions[tree.data_files[0]].items())
        changed = {k for k in parts0 if parts0[k] != parts1.get(k)}
        assert changed == {label}, "only the touched partition changes"
    base, ext = os.path.splitext(first)
    added = f"{base}-added{ext}"
    shutil.copyfile(first, added)
    fp2, _ = fp()
    assert fp2 != fp1, "adding a file changes the fingerprint"
    os.remove(added)
    assert fp()[0] == fp1, "removing it restores the fingerprint"
    renamed = f"{base}-renamed{ext}"
    os.rename(first, renamed)
    assert fp()[0] != fp1, "renaming a file changes the fingerprint"
    os.rename(renamed, first)
    frags = plugin.fragments(tree.root, spec)
    manifest = Manifest(path=None, entries={os.path.relpath(f.uri, tree.root): {"sha256": f"{i:064x}",
                                                                                "bytes": f.size}
                                            for i, f in enumerate(frags)})
    assert plugin.fingerprint(frags, manifest).startswith("fp1:manifest:")


@pytest.mark.parametrize("plugin", _params())
def test_l5_signature_is_stat_only(plugin: Any, tmp_path: Any, monkeypatch: Any) -> None:
    if has_capability(plugin, "upstream_only"):
        spec = layout_cases(plugin).spec()
        tree_root = str(tmp_path)
    else:
        _cases, tree, spec, _fmt = build_tree(plugin, tmp_path)
        tree_root = tree.root

    def refuse(*a: Any, **k: Any) -> Any:
        raise AssertionError(f"signature opened a file: {a[:1]}")

    with monkeypatch.context() as m:
        m.setattr(builtins, "open", refuse)
        m.setattr(io, "open", refuse)
        m.setattr(os, "open", refuse)
        sig0 = plugin.signature(tree_root, spec)
        assert sig0 == plugin.signature(tree_root, spec)
    if has_capability(plugin, "upstream_only"):
        return
    first = os.path.join(tree.root, tree.data_files[0])
    base, ext = os.path.splitext(first)
    added = f"{base}-added{ext}"
    shutil.copyfile(first, added)
    sig1 = plugin.signature(tree_root, spec)
    assert sig1 != sig0, "adding a file changes the signature"
    with open(added, "ab") as fh:
        fh.write(b"\0" * 7)
    sig2 = plugin.signature(tree_root, spec)
    assert sig2 != sig1, "resizing a file changes the signature"
    os.remove(added)
    assert plugin.signature(tree_root, spec) == sig0, "removing it restores the signature"


@pytest.mark.parametrize("plugin", _params(caps=("live", "upstream_only")))
def test_l6_as_of(plugin: Any, tmp_path: Any) -> None:
    as_of = plugin.as_of(str(tmp_path), layout_cases(plugin).spec())
    assert isinstance(as_of, str) and as_of


# ---------------------------------------------------------------------------
# L-7 a truncated shard is an error, never fewer rows
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("plugin", _params(files=True))
def test_l7_truncated_shard(plugin: Any, tmp_path: Any) -> None:
    _cases, tree, spec, fmt_plugin = build_tree(plugin, tmp_path)
    frags = plugin.fragments(tree.root, spec)
    partitions = plugin.partition_columns(spec)
    assert len(scan_all(fmt_plugin, frags, partitions)) == tree.table.num_rows
    victim = frags[-1]
    truncate(victim.uri)
    after = plugin.fragments(tree.root, spec)
    assert [f.uri for f in after] == [f.uri for f in frags], "a damaged file is still listed"
    with pytest.raises(FormatError) as err:
        scan_all(fmt_plugin, after, partitions)
    assert err.value.fragment == victim.uri
    assert dict(err.value.partition) == dict(victim.partition)
