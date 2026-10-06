"""Format conformance suite (§9.3, F-1..F-11), parametrized over the registered format plugins.

A plugin runs only the golden cases its capabilities cover (``tabular`` goldens for every
format, ``nested`` goldens, ``pushdown`` for F-3/F-9/F-10, ``stats`` for F-5,
``leaf_projection`` for F-2's leaf reads and F-11). Goldens are written with the plugin's own
``conformance_cases().write``. The oracle is :func:`~vbt.datalayer.predicate.evaluate` on the
golden rows with storage-typed literals; a scan's result is the pushdown rows with the plugin's
residual applied, so a pushdown that returns one row too many or too few fails.

F-6 runs a 100k-row variant by default and the 1M-row variant when ``VBT_DL_SLOW=1`` (marked
``slow``); peak allocation is measured on a dedicated Arrow proxy pool when the plugin's ``scan``
accepts ``memory_pool`` (else by sampling the default pool), and ``batch_bytes`` counts a batch
both decoded and encoded (the footer's uncompressed bytes per row), since a decode holds both. External format packages import this module from a test file to run the same suite.
"""

from __future__ import annotations

import gc
import heapq
import inspect
import math
import os
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Sequence

import pytest

pa = pytest.importorskip("pyarrow")

from ...predicate import And, Cmp, Eq, In, Predicate, RankKey, TextMatch, evaluate  # noqa: E402
from ...rowkey import canonical, render_float  # noqa: E402
from ..base import FormatError, Fragment  # noqa: E402
from ..formats import nan_to_none, storage_typed  # noqa: E402
from . import applicable, has_capability, selected_plugins  # noqa: E402
from .golden import (  # noqa: E402
    CORRELATED_ANY,
    F9_LITERALS,
    GOLDEN,
    LEGACY_LEAF_CASES,
    FormatCases,
    GoldenCase,
    build_big,
    corrupt,
    empty_variant,
    golden,
    nested_predicates,
    random_predicates,
    truncate,
    zero_length,
)

FORMAT_PLUGINS = selected_plugins("format")
_POOLS: list[Any] = []
SLOW = os.environ.get("VBT_DL_SLOW") == "1"


def format_cases(plugin: Any) -> FormatCases:
    cases = plugin.conformance_cases()
    if not isinstance(cases, FormatCases):
        pytest.skip(f"{plugin.name} provides no FormatCases (it cannot write goldens)")
    return cases


def _golden_params() -> list[Any]:
    out = []
    for plugin in FORMAT_PLUGINS:
        for case in GOLDEN:
            if applicable(plugin, case.requires):
                out.append(pytest.param(plugin, case, id=f"{plugin.name}-{case.name}"))
    return out


def _plugin_params(*caps: str) -> list[Any]:
    return [pytest.param(p, id=p.name) for p in FORMAT_PLUGINS if applicable(p, caps)]


GOLDEN_PARAMS = _golden_params()


# ---------------------------------------------------------------------------
# Helpers (public: format tests and other suites use them)
# ---------------------------------------------------------------------------

def write_golden(plugin: Any, case: GoldenCase | Any, directory: Any, *, table: Any = None, name: str | None = None,
                 row_group_size: int | None = None) -> Fragment:
    """Write a golden with the plugin's writer; returns its fragment."""
    fmt = format_cases(plugin)
    if isinstance(case, GoldenCase) and case.name in fmt.skip:
        pytest.skip(f"{plugin.name}: {fmt.skip[case.name]}")
    table = table if table is not None else case.build()
    path = os.path.join(str(directory), (name or getattr(case, "name", "golden")) + fmt.extension)
    if row_group_size is None:
        fmt.write(table, path)
    else:
        fmt.write(table, path, row_group_size=row_group_size)
    st = os.stat(path)
    return Fragment(uri=path, size=st.st_size, mtime_ns=st.st_mtime_ns)


def scan_schema(plugin: Any, frags: Sequence[Fragment], partitions: Any = None) -> Any:
    fn = getattr(plugin, "scan_schema", None)
    if callable(fn):
        return fn(frags, partitions or {})
    return plugin.logical_schema(frags[0])


def type_of(plugin: Any, schema: Any) -> Any:
    fn = getattr(plugin, "storage_type_of", None)
    if callable(fn):
        return fn(schema)
    from ..formats.parquet import type_at

    def generic(path: str) -> str | None:
        t = type_at(schema, path)
        return None if t is None else str(t)

    return generic


def scan_rows(plugin: Any, frags: Sequence[Fragment], *, columns: list[str] | None = None,
              predicate: Predicate | None = None, partitions: Any = None, batch_rows: int = 3,
              row_groups: Any = None) -> list[dict[str, Any]]:
    """Rows of a scan: the pushdown's batches, natively converted, with the residual applied."""
    rows: list[dict[str, Any]] = []
    for batch in plugin.scan(list(frags), columns=columns, predicate=predicate, partitions=partitions or {},
                             batch_rows=batch_rows, row_groups=row_groups):
        rows.extend(plugin.to_native(batch))
    if predicate is not None and rows:
        from ..layouts import prune_fragments

        kept = prune_fragments(list(frags), predicate) or list(frags)
        _, residual = plugin.compile(predicate, scan_schema(plugin, kept, partitions))
        if residual is not None:
            rows = [r for r in rows if evaluate(residual, r) is True]
    return rows


def oracle_rows(plugin: Any, schema: Any, rows: Sequence[dict[str, Any]], predicate: Predicate | None
                ) -> list[dict[str, Any]]:
    """The rows ``predicate`` selects under ``evaluate`` with storage-typed literals."""
    if predicate is None:
        return list(rows)
    typed = storage_typed(predicate, type_of(plugin, schema))
    return [r for r in rows if evaluate(typed, r) is True]


def golden_rows(table: Any) -> list[dict[str, Any]]:
    """The expected native rows of a golden table (NaN -> None at every level)."""
    return nan_to_none(table.to_pylist())


_NATIVE = (type(None), bool, int, float, str, bytes, datetime, date, Decimal)


def assert_native(value: Any, where: str = "row") -> None:
    """Only plain Python values: lists, dicts and scalars (any numpy object fails F-1)."""
    if isinstance(value, dict):
        for k, v in value.items():
            assert isinstance(k, str), f"{where}: non-string key {k!r}"
            assert_native(v, f"{where}.{k}")
        return
    if isinstance(value, list):
        for i, v in enumerate(value):
            assert_native(v, f"{where}[{i}]")
        return
    assert isinstance(value, _NATIVE), f"{where}: {type(value).__module__}.{type(value).__name__} is not native"
    assert not (isinstance(value, float) and math.isnan(value)), f"{where}: NaN must come back as None"


def project_path(value: Any, tokens: Sequence[str]) -> Any:
    """``value`` keeping only the leaf at ``tokens`` (names and ``[]``), as a leaf read returns it."""
    if not tokens or value is None:
        return value
    head, rest = tokens[0], tokens[1:]
    if head == "[]":
        return [project_path(v, rest) for v in value]
    if not isinstance(value, dict):
        return value
    return {head: project_path(value.get(head), rest)}


def path_tokens(path: str) -> tuple[str, ...]:
    from ...roles import parse_path

    out: list[str] = []
    for seg in parse_path(path).segments:
        if seg.name:
            out.append(seg.name)
        out.extend("[]" for _ in seg.brackets)
    return tuple(out)


def _rank(value: Any, direction: str) -> tuple[int, float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 1, 0.0                                      # nulls last whatever the direction
    v = abs(value) if direction.endswith("_abs") else value
    return 0, (-v if direction.startswith("desc") else v) + 0.0


def scan_top_rows(plugin: Any, frags: Sequence[Fragment], *, order: RankKey | None, k: int,
                  key_columns: Sequence[str], storage_types: Sequence[str | None] | None = None,
                  predicate: Predicate | None = None, batch_rows: int = 1024,
                  columns: list[str] | None = None) -> tuple[list[dict[str, Any]], int]:
    """Bounded global top-k (the ``scan_top_rows`` property): ``(rows, peak Arrow bytes)``. Holds at most
    ``k`` rows plus one batch; ties are broken by the canonical key, nulls last."""
    cols = columns or sorted({*key_columns, *((order.column,) if order else ())})
    gc.collect()
    extra: dict[str, Any] = {}
    pool = None
    if "memory_pool" in inspect.signature(plugin.scan).parameters:
        pool = pa.proxy_memory_pool(pa.default_memory_pool())   # counts this scan's allocations only
        _POOLS.append(pool)                                 # Arrow may free into it later: never drop it
        extra["memory_pool"] = pool
    base = pa.total_allocated_bytes()
    peak = 0
    heap: list[tuple[Any, ...]] = []
    seq = 0
    _, residual = plugin.compile(predicate, scan_schema(plugin, frags)) if predicate is not None else (None, None)
    for batch in plugin.scan(list(frags), columns=cols, predicate=predicate, partitions={}, batch_rows=batch_rows,
                             **extra):
        peak = pool.max_memory() if pool is not None else max(peak, pa.total_allocated_bytes() - base)
        for row in plugin.to_native(batch):
            if residual is not None and evaluate(residual, row) is not True:
                continue
            rank = _rank(row.get(order.column), order.direction) if order else (0, 0.0)
            tie = canonical([row.get(c) for c in key_columns], storage_types)
            seq += 1
            item = (_Neg(rank), _Neg(tie), seq, row)        # max-heap of the k best: evict the worst
            if len(heap) < k:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
        del batch
    if pool is not None:
        peak = pool.max_memory()
    best = sorted(heap, key=lambda it: (it[0].v, it[1].v))
    return [it[3] for it in best], peak


class _Neg:
    """Reverses ordering inside the heap tuple."""

    __slots__ = ("v",)

    def __init__(self, v: Any) -> None:
        self.v = v

    def __lt__(self, other: "_Neg") -> bool:
        return self.v > other.v

    def __gt__(self, other: "_Neg") -> bool:
        return self.v < other.v

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Neg) and self.v == other.v


def sorted_head(rows: Sequence[dict[str, Any]], order: RankKey | None, k: int, key_columns: Sequence[str],
                storage_types: Sequence[str | None] | None = None) -> list[dict[str, Any]]:
    """The full sort-then-head oracle of F-7."""
    def key(r: dict[str, Any]) -> tuple[Any, ...]:
        rank = _rank(r.get(order.column), order.direction) if order else (0, 0.0)
        return rank, canonical([r.get(c) for c in key_columns], storage_types)

    return sorted(rows, key=key)[:k]


# ---------------------------------------------------------------------------
# F-1 round trip, F-2 projection, F-5 stats, F-11 leaf paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("plugin,case", GOLDEN_PARAMS)
def test_f1_round_trip(plugin: Any, case: GoldenCase, tmp_path: Any) -> None:
    table = case.build()
    frag = write_golden(plugin, case, tmp_path)
    rows = scan_rows(plugin, [frag])
    assert rows == golden_rows(table)
    for i, row in enumerate(rows):
        assert_native(row, f"{case.name}[{i}]")
    empty = write_golden(plugin, case, tmp_path, table=empty_variant(table), name=f"{case.name}_empty")
    assert scan_rows(plugin, [empty]) == []


@pytest.mark.parametrize("plugin,case", GOLDEN_PARAMS)
def test_f2_projection(plugin: Any, case: GoldenCase, tmp_path: Any) -> None:
    table = case.build()
    frag = write_golden(plugin, case, tmp_path)
    rows = scan_rows(plugin, [frag], columns=list(case.key))
    assert rows == [{k: r[k] for k in case.key} for r in golden_rows(table)]
    if not has_capability(plugin, "leaf_projection"):
        return
    full = golden_rows(table)
    schema = plugin.logical_schema(frag)
    for path in case.leaf_paths:
        tokens = path_tokens(path)
        if "[]" not in tokens or len(tokens) < 3:
            continue                                        # leaves inside lists, one level at a time
        leaf = plugin.leaf_path(path, schema)
        part = plugin.read_leaves(frag, [leaf], None)
        got = plugin.to_native(part)
        want = [{tokens[0]: project_path(r[tokens[0]], tokens[1:])} for r in full]
        assert got == want, f"{path} -> {leaf}"
        whole = plugin.read_leaves(frag, [tokens[0]], None)
        assert part.nbytes <= whole.nbytes, f"{path}: reading one leaf allocated more than its column"


@pytest.mark.parametrize("plugin,case", [p for p in GOLDEN_PARAMS if has_capability(p.values[0], "stats")])
def test_f5_stats_rows(plugin: Any, case: GoldenCase, tmp_path: Any) -> None:
    frag = write_golden(plugin, case, tmp_path)
    stats = plugin.stats(frag)
    if stats.rows is not None:
        assert stats.rows == len(scan_rows(plugin, [frag]))
    for leaf, cs in stats.columns.items():
        assert cs.uncompressed_bytes >= 0 and cs.storage_type, leaf
        assert cs.kind in ("flat", "string", "nested", "dense_matrix", "sparse_matrix")


@pytest.mark.parametrize("plugin,case", [p for p in GOLDEN_PARAMS if has_capability(p.values[0], "leaf_projection")])
def test_f11_leaf_paths(plugin: Any, case: GoldenCase, tmp_path: Any) -> None:
    frag = write_golden(plugin, case, tmp_path)
    schema = plugin.logical_schema(frag)
    for path, expected in case.leaf_paths.items():
        assert plugin.leaf_path(path, schema) == expected, path


@pytest.mark.parametrize("plugin", _plugin_params("nested", "leaf_projection"))
def test_f11_legacy_list_encodings(plugin: Any) -> None:
    fmt = format_cases(plugin)
    if fmt.with_physical_paths is None:
        pytest.skip(f"{plugin.name} has no legacy footer encodings")
    for case in LEGACY_LEAF_CASES:
        schema = fmt.with_physical_paths(case.schema(), case.physical)
        for path, expected in case.expected.items():
            assert plugin.leaf_path(path, schema) == expected, f"{case.name}: {path}"


# ---------------------------------------------------------------------------
# F-3 pushdown equivalence, F-4 nested membership, F-9 literals, F-10 float32 keys
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("plugin", _plugin_params("pushdown"))
def test_f3_pushdown_equivalence(plugin: Any, tmp_path: Any) -> None:
    case = golden("scalars")
    frag = write_golden(plugin, case, tmp_path)
    schema = scan_schema(plugin, [frag])
    full = scan_rows(plugin, [frag])
    for i, p in enumerate(random_predicates(full, n=50, seed=7)):
        got = [r["id"] for r in scan_rows(plugin, [frag], predicate=p)]
        want = [r["id"] for r in oracle_rows(plugin, schema, full, p)]
        assert got == want, f"predicate {i}: {p}"
    for p in (Cmp("f32", "!=", 0.05), Cmp("f64", "!=", 2.5), In("f64", (0, 7.0)), Eq("flag", False)):
        assert [r["id"] for r in scan_rows(plugin, [frag], predicate=p)] == \
            [r["id"] for r in oracle_rows(plugin, schema, full, p)], f"null/NaN never satisfies: {p}"


@pytest.mark.parametrize("plugin", _plugin_params("pushdown", "nested"))
def test_f3_correlated_any_three_levels(plugin: Any, tmp_path: Any) -> None:
    frag = write_golden(plugin, golden("essentiality"), tmp_path)
    full = scan_rows(plugin, [frag])
    got = [r["id"] for r in scan_rows(plugin, [frag], predicate=CORRELATED_ANY)]
    assert got == [r["id"] for r in full if evaluate(CORRELATED_ANY, r) is True] == ["g2"]


@pytest.mark.parametrize("plugin,case", [p for p in GOLDEN_PARAMS if p.values[1].name in nested_predicates()])
def test_f4_nested_membership(plugin: Any, case: GoldenCase, tmp_path: Any) -> None:
    frag = write_golden(plugin, case, tmp_path)
    schema = scan_schema(plugin, [frag])
    full = scan_rows(plugin, [frag])
    for p in nested_predicates()[case.name]:
        got = [r["id"] for r in scan_rows(plugin, [frag], predicate=p)]
        assert got == [r["id"] for r in oracle_rows(plugin, schema, full, p)], str(p)


@pytest.mark.parametrize("plugin", _plugin_params("pushdown"))
def test_f9_string_literals(plugin: Any, tmp_path: Any) -> None:
    frag = write_golden(plugin, golden("scalars"), tmp_path)
    for lit in F9_LITERALS:
        for p in (Eq("id", lit), In("id", (lit, "no such id")), TextMatch("id", lit, "exact")):
            assert [r["id"] for r in scan_rows(plugin, [frag], predicate=p)] == [lit], str(p)
        assert scan_rows(plugin, [frag], predicate=Eq("id", lit + " ")) == []
    if has_capability(plugin, "string_compile"):
        for lit in F9_LITERALS:
            quoted = plugin.quote(lit)
            assert isinstance(quoted, str) and quoted != lit


@pytest.mark.parametrize("plugin", _plugin_params("pushdown"))
def test_f10_float32_keys(plugin: Any, tmp_path: Any) -> None:
    case = golden("scalars")
    frag = write_golden(plugin, case, tmp_path)
    schema = scan_schema(plugin, [frag])
    full = scan_rows(plugin, [frag])
    storage = type_of(plugin, schema)("concentration")
    assert storage in ("float", "float32"), storage
    for p, n in ((Eq("concentration", 0.05), 4), (In("concentration", (0.05, 5.0)), 6),
                 (And((Eq("concentration", 0.5), Cmp("i32", ">", -5))), 2)):
        got = scan_rows(plugin, [frag], predicate=p)
        assert [r["id"] for r in got] == [r["id"] for r in oracle_rows(plugin, schema, full, p)], str(p)
        assert len(got) == n, str(p)
    for r in scan_rows(plugin, [frag], predicate=Eq("concentration", 0.05)):
        assert render_float(r["concentration"], storage) == "0.05"
        assert canonical([r["concentration"]], [storage]) == "[0.05]"


# ---------------------------------------------------------------------------
# F-6 bounded top-k, F-7 top-k oracle
# ---------------------------------------------------------------------------

def _f6(plugin: Any, tmp_path: Any, n: int, batch_rows: int, k: int = 100) -> None:
    table = build_big(n).select(["id", "score"])
    row_bytes = table.nbytes / table.num_rows
    frag = write_golden(plugin, None, tmp_path, table=table, name=f"big{n}", row_group_size=batch_rows)
    del table
    # a batch is held twice while it decodes: as Arrow arrays and as its encoded pages (footer bytes)
    encoded = row_bytes
    if has_capability(plugin, "stats"):
        stats = plugin.stats(frag)
        encoded = sum(c.uncompressed_bytes for c in stats.columns.values()) / max(1, stats.rows or n)
    batch_bytes = batch_rows * (row_bytes + encoded)
    order = RankKey("score", "desc")
    rows, peak = scan_top_rows(plugin, [frag], order=order, k=k, key_columns=["id"], batch_rows=batch_rows)
    bound = k * row_bytes * 4 + 2 * batch_bytes
    assert peak <= bound, f"peak Arrow allocation {peak} > bound {bound:.0f}"
    assert len(rows) == k
    scores = [r["score"] for r in rows]
    assert all(s is not None for s in scores) and scores == sorted(scores, reverse=True)
    ids = [r["id"] for r in rows]
    ties = {}
    for r in rows:
        ties.setdefault(r["score"], []).append(r["id"])
    assert all(v == sorted(v) for v in ties.values()), "ties broken by the canonical key"
    assert len(set(ids)) == k


@pytest.mark.parametrize("plugin", _plugin_params("tabular"))
def test_f6_bounded_topk(plugin: Any, tmp_path: Any) -> None:
    _f6(plugin, tmp_path, 100_000, 8192)


@pytest.mark.slow
@pytest.mark.skipif(not SLOW, reason="the 1M-row variant runs with VBT_DL_SLOW=1")
@pytest.mark.parametrize("plugin", _plugin_params("tabular"))
def test_f6_bounded_topk_1m(plugin: Any, tmp_path: Any) -> None:
    _f6(plugin, tmp_path, 1_000_000, 65536)


@pytest.mark.parametrize("plugin", _plugin_params("tabular"))
def test_f7_topk_matches_sort(plugin: Any, tmp_path: Any) -> None:
    frag = write_golden(plugin, golden("scalars"), tmp_path)
    full = scan_rows(plugin, [frag])
    for order, keys in ((RankKey("f64", "desc"), ["grp", "id"]), (RankKey("score", "desc"), ["grp", "id"]),
                        (RankKey("f32", "asc"), ["i32", "id"]), (RankKey("f64", "desc_abs"), ["id"]),
                        (None, ["grp", "i32"])):
        for k in (1, 3, 10):
            got, _ = scan_top_rows(plugin, [frag], order=order, k=k, key_columns=keys, batch_rows=3,
                                   columns=list(full[0]))
            assert got == sorted_head(full, order, k, keys), f"{order} k={k}"


@pytest.mark.parametrize("plugin", _plugin_params("nested"))
def test_f7_topk_nested_key_parts(plugin: Any, tmp_path: Any) -> None:
    frag = write_golden(plugin, golden("lists"), tmp_path)
    full = scan_rows(plugin, [frag])
    for k in (1, 2, 6):
        got, _ = scan_top_rows(plugin, [frag], order=None, k=k, key_columns=["tags", "id"], batch_rows=2,
                               columns=["id", "tags", "ints"])
        assert got == sorted_head(full, None, k, ["tags", "id"])


# ---------------------------------------------------------------------------
# F-8 corrupt, truncated and zero-length files
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("plugin", _plugin_params("tabular"))
def test_f8_unreadable_files_raise(plugin: Any, tmp_path: Any) -> None:
    good = write_golden(plugin, golden("scalars"), tmp_path, name="good")
    for name, damage in (("zero", zero_length), ("truncated", truncate), ("corrupt", corrupt)):
        bad = write_golden(plugin, golden("scalars"), tmp_path, name=name)
        damage(bad.uri)
        with pytest.raises(FormatError) as err:
            plugin.logical_schema(bad)
        assert err.value.fragment == bad.uri
        if has_capability(plugin, "stats"):
            with pytest.raises(FormatError):
                plugin.stats(bad)
        with pytest.raises(FormatError):
            scan_rows(plugin, [bad])
        with pytest.raises(FormatError) as err:
            scan_rows(plugin, [good, bad])                  # never fewer rows: the whole scan fails
        assert err.value.fragment == bad.uri

