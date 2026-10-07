"""Golden tables, golden trees and the case types of the conformance suites (§9.3).

pyarrow is imported inside the builders only, so plugins can name these case types from
``conformance_cases()`` without importing it.

* :data:`GOLDEN` lists the golden tables, each tagged with the capabilities a format needs to
  hold it (``tabular`` or ``nested``) and the §6.4 paths whose ``path_in_schema`` the F-11
  case checks: unicode string IDs, int32/int64, float32/float64 with NaN and null, bool with
  null, timestamp, ``list<string>``, ``large_list``/``large_string``, ``list<struct<drugId,
  x>>``, the three-level ``target_essentiality`` shape, a null struct parent versus a null child,
  a null list versus ``[]`` versus a list of null items, ``struct<rows, count>``, a struct of lists
  (``disease.synonyms``) and ``list<list<string>>``; :func:`empty_variant` and
  :func:`build_big` give the 0-row and 1M-row variants. :data:`LEGACY_LEAF_CASES` check the
  legacy list encodings against hand-written footers.
* :func:`write_tree` writes a golden table as a ``single`` (one file per cohort), ``sharded``
  or ``hive`` tree with junk next to the data (``*.part``, ``*.part.json``, ``_SUCCESS``,
  dotfiles, ``_temporary``); :func:`truncate` makes a truncated shard.
* :func:`random_predicates` draws seeded predicates over the ``scalars`` golden (F-3).
* :class:`FormatCases`, :class:`LayoutCases` and :class:`StatisticCase` are what a plugin's
  ``conformance_cases()`` returns.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Mapping, Sequence

from ...predicate import (
    All,
    And,
    Any as AnyItem,
    Cmp,
    CmpAbs,
    Contains,
    Eq,
    In,
    IsNull,
    NonEmpty,
    Not,
    Or,
    Predicate,
    Range,
    TextMatch,
)
from ..base import LayoutSpec

__all__ = [
    "FormatCases", "LayoutCases", "GoldenTree", "GoldenCase", "LegacyLeafCase", "StatisticCase", "GOLDEN",
    "LEGACY_LEAF_CASES", "golden", "empty_variant", "build_big", "write_parquet", "parquet_with_paths",
    "write_tree", "truncate", "corrupt", "zero_length", "random_predicates", "nested_predicates", "CORRELATED_ANY",
    "F9_LITERALS", "HIVE_FILES",
]

NAN = float("nan")


# ---------------------------------------------------------------------------
# Case types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FormatCases:
    """How the format suite writes goldens for a format plugin."""

    extension: str                                     # ".parquet"
    write: Callable[..., None]                         # write(pa_table, path, row_group_size=None)
    #: (arrow schema, footer leaf paths) -> a schema ``leaf_path`` maps through (legacy encodings, F-11)
    with_physical_paths: Callable[[Any, Sequence[str]], Any] | None = None
    #: golden case name -> reason a format cannot hold it (beyond capability tags)
    skip: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class GoldenTree:
    root: str
    location: str                                      # the table's path under root (LayoutSpec.path)
    data_files: tuple[str, ...]                        # relpaths under root, sorted
    partitions: Mapping[str, Mapping[str, Any]]        # relpath -> partition values
    junk: tuple[str, ...]
    table: Any                                         # every row written, partition columns included


@dataclass(frozen=True)
class LayoutCases:
    """How the layout suite exercises a layout plugin."""

    tree: str                                          # "single" | "sharded" | "hive" | "none"
    path: str | None = None                            # LayoutSpec.path under the tree root
    options: Mapping[str, Any] = field(default_factory=dict)
    partitions: Mapping[str, str] = field(default_factory=dict)
    fragment_key: Mapping[str, Any] | None = None
    format: str = "parquet"
    #: a custom tree writer (root, FormatCases) -> GoldenTree for layouts the standard trees do not fit
    build: Callable[[str, FormatCases], GoldenTree] | None = None

    def spec(self, table: str = "golden.t", **changes: Any) -> LayoutSpec:
        data = {"table": table, "path": self.path, "options": dict(self.options), "partitions": dict(self.partitions),
                "fragment_key": dict(self.fragment_key) if self.fragment_key else None, "format": self.format}
        data.update(changes)
        return LayoutSpec(**data)


@dataclass(frozen=True)
class GoldenCase:
    name: str
    build: Callable[[], Any]
    requires: tuple[str, ...] = ("tabular",)
    key: tuple[str, ...] = ("id",)
    #: §6.4 path -> path_in_schema as pyarrow writes it (F-11)
    leaf_paths: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LegacyLeafCase:
    name: str
    schema: Callable[[], Any]
    physical: tuple[str, ...]
    expected: Mapping[str, str]
    requires: tuple[str, ...] = ("nested", "leaf_projection")


@dataclass(frozen=True)
class StatisticCase:
    """One statistic spec with values that exercise S-1..S-6 and S-8.

    ``thresholds`` are in-scale ``(op, value)`` pairs (boundary checks use the ``lt/le/gt/ge``
    ones); ``refused`` must raise ``UnsupportedFilter`` (outside the scale, an undeclared value, a
    negated form); ``unconfirmed`` must raise when the spec is ``verified: false`` and no
    confirmed facts are given; ``group`` is ``(column, a, b)`` for the comparable_within checks.
    """

    name: str
    spec: Mapping[str, Any]
    column: str = "value"
    values: tuple[Any, ...] = ()
    best: Any = None
    unknown: tuple[Any, ...] = (None, NAN)
    unknown_rows: tuple[Mapping[str, Any], ...] = ()
    thresholds: tuple[tuple[str, Any], ...] = ()
    refused: tuple[tuple[str, Any], ...] = ()
    unconfirmed: tuple[tuple[str, Any], ...] = ()
    confirmed: Mapping[str, Any] | None = None
    group: tuple[str, Any, Any] | None = None
    fixed_scope: Mapping[str, Any] = field(default_factory=dict)
    agg: tuple[str, ...] = ("max",)
    row: Callable[[Any], dict[str, Any]] | None = None
    requires: tuple[str, ...] = ()

    def make_row(self, value: Any, **extra: Any) -> dict[str, Any]:
        row = dict(self.row(value)) if self.row is not None else {self.column: value}
        row.update(extra)
        return row

    def column_spec(self, **override: Any) -> Any:
        from ...descriptor.columns import validate_column

        data = dict(self.spec)
        data.update(override)
        return validate_column(data)


# ---------------------------------------------------------------------------
# Golden tables
# ---------------------------------------------------------------------------

def _pa() -> Any:
    import pyarrow

    return pyarrow


#: String literals F-9 must match exactly (quotes, parentheses, backslash, unicode, SQL-ish text).
F9_LITERALS = ("it's", 'say "hi"', "a(b)", "back\\slash", "日本語", "x'); DROP", "Straße")


def build_scalars() -> Any:
    pa = _pa()
    return pa.table({
        "id": pa.array(["ENSG00000169174", "Straße", "it's", 'say "hi"', "a(b)", "back\\slash", "日本語", "ẞ-capital",
                        "x'); DROP", "zz"], pa.string()),
        "i32": pa.array([1, -2, None, 4, 4, 0, 7, 2147483647, -2147483648, 3], pa.int32()),
        "i64": pa.array([10, None, 30, 40, 40, -50, 2 ** 40, 0, 5, 6], pa.int64()),
        "f32": pa.array([0.05, NAN, None, 0.5, 5.0, 0.05, -1.5, 0.0, 1e-8, 3.25], pa.float32()),
        "f64": pa.array([0.1, None, NAN, -0.5, 2.5, 2.5, 1e300, -0.0, 0.3, 7.0], pa.float64()),
        "flag": pa.array([True, False, None, True, False, None, True, True, False, False], pa.bool_()),
        "ts": pa.array([datetime(2020, 1, 1), None, datetime(2021, 6, 30, 12), datetime(2019, 2, 3),
                        datetime(2020, 1, 1), datetime(2024, 12, 31, 23, 59), None, datetime(2001, 9, 9),
                        datetime(2020, 5, 5), datetime(2022, 2, 2)], pa.timestamp("us")),
        "concentration": pa.array([0.05, 0.5, 5.0, 0.05, 0.05, 0.5, 5.0, None, NAN, 0.05], pa.float32()),
        "score": pa.array([0.9, 0.9, None, 0.5, NAN, 0.9, 0.1, 0.5, None, 1.0], pa.float64()),
        "name": pa.array(["BRCA1", "brca1 gene", "Straße", "STRASSE", None, "the BRCA1 locus", "ſtar", "Kelvin K",
                          "", "x"], pa.string()),
        "grp": pa.array(["a", "b", "a", None, "b", "a", "c", "c", "a", "b"], pa.string()),
    })


def build_lists() -> Any:
    pa = _pa()
    return pa.table({
        "id": [f"l{i}" for i in range(1, 7)],
        "tags": pa.array([["a", "b"], [], None, [None], ["b", None], ["c"]], pa.list_(pa.string())),
        "ints": pa.array([[1, 2], [], None, [None], [3, None], [5]], pa.list_(pa.int64())),
    })


def build_large() -> Any:
    pa = _pa()
    return pa.table({
        "id": pa.array([f"L{i}" for i in range(1, 6)], pa.large_string()),
        "ltags": pa.array([["x"], None, [], [None], ["y", "x"]], pa.large_list(pa.large_string())),
        "lname": pa.array(["Alpha", None, "beta", "", "GAMMA"], pa.large_string()),
    })


def build_drugs() -> Any:
    pa = _pa()
    drug = pa.struct([("drugId", pa.string()), ("x", pa.float64())])
    return pa.table({
        "id": [f"p{i}" for i in range(1, 7)],
        "drugs": pa.array([[{"drugId": "CHEMBL1", "x": 1.0}, {"drugId": "CHEMBL2", "x": NAN}], [], None, [None],
                           [{"drugId": None, "x": 2.0}], [{"drugId": "CHEMBL1", "x": None}, {"drugId": "CHEMBL3",
                                                                                            "x": 0.5}]],
                          pa.list_(drug)),
    })


def _screen(name: str | None, effect: float | None, expression: float | None) -> dict[str, Any]:
    return {"cellLineName": name, "geneEffect": effect, "expression": expression}


def build_essentiality() -> Any:
    """The ``target_essentiality`` shape: gene -> depMapEssentiality (tissue) -> screens."""
    pa = _pa()
    screen = pa.struct([("cellLineName", pa.string()), ("geneEffect", pa.float64()), ("expression", pa.float64())])
    tissue = pa.struct([("tissueName", pa.string()), ("screens", pa.list_(screen))])
    gene = pa.struct([("isEssential", pa.bool_()), ("depMapEssentiality", pa.list_(tissue))])
    rows = [
        # liver has a strong screen and a highly expressed one, but no single screen is both
        [{"isEssential": True, "depMapEssentiality": [
            {"tissueName": "liver", "screens": [_screen("A", -1.5, 3.0), _screen("B", -0.2, 12.0)]},
            {"tissueName": "lung", "screens": [_screen("C", -2.0, 15.0)]}]}],
        [{"isEssential": False, "depMapEssentiality": [
            {"tissueName": "liver", "screens": [_screen("D", -1.2, 11.0)]}]}],
        None,
        [],
        [{"isEssential": None, "depMapEssentiality": None}],
        [{"isEssential": True, "depMapEssentiality": [
            {"tissueName": "liver", "screens": []},
            {"tissueName": None, "screens": [_screen("E", -3.0, 20.0)]}]}],
        [{"isEssential": True, "depMapEssentiality": [
            {"tissueName": "liver", "screens": [None, _screen("F", NAN, 30.0)]}]}],
    ]
    return pa.table({"id": [f"g{i}" for i in range(1, len(rows) + 1)],
                     "geneEssentiality": pa.array(rows, pa.list_(gene))})


def build_struct_nulls() -> Any:
    pa = _pa()
    s = pa.struct([("a", pa.string()), ("b", pa.int64())])
    return pa.table({"id": ["s1", "s2", "s3", "s4"],
                     "s": pa.array([{"a": "x", "b": 1}, None, {"a": None, "b": 2}, {"a": "y", "b": None}], s)})


def build_linked() -> Any:
    pa = _pa()
    t = pa.struct([("rows", pa.list_(pa.string())), ("count", pa.int64())])
    return pa.table({"id": [f"d{i}" for i in range(1, 6)],
                     "linkedTargets": pa.array([{"rows": ["T1", "T2"], "count": 2}, {"rows": [], "count": 0}, None,
                                                {"rows": None, "count": None}, {"rows": ["T3"], "count": 1}], t)})


def build_synonyms() -> Any:
    pa = _pa()
    t = pa.struct([("hasExactSynonym", pa.list_(pa.string())), ("hasBroadSynonym", pa.list_(pa.string()))])
    return pa.table({"id": ["MONDO_1", "MONDO_2", "MONDO_3", "MONDO_4"],
                     "synonyms": pa.array([{"hasExactSynonym": ["T2D", "NIDDM"], "hasBroadSynonym": ["diabetes"]},
                                           {"hasExactSynonym": [], "hasBroadSynonym": None}, None,
                                           {"hasExactSynonym": None, "hasBroadSynonym": ["x"]}], t)})


def build_list_list() -> Any:
    pa = _pa()
    return pa.table({"id": [f"r{i}" for i in range(1, 7)],
                     "path": pa.array([[["a", "b"], ["c"]], [[]], [], None, [None], [["a", None]]],
                                      pa.list_(pa.list_(pa.string())))})


_ESS = "geneEssentiality[].depMapEssentiality[]"
_ESS_P = "geneEssentiality.list.element.depMapEssentiality.list.element"

GOLDEN: tuple[GoldenCase, ...] = (
    GoldenCase("scalars", build_scalars, ("tabular",),
               leaf_paths={"id": "id", "f32": "f32", "concentration": "concentration", "ts": "ts"}),
    GoldenCase("lists", build_lists, ("nested",),
               leaf_paths={"tags[]": "tags.list.element", "ints[]": "ints.list.element", "tags": "tags.list.element"}),
    GoldenCase("large", build_large, ("nested",),
               leaf_paths={"ltags[]": "ltags.list.element", "lname": "lname", "id": "id"}),
    GoldenCase("drugs", build_drugs, ("nested",),
               leaf_paths={"drugs[].drugId": "drugs.list.element.drugId", "drugs[].x": "drugs.list.element.x",
                           "drugs[drugId=CHEMBL1].x": "drugs.list.element.x", "drugs": "drugs",
                           "drugs[]": "drugs.list.element"}),
    GoldenCase("essentiality", build_essentiality, ("nested",),
               leaf_paths={f"{_ESS}.screens[].geneEffect": f"{_ESS_P}.screens.list.element.geneEffect",
                           f"{_ESS}.tissueName": f"{_ESS_P}.tissueName",
                           "geneEssentiality[].isEssential": "geneEssentiality.list.element.isEssential",
                           f"{_ESS}.screens": f"{_ESS_P}.screens"}),
    GoldenCase("struct_nulls", build_struct_nulls, ("nested",), leaf_paths={"s.a": "s.a", "s.b": "s.b", "s": "s"}),
    GoldenCase("linked", build_linked, ("nested",),
               leaf_paths={"linkedTargets.rows[]": "linkedTargets.rows.list.element",
                           "linkedTargets.count": "linkedTargets.count"}),
    GoldenCase("synonyms", build_synonyms, ("nested",),
               leaf_paths={"synonyms.hasExactSynonym[]": "synonyms.hasExactSynonym.list.element",
                           "synonyms.hasBroadSynonym[]": "synonyms.hasBroadSynonym.list.element"}),
    GoldenCase("list_list", build_list_list, ("nested",),
               leaf_paths={"path[][]": "path.list.element.list.element"}),
)


def golden(name: str) -> GoldenCase:
    for case in GOLDEN:
        if case.name == name:
            return case
    raise KeyError(f"no golden case {name!r}")


def empty_variant(table: Any) -> Any:
    """The 0-row variant of a golden table (same schema)."""
    return table.slice(0, 0)


def build_big(n: int = 1_000_000, seed: int = 11) -> Any:
    """``n`` rows: ``id`` (unique), ``score`` (float64 with ~1% null and ~1% NaN, many ties), ``grp`` (int32)."""
    pa = _pa()
    rng = random.Random(seed)
    scores: list[float | None] = []
    for _ in range(n):
        r = rng.random()
        scores.append(None if r < 0.01 else NAN if r < 0.02 else round(rng.random() * 1000) / 10)
    return pa.table({"id": pa.array([f"r{i:07d}" for i in range(n)], pa.string()),
                     "score": pa.array(scores, pa.float64()),
                     "grp": pa.array([i % 97 for i in range(n)], pa.int32())})


def _legacy_drugs() -> Any:
    pa = _pa()
    return pa.schema([("id", pa.string()),
                      ("drugs", pa.list_(pa.struct([("drugId", pa.string()), ("x", pa.float64())])))])


def _legacy_tags() -> Any:
    pa = _pa()
    return pa.schema([("id", pa.string()), ("tags", pa.list_(pa.string()))])


def _legacy_path() -> Any:
    pa = _pa()
    return pa.schema([("path", pa.list_(pa.list_(pa.string())))])


LEGACY_LEAF_CASES: tuple[LegacyLeafCase, ...] = (
    LegacyLeafCase("two_level_array", _legacy_drugs, ("id", "drugs.array.drugId", "drugs.array.x"),
                   {"drugs[].drugId": "drugs.array.drugId", "drugs[].x": "drugs.array.x", "drugs": "drugs"}),
    LegacyLeafCase("bag_array_element", _legacy_drugs,
                   ("id", "drugs.bag.array_element.drugId", "drugs.bag.array_element.x"),
                   {"drugs[].drugId": "drugs.bag.array_element.drugId", "drugs[]": "drugs.bag.array_element"}),
    LegacyLeafCase("repeated_element", _legacy_tags, ("id", "tags.element"), {"tags[]": "tags.element",
                                                                              "tags": "tags.element"}),
    LegacyLeafCase("repeated_array", _legacy_tags, ("id", "tags.array"), {"tags[]": "tags.array"}),
    LegacyLeafCase("list_item", _legacy_tags, ("id", "tags.list.item"), {"tags[]": "tags.list.item"}),
    LegacyLeafCase("nested_two_level", _legacy_path, ("path.array.array",), {"path[][]": "path.array.array"}),
)


# ---------------------------------------------------------------------------
# Writers and trees
# ---------------------------------------------------------------------------

def write_parquet(table: Any, path: str, row_group_size: int | None = None) -> None:
    import pyarrow.parquet as pq

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pq.write_table(table, path, row_group_size=row_group_size)


def parquet_with_paths(schema: Any, physical: Sequence[str]) -> Any:
    """A schema carrying footer leaf paths the way :meth:`ParquetFormat.logical_schema` records them."""
    from ..formats.parquet import PATHS_KEY

    meta = dict(schema.metadata or {})
    meta[PATHS_KEY] = json.dumps(list(physical)).encode()
    return schema.with_metadata(meta)


def _touch(path: str, data: bytes = b"") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def _take(table: Any, indexes: Sequence[int]) -> Any:
    pa = _pa()
    return table.take(pa.array(list(indexes), pa.int64()))


#: hive tree partitions: (sourceId, year) per file, and the scalars rows each file holds
HIVE_FILES: tuple[tuple[str | None, int, int, tuple[int, ...]], ...] = (
    ("chembl", 2020, 0, (0, 1)),
    ("chembl", 2021, 0, (2, 3)),
    ("europepmc", 2020, 0, (4, 5)),
    ("europepmc", 2020, 1, (6, 7)),
    (None, 2021, 0, (8, 9)),
)


def write_tree(kind: str, root: str, fmt: FormatCases, table: Any | None = None) -> GoldenTree:
    """Write ``table`` (default: the scalars golden) as a ``single``, ``sharded`` or ``hive`` tree."""
    pa = _pa()
    table = table if table is not None else build_scalars()
    ext = fmt.extension
    n = table.num_rows
    data: list[str] = []
    parts: dict[str, Mapping[str, Any]] = {}
    junk: list[str] = []
    if kind == "single":
        location = "cohorts"
        half = n // 2
        for name, idx in (("GSE1001", range(0, half)), ("GSE2002", range(half, n))):
            rel = f"{location}/{name}{ext}"
            fmt.write(_take(table, list(idx)), os.path.join(root, rel))
            data.append(rel)
        junk = [f"{location}/GSE3003{ext}.part", f"{location}/.GSE9{ext}", f"{location}/notes.txt",
                f"{location}/_SUCCESS"]
        full = table
    elif kind == "sharded":
        location = "target"
        thirds = [range(0, n // 3), range(n // 3, 2 * n // 3), range(2 * n // 3, n)]
        for rel, idx in zip((f"{location}/part-00000{ext}", f"{location}/part-00001{ext}",
                             f"{location}/sub/part-00002{ext}"), thirds):
            fmt.write(_take(table, list(idx)), os.path.join(root, rel))
            data.append(rel)
        junk = [f"{location}/part-00003{ext}.part", f"{location}/part-00003.part.json", f"{location}/_SUCCESS",
                f"{location}/.part-00000{ext}.crc", f"{location}/.hidden/part-00009{ext}",
                f"{location}/_temporary/part-00008{ext}"]
        full = table
    elif kind == "hive":
        location = "evidence"
        pieces = []
        for source, year, k, idx in HIVE_FILES:
            src = "__HIVE_DEFAULT_PARTITION__" if source is None else source
            rel = f"{location}/sourceId={src}/year={year}/part-{k}{ext}"
            idx = tuple(i for i in idx if i < n)
            fmt.write(_take(table, idx), os.path.join(root, rel))
            data.append(rel)
            parts[rel] = {"sourceId": source, "year": year}
            piece = _take(table, idx)
            piece = piece.append_column("sourceId", pa.array([source] * len(idx), pa.string()))
            piece = piece.append_column("year", pa.array([year] * len(idx), pa.int64()))
            pieces.append(piece)
        junk = [f"{location}/sourceId=europepmc/year=2020/part-2{ext}.part", f"{location}/_SUCCESS"]
        full = pa.concat_tables(pieces)
    else:
        raise ValueError(f"unknown golden tree {kind!r} (single, sharded, hive)")
    for rel in junk:
        _touch(os.path.join(root, rel), b"not data")
    return GoldenTree(root=root, location=location, data_files=tuple(sorted(data)), partitions=parts,
                      junk=tuple(junk), table=full)


def truncate(path: str, keep: float = 0.5) -> None:
    """Cut a file to ``keep`` of its bytes (a shard whose download stopped)."""
    size = os.path.getsize(path)
    with open(path, "rb+") as fh:
        fh.truncate(max(1, int(size * keep)))


def corrupt(path: str) -> None:
    """Overwrite a file with bytes that start and end like Parquet but hold no valid footer."""
    with open(path, "wb") as fh:
        fh.write(b"PAR1" + b"\x00garbage\x01" * 64 + b"\x10\x00\x00\x00PAR1")


def zero_length(path: str) -> None:
    _touch(path, b"")


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

_NUMERIC = ("i32", "i64", "f32", "f64", "concentration", "score")
_STRING = ("id", "name", "grp")
_TEXT_NEEDLES = ("brca1", "BRCA1", "ss", "Straße", "star", "k", "", "the", "x", "日本")


def _value(rng: random.Random, column: str, rows: Sequence[Mapping[str, Any]]) -> Any:
    seen = [r[column] for r in rows if r.get(column) is not None]
    if column in _NUMERIC and rng.random() < 0.3:
        return rng.choice([0, 1, -1, 0.05, 0.5, 2.5, 4, 1e-8, 2 ** 40, 3.25, 7, 10, 2147483647])
    if column in _STRING and rng.random() < 0.2:
        return rng.choice(["nope", "a", "B", "ZZ"])
    return rng.choice(seen) if seen else 0


def _leaf(rng: random.Random, rows: Sequence[Mapping[str, Any]]) -> Predicate:
    kind = rng.choice(["eq", "in", "cmp", "cmp", "abs", "range", "null", "text", "flag"])
    if kind == "flag":
        return Eq("flag", rng.choice([True, False])) if rng.random() < 0.7 else IsNull("flag")
    if kind == "text":
        return TextMatch(rng.choice(["name", "id"]), rng.choice(_TEXT_NEEDLES),
                         rng.choice(["exact", "casefold", "substring", "casefold_substring", "word"]))
    if kind == "null":
        return IsNull(rng.choice(_NUMERIC + _STRING + ("flag", "ts")))
    col = rng.choice(_NUMERIC + (_STRING if kind in ("eq", "in", "cmp") else ()))
    if kind == "eq":
        return Eq(col, _value(rng, col, rows))
    if kind == "in":
        vals = [_value(rng, col, rows) for _ in range(rng.randint(1, 3))]
        if rng.random() < 0.2:
            vals.append(None)
        return In(col, tuple(vals))
    if kind == "cmp":
        return Cmp(col, rng.choice(["<", "<=", ">", ">=", "!="]), _value(rng, col, rows))
    if kind == "abs":
        col = rng.choice(_NUMERIC)
        return CmpAbs(col, rng.choice(["<", "<=", ">", ">=", "!="]), abs(_value(rng, col, rows) or 0))
    lo, hi = sorted([_value(rng, col, rows), _value(rng, col, rows)], key=lambda v: (str(type(v)), v))
    return Range(col, lo, hi, rng.random() < 0.7, rng.random() < 0.7)


def _tree(rng: random.Random, rows: Sequence[Mapping[str, Any]], depth: int) -> Predicate:
    r = rng.random()
    if depth <= 0 or r < 0.45:
        return _leaf(rng, rows)
    if r < 0.6:
        return Not(_tree(rng, rows, depth - 1))
    parts = tuple(_tree(rng, rows, depth - 1) for _ in range(rng.randint(2, 3)))
    return And(parts) if r < 0.8 else Or(parts)


def random_predicates(rows: Sequence[Mapping[str, Any]], n: int = 50, seed: int = 7) -> list[Predicate]:
    """``n`` seeded random predicates over the ``scalars`` golden (nulls, NaN, CmpAbs, text, Not/And/Or)."""
    rng = random.Random(seed)
    return [_tree(rng, rows, 2) for _ in range(n)]


#: F-4/F-3: a correlated Any over the three-level golden: one screen in liver both strong and expressed.
CORRELATED_ANY = AnyItem("geneEssentiality[]", AnyItem(
    "depMapEssentiality[]", And((Eq("tissueName", "liver"),
                                 AnyItem("screens[]", And((Cmp("geneEffect", "<=", -1.0),
                                                           Cmp("expression", ">=", 10.0))))))))


def nested_predicates() -> dict[str, list[Predicate]]:
    """Membership predicates per nested golden (F-4): Contains/Any/All on one, two and three levels."""
    ess = "geneEssentiality[]"
    return {
        "lists": [Contains("tags", "b"), Contains("tags[]", "a"), AnyItem("ints[]", Cmp("[]", ">", 1)),
                  All("ints[]", Cmp("[]", ">", 0)), NonEmpty("tags"), Not(Contains("tags", "a")),
                  All("tags[]", Eq("[]", "c")), Eq("tags", "b"), In("ints", (2, 5))],
        "large": [Contains("ltags", "x"), All("ltags[]", Eq("[]", "x")), NonEmpty("ltags")],
        "drugs": [AnyItem("drugs[]", Eq("drugId", "CHEMBL1")), Contains("drugs[].drugId", "CHEMBL3"),
                  AnyItem("drugs[]", And((Eq("drugId", "CHEMBL1"), Cmp("x", ">", 0.5)))),
                  All("drugs[]", Cmp("x", ">=", 0)), AnyItem("drugs[]", IsNull("x"), skip_null_items=True),
                  Not(AnyItem("drugs[]", Eq("drugId", "CHEMBL2")))],
        "essentiality": [CORRELATED_ANY,
                         AnyItem(ess, Eq("isEssential", True)),
                         AnyItem(f"{ess}.depMapEssentiality[]", Eq("tissueName", "liver")),
                         All(f"{ess}.depMapEssentiality[].screens[]", Cmp("geneEffect", "<", 0)),
                         Contains(f"{ess}.depMapEssentiality[].tissueName", "lung"),
                         Cmp(f"{ess}.depMapEssentiality[].screens[].expression", ">", 25.0)],
        "struct_nulls": [Eq("s.a", "x"), IsNull("s.a"), Not(IsNull("s.b")), Cmp("s.b", ">=", 2), IsNull("s")],
        "linked": [Contains("linkedTargets.rows", "T1"), Cmp("linkedTargets.count", ">", 0),
                   NonEmpty("linkedTargets.rows"), IsNull("linkedTargets.count")],
        "synonyms": [Contains("synonyms.hasExactSynonym", "NIDDM"), NonEmpty("synonyms.hasBroadSynonym"),
                     AnyItem("synonyms.hasExactSynonym[]", TextMatch("[]", "t2d", "casefold"))],
        "list_list": [Contains("path[][]", "c"), AnyItem("path[]", Contains("[]", "a")),
                      AnyItem("path[]", NonEmpty("[]")), All("path[][]", Eq("[]", "a"))],
    }
