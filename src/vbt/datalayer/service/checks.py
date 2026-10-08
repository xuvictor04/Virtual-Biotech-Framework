"""Readiness checks R1-R10 (§13) per table, column, container, item table and partition.

Each ``r*`` function takes a :class:`CheckRun` (the reader, the depth and the result being
filled) and records :class:`~vbt.datalayer.ipc.CheckItemModel` findings plus the status of the
part they concern. Statuses (§13), from best to worst::

    ready < unbound < awaiting_producer < stale < partial < key_violation < encoding_drift
          < schema_drift < missing < unreachable < plugin_unavailable

A table's status is the worst status of its error findings; warnings never change it. Which
status a finding implies:

=====  =========================================================  ===============================
R1/R2  layout and manifest probe (``PROBE_STATUS`` of the layout)  missing / partial / schema_drift
R3     an unreadable fragment                                      partial (its partition)
R4     a missing column / an incompatible type                     schema_drift (optional: warning)
R4b    universe keys outside the plugin's canonical form           encoding_drift (failing prefixes named)
R5     nulls in a non-nullable key part                            key_violation
R5b    duplicate keys or item keys                                 key_violation
R6     a refuted ``verified: false`` fact or literal constraint    encoding_drift (``on_refute`` kept)
R7     vocabulary snapshots; mirrored partition columns            warning / schema_drift
R8     a sentinel that is missing, or present when it must not be  partial
R9     dangling references                                         key_violation (``integrity: partial``: warning)
R10    refuted relations, cyclic or non-closed hierarchies         encoding_drift (``on_refute`` kept)
=====  =========================================================  ===============================

Refuted facts are recorded in ``confirmed`` (``{"<column>": {"confirmed": false, ...}}``,
``{"constraint:<column> <op> <value>": {"confirmed": false, "on_refute": "drop_field"}}``) so
the gateway can drop or recompute the fields bound to them.

Depths: ``shallow`` runs R1/R2 only (stat calls, no file opened); ``standard`` runs everything with
bounded samples (``check: full`` keys only up to ``data.readiness.key_check_full_max_rows``);
``deep`` runs full uniqueness and full referential passes.
"""

from __future__ import annotations

import ast
import hashlib
import math
import os
import random
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..descriptor.columns import CompositeRef, is_container
from ..ipc import CheckItemModel, KeyCheckModel, TableCheckModel
from ..plugins.base import ColumnStats, FormatError, Normalized, ValueSnapshot
from ..plugins.layouts import PROBE_STATUS, partition_label
from ..predicate import And, Eq, In, is_null
from ..roles import arrow_compatible, parse_path
from ..rowkey import canonical, content_hash, render_value
from . import ServiceContext, ServiceError, json_path, layout_spec
from . import items as _items
from .reader import BudgetExceeded, TableReader, TableUnavailable, logical_leaves

__all__ = [
    "STATUS_ORDER", "worst", "CheckRun", "check_table", "check_tables", "aggregate_stats", "eval_expr",
    "SAMPLE_ROWS", "SAMPLE_VALUES", "SPILL_KEYS", "sweep_spills",
]

STATUS_ORDER = ("ready", "unbound", "awaiting_producer", "stale", "partial", "key_violation", "encoding_drift",
                "schema_drift", "missing", "unreachable", "plugin_unavailable")
#: Rows a sampled check reads (key prefix blocks, relation samples).
SAMPLE_ROWS = 200_000
#: Distinct values a referential or universe sample checks.
SAMPLE_VALUES = 2_000
_RELATION_SAMPLE = 2_000
_SUSPICIOUS = ("nan", "None", "")


def worst(statuses: Iterable[str]) -> str:
    out = "ready"
    for s in statuses:
        if s in STATUS_ORDER and STATUS_ORDER.index(s) > STATUS_ORDER.index(out):
            out = s
    return out


# ---------------------------------------------------------------------------
# Safe relation expressions (R10): len(x), l2(x), abs, arithmetic and comparisons
# ---------------------------------------------------------------------------

def _l2(v: Any) -> float | None:
    if v is None:
        return None
    return math.sqrt(sum(float(x) * float(x) for x in v if x is not None))


_FUNCS = {"len": lambda v: None if v is None else len(v), "l2": _l2, "abs": lambda v: None if v is None else abs(v),
          "sum": lambda v: None if v is None else sum(x for x in v if x is not None),
          "min": lambda v: None if not v else min(v), "max": lambda v: None if not v else max(v)}


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    raise ValueError(f"unsupported expression node {type(node).__name__}")


def eval_expr(expr: str, scope: Mapping[str, Any]) -> Any:
    """Evaluate a relation expression (``len(children) == 0``, ``l2(vector)``) on one row; only names,
    attributes (paths), constants, ``len/l2/abs/sum/min/max``, arithmetic and comparisons are allowed."""
    tree = ast.parse(expr, mode="eval")

    def ev(n: ast.AST) -> Any:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, (ast.Name, ast.Attribute)):
            return _items.path_value(scope, _dotted(n))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and len(n.args) == 1:
            return _FUNCS[n.func.id](ev(n.args[0]))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.Not)):
            v = ev(n.operand)
            return None if v is None else (-v if isinstance(n.op, ast.USub) else not v)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = ev(n.left), ev(n.right)
            if a is None or b is None:
                return None
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            return a / b if b else None
        if isinstance(n, ast.Compare) and len(n.ops) == 1:
            a, b = ev(n.left), ev(n.comparators[0])
            if a is None or b is None:
                return None
            op = type(n.ops[0])
            return {ast.Eq: a == b, ast.NotEq: a != b, ast.Lt: a < b, ast.LtE: a <= b, ast.Gt: a > b,
                    ast.GtE: a >= b}[op]
        if isinstance(n, ast.BoolOp):
            vals = [ev(v) for v in n.values]
            return all(vals) if isinstance(n.op, ast.And) else any(vals)
        raise ValueError(f"unsupported expression {ast.dump(n)[:80]}")

    return ev(tree)


# ---------------------------------------------------------------------------
# Column statistics aggregated over fragments
# ---------------------------------------------------------------------------

def aggregate_stats(reader: TableReader) -> tuple[dict[str, ColumnStats], int | None, int]:
    """``({§6.4 leaf path: ColumnStats}, rows, bytes on disk)`` summed over the table's fragments."""
    schema = reader.schema()
    to_path: dict[str, str] = {}
    for path, _t in logical_leaves(schema):
        if parse_path(path).head in reader.partitions:
            continue
        try:
            to_path.setdefault(reader.fmt.leaf_path(path, schema), path)
        except (ValueError, KeyError):
            continue
    acc: dict[str, dict[str, Any]] = {}
    rows: int | None = 0
    size = 0
    for frag in reader.fragments():
        st = reader.fragment_stats(frag)
        size += int(frag.size or 0)
        rows = None if rows is None or st.rows is None else rows + int(st.rows)
        for leaf, cs in st.columns.items():
            path = to_path.get(leaf, leaf)
            a = acc.setdefault(path, {"uncompressed_bytes": 0, "num_values": 0, "null_count": 0, "min": None,
                                      "max": None, "minmax": True, "max_rep_level": 0, "max_def_level": 0,
                                      "storage_type": cs.storage_type, "kind": cs.kind})
            a["uncompressed_bytes"] += int(cs.uncompressed_bytes or 0)
            a["num_values"] = None if a["num_values"] is None or cs.num_values is None else \
                a["num_values"] + int(cs.num_values)
            a["null_count"] = None if a["null_count"] is None or cs.null_count is None else \
                a["null_count"] + int(cs.null_count)
            a["max_rep_level"] = max(a["max_rep_level"], cs.max_rep_level)
            a["max_def_level"] = max(a["max_def_level"], cs.max_def_level)
            if cs.min is None or cs.max is None:
                if (st.rows or 0) > 0 and cs.null_count != st.rows:
                    a["minmax"] = False
                continue
            try:
                a["min"] = cs.min if a["min"] is None or cs.min < a["min"] else a["min"]
                a["max"] = cs.max if a["max"] is None or cs.max > a["max"] else a["max"]
            except TypeError:
                a["minmax"] = False
    out = {}
    for path, a in acc.items():
        out[path] = ColumnStats(uncompressed_bytes=a["uncompressed_bytes"], null_count=a["null_count"],
                                num_values=a["num_values"], max_rep_level=a["max_rep_level"],
                                max_def_level=a["max_def_level"], min=a["min"] if a["minmax"] else None,
                                max=a["max"] if a["minmax"] else None, storage_type=a["storage_type"],
                                kind=a["kind"])
    return out, rows, size


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

@dataclass
class CheckRun:
    ctx: ServiceContext
    ref: str
    depth: str = "standard"
    reader: TableReader | None = None
    checks: list[CheckItemModel] = field(default_factory=list)
    columns: dict[str, str] = field(default_factory=dict)
    containers: dict[str, str] = field(default_factory=dict)
    partitions: dict[str, str] = field(default_factory=dict)
    item_tables: dict[str, str] = field(default_factory=dict)
    confirmed: dict[str, Any] = field(default_factory=dict)
    vocab: dict[str, str] = field(default_factory=dict)
    key_check: KeyCheckModel | None = None
    statuses: list[str] = field(default_factory=list)
    stats: dict[str, ColumnStats] = field(default_factory=dict)
    rows: int | None = None
    sample_rows: int = SAMPLE_ROWS
    storage_types: dict[str, str | None] = field(default_factory=dict)   # an item table: its parent's

    def add(self, name: str, ok: bool, detail: str = "", *, status: str | None = None, level: str | None = None,
            hint: str = "", column: str | None = None, partition: str | None = None, container: str | None = None
            ) -> None:
        lvl = level or ("info" if ok else "error")
        self.checks.append(CheckItemModel(name=name, ok=ok, level=lvl, detail=detail[:2000], hint=hint,  # type: ignore[arg-type]
                                          column=column, partition=partition))
        if ok or lvl != "error":
            return
        st = status or "schema_drift"
        self.statuses.append(st)
        if partition:
            self.partitions[partition] = worst([self.partitions.get(partition, "ready"), st])
        if container:
            self.containers[container] = worst([self.containers.get(container, "ready"), st])
        if column:
            self.columns[column] = worst([self.columns.get(column, "ready"), st])

    @property
    def table(self) -> Any:
        return self.ctx.table(self.ref)

    def model(self) -> TableCheckModel:
        status = worst(self.statuses)
        fp = sig = None
        pfs: dict[str, str] = {}
        if self.reader is not None:
            try:
                fp, sig = self.reader.fingerprint(), self.reader._sig
            except (ServiceError, OSError):
                pass
            # per-partition fingerprints let replay ignore drift in partitions a call did not read
            if fp is not None and self.reader.spec.partitions:
                try:
                    pfs = self.reader.partition_fingerprints()
                except (ServiceError, OSError):
                    pfs = {}
        types = self.storage_types or {path: cs.storage_type for path, cs in self.stats.items()}
        return TableCheckModel(status=status, columns=self.columns, containers=self.containers,
                               partitions=self.partitions, item_tables=self.item_tables, checks=self.checks,
                               fingerprint=fp, partition_fingerprints=pfs, signature=sig,
                               confirmed=_jsonable(self.confirmed), vocab=self.vocab, key_check=self.key_check,
                               storage_types=types)


def _jsonable(v: Any) -> Any:
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_jsonable(x) for x in v]
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


def _walk_columns(columns: Mapping[str, Any], prefix: str = "") -> Iterable[tuple[str, Any]]:
    """``(physical path, spec)`` of every declared column and nested field (containers as ``x[]``... are
    yielded before their fields; list-ness is decided by the data, so fields are joined with ``.`` and
    later qualified with the schema)."""
    for name, col in columns.items():
        path = f"{prefix}.{name}" if prefix else name
        yield path, col
        if is_container(col):
            yield from _walk_columns(col.fields, path)


def _schema_path(reader: TableReader, path: str) -> str:
    """A dotted declared path with ``[]`` inserted where the data has lists (``go.id`` -> ``go[].id``)."""
    import pyarrow as pa

    schema = reader.schema()
    names = path.split(".")
    if names[0] not in schema.names:
        return path
    t = schema.field(names[0]).type
    out = names[0]
    for name in names[1:]:
        while pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
            out += "[]"
            t = t.value_type
        if pa.types.is_struct(t) and t.get_field_index(name) >= 0:
            t = t.field(t.get_field_index(name)).type
            out += "." + name
        else:
            return out + "." + ".".join(names[names.index(name):])
    return out


def _arrow_type_of(reader: TableReader, path: str) -> Any:
    """The Arrow type at a declared path (lists kept on the last step); None when absent."""
    import pyarrow as pa

    schema = reader.schema()
    names = path.split(".")
    if names[0] in reader.partitions:
        return "partition"
    if names[0] not in schema.names:
        return None
    t = schema.field(names[0]).type
    for name in names[1:]:
        while pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t) or \
                pa.types.is_dictionary(t):
            t = t.value_type
        if not pa.types.is_struct(t) or t.get_field_index(name) < 0:
            return None
        t = t.field(t.get_field_index(name)).type
    return t


# ---------------------------------------------------------------------------
# R1, R2: layout probe and manifests
# ---------------------------------------------------------------------------

def r1_r2(run: CheckRun) -> bool:
    """Layout and manifest findings; False when the table has no data to read further."""
    t = run.table
    desc = t.descriptor
    layout = run.ctx.plugin("layout", t.layout)
    lspec = layout_spec(t)
    manifest = run.ctx.manifest(desc.source)
    upstream = "scan" not in (getattr(layout, "capabilities", ()) or ())
    for item in layout.probe(desc.root or "", lspec, manifest):
        status = PROBE_STATUS.get(item.name, "schema_drift")
        name = ("R2:" if item.name.startswith("manifest") else "R1:") + item.name
        run.add(name, item.ok, item.detail, status=status, level=item.level if not item.ok else "info",
                hint=item.hint, column=item.column if item.name.startswith("partition_") and not item.partition
                else None, partition=item.partition)
    for spec in desc.manifests:
        if manifest is None:
            if spec.required:
                run.add("R2:manifest_absent", False, f"required manifest {spec.path or 'inline'} is absent or "
                        "unreadable", status="missing", hint="restore the manifest written by the downloader")
            else:
                run.add("R2:manifest_absent", False, f"manifest {spec.path} is absent: release_verified false",
                        level="warning")
            break
        for key, want in (spec.require or {}).items():
            got = manifest.data.get(key) if manifest.data else None
            if key == "complete":
                got = manifest.complete
            if got != want:
                run.add("R2:manifest_require", False, f"manifest {key} is {got!r}, required {want!r}",
                        status="partial")
        if desc.release.expect is not None:
            got_rel = _release_from(desc.release.from_, manifest)
            if got_rel is not None and str(got_rel) != str(desc.release.expect):
                run.add("R2:release", False, f"release {got_rel} differs from the expected {desc.release.expect}",
                        status="stale")
        break
    if upstream:
        return False
    return not any(c.name in ("R1:location", "R1:fragments") and not c.ok for c in run.checks)


def _release_from(spec: Any, manifest: Any) -> Any:
    parts = spec if isinstance(spec, list) else [spec]
    for p in parts:
        if isinstance(p, str) and p.startswith("manifest."):
            return json_path(manifest.data, "$." + p[len("manifest."):])
    return None


_HASHED: dict[tuple[str, int, int], dict[str, str]] = {}


def _file_hash(path: str, algo: str) -> str | None:
    """Whole-file hash, recomputed only when the file's (size, mtime) signature changes."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (path, st.st_size, st.st_mtime_ns)
    done = _HASHED.setdefault(key, {})
    if algo not in done:
        h = hashlib.new(algo)
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        done[algo] = h.hexdigest()
    return done[algo]


def r2_manifest_checks(run: CheckRun) -> None:
    """Inline manifests checked by whole-file hash; ``ManifestSpec.checks`` (manifest row counts, recorded
    filters) compared with the data."""
    desc = run.table.descriptor
    lspec = layout_spec(run.table)
    prefix = (lspec.path or "").strip("/")
    for spec in desc.manifests:
        for name, entry in (spec.inline or {}).items():
            rel = str(name).lstrip("./")
            if prefix and not (rel == prefix or rel.startswith(prefix + "/")):
                continue
            path = os.path.join(desc.root or "", rel)
            for algo in ("sha256", "md5"):
                if entry.get(algo):
                    got = _file_hash(path, algo)
                    if got is None:
                        run.add("R2:inline", False, f"{rel} is listed in the inline manifest but absent",
                                status="missing")
                    elif got != str(entry[algo]).lower():
                        run.add("R2:inline", False, f"{rel}: {algo} {got[:12]}... differs from the inline manifest",
                                status="partial")
                    else:
                        run.add("R2:inline", True, f"{rel}: {algo} matches the inline manifest")
                    break
    manifest = run.ctx.manifest(desc.source)
    if manifest is None:
        return
    for spec in desc.manifests:
        for key, path in (spec.checks or {}).items():
            # "<table>.rows" scopes a check to one table; a bare "rows" applies to every table
            table, _, what = key.rpartition(".")
            if table and table != run.table.ref.table:
                continue
            want = json_path(manifest.data, path)
            if what == "rows" and run.rows is not None and isinstance(want, int) and not isinstance(want, bool):
                run.add("R2:checks.rows", want == run.rows, f"manifest {path} says {want} rows, the data has "
                        f"{run.rows}", status="partial")
            elif what != "rows" and want is not None:
                run.add(f"R2:checks.{what}", True, f"manifest {path}: {want}")


# ---------------------------------------------------------------------------
# R3, R4, R4b
# ---------------------------------------------------------------------------

def r3_footers(run: CheckRun) -> None:
    reader = run.reader
    assert reader is not None
    bad = 0
    for frag in reader.fragments():
        try:
            reader.fragment_stats(frag)
            reader.footer(frag)
        except FormatError as exc:
            bad += 1
            part = partition_label(frag.partition) or None
            run.add("R3", False, str(exc), status="partial", partition=part,
                    hint="re-download the fragment; an unreadable file is never an empty table")
    if not bad:
        run.add("R3", True, f"{len(reader.fragments())} fragment footer(s) readable")


def r4_types(run: CheckRun) -> None:
    reader = run.reader
    assert reader is not None
    spec = reader.spec
    strict = reader.desc.table_strict(reader.table.physical.table)
    declared_top = set(spec.columns) | set(spec.partitions)
    absent_optional: list[str] = []
    for path, col in _walk_columns(spec.columns):
        role = getattr(col, "role", None)
        if any(path.startswith(a + ".") for a in absent_optional):
            run.columns.setdefault(path, "missing")     # a field of an optional container the data lacks
            continue
        t = _arrow_type_of(reader, path)
        optional = bool(getattr(col, "optional", False))
        if t is None:
            if optional:
                run.add("R4", False, f"optional column {path} is absent from the data", level="warning",
                        column=path)
                run.columns.setdefault(path, "missing")
                absent_optional.append(path)
            else:
                run.add("R4", False, f"declared column {path} is absent from the data", status="schema_drift",
                        column=path, hint="the release changed; update the descriptor or the data")
            continue
        if t == "partition" or role is None:
            continue
        ok = arrow_compatible(role, str(t), parse=getattr(col, "parse", None),
                              stored_as=getattr(col, "stored_as", None), encoding=getattr(col, "encoding", None),
                              list_delimiter=getattr(col, "list_delimiter", None))
        if not ok:
            run.add("R4", False, f"{path}: Arrow type {t} does not fit role {role}", status="schema_drift",
                    column=path)
        else:
            run.columns.setdefault(path, "ready")
    # columns a ``column_patterns`` template declares: checked against the pattern's role, never undeclared
    patterned = set()
    for name in reader.top_columns():
        pspec = spec.pattern_column(name) if name not in declared_top else None
        if pspec is None:
            continue
        patterned.add(name)
        t = _arrow_type_of(reader, name)
        role = getattr(pspec, "role", None)
        if t is None or t == "partition" or role is None:
            continue
        if arrow_compatible(role, str(t), parse=getattr(pspec, "parse", None),
                            stored_as=getattr(pspec, "stored_as", None), encoding=getattr(pspec, "encoding", None),
                            list_delimiter=getattr(pspec, "list_delimiter", None)):
            run.columns.setdefault(name, "ready")
        else:
            run.add("R4", False, f"{name} (pattern column): Arrow type {t} does not fit role {role}",
                    status="schema_drift", column=name)
    if strict:
        extra = [n for n in reader.top_columns() if n not in declared_top and n not in patterned]
        if extra:
            run.add("R4:undeclared", False, f"undeclared physical columns under strict: {', '.join(extra[:20])}",
                    status="schema_drift")


#: Matrix findings (``fmt.matrix_checks``) that are key violations (R5); every other error is schema drift (R4).
MATRIX_KEY_FINDINGS = frozenset({"duplicate_key", "row_key_null", "null_key"})


def is_matrix(reader: TableReader) -> bool:
    return reader.spec.matrix is not None and callable(getattr(reader.fmt, "matrix_checks", None))


def r4_r5_matrix(run: CheckRun) -> None:
    """R4/R5 of a matrix table (§6.7): the format's header-axis findings per fragment (unparseable or
    duplicate headers, positional indexes, missing values; duplicate or null axis keys). Matrix columns
    are axis members, never declared columns; under ``strict`` the attribute columns of an axis that
    declares ``columns`` (an h5ad's obs/var) must be declared, as a table's columns must."""
    reader = run.reader
    assert reader is not None
    strict = reader.desc.table_strict(reader.table.physical.table)
    matrix = reader.spec.matrix
    n = 0
    for frag in reader.fragments():
        part = partition_label(frag.partition) or None
        where = reader.fragment_name(frag)
        for item in reader.fmt.matrix_checks(frag):
            n += 1
            key = item.name in MATRIX_KEY_FINDINGS
            run.add(f"{'R5' if key else 'R4'}:{item.name}", item.ok, f"{where}: {item.detail}",
                    status="key_violation" if key else "schema_drift", level=item.level, hint=item.hint,
                    column=item.column, partition=part)
        if strict and matrix is not None:
            for axis, spec in matrix.axes.items():
                if not spec.columns or spec.from_ not in ("index", "column", "table"):
                    continue
                try:
                    names = list(reader.fmt.axis_values(frag, axis).schema.names)
                except (FormatError, ValueError, AttributeError, NotImplementedError):
                    continue
                known = {*spec.columns, *spec.key.columns, *(spec.parse.fields if spec.parse else ()),
                         *([spec.index_name] if spec.index_name else []),
                         *([spec.column] if isinstance(spec.column, str) else []), "position"}
                extra = [c for c in names if c not in known and not str(c).startswith("_")]
                if extra:
                    n += 1
                    run.add("R4:undeclared", False, f"{where}: undeclared {axis} axis columns under strict: "
                            f"{', '.join(map(str, extra[:20]))}", status="schema_drift", column=f"@{axis}",
                            partition=part)
    bad = [c for c in run.checks if not c.ok and c.name.startswith(("R4:", "R5:"))]
    if not bad:
        run.add("R4", True, f"matrix axes of {len(reader.fragments())} fragment(s) parse"
                + ("" if n else " with no findings"))


def _universe_columns(run: CheckRun) -> list[tuple[str, str, Any, Any]]:
    """``(qualified id_type, physical column, IdTypeSpec, where)`` whose identity universe is a column of this
    table (``where``: the universe's row or item filter, JSON form, or None)."""
    t = run.table
    desc = t.descriptor
    out = []
    for name, it in desc.id_types.items():
        u = it.universe
        refs: list[Any] = u if isinstance(u, list) else [u]
        for r in refs:
            if r is None:
                continue
            where = None
            if isinstance(r, str):
                table, _, column = r.partition(".")
                cols = [column]
            else:
                table, cols, where = r.table, list(r.keys), r.where
            if table.count(".") == 1:
                src, table = table.split(".")
                if src != desc.source:
                    continue
            if table == t.ref.table:
                for c in cols[-1:]:
                    out.append((f"{desc.source}:{name}", c, it, where))
    return out


#: Distinct universe keys R4b reads before it samples them (see :func:`_spread`).
UNIVERSE_SCAN_VALUES = 100_000
#: Keys of every prefix R4b always checks, however rare the prefix.
_PER_PREFIX = 20


def _excluded_values(reader: TableReader, column: str) -> set[str]:
    """Rendered ``missing_values`` and ``placeholders`` of a column: declared non-keys, never universe keys."""
    col = reader.column_spec(column)
    return {render_value(x) for x in [*(getattr(col, "missing_values", None) or []),
                                      *(getattr(col, "placeholders", None) or [])]}


def _spread(values: Sequence[str], n: int) -> list[str]:
    """At most ``n`` of ``values``: the first few of every prefix (a rare prefix at the end of a file is still
    checked), the rest evenly spaced over the sorted values."""
    ordered = sorted(set(values))
    if len(ordered) <= n:
        return ordered
    by_prefix: dict[str, list[str]] = {}
    for v in ordered:
        by_prefix.setdefault(_prefix(v), []).append(v)
    picked: set[str] = set()
    for vs in by_prefix.values():
        picked.update(vs[:_PER_PREFIX])
        if len(picked) >= n:
            break
    need = n - len(picked)
    if need > 0:
        others = [v for v in ordered if v not in picked]
        step = len(others) / need                       # >= 1: there are more values than n
        picked.update(others[int(i * step)] for i in range(need))
    return sorted(picked)[:n]


def _universe_sample(run: CheckRun, column: str, where: Any) -> list[str]:
    """The universe keys of ``column`` R4b checks: the values the universe's ``where`` keeps (applied to each
    item when the column is inside a list), without the column's declared missing values and placeholders."""
    reader = run.reader
    assert reader is not None
    excluded = _excluded_values(reader, column)
    budget = int(run.ctx.settings.readiness.vocab_budget_bytes)
    if where is None:
        snap = reader.snapshot(column, budget_bytes=budget, max_values=UNIVERSE_SCAN_VALUES)
        raw: list[Any] = list(snap.values)
    else:
        from ..predicate import evaluate, from_json
        from .reader import predicate_paths

        pred = from_json(where)
        physical = reader.physical_path(column)
        container = physical[: physical.rindex("[]") + 2] if "[]" in physical else None
        lvls = _items.levels(container) if container else []
        found: dict[str, Any] = {}
        for m in reader.scan(pred, columns=[physical, *predicate_paths(pred)], budget_bytes=budget,
                             attribute_unknown=False):
            views = [v for v, _ in _items.explode(m.row, lvls)] if lvls else [m.row]
            for view in views:
                if lvls and evaluate(pred, view, None) is not True:
                    continue                        # another item of the same row matched the filter
                for v in _items.path_values(view, physical):
                    if v is not None:
                        found.setdefault(render_value(v), v)
            if len(found) >= UNIVERSE_SCAN_VALUES:
                break
        raw = list(found.values())
    keep = [str(v) for v in raw if v is not None and render_value(v) not in excluded]
    return _spread(keep, SAMPLE_VALUES)


def r4b_universe(run: CheckRun) -> None:
    reader = run.reader
    assert reader is not None
    for qid, column, it, where in _universe_columns(run):
        try:
            sample = _universe_sample(run, column, where)
        except (ServiceError, BudgetExceeded, FormatError) as exc:
            run.add("R4b", False, f"{qid}: universe sample not read ({exc})", level="warning", column=column)
            continue
        try:
            plugin = run.ctx.identifier(qid, sample)
        except Exception as exc:  # noqa: BLE001 - a plugin that cannot be configured is reported, not raised
            run.add("R4b", False, f"{qid}: identifier plugin {it.plugin} cannot be configured ({exc})",
                    status="plugin_unavailable", column=column)
            continue
        bad: dict[str, list[str]] = {}
        for v in sample:
            n = plugin.normalize(v)
            again = plugin.normalize(n.value) if isinstance(n, Normalized) else None
            if not isinstance(n, Normalized) or n.value != v or not isinstance(again, Normalized) or \
                    again.value != n.value:
                bad.setdefault(_prefix(v), []).append(v)
        scope = " (filtered by the universe's where)" if where is not None else ""
        if bad:
            total = sum(len(v) for v in bad.values())
            share = total / max(len(sample), 1)
            names = ", ".join(f"{p} ({len(v)})" for p, v in sorted(bad.items(), key=lambda kv: -len(kv[1]))[:10])
            run.add("R4b", False, f"{qid}: {total} of {len(sample)} sampled universe keys{scope} ({share:.1%}) are "
                    f"not in the canonical form of {it.plugin}; failing prefixes: {names}", status="encoding_drift",
                    column=column, hint="extend the plugin options (e.g. prefixes) or use prefixes: from_universe")
        else:
            run.add("R4b", True, f"{qid}: {len(sample)} universe keys canonical{scope}", column=column)


def _prefix(value: str) -> str:
    out = []
    for ch in value:
        if ch.isalpha():
            out.append(ch)
        else:
            break
    return "".join(out) or value[:3]


# ---------------------------------------------------------------------------
# R5, R5b: key nulls and uniqueness
# ---------------------------------------------------------------------------

def _hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def r5_keys(run: CheckRun, ref: str | None = None) -> KeyCheckModel:
    """R5 (non-nullable key parts without nulls) and R5b (uniqueness) for a table or item table.

    With ``row_identity: content_hash`` the key is a grouping key that may repeat: R5b then tests that
    no two rows are equal (their content hashes), sampled on the same key-prefix blocks (equal rows
    share every key part, so a sampled block holds every copy of its rows). With ``row_identity: none``
    the release holds exact copies of rows (25.09 interaction_evidence: 24,280 of 27,286,700 rows), so
    there is no uniqueness to test: R5b passes saying so, and the rows are scanned only for the null
    counts of nested non-nullable parts (flat parts are read from the footers)."""
    reader = run.ctx.reader(ref or run.ref)
    t = reader.table
    key = list(reader.key)
    spec_key = t.spec.key
    nullable = set(t.nullable_key)
    content = spec_key.row_identity == "content_hash" and not t.is_item_table
    copies = spec_key.row_identity == "none" and not t.is_item_table
    method = spec_key.check
    full_max = int(run.ctx.settings.readiness.key_check_full_max_rows)
    total_rows = run.rows if not t.is_item_table else None
    if method == "full" and run.depth != "deep" and total_rows is not None and total_rows > full_max:
        method = "sampled"
    if run.depth == "deep" and method != "none":
        method = "full"
    model = KeyCheckModel(method=method, at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    if not key or method == "none":
        model.detail = "no key check declared"
        return model
    # positional item keys ('col[]#') are not §6.4 paths: test the suffix before parsing
    flat = [k for k in key if not k.endswith("#") and not parse_path(k).crosses_list]
    nulls: dict[str, int] = {k: 0 for k in key if k not in nullable}
    # R5: footer null counts for flat parts; nested parts are counted from the scanned values below
    if not t.is_item_table:
        for k in flat:
            if k in nullable or k in reader.partitions:
                continue
            leaf_path = k
            st = run.stats.get(leaf_path)
            if st is not None and st.null_count is not None:
                nulls[k] = int(st.null_count)
    prefix = list(spec_key.sample_prefix) or key[:-1]
    prefix_idx = [key.index(p) for p in prefix if p in key]
    rows_hint = total_rows or 0
    if t.is_item_table and method == "sampled":
        rows_hint = _item_count_hint(run.ctx, reader)
    fraction = 1.0 if method == "full" or rows_hint <= run.sample_rows else run.sample_rows / rows_hint
    buckets = 1 << 16
    keep_below = max(1, int(buckets * fraction))
    row_filter = None
    parent_parts = [k for k in flat if k not in reader.partitions]
    if t.is_item_table and fraction < 1.0 and parent_parts:
        # every item of a parent row shares the parent's key parts: whole parent rows are sampled, decided before
        # their items are expanded. Deciding per item expanded and rendered every item: 5.7 M items of the 25.09
        # target item tables took 12 minutes of a standard check (an item table had no row count, so its
        # declared check: sampled always ran in full).
        def row_filter(row: Mapping[str, Any]) -> bool:
            return _hash(canonical([_items.path_value(row, k) for k in parent_parts])) % buckets < keep_below

        prefix_idx = []
    arrow_filter = None
    if fraction < 1.0 and prefix_idx and not t.is_item_table and len(flat) == len(key):
        # the prefix blocks are sampled on the Arrow arrays, before a row is converted: the content check of the
        # 27.3 M rows of 25.09 interaction_evidence converted every column of every row to hash 0.7% of them
        # (more than 15 minutes of the genetics and interaction session's first check)
        arrow_filter = _arrow_prefix_sample(reader, [key[i] for i in prefix_idx], buckets, keep_below)
        if arrow_filter is not None:
            prefix_idx = []
    presampled = row_filter is not None or arrow_filter is not None
    seen: set[str] = set()
    spill: _Spill | None = None
    dups = 0
    examples: list[str] = []
    scanned = 0
    types = reader.storage_types(key)
    fast = None
    if copies:
        # nothing to deduplicate; scan only when a non-nullable nested part has no footer null count
        fast = (0, [], 0) if all(k in flat for k in nulls) else None
        arrow_filter = None
    elif not content and not t.is_item_table and fraction >= 1.0 and len(flat) == len(key) and \
            total_rows is not None and total_rows <= SPILL_KEYS * ARROW_KEY_ROWS_PER_SPILL_KEY:
        fast = _arrow_key_duplicates(reader, key, types)
    try:
        if fast is None and t.is_item_table:
            # the items are counted in Arrow (whole parent rows sampled by their key values): the scan rendered every
            # item in Python, 44.6 M for the 25.09 l2g_prediction features
            counted = dict.fromkeys(nulls, 0)
            fast = _arrow_item_keys(reader, key, counted, types,
                                    None if fraction >= 1.0 else (buckets, keep_below))
            if fast is not None:
                for k, v in counted.items():
                    nulls[k] += v
        for m in (reader.scan(None, columns=None if content else [], attribute_unknown=False, row_filter=row_filter,
                              arrow_filter=arrow_filter)
                  if fast is None else ()):
            scanned += 1
            for k, v in zip(key, m.key):
                if v is None and k in nulls and (t.is_item_table or k not in flat):
                    nulls[k] += 1
            if copies:
                continue
            if fraction < 1.0 and prefix_idx:
                # the sample is decided on the prefix block before the full key is rendered
                if _hash(canonical([m.key[i] for i in prefix_idx])) % buckets >= keep_below:
                    continue
            ck = content_hash(m.row) if content else canonical(list(m.key), types)
            if fraction < 1.0 and not prefix_idx and not presampled and _hash(ck) % buckets >= keep_below:
                continue
            if spill is not None:
                spill.add(ck)
                continue
            if ck in seen:
                dups += 1
                if len(examples) < 5:
                    examples.append(ck)
                continue
            seen.add(ck)
            if len(seen) > SPILL_KEYS:
                spill = _Spill(run.ctx.settings.cache_dir)
                for s in seen:
                    spill.add(s)
                seen = set()
        if spill is not None:
            d, ex = spill.duplicates()
            dups += d
            examples.extend(ex[: 5 - len(examples)])
        if fast is not None:
            dups, examples, scanned = fast
    except (BudgetExceeded, TableUnavailable, FormatError) as exc:
        model.ok = None
        model.detail = f"key check not completed: {exc}"
        run.add("R5b", False, model.detail, level="warning")
        return model
    finally:
        # every exit (an unfinished scan, any other error, cancellation) removes the spill files
        if spill is not None:
            spill.close()
    model.duplicates = dups
    model.null_counts = {k: v for k, v in nulls.items() if v}
    null_bad = {k: v for k, v in nulls.items() if v and k not in nullable}
    where = f" ({t.ref})" if t.is_item_table else ""
    if null_bad:
        run.add("R5", False, f"non-nullable key parts with nulls{where}: " +
                ", ".join(f"{k}={v}" for k, v in null_bad.items()), status="key_violation",
                column=next(iter(null_bad)))
    else:
        basis = "footer null counts" if copies and not scanned else f"{scanned} keys"
        run.add("R5", True, f"no nulls in non-nullable key parts{where} ({basis})")
    sampled = (f"sampled {fraction:.1%} of {'parent rows' if row_filter is not None else 'prefix blocks'}"
               if fraction < 1.0 else "every key")
    if copies:
        run.add("R5b", True, "no uniqueness to test: rows have no identity (row_identity: none; exact copies "
                "occur in the release and are counted as stored)")
    elif dups and content:
        run.add("R5b", False, f"{dups} row(s) equal to another row ({sampled}; content hashes {', '.join(examples)})",
                status="key_violation", hint="row_identity content_hash needs rows that are never equal; every "
                "count over the rows would be wrong")
    elif dups:
        run.add("R5b", False, f"{dups} duplicate key(s){where} ({sampled}): {', '.join(examples)}",
                status="key_violation", hint="the declared key is not unique; every count over it would be wrong")
    elif content:
        run.add("R5b", True, f"no two rows equal (content identity; {method}: {sampled})")
    else:
        run.add("R5b", True, f"key unique{where} ({method}: {sampled})")
    model.ok = not dups and not null_bad
    model.detail = f"{scanned} keys scanned; {sampled}" if not copies else \
        f"{scanned} keys scanned for nulls; rows have no identity (row_identity: none)"
    return model


def _arrow_prefix_sample(reader: TableReader, parts: Sequence[str], buckets: int, keep_below: int
                         ) -> Callable[[Any], Any] | None:
    """A scan ``arrow_filter`` keeping the rows whose key prefix ``parts`` hash below ``keep_below`` of ``buckets``:
    rows with equal prefixes are kept or dropped together (the hash is of the values, so in every row group alike).
    Each distinct prefix of a row group is hashed once. Only the parts stored as compared are hashed: top-level
    columns that are neither partitions nor cleaned (25.09 interaction_evidence cleans targetA, a non-entity
    endpoint). Rows equal on the key, or equal outright, are equal on any of its parts, so a sample of fewer parts
    still holds every copy. None when no part qualifies, or for a reader that cannot say how its columns are stored:
    the caller samples the scanned rows."""
    import pyarrow as pa
    import pyarrow.compute as pc

    physical = getattr(reader, "physical_path", None)
    unclean = getattr(reader, "_unclean", None)
    if physical is None or unclean is None:
        return None
    names = []
    for p in parts:
        phys = physical(p)
        if not (p in reader.partitions or unclean(p) or "." in phys or "[" in phys):
            names.append(phys)
    if not names:
        return None

    def keep(tbl: Any) -> Any:
        cols = []
        for n in names:
            col = tbl.column(n)
            if not (pa.types.is_string(col.type) or pa.types.is_large_string(col.type)):
                col = pc.cast(col, pa.large_string())
            cols.append(col)
        joined = pc.binary_join_element_wise(*cols, "\x1f", null_handling="replace", null_replacement="\x00")
        wanted = [v for v in pc.unique(joined).to_pylist() if _hash(v) % buckets < keep_below]
        return pc.is_in(joined, value_set=pa.array(wanted, joined.type))

    return keep


def _arrow_key_duplicates(reader: TableReader, key: Sequence[str], types: Sequence[str | None]
                          ) -> tuple[int, list[str], int] | None:
    """``(duplicates, up to five rendered duplicate keys, rows)`` of a key of flat columns, counted on Arrow arrays
    with NULLS NOT DISTINCT: each part is dictionary-encoded over the whole table and the codes are combined into one
    integer per row, whose repeats are the duplicate keys. Rendering every key in Python took 3.4 minutes of a
    standard check of the 4.0 M-row 25.09 association_overall_direct table, and the indirect tables hold 13 M rows.
    When the product of the parts' cardinalities passes 63 bits (the seven-part key of the 14.5 M-row interaction
    table: about 5e21) the codes combined so far are encoded again, which bounds them by the row count.
    None when a part is a partition or a cleaned column (in-band unknowns) or the parts do not read as aligned
    arrays: the caller scans the rows instead."""
    import pyarrow as pa
    import pyarrow.compute as pc

    if not key or any(k in reader.partitions or reader._unclean(k) for k in key):
        return None
    indices: list[Any] = []
    dictionaries: list[Any] = []
    try:
        layout = None
        for part in key:
            chunks, where = [], []
            for frag, rg, arr in reader.leaf_arrays(part):
                where.append((frag.uri, rg, len(arr)))
                chunks.append(arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr)
            if layout is None:
                layout = where
            elif where != layout:
                return None
            if not chunks:
                return 0, [], 0
            enc = pc.dictionary_encode(pa.chunked_array(chunks).combine_chunks())
            del chunks
            dictionaries.append(enc.dictionary)
            indices.append(enc.indices)                 # int32, null where the part is null
        combined: Any = None
        width = 1
        for idx, d in zip(indices, dictionaries):
            w = len(d) + 1
            code = pc.add(pc.fill_null(idx.cast(pa.int64()), -1), 1)                  # 0 is null
            if combined is None:
                combined, width = code, w
                continue
            if width * w >= 2 ** 63:
                again = pc.dictionary_encode(combined)
                combined, width = again.indices.cast(pa.int64()), len(again.dictionary)
                if width * w >= 2 ** 63:
                    return None
            combined = pc.add(pc.multiply(combined, w), code)
            width *= w
        rows = len(combined)
        counts = pc.value_counts(combined)
        repeated = counts.filter(pc.greater(counts.field("counts"), 1))
        dups = int(pc.sum(pc.subtract(repeated.field("counts"), 1)).as_py() or 0)
        examples: list[str] = []
        for value, n in zip(repeated.field("values").slice(0, 5).to_pylist(),
                            repeated.field("counts").slice(0, 5).to_pylist()):
            row = pc.index(combined, value).as_py()
            parts = [None if not idx[row].is_valid else d[idx[row].as_py()].as_py()
                     for idx, d in zip(indices, dictionaries)]
            examples.extend([canonical(parts, types)] * (n - 1))   # one per repeat, as a scan lists
        examples = examples[:5]
    except MemoryError:
        raise                                          # out of memory, not an unanswerable check
    except (ServiceError, FormatError, ValueError, TypeError, NotImplementedError, pa.ArrowException):
        return None
    return dups, examples, rows


#: Items of parent rows that share their parent key (a broken parent key) are compared across rows in memory up to
#: this many; beyond it the item key check scans rows, which spills.
ARROW_CROSS_ITEMS = 2_000_000
_PART_HASH_MULT = 0x100000001B3                        # FNV-1a 64-bit prime: mixes the part hashes of a row


def _cleaner(reader: TableReader, path: str) -> frozenset[str] | None:
    """The rendered values cleaning reads as null at a key part (missing values, statistic codes, placeholders),
    as ``TableReader.clean`` applies them to rows; None when cleaning leaves the part alone. Raises
    ArrowUnsupported for conditional cleaning (``unknown_when``, ``placeholder_when``) and for a container whose
    fields are cleaned: those are decided on rows."""
    spec = reader.column_spec(path)
    if spec is None:
        return None
    tree = reader._clean_tree({"part": spec})
    if not tree:
        return None
    _name, _col, kids, rendered, _holders, conds, of = tree[0]
    if kids is not None or conds or of is not None:
        raise _items.ArrowUnsupported(f"{path}: cleaned by a condition or inside its fields")
    return frozenset(rendered) or None


def _cleaned(arr: Any, bad: frozenset[str] | None) -> Any:
    """``arr`` with the values cleaning reads as null set to null (each distinct value rendered once; a list's
    elements one by one, as rows are cleaned)."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    if not bad:
        return arr
    arr = arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr
    if pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type) or pa.types.is_fixed_size_list(arr.type):
        values, lengths = _items._list_values(arr)
        offsets = pa.array(np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64))
        return pa.LargeListArray.from_arrays(offsets, _cleaned(values, bad), mask=arr.is_null())
    if pa.types.is_nested(arr.type):
        raise _items.ArrowUnsupported(f"cleaning a {arr.type} key part")
    enc = pc.dictionary_encode(arr)
    flags = pa.array([v is not None and render_value(v) in bad for v in enc.dictionary.to_pylist()], pa.bool_())
    hit = pc.fill_null(pc.take(flags, enc.indices), False)
    return pc.if_else(hit, pa.scalar(None, arr.type), arr)


def _comparable(arr: Any) -> Any:
    """A key part's values as compared in canonical keys: NaN is null; a struct is its fields (by name) joined into
    one string; a list is its elements sorted (canonical keys render lists order-insensitively), joined into one
    string with its length (25.09 chemicalProbes ``origin`` is a list of strings and ``urls`` a list of
    ``{niceName, url}`` structs; drug_indication ``references`` a list of ``{source, ids[]}``). Raises
    ArrowUnsupported for maps and dictionaries (the caller reads rows)."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    t = arr.type
    if pa.types.is_floating(t):
        return pc.if_else(pc.is_nan(arr), pa.scalar(None, t), arr)
    if pa.types.is_struct(t):
        return _struct_text(arr)
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        values, lengths = _items._list_values(arr)
        if pa.types.is_nested(values.type):
            values = _comparable(values)                 # structs and lists inside, compared the same way
        if not (pa.types.is_string(values.type) or pa.types.is_large_string(values.type)):
            values = pc.cast(_comparable(values), pa.large_string())
        values = pc.fill_null(values.cast(pa.large_string()), "\x00")
        parent = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
        ordered = pa.table({"p": parent, "v": values}).sort_by([("p", "ascending"), ("v", "ascending")])
        offsets = pa.array(np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64))
        rebuilt = pa.LargeListArray.from_arrays(offsets, ordered.column("v").combine_chunks(), mask=arr.is_null())
        count = pc.cast(pc.list_value_length(arr), pa.large_string())
        joined = pc.binary_join(rebuilt, pa.scalar("\x1f", pa.large_string()))
        return pc.binary_join_element_wise(count, joined, pa.scalar("\x1e", pa.large_string()))
    if pa.types.is_nested(t) or pa.types.is_dictionary(t):
        raise _items.ArrowUnsupported(f"key part of type {t}")
    return arr


def _struct_text(arr: Any) -> Any:
    """One string per struct value: its fields (sorted by name, as canonical keys render a dict; lists and structs
    inside as :func:`_comparable` makes them) joined; null where the struct is null."""
    import pyarrow as pa
    import pyarrow.compute as pc

    names = sorted(f.name for f in arr.type)
    if not names:
        raise _items.ArrowUnsupported("a struct without fields")
    parts = []
    for name in names:
        child = _comparable(_items._child(arr, name))
        parts.append(pc.fill_null(pc.cast(child, pa.large_string()), "\x00"))
    joined = pc.binary_join_element_wise(*parts, pa.scalar("\x1d", pa.large_string())) if len(parts) > 1 \
        else parts[0]
    return pc.if_else(arr.is_valid(), joined, pa.scalar(None, pa.large_string()))


def _codes(arr: Any) -> tuple[Any, int]:
    """``(codes, width)``: dictionary codes of a comparable array, 0 for null and 1.. for its distinct values."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    arr = arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr
    enc = pc.dictionary_encode(arr)
    codes = np.asarray(pc.fill_null(enc.indices.cast(pa.int64()), -1).to_numpy(zero_copy_only=False), dtype=np.int64)
    return codes + 1, len(enc.dictionary) + 1


def _combine(parts: Sequence[tuple[Any, int]]) -> Any:
    """One int64 per item from per-part codes (``(codes, width)``), equal exactly when every part is equal. When
    the product of the widths passes 63 bits the codes combined so far are encoded again (bounded by the items)."""
    import numpy as np

    combined: Any = None
    width = 1
    for codes, w in parts:
        if combined is None:
            combined, width = np.asarray(codes, dtype=np.int64), max(int(w), 1)
            continue
        if width * w >= 2 ** 63:
            uniq, inverse = np.unique(combined, return_inverse=True)
            combined, width = inverse.reshape(-1).astype(np.int64), max(len(uniq), 1)
            if width * w >= 2 ** 63:
                raise _items.ArrowUnsupported("key too wide to combine")
        combined = combined * int(w) + codes
        width *= int(w)
    return combined


def _repeats(combined: Any) -> tuple[int, Any]:
    """``(repeats, the index of each repeat)``: the items whose combined code an earlier item already has (a key
    held by n items repeats n - 1 times, as a scan counts it)."""
    import numpy as np

    if combined is None or not len(combined):
        return 0, np.zeros(0, dtype=np.int64)
    order = np.argsort(combined, kind="stable")
    ordered = combined[order]
    same = ordered[1:] == ordered[:-1]
    return int(np.count_nonzero(same)), order[1:][same]


def _arrow_item_keys(reader: TableReader, key: Sequence[str], nulls: dict[str, int],
                     types: Sequence[str | None], sample: tuple[int, int] | None = None
                     ) -> tuple[int, list[str], int] | None:
    """``(repeats, up to five rendered repeated keys, items)`` of an item table's composed key, counted in Arrow
    and numpy with NULLS NOT DISTINCT; ``nulls`` (non-nullable parts) gets the per-item null counts.

    The composed key is the parent row's key plus the item key parts along the container path. Items of different
    rows can only share a key when their rows share the parent key, so the parent key is encoded first over every
    row (its flat columns only). Items of a row whose parent key is unique are compared within their row group,
    as ``(row, item parts)`` dictionary codes; the items of rows sharing a parent key are compared across rows at
    the end (up to :data:`ARROW_CROSS_ITEMS`). Memory is one row group's items plus a few integers per row: the
    rows path rendered each of the 44.6 M items of 25.09 l2g_prediction features in Python and had not finished
    after 15 minutes. ``sample`` (``(buckets, keep_below)``) keeps whole parent rows by a hash of their key values.
    In-band codes of a part read as null, as rows are cleaned (25.09 drug_indications ``maxPhaseForIndication``
    -1). None when a part is a partition, a nested parent column or cleaned by a condition, the files have no
    footers, or the parent has more rows than a flat key is counted for in Arrow: the caller scans rows (and
    spills)."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    from ..catalog import POSITION_MARK

    lvls = reader.levels
    parent = [k for k in key if not k.endswith(POSITION_MARK) and not parse_path(k).crosses_list]
    inner = [k for k in key if k not in parent]
    if not lvls or not inner or any(k in reader.partitions or "." in k or "[" in k for k in parent):
        return None
    if sample is not None and not parent:
        return None
    try:
        cleaning = {k: _cleaner(reader, k) for k in key if not k.endswith(POSITION_MARK)}
        placed = {k: _items.part_level(k, lvls) for k in inner}
        item_leaves = []
        for k, (d, rest) in placed.items():
            if rest != POSITION_MARK:
                item_leaves.append(reader.leaf(k))
        if not any(d == len(lvls) - 1 and rest != POSITION_MARK for d, rest in placed.values()):
            item_leaves.append(reader.leaf(lvls[-1].text))               # the innermost lists must be read
        parent_leaves = [reader.leaf(k) for k in parent]
        if any(x is None for x in [*item_leaves, *parent_leaves]):
            return None
        plan, _ = reader.plan(None, sorted({*item_leaves, *parent_leaves}), use_sidecars=False)
        groups: list[tuple[Any, int, int]] = []
        for frag, rgs in plan:
            info = reader.footer(frag)
            if info is None or rgs is None:
                return None
            groups.extend((frag, rg, int(info.row_groups[rg].rows)) for rg in rgs)
        if sum(n for _f, _r, n in groups) > SPILL_KEYS * ARROW_KEY_ROWS_PER_SPILL_KEY:
            return None                                # the parent keys of every row are held: as for flat keys
        # the parent key of every row: its dictionary codes, the group of rows sharing it, the null rows
        chunks: dict[str, list[Any]] = {k: [] for k in parent}
        for frag, rg, n in groups:
            tbl = reader.fmt.read_leaves(frag, parent_leaves, [rg]) if parent_leaves else None
            for k in parent:
                col = tbl.column(k)
                chunks[k].append(col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col)
        total_rows = sum(n for _f, _r, n in groups)
        pcodes: dict[str, tuple[Any, int]] = {}
        dictionaries: dict[str, Any] = {}
        for k in parent:
            arr = _comparable(_cleaned(pa.chunked_array(chunks.pop(k)).combine_chunks(), cleaning[k])) if total_rows \
                else pa.array([])
            enc = pc.dictionary_encode(arr)
            dictionaries[k] = enc.dictionary
            pcodes[k] = (np.asarray(pc.fill_null(enc.indices.cast(pa.int64()), -1).to_numpy(zero_copy_only=False),
                                    dtype=np.int64) + 1, len(enc.dictionary) + 1)
        row_code = _combine([pcodes[k] for k in parent]) if parent else np.zeros(total_rows, dtype=np.int64)
        _u, group_of, sizes = np.unique(row_code, return_inverse=True, return_counts=True)
        group_of = group_of.reshape(-1)
        shared = sizes[group_of] > 1
        del row_code, _u
        keep_row = None
        if sample is not None:
            buckets, keep_below = sample
            mixed = np.zeros(total_rows, dtype=np.uint64)
            for k, t in zip(parent, [types[key.index(k)] for k in parent]):
                hashes = np.array([_hash("null")] + [_hash(render_value(v, t)) for v in dictionaries[k].to_pylist()],
                                  dtype=np.uint64)
                mixed = mixed * np.uint64(_PART_HASH_MULT) ^ hashes[pcodes[k][0]]
            keep_row = (mixed % np.uint64(buckets)) < np.uint64(keep_below)
            del mixed
        repeats = 0
        examples: list[str] = []
        scanned = 0
        cross: dict[str, list[Any]] = {"": [], "#rows": [], **{k: [] for k in inner}}
        cross_items = 0
        base = 0

        def render(row: int, values: Mapping[str, Any]) -> str:
            got = []
            for k in key:
                if k in values:
                    got.append(values[k])
                else:
                    c = int(pcodes[k][0][row])
                    got.append(None if c == 0 else dictionaries[k][c - 1].as_py())
            return canonical(got, types)

        for frag, rg, n in groups:
            rows = None
            if keep_row is not None:
                rows = np.flatnonzero(keep_row[base:base + n])
                if not len(rows):
                    base += n
                    continue
            tbl = reader.fmt.read_leaves(frag, item_leaves, [rg])
            items = _items.arrow_items(tbl, lvls, inner, rows=rows)
            del tbl
            for k in inner:
                items.parts[k] = _cleaned(items.parts[k], cleaning.get(k))      # in-band codes read as null
            at = base + items.rows                                           # each item's row in the table
            scanned += len(items)
            for k in parent:
                if k in nulls:
                    nulls[k] += int(np.count_nonzero(pcodes[k][0][at] == 0))
            values = {k: _comparable(items.parts[k]) for k in inner}
            codes = {k: _codes(values[k]) for k in inner}
            for k in inner:
                if k in nulls:
                    nulls[k] += int(np.count_nonzero(codes[k][0] == 0))
            across = shared[at]
            if across.any():
                idx = pa.array(np.flatnonzero(across))
                cross[""].append(group_of[at[across]])
                cross["#rows"].append(at[across])
                for k in inner:
                    cross[k].append(items.parts[k].take(idx))
                cross_items += int(np.count_nonzero(across))
                if cross_items > ARROW_CROSS_ITEMS:
                    return None
            alone = np.flatnonzero(~across)
            if len(alone):
                local = items.rows[alone]
                combined = _combine([(local, int(local.max()) + 1)] + [(codes[k][0][alone], codes[k][1]) for k in inner])
                found, where = _repeats(combined)
                repeats += found
                for i in where[: 5 - len(examples)]:
                    j = int(alone[i])
                    examples.append(render(int(at[j]), {k: items.parts[k][j].as_py() for k in inner}))
            base += n
        if cross_items:
            group = np.concatenate(cross[""])
            at_rows = np.concatenate(cross["#rows"])
            parts = {k: pa.chunked_array(cross[k]).combine_chunks() for k in inner}
            combined = _combine([(group, int(group.max()) + 1)] + [_codes(_comparable(parts[k])) for k in inner])
            found, where = _repeats(combined)
            repeats += found
            for i in where[: 5 - len(examples)]:
                examples.append(render(int(at_rows[int(i)]), {k: parts[k][int(i)].as_py() for k in inner}))
    except MemoryError:
        raise                                          # out of memory, not an unanswerable check
    except (_items.ArrowUnsupported, ValueError, TypeError, KeyError, IndexError, NotImplementedError,
            pa.ArrowException):
        return None                                    # unreadable files (FormatError) are the caller's finding
    return repeats, examples, scanned


def _arrow_nested_repeats(reader: TableReader, container: str, columns: Sequence[str], identity: str,
                          max_rows: int | None = None) -> tuple[int, str, int, int, int] | None:
    """``(rows that repeat an item key, one repeated key, items, rows read, row groups read)`` of a nested container,
    in Arrow. An item key is unique under its parent: the row for a list in the row, the enclosing item for a list
    in a list (one source cited under two indications of a drug is not a repeat), so each row group is counted on
    its own. Every row group is read, or with ``max_rows`` whole row groups in a fixed pseudo-random order until
    that many rows are read: the 2,000-row sample converted to Python held 2.3 M screens of the 25.09
    target_essentiality table (1.97 GB, most of its standard check). None when the container or a key column
    cannot be read as Arrow lists and scalars (the caller samples rows)."""
    import numpy as np
    import pyarrow as pa

    path = container if container.endswith("]") else container + "[]"
    try:
        lvls = _items.levels(path)
        inner = lvls[-1].text
        parts = [inner] if identity == "value" else [f"{inner}.{c}" for c in columns]
        if not parts or lvls[0].names[0] in reader.partitions:
            return None
        cleaning = {p: _cleaner(reader, p) for p in parts}
        leaves = [reader.leaf(p) for p in parts]
        if any(x is None for x in leaves):
            return None
        plan, _ = reader.plan(None, sorted(set(leaves)), use_sidecars=False)
        groups: list[tuple[Any, int, int]] = []
        for frag, rgs in plan:
            info = reader.footer(frag)
            if info is None or rgs is None:
                return None
            groups.extend((frag, rg, int(info.row_groups[rg].rows)) for rg in rgs)
        if max_rows is not None:
            chosen, rows_read = [], 0
            for g in sorted(groups, key=lambda g: _hash(f"{reader.fragment_name(g[0])}:{g[1]}")):
                if rows_read >= max_rows:
                    break
                chosen.append(g)
                rows_read += g[2]
            groups = chosen
        bad = 0
        total = 0
        example = ""
        for frag, rg, _n in groups:
            items = _items.arrow_items(reader.fmt.read_leaves(frag, leaves, [rg]), lvls, parts)
            items.parts = {p: _cleaned(a, cleaning[p]) for p, a in items.parts.items()}
            total += len(items)
            if not len(items):
                continue
            values = [_comparable(items.parts[p]) for p in parts]
            combined = _combine([(items.parents, int(items.parents.max()) + 1)] + [_codes(v) for v in values])
            found, where = _repeats(combined)
            if not found:
                continue
            bad += len(np.unique(items.rows[where]))
            if not example:
                j = int(where[0])
                example = canonical([items.parts[p][j].as_py() for p in parts])
    except MemoryError:
        raise
    except (_items.ArrowUnsupported, ValueError, TypeError, KeyError, IndexError, NotImplementedError,
            pa.ArrowException):
        return None
    return bad, example, total, sum(n for _f, _r, n in groups), len(groups)


def _item_count_hint(ctx: ServiceContext, reader: TableReader) -> int:
    """The items of an item table as its footers count them: the most values of a field directly in the innermost
    list (a null or empty list adds one null value, so this errs high). 0 when the footers do not say."""
    if not reader.levels:
        return 0
    inner = reader.levels[-1].text + "."
    try:
        cols, _rows, _size = aggregate_stats(ctx.reader(str(reader.table.physical)))
    except (ServiceError, FormatError, TableUnavailable):
        return 0
    counts = [int(cs.num_values or 0) for path, cs in cols.items()
              if path.startswith(inner) and "[" not in path[len(inner):]]
    return max(counts, default=0)


#: Distinct keys held in memory before the uniqueness pass spills to hash-partitioned files.
SPILL_KEYS = 2_000_000
#: A flat key is counted on Arrow codes (8 bytes per row and part) when the table has at most this many rows per
#: spilled key; a larger one takes the bounded-memory pass (25.09 literature: 152 M rows).
ARROW_KEY_ROWS_PER_SPILL_KEY = 8
SPILL_PREFIX = "keycheck."


class _Spill:
    """Hash-partitioned temporary files for a bounded-memory full uniqueness pass. The directory is
    ``<cache_dir>/keycheck.<pid>.<random>``: :meth:`close` removes it on every exit of the check, and
    :func:`sweep_spills` removes those of data children that died before they could."""

    def __init__(self, cache_dir: Any, parts: int = 64) -> None:
        os.makedirs(str(cache_dir), exist_ok=True)
        self.dir = tempfile.mkdtemp(prefix=f"{SPILL_PREFIX}{os.getpid()}.", dir=str(cache_dir))
        self.parts = parts
        self.files = [open(os.path.join(self.dir, f"{i:02d}"), "w", encoding="utf-8") for i in range(parts)]

    def close(self) -> None:
        """Close the files and remove the directory (idempotent)."""
        import shutil

        for f in self.files:
            try:
                f.close()
            except OSError:
                pass
        shutil.rmtree(self.dir, ignore_errors=True)

    def add(self, key: str) -> None:
        self.files[_hash(key) % self.parts].write(key.replace("\n", "\\n") + "\n")

    def duplicates(self) -> tuple[int, list[str]]:
        for f in self.files:
            f.close()
        dups = 0
        examples: list[str] = []
        try:
            for i in range(self.parts):
                seen: set[str] = set()
                with open(os.path.join(self.dir, f"{i:02d}"), encoding="utf-8") as fh:
                    for line in fh:
                        if line in seen:
                            dups += 1
                            if len(examples) < 5:
                                examples.append(line.strip())
                        seen.add(line)
        finally:
            self.close()
        return dups, examples


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:          # EPERM: alive, another user's
        return True
    return True


def sweep_spills(cache_dir: Any) -> list[str]:
    """Remove spill directories left by data children that were killed mid-check (their pid is gone),
    and pre-pid ``keycheck.<random>`` ones. Called when a data child starts; returns the removed paths."""
    import shutil

    removed: list[str] = []
    try:
        entries = os.listdir(str(cache_dir))
    except OSError:
        return removed
    for name in entries:
        if not name.startswith(SPILL_PREFIX):
            continue
        path = os.path.join(str(cache_dir), name)
        head = name[len(SPILL_PREFIX):].split(".", 1)
        if len(head) == 2 and head[0].isdigit() and _pid_alive(int(head[0])):
            continue
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)
    return removed


def access_indexes(run: CheckRun) -> None:
    """``access_paths`` with ``via: sidecar_index, build: readiness`` are built here, within the scan budget
    (``build: on_demand`` ones are built on the first read). An over-budget or failed build is a per-column
    warning naming the offline command: the tools reading through it answer ``too_large`` until it is built."""
    from .sidecar import BUILD_HINT, build_access_index

    reader = run.reader
    assert reader is not None
    for ap in reader.spec.access_paths:
        if ap.via != "sidecar_index" or ap.build != "readiness":
            continue
        for column in ap.columns:
            try:
                path, n = build_access_index(reader, column)
            except (BudgetExceeded, TableUnavailable, FormatError, ServiceError, OSError) as exc:
                run.add("access_index", False, f"{column}: sidecar index not built ({exc})", level="warning",
                        column=column, hint=f"build it offline with `{BUILD_HINT}`")
                continue
            run.add("access_index", True, f"{column}: sidecar index ready ({n} values, {path.name})", column=column)


def r5b_item_keys(run: CheckRun) -> None:
    """Item keys of nested containers are unique under their parent (the row, or the enclosing item of a list in a
    list), counted in Arrow (``_arrow_nested_repeats``): every row at depth deep, whole row groups holding at least
    as many rows as the relation sample otherwise. A container the Arrow path cannot read is checked on a sample
    of rows."""
    reader = run.reader
    assert reader is not None
    containers = [(p, c) for p, c in _walk_columns(reader.spec.columns) if is_container(c) and c.item_key is not None
                  and c.item_key.check != "none" and c.item_key.identity != "position"]   # positions never repeat
    if not containers:
        return
    deep = run.depth == "deep"
    on_rows = []
    for path, col in containers:
        try:
            got = _arrow_nested_repeats(reader, _schema_path(reader, path), col.item_key.columns,
                                        col.item_key.identity, None if deep else _RELATION_SAMPLE)
        except (ServiceError, FormatError) as exc:
            run.add("R5b:items", False, f"{path}: item keys not checked ({exc})", level="warning", container=path)
            continue
        if got is None:
            on_rows.append((path, col))
            continue
        bad, example, items, rows, groups = got
        where = "every row" if deep else f"{rows} rows in {groups} sampled row group(s)"
        if bad:
            run.add("R5b:items", False, f"{path}: {bad} row(s) repeat an item key under one parent ({where}, "
                    f"{items} items; e.g. {example})", status="key_violation", container=path)
        else:
            run.containers.setdefault(path, "ready")
    containers = on_rows
    if not containers:
        return
    paths = [p.split(".")[0] for p, _ in containers]
    try:
        rows = reader.sample_rows(_RELATION_SAMPLE, seed=1, columns=sorted(set(paths)))
    except (ServiceError, FormatError) as exc:
        run.add("R5b:items", False, f"item keys not checked ({exc})", level="warning")
        return
    for path, col in containers:
        sp = _schema_path(reader, path)
        container = sp if sp.endswith("]") else sp
        bad = 0
        example = ""
        for row in rows:
            d = _items.item_key_duplicates(row, container, col.item_key.columns, col.item_key.identity)
            if d:
                bad += 1
                example = example or d[0]
        if bad:
            run.add("R5b:items", False, f"{path}: {bad} sampled row(s) repeat an item key (e.g. {example})",
                    status="key_violation", container=path)
        else:
            run.containers.setdefault(path, "ready")


# ---------------------------------------------------------------------------
# R6: verified:false facts and literal constraints
# ---------------------------------------------------------------------------

_LITERAL_OPS = ("<", "<=", ">", ">=", "==", "!=", "in", "is_finite", "not_null")


def r6_facts(run: CheckRun) -> None:
    reader = run.reader
    assert reader is not None
    for path, col in _walk_columns(reader.spec.columns):
        role = getattr(col, "role", None)
        if _arrow_type_of(reader, path) is None:
            continue
        sp = _schema_path(reader, path)
        if role == "measure" and getattr(col, "verified", True) is False:
            plugin = run.ctx.statistic(getattr(col, "statistic", None)) or \
                run.ctx.statistic(getattr(col, "fallback", None))
            if plugin is None:
                run.add("R6", False, f"{path}: statistic {col.statistic} is not registered", status="plugin_unavailable",
                        column=path)
                continue
            snap = _snap(run, sp)
            stats = run.stats.get(sp) or ColumnStats(0, None)
            res = plugin.validate(stats, snap, col)
            run.confirmed[path] = dict(res.facts)
            if res.confirmed is False:
                run.add("R6", False, f"{path}: {res.detail}", status="encoding_drift", column=path,
                        hint="the descriptor's scale or encoding is refuted by the data")
            elif res.confirmed is None:
                run.add("R6", False, f"{path}: not confirmed ({res.detail}); used for disclosure only (I9)",
                        level="warning", column=path)
            else:
                run.add("R6", True, f"{path}: facts confirmed", column=path)
        elif role == "flag" and getattr(col, "encoding", None) and getattr(col, "verified", True) is False:
            snap = _snap(run, sp)
            codes = list(col.encoding)
            _confirm_set(run, path, snap, codes, "encoding")
        elif role in ("category", "scope") and isinstance(getattr(col, "vocab", None), list) and \
                getattr(col, "verified", True) is False:
            snap = _snap(run, sp)
            allowed = list(col.vocab) + list((getattr(col, "aliases", {}) or {}).keys())
            _confirm_set(run, path, snap, allowed, "vocab", every=list(col.vocab))
        if is_container(col) and (getattr(col, "null_means", None) or getattr(col, "empty_means", None) or
                                  (col.coverage is not None and col.coverage.verified is False)):
            _container_counts(run, path, sp)
    for c in reader.spec.constraints:
        if c.op in _LITERAL_OPS:
            _literal_constraint(run, c)


def _snap(run: CheckRun, path: str) -> ValueSnapshot | None:
    reader = run.reader
    assert reader is not None
    try:
        return reader.snapshot(path)
    except (ServiceError, FormatError):
        return None


def _confirm_set(run: CheckRun, path: str, snap: ValueSnapshot | None, allowed: Sequence[Any], what: str,
                 every: Sequence[Any] | None = None) -> None:
    if snap is None or not snap.complete:
        run.confirmed[path] = {"confirmed": None}
        run.add("R6", False, f"{path}: {what} not confirmed (no complete snapshot); disclosure only",
                level="warning", column=path)
        return
    keys = {render_value(a) for a in allowed}
    extra = [v for v in snap.values if render_value(v) not in keys and str(v) not in {str(a) for a in allowed}]
    never = [a for a in (every if every is not None else allowed)
             if all(render_value(v) != render_value(a) and str(v) != str(a) for v in snap.values)]
    ok = not extra and not never
    run.confirmed[path] = {"confirmed": ok, "codes": list(snap.values), "null_count": snap.null_count}
    if ok:
        run.add("R6", True, f"{path}: {what} confirmed", column=path)
    else:
        detail = []
        if extra:
            detail.append(f"undeclared value(s) {', '.join(map(repr, extra[:10]))}")
        if never:
            detail.append(f"declared value(s) never observed {', '.join(map(repr, never[:10]))}")
        run.add("R6", False, f"{path}: {what} refuted: {'; '.join(detail)}", status="encoding_drift", column=path)


def _container_counts(run: CheckRun, path: str, schema_path: str) -> None:
    reader = run.reader
    assert reader is not None
    container = schema_path if schema_path.endswith("]") else schema_path + "[]"
    try:
        struct = _arrow_struct_counts(reader, schema_path)
        counts = struct
        if counts is None:
            lvls = _items.levels(container)
            leaf = reader.leaf(container)
            if leaf is None:
                # reading no column made every row an empty dict: each container counted as null
                raise ValueError(f"{container} is not a list in the data")
            counts = _arrow_container_counts(reader, lvls, leaf)
        if counts is None:
            counts = _items.ContainerCounts()
            for frag, rgs in reader.plan(None, [leaf], use_sidecars=False)[0]:
                info = reader.footer(frag)
                for rg in (rgs if rgs is not None else [None]):
                    for row in reader._read(frag, rg, [leaf], info):
                        for _ in _items.explode(row, lvls, counts):
                            pass
    except (ServiceError, FormatError, ValueError) as exc:
        run.add("R6:containers", False, f"{path}: null vs empty counts not read ({exc})", level="warning",
                container=path)
        return
    run.confirmed[f"{path}[]"] = {"null": counts.null, "empty": counts.empty, "nonempty": counts.nonempty,
                                  "items": counts.items, "null_items": counts.null_items}
    if struct is not None:
        detail = f"{path}: {counts.null} null, {counts.nonempty} present (a struct, not a list)"
    else:
        detail = (f"{path}: {counts.null} null, {counts.empty} empty, {counts.nonempty} non-empty list(s); "
                  f"{counts.null_items} null item(s)")
    run.add("R6:containers", True, detail, container=path)
    run.containers.setdefault(path, "ready")


def _arrow_struct_counts(reader: TableReader, schema_path: str) -> Any:
    """``ContainerCounts`` of a struct container outside any list (25.09 target ``tep``, ``hallmarks``): a row's
    struct is null or present (one item). The list path counted ``tep[]`` and read no column for it, so every one of
    the 78,726 genes read as null, 41 TEPs included. None when the path is not a struct reached without a list."""
    import pyarrow as pa

    if "[" in schema_path or not schema_path or schema_path.split(".")[0] in reader.partitions:
        return None
    t = _arrow_type_of(reader, schema_path)
    if t is None or t == "partition" or not pa.types.is_struct(t):
        return None
    leaf = reader.leaf(schema_path)
    if leaf is None:
        return None
    names = schema_path.split(".")
    counts = _items.ContainerCounts()
    try:
        for frag, rgs in reader.plan(None, [leaf], use_sidecars=False)[0]:
            if reader.footer(frag) is None:
                return None
            for rg in (rgs if rgs is not None else [None]):
                arr = reader.fmt.read_leaves(frag, [leaf], [rg]).column(names[0]).combine_chunks()
                for name in names[1:]:
                    arr = _items._child(arr, name)
                present = len(arr) - arr.null_count
                counts.rows += len(arr)
                counts.null += arr.null_count
                counts.nonempty += present
                counts.items += present
                tally = counts.by_level.setdefault(schema_path, {})
                tally["rows"] = tally.get("rows", 0) + len(arr)
                tally["null"] = tally.get("null", 0) + arr.null_count
    except MemoryError:
        raise
    except (_items.ArrowUnsupported, pa.ArrowException, KeyError, IndexError, TypeError):
        return None
    return counts


def _arrow_container_counts(reader: TableReader, lvls: Sequence[Any], leaf: str | None) -> Any:
    """``ContainerCounts`` of a list of structs counted on Arrow arrays: null, empty and non-empty lists at the
    innermost level, its items and null items (``_items.explode``'s counts without building a Python row).
    Converting every row of the 25.09 expression table (43,804 genes with nested tissues) to count its containers
    was most of a three-minute standard check. None when the path or the items are not structs and lists, or the
    table has no footers: cleaning can turn a coded scalar item into a null, so those are counted on rows."""
    import pyarrow as pa
    import pyarrow.compute as pc

    if not leaf or not lvls or reader.partitions:
        return None
    counts = _items.ContainerCounts()
    try:
        for frag, rgs in reader.plan(None, [leaf], use_sidecars=False)[0]:
            if reader.footer(frag) is None:
                return None
            for rg in (rgs if rgs is not None else [None]):
                tbl = reader.fmt.read_leaves(frag, [leaf], [rg])
                head = lvls[0].names[0]
                if head not in tbl.column_names:
                    return None
                arr = tbl.column(head).combine_chunks()
                counts.rows += len(arr)
                for depth, lvl in enumerate(lvls):
                    for name in (lvl.names[1:] if depth == 0 else lvl.names):
                        index = arr.type.get_field_index(name) if pa.types.is_struct(arr.type) else -1
                        if index < 0:
                            return None
                        arr = arr.flatten()[index]                            # the struct's nulls applied
                    if not (pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type)):
                        return None
                    values = arr.flatten()                                    # items of the valid lists
                    if depth < len(lvls) - 1:
                        arr = values.filter(values.is_valid())                # a null item has no lists below
                        continue
                    if not pa.types.is_struct(values.type):
                        return None
                    lengths = pc.list_value_length(arr)
                    counts.null += arr.null_count
                    counts.empty += int(pc.sum(pc.equal(lengths, 0)).as_py() or 0)
                    counts.nonempty += int(pc.sum(pc.greater(lengths, 0)).as_py() or 0)
                    counts.null_items += values.null_count
                    counts.items += len(values) - values.null_count
    except MemoryError:
        raise                                          # out of memory, not an unanswerable check
    except (pa.ArrowException, KeyError, IndexError, TypeError):
        return None
    return counts


def _literal_constraint(run: CheckRun, c: Any) -> None:
    reader = run.reader
    assert reader is not None
    path = _schema_path(reader, c.column)
    name = f"constraint:{c.column} {c.op}" + (f" {c.value}" if c.value is not None else "")
    st = run.stats.get(path)
    refuted: bool | None = None
    detail = ""
    if c.op in ("<", "<=", ">", ">=", "==", "!="):
        lo, hi = (st.min, st.max) if st is not None else (None, None)
        try:
            v = float(c.value) if isinstance(c.value, (int, float)) and not isinstance(c.value, bool) else c.value
            if lo is None or hi is None:
                refuted = None
            elif c.op == "<":
                refuted = not hi < v
            elif c.op == "<=":
                refuted = not hi <= v
            elif c.op == ">":
                refuted = not lo > v
            elif c.op == ">=":
                refuted = not lo >= v
            elif c.op == "==":
                refuted = not (lo == hi == v)
            else:
                refuted = lo == hi == v
            detail = f"observed range [{lo}, {hi}]"
        except TypeError:
            refuted = None
    elif c.op == "not_null":
        n = st.null_count if st is not None else None
        refuted = None if n is None else n > 0
        detail = f"{n} null(s)"
    elif c.op in ("in", "is_finite"):
        snap = _snap(run, path)
        if snap is not None and snap.complete:
            if c.op == "in":
                allowed = {render_value(x) for x in (c.value or [])}
                extra = [v for v in snap.values if render_value(v) not in allowed]
                refuted = bool(extra)
                detail = f"values outside the set: {extra[:10]}" if extra else "all values in the set"
            else:
                inf = [v for v in snap.values if isinstance(v, float) and math.isinf(v)]
                refuted = bool(inf or snap.nan_count or snap.null_count)
                detail = f"{snap.nan_count} NaN, {snap.null_count} null, {len(inf)} infinite value(s)"
    if refuted is None:
        run.confirmed[name] = {"confirmed": None}
        if c.verified is False:
            run.add("R6:constraint", False, f"{name}: not confirmed ({detail or 'no statistics'})", level="warning",
                    column=c.column)
        return
    run.confirmed[name] = {"confirmed": not refuted, "on_refute": c.on_refute, "origin": c.origin}
    if not refuted:
        run.add("R6:constraint", True, f"{name}: holds ({detail})", column=c.column)
    elif c.on_refute == "not_ready":
        run.add("R6:constraint", False, f"{name} is refuted ({detail}; {c.origin})", status="encoding_drift",
                column=c.column)
    else:
        run.add("R6:constraint", False, f"{name} is refuted ({detail}); on_refute: {c.on_refute}", level="warning",
                column=c.column)


# ---------------------------------------------------------------------------
# R7: vocabulary snapshots, mirrored partition columns
# ---------------------------------------------------------------------------

def r7_vocab(run: CheckRun) -> None:
    reader = run.reader
    assert reader is not None
    budget = int(run.ctx.settings.readiness.vocab_budget_bytes)
    for path, col in _walk_columns(reader.spec.columns):
        role = getattr(col, "role", None)
        if role not in ("category", "scope") or _arrow_type_of(reader, path) is None:
            continue
        sp = _schema_path(reader, path)
        try:
            snap = reader.snapshot(sp, budget_bytes=budget)
        except (ServiceError, FormatError) as exc:
            run.add("R7", False, f"{path}: no vocabulary snapshot ({exc})", level="warning", column=path)
            continue
        excluded = {render_value(x) for x in [*getattr(col, "placeholders", []), *getattr(col, "missing_values", [])]}
        values = tuple(v for v in snap.values if render_value(v) not in excluded)
        snap = ValueSnapshot(values, snap.counts, snap.null_count, snap.nan_count, snap.complete,
                             snap.fingerprint, snap.storage_type)
        if snap.fingerprint:
            run.vocab[path] = snap.fingerprint
            run.ctx.vocab[snap.fingerprint] = snap
        meanings = getattr(col, "values", None) or {}
        # a placeholder-like word the descriptor gives a meaning (`values: {"None": "no coating"}`) is a value
        odd = [v for v in values if isinstance(v, str) and v.strip() in _SUSPICIOUS and v not in meanings]
        if odd:
            run.add("R7", False, f"{path}: placeholder-like value(s) {odd!r} in a categorical column; declare them as "
                    "placeholders or missing_values", level="warning", column=path)
        else:
            run.add("R7", True, f"{path}: {len(values)} value(s){'' if snap.complete else ' (partial snapshot)'}",
                    column=path)
    for name, part in reader.spec.partitions.items():
        for mirror in part.mirrored_by:
            _mirror_check(run, name, mirror)


def _mirror_check(run: CheckRun, partition: str, mirror: str) -> None:
    reader = run.reader
    assert reader is not None
    bad: list[str] = []
    try:
        for frag in reader.fragments():
            want = frag.partition.get(partition)
            info = reader.footer(frag)
            leaf = reader.leaf(mirror)
            if leaf is None or info is None or not info.row_groups:
                continue
            for _f, _rg, vals in reader.leaf_values(mirror, Eq(partition, want) if want is not None else None):
                if any(v is not None and str(v) != str(want) for v in vals):
                    bad.append(partition_label(frag.partition))
                break
    except (ServiceError, FormatError) as exc:
        run.add("R7:mirror", False, f"{mirror} vs partition {partition} not checked ({exc})", level="warning")
        return
    if bad:
        run.add("R7:mirror", False, f"{mirror} differs from partition {partition} in {', '.join(sorted(set(bad))[:10])}",
                status="schema_drift", column=mirror)
    else:
        run.add("R7:mirror", True, f"{mirror} equals partition {partition}")


# ---------------------------------------------------------------------------
# R8: sentinels
# ---------------------------------------------------------------------------

_SENTINEL_OPS = {"eq": lambda a, b: a == b, "ne": lambda a, b: a != b, "lt": lambda a, b: a < b,
                 "le": lambda a, b: a <= b, "gt": lambda a, b: a > b, "ge": lambda a, b: a >= b,
                 "in": lambda a, b: a in b}


def _sentinel_rows(reader: TableReader, key: Mapping[str, Any]) -> list[dict[str, Any]]:
    preds = []
    for name, value in key.items():
        path = name if name in reader.key else reader.physical_path(name)
        preds.append(Eq("/" + path, value) if value is not None else _null(path))
    pred = preds[0] if len(preds) == 1 else And(tuple(preds))
    return [m.row for m in reader.scan(pred, columns=["*"], attribute_unknown=False)]


def _null(path: str) -> Any:
    from ..predicate import IsNull
    return IsNull("/" + path)


def _expect_ok(reader: TableReader, rows: Sequence[Mapping[str, Any]], expect: Mapping[str, Any]
               ) -> tuple[bool, str]:
    def vals(row: Mapping[str, Any], name: str) -> list[Any]:
        path = name if name in reader.key else reader.physical_path(name)
        return _items.path_values(row, path)

    for word, spec in expect.items():
        if word == "min_rows":
            if len(rows) < int(spec):
                return False, f"{len(rows)} row(s), expected at least {spec}"
        elif word == "nonempty":
            for p in spec:
                if not any(any(v not in (None, [], "") for v in vals(r, p)) for r in rows):
                    return False, f"{p} is empty"
        elif word == "is_null":
            for p in spec:
                if not all(all(is_null(v) for v in vals(r, p)) for r in rows):
                    return False, f"{p} is not null"
        elif word == "is_empty":
            for p in spec:
                if not all(not vals(r, p) or all(v in (None, []) for v in vals(r, p)) for r in rows):
                    return False, f"{p} is not empty"
        elif word == "contains":
            for p, want in spec.items():
                if not any(want in vals(r, p) for r in rows):
                    return False, f"{p} does not contain {want!r}"
        elif word == "items":
            for p, cond in spec.items():
                n = max((len(vals(r, p if p.endswith("]") else p + "[]")) for r in rows), default=0)
                text = str(cond).strip()
                op, num = ("==", int(text)) if text.lstrip("-").isdigit() else (text.rstrip("0123456789 ").strip(),
                                                                                int(text.lstrip("<>=! ")))
                ok = {"==": n == num, ">=": n >= num, "<=": n <= num, ">": n > num, "<": n < num}.get(op, False)
                if not ok:
                    return False, f"{p} has {n} item(s), expected {cond}"
        else:
            if isinstance(spec, Mapping):
                op, want = next(iter(spec.items()))
                test = _SENTINEL_OPS[op]
                if not any(any(v is not None and _safe(test, v, want) for v in vals(r, word)) for r in rows):
                    return False, f"{word} is not {op} {want!r}"
            elif not any(spec in vals(r, word) or (spec is None and not vals(r, word)) for r in rows):
                got = [vals(r, word) for r in rows][:3]
                return False, f"{word} is {got!r}, expected {spec!r}"
    return True, ""


def _safe(test: Any, a: Any, b: Any) -> bool:
    try:
        return bool(test(a, b))
    except TypeError:
        return False


def r8_sentinels(run: CheckRun, ref: str | None = None) -> None:
    reader = run.ctx.reader(ref or run.ref)
    spec = reader.table.spec.sentinels
    if spec is None or not run.ctx.settings.readiness.sentinels:
        return
    where = f" ({reader.ref})" if reader.table.is_item_table else ""
    for s in spec.present:
        if s.via is not None:
            continue                                   # remote sentinels are probed by the gateway
        try:
            rows = _sentinel_rows(reader, s.key)
        except (ServiceError, FormatError) as exc:
            run.add("R8", False, f"sentinel {s.key}{where} not checked ({exc})", level="warning")
            continue
        if not rows:
            run.add("R8", False, f"present sentinel {s.key}{where} is missing", status="partial",
                    hint="a positive control is absent: the data is incomplete or from another release")
            continue
        ok, why = _expect_ok(reader, rows, s.expect)
        run.add("R8", ok, f"present sentinel {s.key}{where}" + ("" if ok else f": {why}"), status="partial")
    for s in spec.absent:
        if s.via is not None:
            continue
        try:
            rows = _sentinel_rows(reader, s.key)
        except (ServiceError, FormatError):
            continue
        run.add("R8", not rows, f"absent sentinel {s.key}{where}" + (f" found {len(rows)} row(s)" if rows else ""),
                status="partial")


def r8_matrix_sentinels(run: CheckRun) -> None:
    """R8 for a matrix: each sentinel key names axis members (``{ModelID: ACH-000001, entrez_id: "6122"}``); the
    long view reads those cells (no identifier resolution: sentinels are stored values)."""
    from .verbs.public import long_view

    spec = run.table.spec.sentinels
    if spec is None or not run.ctx.settings.readiness.sentinels:
        return
    try:
        view = long_view(run.ctx, run.ref)
    except (ServiceError, FormatError) as exc:
        run.add("R8", False, f"sentinels not checked ({exc})", level="warning")
        return

    def cells(key: Mapping[str, Any]) -> list[dict[str, Any]]:
        parts = [Eq(str(c), v) for c, v in key.items()]
        pred = parts[0] if len(parts) == 1 else And(tuple(parts))
        rk = [str(key[view.row_key[0]])] if view.row_key and view.row_key[0] in key else None
        ck = [str(key[view.col_key[0]])] if view.col_key and view.col_key[0] in key else None
        rows, _total, _eu = view.rows(pred, order=[], limit=10, row_keys=rk, col_keys=ck)
        return rows

    for s in spec.present:
        if s.via is not None:
            continue
        try:
            rows = cells(s.key)
        except (ServiceError, FormatError) as exc:
            run.add("R8", False, f"sentinel {s.key} not checked ({exc})", level="warning")
            continue
        if not rows:
            run.add("R8", False, f"present sentinel {s.key} is missing", status="partial",
                    hint="a positive control is absent: the data is incomplete or from another release")
            continue
        ok, why = _expect_ok(run.reader, rows, s.expect)     # type: ignore[arg-type]
        run.add("R8", ok, f"present sentinel {s.key}" + ("" if ok else f": {why}"), status="partial")
    for s in spec.absent:
        if s.via is not None:
            continue
        try:
            rows = cells(s.key)
        except (ServiceError, FormatError):
            continue
        run.add("R8", not rows, f"absent sentinel {s.key}" + (f" found {len(rows)} cell(s)" if rows else ""),
                status="partial")


# ---------------------------------------------------------------------------
# R9: referential samples
# ---------------------------------------------------------------------------

def _ref_target(run: CheckRun, ref: str) -> tuple[str, str] | None:
    """``(source.table, column)`` of a ``ref`` (``table.column`` or ``source.table.column``)."""
    parts = ref.split(".")
    src = run.table.descriptor.source
    if len(parts) >= 3 and parts[0] in run.ctx.catalog.sources:
        return f"{parts[0]}.{parts[1]}", ".".join(parts[2:])
    if len(parts) >= 2:
        return f"{src}.{parts[0]}", ".".join(parts[1:])
    return None


def r9_refs(run: CheckRun) -> None:
    reader = run.reader
    assert reader is not None
    max_values = None if run.depth == "deep" else SAMPLE_VALUES
    # declared stored forms ('Erdafitinib ') are matched through normalize_stored in _stored_forms
    stored = {r.partition(".")[2] for it in run.table.descriptor.id_types.values() for r in it.stored_forms
              if r.partition(".")[0] == run.table.ref.table}
    for path, col in _walk_columns(reader.spec.columns):
        ref = getattr(col, "ref", None)
        if ref is None and getattr(col, "role", None) == "hierarchy" and getattr(col, "of", None):
            of = reader.spec.columns.get(col.of) if "." not in str(col.of) else None
            ref = getattr(of, "ref", None) if of is not None else None
        if ref is None or _arrow_type_of(reader, path) is None or path in stored:
            continue
        if isinstance(ref, CompositeRef):
            _composite_ref(run, path, ref)
            continue
        target = _ref_target(run, str(ref))
        if target is None:
            continue
        sp = _schema_path(reader, path)
        try:
            snap = reader.snapshot(sp, max_values=max_values)
            values = [v for v in snap.values if v is not None]
            if max_values is not None and len(values) > max_values:
                values = random.Random(0).sample(values, max_values)
            treader = run.ctx.reader(target[0])
            found = _found_in_column(treader, target[1], values)
            if found is None:
                found = set()
                for i in range(0, len(values), 1000):
                    chunk = values[i:i + 1000]
                    for m in treader.scan(In("/" + target[1], tuple(chunk)), columns=[target[1]],
                                          attribute_unknown=False):
                        for v in _items.path_values(m.row, target[1]):
                            found.add(render_value(v))
        except (ServiceError, FormatError) as exc:
            run.add("R9", False, f"{path} -> {ref}: not checked ({exc})", level="warning", column=path)
            continue
        dangling = [v for v in values if render_value(v) not in found]
        partial = getattr(col, "integrity", "full") == "partial"
        if not dangling:
            run.add("R9", True, f"{path} -> {ref}: {len(values)} sampled value(s) resolve", column=path)
        elif partial:
            run.add("R9", False, f"{path} -> {ref}: {len(dangling)} of {len(values)} sampled value(s) dangle "
                    f"(integrity: partial), e.g. {dangling[:5]}", level="warning", column=path)
            run.confirmed[f"ref:{path}"] = {"dangling": len(dangling), "checked": len(values)}
        else:
            run.add("R9", False, f"{path} -> {ref}: {len(dangling)} of {len(values)} sampled value(s) dangle, e.g. "
                    f"{dangling[:5]}", status="key_violation", column=path,
                    hint="declare integrity: partial if dangling references are expected")
    _stored_forms(run)


def _found_in_column(treader: TableReader, column: str, values: Sequence[Any]) -> set[str] | None:
    """The rendered ``values`` that occur in the flat ``column`` of ``treader``: the column is read row group by
    row group and matched with Arrow's ``is_in``. A ``scan`` with ``In`` tests every row it keeps against every
    value of the chunk in Python: the deep R9 of the real target_prioritisation (78,726 targetIds -> target.id)
    took 51 s that way, 39 M comparisons. None when the column is nested, an item table's, cleaned (in-band
    unknowns are rewritten before matching), mixes value types or is not a string or integer column: the caller
    scans instead."""
    if treader.levels or any(ch in column for ch in ".[]") or treader._unclean(column):
        return None
    import pyarrow as pa
    import pyarrow.compute as pc

    if all(isinstance(v, str) for v in values):
        wanted, as_type, ok = pa.array(values, type=pa.large_string()), pa.large_string(), pa.types.is_string
    elif all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        wanted, as_type, ok = pa.array(values, type=pa.int64()), pa.int64(), pa.types.is_integer
    else:
        return None
    found: set[str] = set()
    try:
        for _frag, _rg, arr in treader.leaf_arrays(column):
            if isinstance(arr, pa.ChunkedArray):
                arr = arr.combine_chunks()
            if pa.types.is_dictionary(arr.type):
                arr = arr.dictionary_decode()
            if not (ok(arr.type) or pa.types.is_large_string(arr.type) and as_type == pa.large_string()):
                return None
            hits = pc.filter(arr, pc.is_in(pc.cast(arr, as_type), value_set=wanted))
            found.update(render_value(v) for v in pc.unique(hits).to_pylist() if v is not None)
    except MemoryError:
        raise                                          # out of memory, not an unanswerable check
    except (ServiceError, FormatError, ValueError, TypeError, NotImplementedError, pa.ArrowException):
        return None
    return found


def _composite_ref(run: CheckRun, path: str, ref: CompositeRef) -> None:
    reader = run.reader
    assert reader is not None
    target = run.ctx.reader(ref.table if "." in ref.table else f"{run.table.descriptor.source}.{ref.table}")
    local = list(ref.on.values())
    remote = list(ref.on.keys())
    rows = reader.sample_rows(min(SAMPLE_VALUES, 500), seed=2, columns=local)
    tuples = {tuple(_items.path_value(r, c) for c in local) for r in rows}
    tuples = {t for t in tuples if all(v is not None for v in t)}
    # only the sampled tuples are looked up (the scan is pruned on the first part and nothing else is kept):
    # holding every target tuple took 10.3 GiB for interaction_evidence -> interaction (14.5M rows)
    found: set[tuple[Any, ...]] = set()
    firsts = sorted({t[0] for t in tuples}, key=str)
    for i in range(0, len(firsts), 1000):
        pred = In("/" + remote[0], tuple(firsts[i:i + 1000]))
        for m in target.scan(pred, columns=remote, attribute_unknown=False):
            t = tuple(_items.path_value(m.row, c) for c in remote)
            if t in tuples:
                found.add(t)
        if len(found) == len(tuples):
            break
    dangling = [t for t in tuples if t not in found]
    level = "warning" if getattr(reader.column_spec(path), "integrity", "full") == "partial" else "error"
    run.add("R9", not dangling, f"{path} -> {ref.table}{list(ref.on)}: {len(dangling)} of {len(tuples)} sampled "
            f"tuple(s) dangle", status="key_violation", level=None if not dangling else level, column=path)


def _stored_forms(run: CheckRun) -> None:
    """Stored forms of an id_type in this table must normalise into its universe (names the unmatched values).
    On a column declared ``integrity: partial`` an unmatched value is the dangling reference that declaration
    expects, a warning as R9 reports it: the version-suffixed interaction endpoints of 25.09 are declared stored
    forms for the 61 human genes among them, and the other species' versioned IDs on the same column
    (ENSAMEG00000026011.1) made interaction_evidence not_ready when they were key violations."""
    t = run.table
    desc = t.descriptor
    reader = run.reader
    assert reader is not None
    specs = dict(_walk_columns(reader.spec.columns))
    for name, it in desc.id_types.items():
        for ref in it.stored_forms:
            table, _, column = ref.partition(".")
            if table != t.ref.table:
                continue
            qid = f"{desc.source}:{name}"
            try:
                snap = reader.snapshot(column, max_values=SAMPLE_VALUES)
                plugin = run.ctx.identifier(qid)
                universe = _universe_values(run, it)
            except (ServiceError, FormatError, BudgetExceeded) as exc:
                run.add("R9:stored_form", False, f"{qid} stored forms in {column}: not checked ({exc})",
                        level="warning", column=column)
                continue
            bad = []
            excluded = _excluded_values(reader, column)
            for v in snap.values:
                if v is None or render_value(v) in excluded:
                    continue                            # a declared missing value or placeholder, not an id
                n = plugin.normalize_stored(str(v))
                if not isinstance(n, Normalized) or (universe is not None and n.value not in universe):
                    bad.append(v)
            partial = getattr(specs.get(column), "integrity", "full") == "partial"
            if bad and partial:
                run.add("R9:stored_form", False, f"{qid}: {len(bad)} sampled stored value(s) in {column} outside the "
                        f"universe (integrity: partial), e.g. {[str(b) for b in bad[:5]]}", level="warning",
                        column=column)
            elif bad:
                run.add("R9:stored_form", False, f"{qid}: stored value(s) in {column} outside the universe: "
                        f"{[str(b) for b in bad[:10]]}", status="key_violation", column=column)
            else:
                run.add("R9:stored_form", True, f"{qid}: stored forms in {column} normalise into the universe",
                        column=column)


def _universe_values(run: CheckRun, it: Any) -> set[str] | None:
    u = it.universe
    if isinstance(u, list):
        u = u[0] if u else None
    if u is None:
        return None
    if isinstance(u, str):
        table, _, column = u.partition(".")
    else:
        table, column = u.table, u.keys[-1]
    if "." not in table:
        table = f"{run.table.descriptor.source}.{table}"
    snap = run.ctx.reader(table).snapshot(column)
    return {str(v) for v in snap.values if v is not None}


# ---------------------------------------------------------------------------
# R10: relations, hierarchies, edges, censoring, functional dependencies
# ---------------------------------------------------------------------------

def r10_relations(run: CheckRun) -> None:
    reader = run.reader
    assert reader is not None
    spec = reader.spec
    relations = [c for c in spec.constraints if c.op in ("equals_expr", "len_eq")]
    hierarchies = [(p, c) for p, c in _walk_columns(spec.columns) if getattr(c, "role", None) == "hierarchy"]
    events = [(p, c) for p, c in _walk_columns(spec.columns) if getattr(c, "role", None) == "flag" and
              getattr(c, "event_of", None)]
    determined = [(p, c) for p, c in _walk_columns(spec.columns) if getattr(getattr(c, "scope", None),
                                                                              "determined_by", None)]
    if not (relations or hierarchies or events or determined or spec.edge is not None):
        return
    n = _RELATION_SAMPLE if run.depth != "deep" else 10 * _RELATION_SAMPLE
    try:
        rows = reader.sample_rows(n, seed=3)
    except (ServiceError, FormatError) as exc:
        run.add("R10", False, f"relations not checked ({exc})", level="warning")
        return
    for c in relations:
        bad = 0
        example = ""
        for row in rows:
            try:
                got = _items.path_value(row, _schema_path(reader, c.column))
                want = eval_expr(str(c.expr), row)
            except (ValueError, TypeError, ZeroDivisionError):
                continue
            if got is None or want is None:
                continue
            if isinstance(got, (int, float)) and isinstance(want, (int, float)) and not isinstance(got, bool):
                ok = abs(float(got) - float(want)) <= float(c.tolerance or 0.0)
            else:
                ok = got == want
            if not ok:
                bad += 1
                example = example or f"{c.column}={got!r} but {c.expr} is {want!r}"
        name = f"constraint:{c.column} {c.op} {c.expr}"
        run.confirmed[name] = {"confirmed": bad == 0, "on_refute": c.on_refute, "checked": len(rows)}
        if not bad:
            run.add("R10", True, f"{name}: holds on {len(rows)} sampled row(s)", column=c.column)
        elif c.on_refute == "not_ready":
            run.add("R10", False, f"{name} is refuted on {bad} of {len(rows)} sampled row(s): {example}",
                    status="encoding_drift", column=c.column)
        else:
            run.add("R10", False, f"{name} is refuted on {bad} of {len(rows)} sampled row(s) ({example}); "
                    f"on_refute: {c.on_refute}", level="warning", column=c.column)
    for path, col in hierarchies:
        _hierarchy(run, path, col, rows)
    for path, col in events:
        _censoring(run, path, col, rows)
    for path, col in determined:
        _determined(run, path, col, rows)
    if spec.edge is not None and spec.edge.orientation in ("both", "canonical"):
        _edges(run, rows)


def _hierarchy(run: CheckRun, path: str, col: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    reader = run.reader
    assert reader is not None
    of = getattr(col, "of", None) or (reader.key[0] if reader.key else None)
    if of is None:
        return
    if col.reflexive is False:
        selfref = sum(1 for r in rows if _items.path_value(r, of) is not None and
                      _items.path_value(r, of) in (_items.path_value(r, path) or []))
        if selfref:
            run.add("R10:hierarchy", False, f"{path}: {selfref} sampled row(s) list their own {of} although "
                    "reflexive: false", status="encoding_drift", column=path)
        run.confirmed[f"{path}.reflexive"] = {"confirmed": selfref == 0}
    if col.relation not in ("parent", "child", "ancestor"):
        return
    graph: dict[Any, list[Any]] = {}
    try:
        for m in reader.scan(None, columns=[of, path], attribute_unknown=False):
            k = _items.path_value(m.row, of)
            vals = _items.path_value(m.row, path)
            if k is None:
                continue
            graph.setdefault(k, []).extend(v for v in (vals if isinstance(vals, list) else [vals]) if v is not None)
    except (BudgetExceeded, ServiceError, FormatError) as exc:
        run.add("R10:hierarchy", False, f"{path}: graph not read ({exc})", level="warning", column=path)
        return
    inverse = getattr(col, "inverse_of", None)
    if inverse:
        other: dict[Any, set[Any]] = {}
        for m in reader.scan(None, columns=[of, inverse], attribute_unknown=False):
            vals = _items.path_value(m.row, inverse)
            other[_items.path_value(m.row, of)] = set(vals if isinstance(vals, list) else [vals]) - {None}
        broken = sum(1 for x, ys in graph.items() for y in ys if y in other and x not in other[y])
        run.confirmed[f"{path}.inverse_of"] = {"confirmed": broken == 0}
        run.add("R10:hierarchy", broken == 0, f"{path}: {broken} link(s) without the inverse link in {inverse}",
                status="encoding_drift", column=path)
    if col.relation in ("parent", "child"):
        cycle = _find_cycle(graph)
        if cycle:
            run.add("R10:hierarchy", False, f"{path}: the {col.relation} relation has a cycle: "
                    f"{' -> '.join(map(str, cycle[:8]))}", status="encoding_drift", column=path,
                    hint="a hierarchy must be acyclic; expansion over it would not terminate")
        else:
            run.add("R10:hierarchy", True, f"{path}: {len(graph)} term(s), acyclic", column=path)
        run.confirmed[f"{path}.acyclic"] = {"confirmed": not cycle}
    closure_of = getattr(col, "closure_of", None)
    if col.relation == "ancestor" and closure_of:
        parents: dict[Any, list[Any]] = {}
        for m in reader.scan(None, columns=[of, closure_of], attribute_unknown=False):
            k = _items.path_value(m.row, of)
            vals = _items.path_value(m.row, closure_of)
            parents[k] = [v for v in (vals if isinstance(vals, list) else [vals]) if v is not None]
        bad = 0
        for k in list(graph)[:_RELATION_SAMPLE]:
            if set(_closure(parents, k)) != set(graph.get(k, [])):
                bad += 1
        run.confirmed[f"{path}.closure"] = {"confirmed": bad == 0}
        run.add("R10:hierarchy", bad == 0, f"{path}: {'equals' if not bad else f'differs on {bad} term(s) from'} the "
                f"transitive closure of {closure_of}", status="encoding_drift", column=path)


def _closure(parents: Mapping[Any, list[Any]], start: Any) -> list[Any]:
    out: list[Any] = []
    seen = {start}
    stack = list(parents.get(start, []))
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
        stack.extend(parents.get(x, []))
    return out


def _find_cycle(graph: Mapping[Any, list[Any]]) -> list[Any]:
    """One cycle of a directed graph (iterative DFS), [] when acyclic."""
    color: dict[Any, int] = {}
    for start in graph:
        if color.get(start):
            continue
        stack: list[tuple[Any, int]] = [(start, 0)]
        path: list[Any] = []
        while stack:
            node, i = stack.pop()
            if i == 0:
                color[node] = 1
                path.append(node)
            nbrs = graph.get(node, [])
            if i < len(nbrs):
                stack.append((node, i + 1))
                nxt = nbrs[i]
                if color.get(nxt) == 1:
                    return path[path.index(nxt):] + [nxt]
                if not color.get(nxt):
                    stack.append((nxt, 0))
            else:
                color[node] = 2
                path.pop()
    return []


def _censoring(run: CheckRun, path: str, col: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    time_col = str(col.event_of)
    no_status = sum(1 for r in rows if _items.path_value(r, time_col) is not None and _items.path_value(r, path) is None)
    no_time = sum(1 for r in rows if _items.path_value(r, path) is not None and _items.path_value(r, time_col) is None)
    nonpos = sum(1 for r in rows if isinstance(_items.path_value(r, time_col), (int, float)) and
                 _items.path_value(r, time_col) <= 0)
    run.confirmed[f"{path}.censoring"] = {"time_without_status": no_status, "status_without_time": no_time,
                                          "non_positive_time": nonpos, "checked": len(rows)}
    ok = not (no_status or no_time or nonpos)
    run.add("R10:censoring", ok, f"{path}/{time_col}: {no_status} time(s) without status, {no_time} status without "
            f"time, {nonpos} non-positive time(s) in {len(rows)} sampled row(s)", level=None if ok else "warning",
            column=path)


def _determined(run: CheckRun, path: str, col: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    by = list(col.scope.determined_by)
    seen: dict[str, str] = {}
    bad = 0
    for r in rows:
        k = canonical([_items.path_value(r, b) for b in by])
        v = render_value(_items.path_value(r, path))
        if seen.setdefault(k, v) != v:
            bad += 1
    run.confirmed[f"{path}.determined_by"] = {"confirmed": bad == 0}
    run.add("R10:determined_by", bad == 0, f"{path} is {'' if not bad else 'not '}determined by {by} on the sample",
            level=None if not bad else "warning", column=path)


def _edges(run: CheckRun, rows: Sequence[Mapping[str, Any]]) -> None:
    reader = run.reader
    assert reader is not None
    edge = reader.spec.edge
    assert edge is not None
    pairs = {(_items.path_value(r, edge.a), _items.path_value(r, edge.b)) for r in rows}
    pairs = {p for p in pairs if p[0] is not None and p[1] is not None}
    if edge.orientation == "canonical":
        bad = sum(1 for a, b in pairs if str(a) > str(b))
        detail = f"{bad} sampled edge(s) not in canonical (a <= b) order"
    else:
        # the sample holds whole row groups, so the reverse of a sampled edge is usually elsewhere: look the
        # reverses up in the table (one scan of the two endpoint columns, restricted to the sampled endpoints)
        missing = [(a, b) for a, b in pairs if a != b and (b, a) not in pairs]
        found = _reverse_edges(reader, edge, missing) if missing else set()
        if found is not None:
            missing = [(a, b) for a, b in missing if (b, a) not in found]
        bad = len(missing)
        detail = (f"{bad} of {len(pairs)} sampled edge(s) without their reverse" +
                  ("" if found is not None else " in the sample (the table could not be scanned for them)") +
                  (f", e.g. {missing[0]}" if missing else ""))
    run.confirmed["edge.orientation"] = {"confirmed": bad == 0, "orientation": edge.orientation}
    run.add("R10:edges", bad == 0, detail, level=None if not bad else "warning")


def _reverse_edges(reader: TableReader, edge: Any, pairs: Sequence[tuple[Any, Any]]) -> set[tuple[Any, Any]] | None:
    """The ``(a, b)`` edges of the table whose ``a`` is the ``b`` of a pair and whose ``b`` is its ``a``: the two
    endpoint columns read row group by row group and matched with Arrow's ``is_in`` (14.5 M interaction rows
    would take minutes as Python rows). None when the endpoints are nested or the columns cannot be read."""
    if any(ch in str(edge.a) + str(edge.b) for ch in ".[]"):
        return None
    import pyarrow as pa
    import pyarrow.compute as pc

    sep = "\x1f"
    wanted = pa.array(sorted({f"{b}{sep}{a}" for a, b in pairs}), type=pa.large_string())

    def text(arr: Any) -> Any:
        return arr if pa.types.is_large_string(arr.type) else pc.cast(arr, pa.large_string())

    out: set[tuple[Any, Any]] = set()
    try:
        for (fa, ra, xa), (fb, rb, xb) in zip(reader.leaf_arrays(edge.a), reader.leaf_arrays(edge.b), strict=True):
            if (fa.uri, ra) != (fb.uri, rb) or len(xa) != len(xb):
                return None
            # only the exact reverses: "a<US>b" of each row against "b<US>a" of each pair
            joined = pc.binary_join_element_wise(text(xa), text(xb), pa.scalar(sep, pa.large_string()))
            mask = pc.is_in(joined, value_set=wanted)
            if pc.any(mask).as_py():
                out.update(zip(pc.filter(xa, mask).to_pylist(), pc.filter(xb, mask).to_pylist()))
    except (ServiceError, FormatError, ValueError, TypeError, NotImplementedError):
        return None
    return out


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def check_table(ctx: ServiceContext, ref: str, depth: str = "standard", *,
                sample_rows: int = SAMPLE_ROWS) -> TableCheckModel:
    """Every check of ``depth`` for one table (an item table runs its parent's file checks plus its own key
    and sentinel checks)."""
    t = ctx.table(ref)
    run = CheckRun(ctx, str(t.ref), depth, sample_rows=sample_rows)
    if t.is_item_table:
        parent = check_table(ctx, str(t.physical), "shallow" if depth == "shallow" else "standard_files",
                             sample_rows=sample_rows)
        run.checks.extend(parent.checks)
        run.statuses.append(parent.status)
        run.partitions.update(parent.partitions)
        run.storage_types = dict(parent.storage_types)
        if depth == "shallow" or parent.status in ("missing", "plugin_unavailable"):
            return run.model()
        try:
            run.reader = ctx.reader(ref)
            run.key_check = r5_keys(run)
            r8_sentinels(run)
        except (ServiceError, FormatError) as exc:
            run.add("R5", False, f"item table not checked ({exc})", status="partial")
        return run.model()
    try:
        if not r1_r2(run):
            return run.model()
    except ServiceError as exc:
        run.add("R1", False, str(exc), status="plugin_unavailable")
        return run.model()
    if depth == "shallow":
        try:
            run.reader = ctx.reader(ref)
        except ServiceError:
            pass
        return run.model()
    try:
        run.reader = ctx.reader(ref)
        r3_footers(run)
        if any(c.name == "R3" and not c.ok for c in run.checks) and \
                not any(c.name == "R3" and c.partition for c in run.checks if not c.ok):
            return run.model()
        run.stats, run.rows, _ = aggregate_stats(run.reader)
        r2_manifest_checks(run)
        if is_matrix(run.reader):
            r4_r5_matrix(run)                       # the long view's keys are axis members, not columns
            if depth != "standard_files" and not any(c.name in ("R4", "R5") and not c.ok and c.level == "error"
                                                     for c in run.checks):
                r8_matrix_sentinels(run)            # read as long cells (axis keys are not columns)
            return run.model()
        r4_types(run)
        if depth == "standard_files":
            return run.model()
        r4b_universe(run)
        run.key_check = r5_keys(run)
        r5b_item_keys(run)
        r6_facts(run)
        r7_vocab(run)
        r8_sentinels(run)
        r9_refs(run)
        r10_relations(run)
        access_indexes(run)
        for item_ref in ctx.item_tables_of(str(t.physical)):
            sub = CheckRun(ctx, item_ref, depth, reader=ctx.reader(item_ref), sample_rows=sample_rows, rows=None)
            sub.key_check = r5_keys(sub)
            r8_sentinels(sub)
            run.checks.extend(sub.checks)
            status = worst(sub.statuses)
            run.item_tables[item_ref.split(".", 1)[1]] = status
    except FormatError as exc:
        run.add("R3", False, str(exc), status="partial", partition=partition_label(exc.partition) or None)
    except TableUnavailable as exc:
        run.add("R1", False, str(exc), status="missing")
    return run.model()


def check_tables(ctx: ServiceContext, refs: Sequence[str] = (), depth: str = "standard"
                 ) -> tuple[dict[str, TableCheckModel], dict[str, str]]:
    """``({table: TableCheckModel}, {table: error})`` for ``refs`` (default: every table of every source)."""
    out: dict[str, TableCheckModel] = {}
    errors: dict[str, str] = {}
    for ref in refs or ctx.table_refs():
        try:
            out[ref] = check_table(ctx, ref, depth)
        except Exception as exc:  # noqa: BLE001 - one broken table never hides the others' findings
            errors[ref] = f"{type(exc).__name__}: {exc}"
    return out, errors


def hash_randomization() -> int:
    return int(sys.flags.hash_randomization)
