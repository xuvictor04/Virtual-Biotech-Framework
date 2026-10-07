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
from typing import Any, Iterable, Mapping, Sequence

from ..descriptor.columns import CompositeRef, is_container
from ..ipc import CheckItemModel, KeyCheckModel, TableCheckModel
from ..plugins.base import ColumnStats, FormatError, Normalized, ValueSnapshot
from ..plugins.layouts import PROBE_STATUS, partition_label
from ..predicate import And, Eq, In, is_null
from ..roles import arrow_compatible, parse_path
from ..rowkey import canonical, render_value
from . import ServiceContext, ServiceError, json_path, layout_spec
from . import items as _items
from .reader import BudgetExceeded, TableReader, TableUnavailable, logical_leaves

__all__ = [
    "STATUS_ORDER", "worst", "CheckRun", "check_table", "check_tables", "aggregate_stats", "eval_expr",
    "SAMPLE_ROWS", "SAMPLE_VALUES",
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
        if self.reader is not None:
            try:
                fp, sig = self.reader.fingerprint(), self.reader._sig
            except (ServiceError, OSError):
                pass
        return TableCheckModel(status=status, columns=self.columns, containers=self.containers,
                               partitions=self.partitions, item_tables=self.item_tables, checks=self.checks,
                               fingerprint=fp, signature=sig, confirmed=_jsonable(self.confirmed), vocab=self.vocab,
                               key_check=self.key_check)


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
        for what, path in (spec.checks or {}).items():
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
    for path, col in _walk_columns(spec.columns):
        role = getattr(col, "role", None)
        t = _arrow_type_of(reader, path)
        optional = bool(getattr(col, "optional", False))
        if t is None:
            if optional:
                run.add("R4", False, f"optional column {path} is absent from the data", level="warning",
                        column=path)
                run.columns.setdefault(path, "missing")
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
    if strict:
        extra = [n for n in reader.top_columns() if n not in declared_top]
        if extra:
            run.add("R4:undeclared", False, f"undeclared physical columns under strict: {', '.join(extra[:20])}",
                    status="schema_drift")


def _universe_columns(run: CheckRun) -> list[tuple[str, str, Any]]:
    """``(qualified id_type, physical column, IdTypeSpec)`` whose identity universe is a column of this table."""
    t = run.table
    desc = t.descriptor
    out = []
    for name, it in desc.id_types.items():
        u = it.universe
        refs: list[Any] = u if isinstance(u, list) else [u]
        for r in refs:
            if r is None:
                continue
            if isinstance(r, str):
                table, _, column = r.partition(".")
                cols = [column]
            else:
                table, cols = r.table, list(r.keys)
            if table.count(".") == 1:
                src, table = table.split(".")
                if src != desc.source:
                    continue
            if table == t.ref.table:
                for c in cols[-1:]:
                    out.append((f"{desc.source}:{name}", c, it))
    return out


def r4b_universe(run: CheckRun) -> None:
    reader = run.reader
    assert reader is not None
    for qid, column, it in _universe_columns(run):
        try:
            snap = reader.snapshot(column, max_values=SAMPLE_VALUES)
        except (ServiceError, BudgetExceeded) as exc:
            run.add("R4b", False, f"{qid}: universe sample not read ({exc})", level="warning", column=column)
            continue
        sample = [str(v) for v in snap.values if v is not None][:SAMPLE_VALUES]
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
        if bad:
            total = sum(len(v) for v in bad.values())
            share = total / max(len(sample), 1)
            names = ", ".join(f"{p} ({len(v)})" for p, v in sorted(bad.items(), key=lambda kv: -len(kv[1]))[:10])
            run.add("R4b", False, f"{qid}: {total} of {len(sample)} sampled universe keys ({share:.1%}) are not in "
                    f"the canonical form of {it.plugin}; failing prefixes: {names}", status="encoding_drift",
                    column=column, hint="extend the plugin options (e.g. prefixes) or use prefixes: from_universe")
        else:
            run.add("R4b", True, f"{qid}: {len(sample)} universe keys canonical", column=column)


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
    """R5 (non-nullable key parts without nulls) and R5b (uniqueness) for a table or item table."""
    reader = run.ctx.reader(ref or run.ref)
    t = reader.table
    key = list(reader.key)
    spec_key = t.spec.key
    nullable = set(t.nullable_key)
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
    flat = [k for k in key if not parse_path(k).crosses_list and not k.endswith("#")]
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
    fraction = 1.0 if method == "full" or rows_hint <= run.sample_rows else run.sample_rows / rows_hint
    buckets = 1 << 16
    keep_below = max(1, int(buckets * fraction))
    seen: set[str] = set()
    spill: _Spill | None = None
    dups = 0
    examples: list[str] = []
    scanned = 0
    types = reader.storage_types(key)
    try:
        for m in reader.scan(None, columns=[], attribute_unknown=False):
            scanned += 1
            for k, v in zip(key, m.key):
                if v is None and k in nulls and (t.is_item_table or k not in flat):
                    nulls[k] += 1
            ck = canonical(list(m.key), types)
            if fraction < 1.0:
                block = canonical([m.key[i] for i in prefix_idx]) if prefix_idx else ck
                if _hash(block) % buckets >= keep_below:
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
            if len(seen) > 2_000_000:
                spill = _Spill(run.ctx.settings.cache_dir)
                for s in seen:
                    spill.add(s)
                seen = set()
    except (BudgetExceeded, TableUnavailable, FormatError) as exc:
        model.ok = None
        model.detail = f"key check not completed: {exc}"
        run.add("R5b", False, model.detail, level="warning")
        return model
    if spill is not None:
        d, ex = spill.duplicates()
        dups += d
        examples.extend(ex[: 5 - len(examples)])
    model.duplicates = dups
    model.null_counts = {k: v for k, v in nulls.items() if v}
    null_bad = {k: v for k, v in nulls.items() if v and k not in nullable}
    where = f" ({t.ref})" if t.is_item_table else ""
    if null_bad:
        run.add("R5", False, f"non-nullable key parts with nulls{where}: " +
                ", ".join(f"{k}={v}" for k, v in null_bad.items()), status="key_violation",
                column=next(iter(null_bad)))
    else:
        run.add("R5", True, f"no nulls in non-nullable key parts{where} ({scanned} keys)")
    sampled = f"sampled {fraction:.1%} of prefix blocks" if fraction < 1.0 else "every key"
    if dups:
        run.add("R5b", False, f"{dups} duplicate key(s){where} ({sampled}): {', '.join(examples)}",
                status="key_violation", hint="the declared key is not unique; every count over it would be wrong")
    else:
        run.add("R5b", True, f"key unique{where} ({method}: {sampled})")
    model.ok = not dups and not null_bad
    model.detail = f"{scanned} keys scanned; {sampled}"
    return model


class _Spill:
    """Hash-partitioned temporary files for a bounded-memory full uniqueness pass."""

    def __init__(self, cache_dir: Any, parts: int = 64) -> None:
        os.makedirs(str(cache_dir), exist_ok=True)
        self.dir = tempfile.mkdtemp(prefix="keycheck.", dir=str(cache_dir))
        self.parts = parts
        self.files = [open(os.path.join(self.dir, f"{i:02d}"), "w", encoding="utf-8") for i in range(parts)]

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
            for i in range(self.parts):
                try:
                    os.unlink(os.path.join(self.dir, f"{i:02d}"))
                except OSError:
                    pass
            try:
                os.rmdir(self.dir)
            except OSError:
                pass
        return dups, examples


def r5b_item_keys(run: CheckRun) -> None:
    """Item keys of nested containers (not declared as item tables) are unique within each row (sampled)."""
    reader = run.reader
    assert reader is not None
    containers = [(p, c) for p, c in _walk_columns(reader.spec.columns) if is_container(c) and c.item_key is not None
                  and c.item_key.check != "none"]
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
    counts = _items.ContainerCounts()
    container = schema_path if schema_path.endswith("]") else schema_path + "[]"
    try:
        lvls = _items.levels(container)
        leaf = reader.leaf(container)
        for frag, rgs in reader.plan(None, [leaf] if leaf else [], use_sidecars=False)[0]:
            info = reader.footer(frag)
            for rg in (rgs if rgs is not None else [None]):
                for row in reader._read(frag, rg, [leaf] if leaf else [], info):
                    for _ in _items.explode(row, lvls, counts):
                        pass
    except (ServiceError, FormatError, ValueError) as exc:
        run.add("R6:containers", False, f"{path}: null vs empty counts not read ({exc})", level="warning",
                container=path)
        return
    run.confirmed[f"{path}[]"] = {"null": counts.null, "empty": counts.empty, "nonempty": counts.nonempty,
                                  "items": counts.items, "null_items": counts.null_items}
    run.add("R6:containers", True, f"{path}: {counts.null} null, {counts.empty} empty, {counts.nonempty} non-empty "
            f"list(s); {counts.null_items} null item(s)", container=path)
    run.containers.setdefault(path, "ready")


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
        odd = [v for v in values if isinstance(v, str) and v.strip() in _SUSPICIOUS]
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
    for path, col in _walk_columns(reader.spec.columns):
        ref = getattr(col, "ref", None)
        if ref is None and getattr(col, "role", None) == "hierarchy" and getattr(col, "of", None):
            of = reader.spec.columns.get(col.of) if "." not in str(col.of) else None
            ref = getattr(of, "ref", None) if of is not None else None
        if ref is None or _arrow_type_of(reader, path) is None:
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
            found: set[str] = set()
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


def _composite_ref(run: CheckRun, path: str, ref: CompositeRef) -> None:
    reader = run.reader
    assert reader is not None
    target = run.ctx.reader(ref.table if "." in ref.table else f"{run.table.descriptor.source}.{ref.table}")
    local = list(ref.on.values())
    remote = list(ref.on.keys())
    rows = reader.sample_rows(min(SAMPLE_VALUES, 500), seed=2, columns=local)
    tuples = {tuple(_items.path_value(r, c) for c in local) for r in rows}
    tuples = {t for t in tuples if all(v is not None for v in t)}
    found = set()
    for m in target.scan(None, columns=remote, attribute_unknown=False):
        found.add(tuple(_items.path_value(m.row, c) for c in remote))
    dangling = [t for t in tuples if t not in found]
    level = "warning" if getattr(reader.column_spec(path), "integrity", "full") == "partial" else "error"
    run.add("R9", not dangling, f"{path} -> {ref.table}{list(ref.on)}: {len(dangling)} of {len(tuples)} sampled "
            f"tuple(s) dangle", status="key_violation", level=None if not dangling else level, column=path)


def _stored_forms(run: CheckRun) -> None:
    """Stored forms of an id_type in this table must normalise into its universe (names the unmatched values)."""
    t = run.table
    desc = t.descriptor
    reader = run.reader
    assert reader is not None
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
            for v in snap.values:
                if v is None:
                    continue
                n = plugin.normalize_stored(str(v))
                if not isinstance(n, Normalized) or (universe is not None and n.value not in universe):
                    bad.append(v)
            if bad:
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
        bad = sum(1 for a, b in pairs if a != b and (b, a) not in pairs)
        detail = f"{bad} sampled edge(s) without their reverse"
    run.confirmed["edge.orientation"] = {"confirmed": bad == 0, "orientation": edge.orientation}
    run.add("R10:edges", bad == 0, detail, level=None if not bad else "warning")


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
