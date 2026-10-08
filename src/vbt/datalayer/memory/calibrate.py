"""Memory calibration: sample-and-scale and measured feedback (§10.4, §14, phase 4 F19). No pyarrow at import.

The phase-1 estimator (:mod:`.estimate`) multiplies footer statistics by **seed** factors. Phase 4
replaces them in two tiers, each stored next to the data it describes:

1. **Sample** (``vbt ds calibrate``; runs in the data child): one to three row groups of the table
   (first, middle, last) are read as Arrow, at most :data:`MAX_SAMPLE_ROWS` evenly spaced rows of them are
   kept (a row group can be a whole shard: 1,964,234 rows of the 25.09 study table, whose measurement ran
   out of memory), then converted to pandas the way upstream loads them.
   Per top-level column the Arrow ``nbytes`` and the pandas size are measured (``memory_usage(deep=True)``
   for flat columns; object columns are walked recursively, because ``deep=True`` counts a list or a
   dict shallowly, which is exactly the nested-item cost that made footer bytes understate the peak
   about 9x). The upstream loader (``to_table().to_pandas()``) holds the Arrow table and the DataFrame
   together, so the estimate's bytes per row are both (``peak_bytes_per_row``); ``bytes_per_row`` stays
   the pandas part. The sample is scaled by ``rows / rows_sampled`` and the per-kind factors are fitted
   against the seed model: expansion and string overhead from flat and string columns, the nested
   item and struct item overheads from the residual of nested columns. The result is written to
   ``<cache>/<source>/<fingerprint>/calibration.json`` (:func:`calibration_path`) and the estimator
   uses its measured bytes per row for that fingerprint.
2. **Measured** (after real loads): the reaper's status file gives the server's peak RSS after a
   cold load; ``(peak - baseline) / estimate`` for the tables of that load is a feedback factor kept
   in ``<cache>/memory_feedback.json`` (:func:`record_feedback`), applied on top of either tier.

``MemoryEstimator.tier(stats)`` reports which tier an estimate came from (``measured``, ``sample``,
``seed``); provenance and ``vbt ds status`` print it.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "CALIBRATION_FILE", "FEEDBACK_FILE", "MAX_ROW_GROUPS", "MAX_SAMPLE_ROWS", "calibration_path", "feedback_path",
    "pick_samples", "deep_bytes", "measure_table", "fit", "calibrate_fragments", "calibrate_table", "write_calibration",
    "load_calibrations", "load_feedback", "record_feedback", "feedback_from_status", "stats_of_fragments",
    "factors_summary", "calibrate",
]

CALIBRATION_FILE = "calibration.json"
FEEDBACK_FILE = "memory_feedback.json"
MAX_ROW_GROUPS = 3
#: Rows measured at most, spread over the sampled row groups (each Open Targets 25.09 shard is one row group).
MAX_SAMPLE_ROWS = 30000
#: The seed model's pandas bytes a sample may hold, and the rows it keeps whatever the estimate.
SAMPLE_BUDGET_BYTES = 256 * 1024 * 1024
MIN_SAMPLE_ROWS = 200
MB = 1024 * 1024


# --------------------------------------------------------------------------- locations


def _safe(fp: str) -> str:
    return str(fp).replace(":", "_").replace(os.sep, "_")


def calibration_path(cache_dir: str | Path, source: str, fingerprint: str) -> Path:
    """``<cache>/<source>/<fingerprint>/calibration.json`` (the sidecar layout of §11.6)."""
    return Path(cache_dir) / source / _safe(fingerprint) / CALIBRATION_FILE


def feedback_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / FEEDBACK_FILE


def _write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1, sort_keys=True, default=str)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


# --------------------------------------------------------------------------- measuring (data child)


def pick_samples(n: int, k: int = MAX_ROW_GROUPS) -> list[int]:
    """Up to ``k`` indices out of ``n``: the first, the middle and the last (spread when ``k`` differs)."""
    if n <= 0:
        return []
    k = max(1, min(int(k), n))
    if k == 1:
        return [0]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def deep_bytes(obj: Any, _seen: set[int] | None = None) -> int:
    """Recursive ``sys.getsizeof`` of a Python object graph (lists, tuples, dicts, numpy arrays of objects),
    counting each object once. pandas ``memory_usage(deep=True)`` counts containers shallowly."""
    seen = _seen if _seen is not None else set()
    stack = [obj]
    total = 0
    while stack:
        o = stack.pop()
        oid = id(o)
        if oid in seen:
            continue
        seen.add(oid)
        nbytes = getattr(o, "nbytes", None)
        dtype = getattr(o, "dtype", None)
        if nbytes is not None and dtype is not None:           # numpy array
            total += int(nbytes) + 112
            if getattr(dtype, "kind", "") == "O":
                stack.extend(o.ravel().tolist() if hasattr(o, "ravel") else list(o))
            continue
        try:
            total += sys.getsizeof(o)
        except TypeError:
            continue
        if isinstance(o, Mapping):
            stack.extend(o.keys())
            stack.extend(o.values())
        elif isinstance(o, (list, tuple, set, frozenset)):
            stack.extend(o)
    return total


def measure_table(table: Any) -> dict[str, dict[str, int]]:
    """``{column: {arrow_bytes, pandas_bytes}}`` of an Arrow table converted the way upstream loads it."""
    out: dict[str, dict[str, int]] = {}
    df = table.to_pandas()
    usage = df.memory_usage(deep=True, index=False)
    for i, name in enumerate(table.column_names):
        col = table.column(i)
        series = df.iloc[:, i]
        if series.dtype == object:
            seen: set[int] = set()
            pandas_bytes = int(series.values.nbytes) + sum(deep_bytes(v, seen) for v in series.values)
        else:
            pandas_bytes = int(usage.iloc[i])
        out[str(name)] = {"arrow_bytes": int(col.nbytes), "pandas_bytes": pandas_bytes}
    return out


def stats_of_fragments(fmt: Any, frags: Sequence[Any]) -> dict[str, Any]:
    """Table statistics (the ``_stats`` JSON shape, §6.4 leaf paths) summed over ``frags``."""
    from ..service.checks import logical_leaves

    schema = fmt.logical_schema(frags[0])
    to_path: dict[str, str] = {}
    for path, _t in logical_leaves(schema):
        try:
            to_path.setdefault(fmt.leaf_path(path, schema), path)
        except (ValueError, KeyError):
            continue
    columns: dict[str, dict[str, Any]] = {}
    rows: int | None = 0
    for frag in frags:
        st = fmt.stats(frag)
        rows = None if rows is None or st.rows is None else rows + int(st.rows)
        for leaf, cs in st.columns.items():
            a = columns.setdefault(to_path.get(leaf, leaf), {"uncompressed_bytes": 0, "num_values": 0,
                                                              "max_rep_level": 0, "kind": cs.kind,
                                                              "storage_type": cs.storage_type})
            a["uncompressed_bytes"] += int(cs.uncompressed_bytes or 0)
            a["num_values"] = None if a["num_values"] is None or cs.num_values is None else \
                a["num_values"] + int(cs.num_values)
            a["max_rep_level"] = max(a["max_rep_level"], int(cs.max_rep_level or 0))
    return {"rows": rows, "columns": columns, "fragments": len(frags)}


def _read_sample(fmt: Any, frags: Sequence[Any], partitions: Mapping[str, str], columns: Sequence[str] | None,
                 footer: Any, k: int, max_rows: int = MAX_SAMPLE_ROWS) -> tuple[Any, list[dict[str, Any]]]:
    import pyarrow as pa

    units: list[tuple[Any, int | None]] = []
    for frag in frags:
        info = footer(frag) if footer is not None else None
        if info is not None and info.row_groups:
            units.extend((frag, i) for i in range(len(info.row_groups)))
        else:
            units.append((frag, None))
    picked = [units[i] for i in pick_samples(len(units), k)]
    per_unit = max(1, int(max_rows) // max(1, len(picked)))
    tables = []
    where = []
    for frag, rg in picked:
        kwargs: dict[str, Any] = {"columns": list(columns) if columns else None, "predicate": None,
                                  "partitions": dict(partitions)}
        if rg is not None:
            kwargs["row_groups"] = {frag.uri: [rg]}
        batches = list(fmt.scan([frag], **kwargs))
        entry: dict[str, Any] = {"fragment": frag.uri, "row_group": rg}
        if batches:
            tbl = pa.Table.from_batches(batches)
            if tbl.num_rows > per_unit:
                # evenly spaced rows: a whole 25.09 study shard (one row group of 1,964,234 rows) did not fit
                step = tbl.num_rows / per_unit
                entry["rows"] = [tbl.num_rows, per_unit]
                tbl = tbl.take(pa.array([int(i * step) for i in range(per_unit)], pa.int64()))
            tables.append(tbl)
        where.append(entry)
    if not tables:
        return None, where
    return (pa.concat_tables(tables, promote_options="permissive") if len(tables) > 1 else tables[0]), where


def calibrate_fragments(fmt: Any, frags: Sequence[Any], *, table_stats: Mapping[str, Any] | None = None,
                        partitions: Mapping[str, str] | None = None, columns: Sequence[str] | None = None,
                        row_groups: int = MAX_ROW_GROUPS, seed: Any = None, table: str | None = None,
                        fingerprint: str | None = None, max_rows: int = MAX_SAMPLE_ROWS) -> dict[str, Any]:
    """Sample up to ``row_groups`` row groups of ``frags``, measure them and :func:`fit` the seed model.
    ``table_stats`` defaults to :func:`stats_of_fragments`. Runs in the data child (pyarrow, pandas).

    At most ``max_rows`` rows are measured, and no more than the seed model expects to fit in
    :data:`SAMPLE_BUDGET_BYTES` as pandas (at least :data:`MIN_SAMPLE_ROWS`): 10,000 rows of each of three
    25.09 expression shards (genes with nested tissues and cell types) ran the data child out of memory."""
    from ..service.sidecar import footer_reader

    stats = dict(table_stats) if table_stats is not None else stats_of_fragments(fmt, frags)
    rows = stats.get("rows")
    if seed is not None and rows:
        per_row = seed.peak_upstream(stats) / float(rows)
        if per_row > 0:
            max_rows = min(int(max_rows), max(MIN_SAMPLE_ROWS, int(SAMPLE_BUDGET_BYTES // per_row)))
    sample, where = _read_sample(fmt, frags, partitions or {}, columns, footer_reader(fmt), row_groups, max_rows)
    if sample is None or sample.num_rows == 0:
        raise ValueError("calibration needs at least one non-empty row group")
    measured = measure_table(sample)
    cal = fit(stats, measured, sample.num_rows, seed=seed)
    cal.update({"table": table, "fingerprint": fingerprint, "sampled": where,
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return cal


def calibrate_table(ctx: Any, ref: str, *, row_groups: int = MAX_ROW_GROUPS, write: bool = True) -> dict[str, Any]:
    """Calibrate one catalog table in the data child and (by default) write its ``calibration.json``."""
    from ..service.checks import aggregate_stats
    from .estimate import MemoryEstimator

    reader = ctx.reader(ref)
    cols, rows, _size = aggregate_stats(reader)
    stats = {"rows": rows, "columns": {p: {"uncompressed_bytes": cs.uncompressed_bytes, "num_values": cs.num_values,
                                           "max_rep_level": cs.max_rep_level, "kind": cs.kind,
                                           "storage_type": cs.storage_type} for p, cs in cols.items()}}
    fp = reader.fingerprint()
    cal = calibrate_fragments(reader.fmt, reader.fragments(), table_stats=stats, partitions=reader.partitions,
                              row_groups=row_groups, seed=MemoryEstimator.from_settings(ctx.settings),
                              table=str(reader.table.physical), fingerprint=fp)
    if write:
        path = write_calibration(ctx.settings.cache_dir, reader.table.physical.source, fp, cal)
        cal["path"] = str(path)
    return cal


def calibrate(ctx: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    """``{table, row_groups?}`` -> the written calibration (``vbt ds calibrate`` runs it in the data child)."""
    return calibrate_table(ctx, str(payload["table"]), row_groups=int(payload.get("row_groups") or MAX_ROW_GROUPS))


# --------------------------------------------------------------------------- fitting (pure)


def _column_of(path: str) -> str:
    from ..roles import PathError, parse_path

    try:
        return parse_path(path).head
    except PathError:
        return path.split(".", 1)[0].split("[", 1)[0]


def fit(table_stats: Mapping[str, Any], measured: Mapping[str, Mapping[str, int]], rows_sampled: int, *,
        seed: Any = None) -> dict[str, Any]:
    """Fit the per-kind factors of ``seed`` (a :class:`~.estimate.MemoryEstimator`) to a measured sample.

    ``measured`` is :func:`measure_table`'s ``{column: {arrow_bytes, pandas_bytes}}`` of ``rows_sampled``
    rows; ``table_stats`` the whole table's footer statistics. Returns the calibration record:
    ``bytes_per_row`` (pandas, all measured columns), ``peak_bytes_per_row`` (pandas plus the Arrow table
    it is converted from: the upstream loader holds both at its peak, and pandas alone was 0.53-0.89 of the
    measured peak of the Open Targets 25.09 loads over 50 MB, both together 0.98-1.47), per-column measured
    and seed bytes (scaled to the table), the fitted ``expansion``, ``object_overhead_bytes`` and ``decode``
    factors, and ``kinds`` (the column kind each factor came from)."""
    from .estimate import MemoryEstimator, leaves_of

    est = seed if seed is not None else MemoryEstimator()
    rows_total = table_stats.get("rows") if isinstance(table_stats, Mapping) else getattr(table_stats, "rows", None)
    rows_total = int(rows_total) if rows_total else int(rows_sampled)
    scale = rows_total / float(max(1, rows_sampled))
    leaves = leaves_of(table_stats)
    by_col: dict[str, list[Any]] = {}
    for leaf in leaves:
        by_col.setdefault(_column_of(leaf.path), []).append(leaf)

    columns: dict[str, dict[str, Any]] = {}
    sums: dict[str, dict[str, float]] = {}
    residual: dict[str, float] = {"string": 0.0, "nested": 0.0}
    weights: dict[str, float] = {"string": 0.0, "nested": 0.0}
    for col, m in sorted(measured.items()):
        col_leaves = by_col.get(col, [])
        sub = {"rows": rows_total, "columns": {lf.path: {"uncompressed_bytes": lf.uncompressed_bytes,
                                                         "num_values": lf.num_values,
                                                         "max_rep_level": lf.max_rep_level, "kind": lf.kind,
                                                         "storage_type": lf.storage_type} for lf in col_leaves}}
        bytes_term = est.bytes_term(sub)
        objects_term = est.objects_term(sub)
        pandas_full = float(m.get("pandas_bytes", 0)) * scale
        arrow_full = float(m.get("arrow_bytes", 0)) * scale
        nested = any(lf.depth > 0 or len(lf.segments) > 1 for lf in col_leaves)
        string = not nested and bool(col_leaves) and all(lf.is_string for lf in col_leaves)
        kind = "nested" if nested else ("string" if string else "flat")
        footer = float(sum(lf.uncompressed_bytes for lf in col_leaves))
        columns[col] = {"kind": kind, "pandas_bytes": int(round(pandas_full)), "arrow_bytes": int(round(arrow_full)),
                        "seed_bytes": int(round(bytes_term + objects_term)), "footer_bytes": int(footer)}
        s = sums.setdefault(kind, {"pandas": 0.0, "seed": 0.0, "arrow": 0.0, "footer": 0.0, "objects": 0.0})
        s["pandas"] += pandas_full
        s["seed"] += bytes_term + objects_term
        s["arrow"] += arrow_full
        s["footer"] += footer
        s["objects"] += objects_term
        if kind in residual:
            residual[kind] += max(0.0, pandas_full - bytes_term)
            weights[kind] += objects_term

    def ratio(kind: str, a: str, b: str) -> float | None:
        s = sums.get(kind)
        if not s or s[b] <= 0:
            return None
        return s[a] / s[b]

    expansion = dict(est.expansion)
    flat_sums = sums.get("flat")
    if flat_sums and flat_sums["footer"] > 0:
        # what the per-value ``flat_value`` term does not explain is left to the bytes (all of it without one)
        expansion["flat"] = round(max(0.0, flat_sums["pandas"] - flat_sums["objects"]) / flat_sums["footer"], 4)
    overhead = dict(est.object_overhead)
    for kind, keys in (("string", ("string",)), ("nested", ("nested_item", "struct_item"))):
        if weights[kind] > 0:
            r = residual[kind] / weights[kind]
            for key in keys:
                overhead[key] = max(0, int(round(overhead[key] * r)))
    decode = dict(est.decode)
    for kind in ("flat", "string", "nested"):
        dr = ratio(kind, "arrow", "footer")
        if dr is not None:
            decode[kind] = round(dr, 4)
    total_pandas = sum(c["pandas_bytes"] for c in columns.values())
    total_arrow = sum(c["arrow_bytes"] for c in columns.values())
    return {"method": "sample", "rows": rows_total, "rows_sampled": int(rows_sampled), "scale": round(scale, 6),
            "bytes_per_row": total_pandas / float(max(1, rows_total)),
            "peak_bytes_per_row": (total_pandas + total_arrow) / float(max(1, rows_total)), "columns": columns,
            "expansion": expansion, "object_overhead_bytes": overhead, "decode": decode,
            "factors": {k: round(v["pandas"] / v["seed"], 4) for k, v in sums.items() if v["seed"] > 0}}


# --------------------------------------------------------------------------- storage


def write_calibration(cache_dir: str | Path, source: str, fingerprint: str, cal: Mapping[str, Any]) -> Path:
    return _write_json(calibration_path(cache_dir, source, fingerprint), dict(cal))


def load_calibrations(cache_dir: str | Path | None) -> dict[str, dict[str, Any]]:
    """``{fingerprint: calibration}`` of every ``calibration.json`` under ``cache_dir``."""
    out: dict[str, dict[str, Any]] = {}
    if not cache_dir:
        return out
    base = Path(cache_dir)
    if not base.is_dir():
        return out
    for path in sorted(base.glob(f"*/*/{CALIBRATION_FILE}")):
        try:
            cal = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        fp = cal.get("fingerprint") if isinstance(cal, dict) else None
        if fp:
            out[str(fp)] = cal
    return out


def load_feedback(cache_dir: str | Path | None) -> dict[str, dict[str, Any]]:
    """``{fingerprint or table: {factor, ...}}`` from ``memory_feedback.json``."""
    if not cache_dir:
        return {}
    try:
        data = json.loads(feedback_path(cache_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): v for k, v in (data.get("tables") or {}).items() if isinstance(v, dict)}


def record_feedback(cache_dir: str | Path, tables: Mapping[str, Mapping[str, Any]], measured_mb: float, *,
                    server: str | None = None, max_observations: int = 20) -> dict[str, Any]:
    """Record one measured cold load: ``tables`` maps ``table`` -> ``{estimated_mb, fingerprint}``; the factor
    ``measured / sum(estimated)`` is kept per fingerprint (else table name). The stored factor is the
    largest of the last ``max_observations`` (memory estimates err high, never low)."""
    est_total = sum(float(t.get("estimated_mb") or 0.0) for t in tables.values())
    if est_total <= 0 or measured_mb is None or measured_mb <= 0:
        return {}
    factor = float(measured_mb) / est_total
    path = feedback_path(cache_dir)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    store = data.setdefault("tables", {})
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for table, t in tables.items():
        key = str(t.get("fingerprint") or table)
        entry = store.setdefault(key, {"table": table, "observations": []})
        obs = list(entry.get("observations") or [])
        obs.append({"factor": round(factor, 4), "measured_mb": round(float(measured_mb), 1),
                    "estimated_mb": round(est_total, 1), "server": server, "at": now})
        entry["observations"] = obs[-max_observations:]
        entry["factor"] = max(o["factor"] for o in entry["observations"])
        entry["table"] = table
    _write_json(path, data)
    return data


def feedback_from_status(status: Mapping[str, Any] | None, baseline_mb: float | None) -> float | None:
    """MB a load added according to a reaper status record: ``peak_rss_mb - baseline`` (None when unknown)."""
    if not status:
        return None
    peak = status.get("peak_rss_mb")
    if not isinstance(peak, (int, float)):
        return None
    base = float(baseline_mb or 0.0)
    value = float(peak) - base
    return value if value > 0 and math.isfinite(value) else None


def factors_summary(calibrations: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """The median per-kind factor over several calibrations (``vbt ds status``)."""
    acc: dict[str, list[float]] = {}
    for cal in calibrations:
        for k, v in (cal.get("factors") or {}).items():
            acc.setdefault(k, []).append(float(v))
    return {k: sorted(v)[len(v) // 2] for k, v in sorted(acc.items())}
