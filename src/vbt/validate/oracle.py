#!/usr/bin/env python3
"""The independent oracle of ``vbt validate`` (stdlib and pyarrow only; never imports ``vbt``).

Usage::

    python -E oracle.py <queries.json> <answers.json>

It answers questions about a table straight from its files with pyarrow, so the expected answers of the
correctness cases never come from the code under test (the data layer's readers, witness or gateway). The
harness runs it under the reaper with the data child's memory limit (``vbt.preflight.run_contained``): decoding
reference data never happens in, or next to, the harness. Each query names its files as the descriptor declares
them (``source`` = a directory of shards or one file; ``format`` parquet, csv or tsv; ``partitioning`` hive or
none) and one of:

* ``sample``: up to ``n`` distinct non-null values of a top-level ``column``, read from evenly spaced row groups
  (with ``items``, a container path such as ``a[]`` or ``a.b[]``: only values whose row holds at least one item);
* ``count``: rows whose ``column`` equals each of ``values`` (compared as text), and with ``items`` the items of
  those rows' containers;
* ``topk``: the rows whose ``filter.column`` equals ``filter.value``, sorted by ``order`` (``[{column, direction,
  nulls}]``; ties keep file order), the first ``k`` rows' order values, the matched total and the order column's
  null count among them;
* ``nulls``: null and non-null counts of ``column`` (among the rows of ``filter`` when given) and the median of the
  non-null values;
* ``distinct``: up to ``max`` distinct values of ``column``;
* ``load``: the table loaded whole the way the upstream servers load it (``dataset.to_table().to_pandas()``):
  rows, the resident memory before and the peak after (``ru_maxrss``).

A failed query answers ``{"error": "..."}``; the others still run.
"""

from __future__ import annotations

import json
import resource
import sys
import time
from typing import Any


def _rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _dataset(q: dict[str, Any]) -> Any:
    import pyarrow.dataset as ds

    fmt = str(q.get("format") or "parquet")
    if fmt in ("csv", "tsv"):
        import pyarrow.csv as pcsv

        delimiter = q.get("delimiter") or ("\t" if fmt == "tsv" else ",")
        fmt_obj: Any = ds.CsvFileFormat(parse_options=pcsv.ParseOptions(delimiter=delimiter))
    else:
        fmt_obj = "parquet"
    partitioning = "hive" if q.get("partitioning") == "hive" else None
    return ds.dataset(q["source"], format=fmt_obj, partitioning=partitioning)


def _text(arr: Any) -> Any:
    import pyarrow as pa
    import pyarrow.compute as pc

    if pa.types.is_dictionary(arr.type):
        arr = arr.dictionary_decode() if hasattr(arr, "dictionary_decode") else pc.cast(arr, arr.type.value_type)
    return arr if pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type) else pc.cast(arr, pa.string())


def _scalar(v: Any) -> Any:
    if isinstance(v, float) and v != v:
        return None
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "replace")
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def _equal_mask(table: Any, column: str, value: Any) -> Any:
    import pyarrow as pa
    import pyarrow.compute as pc

    col = table.column(column)
    col = pa.chunked_array([_text(c) for c in col.chunks], type=pa.string()) if col.num_chunks else \
        pa.chunked_array([], type=pa.string())
    return pc.equal(col, pa.scalar(str(value), pa.string()))


def _segments(path: str) -> list[tuple[str, bool]]:
    """``a.b[]`` -> ``[("a", False), ("b", True)]``."""
    out = []
    for part in [p for p in str(path).split(".") if p]:
        out.append((part.split("[", 1)[0], "[" in part))
    return out


def _items(value: Any, segments: list[tuple[str, bool]]) -> int:
    """How many items the container at ``segments`` holds in one row (``value`` is the first segment's column
    value); lists along the path are flattened, missing containers hold none, a path that ends at a struct or a
    scalar holds one item when it is not null."""
    current = [value]
    for depth, (name, is_list) in enumerate(segments):
        if depth:
            current = [v.get(name) if isinstance(v, dict) else None for v in current]
        if is_list:
            current = [x for v in current if isinstance(v, list) for x in v]
        else:
            current = [v for v in current if v is not None]
    return len(current)


def q_sample(q: dict[str, Any]) -> dict[str, Any]:
    import pyarrow.compute as pc

    dset = _dataset(q)
    n = int(q.get("n") or 3)
    column = q["column"]
    segments = _segments(q["items"]) if q.get("items") else []
    columns = list(dict.fromkeys([column, segments[0][0]])) if segments else [column]
    frags = sorted(dset.get_fragments(), key=lambda f: str(getattr(f, "path", "")))
    if not frags:
        return {"values": [], "rows": 0}
    spread = min(len(frags), 8 if segments else 3)
    picks = sorted({round(i * (len(frags) - 1) / max(1, spread - 1)) for i in range(spread)})
    values: list[Any] = []
    items: dict[str, int] = {}
    seen: set[str] = set()
    tries = (16 if segments else 4) * n
    for i in picks:
        tbl = frags[i].to_table(columns=columns)
        tbl = tbl.filter(pc.is_valid(tbl.column(column)))
        m = tbl.num_rows
        for j in ([int(k * (m - 1) / max(1, tries - 1)) for k in range(tries)] if m else []):
            v = _scalar(tbl.column(column)[j].as_py())
            if v is None or isinstance(v, (list, dict)) or str(v) == "" or str(v) in seen:
                continue
            if segments:
                count = _items(tbl.column(segments[0][0])[j].as_py(), segments)
                if count == 0:
                    continue
                items[str(v)] = count
            seen.add(str(v))
            values.append(v)
            if len(values) >= n:
                break
        if len(values) >= n:
            break
    out: dict[str, Any] = {"values": values, "type": str(dset.schema.field(column).type)}
    if segments:
        out["items"] = items
    return out


def q_count(q: dict[str, Any]) -> dict[str, Any]:
    import pyarrow.compute as pc

    dset = _dataset(q)
    column = q["column"]
    wanted = [str(v) for v in q.get("values") or []]
    segments = _segments(q["items"]) if q.get("items") else []
    counts = {v: 0 for v in wanted}
    items = {v: 0 for v in wanted}
    rows = 0
    columns = list(dict.fromkeys([column, segments[0][0]])) if segments else [column]
    for batch in dset.to_batches(columns=columns):
        rows += batch.num_rows
        col = _text(batch.column(0))
        for v in wanted:
            mask = pc.equal(col, v)
            hits = int(pc.sum(mask).as_py() or 0)
            counts[v] += hits
            if hits and segments:
                kept = batch.filter(mask).column(columns.index(segments[0][0])).to_pylist()
                items[v] += sum(_items(x, segments) for x in kept)
    out: dict[str, Any] = {"counts": counts, "rows": rows}
    if segments:
        out["items"] = items
    return out


def _sort_key(values: list[Any], direction: str, nulls: str) -> list[int]:
    """Positions of ``values`` sorted by ``direction`` with nulls first or last; ties keep file order."""
    absolute = direction.endswith("_abs")
    desc = direction.startswith("desc")

    def key(i: int) -> tuple[int, Any]:
        v = values[i]
        missing = v is None or (isinstance(v, float) and v != v)
        if missing:
            return (0 if nulls == "first" else 2, 0)
        x = abs(v) if absolute and isinstance(v, (int, float)) else v
        return (1, x)

    present = [i for i in range(len(values)) if key(i)[0] == 1]
    present.sort(key=lambda i: key(i)[1], reverse=desc)
    missing = [i for i in range(len(values)) if key(i)[0] != 1]
    return (missing + present) if nulls == "first" else (present + missing)


def q_topk(q: dict[str, Any]) -> dict[str, Any]:
    dset = _dataset(q)
    flt = q.get("filter") or {}
    order = list(q.get("order") or [])
    first = order[0]
    cols = list(dict.fromkeys([flt["column"], first["column"]])) if flt else [first["column"]]
    tbl = dset.to_table(columns=cols)
    if flt:
        tbl = tbl.filter(_equal_mask(tbl, flt["column"], flt["value"]))
    values = [_scalar(v) for v in tbl.column(first["column"]).to_pylist()]
    idx = _sort_key(values, str(first.get("direction") or "desc"), str(first.get("nulls") or "last"))
    k = int(q.get("k") or 5)
    return {"top": [values[i] for i in idx[:k]], "total": len(values),
            "nulls": sum(1 for v in values if v is None)}


def q_nulls(q: dict[str, Any]) -> dict[str, Any]:
    import statistics

    dset = _dataset(q)
    flt = q.get("filter")
    column = q["column"]
    cols = list(dict.fromkeys([flt["column"], column])) if flt else [column]
    tbl = dset.to_table(columns=cols)
    if flt:
        tbl = tbl.filter(_equal_mask(tbl, flt["column"], flt["value"]))
    values = [_scalar(v) for v in tbl.column(column).to_pylist()]
    present = [v for v in values if v is not None]
    numeric = [v for v in present if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return {"rows": len(values), "nulls": len(values) - len(present), "non_null": len(present),
            "median": statistics.median(numeric) if numeric else None}


def q_distinct(q: dict[str, Any]) -> dict[str, Any]:
    import pyarrow.compute as pc

    dset = _dataset(q)
    out: set[Any] = set()
    cap = int(q.get("max") or 200)
    for batch in dset.to_batches(columns=[q["column"]]):
        for v in pc.unique(batch.column(0)).to_pylist():
            if v is not None:
                out.add(_scalar(v))
        if len(out) > cap:
            return {"values": sorted(out, key=str)[:cap], "complete": False}
    return {"values": sorted(out, key=str), "complete": True}


def q_load(q: dict[str, Any]) -> dict[str, Any]:
    import gc

    before = _rss_mb()
    t0 = time.monotonic()
    df = _dataset(q).to_table().to_pandas()
    rows = len(df)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    del df
    gc.collect()
    return {"rows": rows, "rss_before_mb": round(before, 1), "peak_mb": round(peak, 1),
            "seconds": round(time.monotonic() - t0, 2)}


QUERIES = {"sample": q_sample, "count": q_count, "topk": q_topk, "nulls": q_nulls, "distinct": q_distinct,
           "load": q_load}


def main(argv: list[str]) -> int:
    queries = json.loads(open(argv[1], encoding="utf-8").read())
    answers: dict[str, Any] = {}
    for q in queries:
        fn = QUERIES.get(str(q.get("kind")))
        try:
            answers[str(q["id"])] = fn(q) if fn else {"error": f"unknown query kind {q.get('kind')!r}"}
        except Exception as exc:  # noqa: BLE001 - one bad query never hides the others' answers
            answers[str(q["id"])] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
    answers["_peak_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    with open(argv[2], "w", encoding="utf-8") as f:
        json.dump(answers, f, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
