"""``_stats``: fingerprints, partition fingerprints, rows, bytes and per-leaf statistics (§11.8, §10.4).

The gateway's memory estimates and admission read these. ``row_bytes_p99`` is the 99th percentile
of the JSON size of rows from a seeded sample, per output grain: ``row`` for the table itself and
one entry per item table over it (an item table requested by name reports its items as ``row``).
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from ...ipc import VERB_STATS, ColumnStatsModel, StatsRequest, StatsResponse, TableStatsModel
from ...plugins.base import FormatError
from .. import ServiceContext, ServiceError
from ..checks import aggregate_stats
from ..reader import BudgetExceeded, rendered

__all__ = ["stats", "table_stats", "row_bytes_p99", "VERBS"]

SAMPLE_ROWS = 500


def row_bytes_p99(reader: Any, n: int = SAMPLE_ROWS, seed: int = 0) -> int | None:
    rows = reader.sample_rows(n, seed=seed)
    if not rows:
        return None
    sizes = sorted(len(json.dumps(r, default=str, ensure_ascii=False).encode("utf-8")) for r in rows)
    return sizes[min(len(sizes) - 1, int(0.99 * len(sizes)))]


def table_stats(ctx: ServiceContext, ref: str) -> TableStatsModel:
    reader = ctx.reader(ref)
    cols, rows, size = aggregate_stats(reader)
    columns = {}
    for path, cs in cols.items():
        columns[path] = ColumnStatsModel(uncompressed_bytes=cs.uncompressed_bytes, num_values=cs.num_values,
                                         max_rep_level=cs.max_rep_level, max_def_level=cs.max_def_level,
                                         kind=cs.kind, storage_type=cs.storage_type, null_count=cs.null_count,
                                         min=rendered(cs.min, cs.storage_type), max=rendered(cs.max, cs.storage_type))
    p99: dict[str, int] = {}
    if reader.table.is_item_table:
        try:
            rows = reader.count()[0]
        except (BudgetExceeded, ServiceError):
            rows = None
        v = row_bytes_p99(reader)
        if v is not None:
            p99["row"] = v
    else:
        v = row_bytes_p99(reader)
        if v is not None:
            p99["row"] = v
        for item_ref in ctx.item_tables_of(str(reader.table.physical)):
            try:
                iv = row_bytes_p99(ctx.reader(item_ref))
            except (ServiceError, FormatError):
                continue
            if iv is not None:
                p99[item_ref.split(".", 1)[1]] = iv
    return TableStatsModel(fingerprint=reader.fingerprint(), signature=reader.signature(),
                           partition_fingerprints=reader.partition_fingerprints(), rows=rows,
                           fragments=len(reader.fragments()), bytes_on_disk=size, columns=columns, row_bytes_p99=p99)


def stats(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    req = StatsRequest.model_validate(dict(payload))
    out = StatsResponse()
    for ref in req.tables or ctx.table_refs(item_tables=False):
        try:
            out.tables[ref] = table_stats(ctx, ref)
        except (ServiceError, FormatError, OSError) as exc:
            out.errors[ref] = f"{type(exc).__name__}: {exc}"
    return out.model_dump(mode="json", by_alias=True)


VERBS = {VERB_STATS: stats}
