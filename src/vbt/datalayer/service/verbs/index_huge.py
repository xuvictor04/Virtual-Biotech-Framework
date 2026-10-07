"""``_build_index_huge``: row-group value indexes over huge tables, within time and byte budgets (F19).

``_build_index`` builds an ``access_paths`` sidecar in one pass and refuses when the indexed leaf
decodes to more than the scan budget (2 GB). The ``literature`` table (about 90 GB EST, every
``keywordId`` in every shard) needs more, so this verb builds the same sidecar
(:func:`vbt.datalayer.service.sidecar.access_index_path`, readable by ``load_access_index``) in
**resumable** steps:

* row groups are read one at a time (only the indexed leaf), in a fixed order;
* ``value -> (fragment, row group)`` pairs accumulate in memory up to ``max_entries`` pairs, then
  are spilled as a sorted gzip run under ``<index>.work/``; memory stays bounded by ``max_entries``;
* each call stops at its **time budget** (``time_budget_s``) or **byte budget** (``budget_bytes``
  decoded bytes) and records its progress (``progress.json``: the table fingerprint and the row
  groups done); the next call resumes there. A fingerprint change discards the work;
* when every row group is done, the runs are merged (k-way, streaming) into the final index, which
  the reader then uses for ``via: sidecar_index`` pruning.

Response: ``{status: complete|incomplete, path, rows (distinct values), fingerprint, units_done,
units_total, scanned_bytes, elapsed_s, reason?}``. ``vbt ds index build --access-paths --huge``
runs it from the command line, repeatedly if needed.
"""

from __future__ import annotations

import csv
import gzip
import heapq
import io
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from pydantic import BaseModel, ConfigDict

from ...rowkey import render_value
from .. import ServiceContext, ServiceError
from ..sidecar import _HEADER, _TSV, access_index_path

__all__ = ["VERB", "HugeIndexRequest", "build_huge_index", "build_index_huge", "VERBS"]

VERB = "_build_index_huge"
DEFAULT_TIME_BUDGET_S = 3600.0
DEFAULT_MAX_ENTRIES = 2_000_000


class HugeIndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table: str
    column: str | None = None                  # default: the table's first sidecar_index access path
    budget_bytes: int | None = None            # decoded bytes this call may read (default: unbounded)
    time_budget_s: float | None = None         # wall time this call may take (default 3600 s)
    max_entries: int | None = None             # pairs held in memory before a run is spilled
    force: bool = False


def _units(reader: Any) -> list[tuple[Any, int | None]]:
    out: list[tuple[Any, int | None]] = []
    for frag in reader.fragments():
        info = reader.footer(frag)
        if info is not None and info.row_groups:
            out.extend((frag, i) for i in range(len(info.row_groups)))
        else:
            out.append((frag, None))
    return out


def _unit_values(reader: Any, leaf: str, physical: str, frag: Any, rg: int | None) -> list[Any]:
    import pyarrow as pa

    from ..reader import _leaf_array, _tokens

    tokens = _tokens(physical)
    if rg is None:
        batches = list(reader.fmt.scan([frag], columns=[tokens[0]], predicate=None, partitions=reader.partitions))
        if not batches:
            return []
        tbl = pa.Table.from_batches(batches)
    else:
        tbl = reader.fmt.read_leaves(frag, [leaf], [rg])
    return _leaf_array(tbl.column(tokens[0]), tokens[1:]).to_pylist()


def _write_run(path: Path, pairs: set[tuple[str, str, int]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, dialect=_TSV)
        for value, frag, rg in sorted(pairs):
            w.writerow((value, frag, rg))


def _read_run(path: Path) -> Iterator[tuple[str, str, int]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh, dialect=_TSV):
            if len(row) == 3:
                yield row[0], row[1], int(row[2])


def _merge(runs: list[Path], out: Path, column: str, storage: str | None) -> int:
    """K-way merge of sorted runs into the sidecar format; returns the number of distinct values."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.merge.tmp")
    distinct = 0
    with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
        text = io.TextIOWrapper(gz, encoding="utf-8", newline="")
        w = csv.writer(text, dialect=_TSV)
        w.writerow(("#column", column, storage or ""))
        w.writerow(_HEADER)
        cur_value: str | None = None
        cur_frag: str | None = None
        rgs: list[int] = []

        def flush() -> None:
            if cur_value is not None and cur_frag is not None and rgs:
                w.writerow((cur_value, cur_frag, ",".join(str(r) for r in rgs)))

        last: tuple[str, str, int] | None = None
        for item in heapq.merge(*(_read_run(r) for r in runs)):
            if item == last:
                continue
            last = item
            value, frag, rg = item
            if value != cur_value or frag != cur_frag:
                flush()
                if value != cur_value:
                    distinct += 1
                cur_value, cur_frag, rgs = value, frag, []
            rgs.append(rg)
        flush()
        text.flush()
        text.detach()
    os.replace(tmp, out)
    return distinct


def build_huge_index(reader: Any, column: str, *, budget_bytes: int | None = None,
                     time_budget_s: float | None = None, max_entries: int | None = None, force: bool = False,
                     clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Build (or continue building) the sidecar index of ``column`` (see the module docstring)."""
    start = clock()
    fp = reader.fingerprint()
    t = reader.table
    path = access_index_path(reader.ctx.settings.cache_dir, t.physical.source, fp, t.physical.table, column)
    work = path.with_name(path.name + ".work")
    if force:
        shutil.rmtree(work, ignore_errors=True)
        try:
            path.unlink()
        except OSError:
            pass
    units = _units(reader)
    if path.exists():
        return {"status": "complete", "path": str(path), "fingerprint": fp, "units_done": len(units),
                "units_total": len(units), "scanned_bytes": 0, "elapsed_s": 0.0, "rows": None}
    progress_file = work / "progress.json"
    progress: dict[str, Any] = {}
    try:
        progress = json.loads(progress_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        progress = {}
    if progress.get("fingerprint") != fp or progress.get("column") != column:
        shutil.rmtree(work, ignore_errors=True)
        progress = {"fingerprint": fp, "column": column, "done": [], "runs": [], "scanned_bytes": 0}
    work.mkdir(parents=True, exist_ok=True)
    done = {(d[0], d[1]) for d in progress.get("done", [])}
    leaf_physical = reader.physical_path(column)
    leaf = reader.leaf(leaf_physical)
    if leaf is None:
        raise ServiceError(f"{t.physical}: column {column!r} is not in the data")
    storage = reader.storage_type(leaf_physical)
    limit_entries = int(max_entries or DEFAULT_MAX_ENTRIES)
    time_limit = DEFAULT_TIME_BUDGET_S if time_budget_s is None else float(time_budget_s)
    scanned = 0
    pairs: set[tuple[str, str, int]] = set()
    reason = None

    def spill() -> None:
        if not pairs:
            return
        run = work / f"run-{len(progress['runs']):05d}.tsv.gz"
        _write_run(run, pairs)
        progress["runs"].append(run.name)
        pairs.clear()

    def checkpoint() -> None:
        spill()
        progress["done"] = sorted([n, r] for n, r in done)
        progress["scanned_bytes"] = int(progress.get("scanned_bytes", 0)) + scanned
        tmp = progress_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(progress, sort_keys=True), encoding="utf-8")
        os.replace(tmp, progress_file)

    for frag, rg in units:
        name = reader.fragment_name(frag)
        unit_key = (name, -1 if rg is None else rg)
        if unit_key in done:
            continue
        info = reader.footer(frag)
        need = info.row_groups[rg].leaf_bytes(leaf) if (info is not None and rg is not None) else int(frag.size or 0)
        if clock() - start > time_limit:
            reason = f"time budget of {time_limit:g} s reached"
            break
        if budget_bytes is not None and scanned + need > int(budget_bytes):
            reason = f"byte budget of {int(budget_bytes)} decoded bytes reached"
            break
        for v in _unit_values(reader, leaf, leaf_physical, frag, rg):
            if v is None:
                continue
            pairs.add((render_value(v, storage), name, 0 if rg is None else rg))
        scanned += need
        done.add(unit_key)
        if len(pairs) >= limit_entries:
            checkpoint()
            scanned = 0
    checkpoint()
    elapsed = round(clock() - start, 3)
    base = {"path": str(path), "fingerprint": fp, "units_done": len(done), "units_total": len(units),
            "scanned_bytes": int(progress.get("scanned_bytes", 0)), "elapsed_s": elapsed}
    if len(done) < len(units):
        return {"status": "incomplete", "rows": None, "reason": reason, **base}
    rows = _merge([work / r for r in progress["runs"]], path, column, storage)
    shutil.rmtree(work, ignore_errors=True)
    reader.ctx.sidecars.pop(str(path), None)
    return {"status": "complete", "rows": rows, **base}


def build_index_huge(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    req = HugeIndexRequest.model_validate(dict(payload))
    reader = ctx.reader(req.table)
    phys = ctx.reader(str(reader.table.physical))
    column = req.column
    declared = [ap for ap in phys.spec.access_paths if ap.via == "sidecar_index"]
    if column is None:
        if not declared:
            raise ServiceError(f"{req.table} declares no sidecar_index access path")
        column = declared[0].columns[0]
    elif not any(ap.columns[:1] == [column] for ap in declared):
        raise ServiceError(f"{req.table} declares no sidecar_index access path on {column!r} "
                           f"(declared: {[ap.columns for ap in declared]})")
    return build_huge_index(phys, column, budget_bytes=req.budget_bytes, time_budget_s=req.time_budget_s,
                            max_entries=req.max_entries, force=req.force)


VERBS = {VERB: build_index_huge}
