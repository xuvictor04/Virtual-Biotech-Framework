"""The memory step of ``vbt validate``: the admission estimate of a whole-table load against the measured peak.

For the tables the enabled servers load whole (``reads.<table>.access: full_table``) that are ready and local:
the table statistics come from the data child (``_stats``), the estimate from :class:`MemoryEstimator` with the
shipped factors, then again after ``vbt ds calibrate`` wrote the table's sample-tier calibration; the measurement
is the oracle loading the table the way the upstream servers do (``dataset.to_table().to_pandas()``) under the
reaper at the server limit, minus the interpreter's resident memory before the load. The largest tables whose
estimate (with the admission safety) fits the server limit are measured, at most ``--memory-tables``.
"""

from __future__ import annotations

from typing import Any

from .cases import table_source
from .contained import run_oracle
from .report import PASS, WARN, StepResult

__all__ = ["run_memory", "UNDER", "OVER"]

TITLE = "Memory estimate against the measured load"
#: estimate / measured below this is an under-estimate admission could be killed by; above OVER refuses too early.
UNDER, OVER = 0.7, 1.5


def _full_table_reads(ctx: Any) -> dict[str, list[str]]:
    """``{table: [server.tool, ...]}`` of the tables loaded whole by the enabled servers' tools."""
    out: dict[str, list[str]] = {}
    for server in ctx.servers():
        for tool in ctx.catalog.tools(server):
            try:
                c = ctx.catalog.contract(server, tool)
            except Exception:  # noqa: BLE001
                continue
            for ref in c.full_table_reads:
                out.setdefault(ref, []).append(f"{server}.{tool}")
    return out


def run_memory(ctx: Any) -> StepResult:
    from ..datalayer.cli import _call_child, _table_stats
    from ..datalayer.ipc import TableStatsModel
    from ..datalayer.launch import server_limit_mb
    from ..datalayer.memory import MB, MemoryEstimator

    if ctx.opts.memory_tables <= 0:
        return StepResult.skipped("memory", TITLE, "--memory-tables 0")
    reads = _full_table_reads(ctx)
    ready = ctx.ready()
    local = {t: users for t, users in reads.items() if t in ready and table_source(ctx.catalog, t)[0] is not None}
    if not local:
        return StepResult.skipped("memory", TITLE, "no table an enabled server loads whole is ready and local here")
    stats = _table_stats(ctx.config, sorted(local))
    est = MemoryEstimator.from_settings(ctx.settings)
    limit = server_limit_mb("validate-load", ctx.settings)
    sized = []
    for ref, ts in (stats.get("tables") or {}).items():
        model = TableStatsModel.model_validate(ts)
        mb = est.peak_upstream(model) / MB
        sized.append((mb, ref, model))
    fits = sorted([s for s in sized if not limit or s[0] * est.safety <= limit], key=lambda s: -s[0])
    chosen = fits[:ctx.opts.memory_tables]
    rows = []
    worst = None
    peak = 0.0
    for shipped_mb, ref, model in chosen:
        calibration: dict[str, Any] = {}
        try:
            calibration = _call_child(ctx.config, "vbt.datalayer.memory.calibrate:calibrate", {"table": ref})
        except Exception as exc:  # noqa: BLE001 - the measurement still runs
            calibration = {"error": str(exc)[:300]}
        sample = MemoryEstimator.from_settings(ctx.settings, load_calibrations=True)
        sample_mb = sample.peak_upstream(model) / MB
        tier = sample.tier(model)
        source, _why = table_source(ctx.catalog, ref)
        answers, status = run_oracle(ctx.config, [{"id": "load", "kind": "load", **(source or {})}],
                                     limit_mb=limit or None, name="validate-load")
        got = answers.get("load") or {}
        peak = max(peak, float(status.get("peak_rss_mb") or 0))
        if got.get("error") or not got.get("peak_mb"):
            rows.append({"table": ref, "rows": model.rows, "shipped estimate MB": round(shipped_mb),
                         "sample-tier MB": round(sample_mb), "measured MB": None,
                         "note": got.get("error") or answers.get("_error") or f"exit {status.get('exit')}"})
            continue
        reaper_peak = float(status.get("peak_rss_mb") or got["peak_mb"])
        measured = max(reaper_peak, float(got["peak_mb"])) - float(got.get("rss_before_mb") or 0)
        r_shipped, r_sample = shipped_mb / measured, sample_mb / measured
        worst = r_sample if worst is None else min(worst, r_sample)
        rows.append({"table": ref, "rows": model.rows, "loaded by": ", ".join(local.get(ref, [])[:3]),
                     "shipped estimate MB": round(shipped_mb), "sample-tier MB": round(sample_mb), "tier": tier,
                     "measured MB": round(measured), "shipped/measured": round(r_shipped, 2),
                     "sample/measured": round(r_sample, 2), "load s": got.get("seconds"),
                     "rows sampled": calibration.get("rows_sampled")})
    skipped = sorted(ref for mb, ref, _m in sized if limit and mb * est.safety > limit)
    status = PASS if worst is None or worst >= UNDER else WARN
    summary = (f"{len([r for r in rows if r.get('measured MB')])} table(s) measured at the {limit:,} MB server limit"
               + (f"; sample-tier estimate / measured from {min(r['sample/measured'] for r in rows if 'sample/measured' in r):.2f} "
                  f"to {max(r['sample/measured'] for r in rows if 'sample/measured' in r):.2f}"
                  if any("sample/measured" in r for r in rows) else "")
               + (f"; {len(skipped)} larger than the limit allows: refused too_large on this host" if skipped else ""))
    return StepResult("memory", TITLE, status, summary, rows=rows, peak_mb=peak or None,
                      details={"server_limit_mb": limit, "too_large_here": skipped,
                               "candidates": len(local), "under": UNDER, "over": OVER})
