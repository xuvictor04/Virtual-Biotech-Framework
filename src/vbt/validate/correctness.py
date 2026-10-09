"""The correctness step of ``vbt validate``: the six tests on the real data, off and enforce, one server at a time.

For each enabled server whose bound tables are ready on this host: the cases are planned from the bindings and
the upstream signatures (read offline from the servers' source, :func:`vbt.datalayer.cli.upstream_tool_schemas`),
the oracle answers their questions (two contained runs: the samples, then the counts, top k and null counts), and
only then the server starts behind the gateway with its data child, so the oracle and the server are never
resident together. Each case is called ``off`` then ``enforce`` on the same server process; the enforce answer is
judged (:func:`.cases.judge`), the off answer recorded. The data child gets the check step's results
(``DataGateway.set_readiness``) instead of a second session check.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .cases import complete_cases, judge, judge_off, oracle_queries, plan_cases, table_source
from .contained import Calls, run_oracle
from .report import FAIL, PASS, WARN, StepResult

__all__ = ["run_correctness"]

TITLE = "The six correctness tests (off vs enforce)"


def _servers_with_local_tables(ctx: Any) -> dict[str, list[str]]:
    """``{server: [ready local tables its bound tools read]}``."""
    ready = ctx.ready()
    out: dict[str, list[str]] = {}
    for server in ctx.servers():
        tables = set()
        for tool in ctx.catalog.tools(server):
            try:
                table = ctx.catalog.contract(server, tool).bound_table
            except Exception:  # noqa: BLE001
                continue
            if table and table in ready and table_source(ctx.catalog, table)[0] is not None:
                tables.add(table)
        if tables:
            out[server] = sorted(tables)
    return out


async def run_correctness(ctx: Any) -> StepResult:
    from ..datalayer.cli import upstream_tool_schemas

    targets = _servers_with_local_tables(ctx)
    if not targets:
        return StepResult.skipped("correctness", TITLE, "no enabled server reads a ready local table on this host")
    up = (ctx.config.get("vars") or {}).get("upstream")
    schemas_all = upstream_tool_schemas(up, ctx.settings.project_root)
    rows: list[dict[str, Any]] = []
    verdicts: Counter[str] = Counter()
    by_ct: dict[str, Counter[str]] = {}
    servers_out: dict[str, Any] = {}
    skipped_tools: dict[str, str] = {}
    oracle_peak = 0.0
    for server, tables in sorted(targets.items()):
        schemas = {k.split(".", 1)[1]: v for k, v in schemas_all.items() if k.split(".", 1)[0] == server}
        plan = plan_cases(ctx.catalog, ctx.registry, server, schemas, ctx.ready(), max_tools=ctx.opts.max_tools)
        skipped_tools.update(plan.skipped)
        if not plan.cases:
            servers_out[server] = {"cases": 0, "tables": tables}
            continue
        answers, st1 = run_oracle(ctx.config, plan.queries) if plan.queries else ({}, {})
        second = oracle_queries(plan, answers, ctx.catalog)
        more, st2 = run_oracle(ctx.config, second) if second else ({}, {})
        oracle_peak = max(oracle_peak, float(st1.get("peak_rss_mb") or 0), float(st2.get("peak_rss_mb") or 0))
        complete_cases(plan, {**answers, **more})
        live = [c for c in plan.cases if not c.skip]
        for c in plan.cases:
            if c.skip:
                rows.append({"test": c.ct, "tool": f"{server}.{c.tool}", "arguments": c.args, "oracle": c.skip,
                             "off": "", "enforce": "skipped", "detail": ""})
                verdicts["skipped"] += 1
                by_ct.setdefault(c.ct, Counter())["skipped"] += 1
        if not live:
            servers_out[server] = {"cases": 0, "tables": tables, "note": "the oracle backed no case"}
            continue
        async with Calls(ctx.config, [server], ctx.out / "correctness" / server, readiness=ctx.check,
                         timeout_s=ctx.opts.timeout_s) as calls:
            if calls.failures:
                servers_out[server] = {"failed_to_start": calls.failures}
                for c in live:
                    rows.append({"test": c.ct, "tool": f"{server}.{c.tool}", "arguments": c.args, "oracle": "",
                                 "off": "", "enforce": "wrong", "detail": f"server failed to start: {calls.failures}"})
                    verdicts["wrong"] += 1
                continue
            called: set[str] = set()
            for i, c in enumerate(live):
                binding = ctx.catalog.contract(server, c.tool).binding
                first = c.tool not in called          # the first call of a tool loads its tables (cold)
                called.add(c.tool)
                enf = await calls.call(server, c.tool, c.args, mode="enforce")
                verdict, detail = judge(c, enf, binding)
                verdicts[verdict] += 1
                by_ct.setdefault(c.ct, Counter())[verdict] += 1
                key = f"{server}.{c.tool}#{i}"
                ctx.calls.append({"server": server, "tool": c.tool, "case": key, "mode": "enforce",
                                  "seconds": enf.seconds, "verdict": verdict, "first": first})
                guard = None
                whole, scanned = _upstream_reads(ctx.catalog, binding)
                if enf.is_error and enf.kind == "too_large":
                    # the gateway refused the load this host cannot hold: the same call without it would load
                    # the tables anyway and be killed at the server's limit (or take the host with it)
                    guard = "not called: the enforce call was refused too_large on this host"
                else:
                    guard = await calls.off_guard(server, whole, scanned)
                if guard:
                    off_text = guard
                else:
                    off = await calls.call(server, c.tool, c.args, mode="off")
                    calls.after_off(server, whole, off)
                    ctx.calls.append({"server": server, "tool": c.tool, "case": key, "mode": "off",
                                      "seconds": off.seconds, "first": first})
                    off_text = judge_off(c, off, binding)
                rows.append({"test": c.ct, "tool": f"{server}.{c.tool}", "arguments": c.args,
                             "oracle": _oracle_text(c), "off": off_text, "enforce": verdict, "detail": detail})
            ctx.verbs.extend(calls.verb_times)
            status = calls.server_status()
            servers_out[server] = {"tables": tables, "cases": len(live), "start_s": round(calls.started_s, 1),
                                   "peak_mb": {n: s.get("peak_rss_mb") for n, s in status.items()},
                                   "containment": {n: s.get("containment") for n, s in status.items()},
                                   "limit_mb": {n: s.get("limit_mb") for n, s in status.items()},
                                   "exits": {n: s.get("exit") for n, s in status.items() if s.get("exit")}}
        # each server's cases are on disk as soon as it is done (a long run that stops keeps what it judged)
        mine = [r for r in rows if r["tool"].startswith(f"{server}.")]
        out_dir = ctx.out / "correctness" / server
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "cases.json").write_text(json.dumps({"server": servers_out.get(server), "cases": mine},
                                                       default=str, indent=1), encoding="utf-8")
        release_memory()
    wrong = verdicts.get("wrong", 0)
    judged = sum(v for k, v in verdicts.items() if k != "skipped")
    summary = (f"{judged} case(s) on {len([s for s in servers_out if servers_out[s].get('cases')])} server(s): "
               f"{verdicts.get('correct', 0)} correct, {verdicts.get('refused', 0)} typed refusals, {wrong} wrong; "
               f"{verdicts.get('skipped', 0)} not backed by the oracle")
    status = FAIL if wrong else (PASS if judged else WARN)
    return StepResult("correctness", TITLE, status, summary, rows=rows, peak_mb=oracle_peak or None,
                      details={"by_test": {k: dict(v) for k, v in sorted(by_ct.items())}, "servers": servers_out,
                               "tools_without_cases": skipped_tools})


def release_memory() -> None:
    """Hand what one server's calls freed back to the system before the next server loads its tables: the parsed
    upstream answers and the gateway's results stay in this process's heap otherwise (glibc keeps freed pages)."""
    import ctypes
    import gc
    import sys

    gc.collect()
    pa = sys.modules.get("pyarrow")
    if pa is not None:
        try:
            pa.default_memory_pool().release_unused()
        except Exception:  # noqa: BLE001
            pass
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass


def _upstream_reads(catalog: Any, binding: Any) -> tuple[list[str], list[str]]:
    """``(whole, scanned)``: the physical tables a tool's upstream call loads whole (``access: full_table``) and those
    it scans with a filter (``bounded_scan``)."""
    whole: list[str] = []
    scanned: list[str] = []
    for ref, rs in (getattr(binding, "reads", None) or {}).items():
        access = getattr(rs, "access", None)
        if access not in ("full_table", "bounded_scan"):
            continue
        try:
            t = catalog.table(str(ref))
            name = str(t.physical) if t.is_item_table else str(ref)
        except Exception:  # noqa: BLE001
            name = str(ref)
        (whole if access == "full_table" else scanned).append(name)
    return whole, scanned


def _oracle_text(c: Any) -> str:
    o = c.oracle
    if c.ct == "CT-3":
        if c.items:
            return f"{c.value!r}: {o.get('count')} row(s) holding {o.get('items')} item(s) at {c.items}"
        return f"{c.value!r}: {o.get('count')} row(s)"
    if c.ct == "CT-1":
        return f"{o.get('variants')} -> {o.get('canonical')!r}"
    if c.ct == "CT-2":
        return f"{o.get('absent')!r} in no row"
    if c.ct == "CT-4":
        return f"top {o.get('top')} of {o.get('total')}"
    if c.ct == "CT-6":
        return f"median {o.get('median')}, {o.get('nulls')} null(s)"
    return json.dumps({k: v for k, v in o.items() if not callable(v)}, default=str)[:200]
