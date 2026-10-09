"""``vbt validate``: certify a host with what its descriptors declare and what is present (docs/DEPLOYMENT.md).

One command the owners run on their own host. Each step runs on whatever the descriptors, overlays and MCP server
file declare and the host holds, and is skipped, with the reason, when its prerequisite is missing:

=============  =====================================================================================================
step           what it checks
=============  =====================================================================================================
host           the memory this harness plans with and every host-scaled limit (:mod:`vbt.datalayer.memory.sizing`),
               the containment the reaper can use, CPUs and free disk
lint           ``vbt ds lint``: descriptors, overlays and the MCP server file
check          ``vbt ds check --depth deep`` on every declared table, in the data child under the reaper; a
               present table that is not ready fails, an absent one is listed
correctness    the six correctness tests (§19) on the real data, gateway off and enforce through one set of the
               unmodified servers, cases generated from the bindings and the descriptors' roles and answered by
               an independent pyarrow oracle (:mod:`.cases`, :mod:`.oracle`)
latency        per server: p50/p95 of the enforced and the off calls, the gateway's overhead and its witness
               requests to the data child, from the correctness calls
memory         the memory estimate (sample-tier calibration, ``vbt ds calibrate``) against the measured peak of
               loading each present table the way upstream loads it (whole, to pandas), under the reaper
live           the live sources (remote descriptors): their endpoints, then the servers that read them through the
               gateway with their sentinel controls
replication    with the Zenodo archive present: the Case 1 statistics on the authors' inputs, compared row by row
model          with a model server answering: ``vbt local check``
=============  =====================================================================================================

The report is written as ``validate.md`` and ``validate.json`` under ``--out`` (default ``<state>/validate/<UTC
time>``, the ``vbt setup`` state directory). The exit status is 1 when a step failed, else 0.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .report import ERROR, FAIL, PASS, WARN, StepResult, ValidationReport, host_facts, percentile

__all__ = ["STEPS", "Options", "run_validate", "add_validate_parser", "cmd_validate"]

STEPS = ("host", "lint", "check", "correctness", "latency", "memory", "live", "replication", "model")
_NOT_READY = {"schema_drift", "encoding_drift", "key_violation", "partial", "stale", "plugin_unavailable"}
_ABSENT = {"missing"}


class Options(argparse.Namespace):
    """The step options (the CLI flags; tests build them directly)."""

    def __init__(self, **kw: Any) -> None:
        super().__init__()
        self.steps: Sequence[str] = kw.pop("steps", STEPS)
        self.depth: str = kw.pop("depth", "deep")
        self.servers: Sequence[str] | None = kw.pop("servers", None)
        self.max_tools: int = int(kw.pop("max_tools", 40))
        self.memory_tables: int = int(kw.pop("memory_tables", 4))
        self.replicate: str = kw.pop("replicate", "quick")
        self.out: Path | None = kw.pop("out", None)
        self.timeout_s: float = float(kw.pop("timeout_s", 600))
        self.network: bool | None = kw.pop("network", None)
        self.check_from: Path | None = kw.pop("check_from", None)
        for k, v in kw.items():
            setattr(self, k, v)


class _Ctx:
    """What the steps share: the configuration, the catalog, the deep check's response and the call timings."""

    def __init__(self, config: Mapping[str, Any], opts: Options, out: Path) -> None:
        from ..preflight import data_catalog

        self.config = dict(config)
        self.opts = opts
        self.out = out
        self.settings, self.catalog, self.registry = data_catalog(self.config)
        self.check: dict[str, Any] | None = None
        self.calls: list[dict[str, Any]] = []            # correctness calls: server, tool, mode, seconds, verdict
        self.verbs: list[tuple[str, str, str, float]] = []

    def statuses(self) -> dict[str, str]:
        return {k: str((v or {}).get("status")) for k, v in ((self.check or {}).get("tables") or {}).items()}

    def ready(self) -> set[str]:
        return {k for k, v in self.statuses().items() if v in ("ready", "awaiting_producer")}

    def servers(self) -> list[str]:
        configured = [s.get("name") for s in (self.config.get("mcp_servers") or {}).get("servers", [])
                      if s.get("enabled", True)]
        wanted = set(self.opts.servers) if self.opts.servers else None
        return [s for s in configured if s and s in self.catalog.servers() and (wanted is None or s in wanted)]


# ---------------------------------------------------------------------------------------------------- steps


def step_host(ctx: _Ctx) -> StepResult:
    from ..datalayer.memory import sizing

    servers = {s["name"]: s.get("mem_limit_mb") for s in (ctx.config.get("mcp_servers") or {}).get("servers", [])
               if s.get("mem_limit_mb") is not None}
    parallel = int((ctx.config.get("limits") or {}).get("max_parallel_agents") or 8)
    desc = sizing.describe(ctx.settings.raw, servers=servers, parallel=parallel)
    try:
        from ..setup.probe import containment_facts

        containment = containment_facts()
    except Exception as exc:  # noqa: BLE001
        containment = {"note": f"not probed ({exc})"}
    disks = {}
    for name, path in (("data", ctx.settings.cache_dir.parent), ("out", ctx.out)):
        p = Path(path)
        while not p.exists() and p != p.parent:
            p = p.parent
        usage = shutil.disk_usage(p)
        disks[name] = {"path": str(path), "free_gb": round(usage.free / 1e9, 1), "total_gb": round(usage.total / 1e9, 1)}
    rows = [{"setting": k, "configured": v["configured"], "on this host": v["effective"], "rule": v["rule"]}
            for k, v in desc["settings"].items()]
    host = desc["host"]
    summary = (f"plans with {host['plan_mb']:,} MB ({host['plan_from']}; MemTotal {host['memtotal_mb'] or '?'} MB, "
               f"cgroup limit {host['cgroup_limit_mb'] or 'none'}); containment {containment.get('limit_kind')}"
               if host.get("plan_mb") else "the host's memory is unknown: the shipped limits apply")
    status = PASS if host.get("plan_mb") else WARN
    return StepResult("host", "Host and host-scaled limits", status, summary, rows=rows,
                      details={"sizing": desc, "containment": containment, "disk": disks, "cpus": os.cpu_count()})


def step_lint(ctx: _Ctx) -> StepResult:
    from ..datalayer.descriptor.lint import Finding
    from ..datalayer.launch import LIMIT_KINDS

    findings = list(ctx.catalog.lint(ctx.registry, strict=None))
    for spec in (ctx.config.get("mcp_servers") or {}).get("servers") or []:
        kind = spec.get("limit_kind") if isinstance(spec, Mapping) else None
        if kind is not None and kind not in LIMIT_KINDS:
            findings.append(Finding("error", f"mcp_servers.{spec.get('name')}.limit_kind",
                                    f"limit_kind {kind!r} is not one of {', '.join(LIMIT_KINDS)}", rule="server"))
    errors = [f for f in findings if f.level == "error"]
    n_tools = sum(len(ctx.catalog.tools(s)) for s in ctx.catalog.servers())
    rows = [{"level": f.level, "where": getattr(f, "where", None) or getattr(f, "path", None), "finding": str(f)}
            for f in findings[:200]]
    return StepResult("lint", "Descriptors and overlays (vbt ds lint)", FAIL if errors else PASS,
                      f"{len(ctx.catalog.sources)} sources, {n_tools} bound tools: {len(errors)} error(s), "
                      f"{len(findings) - len(errors)} warning(s)", rows=rows,
                      details={"quarantined": [q.to_json() for q in getattr(ctx.catalog, "quarantined", []) or []]})


def step_check(ctx: _Ctx) -> StepResult:
    from .contained import run_data_check_contained

    t0 = time.monotonic()
    tables: list[str] = []
    if ctx.opts.servers:                 # only what the chosen servers read (and their item tables' parents)
        for server in ctx.servers():
            for tool in ctx.catalog.tools(server):
                try:
                    contract = ctx.catalog.contract(server, tool)
                except Exception:  # noqa: BLE001
                    continue
                for ref, t in contract.tables.items():
                    tables += [str(ref), str(t.physical)] if t.is_item_table else [str(ref)]
        tables = sorted(set(tables))
    if ctx.opts.check_from:
        # an earlier run's check.json (the same CheckResponse): the steps after it run without a second deep check
        response = json.loads(Path(ctx.opts.check_from).read_text(encoding="utf-8"))
        status = {"reused": str(ctx.opts.check_from)}
    else:
        response, status = run_data_check_contained(ctx.config, depth=ctx.opts.depth, tables=tables,
                                                    timeout=24 * 3600)
    (ctx.out / "check.json").write_text(json.dumps(response, default=str), encoding="utf-8")
    ctx.check = response
    statuses = ctx.statuses()
    absent = sorted(k for k, v in statuses.items() if v in _ABSENT)
    bad = sorted(k for k, v in statuses.items() if v in _NOT_READY)
    errors = dict(response.get("table_errors") or {})
    rows = []
    for ref, m in sorted((response.get("tables") or {}).items()):
        if statuses.get(ref) in _ABSENT:
            continue
        failing = [c for c in (m.get("checks") or []) if not c.get("ok") and c.get("level") in ("error", "warning")]
        rows.append({"table": ref, "status": statuses.get(ref),
                     "findings": "; ".join(f"[{c.get('level')}] {c.get('name')} {c.get('column') or ''}: "
                                           f"{str(c.get('detail'))[:160]}" for c in failing[:3]) or ""})
    for ref, err in sorted(errors.items()):
        rows.append({"table": ref, "status": "check failed", "findings": str(err)[:300]})
    ready = len(statuses) - len(absent) - len(bad)
    depth = response.get("depth") or ctx.opts.depth
    summary = (f"depth {depth}: {ready} ready, {len(bad)} present but not ready, {len(absent)} absent "
               f"(not on this host), {len(errors)} check error(s)"
               + (f"; reused from {ctx.opts.check_from}" if ctx.opts.check_from else
                  f"; {status.get('groups', 1)} data-child process(es), the largest peak "
                  f"{float(status.get('peak_rss_mb') or 0):,.0f} MB"))
    return StepResult("check", f"Readiness (vbt ds check --depth {ctx.opts.depth})",
                      FAIL if bad or errors else PASS, summary, rows=rows, seconds=time.monotonic() - t0,
                      peak_mb=status.get("peak_rss_mb"),
                      details={"absent": absent, "not_ready": bad, "errors": errors, "containment": status})


def step_correctness(ctx: _Ctx) -> StepResult:
    from .correctness import run_correctness

    if ctx.check is None:
        return StepResult.skipped("correctness", "The six correctness tests (off vs enforce)",
                                  "needs the check step (which tables are ready)")
    if not ctx.ready():
        return StepResult.skipped("correctness", "The six correctness tests (off vs enforce)",
                                  "no declared table is ready on this host")
    why = _upstream_missing(ctx.config)
    if why:
        return StepResult.skipped("correctness", "The six correctness tests (off vs enforce)", why)
    return asyncio.run(run_correctness(ctx))


def step_latency(ctx: _Ctx) -> StepResult:
    title = "Gateway and witness latency per server"
    if not ctx.calls:
        return StepResult.skipped("latency", title, "no correctness calls were made (the correctness step did not "
                                  "run or generated no case)")
    rows = []
    servers = sorted({c["server"] for c in ctx.calls})
    for server in servers:
        mine = [c for c in ctx.calls if c["server"] == server]
        enf = [c["seconds"] for c in mine if c["mode"] == "enforce"]
        off = [c["seconds"] for c in mine if c["mode"] == "off"]
        cold = [c["seconds"] for c in mine if c["mode"] == "enforce" and c.get("first")]
        pairs: dict[str, dict[str, float]] = {}
        for c in mine:
            if not c.get("first"):             # a tool's first (enforce) call loads its tables: not overhead
                pairs.setdefault(c["case"], {})[c["mode"]] = c["seconds"]
        over = [p["enforce"] - p["off"] for p in pairs.values() if "enforce" in p and "off" in p]
        wit = [s for (srv, _t, verb, s) in ctx.verbs if srv == server and verb.startswith("_witness")]
        allv = [s for (srv, _t, _v, s) in ctx.verbs if srv == server]

        def ms(x: float | None) -> float | None:
            return None if x is None else round(1000 * x, 1)

        rows.append({"server": server, "calls": len(enf), "enforce p50 ms": ms(percentile(enf, 50)),
                     "enforce p95 ms": ms(percentile(enf, 95)), "off p50 ms": ms(percentile(off, 50)),
                     "off p95 ms": ms(percentile(off, 95)), "first calls": len(cold),
                     "first call p50 ms": ms(percentile(cold, 50)), "warm pairs": len(over),
                     "overhead p50 ms": ms(percentile(over, 50)),
                     "overhead p95 ms": ms(percentile(over, 95)), "witness requests": len(wit),
                     "witness p50 ms": ms(percentile(wit, 50)), "witness p95 ms": ms(percentile(wit, 95)),
                     "data-child requests": len(allv), "data-child p95 ms": ms(percentile(allv, 95))})
    return StepResult("latency", title, PASS, f"{len(ctx.calls)} timed calls on {len(servers)} server(s)", rows=rows)


def step_memory(ctx: _Ctx) -> StepResult:
    from .memory import run_memory

    title = "Memory estimate against the measured load"
    if ctx.check is None or not ctx.ready():
        return StepResult.skipped("memory", title, "no ready table to load (the check step did not run or found none)")
    return run_memory(ctx)


def step_live(ctx: _Ctx) -> StepResult:
    from .live import run_live

    return run_live(ctx)


def step_replication(ctx: _Ctx) -> StepResult:
    title = "Paper replication (Case 1 on the Zenodo archive)"
    if ctx.opts.replicate == "off":
        return StepResult.skipped("replication", title, "--replicate off")
    try:
        from ..data.zenodo import zenodo_root

        root = zenodo_root(ctx.config)
    except Exception as exc:  # noqa: BLE001
        return StepResult.skipped("replication", title, f"no Zenodo root ({exc})")
    if not root or not Path(root, "clinical_trials").is_dir():
        return StepResult.skipped("replication", title, f"the Zenodo archive is not at {root} (VBT_ZENODO_DIR; "
                                  "`vbt data zenodo fetch --preset case1`)")
    from ..case_studies.trial_outcomes.replicate import replicate_case1

    quick = ctx.opts.replicate != "full"
    t0 = time.monotonic()
    try:
        rep = replicate_case1(root, ctx.out / "case1_replication", n_perm=0 if quick else 1000,
                              gene_perm=0 if quick else 200, mixed=not quick, expr=not quick, progress=None)
    except FileNotFoundError as exc:
        return StepResult.skipped("replication", title, f"the archive lacks a Case 1 input: {exc}")
    agree = rep.agreement()
    rows = [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in agree.to_dict("records")]
    rows_total = int(agree["rows"].sum()) if len(agree) else 0
    matched = int(agree["matched"].sum()) if len(agree) else 0
    status = PASS if rows_total and matched == rows_total else (WARN if matched else FAIL)
    return StepResult("replication", title, status, f"{matched} of {rows_total} compared rows match the authors' "
                      f"tables ({'quick: no permutations, GLMMs or expression models' if quick else 'full'})",
                      rows=rows, seconds=time.monotonic() - t0, details={"zenodo_root": str(root)})


def step_model(ctx: _Ctx) -> StepResult:
    title = "Model server (vbt local check)"
    provider = (ctx.config.get("provider") or {}).get("name")
    if provider in (None, "mock", "anthropic"):
        return StepResult.skipped("model", title, f"the provider is {provider!r}: no local model server to check")
    from ..local.check import options_from_config, run_checks

    opts = options_from_config(ctx.config)
    base = str(getattr(opts, "base_url", "") or (ctx.config.get("provider") or {}).get("options", {}).get("base_url")
               or "")
    if not _reachable(base.rstrip("/") + "/models"):
        return StepResult.skipped("model", title, f"no model server answers at {base or '(no base_url)'}")
    t0 = time.monotonic()
    report = asyncio.run(run_checks(opts))
    data = report.to_json() if hasattr(report, "to_json") else getattr(report, "__dict__", {})
    results = list(getattr(report, "results", []) or [])
    failed = [r for r in results if not getattr(r, "ok", True)]
    rows = [{"check": getattr(r, "name", ""), "ok": getattr(r, "ok", None), "detail": getattr(r, "detail", "")}
            for r in results]
    return StepResult("model", title, FAIL if failed else PASS, f"{len(results) - len(failed)} of {len(results)} "
                      f"checks passed at {base}", rows=rows, seconds=time.monotonic() - t0, details={"report": data})


def _upstream_missing(config: Mapping[str, Any]) -> str | None:
    """Why the upstream servers cannot be started (no checkout at ``vars.upstream``), else None."""
    up = (config.get("vars") or {}).get("upstream")
    if not up or not Path(str(up), "src", "mcp_servers").is_dir():
        return f"the upstream servers are not at {up or '(vars.upstream unset)'} (VBT_UPSTREAM; git submodule update)"
    return None


def _reachable(url: str, timeout: float = 5.0) -> bool:
    import urllib.error
    import urllib.request

    if not url.startswith(("http://", "https://")):
        return False
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True                      # the server answered
    except Exception:  # noqa: BLE001
        return False


_STEP_FUNCS: dict[str, tuple[str, Callable[[_Ctx], StepResult]]] = {
    "host": ("Host and host-scaled limits", step_host),
    "lint": ("Descriptors and overlays (vbt ds lint)", step_lint),
    "check": ("Readiness (vbt ds check)", step_check),
    "correctness": ("The six correctness tests (off vs enforce)", step_correctness),
    "latency": ("Gateway and witness latency per server", step_latency),
    "memory": ("Memory estimate against the measured load", step_memory),
    "live": ("Live sources", step_live),
    "replication": ("Paper replication (Case 1 on the Zenodo archive)", step_replication),
    "model": ("Model server (vbt local check)", step_model),
}


def run_validate(config: Mapping[str, Any], opts: Options | None = None, *,
                 progress: Callable[[str], None] | None = None) -> tuple[ValidationReport, Path]:
    """Run the selected steps in order; returns the report and the directory it was written to."""
    from ..datalayer.memory import sizing

    opts = opts or Options()
    out = Path(opts.out) if opts.out else _default_out()
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.monotonic()
    report = ValidationReport(started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              host=host_facts(sizing.plan_mb(_memory_raw(config))),
                              profiles=list(config.get("_profiles") or []))
    ctx = _Ctx(config, opts, out)
    for name in STEPS:
        if name not in opts.steps:
            continue
        title, fn = _STEP_FUNCS[name]
        if progress:
            progress(f"[validate] {name} ...")
        t0 = time.monotonic()
        try:
            res = fn(ctx)
        except Exception as exc:  # noqa: BLE001 - a broken step is reported, the next steps still run
            res = StepResult(name, title, ERROR, f"{type(exc).__name__}: {exc}"[:500],
                             details={"traceback": traceback.format_exc()[-4000:]})
        res.seconds = res.seconds or (time.monotonic() - t0)
        report.steps.append(res)
        if progress:
            progress(f"[validate] {name}: {res.status} ({res.seconds:,.1f} s) {res.summary[:160]}")
        report.seconds = time.monotonic() - t_all
        report.write(out)
    report.finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report.seconds = time.monotonic() - t_all
    report.write(out)
    return report, out


def _memory_raw(config: Mapping[str, Any]) -> dict[str, Any]:
    mem = ((config.get("data") or {}).get("memory") or {})
    return dict(mem) if isinstance(mem, Mapping) else {}


def _default_out() -> Path:
    try:
        from ..setup.layout import resolve_layout

        state = resolve_layout().state
    except Exception:  # noqa: BLE001
        state = Path("data") / ".vbt-setup"
    return Path(state) / "validate" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


# ---------------------------------------------------------------------------------------------------- CLI


def add_validate_parser(sub: Any) -> argparse.ArgumentParser:
    p = sub.add_parser("validate", help="certify this host: lint, deep check, the six correctness tests on the real "
                       "data, memory and latency, live sources, replication, the model server (markdown + JSON)")
    p.add_argument("--only", help=f"comma-separated steps to run ({', '.join(STEPS)})")
    p.add_argument("--skip", help="comma-separated steps to leave out")
    p.add_argument("--depth", choices=["standard", "deep"], default="deep", help="readiness depth (default deep)")
    p.add_argument("--servers", help="comma-separated MCP servers for the correctness and live steps (default: all)")
    p.add_argument("--max-tools", type=int, default=40, help="tools per server the correctness step generates "
                   "cases for (default 40)")
    p.add_argument("--memory-tables", type=int, default=4, help="largest present tables whose load the memory step "
                   "measures, among those that fit the server limit (default 4; 0 = none)")
    p.add_argument("--replicate", choices=["quick", "full", "off"], default="quick",
                   help="Case 1 replication: quick (no permutations, GLMMs or expression models), full, off")
    p.add_argument("--timeout-s", type=float, default=600, help="per tool call (default 600)")
    p.add_argument("--out", help="report directory (default <state>/validate/<UTC time>)")
    p.add_argument("--check-from", help="reuse the check.json an earlier run wrote (its CheckResponse) instead of "
                   "checking again")
    p.add_argument("--json", action="store_true", help="print the JSON report instead of the Markdown one")
    p.set_defaults(handler=cmd_validate)
    return p


def cmd_validate(args: argparse.Namespace, config: dict[str, Any]) -> int:
    steps = [s.strip() for s in (args.only or ",".join(STEPS)).split(",") if s.strip()]
    skip = {s.strip() for s in (args.skip or "").split(",") if s.strip()}
    unknown = [s for s in [*steps, *skip] if s not in STEPS]
    if unknown:
        print(f"error: unknown step(s) {', '.join(unknown)}; steps: {', '.join(STEPS)}", file=sys.stderr)
        return 2
    opts = Options(steps=[s for s in steps if s not in skip], depth=args.depth,
                   servers=[s for s in (args.servers or "").split(",") if s] or None, max_tools=args.max_tools,
                   memory_tables=args.memory_tables, replicate=args.replicate,
                   out=Path(args.out) if args.out else None, timeout_s=args.timeout_s,
                   check_from=Path(args.check_from) if args.check_from else None)
    config = dict(config)
    config.setdefault("_profiles", list(getattr(args, "profile", None) or []))
    report, out = run_validate(config, opts, progress=lambda m: print(m, file=sys.stderr, flush=True))
    if args.json:
        print(json.dumps(report.to_json(), indent=1, sort_keys=True, default=str))
    else:
        print(report.to_markdown())
    print(f"report: {out / 'validate.md'} and {out / 'validate.json'}", file=sys.stderr)
    return 0 if report.ok else 1
