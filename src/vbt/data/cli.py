"""``vbt data acquire`` and ``vbt data status`` (registered under ``vbt data`` by :func:`add_acquire_parsers`).

::

    vbt data acquire SOURCE[.TABLE] ...            the tables (or extra download groups) named
    vbt data acquire --for-tools TOOL ...          every table those tools read (server.tool, mcp__s__t, globs)
    vbt data acquire --for-agents AGENT ...        every table the agents' tools read (configs/agents.yaml)
    vbt data acquire --all | --missing | --pending every declared source | what readiness reports missing |
                                                   what data.acquisition.auto queued for approval
        [--dest ROOT] [--plan] [--offline] [--max-gb N] [--workers N] [--no-prepare] [--include-optional]
        [--env-file .env] [--json]
    vbt data status [SOURCE ...] [--check] [--json] [-v]

``acquire`` prints the plan (files, bytes, time at the measured rate, prepare steps, licence and login notes,
the variables that point the data layer at the files) and, without ``--plan``, carries it out: parallel resumable
downloads, verification, manifests, prepare steps, the pinned release in ``<home>/.vbt-acquisition.json`` and a
provenance record.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

__all__ = ["add_acquire_parsers", "cmd_acquire", "cmd_status"]


def add_acquire_parsers(ds: Any) -> None:
    a = ds.add_parser("acquire", help="acquire the files of declared tables (descriptor acquisition sections): "
                                      "plan, download resumably, verify, prepare, write manifests")
    a.add_argument("targets", nargs="*", metavar="SOURCE[.TABLE]",
                   help="sources, tables, or extra download groups (SOURCE.GROUP)")
    a.add_argument("--for-tools", nargs="+", default=[], metavar="TOOL",
                   help="tables these tools read (server.tool, mcp__server__tool, globs such as target.*)")
    a.add_argument("--for-agents", nargs="+", default=[], metavar="AGENT",
                   help="tables the tools of these agents read (configs/agents.yaml)")
    a.add_argument("--all", action="store_true", help="every source that declares an acquisition section")
    a.add_argument("--missing", action="store_true",
                   help="the tables the readiness cache reports missing, partial or stale")
    a.add_argument("--pending", action="store_true",
                   help="the tables data.acquisition.auto queued for approval")
    a.add_argument("--dest", metavar="ROOT", help="acquisition root (default: data.acquisition.root)")
    a.add_argument("--plan", action="store_true", help="print the plan and stop")
    a.add_argument("--offline", action="store_true", help="plan from the declared sizes, without listing")
    a.add_argument("--max-gb", type=float, help="refuse when more than this would be transferred")
    a.add_argument("--workers", type=int, help="parallel transfers (default: data.acquisition.workers)")
    a.add_argument("--no-prepare", action="store_true", help="do not run prepare steps")
    a.add_argument("--include-optional", action="store_true",
                   help="with a bare SOURCE, also the extra groups marked optional")
    a.add_argument("--env-file", metavar="PATH",
                   help="write the variables that point the data layer at the files (KEY=\"value\" lines)")
    a.add_argument("--json", action="store_true", help="the plan (and report) as JSON")
    a.set_defaults(handler=cmd_acquire)
    s = ds.add_parser("status", help="every declared table: present, verified, ready, and the tools it unlocks")
    s.add_argument("sources", nargs="*", metavar="SOURCE")
    s.add_argument("--check", action="store_true", help="run the readiness check (vbt ds check) on present tables")
    s.add_argument("--json", action="store_true")
    s.add_argument("--tools", action="store_true", help="name the tools each table unlocks")
    s.set_defaults(handler=cmd_status)


def _err(text: str) -> None:
    print(text, file=sys.stderr)


def cmd_acquire(args: Any, config: Mapping[str, Any]) -> int:
    from ..datalayer.plugins.base import AcquisitionError
    from ..preflight import data_catalog
    from . import acquire as A
    from .ondemand import clear_pending, missing_tables, pending
    from .targets import Targets, resolve_targets

    settings = A.AcquisitionSettings.from_config(config)
    _s, catalog, _r = data_catalog(dict(config))
    names = list(args.targets)
    if args.all:
        names += [s for s, d in sorted(catalog.sources.items()) if d.acquisition is not None
                  and d.acquisition.mode == "download"]
    targets: Targets = resolve_targets(catalog, config, names, tools=args.for_tools, agents=args.for_agents,
                                       include_optional=args.include_optional)
    extra: dict[str, list[str]] = {}
    if args.missing:
        extra = missing_tables(config)
    if args.pending:
        for s, ts in pending(settings).items():
            extra.setdefault(s, []).extend(t for t in ts if t not in extra.get(s, []))
    for s, ts in extra.items():
        for t in ts:
            targets.add(catalog, f"{s}.{t}", "missing" if args.missing else "pending")
    for e in targets.errors:
        _err(f"error: {e}")
    if targets.errors:
        return 2
    if not targets.wanted:
        if not (names or args.for_tools or args.for_agents or args.missing or args.pending):
            _err("error: name sources or tables, or give --for-tools, --for-agents, --all, --missing or --pending")
            return 2
        if args.json and args.plan:
            # the same document as a plan with work in it, so a caller parses one shape
            print(json.dumps({"root": str(Path(args.dest).expanduser().resolve() if args.dest else settings.root),
                              "bytes_remaining": 0, "seconds": 0, "sources": [], "notes": list(targets.notes),
                              "why": targets.why}, indent=1, default=str, sort_keys=True))
            return 0
        for n in targets.notes:
            print(f"note: {n}")
        print("nothing to acquire")
        return 0
    root = Path(args.dest).expanduser().resolve() if args.dest else settings.root
    try:
        plan = A.plan_acquisition(catalog, targets.wanted, settings, root=root, offline=args.offline,
                                  write_index=not args.plan)
    except AcquisitionError as exc:
        _err(f"error: {exc}")
        return 2
    plan.notes.extend(targets.notes)
    if args.json and args.plan:
        body = plan.to_dict()
        body["why"] = targets.why
        print(json.dumps(body, indent=1, default=str, sort_keys=True))
        return 1 if any(s.listing_error for s in plan.sources) else 0
    for line in plan.lines():
        print(line)
    if (args.for_tools or args.for_agents) and getattr(args, "verbose", False):
        for ref, why in sorted(targets.why.items()):
            print(f"#   {ref}: {', '.join(why[:6])}" + (f" (+{len(why) - 6})" if len(why) > 6 else ""))
    if args.plan:
        return 1 if any(s.listing_error for s in plan.sources) else 0
    blocked = [s for s in plan.sources if s.mode == "download" and s.groups and s.listing_error]
    for s in blocked:
        _err(f"error: {s.source}: {s.listing_error}")
    verbose = bool(getattr(args, "verbose", False))

    def event(kind: str, info: dict[str, Any]) -> None:
        if kind == "file" and (verbose or info.get("status") == "failed"):
            print(f"  {info.get('status'):10s} {info.get('bytes') or 0:>14,d}  {info.get('path')}"
                  + (f"  {info.get('error')}" if info.get("error") else ""), flush=True)
        elif kind == "progress":
            print(f"  ... {info['source']}: {info['done']} of {info['of']} file(s), {A.fmt_bytes(info['received'])} "
                  "received", flush=True)
        elif kind == "prepare":
            print(f"  prepare {info['step']}: {' '.join(info['argv'])}", flush=True)

    try:
        report = A.execute(plan, settings, workers=args.workers,
                           max_bytes=int(args.max_gb * 1e9) if args.max_gb is not None else None,
                           run_prepare=not args.no_prepare, config=config, on_event=event)
    except ValueError as exc:
        _err(f"error: {exc}")
        return 2
    A.record_provenance(settings, report, by="cli", extra={"wanted": targets.wanted, "argv": sys.argv[1:]})
    if args.pending and report.ok:
        clear_pending(settings, targets.wanted)
    env: dict[str, str] = {}
    for s in report.sources:
        if s.ok:
            env.update(s.env)
    if args.env_file and env:
        A.write_env_file(Path(args.env_file), env)
        print(f"# wrote {', '.join(env)} to {args.env_file}")
    if args.json:
        print(json.dumps(A.report_dict(report), indent=1, default=str, sort_keys=True))
    else:
        for line in report.lines():
            print(line)
    return 0 if report.ok and not blocked else 1


def cmd_status(args: Any, config: Mapping[str, Any]) -> int:
    from .status import collect_status, status_dict, status_lines

    statuses = collect_status(config, sources=args.sources, check=args.check)
    if args.sources:
        known = {s.source for s in statuses}
        unknown = [s for s in args.sources if s not in known]
        if unknown:
            _err(f"error: unknown source(s): {', '.join(unknown)}")
            return 2
    if args.json:
        print(json.dumps(status_dict(statuses), indent=1, default=str, sort_keys=True))
        return 0
    for line in status_lines(statuses, verbose=args.tools):
        print(line)
    return 0
