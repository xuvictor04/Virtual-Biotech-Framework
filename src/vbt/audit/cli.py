"""Audit subcommands: ``vbt verify | list | index | export | audit | show``.

``add_audit_parsers(sub)`` registers them on the main ``vbt`` subparsers; every
handler has the signature ``handler(args, config)`` and resolves its run
argument with ``vbt.audit.index.resolve_run`` against ``config.paths.runs_dir``
(a path, a run id, a unique id prefix, the id's hex suffix, or ``latest``).

Every command that reads runs takes ``--project NAME``: a project's runs are under
``<project>/runs`` (``projects.runs_in_project``) and ``verify --data`` re-reads the
project's own tables, so the project is activated as ``vbt chat --project`` does.

Exit codes: 0 success; ``verify`` returns 0 only for COMPLETE (1 for INCOMPLETE
or FAIL); 2 when the run cannot be resolved or the command fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .index import AmbiguousRunError, RunNotFound, resolve_run, runs_dir_from_config


def _runs_dir(config: dict[str, Any]) -> Path:
    return runs_dir_from_config(config)


def _resolve(args, config) -> Path | None:
    try:
        return resolve_run(args.run, _runs_dir(config))
    except (RunNotFound, AmbiguousRunError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


# ----------------------------------------------------------------- handlers

def cmd_verify(args, config) -> int:
    from ..verify import format_report, verify_run

    run = _resolve(args, config)
    if run is None:
        return 2
    data = bool(getattr(args, "data", False))
    report = verify_run(run, rerun=args.rerun, python=args.python, timeout=args.timeout, data=data,
                        config=config if data else None, backend=getattr(args, "backend", "auto"))
    if args.json:
        _print_json(report)
    else:
        print(format_report(report))
        if (run / "audit.html").is_file():
            print(f"Audit report: {run / 'audit.html'}")
    return 0 if report.get("status") == "COMPLETE" else 1


def cmd_list(args, config) -> int:
    from .index import scan_runs

    rows = scan_runs(_runs_dir(config))
    if args.limit:
        rows = rows[: args.limit]
    if args.json:
        _print_json(rows)
        return 0
    if not rows:
        print(f"No runs in {_runs_dir(config)}")
        return 0
    print(f"{'RUN':34s} {'STATUS':13s} {'STARTED':16s} {'TURNS':>5s} {'ARTS':>5s} {'CLAIMS':>6s} {'COST':>8s}  QUERY")
    for r in rows:
        q = " ".join(str(r.get("query") or "").split())
        q = q[:60] + ("…" if len(q) > 60 else "")
        cost = f"${r['cost_usd']:.2f}" if r.get("cost_usd") is not None else "-"
        started = str(r.get("created") or "")[:16].replace("T", " ")
        print(f"{str(r['run_id'])[:34]:34s} {str(r['status'])[:13]:13s} {started:16s} {r.get('n_turns') or 0:5d} "
              f"{r.get('n_artifacts') or 0:5d} {r.get('n_claims') or 0:6d} {cost:>8s}  {q}")
    return 0


def cmd_index(args, config) -> int:
    from .index import INDEX_JSON, INDEX_MD, update_index

    root = _runs_dir(config)
    rows = update_index(root)
    print(f"Indexed {len(rows)} run(s): {root / INDEX_MD} and {root / INDEX_JSON}")
    return 0


def cmd_export(args, config) -> int:
    from .export import ExportError, excluded_members, export_chat_markdown, export_run

    run = _resolve(args, config)
    if run is None:
        return 2
    try:
        if args.chat:
            text = export_chat_markdown(run)
            if args.output in (None, "-"):
                sys.stdout.write(text)
                return 0
            out = Path(args.output).expanduser()
            from .storage import write_text_atomic
            write_text_atomic(out, text)
            print(f"Chat transcript: {out}")
            return 0
        path = export_run(run, args.output, no_data=args.no_data, max_file_mb=args.max_file_mb)
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    size = path.stat().st_size
    print(f"Run bundle: {path} ({size / (1024 * 1024):.1f} MB)")
    left_out = excluded_members(path)
    if left_out:
        print(f"Left out {len(left_out)} file(s) (listed in logs/export_excluded.json inside the zip):")
        for e in left_out[:10]:
            print(f"  {e['path']}  ({e['bytes'] / (1024 * 1024):.1f} MB, {e['reason']})")
        if len(left_out) > 10:
            print(f"  … and {len(left_out) - 10} more")
    return 0


def cmd_audit(args, config) -> int:
    from .retrofit import AuditError, audit_all, audit_run

    if args.all == bool(args.run):
        print("error: give a RUN or --all", file=sys.stderr)
        return 2
    out_dir = Path(args.out).expanduser() if args.out else None
    if args.all:
        results = audit_all(_runs_dir(config), in_place=args.in_place, out_dir=out_dir, link=args.link)
    else:
        run = _resolve(args, config)
        if run is None:
            return 2
        try:
            results = [audit_run(run, in_place=args.in_place, out_dir=out_dir, link=args.link)]
        except AuditError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    if args.json:
        _print_json(results)
    else:
        for r in results:
            if "error" in r:
                print(f"[failed] {r['run_id']}: {r['error']}")
                continue
            att = ", ".join(f"{k} {v}" for k, v in r["attribution"].items() if v) or "no artifacts"
            print(f"[ok] {r['run_id']}: status {r['status']} (was {r['status_before'] or 'unrecorded'}); "
                  f"{r['n_artifacts']} artifact(s) [{att}]; {r['n_tool_calls']} tool call(s); {r['n_claims']} claim(s)")
            if r.get("reconstructed_turns"):
                print(f"     reconstructed turn record(s): {', '.join(map(str, r['reconstructed_turns']))}")
            if r.get("html"):
                print(f"     -> {r['html']}")
    if args.in_place and not args.all:
        try:
            from .index import update_index
            update_index(_runs_dir(config))
        except OSError:
            pass
    return 0 if all("error" not in r for r in results) else 1


def cmd_show(args, config) -> int:
    from .export import format_file_listing, list_session_files
    from .index import summarize_run

    run = _resolve(args, config)
    if run is None:
        return 2
    summary = summarize_run(run)
    listing = list_session_files(run) if args.files else None
    if args.json:
        _print_json({"run": summary, **({"files": listing} if listing is not None else {})})
        return 0
    print(f"Run:      {summary['run_id']}  ({summary['status']})")
    if summary.get("query"):
        print(f"Query:    {' '.join(str(summary['query']).split())[:200]}")
    print(f"Started:  {summary.get('created') or '-'}")
    cost = f"${summary['cost_usd']:.2f}" if summary.get("cost_usd") is not None else "-"
    print(f"Turns:    {summary['n_turns']}   cost {cost}   specialists: {', '.join(summary['agents']) or '-'}")
    print(f"Evidence: {summary['n_artifacts']} artifact(s), {summary['n_claims']} claim(s)"
          + (f", {summary['n_unresolved_evidence']} unresolved evidence item(s)"
             if summary.get("n_unresolved_evidence") else ""))
    print(f"Path:     {run}")
    for name in ("README.md", "audit.html"):
        if (run / name).is_file():
            print(f"          {run / name}")
    if listing is not None:
        print()
        print(format_file_listing(listing))
    return 0


# ----------------------------------------------------------------- parsers

def add_audit_parsers(sub) -> None:
    """Register verify, list, index, export, audit and show on ``vbt``'s subparsers (each with ``--project``)."""
    from ..projects.cli import add_project_argument

    v = sub.add_parser("verify", help="check a run's artifact integrity and evidence coverage",
                       description="Exit status 0 only when the run verifies COMPLETE.")
    v.add_argument("run", help="run directory, run id, unique id prefix, or 'latest'")
    v.add_argument("--rerun", action="store_true", help="re-execute agent scripts in a scratch copy")
    v.add_argument("--python", metavar="EXE", help="interpreter for --rerun (default: this one)")
    v.add_argument("--timeout", type=float, default=600, metavar="S", help="per-script timeout for --rerun")
    v.add_argument("--json", action="store_true", help="print the machine-readable report")
    v.add_argument("--data", action="store_true",
                   help="check reference data: current table fingerprints (data_version_drift) and a replay of "
                        "every cited data call (replay_mismatch, source_updated)")
    v.add_argument("--backend", choices=("auto", "inprocess", "bridge"), default="auto",
                   help="--data: how cited calls are replayed (see `vbt ds replay`)")
    add_project_argument(v)
    v.set_defaults(handler=cmd_verify)

    ls = sub.add_parser("list", help="list runs (newest first)")
    ls.add_argument("--json", action="store_true")
    ls.add_argument("--limit", type=int, metavar="N")
    add_project_argument(ls)
    ls.set_defaults(handler=cmd_list)

    ix = sub.add_parser("index", help="rebuild runs/INDEX.md and runs/INDEX.json")
    add_project_argument(ix)
    ix.set_defaults(handler=cmd_index)

    ex = sub.add_parser("export", help="zip a run (or write its chat transcript)")
    ex.add_argument("run")
    ex.add_argument("-o", "--output", metavar="FILE", help="output path (default: runs/.downloads/<run>.zip; "
                                                            "with --chat the transcript goes to stdout)")
    ex.add_argument("--no-data", action="store_true",
                    help="leave out large raw data (*.h5ad, *.h5, *.loom; *.parquet over the size cap)")
    ex.add_argument("--max-file-mb", type=float, metavar="N", help="leave out analysis files larger than N MB")
    ex.add_argument("--chat", action="store_true", help="export the conversation as Markdown instead")
    add_project_argument(ex)
    ex.set_defaults(handler=cmd_export)

    au = sub.add_parser("audit", help="rebuild a run's MANIFEST, provenance and README/audit.html from its trace")
    au.add_argument("run", nargs="?", help="run directory, id or prefix")
    au.add_argument("--all", action="store_true", help="audit every run in the runs directory")
    au.add_argument("--in-place", action="store_true", help="rewrite the run's own records (default: a copy)")
    au.add_argument("-o", "--out", metavar="DIR", help="where audit copies go (default: <runs>/../audits)")
    au.add_argument("--link", action="store_true", help="hard-link files into the copy instead of copying")
    au.add_argument("--json", action="store_true")
    add_project_argument(au)
    au.set_defaults(handler=cmd_audit)

    sh = sub.add_parser("show", help="summarise a run; --files lists its figures, tables and code by agent")
    sh.add_argument("run")
    sh.add_argument("--files", action="store_true", help="list figures, tables and code grouped by agent")
    sh.add_argument("--json", action="store_true")
    add_project_argument(sh)
    sh.set_defaults(handler=cmd_show)


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point (``python -m vbt.audit.cli``) using the default config."""
    from ..config import load_config
    from ..projects import ProjectError
    from ..projects.cli import apply_project

    p = argparse.ArgumentParser(prog="python -m vbt.audit.cli")
    p.add_argument("--profile", action="append", default=[])
    p.add_argument("--runs-dir")
    sub = p.add_subparsers(dest="cmd", required=True)
    add_audit_parsers(sub)
    args = p.parse_args(argv)
    overrides = {"paths": {"runs_dir": args.runs_dir}} if args.runs_dir else None
    try:
        config = apply_project(args, load_config(args.profile, overrides))
    except ProjectError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return args.handler(args, config)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
