"""``vbt project``: create, inspect, check and approve projects (docs/PROJECTS.md).

* ``vbt project init NAME [--root DIR] [--description TEXT] [--review none|reviewer|human]``
* ``vbt project list`` / ``vbt project show NAME`` (``--json``)
* ``vbt project check NAME [--tests]``: every registered file against its provenance record, the project's
  descriptors and overlays linted with the shipped catalog, the generated profile up to date; ``--tests`` re-runs
  every utility's tests in the sandbox
* ``vbt project profile NAME``: (re)write ``profile.yaml`` and print the ``--profile`` argument
* ``vbt project approve|reject NAME KIND ITEM [--by WHO] [--notes TEXT]``: decide an item waiting for review
* ``vbt project memory NAME --from-run RUN [--agent A]``: append a run's agent notes to the project notes

:func:`add_project_parser` registers the command; :func:`add_project_argument` and :func:`apply_project` give a
session command ``--project NAME``. ``python -m vbt.projects ...`` runs the same commands.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from . import ledger
from .model import (
    REVIEW_MODES,
    Project,
    ProjectError,
    activate,
    init_project,
    list_projects,
    profile_is_current,
    projects_root,
    resolve_project,
    write_profile,
)

__all__ = ["add_project_parser", "add_project_argument", "apply_project", "main", "check_project"]


def _out(text: str) -> None:
    print(text, flush=True)


def _err(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def add_project_argument(p: argparse.ArgumentParser) -> None:
    """``--project NAME`` for a session command (chat, run, setup)."""
    p.add_argument("--project", metavar="NAME", help="run in a project (vbt project init NAME): its descriptors, "
                                                     "overlays, plugins, utilities, skills, notes and runs")


def apply_project(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    """``config`` with ``--project`` activated (unchanged without the flag)."""
    ref = getattr(args, "project", None)
    if not ref:
        return dict(config)
    return activate(config, resolve_project(ref, config))


def _project(args: argparse.Namespace, config: Mapping[str, Any]) -> Project:
    return resolve_project(args.name, config)


def cmd_init(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    project = init_project(args.name, config=config, root=args.root, description=args.description or "",
                           review=args.review)
    _out(f"project {project.name} created at {project.root}")
    _out(f"activate it with: vbt --profile {project.profile_path} chat   (or --project {project.name})")
    return 0


def cmd_list(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    rows = []
    for p in list_projects(config):
        counts = {k: sum(1 for r in ledger.records(p, [k]) if r.get("status") == "registered")
                  for k in ledger.ITEM_KINDS}
        rows.append({**p.summary(), "items": counts})
    if args.json:
        _out(json.dumps({"root": str(projects_root(config)), "projects": rows}, indent=1, default=str))
        return 0
    _out(f"projects under {projects_root(config)}:")
    for r in rows:
        items = ", ".join(f"{n} {k}(s)" for k, n in r["items"].items() if n) or "nothing registered yet"
        _out(f"  {r['name']:<24} {items}  {r['description'][:60]}")
    if not rows:
        _out("  (none; create one with `vbt project init NAME`)")
    return 0


def cmd_show(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    from .authoring import pending_items
    from .prompt import resources

    p = _project(args, config)
    body = {**p.summary(), "resources": resources(p), "pending": pending_items(p),
            "profile": str(p.profile_path), "ledger_events": len(ledger.read_ledger(p))}
    if args.json:
        _out(json.dumps(body, indent=1, default=str))
        return 0
    _out(f"project {p.name} ({p.root})")
    for kind, items in body["resources"].items():
        if items:
            _out(f"  {kind}: " + ", ".join(f"{i['name']}" + (f" v{i['version']}" if i.get("version") else "")
                                         for i in items))
    for item in body["pending"]:
        _out(f"  pending review: {item['kind']} {item['name']} (by {item.get('who')}, {item.get('when')})")
    _out(f"  ledger: {body['ledger_events']} event(s); profile: {body['profile']}")
    return 0


def check_project(project: Project, config: Mapping[str, Any]) -> dict[str, Any]:
    """Records against files, lint of the project's catalog files, and the profile."""
    from ..datalayer.catalog import build_catalog
    from ..datalayer.descriptor.lint import lint_descriptor, lint_overlay
    from ..datalayer.descriptor.load import variables_from_config
    from ..datalayer.plugins.registry import discover
    from ..datalayer.settings import DataSettings

    problems: list[str] = []
    for rec in ledger.records(project):
        for p in ledger.verify(project, rec):
            problems.append(f"{rec.get('kind')} {rec.get('name')}: {p}")
    for sub, kind in (("descriptors", "descriptor"), ("overlays", "overlay")):
        for f in sorted((project.root / sub).glob("*.y*ml")):
            if not ledger.read_record(project, kind, f.stem):
                problems.append(f"{sub}/{f.name}: no provenance record (not registered through the authoring tools)")
    cfg = activate(config, project, runs=False)
    settings = DataSettings.from_config(cfg)
    lint: list[str] = []
    try:
        registry = discover(settings)
        catalog = build_catalog(settings, registry, variables=variables_from_config(cfg))
        for q in catalog.project_refused:
            problems.append(f"refused {q.kind} {q.file}: {q.summary}")
        for q in catalog.quarantined:
            if str(project.root) in q.path:
                problems.append(f"quarantined {q.kind} {q.file}: {q.summary}")
        for name in sorted(catalog.project_sources):
            lint += [str(f) for f in lint_descriptor(catalog.sources[name], registry, None, catalog.sources)]
        for name in sorted(catalog.project_servers):
            lint += [str(f) for f in lint_overlay(catalog.overlays[name], catalog, registry)]
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        problems.append(f"the catalog with the project does not build: {type(exc).__name__}: {exc}")
    problems += [f"lint {f}" for f in lint if f.startswith("error")]
    if not profile_is_current(project, config):
        problems.append(f"{project.profile_path.name} is out of date: run `vbt project profile {project.name}`")
    return {"project": project.name, "ok": not problems, "problems": problems, "lint": lint}


async def _rerun_tests(project: Project, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    import tempfile

    from ..tools.policy import PathPolicy
    from .sandbox import run_sandboxed, runner_argv

    out = []
    with tempfile.TemporaryDirectory(prefix="vbt-project-check-") as tmp:
        run_dir = Path(tmp)
        policy = PathPolicy(run_dir=run_dir, workspace=run_dir / "work" / "_project", agent="_project")
        for rec in ledger.records(project, ["utility"]):
            if rec.get("status") != "registered":
                continue
            res = await run_sandboxed(runner_argv(None, "test", str(project.utilities_dir / str(rec["name"]))),
                                      policy=policy, cwd=policy.own_dir, config=config,
                                      label=f"check_{rec['name']}", timeout_s=600, network=False)
            result = res.result if isinstance(res.result, dict) else {}
            out.append({"utility": rec["name"], "ok": res.ok and not result.get("failed") and
                        int(result.get("passed") or 0) > 0, "passed": result.get("passed"),
                        "failed": result.get("failed"), "sandbox": res.sandbox})
    return out


def cmd_check(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    p = _project(args, config)
    report = check_project(p, config)
    if args.tests:
        report["tests"] = asyncio.run(_rerun_tests(p, config))
        bad = [t for t in report["tests"] if not t["ok"]]
        report["problems"] += [f"utility {t['utility']}: tests failed ({t.get('failed')})" for t in bad]
        report["ok"] = not report["problems"]
    if args.json:
        _out(json.dumps(report, indent=1, default=str))
    else:
        for line in report["problems"]:
            _out(f"problem: {line}")
        for line in report["lint"]:
            if not line.startswith("error"):
                _out(line)
        n = len(report["problems"])
        _out(f"project {p.name}: " + ("ok" if report["ok"] else f"{n} problem(s)"))
    return 0 if report["ok"] else 1


def cmd_profile(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    p = _project(args, config)
    path = write_profile(p, config)
    _out(str(path))
    return 0


def cmd_approve(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    from .authoring import approve_pending, reject_pending

    p = _project(args, config)
    by = args.by or os.environ.get("USER") or "owner"
    if args.decision == "approve":
        rec = approve_pending(p, args.kind, args.item, by=by, notes=args.notes or "", config=config)
        _out(f"{args.kind} {args.item} v{rec.get('version')} approved by {by} and registered")
    else:
        reject_pending(p, args.kind, args.item, by=by, notes=args.notes or "")
        _out(f"{args.kind} {args.item} rejected by {by}")
    return 0


def cmd_memory(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    from ..agents import memory_path
    from ..audit.index import resolve_run

    from ..config import resolve_path

    p = _project(args, config)
    run_dir = None
    for base in (p.runs_dir, resolve_path((config.get("paths") or {}).get("runs_dir") or "runs")):
        try:
            run_dir = Path(resolve_run(args.from_run, base))
            break
        except Exception:  # noqa: BLE001 - RunNotFound / AmbiguousRunError: try the next directory
            continue
    if run_dir is None:
        raise ProjectError(f"no run {args.from_run!r} under {p.runs_dir} or the runs directory")
    agents = [args.agent] if args.agent else sorted(d.name for d in (run_dir / "memory").glob("*") if d.is_dir())
    added = 0
    for agent in agents:
        src = memory_path(run_dir, agent)
        if not src.is_file():
            continue
        dest = p.memory_path(agent)
        dest.parent.mkdir(parents=True, exist_ok=True)
        old = dest.read_text(encoding="utf-8") if dest.is_file() else ""
        seen = set(old.splitlines())
        new = [ln for ln in src.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip() and
               ln not in seen]
        if new:
            dest.write_text(old + ("" if not old or old.endswith("\n") else "\n")
                            + f"<!-- from run {run_dir.name} -->\n" + "\n".join(new) + "\n", encoding="utf-8")
            added += len(new)
            ledger.append_ledger(p, {"event": "memory", "kind": "memory", "name": agent, "who": {"user": "cli"},
                                     "why": f"notes from run {run_dir.name}", "lines": len(new)})
    _out(f"{added} new line(s) of notes added to project {p.name}")
    return 0


def add_project_parser(sub: Any) -> Any:
    """Register ``vbt project`` on an argparse subparsers object."""
    d = sub.add_parser("project", help="projects: what the system creates for one project (docs/PROJECTS.md)")
    _add_commands(d.add_subparsers(dest="project_cmd", required=True))
    return d


def _add_commands(ps: Any) -> None:
    p = ps.add_parser("init", help="create a project directory")
    p.add_argument("name")
    p.add_argument("--root", help="projects directory (default: projects.root, else $VBT_PROJECTS_DIR, "
                                  "<VBT_HOME>/projects or <data>/projects)")
    p.add_argument("--description")
    p.add_argument("--review", choices=REVIEW_MODES, help="who approves registrations in this project (at least "
                                                          "the host's projects.review)")
    p.set_defaults(handler=cmd_init)

    p = ps.add_parser("list", help="projects and what they registered")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=cmd_list)

    p = ps.add_parser("show", help="a project's resources, pending items and ledger")
    p.add_argument("name")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=cmd_show)

    p = ps.add_parser("check", help="files against provenance, lint, profile (and --tests)")
    p.add_argument("name")
    p.add_argument("--tests", action="store_true", help="re-run every utility's tests in the sandbox")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=cmd_check)

    p = ps.add_parser("profile", help="(re)write the project's profile.yaml and print its path")
    p.add_argument("name")
    p.set_defaults(handler=cmd_profile)

    for decision in ("approve", "reject"):
        p = ps.add_parser(decision, help=f"{decision} an item waiting for review (projects.review: human)")
        p.add_argument("name")
        p.add_argument("kind", choices=ledger.ITEM_KINDS)
        p.add_argument("item")
        p.add_argument("--by", help="who decides (default $USER)")
        p.add_argument("--notes")
        p.set_defaults(handler=cmd_approve, decision=decision)

    p = ps.add_parser("memory", help="append a run's agent notes to the project's notes")
    p.add_argument("name")
    p.add_argument("--from-run", required=True, metavar="RUN", help="run id, prefix, path or 'latest'")
    p.add_argument("--agent", help="one agent (default: every agent with notes)")
    p.set_defaults(handler=cmd_memory)


def main(argv: list[str] | None = None) -> int:
    """``python -m vbt.projects [--profile P ...] <command> ...`` (the commands of ``vbt project``)."""
    from ..config import load_config

    parser = argparse.ArgumentParser(prog="python -m vbt.projects", description="Virtual Biotech projects "
                                                                                "(docs/PROJECTS.md)")
    parser.add_argument("--profile", action="append", default=[])
    _add_commands(parser.add_subparsers(dest="project_cmd", required=True))
    args = parser.parse_args(argv)
    config = load_config(args.profile)
    config["profiles"] = list(args.profile)
    try:
        return int(args.handler(args, config))
    except ProjectError as exc:
        _err(f"error: {exc}")
        return 2
