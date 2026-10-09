"""Validation and registration of what the system creates in a project (docs/PROJECTS.md).

:class:`Author` is used by the agent tools (``vbt.tools.utilities``) and by ``vbt project``. Nothing is registered
unless it passed its kind's validation, and every attempt is recorded in the project's ledger:

* **data specs** (a descriptor, an overlay, or an acquisition spec merged into a project descriptor): parsed and
  model-validated, linted (``vbt ds lint``'s rules: :func:`~vbt.datalayer.descriptor.lint.lint_descriptor` /
  ``lint_overlay`` against the shipped catalog plus the project's) and checked (``vbt ds check``: the data child's
  ``--check`` at ``projects.check_depth`` under its memory limit) on a staged copy of the project. Lint errors,
  check errors, a table that is not ready, a refused or quarantined file, or a name the core ships refuse it.
  A table whose files are missing passes only when the descriptor says how to acquire them;
* **plugins** of an existing kind: a module whose ``@register`` class declares a literal ``name``; a name another
  plugin of the kind already has is refused; the kind's conformance suite runs on it in the sandbox
  (:mod:`vbt.projects.sandbox`), and its conformance stamp is written on success
  (``data.plugins.require_conformance`` accepts it);
* **utilities**: ``utility.py`` (a function ``entry`` with a docstring, or a script with a module docstring), a JSON
  schema for its arguments, and ``test_utility.py`` with ``test_*`` functions, which must all pass in the sandbox.

``projects.review`` (and the project's own ``settings.review``, which may only be stricter) decides who approves:
``none``; ``reviewer``: the ``reviewer`` callable (the scientific reviewer, in a session) must answer APPROVE;
``human``: the item waits under ``provenance/pending/`` for ``vbt project approve``. Every file lands under the
project directory only (:meth:`Project.inside`); staging copies live in ``.staging/`` (data specs) or the calling
agent's own work directory (code), and are removed afterwards.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import fcntl
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

import yaml

from . import ledger
from .model import ITEM_NAME_RE, Project, ProjectError, ProjectSettings, now_iso, write_profile
from .sandbox import run_sandboxed, runner_argv

__all__ = ["Outcome", "Review", "Author", "UTILITY_TOOL_PREFIX", "DATA_SPEC_KINDS", "UTILITY_MODES",
           "utility_manifest", "approve_pending", "reject_pending", "pending_items", "unsafe_spec"]

UTILITY_TOOL_PREFIX = "util__"
DATA_SPEC_KINDS = ("descriptor", "overlay", "acquisition")
UTILITY_MODES = ("function", "script")
UTILITY_FILE = "utility.py"
TESTS_FILE = "test_utility.py"
MANIFEST_FILE = "utility.json"
MIN_DESCRIPTION = 20

Reviewer = Callable[[dict[str, Any]], Awaitable["Review"]]


@dataclass
class Review:
    verdict: str                      # approve | reject
    by: str
    notes: str = ""
    invocation_id: str | None = None


@dataclass
class Outcome:
    """The result of one registration attempt."""

    ok: bool
    status: str                       # registered | pending_review | refused
    kind: str
    name: str
    message: str
    record: dict[str, Any] | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        out = {"ok": self.ok, "status": self.status, "kind": self.kind, "name": self.name, "message": self.message}
        if self.record:
            out["record"] = {k: self.record.get(k) for k in ("version", "source_hash", "files", "when", "status")}
        out.update({k: v for k, v in self.details.items() if v not in (None, [], {})})
        return out


def _refused(kind: str, name: str, message: str, **details: Any) -> Outcome:
    return Outcome(False, "refused", kind, name, message, details=details)


# ---------------------------------------------------------------------------- locking


@contextlib.asynccontextmanager
async def _locked(project: Project):
    """One registration at a time per project, across processes (``.lock`` under the project)."""
    path = project.root / ".lock"
    fh = open(path, "a+")  # noqa: SIM115 - held for the block
    try:
        await asyncio.to_thread(fcntl.flock, fh, fcntl.LOCK_EX)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


# ---------------------------------------------------------------------------- the author


class Author:
    """Registers items in ``project`` on behalf of ``who`` (``{agent, run_id, invocation_id, tool_use_id}`` in a
    session, ``{user}`` from the CLI).

    ``policy`` (a :class:`~vbt.tools.policy.PathPolicy`) and ``work_dir`` place the code sandbox: the staging
    copies are written under ``work_dir`` (the calling agent's own work directory, the only writable place under
    bwrap). ``reviewer`` answers ``projects.review: reviewer``."""

    def __init__(self, project: Project, config: Mapping[str, Any], who: Mapping[str, Any], *, policy: Any,
                 work_dir: Path, reviewer: Reviewer | None = None) -> None:
        self.project = project
        self.config = dict(config)
        self.who = dict(who)
        self.policy = policy
        self.work_dir = Path(work_dir)
        self.reviewer = reviewer
        self.settings = ProjectSettings.from_config(config, project)

    # ------------------------------------------------------------------ shared

    def _event(self, outcome: Outcome, why: str, **extra: Any) -> None:
        ledger.append_ledger(self.project, {"event": outcome.status, "kind": outcome.kind, "name": outcome.name,
                                            "who": self.who, "why": why, "message": outcome.message[:2000],
                                            **({"source_hash": outcome.record.get("source_hash")}
                                               if outcome.record else {}), **extra})

    def _finish(self, outcome: Outcome, why: str) -> Outcome:
        self._event(outcome, why)
        return outcome

    def _size_ok(self, kind: str, name: str, **texts: str) -> Outcome | None:
        for label, text in texts.items():
            if len(text.encode("utf-8")) > self.settings.max_source_bytes:
                return _refused(kind, name, f"{label} is larger than projects.max_source_bytes "
                                            f"({self.settings.max_source_bytes:,} bytes)")
        return None

    async def _review(self, kind: str, name: str, why: str, files: Mapping[str, str],
                      validation: Mapping[str, Any]) -> Review | None:
        """None: no review needed. A ``human`` review is answered with ``pending``."""
        mode = self.settings.review_for(kind)
        if mode == "none":
            return None
        if mode == "human" or self.reviewer is None:
            return Review("pending", "human", "awaiting `vbt project approve`")
        summary = {"kind": kind, "name": name, "why": why, "project": self.project.name,
                   "files": {k: v[:12000] for k, v in files.items()}, "validation": validation,
                   "who": self.who}
        try:
            return await self.reviewer(summary)
        except Exception as exc:  # noqa: BLE001 - an unavailable reviewer never approves
            return Review("reject", "scientific-reviewer", f"the review could not run: {type(exc).__name__}: {exc}")

    def _record(self, kind: str, name: str, files: Mapping[str, str], why: str, validation: Mapping[str, Any],
                review: Review | None, status: str, **extra: Any) -> dict[str, Any]:
        version, previous = ledger.next_version(self.project, kind, name)
        rec = {"schema": ledger.ITEM_SCHEMA, "kind": kind, "name": name, "version": version, "status": status,
               "files": dict(sorted(files.items())), "source_hash": ledger.files_hash(files), "who": self.who,
               "when": now_iso(), "why": why, "validation": dict(validation),
               "review": ({"mode": self.settings.review_for(kind), "verdict": review.verdict, "by": review.by,
                           "notes": review.notes[:4000], "invocation_id": review.invocation_id, "at": now_iso()}
                          if review is not None else {"mode": "none"}),
               "previous": previous, **extra}
        return rec

    def _commit(self, kind: str, name: str, plan: Mapping[str, Path], record: dict[str, Any], *,
                replace_dir: str | None = None, remove: Sequence[str] = (), move: bool = False) -> None:
        """Put ``plan`` ({project-relative destination: source file}) into the project and write the record; an
        item pending review goes under ``provenance/pending/<kind>/<name>/`` instead (the registered version, if
        any, stays current until ``vbt project approve``). ``remove``: project paths replaced whole (a re-imported
        data directory); ``move``: the sources are staging copies, moved rather than copied again."""
        put = shutil.move if move else shutil.copyfile
        if record["status"] == "pending_review":
            base = self.project.provenance_dir / "pending" / kind / name
            if base.exists():
                shutil.rmtree(base)
            for rel, src in plan.items():
                dest = self.project.inside(base / "files" / rel)
                dest.parent.mkdir(parents=True, exist_ok=True)
                put(str(src), str(dest))
            (base / "pending.json").write_text(json.dumps({"record": record, "files": sorted(plan),
                                                           "replace_dir": replace_dir, "remove": list(remove)},
                                                          indent=1, default=str))
            return
        _install(self.project, plan, replace_dir, remove=remove, move=move)
        ledger.write_record(self.project, record)

    # ------------------------------------------------------------------ data specs

    async def register_data_spec(self, kind: str, text: str, *, why: str, source: str | None = None,
                                 files: Sequence[Path] | Mapping[str, Path] = ()) -> Outcome:
        """Register a descriptor, an overlay, or an acquisition spec (merged into project descriptor ``source``).
        ``files`` are data files copied into ``data/<source>/`` (descriptors only): paths (each keeps its name) or
        ``{path under data/<source>/: file}`` (a directory's files under its name; importing a directory again
        replaces it whole)."""
        kind = str(kind or "").strip()
        if kind not in DATA_SPEC_KINDS:
            return _refused(kind, "?", f"kind must be one of {', '.join(DATA_SPEC_KINDS)}")
        if not str(why or "").strip():
            return _refused(kind, "?", "why is required: say what the dataset is for")
        big = self._size_ok(kind, source or "?", spec=text)
        if big is not None:
            return self._finish(big, why)
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            return self._finish(_refused(kind, source or "?", f"not YAML: {exc}"), why)
        if not isinstance(data, dict):
            return self._finish(_refused(kind, source or "?", "the spec must be a YAML mapping"), why)
        try:
            imports = _import_map(files)
        except ValueError as exc:
            return self._finish(_refused(kind, source or "?", str(exc)), why)
        async with _locked(self.project):
            if kind == "acquisition":
                return await self._acquisition(data, why=why, source=source)
            return await self._data_spec(kind, text, data, why=why, files=imports)

    def _shipped_names(self) -> tuple[set[str], set[str]]:
        from ..datalayer.descriptor.load import load_descriptors, load_overlays
        from ..datalayer.settings import DataSettings

        settings = DataSettings.from_config(self.config)
        q: list[Any] = []
        sources = set(load_descriptors(settings.descriptors_dir, quarantine=q)) | {
            x.name for x in q if x.kind == "descriptor" and x.name}
        reviewed, _generic = load_overlays(settings.overlays_dir, quarantine=q)
        servers = set(reviewed) | {x.name for x in q if x.kind == "overlay" and x.name}
        return sources, servers

    async def _acquisition(self, data: dict[str, Any], *, why: str, source: str | None) -> Outcome:
        spec = data.get("acquisition", data)
        name = str(source or "").strip()
        if not ITEM_NAME_RE.match(name):
            return self._finish(_refused("acquisition", name or "?", "source must name a project descriptor (an "
                                         "acquisition spec is merged into it)"), why)
        current = self.project.descriptors_dir / f"{name}.yaml"
        if not current.is_file():
            return self._finish(_refused("acquisition", name, f"the project has no descriptor {name!r}: register "
                                         "the descriptor first (kind descriptor, with or without acquisition)"), why)
        existing = yaml.safe_load(current.read_text(encoding="utf-8")) or {}
        if not isinstance(spec, dict):
            return self._finish(_refused("acquisition", name, "the acquisition spec must be a mapping"), why)
        merged_text = _merge_acquisition(current.read_text(encoding="utf-8"), existing, spec)
        outcome = await self._data_spec("descriptor", merged_text, yaml.safe_load(merged_text), why=why, files={},
                                        change="acquisition")
        outcome.details.setdefault("merged_into", f"descriptors/{name}.yaml")
        return outcome

    async def _data_spec(self, kind: str, text: str, data: dict[str, Any], *, why: str, files: Mapping[str, Path],
                         change: str = "spec") -> Outcome:
        key = "source" if kind == "descriptor" else "server"
        name = str(data.get(key) or "").strip()
        if not ITEM_NAME_RE.match(name):
            return self._finish(_refused(kind, name or "?", f"`{key}:` must be a lowercase identifier (got "
                                                            f"{name!r})"), why)
        sources, servers = self._shipped_names()
        if kind == "descriptor" and name in sources:
            return self._finish(_refused(kind, name, f"source {name!r} is shipped with the harness: a project adds "
                                                     "sources and never redefines one; choose another name"), why)
        if kind == "overlay" and (name in servers or name == "*"):
            return self._finish(_refused(kind, name, f"server {name!r} has a shipped overlay: a project overlay "
                                                     "binds only servers the core does not"), why)
        if files and kind != "descriptor":
            return self._finish(_refused(kind, name, "files can only be imported with a descriptor"), why)
        full = _disk_problem(self.project, sum(f.stat().st_size for f in files.values()), self.settings)
        if full:
            return self._finish(_refused(kind, name, full), why)
        tops = {Path(rel).parts[0] for rel in files}         # what the import replaces under data/<name>/
        stage = self.project.staging_dir / f"{kind}-{name}-{uuid.uuid4().hex[:8]}"
        try:
            imported = await asyncio.to_thread(_stage_project, self.project, stage, kind, name, text, files)
            report = await asyncio.to_thread(self._validate_spec, kind, name, stage)
            if report["errors"]:
                return self._finish(_refused(kind, name, "validation failed: " + "; ".join(report["errors"][:20]),
                                             lint=report["lint"], check=report["check"]), why)
            sub = "descriptors" if kind == "descriptor" else "overlays"
            plan: dict[str, Path] = {f"{sub}/{name}.yaml": stage / sub / f"{name}.yaml"}
            for rel, src in imported.items():
                plan[rel] = src
            hashes = await asyncio.to_thread(lambda: {rel: ledger.sha256_file(src) for rel, src in plan.items()})
            if kind == "descriptor":                  # data imported earlier and not replaced stays with the source
                prev = ledger.read_record(self.project, kind, name) or {}
                for rel, digest in (prev.get("files") or {}).items():
                    parts = Path(rel).parts
                    if parts[:2] == ("data", name) and len(parts) > 2 and parts[2] not in tops and \
                            rel not in hashes and (self.project.root / rel).is_file():
                        hashes[rel] = digest
            validation = {"lint": report["lint"], "check": report["check"]}
            review = await self._review(kind, name, why, {f"{sub}/{name}.yaml": text}, validation)
            if review is not None and review.verdict == "reject":
                return self._finish(_refused(kind, name, f"rejected by {review.by}: {review.notes}",
                                             review={"by": review.by, "notes": review.notes}), why)
            status = "pending_review" if review is not None and review.verdict == "pending" else "registered"
            record = self._record(kind, name, hashes, why, validation, review, status, change=change,
                                  tables=report.get("tables"))
            remove = [f"data/{name}/{t}" for t in sorted(tops) if (self.project.data_dir / name / t).is_dir()]
            await asyncio.to_thread(lambda: self._commit(kind, name, plan, record, remove=remove, move=True))
            msg = (f"{kind} {name} {'registered' if status == 'registered' else 'staged for review'} "
                   f"(version {record['version']})")
            return self._finish(Outcome(True, status, kind, name, msg, record,
                                        details={"lint": report["lint"], "check": report["check"],
                                                 "tables": report.get("tables")}), why)
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def _validate_spec(self, kind: str, name: str, stage: Path) -> dict[str, Any]:
        """Lint and check the staged project (blocking: runs the data child's check)."""
        from ..datalayer.catalog import build_catalog
        from ..datalayer.descriptor.lint import lint_descriptor, lint_overlay
        from ..datalayer.descriptor.load import PROJECT_ENV, variables_from_config
        from ..datalayer.plugins.registry import discover
        from ..datalayer.settings import DataSettings
        from ..preflight import DataCheckUnavailable, run_data_check

        cfg = _staged_config(self.config, self.project, stage)
        settings = DataSettings.from_config(cfg)
        errors: list[str] = []
        lint: list[str] = []
        check: dict[str, Any] = {}
        try:
            registry = discover(settings)
        except Exception as exc:  # noqa: BLE001 - a broken plugin path is the first error
            return {"errors": [f"plugin discovery failed: {exc}"], "lint": [], "check": {}}
        variables = variables_from_config(cfg)
        variables[f"env.{PROJECT_ENV}"] = str(stage)
        catalog = build_catalog(settings, registry, variables=variables)
        sub = "descriptors" if kind == "descriptor" else "overlays"
        mine = str(stage / sub / f"{name}.yaml")
        for q in [*catalog.project_refused, *catalog.quarantined]:
            if q.path == mine or (q.name == name and q.kind == kind):
                errors.append(f"{q.file}: {q.summary}")
        if errors:
            return {"errors": errors, "lint": lint, "check": check}
        desc: Any = None
        if kind == "descriptor":
            desc = catalog.sources.get(name)
            if desc is None:
                return {"errors": [f"{name}.yaml did not load"], "lint": [], "check": {}}
            unsafe = unsafe_spec(desc, self.policy, self.project, stage)     # before the check reads anything
            if unsafe:
                return {"errors": unsafe, "lint": [], "check": {}}
            findings = lint_descriptor(desc, registry, None, catalog.sources)
            tables = [f"{name}.{t}" for t, spec in desc.tables.items() if spec.items_of is None]
            acquirable = set((desc.acquisition.tables if desc.acquisition is not None else {}) or {})
        else:
            ov = catalog.overlays.get(name)
            if ov is None:
                return {"errors": [f"{name}.yaml did not load"], "lint": [], "check": {}}
            findings = lint_overlay(ov, catalog, registry)
            tables = sorted({str(catalog.table(r).physical) for tool in ov.tools
                             for r in catalog.contract(name, tool).tables})
            acquirable = set()
        lint = [f"{f}" + (f" [{f.rule}]" if f.rule else "") for f in findings]
        errors += [f"lint {f}" for f in findings if f.level == "error"]
        if errors or not tables:
            return {"errors": errors, "lint": lint, "check": check, "tables": tables}
        try:
            response = run_data_check(cfg, tables=tables, depth=self.settings.check_depth)
        except DataCheckUnavailable as exc:
            return {"errors": [f"vbt ds check could not run: {exc}"], "lint": lint, "check": {}, "tables": tables}
        for ref, err in (response.get("table_errors") or {}).items():
            errors.append(f"check {ref}: {err}")
        for ref in tables:
            t = (response.get("tables") or {}).get(ref)
            if t is None:
                errors.append(f"check {ref}: no result")
                continue
            failed = [c for c in t.get("checks") or [] if not c.get("ok") and c.get("level") == "error"]
            status = str(t.get("status"))
            check[ref] = {"status": status, "fingerprint": t.get("fingerprint"),
                          "failed": [f"{c.get('name')} {c.get('column') or ''}: {c.get('detail')}".replace("  ", " ")
                                     for c in failed]}
            table = ref.split(".", 1)[1]
            if status == "missing" and table in acquirable:
                check[ref]["note"] = (f"files not present yet: `{_acquire_command(desc, self.project, stage)}` "
                                      "fetches them (the acquisition spec is part of the descriptor)")
                continue
            if status != "ready":
                errors.append(f"check {ref}: {status}" + (f" ({'; '.join(check[ref]['failed'][:5])})"
                                                          if failed else ""))
            elif failed:
                errors.append(f"check {ref}: " + "; ".join(check[ref]["failed"][:5]))
        for q in response.get("quarantined") or []:
            if str(q.get("file")) == mine or q.get("name") == name:
                errors.append(f"check: {q.get('file')} quarantined by the data child: {q.get('error')}")
        return {"errors": errors, "lint": lint, "check": check, "tables": tables}

    # ------------------------------------------------------------------ plugins

    async def register_plugin(self, kind: str, module_text: str, *, why: str) -> Outcome:
        from ..datalayer.plugins import HARNESS_KINDS, KINDS
        from ..datalayer.plugins.registry import discover, discover_harness
        from ..datalayer.settings import DataSettings

        kind = str(kind or "").strip()
        known = (*KINDS, *HARNESS_KINDS)
        if kind not in known:
            return _refused("plugin", "?", f"unknown plugin kind {kind!r}: a project adds plugins of the existing "
                                           f"kinds only ({', '.join(known)}); a new kind is a core change")
        if not str(why or "").strip():
            return _refused("plugin", "?", "why is required")
        big = self._size_ok("plugin", "?", module=module_text)
        if big is not None:
            return self._finish(big, why)
        name, problem = _plugin_name(module_text)
        if problem:
            return self._finish(_refused("plugin", name or "?", problem), why)
        assert name is not None
        if not ITEM_NAME_RE.match(name):
            return self._finish(_refused("plugin", name, "the plugin name must be a lowercase identifier"), why)
        own_file = self.project.plugins_dir / kind / f"{name}.py"
        settings = DataSettings.from_config({**self.config, "data": {**dict(self.config.get("data") or {}),
                                                                     "plugins": _shipped_plugins(self.config,
                                                                                                 self.project)}})
        try:
            reg = discover_harness(settings) if kind in HARNESS_KINDS else discover(settings)
        except Exception as exc:  # noqa: BLE001
            return self._finish(_refused("plugin", name, f"plugin discovery failed: {exc}"), why)
        if reg.has(kind, name):
            return self._finish(_refused("plugin", name, f"a {kind} plugin named {name!r} already exists "
                                                         f"({reg.origins().get(f'{kind}/{name}')}): choose another "
                                                         "name"), why)
        for other in ledger.records(self.project, ["plugin"]):
            if other.get("name") == name and other.get("plugin_kind") != kind:
                return self._finish(_refused("plugin", name, f"the project already has a "
                                                             f"{other.get('plugin_kind')} plugin {name!r}"), why)
        async with _locked(self.project):
            stage = self.work_dir / ".vbt-staging" / f"plugin-{name}-{uuid.uuid4().hex[:8]}"
            try:
                stage.mkdir(parents=True)
                staged = stage / f"{name}.py"
                staged.write_text(module_text, encoding="utf-8")
                others = [p for p in self.project.plugin_files() if Path(p) != own_file]
                data = dict(self.config.get("data") or {})
                data["plugins"] = {**dict(data.get("plugins") or {}),
                                   "paths": [*_shipped_plugins(self.config, self.project)["paths"], *others,
                                             str(staged)]}
                data["cache_dir"] = str(stage / ".cache")
                child = DataSettings.from_config({**self.config, "data": data})
                settings_file = stage / "settings.json"
                settings_file.write_text(child.to_json(), encoding="utf-8")
                res = await run_sandboxed(
                    runner_argv(None, "conformance", kind, str(staged), name, str(settings_file)),
                    policy=self.policy, cwd=stage, config=self.config, label=f"conformance_{name}",
                    timeout_s=self.settings.test_timeout_s, network=self.settings.test_network)
                result = res.result if isinstance(res.result, dict) else {}
                conformance = {"exit_code": result.get("exit_code", res.exit_code), "sandbox": res.sandbox,
                               "duration_s": res.duration_s, "timed_out": res.timed_out, "notes": res.notes,
                               "output_tail": res.output[-6000:]}
                if not res.ok or result.get("exit_code") != 0:
                    why_failed = result.get("error") or ("timed out" if res.timed_out else
                                                         f"the {kind} conformance suite failed")
                    return self._finish(_refused("plugin", name, f"{why_failed}", conformance=conformance), why)
                validation = {"conformance": {**conformance, "stamp": result.get("stamp")}}
                review = await self._review("plugin", name, why, {f"plugins/{kind}/{name}.py": module_text},
                                            validation)
                if review is not None and review.verdict == "reject":
                    return self._finish(_refused("plugin", name, f"rejected by {review.by}: {review.notes}"), why)
                status = "pending_review" if review is not None and review.verdict == "pending" else "registered"
                rel = f"plugins/{kind}/{name}.py"
                record = self._record("plugin", name, {rel: ledger.sha256_file(staged)}, why, validation, review,
                                      status, plugin_kind=kind)
                self._commit("plugin", name, {rel: staged}, record)
                if status == "registered":
                    _write_stamp(self.config, result.get("stamp"))
                    write_profile(self.project, self.config)
                return self._finish(Outcome(True, status, "plugin", name,
                                            f"{kind} plugin {name} {status.replace('_', ' ')} (version "
                                            f"{record['version']})", record,
                                            details={"conformance": conformance}), why)
            finally:
                shutil.rmtree(stage, ignore_errors=True)

    # ------------------------------------------------------------------ utilities

    async def register_utility(self, name: str, *, description: str, module_text: str, tests_text: str,
                               input_schema: Mapping[str, Any] | None, why: str, entry: str = "run",
                               mode: str = "function", timeout_s: float | None = None,
                               fixtures: Mapping[str, bytes] | None = None) -> Outcome:
        name = str(name or "").strip()
        if not ITEM_NAME_RE.match(name):
            return _refused("utility", name or "?", "name must be a lowercase identifier (letters, digits, _)")
        if not str(why or "").strip():
            return _refused("utility", name, "why is required: say which repeated need the utility serves")
        big = self._size_ok("utility", name, module=module_text, tests=tests_text)
        if big is not None:
            return self._finish(big, why)
        problems = _utility_problems(name, description, module_text, tests_text, input_schema, entry, mode,
                                     fixtures or {})
        if problems:
            return self._finish(_refused("utility", name, "; ".join(problems)), why)
        schema = dict(input_schema or {"type": "object", "properties": {}})
        manifest = utility_manifest(name, description, entry, mode, schema, timeout_s)
        async with _locked(self.project):
            stage = self.work_dir / ".vbt-staging" / f"utility-{name}-{uuid.uuid4().hex[:8]}"
            try:
                stage.mkdir(parents=True)
                written = [UTILITY_FILE, TESTS_FILE]
                (stage / UTILITY_FILE).write_text(module_text, encoding="utf-8")
                (stage / TESTS_FILE).write_text(tests_text, encoding="utf-8")
                for rel, blob in (fixtures or {}).items():
                    dest = (stage / "fixtures" / rel).resolve()
                    if stage.resolve() not in dest.parents:
                        return self._finish(_refused("utility", name, f"fixture {rel!r} leaves the utility "
                                                                      "directory"), why)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(blob)
                    written.append(dest.relative_to(stage.resolve()).as_posix())
                before = {rel: ledger.sha256_file(stage / rel) for rel in written}
                res = await run_sandboxed(runner_argv(None, "test", str(stage)), policy=self.policy,
                                          cwd=stage / "scratch", config=self.config, label=f"utility_test_{name}",
                                          timeout_s=self.settings.test_timeout_s, network=self.settings.test_network)
                result = res.result if isinstance(res.result, dict) else {}
                tests = {"passed": int(result.get("passed") or 0), "failed": result.get("failed") or [],
                         "tests": result.get("tests") or [], "sandbox": res.sandbox, "duration_s": res.duration_s,
                         "timed_out": res.timed_out, "exit_code": res.exit_code, "notes": res.notes}
                if not res.ok or tests["failed"] or tests["passed"] == 0:
                    tests["output_tail"] = res.output[-6000:]
                    if res.timed_out:
                        msg = f"the tests timed out after {self.settings.test_timeout_s:g}s"
                    elif tests["failed"]:
                        msg = "tests failed: " + "; ".join(f"{f.get('name')}: {f.get('error')}"
                                                           for f in tests["failed"][:5])
                    elif tests["passed"] == 0:
                        msg = "no test ran (write test_* functions in test_utility.py)"
                    else:
                        msg = f"the test run exited {res.exit_code}"
                    return self._finish(_refused("utility", name, msg, tests=tests), why)
                after = {rel: ledger.sha256_file(stage / rel) if (stage / rel).is_file() else None for rel in written}
                if after != before:
                    return self._finish(_refused("utility", name, "the utility's files changed while its tests ran "
                                                                  "(tests must not rewrite the code they test)",
                                                 tests=tests), why)
                (stage / MANIFEST_FILE).write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
                base = f"utilities/{name}"
                plan = {f"{base}/{rel}": stage / rel for rel in [*written, MANIFEST_FILE]}
                hashes = {rel: ledger.sha256_file(src) for rel, src in plan.items()}
                validation = {"tests": tests, "schema": schema}
                review = await self._review("utility", name, why, {f"{base}/{UTILITY_FILE}": module_text,
                                                                   f"{base}/{TESTS_FILE}": tests_text}, validation)
                if review is not None and review.verdict == "reject":
                    return self._finish(_refused("utility", name, f"rejected by {review.by}: {review.notes}",
                                                 tests=tests), why)
                status = "pending_review" if review is not None and review.verdict == "pending" else "registered"
                record = self._record("utility", name, hashes, why, validation, review, status,
                                      tool=UTILITY_TOOL_PREFIX + name, description=description)
                self._commit("utility", name, plan, record, replace_dir=base)
                return self._finish(Outcome(True, status, "utility", name,
                                            f"utility {name} {status.replace('_', ' ')} (version {record['version']}"
                                            f"; {tests['passed']} test(s) passed in {res.sandbox}); "
                                            + (f"call it as {UTILITY_TOOL_PREFIX}{name}" if status == "registered"
                                               else "it becomes a tool once approved"), record,
                                            details={"tests": tests}), why)
            finally:
                shutil.rmtree(stage, ignore_errors=True)


# ---------------------------------------------------------------------------- helpers


def utility_manifest(name: str, description: str, entry: str, mode: str, schema: Mapping[str, Any],
                     timeout_s: float | None) -> dict[str, Any]:
    return {"schema": "vbt.utility/1", "name": name, "tool": UTILITY_TOOL_PREFIX + name,
            "description": " ".join(str(description).split()), "entry": entry, "mode": mode,
            "input_schema": dict(schema), **({"timeout_s": float(timeout_s)} if timeout_s else {})}


def _install(project: Project, plan: Mapping[str, Path], replace_dir: str | None, *, remove: Sequence[str] = (),
             move: bool = False) -> None:
    """Copy (``move``: move) the files into the project; ``replace_dir`` (a utility directory) is swapped in whole,
    and the ``remove`` paths (re-imported data directories) are deleted first."""
    put = shutil.move if move else shutil.copyfile
    for rel in remove:
        target = project.root / rel
        project.inside(target.parent)
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
    if replace_dir:
        target = project.inside(replace_dir)
        tmp = target.with_name(target.name + ".new")
        if tmp.exists():
            shutil.rmtree(tmp)
        for rel, src in plan.items():
            dest = project.inside(tmp / Path(rel).relative_to(replace_dir))
            dest.parent.mkdir(parents=True, exist_ok=True)
            put(str(src), str(dest))
        old = target.with_name(target.name + ".old")
        if old.exists():
            shutil.rmtree(old)
        if target.exists():
            target.rename(old)
        tmp.rename(target)
        shutil.rmtree(old, ignore_errors=True)
        return
    for rel, src in plan.items():
        dest = project.inside(rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if Path(src).resolve() == dest:
            continue
        tmp = dest.with_name(dest.name + ".tmp")
        put(str(src), str(tmp))
        tmp.replace(dest)


def _staged_config(config: Mapping[str, Any], project: Project, stage: Path) -> dict[str, Any]:
    """The configuration the staged project is validated with: ``VBT_PROJECT_DIR`` at the staging copy and the
    project's plugins discoverable."""
    from ..datalayer.descriptor.load import PROJECT_ENV

    cfg = json.loads(json.dumps(dict(config), default=str))
    cfg.setdefault("tool_env", {})[PROJECT_ENV] = str(stage)
    data = cfg.setdefault("data", {})
    plugins = data.setdefault("plugins", {})
    paths = [p for p in plugins.get("paths") or [] if not str(p).startswith(str(project.plugins_dir) + os.sep)]
    plugins["paths"] = [*paths, *project.plugin_files()]
    return cfg


_URL = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
#: Directory entries scanned for links that lead out of a named directory (a bound for the registration's time).
MAX_LINK_SCAN = 1_000_000


def _named_paths(desc: Any) -> list[tuple[str, str]]:
    """``(what, path)`` of every local file or directory a descriptor names: its root, each table's and manifest's
    path under it, and any other absolute path anywhere in it (layout and format options). URLs are left out."""
    root = str(desc.root or "")
    remote_root = bool(_URL.match(root))
    out: list[tuple[str, str]] = []
    if root and not remote_root:
        out.append(("root", root))
    named = [(f"table {t}", spec.path) for t, spec in desc.tables.items()] + \
        [(f"manifest {i}", m.path) for i, m in enumerate(desc.manifests)]
    for what, path in named:
        if path and not _URL.match(path) and not (remote_root and not os.path.isabs(path)):
            out.append((what, path if os.path.isabs(path) or not root else os.path.join(root, path)))
    seen = {p for _, p in out}

    def walk(value: Any, where: str) -> None:
        if isinstance(value, str):
            if value.startswith("/") and value not in seen:
                seen.add(value)
                out.append((where or "a field", value))
        elif isinstance(value, Mapping):
            for k, v in value.items():
                walk(v, f"{where}.{k}" if where else str(k))
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v, where)

    walk(desc.model_dump(mode="json", by_alias=True, exclude={"acquisition"}), "")
    return out


def _path_problem(what: str, raw: str, policy: Any, project: Project) -> str | None:
    """Why the registering agent may not have the data layer read ``raw`` (None: it may). A project descriptor
    reads the project's own files and what its author may read itself (``paths.read_roots``, never
    ``paths.blocked_read``); both the path as written and where its links lead are checked."""
    lexical = os.path.normpath(raw)
    if not os.path.isabs(lexical):
        return (f"{what}: {raw!r} is relative, so it would be read from the data child's working directory; give "
                "the descriptor an absolute root (`${VBT_PROJECT_DIR}/data/<source>` for imported files)")
    proj = str(project.root.resolve())
    for form in dict.fromkeys([lexical, os.path.realpath(lexical)]):
        blocked = policy is not None and policy.is_blocked(form)
        if not blocked and (form == proj or form.startswith(proj + os.sep)):
            continue
        why = (policy.read_denial(form) if policy is not None else f"{form} is outside the project")
        if why:
            return (f"{what} names {form}, which the registering agent may not read ({why}); a project descriptor "
                    "reads the project's own data (import files with `files`) and paths its author can read")
    return None


def _link_problems(what: str, raw: str, policy: Any, project: Project) -> list[str]:
    """Links inside a named directory that lead where the agent may not read (the data layer follows them)."""
    base = os.path.realpath(raw)
    if not os.path.isdir(base):
        return []
    problems: list[str] = []
    seen = 0
    for dirpath, dirnames, filenames in os.walk(base):
        for name in [*dirnames, *filenames]:
            seen += 1
            full = os.path.join(dirpath, name)
            if os.path.islink(full):
                why = _path_problem(f"{what} (link {os.path.relpath(full, base)})", os.path.realpath(full), policy,
                                    project)
                if why:
                    problems.append(why)
                    if len(problems) >= 5:
                        return problems
        if seen >= MAX_LINK_SCAN:
            problems.append(f"{what}: more than {MAX_LINK_SCAN:,} entries under {raw}; its links cannot be "
                            "verified, so it is not registered (point the root at a narrower directory)")
            break
    return problems


def unsafe_spec(desc: Any, policy: Any, project: Project, stage: Path) -> list[str]:
    """What a project descriptor may not do, checked before the data child reads anything:

    * name a path its author may not read (:func:`_path_problem`; links included): the data child would serve it;
    * declare ``acquisition.prepare`` (a command ``vbt data acquire`` runs as the operator, outside the sandbox) or
      ``acquisition.env`` (variables ``vbt setup`` writes into ``host.env``, which every ``vbt`` command loads);
    * acquire into a directory other than the one its root reads (``vbt data acquire --dest <project>/data`` would
      put the files where the descriptor never looks)."""
    problems: list[str] = []
    for what, raw in _named_paths(desc):
        why = _path_problem(what, raw, policy, project)
        if why:
            problems.append(why)
        elif what != "root":                  # the layouts read under each table's location, not the whole root
            problems += _link_problems(what, raw, policy, project)
    acq = desc.acquisition
    if acq is not None:
        if acq.prepare:
            problems.append(f"acquisition.prepare ({', '.join(acq.prepare)}): a prepare step is a command `vbt data "
                            "acquire` runs as the operator, outside the sandbox, so a project descriptor cannot "
                            "declare one; read the downloaded files as they are (a format or layout plugin) or "
                            "transform them with a utility")
        if acq.env:
            problems.append(f"acquisition.env ({', '.join(acq.env)}): these variables are written into the host "
                            "configuration every vbt command loads, so a project descriptor cannot set any; its root "
                            "names the project's data directory instead (`${VBT_PROJECT_DIR}/data/<source>`)")
        home = _acquired_into(desc, stage)
        root = os.path.normpath(str(desc.root or ""))
        data = os.path.normpath(str(stage / "data"))
        if acq.mode != "remote" and home is not None and root.startswith(data + os.sep) and home != root:
            want, got = os.path.relpath(root, data), os.path.relpath(home, data)
            problems.append(f"acquisition: `vbt data acquire {desc.source} --dest {project.data_dir}` puts the files in "
                            f"data/{got}, but the root reads data/{want}: set `dir: {want}` (and `downloads: .`)")
    return problems


def _acquired_into(desc: Any, stage: Path) -> str | None:
    """Where ``vbt data acquire --dest <project>/data`` puts the source's files (in the staged copy)."""
    try:
        from ..data.acquire import source_home
    except ImportError:                                    # pragma: no cover - part of the harness
        return None
    acq = desc.acquisition
    return os.path.normpath(str(source_home(desc, stage / "data") / (acq.downloads if acq is not None else ".")))


def _acquire_command(desc: Any, project: Project, stage: Path) -> str:
    """The command that fetches a project source's files to where its descriptor reads them (validated against
    the staged copy, whose ``data/`` stands for the project's)."""
    root = os.path.normpath(str(desc.root or ""))
    data = os.path.normpath(str(stage / "data"))
    dest = f" --dest {project.data_dir}" if root == data or root.startswith(data + os.sep) else ""
    return f"vbt --profile {project.profile_path} data acquire {desc.source}{dest}"


def _shipped_plugins(config: Mapping[str, Any], project: Project) -> dict[str, Any]:
    """``data.plugins`` without the project's own plugin files."""
    plugins = dict(((config.get("data") or {}).get("plugins") or {}))
    plugins["paths"] = [p for p in plugins.get("paths") or []
                        if not str(p).startswith(str(project.plugins_dir) + os.sep)]
    return plugins


def _import_map(files: Sequence[Path] | Mapping[str, Path]) -> dict[str, Path]:
    """``{path under data/<source>/: file}``; a plain list keeps each file's name."""
    pairs = list(files.items()) if isinstance(files, Mapping) else [(Path(f).name, Path(f)) for f in files]
    out: dict[str, Path] = {}
    for rel, src in pairs:
        parts = Path(str(rel)).parts
        if not parts or Path(str(rel)).is_absolute() or any(x in ("..", ".", "") for x in parts):
            raise ValueError(f"import name {rel!r} must be a relative path inside data/<source>/")
        if not Path(src).is_file():
            raise ValueError(f"no such file to import: {src}")
        out[Path(str(rel)).as_posix()] = Path(src)
    return out


def _disk_problem(project: Project, need: int, settings: ProjectSettings) -> str | None:
    """Why importing ``need`` bytes into the project is refused: above ``projects.max_import_bytes``, or more than
    half of what is free on the project's file system beyond 1 GiB (the rest stays for the analyses)."""
    if not need:
        return None
    if settings.max_import_bytes and need > settings.max_import_bytes:
        return (f"the import is {need:,} bytes, more than projects.max_import_bytes ({settings.max_import_bytes:,}): "
                "keep the data where it is and name it with the descriptor's root (a directory you can read)")
    free = shutil.disk_usage(project.root).free
    allowed = max(0, (free - (1 << 30)) // 2)
    if need > allowed:
        return (f"the import is {need:,} bytes and an import may take at most half of the free space beyond 1 GiB "
                f"({allowed:,} of {free:,} bytes free on the project's file system): keep the data where it is and "
                "name it with the descriptor's root (a directory you can read), or free space")
    return None


def _stage_project(project: Project, stage: Path, kind: str, name: str, text: str,
                   files: Mapping[str, Path]) -> dict[str, Path]:
    """A staging copy of the project's catalog with the candidate in place. ``data/`` links to the project's data
    directories; files imported for the candidate source are copied into the staged ``data/<name>/`` (an imported
    directory replaces the one there). Returns ``{project-relative destination: staged file}`` of the imports."""
    stage.mkdir(parents=True)
    for sub in ("descriptors", "overlays"):
        (stage / sub).mkdir()
        src = project.root / sub
        for p in sorted(src.glob("*.y*ml")) if src.is_dir() else []:
            shutil.copyfile(p, stage / sub / p.name)
    sub = "descriptors" if kind == "descriptor" else "overlays"
    for old in (stage / sub).glob(f"{name}.y*ml"):
        old.unlink()
    (stage / sub / f"{name}.yaml").write_text(text, encoding="utf-8")
    data = stage / "data"
    data.mkdir()
    real = project.data_dir
    for d in sorted(real.iterdir()) if real.is_dir() else []:
        if d.name != name or not files:
            os.symlink(d, data / d.name)
    imported: dict[str, Path] = {}
    if files:
        own = data / name
        own.mkdir()
        existing = real / name
        tops = {Path(rel).parts[0] for rel in files}
        for p in sorted(existing.iterdir()) if existing.is_dir() else []:
            if p.name not in tops:
                os.symlink(p, own / p.name)
        for rel, f in files.items():
            dest = own / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dest)
            imported[f"data/{name}/{rel}"] = dest
    if project.plugins_dir.is_dir():
        os.symlink(project.plugins_dir, stage / "plugins")
    return imported


def _merge_acquisition(text: str, existing: Mapping[str, Any], spec: Mapping[str, Any]) -> str:
    """The descriptor text with ``acquisition:`` set to ``spec`` (appended when absent, so comments stay)."""
    block = yaml.safe_dump({"acquisition": dict(spec)}, sort_keys=False, width=120)
    if "acquisition" not in existing:
        return text.rstrip() + "\n\n" + block
    merged = dict(existing)
    merged["acquisition"] = dict(spec)
    return yaml.safe_dump(merged, sort_keys=False, width=120)


def _write_stamp(config: Mapping[str, Any], stamp: Any) -> None:
    """The conformance stamp the sandbox computed, under ``data.cache_dir/conformance/``."""
    if not isinstance(stamp, dict) or "/" not in str(stamp.get("plugin") or ""):
        return
    from ..datalayer.settings import DataSettings

    kind, name = str(stamp["plugin"]).split("/", 1)
    path = Path(DataSettings.from_config(config).cache_dir) / "conformance" / f"{kind}.{name}.json"
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**stamp, "at": now_iso()}, indent=1, sort_keys=True))


def _plugin_name(text: str) -> tuple[str | None, str | None]:
    """``(name, problem)``: the literal ``name`` of the single ``@register`` class of a plugin module."""
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return None, f"the module does not parse: {exc}"
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and any(
        (isinstance(d, ast.Name) and d.id == "register") or (isinstance(d, ast.Attribute) and d.attr == "register")
        or (isinstance(d, ast.Call) and getattr(d.func, "id", getattr(d.func, "attr", None)) == "register")
        for d in n.decorator_list)]
    if len(classes) != 1:
        return None, (f"a plugin module defines exactly one @register class (found {len(classes)}); import "
                      "`register` from vbt.datalayer.plugins.registry")
    for stmt in classes[0].body:
        target = stmt.target if isinstance(stmt, ast.AnnAssign) else (
            stmt.targets[0] if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 else None)
        value = stmt.value if isinstance(stmt, (ast.Assign, ast.AnnAssign)) else None
        if isinstance(target, ast.Name) and target.id == "name":
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value, None
            return None, "the plugin's `name` must be a string literal"
    return None, f"class {classes[0].name} declares no `name = \"...\"`"


def _utility_problems(name: str, description: str, module_text: str, tests_text: str, schema: Any, entry: str,
                      mode: str, fixtures: Mapping[str, bytes]) -> list[str]:
    """Static checks of a utility before its tests run."""
    problems: list[str] = []
    if mode not in UTILITY_MODES:
        problems.append(f"mode must be one of {', '.join(UTILITY_MODES)}")
    if len(" ".join(str(description or "").split())) < MIN_DESCRIPTION:
        problems.append(f"description must say what the utility does and returns (at least {MIN_DESCRIPTION} "
                        "characters): it becomes the tool's description")
    if schema is None:
        schema = {"type": "object", "properties": {}}
    if not isinstance(schema, Mapping) or schema.get("type") != "object" or not isinstance(
            schema.get("properties", {}), Mapping):
        problems.append("input_schema must be a JSON schema object: {type: object, properties: {...}, required: "
                        "[...]}")
        schema = {"type": "object", "properties": {}}
    else:
        try:
            import jsonschema  # type: ignore

            jsonschema.validators.validator_for(schema).check_schema(schema)
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001 - jsonschema.SchemaError and friends
            problems.append(f"input_schema is not a valid JSON schema: {str(exc).splitlines()[0]}")
        unknown = [r for r in schema.get("required") or [] if r not in (schema.get("properties") or {})]
        if unknown:
            problems.append(f"input_schema requires properties it does not declare: {unknown}")
    try:
        mod = ast.parse(module_text)
    except SyntaxError as exc:
        return [*problems, f"utility.py does not parse: {exc}"]
    if mode == "script":
        if not ast.get_docstring(mod):
            problems.append("a script utility needs a module docstring (what it reads on stdin and prints)")
    else:
        fn = next((n for n in mod.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name == entry), None)
        if fn is None:
            problems.append(f"utility.py defines no top-level function {entry!r} (the entry)")
        else:
            if isinstance(fn, ast.AsyncFunctionDef):
                problems.append(f"{entry} must be a plain function, not async")
            if not ast.get_docstring(fn):
                problems.append(f"{entry} needs a docstring (what it computes, its arguments and what it returns)")
            args = fn.args
            names = [a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]]
            n_defaults = len(args.defaults)
            required_pos = [a.arg for a in [*args.posonlyargs, *args.args][:len(args.args) + len(args.posonlyargs)
                                                                             - n_defaults]]
            required_kw = [a.arg for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is None]
            props = set((schema.get("properties") or {}) if isinstance(schema, Mapping) else ())
            required = set((schema.get("required") or []) if isinstance(schema, Mapping) else ())
            missing = [p for p in [*required_pos, *required_kw] if p not in required]
            if missing:
                problems.append(f"parameters without defaults must be required in input_schema: {missing}")
            if args.kwarg is None:
                extra = sorted(props - set(names))
                if extra:
                    problems.append(f"input_schema declares properties {entry} does not take: {extra}")
    try:
        tests = ast.parse(tests_text)
    except SyntaxError as exc:
        return [*problems, f"test_utility.py does not parse: {exc}"]
    if not any(isinstance(n, ast.FunctionDef) and n.name.startswith("test_") for n in tests.body):
        problems.append("test_utility.py defines no test_* function")
    for rel in fixtures:
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            problems.append(f"fixture name {rel!r} must be a relative path inside fixtures/")
    if name.startswith("_"):
        problems.append("name must not start with '_'")
    return problems


# ---------------------------------------------------------------------------- pending review


def pending_items(project: Project) -> list[dict[str, Any]]:
    out = []
    base = project.provenance_dir / "pending"
    for p in sorted(base.glob("*/*/pending.json")) if base.is_dir() else []:
        try:
            data = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        rec = data.get("record") or {}
        out.append({"kind": rec.get("kind"), "name": rec.get("name"), "version": rec.get("version"),
                    "who": rec.get("who"), "why": rec.get("why"), "when": rec.get("when"), "dir": str(p.parent)})
    return out


def approve_pending(project: Project, kind: str, name: str, *, by: str, notes: str = "",
                    config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Install a pending item (``projects.review: human``) after checking its files are the ones validated."""
    base = project.provenance_dir / "pending" / kind / name
    try:
        data = json.loads((base / "pending.json").read_text())
    except (OSError, ValueError):
        raise ProjectError(f"no pending {kind} {name!r} in project {project.name}") from None
    record = dict(data["record"])
    plan = {rel: base / "files" / rel for rel in data.get("files") or []}
    for rel, src in plan.items():
        digest = (record.get("files") or {}).get(rel)
        if digest and ledger.sha256_file(src) != digest:
            raise ProjectError(f"{rel} changed after validation; it was not installed")
    _install(project, plan, data.get("replace_dir"), remove=data.get("remove") or (), move=True)
    record["status"] = "registered"
    record["review"] = {**dict(record.get("review") or {}), "verdict": "approve", "by": by, "notes": notes,
                        "at": now_iso()}
    ledger.write_record(project, record)
    if kind == "plugin":
        stamp = ((record.get("validation") or {}).get("conformance") or {}).get("stamp")
        if config is not None:
            _write_stamp(config, stamp)
        write_profile(project, config)
    shutil.rmtree(base, ignore_errors=True)
    ledger.append_ledger(project, {"event": "approved", "kind": kind, "name": name, "who": {"user": by},
                                   "why": notes, "source_hash": record.get("source_hash")})
    return record


def reject_pending(project: Project, kind: str, name: str, *, by: str, notes: str = "") -> None:
    base = project.provenance_dir / "pending" / kind / name
    if not (base / "pending.json").is_file():
        raise ProjectError(f"no pending {kind} {name!r} in project {project.name}")
    shutil.rmtree(base, ignore_errors=True)
    ledger.append_ledger(project, {"event": "rejected", "kind": kind, "name": name, "who": {"user": by},
                                   "why": notes})
