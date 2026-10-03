"""Re-execute a run's agent-written scripts in a scratch copy and compare outputs.

The agents themselves are not bit-reproducible (LLM sampling), but the *code*
they wrote is ordinary deterministic software: if ``02_de.py`` produced
``de_results.csv`` once it should produce the same bytes again.
``rerun_scripts``:

1. copies the whole run directory to a scratch directory (the original run is
   never written; a before/after snapshot of the original proves it);
2. rewrites the literal original run-directory prefix to the scratch prefix in
   the copied scripts and small config files, and in the environment
   (``VBT_RUN_DIR``, ``VBT_WORKSPACE``, ``MCP_OUTPUT_DIR`` and the working
   directory), with ``VBT_VERIFY=1``;
3. executes the scripts in the order the trace shows they ran (Bash calls that
   ran ``python <script>`` / ``Rscript <script>`` and succeeded), falling back to
   lexical order of the recorded code files;
4. compares each script's reproduced outputs with the MANIFEST hashes, by
   run-relative path within the same agent workspace, and reports ``matched``,
   ``differed``, ``missing``, ``failed``, ``timeout`` or ``copy_failed``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .. import envpolicy
from .provenance import build_index, parse_bash, workspace_rel
from .storage import diff_snapshots, read_json, sha256_file, snapshot_dir, to_rel, work_owner

_REWRITE_SUFFIXES = {".py", ".r", ".sh", ".ipynb", ".rmd", ".sql", ".yaml", ".yml", ".toml", ".cfg", ".ini"}
_SCRIPT_SUFFIXES = (".py", ".r", ".R")


def _artifacts(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    arts = manifest.get("artifacts") if isinstance(manifest, Mapping) else None
    for rel, e in (arts or {}).items():
        if isinstance(e, str):
            out[rel] = {"path": rel, "sha256": e}
        elif isinstance(e, Mapping):
            out[rel] = dict(e)
    return out


def _join(base: str, p: str) -> str:
    return os.path.normpath(f"{base}/{p}" if base else p).replace(os.sep, "/")


def script_runs(run_dir: Path, events: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Scripts executed successfully, in trace order (deduplicated by script, args and cwd)."""
    prov = build_index(events, run_dir) if events is not None else build_index(None, run_dir)
    seen: set[tuple] = set()
    runs: list[dict[str, Any]] = []

    def add(agent: str | None, command: str, tool_use_id: str | None) -> None:
        base = workspace_rel(agent)
        for cwd_change, interp, script, args in parse_bash(command)["scripts"]:
            cwd = base
            if cwd_change:
                if os.path.isabs(cwd_change):
                    cwd = to_rel(cwd_change, run_dir)
                    if cwd is None:
                        continue
                else:
                    cwd = _join(base, cwd_change)
            if os.path.isabs(script):
                rel = to_rel(script, run_dir)
                if rel is None:
                    continue
            else:
                rel = _join(cwd, script)
            if rel.startswith("../"):
                continue
            key = (rel, tuple(args), cwd)
            if key in seen:
                continue
            seen.add(key)
            runs.append({"script": rel, "interpreter": interp, "args": list(args), "cwd": cwd or "",
                         "agent": agent or work_owner(rel), "tool_use_id": tool_use_id})

    calls = sorted(prov.calls.values(), key=lambda c: (c.get("start_t") is None, c.get("start_t") or 0))
    for c in calls:
        if c.get("tool") == "Bash" and not c.get("is_error") and not c.get("pending") and c.get("command"):
            add(c.get("agent"), c["command"], c["tool_use_id"])
    if not runs:
        for ev in prov.bash_events:
            if ev.get("exit_code") == 0 and ev.get("command"):
                add(ev.get("agent"), str(ev["command"]), ev.get("tool_use_id"))
    return runs


def _lexical_runs(arts: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    runs = []
    for rel in sorted(arts):
        owner = work_owner(rel)
        if owner is None or owner.startswith("_") or not rel.endswith(_SCRIPT_SUFFIXES):
            continue
        runs.append({"script": rel, "interpreter": "Rscript" if rel.lower().endswith(".r") else "python",
                     "args": [], "cwd": f"work/{owner}", "agent": owner, "tool_use_id": None})
    return runs


def _rewrite_prefix(scratch: Path, prefixes: list[str]) -> int:
    n = 0
    work = scratch / "work"
    if not work.is_dir():
        return 0
    for p in work.rglob("*"):
        try:
            if not p.is_file() or p.is_symlink() or p.stat().st_size > 5_000_000:
                continue
        except OSError:
            continue
        suffix = p.suffix.lower()
        if suffix not in _REWRITE_SUFFIXES and not (suffix == ".json" and "code" in p.parts):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        new = text
        for old in prefixes:
            new = new.replace(old, str(scratch))
        if new != text:
            st = p.stat()
            p.write_text(new, encoding="utf-8")
            os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
            n += 1
    return n


def _expected_outputs(run: Mapping[str, Any], arts: Mapping[str, Mapping[str, Any]], ws: str) -> list[str]:
    name = PurePosixPath(run["script"]).name
    out = []
    for rel, e in arts.items():
        if rel == run["script"] or not rel.startswith(ws + "/"):
            continue
        if (run.get("tool_use_id") and e.get("tool_use_id") == run["tool_use_id"]) or \
                str(e.get("created_by") or "").startswith(name + ":"):
            if str(rel).endswith(_SCRIPT_SUFFIXES) and e.get("kind") == "code":
                continue
            out.append(rel)
    return sorted(set(out))


def _default_passthrough() -> list[str]:
    """``bash.env_passthrough`` from the local configuration (never the run's own copy)."""
    try:
        from ..config import load_config
        return [str(p) for p in (load_config().get("bash") or {}).get("env_passthrough") or [] if p]
    except Exception:  # noqa: BLE001 - a broken local config must not widen the environment
        return []


def _run_one(run: Mapping[str, Any], scratch: Path, prefixes: list[str], arts: Mapping[str, Mapping[str, Any]],
             python: str, timeout: float, passthrough: Iterable[str] = ()) -> dict[str, Any]:
    agent = run.get("agent") or work_owner(run["script"]) or ""
    ws = f"work/{work_owner(run['script'])}" if work_owner(run["script"]) else (workspace_rel(agent) or "")
    res: dict[str, Any] = {"script": run["script"], "agent": agent, "cwd": run.get("cwd", ""),
                           "args": list(run.get("args") or []), "tool_use_id": run.get("tool_use_id"),
                           "status": None, "outputs_matched": [], "outputs_differed": [], "outputs_missing": [],
                           "outputs_new": [], "outputs_elsewhere": []}
    target = scratch / run["script"]
    if not target.is_file():
        res.update(status="failed", detail="script not found in the run directory")
        return res
    expected = _expected_outputs(run, arts, ws) if ws else []
    for rel in expected:
        try:
            (scratch / rel).unlink()
        except OSError:
            pass
    cwd = scratch / run.get("cwd", "") if run.get("cwd") else scratch
    cwd.mkdir(parents=True, exist_ok=True)

    def sub(v: str) -> str:
        for old in prefixes:
            v = v.replace(old, str(scratch))
        return v

    # Same allow-list as the live Bash tool: provider keys and tokens never reach agent code.
    base = envpolicy.child_env(os.environ, passthrough=passthrough,
                               home=scratch.parent / ".home", tmp=scratch.parent / ".tmp")
    env = {k: sub(v) for k, v in base.items()}
    env.update(VBT_RUN_DIR=str(scratch), VBT_WORKSPACE=str(scratch / ws) if ws else str(scratch),
               MCP_OUTPUT_DIR=str(scratch / "work" / "_mcp" / "data" / "processed"),
               VBT_AGENT=agent, VBT_VERIFY="1", MPLBACKEND="Agg")
    args = [sub(a) for a in run.get("args") or []]
    if run.get("interpreter") == "Rscript":
        exe = shutil.which("Rscript")
        if not exe:
            res.update(status="failed", detail="Rscript is not available on this machine")
            return res
        cmd = [exe, str(target), *args]
    else:
        cmd = [python, str(target), *args]
    before = snapshot_dir(scratch)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        res.update(status="timeout", detail=f"exceeded {timeout:g}s", duration_s=round(time.time() - t0, 2))
        return res
    except OSError as exc:
        res.update(status="failed", detail=f"could not start: {exc}")
        return res
    res["duration_s"] = round(time.time() - t0, 2)
    res["returncode"] = proc.returncode
    if proc.returncode != 0:
        res.update(status="failed", detail=envpolicy.redact((proc.stderr or proc.stdout or "")[-2000:],
                                                            dict(os.environ)))
        return res
    after = snapshot_dir(scratch)
    changed, _deleted = diff_snapshots(before, after)
    for rel in changed:
        if ws and not rel.startswith(ws + "/"):
            res["outputs_elsewhere"].append(rel)
            continue
        e = arts.get(rel)
        if e is None:
            res["outputs_new"].append(rel)
            continue
        try:
            actual = sha256_file(scratch / rel)
        except OSError:
            continue
        (res["outputs_matched"] if actual == e.get("sha256") else res["outputs_differed"]).append(rel)
    res["outputs_missing"] = [rel for rel in expected if rel not in changed]
    if res["outputs_differed"]:
        res["status"] = "differed"
    elif res["outputs_missing"]:
        res["status"] = "missing"
    elif res["outputs_matched"]:
        res["status"] = "matched"
    else:
        res["status"] = "no_outputs"
    return res


def rerun_scripts(run_dir: str | Path, python: str | None = None, timeout: float = 600, *,
                  keep_scratch: bool = False, passthrough: Iterable[str] | None = None) -> dict[str, Any]:
    """Re-execute the run's scripts in a scratch copy; never writes into ``run_dir``.

    Scripts get the same allow-listed environment as the live Bash tool
    (:func:`vbt.envpolicy.child_env`); ``passthrough`` defaults to the local
    configuration's ``bash.env_passthrough``.
    """
    passthrough = _default_passthrough() if passthrough is None else [p for p in passthrough if p]
    given = Path(run_dir).expanduser()
    run_dir = given.resolve()
    python = python or sys.executable
    out: dict[str, Any] = {"ok": True, "python": python, "timeout_s": timeout, "n_scripts": 0, "scripts": [],
                           "order": None, "scratch_dir": None, "summary": {}, "original_modified": []}
    manifest = read_json(run_dir / "MANIFEST.json", None)
    if not isinstance(manifest, Mapping):
        out.update(ok=False, note="no readable MANIFEST.json; nothing to compare against")
        return out
    arts = _artifacts(manifest)
    runs = script_runs(run_dir)
    out["order"] = "trace"
    if not runs:
        runs = _lexical_runs(arts)
        out["order"] = "lexical"
    out["n_scripts"] = len(runs)
    if not runs:
        out["note"] = "This run contains no agent-written scripts to re-execute."
        return out
    prefixes = sorted({str(given.absolute()), str(run_dir)}, key=len, reverse=True)
    original_before = snapshot_dir(run_dir)
    scratch_root = Path(tempfile.mkdtemp(prefix="vbt-rerun-"))
    scratch = scratch_root / run_dir.name
    out["scratch_dir"] = str(scratch)
    try:
        try:
            shutil.copytree(run_dir, scratch, symlinks=True,
                            ignore=shutil.ignore_patterns(".audit.lock", ".tmp", ".home", ".downloads"))
        except (OSError, shutil.Error) as exc:
            out["ok"] = False
            out["scripts"] = [{"script": r["script"], "agent": r.get("agent"), "status": "copy_failed",
                               "detail": envpolicy.redact(str(exc)[:500], dict(os.environ))} for r in runs]
            return out
        out["rewritten_files"] = _rewrite_prefix(scratch, prefixes)
        for r in runs:
            out["scripts"].append(_run_one(r, scratch, prefixes, arts, python, timeout, passthrough))
    finally:
        if not keep_scratch:
            shutil.rmtree(scratch_root, ignore_errors=True)
            out["scratch_dir"] = None
        changed, deleted = diff_snapshots(original_before, snapshot_dir(run_dir))
        out["original_modified"] = sorted(set(changed) | set(deleted))
    counts: dict[str, int] = {}
    for s in out["scripts"]:
        counts[s["status"]] = counts.get(s["status"], 0) + 1
    out["summary"] = counts
    out["ok"] = not any(s["status"] in ("differed", "missing", "failed", "timeout", "copy_failed")
                        for s in out["scripts"]) and not out["original_modified"]
    return out


__all__ = ["rerun_scripts", "script_runs"]
