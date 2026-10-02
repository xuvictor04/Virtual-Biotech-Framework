"""Rebuild a run's audit record from its trace and files: ``vbt audit RUN | --all``.

A run killed before ``Run.close()`` (or recorded by an older harness) still has
``logs/trace.jsonl`` and its ``work/`` files. ``audit_run`` reconstructs from
them, in a copy (default ``<runs>/../audits/<run_id>``) or ``in_place``:

* every file under ``work/`` hashed (a recorded hash is reused while the file's
  size and mtime are unchanged) and attributed, strongest evidence first:
  the input of a tool call that named it, an MCP tool that returned its path, a
  unique writer line in an agent's script (``created_by = 'script.py:LINE'``),
  then the one agent whose agent_start/agent_end window contains its mtime
  (``ambiguous`` when several windows overlap; never guessed). Attribution
  recorded live during the run is kept;
* claims re-validated non-strictly, so a stale pointer stays on record as
  ``unresolved`` instead of disappearing;
* turns that never got a turn record are reconstructed from ``turn_start`` and
  the CSO's last text in the trace, marked ``reconstructed`` with their real
  status (``unfinished`` when no ``turn_end`` exists);
* evidence/provenance.json, report/plan_reconciliation.json, the MANIFEST
  (status ``reconstructed`` when the run never finalised) and README.md /
  audit.html, whose notes state the attribution counts and anything the
  retrofit had to re-record (changed artifacts or harness records).

The source run is never modified unless ``in_place`` is set.
"""

from __future__ import annotations

import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import storage as st
from .claims import EvidenceContext, claims_payload, link_cited_by, read_claims_file, validate_claims
from .plan import reconcile, render_plan_md
from .provenance import build_index, read_trace, research_turns
from .render import render_report_md

FINAL_STATUSES = ("completed", "incomplete", "interrupted", "empty", "reconstructed")
COPY_MARKER = ".vbt_audit_copy"
METHODS = ("tool_capture", "tool_input", "mcp_return", "script", "window", "ambiguous", "workspace",
           "unattributed")
METHOD_TEXT = {
    "tool_capture": "recorded during the run by the harness's per-call capture",
    "tool_input": "named in a tool call's input",
    "mcp_return": "returned by an MCP tool",
    "script": "traced to a writer line in an agent's script",
    "window": "inferred from the only agent running when the file was written",
    "ambiguous": "ambiguous (several agents were running when the file was written)",
    "workspace": "attributed only by the work/<agent>/ directory they are in",
    "unattributed": "could not be attributed",
}


class AuditError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------- copying

def _link_or_copy(src: str, dst: str) -> str:
    try:
        os.link(src, dst)
        return dst
    except OSError:
        return shutil.copy2(src, dst)


def _copy_run(src: Path, target: Path, *, link: bool) -> None:
    if target == src or src in target.parents:
        raise AuditError(f"the audit copy {target} cannot be inside the run being audited")
    if target.exists():
        if not (target / COPY_MARKER).exists():
            raise AuditError(f"{target} exists and is not an earlier `vbt audit` copy; choose another --out")
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.partial-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp, symlinks=True,
                    ignore=shutil.ignore_patterns(".*", "__pycache__", "*.tmp"),
                    copy_function=_link_or_copy if link else shutil.copy2)
    (tmp / COPY_MARKER).write_text(f"audit copy of {src} made {_now()}\n")
    os.replace(tmp, target)


# ----------------------------------------------------------------- manifest

def _normalize_manifest(raw: Any, run_id: str) -> dict[str, Any]:
    m = dict(raw) if isinstance(raw, Mapping) else {}
    m.setdefault("run_id", run_id)
    arts = m.get("artifacts")
    norm: dict[str, dict[str, Any]] = {}
    if isinstance(arts, Mapping):
        for rel, e in arts.items():
            if isinstance(e, str):
                norm[str(rel)] = {"path": str(rel), "sha256": e}
            elif isinstance(e, Mapping):
                norm[str(rel)] = dict(e, path=str(rel))
    m["artifacts"] = norm
    for k in ("agents", "execution", "research_turns", "interrupted_turns", "data_source_errors", "audit_errors",
              "misplaced_files", "deleted_artifacts"):
        if not isinstance(m.get(k), list):
            m[k] = []
    if not isinstance(m.get("harness_files"), dict):
        m["harness_files"] = {}
    if not isinstance(m.get("config"), dict):
        m["config"] = {}
    return m


def _hash_harness(run_dir: Path, previous: Mapping[str, str]) -> tuple[dict[str, str], list[str]]:
    """(hashes, changed): every harness record hashed; ``changed`` lists records whose
    content differs from the previous record (the retrofit re-records them)."""
    hashes = {k: v for k, v in previous.items() if not (run_dir / k).is_file()}  # keep missing: verify reports them
    changed = []
    for sub in st.HARNESS_HASH_DIRS:
        base = run_dir / sub
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(run_dir).as_posix()
            if st.is_ignored_rel(rel):
                continue
            hashes[rel] = st.sha256_file(p)
    for name in st.HARNESS_ROOT_HASHED:
        if (run_dir / name).is_file():
            hashes[name] = st.sha256_file(run_dir / name)
    for rel, old in previous.items():
        if rel in hashes and hashes[rel] != old:
            changed.append(rel)
    return dict(sorted(hashes.items())), changed


# ----------------------------------------------------------------- turns

def _reconstruct_turns(events: list[dict[str, Any]], recorded: set[Any]) -> list[dict[str, Any]]:
    """Turn records for turns seen in the trace but never saved."""
    out: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None

    def close(rec: dict[str, Any] | None) -> None:
        if rec is not None and rec["turn"] not in recorded:
            rec["response"] = rec.pop("_final", None) or rec.pop("_last", None) or ""
            rec.pop("_final", None)
            rec.pop("_last", None)
            rec["agents"] = list(dict.fromkeys(rec["agents"]))
            out.append(rec)

    for ev in events:
        t = ev.get("type")
        if t == "turn_start":
            close(cur)
            try:
                n = int(ev.get("turn"))
            except (TypeError, ValueError):
                cur = None
                continue
            cur = {"turn": n, "prompt": str(ev.get("prompt") or ""), "status": "unfinished", "agents": [],
                   "started": ev.get("ts"), "reconstructed": True, "_final": None, "_last": None}
        elif cur is None:
            continue
        elif t == "turn_end":
            cur["status"] = str(ev.get("status") or "completed")
            cur["ended"] = ev.get("ts")
            close(cur)
            cur = None
        elif t == "agent_start" and int(ev.get("depth") or 0) >= 1 and ev.get("agent"):
            cur["agents"].append(str(ev["agent"]))
        elif t == "agent_end" and str(ev.get("agent")) == "cso" and int(ev.get("depth") or 0) == 0 and ev.get("text"):
            cur["_final"] = str(ev["text"])
        elif t == "model_call" and str(ev.get("agent")) == "cso" and ev.get("text"):
            cur["_last"] = str(ev["text"])
    close(cur)
    return out


def _render_final(run_id: str, turns: list[Mapping[str, Any]]) -> str:
    try:
        from ..session import render_final_report
        return render_final_report(run_id, turns)
    except Exception:  # noqa: BLE001 - keep the retrofit usable without the session module
        lines = [f"# Final report — {run_id}", ""]
        for t in turns:
            lines += [f"## Turn {t.get('turn')}: {str(t.get('prompt') or '')[:120]}", "", str(t.get("response") or ""),
                      ""]
        return "\n".join(lines)


# ----------------------------------------------------------------- the retrofit

def _turn_for(prov, tool_use_id: str | None, mtime: float | None) -> int | None:
    if tool_use_id and tool_use_id in prov.calls and isinstance(prov.calls[tool_use_id].get("turn"), int):
        return prov.calls[tool_use_id]["turn"]
    if mtime is None:
        return None
    best = None
    for n, w in sorted(prov.turns.items()):
        s = w.get("start_t")
        if s is not None and mtime >= float(s):
            best = n
    return best


def audit_run(run_dir: str | Path, *, in_place: bool = False, out_dir: str | Path | None = None,
              link: bool = False, render: bool = True) -> dict[str, Any]:
    """Reconstruct the audit record of one run. Returns a summary dict.

    ``in_place=False`` (default) copies the run to ``out_dir/<run_id>`` (default
    ``<runs>/../audits``) and rebuilds the copy; ``link=True`` hard-links files
    into the copy instead of copying them (records are replaced atomically, so
    the source is never modified through a link).
    """
    src = Path(run_dir).expanduser().resolve()
    if not src.is_dir():
        raise AuditError(f"no run directory at {src}")
    if not ((src / "MANIFEST.json").is_file() or (src / "logs" / "trace.jsonl").is_file()
            or (src / "work").is_dir()):
        raise AuditError(f"{src} has no MANIFEST.json, logs/trace.jsonl or work/ directory; it is not a run")
    if in_place:
        target = src
    else:
        base = Path(out_dir).expanduser().resolve() if out_dir is not None else src.parent.parent / "audits"
        target = base / src.name
        _copy_run(src, target, link=link)

    t0 = time.time()
    notes: list[str] = []
    with st.run_lock(target):
        raw = st.read_json(target / "MANIFEST.json", None)
        m = _normalize_manifest(raw, target.name)
        status_before = str(m.get("status") or ("" if raw is not None else "no_manifest"))
        schema_before = m.get("schema", 1 if raw is not None else None)
        trace_path = target / "logs" / "trace.jsonl"
        events, bad = read_trace(trace_path)
        prov = build_index(events, target, malformed_lines=bad)
        try:
            recent = time.time() - trace_path.stat().st_mtime < 300
        except OSError:
            recent = False

        # ------------------------------------------------------ artifacts
        snap = st.snapshot_dir(target)
        work = {rel: s for rel, s in snap.items() if st.is_work_rel(rel) and not st.is_ignored_rel(rel)}
        prev_arts = m["artifacts"]
        reg_rows = st.read_json(target / "evidence" / "artifacts.json", [])
        registered = {str(r.get("path")): r for r in reg_rows if isinstance(r, Mapping) and r.get("path")} \
            if isinstance(reg_rows, list) else {}
        keys = sorted(work)
        prov.index_script_outputs([k for k in keys if st.classify(k) == "code"])
        counts = {k: 0 for k in METHODS}
        changed_artifacts: list[str] = []
        arts: dict[str, dict[str, Any]] = {}
        attribution: dict[str, Any] = {}
        for rel in keys:
            mtime_ns, size = work[rel]
            prev = prev_arts.get(rel) or {}
            if prev.get("sha256") and prev.get("mtime_ns") == mtime_ns and prev.get("bytes") == size:
                sha = str(prev["sha256"])
            else:
                try:
                    sha = st.sha256_file(target / rel)
                except OSError:
                    continue
                if prev.get("sha256") and prev["sha256"] != sha:
                    changed_artifacts.append(rel)
            mtime = mtime_ns / 1e9
            a = prov.attribute_path(rel, mtime, keys)
            owner = st.work_owner(rel)
            unowned = owner is None or owner in st.HARNESS_WORK_DIRS
            method = a.get("method")
            tuid, created_by, agent = prev.get("tool_use_id"), prev.get("created_by"), None
            if method in ("tool_input", "mcp_return", "script") and a.get("agent"):
                agent = a["agent"]
                if unowned or agent == owner:
                    tuid = a.get("tool_use_id") or tuid
                    created_by = a.get("created_by") or created_by
                elif prev.get("tool_use_id"):
                    method = "tool_capture"
                else:
                    method = "workspace"
            elif prev.get("tool_use_id") and prev.get("sha256") == sha:
                method = "tool_capture"
                agent = prev.get("produced_by")
            elif method == "window" and a.get("confidence") == "ambiguous":
                method = "ambiguous"
            elif method == "window" and a.get("agent"):
                agent = a["agent"]
                if not unowned and agent != owner:
                    method = "workspace"
            elif not unowned:
                method = "workspace"
            else:
                method = "unattributed"
            if method not in counts:
                method = "unattributed"
            counts[method] += 1
            if unowned:
                produced_by = agent if (agent and method not in ("ambiguous", "unattributed")) else (
                    prev.get("produced_by") or owner or "unknown")
            else:
                produced_by = owner
            reg = registered.get(rel) or {}
            entry = {
                "path": rel, "kind": prev.get("kind") or reg.get("kind") or st.classify(rel), "bytes": size,
                "sha256": sha, "produced_by": produced_by,
                "registered_by": prev.get("registered_by") or reg.get("registered_by"),
                "tool_use_id": tuid, "created_by": created_by,
                "produced_at": prev.get("produced_at") or _iso(mtime), "modified_at": _iso(mtime),
                "mtime_ns": mtime_ns, "description": prev.get("description") or reg.get("description"),
                "registered": bool(prev.get("registered") or reg), "cited_by": [],
                "turn": prev.get("turn") or _turn_for(prov, tuid, mtime),
                "modified_turn": prev.get("modified_turn") or _turn_for(prov, tuid, mtime),
                "attribution": method,
            }
            if prev.get("registered_at"):
                entry["registered_at"] = prev["registered_at"]
            arts[rel] = entry
            attribution[rel] = dict(a, method=method)

        deleted = [d for d in m["deleted_artifacts"] if isinstance(d, Mapping)]
        known_deleted = {d.get("path") for d in deleted}
        for rel, e in prev_arts.items():
            if rel not in arts and rel not in known_deleted:
                deleted.append({"path": rel, "sha256": e.get("sha256"), "produced_by": e.get("produced_by"),
                                "cited_by": list(e.get("cited_by") or []), "deleted_at": None, "turn": None,
                                "agent": None, "tool": None, "tool_use_id": None, "detected_by": "vbt audit"})

        misplaced = [x for x in m["misplaced_files"]]
        known_mis = {x.get("path") if isinstance(x, Mapping) else x for x in misplaced}
        for rel, (mtime_ns, size) in snap.items():
            parts = st.rel_parts(rel)
            if st.is_work_rel(rel) or st.is_harness_rel(rel) or rel in known_mis or rel == COPY_MARKER:
                continue
            if parts and parts[0] in ("logs", "memory"):
                continue
            a = prov.attribute_path(rel, mtime_ns / 1e9)
            misplaced.append({"path": rel, "reason": "outside_work", "agent": a.get("agent"), "tool": a.get("tool"),
                              "tool_use_id": a.get("tool_use_id"), "bytes": size, "detected_at": _now(),
                              "turn": None, "attribution": a.get("method")})

        # ------------------------------------------------------ claims
        claims_path = target / "evidence" / "claims.json"
        claims, claims_err = read_claims_file(claims_path)
        n_claims_unresolved = 0
        if claims_err and claims_path.exists():
            notes.append(f"evidence/claims.json could not be read ({claims_err}); it was left untouched.")
            merged = claims
        else:
            ctx = EvidenceContext(target, arts, prov.calls)
            res = validate_claims(claims, ctx, strict=False, stored=True)
            by_id = {c["id"]: c for c in res.claims}
            merged = []
            for c in claims:
                new = by_id.get(str(c.get("id")))
                if new is None:  # malformed beyond repair: keep it, marked unresolved
                    new = dict(c, n_verified=0, evidence=[dict(ev, verified=False, evidence_status="unresolved")
                                                          for ev in c.get("evidence") or [] if isinstance(ev, Mapping)])
                merged.append(new)
            n_claims_unresolved = sum(1 for c in merged for ev in c.get("evidence") or []
                                      if isinstance(ev, Mapping) and ev.get("evidence_status") == "unresolved")
            if claims_path.exists() or merged:
                st.write_json_atomic(claims_path, claims_payload(merged))
        link_cited_by(arts, merged)

        # ------------------------------------------------------ turns
        sr = st.read_json(target / "session_report.json", None)
        sr = sr if isinstance(sr, dict) else {}
        turns = [t for t in (sr.get("turns") or []) if isinstance(t, dict)]
        rebuilt = _reconstruct_turns(events, {t.get("turn") for t in turns})
        if rebuilt:
            turns = sorted(turns + rebuilt, key=lambda t: (t.get("turn") or 0))
            claims_list = merged
            st.write_json_atomic(target / "session_report.json", {
                **sr, "run_id": sr.get("run_id") or m.get("run_id"), "turns": turns, "updated": _now(),
                "reconstructed_turns": [t["turn"] for t in rebuilt]})
            st.write_text_atomic(target / "inputs" / "query.txt", "\n\n".join(
                f"--- turn {t.get('turn')} ---\n{t.get('prompt', '')}" for t in turns) + "\n")
            raw_report = _render_final(str(m.get("run_id")), turns)
            st.write_text_atomic(target / "report" / "FINAL_REPORT.md", raw_report)
            st.write_text_atomic(target / "report" / "FINAL_REPORT.rendered.md",
                                 render_report_md(raw_report, claims_list))
            notes.append(f"{len(rebuilt)} turn(s) had no turn record and were reconstructed from the trace "
                         f"(turn {', '.join(str(t['turn']) for t in rebuilt)}); a reconstructed answer is the "
                         "CSO's last recorded text, which may not be what the user saw.")

        # ------------------------------------------------------ execution, plan, provenance
        execution = prov.execution()
        research = research_turns(prov, arts, turns)
        plan_doc = st.read_json(target / "inputs" / "plan.json", None)
        if isinstance(plan_doc, dict) and plan_doc.get("steps"):
            pt = plan_doc.get("turn")
            execs = [e for e in execution if not e.get("orientation")
                     and (not isinstance(pt, int) or (e.get("turn") or 0) >= pt)]
            rec = reconcile(plan_doc, execs, arts)
            rec.update(plan_version=plan_doc.get("version"), plan_turn=pt, plan_written=plan_doc.get("written"))
        else:
            rec = reconcile(None, execution)
        rec["computed_at"] = _now()
        rec["turn"] = turns[-1].get("turn") if turns else None
        st.write_json_atomic(target / "report" / "plan_reconciliation.json", rec)
        if isinstance(plan_doc, dict) and plan_doc.get("steps"):
            st.write_text_atomic(target / "report" / "plan_reconciliation.md", render_plan_md(plan_doc, rec))
        st.write_json_atomic(target / "evidence" / "provenance.json", prov.to_dict(attribution))

        # ------------------------------------------------------ notes
        finalised = status_before in FINAL_STATUSES
        if not finalised and schema_before == 1 and prev_arts:
            finalised = True  # a schema-1 run that closed (its MANIFEST has artifact hashes)
        status = status_before if finalised and status_before else ("completed" if finalised else "reconstructed")
        if not (target / "logs" / "trace.jsonl").is_file():
            notes.append(f"No execution trace was recorded, so none of the {len(arts)} artifact(s) can be "
                         "attributed beyond their work/<agent>/ directory and the order of the analysis cannot be "
                         "reconstructed.")
        when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        notes.insert(0, f"Reconstructed by `vbt audit` on {when} from logs/trace.jsonl and the files on disk"
                        + ("" if in_place else f" (an audit copy of {src})") + ". "
                        + ("The run never finalised its audit record (status was "
                           f"'{status_before or 'missing'}')." if status == "reconstructed"
                           else f"The run had finalised with status '{status_before}'."))
        parts = [f"{counts[k]} {METHOD_TEXT[k]}" for k in METHODS if counts[k]]
        notes.insert(1, f"Attribution of {len(arts)} artifact(s): " + ("; ".join(parts) if parts else "none") + ".")
        if changed_artifacts:
            notes.append(f"{len(changed_artifacts)} artifact(s) changed after their hash was recorded and were "
                         "re-hashed: " + ", ".join(changed_artifacts[:10])
                         + (" …" if len(changed_artifacts) > 10 else ""))
        missing_prev = [rel for rel in prev_arts if rel not in arts]
        if missing_prev:
            notes.append(f"{len(missing_prev)} previously recorded artifact(s) no longer exist and are listed as "
                         "deleted.")
        if n_claims_unresolved:
            notes.append(f"{n_claims_unresolved} claim evidence item(s) no longer resolve against the run and stay "
                         "on record as unresolved.")
        if not merged:
            notes.append("This run has no filed claims; reconstructing artifacts and tool activity cannot recover "
                         "claims that were never filed.")
        if bad:
            notes.append(f"{bad} malformed or partial trace line(s) were skipped.")
        if recent and in_place:
            notes.append("logs/trace.jsonl changed within the last 5 minutes; if the run is still active this "
                         "reconstruction may already be out of date.")

        # ------------------------------------------------------ manifest
        cost_report = st.read_json(target / "logs" / "cost_report.json", {}) or {}
        agents = list(dict.fromkeys([str(a) for a in m.get("agents") or []]
                                    + [str(e["agent"]) for e in execution]
                                    + [e["produced_by"] for e in arts.values()
                                       if e.get("produced_by") and not str(e["produced_by"]).startswith("_")
                                       and e["produced_by"] != "unknown"]))
        last_ts = next((ev.get("ts") for ev in reversed(events) if ev.get("ts")), None)
        query = m.get("query") or (turns[0].get("prompt") if turns else "") or ""
        m.update({
            "schema": 2, "status": status, "query": str(query)[:2000], "agents": agents, "artifacts": arts,
            "turns": len(turns), "execution": [{k: e.get(k) for k in (
                "agent", "agent_run_id", "tool_use_id", "description", "start", "end", "duration_s", "status", "turn",
                "orientation")} for e in execution],
            "research_turns": sorted(research), "misplaced_files": misplaced, "deleted_artifacts": deleted,
            "updated": _now(),
            "plan_reconciliation": {"has_plan": rec.get("has_plan", False), "summary": rec.get("summary"),
                                    "n_deviations": rec.get("n_deviations", 0)},
        })
        if not m.get("created"):
            first_ts = next((ev.get("ts") for ev in events if ev.get("ts")), None)
            m["created"] = m.get("started") or first_ts or _now()
        if status == "reconstructed" or not m.get("completed"):
            m["completed"] = m.get("completed") or last_ts or _now()
        if m.get("cost_usd") in (None, 0, 0.0) and isinstance(cost_report, dict) and cost_report.get("total_usd"):
            m["cost_usd"] = cost_report.get("total_usd")
        for n in [t.get("turn") for t in turns if t.get("status") == "interrupted"]:
            if n not in m["interrupted_turns"]:
                m["interrupted_turns"].append(n)
        m["reconstructed"] = {"at": _now(), "by": "vbt audit", "source": str(src), "in_place": bool(in_place),
                              "status_before": status_before or None, "schema_before": schema_before,
                              "attribution": counts, "n_artifacts": len(arts), "n_tool_calls": len(prov.calls),
                              "n_claims": len(merged), "reconstructed_turns": [t["turn"] for t in rebuilt],
                              "changed_artifacts": changed_artifacts}
        hashes, changed_harness = _hash_harness(target, m.get("harness_files") or {})
        m["harness_files"] = hashes
        if changed_harness:
            notes.append(f"{len(changed_harness)} harness record(s) differ from their recorded hash and were "
                         "re-recorded: " + ", ".join(changed_harness[:10]))
        m["audit_notes"] = notes
        st.write_json_atomic(target / "MANIFEST.json", m)

        # ------------------------------------------------------ reports
        paths: dict[str, Path] = {}
        if render:
            from .report import write_reports
            paths = write_reports(target, update_manifest=True)

    return {
        "run_id": m.get("run_id"), "source": str(src), "run_dir": str(target), "in_place": bool(in_place),
        "status": status, "status_before": status_before or None, "n_artifacts": len(arts),
        "attribution": counts, "n_tool_calls": len(prov.calls), "n_claims": len(merged),
        "reconstructed_turns": [t["turn"] for t in rebuilt], "notes": notes,
        "readme": str(paths.get("readme")) if paths else None, "html": str(paths.get("html")) if paths else None,
        "seconds": round(time.time() - t0, 2),
    }


def audit_all(runs_dir: str | Path, *, in_place: bool = False, out_dir: str | Path | None = None,
              link: bool = False) -> list[dict[str, Any]]:
    """``audit_run`` for every run under ``runs_dir``; one failure never stops the batch."""
    from .index import _run_dirs, update_index

    root = Path(runs_dir).expanduser().resolve()
    out_base = Path(out_dir).expanduser().resolve() if out_dir is not None else root.parent / "audits"
    results = []
    for d in sorted(_run_dirs(root)):
        try:
            results.append(audit_run(d, in_place=in_place, out_dir=out_base, link=link))
        except Exception as exc:  # noqa: BLE001
            results.append({"run_id": d.name, "source": str(d), "error": f"{type(exc).__name__}: {exc}"})
    try:
        update_index(root if in_place else out_base)
    except OSError:
        pass
    return results


__all__ = ["audit_run", "audit_all", "AuditError", "METHODS"]
