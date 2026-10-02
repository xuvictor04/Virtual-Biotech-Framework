"""Verify a run record: artifact integrity and evidence coverage, kept separate.

**Integrity** (FAIL when violated): every artifact and harness record hashed in
MANIFEST.json is re-hashed. A changed or missing file means the record no longer
describes the run.

**Evidence coverage** (INCOMPLETE when violated): every filed claim is
re-validated strictly against the artifact registry and the trace; research
turns must cite filed claims; lifecycle problems (unfinished, interrupted or
failed turns, unavailable data sources, audit-capture errors, misplaced files,
unreadable trace lines) are reported.

**Re-execution** (opt-in, ``rerun=True``): agent scripts are re-run in a scratch
copy and their outputs compared with the recorded hashes.

These checks establish a recorded evidence trail; they do not verify scientific
correctness. Every read is tolerant: a missing or malformed record becomes a
reported problem, never an exception.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from .audit.claims import EvidenceContext, read_claims_file, validate_claims
from .audit.plan import reconcile
from .audit.provenance import SUPPORT_AGENTS, build_index, read_trace, research_turns
from .audit.render import find_refs
from .audit.storage import read_json, sha256_file, to_rel

CAVEAT = "These checks do not verify scientific correctness."

#: Problem kinds that are integrity failures (status FAIL).
INTEGRITY_KINDS = frozenset({"no_manifest", "invalid_manifest", "hash_mismatch", "missing", "harness_changed",
                             "harness_missing", "invalid_artifact_path"})


def _sha(p: Path) -> str | None:
    try:
        return sha256_file(p)
    except OSError:
        return None


def _problem(kind: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"kind": kind, "detail": detail, **extra}


def _empty_report(run_dir: Path) -> dict[str, Any]:
    return {
        "run_id": None, "run_dir": str(run_dir), "query": "", "manifest_status": None, "status": "FAIL",
        "integrity": {"status": "unavailable", "checked": 0, "changed": [], "missing": [], "harness_changed": [],
                      "harness_missing": [], "n_artifacts": 0, "n_harness_files": 0},
        "evidence": {"status": "unavailable", "problems": [], "warnings": [], "claims": 0,
                     "research_turns": [], "reference_count": 0},
        "problems": [],
    }


def _finish(report: dict[str, Any]) -> dict[str, Any]:
    integ = report["integrity"]
    ev = report["evidence"]
    integrity_problems = []
    for rel in integ["changed"]:
        integrity_problems.append(_problem("hash_mismatch", f"{rel} has changed since it was recorded", path=rel))
    for rel in integ["missing"]:
        integrity_problems.append(_problem("missing", f"{rel} is missing", path=rel))
    for rel in integ["harness_changed"]:
        integrity_problems.append(_problem("harness_changed", f"harness record {rel} has changed since it was "
                                                              "written", path=rel))
    for rel in integ["harness_missing"]:
        integrity_problems.append(_problem("harness_missing", f"harness record {rel} is missing", path=rel))
    fatal = any(p["kind"] in INTEGRITY_KINDS for p in ev["problems"])
    if integ["status"] != "unavailable":
        integ["status"] = "failed" if integrity_problems or fatal else "passed"
    if ev["status"] != "unavailable":
        coverage = [p for p in ev["problems"] if p["kind"] not in INTEGRITY_KINDS]
        ev["status"] = "incomplete" if coverage else (
            "complete" if (ev.get("research_turns") or ev.get("claims") or ev.get("reference_count"))
            else "not_required")
    report["problems"] = integrity_problems + list(ev["problems"])
    if fatal or integrity_problems or integ["status"] in ("failed", "unavailable"):
        report["status"] = "FAIL"
    elif ev["problems"] or ev["status"] in ("incomplete", "unavailable"):
        report["status"] = "INCOMPLETE"
    else:
        report["status"] = "COMPLETE"
    report["ok"] = report["status"] == "COMPLETE"
    return report


def _artifact_map(raw: Any, report: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], bool]:
    """Normalise v1 ({rel: sha}) and v2 artifact maps. Returns (map, present)."""
    if not isinstance(raw, Mapping):
        return {}, False
    out: dict[str, dict[str, Any]] = {}
    for rel, e in raw.items():
        if isinstance(e, str):
            out[str(rel)] = {"path": str(rel), "sha256": e}
        elif isinstance(e, Mapping):
            out[str(rel)] = dict(e, path=str(rel))
        else:
            out[str(rel)] = {"path": str(rel)}
    return out, True


def verify_run(run_dir: str | Path, *, rerun: bool = False, python: str | None = None,
               timeout: float = 600) -> dict[str, Any]:
    """Verify a run directory. See the module docstring for the checks."""
    run_dir = Path(run_dir).expanduser()
    report = _empty_report(run_dir)
    ev = report["evidence"]
    problems: list[dict[str, Any]] = ev["problems"]
    warnings: list[dict[str, Any]] = ev["warnings"]
    integ = report["integrity"]

    mpath = run_dir / "MANIFEST.json"
    if not mpath.is_file():
        problems.append(_problem("no_manifest", f"{run_dir} has no MANIFEST.json; it is not an auditable run."))
        return _finish(report)
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("MANIFEST.json is not a JSON object")
    except (OSError, ValueError, UnicodeError) as exc:
        problems.append(_problem("invalid_manifest", f"MANIFEST.json cannot be read: {exc}"))
        return _finish(report)
    report["run_id"] = manifest.get("run_id") or run_dir.name
    report["query"] = str(manifest.get("query") or "")
    status = manifest.get("status")
    report["manifest_status"] = status
    report["schema"] = manifest.get("schema", 1)
    integ["status"] = "passed"
    ev["status"] = "complete"

    # ---------------------------------------------------------- integrity
    arts, present = _artifact_map(manifest.get("artifacts"), report)
    if not present:
        problems.append(_problem("artifacts_not_hashed", "MANIFEST.json has no artifact hashes (the run never "
                                                         "saved its audit record), so integrity cannot be checked."))
    unhashed = []
    for rel, e in sorted(arts.items()):
        if to_rel(run_dir / rel, run_dir) in (None, "") or rel.startswith("/"):
            problems.append(_problem("invalid_artifact_path", f"{rel!r} is not a path inside the run", path=rel))
            continue
        digest = e.get("sha256")
        if not digest:
            unhashed.append(rel)
            continue
        p = run_dir / rel
        if not p.is_file():
            integ["missing"].append(rel)
        elif _sha(p) != digest:
            integ["changed"].append(rel)
    if unhashed:
        problems.append(_problem("artifacts_not_hashed", f"{len(unhashed)} artifact(s) have no recorded hash: "
                                 + ", ".join(unhashed[:5]), paths=unhashed))
    harness = manifest.get("harness_files") if isinstance(manifest.get("harness_files"), Mapping) else {}
    for rel, digest in sorted(harness.items()):
        if to_rel(run_dir / rel, run_dir) in (None, ""):
            problems.append(_problem("invalid_artifact_path", f"{rel!r} is not a path inside the run", path=rel))
            continue
        p = run_dir / rel
        if not p.is_file():
            integ["harness_missing"].append(rel)
        elif _sha(p) != digest:
            integ["harness_changed"].append(rel)
    integ["n_artifacts"] = len(arts)
    integ["n_harness_files"] = len(harness)
    integ["checked"] = len(arts) - len(unhashed) + len(harness)

    # ---------------------------------------------------------- lifecycle
    if status == "in_progress":
        problems.append(_problem("unfinished_run", "The run is still in progress, or it stopped (crash, kill) "
                                                   "before its record was finalised."))
    events, bad = read_trace(run_dir / "logs" / "trace.jsonl")
    if bad:
        problems.append(_problem("unreadable_trace_lines", f"logs/trace.jsonl has {bad} malformed or partial "
                                                           "line(s); the execution record is incomplete.", count=bad))
    prov = build_index(events, run_dir, malformed_lines=bad)

    sr = read_json(run_dir / "session_report.json", None)
    turns: list[dict[str, Any]] = []
    if sr is not None:
        raw_turns = sr.get("turns") if isinstance(sr, Mapping) else None
        if not isinstance(raw_turns, list):
            problems.append(_problem("invalid_turn_record", "session_report.json has no list of turns."))
        else:
            turns = [t for t in raw_turns if isinstance(t, Mapping)]
            if len(turns) != len(raw_turns):
                problems.append(_problem("invalid_turn_record", "session_report.json has malformed turn records."))
    elif (run_dir / "session_report.json").exists():
        problems.append(_problem("invalid_turn_record", "session_report.json cannot be read."))

    interrupted = []
    for n in manifest.get("interrupted_turns") or []:
        if n not in interrupted:
            interrupted.append(n)
    for t in turns:
        s = str(t.get("status") or "completed")
        n = t.get("turn")
        if s == "interrupted":
            if n not in interrupted:
                interrupted.append(n)
        elif s != "completed":
            problems.append(_problem("failed_turn", f"Turn {n} ended with status {s!r}; its answer and evidence "
                                                    "may be incomplete.", turn=n))
    for n in interrupted:
        problems.append(_problem("interrupted_turn", f"Turn {n} was interrupted; its response or evidence record "
                                                     "may be incomplete.", turn=n))
    if status == "interrupted" and not interrupted:
        problems.append(_problem("interrupted_turn", "The run was interrupted; its response or evidence record "
                                                     "may be incomplete."))
    for err in manifest.get("data_source_errors") or []:
        if isinstance(err, Mapping):
            name = err.get("tool_name") or err.get("tool") or "data source"
            detail = f"{name}: {str(err.get('error') or 'unavailable')[:300]}"
            if err.get("turn") is not None:
                detail = f"turn {err['turn']}: " + detail
        else:
            detail = str(err)[:300]
        problems.append(_problem("data_source_unavailable", detail))
    for err in manifest.get("audit_errors") or []:
        problems.append(_problem("audit_capture_error", str(err)[:500]))
    for m in manifest.get("misplaced_files") or []:
        if isinstance(m, Mapping):
            who = f" (by {m['agent']}" + (f" via {m['tool']}" if m.get("tool") else "") + ")" if m.get("agent") else ""
            problems.append(_problem("misplaced_files", f"{m.get('path')}: {m.get('reason', 'outside work/')}{who}",
                                     path=m.get("path")))
        else:
            problems.append(_problem("misplaced_files", f"{m}: outside work/", path=str(m)))

    # ---------------------------------------------------------- claims
    claims, err = read_claims_file(run_dir / "evidence" / "claims.json")
    if err:
        problems.append(_problem("invalid_claims", f"evidence/claims.json: {err}"))
    ev["claims"] = len(claims)
    ctx = EvidenceContext(run_dir, arts, prov.calls)
    res = validate_claims(claims, ctx, strict=True, stored=True)
    for e in res.errors:
        kind = "unresolved_evidence" if "evidence" in e else "invalid_claims"
        problems.append(_problem(kind, e))
    filed = {c["id"] for c in res.claims}
    ev["valid_claims"] = len(filed)
    ev["claims_without_verified_evidence"] = [c["id"] for c in res.claims if not c.get("n_verified")]

    # ---------------------------------------------------------- coverage
    research = research_turns(prov, arts, turns)
    for n in manifest.get("research_turns") or []:
        if isinstance(n, int) and n not in research:
            research[n] = ["recorded as a research turn"]
    ev["research_turns"] = sorted(research)
    if research and not claims:
        problems.append(_problem("no_claims", "Research was recorded (" + "; ".join(
            f"turn {n}: {', '.join(r)}" for n, r in sorted(research.items())[:3])
            + "), but no claims were filed; the evidence audit is incomplete."))
    final_path = run_dir / "report" / "FINAL_REPORT.md"
    final_text = ""
    if final_path.is_file():
        try:
            final_text = final_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            problems.append(_problem("unreadable_final_report", f"report/FINAL_REPORT.md: {exc}"))
    refs = find_refs(final_text)
    ev["reference_count"] = len(refs)
    dangling = sorted(set(refs) - filed)
    if dangling:
        problems.append(_problem("dangling_claim_reference", "The final report references claims that were not "
                                 "validly filed: " + ", ".join(dangling), ids=dangling))
    if research and final_text.strip() and not refs:
        problems.append(_problem("missing_claim_references", "The final report contains no [[claim:ID]] "
                                 "references, so its conclusions are not linked to the recorded evidence."))
    seen_turns = set()
    for t in turns:
        n = t.get("turn")
        seen_turns.add(n)
        if n not in research:
            continue
        trefs = set(find_refs(str(t.get("response") or "")))
        if not trefs & filed:
            problems.append(_problem("missing_turn_claim_references", f"Research turn {n} has no valid claim "
                                     "references; an earlier turn's citations do not cover its findings.", turn=n))
        unresolved = sorted(trefs - filed)
        if unresolved:
            problems.append(_problem("dangling_turn_claim_references", f"Turn {n} references claims that are not "
                                     "validly filed: " + ", ".join(unresolved), turn=n, ids=unresolved))
    for n in sorted(set(research) - seen_turns):
        if status != "in_progress":
            problems.append(_problem("missing_turn_record", f"Research turn {n} has no saved turn record.", turn=n))

    # ---------------------------------------------------------- plan warnings
    plan_doc = read_json(run_dir / "inputs" / "plan.json", None)
    plan_turns = set()
    if isinstance(plan_doc, Mapping):
        for h in (plan_doc.get("history") or [plan_doc]):
            if isinstance(h, Mapping) and isinstance(h.get("turn"), int):
                plan_turns.add(h["turn"])
    for n, a in prov.turn_activity().items():
        specs = [s for s in a["specialists"] if s not in SUPPORT_AGENTS]
        if len(specs) >= 2 and not a["plan_writes"] and n not in plan_turns:
            warnings.append(_problem("no_plan_for_multi_specialist_turn",
                                     f"Turn {n} dispatched {len(specs)} specialists ({', '.join(specs)}) without "
                                     "recording a plan (mcp__provenance__write_plan).", turn=n))
    if isinstance(plan_doc, Mapping) and plan_doc.get("steps"):
        pt = plan_doc.get("turn")
        execs = [e for e in prov.execution() if not e.get("orientation")
                 and (not isinstance(pt, int) or (e.get("turn") or 0) >= pt)]
        rec = reconcile(plan_doc, execs, arts)
        ev["plan_reconciliation"] = {"summary": rec["summary"], "n_deviations": rec["n_deviations"]}
        if rec["n_deviations"]:
            warnings.append(_problem("plan_deviation", rec["summary"],
                                     deviations=[d["detail"] for d in rec["deviations"][:20]]))

    # ---------------------------------------------------------- rerun
    if rerun:
        from .audit.rerun import rerun_scripts
        rr = rerun_scripts(run_dir, python=python, timeout=timeout)
        report["rerun"] = rr
        for s in rr.get("scripts") or []:
            if s.get("status") in ("differed", "missing", "failed", "timeout", "copy_failed"):
                detail = s.get("detail") or ""
                if s["status"] == "differed":
                    detail = "re-execution produced different bytes for: " + ", ".join(s["outputs_differed"][:5])
                elif s["status"] == "missing":
                    detail = "re-execution did not reproduce: " + ", ".join(s["outputs_missing"][:5])
                problems.append(_problem(f"rerun_{s['status']}", f"{s['script']}: {str(detail)[:300]}",
                                         path=s["script"]))
        if rr.get("original_modified"):
            problems.append(_problem("rerun_modified_original", "re-execution changed files in the original run: "
                                     + ", ".join(rr["original_modified"][:5])))
    return _finish(report)


def format_report(report: Mapping[str, Any]) -> str:
    """Human-readable verification summary for the terminal."""
    lines: list[str] = []
    integ = report.get("integrity") or {}
    ev = report.get("evidence") or {}
    lines.append(f"Run:    {report.get('run_id') or report.get('run_dir')}")
    if report.get("query"):
        lines.append(f"Query:  {str(report['query'])[:90]}")
    if report.get("manifest_status"):
        lines.append(f"Record: {report['manifest_status']}")
    if integ.get("status") != "unavailable":
        lines.append(f"Files:  {integ.get('n_artifacts', 0)} artifact(s) and {integ.get('n_harness_files', 0)} harness "
                     f"record(s); {len(integ.get('changed') or []) + len(integ.get('harness_changed') or [])} changed, "
                     f"{len(integ.get('missing') or []) + len(integ.get('harness_missing') or [])} missing")
        lines.append(f"Claims: {ev.get('claims', 0)} filed, {ev.get('valid_claims', 0)} with valid evidence; "
                     f"research turns: {', '.join(str(n) for n in ev.get('research_turns') or []) or 'none'}")
    lines.append(f"Artifact integrity: {str(integ.get('status', 'unavailable')).upper()}")
    lines.append(f"Evidence coverage:  {str(ev.get('status', 'unavailable')).replace('_', ' ').upper()}")
    rr = report.get("rerun")
    if rr:
        summ = rr.get("summary") or {}
        lines.append(f"Rerun:  {summ.get('matched', 0)}/{rr.get('n_scripts', 0)} script(s) reproduced their "
                     f"recorded outputs ({', '.join(f'{k} {v}' for k, v in sorted(summ.items())) or 'none run'})")
        if rr.get("note"):
            lines.append(f"        {rr['note']}")
    lines.append("")
    status = report.get("status", "FAIL")
    probs = report.get("problems") or []
    if status == "COMPLETE":
        if ev.get("status") == "not_required":
            lines.append("COMPLETE — recorded files are unchanged; no research requiring claims was recorded.")
        else:
            lines.append("COMPLETE — recorded files, claim evidence and report references passed the checks.")
    else:
        lines.append(f"{status} — {len(probs)} problem(s):")
        for p in probs[:25]:
            lines.append(f"  [{p.get('kind')}] {p.get('detail', p.get('path', ''))}")
        if len(probs) > 25:
            lines.append(f"  … and {len(probs) - 25} more")
    warns = ev.get("warnings") or []
    if warns:
        lines.append(f"Warnings ({len(warns)}):")
        for w in warns[:10]:
            lines.append(f"  [{w.get('kind')}] {w.get('detail')}")
    lines.append(CAVEAT)
    return "\n".join(lines)


def _resolve_run(arg: str) -> Path:
    p = Path(arg).expanduser()
    if p.is_dir():
        return p
    try:  # P8's resolver (path | id | unique prefix), when installed
        from .audit.index import resolve_run  # type: ignore[import-not-found]
        from .config import load_config, resolve_path
        return Path(resolve_run(arg, resolve_path(load_config()["paths"]["runs_dir"])))
    except ImportError:
        pass
    try:
        from .config import load_config, resolve_path
        runs = resolve_path(load_config()["paths"]["runs_dir"])
    except Exception:  # noqa: BLE001
        return p
    if (runs / arg).is_dir():
        return runs / arg
    hits = sorted(d for d in runs.glob(f"{arg}*") if d.is_dir()) if runs.is_dir() else []
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise SystemExit(f"ambiguous run {arg!r}: " + ", ".join(h.name for h in hits[:10]))
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vbt.verify",
                                 description="Check a run's artifact integrity and evidence coverage.")
    ap.add_argument("run", help="run directory, run id or unique id prefix")
    ap.add_argument("--rerun", action="store_true", help="re-execute agent scripts in a scratch copy")
    ap.add_argument("--python", help="interpreter for --rerun (default: this one)")
    ap.add_argument("--timeout", type=float, default=600, help="per-script timeout for --rerun, seconds")
    ap.add_argument("--json", action="store_true", help="print the machine-readable report")
    args = ap.parse_args(argv)
    report = verify_run(_resolve_run(args.run), rerun=args.rerun, python=args.python, timeout=args.timeout)
    print(json.dumps(report, indent=2, default=str) if args.json else format_report(report))
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
