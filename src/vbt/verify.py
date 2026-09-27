"""Verify a run record: artifact integrity and claim-evidence coverage."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "MANIFEST.json").read_text())
    changed, missing = [], []
    for rel, digest in (manifest.get("artifacts") or {}).items():
        p = run_dir / rel
        if not p.exists():
            missing.append(rel)
        elif _sha(p) != digest:
            changed.append(rel)

    claims_path = run_dir / "evidence" / "claims.json"
    claims = json.loads(claims_path.read_text()) if claims_path.exists() else []
    ids = {c["id"] for c in claims}
    stale = []
    for c in claims:
        for e in c.get("evidence", []):
            if e.get("path") and e.get("sha256"):
                p = run_dir / e["path"]
                if not p.exists() or _sha(p) != e["sha256"]:
                    stale.append(f"{c['id']}:{e['path']}")
    report_path = run_dir / "report" / "FINAL_REPORT.md"
    anchors = set(re.findall(r"\[\[claim:([\w\-]+)\]\]", report_path.read_text())) if report_path.exists() else set()
    dangling = sorted(anchors - ids)

    delegated = False
    trace = run_dir / "logs" / "trace.jsonl"
    if trace.exists():
        delegated = any('"type": "delegation"' in line for line in trace.read_text().splitlines())
    evidence_ok = (not delegated) or bool(claims)
    status = "COMPLETE" if not (changed or missing or stale or dangling) and evidence_ok else "INCOMPLETE"
    return {
        "run_id": manifest.get("run_id"), "status": status,
        "artifact_integrity": {"checked": len(manifest.get("artifacts") or {}), "changed": changed, "missing": missing},
        "evidence_coverage": {"claims": len(claims), "stale_evidence": stale, "dangling_anchors": dangling,
                              "research_run_without_claims": delegated and not claims},
    }
