"""Run directory, trace log, cost ledger and provenance records.

Layout (mirrors the upstream Virtual Biotech session layout)::

    runs/<RUN_ID>/
      MANIFEST.json              run config + artifact hashes
      inputs/query.txt, plan.json
      work/<agent>/...           each agent's scripts, data, results
      work/_mcp/data/processed/  MCP tool outputs
      logs/trace.jsonl           every model call, tool call, delegation
      logs/cost_report.json      per-agent tokens and USD
      logs/transcript.md         human-readable conversation
      evidence/artifacts.json    registered artifacts
      evidence/claims.json       claim -> evidence objects filed by the CSO
      report/FINAL_REPORT.md     CSO responses across turns
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .providers.base import Usage


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


@dataclass
class CostLedger:
    by_agent: dict[str, Usage] = field(default_factory=lambda: defaultdict(Usage))
    usd_by_agent: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    calls_by_agent: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    extra_usd: float = 0.0  # e.g. web-search sub-requests

    def add(self, agent: str, usage: Usage, usd: float) -> None:
        self.by_agent[agent] = self.by_agent[agent] + usage
        self.usd_by_agent[agent] += usd
        self.calls_by_agent[agent] += 1

    @property
    def total_usd(self) -> float:
        return sum(self.usd_by_agent.values()) + self.extra_usd

    def report(self) -> dict[str, Any]:
        return {
            "total_usd": round(self.total_usd, 6),
            "extra_usd": round(self.extra_usd, 6),
            "agents": {
                a: {"usd": round(self.usd_by_agent[a], 6), "model_calls": self.calls_by_agent[a],
                    **self.by_agent[a].as_dict()}
                for a in sorted(self.by_agent)
            },
        }


class Run:
    """One research session (interactive or headless) and its audit record."""

    def __init__(self, root: Path, run_id: str | None = None, config: dict[str, Any] | None = None):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.dir = (Path(root) / self.run_id).resolve()
        for sub in ("inputs", "work/_mcp/data/processed", "logs", "evidence", "report"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self.cost = CostLedger()
        self.todos: dict[str, list[dict[str, Any]]] = {}
        self.turns: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._trace = open(self.dir / "logs" / "trace.jsonl", "a", buffering=1)
        self.started = _now()
        self._write_manifest()

    # ------------------------------------------------------------------ paths

    def agent_dir(self, agent: str) -> Path:
        d = self.dir / "work" / agent
        for sub in ("code/scripts", "data/raw", "data/processed", "results/figures",
                    "results/tables", "results/reports"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        return d

    @property
    def mcp_output_dir(self) -> Path:
        return self.dir / "work" / "_mcp" / "data" / "processed"

    def rel(self, path: str | Path) -> str:
        p = Path(path).resolve()
        try:
            return str(p.relative_to(self.dir))
        except ValueError:
            return str(p)

    # ------------------------------------------------------------------ trace

    def trace(self, type: str, **data: Any) -> None:
        event = {"ts": _now(), "t": round(time.time(), 3), "type": type, **data}
        with self._lock:
            self._trace.write(json.dumps(event, default=str) + "\n")

    def events(self) -> list[dict[str, Any]]:
        path = self.dir / "logs" / "trace.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    # ------------------------------------------------------------------ evidence

    def _load(self, name: str, default):
        p = self.dir / "evidence" / name
        return json.loads(p.read_text()) if p.exists() else default

    def register_artifact(self, path: str, description: str, agent: str, kind: str | None = None) -> dict:
        p = Path(path)
        if not p.is_absolute():
            p = self.dir / p
        p = p.resolve()
        if not p.exists():
            return {"ok": False, "error": f"no such file: {path}"}
        if self.dir not in p.parents:
            return {"ok": False, "error": "artifacts must live inside the run directory"}
        kind = kind or _kind(p)
        with self._lock:
            arts = self._load("artifacts.json", [])
            entry = {"path": self.rel(p), "description": description, "agent": agent, "kind": kind,
                     "sha256": _sha256(p), "bytes": p.stat().st_size, "registered": _now()}
            arts = [a for a in arts if a["path"] != entry["path"]] + [entry]
            write_json_atomic(self.dir / "evidence" / "artifacts.json", arts)
        return {"ok": True, "artifact": entry}

    def list_artifacts(self, agent: str | None = None, kind: str | None = None) -> list[dict]:
        registered = {a["path"]: a for a in self._load("artifacts.json", [])}
        # Files written but never registered are still evidence candidates.
        for p in (self.dir / "work").rglob("*"):
            if p.is_file() and not p.name.endswith(".tmp"):
                r = self.rel(p)
                if r not in registered:
                    parts = Path(r).parts
                    registered[r] = {"path": r, "agent": parts[1] if len(parts) > 1 else "?",
                                     "kind": _kind(p), "description": "(unregistered)"}
        out = list(registered.values())
        if agent:
            out = [a for a in out if a.get("agent") == agent]
        if kind:
            out = [a for a in out if a.get("kind") == kind]
        return sorted(out, key=lambda a: a["path"])

    def write_plan(self, goal: str, steps: list[dict]) -> dict:
        ids = [s.get("id") for s in steps]
        if len(set(ids)) != len(ids) or None in ids:
            return {"ok": False, "error": "every step needs a unique id"}
        known = set(ids)
        for s in steps:
            bad = [d for d in s.get("depends_on", []) if d not in known]
            if bad:
                return {"ok": False, "error": f"step {s['id']} depends on unknown steps {bad}"}
        # cycle check (Kahn)
        deps = {s["id"]: set(s.get("depends_on", [])) for s in steps}
        done: set[str] = set()
        while True:
            ready = [k for k, v in deps.items() if k not in done and v <= done]
            if not ready:
                break
            done.update(ready)
        if len(done) != len(deps):
            return {"ok": False, "error": "plan contains a dependency cycle"}
        with self._lock:
            history = json.loads((self.dir / "inputs" / "plan.json").read_text()).get("history", []) \
                if (self.dir / "inputs" / "plan.json").exists() else []
            plan = {"goal": goal, "steps": steps, "written": _now()}
            write_json_atomic(self.dir / "inputs" / "plan.json", {**plan, "history": history + [plan]})
        return {"ok": True, "n_steps": len(steps)}

    def record_claims(self, claims: list[dict], tool_ids: set[str]) -> dict:
        errors = []
        for c in claims:
            if not c.get("id") or not c.get("text"):
                errors.append("claims need 'id' and 'text'")
                continue
            if not c.get("evidence"):
                errors.append(f"{c['id']}: at least one evidence item is required")
            for e in c.get("evidence", []):
                if e.get("path"):
                    p = self.dir / e["path"] if not Path(e["path"]).is_absolute() else Path(e["path"])
                    if not p.exists():
                        errors.append(f"{c['id']}: evidence path does not exist: {e['path']}")
                    else:
                        e["sha256"] = _sha256(p)
                elif e.get("tool_use_id"):
                    if e["tool_use_id"] not in tool_ids:
                        errors.append(f"{c['id']}: unknown tool_use_id {e['tool_use_id']}")
                elif not e.get("url"):
                    errors.append(f"{c['id']}: evidence needs a path, tool_use_id or url")
        if errors:
            return {"ok": False, "errors": errors}
        with self._lock:
            existing = {c["id"]: c for c in self._load("claims.json", [])}
            for c in claims:
                existing[c["id"]] = {**c, "filed": _now()}
            write_json_atomic(self.dir / "evidence" / "claims.json", list(existing.values()))
        return {"ok": True, "n_claims": len(claims)}

    # ------------------------------------------------------------------ reports

    def _write_manifest(self) -> None:
        write_json_atomic(self.dir / "MANIFEST.json", {
            "run_id": self.run_id, "started": self.started, "updated": _now(),
            "config": self.config, "turns": len(self.turns),
            "cost_usd": round(self.cost.total_usd, 6),
        })

    def finish_turn(self, turn: dict[str, Any]) -> None:
        self.turns.append(turn)
        write_json_atomic(self.dir / "logs" / "cost_report.json", self.cost.report())
        write_json_atomic(self.dir / "session_report.json", {"run_id": self.run_id, "turns": self.turns,
                                                             "cost": self.cost.report()})
        lines = [f"# Virtual Biotech run {self.run_id}", ""]
        report = [f"# Final report — {self.run_id}", ""]
        for t in self.turns:
            lines += [f"## Turn {t['turn']}", f"**User:** {t['prompt']}", "",
                      f"**CSO:** {t['response']}", "",
                      f"*Agents:* {', '.join(t['agents']) or '(none)'} · *cost:* ${t['cost_usd']:.2f}", ""]
            report += [f"## Turn {t['turn']}: {t['prompt'][:120]}", "", t["response"], ""]
        (self.dir / "logs" / "transcript.md").write_text("\n".join(lines))
        (self.dir / "report" / "FINAL_REPORT.md").write_text("\n".join(report))
        self._write_manifest()

    def close(self) -> None:
        self._write_manifest()
        manifest = json.loads((self.dir / "MANIFEST.json").read_text())
        manifest["artifacts"] = {self.rel(p): _sha256(p) for p in sorted(self.dir.rglob("*"))
                                 if p.is_file() and "logs" not in p.parts and p.name != "MANIFEST.json"}
        write_json_atomic(self.dir / "MANIFEST.json", manifest)
        self._trace.close()


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _kind(p: Path) -> str:
    ext = p.suffix.lower()
    if ext in (".png", ".pdf", ".svg", ".jpg", ".jpeg"):
        return "figure"
    if ext in (".csv", ".tsv", ".parquet", ".xlsx"):
        return "table"
    if ext in (".py", ".r", ".R", ".sh", ".ipynb"):
        return "code"
    if ext in (".md", ".txt", ".html"):
        return "report"
    if ext in (".h5ad", ".h5", ".json", ".jsonl", ".pkl"):
        return "data"
    return "other"
