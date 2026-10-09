"""The ``vbt validate`` report: one :class:`StepResult` per step, rendered as Markdown and JSON."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

__all__ = ["PASS", "FAIL", "WARN", "SKIPPED", "ERROR", "StepResult", "ValidationReport", "percentile"]

PASS, FAIL, WARN, SKIPPED, ERROR = "pass", "fail", "warn", "skipped", "error"
_MARK = {PASS: "PASS", FAIL: "FAIL", WARN: "WARN", SKIPPED: "SKIPPED", ERROR: "ERROR"}


def percentile(values: Iterable[float], q: float) -> float | None:
    """The ``q`` percentile (0-100) by linear interpolation, None for no values."""
    xs = sorted(float(v) for v in values)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


@dataclass
class StepResult:
    """One step: ``status`` pass | fail | warn | skipped | error. A skipped step says why in ``reason`` (its
    prerequisite is missing); ``details`` is the step's own JSON; ``rows`` a table for the Markdown report."""

    name: str
    title: str
    status: str = PASS
    summary: str = ""
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    peak_mb: float | None = None

    @classmethod
    def skipped(cls, name: str, title: str, reason: str) -> "StepResult":
        return cls(name, title, SKIPPED, summary=reason, reason=reason)


@dataclass
class ValidationReport:
    started: str
    host: dict[str, Any]
    profiles: list[str] = field(default_factory=list)
    steps: list[StepResult] = field(default_factory=list)
    finished: str | None = None
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not any(s.status in (FAIL, ERROR) for s in self.steps)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.steps:
            out[s.status] = out.get(s.status, 0) + 1
        return out

    def to_json(self) -> dict[str, Any]:
        return {"schema": "vbt.validate/1", "ok": self.ok, "started": self.started, "finished": self.finished,
                "seconds": round(self.seconds, 1), "host": self.host, "profiles": self.profiles,
                "counts": self.counts(), "steps": [asdict(s) for s in self.steps]}

    def to_markdown(self) -> str:
        lines = [f"# vbt validate: {'PASS' if self.ok else 'FAIL'}", "",
                 f"Started {self.started}, finished {self.finished or '-'} ({self.seconds:,.0f} s). "
                 f"Host: {self.host.get('hostname')} ({self.host.get('platform')}), "
                 f"{self.host.get('cpus')} CPUs, plan memory {_mb(self.host.get('plan_mb'))}"
                 + (f", profiles {', '.join(self.profiles)}" if self.profiles else "") + ".", "",
                 "| Step | Status | Summary | Time |", "|---|---|---|---:|"]
        for s in self.steps:
            lines.append(f"| {s.title} | {_MARK.get(s.status, s.status)} | {_cell(s.summary)} | {s.seconds:,.1f} s |")
        for s in self.steps:
            lines += ["", f"## {s.title}", "", f"**{_MARK.get(s.status, s.status)}**: {s.summary}"]
            if s.reason and s.status == SKIPPED:
                lines.append("")
                lines.append(f"Skipped: {s.reason}")
            if s.peak_mb is not None:
                lines += ["", f"Peak resident memory of the contained process: {s.peak_mb:,.0f} MB."]
            if s.rows:
                cols = list(dict.fromkeys(k for r in s.rows for k in r))
                lines += ["", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                for r in s.rows:
                    lines.append("| " + " | ".join(_cell(r.get(c)) for c in cols) + " |")
        return "\n".join(lines) + "\n"

    def write(self, out_dir: Path) -> tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        md, js = out_dir / "validate.md", out_dir / "validate.json"
        md.write_text(self.to_markdown(), encoding="utf-8")
        js.write_text(json.dumps(self.to_json(), indent=1, sort_keys=True, default=str), encoding="utf-8")
        return md, js


def _mb(value: Any) -> str:
    return f"{value:,.0f} MB" if isinstance(value, (int, float)) else "unknown"


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        text = f"{value:,.3g}" if abs(value) < 1000 else f"{value:,.0f}"
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, default=str, sort_keys=True)
    else:
        text = str(value)
    text = text.replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= 300 else text[:297] + "..."


def host_facts(plan_mb: float | None) -> dict[str, Any]:
    import os
    import socket

    return {"hostname": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version(),
            "cpus": os.cpu_count(), "plan_mb": round(plan_mb) if plan_mb else None,
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
