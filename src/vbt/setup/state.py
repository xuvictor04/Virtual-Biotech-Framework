"""The setup state (``<state>/setup-state.json``): the last probe, the outcome of every step, and the rates
measured on this host, so ``vbt setup`` resumes where it stopped and plans with this host's own timings.

A step is skipped on the next run when it finished (``done``) with the same input fingerprint; anything that
changes its inputs (the host configuration, the step's command, the data it reads) runs it again.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

__all__ = ["SetupState", "STATE_FILE", "fingerprint"]

STATE_FILE = "setup-state.json"
SCHEMA = "vbt.setup.state/1"


def fingerprint(*parts: Any) -> str:
    """A stable digest of JSON-able ``parts``."""
    blob = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class SetupState:
    """Load, update and atomically save the state file."""

    def __init__(self, state_dir: str | Path) -> None:
        self.dir = Path(state_dir)
        self.path = self.dir / STATE_FILE
        self.data: dict[str, Any] = {"schema": SCHEMA, "steps": {}, "rates": {}, "probe": None, "plan": None}
        if self.path.is_file():
            try:
                loaded = json.loads(self.path.read_text())
            except (OSError, ValueError):
                loaded = None
            if isinstance(loaded, dict) and loaded.get("schema") == SCHEMA:
                self.data.update(loaded)

    @property
    def steps(self) -> dict[str, Any]:
        return self.data.setdefault("steps", {})

    def step(self, name: str) -> dict[str, Any]:
        return dict(self.steps.get(name) or {})

    def is_current(self, name: str, fp: str) -> bool:
        rec = self.steps.get(name) or {}
        return rec.get("status") == "done" and rec.get("fingerprint") == fp

    def record(self, name: str, **fields: Any) -> dict[str, Any]:
        rec = dict(self.steps.get(name) or {})
        rec.update(fields)
        rec["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.steps[name] = rec
        self.save()
        return rec

    def rate(self, name: str) -> float | None:
        value = (self.data.get("rates") or {}).get(name)
        return float(value) if isinstance(value, (int, float)) and value > 0 else None

    def observe_rate(self, name: str, value: float) -> None:
        """Keep a moving average (weight 0.5 on the newest) of a measured rate (seconds per GB, bytes/s)."""
        if not value or value <= 0:
            return
        old = self.rate(name)
        self.data.setdefault("rates", {})[name] = round(value if old is None else 0.5 * old + 0.5 * value, 4)
        self.save()

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True, default=str))
        os.replace(tmp, self.path)

    def summary(self) -> dict[str, Any]:
        return {name: {k: rec.get(k) for k in ("status", "seconds", "finished", "detail") if rec.get(k) is not None}
                for name, rec in self.steps.items() if isinstance(rec, Mapping)}
