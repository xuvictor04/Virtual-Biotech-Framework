"""A network-free ``requests`` for the ClinicalTrials.gov defect detectors (imported by the unmodified
upstream ``clinicaltrials_mcp.tools`` when ``stubs/ctgov`` is first on ``sys.path``).

``VBT_CTGOV_PAGES`` names a JSON file: a list of ``{"status": int, "json": {...}}`` replies served in
order (the last one repeats). Every request's ``params`` are appended to ``VBT_CTGOV_LOG`` (JSONL).
"""

from __future__ import annotations

import json
import os
from typing import Any

from . import exceptions

__all__ = ["get", "exceptions", "Response"]

_served = 0


class Response:
    def __init__(self, status: int, body: Any) -> None:
        self.status_code = status
        self._body = body
        self.headers: dict[str, str] = {}
        self.text = json.dumps(body)

    def json(self) -> Any:
        return self._body


def get(url: str, params: Any = None, timeout: Any = None, **_: Any) -> Response:
    global _served
    log = os.environ.get("VBT_CTGOV_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"url": url, "params": params}, default=str) + "\n")
    with open(os.environ["VBT_CTGOV_PAGES"], encoding="utf-8") as fh:
        pages = json.load(fh)
    page = pages[min(_served, len(pages) - 1)]
    _served += 1
    return Response(int(page.get("status", 200)), page.get("json"))
