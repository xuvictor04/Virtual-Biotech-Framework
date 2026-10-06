"""``reactome``: Reactome stable IDs (``R-HSA-109582``); the ``.N`` version is stripped."""

from __future__ import annotations

import re

from ..base import Normalized, Rejected
from ..registry import register
from . import KeyIdentifier, Trace


@register
class Reactome(KeyIdentifier):
    name = id_type = "reactome"
    canonical = r"^R-[A-Z]{3}-\d+$"
    examples = ("R-HSA-109582", "R-HSA-1643685")
    description = "Reactome stable ID R-SPECIES-N (version stripped)"
    cases = (
        {"raw": "R-HSA-109582", "expected": "R-HSA-109582", "steps": []},
        {"raw": "R-HSA-109582.3", "expected": "R-HSA-109582", "steps": ["strip_version"]},
        {"raw": "r-hsa-109582", "expected": "R-HSA-109582", "steps": ["upper"]},
        {"raw": "Hemostasis", "rejected": True},
        {"raw": "109582", "rejected": True},
    )

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        if not re.fullmatch(r"R-[A-Z]{3}-\d+(?:\.\d+)?", t.value, re.IGNORECASE):
            return self.reject(t.value)
        t.upper()
        m = re.fullmatch(r"(R-[A-Z]{3}-\d+)\.\d+", t.value)
        if m:
            t.apply("strip_version", m.group(1))
        return t.done()
