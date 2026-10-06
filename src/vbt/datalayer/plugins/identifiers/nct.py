"""``nct_id``: ClinicalTrials.gov registry IDs, strictly ``NCT`` + 8 digits (``NCT05653258``).

``nct05653258`` is folded (``upper``); a wrong digit count (``NCT0565325``) is rejected, never
padded. Trial existence is decided upstream (``resolvable: false`` id_types check the pattern only).
"""

from __future__ import annotations

import re

from ..base import Normalized, Rejected
from ..registry import register
from . import KeyIdentifier, Trace


@register
class NctId(KeyIdentifier):
    name = id_type = "nct_id"
    canonical = r"^NCT\d{8}$"
    examples = ("NCT05653258", "NCT00000102")
    description = "ClinicalTrials.gov ID NCT + 8 digits"
    cases = (
        {"raw": "NCT05653258", "expected": "NCT05653258", "steps": []},
        {"raw": "nct05653258", "expected": "NCT05653258", "steps": ["upper"]},
        {"raw": " NCT05653258\n", "expected": "NCT05653258", "steps": ["strip"]},
        {"raw": "NCT0565325", "rejected": True},
        {"raw": "NCT056532581", "rejected": True},
        {"raw": "05653258", "rejected": True},
    )

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        if re.fullmatch(r"nct\d+", t.value, re.IGNORECASE):
            t.upper()
            if not self.matches(t.value):
                return self.reject(t.value, f"NCT IDs have exactly 8 digits ({t.value} has {len(t.value) - 3})")
        return t.done() if self.matches(t.value) else self.reject(t.value)
