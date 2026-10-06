"""``europepmc_ppr``: Europe PMC preprint IDs (``PPR123456``), which OT evidence ``literature`` lists
next to PMIDs."""

from __future__ import annotations

import re

from ..base import Normalized, Rejected
from ..registry import register
from . import KeyIdentifier, Trace


@register
class EuropePmcPreprint(KeyIdentifier):
    name = id_type = "europepmc_ppr"
    canonical = r"^PPR\d+$"
    examples = ("PPR123456",)
    description = "Europe PMC preprint ID PPRn"
    cases = (
        {"raw": "PPR123456", "expected": "PPR123456", "steps": []},
        {"raw": "ppr123456", "expected": "PPR123456", "steps": ["upper"]},
        {"raw": "123456", "rejected": True},
    )

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        if re.fullmatch(r"ppr\d+", t.value, re.IGNORECASE):
            t.upper()
        return t.done() if self.matches(t.value) else self.reject(t.value)
