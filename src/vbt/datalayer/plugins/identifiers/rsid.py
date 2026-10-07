"""``rsid``: dbSNP reference SNP IDs (``rs7412``). ``RS7412`` is folded (``lower``).

Cardinality ``many``: one rsID can name several variants (multi-allelic sites), so a lookup that
finds several is ``ambiguous``, never its first row (C11; upstream takes ``iloc[0]``).
"""

from __future__ import annotations

import re

from ..base import Normalized, Rejected
from ..registry import register
from . import KeyIdentifier, Trace


@register
class Rsid(KeyIdentifier):
    name = id_type = "rsid"
    canonical = r"^rs\d+$"
    examples = ("rs7412", "rs429358")
    cardinality = "many"
    description = "dbSNP rsID (may name several variants)"
    cases = (
        {"raw": "rs7412", "expected": "rs7412", "steps": []},
        {"raw": "RS7412", "expected": "rs7412", "steps": ["lower"]},
        {"raw": "7412", "rejected": True},
        {"raw": "19_44908822_C_T", "rejected": True, "looks_like": "ot_variant"},
    )

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        if re.fullmatch(r"rs\d+", t.value, re.IGNORECASE):
            t.lower()
        return t.done() if self.matches(t.value) else self.reject(t.value)
