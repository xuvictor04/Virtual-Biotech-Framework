"""``inchikey``: standard InChIKeys (``BSYNRYMUTXBXSQ-UHFFFAOYSA-N``), an alternate molecule key."""

from __future__ import annotations

import re

from ..base import Normalized, Rejected
from ..registry import register
from . import KeyIdentifier, Trace


@register
class InchiKey(KeyIdentifier):
    name = id_type = "inchikey"
    canonical = r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$"
    examples = ("BSYNRYMUTXBXSQ-UHFFFAOYSA-N",)
    description = "standard InChIKey (27 characters, case folded)"
    cases = (
        {"raw": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "expected": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "steps": []},
        {"raw": "bsynrymutxbxsq-uhfffaoysa-n", "expected": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "steps": ["upper"]},
        {"raw": "InChIKey=BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "expected": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
         "steps": ["strip_prefix"]},
        {"raw": "BSYNRYMUTXBXSQ-UHFFFAOYSA", "rejected": True},
    )

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        m = re.fullmatch(r"InChIKey=(\S+)", t.value, re.IGNORECASE)
        if m:
            t.apply("strip_prefix", m.group(1))
        if re.fullmatch(r"[A-Za-z]{14}-[A-Za-z]{10}-[A-Za-z]", t.value):
            t.upper()
        return t.done() if self.matches(t.value) else self.reject(t.value)
