"""``cellosaurus``: Cellosaurus accessions (``CVCL_0023``); ``CVCL:0023`` is folded."""

from __future__ import annotations

import re

from ..base import Normalized, Rejected
from ..registry import register
from . import KeyIdentifier, Trace


@register
class Cellosaurus(KeyIdentifier):
    name = id_type = "cellosaurus"
    canonical = r"^CVCL_[A-Z0-9]{4}$"
    examples = ("CVCL_0023", "CVCL_1B44")
    description = "Cellosaurus accession CVCL_XXXX"
    cases = (
        {"raw": "CVCL_0023", "expected": "CVCL_0023", "steps": []},
        {"raw": "cvcl_0023", "expected": "CVCL_0023", "steps": ["upper"]},
        {"raw": "CVCL:0023", "expected": "CVCL_0023", "steps": ["curie_colon_to_underscore"]},
        {"raw": "A549", "rejected": True},
    )

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        m = re.fullmatch(r"(CVCL)([_:])([A-Z0-9]{4})", t.value, re.IGNORECASE)
        if m is None:
            return self.reject(t.value)
        if m.group(2) == ":":
            t.apply("curie_colon_to_underscore", f"{m.group(1)}_{m.group(3)}")
        t.upper()
        return t.done() if self.matches(t.value) else self.reject(t.value)
