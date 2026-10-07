"""``pmcid``: PubMed Central IDs (``PMC1234``). Validation only: no PubMed tool resolves them, but the
kind lets the resolver say "looks like a PMCID" instead of reading ``PMC1234`` as PMID 1234."""

from __future__ import annotations

import re

from ..base import Normalized, Rejected
from ..registry import register
from . import KeyIdentifier, Trace


@register
class Pmcid(KeyIdentifier):
    name = id_type = "pmcid"
    canonical = r"^PMC\d+$"
    examples = ("PMC1234", "PMC6309485")
    description = "PubMed Central ID PMCn (validation only)"
    cases = (
        {"raw": "PMC1234", "expected": "PMC1234", "steps": []},
        {"raw": "pmc1234", "expected": "PMC1234", "steps": ["upper"]},
        {"raw": "PMCID: PMC1234", "expected": "PMC1234", "steps": ["strip_prefix"]},
        {"raw": "1234", "rejected": True, "looks_like": "pmid"},
    )

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        m = re.fullmatch(r"PMCID\s*:?\s*(\S+)", t.value, re.IGNORECASE)
        if m:
            t.apply("strip_prefix", m.group(1))
        if re.fullmatch(r"pmc\d+", t.value, re.IGNORECASE):
            t.upper()
        return t.done() if self.matches(t.value) else self.reject(t.value)
