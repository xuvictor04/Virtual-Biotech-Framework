"""``doi``: Digital Object Identifiers (``10.1056/NEJMoa1615664``). Validation only; ``doi:`` and
``https://doi.org/`` prefixes are removed (``strip_prefix``). DOIs are case-insensitive by
definition, but the stored case is kept."""

from __future__ import annotations

import re

from ..base import Normalized, Rejected
from ..registry import register
from . import KeyIdentifier, Trace

_PREFIX = re.compile(r"(?:doi:\s*|https?://(?:dx\.)?doi\.org/)(\S+)", re.IGNORECASE)


@register
class Doi(KeyIdentifier):
    name = id_type = "doi"
    canonical = r"^10\.\d{4,9}/\S+$"
    examples = ("10.1056/NEJMoa1615664",)
    description = "DOI 10.NNNN/suffix (validation only; doi: and doi.org prefixes removed)"
    cases = (
        {"raw": "10.1056/NEJMoa1615664", "expected": "10.1056/NEJMoa1615664", "steps": []},
        {"raw": "https://doi.org/10.1056/NEJMoa1615664", "expected": "10.1056/NEJMoa1615664",
         "steps": ["strip_prefix"]},
        {"raw": "doi:10.1056/NEJMoa1615664", "expected": "10.1056/NEJMoa1615664", "steps": ["strip_prefix"]},
        {"raw": "30595370", "rejected": True, "looks_like": "pmid"},
    )

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        m = _PREFIX.fullmatch(t.value)
        if m:
            t.apply("strip_prefix", m.group(1))
        return t.done() if self.matches(t.value) else self.reject(t.value)
