"""``pmid``: PubMed IDs (digits). ``PMID:30595370`` loses its prefix (``strip_prefix``).

A PMCID (``PMC1234``) or a DOI is rejected and named in ``looks_like``: upstream
``fetch_abstracts(['PMC1234'])`` returns PMID 1234, a different paper (C12, VERIFIED). ``n.v``
and other non-digit values are rejected. ``options.input_requires_prefix: true`` makes agent input
carry the ``PMID:`` prefix (for arguments that also accept other digit kinds); stored values stay
bare digits.
"""

from __future__ import annotations

import re

from ..base import Normalized, Rejected
from ..registry import register
from . import KeyIdentifier, Trace

_PREFIX = re.compile(r"PMID\s*[:_]?\s*(\S+)", re.IGNORECASE)


@register
class Pmid(KeyIdentifier):
    name = id_type = "pmid"
    canonical = r"^[1-9]\d{0,8}$"
    examples = ("30595370", "1234")
    capabilities = frozenset({"options"})
    overlaps = frozenset({"ncbi_taxon", "chromosome"})
    description = "PubMed ID (digits; PMID: prefix accepted; PMCIDs and DOIs are rejected)"
    cases = (
        {"raw": "30595370", "expected": "30595370", "steps": []},
        {"raw": 30595370, "expected": "30595370", "steps": []},
        {"raw": "PMID:30595370", "expected": "30595370", "steps": ["strip_prefix"]},
        {"raw": "pmid 1234", "expected": "1234", "steps": ["strip_prefix"]},
        {"raw": "PMC1234", "rejected": True, "looks_like": "pmcid"},
        {"raw": "10.1056/NEJMoa1615664", "rejected": True, "looks_like": "doi"},
        {"raw": "n.v", "rejected": True},
        {"raw": "0123", "rejected": True},
        {"raw": "1234", "rejected": True, "options": {"input_requires_prefix": True}, "requires": "options"},
        {"raw": "PMID:1234", "expected": "1234", "options": {"input_requires_prefix": True}, "requires": "options"},
        {"stored": "1234", "expected": "1234", "options": {"input_requires_prefix": True}, "requires": "options"},
    )

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        m = _PREFIX.fullmatch(t.value)
        if m:
            t.apply("strip_prefix", m.group(1))
        elif not stored and self.options.get("input_requires_prefix"):
            return self.reject(t.value, "give PubMed IDs with their prefix (PMID:30595370) for this argument")
        if re.fullmatch(r"PMC\d+", t.value, re.IGNORECASE):
            return Rejected(f"{t.value} is a PMCID (PubMed Central), not a PMID; it names a different paper",
                            ("pmcid",))
        if not self.matches(t.value):
            return self.reject(t.value, "PubMed IDs are 1-9 digits without leading zeros (e.g. 30595370)")
        return t.done()
