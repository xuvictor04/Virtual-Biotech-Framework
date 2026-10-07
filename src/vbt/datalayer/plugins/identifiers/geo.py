"""``geo_gsm``: GEO sample accessions (``GSM1234567``), the obs index of the Zenodo GEO cohort files (phase 2).

Case folds to the canonical upper case (``gsm1234567`` -> ``GSM1234567``, ``upper``); series (``GSE``)
and platform (``GPL``) accessions are other kinds and are rejected. A ``GSM`` accession is also a
syntactically valid gene-symbol or free-text string, so the free-syntax kinds are declared overlaps.
"""

from __future__ import annotations

import re

from ..base import Normalized, Rejected
from ..registry import register
from . import TEXT_KINDS, KeyIdentifier, Trace


@register
class GeoGsm(KeyIdentifier):
    name = id_type = "geo_gsm"
    canonical = r"^GSM\d+$"
    examples = ("GSM1234567", "GSM1798004")
    overlaps = TEXT_KINDS
    description = "GEO sample accession GSMnnnnnnn (GSE series and GPL platforms are other kinds)"
    cases = (
        {"raw": "GSM1234567", "expected": "GSM1234567", "steps": []},
        {"raw": " gsm1234567", "expected": "GSM1234567", "steps": ["strip", "upper"]},
        {"raw": "GSE73661", "rejected": True},
        {"raw": "GPL570", "rejected": True},
        {"raw": "1234567", "rejected": True},
    )

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        if re.fullmatch(r"gsm\d+", t.value, re.IGNORECASE):
            t.upper()
        if re.fullmatch(r"(GSE|GPL|GDS)\d+", t.value, re.IGNORECASE):
            return Rejected(f"{t.value} is a GEO {t.value[:3].upper()} accession, not a sample (GSM)")
        return t.done() if self.matches(t.value) else self.reject(t.value)
