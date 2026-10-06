"""``chromosome``: human chromosome names in the bare style (``19``, ``X``, ``MT``).

``chr19`` loses its prefix (``strip_chr``) and ``x`` is upper-cased. ``M`` is rejected (not
renamed): the canonical mitochondrial name is ``MT``.
"""

from __future__ import annotations

import re

from ..base import Normalized, Rejected
from ..registry import register
from . import KeyIdentifier, Trace


@register
class Chromosome(KeyIdentifier):
    name = id_type = "chromosome"
    canonical = r"^(?:[1-9]|1\d|2[0-2]|X|Y|MT)$"
    examples = ("19", "X", "MT")
    overlaps = frozenset({"pmid", "ncbi_taxon"})
    description = "human chromosome, bare style (1-22, X, Y, MT; chr prefix removed)"
    cases = (
        {"raw": "19", "expected": "19", "steps": []},
        {"raw": "chr19", "expected": "19", "steps": ["strip_chr"]},
        {"raw": "chrx", "expected": "X", "steps": ["strip_chr", "upper"]},
        {"raw": 7, "expected": "7"},
        {"raw": "chrM", "rejected": True},
        {"raw": "23", "rejected": True},
        {"raw": "chr19:44908822", "rejected": True},
    )

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        m = re.fullmatch(r"chr(\w+)", t.value, re.IGNORECASE)
        if m:
            t.apply("strip_chr", m.group(1))
        if t.value.upper() in ("M", "CHRM"):
            return self.reject(t.value, "the mitochondrial chromosome is named MT")
        if re.fullmatch(r"[xy]|mt", t.value, re.IGNORECASE):
            t.upper()
        return t.done() if self.matches(t.value) else self.reject(t.value)
