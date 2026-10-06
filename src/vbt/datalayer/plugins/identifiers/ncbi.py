"""``ncbi_taxon``: NCBI Taxonomy IDs as digit strings. Integer and string forms agree (``9606`` and
``"9606"`` give ``"9606"``); ``NCBITaxon:9606`` and ``taxon:9606`` lose their prefix.
``options.input_requires_prefix: true`` makes agent input carry a prefix (for arguments that also
accept other digit kinds, such as PMIDs).
"""

from __future__ import annotations

import re

from ..base import Normalized, Rejected
from ..registry import register
from . import KeyIdentifier, Trace

_PREFIX = re.compile(r"(?:NCBITaxon[:_]|taxon:|txid)\s*(\S+)", re.IGNORECASE)


@register
class NcbiTaxon(KeyIdentifier):
    name = id_type = "ncbi_taxon"
    canonical = r"^[1-9]\d*$"
    examples = ("9606", "10090")
    capabilities = frozenset({"options"})
    overlaps = frozenset({"pmid", "chromosome"})
    description = "NCBI Taxonomy ID (digits; integers and NCBITaxon: prefixes accepted)"
    cases = (
        {"raw": 9606, "expected": "9606", "steps": []},
        {"raw": "9606", "expected": "9606", "steps": []},
        {"raw": "NCBITaxon:10090", "expected": "10090", "steps": ["strip_prefix"]},
        {"raw": "Homo sapiens", "rejected": True},
        {"raw": "9606", "rejected": True, "options": {"input_requires_prefix": True}, "requires": "options"},
        {"stored": 9606, "expected": "9606", "options": {"input_requires_prefix": True}, "requires": "options"},
    )

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        m = _PREFIX.fullmatch(t.value)
        if m:
            t.apply("strip_prefix", m.group(1))
        elif not stored and self.options.get("input_requires_prefix"):
            return self.reject(t.value, "give taxon IDs with their prefix (NCBITaxon:9606) for this argument")
        return t.done() if self.matches(t.value) else self.reject(t.value, "NCBI taxon IDs are digits (e.g. 9606)")
