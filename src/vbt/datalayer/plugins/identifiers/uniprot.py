"""``uniprot_accession``: UniProtKB accessions (``P04637``, ``A0A024RBG1``) in the official pattern.

An isoform (``P04637-2``) is rejected rather than folded onto its accession: isoforms are different
entries. Some gene symbols share the syntax (``P2RY12``), so symbol kinds declare the overlap.
"""

from __future__ import annotations

import re

from ..base import Normalized, Rejected
from ..registry import register
from . import KeyIdentifier, Trace

_ACC = r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}"


@register
class UniprotAccession(KeyIdentifier):
    name = id_type = "uniprot_accession"
    canonical = rf"^(?:{_ACC})$"
    examples = ("P04637", "Q8NBP7", "A0A024RBG1")
    description = "UniProtKB accession (case folded; isoforms rejected)"
    cases = (
        {"raw": "P04637", "expected": "P04637", "steps": []},
        {"raw": "p04637", "expected": "P04637", "steps": ["upper"]},
        {"raw": "P04637-2", "rejected": True},
        {"raw": "TP53", "rejected": True},
    )

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        if re.fullmatch(rf"(?:{_ACC})-\d+", t.value, re.IGNORECASE):
            return self.reject(t.value, f"{t.value} is an isoform ID; give the accession or the isoform's own entry")
        if re.fullmatch(_ACC, t.value, re.IGNORECASE):
            t.upper()
        return t.done() if self.matches(t.value) else self.reject(t.value)
