"""``depmap_cell_line``: DepMap model IDs (``ACH-000001``).

Cell-line names are labels (``resolve_via: [cell_line_metadata.cell_name]``) matched exactly but
insensitive to punctuation and case: ``label_key`` keeps letters and digits only, so ``NCI-H460``,
``NCIH460`` and ``nci h460`` share one key. Names are never matched as substrings (``PC-3`` is not
``BxPC-3``).
"""

from __future__ import annotations

import re
import unicodedata

from ..base import Normalized, Rejected
from ..registry import register
from . import KeyIdentifier, Trace


@register
class DepmapCellLine(KeyIdentifier):
    name = id_type = "depmap_cell_line"
    canonical = r"^ACH-\d{6}$"
    examples = ("ACH-000001", "ACH-000681")
    description = "DepMap model ID ACH-NNNNNN (cell-line names resolve by punctuation-insensitive exact match)"
    cases = (
        {"raw": "ACH-000001", "expected": "ACH-000001", "steps": []},
        {"raw": "ach-000001", "expected": "ACH-000001", "steps": ["upper"]},
        {"raw": "NCI-H460", "rejected": True},
        {"raw": "ACH-1", "rejected": True},
        {"label": "NCI-H460", "key": "ncih460"},
        {"label": "nci h460", "key": "ncih460"},
        {"label": "PC-3", "key": "pc3"},
    )

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        if re.fullmatch(r"ach-\d{6}", t.value, re.IGNORECASE):
            t.upper()
        return t.done() if self.matches(t.value) else self.reject(t.value)

    def label_key(self, raw: str) -> str:
        text = unicodedata.normalize("NFKC", str(raw)).casefold()
        return "".join(ch for ch in text if ch.isalnum())
