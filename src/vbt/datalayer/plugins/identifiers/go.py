"""``go``: Gene Ontology term IDs, canonical ``GO:0005737``.

``GO_0005737`` (the OBO-IRI and Open Targets disease-table form) becomes ``GO:0005737``
(``curie_underscore_to_colon``); IRIs lose their namespace. OT ``disease.id`` holds ``GO_`` terms,
so ``go`` and ``ot_disease`` overlap.
"""

from __future__ import annotations

import re

from ..base import Normalized, Rejected
from ..registry import register
from . import KeyIdentifier, Trace

_IRI = re.compile(r"https?://\S+/([A-Za-z]+[_:]\d+)")


class OboTerm(KeyIdentifier):
    """``PREFIX:NNNNNNN`` OBO term IDs (``prefix`` and ``digits`` set by subclasses)."""

    prefix = "GO"
    digits = 7

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        m = _IRI.fullmatch(t.value)
        if m:
            t.apply("strip_prefix", m.group(1))
        m = re.fullmatch(rf"({self.prefix})([_:])(\d{{{self.digits}}})", t.value, re.IGNORECASE)
        if m is None:
            return self.reject(t.value)
        if m.group(2) == "_":
            t.apply("curie_underscore_to_colon", f"{m.group(1)}:{m.group(3)}")
        t.upper()
        return t.done() if self.matches(t.value) else self.reject(t.value)


@register
class GoTerm(OboTerm):
    name = id_type = "go"
    prefix = "GO"
    canonical = r"^GO:\d{7}$"
    examples = ("GO:0005737", "GO:0008150")
    overlaps = frozenset({"ot_disease", "ot_entity_any"})
    description = "Gene Ontology term ID GO:NNNNNNN (GO_ and IRIs folded)"
    cases = (
        {"raw": "GO:0005737", "expected": "GO:0005737", "steps": []},
        {"raw": "GO_0005737", "expected": "GO:0005737", "steps": ["curie_underscore_to_colon"]},
        {"raw": "go:0005737", "expected": "GO:0005737", "steps": ["upper"]},
        {"raw": "http://purl.obolibrary.org/obo/GO_0005737", "expected": "GO:0005737",
         "steps": ["strip_prefix", "curie_underscore_to_colon"]},
        {"raw": "GO:1", "rejected": True},
        {"raw": "cytoplasm", "rejected": True},
    )
