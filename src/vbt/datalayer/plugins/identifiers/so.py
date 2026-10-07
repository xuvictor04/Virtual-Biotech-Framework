"""``so``: Sequence Ontology term IDs (``SO:0001583``), e.g. variant consequences. Open Targets stores
``SO_0001583``: its descriptor sets ``options.separator: "_"``."""

from __future__ import annotations

from ..registry import register
from .go import OboTerm


@register
class SoTerm(OboTerm):
    name = id_type = "so"
    prefix = "SO"
    canonical = r"^SO:\d{7}$"
    examples = ("SO:0001583", "SO:0001587")
    description = "Sequence Ontology term ID SO:NNNNNNN (SO_ folded)"
    cases = (
        {"raw": "SO_0001583", "expected": "SO:0001583", "steps": ["curie_underscore_to_colon"]},
        {"raw": "SO:0001583", "expected": "SO_0001583", "options": {"separator": "_"}, "requires": "options"},
        {"raw": "so:0001583", "expected": "SO:0001583", "steps": ["upper"]},
        {"raw": "missense_variant", "rejected": True},
    )
