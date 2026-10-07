"""``hpo``: Human Phenotype Ontology term IDs. Canonical ``HP_0001250`` (the form OT tables store);
``HP:0001250`` is accepted (``curie_colon_to_underscore``). ``options.separator: ":"`` makes
``HP:0001250`` canonical instead, for sources that store the OBO form. OT ``disease.id`` holds HP
terms, so ``hpo`` and ``ot_disease`` overlap.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Self, Sequence

from ..base import Normalized, Rejected
from ..registry import register
from . import KeyIdentifier, Trace, set_attr

_IRI = re.compile(r"https?://\S+/(HP[_:]\d+)", re.IGNORECASE)


@register
class Hpo(KeyIdentifier):
    name = id_type = "hpo"
    canonical = r"^HP_\d{7}$"
    examples = ("HP_0001250", "HP_0001370")
    capabilities = frozenset({"options"})
    overlaps = frozenset({"ot_disease", "ot_entity_any"})
    description = "Human Phenotype Ontology term ID (HP_NNNNNNN; HP: accepted)"
    cases = (
        {"raw": "HP_0001250", "expected": "HP_0001250", "steps": []},
        {"raw": "HP:0001250", "expected": "HP_0001250", "steps": ["curie_colon_to_underscore"]},
        {"raw": "hp:0001250", "expected": "HP_0001250", "steps": ["curie_colon_to_underscore", "upper"]},
        {"raw": "HP_0001250", "expected": "HP:0001250", "options": {"separator": ":"}, "requires": "options"},
        {"raw": "HP_123", "rejected": True},
        {"raw": "seizure", "rejected": True},
    )

    def configure(self, options: Mapping[str, Any], universe_sample: Sequence[str] | None) -> Self:
        other = super().configure(options, universe_sample)
        if (options or {}).get("separator") == ":":
            set_attr(other, "canonical", r"^HP:\d{7}$")
        return other

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        m = _IRI.fullmatch(t.value)
        if m:
            t.apply("strip_prefix", m.group(1))
        m = re.fullmatch(r"(HP)([_:])(\d{7})", t.value, re.IGNORECASE)
        if m is None:
            return self.reject(t.value)
        colon = self.options.get("separator") == ":"
        if colon and m.group(2) == "_":
            t.apply("curie_underscore_to_colon", f"{m.group(1)}:{m.group(3)}")
        elif not colon and m.group(2) == ":":
            t.apply("curie_colon_to_underscore", f"{m.group(1)}_{m.group(3)}")
        t.upper()
        return t.done() if self.matches(t.value) else self.reject(t.value)
