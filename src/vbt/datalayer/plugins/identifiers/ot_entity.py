"""``ot_entity_any``: the syntax of a union id_type (``IdTypeSpec.union``), e.g. the words of OT
``literature_vector`` (gene, disease and drug IDs in one column).

It accepts what any member accepts, trying members in order (default ``ensembl_gene``,
``ot_disease``, ``chembl_molecule``; ``options.members`` names others, and the remaining options
are passed to every member, e.g. ``prefixes: from_universe``). Which member a stored value must
satisfy is the column's ``kind_from`` facet; resolving a value against several members is the
resolver's job (several distinct canonicals are ``ambiguous``).
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping, Self, Sequence

from ..base import Normalized, PluginError, Rejected
from ..registry import register, registered
from . import KeyIdentifier, bare_pattern, set_attr
from .chembl import ChemblMolecule
from .ensembl import EnsemblGene
from .ot_disease import OtDisease

_DEFAULT = (EnsemblGene, OtDisease, ChemblMolecule)


def _union_pattern(members: Sequence[Any]) -> str:
    return "^(?:" + "|".join(f"(?:{bare_pattern(m.canonical)})" for m in members) + ")$"


def _member_overlaps(classes: Sequence[type]) -> frozenset[str]:
    out: set[str] = set()
    for c in classes:
        out.add(getattr(c, "name"))
        out.update(getattr(c, "overlaps", ()))
    return frozenset(out - {"ot_entity_any"})


@register
class OtEntityAny(KeyIdentifier):
    name = id_type = "ot_entity_any"
    canonical = _union_pattern(_DEFAULT)
    examples = ("ENSG00000169174", "EFO_0000685", "CHEMBL25")
    capabilities = frozenset({"union", "options"})
    overlaps: ClassVar[frozenset[str]] = _member_overlaps(_DEFAULT)
    description = "an Open Targets entity ID of any member kind (gene, disease or drug)"
    cases = (
        {"raw": "ENSG00000169174.3", "expected": "ENSG00000169174", "steps": ["strip_version"]},
        {"raw": "EFO:0000685", "expected": "EFO_0000685", "steps": ["curie_colon_to_underscore"]},
        {"raw": "chembl25", "expected": "CHEMBL25", "steps": ["upper"]},
        {"raw": "PCSK9", "rejected": True},
        {"raw": "R-HSA-109582", "expected": "R-HSA-109582", "options": {"members": ["reactome"]},
         "requires": "options"},
    )

    def __init__(self) -> None:
        super().__init__()
        self.members: tuple[Any, ...] = tuple(c() for c in _DEFAULT)

    def configure(self, options: Mapping[str, Any], universe_sample: Sequence[str] | None) -> Self:
        options = dict(options or {})
        names = options.pop("members", None)
        classes: Sequence[type] = _DEFAULT
        if names:
            known = {getattr(c, "name", None): c for c in registered() if getattr(c, "kind", None) == "identifier"}
            missing = [n for n in names if n not in known or n == self.name]
            if missing:
                raise PluginError(f"ot_entity_any options.members: unknown identifier plugins {missing}")
            classes = [known[n] for n in names]
        other = super().configure(options, universe_sample)
        other.members = tuple(c().configure(options, universe_sample) for c in classes)
        set_attr(other, "canonical", _union_pattern(other.members))
        return other

    def member_of(self, raw: Any) -> str | None:
        """The first member kind that accepts ``raw``."""
        for m in self.members:
            if isinstance(m.normalize(raw), Normalized):
                return str(m.name)
        return None

    def normalize(self, raw: Any) -> Normalized | Rejected:
        return self._first(raw, False)

    def normalize_stored(self, stored: Any) -> Normalized | Rejected:
        return self._first(stored, True)

    def _first(self, raw: Any, stored: bool) -> Normalized | Rejected:
        reasons: list[str] = []
        hints: list[str] = []
        for m in self.members:
            n = m.normalize_stored(raw) if stored else m.normalize(raw)
            if isinstance(n, Normalized):
                return n
            reasons.append(f"{m.name}: {n.reason}")
            hints.extend(h for h in n.looks_like if h not in hints)
        return Rejected("not an ID of any member kind (" + "; ".join(reasons) + ")", tuple(hints))
