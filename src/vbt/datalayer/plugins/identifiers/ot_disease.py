"""``ot_disease``: Open Targets disease/phenotype IDs (``EFO_0000685``), and ``disease_name`` (its label kind).

OT ``disease.id`` holds terms of many ontologies: 29.5% of the 25.09 IDs use prefixes outside a
short EFO/MONDO/Orphanet list (OBA, GO, MP, NCIT, OBI, GSSO, PATO, OGMS, UBERON ...). The prefix
set is therefore an option: ``prefixes: [..]`` or ``prefixes: from_universe`` (learned from the
universe sample passed to :meth:`OtDisease.configure`). Normalisation: IRIs lose their namespace
(``strip_prefix``), ``EFO:0000685`` becomes ``EFO_0000685`` (``curie_colon_to_underscore``) and the
prefix gets its canonical case (``orphanet_558`` -> ``Orphanet_558``, ``canonical_prefix_case``).
GO, HP and UBERON terms are legitimately disease IDs here, so those kinds overlap.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Self, Sequence

from ..base import Normalized, Rejected
from ..registry import register
from . import KeyIdentifier, TextIdentifier, Trace, set_attr, text_overlaps

DEFAULT_PREFIXES: tuple[str, ...] = ("EFO", "MONDO", "Orphanet", "HP", "OTAR", "DOID", "OBA", "GO", "MP", "NCIT",
                                     "OBI", "GSSO", "PATO", "OGMS", "UBERON")
_IRI = re.compile(r"https?://\S+/([A-Za-z][A-Za-z0-9]*[_:][A-Za-z0-9]+)")
_CURIE = re.compile(r"([A-Za-z][A-Za-z0-9]*)([_:])([A-Za-z0-9]+)")


def _pattern(prefixes: Sequence[str]) -> str:
    alts = "|".join(re.escape(p) for p in sorted(prefixes, key=lambda p: (-len(p), p)))
    return rf"^(?:{alts})_[A-Za-z0-9]+$"


def learn_prefixes(sample: Sequence[str] | None) -> tuple[str, ...]:
    """Prefixes of the ``PREFIX_local`` keys in a universe sample, in first-seen order and case."""
    seen: dict[str, str] = {}
    for key in sample or ():
        m = _CURIE.fullmatch(str(key).strip())
        if m and m.group(2) == "_":
            seen.setdefault(m.group(1).casefold(), m.group(1))
    return tuple(seen.values())


@register
class OtDisease(KeyIdentifier):
    name = id_type = "ot_disease"
    canonical = _pattern(DEFAULT_PREFIXES)
    examples = ("EFO_0000685", "MONDO_0005148", "Orphanet_558")
    capabilities = frozenset({"options"})
    overlaps = frozenset({"go", "hpo", "uberon", "ot_entity_any"})
    description = "Open Targets disease or phenotype ID PREFIX_local (EFO, MONDO, Orphanet, ...; CURIEs and IRIs folded)"
    example_options = {"prefixes": "from_universe"}
    example_universe = ("EFO_0000685", "MONDO_0005148", "Orphanet_558", "OBA_VT0000047", "GO_0005737", "HP_0001250",
                        "MP_0001186", "NCIT_C4872", "OTAR_0000006", "UBERON_0002107", "DOID_9352", "OGMS_0000031")
    cases = (
        {"raw": "EFO_0000685", "expected": "EFO_0000685", "steps": []},
        {"raw": "EFO:0000685", "expected": "EFO_0000685", "steps": ["curie_colon_to_underscore"]},
        {"raw": "orphanet_558", "expected": "Orphanet_558", "steps": ["canonical_prefix_case"]},
        {"raw": "ORPHANET:558", "expected": "Orphanet_558",
         "steps": ["curie_colon_to_underscore", "canonical_prefix_case"]},
        {"raw": "http://www.ebi.ac.uk/efo/EFO_0000685", "expected": "EFO_0000685", "steps": ["strip_prefix"]},
        {"raw": "http://purl.obolibrary.org/obo/MONDO_0005148", "expected": "MONDO_0005148"},
        {"raw": "GO_0005737", "expected": "GO_0005737"},
        {"raw": "NCIT_c4872", "expected": "NCIT_C4872", "steps": ["upper"]},
        {"raw": "MESH:D003924", "rejected": True},
        {"raw": "CHEMBL25", "rejected": True, "looks_like": "chembl_molecule"},
        {"raw": "rheumatoid arthritis", "rejected": True},
        {"raw": "OBA_VT0000047", "expected": "OBA_VT0000047", "options": {"prefixes": "from_universe"},
         "universe_sample": ["OBA_VT0000047", "EFO_0000685"], "requires": "options"},
        {"raw": "MONDO_0005148", "rejected": True, "options": {"prefixes": "from_universe"},
         "universe_sample": ["OBA_VT0000047", "EFO_0000685"], "requires": "options"},
        {"raw": "efo_0000685", "expected": "EFO_0000685", "options": {"prefixes": ["EFO"]}, "requires": "options"},
    )

    def __init__(self) -> None:
        super().__init__()
        self.prefixes: tuple[str, ...] = DEFAULT_PREFIXES
        self._case = {p.casefold(): p for p in DEFAULT_PREFIXES}

    def configure(self, options: Mapping[str, Any], universe_sample: Sequence[str] | None) -> Self:
        other = super().configure(options, universe_sample)
        spec = (options or {}).get("prefixes")
        if spec == "from_universe":
            prefixes = learn_prefixes(universe_sample) or DEFAULT_PREFIXES
        elif spec:
            prefixes = tuple(str(p) for p in spec)
        else:
            prefixes = DEFAULT_PREFIXES
        other.prefixes = prefixes
        other._case = {p.casefold(): p for p in prefixes}
        set_attr(other, "canonical", _pattern(prefixes))
        return other

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        m = _IRI.fullmatch(t.value)
        if m:
            t.apply("strip_prefix", m.group(1))
        m = _CURIE.fullmatch(t.value)
        if m is None:
            return self.reject(t.value)
        prefix, sep, local = m.groups()
        canon = self._case.get(prefix.casefold())
        if canon is None:
            shown = ", ".join(self.prefixes[:8]) + (", ..." if len(self.prefixes) > 8 else "")
            return self.reject(t.value, f"prefix {prefix!r} is not a disease ID prefix of this source ({shown})",
                               form=f"a cross-reference in the {prefix} namespace")
        if sep == ":":
            t.apply("curie_colon_to_underscore", f"{prefix}_{local}")
        t.apply("canonical_prefix_case", f"{canon}_{local}")
        t.apply("upper", f"{canon}_{local.upper()}")
        return t.done()


@register
class DiseaseName(TextIdentifier):
    name = id_type = "disease_name"
    canonical = r"^\S(?:[^\t\r\n]*\S)?$"
    examples = ("rheumatoid arthritis", "type 2 diabetes mellitus")
    capabilities = frozenset({"label"})
    overlaps = text_overlaps("disease_name")
    description = "disease or phenotype name (a label resolved to the disease ID by name and synonyms)"
    cases = (
        {"raw": " Rheumatoid arthritis ", "expected": "Rheumatoid arthritis", "steps": ["strip"]},
        {"raw": "EFO_0000685", "rejected": True, "looks_like": "ot_disease"},
        {"raw": "MONDO:0005148", "rejected": True, "looks_like": "ot_disease"},
        {"label": "Type 2 Diabetes Mellitus", "key": "type 2 diabetes mellitus"},
    )
