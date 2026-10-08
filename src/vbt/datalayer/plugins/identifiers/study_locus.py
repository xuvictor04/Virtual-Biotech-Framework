"""Open Targets genetics keys: ``study_locus_id`` (32 hex characters) and ``gwas_study``.

``gwas_study`` accepts GWAS Catalog accessions (``GCST004988``) and the other study ID families of
OT 25.09 (``FINNGEN_R12_I9_HYPTENS``, ``UKB_PPP_EUR_...``); ``options.patterns`` adds more.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Self, Sequence

from ..base import Normalized, Rejected
from ..registry import register
from . import KeyIdentifier, Trace, bare_pattern, set_attr


@register
class StudyLocusId(KeyIdentifier):
    name = id_type = "study_locus_id"
    canonical = r"^[0-9a-f]{32}$"
    examples = ("0a1b2c3d4e5f60718293a4b5c6d7e8f9",)
    description = "Open Targets study-locus ID (32 hex characters, lower case)"
    cases = (
        {"raw": "0A1B2C3D4E5F60718293A4B5C6D7E8F9", "expected": "0a1b2c3d4e5f60718293a4b5c6d7e8f9",
         "steps": ["lower"]},
        {"raw": "0a1b2c3d4e5f6071", "rejected": True},
    )

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        if re.fullmatch(r"[0-9a-fA-F]{32}", t.value):
            t.lower()
        return t.done() if self.matches(t.value) else self.reject(t.value)


#: UKB-PPP pQTL studies name the protein's gene, which may hold a hyphen (UKB_PPP_EUR_HLA-DRA_P01903_OID20520_v1:
#: 4 of the 2,954 in 25.09)
_STUDY = (r"GCST\d+", r"FINNGEN_R\d+_[A-Za-z0-9_]+", r"UKB_PPP_[A-Za-z0-9_\-]+")


@register
class GwasStudy(KeyIdentifier):
    name = id_type = "gwas_study"
    canonical = "^(?:" + "|".join(_STUDY) + ")$"
    examples = ("GCST004988", "FINNGEN_R12_I9_HYPTENS")
    capabilities = frozenset({"options"})
    description = "GWAS study ID (GWAS Catalog GCST accession or an OT study family such as FINNGEN_R12_...)"
    cases = (
        {"raw": "GCST004988", "expected": "GCST004988", "steps": []},
        {"raw": "gcst004988", "expected": "GCST004988", "steps": ["upper"]},
        {"raw": "UKB_PPP_EUR_HLA-DRA_P01903_OID20520_v1", "expected": "UKB_PPP_EUR_HLA-DRA_P01903_OID20520_v1",
         "steps": []},
        {"raw": "QTS000001_ENSG00000169174", "rejected": True},
        {"raw": "QTS000001_ENSG00000169174", "expected": "QTS000001_ENSG00000169174",
         "options": {"patterns": [r"QTS\d+_\w+"]}, "requires": "options"},
    )

    def configure(self, options: Mapping[str, Any], universe_sample: Sequence[str] | None) -> Self:
        other = super().configure(options, universe_sample)
        extra = tuple(bare_pattern(str(p)) for p in (options or {}).get("patterns", ()))
        if extra:
            set_attr(other, "canonical", "^(?:" + "|".join(_STUDY + extra) + ")$")
        return other

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        if re.fullmatch(r"gcst\d+", t.value, re.IGNORECASE):
            t.upper()
        return t.done() if self.matches(t.value) else self.reject(t.value)
