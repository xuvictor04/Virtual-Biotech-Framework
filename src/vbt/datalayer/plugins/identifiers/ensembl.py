"""Ensembl gene IDs: ``ensembl_gene`` (human), ``ensembl_gene_mouse`` and ``ensembl_gene_any`` (homologues).

Agent input is stripped, upper-cased and loses its ``.N`` version (``ENSG00000169174.12`` and
``ensg00000169174`` both give ``ENSG00000169174``). The ``_PAR_Y`` copy suffix is a declared
qualifier (``strip_suffix``); ``options.keep_suffix: true`` keeps it as a key part instead. A mouse
ID given for a human gene is rejected with ``looks_like: [ensembl_gene_mouse, ...]``.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar, Mapping, Self, Sequence

from ..base import Normalized, Rejected
from ..registry import register
from . import KeyIdentifier, Trace, set_attr

_ENS = re.compile(r"ENS[A-Z]*G\d{11}(?:\.\d+)?(?:_PAR_Y)?", re.IGNORECASE)
_VERSION = re.compile(r"(ENS[A-Z]*G\d{11})\.\d+((?:_PAR_Y)?)")
_PAR = "_PAR_Y"


class _Ensembl(KeyIdentifier):
    stem: ClassVar[str]                                # species stem pattern, e.g. "ENSG"

    def configure(self, options: Mapping[str, Any], universe_sample: Sequence[str] | None) -> Self:
        other = super().configure(options, universe_sample)
        if (options or {}).get("keep_suffix"):
            set_attr(other, "canonical", rf"^{self.stem}\d{{11}}(?:{_PAR})?$")
        return other

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        if not _ENS.fullmatch(t.value):
            if re.fullmatch(r"ENS[A-Z]*[TPE]\d{11}(?:\.\d+)?", t.value, re.IGNORECASE):
                return self.reject(t.value, f"an Ensembl transcript/protein/exon ID, not a gene ID "
                                            f"(e.g. {self.examples[0]})")
            return self.reject(t.value)
        t.upper()
        m = _VERSION.fullmatch(t.value)
        if m:
            t.apply("strip_version", m.group(1) + m.group(2))
        if t.value.endswith(_PAR) and not self.options.get("keep_suffix"):
            t.apply("strip_suffix", t.value[: -len(_PAR)])
        if self.matches(t.value):
            return t.done()
        return self.reject(t.value, f"not a {self.id_type} ID (e.g. {self.examples[0]})")


@register
class EnsemblGene(_Ensembl):
    name = id_type = "ensembl_gene"
    stem = "ENSG"
    canonical = r"^ENSG\d{11}$"
    examples = ("ENSG00000169174", "ENSG00000141510")
    capabilities = frozenset({"options"})
    overlaps = frozenset({"ensembl_gene_any", "tahoe_gene_name", "ot_entity_any"})
    description = "Ensembl human gene ID (version and case folded)"
    cases = (
        {"raw": "ENSG00000169174", "expected": "ENSG00000169174", "steps": []},
        {"raw": "ENSG00000169174.12", "expected": "ENSG00000169174", "steps": ["strip_version"]},
        {"raw": "ensg00000169174", "expected": "ENSG00000169174", "steps": ["upper"]},
        {"raw": " ENSG00000169174 ", "expected": "ENSG00000169174", "steps": ["strip"]},
        {"stored": "ENSG00000002586.18_PAR_Y", "expected": "ENSG00000002586"},
        {"raw": "ENSG00000002586_PAR_Y", "expected": "ENSG00000002586_PAR_Y", "options": {"keep_suffix": True}},
        {"raw": "ENSMUSG00000044254", "rejected": True, "looks_like": "ensembl_gene_mouse"},
        {"raw": "ENST00000302118", "rejected": True},
        {"raw": "PCSK9", "rejected": True},
        {"raw": "ENSG0000016917", "rejected": True},
    )


@register
class EnsemblGeneMouse(_Ensembl):
    name = id_type = "ensembl_gene_mouse"
    stem = "ENSMUSG"
    canonical = r"^ENSMUSG\d{11}$"
    examples = ("ENSMUSG00000044254",)
    capabilities = frozenset({"options"})
    overlaps = frozenset({"ensembl_gene_any"})
    description = "Ensembl mouse gene ID"
    cases = (
        {"raw": "ensmusg00000044254.3", "expected": "ENSMUSG00000044254", "steps": ["upper", "strip_version"]},
        {"raw": "ENSG00000169174", "rejected": True, "looks_like": "ensembl_gene"},
    )


@register
class EnsemblGeneAny(_Ensembl):
    name = id_type = "ensembl_gene_any"
    stem = "ENS[A-Z]*G"
    canonical = r"^ENS[A-Z]*G\d{11}$"
    examples = ("ENSMUSG00000044254", "ENSDARG00000000001")
    capabilities = frozenset({"options"})
    overlaps = frozenset({"ensembl_gene", "ensembl_gene_mouse", "tahoe_gene_name", "ot_entity_any"})
    description = "Ensembl gene ID of any species (homologues; syntax only)"
    cases = (
        {"raw": "ENSG00000169174.12", "expected": "ENSG00000169174"},
        {"raw": "ensdarg00000000001", "expected": "ENSDARG00000000001", "steps": ["upper"]},
        {"raw": "ENST00000302118", "rejected": True},
    )
