"""CELLxGENE Census feature IDs (phase 4): ``census_feature``.

The Census ``var`` dataframe keys genes by ``feature_id``, an Ensembl gene ID of the experiment's
organism: ``ENSG...`` for ``homo_sapiens`` (the default), ``ENSMUSG...`` with
``options.organism: mus_musculus`` (``any`` accepts both). AnnData written by Census tools index ``var``
by position, so the feature ID must come from ``var.feature_id``, never from ``var_names`` (``0``,
``1``, ...: the descriptor's ``forbid_positional_index``). Input is normalised like an Ensembl ID
(strip, upper case, ``.N`` version dropped). The syntax is Ensembl's, so the kind declares that
overlap: existence is decided by the Census ``var`` universe (``existence: upstream`` in the
single_cell overlay), not by syntax.
"""

from __future__ import annotations

from typing import Any, Mapping, Self, Sequence

from ..registry import register
from . import set_attr
from .ensembl import _Ensembl

_STEMS = {"homo_sapiens": "ENSG", "mus_musculus": "ENSMUSG", "any": "ENS(?:MUS)?G"}
_EXAMPLES = {"homo_sapiens": ("ENSG00000169174", "ENSG00000141510"), "mus_musculus": ("ENSMUSG00000059552",),
             "any": ("ENSG00000169174", "ENSMUSG00000059552")}


@register
class CensusFeature(_Ensembl):
    name = id_type = "census_feature"
    stem = "ENSG"
    canonical = r"^ENSG\d{11}$"
    examples = _EXAMPLES["homo_sapiens"]
    capabilities = frozenset({"options"})
    overlaps = frozenset({"ensembl_gene", "ensembl_gene_mouse", "ensembl_gene_any", "tahoe_gene_name",
                          "ot_entity_any"})
    description = "CELLxGENE Census var feature_id (an Ensembl gene ID of the Census organism)"
    cases = (
        {"raw": "ENSG00000141510", "expected": "ENSG00000141510", "steps": []},
        {"raw": "ensg00000141510.17", "expected": "ENSG00000141510", "steps": ["upper", "strip_version"]},
        {"raw": "ENSMUSG00000059552", "rejected": True, "looks_like": "ensembl_gene_mouse"},
        {"raw": "ENSMUSG00000059552", "expected": "ENSMUSG00000059552", "options": {"organism": "mus_musculus"},
         "requires": "options"},
        {"raw": "ENSG00000141510", "expected": "ENSG00000141510", "options": {"organism": "any"},
         "requires": "options"},
        {"raw": "0", "rejected": True},
        {"raw": "TP53", "rejected": True},
        {"raw": "ENST00000269305", "rejected": True},
    )

    def configure(self, options: Mapping[str, Any], universe_sample: Sequence[str] | None) -> Self:
        other = super().configure(options, universe_sample)
        organism = str((options or {}).get("organism") or "").strip().lower()
        if organism in _STEMS:
            stem = _STEMS[organism]
            set_attr(other, "stem", stem)
            set_attr(other, "canonical", rf"^{stem}\d{{11}}$")
            set_attr(other, "examples", _EXAMPLES[organism])
        return other
