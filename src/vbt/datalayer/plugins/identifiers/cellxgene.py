"""CELLxGENE identifiers (phase 2): ``census_joinid`` and ``cell_barcode``.

* ``census_joinid``: a SOMA ``soma_joinid`` (a non-negative integer; ``0`` is valid). Integers and
  digit strings agree (``123`` and ``"123"`` give ``"123"``). Joinids are positions in one Census
  release, so they are opaque keys (``resolvable: false`` in descriptors) and overlap every digits-only
  kind.
* ``cell_barcode``: a droplet barcode, 10x style ``AAACCTGAGAAACCAT-1`` (bases ``ACGTN``, an optional
  ``-<gem well>`` suffix), canonical upper case. ``options.canonical`` replaces the pattern for other
  barcoding schemes (``cell_barcode`` keys are unique only within a dataset: descriptors declare
  ``unique_within``). The 10x syntax is in :data:`~vbt.datalayer.plugins.identifiers.STRUCTURED`, so the
  free-syntax kinds reject barcodes.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Self, Sequence

from ..base import Normalized, Rejected
from ..registry import register
from . import KeyIdentifier, Trace, configured_examples, set_attr


@register
class CensusJoinid(KeyIdentifier):
    name = id_type = "census_joinid"
    canonical = r"^(?:0|[1-9]\d*)$"
    examples = ("0", "123456")
    overlaps = frozenset({"pmid", "ncbi_taxon", "chromosome", "ncbi_gene"})
    description = "CELLxGENE Census soma_joinid (a non-negative integer of one Census release)"
    cases = (
        {"raw": "0", "expected": "0", "steps": []},
        {"raw": 123456, "expected": "123456", "steps": []},
        {"raw": " 42 ", "expected": "42", "steps": ["strip"]},
        {"raw": "-1", "rejected": True},
        {"raw": "007", "rejected": True},
        {"raw": "ENSG00000169174", "rejected": True, "looks_like": "ensembl_gene"},
    )


@register
class CellBarcode(KeyIdentifier):
    name = id_type = "cell_barcode"
    canonical = r"^[ACGTN]{8,}(?:-\d+)?$"
    examples = ("AAACCTGAGAAACCAT-1", "TTTGTCATCTTGCAGA-2")
    capabilities = frozenset({"options"})
    description = "droplet cell barcode (10x style ACGT bases with an optional -N gem-well suffix)"
    cases = (
        {"raw": "AAACCTGAGAAACCAT-1", "expected": "AAACCTGAGAAACCAT-1", "steps": []},
        {"raw": "aaacctgagaaaccat-1", "expected": "AAACCTGAGAAACCAT-1", "steps": ["upper"]},
        {"raw": "AAACCTGAGAAACCAT", "expected": "AAACCTGAGAAACCAT", "steps": []},
        {"raw": "ACGT", "rejected": True},
        {"raw": "TP53", "rejected": True},
        {"raw": "cell_17", "expected": "cell_17", "options": {"canonical": "^cell_\\d+$"}, "requires": "options"},
    )

    def configure(self, options: Mapping[str, Any], universe_sample: Sequence[str] | None) -> Self:
        other = super().configure(options, universe_sample)
        canonical = (options or {}).get("canonical")
        if canonical:
            set_attr(other, "canonical", str(canonical))
            set_attr(other, "examples", configured_examples(options, type(self).examples, str(canonical)))
        return other

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        if not self.options.get("canonical") and re.fullmatch(r"[acgtnACGTN]{8,}(?:-\d+)?", t.value):
            t.upper()
        return t.done() if self.matches(t.value) else self.reject(t.value)
