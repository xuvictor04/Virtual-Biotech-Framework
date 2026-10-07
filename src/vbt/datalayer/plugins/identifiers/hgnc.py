"""``hgnc_symbol``: an HGNC gene symbol. A label kind (``label_of`` a gene id_type), not a key.

Symbols keep their case (``label_exact`` compares them as typed); ``label_key`` casefolds for
``label_casefold``. A value with the syntax of a structured kind (``ENSG...``, ``PMC1234``,
``rs7412``) is rejected and named in ``looks_like``. Some real symbols have UniProt accession
syntax (``P2RY12``), so the kinds overlap.
"""

from __future__ import annotations

from ..registry import register
from . import TextIdentifier, text_overlaps


@register
class HgncSymbol(TextIdentifier):
    name = id_type = "hgnc_symbol"
    canonical = r"^[A-Za-z0-9][A-Za-z0-9._@/#\-]*$"
    examples = ("PCSK9", "TP53")
    capabilities = frozenset({"label"})
    overlaps = text_overlaps("hgnc_symbol", "uniprot_accession")
    description = "HGNC gene symbol (a label resolved to the gene's ID; case kept, then casefolded)"
    cases = (
        {"raw": "PCSK9", "expected": "PCSK9", "steps": []},
        {"raw": " pcsk9 ", "expected": "pcsk9", "steps": ["strip"]},
        {"raw": "HLA-A", "expected": "HLA-A"},
        {"raw": "C1orf112", "expected": "C1orf112"},
        {"raw": "P2RY12", "expected": "P2RY12"},
        {"raw": "RS1", "expected": "RS1"},
        {"raw": "ENSG00000169174", "rejected": True, "looks_like": "ensembl_gene"},
        {"raw": "rs7412", "rejected": True, "looks_like": "rsid"},
        {"raw": "PMC1234", "rejected": True, "looks_like": "pmcid"},
        {"raw": "12345", "rejected": True, "looks_like": "pmid"},
        {"raw": "type 2 diabetes", "rejected": True},
        {"label": "PCSK9", "key": "pcsk9"},
    )

    def syntax_reason(self, value: str) -> str:
        return "a gene symbol is one token of letters, digits and -._@/# (e.g. PCSK9)"
