"""Tahoe-100M names: ``tahoe_drug`` and ``tahoe_gene_name``.

``tahoe_drug`` is a name kind: Tahoe's DE table stores ``'Erdafitinib '`` where the metadata stores
``'Erdafitinib'`` (VERIFIED), so ``normalize_stored`` strips whitespace and the resolver index keeps
each table's stored spelling (``stored_forms``) for ``send_as: stored``. ``label_key`` also drops a
trailing parenthesised salt or code suffix (``Erlotinib (hydrochloride)`` -> ``erlotinib``), so
salt variants become candidates of one name instead of misses.

``tahoe_gene_name`` is the DE table's ``gene_name``: a symbol, or a bare Ensembl ID for genes
without one; Ensembl IDs lose their version.
"""

from __future__ import annotations

import re
import unicodedata

from ..base import Normalized, Rejected
from ..registry import register
from . import TextIdentifier, Trace, text_overlaps

_SALT = re.compile(r"\s*\([^()]*\)\s*$")


@register
class TahoeDrug(TextIdentifier):
    name = id_type = "tahoe_drug"
    canonical = r"^\S(?:[^\t\r\n]*\S)?$"
    examples = ("Erdafitinib", "Abemaciclib")
    capabilities = frozenset({"label"})
    overlaps = text_overlaps("tahoe_drug")
    description = "Tahoe-100M drug name (whitespace and case folded; salt suffixes give candidates)"
    cases = (
        {"raw": "Erdafitinib", "expected": "Erdafitinib", "steps": []},
        {"stored": "Erdafitinib ", "expected": "Erdafitinib", "steps": ["strip"]},
        {"raw": " erdafitinib", "expected": "erdafitinib", "steps": ["strip"]},
        {"label": "Erlotinib (hydrochloride)", "key": "erlotinib"},
        {"label": " Erdafitinib ", "key": "erdafitinib"},
        {"raw": "CHEMBL25", "rejected": True, "looks_like": "chembl_molecule"},
    )

    def label_key(self, raw: str) -> str:
        text = unicodedata.normalize("NFKC", str(raw)).strip()
        text = _SALT.sub("", text) or text
        return text.strip().casefold()


@register
class TahoeGeneName(TextIdentifier):
    name = id_type = "tahoe_gene_name"
    canonical = r"^(?:ENSG\d{11}|[A-Za-z0-9][A-Za-z0-9._@/#\-]*)$"
    examples = ("PCSK9", "SEPT9")
    capabilities = frozenset({"label"})
    overlaps = text_overlaps("tahoe_gene_name", "ensembl_gene", "ensembl_gene_any", "uniprot_accession",
                             "ot_entity_any")
    description = "Tahoe-100M gene name (a symbol, or a bare Ensembl ID for genes without one)"
    cases = (
        {"raw": "PCSK9", "expected": "PCSK9", "steps": []},
        {"raw": "ENSG00000169174.3", "expected": "ENSG00000169174", "steps": ["strip_version"]},
        {"raw": "ensg00000169174", "expected": "ENSG00000169174", "steps": ["upper"]},
        {"raw": "ENSMUSG00000044254", "rejected": True},
        {"raw": "rs7412", "rejected": True, "looks_like": "rsid"},
    )

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        if re.fullmatch(r"ENS[A-Z]*G\d{11}(?:\.\d+)?", t.value, re.IGNORECASE):
            t.upper()
            m = re.fullmatch(r"(ENSG\d{11})\.\d+", t.value)
            if m:
                t.apply("strip_version", m.group(1))
            if not re.fullmatch(r"ENSG\d{11}", t.value):
                return Rejected(f"{t.value} is not a human Ensembl gene ID", ("ensembl_gene_any",))
            return t.done()
        return self._check(t, stored)
