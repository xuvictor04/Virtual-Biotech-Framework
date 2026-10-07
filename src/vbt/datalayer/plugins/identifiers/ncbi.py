"""``ncbi_taxon`` and ``ncbi_gene`` (phase 2): NCBI identifiers as digit strings.

``ncbi_taxon``: NCBI Taxonomy IDs as digit strings. Integer and string forms agree (``9606`` and
``"9606"`` give ``"9606"``); ``NCBITaxon:9606`` and ``taxon:9606`` lose their prefix.
``options.input_requires_prefix: true`` makes agent input carry a prefix (for arguments that also
accept other digit kinds, such as PMIDs).
"""

from __future__ import annotations

import re

from ..base import Normalized, Rejected
from ..registry import register
from . import KeyIdentifier, Trace

_PREFIX = re.compile(r"(?:NCBITaxon[:_]|taxon:|txid)\s*(\S+)", re.IGNORECASE)


@register
class NcbiTaxon(KeyIdentifier):
    name = id_type = "ncbi_taxon"
    canonical = r"^[1-9]\d*$"
    examples = ("9606", "10090")
    capabilities = frozenset({"options"})
    overlaps = frozenset({"pmid", "chromosome"})
    description = "NCBI Taxonomy ID (digits; integers and NCBITaxon: prefixes accepted)"
    cases = (
        {"raw": 9606, "expected": "9606", "steps": []},
        {"raw": "9606", "expected": "9606", "steps": []},
        {"raw": "NCBITaxon:10090", "expected": "10090", "steps": ["strip_prefix"]},
        {"raw": "Homo sapiens", "rejected": True},
        {"raw": "9606", "rejected": True, "options": {"input_requires_prefix": True}, "requires": "options"},
        {"stored": 9606, "expected": "9606", "options": {"input_requires_prefix": True}, "requires": "options"},
    )

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        m = _PREFIX.fullmatch(t.value)
        if m:
            t.apply("strip_prefix", m.group(1))
        elif not stored and self.options.get("input_requires_prefix"):
            return self.reject(t.value, "give taxon IDs with their prefix (NCBITaxon:9606) for this argument")
        return t.done() if self.matches(t.value) else self.reject(t.value, "NCBI taxon IDs are digits (e.g. 9606)")


_GENE_PREFIX = re.compile(r"(?:NCBI[_\s]?Gene|Entrez(?:[_\s]?Gene)?|GeneID)\s*[:_]\s*(\S+)", re.IGNORECASE)


@register
class NcbiGene(KeyIdentifier):
    """``ncbi_gene``: NCBI (Entrez) Gene IDs, stored as digits (DepMap ``"A1BG (1)"`` headers, GMT files
    keyed by Entrez). Bare digits look like PMIDs, taxon IDs and joinids, so a descriptor that accepts
    agent input declares ``options.input_requires_prefix: true``: agent input then needs
    ``NCBIGene:``/``entrez:`` (``NCBIGene:3845`` -> ``3845``, ``strip_prefix``) while stored values stay
    bare digits. The syntax overlap with ``pmid`` therefore holds for stored values only (§9.3 I-4)."""

    name = id_type = "ncbi_gene"
    canonical = r"^[1-9]\d*$"
    examples = ("3845", "7157")
    capabilities = frozenset({"options"})
    overlaps = frozenset({"pmid", "ncbi_taxon", "chromosome", "census_joinid"})
    description = "NCBI Gene (Entrez) ID: digits; agent input as NCBIGene:3845 or entrez:3845"
    cases = (
        {"raw": "3845", "expected": "3845", "steps": []},
        {"raw": 7157, "expected": "7157", "steps": []},
        {"raw": "NCBIGene:3845", "expected": "3845", "steps": ["strip_prefix"]},
        {"raw": "entrez:3845", "expected": "3845", "steps": ["strip_prefix"]},
        {"raw": " ncbigene:7157 ", "expected": "7157", "steps": ["strip", "strip_prefix"]},
        {"raw": "KRAS", "rejected": True},
        {"raw": "0123", "rejected": True},
        {"raw": "PMC1234", "rejected": True, "looks_like": "pmcid"},
        {"raw": "3845", "rejected": True, "options": {"input_requires_prefix": True}, "requires": "options"},
        {"raw": "NCBIGene:3845", "expected": "3845", "options": {"input_requires_prefix": True},
         "requires": "options"},
        {"stored": "3845", "expected": "3845", "options": {"input_requires_prefix": True}, "requires": "options"},
        {"stored": 3845.0, "expected": "3845", "options": {"input_requires_prefix": True}, "requires": "options"},
    )

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        m = _GENE_PREFIX.fullmatch(t.value)
        if m:
            t.apply("strip_prefix", m.group(1))
        elif not stored and self.options.get("input_requires_prefix"):
            return self.reject(t.value, "give NCBI Gene IDs with their prefix (NCBIGene:3845) for this argument; "
                                        "bare digits are ambiguous with PMIDs")
        if re.fullmatch(r"PMC\d+", t.value, re.IGNORECASE):
            return Rejected(f"{t.value} is a PMCID, not an NCBI Gene ID", ("pmcid",))
        return t.done() if self.matches(t.value) else self.reject(t.value, "NCBI Gene IDs are digits (e.g. 3845)")
