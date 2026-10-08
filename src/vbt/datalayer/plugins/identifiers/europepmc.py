"""``europepmc_ppr``: Europe PMC preprint IDs (``PPR123456``), which OT evidence ``literature`` lists
next to PMIDs."""

from __future__ import annotations

import re

from ..base import Normalized, Rejected
from ..registry import register
from . import KeyIdentifier, Trace


@register
class EuropePmcPreprint(KeyIdentifier):
    name = id_type = "europepmc_ppr"
    canonical = r"^PPR\d+$"
    examples = ("PPR123456",)
    description = "Europe PMC preprint ID PPRn"
    cases = (
        {"raw": "PPR123456", "expected": "PPR123456", "steps": []},
        {"raw": "ppr123456", "expected": "PPR123456", "steps": ["upper"]},
        {"raw": "123456", "rejected": True},
    )

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        if re.fullmatch(r"ppr\d+", t.value, re.IGNORECASE):
            t.upper()
        return t.done() if self.matches(t.value) else self.reject(t.value)


@register
class EuropePmcId(KeyIdentifier):
    """``europepmc_id``: the ID Europe PMC gives an article, as Open Targets ``literature.pmid`` stores it: the
    PubMed ID (digits) for MEDLINE records, else a source-prefixed ID. All 151,961,320 rows of 25.09: digits
    except 2,349,085 rows (759,406 IDs) with PPR (preprints, 499,875 IDs), IND (Agricola, 178,715), PMC
    (full text without a PMID, 80,801), CAIN (8), c (Chinese Biological Abstracts, 6) and FNI (1). A PMID is
    accepted with its ``PMID:`` prefix; a prefix not listed here is rejected (readiness R4b names it)."""

    name = id_type = "europepmc_id"
    canonical = r"^(?:[1-9]\d{0,8}|(?:PPR|PMC|IND|CAIN|FNI|c)\d+)$"
    examples = ("30595370", "PPR123456", "PMC10028347")
    overlaps = frozenset({"pmid", "pmcid", "europepmc_ppr", "ncbi_taxon", "chromosome", "ncbi_gene", "census_joinid"})
    description = "Europe PMC article ID: a PubMed ID, or PPR/PMC/IND/CAIN/FNI/c followed by digits"
    cases = (
        {"raw": "30595370", "expected": "30595370", "steps": []},
        {"raw": "PMID:30595370", "expected": "30595370", "steps": ["strip_prefix"]},
        {"raw": "PPR100056", "expected": "PPR100056", "steps": []},
        {"raw": "ppr100056", "expected": "PPR100056", "steps": ["upper"]},
        {"raw": "IND20360743", "expected": "IND20360743", "steps": []},
        {"raw": "PMC10028347", "expected": "PMC10028347", "steps": []},
        {"raw": "c8345", "expected": "c8345", "steps": []},
        {"raw": "0123", "rejected": True},
        {"raw": "XYZ123", "rejected": True},
        {"raw": "10.1056/NEJMoa1615664", "rejected": True},
    )

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        m = re.fullmatch(r"PMID\s*[:_]?\s*(\S+)", t.value, re.IGNORECASE)
        if m:
            t.apply("strip_prefix", m.group(1))
        elif re.fullmatch(r"(?:ppr|pmc|ind|cain|fni)\d+", t.value, re.IGNORECASE) and t.value != t.value.upper():
            t.upper()
        return t.done() if self.matches(t.value) else self.reject(
            t.value, "Europe PMC IDs are PubMed IDs (digits) or PPR/PMC/IND/CAIN/FNI/c followed by digits")
