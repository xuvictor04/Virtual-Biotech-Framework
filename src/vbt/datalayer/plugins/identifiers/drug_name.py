"""``drug_name``: a drug name, trade name or synonym (a label kind: ``label_of`` a molecule id_type).

Trade names often name several salt forms of one parent (``Tarceva``); the resolver collapses such
candidates to the parent (rule ``parent_family``, §11.5).
"""

from __future__ import annotations

from ..registry import register
from . import TextIdentifier, text_overlaps


@register
class DrugName(TextIdentifier):
    name = id_type = "drug_name"
    canonical = r"^\S(?:[^\t\r\n]*\S)?$"
    examples = ("erlotinib", "Tarceva")
    capabilities = frozenset({"label"})
    overlaps = text_overlaps("drug_name")
    description = "drug name, trade name or synonym (a label resolved to the molecule ID)"
    cases = (
        {"raw": "Tarceva ", "expected": "Tarceva", "steps": ["strip"]},
        {"raw": "erlotinib hydrochloride", "expected": "erlotinib hydrochloride"},
        {"raw": "CHEMBL553", "rejected": True, "looks_like": "chembl_molecule"},
        {"raw": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "rejected": True, "looks_like": "inchikey"},
        {"label": "TARCEVA", "key": "tarceva"},
    )
