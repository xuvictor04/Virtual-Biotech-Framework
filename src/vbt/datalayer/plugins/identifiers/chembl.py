"""``chembl_molecule``: ChEMBL molecule IDs (``CHEMBL25``). Case and whitespace folded.

Parent/salt families (``canonicalize: {parent: drug_molecule.parentId}``) are an id_type facet the
resolver applies through the index's ``family`` column (§11.5), not plugin syntax.
"""

from __future__ import annotations

from ..base import Normalized, Rejected
from ..registry import register
from . import KeyIdentifier, Trace


@register
class ChemblMolecule(KeyIdentifier):
    name = id_type = "chembl_molecule"
    canonical = r"^CHEMBL\d+$"
    examples = ("CHEMBL25", "CHEMBL553")
    overlaps = frozenset({"ot_entity_any"})
    description = "ChEMBL molecule ID (case folded; salts and parents form families)"
    cases = (
        {"raw": "CHEMBL25", "expected": "CHEMBL25", "steps": []},
        {"raw": "chembl25", "expected": "CHEMBL25", "steps": ["upper"]},
        {"raw": " CHEMBL1079742", "expected": "CHEMBL1079742", "steps": ["strip"]},
        {"raw": "CHEMBL", "rejected": True},
        {"raw": "aspirin", "rejected": True},
        {"raw": "25", "rejected": True},
    )

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        if t.value.upper().startswith("CHEMBL"):
            t.upper()
        return t.done() if self.matches(t.value) else self.reject(t.value)
