"""``chembl_molecule``: ChEMBL molecule IDs (``CHEMBL25``). Case and whitespace folded; a ``:`` or
``_`` between the prefix and the number (``CHEMBL:25``, a CURIE form) is dropped.

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
        {"raw": "CHEMBL:25", "expected": "CHEMBL25", "steps": ["separator_to_underscore"]},
        {"raw": "chembl_25", "expected": "CHEMBL25", "steps": ["upper", "separator_to_underscore"]},
        {"raw": "CHEMBL", "rejected": True},
        {"raw": "aspirin", "rejected": True},
        {"raw": "25", "rejected": True},
    )

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        if t.value.upper().startswith("CHEMBL"):
            t.upper()
            if t.value[6:7] in (":", "_") and t.value[7:].isdigit():
                t.apply("separator_to_underscore", "CHEMBL" + t.value[7:])
        return t.done() if self.matches(t.value) else self.reject(t.value)
