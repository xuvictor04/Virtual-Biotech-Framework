"""``cell_ontology`` and ``uberon``: Cell Ontology and UBERON term IDs (phase 2), wrapping
:mod:`vbt.analysis.ontology`.

Canonical forms are the OBO CURIEs ``CL:0000057`` and ``UBERON:0002048`` (the forms CELLxGENE stores
in ``cell_type_ontology_term_id``/``tissue_ontology_term_id`` and ``mini_cl.obo`` uses); ``CL_0000057``
(``curie_underscore_to_colon``), lower case (``upper``) and OBO PURLs (``strip_prefix``) normalise to
them. ``options.separator: "_"`` makes the underscore form canonical for sources that store it (Open
Targets stores ``UBERON_0002048`` in ``disease.id``, so ``uberon`` and ``ot_disease`` overlap).

The hierarchy helpers wrap the harness's Cell Ontology loader (:func:`vbt.analysis.ontology.load_cell_ontology`,
``is_a`` edges, obsolete terms excluded): :meth:`CellOntologyTerm.ontology` loads the OBO named by
``options.obo`` (default ``$VBT_CL_OBO``) once per path, and :meth:`CellOntologyTerm.ancestors` /
:meth:`CellOntologyTerm.label` answer on canonical IDs. They import that module (and pandas) when
called, so they run in the data child only; normalisation stays stdlib-only.
"""

from __future__ import annotations

import functools
from typing import Any

from ..registry import register
from .go import OboTerm

__all__ = ["CellOntologyTerm", "UberonTerm"]


@functools.lru_cache(maxsize=4)
def _load(path: str | None) -> Any:
    from ....analysis.ontology import load_cell_ontology

    return load_cell_ontology(path, engine="builtin")


@register
class CellOntologyTerm(OboTerm):
    name = id_type = "cell_ontology"
    prefix = "CL"
    canonical = r"^CL:\d{7}$"
    examples = ("CL:0000057", "CL:0000084")
    description = "Cell Ontology term CL:NNNNNNN (CL_ and OBO PURLs folded)"
    cases = (
        {"raw": "CL:0000057", "expected": "CL:0000057", "steps": []},
        {"raw": "CL_0000057", "expected": "CL:0000057", "steps": ["curie_underscore_to_colon"]},
        {"raw": "cl:0000057", "expected": "CL:0000057", "steps": ["upper"]},
        {"raw": "http://purl.obolibrary.org/obo/CL_0000057", "expected": "CL:0000057",
         "steps": ["strip_prefix", "curie_underscore_to_colon"]},
        {"raw": "UBERON:0002048", "rejected": True},
        {"raw": "fibroblast", "rejected": True},
        {"raw": "CL:57", "rejected": True},
        {"raw": "CL:0000057", "expected": "CL_0000057", "options": {"separator": "_"}, "requires": "options"},
        {"raw": "CP:0000001", "rejected": True},
        # cl-basic.obo keeps 9 obsolete CP: terms (moved into CL): the descriptor names the prefix
        {"raw": "cp:0000001", "expected": "CP:0000001", "options": {"also_prefixes": ["CP"]}, "requires": "options"},
    )

    # -- hierarchy (wraps vbt.analysis.ontology; data child only) -------------------------------

    def ontology(self) -> Any:
        """The Cell Ontology of ``options.obo`` (default ``$VBT_CL_OBO``), loaded once per path."""
        return _load(self.options.get("obo"))

    def ancestors(self, term: str, include_self: bool = False) -> frozenset[str]:
        """``is_a`` ancestors of a term (canonical CURIEs); unknown terms have none."""
        from ..base import Normalized

        n = self.normalize(term)
        if not isinstance(n, Normalized):
            return frozenset()
        return frozenset(self.ontology().ancestors(n.value, include_self=include_self))

    def label(self, term: str) -> str | None:
        onto = self.ontology()
        return onto.names.get(term)


@register
class UberonTerm(OboTerm):
    name = id_type = "uberon"
    prefix = "UBERON"
    canonical = r"^UBERON:\d{7}$"
    examples = ("UBERON:0002048", "UBERON:0000955")
    overlaps = frozenset({"ot_disease", "ot_entity_any"})
    description = "UBERON anatomy term UBERON:NNNNNNN (UBERON_ and OBO PURLs folded)"
    cases = (
        {"raw": "UBERON:0002048", "expected": "UBERON:0002048", "steps": []},
        {"raw": "UBERON_0002048", "expected": "UBERON:0002048", "steps": ["curie_underscore_to_colon"]},
        {"raw": "http://purl.obolibrary.org/obo/UBERON_0002048", "expected": "UBERON:0002048",
         "steps": ["strip_prefix", "curie_underscore_to_colon"]},
        {"raw": "CL:0000057", "rejected": True},
        {"raw": "lung", "rejected": True},
        {"raw": "UBERON:0002048", "expected": "UBERON_0002048", "options": {"separator": "_"},
         "requires": "options"},
    )
