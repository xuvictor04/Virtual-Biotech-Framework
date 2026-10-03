"""Cell Ontology (CL) harmonisation of cell-type labels (case studies 2 and 3).

The paper harmonised CELLxGENE cell types hierarchically: DE and LIANA used a
fine "Level 3" annotation (e.g. fibroblast subtypes pooled into *fibroblast*)
and the cross-disease survey a coarse "Level 1" compartment (immune, stromal,
epithelial, endothelial, neural, ...). This module

* loads the CL from an OBO file — with ``pronto`` or ``obonet`` when installed,
  else a small built-in OBO parser (``is_a`` edges only; obsolete terms
  skipped) — see :func:`load_cell_ontology`;
* computes the depth of each term (shortest ``is_a`` path to the root
  ``CL:0000000`` *cell*) and the Level-k ancestor of a term: the anchor of
  level k it descends from when curated anchors are given for that level
  (:data:`LEVEL1_ANCHORS` by default for Level 1), else its ancestor at depth k
  (terms shallower than k map to themselves) — :meth:`CellOntology.level`;
* maps ``cell_type_ontology_term_id`` values (as in CELLxGENE Census ``obs``)
  to Level-1 / Level-3 labels: :meth:`CellOntology.map_terms` and
  :func:`add_level_labels`.

``census_query_plan`` in :mod:`vbt.analysis.cross_disease` groups by the Level-1
label; pass ``level=3`` columns to pseudobulk DE / LIANA.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

__all__ = ["CellOntology", "LEVEL1_ANCHORS", "ROOT", "load_cell_ontology", "parse_obo", "add_level_labels"]

ROOT = "CL:0000000"

#: Curated Level-1 compartments, in priority order (the first anchor a term
#: descends from wins: endothelial cells are is_a epithelial in CL, leukocytes
#: are is_a hematopoietic, so the more specific compartments come first).
LEVEL1_ANCHORS: list[tuple[str, str]] = [
    ("CL:0000738", "immune"),          # leukocyte
    ("CL:0000988", "hematopoietic"),   # hematopoietic cell (erythroid, megakaryocyte, ...)
    ("CL:0000115", "endothelial"),     # endothelial cell
    ("CL:0000066", "epithelial"),      # epithelial cell
    ("CL:0000057", "stromal"),         # fibroblast
    ("CL:0000669", "stromal"),         # pericyte
    ("CL:0000499", "stromal"),         # stromal cell
    ("CL:0000187", "muscle"),          # muscle cell
    ("CL:0002319", "neural"),          # neural cell
    ("CL:0000586", "germ"),            # germ cell
    ("CL:0000034", "stem"),            # stem cell
]


def parse_obo(path: str | Path) -> tuple[dict[str, str], dict[str, set[str]], set[str]]:
    """Minimal OBO parser: ``(names, is_a parents, obsolete ids)`` for ``[Term]`` stanzas."""
    names: dict[str, str] = {}
    parents: dict[str, set[str]] = {}
    obsolete: set[str] = set()
    cur: dict | None = None

    def flush() -> None:
        if cur and cur.get("id"):
            tid = cur["id"]
            if cur.get("obsolete"):
                obsolete.add(tid)
                return
            names[tid] = cur.get("name", tid)
            parents[tid] = set(cur.get("is_a", ()))

    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("["):
                flush()
                cur = {} if line == "[Term]" else None
                continue
            if cur is None or not line or ":" not in line:
                continue
            key, _, val = line.partition(":")
            val = val.split(" ! ")[0].strip()
            if key == "id":
                cur["id"] = val
            elif key == "name":
                cur["name"] = val
            elif key == "is_a":
                cur.setdefault("is_a", []).append(val.split()[0])
            elif key == "is_obsolete" and val.lower() == "true":
                cur["obsolete"] = True
    flush()
    return names, parents, obsolete


@dataclass
class CellOntology:
    names: dict[str, str]
    parents: dict[str, set[str]]
    obsolete: set[str] = field(default_factory=set)
    root: str = ROOT
    source: str = ""

    def __post_init__(self) -> None:
        self._depth: dict[str, int] | None = None
        self._anc: dict[str, frozenset[str]] = {}

    # ------------------------------------------------------------------ structure

    def __contains__(self, term: str) -> bool:
        return term in self.names

    def name(self, term: str) -> str:
        return self.names.get(term, term)

    def ancestors(self, term: str, include_self: bool = True) -> frozenset[str]:
        if term not in self._anc:
            seen: set[str] = set()
            todo = [term]
            while todo:
                t = todo.pop()
                for p in self.parents.get(t, ()):
                    if p not in seen:
                        seen.add(p)
                        todo.append(p)
            self._anc[term] = frozenset(seen)
        a = self._anc[term]
        return a | {term} if include_self else a

    def depths(self) -> dict[str, int]:
        """Shortest is_a distance from the root for every reachable term."""
        if self._depth is None:
            children: dict[str, set[str]] = {}
            for c, ps in self.parents.items():
                for p in ps:
                    children.setdefault(p, set()).add(c)
            depth = {self.root: 0}
            q = deque([self.root])
            while q:
                t = q.popleft()
                for c in children.get(t, ()):
                    if c not in depth:
                        depth[c] = depth[t] + 1
                        q.append(c)
            self._depth = depth
        return self._depth

    def depth(self, term: str) -> int | None:
        return self.depths().get(term)

    # ------------------------------------------------------------------ levels

    def level(self, term: str, k: int, anchors: Mapping[int, Sequence] | None = None) -> str | None:
        """Level-k ancestor id of ``term`` (None for unknown terms).

        With curated anchors for level ``k`` (``anchors[k]``: ids or ``(id,
        label)`` pairs; Level 1 defaults to :data:`LEVEL1_ANCHORS`) the first
        anchor that is an ancestor-or-self of the term is returned (None if
        none). Otherwise the ancestor-or-self at depth ``k``; a term shallower
        than ``k`` maps to itself; several candidates -> the alphabetically
        first name (deterministic).
        """
        if term not in self.names:
            return None
        anc = self.ancestors(term)
        level_anchors = self._anchors_for(k, anchors)
        if level_anchors is not None:
            for a, _label in level_anchors:
                if a in anc:
                    return a
            return None
        d = self.depths()
        if d.get(term) is None:
            return None
        if d[term] <= k:
            return term
        cands = [a for a in anc if d.get(a) == k]
        if not cands:
            return term
        return sorted(cands, key=lambda a: (self.name(a), a))[0]

    def _anchors_for(self, k: int, anchors: Mapping[int, Sequence] | None) -> list[tuple[str, str]] | None:
        src = anchors if anchors is not None else {1: LEVEL1_ANCHORS}
        lst = src.get(k)
        if lst is None:
            return None
        out = []
        for a in lst:
            if isinstance(a, (tuple, list)):
                out.append((str(a[0]), str(a[1])))
            else:
                out.append((str(a), self.name(str(a))))
        return out

    def level_label(self, term: str, k: int, anchors: Mapping[int, Sequence] | None = None) -> str | None:
        """Human-readable Level-k label (the anchor's curated label, or the ancestor's name)."""
        a = self.level(term, k, anchors)
        if a is None:
            return None
        level_anchors = self._anchors_for(k, anchors)
        if level_anchors is not None:
            return dict(level_anchors).get(a, self.name(a))
        return self.name(a)

    def map_terms(self, terms: Iterable[str], levels: Sequence[int] = (1, 3),
                  anchors: Mapping[int, Sequence] | None = None) -> pd.DataFrame:
        """One row per distinct term id: ``term_id, name, depth, level<k>_id, level<k>``."""
        rows = []
        for t in pd.unique(pd.Series(list(terms), dtype=object).dropna().astype(str)):
            row = {"term_id": t, "name": self.names.get(t), "depth": self.depth(t)}
            for k in levels:
                row[f"level{k}_id"] = self.level(t, k, anchors)
                row[f"level{k}"] = self.level_label(t, k, anchors)
            rows.append(row)
        return pd.DataFrame(rows)


def _from_pronto(path: str) -> CellOntology:
    import pronto  # type: ignore

    ont = pronto.Ontology(path)
    names, parents, obsolete = {}, {}, set()
    for term in ont.terms():
        if term.obsolete:
            obsolete.add(term.id)
            continue
        names[term.id] = term.name or term.id
        parents[term.id] = {p.id for p in term.superclasses(distance=1, with_self=False)}
    return CellOntology(names, parents, obsolete, source=f"pronto:{path}")


def _from_obonet(path: str) -> CellOntology:
    import obonet  # type: ignore

    g = obonet.read_obo(path)
    names, parents, obsolete = {}, {}, set()
    for tid, data in g.nodes(data=True):
        if str(data.get("is_obsolete", "")).lower() == "true":
            obsolete.add(tid)
            continue
        names[tid] = data.get("name", tid)
        parents[tid] = {p.split()[0] for p in data.get("is_a", [])}
    return CellOntology(names, parents, obsolete, source=f"obonet:{path}")


def load_cell_ontology(path: str | Path | None = None, engine: str = "auto") -> CellOntology:
    """Load the Cell Ontology from an OBO file.

    ``path`` defaults to ``$VBT_CL_OBO``; download ``cl-basic.obo`` from the OBO
    Foundry (http://purl.obolibrary.org/obo/cl/cl-basic.obo) for real data.
    ``engine``: ``"pronto"``, ``"obonet"``, ``"builtin"`` or ``"auto"`` (pronto,
    then obonet, then the built-in parser). Only ``is_a`` edges define the
    hierarchy.
    """
    path = path or os.environ.get("VBT_CL_OBO")
    if not path:
        raise FileNotFoundError("no Cell Ontology OBO given: pass path= or set VBT_CL_OBO "
                                "(http://purl.obolibrary.org/obo/cl/cl-basic.obo)")
    path = str(path)
    if engine not in ("auto", "pronto", "obonet", "builtin"):
        raise ValueError("engine must be 'auto', 'pronto', 'obonet' or 'builtin'")
    if engine in ("auto", "pronto"):
        try:
            return _from_pronto(path)
        except ImportError:
            if engine == "pronto":
                raise ImportError("pronto is not installed: pip install 'vbt-harness[singlecell]'") from None
    if engine in ("auto", "obonet"):
        try:
            return _from_obonet(path)
        except ImportError:
            if engine == "obonet":
                raise ImportError("obonet is not installed: pip install obonet") from None
    names, parents, obsolete = parse_obo(path)
    return CellOntology(names, parents, obsolete, source=f"builtin:{path}")


def add_level_labels(obs: pd.DataFrame, ontology: CellOntology, term_col: str = "cell_type_ontology_term_id",
                     levels: Sequence[int] = (1, 3), prefix: str = "cell_type_level",
                     fallback_col: str | None = "cell_type",
                     anchors: Mapping[int, Sequence] | None = None) -> pd.DataFrame:
    """Add ``<prefix><k>`` columns (e.g. ``cell_type_level1``, ``cell_type_level3``) to ``obs``.

    Terms the ontology does not know keep ``fallback_col``'s value (or
    ``"unknown"``). Returns ``obs`` (modified in place) for chaining.
    """
    table = ontology.map_terms(obs[term_col].astype(str), levels, anchors).set_index("term_id")
    for k in levels:
        lab = obs[term_col].astype(str).map(table[f"level{k}"])
        if fallback_col and fallback_col in obs.columns:
            lab = lab.where(lab.notna(), obs[fallback_col].astype(str))
        obs[f"{prefix}{k}"] = lab.fillna("unknown").astype(str).to_numpy()
    return obs
