"""``tissue_specificity``: genes expressed more in one tissue than in the median of the others (ASN-5: this was
core code of the data child, ``serve_extension("tissue_specificity")``).

Options (the overlay's ``split.tissue_specificity``):

``columns``   column paths: ``gene``, ``tissue_id``, ``tissue_label``, ``value``, ``unit``; optional ``zscore``
``args``      the tool's argument names for ``tissue`` and ``threshold`` (default: the same name)
``defaults``  ``threshold`` (the fold change a gene must reach) when the call gives none

Unit guard: each gene is compared within the unit of its target-tissue value; a blank unit is not a unit (the gene
is excluded and counted) and values in other units are never pooled. When every other tissue is 0 the fold change
is undefined: the gene is reported as ``expressed_only_in_target`` with ``fold_change: null``, ranked first. The
tissue is an exact label or id (casefolded), never a substring.
"""

from __future__ import annotations

from statistics import median
from typing import Any, ClassVar, Mapping

from ..base import DerivedBase, DerivedOptionsError
from ..registry import register

__all__ = ["TissueSpecificity"]

_COLUMNS = ("gene", "tissue_id", "tissue_label", "value", "unit")


@register
class TissueSpecificity(DerivedBase):
    name: ClassVar[str] = "tissue_specificity"
    version: ClassVar[str] = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset()

    def validate_options(self, options: Mapping[str, Any]) -> list[str]:
        cols = options.get("columns")
        if not isinstance(cols, Mapping):
            return [f"columns must map {', '.join(_COLUMNS)} to column paths"]
        problems = []
        missing = [k for k in _COLUMNS if not cols.get(k)]
        if missing:
            problems.append(f"columns needs {', '.join(missing)}")
        if not isinstance(options.get("args") or {}, Mapping):
            problems.append("args must map names to the tool's argument names")
        defaults = options.get("defaults") or {}
        if not isinstance(defaults, Mapping):
            problems.append("defaults must be a mapping")
        elif defaults.get("threshold") is not None and (isinstance(defaults["threshold"], bool) or
                                                         not isinstance(defaults["threshold"], (int, float))):
            problems.append("defaults.threshold must be a number")
        return problems

    def columns(self, options: Mapping[str, Any]) -> list[str]:
        cols = dict(options.get("columns") or {})
        return [cols[k] for k in (*_COLUMNS, "zscore") if cols.get(k)]

    def serve(self, view: Any, request: Mapping[str, Any], options: Mapping[str, Any]) -> dict[str, Any]:
        from ...errors import ErrorKind, GatewayError

        problems = self.validate_options(options)
        if problems:
            raise DerivedOptionsError("split.tissue_specificity: " + "; ".join(problems))
        cols = dict(options["columns"])
        tissue = self.param(request, options, "tissue")
        if tissue in (None, ""):
            raise GatewayError(ErrorKind.invalid_argument, "tissue is required")
        given = self.param(request, options, "threshold")
        if given is None:
            raise DerivedOptionsError("split.tissue_specificity: the call gives no threshold and the overlay "
                                      "declares no defaults.threshold")
        threshold = float(given)
        rows, _t, _e = view.rows(None, columns=self.columns(options), order=[], limit=None,
                                 budget=request.get("budget_bytes"))
        key = str(tissue).strip().casefold()

        def is_target(r: Mapping[str, Any]) -> bool:
            return str(r.get(cols["tissue_id"]) or "").casefold() == key or \
                str(r.get(cols["tissue_label"]) or "").strip().casefold() == key

        by_gene: dict[str, list[Mapping[str, Any]]] = {}
        for r in rows:
            by_gene.setdefault(str(r.get(cols["gene"])), []).append(r)
        if not any(is_target(r) for r in rows):
            labels = sorted({str(r.get(cols["tissue_label"])) for r in rows if r.get(cols["tissue_label"])})
            raise GatewayError(ErrorKind.invalid_argument, f"tissue {tissue!r} is not a tissue of this table (exact "
                               "label or EFO code)", payload={"argument": "tissue", "valid_values": labels[:50]})
        out: list[dict[str, Any]] = []
        blank_unit = undefined = 0
        for gene, entries in by_gene.items():
            targets = [e for e in entries if is_target(e) and isinstance(e.get(cols["value"]), (int, float))]
            if not targets:
                continue
            t = targets[0]
            unit = t.get(cols["unit"])
            if unit is None or not str(unit).strip():
                blank_unit += 1
                continue
            others = [float(e[cols["value"]]) for e in entries if not is_target(e) and e.get(cols["unit"]) == unit
                      and isinstance(e.get(cols["value"]), (int, float)) and e[cols["value"]] == e[cols["value"]]]
            if not others:
                continue
            med = float(median(others))
            value = float(t[cols["value"]])
            if med == 0:
                if value <= 0:
                    continue
                undefined += 1
                fold, only = None, True
            else:
                fold, only = value / med, False
                if fold < threshold:
                    continue
            out.append({"gene_id": gene, "tissue_expression": value, "median_other_tissues": med,
                        "fold_change": fold, "expressed_only_in_target": only, "n_other_tissues": len(others),
                        "zscore_in_tissue": t.get(cols["zscore"]) if cols.get("zscore") else None, "unit": unit})
        out.sort(key=lambda r: (not r["expressed_only_in_target"], -(r["fold_change"] or 0.0),
                                -r["tissue_expression"], r["gene_id"]))
        total = len(out)
        limit = request.get("limit")
        shown = out[: int(limit)] if limit is not None else out
        meta = {"blank_unit_excluded": blank_unit, "expressed_only_in_target": undefined, "threshold": threshold}
        return self.response(shown, total=total, truncated=len(shown) < total, key_columns=["gene_id"],
                             sections={"_specificity": meta})

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.derived import DerivedCase, DerivedCases

        cols = {"gene": "gene", "tissue_id": "tid", "tissue_label": "label", "value": "value", "unit": "unit"}
        options = {"columns": cols, "args": {"threshold": "fold"}, "defaults": {"threshold": 2.0}}
        rows = [
            {"gene": "G1", "tid": "T1", "label": "liver", "value": 10.0, "unit": "TPM"},
            {"gene": "G1", "tid": "T2", "label": "lung", "value": 2.0, "unit": "TPM"},
            {"gene": "G1", "tid": "T3", "label": "skin", "value": 2.0, "unit": "TPM"},
            {"gene": "G2", "tid": "T1", "label": "liver", "value": 3.0, "unit": "TPM"},
            {"gene": "G2", "tid": "T2", "label": "lung", "value": 0.0, "unit": "TPM"},
            {"gene": "G3", "tid": "T1", "label": "liver", "value": 5.0, "unit": ""},
            {"gene": "G3", "tid": "T2", "label": "lung", "value": 1.0, "unit": ""},
            {"gene": "G4", "tid": "T1", "label": "liver", "value": 4.0, "unit": "TPM"},
            {"gene": "G4", "tid": "T2", "label": "lung", "value": 3.0, "unit": "TPM"},
        ]
        return DerivedCases(
            valid_options=(options, {"columns": {**cols, "zscore": "z"}}),
            invalid_options=({}, {"columns": {"gene": "gene"}}, {"columns": cols, "defaults": {"threshold": "x"}}),
            cases=(
                DerivedCase("only_in_target_first", rows, options, {"tissue": "liver"}, total=2,
                            first={"gene_id": "G2", "fold_change": None, "expressed_only_in_target": True}),
                DerivedCase("threshold_argument", rows, options, {"tissue": "T1", "fold": 1.2}, total=3),
                DerivedCase("limit", rows, options, {"tissue": "liver"}, total=2, limit=1, returned=1),
            ),
            refusals=((rows, options, {"tissue": "liv"}), (rows, options, {})),
        )
