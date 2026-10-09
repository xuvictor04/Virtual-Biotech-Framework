"""``essentiality``: grouped screen statistics over a screens item table (ASN-5: this was core code of the data
child, ``serve_extension("essentiality")``).

Options (the overlay's ``split.essentiality``):

``mode``           ``by_tissue`` (one gene's tissues, with its screens), ``by_disease`` (one gene per disease),
                   ``by_gene`` (genes essential in one disease) or ``selective`` (genes whose mean effect in a
                   target group is lower than in a comparison group)
``columns``        column paths: ``gene``, ``effect``, ``disease`` (always); ``tissue_id`` (by_tissue);
                   optional ``tissue_name``, ``cell_line``, ``essential_flag``
``screen_fields``  by_tissue: the screen record (``{field: column path}``) and ``screen_order``, the field the
                   screens are sorted by
``args``           the tool's argument names for ``target``, ``comparison`` (selective, required), ``gene``,
                   ``min_cell_lines``, ``min_effect_threshold``, ``min_effect_difference`` (default: the same name)
``defaults``       values of those arguments when the call gives none

Diseases are exact vocabulary values (the gateway checked them); groups below ``min_cell_lines`` known effects are
dropped and counted; unknown effects never count as non-essential; the fraction uses the measure's declared cutoff
(its inclusivity), or ``<= threshold`` when the call gives one, and is null (never 0) without a known effect.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping, Sequence

from ..base import DerivedBase, DerivedOptionsError
from ..registry import register

__all__ = ["Essentiality", "MODES", "effect_stats", "cutoff_text"]

MODES = ("by_tissue", "by_disease", "by_gene", "selective")
_COLUMNS = ("gene", "effect", "disease")
_MODE_COLUMNS = {"by_tissue": ("tissue_id",)}
_MODE_ARGS = {"selective": ("target", "comparison")}
_NUMERIC = ("min_cell_lines", "min_effect_threshold", "min_effect_difference")


def effect_stats(effects: Sequence[Any], spec: Any, threshold: Any = None) -> dict[str, Any]:
    """``n``, ``n_unknown``, ``mean_gene_effect``, ``essential_count`` and ``essential_fraction`` of screen
    effects: unknown effects are excluded and counted; ``None`` (never 0) without a known effect."""
    from ..statistics.gene_effect import meets_cutoff

    known = [float(e) for e in effects if isinstance(e, (int, float)) and not isinstance(e, bool) and e == e]
    out: dict[str, Any] = {"n": len(known), "n_unknown": len(effects) - len(known),
                           "mean_gene_effect": (sum(known) / len(known)) if known else None}
    if threshold is not None:
        flags = [e <= float(threshold) for e in known]
    elif getattr(spec, "cutoff", None) is not None:
        flags = [bool(meets_cutoff(e, spec)) for e in known]
    else:
        flags = []
    out["essential_count"] = sum(flags) if flags or known else None
    out["essential_fraction"] = (sum(flags) / len(flags)) if flags else None
    return out


def cutoff_text(spec: Any) -> str | None:
    cutoff = getattr(spec, "cutoff", None)
    if cutoff is None:
        return None
    return f"{cutoff.op} {cutoff.value} ({'inclusive' if cutoff.op in ('le', 'ge') else 'exclusive'})"


@register
class Essentiality(DerivedBase):
    name: ClassVar[str] = "essentiality"
    version: ClassVar[str] = "1.0"
    records: ClassVar[tuple[str, ...]] = ("_essentiality",)
    capabilities: ClassVar[frozenset[str]] = frozenset({"modes"})

    def validate_options(self, options: Mapping[str, Any]) -> list[str]:
        problems: list[str] = []
        mode = options.get("mode")
        if mode not in MODES:
            problems.append(f"mode must be one of {', '.join(MODES)} (got {mode!r})")
        cols = options.get("columns")
        if not isinstance(cols, Mapping):
            return [*problems, "columns must map gene, effect and disease to column paths"]
        missing = [k for k in (*_COLUMNS, *_MODE_COLUMNS.get(str(mode), ())) if not cols.get(k)]
        if missing:
            problems.append(f"columns needs {', '.join(missing)}")
        args = options.get("args") or {}
        if not isinstance(args, Mapping):
            problems.append("args must map names to the tool's argument names")
            args = {}
        need = [k for k in _MODE_ARGS.get(str(mode), ()) if not args.get(k)]
        if need:
            problems.append(f"mode {mode} needs args.{' and args.'.join(need)} (the tool's argument names)")
        defaults = options.get("defaults") or {}
        if not isinstance(defaults, Mapping):
            problems.append("defaults must be a mapping")
            defaults = {}
        for k in _NUMERIC:
            v = defaults.get(k)
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))):
                problems.append(f"defaults.{k} must be a number")
        if mode == "by_tissue" and options.get("screen_order") and \
                options.get("screen_order") not in dict(options.get("screen_fields") or {}):
            problems.append("screen_order must name a field of screen_fields")
        return problems

    def columns(self, options: Mapping[str, Any]) -> list[str]:
        cols = dict(options.get("columns") or {})
        fields = dict(options.get("screen_fields") or {}) if options.get("mode") == "by_tissue" else {}
        return list(dict.fromkeys([*cols.values(), *fields.values()]))

    def serve(self, view: Any, request: Mapping[str, Any], options: Mapping[str, Any]) -> dict[str, Any]:
        from ...errors import ErrorKind, GatewayError
        from ...predicate import In, from_json

        problems = self.validate_options(options)
        if problems:
            raise DerivedOptionsError("split.essentiality: " + "; ".join(problems))
        cols = dict(options["columns"])
        mode = options["mode"]

        def p(name: str) -> Any:
            return self.param(request, options, name)

        spec = view.column(cols["effect"].split(".")[-1]) or view.column(cols["effect"])
        pred = from_json(request["predicate"]) if request.get("predicate") else None
        want = list(dict.fromkeys(cols.values()))
        if mode == "selective":
            target, comparison = p("target"), p("comparison")
            if target == comparison:
                raise GatewayError(ErrorKind.invalid_argument, "the target and comparison diseases are the same "
                                                               "group")
            pred = In(cols["disease"], (target, comparison))
        rows, _total, _extra = view.rows(pred, columns=want, order=[], limit=None,
                                         budget=request.get("budget_bytes"))
        given = p("min_cell_lines")
        min_n = int(given) if given is not None else 1          # a group needs one known effect
        if mode == "by_tissue":
            return self._by_tissue(rows, cols, options, spec, p)
        below = 0
        out: list[dict[str, Any]] = []
        if mode == "by_disease":
            groups: dict[Any, list[Any]] = {}
            for r in rows:
                groups.setdefault(r.get(cols["disease"]), []).append(r.get(cols["effect"]))
            for d, effects in groups.items():
                st = effect_stats(effects, spec)
                if st["n"] < min_n:
                    below += 1
                    continue
                out.append({"disease": d, "num_cell_lines": st["n"], "num_unknown": st["n_unknown"],
                            "mean_gene_effect": st["mean_gene_effect"],
                            "essential_fraction": st["essential_fraction"]})
            out.sort(key=lambda r: (r["mean_gene_effect"] is None, r["mean_gene_effect"] or 0.0, str(r["disease"])))
            for i, r in enumerate(out, start=1):
                r["rank"] = i
        elif mode == "by_gene":
            threshold = p("min_effect_threshold")
            by_gene: dict[Any, list[dict[str, Any]]] = {}
            for r in rows:
                by_gene.setdefault(r.get(cols["gene"]), []).append(r)
            for gene, g in by_gene.items():
                st = effect_stats([r.get(cols["effect"]) for r in g], spec, threshold)
                if st["n"] < min_n:
                    below += 1
                    continue
                if not st["essential_count"]:
                    continue
                top = sorted((r for r in g if isinstance(r.get(cols["effect"]), (int, float))),
                             key=lambda r: (r[cols["effect"]], str(r.get(cols.get("cell_line", "")))))[:5]
                out.append({"gene_id": gene, "mean_gene_effect": st["mean_gene_effect"], "num_cell_lines": st["n"],
                            "num_unknown": st["n_unknown"], "essential_fraction": st["essential_fraction"],
                            "top_cell_lines": [{"cellLineName": r.get(cols.get("cell_line", "")),
                                                "disease": r.get(cols["disease"]), "geneEffect": r.get(cols["effect"])}
                                               for r in top]})
            out.sort(key=lambda r: (r["mean_gene_effect"], str(r["gene_id"])))
        else:  # selective
            given = p("min_effect_difference")
            min_diff = float(given) if given is not None else 0.0
            sides: dict[Any, dict[str, list[Any]]] = {}
            for r in rows:
                side = "target" if r.get(cols["disease"]) == target else "comparison"
                sides.setdefault(r.get(cols["gene"]), {"target": [], "comparison": []})[side].append(
                    r.get(cols["effect"]))
            for gene, both in sides.items():
                t, c = effect_stats(both["target"], spec), effect_stats(both["comparison"], spec)
                if t["n"] < min_n or c["n"] < min_n:
                    below += 1
                    continue
                diff = t["mean_gene_effect"] - c["mean_gene_effect"]
                if diff >= -min_diff:
                    continue
                out.append({"gene_id": gene, "target_effect": t["mean_gene_effect"],
                            "comparison_effect": c["mean_gene_effect"], "effect_difference": diff,
                            "target_cell_lines": t["n"], "comparison_cell_lines": c["n"]})
            out.sort(key=lambda r: (r["effect_difference"], str(r["gene_id"])))
        total = len(out)
        limit = request.get("limit")
        shown = out[: int(limit)] if limit is not None else out
        meta = {"groups_below_min_cell_lines": below, "min_cell_lines": min_n, "cutoff": cutoff_text(spec)}
        return self.response(shown, total=total, truncated=len(shown) < total,
                             key_columns=["disease"] if mode == "by_disease" else ["gene_id"],
                             sections={"_essentiality": meta})

    def _by_tissue(self, rows: list[dict[str, Any]], cols: Mapping[str, str], options: Mapping[str, Any],
                   spec: Any, p: Any) -> dict[str, Any]:
        sections = {"_essentiality": {"cutoff": cutoff_text(spec)}}
        if not rows:
            # no screen of this gene: nothing was found (the gateway reports empty, not screened), never a record
            # that says found
            return self.response([], total=0, sections=sections)
        fields = dict(options.get("screen_fields") or {})
        order = options.get("screen_order")
        genes = {r.get(cols["gene"]) for r in rows}
        groups: dict[Any, list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(r.get(cols["tissue_id"]), []).append(r)
        tissues = []
        for tid in sorted(groups, key=lambda t: str(t)):
            g = groups[tid]
            st = effect_stats([r.get(cols["effect"]) for r in g], spec)
            screens = [{k: r.get(v) for k, v in fields.items()} for r in g]
            if order:
                screens.sort(key=lambda s: str(s.get(order)))
            tissues.append({"tissue": {"id": tid, "name": g[0].get(cols.get("tissue_name", ""))},
                            "num_cell_lines": len(g), "num_with_effect": st["n"],
                            "mean_gene_effect": st["mean_gene_effect"], "essential_fraction": st["essential_fraction"],
                            "cell_lines": screens})
        flags = [r.get(cols["essential_flag"]) for r in rows if cols.get("essential_flag")]
        known_flags = [f for f in flags if f is not None]
        gene = next(iter(genes)) if len(genes) == 1 else p("gene")
        record = {"gene_id": gene, "found": True,
                  "is_essential": known_flags[0] if known_flags else None, "num_tissues": len(tissues),
                  "num_cell_lines": sum(t["num_cell_lines"] for t in tissues), "essentiality_by_tissue": tissues}
        return self.response([record], total=1, sections=sections)

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.derived import DerivedCase, DerivedCases

        cols = {"gene": "gene", "effect": "effect", "disease": "disease", "cell_line": "line",
                "tissue_id": "tissue", "tissue_name": "tissue_name"}
        rows = [
            {"gene": "G1", "effect": -1.2, "disease": "A", "line": "L1", "tissue": "T1", "tissue_name": "lung"},
            {"gene": "G1", "effect": -0.8, "disease": "A", "line": "L2", "tissue": "T1", "tissue_name": "lung"},
            {"gene": "G1", "effect": 0.1, "disease": "B", "line": "L3", "tissue": "T2", "tissue_name": "skin"},
            {"gene": "G2", "effect": None, "disease": "A", "line": "L1", "tissue": "T1", "tissue_name": "lung"},
            {"gene": "G2", "effect": -0.1, "disease": "B", "line": "L3", "tissue": "T2", "tissue_name": "skin"},
        ]
        by_disease = {"mode": "by_disease", "columns": cols, "defaults": {"min_cell_lines": 1}}
        selective = {"mode": "selective", "columns": cols, "args": {"target": "t", "comparison": "c"},
                     "defaults": {"min_cell_lines": 1, "min_effect_difference": 0.3}}
        by_gene = {"mode": "by_gene", "columns": cols, "args": {"min_effect_threshold": "threshold"},
                   "defaults": {"min_cell_lines": 1}}
        return DerivedCases(
            valid_options=(by_disease, selective, by_gene,
                           {"mode": "by_tissue", "columns": cols, "screen_fields": {"line": "line"},
                            "screen_order": "line"}),
            invalid_options=({}, {"mode": "by_month", "columns": cols}, {"mode": "by_gene", "columns": {"gene": "g"}},
                             {"mode": "selective", "columns": cols},
                             {"mode": "by_tissue", "columns": {k: v for k, v in cols.items() if k != "tissue_id"}},
                             {"mode": "by_tissue", "columns": cols, "screen_fields": {"line": "line"},
                              "screen_order": "depmapId"},
                             {"mode": "by_gene", "columns": cols, "defaults": {"min_cell_lines": "one"}}),
            cases=(
                DerivedCase("by_disease", rows, by_disease, {}, total=2, key_columns=("disease",),
                            first={"disease": "A", "num_cell_lines": 2, "num_unknown": 1}),
                DerivedCase("selective_target_lower", rows, selective, {"t": "A", "c": "B"}, total=1,
                            first={"gene_id": "G1", "target_cell_lines": 2, "comparison_cell_lines": 1}),
                DerivedCase("by_gene_threshold", rows, by_gene, {"threshold": -0.5}, total=1,
                            first={"gene_id": "G1", "num_cell_lines": 3}),
                DerivedCase("by_gene_limit", rows, by_gene, {"threshold": 1.0}, total=2, limit=1, returned=1),
                DerivedCase("by_disease_no_rows", [], by_disease, {}, total=0),
            ),
        )
