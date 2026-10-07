"""Paired contrasts over composite keys: Tahoe drug comparisons and cell-line selectivity (§6.6, §8.5; phase 3, F17).

Tahoe DE rows are keyed by ``(drug, concentration, concentration_unit, Cell_ID_DepMap, plate, gene_name)``
and only contrasts with padj < 0.10 are stored (``coverage.absence_means: censored``). Upstream merged two
drugs' rows on ``gene_name`` alone (a cartesian product over doses, lines and plates) and computed
selectivity ratios in which a gene missing from a line (not significant there, or the line not profiled)
silently dropped out. Here:

* the **profiled conditions** are the table's ``conditions`` tuples (``drug, concentration, unit, cell line,
  plate``) present in the data; a requested drug x cell line (x concentration) that was never profiled is
  ``not_found`` with ``subkind: combination_not_profiled`` listing the profiled values, never an empty
  comparison;
* :func:`compare_drugs` pairs rows of drug A and drug B only **within one context** (cell line,
  concentration and unit; plates are reported per pair as ``plate_a``/``plate_b``), so one stored row
  takes part in at most one pair per plate of the other drug: no cartesian product over doses or lines.
  ``signature_correlation`` is the Pearson correlation of the paired log2 fold changes within a context,
  ``None`` when fewer than three genes pair or a side has no variance (never 0 for "not computable");
  ``unique_to_a``/``unique_to_b`` are significant on one side and absent (censored: not significant or
  NA) on the other in a context where both drugs were profiled;
* :func:`selectivity` compares a drug's effect in one line with **every line tested** with that drug at the
  same concentration (or the lines given): per gene the target effect, how many comparison lines were
  tested, in how many the gene was significant, and the mean |log2FC| over those; a gene significant in
  the target line and in no comparison line is ``exclusive`` (its ratio is undefined, not infinite and
  not dropped), otherwise ``selectivity_ratio = |FC_target| / mean |FC_significant elsewhere|``. NaN and
  null effects are excluded and counted.

Hidden verb ``_pairs``: ``{mode: compare | selectivity, table?, drug_a, drug_b | drug, cell_line?,
cell_line_of_interest, comparison_cell_lines?, concentration?, min_abs_log2fc?, max_padj?,
selectivity_threshold?, limit?}``. Derived bindings reach it through ``_serve`` with ``split: {pairs:
{mode, args: {...}}}``: the drugs and lines are the predicate's (already resolved) values, thresholds the
call's parameters.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ...errors import ErrorKind, GatewayError, json_value
from ...ipc import ServeResponse
from ...predicate import And, Cmp, CmpAbs, Eq, In, Predicate, from_json
from ...result import inject_header
from ...rowkey import canonical
from .. import ServiceContext
from .hierarchy import serve_extension
from .public import LongView, _invalid, compile_where, guarded, header, long_view, table_access

__all__ = ["compare_drugs", "selectivity", "profiled_conditions", "pearson", "pairs_verb", "serve_pairs", "VERBS"]

DEFAULT_TABLE = "tahoe_100m.de_permissive"
DRUG, LINE, CONC, UNIT, PLATE, GENE = "drug", "Cell_ID_DepMap", "concentration", "concentration_unit", "plate", \
    "gene_name"
LFC, PADJ = "log2FoldChange", "padj"


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson correlation, ``None`` for fewer than three pairs or a side without variance."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mx = math.fsum(xs) / len(xs)
    my = math.fsum(ys) / len(ys)
    sxy = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = math.fsum((x - mx) ** 2 for x in xs)
    syy = math.fsum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return max(-1.0, min(1.0, sxy / math.sqrt(sxx * syy)))


def _columns(view: LongView) -> tuple[list[str], str]:
    cond = view.table.spec.conditions
    if cond is None:
        raise _invalid("table", view.ref, f"{view.ref} declares no conditions (profiled tuples)")
    gene = next(k for k in view.key if k not in cond.columns)
    return list(cond.columns), gene


def profiled_conditions(view: LongView, drugs: Sequence[Any]) -> list[dict[str, Any]]:
    """The distinct condition tuples of ``drugs`` present in the table (``conditions.from: self``)."""
    cols, _gene = _columns(view)
    rows, _t, _e = view.rows(In(DRUG, tuple(drugs)), columns=cols, order=[], limit=None, distinct=cols)
    return sorted(({c: r.get(c) for c in cols} for r in rows), key=lambda r: canonical([r.get(c) for c in cols]))


def _not_profiled(what: str, profiled: list[dict[str, Any]], fields: Sequence[str]) -> GatewayError:
    values = {f: sorted({json_value(p.get(f)) for p in profiled if p.get(f) is not None}, key=str) for f in fields}
    return GatewayError(ErrorKind.not_found, f"{what} was not profiled (no stored contrast); profiled: "
                        + "; ".join(f"{f}: {', '.join(map(str, v[:20]))}" for f, v in values.items()),
                        subkind="combination_not_profiled", payload={"profiled": values})


def _filters(min_abs: Any, max_padj: Any) -> list[Predicate]:
    out: list[Predicate] = []
    if max_padj is not None:
        out.append(Cmp(PADJ, "<=", float(max_padj)))
    if min_abs not in (None, 0, 0.0):
        out.append(CmpAbs(LFC, "gt_abs", float(min_abs)))  # upstream's strict |log2FC| > min
    return out


def _context(r: Mapping[str, Any]) -> tuple[Any, ...]:
    return (r.get(LINE), r.get(CONC), r.get(UNIT))


def _ctx_dict(c: tuple[Any, ...]) -> dict[str, Any]:
    return {LINE: c[0], CONC: json_value(c[1], "float"), UNIT: c[2]}


def compare_drugs(view: LongView, drug_a: Any, drug_b: Any, *, cell_line: Any = None, concentration: Any = None,
                  min_abs: Any = 0.5, max_padj: Any = 0.10, budget: int | None = None) -> dict[str, Any]:
    """Paired comparison of two drugs within shared contexts (see the module docstring)."""
    profiled = profiled_conditions(view, [drug_a, drug_b])
    a_ctx = {(p[LINE], p[CONC], p[UNIT]) for p in profiled if p[DRUG] == drug_a}
    b_ctx = {(p[LINE], p[CONC], p[UNIT]) for p in profiled if p[DRUG] == drug_b}
    for drug, ctxs in ((drug_a, a_ctx), (drug_b, b_ctx)):
        if not ctxs:
            raise _not_profiled(f"drug {drug!r}", profiled, [DRUG])
        if cell_line is not None and not any(c[0] == cell_line for c in ctxs):
            raise _not_profiled(f"{drug!r} in {cell_line}", [p for p in profiled if p[DRUG] == drug], [LINE, CONC])
    shared = {c for c in a_ctx & b_ctx if (cell_line is None or c[0] == cell_line) and
              (concentration is None or _same(c[1], concentration))}
    if not shared:
        raise _not_profiled(f"{drug_a!r} and {drug_b!r} in a shared cell line and concentration"
                            + (f" ({cell_line})" if cell_line else ""), profiled, [DRUG, LINE, CONC])
    parts: list[Predicate] = [In(DRUG, (drug_a, drug_b)), *_filters(min_abs, max_padj)]
    if cell_line is not None:
        parts.append(Eq(LINE, cell_line))
    rows, _t, eu = view.rows(And(tuple(parts)), order=[], limit=None, budget=budget)
    side: dict[tuple[Any, ...], dict[str, dict[str, list[dict[str, Any]]]]] = {}
    excluded = 0
    for r in rows:
        c = _context(r)
        if c not in shared:
            continue
        if not _finite(r.get(LFC)):
            excluded += 1
            continue
        side.setdefault(c, {"a": {}, "b": {}})["a" if r.get(DRUG) == drug_a else "b"].setdefault(
            str(r.get(GENE)), []).append(r)
    same, opposite, only_a, only_b, contexts = [], [], [], [], []
    for c in sorted(shared, key=lambda x: canonical(list(x))):
        s = side.get(c, {"a": {}, "b": {}})
        xs, ys = [], []
        n_same = n_opp = 0
        for gene in sorted(set(s["a"]) | set(s["b"])):
            ra, rb = s["a"].get(gene, []), s["b"].get(gene, [])
            if ra and rb:
                for x in sorted(ra, key=lambda r: str(r.get(PLATE))):
                    for y in sorted(rb, key=lambda r: str(r.get(PLATE))):
                        pair = {**_ctx_dict(c), "gene_name": gene, "plate_a": x.get(PLATE), "plate_b": y.get(PLATE),
                                "log2FC_drug_a": x[LFC], "log2FC_drug_b": y[LFC], "padj_drug_a": x.get(PADJ),
                                "padj_drug_b": y.get(PADJ)}
                        xs.append(float(x[LFC]))
                        ys.append(float(y[LFC]))
                        if x[LFC] * y[LFC] > 0:
                            same.append(pair)
                            n_same += 1
                        elif x[LFC] * y[LFC] < 0:
                            opposite.append(pair)
                            n_opp += 1
            elif ra:
                only_a.append({**_ctx_dict(c), "gene_name": gene})
            else:
                only_b.append({**_ctx_dict(c), "gene_name": gene})
        contexts.append({**_ctx_dict(c), "num_genes_a": len(s["a"]), "num_genes_b": len(s["b"]),
                         "num_pairs": len(xs), "num_shared_same_direction": n_same, "num_opposite_effects": n_opp,
                         "signature_correlation": pearson(xs, ys)})
    record = {
        "signature_correlation": contexts[0]["signature_correlation"] if len(contexts) == 1 else None,
        "contexts": contexts,
        "shared_targets_same_direction": same, "opposite_effects": opposite,
        "unique_to_a": only_a, "unique_to_b": only_b,
        "stats": {"num_contexts": len(contexts), "num_pairs": sum(c["num_pairs"] for c in contexts),
                  "num_shared_same_direction": len(same), "num_opposite_effects": len(opposite),
                  "num_unique_to_a": len(only_a), "num_unique_to_b": len(only_b),
                  "excluded_non_finite": excluded,
                  "not_shared_contexts": {"a": len(a_ctx - b_ctx), "b": len(b_ctx - a_ctx)}},
    }
    if len(contexts) > 1:
        record["note"] = ("several cell line x concentration contexts are compared separately; "
                          "signature_correlation is per context (contexts[*].signature_correlation)")
    return record


def _same(a: Any, b: Any) -> bool:
    try:
        return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return a == b


def selectivity(view: LongView, drug: Any, line: Any, *, comparison: Sequence[Any] | None = None,
                concentration: Any = None, min_abs: Any = 0.5, max_padj: Any = 0.10, threshold: Any = 2.0,
                budget: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Genes with a line-selective effect of ``drug`` in ``line`` (see the module docstring)."""
    profiled = profiled_conditions(view, [drug])
    if not profiled:
        raise _not_profiled(f"drug {drug!r}", profiled, [DRUG])
    target_ctx = sorted({(p[CONC], p[UNIT]) for p in profiled if p[LINE] == line and
                         (concentration is None or _same(p[CONC], concentration))}, key=lambda x: canonical(list(x)))
    if not target_ctx:
        raise _not_profiled(f"{drug!r} in {line}" + (f" at {concentration}" if concentration is not None else ""),
                            profiled, [LINE, CONC])
    tested: dict[tuple[Any, ...], set[str]] = {}
    for p in profiled:
        if p[LINE] != line:
            tested.setdefault((p[CONC], p[UNIT]), set()).add(str(p[LINE]))
    if comparison:
        missing = [c for c in comparison if not any(str(c) in tested.get(tc, set()) for tc in target_ctx)]
        if missing:
            raise _not_profiled(f"{drug!r} in comparison line(s) {', '.join(map(str, missing))} at the target's "
                                "concentration", profiled, [LINE, CONC])
    rows, _t, _e = view.rows(And((Eq(DRUG, drug), *_filters(None, max_padj))), order=[], limit=None, budget=budget)
    by: dict[tuple[Any, ...], dict[str, dict[str, list[float]]]] = {}
    target: list[dict[str, Any]] = []
    excluded = 0
    for r in rows:
        if not _finite(r.get(LFC)):
            excluded += 1
            continue
        tc = (r.get(CONC), r.get(UNIT))
        if r.get(LINE) == line:
            if tc in target_ctx and abs(float(r[LFC])) >= float(min_abs or 0):   # upstream: inclusive
                target.append(r)
            continue
        by.setdefault(tc, {}).setdefault(str(r.get(GENE)), {}).setdefault(str(r.get(LINE)), []).append(
            abs(float(r[LFC])))
    out = []
    for r in target:
        tc = (r.get(CONC), r.get(UNIT))
        lines = set(tested.get(tc, set()))
        if comparison:
            lines &= {str(c) for c in comparison}
        hits = {ln: vals for ln, vals in by.get(tc, {}).get(str(r.get(GENE)), {}).items() if ln in lines}
        sig = [math.fsum(v) / len(v) for v in hits.values()]
        mean_sig = math.fsum(sig) / len(sig) if sig else None
        ratio = abs(float(r[LFC])) / mean_sig if mean_sig else None
        exclusive = not hits and bool(lines)
        if not exclusive and (ratio is None or ratio < float(threshold)):
            continue
        out.append({"gene_name": r.get(GENE), CONC: json_value(tc[0], "float"), UNIT: tc[1], "plate": r.get(PLATE),
                    "log2FC_target": r[LFC], "padj_target": r.get(PADJ), "n_comparison_cell_lines": len(lines),
                    "n_comparison_significant": len(hits),
                    "mean_abs_log2FC_comparison_significant": mean_sig, "selectivity_ratio": ratio,
                    "exclusive": exclusive})
    out.sort(key=lambda g: (not g["exclusive"], -(g["selectivity_ratio"] or 0.0), -abs(g["log2FC_target"]),
                            canonical([g["gene_name"], g[CONC], g["plate"]])))
    stats = {"comparison_lines": {f"{json_value(k[0], 'float')} {k[1]}": sorted(v) for k, v in sorted(
        tested.items(), key=lambda kv: canonical(list(kv[0]))) if any(_same(k[0], t[0]) for t in target_ctx)},
             "target_contexts": [{CONC: json_value(c[0], "float"), UNIT: c[1]} for c in target_ctx],
             "excluded_non_finite": excluded, "selectivity_threshold": threshold,
             "censoring": "a gene absent from a tested comparison line is not significant there (padj >= 0.10 or "
                          "NA), not 0 and not dropped"}
    return out, stats


# ---------------------------------------------------------------------------- the hidden verb


def pairs_verb(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    """``_pairs``: ``mode: compare`` or ``mode: selectivity`` with resolved arguments."""
    table = table_access(ctx, payload.get("table") or DEFAULT_TABLE, agent=payload.get("agent"), native=False)
    view = long_view(ctx, str(table.ref))
    notes: list[str] = []
    resolved: dict[str, str] = {}

    def one(name: str, column: str) -> Any:
        v = payload.get(name)
        if v is None:
            return None
        _p, keys = compile_where(view, {column: v}, argument=name, notes=notes, resolved=resolved)
        return _stored(ctx, view, column, (keys.get(column) or [v])[0])

    mode = payload.get("mode")
    if mode == "compare":
        a, b = one("drug_a", DRUG), one("drug_b", DRUG)
        if a is None or b is None:
            raise _invalid("drug_a", payload.get("drug_a"), "compare needs drug_a and drug_b")
        record = compare_drugs(view, a, b, cell_line=one("cell_line", LINE), concentration=payload.get("concentration"),
                               min_abs=payload.get("min_abs_log2fc", 0.5), max_padj=payload.get("max_padj", 0.10))
        rows = record["shared_targets_same_direction"] + record["opposite_effects"]
        hdr = header(view, rows=rows, total=len(rows), resolved=resolved, notes=notes,
                     key=[LINE, CONC, UNIT, "gene_name", "plate_a", "plate_b"], extra={"stats": record["stats"]})
        return inject_header(json_value(record), hdr)
    if mode == "selectivity":
        drug = one("drug", DRUG)
        line = one("cell_line_of_interest", LINE)
        comp = payload.get("comparison_cell_lines")
        comp_ids = None
        if comp:
            _p, keys = compile_where(view, {LINE: list(comp)}, argument="comparison_cell_lines", notes=notes,
                                     resolved=resolved)
            comp_ids = list(keys.get(LINE) or comp)
        rows, stats = selectivity(view, drug, line, comparison=comp_ids, concentration=payload.get("concentration"),
                                  min_abs=payload.get("min_abs_log2fc", 0.5), max_padj=payload.get("max_padj", 0.10),
                                  threshold=payload.get("selectivity_threshold", 2.0))
        limit = payload.get("limit")
        shown = rows[: int(limit)] if limit is not None else rows
        hdr = header(view, rows=shown, total=len(rows), truncated=len(shown) < len(rows), resolved=resolved,
                     notes=notes, key=["gene_name", CONC, UNIT, "plate"], order="exclusive first, then selectivity",
                     extra={"selectivity": stats})
        return inject_header({"rows": json_value(shown)}, hdr)
    raise _invalid("mode", mode, "mode is compare or selectivity", ["compare", "selectivity"])


def _stored(ctx: ServiceContext, view: LongView, column: str, canonical_value: Any) -> Any:
    """The table's own spelling of a resolved key (``'Erdafitinib '`` in the DE file for ``Erdafitinib``)."""
    from .public import resolver

    bound = view.id_type(column)
    if bound is None:
        return canonical_value
    r = resolver(ctx)
    res = r.resolve(canonical_value, [bound], bound_id_type=bound)
    if res.canonical is None:
        return canonical_value
    return r.send_value(res, "stored", view.ref)


def _values(pred: Predicate | None, column: str, kind: type) -> list[Any]:
    out: list[Any] = []

    def rec(p: Any) -> None:
        if isinstance(p, kind) and getattr(p, "column", None) == column:
            out.extend([p.value] if kind is Eq else list(p.values))
        elif isinstance(p, And):
            for q in p.preds:
                rec(q)

    rec(pred)
    return out


@serve_extension("pairs", lambda req: bool((req.get("split") or {}).get("pairs")))
def serve_pairs(ctx: ServiceContext, req: Mapping[str, Any]) -> dict[str, Any]:
    """``split: {pairs: {mode, args}}``: ``compare_drug_effects`` (one record) and
    ``find_cell_line_selective_effects`` (rows)."""
    opts = dict((req.get("split") or {}).get("pairs") or {})
    params = dict(req.get("params") or {})
    args = dict(opts.get("args") or {})

    def p(name: str, default: Any) -> Any:
        v = params.get(args.get(name, name))
        return default if v is None else v

    view = long_view(ctx, str(req.get("table")))
    pred = from_json(req["predicate"]) if req.get("predicate") else None
    drugs = _values(pred, DRUG, Eq)
    lines = _values(pred, LINE, Eq)
    if opts.get("mode") == "compare":
        if len(drugs) != 2:
            raise GatewayError(ErrorKind.invalid_argument, "compare_drug_effects needs two drugs")
        record = compare_drugs(view, drugs[0], drugs[1], cell_line=lines[0] if lines else None,
                               concentration=p("concentration", None), min_abs=p("min_abs_log2fc", 0.5),
                               max_padj=p("max_padj", 0.10), budget=req.get("budget_bytes"))
        sections = _sections(ctx, req, params)
        return ServeResponse(rows=[json_value(record)], total=1, key_columns=[], sections=sections,
                             served_by="derived").model_dump(mode="json")
    comparison = _values(pred, LINE, In) or None
    rows, stats = selectivity(view, drugs[0] if drugs else None, lines[0] if lines else None, comparison=comparison,
                              concentration=p("concentration", None), min_abs=p("min_abs_log2fc", 0.5),
                              max_padj=p("max_padj", 0.10), threshold=p("selectivity_threshold", 2.0),
                              budget=req.get("budget_bytes"))
    limit = req.get("limit")
    shown = rows[: int(limit)] if limit is not None else rows
    sections = _sections(ctx, req, params)
    sections["_selectivity"] = json_value(stats)
    return ServeResponse(rows=json_value(shown), total=len(rows), truncated=len(shown) < len(rows),
                         key_columns=["gene_name", CONC, UNIT, "plate"], sections=sections,
                         served_by="derived").model_dump(mode="json")


def _sections(ctx: ServiceContext, req: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """The request's metadata sections (drug and cell line records), read like any ``_serve`` section."""
    if not req.get("sections"):
        return {}
    from .views import serve_sections

    return dict(serve_sections(ctx, dict(req["sections"]), dict(params)))


VERBS = {"_pairs": guarded("pairs", pairs_verb)}
