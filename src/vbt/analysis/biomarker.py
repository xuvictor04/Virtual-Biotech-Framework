"""Predictive-biomarker analysis from case study 3 (gp130-axis score vs
anti-TNF / anti-integrin non-response in ulcerative colitis; Fig. 5C).

Implements the Supplementary Methods section "Biomarker analysis in
independent UC cohorts": microarray probes collapsed to genes (mean across
probes), a composite score equal to the mean of within-cohort z-scored genes,
and ROC AUC for discriminating non-responders (positive class) from
responders, compared between the gp130-axis score, OSMR alone and the Arijs
et al. five-gene signature across GEO cohorts GSE12251, GSE16879, GSE23597 and
GSE73661.
"""

from __future__ import annotations

import gzip
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "GP130_AXIS_GENES", "ARIJS_SIGNATURE", "DEFAULT_SIGNATURES", "GEO_COHORTS",
    "collapse_probes", "composite_score", "delong_auc_ci", "score_auc",
    "compare_scores_across_cohorts", "GeoSeriesMatrix", "load_geo_series_matrix",
]

GP130_AXIS_GENES: list[str] = ["OSMR", "IL6ST", "LIFR", "IL6R", "IL11RA", "IL6", "IL11", "OSM",
                               "LIF", "STAT1"]
ARIJS_SIGNATURE: list[str] = ["IL13RA2", "TNFRSF11B", "STC1", "PTGS2", "IL11"]
DEFAULT_SIGNATURES: dict[str, list[str]] = {
    "gp130_axis": GP130_AXIS_GENES,
    "OSMR_alone": ["OSMR"],
    "arijs_5gene": ARIJS_SIGNATURE,
}

#: Validation cohorts. Response labels are NOT standardised across series: they
#: must be derived per cohort from the ``!Sample_characteristics_ch1`` fields
#: (see :func:`load_geo_series_matrix`), restricting to UC, pre-treatment
#: biopsies and mapping the cohort's response annotation to 1 = non-responder,
#: 0 = responder.
GEO_COHORTS: dict[str, dict] = {
    "GSE12251": {"platform": "GPL570", "drug": "infliximab", "disease": "UC",
                 "note": "Arijs et al. 2009; pre-treatment colonic biopsies; derive response "
                         "from the week-8 response characteristic."},
    "GSE16879": {"platform": "GPL570", "drug": "infliximab", "disease": "UC and Crohn's disease",
                 "note": "Arijs et al. 2009; keep UC samples taken before the first infliximab "
                         "infusion; derive response from the response characteristic."},
    "GSE23597": {"platform": "GPL570", "drug": "infliximab (ACT1)", "disease": "UC",
                 "note": "ACT1 substudy; keep baseline (week 0) biopsies; derive week-8 response "
                         "from the characteristics; decide how to handle the placebo arm."},
    "GSE73661": {"platform": "GPL6244", "drug": "vedolizumab / infliximab", "disease": "UC",
                 "note": "Arijs et al. 2018; keep baseline biopsies; derive response from the "
                         "characteristics; treatment arms may be analysed separately."},
}
# Characteristic field names differ between series and must be checked against
# ``load_geo_series_matrix(path).samples.columns`` before labelling.


def collapse_probes(expr_df: pd.DataFrame, probe_to_gene: Mapping[str, str] | pd.Series,
                    drop_multi: bool = True) -> pd.DataFrame:
    """Collapse a probes x samples matrix to genes x samples by the mean across
    probes mapping to the same gene symbol. Unmapped probes are dropped; probes
    annotated to several genes (``'A /// B'``) are dropped when ``drop_multi``
    else assigned to the first symbol."""
    mapping = pd.Series(probe_to_gene) if not isinstance(probe_to_gene, pd.Series) else probe_to_gene
    mapping = mapping.astype(str).str.strip()
    genes = mapping.reindex(expr_df.index.astype(str))
    multi = genes.str.contains("///", regex=False).fillna(False)
    if drop_multi:
        genes = genes.where(~multi)
    else:
        genes = genes.str.split("///").str[0].str.strip()
    genes = genes.where(~genes.isin(["", "nan", "None", "---"]))
    ok = genes.notna().to_numpy()
    sub = expr_df.loc[ok].apply(pd.to_numeric, errors="coerce")
    out = sub.groupby(genes[ok].to_numpy()).mean()
    out.index.name = "gene"
    return out


def composite_score(expr_df: pd.DataFrame, genes: Sequence[str]) -> pd.Series:
    """Composite score per sample = mean of within-cohort z-scored genes.

    ``expr_df`` is samples x genes (log-scale). Genes absent from the matrix or
    with zero variance are skipped and reported in ``.attrs['genes_missing']``;
    ``.attrs['genes_used']`` lists the genes averaged."""
    present = [g for g in genes if g in expr_df.columns]
    sub = expr_df[present].apply(pd.to_numeric, errors="coerce")
    sd = sub.std(axis=0, ddof=1)
    used = [g for g in present if np.isfinite(sd[g]) and sd[g] > 0]
    missing = [g for g in genes if g not in used]
    if not used:
        s = pd.Series(np.nan, index=expr_df.index, name="score")
    else:
        z = (sub[used] - sub[used].mean(axis=0)) / sd[used]
        s = z.mean(axis=1, skipna=True).rename("score")
    s.attrs["genes_used"] = used
    s.attrs["genes_missing"] = missing
    return s


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong_auc_ci(scores, labels, alpha: float = 0.05) -> tuple[float, float, float, float]:
    """AUC with DeLong variance (Sun & Xu 2014 fast algorithm).
    Returns ``(auc, se, ci_low, ci_high)``; CI is clipped to [0, 1]."""
    from scipy import stats

    scores = np.asarray(scores, float)
    labels = np.asarray(labels).astype(int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    m, n = len(pos), len(neg)
    tx, ty, tz = _midrank(pos), _midrank(neg), _midrank(np.concatenate([pos, neg]))
    auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:m] - tx) / n
    v10 = 1 - (tz[m:] - ty) / m
    var = np.var(v01, ddof=1) / m + np.var(v10, ddof=1) / n if m > 1 and n > 1 else np.nan
    se = float(np.sqrt(var)) if np.isfinite(var) else np.nan
    zq = stats.norm.ppf(1 - alpha / 2)
    return float(auc), se, float(max(0.0, auc - zq * se)), float(min(1.0, auc + zq * se))


def score_auc(scores, labels, ci: str = "delong", n_boot: int = 2000, seed: int = 0,
              alpha: float = 0.05) -> dict:
    """ROC AUC of ``scores`` for non-response (``labels`` == 1 is the positive
    class, i.e. non-responder) with a DeLong or stratified-bootstrap CI (Methods,
    "Biomarker analysis in independent UC cohorts")."""
    from sklearn.metrics import roc_auc_score

    s = pd.Series(np.asarray(scores, float))
    y = pd.Series(np.asarray(labels, float))
    ok = s.notna() & y.notna()
    s, y = s[ok].to_numpy(), y[ok].to_numpy().astype(int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    res = {"auc": np.nan, "ci_low": np.nan, "ci_high": np.nan, "se": np.nan,
           "n_nonresponders": n_pos, "n_responders": n_neg, "ci_method": ci}
    if n_pos == 0 or n_neg == 0:
        return res
    res["auc"] = float(roc_auc_score(y, s))
    if ci == "delong":
        _, se, lo, hi = delong_auc_ci(s, y, alpha)
        res.update(se=se, ci_low=lo, ci_high=hi)
    elif ci == "bootstrap":
        rng = np.random.default_rng(seed)
        pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]
        aucs = np.empty(n_boot)
        for b in range(n_boot):
            idx = np.concatenate([rng.choice(pi, pi.size), rng.choice(ni, ni.size)])
            aucs[b] = roc_auc_score(y[idx], s[idx])
        lo, hi = np.quantile(aucs, [alpha / 2, 1 - alpha / 2])
        res.update(se=float(aucs.std(ddof=1)), ci_low=float(lo), ci_high=float(hi))
    elif ci not in (None, "none"):
        raise ValueError(f"unknown ci {ci!r}")
    return res


def compare_scores_across_cohorts(cohorts: Mapping[str, tuple[pd.DataFrame, pd.Series]],
                                  signatures: Mapping[str, Sequence[str]] | None = None,
                                  ci: str = "delong", **auc_kwargs) -> pd.DataFrame:
    """AUC table (cohort x signature) as in Fig. 5C.

    ``cohorts`` maps a name to ``(expr_df samples x genes, labels)`` with labels
    1 = non-responder, 0 = responder (indexed like ``expr_df`` or positional).
    Default signatures: gp130 axis, OSMR alone, Arijs five-gene signature.
    """
    signatures = dict(signatures or DEFAULT_SIGNATURES)
    rows = []
    for name, (expr, labels) in cohorts.items():
        lab = labels.reindex(expr.index) if isinstance(labels, pd.Series) and \
            labels.index.isin(expr.index).all() else pd.Series(np.asarray(labels), index=expr.index)
        for sig, genes in signatures.items():
            sc = composite_score(expr, genes)
            r = score_auc(sc.to_numpy(), lab.to_numpy(), ci=ci, **auc_kwargs)
            rows.append({"cohort": name, "signature": sig, **r,
                         "n_genes_used": len(sc.attrs["genes_used"]),
                         "genes_missing": ",".join(sc.attrs["genes_missing"])})
    cols = ["cohort", "signature", "auc", "ci_low", "ci_high", "se", "n_nonresponders",
            "n_responders", "n_genes_used", "genes_missing", "ci_method"]
    return pd.DataFrame(rows)[cols]


@dataclass
class GeoSeriesMatrix:
    """Parsed GEO series matrix: ``expression`` (probes x samples),
    ``samples`` (GSM x annotation incl. parsed characteristics) and ``series``
    (series-level metadata)."""
    expression: pd.DataFrame
    samples: pd.DataFrame
    series: dict = field(default_factory=dict)


def _split_line(line: str) -> list[str]:
    parts = line.rstrip("\n").rstrip("\r").split("\t")
    return [p[1:-1] if len(p) >= 2 and p.startswith('"') and p.endswith('"') else p for p in parts]


def load_geo_series_matrix(path) -> GeoSeriesMatrix:
    """Parse a GEO ``*_series_matrix.txt`` or ``.txt.gz`` file.

    Sample attributes (``!Sample_*`` lines) become columns of ``samples``
    (indexed by GSM); each ``!Sample_characteristics_ch1`` line of the form
    ``key: value`` becomes a column named by ``key`` (duplicates suffixed
    ``_2``...). The data table between ``!series_matrix_table_begin`` and
    ``!series_matrix_table_end`` becomes ``expression`` (float, probes x GSM).
    Response labels must then be derived per cohort (see :data:`GEO_COHORTS`).
    """
    path = Path(path)
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    text = raw.decode("utf-8", errors="replace")
    series: dict = {}
    sample_attrs: list[tuple[str, list[str]]] = []
    table_lines: list[str] = []
    in_table = False
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith("!series_matrix_table_begin"):
            in_table = True
            continue
        if line.startswith("!series_matrix_table_end"):
            in_table = False
            continue
        if in_table:
            table_lines.append(line)
        elif line.startswith("!Series_"):
            parts = _split_line(line)
            key = parts[0][len("!Series_"):]
            val = parts[1:] if len(parts) > 2 else (parts[1] if len(parts) > 1 else "")
            if key in series:
                prev = series[key] if isinstance(series[key], list) else [series[key]]
                series[key] = prev + (val if isinstance(val, list) else [val])
            else:
                series[key] = val
        elif line.startswith("!Sample_"):
            parts = _split_line(line)
            sample_attrs.append((parts[0][len("!Sample_"):], parts[1:]))

    if table_lines:
        expr = pd.read_csv(io.StringIO("\n".join(table_lines)), sep="\t", index_col=0,
                           quotechar='"', na_values=["null", "NA", ""])
        expr.index = expr.index.astype(str)
        expr.index.name = "ID_REF"
        expr = expr.apply(pd.to_numeric, errors="coerce")
    else:
        expr = pd.DataFrame()

    gsm = next((v for k, v in sample_attrs if k == "geo_accession"), None)
    if gsm is None:
        gsm = list(expr.columns)
    samples = pd.DataFrame(index=pd.Index(gsm, name="gsm"))
    seen: dict[str, int] = {}

    def _put(col, vals):
        n = seen.get(col, 0) + 1
        seen[col] = n
        name = col if n == 1 else f"{col}_{n}"
        vals = list(vals) + [None] * (len(gsm) - len(vals))
        samples[name] = vals[: len(gsm)]

    for key, vals in sample_attrs:
        if key == "geo_accession":
            continue
        if key.startswith("characteristics"):
            keys = {v.split(":", 1)[0].strip() for v in vals if v and ":" in v}
            if len(keys) == 1:
                k = keys.pop()
                _put(k, [v.split(":", 1)[1].strip() if v and ":" in v else None for v in vals])
                continue
            if len(keys) > 1:  # mixed keys on one line: spread into separate columns
                for k in sorted(keys):
                    _put(k, [v.split(":", 1)[1].strip() if v and ":" in v and
                             v.split(":", 1)[0].strip() == k else None for v in vals])
                continue
        _put(key, vals)
    return GeoSeriesMatrix(expression=expr, samples=samples, series=series)
