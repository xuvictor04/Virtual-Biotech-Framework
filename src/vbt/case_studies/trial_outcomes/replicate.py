"""Exact replication of Case study 1 on the authors' own inputs (Zenodo archive).

The paper's Zenodo record ships ``clinical_trials/{code,data,results}``: the
authors' analysis scripts, the exact inputs (reconciled labels, ChEMBL NCT
mapping, aggregated target features, Open Targets subsets) and every result
table behind Fig. 3 / figs. S2–S4. :func:`replicate_case1` runs *our*
implementation (:mod:`vbt.case_studies.trial_outcomes.stats`) on exactly
those inputs, aligns each of our estimates with the authors' row (feature ×
outcome × model) and writes

* ``case1_replication_comparison.csv`` — ours vs theirs (estimate, CI, p, n,
  absolute / relative deltas, match flag, engine) for every compared row;
* ``case1_replication_summary.md`` — per-table agreement and the headline
  numbers (ours vs theirs vs the paper text);
* ``case1_replication_headline.json`` and our own tables (``ours_*.csv``);
* ``case1_harness_checks.csv`` — the harness's confounding checks on the
  real features (target-count adjustment, gene-label permutation null).

Engines: logistic models are statsmodels GLMs (the authors' engine); beta
regressions use R ``betareg`` through rpy2 when installed, else
:func:`~vbt.case_studies.trial_outcomes.stats.betareg_fit` (same estimator and
expected-information SEs); mixed models use lme4/glmmTMB when available, else
:func:`~vbt.case_studies.trial_outcomes.stats.glmm_laplace`.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from . import stats as st

__all__ = [
    "PAPER_HEADLINES", "SC_PREFIXES", "ReplicationReport", "load_inputs", "build_trials",
    "non_expr_features", "run_univariate", "run_expr", "run_permutation", "run_binary", "run_genetic",
    "run_combined", "run_mixed", "harness_checks", "gene_permutation_null", "compare_tables",
    "load_reference_tables", "replicate_case1", "headline_numbers",
]

#: Numbers stated in the paper text (Results, Case study 1).
PAPER_HEADLINES: dict[str, float] = {
    "or_primary_tau": 1.12,
    "or_secondary_tau": 1.12,
    "or_phase1to2_tau": 1.27,
    "phase4_relative_increase": 0.48,
    "phase1to2_relative_increase": 0.40,
    "ae_relative_difference": -0.32,
    "tau_threshold": 0.69,
    "pearson_tau_bimodality": 0.54,
    "n_trials": 55984,
}

#: Target-feature column prefixes aggregated by MIN across a trial's targets
#: (the authors' ``sc_cols``).
SC_PREFIXES = ("mean_", "detection_", "cv", "p90_", "tau_", "n_expressing", "expr_", "ae_risk_", "tissue_",
               "rare_", "critical_", "variance_", "bimodality", "outlier", "cell_type_selectivity",
               "donor_variance", "peak_cell_type_expr")

_CORE = ["mean_expr", "detection_rate", "cv", "p90_expr", "donor_variance_ratio"]
_SPECIFICITY = ["tau_cell_type", "tau_cell_type_max", "tau_cell_type_min", "tau_cell_type_range",
                "tau_cell_type_variance", "n_tissues_high_tau", "tissue_tau", "cell_type_selectivity",
                "n_expressing_cell_types", "n_tissues_expressing"]
_HETEROGENEITY = ["variance_within_celltype_max", "bimodality_score", "outlier_cell_fraction"]
_OTHER = ["peak_cell_type_expr", "rare_celltype_enrichment", "critical_rare_expr_max", "tissue_entropy"]
FEATURES = ("tau_cell_type", "bimodality_score")


def _log(progress: Callable[[str], None] | None, msg: str) -> None:
    if progress:
        progress(msg)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_inputs(root: str | Path, *, open_targets: bool = True) -> dict[str, Any]:
    """Read ``clinical_trials/data`` of the extracted archive (``root`` =
    ``.../virtualbiotech_submission`` or its ``clinical_trials`` folder)."""
    root = Path(root)
    ct = root if (root / "data" / "clinical_trial_labels_reconciled.csv").exists() else root / "clinical_trials"
    d = ct / "data"
    if not (d / "clinical_trial_labels_reconciled.csv").exists():
        raise FileNotFoundError(f"{d}/clinical_trial_labels_reconciled.csv not found - run "
                                "`vbt data zenodo fetch --preset case1` first")
    out: dict[str, Any] = dict(
        root=ct,
        labels=pd.read_csv(d / "clinical_trial_labels_reconciled.csv"),
        mapping=pd.read_parquet(d / "chembl_clinical_nct_data.parquet"),
        features=pd.read_parquet(d / "comprehensive_features_aggregated_v2_optimized.parquet"),
    )
    if open_targets:
        ga = d / "association_by_datatype_direct"
        if ga.exists():
            g = pd.read_parquet(ga, columns=["targetId", "diseaseId", "datatypeId", "score"])
            out["genetic"] = g[g["datatypeId"] == "genetic_association"][["targetId", "diseaseId", "score"]]
        if (d / "drug_molecule").exists():
            out["drug_types"] = pd.read_parquet(d / "drug_molecule", columns=["id", "drugType"]) \
                .set_index("id")["drugType"]
        if (d / "disease").exists():
            out["disease"] = pd.read_parquet(d / "disease", columns=["id", "name", "therapeuticAreas"])
    return out


def sc_columns(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if c.startswith(SC_PREFIXES)]


def build_trials(inputs: dict[str, Any], *, require: Sequence[str] = FEATURES, how: str = "min") -> pd.DataFrame:
    """Trial-level table as in the authors' scripts (MIN across targets), sorted by NCT ID."""
    t = st.aggregate_trials(inputs["labels"], inputs["mapping"], inputs["features"],
                            feature_cols=sc_columns(inputs["features"]), how=how, require=require)
    n_all = inputs["mapping"][["nct_id", "targetId"]].drop_duplicates().groupby("nct_id").size()
    t["n_targets_all"] = t["nct_id"].map(n_all).fillna(1).astype(int)
    return t.sort_values("nct_id").reset_index(drop=True)


def non_expr_features(trials: pd.DataFrame) -> list[str]:
    ae = [c for c in trials.columns if c.startswith("ae_risk_")]
    return [f for f in (_CORE + _SPECIFICITY + _HETEROGENEITY + ae + _OTHER) if f in trials.columns]


# ---------------------------------------------------------------------------
# Univariate models (figure2_virtualbiotech_analysis.py)
# ---------------------------------------------------------------------------


def _logit_row(sub: pd.DataFrame, x: pd.Series, y: pd.Series, feature: str) -> dict:
    d = pd.DataFrame({feature: x.to_numpy(dtype=float), "_y": y.to_numpy(dtype=float)})
    r = st.univariate_logistic(d, feature, "_y")
    return dict(feature=feature, coefficient=r["beta"], OR=r["odds_ratio"], CI_lower=r["ci_low"],
                CI_upper=r["ci_high"], p_value=r["p"], se=r["se"], engine="statsmodels")


def run_univariate(trials: pd.DataFrame, features: Sequence[str] | None = None, *,
                   beta_engine: str = "auto", progress=None) -> pd.DataFrame:
    """Sections A–F for each feature: univariate logistic (feature NaN → 0, as
    the authors) and beta regression per AE organ (feature NaN dropped). BH
    over the whole table. Columns follow ``figure2_virtualbiotech_all_results.csv``."""
    features = list(features or non_expr_features(trials))
    rows = []
    for section, outcome, otype, value in st.AUTHORS_ANALYSES:
        fr = st.authors_outcome_frame(trials, otype, value)
        if fr is None:
            continue
        sub, y = fr
        npos, nref = int(y.sum()), int((y == 0).sum())
        for f in features:
            r = _logit_row(sub, sub[f].fillna(0), y, f)
            rows.append(dict(section=section, outcome=outcome, regression_type="logistic", n_positive=npos,
                             n_reference=nref, **r))
        _log(progress, f"  univariate {section}: {outcome} (n={npos + nref})")
    for col, label in st.AE_ORGAN_SYSTEMS:
        if col not in trials:
            continue
        sub = trials[trials[col].notna()]
        if len(sub) < 30:
            continue
        for f in features:
            fv = sub[sub[f].notna()]
            r = st.beta_regression(fv, f, col, engine=beta_engine)
            rows.append(dict(section="AE Organ System", outcome=f"{label} (original)", regression_type="beta",
                             feature=f, n_positive=np.nan, n_reference=int(r["n"]), coefficient=r["coef"],
                             OR=np.nan, CI_lower=r["ci_low"], CI_upper=r["ci_high"], p_value=r["p"],
                             se=r["se"], engine=r["engine"], ae_column=col,
                             outcome_mean=float(fv[col].mean())))
        _log(progress, f"  univariate AE: {label}")
    out = pd.DataFrame(rows)
    out["fdr_adjusted_p"] = st.benjamini_hochberg(out["p_value"])
    return out


def run_expr(trials: pd.DataFrame, progress=None) -> pd.DataFrame:
    """Cell-type expression features: log2 fold change vs mean expression
    (univariate) and the raw feature adjusted for ``mean_expr``; BH per method."""
    expr = [c for c in trials.columns if c.startswith("expr_") and not c.startswith("expr_fc_")]
    t = trials.copy()
    me = t["mean_expr"].fillna(0)
    fc = {f.replace("expr_", "expr_fc_"): np.log2((t[f].fillna(0) + 1) / (me + 1)) for f in expr}
    t = pd.concat([t, pd.DataFrame(fc, index=t.index)], axis=1)
    rows = []
    for section, outcome, otype, value in st.AUTHORS_ANALYSES:
        fr = st.authors_outcome_frame(t, otype, value)
        if fr is None:
            continue
        sub, y = fr
        npos, nref = int(y.sum()), int((y == 0).sum())
        d = pd.DataFrame({"_y": y.to_numpy(dtype=float), "mean_expr": sub["mean_expr"].fillna(0).to_numpy()})
        for f in expr:
            fcn = f.replace("expr_", "expr_fc_")
            r = _logit_row(sub, sub[fcn].fillna(0), y, fcn)
            rows.append(dict(section=section, outcome=outcome, method="fold_change", n_positive=npos,
                             n_reference=nref, **r))
            d[f] = sub[f].fillna(0).to_numpy()
            try:
                m = st.multivariable_logistic(d, [f, "mean_expr"], "_y", [True, True])[0]
            except Exception:  # noqa: BLE001
                m = dict(feature=f, coefficient=np.nan, OR=np.nan, CI_lower=np.nan, CI_upper=np.nan,
                         p_value=np.nan, se=np.nan)
            m.pop("n", None)
            rows.append(dict(section=section, outcome=outcome, method="adjusted_mean_expr", n_positive=npos,
                             n_reference=nref, engine="statsmodels", **m))
        _log(progress, f"  expr {section}: {outcome}")
    out = pd.DataFrame(rows)
    out["fdr_adjusted_p"] = np.nan
    for method in ("fold_change", "adjusted_mean_expr"):
        msk = out["method"] == method
        out.loc[msk, "fdr_adjusted_p"] = st.benjamini_hochberg(out.loc[msk, "p_value"])
    return out


# ---------------------------------------------------------------------------
# Permutation test (figure2_permutation_test.py)
# ---------------------------------------------------------------------------


def _fast_logit_coef(x: np.ndarray, y: np.ndarray, iters: int = 30) -> float:
    """Slope of a one-predictor logistic regression by Newton–Raphson."""
    b0 = np.log(max(y.mean(), 1e-12) / max(1 - y.mean(), 1e-12))
    b1 = 0.0
    for _ in range(iters):
        eta = b0 + b1 * x
        p = 1 / (1 + np.exp(-eta))
        w = p * (1 - p)
        r = y - p
        g0, g1 = r.sum(), (r * x).sum()
        h00, h01, h11 = w.sum(), (w * x).sum(), (w * x * x).sum()
        det = h00 * h11 - h01 * h01
        if det <= 0:
            return float("nan")
        d0 = (h11 * g0 - h01 * g1) / det
        d1 = (h00 * g1 - h01 * g0) / det
        b0, b1 = b0 + d0, b1 + d1
        if max(abs(d0), abs(d1)) < 1e-12:
            break
    return float(b1)


def _z(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return (v - v.mean()) / v.std()


#: the authors' FEATURES_TO_TEST order (it fixes which permutations each feature gets)
PERMUTATION_FEATURES = ("bimodality_score", "tau_cell_type")


def run_permutation(trials: pd.DataFrame, n_perm: int = 1000, *, n_perm_beta: int | None = None,
                    seed: int = 42, features: Sequence[str] = PERMUTATION_FEATURES, progress=None,
                    keep_null: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Outcome-permutation nulls exactly as the authors: one
    ``np.random.default_rng(seed)`` consumed in the authors' order (logistic
    analyses, then AE organs; per feature ``n_perm`` × ``rng.permutation(y)``),
    two-sided empirical p = (#|null| ≥ |obs| + 1) / (n + 1). With the
    authors' seed and ``n_perm`` the shuffles are identical to theirs.
    ``n_perm_beta`` (default ``n_perm``) caps the slower beta permutations;
    when it differs from ``n_perm`` the RNG stream (and thus the null draws)
    diverges from the authors' for the AE rows."""
    rng = np.random.default_rng(seed)
    n_perm_beta = n_perm if n_perm_beta is None else n_perm_beta
    rows, nulls = [], []
    for section, outcome, otype, value in st.AUTHORS_ANALYSES:
        fr = st.authors_outcome_frame(trials, otype, value)
        if fr is None:
            continue
        sub, y = fr
        yv = y.to_numpy(dtype=float)
        for f in features:
            x = _z(sub[f].to_numpy())
            obs = _logit_row(sub, sub[f], y, f)
            null = np.array([_fast_logit_coef(x, rng.permutation(yv)) for _ in range(n_perm)])
            ok = null[np.isfinite(null)]
            emp = (np.sum(np.abs(ok) >= abs(obs["coefficient"])) + 1) / (len(ok) + 1) if len(ok) else np.nan
            rows.append(dict(section=section, outcome=outcome, feature=f, regression_type="logistic",
                             n_positive=int(yv.sum()), n_reference=int((yv == 0).sum()),
                             observed_coefficient=obs["coefficient"], observed_OR=obs["OR"],
                             observed_CI_lower=obs["CI_lower"], observed_CI_upper=obs["CI_upper"],
                             observed_p_value=obs["p_value"], null_mean=float(np.nanmean(null)),
                             null_std=float(np.nanstd(null)), null_median=float(np.nanmedian(null)),
                             **{"null_2.5pct": float(np.nanpercentile(null, 2.5)),
                                "null_97.5pct": float(np.nanpercentile(null, 97.5))},
                             empirical_p_value=float(emp), n_permutations=n_perm))
            if keep_null:
                nulls.append(pd.DataFrame(dict(section=section, outcome=outcome, feature=f,
                                               permutation=np.arange(n_perm), null_coefficient=null)))
        _log(progress, f"  permutation {section}: {outcome}")
    for col, label in (st.AE_ORGAN_SYSTEMS if n_perm_beta > 0 else ()):
        if col not in trials:
            continue
        sub = trials[trials[col].notna()]
        if len(sub) < 30:
            continue
        for f in features:
            fv = sub[sub[f].notna()]
            x = _z(fv[f].to_numpy())
            yp = fv[col].to_numpy(dtype=float)
            obs = st.beta_regression(fv, f, col)
            X = np.column_stack([np.ones(len(x)), x])
            null = np.full(n_perm_beta, np.nan)
            for i in range(n_perm_beta):
                yy = st._squeeze(rng.permutation(yp) / 100.0, st.BETA_SQUEEZE, len(yp))
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # full ML fit (quasi-Newton + scoring), as R betareg in the authors' loop
                        null[i] = st.betareg_fit(yy, X)["params"][1]
                except Exception:  # noqa: BLE001
                    pass
            ok = null[np.isfinite(null)]
            emp = (np.sum(np.abs(ok) >= abs(obs["coef"])) + 1) / (len(ok) + 1) if len(ok) else np.nan
            rows.append(dict(section="AE Organ System", outcome=f"{label} (original)", feature=f,
                             regression_type="beta", n_positive=np.nan, n_reference=len(fv),
                             observed_coefficient=obs["coef"], observed_OR=np.nan,
                             observed_CI_lower=obs["ci_low"], observed_CI_upper=obs["ci_high"],
                             observed_p_value=obs["p"], null_mean=float(np.nanmean(null)),
                             null_std=float(np.nanstd(null)), null_median=float(np.nanmedian(null)),
                             **{"null_2.5pct": float(np.nanpercentile(null, 2.5)),
                                "null_97.5pct": float(np.nanpercentile(null, 97.5))},
                             empirical_p_value=float(emp), n_permutations=n_perm_beta,
                             mean_ae_pct=float(yp.mean())))
            if keep_null:
                nulls.append(pd.DataFrame(dict(section="AE Organ System", outcome=f"{label} (original)",
                                               feature=f, permutation=np.arange(n_perm_beta),
                                               null_coefficient=null)))
        _log(progress, f"  permutation AE: {label}")
    out = pd.DataFrame(rows)
    if keep_null:
        return out, (pd.concat(nulls, ignore_index=True) if nulls else pd.DataFrame())
    return out


# ---------------------------------------------------------------------------
# Binary tau / bimodality (figure2_binary_tau_analysis.py)
# ---------------------------------------------------------------------------


def run_binary(inputs: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    """K-means thresholds on the trial-level tau / bimodality distributions and
    the target-level, AE and Phase I→II contrasts (the 48% / 32% / 40% results)."""
    trials = build_trials(inputs, require=("tau_cell_type",))
    thr, (c_lo, c_hi) = st.kmeans_threshold(trials["tau_cell_type"], seed)
    bthr, (b_lo, b_hi) = st.kmeans_threshold(trials["bimodality_score"].dropna(), seed)
    trials["tau_binary"] = (trials["tau_cell_type"] >= thr).astype(int)
    trials["bimodality_binary"] = np.where(trials["bimodality_score"].notna(),
                                           (trials["bimodality_score"] >= bthr).astype(float), np.nan)
    lwt = inputs["labels"].merge(inputs["mapping"][["nct_id", "targetId"]].drop_duplicates(), on="nct_id")
    g = inputs["features"].set_index("ensembl_id")
    tl_tau = st.target_level_table(lwt, g["tau_cell_type"], thr).assign(section="Target-Level")
    tl_bim = st.target_level_table(lwt, g["bimodality_score"], bthr).assign(section="Target-Level-Bimodality")
    ae_tau = st.ae_group_comparison(trials, "tau_binary").assign(section="AE Organ System", threshold=thr)
    ae_bim = st.ae_group_comparison(trials, "bimodality_binary").assign(
        section="AE Organ System (Bimodality)", threshold=bthr)
    p1 = trials[(trials["phase"] == 1.0) & trials["phase2_progression"].isin(["EVER", "NEVER"])].copy()
    p1["progressed"] = (p1["phase2_progression"] == "EVER").astype(int)
    prog = []
    for metric, col, t in (("tau_cell_type", "tau_binary", thr), ("bimodality_score", "bimodality_binary", bthr)):
        pp = p1[p1[col].notna()]
        r = st.contingency_or(pp[col].astype(int).to_numpy(), pp["progressed"].to_numpy())
        prog.append(dict(metric=metric, threshold=t, threshold_method="kmeans_k2_midpoint", n_total=len(pp), **r))
    return dict(threshold=thr, centers=(c_lo, c_hi), bimodality_threshold=bthr, bimodality_centers=(b_lo, b_hi),
                n_trials=len(trials), share_specific=float(trials["tau_binary"].mean()),
                target_tau=tl_tau, target_bimodality=tl_bim, ae_tau=ae_tau, ae_bimodality=ae_bim,
                phase1to2=pd.DataFrame(prog))


# ---------------------------------------------------------------------------
# Genetic evidence (figure2_genetic_association_analysis.py) and combined models
# ---------------------------------------------------------------------------


def _genetic(inputs: dict[str, Any]) -> pd.DataFrame:
    if "genetic" not in inputs:
        raise KeyError("Open Targets association_by_datatype_direct not in the archive extract")
    return st.genetic_evidence_table(inputs["mapping"], inputs["genetic"])


def run_genetic(inputs: dict[str, Any], *, beta_engine: str = "auto") -> pd.DataFrame:
    """Razuvayevskaya-style replication: outcomes on ``has_genetic_evidence_any``
    (raw 0/1) and ``fraction_genetic_evidence`` (z-scored); AE beta models on
    both (the authors z-score both there)."""
    merged = inputs["labels"].merge(_genetic(inputs), on="nct_id", how="inner").sort_values("nct_id")
    feats = {"has_genetic_evidence_any": False, "fraction_genetic_evidence": True}
    rows = []
    for section, outcome, otype, value in st.AUTHORS_ANALYSES:
        fr = st.authors_outcome_frame(merged, otype, value)
        if fr is None:
            continue
        sub, y = fr
        for f, stdz in feats.items():
            d = pd.DataFrame({f: sub[f].to_numpy(dtype=float), "_y": y.to_numpy(dtype=float)})
            if stdz:
                r = st.univariate_logistic(d, f, "_y")
                rr = dict(coefficient=r["beta"], OR=r["odds_ratio"], CI_lower=r["ci_low"], CI_upper=r["ci_high"],
                          p_value=r["p"])
            else:
                r = st.multivariable_logistic(d, [f], "_y", [False])[0]
                rr = {k: r[k] for k in ("coefficient", "OR", "CI_lower", "CI_upper", "p_value")}
            rows.append(dict(section=section, outcome=outcome, regression_type="logistic", feature=f,
                             feature_type="continuous" if stdz else "binary", n_positive=int(y.sum()),
                             n_reference=int((y == 0).sum()), engine="statsmodels", **rr))
    for col, label in st.AE_ORGAN_SYSTEMS:
        if col not in merged:
            continue
        sub = merged[merged[col].notna()]
        if len(sub) < 30:
            continue
        for f, stdz in feats.items():
            r = st.beta_regression(sub, f, col, engine=beta_engine)
            rows.append(dict(section="AE Organ System", outcome=f"{label} (original)", regression_type="beta",
                             feature=f, feature_type="continuous" if stdz else "binary", n_positive=np.nan,
                             n_reference=int(r["n"]), coefficient=r["coef"], OR=np.nan, CI_lower=r["ci_low"],
                             CI_upper=r["ci_high"], p_value=r["p"], engine=r["engine"], ae_column=col))
    out = pd.DataFrame(rows)
    out["fdr_adjusted_p"] = st.benjamini_hochberg(out["p_value"])
    return out


_COMBINED_MODELS = (
    ("univariate", None),
    ("multivariate_3way", ["has_genetic_evidence_any", "tau_cell_type", "bimodality_score"]),
    ("multivariate_2way_sc_only", ["tau_cell_type", "bimodality_score"]),
    ("multivariate_2way_genetic_tau", ["has_genetic_evidence_any", "tau_cell_type"]),
    ("multivariate_2way_genetic_bimodality", ["has_genetic_evidence_any", "bimodality_score"]),
)
_COMBINED_ANALYSES = [a for a in st.AUTHORS_ANALYSES
                      if a[2] != "stop_reason" or a[3] in st.STOP_REASON_CATEGORIES[:4]]


def run_combined(inputs: dict[str, Any], trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correlation of genetic and single-cell features, and the univariate /
    2-way / 3-way models (genetic flag raw, tau and bimodality z-scored)."""
    from scipy.stats import pearsonr, spearmanr

    merged = trials.merge(_genetic(inputs), on="nct_id", how="inner").sort_values("nct_id")
    key = [f for f in ["bimodality_score", "tau_cell_type", "tissue_tau", "mean_expr", "detection_rate", "cv",
                       "p90_expr", "cell_type_selectivity", "n_expressing_cell_types",
                       "variance_within_celltype_max", "outlier_cell_fraction"] if f in merged]
    corr = []
    for gf in ("has_genetic_evidence_any", "fraction_genetic_evidence"):
        for sf in key:
            ok = merged[gf].notna() & merged[sf].notna()
            if ok.sum() < 100:
                continue
            pr, pp = pearsonr(merged.loc[ok, gf], merged.loc[ok, sf])
            sr, sp = spearmanr(merged.loc[ok, gf], merged.loc[ok, sf])
            corr.append(dict(genetic_feature=gf, sc_feature=sf, n_samples=int(ok.sum()), pearson_r=float(pr),
                             pearson_p=float(pp), spearman_r=float(sr), spearman_p=float(sp),
                             abs_spearman_r=abs(float(sr))))
    std = {"has_genetic_evidence_any": False, "tau_cell_type": True, "bimodality_score": True}
    rows = []
    for section, outcome, otype, value in _COMBINED_ANALYSES:
        fr = st.authors_outcome_frame(merged, otype, value)
        if fr is None:
            continue
        sub, y = fr
        d = sub[list(std)].copy()
        d["_y"] = y.to_numpy(dtype=float)
        base = dict(section=section, outcome=outcome, n_positive=int(y.sum()), n_reference=int((y == 0).sum()))
        for model, cols in _COMBINED_MODELS:
            for cs in ([[c] for c in std] if cols is None else [cols]):
                for r in st.multivariable_logistic(d, cs, "_y", [std[c] for c in cs]):
                    r.pop("n", None)
                    rows.append(dict(base, model=model, **r))
    for col, label in st.AE_ORGAN_SYSTEMS:
        if col not in merged:
            continue
        sub = merged[merged[col].notna()]
        if len(sub) < 30:
            continue
        base = dict(section="AE Organ System", outcome=f"{label} (original)", n_trials=len(sub),
                    mean_ae_pct=float(sub[col].mean()))
        for model, cols in _COMBINED_MODELS:
            for cs in ([[c] for c in std] if cols is None else [cols]):
                for r in st.multivariable_beta(sub, cs, col, [std[c] for c in cs]):
                    r.pop("n", None)
                    rows.append(dict(base, model=model, **r))
    out = pd.DataFrame(rows)
    out["fdr_adjusted_p"] = st.benjamini_hochberg(out["p_value"])
    return pd.DataFrame(corr), out


# ---------------------------------------------------------------------------
# Mixed effects (figure2_virtualbiotech_analysis_mixedeffects.py)
# ---------------------------------------------------------------------------


def mixed_dataset(inputs: dict[str, Any], trials: pd.DataFrame) -> pd.DataFrame:
    """Trials with drugType / therapeutic-area combinations and enrolment year."""
    m = inputs["mapping"]
    t = trials.merge(st.drugtype_combo(m, inputs["drug_types"]), on="nct_id", how="left") \
        .merge(st.ta_combo(m, inputs["disease"]), on="nct_id", how="left")
    yr = pd.to_datetime(m["trial_date"], errors="coerce").dt.year.groupby(m["nct_id"]).min().rename("trial_year")
    t = t.merge(yr.reset_index(), on="nct_id", how="left")
    t = t[t["drugType_combo"].notna() & t["ta_combo"].notna() & t["trial_year"].notna()].copy()
    t["years_from_earliest"] = t["trial_year"] - t["trial_year"].min()
    return t.sort_values("nct_id").reset_index(drop=True)


def _independent_columns(X: np.ndarray, tol: float = 1e-9) -> list[int]:
    return st.independent_columns(X, tol)


def run_mixed(inputs: dict[str, Any], trials: pd.DataFrame, *, engine: str = "auto",
              features: Sequence[str] = FEATURES, progress=None) -> pd.DataFrame:
    """GLMMs with crossed random intercepts for drug-type and TA combinations,
    the authors' per-outcome covariates (year_z; phase dummies for status and
    AE; phase3 for endpoints), every fixed-effect term reported; BH over the
    feature terms only."""
    t = mixed_dataset(inputs, trials)
    analyses = [a for a in st.AUTHORS_ANALYSES if a[2] != "stop_reason"]
    rows = []
    use_r = engine == "R"  # only the Python Laplace port is wired here (the R path lives in stats.mixed_effects_*)

    def fit(sub, y, kind, family):
        groups = [sub["drugType_combo"].to_numpy(), sub["ta_combo"].to_numpy()]
        cov = st.mixed_covariates(kind, sub)
        out = []
        for f in features:
            x = _z(sub[f].to_numpy())
            all_names = ["feature_z", *cov]
            full = np.column_stack([np.ones(len(sub)), x, *cov.values()])
            keep = _independent_columns(full)  # R drops aliased columns (NA coefficients)
            X = full[:, keep]
            kept_names = [all_names[j - 1] for j in keep[1:]]
            t0 = time.time()
            if use_r:
                raise NotImplementedError("R engine: use stats.mixed_effects_logistic/beta with engine='lme4'")
            r = st.glmm_laplace(np.asarray(y, dtype=float), X, groups, family=family)
            for term in all_names:
                if term not in kept_names:
                    out.append(dict(feature=f if term == "feature_z" else term, analysis_feature=f, term=term,
                                    coefficient=np.nan, se=np.nan, converged=r["converged"],
                                    engine=f"laplace_py[{family}]", note="aliased (constant or collinear) - dropped"))
                    continue
                j = kept_names.index(term) + 1
                b, se = float(r["params"][j]), float(r["se"][j])
                out.append(dict(feature=f if term == "feature_z" else term, analysis_feature=f, term=term,
                                coefficient=b, se=se, OR=float(np.exp(b)) if family == "binomial" else np.nan,
                                CI_lower=(np.exp(b - 1.96 * se) if family == "binomial" else b - 1.96 * se),
                                CI_upper=(np.exp(b + 1.96 * se) if family == "binomial" else b + 1.96 * se),
                                z_value=b / se, p_value=float(r["p"][j]), RE_variance_drugType=float(r["var"][0]),
                                RE_variance_ta=float(r["var"][1]), converged=r["converged"],
                                n_groups_drugType=len(np.unique(groups[0])), n_groups_ta=len(np.unique(groups[1])),
                                engine=f"laplace_py[{family}]", seconds=round(time.time() - t0, 1)))
        return out

    for section, outcome, otype, value in analyses:
        if otype == "endpoint":
            sub = t[t["phase"].isin([2.0, 3.0])]
        elif otype == "phase1_progression":
            sub = t[t["phase"] == 1.0]
        else:
            sub = t
        y = st.authors_outcome(sub, otype, value)
        if otype == "phase1_progression":
            sub, y = sub[y.notna()], y[y.notna()]
        if int(y.sum()) < 30:
            continue
        for r in fit(sub, y.to_numpy(dtype=float), otype, "binomial"):
            rows.append(dict(section=section, outcome=outcome, regression_type="logistic_glmm",
                             n_positive=int(y.sum()), n_reference=int((y == 0).sum()), **r))
        _log(progress, f"  mixed {section}: {outcome}")
    for col, label in st.AE_ORGAN_SYSTEMS:
        sub = t[t[col].notna()]
        if len(sub) < 30:
            continue
        y = st._squeeze(sub[col].to_numpy(dtype=float) / 100.0, st.BETA_SQUEEZE)
        for r in fit(sub, y, "ae", "beta"):
            rows.append(dict(section="AE Organ System", outcome=label, regression_type="beta_glmmTMB",
                             n_positive=np.nan, n_reference=len(sub), mean_ae_pct=float(sub[col].mean()), **r))
        _log(progress, f"  mixed AE: {label}")
    out = pd.DataFrame(rows)
    msk = (out["term"] == "feature_z") & out["p_value"].notna()
    out["fdr_adjusted_p"] = np.nan
    out.loc[msk, "fdr_adjusted_p"] = st.benjamini_hochberg(out.loc[msk, "p_value"])
    return out


# ---------------------------------------------------------------------------
# Harness confounding checks on the real features
# ---------------------------------------------------------------------------


def gene_permutation_null(inputs: dict[str, Any], trials: pd.DataFrame, *, n_iter: int = 200, seed: int = 0,
                          features: Sequence[str] = FEATURES, analyses=None, progress=None) -> pd.DataFrame:
    """Gene-label permutation null (harness addition).

    Outcome permutation breaks every link, including confounded ones.
    Shuffling feature values *across genes* keeps the trial → target structure
    (number of targets, shared targets), so the MIN-aggregated trial feature
    of an uninformative gene property still carries whatever the aggregation
    itself induces (e.g. multi-target trials get lower minima). Reports the
    observed univariate OR, the null median / 2.5–97.5% ORs and
    ``p_gene_perm`` = (#|null − median| ≥ |obs − median| + 1)/(n + 1) on the
    log-OR scale, plus ``p_gene_perm_one_sided`` (null ≥ obs, in the
    direction of the observed effect).
    """
    rng = np.random.default_rng(seed)
    analyses = analyses or [a for a in st.AUTHORS_ANALYSES if a[2] != "stop_reason"]
    feats = inputs["features"].set_index("ensembl_id")
    genes = feats.index.to_numpy()
    pairs = inputs["mapping"][["nct_id", "targetId"]].drop_duplicates()
    pairs = pairs[pairs["nct_id"].isin(set(trials["nct_id"])) & pairs["targetId"].isin(set(genes))]
    order = trials["nct_id"].to_numpy()
    pos = pd.Series(np.arange(len(order)), index=order)
    tri = pos.reindex(pairs["nct_id"]).to_numpy()
    gidx = pd.Series(np.arange(len(genes)), index=genes).reindex(pairs["targetId"]).to_numpy()
    srt = np.argsort(tri, kind="stable")
    tri, gidx = tri[srt], gidx[srt]
    starts = np.r_[0, np.flatnonzero(np.diff(tri)) + 1]
    trial_of_start = tri[starts]
    frames = []
    for section, outcome, otype, value in analyses:
        fr = st.authors_outcome_frame(trials, otype, value)
        if fr is None:
            continue
        sub, y = fr
        frames.append((section, outcome, pos.reindex(sub["nct_id"]).to_numpy(), y.to_numpy(dtype=float)))
    obs = {}
    for f in features:
        for section, outcome, idx, y in frames:
            obs[(f, outcome)] = _fast_logit_coef(_z(trials[f].to_numpy()[idx]), y)
    null = {k: [] for k in obs}
    for it in range(n_iter):
        for f in features:
            vals = feats[f].to_numpy(dtype=float)[rng.permutation(len(genes))]
            pv = vals[gidx]
            pv = np.where(np.isnan(pv), np.inf, pv)
            mins = np.minimum.reduceat(pv, starts)
            tv = np.full(len(order), np.nan)
            tv[trial_of_start] = np.where(np.isinf(mins), np.nan, mins)
            for section, outcome, idx, y in frames:
                xv = tv[idx]
                ok = np.isfinite(xv)
                null[(f, outcome)].append(_fast_logit_coef(_z(xv[ok]), y[ok]) if ok.sum() > 10 else np.nan)
        if progress and (it + 1) % 25 == 0:
            progress(f"  gene permutation {it + 1}/{n_iter}")
    rows = []
    for (f, outcome), b in obs.items():
        z = np.asarray(null[(f, outcome)], dtype=float)
        z = z[np.isfinite(z)]
        med = float(np.median(z))
        p2 = (np.sum(np.abs(z - med) >= abs(b - med)) + 1) / (len(z) + 1)
        p1 = ((np.sum(z >= b) if b >= med else np.sum(z <= b)) + 1) / (len(z) + 1)
        rows.append(dict(check="gene_permutation_null", feature=f, outcome=outcome, observed_or=float(np.exp(b)),
                         null_median_or=float(np.exp(med)), null_q025=float(np.exp(np.quantile(z, .025))),
                         null_q975=float(np.exp(np.quantile(z, .975))), p_gene_perm=float(p2),
                         p_gene_perm_one_sided=float(p1), n_iter=int(len(z)),
                         survives=bool(p2 < 0.05)))
    return pd.DataFrame(rows)


def harness_checks(inputs: dict[str, Any], trials: pd.DataFrame, *, gene_perm: int = 200, seed: int = 0,
                   progress=None) -> pd.DataFrame:
    """Target-count adjustment (log number of targets, and log targets with
    features) of the univariate tau / bimodality models, plus the gene-label
    permutation null."""
    rows = []
    t = trials.copy()
    t["log_n_targets"] = np.log(t["n_targets_all"].clip(lower=1))
    for section, outcome, otype, value in st.AUTHORS_ANALYSES:
        if otype == "stop_reason":
            continue
        fr = st.authors_outcome_frame(t, otype, value)
        if fr is None:
            continue
        sub, y = fr
        d = sub[[*FEATURES, "log_n_targets"]].copy()
        d["_y"] = y.to_numpy(dtype=float)
        for f in FEATURES:
            un = st.univariate_logistic(d, f, "_y")
            ad = st.adjusted_logistic(d, f, "_y", ["log_n_targets"])
            rows.append(dict(check="adjusted_n_targets", feature=f, outcome=outcome, observed_or=un["odds_ratio"],
                             adjusted_or=ad["odds_ratio"], adjusted_ci_low=ad["ci_low"],
                             adjusted_ci_high=ad["ci_high"], adjusted_p=ad["p"], n=ad["n"],
                             survives=bool(ad["p"] < 0.05 and np.sign(np.log(ad["odds_ratio"])) ==
                                           np.sign(np.log(un["odds_ratio"])))))
    out = pd.DataFrame(rows)
    if gene_perm:
        out = pd.concat([out, gene_permutation_null(inputs, trials, n_iter=gene_perm, seed=seed,
                                                    progress=progress)], ignore_index=True)
    return out


# ---------------------------------------------------------------------------
# Reference tables and comparison
# ---------------------------------------------------------------------------

_REFERENCE_FILES = {
    "all_results": "figure2_virtualbiotech_all_results.csv",
    "expr_results": "figure2_virtualbiotech_expr_results.csv",
    "permutation": "figure2_permutation_results.csv",
    "mixed": "figure2_virtualbiotech_mixedeffects_results.csv",
    "binary_tau_target": "figure2_binary_tau_target_level_results.csv",
    "binary_bimodality_target": "figure2_binary_bimodality_target_level_results.csv",
    "binary_tau_ae": "figure2_binary_tau_ae_results.csv",
    "binary_bimodality_ae": "figure2_binary_bimodality_ae_results.csv",
    "binary_phase1to2": "figure2_binary_phase1to2_progression_results.csv",
    "genetic": "figure2_genetic_association_results.csv",
    "combined_multivariate": "figure2_combined_multivariate_results.csv",
    "combined_correlation": "figure2_combined_correlation_analysis.csv",
}


def load_reference_tables(root: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(root)
    res = root / "results" if (root / "results").exists() else root / "clinical_trials" / "results"
    out = {}
    for k, fn in _REFERENCE_FILES.items():
        if (res / fn).exists():
            out[k] = pd.read_csv(res / fn)
    if "mixed" in out:  # covariate rows belong to the preceding feature_z row's feature
        m = out["mixed"].copy()
        m["analysis_feature"] = m["feature"].where(m["term"] == "feature_z").ffill()
        out["mixed"] = m
    return out


def _n(row: pd.Series, n_cols: Sequence[str]) -> float:
    vals = [row.get(c) for c in n_cols if c in row and pd.notna(row.get(c))]
    return float(sum(vals)) if vals else float("nan")


def compare_tables(ours: pd.DataFrame, theirs: pd.DataFrame, *, table: str, keys: Sequence[str],
                   estimate: str | dict, ci: tuple[str, str] | None = ("CI_lower", "CI_upper"),
                   p: str | None = "p_value", n_cols: Sequence[str] = ("n_positive", "n_reference"),
                   log_scale: bool | str = False, tol: float = 1e-3, abs_tol: float = 5e-4,
                   engine_col: str | None = "engine", se_col: str | None = None,
                   se_tol: float = 0.05) -> pd.DataFrame:
    """Align two tables on ``keys`` and compare one estimate column.

    ``estimate`` is a column name or ``{regression_type value: column}``
    (e.g. OR for logistic rows, coefficient for beta rows). ``log_scale``
    (True or a column-value mapping key) compares log(estimate) for ratios.
    A row *matches* when n agrees and |Δ| ≤ ``abs_tol`` on the (log) scale or
    the relative difference ≤ ``tol`` — or, with ``se_col``, |Δ| ≤ ``se_tol`` ×
    the authors' standard error. Two missing estimates (e.g. an aliased
    covariate that R reports as NA) also match.
    """
    keys = list(keys)
    t = theirs.copy()
    o = ours.copy()
    for k in keys:
        t[k] = t[k].astype(str)
        o[k] = o[k].astype(str)
    t = t.drop_duplicates(keys)
    o = o.drop_duplicates(keys)
    m = t.merge(o, on=keys, how="left", suffixes=("_theirs", "_ours"), indicator=True)
    rows = []
    for _, r in m.iterrows():
        rt = r.get("regression_type_theirs", r.get("regression_type"))
        est = estimate if isinstance(estimate, str) else estimate.get(rt, estimate.get("*"))
        logs = log_scale if isinstance(log_scale, bool) else (est == log_scale)
        th = r.get(f"{est}_theirs", r.get(est))
        ou = r.get(f"{est}_ours", np.nan)
        th = float(th) if pd.notna(th) else np.nan
        ou = float(ou) if pd.notna(ou) else np.nan
        if logs and th > 0 and ou > 0:
            d_abs = abs(np.log(ou) - np.log(th))
        else:
            d_abs = abs(ou - th)
        rel = abs(ou - th) / abs(th) if th not in (0,) and np.isfinite(th) else np.nan
        n_th = _n(pd.Series({c: r.get(f"{c}_theirs", r.get(c)) for c in n_cols}), n_cols)
        n_ou = _n(pd.Series({c: r.get(f"{c}_ours") for c in n_cols}), n_cols)
        n_ok = (not np.isfinite(n_th)) or (np.isfinite(n_ou) and n_th == n_ou)
        both_na = r["_merge"] == "both" and not np.isfinite(ou) and not np.isfinite(th)
        found = r["_merge"] == "both" and np.isfinite(ou)
        se_ok = False
        if se_col is not None and found:
            se_th = r.get(f"{se_col}_theirs", r.get(se_col))
            se_ok = bool(pd.notna(se_th) and d_abs <= se_tol * float(se_th))
        row = dict(table=table, **{k: r[k] for k in keys}, estimate_kind=est, theirs=th, ours=ou,
                   abs_delta=d_abs if found else np.nan, rel_delta=rel if found else np.nan,
                   theirs_n=n_th, ours_n=n_ou, n_match=bool(n_ok and (found or both_na)),
                   match=bool(both_na or (found and n_ok and np.isfinite(th)
                                          and (d_abs <= abs_tol or rel <= tol or se_ok))))
        if ci:
            for side, c in zip(("ci_low", "ci_high"), ci):
                row[f"theirs_{side}"] = r.get(f"{c}_theirs", np.nan)
                row[f"ours_{side}"] = r.get(f"{c}_ours", np.nan)
        if p:
            row["theirs_p"] = r.get(f"{p}_theirs", np.nan)
            row["ours_p"] = r.get(f"{p}_ours", np.nan)
        if engine_col:
            row["engine"] = r.get(f"{engine_col}_ours", r.get(engine_col, "statsmodels"))
        if not found and not both_na:
            row["engine"] = "missing"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Headlines
# ---------------------------------------------------------------------------


def headline_numbers(univ: pd.DataFrame, binary: dict[str, Any], trials: pd.DataFrame,
                     features: pd.DataFrame) -> dict[str, float]:
    def or_(outcome, feat="tau_cell_type"):
        r = univ[(univ["outcome"] == outcome) & (univ["feature"] == feat) & (univ["regression_type"] == "logistic")]
        return float(r["OR"].iloc[0]) if len(r) else float("nan")

    tl = binary["target_tau"]
    p4 = tl[(tl["model"] == "unadjusted") & (tl["outcome"] == "Ever reached Phase IV")]
    pr = binary["phase1to2"].set_index("metric")
    ae = binary["ae_tau"]
    return dict(
        n_trials=int(len(trials)),
        or_primary_tau=or_("Primary Positive"), or_secondary_tau=or_("Secondary Positive"),
        or_phase1to2_tau=or_("Phase 2 Progression"),
        or_primary_bimodality=or_("Primary Positive", "bimodality_score"),
        or_phase1to2_bimodality=or_("Phase 2 Progression", "bimodality_score"),
        phase4_relative_increase=float(p4["fold"].iloc[0] - 1) if len(p4) else float("nan"),
        phase1to2_relative_increase=float(pr.loc["tau_cell_type", "fold"] - 1),
        ae_relative_difference=float((ae["fold"] - 1).mean()) if len(ae) else float("nan"),
        ae_relative_difference_pooled=float(ae["mean_hi"].sum() / ae["mean_lo"].sum() - 1) if len(ae) else np.nan,
        tau_threshold=float(binary["threshold"]),
        pearson_tau_bimodality=float(trials[["tau_cell_type", "bimodality_score"]].corr().iloc[0, 1]),
        pearson_tau_bimodality_gene_level=float(features[["tau_cell_type", "bimodality_score"]].corr().iloc[0, 1]),
    )


def _reference_headlines(ref: dict[str, pd.DataFrame]) -> dict[str, float]:
    out: dict[str, float] = {}
    a = ref.get("all_results")
    if a is not None:
        def g(outcome, feat="tau_cell_type"):
            r = a[(a["outcome"] == outcome) & (a["feature"] == feat) & (a["regression_type"] == "logistic")]
            return float(r["OR"].iloc[0]) if len(r) else float("nan")
        out.update(or_primary_tau=g("Primary Positive"), or_secondary_tau=g("Secondary Positive"),
                   or_phase1to2_tau=g("Phase 2 Progression"),
                   or_primary_bimodality=g("Primary Positive", "bimodality_score"),
                   or_phase1to2_bimodality=g("Phase 2 Progression", "bimodality_score"))
        r = a[(a["outcome"] == "Phase II+") & (a["feature"] == "tau_cell_type")]
        if len(r):
            out["n_trials"] = int(r["n_positive"].iloc[0] + r["n_reference"].iloc[0])
    tl = ref.get("binary_tau_target")
    if tl is not None:
        r = tl[(tl["model"] == "unadjusted") & (tl["outcome"] == "Ever reached Phase IV")]
        out["phase4_relative_increase"] = float(r["fold"].iloc[0] - 1)
        out["tau_threshold"] = float(tl["threshold"].iloc[0])
    p1 = ref.get("binary_phase1to2")
    if p1 is not None:
        out["phase1to2_relative_increase"] = float(p1.set_index("metric").loc["tau_cell_type", "fold"] - 1)
    ae = ref.get("binary_tau_ae")
    if ae is not None:
        out["ae_relative_difference"] = float((ae["fold"] - 1).mean())
        out["ae_relative_difference_pooled"] = float(ae["mean_hi"].sum() / ae["mean_lo"].sum() - 1)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class ReplicationReport:
    out_dir: Path
    comparison: pd.DataFrame
    headline: dict[str, dict[str, float]]
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    checks: pd.DataFrame | None = None
    summary_md: str = ""
    engines: dict[str, str] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def agreement(self) -> pd.DataFrame:
        c = self.comparison
        return (c.groupby("table")
                 .agg(rows=("match", "size"), matched=("match", "sum"), n_matched=("n_match", "sum"),
                      max_abs_delta=("abs_delta", "max"), median_abs_delta=("abs_delta", "median"))
                 .reset_index())


def _fmt(v, nd=4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    return f"{v:.{nd}g}" if abs(v) < 1e4 else f"{v:,.0f}"


def _summary_md(rep: ReplicationReport) -> str:
    lines = ["# Case 1 replication on the authors' Zenodo inputs", ""]
    lines += ["## Headline numbers", "", "| quantity | ours | authors' tables | paper text |", "|---|---|---|---|"]
    h = rep.headline
    labels = {
        "n_trials": "trials analysed (Phase II+ model n)",
        "or_primary_tau": "OR primary endpoint, tau (per SD)",
        "or_secondary_tau": "OR secondary endpoint, tau",
        "or_phase1to2_tau": "OR Phase I→II, tau",
        "or_primary_bimodality": "OR primary endpoint, bimodality",
        "or_phase1to2_bimodality": "OR Phase I→II, bimodality",
        "phase4_relative_increase": "specific targets: ever reach Phase IV (target level, fold − 1)",
        "phase1to2_relative_increase": "specific-target trials: Phase I→II (fold − 1)",
        "ae_relative_difference": "specific-target trials: serious AE rate (mean over organs of fold − 1)",
        "ae_relative_difference_pooled": "  same, pooled over organs (Σ mean_hi / Σ mean_lo − 1)",
        "tau_threshold": "k-means tau threshold",
        "pearson_tau_bimodality": "Pearson tau vs bimodality (trial level, MIN-aggregated)",
        "pearson_tau_bimodality_gene_level": "Pearson tau vs bimodality (gene level)",
    }
    for k, lab in labels.items():
        lines.append(f"| {lab} | {_fmt(h['ours'].get(k))} | {_fmt(h['authors'].get(k))} | "
                     f"{_fmt(h['paper'].get(k))} |")
    lines += ["", "## Agreement per table", "",
              "| table | rows | estimate matches | n matches | max abs Δ | median abs Δ | engine |",
              "|---|---|---|---|---|---|---|"]
    ag = rep.agreement()
    for _, r in ag.iterrows():
        eng = ", ".join(sorted(set(rep.comparison.loc[rep.comparison["table"] == r["table"], "engine"]
                                   .dropna().astype(str))))
        lines.append(f"| {r['table']} | {int(r['rows'])} | {int(r['matched'])} | {int(r['n_matched'])} | "
                     f"{_fmt(r['max_abs_delta'], 3)} | {_fmt(r['median_abs_delta'], 3)} | {eng} |")
    lines += ["", "Δ is on the log-OR scale for odds ratios and on the coefficient scale otherwise; a row "
              "matches when n agrees and |Δ| ≤ 5e-4 or the relative difference ≤ 1e-3 "
              "(mixed models: |Δ| ≤ 5e-3 or ≤ 0.05 × the authors' SE; aliased terms reported NA by R and "
              "dropped by us count as matches; permutation p: ±2/(n+1); permutation null SD: 10%).", ""]
    if rep.checks is not None and len(rep.checks):
        lines += ["## Harness confounding checks (real features)", ""]
        a = rep.checks[rep.checks["check"] == "adjusted_n_targets"]
        if len(a):
            lines += ["| feature | outcome | OR | OR adjusted for log #targets | p adj | survives |",
                      "|---|---|---|---|---|---|"]
            for _, r in a.iterrows():
                lines.append(f"| {r['feature']} | {r['outcome']} | {_fmt(r['observed_or'])} | "
                             f"{_fmt(r['adjusted_or'])} | {_fmt(r['adjusted_p'], 3)} | {r['survives']} |")
            lines.append("")
        g = rep.checks[rep.checks["check"] == "gene_permutation_null"]
        if len(g):
            lines += ["| feature | outcome | observed OR | gene-perm null median OR [2.5%, 97.5%] | "
                      "p (two-sided) | survives |", "|---|---|---|---|---|---|"]
            for _, r in g.iterrows():
                lines.append(f"| {r['feature']} | {r['outcome']} | {_fmt(r['observed_or'])} | "
                             f"{_fmt(r['null_median_or'])} [{_fmt(r['null_q025'])}, {_fmt(r['null_q975'])}] | "
                             f"{_fmt(r['p_gene_perm'], 3)} | {r['survives']} |")
            lines.append("")
    if rep.timings:
        lines += ["Timings (s): " + ", ".join(f"{k} {v:.0f}" for k, v in rep.timings.items()), ""]
    return "\n".join(lines)


def replicate_case1(zenodo_root: str | Path, out_dir: str | Path, *, n_perm: int = 1000,
                    n_perm_beta: int | None = None, gene_perm: int = 200, mixed: bool = True,
                    expr: bool = True, combined: bool = True, seed: int = 42,
                    progress: Callable[[str], None] | None = print) -> ReplicationReport:
    """Run our Case 1 implementation on the authors' inputs and compare every row.

    ``zenodo_root``: the extracted ``virtualbiotech_submission`` folder (see
    :func:`vbt.data.zenodo.zenodo_root`). ``n_perm`` / ``n_perm_beta``: outcome
    permutations (0 skips); ``gene_perm``: gene-label permutations for the
    harness check (0 skips); ``mixed``: fit the GLMMs (slow without R:
    ~10 s per logistic and ~1 min per beta model); ``expr``: the 2,604
    cell-type expression models; ``combined``: correlation and 2-/3-way models.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    t0 = time.time()
    inputs = load_inputs(zenodo_root)
    ref = load_reference_tables(inputs["root"])
    trials = build_trials(inputs)
    _log(progress, f"{len(trials):,} trials with tau and bimodality")
    tables: dict[str, pd.DataFrame] = {}
    comps = []

    def tick(name):
        nonlocal t0
        timings[name] = time.time() - t0
        t0 = time.time()

    tick("load")
    univ = run_univariate(trials, progress=progress)
    tables["all_results"] = univ
    if "all_results" in ref:
        comps.append(compare_tables(univ, ref["all_results"], table="univariate (all_results)",
                                    keys=["section", "outcome", "feature", "regression_type"],
                                    estimate={"logistic": "OR", "beta": "coefficient"}, log_scale="OR",
                                    n_cols=("n_positive", "n_reference")))
    tick("univariate")
    if expr:
        ex = run_expr(trials, progress=progress)
        tables["expr_results"] = ex
        if "expr_results" in ref:
            comps.append(compare_tables(ex, ref["expr_results"], table="expression (expr_results)",
                                        keys=["section", "outcome", "method", "feature"], estimate="OR",
                                        log_scale=True))
        tick("expr")
    binary = run_binary(inputs, seed=seed)
    for k in ("target_tau", "target_bimodality", "ae_tau", "ae_bimodality", "phase1to2"):
        tables[f"binary_{k}"] = binary[k]
    if "binary_tau_target" in ref:
        for ours_k, ref_k in (("target_tau", "binary_tau_target"), ("target_bimodality", "binary_bimodality_target")):
            comps.append(compare_tables(binary[ours_k], ref[ref_k], table=f"binary {ours_k}",
                                        keys=["outcome", "model"], estimate="OR", log_scale=True,
                                        n_cols=("n_total",), engine_col=None))
            comps.append(compare_tables(binary[ours_k], ref[ref_k], table=f"binary {ours_k} (fold)",
                                        keys=["outcome", "model"], estimate="fold", ci=None, p=None,
                                        n_cols=("n_total",), engine_col=None))
        for ours_k, ref_k in (("ae_tau", "binary_tau_ae"), ("ae_bimodality", "binary_bimodality_ae")):
            comps.append(compare_tables(binary[ours_k], ref[ref_k], table=f"binary {ours_k} (fold)",
                                        keys=["ae_column"], estimate="fold", ci=None, p="p_value_ttest",
                                        n_cols=("n_total",), engine_col=None))
        comps.append(compare_tables(binary["phase1to2"], ref["binary_phase1to2"], table="binary phase1to2",
                                    keys=["metric"], estimate="OR", log_scale=True, n_cols=("n_total",),
                                    engine_col=None))
    tick("binary")
    if "genetic" in inputs:
        gen = run_genetic(inputs)
        tables["genetic"] = gen
        if "genetic" in ref:
            comps.append(compare_tables(gen, ref["genetic"], table="genetic evidence",
                                        keys=["section", "outcome", "feature", "regression_type"],
                                        estimate={"logistic": "OR", "beta": "coefficient"}, log_scale="OR",
                                        n_cols=("n_positive", "n_reference")))
        tick("genetic")
        if combined:
            corr, mv = run_combined(inputs, trials)
            tables["combined_correlation"] = corr
            tables["combined_multivariate"] = mv
            if "combined_multivariate" in ref:
                rmv = ref["combined_multivariate"].copy()
                rmv["regression_type"] = np.where(rmv["section"] == "AE Organ System", "beta", "logistic")
                mv2 = mv.assign(regression_type=np.where(mv["section"] == "AE Organ System", "beta", "logistic"))
                comps.append(compare_tables(mv2, rmv, table="combined multivariate",
                                            keys=["section", "outcome", "model", "feature"],
                                            estimate={"logistic": "OR", "beta": "coefficient"}, log_scale="OR",
                                            n_cols=("n_positive", "n_reference", "n_trials"), engine_col=None))
            if "combined_correlation" in ref:
                comps.append(compare_tables(corr, ref["combined_correlation"], table="combined correlation",
                                            keys=["genetic_feature", "sc_feature"], estimate="spearman_r",
                                            ci=None, p="spearman_p", n_cols=("n_samples",), engine_col=None))
            tick("combined")
    if n_perm:
        perm = run_permutation(trials, n_perm, n_perm_beta=n_perm_beta, seed=seed, progress=progress)
        tables["permutation"] = perm
        if "permutation" in ref:
            rp = ref["permutation"]
            c = compare_tables(perm, rp, table="permutation (empirical p)",
                               keys=["section", "outcome", "feature", "regression_type"],
                               estimate="empirical_p_value", ci=None, p=None,
                               n_cols=("n_positive", "n_reference"), engine_col=None)
            n_eff = perm.set_index(["section", "outcome", "feature"])["n_permutations"]
            tol = 2.0 / (n_eff.min() + 1)
            c["match"] = c["n_match"] & (c["abs_delta"] <= tol + 1e-12)
            comps.append(c)
            comps.append(compare_tables(perm, rp, table="permutation (null SD)",
                                        keys=["section", "outcome", "feature", "regression_type"],
                                        estimate="null_std", ci=None, p=None, n_cols=("n_positive", "n_reference"),
                                        tol=0.1, abs_tol=0, engine_col=None))
        tick("permutation")
    if mixed and {"drug_types", "disease"} <= set(inputs):
        mx = run_mixed(inputs, trials, progress=progress)
        tables["mixed"] = mx
        if "mixed" in ref:
            comps.append(compare_tables(mx, ref["mixed"], table="mixed effects",
                                        keys=["section", "outcome", "analysis_feature", "term"],
                                        estimate="coefficient", ci=None, p="p_value",
                                        n_cols=("n_positive", "n_reference"), abs_tol=5e-3, tol=5e-3,
                                        se_col="se", se_tol=0.05))
        tick("mixed")
    checks = harness_checks(inputs, trials, gene_perm=gene_perm, seed=0, progress=progress)
    tick("harness_checks")

    ours_h = headline_numbers(univ, binary, trials, inputs["features"])
    theirs_h = _reference_headlines(ref)
    theirs_h.setdefault("pearson_tau_bimodality_gene_level", ours_h["pearson_tau_bimodality_gene_level"])
    comparison = pd.concat(comps, ignore_index=True) if comps else pd.DataFrame()
    rep = ReplicationReport(out_dir=out_dir, comparison=comparison,
                            headline=dict(ours=ours_h, authors=theirs_h, paper=dict(PAPER_HEADLINES)),
                            tables=tables, checks=checks, timings=timings,
                            engines=dict(logistic="statsmodels GLM", beta=str(univ.loc[univ["regression_type"] ==
                                                                                       "beta", "engine"].iloc[0])
                                         if (univ["regression_type"] == "beta").any() else "-",
                                         mixed="laplace_py" if mixed else "not run"))
    rep.summary_md = _summary_md(rep)
    comparison.to_csv(out_dir / "case1_replication_comparison.csv", index=False)
    (out_dir / "case1_replication_summary.md").write_text(rep.summary_md)
    (out_dir / "case1_replication_headline.json").write_text(json.dumps(rep.headline, indent=2, default=float))
    checks.to_csv(out_dir / "case1_harness_checks.csv", index=False)
    for k, v in tables.items():
        v.to_csv(out_dir / f"ours_{k}.csv", index=False)
    return rep
