"""Association of single-cell target features with clinical-trial outcomes.

Implements the paper's Methods section *"Statistical analysis of single-cell
atlas features with trial outcomes"* (Zhang et al., "The Virtual Biotech",
Science 2026), plus the outcome derivation from the released trial labels:

* :func:`build_outcomes` — trial-level binary outcomes (endpoint success,
  phase progression, early stopping, stop-for-safety/negative, reaching
  Phase IV) and serious-AE percentages from the labels CSV + mapping parquet.
* :func:`trial_level_features` — collapse target-level features to trials
  (minimum across a drug's targets: the least-specific target drives risk).
* :func:`genetic_evidence_flags` — trial has human genetic support for any of
  its (target, disease) pairs.
* :func:`univariate_logistic`, :func:`adjusted_logistic` — binomial GLMs
  with the z-scored feature (odds ratio per 1 SD).
* :func:`beta_regression` — beta regression for AE percentages with the
  Smithson–Verkuilen squeeze.
* :func:`permutation_test` — outcome-permutation null.
* :func:`mixed_effects_logistic`, :func:`mixed_effects_beta` — crossed random
  intercepts for modality and therapeutic area (lme4 / glmmTMB via rpy2 when
  available; statsmodels fallbacks otherwise — the engine used is reported).
* :func:`benjamini_hochberg` — FDR correction.
* :func:`binarize_tau`, :func:`relative_likelihood`,
  :func:`relative_ae_difference` — the "cell-type-specific vs broadly
  expressed" contrasts (paper: τ threshold ≈ 0.69; specific-target trials
  48% more likely to reach Phase IV, 32% lower serious-AE rates).
* :func:`run_association_suite` — tidy table across features × outcomes
  (covariate-adjusted logistic *and* beta models when covariates are given;
  :func:`refdr` recomputes BH within a filtered family).
* :func:`genetic_evidence_replication` — Razuvayevskaya-style replication
  (fig. S2): outcomes and AE rates on the genetic-evidence flag.

Every model row records the ``engine`` that fitted it (statsmodels, or R
``betareg``/``lme4``/``glmmTMB`` via rpy2 when requested and installed).

``rpy2`` is optional and imported lazily; nothing here crashes without it.
"""

from __future__ import annotations

import re
import warnings
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "trial_level_features",
    "genetic_evidence_flags",
    "univariate_logistic",
    "adjusted_logistic",
    "beta_regression",
    "permutation_test",
    "mixed_effects_logistic",
    "mixed_effects_beta",
    "benjamini_hochberg",
    "binarize_tau",
    "relative_likelihood",
    "relative_ae_difference",
    "run_association_suite",
    "build_outcomes",
    "phase_to_numeric",
    "smithson_verkuilen",
    "refdr",
    "genetic_evidence_replication",
    "combine_therapeutic_areas",
]

_Z = 1.959963984540054  # standard-normal 97.5% quantile


# ---------------------------------------------------------------------------
# Feature aggregation to trials
# ---------------------------------------------------------------------------


def trial_level_features(
    mapping: pd.DataFrame,
    gene_features: pd.DataFrame,
    feature_cols: Sequence[str] = ("tau", "bimodality"),
    target_col: str = "targetId",
) -> pd.DataFrame:
    """Collapse target-level features to one row per trial.

    Paper Methods ("Statistical analysis of single-cell atlas features with
    trial outcomes"): for drugs with multiple targets the **minimum** value
    across targets is used for each feature, on the rationale that the least
    cell-type-specific target drives the safety liability.

    ``mapping`` needs ``nct_id`` and ``targetId``; ``gene_features`` is indexed
    by target ID (e.g. the output of
    :func:`vbt.case_studies.trial_outcomes.features.compute_gene_features`
    with ``gene_id_column="ensembl_id"``), or has a ``targetId``/``gene``
    column.  Returns a DataFrame indexed by ``nct_id`` with the feature
    columns plus ``n_targets`` and ``n_targets_with_features``.  Trials with no
    featurized target get NaN features.
    """
    feats = gene_features
    if target_col in feats.columns:
        feats = feats.set_index(target_col)
    elif "gene" in feats.columns and feats.index.name != "gene":
        feats = feats.set_index("gene")
    feature_cols = list(feature_cols)
    feats = feats[feature_cols]
    feats = feats[~feats.index.duplicated(keep="first")]

    pairs = mapping[["nct_id", target_col]].dropna().drop_duplicates()
    merged = pairs.join(feats, on=target_col)
    has = merged[feature_cols].notna().any(axis=1)
    g = merged.groupby("nct_id", sort=True)
    out = g[feature_cols].min()  # skipna: NaN only if all targets missing
    out["n_targets"] = g[target_col].nunique()
    out["n_targets_with_features"] = merged.assign(_h=has).groupby("nct_id")["_h"].sum()
    # Trials in the mapping with no target at all.
    all_ids = pd.Index(mapping["nct_id"].dropna().unique(), name="nct_id")
    out = out.reindex(all_ids.sort_values())
    out["n_targets"] = out["n_targets"].fillna(0).astype(int)
    out["n_targets_with_features"] = out["n_targets_with_features"].fillna(0).astype(int)
    out.index.name = "nct_id"
    return out


def genetic_evidence_flags(
    mapping: pd.DataFrame,
    genetic_pairs: set[tuple[str, str]],
    target_col: str = "targetId",
    disease_col: str = "diseaseId",
) -> pd.Series:
    """Per-trial flag: any (targetId, diseaseId) pair has genetic support.

    Used as the genetic-evidence covariate in the paper's bivariate
    adjustment.  ``genetic_pairs`` is a set of (targetId, diseaseId) tuples
    (e.g. Open Targets genetic-association evidence).  Returns a boolean
    Series indexed by ``nct_id``.
    """
    pairs = set(genetic_pairs)
    sub = mapping[["nct_id", target_col, disease_col]]
    hit = [
        (t, d) in pairs for t, d in zip(sub[target_col].to_numpy(), sub[disease_col].to_numpy())
    ]
    s = pd.Series(hit, index=sub["nct_id"].to_numpy()).groupby(level=0).any()
    s.index.name = "nct_id"
    s.name = "genetic_evidence"
    return s.astype(bool)


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------


def _zscore(x: pd.Series) -> pd.Series:
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        raise ValueError("feature has zero variance; cannot z-score")
    return (x - x.mean()) / sd


def _nan_result(n: int, error: str, **extra) -> dict:
    d = dict(odds_ratio=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, n=int(n),
             beta=np.nan, se=np.nan, error=error)
    d.update(extra)
    return d


def _design(df: pd.DataFrame, covariates: Sequence[str]) -> pd.DataFrame:
    """Numeric design block for covariates (categoricals → drop-first dummies)."""
    parts = []
    for c in covariates:
        col = df[c]
        if pd.api.types.is_bool_dtype(col):
            parts.append(col.astype(float).rename(c))
        elif pd.api.types.is_numeric_dtype(col):
            parts.append(col.astype(float).rename(c))
        else:
            d = pd.get_dummies(col.astype("string"), prefix=c, drop_first=True, dtype=float)
            parts.append(d)
    if not parts:
        return pd.DataFrame(index=df.index)
    return pd.concat(parts, axis=1)


def _binary(y: pd.Series) -> pd.Series:
    y = pd.to_numeric(y.astype("float64") if pd.api.types.is_bool_dtype(y) else y,
                      errors="coerce")
    bad = y.dropna()[~y.dropna().isin([0, 1])]
    if len(bad):
        raise ValueError("binary outcome must be coded 0/1")
    return y


def _fit_logit(y: np.ndarray, X: pd.DataFrame):
    import statsmodels.api as sm

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sm.GLM(y, X, family=sm.families.Binomial()).fit()


def adjusted_logistic(
    df: pd.DataFrame,
    feature: str,
    outcome: str,
    covariates: Sequence[str] = (),
) -> dict:
    """Logistic regression of a binary outcome on a z-scored feature + covariates.

    Paper Methods: binomial GLM (logit link); the feature is z-scored so the
    odds ratio is per 1 SD.  Covariates (e.g. the genetic-evidence flag for
    the bivariate adjustment, or phase/start year) enter unscaled; non-numeric
    covariates are expanded to drop-first dummies.  Rows with any NaN in the
    used columns are dropped before z-scoring.

    Returns a dict with ``odds_ratio``, ``ci_low``, ``ci_high`` (95% Wald),
    ``p``, ``n``, ``beta``, ``se`` for the feature (and ``error`` when the fit
    failed, e.g. zero variance or a single outcome class).
    """
    import statsmodels.api as sm

    covariates = list(covariates)
    cols = [feature, outcome, *covariates]
    d = df[cols].dropna()
    n = len(d)
    try:
        y = _binary(d[outcome]).to_numpy(dtype=float)
        if n < 3 or len(np.unique(y)) < 2:
            return _nan_result(n, "insufficient data or single outcome class")
        X = pd.concat([_zscore(d[feature].astype(float)).rename(feature),
                       _design(d, covariates)], axis=1)
        X = sm.add_constant(X, has_constant="add")
        res = _fit_logit(y, X)
        beta = float(res.params[feature])
        se = float(res.bse[feature])
        p = float(res.pvalues[feature])
    except Exception as exc:  # noqa: BLE001 - report instead of crashing a suite
        return _nan_result(n, f"{type(exc).__name__}: {exc}")
    return dict(engine="statsmodels",
        odds_ratio=float(np.exp(beta)),
        ci_low=float(np.exp(beta - _Z * se)),
        ci_high=float(np.exp(beta + _Z * se)),
        p=p, n=int(n), beta=beta, se=se,
    )


def univariate_logistic(df: pd.DataFrame, feature: str, outcome: str) -> dict:
    """Univariate logistic regression (binomial GLM, logit) on the z-scored feature.

    Paper Methods, "Statistical analysis of single-cell atlas features with
    trial outcomes": odds ratio per 1 SD increase of the feature with 95%
    Wald CI and p-value.  See :func:`adjusted_logistic` for the return keys.
    """
    return adjusted_logistic(df, feature, outcome, covariates=())


def smithson_verkuilen(y: np.ndarray, n: int | None = None) -> np.ndarray:
    """Squeeze proportions in [0, 1] into (0, 1): y' = (y(n − 1) + 0.5)/n.

    Smithson & Verkuilen (2006); ``n`` defaults to the sample size.
    """
    y = np.asarray(y, dtype=float)
    n = len(y) if n is None else n
    return (y * (n - 1) + 0.5) / n


def beta_regression(
    df: pd.DataFrame,
    feature: str,
    outcome_pct: str,
    covariates: Sequence[str] = (),
    *,
    engine: str = "statsmodels",
    standardize: bool = True,
) -> dict:
    """Beta regression of an AE percentage on the z-scored feature.

    Paper Methods: serious-AE rates (percentages 0–100) are converted to
    proportions, squeezed into (0, 1) with the Smithson–Verkuilen transform
    y' = (y(n − 1) + 0.5)/n, and modelled with a logit-link beta regression
    (``statsmodels.othermod.betareg.BetaModel``).  Returns ``coef`` (log-odds
    change of the mean proportion per 1 SD), ``ci_low``/``ci_high`` (95%
    Wald), ``p``, ``n``, ``se``, ``exp_coef`` (odds-ratio scale) and
    ``engine``.

    ``engine``: ``"statsmodels"`` (default), ``"betareg"`` (R ``betareg`` via
    rpy2, the paper's engine) or ``"auto"`` (betareg when rpy2 and the R
    package are installed, else statsmodels). ``standardize=False`` keeps the
    feature on its own scale (e.g. a 0/1 flag: ``exp_coef`` is then the odds
    ratio of the mean proportion between the two groups).
    """
    if engine not in ("statsmodels", "betareg", "auto"):
        raise ValueError("engine must be 'statsmodels', 'betareg' or 'auto'")
    covariates = list(covariates)
    d = df[[feature, outcome_pct, *covariates]].dropna()
    n = len(d)
    base = dict(coef=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, n=int(n),
                se=np.nan, exp_coef=np.nan, engine=engine)
    use_r = engine == "betareg" or (engine == "auto" and _rpy2_available(["betareg"]))
    if engine == "betareg" and not _rpy2_available(["betareg"]):
        base["error"] = "rpy2 or the R package betareg is not installed (pip install 'vbt-harness[r]')"
        return base
    if use_r:
        try:
            dd = d.copy()
            dd[feature] = _zscore(dd[feature].astype(float)) if standardize else dd[feature].astype(float)
            dd[outcome_pct] = smithson_verkuilen(np.clip(dd[outcome_pct].astype(float).to_numpy() / 100.0, 0, 1), n)
            safe, _rhs, rhs_r, _rand = _prep_mixed_fixed(dd, feature, outcome_pct, covariates)
            import rpy2.robjects as ro  # type: ignore

            ro.r("suppressPackageStartupMessages(library(betareg))")
            beta, se, p = _r_fit(safe, "y_out ~ " + " + ".join(rhs_r), "betareg({formula}, data=vbt_df)",
                                 "summary(vbt_fit)$coefficients$mean")
            base.update(coef=beta, se=se, p=p, ci_low=beta - _Z * se, ci_high=beta + _Z * se,
                        exp_coef=float(np.exp(beta)), engine="R:betareg")
            return base
        except Exception as exc:  # noqa: BLE001
            if engine == "betareg":
                base["error"] = f"betareg failed: {type(exc).__name__}: {exc}"
                return base
            base["note"] = f"betareg failed, statsmodels used: {exc}"
    import statsmodels.api as sm
    from statsmodels.othermod.betareg import BetaModel

    base["engine"] = "statsmodels"
    try:
        if n < 5:
            raise ValueError("insufficient data")
        y = np.clip(d[outcome_pct].astype(float).to_numpy() / 100.0, 0.0, 1.0)
        y = smithson_verkuilen(y, n)
        xf = _zscore(d[feature].astype(float)) if standardize else d[feature].astype(float)
        X = pd.concat([xf.rename(feature), _design(d, covariates)], axis=1)
        X = sm.add_constant(X, has_constant="add")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = BetaModel(y, X).fit(disp=0)
        names = list(res.model.exog_names)
        i = names.index(feature)
        coef = float(np.asarray(res.params)[i])
        se = float(np.asarray(res.bse)[i])
        p = float(np.asarray(res.pvalues)[i])
    except Exception as exc:  # noqa: BLE001
        base["error"] = f"{type(exc).__name__}: {exc}"
        return base
    base.update(coef=coef, se=se, p=p, ci_low=coef - _Z * se, ci_high=coef + _Z * se,
                exp_coef=float(np.exp(coef)))
    return base


def permutation_test(
    df: pd.DataFrame,
    feature: str,
    outcome: str,
    kind: str = "logistic",
    n_perm: int = 1000,
    seed: int = 0,
) -> dict:
    """Outcome-permutation test for the feature coefficient.

    Paper Methods: the outcome is shuffled across trials (complete cases),
    the model refitted, and the observed coefficient compared to the null
    distribution.  ``kind`` is ``"logistic"`` (GLM coefficient) or ``"beta"``
    (beta-regression coefficient; ``outcome`` then is a percentage).

    Two p-values are returned:

    * ``p_perm`` = (#{|null| ≥ |obs|} + 1) / (n_perm + 1) — the standard
      Phipson & Smyth (2010) estimator; it counts the observed statistic as
      one permutation so it is never exactly 0 and is a valid (slightly
      conservative) p-value.  This is the recommended value.
    * ``p_perm_raw`` = #{|null| ≥ |obs|} / n_perm — the plain empirical
      proportion (as reported in the paper), which can be 0.

    Also returns ``observed``, ``n_perm`` (successful refits), ``n`` and
    ``null`` (the array of null coefficients).
    """
    if kind not in ("logistic", "beta"):
        raise ValueError("kind must be 'logistic' or 'beta'")
    d = df[[feature, outcome]].dropna().reset_index(drop=True)
    fit = (lambda frame: univariate_logistic(frame, feature, outcome)["beta"]) \
        if kind == "logistic" else \
        (lambda frame: beta_regression(frame, feature, outcome)["coef"])
    obs = fit(d)
    rng = np.random.default_rng(seed)
    null = np.full(n_perm, np.nan)
    if np.isfinite(obs):
        y = d[outcome].to_numpy()
        for i in range(n_perm):
            perm = d.copy()
            perm[outcome] = rng.permutation(y)
            null[i] = fit(perm)
    ok = null[np.isfinite(null)]
    m = len(ok)
    if not np.isfinite(obs) or m == 0:
        return dict(observed=obs, p_perm=np.nan, p_perm_raw=np.nan, n_perm=m, n=len(d),
                    null=null)
    count = int(np.sum(np.abs(ok) >= np.abs(obs)))
    return dict(
        observed=float(obs),
        p_perm=(count + 1) / (m + 1),
        p_perm_raw=count / m,
        n_perm=m,
        n=len(d),
        null=null,
    )


# ---------------------------------------------------------------------------
# Mixed-effects models
# ---------------------------------------------------------------------------


def _rpy2_available(packages: Sequence[str]) -> bool:
    try:
        from rpy2.robjects.packages import isinstalled  # type: ignore
    except Exception:  # noqa: BLE001 - ImportError or R init failure
        return False
    try:
        return all(isinstalled(p) for p in packages)
    except Exception:  # noqa: BLE001
        return False


def _safe_frame(d: pd.DataFrame, feature: str, outcome: str,
                fixed: Sequence[str], random: Sequence[str]):
    """Rename columns to formula-safe names; return frame and name map."""
    names = {feature: "x_feat", outcome: "y_out"}
    for i, c in enumerate(fixed):
        names[c] = f"f{i}_" + re.sub(r"\W", "_", str(c))
    for i, c in enumerate(random):
        names[c] = f"g{i}_" + re.sub(r"\W", "_", str(c))
    out = d.rename(columns=names)
    return out, names


def _prep_mixed(df, feature, outcome, fixed, random, outcome_transform=None):
    fixed = [c for c in (fixed or []) if c in df.columns]
    random = [c for c in (random or []) if c in df.columns]
    d = df[[feature, outcome, *fixed, *random]].dropna().copy()
    d[feature] = _zscore(d[feature].astype(float))
    if outcome_transform is not None:
        d[outcome] = outcome_transform(d[outcome])
    fixed_terms = []
    for c in fixed:
        if pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c]):
            sd = d[c].astype(float).std(ddof=1)
            # Centre/scale numeric covariates (helps convergence; does not
            # change the feature coefficient).
            d[c] = (d[c].astype(float) - d[c].astype(float).mean()) / (sd if sd > 0 else 1.0)
            fixed_terms.append((c, "num"))
        else:
            d[c] = d[c].astype(str)
            fixed_terms.append((c, "cat"))
    for c in random:
        d[c] = d[c].astype(str)
    safe, names = _safe_frame(d, feature, outcome, fixed, random)
    rhs = ["x_feat"] + [
        names[c] if kind == "num" else f"C({names[c]})" for c, kind in fixed_terms
    ]
    rhs_r = ["x_feat"] + [
        names[c] if kind == "num" else f"factor({names[c]})" for c, kind in fixed_terms
    ]
    # Drop constant fixed covariates / single-level factors.
    rhs = [t for t, (c, _k) in zip(rhs[1:], fixed_terms) if safe[names[c]].nunique() > 1]
    rhs_r = [t for t, (c, _k) in zip(rhs_r[1:], fixed_terms) if safe[names[c]].nunique() > 1]
    rand_safe = [names[c] for c in random if safe[names[c]].nunique() > 1]
    return safe, ["x_feat", *rhs], ["x_feat", *rhs_r], rand_safe


def _prep_mixed_fixed(d: pd.DataFrame, feature: str, outcome: str, covariates: Sequence[str]):
    """Formula-safe frame for a fixed-effects-only R model (feature already transformed)."""
    covariates = list(covariates)
    dd = d[[feature, outcome, *covariates]].copy()
    terms = []
    for c in covariates:
        if pd.api.types.is_numeric_dtype(dd[c]) or pd.api.types.is_bool_dtype(dd[c]):
            dd[c] = dd[c].astype(float)
            terms.append((c, "num"))
        else:
            dd[c] = dd[c].astype(str)
            terms.append((c, "cat"))
    safe, names = _safe_frame(dd, feature, outcome, covariates, [])
    rhs_r = ["x_feat"] + [names[c] if k == "num" else f"factor({names[c]})" for c, k in terms
                          if safe[names[c]].nunique() > 1]
    return safe, rhs_r, rhs_r, []


def _r_fit(safe: pd.DataFrame, formula: str, call: str, coef_expr: str) -> tuple[float, float, float]:
    import rpy2.robjects as ro  # type: ignore
    from rpy2.robjects import pandas2ri  # type: ignore
    from rpy2.robjects.conversion import localconverter  # type: ignore

    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["vbt_df"] = ro.conversion.py2rpy(safe.reset_index(drop=True))
    ro.r(f"vbt_fit <- {call.format(formula=formula)}")
    est = ro.r(f"{coef_expr}['x_feat', ]")
    beta, se = float(est[0]), float(est[1])
    p = float(est[3]) if len(est) > 3 else float("nan")
    return beta, se, p


def mixed_effects_logistic(
    df: pd.DataFrame,
    feature: str,
    outcome: str,
    fixed: Sequence[str] = ("phase", "start_year"),
    random: Sequence[str] = ("modality", "therapeutic_area"),
    engine: str = "auto",
) -> dict:
    """Mixed-effects logistic regression with crossed random intercepts.

    Paper Methods (sensitivity analysis): ``outcome ~ z(feature) + phase +
    start_year + (1|modality) + (1|therapeutic_area)``, binomial/logit.

    Engines:

    * ``"lme4"`` — ``lme4::glmer`` through ``rpy2`` (the paper's model).
    * ``"statsmodels"`` — ``BinomialBayesMixedGLM`` fitted by variational
      Bayes (``fit_vb``) with one variance component per grouping factor;
      the feature's posterior mean/SD are reported as beta/se with a normal
      approximation for the CI and p-value.
    * ``"auto"`` (default) — lme4 if rpy2 and lme4 are installed and the fit
      succeeds, else the statsmodels fallback.  Never raises for a missing
      rpy2.

    Fixed covariates that are numeric are centred and scaled (does not change
    the feature coefficient); non-numeric ones are treated as factors.
    Grouping/covariate columns absent from ``df`` are skipped.  Returns
    ``odds_ratio``, ``ci_low``, ``ci_high``, ``p``, ``n``, ``beta``, ``se``
    and ``engine``.
    """
    if engine not in ("auto", "lme4", "statsmodels"):
        raise ValueError("engine must be 'auto', 'lme4' or 'statsmodels'")
    try:
        safe, rhs, rhs_r, rand = _prep_mixed(
            df, feature, outcome, fixed, random,
            outcome_transform=lambda s: _binary(s).astype(float),
        )
    except Exception as exc:  # noqa: BLE001
        return {**_nan_result(0, f"{type(exc).__name__}: {exc}"), "engine": "none"}
    n = len(safe)
    if n < 5 or safe["y_out"].nunique() < 2:
        return {**_nan_result(n, "insufficient data or single outcome class"), "engine": "none"}

    errors = []
    if engine in ("auto", "lme4") and rand:
        if _rpy2_available(["lme4"]):
            try:
                ro_formula = "y_out ~ " + " + ".join(rhs_r) + "".join(
                    f" + (1|{g})" for g in rand)
                import rpy2.robjects as ro  # type: ignore

                ro.r("suppressPackageStartupMessages(library(lme4))")
                beta, se, p = _r_fit(
                    safe, ro_formula,
                    "glmer({formula}, data=vbt_df, family=binomial)",
                    "summary(vbt_fit)$coefficients",
                )
                return dict(odds_ratio=float(np.exp(beta)), ci_low=float(np.exp(beta - _Z * se)),
                            ci_high=float(np.exp(beta + _Z * se)), p=p, n=n, beta=beta,
                            se=se, engine="lme4")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"lme4 failed: {exc}")
        else:
            errors.append("rpy2/lme4 not available")
        if engine == "lme4":
            return {**_nan_result(n, "; ".join(errors)), "engine": "lme4"}

    from scipy import stats as sstats

    formula = "y_out ~ " + " + ".join(rhs)
    try:
        if rand:
            from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

            vc = {g: f"0 + C({g})" for g in rand}
            model = BinomialBayesMixedGLM.from_formula(formula, vc, safe)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = model.fit_vb()
            names = list(model.exog_names)
            i = names.index("x_feat")
            beta = float(res.fe_mean[i])
            se = float(res.fe_sd[i])
            eng = "statsmodels_bayes_mixed_glm_vb"
        else:
            import statsmodels.formula.api as smf
            import statsmodels.api as sm

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = smf.glm(formula, safe, family=sm.families.Binomial()).fit()
            beta, se = float(res.params["x_feat"]), float(res.bse["x_feat"])
            eng = "statsmodels_glm_no_random_effects"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"statsmodels failed: {type(exc).__name__}: {exc}")
        return {**_nan_result(n, "; ".join(errors)), "engine": "none"}
    p = float(2 * sstats.norm.sf(abs(beta / se))) if se > 0 else float("nan")
    out = dict(odds_ratio=float(np.exp(beta)), ci_low=float(np.exp(beta - _Z * se)),
               ci_high=float(np.exp(beta + _Z * se)), p=p, n=n, beta=beta, se=se, engine=eng)
    if errors:
        out["note"] = "; ".join(errors)
    return out


def mixed_effects_beta(
    df: pd.DataFrame,
    feature: str,
    outcome_pct: str,
    fixed: Sequence[str] = ("phase", "start_year"),
    random: Sequence[str] = ("modality", "therapeutic_area"),
    engine: str = "auto",
) -> dict:
    """Mixed-effects beta regression for AE percentages.

    Paper Methods (sensitivity analysis): ``AE% ~ z(feature) + phase +
    start_year + (1|modality) + (1|therapeutic_area)`` with a beta family
    (Smithson–Verkuilen-squeezed proportions).

    Engines: ``"glmmTMB"`` (``glmmTMB(..., family=beta_family())`` via rpy2),
    ``"statsmodels"`` (fallback: ``BetaModel`` with the grouping factors as
    *fixed-effect* drop-first dummies — a fixed-effects approximation to the
    random intercepts), or ``"auto"`` (glmmTMB if available, else fallback).
    Returns ``coef``, ``ci_low``, ``ci_high``, ``p``, ``n``, ``se``,
    ``engine``.
    """
    if engine not in ("auto", "glmmTMB", "statsmodels"):
        raise ValueError("engine must be 'auto', 'glmmTMB' or 'statsmodels'")
    fixed = [c for c in (fixed or []) if c in df.columns]
    random = [c for c in (random or []) if c in df.columns]
    d = df[[feature, outcome_pct, *fixed, *random]].dropna()
    n = len(d)
    errors = []
    if engine in ("auto", "glmmTMB") and random:
        if _rpy2_available(["glmmTMB"]):
            try:
                sv = lambda s: pd.Series(  # noqa: E731
                    smithson_verkuilen(np.clip(s.astype(float) / 100.0, 0, 1), len(s)),
                    index=s.index)
                safe, _rhs, rhs_r, rand = _prep_mixed(
                    d, feature, outcome_pct, fixed, random, outcome_transform=sv)
                import rpy2.robjects as ro  # type: ignore

                ro.r("suppressPackageStartupMessages(library(glmmTMB))")
                formula = "y_out ~ " + " + ".join(rhs_r) + "".join(f" + (1|{g})" for g in rand)
                beta, se, p = _r_fit(
                    safe, formula,
                    "glmmTMB({formula}, data=vbt_df, family=beta_family())",
                    "summary(vbt_fit)$coefficients$cond",
                )
                return dict(coef=beta, ci_low=beta - _Z * se, ci_high=beta + _Z * se, p=p,
                            n=len(safe), se=se, exp_coef=float(np.exp(beta)), engine="glmmTMB")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"glmmTMB failed: {exc}")
        else:
            errors.append("rpy2/glmmTMB not available")
        if engine == "glmmTMB":
            return dict(coef=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, n=n, se=np.nan,
                        exp_coef=np.nan, engine="glmmTMB", error="; ".join(errors))

    d2 = d.copy()
    for c in fixed:
        if pd.api.types.is_numeric_dtype(d2[c]) and not pd.api.types.is_bool_dtype(d2[c]):
            sd = d2[c].astype(float).std(ddof=1)
            d2[c] = (d2[c].astype(float) - d2[c].astype(float).mean()) / (sd if sd > 0 else 1.0)
        else:
            d2[c] = d2[c].astype(str)
    for c in random:
        d2[c] = d2[c].astype(str)
    covs = [c for c in [*fixed, *random] if d2[c].nunique() > 1]
    res = beta_regression(d2, feature, outcome_pct, covariates=covs)
    res["engine"] = "statsmodels_betareg_fixed_dummies"
    if errors:
        res["note"] = "; ".join(errors)
    return res


# ---------------------------------------------------------------------------
# Multiple testing, binarization and effect summaries
# ---------------------------------------------------------------------------


def benjamini_hochberg(pvals: Iterable[float]) -> np.ndarray:
    """Benjamini–Hochberg FDR-adjusted p-values (wraps ``multipletests``).

    NaN p-values are ignored (left NaN) and do not count toward the number of
    tests.
    """
    from statsmodels.stats.multitest import multipletests

    p = np.asarray(list(pvals), dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


def binarize_tau(values, seed: int = 0) -> tuple[float, np.ndarray]:
    """Split trial-level τ into "cell-type-specific" vs "broadly expressed".

    Paper Methods: k-means (k = 2) on the trial-level τ distribution; the
    threshold is the midpoint of the two cluster centres (the paper obtained
    τ ≈ 0.69).  Returns ``(threshold, labels)`` where ``labels`` is a float
    array aligned with ``values``: 1.0 = specific (τ ≥ threshold), 0.0 =
    broad, NaN where the input is NaN.
    """
    from sklearn.cluster import KMeans

    v = np.asarray(values, dtype=float).ravel()
    ok = np.isfinite(v)
    labels = np.full(v.shape, np.nan)
    if np.unique(v[ok]).size < 2:
        raise ValueError("need at least two distinct non-NaN values to binarize")
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(v[ok].reshape(-1, 1))
    centers = np.sort(km.cluster_centers_.ravel())
    threshold = float(centers.mean())
    labels[ok] = (v[ok] >= threshold).astype(float)
    return threshold, labels


def relative_likelihood(df: pd.DataFrame, binary_col: str, outcome: str) -> dict:
    """Relative likelihood of an outcome for specific vs broad targets.

    Paper Results/Methods: e.g. trials with cell-type-specific targets were
    "48% more likely to reach Phase IV" = P(outcome | specific) /
    P(outcome | broad) − 1.  ``binary_col`` is 1 for specific, 0 for broad
    (see :func:`binarize_tau`); ``outcome`` is 0/1.  Returns the two
    proportions, counts, ``relative_increase`` and a Fisher exact p-value.
    """
    from scipy.stats import fisher_exact

    d = df[[binary_col, outcome]].dropna()
    g = d[binary_col].astype(float)
    y = _binary(d[outcome]).astype(float)
    s, b = y[g == 1], y[g == 0]
    p1 = s.mean() if len(s) else np.nan
    p0 = b.mean() if len(b) else np.nan
    table = [[int(s.sum()), int(len(s) - s.sum())], [int(b.sum()), int(len(b) - b.sum())]]
    try:
        _, pf = fisher_exact(table)
    except Exception:  # noqa: BLE001
        pf = np.nan
    rel = p1 / p0 - 1 if (np.isfinite(p0) and p0 > 0) else np.nan
    return dict(
        p_specific=float(p1), p_broad=float(p0), relative_increase=float(rel),
        n_specific=int(len(s)), n_broad=int(len(b)),
        events_specific=int(s.sum()), events_broad=int(b.sum()),
        fisher_p=float(pf),
    )


def relative_ae_difference(df: pd.DataFrame, binary_col: str, ae_cols: Sequence[str],
                           *, with_beta: bool = False) -> dict:
    """Relative difference in serious-AE rates, specific vs broad targets.

    Paper Results: trials with cell-type-specific targets showed ~32% lower
    serious-AE rates "on average". The phrase is ambiguous, so three clearly
    labelled definitions are reported:

    * ``mean_of_per_organ`` — the mean over organ systems of the per-organ
      relative differences (mean_specific / mean_broad − 1 per AE column);
      the natural reading of "on average" and the value compared with −0.32
      (``paper_comparison``);
    * ``pooled_row_mean`` (alias ``overall``, kept for compatibility) — the
      per-trial mean of whichever AE columns are non-missing, then the
      relative difference of those means (mixes organs with different
      baselines across trials);
    * ``exp_beta_pooled`` (``with_beta=True``) — exp(coefficient) of a beta
      regression of the pooled row-mean on the 0/1 group: an odds-ratio of
      the mean proportion, *not* a relative difference.

    ``per_outcome`` holds, per AE column, both means, ``relative_difference``,
    counts and a Mann–Whitney U p-value.
    """
    from scipy.stats import mannwhitneyu

    def one(values: pd.Series, g: pd.Series) -> dict:
        m = values.notna() & g.notna()
        v, gg = values[m].astype(float), g[m].astype(float)
        s, b = v[gg == 1], v[gg == 0]
        ms = s.mean() if len(s) else np.nan
        mb = b.mean() if len(b) else np.nan
        rel = ms / mb - 1 if (np.isfinite(mb) and mb > 0) else np.nan
        try:
            p = mannwhitneyu(s, b).pvalue if len(s) and len(b) else np.nan
        except Exception:  # noqa: BLE001
            p = np.nan
        return dict(mean_specific=float(ms), mean_broad=float(mb),
                    relative_difference=float(rel), n_specific=int(len(s)),
                    n_broad=int(len(b)), mannwhitney_p=float(p))

    ae_cols = list(ae_cols)
    g = df[binary_col]
    per = {c: one(df[c], g) for c in ae_cols}
    pooled = df[ae_cols].astype(float).mean(axis=1, skipna=True) if ae_cols else None
    overall = one(pooled, g) if ae_cols else {}
    rels = [v["relative_difference"] for v in per.values() if np.isfinite(v["relative_difference"])]
    mean_per_organ = dict(relative_difference=float(np.mean(rels)) if rels else float("nan"),
                          n_organs=len(rels),
                          definition="mean over AE columns of (mean_specific / mean_broad - 1)")
    out = dict(per_outcome=per, overall=overall,
               pooled_row_mean={**overall, "definition": "relative difference of per-trial means over the "
                                                         "available AE columns"},
               mean_of_per_organ=mean_per_organ,
               paper_comparison="mean_of_per_organ")
    if with_beta and ae_cols:
        d = pd.DataFrame({"g": g.astype(float), "y": pooled})
        r = beta_regression(d, "g", "y", standardize=False)
        out["exp_beta_pooled"] = dict(exp_coef=r.get("exp_coef"), coef=r.get("coef"), p=r.get("p"), n=r.get("n"),
                                      engine=r.get("engine"),
                                      definition="exp(beta) of a beta regression of the pooled row-mean on the "
                                                 "0/1 group (odds-ratio scale of the mean proportion)")
    return out


def refdr(table: pd.DataFrame) -> pd.DataFrame:
    """Recompute ``p_fdr`` / ``p_perm_fdr`` (BH) over exactly the rows of ``table``.

    Use after filtering a suite table to the family actually reported, e.g.
    ``refdr(t.query("model != 'logistic'"))`` for the adjusted analyses.
    """
    t = table.copy()
    if "p" in t.columns:
        t["p_fdr"] = benjamini_hochberg(t["p"]) if len(t) else []
    if "p_perm" in t.columns:
        t["p_perm_fdr"] = benjamini_hochberg(t["p_perm"]) if len(t) else []
    return t


def run_association_suite(
    df: pd.DataFrame,
    features: Sequence[str],
    binary_outcomes: Sequence[str],
    ae_outcomes: Sequence[str] = (),
    n_perm: int = 1000,
    covariates: Sequence[str] | None = None,
    seed: int = 0,
    *,
    include_unadjusted: bool = True,
    beta_engine: str = "statsmodels",
) -> pd.DataFrame:
    """Run the paper's feature × outcome association grid into a tidy table.

    For each feature and binary outcome: univariate logistic (``model =
    "logistic"``; estimate = OR per SD) and, when ``covariates`` is given,
    the adjusted logistic (``"logistic_adjusted"``). For each AE percentage
    outcome: beta regression (``"beta"``; estimate = logit-scale coefficient
    per SD) — or, when ``covariates`` is given, the covariate-adjusted beta
    regression (``"beta_adjusted"``) *instead* (the unadjusted beta row is
    emitted only without covariates). Univariate models get a permutation
    p-value when ``n_perm > 0``. ``include_unadjusted=False`` drops the
    unadjusted logistic rows when covariates are given, so the table is
    exactly the adjusted family.

    Columns: ``feature, outcome, model, estimate, ci_low, ci_high, p, p_perm,
    p_perm_raw, n, engine, p_fdr, p_perm_fdr`` — FDR (Benjamini–Hochberg) is
    applied across the returned rows, separately to parametric and permutation
    p. After filtering rows, call :func:`refdr` to correct over the family
    actually reported.
    """
    rows = []
    covariates = list(covariates) if covariates else []
    for f in features:
        for o in binary_outcomes:
            if include_unadjusted or not covariates:
                r = univariate_logistic(df, f, o)
                perm = permutation_test(df, f, o, "logistic", n_perm, seed) if n_perm > 0 else {}
                rows.append(dict(feature=f, outcome=o, model="logistic", estimate=r["odds_ratio"],
                                 ci_low=r["ci_low"], ci_high=r["ci_high"], p=r["p"],
                                 p_perm=perm.get("p_perm", np.nan),
                                 p_perm_raw=perm.get("p_perm_raw", np.nan), n=r["n"], engine="statsmodels"))
            if covariates:
                r = adjusted_logistic(df, f, o, covariates)
                rows.append(dict(feature=f, outcome=o, model="logistic_adjusted",
                                 estimate=r["odds_ratio"], ci_low=r["ci_low"],
                                 ci_high=r["ci_high"], p=r["p"], p_perm=np.nan,
                                 p_perm_raw=np.nan, n=r["n"], engine="statsmodels"))
        for o in ae_outcomes:
            if covariates:
                r = beta_regression(df, f, o, covariates=covariates, engine=beta_engine)
                rows.append(dict(feature=f, outcome=o, model="beta_adjusted", estimate=r["coef"],
                                 ci_low=r["ci_low"], ci_high=r["ci_high"], p=r["p"], p_perm=np.nan,
                                 p_perm_raw=np.nan, n=r["n"], engine=r.get("engine")))
                continue
            r = beta_regression(df, f, o, engine=beta_engine)
            perm = permutation_test(df, f, o, "beta", n_perm, seed) if n_perm > 0 else {}
            rows.append(dict(feature=f, outcome=o, model="beta", estimate=r["coef"],
                             ci_low=r["ci_low"], ci_high=r["ci_high"], p=r["p"],
                             p_perm=perm.get("p_perm", np.nan),
                             p_perm_raw=perm.get("p_perm_raw", np.nan), n=r["n"], engine=r.get("engine")))
    cols = ["feature", "outcome", "model", "estimate", "ci_low", "ci_high", "p", "p_perm",
            "p_perm_raw", "n", "engine"]
    out = pd.DataFrame(rows, columns=cols)
    return refdr(out)


# ---------------------------------------------------------------------------
# Genetic-evidence replication (Razuvayevskaya et al.; paper fig. S2)
# ---------------------------------------------------------------------------


def _flag_logistic(d: pd.DataFrame, flag: str, outcome: str) -> dict:
    import statsmodels.api as sm

    d = d[[flag, outcome]].dropna()
    n = len(d)
    try:
        y = _binary(d[outcome]).astype(float).to_numpy()
        x = d[flag].astype(float)
        if n < 5 or x.nunique() < 2 or len(np.unique(y)) < 2:
            raise ValueError("insufficient data or a single class")
        X = sm.add_constant(x.rename(flag).to_frame(), has_constant="add")
        res = _fit_logit(y, X)
        b = float(res.params[flag])
        se = float(res.bse[flag])
        p = float(res.pvalues[flag])
    except Exception as exc:  # noqa: BLE001
        return _nan_result(n, f"{type(exc).__name__}: {exc}")
    return dict(odds_ratio=float(np.exp(b)), ci_low=float(np.exp(b - _Z * se)),
                ci_high=float(np.exp(b + _Z * se)), p=p, n=int(n), beta=b, se=se,
                n_flagged=int(d[flag].astype(float).sum()))


def _split_categories(v) -> list[str]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    if isinstance(v, (list, tuple, np.ndarray)):
        items = [str(x) for x in v]
    else:
        items = re.split(r"[|,;]", re.sub(r"[\[\]'\"]", "", str(v)))
    return [i.strip() for i in items if i.strip() and i.strip().lower() not in ("nan", "none")]


def genetic_evidence_replication(
    df: pd.DataFrame,
    outcomes: Sequence[str],
    ae_cols: Sequence[str] = (),
    *,
    flag: str = "genetic_evidence",
    stop_col: str | None = "stop_categories",
    status_col: str = "status",
    min_events: int = 5,
) -> pd.DataFrame:
    """Replicate Razuvayevskaya et al. (Nat Genet 2024) on this dataset (fig. S2).

    * each binary outcome ~ genetic-evidence flag (binomial GLM; OR with Wald CI);
    * each AE percentage ~ flag (beta regression; ``estimate`` = logit-scale
      coefficient, ``exp_coef`` its odds-ratio scale);
    * optionally, per stop-reason category (``stop_col``, pipe/list values):
      trials stopped for that category (1) vs completed trials (0), on the
      flag — Razuvayevskaya-style stop-category ORs.

    BH FDR across all returned rows. Columns: ``analysis, outcome, model,
    estimate, ci_low, ci_high, p, n, n_flagged, engine, p_fdr``.
    """
    if flag not in df.columns:
        raise KeyError(f"{flag!r} not in the dataset (pass genetic pairs or OPEN_TARGETS_DATA_PATH)")
    rows = []
    for o in outcomes:
        if o not in df.columns:
            continue
        r = _flag_logistic(df, flag, o)
        rows.append(dict(outcome=o, model="logistic", estimate=r["odds_ratio"], ci_low=r["ci_low"],
                         ci_high=r["ci_high"], p=r["p"], n=r["n"], n_flagged=r.get("n_flagged"),
                         engine="statsmodels"))
    for o in ae_cols:
        if o not in df.columns:
            continue
        r = beta_regression(df, flag, o, standardize=False)
        rows.append(dict(outcome=o, model="beta", estimate=r["coef"], ci_low=r["ci_low"], ci_high=r["ci_high"],
                         p=r["p"], n=r["n"], exp_coef=r.get("exp_coef"), engine=r.get("engine")))
    if stop_col and stop_col in df.columns and status_col in df.columns:
        st = df[status_col].astype("string").str.strip().str.lower()
        cats = df[stop_col].map(_split_categories)
        completed = st.isin(_COMPLETED).fillna(False).to_numpy(dtype=bool)
        stopped = st.isin(_STOPPED).fillna(False).to_numpy(dtype=bool)
        all_cats = sorted({c for cs, s_ in zip(cats, stopped) if s_ for c in cs})
        for c in all_cats:
            has = np.array([c in cs for cs in cats], dtype=bool)
            y = np.where(stopped & has, 1.0, np.where(completed, 0.0, np.nan))
            if np.nansum(y) < min_events:
                continue
            d = pd.DataFrame({flag: df[flag].to_numpy(), "y": y})
            r = _flag_logistic(d, flag, "y")
            rows.append(dict(outcome=f"stopped:{c}", model="logistic_stop_category", estimate=r["odds_ratio"],
                             ci_low=r["ci_low"], ci_high=r["ci_high"], p=r["p"], n=r["n"],
                             n_flagged=r.get("n_flagged"), engine="statsmodels"))
    out = pd.DataFrame(rows, columns=["outcome", "model", "estimate", "ci_low", "ci_high", "p", "n", "n_flagged",
                                      "exp_coef", "engine"])
    out.insert(0, "analysis", "genetic_evidence_replication")
    out["feature"] = flag
    out["p_fdr"] = benjamini_hochberg(out["p"]) if len(out) else []
    return out


def combine_therapeutic_areas(values: Iterable) -> str:
    """A trial's therapeutic-area combination: the sorted union of the therapeutic
    areas of *all* its diseases (each value a list or a pipe-joined string)."""
    areas: set[str] = set()
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        items = list(v) if isinstance(v, (list, tuple, np.ndarray)) else str(v).split("|")
        areas |= {str(x).strip() for x in items if str(x).strip() and str(x).strip() not in ("none", "nan")}
    return "|".join(sorted(areas)) if areas else "unknown"


# ---------------------------------------------------------------------------
# Outcome derivation
# ---------------------------------------------------------------------------

_STOPPED = {"terminated", "withdrawn", "suspended"}
_COMPLETED = {"completed"}


def phase_to_numeric(phase) -> float:
    """Parse a trial phase to a number.

    Accepts numbers (Open Targets/ChEMBL style: 0.5 = early phase 1, 1–4) and
    strings such as ``"PHASE2"``, ``"Phase 1/Phase 2"``, ``"EARLY_PHASE1"``,
    ``"Phase IV"``.  Combined phases (1/2, 2/3) map to the higher phase (the
    ChEMBL convention).  Returns NaN if unparseable.
    """
    if phase is None or (isinstance(phase, float) and np.isnan(phase)):
        return float("nan")
    if isinstance(phase, (int, float, np.integer, np.floating)) and not isinstance(phase, bool):
        return float(phase)
    s = str(phase).strip().upper()
    if not s or s in {"NA", "NAN", "NONE", "N/A"}:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        pass
    if "EARLY" in s and "1" in s or s in {"PHASE0", "PHASE 0", "EARLY_PHASE1"}:
        return 0.5
    roman = {"IV": 4, "III": 3, "II": 2, "I": 1}
    nums = [float(x) for x in re.findall(r"[0-4]", s)]
    for tok in re.findall(r"\b(IV|III|II|I)\b", s.replace("PHASE", " ")):
        nums.append(float(roman[tok]))
    return max(nums) if nums else float("nan")


def _stop_text(row_vb, row_ot) -> str | None:
    for v in (row_vb, row_ot):
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        if isinstance(v, (list, tuple, np.ndarray)):
            if len(v) == 0:
                continue
            return "|".join(map(str, v)).lower()
        s = str(v).strip()
        if s and s.lower() not in {"nan", "none", "[]"}:
            return s.lower()
    return None


def _has_negative(text: str | None) -> bool:
    return bool(text) and bool(re.search(r"negative|safety|side[\s_-]?effect", text))


def _has_safety(text: str | None) -> bool:
    return bool(text) and bool(re.search(r"safety|side[\s_-]?effect", text))


def build_outcomes(labels: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Derive trial-level outcomes from the released labels + trial mapping.

    ``labels`` (one row per trial; duplicates keep the first) must contain
    ``nct_id`` and may contain ``phase``, ``status``,
    ``studyStopReasonCategories``, ``virtualbiotech_stop_reason_categories``
    (pipe-separated), ``phase2_progression`` (EVER/NEVER),
    ``primary_endpoint_result``/``secondary_endpoint_result``
    (POSITIVE/NEGATIVE/UNKNOWN) and ``ae_serious_*_pct`` columns.
    ``mapping`` has one row per (trial, drug, target, disease): ``nct_id``,
    ``drugId``, ``targetId``, ``diseaseId``, ``phase``, ``status``,
    ``trial_date`` ...

    Derived columns (NaN = not applicable / unknown):

    * ``primary_success`` / ``secondary_success`` — POSITIVE → 1,
      NEGATIVE → 0, otherwise NaN.
    * ``phase1_to_2`` — for phase ≤ 1 trials: EVER → 1, NEVER → 0.
    * ``stopped_early`` — Terminated/Withdrawn/Suspended → 1, Completed → 0.
    * ``stopped_negative`` — stopped with a stop category mentioning
      Negative, Safety or side effects → 1; Completed → 0; stopped for other
      reasons → NaN.  Categories come from
      ``virtualbiotech_stop_reason_categories`` when present, else
      ``studyStopReasonCategories``.
    * ``stopped_safety`` — as above but only Safety/side-effect categories.
    * ``ever_phase4`` — **definition**: 1 if any (drugId, diseaseId) pair of
      the trial appears in *any* mapping row with phase ≥ 4 (i.e. the drug
      reached Phase IV for that indication, in this or another trial), else
      0; NaN if the trial has no mapping rows.
    * ``ever_phase_ge_2`` (phase < 2 trials) and ``ever_phase_ge_3``
      (phase < 3 trials) — same definition with thresholds 2 and 3.
    * ``start_year`` — year of the earliest ``trial_date`` of the trial.
    * ``phase_num`` — numeric phase (labels' phase, else max mapping phase).

    ``ae_serious_*`` columns and ``modality``/``therapeutic_area`` (if present
    in either input) are carried over.  Returns one row per trial (union of
    labels and mapping trials), indexed by position with an ``nct_id``
    column.
    """
    lab = labels.drop_duplicates("nct_id", keep="first").set_index("nct_id")
    ids = pd.Index(lab.index).union(pd.Index(mapping["nct_id"].dropna().unique()))
    out = pd.DataFrame(index=ids)
    out.index.name = "nct_id"

    mp = mapping.copy()
    mp["_phase_num"] = mp["phase"].map(phase_to_numeric) if "phase" in mp else np.nan
    map_phase = mp.groupby("nct_id")["_phase_num"].max()

    lab_phase = lab["phase"] if "phase" in lab else pd.Series(dtype=object)
    out["phase"] = lab_phase.reindex(ids)
    if "phase" in mp:
        first_map_phase = mp.groupby("nct_id")["phase"].first().reindex(ids)
        out["phase"] = out["phase"].where(out["phase"].notna(), first_map_phase)
    pn = out["phase"].map(phase_to_numeric).astype(float)
    out["phase_num"] = pn.where(pn.notna(), map_phase.reindex(ids))

    status = lab["status"].reindex(ids) if "status" in lab else pd.Series(np.nan, index=ids)
    if "status" in mp:
        status = status.where(status.notna(), mp.groupby("nct_id")["status"].first().reindex(ids))
    out["status"] = status
    st = status.astype("string").str.strip().str.lower()

    def endpoint(col):
        if col not in lab:
            return pd.Series(np.nan, index=ids)
        s = lab[col].reindex(ids).astype("string").str.strip().str.upper()
        return s.map({"POSITIVE": 1.0, "NEGATIVE": 0.0}).astype(float)

    out["primary_success"] = endpoint("primary_endpoint_result")
    out["secondary_success"] = endpoint("secondary_endpoint_result")

    if "phase2_progression" in lab:
        prog = lab["phase2_progression"].reindex(ids).astype("string").str.strip().str.upper()
        p12 = prog.map({"EVER": 1.0, "NEVER": 0.0}).astype(float)
        out["phase1_to_2"] = p12.where(~(out["phase_num"] > 1))
    else:
        out["phase1_to_2"] = np.nan

    stopped = st.isin(_STOPPED).fillna(False).astype(bool)
    completed = st.isin(_COMPLETED).fillna(False).astype(bool)
    out["stopped_early"] = np.where(stopped, 1.0, np.where(completed, 0.0, np.nan))

    vb = lab["virtualbiotech_stop_reason_categories"].reindex(ids) \
        if "virtualbiotech_stop_reason_categories" in lab else pd.Series(None, index=ids)
    ot = lab["studyStopReasonCategories"].reindex(ids) \
        if "studyStopReasonCategories" in lab else pd.Series(None, index=ids)
    texts = [_stop_text(a, b) for a, b in zip(vb.to_numpy(dtype=object), ot.to_numpy(dtype=object))]
    neg = np.array([_has_negative(t) for t in texts], dtype=bool)
    saf = np.array([_has_safety(t) for t in texts], dtype=bool)
    stp = stopped.to_numpy()
    cmp_ = completed.to_numpy()
    out["stop_categories"] = texts  # lower-cased category text (for stop-category analyses)
    out["stopped_negative"] = np.where(stp & neg, 1.0, np.where(cmp_, 0.0, np.nan))
    out["stopped_safety"] = np.where(stp & saf, 1.0, np.where(cmp_, 0.0, np.nan))

    # Phase-reached outcomes from (drug, indication) pairs.
    if {"drugId", "diseaseId"}.issubset(mp.columns):
        pairs = mp[["nct_id", "drugId", "diseaseId", "_phase_num"]].dropna(
            subset=["nct_id", "drugId", "diseaseId"])
        pair_max = pairs.groupby(["drugId", "diseaseId"])["_phase_num"].max()
        pairs = pairs.join(pair_max.rename("_pair_max"), on=["drugId", "diseaseId"])
        trial_max = pairs.groupby("nct_id")["_pair_max"].max().reindex(ids)
        has_map = pd.Series(ids.isin(pairs["nct_id"].unique()), index=ids)
        for k, name, elig in ((4, "ever_phase4", None),
                              (2, "ever_phase_ge_2", 2),
                              (3, "ever_phase_ge_3", 3)):
            v = (trial_max >= k).astype(float).where(has_map)
            if elig is not None:
                v = v.where(out["phase_num"] < elig)
            out[name] = v
    else:
        out["ever_phase4"] = out["ever_phase_ge_2"] = out["ever_phase_ge_3"] = np.nan

    if "trial_date" in mp:
        dates = pd.to_datetime(mp["trial_date"], errors="coerce")
        out["start_year"] = dates.groupby(mp["nct_id"]).min().dt.year.reindex(ids).astype(float)
    else:
        out["start_year"] = np.nan

    for c in ("modality", "therapeutic_area"):
        src = None
        if c in lab:
            src = lab[c].reindex(ids)
        if c in mp:
            m = mp.groupby("nct_id")[c].first().reindex(ids)
            src = m if src is None else src.where(src.notna(), m)
        if src is not None:
            out[c] = src

    ae_cols = [c for c in lab.columns if str(c).startswith("ae_serious_")]
    for c in ae_cols:
        out[c] = pd.to_numeric(lab[c].reindex(ids), errors="coerce")

    return out.reset_index()
