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
from typing import Any, Iterable, Sequence

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
    # P9: the authors' analysis code (Zenodo archive) made reusable
    "betareg_fit", "glmm_laplace", "BETA_SQUEEZE", "ZSCORE_DDOF",
    "STOP_REASON_CATEGORIES", "AUTHORS_ANALYSES", "AE_ORGAN_SYSTEMS", "AUTHORS_OUTCOME_COLUMNS",
    "stop_categories", "authors_outcome", "authors_subset", "authors_outcome_frame", "aggregate_trials",
    "genetic_evidence_table", "drugtype_combo", "ta_combo", "mixed_covariates",
    "contingency_or", "multivariable_logistic", "multivariable_beta", "kmeans_threshold",
    "target_level_table", "ae_group_comparison", "independent_columns",
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


#: z-scoring uses the population SD (ddof=0), i.e. ``sklearn.StandardScaler`` as in the
#: authors' analysis code (``clinical_trials/code/figure2_*.py``).
ZSCORE_DDOF = 0


def _zscore(x: pd.Series, ddof: int | None = None) -> pd.Series:
    sd = x.std(ddof=ZSCORE_DDOF if ddof is None else ddof)
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


#: Response transform for beta regression: ``"clip"`` = clip proportions to
#: [1e-6, 1 − 1e-6] (the authors' code, ``np.clip(y / 100, 1e-6, 1 - 1e-6)``);
#: ``"smithson_verkuilen"`` = (y(n − 1) + 0.5)/n.
BETA_SQUEEZE = "clip"
CLIP_EPS = 1e-6


def _squeeze(y: np.ndarray, squeeze: str, n: int | None = None) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if squeeze == "clip":
        return np.clip(y, CLIP_EPS, 1 - CLIP_EPS)
    if squeeze in ("smithson_verkuilen", "sv"):
        return smithson_verkuilen(np.clip(y, 0.0, 1.0), n)
    raise ValueError("squeeze must be 'clip' or 'smithson_verkuilen'")


def betareg_fit(y: np.ndarray, X: np.ndarray, *, maxiter: int = 200, tol: float = 1e-10,
                optimizer: str = "bfgs", start: tuple | None = None) -> dict:
    """Maximum-likelihood beta regression, logit mean link, constant precision.

    A Python port of what R ``betareg::betareg(y ~ x)`` computes by default
    (Ferrari & Cribari-Neto 2004; Cribari-Neto & Zeileis 2010): ML estimates
    (quasi-Newton from the ``lm(logit(y) ~ x)`` start, then Fisher scoring)
    and the covariance from the analytic **expected** information matrix,
    which is what ``summary(betareg(...))`` reports. (``statsmodels``'
    ``BetaModel`` gives the same point estimates but observed-Hessian
    standard errors, ~0.5% different on the paper's data.)

    ``y`` must lie strictly in (0, 1); ``X`` includes the intercept column.
    Returns ``params`` (mean coefficients), ``se``, ``z``, ``p``, ``phi``,
    ``phi_se``, ``cov`` (mean block), ``loglik``, ``pseudo_r2`` (squared
    correlation of the linear predictor and logit(y), as betareg) and
    ``converged``. ``optimizer="scoring"`` skips the quasi-Newton stage
    (faster, but only safe from a good ``start``); ``start=(beta, phi)``
    replaces the ``lm(logit(y) ~ x)`` start values.
    """
    from scipy import optimize
    from scipy import stats as sstats
    from scipy.special import digamma, expit, gammaln, polygamma

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    if np.any((y <= 0) | (y >= 1)):
        raise ValueError("beta regression needs 0 < y < 1")
    ylogit = np.log(y / (1 - y))
    l1my = np.log1p(-y)
    ly = np.log(y)
    # betareg starting values: lm(logit(y) ~ X) and the moment estimate of phi
    b0, *_ = np.linalg.lstsq(X, ylogit, rcond=None)
    mu0 = expit(X @ b0)
    res = ylogit - X @ b0
    s2 = float(res @ res) / max(n - k, 1)
    sigma2 = s2 * (mu0 * (1 - mu0)) ** 2
    phi0 = float(np.mean(mu0 * (1 - mu0) / sigma2) - 1)
    if not np.isfinite(phi0) or phi0 <= 0:
        phi0 = 1.0

    def unpack(theta):
        return theta[:k], float(np.exp(theta[k]))

    def nll(theta):
        b, phi = unpack(theta)
        mu = np.clip(expit(X @ b), 1e-15, 1 - 1e-15)
        ll = (gammaln(phi) - gammaln(mu * phi) - gammaln((1 - mu) * phi)
              + (mu * phi - 1) * ly + ((1 - mu) * phi - 1) * l1my)
        return -float(np.sum(ll))

    def score(b, phi):
        mu = np.clip(expit(X @ b), 1e-15, 1 - 1e-15)
        mustar = digamma(mu * phi) - digamma((1 - mu) * phi)
        T = mu * (1 - mu)
        gb = phi * (X.T @ (T * (ylogit - mustar)))
        gphi = float(np.sum(mu * (ylogit - mustar) + l1my - digamma((1 - mu) * phi) + digamma(phi)))
        return gb, gphi

    def grad(theta):
        b, phi = unpack(theta)
        gb, gphi = score(b, phi)
        return -np.concatenate([gb, [gphi * phi]])

    def info(b, phi):
        mu = np.clip(expit(X @ b), 1e-15, 1 - 1e-15)
        a = polygamma(1, mu * phi)
        bb = polygamma(1, (1 - mu) * phi)
        T = mu * (1 - mu)
        K = np.empty((k + 1, k + 1))
        K[:k, :k] = phi ** 2 * (X * ((a + bb) * T ** 2)[:, None]).T @ X
        c = phi * (a * mu - bb * (1 - mu))
        K[:k, k] = K[k, :k] = X.T @ (T * c)
        K[k, k] = float(np.sum(a * mu ** 2 + bb * (1 - mu) ** 2 - polygamma(1, phi)))
        return K

    if start is not None:  # warm start (e.g. the observed fit, for permutation nulls)
        b0, phi0 = np.asarray(start[0], dtype=float), float(start[1])
    theta0 = np.concatenate([b0, [np.log(phi0)]])
    if optimizer == "scoring":  # fast path (permutations): Fisher scoring from the start values
        b, phi, converged = b0.copy(), phi0, False
    else:
        opt = optimize.minimize(nll, theta0, jac=grad, method="BFGS",
                                options={"maxiter": 5000, "gtol": 1e-8})
        b, phi = unpack(opt.x)
        converged = bool(opt.success)
    # Fisher scoring on (beta, phi), as betareg does after optim
    for _ in range(maxiter):
        gb, gphi = score(b, phi)
        try:
            step = np.linalg.solve(info(b, phi), np.concatenate([gb, [gphi]]))
        except np.linalg.LinAlgError:
            break
        if phi + step[k] <= 0 or not np.all(np.isfinite(step)):
            break
        b, phi = b + step[:k], phi + step[k]
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    V = np.linalg.inv(info(b, phi))
    se = np.sqrt(np.diag(V))
    z = b / se[:k]
    eta = X @ b
    pr2 = float(np.corrcoef(eta, ylogit)[0, 1] ** 2) if np.std(eta) > 0 else float("nan")
    return dict(params=b, se=se[:k], z=z, p=2 * sstats.norm.sf(np.abs(z)), phi=float(phi),
                phi_se=float(se[k]), cov=V[:k, :k], loglik=-nll(np.concatenate([b, [np.log(phi)]])),
                pseudo_r2=pr2, converged=converged)


def beta_regression(
    df: pd.DataFrame,
    feature: str,
    outcome_pct: str,
    covariates: Sequence[str] = (),
    *,
    engine: str = "auto",
    standardize: bool = True,
    squeeze: str | None = None,
) -> dict:
    """Beta regression of an AE percentage on the z-scored feature.

    Paper Methods: serious-AE rates (percentages 0–100) are modelled with R's
    ``betareg`` (logit mean link); the coefficient is the change in the logit
    of the mean AE proportion per 1 SD of the predictor. Following the
    authors' code, percentages are divided by 100 and **clipped** to
    [1e-6, 1 − 1e-6] (``squeeze="clip"``, default; ``"smithson_verkuilen"``
    is the alternative (y(n − 1) + 0.5)/n squeeze). Returns ``coef``,
    ``ci_low``/``ci_high`` (coef ± 1.96·SE as in the authors' code), ``p``,
    ``n``, ``se``, ``exp_coef``, ``phi`` and ``engine``.

    ``engine``: ``"auto"`` (default: R ``betareg`` via rpy2 when installed,
    else ``"betareg_py"``), ``"betareg"`` (R only), ``"betareg_py"``
    (:func:`betareg_fit`: the same ML estimates and expected-information SEs
    as R betareg) or ``"statsmodels"`` (``BetaModel``; observed-Hessian SEs).
    ``standardize=False`` keeps the feature on its own scale (e.g. a 0/1 flag).
    """
    if engine not in ("statsmodels", "betareg", "betareg_py", "auto"):
        raise ValueError("engine must be 'auto', 'betareg', 'betareg_py' or 'statsmodels'")
    squeeze = squeeze or BETA_SQUEEZE
    covariates = list(covariates)
    d = df[[feature, outcome_pct, *covariates]].dropna()
    n = len(d)
    base = dict(coef=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, n=int(n),
                se=np.nan, exp_coef=np.nan, engine=engine, squeeze=squeeze)
    use_r = engine == "betareg" or (engine == "auto" and _rpy2_available(["betareg"]))
    if engine == "betareg" and not _rpy2_available(["betareg"]):
        base["error"] = "rpy2 or the R package betareg is not installed (pip install 'vbt-harness[r]')"
        return base
    if use_r:
        try:
            dd = d.copy()
            dd[feature] = _zscore(dd[feature].astype(float)) if standardize else dd[feature].astype(float)
            dd[outcome_pct] = _squeeze(dd[outcome_pct].astype(float).to_numpy() / 100.0, squeeze, n)
            safe, _rhs, rhs_r, _rand = _prep_mixed_fixed(dd, feature, outcome_pct, covariates)
            import rpy2.robjects as ro  # type: ignore

            ro.r("suppressPackageStartupMessages(library(betareg))")
            beta, se, p = _r_fit(safe, "y_out ~ " + " + ".join(rhs_r), "betareg({formula}, data=vbt_df)",
                                 "summary(vbt_fit)$coefficients$mean")
            base.update(coef=beta, se=se, p=p, ci_low=beta - 1.96 * se, ci_high=beta + 1.96 * se,
                        exp_coef=float(np.exp(beta)), engine="R:betareg")
            return base
        except Exception as exc:  # noqa: BLE001
            if engine == "betareg":
                base["error"] = f"betareg failed: {type(exc).__name__}: {exc}"
                return base
            base["note"] = f"R betareg failed, betareg_py used: {exc}"
    import statsmodels.api as sm

    try:
        if n < 5:
            raise ValueError("insufficient data")
        y = _squeeze(d[outcome_pct].astype(float).to_numpy() / 100.0, squeeze, n)
        xf = _zscore(d[feature].astype(float)) if standardize else d[feature].astype(float)
        X = pd.concat([xf.rename(feature), _design(d, covariates)], axis=1)
        X = sm.add_constant(X, has_constant="add")
        i = list(X.columns).index(feature)
        if engine == "statsmodels":
            from statsmodels.othermod.betareg import BetaModel

            base["engine"] = "statsmodels"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = BetaModel(y, X).fit(disp=0)
            coef = float(np.asarray(res.params)[i])
            se = float(np.asarray(res.bse)[i])
            p = float(np.asarray(res.pvalues)[i])
        else:
            base["engine"] = "betareg_py"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = betareg_fit(y, X.to_numpy(dtype=float))
            coef, se, p = float(fit["params"][i]), float(fit["se"][i]), float(fit["p"][i])
            base.update(phi=fit["phi"], pseudo_r2=fit["pseudo_r2"], converged=fit["converged"])
    except Exception as exc:  # noqa: BLE001
        base["error"] = f"{type(exc).__name__}: {exc}"
        return base
    base.update(coef=coef, se=se, p=p, ci_low=coef - 1.96 * se, ci_high=coef + 1.96 * se,
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


class _Binomial:
    n_extra = 0

    def __init__(self, y):
        self.y = y

    def loglik(self, eta, extra):
        return float(np.sum(self.y * eta - np.logaddexp(0.0, eta)))

    def derivs(self, eta, extra, exact=False):
        from scipy.special import expit

        mu = expit(eta)
        return self.y - mu, mu * (1 - mu)  # dl/deta, -d2l/deta2 (canonical link: Fisher = exact)


class _Beta:
    """Beta family, logit mean link, log precision (glmmTMB ``beta_family()``)."""

    n_extra = 1

    def __init__(self, y):
        self.y = y
        self.ylogit = np.log(y / (1 - y))
        self.ly = np.log(y)
        self.l1my = np.log1p(-y)

    def loglik(self, eta, extra):
        from scipy.special import expit, gammaln

        phi = float(np.exp(extra[0]))
        mu = np.clip(expit(eta), 1e-15, 1 - 1e-15)
        return float(np.sum(gammaln(phi) - gammaln(mu * phi) - gammaln((1 - mu) * phi)
                            + (mu * phi - 1) * self.ly + ((1 - mu) * phi - 1) * self.l1my))

    def derivs(self, eta, extra, exact=False):
        from scipy.special import digamma, expit, polygamma

        phi = float(np.exp(extra[0]))
        mu = np.clip(expit(eta), 1e-15, 1 - 1e-15)
        T = mu * (1 - mu)
        resid = self.ylogit - (digamma(mu * phi) - digamma((1 - mu) * phi))
        a, b = polygamma(1, mu * phi), polygamma(1, (1 - mu) * phi)
        w = phi ** 2 * (a + b) * T ** 2
        if exact:
            w = w - phi * resid * T * (1 - 2 * mu)
        return phi * resid * T, w


def glmm_laplace(y: np.ndarray, X: np.ndarray, groups: Sequence[np.ndarray], *, family: str = "binomial",
                 nagq: int = 1, vcov: str | None = None, offset: np.ndarray | None = None,
                 max_inner: int = 100, tol: float = 1e-10) -> dict:
    """GLMM with crossed random intercepts by the Laplace approximation.

    ``y ~ X + (1|g1) + (1|g2) + ...`` with ``family="binomial"`` (logit; the
    authors' ``lme4::glmer(..., family=binomial)``) or ``family="beta"``
    (logit mean, constant precision φ; the authors' ``glmmTMB(...,
    family=beta_family())``). The marginal likelihood is approximated as in
    lme4/TMB: with spherical random effects v (u = Λv, Λ = diag(θ)), the
    Laplace deviance

        −2 log p(y | β, v̂, φ) + ‖v̂‖² + log det(ΛZᵀWZΛ + I)

    with v̂ the conditional mode (Newton / penalized IRLS).

    1. ``nAGQ = 0`` stage: (β, v) found jointly for each (θ, φ); the deviance
       is minimized over θ ≥ 0 (and log φ).
    2. ``nAGQ = 1`` stage (default; glmer's Laplace fit and glmmTMB's
       objective): the deviance minimized jointly over (θ, β, log φ) from the
       stage-1 solution (L-BFGS-B).

    ``vcov``: ``"conditional"`` (binomial default) — the β block of the
    inverse joint penalized Hessian in (β, v), lme4's ``RX``-based
    ``vcov(glmer)``, which reproduces the authors' glmer SEs to ~5 digits;
    ``"hessian"`` (beta default) — twice the inverse numerical Hessian of the
    deviance in (θ, β, log φ), the marginal covariance glmmTMB's ``sdreport``
    gives. Returns ``params``, ``se``, ``z``, ``p``, ``sd``/``var``
    (random-effect SDs/variances), ``phi`` (beta), ``deviance``,
    ``n_groups``, ``nagq``, ``vcov`` and ``converged``.
    """
    import scipy.sparse as sps
    from scipy import optimize
    from scipy import stats as sstats

    if family not in ("binomial", "beta"):
        raise ValueError("family must be 'binomial' or 'beta'")
    vcov = vcov or ("conditional" if family == "binomial" else "hessian")
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    off0 = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    fam = _Binomial(y) if family == "binomial" else _Beta(y)
    e = fam.n_extra
    blocks, sizes = [], []
    for g in groups:
        codes, uniq = pd.factorize(pd.Series(np.asarray(g)).astype(str), sort=True)
        sizes.append(len(uniq))
        blocks.append(sps.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, len(uniq))))
    Z = sps.hstack(blocks).tocsr()
    q = Z.shape[1]
    m = len(groups)
    owner = np.concatenate([np.full(s, j) for j, s in enumerate(sizes)])

    def zwz(ZL, w):
        return (ZL.T @ sps.diags(w) @ ZL).toarray() + np.eye(q)

    inner = {"tol": tol, "obj": True}

    def newton(ogh, x0):
        x = x0.copy()
        obj, g, H = ogh(x)
        for _ in range(max_inner):
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            t = 1.0
            while True:
                x_new = x + t * step
                obj_new, g_new, H_new = ogh(x_new)
                if obj_new >= obj - 1e-10 or t < 1e-8:
                    break
                t /= 2
            # converged: tiny step, or no representable change left (line search
            # exhausted / objective unchanged at double precision)
            done = (float(np.max(np.abs(t * step))) < inner["tol"] * (1 + float(np.max(np.abs(x))))
                    or t < 1e-8 or abs(obj_new - obj) <= 4e-16 * (1 + abs(obj))
                    or (inner["obj"] and abs(obj_new - obj) < 1e-14 * (1 + abs(obj))))
            x, obj, g, H = x_new, obj_new, g_new, H_new
            if done:
                break
        return x

    def logdet_at(ZL, eta, extra):
        _, w_exact = fam.derivs(eta, extra, exact=True)
        sign, ld = np.linalg.slogdet(zwz(ZL, w_exact))
        if sign <= 0:
            _, w = fam.derivs(eta, extra)
            ld = np.linalg.slogdet(zwz(ZL, w))[1]
        return ld

    # ---- start values: fixed-effects fit
    import statsmodels.api as sm

    if family == "binomial":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b_start = np.asarray(sm.GLM(y, X, family=sm.families.Binomial()).fit().params, dtype=float)
        extra_start = np.zeros(0)
    else:
        fit = betareg_fit(y, X)
        b_start = np.asarray(fit["params"], dtype=float)
        extra_start = np.array([np.log(fit["phi"])])
    state = {"b": b_start, "v": np.zeros(q)}

    # ---- stage 1: joint (beta, v) for each (theta, extra)
    def joint(theta, extra):
        ZL = Z @ sps.diags(theta[owner])

        def ogh(x):
            b, v = x[:k], x[k:]
            eta = off0 + X @ b + ZL @ v
            g_eta, w = fam.derivs(eta, extra)
            g = np.concatenate([X.T @ g_eta, ZL.T @ g_eta - v])
            XtWZ = np.asarray(ZL.T @ (X * w[:, None])).T
            H = np.block([[X.T @ (X * w[:, None]), XtWZ], [XtWZ.T, zwz(ZL, w)]])
            return fam.loglik(eta, extra) - 0.5 * v @ v, g, H

        x = newton(ogh, np.concatenate([state["b"], state["v"]]))
        b, v = x[:k], x[k:]
        eta = off0 + X @ b + ZL @ v
        return -2 * fam.loglik(eta, extra) + v @ v + logdet_at(ZL, eta, extra), b, v

    def f0(params):
        dev, b, v = joint(np.asarray(params[:m], dtype=float), np.asarray(params[m:], dtype=float))
        state["b"], state["v"] = b, v
        return dev

    p0 = np.concatenate([np.full(m, 0.5), extra_start])
    opt = optimize.minimize(f0, p0, method="L-BFGS-B", bounds=[(0.0, 20.0)] * m + [(-10, 15)] * e,
                            options={"maxiter": 500, "eps": 1e-4})
    theta, extra = np.asarray(opt.x[:m], dtype=float), np.asarray(opt.x[m:], dtype=float)
    dev, b, v = joint(theta, extra)
    converged = bool(opt.success)

    # ---- stage 2: Laplace deviance over (theta, beta, extra), v at its conditional mode
    vstate = {"v": v.copy()}

    def laplace(params):
        th, bb, ex = np.maximum(params[:m], 0.0), params[m:m + k], params[m + k:]
        ZL = Z @ sps.diags(th[owner])
        off = off0 + X @ bb

        def ogh(vv):
            eta = off + ZL @ vv
            g_eta, w = fam.derivs(eta, ex)
            return fam.loglik(eta, ex) - 0.5 * vv @ vv, ZL.T @ g_eta - vv, zwz(ZL, w)

        vv = newton(ogh, vstate["v"])
        vstate["v"] = vv
        eta = off + ZL @ vv
        return -2 * fam.loglik(eta, ex) + vv @ vv + logdet_at(ZL, eta, ex)

    x_opt = np.concatenate([theta, b, extra])
    if nagq >= 1:
        f_x0 = laplace(x_opt)
        opt2 = optimize.minimize(laplace, x_opt, method="L-BFGS-B",
                                 bounds=[(0.0, 20.0)] * m + [(None, None)] * k + [(-10, 15)] * e,
                                 options={"maxiter": 1000, "eps": 1e-5, "ftol": 1e-14, "gtol": 1e-6})
        if opt2.fun <= f_x0 + 1e-8:
            x_opt = np.asarray(opt2.x, dtype=float)
            x_opt[:m] = np.maximum(x_opt[:m], 0.0)
            converged = converged and bool(opt2.success)
    theta, b, extra = x_opt[:m], x_opt[m:m + k], x_opt[m + k:]
    dev = float(laplace(x_opt))  # leaves vstate at the conditional mode of the optimum
    ZL = Z @ sps.diags(theta[owner])
    eta = off0 + X @ b + ZL @ vstate["v"]
    _, w = fam.derivs(eta, extra)
    XtWZ = np.asarray(ZL.T @ (X * w[:, None])).T
    H = np.block([[X.T @ (X * w[:, None]), XtWZ], [XtWZ.T, zwz(ZL, w)]])
    cov = np.linalg.inv(H)[:k, :k]
    used = "conditional"
    if vcov == "hessian":
        inner.update(tol=1e-12, obj=False)  # tight conditional modes for finite differences
        dev = float(laplace(x_opt))
        free = list(range(m, m + k + e))
        pf = len(free)
        hstep = 1e-3 * np.maximum(1.0, np.abs(x_opt[free]))

        def at(i, di, j=None, dj=0.0):
            xx = x_opt.copy()
            xx[free[i]] += di
            if j is not None:
                xx[free[j]] += dj
            return laplace(xx)

        Hd = np.empty((pf, pf))
        for i in range(pf):
            for j in range(i, pf):
                if i == j:
                    val = (at(i, hstep[i]) - 2 * dev + at(i, -hstep[i])) / hstep[i] ** 2
                else:
                    val = (at(i, hstep[i], j, hstep[j]) - at(i, hstep[i], j, -hstep[j])
                           - at(i, -hstep[i], j, hstep[j]) + at(i, -hstep[i], j, -hstep[j])) \
                        / (4 * hstep[i] * hstep[j])
                Hd[i, j] = Hd[j, i] = val
        try:
            full = 2 * np.linalg.inv(Hd)
            nb = pf - k - e  # free thetas come first
            c = full[nb:nb + k, nb:nb + k]
            if np.all(np.diag(c) > 0):
                cov, used = c, "hessian"
        except np.linalg.LinAlgError:
            pass
    se = np.sqrt(np.diag(cov))
    z = b / se
    out = dict(params=b, se=se, z=z, p=2 * sstats.norm.sf(np.abs(z)), sd=theta, var=theta ** 2,
               deviance=dev, n_groups=sizes, nagq=int(nagq), vcov=used, converged=converged)
    if family == "beta":
        out["phi"] = float(np.exp(extra[0]))
    return out


def independent_columns(X: np.ndarray, tol: float = 1e-9) -> list[int]:
    """Indices of the columns R keeps for a rank-deficient design: walk left to
    right and drop a column that is (numerically) a linear combination of the
    columns already kept (R reports its coefficient as NA)."""
    X = np.asarray(X, dtype=float)
    keep: list[int] = []
    for j in range(X.shape[1]):
        cand = X[:, keep + [j]]
        if np.linalg.matrix_rank(cand, tol=tol * max(1.0, float(np.abs(cand).max()))) == len(keep) + 1:
            keep.append(j)
    return keep


def _laplace_fit(safe: pd.DataFrame, rhs: Sequence[str], rand: Sequence[str], family: str) -> tuple:
    """Fit ``y_out ~ rhs + (1|g)...`` with :func:`glmm_laplace`; return (beta, se, fit) of ``x_feat``."""
    from patsy import dmatrix

    X = dmatrix("~ " + " + ".join(rhs), safe, return_type="dataframe")
    keep = [c for c in X.columns if c == "Intercept" or X[c].nunique() > 1]
    X = X[keep]
    X = X.iloc[:, independent_columns(X.to_numpy(dtype=float))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = glmm_laplace(safe["y_out"].to_numpy(dtype=float), X.to_numpy(dtype=float),
                           [safe[g].to_numpy() for g in rand], family=family)
    i = list(X.columns).index("x_feat")
    return float(fit["params"][i]), float(fit["se"][i]), fit


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
    * ``"laplace"`` — :func:`glmm_laplace` (Python port of glmer's Laplace
      fit; reproduces the authors' glmer estimates and SEs to ~4–5 digits).
    * ``"auto"`` (default) — lme4 if rpy2 and lme4 are installed and the fit
      succeeds, else ``"laplace"``.  Never raises for a missing rpy2.

    Fixed covariates that are numeric are centred and scaled (does not change
    the feature coefficient); non-numeric ones are treated as factors.
    Grouping/covariate columns absent from ``df`` are skipped.  Returns
    ``odds_ratio``, ``ci_low``, ``ci_high``, ``p``, ``n``, ``beta``, ``se``
    and ``engine``.
    """
    if engine not in ("auto", "lme4", "statsmodels", "laplace"):
        raise ValueError("engine must be 'auto', 'lme4', 'laplace' or 'statsmodels'")
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

    if engine in ("auto", "laplace") and rand:
        try:
            beta, se, fit = _laplace_fit(safe, rhs, rand, "binomial")
            p = float(2 * sstats.norm.sf(abs(beta / se)))
            out = dict(odds_ratio=float(np.exp(beta)), ci_low=float(np.exp(beta - _Z * se)),
                       ci_high=float(np.exp(beta + _Z * se)), p=p, n=n, beta=beta, se=se, engine="laplace",
                       re_variance=[float(v) for v in fit["var"]], converged=fit["converged"])
            if errors:
                out["note"] = "; ".join(errors)
            return out
        except Exception as exc:  # noqa: BLE001
            errors.append(f"laplace failed: {type(exc).__name__}: {exc}")
            if engine == "laplace":
                return {**_nan_result(n, "; ".join(errors)), "engine": "laplace"}

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

    Proportions are squeezed with :data:`BETA_SQUEEZE` (authors: clip to
    [1e-6, 1 − 1e-6]). Engines: ``"glmmTMB"`` (``glmmTMB(...,
    family=beta_family())`` via rpy2, the authors' model), ``"laplace"``
    (:func:`glmm_laplace` with ``family="beta"``: the same Laplace objective
    in Python), ``"statsmodels"`` (``BetaModel`` with the grouping factors as
    *fixed-effect* drop-first dummies — a fixed-effects approximation), or
    ``"auto"`` (glmmTMB if available, else laplace).
    Returns ``coef``, ``ci_low``, ``ci_high``, ``p``, ``n``, ``se``,
    ``engine``.
    """
    if engine not in ("auto", "glmmTMB", "statsmodels", "laplace"):
        raise ValueError("engine must be 'auto', 'glmmTMB', 'laplace' or 'statsmodels'")
    fixed = [c for c in (fixed or []) if c in df.columns]
    random = [c for c in (random or []) if c in df.columns]
    d = df[[feature, outcome_pct, *fixed, *random]].dropna()
    n = len(d)
    errors = []
    if engine in ("auto", "glmmTMB") and random:
        if _rpy2_available(["glmmTMB"]):
            try:
                sv = lambda s: pd.Series(  # noqa: E731
                    _squeeze(s.astype(float).to_numpy() / 100.0, BETA_SQUEEZE, len(s)),
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

    if engine in ("auto", "laplace") and random:
        try:
            from scipy import stats as sstats

            sq = lambda s: pd.Series(  # noqa: E731
                _squeeze(s.astype(float).to_numpy() / 100.0, BETA_SQUEEZE, len(s)), index=s.index)
            safe, rhs, _rhs_r, rand = _prep_mixed(d, feature, outcome_pct, fixed, random, outcome_transform=sq)
            if not rand:
                raise ValueError("no grouping factor with more than one level")
            beta, se, fit = _laplace_fit(safe, rhs, rand, "beta")
            out = dict(coef=beta, ci_low=beta - 1.96 * se, ci_high=beta + 1.96 * se,
                       p=float(2 * sstats.norm.sf(abs(beta / se))), n=len(safe), se=se,
                       exp_coef=float(np.exp(beta)), engine="laplace", phi=fit.get("phi"),
                       re_variance=[float(v) for v in fit["var"]], converged=fit["converged"])
            if errors:
                out["note"] = "; ".join(errors)
            return out
        except Exception as exc:  # noqa: BLE001
            errors.append(f"laplace failed: {type(exc).__name__}: {exc}")
            if engine == "laplace":
                return dict(coef=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, n=n, se=np.nan,
                            exp_coef=np.nan, engine="laplace", error="; ".join(errors))

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
    res = beta_regression(d2, feature, outcome_pct, covariates=covs, engine="statsmodels")
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


def binarize_tau(values, seed: int = 42) -> tuple[float, np.ndarray]:
    """Split trial-level τ into "cell-type-specific" vs "broadly expressed".

    Paper Methods: k-means (k = 2) on the trial-level τ distribution; the
    threshold is the midpoint of the two cluster centres (the paper obtained
    τ ≈ 0.69; the authors' code uses ``KMeans(random_state=42, n_init=10)``).  Returns ``(threshold, labels)`` where ``labels`` is a float
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
    beta_engine: str = "auto",
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


# ---------------------------------------------------------------------------
# The authors' "6-section" outcome structure (clinical_trials/code/figure2_*.py)
# ---------------------------------------------------------------------------

#: Stop-reason categories analysed by the authors (ChEMBL/Open Targets
#: ``studyStopReasonCategories`` vocabulary).
STOP_REASON_CATEGORIES = (
    "Negative", "Safety or side effects", "Insufficient enrollment", "Business or administrative",
    "Study design", "Logistics or resources", "Another study", "Regulatory", "COVID-19",
    "Interim analysis", "Success",
)

#: (section, outcome label, outcome type, value) exactly as in the authors'
#: ``figure2_virtualbiotech_analysis.py`` (sections A–E; F = AE beta models).
AUTHORS_ANALYSES: tuple[tuple[str, str, str, Any], ...] = (
    ("Phase", "Phase II+", "phase", 2.0),
    ("Phase", "Phase III+", "phase", 3.0),
    ("Phase", "Phase IV", "phase", 4.0),
    ("Stopped Status", "Terminated", "status", "Terminated"),
    ("Stopped Status", "Withdrawn", "status", "Withdrawn"),
    ("Stopped Status", "Suspended", "status", "Suspended"),
    *(("Stop Reason", c, "stop_reason", c) for c in STOP_REASON_CATEGORIES),
    ("Endpoint", "Primary Positive", "endpoint", "primary"),
    ("Endpoint", "Secondary Positive", "endpoint", "secondary"),
    ("Endpoint", "Either Positive", "endpoint", "either"),
    ("Phase 1 Progression", "Phase 2 Progression", "phase1_progression", "EVER"),
)

#: (AE column, label) — section F, R betareg on AE % per organ system.
AE_ORGAN_SYSTEMS: tuple[tuple[str, str], ...] = (
    ("ae_serious_infections_pct", "Infections"),
    ("ae_serious_gastrointestinal_pct", "Gastrointestinal"),
    ("ae_serious_cardiac_pct", "Cardiac"),
    ("ae_serious_blood_lymphatic_pct", "Blood/Lymphatic"),
    ("ae_serious_nervous_pct", "Nervous"),
    ("ae_serious_respiratory_pct", "Respiratory"),
    ("ae_serious_general_pct", "General"),
    ("ae_serious_vascular_pct", "Vascular"),
    ("ae_serious_renal_pct", "Renal"),
    ("ae_serious_injury_pct", "Injury"),
)

#: build_outcomes column for each authors' analysis (outcome label -> column).
AUTHORS_OUTCOME_COLUMNS: dict[str, str] = {
    "Phase II+": "phase_ge_2", "Phase III+": "phase_ge_3", "Phase IV": "phase_ge_4",
    "Terminated": "status_terminated", "Withdrawn": "status_withdrawn", "Suspended": "status_suspended",
    **{c: "stop_" + re.sub(r"\W+", "_", c.lower()).strip("_") for c in STOP_REASON_CATEGORIES},
    "Primary Positive": "primary_success", "Secondary Positive": "secondary_success",
    "Either Positive": "either_success", "Phase 2 Progression": "phase1_to_2",
}


def _is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    try:
        return bool(pd.isna(v)) if np.ndim(v) == 0 else False
    except (TypeError, ValueError):
        return False


def stop_categories(chembl, vb, source: str = "chembl_first") -> list[str] | None:
    """A trial's stop-reason categories with the authors' hierarchy.

    ``source="chembl_first"`` (authors' code): ChEMBL/Open Targets
    ``studyStopReasonCategories`` when present, else the Virtual Biotech
    categories. ``"vb_first"`` reverses the order. Values are pipe-separated
    strings (lists are accepted). Returns ``None`` when neither is present.
    """
    order = (chembl, vb) if source == "chembl_first" else (vb, chembl)
    for v in order:
        if _is_missing(v):
            continue
        if isinstance(v, (list, tuple, np.ndarray)):
            return [str(x).strip() for x in v]
        return [c.strip() for c in str(v).split("|")]
    return None


def authors_outcome(df: pd.DataFrame, outcome_type: str, value=None,
                    stop_reason_source: str = "chembl_first") -> pd.Series:
    """The authors' ``create_outcome_variable`` (figure2_virtualbiotech_analysis.py).

    * ``phase``: ``phase >= value`` (cross-sectional: the trial's own phase);
    * ``status``: ``status == value``;
    * ``stop_reason``: category membership with :func:`stop_categories`
      (0 when the trial has no categories at all);
    * ``endpoint``: POSITIVE → 1, *anything else* (NEGATIVE, UNKNOWN, NA) → 0
      for ``primary``/``secondary``/``either``;
    * ``phase1_progression``: EVER → 1, NEVER → 0, otherwise NaN.
    """
    if outcome_type == "phase":
        return (pd.to_numeric(df["phase"], errors="coerce") >= value).astype(int)
    if outcome_type == "status":
        return (df["status"] == value).astype(int)
    if outcome_type == "stop_reason":
        ch = df["studyStopReasonCategories"] if "studyStopReasonCategories" in df else pd.Series(None, index=df.index)
        vb = df["virtualbiotech_stop_reason_categories"] \
            if "virtualbiotech_stop_reason_categories" in df else pd.Series(None, index=df.index)
        hits = [int(value in (stop_categories(a, b, stop_reason_source) or []))
                for a, b in zip(ch.to_numpy(dtype=object), vb.to_numpy(dtype=object))]
        return pd.Series(hits, index=df.index, dtype=int)
    if outcome_type == "endpoint":
        prim = df["primary_endpoint_result"] == "POSITIVE"
        sec = df["secondary_endpoint_result"] == "POSITIVE"
        if value == "primary":
            return prim.astype(int)
        if value == "secondary":
            return sec.astype(int)
        if value == "either":
            return (prim | sec).astype(int)
        raise ValueError(f"unknown endpoint outcome value: {value}")
    if outcome_type == "phase1_progression":
        return df["phase2_progression"].map(lambda x: 1.0 if x == "EVER" else (0.0 if x == "NEVER" else np.nan))
    raise ValueError(f"unknown outcome type: {outcome_type}")


def authors_subset(df: pd.DataFrame, outcome_type: str) -> pd.DataFrame:
    """Trials entering each analysis, as in the authors' code.

    Stop reasons: trials with ChEMBL *or* VB stop categories; endpoints:
    Phase II/III trials (``phase in {2, 3}``); everything else: all trials
    (Phase I→II drops NaN outcomes afterwards).
    """
    if outcome_type == "stop_reason":
        m = pd.Series(False, index=df.index)
        for c in ("studyStopReasonCategories", "virtualbiotech_stop_reason_categories"):
            if c in df:
                m |= df[c].notna()
        return df[m]
    if outcome_type == "endpoint":
        return df[pd.to_numeric(df["phase"], errors="coerce").isin([2.0, 3.0])]
    return df


def authors_outcome_frame(df: pd.DataFrame, outcome_type: str, value=None, *,
                          stop_reason_source: str = "chembl_first", min_positive: int = 30):
    """``(subset, y)`` for one authors' analysis, or ``None`` if fewer than
    ``min_positive`` (30) positive trials (the authors skip those outcomes)."""
    sub = authors_subset(df, outcome_type)
    y = authors_outcome(sub, outcome_type, value, stop_reason_source)
    if outcome_type == "phase1_progression":
        ok = y.notna()
        sub, y = sub[ok], y[ok].astype(int)
    if int(y.sum()) < min_positive:
        return None
    return sub, y


def aggregate_trials(labels: pd.DataFrame, mapping: pd.DataFrame, gene_features: pd.DataFrame, *,
                     feature_cols: Sequence[str] | None = None, how: str = "min",
                     id_col: str = "ensembl_id", target_col: str = "targetId",
                     require: Sequence[str] = ("tau_cell_type", "bimodality_score")) -> pd.DataFrame:
    """Trial-level dataset exactly as the authors build it.

    labels ⋈ (nct_id, targetId) pairs of ``mapping`` ⋈ gene features (inner
    joins), then one row per trial: clinical/AE columns take the first value,
    feature columns are aggregated with ``how`` (``"min"`` — the paper's
    least-specific-target rule; ``"mean"``/``"max"`` for sensitivity). Trials
    missing any ``require`` feature are dropped. Adds ``n_targets`` (targets
    with features) as a harness covariate.
    """
    feats = gene_features
    if id_col not in feats.columns:
        feats = feats.rename_axis(id_col).reset_index()
    if feature_cols is None:
        feature_cols = [c for c in feats.columns if c != id_col and pd.api.types.is_numeric_dtype(feats[c])]
    feature_cols = list(feature_cols)
    pairs = mapping[["nct_id", target_col]].drop_duplicates()
    lab_cols = [c for c in labels.columns if c != "nct_id"]
    merged = labels.merge(pairs, on="nct_id", how="inner").merge(
        feats[[id_col, *feature_cols]], left_on=target_col, right_on=id_col, how="inner")
    agg: dict[str, Any] = {c: "first" for c in lab_cols}
    agg.update({c: how for c in feature_cols})
    g = merged.groupby("nct_id")
    out = g.agg(agg)
    out["n_targets"] = g[target_col].nunique()
    out = out.reset_index()
    req = [c for c in require if c in out.columns]
    if req:
        out = out[out[req].notna().all(axis=1)].copy()
    return out.reset_index(drop=True)


def genetic_evidence_table(mapping: pd.DataFrame, genetic: pd.DataFrame | set,
                           target_col: str = "targetId", disease_col: str = "diseaseId") -> pd.DataFrame:
    """Per-trial genetic evidence as in the authors' code.

    ``genetic``: Open Targets ``association_by_datatype_direct`` rows already
    filtered to ``datatypeId == 'genetic_association'`` (any row counts — no
    score threshold), or a set of (targetId, diseaseId) pairs. Returns
    ``nct_id, n_pairs, n_with_evidence, has_genetic_evidence_any,
    fraction_genetic_evidence`` over the trial's unique (target, disease)
    pairs.
    """
    if isinstance(genetic, (set, frozenset)):
        gp = pd.DataFrame(list(genetic), columns=[target_col, disease_col])
    else:
        gp = genetic[[target_col, disease_col]]
    gp = gp.drop_duplicates().assign(_g=1)
    pairs = mapping[["nct_id", target_col, disease_col]].drop_duplicates()
    pairs = pairs.merge(gp, on=[target_col, disease_col], how="left")
    pairs["_g"] = pairs["_g"].fillna(0).astype(int)
    t = pairs.groupby("nct_id")["_g"].agg(n_pairs="count", n_with_evidence="sum",
                                           has_genetic_evidence_any="max").reset_index()
    t["fraction_genetic_evidence"] = t["n_with_evidence"] / t["n_pairs"]
    return t


#: Therapeutic areas dropped / collapsed when building the TA combination
#: (authors' ``build_ta_combo``).
TA_DROP = frozenset({"EFO_0000651", "GO_0008150", "EFO_0001444", "EFO_0002571"})
TA_COLLAPSE = {"EFO_0009605": "EFO_0001379", "MONDO_0002025": "EFO_0000618"}


def drugtype_combo(mapping: pd.DataFrame, drug_types: pd.Series | dict) -> pd.DataFrame:
    """``nct_id, drugType_combo``: sorted unique drug types joined with '-'
    ('Unknown' when none), from ``drugId`` -> ``drugType``."""
    dt = mapping["drugId"].map(drug_types)
    combo = dt.groupby(mapping["nct_id"]).agg(
        lambda x: "-".join(sorted(x.dropna().unique())) if x.notna().any() else "Unknown")
    return combo.rename("drugType_combo").rename_axis("nct_id").reset_index()


def ta_combo(mapping: pd.DataFrame, disease: pd.DataFrame) -> pd.DataFrame:
    """``nct_id, ta_combo``: union of the cleaned therapeutic areas of all the
    trial's diseases (junk areas dropped, two collapsed), as names sorted by
    name and joined with '|' ('Unknown' when none). ``disease`` has
    ``id, name, therapeuticAreas``."""
    names = disease.set_index("id")["name"].to_dict()
    tas = disease.set_index("id")["therapeuticAreas"]
    m = mapping[["nct_id", "diseaseId"]].copy()
    m["tas"] = m["diseaseId"].map(tas)

    def combo(series) -> str:
        allt: set[str] = set()
        for v in series.dropna():
            if not hasattr(v, "__iter__") or isinstance(v, str):
                continue
            for t in v:
                if t in TA_DROP:
                    continue
                allt.add(TA_COLLAPSE.get(t, t))
        if not allt:
            return "Unknown"
        return "|".join(names.get(t, t) for t in sorted(allt, key=lambda x: names.get(x, x)))

    out = m.groupby("nct_id")["tas"].agg(combo)
    return out.rename("ta_combo").rename_axis("nct_id").reset_index()


def mixed_covariates(outcome_type: str, sub: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fixed-effect covariates of the authors' GLMMs for one analysis.

    ``year_z`` (z-scored years since the earliest trial, within the subset)
    always; phase dummies (Phase I reference) for status and AE outcomes;
    ``phase3`` only for the Phase II/III endpoint subset; nothing else for
    phase outcomes (phase *is* the outcome) and Phase I→II.
    """
    yr = sub["years_from_earliest"].astype(float).to_numpy()
    sd = yr.std()
    cov = {"year_z": (yr - yr.mean()) / (sd if sd > 0 else 1.0)}
    ph = pd.to_numeric(sub["phase"], errors="coerce")
    if outcome_type in ("status", "ae"):
        for k in (2, 3, 4):
            cov[f"phase{k}"] = (ph == float(k)).astype(float).to_numpy()
    elif outcome_type == "endpoint":
        cov["phase3"] = (ph == 3.0).astype(float).to_numpy()
    return cov


def build_outcomes(labels: pd.DataFrame, mapping: pd.DataFrame, *, endpoint_coding: str = "authors",
                   stop_reason_source: str = "chembl_first") -> pd.DataFrame:
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

    * ``primary_success`` / ``secondary_success`` / ``either_success`` —
      ``endpoint_coding="authors"`` (default; the authors' code): Phase II/III
      trials only, POSITIVE → 1 and *everything else* (NEGATIVE, UNKNOWN,
      missing) → 0; other phases NaN. ``"strict"``: POSITIVE → 1,
      NEGATIVE → 0, otherwise NaN, any phase.
    * ``phase_ge_2`` / ``phase_ge_3`` / ``phase_ge_4`` — the authors' Phase
      II+/III+/IV outcomes: the trial's *own* phase ≥ k (cross-sectional).
    * ``status_terminated`` / ``status_withdrawn`` / ``status_suspended`` —
      status equals that value (vs every other trial).
    * ``stop_<category>`` for each of :data:`STOP_REASON_CATEGORIES` — trials
      with stop categories (hierarchy ``stop_reason_source``, default
      ChEMBL first then Virtual Biotech, as in the authors' code): 1 if the
      category is among them, else 0; NaN for trials without categories.
    * ``phase1_to_2`` — for phase ≤ 1 trials: EVER → 1, NEVER → 0.
    * ``stopped_early`` — Terminated/Withdrawn/Suspended → 1, Completed → 0.
    * ``stopped_negative`` — stopped with a stop category mentioning
      Negative, Safety or side effects → 1; Completed → 0; stopped for other
      reasons → NaN.  Categories follow ``stop_reason_source``
      (``"chembl_first"``: ``studyStopReasonCategories`` when present, else
      ``virtualbiotech_stop_reason_categories``; ``"vb_first"``: reversed).
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
    if endpoint_coding not in ("authors", "strict"):
        raise ValueError("endpoint_coding must be 'authors' or 'strict'")
    if stop_reason_source not in ("chembl_first", "vb_first"):
        raise ValueError("stop_reason_source must be 'chembl_first' or 'vb_first'")
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

    def endpoint_raw(col):
        if col not in lab:
            return pd.Series(pd.NA, index=ids, dtype="string")
        return lab[col].reindex(ids).astype("string").str.strip().str.upper()

    prim, sec = endpoint_raw("primary_endpoint_result"), endpoint_raw("secondary_endpoint_result")
    if endpoint_coding == "authors":
        p23 = out["phase_num"].isin([2.0, 3.0])
        for name, v in (("primary_success", prim.eq("POSITIVE")), ("secondary_success", sec.eq("POSITIVE")),
                        ("either_success", prim.eq("POSITIVE") | sec.eq("POSITIVE"))):
            out[name] = v.fillna(False).astype(float).where(p23)
    else:
        codes = {"POSITIVE": 1.0, "NEGATIVE": 0.0}
        out["primary_success"] = prim.map(codes).astype(float)
        out["secondary_success"] = sec.map(codes).astype(float)
        e = np.fmax(out["primary_success"], out["secondary_success"])
        out["either_success"] = e.where(~(prim.eq("POSITIVE") | sec.eq("POSITIVE")).fillna(False), 1.0)

    pn_ok = out["phase_num"].notna()
    for k in (2, 3, 4):
        out[f"phase_ge_{k}"] = (out["phase_num"] >= k).astype(float).where(pn_ok)
    st_ok = status.notna()
    for v in ("Terminated", "Withdrawn", "Suspended"):
        out[f"status_{v.lower()}"] = (status == v).astype(float).where(st_ok)

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
    pairs_ = list(zip(vb.to_numpy(dtype=object), ot.to_numpy(dtype=object)))
    texts = [_stop_text(a, b) if stop_reason_source == "vb_first" else _stop_text(b, a) for a, b in pairs_]
    cats = [stop_categories(b, a, stop_reason_source) for a, b in pairs_]
    for c in STOP_REASON_CATEGORIES:
        out[AUTHORS_OUTCOME_COLUMNS[c]] = [np.nan if cs is None else float(c in cs) for cs in cats]
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


# ---------------------------------------------------------------------------
# Binary contrasts and multivariable models (authors' figure2_binary_tau_analysis.py
# and figure2_combined_genetic_sc_analysis.py)
# ---------------------------------------------------------------------------


def kmeans_threshold(values, seed: int = 42) -> tuple[float, tuple[float, float]]:
    """K-means (k = 2) on a 1-D distribution: (midpoint threshold, (low, high) centres)."""
    from sklearn.cluster import KMeans

    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    km = KMeans(n_clusters=2, random_state=seed, n_init=10).fit(v.reshape(-1, 1))
    c = sorted(float(x) for x in km.cluster_centers_.ravel())
    return (c[0] + c[1]) / 2, (c[0], c[1])


def contingency_or(x, y) -> dict:
    """Odds ratio of a 2×2 table (x = exposure 0/1, y = outcome 0/1), Wald CI on
    log OR, normal-approximation p — the authors' ``compute_or_from_contingency``.
    Also returns rates, counts, ``fold`` (rate_hi / rate_lo) and ``pct_change``."""
    from scipy import stats as sstats

    x = np.asarray(x)
    y = np.asarray(y)
    a = int(((x == 1) & (y == 1)).sum())
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    d = int(((x == 0) & (y == 0)).sum())
    rate_hi = a / (a + b) if a + b else 0.0
    rate_lo = c / (c + d) if c + d else 0.0
    if min(a, b, c, d) > 0:
        orr = a * d / (b * c)
        se = float(np.sqrt(1 / a + 1 / b + 1 / c + 1 / d))
        lo, hi = float(np.exp(np.log(orr) - 1.96 * se)), float(np.exp(np.log(orr) + 1.96 * se))
        p = float(2 * sstats.norm.sf(abs(np.log(orr) / se)))
    else:
        orr = lo = hi = p = float("nan")
    fold = rate_hi / rate_lo if rate_lo > 0 else float("nan")
    return dict(OR=float(orr), CI_lower=lo, CI_upper=hi, p_value=p, rate_hi=rate_hi, rate_lo=rate_lo,
                n_hi=a + b, n_lo=c + d, n_pos_hi=a, n_pos_lo=c, fold=fold, pct_change=(fold - 1) * 100)


def _standardize_columns(df: pd.DataFrame, cols: Sequence[str], standardize: Sequence[bool]) -> np.ndarray:
    X = np.column_stack([df[c].astype(float).to_numpy() for c in cols])
    for i, st_ in enumerate(standardize):
        if st_:
            col = X[:, i]
            sd = col.std()
            X[:, i] = (col - col.mean()) / (sd if sd > 0 else 1.0)
    return X


def multivariable_logistic(df: pd.DataFrame, cols: Sequence[str], outcome: str,
                           standardize: Sequence[bool] | None = None) -> list[dict]:
    """Binomial GLM of ``outcome`` on several predictors (z-scored where
    ``standardize``; a 0/1 flag stays raw), one dict per predictor."""
    import statsmodels.api as sm

    cols = list(cols)
    standardize = list(standardize) if standardize is not None else [True] * len(cols)
    d = df[[*cols, outcome]].dropna()
    X = sm.add_constant(_standardize_columns(d, cols, standardize), has_constant="add")
    res = _fit_logit(d[outcome].astype(float).to_numpy(), X)
    ci = np.asarray(res.conf_int())
    out = []
    for i, c in enumerate(cols, start=1):
        b = float(np.asarray(res.params)[i])
        out.append(dict(feature=c, coefficient=b, OR=float(np.exp(b)), CI_lower=float(np.exp(ci[i, 0])),
                        CI_upper=float(np.exp(ci[i, 1])), p_value=float(np.asarray(res.pvalues)[i]),
                        se=float(np.asarray(res.bse)[i]), n=len(d)))
    return out


def multivariable_beta(df: pd.DataFrame, cols: Sequence[str], outcome_pct: str,
                       standardize: Sequence[bool] | None = None, squeeze: str | None = None) -> list[dict]:
    """Beta regression (:func:`betareg_fit`) of an AE percentage on several predictors."""
    cols = list(cols)
    standardize = list(standardize) if standardize is not None else [True] * len(cols)
    d = df[[*cols, outcome_pct]].dropna()
    y = _squeeze(d[outcome_pct].astype(float).to_numpy() / 100.0, squeeze or BETA_SQUEEZE, len(d))
    X = np.column_stack([np.ones(len(d)), _standardize_columns(d, cols, standardize)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = betareg_fit(y, X)
    out = []
    for i, c in enumerate(cols, start=1):
        b, se = float(fit["params"][i]), float(fit["se"][i])
        out.append(dict(feature=c, coefficient=b, CI_lower=b - 1.96 * se, CI_upper=b + 1.96 * se,
                        p_value=float(fit["p"][i]), se=se, n=len(d), engine="betareg_py"))
    return out


def target_level_table(labels_with_targets: pd.DataFrame, gene_feature: pd.Series, threshold: float,
                       thresholds=((("Ever reached Phase II+"), 2.0), ("Ever reached Phase III+", 3.0),
                                   ("Ever reached Phase IV", 4.0))) -> pd.DataFrame:
    """Target-level analysis behind the paper's "48% more likely to ever reach Phase IV".

    For each target: highest phase over all its trials (labels ⋈ mapping)
    and number of trials; targets are split by ``gene_feature >= threshold``
    (the *trial-level* k-means threshold). Unadjusted 2×2 OR (Wald) and a
    logistic model adjusted for log(1 + n_trials), as the authors' code.
    """
    import statsmodels.api as sm

    lt = labels_with_targets
    tl = lt.groupby("targetId").agg(max_phase=("phase", "max"), n_trials=("nct_id", "nunique")).reset_index()
    g = gene_feature.dropna()
    tl = tl[tl["targetId"].isin(g.index)].copy()
    tl["hi"] = (tl["targetId"].map(g) >= threshold).astype(int)
    rows = []
    for name, k in thresholds:
        y = (tl["max_phase"] >= k).astype(int).to_numpy()
        r = contingency_or(tl["hi"].to_numpy(), y)
        rows.append(dict(outcome=name, model="unadjusted", threshold=threshold, n_total=len(tl), **r))
    for name, k in thresholds:
        y = (tl["max_phase"] >= k).astype(float).to_numpy()
        X = sm.add_constant(np.column_stack([tl["hi"].astype(float), np.log1p(tl["n_trials"].astype(float))]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = sm.Logit(y, X).fit(disp=False)
        ci = np.asarray(m.conf_int())
        un = next(r for r in rows if r["outcome"] == name)
        rows.append(dict(outcome=name, model="adjusted_n_trials", threshold=threshold, n_total=len(tl),
                         OR=float(np.exp(m.params[1])), CI_lower=float(np.exp(ci[1, 0])),
                         CI_upper=float(np.exp(ci[1, 1])), p_value=float(m.pvalues[1]),
                         rate_hi=un["rate_hi"], rate_lo=un["rate_lo"], fold=un["fold"],
                         n_hi=int(tl["hi"].sum()), n_lo=int((tl["hi"] == 0).sum()),
                         covariate_log_ntrials_OR=float(np.exp(m.params[2])),
                         covariate_log_ntrials_p=float(m.pvalues[2])))
    out = pd.DataFrame(rows)
    for model in ("unadjusted", "adjusted_n_trials"):
        msk = out["model"] == model
        out.loc[msk, "fdr_adjusted_p"] = benjamini_hochberg(out.loc[msk, "p_value"])
    return out


def ae_group_comparison(df: pd.DataFrame, group_col: str, organs=AE_ORGAN_SYSTEMS,
                        min_n: int = 30) -> pd.DataFrame:
    """Serious-AE % by organ system, group 1 vs 0: means, medians, fold,
    Welch t-test and Mann–Whitney U, BH over organs (authors' Part 2/4)."""
    from scipy import stats as sstats

    rows = []
    for col, label in organs:
        if col not in df:
            continue
        v = df[df[col].notna() & df[group_col].notna()]
        if len(v) < min_n:
            continue
        hi, lo = v.loc[v[group_col] == 1, col], v.loc[v[group_col] == 0, col]
        t, pt = sstats.ttest_ind(hi, lo, equal_var=False)
        _, pm = sstats.mannwhitneyu(hi, lo, alternative="two-sided")
        mh, ml = float(hi.mean()), float(lo.mean())
        rows.append(dict(outcome=label, ae_column=col, n_total=len(v), n_hi=len(hi), n_lo=len(lo),
                         mean_hi=mh, mean_lo=ml, median_hi=float(hi.median()), median_lo=float(lo.median()),
                         diff=mh - ml, fold=mh / ml if ml > 0 else float("nan"), t_statistic=float(t),
                         p_value_ttest=float(pt), p_value_mannwhitney=float(pm)))
    out = pd.DataFrame(rows)
    if len(out):
        out["fdr_adjusted_p_ttest"] = benjamini_hochberg(out["p_value_ttest"])
        out["fdr_adjusted_p_mannwhitney"] = benjamini_hochberg(out["p_value_mannwhitney"])
    return out
