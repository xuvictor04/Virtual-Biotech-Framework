"""Relative-importance (variance decomposition) analysis from case study 3.

Implements the Supplementary Methods section "Variance decomposition of STAT
activity": Lindeman-Merenda-Gold (LMG) decomposition of the variance in an
outcome (e.g. per-sample STAT3/STAT1 activity) explained by the gp130-family
receptors, by exhaustive enumeration of all 2^p predictor subsets with Shapley
averaging over orderings; R^2 from OLS or, when a grouping (patient) is given,
the Nakagawa marginal R^2 of a random-intercept mixed model; 95% CIs from a
patient-level cluster bootstrap using OLS for tractability.

MuMIn note: :func:`marginal_r2_mixed` re-implements
``MuMIn::r.squaredGLMM`` (marginal R^2) in Python. The fixed-effect variance is
``np.var(X b)`` (population variance, ddof=0) whereas R's ``var`` uses ddof=1;
the difference is a factor n/(n-1) on var(Xb) only (negligible for the
per-sample tables used here). ``engine="MuMIn"`` (or ``"auto"`` with rpy2 + lme4
+ MuMIn installed) fits ``lme4::lmer`` and calls ``MuMIn::r.squaredGLMM``
itself; the engine is recorded in ``attrs['r2_engine']``.
"""

from __future__ import annotations

import warnings
from math import factorial
from typing import Sequence

import numpy as np
import pandas as pd

from ._utils import fit_mixedlm

__all__ = ["GP130_RECEPTORS", "subset_r2_ols", "marginal_r2_mixed", "lmg_from_r2",
           "lmg_shares", "bootstrap_lmg", "r_packages_available"]


def r_packages_available(*pkgs: str) -> bool:
    """rpy2 importable and every R package installed (``isinstalled``)."""
    try:
        from rpy2.robjects.packages import isinstalled  # type: ignore
    except Exception:  # noqa: BLE001
        return False
    try:
        return all(isinstalled(p) for p in pkgs)
    except Exception:  # noqa: BLE001
        return False


def _r2_mumin(df: pd.DataFrame, outcome: str, predictors: Sequence[str], groups: str) -> float:
    import rpy2.robjects as ro  # type: ignore
    from rpy2.robjects import pandas2ri  # type: ignore
    from rpy2.robjects.conversion import localconverter  # type: ignore

    safe = pd.DataFrame({"y": df[outcome].to_numpy(float), "g": df[groups].astype(str).to_numpy()})
    for i, c in enumerate(predictors):
        safe[f"x{i}"] = df[c].to_numpy(float)
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["vbt_df"] = ro.conversion.py2rpy(safe)
    rhs = " + ".join(f"x{i}" for i in range(len(predictors))) or "1"
    ro.r("suppressPackageStartupMessages({library(lme4); library(MuMIn)})")
    r2 = ro.r(f"MuMIn::r.squaredGLMM(lme4::lmer(y ~ {rhs} + (1|g), data=vbt_df, REML=TRUE))")
    return float(r2[0])

GP130_RECEPTORS: list[str] = ["OSMR", "IL6ST", "LIFR", "IL6R", "IL11RA", "IL27RA", "CNTFR"]


def subset_r2_ols(X: np.ndarray, y: np.ndarray) -> dict[int, float]:
    """R^2 of OLS (with intercept) for every predictor subset, keyed by bitmask.

    Uses the correlation-matrix identity ``R^2_S = r_yS' R_SS^+ r_yS`` which is
    exact for OLS with an intercept and very fast for exhaustive enumeration.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    p = X.shape[1]
    Z = np.column_stack([X, y])
    Z = Z - Z.mean(axis=0)
    sd = Z.std(axis=0)
    sd[sd == 0] = 1.0
    C = (Z.T @ Z) / (Z.shape[0] * np.outer(sd, sd))
    Rxx, rxy = C[:p, :p], C[:p, p]
    out = {0: 0.0}
    for mask in range(1, 1 << p):
        idx = [j for j in range(p) if mask >> j & 1]
        r = rxy[idx]
        coef = np.linalg.lstsq(Rxx[np.ix_(idx, idx)], r, rcond=None)[0]
        out[mask] = float(np.clip(r @ coef, 0.0, 1.0))
    return out


def marginal_r2_mixed(df: pd.DataFrame, outcome: str, predictors: Sequence[str], groups: str,
                      engine: str = "statsmodels") -> float:
    """Nakagawa & Schielzeth marginal R^2 of ``outcome ~ predictors + (1|groups)``:
    var(X b) / (var(X b) + sigma^2_group + sigma^2_resid), REML fit via
    statsmodels ``MixedLM`` (``np.var``, ddof=0 — see the module note), or
    ``MuMIn::r.squaredGLMM`` with ``engine="MuMIn"`` (rpy2)."""
    import statsmodels.api as sm

    if engine == "MuMIn":
        return _r2_mumin(df, outcome, predictors, groups) if len(predictors) else 0.0

    y = df[outcome].to_numpy(float)
    g = df[groups].to_numpy()
    if len(predictors):
        X = sm.add_constant(df[list(predictors)].to_numpy(float), has_constant="add")
    else:
        X = np.ones((len(df), 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.MixedLM(y, X, groups=g)
    res = fit_mixedlm(model)
    fe = np.asarray(res.fe_params)
    var_f = float(np.var(X @ fe)) if len(predictors) else 0.0
    var_g = float(np.asarray(res.cov_re).ravel()[0])
    var_e = float(res.scale)
    denom = var_f + max(var_g, 0.0) + var_e
    return var_f / denom if denom > 0 else 0.0


def lmg_from_r2(r2: dict[int, float], p: int) -> np.ndarray:
    """Shapley/LMG shares from subset R^2 values (bitmask -> R^2):
    ``share_j = sum_{S not containing j} |S|!(p-|S|-1)!/p! * (R^2(S+j) - R^2(S))``."""
    w = np.array([factorial(k) * factorial(p - k - 1) / factorial(p) for k in range(p)])
    shares = np.zeros(p)
    for mask, val in r2.items():
        k = bin(mask).count("1")
        for j in range(p):
            if mask >> j & 1:
                continue
            shares[j] += w[k] * (r2[mask | (1 << j)] - val)
    return shares


def lmg_shares(df: pd.DataFrame, outcome: str, predictors: Sequence[str] = GP130_RECEPTORS,
               groups: str | None = None, r2_engine: str = "statsmodels") -> pd.DataFrame:
    """LMG relative importance (Methods, "Variance decomposition of STAT activity").

    Exhaustively fits all 2^p predictor subsets; R^2 is OLS R^2, or the
    Nakagawa marginal R^2 of a random-intercept model when ``groups`` is given.
    Shares are Shapley values over predictor orderings and sum exactly to the
    full-model R^2 (``df.attrs['r2_total']``).

    ``r2_engine`` (mixed models only): ``"statsmodels"`` (default),
    ``"MuMIn"`` (rpy2 + lme4 + MuMIn) or ``"auto"`` (MuMIn when installed).

    Returns ``predictor, share, pct_of_r2, rank, engine`` sorted by share (descending).
    """
    if r2_engine == "auto":
        r2_engine = "MuMIn" if groups is not None and r_packages_available("lme4", "MuMIn") else "statsmodels"
    if r2_engine not in ("statsmodels", "MuMIn"):
        raise ValueError("r2_engine must be 'statsmodels', 'MuMIn' or 'auto'")
    predictors = list(predictors)
    cols = [outcome] + predictors + ([groups] if groups else [])
    dat = df[cols].dropna()
    p = len(predictors)
    if p == 0:
        raise ValueError("need at least one predictor")
    if groups is None:
        r2 = subset_r2_ols(dat[predictors].to_numpy(float), dat[outcome].to_numpy(float))
    else:
        r2 = {}
        for mask in range(1 << p):
            sub = [predictors[j] for j in range(p) if mask >> j & 1]
            r2[mask] = marginal_r2_mixed(dat, outcome, sub, groups, engine=r2_engine) if sub else 0.0
    shares = lmg_from_r2(r2, p)
    total = r2[(1 << p) - 1]
    out = pd.DataFrame({"predictor": predictors, "share": shares})
    out["pct_of_r2"] = out["share"] / total * 100 if total > 0 else np.nan
    out = out.sort_values("share", ascending=False, kind="stable").reset_index(drop=True)
    out["rank"] = np.arange(1, p + 1)
    engine_used = "ols" if groups is None else ("R:MuMIn" if r2_engine == "MuMIn" else "statsmodels_mixedlm_np_var")
    out["engine"] = engine_used
    out.attrs["r2_engine"] = engine_used
    out.attrs["r2_total"] = total
    out.attrs["n"] = len(dat)
    out.attrs["r2_type"] = "ols" if groups is None else "marginal_mixed"
    return out


def bootstrap_lmg(df: pd.DataFrame, outcome: str, predictors: Sequence[str] = GP130_RECEPTORS,
                  cluster_col: str | None = "patient", n_boot: int = 2000, seed: int = 0,
                  ci: float = 0.95) -> pd.DataFrame:
    """Cluster (patient-level) bootstrap CIs for OLS LMG shares (Methods,
    "Variance decomposition of STAT activity": 2,000 resamples, OLS used inside
    the bootstrap for tractability).

    Returns ``predictor, share, ci_low, ci_high, pct_of_r2, pct_ci_low,
    pct_ci_high, p_top`` (fraction of resamples in which the predictor ranks
    first); ``attrs`` carries the R^2 point estimate and CI.
    """
    predictors = list(predictors)
    cols = [outcome] + predictors + ([cluster_col] if cluster_col else [])
    dat = df[cols].dropna().reset_index(drop=True)
    p = len(predictors)
    X = dat[predictors].to_numpy(float)
    y = dat[outcome].to_numpy(float)
    point_r2 = subset_r2_ols(X, y)
    point = lmg_from_r2(point_r2, p)
    total = point_r2[(1 << p) - 1]
    rng = np.random.default_rng(seed)
    if cluster_col:
        codes, uniq = pd.factorize(dat[cluster_col])
        members = [np.where(codes == i)[0] for i in range(len(uniq))]
    else:
        members = [np.array([i]) for i in range(len(dat))]
    boots = np.full((n_boot, p), np.nan)
    boot_r2 = np.full(n_boot, np.nan)
    for b in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[i] for i in pick])
        r2b = subset_r2_ols(X[idx], y[idx])
        boots[b] = lmg_from_r2(r2b, p)
        boot_r2[b] = r2b[(1 << p) - 1]
    a = (1 - ci) / 2
    lo, hi = np.nanquantile(boots, [a, 1 - a], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = boots / boot_r2[:, None] * 100
    plo, phi = np.nanquantile(pct, [a, 1 - a], axis=0)
    top = np.bincount(np.argmax(boots, axis=1), minlength=p) / n_boot
    out = pd.DataFrame({"predictor": predictors, "share": point, "ci_low": lo, "ci_high": hi,
                        "pct_of_r2": point / total * 100 if total > 0 else np.nan,
                        "pct_ci_low": plo, "pct_ci_high": phi, "p_top": top})
    out = out.sort_values("share", ascending=False, kind="stable").reset_index(drop=True)
    out["engine"] = "ols_cluster_bootstrap"
    out.attrs.update({"r2_total": total, "r2_ci": tuple(np.nanquantile(boot_r2, [a, 1 - a])),
                      "n_boot": n_boot, "n_clusters": len(members)})
    return out
