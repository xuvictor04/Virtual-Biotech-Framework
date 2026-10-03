"""Relative-importance (variance decomposition) analysis from case study 3.

Implements the Supplementary Methods section "Variance decomposition of STAT
activity": Lindeman-Merenda-Gold (LMG) decomposition of the variance in an
outcome (e.g. per-sample STAT3/STAT1 activity) explained by the gp130-family
receptors, by exhaustive enumeration of all 2^p predictor subsets with Shapley
averaging over orderings; R^2 from OLS or, when a grouping (patient) is given,
the Nakagawa marginal R^2 of a random-intercept mixed model; 95% CIs from a
patient-level cluster bootstrap using OLS for tractability.

MuMIn note: :func:`marginal_r2_mixed` re-implements
``MuMIn::r.squaredGLMM`` (marginal R^2) in Python: var(X b) with R's ``var``
(ddof=1, ``var_ddof``) over var(X b) + sigma^2_patient + sigma^2_resid, the
model fitted by **maximum likelihood** (``reml=False``) as in the authors'
``osmr/code/02b_stat1_dominance.py`` (``lmer(..., REML=FALSE)``; Zenodo
archive). ``engine="MuMIn"`` (or ``"auto"`` with rpy2 + lme4 + MuMIn
installed) fits ``lme4::lmer`` and calls ``MuMIn::r.squaredGLMM`` itself; the
engine is recorded in ``attrs['r2_engine']``. :func:`stat1_dominance`
reproduces the authors' Fig. 5B table from their pseudobulk file.
"""

from __future__ import annotations

import warnings
from math import factorial
from typing import Sequence

import numpy as np
import pandas as pd

from ._utils import fit_mixedlm

__all__ = ["GP130_RECEPTORS", "subset_r2_ols", "marginal_r2_mixed", "lmg_from_r2",
           "lmg_shares", "bootstrap_lmg", "r_packages_available", "stat1_dominance",
           "DOMINANCE_CELL_STATES", "RECEPTORS_REDUCED", "lmm_random_intercept_ml"]


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


def _r2_mumin(df: pd.DataFrame, outcome: str, predictors: Sequence[str], groups: str,
              reml: bool = False) -> float:
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
    r2 = ro.r(f"MuMIn::r.squaredGLMM(lme4::lmer(y ~ {rhs} + (1|g), data=vbt_df, "
              f"REML={'TRUE' if reml else 'FALSE'}))")
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


def lmm_random_intercept_ml(y: np.ndarray, X: np.ndarray, groups) -> dict:
    """Maximum-likelihood fit of ``y ~ X + (1|groups)`` (lme4 ``lmer(REML=FALSE)``).

    The likelihood is profiled over the variance ratio λ = σ²_group / σ²_resid
    (closed-form GLS for β and σ²_resid given λ; per-group Woodbury inverse),
    and λ is optimised on [0, ∞) including the boundary — so tiny or singular
    designs (lmer's ``boundary (singular) fit``) are handled where
    statsmodels' ``MixedLM`` fails. Returns ``beta``, ``var_group``,
    ``var_resid``, ``loglik``.
    """
    from scipy import optimize

    y = np.asarray(y, float)
    X = np.asarray(X, float)
    codes, uniq = pd.factorize(pd.Series(np.asarray(groups)).astype(str))
    idx = [np.where(codes == g)[0] for g in range(len(uniq))]
    n = len(y)

    def solve(lam):
        XtVX = np.zeros((X.shape[1], X.shape[1]))
        XtVy = np.zeros(X.shape[1])
        logdet = 0.0
        for ii in idx:
            Xg, yg = X[ii], y[ii]
            c = lam / (1 + lam * len(ii))
            sx, sy = Xg.sum(0), yg.sum()
            XtVX += Xg.T @ Xg - c * np.outer(sx, sx)
            XtVy += Xg.T @ yg - c * sx * sy
            logdet += np.log1p(lam * len(ii))
        beta = np.linalg.lstsq(XtVX, XtVy, rcond=None)[0]
        r = y - X @ beta
        q = 0.0
        for ii in idx:
            rg = r[ii]
            q += rg @ rg - lam / (1 + lam * len(ii)) * rg.sum() ** 2
        s2 = max(q / n, 1e-300)
        dev = n * np.log(2 * np.pi * s2) + logdet + n
        return dev, beta, s2

    f = lambda t: solve(np.exp(t))[0]  # noqa: E731
    best = optimize.minimize_scalar(f, bounds=(-25.0, 15.0), method="bounded", options={"xatol": 1e-10})
    lam = float(np.exp(best.x))
    dev0 = solve(0.0)[0]
    if dev0 <= best.fun:
        lam = 0.0
    dev, beta, s2 = solve(lam)
    return dict(beta=beta, var_group=lam * s2, var_resid=s2, loglik=-dev / 2)


def marginal_r2_mixed(df: pd.DataFrame, outcome: str, predictors: Sequence[str], groups: str,
                      engine: str = "statsmodels", *, reml: bool = False, var_ddof: int = 1) -> float:
    """Nakagawa & Schielzeth marginal R^2 of ``outcome ~ predictors + (1|groups)``:
    var(X b) / (var(X b) + sigma^2_group + sigma^2_resid) via statsmodels
    ``MixedLM`` (REML) or :func:`lmm_random_intercept_ml` (ML, the default, as the
    authors' ``lmer(REML=FALSE)``; ``var``
    with ``var_ddof`` = 1 as R), or ``MuMIn::r.squaredGLMM`` with
    ``engine="MuMIn"`` (rpy2)."""
    import statsmodels.api as sm

    if engine == "MuMIn":
        return _r2_mumin(df, outcome, predictors, groups, reml=reml) if len(predictors) else 0.0

    y = df[outcome].to_numpy(float)
    g = df[groups].to_numpy()
    if len(predictors):
        X = sm.add_constant(df[list(predictors)].to_numpy(float), has_constant="add")
    else:
        X = np.ones((len(df), 1))
    if not reml:  # profiled ML, robust on tiny designs (lmer REML=FALSE)
        fit = lmm_random_intercept_ml(y, X, g)
        var_f = float(np.var(X @ fit["beta"], ddof=var_ddof)) if len(predictors) else 0.0
        denom = var_f + fit["var_group"] + fit["var_resid"]
        return var_f / denom if denom > 0 else 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.MixedLM(y, X, groups=g)
    res = fit_mixedlm(model, reml=reml)
    fe = np.asarray(res.fe_params)
    var_f = float(np.var(X @ fe, ddof=var_ddof)) if len(predictors) else 0.0
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
               groups: str | None = None, r2_engine: str = "statsmodels", *, reml: bool = False,
               var_ddof: int = 1) -> pd.DataFrame:
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
            r2[mask] = (marginal_r2_mixed(dat, outcome, sub, groups, engine=r2_engine, reml=reml,
                                          var_ddof=var_ddof) if sub else 0.0)
    shares = lmg_from_r2(r2, p)
    total = r2[(1 << p) - 1]
    out = pd.DataFrame({"predictor": predictors, "share": shares})
    out["pct_of_r2"] = out["share"] / total * 100 if total > 0 else np.nan
    out = out.sort_values("share", ascending=False, kind="stable").reset_index(drop=True)
    out["rank"] = np.arange(1, p + 1)
    engine_used = "ols" if groups is None else ("R:MuMIn" if r2_engine == "MuMIn" else
                                                 ("statsmodels_mixedlm_reml" if reml else "python_lmm_ml"))
    out["engine"] = engine_used
    out.attrs["r2_engine"] = engine_used
    out.attrs["r2_total"] = total
    out.attrs["n"] = len(dat)
    out.attrs["r2_type"] = "ols" if groups is None else "marginal_mixed"
    return out


def bootstrap_lmg(df: pd.DataFrame, outcome: str, predictors: Sequence[str] = GP130_RECEPTORS,
                  cluster_col: str | None = "patient", n_boot: int = 2000, seed: int = 42,
                  ci: float = 0.95) -> pd.DataFrame:
    """Cluster (patient-level) bootstrap CIs for OLS LMG shares (Methods,
    "Variance decomposition of STAT activity": 2,000 resamples, OLS used inside
    the bootstrap for tractability).

    Returns ``predictor, share, ci_low, ci_high, pct_of_r2, pct_ci_low,
    pct_ci_high, p_top`` (fraction of resamples in which the predictor ranks
    first); ``attrs`` carries the R^2 point estimate and CI. With the authors'
    seed (42) and cluster order (first appearance) the resamples are the
    same as in ``02b_stat1_dominance.py`` (``rng.choice(n, n)`` draws the
    same integers as ``rng.integers(0, n, n)``).
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


#: Cell states of the authors' STAT1 dominance analysis (OSMR >= 5% detection in NR-Post).
DOMINANCE_CELL_STATES = ("THY1pos_FAPpos_PDPNpos_fibroblast", "Fibroblast", "LP_fibroblast", "C3pos_fibroblast",
                         "Epi_fibroblast", "Myofibroblast", "Pericyte", "Endothelium", "Cycling_stroma", "Glial")
#: Reduced receptor set for cell states with fewer than 7 patients.
RECEPTORS_REDUCED = ("OSMR", "IL6ST", "LIFR", "IL11RA")


def stat1_dominance(pb: pd.DataFrame, *, y_col: str = "STAT1_clean", cell_col: str = "major",
                    patient_col: str = "Patient", cell_states: Sequence[str] = DOMINANCE_CELL_STATES,
                    receptors: Sequence[str] = GP130_RECEPTORS, receptors_reduced: Sequence[str] = RECEPTORS_REDUCED,
                    min_patients_full: int = 7, min_patients: int = 5, n_boot: int = 2000, seed: int = 42,
                    r2_engine: str = "statsmodels", subset: dict | None = None) -> pd.DataFrame:
    """The authors' Fig. 5B LMG table (``02b_stat1_dominance.py``) from a pseudobulk table.

    For each cell state among NR-Post samples (``subset``, default
    ``Remission_status == 'Non_Remission'`` and ``Treatment == 'Post'``):
    skip < 6 samples or < ``min_patients`` patients; use all receptors
    (``>= min_patients_full`` patients) or the reduced set, dropping constant
    ones; z-score receptors (``std`` ddof=1, + 1e-9); point LMG shares from
    the ML random-intercept marginal R^2 (:func:`lmg_shares`); patient
    cluster-bootstrap CIs (OLS on the unscaled receptors, ``seed`` 42; CI only
    with >= 100 valid resamples). Returns one row per (cell state, receptor):
    ``cell_state, n_samples, n_patients, predictor_set, receptor, lmg_share,
    lmg_pct, ci_lo, ci_hi, r2m_full, engine``.
    """
    subset = {"Remission_status": "Non_Remission", "Treatment": "Post"} if subset is None else subset
    d = pb.copy()
    for k, v in subset.items():
        d = d[d[k] == v]
    rows = []
    for ct in cell_states:
        sub = d[d[cell_col] == ct].copy()
        n_pat = sub[patient_col].nunique()
        if len(sub) < 6 or n_pat < min_patients:
            continue
        recs = list(receptors if n_pat >= min_patients_full else receptors_reduced)
        recs = [r for r in recs if r in sub.columns and sub[r].std() > 1e-9]
        if len(recs) < 2:
            continue
        z = sub.copy()
        for r in recs:
            z[r] = (z[r] - z[r].mean()) / (z[r].std() + 1e-9)
        pt = lmg_shares(z, y_col, recs, groups=patient_col, r2_engine=r2_engine).set_index("predictor")
        total = float(pt.attrs.get("r2_total", np.nan))
        boot = bootstrap_lmg(sub, y_col, recs, cluster_col=patient_col, n_boot=n_boot, seed=seed).set_index("predictor")
        for r in recs:
            rows.append(dict(cell_state=ct, n_samples=len(sub), n_patients=int(n_pat), predictor_set="|".join(recs),
                             receptor=r, lmg_share=float(pt.loc[r, "share"]),
                             lmg_pct=float(pt.loc[r, "share"] / total * 100) if total > 0 else np.nan,
                             ci_lo=float(boot.loc[r, "ci_low"]), ci_hi=float(boot.loc[r, "ci_high"]),
                             r2m_full=total, engine=pt.attrs.get("r2_engine")))
    return pd.DataFrame(rows)
