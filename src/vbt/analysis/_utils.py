"""Shared helpers for :mod:`vbt.analysis` (lazy imports, multiple-testing correction)."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

_EXTRA_FOR = {
    "anndata": "singlecell",
    "scanpy": "singlecell",
    "harmonypy": "singlecell",
    "pydeseq2": "singlecell",
    "decoupler": "singlecell",
    "liana": "singlecell",
    "cellxgene_census": "singlecell",
    "cell2location": "spatial",
    "lifelines": "survival",
    "rpy2": "r",
}


def require(module: str, extra: str | None = None) -> Any:
    """Import ``module`` lazily, raising an ImportError that names the pip extra."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on environment
        top = module.split(".")[0]
        extra = extra or _EXTRA_FOR.get(top, "all")
        raise ImportError(
            f"'{top}' is required for this analysis but is not installed. "
            f"Install it with: pip install 'vbt-harness[{extra}]'"
        ) from exc


def has_module(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def bh_fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values; NaNs are propagated and ignored."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0, 1)
    out[ok] = adj
    return out


def zscore(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x, ddof=1) if x.size > 1 else 0.0
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(x)
    return (x - np.nanmean(x)) / sd


def fit_mixedlm(model, reml: bool = True, methods=("bfgs", "lbfgs"),
                fallback_methods=("powell", "nm")):
    """Fit a statsmodels ``MixedLM`` robustly.

    statsmodels' default L-BFGS occasionally returns a degenerate boundary
    solution (``llf = inf``, random-effect variance 0, wrong fixed effects) when
    the fixed effect is nested within groups. We fit with several optimisers
    and keep the finite solution with the highest (restricted) log-likelihood.
    """
    import warnings

    fits = []

    def _try(methods_):
        for m in methods_:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    f = model.fit(reml=reml, method=m, maxiter=2000)
            except Exception:
                continue
            bse = np.asarray(f.bse_fe) if hasattr(f, "bse_fe") else np.asarray(f.bse)
            if np.isfinite(f.llf) and np.all(np.isfinite(np.asarray(f.fe_params))) \
                    and np.all(np.isfinite(bse)):
                fits.append(f)

    _try(methods)
    if not fits:
        _try(fallback_methods)
    if not fits:
        raise RuntimeError("MixedLM failed to converge with all optimisers")
    return max(fits, key=lambda f: f.llf)
