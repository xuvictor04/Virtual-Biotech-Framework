"""Transcription-factor activity analysis from case study 3 (OSMR in ulcerative
colitis).

Implements the Supplementary Methods section "Transcription factor activity":
per-cell TF activities inferred with decoupler (multivariate linear model,
``mlm``, or univariate ``ulm``) on the CollecTRI regulon collection, with the
target IL6ST removed from the STAT1 regulon to avoid circularity with the
receptor under study; per TF and cell type a linear mixed model
``activity ~ condition + (1 | sample)`` fitted with lmerTest (Satterthwaite
df) and Benjamini-Hochberg correction within each cell type.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ._utils import bh_fdr, fit_mixedlm, has_module, require

__all__ = ["load_collectri", "remove_targets_from_regulon", "score_tf_activity",
           "tf_mixed_model_screen", "PAPER_REGULON_EXCLUSIONS"]

#: Regulon edits made in the paper (TF -> targets removed).
PAPER_REGULON_EXCLUSIONS: dict[str, list[str]] = {"STAT1": ["IL6ST"]}


def remove_targets_from_regulon(net: pd.DataFrame, genes: Iterable[str] | Mapping[str, Iterable[str]],
                                tfs: Iterable[str] | None = None, source_col: str = "source",
                                target_col: str = "target") -> pd.DataFrame:
    """Drop edges to ``genes`` from a decoupler-style network (``source``,
    ``target``, ``weight``).

    ``genes`` may be a list (removed from the regulons of ``tfs``, or from all
    TFs if ``tfs`` is None) or a mapping ``{tf: [targets]}`` such as
    :data:`PAPER_REGULON_EXCLUSIONS` (paper: IL6ST removed from STAT1).
    """
    if isinstance(genes, Mapping):
        drop = np.zeros(len(net), dtype=bool)
        for tf, tg in genes.items():
            drop |= (net[source_col] == tf).to_numpy() & net[target_col].isin(list(tg)).to_numpy()
    else:
        drop = np.array(net[target_col].isin(list(genes)), dtype=bool)
        if tfs is not None:
            drop &= net[source_col].isin(list(tfs)).to_numpy()
    return net.loc[~drop].reset_index(drop=True)


def load_collectri(organism: str = "human") -> pd.DataFrame:
    """Load the CollecTRI network with decoupler (v2 ``dc.op.collectri`` or v1
    ``dc.get_collectri``). Requires network access on first use."""
    dc = require("decoupler")
    if hasattr(dc, "op") and hasattr(dc.op, "collectri"):
        return dc.op.collectri(organism=organism)
    return dc.get_collectri(organism=organism, split_complexes=False)


def score_tf_activity(adata, method: str = "mlm", regulons: pd.DataFrame | None = None,
                      exclusions: Mapping[str, Iterable[str]] | None = PAPER_REGULON_EXCLUSIONS,
                      tmin: int = 5) -> pd.DataFrame:
    """Per-cell TF activity scores (cells x TFs) with decoupler ``mlm``/``ulm``
    on CollecTRI (Methods, "Transcription factor activity").

    ``adata.X`` should be log-normalised expression. ``regulons`` defaults to
    CollecTRI; ``exclusions`` are applied with
    :func:`remove_targets_from_regulon` (default: IL6ST removed from STAT1).
    """
    if method not in {"mlm", "ulm"}:
        raise ValueError("method must be 'mlm' or 'ulm'")
    dc = require("decoupler")
    net = regulons if regulons is not None else load_collectri()
    if exclusions:
        net = remove_targets_from_regulon(net, exclusions)
    if hasattr(dc, "mt") and hasattr(dc.mt, method):  # decoupler >= 2
        getattr(dc.mt, method)(data=adata, net=net, tmin=tmin)
        key = f"score_{method}"
    else:  # decoupler 1.x
        getattr(dc, f"run_{method}")(mat=adata, net=net, min_n=tmin, use_raw=False)
        key = f"{method}_estimate"
    scores = adata.obsm[key]
    if not isinstance(scores, pd.DataFrame):
        scores = pd.DataFrame(np.asarray(scores), index=adata.obs_names)
    return scores


def _lmm_statsmodels(data: pd.DataFrame):
    import statsmodels.formula.api as smf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm("y ~ x", data, groups=data["g"])
    fit = fit_mixedlm(model)
    return float(fit.params["x"]), float(fit.bse["x"]), float(fit.pvalues["x"])


def _lmm_lmertest(data: pd.DataFrame):
    require("rpy2", "r")
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    importr("lmerTest")
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["dat"] = data
    co = ro.r('coef(summary(lmerTest::lmer(y ~ x + (1|g), data=dat, REML=TRUE)))["x", ]')
    return float(co[0]), float(co[1]), float(co[4])


def tf_mixed_model_screen(activity_df: pd.DataFrame, obs: pd.DataFrame, condition_col: str,
                          sample_col: str, celltype_col: str, ref_level: str | None = None,
                          test_level: str | None = None, tfs: Sequence[str] | None = None,
                          celltypes: Sequence[str] | None = None, min_samples_per_group: int = 2,
                          engine: str = "auto") -> pd.DataFrame:
    """Per TF x cell type LMM ``activity ~ condition + (1 | sample)`` with BH FDR
    within each cell type (Methods, "Transcription factor activity").

    ``engine="lmerTest"`` (rpy2) reproduces the paper's Satterthwaite t-tests;
    ``engine="statsmodels"`` uses ``MixedLM`` (REML) with Wald z-tests —
    statsmodels has no Satterthwaite/Kenward-Roger df, so p-values are somewhat
    anti-conservative with few samples. ``"auto"`` picks lmerTest when rpy2 is
    installed. Cell types with fewer than ``min_samples_per_group`` samples in
    either condition are skipped.

    Returns ``cell_type, tf, beta, se, p, fdr, n_cells, n_samples_ref,
    n_samples_test, engine`` where ``beta`` is the mean activity difference
    ``test_level - ref_level``.
    """
    if engine == "auto":
        engine = "lmerTest" if has_module("rpy2") else "statsmodels"
    obs = obs.loc[activity_df.index]
    cond = obs[condition_col].astype(str)
    levels = sorted(pd.unique(cond))
    ref_level = str(ref_level) if ref_level is not None else levels[0]
    if test_level is None:
        others = [lv for lv in levels if lv != ref_level]
        if len(others) != 1:
            raise ValueError(f"specify test_level; levels={levels}")
        test_level = others[0]
    test_level = str(test_level)
    tfs = list(tfs) if tfs is not None else list(activity_df.columns)
    cts = celltypes if celltypes is not None else sorted(pd.unique(obs[celltype_col].astype(str)))
    fit = _lmm_lmertest if engine == "lmerTest" else _lmm_statsmodels
    rows = []
    for ct in cts:
        m = (obs[celltype_col].astype(str) == str(ct)).to_numpy() & cond.isin(
            [ref_level, test_level]).to_numpy()
        if not m.any():
            continue
        sub_obs = obs.loc[m]
        x = (sub_obs[condition_col].astype(str) == test_level).astype(float).to_numpy()
        g = sub_obs[sample_col].astype(str).to_numpy()
        n_ref = len(set(g[x == 0]))
        n_test = len(set(g[x == 1]))
        if min(n_ref, n_test) < min_samples_per_group:
            continue
        block = []
        for tf in tfs:
            data = pd.DataFrame({"y": activity_df.loc[m, tf].to_numpy(float), "x": x, "g": g})
            data = data.dropna()
            try:
                beta, se, p = fit(data)
            except Exception:  # pragma: no cover - numerical failure
                beta, se, p = np.nan, np.nan, np.nan
            block.append({"cell_type": ct, "tf": tf, "beta": beta, "se": se, "p": p,
                          "n_cells": int(len(data)), "n_samples_ref": n_ref,
                          "n_samples_test": n_test, "engine": engine})
        blk = pd.DataFrame(block)
        blk.insert(5, "fdr", bh_fdr(blk["p"].to_numpy()))
        rows.append(blk)
    if not rows:
        return pd.DataFrame(columns=["cell_type", "tf", "beta", "se", "p", "fdr", "n_cells",
                                     "n_samples_ref", "n_samples_test", "engine"])
    return pd.concat(rows, ignore_index=True)
