"""Transcription-factor activity analysis from case study 3 (OSMR in ulcerative
colitis).

Implements the Supplementary Methods section "Transcription factor activity":
per-cell TF activities inferred with decoupler (multivariate linear model,
``mlm``, or univariate ``ulm``) on the CollecTRI regulon collection, with the
target IL6ST removed from the STAT1 regulon to avoid circularity with the
receptor under study; per TF and cell type a linear mixed model
``activity ~ condition + (1 | sample)`` fitted with lmerTest (Satterthwaite
df) and Benjamini-Hochberg correction within each cell type.

The genome-wide MLM screen uses the unedited regulons (``exclusions=None``);
the receptor-cleaned STAT1 regulon (:func:`clean_regulon`, all seven gp130
receptors removed) is for the targeted ULM step that feeds the LMG
decomposition (:func:`select_celltypes_by_detection`, :func:`pseudobulk_means`,
:func:`run_lmg_per_celltype`).
"""

from __future__ import annotations

import warnings
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ._utils import bh_fdr, fit_mixedlm, has_module, require, zscore
from .variance_decomposition import GP130_RECEPTORS, bootstrap_lmg, lmg_shares, r_packages_available

__all__ = ["load_collectri", "remove_targets_from_regulon", "score_tf_activity",
           "tf_mixed_model_screen", "PAPER_REGULON_EXCLUSIONS", "clean_regulon",
           "select_celltypes_by_detection", "pseudobulk_means", "run_lmg_per_celltype"]

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


def clean_regulon(net: pd.DataFrame, tf: str = "STAT1", remove: Iterable[str] = GP130_RECEPTORS,
                  source_col: str = "source", target_col: str = "target") -> pd.DataFrame:
    """Remove the receptors under study (default: all seven gp130-family receptors)
    from ``tf``'s regulon, so its activity is not computed from their own
    expression (circularity). Other TFs' regulons are untouched."""
    return remove_targets_from_regulon(net, {tf: list(remove)}, source_col=source_col, target_col=target_col)


def load_collectri(organism: str = "human") -> pd.DataFrame:
    """Load the CollecTRI network with decoupler (v2 ``dc.op.collectri`` or v1
    ``dc.get_collectri``). Requires network access on first use."""
    dc = require("decoupler")
    if hasattr(dc, "op") and hasattr(dc.op, "collectri"):
        return dc.op.collectri(organism=organism)
    return dc.get_collectri(organism=organism, split_complexes=False)


def score_tf_activity(adata, method: str = "mlm", regulons: pd.DataFrame | None = None,
                      exclusions: Mapping[str, Iterable[str]] | None = None,
                      tmin: int = 5) -> pd.DataFrame:
    """Per-cell TF activity scores (cells x TFs) with decoupler ``mlm``/``ulm``
    on CollecTRI (Methods, "Transcription factor activity").

    ``adata.X`` should be log-normalised expression. ``regulons`` defaults to
    CollecTRI; ``exclusions`` (default None: the genome-wide MLM screen uses the
    unedited regulons) are applied with :func:`remove_targets_from_regulon` —
    e.g. :data:`PAPER_REGULON_EXCLUSIONS`, or pass a :func:`clean_regulon`-ed
    network as ``regulons`` for the targeted ULM step.
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


def _lmm_statsmodels(data: pd.DataFrame, reml: bool = False):
    import statsmodels.formula.api as smf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm("y ~ x", data, groups=data["g"])
    fit = fit_mixedlm(model, reml=reml)
    return float(fit.params["x"]), float(fit.bse["x"]), float(fit.pvalues["x"])


def _lmm_lmertest(data: pd.DataFrame, reml: bool = False):
    require("rpy2", "r")
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    importr("lmerTest")
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["dat"] = data
    co = ro.r(f'coef(summary(lmerTest::lmer(y ~ x + (1|g), data=dat, REML={"TRUE" if reml else "FALSE"})))["x", ]')
    return float(co[0]), float(co[1]), float(co[4])


def tf_mixed_model_screen(activity_df: pd.DataFrame, obs: pd.DataFrame, condition_col: str,
                          sample_col: str, celltype_col: str, ref_level: str | None = None,
                          test_level: str | None = None, tfs: Sequence[str] | None = None,
                          celltypes: Sequence[str] | None = None, min_samples_per_group: int = 2,
                          engine: str = "auto", *, reml: bool = False, min_samples: int = 4) -> pd.DataFrame:
    """Per TF x cell type LMM ``activity ~ condition + (1 | sample)`` with BH FDR
    within each cell type (Methods, "Transcription factor activity").

    ``engine="lmerTest"`` (rpy2) reproduces the paper's Satterthwaite t-tests;
    ``engine="statsmodels"`` uses ``MixedLM`` (REML) with Wald z-tests —
    statsmodels has no Satterthwaite/Kenward-Roger df, so p-values are somewhat
    anti-conservative with few samples. ``"auto"`` picks lmerTest when rpy2 is
    installed. Cell types with fewer than ``min_samples_per_group`` samples in
    either condition, or fewer than ``min_samples`` (4) samples in total, are
    skipped. As in the authors' ``osmr/code/01c_tf_activity_screen.py``
    (Zenodo archive) the model is fitted by maximum likelihood
    (``lmer(..., REML=FALSE)``; ``reml=True`` for REML).

    Returns ``cell_type, tf, beta, se, p, fdr, n_cells, n_samples_ref,
    n_samples_test, engine`` where ``beta`` is the mean activity difference
    ``test_level - ref_level``.
    """
    if engine == "auto":
        engine = "lmerTest" if r_packages_available("lmerTest") else "statsmodels"
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
        if min(n_ref, n_test) < min_samples_per_group or n_ref + n_test < min_samples:
            continue
        block = []
        for tf in tfs:
            data = pd.DataFrame({"y": activity_df.loc[m, tf].to_numpy(float), "x": x, "g": g})
            data = data.dropna()
            try:
                beta, se, p = fit(data, reml=reml)
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


# --------------------------------------------------------------------------- LMG helpers
def _column(adata, gene: str, layer: str | None = None) -> np.ndarray:
    names = list(map(str, adata.var_names))
    if gene not in names:
        raise KeyError(f"{gene!r} not in var_names")
    X = adata.layers[layer] if layer else adata.X
    col = X[:, names.index(gene)]
    col = col.toarray() if hasattr(col, "toarray") else np.asarray(col)
    return np.asarray(col, dtype=float).ravel()


def select_celltypes_by_detection(adata, gene: str = "OSMR", min_frac: float = 0.05,
                                  celltype_key: str = "cell_type", layer: str | None = None,
                                  return_table: bool = False):
    """Cell types in which ``gene`` is detected (> 0) in at least ``min_frac`` of
    cells (paper: OSMR >= 5%). Returns a sorted list (and optionally the
    per-cell-type table ``n_cells, frac_detected``)."""
    v = _column(adata, gene, layer)
    ct = adata.obs[celltype_key].astype(str).to_numpy()
    table = (pd.DataFrame({"ct": ct, "det": v > 0}).groupby("ct")["det"].agg(["size", "mean"])
             .rename(columns={"size": "n_cells", "mean": "frac_detected"}))
    keep = sorted(table.index[table["frac_detected"] >= min_frac])
    return (keep, table) if return_table else keep


def pseudobulk_means(adata, activity: pd.DataFrame | None = None, genes: Sequence[str] = (),
                     by: Sequence[str] = ("sample", "cell_type"), min_cells: int = 10,
                     layer: str | None = None, carry_cols: Sequence[str] = ()) -> pd.DataFrame:
    """Per-group (default sample x cell type) mean expression of ``genes`` and
    mean activity (``activity``: cells x TFs, indexed like ``adata.obs``).

    Groups with fewer than ``min_cells`` cells are dropped. Returns one row per
    group: the ``by`` columns, ``n_cells``, one column per gene and per TF, and
    ``carry_cols`` (first value).
    """
    obs = adata.obs
    by = list(by)
    frame = obs[by + [c for c in carry_cols if c not in by]].copy()
    for g in genes:
        frame[g] = _column(adata, g, layer)
    tfs: list[str] = []
    if activity is not None:
        act = activity.loc[obs.index] if not activity.index.equals(obs.index) else activity
        for c in act.columns:
            name = str(c) if str(c) not in frame.columns else f"{c}_activity"
            frame[name] = act[c].to_numpy(float)
            tfs.append(name)
    agg = {g: "mean" for g in list(genes) + tfs}
    for c in carry_cols:
        if c not in by:
            agg[c] = "first"
    grouped = frame.groupby(by, observed=True, sort=True)
    out = grouped.agg(agg)
    out.insert(0, "n_cells", grouped.size())
    out = out[out["n_cells"] >= min_cells].reset_index()
    return out


def run_lmg_per_celltype(table: pd.DataFrame, outcome: str = "STAT1", predictors: Sequence[str] = GP130_RECEPTORS,
                         celltype_col: str = "cell_type", cluster_col: str | None = "patient",
                         standardize: bool = True, n_boot: int = 2000, seed: int = 0,
                         min_rows: int = 6) -> pd.DataFrame:
    """LMG (Shapley) shares of ``outcome`` variance across ``predictors`` per cell type.

    ``table`` is a per-sample table (e.g. :func:`pseudobulk_means`). Within each
    cell type the outcome and predictors are z-scored (``standardize``),
    constant predictors dropped, and shares with ``cluster_col`` bootstrap CIs
    computed with :func:`~vbt.analysis.variance_decomposition.bootstrap_lmg`.
    Returns ``cell_type, predictor, share, ci_low, ci_high, pct_of_r2, p_top,
    rank, r2_total, n, engine``.
    """
    rows = []
    for ct, d in table.groupby(celltype_col, observed=True, sort=True):
        preds = [p for p in predictors if p in d.columns and d[p].nunique(dropna=True) > 1]
        cols = [outcome, *preds] + ([cluster_col] if cluster_col and cluster_col in d.columns else [])
        d = d[cols].dropna()
        if not preds or len(d) < max(min_rows, len(preds) + 2):
            continue
        d = d.copy()
        if standardize:
            for c in [outcome, *preds]:
                d[c] = zscore(d[c].to_numpy(float))
        cl = cluster_col if cluster_col and cluster_col in d.columns else None
        boot = bootstrap_lmg(d, outcome, preds, cluster_col=cl, n_boot=n_boot, seed=seed)
        point = lmg_shares(d, outcome, preds).set_index("predictor")
        boot["rank"] = boot["predictor"].map(point["rank"])
        boot.insert(0, celltype_col, ct)
        boot["r2_total"] = boot.attrs.get("r2_total")
        boot["n"] = len(d)
        rows.append(boot)
    if not rows:
        return pd.DataFrame(columns=[celltype_col, "predictor", "share", "ci_low", "ci_high", "pct_of_r2",
                                     "p_top", "rank", "r2_total", "n", "engine"])
    out = pd.concat(rows, ignore_index=True)
    keep = [celltype_col, "predictor", "share", "ci_low", "ci_high", "pct_of_r2", "pct_ci_low", "pct_ci_high",
            "p_top", "rank", "r2_total", "n", "engine"]
    return out[[c for c in keep if c in out.columns]]
