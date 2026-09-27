"""Cross-disease OSMR expression analysis from case study 3 (CELLxGENE Census).

Implements the Supplementary Methods section "Cross-disease expression
analysis": cells from CELLxGENE Census grouped into donor x cell-type
pseudobulks (>= 25 cells) summarised as the mean per-cell log2 counts per
10,000 (log2 CP10K) of OSMR; disease vs normal compared within each disease
and cell type with a weighted linear mixed model (weights = number of cells;
covariates log sequencing depth and assay group; random intercept for dataset)
fitted with lmerTest, falling back to weighted OLS when fewer than two
datasets contribute; Benjamini-Hochberg correction across all tests.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from ._utils import bh_fdr, has_module, require

__all__ = ["pseudobulk_log2cp10k", "pair_with_normals", "disease_vs_normal_lmm", "census_query_plan"]


def pseudobulk_log2cp10k(counts_gene, total_counts, obs: pd.DataFrame, donor_col: str = "donor_id",
                         celltype_col: str = "cell_type", min_cells: int = 25,
                         carry_cols: Sequence[str] = ("dataset_id", "disease", "condition", "assay",
                                                      "assay_group", "tissue_general"),
                         gene: str = "OSMR") -> pd.DataFrame:
    """Donor x cell-type summaries of one gene (Methods, "Cross-disease
    expression analysis").

    ``counts_gene`` and ``total_counts`` are per-cell raw counts of the gene and
    per-cell library sizes. Per group returns ``n_cells`` (groups with fewer
    than ``min_cells`` dropped), ``mean_log2_<gene>_cp10k`` (mean over cells of
    log2(1 + 1e4 * count / total)), ``pseudobulk_log2_<gene>_cp10k`` (log2(1 +
    1e4 * sum counts / sum totals)), ``frac_expressing`` and ``log_depth``
    (log10 mean total counts per cell), plus the first value of any
    ``carry_cols`` present in ``obs``.
    """
    c = np.asarray(counts_gene, dtype=float).ravel()
    t = np.asarray(total_counts, dtype=float).ravel()
    cp10k = np.divide(c * 1e4, t, out=np.zeros_like(c), where=t > 0)
    df = pd.DataFrame({"_g": c, "_t": t, "_l": np.log2(1 + cp10k), "_e": (c > 0).astype(float)})
    keys = [donor_col, celltype_col]
    for k in keys:
        df[k] = obs[k].to_numpy()
    carry = [col for col in carry_cols if col in obs.columns and col not in keys]
    for col in carry:
        df[col] = obs[col].to_numpy()
    agg = {"_g": "sum", "_t": ["sum", "mean"], "_l": ["mean", "size"], "_e": "mean"}
    for col in carry:
        agg[col] = "first"
    g = df.groupby(keys, observed=True, sort=True).agg(agg)
    g.columns = ["_".join(x) if isinstance(x, tuple) else x for x in g.columns]
    out = pd.DataFrame({
        donor_col: g.index.get_level_values(0),
        celltype_col: g.index.get_level_values(1),
        "n_cells": g["_l_size"].to_numpy().astype(int),
        f"mean_log2_{gene}_cp10k": g["_l_mean"].to_numpy(),
        f"pseudobulk_log2_{gene}_cp10k": np.log2(1 + 1e4 * g["_g_sum"].to_numpy()
                                                 / np.where(g["_t_sum"] > 0, g["_t_sum"], np.nan)),
        "frac_expressing": g["_e_mean"].to_numpy(),
        "log_depth": np.log10(g["_t_mean"].clip(lower=1).to_numpy()),
    })
    for col in carry:
        out[col] = g[f"{col}_first"].to_numpy()
    return out[out["n_cells"] >= min_cells].reset_index(drop=True)


def pair_with_normals(df: pd.DataFrame, disease_col: str = "disease_name",
                      ref_level: str = "normal", match_on: Sequence[str] = ("tissue_general",)
                      ) -> pd.DataFrame:
    """Build per-disease comparison sets: for every non-normal disease, stack its
    pseudobulks with the ``ref_level`` pseudobulks matching on ``match_on``
    columns (e.g. same tissue), relabelling the normal rows' ``disease_col`` to
    that disease so strata ``by=(disease_col, "cell_type")`` contain both arms.
    Adds ``condition`` ("disease"/``ref_level``) if absent."""
    d = df.copy()
    if "condition" not in d.columns:
        d["condition"] = np.where(d[disease_col].astype(str) == ref_level, ref_level, "disease")
    is_ref = d[disease_col].astype(str) == ref_level
    normals, dis = d[is_ref], d[~is_ref]
    match_on = [m for m in match_on if m in d.columns]
    parts = []
    for name, grp in dis.groupby(disease_col, observed=True, sort=True):
        nm = normals
        if match_on:
            keys = grp[match_on].drop_duplicates()
            nm = normals.merge(keys, on=match_on, how="inner")
        nm = nm.assign(**{disease_col: name})
        parts.append(pd.concat([grp, nm], ignore_index=True))
    return pd.concat(parts, ignore_index=True) if parts else d.iloc[0:0]


def _fit_weighted_python(d: pd.DataFrame, formula: str, group: str, weights: str):
    import statsmodels.formula.api as smf

    n_groups = d[group].nunique()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.wls(formula, data=d, weights=d[weights].to_numpy(float))
        if n_groups >= 2:
            groups = pd.factorize(d[group])[0]
            fit = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
            method = "wls_cluster_robust"
        else:
            fit = model.fit()
            method = "wls"
    return (float(fit.params["disease"]), float(fit.bse["disease"]),
            float(fit.pvalues["disease"]), method)


def _fit_lmertest(d: pd.DataFrame, rhs: str, value: str, group: str, weights: str):
    require("rpy2", "r")
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    importr("lmerTest")
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["dat"] = d
    co = ro.r(f'coef(summary(lmerTest::lmer({value} ~ {rhs} + (1|{group}), data=dat, '
              f'weights={weights}, REML=TRUE)))["disease", ]')
    return float(co[0]), float(co[1]), float(co[4]), "lmerTest_weighted_lmm"


def disease_vs_normal_lmm(df: pd.DataFrame, value: str = "mean_log2_OSMR_cp10k",
                          condition: str = "condition", covars: Sequence[str] = ("log_depth",
                                                                                 "assay_group"),
                          group: str = "dataset_id", weights: str = "n_cells",
                          by: Sequence[str] | None = ("disease_name", "cell_type"),
                          ref_level: str = "normal", engine: str = "auto",
                          min_per_arm: int = 2) -> pd.DataFrame:
    """Weighted disease-vs-normal model per stratum (Methods, "Cross-disease
    expression analysis").

    Model: ``value ~ disease + covars`` with observation weights ``weights``
    (cells per pseudobulk) and a random intercept for ``group`` (dataset).
    ``disease`` = 1 when ``condition != ref_level``. Engines:

    * ``"lmerTest"`` (rpy2): weighted LMM with Satterthwaite df, as in the paper.
    * ``"python"``: statsmodels ``MixedLM`` does not support observation
      weights, so the fallback is WLS with dataset-clustered robust SEs when
      >= 2 datasets contribute, and plain weighted OLS when < 2 datasets (the
      paper's own fallback for the single-dataset case).
    * ``"auto"``: lmerTest if rpy2 is installed, else python.

    Covariates that are constant within a stratum are dropped. ``by`` defines
    strata (``None`` = one model). ``fdr`` is BH across all strata (global).
    Returns ``[by...], beta, se, p, fdr, n_obs, n_disease, n_normal,
    n_datasets, method``.
    """
    if engine == "auto":
        engine = "lmerTest" if has_module("rpy2") else "python"
    by = list(by) if by else []
    strata = df.groupby(by, observed=True, sort=True) if by else [((), df)]
    rows = []
    for key, d in strata:
        d = d.dropna(subset=[value, condition, weights]).copy()
        d["disease"] = (d[condition].astype(str) != str(ref_level)).astype(float)
        n_dis, n_norm = int(d["disease"].sum()), int((1 - d["disease"]).sum())
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        row.update(n_obs=len(d), n_disease=n_dis, n_normal=n_norm,
                   n_datasets=int(d[group].nunique()) if group in d else 0)
        if min(n_dis, n_norm) < min_per_arm:
            rows.append({**row, "beta": np.nan, "se": np.nan, "p": np.nan, "method": "skipped"})
            continue
        use = []
        for c in covars:
            if c in d.columns and d[c].nunique(dropna=True) > 1:
                d = d.dropna(subset=[c])
                use.append(f"C({c})" if not pd.api.types.is_numeric_dtype(d[c]) else c)
        rhs = " + ".join(["disease"] + use)
        try:
            if engine == "lmerTest" and d[group].nunique() >= 2:
                beta, se, p, method = _fit_lmertest(d, rhs.replace("C(", "factor("), value, group,
                                                    weights)
            else:
                beta, se, p, method = _fit_weighted_python(d, f"{value} ~ {rhs}", group, weights)
        except Exception as exc:  # pragma: no cover - numerical failure
            beta, se, p, method = np.nan, np.nan, np.nan, f"failed: {exc}"
        rows.append({**row, "beta": beta, "se": se, "p": p, "method": method})
    out = pd.DataFrame(rows)
    out["fdr"] = bh_fdr(out["p"].to_numpy())
    cols = by + ["beta", "se", "p", "fdr", "n_obs", "n_disease", "n_normal", "n_datasets", "method"]
    return out[cols]


def census_query_plan(gene: str = "OSMR", diseases: Sequence[str] | None = None,
                      tissue_general: Sequence[str] | None = None,
                      census_version: str = "stable", organism: str = "homo_sapiens",
                      execute: bool = False, min_cells: int = 25):
    """Build (and optionally run) the CELLxGENE Census query used for the
    cross-disease analysis (Methods, "Cross-disease expression analysis").

    The plan restricts to primary (``is_primary_data == True``) cells of the
    requested diseases plus ``normal``, and fetches the gene's raw counts with
    per-cell ``raw_sum`` (library size) and donor / cell type / dataset / assay
    / disease annotations. With ``execute=True`` (requires ``pip install
    'vbt-harness[singlecell]'`` and network) returns the pseudobulk table from
    :func:`pseudobulk_log2cp10k`; otherwise returns the plan dict.
    """
    filt = ["is_primary_data == True"]
    if diseases:
        dl = sorted(set(diseases) | {"normal"})
        filt.append(f"disease in {dl!r}")
    if tissue_general:
        filt.append(f"tissue_general in {sorted(tissue_general)!r}")
    plan = {
        "census_version": census_version,
        "organism": organism,
        "var_value_filter": f"feature_name == '{gene}'",
        "obs_value_filter": " and ".join(filt),
        "obs_column_names": ["soma_joinid", "dataset_id", "donor_id", "cell_type", "disease",
                             "assay", "tissue_general", "raw_sum", "is_primary_data"],
        "min_cells": min_cells,
    }
    if not execute:
        return plan
    census_mod = require("cellxgene_census")
    with census_mod.open_soma(census_version=census_version) as census:
        adata = census_mod.get_anndata(census, organism=organism,
                                       var_value_filter=plan["var_value_filter"],
                                       obs_value_filter=plan["obs_value_filter"],
                                       obs_column_names=plan["obs_column_names"])
    X = adata.X
    counts = np.asarray(X.toarray() if hasattr(X, "toarray") else X).ravel()
    obs = adata.obs.copy()
    obs["condition"] = np.where(obs["disease"].astype(str) == "normal", "normal", "disease")
    obs["disease_name"] = obs["disease"].astype(str)
    obs["assay_group"] = np.where(obs["assay"].astype(str).str.contains("10x", case=False),
                                  "10x", "other")
    pb = pseudobulk_log2cp10k(counts, obs["raw_sum"].to_numpy(), obs, donor_col="donor_id",
                              celltype_col="cell_type", min_cells=min_cells,
                              carry_cols=("dataset_id", "disease_name", "condition", "assay",
                                          "assay_group", "tissue_general"), gene=gene)
    pb.attrs["plan"] = plan
    return pb
