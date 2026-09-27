"""Single-cell analyses from case studies 2 and 3 (Zhang et al., Science 2026).

Implements the Supplementary Methods sections
"Single-cell RNA-seq preprocessing" (QC thresholds, per-donor down-sampling,
normalisation, HVG/PCA/Harmony), "Pseudobulk differential expression"
(donor x cell-type pseudobulks, cell-type eligibility, PyDESeq2 Wald tests,
FDR < 0.05 and |log2FC| > 1) and "Ligand-receptor analysis" (LIANA
``rank_aggregate`` with 1,000 permutations comparing B7-H3-high vs B7-H3-low
fibroblasts; interactions kept if in the top 10% by magnitude, CellPhoneDB
p < 0.01 and supported by >= 3 methods).

Functions that need scanpy / anndata / pydeseq2 / liana import them lazily;
everything else is pure numpy/pandas and is unit-tested.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ._utils import bh_fdr, require

__all__ = [
    "qc_filter",
    "downsample_indices",
    "downsample_by_donor",
    "normalize_log1p",
    "preprocess",
    "eligible_celltypes",
    "pseudobulk_counts",
    "pseudobulk_de",
    "quartile_groups",
    "ligand_receptor_contrast",
    "filter_lr_results",
    "group_specific_interactions",
    "DEFAULT_LR_METHOD_RULES",
]


# --------------------------------------------------------------------------- QC
def _is_sparse(x) -> bool:
    return hasattr(x, "tocsr") and hasattr(x, "nnz")


def qc_filter(adata, min_genes: int = 300, max_genes: int = 9000, max_pct_mt: float = 15,
              mt_prefix: str = "MT-"):
    """Cell-level QC (Methods, "Single-cell RNA-seq preprocessing").

    Keeps cells with ``min_genes <= n_genes_detected <= max_genes`` and
    mitochondrial read fraction ``<= max_pct_mt`` percent. Works on any
    AnnData-like object exposing ``.X`` (raw counts), ``.var_names`` and boolean
    indexing; QC metrics are written to ``.obs`` (``n_genes``, ``total_counts``,
    ``pct_counts_mt``). Returns a filtered copy.
    """
    X = adata.X
    if _is_sparse(X):
        X = X.tocsr()
        n_genes = np.asarray((X > 0).sum(axis=1)).ravel()
        total = np.asarray(X.sum(axis=1)).ravel()
    else:
        X = np.asarray(X)
        n_genes = (X > 0).sum(axis=1)
        total = X.sum(axis=1)
    mt = np.asarray([str(g).upper().startswith(mt_prefix.upper()) for g in adata.var_names])
    if mt.any():
        mt_counts = np.asarray(X[:, np.where(mt)[0]].sum(axis=1)).ravel()
    else:
        mt_counts = np.zeros_like(total, dtype=float)
    pct_mt = np.divide(100.0 * mt_counts, total, out=np.zeros(total.shape, float), where=total > 0)
    adata.obs["n_genes"] = n_genes
    adata.obs["total_counts"] = total
    adata.obs["pct_counts_mt"] = pct_mt
    keep = (n_genes >= min_genes) & (n_genes <= max_genes) & (pct_mt <= max_pct_mt)
    return adata[keep].copy()


def downsample_indices(obs: pd.DataFrame, donor_key: str, celltype_key: str,
                       max_cells: int = 10000, seed: int = 0) -> np.ndarray:
    """Positional indices of cells kept by per-donor stratified down-sampling.

    Donors with more than ``max_cells`` cells are down-sampled to ``max_cells``
    with the per-cell-type quota proportional to the donor's cell-type
    composition (largest-remainder rounding), preserving proportions (Methods,
    "Single-cell RNA-seq preprocessing"). Pure pandas.
    """
    rng = np.random.default_rng(seed)
    pos = np.arange(len(obs))
    donors = obs[donor_key].to_numpy()
    cts = obs[celltype_key].astype(str).to_numpy()
    keep: list[np.ndarray] = []
    for d in pd.unique(donors):
        idx = pos[donors == d]
        if idx.size <= max_cells:
            keep.append(idx)
            continue
        ct = cts[idx]
        levels, counts = np.unique(ct, return_counts=True)
        exact = counts / counts.sum() * max_cells
        quota = np.floor(exact).astype(int)
        rem = max_cells - quota.sum()
        if rem > 0:
            order = np.argsort(-(exact - quota), kind="stable")
            quota[order[:rem]] += 1
        for lev, q in zip(levels, quota):
            members = idx[ct == lev]
            q = min(q, members.size)
            keep.append(rng.choice(members, size=q, replace=False))
    return np.sort(np.concatenate(keep)) if keep else np.array([], dtype=int)


def downsample_by_donor(adata, donor_key: str, celltype_key: str, max_cells: int = 10000,
                        seed: int = 0):
    """AnnData wrapper around :func:`downsample_indices` (returns a copy)."""
    idx = downsample_indices(adata.obs, donor_key, celltype_key, max_cells=max_cells, seed=seed)
    return adata[idx].copy()


def normalize_log1p(adata, target_sum: float = 1e4, counts_layer: str = "counts"):
    """Library-size normalisation to 10,000 counts + log1p (scanpy), keeping raw
    counts in ``adata.layers[counts_layer]`` for pseudobulking."""
    sc = require("scanpy")
    if counts_layer not in adata.layers:
        adata.layers[counts_layer] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


def preprocess(adata, n_hvg: int = 3000, n_pcs: int = 40, batch_key: str | None = None,
               n_neighbors: int = 15, seed: int = 0, run_umap: bool = True):
    """HVG selection (3,000), PCA (40 PCs), optional Harmony on ``batch_key``,
    kNN graph and UMAP (Methods, "Single-cell RNA-seq preprocessing").

    Expects log-normalised ``adata.X`` (see :func:`normalize_log1p`).
    """
    sc = require("scanpy")
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, batch_key=batch_key)
    sc.pp.pca(adata, n_comps=n_pcs, mask_var="highly_variable", random_state=seed)
    rep = "X_pca"
    if batch_key is not None:
        require("harmonypy")
        sc.external.pp.harmony_integrate(adata, key=batch_key, basis="X_pca",
                                         adjusted_basis="X_pca_harmony")
        rep = "X_pca_harmony"
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=rep, random_state=seed)
    if run_umap:
        sc.tl.umap(adata, random_state=seed)
    return adata


# ------------------------------------------------------------------ pseudobulk
def eligible_celltypes(obs: pd.DataFrame, celltype_key: str, donor_key: str, condition_key: str,
                       min_donors: int = 3, min_cells_per_donor: int = 20,
                       return_table: bool = False):
    """Cell types eligible for pseudobulk DE (Methods, "Pseudobulk differential
    expression").

    A donor contributes to a cell type if it has ``>= min_cells_per_donor`` cells
    of that type; a cell type is eligible when *every* condition level has
    ``>= min_donors`` contributing donors. Returns a sorted list (and optionally
    the celltype x condition table of qualifying-donor counts).
    """
    counts = obs.groupby([celltype_key, condition_key, donor_key], observed=True).size()
    ok = counts[counts >= min_cells_per_donor].reset_index()
    levels = pd.unique(obs[condition_key])
    table = (ok.groupby([celltype_key, condition_key], observed=True)[donor_key].nunique()
             .unstack(condition_key).reindex(columns=levels).fillna(0).astype(int))
    all_ct = pd.unique(obs[celltype_key])
    table = table.reindex(all_ct).fillna(0).astype(int)
    eligible = sorted(table.index[(table >= min_donors).all(axis=1)], key=str)
    return (eligible, table) if return_table else eligible


def pseudobulk_counts(X_counts, obs: pd.DataFrame, groupby: Sequence[str] = ("donor", "celltype"),
                      var_names: Iterable[str] | None = None, min_cells: int = 10,
                      carry_cols: Sequence[str] | None = None):
    """Sum raw counts per ``groupby`` combination (e.g. donor x cell type).

    ``X_counts`` is a cells x genes numpy array or scipy sparse matrix of raw
    counts. Groups with fewer than ``min_cells`` cells are dropped. Returns
    ``(counts_df, meta_df)``: pseudobulk samples x genes (int if input is int) and
    per-sample metadata with the ``groupby`` columns, ``n_cells`` and any
    ``carry_cols`` that are constant within the group (e.g. condition).
    Methods, "Pseudobulk differential expression".
    """
    groupby = list(groupby)
    n = len(obs)
    if X_counts.shape[0] != n:
        raise ValueError("X_counts rows must match obs rows")
    keys = obs[groupby].astype(str)
    sample_id = keys.agg("|".join, axis=1).to_numpy() if len(groupby) > 1 else keys.iloc[:, 0].to_numpy()
    codes, uniques = pd.factorize(sample_id, sort=True)
    n_groups = len(uniques)
    from scipy import sparse

    indicator = sparse.csr_matrix((np.ones(n), (codes, np.arange(n))), shape=(n_groups, n))
    if _is_sparse(X_counts):
        summed = (indicator @ X_counts.tocsr()).toarray()
    else:
        summed = np.asarray(indicator @ np.asarray(X_counts))
    if np.allclose(summed, np.round(summed)):
        summed = np.round(summed).astype(np.int64)
    n_cells = np.bincount(codes, minlength=n_groups)
    genes = list(var_names) if var_names is not None else [f"g{i}" for i in range(X_counts.shape[1])]
    counts_df = pd.DataFrame(summed, index=pd.Index(uniques, name="sample"), columns=genes)

    first = pd.DataFrame({"_code": codes})
    first_pos = first.drop_duplicates("_code").sort_values("_code").index.to_numpy()
    meta = obs.iloc[first_pos][groupby].copy()
    meta.index = counts_df.index
    meta["n_cells"] = n_cells
    for col in carry_cols or []:
        nun = obs.groupby(codes)[col].nunique(dropna=False)
        if (nun > 1).any():
            raise ValueError(f"carry column {col!r} is not constant within pseudobulk groups")
        meta[col] = obs.iloc[first_pos][col].to_numpy()
    keep = n_cells >= min_cells
    return counts_df.loc[keep], meta.loc[keep]


def _ols_log_cpm_de(counts_df, meta_df, design_factor, ref_level, test_level, covariates,
                    min_total_counts, prior_count):
    y_all = counts_df.to_numpy(dtype=float)
    lib = y_all.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    keep_genes = y_all.sum(axis=0) >= min_total_counts
    logcpm = np.log2(y_all[:, keep_genes] / lib * 1e6 + prior_count)
    cond = meta_df[design_factor].astype(str).to_numpy()
    x = (cond == str(test_level)).astype(float)
    cols = [np.ones_like(x), x]
    for cov in covariates or []:
        v = meta_df[cov]
        if pd.api.types.is_numeric_dtype(v):
            cols.append(v.to_numpy(float) - v.mean())
        else:
            d = pd.get_dummies(v.astype(str), drop_first=True, dtype=float)
            cols.extend(d.to_numpy().T)
    X = np.column_stack(cols)
    n, p = X.shape
    dof = n - p
    if dof < 1:
        raise ValueError("not enough pseudobulk samples for the OLS design")
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ logcpm
    resid = logcpm - X @ beta
    sigma2 = (resid ** 2).sum(axis=0) / dof
    se = np.sqrt(sigma2 * XtX_inv[1, 1])
    from scipy import stats

    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta[1] / se
    pval = 2 * stats.t.sf(np.abs(t), dof)
    return pd.DataFrame({
        "gene": counts_df.columns[keep_genes],
        "baseMean_logcpm": logcpm.mean(axis=0),
        "log2FC": beta[1],
        "lfcSE": se,
        "stat": t,
        "pvalue": pval,
    })


def pseudobulk_de(counts_df: pd.DataFrame, meta_df: pd.DataFrame, design_factor: str = "condition",
                  ref_level: str | None = None, test_level: str | None = None,
                  engine: str = "pydeseq2", covariates: Sequence[str] | None = None,
                  fdr: float = 0.05, min_lfc: float = 1.0, min_total_counts: int = 10,
                  prior_count: float = 1.0) -> pd.DataFrame:
    """Pseudobulk differential expression for one cell type (Methods,
    "Pseudobulk differential expression").

    ``engine="pydeseq2"`` fits the negative-binomial GLM ``~ [covariates +]
    design_factor`` with PyDESeq2 and a Wald test ``test_level`` vs ``ref_level``
    (as in the paper). ``engine="ols_log_cpm"`` is a dependency-free fallback:
    per-gene OLS of log2(CPM + prior_count) on the condition indicator (+
    covariates), t-test p-values. Both use Benjamini-Hochberg FDR and call a
    gene DE when ``padj < fdr`` and ``|log2FC| > min_lfc``.

    Returns a table with ``gene, log2FC, pvalue, padj, de, direction``.
    """
    meta_df = meta_df.loc[counts_df.index]
    levels = list(pd.unique(meta_df[design_factor].astype(str)))
    if ref_level is None:
        ref_level = sorted(levels)[0]
    ref_level = str(ref_level)
    if test_level is None:
        others = [lv for lv in levels if lv != ref_level]
        if len(others) != 1:
            raise ValueError(f"specify test_level; levels={levels}")
        test_level = others[0]
    test_level = str(test_level)
    sub = meta_df[design_factor].astype(str).isin([ref_level, test_level]).to_numpy()
    counts_df, meta_df = counts_df.loc[sub], meta_df.loc[sub].copy()

    if engine == "ols_log_cpm":
        res = _ols_log_cpm_de(counts_df, meta_df, design_factor, ref_level, test_level,
                              covariates, min_total_counts, prior_count)
    elif engine == "pydeseq2":
        require("pydeseq2")
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats

        keep = counts_df.sum(axis=0) >= min_total_counts
        counts = counts_df.loc[:, keep].round().astype(int)
        meta = meta_df.copy()
        meta[design_factor] = meta[design_factor].astype(str)
        factors = list(covariates or []) + [design_factor]
        try:  # pydeseq2 >= 0.5 formula API
            dds = DeseqDataSet(counts=counts, metadata=meta, design="~" + " + ".join(factors),
                               quiet=True)
        except TypeError:  # older API
            dds = DeseqDataSet(counts=counts, metadata=meta, design_factors=factors,
                               ref_level=[design_factor, ref_level], quiet=True)
        dds.deseq2()
        stat = DeseqStats(dds, contrast=[design_factor, test_level, ref_level], quiet=True)
        stat.summary()
        r = stat.results_df
        res = pd.DataFrame({"gene": r.index, "baseMean": r["baseMean"].to_numpy(),
                            "log2FC": r["log2FoldChange"].to_numpy(), "lfcSE": r["lfcSE"].to_numpy(),
                            "stat": r["stat"].to_numpy(), "pvalue": r["pvalue"].to_numpy()})
    else:
        raise ValueError(f"unknown engine {engine!r}")
    res["padj"] = bh_fdr(res["pvalue"].to_numpy())
    res["de"] = (res["padj"] < fdr) & (res["log2FC"].abs() > min_lfc)
    res["direction"] = np.where(res["de"], np.where(res["log2FC"] > 0, "up", "down"), "ns")
    res["engine"] = engine
    res["contrast"] = f"{test_level}_vs_{ref_level}"
    return res.sort_values("pvalue", kind="stable").reset_index(drop=True)


def quartile_groups(values, top: float = 0.75, bottom: float = 0.25):
    """Label values in the top quantile ``'high'``, bottom quantile ``'low'``, else
    ``None`` (middle excluded; NaN -> None). Used for B7-H3-high vs -low
    fibroblasts and all quartile contrasts in the paper. Returns a pandas Series
    (index preserved if input is a Series)."""
    s = pd.Series(values) if not isinstance(values, pd.Series) else values
    v = pd.to_numeric(s, errors="coerce")
    hi, lo = v.quantile(top), v.quantile(bottom)
    lab = pd.Series([None] * len(v), index=s.index, dtype=object)
    lab[v >= hi] = "high"
    lab[(v <= lo) & ~(v >= hi)] = "low"
    return lab


# ---------------------------------------------------------- ligand-receptor
#: How each LIANA method "supports" an interaction. Columns absent from the
#: result table are ignored. Operationalisation of the paper's ">= 3 methods".
DEFAULT_LR_METHOD_RULES: dict[str, tuple[str, str, float]] = {
    "CellPhoneDB": ("cellphone_pvals", "<=", 0.05),
    "CellChat": ("cellchat_pvals", "<=", 0.05),
    "geometric_mean": ("gmean_pvals", "<=", 0.05),
    "SingleCellSignalR": ("lrscore", ">=", 0.6),
    "log2FC": ("lr_logfc", ">", 0.0),
    "NATMI": ("spec_weight", "top_frac", 0.10),
    "Connectome": ("scaled_weight", "top_frac", 0.10),
}


def _method_support(df: pd.DataFrame, rules) -> pd.Series:
    n = pd.Series(0, index=df.index, dtype=int)
    for _, (col, op, thr) in rules.items():
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        if op == "<=":
            hit = v <= thr
        elif op == "<":
            hit = v < thr
        elif op == ">=":
            hit = v >= thr
        elif op == ">":
            hit = v > thr
        elif op == "top_frac":
            hit = v >= v.quantile(1 - thr)
        else:
            raise ValueError(op)
        n += hit.fillna(False).astype(int)
    return n


def filter_lr_results(df: pd.DataFrame, top_frac: float = 0.10, cpdb_p: float = 0.01,
                      min_methods: int = 3, method_rules: dict | None = None) -> pd.DataFrame:
    """Apply the paper's ligand-receptor filter to a LIANA result table (Methods,
    "Ligand-receptor analysis").

    Keeps interactions (i) in the top ``top_frac`` by magnitude — using
    ``magnitude_rank`` (lower is better) when present, else ``expr_prod``
    (higher is better); (ii) with CellPhoneDB ``cellphone_pvals < cpdb_p``; and
    (iii) supported by ``>= min_methods`` methods. Support is taken from an
    ``n_methods`` column if present, otherwise computed from ``method_rules``
    (default :data:`DEFAULT_LR_METHOD_RULES`). Adds ``n_methods``.
    """
    out = df.copy()
    if "magnitude_rank" in out.columns:
        mag = pd.to_numeric(out["magnitude_rank"], errors="coerce")
        top = mag <= mag.quantile(top_frac)
    elif "expr_prod" in out.columns:
        mag = pd.to_numeric(out["expr_prod"], errors="coerce")
        top = mag >= mag.quantile(1 - top_frac)
    else:
        raise KeyError("need 'magnitude_rank' or 'expr_prod' column")
    if "cellphone_pvals" not in out.columns:
        raise KeyError("need 'cellphone_pvals' column")
    sig = pd.to_numeric(out["cellphone_pvals"], errors="coerce") < cpdb_p
    if "n_methods" not in out.columns:
        out["n_methods"] = _method_support(out, method_rules or DEFAULT_LR_METHOD_RULES)
    keep = top.fillna(False) & sig.fillna(False) & (out["n_methods"] >= min_methods)
    return out.loc[keep].reset_index(drop=True)


def group_specific_interactions(df_a: pd.DataFrame, df_b: pd.DataFrame,
                                key_cols: Sequence[str] = ("source", "target", "ligand_complex",
                                                           "receptor_complex")) -> dict:
    """Split filtered interactions into ``a_only``, ``b_only`` and ``shared`` by
    ``key_cols`` (e.g. B7-H3-high-specific vs -low-specific fibroblast
    interactions; Methods, "Ligand-receptor analysis")."""
    key_cols = list(key_cols)
    ka = df_a[key_cols].astype(str).agg("|".join, axis=1)
    kb = df_b[key_cols].astype(str).agg("|".join, axis=1)
    sa, sb = set(ka), set(kb)
    return {
        "a_only": df_a.loc[~ka.isin(sb).to_numpy()].reset_index(drop=True),
        "b_only": df_b.loc[~kb.isin(sa).to_numpy()].reset_index(drop=True),
        "shared": df_a.loc[ka.isin(sb).to_numpy()].reset_index(drop=True),
    }


def ligand_receptor_contrast(adata, celltype_key: str, focal_celltype: str, group_key: str,
                             group_a: str = "high", group_b: str = "low", n_perms: int = 1000,
                             resource_name: str = "consensus", expr_prop: float = 0.1,
                             seed: int = 0, use_raw: bool = False, filter_kwargs: dict | None = None,
                             key_cols: Sequence[str] = ("source", "target", "ligand_complex",
                                                        "receptor_complex")) -> dict:
    """LIANA ``rank_aggregate`` for two states of a focal cell type (e.g.
    B7-H3-high vs B7-H3-low fibroblasts) against all other cell types (Methods,
    "Ligand-receptor analysis").

    ``adata.obs[group_key]`` labels focal cells ``group_a`` / ``group_b`` (e.g.
    from :func:`quartile_groups` on CD276 expression; other focal cells are
    dropped). LIANA is run separately for each state with ``n_perms`` (1,000)
    permutations, filtered with :func:`filter_lr_results`, and contrasted with
    :func:`group_specific_interactions`. Returns dict with ``raw_a``, ``raw_b``,
    ``filtered_a``, ``filtered_b``, ``a_only``, ``b_only``, ``shared``.
    """
    li = require("liana")
    obs = adata.obs
    is_focal = obs[celltype_key].astype(str) == str(focal_celltype)
    results = {}
    for tag, grp in (("a", group_a), ("b", group_b)):
        mask = (~is_focal) | (is_focal & (obs[group_key].astype(str) == str(grp)))
        sub = adata[mask.to_numpy()].copy()
        li.mt.rank_aggregate(sub, groupby=celltype_key, resource_name=resource_name,
                             expr_prop=expr_prop, n_perms=n_perms, seed=seed, use_raw=use_raw,
                             verbose=False)
        raw = sub.uns["liana_res"].copy()
        raw = raw[(raw["source"].astype(str) == str(focal_celltype))
                  | (raw["target"].astype(str) == str(focal_celltype))]
        results[f"raw_{tag}"] = raw.reset_index(drop=True)
        results[f"filtered_{tag}"] = filter_lr_results(raw, **(filter_kwargs or {}))
    results.update(group_specific_interactions(results["filtered_a"], results["filtered_b"],
                                               key_cols))
    return results
