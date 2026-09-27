"""Spatial transcriptomics analyses from case study 2 (B7-H3 in lung cancer).

Implements the Supplementary Methods sections "Spatial deconvolution"
(cell2location: reference model 250 epochs, spatial model 10,000 epochs,
N_cells_per_location = 8, gene of interest excluded from the gene set, q05
abundances) and "Spatial immune-neighbourhood analysis" (k-nearest-neighbour
rings of neighbour ranks 1-6, 7-15 and 16-30 computed per sample; CD276-
expressing spots stratified per sample into top vs bottom quartile with the
middle 50% excluded; per immune cell type a linear mixed model
``Y ~ high + z(UMI) + z(fibroblast) + z(epithelial) + z(endothelial)`` with
random intercepts for patient and sample-within-patient; percentage change of
mean abundance high vs low with bootstrap 95% CIs over spots (10,000 resamples
in the paper)).
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from ._utils import bh_fdr, fit_mixedlm, has_module, require, zscore

__all__ = [
    "DEFAULT_RINGS",
    "ring_label",
    "knn_rings",
    "neighbor_mean_abundance",
    "immune_neighborhood_analysis",
    "run_cell2location",
]

#: Neighbour-rank rings used in the paper (1-based ranks, self excluded).
DEFAULT_RINGS: list[tuple[int, int]] = [(1, 6), (7, 15), (16, 30)]


def ring_label(ring: tuple[int, int]) -> str:
    return f"{ring[0]}-{ring[1]}"


def knn_rings(coords, k_rings: Sequence[tuple[int, int]] = DEFAULT_RINGS,
              sample_ids=None) -> dict[str, np.ndarray]:
    """Neighbour index sets per ring (Methods, "Spatial immune-neighbourhood analysis").

    For each sample separately (``sample_ids``; all spots one sample if None) a
    :class:`sklearn.neighbors.NearestNeighbors` model is fitted on the spot
    coordinates and neighbours are ranked by distance (self excluded). Returns
    ``{"1-6": array(n_spots, 6), "7-15": array(n_spots, 9), ...}`` of *global*
    row indices; if a sample has too few spots, missing slots are ``-1``.
    """
    from sklearn.neighbors import NearestNeighbors

    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    sample_ids = np.zeros(n, dtype=int) if sample_ids is None else np.asarray(sample_ids)
    max_k = max(hi for _, hi in k_rings)
    ranked = np.full((n, max_k), -1, dtype=np.int64)
    for s in pd.unique(sample_ids):
        idx = np.where(sample_ids == s)[0]
        if idx.size < 2:
            continue
        k = min(max_k + 1, idx.size)
        nn = NearestNeighbors(n_neighbors=k).fit(coords[idx])
        _, nbr = nn.kneighbors(coords[idx])
        # drop self explicitly (robust to duplicated coordinates)
        for row, local in enumerate(nbr):
            others = local[local != row][:max_k]
            ranked[idx[row], : others.size] = idx[others]
    return {ring_label(r): ranked[:, r[0] - 1: r[1]] for r in k_rings}


def neighbor_mean_abundance(abund_df: pd.DataFrame, coords, sample_ids, ring: tuple[int, int],
                            rings: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """Mean abundance of each column of ``abund_df`` over the neighbours of each
    spot within ``ring`` (e.g. (1, 6)); ``ring=(0, 0)`` returns the spot itself.
    ``rings`` may be a precomputed :func:`knn_rings` result."""
    if tuple(ring) == (0, 0):
        return abund_df.copy()
    lab = ring_label(ring)
    if rings is None or lab not in rings:
        rings = knn_rings(coords, [tuple(ring)], sample_ids)
    nbr = rings[lab]
    vals = abund_df.to_numpy(dtype=float)
    valid = nbr >= 0
    gathered = vals[np.where(valid, nbr, 0)]  # (n, k, m)
    gathered[~valid] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(gathered, axis=1)
    return pd.DataFrame(mean, index=abund_df.index, columns=abund_df.columns)


def _bootstrap_pct_change(y_high, y_low, n_boot, rng, alpha=0.05):
    y_high = np.asarray(y_high, float)
    y_low = np.asarray(y_low, float)
    m_lo = y_low.mean()
    pct = (y_high.mean() - m_lo) / m_lo * 100 if m_lo != 0 else np.nan
    if n_boot <= 0:
        return pct, np.nan, np.nan
    bh = np.empty(n_boot)
    bl = np.empty(n_boot)
    chunk = max(1, int(2e7 // max(1, y_high.size + y_low.size)))
    for start in range(0, n_boot, chunk):
        stop = min(n_boot, start + chunk)
        m = stop - start
        bh[start:stop] = y_high[rng.integers(0, y_high.size, (m, y_high.size))].mean(axis=1)
        bl[start:stop] = y_low[rng.integers(0, y_low.size, (m, y_low.size))].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boots = (bh - bl) / bl * 100
    lo, hi = np.nanquantile(boots, [alpha / 2, 1 - alpha / 2])
    return pct, lo, hi


def _fit_mixedlm(data: pd.DataFrame, formula: str, patient_col: str, sample_col: str):
    import statsmodels.formula.api as smf

    n_samples_per_patient = data.groupby(patient_col)[sample_col].nunique()
    vc = {"sample": f"0 + C({sample_col})"} if (n_samples_per_patient > 1).any() else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(formula, data, groups=data[patient_col], re_formula="1",
                            vc_formula=vc)
    fit = fit_mixedlm(model)
    return fit.params["high"], fit.bse["high"], fit.pvalues["high"]


def _fit_lmertest(data: pd.DataFrame, formula_rhs: str, patient_col: str, sample_col: str):
    """lmerTest (Satterthwaite df) via rpy2: Y ~ rhs + (1|patient) + (1|patient:sample)."""
    require("rpy2", "r")
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    importr("lmerTest")
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["dat"] = data
    ro.r(f"fit <- lmerTest::lmer(Y ~ {formula_rhs} + (1|{patient_col}) + "
         f"(1|{patient_col}:{sample_col}), data=dat, REML=TRUE)")
    co = ro.r('coef(summary(fit))["high", ]')
    return float(co[0]), float(co[1]), float(co[4])


def immune_neighborhood_analysis(spots_df: pd.DataFrame, gene_col: str, immune_cols: Sequence[str],
                                 coord_cols: Sequence[str] = ("x", "y"),
                                 covariate_cols: Sequence[str] = ("total_umi", "fibroblast",
                                                                  "epithelial", "endothelial"),
                                 sample_col: str = "sample", patient_col: str = "patient",
                                 rings: Sequence[tuple[int, int]] = DEFAULT_RINGS,
                                 include_self: bool = False, min_expressing_spots: int = 25,
                                 expression_threshold: float = 0.0, n_boot: int = 1000,
                                 seed: int = 0, engine: str = "statsmodels") -> pd.DataFrame:
    """Immune neighbourhood of gene-high vs gene-low spots (Methods, "Spatial
    immune-neighbourhood analysis"; Fig. 4E/F).

    Steps: (1) kNN rings per sample (:func:`knn_rings`); outcome ``Y`` for each
    index spot and immune type is the mean cell2location abundance of that
    immune type over the ring's neighbours (``include_self`` adds ring "0-0" =
    the spot itself); (2) within each sample with ``>= min_expressing_spots``
    spots expressing ``gene_col`` (> ``expression_threshold``), spots in the
    top quartile of expression are ``high=1`` and bottom quartile ``high=0``
    (middle 50% excluded); (3) per immune type and ring fit
    ``Y ~ high + z(cov_1) + ...`` (covariates of the index spot, z-scored over
    analysed spots) with random intercepts for patient and sample-within-patient
    — statsmodels ``MixedLM`` (groups=patient, variance component for sample,
    REML; Wald z p-values) or ``engine="lmerTest"`` (rpy2; Satterthwaite t-test,
    as in the paper); (4) percentage change of mean ``Y`` high vs low with a
    percentile bootstrap 95% CI resampling spots within each group
    (``n_boot``; the paper used 10,000).

    Returns a tidy table with ``immune_type, ring, beta_high, se, p, fdr,
    pct_change, pct_ci_low, pct_ci_high, mean_high, mean_low, n_high, n_low,
    n_samples, n_patients, engine``; ``fdr`` is BH across all rows.
    """
    if engine == "lmerTest" and not has_module("rpy2"):
        require("rpy2", "r")
    df = spots_df.reset_index(drop=True)
    immune_cols = list(immune_cols)
    covariate_cols = list(covariate_cols)
    coords = df[list(coord_cols)].to_numpy(float)
    sids = df[sample_col].astype(str).to_numpy()

    # (2) per-sample quartile stratification among expressing spots
    high = pd.Series(np.nan, index=df.index)
    for s, idx in df.groupby(sids).groups.items():
        g = df.loc[idx, gene_col].astype(float)
        expr = g[g > expression_threshold]
        if expr.size < min_expressing_spots:
            continue
        q_lo, q_hi = expr.quantile(0.25), expr.quantile(0.75)
        high.loc[expr.index[expr >= q_hi]] = 1.0
        high.loc[expr.index[(expr <= q_lo) & (expr < q_hi)]] = 0.0
    analysed = high.notna().to_numpy()
    if analysed.sum() == 0:
        raise ValueError("no sample had enough expressing spots")

    ring_list = ([(0, 0)] if include_self else []) + [tuple(r) for r in rings]
    knn = knn_rings(coords, [r for r in ring_list if r != (0, 0)], sids)

    base = pd.DataFrame({
        "high": high[analysed].to_numpy(),
        sample_col: sids[analysed],
        patient_col: df.loc[analysed, patient_col].astype(str).to_numpy(),
    })
    z_names = []
    for c in covariate_cols:
        zn = f"z_{c}"
        base[zn] = zscore(df.loc[analysed, c].to_numpy(float))
        z_names.append(zn)
    rhs = " + ".join(["high"] + z_names)
    rng = np.random.default_rng(seed)
    rows = []
    for ring in ring_list:
        nbr_mean = neighbor_mean_abundance(df[immune_cols], coords, sids, ring, rings=knn)
        for imm in immune_cols:
            data = base.copy()
            data["Y"] = nbr_mean[imm].to_numpy()[analysed]
            data = data.dropna(subset=["Y"])
            if engine == "lmerTest":
                beta, se, p = _fit_lmertest(data, rhs, patient_col, sample_col)
            else:
                beta, se, p = _fit_mixedlm(data, "Y ~ " + rhs, patient_col, sample_col)
            yh = data.loc[data["high"] == 1, "Y"].to_numpy()
            yl = data.loc[data["high"] == 0, "Y"].to_numpy()
            pct, lo, hi = _bootstrap_pct_change(yh, yl, n_boot, rng)
            rows.append({
                "immune_type": imm, "ring": ring_label(ring), "beta_high": beta, "se": se, "p": p,
                "pct_change": pct, "pct_ci_low": lo, "pct_ci_high": hi,
                "mean_high": yh.mean(), "mean_low": yl.mean(), "n_high": yh.size, "n_low": yl.size,
                "n_samples": data[sample_col].nunique(), "n_patients": data[patient_col].nunique(),
                "engine": engine,
            })
    out = pd.DataFrame(rows)
    out.insert(out.columns.get_loc("p") + 1, "fdr", bh_fdr(out["p"].to_numpy()))
    return out


def run_cell2location(adata_ref, adata_vis, labels_key: str, batch_key: str | None = None,
                      exclude_genes: Sequence[str] = ("CD276",), ref_max_epochs: int = 250,
                      spatial_max_epochs: int = 10000, n_cells_per_location: float = 8,
                      detection_alpha: float = 20, use_gpu: bool | None = None):
    """cell2location deconvolution with the paper's settings (Methods, "Spatial
    deconvolution"): negative-binomial regression reference signatures (250
    epochs), spatial mapping (10,000 epochs, ``N_cells_per_location=8``,
    ``detection_alpha=20``), the gene of interest excluded from the shared gene
    set to avoid circularity, and q05 posterior abundances returned as a
    spots x cell-types DataFrame (also stored in ``adata_vis.obs``).

    Not unit-tested (requires ``pip install 'vbt-harness[spatial]'``).
    """
    require("cell2location", "spatial")
    from cell2location.models import Cell2location, RegressionModel

    excl = {g.upper() for g in exclude_genes}
    keep_ref = [g for g in adata_ref.var_names if g.upper() not in excl]
    adata_ref = adata_ref[:, keep_ref].copy()
    RegressionModel.setup_anndata(adata_ref, batch_key=batch_key, labels_key=labels_key)
    ref_model = RegressionModel(adata_ref)
    train_kw = {} if use_gpu is None else {"accelerator": "gpu" if use_gpu else "cpu"}
    ref_model.train(max_epochs=ref_max_epochs, **train_kw)
    adata_ref = ref_model.export_posterior(adata_ref, sample_kwargs={"num_samples": 1000,
                                                                     "batch_size": 2500})
    fact = adata_ref.uns["mod"]["factor_names"]
    if "means_per_cluster_mu_fg" in adata_ref.varm:
        inf_aver = adata_ref.varm["means_per_cluster_mu_fg"][
            [f"means_per_cluster_mu_fg_{i}" for i in fact]].copy()
    else:
        inf_aver = adata_ref.var[[f"means_per_cluster_mu_fg_{i}" for i in fact]].copy()
    inf_aver.columns = fact

    shared = [g for g in adata_vis.var_names if g in inf_aver.index and g.upper() not in excl]
    adata_vis = adata_vis[:, shared].copy()
    inf_aver = inf_aver.loc[shared]
    Cell2location.setup_anndata(adata_vis, batch_key=None)
    model = Cell2location(adata_vis, cell_state_df=inf_aver,
                          N_cells_per_location=n_cells_per_location,
                          detection_alpha=detection_alpha)
    model.train(max_epochs=spatial_max_epochs, batch_size=None, train_size=1, **train_kw)
    adata_vis = model.export_posterior(adata_vis, sample_kwargs={"num_samples": 1000,
                                                                 "batch_size": model.adata.n_obs})
    q05 = pd.DataFrame(adata_vis.obsm["q05_cell_abundance_w_sf"], index=adata_vis.obs_names)
    q05.columns = [str(c).replace("q05cell_abundance_w_sf_", "") for c in q05.columns]
    return q05
