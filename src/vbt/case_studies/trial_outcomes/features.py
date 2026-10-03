"""Single-cell target features from a cell atlas (Tabula Sapiens).

Implements the paper's Methods section *"Single-cell feature extraction from
Tabula Sapiens"* (Zhang et al., "The Virtual Biotech", Science 2026):

* **Cell-type specificity (tau, τ)** — computed *within each tissue* over the
  mean log-normalized expression of each cell type (cell types with at least
  ``min_cells_per_type`` cells, default 20)::

      τ = Σ_j (1 − x_j / x_max) / (n − 1)

  τ = 0 means uniform expression across cell types, τ = 1 means expression in
  exactly one cell type.  Tissues where the gene is not expressed
  (``x_max == 0``) are excluded; the gene-level τ is the mean over the tissues
  where it is expressed.

* **Bimodality coefficient (BC)** — computed within each tissue over the
  *expressing* cells only (x > 0), using the Pfister et al. (2013) formula::

      BC = (m3² + 1) / (m4 + 3 (n − 1)² / ((n − 2)(n − 3)))

  with ``m3`` the bias-corrected sample skewness and ``m4`` the bias-corrected
  sample *excess* kurtosis.  BC > 5/9 ≈ 0.555 suggests bi-/multimodality.
  The gene-level BC is the mean of the per-tissue BCs.

  **Ambiguity.** The paper's text says "Pearson kurtosis" while Pfister's
  formula uses excess (Fisher) kurtosis; Pearson = excess + 3 inflates the
  denominator by 3, which is not a monotone transform across genes (per-gene
  ranks and odds ratios can change). ``kurtosis='excess'|'pearson'`` and
  ``bias_correction`` select the variant; check against the paper's Pearson
  ρ(τ, BC) = 0.54 and τ threshold 0.69 (``case1_headline.json``).

The main entry point :func:`compute_gene_features` accepts an ``AnnData`` or any
duck-typed object exposing ``.X`` (dense ``numpy`` array or ``scipy.sparse``
matrix), ``.obs`` (DataFrame), ``.var`` (DataFrame) and ``.var_names`` — so it
can be used (and tested) without ``anndata`` installed.  ``anndata`` is only
imported lazily in :func:`compute_features_from_h5ad`.

Sparse matrices are never densified as a whole: the matrix is processed one
tissue (row block) at a time, restricted to the requested genes, and within a
tissue in gene chunks.  Skewness/kurtosis are computed in a vectorized way
directly on the non-zero entries of CSC chunks.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

import logging
import re

log = logging.getLogger(__name__)

KURTOSIS_MODES = ("excess", "pearson")

__all__ = [
    "BC_THRESHOLD",
    "KURTOSIS_MODES",
    "detect_gene_id_column",
    "looks_like_counts",
    "tau_index",
    "bimodality_coefficient",
    "compute_gene_features",
    "compute_features_from_h5ad",
]

#: Conventional bimodality threshold (5/9) — BC above this suggests bimodality.
BC_THRESHOLD = 5.0 / 9.0


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------


def tau_index(mean_expr: np.ndarray) -> float:
    """Tissue-specificity index τ over per-cell-type mean expression.

    Paper Methods, "Single-cell feature extraction from Tabula Sapiens":
    τ = Σ_j (1 − x_j/x_max)/(n − 1), where x_j is the mean log-normalized
    expression of the gene in cell type j within one tissue.

    Returns ``nan`` if fewer than two cell types are available, or if the gene
    is not expressed in the tissue (``x_max == 0``; such tissues are excluded
    from the gene-level average).  NaN entries are ignored.
    """
    x = np.asarray(mean_expr, dtype=float).ravel()
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2:
        return float("nan")
    x_max = x.max()
    if not x_max > 0:
        return float("nan")
    return float(np.sum(1.0 - x / x_max) / (n - 1))


def _check_kurtosis(kurtosis: str) -> None:
    if kurtosis not in KURTOSIS_MODES:
        raise ValueError(f"kurtosis must be one of {KURTOSIS_MODES}, got {kurtosis!r}")


def _bc_from_moments(n: np.ndarray, g1: np.ndarray, g2: np.ndarray, kurtosis: str = "excess") -> np.ndarray:
    """BC given n, skewness g1 and *excess* kurtosis g2 (``kurtosis='pearson'`` adds 3)."""
    _check_kurtosis(kurtosis)
    n = np.asarray(n, dtype=float)
    k = g2 + 3.0 if kurtosis == "pearson" else g2
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = 3.0 * (n - 1.0) ** 2 / ((n - 2.0) * (n - 3.0))
        bc = (g1**2 + 1.0) / (k + corr)
    bc = np.where(n < 4, np.nan, bc)
    return bc


def bimodality_coefficient(x: np.ndarray, kurtosis: str = "excess", bias_correction: bool = True) -> float:
    """Bimodality coefficient (Pfister et al. 2013) over expressing cells.

    Paper Methods, "Single-cell feature extraction from Tabula Sapiens":
    BC = (m3² + 1) / (m4 + 3(n − 1)²/((n − 2)(n − 3))), computed on the cells
    with x > 0 only (n = number of expressing cells), where m3 is the
    bias-corrected skewness (``scipy.stats.skew(bias=False)``) and m4 the
    bias-corrected *excess* kurtosis
    (``scipy.stats.kurtosis(fisher=True, bias=False)``).

    ``kurtosis='pearson'`` uses Pearson kurtosis (excess + 3) in the
    denominator, as the paper's text says; ``bias_correction=False`` uses the
    biased (population) skewness and kurtosis.

    Returns ``nan`` if n < 4 or the expressing values have zero variance.
    """
    from scipy import stats

    _check_kurtosis(kurtosis)
    v = np.asarray(x, dtype=float).ravel()
    v = v[np.isfinite(v) & (v > 0)]
    n = v.size
    if n < 4 or np.ptp(v) == 0:
        return float("nan")
    g1 = stats.skew(v, bias=not bias_correction)
    g2 = stats.kurtosis(v, fisher=True, bias=not bias_correction)
    return float(_bc_from_moments(np.array([n]), np.array([g1]), np.array([g2]), kurtosis)[0])


# ---------------------------------------------------------------------------
# Vectorized helpers for sparse / dense chunks
# ---------------------------------------------------------------------------


def _sparse_module():
    import scipy.sparse as sp

    return sp


def _is_sparse(m) -> bool:
    sp = _sparse_module()
    return sp.issparse(m)


def _bc_columns(chunk, kurtosis: str = "excess", bias_correction: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized BC per column of ``chunk`` over entries > 0.

    ``chunk`` is a scipy.sparse matrix (any format) or a dense array
    (cells × genes).  Returns (bc, n_expressing) arrays of length n_genes.
    """
    sp = _sparse_module()
    n_genes = chunk.shape[1]
    if sp.issparse(chunk):
        csc = sp.csc_matrix(chunk)
        counts_all = np.diff(csc.indptr)
        col_ids = np.repeat(np.arange(n_genes), counts_all)
        data = np.asarray(csc.data, dtype=float)
    else:
        dense = np.asarray(chunk, dtype=float)
        rows, col_ids = np.nonzero(dense)
        data = dense[rows, col_ids]
    keep = np.isfinite(data) & (data > 0)
    data = data[keep]
    col_ids = col_ids[keep]

    n = np.bincount(col_ids, minlength=n_genes).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.bincount(col_ids, weights=data, minlength=n_genes) / n
        d = data - mean[col_ids]
        m2 = np.bincount(col_ids, weights=d**2, minlength=n_genes) / n
        m3 = np.bincount(col_ids, weights=d**3, minlength=n_genes) / n
        m4 = np.bincount(col_ids, weights=d**4, minlength=n_genes) / n
        # Treat numerically-zero variance as constant -> undefined BC.
        scale = np.maximum(np.abs(mean), 1e-300)
        const = m2 <= (1e-12 * scale) ** 2
        # Biased (population) moments -> bias-corrected estimators (scipy's
        # skew(bias=False) and kurtosis(fisher=True, bias=False)).
        g1_b = m3 / m2**1.5
        if bias_correction:
            g1 = np.sqrt(n * (n - 1.0)) / (n - 2.0) * g1_b
            g2 = ((n**2 - 1.0) * m4 / m2**2 - 3.0 * (n - 1.0) ** 2) / ((n - 2.0) * (n - 3.0))
        else:
            g1 = g1_b
            g2 = m4 / m2**2 - 3.0
    bc = _bc_from_moments(n, g1, g2, kurtosis)
    bc = np.where(const, np.nan, bc)
    return bc, n.astype(int)


def _tau_matrix(means: np.ndarray) -> np.ndarray:
    """τ per column of a (n_celltypes × n_genes) matrix of means."""
    n_types = means.shape[0]
    if n_types < 2:
        return np.full(means.shape[1], np.nan)
    x_max = means.max(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = np.sum(1.0 - means / x_max[None, :], axis=0) / (n_types - 1)
    return np.where(x_max > 0, tau, np.nan)


def _get_matrix(adata, layer: str | None):
    if layer is None:
        return adata.X
    layers = getattr(adata, "layers", None)
    if layers is None or layer not in layers:
        raise KeyError(f"layer {layer!r} not found on the AnnData object")
    return layers[layer]


def _row_block(matrix, rows: np.ndarray, cols: np.ndarray | None):
    """Load the rows × cols block of ``matrix`` (in-memory or backed)."""
    sp = _sparse_module()
    rows = np.sort(np.asarray(rows))
    if sp.issparse(matrix):
        block = matrix[rows]
        if cols is not None:
            block = block[:, cols]
        return sp.csc_matrix(block)
    # Backed anndata sparse dataset / h5py dataset / numpy array.
    try:
        block = matrix[rows]
    except TypeError:  # pragma: no cover - exotic backends
        block = matrix[list(rows)]
    if hasattr(block, "to_memory"):  # pragma: no cover - anndata backed
        block = block.to_memory()
    if sp.issparse(block):
        if cols is not None:
            block = block[:, cols]
        return sp.csc_matrix(block)
    block = np.asarray(block)
    if cols is not None:
        block = block[:, cols]
    return block


_ENSG = re.compile(r"^ENS[A-Z]*G\d+(\.\d+)?$")


def detect_gene_id_column(adata, preferred: str | None = None) -> str | None:
    """Where the Ensembl gene IDs live: ``None`` (= ``var_names``) when most
    var_names are ENSG IDs (CELLxGENE-distributed h5ad files), else
    ``preferred`` / ``ensembl_id`` / ``feature_id`` / ``gene_ids`` if present in
    ``var``. Raises when none is found."""
    names = pd.Index(np.asarray(adata.var_names).astype(str))
    sample = names[: min(len(names), 2000)]
    if len(sample) and np.mean([bool(_ENSG.match(x)) for x in sample]) > 0.8:
        return None
    var = adata.var
    for c in [preferred, "ensembl_id", "feature_id", "gene_ids", "gene_id"]:
        if c and c in var.columns:
            vals = var[c].astype(str).head(2000)
            if len(vals) and np.mean([bool(_ENSG.match(x)) for x in vals]) > 0.5:
                return c
    raise KeyError("no Ensembl gene IDs found: var_names are not ENSG IDs and none of "
                   f"{[c for c in [preferred, 'ensembl_id', 'feature_id', 'gene_ids'] if c]} holds them; "
                   "pass --gene-id-column")


def looks_like_counts(matrix, n_rows: int = 500) -> bool:
    """True when a sample of ``matrix`` is integer-valued (raw counts, not log-normalised)."""
    sp = _sparse_module()
    n = matrix.shape[0]
    rows = np.unique(np.linspace(0, max(n - 1, 0), num=min(n, n_rows)).astype(int))
    block = _row_block(matrix, rows, None)
    data = np.asarray(block.data if sp.issparse(block) else block, dtype=float).ravel()
    data = data[np.isfinite(data) & (data != 0)]
    if data.size == 0:
        return False
    return bool(np.all(np.mod(data, 1.0) == 0) and data.max() > 1)


def _normalize_log1p(block, target_sum: float = 1e4):
    """normalize_total(target_sum) + log1p on a cells x genes block (all genes)."""
    sp = _sparse_module()
    if sp.issparse(block):
        csr = sp.csr_matrix(block, dtype=float)
        lib = np.asarray(csr.sum(axis=1)).ravel()
        scale = np.divide(target_sum, lib, out=np.zeros_like(lib), where=lib > 0)
        out = sp.diags(scale) @ csr
        out = sp.csr_matrix(out)
        out.data = np.log1p(out.data)
        return sp.csc_matrix(out)
    dense = np.asarray(block, dtype=float)
    lib = dense.sum(axis=1)
    scale = np.divide(target_sum, lib, out=np.zeros_like(lib), where=lib > 0)
    return np.log1p(dense * scale[:, None])


def _gene_index(adata, gene_id_column: str | None) -> pd.Index:
    var_names = pd.Index(np.asarray(adata.var_names).astype(str))
    if gene_id_column is None:
        return var_names
    var = adata.var
    if gene_id_column not in var.columns:
        raise KeyError(f"gene_id_column {gene_id_column!r} not in adata.var")
    ids = var[gene_id_column].astype("string")
    # Strip Ensembl version suffixes (ENSG00000123.4 -> ENSG00000123).
    ids = ids.str.replace(r"^(ENS[A-Z]*G\d+)\.\d+$", r"\1", regex=True)
    return pd.Index(ids.to_numpy(dtype=object))


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def compute_gene_features(
    adata,
    genes: Iterable[str] | None = None,
    tissue_key: str = "tissue",
    celltype_key: str = "cell_ontology_class",
    min_cells_per_type: int = 20,
    layer: str | None = None,
    gene_id_column: str | None = None,
    per_tissue: bool = False,
    chunk_size: int = 2000,
    kurtosis: str = "excess",
    bias_correction: bool = True,
    normalize: bool = False,
) -> pd.DataFrame:
    """Compute τ and bimodality per gene from a (log-normalized) cell atlas.

    Implements paper Methods "Single-cell feature extraction from Tabula
    Sapiens".  For every tissue in ``adata.obs[tissue_key]``:

    * τ is computed over cell types (``adata.obs[celltype_key]``) having at
      least ``min_cells_per_type`` cells in that tissue, from the mean
      expression of each cell type (``adata.X`` is assumed already
      log-normalized unless ``layer`` is given);
    * BC is computed over the expressing cells (x > 0) of the whole tissue.

    Gene-level aggregates: ``tau`` = mean τ over tissues where the gene is
    expressed (tissue τ not NaN); ``bimodality`` = mean of the (non-NaN)
    per-tissue BCs; ``n_tissues_expressed`` = number of tissues with a
    non-NaN τ.

    Parameters
    ----------
    adata
        ``AnnData`` or duck-typed object with ``X``, ``obs``, ``var`` and
        ``var_names`` (and ``layers`` if ``layer`` is used).  ``X`` may be a
        dense array or any ``scipy.sparse`` matrix; it is never densified as a
        whole.
    genes
        Optional subset of genes, matched against the output key (Ensembl IDs
        when ``gene_id_column`` is given, else ``var_names``) and also against
        ``var_names``.
    gene_id_column
        Column of ``adata.var`` holding the identifiers to key the output by,
        e.g. ``"ensembl_id"`` so features join to Open Targets ``targetId``
        (ENSG...).  Version suffixes are stripped; genes with missing IDs are
        dropped; for duplicated IDs the first occurrence is kept.
    per_tissue
        If True, return the long per-(gene, tissue) table instead, with columns
        ``gene, tissue, tau, bimodality, n_celltypes, n_expressing``.
    chunk_size
        Number of genes processed at once inside a tissue.
    kurtosis, bias_correction
        Bimodality-coefficient variant (see :func:`bimodality_coefficient`).
    normalize
        The matrix holds raw counts: apply normalize_total(1e4) + log1p per
        cell (library size over all genes) before computing features.

    Returns
    -------
    DataFrame indexed by gene (name ``"gene"``) with columns ``tau``,
    ``bimodality``, ``n_tissues_expressed`` (or the long table if
    ``per_tissue``).
    """
    _check_kurtosis(kurtosis)
    obs = adata.obs
    for key in (tissue_key, celltype_key):
        if key not in obs.columns:
            raise KeyError(f"{key!r} not found in adata.obs")
    keys = _gene_index(adata, gene_id_column)
    var_names = pd.Index(np.asarray(adata.var_names).astype(str))

    valid = ~pd.isna(pd.Series(keys, dtype=object)).to_numpy()
    valid &= ~pd.Index(keys).duplicated(keep="first")
    if genes is not None:
        wanted = set(map(str, genes))
        in_set = np.array([str(k) in wanted for k in keys]) | var_names.isin(wanted)
        valid &= in_set
    cols = np.flatnonzero(valid)
    out_keys = pd.Index([str(k) for k in keys[cols]], name="gene")
    n_genes = len(cols)

    matrix = _get_matrix(adata, layer)
    tissues = obs[tissue_key].astype("string").to_numpy(dtype=object)
    celltypes = obs[celltype_key].astype("string").to_numpy(dtype=object)
    tissue_levels = [t for t in pd.unique(tissues) if not pd.isna(t)]

    tau_rows, bc_rows, long_parts = [], [], []
    kept_log: dict[str, dict[str, int]] = {}
    for tissue in tissue_levels:
        rows = np.flatnonzero(tissues == tissue)
        rows = np.sort(rows)
        if normalize:
            block = _normalize_log1p(_row_block(matrix, rows, None))
            if n_genes < matrix.shape[1]:
                block = block[:, cols]
        else:
            block = _row_block(matrix, rows, cols if n_genes < matrix.shape[1] else None)
        ct = celltypes[rows]
        ct_series = pd.Series(ct, dtype=object)
        sizes = ct_series.value_counts(dropna=True)
        kept_types = [c for c, s in sizes.items() if s >= min_cells_per_type]
        kept_log[str(tissue)] = {"n_cells": int(len(rows)), "n_celltypes": int(len(sizes)),
                                 "n_celltypes_kept": int(len(kept_types))}
        tau_t = np.full(n_genes, np.nan)
        bc_t = np.full(n_genes, np.nan)
        nexp_t = np.zeros(n_genes, dtype=int)

        indicator = None
        if len(kept_types) >= 2:
            sp = _sparse_module()
            code = {c: i for i, c in enumerate(kept_types)}
            r_idx = np.array([code.get(c, -1) for c in ct])
            m = r_idx >= 0
            weights = 1.0 / np.array([sizes[c] for c in kept_types], dtype=float)
            indicator = sp.csr_matrix(
                (weights[r_idx[m]], (r_idx[m], np.flatnonzero(m))),
                shape=(len(kept_types), len(rows)),
            )

        for start in range(0, n_genes, chunk_size):
            stop = min(start + chunk_size, n_genes)
            sub = block[:, start:stop]
            if indicator is not None:
                means = indicator @ sub
                means = means.toarray() if _is_sparse(means) else np.asarray(means)
                tau_t[start:stop] = _tau_matrix(means)
            bc, nexp = _bc_columns(sub, kurtosis, bias_correction)
            bc_t[start:stop] = bc
            nexp_t[start:stop] = nexp

        tau_rows.append(tau_t)
        bc_rows.append(bc_t)
        if per_tissue:
            long_parts.append(
                pd.DataFrame(
                    {
                        "gene": out_keys,
                        "tissue": tissue,
                        "tau": tau_t,
                        "bimodality": bc_t,
                        "n_celltypes": len(kept_types),
                        "n_expressing": nexp_t,
                    }
                )
            )

    n_tau = sum(1 for v in kept_log.values() if v["n_celltypes_kept"] >= 2)
    log.info("features: %d tissues, %d with >=2 cell types of >=%d cells (tau computed); "
             "%d of %d cell types kept by the min-%d rule", len(kept_log), n_tau, min_cells_per_type,
             sum(v["n_celltypes_kept"] for v in kept_log.values()),
             sum(v["n_celltypes"] for v in kept_log.values()), min_cells_per_type)
    attrs = {"kept_by_tissue": kept_log, "n_tissues": len(kept_log), "n_tissues_with_tau": n_tau,
             "min_cells_per_type": min_cells_per_type, "bimodality_kurtosis": kurtosis,
             "bimodality_bias_correction": bias_correction, "normalized_from_counts": bool(normalize)}

    if per_tissue:
        if not long_parts:
            return pd.DataFrame(
                columns=["gene", "tissue", "tau", "bimodality", "n_celltypes", "n_expressing"]
            )
        out = pd.concat(long_parts, ignore_index=True)
        out.attrs.update(attrs)
        return out

    if tau_rows:
        tau_mat = np.vstack(tau_rows)
        bc_mat = np.vstack(bc_rows)
        n_expr = np.sum(~np.isnan(tau_mat), axis=0)
        with np.errstate(invalid="ignore"), _silence_mean_warning():
            tau = np.nanmean(tau_mat, axis=0)
            bcm = np.nanmean(bc_mat, axis=0)
    else:
        tau = bcm = np.full(n_genes, np.nan)
        n_expr = np.zeros(n_genes, dtype=int)
    out = pd.DataFrame(
        {"tau": tau, "bimodality": bcm, "n_tissues_expressed": n_expr.astype(int)},
        index=out_keys,
    )
    out.attrs.update(attrs)
    return out


class _silence_mean_warning:
    """Context manager silencing 'Mean of empty slice' RuntimeWarnings."""

    def __enter__(self):
        import warnings

        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return self

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


def compute_features_from_h5ad(
    path: str,
    gene_ids: Sequence[str] | None = None,
    backed: str | bool | None = "r",
    *,
    gene_id_column: str | None = None,
    layer: str | None = None,
    normalize: str = "auto",
    **kwargs,
) -> pd.DataFrame:
    """Compute τ / BC features directly from an ``.h5ad`` atlas file.

    Lazily imports ``anndata`` (not a hard dependency).  By default the file is
    opened in backed read-only mode and processed tissue by tissue (row
    blocks), so the full expression matrix is never loaded; pass
    ``backed=None`` to load it into memory instead.  ``gene_ids`` restricts the
    computation (matched against ``gene_id_column`` values or ``var_names``);
    all other keyword arguments go to :func:`compute_gene_features`.

    Gene IDs: ``gene_id_column=None`` auto-detects (``var_names`` when they
    are ENSG IDs, as in CELLxGENE downloads; else ``ensembl_id`` /
    ``feature_id``) so the output joins to Open Targets ``targetId``.
    ``layer`` selects the matrix. Integer-valued (raw-count) matrices are
    normalised (normalize_total 1e4 + log1p) when ``normalize='auto'`` or
    ``'always'``; ``'never'`` raises instead. The tissue and cell-type counts
    kept by the min-cells rule are logged and stored in ``df.attrs``.
    """
    if normalize not in ("auto", "always", "never"):
        raise ValueError("normalize must be 'auto', 'always' or 'never'")
    try:
        import anndata  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "anndata is required to read .h5ad files: pip install anndata"
        ) from exc
    adata = anndata.read_h5ad(path, backed=backed if backed else None)
    try:
        gid = detect_gene_id_column(adata, gene_id_column)
        log.info("gene IDs from %s", "var_names" if gid is None else f"var[{gid!r}]")
        matrix = _get_matrix(adata, layer)
        counts = normalize == "always" or looks_like_counts(matrix)
        if counts and normalize == "never":
            raise ValueError(f"{'layer ' + layer if layer else 'X'} looks like raw counts (integer-valued); "
                             "tau/bimodality need log-normalised expression: pass --normalize auto or a "
                             "log-normalised --layer")
        if counts:
            log.warning("%s holds raw counts: applying normalize_total(1e4) + log1p per cell",
                        f"layer {layer!r}" if layer else "X")
        df = compute_gene_features(adata, genes=gene_ids, gene_id_column=gid, layer=layer,
                                   normalize=counts, **kwargs)
        print(f"features: {df.attrs.get('n_tissues')} tissues ({df.attrs.get('n_tissues_with_tau')} with tau); "
              + ", ".join(f"{t}: {v['n_celltypes_kept']}/{v['n_celltypes']} cell types"
                          for t, v in list(df.attrs.get("kept_by_tissue", {}).items())[:40]))
        return df
    finally:
        file = getattr(adata, "file", None)
        if backed and file is not None:
            try:
                file.close()
            except Exception:  # pragma: no cover
                pass
