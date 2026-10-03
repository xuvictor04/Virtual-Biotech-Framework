"""Cross-disease OSMR expression analysis from case study 3 (CELLxGENE Census).

Implements the Supplementary Methods section "Cross-disease expression
analysis": cells from CELLxGENE Census grouped into donor x cell-type
pseudobulks (>= 25 cells) summarised as the mean per-cell log2 counts per
10,000 (log2 CP10K) of OSMR; disease vs normal compared within each disease
and cell type with a weighted linear mixed model (weights = number of cells;
covariates log sequencing depth and assay group; random intercept for dataset)
fitted with lmerTest, falling back to weighted OLS when fewer than two
datasets contribute; Benjamini-Hochberg correction across all tests.

Harness details (see each function): pseudobulks are keyed by
(dataset, donor, disease, tissue, cell type) so one donor's disease and normal
cells or several tissues are never pooled; normal arms are chosen with
same-dataset > same-assay > same-tissue priority; diseases pass a QC filter
(>= 3 donors per arm by default; the paper surveyed 112 diseases); several
genes can be pseudobulked at once (``gene`` column) with one global BH; cell
types are harmonised to Cell Ontology Level 1 (:mod:`vbt.analysis.ontology`).
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from ._utils import bh_fdr, has_module, require

__all__ = ["pseudobulk_log2cp10k", "pair_with_normals", "disease_qc", "disease_vs_normal_lmm", "census_query_plan",
           "DEFAULT_PSEUDOBULK_KEYS", "r_package_available"]

#: Pseudobulk keys: a donor's disease and normal cells, tissues and datasets stay separate.
DEFAULT_PSEUDOBULK_KEYS: tuple[str, ...] = ("dataset_id", "donor_id", "disease", "tissue_general", "cell_type")


def r_package_available(pkg: str) -> bool:
    """rpy2 importable *and* the R package installed (``rpy2.robjects.packages.isinstalled``)."""
    try:
        from rpy2.robjects.packages import isinstalled  # type: ignore
    except Exception:  # noqa: BLE001 - ImportError or R initialisation failure
        return False
    try:
        return bool(isinstalled(pkg))
    except Exception:  # noqa: BLE001
        return False


def pseudobulk_log2cp10k(counts_gene, total_counts, obs: pd.DataFrame, donor_col: str = "donor_id",
                         celltype_col: str = "cell_type", min_cells: int = 25,
                         carry_cols: Sequence[str] = ("dataset_id", "disease", "condition", "assay",
                                                      "assay_group", "tissue_general"),
                         gene: str | Sequence[str] = "OSMR", keys: Sequence[str] | None = None,
                         strict_carry: bool = True) -> pd.DataFrame:
    """Per-group summaries of one or more genes (Methods, "Cross-disease
    expression analysis").

    ``counts_gene`` are per-cell raw counts (a vector for one gene, or a cells x
    genes matrix with ``gene`` a list of names) and ``total_counts`` per-cell
    library sizes. Groups are ``keys`` (default :data:`DEFAULT_PSEUDOBULK_KEYS`
    = dataset x donor x disease x tissue x cell type, restricted to the columns
    present in ``obs``; ``donor_col`` and ``celltype_col`` are always keys), so
    a donor's tumour and adjacent-normal cells, several tissues or datasets are
    never pooled. Per group: ``n_cells`` (groups with fewer than ``min_cells``
    dropped), ``mean_log2_<gene>_cp10k`` (mean over cells of log2(1 + 1e4 *
    count / total)), ``pseudobulk_log2_<gene>_cp10k`` (log2(1 + 1e4 * sum counts
    / sum totals)), ``frac_expressing``, ``log_depth`` (log10 mean total counts
    per cell) and the ``carry_cols`` present in ``obs`` (which must be constant
    within a group: ValueError when ``strict_carry``). Several genes give a long
    table with a ``gene`` column and generic ``mean_log2_cp10k`` /
    ``pseudobulk_log2_cp10k`` columns (a single gene also gets a ``gene``
    column).
    """
    genes = [gene] if isinstance(gene, str) else list(gene)
    C = counts_gene.toarray() if hasattr(counts_gene, "toarray") else np.asarray(counts_gene, dtype=float)
    C = np.asarray(C, dtype=float)
    if C.ndim == 1:
        C = C.reshape(-1, 1)
    if C.shape[1] != len(genes):
        raise ValueError(f"counts have {C.shape[1]} gene column(s) but {len(genes)} gene name(s) were given")
    t = np.asarray(total_counts, dtype=float).ravel()
    if keys is None:
        keys = DEFAULT_PSEUDOBULK_KEYS
    keys = [k for k in dict.fromkeys(keys) if k in obs.columns]
    for k in (donor_col, celltype_col):
        if k not in keys:
            keys.append(k)
    carry = [c for c in carry_cols if c in obs.columns and c not in keys]
    frame = pd.DataFrame({k: obs[k].astype(str).to_numpy() for k in keys})
    codes, uniq = pd.factorize(frame.agg("|".join, axis=1) if len(keys) > 1 else frame.iloc[:, 0], sort=True)
    for c in carry:
        nun = pd.Series(obs[c].to_numpy()).groupby(codes).nunique(dropna=False)
        if strict_carry and (nun > 1).any():
            raise ValueError(f"carry column {c!r} varies within pseudobulk groups {keys}; add it to keys")
    n_groups = len(uniq)
    n_cells = np.bincount(codes, minlength=n_groups)
    t_sum = np.bincount(codes, weights=t, minlength=n_groups)
    t_mean = t_sum / np.maximum(n_cells, 1)
    first = pd.Series(np.arange(len(codes))).groupby(codes).first().to_numpy()
    meta = pd.DataFrame({k: obs[k].to_numpy()[first] for k in keys})
    for c in carry:
        meta[c] = obs[c].to_numpy()[first]
    meta["n_cells"] = n_cells
    meta["log_depth"] = np.log10(np.clip(t_mean, 1, None))
    parts = []
    for j, g in enumerate(genes):
        c = C[:, j]
        cp10k = np.divide(c * 1e4, t, out=np.zeros_like(c), where=t > 0)
        l = np.log2(1 + cp10k)
        g_sum = np.bincount(codes, weights=c, minlength=n_groups)
        part = meta.copy()
        part["gene"] = g
        mean_l = np.bincount(codes, weights=l, minlength=n_groups) / np.maximum(n_cells, 1)
        pb = np.log2(1 + 1e4 * g_sum / np.where(t_sum > 0, t_sum, np.nan))
        if len(genes) == 1:
            part[f"mean_log2_{g}_cp10k"] = mean_l
            part[f"pseudobulk_log2_{g}_cp10k"] = pb
        else:
            part["mean_log2_cp10k"] = mean_l
            part["pseudobulk_log2_cp10k"] = pb
        part["frac_expressing"] = np.bincount(codes, weights=(c > 0).astype(float), minlength=n_groups) \
            / np.maximum(n_cells, 1)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out = out[out["n_cells"] >= min_cells].reset_index(drop=True)
    out.attrs["keys"] = keys
    return out


def pair_with_normals(df: pd.DataFrame, disease_col: str = "disease_name",
                      ref_level: str = "normal", match_on: Sequence[str] = ("tissue_general",),
                      priority: Sequence[str] = ("dataset_id", "assay"),
                      by: Sequence[str] | None = ("cell_type",)) -> pd.DataFrame:
    """Build per-disease comparison sets with prioritised normal arms.

    For every non-normal disease (and, when present, each ``by`` stratum such
    as cell type) the ``ref_level`` pseudobulks used as controls are, in order
    of preference: those from the **same dataset** (``priority[0]``), else the
    **same assay** (``priority[1]``), else any — always restricted to the same
    ``match_on`` columns (tissue). Normal rows are relabelled with the disease
    so strata ``by=(disease_col, "cell_type")`` contain both arms. Adds
    ``condition`` ("disease"/``ref_level``) if absent and ``normal_match``
    (``dataset_id`` | ``assay`` | ``tissue``/``any``) on every row.
    """
    d = df.copy()
    if "condition" not in d.columns:
        d["condition"] = np.where(d[disease_col].astype(str) == ref_level, ref_level, "disease")
    is_ref = d[disease_col].astype(str) == ref_level
    normals, dis = d[is_ref], d[~is_ref]
    match_on = [m for m in match_on if m in d.columns]
    priority = [c for c in priority if c in d.columns]
    by = [b for b in (by or []) if b in d.columns]
    base_label = "tissue" if match_on else "any"
    parts = []
    for name, grp in dis.groupby(disease_col, observed=True, sort=True):
        strata = grp.groupby(by, observed=True, sort=True) if by else [((), grp)]
        for key, sub in strata:
            nm = normals
            if by:
                kk = key if isinstance(key, tuple) else (key,)
                for b, v in zip(by, kk):
                    nm = nm[nm[b] == v]
            if match_on:
                nm = nm.merge(sub[match_on].drop_duplicates(), on=match_on, how="inner")
            label = base_label
            for col in priority:
                cand = nm[nm[col].isin(set(sub[col].dropna()))]
                if len(cand):
                    nm, label = cand, col
                    break
            parts.append(pd.concat([sub.assign(normal_match=label),
                                    nm.assign(**{disease_col: name, "normal_match": label})], ignore_index=True))
    return pd.concat(parts, ignore_index=True) if parts else d.iloc[0:0]


def disease_qc(paired: pd.DataFrame, disease_col: str = "disease_name", condition: str = "condition",
               donor_col: str = "donor_id", ref_level: str = "normal", min_donors_per_arm: int = 3,
               min_datasets: int = 1, dataset_col: str = "dataset_id") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Disease-level quality filter for the cross-disease survey.

    A disease passes when each arm has ``>= min_donors_per_arm`` distinct
    donors (default 3) and its disease arm comes from ``>= min_datasets``
    datasets. Returns ``(filtered_paired_table, report)``; the report has one
    row per disease with donor/dataset counts and ``passed``, and
    ``report.attrs['n_passing']`` (paper: 112 diseases surveyed).
    """
    rows = []
    for name, g in paired.groupby(disease_col, observed=True, sort=True):
        is_ref = g[condition].astype(str) == ref_level
        dcol = donor_col if donor_col in g.columns else None
        n_dis = g.loc[~is_ref, dcol].nunique() if dcol else int((~is_ref).sum())
        n_norm = g.loc[is_ref, dcol].nunique() if dcol else int(is_ref.sum())
        n_ds = g.loc[~is_ref, dataset_col].nunique() if dataset_col in g.columns else 1
        ok = min(n_dis, n_norm) >= min_donors_per_arm and n_ds >= min_datasets
        rows.append({disease_col: name, "n_donors_disease": int(n_dis), "n_donors_normal": int(n_norm),
                     "n_datasets_disease": int(n_ds), "passed": bool(ok)})
    report = pd.DataFrame(rows, columns=[disease_col, "n_donors_disease", "n_donors_normal",
                                         "n_datasets_disease", "passed"])
    keep = set(report.loc[report["passed"], disease_col]) if len(report) else set()
    report.attrs["n_passing"] = len(keep)
    report.attrs["n_total"] = len(report)
    return paired[paired[disease_col].isin(keep)].reset_index(drop=True), report


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


def _fit_sqrtw_mixedlm(d: pd.DataFrame, rhs: str, value: str, group: str, weights: str):
    """APPROXIMATE weighted LMM: statsmodels MixedLM has no observation weights, so
    rows (response, fixed design and random-intercept design) are scaled by
    sqrt(weight). This reproduces weighted least squares for the fixed part but
    also rescales the random intercept per row, so it only approximates lmerTest's
    weighted model (flagged in ``method``)."""
    import patsy
    import statsmodels.api as sm

    from ._utils import fit_mixedlm

    y, X = patsy.dmatrices(f"{value} ~ {rhs}", d, return_type="dataframe")
    sw = np.sqrt(d.loc[y.index, weights].to_numpy(float))
    Xs = X.to_numpy(float) * sw[:, None]
    ys = y.to_numpy(float).ravel() * sw
    groups = d.loc[y.index, group].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.MixedLM(ys, Xs, groups=groups, exog_re=sw.reshape(-1, 1))
    fit = fit_mixedlm(model)
    j = list(X.columns).index("disease")
    fe, bse = np.asarray(fit.fe_params), np.asarray(fit.bse_fe)
    from scipy import stats

    z = fe[j] / bse[j]
    return float(fe[j]), float(bse[j]), float(2 * stats.norm.sf(abs(z))), "mixedlm_sqrt_weight_APPROX"


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
    * ``"mixedlm_sqrtw"``: an *approximate* weighted LMM (statsmodels MixedLM
      on sqrt(weight)-scaled rows), flagged ``mixedlm_sqrt_weight_APPROX``;
      WLS when < 2 datasets.
    * ``"auto"``: lmerTest only when rpy2 *and* the R package lmerTest are
      installed (``isinstalled('lmerTest')``), else python.

    A ``gene`` column (multi-gene pseudobulks) is added to the strata
    automatically, so ``fdr`` is one global BH over genes x diseases x cell types.

    Covariates that are constant within a stratum are dropped. ``by`` defines
    strata (``None`` = one model). ``fdr`` is BH across all strata (global).
    Returns ``[by...], beta, se, p, fdr, n_obs, n_disease, n_normal,
    n_datasets, method``.
    """
    if engine == "auto":
        engine = "lmerTest" if r_package_available("lmerTest") else "python"
    if engine not in ("lmerTest", "python", "mixedlm_sqrtw"):
        raise ValueError("engine must be 'auto', 'lmerTest', 'python' or 'mixedlm_sqrtw'")
    by = list(by) if by else []
    if "gene" in df.columns and "gene" not in by and df["gene"].nunique() > 1:
        by = ["gene", *by]
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
            elif engine == "mixedlm_sqrtw" and d[group].nunique() >= 2:
                beta, se, p, method = _fit_sqrtw_mixedlm(d, rhs, value, group, weights)
            else:
                beta, se, p, method = _fit_weighted_python(d, f"{value} ~ {rhs}", group, weights)
        except Exception as exc:  # pragma: no cover - numerical failure
            beta, se, p, method = np.nan, np.nan, np.nan, f"failed: {exc}"
        rows.append({**row, "beta": beta, "se": se, "p": p, "method": method})
    out = pd.DataFrame(rows)
    out["fdr"] = bh_fdr(out["p"].to_numpy())
    out["engine"] = engine
    cols = by + ["beta", "se", "p", "fdr", "n_obs", "n_disease", "n_normal", "n_datasets", "method", "engine"]
    return out[cols]


def census_query_plan(gene: str | Sequence[str] = "OSMR", diseases: Sequence[str] | None = None,
                      tissue_general: Sequence[str] | None = None,
                      census_version: str = "stable", organism: str = "homo_sapiens",
                      execute: bool = False, min_cells: int = 25, ontology=None, celltype_level: int = 1,
                      keys: Sequence[str] | None = None):
    """Build (and optionally run) the CELLxGENE Census query used for the
    cross-disease analysis (Methods, "Cross-disease expression analysis").

    The plan restricts to primary (``is_primary_data == True``) cells of the
    requested diseases plus ``normal`` and fetches the raw counts of ``gene``
    (one name or a list) with per-cell ``raw_sum`` (library size) and donor /
    cell type (+ ontology term id) / dataset / assay / disease / tissue
    annotations. Cell types are harmonised to Cell Ontology Level
    ``celltype_level`` (default 1; ``ontology`` from
    :func:`vbt.analysis.ontology.load_cell_ontology`, or ``$VBT_CL_OBO``; raw
    Census ``cell_type`` when no ontology is available). With ``execute=True``
    (requires ``pip install 'vbt-harness[singlecell]'`` and network) returns the
    pseudobulk table from :func:`pseudobulk_log2cp10k` (keys: dataset x donor x
    disease x tissue x Level-k cell type; one row per gene); otherwise returns
    the plan dict.
    """
    genes = [gene] if isinstance(gene, str) else list(gene)
    filt = ["is_primary_data == True"]
    if diseases:
        dl = sorted(set(diseases) | {"normal"})
        filt.append(f"disease in {dl!r}")
    if tissue_general:
        filt.append(f"tissue_general in {sorted(tissue_general)!r}")
    celltype_col = f"cell_type_level{celltype_level}"
    plan = {
        "census_version": census_version,
        "organism": organism,
        "genes": genes,
        "var_value_filter": (f"feature_name == '{genes[0]}'" if len(genes) == 1
                             else f"feature_name in {genes!r}"),
        "obs_value_filter": " and ".join(filt),
        "obs_column_names": ["soma_joinid", "dataset_id", "donor_id", "cell_type", "cell_type_ontology_term_id",
                             "disease", "assay", "tissue_general", "raw_sum", "is_primary_data"],
        "min_cells": min_cells,
        "celltype_level": celltype_level,
        "celltype_col": celltype_col,
        "pseudobulk_keys": list(keys or ("dataset_id", "donor_id", "disease_name", "tissue_general",
                                         celltype_col)),
    }
    if not execute:
        return plan
    census_mod = require("cellxgene_census")
    with census_mod.open_soma(census_version=census_version) as census:
        adata = census_mod.get_anndata(census, organism=organism,
                                       var_value_filter=plan["var_value_filter"],
                                       obs_value_filter=plan["obs_value_filter"],
                                       obs_column_names=plan["obs_column_names"])
    names = list(map(str, adata.var["feature_name"])) if "feature_name" in adata.var else list(genes)
    order = [names.index(g) for g in genes if g in names]
    X = adata.X[:, order]
    counts = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    obs = adata.obs.copy()
    obs["condition"] = np.where(obs["disease"].astype(str) == "normal", "normal", "disease")
    obs["disease_name"] = obs["disease"].astype(str)
    obs["assay_group"] = np.where(obs["assay"].astype(str).str.contains("10x", case=False),
                                  "10x", "other")
    onto = ontology
    if onto is None:
        try:
            from .ontology import load_cell_ontology
            onto = load_cell_ontology()
        except Exception as exc:  # noqa: BLE001 - no OBO configured
            warnings.warn(f"no Cell Ontology ({exc}); using raw Census cell_type labels", stacklevel=2)
            onto = None
    if onto is not None:
        from .ontology import add_level_labels
        add_level_labels(obs, onto, levels=(celltype_level,))
        plan["ontology"] = getattr(onto, "source", "")
    else:
        obs[celltype_col] = obs["cell_type"].astype(str)
        plan["ontology"] = None
    pb = pseudobulk_log2cp10k(counts, obs["raw_sum"].to_numpy(), obs, donor_col="donor_id",
                              celltype_col=celltype_col, min_cells=min_cells,
                              carry_cols=("condition", "assay", "assay_group"),
                              gene=[genes[i] for i in range(len(order))] if len(order) > 1 else genes[0],
                              keys=plan["pseudobulk_keys"], strict_carry=False)
    pb.attrs["plan"] = plan
    return pb
