"""Fidelity of vbt.analysis against the authors' Case 2/3 code (Zenodo archive).

Unit tests run everywhere; numeric checks against the archived outputs skip
when the archive extract is absent (set VBT_ZENODO_DIR or run
`vbt data zenodo fetch --preset b7h3-results` / `--preset osmr-results`).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from vbt.analysis import biomarker, cross_disease, singlecell, survival
from vbt.analysis import variance_decomposition as vd


def _root():
    from vbt.data.zenodo import zenodo_root

    return zenodo_root()


def _need(rel: str):
    p = _root() / rel
    if not p.exists():
        pytest.skip(f"archive file not present: {rel}")
    return p


# ---------------------------------------------------------------------------
# unit tests (synthetic)
# ---------------------------------------------------------------------------


def test_survival_authors_qc():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"expr": rng.normal(size=n), "age": rng.normal(60, 8, n),
                       "stage_advanced": rng.integers(0, 2, n).astype(float), "sex": rng.integers(0, 2, n).astype(float),
                       "os_time": 1 + rng.exponential(30, n), "os_event": rng.integers(0, 2, n).astype(float),
                       "msi_sensor": np.r_[np.full(10, 5.0), np.full(n - 10, 0.1)]})
    df.loc[:19, "os_time"] = 0.5  # < 1 month -> endpoint dropped -> patient dropped
    r = survival.quartile_cox_all_endpoints(df, "expr", endpoints=("OS",), engine="statsmodels")
    assert r.loc[0, "n_cohort"] == n - 20  # 20 short follow-up (incl. the 10 MSI-H) removed
    plain = survival.quartile_cox_all_endpoints(df, "expr", endpoints=("OS",), engine="statsmodels",
                                                min_followup=None, msi_high=None)
    assert plain.loc[0, "n_cohort"] == n
    df.loc[:19, "os_time"] = 5.0
    msi = survival.quartile_cox_all_endpoints(df, "expr", endpoints=("OS",), engine="statsmodels")
    assert msi.loc[0, "n_cohort"] == n - 10


def test_filter_lr_authors_rule_and_specificity():
    n = 100
    df = pd.DataFrame({"source": "F", "target": "T", "ligand_complex": [f"L{i}" for i in range(n)],
                       "receptor_complex": "R", "magnitude_rank": np.linspace(0.001, 1, n),
                       "specificity_rank": np.r_[np.full(5, 0.9), np.linspace(0.001, 1, n - 5)],
                       "cellphone_pvals": 0.001, "lrscore": 0.5, "lr_logfc": 1.0, "expr_prod": 1.0,
                       "scaled_weight": 0.0})
    out = singlecell.filter_lr_results(df)  # authors: 4 methods detected; specificity top 10% required
    assert set(out.ligand_complex) == {f"L{i}" for i in range(5, 10)}
    assert (out.n_methods == 4).all()
    no_spec = singlecell.filter_lr_results(df, specificity_top_frac=None)
    assert len(no_spec) == 10
    df2 = df.assign(lr_logfc=0.0, expr_prod=0.0)  # only 2 methods detect
    assert singlecell.filter_lr_results(df2).empty


def test_assay_design():
    meta = pd.DataFrame({"condition": ["d"] * 6 + ["n"] * 6,
                         "assay": ["v3"] * 4 + ["v2"] * 2 + ["v3"] * 3 + ["v2"] * 2 + ["v1"]},
                        index=[f"s{i}" for i in range(12)])
    m, cov, diag = singlecell.assay_design(meta)
    assert cov == ["assay_grp"] and diag["assay_included"] and "s11" not in m.index  # v1 (n=1) -> other, dropped
    assert diag["design"] == "~condition + assay_grp"
    conf = pd.DataFrame({"condition": ["d"] * 4 + ["n"] * 4, "assay": ["a"] * 4 + ["b"] * 4})
    m2, cov2, diag2 = singlecell.assay_design(conf)
    assert cov2 == [] and diag2["fallback_reason"] == "assay_perfectly_predicts_condition" and len(m2) == 8


def test_pseudobulk_majority_column():
    obs = pd.DataFrame({"donor": ["a"] * 60, "celltype": ["x"] * 60, "assay": ["v3"] * 40 + ["v2"] * 20})
    counts, meta = singlecell.pseudobulk_counts(np.ones((60, 2)), obs, majority_cols=["assay"])
    assert meta["assay"].iloc[0] == "v3" and meta["n_cells"].iloc[0] == 60


def test_lmm_ml_matches_statsmodels():
    import statsmodels.api as sm

    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(8), 6)
    x = rng.normal(size=48)
    y = 0.5 * x + rng.normal(0, 0.7, 8)[g] + rng.normal(0, 0.5, 48)
    X = sm.add_constant(x)
    fit = vd.lmm_random_intercept_ml(y, X, g)
    ref = sm.MixedLM(y, X, groups=g).fit(reml=False)
    np.testing.assert_allclose(fit["beta"], ref.fe_params, rtol=1e-4)
    assert fit["var_group"] == pytest.approx(float(np.asarray(ref.cov_re).ravel()[0]), rel=1e-3)
    assert fit["var_resid"] == pytest.approx(ref.scale, rel=1e-3)
    # singular design (no group effect) is handled at the boundary
    y2 = 0.5 * x + rng.normal(0, 0.5, 48)
    f2 = vd.lmm_random_intercept_ml(y2 - (np.bincount(g, y2)[g] / 6 - y2.mean()), X, g)
    assert f2["var_group"] >= 0


def test_weighted_reml_reduces_to_wls_and_df():
    import statsmodels.api as sm

    rng = np.random.default_rng(2)
    n = 60
    g = np.repeat(np.arange(6), 10)
    w = rng.integers(20, 500, n).astype(float)
    x = rng.integers(0, 2, n).astype(float)
    y = 0.3 * x + rng.normal(0, 1, n) / np.sqrt(w)
    X = sm.add_constant(x)
    r = cross_disease.weighted_lmm_reml(y, X, g, w)
    wls = sm.WLS(y, X, weights=w).fit()
    if r["singular"]:  # no study variance -> WLS with n - p df
        assert r["beta"] == pytest.approx(float(wls.params[1]), rel=1e-6)
        assert r["df"] == n - 2
    y3 = y + rng.normal(0, 0.5, 6)[g]
    r3 = cross_disease.weighted_lmm_reml(y3, X, g, w)
    assert r3["var_group"] > 0 and 1 < r3["df"] < n and 0 <= r3["p"] <= 1


def test_biomarker_orientation():
    rng = np.random.default_rng(3)
    lab = pd.Series(rng.integers(0, 2, 60))
    expr = pd.DataFrame({"OSMR": -lab.to_numpy() + rng.normal(0, 0.5, 60)})
    t = biomarker.compare_scores_across_cohorts({"c": (expr, lab)}, {"OSMR_alone": ["OSMR"]})
    assert t.loc[0, "auc"] > 0.5 and bool(t.loc[0, "flipped"])
    f = biomarker.compare_scores_across_cohorts({"c": (expr, lab)}, {"OSMR_alone": ["OSMR"]}, orientation="fixed")
    assert f.loc[0, "auc"] < 0.5 and not bool(f.loc[0, "flipped"])


# ---------------------------------------------------------------------------
# archive checks
# ---------------------------------------------------------------------------


def test_survival_reproduces_archived_hrs():
    raw = _need("b7-h3/data/inputs/tcga/raw_data.csv")
    res_dir = _root() / "b7-h3/data/survival_outputs/results"
    df = survival.prepare_tcga_clinical(survival.load_cbioportal_raw(raw))
    out = survival.quartile_cox_all_endpoints(df, "expr", engine="statsmodels").set_index("endpoint")
    strat = json.loads((res_dir / "stratification_info.json").read_text())
    for ep in ("OS", "PFS", "DSS", "DFS"):
        ref = json.loads((res_dir / f"{ep.lower()}_analysis_results.json").read_text())
        mv = ref["multivariable_cox"]
        assert out.loc[ep, "n"] == ref["n_samples"] and out.loc[ep, "n_events"] == ref["n_events"]
        assert out.loc[ep, "hr"] == pytest.approx(mv["b7h3_hazard_ratio"], rel=1e-5)
        assert out.loc[ep, "p"] == pytest.approx(mv["b7h3_p_value"], rel=1e-4)
        assert out.loc[ep, "ci_low"] == pytest.approx(mv["b7h3_ci_lower"], rel=1e-4)
        assert out.loc[ep, "q_low"] == pytest.approx(strat["q25_threshold"])
        assert out.loc[ep, "q_high"] == pytest.approx(strat["q75_threshold"])
    assert out.loc["OS", "hr"] == pytest.approx(1.62, abs=0.005)  # paper
    assert out.loc["DFS", "hr"] == pytest.approx(2.06, abs=0.01)


AUTHORS_AUC = {  # osmr/traces/agent_reports/single_cell_analyst_validation_report.md
    "GSE16879": (0.773, 0.914, 0.914), "GSE12251": (0.788, 0.909, 0.962),
    "GSE23597": (0.651, 0.834, 0.846), "GSE73661": (0.742, 0.850, 0.858),
}


def test_biomarker_aucs_reproduce_authors():
    pytest.importorskip("anndata")
    d = _need("osmr/code/data/GSE12251.h5ad").parent
    cohorts = {k: biomarker.load_processed_cohort(d / f"{k}.h5ad", drug=v[0], response_col=v[1], disease=v[2])
               for k, v in biomarker.AUTHORS_IFX_COHORTS.items()}
    t = biomarker.compare_scores_across_cohorts(cohorts, ci="none").set_index(["cohort", "signature"])
    for c, (osmr, gp130, arijs) in AUTHORS_AUC.items():
        assert t.loc[(c, "OSMR_alone"), "auc"] == pytest.approx(osmr, abs=6e-4)
        assert t.loc[(c, "gp130_axis"), "auc"] == pytest.approx(gp130, abs=6e-4)
        assert t.loc[(c, "arijs_5gene"), "auc"] == pytest.approx(arijs, abs=6e-4)


IMMUNE = ["T cell", "B cell", "natural killer cell", "monocyte", "macrophage", "dendritic cell", "neutrophil"]


@pytest.mark.parametrize("cancer,n_specific,n_fib_immune", [("luad", 793, 226), ("sclc", 719, 180)])
def test_liana_specific_interaction_counts(cancer, n_specific, n_fib_immune):
    d = _need(f"b7-h3/data/single_cell_outputs/liana_{cancer}_normalized/liana_cd276_high_consensus.csv").parent
    hi_raw, lo_raw = d / "liana_cd276_high_raw.csv", d / "liana_cd276_low_raw.csv"
    if hi_raw.exists() and lo_raw.exists():  # full archive: re-filter the raw LIANA tables
        hi = singlecell.filter_lr_results(pd.read_csv(hi_raw))
        lo = singlecell.filter_lr_results(pd.read_csv(lo_raw))
        assert len(hi) == len(pd.read_csv(d / "liana_cd276_high_consensus.csv"))
    else:
        hi = pd.read_csv(d / "liana_cd276_high_consensus.csv")
        lo = pd.read_csv(d / "liana_cd276_low_consensus.csv")
    res = singlecell.group_specific_interactions(hi, lo)
    a = res["a_only"]
    assert len(a) == n_specific == len(pd.read_csv(d / "cd276_high_specific_interactions.csv"))
    assert int(((a.source == "fibroblast") & a.target.isin(IMMUNE)).sum()) == n_fib_immune


def test_lmg_shares_reproduce_authors():
    pb = pd.read_csv(_need("osmr/code/pseudobulk.tsv"), sep="\t")
    ref = pd.read_csv(_root() / "osmr/code/stat1_dominance_lmg_shares.tsv", sep="\t")
    cells = ["LP_fibroblast", "Pericyte", "Endothelium"]
    ours = vd.stat1_dominance(pb, cell_states=cells, n_boot=2000)
    m = ref.merge(ours, on=["cell_state", "receptor"], suffixes=("_a", "_o"))
    assert len(m) == len(ref[ref.cell_state.isin(cells)]) > 0
    np.testing.assert_allclose(m["lmg_share_o"], m["lmg_share_a"], atol=1e-5)
    np.testing.assert_allclose(m["ci_lo_o"], m["ci_lo_a"], atol=1e-6)
    np.testing.assert_allclose(m["ci_hi_o"], m["ci_hi_a"], atol=1e-6)
    np.testing.assert_allclose(m["r2m_full_o"], m["r2m_full_a"], atol=1e-5)


def test_cross_disease_lmm_reproduces_authors():
    d = _need("osmr/data/results/cross_disease_v3/tests_lmm_v2.csv").parent
    uc = d / "disease_ulcerative_colitis.csv"
    if not uc.exists():
        pytest.skip("cross-disease pseudobulk inputs not fetched")
    t = cross_disease.cross_disease_tests(cross_disease.prepare_cross_disease(pd.read_csv(uc)))
    ref = pd.read_csv(d / "tests_lmm_v2.csv")
    m = ref.merge(t, on=["disease_term", "cell_type", "gene"], suffixes=("_a", "_o"))
    assert len(m) == len(ref[ref.disease_term == "ulcerative colitis"]) > 0
    np.testing.assert_allclose(m["log2FC_o"], m["log2FC_a"], rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(m["se_o"], m["se_a"], rtol=1e-4, atol=1e-8)
    np.testing.assert_allclose(m["df_satterthwaite_o"], m["df_satterthwaite_a"], rtol=1e-3)
