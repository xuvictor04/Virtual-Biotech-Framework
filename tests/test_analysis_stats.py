import gzip

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from vbt.analysis import biomarker, cross_disease, survival, tf_activity
from vbt.analysis import variance_decomposition as vd
from vbt.analysis._utils import bh_fdr


def test_bh_fdr_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests

    p = np.array([0.001, 0.02, 0.03, 0.5, np.nan, 0.04])
    ok = ~np.isnan(p)
    np.testing.assert_allclose(bh_fdr(p)[ok], multipletests(p[ok], method="fdr_bh")[1])
    assert np.isnan(bh_fdr(p)[4])


# ------------------------------------------------------------------ survival
def test_stage_and_status_mapping():
    f = survival.stage_to_advanced
    assert f("STAGE IIIA") == 1 and f("Stage IV") == 1 and f("STAGE IB") == 0 and f("STAGE II") == 0
    assert np.isnan(f("STAGE X")) and np.isnan(f(None)) and np.isnan(f("[Not Available]"))
    clin = pd.DataFrame({
        "AJCC_PATHOLOGIC_TUMOR_STAGE": ["STAGE IIB", "STAGE IIIB", None],
        "SEX": ["Male", "Female", "Male"], "AGE": ["65", "70", "n/a"],
        "OS_MONTHS": [10.0, 20.0, 5.0], "OS_STATUS": ["1:DECEASED", "0:LIVING", "1:DECEASED"],
        "DFS_MONTHS": [3, 4, 5], "DFS_STATUS": ["1:Recurred/Progressed", "0:DiseaseFree", None],
        "PFS_MONTHS": [1, 2, 3], "PFS_STATUS": ["1:PROGRESSION", "0:CENSORED", "0:CENSORED"],
        "DSS_MONTHS": [1, 2, 3], "DSS_STATUS": ["1:DEAD WITH TUMOR", "0:ALIVE OR DEAD TUMOR FREE", None],
    })
    out = survival.prepare_tcga_clinical(clin)
    assert out["stage_advanced"].tolist()[:2] == [0.0, 1.0] and np.isnan(out["stage_advanced"][2])
    assert out["sex"].tolist() == [1.0, 0.0, 1.0]
    assert out["os_event"].tolist() == [1.0, 0.0, 1.0]
    assert out["dfs_event"].tolist()[:2] == [1.0, 0.0] and np.isnan(out["dfs_event"][2])
    assert out["pfs_event"].tolist() == [1.0, 0.0, 0.0]
    assert out["dss_event"].tolist()[:2] == [1.0, 0.0]
    assert np.isnan(out["age"][2])


def test_quartile_cox_recovers_hr_gt_1():
    rng = np.random.default_rng(0)
    n = 800
    df = pd.DataFrame({"expr": rng.normal(size=n), "age": rng.normal(65, 8, n),
                       "stage_advanced": rng.integers(0, 2, n).astype(float),
                       "sex": rng.integers(0, 2, n).astype(float)})
    lin = 0.6 * df.expr + 0.5 * df.stage_advanced + 0.02 * (df.age - 65)
    t = rng.exponential(1 / (0.05 * np.exp(lin)))
    c = rng.exponential(40, n)
    df["time"], df["event"] = np.minimum(t, c), (t <= c).astype(float)
    df.loc[:9, "age"] = np.nan  # incomplete covariates are dropped
    res = survival.quartile_cox(df, "expr", "time", "event", engine="statsmodels")
    assert res["hr"] > 1.5 and res["p"] < 1e-4 and res["ci_low"] > 1
    assert res["n"] == res["n_high"] + res["n_low"]
    assert abs(res["n_high"] - (n - 10) / 4) <= 2
    assert res["engine"] == "statsmodels"


# ---------------------------------------------------------------- TF activity
def test_remove_targets_from_regulon():
    net = pd.DataFrame({"source": ["STAT1", "STAT1", "STAT3", "STAT3"],
                        "target": ["IL6ST", "IRF1", "IL6ST", "SOCS3"], "weight": 1.0})
    out = tf_activity.remove_targets_from_regulon(net, tf_activity.PAPER_REGULON_EXCLUSIONS)
    assert len(out) == 3 and not ((out.source == "STAT1") & (out.target == "IL6ST")).any()
    assert ((out.source == "STAT3") & (out.target == "IL6ST")).any()
    assert len(tf_activity.remove_targets_from_regulon(net, ["IL6ST"])) == 2
    assert len(tf_activity.remove_targets_from_regulon(net, ["IL6ST"], tfs=["STAT3"])) == 3


def test_tf_mixed_model_screen():
    rng = np.random.default_rng(1)
    rows, acts = [], []
    for ct in ["Fib", "Epi"]:
        for s in range(8):
            cond = "UC" if s >= 4 else "healthy"
            sample_eff = rng.normal(0, 0.3)
            for _ in range(30):
                rows.append((ct, f"S{s}", cond))
                stat3 = 1.5 * (cond == "UC" and ct == "Fib") + sample_eff + rng.normal(0, 0.5)
                acts.append((stat3, rng.normal(0, 0.5) + sample_eff, rng.normal()))
    obs = pd.DataFrame(rows, columns=["ct", "sample", "cond"])
    act = pd.DataFrame(acts, columns=["STAT3", "STAT1", "NFKB1"], index=obs.index)
    res = tf_activity.tf_mixed_model_screen(act, obs, "cond", "sample", "ct", ref_level="healthy",
                                            engine="statsmodels")
    assert len(res) == 6
    r = res.set_index(["cell_type", "tf"])
    assert r.loc[("Fib", "STAT3"), "beta"] > 1 and r.loc[("Fib", "STAT3"), "fdr"] < 0.01
    assert r.loc[("Epi", "STAT3"), "fdr"] > 0.01
    assert r.loc[("Fib", "STAT3"), "n_samples_ref"] == 4


# ------------------------------------------------------------------------ LMG
def _lmg_data(seed=0, n=300):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    X[:, 1] += 0.5 * X[:, 0]
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] + 0.2 * X[:, 2] + rng.normal(size=n)
    df = pd.DataFrame(X, columns=["OSMR", "IL6ST", "LIFR", "IL6R"])
    df["y"] = y
    df["patient"] = np.repeat(np.arange(30), n // 30)
    return df


def _ols_r2(df, cols):
    return sm.OLS(df["y"], sm.add_constant(df[cols])).fit().rsquared


def test_lmg_shares_sum_and_rank():
    df = _lmg_data()
    preds = ["OSMR", "IL6ST", "LIFR", "IL6R"]
    out = vd.lmg_shares(df, "y", preds)
    assert out.share.sum() == pytest.approx(_ols_r2(df, preds), abs=1e-10)
    assert out.share.sum() == pytest.approx(out.attrs["r2_total"])
    assert out.predictor.iloc[0] == "OSMR" and out["rank"].iloc[0] == 1
    assert out.pct_of_r2.sum() == pytest.approx(100)


def test_lmg_two_predictor_hand_computed():
    df = _lmg_data(seed=2)
    r1, r2, r12 = _ols_r2(df, ["OSMR"]), _ols_r2(df, ["IL6ST"]), _ols_r2(df, ["OSMR", "IL6ST"])
    out = vd.lmg_shares(df, "y", ["OSMR", "IL6ST"]).set_index("predictor")
    assert out.loc["OSMR", "share"] == pytest.approx((r1 + (r12 - r2)) / 2, abs=1e-10)
    assert out.loc["IL6ST", "share"] == pytest.approx((r2 + (r12 - r1)) / 2, abs=1e-10)


def test_lmg_mixed_and_bootstrap():
    df = _lmg_data()
    preds = ["OSMR", "IL6ST", "LIFR"]
    mixed = vd.lmg_shares(df, "y", preds, groups="patient")
    assert mixed.share.sum() == pytest.approx(mixed.attrs["r2_total"], abs=1e-10)
    assert mixed.predictor.iloc[0] == "OSMR" and 0 < mixed.attrs["r2_total"] < 1
    boot = vd.bootstrap_lmg(df, "y", preds, cluster_col="patient", n_boot=100, seed=0)
    top = boot.iloc[0]
    assert top.predictor == "OSMR" and top.ci_low < top.share < top.ci_high
    assert top.p_top > 0.9 and boot.attrs["n_clusters"] == 30
    assert vd.GP130_RECEPTORS[0] == "OSMR" and len(vd.GP130_RECEPTORS) == 7


# ------------------------------------------------------------------ biomarker
def test_collapse_probes_and_composite_score():
    expr = pd.DataFrame({"s1": [1.0, 3.0, 5.0, 9.0], "s2": [2.0, 4.0, 6.0, 9.0]},
                        index=["p1", "p2", "p3", "p4"])
    genes = biomarker.collapse_probes(expr, {"p1": "OSMR", "p2": "OSMR", "p3": "A /// B", "p4": "---"})
    assert list(genes.index) == ["OSMR"] and genes.loc["OSMR"].tolist() == [2.0, 3.0]

    m = pd.DataFrame({"OSMR": [1.0, 2.0, 3.0], "IL6": [2.0, 2.0, 2.0], "LIFR": [3.0, 2.0, 1.0]})
    sc = biomarker.composite_score(m, ["OSMR", "LIFR", "IL6", "NOTHERE"])
    np.testing.assert_allclose(sc.to_numpy(), [0.0, 0.0, 0.0], atol=1e-12)
    assert sc.attrs["genes_used"] == ["OSMR", "LIFR"]
    assert set(sc.attrs["genes_missing"]) == {"IL6", "NOTHERE"}


def test_score_auc_delong_and_bootstrap():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(40), np.zeros(60)]
    s = rng.normal(size=100) + 1.2 * y
    r = biomarker.score_auc(s, y)
    assert 0.7 < r["auc"] < 0.95 and r["ci_low"] < r["auc"] < r["ci_high"]
    assert r["n_nonresponders"] == 40
    rb = biomarker.score_auc(s, y, ci="bootstrap", n_boot=200)
    assert rb["auc"] == pytest.approx(r["auc"]) and abs(rb["ci_low"] - r["ci_low"]) < 0.08
    # perfect separation
    assert biomarker.score_auc([1, 2, 3, 4], [0, 0, 1, 1])["auc"] == 1.0


def test_compare_scores_across_cohorts():
    rng = np.random.default_rng(5)
    cohorts = {}
    genes = sorted(set(biomarker.GP130_AXIS_GENES + biomarker.ARIJS_SIGNATURE))
    for name in ["GSE12251", "GSE16879"]:
        lab = pd.Series(rng.integers(0, 2, 40), index=[f"{name}_{i}" for i in range(40)])
        expr = pd.DataFrame(rng.normal(size=(40, len(genes))), index=lab.index, columns=genes)
        expr[biomarker.GP130_AXIS_GENES] += lab.to_numpy()[:, None] * 1.0
        expr = expr.drop(columns=["CNTF"], errors="ignore").drop(columns=["STC1"])
        cohorts[name] = (expr, lab)
    tab = biomarker.compare_scores_across_cohorts(cohorts)
    assert tab.shape == (6, 11)
    assert set(tab.signature) == {"gp130_axis", "OSMR_alone", "arijs_5gene"}
    g = tab.set_index(["cohort", "signature"])
    assert g.loc[("GSE12251", "gp130_axis"), "auc"] > 0.85
    assert g.loc[("GSE12251", "arijs_5gene"), "n_genes_used"] == 4
    assert "STC1" in g.loc[("GSE12251", "arijs_5gene"), "genes_missing"]


SERIES_MATRIX = """!Series_title\t"Toy UC infliximab cohort"
!Series_geo_accession\t"GSE00001"
!Sample_title\t"UC1 pre"\t"UC2 pre"\t"UC3 pre"
!Sample_geo_accession\t"GSM1"\t"GSM2"\t"GSM3"
!Sample_source_name_ch1\t"colon"\t"colon"\t"colon"
!Sample_characteristics_ch1\t"disease: UC"\t"disease: UC"\t"disease: UC"
!Sample_characteristics_ch1\t"response: Yes"\t"response: No"\t"response: No"
!Sample_characteristics_ch1\t"time: W0"\t"time: W0"\t"time: W8"
!series_matrix_table_begin
"ID_REF"\t"GSM1"\t"GSM2"\t"GSM3"
"1000_at"\t7.5\t8.1\tnull
"1001_at"\t5.0\t5.5\t6.0
!series_matrix_table_end
"""


@pytest.mark.parametrize("gz", [False, True])
def test_load_geo_series_matrix(tmp_path, gz):
    path = tmp_path / ("GSE00001_series_matrix.txt" + (".gz" if gz else ""))
    data = SERIES_MATRIX.encode()
    path.write_bytes(gzip.compress(data) if gz else data)
    gsm = biomarker.load_geo_series_matrix(path)
    assert gsm.expression.shape == (2, 3)
    assert list(gsm.expression.columns) == ["GSM1", "GSM2", "GSM3"]
    assert np.isnan(gsm.expression.loc["1000_at", "GSM3"])
    assert gsm.expression.loc["1001_at", "GSM2"] == 5.5
    s = gsm.samples
    assert list(s.index) == ["GSM1", "GSM2", "GSM3"]
    assert s["response"].tolist() == ["Yes", "No", "No"]
    assert s.loc["GSM3", "time"] == "W8" and s.loc["GSM1", "title"] == "UC1 pre"
    assert gsm.series["geo_accession"] == "GSE00001"
    assert set(biomarker.GEO_COHORTS) == {"GSE12251", "GSE16879", "GSE23597", "GSE73661"}


# -------------------------------------------------------------- cross-disease
def test_pseudobulk_log2cp10k():
    obs = pd.DataFrame({"donor_id": ["d1"] * 30 + ["d2"] * 10, "cell_type": "Fib",
                        "dataset_id": "ds1", "condition": "normal"})
    counts = np.r_[np.full(30, 1.0), np.zeros(10)]
    total = np.full(40, 1e4)
    pb = cross_disease.pseudobulk_log2cp10k(counts, total, obs, min_cells=25)
    assert len(pb) == 1 and pb.n_cells.iloc[0] == 30
    assert pb["mean_log2_OSMR_cp10k"].iloc[0] == pytest.approx(1.0)
    assert pb["pseudobulk_log2_OSMR_cp10k"].iloc[0] == pytest.approx(1.0)
    assert pb["log_depth"].iloc[0] == pytest.approx(4.0) and pb.dataset_id.iloc[0] == "ds1"


def _cross_df(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for disease, eff, nds in [("UC", 1.0, 3), ("psoriasis", 0.0, 3), ("CD", 0.8, 1)]:
        for ds in range(nds):
            ds_eff = rng.normal(0, 0.2)
            for i in range(12):
                cond = "disease" if i % 2 else "normal"
                n_cells = int(rng.integers(25, 400))
                depth = rng.normal(3.5, 0.2)
                val = 1 + eff * (cond == "disease") + ds_eff + 0.3 * (depth - 3.5) + rng.normal(0, 0.3)
                rows.append({"disease_name": disease, "cell_type": "Fib", "condition": cond,
                             "dataset_id": f"{disease}_{ds}", "n_cells": n_cells,
                             "log_depth": depth, "assay_group": "10x",
                             "mean_log2_OSMR_cp10k": val})
    return pd.DataFrame(rows)


def test_disease_vs_normal_weighted_model():
    df = _cross_df()
    res = cross_disease.disease_vs_normal_lmm(df, engine="python").set_index("disease_name")
    assert len(res) == 3
    assert res.loc["UC", "method"] == "wls_cluster_robust" and res.loc["CD", "method"] == "wls"
    assert res.loc["UC", "beta"] > 0.6 and res.loc["UC", "fdr"] < 0.05
    assert res.loc["CD", "beta"] > 0.4 and res.loc["CD", "fdr"] < 0.05
    assert res.loc["psoriasis", "fdr"] > 0.05
    assert res.loc["UC", "n_datasets"] == 3


def test_pair_with_normals_and_query_plan():
    df = pd.DataFrame({"disease_name": ["UC", "normal", "normal", "psoriasis"],
                       "tissue_general": ["colon", "colon", "skin", "skin"], "v": [1, 2, 3, 4]})
    paired = cross_disease.pair_with_normals(df)
    assert len(paired) == 4
    assert paired.groupby("disease_name").condition.apply(set).to_dict() == {
        "UC": {"disease", "normal"}, "psoriasis": {"disease", "normal"}}
    plan = cross_disease.census_query_plan("OSMR", diseases=["ulcerative colitis"])
    assert plan["var_value_filter"] == "feature_name == 'OSMR'"
    assert "normal" in plan["obs_value_filter"] and "is_primary_data" in plan["obs_value_filter"]


def test_optional_dependency_error_message():
    from vbt.analysis._utils import require

    with pytest.raises(ImportError, match=r"pip install 'vbt-harness\[singlecell\]'"):
        require("liana_not_a_real_module_xyz", "singlecell")
