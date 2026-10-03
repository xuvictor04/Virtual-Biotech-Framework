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


# ------------------------------------------------------------------ P7 additions
def test_cross_disease_pseudobulk_keeps_states_and_tissues_split():
    # one donor contributes tumour and adjacent-normal cells, in two tissues
    obs = pd.DataFrame({
        "dataset_id": "ds1", "donor_id": "d1", "cell_type": "fibroblast",
        "disease": ["LUAD"] * 30 + ["normal"] * 30 + ["normal"] * 30,
        "tissue_general": ["lung"] * 60 + ["blood"] * 30,
        "condition": ["disease"] * 30 + ["normal"] * 60,
    })
    counts = np.r_[np.full(30, 4.0), np.zeros(30), np.full(30, 1.0)]
    pb = cross_disease.pseudobulk_log2cp10k(counts, np.full(90, 1e4), obs, min_cells=25)
    assert len(pb) == 3
    by = pb.set_index(["disease", "tissue_general"])
    assert by.loc[("LUAD", "lung"), "mean_log2_OSMR_cp10k"] == pytest.approx(np.log2(5))
    assert by.loc[("normal", "lung"), "mean_log2_OSMR_cp10k"] == 0
    # carried columns must be constant within a group
    bad = obs.assign(assay=["10x"] * 15 + ["smart"] * 75)
    with pytest.raises(ValueError, match="assay"):
        cross_disease.pseudobulk_log2cp10k(counts, np.full(90, 1e4), bad, min_cells=25)
    # multi-gene long table with a gene column
    multi = cross_disease.pseudobulk_log2cp10k(np.column_stack([counts, counts * 2]), np.full(90, 1e4), obs,
                                               gene=["OSMR", "IL6ST"], min_cells=25)
    assert set(multi.gene) == {"OSMR", "IL6ST"} and len(multi) == 6 and "mean_log2_cp10k" in multi


def test_pair_with_normals_priority_and_disease_qc():
    rows = []
    for ds, assay in [("dsA", "10x 3' v3"), ("dsB", "10x 3' v3"), ("dsC", "Smart-seq2")]:
        for donor in range(4):
            rows.append({"disease_name": "normal", "dataset_id": ds, "assay": assay, "tissue_general": "colon",
                         "cell_type": "fib", "donor_id": f"{ds}_n{donor}"})
    for donor in range(4):
        rows.append({"disease_name": "UC", "dataset_id": "dsA", "assay": "10x 3' v3", "tissue_general": "colon",
                     "cell_type": "fib", "donor_id": f"uc{donor}"})
        rows.append({"disease_name": "CD", "dataset_id": "dsX", "assay": "10x 3' v3", "tissue_general": "colon",
                     "cell_type": "fib", "donor_id": f"cd{donor}"})
    rows.append({"disease_name": "rare", "dataset_id": "dsY", "assay": "other", "tissue_general": "colon",
                 "cell_type": "fib", "donor_id": "r0"})
    df = pd.DataFrame(rows)
    paired = cross_disease.pair_with_normals(df)
    uc = paired[(paired.disease_name == "UC") & (paired.condition == "normal")]
    assert set(uc.dataset_id) == {"dsA"} and set(uc.normal_match) == {"dataset_id"}  # same study first
    cd = paired[(paired.disease_name == "CD") & (paired.condition == "normal")]
    assert set(cd.dataset_id) == {"dsA", "dsB"} and set(cd.normal_match) == {"assay"}  # then same assay
    rare = paired[(paired.disease_name == "rare") & (paired.condition == "normal")]
    assert set(rare.dataset_id) == {"dsA", "dsB", "dsC"} and set(rare.normal_match) == {"tissue"}
    kept, report = cross_disease.disease_qc(paired, min_donors_per_arm=3)
    assert report.attrs["n_passing"] == 2 and set(kept.disease_name) == {"UC", "CD"}
    assert not report.set_index("disease_name").loc["rare", "passed"]


def test_cross_disease_engines_and_global_fdr():
    df = _cross_df()
    approx = cross_disease.disease_vs_normal_lmm(df, engine="mixedlm_sqrtw").set_index("disease_name")
    assert approx.loc["UC", "method"] == "mixedlm_sqrt_weight_APPROX" and approx.loc["CD", "method"] == "wls"
    assert approx.loc["UC", "beta"] > 0.6 and (approx["engine"] == "mixedlm_sqrtw").all()
    two = pd.concat([df.assign(gene="OSMR"), df.assign(gene="IL6ST")], ignore_index=True)
    res = cross_disease.disease_vs_normal_lmm(two, engine="python")
    assert len(res) == 6 and "gene" in res.columns
    one = cross_disease.disease_vs_normal_lmm(df, engine="python")
    assert (res.groupby("gene").fdr.min() >= one.fdr.min() - 1e-12).all()  # family of 6, not 3
    auto = cross_disease.disease_vs_normal_lmm(df)  # no lmerTest -> python, recorded
    assert auto["engine"].iloc[0] in {"python", "lmerTest"}
    plan = cross_disease.census_query_plan(["OSMR", "IL6ST"])
    assert plan["var_value_filter"] == "feature_name in ['OSMR', 'IL6ST']"
    assert plan["celltype_col"] == "cell_type_level1" and "cell_type_ontology_term_id" in plan["obs_column_names"]
    assert plan["pseudobulk_keys"][:2] == ["dataset_id", "donor_id"]


def _surv_df(seed=0, n=600):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"expr": rng.normal(size=n), "age": rng.normal(65, 8, n),
                       "stage_advanced": rng.integers(0, 2, n).astype(float),
                       "sex": rng.integers(0, 2, n).astype(float)})
    for ep, b in [("os", 0.6), ("dfs", 0.8), ("pfs", 0.5), ("dss", 0.4)]:
        t = rng.exponential(1 / (0.05 * np.exp(b * df.expr)))
        c = rng.exponential(40, n)
        df[f"{ep}_time"], df[f"{ep}_event"] = np.minimum(t, c), (t <= c).astype(float)
    df.loc[: n // 3, ["dfs_time", "dfs_event"]] = np.nan  # DFS missing for many, low-expression-biased
    df.loc[:9, "age"] = np.nan
    return df


def test_quartile_cox_all_endpoints_fixed_groups():
    df = _surv_df()
    res = survival.quartile_cox_all_endpoints(df, "expr", engine="statsmodels").set_index("endpoint")
    assert list(res.index) == ["OS", "PFS", "DSS", "DFS"]
    assert (res["n_cohort"] == 590).all() and (res["quartile_scope"] == "cohort").all()
    assert res["q_high"].nunique() == 1 and res["q_low"].nunique() == 1  # fixed stratification
    assert res.loc["OS", "n"] == res.loc["OS", "n_high"] + res.loc["OS", "n_low"]
    assert res.loc["OS", "n"] > res.loc["DFS", "n"]  # DFS fitted on the cohort's groups with DFS data
    assert res.loc["OS", "hr"] > 1.5 and res.loc["DFS", "hr"] > 1.5
    per_ep = survival.quartile_cox_all_endpoints(df, "expr", engine="statsmodels",
                                                 quartile_scope="endpoint").set_index("endpoint")
    assert per_ep.loc["DFS", "q_low"] != res.loc["DFS", "q_low"]
    with pytest.raises(ValueError):
        survival.quartile_cox_all_endpoints(df, "expr", quartile_scope="x")


def test_tf_lmg_helpers():
    import inspect

    assert inspect.signature(tf_activity.score_tf_activity).parameters["exclusions"].default is None
    net = pd.DataFrame({"source": ["STAT1"] * 4 + ["STAT3"], "target": ["IL6ST", "OSMR", "IRF1", "LIFR", "OSMR"],
                        "weight": 1.0})
    clean = tf_activity.clean_regulon(net)
    assert list(clean[clean.source == "STAT1"].target) == ["IRF1"] and len(clean) == 2

    rng = np.random.default_rng(0)
    rows, X = [], []
    for patient in range(12):
        for ct in ["Fib", "Endo", "Epi"]:
            for _ in range(15):
                rows.append({"sample": f"S{patient}", "patient": f"P{patient}", "cell_type": ct})
                osmr = rng.random() < (0.5 if ct != "Epi" else 0.01)
                X.append([float(osmr) * rng.gamma(2), rng.gamma(2), rng.gamma(1)])

    class A:
        pass

    a = A()
    a.obs = pd.DataFrame(rows, index=[f"c{i}" for i in range(len(rows))])
    a.X = np.array(X)
    a.var_names = pd.Index(["OSMR", "IL6ST", "LIFR"])
    keep, table = tf_activity.select_celltypes_by_detection(a, "OSMR", 0.05, return_table=True)
    assert keep == ["Endo", "Fib"] and table.loc["Epi", "frac_detected"] < 0.05
    act = pd.DataFrame({"STAT1": 0.8 * a.X[:, 1] + rng.normal(0, 0.3, len(rows))}, index=a.obs.index)
    tab = tf_activity.pseudobulk_means(a, act, ["OSMR", "IL6ST", "LIFR"], by=("sample", "cell_type"),
                                       min_cells=10, carry_cols=("patient",))
    assert len(tab) == 36 and (tab.n_cells == 15).all() and {"STAT1", "OSMR", "patient"} <= set(tab.columns)
    small = tf_activity.pseudobulk_means(a, act, ["OSMR"], min_cells=16)
    assert small.empty
    res = tf_activity.run_lmg_per_celltype(tab[tab.cell_type.isin(keep)], "STAT1", ["OSMR", "IL6ST", "LIFR"],
                                           cluster_col="patient", n_boot=50)
    fib = res[res.cell_type == "Fib"].set_index("predictor")
    assert set(res.cell_type) == {"Endo", "Fib"}
    assert fib.loc["IL6ST", "rank"] == 1 and fib["share"].sum() == pytest.approx(fib["r2_total"].iloc[0])
    assert (fib["ci_low"] <= fib["share"]).all()


def test_lmg_engine_recorded():
    df = _lmg_data()
    out = vd.lmg_shares(df, "y", ["OSMR", "IL6ST"], groups="patient")
    assert out.attrs["r2_engine"] == "statsmodels_mixedlm_np_var" and set(out.engine) == {out.attrs["r2_engine"]}
    assert set(vd.lmg_shares(df, "y", ["OSMR"]).engine) == {"ols"}
    auto = vd.lmg_shares(df, "y", ["OSMR", "IL6ST"], groups="patient", r2_engine="auto")
    assert auto.attrs["r2_engine"] in {"statsmodels_mixedlm_np_var", "R:MuMIn"}
