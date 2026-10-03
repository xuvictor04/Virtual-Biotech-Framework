"""Tests for trial-outcome single-cell features and association statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy import stats as sstats

from vbt.case_studies.trial_outcomes import features as F
from vbt.case_studies.trial_outcomes import stats as S


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------


def test_tau_uniform_and_one_hot():
    assert F.tau_index(np.array([2.0, 2.0, 2.0, 2.0])) == pytest.approx(0.0)
    assert F.tau_index(np.array([0.0, 0.0, 3.0, 0.0])) == pytest.approx(1.0)
    assert np.isnan(F.tau_index(np.zeros(4)))
    assert np.isnan(F.tau_index(np.array([1.0])))


def test_bimodality_coefficient():
    rng = np.random.default_rng(0)
    bimodal = np.concatenate([rng.normal(1, 0.1, 500), rng.normal(4, 0.1, 500)])
    normal = rng.normal(5, 1, 2000)
    assert F.bimodality_coefficient(bimodal) > F.BC_THRESHOLD
    assert F.bimodality_coefficient(normal) < F.BC_THRESHOLD
    assert np.isnan(F.bimodality_coefficient(np.array([0, 0, 1.0, 2.0, 3.0])))  # n=3


class FakeAnnData:
    def __init__(self, X, obs, var):
        self.X = X
        self.obs = obs
        self.var = var
        self.var_names = var.index


def _fake_adata():
    rng = np.random.default_rng(1)
    obs_rows, rows = [], []
    # Tissue A: 3 cell types x 30 cells; tissue B: 2 types x 25 + 1 small type (5).
    layout = [("A", "t1", 30), ("A", "t2", 30), ("A", "t3", 30),
              ("B", "t1", 25), ("B", "t4", 25), ("B", "t5", 5)]
    for tissue, ct, n in layout:
        for _ in range(n):
            obs_rows.append((tissue, ct))
            g0 = 1.0  # uniform everywhere
            g1 = 2.0 if ct == "t1" else 0.0  # one-hot in both tissues
            g2 = 0.0  # never expressed
            g3 = rng.choice([0.5, 3.0]) + rng.normal(0, 0.05)  # bimodal
            rows.append([g0, g1, g2, g3])
    X = sp.csr_matrix(np.array(rows))
    obs = pd.DataFrame(obs_rows, columns=["tissue", "cell_ontology_class"])
    var = pd.DataFrame({"ensembl_id": ["ENSG01.3", "ENSG02", "ENSG03", "ENSG04"]},
                       index=["G0", "G1", "G2", "G3"])
    return FakeAnnData(X, obs, var), np.array(rows), obs


def test_compute_gene_features_duck_typed():
    adata, dense, obs = _fake_adata()
    res = F.compute_gene_features(adata, chunk_size=2)
    assert list(res.columns) == ["tau", "bimodality", "n_tissues_expressed"]
    assert res.loc["G0", "tau"] == pytest.approx(0.0)
    assert res.loc["G1", "tau"] == pytest.approx(1.0)
    assert np.isnan(res.loc["G2", "tau"]) and res.loc["G2", "n_tissues_expressed"] == 0
    assert res.loc["G1", "n_tissues_expressed"] == 2
    # BC matches scalar reference, averaged over tissues.
    ref = np.mean([F.bimodality_coefficient(dense[(obs.tissue == t).to_numpy(), 3])
                   for t in ["A", "B"]])
    assert res.loc["G3", "bimodality"] == pytest.approx(ref, rel=1e-8)
    assert res.loc["G3", "bimodality"] > F.BC_THRESHOLD
    # Vectorized BC agrees with scipy for a skewed column.
    x = np.random.default_rng(3).gamma(2.0, size=200)
    bc, n = F._bc_columns(sp.csr_matrix(x.reshape(-1, 1)))
    g1, g2 = sstats.skew(x, bias=False), sstats.kurtosis(x, bias=False)
    expected = (g1**2 + 1) / (g2 + 3 * (199**2) / (198 * 197))
    assert bc[0] == pytest.approx(expected, rel=1e-8) and n[0] == 200

    # Ensembl keying + gene subset + per-tissue long table.
    res2 = F.compute_gene_features(adata, gene_id_column="ensembl_id", genes=["ENSG01", "G1"])
    assert list(res2.index) == ["ENSG01", "ENSG02"]
    long = F.compute_gene_features(adata, per_tissue=True)
    assert set(long.tissue) == {"A", "B"}
    # Tissue B: t5 (5 cells) excluded -> 2 cell types.
    assert long.loc[long.tissue == "B", "n_celltypes"].iloc[0] == 2


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def _sim(n=400, beta=1.2, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(beta * x - 0.2)))
    y = rng.binomial(1, p)
    ae = 100 / (1 + np.exp(-(-1.5 - 0.6 * x + rng.normal(0, 0.3, n))))
    return pd.DataFrame({
        "tau": x, "y": y, "ae": ae,
        "gen": rng.binomial(1, 0.3, n),
        "phase": rng.choice([1, 2, 3], n),
        "start_year": rng.integers(2000, 2020, n),
        "modality": rng.choice(["sm", "ab", "other"], n),
        "therapeutic_area": rng.choice(["onc", "neuro", "cv", "imm"], n),
    })


def test_logistic_direction_and_adjusted():
    df = _sim()
    r = S.univariate_logistic(df, "tau", "y")
    assert r["odds_ratio"] > 1.5 and r["p"] < 1e-6 and r["ci_low"] > 1 and r["n"] == 400
    neg = df.assign(tau=-df.tau)
    assert S.univariate_logistic(neg, "tau", "y")["odds_ratio"] < 1
    a = S.adjusted_logistic(df, "tau", "y", ["gen", "modality"])
    assert a["odds_ratio"] > 1.5
    df.loc[:5, "tau"] = np.nan
    assert S.univariate_logistic(df, "tau", "y")["n"] == 394


def test_beta_regression_and_permutation():
    df = _sim()
    b = S.beta_regression(df, "tau", "ae")
    assert b["coef"] < 0 and b["p"] < 1e-6 and b["n"] == 400
    y = S.smithson_verkuilen(np.array([0.0, 1.0]), 10)
    assert 0 < y[0] < y[1] < 1
    perm = S.permutation_test(df, "tau", "y", "logistic", n_perm=50, seed=1)
    assert perm["p_perm"] == pytest.approx(1 / 51) and perm["p_perm_raw"] == 0.0
    permb = S.permutation_test(df.iloc[:120], "tau", "ae", "beta", n_perm=20, seed=1)
    assert permb["p_perm"] <= 1 / 21 + 1e-12


def test_benjamini_hochberg_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests

    p = np.array([0.001, 0.02, 0.03, 0.2, 0.5, 0.04])
    np.testing.assert_allclose(S.benjamini_hochberg(p), multipletests(p, method="fdr_bh")[1])
    out = S.benjamini_hochberg([0.01, np.nan, 0.04])
    assert np.isnan(out[1])
    np.testing.assert_allclose(out[[0, 2]], multipletests([0.01, 0.04], method="fdr_bh")[1])


def test_binarize_tau_and_contrasts():
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.normal(0.4, 0.03, 100), rng.normal(0.9, 0.03, 60), [np.nan]])
    thr, lab = S.binarize_tau(v)
    assert 0.5 < thr < 0.8
    assert np.isnan(lab[-1]) and lab[:100].sum() == 0 and lab[100:160].sum() == 60

    df = pd.DataFrame({"spec": [1, 1, 1, 1, 0, 0, 0, 0],
                       "ph4": [1, 1, 1, 0, 1, 1, 0, 0],
                       "ae1": [1.0, 2, 3, 2, 4, 4, 4, 4],
                       "ae2": [np.nan, 2, 1, 3, 4, 2, 6, 4]})
    rl = S.relative_likelihood(df, "spec", "ph4")
    assert rl["p_specific"] == 0.75 and rl["p_broad"] == 0.5
    assert rl["relative_increase"] == pytest.approx(0.5)
    ae = S.relative_ae_difference(df, "spec", ["ae1", "ae2"])
    assert ae["per_outcome"]["ae1"]["relative_difference"] == pytest.approx(0.5 - 1)
    assert ae["overall"]["relative_difference"] < 0


def test_trial_level_features_and_genetic_flags():
    mapping = pd.DataFrame({
        "nct_id": ["N1", "N1", "N2", "N3"],
        "targetId": ["T1", "T2", "T2", "T9"],
        "diseaseId": ["D1", "D1", "D2", "D3"],
    })
    gf = pd.DataFrame({"tau": [0.9, 0.4], "bimodality": [0.3, 0.7]},
                      index=pd.Index(["T1", "T2"], name="gene"))
    tl = S.trial_level_features(mapping, gf)
    assert tl.loc["N1", "tau"] == 0.4 and tl.loc["N1", "bimodality"] == 0.3
    assert tl.loc["N1", "n_targets"] == 2
    assert np.isnan(tl.loc["N3", "tau"]) and tl.loc["N3", "n_targets_with_features"] == 0
    flags = S.genetic_evidence_flags(mapping, {("T2", "D1")})
    assert flags.to_dict() == {"N1": True, "N2": False, "N3": False}


def test_run_association_suite():
    df = _sim(n=200)
    res = S.run_association_suite(df, ["tau"], ["y"], ["ae"], n_perm=10, covariates=["gen"])
    # with covariates the AE model is the adjusted beta regression (no unadjusted beta row)
    assert list(res.model) == ["logistic", "logistic_adjusted", "beta_adjusted"]
    assert {"p_fdr", "p_perm", "estimate", "ci_low", "ci_high", "n", "engine"} <= set(res.columns)
    assert set(res.engine) == {"statsmodels"}
    plain = S.run_association_suite(df, ["tau"], ["y"], ["ae"], n_perm=0)
    assert list(plain.model) == ["logistic", "beta"]
    assert (res.p_fdr >= res.p - 1e-15).all()


def test_mixed_effects_fallback():
    df = _sim(n=300)
    r = S.mixed_effects_logistic(df, "tau", "y", engine="statsmodels")
    assert r["engine"] == "statsmodels_bayes_mixed_glm_vb"
    assert r["odds_ratio"] > 1.5 and r["n"] == 300
    auto = S.mixed_effects_logistic(df, "tau", "y")  # no rpy2 -> fallback, no crash
    assert auto["engine"] in {"lme4", "statsmodels_bayes_mixed_glm_vb"}
    b = S.mixed_effects_beta(df, "tau", "ae", engine="statsmodels")
    assert b["engine"] == "statsmodels_betareg_fixed_dummies" and b["coef"] < 0
    b2 = S.mixed_effects_beta(df, "tau", "ae")
    assert b2["engine"] in {"glmmTMB", "statsmodels_betareg_fixed_dummies"}


def test_phase_to_numeric():
    assert S.phase_to_numeric("PHASE2") == 2
    assert S.phase_to_numeric("Phase 1/Phase 2") == 2
    assert S.phase_to_numeric("EARLY_PHASE1") == 0.5
    assert S.phase_to_numeric(4.0) == 4
    assert S.phase_to_numeric("Phase IV") == 4
    assert np.isnan(S.phase_to_numeric(None))


def test_build_outcomes():
    labels = pd.DataFrame({
        "nct_id": ["N1", "N2", "N3", "N4", "N5"],
        "phase": ["PHASE1", "PHASE2", "PHASE3", "PHASE2", "PHASE1"],
        "status": ["Completed", "Terminated", "Completed", "Withdrawn", "Terminated"],
        "studyStopReasonCategories": [None, None, None, "['Business_Administrative']", None],
        "phase2_progression": ["EVER", None, None, None, "NEVER"],
        "primary_endpoint_result": ["POSITIVE", "NEGATIVE", "UNKNOWN", np.nan, "POSITIVE"],
        "secondary_endpoint_result": [np.nan, "POSITIVE", "NEGATIVE", np.nan, np.nan],
        "virtualbiotech_stop_reason_categories": [None, "Safety_Sideeffects|Negative", None,
                                                  None, "Negative"],
        "ae_serious_any_pct": [1.0, 5.0, np.nan, 0.0, 2.0],
    })
    mapping = pd.DataFrame({
        "nct_id": ["N1", "N2", "N3", "N4", "N5", "N6"],
        "drugId": ["D_A", "D_B", "D_C", "D_A", "D_E", "D_B"],
        "targetId": ["T1"] * 6,
        "diseaseId": ["X", "Y", "Z", "X", "W", "Y"],
        "phase": [1.0, 2.0, 3.0, 2.0, 1.0, 4.0],
        "status": ["Completed"] * 6,
        "trial_date": ["2010-01-05", "2012-03-01", "2015-06-01", "2011-01-01", "2019-02-02",
                       "2020-01-01"],
    })
    out = S.build_outcomes(labels, mapping).set_index("nct_id")
    assert out.loc["N1", "primary_success"] == 1 and out.loc["N2", "primary_success"] == 0
    assert np.isnan(out.loc["N3", "primary_success"])
    assert out.loc["N3", "secondary_success"] == 0
    assert out.loc["N1", "phase1_to_2"] == 1 and out.loc["N5", "phase1_to_2"] == 0
    assert np.isnan(out.loc["N2", "phase1_to_2"])
    assert out.loc["N2", "stopped_early"] == 1 and out.loc["N1", "stopped_early"] == 0
    assert out.loc["N2", "stopped_safety"] == 1 and out.loc["N2", "stopped_negative"] == 1
    assert np.isnan(out.loc["N5", "stopped_safety"]) and out.loc["N5", "stopped_negative"] == 1
    assert np.isnan(out.loc["N4", "stopped_negative"])  # stopped for business reasons
    # (D_B, Y) reached phase 4 via N6.
    assert out.loc["N2", "ever_phase4"] == 1 and out.loc["N1", "ever_phase4"] == 0
    # (D_A, X) reached phase 2 via N4 -> N1 progressed.
    assert out.loc["N1", "ever_phase_ge_2"] == 1 and out.loc["N5", "ever_phase_ge_2"] == 0
    assert np.isnan(out.loc["N3", "ever_phase_ge_3"])  # phase 3 not eligible
    assert out.loc["N1", "start_year"] == 2010
    assert out.loc["N2", "ae_serious_any_pct"] == 5.0
    assert "N6" in out.index and out.loc["N6", "phase_num"] == 4
    assert "negative" in out.loc["N2", "stop_categories"]


# ---------------------------------------------------------------------------
# P7: adjusted models, FDR families, contrasts, replication, Phase I scope
# ---------------------------------------------------------------------------


def _confounded(n=3000, seed=0):
    """Genetic evidence drives both tau and AE%; tau itself has no effect on AE."""
    rng = np.random.default_rng(seed)
    gen = rng.binomial(1, 0.4, n)
    tau = 0.8 * gen + rng.normal(0, 0.5, n)
    ae = 100 / (1 + np.exp(-(-2.0 + 1.0 * gen + rng.normal(0, 0.3, n))))
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.8 * gen - 0.3))), n)
    return pd.DataFrame({"tau": tau, "gen": gen, "ae": ae, "y": y, "y2": rng.binomial(1, 0.4, n)})


def test_adjusted_beta_is_adjusted():
    df = _confounded()
    unadj = S.run_association_suite(df, ["tau"], [], ["ae"], n_perm=0).set_index("model")
    adj = S.run_association_suite(df, ["tau"], [], ["ae"], n_perm=0, covariates=["gen"]).set_index("model")
    assert unadj.loc["beta", "estimate"] > 0.15  # confounded association
    assert "beta" not in adj.index
    assert abs(adj.loc["beta_adjusted", "estimate"]) < 0.03  # adjustment removes it
    direct = S.beta_regression(df, "tau", "ae", covariates=["gen"])
    assert adj.loc["beta_adjusted", "estimate"] == pytest.approx(direct["coef"])


def test_fdr_family_after_filtering():
    from statsmodels.stats.multitest import multipletests

    df = _confounded(n=600)
    full = S.run_association_suite(df, ["tau"], ["y", "y2"], [], n_perm=0, covariates=["gen"])
    fam = S.refdr(full.query("model != 'logistic'"))
    assert len(fam) == 2
    np.testing.assert_allclose(fam["p_fdr"], multipletests(fam["p"], method="fdr_bh")[1])
    only = S.run_association_suite(df, ["tau"], ["y", "y2"], [], n_perm=0, covariates=["gen"],
                                   include_unadjusted=False)
    assert list(only.model) == ["logistic_adjusted"] * 2
    np.testing.assert_allclose(only["p_fdr"], fam["p_fdr"])


def test_therapeutic_area_union():
    assert S.combine_therapeutic_areas(["onc|imm", "cv", None, "imm"]) == "cv|imm|onc"
    assert S.combine_therapeutic_areas([["b", "a"], ["c"]]) == "a|b|c"
    assert S.combine_therapeutic_areas([None]) == "unknown"


def test_relative_ae_difference_definitions():
    df = pd.DataFrame({"spec": [1, 1, 0, 0],
                       "ae1": [1.0, 1.0, 2.0, 2.0],     # -50% per organ
                       "ae2": [np.nan, 9.0, 10.0, np.nan]})  # -10% per organ
    r = S.relative_ae_difference(df, "spec", ["ae1", "ae2"], with_beta=True)
    assert r["mean_of_per_organ"]["relative_difference"] == pytest.approx(-0.3)
    assert r["paper_comparison"] == "mean_of_per_organ"
    # pooled row mean: specific (1 + 5)/2 = 3, broad (6 + 2)/2 = 4 -> -25%
    assert r["pooled_row_mean"]["relative_difference"] == pytest.approx(-0.25)
    assert r["overall"]["relative_difference"] == pytest.approx(-0.25)
    assert "exp_beta_pooled" in r and "definition" in r["exp_beta_pooled"]


def test_genetic_evidence_replication():
    df = _confounded(n=1500).rename(columns={"gen": "genetic_evidence"})
    rng = np.random.default_rng(2)
    df["status"] = rng.choice(["Completed", "Terminated"], len(df), p=[0.7, 0.3])
    df["stop_categories"] = np.where(df["status"] == "Terminated",
                                     rng.choice(["negative", "business_administrative"], len(df)), None)
    rep = S.genetic_evidence_replication(df, ["y", "y2"], ["ae"])
    r = rep.set_index("outcome")
    assert r.loc["y", "estimate"] > 1.5 and r.loc["y", "p"] < 1e-6  # OR for the flag
    assert r.loc["ae", "model"] == "beta" and r.loc["ae", "estimate"] > 0.5
    assert {"stopped:negative", "stopped:business_administrative"} <= set(r.index)
    assert (rep["analysis"] == "genetic_evidence_replication").all() and rep["p_fdr"].notna().all()


def test_bimodality_kurtosis_variants():
    rng = np.random.default_rng(0)
    x = rng.gamma(2.0, size=300)
    exc = F.bimodality_coefficient(x)
    pea = F.bimodality_coefficient(x, kurtosis="pearson")
    g1, g2 = sstats.skew(x, bias=False), sstats.kurtosis(x, bias=False)
    corr = 3 * 299**2 / (298 * 297)
    assert pea == pytest.approx((g1**2 + 1) / (g2 + 3 + corr)) and pea < exc
    biased = F.bimodality_coefficient(x, bias_correction=False)
    b1, b2 = sstats.skew(x), sstats.kurtosis(x)
    assert biased == pytest.approx((b1**2 + 1) / (b2 + corr))
    bc, _ = F._bc_columns(sp.csr_matrix(x.reshape(-1, 1)), "pearson", False)
    assert bc[0] == pytest.approx((b1**2 + 1) / (b2 + 3 + corr))
    with pytest.raises(ValueError):
        F.bimodality_coefficient(x, kurtosis="fisher")
    adata, _, _ = _fake_adata()
    a = F.compute_gene_features(adata, kurtosis="pearson")
    assert a.attrs["bimodality_kurtosis"] == "pearson" and a.attrs["n_tissues_with_tau"] == 2
    assert a.attrs["kept_by_tissue"]["B"] == {"n_cells": 55, "n_celltypes": 3, "n_celltypes_kept": 2}


def test_feature_input_detection():
    adata, dense, obs = _fake_adata()
    assert F.detect_gene_id_column(adata) == "ensembl_id"
    ens = FakeAnnData(adata.X, obs, pd.DataFrame(index=["ENSG0001", "ENSG0002", "ENSG0003", "ENSG0004"]))
    assert F.detect_gene_id_column(ens) is None
    none = FakeAnnData(adata.X, obs, pd.DataFrame(index=["A", "B", "C", "D"]))
    with pytest.raises(KeyError):
        F.detect_gene_id_column(none)
    counts = sp.csr_matrix(np.random.default_rng(0).poisson(3, size=(50, 4)).astype(float))
    assert F.looks_like_counts(counts) and not F.looks_like_counts(sp.csr_matrix(np.log1p(counts.toarray())))
    # normalize=True equals normalising first
    raw = FakeAnnData(counts, pd.DataFrame({"tissue": ["A"] * 50, "cell_ontology_class": ["x"] * 25 + ["y"] * 25}),
                      pd.DataFrame(index=["G0", "G1", "G2", "G3"]))
    lib = counts.toarray().sum(1, keepdims=True)
    normed = FakeAnnData(sp.csr_matrix(np.log1p(counts.toarray() / lib * 1e4)), raw.obs, raw.var)
    a = F.compute_gene_features(raw, normalize=True)
    b = F.compute_gene_features(normed)
    np.testing.assert_allclose(a["tau"], b["tau"], rtol=1e-10)


def test_phase1_status_scope():
    from vbt.case_studies.trial_outcomes.phase1 import phase1_progression

    m = pd.DataFrame({
        "nct_id": ["P1", "P2", "P3", "Q1"], "drugId": ["D"] * 4, "diseaseId": ["E"] * 4,
        "phase": [1.0, 1.0, 1.0, 2.0], "status": ["Completed", "Recruiting", "Terminated", "Completed"],
        "trial_date": ["2010-01-01", "2010-01-01", "2010-01-01", "2012-01-01"]})
    assert list(phase1_progression(m).nct_id) == ["P1"]
    assert len(phase1_progression(m, statuses=None)) == 3
    assert list(phase1_progression(m, statuses=None, universe={"P2"}).nct_id) == ["P2"]


def test_phase1_released_scope_11412():
    import os
    from pathlib import Path

    up = Path(os.environ.get("VBT_UPSTREAM", "third_party/TheVirtualBiotech")) / "datasets" / "clinical_trials"
    if not (up / "chembl_clinical_nct_data.parquet").exists():
        pytest.skip("upstream clinical-trial data not available")
    from vbt.case_studies.trial_outcomes.phase1 import phase1_progression

    res = phase1_progression(pd.read_parquet(up / "chembl_clinical_nct_data.parquet"))
    assert len(res) == 11412
    released = pd.read_csv(up / "clinical_trial_labels_reconciled.csv")
    rel = released[(released.phase == 1.0) & released.phase2_progression.notna()]
    assert set(res.nct_id) == set(rel.nct_id)


def test_agreement_applicability():
    from vbt.case_studies.trial_outcomes.validation import agreement_report

    pred = pd.DataFrame({"nct_id": list("ABCDE"),
                         "primary_endpoint_result": ["POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE"],
                         "ae_serious_cardiac_pct": [1.0, 2.0, 3.0, np.nan, 5.0]})
    ref = pd.DataFrame({"nct_id": list("ABCDE"),
                        "status": ["Completed", "Completed", "Terminated", "Completed", "Withdrawn"],
                        "primary_endpoint_result": ["POSITIVE", "POSITIVE", "NEGATIVE", "NOT_APPLICABLE",
                                                    "POSITIVE"],
                        "ae_serious_cardiac_pct": [1.2, np.nan, 3.0, np.nan, 9.0]})
    rep = agreement_report(pred, ref).set_index("field")
    p = rep.loc["primary_endpoint_result"]
    assert p["n_applicable"] == 2 and p["n_agree"] == 1  # C, E stopped; D not applicable
    assert p["n_excluded_stopped"] == 2 and p["n_excluded_not_applicable"] == 1
    inc = agreement_report(pred, ref, exclude_stopped=False).set_index("field")
    assert inc.loc["primary_endpoint_result", "n_applicable"] == 4
    ae = rep.loc["serious_ae_rates"]
    assert ae["n_applicable"] == 3 and ae["n_agree"] == 2 and ae["n_excluded_no_exact_statistics"] == 2
