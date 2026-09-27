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
    assert list(res.model) == ["logistic", "logistic_adjusted", "beta"]
    assert {"p_fdr", "p_perm", "estimate", "ci_low", "ci_high", "n"} <= set(res.columns)
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
