"""Case 1 replication against the authors' code: unit tests + archive integration (skip if absent)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vbt.case_studies.trial_outcomes import replicate as R
from vbt.case_studies.trial_outcomes import stats as S


# ---------------------------------------------------------------------------
# authors' outcome structure
# ---------------------------------------------------------------------------


def _labels():
    return pd.DataFrame({
        "nct_id": ["N1", "N2", "N3", "N4", "N5", "N6"],
        "phase": [1.0, 2.0, 3.0, 2.0, 4.0, 0.5],
        "status": ["Completed", "Terminated", "Completed", "Withdrawn", "Completed", "Suspended"],
        "studyStopReasonCategories": [None, "Negative", None, "Business or administrative", None, None],
        "virtualbiotech_stop_reason_categories": [None, "Safety or side effects", None, "Negative", None,
                                                  "Regulatory|COVID-19"],
        "phase2_progression": ["EVER", None, None, None, None, None],
        "primary_endpoint_result": [None, "NEGATIVE", "UNKNOWN", np.nan, "POSITIVE", None],
        "secondary_endpoint_result": [None, "POSITIVE", "NEGATIVE", np.nan, None, None],
    })


def test_authors_outcomes():
    lab = _labels()
    assert S.authors_outcome(lab, "phase", 2.0).tolist() == [0, 1, 1, 1, 1, 0]
    assert S.authors_outcome(lab, "status", "Withdrawn").tolist() == [0, 0, 0, 1, 0, 0]
    # ChEMBL first: N2 is Negative (not Safety), N4 Business (VB 'Negative' ignored), N6 falls back to VB
    assert S.authors_outcome(lab, "stop_reason", "Negative").tolist() == [0, 1, 0, 0, 0, 0]
    assert S.authors_outcome(lab, "stop_reason", "COVID-19").tolist() == [0, 0, 0, 0, 0, 1]
    assert S.authors_outcome(lab, "stop_reason", "Negative", "vb_first").tolist() == [0, 0, 0, 1, 0, 0]
    sub = S.authors_subset(lab, "endpoint")
    assert sub["nct_id"].tolist() == ["N2", "N3", "N4"]
    # everything that is not POSITIVE is a failure (UNKNOWN / missing -> 0)
    assert S.authors_outcome(sub, "endpoint", "primary").tolist() == [0, 0, 0]
    assert S.authors_outcome(sub, "endpoint", "either").tolist() == [1, 0, 0]
    assert S.authors_subset(lab, "stop_reason")["nct_id"].tolist() == ["N2", "N4", "N6"]
    p = S.authors_outcome(lab, "phase1_progression")
    assert p.iloc[0] == 1 and p.iloc[1:].isna().all()
    assert S.authors_outcome_frame(lab, "phase", 2.0, min_positive=30) is None


def test_build_outcomes_authors_columns():
    lab = _labels()
    mapping = pd.DataFrame({"nct_id": lab["nct_id"], "drugId": "D", "targetId": "T", "diseaseId": "X",
                            "phase": lab["phase"], "status": lab["status"], "trial_date": "2015-01-01"})
    out = S.build_outcomes(lab, mapping).set_index("nct_id")
    assert out.loc["N3", "primary_success"] == 0 and np.isnan(out.loc["N1", "primary_success"])
    assert out.loc["N2", "either_success"] == 1
    assert out.loc["N5", "phase_ge_4"] == 1 and out.loc["N6", "phase_ge_2"] == 0
    assert out.loc["N4", "status_withdrawn"] == 1 and out.loc["N2", "status_withdrawn"] == 0
    assert out.loc["N2", "stop_negative"] == 1 and out.loc["N2", "stop_safety_or_side_effects"] == 0
    assert np.isnan(out.loc["N1", "stop_negative"])
    assert out.loc["N6", "stop_covid_19"] == 1
    strict = S.build_outcomes(lab, mapping, endpoint_coding="strict").set_index("nct_id")
    assert np.isnan(strict.loc["N3", "primary_success"]) and strict.loc["N2", "primary_success"] == 0


def test_aggregate_trials_min_and_drop():
    lab = pd.DataFrame({"nct_id": ["A", "B", "C"], "phase": [2.0, 3.0, 1.0]})
    mapping = pd.DataFrame({"nct_id": ["A", "A", "B", "C", "C"], "targetId": ["g1", "g2", "g2", "g3", "g9"]})
    feats = pd.DataFrame({"ensembl_id": ["g1", "g2", "g3"], "tau_cell_type": [0.9, 0.4, np.nan],
                          "bimodality_score": [0.2, 0.7, 0.5]})
    t = S.aggregate_trials(lab, mapping, feats).set_index("nct_id")
    assert list(t.index) == ["A", "B"]  # C lacks tau
    assert t.loc["A", "tau_cell_type"] == 0.4 and t.loc["A", "bimodality_score"] == 0.2
    assert t.loc["A", "n_targets"] == 2
    tm = S.aggregate_trials(lab, mapping, feats, how="mean").set_index("nct_id")
    assert tm.loc["A", "tau_cell_type"] == pytest.approx(0.65)


def test_genetic_and_combos():
    mapping = pd.DataFrame({"nct_id": ["A", "A", "B"], "targetId": ["t1", "t2", "t1"],
                            "diseaseId": ["d1", "d1", "d2"], "drugId": ["x", "y", "x"]})
    g = S.genetic_evidence_table(mapping, pd.DataFrame({"targetId": ["t2"], "diseaseId": ["d1"]})).set_index("nct_id")
    assert g.loc["A", "has_genetic_evidence_any"] == 1 and g.loc["A", "fraction_genetic_evidence"] == 0.5
    assert g.loc["B", "has_genetic_evidence_any"] == 0
    dt = S.drugtype_combo(mapping, {"x": "Small molecule", "y": "Antibody"}).set_index("nct_id")
    assert dt.loc["A", "drugType_combo"] == "Antibody-Small molecule"
    dis = pd.DataFrame({"id": ["d1", "d2"], "name": ["D1", "D2"],
                        "therapeuticAreas": [["EFO_0000651", "MONDO_0002025"], ["EFO_0009605"]]})
    ta = S.ta_combo(mapping, dis).set_index("nct_id")
    assert ta.loc["A", "ta_combo"] == "EFO_0000618"  # phenotype dropped, psychiatric -> nervous system
    assert ta.loc["B", "ta_combo"] == "EFO_0001379"


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------


def test_betareg_fit_matches_statsmodels_point_estimates():
    from statsmodels.othermod.betareg import BetaModel

    rng = np.random.default_rng(0)
    n = 800
    x = rng.normal(size=n)
    mu = 1 / (1 + np.exp(-(-1.0 + 0.4 * x)))
    y = rng.beta(mu * 12, (1 - mu) * 12)
    X = np.column_stack([np.ones(n), x])
    fit = S.betareg_fit(y, X)
    sm = BetaModel(y, X).fit(disp=0)
    np.testing.assert_allclose(fit["params"], np.asarray(sm.params)[:2], rtol=1e-5)
    assert fit["phi"] == pytest.approx(np.exp(np.asarray(sm.params)[2]), rel=1e-4)
    assert abs(fit["params"][1] - 0.4) < 0.1 and fit["converged"]
    # expected-information SEs are close to (not identical with) the observed-Hessian ones
    assert fit["se"][1] == pytest.approx(np.asarray(sm.bse)[1], rel=0.05)
    fast = S.betareg_fit(y, X, optimizer="scoring")
    np.testing.assert_allclose(fast["params"], fit["params"], rtol=1e-6)


def test_beta_regression_clip_default():
    df = pd.DataFrame({"f": np.linspace(0, 1, 60), "ae": np.r_[np.zeros(10), np.linspace(1, 30, 50)]})
    r = S.beta_regression(df, "f", "ae")
    assert r["squeeze"] == "clip" and r["engine"] in ("betareg_py", "R:betareg") and r["coef"] > 0
    sv = S.beta_regression(df, "f", "ae", squeeze="smithson_verkuilen")
    assert sv["coef"] != pytest.approx(r["coef"])


def test_glmm_laplace_recovers_effects():
    rng = np.random.default_rng(1)
    n, g1, g2 = 3000, 8, 25
    a = rng.integers(0, g1, n)
    b = rng.integers(0, g2, n)
    x = rng.normal(size=n)
    eta = -0.3 + 0.5 * x + rng.normal(0, 0.6, g1)[a] + rng.normal(0, 0.4, g2)[b]
    y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    X = np.column_stack([np.ones(n), x])
    fit = S.glmm_laplace(y, X, [a, b], family="binomial")
    assert fit["converged"] and abs(fit["params"][1] - 0.5) < 0.1
    assert fit["se"][1] > 0 and fit["n_groups"] == [g1, g2] and fit["sd"][0] > 0.2
    # nAGQ=0 and nAGQ=1 agree closely at this size
    f0 = S.glmm_laplace(y, X, [a, b], family="binomial", nagq=0)
    assert abs(f0["params"][1] - fit["params"][1]) < 0.01


def test_mixed_effects_laplace_engine():
    rng = np.random.default_rng(2)
    n = 600
    df = pd.DataFrame({"tau": rng.normal(size=n), "mod": rng.choice(list("abcd"), n),
                       "ta": rng.choice(list("pqrstu"), n), "year": rng.integers(2000, 2020, n)})
    df["y"] = rng.binomial(1, 1 / (1 + np.exp(-(0.8 * df["tau"]))))
    df["ae"] = 100 / (1 + np.exp(-(-2 - 0.3 * df["tau"] + rng.normal(0, 0.5, n))))
    r = S.mixed_effects_logistic(df, "tau", "y", fixed=("year",), random=("mod", "ta"), engine="laplace")
    assert r["engine"] == "laplace" and r["odds_ratio"] > 1.5
    b = S.mixed_effects_beta(df, "tau", "ae", fixed=("year",), random=("mod", "ta"), engine="laplace")
    assert b["engine"] == "laplace" and b["coef"] < 0 and b["phi"] > 0


def test_contingency_and_fast_logit():
    x = np.array([1] * 60 + [0] * 40)
    y = np.array([1] * 30 + [0] * 30 + [1] * 10 + [0] * 30)
    r = S.contingency_or(x, y)
    assert r["OR"] == pytest.approx(30 * 30 / (30 * 10)) and r["fold"] == pytest.approx(0.5 / 0.25)
    assert r["pct_change"] == pytest.approx(100.0)
    rng = np.random.default_rng(3)
    xx = rng.normal(size=500)
    yy = rng.binomial(1, 1 / (1 + np.exp(-(0.2 + 0.7 * xx)))).astype(float)
    d = pd.DataFrame({"x": xx, "y": yy})
    assert R._fast_logit_coef(R._z(xx), yy) == pytest.approx(S.univariate_logistic(d, "x", "y")["beta"], rel=1e-8)


def test_kmeans_threshold_and_target_level():
    rng = np.random.default_rng(0)
    v = np.r_[rng.normal(0.4, 0.03, 200), rng.normal(0.9, 0.03, 100)]
    thr, (lo, hi) = S.kmeans_threshold(v)
    assert lo < thr < hi and thr == pytest.approx((lo + hi) / 2)
    lwt = pd.DataFrame({"targetId": ["a", "a", "b", "c", "d"], "nct_id": ["1", "2", "3", "4", "5"],
                        "phase": [1.0, 4.0, 2.0, 4.0, 1.0]})
    tl = S.target_level_table(lwt, pd.Series({"a": 0.9, "b": 0.95, "c": 0.3, "d": 0.2}), 0.5)
    p4 = tl[(tl.model == "unadjusted") & (tl.outcome == "Ever reached Phase IV")].iloc[0]
    assert p4["rate_hi"] == 0.5 and p4["rate_lo"] == 0.5 and p4["n_total"] == 4


# ---------------------------------------------------------------------------
# comparison logic
# ---------------------------------------------------------------------------


def test_compare_tables_alignment():
    theirs = pd.DataFrame({"section": ["Phase", "Phase", "AE"], "outcome": ["II+", "IV", "Inf"],
                           "feature": ["tau"] * 3, "regression_type": ["logistic", "logistic", "beta"],
                           "OR": [1.2, 1.4, np.nan], "coefficient": [np.log(1.2), np.log(1.4), -0.1],
                           "CI_lower": [1.1, 1.3, -0.2], "CI_upper": [1.3, 1.5, 0.0], "p_value": [1e-5, 1e-9, 0.01],
                           "n_positive": [10, 5, np.nan], "n_reference": [90, 95, 50]})
    ours = theirs.copy()
    ours.loc[1, "OR"] = 1.41  # 0.7% off -> mismatch
    ours.loc[2, "n_reference"] = 49  # n mismatch
    ours = ours.drop(index=0).assign(engine="statsmodels")  # row 0 missing
    c = R.compare_tables(ours, theirs, table="t", keys=["section", "outcome", "feature", "regression_type"],
                         estimate={"logistic": "OR", "beta": "coefficient"}, log_scale="OR")
    c = c.set_index("outcome")
    assert not c.loc["II+", "match"] and c.loc["II+", "engine"] == "missing"
    assert not c.loc["IV", "match"] and c.loc["IV", "abs_delta"] == pytest.approx(np.log(1.41 / 1.4))
    assert c.loc["Inf", "abs_delta"] == 0 and not c.loc["Inf", "n_match"] and not c.loc["Inf", "match"]
    ours.loc[1, "OR"] = 1.4 * (1 + 1e-5)
    ours.loc[2, "n_reference"] = 50
    c2 = R.compare_tables(ours, theirs, table="t", keys=["section", "outcome", "feature", "regression_type"],
                          estimate={"logistic": "OR", "beta": "coefficient"}, log_scale="OR")
    assert c2.set_index("outcome").loc[["IV", "Inf"], "match"].all()


def _synthetic_inputs(n_trials=400, n_genes=60, seed=0):
    rng = np.random.default_rng(seed)
    genes = [f"ENSG{i:05d}" for i in range(n_genes)]
    tau = rng.uniform(0.2, 1.0, n_genes)
    feats = pd.DataFrame({"ensembl_id": genes, "tau_cell_type": tau,
                          "bimodality_score": np.clip(tau * 0.5 + rng.normal(0, 0.1, n_genes), 0, 1),
                          "mean_expr": rng.gamma(2, 1, n_genes)})
    ids = [f"NCT{i:08d}" for i in range(n_trials)]
    rows = []
    for i, t in enumerate(ids):
        for g in rng.choice(n_genes, rng.integers(1, 4), replace=False):
            rows.append((t, genes[g], f"D{i % 7}", f"drug{i % 13}"))
    mapping = pd.DataFrame(rows, columns=["nct_id", "targetId", "diseaseId", "drugId"])
    mapping["trial_date"] = "2010-01-01"
    tmin = mapping.merge(feats, left_on="targetId", right_on="ensembl_id").groupby("nct_id")["tau_cell_type"].min()
    p = 1 / (1 + np.exp(-(-0.5 + 2 * (tmin.reindex(ids).to_numpy() - 0.5))))
    phase = rng.choice([1.0, 2.0, 3.0, 4.0], n_trials)
    lab = pd.DataFrame({"nct_id": ids, "phase": phase,
                        "status": rng.choice(["Completed", "Terminated"], n_trials, p=[0.8, 0.2]),
                        "studyStopReasonCategories": None, "virtualbiotech_stop_reason_categories": None,
                        "phase2_progression": np.where(phase == 1.0, np.where(rng.random(n_trials) < p, "EVER",
                                                                               "NEVER"), None),
                        "primary_endpoint_result": np.where(rng.random(n_trials) < p, "POSITIVE", "NEGATIVE"),
                        "secondary_endpoint_result": "UNKNOWN",
                        "ae_serious_infections_pct": np.where(rng.random(n_trials) < 0.5,
                                                              rng.uniform(0, 20, n_trials), np.nan)})
    return dict(labels=lab, mapping=mapping, features=feats,
                genetic=pd.DataFrame({"targetId": genes[:10], "diseaseId": ["D0"] * 10}))


def test_run_univariate_and_checks_on_synthetic():
    inputs = _synthetic_inputs()
    trials = R.build_trials(inputs)
    assert trials["nct_id"].is_monotonic_increasing and "n_targets_all" in trials
    u = R.run_univariate(trials, ["tau_cell_type", "bimodality_score"])
    prim = u[(u.outcome == "Primary Positive") & (u.feature == "tau_cell_type")].iloc[0]
    assert prim["OR"] > 1 and prim["n_positive"] + prim["n_reference"] == ((trials.phase == 2) | (trials.phase == 3)).sum()
    assert (u.regression_type == "beta").any() and u["fdr_adjusted_p"].notna().all()
    perm = R.run_permutation(trials, n_perm=20, n_perm_beta=5)
    assert {"empirical_p_value", "null_std"} <= set(perm.columns)
    assert (perm["empirical_p_value"] >= 1 / 21 - 1e-12).all()
    # identical seed -> identical nulls (the authors' single-RNG order)
    perm2 = R.run_permutation(trials, n_perm=20, n_perm_beta=5)
    np.testing.assert_allclose(perm["null_mean"], perm2["null_mean"])
    chk = R.harness_checks(inputs, trials, gene_perm=10)
    assert set(chk["check"]) == {"adjusted_n_targets", "gene_permutation_null"}
    g = chk[chk.check == "gene_permutation_null"]
    assert (g["n_iter"] == 10).all() and g["null_q025"].le(g["null_q975"]).all()
    gen = R.run_genetic(inputs)
    assert set(gen["feature"]) == {"has_genetic_evidence_any", "fraction_genetic_evidence"}


# ---------------------------------------------------------------------------
# integration on the archive
# ---------------------------------------------------------------------------


def _archive_root():
    from vbt.data.zenodo import zenodo_root

    root = zenodo_root()
    if not (root / "clinical_trials" / "data" / "clinical_trial_labels_reconciled.csv").exists():
        pytest.skip("Zenodo archive extract not present (set VBT_ZENODO_DIR or run `vbt data zenodo fetch "
                    "--preset case1`)")
    return root


def test_archive_univariate_matches_authors(tmp_path):
    root = _archive_root()
    rep = R.replicate_case1(root, tmp_path, n_perm=0, gene_perm=3, mixed=False, expr=False, combined=False,
                            progress=None)
    c = rep.comparison
    uni = c[c.table == "univariate (all_results)"].set_index(["outcome", "feature", "regression_type"])
    for outcome in ("Primary Positive", "Phase 2 Progression"):
        for feat in ("tau_cell_type", "bimodality_score"):
            r = uni.loc[(outcome, feat, "logistic")]
            assert r["n_match"] and r["rel_delta"] < 1e-6, (outcome, feat, r.to_dict())
    beta = uni.xs("beta", level="regression_type")
    assert beta["n_match"].all() and (beta["abs_delta"] < 1e-4).all()
    assert uni["match"].all()
    h = rep.headline["ours"]
    assert h["or_primary_tau"] == pytest.approx(1.1158, abs=1e-4)
    assert h["or_phase1to2_tau"] == pytest.approx(1.2694, abs=1e-4)
    assert h["phase4_relative_increase"] == pytest.approx(0.4757, abs=1e-3)
    assert h["phase1to2_relative_increase"] == pytest.approx(0.3968, abs=1e-3)
    assert h["ae_relative_difference"] == pytest.approx(-0.3199, abs=1e-3)
    assert h["tau_threshold"] == pytest.approx(0.689, abs=1e-3)
    assert c[c.table.str.startswith("binary")]["match"].all()
    assert (tmp_path / "case1_replication_comparison.csv").exists()
    assert "Headline numbers" in (tmp_path / "case1_replication_summary.md").read_text()


def test_archive_permutation_identical_to_authors(tmp_path):
    root = _archive_root()
    inputs = R.load_inputs(root, open_targets=False)
    trials = R.build_trials(inputs)
    ref = R.load_reference_tables(inputs["root"])["permutation"]
    # The first analysis (Phase II+, bimodality then tau) consumes the RNG first: with the
    # authors' seed and 1000 permutations the null draws are the same as theirs.
    sub = [a for a in S.AUTHORS_ANALYSES if a[1] == "Phase II+"]
    import vbt.case_studies.trial_outcomes.replicate as mod

    orig = S.AUTHORS_ANALYSES
    try:
        mod.st.AUTHORS_ANALYSES = tuple(sub)
        perm = R.run_permutation(trials, n_perm=1000, n_perm_beta=0, features=("bimodality_score",))
    finally:
        mod.st.AUTHORS_ANALYSES = orig
    r = ref[(ref.outcome == "Phase II+") & (ref.feature == "bimodality_score")].iloc[0]
    ours = perm[perm.regression_type == "logistic"].iloc[0]
    assert ours["null_mean"] == pytest.approx(r["null_mean"], rel=1e-6, abs=1e-9)
    assert ours["null_std"] == pytest.approx(r["null_std"], rel=1e-6)
    assert ours["empirical_p_value"] == pytest.approx(r["empirical_p_value"])


def test_independent_columns_and_aliased_rows():
    x = np.arange(6.0)
    X = np.column_stack([np.ones(6), x, [0, 1, 1, 0, 0, 0], [1, 0, 0, 1, 1, 1], np.zeros(6)])
    # col 3 = intercept - col 2 (aliased, as R's phase dummies when one phase is absent); col 4 constant 0
    assert S.independent_columns(X) == [0, 1, 2]
    theirs = pd.DataFrame({"k": ["a", "b"], "coefficient": [0.1, np.nan], "se": [0.5, np.nan],
                           "n_reference": [10, 10]})
    ours = pd.DataFrame({"k": ["a", "b"], "coefficient": [0.1 + 0.02, np.nan], "n_reference": [10, 10]})
    c = R.compare_tables(ours, theirs, table="t", keys=["k"], estimate="coefficient", ci=None, p=None,
                         n_cols=("n_reference",), engine_col=None, se_col="se", se_tol=0.05).set_index("k")
    assert c.loc["a", "match"]  # 0.02 <= 0.05 * 0.5
    assert c.loc["b", "match"] and c.loc["b", "n_match"]  # both NA (aliased)


def test_archive_glmm_matches_lme4():
    root = _archive_root()
    inputs = R.load_inputs(root)
    if not {"drug_types", "disease"} <= set(inputs):
        pytest.skip("Open Targets drug_molecule/disease subsets not present")
    t = R.mixed_dataset(inputs, R.build_trials(inputs))
    sub = t[t["phase"].isin([2.0, 3.0])]
    y = S.authors_outcome(sub, "endpoint", "primary").to_numpy(float)
    cov = S.mixed_covariates("endpoint", sub)
    X = np.column_stack([np.ones(len(sub)), R._z(sub["tau_cell_type"].to_numpy()), *cov.values()])
    fit = S.glmm_laplace(y, X, [sub["drugType_combo"].to_numpy(), sub["ta_combo"].to_numpy()])
    # authors (lme4::glmer): 0.076507 (SE 0.013454); year_z -0.119945; phase3 0.420019
    assert len(sub) == 19413 + 17157
    assert fit["params"][1] == pytest.approx(0.076507, abs=2e-5)
    assert fit["se"][1] == pytest.approx(0.013454, rel=1e-3)
    assert fit["params"][3] == pytest.approx(0.420019, abs=1e-4)
