"""Table S2 benchmark agreement on synthetic frames (+ the archive when present)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vbt.case_studies.trial_outcomes import benchmarks as B


def _archive(tmp_path):
    ct = tmp_path / "clinical_trials"
    ids = [f"NCT{i:08d}" for i in range(10)]
    status = ["Completed"] * 8 + ["Terminated", "Withdrawn"]
    vb_primary = ["POSITIVE"] * 5 + ["NEGATIVE"] * 2 + ["UNKNOWN"] + ["POSITIVE", "NEGATIVE"]
    (ct / "data").mkdir(parents=True)
    pd.DataFrame({"nct_id": ids, "phase": [2.0] * 5 + [3.0] * 5, "status": status,
                  "primary_endpoint_result": vb_primary,
                  "secondary_endpoint_result": vb_primary,
                  "ae_has_safety_signals": [True, False] * 4 + [np.nan, True]}) \
        .to_csv(ct / "data" / "clinical_trial_labels_reconciled.csv", index=False)
    # system A: perfect on positives, answers UNKNOWN for one negative; system B misses half
    a = pd.DataFrame({"nct_id": ids, "primary": vb_primary[:5] + ["UNKNOWN", "NEGATIVE", "NEGATIVE",
                                                                   "NEGATIVE", "NEGATIVE"],
                      "secondary": vb_primary, "ae_binary": [True, False] * 5})
    b = pd.DataFrame({"nct_id": ids[:5], "primary": ["POSITIVE"] * 5, "secondary": [None] * 5,
                      "ae_binary": [False] * 5})
    for name, df in (("Biomni", a), ("Kosmos", b)):
        d = ct / "benchmarks" / name
        d.mkdir(parents=True)
        df.to_csv(d / "manual_review_set_annotations.csv", index=False)
        df[["nct_id", "primary"]].to_csv(d / "tdc_annotations.csv", index=False)
    return tmp_path, ids


def test_binarize_positive():
    assert B.binarize_positive("POSITIVE") == 1.0
    assert B.binarize_positive("unknown") == 0.0 and B.binarize_positive("MIXED") == 0.0
    assert np.isnan(B.binarize_positive(None)) and np.isnan(B.binarize_positive(float("nan")))
    assert B.binarize_positive(True) == 1.0 and B.binarize_positive("False") == 0.0


def test_agreement_vs_vb_labels(tmp_path):
    root, ids = _archive(tmp_path)
    data = B.load_benchmarks(root)
    t = B.benchmark_agreement(data).set_index(["system", "trial_set", "field"])
    assert set(t["reference"]) == {"virtual_biotech_labels"}
    # stopped trials (2) excluded for endpoints -> 8 applicable; UNKNOWN == not positive
    r = t.loc[("Biomni", "manual_review", "primary")]
    assert r["n_applicable"] == 8 and r["n_agree"] == 8 and r["agreement"] == 1.0
    k = t.loc[("Kosmos", "manual_review", "primary")]
    # Kosmos answered 5 positives (all VB-positive) and nothing else: missing = not positive
    assert k["n_answered"] == 5 and k["coverage"] == pytest.approx(5 / 8)
    assert k["n_agree"] == 8 and k["agreement_answered"] == 1.0
    ks = t.loc[("Kosmos", "manual_review", "secondary")]
    assert ks["n_answered"] == 0 and ks["n_agree"] == 3  # 3 VB-not-positive among applicable
    # AE: VB reference missing for one trial -> 9 applicable, stopped trials kept
    ae = t.loc[("Biomni", "manual_review", "ae_binary")]
    assert ae["n_applicable"] == 9 and ae["n_agree"] == 8
    assert ("Biomni", "tdc", "primary") in t.index
    md = B.format_table_s2(B.benchmark_agreement(data), data)
    assert "NOTE" in md and "Biomni" in md


def test_agreement_with_ground_truth(tmp_path):
    root, ids = _archive(tmp_path)
    truth = pd.DataFrame({"nctid": ids, "label": [1] * 4 + [0] * 6})
    truth.to_csv(tmp_path / "tdc.csv", index=False)
    manual = pd.DataFrame({"nct_id": ids, "primary": ["POSITIVE"] * 10, "secondary": ["NEGATIVE"] * 10,
                           "ae_binary": [False] * 10})
    manual.to_csv(tmp_path / "manual.csv", index=False)
    data = B.load_benchmarks(root, tdc=tmp_path / "tdc.csv", manual=tmp_path / "manual.csv")
    t = B.benchmark_agreement(data)
    refs = set(t["reference"])
    assert {"tdc", "manual_review", "virtual_biotech_labels"} <= refs
    vb = t[(t.system == B.VB) & (t.reference == "tdc")].iloc[0]
    # VB positive on ids 0-4 and 8; truth positive on 0-3 -> disagreements on id 4 and 8 (all completed? 8 is
    # Terminated but TDC rows are filtered by stopped status too)
    assert vb["n_applicable"] == 8 and vb["n_agree"] == 7
    assert not ((t.system == B.VB) & (t.reference == "virtual_biotech_labels")).any()


def test_archive_benchmarks_if_present():
    from vbt.data.zenodo import zenodo_root

    root = zenodo_root()
    if not (root / "clinical_trials" / "benchmarks").exists():
        pytest.skip("Zenodo archive extract not present")
    t = B.benchmark_agreement(B.load_benchmarks(root))
    prim = t[(t.trial_set == "manual_review") & (t.field == "primary")]
    assert set(prim["n_applicable"]) == {86}  # the paper's 76/86 denominator
    assert set(t[(t.trial_set == "tdc")]["n_trials"]) == {500}
