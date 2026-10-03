"""Annotation quality: agreement with manual review, TDC, or the released labels.

Paper: 88.4% (primary), 88.4% (secondary), 92.4% (AE rates, exact statistics)
agreement with manual review of 100 trials; 85.6% primary-endpoint agreement
with the TDC Trial Outcome Prediction labels on 7,666 overlapping trials.
Endpoints are compared as a binary "sufficient evidence of a positive
endpoint" vs not, restricted to trials where the endpoint is applicable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

AE_COLS_PREFIX = "ae_serious_"
STOPPED_STATUSES = ("Terminated", "Withdrawn", "Suspended")
NOT_APPLICABLE = {"NOT_APPLICABLE", "NOT APPLICABLE", "N/A", "NA", "NONE", ""}


def _positive(s: pd.Series) -> pd.Series:
    return s.map(lambda v: np.nan if pd.isna(v) else float(v == "POSITIVE"))


def _not_applicable(s: pd.Series) -> pd.Series:
    return s.map(lambda v: pd.isna(v) or str(v).strip().upper() in NOT_APPLICABLE)


def _status(j: pd.DataFrame) -> pd.Series:
    """Registry status per joined row (reference first, then prediction)."""
    st = pd.Series(np.nan, index=j.index, dtype=object)
    for c in ("status_ref", "status", "status_pred", "overall_status_pred", "overall_status"):
        if c in j.columns:
            st = st.where(st.notna(), j[c])
    return st


def endpoint_agreement(pred: pd.DataFrame, ref: pd.DataFrame, col: str, *, exclude_stopped: bool = True,
                       stopped_statuses=STOPPED_STATUSES) -> dict:
    """Agreement on "sufficient evidence of a positive endpoint", over applicable trials.

    Applicability (paper Methods): the reference must carry an applicable label
    (missing / NOT_APPLICABLE rows are excluded), the prediction must exist, and
    — with ``exclude_stopped`` (default) — trials that stopped early
    (Terminated / Withdrawn / Suspended) are excluded for endpoints.
    """
    keep = ["nct_id", col]
    extra_ref = [c for c in ("status",) if c in ref.columns]
    extra_pred = [c for c in ("status", "overall_status") if c in pred.columns]
    j = pred[[*keep, *extra_pred]].merge(ref[[*keep, *extra_ref]], on="nct_id", suffixes=("_pred", "_ref"))
    if "status" in extra_pred and "status" not in extra_ref:
        j = j.rename(columns={"status": "status_pred"})
    elif "status" in extra_ref and "status" not in extra_pred:
        j = j.rename(columns={"status": "status_ref"})
    ref_na = _not_applicable(j[f"{col}_ref"])
    pred_missing = j[f"{col}_pred"].isna()
    stopped = _status(j).astype("string").str.strip().str.lower().isin(
        {s.lower() for s in stopped_statuses}).fillna(False).astype(bool)
    if not exclude_stopped:
        stopped = pd.Series(False, index=j.index)
    mask = ~ref_na & ~pred_missing & ~stopped
    a, b = _positive(j.loc[mask, f"{col}_pred"]), _positive(j.loc[mask, f"{col}_ref"])
    n = int(mask.sum())
    agree = int((a == b).sum())
    return {"field": col, "n_applicable": n, "n_agree": agree, "agreement": round(agree / n, 4) if n else None,
            "n_compared": int(len(j)), "n_excluded_stopped": int((stopped & ~ref_na).sum()),
            "n_excluded_not_applicable": int(ref_na.sum()), "n_excluded_missing_prediction": int(
                (pred_missing & ~ref_na & ~stopped).sum())}


def ae_agreement(pred: pd.DataFrame, ref: pd.DataFrame, tol_pct_points: float = 1.0) -> dict:
    """Serious-AE agreement over trials whose reference reports exact statistics.

    A trial is applicable only when the reference has at least one numeric
    ``ae_serious_*`` value (no exact statistics -> excluded); every reference
    value must then be matched within ``tol_pct_points``.
    """
    cols = [c for c in ref.columns if c.startswith(AE_COLS_PREFIX) and c in pred.columns]
    j = pred[["nct_id", *cols]].merge(ref[["nct_id", *cols]], on="nct_id", suffixes=("_pred", "_ref"))
    n = agree = no_stats = 0
    for _, r in j.iterrows():
        ref_vals = {c: r[f"{c}_ref"] for c in cols if pd.notna(r[f"{c}_ref"])}
        if not ref_vals:  # reference reports no exact statistics for this trial
            no_stats += 1
            continue
        n += 1
        ok = all(pd.notna(r[f"{c}_pred"]) and abs(r[f"{c}_pred"] - v) <= tol_pct_points for c, v in ref_vals.items())
        agree += ok
    return {"field": "serious_ae_rates", "n_applicable": n, "n_agree": agree,
            "agreement": round(agree / n, 4) if n else None, "tolerance_pct_points": tol_pct_points,
            "n_compared": int(len(j)), "n_excluded_no_exact_statistics": no_stats}


def agreement_report(pred: pd.DataFrame, ref: pd.DataFrame, *, exclude_stopped: bool = True,
                     tol_pct_points: float = 1.0) -> pd.DataFrame:
    """Per-field agreement with ``n_applicable`` denominators (paper: 88.4% = 76/86 primary)."""
    rows = [endpoint_agreement(pred, ref, c, exclude_stopped=exclude_stopped)
            for c in ("primary_endpoint_result", "secondary_endpoint_result")
            if c in pred.columns and c in ref.columns]
    if any(c.startswith(AE_COLS_PREFIX) for c in ref.columns):
        rows.append(ae_agreement(pred, ref, tol_pct_points))
    return pd.DataFrame(rows)


def load_tdc(path: str | Path) -> pd.DataFrame:
    """TDC / HINT trial-outcome CSV (columns nctid, label) -> primary_endpoint_result."""
    df = pd.read_csv(path)
    id_col = next(c for c in df.columns if c.lower() in ("nctid", "nct_id"))
    return pd.DataFrame({"nct_id": df[id_col],
                         "primary_endpoint_result": df["label"].map({1: "POSITIVE", 0: "NEGATIVE"})})


def manual_review_sample(labels: pd.DataFrame, n_per_phase: int = 50, seed: int = 0) -> pd.DataFrame:
    """The paper's manual-review design: 50 random Phase II + 50 random Phase III trials."""
    parts = [labels[labels["phase"] == p].sample(n=min(n_per_phase, int((labels["phase"] == p).sum())),
                                                  random_state=seed) for p in (2.0, 3.0)]
    return pd.concat(parts)[["nct_id", "phase", "status"]]
