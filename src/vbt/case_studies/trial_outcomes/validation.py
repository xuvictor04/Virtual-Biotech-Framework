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


def _positive(s: pd.Series) -> pd.Series:
    return s.map(lambda v: np.nan if pd.isna(v) else float(v == "POSITIVE"))


def endpoint_agreement(pred: pd.DataFrame, ref: pd.DataFrame, col: str) -> dict:
    j = pred[["nct_id", col]].merge(ref[["nct_id", col]], on="nct_id", suffixes=("_pred", "_ref"))
    a, b = _positive(j[f"{col}_pred"]), _positive(j[f"{col}_ref"])
    mask = a.notna() & b.notna()
    n = int(mask.sum())
    agree = int((a[mask] == b[mask]).sum())
    return {"field": col, "n_applicable": n, "n_agree": agree, "agreement": round(agree / n, 4) if n else None}


def ae_agreement(pred: pd.DataFrame, ref: pd.DataFrame, tol_pct_points: float = 1.0) -> dict:
    cols = [c for c in ref.columns if c.startswith(AE_COLS_PREFIX) and c in pred.columns]
    j = pred[["nct_id", *cols]].merge(ref[["nct_id", *cols]], on="nct_id", suffixes=("_pred", "_ref"))
    n = agree = 0
    for _, r in j.iterrows():
        ref_vals = {c: r[f"{c}_ref"] for c in cols if pd.notna(r[f"{c}_ref"])}
        if not ref_vals:  # reference reports no exact statistics for this trial
            continue
        n += 1
        ok = all(pd.notna(r[f"{c}_pred"]) and abs(r[f"{c}_pred"] - v) <= tol_pct_points for c, v in ref_vals.items())
        agree += ok
    return {"field": "serious_ae_rates", "n_applicable": n, "n_agree": agree,
            "agreement": round(agree / n, 4) if n else None, "tolerance_pct_points": tol_pct_points}


def agreement_report(pred: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    rows = [endpoint_agreement(pred, ref, c) for c in ("primary_endpoint_result", "secondary_endpoint_result")
            if c in pred.columns and c in ref.columns]
    if any(c.startswith(AE_COLS_PREFIX) for c in ref.columns):
        rows.append(ae_agreement(pred, ref))
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
