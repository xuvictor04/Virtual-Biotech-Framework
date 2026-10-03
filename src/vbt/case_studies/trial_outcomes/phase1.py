"""Algorithmic Phase I success = progression to Phase II (paper Methods).

"Phase I trials were classified as having progressed to Phase II if there
existed at least one Phase II trial which tested the same set of drugs for the
same indications, with a subsequent or concurrent trial start date."
"""

from __future__ import annotations

import pandas as pd


def phase1_progression(mapping: pd.DataFrame, *, indication_match: str = "all",
                       statuses: tuple[str, ...] | None = ("Completed",),
                       universe: set[str] | None = None) -> pd.DataFrame:
    """Return nct_id -> phase2_progression ('EVER' / 'NEVER') for Phase I trials.

    indication_match: 'all' (identical indication sets, default) or 'any' (share at
    least one indication). Drug sets must be identical in both modes. On the
    released dataset, 'all' reproduces the published Phase I labels for 99.8% of
    11,412 trials ('any': 96.2%).

    statuses: registry statuses of the Phase I trials labelled (default
    ``('Completed',)``: the paper's 11,412 completed Phase I trials; recruiting,
    active or unknown trials are right-censored and almost all NEVER, and
    terminated ones did not finish). ``None`` keeps every status (18,304 trials).
    universe: optional explicit set of NCT IDs to label (e.g. the released
    Phase I labels); applied after the status filter.
    """
    m = mapping.dropna(subset=["drugId", "diseaseId"]).copy()
    m["trial_date"] = pd.to_datetime(m["trial_date"], errors="coerce")
    agg = dict(phase=("phase", "max"), date=("trial_date", "min"),
               drugs=("drugId", lambda s: frozenset(s)), diseases=("diseaseId", lambda s: frozenset(s)))
    if "status" in m.columns:
        agg["status"] = ("status", "first")
    per = m.groupby("nct_id").agg(**agg).reset_index()
    p1 = per[per["phase"] == 1.0]
    if statuses is not None and "status" in p1.columns:
        p1 = p1[p1["status"].isin(set(statuses))]
    if universe is not None:
        p1 = p1[p1["nct_id"].isin(set(universe))]
    p2 = per[per["phase"] == 2.0]
    by_drugs: dict[frozenset, list[tuple]] = {}
    for r in p2.itertuples(index=False):
        by_drugs.setdefault(r.drugs, []).append((r.date, r.diseases))

    out = []
    for r in p1.itertuples(index=False):
        ever = False
        for date, diseases in by_drugs.get(r.drugs, []):
            if pd.notna(r.date) and pd.notna(date) and date < r.date:
                continue
            if (indication_match == "all" and diseases == r.diseases) or \
               (indication_match == "any" and diseases & r.diseases):
                ever = True
                break
        out.append({"nct_id": r.nct_id, "phase2_progression": "EVER" if ever else "NEVER"})
    return pd.DataFrame(out)
