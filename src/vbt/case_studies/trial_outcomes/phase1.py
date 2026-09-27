"""Algorithmic Phase I success = progression to Phase II (paper Methods).

"Phase I trials were classified as having progressed to Phase II if there
existed at least one Phase II trial which tested the same set of drugs for the
same indications, with a subsequent or concurrent trial start date."
"""

from __future__ import annotations

import pandas as pd


def phase1_progression(mapping: pd.DataFrame, *, indication_match: str = "all") -> pd.DataFrame:
    """Return nct_id -> phase2_progression ('EVER' / 'NEVER') for Phase I trials.

    indication_match: 'all' (identical indication sets, default) or 'any' (share at
    least one indication). Drug sets must be identical in both modes. On the
    released dataset, 'all' reproduces the published Phase I labels for 99.8% of
    11,412 trials ('any': 96.2%).
    """
    m = mapping.dropna(subset=["drugId", "diseaseId"]).copy()
    m["trial_date"] = pd.to_datetime(m["trial_date"], errors="coerce")
    per = (m.groupby("nct_id")
             .agg(phase=("phase", "max"), date=("trial_date", "min"),
                  drugs=("drugId", lambda s: frozenset(s)), diseases=("diseaseId", lambda s: frozenset(s)))
             .reset_index())
    p1 = per[per["phase"] == 1.0]
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
