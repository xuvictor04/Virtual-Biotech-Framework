"""Table S2: clinical-trial annotation by competitor agentic systems.

The Zenodo archive ships, for Biomni, Kosmos and PantheonOS,
``clinical_trials/benchmarks/<system>/manual_review_set_annotations.csv``
(the 100-trial manual-review set: 50 Phase II + 50 Phase III; columns
``nct_id, primary, secondary, ae_binary``) and ``tdc_annotations.csv`` (500
trials overlapping the TDC Trial Outcome Prediction labels; ``nct_id,
primary``).

What the archive does **not** ship: the two human annotators' manual-review
labels and the TDC/HINT labels themselves. So, by default,
:func:`benchmark_agreement` scores every system against the **Virtual
Biotech's reconciled labels** on the same trial sets (and says so in the
``reference`` column); pass ``manual=`` / ``tdc=`` files to score every
system — the Virtual Biotech included — against the real ground truth.

Applicability follows the paper (Methods, "annotation quality"):

* endpoints are binarised as "sufficient evidence of a positive endpoint"
  (POSITIVE) vs not (NEGATIVE, UNKNOWN, MIXED, missing);
* primary/secondary endpoints are not counted for trials stopped early
  (Terminated / Withdrawn / Suspended): on the archive's manual-review set
  this leaves 86 applicable trials, the paper's denominator (76/86);
* the reference label must exist (for VB-referenced AE agreement:
  ``ae_has_safety_signals`` must be non-missing).

``agreement`` counts a missing system answer as "not positive" (the system
found no evidence of success); ``agreement_answered`` restricts to trials the
system answered and ``coverage`` is the answered fraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SYSTEMS = ("Biomni", "Kosmos", "PantheonOS")
STOPPED = ("Terminated", "Withdrawn", "Suspended")
VB = "Virtual Biotech"

__all__ = ["SYSTEMS", "load_benchmarks", "benchmark_agreement", "format_table_s2", "binarize_positive"]


def binarize_positive(v) -> float:
    """POSITIVE -> 1, any other non-missing label -> 0, missing -> NaN."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and not v.strip()):
        return np.nan
    if isinstance(v, (bool, np.bool_)):
        return float(v)
    s = str(v).strip().upper()
    if s in ("TRUE", "1", "1.0"):
        return 1.0
    if s in ("FALSE", "0", "0.0"):
        return 0.0
    return float(s == "POSITIVE")


def _ct(root: str | Path) -> Path:
    root = Path(root)
    return root if (root / "benchmarks").exists() else root / "clinical_trials"


def _read_truth(path: str | Path, kind: str) -> pd.DataFrame:
    """Ground-truth file -> nct_id + primary/secondary/ae_binary columns.

    Accepts the benchmark column names, the released-label names
    (``primary_endpoint_result`` ...) or TDC/HINT (``nctid``, ``label`` 0/1).
    """
    path = Path(path)
    df = pd.read_csv(path, sep="\t" if path.suffix in (".tsv", ".txt") else ",")
    cols = {c.lower(): c for c in df.columns}
    idc = cols.get("nct_id") or cols.get("nctid")
    if idc is None:
        raise ValueError(f"{path}: no nct_id / nctid column")
    out = pd.DataFrame({"nct_id": df[idc].astype(str)})
    for name, alts in (("primary", ("primary", "primary_endpoint_result", "label")),
                       ("secondary", ("secondary", "secondary_endpoint_result")),
                       ("ae_binary", ("ae_binary", "ae_has_safety_signals"))):
        src = next((cols[a] for a in alts if a in cols), None)
        if src is not None:
            col = df[src]
            if src.lower() == "label":
                col = col.map({1: "POSITIVE", 0: "NEGATIVE", "1": "POSITIVE", "0": "NEGATIVE"})
            out[name] = col.to_numpy()
    out.attrs["source"] = str(path)
    out.attrs["kind"] = kind
    return out


def load_benchmarks(root: str | Path, *, labels: str | Path | None = None, manual: str | Path | None = None,
                    tdc: str | Path | None = None) -> dict[str, Any]:
    """Competitor annotations, the Virtual Biotech labels and optional ground truth."""
    ct = _ct(root)
    systems: dict[str, dict[str, pd.DataFrame]] = {}
    for s in SYSTEMS:
        d = ct / "benchmarks" / s
        if not d.exists():
            continue
        systems[s] = {}
        for key, fn in (("manual_review", "manual_review_set_annotations.csv"), ("tdc", "tdc_annotations.csv")):
            if (d / fn).exists():
                systems[s][key] = pd.read_csv(d / fn, dtype={"nct_id": str})
    lab_path = Path(labels) if labels else ct / "data" / "clinical_trial_labels_reconciled.csv"
    vb = pd.read_csv(lab_path)
    vbf = pd.DataFrame({"nct_id": vb["nct_id"].astype(str),
                        "primary": vb.get("primary_endpoint_result"),
                        "secondary": vb.get("secondary_endpoint_result"),
                        "ae_binary": vb.get("ae_has_safety_signals"),
                        "status": vb.get("status"), "phase": vb.get("phase")})
    return dict(systems=systems, vb=vbf, labels_source=str(lab_path),
                manual=_read_truth(manual, "manual") if manual else None,
                tdc=_read_truth(tdc, "tdc") if tdc else None)


def _agree(pred: pd.DataFrame | None, ref: pd.DataFrame, ids: list[str], field: str, status: pd.Series,
           exclude_stopped: bool) -> dict:
    r = ref.set_index("nct_id").reindex(ids)
    if field not in r:
        return {}
    rb = r[field].map(binarize_positive)
    applicable = rb.notna()
    if exclude_stopped and field in ("primary", "secondary"):
        applicable &= ~status.reindex(ids).isin(STOPPED).to_numpy()
    if pred is None or field not in pred:
        return {}
    p = pred.drop_duplicates("nct_id").set_index("nct_id").reindex(ids)[field].map(binarize_positive)
    a = applicable.to_numpy()
    pv, rv = p.to_numpy()[a], rb.to_numpy()[a]
    answered = ~np.isnan(pv)
    n = int(a.sum())
    agree = int(np.sum(np.nan_to_num(pv, nan=0.0) == rv))
    agree_ans = int(np.sum(pv[answered] == rv[answered]))
    return dict(n_applicable=n, n_agree=agree, agreement=agree / n if n else np.nan,
                n_answered=int(answered.sum()), agreement_answered=agree_ans / answered.sum() if answered.any()
                else np.nan, coverage=answered.mean() if n else np.nan,
                reference_positive_rate=float(np.mean(rv)) if n else np.nan)


def benchmark_agreement(data: dict[str, Any], *, exclude_stopped: bool = True) -> pd.DataFrame:
    """Table-S2-style agreement per system × trial set × field × reference."""
    vb = data["vb"]
    status = vb.set_index("nct_id")["status"]
    rows = []
    for set_name, fields in (("manual_review", ("primary", "secondary", "ae_binary")), ("tdc", ("primary",))):
        ids = sorted({i for s in data["systems"].values() if set_name in s for i in s[set_name]["nct_id"]})
        if not ids:
            continue
        truth = data.get("manual" if set_name == "manual_review" else "tdc")
        refs = [("virtual_biotech_labels", vb)]
        if truth is not None:
            refs.insert(0, ("manual_review" if set_name == "manual_review" else "tdc", truth))
        for ref_name, ref in refs:
            systems = list(data["systems"].items())
            if ref_name != "virtual_biotech_labels":
                systems.append((VB, {set_name: vb}))
            for sys_name, ann in systems:
                pred = ann.get(set_name)
                for f in fields:
                    r = _agree(pred, ref, ids, f, status, exclude_stopped)
                    if r:
                        rows.append(dict(system=sys_name, trial_set=set_name, field=f, reference=ref_name,
                                         n_trials=len(ids), **r))
    return pd.DataFrame(rows)


def format_table_s2(table: pd.DataFrame, data: dict[str, Any] | None = None) -> str:
    """Markdown rendering with the reference made explicit."""
    if table.empty:
        return "(no benchmark annotations found)"
    lines = []
    refs = sorted(table["reference"].unique())
    if refs == ["virtual_biotech_labels"]:
        lines += ["NOTE: the archive ships neither the manual-review labels nor the TDC labels; agreement is "
                  "computed against the Virtual Biotech's reconciled labels on the same trials (pass --manual / "
                  "--tdc for ground truth).", ""]
    lines += ["| trial set | reference | field | system | agreement | n agree / applicable | coverage | "
              "agreement (answered only) |", "|---|---|---|---|---|---|---|---|"]
    for _, r in table.sort_values(["trial_set", "reference", "field", "system"]).iterrows():
        lines.append(f"| {r['trial_set']} | {r['reference']} | {r['field']} | {r['system']} | "
                     f"{r['agreement']:.1%} | {r['n_agree']}/{r['n_applicable']} | {r['coverage']:.0%} | "
                     f"{r['agreement_answered']:.1%} |")
    return "\n".join(lines)
