"""Survival analysis from case study 2 (CD276 / B7-H3 expression and outcome in
TCGA lung adenocarcinoma).

Implements the Supplementary Methods section "Survival analysis": TCGA
PanCancer Atlas RNA-seq (RSEM, log2(x+1)) and clinical data from cBioPortal;
patients restricted to complete covariates (age, AJCC stage dichotomised as
advanced III/IV vs early I/II, sex); top vs bottom expression quartile
compared with a multivariable Cox proportional-hazards model for OS, PFS, DSS
and DFS endpoints.

The paper stratifies one complete-case cohort (477 LUAD patients) once and
fits every endpoint on those fixed high/low groups:
:func:`quartile_cox_all_endpoints` (``quartile_scope='cohort'``, default).
``quartile_scope='endpoint'`` recomputes quartiles on each endpoint's own
complete cases (high/low membership then differs between OS and DFS).
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd

from ._utils import has_module, require

__all__ = [
    "ENDPOINTS",
    "stage_to_advanced",
    "parse_status",
    "prepare_tcga_clinical",
    "quartile_cox",
    "quartile_cox_all_endpoints",
    "fetch_cbioportal_expression_and_clinical",
]

#: cBioPortal PanCancer Atlas endpoint attribute stems.
ENDPOINTS = ("OS", "PFS", "DSS", "DFS")

_STAGE_RE = re.compile(r"(?:STAGE\s*)?\b(IV|III|II|I)[ABC]?\d?\b", re.IGNORECASE)


def stage_to_advanced(stage) -> float:
    """Map an AJCC stage string to 1.0 (III/IV, advanced), 0.0 (I/II, early) or
    NaN (missing / 'Stage X' / unparseable). E.g. 'STAGE IIIA' -> 1, 'Stage IB' -> 0."""
    if stage is None or (isinstance(stage, float) and np.isnan(stage)):
        return np.nan
    s = str(stage).strip().upper()
    if not s or s in {"NA", "NAN", "[NOT AVAILABLE]", "[DISCREPANCY]", "STAGE X"}:
        return np.nan
    m = _STAGE_RE.search(s)
    if not m:
        return np.nan
    return 1.0 if m.group(1).upper() in {"III", "IV"} else 0.0


def parse_status(status) -> float:
    """Parse a cBioPortal status such as '1:DECEASED', '0:LIVING',
    '1:Recurred/Progressed', '0:DiseaseFree', '1:PROGRESSION', '0:CENSORED',
    '1:DEAD WITH TUMOR' to 1.0 / 0.0 (NaN if missing)."""
    if status is None or (isinstance(status, float) and np.isnan(status)):
        return np.nan
    s = str(status).strip()
    m = re.match(r"^([01])\s*:", s)
    if m:
        return float(m.group(1))
    u = s.upper()
    if u in {"DECEASED", "DEAD", "RECURRED/PROGRESSED", "PROGRESSION", "DEAD WITH TUMOR"}:
        return 1.0
    if u in {"LIVING", "ALIVE", "DISEASEFREE", "DISEASE FREE", "CENSORED",
             "ALIVE OR DEAD TUMOR FREE"}:
        return 0.0
    return np.nan


def prepare_tcga_clinical(df: pd.DataFrame, stage_col: str | None = None,
                          sex_col: str = "SEX", age_col: str | None = None) -> pd.DataFrame:
    """Derive analysis columns from cBioPortal TCGA clinical attributes.

    Adds ``age`` (float), ``sex`` (1 = male, 0 = female), ``stage_advanced``
    (1 = AJCC III/IV, 0 = I/II; see :func:`stage_to_advanced`) and for each
    endpoint in :data:`ENDPOINTS` present as ``<EP>_MONTHS`` / ``<EP>_STATUS``
    the columns ``<ep>_time`` (months) and ``<ep>_event`` (1 = event). Original
    columns are kept. Methods, "Survival analysis".
    """
    out = df.copy()
    if stage_col is None:
        for cand in ("AJCC_PATHOLOGIC_TUMOR_STAGE", "PATH_STAGE", "TUMOR_STAGE", "STAGE", "stage"):
            if cand in out.columns:
                stage_col = cand
                break
    if age_col is None:
        for cand in ("AGE", "age", "AGE_AT_DIAGNOSIS"):
            if cand in out.columns:
                age_col = cand
                break
    out["stage_advanced"] = out[stage_col].map(stage_to_advanced) if stage_col else np.nan
    out["age"] = pd.to_numeric(out[age_col], errors="coerce") if age_col else np.nan
    if sex_col in out.columns:
        sx = out[sex_col].astype(str).str.strip().str.upper()
        out["sex"] = np.where(sx.isin(["MALE", "M"]), 1.0, np.where(sx.isin(["FEMALE", "F"]), 0.0,
                                                                     np.nan))
    for ep in ENDPOINTS:
        mcol, scol = f"{ep}_MONTHS", f"{ep}_STATUS"
        if mcol in out.columns and scol in out.columns:
            out[f"{ep.lower()}_time"] = pd.to_numeric(out[mcol], errors="coerce")
            out[f"{ep.lower()}_event"] = out[scol].map(parse_status)
    return out


def _cox_phreg(dat: pd.DataFrame, time_col, event_col, xcols):
    from statsmodels.duration.hazard_regression import PHReg

    model = PHReg(dat[time_col].to_numpy(float), dat[xcols].to_numpy(float),
                  status=dat[event_col].to_numpy(float), ties="efron")
    res = model.fit()
    return float(res.params[0]), float(res.bse[0]), float(res.pvalues[0])


def _cox_lifelines(dat: pd.DataFrame, time_col, event_col, xcols):
    lifelines = require("lifelines", "survival")
    cph = lifelines.CoxPHFitter()
    cph.fit(dat[[time_col, event_col] + xcols], duration_col=time_col, event_col=event_col)
    row = cph.summary.loc["high"]
    return float(row["coef"]), float(row["se(coef)"]), float(row["p"])


def quartile_cox(df: pd.DataFrame, expr_col: str, time_col: str, event_col: str,
                 covariates: Sequence[str] = ("age", "stage_advanced", "sex"),
                 top: float = 0.75, bottom: float = 0.25, engine: str = "auto") -> dict:
    """Multivariable Cox PH model, top vs bottom expression quartile (Methods,
    "Survival analysis").

    Rows are restricted to complete ``covariates``, time, event and expression;
    quartiles are computed in that complete-case set, the middle 50% dropped,
    and ``event ~ high + covariates`` fitted. ``engine``: ``"lifelines"``,
    ``"statsmodels"`` (``PHReg``, Efron ties) or ``"auto"`` (lifelines if
    installed). Returns dict with ``hr, ci_low, ci_high, p, coef, se, n,
    n_events, n_high, n_low, engine``.
    """
    covariates = list(covariates)
    cols = [expr_col, time_col, event_col] + covariates
    dat = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    dat = dat[dat[time_col] > 0] if (dat[time_col] <= 0).any() else dat
    q_hi, q_lo = dat[expr_col].quantile(top), dat[expr_col].quantile(bottom)
    hi = dat[expr_col] >= q_hi
    lo = (dat[expr_col] <= q_lo) & ~hi
    dat = dat[hi | lo].copy()
    dat["high"] = (dat[expr_col] >= q_hi).astype(float)
    # covariates constant in the analysed set are dropped (would make the model singular)
    return _fit_groups(dat, time_col, event_col, covariates, engine)


def _fit_groups(dat: pd.DataFrame, time_col: str, event_col: str, covariates: list[str], engine: str) -> dict:
    """Cox fit of ``event ~ high + covariates`` on rows already labelled ``high`` (1/0)."""
    xcols = ["high"] + [c for c in covariates if dat[c].nunique() > 1]
    if engine == "auto":
        engine = "lifelines" if has_module("lifelines") else "statsmodels"
    if engine == "lifelines":
        coef, se, p = _cox_lifelines(dat, time_col, event_col, xcols)
    elif engine == "statsmodels":
        coef, se, p = _cox_phreg(dat, time_col, event_col, xcols)
    else:
        raise ValueError(f"unknown engine {engine!r}")
    return {
        "hr": float(np.exp(coef)), "ci_low": float(np.exp(coef - 1.959964 * se)),
        "ci_high": float(np.exp(coef + 1.959964 * se)), "p": p, "coef": coef, "se": se,
        "n": int(len(dat)), "n_events": int(dat[event_col].sum()),
        "n_high": int(dat["high"].sum()), "n_low": int((1 - dat["high"]).sum()),
        "covariates": [c for c in xcols if c != "high"], "engine": engine,
    }


def quartile_cox_all_endpoints(df: pd.DataFrame, expr_col: str = "expr",
                               endpoints: Sequence[str] = ENDPOINTS,
                               covariates: Sequence[str] = ("age", "stage_advanced", "sex"),
                               top: float = 0.75, bottom: float = 0.25, engine: str = "auto",
                               quartile_scope: str = "cohort") -> pd.DataFrame:
    """Top- vs bottom-quartile Cox models for every endpoint on fixed groups.

    ``quartile_scope='cohort'`` (paper): restrict once to patients with
    complete ``covariates`` and expression (the cohort; paper: 477 LUAD
    patients), compute the quartiles once, and fit each endpoint
    (``<ep>_time`` / ``<ep>_event`` columns from :func:`prepare_tcga_clinical`)
    on the cohort's high/low patients that have that endpoint recorded.
    ``'endpoint'`` recomputes the quartiles on each endpoint's complete cases
    (as :func:`quartile_cox`). Returns one row per endpoint with ``endpoint,
    hr, ci_low, ci_high, p, coef, se, n, n_events, n_high, n_low, n_cohort,
    q_low, q_high, quartile_scope, engine`` (``error`` when a fit fails).
    """
    if quartile_scope not in ("cohort", "endpoint"):
        raise ValueError("quartile_scope must be 'cohort' or 'endpoint'")
    covariates = list(covariates)
    base = df.copy()
    for c in [expr_col, *covariates]:
        base[c] = pd.to_numeric(base[c], errors="coerce")
    cohort = base.dropna(subset=[expr_col, *covariates])
    q_hi, q_lo = cohort[expr_col].quantile(top), cohort[expr_col].quantile(bottom)
    rows = []
    for ep in endpoints:
        tcol, ecol = f"{ep.lower()}_time", f"{ep.lower()}_event"
        row: dict = {"endpoint": ep, "n_cohort": int(len(cohort)), "quartile_scope": quartile_scope}
        if tcol not in cohort.columns or ecol not in cohort.columns:
            rows.append({**row, "error": f"missing {tcol}/{ecol}"})
            continue
        try:
            if quartile_scope == "endpoint":
                r = quartile_cox(cohort, expr_col, tcol, ecol, covariates, top, bottom, engine)
                sub = cohort[[expr_col, tcol, ecol]].apply(pd.to_numeric, errors="coerce").dropna()
                r.update(q_high=float(sub[expr_col].quantile(top)), q_low=float(sub[expr_col].quantile(bottom)))
            else:
                d = cohort[[expr_col, tcol, ecol, *covariates]].apply(pd.to_numeric, errors="coerce")
                d = d.dropna(subset=[tcol, ecol])
                d = d[d[tcol] > 0] if (d[tcol] <= 0).any() else d
                hi = d[expr_col] >= q_hi
                lo = (d[expr_col] <= q_lo) & ~hi
                d = d[hi | lo].copy()
                d["high"] = (d[expr_col] >= q_hi).astype(float)
                r = _fit_groups(d, tcol, ecol, covariates, engine)
                r.update(q_high=float(q_hi), q_low=float(q_lo))
        except Exception as exc:  # noqa: BLE001 - report per endpoint
            rows.append({**row, "error": f"{type(exc).__name__}: {exc}"})
            continue
        r.pop("covariates", None)
        rows.append({**row, **r})
    cols = ["endpoint", "hr", "ci_low", "ci_high", "p", "coef", "se", "n", "n_events", "n_high", "n_low",
            "n_cohort", "q_low", "q_high", "quartile_scope", "engine", "error"]
    out = pd.DataFrame(rows)
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]


_ENTREZ = {"CD276": 80381, "OSMR": 9180}


def fetch_cbioportal_expression_and_clinical(study_id: str = "luad_tcga_pan_can_atlas_2018",
                                             gene: str = "CD276", entrez_id: int | None = None,
                                             base_url: str = "https://www.cbioportal.org/api",
                                             primary_tumor_only: bool = True, log2: bool = True,
                                             timeout: float = 60.0) -> pd.DataFrame:
    """Download RSEM expression of ``gene`` and patient + sample clinical data for
    a cBioPortal study (Methods, "Survival analysis" data source).

    Uses the public REST API: ``POST /molecular-profiles/{study}_rna_seq_v2_mrna/
    molecular-data/fetch`` (``entrezGeneIds``, ``sampleListId={study}_all``) and
    ``GET /studies/{study}/clinical-data?clinicalDataType=PATIENT|SAMPLE``.
    Returns one row per patient with ``expr`` (log2(RSEM+1) if ``log2``), the
    pivoted clinical attributes (e.g. ``OS_MONTHS``, ``OS_STATUS``,
    ``AJCC_PATHOLOGIC_TUMOR_STAGE``, ``SEX``, ``AGE``), ready for
    :func:`prepare_tcga_clinical`. Requires network access (not unit-tested).
    """
    import httpx

    with httpx.Client(base_url=base_url, timeout=timeout,
                      headers={"Accept": "application/json"}) as client:
        if entrez_id is None:
            entrez_id = _ENTREZ.get(gene.upper())
        if entrez_id is None:
            r = client.get(f"/genes/{gene}")
            r.raise_for_status()
            entrez_id = int(r.json()["entrezGeneId"])
        profile = f"{study_id}_rna_seq_v2_mrna"
        r = client.post(f"/molecular-profiles/{profile}/molecular-data/fetch",
                        params={"projection": "SUMMARY"},
                        json={"entrezGeneIds": [int(entrez_id)], "sampleListId": f"{study_id}_all"})
        r.raise_for_status()
        mol = pd.DataFrame(r.json())
        if mol.empty:
            raise ValueError(f"no expression returned for {gene} in {profile}")
        clin = {}
        for kind in ("PATIENT", "SAMPLE"):
            r = client.get(f"/studies/{study_id}/clinical-data",
                           params={"clinicalDataType": kind, "projection": "SUMMARY",
                                   "pageSize": 10_000_000})
            r.raise_for_status()
            clin[kind] = pd.DataFrame(r.json())

    expr = mol[["sampleId", "patientId", "value"]].copy()
    expr["value"] = pd.to_numeric(expr["value"], errors="coerce")
    if primary_tumor_only:
        # TCGA barcodes: sample type 01 = primary solid tumour
        prim = expr["sampleId"].str.match(r"^TCGA-..-....-01")
        if prim.any():
            expr = expr[prim]
    expr = expr.sort_values("sampleId").drop_duplicates("patientId")
    expr["expr"] = np.log2(expr["value"].clip(lower=0) + 1) if log2 else expr["value"]

    pat = clin["PATIENT"]
    pat_w = (pat.pivot_table(index="patientId", columns="clinicalAttributeId", values="value",
                             aggfunc="first") if not pat.empty else pd.DataFrame())
    smp = clin["SAMPLE"]
    if not smp.empty:
        smp_w = smp.pivot_table(index="sampleId", columns="clinicalAttributeId", values="value",
                                aggfunc="first")
        smp_w = smp_w.loc[smp_w.index.intersection(expr["sampleId"])]
    else:
        smp_w = pd.DataFrame()
    out = expr[["patientId", "sampleId", "expr"]].set_index("patientId")
    out = out.join(pat_w, how="left")
    if not smp_w.empty:
        dup = [c for c in smp_w.columns if c in out.columns]
        out = out.join(smp_w.drop(columns=dup), on="sampleId", how="left")
    out.index.name = "patientId"
    out.attrs.update({"study_id": study_id, "gene": gene, "entrez_id": int(entrez_id)})
    return out.reset_index()
