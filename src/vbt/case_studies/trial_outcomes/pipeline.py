"""Case study 1, step 3: target single-cell features vs clinical-trial outcomes.

Assembles the analysis dataset (paper Fig. 3 and figs. S2-S4):
labels (released or agent-annotated) + Open Targets trial mapping + target
features (tau, bimodality) + genetic evidence, then runs
  * univariate logistic / beta regressions on z-scored features (main results),
  * permutation tests (1,000 iterations; fig. S3A-B),
  * mixed-effects models with modality and therapeutic area (fig. S3C-D),
  * genetic-evidence-adjusted models and the no-genetic-evidence subset (fig. S4),
  * interpretable contrasts after k-means binarisation of tau (48% / 32% results).

Outcome definitions, trial set, covariates and binarisation follow the
authors' analysis code in the paper's Zenodo archive by default
(``definitions="authors"``; see docs/ZENODO_REPLICATION.md): the six-section
outcome structure (Phase II+/III+/IV of the trial's own phase, stopped
status, ChEMBL-first stop reasons, endpoints coded POSITIVE vs everything
else within Phase II/III, Phase I->II), trials without tau *or* bimodality
dropped, the 48% computed at the target level, glmer/glmmTMB-style mixed
models with drug-type and therapeutic-area combinations and per-outcome
covariates. ``definitions="methods"`` keeps the harness's earlier
Methods-text reading (``ever_phase4`` per drug-indication, NEGATIVE-only
failures, early-stopping outcomes).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import stats as st
from .annotate import labels_path, mapping_path
from .phase1 import phase1_progression

FEATURES = ("tau", "bimodality")
#: the authors' six-section outcome structure (build_outcomes columns, in their order)
BINARY_OUTCOMES = tuple(st.AUTHORS_OUTCOME_COLUMNS[a[1]] for a in st.AUTHORS_ANALYSES)
#: the harness's earlier Methods-text outcomes (``definitions="methods"``)
METHODS_OUTCOMES = ("primary_success", "secondary_success", "phase1_to_2", "ever_phase4",
                    "stopped_early", "stopped_negative", "stopped_safety")
#: build_outcomes column -> the authors' covariate set for the mixed models
_MIXED_KIND = {**{c: "phase" for c in ("phase_ge_2", "phase_ge_3", "phase_ge_4", "ever_phase4")},
               **{c: "status" for c in ("status_terminated", "status_withdrawn", "status_suspended",
                                         "stopped_early", "stopped_negative", "stopped_safety")},
               **{c: "endpoint" for c in ("primary_success", "secondary_success", "either_success")},
               "phase1_to_2": "phase1"}
# outcomes for the genetic-evidence replication (fig. S2; Razuvayevskaya et al.)
GENETIC_OUTCOMES = ("ever_phase_ge_2", "ever_phase_ge_3", "ever_phase4", "phase1_to_2", "stopped_early",
                    "stopped_negative", "stopped_safety", "primary_success", "secondary_success")


def genetic_pairs_from_open_targets(ot_path: str | Path) -> set[tuple[str, str]]:
    """(targetId, diseaseId) pairs with direct genetic-association evidence in Open Targets.

    Follows Razuvayevskaya et al.: GWAS, PheWAS, gene burden and ClinVar/EVA
    evidence aggregated into the ``genetic_association`` datatype.
    """
    import pyarrow.dataset as ds

    d = ds.dataset(Path(ot_path) / "association_by_datatype_direct", format="parquet")
    # any genetic_association row counts (the authors' code; no score threshold)
    t = d.to_table(columns=["targetId", "diseaseId", "datatypeId", "score"],
                   filter=ds.field("datatypeId") == "genetic_association")
    df = t.to_pandas()
    return set(zip(df["targetId"], df["diseaseId"]))


def covariates_from_open_targets(ot_path: str | Path, mapping: pd.DataFrame, *,
                                 definitions: str = "authors") -> pd.DataFrame:
    """Per-trial drug modality and therapeutic area (for the mixed-effects models).

    ``definitions="authors"``: the authors' ``drugType_combo`` (sorted drug
    types joined with '-') and cleaned ``ta_combo`` (junk areas dropped,
    pancreas/psychiatric collapsed, names joined with '|'); ``"methods"``:
    the earlier union of raw therapeutic-area IDs.
    """
    if definitions == "authors":
        drugs = pd.read_parquet(Path(ot_path) / "drug_molecule", columns=["id", "drugType"]).set_index("id")
        dis = pd.read_parquet(Path(ot_path) / "disease", columns=["id", "name", "therapeuticAreas"])
        return (st.drugtype_combo(mapping, drugs["drugType"]).rename(columns={"drugType_combo": "modality"})
                .merge(st.ta_combo(mapping, dis).rename(columns={"ta_combo": "therapeutic_area"}), on="nct_id"))
    drugs = pd.read_parquet(Path(ot_path) / "drug_molecule", columns=["id", "drugType"])
    dis = pd.read_parquet(Path(ot_path) / "disease", columns=["id", "therapeuticAreas"])
    dis["ta"] = dis["therapeuticAreas"].map(lambda v: "|".join(sorted(v)) if v is not None and len(v) else None)
    m = (mapping[["nct_id", "drugId", "diseaseId"]]
         .merge(drugs.rename(columns={"id": "drugId"}), on="drugId", how="left")
         .merge(dis[["id", "ta"]].rename(columns={"id": "diseaseId"}), on="diseaseId", how="left"))
    return (m.groupby("nct_id")
             .agg(modality=("drugType", lambda s: "|".join(sorted(set(s.dropna()))) or "unknown"),
                  # the union of therapeutic areas across *all* the trial's diseases
                  therapeutic_area=("ta", st.combine_therapeutic_areas))
             .reset_index())


def build_dataset(config: dict[str, Any], *, features_path: Path, labels: str = "released",
                  genetic_pairs: str | None = None, definitions: str = "authors") -> pd.DataFrame:
    mapping = pd.read_parquet(mapping_path(config))
    released = pd.read_csv(labels_path(config))
    if labels == "released":
        lab = released
    else:
        pred = pd.read_csv(labels)
        # Agent labels cover Phase II/III endpoints; phase/status come from the registry
        # snapshot, and Phase I progression is algorithmic (paper Methods).
        meta = released[["nct_id", "phase", "status", "studyStopReasonCategories"]]
        lab = pred.merge(meta, on="nct_id", how="left")
        # Phase I: completed trials of the released-label universe only (11,412 in the paper)
        rel_p1 = released.loc[(released["phase"] == 1.0) & released["phase2_progression"].notna(), "nct_id"]
        p1 = phase1_progression(mapping, statuses=("Completed",), universe=set(rel_p1))
        p1 = p1.merge(released[["nct_id", "phase", "status"]], on="nct_id", how="left")
        lab = pd.concat([lab, p1], ignore_index=True)
    if definitions not in ("authors", "methods"):
        raise ValueError("definitions must be 'authors' or 'methods'")
    outcomes = (st.build_outcomes(lab, mapping) if definitions == "authors" else
                st.build_outcomes(lab, mapping, endpoint_coding="strict", stop_reason_source="vb_first"))
    outcomes = outcomes[outcomes["nct_id"].isin(set(lab["nct_id"]))]  # the curated trial set only

    gene_features = pd.read_csv(features_path, index_col=0)
    feats = st.trial_level_features(mapping, gene_features, FEATURES)
    df = outcomes.merge(feats, on="nct_id", how="inner")
    if definitions == "authors":  # the authors drop trials missing either feature
        df = df.dropna(subset=list(FEATURES)).reset_index(drop=True)
    for k in (2, 3, 4):  # phase dummies (Phase I / early Phase I reference) for the mixed models
        df[f"phase{k}"] = (df["phase_num"] == float(k)).astype(float)

    ot = config.get("tool_env", {}).get("OPEN_TARGETS_DATA_PATH")
    pairs: set[tuple[str, str]] | None = None
    if genetic_pairs:
        g = pd.read_csv(genetic_pairs)
        pairs = set(zip(g["targetId"], g["diseaseId"]))
    elif ot and Path(ot).exists():
        pairs = genetic_pairs_from_open_targets(ot)
    if pairs is not None:
        flags = st.genetic_evidence_flags(mapping, pairs).rename("genetic_evidence").astype(float)
        df = df.merge(flags, left_on="nct_id", right_index=True, how="left")
    if ot and Path(ot).exists() and "modality" not in df.columns:
        df = df.merge(covariates_from_open_targets(ot, mapping, definitions=definitions), on="nct_id",
                      how="left")
    thr, lab_bin = st.binarize_tau(df["tau"].to_numpy())
    df["tau_specific"] = lab_bin
    df.attrs["tau_threshold"] = thr
    df.attrs["definitions"] = definitions
    return df


def _mixed_fixed(outcome: str) -> tuple[str, ...]:
    """The authors' fixed covariates for one outcome (year always; phase dummies
    for status and AE outcomes; phase3 for endpoints)."""
    kind = _MIXED_KIND.get(outcome, "status" if outcome.startswith("stop_") else "phase")
    if kind == "status":
        return ("start_year", "phase2", "phase3", "phase4")
    if kind == "endpoint":
        return ("start_year", "phase3")
    return ("start_year",)


def gene_permutation_null(config: dict[str, Any], df: pd.DataFrame, features_path: Path,
                          outcomes: list[str], n_iter: int = 100, seed: int = 0) -> pd.DataFrame:
    """Calibration null (harness addition): shuffle feature values across genes.

    Outcome permutation (the paper's test) breaks every feature-outcome link,
    including confounded ones. Shuffling *genes* keeps the trial -> target
    structure (number of targets, shared targets across trials), so the null
    reflects what an uninformative feature would show through the same
    aggregation. Returns observed OR, null quantiles and empirical p.
    """
    mapping = pd.read_parquet(mapping_path(config), columns=["nct_id", "targetId"])
    genes = pd.read_csv(features_path, index_col=0)[list(FEATURES)]
    base = df.drop(columns=[c for c in FEATURES if c in df.columns])
    rng = np.random.default_rng(seed)
    observed = {(f, o): st.univariate_logistic(df, f, o)["odds_ratio"] for f in FEATURES for o in outcomes}
    null: dict[tuple[str, str], list[float]] = {k: [] for k in observed}
    for _ in range(n_iter):
        shuffled = genes.copy()
        shuffled.index = rng.permutation(genes.index.to_numpy())
        feats = st.trial_level_features(mapping, shuffled, FEATURES)
        if "nct_id" not in feats.columns:
            feats = feats.reset_index()
        feats = feats[["nct_id", *FEATURES]]
        d = base.merge(feats, on="nct_id", how="inner")
        for f, o in observed:
            null[(f, o)].append(st.univariate_logistic(d, f, o)["odds_ratio"])
    rows = []
    for (f, o), obs in observed.items():
        z = np.log(np.asarray(null[(f, o)], dtype=float))
        lo = np.log(obs)
        rows.append(dict(feature=f, outcome=o, observed_or=obs, null_median_or=float(np.exp(np.nanmedian(z))),
                         null_q025=float(np.exp(np.nanquantile(z, .025))), null_q975=float(np.exp(np.nanquantile(z, .975))),
                         p_gene_perm=(np.sum(np.abs(z - np.nanmedian(z)) >= abs(lo - np.nanmedian(z))) + 1) / (n_iter + 1),
                         n_iter=n_iter))
    return pd.DataFrame(rows)


def run_stats(config: dict[str, Any], *, features_path: Path, labels: str = "released",
              genetic_pairs: str | None = None, n_perm: int = 1000, gene_perm: int = 0,
              out_dir: Path | None = None, definitions: str = "authors") -> pd.DataFrame:
    df = build_dataset(config, features_path=features_path, labels=labels, genetic_pairs=genetic_pairs,
                       definitions=definitions)
    ae_cols = [c for c in df.columns if c.startswith("ae_serious_") and df[c].notna().sum() >= 30]
    wanted = BINARY_OUTCOMES if definitions == "authors" else METHODS_OUTCOMES
    # the authors skip outcomes with fewer than 30 positive trials
    outcomes = [o for o in wanted if o in df.columns and df[o].notna().sum() >= 30 and (df[o] == 1).sum() >= 30]

    tables = [st.run_association_suite(df, FEATURES, outcomes, ae_cols, n_perm=n_perm).assign(analysis="main")]
    # Sensitivity analysis (harness addition): the min-across-targets aggregation
    # makes the trial-level feature depend on how many targets a drug has, and
    # multi-target drugs differ systematically in outcomes. Adjust for it.
    if "n_targets" in df.columns:
        df["log_n_targets"] = np.log(df["n_targets"].clip(lower=1))
        tables.append(st.run_association_suite(df, FEATURES, outcomes, ae_cols, n_perm=0,
                                               covariates=["log_n_targets"], include_unadjusted=False)
                      .assign(analysis="adjusted_n_targets"))
    if "genetic_evidence" in df.columns:
        # covariate-adjusted logistic AND beta models; BH over exactly this family
        tables.append(st.run_association_suite(df, FEATURES, outcomes, ae_cols, n_perm=0,
                                               covariates=["genetic_evidence"], include_unadjusted=False)
                      .assign(analysis="adjusted_genetic"))
        no_gen = df[df["genetic_evidence"] == 0]
        tables.append(st.run_association_suite(no_gen, FEATURES, outcomes, ae_cols, n_perm=0)
                      .assign(analysis="no_genetic_evidence_subset"))
        # fig. S2: Razuvayevskaya-style replication of the genetic-evidence associations
        gen_out = BINARY_OUTCOMES if definitions == "authors" else GENETIC_OUTCOMES
        rep = st.genetic_evidence_replication(df, [o for o in gen_out if o in df.columns], ae_cols)
        tables.append(rep.drop(columns=["exp_coef"], errors="ignore"))
    if {"modality", "therapeutic_area", "start_year"} <= set(df.columns):
        rows = []
        for f in FEATURES:
            for o in outcomes:
                r = st.mixed_effects_logistic(df, f, o, fixed=_mixed_fixed(o))
                rows.append(dict(feature=f, outcome=o, model=f"mixed_logistic[{r.get('engine')}]",
                                 estimate=r.get("odds_ratio"), ci_low=r.get("ci_low"),
                                 ci_high=r.get("ci_high"), p=r.get("p"), n=r.get("n"), engine=r.get("engine")))
            for o in ae_cols:
                r = st.mixed_effects_beta(df, f, o, fixed=("start_year", "phase2", "phase3", "phase4"))
                rows.append(dict(feature=f, outcome=o, model=f"mixed_beta[{r.get('engine')}]",
                                 estimate=r.get("coef"), ci_low=r.get("ci_low"), ci_high=r.get("ci_high"),
                                 p=r.get("p"), n=r.get("n"), engine=r.get("engine")))
        mixed = pd.DataFrame(rows)
        mixed["p_fdr"] = st.benjamini_hochberg(mixed["p"])
        tables.append(mixed.assign(analysis="mixed_effects"))
    table = pd.concat(tables, ignore_index=True)

    headline = {
        "n_trials": int(len(df)),
        "tau_threshold": float(df.attrs.get("tau_threshold", np.nan)),
        "paper_tau_threshold": 0.69,
        "pearson_tau_bimodality": float(df[["tau", "bimodality"]].corr().iloc[0, 1]),
        "paper_pearson": 0.54,
        "ever_phase4_specific_vs_broad": (st.relative_likelihood(df, "tau_specific", "ever_phase4")
                                          if "ever_phase4" in df else None),
        # the paper's 48%: *targets* (not trials) above the trial-level threshold ever reaching Phase IV
        "target_level_phase4": _target_level(mapping_frame(config), features_path, df),
        "paper_ever_phase4_relative_increase": 0.48,
        "phase1_to_2_specific_vs_broad": st.relative_likelihood(df, "tau_specific", "phase1_to_2"),
        "paper_phase1_to_2_relative_increase": 0.40,
        "ae_specific_vs_broad": (st.relative_ae_difference(df, "tau_specific", ae_cols, with_beta=True)
                                 if ae_cols else None),
        "paper_ae_relative_difference": -0.32,
        "paper_ae_relative_difference_compared_with": "ae_specific_vs_broad.mean_of_per_organ",
        "bimodality_kurtosis": df.attrs.get("bimodality_kurtosis", "see target_features.csv provenance"),
    }
    if out_dir is None:
        out_dir = Path(features_path).parent
    if gene_perm:
        calib = gene_permutation_null(config, df, features_path, outcomes, n_iter=gene_perm)
        calib.to_csv(Path(out_dir) / "case1_gene_permutation_null.csv", index=False)
        print(calib.to_string(index=False))
    (Path(out_dir) / "case1_headline.json").write_text(json.dumps(headline, indent=2, default=_json))
    df.to_parquet(Path(out_dir) / "case1_analysis_dataset.parquet")
    print(json.dumps(headline, indent=2, default=_json))
    return table


def mapping_frame(config: dict[str, Any]) -> pd.DataFrame:
    return pd.read_parquet(mapping_path(config), columns=["nct_id", "targetId", "phase"])


def _target_level(mapping: pd.DataFrame, features_path: Path, df: pd.DataFrame) -> dict | None:
    """Target-level Phase IV contrast (the authors' Part 1): fold − 1 of the share
    of high-tau vs low-tau targets whose highest phase over all trials is IV."""
    try:
        genes = pd.read_csv(features_path, index_col=0)["tau"]
        lwt = mapping.dropna(subset=["targetId"])
        lwt = lwt[lwt["nct_id"].isin(set(df["nct_id"]))] if len(df) else lwt
        lwt = lwt.assign(phase=lwt["phase"].map(st.phase_to_numeric))
        tl = st.target_level_table(lwt, genes, float(df.attrs.get("tau_threshold", np.nan)))
        r = tl[(tl["model"] == "unadjusted") & (tl["outcome"] == "Ever reached Phase IV")].iloc[0]
        return dict(relative_increase=float(r["fold"] - 1), odds_ratio=float(r["OR"]), n_targets=int(r["n_total"]),
                    rate_specific=float(r["rate_hi"]), rate_broad=float(r["rate_lo"]))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
