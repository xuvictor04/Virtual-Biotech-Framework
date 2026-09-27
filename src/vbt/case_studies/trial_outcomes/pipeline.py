"""Case study 1, step 3: target single-cell features vs clinical-trial outcomes.

Assembles the analysis dataset (paper Fig. 3 and figs. S2-S4):
labels (released or agent-annotated) + Open Targets trial mapping + target
features (tau, bimodality) + genetic evidence, then runs
  * univariate logistic / beta regressions on z-scored features (main results),
  * permutation tests (1,000 iterations; fig. S3A-B),
  * mixed-effects models with modality and therapeutic area (fig. S3C-D),
  * genetic-evidence-adjusted models and the no-genetic-evidence subset (fig. S4),
  * interpretable contrasts after k-means binarisation of tau (48% / 32% results).
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
BINARY_OUTCOMES = ("primary_success", "secondary_success", "phase1_to_2", "ever_phase4",
                   "stopped_early", "stopped_negative", "stopped_safety")


def genetic_pairs_from_open_targets(ot_path: str | Path) -> set[tuple[str, str]]:
    """(targetId, diseaseId) pairs with direct genetic-association evidence in Open Targets.

    Follows Razuvayevskaya et al.: GWAS, PheWAS, gene burden and ClinVar/EVA
    evidence aggregated into the ``genetic_association`` datatype.
    """
    import pyarrow.dataset as ds

    d = ds.dataset(Path(ot_path) / "association_by_datatype_direct", format="parquet")
    t = d.to_table(columns=["targetId", "diseaseId", "datatypeId", "score"],
                   filter=(ds.field("datatypeId") == "genetic_association") & (ds.field("score") > 0))
    df = t.to_pandas()
    return set(zip(df["targetId"], df["diseaseId"]))


def covariates_from_open_targets(ot_path: str | Path, mapping: pd.DataFrame) -> pd.DataFrame:
    """Per-trial drug modality and therapeutic area (for the mixed-effects models)."""
    drugs = pd.read_parquet(Path(ot_path) / "drug_molecule", columns=["id", "drugType"])
    dis = pd.read_parquet(Path(ot_path) / "disease", columns=["id", "therapeuticAreas"])
    dis["ta"] = dis["therapeuticAreas"].map(lambda v: "|".join(sorted(v)) if v is not None and len(v) else "none")
    m = (mapping[["nct_id", "drugId", "diseaseId"]]
         .merge(drugs.rename(columns={"id": "drugId"}), on="drugId", how="left")
         .merge(dis[["id", "ta"]].rename(columns={"id": "diseaseId"}), on="diseaseId", how="left"))
    return (m.groupby("nct_id")
             .agg(modality=("drugType", lambda s: "|".join(sorted(set(s.dropna()))) or "unknown"),
                  therapeutic_area=("ta", lambda s: sorted(set(s.dropna()))[0] if s.notna().any() else "unknown"))
             .reset_index())


def build_dataset(config: dict[str, Any], *, features_path: Path, labels: str = "released",
                  genetic_pairs: str | None = None) -> pd.DataFrame:
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
        p1 = phase1_progression(mapping)
        p1 = p1.merge(released[["nct_id", "phase", "status"]], on="nct_id", how="left")
        lab = pd.concat([lab, p1], ignore_index=True)
    outcomes = st.build_outcomes(lab, mapping)
    outcomes = outcomes[outcomes["nct_id"].isin(set(lab["nct_id"]))]  # the curated trial set only

    gene_features = pd.read_csv(features_path, index_col=0)
    feats = st.trial_level_features(mapping, gene_features, FEATURES)
    df = outcomes.merge(feats, on="nct_id", how="inner")

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
        df = df.merge(covariates_from_open_targets(ot, mapping), on="nct_id", how="left")
    thr, lab_bin = st.binarize_tau(df["tau"].to_numpy())
    df["tau_specific"] = lab_bin
    df.attrs["tau_threshold"] = thr
    return df


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
              out_dir: Path | None = None) -> pd.DataFrame:
    df = build_dataset(config, features_path=features_path, labels=labels, genetic_pairs=genetic_pairs)
    ae_cols = [c for c in df.columns if c.startswith("ae_serious_") and df[c].notna().sum() >= 30]
    outcomes = [o for o in BINARY_OUTCOMES if o in df.columns and df[o].notna().sum() >= 30]

    tables = [st.run_association_suite(df, FEATURES, outcomes, ae_cols, n_perm=n_perm).assign(analysis="main")]
    # Sensitivity analysis (harness addition): the min-across-targets aggregation
    # makes the trial-level feature depend on how many targets a drug has, and
    # multi-target drugs differ systematically in outcomes. Adjust for it.
    if "n_targets" in df.columns:
        df["log_n_targets"] = np.log(df["n_targets"].clip(lower=1))
        tables.append(st.run_association_suite(df, FEATURES, outcomes, (), n_perm=0,
                                               covariates=["log_n_targets"])
                      .query("model != 'logistic'").assign(analysis="adjusted_n_targets"))
    if "genetic_evidence" in df.columns:
        tables.append(st.run_association_suite(df, FEATURES, outcomes, ae_cols, n_perm=0,
                                               covariates=["genetic_evidence"])
                      .query("model != 'logistic'").assign(analysis="adjusted_genetic"))
        no_gen = df[df["genetic_evidence"] == 0]
        tables.append(st.run_association_suite(no_gen, FEATURES, outcomes, ae_cols, n_perm=0)
                      .assign(analysis="no_genetic_evidence_subset"))
    if {"modality", "therapeutic_area", "start_year"} <= set(df.columns):
        rows = []
        for f in FEATURES:
            for o in outcomes:
                r = st.mixed_effects_logistic(df, f, o, fixed=("phase_num", "start_year"))
                rows.append(dict(feature=f, outcome=o, model=f"mixed_logistic[{r.get('engine')}]",
                                 estimate=r.get("odds_ratio"), ci_low=r.get("ci_low"),
                                 ci_high=r.get("ci_high"), p=r.get("p"), n=r.get("n")))
            for o in ae_cols:
                r = st.mixed_effects_beta(df, f, o, fixed=("phase_num", "start_year"))
                rows.append(dict(feature=f, outcome=o, model=f"mixed_beta[{r.get('engine')}]",
                                 estimate=r.get("coef"), ci_low=r.get("ci_low"), ci_high=r.get("ci_high"),
                                 p=r.get("p"), n=r.get("n")))
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
        "ever_phase4_specific_vs_broad": st.relative_likelihood(df, "tau_specific", "ever_phase4"),
        "paper_ever_phase4_relative_increase": 0.48,
        "phase1_to_2_specific_vs_broad": st.relative_likelihood(df, "tau_specific", "phase1_to_2"),
        "paper_phase1_to_2_relative_increase": 0.40,
        "ae_specific_vs_broad": st.relative_ae_difference(df, "tau_specific", ae_cols) if ae_cols else None,
        "paper_ae_relative_difference": -0.32,
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


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
