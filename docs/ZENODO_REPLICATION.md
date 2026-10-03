# Replication against the authors' Zenodo archive

The GitHub repository of *The Virtual Biotech* (Zhang et al., Science 2026) contains the
agentic system but not the case-study analysis code. The authors deposited that code,
their result tables, the specialist agents' reports and the data subsets in a Zenodo
record: **doi:10.5281/zenodo.22259123**, file `virtualbiotech_submission.zip` (2.9 GB,
CC-BY-4.0, 3,051 members). This document describes how the harness reads that archive
and how closely our implementation (`src/vbt/case_studies/trial_outcomes/*`,
`src/vbt/analysis/*`) reproduces the authors' own outputs.

All tables below come from real runs on the archive (October 2026). Each one says which
command produced it.

## 1. Getting the archive: `vbt data zenodo`

`src/vbt/data/zenodo.py` reads the zip remotely. Zenodo serves the file with HTTP
`Range` support, so `HTTPRangeFile` lets `zipfile` read the central directory and pull
single members without downloading 2.9 GB.

```bash
vbt data zenodo presets                              # list the named selections
vbt data zenodo list --pattern 'clinical_trials/**'  # reads only the zip directory (~0.4 MB)
vbt data zenodo fetch --preset case1                 # Case 1 code, results, data, competitor annotations
vbt data zenodo fetch --preset b7h3-results              # one preset per call
vbt data zenodo fetch --pattern 'b7-h3/data/single_cell_outputs/liana_*/liana_cd276_*_raw.csv'
vbt data zenodo download                             # the whole zip, md5-checked against the record
```

* **Where files go.** Members keep their archive paths under `--dest`. The default is
  `${VBT_ZENODO_DIR:-<project>/data/zenodo}`, so files land in
  `.../virtualbiotech_submission/...`. `vbt.data.zenodo.zenodo_root(config)` returns that
  folder, and `config['zenodo']['dir']` can override it (no default config key is
  needed).
* **Skipping files already present.** A member is skipped when a local file has the same
  size and the same zip CRC32. Every file is first written under a temporary name and
  then renamed into place.
* **Retries and network settings.** Failed range requests (transport errors, 429, 5xx)
  are retried with exponential backoff. Proxies and CA bundles come from the environment
  (plain `httpx` defaults).
* **Presets.**

  | Preset | Contents |
  |---|---|
  | `case1` | `clinical_trials/{code,results,data}/**`, `clinical_trials/benchmarks/*/*.csv` and the README |
  | `b7h3-code` | B7-H3 code |
  | `b7h3-results` | Small csv/json/md files under `b7-h3/data`, without the 2.5 GB h5ad |
  | `b7h3-scrnaseq` | The 2.5 GB h5ad (opt-in) |
  | `osmr-code` | OSMR code |
  | `osmr-results` | OSMR tables, processed bulk cohorts and inputs |
  | `traces` | Agent reports |
  | `benchmarks` | Competitor runs, without images |
  | `system-snapshot` | Frozen snapshot of the agentic-system source |
  | `small` | Every non-image member under 5 MB |
  | `all` | The whole archive |

* **Real use.** `fetch --pattern '.../liana_cd276_*_raw.csv'` fetched four members
  (93.3 MB uncompressed) with 7.3 MB of HTTP transfer. The cross-disease inputs needed
  8.2 MB uncompressed and 4.1 MB transferred.
* **Tests.** `tests/test_zenodo.py` covers listing, presets, CRC skip, atomic writes,
  retries, md5 download and the CLI. It runs against a local Range-capable
  `http.server` and a fake records API.

## 2. Case 1: exact replication (`vbt case1 replicate`)

```bash
vbt case1 replicate [--zenodo-dir DIR] [--n-perm 1000] [--n-perm-beta N] [--gene-perm 200] \
                    [--no-mixed] [--no-expr] [--out results/case1_replication]
```

`replicate_case1()` (`src/vbt/case_studies/trial_outcomes/replicate.py`) loads
`clinical_trials/data/*`. It runs our statistics on exactly those inputs and aligns every
one of our estimates with the authors' row (feature × outcome × model) in
`clinical_trials/results/*.csv`.

It writes these files:

* `case1_replication_comparison.csv`: ours and theirs for every row (estimate, CI, p,
  n), with absolute and relative deltas, a match flag and the engine used.
* `case1_replication_summary.md`
* `case1_replication_headline.json`
* `ours_*.csv`: our full tables.
* `case1_harness_checks.csv`

### 2.1 What the authors' code does, and what we changed

We read the seven scripts in `clinical_trials/code/` line by line. Our implementation
now follows them by default. Where the authors' code goes beyond or contradicts the
Methods text, the behaviour can be switched with a parameter; the default is the
authors' code.

**Trial set.** Labels (56,707 trials) are inner-joined with ChEMBL (NCT, target) pairs
and then with the 1,511 featurised genes. Feature columns are aggregated with MIN per
trial and clinical columns take the first value. Trials missing tau *or* bimodality are
dropped, which leaves **56,555** trials.

* Paper text: "55,984 clinical trials". That number is the count of Phase I–IV labels
  (56,707 minus 723 early-Phase-I trials). It is not the analysed n.
* Ours: `stats.aggregate_trials`. The pipeline default now also drops trials missing
  either feature.

**Outcomes: the "6-section structure"** (`stats.AUTHORS_ANALYSES`, `authors_outcome`,
`authors_subset`; `build_outcomes(endpoint_coding="authors", stop_reason_source="chembl_first")`):

| Section | Authors' definition | Our earlier harness reading (now `endpoint_coding="strict"` / `definitions="methods"`) |
|---|---|---|
| A. Phase | Phase II+, III+, IV: the trial's *own* phase ≥ k, over all trials (cross-sectional) | `ever_phase4`: a drug–indication pair ever reached Phase IV |
| B. Stopped status | `status == Terminated` (or `Withdrawn`, `Suspended`) vs every other trial | `stopped_early` vs Completed |
| C. Stop reasons | 11 ChEMBL categories. ChEMBL `studyStopReasonCategories` is used first and the Virtual Biotech categories only when ChEMBL has none. Only trials with any category are included. | Virtual Biotech categories first; Negative/Safety only |
| D. Endpoints | Phase II/III trials only. POSITIVE = 1; **NEGATIVE, UNKNOWN or missing = 0**. Includes an "Either Positive" outcome. | POSITIVE = 1, NEGATIVE = 0, otherwise excluded, any phase |
| E. Phase I→II | Label EVER = 1, NEVER = 0 (the 11,412 labelled Phase I trials, 11,359 with features) | Same |
| F. Adverse events | 10 organ-system serious-AE %, beta regression | Same outcomes |

An outcome is skipped when it has fewer than 30 positives, as in the authors' code.

**Model details.**

* **z-scoring.** `StandardScaler` uses the population SD (ddof = 0), so the stats module
  now does too (`stats.ZSCORE_DDOF = 0`). For logistic models, a missing feature is set
  to 0 before z-scoring (moot for tau and bimodality, which are never missing after the
  trial filter). For beta models, rows with a missing feature are dropped.
* **Beta regression.** The response is `np.clip(y/100, 1e-6, 1 - 1e-6)`; the earlier
  harness used the Smithson–Verkuilen squeeze. The default is now `squeeze="clip"`, and
  `"smithson_verkuilen"` remains selectable. CI = coef ± 1.96·SE.
* **Beta regression engine.** R `betareg` was not available, so we wrote
  `stats.betareg_fit`, a Python port of `betareg()`:
  * ML estimates, then Fisher scoring.
  * Standard errors from the **expected** information matrix, as `summary(betareg)`
    reports them. statsmodels' `BetaModel` gives the same estimates, but its
    observed-Hessian SEs are about 0.5% off.
  * `beta_regression(engine="auto")` uses R betareg when rpy2 is available, otherwise
    this port (engine label `betareg_py`).
* **Permutation test** (fig. S3A–B).
  * Empirical p = (#|null| ≥ |obs| + 1)/(n + 1). The Methods text says "proportion"
    without the +1. Our `p_perm` already used +1, and `p_perm_raw` gives the plain
    proportion.
  * The authors draw from a single `np.random.default_rng(42)` across all analyses in a
    fixed order. `replicate.run_permutation` reproduces that order, so with 1,000
    permutations the null draws are **identical** to theirs.
* **Mixed models** (fig. S3C–D).
  * Random intercepts: crossed `(1|drugType_combo) + (1|ta_combo)`.
    * `drugType_combo`: the sorted drug types joined with "-".
    * `ta_combo`: the union of the therapeutic areas of all the trial's diseases. Four
      junk areas are dropped (phenotype, biological process, measurement, medical
      procedure), and pancreas → endocrine and psychiatric → nervous system are
      collapsed.
  * Fixed effects: `year_z` (z-scored first trial year) in every model. Status and AE
    models add Phase 2, 3 and 4 dummies; endpoint models add only `phase3`. The Methods
    text says "adjusted for trial phase and enrollment year" for all models.
  * Phase I→II mixed models use `phase == 1` trials only.
  * Ours: `stats.drugtype_combo`, `ta_combo`, `mixed_covariates`, and
    `pipeline._mixed_fixed` for the pipeline.
  * Without R, `stats.glmm_laplace` is a Python port of the Laplace fit:
    * Binomial (glmer): penalized IRLS for the conditional modes, the `nAGQ = 0` stage,
      then joint optimisation of (θ, β). SEs are lme4's conditional (RX-based) `vcov`.
    * Beta family (glmmTMB): the same Laplace objective plus log φ, with SEs from the
      numerical Hessian.
    * `mixed_effects_logistic` and `mixed_effects_beta` gain `engine="laplace"`, and
      their `"auto"` setting falls back to it instead of the earlier variational-Bayes /
      fixed-dummy approximations.
* **Binarisation.** k-means (k = 2, `random_state = 42`) on the trial-level MIN tau
  gives a threshold of **0.689**. The binary analysis keeps trials with tau even if
  bimodality is missing.
* **"48% more likely to ever reach Phase IV"** is a **target-level** contrast, not a
  trial-level one. For each target, the highest phase over all its labelled trials is
  compared between targets with gene-level tau ≥ the trial-level threshold and the rest:
  39.1% vs 26.5% (fold 1.476). The adjusted model uses log(1 + n_trials). Ours:
  `stats.target_level_table`; the pipeline headline `target_level_phase4`.
* **"40% more likely to progress from Phase I to Phase II"** is the trial-level fold
  30.2% / 21.6% = 1.397 (`stats.contingency_or`).
* **"32% lower adverse-event rates"** is the mean over the 10 organ systems of
  (mean_specific / mean_broad − 1) = −0.320 (`ae_group_comparison`). Our
  `relative_ae_difference(...)["mean_of_per_organ"]` already reported this definition.
* **Pearson ρ = 0.54** is the *gene-level* correlation of tau and bimodality (0.539).
  At trial level (MIN-aggregated) it is 0.571.
* **Genetic evidence.**
  * A trial-level flag of 1 when any (target, disease) pair has *any*
    `genetic_association` row (no score threshold; the pipeline's former `score > 0`
    filter was a no-op on these data and was removed).
  * `fraction_genetic_evidence` is also used.
  * Logistic models use the flag raw (0/1). Beta models z-score both genetic features.
  * Ours: `stats.genetic_evidence_table`, `replicate.run_genetic`.
* **Combined models.**
  * Correlation table.
  * Univariate, 3-way and three 2-way models (genetic flag raw; tau and bimodality
    z-scored) over the outcomes with the first four stop reasons.
  * One BH over the whole table.
  * Ours: `stats.multivariable_logistic` and `multivariable_beta`, `replicate.run_combined`.
* **FDR families.**
  * Main table: one BH over all logistic and beta rows.
  * Expression table: BH per method.
  * Mixed models: BH over the feature terms only.
  * Target level: BH per model.
  * AE contrasts: BH per test.
  * All of these are reproduced.

`pipeline.run_stats(definitions="authors")` (CLI `vbt case1 stats --definitions
{authors,methods}`) now uses these outcomes, covariates and contrasts.

### 2.2 Results

These results come from the full run of `vbt case1 replicate` on the archive extract, in
about 20 minutes on 4 CPUs without R: 1,000 outcome permutations, 200 gene-label
permutations, all GLMMs and the 2,604 expression models. The engines were statsmodels
GLM for logistic models, `betareg_py` for beta regression and `laplace_py` for GLMMs.

**Headline numbers: ours, the authors' result tables, and the paper text**

| quantity | ours | authors' tables | paper text |
|---|---|---|---|
| trials analysed (Phase II+ model n) | 56,555 | 56,555 | 55,984 |
| OR primary endpoint, tau (per SD) | 1.116 | 1.116 | 1.12 |
| OR secondary endpoint, tau | 1.117 | 1.117 | 1.12 |
| OR Phase I→II, tau | 1.269 | 1.269 | 1.27 |
| OR primary endpoint, bimodality | 1.126 | 1.126 | – |
| OR Phase I→II, bimodality | 1.181 | 1.181 | – |
| specific targets: ever reach Phase IV (target level, fold − 1) | 0.4757 | 0.4757 | 0.48 |
| specific-target trials: Phase I→II (fold − 1) | 0.3968 | 0.3968 | 0.4 |
| specific-target trials: serious AE rate (mean over organs of fold − 1) | -0.3199 | -0.3199 | -0.32 |
|   same, pooled over organs (Σ mean_hi / Σ mean_lo − 1) | -0.3421 | -0.3421 | – |
| k-means tau threshold | 0.689 | 0.689 | 0.69 |
| Pearson tau vs bimodality (trial level, MIN-aggregated) | 0.5706 | – | 0.54 |
| Pearson tau vs bimodality (gene level) | 0.5393 | 0.5393 | – |

**Agreement per table.** Every row of every result table the authors shipped was
compared: 4,353 rows in all.

| table | rows | estimate matches | n matches | max abs Δ | median abs Δ | engine |
|---|---|---|---|---|---|---|
| binary ae_bimodality (fold) | 10 | 10 | 10 | 0 | 0 |  |
| binary ae_tau (fold) | 10 | 10 | 10 | 0 | 0 |  |
| binary phase1to2 | 2 | 2 | 2 | 0 | 0 |  |
| binary target_bimodality | 6 | 6 | 6 | 4.44e-16 | 1.96e-16 |  |
| binary target_bimodality (fold) | 6 | 6 | 6 | 0 | 0 |  |
| binary target_tau | 6 | 6 | 6 | 4.44e-16 | 0 |  |
| binary target_tau (fold) | 6 | 6 | 6 | 0 | 0 |  |
| combined correlation | 22 | 22 | 22 | 9.71e-17 | 4.47e-17 |  |
| combined multivariate | 288 | 288 | 288 | 1.14e-09 | 4.65e-16 |  |
| expression (expr_results) | 2604 | 2604 | 2604 | 5.98e-08 | 2.07e-09 | statsmodels |
| genetic evidence | 62 | 62 | 62 | 2.08e-10 | 2.21e-16 | betareg_py, statsmodels |
| mixed effects | 164 | 164 | 164 | 0.0148 | 9.39e-06 | laplace_py[beta], laplace_py[binomial] |
| permutation (empirical p) | 62 | 62 | 62 | 1.11e-16 | 9.91e-17 |  |
| permutation (null SD) | 62 | 62 | 62 | 5.68e-12 | 1.52e-14 |  |
| univariate (all_results) | 1023 | 1023 | 1023 | 1.1e-08 | 4.23e-16 | betareg_py, statsmodels |

Δ is on the log-OR scale for odds ratios and on the coefficient scale otherwise; a row matches when n agrees and |Δ| ≤ 5e-4 or the relative difference ≤ 1e-3 (mixed models: |Δ| ≤ 5e-3 or ≤ 0.05 × the authors' SE; aliased terms reported NA by R and dropped by us count as matches; permutation p: ±2/(n+1); permutation null SD: 10%).

* **Logistic models, beta models and contrasts** match to about 1e-8. That covers the
  1,023-row main table (logistic and beta), the 2,604-row expression table, the genetic,
  combined and correlation tables, and all binary and target-level contrasts. The n per
  model matches for every row.
* **R betareg.** The pure-Python `betareg_fit` reproduces R `betareg` coefficients and
  SEs to about 1e-8.
* **Permutation test.** The empirical p and the null SD agree for all 62 rows; the null
  SDs agree to 6e-12. With the authors' seed, feature order and 1,000 permutations, the
  shuffles (and therefore the null distributions) are identical.
* **Mixed models: estimates.** All 40 feature terms are within 8.1e-5 absolute, a median
  relative difference of 1e-4. That is lme4/glmmTMB versus our Laplace port. The largest
  covariate difference is 0.015, on the Phase 2/3 dummies of the AE models: they are
  nearly collinear with the intercept (few Phase I/IV trials report AEs) and have
  SE ≈ 0.7–0.8.
* **Mixed models: aliased terms.** Thirty-two covariate terms are aliased (all-zero
  `phase4`, or `phase3` collinear when an AE subset has no Phase I or IV trial). R reports
  them as NA and we drop them the same way (`stats.independent_columns`).
* **Mixed models: significance.** Significance at 0.05 agrees for all feature terms.

**Discrepancies with the paper text.** Each one is explained by the authors' code:

* **n = 55,984 vs 56,555.** The paper quotes the number of labelled Phase I–IV trials.
  The models use every labelled trial with features, including 723 early-Phase-I trials
  (which count as phase < 2).
* **ρ = 0.54** is the gene-level Pearson correlation (0.539). At trial level, after MIN
  aggregation, it is 0.571.
* **"48% more likely to ever reach Phase IV"** is computed at the target level (fold
  1.476). It is not the trial-level Phase IV model (per-SD OR 1.43).
* **"32% lower adverse event rates"** is the mean over organs of the per-organ fold − 1
  (−0.320). The pooled definition gives −0.342.

**Harness confounding checks on the real features** (`case1_harness_checks.csv`):

| feature | outcome | OR | OR adjusted for log #targets | p adj | survives |
|---|---|---|---|---|---|
| tau_cell_type | Phase II+ | 1.165 | 1.139 | 1.1e-31 | True |
| bimodality_score | Phase II+ | 1.193 | 1.173 | 1.22e-44 | True |
| tau_cell_type | Phase III+ | 1.323 | 1.249 | 7.68e-96 | True |
| bimodality_score | Phase III+ | 1.356 | 1.293 | 3.13e-137 | True |
| tau_cell_type | Phase IV | 1.431 | 1.296 | 2.55e-40 | True |
| bimodality_score | Phase IV | 1.311 | 1.186 | 6.22e-25 | True |
| tau_cell_type | Terminated | 0.8783 | 0.9036 | 2.83e-13 | True |
| bimodality_score | Terminated | 0.812 | 0.8163 | 2.38e-45 | True |
| tau_cell_type | Withdrawn | 0.9652 | 0.9693 | 0.173 | False |
| bimodality_score | Withdrawn | 0.8849 | 0.8646 | 2.9e-10 | True |
| tau_cell_type | Suspended | 0.908 | 1.019 | 0.792 | False |
| bimodality_score | Suspended | 0.7012 | 0.7188 | 1.41e-05 | True |
| tau_cell_type | Primary Positive | 1.116 | 1.073 | 9.65e-09 | True |
| bimodality_score | Primary Positive | 1.126 | 1.088 | 5.21e-12 | True |
| tau_cell_type | Secondary Positive | 1.117 | 1.065 | 3.37e-07 | True |
| bimodality_score | Secondary Positive | 1.134 | 1.089 | 3.27e-12 | True |
| tau_cell_type | Either Positive | 1.119 | 1.074 | 8.94e-09 | True |
| bimodality_score | Either Positive | 1.138 | 1.099 | 2.47e-14 | True |
| tau_cell_type | Phase 2 Progression | 1.269 | 1.123 | 9.53e-06 | True |
| bimodality_score | Phase 2 Progression | 1.181 | 1.042 | 0.0878 | False |

| feature | outcome | observed OR | gene-perm null median OR [2.5%, 97.5%] | p (two-sided) | survives |
|---|---|---|---|---|---|
| tau_cell_type | Phase II+ | 1.165 | 1.078 [0.9928, 1.154] | 0.0697 | False |
| tau_cell_type | Phase III+ | 1.323 | 1.155 [1.005, 1.311] | 0.0498 | True |
| tau_cell_type | Phase IV | 1.431 | 1.203 [0.9943, 1.442] | 0.0647 | False |
| tau_cell_type | Terminated | 0.8783 | 0.927 [0.868, 0.984] | 0.109 | False |
| tau_cell_type | Withdrawn | 0.9652 | 0.9835 [0.9206, 1.039] | 0.517 | False |
| tau_cell_type | Suspended | 0.908 | 0.879 [0.7662, 0.995] | 0.607 | False |
| tau_cell_type | Primary Positive | 1.116 | 1.073 [1.007, 1.152] | 0.249 | False |
| tau_cell_type | Secondary Positive | 1.117 | 1.084 [1.015, 1.166] | 0.388 | False |
| tau_cell_type | Either Positive | 1.119 | 1.079 [1.011, 1.155] | 0.279 | False |
| tau_cell_type | Phase 2 Progression | 1.269 | 1.198 [1.093, 1.302] | 0.219 | False |
| bimodality_score | Phase II+ | 1.193 | 1.073 [0.9831, 1.153] | 0.0149 | True |
| bimodality_score | Phase III+ | 1.356 | 1.138 [0.9988, 1.265] | 0.00995 | True |
| bimodality_score | Phase IV | 1.311 | 1.154 [0.9747, 1.331] | 0.129 | False |
| bimodality_score | Terminated | 0.812 | 0.9359 [0.8808, 1.001] | 0.00498 | True |
| bimodality_score | Withdrawn | 0.8849 | 0.9867 [0.915, 1.057] | 0.00995 | True |
| bimodality_score | Suspended | 0.7012 | 0.8799 [0.7571, 1.027] | 0.0199 | True |
| bimodality_score | Primary Positive | 1.126 | 1.068 [0.9994, 1.139] | 0.144 | False |
| bimodality_score | Secondary Positive | 1.134 | 1.073 [1.009, 1.151] | 0.139 | False |
| bimodality_score | Either Positive | 1.138 | 1.072 [1.007, 1.143] | 0.109 | False |
| bimodality_score | Phase 2 Progression | 1.181 | 1.176 [1.078, 1.266] | 0.905 | False |

**Interpretation.**

1. **Number of targets.** Adjusting for log(number of targets per trial) shrinks every
   association: for example, tau and Phase I→II goes from OR 1.27 to 1.12, and bimodality
   and Phase I→II from 1.18 to 1.04 (p = 0.09, no longer significant). The phase,
   endpoint and termination associations remain significant. MIN aggregation makes a
   trial's feature depend on how many targets it has, and multi-target trials differ
   systematically in outcome.
2. **Gene-label permutation.**
   * The paper's outcome-permutation test (the authors' 1,000 shuffles, reproduced
     exactly above) is centred on OR = 1 by construction. Shuffling the feature values
     *across genes* instead keeps the trial → target structure.
   * Under that null, an uninformative gene property already shows ORs of 1.07–1.20 for
     the phase, endpoint and Phase I→II outcomes. That is the size of the published
     associations.
   * **Tau:** only Phase III+ exceeds this null (p = 0.05). The primary-endpoint OR of
     1.12 (null median 1.07, 95% range 1.01–1.15) and the Phase I→II OR of 1.27 (null
     1.20, range 1.09–1.30) do not.
   * **Bimodality:** Phase II+/III+ and the three stopped-status outcomes survive.
     Endpoint and Phase I→II do not.
   * In short, a large part of the reported single-cell–outcome associations is
     explained by the trial-to-target aggregation itself. The paper's permutation test
     cannot detect that, because it breaks the aggregation along with the signal.
   * These are harness additions (`replicate.gene_permutation_null`,
     `harness_checks`); they are not part of the authors' analysis.

## 3. Table S2: competitor annotation agreement (`vbt case1 benchmarks`)

**What the archive contains.** `clinical_trials/benchmarks/{Biomni,Kosmos,PantheonOS}/` holds:

* `manual_review_set_annotations.csv`: 89–100 rows of `nct_id, primary, secondary,
  ae_binary`. This is the 100-trial manual-review set (50 Phase II + 50 Phase III).
* `tdc_annotations.csv`: 482–500 trials overlapping the TDC labels.
* 1,780 per-trial JSON files (not used).

**What it does not contain.** The archive ships **neither the two annotators'
manual-review labels nor the TDC/HINT labels.** By default,
`benchmarks.benchmark_agreement()` therefore scores each system against the **Virtual
Biotech's reconciled labels** on the same trials. The `reference` column says so, and
`--manual` / `--tdc` take the real ground truth when you have it; with it, the Virtual
Biotech itself is also scored.

**Applicability rules** (paper Methods):

* Endpoints are reduced to "positive" vs not.
* Endpoints are not counted for trials stopped early. On the archive's manual-review set
  this leaves **86** applicable trials, matching the paper's 76/86 denominator.
* AE agreement needs a reference value (`ae_has_safety_signals` for the Virtual Biotech
  labels). The paper's "61/66 exact statistics" denominator cannot be rebuilt from the
  shipped columns.
* A missing system answer counts as "not positive". `agreement_answered` and `coverage`
  report the alternative view.

`vbt case1 benchmarks` on the archive:

| trial set | reference | field | system | agreement | n agree / applicable | coverage | agreement (answered only) |
|---|---|---|---|---|---|---|---|
| manual_review | virtual_biotech_labels | ae_binary | Biomni | 61.6% | 53/86 | 90% | 62.3% |
| manual_review | virtual_biotech_labels | ae_binary | Kosmos | 60.5% | 52/86 | 100% | 60.5% |
| manual_review | virtual_biotech_labels | ae_binary | PantheonOS | 62.8% | 54/86 | 100% | 62.8% |
| manual_review | virtual_biotech_labels | primary | Biomni | 69.8% | 60/86 | 87% | 74.7% |
| manual_review | virtual_biotech_labels | primary | Kosmos | 47.7% | 41/86 | 55% | 57.4% |
| manual_review | virtual_biotech_labels | primary | PantheonOS | 75.6% | 65/86 | 97% | 77.1% |
| manual_review | virtual_biotech_labels | secondary | Biomni | 62.8% | 54/86 | 87% | 66.7% |
| manual_review | virtual_biotech_labels | secondary | Kosmos | 38.4% | 33/86 | 55% | 46.8% |
| manual_review | virtual_biotech_labels | secondary | PantheonOS | 69.8% | 60/86 | 97% | 71.1% |
| tdc | virtual_biotech_labels | primary | Biomni | 83.6% | 418/500 | 96% | 86.1% |
| tdc | virtual_biotech_labels | primary | Kosmos | 61.8% | 309/500 | 80% | 70.5% |
| tdc | virtual_biotech_labels | primary | PantheonOS | 84.4% | 422/500 | 97% | 85.6% |

These are agreements with the Virtual Biotech's labels, not accuracies. They support the
paper's ordering (every competitor disagrees with the Virtual Biotech labels on a sizeable
share of trials). Without the ground truth they cannot say which system is right.

## 4. Case 2/3: fidelity of `vbt.analysis` with the authors' code

For each analysis we compared our module with the authors' scripts (`b7-h3/code/**`,
`osmr/code/*.py`). We fixed the defaults to match their code and kept the Methods-text
variants selectable.

| Analysis | Authors' code | Change in `vbt.analysis` |
|---|---|---|
| **Survival** (`survival_analysis.py`) | Complete covariates (age, AJCC stage I/II vs III/IV, sex). Each endpoint with < 1 month follow-up is set to missing, and patients without any endpoint are dropped. **MSI-H (MSIsensor ≥ 3.5) is excluded.** Quartiles (top ≥ q75, bottom ≤ q25) are computed once on that cohort. Multivariable Cox (lifelines, Efron ties) on `high + age + stage_advanced + sex_male`. | `quartile_cox_all_endpoints(min_followup=1.0, msi_high=3.5)` (defaults; `None` turns them off). `prepare_tcga_clinical` adds `msi_sensor`. New `load_cbioportal_raw()` reads the authors' `raw_data.csv`. |
| **LIANA** (`liana_03b_*.py`) | Each run contains the fibroblasts of one state plus 7 immune types only. Expressing fibroblasts are compared with non-expressing ones in LUAD (CP10K + log1p > 0), and top vs bottom quartile in SCLC. `rank_aggregate(n_perms=1000, return_all_lrs=True)`. Filter: magnitude_rank ≤ q10 **and specificity_rank ≤ q10** and CellPhoneDB p < 0.01 and ≥ 3 of 5 methods "detecting" (p < 1 or score ≠ 0). The filter runs on the **whole** table; fibroblast→immune is taken afterwards. | `filter_lr_results(method_rules="authors", specificity_top_frac=0.10)` (defaults; `method_rules="harness"` keeps the earlier rules). `ligand_receptor_contrast(immune_celltypes=..., restrict="after_filter")` builds the authors' cell sets and returns `a_only_focal_to_partner`. |
| **Pseudobulk DE** (`de_analysis_*.py`) | ≥ 50 cells per donor (for both eligibility and pseudobulk), ≥ 3 donors per condition. Each pseudobulk is given its majority 10x assay. Assay levels with < 3 pseudobulks are collapsed to "other", and "other" is dropped if it is still < 3. `~ condition + assay_grp` is used only when identifiable (≥ 2 levels and ≥ 1 level spanning both conditions). PyDESeq2 with Cook's refit; padj < 0.05 and abs(log2FC) > 1. | `MIN_CELLS_PER_DONOR = 50` is now the default of `eligible_celltypes` and `pseudobulk_counts` (it was 20). `pseudobulk_counts(majority_cols=["assay"])`. New `assay_design()`, whose covariates go into `pseudobulk_de(covariates=...)`. |
| **Spatial immune exclusion** (`05_lmer_cd276_exclusion.py`) | All LUAD tumour spots. CD276 as CP10K of raw counts. Per sample, among expressing spots: high = **> q75**, low ≤ q25; skip the sample if < 25 expressing spots or < 3 per group. Rings 1–6, 7–15, 16–30. `lmerTest` REML `Y ~ high + z(fib) + z(epi) + z(endo) + z(UMI) + (1|patient/sample)`. | `immune_neighborhood_analysis(high_rule="gt", min_per_group=3)` (defaults). Model and covariates already matched. |
| **TF screen** (`01c_tf_activity_screen.py`) | decoupler ULM/MLM on CollecTRI (`tmin = 5`), post-treatment cells. `lmer(TF ~ Remission + (1|sample_id), REML=FALSE)`, ≥ 4 samples per cell type, BH within cell type. | `tf_mixed_model_screen(reml=False, min_samples=4)` (defaults). |
| **STAT1 dominance / LMG** (`02b_stat1_dominance.py`) | Non-Remission post-treatment pseudobulks in 10 stromal cell states. All 7 receptors (≥ 7 patients) or 4 receptors (fewer patients), constant ones dropped, z-scored. LMG on **ML** `lmer` marginal R² (MuMIn: R `var`, ddof = 1). Patient cluster bootstrap with OLS on unscaled receptors, `default_rng(42)`, 2,000 draws. | `marginal_r2_mixed`/`lmg_shares(reml=False, var_ddof=1)` (defaults; it was REML with ddof = 0). New `lmm_random_intercept_ml` (profiled ML, robust on the 11-sample designs where statsmodels fails). `bootstrap_lmg(seed=42)`. New `stat1_dominance()` runs the whole table. |
| **Bulk biomarker AUCs** (`03a/03b`) | Processed cohorts (baseline UC infliximab, R/NR; GSE73661 uses mucosal healing). Equal-weight within-cohort z-score composite. **A score is negated when its AUC < 0.5** (not in the Methods text). | `compare_scores_across_cohorts(orientation="authors")` (default; `"fixed"` turns the flip off; `flipped` column). New `load_processed_cohort()` and `AUTHORS_IFX_COHORTS`. |
| **Cross-disease** (`04b_cross_disease_lmm.py`) | Drop non-UMI assays and pseudobulks > 50,000 counts per cell; `log_depth` = log10(counts per cell); assay groups 10x 3′, 10x 5′ and other. Per disease, donors present in both arms are removed from the normal arm. ≥ 5 donors per arm. `lmerTest` REML `y ~ condition + log_depth [+ assay_group] + (1|dataset_id)`, weights = n_cells, Satterthwaite df. `lm` is used when there is one dataset. Global BH. | New `prepare_cross_disease()`, `cross_disease_tests()` and `weighted_lmm_reml()` (a Python port of weighted REML `lmer` with Satterthwaite df). `disease_vs_normal_lmm` is unchanged. |

### 4.1 Numeric checks on shipped intermediates

| Analysis | Reference value (authors' output) | Ours | Match? | Notes |
|---|---|---|---|---|
| Survival OS, multivariable Cox HR (top vs bottom CD276 quartile) (Fig. 4F) | 1.6194 [1.054, 2.489], p = 0.0279, n = 238, 95 events | 1.6194 [1.054, 2.489], p = 0.0279, n = 238, 95 events | yes | Without the authors' QC (Methods-text cohort): HR 1.615, n = 243; paper: HR 1.62, p = 0.028 |
| Survival PFS, multivariable Cox HR (top vs bottom CD276 quartile) (Fig. 4F) | 1.3853 [0.930, 2.063], p = 0.1088, n = 235, 101 events | 1.3853 [0.930, 2.063], p = 0.1088, n = 235, 101 events | yes | Without the authors' QC (Methods-text cohort): HR 1.288, n = 243 |
| Survival DSS, multivariable Cox HR (top vs bottom CD276 quartile) (Fig. 4F) | 1.6253 [0.936, 2.821], p = 0.0843, n = 226, 57 events | 1.6253 [0.936, 2.821], p = 0.0843, n = 226, 57 events | yes | Without the authors' QC (Methods-text cohort): HR 1.510, n = 230 |
| Survival DFS, multivariable Cox HR (top vs bottom CD276 quartile) (Fig. 4F) | 2.0553 [1.084, 3.897], p = 0.0273, n = 142, 43 events | 2.0553 [1.084, 3.897], p = 0.0273, n = 142, 43 events | yes | Without the authors' QC (Methods-text cohort): HR 1.822, n = 144; paper: HR 2.06, p = 0.027 |
| Biomarker AUCs, 4 cohorts × {OSMR, gp130-axis, Arijs} (Fig. 5C) | e.g. GSE16879 0.773 / 0.914 / 0.914; GSE23597 0.651 / 0.834 / 0.846 (validation report) | 12/12 identical to 3 decimals (e.g. 0.773438 / 0.914062 / 0.914062) | yes | `osmr/code/data/GSE*.h5ad`; no score needed flipping; n = 24/23/32/23 |
| LIANA CD276-high-specific interactions, LUAD | 793 specific; **226** fibroblast→immune | 793; 226 (from the 190,561-row raw table: 5,631 / 4,873 consensus rows as archived) | yes | Raw tables fetched with `vbt data zenodo fetch` |
| LIANA, SCLC | 719 specific; **180** fibroblast→immune | 719; 180 (7,334 / 6,674 consensus rows) | yes | The specificity criterion does not bind on these tables |
| LMG shares, STAT1 ~ 7 receptors (Fig. 5B), 10 cell states, 68 rows | `stat1_dominance_lmg_shares.tsv` | 62/68 shares within 1e-6 (9 cell states, max abs Δ 4e-7); bootstrap CIs within 3e-8 for all 68 | 9/10 cell states | THY1⁺FAP⁺PDPN⁺ fibroblasts (n = 11 samples, 7 patients, 6 receptors): shares differ by ≤ 0.0093 and the full-model R² matches (0.4541). Some of lme4's 63 sub-model fits on 11 rows are singular or failed (the authors' code skips NaN sub-models); our profiled ML fits them all. |
| Cross-disease LMM (Fig. 5D): UC, Crohn's disease, atherosclerosis, 308 tests | `tests_lmm_v2.csv` (lmerTest) | 308/308 tests selected; log2FC and SE: 307/308 within 1e-4 relative (median 4e-9); Satterthwaite df within 1e-3 for 307/308; significance at 0.05 agrees 308/308 | yes | One test (Crohn's, ILC, OSM: σ²_study ≈ 9e-6) has df 3.41 vs 3.27. Global FDR needs all 130 disease files. |
| Pseudobulk DE (`de_summary.csv`, `de_design_per_cell_type.csv`) | Design per cell type, n_up / n_down | Not re-run | n/a | Needs the 2.5 GB scRNA-seq h5ad and PyDESeq2. The design logic is ported (`assay_design`) and unit-tested. |
| Spatial `lmer_cd276_exclusion.csv` (37–69% depletion) | Per immune type and ring | Not re-run | n/a | Needs the E-MTAB-13530 Visium data and cell2location (GPU). Grouping rule fixed as above. |

Skip-if-absent tests: `tests/test_analysis_fidelity.py`. They check survival, AUCs,
LIANA counts, LMG shares, and cross-disease when the inputs are present.

## 5. Reproducing

```bash
export VBT_ZENODO_DIR=$PWD/data/zenodo
vbt data zenodo fetch --preset case1
vbt data zenodo fetch --preset b7h3-results
vbt data zenodo fetch --preset osmr-results
vbt case1 replicate --out results/case1_replication   # ~20 min on 4 CPUs without R (beta permutations + GLMMs dominate)
vbt case1 benchmarks
VBT_UPSTREAM=third_party/TheVirtualBiotech python -m pytest -q tests/test_replicate.py tests/test_analysis_fidelity.py
```
