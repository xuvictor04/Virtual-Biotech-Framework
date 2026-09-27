"""Case study 1, step 1: massively parallel trial-outcome annotation.

Reproduces "Massively parallel, source-tracked extraction of trial outcomes":
the virtual CSO dispatched one clinical trialist agent per NCT ID for the
Phase II/III trials of the Open Targets clinical-trial dataset; each followed a
three-tier evidence cascade (ClinicalTrials.gov -> PubMed -> press releases)
and returned Pydantic-validated JSON. Median cost in the paper: $0.23/trial.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ...agents import AgentDefinition
from ...bulk import BulkItem, BulkRunner, load_results
from ...runtime import Runtime
from .schema import TrialAnnotation

HERE = Path(__file__).resolve().parent
STOPPED_OR_DONE = ("Completed", "Terminated", "Withdrawn", "Suspended")

ANNOTATOR_TOOLS = [
    "mcp__clinicaltrials__get_clinical_trial_details",
    "mcp__pubmed__search_pubmed",
    "mcp__pubmed__fetch_abstracts",
    "WebSearch",
    "WebFetch",
    "TodoWrite",
]


def annotator_agent(config: dict[str, Any]) -> AgentDefinition:
    tools = list(ANNOTATOR_TOOLS)
    if not config.get("web", {}).get("enabled", True):
        tools = [t for t in tools if t not in ("WebSearch", "WebFetch")]
    return AgentDefinition(
        name="trial-annotator",
        description="Clinical trialist agent assigned to a single NCT ID (bulk annotation).",
        prompt=(HERE / "annotator_prompt.md").read_text(),
        tier="bulk",
        tools=tools,
        division="Clinical Officers",
        role="Clinical trialist agent (single-trial annotation)",
    )


def mapping_path(config: dict[str, Any]) -> Path:
    return Path(config["vars"]["upstream"]) / "datasets" / "clinical_trials" / "chembl_clinical_nct_data.parquet"


def labels_path(config: dict[str, Any]) -> Path:
    return Path(config["vars"]["upstream"]) / "datasets" / "clinical_trials" / "clinical_trial_labels_reconciled.csv"


def select_trials(mapping: pd.DataFrame, phases=(2.0, 3.0), statuses=STOPPED_OR_DONE,
                  sample: int | None = None, seed: int = 0, ids: list[str] | None = None) -> pd.DataFrame:
    """One row per trial with its drugs, targets and indications for the prompt."""
    m = mapping[mapping["phase"].isin(phases) & mapping["status"].isin(statuses)]
    if ids:
        m = m[m["nct_id"].isin(ids)]
    trials = (m.groupby("nct_id")
               .agg(phase=("phase", "max"), status=("status", "first"), start=("trial_date", "min"),
                    drugs=("drugId", lambda s: sorted(set(s))),
                    targets=("targetFromSourceId", lambda s: sorted(set(map(str, s)))),
                    indications=("disease_name", lambda s: sorted(set(map(str, s)))),
                    stop_reason=("studyStopReason", "first"))
               .reset_index())
    if sample:
        trials = trials.sample(n=min(sample, len(trials)), random_state=seed)
    return trials


def trial_prompt(row: pd.Series) -> str:
    stop = row["stop_reason"] if isinstance(row["stop_reason"], str) else "none recorded"
    return (
        f"NCT ID: {row['nct_id']}\nPhase: {row['phase']:g}\nRegistry status (Open Targets snapshot): {row['status']}\n"
        f"Start date: {row['start']}\nDrugs (ChEMBL): {', '.join(row['drugs'])}\n"
        f"Targets (UniProt): {', '.join(row['targets'])}\nIndications: {', '.join(row['indications'])}\n"
        f"Stated stop reason: {stop}\n\nFollow the cascade and call submit_result."
    )


async def annotate(runtime: Runtime, trials: pd.DataFrame, out_path: Path, *, concurrency: int = 64,
                   budget_usd: float | None = None, on_progress=None) -> dict[str, Any]:
    agent = annotator_agent(runtime.config)
    items = [BulkItem(r["nct_id"], trial_prompt(r), {"phase": r["phase"], "status": r["status"]})
             for _, r in trials.iterrows()]
    runner = BulkRunner(runtime, agent, TrialAnnotation, out_path, concurrency=concurrency,
                        budget_usd=budget_usd, on_progress=on_progress)
    return await runner.run(items)


def results_to_labels(results_path: Path) -> pd.DataFrame:
    """Convert bulk JSONL output into the released-label column layout."""
    rows = []
    for rec in load_results(results_path):
        if rec.get("ok"):
            row = TrialAnnotation.model_validate(rec["result"]).to_label_row()
            row["cost_usd"] = rec.get("cost_usd")
            rows.append(row)
    return pd.DataFrame(rows)
