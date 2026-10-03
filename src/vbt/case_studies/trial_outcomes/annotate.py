"""Case study 1, step 1: massively parallel trial-outcome annotation.

Reproduces "Massively parallel, source-tracked extraction of trial outcomes":
the virtual CSO dispatched one clinical trialist agent per NCT ID for the
Phase II/III trials of the Open Targets clinical-trial dataset; each followed a
three-tier evidence cascade (ClinicalTrials.gov -> PubMed -> press releases)
and returned Pydantic-validated JSON. Median cost in the paper: $0.23/trial.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from ...agents import AgentDefinition
from ...bulk import BulkItem, BulkRunner, load_results
from ...runtime import Runtime
from .schema import TrialAnnotation

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
STOPPED_OR_DONE = ("Completed", "Terminated", "Withdrawn", "Suspended")

#: Tools of the bulk trial annotator. ``QueryToolOutput`` is not listed because the
#: runtime adds it to every agent automatically: large CT.gov records are truncated
#: structure-aware (every top-level key kept, long arrays/strings cut with markers
#: naming the exact ``QueryToolOutput(path, json_path)`` call) and spilled to
#: ``logs/tool_outputs``, so the annotator can drill into ``adverseEvents`` and late
#: outcome sections that fall past the inline limit (annotator_prompt.md says how).
ANNOTATOR_TOOLS = [
    "mcp__clinicaltrials__get_clinical_trial_details",
    "mcp__pubmed__search_pubmed",
    "mcp__pubmed__fetch_abstracts",
    "WebSearch",
    "WebFetch",
    "TodoWrite",
]


SUBMIT_TAIL = (
    "\n\n## Harness notes\n\n"
    "- Large tool results are truncated in your context; markers name the exact "
    "`QueryToolOutput(path, json_path)` call that reads the rest (e.g. `json_path=$.adverseEvents`). "
    "Use it before treating any field as missing.\n"
    "- Your only deliverable is one `submit_result` call; it ends your task.\n"
)


def annotator_agent(config: dict[str, Any], protocol: str | None = None) -> AgentDefinition:
    """The bulk clinical-trialist agent.

    ``protocol``: a protocol designed by the clinical trialist in a
    ``trial_curation`` run (see :func:`load_protocol`); the reconstructed
    ``annotator_prompt.md`` is the fallback.
    """
    tools = list(ANNOTATOR_TOOLS)
    if not config.get("web", {}).get("enabled", True):
        tools = [t for t in tools if t not in ("WebSearch", "WebFetch")]
    prompt = protocol.rstrip() + SUBMIT_TAIL if protocol else (HERE / "annotator_prompt.md").read_text()
    return AgentDefinition(
        name="trial-annotator",
        description="Clinical trialist agent assigned to a single NCT ID (bulk annotation).",
        prompt=prompt,
        tier="bulk",
        tools=tools,
        division="Clinical Officers",
        role="Clinical trialist agent (single-trial annotation)",
        memory="none",
        prompt_ref="protocol" if protocol else str(HERE / "annotator_prompt.md"),
    )


def load_protocol(source: str | Path) -> tuple[str, type[BaseModel] | None, dict[str, str]]:
    """Protocol text (and JSON Schema model, if any) from a trial_curation run or a file.

    ``source`` is a markdown file, or a run directory in which the newest
    ``*protocol*.md`` under ``work/clinical-trialist/`` (else anywhere under
    ``work/``) is used, with a ``*schema*.json`` JSON Schema next to it or in
    the same agent's tree when present. Returns ``(text, model_or_None, paths)``.
    """
    from ...bulk_dispatch import resolve_schema

    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"protocol source not found: {src}")
    paths: dict[str, str] = {}
    if src.is_file():
        proto = src
        search = [src.parent]
    else:
        roots = [src / "work" / "clinical-trialist", src / "work"]
        cands: list[Path] = []
        for r in roots:
            if r.is_dir():
                cands = sorted((p for p in r.rglob("*protocol*.md") if p.is_file()),
                               key=lambda p: p.stat().st_mtime, reverse=True)
                if cands:
                    break
        if not cands:
            raise FileNotFoundError(f"no *protocol*.md under {src}/work (is this a trial_curation run?)")
        proto = cands[0]
        search = [proto.parent, *[r for r in roots if r.is_dir()]]
    paths["protocol"] = str(proto)
    model = None
    for d in search:
        schemas = sorted((p for p in d.rglob("*schema*.json") if p.is_file()),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        if schemas:
            model = resolve_schema(str(schemas[0]))
            paths["schema"] = str(schemas[0])
            break
    return proto.read_text(), model, paths


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
                   budget_usd: float | None = None, on_progress=None, protocol: str | None = None,
                   schema: type[BaseModel] | None = None, **runner_kw: Any) -> dict[str, Any]:
    """Bulk annotation; raises a non-retryable ProviderError (with ``.bulk_summary``)
    when the batch hits a fatal provider error."""
    agent = annotator_agent(runtime.config, protocol=protocol)
    items = [BulkItem(r["nct_id"], trial_prompt(r), {"phase": r["phase"], "status": r["status"]})
             for _, r in trials.iterrows()]
    runner = BulkRunner(runtime, agent, schema or TrialAnnotation, out_path, concurrency=concurrency,
                        budget_usd=budget_usd, on_progress=on_progress, **runner_kw)
    return await runner.run(items)


def results_to_labels(results_path: Path) -> pd.DataFrame:
    """Convert bulk JSONL output into the released-label column layout."""
    rows = []
    for rec in load_results(results_path):
        if rec.get("ok"):
            try:
                row = TrialAnnotation.model_validate(rec["result"]).to_label_row()
            except ValueError as exc:  # a protocol-specific schema that is not TrialAnnotation-shaped
                log.warning("result for %s is not a TrialAnnotation: %s", rec.get("id"), exc)
                continue
            row["cost_usd"] = rec.get("cost_usd")
            rows.append(row)
    return pd.DataFrame(rows)
