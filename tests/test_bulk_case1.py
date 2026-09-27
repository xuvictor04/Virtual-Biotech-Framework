"""Bulk per-trial annotation with the scripted provider, plus Case 1 helpers."""

import json

import pandas as pd

from vbt.bulk import BulkRunner, load_results
from vbt.case_studies.trial_outcomes import annotate as ann
from vbt.case_studies.trial_outcomes.phase1 import phase1_progression
from vbt.case_studies.trial_outcomes.schema import TrialAnnotation
from vbt.case_studies.trial_outcomes.validation import agreement_report
from vbt.orchestrator import open_session
from vbt.providers.mock import ScriptedProvider, call, reply


def _record(nct, primary="POSITIVE"):
    return dict(nct_id=nct, overall_status="Completed", primary_endpoint_result=primary,
                primary_endpoint_rationale="met", secondary_endpoint_result="UNKNOWN",
                secondary_endpoint_rationale="n/a", results_source="ClinicalTrials.gov",
                ae_source="ClinicalTrials.gov", tiers_consulted=["ClinicalTrials.gov"], confidence="high",
                serious_ae_pct={"cardiac": 2.5})


def _mapping():
    return pd.DataFrame({
        "nct_id": ["NCT00000001", "NCT00000002", "NCT00000003", "NCT00000004"],
        "drugId": ["D1", "D1", "D2", "D1"], "targetId": ["T1", "T1", "T2", "T1"],
        "diseaseId": ["E1", "E1", "E2", "E1"], "phase": [1.0, 2.0, 2.0, 3.0],
        "status": ["Completed"] * 4, "trial_date": ["2010-01-01", "2012-01-01", "2012-01-01", "2014-01-01"],
        "disease_name": ["d1", "d1", "d2", "d1"], "targetFromSourceId": ["P1", "P1", "P2", "P1"],
        "studyStopReason": [None] * 4,
    })


async def test_bulk_annotation_validates_and_resumes(config, tmp_path):
    def script(agent, system, messages, tools):
        assert "submit_result" in [t.name for t in tools]
        nct = messages[0].text.split("NCT ID: ")[1].split()[0]
        n_submits = sum(1 for m in messages for b in m.content if b.type == "tool_call")
        if n_submits == 0:  # first try: invalid (bad PMID) -> validation error returned to agent
            return reply(call("submit_result", **{**_record(nct), "pubmed_ids": ["abc"]}))
        if n_submits == 1:
            return reply(call("submit_result", **_record(nct, "NEGATIVE" if nct.endswith("3") else "POSITIVE")))
        return reply("done")

    session = await open_session(config, provider=ScriptedProvider(script), start_mcp=False)
    trials = ann.select_trials(_mapping(), phases=(2.0, 3.0))
    assert set(trials["nct_id"]) == {"NCT00000002", "NCT00000003", "NCT00000004"}
    out = tmp_path / "ann.jsonl"
    summary = await ann.annotate(session.rt, trials, out, concurrency=2)
    assert summary["completed"] == 3 and summary["failed"] == 0
    # resumable: nothing re-queued
    again = await ann.annotate(session.rt, trials, out, concurrency=2)
    assert again["queued"] == 0 and again["skipped_existing"] == 3
    labels = ann.results_to_labels(out)
    assert labels.set_index("nct_id").loc["NCT00000003", "primary_endpoint_result"] == "NEGATIVE"
    assert labels["ae_serious_cardiac_pct"].eq(2.5).all()
    await session.close()


def test_phase1_progression():
    res = phase1_progression(_mapping()).set_index("nct_id")
    assert res.loc["NCT00000001", "phase2_progression"] == "EVER"


def test_agreement_report():
    pred = pd.DataFrame({"nct_id": ["A", "B", "C"], "primary_endpoint_result": ["POSITIVE", "NEGATIVE", "UNKNOWN"],
                         "ae_serious_cardiac_pct": [1.0, 5.0, None]})
    ref = pd.DataFrame({"nct_id": ["A", "B", "C"], "primary_endpoint_result": ["POSITIVE", "POSITIVE", None],
                        "ae_serious_cardiac_pct": [1.2, 9.0, None]})
    rep = agreement_report(pred, ref).set_index("field")
    assert rep.loc["primary_endpoint_result", "n_applicable"] == 2
    assert rep.loc["primary_endpoint_result", "agreement"] == 0.5
    assert rep.loc["serious_ae_rates", "n_agree"] == 1


def test_schema_roundtrip():
    rec = TrialAnnotation.model_validate(_record("NCT12345678"))
    row = rec.to_label_row()
    assert row["primary_endpoint_result"] == "POSITIVE" and row["ae_serious_cardiac_pct"] == 2.5
