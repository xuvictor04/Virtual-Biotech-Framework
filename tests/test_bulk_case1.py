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


# ---------------------------------------------------------------- bulk budgets, cost and fatal errors

import pytest  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from vbt.agents import AgentDefinition  # noqa: E402
from vbt.bulk import BulkItem  # noqa: E402
from vbt.providers.base import ProviderError, StopReason  # noqa: E402
from vbt.providers.mock import fail, turn  # noqa: E402


class _Out(BaseModel):
    value: int


def _agent():
    return AgentDefinition(name="bulk-probe", description="probe", prompt="Answer.", tier="bulk", tools=[],
                           memory="none")


def _item_id(messages):
    return messages[0].text.split("ITEM ")[1].split()[0]


def _items(n):
    return [BulkItem(f"i{k:02d}", f"ITEM i{k:02d} please") for k in range(n)]


def _paid_submit(messages):
    return turn(call("submit_result", value=1), cost_usd=1.0)


async def _session(config, script, **limits):
    config["limits"].update(limits)
    return await open_session(config, provider=ScriptedProvider(script), start_mcp=False)


def _recs(out):
    return {r["id"]: r for r in map(json.loads, out.read_text().splitlines())}


async def test_bulk_not_capped_by_turn_budget(config, tmp_path):
    session = await _session(config, lambda a, s, m, t: _paid_submit(m), max_turn_cost_usd=3)
    out = tmp_path / "b.jsonl"
    # even when started inside a CSO turn scope the per-turn cap never applies
    with session.rt.cost_scope("turn", 3) as turn_scope:
        summary = await BulkRunner(session.rt, _agent(), _Out, out, concurrency=3).run(_items(10))
    assert summary["completed"] == 10 and summary["failed"] == 0
    assert summary["budget_spent"] == pytest.approx(10.0)
    assert turn_scope.spent == pytest.approx(10.0)  # accounted to the turn afterwards
    # submit_result is terminal: exactly one provider call per item
    assert len(session.rt.provider.calls) == 10
    await session.close()


async def test_bulk_budget_stops_without_writing_unrun_items(config, tmp_path):
    session = await _session(config, lambda a, s, m, t: _paid_submit(m))
    out = tmp_path / "b.jsonl"
    summary = await BulkRunner(session.rt, _agent(), _Out, out, concurrency=1, budget_usd=4).run(_items(10))
    recs = _recs(out)
    assert summary["budget_stop"] is True
    assert len(recs) == 4 and all(r["ok"] for r in recs.values())
    assert summary["unrun"] == 6 and summary["budget_spent"] == pytest.approx(4.0)
    await session.close()


async def test_bulk_item_budget_fails_item_without_retry(config, tmp_path):
    def script(agent, system, messages, tools):
        if _item_id(messages) == "i00":
            return turn(call("submit_result", value="bad"), cost_usd=2.0)  # invalid -> would loop
        return _paid_submit(messages)

    session = await _session(config, script)
    out = tmp_path / "b.jsonl"
    summary = await BulkRunner(session.rt, _agent(), _Out, out, concurrency=1, max_item_cost_usd=1.5,
                               prewarm="none").run(_items(3))
    recs = _recs(out)
    assert recs["i00"]["ok"] is False and recs["i00"]["status"] == "item_budget"
    assert recs["i00"]["attempts"] == 1 and recs["i00"]["cost_usd"] == pytest.approx(2.0)
    assert summary["completed"] == 2
    await session.close()


async def test_bulk_auth_error_is_batch_fatal(config, tmp_path):
    session = await _session(config, lambda a, s, m, t: fail(ProviderError("authentication_error: bad key")))
    out = tmp_path / "b.jsonl"
    with pytest.raises(ProviderError) as ei:
        await BulkRunner(session.rt, _agent(), _Out, out, concurrency=4).run(_items(20))
    assert "fatal_error" in ei.value.bulk_summary
    assert not out.exists() or out.read_text().strip() == ""
    assert len(session.rt.provider.calls) == 1  # prewarm item fails: no retries, no fan-out
    await session.close()


async def test_bulk_failed_attempt_cost_and_success_median(config, tmp_path):
    seen: dict[str, int] = {}

    def script(agent, system, messages, tools):
        iid = _item_id(messages)
        seen[iid] = seen.get(iid, 0) + 1
        if iid in ("i00", "i01"):
            return fail(RuntimeError("boom"))  # zero-cost failures
        if iid == "i02" and seen[iid] == 1:
            return turn(call("Nope"), cost_usd=1.0)  # paid call, then the next call raises
        if iid == "i02" and seen[iid] == 2:
            return fail(RuntimeError("dropped"))
        return turn(call("submit_result", value=1), cost_usd=1.0 if iid == "i02" else 3.0)

    session = await _session(config, script)
    out = tmp_path / "b.jsonl"
    summary = await BulkRunner(session.rt, _agent(), _Out, out, concurrency=1, prewarm="none").run(_items(4))
    recs = _recs(out)
    assert recs["i02"]["ok"] and recs["i02"]["cost_usd"] == pytest.approx(2.0)  # failed attempt included
    assert recs["i00"]["ok"] is False and recs["i00"]["cost_usd"] == 0
    assert summary["median_cost_usd"] == pytest.approx(2.5)   # over successes (2, 3)
    assert summary["cost_all"]["median_usd"] == pytest.approx(1.0)  # over all (0, 0, 2, 3)
    assert summary["cost_successful"]["total_usd"] == pytest.approx(5.0)
    await session.close()


async def test_bulk_prewarm_runs_first_item_alone(config, tmp_path):
    order: list[str] = []

    def script(agent, system, messages, tools):
        order.append(_item_id(messages))
        n = sum(1 for m in messages for b in m.content if b.type == "tool_call")
        if n == 0:
            return reply(call("submit_result", value="x"))  # invalid first: two calls per item
        return reply(call("submit_result", value=1))

    session = await _session(config, script)
    out = tmp_path / "b.jsonl"
    summary = await BulkRunner(session.rt, _agent(), _Out, out, concurrency=4).run(_items(4))
    assert summary["completed"] == 4 and summary["prewarm"] == "first_item"
    assert order[:2] == ["i00", "i00"]
    await session.close()


async def test_bulk_records_refusals(config, tmp_path):
    def script(agent, system, messages, tools):
        if _item_id(messages) == "i00":
            return turn("I can't help with that.", stop=StopReason.REFUSAL)
        return _paid_submit(messages)

    session = await _session(config, script)
    out = tmp_path / "b.jsonl"
    summary = await BulkRunner(session.rt, _agent(), _Out, out, concurrency=1, prewarm="none").run(_items(2))
    recs = _recs(out)
    assert recs["i00"]["refusals"] == 1 and recs["i00"]["attempts"] == 1
    assert summary["refusals"] == 1 and summary["refused_items"] == ["i00"]
    await session.close()


def test_annotator_protocol_and_truncation_guidance(config, tmp_path):
    base = ann.annotator_agent(config)
    assert "QueryToolOutput" in base.prompt and "adverseEvents" in base.prompt
    assert base.memory == "none"
    run = tmp_path / "run"
    res = run / "work" / "clinical-trialist" / "results" / "reports"
    res.mkdir(parents=True)
    (res / "annotation_protocol.md").write_text("# Designed protocol\nUse the cascade.\n")
    (res / "annotation_schema.json").write_text(json.dumps(
        {"type": "object", "properties": {"nct_id": {"type": "string"}}, "required": ["nct_id"]}))
    text, model, paths = ann.load_protocol(run)
    assert "Designed protocol" in text and paths["protocol"].endswith("annotation_protocol.md")
    assert model is not None and model.model_validate({"nct_id": "N1"}).model_dump()["nct_id"] == "N1"
    agent = ann.annotator_agent(config, protocol=text)
    assert agent.prompt.startswith("# Designed protocol") and "submit_result" in agent.prompt
    assert "QueryToolOutput" in agent.prompt
    text2, model2, _ = ann.load_protocol(res / "annotation_protocol.md")
    assert text2 == text and model2 is not None
