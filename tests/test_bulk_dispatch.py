"""BulkDispatch / BulkStatus: pilot projection, confirm gate, background run (offline)."""

import asyncio
import json

import pytest

from vbt.bulk import bulk_settings
from vbt.bulk_dispatch import bulk_dispatch_tools, render_prompt, resolve_schema
from vbt.orchestrator import open_session
from vbt.providers.mock import ScriptedProvider, call, turn
from vbt.tools.base import ToolContext, ToolFailure


def _script(agent, system, messages, tools):
    assert agent == "clinical-trialist-bulk"
    assert "PROTOCOL-MARKER" in system  # protocol appended to the agent prompt
    nct = messages[0].text.split("Trial ")[1].split()[0]
    return turn(call("submit_result", nct_id=nct, verdict="POSITIVE"), cost_usd=0.5)


async def _setup(config, tmp_path):
    session = await open_session(config, provider=ScriptedProvider(_script), start_mcp=False)
    run = session.run.dir
    (run / "work" / "clinical-trialist" / "results").mkdir(parents=True, exist_ok=True)
    items = run / "work" / "clinical-trialist" / "results" / "trials.csv"
    items.write_text("nct_id,phase\n" + "\n".join(f"NCT{k:08d},{2 if k % 2 else 3}" for k in range(12)) + "\n")
    protocol = run / "work" / "clinical-trialist" / "results" / "protocol.md"
    protocol.write_text("# Protocol\nPROTOCOL-MARKER: follow the cascade.\n")
    schema = run / "work" / "clinical-trialist" / "results" / "schema.json"
    schema.write_text(json.dumps({"type": "object", "properties": {
        "nct_id": {"type": "string"}, "verdict": {"type": "string", "enum": ["POSITIVE", "NEGATIVE"]}},
        "required": ["nct_id", "verdict"]}))
    tools = {t.name: t for t in bulk_dispatch_tools(session.rt)}
    ctx = ToolContext(agent="cso", run=session.run, runtime=session.rt)
    args = dict(subagent_type="clinical-trialist", items_path="work/clinical-trialist/results/trials.csv",
                query="phase == 2", prompt_template="Trial {nct_id} (phase {phase})",
                protocol_path="work/clinical-trialist/results/protocol.md",
                schema="work/clinical-trialist/results/schema.json", budget_usd=10, pilot_size=2, concurrency=3)
    return session, tools, ctx, args


def test_defaults_and_helpers():
    s = bulk_settings({})
    assert s == {"dispatch_enabled": False, "dispatch_max_budget_usd": 50.0, "pilot_size": 5,
                 "prewarm": "first_item"}
    assert render_prompt("Trial {nct_id}", {"nct_id": "X"}) == "Trial X"
    with pytest.raises(ToolFailure):
        render_prompt("Trial {missing}", {"nct_id": "X"})
    assert resolve_schema("trial_annotation").__name__ == "TrialAnnotation"


async def test_pilot_then_confirm(config, tmp_path):
    session, tools, ctx, args = await _setup(config, tmp_path)
    dispatch, status = tools["BulkDispatch"], tools["BulkStatus"]

    with pytest.raises(ToolFailure, match="pilot first"):
        await dispatch(ctx, {**args, "confirm": True})

    pilot = await dispatch(ctx, args)
    assert pilot["state"] == "piloted" and len(pilot["pilot_results"]) == 2
    proj = pilot["projection"]
    assert proj["n_items"] == 6 and proj["n_remaining"] == 4
    assert proj["mean_cost_per_item_usd"] == pytest.approx(0.5)
    assert proj["projected_remaining_cost_usd"] == pytest.approx(2.0)
    assert proj["fits_budget"] is True
    assert len(session.rt.provider.calls) == 2

    started = await dispatch(ctx, {**args, "confirm": True})
    assert started["state"] == "running" and started["job_id"] == pilot["job_id"]
    job = session.rt._bulk_jobs[started["job_id"]]
    await asyncio.wait_for(job.task, 10)
    st = await status(ctx, {"job_id": started["job_id"]})
    assert st["state"] == "completed"
    assert st["summary"]["completed"] == 4 and st["summary"]["skipped_existing"] == 2
    assert len(session.rt.provider.calls) == 6  # pilot items reused, not rerun
    lines = [json.loads(l) for l in open(started["results_path"])]
    assert {r["id"] for r in lines if r["ok"]} == {f"NCT{k:08d}" for k in range(1, 12, 2)}
    assert started["results_path"].endswith(f"work/clinical-trialist/results/bulk/{started['job_id']}.jsonl")
    assert json.loads(open(started["summary_path"]).read())["state"] == "completed"
    await session.close()


async def test_budget_validation_and_schema_errors(config, tmp_path):
    session, tools, ctx, args = await _setup(config, tmp_path)
    dispatch = tools["BulkDispatch"]
    with pytest.raises(ToolFailure, match="dispatch_max_budget_usd"):
        await dispatch(ctx, {**args, "budget_usd": 1000})
    with pytest.raises(ToolFailure, match="read root"):
        await dispatch(ctx, {**args, "items_path": "/etc/passwd"})
    with pytest.raises(ToolFailure, match="unknown subagent_type"):
        await dispatch(ctx, {**args, "subagent_type": "nobody"})
    model = resolve_schema(str(session.run.dir / "work/clinical-trialist/results/schema.json"))
    with pytest.raises(ValueError):
        model.model_validate({"nct_id": "X", "verdict": "MAYBE"})
    assert model.model_validate({"nct_id": "X", "verdict": "NEGATIVE"}).model_dump()["verdict"] == "NEGATIVE"
    await session.close()


async def test_job_budget_is_cumulative_across_pilots_and_confirms(config, tmp_path):
    """budget_usd caps the job's total spend: re-piloting or re-confirming cannot reset it."""
    session, tools, ctx, args = await _setup(config, tmp_path)
    dispatch, status = tools["BulkDispatch"], tools["BulkStatus"]
    args = {**args, "budget_usd": 2.0, "concurrency": 1}  # $0.50 per item, 6 items selected

    pilot = await dispatch(ctx, args)  # 2 items -> $1.00
    job = session.rt._bulk_jobs[pilot["job_id"]]
    assert job.spent_usd == pytest.approx(1.0)
    assert pilot["projection"]["budget_remaining_usd"] == pytest.approx(1.0)
    assert pilot["projection"]["fits_budget"] is False  # 4 x $0.50 > $1.00 left

    again = await dispatch(ctx, args)  # pilot items already done: costs nothing, keeps the first spend
    assert again["projection"]["job_spent_usd"] == pytest.approx(1.0)
    assert job.spent_usd == pytest.approx(1.0)

    started = await dispatch(ctx, {**args, "confirm": True})
    await asyncio.wait_for(job.task, 10)
    st = await status(ctx, {"job_id": started["job_id"]})
    assert st["state"] == "stopped" and st["spent_usd"] == pytest.approx(2.0)
    calls = len(session.rt.provider.calls)
    assert calls == 4

    for extra in ({"confirm": True}, {}):  # neither a re-confirm nor a re-pilot may spend more
        with pytest.raises(ToolFailure, match="already spent"):
            await dispatch(ctx, {**args, **extra})
    assert len(session.rt.provider.calls) == calls

    # the spend survives a new session resuming the same job (persisted in summary_path)
    assert json.loads(open(started["summary_path"]).read())["spent_usd"] == pytest.approx(2.0)
    del session.rt._bulk_jobs[started["job_id"]]
    with pytest.raises(ToolFailure, match="already spent"):
        await dispatch(ctx, {**args, "confirm": True, "pilot_size": 0})
    # raising the cap lets it continue, but only by the difference
    started = await dispatch(ctx, {**args, "budget_usd": 2.5, "confirm": True, "pilot_size": 0})
    job = session.rt._bulk_jobs[started["job_id"]]
    await asyncio.wait_for(job.task, 10)
    assert len(session.rt.provider.calls) == calls + 1
    assert job.spent_usd == pytest.approx(2.5)
    await session.close()


async def test_pilot_walltime_projection_uses_per_item_latency(config, tmp_path, monkeypatch):
    session, tools, ctx, args = await _setup(config, tmp_path)
    import vbt.bulk_dispatch as bd

    real = bd.load_results

    def slow_records(path):  # every pilot item took 60 s
        return [{**r, "elapsed_s": 60.0} for r in real(path)]

    monkeypatch.setattr(bd, "load_results", slow_records)
    pilot = await tools["BulkDispatch"](ctx, {**args, "pilot_size": 2, "concurrency": 3})
    # 4 remaining items at concurrency 3 -> 2 waves of 60 s
    assert pilot["projection"]["projected_wall_time_s"] == pytest.approx(120.0)
    await session.close()
