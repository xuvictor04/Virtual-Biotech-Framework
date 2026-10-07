"""Harness plumbing of the typed outcome channel (DATA_LAYER.md §11.1, §12, §15.1-15.2): what the
runtime does with a ``DataResult`` and a ``GatewayError``, failures exempting model-side kinds,
``shrink_json`` keeping ``_vbt``, late tools reaching the registry, tool-scoped readiness in
ListTools and the prompts, and ``start_mcp`` with the data layer enabled but no gateway."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from vbt import datalayer
from vbt import failures as fl
from vbt import runtime as runtime_mod
from vbt.agents import AgentDefinition, _unavailable_text, system_prompt_parts
from vbt.context import _SPILLED_RE
from vbt.datalayer.errors import ErrorKind, GatewayError, not_found_payload
from vbt.datalayer.record import DataProvenance, OrderInfo, ResultInfo, SourceInfo, TableInfo
from vbt.datalayer.result import DataResult, Header
from vbt.providers.mock import ScriptedProvider, call, reply
from vbt.tools.base import Tool, ToolContext, schema, shrink_json

TOOL = "mcp__fake__lookup"


# --------------------------------------------------------------------------- helpers


def _probe(tools=(TOOL,)):
    return AgentDefinition(name="probe", description="test agent", prompt="You are a test agent.",
                           tier="scientist", tools=list(tools))


def _prov(status="ok", *, coverage=None, statement=None, returned=2, total=2, nature=None, leakage=None):
    p = DataProvenance(tool=TOOL, server="fake", mode="enforce", served_by="upstream")
    p.source = SourceInfo(name="open_targets", release="25.09")
    p.tables = [TableInfo(name="target", fingerprint="fp1:sha256:abc")]
    p.result = ResultInfo(status=status, returned=returned, total=total, coverage=coverage,
                          coverage_statement=statement, truncated=returned != total,
                          order=OrderInfo(by="score", direction="desc", verified=True))
    p.evidence_nature = nature
    p.leakage = leakage
    return p.finalize()


def _result(status="ok", rows=None, *, full_rows=None, **kw):
    rows = [{"id": "ENSG00000169174", "score": 0.9}, {"id": "ENSG00000141510", "score": 0.5}] if rows is None \
        else rows
    prov = _prov(status, returned=len(rows), total=len(full_rows if full_rows is not None else rows), **kw)
    header = Header(status=status, source="open_targets@25.09", tables=["target"], returned=prov.result.returned,
                    total=prov.result.total, coverage=prov.result.coverage,
                    coverage_statement=prov.result.coverage_statement, prov=prov.id)
    full = {"rows": full_rows} if full_rows is not None else None
    return DataResult.build({"rows": rows}, header, provenance=prov, full_obj=full)


async def _run_tool(scripted_session, config, handler, **limits):
    provider = ScriptedProvider.from_rules({"probe": [reply(call(TOOL, gene="PCSK9")), reply("done")]})
    session = await scripted_session(config, provider, **limits)
    rt = session.rt
    tool = Tool(TOOL, "lookup", schema({"gene": {"type": "string"}}), handler, source="mcp:fake")
    history: list = []
    res = await rt.run_agent(_probe(), "look it up", depth=1, extra_tools=[tool], history=history)
    end = next(e for e in rt.run.events() if e["type"] == "tool_end" and e["tool"] == TOOL)
    text = next(b for m in history if m.role == "user" for b in m.content
                if getattr(b, "tool_call_id", None) == end["tool_use_id"]).content
    return session, res, end, text


# --------------------------------------------------------------------------- DataResult


async def test_data_result_records_status_provenance_and_header_first(config, scripted_session):
    async def handler(ctx, a):
        return _result("ok")

    session, res, end, text = await _run_tool(scripted_session, config, handler)
    rt = session.rt
    assert end["result_status"] == "ok" and not end["is_error"]
    dp = end["data_provenance"]
    assert dp["status"] == "ok" and dp["source"] == "open_targets@25.09" and dp["returned"] == 2
    assert dp["tables"] == [{"name": "target", "fingerprint": "fp1:sha256:abc"}]
    assert "coverage" in dp and "leakage_risk" in dp and "evidence_nature" in dp and dp["order_verified"] is True
    # the full vbt.dataprov/1 record, stamped with this call's tool_use_id
    path = rt.run.dir / "logs" / "data_provenance" / f"{end['tool_use_id']}.json"
    assert dp["record"] == f"logs/data_provenance/{path.name}"
    record = json.loads(path.read_text())
    assert record["schema"] == "vbt.dataprov/1" and record["tool_use_id"] == end["tool_use_id"]
    assert record["id"] == dp["prov"] and record["id"].startswith("dp_")
    # the model sees the gateway's text: the header is the first key
    assert text.startswith('{"_vbt": {') and json.loads(text)["_vbt"]["prov"] == dp["prov"]
    assert end["output"].startswith('{"_vbt"')
    # the live call index carries status, prov and coverage (§15.2)
    rec = rt.run.tool_call_status(end["tool_use_id"])
    assert rec["result_status"] == "ok" and rec["prov"] == dp["prov"]
    assert res.tool_calls == 1
    await session.close()


async def test_empty_result_carries_coverage_to_the_trace_and_index(config, scripted_session):
    async def handler(ctx, a):
        return _result("empty", rows=[], coverage="covered", statement="every target is assessed")

    session, _res, end, _text = await _run_tool(scripted_session, config, handler)
    assert end["result_status"] == "empty"
    assert end["data_provenance"]["coverage"] == "covered"
    assert end["data_provenance"]["coverage_statement"] == "every target is assessed"
    rec = session.rt.run.tool_call_status(end["tool_use_id"])
    assert rec["coverage"] == "covered" and rec["data_provenance"]["coverage"] == "covered"
    await session.close()


async def test_spill_keeps_the_unshrunk_payload_when_the_gateway_shrank_the_text(config, scripted_session):
    full = [{"id": f"ENSG{i:011d}", "score": 1 - i / 1000} for i in range(300)]

    async def handler(ctx, a):
        return _result("partial", rows=full[:3], full_rows=full)

    session, _res, end, text = await _run_tool(scripted_session, config, handler)
    rt = session.rt
    assert end["output_path"].startswith("logs/tool_outputs/")
    spilled = json.loads((rt.run.dir / end["output_path"]).read_text())
    assert list(spilled) == ["_vbt", "rows"] and spilled["rows"] == full     # the complete record
    # the model text is the gateway's short text, header first, truncation note last
    assert text.startswith('{"_vbt"') and json.loads(text.split("\n\n[Output truncated")[0])["rows"] == full[:3]
    m = _SPILLED_RE.search(text)
    assert m and Path(m.group(1)) == rt.run.dir / end["output_path"]
    await session.close()


async def test_large_data_result_is_shrunk_with_the_header_intact(config, scripted_session):
    rows = [{"id": f"ENSG{i:011d}", "text": "x" * 200} for i in range(400)]

    async def handler(ctx, a):
        return _result("ok", rows=rows)

    session, _res, end, text = await _run_tool(scripted_session, config, handler, tool_output_max_chars=6000)
    body = text.split("\n\n[Output truncated")[0]
    shown = json.loads(body)
    assert list(shown)[0] == "_vbt" and shown["_vbt"]["returned"] == 400 and shown["_vbt"]["status"] == "ok"
    assert len(shown["rows"]) < 400
    assert _SPILLED_RE.search(text)
    await session.close()


async def test_audit_note_never_follows_a_truncation_note(config, scripted_session):
    async def handler(ctx, a):
        ctx.run.note_audit_error("capture failed for a test")
        return "line\n" * 5000

    session, _res, _end, text = await _run_tool(scripted_session, config, handler, tool_output_max_chars=3000)
    assert "Audit recording failed for this call" in text
    assert text.rstrip().endswith("]") and text.index("Audit recording failed") < text.index("[Output truncated")
    assert _SPILLED_RE.search(text)
    await session.close()


# --------------------------------------------------------------------------- GatewayError


async def test_gateway_error_is_an_error_with_its_kind(config, scripted_session):
    async def handler(ctx, a):
        raise GatewayError(ErrorKind.not_found, "no target PCSK99", tool=TOOL, payload=not_found_payload(
            "gene", "PCSK99", "ensembl_gene", ["ensembl_gene", "hgnc_symbol"], ["hgnc_symbol exact: none"], [],
            "open_targets@25.09", "target"))

    session, res, end, text = await _run_tool(scripted_session, config, handler)
    assert end["is_error"] and end["error_kind"] == "not_found"
    assert text.startswith("Error: {") and json.loads(text[len("Error: "):])["kind"] == "not_found"
    assert res.tool_errors[0]["error_kind"] == "not_found"
    assert session.rt.run.tool_call_status(end["tool_use_id"])["error_kind"] == "not_found"
    # a model-side kind is not a data-source failure
    assert res.unresolved_data_failures == []
    await session.close()


async def test_data_side_gateway_error_is_a_data_source_failure(config, scripted_session):
    async def handler(ctx, a):
        raise GatewayError("not_ready", "table target not ready", tool=TOOL)

    session, res, end, _text = await _run_tool(scripted_session, config, handler)
    assert end["error_kind"] == "not_ready"
    assert [f["tool"] for f in res.unresolved_data_failures] == [TOOL]
    await session.close()


def test_failures_exempt_model_side_kinds_and_copy_kind_and_status():
    events = [
        {"type": "tool_start", "tool_use_id": "a", "tool": TOOL, "input": {"gene": "X"}, "agent": "g"},
        {"type": "tool_end", "tool_use_id": "a", "tool": TOOL, "is_error": True, "error_kind": "not_found",
         "output": "Error: not found"},
        {"type": "tool_start", "tool_use_id": "b", "tool": TOOL, "input": {"gene": "Y"}, "agent": "g"},
        {"type": "tool_end", "tool_use_id": "b", "tool": TOOL, "is_error": True, "error_kind": "source_error",
         "output": "Error: upstream 500"},
        {"type": "tool_start", "tool_use_id": "c", "tool": TOOL, "input": {"gene": "Z"}, "agent": "g"},
        {"type": "tool_end", "tool_use_id": "c", "tool": TOOL, "is_error": False, "result_status": "empty"},
    ]
    recs = fl.records_from_trace(events)
    assert recs[0]["error_kind"] == "not_found" and recs[2]["result_status"] == "empty"
    summary = fl.summarize(recs)
    assert [f["tool_use_id"] for f in summary["unresolved_data"]] == ["b"]
    assert len(summary["all"]) == 2           # both stay failed calls in the audit record
    assert [f["tool_use_id"] for f in fl.unresolved_from_trace(events, data_only=True)] == ["b"]
    assert fl.data_failure_notice(summary["all"]).count(TOOL) == 1
    for kind in ("ambiguous", "invalid_argument", "unsupported_combination", "incomplete_key",
                 "unsupported_filter", "insufficient_resolution"):
        assert not fl._data_failure({"tool": TOOL, "is_error": True, "error_kind": kind})
    assert fl._data_failure({"tool": TOOL, "is_error": True, "error_kind": "tool_defect"})


# --------------------------------------------------------------------------- shrink_json


def test_shrink_json_never_shrinks_the_header():
    header = {"v": 1, "status": "partial", "returned": 20, "total": 61, "notes": ["n" * 400],
              "coverage_statement": "s" * 500, "prov": "dp_0123456789ab"}
    obj = {"_vbt": header, "rows": [{"a": "y" * 600} for _ in range(300)]}
    for limit in (40_000, 6_000, 2_500):
        shown = json.loads(shrink_json(obj, limit, spill_path="/x.json"))
        assert shown["_vbt"] == header and list(shown)[0] == "_vbt"
    many = {"_vbt": header, **{f"k{i}": i for i in range(3000)}}
    shown = json.loads(shrink_json(many, 1500, spill_path="/x.json"))
    assert shown["_vbt"] == header
    # nested _vbt keys are ordinary values
    nested = {"x": {"_vbt": "z" * 5000}}
    assert len(shrink_json(nested, 1000)) <= 1100


# --------------------------------------------------------------------------- tools, readiness, prompts


async def test_on_tools_changed_extends_the_registry_and_clears_prompts(config, scripted_session):
    session = await scripted_session(config, {"cso": []})
    rt = session.rt
    rt._system_cache["cso"] = ["stale"]
    late = Tool("mcp__late__tool", "late", schema({}), lambda ctx, a: "ok", source="mcp:late")
    rt._on_tools_changed([late])
    assert rt.registry.get("mcp__late__tool") is late and rt._system_cache == {}
    updated = Tool("mcp__late__tool", "updated listing", schema({}), lambda ctx, a: "ok", source="mcp:late")
    rt._on_tools_changed([updated])
    assert rt.registry.get("mcp__late__tool").description == "updated listing"
    await session.close()


async def test_tool_readiness_in_list_tools_and_prompt(config, scripted_session):
    session = await scripted_session(config, {"cso": []})
    rt = session.rt
    rt.set_tool_readiness({
        "mcp__functional_genomics__query_drug_perturbation": "tahoe_100m.de_permissive missing",
        "mcp__functional_genomics__get_drug_targets": {"table": "tahoe_100m.de_permissive", "check": "missing"},
        "mcp__target__get_target_info": {"table": "open_targets.target", "column": "homologues.speciesId",
                                         "check": "schema_drift"}})
    inv = await rt.registry.get("ListTools")(ToolContext(agent="cso", run=session.run, runtime=rt), {})
    assert set(inv["tools_not_ready"]) == {"mcp__functional_genomics__query_drug_perturbation",
                                           "mcp__functional_genomics__get_drug_targets",
                                           "mcp__target__get_target_info"}
    g = rt.agents["genomics-analyst"]
    volatile = rt._system_for(g, rt.workspace_for(g.name))[1].text
    assert ("`functional_genomics`: `get_drug_targets`, `query_drug_perturbation` unavailable "
            "(tahoe_100m.de_permissive missing)") in volatile
    assert "`target`: `get_target_info` unavailable (open_targets.target.homologues.speciesId schema_drift)" \
        in volatile
    await session.close()


def test_unavailable_text_accepts_old_and_split_shapes():
    old = _unavailable_text({"genetics": "Connection closed\nmore"})
    assert old.startswith("- Unavailable data servers: `genetics` (Connection closed)")
    assert _unavailable_text(["genetics"]).startswith("- Unavailable data servers: `genetics`.")
    split = _unavailable_text({"servers": {"genetics": "Connection closed"},
                               "tools": {"mcp__target__get_target_info": "open_targets.target missing"}})
    assert "`genetics` (Connection closed)" in split
    assert "`target`: `get_target_info` unavailable (open_targets.target missing)" in split
    assert _unavailable_text({"servers": {}, "tools": {}}) == ""
    assert "Unavailable data servers" not in _unavailable_text({"servers": {}, "tools": {"mcp__a__b": "x"}})


def test_data_layer_addendum_only_for_data_agents_when_enforcing(config, tmp_path):
    data_agent = _probe()
    harness_only = _probe(tools=["Read", "mcp__provenance__record_claims"])
    marker = "Every `mcp__*` data result starts with a `_vbt` header"
    _s, on = system_prompt_parts(data_agent, run_dir=tmp_path, workspace=tmp_path, config=config, data_layer=True)
    _s, off = system_prompt_parts(data_agent, run_dir=tmp_path, workspace=tmp_path, config=config)
    _s, other = system_prompt_parts(harness_only, run_dir=tmp_path, workspace=tmp_path, config=config,
                                    data_layer=True)
    assert marker in on and marker not in off and marker not in other
    stable_on, _v = system_prompt_parts(data_agent, run_dir=tmp_path, workspace=tmp_path, config=config,
                                        data_layer=True)
    assert marker not in stable_on            # volatile only: the cached stable prefix is unchanged


def test_prompts_carry_the_absence_rule():
    prompts = Path(runtime_mod.__file__).resolve().parent / "prompts"
    addendum = (prompts / "data_layer_addendum.md").read_text()
    assert 10 <= len(addendum.strip().splitlines()) <= 20
    flat = " ".join(addendum.split())
    for needle in ("`status`", "`returned`", "`total`", "`truncated`", "`order`", "`coverage`", "`evidence`",
                   "`prov`", "not evidence", "`supports: \"absence\"`", "`covered`", "`censored`", "`_vbt.cite`"):
        assert needle in flat, needle
    burden = (prompts / "genomics_burden_addendum.md").read_text()
    assert "valid outcome" not in burden
    assert ("A resolved query with `status: empty` means no rows in this source's coverage;\nrecord it only as an "
            "absence finding (`supports: absence`), never as support. A\n`not_found` error means the identifier is "
            "wrong.") in burden


# --------------------------------------------------------------------------- start_mcp


def _no_gateway_module(monkeypatch):
    vars(datalayer).pop("build_gateway", None)  # a cached real gateway (later waves) would short-cut __getattr__
    monkeypatch.setattr(datalayer, "_GATEWAY_MODULE", "vbt.datalayer._no_such_gateway_module")


async def test_start_mcp_with_data_enabled_and_no_gateway_module_still_starts(config, scripted_session,
                                                                               monkeypatch):
    _no_gateway_module(monkeypatch)
    config["data"] = {"enabled": True, "gateway": {"mode": "enforce"}}
    config["mcp_servers"] = {"servers": []}
    session = await scripted_session(config, {"cso": []})
    rt = session.rt
    assert await rt.start_mcp() == {}
    assert rt.mcp is not None and rt.gateway is None and "not installed" in rt.gateway_error
    events = [e for e in rt.run.events() if e["type"] in ("data_gateway_unavailable", "mcp_started")]
    assert [e["type"] for e in events] == ["data_gateway_unavailable", "mcp_started"]
    assert not rt.data_layer_enforcing
    await session.close()


async def test_start_mcp_without_the_data_layer_builds_no_gateway(config, scripted_session, monkeypatch):
    built = []
    monkeypatch.setitem(vars(datalayer), "build_gateway", lambda *a, **k: built.append(1))
    config["data"] = {"enabled": False}
    config["mcp_servers"] = {"servers": []}
    session = await scripted_session(config, {"cso": []})
    assert await session.rt.start_mcp() == {}
    assert built == [] and session.rt.gateway is None and session.rt.gateway_error is None
    config["data"] = {"enabled": True, "gateway": {"mode": "off"}}
    assert await session.rt.start_mcp() == {}
    assert built == []
    await session.close()


class _FakeGateway:
    mode = "enforce"

    def extra_servers(self):
        return [{"name": "data", "command": "python", "args": ["-c", "pass"]},
                {"name": "configured", "command": "ignored"}]

    def pinned(self):
        return {"mode": "enforce", "catalog_sha256": "sha256:abc"}


class _FakeBridge:
    """Stands in for an MCPBridge with the gateway seam (§11.1)."""

    made: list = []

    def __init__(self, specs, *, extra_env=None, log_dir=None, options=None, on_event=None, gateway=None,
                 on_tools_changed=None):
        self.specs, self.gateway, self.on_tools_changed = specs, gateway, on_tools_changed
        self.sessions: dict = {}
        self.failures: dict = {}
        self.started_with = None
        _FakeBridge.made.append(self)

    async def start(self, only=None):
        self.started_with = only
        return [Tool("mcp__configured__t", "t", schema({}), lambda ctx, a: "ok", source="mcp:configured")]

    async def aclose(self):
        pass


async def test_start_mcp_wires_the_gateway_into_the_bridge(config, scripted_session, monkeypatch):
    seen = {}

    def build(cfg, run=None):
        seen["run"] = run
        return _FakeGateway()

    monkeypatch.setitem(vars(datalayer), "build_gateway", build)
    monkeypatch.setattr(runtime_mod, "MCPBridge", _FakeBridge)
    config["data"] = {"enabled": True}
    config["mcp_servers"] = {"servers": [{"name": "configured", "command": "python"}]}
    session = await scripted_session(config, {"cso": []})
    rt = session.rt
    await rt.start_mcp(["configured"])
    bridge = _FakeBridge.made[-1]
    assert bridge.gateway is rt.gateway and isinstance(rt.gateway, _FakeGateway)
    assert bridge.on_tools_changed == rt._on_tools_changed
    assert [s.name for s in bridge.specs] == ["configured", "data"]    # explicit entries win
    assert bridge.specs[0].command == "python"
    assert bridge.started_with == {"configured", "data"}               # the data child starts with a subset
    assert seen["run"]["mcp_output_dir"] == str(rt.run.mcp_output_dir)
    assert rt.registry.get("mcp__configured__t") is not None and rt.data_layer_enforcing
    await session.close()


def test_accepted_kwargs_filters_by_signature():
    def f(a, *, gateway=None):
        return a

    def g(a, **kw):
        return a

    assert runtime_mod._accepted_kwargs(f, {"gateway": 1, "on_tools_changed": 2}) == {"gateway": 1}
    assert runtime_mod._accepted_kwargs(g, {"x": 1}) == {"x": 1}


@pytest.mark.parametrize("text,expected_last", [("body", "note"), ("body\n\n[Output truncated: 9 chars (x). "
                                                                    "Full output saved to /p; y.]", "]")])
def test_add_note_keeps_truncation_note_last(text, expected_last):
    out = runtime_mod._add_note(text, "note")
    assert out.endswith(expected_last) and "note" in out
    assert re.search(r"note\n\n\[Output truncated", out) or "[Output truncated" not in text
