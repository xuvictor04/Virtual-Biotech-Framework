"""The whole stack end to end (docs/E2E_RUN.md).

Offline (the default suite), with the ``e2e-cpu`` profile and a scripted model in place of the served one:

* the session preflight's data check reaches the gateway before the data child lists its tools (a regression
  the first CPU run found: the listing sent the data child a second full check that kept running for minutes);
* a ClinVar-shaped TSV whose header is not a word (``#GeneID``) and whose rows repeat registers as
  ``InspectDataset`` drafted it (the real ``gene_condition_source_id`` file did not);
* session 1 of the end-to-end run on the Open Targets fixtures, through the UNMODIFIED upstream MCP servers, the
  gateway in enforce mode, the data child and the reaper: orientation (Chief of Staff brief, clarification),
  delegation to two specialists, a resolved symbol and a ``not_found`` through the gateway, the reviewer, claims
  with evidence, MANIFEST/audit.html, ``vbt verify --data``, ``vbt ds retro-audit`` and ``vbt ds graduate``;
* session 2 in the same project: a dataset the harness has no descriptor for; the CSO delegates to the
  data-engineer, which drafts and registers the descriptor and a tested helper utility; a specialist then reads the
  new table through the gateway and calls the new tool.

Opt-in, against a served model (``VBT_E2E_MODEL_URL=http://127.0.0.1:8012/v1``, optionally ``VBT_E2E_MODEL``
(default ``qwen3.5-2b``) and ``VBT_E2E_TIMEOUT_S``): the same two sessions driven by the model, on the fixtures, through
``vbt run``. A small model is weak: the assertions are the harness's (every turn recorded, every tool call
answered, no harness error, the run verifiable), never the model's choices.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vbt.config import load_config
from vbt.providers.base import ToolCall
from vbt.providers.mock import ScriptedProvider, reply

REPO = Path(__file__).resolve().parents[1]
ECHO = str(REPO / "tests" / "fixtures" / "echo_env_mcp_server.py")
HAVE_MCP = importlib.util.find_spec("fastmcp") is not None and importlib.util.find_spec("mcp") is not None
HAVE_ARROW = importlib.util.find_spec("pyarrow") is not None
needs_mcp = pytest.mark.skipif(not HAVE_MCP, reason="fastmcp and mcp are needed to start MCP servers")
needs_arrow = pytest.mark.skipif(not HAVE_ARROW, reason="pyarrow is needed to read the datasets")

#: The unmodified upstream servers the e2e-cpu profile starts (the Open Targets servers).
E2E_SERVERS = ("target", "disease", "drug", "association", "genetics", "interaction", "pathway")
ENGINEER = "data-engineer"
MODEL_URL = os.environ.get("VBT_E2E_MODEL_URL", "").strip()

# ClinVar's gene_condition_source_id, in its shape: a header that is not a word (``#GeneID``), sources and source ids
# that may be empty, and rows that repeat (484 exact copies in the whole file). Rows copied from
# https://ftp.ncbi.nlm.nih.gov/pub/clinvar/gene_condition_source_id (Last-Modified 2026-10-07, retrieved 2026-10-09:
# 14,211 rows, 1,323,448 bytes, sha256 36ea1e52...a1206; public domain, NCBI).
CLINVAR = "\n".join("\t".join(r) for r in [
    ["#GeneID", "AssociatedGenes", "RelatedGenes", "ConceptID", "DiseaseName", "SourceName", "SourceID", "DiseaseMIM",
     "LastUpdated"],
    ["348", "APOE", "", "C1863051", "Alzheimer disease 2", "MONDO", "MONDO:0007089", "104310", "Mar  2 2016"],
    ["348", "APOE", "", "C0020479", "Familial type 3 hyperlipoproteinemia", "", "", "617347", "Feb 16 2016"],
    ["348", "APOE", "", "C0002395", "Alzheimer disease", "MONDO", "MONDO:0004975", "", "Feb 19 2020"],
    ["348", "APOE", "", "C0002395", "Alzheimer disease", "Human Phenotype Ontology", "HP:0002511", "", "Feb 19 2020"],
    ["3949", "LDLR", "", "C0745103", "Hypercholesterolemia, familial, 1", "MONDO", "MONDO:0007750", "143890",
     "Apr 19 2022"],
    ["3949", "LDLR", "", "C0020445", "Familial hypercholesterolemia", "MONDO", "MONDO:0005439", "", "Jan 15 2020"],
    ["255738", "PCSK9", "", "C0020445", "Familial hypercholesterolemia", "MONDO", "MONDO:0005439", "", "Jan 15 2020"],
    ["255738", "PCSK9", "", "C1863551", "Hypercholesterolemia, autosomal dominant, 3", "MONDO", "MONDO:0011369",
     "603776", "Apr 19 2022"],
    ["1290", "COL5A2", "", "C0268336", "Ehlers-Danlos syndrome, classic type, 2", "MONDO", "MONDO:0019568", "130010",
     "Apr 19 2022"],
    ["1290", "COL5A2", "", "C0268336", "Ehlers-Danlos syndrome, classic type, 2", "MONDO", "MONDO:0019568", "130010",
     "Apr 19 2022"],
]) + "\n"

UTILITY = '''"""Summaries of gene-condition rows."""


def run(rows: list) -> dict:
    """Distinct conditions of gene-condition rows (each a mapping with DiseaseName and SourceName) and how many
    rows each source contributed; an empty source is counted as "(none)". Returns n_rows, n_conditions,
    conditions (sorted) and by_source."""
    conditions = sorted({str(r.get("DiseaseName")) for r in rows if r.get("DiseaseName")})
    by_source: dict = {}
    for r in rows:
        key = r.get("SourceName") or "(none)"
        by_source[key] = by_source.get(key, 0) + 1
    return {"n_rows": len(rows), "n_conditions": len(conditions), "conditions": conditions, "by_source": by_source}
'''
UTILITY_TESTS = '''import utility


def test_known_answer():
    rows = [{"DiseaseName": "B", "SourceName": "MONDO"}, {"DiseaseName": "A", "SourceName": ""},
            {"DiseaseName": "B", "SourceName": "MONDO"}]
    out = utility.run(rows)
    assert out == {"n_rows": 3, "n_conditions": 2, "conditions": ["A", "B"], "by_source": {"MONDO": 2, "(none)": 1}}


def test_no_rows():
    assert utility.run([]) == {"n_rows": 0, "n_conditions": 0, "conditions": [], "by_source": {}}
'''
UTILITY_SCHEMA = {"type": "object", "properties": {"rows": {"type": "array", "items": {"type": "object"},
                                                            "description": "rows with DiseaseName and SourceName"}},
                  "required": ["rows"]}


def tc(tid: str, tool: str, **args: Any) -> ToolCall:
    return ToolCall(id=tid, name=tool, input=args)


class _Scripted(ScriptedProvider):
    """A scripted model under a name other than ``mock``: the session preflight (credentials, the data check,
    MCP commands) runs as it does for a served model, which ``require_ready`` skips for ``mock``."""

    name = "scripted"


# ---------------------------------------------------------------------------- the regression tests


def _echo_config(tmp_path: Path) -> dict:
    """The mock profile with one small stdio server behind an enforcing gateway (so the data child starts). It is
    named after a shipped overlay's server, so the session's tables are that overlay's (what a listing checks)."""
    config = load_config(["mock"], overrides={"paths": {"runs_dir": str(tmp_path / "runs")},
                                              "data": {"enabled": True, "gateway": {"mode": "enforce"},
                                                       "cache_dir": str(tmp_path / "dl-cache")}})
    config["mcp_servers"] = {"servers": [{"name": "pathway", "command": sys.executable, "args": ["-E", ECHO],
                                          "timeout_s": 30, "enabled": True}]}
    return config


@needs_mcp
@pytest.mark.parametrize("supplied", [True, False])
async def test_the_preflight_check_reaches_the_gateway_before_the_data_child_lists_its_tools(
        tmp_path, monkeypatch, supplied):
    """The session's preflight already ran the data check: the gateway must decide calls from it and never send
    the data child a second session check. Handing it over after the servers started cancelled only the harness's
    task: the data child kept computing the whole check (minutes and GBs on a real release, observed in the first
    CPU end-to-end run). Without a supplied check the listing still starts one."""
    from vbt.datalayer.gateway.service_client import ServiceClient
    from vbt.runtime import Runtime
    from vbt.session import Run

    sent: list[tuple[list[str], str]] = []

    async def check(self, tables=(), depth="standard"):
        sent.append((list(tables), depth))
        return {"tables": {}}

    monkeypatch.setattr(ServiceClient, "check", check)
    run = Run(tmp_path / "runs")
    rt = Runtime(_echo_config(tmp_path), run)
    try:
        if supplied:
            assert await rt.start_mcp(readiness={"tables": {}}) == {}
        else:
            assert await rt.start_mcp() == {}
        assert rt.gateway is not None and "data" in rt.mcp.sessions
        task = rt.gateway._check_task                                   # noqa: SLF001
        if task is not None:
            await task
        assert bool(sent) is not supplied, sent
        assert rt.gateway._readiness_supplied is supplied               # noqa: SLF001
    finally:
        await rt.aclose()
        run.close()


@pytest.mark.parametrize("name,expected,absent", [("llamacpp", "llama-server", "vbt local serve"),
                                                   ("vllm", "vbt local serve", "llama-server")])
async def test_a_server_that_is_down_is_named_with_the_command_that_starts_it(monkeypatch, name, expected, absent):
    """When the CPU run's llama-server stopped mid-session, every retry told the operator to run `vbt local serve`,
    which starts vLLM: a llama.cpp provider names llama-server instead (vLLM and SGLang keep the vLLM hint)."""
    import socket

    from vbt.providers.base import Message, ModelSettings, ProviderError, RetryableProviderError, TextBlock
    from vbt.providers.openai_compat import OpenAICompatProvider

    monkeypatch.delenv("VBT_LLM_BASE_URL", raising=False)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    p = OpenAICompatProvider(base_url=f"http://127.0.0.1:{port}/v1", name=name, auto_discover=False)
    try:
        with pytest.raises(RetryableProviderError) as ei:
            await p.complete(settings=ModelSettings(name, "qwen3.5-2b", max_tokens=16), system="s",
                             messages=[Message("user", [TextBlock("hi")])], tools=[])
        assert expected in str(ei.value) and absent not in str(ei.value)
        with pytest.raises(ProviderError) as ei:
            await p.prepare()
        assert expected in str(ei.value) and absent not in str(ei.value)
    finally:
        await p.aclose()


@pytest.mark.parametrize("argv", [["ds", "retro-audit", "latest"], ["ds", "graduate", "target", "--run", "latest"],
                                  ["ds", "replay", "latest", "--all"]])
def test_the_commands_that_read_a_project_run_take_the_project(tmp_path, monkeypatch, argv):
    """A project's runs are under <project>/runs: `vbt ds retro-audit RUN` of the end-to-end run did not find it
    (only `--profile <project>/profile.yaml` did). The data commands that read a recorded run take --project, which
    activates the project as `vbt run --project` does (its runs, descriptors and overlays)."""
    from vbt import cli
    from vbt.projects.model import init_project

    monkeypatch.setenv("VBT_PROJECTS_DIR", str(tmp_path / "projects"))
    project = init_project("demo", config=load_config(["mock"]))
    args = cli.build_parser().parse_args(["--profile", "mock", *argv[:2], "--project", "demo", *argv[2:]])
    cfg = cli.build_config(args)
    assert Path(cfg["paths"]["runs_dir"]).resolve() == (project.root / "runs").resolve()
    assert Path(cfg["data"]["project_dir"]).resolve() == project.root.resolve()


@needs_arrow
async def test_a_header_that_is_not_a_word_registers_as_drafted(config, tmp_path):
    """ClinVar's ``gene_condition_source_id`` (2026-10-07): the key InspectDataset drafted named ``#GeneID``, which
    the descriptor reads as a path, and lint refused the draft ("'#GeneID' is not a valid path: empty segment at
    0"). References are now written as paths (backtick-quoted), the column keeps its literal name, and the repeated
    rows are declared ``row_identity: none``."""
    from vbt.projects.model import activate, init_project
    from vbt.runtime import Runtime
    from vbt.session import Run
    from vbt.tools.base import ToolContext

    config["projects"] = {"root": str(tmp_path / "projects")}
    config["data"]["cache_dir"] = str(tmp_path / "cache")
    config.setdefault("preflight", {})["skip"] = True
    cfg = activate(config, init_project("clinvar", config=config))
    run = Run(Path(cfg["paths"]["runs_dir"]), config=cfg)
    rt = Runtime(cfg, run, provider=ScriptedProvider.from_rules({}))
    try:
        ws = rt.workspace_for(ENGINEER)
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "gene_condition_source_id.tsv").write_text(CLINVAR)

        async def call(tool: str, **args: Any) -> Any:
            return await rt.registry.get(tool)(ToolContext(ENGINEER, rt.run, rt, tool_call_id=f"tu_{tool}"), args)

        info = await call("InspectDataset", path="gene_condition_source_id.tsv", source="clinvar_gcs",
                          table="gene_conditions")
        assert info["key"][0] == "#GeneID" and info["key_identity"] == "none", info
        assert "- '`#GeneID`'" in info["draft_descriptor"] and "'#GeneID':" in info["draft_descriptor"]
        out = await call("RegisterDataSpec", kind="descriptor", content=info["draft_descriptor"],
                         files=["gene_condition_source_id.tsv"], why="the draft as is")
        assert out["status"] == "registered", out
        assert out["check"]["clinvar_gcs.gene_conditions"]["status"] == "ready", out
    finally:
        run.close()


# ---------------------------------------------------------------------------- the stack on fixtures


def _stack_missing() -> str | None:
    if not (HAVE_MCP and HAVE_ARROW):
        return "fastmcp, mcp and pyarrow are needed"
    datalayer_tests = str(REPO / "tests" / "datalayer")
    if datalayer_tests not in sys.path:
        sys.path.append(datalayer_tests)     # after tests/: `conftest` stays the top-level one
    import dl_upstream

    return dl_upstream.upstream_missing() or dl_upstream.gateway_missing()


def _stack_config(tmp: Path, root: Path, env: dict[str, str], zenodo: Path, **extra: Any) -> dict[str, Any]:
    """The e2e-cpu profile on the fixtures: the profile's servers (unmodified upstream, ``-B``: nothing is written
    under the upstream checkout), its data layer (enforce, data child, reaper) and its limits."""
    import dl_upstream

    overrides: dict[str, Any] = {
        "paths": {"runs_dir": str(tmp / "runs")},
        "vars": {"upstream": str(dl_upstream.upstream_root())},
        "projects": {"root": str(tmp / "projects")},
        "data": {"cache_dir": str(tmp / "dl-cache")},
        "tool_env": {"OPEN_TARGETS_DATA_PATH": str(root), "VBT_ZENODO_DIR": str(zenodo),
                     **dl_upstream.offline_bases()},
        "mcp": dict(dl_upstream.FAST_START),
    }
    for key, value in extra.items():
        overrides[key] = {**overrides.get(key, {}), **value} if isinstance(value, dict) else value
    cfg = load_config(["e2e-cpu"], overrides=overrides)
    cfg["mcp_servers"] = {"servers": dl_upstream.server_specs(cfg, E2E_SERVERS, env)}
    return cfg


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    reason = _stack_missing()
    if reason:
        pytest.skip(reason)
    import dl_fixtures
    from dl_upstream import DataEnv
    from vbt.projects.model import init_project

    tmp = tmp_path_factory.mktemp("e2e")
    root = dl_fixtures.build_ot_fixture(tmp / "ot" / "25.09")
    zenodo = tmp / "no-zenodo"
    zenodo.mkdir()
    env = DataEnv(ot_root=root).env()
    env["VBT_ZENODO_DIR"] = str(zenodo)
    config = _stack_config(tmp, root, env, zenodo)
    project = init_project("e2e", config=config)
    incoming = project.root / "incoming"           # what the user downloaded into the project (a read root)
    incoming.mkdir()
    (incoming / "gene_condition_source_id.tsv").write_text(CLINVAR)
    return SimpleNamespace(tmp=tmp, root=root, env=env, zenodo=zenodo, config=config, project=project,
                           tsv=incoming / "gene_condition_source_id.tsv", runs={})


def _trace(run_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (run_dir / "logs" / "trace.jsonl").read_text().splitlines() if line.strip()]


def _ends(run_dir: Path) -> dict[str, dict[str, Any]]:
    return {e["tool_use_id"]: e for e in _trace(run_dir) if e.get("type") == "tool_end"}


def _payload(text: str) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _last_result(messages: list[Any]) -> str:
    return "\n".join(str(getattr(b, "content", "")) for b in messages[-1].content)


QUESTION_1 = ("Is PCSK9 a genetically supported and tractable drug target for hypercholesterolaemia? Delegate the "
              "genetic evidence to the genomics-analyst and its pathway context to the bio-pathways-ppi-analyst. A "
              "collaborator also mentioned a gene called PCSK99: check whether it exists.")


def _session1_rules(seen: dict[str, Any]) -> dict[str, list[Any]]:
    claims = [
        {"id": "C1", "text": "PCSK9 is the Open Targets target ENSG00000169174.", "agent": "genomics-analyst",
         "confidence": "strong", "evidence": [{"kind": "tool_call", "tool_use_id": "ga_symbol"}]},
        {"id": "C2", "text": "PCSK9 has Reactome pathway annotations in Open Targets.",
         "agent": "bio-pathways-ppi-analyst", "confidence": "moderate",
         "evidence": [{"kind": "tool_call", "tool_use_id": "bp_pathways"}]},
    ]

    def keep(key: str, item: Any):
        def f(messages):
            seen[key] = _last_result(messages)
            return item
        return f

    return {
        "chief-of-staff": [reply(tc("cos_tools", "ListTools")),
                           reply("## Brief\nOpen Targets target, disease, association, genetics, pathway and "
                                 "interaction data are available.")],
        "cso": [
            reply("NO_CLARIFICATION_NEEDED"),
            reply("Dispatching genetics and pathway context.",
                  tc("cso_task_ga", "Task", subagent_type="genomics-analyst", description="genetic evidence",
                     prompt="Look up PCSK9 and PCSK99 in Open Targets."),
                  tc("cso_task_bp", "Task", subagent_type="bio-pathways-ppi-analyst", description="pathways",
                     prompt="Give the pathway context of PCSK9.")),
            reply(tc("cso_review", "Task", subagent_type="scientific-reviewer", description="review",
                     prompt="Review: PCSK9 resolves to ENSG00000169174 and has Reactome pathways; PCSK99 is not "
                            "a known target.")),
            keep("review", reply(tc("cso_claims", "mcp__provenance__record_claims", claims=claims))),
            keep("claims", reply("PCSK9 is a well-annotated target [[claim:C1]] with pathway context [[claim:C2]]; "
                                 "PCSK99 is not a known gene.")),
        ],
        "genomics-analyst": [
            reply(tc("ga_symbol", "mcp__target__get_target_info", target_id="PCSK9")),
            keep("symbol", reply(tc("ga_unknown", "mcp__target__get_target_info", target_id="PCSK99"))),
            keep("unknown", reply("PCSK9 resolves to ENSG00000169174; PCSK99 is not a known target (not_found).")),
        ],
        "bio-pathways-ppi-analyst": [
            reply(tc("bp_pathways", "mcp__pathway__get_gene_pathways", target_id="PCSK9")),
            keep("pathways", reply("PCSK9 has Reactome pathway annotations.")),
        ],
        "scientific-reviewer": [reply("VERDICT: APPROVE. The conclusions follow from the cited calls.")],
    }


@needs_mcp
@needs_arrow
async def test_session_1_research_through_the_whole_stack(stack, monkeypatch):
    """Orientation, two specialists, a resolved symbol and a not_found through the enforcing gateway on the unmodified
    upstream servers, review, claims, the run record, verify --data, retro-audit and graduate."""
    import dl_upstream
    from vbt.datalayer.cli import graduation_checklist
    from vbt.datalayer.gateway.service_client import ServiceClient
    from vbt.datalayer.retro_audit import retro_audit
    from vbt.orchestrator import open_session
    from vbt.projects.model import activate
    from vbt.verify import verify_run

    checks: list[Any] = []
    original = ServiceClient.check

    async def counted(self, tables=(), depth="standard"):
        checks.append((list(tables), depth))
        return await original(self, tables, depth)

    monkeypatch.setattr(ServiceClient, "check", counted)
    cfg = activate(stack.config, stack.project)
    seen: dict[str, Any] = {}
    provider = _Scripted.from_rules(_session1_rules(seen))
    with dl_upstream._no_bytecode():                                    # noqa: SLF001 - preflight imports upstream
        session = await open_session(cfg, provider=provider, start_mcp=True, profiles=["e2e-cpu"])
    try:
        assert checks == [], "the data child got a second session check: the preflight's was not handed over"
        assert {"mcp__target__get_target_info", "mcp__pathway__get_gene_pathways", "mcp__data__find"} <= set(
            session.rt.registry.names())
        answer = await session.ask(QUESTION_1)
        rec = session.last_turn
    finally:
        await session.close()
    run_dir = session.run.dir
    stack.runs["session1"] = run_dir
    assert "PCSK99 is not a known gene" in answer, answer
    assert rec["status"] == "completed" and rec.get("reviewed") is True, rec

    trace = _trace(run_dir)
    agents = {e["agent"] for e in trace if e.get("type") == "agent_start"}
    assert {"chief-of-staff", "cso", "genomics-analyst", "bio-pathways-ppi-analyst", "scientific-reviewer"} <= agents
    assert any(e.get("type") == "briefing" for e in trace)
    ends = _ends(run_dir)
    resolved = _payload(seen["symbol"])
    assert resolved["id"] == "ENSG00000169174" and resolved["_vbt"]["status"] == "ok", seen["symbol"][:2000]
    assert "approvedSymbol" in json.dumps(resolved["_vbt"].get("resolved") or resolved["_vbt"]), resolved["_vbt"]
    assert ends["ga_symbol"]["is_error"] is False and ends["ga_symbol"]["result_status"] == "ok"
    assert ends["ga_unknown"]["is_error"] is True and ends["ga_unknown"]["error_kind"] == "not_found", \
        ends["ga_unknown"]
    assert ends["bp_pathways"]["is_error"] is False
    assert '"ok": true' in seen["claims"].replace(" ", " "), seen["claims"][:2000]

    manifest = json.loads((run_dir / "MANIFEST.json").read_text())
    assert manifest["config"]["data"]["mode"] == "enforce"
    assert set(manifest["config"]["mcp"]["started"]) == set(E2E_SERVERS) | {"data"}
    assert manifest["config"]["project"]["name"] == "e2e"
    assert (run_dir / "audit.html").is_file() and (run_dir / "evidence" / "claims.json").is_file()
    assert {p.stem for p in (run_dir / "logs" / "data_provenance").glob("*.json")} >= {"ga_symbol", "bp_pathways"}

    with dl_upstream._no_bytecode():                                    # noqa: SLF001
        report = await asyncio.to_thread(verify_run, run_dir, data=True, config=cfg)
    assert report["status"] == "COMPLETE", json.dumps(report["problems"], indent=1)[:4000]

    audit = retro_audit(run_dir, cfg)
    by_id = {c["tool_use_id"]: c for c in audit["calls"]}
    assert by_id["ga_unknown"]["outcome"] == "not_found" and by_id["ga_symbol"]["outcome"] == "ok", by_id
    assert by_id["ga_symbol"]["cited_by"] == ["C1"]
    grad = graduation_checklist(cfg, ["target", "pathway"], [run_dir])
    for server in ("target", "pathway"):
        evidence = grad[server]["items"]["observe_evidence"]
        assert evidence["ok"] is True and evidence["calls"] >= 1, grad[server]
        assert grad[server]["items"]["overlay"]["ok"] and grad[server]["items"]["lint"]["ok"], grad[server]


QUESTION_2 = ("Which conditions does ClinVar associate with PCSK9? I downloaded ClinVar's gene_condition_source_id "
              "into the project's incoming/ directory; none of the data tools know it yet.")


def _session2_rules(seen: dict[str, Any], tsv: Path) -> dict[str, list[Any]]:
    def keep(key: str, item: Any):
        def f(messages):
            seen[key] = _last_result(messages)
            return item
        return f

    def register_draft(messages):
        info = json.loads(_last_result(messages))
        seen["inspect"] = info
        return reply(tc("de_register", "RegisterDataSpec", kind="descriptor", content=info["draft_descriptor"],
                        files=[str(tsv)], why="ClinVar gene-condition associations the user downloaded"))

    def summarise(messages):
        found = json.loads(_last_result(messages))
        seen["find"] = found
        return reply(tc("gx_util", "util__condition_summary", rows=found["rows"]))

    return {
        "chief-of-staff": [reply("## Brief\nThe user's ClinVar file has no descriptor yet.")],
        "cso": [
            reply("NO_CLARIFICATION_NEEDED"),
            reply(tc("cso_engineer", "Task", subagent_type=ENGINEER, description="register ClinVar",
                     prompt=f"Register {tsv} as a data source (clinvar_gcs.gene_conditions) and a tested utility "
                            "condition_summary that summarises gene-condition rows.")),
            reply(tc("cso_task_gx", "Task", subagent_type="genomics-analyst", description="ClinVar conditions",
                     prompt="Find PCSK9's rows in clinvar_gcs.gene_conditions and summarise them with "
                            "util__condition_summary.")),
            reply(tc("cso_review2", "Task", subagent_type="scientific-reviewer", description="review",
                     prompt="Review: ClinVar lists two conditions for PCSK9.")),
            reply(tc("cso_claims2", "mcp__provenance__record_claims", claims=[
                {"id": "C1", "text": "ClinVar associates PCSK9 with familial hypercholesterolemia.",
                 "agent": "genomics-analyst", "confidence": "strong",
                 "evidence": [{"kind": "tool_call", "tool_use_id": "gx_find"}]}])),
            keep("claims", reply("ClinVar associates PCSK9 with two conditions [[claim:C1]].")),
        ],
        ENGINEER: [
            reply(tc("de_info", "ProjectInfo")),
            reply(tc("de_inspect", "InspectDataset", path=str(tsv), source="clinvar_gcs", table="gene_conditions")),
            register_draft,
            keep("register", reply(tc("de_write_code", "Write", file_path="utilities/condition_summary/utility.py",
                                      content=UTILITY))),
            reply(tc("de_write_tests", "Write", file_path="utilities/condition_summary/test_utility.py",
                     content=UTILITY_TESTS)),
            reply(tc("de_utility", "RegisterUtility", name="condition_summary",
                     description="Distinct conditions of gene-condition rows and rows per source.",
                     input_schema=UTILITY_SCHEMA, directory="utilities/condition_summary",
                     why="every ClinVar question summarises the same rows")),
            keep("utility", reply("Registered clinvar_gcs.gene_conditions and util__condition_summary.")),
        ],
        "genomics-analyst": [
            reply(tc("gx_find", "mcp__data__find", table="clinvar_gcs.gene_conditions",
                     where={"AssociatedGenes": "PCSK9"})),
            summarise,
            keep("util", reply("ClinVar lists 2 conditions for PCSK9.")),
        ],
        "scientific-reviewer": [reply("VERDICT: APPROVE.")],
    }


@needs_mcp
@needs_arrow
async def test_session_2_the_engineer_creates_what_a_specialist_then_uses(stack):
    """A dataset the harness has no descriptor for: the engineer drafts and registers the descriptor and a tested
    utility in the running session; a specialist reads the new table through the gateway and calls the new tool."""
    import dl_upstream
    from vbt.orchestrator import open_session
    from vbt.projects import ledger
    from vbt.projects.model import activate

    cfg = activate(stack.config, stack.project)
    seen: dict[str, Any] = {}
    provider = _Scripted.from_rules(_session2_rules(seen, stack.tsv))
    with dl_upstream._no_bytecode():                                    # noqa: SLF001
        session = await open_session(cfg, provider=provider, start_mcp=True, profiles=["e2e-cpu"])
    try:
        assert ENGINEER in session.rt.agents and "util__condition_summary" not in session.rt.registry.names()
        await session.ask(QUESTION_2)
        rec = session.last_turn
        assert "util__condition_summary" in session.rt.registry.names()
    finally:
        await session.close()
    run_dir = session.run.dir
    assert rec["status"] == "completed" and rec.get("reviewed") is True, rec

    assert seen["inspect"]["key"][0] == "#GeneID" and seen["inspect"]["rows"] == 10
    assert '"status": "registered"' in seen["register"] and '"available": "now' in seen["register"], \
        seen["register"][:3000]
    assert '"status": "registered"' in seen["utility"], seen["utility"][:3000]
    found = seen["find"]
    assert found["_vbt"]["status"] == "ok" and found["_vbt"]["total"] == 2, found
    assert sorted(r["DiseaseName"] for r in found["rows"]) == ["Familial hypercholesterolemia",
                                                               "Hypercholesterolemia, autosomal dominant, 3"]
    util = json.loads(seen["util"])
    assert util["n_conditions"] == 2 and util["by_source"] == {"MONDO": 2}, util

    for kind, name in (("descriptor", "clinvar_gcs"), ("utility", "condition_summary")):
        r = ledger.read_record(stack.project, kind, name)
        assert r["status"] == "registered" and r["who"]["agent"] == ENGINEER and not ledger.verify(stack.project, r)
    tests = ledger.read_record(stack.project, "utility", "condition_summary")["validation"]["tests"]
    assert tests["passed"] == 2 and tests["failed"] == []
    trace = _trace(run_dir)
    assert {e.get("name") for e in trace if e.get("type") == "project_registration"} >= {"clinvar_gcs",
                                                                                         "condition_summary"}
    assert any(e.get("type") == "project_utility_call" for e in trace)
    manifest = json.loads((run_dir / "MANIFEST.json").read_text())
    kinds = {e.get("kind") for e in manifest["artifacts"].values() if e.get("registered")}
    assert {"project_descriptor", "project_utility"} <= kinds


# ---------------------------------------------------------------------------- opt-in: a served model


needs_model = pytest.mark.skipif(not MODEL_URL, reason="set VBT_E2E_MODEL_URL to drive the stack with a served model")


def _served_config(stack: Any) -> dict[str, Any]:
    from vbt.projects.model import activate

    model = os.environ.get("VBT_E2E_MODEL", "").strip() or "qwen3.5-2b"
    timeout = float(os.environ.get("VBT_E2E_TIMEOUT_S", "") or 1800)
    cfg = _stack_config(stack.tmp, stack.root, stack.env, stack.zenodo,
                        provider={"name": "llamacpp", "options": {"base_url": MODEL_URL, "read_timeout_s": timeout}})
    for tier in cfg["models"].values():
        tier["model"] = model
    return activate(cfg, stack.project)


def _harness_failures(run_dir: Path) -> list[str]:
    """What a harness bug leaves in a run: a turn that failed with a Python exception, an unrecorded tool call, an
    audit error. A weak model's own failures (no delegation, a refused call, a turn limit) are not listed."""
    out = []
    trace = _trace(run_dir)
    starts = {e["tool_use_id"] for e in trace if e.get("type") == "tool_start"}
    ends = {e["tool_use_id"] for e in trace if e.get("type") == "tool_end"}
    out += [f"tool call without an end: {t}" for t in sorted(starts - ends)]
    for e in trace:
        if e.get("type") == "turn_end" and str(e.get("status", "")).startswith("failed"):
            out.append(f"turn {e.get('turn')} failed: {e.get('status')}")
    manifest = json.loads((run_dir / "MANIFEST.json").read_text())
    out += [f"audit error: {a}" for a in manifest.get("audit_errors") or []]
    return out


@needs_model
@needs_mcp
@needs_arrow
async def test_a_served_model_drives_the_stack(stack):
    """Session 1 driven by the served model: every mechanism runs and every failure is the model's."""
    import dl_upstream
    from vbt.orchestrator import open_session

    cfg = _served_config(stack)
    t0 = time.monotonic()
    with dl_upstream._no_bytecode():                                    # noqa: SLF001
        session = await open_session(cfg, start_mcp=True, profiles=["e2e-cpu"], interface="run")
    try:
        await session.ask(QUESTION_1)
        if session.last_turn and session.last_turn.get("status") == "completed" and not any(
                e.get("type") == "delegation" and e.get("agent") not in ("chief-of-staff",)
                for e in session.run.events()):
            await session.ask("No further clarification is needed: proceed with the analysis as specified.")
    finally:
        await session.close()
    run_dir = session.run.dir
    assert not _harness_failures(run_dir), _harness_failures(run_dir)
    assert (run_dir / "audit.html").is_file()
    trace = _trace(run_dir)
    assert any(e.get("type") == "model_call" and e.get("agent") == "cso" for e in trace)
    assert time.monotonic() - t0 > 0
