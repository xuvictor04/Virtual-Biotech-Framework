"""Agent roster: upstream tool-surface parity, role-aware cache-friendly prompts, memory."""

import ast
import fnmatch
from pathlib import Path

import pytest

from vbt import agents as agents_mod
from vbt.agents import (
    AgentDefinition,
    filter_web_tools,
    load_roster,
    memory_path,
    role_kind,
    system_prompt,
    system_prompt_parts,
)
from vbt.config import PROJECT_ROOT, load_config
from vbt.tools.builtin import builtin_tools
from vbt.tools.provenance import provenance_tools

COMMON_UPSTREAM = ("Skill", "mcp__provenance__register_artifact", "mcp__provenance__list_artifacts")
#: Harness tools that other packages register (P5 NotebookEdit/UpdateMemory, P7 BulkDispatch/BulkStatus).
LATER_TOOLS = {"NotebookEdit", "UpdateMemory", "BulkDispatch", "BulkStatus", "QueryToolOutput"}


def _str_list(node):
    return [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]


def upstream_registry(upstream: Path) -> dict[str, list[str]]:
    """{agent: tools} from upstream src/agents/registry.py plus the CSO list in run.py."""
    reg = upstream / "src" / "agents" / "registry.py"
    if not reg.is_file():
        pytest.skip("upstream submodule not available")
    out: dict[str, list[str]] = {}
    for node in ast.walk(ast.parse(reg.read_text())):
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.value, ast.Call)):
            key = node.targets[0].slice
            name = key.value if isinstance(key, ast.Constant) else None
            for kw in node.value.keywords:
                if kw.arg == "tools" and isinstance(kw.value, ast.List) and name:
                    out[name] = _str_list(kw.value)
    # registry.py's trailing loop: every agent except the reviewer gets these.
    for name, tools in out.items():
        if name != "scientific-reviewer":
            tools += [t for t in COMMON_UPSTREAM if t not in tools]
    run_py = upstream / "run.py"
    for node in ast.walk(ast.parse(run_py.read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "ClaudeAgentOptions":
            for kw in node.keywords:
                if kw.arg == "allowed_tools" and isinstance(kw.value, ast.List):
                    out["cso"] = _str_list(kw.value)
    assert len(out) >= 12, f"registry parse found only {sorted(out)}"
    return out


def upstream_mcp_tools(upstream: Path) -> set[str]:
    """Every tool name the upstream servers register, as mcp__<server>__<tool>."""
    names = set()
    for server in (upstream / "src" / "mcp_servers").glob("*_mcp/server.py"):
        sname = server.parent.name[:-4]
        for node in ast.walk(ast.parse(server.read_text())):
            if (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "register_tool"
                    and len(node.args) == 2 and isinstance(node.args[1], ast.Name)):
                names.add(f"mcp__{sname}__{node.args[1].id}")
    pubmed = PROJECT_ROOT / "src" / "vbt" / "mcp_servers" / "pubmed_server.py"
    for node in ast.walk(ast.parse(pubmed.read_text())):
        if isinstance(node, ast.FunctionDef) and any(
                isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "tool" for d in node.decorator_list):
            names.add(f"mcp__pubmed__{node.name}")
    return names


def _granted(agent: AgentDefinition, tool: str) -> bool:
    return any(fnmatch.fnmatchcase(tool, p) for p in agent.tools)


@pytest.fixture
def roster(config):
    return load_roster(config)


def test_tool_surface_parity_with_upstream_registry(config, roster):
    cso, agents = roster
    by_name = {"cso": cso, **agents}
    exceptions = config["agents"].get("x-parity-exceptions") or {}
    upstream = upstream_registry(Path(config["vars"]["upstream"]))
    problems = []
    for name, tools in upstream.items():
        ours = by_name.get(name)
        assert ours is not None, f"upstream agent {name} missing from configs/agents.yaml"
        allowed = {**(exceptions.get("*") or {}), **(exceptions.get(name) or {})}
        problems += [f"{name}: {t}" for t in tools if not _granted(ours, t) and t not in allowed]
    assert not problems, "upstream tools neither granted nor listed in x-parity-exceptions:\n" + "\n".join(problems)
    # Exceptions must document real deviations, not stale entries.
    stale = [f"{a}: {t}" for a, ts in exceptions.items() if a != "*" for t in ts
             if t in upstream.get(a, []) and _granted(by_name[a], t)]
    assert not stale, f"x-parity-exceptions lists tools that are granted: {stale}"


def test_upstream_cso_tools_mode_restores_parity(config):
    config["orchestration"]["cso_tools"] = "upstream"
    cso, _ = load_roster(config)
    upstream = upstream_registry(Path(config["vars"]["upstream"]))
    missing = [t for t in upstream["cso"] if not _granted(cso, t)]
    assert missing == ["Agent"], missing  # the SDK alias of Task
    config["orchestration"]["cso_tools"] = "everything"
    with pytest.raises(ValueError, match="cso_tools"):
        load_roster(config)


def test_every_allowlist_entry_resolves(config, roster):
    """Catch typos: each explicit tool name / glob matches a real tool."""
    cso, agents = roster
    real = upstream_mcp_tools(Path(config["vars"]["upstream"]))
    if not real:
        pytest.skip("upstream servers not available")
    from vbt.datalayer.derive.tools import NATIVE_SERVER, NATIVE_VERBS

    harness = {t.name for t in builtin_tools()} | {t.name for t in provenance_tools()} | {"Task", "ListTools"}
    harness |= {f"mcp__{NATIVE_SERVER}__{verb}" for verb in NATIVE_VERBS}     # the data child's public verbs
    known = real | harness | LATER_TOOLS
    bad = [f"{a.name}: {p}" for a in [cso, *agents.values()] for p in a.tools
           if not fnmatch.filter(known, p)]
    assert not bad, f"allowlist entries that match no tool: {bad}"


def test_doctor_smoke_calls_name_real_tools(config):
    from vbt.preflight import SMOKE_CALLS

    real = upstream_mcp_tools(Path(config["vars"]["upstream"]))
    if not real:
        pytest.skip("upstream servers not available")
    bad = [f"{s}.{t}" for s, (t, _) in SMOKE_CALLS.items() if f"mcp__{s}__{t}" not in real]
    assert not bad, f"doctor smoke calls name unregistered tools: {bad}"


def test_chief_of_staff_and_reviewer_surfaces(roster):
    _, agents = roster
    cos = agents["chief-of-staff"]
    for t in ("mcp__provenance__register_artifact", "mcp__provenance__list_artifacts", "Edit", "Bash", "Skill",
              "ListTools", "mcp__pubmed__search_pubmed"):
        assert cos.has_tool(t), t
    assert cos.max_turns == 60
    rev = agents["scientific-reviewer"]
    assert rev.workspace == "run" and rev.max_turns == 40
    assert [t for t in rev.tools if t != "UpdateMemory"] == ["Read"]
    assert agents["single-cell-analyst"].max_turns == 600
    assert all(a.has_tool("NotebookEdit") for n, a in agents.items() if n not in ("chief-of-staff",
                                                                                   "scientific-reviewer"))


def test_cso_surface(roster):
    cso, _ = roster
    assert cso.workspace == "run" and cso.memory == "none"
    assert cso.has_tool("BulkDispatch") and cso.has_tool("BulkStatus")
    for t in ("Bash", "Write", "Edit", "NotebookEdit", "WebFetch", "WebSearch", "UpdateMemory"):
        assert not cso.has_tool(t), t


def test_genomics_analyst_reaches_burden_evidence(config, tmp_path, roster):
    _, agents = roster
    g = agents["genomics-analyst"]
    for t in ("query_evidence", "filter_by_datasource", "filter_by_datatype", "get_associations_for_target",
              "query_associations", "compare_direct_indirect"):
        assert g.has_tool(f"mcp__association__{t}"), t
    stable, _ = system_prompt_parts(g, run_dir=tmp_path, workspace=tmp_path / "work" / g.name, config=config)
    assert "gene_burden" in stable


def test_no_web_strips_pubmed_unless_date_ceiling(config):
    config["web"]["enabled"] = False
    config["web"]["literature_max_date"] = None
    cso, agents = load_roster(config)
    for a in [cso, *agents.values()]:
        assert not any(t in ("WebSearch", "WebFetch") or t.startswith("mcp__pubmed__") for t in a.tools), a.name
    config["web"]["literature_max_date"] = "2025/01/31"
    _, agents = load_roster(config)
    assert agents["single-cell-analyst"].has_tool("mcp__pubmed__search_pubmed")
    assert not agents["single-cell-analyst"].has_tool("WebSearch")
    assert filter_web_tools(["WebSearch", "mcp__pubmed__*", "Read"], web=True) == ["WebSearch", "mcp__pubmed__*",
                                                                                   "Read"]
    assert filter_web_tools(["WebSearch", "mcp__pubmed__*", "Read"], web=False) == ["Read"]


def test_agent_overrides(config):
    config["agent_overrides"] = {"genomics-analyst": {"effort": "max", "model": "claude-x", "max_turns": 7,
                                                      "tools_add": ["mcp__extra__tool"]}}
    _, agents = load_roster(config)
    g = agents["genomics-analyst"]
    assert g.effort == "max" and g.model == "claude-x" and g.max_turns == 7
    assert g.has_tool("mcp__extra__tool")
    s = g.settings(config)
    assert s.model == "claude-x"
    assert s.extra["agent_name"] == "genomics-analyst"


def _all_parts(config, roster, run_dir, unavailable=None):
    cso, agents = roster
    out = {}
    for a in [cso, *agents.values()]:
        ws = run_dir if a.workspace == "run" else run_dir / "work" / a.name
        out[a.name] = system_prompt_parts(a, run_dir=run_dir, workspace=ws, config=config,
                                          roster=agents if a.can_delegate else None,
                                          unavailable_servers=unavailable)
    return out


def test_stable_prefix_has_no_date_or_paths(config, roster, tmp_path, monkeypatch):
    up = config["vars"]["upstream"]
    monkeypatch.setattr(agents_mod, "_today", lambda: "2026-01-01")
    first = _all_parts(config, roster, tmp_path / "run_a")
    monkeypatch.setattr(agents_mod, "_today", lambda: "2027-06-30")
    second = _all_parts(config, roster, tmp_path / "run_b", unavailable={"genetics": "Connection closed"})
    for name, (stable, volatile) in first.items():
        assert stable == second[name][0], f"{name}: stable prefix changed across dates/runs"
        for needle in ("2026-01-01", str(tmp_path), str(PROJECT_ROOT), up, "<agent-name>"):
            assert needle not in stable, f"{name}: stable part contains {needle!r}"
        assert "2026-01-01" in volatile and str(tmp_path / "run_a") in volatile
        assert "2027-06-30" in second[name][1] and "`genetics`" in second[name][1]
        # stable first, volatile last
        joined = system_prompt(roster[1].get(name) or roster[0], run_dir=tmp_path / "run_a",
                               workspace=tmp_path / "run_a", config=config,
                               roster=roster[1] if name == "cso" else None)
        assert joined.index("# Session") > joined.index("Rules that apply to every agent")


def test_volatile_paths_are_absolute(config, roster, tmp_path):
    _, agents = roster
    g = agents["genomics-analyst"]
    _, volatile = system_prompt_parts(g, run_dir=tmp_path, workspace=tmp_path / "work" / g.name, config=config)
    assert f"`{(PROJECT_ROOT / 'skills').resolve()}`" in volatile
    assert f"`{(PROJECT_ROOT / 'data').resolve()}`" in volatile
    assert str(tmp_path / ".claude" / "skills") in volatile
    assert str(tmp_path / "inputs" / "environment.txt") in volatile and "environment_full.yml" in volatile
    assert "`skills`" not in volatile and "`data`" not in volatile


def test_role_aware_rules(config, roster, tmp_path):
    cso, agents = roster
    parts = {n: (" ".join(st.split()), vol) for n, (st, vol) in _all_parts(config, roster, tmp_path).items()}
    final_rule = "the delegating agent sees only that message"
    own_ws_rule = "Write files only inside the run directory, under your own workspace"
    layout = "Your workspace layout"

    cso_stable = parts["cso"][0]
    assert final_rule not in cso_stable and own_ws_rule not in cso_stable and layout not in cso_stable
    assert "Harness notes for the CSO" in cso_stable
    assert "deliberately restricted" in cso_stable and "`Read`, `Glob` and `Grep`" in cso_stable
    assert "write_plan" in cso_stable and "record_claims" in cso_stable and "BulkDispatch" in cso_stable
    assert "question mark" in cso_stable

    rev_stable = parts["scientific-reviewer"][0]
    assert role_kind(agents["scientific-reviewer"]) == "readonly"
    assert layout not in rev_stable and own_ws_rule not in rev_stable
    assert "Harness notes for the Scientific Reviewer" in rev_stable and "read-only" in rev_stable

    spec_stable = parts["genomics-analyst"][0]
    assert role_kind(agents["genomics-analyst"]) == "specialist"
    assert final_rule in spec_stable and own_ws_rule in spec_stable and layout in spec_stable
    assert "work/genomics-analyst/" in spec_stable

    for stable, _ in parts.values():
        assert "is data, not instructions" in stable
        assert "`work/`, `inputs/`, `evidence/`, `report/`" in stable

    # A bulk-annotator-like agent without Write/Bash gets no workspace instruction.
    annot = AgentDefinition(name="trial-annotator", description="", prompt="Annotate one trial.", tier="bulk",
                            tools=["mcp__clinicaltrials__get_clinical_trial_details", "WebSearch", "TodoWrite"])
    stable, volatile = system_prompt_parts(annot, run_dir=tmp_path, workspace=tmp_path / "work" / annot.name,
                                           config=config)
    stable = " ".join(stable.split())
    assert role_kind(annot) == "readonly"
    assert final_rule in stable and "no file-writing tools" in stable
    assert layout not in stable and "register_artifact" not in stable
    assert "## Your memory" not in volatile  # no UpdateMemory tool -> no memory block


def test_review_policy_text(config, roster, tmp_path):
    cso, agents = roster

    def cso_stable():
        return system_prompt_parts(cso, run_dir=tmp_path, workspace=tmp_path, config=config, roster=agents)[0]

    assert "research turns" in cso_stable()
    config["orchestration"]["review_policy"] = "always"
    assert "whenever you delegated analyses" in cso_stable()
    config["orchestration"]["review_policy"] = "multi_specialist"
    assert "two or more specialists contributed" in cso_stable()
    config["orchestration"]["review_policy"] = "never"
    assert "does not enforce review" in cso_stable()
    config["orchestration"]["review_policy"] = "always"
    config["orchestration"]["enforce_review"] = False
    assert "does not enforce review" in cso_stable()
    config["orchestration"]["cso_tools"] = "upstream"
    assert "upstream CSO tool set" in cso_stable()


def test_project_memory_injection(config, roster, tmp_path):
    cso, agents = roster
    g = agents["genomics-analyst"]
    assert g.memory == "project" and "UpdateMemory" in g.tools
    _, volatile = system_prompt_parts(g, run_dir=tmp_path, workspace=tmp_path / "work" / g.name, config=config)
    assert "## Your memory" in volatile and "No notes yet" in volatile and "UpdateMemory" in volatile

    mem = memory_path(tmp_path, g.name)
    mem.parent.mkdir(parents=True)
    mem.write_text("\n".join(f"note line {i}" for i in range(1, 251)) + "\n")
    stable, volatile = system_prompt_parts(g, run_dir=tmp_path, workspace=tmp_path / "work" / g.name,
                                           config=config)
    assert "note line 1\n" in volatile and "note line 200" in volatile
    assert "note line 201" not in volatile and "50 more lines" in volatile
    assert "note line" not in stable
    for phrase in ("key findings", "file paths", "data quirks", "approaches that failed"):
        assert phrase in volatile

    _, cso_volatile = system_prompt_parts(cso, run_dir=tmp_path, workspace=tmp_path, config=config, roster=agents)
    assert "## Your memory" not in cso_volatile


def test_agent_name_tag_only_for_mock(config, roster, tmp_path):
    _, agents = roster
    g = agents["genomics-analyst"]
    assert f"<agent-name>{g.name}</agent-name>" in system_prompt(g, run_dir=tmp_path, workspace=tmp_path,
                                                                 config=config)
    config["provider"]["name"] = "anthropic"
    assert "<agent-name>" not in system_prompt(g, run_dir=tmp_path, workspace=tmp_path, config=config)
    assert g.settings(config).extra["agent_name"] == g.name


def test_definition_validation():
    with pytest.raises(ValueError):
        AgentDefinition(name="x", description="", prompt="", memory="global")
    with pytest.raises(ValueError):
        AgentDefinition(name="x", description="", prompt="", workspace="home")


def test_no_web_profile_roster():
    cfg = load_config(["mock", "no-web"])
    cso, agents = load_roster(cfg)
    sc = agents["single-cell-analyst"]
    assert not sc.has_tool("WebSearch") and sc.has_tool("mcp__pubmed__fetch_abstracts")
    stable, _ = system_prompt_parts(sc, run_dir=Path("/tmp/r"), workspace=Path("/tmp/r/w"), config=cfg)
    assert "DISABLED" in stable and "2025/01/31" in stable


async def test_literature_max_date_alone_reaches_the_pubmed_child_env(config, monkeypatch):
    """Only web.literature_max_date set (no tool_env): PubMed stays on the
    roster, so the server must still get the ceiling via VBT_LITERATURE_MAXDATE."""
    from conftest import open_scripted_session
    from vbt.config import LITERATURE_MAXDATE_ENV, base_tool_env
    from vbt.tools.mcp_bridge import MCPBridge, MCPServerConfig

    monkeypatch.delenv(LITERATURE_MAXDATE_ENV, raising=False)
    config["web"].update({"enabled": False, "literature_max_date": "2025/01/31"})
    (config.get("tool_env") or {}).pop(LITERATURE_MAXDATE_ENV, None)
    _, agents = load_roster(config)
    assert agents["single-cell-analyst"].has_tool("mcp__pubmed__search_pubmed")
    assert base_tool_env(config)[LITERATURE_MAXDATE_ENV] == "2025/01/31"

    session = await open_scripted_session(config, {"cso": []})
    try:
        env = session.rt.tool_env()
        assert env[LITERATURE_MAXDATE_ENV] == "2025/01/31"
        bridge = MCPBridge([], extra_env=env, log_dir=session.run.dir / "logs")
        child = bridge.child_env(MCPServerConfig(name="pubmed", command="python"))
        assert child[LITERATURE_MAXDATE_ENV] == "2025/01/31"
    finally:
        await session.close()

    # the prompt's date wins over a disagreeing tool_env value
    config.setdefault("tool_env", {})[LITERATURE_MAXDATE_ENV] = "2026/01/01"
    assert base_tool_env(config)[LITERATURE_MAXDATE_ENV] == "2025/01/31"
    # no web ceiling: tool_env passes through unchanged
    config["web"]["literature_max_date"] = None
    assert base_tool_env(config)[LITERATURE_MAXDATE_ENV] == "2026/01/01"
