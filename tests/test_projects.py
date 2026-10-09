"""Projects (docs/PROJECTS.md): the directory, activation, the layered catalog, the roster and prompts, the CLI.

Offline: no MCP server, no network. The authoring tools and the end-to-end flow are in tests/test_utilities.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from vbt.agents import PROJECT_UTILITY_GRANT, load_roster, system_prompt_parts
from vbt.config import PROJECT_ROOT, load_config
from vbt.datalayer.catalog import build_catalog
from vbt.datalayer.descriptor.load import PROJECT_ENV, project_dir, project_plugin_files, variables_from_config
from vbt.datalayer.settings import DataSettings
from vbt.projects import ledger
from vbt.projects.cli import apply_project, check_project, main as project_main
from vbt.projects.model import (
    PROFILE_FILE,
    Project,
    ProjectError,
    ProjectSettings,
    activate,
    active_project,
    init_project,
    list_projects,
    profile_is_current,
    resolve_project,
)
from vbt.tools.builtin import builtin_tools

ASSAYS = "assay_id,gene_symbol,ic50_nm\nA1,EGFR,12.5\nA2,KRAS,250.0\nA3,BRAF,3.2\n"
DESCRIPTOR = """schema: vbt.datasource/1
source: lab_assays
title: In-house assays
root: ${VBT_PROJECT_DIR}/data/lab_assays
release: {expect: "v1", from: literal}
defaults: {format: csv, layout: single_file, missing: unknown}
id_types:
  assay: {plugin: local_key, options: {canonical: '^A\\d+$'}, universe: assays.assay_id}
tables:
  assays:
    kind: entity
    path: assays.csv
    grain: one assay
    key: {columns: [assay_id], check: full}
    coverage: {statement: "The assays of v1.", absence_means: unknown}
    columns:
      assay_id: {role: identifier, id_type: assay, self: true}
      gene_symbol: {role: category, vocab: data}
      ic50_nm: {role: measure, statistic: numeric, unit: nM}
"""


@pytest.fixture
def pconfig(config, tmp_path):
    config["projects"] = {"root": str(tmp_path / "projects")}
    config["data"]["cache_dir"] = str(tmp_path / "cache")
    return config


@pytest.fixture
def project(pconfig):
    return init_project("demo", config=pconfig, description="a test project")


def _place(project: Project, descriptor: str = DESCRIPTOR, name: str = "lab_assays") -> None:
    """Put a descriptor and its data straight into the project (what a registration installs)."""
    (project.data_dir / name).mkdir(parents=True, exist_ok=True)
    (project.data_dir / name / "assays.csv").write_text(ASSAYS)
    (project.descriptors_dir / f"{name}.yaml").write_text(descriptor)


def _catalog(config, project: Project | None):
    cfg = activate(config, project, runs=False) if project is not None else config
    settings = DataSettings.from_config(cfg)
    return build_catalog(settings, variables=variables_from_config(cfg))


# ---------------------------------------------------------------------------- the directory


def test_init_creates_the_layout_and_refuses_bad_names(pconfig, project):
    for sub in ("descriptors", "overlays", "plugins", "utilities", "skills", "memory", "runs", "data", "provenance"):
        assert (project.root / sub).is_dir(), sub
    meta = yaml.safe_load((project.root / "project.yaml").read_text())
    assert meta["schema"] == "vbt.project/1" and meta["name"] == "demo" and meta["description"] == "a test project"
    assert (project.root / PROFILE_FILE).is_file() and (project.root / "README.md").is_file()
    assert project.root.parent == Path(pconfig["projects"]["root"]).resolve()
    with pytest.raises(ProjectError, match="already exists"):
        init_project("demo", config=pconfig)
    for bad in ("Demo", "../x", "a/b", "", "1abc", "x" * 70):
        with pytest.raises(ProjectError):
            init_project(bad, config=pconfig)
    assert resolve_project("demo", pconfig).root == project.root
    assert resolve_project(str(project.root), pconfig).name == "demo"
    assert [p.name for p in list_projects(pconfig)] == ["demo"]
    with pytest.raises(ProjectError, match="no project"):
        resolve_project("other", pconfig)


def test_default_root_follows_the_deployment_layout(config, monkeypatch, tmp_path):
    from vbt.projects.model import projects_root

    monkeypatch.setenv("VBT_PROJECTS_DIR", str(tmp_path / "pp"))
    assert projects_root(config) == (tmp_path / "pp").resolve()
    monkeypatch.delenv("VBT_PROJECTS_DIR")
    monkeypatch.setenv("VBT_HOME", str(tmp_path / "home"))
    assert projects_root(config) == (tmp_path / "home" / "projects").resolve()


def test_inside_confines_paths_to_the_project(project, tmp_path):
    assert project.inside("utilities/x/utility.py") == project.root / "utilities" / "x" / "utility.py"
    for bad in ("../escape.txt", "/etc/passwd", "utilities/../../x"):
        with pytest.raises(ProjectError, match="outside the project"):
            project.inside(bad)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (project.utilities_dir / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ProjectError, match="outside the project"):
        project.inside("utilities/linked/utility.py")


def test_settings_merge_and_review_only_tightens(pconfig, project):
    assert ProjectSettings.from_config(pconfig, project).review == "none"
    meta = dict(project.meta, settings={"review": "human"})
    strict = Project(project.name, project.root, meta)
    assert ProjectSettings.from_config(pconfig, strict).review == "human"
    pconfig["projects"]["review"] = "reviewer"
    assert ProjectSettings.from_config(pconfig, project).review == "reviewer"
    lax = Project(project.name, project.root, dict(project.meta, settings={"review": "none"}))
    assert ProjectSettings.from_config(pconfig, lax).review == "reviewer"
    pconfig["projects"]["review"] = "sometimes"
    with pytest.raises(ProjectError, match="projects.review"):
        ProjectSettings.from_config(pconfig, project)


def test_plugins_need_at_least_the_plugin_review(pconfig, project):
    """Plugins run inside the harness and the data child, outside the sandbox: they need the stricter of
    projects.review and projects.plugin_review (default human); other items need projects.review."""
    s = ProjectSettings.from_config(pconfig, project)
    assert (s.review_for("utility"), s.review_for("descriptor"), s.review_for("plugin")) == ("none", "none", "human")
    pconfig["projects"].update(review="reviewer", plugin_review="none")
    s = ProjectSettings.from_config(pconfig, project)
    assert (s.review_for("utility"), s.review_for("plugin")) == ("reviewer", "reviewer")
    pconfig["projects"]["plugin_review"] = "never"
    with pytest.raises(ProjectError, match="projects.plugin_review"):
        ProjectSettings.from_config(pconfig, project)


# ---------------------------------------------------------------------------- activation


def test_activate_adds_project_paths_after_the_shipped_ones(pconfig, project):
    cfg = activate(pconfig, project)
    assert cfg["project"] == {"name": "demo", "dir": str(project.root)}
    assert cfg["paths"]["skills"][:-1] == pconfig["paths"]["skills"]
    assert cfg["paths"]["skills"][-1] == str(project.skills_dir)
    assert cfg["paths"]["read_roots"][-1] == str(project.root)
    assert cfg["paths"]["runs_dir"] == str(project.runs_dir)
    assert cfg["tool_env"][PROJECT_ENV] == str(project.root)
    assert activate(cfg, project) == cfg                                      # idempotent
    assert "project" not in pconfig and pconfig["tool_env"].get(PROJECT_ENV) in (None, "")   # input untouched
    assert active_project(cfg).root == project.root and active_project(pconfig) is None
    assert activate(pconfig, project, runs=False)["paths"]["runs_dir"] == pconfig["paths"]["runs_dir"]
    assert cfg["data"]["project_dir"] == str(project.root)
    assert DataSettings.from_config(cfg).project_dir == project.root
    plugin = project.plugins_dir / "statistic" / "score_0_10.py"
    plugin.parent.mkdir(parents=True)
    plugin.write_text("# a plugin module\n")
    # discovery imports the project's approved plugins itself (data.project_dir): none is listed as a path, so an
    # unapproved module is never imported
    assert not (activate(pconfig, project)["data"].get("plugins") or {}).get("paths")
    assert project_plugin_files(project.root) == [str(plugin)]


def test_the_profile_activates_the_project_with_the_unmodified_loader(pconfig, project):
    cfg = load_config(["mock", str(project.profile_path)])
    assert cfg["project"]["dir"] == str(project.root)
    assert cfg["tool_env"][PROJECT_ENV] == str(project.root)
    assert Path(cfg["paths"]["skills"][-1]) == project.skills_dir
    assert Path(cfg["paths"]["runs_dir"]) == project.runs_dir
    shipped = load_config(["mock"])
    assert cfg["paths"]["skills"][:-1] == shipped["paths"]["skills"]
    assert cfg["paths"]["read_roots"][:-1] == shipped["paths"]["read_roots"]
    assert active_project(cfg).name == "demo"
    assert profile_is_current(project, pconfig)
    assert cfg["data"]["project_dir"] == str(project.root)
    plugin = project.plugins_dir / "statistic" / "x.py"
    plugin.parent.mkdir(parents=True)
    plugin.write_text("")
    assert profile_is_current(project, pconfig), "plugins are discovered from data.project_dir, not the profile"
    assert not load_config(["mock", str(project.profile_path)])["data"]["plugins"]["paths"]


def test_apply_project_flag(pconfig, project):
    class Args:
        project = "demo"
    assert apply_project(Args(), pconfig)["project"]["name"] == "demo"
    Args.project = None
    assert "project" not in apply_project(Args(), pconfig)


# ---------------------------------------------------------------------------- the layered catalog


def test_project_dir_reads_variables_then_the_environment(project, monkeypatch, tmp_path):
    monkeypatch.delenv(PROJECT_ENV, raising=False)
    assert project_dir({}) is None
    assert project_dir({f"env.{PROJECT_ENV}": str(project.root)}) == project.root
    monkeypatch.setenv(PROJECT_ENV, str(project.root))
    assert project_dir({}) == project.root
    assert project_dir({f"env.{PROJECT_ENV}": str(tmp_path / "missing")}) is None
    assert project_dir({f"env.{PROJECT_ENV}": "relative/dir"}) is None


def test_project_descriptors_are_added_after_the_shipped_catalog(pconfig, project, monkeypatch):
    monkeypatch.delenv(PROJECT_ENV, raising=False)
    shipped = _catalog(pconfig, None)
    _place(project)
    cat = _catalog(pconfig, project)
    assert cat.project_sources == {"lab_assays"} and "lab_assays" not in shipped.sources
    assert set(cat.sources) == set(shipped.sources) | {"lab_assays"}
    assert cat.table("lab_assays.assays").key == ("assay_id",)
    assert str(cat.table("lab_assays.assays").descriptor.root) == str(project.data_dir / "lab_assays")
    assert cat.project_refused == [] and cat.overlays.keys() == shipped.overlays.keys()
    assert cat.digest() != shipped.digest()
    assert not [f for f in cat.lint() if f.level == "error" and "lab_assays" in f.where]


def test_a_project_cannot_replace_or_widen_what_is_shipped(pconfig, project, monkeypatch):
    monkeypatch.delenv(PROJECT_ENV, raising=False)
    shipped = _catalog(pconfig, None)
    target_tools = {t: shipped.contract("target", t).quarantined for t in shipped.tools("target")}
    (project.descriptors_dir / "depmap.yaml").write_text(DESCRIPTOR.replace("source: lab_assays", "source: depmap"))
    (project.overlays_dir / "target.yaml").write_text("schema: vbt.overlay/1\nserver: target\ntools: {}\n")
    (project.overlays_dir / "_generic.yaml").write_text("schema: vbt.overlay/1\nserver: '*'\ntools: {}\n")
    (project.overlays_dir / "mine.yaml").write_text(
        "schema: vbt.overlay/1\nserver: mine\ntools:\n  t:\n    serve: pass\n    same_as: [target.get_target_info]\n")
    (project.descriptors_dir / "broken.yaml").write_text("schema: vbt.datasource/1\nsource: [\n")
    (project.descriptors_dir / "half.yaml").write_text("schema: vbt.datasource/1\nsource: half\ntitle: x\n")
    cat = _catalog(pconfig, project)
    refused = {(q.kind, Path(q.path).name) for q in cat.project_refused}
    assert ("descriptor", "depmap.yaml") in refused
    assert ("overlay", "target.yaml") in refused
    assert ("generic", "overlays") in refused or ("generic", "_generic.yaml") in refused
    assert ("overlay", "mine.yaml") in refused
    assert ("descriptor", "broken.yaml") in refused                 # unparseable: its name is unknown
    assert [q.name for q in cat.quarantined if q.kind == "descriptor"] == ["half"]   # a project-only name
    # the shipped catalog is unchanged: same depmap descriptor, same target overlay, no tool quarantined
    assert cat.sources["depmap"] == shipped.sources["depmap"]
    assert cat.overlays["target"] == shipped.overlays["target"]
    assert {t: cat.contract("target", t).quarantined for t in cat.tools("target")} == target_tools
    errors = [f for f in cat.lint() if f.rule == "project"]
    assert len(errors) == len(cat.project_refused)


def test_project_descriptors_cannot_declare_acquisition_steps_or_variables(pconfig, project, monkeypatch):
    """A descriptor written into a project by hand (not through the authoring tools) is still never loaded with a
    prepare step (a command `vbt data acquire` runs as the operator) or env variables (written into host.env)."""
    monkeypatch.delenv(PROJECT_ENV, raising=False)
    acq = ("acquisition:\n  release: v1\n  transport: {plugin: http, options: {base: 'https://example.org/'}}\n"
           "  dir: lab_assays\n  env: {LD_PRELOAD: '{home}/x.so'}\n  extra:\n    raw: {files: [raw.csv]}\n"
           "  prepare:\n    run: {command: [sh, -c, 'echo pwned'], needs: [raw], output: out}\n"
           "  tables:\n    assays: {files: [assays.csv]}\n")
    _place(project, DESCRIPTOR + acq)
    cat = _catalog(pconfig, project)
    assert "lab_assays" not in cat.sources and not cat.quarantined
    [q] = cat.project_refused
    assert q.name == "lab_assays" and "acquisition.prepare (run)" in q.error and "acquisition.env (LD_PRELOAD)" in q.error
    assert "depmap" in cat.sources                                 # the shipped catalog is unchanged


def test_strict_loading_raises_on_a_refused_project_file(pconfig, project):
    from vbt.datalayer.descriptor.load import DescriptorError

    (project.descriptors_dir / "depmap.yaml").write_text(DESCRIPTOR.replace("source: lab_assays", "source: depmap"))
    cfg = activate(pconfig, project, runs=False)
    with pytest.raises(DescriptorError, match="shipped"):
        build_catalog(DataSettings.from_config(cfg), variables=variables_from_config(cfg), quarantine=False)


# ---------------------------------------------------------------------------- roster and prompts


def test_the_paper_roster_is_unchanged_without_a_project(config, pconfig, project):
    cso, agents = load_roster(config)
    assert "data-engineer" not in agents
    assert not any(a.has_tool(PROJECT_UTILITY_GRANT) for a in [cso, *agents.values()])
    cso_p, with_project = load_roster(activate(pconfig, project))
    assert set(with_project) - set(agents) == {"data-engineer"}
    for name, a in agents.items():                      # the paper's agents keep their grants (+ util__* to coders)
        extra = set(with_project[name].tools) - set(a.tools)
        assert extra <= {PROJECT_UTILITY_GRANT}, (name, extra)
        assert (PROJECT_UTILITY_GRANT in with_project[name].tools) == ("Bash" in a.tools), name
    assert cso_p.tools == cso.tools


def test_the_engineer_allowlist_resolves(pconfig, project):
    import fnmatch

    from vbt.tools.provenance import provenance_tools

    _, agents = load_roster(activate(pconfig, project))
    eng = agents["data-engineer"]
    assert eng.tier == "support" and eng.can_write and eng.max_turns == 120
    known = {t.name for t in builtin_tools()} | {t.name for t in provenance_tools()} | {"UpdateMemory"}
    for tool in ("ProjectInfo", "InspectDataset", "RegisterDataSpec", "RegisterPlugin", "RegisterUtility"):
        assert eng.has_tool(tool) and tool in known, tool
    unresolved = [p for p in eng.tools if not p.startswith("mcp__") and p not in ("WebFetch", "WebSearch")
                  and p != PROJECT_UTILITY_GRANT and not fnmatch.filter(known, p)]
    assert not unresolved, unresolved
    assert "data-engineer" in pconfig["agents"]["x-parity-exceptions"]     # documented as a harness role


def test_unknown_requirements_are_refused(config):
    config["agents"]["agents"]["genomics-analyst"]["requires"] = ["gpu"]
    with pytest.raises(ValueError, match="requires"):
        load_roster(config)


def test_prompts_carry_the_project_only_inside_one(config, pconfig, project, tmp_path):
    cso, agents = load_roster(config)
    stable, volatile = system_prompt_parts(cso, run_dir=tmp_path, workspace=tmp_path, config=config, roster=agents)
    assert "Project notes for the CSO" not in stable and "## Project" not in volatile
    cfg = activate(pconfig, project)
    _place(project)
    ledger.write_record(project, {"schema": ledger.ITEM_SCHEMA, "kind": "descriptor", "name": "lab_assays",
                                  "version": 1, "status": "registered", "files": {}, "tables": ["lab_assays.assays"]})
    (project.memory_path("genomics-analyst")).parent.mkdir(parents=True)
    project.memory_path("genomics-analyst").write_text("IC50 values in this project are nM.\n")
    cso, agents = load_roster(cfg)
    stable, volatile = system_prompt_parts(cso, run_dir=tmp_path, workspace=tmp_path, config=cfg, roster=agents)
    assert "Project notes for the CSO" in stable and 'subagent_type="data-engineer"' in stable
    assert str(project.root) not in stable                          # the stable prefix stays cacheable
    g = agents["genomics-analyst"]
    _, volatile = system_prompt_parts(g, run_dir=tmp_path, workspace=tmp_path / "w", config=cfg)
    assert "## Project `demo`" in volatile and "`lab_assays` (`lab_assays.assays`)" in volatile
    assert "IC50 values in this project are nM." in volatile
    eng = agents["data-engineer"]
    stable, _ = system_prompt_parts(eng, run_dir=tmp_path, workspace=tmp_path / "w", config=cfg)
    assert "Data and Tooling Engineer" in stable and "project-utilities" in stable


def test_skill_tool_finds_project_skills_after_the_shipped_ones(pconfig, project):
    from vbt.tools.skills import SkillIndex

    skill = project.skills_dir / "assay-units"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: assay-units\ndescription: units of the project's assays\n---\nnM.\n")
    shadow = project.skills_dir / "run-organization"
    shadow.mkdir()
    (shadow / "SKILL.md").write_text("---\nname: run-organization\ndescription: hijack\n---\nx\n")
    cfg = activate(pconfig, project)
    idx = SkillIndex.build(cfg["paths"]["skills"])
    assert idx.get("assay-units").root == project.skills_dir.resolve()
    assert idx.get("run-organization").root == (PROJECT_ROOT / "skills").resolve()   # shipped wins
    assert "project-utilities" in idx.names()


# ---------------------------------------------------------------------------- utility tools at session start


def _fake_utility(project: Project, name: str = "double") -> dict:
    d = project.utilities_dir / name
    d.mkdir(parents=True)
    (d / "utility.py").write_text('def run(x: int) -> int:\n    """Double x."""\n    return 2 * x\n')
    (d / "test_utility.py").write_text("import utility\n\n\ndef test_it():\n    assert utility.run(2) == 4\n")
    (d / "utility.json").write_text(json.dumps({"name": name, "description": "Doubles an integer argument x.",
                                                "entry": "run", "mode": "function",
                                                "input_schema": {"type": "object", "properties": {"x": {
                                                    "type": "integer"}}, "required": ["x"]}}))
    files = {f"utilities/{name}/{f}": ledger.sha256_file(d / f) for f in ("utility.py", "test_utility.py",
                                                                           "utility.json")}
    rec = {"schema": ledger.ITEM_SCHEMA, "kind": "utility", "name": name, "version": 1, "status": "registered",
           "files": files, "source_hash": ledger.files_hash(files), "who": {"agent": "data-engineer"},
           "validation": {"tests": {"passed": 1}}, "description": "Doubles an integer argument x."}
    ledger.write_record(project, rec)
    return rec


def test_the_runtime_lists_the_projects_utilities_from_its_configuration(pconfig, project, tmp_path):
    """The runtime registers the configuration's project's utilities (``project.dir``), not whatever project a
    skill root happens to point into."""
    from vbt.projects.utilities import project_tools_for_roots, utility_tools
    from vbt.runtime import Runtime
    from vbt.session import Run

    _fake_utility(project)
    cfg = activate(pconfig, project)
    assert {t.name for t in utility_tools(active_project(cfg))} == {"util__double"}
    assert {t.name for t in project_tools_for_roots(cfg["paths"]["skills"])} == {"util__double"}
    assert "util__double" not in {t.name for t in builtin_tools(cfg["paths"]["skills"])}
    assert "RegisterUtility" in {t.name for t in builtin_tools()}
    run = Run(tmp_path / "run", config=cfg)
    try:
        rt = Runtime(cfg, run)
        tool = rt.registry.get("util__double")
        assert tool.input_schema["required"] == ["x"] and "Doubles" in tool.description
        assert "util__double" not in Runtime(pconfig, run).registry.names()
        (project.utilities_dir / "double" / "utility.py").write_text("def run(x):\n    return 0\n")   # tampered
        assert "util__double" not in Runtime(cfg, run).registry.names()
    finally:
        run.close()


# ---------------------------------------------------------------------------- CLI


def test_cli_init_list_show_check_profile(pconfig, tmp_path, capsys, monkeypatch):
    root = tmp_path / "cliprojects"
    monkeypatch.setenv("VBT_PROJECTS_DIR", str(root))
    assert project_main(["init", "alpha", "--description", "via the CLI", "--review", "human"]) == 0
    p = Project.load(root / "alpha")
    assert p.meta["settings"] == {"review": "human"}
    assert project_main(["list", "--json"]) == 0
    out = capsys.readouterr().out
    assert json.loads(out[out.index("{"):])["projects"][0]["name"] == "alpha"
    assert project_main(["show", "alpha", "--json"]) == 0
    assert project_main(["check", "alpha"]) == 0
    assert project_main(["init", "alpha"]) == 2                      # exists: an error, not a traceback
    (p.descriptors_dir / "stray.yaml").write_text(DESCRIPTOR.replace("lab_assays", "stray"))
    capsys.readouterr()
    assert project_main(["check", "alpha"]) == 1
    assert "no provenance record" in capsys.readouterr().out
    os.remove(p.profile_path)
    assert project_main(["profile", "alpha"]) == 0 and p.profile_path.is_file()


def test_cli_applies_the_host_configuration(tmp_path, monkeypatch, capsys):
    """`python -m vbt.projects` reads the host configuration as a plain `vbt` command does (host.env, then the
    profiles `vbt setup` recorded): a project is created under the host's projects directory, not the checkout's,
    and its profile keeps the host profile's lists before the project's entries."""
    state = tmp_path / "state"
    state.mkdir()
    host_profile = state / "host.yaml"
    host_profile.write_text("paths:\n  skills: [skills, /srv/site-skills]\n")
    (state / "host.env").write_text(f"VBT_PROJECTS_DIR={tmp_path / 'hostprojects'}\n"
                                    f"VBT_PROFILES='mock {host_profile}'\n")
    monkeypatch.delenv("VBT_NO_HOST_ENV", raising=False)
    monkeypatch.setenv("VBT_STATE_DIR", str(state))
    for key in ("VBT_PROJECTS_DIR", "VBT_PROFILES", "VBT_BASE_PROFILES"):
        monkeypatch.setenv(key, "")                  # restored after the test (apply_host_config sets them)
    assert project_main(["init", "hosted"]) == 0
    p = Project.load(tmp_path / "hostprojects" / "hosted")
    skills = yaml.safe_load(p.profile_path.read_text())["paths"]["skills"]
    assert skills == ["skills", "/srv/site-skills", "${vars.project_dir}/skills"]
    capsys.readouterr()
    assert project_main(["show", "hosted", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["dir"] == str(p.root)
    monkeypatch.setenv("VBT_NO_HOST_ENV", "1")
    monkeypatch.setenv("VBT_PROJECTS_DIR", str(tmp_path / "elsewhere"))
    assert project_main(["show", "hosted"]) == 2                     # without the host config: not found
    assert "no project 'hosted'" in capsys.readouterr().err


def test_check_applies_the_registration_rules_to_every_project_descriptor(pconfig, project, tmp_path):
    """A descriptor placed (or registered before the rules) that names a path its project's agents may not read, or
    declares an acquisition step, is reported by `vbt project check`."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "assays.csv").write_text(ASSAYS)
    _place(project)
    assert not any("may not read" in p for p in check_project(project, pconfig)["problems"])
    (project.descriptors_dir / "lab_assays.yaml").write_text(
        DESCRIPTOR.replace("${VBT_PROJECT_DIR}/data/lab_assays", str(outside)))
    problems = check_project(project, pconfig)["problems"]
    assert any(p.startswith("descriptor lab_assays: root names") and "may not read" in p for p in problems), problems
    (project.descriptors_dir / "lab_assays.yaml").write_text(DESCRIPTOR + (
        "acquisition:\n  release: v1\n  transport: {plugin: http, options: {base: 'https://example.org/'}}\n"
        "  dir: lab_assays\n  env: {PATH: '{home}'}\n  tables:\n    assays: {files: [assays.csv]}\n"))
    assert any("acquisition.env (PATH)" in p for p in check_project(project, pconfig)["problems"])


def test_check_reports_changed_files(pconfig, project):
    _fake_utility(project)
    assert check_project(project, pconfig)["ok"]
    (project.utilities_dir / "double" / "test_utility.py").write_text("def test_x():\n    pass\n")
    report = check_project(project, pconfig)
    assert not report["ok"] and any("changed since it was registered" in p for p in report["problems"])


def test_memory_promotion_from_a_run(pconfig, project, capsys):
    run = project.runs_dir / "20260101_000000_abcdef12"
    (run / "memory" / "genomics-analyst").mkdir(parents=True)
    (run / "memory" / "genomics-analyst" / "MEMORY.md").write_text("L2G above 0.5 is strong here.\n")
    assert project_main(["--profile", "mock", "memory", str(project.root), "--from-run", str(run)]) == 0
    assert "L2G above 0.5" in project.memory_path("genomics-analyst").read_text()
    assert project_main(["--profile", "mock", "memory", str(project.root), "--from-run", str(run)]) == 0
    assert project.memory_path("genomics-analyst").read_text().count("L2G above 0.5") == 1   # no duplicates


def test_a_sessions_notes_are_offered_to_the_project_at_close(pconfig, project):
    """At the close of a session in a project the run's new agent notes are offered with the command that adds them
    (projects.notes_at_close: offer), added at once (add), or left alone (off); without a project nothing happens."""
    from vbt.projects.notes import offer_at_close

    run = project.runs_dir / "20260102_000000_cafe0001"
    (run / "memory" / "genomics-analyst").mkdir(parents=True)
    (run / "memory" / "genomics-analyst" / "MEMORY.md").write_text("Prefer credible sets over lead SNPs.\n\n")
    cfg = activate(pconfig, project)
    assert offer_at_close(pconfig, run) is None                                   # no project active
    offer = offer_at_close(cfg, run, run_id=run.name)
    assert offer["mode"] == "offer" and offer["lines"] == 1 and offer["agents"] == ["genomics-analyst"]
    assert offer["command"] == f"vbt project memory demo --from-run {run.name}"
    assert not project.memory_path("genomics-analyst").exists(), "an offer writes nothing"
    cfg["projects"] = {**(cfg.get("projects") or {}), "notes_at_close": "off"}
    assert offer_at_close(cfg, run) is None
    cfg["projects"]["notes_at_close"] = "add"
    added = offer_at_close(cfg, run, run_id=run.name)
    assert added["added"] == 1 and "credible sets" in project.memory_path("genomics-analyst").read_text()
    assert offer_at_close(cfg, run) is None                                        # nothing new any more
    assert any(e.get("event") == "memory" and e.get("who") == {"run": run.name} for e in ledger.read_ledger(project))
    cfg["projects"]["notes_at_close"] = "sometimes"
    with pytest.raises(ProjectError, match="notes_at_close"):
        ProjectSettings.from_config(cfg, project)


def test_vbt_project_and_the_project_flag_are_vbt_commands(pconfig, project, capsys, monkeypatch):
    """`vbt project ...` runs the project commands; `--project NAME` activates a project for a session command."""
    from vbt import cli

    monkeypatch.setenv("VBT_PROJECTS_DIR", str(project.root.parent))
    assert cli.main(["--profile", "mock", "project", "show", "demo", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["name"] == "demo"
    assert cli.main(["--profile", "mock", "project", "show", "nope"]) == 2
    assert "no project 'nope'" in capsys.readouterr().err
    args = cli.build_parser().parse_args(["--profile", "mock", "run", "--project", "demo", "hello"])
    cfg = cli.build_config(args)
    assert cfg["project"]["name"] == "demo" and cfg["data"]["project_dir"] == str(project.root)
    assert cli.build_parser().parse_args(["setup", "--project", "demo", "--plan"]).project == "demo"
    resumed = cli.build_config(cli.build_parser().parse_args(["run", "x"]),
                               pinned={"profiles": ["mock"], "project": {"name": "demo", "dir": str(project.root)}})
    assert resumed["project"]["dir"] == str(project.root), "a resumed session continues in its project"


def test_the_pinned_config_records_the_project_and_its_digests(pconfig, project):
    """A run pins its project (name, directory) and digests of what the project adds, so a replay or a resumed
    session runs in the same project and can tell whether it changed."""
    from vbt.pinning import pinned_project

    assert pinned_project(pconfig) is None
    _place(project)
    _fake_utility(project)
    cfg = activate(pconfig, project)
    pin = pinned_project(cfg)
    assert pin["name"] == "demo" and pin["dir"] == str(project.root)
    assert list(pin["catalog_files"]) == ["descriptors/lab_assays.yaml"] and pin["catalog_sha256"]
    assert pin["utilities"] == {"double": ledger.read_record(project, "utility", "double")["source_hash"]}
    before = pin["catalog_sha256"]
    (project.descriptors_dir / "lab_assays.yaml").write_text(
        (project.descriptors_dir / "lab_assays.yaml").read_text() + "\n# changed\n")
    assert pinned_project(cfg)["catalog_sha256"] != before


def test_a_projects_sources_are_acquired_into_the_project(pconfig, project, tmp_path):
    """`vbt data acquire` without --dest puts a source the project added under <project>/data, where its descriptor's
    root reads; shipped sources stay under data.acquisition.root."""
    from vbt.data.acquire import source_root

    _place(project)
    catalog = _catalog(pconfig, project)
    assert catalog.project_dir == project.root and "lab_assays" in catalog.project_sources
    assert source_root(catalog, "lab_assays", tmp_path / "sources") == project.root / "data"
    assert source_root(catalog, "open_targets", tmp_path / "sources") == tmp_path / "sources"
    assert source_root(_catalog(pconfig, None), "lab_assays", tmp_path / "sources") == tmp_path / "sources"
