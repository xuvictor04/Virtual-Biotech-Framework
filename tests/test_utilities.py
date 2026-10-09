"""The system creates project-specific utilities as needed (docs/PROJECTS.md): the authoring tools.

Offline. A scripted (mock-provider) data engineer is handed an unknown CSV and Parquet dataset and a missing helper,
registers descriptors and a utility (both validated: ``vbt ds lint``/``check`` on the data, the utility's tests in
the sandbox), and a later turn uses them through the gateway (the real data child, started by the session) and the
``util__`` tool. Refusals: a failing lint, a failing check, a shipped name, a failing conformance suite, failing
tests, writes outside the project, and no project at all. Review: the scientific reviewer and a human.
"""

from __future__ import annotations

import importlib.util
import os
import itertools
import json
import shutil
from pathlib import Path

import pytest

from conftest import scripted_provider
from vbt.projects import ledger
from vbt.projects.authoring import approve_pending, pending_items
from vbt.projects.model import ProjectError, activate, init_project
from vbt.projects.sandbox import bwrap_works, sandbox_plan
from vbt.projects.utilities import utility_tools
from vbt.providers.base import ToolCall
from vbt.providers.mock import ScriptedProvider, reply
from vbt.runtime import Runtime
from vbt.session import Run
from vbt.tools.base import ToolContext, ToolFailure

ENGINEER = "data-engineer"
HAVE_ARROW = importlib.util.find_spec("pyarrow") is not None
HAVE_MCP = importlib.util.find_spec("fastmcp") is not None and importlib.util.find_spec("mcp") is not None
needs_arrow = pytest.mark.skipif(not HAVE_ARROW, reason="pyarrow is needed to read the datasets")

_ids = itertools.count(1)

ASSAYS = ("assay_id,gene_symbol,cell_line,ic50_nm,tissue\nA1,EGFR,A549,12.5,lung\nA2,KRAS,HCT116,250.0,colon\n"
          "A3,BRAF,A375,3.2,skin\nA4,EGFR,PC9,0.8,lung\n")
DESCRIPTOR = """schema: vbt.datasource/1
source: lab_assays
title: In-house drug sensitivity assays
root: ${VBT_PROJECT_DIR}/data/lab_assays
release: {expect: "v1", from: literal}
defaults: {format: csv, layout: single_file, missing: unknown}
id_types:
  assay: {plugin: local_key, options: {canonical: '^A\\d+$'}, universe: assays.assay_id}
tables:
  assays:
    kind: entity
    path: assays.csv
    grain: one assay measurement
    key: {columns: [assay_id], check: full}
    coverage: {statement: "The in-house assays of release v1.", absence_means: unknown}
    columns:
      assay_id: {role: identifier, id_type: assay, self: true}
      gene_symbol: {role: category, vocab: data}
      cell_line: {role: category, vocab: data}
      ic50_nm: {role: measure, statistic: numeric, unit: nM}
      tissue: {role: category, vocab: data}
"""
CODE = '''"""IC50 summaries."""
import math


def run(values: list) -> dict:
    """Geometric mean of the positive IC50 values (nM) and how many were used."""
    vals = [float(v) for v in values if v is not None and float(v) > 0]
    if not vals:
        return {"n": 0, "geomean": None}
    return {"n": len(vals), "geomean": math.exp(sum(math.log(v) for v in vals) / len(vals))}
'''
TESTS = '''import utility


def test_known_answer():
    r = utility.run([1, 100])
    assert r["n"] == 2 and abs(r["geomean"] - 10.0) < 1e-9


def test_empty_is_not_zero(tmp_path):
    assert utility.run([None, -1]) == {"n": 0, "geomean": None}
'''
SCHEMA = {"type": "object", "properties": {"values": {"type": "array", "items": {"type": ["number", "null"]}}},
          "required": ["values"]}


def tc(tool: str, **args) -> ToolCall:
    return ToolCall(id=f"call_d5_{next(_ids)}", name=tool, input=args)


@pytest.fixture
def pconfig(config, tmp_path):
    config["projects"] = {"root": str(tmp_path / "projects")}
    config["data"]["cache_dir"] = str(tmp_path / "cache")
    config.setdefault("preflight", {})["skip"] = True
    config["orchestration"]["enforce_review"] = False
    return config


@pytest.fixture
def project(pconfig):
    return init_project("demo", config=pconfig)


def _runtime(pconfig, project, rules=None):
    cfg = activate(pconfig, project)
    run = Run(Path(cfg["paths"]["runs_dir"]), config=cfg)
    return Runtime(cfg, run, provider=ScriptedProvider.from_rules(rules or {}))


@pytest.fixture
def rt(pconfig, project):
    runtime = _runtime(pconfig, project)
    yield runtime
    runtime.run.close()


def _ctx(rt, agent=ENGINEER, tid=None):
    return ToolContext(agent, rt.run, rt, tool_call_id=tid or f"tu_{next(_ids)}")


async def _call(rt, tool, agent=ENGINEER, **args):
    return await rt.registry.get(tool)(_ctx(rt, agent), args)


async def _refused(rt, tool, agent=ENGINEER, **args) -> str:
    with pytest.raises(ToolFailure) as ei:
        await _call(rt, tool, agent, **args)
    return str(ei.value)


def _write(rt, rel: str, text: str, agent=ENGINEER) -> Path:
    p = rt.workspace_for(agent) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def _utility_dir(rt, name="ic50_geomean", code=CODE, tests=TESTS) -> Path:
    _write(rt, f"utilities/{name}/utility.py", code)
    _write(rt, f"utilities/{name}/test_utility.py", tests)
    return rt.workspace_for(ENGINEER) / "utilities" / name


def _events(project, kind=None):
    return [e for e in ledger.read_ledger(project) if kind is None or e.get("kind") == kind]


# ---------------------------------------------------------------------------- end to end (gateway + data child)


@pytest.mark.skipif(not (HAVE_ARROW and HAVE_MCP), reason="the data child needs pyarrow, fastmcp and mcp")
async def test_engineer_creates_descriptors_and_a_utility_that_a_later_turn_uses(pconfig, project, tmp_path):
    """Turn 1: the engineer is handed an unknown CSV and Parquet dataset and a missing helper; it inspects them,
    registers two descriptors (lint + check) and a utility (tests in the sandbox). Turn 2: a specialist reads the
    new tables through the gateway and calls the new tool."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from vbt.datalayer import build_gateway
    from vbt.orchestrator import open_session

    cfg = activate(pconfig, project)
    cfg["mcp_servers"] = {"servers": build_gateway(cfg).extra_servers()}
    incoming = tmp_path / "incoming"                       # the user dropped files here (a read root)
    incoming.mkdir()
    pq.write_table(pa.table({"line_id": ["L001", "L002", "L003"], "doubling_h": [21.5, 33.0, None],
                             "lineage": ["lung", "lung", "skin"]}), incoming / "lines.parquet")
    cfg["paths"]["read_roots"].append(str(incoming))
    seen: dict[str, str] = {}

    def keep(key, item):
        def f(msgs):
            seen[key] = "\n".join(str(getattr(b, "content", "")) for b in msgs[-1].content)
            return item
        return f

    def draft_parquet(msgs):
        info = json.loads(next(str(b.content) for b in msgs[-1].content))
        seen["inspect"] = info
        return reply(tc("RegisterDataSpec", kind="descriptor", content=info["draft_descriptor"],
                        files=[str(incoming / "lines.parquet")], why="user-supplied cell line table"))

    rules = {
        "cso": [reply(tc("Task", subagent_type=ENGINEER, prompt="Register the assay CSV, lines.parquet and an IC50 "
                                                                 "helper")),
                reply("Registered."),
                reply(tc("Task", subagent_type="genomics-analyst", prompt="Summarise lung IC50s")),
                reply("Lung IC50 geometric mean computed.")],
        ENGINEER: [
            reply(tc("ProjectInfo")),
            reply(tc("Write", file_path="incoming/assays.csv", content=ASSAYS)),
            reply(tc("Write", file_path="drafts/lab_assays.yaml", content=DESCRIPTOR)),
            reply(tc("RegisterDataSpec", kind="descriptor", path="drafts/lab_assays.yaml",
                     files=["incoming/assays.csv"], why="user-supplied assay table")),
            keep("register_csv", reply(tc("InspectDataset", path=str(incoming / "lines.parquet"), source="cell_lines",
                                          table="lines"))),
            draft_parquet,
            keep("register_parquet", reply(tc("Write", file_path="utilities/ic50_geomean/utility.py", content=CODE))),
            reply(tc("Write", file_path="utilities/ic50_geomean/test_utility.py", content=TESTS)),
            reply(tc("RegisterUtility", name="ic50_geomean", description="Geometric mean of positive IC50 values "
                     "(nM), with the number of values used.", input_schema=SCHEMA,
                     directory="utilities/ic50_geomean", why="IC50 summaries are recomputed in every delegation")),
            keep("register_util", reply("Registered lab_assays.assays, cell_lines.lines and util__ic50_geomean.")),
        ],
        "genomics-analyst": [
            reply(tc("mcp__data__find", table="lab_assays.assays", where={"tissue": "lung"})),
            keep("find", reply(tc("mcp__data__find", table="cell_lines.lines", where={"lineage": "lung"}))),
            keep("find_parquet", reply(tc("util__ic50_geomean", values=[12.5, 0.8]))),
            keep("util", reply("The lung geometric mean IC50 is 3.16 nM over 2 assays.")),
        ],
    }
    session = await open_session(cfg, provider=scripted_provider(rules), start_mcp=True)
    try:
        assert "mcp__data__find" in session.rt.registry.names()
        assert not [n for n in session.rt.registry.names() if n.startswith("util__")]
        assert await session.ask("Register our assay data and an IC50 helper.") == "Registered."
        assert "util__ic50_geomean" in session.rt.registry.names()
        assert await session.ask("Summarise lung IC50s.") == "Lung IC50 geometric mean computed."
    finally:
        await session.close()

    for key in ("register_csv", "register_parquet", "register_util"):
        assert '"status": "registered"' in seen[key], (key, seen[key][:2000])
    assert '"available": "now' in seen["register_csv"]
    assert seen["inspect"]["rows"] == 3 and seen["inspect"]["format"] == "parquet"
    found = json.loads(seen["find"])
    assert found["_vbt"]["status"] == "ok" and found["_vbt"]["total"] == 2
    assert sorted(r["assay_id"] for r in found["rows"]) == ["A1", "A4"]
    parquet_rows = json.loads(seen["find_parquet"])
    assert sorted(r["line_id"] for r in parquet_rows["rows"]) == ["L001", "L002"]
    util = json.loads(seen["util"])
    assert util["n"] == 2 and abs(util["geomean"] - 10 ** 0.5) < 1e-9

    # stored with the project, with provenance, and listed in the run's MANIFEST and audit.html
    for kind, name in (("descriptor", "lab_assays"), ("descriptor", "cell_lines"), ("utility", "ic50_geomean")):
        rec = ledger.read_record(project, kind, name)
        assert rec["status"] == "registered" and rec["version"] == 1 and not ledger.verify(project, rec)
        assert rec["who"]["agent"] == ENGINEER and rec["who"]["run_id"] == session.run.run_id
        assert rec["why"] and rec["when"] and rec["source_hash"].startswith("sha256:")
    tests = ledger.read_record(project, "utility", "ic50_geomean")["validation"]["tests"]
    assert tests["passed"] == 2 and tests["failed"] == [] and tests["sandbox"]
    check = ledger.read_record(project, "descriptor", "lab_assays")["validation"]["check"]
    assert check["lab_assays.assays"]["status"] == "ready"
    assert (project.data_dir / "lab_assays" / "assays.csv").read_text() == ASSAYS
    manifest = json.loads((session.run.dir / "MANIFEST.json").read_text())
    receipts = {e["kind"]: rel for rel, e in manifest["artifacts"].items() if e.get("registered")
                and str(e.get("kind", "")).startswith("project_")}
    assert set(receipts) == {"project_descriptor", "project_utility"}
    audit = (session.run.dir / "audit.html").read_text()
    assert "project_registrations/utility-ic50_geomean-v1.json" in audit
    trace = [json.loads(line) for line in (session.run.dir / "logs" / "trace.jsonl").read_text().splitlines()]
    assert {e.get("name") for e in trace if e.get("type") == "project_registration"} >= {"lab_assays", "cell_lines",
                                                                                         "ic50_geomean"}
    assert any(e.get("type") == "project_utility_call" for e in trace)


async def test_a_new_session_of_the_project_starts_with_its_utilities_and_prompt(pconfig, project, rt):
    await _call(rt, "RegisterUtility", name="ic50_geomean", description="Geometric mean of positive IC50 values.",
                input_schema=SCHEMA, directory=str(_utility_dir(rt)), why="repeated summaries")
    later = _runtime(pconfig, project)
    try:
        assert "util__ic50_geomean" in later.registry.names()
        g = later.agents["genomics-analyst"]
        assert any(t.name == "util__ic50_geomean" for t in later.tools_for(g))
        assert not any(t.name == "util__ic50_geomean" for t in later.tools_for(later.agents["scientific-reviewer"]))
        _, volatile = later._system_for(g, later.workspace_for(g.name))  # noqa: SLF001
        assert "`util__ic50_geomean`" in volatile.text
        out = await later.registry.get("util__ic50_geomean")(_ctx(later, "genomics-analyst"), {"values": [1, 100]})
        assert out == {"n": 2, "geomean": pytest.approx(10.0)}
    finally:
        later.run.close()


# ---------------------------------------------------------------------------- data specs


@needs_arrow
async def test_a_lint_error_refuses_the_descriptor_and_writes_nothing(rt, project):
    _write(rt, "assays.csv", ASSAYS)
    bad = DESCRIPTOR.replace("key: {columns: [assay_id], check: full}", "key: {columns: [no_such_column]}")
    _write(rt, "bad.yaml", bad)
    msg = await _refused(rt, "RegisterDataSpec", kind="descriptor", path="bad.yaml", files=["assays.csv"],
                         why="test")
    assert "lint error" in msg and "no_such_column" in msg
    assert not (project.descriptors_dir / "lab_assays.yaml").exists()
    assert not (project.data_dir / "lab_assays").exists()
    assert ledger.read_record(project, "descriptor", "lab_assays") is None
    assert _events(project, "descriptor")[-1]["event"] == "refused"
    assert not list(project.staging_dir.glob("*"))


@needs_arrow
async def test_a_failing_check_refuses_the_descriptor(rt, project):
    _write(rt, "assays.csv", ASSAYS)
    _write(rt, "wrong.yaml", DESCRIPTOR.replace("'^A\\d+$'", "'^ASSAY-\\d+$'"))     # the keys do not match
    msg = await _refused(rt, "RegisterDataSpec", kind="descriptor", path="wrong.yaml", files=["assays.csv"],
                         why="test")
    assert "check lab_assays.assays" in msg and "R4b" in msg
    assert ledger.read_record(project, "descriptor", "lab_assays") is None
    _write(rt, "missing.yaml", DESCRIPTOR)                     # no files imported: the table has no data
    msg = await _refused(rt, "RegisterDataSpec", kind="descriptor", path="missing.yaml", why="test")
    assert "check lab_assays.assays" in msg


async def test_shipped_names_and_bad_specs_are_refused(rt, project):
    msg = await _refused(rt, "RegisterDataSpec", kind="descriptor",
                         content=DESCRIPTOR.replace("source: lab_assays", "source: depmap"), why="test")
    assert "shipped" in msg
    msg = await _refused(rt, "RegisterDataSpec", kind="overlay",
                         content="schema: vbt.overlay/1\nserver: target\ntools: {}\n", why="test")
    assert "shipped overlay" in msg
    msg = await _refused(rt, "RegisterDataSpec", kind="descriptor",
                         content=DESCRIPTOR.replace("source: lab_assays", "source: ../../escape"), why="test")
    assert "lowercase identifier" in msg
    msg = await _refused(rt, "RegisterDataSpec", kind="descriptor",
                         content="schema: vbt.datasource/1\nsource: half\ntitle: x\n", why="test")
    assert "half.yaml" in msg and "Field required" in msg
    assert "why is required" in await _refused(rt, "RegisterDataSpec", kind="descriptor", content=DESCRIPTOR,
                                               why="")
    assert not list(project.descriptors_dir.glob("*"))


@needs_arrow
async def test_acquisition_specs_merge_into_the_project_descriptor(rt, project):
    _write(rt, "assays.csv", ASSAYS)
    _write(rt, "d.yaml", DESCRIPTOR)
    await _call(rt, "RegisterDataSpec", kind="descriptor", path="d.yaml", files=["assays.csv"], why="test")
    acq = ("acquisition:\n  release: v1\n  transport: {plugin: http, options: {base: 'https://example.org/assays/', "
           "files: {assays.csv: {bytes: 139}}}}\n  dir: lab_assays/{release}\n"
           "  tables:\n    assays: {files: [assays.csv], count: 1, bytes: 139}\n")
    out = await _call(rt, "RegisterDataSpec", kind="acquisition", source="lab_assays", content=acq,
                      why="fetch the next export from the lab server")
    assert out["status"] == "registered" and out["record"]["version"] == 2
    rec = ledger.read_record(project, "descriptor", "lab_assays")
    assert rec["change"] == "acquisition" and rec["previous"]["version"] == 1
    assert "data/lab_assays/assays.csv" in rec["files"]               # the imported data stays recorded
    text = (project.descriptors_dir / "lab_assays.yaml").read_text()
    assert "acquisition:" in text and "In-house drug sensitivity assays" in text
    msg = await _refused(rt, "RegisterDataSpec", kind="acquisition", source="nope", content=acq, why="x")
    assert "no descriptor 'nope'" in msg


@needs_arrow
async def test_a_descriptor_whose_files_are_to_be_acquired_registers_with_a_note(rt, project):
    remote = DESCRIPTOR.replace("lab_assays", "remote_assays") + (
        "acquisition:\n  release: v1\n  transport: {plugin: http, options: {base: 'https://example.org/a/', "
        "files: {assays.csv: {bytes: 139}}}}\n  tables:\n    assays: {files: [assays.csv]}\n")
    out = await _call(rt, "RegisterDataSpec", kind="descriptor", content=remote, why="acquired later")
    assert out["status"] == "registered"
    assert "vbt data acquire --source remote_assays" in out["check"]["remote_assays.assays"]["note"]


@needs_arrow
async def test_inspect_dataset_drafts_a_descriptor_that_registers(rt, project, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = rt.workspace_for(ENGINEER) / "scores.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"sample_id": ["S1", "S2", "S3", "S4"], "score": [0.1, 0.5, 0.9, None],
                             "batch": ["b1", "b1", "b2", "b2"]}), path)
    info = await _call(rt, "InspectDataset", path="scores.parquet", source="screen_scores")
    cols = {c["name"]: c for c in info["columns"]}
    assert info["rows"] == 4 and cols["sample_id"]["unique"] and cols["score"]["nulls"] == 1
    assert "local_key" in info["draft_descriptor"] and "screen_scores" in info["draft_descriptor"]
    out = await _call(rt, "RegisterDataSpec", kind="descriptor", content=info["draft_descriptor"],
                      files=["scores.parquet"], why="the draft as is")
    assert out["status"] == "registered" and out["check"]["screen_scores.rows"]["status"] == "ready"


@needs_arrow
@pytest.mark.parametrize("case", ["late_text_type", "dotted_column", "nullable_composite_key", "names_with_spaces"])
async def test_drafts_of_real_shapes_register_as_drafted(rt, project, case):
    """Shapes the real files showed (2026-10-09): the HGNC complete set is a tab-separated ``.txt`` whose column
    ``pseudogene.org`` holds a dot and whose sparse columns are empty or numeric in the first MiB the CSV plugin
    infers types from; the Tahoe-100M cell-line driver table has no null-free key (its rows are unique only with a
    nullable column); Tahoe drug names hold spaces and parentheses."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    ws = rt.workspace_for(ENGINEER)
    ws.mkdir(parents=True, exist_ok=True)
    if case == "late_text_type":                               # int in the first MiB, text afterwards
        rows = [f"HGNC:{i}\tG{i}\t{i}" for i in range(1, 90_000)] + ["HGNC:90000\tG90000\tn/a"]
        path = ws / "genes.txt"
        path.write_text("hgnc_id\tsymbol\tlate\n" + "\n".join(rows) + "\n")
    elif case == "dotted_column":
        path = ws / "dotted.tsv"
        path.write_text("hgnc_id\tsymbol\tpseudogene.org\nHGNC:1\tA1BG\t\nHGNC:2\tA2M\tPGOHUM1\n")
    elif case == "nullable_composite_key":
        path = ws / "drivers.parquet"
        pq.write_table(pa.table({"cell_name": ["A549", "A549", "A549", "A549", "PC9", "PC9"],
                                 "Driver_Gene_Symbol": ["KRAS", "KRAS", "STK11", "NRAS", "EGFR", "KRAS"],
                                 "Driver_ProtEffect": ["p.G12S", None, "p.Q37*", None, "p.E746_A750del", "p.G12S"],
                                 "Organ": ["Lung"] * 6}), path)
    else:
        path = ws / "drugs.parquet"
        pq.write_table(pa.table({"drug": ["Almonertinib (mesylate)", "18β-Glycyrrhetinic acid", "Erlotinib"],
                                 "targets": ["EGFR", None, "EGFR"]}), path)
    info = await _call(rt, "InspectDataset", path=path.name, source=f"real_{case}")
    if case == "late_text_type":
        assert info["format"] == "tsv" and info["column_types"] == {"late": "string"}
        assert "column_types" in info["draft_descriptor"]
        import yaml

        without = yaml.safe_load(info["draft_descriptor"])
        without["defaults"]["format"] = "tsv"                  # the plugin's own inference: int64 from the first MiB
        msg = await _refused(rt, "RegisterDataSpec", kind="descriptor", content=yaml.safe_dump(without),
                             files=[path.name], why="without the override")
        assert "check real_late_text_type.rows" in msg
    elif case == "dotted_column":
        assert "pseudogene.org" not in info["draft_descriptor"] and any("pseudogene.org" in n for n in info["notes"])
    elif case == "nullable_composite_key":
        assert info["key"] == ["cell_name", "Driver_Gene_Symbol", "Driver_ProtEffect"]
        assert info["key_nullable"] == ["Driver_ProtEffect"]
    else:
        assert info["key"] == ["drug"] and info["key_pattern"] == r"^\S(?:.*\S)?$"
    out = await _call(rt, "RegisterDataSpec", kind="descriptor", content=info["draft_descriptor"],
                      files=[path.name], why="the draft as is")
    assert out["status"] == "registered", out


@needs_arrow
def test_inspect_dataset_is_bounded_in_rows_and_bytes(tmp_path):
    from vbt.tools.utilities import inspect_dataset

    path = tmp_path / "big.csv"
    path.write_text("id,value\n" + "".join(f"R{i},{i}\n" for i in range(200_000)))
    assert inspect_dataset(path)["complete"] is True
    cut = inspect_dataset(path, max_bytes=64 << 10)
    assert cut["complete"] is False and cut["rows_profiled"] < 200_000
    assert inspect_dataset(path, max_rows=1000)["complete"] is False


async def test_imports_respect_the_read_policy(rt, project, tmp_path):
    outside = tmp_path / "outside.csv"
    outside.write_text(ASSAYS)
    msg = await _refused(rt, "RegisterDataSpec", kind="descriptor", content=DESCRIPTOR, files=[str(outside)],
                         why="x")
    assert "outside permitted roots" in msg


# ---------------------------------------------------------------------------- utilities


async def test_a_utility_whose_tests_fail_is_refused_and_never_becomes_a_tool(rt, project):
    failing = TESTS.replace("- 10.0) < 1e-9", "- 11.0) < 1e-9")
    msg = await _refused(rt, "RegisterUtility", name="ic50_geomean", description="Geometric mean of IC50 values.",
                         input_schema=SCHEMA, directory=str(_utility_dir(rt, tests=failing)), why="x")
    assert "tests failed" in msg and "test_known_answer" in msg
    assert "util__ic50_geomean" not in rt.registry.names()
    assert not (project.utilities_dir / "ic50_geomean").exists()
    assert ledger.read_record(project, "utility", "ic50_geomean") is None
    assert _events(project, "utility")[-1]["event"] == "refused"


@pytest.mark.parametrize("code,tests,schema,expected", [
    (CODE.replace('    """Geometric mean of the positive IC50 values (nM) and how many were used."""\n', ""), TESTS,
     SCHEMA, "needs a docstring"),
    (CODE, "def helper():\n    pass\n", SCHEMA, "no test_* function"),
    (CODE, TESTS, {"type": "object", "properties": {"values": {}}}, "must be required"),
    (CODE, TESTS, {"type": "object", "properties": {"values": {}, "extra": {}}, "required": ["values"]},
     "does not take"),
    (CODE, TESTS, {"type": "array"}, "JSON schema object"),
    ("def run(values:\n", TESTS, SCHEMA, "does not parse"),
])
async def test_static_checks_refuse_before_tests_run(rt, code, tests, schema, expected):
    msg = await _refused(rt, "RegisterUtility", name="ic50_geomean", description="Geometric mean of IC50 values.",
                         input_schema=schema, code=code, tests=tests, why="x")
    assert expected in msg


async def test_a_registered_utility_is_called_in_the_sandbox_and_refused_when_changed(rt, project):
    out = await _call(rt, "RegisterUtility", name="ic50_geomean", description="Geometric mean of IC50 values (nM).",
                      input_schema=SCHEMA, directory=str(_utility_dir(rt)), why="x")
    assert out["tool"] == "util__ic50_geomean" and out["receipt"].endswith("utility-ic50_geomean-v1.json")
    result = await _call(rt, "util__ic50_geomean", agent="genomics-analyst", values=[1, 100])
    assert result == {"n": 2, "geomean": pytest.approx(10.0)}
    (project.utilities_dir / "ic50_geomean" / "utility.py").write_text(CODE.replace("math.exp", "math.log"))
    msg = await _refused(rt, "util__ic50_geomean", agent="genomics-analyst", values=[1, 100])
    assert "changed since it was registered" in msg
    assert not utility_tools(project)


async def test_a_new_version_replaces_the_utility_and_keeps_the_ledger(rt, project):
    await _call(rt, "RegisterUtility", name="ic50_geomean", description="Geometric mean of IC50 values (nM).",
                input_schema=SCHEMA, directory=str(_utility_dir(rt)), why="v1")
    v2 = CODE.replace('return {"n": len(vals)', 'return {"version": 2, "n": len(vals)')
    out = await _call(rt, "RegisterUtility", name="ic50_geomean", description="Geometric mean of IC50 values (nM).",
                      input_schema=SCHEMA, code=v2, tests=TESTS, why="v2")
    assert out["record"]["version"] == 2
    rec = ledger.read_record(project, "utility", "ic50_geomean")
    assert rec["previous"]["version"] == 1 and not ledger.verify(project, rec)
    assert (await _call(rt, "util__ic50_geomean", agent="genomics-analyst", values=[4]))["version"] == 2
    assert [e["why"] for e in _events(project, "utility") if e["event"] == "registered"] == ["v1", "v2"]


async def test_writes_outside_the_project_are_refused(rt, project, tmp_path):
    msg = await _refused(rt, "RegisterUtility", name="../evil", description="Geometric mean of IC50 values (nM).",
                         input_schema=SCHEMA, code=CODE, tests=TESTS, why="x")
    assert "lowercase identifier" in msg
    d = _utility_dir(rt)
    (d / "fixtures").mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("not yours")
    (d / "fixtures" / "link.txt").symlink_to(secret)
    msg = await _refused(rt, "RegisterUtility", name="ic50_geomean", description="Geometric mean of IC50 values.",
                         input_schema=SCHEMA, directory=str(d), why="x")
    assert "outside permitted roots" in msg
    (d / "fixtures" / "link.txt").unlink()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    shutil.rmtree(project.utilities_dir)
    project.utilities_dir.symlink_to(elsewhere, target_is_directory=True)   # the project dir is redirected
    msg = await _refused(rt, "RegisterUtility", name="ic50_geomean", description="Geometric mean of IC50 values.",
                         input_schema=SCHEMA, directory=str(d), why="x")
    assert "outside the project" in msg
    assert not list(elsewhere.iterdir())


@pytest.mark.skipif(not bwrap_works(), reason="bubblewrap is not available here")
async def test_utility_code_cannot_write_the_project_or_reach_the_network_under_bwrap(rt, project, pconfig):
    rt.config["projects"]["sandbox"] = "bwrap"
    sneaky = (f'from pathlib import Path\n\n\ndef test_writes_the_project():\n'
              f'    Path({str(project.root / "descriptors" / "x.yaml")!r}).write_text("x")\n')
    msg = await _refused(rt, "RegisterUtility", name="ic50_geomean", description="Geometric mean of IC50 values.",
                         input_schema=SCHEMA, code=CODE, tests=sneaky, why="x")
    assert "tests failed" in msg and ("Read-only file system" in msg or "Permission denied" in msg)
    assert not (project.descriptors_dir / "x.yaml").exists()
    net = ('import socket\n\n\ndef test_reaches_out():\n'
           '    socket.create_connection(("1.1.1.1", 53), timeout=3)\n')
    msg = await _refused(rt, "RegisterUtility", name="ic50_geomean", description="Geometric mean of IC50 values.",
                         input_schema=SCHEMA, code=CODE, tests=net, why="x")
    assert "tests failed" in msg and "test_reaches_out" in msg


def test_sandbox_plan_names_what_applies(pconfig):
    label, _exe, _notes = sandbox_plan(pconfig, network=False)
    assert label.startswith(("bwrap", "rlimit"))
    pconfig["projects"]["sandbox"] = "none"
    assert sandbox_plan(pconfig, network=True)[0] == "rlimit"


async def test_without_a_project_the_tools_refuse(config):
    run = Run(Path(config["paths"]["runs_dir"]), config=config)
    runtime = Runtime(config, run, provider=ScriptedProvider.from_rules({}))
    try:
        for tool, args in (("ProjectInfo", {}), ("RegisterUtility", {"name": "x", "why": "y"}),
                           ("RegisterDataSpec", {"kind": "descriptor", "content": DESCRIPTOR, "why": "y"})):
            with pytest.raises(ToolFailure, match="no project is active"):
                await runtime.registry.get(tool)(ToolContext(ENGINEER, run, runtime, tool_call_id="t"), args)
        assert ENGINEER not in runtime.agents
    finally:
        run.close()


# ---------------------------------------------------------------------------- plugins

SCORE_0_10 = '''"""score_0_10: a bounded 0-10 score."""
from typing import ClassVar

from vbt.datalayer.plugins.registry import register
from vbt.datalayer.plugins.statistics.score import Score01Statistic


@register
class Score010Statistic(Score01Statistic):
    name: ClassVar[str] = "score_0_10"
    version: ClassVar[str] = "1.0"
    default_scale: ClassVar[tuple] = (0.0, 10.0)
'''

BROKEN = SCORE_0_10.replace('"score_0_10"', '"score_broken"') + '''
    def sort_key(self, value, spec=None, **kw):
        raise RuntimeError("broken ordering")

    def predicate(self, *a, **kw):
        raise RuntimeError("broken predicate")
'''


async def test_a_plugin_registers_only_after_its_conformance_suite_passes(rt, project, pconfig):
    out = await _call(rt, "RegisterPlugin", kind="statistic", content=SCORE_0_10, why="an in-house 0-10 score")
    assert out["status"] == "registered"
    rec = ledger.read_record(project, "plugin", "score_0_10")
    assert rec["plugin_kind"] == "statistic" and rec["validation"]["conformance"]["exit_code"] == 0
    assert (project.plugins_dir / "statistic" / "score_0_10.py").is_file()
    stamp = Path(pconfig["data"]["cache_dir"]) / "conformance" / "statistic.score_0_10.json"
    assert json.loads(stamp.read_text())["plugin"] == "statistic/score_0_10"
    assert "plugins/statistic/score_0_10.py" in project.profile_path.read_text()
    assert str(project.plugins_dir / "statistic" / "score_0_10.py") in rt.config["data"]["plugins"]["paths"]

    msg = await _refused(rt, "RegisterPlugin", kind="statistic", content=BROKEN, why="x")
    assert "conformance suite failed" in msg
    assert not (project.plugins_dir / "statistic" / "score_broken.py").exists()
    msg = await _refused(rt, "RegisterPlugin", kind="statistic",
                         content=SCORE_0_10.replace('"score_0_10"', '"score_0_1"'), why="x")
    assert "already exists" in msg
    msg = await _refused(rt, "RegisterPlugin", kind="quantum", content=SCORE_0_10, why="x")
    assert "existing kinds only" in msg
    msg = await _refused(rt, "RegisterPlugin", kind="statistic", content="x = 1\n", why="x")
    assert "exactly one @register class" in msg


# ---------------------------------------------------------------------------- review


async def test_the_reviewer_must_approve_when_review_is_required(pconfig, project):
    pconfig["projects"]["review"] = "reviewer"
    runtime = _runtime(pconfig, project, {"scientific-reviewer": [
        reply("The tests check known answers.\nVERDICT: APPROVE"),
        reply("The test only checks an empty input.\nVERDICT: REJECT")]})
    try:
        out = await runtime.registry.get("RegisterUtility")(_ctx(runtime), dict(
            name="ic50_geomean", description="Geometric mean of IC50 values (nM).", input_schema=SCHEMA,
            directory=str(_utility_dir(runtime)), why="x"))
        assert out["status"] == "registered"
        review = ledger.read_record(project, "utility", "ic50_geomean")["review"]
        assert review["verdict"] == "approve" and review["by"] == "scientific-reviewer" and review["invocation_id"]
        with pytest.raises(ToolFailure, match="rejected by scientific-reviewer"):
            await runtime.registry.get("RegisterUtility")(_ctx(runtime), dict(
                name="other_helper", description="Geometric mean of IC50 values (nM).", input_schema=SCHEMA,
                code=CODE, tests=TESTS, why="x"))
        assert ledger.read_record(project, "utility", "other_helper") is None
        assert any(e["agent"] == "scientific-reviewer" for e in runtime.delegation_log)
    finally:
        runtime.run.close()


async def test_human_review_keeps_the_item_pending_until_approved(pconfig, project):
    pconfig["projects"]["review"] = "human"
    runtime = _runtime(pconfig, project)
    try:
        out = await runtime.registry.get("RegisterUtility")(_ctx(runtime), dict(
            name="ic50_geomean", description="Geometric mean of IC50 values (nM).", input_schema=SCHEMA,
            directory=str(_utility_dir(runtime)), why="x"))
        assert out["status"] == "pending_review"
        assert "util__ic50_geomean" not in runtime.registry.names()
        assert ledger.read_record(project, "utility", "ic50_geomean") is None
        assert [p["name"] for p in pending_items(project)] == ["ic50_geomean"]
        info = await runtime.registry.get("ProjectInfo")(_ctx(runtime), {})
        assert info["pending"][0]["name"] == "ic50_geomean" and info["review"] == "human"
        rec = approve_pending(project, "utility", "ic50_geomean", by="owner", config=pconfig)
        assert rec["status"] == "registered" and rec["review"]["by"] == "owner"
        assert [t.name for t in utility_tools(project)] == ["util__ic50_geomean"]
        assert not pending_items(project)
        with pytest.raises(ProjectError, match="no pending"):
            approve_pending(project, "utility", "ic50_geomean", by="owner")
    finally:
        runtime.run.close()


# ---------------------------------------------------------------------------- opt-in: real files, live sources

REAL = os.environ.get("VBT_DL_REAL_DATA", "").strip()
NETWORK = os.environ.get("VBT_DL_NETWORK", "").strip() == "1"


@pytest.mark.skipif(not (REAL and HAVE_ARROW), reason="set VBT_DL_REAL_DATA=<data/real> to run on the real files")
@pytest.mark.parametrize("rel,source,rows,key", [
    ("depmap/24Q4/Model.csv", "depmap_models", 2105, ["ModelID"]),
    ("tahoe/2dc57900b7981cfcf5e211527169a0b006546a95/metadata/drug_metadata.parquet", "tahoe_drugs", 379, ["drug"]),
    ("tahoe/2dc57900b7981cfcf5e211527169a0b006546a95/metadata/cell_line_metadata.parquet", "tahoe_cell_lines",
     1000, ["cell_name", "Driver_Gene_Symbol", "Driver_ProtEffect_or_CdnaEffect"]),
])
async def test_real_files_register_from_their_drafts(rt, project, rel, source, rows, key):
    path = Path(REAL) / rel
    if not path.is_file():
        pytest.skip(f"{path} is not downloaded")
    rt.read_roots.append(Path(REAL).resolve())
    info = await _call(rt, "InspectDataset", path=str(path), source=source)
    assert info["rows"] == rows and info["key"] == key
    out = await _call(rt, "RegisterDataSpec", kind="descriptor", content=info["draft_descriptor"],
                      files=[str(path)], why="real-data check")
    assert out["status"] == "registered" and list(out["check"].values())[0]["status"] == "ready"


HGNC_URL = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"


@pytest.mark.skipif(not (NETWORK and HAVE_ARROW), reason="set VBT_DL_NETWORK=1 to fetch the HGNC complete set")
async def test_a_live_dataset_registers_from_its_draft(rt, project):
    import urllib.request

    dest = rt.workspace_for(ENGINEER) / "hgnc_complete_set.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(HGNC_URL, timeout=120) as r:
        dest.write_bytes(r.read())
    info = await _call(rt, "InspectDataset", path=dest.name, source="hgnc_genes", table="genes")
    assert info["format"] == "tsv" and info["key"] == ["hgnc_id"] and info["rows"] > 40_000
    out = await _call(rt, "RegisterDataSpec", kind="descriptor", content=info["draft_descriptor"],
                      files=[dest.name], why="live check")
    assert out["status"] == "registered" and out["check"]["hgnc_genes.genes"]["status"] == "ready"
