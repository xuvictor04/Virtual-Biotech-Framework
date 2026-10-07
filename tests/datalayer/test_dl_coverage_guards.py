"""Coverage and drift guards (§18 F24, §21, §22 steps 5-6).

* Every tool upstream registers (AST of its ``server.py``, never imported) and the two PubMed tools have a
  reviewed binding, and every upstream parameter is bound by an overlay argument.
* The upstream input schemas equal the reviewed snapshot ``upstream_tool_schemas.json`` (written by
  ``vbt ds explain --all --json --schemas``): a new tool, a removed tool, a new or retyped parameter or a
  changed default fails CI until the overlay is reviewed and the snapshot regenerated
  (``VBT_UPDATE_GOLDEN=1`` rewrites it).
* Every column of every strict descriptor has a role (no ``strict_roles`` finding), and the snapshot's
  hashes are its schemas'.
* The other CI guards exist and are not skipped unconditionally: defect detectors, architecture,
  determinism across hash seeds, documentation examples, fixture columns roled, upstream submodule clean.
* ``vbt ds diff-release`` reports role columns, Arrow types, vocabularies and matrix axes between two
  data roots, and nothing for identical ones; ``vbt ds graduate`` reports its checklist.
* ``docs/DATA_LAYER_RUNBOOK.md`` names only commands the CLI has.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from dl_upstream import REPO, needs_arrow, upstream_root
from vbt.datalayer.cli import (
    COMMANDS,
    add_datasource_parsers,
    explain_json,
    graduation_checklist,
    schema_sha256,
    upstream_tool_schemas,
)

HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE / "upstream_tool_schemas.json"
RUNBOOK = REPO / "docs" / "DATA_LAYER_RUNBOOK.md"
UPDATE = os.environ.get("VBT_UPDATE_GOLDEN") == "1"
UPSTREAM_SCHEMAS = upstream_tool_schemas(upstream_root(), REPO) \
    if (upstream_root() / "src" / "mcp_servers").is_dir() else {}
needs_upstream = pytest.mark.skipif(not UPSTREAM_SCHEMAS,
                                    reason="the upstream checkout is missing (git submodule update --init)")


@pytest.fixture(scope="module")
def shipped() -> tuple[Any, Any]:
    from vbt.datalayer.catalog import Catalog
    from vbt.datalayer.descriptor.load import load_descriptors, load_overlays
    from vbt.datalayer.plugins.registry import discover

    registry = discover(entry_points=False)
    variables = {"project_root": str(REPO), "upstream_commit": ""}
    descriptors = load_descriptors(REPO / "configs" / "data" / "sources", variables)
    overlays, generic = load_overlays(REPO / "configs" / "data" / "overlays", variables)
    return Catalog(descriptors, overlays, generic, registry=registry), registry


def _snapshot() -> dict[str, Any]:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------- bindings and the schema snapshot


@needs_upstream
def test_every_registered_tool_has_a_reviewed_binding_and_every_parameter_is_bound(shipped) -> None:
    catalog, _registry = shipped
    body = explain_json({}, catalog, upstream=upstream_root(), project_root=REPO)
    upstream = {name: e for name, e in body["tools"].items() if "input_schema" in e}
    assert len(upstream) == len(UPSTREAM_SCHEMAS) == 103
    unbound = sorted(n for n, e in upstream.items() if not e["bound"] or e.get("status") != "reviewed")
    assert not unbound, f"registered tools without a reviewed binding (the generic guard applies): {unbound}"
    params = {n: e["unbound_params"] for n, e in upstream.items() if e.get("unbound_params")}
    assert not params, f"upstream parameters no overlay argument binds: {params}"


@needs_upstream
def test_upstream_input_schemas_match_the_reviewed_snapshot(shipped) -> None:
    catalog, _registry = shipped
    current = explain_json({}, catalog, upstream=upstream_root(), project_root=REPO, schemas_only=True)
    if UPDATE:
        SNAPSHOT.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    recorded = _snapshot()["tools"]
    now = current["tools"]
    added = sorted(set(now) - set(recorded))
    removed = sorted(set(recorded) - set(now))
    changed = {}
    for name in sorted(set(now) & set(recorded)):
        if now[name]["input_sha256"] != recorded[name]["input_sha256"]:
            a = recorded[name]["input_schema"].get("properties") or {}
            b = now[name]["input_schema"].get("properties") or {}
            changed[name] = {"added": sorted(set(b) - set(a)), "removed": sorted(set(a) - set(b)),
                             "changed": sorted(k for k in set(a) & set(b) if a[k] != b[k])}
    assert not (added or removed or changed), (
        "upstream tool schemas drifted from tests/datalayer/upstream_tool_schemas.json: "
        f"new tools {added}, removed tools {removed}, changed {changed}. Review the overlays "
        "(configs/data/overlays), then regenerate with `vbt ds explain --all --json --schemas > "
        "tests/datalayer/upstream_tool_schemas.json` (or VBT_UPDATE_GOLDEN=1).")


def test_the_snapshot_hashes_its_schemas_and_covers_the_bound_tools(shipped) -> None:
    catalog, _registry = shipped
    snap = _snapshot()
    assert snap["schema"] == "vbt.upstream_tool_schemas/1" and re.fullmatch(r"[0-9a-f]{40}", snap["upstream_commit"])
    for name, entry in snap["tools"].items():
        assert entry["input_sha256"] == schema_sha256(entry["input_schema"]), name
    bridged = {f"{s}.{t}" for s, ov in catalog.overlays.items() if s != "data" for t in ov.tools}
    bridged |= {a for s, ov in catalog.overlays.items() if s != "data" for b in ov.tools.values() for a in b.same_as}
    assert bridged == set(snap["tools"]), sorted(bridged ^ set(snap["tools"]))
    commits = {getattr(ov, "upstream_commit", None) for s, ov in catalog.overlays.items() if s != "data"} - {None, ""}
    assert commits <= {snap["upstream_commit"]}, "the overlays were reviewed against another upstream commit"


def test_schema_parsing_matches_the_signature(tmp_path: Path) -> None:
    server = tmp_path / "src" / "mcp_servers" / "toy_mcp"
    server.mkdir(parents=True)
    (server / "server.py").write_text("from src.mcp_servers.toy_mcp.tools import find_things\n"
                                      "register_tool(mcp, find_things)\n")
    (server / "tools.py").write_text(
        "from typing import Optional, List\n"
        "def find_things(target_id: str, limit: int = 20, tags: Optional[List[str]] = None, x: float | None = 0.5):\n"
        "    '''doc'''\n")
    got = upstream_tool_schemas(tmp_path, tmp_path)
    assert got == {"toy.find_things": {"type": "object", "required": ["target_id"], "properties": {
        "target_id": {"type": "string"}, "limit": {"type": "integer", "default": 20},
        "tags": {"type": "array", "items": {"type": "string"}, "default": None},
        "x": {"type": "number", "default": 0.5}}}}


# ---------------------------------------------------------------------------- roles


def test_every_column_of_every_strict_descriptor_has_a_role(shipped) -> None:
    from vbt.datalayer.descriptor.lint import lint_descriptor

    catalog, registry = shipped
    strict = [d for d in catalog.sources.values() if d.strict]
    assert strict, "no strict descriptor is shipped"
    for desc in strict:
        findings = [f for f in lint_descriptor(desc, registry, strict=True, loaded_sources=catalog.sources)
                    if f.rule == "strict_roles"]
        assert not findings, "\n".join(map(str, findings))

        def walk(cols: Any, where: str) -> None:
            for name, col in (cols or {}).items():
                assert getattr(col, "role", None), f"{where}.{name} has no role"
                walk(getattr(col, "fields", None), f"{where}.{name}")

        for name, spec in desc.tables.items():
            walk(spec.columns, f"{desc.source}.{name}")


# ---------------------------------------------------------------------------- the other guards exist


#: guard -> (test file, test functions that implement it)
GUARDS = {
    "defect detectors reproduce": ("test_dl_defect_detectors.py", ("test_ot_drug_003", "test_ct_gov_001")),
    "every defect has its detector": ("test_dl_shipped_configs.py", ("test_defect_detectors_exist",)),
    "architecture: no pyarrow in the harness": ("test_dl_architecture.py",
                                                ("test_harness_modules_import_without_pyarrow_or_pandas",
                                                 "test_core_modules_name_no_plugin_or_table")),
    "determinism across hash seeds": ("test_dl_determinism.py",
                                      ("test_serve_is_independent_of_the_hash_seed",
                                       "test_gateway_results_are_independent_of_the_hash_seed")),
    "documentation examples lint": ("test_dl_doc_examples.py", ("test_descriptor_examples_validate_and_lint_clean",
                                                                "test_overlay_examples_validate_and_lint_clean")),
    "fixture columns roled": ("test_dl_strict_descriptors.py",
                              ("test_every_fixture_column_and_nested_field_is_roled",
                               "test_lint_strict_is_clean_and_phase2_descriptors_are_strict")),
    "submodule clean": ("test_dl_upstream_untouched.py", ("test_upstream_worktree_is_clean",
                                                          "test_upstream_head_is_the_recorded_commit")),
    "every tool bound": ("test_dl_shipped_configs.py", ("test_every_bridged_tool_has_a_reviewed_binding",)),
}


def _unconditional_skip(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for d in fn.decorator_list:
        text = ast.unparse(d)
        if text in ("pytest.mark.skip", "pytest.skip") or text.startswith("pytest.mark.skip("):
            return True
    return False


@pytest.mark.parametrize("guard", sorted(GUARDS))
def test_ci_guard_exists_and_is_not_skipped(guard: str) -> None:
    file, names = GUARDS[guard]
    tree = ast.parse((HERE / file).read_text(encoding="utf-8"))
    fns = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for name in names:
        assert name in fns, f"{guard}: {file}::{name} is missing"
        assert not _unconditional_skip(fns[name]), f"{guard}: {file}::{name} is skipped unconditionally"


# ---------------------------------------------------------------------------- diff-release


def _ot_ctx(root: Path, cache: Path) -> Any:
    from vbt.datalayer.service import ServiceContext
    from vbt.datalayer.settings import DataSettings

    settings = DataSettings.from_dict({"descriptors_dir": str(REPO / "configs" / "data" / "sources"),
                                       "overlays_dir": str(REPO / "configs" / "data" / "overlays"),
                                       "cache_dir": str(cache)}, project_root=REPO)
    return ServiceContext(settings)


@pytest.fixture
def releases(ot_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Two releases side by side: 25.09 (the fixture) and 25.12 (a copy the test edits)."""
    import dl_fixtures as F

    old = F.copy_fixture(ot_root, tmp_path / "ot" / "25.09")
    new = F.copy_fixture(ot_root, tmp_path / "ot" / "25.12")
    monkeypatch.setenv("OPEN_TARGETS_DATA_PATH", str(old))
    return old, new, _ot_ctx(old, tmp_path / "cache")


@needs_arrow
def test_diff_release_of_identical_releases_is_empty(releases) -> None:
    from vbt.datalayer.diff_release import diff_release

    _old, _new, ctx = releases
    report = diff_release(ctx, {"from": "25.09", "to": "25.12", "tables": ["open_targets.known_drug",
                                                                            "open_targets.target"]})
    tables = report["sources"]["open_targets"]["tables"]
    assert set(tables) == {"open_targets.known_drug", "open_targets.target"}
    assert all(t["status"] == "same" for t in tables.values()), tables
    assert report["summary"] == {"changes": {}, "breaking_tables": []}


@needs_arrow
def test_diff_release_reports_role_columns_types_and_vocabularies(releases) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    import dl_fixtures as F
    from vbt.datalayer.diff_release import diff_release, format_report

    _old, new, ctx = releases
    for path in sorted((new / "known_drug").glob("*.parquet")):
        tbl = pq.read_table(path)
        rows = tbl.to_pylist()
        for r in rows:
            r.pop("drugType")
            r["status"] = "Withdrawn" if r["status"] == "Completed" else r["status"]
            r["newColumn"] = 1
        schema = tbl.schema.remove(tbl.schema.get_field_index("drugType")).append(pa.field("newColumn", pa.int64()))
        schema = schema.set(schema.get_field_index("phase"), pa.field("phase", pa.float32()))
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    shutil.rmtree(new / "mouse_phenotype")
    F.write_manifest(new)
    report = diff_release(ctx, {"from": "25.09", "to": str(new), "sources": ["open_targets"],
                                "tables": ["open_targets.known_drug", "open_targets.mouse_phenotype",
                                           "open_targets.target"]})
    tables = report["sources"]["open_targets"]["tables"]
    kd = {c["kind"]: c for c in tables["open_targets.known_drug"]["changes"]}
    assert kd["column_removed"]["column"] == "drugType" and kd["column_removed"]["role"] == "category"
    assert kd["column_added"]["column"] == "newColumn"
    assert (kd["type_changed"]["column"], kd["type_changed"]["before"], kd["type_changed"]["after"]) == \
        ("phase", "double", "float")
    assert kd["vocab_changed"]["column"] == "status"
    assert kd["vocab_changed"]["added"] == ["Withdrawn"] and kd["vocab_changed"]["removed"] == ["Completed"]
    assert [c["kind"] for c in tables["open_targets.mouse_phenotype"]["changes"]] == ["table_removed"]
    assert tables["open_targets.target"]["status"] == "same"
    assert report["summary"]["breaking_tables"] == ["open_targets.known_drug", "open_targets.mouse_phenotype"]
    text = "\n".join(format_report(report))
    assert "known_drug (breaking)" in text and "drugType" in text


@needs_arrow
def test_diff_release_reports_matrix_axis_membership(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import yaml

    import test_dl_native_tools as N
    from vbt.datalayer.diff_release import diff_release

    for label in ("24Q4", "25Q2"):
        (tmp_path / "depmap" / label).mkdir(parents=True)
        N._write_depmap(tmp_path / "depmap" / label)
    new = tmp_path / "depmap" / "25Q2" / "CRISPRGeneEffect.csv"
    lines = new.read_text().splitlines()
    lines[0] += ",BRCA1 (672)"
    lines[1:] = [ln + ",-0.5" for ln in lines[1:-1]]                     # one model fewer, one gene more
    new.write_text("\n".join(lines) + "\n")
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "depmap.yaml").write_text(yaml.safe_dump(N._depmap_descriptor(), sort_keys=False))
    monkeypatch.setenv("DEPMAP_DATA_PATH", str(tmp_path / "depmap" / "24Q4"))
    ctx = N._ctx(tmp_path, tmp_path / "sources")
    report = diff_release(ctx, {"from": "24Q4", "to": "25Q2", "tables": ["depmap.gene_effect"]})
    changes = {c["axis"]: c for c in report["sources"]["depmap"]["tables"]["depmap.gene_effect"]["changes"]
               if c["kind"] == "axis_changed"}
    assert changes["col"]["n_added"] == 1 and changes["col"]["n_removed"] == 0
    assert changes["row"]["n_removed"] == 1 and "ACH-000004" in changes["row"]["removed"][0]


def test_side_roots_resolve_labels_as_siblings(tmp_path: Path) -> None:
    from vbt.datalayer.diff_release import side_root

    (tmp_path / "25.09").mkdir()
    assert side_root(str(tmp_path / "25.09"), "25.09", "25.12") == tmp_path / "25.12"
    assert side_root(str(tmp_path / "25.09"), "25.09", str(tmp_path / "25.09")) == tmp_path / "25.09"
    with pytest.raises(ValueError):
        side_root(str(tmp_path / "current"), "25.09", "25.12")


# ---------------------------------------------------------------------------- CLI, graduation and the runbook


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    add_datasource_parsers(sub)
    return ap


def test_phase5_commands_are_wired() -> None:
    ap = _parser()
    a = ap.parse_args(["ds", "diff-release", "--from", "25.09", "--to", "25.12", "--source", "open_targets"])
    assert (a.from_, a.to, a.source, a.ds_cmd) == ("25.09", "25.12", ["open_targets"], "diff-release")
    a = ap.parse_args(["ds", "explain", "--all", "--json", "--schemas"])
    assert a.all and a.json and a.schemas
    a = ap.parse_args(["ds", "graduate", "drug", "--run", "r1"])
    assert a.server == ["drug"] and a.run == ["r1"]
    assert {"replay", "diff-release", "graduate"} <= set(COMMANDS)


def test_graduation_checklist_with_and_without_retro_audit_evidence(shipped, tmp_path: Path) -> None:
    catalog, registry = shipped
    bare = graduation_checklist({}, ["target", "drug"], catalog=catalog, registry=registry)
    for server in ("target", "drug"):
        items = bare[server]["items"]
        assert items["overlay"]["ok"] and items["lint"]["ok"] and items["reviewed"]["ok"], items
        assert items["blocked_alternatives"]["ok"]
        assert items["observe_evidence"]["ok"] is None and bare[server]["graduated"] is False
    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    events = [{"type": "tool_start", "tool": "mcp__target__get_target_info", "tool_use_id": "t1",
               "input": {"target_id": "ENSG00000169174"}},
              {"type": "tool_end", "tool": "mcp__target__get_target_info", "tool_use_id": "t1", "is_error": False,
               "output": json.dumps({"id": "ENSG00000169174", "approvedSymbol": "PCSK9"})}]
    (run / "logs" / "trace.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    out = graduation_checklist({}, ["target", "drug"], [run], catalog=catalog, registry=registry)
    assert out["target"]["items"]["observe_evidence"]["calls"] == 1 and out["target"]["graduated"] is True
    assert out["drug"]["items"]["observe_evidence"]["ok"] is False and out["drug"]["graduated"] is False
    assert graduation_checklist({}, ["nosuch"], catalog=catalog, registry=registry)["nosuch"]["graduated"] is False


def test_runbook_commands_exist() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    ap = _parser()
    sub = next(a for a in ap._actions if isinstance(a, argparse._SubParsersAction))
    ds = next(a for a in sub.choices["ds"]._actions if isinstance(a, argparse._SubParsersAction))
    commands = set(ds.choices)
    used = re.findall(r"vbt ds ([a-z][a-z-]*)", text)
    assert used, "the runbook names no vbt ds command"
    unknown = sorted(set(used) - commands)
    assert not unknown, f"the runbook names commands the CLI does not have: {unknown}"
    for symptom in ("not_ready", "too_large", "unranked_truncation", "expansion", "oom", "tool_defect",
                    "incomplete_key", "withheld", "data_version_drift", "replay_mismatch", "slow witness",
                    "sidecar", "third-party"):
        assert symptom in text, f"the runbook has no entry for {symptom}"
    for check in ("W1", "W2", "W3", "W4", "W5", "W6"):
        assert check in text, check
