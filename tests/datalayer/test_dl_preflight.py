"""Preflight with tool-scoped readiness (§13): the data child's ``--check --json`` runs as a subprocess
on the OT/Tahoe fixtures, and one unready table, partition or file degrades only the tools that read it.

The smoke tests start the unmodified upstream ``target`` server behind the gateway (``needs_fastmcp``):
a positive sentinel control must come back populated and a negative control must be ``not_found``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dl_upstream import DataEnv, harness_config, needs_arrow, needs_fastmcp, server_specs
from vbt import preflight
from vbt.preflight import DATA_TOOLS_LABEL, DataReadinessError, degraded_servers, degraded_tools, require_ready

pytestmark = needs_arrow

PROVIDER = SimpleNamespace(name="fakeprov", check_credentials=lambda: None)
KNOWN_DRUG_READERS = {"mcp__drug__search_known_drugs"}
EVIDENCE = "evidence"


def _config(tmp_path: Path, ot: Path | None, tahoe: Path | None = None, *, servers: list[str] | None = None,
            env: DataEnv | None = None) -> dict[str, Any]:
    env = env or DataEnv(ot_root=ot, tahoe_root=tahoe)
    cfg = harness_config(gateway=True, tmp_path=tmp_path, env=env,
                         overrides={"orchestration": {"require_reference_data": True}})
    cfg["tool_env"] = {"OPEN_TARGETS_DATA_PATH": str(ot) if ot else "", "TAHOE_DATA_PATH": str(tahoe) if tahoe else ""}
    if servers is not None:
        cfg["mcp_servers"] = {"servers": server_specs(cfg, servers, env.env())}
    return cfg


def _copy(src: Path, tmp_path_factory: pytest.TempPathFactory, label: str) -> Path:
    import dl_fixtures as F
    return F.copy_fixture(src, tmp_path_factory.mktemp(label) / src.name)


def _findings(results: list[Any]) -> list[Any]:
    return [r for r in results if r.kind == "data" and r.scope and "table" in r.scope]


@pytest.fixture(scope="module")
def baseline(ot_root: Path, tahoe_root: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Readiness of the intact fixtures (every enabled server)."""
    cfg = _config(tmp_path_factory.mktemp("pf-base"), ot_root, tahoe_root)
    results = preflight.check_reference_data(cfg)
    return {"config": cfg, "results": results, "tools": degraded_tools(cfg, results),
            "response": preflight.last_data_check(cfg)}


def test_subprocess_check_gives_one_result_per_part(baseline: dict[str, Any]) -> None:
    results, cfg = baseline["results"], baseline["config"]
    response = baseline["response"]
    assert response and "open_targets.known_drug" in response["tables"], "the data child's --check did not run"
    assert all(r.scope is not None for r in results if r.kind == "data")
    for r in _findings(results):
        assert r.scope["source"] and r.scope["table"] and r.label.startswith("data: ")
        assert set(r.scope) <= {"source", "table", "column", "columns", "partition"}
    # a column-scoped finding names its column and only the tools that read it
    col = [r for r in _findings(results) if "column" in r.scope and r.tools]
    assert col, [r.line() for r in _findings(results)]
    for r in col:
        assert all(t.startswith("mcp__") for t in r.tools)
    labels = {r.label for r in results}
    assert "Open Targets reference data (OPEN_TARGETS_DATA_PATH)" in labels
    assert "Tahoe-100M data (TAHOE_DATA_PATH)" in labels and DATA_TOOLS_LABEL in labels
    assert all(r.ok for r in results if r.scope and "table" not in r.scope)    # aggregates and summary pass
    assert degraded_servers(cfg, results) == {}
    assert "mcp__drug__search_known_drugs" not in baseline["tools"]
    assert "mcp__target__get_target_info" not in baseline["tools"]


def test_deleting_known_drug_degrades_only_its_readers(baseline: dict[str, Any], ot_root: Path, tahoe_root: Path,
                                                       tmp_path_factory: pytest.TempPathFactory) -> None:
    import dl_fixtures as F

    root = _copy(ot_root, tmp_path_factory, "pf-nokd")
    F.delete_table(root, "known_drug")
    cfg = _config(tmp_path_factory.mktemp("pf-nokd-cfg"), root, tahoe_root)
    results = require_ready(cfg, provider=PROVIDER)            # does not block: other tools are ready
    tools = degraded_tools(cfg, results)
    assert set(tools) - set(baseline["tools"]) == KNOWN_DRUG_READERS, tools
    assert "known_drug" in tools["mcp__drug__search_known_drugs"]
    assert degraded_servers(cfg, results) == {}
    finding = next(r for r in _findings(results) if r.scope["table"] == "known_drug")
    assert finding.required and not finding.ok and set(finding.tools) == KNOWN_DRUG_READERS


def test_stray_part_file_marks_only_its_partition(baseline: dict[str, Any], ot_root: Path, tahoe_root: Path,
                                                  tmp_path_factory: pytest.TempPathFactory) -> None:
    root = _copy(ot_root, tmp_path_factory, "pf-part")
    part = root / EVIDENCE / "sourceId=europepmc" / "part-00009-fixture-c000.snappy.parquet.part"
    part.write_bytes(b"PAR1 partial")
    cfg = _config(tmp_path_factory.mktemp("pf-part-cfg"), root, tahoe_root)
    results = preflight.check_reference_data(cfg)
    ev = [r for r in _findings(results) if r.scope["table"] == EVIDENCE]
    assert ev, [r.line() for r in _findings(results)]
    assert all(r.scope.get("partition") for r in ev), [r.line() for r in ev]
    assert {r.scope["partition"] for r in ev} == {"sourceId=europepmc"}
    assert not any(r.required for r in ev)                 # a call can exclude the partition
    assert set(degraded_tools(cfg, results)) == set(baseline["tools"])


def test_missing_hive_partition_is_not_ready(baseline: dict[str, Any], ot_root: Path, tahoe_root: Path,
                                             tmp_path_factory: pytest.TempPathFactory) -> None:
    import shutil

    import dl_fixtures as F

    root = _copy(ot_root, tmp_path_factory, "pf-hive")
    shutil.rmtree(root / EVIDENCE / "sourceId=chembl")
    F.write_manifest(root)
    cfg = _config(tmp_path_factory.mktemp("pf-hive-cfg"), root, tahoe_root)
    results = preflight.check_reference_data(cfg)
    ev = [r for r in _findings(results) if r.scope["table"] == EVIDENCE]
    assert any(r.scope.get("partition") == "sourceId=chembl" for r in ev), [r.line() for r in ev]
    assert set(degraded_tools(cfg, results)) == set(baseline["tools"])


def _with_essentiality(root: Path) -> Path:
    """Write a small DepMap ``target_essentiality`` table (the shared fixture has only a placeholder)."""
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq

    import dl_fixtures as F

    screen = pa.struct([("depmapId", pa.string()), ("cellLineName", pa.string()), ("diseaseFromSource", pa.string()),
                        ("diseaseCellLineId", pa.string()), ("expression", pa.float64()),
                        ("geneEffect", pa.float64()), ("mutation", pa.string())])
    tissue = pa.struct([("tissueId", pa.string()), ("tissueName", pa.string()), ("screens", pa.list_(screen))])
    essentiality = pa.struct([("isEssential", pa.bool_()), ("depMapEssentiality", pa.list_(tissue))])
    schema = pa.schema([("id", pa.string()), ("geneEssentiality", pa.list_(essentiality))])
    rows = [{"id": gene, "geneEssentiality": [{"isEssential": essential, "depMapEssentiality": [
        {"tissueId": "UBERON_0002048", "tissueName": "lung", "screens": [
            {"depmapId": F.A549, "cellLineName": "A549", "diseaseFromSource": "Lung Cancer",
             "diseaseCellLineId": "CVCL_0023", "expression": 1.5, "geneEffect": effect, "mutation": None}]}]}]}
        for gene, essential, effect in ((F.PCSK9, False, -0.1), (F.TP53, True, -1.2))]
    shutil.rmtree(root / "target_essentiality", ignore_errors=True)
    (root / "target_essentiality").mkdir()
    pq.write_table(pa.Table.from_pylist(rows, schema=schema),
                   root / "target_essentiality" / "part-00000-fixture-c000.snappy.parquet")
    F.write_manifest(root)
    return root


def test_missing_tahoe_leaves_depmap_tools_ready(ot_root: Path, tahoe_root: Path,
                                                 tmp_path_factory: pytest.TempPathFactory) -> None:
    import dl_fixtures as F

    ot = _with_essentiality(_copy(ot_root, tmp_path_factory, "pf-depmap"))
    before_cfg = _config(tmp_path_factory.mktemp("pf-depmap-cfg"), ot, tahoe_root)
    before = degraded_tools(before_cfg, preflight.check_reference_data(before_cfg))
    _settings, catalog, _registry = preflight.data_catalog(before_cfg)
    depmap = {f"mcp__functional_genomics__{t}" for t in catalog.tools("functional_genomics")
              if catalog.contract("functional_genomics", t).binding.serve != "block"
              and all(r.startswith("open_targets.") for r in catalog.contract("functional_genomics", t).tables)}
    assert depmap and not depmap & set(before), before          # DepMap tools ready on the full fixture

    tahoe = _copy(tahoe_root, tmp_path_factory, "pf-tahoe")
    (tahoe / F.TAHOE_DE).unlink()
    cfg = _config(tmp_path_factory.mktemp("pf-tahoe-cfg"), ot, tahoe)
    results = require_ready(cfg, provider=PROVIDER)
    tools = degraded_tools(cfg, results)
    new = set(tools) - set(before)
    assert new and all(t.startswith("mcp__functional_genomics__") for t in new), new
    assert all("tahoe_100m.de_permissive" in tools[t] for t in new)
    assert not depmap & set(tools)                               # a missing Tahoe file leaves DepMap ready
    tahoe_label = next(r for r in results if "TAHOE_DATA_PATH" in r.label)
    assert not tahoe_label.ok and tahoe_label.scope == {"source": "tahoe_100m"}
    assert "functional_genomics" not in degraded_servers(cfg, results)


def test_require_ready_blocks_only_when_no_granted_tool_is_ready(tahoe_root: Path, tmp_path: Path) -> None:
    empty = tmp_path / "no-ot"
    empty.mkdir()
    cfg = _config(tmp_path / "cfg", empty, tahoe_root)
    results = require_ready(cfg, provider=PROVIDER)            # PubMed / ClinicalTrials.gov tools are ready
    servers = degraded_servers(cfg, results)
    assert {"target", "disease", "drug"} <= set(servers), servers
    assert "pubmed" not in servers and "clinicaltrials" not in servers
    ot = next(r for r in results if "OPEN_TARGETS_DATA_PATH" in r.label)
    assert not ot.ok                                          # no granted Open Targets tool is ready
    only_ot = _config(tmp_path / "cfg-ot", empty, tahoe_root)
    only_ot["mcp_servers"]["servers"] = [s for s in only_ot["mcp_servers"]["servers"] if s["name"] in ("target", "drug")]
    with pytest.raises(DataReadinessError, match=DATA_TOOLS_LABEL):
        require_ready(only_ot, provider=PROVIDER)
    assert require_ready(only_ot, provider=PROVIDER, allow_missing_data=True)


# ---------------------------------------------------------------------------- smoke through the gateway


def _smoke(cfg: dict[str, Any], tmp_path: Path) -> list[Any]:
    return asyncio.run(preflight.smoke_mcp(cfg, log_dir=tmp_path / "logs", servers=["target"]))


def _target_variant(tmp_path_factory: pytest.TempPathFactory, label: str, edit: Any) -> Path:
    import dl_fixtures as F

    root = F.build_ot_fixture(tmp_path_factory.mktemp(label) / "25.09")
    rows = edit(F.ot_rows()["target"])
    F.delete_table(root, "target")
    F.write_table(root, "target", F.table("target", rows), shards=F.TABLE_SHARDS.get("target", 1))
    F.write_manifest(root)
    return root


def _controls(results: list[Any]) -> dict[str, Any]:
    return {r.label.rsplit("[", 1)[1].split()[0]: r for r in results if r.label.endswith(" control]")}


@needs_fastmcp
def test_smoke_runs_sentinel_controls(ot_root: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path, ot_root, servers=["target"])
    results = _smoke(cfg, tmp_path)
    controls = _controls(results)
    assert set(controls) == {"positive", "negative"}, [r.line() for r in results]
    assert all(r.ok for r in controls.values()), [r.line() for r in results]
    assert "ENSG00000169174" in controls["positive"].label


@needs_fastmcp
def test_smoke_fails_on_an_empty_sentinel(tmp_path_factory: pytest.TempPathFactory) -> None:
    import dl_fixtures as F

    root = _target_variant(tmp_path_factory, "pf-nosentinel", lambda rows: [r for r in rows if r["id"] != F.PCSK9])
    tmp = tmp_path_factory.mktemp("pf-smoke-empty")
    results = _smoke(_config(tmp, root, servers=["target"]), tmp)
    controls = _controls(results)
    assert not controls["positive"].ok, [r.line() for r in results]


@needs_fastmcp
def test_smoke_fails_on_a_phantom_negative(tmp_path_factory: pytest.TempPathFactory) -> None:
    import dl_fixtures as F

    def add_phantom(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pcsk9 = next(r for r in rows if r["id"] == F.PCSK9)
        return rows + [{**pcsk9, "id": "ENSG00000000000", "approvedSymbol": "PHANTOM"}]

    root = _target_variant(tmp_path_factory, "pf-phantom", add_phantom)
    tmp = tmp_path_factory.mktemp("pf-smoke-phantom")
    results = _smoke(_config(tmp, root, servers=["target"]), tmp)
    controls = _controls(results)
    assert controls["positive"].ok, [r.line() for r in results]
    assert not controls["negative"].ok and "phantom" in controls["negative"].detail, [r.line() for r in results]


def test_judge_control() -> None:
    from vbt.datalayer.errors import ErrorKind, GatewayError

    pos = {"kind": "positive", "key": {"id": "ENSG00000169174"}, "tool": "get_target_info", "args": {}}
    neg = {"kind": "negative", "key": {"id": "ENSG00000000000"}, "tool": "get_target_info", "args": {}}
    ok_result = SimpleNamespace(is_data_result=True, status="ok", text='{"id": "ENSG00000169174"}', full_text=None)
    empty = SimpleNamespace(is_data_result=True, status="empty", text="{}", full_text=None)
    assert preflight.judge_control(pos, ok_result)[0]
    assert not preflight.judge_control(pos, empty)[0]
    assert not preflight.judge_control(pos, SimpleNamespace(is_data_result=True, status="ok", text="{}",
                                                            full_text=None))[0]
    nf = GatewayError(ErrorKind.not_found, "no such target")
    assert not preflight.judge_control(pos, None, nf)[0]
    assert preflight.judge_control(neg, None, nf)[0]
    assert not preflight.judge_control(neg, ok_result)[0]
    assert not preflight.judge_control(neg, None, GatewayError(ErrorKind.source_error, "boom"))[0]
