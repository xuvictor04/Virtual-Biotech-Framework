"""Preflight / doctor: credentials, reference data, MCP commands, smoke calls."""

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

from vbt import preflight
from vbt.config import load_config
from vbt.preflight import (
    DATA_TOOLS_LABEL,
    TURN_NOT_SENT,
    DataReadinessError,
    check_credentials,
    check_mcp_commands,
    check_open_targets,
    check_reference_data,
    check_tahoe,
    degraded_servers,
    degraded_tables,
    degraded_tools,
    mcp_modules,
    require_ready,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _live_config(monkeypatch, tmp_path, *, data=False, **env):
    """Anthropic-provider config with the default MCP servers (nothing is started). The whole-release
    checks need the data layer off (``data.enabled: false``); ``data=True`` keeps it on, with its
    cache under ``tmp_path``."""
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    # the default is a local model now; these checks exercise the Anthropic key handling
    cfg = load_config(["claude"], overrides={"paths": {"runs_dir": str(tmp_path / "runs")},
                                             "data": {"enabled": bool(data), "cache_dir": str(tmp_path / "dl")}})
    assert cfg["provider"]["name"] == "anthropic"
    return cfg


def _fake_open_targets(root: Path, config) -> Path:
    doctor = preflight.upstream_doctor(config)
    if doctor is None:
        pytest.skip("upstream tools/doctor.py not importable")
    for name in doctor.OPEN_TARGETS_DATASETS:
        (root / name).mkdir(parents=True)
        (root / name / "part-00000.parquet").write_bytes(b"PAR1" + b"\0" * 16 + b"PAR1")
    return root


def test_blank_key_fails(monkeypatch, tmp_path):
    cfg = _live_config(monkeypatch, tmp_path, ANTHROPIC_API_KEY="   ")
    r = check_credentials(cfg)
    assert not r.ok and "blank" in r.hint
    with pytest.raises(DataReadinessError) as exc:
        require_ready(cfg)
    assert "ANTHROPIC_API_KEY" in str(exc.value) and TURN_NOT_SENT in str(exc.value)
    # credentials are never waived by allow_missing_data
    with pytest.raises(DataReadinessError):
        require_ready(cfg, allow_missing_data=True)

    class Provider:
        name = "anthropic"

        def check_credentials(self):
            return "ANTHROPIC_API_KEY is blank"

    assert not check_credentials(cfg, Provider()).ok
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real")
    assert check_credentials(cfg).ok


def test_missing_open_targets_raises_with_fix(monkeypatch, tmp_path):
    cfg = _live_config(monkeypatch, tmp_path, ANTHROPIC_API_KEY="sk-ant-test", OPEN_TARGETS_DATA_PATH="",
                       TAHOE_DATA_PATH="")
    for per_turn in (False, True):
        with pytest.raises(DataReadinessError) as exc:
            require_ready(cfg, per_turn=per_turn)
        msg = str(exc.value)
        assert "OPEN_TARGETS_DATA_PATH" in msg and "download_open_targets.py" in msg
        assert msg.endswith(TURN_NOT_SENT)
        assert any(not r.ok and r.kind == "data" for r in exc.value.results)

    results = require_ready(cfg, allow_missing_data=True)
    degraded = degraded_servers(cfg, results)
    assert {"genetics", "target", "association"} <= set(degraded)
    assert "single_cell" not in degraded and "clinicaltrials" not in degraded

    monkeypatch.setenv("OPEN_TARGETS_DATA_PATH", str(tmp_path / "nowhere"))
    cfg["tool_env"]["OPEN_TARGETS_DATA_PATH"] = ""   # falls back to the live environment
    with pytest.raises(DataReadinessError, match="not a directory"):
        require_ready(cfg, per_turn=True)


def test_no_mcp_checks_credentials_only(monkeypatch, tmp_path):
    """--no-mcp: missing reference data / broken server commands do not block or degrade."""
    cfg = _live_config(monkeypatch, tmp_path, ANTHROPIC_API_KEY="sk-ant-test", OPEN_TARGETS_DATA_PATH="",
                       TAHOE_DATA_PATH="")
    with pytest.raises(DataReadinessError):
        require_ready(cfg)
    for per_turn in (False, True):
        results = require_ready(cfg, per_turn=per_turn, start_mcp=False)
        assert [r.kind for r in results] == ["credentials"] and results[0].ok
        assert degraded_servers(cfg, results) == {}
    # credentials still gate a --no-mcp session
    monkeypatch.setenv("ANTHROPIC_API_KEY", " ")
    with pytest.raises(DataReadinessError, match="ANTHROPIC_API_KEY"):
        require_ready(cfg, start_mcp=False)


def test_mock_and_disabled_preflight_skip(monkeypatch, config, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.delenv("OPEN_TARGETS_DATA_PATH", raising=False)
    assert config["provider"]["name"] == "mock"
    assert require_ready(config) == []
    assert require_ready(config, per_turn=True) == []

    live = _live_config(monkeypatch, tmp_path, ANTHROPIC_API_KEY="")

    class Mock:
        name = "mock"

    assert require_ready(live, provider=Mock()) == []
    live["orchestration"]["require_reference_data"] = False
    assert require_ready(live) == []


def test_open_targets_layout_via_upstream_doctor(monkeypatch, tmp_path):
    cfg = _live_config(monkeypatch, tmp_path, ANTHROPIC_API_KEY="sk-ant-test", TAHOE_DATA_PATH="")
    root = _fake_open_targets(tmp_path / "ot", cfg)
    cfg["tool_env"]["OPEN_TARGETS_DATA_PATH"] = str(root)
    r = check_open_targets(cfg)
    assert r.ok, r.detail
    assert "datasets" in r.detail
    results = require_ready(cfg)
    assert all(x.ok for x in results if x.required), [x.line() for x in results]
    assert "src" not in sys.modules or "TheVirtualBiotech" not in str(getattr(sys.modules["src"], "__path__", "")), \
        "upstream packages must not stay imported"

    (root / "target" / "part-00000.parquet").write_bytes(b"PAR1")    # truncated file
    r = check_open_targets(cfg)
    assert not r.ok and "truncated" in r.detail.lower()
    (root / "target" / "part-00000.parquet").write_bytes(b"PAR1" + b"\0" * 16 + b"PAR1")
    (root / "disease" / "part-00001.parquet.part").write_bytes(b"x")  # partial download
    r = check_open_targets(cfg)
    assert not r.ok and "partial" in r.detail.lower()


def test_tahoe_layout(monkeypatch, tmp_path):
    cfg = _live_config(monkeypatch, tmp_path, ANTHROPIC_API_KEY="sk-ant-test", TAHOE_DATA_PATH="")
    assert check_tahoe(cfg) is None
    tahoe = tmp_path / "tahoe"
    tahoe.mkdir()
    cfg["tool_env"]["TAHOE_DATA_PATH"] = str(tahoe)
    r = check_tahoe(cfg)
    assert not r.ok
    for needle in ("tahoe_permissive_padj010.parquet", "pseudobulk_de_significant/",
                   "pseudobulk_de_high_quality/", "metadata/gene_metadata.parquet",
                   "metadata/sample_metadata.parquet"):
        assert needle in r.detail
    (tahoe / "tahoe_permissive_padj010.parquet").write_bytes(b"x" * 20)
    for d in ("pseudobulk_de_significant", "pseudobulk_de_high_quality"):
        (tahoe / d).mkdir()
        (tahoe / d / "part-00000.parquet").write_bytes(b"x" * 20)
    (tahoe / "metadata").mkdir()
    for m in ("gene", "drug", "cell_line", "sample"):
        (tahoe / "metadata" / f"{m}_metadata.parquet").write_bytes(b"x" * 20)
    assert check_tahoe(cfg).ok
    assert any("Tahoe" in x.label for x in check_reference_data(cfg))


def test_mcp_command_checks(monkeypatch, tmp_path):
    cfg = _live_config(monkeypatch, tmp_path, ANTHROPIC_API_KEY="sk-ant-test")
    assert all(r.ok for r in check_mcp_commands(cfg)), [r.line() for r in check_mcp_commands(cfg)]
    mods = mcp_modules(cfg)
    for m in ("cellxgene_census", "tiledbsoma", "pybioportal", "pyarrow", "fastmcp"):
        assert m in mods
    cfg["mcp_servers"] = {"servers": [
        {"name": "blank", "command": "", "args": []},
        {"name": "noscript", "command": sys.executable, "args": [str(tmp_path / "missing_server.py")]},
        {"name": "nocmd", "command": "definitely-not-a-python-xyz", "args": []},
    ]}
    res = {r.label.split()[2].rstrip(":"): r for r in check_mcp_commands(cfg)}
    assert not res["blank"].ok and "empty command" in res["blank"].detail
    assert not res["noscript"].ok and "server script not found" in res["noscript"].detail
    assert not res["nocmd"].ok and "not found on PATH" in res["nocmd"].detail


def test_doctor_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    preflight.add_doctor_parser(sub)
    args = p.parse_args(["doctor", "--smoke", "--analysis"])
    assert args.smoke and args.analysis and args.handler is preflight._doctor_handler
    assert args.smoke == "gateway" and not args.data
    args = p.parse_args(["doctor", "--smoke=upstream", "--data"])
    assert args.smoke == "upstream" and args.data
    assert p.parse_args(["doctor"]).smoke is False


@pytest.mark.skipif(importlib.util.find_spec("fastmcp") is None, reason="fastmcp not installed")
def test_doctor_smoke_fails_when_a_tool_call_errors(config, tmp_path):
    echo = {"name": "echo", "command": sys.executable, "args": ["-E", str(FIXTURES / "echo_env_mcp_server.py")],
            "enabled": True}
    config["mcp"] = {**config["mcp"], "start_backoff_s": 0.01}
    config["mcp_servers"] = {"servers": [{**echo, "smoke": {"tool": "raise_error", "args": {"message": "no data"}}}]}
    lines: list[str] = []
    assert preflight.run_doctor(config, smoke=True, out=lines.append) == 1
    text = "\n".join(lines)
    assert "[ok] MCP echo: starts" in text
    assert "[!!] MCP echo: raise_error" in text and "no data" in text

    config["mcp_servers"] = {"servers": [{**echo, "smoke": {"tool": "legacy_error"}}]}
    lines.clear()
    assert preflight.run_doctor(config, smoke=True, out=lines.append) == 1

    config["mcp_servers"] = {"servers": [{**echo, "smoke": {"tool": "say", "args": {"text": "ok"}}}]}
    lines.clear()
    assert preflight.run_doctor(config, smoke=True, out=lines.append) == 0, "\n".join(lines)
    assert "[ok] MCP echo: say(text='ok')" in "\n".join(lines)


# ---------------------------------------------------------------------------
# tool-scoped readiness (data layer): the data child's check is replaced by a synthetic response
# ---------------------------------------------------------------------------

KNOWN_DRUG_READERS = {"mcp__drug__search_known_drugs"}


def _check(**tables):
    """A ``CheckResponse`` JSON: ``source__table=status`` (failing tables get one table-level R1 finding)."""
    out = {}
    for ref, status in tables.items():
        ref = ref.replace("__", ".")
        checks = [] if status == "ready" else [{"name": "R1:location", "ok": False, "level": "error",
                                                "detail": f"{ref} files absent"}]
        out[ref] = {"status": status, "checks": checks, "fingerprint": "fp1:test"}
    return {"tables": out, "depth": "standard", "table_errors": {}}


def _tool_scoped(monkeypatch, tmp_path, response, servers=None):
    cfg = _live_config(monkeypatch, tmp_path, data=True, ANTHROPIC_API_KEY="sk-ant-test",
                       OPEN_TARGETS_DATA_PATH=str(tmp_path / "ot"), TAHOE_DATA_PATH="")
    if servers is not None:
        cfg["mcp_servers"]["servers"] = [s for s in cfg["mcp_servers"]["servers"] if s["name"] in servers]
    calls = []

    def fake_check(config, *, tables=(), depth=None, timeout=None):
        calls.append(list(tables))
        return response
    monkeypatch.setattr(preflight, "run_data_check", fake_check)
    monkeypatch.setattr(preflight, "check_mcp_commands", lambda cfg: [])
    return cfg, calls


def test_one_missing_table_degrades_only_its_readers(monkeypatch, tmp_path):
    cfg, calls = _tool_scoped(monkeypatch, tmp_path, _check(open_targets__known_drug="missing",
                                                             open_targets__target="ready"))
    results = require_ready(cfg)          # one missing table never blocks the session
    # one check, over the tables the enabled servers' tools read (never the unbound Zenodo archive)
    assert len(calls) == 1 and "open_targets.known_drug" in calls[0]
    assert not [t for t in calls[0] if t.startswith("zenodo")]
    finding = next(r for r in results if r.label.startswith("data: open_targets.known_drug"))
    assert not finding.ok and finding.kind == "data" and finding.required
    assert finding.scope == {"source": "open_targets", "table": "known_drug"}
    assert set(finding.tools) == KNOWN_DRUG_READERS
    assert set(degraded_tools(cfg, results)) == KNOWN_DRUG_READERS
    assert degraded_servers(cfg, results) == {}           # drug and target keep their other tools
    assert degraded_tables(results) == {"open_targets.known_drug": "missing"}
    ot = next(r for r in results if "OPEN_TARGETS_DATA_PATH" in r.label)
    assert ot.ok and ot.scope == {"source": "open_targets"}            # the legacy label, now an aggregate
    summary = next(r for r in results if r.label == DATA_TOOLS_LABEL)
    assert summary.ok and summary.scope["ready"] == summary.scope["granted"] - 1


def test_blocks_only_when_no_granted_data_tool_is_ready(monkeypatch, tmp_path):
    from vbt.preflight import data_catalog

    cfg, _ = _tool_scoped(monkeypatch, tmp_path, {"tables": {}}, servers={"drug"})
    _settings, catalog, _registry = data_catalog(cfg)
    tables = {ref for tool in catalog.tools("drug") for ref in catalog.contract("drug", tool).tables}
    response = _check(**{ref.replace(".", "__"): "missing" for ref in tables})
    monkeypatch.setattr(preflight, "run_data_check", lambda config, **kw: response)
    with pytest.raises(DataReadinessError) as exc:
        require_ready(cfg)
    assert DATA_TOOLS_LABEL in str(exc.value) and TURN_NOT_SENT in str(exc.value)
    results = require_ready(cfg, allow_missing_data=True)
    assert set(degraded_servers(cfg, results)) == {"drug"}           # every drug tool is unready
    cfg["data"]["readiness"]["block_when"] = "never"
    assert require_ready(cfg)


def test_per_turn_reuses_the_session_check(monkeypatch, tmp_path):
    cfg, calls = _tool_scoped(monkeypatch, tmp_path, _check(open_targets__known_drug="missing"))
    require_ready(cfg)
    assert len(calls) == 1 and "open_targets.known_drug" in calls[0]
    calls.clear()
    results = require_ready(cfg, per_turn=True)
    # cached results are reused (stat-only signatures); only tables without a cached result are re-checked
    assert all("open_targets.known_drug" not in c for c in calls)
    assert set(degraded_tools(cfg, results)) == KNOWN_DRUG_READERS


def test_data_check_unavailable_falls_back_to_the_release_checks(monkeypatch, tmp_path):
    cfg = _live_config(monkeypatch, tmp_path, data=True, ANTHROPIC_API_KEY="sk-ant-test", OPEN_TARGETS_DATA_PATH="",
                       TAHOE_DATA_PATH="")

    def broken(config, **kw):
        raise preflight.DataCheckUnavailable("no interpreter")
    monkeypatch.setattr(preflight, "run_data_check", broken)
    results = check_reference_data(cfg)
    note = next(r for r in results if "tool-scoped" in r.label)
    assert not note.ok and not note.required and "no interpreter" in note.detail
    assert any("OPEN_TARGETS_DATA_PATH" in r.label and not r.ok and r.scope is None for r in results)
    with pytest.raises(DataReadinessError, match="OPEN_TARGETS_DATA_PATH"):
        require_ready(cfg)                 # the legacy checks keep their blocking semantics


def test_doctor_lists_unready_tools(monkeypatch, tmp_path):
    cfg, _ = _tool_scoped(monkeypatch, tmp_path, _check(open_targets__known_drug="missing"))
    results = check_reference_data(cfg)
    lines = preflight.tool_readiness_lines(results)
    assert any(ln.strip().startswith("drug:") and "tools ready" in ln for ln in lines)
    assert any("[!!] search_known_drugs" in ln and "known_drug" in ln for ln in lines)
    every = preflight.tool_readiness_lines(results, every_tool=True)
    assert any("[ok] get_drug_info" in ln for ln in every)
