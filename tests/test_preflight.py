"""Preflight / doctor: credentials, reference data, MCP commands, smoke calls."""

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

from vbt import preflight
from vbt.config import load_config
from vbt.preflight import (
    TURN_NOT_SENT,
    DataReadinessError,
    check_credentials,
    check_mcp_commands,
    check_open_targets,
    check_reference_data,
    check_tahoe,
    degraded_servers,
    mcp_modules,
    require_ready,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _live_config(monkeypatch, tmp_path, **env):
    """Anthropic-provider config with the default MCP servers (nothing is started)."""
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    # the default is a local model now; these checks exercise the Anthropic key handling
    cfg = load_config(["claude"], overrides={"paths": {"runs_dir": str(tmp_path / "runs")}})
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
