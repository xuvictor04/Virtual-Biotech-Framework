"""Configuration loading: env expansion semantics, .env handling, in-code defaults."""

import os
import sys

import pytest

from vbt import config as vconfig
from vbt.config import PROJECT_ROOT, _expand, load_config, resolve_path


def _unset(monkeypatch, *names):
    """Make ``names`` absent now and restore their original state afterwards,
    even if load_dotenv sets them during the test."""
    for n in names:
        monkeypatch.setenv(n, "placeholder")
        monkeypatch.delenv(n)


def test_env_example_blank_values_keep_defaults(monkeypatch):
    dotenv = pytest.importorskip("dotenv")
    values = dotenv.dotenv_values(PROJECT_ROOT / ".env.example")
    assert "VBT_MCP_PYTHON" in values and "VBT_DATA_DIR" in values
    for k, v in values.items():
        assert v == "", f"{k} should ship blank in .env.example"
        monkeypatch.setenv(k, v)

    cfg = load_config([])
    servers = cfg["mcp_servers"]["servers"]
    assert servers, "default MCP servers should be configured"
    for s in servers:
        assert s["command"].strip(), f"server {s['name']} has an empty command"
        assert s["args"][-1].endswith(".py")
    assert cfg["vars"]["mcp_python"].strip()

    roots = cfg["paths"]["read_roots"]
    assert "data" in roots, "VBT_DATA_DIR='' must fall back to the default 'data' root"
    assert "skills" in roots
    assert f"{cfg['vars']['upstream']}/.claude/skills" in roots
    # Blank optional data paths stay blank (filtered by the runtime), never a bogus default.
    assert cfg["tool_env"]["OPEN_TARGETS_DATA_PATH"] == ""


def test_expand_bash_semantics(monkeypatch):
    _unset(monkeypatch, "VBT_T_UNSET")
    monkeypatch.setenv("VBT_T_EMPTY", "")
    monkeypatch.setenv("VBT_T_SET", "x")
    v = {"empty": "", "full": "val"}
    assert _expand("${VBT_T_UNSET:-d}", v) == "d"
    assert _expand("${VBT_T_EMPTY:-d}", v) == "d"
    assert _expand("${VBT_T_SET:-d}", v) == "x"
    assert _expand("${VBT_T_UNSET-d}", v) == "d"
    assert _expand("${VBT_T_EMPTY-d}", v) == ""
    assert _expand("${VBT_T_SET-d}", v) == "x"
    assert _expand("${VBT_T_UNSET}", v) == ""
    assert _expand("${VBT_T_UNSET:-}", v) == ""
    assert _expand("${vars.empty:-d}", v) == "d"
    assert _expand("${vars.empty-d}", v) == ""
    assert _expand("${vars.missing-d}", v) == "d"
    assert _expand("${vars.full:-d}/sub", v) == "val/sub"
    # nested defaults expand inside-out
    assert _expand("${VBT_T_EMPTY:-${vars.full}/x}", v) == "val/x"
    assert _expand(["${VBT_T_SET}", {"k": "${VBT_T_EMPTY:-z}"}], v) == ["x", {"k": "z"}]


def test_python_vars_and_blank_mcp_python(monkeypatch):
    monkeypatch.setenv("VBT_MCP_PYTHON", "")
    cfg = load_config([], overrides={"vars": {"mcp_python": "${VBT_MCP_PYTHON-}"}})
    assert cfg["vars"]["python"] == sys.executable
    assert cfg["vars"]["python_prefix"] == sys.prefix
    assert cfg["vars"]["mcp_python"] == sys.executable
    cfg = load_config([], overrides={"vars": {"mcp_python": "${vars.python}"}})
    assert cfg["vars"]["mcp_python"] == sys.executable


def test_upstream_env_file_is_loaded(monkeypatch, tmp_path):
    up = tmp_path / "upstream"
    up.mkdir()
    (up / ".env").write_text('VBT_T_UPSTREAM_ONLY="from-upstream"\nVBT_T_EXPORTED="from-file"\n'
                             'OPEN_TARGETS_DATA_PATH="/data/from-upstream-env"\n')
    _unset(monkeypatch, "VBT_T_UPSTREAM_ONLY", "OPEN_TARGETS_DATA_PATH")
    monkeypatch.setenv("VBT_T_EXPORTED", "from-shell")
    monkeypatch.setenv("VBT_UPSTREAM", str(up))
    cfg = load_config([])
    assert cfg["vars"]["upstream"] == str(up)
    assert os.environ["VBT_T_UPSTREAM_ONLY"] == "from-upstream"
    assert os.environ["VBT_T_EXPORTED"] == "from-shell", "exported variables win over .env files"
    assert cfg["tool_env"]["OPEN_TARGETS_DATA_PATH"] == "/data/from-upstream-env"
    assert "/data/from-upstream-env" in cfg["paths"]["read_roots"]
    assert vconfig.env_files(cfg)[-1] == up / ".env"


def test_resolve_path_resolves_absolute_symlinks(tmp_path):
    real = tmp_path / "real_ot"
    real.mkdir()
    link = tmp_path / "ot_link"
    link.symlink_to(real)
    assert resolve_path(str(link)) == real.resolve()
    assert resolve_path(link / "x" / ".." / "y") == real.resolve() / "y"
    assert resolve_path("skills") == (PROJECT_ROOT / "skills").resolve()


def test_in_code_defaults(config):
    assert config["mcp"]["start_attempts"] == 3
    assert config["mcp"]["start_timeout_s"] == 180
    assert config["mcp"]["default_timeout_s"] == 1800
    assert config["mcp"]["max_restarts"] == 3
    assert config["mcp"]["inherit_env"] is False
    assert config["web"]["literature_max_date"] is None
    assert config["orchestration"]["cso_tools"] == "restricted"
    # the mock profile turns the readiness gate off; the default config keeps it on
    assert config["orchestration"]["require_reference_data"] is False
    assert load_config([])["orchestration"]["require_reference_data"] is True
    assert config["agent_overrides"] == {}
    # YAML still wins over the in-code defaults
    cfg = load_config(["mock"], overrides={"mcp": {"max_restarts": 1}})
    assert cfg["mcp"]["max_restarts"] == 1 and cfg["mcp"]["start_attempts"] == 3


def test_no_web_profile(monkeypatch):
    cfg = load_config(["mock", "no-web"])
    assert cfg["web"]["enabled"] is False
    assert cfg["web"]["literature_max_date"] == "2025/01/31"
    assert vconfig.base_tool_env(cfg)["VBT_LITERATURE_MAXDATE"] == "2025/01/31"  # derived, single source
    assert cfg["bash"]["network"] is False
