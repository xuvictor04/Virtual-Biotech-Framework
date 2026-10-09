"""The host configuration and image facts every ``vbt`` command applies (docs/DEPLOYMENT.md), and the session's
between-turns acquisition (docs/DATA_SETUP.md):

* a plain ``vbt`` loads ``host.env`` and the profiles ``vbt setup`` recorded, as ``deploy/full/vbt-host`` does;
* run records made in the image (no ``.git``) pin the commits the image recorded;
* ``vbt doctor --analysis`` reads one line per R package;
* after a turn with ``not_ready`` refusals, ``data.acquisition.auto`` hands them to ``between_turns`` and the next
  turn waits for it and records it.
"""

from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path
from types import SimpleNamespace

from vbt import cli, pinning, preflight

# ------------------------------------------------------------------------------------------------ host.env


def _write_env(state: Path, **values: str) -> Path:
    state.mkdir(parents=True, exist_ok=True)
    path = state / "host.env"
    path.write_text("# written by vbt setup\n" + "".join(f"{k}='{v}'\n" for k, v in values.items()))
    return path


def test_a_plain_vbt_applies_the_host_configuration(tmp_path):
    state = tmp_path / "state"
    _write_env(state, VBT_PROFILES=f"production {state}/host.yaml", OPEN_TARGETS_DATA_PATH="/d/ot/25.09",
               VBT_LLM_BASE_URL="http://gpu:8000/v1")
    (state / "host.yaml").write_text("{}\n")                                 # a profile named by path exists
    env = {"VBT_STATE_DIR": str(state), "VBT_LLM_BASE_URL": "http://mine:8000/v1", "TAHOE_DATA_PATH": ""}
    args = SimpleNamespace(cmd="chat", profile=["no-web"], resume=None)
    assert cli.apply_host_config(args, env) == state / "host.env"
    assert args.profile == ["production", f"{state}/host.yaml", "no-web"]     # the command's own flags last
    assert env["OPEN_TARGETS_DATA_PATH"] == "/d/ot/25.09"
    assert env["VBT_LLM_BASE_URL"] == "http://mine:8000/v1"                  # a variable already set wins


def test_resume_replay_setup_and_the_switch(tmp_path):
    state = tmp_path / "home" / "state"
    _write_env(state, VBT_PROFILES="production host.yaml")
    base = {"VBT_HOME": str(tmp_path / "home")}
    resumed = SimpleNamespace(cmd="chat", profile=[], resume="latest")
    cli.apply_host_config(resumed, dict(base))
    assert resumed.profile == []                                             # keeps its pinned profiles
    replay = SimpleNamespace(cmd="replay", profile=[])
    cli.apply_host_config(replay, dict(base))
    assert replay.profile == []
    setup = SimpleNamespace(cmd="setup", profile=[])
    cli.apply_host_config(setup, {**base, "VBT_BASE_PROFILES": "production"})
    assert setup.profile == ["production"]                                   # never the previous host.yaml
    off = SimpleNamespace(cmd="chat", profile=[])
    env = {**base, cli.NO_HOST_ENV: "1"}
    assert cli.apply_host_config(off, env) is None and off.profile == [] and "VBT_PROFILES" not in env
    # before the first setup: the operator's base profiles
    fresh = SimpleNamespace(cmd="chat", profile=[])
    assert cli.apply_host_config(fresh, {"VBT_STATE_DIR": str(tmp_path / "none"), "VBT_BASE_PROFILES": "x"}) is None
    assert fresh.profile == ["x"]


# ------------------------------------------------------------------------------------------------ pinning


def test_git_info_falls_back_to_the_commits_the_image_recorded(monkeypatch):
    monkeypatch.setattr(pinning, "_git", lambda args, cwd: None)                # no .git in the image
    monkeypatch.setenv(pinning.HARNESS_COMMIT_ENV, "a" * 40)
    monkeypatch.setenv(pinning.UPSTREAM_COMMIT_ENV, "b" * 40)
    info = pinning.git_info("/nonexistent/upstream")
    assert info["commit"] == "a" * 40 and info["dirty"] is None
    assert info["upstream_commit"] == "b" * 40
    assert info["commit_source"] == "environment (VBT_HARNESS_COMMIT)"
    monkeypatch.delenv(pinning.HARNESS_COMMIT_ENV)
    monkeypatch.delenv(pinning.UPSTREAM_COMMIT_ENV)
    assert pinning.git_info(None)["commit"] is None


# ------------------------------------------------------------------------------------------------ R packages


def test_the_r_check_reads_one_line_per_package(tmp_path, monkeypatch):
    """The R code prints one line per package; output with leading blanks (an older ``cat`` of a vector) parses
    too."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    argv_file = tmp_path / "argv.txt"
    rscript = bindir / "Rscript"
    lines = "".join(f"{' ' if i else ''}VBT_R {p}=TRUE\\n" for i, p in enumerate(preflight.R_PACKAGES[:-1]))
    rscript.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$@\" > {argv_file}\nprintf '{lines}VBT_R "
                       f"{preflight.R_PACKAGES[-1]}=FALSE\\n'\n")
    rscript.chmod(rscript.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setattr(preflight, "probe_imports", lambda *a, **k: {})
    out = {r.label.removeprefix("analysis: R package "): r.ok for r in preflight.check_analysis_stack({})
           if r.label.startswith("analysis: R package")}
    assert out == {**{p: True for p in preflight.R_PACKAGES[:-1]}, preflight.R_PACKAGES[-1]: False}
    assert "writeLines(" in argv_file.read_text()


# ------------------------------------------------------------------------------------------------ between turns


class _Run:
    def __init__(self, events):
        self._events = list(events)
        self.dir = Path("/tmp/run")

    def events(self):
        return list(self._events)

    def trace(self, kind, **data):
        self._events.append({"type": kind, **data})


class _Gateway:
    def __init__(self):
        self.refreshed = []

    async def refresh_readiness(self, tables=(), depth=None):
        self.refreshed.append(list(tables))
        return True


def _session(auto, events):
    from vbt.orchestrator import CSOSession

    emitted = []
    rt = SimpleNamespace(config={"data": {"acquisition": {"auto": auto}}, "orchestration": {}}, run=_Run(events),
                         gateway=_Gateway(), emit=lambda kind, **d: emitted.append((kind, d)))
    return CSOSession(rt), rt, emitted


REFUSED = {"type": "tool_end", "tool": "mcp__target__get_target_info", "error_kind": "not_ready",
           "not_ready": [{"name": "open_targets.target", "check": "missing", "detail": "no files", "hint": "h",
                          "acquire": {"table": "open_targets.target", "command": "vbt data acquire open_targets.target"}}]}


def test_between_turns_hands_the_refusals_over_and_records_the_outcome(monkeypatch):
    from vbt.data import ondemand

    calls = []

    def fake(config, *, refusals=None, run_dir=None, **kw):
        calls.append((refusals, run_dir))
        report = SimpleNamespace(bytes_downloaded=1234, seconds=0.5,
                                 sources=[SimpleNamespace(env={"OPEN_TARGETS_DATA_PATH": "/acq/ot/25.09"})])
        return {"policy": "under_budget", "budget_bytes": 10**9, "wanted": {"open_targets": ["target"]},
                "bytes": 1234, "decision": "acquired", "report": report}

    monkeypatch.setattr(ondemand, "between_turns", fake)
    monkeypatch.delenv("OPEN_TARGETS_DATA_PATH", raising=False)
    session, rt, emitted = _session("under_budget", [{"type": "tool_end", "tool": "x"}, REFUSED])

    async def go():
        session._start_acquisition(0)
        assert session._acquisition is not None
        return await session.settle_acquisition()

    rec = asyncio.run(go())
    assert calls == [([{"tables": REFUSED["not_ready"]}], rt.run.dir)]
    assert rec["decision"] == "acquired" and rec["bytes_downloaded"] == 1234
    assert rec["env_needed"] == {"OPEN_TARGETS_DATA_PATH": "/acq/ot/25.09"}
    assert rt.gateway.refreshed == [["open_targets.target"]]
    kinds = [e["type"] for e in rt.run.events()]
    assert kinds[-2:] == ["data_acquisition_start", "data_acquisition"]
    assert emitted and emitted[-1][0] == "data_acquisition" and session.last_acquisition == rec


def test_between_turns_is_off_by_default_and_without_refusals(monkeypatch):
    from vbt.data import ondemand

    monkeypatch.setattr(ondemand, "between_turns", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    for auto, events in ((False, [REFUSED]), ("off", [REFUSED]), ("ask", [{"type": "tool_end", "tool": "x"}])):
        session, _rt, _ = _session(auto, events)
        session._start_acquisition(0)
        assert session._acquisition is None
        assert asyncio.run(session.settle_acquisition()) is None


def test_a_failed_acquisition_is_recorded_not_raised(monkeypatch):
    from vbt.data import ondemand

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(ondemand, "between_turns", boom)
    session, rt, _ = _session("ask", [REFUSED])

    async def go():
        session._start_acquisition(0)
        return await session.settle_acquisition()

    rec = asyncio.run(go())
    assert rec["decision"].startswith("failed: RuntimeError: disk full") and rt.gateway.refreshed == []
