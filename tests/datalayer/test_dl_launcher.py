"""Reaper launcher (§14.2): RLIMIT_DATA, exit marker, signal forwarding, status file, and an
effective hash seed for ``python -E`` children. Linux only; servers need fastmcp."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from vbt.datalayer.api import CallPlan, ListingDecision
from vbt.datalayer.errors import ErrorKind, GatewayError
from vbt.datalayer.launch import CHILD_ENV, REAPER, build_launch_spec, server_limit_mb
from vbt.datalayer.launch import reaper as reaper_mod
from vbt.datalayer.memory.crash import classify_error_text, crash_decision, parse_exit_marker
from vbt.datalayer.settings import DataSettings
from vbt.tools.mcp_bridge import MCPBridge, MCPServerConfig

SERVERS = Path(__file__).resolve().parent / "servers"
OOM = str(SERVERS / "oom_fixture_server.py")
HASHSEED = str(SERVERS / "hashseed_fixture_server.py")

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the reaper needs Linux")
needs_fastmcp = pytest.mark.skipif(
    importlib.util.find_spec("fastmcp") is None or importlib.util.find_spec("mcp") is None,
    reason="fastmcp/mcp not installed (pip install -e '.[mcp]')")

FAST = {"start_backoff_s": 0.01, "start_backoff_factor": 1.0, "start_timeout_s": 60, "start_attempts": 1}


class LaunchGateway:
    """A pass-through gateway that launches every stdio server under the reaper and classifies
    crashes with ``memory.crash`` (what DataGateway does, minus everything else)."""

    mode = "enforce"

    def __init__(self, settings: DataSettings | None = None) -> None:
        self.settings = settings or DataSettings.from_dict({})
        self.bridge = None
        self.crashes: list = []

    def bind_bridge(self, bridge):
        self.bridge = bridge

    def extra_servers(self):
        return []

    def launch_spec(self, cfg):
        return build_launch_spec(cfg, self.settings, self.bridge.log_root())

    def rewrite_listing(self, server, tool, description, input_schema):
        return ListingDecision(True, description, input_schema)

    async def prepare(self, server, tool, args, ctx):
        return CallPlan(server, tool, dict(args), dict(args), "upstream", None, [], None)

    async def finish(self, plan, raw):
        return raw

    async def on_crash(self, server, plan, reason, log_tail):
        decision = crash_decision(reason, log_tail, server=server, tool=plan.tool)
        self.crashes.append((reason, log_tail, decision))
        return decision

    def pinned(self):
        return {}

    def readiness_snapshot(self):
        return {}


def py_server(name, script, *args, **kw):
    return MCPServerConfig(name=name, command=sys.executable, args=["-E", script, *map(str, args)], **kw)


# --------------------------------------------------------------------------- pure


def test_strip_python_flags():
    strip = reaper_mod.strip_python_flags
    assert strip(["-E", "server.py", "-E"]) == (["server.py", "-E"], ["-E"])
    assert strip(["-I", "-B", "s.py"]) == (["-B", "s.py"], ["-I"])
    assert strip(["-EsB", "-W", "ignore", "-Xdev", "s.py", "--x"]) == (["-sB", "-W", "ignore", "-Xdev", "s.py", "--x"],
                                                                       ["-E"])
    assert strip(["-IE", "-c", "print(1)", "-E"]) == (["-c", "print(1)", "-E"], ["-I", "-E"])
    assert strip(["-Ec", "code"]) == (["-c", "code"], ["-E"])
    assert strip(["-m", "pkg.mod", "-I"]) == (["-m", "pkg.mod", "-I"], [])
    assert strip(["--check-hash-based-pycs", "always", "-E", "s.py"]) == (
        ["--check-hash-based-pycs", "always", "s.py"], ["-E"])
    assert reaper_mod.is_python("/usr/bin/python3.11") and reaper_mod.is_python("python")
    assert not reaper_mod.is_python("/usr/bin/node")


def test_child_environment_is_explicit():
    env = {"PATH": "/bin", "PYTHONPATH": "/evil", "PYTHONHOME": "/evil", "PYTHONSTARTUP": "/evil/s.py",
           "PYTHONHASHSEED": "random", "PYTHONUSERBASE": "/home/u/.local", "OPEN_TARGETS_DATA_PATH": "/ot"}
    out = reaper_mod.child_environment(env, isolated=False)
    assert out == {"PATH": "/bin", "OPEN_TARGETS_DATA_PATH": "/ot", "PYTHONHASHSEED": "0",
                   "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUSERBASE": "/home/u/.local"}
    iso = reaper_mod.child_environment({"PATH": "/bin"}, isolated=True)
    assert iso["PYTHONNOUSERSITE"] == "1" and iso["PYTHONSAFEPATH"] == "1" and "PYTHONUSERBASE" not in iso


def test_build_launch_spec(tmp_path):
    settings = DataSettings.from_dict({"memory": {"default_server_mb": 4096}, "service": {"mem_limit_mb": 1500}})
    cfg = py_server("target", "/srv/target/server.py")
    spec = build_launch_spec(cfg, settings, tmp_path)
    if not sys.platform.startswith("linux"):
        assert spec is None
        return
    assert spec.command == sys.executable
    assert spec.args[:2] == ["-E", str(REAPER)] and Path(spec.args[1]).is_absolute()
    cut = spec.args.index("--")
    assert spec.args[cut + 1:] == [sys.executable, "-E", "/srv/target/server.py"]
    assert spec.args[spec.args.index("--limit-mb") + 1] == "4096"
    assert spec.args[spec.args.index("--server") + 1] == "target"
    assert spec.status_path == str(tmp_path / "target.status.json")
    assert spec.env == CHILD_ENV == {"ARROW_DEFAULT_MEMORY_POOL": "system", "MALLOC_ARENA_MAX": "2",
                                     "PRELOAD_MCP_DATA": "0"}
    # per-server limit, the data child's default, the kill switch, HTTP servers
    assert server_limit_mb(MCPServerConfig("t", command="x", mem_limit_mb=512), settings) == 512
    assert server_limit_mb(MCPServerConfig("data", command="x"), settings) == 1500
    assert server_limit_mb(cfg, DataSettings.from_dict({"memory": {"limit_kind": "none"}})) == 0
    assert build_launch_spec(MCPServerConfig("t", command="x", launcher=False), settings, tmp_path) is None
    assert build_launch_spec(MCPServerConfig("h", url="http://localhost:1/mcp"), settings, tmp_path) is None
    auto = server_limit_mb(cfg, DataSettings.from_dict({"memory": {"default_server_mb": "auto"}}))
    assert auto > 0


# --------------------------------------------------------------------------- the reaper process


def _wait_for(predicate, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("timed out")


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


@linux_only
def test_sigterm_forwarded_status_file_and_exit_marker(tmp_path):
    ready, got = tmp_path / "ready", tmp_path / "got_term"
    child = ("import os, signal, sys, time\n"
             f"def on_term(*a):\n    open({str(got)!r}, 'w').write(os.environ.get('PYTHONHASHSEED', '?'))\n"
             "    os._exit(7)\n"
             "signal.signal(signal.SIGTERM, on_term)\n"
             f"open({str(ready)!r}, 'w').write(str(os.getpid()))\n"
             "time.sleep(60)\n")
    status = tmp_path / "logs" / "probe.status.json"
    proc = subprocess.Popen([sys.executable, "-E", str(REAPER), "--limit-mb", "256", "--status", str(status),
                             "--server", "probe", "--", sys.executable, "-E", "-c", child],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        child_pid = int(_wait_for(lambda: ready.exists() and ready.read_text()))
        data = _wait_for(lambda: _read_json(status))
        assert data["pid"] == child_pid and data["server"] == "probe" and data["limit_mb"] == 256
        assert data["containment"] == "rlimit_data" and data["hash_seed"] == 0
        assert data["flags_stripped"] == ["-E"] and "ts" in data
        assert _wait_for(lambda: (_read_json(status) or {}).get("rss_mb"))
        # the child runs in its own process group under RLIMIT_DATA
        assert os.getpgid(child_pid) == child_pid != os.getpgid(proc.pid)
        limits = Path(f"/proc/{child_pid}/limits").read_text()
        assert any(line.startswith("Max data size") and str(256 * 1024 * 1024) in line
                   for line in limits.splitlines())
        proc.send_signal(signal.SIGTERM)
        _, err = proc.communicate(timeout=20)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert got.read_text() == "0", "SIGTERM reached the child, whose PYTHONHASHSEED is 0"
    assert proc.returncode == 7
    marker = parse_exit_marker(err.decode())
    assert marker == {"pid": child_pid, "code": 7, "signal": None, "maxrss_kb": marker["maxrss_kb"],
                      "reason": "exit_code"}
    assert marker["maxrss_kb"] > 0
    final = _read_json(status)
    assert final["exit"] == marker


@linux_only
def test_child_dies_with_a_killed_reaper(tmp_path):
    """The bridge's last resort is SIGKILL to the reaper's group; the child (in its own group)
    must not outlive it."""
    ready = tmp_path / "ready"
    proc = subprocess.Popen([sys.executable, "-E", str(REAPER), "--limit-mb", "0", "--server", "orphan", "--",
                             sys.executable, "-c",
                             f"import os, time; open({str(ready)!r}, 'w').write(str(os.getpid())); time.sleep(60)"],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    child_pid = int(_wait_for(lambda: ready.exists() and ready.read_text()))
    proc.kill()
    proc.wait(timeout=10)

    def gone():
        try:
            state = Path(f"/proc/{child_pid}/stat").read_text().split(")")[-1].split()[0]
        except OSError:
            return True
        return state == "Z"

    assert _wait_for(gone, timeout=10)


@linux_only
def test_killed_child_gives_signal_marker_and_rc(tmp_path):
    proc = subprocess.run([sys.executable, "-E", str(REAPER), "--limit-mb", "0", "--status", str(tmp_path / "s.json"),
                           "--server", "k", "--", sys.executable, "-c",
                           "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 128 + 9
    marker = parse_exit_marker(proc.stderr)
    assert marker["signal"] == 9 and marker["code"] is None and marker["reason"] == "signal"
    assert _read_json(tmp_path / "s.json")["containment"] == "none"
    missing = subprocess.run([sys.executable, "-E", str(REAPER), "--limit-mb", "0", "--server", "m", "--",
                              str(tmp_path / "no-such-binary")], capture_output=True, text=True, timeout=30)
    assert missing.returncode == 127 and "cannot start" in missing.stderr
    assert parse_exit_marker(missing.stderr)["code"] == 127


# --------------------------------------------------------------------------- through the bridge


@linux_only
@needs_fastmcp
async def test_allocation_over_limit_is_oom_and_server_survives(tmp_path):
    gw = LaunchGateway()
    bridge = MCPBridge([py_server("oom", OOM, tmp_path / "state", mem_limit_mb=512)], log_dir=tmp_path / "logs",
                       options=FAST, gateway=gw)
    try:
        await bridge.start()
        assert bridge.failures == {}
        ping = await bridge.call("oom", "ping", {})
        info = json.loads(ping.text)
        assert info["rlimit_data"] == 512 * 1024 * 1024
        raw = await bridge.call("oom", "allocate", {"mb": 2048})
        assert raw.envelope == "is_error" and "Unable to allocate" in raw.error_text
        assert classify_error_text(raw.error_text) == "oom"
        small = await bridge.call("oom", "allocate", {"mb": 16})
        assert small.envelope == "ok" and "allocated 16 MiB" in small.text
        after = json.loads((await bridge.call("oom", "ping", {})).text)
        assert after["start"] == 1 and after["pid"] == info["pid"], "the server survived"
        st = bridge.status()["oom"]
        assert st["restarts"] == 0 and st["oom_kills"] == 0 and gw.crashes == []
        assert st["status_file"] and _read_json(st["status_file"])["hash_seed"] == 0
    finally:
        await bridge.aclose()


@linux_only
@needs_fastmcp
async def test_sigkill_is_oom_killed_not_retried_and_separately_budgeted(tmp_path):
    events = []
    gw = LaunchGateway()
    state = tmp_path / "state"
    bridge = MCPBridge([py_server("oom", OOM, state)], log_dir=tmp_path / "logs",
                       options={**FAST, "max_restarts": 0, "max_oom_kills": 1}, gateway=gw,
                       on_event=lambda kind, **d: events.append((kind, d)))
    try:
        await bridge.start()
        with pytest.raises(GatewayError) as exc:
            await bridge.call("oom", "sigkill_self", {})
        err = exc.value
        assert err.kind is ErrorKind.oom and err.subkind == "oom_killed" and err.retryable == "no"
        assert err.payload["exit"]["signal"] == 9
        st = bridge.status()["oom"]
        assert st["state"] == "broken" and st["oom_kills"] == 1 and st["restarts"] == 0
        assert (state / "starts").read_text() == "1", "an OOM kill is not retried"
        log = Path(st["log"]).read_text()
        assert parse_exit_marker(log)["signal"] == 9
        crash = [d for k, d in events if k == "mcp_crash"][0]
        assert "VBT_CHILD_EXIT" in crash["stderr_tail"], "the bridge waited for the exit marker"

        # The next call restarts the server first; max_restarts=0 does not apply to OOM kills.
        assert json.loads((await bridge.call("oom", "ping", {})).text)["start"] == 2
        st = bridge.status()["oom"]
        assert st["state"] == "ready" and st["restarts"] == 0 and st["generation"] == 2

        # A second kill exceeds max_oom_kills=1: the server is given up.
        with pytest.raises(GatewayError):
            await bridge.call("oom", "sigkill_self", {})
        st = bridge.status()["oom"]
        assert st["state"] == "failed" and st["oom_kills"] == 2 and "OOM kill limit" in bridge.failures["oom"]
    finally:
        await bridge.aclose()


@linux_only
@needs_fastmcp
async def test_internal_call_to_launched_server_is_not_retried_after_a_memory_kill(tmp_path):
    bridge = MCPBridge([py_server("data", OOM, tmp_path / "state")], log_dir=tmp_path / "logs", options=FAST,
                       gateway=LaunchGateway())
    try:
        await bridge.start()
        with pytest.raises(GatewayError) as exc:
            await bridge.call_raw("data", "sigkill_self", {})
        assert exc.value.kind is ErrorKind.oom
        assert (tmp_path / "state" / "starts").read_text() == "1"
        assert bridge.status()["data"]["oom_kills"] == 1
    finally:
        await bridge.aclose()


async def _set_order(tmp_path, name, cfg_kwargs, gateway):
    bridge = MCPBridge([py_server(name, HASHSEED, **cfg_kwargs)], log_dir=tmp_path / f"logs-{name}", options=FAST,
                       gateway=gateway)
    try:
        await bridge.start()
        assert bridge.failures == {}, bridge.failures
        out = await bridge.call(name, "set_order", {})
        text = out.text if hasattr(out, "text") else out
        return json.loads(text), bridge.status()[name]["status_file"]
    finally:
        await bridge.aclose()


@linux_only
@needs_fastmcp
async def test_hash_seed_effective(tmp_path):
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    marker = tmp_path / "hostile_ran"
    (hostile / "sitecustomize.py").write_text(
        f"import sys, types\nsys.modules['vbt_hostile_probe'] = types.ModuleType('vbt_hostile_probe')\n"
        f"open({str(marker)!r}, 'w').write('ran')\n")
    env = {"PYTHONPATH": str(hostile), "PYTHONHASHSEED": "random"}

    first, status = await _set_order(tmp_path, "seed1", {"env": env}, LaunchGateway())
    second, _ = await _set_order(tmp_path, "seed2", {"env": env}, LaunchGateway())
    assert first["hash_randomization"] == 0 and second["hash_randomization"] == 0
    assert first["order"] == second["order"], "two launches give the same set order"
    assert first["ignore_environment"] == 0, "-E was stripped from the child"
    assert first["python_env"] == {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}
    assert str(hostile) not in first["sys_path"] and not first["hostile_loaded"]
    assert not marker.exists(), "a hostile PYTHONPATH in the parent env does not reach the child"
    data = _read_json(status)
    assert data["hash_seed"] == 0 and data["flags_stripped"] == ["-E"]

    # Positive control: without the reaper, -E makes CPython ignore PYTHONHASHSEED.
    plain, _ = await _set_order(tmp_path, "plain", {"env": {"PYTHONHASHSEED": "0"}}, None)
    assert plain["hash_randomization"] == 1 and plain["ignore_environment"] == 1


@linux_only
@needs_fastmcp
async def test_launcher_false_runs_server_directly(tmp_path):
    out, status = await _set_order(tmp_path, "direct", {"launcher": False}, LaunchGateway())
    assert status is None and out["ignore_environment"] == 1


@linux_only
async def test_reaper_exits_when_stdin_closes(tmp_path):
    """The bridge stops a server by closing its stdin; the reaper must follow its child out."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-E", str(REAPER), "--limit-mb", "0", "--server", "cat", "--",
        sys.executable, "-c", "import sys; sys.stdin.read()",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    proc.stdin.close()
    rc = await asyncio.wait_for(proc.wait(), 20)
    err = (await proc.stderr.read()).decode()
    assert rc == 0 and parse_exit_marker(err)["code"] == 0
