"""The sandbox project code runs in: utility tests, utility calls and plugin conformance suites.

The same containment as the Bash tool, applied to a command the harness writes (``python -I runner.py ...``):

* **filesystem** (``projects.sandbox``): ``bwrap`` mounts the whole filesystem read-only except the calling
  agent's own work directory and the run's ``.tmp``/``.home``, and hides ``paths.blocked_read``
  (:func:`vbt.tools.builtin.bwrap_argv`); ``auto`` (the default) uses bwrap when it works on this host and
  falls back to the next point; ``none`` never uses it;
* **memory**: the reaper's ``RLIMIT_DATA`` limit (``data.memory.workspace_mb``, as for Bash);
* **network**: off for tests and conformance suites unless ``projects.test_network`` (bwrap ``--unshare-net``,
  else ``unshare -rn`` when available); utility calls follow ``bash.network_isolation`` like Bash;
* **environment**: the allow-listed child environment (no provider keys) plus the harness ``tool_env``;
* **time and output**: a timeout that kills the process group, and an output cap that keeps head and tail.

:func:`sandbox_label` names what applied (``bwrap+netns``, ``rlimit+netns``, ``rlimit``, ``none``); it goes into
the provenance of every test run.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .. import envpolicy
from ..tools.policy import PathPolicy

__all__ = ["RUNNER", "MARKER", "SandboxResult", "bwrap_works", "sandbox_plan", "run_sandboxed", "runner_argv"]

RUNNER = Path(__file__).resolve().with_name("runner.py")
MARKER = "@@VBT_RESULT@@"
MAX_OUTPUT = 400_000
_BWRAP_OK: dict[str, bool] = {}


@dataclass
class SandboxResult:
    exit_code: int | None
    output: str                       # stdout + stderr without the result line (head and tail when long)
    result: Any = None                # the runner's result object (None when it printed none)
    timed_out: bool = False
    duration_s: float = 0.0
    sandbox: str = "none"
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def bwrap_works(exe: str | None = None) -> bool:
    """bubblewrap is installed and can create its namespaces here (probed once per binary)."""
    exe = exe or shutil.which("bwrap")
    if not exe:
        return False
    if exe not in _BWRAP_OK:
        try:
            r = subprocess.run([exe, "--die-with-parent", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
                                "--unshare-net", "--", "/bin/true"], capture_output=True, timeout=20)
            _BWRAP_OK[exe] = r.returncode == 0
        except (OSError, subprocess.SubprocessError):
            _BWRAP_OK[exe] = False
    return _BWRAP_OK[exe]


def sandbox_plan(config: Mapping[str, Any], *, network: bool) -> tuple[str, str | None, list[str]]:
    """``(label, bwrap binary or None, notes)`` for ``projects.sandbox`` on this host. ``bwrap`` mode without a
    working bubblewrap raises RuntimeError (it fails closed, as ``bash.sandbox.os: bwrap`` does)."""
    from ..tools.builtin import _unshare_available

    mode = str(((config.get("projects") or {}).get("sandbox")) or "auto")
    notes: list[str] = []
    exe = shutil.which("bwrap")
    use_bwrap = False
    if mode == "bwrap":
        if not bwrap_works(exe):
            raise RuntimeError("projects.sandbox is 'bwrap' but bubblewrap is not installed or cannot create its "
                               "namespaces here; install it or set projects.sandbox: auto")
        use_bwrap = True
    elif mode == "auto":
        use_bwrap = bwrap_works(exe)
        if not use_bwrap:
            notes.append("bubblewrap is not available: the filesystem is not isolated (memory limit and network "
                         "isolation still apply)")
    label = "bwrap" if use_bwrap else "rlimit"
    if not network:
        if use_bwrap or _unshare_available():
            label += "+netns"
        else:
            notes.append("network isolation is unavailable here (no bwrap, no unshare -rn)")
    return label, (exe if use_bwrap else None), notes


def runner_argv(python: str | None, *args: str) -> list[str]:
    return [python or sys.executable, "-I", str(RUNNER), *args]


def _cap(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n[... {len(text) - limit:,} chars omitted ...]\n" + text[-half:]


def _split_result(text: str) -> tuple[str, Any]:
    """``(output without the result line, the result object)``: the last ``@@VBT_RESULT@@`` line wins."""
    lines = text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith(MARKER + " "):
            try:
                obj = json.loads(lines[i][len(MARKER) + 1:])
            except ValueError:
                return text, None
            return "\n".join(lines[:i] + lines[i + 1:]), obj
    return text, None


async def run_sandboxed(argv: list[str], *, policy: PathPolicy, cwd: Path, config: Mapping[str, Any],
                        label: str, timeout_s: float, network: bool, stdin: bytes | None = None,
                        extra_env: Mapping[str, str] | None = None) -> SandboxResult:
    """Run ``argv`` (the harness's own command line) contained as described in the module docstring.

    ``policy`` is the calling agent's path policy: under bwrap its own work directory and the run's
    ``.tmp``/``.home`` are the only writable places and its blocked paths are hidden. ``label`` names the call
    in the reaper's status file."""
    from ..tools.builtin import _kill_group, _unshare_available, _workspace_limit, bwrap_argv

    kind, exe, notes = sandbox_plan(config, network=network)
    bash_cfg = dict(config.get("bash") or {})
    run_dir = Path(policy.run_dir)
    cwd.mkdir(parents=True, exist_ok=True)
    extra = {**{k: str(v) for k, v in (config.get("tool_env") or {}).items() if v},
             "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg", "PWD": str(cwd), **dict(extra_env or {})}
    env = envpolicy.child_env(os.environ, passthrough=bash_cfg.get("env_passthrough") or [], extra=extra,
                              home=run_dir / ".home", tmp=run_dir / ".tmp")
    if exe:
        cmd = bwrap_argv(policy, list(argv), cwd, bwrap=exe, unshare_net=not network)
    elif not network and _unshare_available():
        cmd = ["unshare", "-rn", "--", *argv]
    else:
        cmd = list(argv)
    limit_mb, launched, _status = _workspace_limit(dict(config), cmd, run_dir, label)
    if launched:
        cmd = launched
        kind += f"+{limit_mb}MB"
    t0 = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=str(cwd), env=env, stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
    timed_out = False
    try:
        out, _ = await asyncio.wait_for(proc.communicate(stdin), timeout=max(float(timeout_s), 1.0))
    except asyncio.TimeoutError:
        timed_out = True
        await _kill_group(proc)
        out = b""
    except BaseException:
        await _kill_group(proc)
        raise
    text = out.decode("utf-8", errors="replace")
    secrets = dict(os.environ)
    text = envpolicy.redact(text, secrets)
    rest, result = _split_result(text)
    from ..tools.builtin import _strip_exit_marker

    rest, reason = _strip_exit_marker(rest)
    if reason == "memory_limit":
        notes.append(f"the process reached the workspace memory limit ({limit_mb} MB)")
    return SandboxResult(exit_code=proc.returncode if not timed_out else None, output=_cap(rest.strip()),
                         result=result, timed_out=timed_out, duration_s=round(time.monotonic() - t0, 3),
                         sandbox=kind, notes=notes)
