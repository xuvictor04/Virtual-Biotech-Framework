#!/usr/bin/env python3
"""Stdlib-only launcher for MCP server children (§14.2). Never imports ``vbt``.

Usage::

    python -E reaper.py --limit-mb N --status PATH --server NAME [--strip-python-flags] -- cmd args...

The reaper forks. The **child** puts itself in its own process group, asks the kernel to
kill it when the reaper dies, sets ``RLIMIT_DATA`` to N MB when N > 0 (Linux >= 4.7 counts
brk and private writable mappings, so pandas and Arrow heaps are bounded without the false
failures ``RLIMIT_AS`` causes) and execs the command, inheriting fds 0, 1 and 2.

The **parent** points its own fds 0 and 1 at ``/dev/null`` (so pipe EOF reaches the bridge as
soon as the child dies), forwards SIGTERM, SIGINT and SIGHUP to the child's process group,
polls ``/proc/<pid>/status`` every 250 ms, writes the status JSON ``{pid, rss_mb, peak_rss_mb,
limit_mb, containment, server, hash_seed, flags_stripped, ts}`` atomically every second, and
after ``waitpid`` writes one line ``VBT_CHILD_EXIT {"pid", "code", "signal", "maxrss_kb",
"reason"}`` to stderr (the server log), exiting with the child's code or 128 + signal.

Hash seed (rev 2): every server runs as ``python -E``, and ``-E`` makes CPython ignore
``PYTHONHASHSEED``. For a Python child (interpreter basename ``python*``, or
``--strip-python-flags``) the reaper removes ``-E`` and ``-I`` from the child's argv and builds
its environment explicitly instead: every ``PYTHON*`` variable is dropped and only
``PYTHONHASHSEED=0``, ``PYTHONDONTWRITEBYTECODE=1``, ``PYTHONNOUSERSITE=1`` and
``PYTHONSAFEPATH=1`` (when ``-I`` was given) and ``PYTHONUSERBASE`` (when the launcher's
environment has it) are added back. ``PYTHONPATH`` and ``PYTHONHOME`` therefore still cannot
redirect imports, while the seed takes effect.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import resource
import signal
import sys
import time
from typing import Any

EXIT_MARKER = "VBT_CHILD_EXIT"
HASH_SEED = "0"
MB = 1024 * 1024

WAIT_POLL_S = 0.05            # waitpid granularity (the exit marker follows a death within this)
RSS_POLL_S = 0.25             # /proc/<pid>/status
STATUS_EVERY_S = 1.0          # status file rewrite
MEMORY_LIMIT_FRACTION = 0.9   # an abnormal exit at this share of the limit is a memory exit
FORWARDED = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)

_VALUE_FLAGS = frozenset("WX")      # -W arg, -X arg (attached or the next argv element)
_END_FLAGS = frozenset("cm")        # -c cmd, -m mod: option parsing ends here
_LONG_VALUE_FLAGS = frozenset({"--check-hash-based-pycs"})
_PR_SET_PDEATHSIG = 1


def is_python(command: str) -> bool:
    """True when ``command`` names a CPython interpreter (basename ``python*``)."""
    return os.path.basename(command).startswith("python")


def strip_python_flags(args: list[str]) -> tuple[list[str], list[str]]:
    """Remove ``-E`` and ``-I`` from interpreter options (also inside clusters such as ``-IB``).

    ``args`` excludes the interpreter itself. Options end at the script, ``-c``, ``-m``, ``-``
    or ``--``; everything after is passed through unchanged. Returns ``(args, stripped)`` with
    ``stripped`` the removed flags in first-seen order, without duplicates.
    """
    out: list[str] = []
    stripped: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--" or a == "-" or not a.startswith("-"):
            return out + args[i:], stripped
        if a.startswith("--"):
            out.append(a)
            if a in _LONG_VALUE_FLAGS and i + 1 < len(args):
                out.append(args[i + 1])
                i += 1
            i += 1
            continue
        letters, kept = a[1:], ""
        takes_value = ends = False
        for j, ch in enumerate(letters):
            if ch in "EI":
                if f"-{ch}" not in stripped:
                    stripped.append(f"-{ch}")
                continue
            if ch in _VALUE_FLAGS or ch in _END_FLAGS:
                kept += letters[j:]
                takes_value = ch in _VALUE_FLAGS and j == len(letters) - 1
                ends = ch in _END_FLAGS
                break
            kept += ch
        if kept:
            out.append("-" + kept)
        if ends:
            return out + args[i + 1:], stripped
        if takes_value and i + 1 < len(args):
            out.append(args[i + 1])
            i += 1
        i += 1
    return out, stripped


def child_environment(environ: dict[str, str], *, isolated: bool) -> dict[str, str]:
    """The explicit environment of a Python child: no ``PYTHON*`` variable except the allow-list."""
    env = {k: v for k, v in environ.items() if not k.startswith("PYTHON")}
    env["PYTHONHASHSEED"] = HASH_SEED
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if isolated:
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONSAFEPATH"] = "1"      # -I also kept the script directory off sys.path
    if environ.get("PYTHONUSERBASE"):
        env["PYTHONUSERBASE"] = environ["PYTHONUSERBASE"]
    return env


def read_proc_status(pid: int) -> tuple[float | None, float | None]:
    """``(VmRSS, VmHWM)`` of ``pid`` in MB, or ``(None, None)`` when unreadable."""
    rss = hwm = None
    try:
        with open(f"/proc/{pid}/status", encoding="ascii", errors="replace") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / 1024.0
                elif line.startswith("VmHWM:"):
                    hwm = int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        return None, None
    return rss, hwm


def write_status(path: str | None, data: dict[str, Any]) -> None:
    """Write ``data`` as JSON to ``path`` atomically (best effort)."""
    if not path:
        return
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    if "--" not in argv:
        raise SystemExit("reaper: usage: reaper.py --limit-mb N --status PATH --server NAME -- cmd args...")
    cut = argv.index("--")
    p = argparse.ArgumentParser(prog="reaper.py", add_help=False)
    p.add_argument("--limit-mb", type=int, default=0)
    p.add_argument("--status", default=None)
    p.add_argument("--server", default="")
    p.add_argument("--strip-python-flags", action="store_true")
    ns = p.parse_args(argv[:cut])
    command = argv[cut + 1:]
    if not command:
        raise SystemExit("reaper: no command after --")
    return ns, command


def _set_pdeathsig(parent: int) -> None:
    """Have the kernel SIGKILL this (child) process when the reaper dies (Linux)."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0)
    except (OSError, AttributeError):
        return
    if os.getppid() != parent:  # the reaper died before prctl took effect
        os._exit(1)


def _exec_child(command: list[str], env: dict[str, str], limit_mb: int, parent: int) -> None:
    """In the forked child: process group, death signal, RLIMIT_DATA, exec. Never returns."""
    try:
        os.setpgid(0, 0)
        _set_pdeathsig(parent)
        if limit_mb > 0:
            want = limit_mb * MB
            _, hard = resource.getrlimit(resource.RLIMIT_DATA)
            if hard != resource.RLIM_INFINITY:
                want = min(want, hard)
            resource.setrlimit(resource.RLIMIT_DATA, (want, want))
        os.execvpe(command[0], command, env)
    except BaseException as exc:  # noqa: BLE001 - report and leave with "command not runnable"
        try:
            os.write(2, f"reaper: cannot start {command[0]!r}: {exc}\n".encode("utf-8", "replace"))
        finally:
            os._exit(127)


def main(argv: list[str] | None = None) -> int:
    ns, command = parse_args(list(sys.argv[1:] if argv is None else argv))
    python = ns.strip_python_flags or is_python(command[0])
    stripped: list[str] = []
    env = dict(os.environ)
    if python:
        rest, stripped = strip_python_flags(command[1:])
        command = [command[0], *rest]
        env = child_environment(env, isolated="-I" in stripped)
    limit = max(0, int(ns.limit_mb))

    parent = os.getpid()
    pid = os.fork()
    if pid == 0:
        _exec_child(command, env, limit, parent)

    # ---- parent (the reaper)
    try:
        os.setpgid(pid, pid)       # also done by the child; whichever runs first wins the race
    except OSError:
        pass
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)
    os.close(devnull)

    def forward(signum: int, _frame: Any) -> None:
        try:
            os.killpg(pid, signum)
        except OSError:
            try:
                os.kill(pid, signum)
            except OSError:
                pass

    for sig in FORWARDED:
        signal.signal(sig, forward)

    status: dict[str, Any] = {
        "pid": pid, "server": ns.server, "limit_mb": limit,
        "containment": "rlimit_data" if limit > 0 else "none",
        "hash_seed": int(HASH_SEED) if python else None, "flags_stripped": stripped,
        "rss_mb": None, "peak_rss_mb": None, "ts": round(time.time(), 3),
    }
    write_status(ns.status, status)
    peak = 0.0
    last_rss = last_write = time.monotonic()
    while True:
        try:
            wpid, wstatus, usage = os.wait4(pid, os.WNOHANG)
        except ChildProcessError:  # pragma: no cover - reaped elsewhere
            wpid, wstatus, usage = pid, 0, None
        if wpid == pid:
            break
        now = time.monotonic()
        if now - last_rss >= RSS_POLL_S:
            last_rss = now
            rss, hwm = read_proc_status(pid)
            if rss is not None:
                peak = max(peak, rss, hwm or 0.0)
                status["rss_mb"], status["peak_rss_mb"] = round(rss, 1), round(peak, 1)
        if now - last_write >= STATUS_EVERY_S:
            last_write = now
            status["ts"] = round(time.time(), 3)
            write_status(ns.status, status)
        time.sleep(WAIT_POLL_S)

    if os.WIFSIGNALED(wstatus):
        code, signum = None, os.WTERMSIG(wstatus)
        rc = 128 + signum
    else:
        code, signum = os.WEXITSTATUS(wstatus), None
        rc = code
    maxrss_kb = int(getattr(usage, "ru_maxrss", 0) or 0)
    peak = max(peak, maxrss_kb / 1024.0)
    reason = "signal" if signum is not None else "exit_code"
    if limit > 0 and rc != 0 and peak >= MEMORY_LIMIT_FRACTION * limit:
        reason = "memory_limit"
    marker = {"pid": pid, "code": code, "signal": signum, "maxrss_kb": maxrss_kb, "reason": reason}
    status.update({"rss_mb": None, "peak_rss_mb": round(peak, 1), "ts": round(time.time(), 3), "exit": marker})
    write_status(ns.status, status)
    try:
        sys.stderr.write(f"{EXIT_MARKER} {json.dumps(marker, sort_keys=True)}\n")
        sys.stderr.flush()
    except (OSError, ValueError):
        pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
