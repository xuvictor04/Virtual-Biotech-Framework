#!/usr/bin/env python3
"""Stdlib-only launcher for MCP server children (§14.2). Never imports ``vbt``.

Usage::

    python -E reaper.py --limit-mb N --status PATH --server NAME [--strip-python-flags]
                        [--containment rlimit_data|cgroup|watchdog|auto|rss|none] [--relay-max-mb M] -- cmd args...

The reaper forks. The **child** puts itself in its own process group, asks the kernel to
kill it when the reaper dies, sets ``RLIMIT_DATA`` to N MB when N > 0 (Linux >= 4.7 counts
brk and private writable mappings, so pandas and Arrow heaps are bounded without the false
failures ``RLIMIT_AS`` causes) and execs the command, inheriting fd 0 (and 1 unless the relay is
on); its fd 2 is a pipe to the reaper. When the reaper's own fds 1 and 2 are one pipe or file (a caller that
merges the streams: the Bash tool runs commands with ``stderr=STDOUT``), the child's fds 1 and 2 are both that
pipe to the reaper, which copies it in order to the merged stream: a separate stderr pipe would deliver stderr
late and out of place, and its 64 KiB copies could land inside a stdout line.

The **parent** points its own fds 0 and 1 at ``/dev/null`` (so pipe EOF reaches the bridge as
soon as the child dies), forwards SIGTERM, SIGINT and SIGHUP to the child's process group, dies with
its own launcher (when ``getppid()`` changes, the launcher is gone: the child's group gets SIGTERM, and
SIGKILL :data:`ORPHAN_GRACE_S` later; the exit marker then says ``"orphaned": true``. A killed bridge
left a ``vbt ds check`` data child, which reads no stdin, checking for minutes under a reaper
re-parented to init),
tees the child's stderr to its own (the server log) as it arrives, keeping only the last 64 KiB,
polls ``/proc/<pid>/status`` every 250 ms, writes the status JSON ``{pid, rss_mb, peak_rss_mb,
limit_mb, containment, server, hash_seed, flags_stripped, ts}`` atomically every second, and
after ``waitpid`` (and draining the stderr pipe for at most a second) writes one line
``VBT_CHILD_EXIT {"pid", "code", "signal", "maxrss_kb", "reason", "cause"?}`` to stderr, exiting
with the child's code or 128 + signal.

Memory exits (INV-1): ``reason`` is ``memory_limit`` when the cause is known, and ``cause`` says which:
``watchdog`` (the RSS watchdog's SIGKILL), ``cgroup_oom_kill`` (the child cgroup's ``oom_kill`` count
moved), ``memory_error`` (an abnormal exit whose last stderr line carries a memory signature: an
uncaught ``MemoryError`` or ``std::bad_alloc`` under ``RLIMIT_DATA``, which fails the allocation
instead of killing, or, under a data limit, a thread that could not be created, since thread stacks count
against ``RLIMIT_DATA``; ``memory_error`` holds that line), ``kernel_oom_kill`` (a SIGKILL the kernel log
``/dev/kmsg`` records as an OOM kill of the child, e.g. by an enclosing cgroup) or, only when none of these
applies, ``peak_rss`` (an abnormal exit at 90% of the limit). Any other exit is ``signal`` or
``exit_code`` with no ``cause``. Without a readable kernel log (``dmesg_restrict`` without ``CAP_SYSLOG``), a
SIGKILL while the host's ``/proc/vmstat`` ``oom_kill`` count moved is not attributed to the child (an OOM kill
anywhere on the host moves it): the marker says ``"possible_kernel_oom": true`` and the reason stays ``signal``.

Containment (phase 4, F19; ``--containment``, default ``rlimit_data`` or ``$VBT_REAPER_CONTAINMENT``):

* ``rlimit_data``: ``RLIMIT_DATA`` only (phases 1-3).
* ``cgroup`` / ``auto``: a memory cgroup for the child, in order: a delegated cgroup v2 subtree
  (the reaper's own cgroup is writable: a child cgroup gets ``memory.max``, ``memory.swap.max = 0``
  and ``memory.oom.group = 1``, and its ``memory.events`` ``oom_kill`` count tells a memory kill
  apart); a writable cgroup v1 ``memory`` hierarchy (``memory.limit_in_bytes``, ``memory.failcnt``);
  a ``systemd-run --user --scope`` with ``MemoryMax`` (cgroup v2 hosts with a user manager). Without
  any (no writable memory controller, e.g. an unprivileged user), the **RSS watchdog** contains the child.
* ``watchdog``: ``RLIMIT_DATA`` plus a watchdog that SIGKILLs the child's process group when the
  group's RSS reaches ``limit - max(512 MB, 5%)`` (at least half the limit), and reports
  ``reason: memory_limit``, which the bridge classifies ``oom`` and never retries.
* ``rss``: resident memory only, no ``RLIMIT_DATA``: the memory cgroup as ``cgroup`` does, else the RSS
  watchdog. For children whose libraries reserve address space far beyond what they touch (TileDB read
  buffers in Census pulls fail with ``std::bad_alloc`` under any tested ``RLIMIT_DATA`` while RSS stays
  under 3 GB; Arrow thread stacks count against it too).
``RLIMIT_DATA`` stays set under every other mode with a limit. ``VBT_REAPER_CGROUP_ROOT`` (v2) and
``VBT_REAPER_CGROUP_V1_ROOT`` (v1) point the cgroup code at another mount (tests).

Relay (optional, off by default; ``--relay-max-mb`` or ``$VBT_REAPER_RELAY_MAX_MB``): the child's
stdout goes through the reaper line by line (MCP stdio frames are newline-delimited JSON-RPC). A
message over the cap is never held whole: the reaper keeps its first and last 64 KiB, discards the
rest, and writes in its place an error response carrying the **same JSON-RPC id** (``code -32001``),
so the harness never buffers a multi-GB reply and the waiting request still completes.

Hash seed (rev 2): every server runs as ``python -E``, and ``-E`` makes CPython ignore
``PYTHONHASHSEED``. For a Python child (interpreter basename ``python*``, or
``--strip-python-flags``) the reaper removes ``-E`` and ``-I`` from the child's argv and builds
its environment explicitly instead: every ``PYTHON*`` variable is dropped and only
``PYTHONHASHSEED=0``, ``PYTHONDONTWRITEBYTECODE=1``, ``PYTHONNOUSERSITE=1`` and
``PYTHONSAFEPATH=1`` (when ``-I`` was given) and ``PYTHONUSERBASE`` (when the launcher's
environment has it) are added back. ``PYTHONPATH`` and ``PYTHONHOME`` therefore still cannot
redirect imports, while the seed takes effect. This holds under every containment mode.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import re
import resource
import select
import shutil
import signal
import stat
import sys
import threading
import time
from typing import Any

EXIT_MARKER = "VBT_CHILD_EXIT"
HASH_SEED = "0"
MB = 1024 * 1024

WAIT_POLL_S = 0.05            # waitpid granularity (the exit marker follows a death within this)
ORPHAN_GRACE_S = 5.0          # after the launcher dies: SIGTERM to the child's group, SIGKILL this much later
RSS_POLL_S = 0.25             # /proc/<pid>/status
STATUS_EVERY_S = 1.0          # status file rewrite
MEMORY_LIMIT_FRACTION = 0.9   # an abnormal exit at this share of the limit is a memory exit
FORWARDED = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)

CONTAINMENTS = ("rlimit_data", "cgroup", "watchdog", "auto", "rss", "none")
CONTAINMENT_ENV = "VBT_REAPER_CONTAINMENT"
RELAY_ENV = "VBT_REAPER_RELAY_MAX_MB"
CGROUP_ROOT_ENV = "VBT_REAPER_CGROUP_ROOT"
CGROUP_V1_ROOT_ENV = "VBT_REAPER_CGROUP_V1_ROOT"
WATCHDOG_MARGIN_MB = 512      # the watchdog kills at limit - max(512 MB, 5%)
WATCHDOG_MARGIN_FRACTION = 0.05
RELAY_KEEP = 64 * 1024        # bytes of an oversized message kept to find its id
RELAY_ERROR_CODE = -32001
TEE_KEEP = 64 * 1024          # bytes of the child's stderr kept (its tail) to name the cause of an exit
TEE_DRAIN_S = 1.0             # after the child's exit, the stderr pipe is drained for at most this long
TEE_IDLE_S = 0.05             # ... and the drain ends once the pipe has been quiet this long
KMSG_CLOCK_MARGIN_US = 5_000_000   # the kernel log's clock runs within ~0.3 s of CLOCK_MONOTONIC

#: Error text of a failed allocation (Python, numpy, Arrow, C++, Rust, errno ENOMEM). The same patterns as
#: ``vbt.datalayer.memory.crash.MEMORY_PATTERNS`` (a test keeps them equal; this file never imports vbt).
MEMORY_PATTERNS: tuple[str, ...] = (
    r"MemoryError", r"Unable to allocate", r"bad_alloc", r"Cannot allocate memory", r"\bENOMEM\b",
    r"(?i:\bmemory allocation\b.{0,40}\bfailed\b)", r"\b(?:m|re|c)alloc of size \d+ failed", r"(?i:\bout of memory\b)",
)
MEMORY_RE = re.compile("|".join(MEMORY_PATTERNS))
#: Under a data limit only: thread creation that fails because thread stacks count against RLIMIT_DATA
#: (Arrow's pool on a real Open Targets load: ``std::system_error`` / ``Resource temporarily unavailable``).
LIMIT_PATTERNS: tuple[str, ...] = (r"what\(\):\s+Resource temporarily unavailable", r"can't start new thread",
                                   r"pthread_create")
LIMIT_RE = re.compile("|".join(LIMIT_PATTERNS))

_VALUE_FLAGS = frozenset("WX")      # -W arg, -X arg (attached or the next argv element)
_END_FLAGS = frozenset("cm")        # -c cmd, -m mod: option parsing ends here
_LONG_VALUE_FLAGS = frozenset({"--check-hash-based-pycs"})
_PR_SET_PDEATHSIG = 1
_ID_RE = re.compile(rb'"id"\s*:\s*(-?\d+|"(?:[^"\\]|\\.)*"|null)')


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


def group_rss_mb(pgid: int) -> float | None:
    """Summed VmRSS (MB) of every process in process group ``pgid``; None when none is readable."""
    total, seen = 0.0, False
    try:
        names = os.listdir("/proc")
    except OSError:
        return None
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", encoding="ascii", errors="replace") as f:
                fields = f.read().rsplit(")", 1)[-1].split()
            if int(fields[2]) != pgid:           # fields after ")": state ppid pgrp ...
                continue
        except (OSError, ValueError, IndexError):
            continue
        rss, _ = read_proc_status(int(name))
        if rss is not None:
            total += rss
            seen = True
    return total if seen else None


def memory_signature(text: str, *, limited: bool = False) -> str | None:
    """The last non-empty line of ``text`` when it carries a memory signature (the error that ended the
    process; a memory error a server survived is followed by more output), else None. ``limited`` (a data
    limit is set) also counts a thread that could not be created (:data:`LIMIT_PATTERNS`)."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return None
    hit = MEMORY_RE.search(lines[-1]) or (limited and LIMIT_RE.search(lines[-1]))
    return lines[-1][:300] if hit else None


def host_oom_kills() -> int | None:
    """The host's kernel OOM kills so far (``/proc/vmstat`` ``oom_kill``, Linux >= 4.13), or None."""
    for line in (_read("/proc/vmstat") or "").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "oom_kill":
            try:
                return int(parts[1])
            except ValueError:
                return None
    return None


def kernel_oom_killed(pid: int, since_us: int = 0, *, path: str = "/dev/kmsg",
                      max_records: int = 200000) -> bool | None:
    """True when the kernel log records an OOM kill of ``pid`` (``Killed process <pid> (``) at or after
    ``since_us`` (microseconds of the log's monotonic clock), False when the log is readable and has none,
    None when it cannot be read (no permission, no ``/dev/kmsg``)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return None
    needle = f"Killed process {pid} (".encode()
    found = False
    try:
        for _ in range(max_records):
            try:
                rec = os.read(fd, 8192)
            except BlockingIOError:
                break
            except InterruptedError:
                continue
            except OSError as exc:
                if exc.errno == errno.EPIPE:          # a record was overwritten while reading: go on
                    continue
                return None
            if not rec:
                break
            if needle not in rec:
                continue
            try:                                     # "prio,seq,ts_usec,flags;message"
                ts = int(rec.split(b";", 1)[0].split(b",")[2])
            except (IndexError, ValueError):
                ts = since_us
            if ts >= since_us:
                found = True
    finally:
        os.close(fd)
    return found


def exit_cause(*, abnormal: bool, signum: int | None, limit_mb: int, peak_mb: float, watchdog: bool = False,
               cgroup_oom: bool = False, stderr_tail: str = "", kernel_oom: bool = False,
               data_limited: bool | None = None) -> tuple[str | None, str | None]:
    """``(cause, memory_error line)`` of a child's exit; cause None when it was not a memory exit.
    ``data_limited``: the child ran under ``RLIMIT_DATA`` (default: whenever ``limit_mb`` is set; not under
    the ``rss`` containment)."""
    if watchdog:
        return "watchdog", None
    if cgroup_oom:
        return "cgroup_oom_kill", None
    if not abnormal:
        return None, None
    line = memory_signature(stderr_tail, limited=limit_mb > 0 if data_limited is None else data_limited)
    if line is not None:
        return "memory_error", line
    if signum == signal.SIGKILL and kernel_oom:
        return "kernel_oom_kill", None
    if limit_mb > 0 and peak_mb >= MEMORY_LIMIT_FRACTION * limit_mb:
        return "peak_rss", None
    return None, None


def kill_attribution(logged: bool | None, host_before: int | None, host_after: int | None) -> tuple[bool, bool]:
    """``(kernel_oom, possible_kernel_oom)`` of a SIGKILL nobody in the reaper sent. ``logged``: the kernel log
    names the child's OOM kill (None: no readable log). Without the log the host's ``oom_kill`` count moving says
    only that some process was OOM-killed meanwhile (another agent's, in its own cgroup, moves it too): possible,
    never the child's cause."""
    if logged is not None:
        return bool(logged), False
    moved = host_before is not None and host_after is not None and host_after > host_before
    return False, moved


def watchdog_threshold_mb(limit_mb: int) -> float:
    """The group RSS at which the watchdog kills: ``limit - max(512 MB, 5%)``, at least half the limit."""
    margin = max(WATCHDOG_MARGIN_MB, WATCHDOG_MARGIN_FRACTION * limit_mb)
    return max(limit_mb - margin, 0.5 * limit_mb)


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


# --------------------------------------------------------------------------- cgroups


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="ascii", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _write(path: str, value: str) -> bool:
    try:
        with open(path, "w", encoding="ascii") as f:
            f.write(value)
        return True
    except OSError:
        return False


def _own_cgroup(version: int) -> str | None:
    """The reaper's cgroup path (v2: the ``0::`` line; v1: the ``memory`` controller line)."""
    text = _read("/proc/self/cgroup") or ""
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        if version == 2 and parts[0] == "0" and parts[1] == "":
            return parts[2] or "/"
        if version == 1 and "memory" in parts[1].split(","):
            return parts[2] or "/"
    return None


def _cgroup2_mounted(root: str) -> bool:
    if os.environ.get(CGROUP_ROOT_ENV):
        return os.path.isdir(root)
    for line in (_read("/proc/mounts") or "").splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[1] == root and fields[2] == "cgroup2":
            return True
    return False


class Cgroup:
    """A memory cgroup made for one child: ``version`` 2 or 1, ``path`` its directory."""

    def __init__(self, version: int, path: str) -> None:
        self.version = version
        self.path = path

    def add(self, pid: int) -> bool:
        name = "cgroup.procs" if self.version == 2 else "tasks"
        ok = _write(os.path.join(self.path, name), str(pid))
        if not ok and self.version == 1:
            ok = _write(os.path.join(self.path, "cgroup.procs"), str(pid))
        return ok

    def oom_kills(self) -> int:
        """Memory kills inside the cgroup so far (v2 ``memory.events`` ``oom_kill``; v1 ``failcnt``
        is only a hint, so v1 reads ``memory.oom_control`` ``oom_kill`` when the kernel has it)."""
        if self.version == 2:
            text = _read(os.path.join(self.path, "memory.events")) or ""
        else:
            text = _read(os.path.join(self.path, "memory.oom_control")) or ""
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == "oom_kill":
                try:
                    return int(parts[1])
                except ValueError:
                    return 0
        return 0

    def remove(self) -> None:
        try:
            os.rmdir(self.path)
        except OSError:
            pass


def make_cgroup_v2(server: str, limit_mb: int) -> Cgroup | None:
    """A child cgroup under the reaper's own (delegated) v2 cgroup, or None without delegation."""
    root = os.environ.get(CGROUP_ROOT_ENV) or "/sys/fs/cgroup"
    if not _cgroup2_mounted(root):
        return None
    own = "/" if os.environ.get(CGROUP_ROOT_ENV) else _own_cgroup(2)
    if own is None:
        return None
    parent = os.path.join(root, own.lstrip("/"))
    if "memory" not in (_read(os.path.join(parent, "cgroup.subtree_control")) or "").split() and \
            "memory" not in (_read(os.path.join(parent, "cgroup.controllers")) or "").split():
        return None
    if not os.access(parent, os.W_OK):
        return None
    path = os.path.join(parent, f"vbt-{_safe(server)}-{os.getpid()}")
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        return None
    if not _write(os.path.join(path, "memory.max"), str(limit_mb * MB)):
        Cgroup(2, path).remove()
        return None
    _write(os.path.join(path, "memory.swap.max"), "0")
    _write(os.path.join(path, "memory.oom.group"), "1")
    return Cgroup(2, path)


def make_cgroup_v1(server: str, limit_mb: int) -> Cgroup | None:
    """A child cgroup in a writable v1 ``memory`` hierarchy, or None."""
    root = os.environ.get(CGROUP_V1_ROOT_ENV) or "/sys/fs/cgroup/memory"
    if not os.path.isfile(os.path.join(root, "memory.limit_in_bytes")) and not os.environ.get(CGROUP_V1_ROOT_ENV):
        return None
    own = "/" if os.environ.get(CGROUP_V1_ROOT_ENV) else (_own_cgroup(1) or "/")
    parent = os.path.join(root, own.lstrip("/"))
    if not os.path.isdir(parent) or not os.access(parent, os.W_OK):
        return None
    path = os.path.join(parent, f"vbt-{_safe(server)}-{os.getpid()}")
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        return None
    if not _write(os.path.join(path, "memory.limit_in_bytes"), str(limit_mb * MB)):
        Cgroup(1, path).remove()
        return None
    _write(os.path.join(path, "memory.memsw.limit_in_bytes"), str(limit_mb * MB))
    return Cgroup(1, path)


def systemd_scope_prefix(server: str, limit_mb: int) -> list[str] | None:
    """``systemd-run --user --scope`` with ``MemoryMax`` when a user manager is reachable, else None."""
    exe = shutil.which("systemd-run")
    if not exe or not os.environ.get("XDG_RUNTIME_DIR") or \
            not os.path.exists(os.path.join(os.environ["XDG_RUNTIME_DIR"], "systemd", "private")):
        return None
    return [exe, "--user", "--scope", "--quiet", f"--unit=vbt-{_safe(server)}-{os.getpid()}",
            "-p", f"MemoryMax={limit_mb}M", "-p", "MemorySwapMax=0", "--"]


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name or "server")[:64]


# --------------------------------------------------------------------------- stderr tee


def merged_output(out_fd: int = 1, err_fd: int = 2) -> bool:
    """The caller merges stdout and stderr: both fds are one pipe or one regular file (``stderr=STDOUT``).
    A terminal is left alone (the child keeps a tty stdout; a late stderr line there is cosmetic)."""
    try:
        a, b = os.fstat(out_fd), os.fstat(err_fd)
    except OSError:
        return False
    if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
        return False
    return stat.S_ISFIFO(a.st_mode) or stat.S_ISREG(a.st_mode) or stat.S_ISSOCK(a.st_mode)


class StderrTee:
    """Copies the child's stderr (the read end of a pipe) to the reaper's own fd 2 (the server log) as it
    arrives and keeps only its last ``keep`` bytes, so the exit can be labelled from what the child said
    last. Memory is bounded (at most twice ``keep``); a log that cannot be written is still drained."""

    def __init__(self, src_fd: int, dst_fd: int = 2, keep: int = TEE_KEEP) -> None:
        self.src, self.dst, self.keep = src_fd, dst_fd, keep
        self.tail = bytearray()
        self.total = 0
        self.eof = False
        self._exited = threading.Event()
        self._thread = threading.Thread(target=self._run, name="vbt-stderr-tee", daemon=True)

    def start(self) -> "StderrTee":
        self._thread.start()
        return self

    def _write(self, data: bytes) -> None:
        view = memoryview(data)
        while view:
            try:
                n = os.write(self.dst, view)
            except InterruptedError:
                continue
            except OSError:
                return
            view = view[n:]

    def _run(self) -> None:
        while True:
            try:
                ready, _, _ = select.select([self.src], [], [], TEE_IDLE_S)
            except InterruptedError:
                continue
            except (OSError, ValueError):
                return
            if not ready:
                if self._exited.is_set():
                    return                         # the child is gone and the pipe is quiet: drained
                continue
            try:
                data = os.read(self.src, 1 << 16)
            except InterruptedError:
                continue
            except OSError:
                data = b""
            if not data:
                self.eof = True
                return
            self.total += len(data)
            self._write(data)
            self.tail += data
            if len(self.tail) > 2 * self.keep:
                del self.tail[:len(self.tail) - self.keep]

    def finish(self, timeout: float = TEE_DRAIN_S) -> str:
        """Drain what the dead child wrote (a grandchild holding the pipe open ends the wait after
        ``timeout``) and return the kept tail as text."""
        self._exited.set()
        self._thread.join(timeout)
        return bytes(self.tail[-self.keep:]).decode("utf-8", "replace")


# --------------------------------------------------------------------------- relay


def _message_id(head: bytes, tail: bytes) -> Any:
    """The JSON-RPC id of a message from its first and last bytes (``_NO_ID`` when absent)."""
    for chunk in (head, tail):
        m = _ID_RE.search(chunk)
        if m:
            try:
                return json.loads(m.group(1).decode("utf-8", "replace"))
            except ValueError:
                continue
    return _NO_ID


_NO_ID: Any = object()


def oversized_error(msg_id: Any, size: int, cap_bytes: int) -> bytes:
    """The error response written in place of an oversized message."""
    body = {"jsonrpc": "2.0", "id": msg_id, "error": {
        "code": RELAY_ERROR_CODE,
        "message": f"the server's response is {size / MB:.1f} MB, over the relay cap of {cap_bytes / MB:.0f} MB "
                   "(data.memory.relay_max_message_mb); it was discarded. Narrow the request.",
        "data": {"bytes": size, "cap_bytes": cap_bytes}}}
    return (json.dumps(body, sort_keys=True) + "\n").encode("utf-8")


def relay(src_fd: int, dst_fd: int, cap_bytes: int, *, chunk: int = 1 << 16,
          note: Any = None) -> dict[str, int]:
    """Copy newline-delimited messages from ``src_fd`` to ``dst_fd``; a message over ``cap_bytes`` is
    replaced by :func:`oversized_error` with its id (a message without an id is dropped). Returns
    counts ``{messages, replaced, dropped}``. Never holds more than ``cap_bytes + chunk`` of a message."""
    counts = {"messages": 0, "replaced": 0, "dropped": 0}
    buf = bytearray()
    over = False
    size = 0
    head = b""
    tail = bytearray()

    def write(data: bytes) -> None:
        view = memoryview(data)
        while view:
            try:
                n = os.write(dst_fd, view)
            except InterruptedError:
                continue
            view = view[n:]

    def finish_oversized() -> None:
        msg_id = _message_id(head, bytes(tail))
        if msg_id is _NO_ID:
            counts["dropped"] += 1
            if note:
                note(f"reaper: dropped a {size}-byte message without an id (over the relay cap)")
        else:
            counts["replaced"] += 1
            write(oversized_error(msg_id, size, cap_bytes))
            if note:
                note(f"reaper: replaced a {size}-byte response (id {msg_id!r}) over the relay cap")

    while True:
        try:
            data = os.read(src_fd, chunk)
        except InterruptedError:
            continue
        except OSError:
            data = b""
        if not data:
            break
        start = 0
        while start < len(data):
            nl = data.find(b"\n", start)
            piece = data[start:] if nl < 0 else data[start:nl + 1]
            start = len(data) if nl < 0 else nl + 1
            if not over:
                buf += piece
                if len(buf) > cap_bytes:
                    over = True
                    size = len(buf)
                    head = bytes(buf[:RELAY_KEEP])
                    tail = bytearray(buf[-RELAY_KEEP:])
                    buf = bytearray()
            else:
                size += len(piece)
                tail += piece
                if len(tail) > RELAY_KEEP:
                    del tail[:len(tail) - RELAY_KEEP]
            if nl >= 0:
                counts["messages"] += 1
                if over:
                    finish_oversized()
                    over, size, head, tail = False, 0, b"", bytearray()
                else:
                    write(bytes(buf))
                    buf = bytearray()
    if over:
        counts["messages"] += 1
        finish_oversized()
    elif buf:
        counts["messages"] += 1
        write(bytes(buf))
    return counts


# --------------------------------------------------------------------------- launch


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    if "--" not in argv:
        raise SystemExit("reaper: usage: reaper.py --limit-mb N --status PATH --server NAME -- cmd args...")
    cut = argv.index("--")
    p = argparse.ArgumentParser(prog="reaper.py", add_help=False)
    p.add_argument("--limit-mb", type=int, default=0)
    p.add_argument("--status", default=None)
    p.add_argument("--server", default="")
    p.add_argument("--strip-python-flags", action="store_true")
    p.add_argument("--containment", choices=CONTAINMENTS, default=None)
    p.add_argument("--relay-max-mb", type=float, default=None)
    ns = p.parse_args(argv[:cut])
    command = argv[cut + 1:]
    if not command:
        raise SystemExit("reaper: no command after --")
    if ns.containment is None:
        env = (os.environ.get(CONTAINMENT_ENV) or "rlimit_data").strip()
        ns.containment = env if env in CONTAINMENTS else "rlimit_data"
    if ns.relay_max_mb is None:
        try:
            ns.relay_max_mb = float(os.environ.get(RELAY_ENV) or 0)
        except ValueError:
            ns.relay_max_mb = 0.0
    return ns, command


def _signal_group(pid: int, signum: int) -> None:
    """``signum`` to the child's process group (to the child alone when the group is gone)."""
    try:
        os.killpg(pid, signum)
    except OSError:
        try:
            os.kill(pid, signum)
        except OSError:
            pass


def _set_pdeathsig(parent: int) -> None:
    """Have the kernel SIGKILL this (child) process when the reaper dies (Linux)."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0)
    except (OSError, AttributeError):
        return
    if os.getppid() != parent:  # the reaper died before prctl took effect
        os._exit(1)


def _exec_child(command: list[str], env: dict[str, str], limit_mb: int, parent: int, *,
                cgroup: Cgroup | None = None, stdout_fd: int | None = None, stderr_fd: int | None = None,
                close_fds: tuple[int, ...] = ()) -> None:
    """In the forked child: process group, death signal, cgroup, RLIMIT_DATA, exec. Never returns."""
    try:
        os.setpgid(0, 0)
        _set_pdeathsig(parent)
        if cgroup is not None:
            cgroup.add(os.getpid())
        if stdout_fd is not None:
            os.dup2(stdout_fd, 1)
        if stderr_fd is not None:
            os.dup2(stderr_fd, 2)
        for fd in close_fds:
            try:
                os.close(fd)
            except OSError:
                pass
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


def contain(mode: str, server: str, limit: int) -> tuple[str, Cgroup | None, list[str] | None, bool]:
    """``(containment, cgroup, command prefix, watchdog)`` for ``mode`` on this host."""
    if limit <= 0 or mode == "none":
        return "none", None, None, False
    if mode == "rlimit_data":
        return "rlimit_data", None, None, False
    if mode in ("cgroup", "auto", "rss"):
        cg = make_cgroup_v2(server, limit)
        if cg is not None:
            return "cgroup_v2", cg, None, False
        cg = make_cgroup_v1(server, limit)
        if cg is not None:
            return "cgroup_v1", cg, None, False
        prefix = systemd_scope_prefix(server, limit)
        if prefix is not None:
            return "systemd_scope", None, prefix, False
    return "watchdog", None, None, True


def main(argv: list[str] | None = None) -> int:
    ns, command = parse_args(list(sys.argv[1:] if argv is None else argv))
    python = ns.strip_python_flags or is_python(command[0])
    stripped: list[str] = []
    env = dict(os.environ)
    for key in (CONTAINMENT_ENV, RELAY_ENV, CGROUP_ROOT_ENV, CGROUP_V1_ROOT_ENV):
        env.pop(key, None)                 # launcher settings never reach the server
    if python:
        rest, stripped = strip_python_flags(command[1:])
        command = [command[0], *rest]
        env = child_environment(env, isolated="-I" in stripped)
    limit = max(0, int(ns.limit_mb))
    containment, cgroup, prefix, watchdog = contain(ns.containment, ns.server, limit)
    data_limit = 0 if ns.containment == "rss" else limit       # rss: contained by resident memory only
    if prefix:
        command = [*prefix, *command]
    relay_cap = int(float(ns.relay_max_mb or 0) * MB)
    relay_r = relay_w = None
    if relay_cap > 0:
        relay_r, relay_w = os.pipe()
    # a caller that merges the streams gets them merged in the child: one pipe for fds 1 and 2, copied in order
    merged = relay_cap <= 0 and merged_output()
    out_fd = os.dup(1) if relay_cap > 0 or merged else None
    err_r, err_w = os.pipe()                       # the child's stderr, teed to ours (INV-1)
    oom_host_before = host_oom_kills()
    started_us = time.clock_gettime_ns(time.CLOCK_MONOTONIC) // 1000 - KMSG_CLOCK_MARGIN_US

    launcher = os.getppid()
    parent = os.getpid()
    pid = os.fork()
    if pid == 0:
        _exec_child(command, env, data_limit, parent, cgroup=cgroup, stdout_fd=err_w if merged else relay_w,
                    stderr_fd=err_w,
                    close_fds=tuple(fd for fd in (relay_r, relay_w, out_fd, err_r, err_w) if fd is not None))

    # ---- parent (the reaper)
    try:
        os.setpgid(pid, pid)       # also done by the child; whichever runs first wins the race
    except OSError:
        pass
    if cgroup is not None:
        cgroup.add(pid)            # the child also adds itself before exec; whichever runs first
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)
    os.close(devnull)
    os.close(err_w)
    tee = StderrTee(err_r, dst_fd=out_fd if merged and out_fd is not None else 2).start()
    relay_thread = None
    relay_counts: dict[str, int] = {}
    if relay_cap > 0 and relay_r is not None and relay_w is not None and out_fd is not None:
        os.close(relay_w)

        def _note(text: str) -> None:
            try:
                sys.stderr.write(text + "\n")
                sys.stderr.flush()
            except (OSError, ValueError):
                pass

        def _relay() -> None:
            relay_counts.update(relay(relay_r, out_fd, relay_cap, note=_note))

        relay_thread = threading.Thread(target=_relay, name="vbt-relay", daemon=True)
        relay_thread.start()

    def forward(signum: int, _frame: Any) -> None:
        _signal_group(pid, signum)

    for sig in FORWARDED:
        signal.signal(sig, forward)

    status: dict[str, Any] = {
        "pid": pid, "server": ns.server, "limit_mb": limit,
        "containment": containment,
        "hash_seed": int(HASH_SEED) if python else None, "flags_stripped": stripped,
        "rss_mb": None, "peak_rss_mb": None, "ts": round(time.time(), 3),
    }
    if containment not in ("none", "rlimit_data"):
        status["rlimit_data"] = data_limit > 0
    if cgroup is not None:
        status["cgroup"] = cgroup.path
    if watchdog:
        status["watchdog_kill_mb"] = round(watchdog_threshold_mb(limit), 1)
    if relay_cap > 0:
        status["relay_max_mb"] = float(ns.relay_max_mb)
    write_status(ns.status, status)
    peak = 0.0
    killed_by_watchdog = False
    orphaned_at: float | None = None
    orphan_killed = False
    oom_before = cgroup.oom_kills() if cgroup is not None else 0
    last_rss = last_write = time.monotonic()
    while True:
        try:
            wpid, wstatus, usage = os.wait4(pid, os.WNOHANG)
        except ChildProcessError:  # pragma: no cover - reaped elsewhere
            wpid, wstatus, usage = pid, 0, None
        if wpid == pid:
            break
        now = time.monotonic()
        if orphaned_at is None:
            if os.getppid() != launcher:                # re-parented: the launcher (the bridge) died
                orphaned_at = now
                _signal_group(pid, signal.SIGTERM)
        elif not orphan_killed and now - orphaned_at >= ORPHAN_GRACE_S:
            orphan_killed = True
            _signal_group(pid, signal.SIGKILL)
        if now - last_rss >= RSS_POLL_S:
            last_rss = now
            rss, hwm = read_proc_status(pid)
            if rss is not None:
                peak = max(peak, rss, hwm or 0.0)
                status["rss_mb"], status["peak_rss_mb"] = round(rss, 1), round(peak, 1)
            if watchdog and not killed_by_watchdog:
                group = group_rss_mb(pid)
                if group is not None and group >= watchdog_threshold_mb(limit):
                    killed_by_watchdog = True
                    peak = max(peak, group)
                    status["watchdog_rss_mb"] = round(group, 1)
                    _signal_group(pid, signal.SIGKILL)
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
    if relay_thread is not None:
        relay_thread.join(timeout=10.0)
        if out_fd is not None:
            try:
                os.close(out_fd)
            except OSError:
                pass
    maxrss_kb = int(getattr(usage, "ru_maxrss", 0) or 0)
    peak = max(peak, maxrss_kb / 1024.0)
    stderr_tail = tee.finish()
    if merged and out_fd is not None:
        try:
            os.close(out_fd)
        except OSError:
            pass
    cgroup_oom = cgroup is not None and cgroup.oom_kills() > oom_before
    kernel_oom = possible_kernel_oom = False
    if signum == signal.SIGKILL and not killed_by_watchdog and not cgroup_oom and not orphan_killed:
        logged = kernel_oom_killed(pid, started_us)
        kernel_oom, possible_kernel_oom = kill_attribution(logged, oom_host_before,
                                                           host_oom_kills() if logged is None else None)
    cause, memory_line = exit_cause(
        abnormal=rc != 0, signum=signum, limit_mb=limit, peak_mb=peak, watchdog=killed_by_watchdog,
        cgroup_oom=cgroup_oom, stderr_tail=stderr_tail, kernel_oom=kernel_oom, data_limited=data_limit > 0) \
        if orphaned_at is None else (None, None)        # an orphan is stopped by the reaper, not by its memory
    reason = "memory_limit" if cause else ("signal" if signum is not None else "exit_code")
    marker: dict[str, Any] = {"pid": pid, "code": code, "signal": signum, "maxrss_kb": maxrss_kb, "reason": reason}
    if cause:
        marker["cause"] = cause
    if memory_line:
        marker["memory_error"] = memory_line
    if killed_by_watchdog:
        marker["watchdog"] = True
    if orphaned_at is not None:
        marker["orphaned"] = True
    if cgroup_oom:
        marker["cgroup_oom"] = True
    if possible_kernel_oom and not cause:
        marker["possible_kernel_oom"] = True
    status.update({"rss_mb": None, "peak_rss_mb": round(peak, 1), "ts": round(time.time(), 3), "exit": marker})
    if relay_counts:
        status["relay"] = dict(relay_counts)
    write_status(ns.status, status)
    if cgroup is not None:
        cgroup.remove()
    try:
        sys.stderr.write(f"{EXIT_MARKER} {json.dumps(marker, sort_keys=True)}\n")
        sys.stderr.flush()
    except (OSError, ValueError):
        pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
