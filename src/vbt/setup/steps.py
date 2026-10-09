"""The generic steps of ``vbt setup``, in order:

==========  =================================================================================================
probe       host facts (:mod:`vbt.setup.probe`)
configure   serving profile, sizing, data roots -> ``host.yaml``, ``host.env``, ``compose.vllm.yaml``
acquire     ``vbt data acquire`` for the tools the enabled agents may call (what to fetch comes from the
            descriptors' acquisition sections and the roster, never from setup)
size        ``vbt ds estimate --json`` of the tables the enabled servers load whole -> per-server memory, then
            the host configuration is written again with it
index       ``vbt ds index build --table T ...``: the resolver sidecars of the id types the enabled tools' tables hold
check       ``vbt ds check`` of the tables the enabled tools read (readiness R1-R10)
calibrate   ``vbt ds calibrate`` of the local tables loaded whole (memory admission on this host's data)
smoke       an offline mock session; ``vbt doctor`` (``--smoke`` once the model server answers); the analysis stack
==========  =================================================================================================

Every step after ``configure`` runs a ``vbt`` command as a child process with the host configuration (its
profiles and ``host.env``), so what setup checks is what sessions will run. The commands are configuration:
``setup.steps.<name>.run`` / ``.plan`` replace a step's ``vbt`` arguments (``{tools}``, ``{tables}``,
``{local_tables}``, ``{full_tables}``, ``{data_dir}`` and ``{state_dir}`` are substituted).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TextIO

from . import hostconfig
from .needs import Needs, compute_needs
from .state import SetupState, fingerprint

__all__ = ["STEPS", "STEP_NAMES", "SetupContext", "StepResult", "default_commands", "data_signature",
           "DEFAULT_RATES", "parse_estimate"]

#: Seconds per GB of local data each data step took on the reference run (docs/DEPLOYMENT.md §9: the 31 downloaded
#: Open Targets 25.09 tables, 1.76 GB, 4 CPUs). A host's own measurements replace them after its first run.
DEFAULT_RATES: dict[str, float] = {"size": 10.8, "index": 194.5, "check": 81.2, "calibrate": 34.0}
#: Download rate assumed until this host has measured one (bytes per second).
DEFAULT_DOWNLOAD_BPS = 50e6
GB = 1e9

#: The ``vbt`` arguments of each step (``setup.steps.<name>.run|plan`` in the config override them).
_COMMANDS: dict[str, dict[str, list[str]]] = {
    "acquire": {"run": ["data", "acquire", "--for-tools", "{tools}"],
                "plan": ["data", "acquire", "--for-tools", "{tools}", "--plan", "--json"]},
    "size": {"run": ["ds", "estimate", "--json", "{full_tables}"]},
    "index": {"run": ["ds", "index", "build", "{tables}"]},     # only the id types of the tables tools read
    "check": {"run": ["ds", "check", "--json", "{tables}"]},
    "calibrate": {"run": ["ds", "calibrate", "--json", "{full_local_tables}"]},
}


def default_commands() -> dict[str, dict[str, list[str]]]:
    return json.loads(json.dumps(_COMMANDS))


@dataclass
class StepResult:
    status: str                      # done | failed | skipped | deferred | unavailable
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class SetupContext:
    """Everything a step needs. ``runner`` runs a child (tests replace it)."""

    config: dict[str, Any]
    layout: Any
    state: SetupState
    options: dict[str, Any]
    out: TextIO = sys.stdout
    facts: dict[str, Any] | None = None
    needs: Needs | None = None
    serving: dict[str, Any] | None = None
    sizing: dict[str, Any] | None = None
    runner: Callable[..., subprocess.CompletedProcess] | None = None
    verbose: bool = False
    #: ``host.env`` as this run computed it (before it is written, e.g. under ``--plan``)
    pending_env: dict[str, str] | None = None
    #: bytes the acquisition plan will add (the data steps' time estimates include them)
    planned_fetch_bytes: int = 0

    # ------------------------------------------------------------------ helpers
    @property
    def base_profiles(self) -> list[str]:
        return [str(p) for p in (self.config.get("profiles") or [])]

    def profiles(self) -> list[str]:
        """The profiles sessions use: the serving profile's harness profile, the operator's, then host.yaml."""
        harness = (self.serving or {}).get("harness_profile")
        out = [harness] if harness and harness not in self.base_profiles else []
        out += [p for p in self.base_profiles if not p.endswith(hostconfig.HOST_PROFILE)]
        host = Path(self.layout.state) / hostconfig.HOST_PROFILE
        if host.is_file():
            out.append(str(host))
        return out

    def host_env(self) -> dict[str, str]:
        if self.pending_env is not None:
            return dict(self.pending_env)
        return hostconfig.read_env_file(Path(self.layout.state) / hostconfig.HOST_ENV)

    def commands(self, step: str) -> dict[str, list[str]]:
        cmds = default_commands().get(step, {})
        own = (((self.config.get("setup") or {}).get("steps") or {}).get(step) or {})
        for key in ("run", "plan"):
            if isinstance(own.get(key), list):
                cmds[key] = [str(a) for a in own[key]]
        return cmds

    def render(self, argv: Sequence[str]) -> list[str]:
        """Substitute the placeholders. A list placeholder alone in an argument expands to one argument per item:
        ``{tools}`` to the bare names (``--for-tools {tools}`` -> ``--for-tools a.x b.y``), the table lists to
        repeated ``--table T``. Elsewhere ``{tools}`` is the comma-joined names."""
        needs = self.needs or Needs()
        full = sorted({t for ts in needs.server_full_loads().values() for t in ts})
        local = needs.local_tables()
        lists = {"tables": sorted(needs.tables), "local_tables": local, "full_tables": full,
                 "full_local_tables": [t for t in full if t in local]}
        scalars = {"tools": ",".join(needs.tools), "data_dir": str(self.layout.data),
                   "state_dir": str(self.layout.state)}
        out: list[str] = []
        for arg in argv:
            m = re.fullmatch(r"\{(\w+)\}", arg)
            if m and m.group(1) == "tools":
                out += list(needs.tools)
                continue
            if m and m.group(1) in lists:
                for t in lists[m.group(1)]:
                    out += ["--table", t]
                continue
            for k, v in scalars.items():
                arg = arg.replace("{" + k + "}", v)
            out.append(arg)
        return out

    def vbt(self, argv: Sequence[str], *, log: str, timeout: float | None = None,
            profiles: Sequence[str] | None = None) -> subprocess.CompletedProcess:
        """Run ``python -m vbt.cli --profile ... <argv>`` with ``host.env`` applied; stdout and stderr are kept
        in ``<state>/logs/<log>.log`` (and returned)."""
        prof = list(self.profiles() if profiles is None else profiles)
        cmd = [sys.executable, "-m", "vbt.cli"]
        for p in prof:
            cmd += ["--profile", p]
        cmd += list(argv)
        env = dict(os.environ)
        env.update(self.host_env())
        src = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        logs = Path(self.layout.state) / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        run = self.runner or _run
        t0 = time.monotonic()
        res = run(cmd, env=env, timeout=timeout)
        took = time.monotonic() - t0
        with open(logs / f"{log}.log", "a", encoding="utf-8") as f:
            f.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ({took:.1f} s, exit "
                    f"{res.returncode}) $ {' '.join(cmd)}\n")
            f.write(res.stdout or "")
            if res.stderr:
                f.write("\n--- stderr\n" + res.stderr)
        if self.verbose:
            self.out.write((res.stdout or "") + (res.stderr or ""))
        return res

    def say(self, text: str) -> None:
        self.out.write(text + "\n")
        self.out.flush()


def _run(cmd: Sequence[str], *, env: Mapping[str, str], timeout: float | None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(list(cmd), env=dict(env), capture_output=True, text=True, timeout=timeout,
                              check=False)
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(list(cmd), 124, _text(exc.stdout), _text(exc.stderr) + "\n(timed out)")


def _text(value: Any) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def _unavailable(res: subprocess.CompletedProcess) -> bool:
    """The ``vbt`` command or one of its options does not exist in this installation."""
    err = res.stderr or ""
    return res.returncode == 2 and ("invalid choice" in err or "unrecognized arguments" in err)


def _tail(text: str, n: int = 6) -> str:
    lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    return " | ".join(lines[-n:])[-800:]


def _json_out(text: str) -> Any:
    """The JSON document of a command's stdout (the whole text, else its last line that parses)."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    start = text.find("{")
    if start >= 0:
        try:
            return json.loads(text[start:])
        except ValueError:
            pass
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith(("{", "[")):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


def data_signature(ctx: SetupContext) -> list[Any]:
    """A cheap signature of the local data the needs read: per source root, its entries' names, sizes and
    modification times (two levels), so new or replaced files re-run the data steps."""
    sig: list[Any] = []
    for source, rec in sorted((ctx.needs.sources if ctx.needs else {}).items()):
        root = _source_root(ctx, source, rec)
        if not root:
            continue
        p = Path(root)
        entries: list[Any] = []
        if p.is_dir():
            for child in sorted(p.iterdir())[:2000]:
                try:
                    st = child.stat()
                except OSError:
                    continue
                entries.append([child.name, st.st_size, st.st_mtime_ns])
                if child.is_dir():
                    try:
                        inner = sorted(child.iterdir())[:200]
                    except OSError:
                        inner = []
                    entries.append([[c.name, c.stat().st_size, c.stat().st_mtime_ns] for c in inner if c.exists()])
        sig.append([source, str(p), entries])
    return sig


def _source_root(ctx: SetupContext, source: str, rec: Mapping[str, Any]) -> str | None:
    var = rec.get("root_var")
    if var:
        value = ctx.host_env().get(var) or os.environ.get(var)
        if value:
            return value
    return rec.get("root")


def local_bytes(ctx: SetupContext) -> int:
    """Bytes on disk of the local sources the needs read (what the data steps scan)."""
    total = 0
    for source, rec in (ctx.needs.sources if ctx.needs else {}).items():
        if rec.get("kind") != "local":
            continue
        root = _source_root(ctx, source, rec)
        if not root or not Path(root).exists():
            continue
        tables = set(rec.get("tables") or [])
        for child in Path(root).iterdir() if Path(root).is_dir() else []:
            if child.is_dir() and child.name not in tables:
                continue
            for p in child.rglob("*") if child.is_dir() else [child]:
                try:
                    if p.is_file():
                        total += p.stat().st_size
                except OSError:
                    continue
    return total


# ============================================================================ steps

class Step:
    name = ""
    help = ""
    needs_data_layer = False

    def inputs(self, ctx: SetupContext) -> list[Any]:
        return []

    def plan(self, ctx: SetupContext) -> dict[str, Any]:
        return {"action": "run"}

    def run(self, ctx: SetupContext) -> StepResult:  # pragma: no cover - every step overrides it
        raise NotImplementedError


class ProbeStep(Step):
    name = "probe"
    help = "host facts: CPUs, memory, GPUs, disk, memory containment, sandbox, container, network"

    def plan(self, ctx: SetupContext) -> dict[str, Any]:
        return {"action": "run", "seconds": 15, "detail": "always runs (cheap)"}

    def run(self, ctx: SetupContext) -> StepResult:
        from .probe import probe_host

        if ctx.facts is None:
            ctx.facts = probe_host(ctx.layout, ctx.config, **ctx.options.get("probe_kwargs", {}))
        ctx.state.data["probe"] = ctx.facts
        mem = ctx.facts.get("memory") or {}
        gpu = ctx.facts.get("gpu") or {}
        unreachable = sorted(k for k, v in (ctx.facts.get("network") or {}).items() if v.get("reachable") is False)
        detail = (f"{(ctx.facts.get('cpu') or {}).get('effective')} CPUs, {mem.get('effective_mb')} MB RAM, "
                  f"{gpu.get('count', 0)} GPU(s)")
        if unreachable:
            detail += f"; unreachable: {', '.join(unreachable)}"
        return StepResult("done", detail, {"unreachable": unreachable})


class ConfigureStep(Step):
    name = "configure"
    help = "pick the serving profile, size memory, place the data roots; write host.yaml, host.env, compose.vllm.yaml"

    def plan(self, ctx: SetupContext) -> dict[str, Any]:
        ensure_configured(ctx, write=False)
        return {"action": "run", "seconds": 1, "detail": _serving_line(ctx)}

    def run(self, ctx: SetupContext) -> StepResult:
        written = ensure_configured(ctx, write=True)
        notes = list((ctx.sizing or {}).get("notes") or []) + list((ctx.serving or {}).get("warnings") or [])
        return StepResult("done", _serving_line(ctx) + (f"; {len(notes)} note(s)" if notes else ""),
                          {"files": [str(p) for p in written], "serving": ctx.serving, "sizing": ctx.sizing,
                           "notes": notes})


def _serving_line(ctx: SetupContext) -> str:
    s = ctx.serving or {}
    z = ctx.sizing or {}
    parts = [f"serving profile {s.get('serving_profile') or 'none'}", f"harness profile {s.get('harness_profile') or 'default'}"]
    if z.get("host_budget_mb"):
        parts.append(f"host budget {z['host_budget_mb']:,} MB, server limit {z.get('default_server_mb', 0):,} MB")
    return ", ".join(parts)


def data_roots(ctx: SetupContext) -> dict[str, str]:
    """The root variable of every local source the needs read: what the acquisition reported, else the current
    value, else ``<data>/sources/<source>/<release or current>`` (the home ``vbt data acquire`` uses by default:
    ``data.acquisition.root`` = ``${VBT_DATA_DIR}/sources``)."""
    roots: dict[str, str] = {}
    acquired = ((ctx.state.step("acquire").get("data") or {}).get("roots") or {})
    for source, rec in sorted((ctx.needs.sources if ctx.needs else {}).items()):
        var = rec.get("root_var")
        if rec.get("kind") != "local" or not var:
            continue
        value = acquired.get(var) or os.environ.get(var) or ""
        if not value.strip():
            value = str(Path(ctx.layout.data) / "sources" / source / (rec.get("release") or "current"))
        roots[var] = value
    return roots


def ensure_configured(ctx: SetupContext, *, write: bool) -> list[Path]:
    """Compute serving, sizing and data roots into ``ctx`` (probing first when needed) and, with ``write``,
    write the host files."""
    if ctx.facts is None:
        ProbeStep().run(ctx)
    if ctx.needs is None:
        ctx.needs = compute_needs(ctx.config)
    opts = ctx.options
    if ctx.serving is None:
        ctx.serving = hostconfig.pick_serving(
            ctx.facts or {}, serving_profile=opts.get("serving_profile"), variants=opts.get("variants") or (),
            harness_profile=opts.get("harness_profile"), llm_url=opts.get("llm_url"))
    need = (ctx.state.step("size").get("data") or {}).get("server_need_mb")
    ctx.sizing = hostconfig.size_host(ctx.facts or {}, ctx.config, server_need_mb=need)
    roots = data_roots(ctx)
    profile = hostconfig.host_profile(ctx.sizing, ctx.layout, tool_env=roots)
    deploy = opts.get("deploy") or "host"
    host_yaml = str(Path(ctx.layout.state) / hostconfig.HOST_PROFILE)
    profiles = [p for p in ctx.profiles() if p != host_yaml] + [host_yaml]
    ctx.pending_env = hostconfig.host_env(ctx.layout, ctx.serving, data_roots=roots, profiles=profiles,
                                          deploy=deploy, llm_url=opts.get("llm_url"))
    if not write:
        return []
    compose = hostconfig.render_compose_vllm(ctx.serving) if deploy == "compose" else None
    header = (f"Written by `vbt setup` on {(ctx.facts or {}).get('hostname') or 'this host'} at "
              f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}. Regenerated by every `vbt setup` run:\n"
              "put your own settings in another profile (e.g. configs/profiles/production-local.yaml).")
    for d in ctx.layout.dirs().values():
        Path(d).mkdir(parents=True, exist_ok=True)
    return hostconfig.write_host_files(Path(ctx.layout.state), profile, ctx.pending_env, compose, header=header)


class AcquireStep(Step):
    name = "acquire"
    help = "fetch the data the enabled tools read (`vbt data acquire`)"
    needs_data_layer = True

    #: the step moves the data roots: its fingerprint is taken again after it ran (cli.run_steps)
    refingerprint = True

    def inputs(self, ctx: SetupContext) -> list[Any]:
        # files deleted or left partial since the last run change the signature: the step runs again
        return [ctx.render(ctx.commands(self.name).get("run") or []), data_roots(ctx), data_signature(ctx)]

    def plan(self, ctx: SetupContext) -> dict[str, Any]:
        argv = ctx.commands(self.name).get("plan")
        if not argv:
            return {"action": "run", "detail": "no plan command configured"}
        if ctx.needs is not None and not ctx.needs.tools:
            return {"action": "nothing", "bytes": 0, "detail": "no enabled agent may call an upstream tool"}
        argv = ctx.render(argv)
        if not (ctx.options.get("probe_kwargs") or {}).get("network", True) and "--offline" not in argv:
            argv.append("--offline")              # --no-network: plan from the declared sizes
        res = ctx.vbt(argv, log="acquire-plan", timeout=ctx.options.get("plan_timeout_s", 900))
        if _unavailable(res):
            return {"action": "unavailable", "detail": "`vbt data acquire` is not available in this installation: "
                                                       f"{_tail(res.stderr, 2)}"}
        doc = _json_out(res.stdout)
        if res.returncode != 0 and doc is None:
            return {"action": "run", "detail": f"the acquisition plan failed (exit {res.returncode}): "
                                               f"{_tail(res.stderr or res.stdout, 3)}"}
        total = _bytes_to_fetch(doc)
        todo = _plan_pending(doc)
        # transfer rate: the one `vbt data acquire` measured on this host, else setup's, else an assumption
        measured = ctx.state.rate("download_bps")
        if isinstance(doc, Mapping) and doc.get("rate_measured") and isinstance(doc.get("rate_mbps"), (int, float)):
            measured = float(doc["rate_mbps"]) * 1e6
        bps = measured or DEFAULT_DOWNLOAD_BPS
        out: dict[str, Any] = {"action": "run" if total or todo else "nothing", "bytes": total,
                               "disk_bytes": total, "plan": doc}
        if total:
            out["seconds"] = round(total / bps)
            out["detail"] = (f"{_fmt_bytes(total)} to fetch at {'the measured' if measured else 'an assumed'} "
                             f"{bps / 1e6:,.0f} MB/s")
        else:
            out["detail"] = "; ".join(todo) if todo else "nothing to fetch"
        return out

    def run(self, ctx: SetupContext) -> StepResult:
        plan = self.plan(ctx)
        if plan.get("action") == "unavailable":
            return StepResult("unavailable", plan["detail"])
        need = plan.get("disk_bytes") or 0
        free = _free_bytes(ctx.layout.data)
        if need and free is not None and need > free:
            return StepResult("failed", f"{need / GB:,.1f} GB to fetch but {free / GB:,.1f} GB free under "
                                        f"{ctx.layout.data}")
        data: dict[str, Any] = {"bytes": 0, "roots": _plan_roots(plan.get("plan"))}
        if plan.get("action") == "nothing":
            detail = "nothing to fetch"
        else:
            t0 = time.monotonic()
            res = ctx.vbt(ctx.render(ctx.commands(self.name)["run"]), log="acquire",
                          timeout=ctx.options.get("acquire_timeout_s"))
            took = time.monotonic() - t0
            if _unavailable(res):
                return StepResult("unavailable", f"`vbt data acquire` is not available: {_tail(res.stderr, 2)}")
            if res.returncode != 0:
                return StepResult("failed", f"exit {res.returncode}: {_tail(res.stderr or res.stdout)}")
            if need >= 50e6 and took > 1:          # small transfers measure process start-up, not the network
                ctx.state.observe_rate("download_bps", need / took)
            data["bytes"] = need
            detail = (f"fetched {_fmt_bytes(need)} in {took:,.0f} s" if need else
                      f"{plan.get('detail') or 'acquisition'} done in {took:,.0f} s")
        # the roots the acquisition reported go into host.yaml / host.env before the data steps read them
        ctx.state.record(self.name, data=data)
        data["files"] = [str(f) for f in ensure_configured(ctx, write=True)]
        return StepResult("done", detail, data)


def _fmt_bytes(n: float) -> str:
    for unit, size in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= size:
            return f"{n / size:,.2f} {unit}"
    return f"{n:,.0f} bytes"


def _bytes_to_fetch(doc: Any) -> int:
    """Bytes a ``vbt data acquire --plan --json`` document says remain to transfer (``bytes_remaining`` at the
    top, else summed over its download-mode sources; a source whose size is unknown counts 0)."""
    if not isinstance(doc, Mapping) or doc.get("mode") not in (None, "download"):
        return 0
    for key in ("bytes_remaining", "bytes_to_fetch"):
        if isinstance(doc.get(key), (int, float)):
            return int(doc[key])
    total = 0
    for key in ("sources", "items"):
        items = doc.get(key)
        if isinstance(items, Mapping):
            items = list(items.values())
        for it in items or []:
            if isinstance(it, Mapping):
                total += _bytes_to_fetch(it)
    return total


def _plan_pending(doc: Any) -> list[str]:
    """What an acquisition plan leaves to do besides bytes it can size: download-mode sources with files missing
    or partial (or of unknown size) and prepare steps not done."""
    out: list[str] = []
    sources = doc.get("sources") if isinstance(doc, Mapping) else None
    for s in sources if isinstance(sources, list) else []:
        if not isinstance(s, Mapping) or s.get("mode") not in (None, "download"):
            continue
        name = s.get("source") or "?"
        states = s.get("states") if isinstance(s.get("states"), Mapping) else {}
        if s.get("bytes_remaining") is None or states.get("missing") or states.get("partial"):
            out.append(f"{name}: files to fetch")
        steps = [k for k, v in (s.get("prepare") or {}).items() if v != "done"]
        if steps:
            out.append(f"{name}: prepare {', '.join(steps)}")
    return out


def _plan_roots(doc: Any) -> dict[str, str]:
    """``{variable: directory}``: the variables an acquisition plan says point the data layer at the files
    (``sources[*].env``)."""
    out: dict[str, str] = {}
    if not isinstance(doc, Mapping):
        return out
    sources = doc.get("sources")
    if isinstance(sources, Mapping):
        sources = list(sources.values())
    for s in sources or []:
        if isinstance(s, Mapping) and isinstance(s.get("env"), Mapping):
            out.update({str(k): str(v) for k, v in s["env"].items() if v})
    return out


def _free_bytes(path: str | Path) -> int | None:
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    try:
        return shutil.disk_usage(p).free
    except OSError:
        return None


_EST_LINE = re.compile(r"^(?P<table>[\w.]+): .*?upstream full load ~(?P<mb>[\d,]+) MB")
_EST_ERR = re.compile(r"^(?P<table>[\w.]+): error: (?P<err>.*)$")


def parse_estimate(text: str) -> tuple[dict[str, float], dict[str, str]]:
    """``({table: MB}, {table: error})`` from ``vbt ds estimate --json`` (``tables.<ref>.upstream_mb``,
    ``errors``), or from its text lines."""
    sizes: dict[str, float] = {}
    errors: dict[str, str] = {}
    doc = _json_out(text)
    if isinstance(doc, Mapping) and isinstance(doc.get("tables"), Mapping):
        for ref, rec in doc["tables"].items():
            if isinstance(rec, Mapping) and isinstance(rec.get("upstream_mb"), (int, float)):
                sizes[str(ref)] = float(rec["upstream_mb"])
        errors = {str(k): str(v) for k, v in (doc.get("errors") or {}).items()}
        return sizes, errors
    for line in (text or "").splitlines():
        m = _EST_LINE.match(line.strip())
        if m:
            sizes[m.group("table")] = float(m.group("mb").replace(",", ""))
            continue
        m = _EST_ERR.match(line.strip())
        if m:
            errors[m.group("table")] = m.group("err")
    return sizes, errors


class _DataStep(Step):
    """A step that runs one ``vbt ds`` command over the local data; time = this host's rate x local GB."""

    needs_data_layer = True
    timeout_key = ""

    def inputs(self, ctx: SetupContext) -> list[Any]:
        return [ctx.render(ctx.commands(self.name).get("run") or []), data_signature(ctx),
                _file_digest(Path(ctx.layout.state) / hostconfig.HOST_PROFILE)]

    def plan(self, ctx: SetupContext) -> dict[str, Any]:
        gb = (local_bytes(ctx) + ctx.planned_fetch_bytes) / GB
        rate = ctx.state.rate(self.name) or DEFAULT_RATES.get(self.name, 60.0)
        return {"action": "run", "seconds": round(max(5.0, gb * rate)),
                "detail": f"{gb:,.2f} GB of local data at {rate:,.0f} s/GB"
                          f"{' (measured here)' if ctx.state.rate(self.name) else ' (reference run)'}"}

    def execute(self, ctx: SetupContext, argv_key: str = "run") -> tuple[subprocess.CompletedProcess, float]:
        t0 = time.monotonic()
        res = ctx.vbt(ctx.render(ctx.commands(self.name)[argv_key]), log=self.name,
                      timeout=ctx.options.get(f"{self.name}_timeout_s"))
        took = time.monotonic() - t0
        gb = local_bytes(ctx) / GB
        if res.returncode in (0, 1) and gb > 0.01:
            ctx.state.observe_rate(self.name, took / gb)
        return res, took


def _file_digest(path: Path) -> str | None:
    """Digest of a YAML file's content (comments, such as the generation time, do not count)."""
    import yaml

    try:
        return fingerprint(yaml.safe_load(path.read_text()))
    except (OSError, yaml.YAMLError):
        return None


class SizeStep(_DataStep):
    name = "size"
    help = "estimate the tables each enabled server loads whole; size the server memory limit to the largest"

    def run(self, ctx: SetupContext) -> StepResult:
        full = (ctx.needs or Needs()).server_full_loads()
        if not full:
            return StepResult("done", "no enabled server loads whole tables", {"server_need_mb": {}})
        res, took = self.execute(ctx)
        sizes, errors = parse_estimate(res.stdout)
        if not sizes:
            return StepResult("failed", f"no estimate (exit {res.returncode}): {_tail(res.stderr or res.stdout)}")
        need = {server: round(sum(sizes.get(t, 0.0) for t in tables), 1) for server, tables in full.items()}
        missing = sorted({t for tables in full.values() for t in tables if t not in sizes})
        data = {"server_need_mb": {k: v for k, v in need.items() if v > 0}, "table_mb": sizes,
                "unmeasured": missing, "errors": errors}
        ctx.state.record(self.name, data=data)   # configure reads it
        written = ensure_configured(ctx, write=True)
        largest = max(need.items(), key=lambda kv: kv[1]) if need else ("-", 0)
        detail = (f"{len(sizes)} table(s) estimated in {took:,.0f} s; largest server {largest[0]} "
                  f"~{largest[1]:,.0f} MB; server limit {ctx.sizing.get('default_server_mb', 0):,} MB")
        if missing:
            detail += f"; {len(missing)} table(s) not on disk (not counted)"
        data["files"] = [str(p) for p in written]
        return StepResult("done", detail, data)


class IndexStep(_DataStep):
    name = "index"
    help = "build the resolver and access-path sidecars (`vbt ds index build`)"

    def run(self, ctx: SetupContext) -> StepResult:
        if ctx.needs is not None and not ctx.needs.tables:
            # without --table, `vbt ds index build` would build every local id type of every source
            return StepResult("done", "no enabled tool reads a table", {"built": []})
        res, took = self.execute(ctx)
        lines = [ln.strip() for ln in (res.stdout or "").splitlines()]
        built = [ln.split(":", 2)[0].removeprefix("built ") + ":" + ln.split(":", 2)[1]
                 for ln in lines if ln.startswith("built ") and ln.count(":") >= 2]
        failed = [ln.removeprefix("failed ") for ln in lines if ln.startswith("failed ")]
        # an index whose table is not on this host fails alone (TableUnavailable): `check` reports the data
        absent = [f.split(":", 2)[0] + ":" + f.split(":", 2)[1] for f in failed if "TableUnavailable" in f]
        broken = [f for f in failed if "TableUnavailable" not in f]
        data = {"built": built, "absent_data": absent, "errors": broken}
        detail = f"{len(built)} index(es) built in {took:,.0f} s; {len(absent)} without data on this host"
        if broken or (res.returncode != 0 and not built):
            return StepResult("failed", detail + f"; errors: {_tail(chr(10).join(broken) or res.stderr, 3)}", data)
        return StepResult("done", detail, data)


class CheckStep(_DataStep):
    name = "check"
    help = "readiness checks R1-R10 of the tables the enabled tools read (`vbt ds check`)"

    def run(self, ctx: SetupContext) -> StepResult:
        if ctx.needs is not None and not ctx.needs.tables:
            # without --table, `vbt ds check` would check every declared table
            return StepResult("done", "no enabled tool reads a table", {"tables": {}})
        res, took = self.execute(ctx)
        doc = _json_out(res.stdout)
        if not isinstance(doc, Mapping):
            return StepResult("failed", f"exit {res.returncode}: {_tail(res.stderr or res.stdout)}")
        tables = doc.get("tables") or {}
        status: dict[str, int] = {}
        for rec in tables.values():
            st = str((rec or {}).get("status") or "unknown")
            status[st] = status.get(st, 0) + 1
        unready = (doc.get("tools") or {}).get("unready") or {}
        errors = doc.get("table_errors") or []
        data = {"tables": status, "unready_tools": sorted(unready), "table_errors": errors,
                "quarantined": doc.get("quarantined") or []}
        detail = (f"{len(tables)} table(s) in {took:,.0f} s: "
                  + ", ".join(f"{n} {s}" for s, n in sorted(status.items()))
                  + f"; {len(unready)} tool(s) unready")
        ready = status.get("ready", 0)
        if tables and ready == 0:
            return StepResult("failed", detail, data)
        return StepResult("done", detail, data)


class CalibrateStep(_DataStep):
    name = "calibrate"
    help = "sample-and-scale memory calibration of the local tables loaded whole (`vbt ds calibrate`)"

    def run(self, ctx: SetupContext) -> StepResult:
        full = {t for ts in (ctx.needs or Needs()).server_full_loads().values() for t in ts}
        present = [t for t in sorted(full) if t in (ctx.needs or Needs()).local_tables()
                   and t in ((ctx.state.step("size").get("data") or {}).get("table_mb") or {})]
        if not present:
            return StepResult("done", "no local table loaded whole is on disk")
        cmd = list(ctx.commands(self.name)["run"])
        cmd = [a for a in cmd if a != "{full_local_tables}"]
        for t in present:
            cmd += ["--table", t]
        t0 = time.monotonic()
        res = ctx.vbt(ctx.render(cmd), log=self.name, timeout=ctx.options.get("calibrate_timeout_s"))
        took = time.monotonic() - t0
        gb = local_bytes(ctx) / GB
        if gb > 0.01:
            ctx.state.observe_rate(self.name, took / gb)
        if res.returncode != 0:
            return StepResult("failed", f"exit {res.returncode}: {_tail(res.stderr or res.stdout)}")
        return StepResult("done", f"{len(present)} table(s) calibrated in {took:,.0f} s")


class SmokeStep(Step):
    name = "smoke"
    help = "offline mock session, `vbt doctor` (with --smoke when the model server answers), analysis stack"

    def inputs(self, ctx: SetupContext) -> list[Any]:
        return [time.time()]          # always runs

    def plan(self, ctx: SetupContext) -> dict[str, Any]:
        return {"action": "run", "seconds": 120,
                "detail": "mock session + doctor; the live part needs the model server up"}

    def run(self, ctx: SetupContext) -> StepResult:
        results: dict[str, Any] = {}
        mock = ctx.vbt(["--skip-preflight", "run", "-q", "setup smoke test: say hello"], log="smoke-mock",
                       profiles=["mock", str(Path(ctx.layout.state) / hostconfig.HOST_PROFILE)], timeout=600)
        results["mock_session"] = {"exit": mock.returncode, "tail": _tail(mock.stdout or mock.stderr, 3)}
        server = _model_server_up(ctx)
        results["model_server"] = server
        doctor = ctx.vbt(["doctor", "--smoke"] if server.get("up") else ["doctor"], log="smoke-doctor",
                         timeout=ctx.options.get("smoke_timeout_s", 1800))
        results["doctor"] = {"exit": doctor.returncode, "smoke": bool(server.get("up")),
                             "tail": _tail(doctor.stdout, 3)}
        analysis_failed: list[str] = []
        if ctx.options.get("analysis", True):
            ana = ctx.vbt(["doctor", "--analysis"], log="smoke-analysis", timeout=1800)
            lines = [ln.strip() for ln in (ana.stdout or "").splitlines()]
            analysis_failed = [ln.removeprefix("[!!] analysis: ") for ln in lines if ln.startswith("[!!] analysis")]
            analysis_failed, confirmed_r = _confirm_r_packages(ctx, analysis_failed)
            results["analysis"] = {"ok": sum(ln.startswith("[ok] analysis") for ln in lines),
                                   "failed": analysis_failed}
            if confirmed_r:
                results["analysis"]["r_loaded_on_recheck"] = confirmed_r
        if mock.returncode != 0:
            return StepResult("failed", f"the offline mock session failed: {results['mock_session']['tail']}",
                              results)
        if analysis_failed:
            return StepResult("failed", f"the analysis stack is incomplete ({len(analysis_failed)}): "
                                        + "; ".join(analysis_failed[:4]), results)
        if not server.get("up"):
            return StepResult("deferred", f"offline checks passed; the model server ({server.get('url')}) does not "
                                          "answer yet: run `vbt setup --only smoke` once it serves", results)
        if doctor.returncode != 0:
            return StepResult("failed", f"vbt doctor --smoke: {results['doctor']['tail']}", results)
        return StepResult("done", "mock session, doctor --smoke"
                          + (" and the analysis stack" if ctx.options.get("analysis", True) else "") + " passed",
                          results)


_R_MISSING = re.compile(r"^R package ([A-Za-z][A-Za-z0-9.]*): not installed$")


def _confirm_r_packages(ctx: SetupContext, failed: list[str]) -> tuple[list[str], list[str]]:
    """Confirm each R package ``vbt doctor --analysis`` reports missing with its own ``Rscript`` call, so the step
    fails only on a package R cannot load. (The doctor reads all R packages from one ``cat`` of a vector, whose
    elements after the first come out behind a space; it then reports every package after the first missing.)
    Returns ``(still_failed, loaded_on_recheck)``."""
    env = {**os.environ, **ctx.host_env()}
    rscript = shutil.which("Rscript", path=env.get("PATH"))
    if not rscript:
        return failed, []
    run = ctx.runner or _run
    still, loaded = [], []
    for item in failed:
        m = _R_MISSING.match(item)
        if m is None:
            still.append(item)
            continue
        res = run([rscript, "-e", f'cat(requireNamespace("{m.group(1)}", quietly=TRUE))'], env=env, timeout=300)
        if res.returncode == 0 and (res.stdout or "").strip().endswith("TRUE"):
            loaded.append(m.group(1))
        else:
            still.append(item)
    return still, loaded


def _model_server_up(ctx: SetupContext) -> dict[str, Any]:
    """Whether the configured model server answers (``/health`` of a local server; the Anthropic API counts as up
    when a key is set)."""
    import httpx

    env = {**os.environ, **ctx.host_env()}
    harness = (ctx.serving or {}).get("harness_profile") or ""
    if harness in ("claude", "paper"):
        return {"up": bool(env.get("ANTHROPIC_API_KEY")), "url": "Anthropic API"}
    base = env.get("VBT_LLM_BASE_URL") or ctx.options.get("llm_url") or "http://localhost:8000/v1"
    url = base.rstrip("/").removesuffix("/v1") + "/health"
    try:
        with httpx.Client(timeout=5.0, trust_env=True) as client:
            code = client.get(url).status_code
        return {"up": code == 200, "url": url, "status": code}
    except Exception as exc:  # noqa: BLE001
        return {"up": False, "url": url, "error": f"{type(exc).__name__}: {exc}"[:200]}


STEPS: list[Step] = [ProbeStep(), ConfigureStep(), AcquireStep(), SizeStep(), IndexStep(), CheckStep(),
                     CalibrateStep(), SmokeStep()]
STEP_NAMES = [s.name for s in STEPS]
