"""Readiness checks: credentials, reference data, MCP servers and the analysis stack.

Two entry points:

* :func:`require_ready` -- a cheap gate run before a session starts and before
  every CSO turn. It raises :class:`DataReadinessError` *before* any model call
  when the provider credentials or the reference data the configured MCP
  servers need are missing, so no money is spent on a turn whose data tools
  would all fail.
* :func:`run_doctor` (``vbt doctor [--smoke] [--analysis]``) -- the full
  installation report, including the MCP interpreter's imports, a live smoke
  call per MCP server (fails on any tool error) and the Python/R analysis
  stack.

The Open Targets check reuses the upstream ``tools/doctor.py::reference_files``
(layout, truncated files, partial downloads, download manifest).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .config import base_tool_env, env_files, resolve_path
from .envpolicy import child_env

log = logging.getLogger(__name__)

TURN_NOT_SENT = "This turn has not been sent to the model."

#: Upstream servers that read the Open Targets parquet release.
OPEN_TARGETS_SERVERS = frozenset({
    "expression", "functional_genomics", "genetics", "target", "drug",
    "association", "disease", "interaction", "pathway",
})

#: Third-party modules each server imports in the MCP interpreter.
SERVER_MODULES: dict[str, tuple[str, ...]] = {
    "_stdio": ("fastmcp", "mcp"),
    "_opentargets": ("pandas", "numpy", "pyarrow", "dotenv"),
    "pathway": ("scipy",),
    "clinicaltrials": ("pandas", "requests", "pybioportal", "dotenv"),
    "single_cell": ("pandas", "numpy", "anndata", "cellxgene_census", "tiledbsoma", "dotenv"),
    "pubmed": ("httpx",),
}
FASTMCP_MAJORS = (3, 4)   # pyproject: fastmcp>=3.2,<5 (upstream pins 3.2.0)

ANALYSIS_MODULES = ("scanpy", "anndata", "pydeseq2", "gseapy", "decoupler", "liana", "harmonypy",
                    "matplotlib", "seaborn", "lifelines", "statsmodels", "rpy2")
R_PACKAGES = ("lme4", "lmerTest", "glmmTMB", "betareg", "MuMIn")

PCSK9 = "ENSG00000169174"
#: One cheap call per server for ``vbt doctor --smoke`` (override per server
#: in mcp_servers.yaml with ``smoke: {tool: ..., args: {...}}`` or ``smoke: false``).
SMOKE_CALLS: dict[str, tuple[str, dict[str, Any]]] = {
    "single_cell": ("get_census_info", {}),
    "target": ("search_targets_by_name", {"query": "PCSK9", "limit": 1}),
    "clinicaltrials": ("count_clinical_trials", {"condition": "hypercholesterolemia", "intervention": "PCSK9"}),
    "pubmed": ("search_pubmed", {"query": "PCSK9", "max_results": 1}),
    "expression": ("list_available_tissues", {}),
    "functional_genomics": ("query_gene_essentiality", {"gene_id": PCSK9}),
    "genetics": ("query_l2g_predictions", {"gene_id": PCSK9, "limit": 1}),
    "drug": ("search_known_drugs", {"target_id": PCSK9, "limit": 1}),
    "association": ("query_associations", {"output_path": "doctor_smoke_associations.parquet",
                                           "target_id": PCSK9, "limit": 1}),
    "disease": ("search_diseases_by_name", {"query": "hypercholesterolemia", "limit": 1}),
    "interaction": ("get_interactions", {"target_id": PCSK9, "limit": 1}),
    "pathway": ("search_pathways", {"query": "cholesterol", "limit": 1}),
}
SMOKE_CALL_TIMEOUT_S = 600.0

TAHOE_FILES = ("tahoe_permissive_padj010.parquet",)
TAHOE_DIRS = ("pseudobulk_de_significant", "pseudobulk_de_high_quality")
TAHOE_METADATA = ("gene", "drug", "cell_line", "sample")


class DataReadinessError(RuntimeError):
    """Raised before any model call when the session or turn cannot be served."""

    def __init__(self, message: str, results: list["CheckResult"] | None = None):
        super().__init__(message)
        self.results = results or []


@dataclass
class CheckResult:
    label: str
    ok: bool
    hint: str = ""
    detail: str = ""
    required: bool = True          # False: informational/optional, never fails the doctor
    kind: str = "general"          # credentials | data | mcp | analysis | general

    def line(self) -> str:
        mark = "ok" if self.ok else ("!!" if self.required else "--")
        out = f"[{mark}] {self.label}"
        if self.detail:
            out += f": {self.detail}"
        if not self.ok and self.hint:
            out += f"\n     -> {self.hint}"
        return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _servers(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in ((config.get("mcp_servers") or {}).get("servers") or []) if s.get("enabled", True)]


def _env_value(config: dict[str, Any], key: str) -> str:
    """The value MCP servers will see: tool_env first, then the process environment."""
    v = base_tool_env(config).get(key)
    if v is None or not str(v).strip():
        v = os.environ.get(key, "")
    return str(v or "").strip()


def _provider_name(config: dict[str, Any], provider: Any = None) -> str:
    if provider is not None and getattr(provider, "name", None):
        return str(provider.name)
    return str((config.get("provider") or {}).get("name", ""))


_DOCTOR_CACHE: dict[str, Any] = {}


def upstream_doctor(config: dict[str, Any]) -> Any | None:
    """Import the upstream ``tools/doctor.py`` (with the upstream root on sys.path)."""
    up = resolve_path((config.get("vars") or {}).get("upstream") or "third_party/TheVirtualBiotech")
    key = str(up)
    if key in _DOCTOR_CACHE:
        return _DOCTOR_CACHE[key]
    path = up / "tools" / "doctor.py"
    mod = None
    if path.is_file():
        before_path = list(sys.path)
        before_mods = set(sys.modules)
        sys.path.insert(0, key)
        try:
            spec = importlib.util.spec_from_file_location("vbt_upstream_doctor", path)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
        except Exception as exc:  # noqa: BLE001 - fall back to the basic check
            log.debug("upstream doctor import failed: %s", exc)
            mod = None
        finally:
            # The upstream module only needs these at import time; do not leave its
            # `src`/`tools` packages shadowing anything for the rest of the process.
            sys.path[:] = before_path
            for name in set(sys.modules) - before_mods:
                if name.split(".")[0] in ("src", "tools"):
                    sys.modules.pop(name, None)
    _DOCTOR_CACHE[key] = mod
    return mod


def _basic_reference_files(value: str) -> dict[str, list[Path]]:
    root = Path(value)
    if not root.is_dir():
        raise ValueError(f"OPEN_TARGETS_DATA_PATH is not a directory: {root}")
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise ValueError(f"no Parquet files under {root}; rerun the data downloader")
    return {"(all)": files}


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------

def check_credentials(config: dict[str, Any], provider: Any = None) -> CheckResult:
    """Provider credentials. Uses ``provider.check_credentials()`` when available."""
    name = _provider_name(config, provider)
    if name == "mock":
        return CheckResult("model credentials", True, detail="mock provider (no credentials needed)",
                           kind="credentials")
    hook = getattr(provider, "check_credentials", None) if provider is not None else None
    if callable(hook):
        try:
            problem = hook()
        except Exception as exc:  # noqa: BLE001
            problem = f"{type(exc).__name__}: {exc}"
        return CheckResult(f"{name} credentials", not problem, hint=str(problem or ""),
                           detail="" if problem else "configured (model access not tested)", kind="credentials")
    if name == "anthropic":
        key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        token = (os.environ.get("ANTHROPIC_AUTH_TOKEN") or "").strip()
        ok = bool(key or token)
        return CheckResult("ANTHROPIC_API_KEY set", ok,
                           hint="export ANTHROPIC_API_KEY or set it in .env (a blank value counts as missing)",
                           detail="configured (model access not tested)" if ok else "missing or blank",
                           kind="credentials")
    return CheckResult(f"{name} credentials", True, detail="not checked (provider has no check_credentials hook)",
                       required=False, kind="credentials")


def check_open_targets(config: dict[str, Any]) -> CheckResult:
    value = _env_value(config, "OPEN_TARGETS_DATA_PATH")
    hint = ("download the Open Targets 25.09 release: python third_party/TheVirtualBiotech/tools/"
            "download_open_targets.py <dir> --workers 8, then set OPEN_TARGETS_DATA_PATH in .env")
    if not value:
        return CheckResult("Open Targets reference data (OPEN_TARGETS_DATA_PATH)", False, hint=hint,
                           detail="OPEN_TARGETS_DATA_PATH is not set", kind="data")
    doctor = upstream_doctor(config)
    try:
        if doctor is not None and hasattr(doctor, "reference_files"):
            files = doctor.reference_files(value)
            detail = (f"{value}: {len(files)} datasets, {sum(map(len, files.values()))} Parquet files "
                      "(layout and sizes checked)")
        else:
            files = _basic_reference_files(value)
            detail = f"{value}: {len(files['(all)'])} Parquet files (basic check; upstream doctor unavailable)"
    except Exception as exc:  # noqa: BLE001 - the upstream check raises ValueError with a fix
        return CheckResult("Open Targets reference data (OPEN_TARGETS_DATA_PATH)", False, hint=hint,
                           detail=str(exc), kind="data")
    return CheckResult("Open Targets reference data (OPEN_TARGETS_DATA_PATH)", True, detail=detail, kind="data")


def check_tahoe(config: dict[str, Any]) -> CheckResult | None:
    value = _env_value(config, "TAHOE_DATA_PATH")
    if not value:
        return None
    root = Path(value)
    missing = [f for f in TAHOE_FILES if not (root / f).is_file()]
    missing += [d + "/" for d in TAHOE_DIRS if not (root / d).is_dir() or not any((root / d).glob("*.parquet"))]
    missing += [f"metadata/{m}_metadata.parquet" for m in TAHOE_METADATA
                if not (root / "metadata" / f"{m}_metadata.parquet").is_file()]
    hint = ("prepare the Tahoe-100M pseudobulk DE files (third_party/TheVirtualBiotech/docs/TAHOE_SETUP.md, "
            "tools/prepare_tahoe.py) or unset TAHOE_DATA_PATH")
    if missing:
        return CheckResult("Tahoe-100M data (TAHOE_DATA_PATH)", False, hint=hint,
                           detail=f"{value}: missing {', '.join(missing)}", kind="data")
    return CheckResult("Tahoe-100M data (TAHOE_DATA_PATH)", True, detail=value, kind="data")


def check_reference_data(config: dict[str, Any]) -> list[CheckResult]:
    """Reference data needed by the enabled MCP servers (Open Targets; Tahoe when configured)."""
    names = {s.get("name") for s in _servers(config)}
    out: list[CheckResult] = []
    if names & OPEN_TARGETS_SERVERS:
        out.append(check_open_targets(config))
    if "functional_genomics" in names:
        tahoe = check_tahoe(config)
        if tahoe is not None:
            out.append(tahoe)
    return out


def check_mcp_commands(config: dict[str, Any]) -> list[CheckResult]:
    """Each stdio server's interpreter is executable and its script exists."""
    from .tools.mcp_bridge import MCPBridge, MCPServerConfig, _ConfigError

    out = []
    for s in _servers(config):
        cfg = MCPServerConfig(**{k: v for k, v in s.items() if k in MCPServerConfig.__dataclass_fields__})
        try:
            MCPBridge.check_command(cfg)
        except _ConfigError as exc:
            out.append(CheckResult(f"MCP server {cfg.name}: command", False, detail=str(exc),
                                   hint="set VBT_MCP_PYTHON (or vars.mcp_python) to the environment that has "
                                        "the MCP dependencies, and run `git submodule update --init`",
                                   kind="mcp"))
            continue
        target = cfg.url or " ".join([cfg.command or "", *[str(a) for a in cfg.args]])
        out.append(CheckResult(f"MCP server {cfg.name}: command", True, detail=target, kind="mcp"))
    return out


_IMPORT_PROBE = r"""
import importlib, json, sys
out = {}
for m in sys.argv[1:]:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", None)
        if v is None:
            try:
                from importlib.metadata import version
                v = version({"dotenv": "python-dotenv", "cellxgene_census": "cellxgene-census"}.get(m, m))
            except Exception:
                v = "?"
        out[m] = {"ok": True, "version": str(v)}
    except BaseException as e:
        out[m] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
print("VBT_PROBE" + json.dumps(out))
"""


def probe_imports(python: str, modules: Iterable[str], *, flags: Iterable[str] = ("-E",),
                  timeout: float = 300.0) -> dict[str, dict[str, Any]]:
    """Import ``modules`` in a subprocess of ``python``; {module: {ok, version|error}}."""
    modules = list(dict.fromkeys(modules))
    try:
        proc = subprocess.run([python, *flags, "-c", _IMPORT_PROBE, *modules], capture_output=True, text=True,
                              timeout=timeout, env=child_env(os.environ))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {m: {"ok": False, "error": f"could not run {python}: {exc}"} for m in modules}
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("VBT_PROBE")), None)
    if line is None:
        err = (proc.stderr or proc.stdout).strip()[-300:]
        return {m: {"ok": False, "error": f"{python} failed: {err}"} for m in modules}
    return json.loads(line[len("VBT_PROBE"):])


def mcp_modules(config: dict[str, Any]) -> list[str]:
    """Modules the enabled servers launched with vars.mcp_python import."""
    py = (config.get("vars") or {}).get("mcp_python") or sys.executable
    mods: list[str] = []
    for s in _servers(config):
        if s.get("url") or str(s.get("command") or "") != str(py):
            continue
        mods += SERVER_MODULES["_stdio"]
        name = s.get("name")
        if name in OPEN_TARGETS_SERVERS:
            mods += SERVER_MODULES["_opentargets"]
        mods += SERVER_MODULES.get(name, ())
    return list(dict.fromkeys(mods))


def check_mcp_imports(config: dict[str, Any]) -> CheckResult:
    """Run ``<mcp_python> -E -c 'import ...'`` for the modules the enabled servers need."""
    py = (config.get("vars") or {}).get("mcp_python") or sys.executable
    mods = mcp_modules(config)
    label = f"MCP interpreter imports ({py})"
    if not mods:
        return CheckResult(label, True, detail="no stdio servers use vars.mcp_python", required=False, kind="mcp")
    res = probe_imports(py, mods)
    missing = {m: r.get("error", "") for m, r in res.items() if not r.get("ok")}
    fm = res.get("fastmcp", {})
    detail_bits = []
    problems = []
    if fm.get("ok"):
        ver = str(fm.get("version", "?"))
        detail_bits.append(f"fastmcp {ver}")
        try:
            major = int(ver.split(".")[0])
        except ValueError:
            major = None
        if major is not None and major not in FASTMCP_MAJORS:
            problems.append(f"fastmcp major version {major} is untested (expected {FASTMCP_MAJORS})")
    if missing:
        problems.append("missing: " + ", ".join(f"{m} ({e})" for m, e in missing.items()))
    detail = "; ".join(detail_bits + problems) or f"{len(mods)} modules import"
    return CheckResult(label, not problems, detail=detail,
                       hint="use the conda env from environment.yml (conda env create -f environment.yml) or "
                            "pip install -e '.[mcp,singlecell]' into the interpreter named by VBT_MCP_PYTHON",
                       kind="mcp")


def _bash_python() -> str:
    path = child_env(os.environ).get("PATH")
    return shutil.which("python", path=path) or shutil.which("python3", path=path) or sys.executable


def check_analysis_stack(config: dict[str, Any]) -> list[CheckResult]:
    """The Python and R stack agents use from Bash (and the Case 2/3 statistics)."""
    py = _bash_python()
    res = probe_imports(py, ANALYSIS_MODULES, flags=())
    out = []
    for m in ANALYSIS_MODULES:
        r = res.get(m, {})
        out.append(CheckResult(f"analysis: python import {m}", bool(r.get("ok")),
                               detail=(f"{r.get('version')} ({py})" if r.get("ok") else r.get("error", "")),
                               hint="install the conda env from environment.yml (or pip install -e '.[full]')",
                               kind="analysis"))
    rscript = shutil.which("Rscript", path=child_env(os.environ).get("PATH"))
    if not rscript:
        out.append(CheckResult("analysis: Rscript", False, detail="Rscript not found on PATH",
                               hint="install R (r-base 4.4) with lme4, lmerTest, glmmTMB, betareg and MuMIn "
                                    "(environment.yml)", kind="analysis"))
        return out
    code = ("pk <- c(" + ",".join(f'"{p}"' for p in R_PACKAGES) + "); "
            "ok <- sapply(pk, requireNamespace, quietly=TRUE); cat(paste0('VBT_R ', pk, '=', ok, '\\n'))")
    try:
        proc = subprocess.run([rscript, "-e", code], capture_output=True, text=True, timeout=300,
                              env=child_env(os.environ))
        found = dict(ln[6:].split("=", 1) for ln in proc.stdout.splitlines() if ln.startswith("VBT_R "))
    except (OSError, subprocess.TimeoutExpired) as exc:
        found = {}
        out.append(CheckResult("analysis: Rscript runs", False, detail=str(exc), kind="analysis"))
    for p in R_PACKAGES:
        out.append(CheckResult(f"analysis: R package {p}", found.get(p) == "TRUE",
                               detail=rscript if found.get(p) == "TRUE" else "not installed",
                               hint=f"conda install -c conda-forge r-{p.lower()}", kind="analysis"))
    return out


async def smoke_mcp(config: dict[str, Any], *, log_dir: str | os.PathLike | None = None,
                    servers: Iterable[str] | None = None) -> list[CheckResult]:
    """Start every enabled MCP server and make one cheap call per server.

    A server fails the smoke test when it does not start, advertises no tools,
    or its smoke call returns an error (isError or a legacy failure envelope).
    """
    from .tools.base import ToolFailure
    from .tools.mcp_bridge import MCPBridge, MCPServerConfig

    specs_raw = [s for s in _servers(config) if servers is None or s.get("name") in set(servers)]
    specs = [MCPServerConfig(**{k: v for k, v in s.items() if k in MCPServerConfig.__dataclass_fields__})
             for s in specs_raw]
    tmp = tempfile.mkdtemp(prefix="vbt-doctor-")
    extra = base_tool_env(config)
    extra.update({"VBT_RUN_DIR": tmp, "MCP_OUTPUT_DIR": str(Path(tmp) / "mcp")})
    bridge = MCPBridge(specs, extra_env=extra, log_dir=log_dir or Path(tmp) / "logs",
                       options=config.get("mcp") or {})
    results: list[CheckResult] = []
    try:
        await bridge.start()
        status = bridge.status()
        for raw, spec in zip(specs_raw, specs):
            name = spec.name
            st = status.get(name, {})
            if st.get("state") != "ready":
                results.append(CheckResult(f"MCP {name}: starts", False, kind="mcp",
                                           detail=(bridge.failures.get(name) or "not started")[:1500],
                                           hint=f"see the server log {st.get('log') or ''}".strip()))
                continue
            if not st.get("tools"):
                results.append(CheckResult(f"MCP {name}: starts", False, detail="server advertised no tools",
                                           kind="mcp"))
                continue
            results.append(CheckResult(f"MCP {name}: starts", True, kind="mcp",
                                       detail=f"{st['tools']} tools in {st.get('start_duration_s')}s"))
            smoke = raw.get("smoke", None)
            if smoke is False:
                continue
            if isinstance(smoke, dict) and smoke.get("tool"):
                tool, args = smoke["tool"], dict(smoke.get("args") or {})
            elif name in SMOKE_CALLS:
                tool, args = SMOKE_CALLS[name]
            else:
                continue
            label = f"MCP {name}: {tool}({', '.join(f'{k}={v!r}' for k, v in args.items())})"
            try:
                timeout = min(float(spec.timeout_s or SMOKE_CALL_TIMEOUT_S), SMOKE_CALL_TIMEOUT_S)
                out = await asyncio.wait_for(bridge.call(name, tool, args), timeout + 5)
                text = out if isinstance(out, str) else str(out)
                results.append(CheckResult(label, True, detail=text.replace("\n", " ")[:160], kind="mcp"))
            except ToolFailure as exc:
                results.append(CheckResult(label, False, detail=str(exc)[:1500], kind="mcp",
                                           hint="the tool returned an error: check the reference data and the "
                                                "server log"))
            except Exception as exc:  # noqa: BLE001
                results.append(CheckResult(label, False, detail=f"{type(exc).__name__}: {exc}"[:1500], kind="mcp"))
    finally:
        await bridge.aclose()
        if log_dir:  # otherwise the server logs stay in `tmp` for the hints above
            shutil.rmtree(tmp, ignore_errors=True)
    return results


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def require_ready(config: dict[str, Any], *, per_turn: bool = False, provider: Any = None,
                  allow_missing_data: bool = False, start_mcp: bool = True) -> list[CheckResult]:
    """Raise :class:`DataReadinessError` unless the session/turn can be served.

    Skipped (returns []) for the mock provider or when
    ``orchestration.require_reference_data`` is false. Missing credentials
    always raise; missing reference data or broken MCP commands raise unless
    ``allow_missing_data`` (the run is then degraded: the caller should mark it
    and tell the CSO which servers lack data). With ``start_mcp=False``
    (``--no-mcp``: no MCP server is started) only the credentials are checked,
    since the reference data and server commands are needed only by the
    servers. Returns the check results.
    """
    if _provider_name(config, provider) == "mock":
        return []
    if not (config.get("orchestration") or {}).get("require_reference_data", True):
        return []
    results = [check_credentials(config, provider)]
    if start_mcp:
        results += check_reference_data(config)
        if not per_turn:
            results += check_mcp_commands(config)
    failed = [r for r in results if r.required and not r.ok]
    blocking = [r for r in failed if r.kind == "credentials" or not allow_missing_data]
    if blocking:
        lines = "; ".join(f"{r.label}: {r.detail or 'failed'}" + (f" (fix: {r.hint})" if r.hint else "")
                          for r in blocking)
        raise DataReadinessError(f"Not ready: {lines}. {TURN_NOT_SENT}", results)
    return results


def degraded_servers(config: dict[str, Any], results: Iterable[CheckResult]) -> dict[str, str]:
    """Servers that lack their reference data, given ``require_ready`` results ({server: reason})."""
    out: dict[str, str] = {}
    names = {s.get("name") for s in _servers(config)}
    for r in results:
        if r.ok or r.kind != "data":
            continue
        if "OPEN_TARGETS" in r.label:
            for n in sorted(names & OPEN_TARGETS_SERVERS):
                out[n] = f"Open Targets data unavailable: {r.detail}"
        elif "Tahoe" in r.label and "functional_genomics" in names:
            out.setdefault("functional_genomics", f"Tahoe-100M data unavailable: {r.detail}")
    return out


def check_bash_network(config: dict[str, Any]) -> CheckResult | None:
    """With ``bash.network: false``, report whether Bash commands are actually network-isolated."""
    bash = config.get("bash") or {}
    if bash.get("network", True) or not bash.get("enabled", True):
        return None
    from .tools.builtin import network_isolation_status
    ok, why = network_isolation_status(config)
    return CheckResult("Bash network isolation", ok, detail=why, required=False,
                       hint="set bash.network_isolation: unshare and run where `unshare -rn true` works (or "
                            "bash.sandbox.os: bwrap); otherwise bash.network: false is only a pattern guardrail")


# ---------------------------------------------------------------------------
# vbt doctor
# ---------------------------------------------------------------------------

def run_doctor(config: dict[str, Any], *, smoke: bool = False, analysis: bool = False,
               out: Callable[[str], Any] = print) -> int:
    """Print an installation report; return 0 when every required check passes."""
    results: list[CheckResult] = []

    def add(r: CheckResult | list[CheckResult] | None) -> None:
        for x in ([r] if isinstance(r, CheckResult) else (r or [])):
            results.append(x)
            out(x.line())

    up = resolve_path((config.get("vars") or {}).get("upstream") or "third_party/TheVirtualBiotech")
    out("The Virtual Biotech harness -- environment check")
    out(f"  python:   {sys.executable} ({sys.version.split()[0]})")
    out(f"  upstream: {up}")
    for f in env_files(config):
        out(f"  env file: {f} ({'loaded' if f.is_file() else 'absent'})")
    add(CheckResult("upstream submodule present", (up / "src" / "agents" / "cso" / "system_prompt.md").exists(),
                    hint="git submodule update --init"))
    try:
        from .agents import load_roster
        _, agents = load_roster(config)
        add(CheckResult("agent roster loads", True, detail=f"{len(agents)} agents + CSO"))
    except Exception as exc:  # noqa: BLE001
        add(CheckResult("agent roster loads", False, detail=str(exc)))
    try:
        from .providers import create_provider
        provider = create_provider(config["provider"]["name"], **(config["provider"].get("options") or {}))
    except Exception:  # noqa: BLE001 - fall back to the env-var check
        provider = None
    add(check_credentials(config, provider))
    add(check_reference_data(config))
    add(CheckResult("upstream clinical-trial labels present",
                    (up / "datasets" / "clinical_trials" / "clinical_trial_labels_reconciled.csv").exists(),
                    hint="git submodule update --init", required=False))
    add(check_mcp_commands(config))
    add(check_bash_network(config))
    if _servers(config):
        add(check_mcp_imports(config))
    if analysis:
        add(check_analysis_stack(config))
    if smoke:
        if _servers(config):
            add(asyncio.run(smoke_mcp(config)))
        else:
            add(CheckResult("MCP smoke test", True, detail="no MCP servers enabled", required=False))
    failed = [r for r in results if r.required and not r.ok]
    out("PASS" if not failed else f"FAIL -- {len(failed)} check(s) failed; fix the items marked [!!].")
    return 0 if not failed else 1


def _doctor_handler(args: Any, config: dict[str, Any]) -> int:
    return run_doctor(config, smoke=bool(getattr(args, "smoke", False)),
                      analysis=bool(getattr(args, "analysis", False)))


def add_doctor_parser(sub: Any) -> Any:
    """Register ``vbt doctor`` on an argparse subparsers object."""
    d = sub.add_parser("doctor", help="check installation, credentials, reference data and MCP servers")
    d.add_argument("--smoke", action="store_true",
                   help="also start every MCP server and make one cheap call each (fails on tool errors)")
    d.add_argument("--analysis", action="store_true",
                   help="also check the Python/R analysis stack (scanpy, pydeseq2, rpy2, lme4, glmmTMB, ...)")
    d.set_defaults(handler=_doctor_handler)
    return d


__all__ = [
    "CheckResult", "DataReadinessError", "TURN_NOT_SENT", "add_doctor_parser", "check_analysis_stack",
    "check_credentials", "check_mcp_commands", "check_mcp_imports", "check_reference_data",
    "degraded_servers", "require_ready", "run_doctor", "smoke_mcp", "upstream_doctor",
]
