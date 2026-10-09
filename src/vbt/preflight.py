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
  stack. For local model servers (``vllm``, ``sglang``, ...) the credentials
  check is offline (a base URL is configured) and one bounded ``GET /health``
  says whether the server is running at all; ``--smoke`` also runs
  ``provider.prepare()`` (``/health``, the configured models are served) and
  compares the server's ``max_model_len`` with the configured context windows,
  and runs one query through the WebSearch backend.
* :func:`prepare_provider` / :func:`provider_server_info` -- used by
  ``open_session``: the model server's readiness check before a run starts
  (:class:`ProviderNotReadyError`) and its facts for the pinned config.

With the data layer (``data.enabled``, gateway mode not ``off``; docs/DATA_LAYER.md
§13) reference-data readiness is scoped to tools: :func:`check_reference_data`
runs the data child's ``server.py --check --json`` once as a subprocess and
returns one result per unready table, column or partition (``scope``) with the
tools it makes unready; :func:`degraded_tools` and :func:`degraded_servers`
(only servers whose every tool is unready) feed ``open_session``, and the
session blocks on data only when no granted data tool is ready. Without the
data layer (or when the data child cannot run) the Open Targets check reuses
the upstream ``tools/doctor.py::reference_files`` (layout, truncated files,
partial downloads, download manifest).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

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
#: Third-party modules the data-layer child (src/vbt/datalayer/service/server.py) imports.
DATA_CHILD_MODULES = ("pydantic", "yaml", "pyarrow")

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


class ProviderNotReadyError(DataReadinessError):
    """The model server is not ready (``provider.prepare()`` failed): not reachable,
    unhealthy, or the configured model is not served. Raised by ``open_session``
    before a run directory exists; the message carries the provider's fix hint."""


#: Provider names served by an OpenAI-compatible inference server (vbt.providers.openai_compat).
LOCAL_PROVIDERS = frozenset({"vllm", "sglang", "openai_compat", "llamacpp"})


def provider_options(config: dict[str, Any]) -> dict[str, Any]:
    """``provider.options`` without null values: profiles are deep-merged, so a
    profile that switches provider (``--profile claude`` over the local default)
    resets the other adapter's options to null, and null means "not set"."""
    opts = (config.get("provider") or {}).get("options") or {}
    return {k: v for k, v in opts.items() if v is not None}


def create_configured_provider(config: dict[str, Any]) -> Any:
    """The configured provider (``provider.name`` + non-null ``provider.options``)."""
    from .providers import create_provider
    return create_provider(config["provider"]["name"], **provider_options(config))


def has_prepare(provider: Any) -> bool:
    """The provider overrides ``LLMProvider.prepare`` (a real readiness check)."""
    if provider is None:
        return False
    try:
        from .providers.base import LLMProvider
        fn = getattr(type(provider), "prepare", None)
        return fn is not None and fn is not LLMProvider.prepare
    except Exception:  # noqa: BLE001
        return False


def configured_models(config: dict[str, Any]) -> list[str]:
    """Every model id the tiers and agent overrides name (what the server must serve)."""
    out: list[str] = []
    for t in (config.get("models") or {}).values():
        if isinstance(t, dict) and t.get("model"):
            out.append(str(t["model"]))
    for o in (config.get("agent_overrides") or {}).values():
        if isinstance(o, dict) and o.get("model"):
            out.append(str(o["model"]))
    return list(dict.fromkeys(out))


def configured_window(config: dict[str, Any]) -> int | None:
    """The largest ``models.<tier>.context_window_tokens`` (None when no tier sets one)."""
    wins = []
    for t in (config.get("models") or {}).values():
        if isinstance(t, dict):
            try:
                if t.get("context_window_tokens"):
                    wins.append(int(t["context_window_tokens"]))
            except (TypeError, ValueError):
                continue
    return max(wins) if wins else None


async def prepare_provider(provider: Any, config: dict[str, Any], *, wait_s: float | None = None) -> None:
    """``await provider.prepare()`` (with the configured models when the provider
    takes them). Raises :class:`ProviderNotReadyError` with the provider's text on
    a ``ProviderError``; providers without a real ``prepare`` are a no-op."""
    if not has_prepare(provider):
        return
    import inspect

    from .providers.base import ProviderError
    kwargs: dict[str, Any] = {}
    try:
        params = inspect.signature(provider.prepare).parameters
    except (TypeError, ValueError):
        params = {}
    if "models" in params:
        kwargs["models"] = configured_models(config) or None
    if wait_s is not None and "wait_s" in params:
        kwargs["wait_s"] = wait_s
    try:
        await provider.prepare(**kwargs)
    except ProviderError as exc:
        name = getattr(provider, "name", None) or (config.get("provider") or {}).get("name")
        raise ProviderNotReadyError(f"model server not ready ({name}): {exc}") from exc


async def provider_server_info(provider: Any, *, timeout: float = 45.0) -> dict[str, Any]:
    """``provider.server_info()`` bounded by ``timeout``; ``{}`` on any failure."""
    fn = getattr(provider, "server_info", None)
    if not callable(fn):
        return {}
    try:
        info = await asyncio.wait_for(fn(), timeout)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - informational only
        return {"error": f"{type(exc).__name__}: {exc}"[:500]}
    return dict(info) if isinstance(info, dict) else {}


@dataclass
class CheckResult:
    label: str
    ok: bool
    hint: str = ""
    detail: str = ""
    required: bool = True          # False: informational/optional, never fails the doctor
    kind: str = "general"          # credentials | data | mcp | analysis | general
    # Data-layer results only: what the result is about ({source, table, column?, partition?} for a
    # readiness finding, {source} for an aggregate, {granted, ready, bound} for the tools summary).
    # None for the legacy checks, which keep their old blocking semantics.
    scope: dict[str, Any] | None = None
    tools: dict[str, str] = field(default_factory=dict)   # tools this finding makes unready {tool: reason}

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
        before_bytecode = sys.dont_write_bytecode
        sys.path.insert(0, key)
        # Never write __pycache__ into the upstream checkout (it stays byte-identical).
        sys.dont_write_bytecode = True
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
            sys.dont_write_bytecode = before_bytecode
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
        raise ValueError(f"no Parquet files under {root}; vbt data acquire open_targets fetches them")
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
    hint = OPEN_TARGETS_HINT
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
    hint = TAHOE_HINT
    if missing:
        return CheckResult("Tahoe-100M data (TAHOE_DATA_PATH)", False, hint=hint,
                           detail=f"{value}: missing {', '.join(missing)}", kind="data")
    return CheckResult("Tahoe-100M data (TAHOE_DATA_PATH)", True, detail=value, kind="data")


def check_reference_data(config: dict[str, Any], *, per_turn: bool = False) -> list[CheckResult]:
    """Reference data needed by the enabled MCP servers.

    With the data layer (``data.enabled`` and a gateway mode other than ``off``) readiness is
    scoped to each tool (§13): the data child's ``--check`` runs once as a subprocess and every
    unready part (table, column, container, partition) becomes one result with ``scope`` and the
    tools it makes unready; one summary per source (``data source: <id>``) fails only when no granted
    tool reading that source is ready. ``per_turn`` reuses the session's
    cached results and re-checks only tables whose stat-only signature moved. Without the data
    layer, or when the data child cannot run, the legacy whole-release checks apply (with a note).
    A descriptor or overlay file that does not load is quarantined on its own (R8): one required
    ``data catalog`` result names each file, its error and the servers and tools it refuses, and the
    rest of the catalog is checked as usual.
    """
    if not _servers(config):
        return []
    if data_layer_active(config):
        quarantine = _catalog_quarantine(config)
        try:
            return [*check_data_readiness(config, per_turn=per_turn), *quarantine, *_leakage_ceiling(config)]
        except DataCatalogError as exc:
            # the catalog itself is broken: the gateway cannot guard any server, which is not a fallback
            fail = CheckResult("data catalog", False, required=True, kind="data",
                               detail=f"the data catalog cannot be loaded: {exc}",
                               hint="fix the descriptor or overlay (`vbt ds lint` names the file and field)")
            return [*_legacy_reference_data(config), fail]
        except DataCheckUnavailable as exc:
            note = CheckResult("data layer readiness (tool-scoped)", False, required=False, kind="data",
                               detail=f"the data child's check could not run ({exc}); falling back to the "
                                      "whole-release checks",
                               hint="check vars.mcp_python has pyarrow and the vbt data-layer dependencies "
                                    "(`vbt doctor` lists the data child's imports)")
            return [*_legacy_reference_data(config), *quarantine, note]
    return _legacy_reference_data(config)


def _catalog_quarantine(config: dict[str, Any]) -> list[CheckResult]:
    """One required ``data catalog`` result when descriptor or overlay files are quarantined (R8): the files
    and their errors, the enabled servers refused (their overlay) and the tools refused (they read a table or
    id_type of a quarantined descriptor). Tool-scoped (``scope``), so it blocks no session on its own."""
    try:
        _settings, catalog, _registry = data_catalog(config)
    except Exception:  # noqa: BLE001 - a catalog that cannot be built at all is reported by the caller
        return []
    files = list(getattr(catalog, "quarantined", None) or [])
    if not files:
        return []
    names = sorted({str(s.get("name")) for s in _servers(config)})
    servers = sorted(n for n in names if n in catalog.quarantined_servers())
    tools = {_tool_name(*name.split(".", 1)): "quarantined: " + "; ".join(f"{q.file}: {q.summary}" for q in qs)
             for name, qs in catalog.quarantined_tools(names).items()}
    detail = (f"{len(files)} file(s) do not load and are quarantined (the rest of the catalog is served): "
              + "; ".join(f"{q.path}: {q.summary}" for q in files))
    if servers:
        detail += f" -- servers refused: {', '.join(servers)}"
    if tools:
        short = sorted(t.split("__", 2)[-1] for t in tools)
        detail += f" -- tools refused: {', '.join(short[:8])}" + (f" (+{len(short) - 8})" if len(short) > 8 else "")
    return [CheckResult("data catalog", False, required=True, kind="data", detail=detail[:1500],
                        hint="fix the file (`vbt ds lint` names the file and field)",
                        scope={"quarantined": [q.to_json() for q in files], "servers": servers}, tools=tools)]


def _leakage_ceiling(config: dict[str, Any]) -> list[CheckResult]:
    """A no-web run (``web.enabled: false``) whose sources can return records past the evidence ceiling
    (a descriptor declares ``leakage``) while ``data.leakage.ceiling`` is unset: the ceiling is not applied."""
    if (config.get("web") or {}).get("enabled", True) is not False:
        return []
    if ((config.get("data") or {}).get("leakage") or {}).get("ceiling"):
        return []
    from .datalayer.descriptor.models import DATA_CEILING

    try:
        _settings, catalog, _registry = data_catalog(config)
    except Exception:  # noqa: BLE001 - the catalog's own failure is reported elsewhere
        return []
    # a source whose server applies its own ceiling (PubMed: web.literature_max_date) is not bounded by this one
    leaky = sorted(src for src, d in catalog.sources.items() if getattr(d, "leakage", None) is not None and
                   getattr(d.leakage, "ceiling_from", DATA_CEILING) == DATA_CEILING)
    if not leaky:
        return []
    return [CheckResult("data: evidence ceiling", False, required=False, kind="data",
                        detail=f"web is disabled but data.leakage.ceiling is unset: {', '.join(leaky)} can return "
                               "records past the evidence ceiling",
                        hint="set data.leakage.ceiling (ISO date, e.g. the web.literature_max_date)")]


def _legacy_reference_data(config: dict[str, Any]) -> list[CheckResult]:
    """The pre-datalayer checks: whole-release layout of Open Targets (and Tahoe when configured)."""
    names = {s.get("name") for s in _servers(config)}
    out: list[CheckResult] = []
    if names & OPEN_TARGETS_SERVERS:
        out.append(check_open_targets(config))
    if "functional_genomics" in names:
        tahoe = check_tahoe(config)
        if tahoe is not None:
            out.append(tahoe)
    return out


# ---------------------------------------------------------------------------
# tool-scoped data readiness (the data layer, docs/DATA_LAYER.md §13)
# ---------------------------------------------------------------------------

#: The data child's entry point (``--check --json`` runs its readiness checks without MCP).
DATA_SERVICE_SCRIPT = ("src", "vbt", "datalayer", "service", "server.py")
DATA_CHECK_TIMEOUT_S = 1800.0
DATA_TOOLS_LABEL = "data tools ready"
# the whole-release checks' labels (no data layer, or its check cannot run); with the data layer every source has
# the same per-source summary (source_label)
OPEN_TARGETS_LABEL = "Open Targets reference data (OPEN_TARGETS_DATA_PATH)"
TAHOE_LABEL = "Tahoe-100M data (TAHOE_DATA_PATH)"
# the acquisition engine's commands (the not_ready payload names the same; DEP-16): they fetch the pinned release
# into the acquisition home and write the root variable
OPEN_TARGETS_HINT = ("fetch the pinned Open Targets release: vbt data acquire open_targets --env-file .env "
                     "(or --for-agents to fetch only what the enabled agents read)")
TAHOE_HINT = ("fetch and prepare Tahoe-100M (about 83 GiB): vbt data acquire tahoe_100m --env-file .env, "
              "or unset TAHOE_DATA_PATH (Tahoe is optional)")
#: Sources whose data is optional: unset, their findings are informational (as the legacy Tahoe check).
OPTIONAL_SOURCES = {"tahoe_100m": "TAHOE_DATA_PATH"}
_READY = frozenset({"ready", "awaiting_producer", "unbound"})
_STATUS_HINTS = {
    "missing": "the table's files are absent: `vbt data acquire {source}.{table} --env-file <file>` fetches them "
               "(then rerun `vbt ds check`)",
    "partial": "a partial download or unreadable fragment: `vbt data acquire {source}.{table}` completes it, then "
               "rerun `vbt ds check`",
    "schema_drift": "the columns differ from the descriptor: update the descriptor or the data",
    "encoding_drift": "stored codes differ from the descriptor's encoding",
    "key_violation": "the declared key is not unique or has nulls",
    "stale": "the release differs from the one the descriptor pins: `vbt data acquire {source}.{table}` fetches it",
}


def _status_hint(status: str, source: str, table: str) -> str:
    """The fix for a table status, naming the acquisition command for the table (DEP-16)."""
    return _STATUS_HINTS.get(status, "run `vbt ds check`").format(source=source, table=table)
#: Tables whose ``_check`` failed in this process, with their signature (a per-turn check skips them).
_CHECK_ERRORS: dict[tuple[str, str], tuple[str | None, str]] = {}


#: The last full ``_check`` response per cache directory (``open_session`` hands it to the gateway).
_LAST_CHECK: dict[str, dict[str, Any]] = {}


def last_data_check(config: dict[str, Any]) -> dict[str, Any] | None:
    """The ``CheckResponse`` JSON of this process's last session-start data check for ``config``."""
    try:
        from .datalayer.settings import DataSettings
        return _LAST_CHECK.get(str(DataSettings.from_config(config).cache_dir))
    except Exception:  # noqa: BLE001
        return None


class DataCheckUnavailable(RuntimeError):
    """The data child's readiness check could not run (no script, no interpreter, a crash)."""


class DataCatalogError(DataCheckUnavailable):
    """The descriptors or overlays cannot be loaded (missing directory, malformed YAML, a schema error)."""


def data_layer_active(config: dict[str, Any]) -> bool:
    """``data.enabled`` with a gateway mode other than ``off``: readiness is scoped to tools."""
    try:
        from .datalayer.settings import DataSettings
        settings = DataSettings.from_config(config)
    except Exception:  # noqa: BLE001 - a broken data section falls back to the legacy checks
        log.debug("data settings unreadable", exc_info=True)
        return False
    return bool(settings.enabled) and settings.gateway.mode != "off"


def data_child_command(config: dict[str, Any], *args: str) -> tuple[list[str], dict[str, str]]:
    """``(argv, env)`` of the data child's command line (``<mcp_python> -E server.py ...``)."""
    from .datalayer.settings import SETTINGS_ENV, DataSettings

    settings = DataSettings.from_config(config)
    python = (config.get("vars") or {}).get("mcp_python") or sys.executable
    script = Path(settings.project_root).joinpath(*DATA_SERVICE_SCRIPT)
    if not script.is_file():
        raise DataCheckUnavailable(f"the data child script {script} is missing")
    env = child_env(os.environ, extra={**base_tool_env(config), SETTINGS_ENV: settings.to_json()})
    return [str(python), "-E", str(script), *args], env


def run_contained(config: Mapping[str, Any], argv: list[str], env: Mapping[str, str], *,
                  timeout: float) -> subprocess.CompletedProcess[str]:
    """Run a data-child command line under the reaper with the data child's memory limit
    (``data.service.mem_limit_mb``, ``data.memory.limit_kind``), as the bridge launches it: decoding
    reference data never runs unlimited in, or next to, the harness (I12). A deep ``--check`` of a 27M-row
    table reached 10.3 GiB resident when it ran uncontained. Where the reaper is not available (not Linux,
    ``limit_kind: none`` aside) the command runs as given."""
    import tempfile

    from .datalayer.launch import DATA_SERVER, build_launch_spec
    from .datalayer.settings import DataSettings
    from .tools.mcp_bridge import MCPServerConfig

    with tempfile.TemporaryDirectory(prefix="vbt-ds-") as status_dir:
        try:
            spec = build_launch_spec(MCPServerConfig(DATA_SERVER, command=argv[0], args=list(argv[1:])),
                                     DataSettings.from_config(dict(config)), status_dir)
        except Exception:  # noqa: BLE001 - run unguarded rather than not at all
            spec = None
        if spec is not None:
            argv, env = [spec.command, *spec.args], {**env, **spec.env}
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=dict(env))


#: A local table whose files take at least this much disk is checked in a data child of its own (see
#: :func:`check_groups`).
CHECK_ISOLATE_BYTES = 32 * 1024 * 1024


def _local_bytes(catalog: Any, ref: str) -> int:
    """Bytes on disk of a local table's files (an item table: its parent's), from ``stat`` alone; 0 when remote or
    absent."""
    try:
        t = catalog.table(ref)
        if t.is_item_table:
            t = catalog.table(str(t.physical))
        root = getattr(catalog.source(t.ref.source), "root", None)
        path = t.physical_spec.path
    except Exception:  # noqa: BLE001
        return 0
    if not root or not path:
        return 0
    base = Path(str(root)) / str(path)
    total = 0
    try:
        if base.is_file():
            return base.stat().st_size
        for dirpath, _dirs, files in os.walk(base):
            for name in files:
                try:
                    total += (Path(dirpath) / name).stat().st_size
                except OSError:
                    continue
    except OSError:
        return 0
    return total


def check_groups(config: dict[str, Any], tables: Iterable[str] = ()) -> list[list[str]]:
    """The tables of one session check, split into the data-child processes that check them: every table whose
    files take :data:`CHECK_ISOLATE_BYTES` or more (with its item tables) in a process of its own, the rest
    together. One process checking every table kept what each check cached (readers, footers, vocabulary
    snapshots) until the end: on the 31 Open Targets 25.09 tables its peak was 2,681 MB, against 2,233 MB for the
    largest single table (interaction). ``[[]]`` (every table, one process) when the catalog does not load."""
    try:
        _settings, catalog, _registry = data_catalog(config)
        refs = [str(r) for r in (tables or catalog.table_refs())]
    except Exception:  # noqa: BLE001 - the child reports a broken catalog itself
        return [list(tables)]
    big: dict[str, list[str]] = {}
    rest: list[str] = []
    for ref in refs:
        try:
            t = catalog.table(ref)
            parent = str(t.physical) if t.is_item_table else ref
        except Exception:  # noqa: BLE001
            rest.append(ref)
            continue
        if _local_bytes(catalog, parent) >= CHECK_ISOLATE_BYTES:
            big.setdefault(parent, []).append(ref)
        else:
            rest.append(ref)
    groups = [sorted(g) for _parent, g in sorted(big.items())]
    if rest:
        groups.append(rest)
    return groups or [[]]


def _merge_checks(responses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"tables": {}, "table_errors": {}, "quarantined": []}
    seen: set[str] = set()
    for r in responses:
        out["tables"].update(r.get("tables") or {})
        out["table_errors"].update(r.get("table_errors") or r.get("errors") or {})
        for q in r.get("quarantined") or []:
            key = json.dumps(q, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                out["quarantined"].append(q)
        for k, v in r.items():
            if k not in ("tables", "table_errors", "errors", "quarantined"):
                out.setdefault(k, v)
    return out


def run_data_check(config: dict[str, Any], *, tables: Iterable[str] = (), depth: str | None = None,
                   timeout: float = DATA_CHECK_TIMEOUT_S) -> dict[str, Any]:
    """Run ``server.py --check --json`` under the data child's memory limit and return its ``CheckResponse``
    JSON: one process per group of :func:`check_groups` (the large tables each in their own), merged."""
    from .datalayer.settings import DataSettings

    depth = depth or DataSettings.from_config(config).readiness.session_depth
    groups = check_groups(config, tables)
    if len(groups) == 1:
        return _run_check_once(config, groups[0], depth, timeout)
    deadline = time.monotonic() + timeout
    responses: list[dict[str, Any]] = []
    failed: list[str] = []
    for group in groups:
        try:
            responses.append(_run_check_once(config, group, depth, max(60.0, deadline - time.monotonic())))
        except DataCheckUnavailable as exc:
            # one group's child failing (a table killed at the data child's limit) never hides the other groups'
            # results: its tables carry the error, as a table the child could not check does
            failed.append(str(exc))
            responses.append({"tables": {}, "table_errors": {str(t): f"the data child checking it failed: {exc}"[:800]
                                                             for t in group}})
    if len(failed) == len(groups):
        raise DataCheckUnavailable(failed[0])
    return _merge_checks(responses)


def _run_check_once(config: dict[str, Any], tables: Sequence[str], depth: str, timeout: float) -> dict[str, Any]:
    args = ["--check", "--json", "--depth", depth]
    for t in tables:
        args += ["--table", str(t)]
    cmd, env = data_child_command(config, *args)
    try:
        proc = run_contained(config, cmd, env, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DataCheckUnavailable(f"could not run {cmd[0]}: {exc}") from exc
    line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("{")), None)
    if proc.returncode != 0 or line is None:
        err = (proc.stderr or proc.stdout).strip()[-500:]
        raise DataCheckUnavailable(f"{cmd[0]} exited {proc.returncode}: {err}")
    try:
        return json.loads(line)
    except ValueError as exc:
        raise DataCheckUnavailable(f"unreadable --check output: {exc}") from exc


def _tool_name(server: str, tool: str) -> str:
    return f"mcp__{server}__{tool}"


def granted_tools(config: dict[str, Any], tools: Iterable[str]) -> set[str]:
    """The ``tools`` some agent (or the CSO) may call; all of them when the roster cannot be loaded."""
    tools = list(tools)
    try:
        from .agents import load_roster
        cso, agents = load_roster(config)
    except Exception:  # noqa: BLE001 - without a roster every tool counts as granted
        return set(tools)
    roster = [cso, *agents.values()]
    return {t for t in tools if any(a.has_tool(t) for a in roster)}


@dataclass
class DataReadiness:
    """Tool-scoped readiness of the enabled servers, from one set of ``_check`` results.

    ``unready``: tools every call of which is unready, with the reason (``{table, column?, check,
    detail, status}``); ``partial``: tools that only some partitions block (``{tool: [labels]}``);
    ``bound``: ``{server: [tools with a reviewed, servable binding]}``; ``reads``: the physical
    tables each tool may read; ``errors``: tables whose check failed (left unchecked)."""

    tables: dict[str, Any] = field(default_factory=dict)          # ref -> TableCheckModel
    errors: dict[str, str] = field(default_factory=dict)
    unready: dict[str, dict[str, Any]] = field(default_factory=dict)
    partial: dict[str, list[str]] = field(default_factory=dict)
    bound: dict[str, list[str]] = field(default_factory=dict)
    reads: dict[str, set[str]] = field(default_factory=dict)
    granted: set[str] = field(default_factory=set)
    awaiting: set[str] = field(default_factory=set)               # tables a tool writes during the run

    @property
    def used(self) -> set[str]:
        return set().union(*self.reads.values()) if self.reads else set()

    def reason(self, tool: str) -> str:
        r = self.unready.get(tool)
        if not r:
            return ""
        where = r["table"] + (f".{r['column']}" if r.get("column") else "")
        return f"{where} {r.get('status') or 'not ready'} ({r.get('check')}: {r.get('detail')})"[:300]


_CATALOGS: dict[str, tuple[Any, Any, Any]] = {}
_ENV_REFS: dict[tuple[str, int], tuple[str, ...]] = {}
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)")


def _env_refs(path: Path, mtime_ns: int) -> tuple[str, ...]:
    """The environment variables a descriptor or overlay file expands (``${NAME}``, ``${NAME:-x}``)."""
    key = (str(path), mtime_ns)
    if key not in _ENV_REFS:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            text = ""
        _ENV_REFS[key] = tuple(sorted({m for m in _ENV_REF.findall(text) if m != "vars"}))
    return _ENV_REFS[key]


def _mtime(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def data_catalog(config: dict[str, Any]) -> tuple[Any, Any, Any]:
    """``(settings, catalog, registry)`` of the config's data layer (no pyarrow). Kept per process
    while the settings, variables and descriptor/overlay files are unchanged (the per-turn check): the shipped ones
    and the active project's, with its approved plugin modules and their provenance records."""
    from .datalayer.catalog import build_catalog
    from .datalayer.descriptor.load import (
        PROJECT_DESCRIPTORS,
        PROJECT_OVERLAYS,
        approved_project_plugins,
        project_search_dirs,
        variables_from_config,
    )
    from .datalayer.plugins.registry import discover
    from .datalayer.settings import DataSettings

    settings = DataSettings.from_config(config)
    variables = variables_from_config(config)
    for what, d in (("descriptors_dir", settings.descriptors_dir), ("overlays_dir", settings.overlays_dir)):
        # a missing or empty catalog would make every tool look ready and silence the legacy checks
        if not Path(d).is_dir() or not any(Path(d).glob("*.y*ml")):
            raise DataCatalogError(f"data.{what} {d} is missing or holds no YAML files: the data catalog "
                                       "cannot be loaded")
    stamps = []
    env: dict[str, str | None] = {}
    dirs = [settings.descriptors_dir, settings.overlays_dir]
    project = Path(settings.project_dir) if settings.project_dir else None
    if project is None:
        found = project_search_dirs(variables)
        project = found[0].parent if found else None
    if project is not None:
        # a registration made in this process (RegisterDataSpec, RegisterPlugin) is a different catalog
        dirs += [project / PROJECT_DESCRIPTORS, project / PROJECT_OVERLAYS]
        stamps += [(f, _mtime(Path(f))) for f in approved_project_plugins(project)[0]]
        stamps.append((str(project / "provenance" / "plugin"), _mtime(project / "provenance" / "plugin")))
    for d in dirs:
        for p in sorted(Path(d).glob("*.y*ml")) if Path(d).is_dir() else []:
            try:
                mtime = p.stat().st_mtime_ns
            except OSError:
                continue
            stamps.append((str(p), mtime))
            # the descriptors expand these at load: a changed value (VBT_CL_OBO, OPEN_TARGETS_DATA_PATH) is a
            # different catalog
            env.update({name: os.environ.get(name) for name in _env_refs(p, mtime)})
    key = json.dumps([settings.to_json(), variables, stamps, env], sort_keys=True, default=str)
    if key not in _CATALOGS:
        registry = discover(settings)
        _CATALOGS.clear()
        _CATALOGS[key] = (settings, build_catalog(settings, registry, variables=variables), registry)
    return _CATALOGS[key]


def data_readiness(config: dict[str, Any], cache: Any, catalog: Any, *, errors: Mapping[str, str] | None = None,
                   servers: Iterable[str] | None = None) -> DataReadiness:
    """Decide every bound tool of the enabled servers (or of ``servers``) from the cached ``_check`` results."""
    from .datalayer.gateway.readiness import call_readiness

    out = DataReadiness(tables=dict(cache.tables), errors=dict(errors or {}))
    enabled = set(servers) if servers is not None else {s.get("name") for s in _servers(config)}
    for server in catalog.servers():
        if server not in enabled:
            continue
        for tool in catalog.tools(server):
            try:
                contract = catalog.contract(server, tool)
            except Exception:  # noqa: BLE001 - a broken binding is reported by `vbt ds lint`
                continue
            b = contract.binding
            if b is None or b.serve == "block" or b.hidden:
                continue
            name = _tool_name(server, tool)
            out.bound.setdefault(server, []).append(name)
            out.reads[name] = {cache.physical(ref)[0] for ref in contract.tables}
            if getattr(contract, "quarantined", None):
                # R8: it reads a table or id_type of a descriptor that does not load
                q = contract.quarantined[0]
                out.unready[name] = {"table": q.name or q.file, "check": "quarantined",
                                     "detail": f"{q.file}: {q.summary}", "status": "quarantined"}
                continue
            r = call_readiness(contract, cache, bound_table=contract.bound_table)
            always = [x for x in r.hard if "partition" not in x]       # a section table only degrades its section
            if always:
                x = always[0]
                m = cache.get(x["name"])
                col = x.get("column")
                status = ((m.columns.get(col) or m.containers.get(col)) if (m is not None and col) else None) or \
                    (m.status if m is not None else None)
                out.unready[name] = {"table": x["name"], **({"column": col} if col else {}), "check": x.get("check"),
                                     "detail": x.get("detail"), "status": status}
            elif r.unavailable_partitions:
                out.partial[name] = sorted({p for parts in r.unavailable_partitions.values() for p in parts})
    out.granted = granted_tools(config, [t for tools in out.bound.values() for t in tools])
    return out


def _mark_awaiting(response: dict[str, Any], catalog: Any) -> set[str]:
    """Tables a tool materialises during the run (``materialized_by``) are ``awaiting_producer``,
    never missing, before the run exists."""
    out = set()
    for ref, m in (response.get("tables") or {}).items():
        try:
            spec = catalog.table(ref).spec
        except Exception:  # noqa: BLE001
            continue
        if getattr(spec, "materialized_by", None) is not None and m.get("status") not in _READY:
            m["status"] = "awaiting_producer"
            for c in m.get("checks") or []:
                if not c.get("ok"):
                    c["level"] = "info"
            out.add(ref)
    return out


def load_data_readiness(config: dict[str, Any], *, per_turn: bool = False,
                        response: Mapping[str, Any] | None = None, servers: Iterable[str] | None = None
                        ) -> tuple[DataReadiness, Any, Any]:
    """Run (or, per turn, reuse) the data child's check and decide the enabled servers' tools
    (``servers``: these instead). Returns ``(readiness, cache, catalog)``; the results are persisted
    in ``data.cache_dir``, where the session's gateway finds them."""
    from .datalayer.gateway.readiness import ReadinessCache

    settings, catalog, registry = data_catalog(config)
    cache = ReadinessCache(settings.cache_dir, catalog, registry, acquisition=settings.acquisition.policy())
    servers = list(servers) if servers is not None else None
    enabled = set(servers) if servers is not None else {s.get("name") for s in _servers(config)}
    wanted: set[str] = set()
    for server in catalog.servers():
        if server in enabled:
            for tool in catalog.tools(server):
                try:
                    wanted.update(cache.physical(ref)[0] for ref in catalog.contract(server, tool).tables)
                except Exception:  # noqa: BLE001
                    continue
    errors: dict[str, str] = {}
    key = str(settings.cache_dir)
    if response is None and per_turn:
        cache.load()   # stat-only: results whose descriptor digest and layout signature still match
        stale = []
        for ref in sorted(wanted - set(cache.tables)):
            known = _CHECK_ERRORS.get((key, ref))
            if known is not None and known[0] == cache._current_signature(ref):
                errors[ref] = known[1]
            else:
                stale.append(ref)
        if stale:
            response = run_data_check(config, tables=stale, depth=settings.readiness.session_depth)
    elif response is None:
        # session start and doctor: only the tables the enabled servers' tools read, and only those
        # without a persisted result whose descriptor digest and layout signature still match
        cache.load()
        todo = sorted(wanted - set(cache.tables))
        if todo:
            response = run_data_check(config, tables=todo, depth=settings.readiness.session_depth)
        reused = {ref: cache.tables[ref].model_dump(mode="json") for ref in sorted(wanted & set(cache.tables))
                  if ref not in todo}
        if reused:
            response = dict(response or {})
            response["tables"] = {**reused, **dict(response.get("tables") or {})}
    awaiting: set[str] = set()
    if response is not None:
        response = json.loads(json.dumps(response))
        awaiting = _mark_awaiting(response, catalog)
        cache.load_check_results(response)
        if not per_turn:
            _LAST_CHECK[key] = response
        for ref, err in (response.get("table_errors") or response.get("errors") or {}).items():
            errors[ref] = str(err)
            _CHECK_ERRORS[(key, ref)] = (cache._current_signature(ref), str(err))
    out = data_readiness(config, cache, catalog, errors=errors, servers=servers)
    out.awaiting = awaiting | {r for r, m in cache.tables.items() if m.status == "awaiting_producer"}
    return out, cache, catalog


def _failed_parts(m: Any) -> dict[tuple[str | None, str | None], list[Any]]:
    """``{(column, partition): [failed error checks]}`` of one table's check, plus parts whose
    status is not ready without a failed check naming them."""
    parts: dict[tuple[str | None, str | None], list[Any]] = {}
    for c in m.checks:
        if not c.ok and c.level == "error":
            parts.setdefault((c.column, c.partition), []).append(c)
    named = {col for col, _ in parts} | {part for _, part in parts}
    for col, st in {**m.columns, **m.containers}.items():
        if st not in _READY and col not in named:
            parts.setdefault((col, None), [])
    for label, st in m.partitions.items():
        if st not in _READY and label not in named:
            parts.setdefault((None, label), [])
    if not parts and m.status not in _READY:
        parts[(None, None)] = []
    return parts


def data_findings(config: dict[str, Any], dr: DataReadiness) -> list[CheckResult]:
    """One :class:`CheckResult` per unready part of a table some enabled tool reads (tables no
    enabled tool reads are ``unbound`` and never reported), then the legacy aggregates and the
    granted-tools summary. A part is ``required`` when it makes a granted tool unready."""
    out: list[CheckResult] = []
    used = dr.used
    optional = {src for src, env in OPTIONAL_SOURCES.items() if not _env_value(config, env)}
    for ref in sorted(dr.tables):
        m = dr.tables[ref]
        if ref not in used or ref in dr.awaiting:
            continue
        parts = _failed_parts(m)
        if m.status in _READY and not parts:
            continue
        source, _, table = ref.partition(".")
        quiet: dict[str, list[str]] = {}       # columns no enabled tool reads, by status: one result per table
        for (col, part), checks in sorted(parts.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
            status = (m.partitions.get(part) if part else None) or \
                ((m.columns.get(col) or m.containers.get(col)) if col else None) or m.status
            if status in _READY:
                status = m.status if m.status not in _READY else "partial"
            if part is not None:
                tools = {t: f"partition {part} of {ref} is {status}" for t, labels in dr.partial.items()
                         if part in labels}
            else:
                tools = {t: dr.reason(t) for t, r in dr.unready.items()
                         if r["table"] == ref and (r.get("column") == col or (col is None and "column" not in r))}
            if col and part is None and not tools:
                quiet.setdefault(status, []).append(col)
                continue
            detail = "; ".join(f"{c.name}: {c.detail}" for c in checks[:3]) or status
            if tools:
                names = sorted(t.split("__", 2)[-1] for t in tools)
                verb = "partial for" if part is not None else "unready"
                detail += f" -- {verb}: {', '.join(names[:6])}" + (f" (+{len(names) - 6})" if len(names) > 6 else "")
            hint = next((c.hint for c in checks if c.hint), "") or _status_hint(status, source, table)
            scope: dict[str, Any] = {"source": source, "table": table}
            if col:
                scope["column"] = col
            if part:
                scope["partition"] = part
            label = f"data: {ref}" + (f".{col}" if col else "") + (f" [{part}]" if part else "")
            required = part is None and source not in optional and any(t in dr.granted for t in tools)
            out.append(CheckResult(label, False, hint=hint, detail=f"{status}: {detail}"[:1500], required=required,
                                   kind="data", scope=scope, tools=tools))
        for status, cols in sorted(quiet.items()):
            listed = ", ".join(cols[:8]) + (f" (+{len(cols) - 8})" if len(cols) > 8 else "")
            out.append(CheckResult(f"data: {ref} ({len(cols)} column{'s' if len(cols) > 1 else ''})", False,
                                   required=False, kind="data",
                                   detail=f"{status}: {listed} -- no enabled tool's calls read them",
                                   hint=_status_hint(status, source, table),
                                   scope={"source": source, "table": table, "columns": cols}))
    for ref, err in sorted(dr.errors.items()):
        if ref in used:
            source, _, table = ref.partition(".")
            out.append(CheckResult(f"data: {ref}", False, required=False, kind="data",
                                   detail=f"the readiness check failed ({err[:300]}); the table is unchecked",
                                   hint="calls reading the table run without a cached readiness result",
                                   scope={"source": source, "table": table}))
    out.extend(_data_aggregates(config, dr))
    return out


def _source_tools(dr: DataReadiness, source: str) -> list[str]:
    return sorted(t for t, refs in dr.reads.items() if any(r.split(".")[0] == source for r in refs))


def source_label(source: str) -> str:
    """The per-source readiness summary's label (the same shape for every source)."""
    return f"data source: {source}"


def _source_hint(source: str, config: dict[str, Any]) -> str:
    env = OPTIONAL_SOURCES.get(source)
    hint = (f"`vbt ds check` lists the unready tables; `vbt data acquire {source} --env-file .env` fetches the "
            "pinned release into the acquisition home and writes its root variable")
    return hint + (f" ({source} is optional: unset {env})" if env and _env_value(config, env) else "")


def _data_aggregates(config: dict[str, Any], dr: DataReadiness) -> list[CheckResult]:
    """One summary per source the granted tools read (ok unless no granted tool reading it is ready; ASN-6: the
    same for every source, where Open Targets and Tahoe had labels and branches of their own) and the summary
    over every granted data tool, which is what ``require_ready`` blocks on."""
    from .datalayer.settings import DataSettings

    out = []
    optional = {src for src, env in OPTIONAL_SOURCES.items() if not _env_value(config, env)}
    sources = sorted({str(r).split(".")[0] for t in dr.granted for r in (dr.reads.get(t) or ())})
    for source in sources:
        tools = [t for t in _source_tools(dr, source) if t in dr.granted]
        ready = [t for t in tools if t not in dr.unready]
        bad = sorted({str(dr.unready[t].get("table")) for t in tools if t in dr.unready})
        detail = f"{len(ready)} of {len(tools)} granted tools reading {source} ready"
        if bad:
            detail += f"; not ready: {', '.join(bad[:6])}" + (f" (+{len(bad) - 6})" if len(bad) > 6 else "")
        out.append(CheckResult(source_label(source), bool(ready), hint=_source_hint(source, config), detail=detail,
                               kind="data", required=source not in optional, scope={"source": source}))
    block_when = DataSettings.from_config(config).readiness.block_when
    granted = sorted(dr.granted)
    ready = [t for t in granted if t not in dr.unready]
    ok = (len(ready) == len(granted)) if block_when == "any_unready" else (bool(ready) or not granted)
    unready = sorted(t for t in granted if t in dr.unready)
    detail = f"{len(ready)} of {len(granted)} granted data tools ready"
    if unready:
        detail += f"; unready: {', '.join(unready[:8])}" + (f" (+{len(unready) - 8})" if len(unready) > 8 else "")
    if dr.partial:
        detail += f"; {len(dr.partial)} tool(s) with unavailable partitions"
    out.append(CheckResult(DATA_TOOLS_LABEL, ok, kind="data", required=block_when != "never", detail=detail[:1500],
                           hint="no granted data tool can be served: `vbt ds check` lists the unready tables, "
                                "columns and partitions (or start degraded with --allow-missing-data)",
                           scope={"granted": len(granted), "ready": len(ready),
                                  "bound": {s: list(t) for s, t in sorted(dr.bound.items())}},
                           tools={t: dr.reason(t) for t in unready}))
    return out


def check_data_readiness(config: dict[str, Any], *, per_turn: bool = False) -> list[CheckResult]:
    """Tool-scoped readiness results (raises :class:`DataCheckUnavailable` when the data child
    cannot run its check or the catalog cannot be loaded)."""
    try:
        dr, _cache, _catalog = load_data_readiness(config, per_turn=per_turn)
    except DataCheckUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - a broken catalog is a required failure; anything else a note
        from .datalayer.catalog import CatalogError
        from .datalayer.descriptor.load import DescriptorError

        cls = DataCatalogError if isinstance(exc, (DescriptorError, CatalogError)) else DataCheckUnavailable
        raise cls(f"{type(exc).__name__}: {exc}") from exc
    return data_findings(config, dr)


def _data_layer_results(results: Iterable[CheckResult]) -> list[CheckResult]:
    return [r for r in results if r.kind == "data" and r.scope is not None]


def degraded_tools(config: dict[str, Any], results: Iterable[CheckResult]) -> dict[str, str]:
    """``{mcp__server__tool: reason}``: tools every call of which reads an unready part, given
    ``require_ready`` results (data layer only; tools that only some partitions block, and the
    tools of servers the config does not enable, are left out)."""
    names = {s.get("name") for s in _servers(config)}
    out: dict[str, str] = {}
    for r in _data_layer_results(results):
        if r.scope.get("partition"):
            continue
        for tool, why in r.tools.items():
            if tool.split("__")[1:2] and tool.split("__")[1] in names:
                out.setdefault(tool, why)
    return out


def degraded_tables(results: Iterable[CheckResult]) -> dict[str, str]:
    """``{source.table: status}`` of the unready tables that make some tool unready."""
    out: dict[str, str] = {}
    for r in _data_layer_results(results):
        if not r.ok and r.scope.get("table") and r.tools and not r.scope.get("partition"):
            out.setdefault(f"{r.scope['source']}.{r.scope['table']}", r.detail.split(":", 1)[0])
    return out


def check_mcp_commands(config: dict[str, Any]) -> list[CheckResult]:
    """Each stdio server's interpreter is executable and its script exists."""
    from .tools.mcp_bridge import MCPBridge, MCPServerConfig, _ConfigError

    from .datalayer.launch import LIMIT_KINDS
    from .datalayer.memory.sizing import memory_problems

    out = []
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    memory = data.get("memory") if isinstance(data.get("memory"), dict) else {}
    for problem in memory_problems(memory, LIMIT_KINDS):
        # an unparsable size or containment would otherwise be ignored without a word (RR-5)
        out.append(CheckResult("data.memory settings", False, detail=problem,
                               hint="use auto, off or a size such as 6144 or '6 GB'; limit_kind one of "
                                    + ", ".join(LIMIT_KINDS), kind="mcp"))
    for s in _servers(config):
        cfg = MCPServerConfig(**{k: v for k, v in s.items() if k in MCPServerConfig.__dataclass_fields__})
        if cfg.limit_kind is not None and cfg.limit_kind not in LIMIT_KINDS:
            # a typo would otherwise launch the server under data.memory.limit_kind, not what was configured
            out.append(CheckResult(f"MCP server {cfg.name}: limit_kind", False,
                                   detail=f"limit_kind {cfg.limit_kind!r} is not one of {', '.join(LIMIT_KINDS)}",
                                   hint="fix limit_kind in the MCP server file (configs/mcp_servers.yaml)", kind="mcp"))
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


def data_plugin_requires(config: dict[str, Any]) -> dict[str, list[str]]:
    """``{module: [kind/plugin, ...]}``: the modules registered data-layer plugins declare in ``requires``."""
    try:
        from .datalayer.plugins.registry import discover
        from .datalayer.settings import DataSettings
        registry = discover(DataSettings.from_config(config))
    except Exception:  # noqa: BLE001 - a broken registry is reported by `vbt ds lint`
        log.debug("plugin discovery failed", exc_info=True)
        return {}
    out: dict[str, list[str]] = {}
    for kind in registry.kinds:
        for plugin in registry.all(kind):
            for mod in getattr(plugin, "requires", ()) or ():
                out.setdefault(str(mod), []).append(f"{kind}/{plugin.name}")
    return out


def mcp_modules(config: dict[str, Any]) -> list[str]:
    """Modules the enabled servers launched with vars.mcp_python import (and, with the data layer,
    the data child and the plugins' ``requires``)."""
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
    if mods and data_layer_active(config):
        mods += [*SERVER_MODULES["_stdio"], *DATA_CHILD_MODULES, *data_plugin_requires(config)]
    return list(dict.fromkeys(mods))


def check_mcp_imports(config: dict[str, Any]) -> CheckResult:
    """Run ``<mcp_python> -E -c 'import ...'`` for the modules the enabled servers need, the data
    child's own imports and every data-layer plugin's ``requires`` (a missing one names the plugin)."""
    py = (config.get("vars") or {}).get("mcp_python") or sys.executable
    mods = mcp_modules(config)
    label = f"MCP interpreter imports ({py})"
    if not mods:
        return CheckResult(label, True, detail="no stdio servers use vars.mcp_python", required=False, kind="mcp")
    res = probe_imports(py, mods)
    missing = {m: r.get("error", "") for m, r in res.items() if not r.get("ok")}
    needed_by = data_plugin_requires(config) if data_layer_active(config) else {}
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
        problems.append("missing: " + ", ".join(
            f"{m} ({e})" + (f" [needed by the data-layer plugin(s) {', '.join(needed_by[m])}]" if m in needed_by
                            else " [needed by the data child]" if m in DATA_CHILD_MODULES else "")
            for m, e in missing.items()))
    if needed_by and not missing:
        detail_bits.append(f"data child and {sum(map(len, needed_by.values()))} plugin requirement(s) import")
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
            "ok <- sapply(pk, requireNamespace, quietly=TRUE); writeLines(paste0('VBT_R ', pk, '=', ok))")
    try:
        proc = subprocess.run([rscript, "-e", code], capture_output=True, text=True, timeout=300,
                              env=child_env(os.environ))
        # one line per package (an earlier cat() of the vector put a space before every line after the first)
        found = dict(ln.strip()[6:].split("=", 1) for ln in proc.stdout.splitlines() if ln.strip().startswith("VBT_R "))
    except (OSError, subprocess.TimeoutExpired) as exc:
        found = {}
        out.append(CheckResult("analysis: Rscript runs", False, detail=str(exc), kind="analysis"))
    for p in R_PACKAGES:
        out.append(CheckResult(f"analysis: R package {p}", found.get(p) == "TRUE",
                               detail=rscript if found.get(p) == "TRUE" else "not installed",
                               hint=f"conda install -c conda-forge r-{p.lower()}", kind="analysis"))
    return out


#: Smoke modes: ``gateway`` (sentinel positive and negative controls through the data gateway,
#: the default with the data layer) and ``upstream`` (one ``SMOKE_CALLS`` call per server).
SMOKE_MODES = ("gateway", "upstream")
#: How long a gateway smoke waits for the data child's readiness check.
SMOKE_READINESS_WAIT_S = 600.0


def sentinel_controls(catalog: Any, server: str, schemas: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The smoke controls of ``server`` from its tables' sentinels (§13): one bound tool whose
    identifier argument (``eq``/``in``) binds the single key column of a ``sentinels.present`` entry
    and that needs no other argument. Returns ``[{tool, args, kind: positive|negative, key, table}]``
    (the negative from ``sentinels.absent`` when the table declares one); [] when no tool qualifies."""
    candidates = []
    for tool in catalog.tools(server):
        schema = schemas.get(tool)
        if schema is None:
            continue
        try:
            contract = catalog.contract(server, tool)
        except Exception:  # noqa: BLE001
            continue
        b = contract.binding
        bound = contract.bound_table
        if b is None or b.serve == "block" or b.hidden or not bound or getattr(contract, "quarantined", None):
            continue
        try:
            sentinels = catalog.table(bound).spec.sentinels
        except Exception:  # noqa: BLE001
            continue
        if sentinels is None or not sentinels.present:
            continue
        required = set(schema.get("required") or [])
        for name, a in contract.identifier_args.items():
            cols = [c for t, c in contract.arg_columns(name) if t == bound]
            if not cols or a.op not in ("eq", "in") or required - {name}:
                continue
            pos = next((x for x in sentinels.present if list(x.key) == cols[:1] and x.via is None), None)
            if pos is None:
                continue
            neg = next((x for x in sentinels.absent if list(x.key) == cols[:1] and x.via is None), None)

            def value(v: Any, a: Any = a) -> Any:
                return [v] if a.op == "in" else v

            rank = (neg is None, not (tool.startswith("get_") and tool.endswith("_info")), tool)
            controls = [{"tool": tool, "args": {name: value(pos.key[cols[0]])}, "kind": "positive",
                         "key": dict(pos.key), "table": bound}]
            if neg is not None:
                controls.append({"tool": tool, "args": {name: value(neg.key[cols[0]])}, "kind": "negative",
                                 "key": dict(neg.key), "table": bound})
            candidates.append((rank, controls))
            break
    return min(candidates, key=lambda c: c[0])[1] if candidates else []


def judge_control(control: Mapping[str, Any], out: Any = None, exc: BaseException | None = None) -> tuple[bool, str]:
    """``(passed, detail)`` of one sentinel control. A positive control passes only with a success
    (``ok``/``partial``, never ``empty``, ``empty_unverified`` or an error) that carries the sentinel's
    key values; a negative control passes only with a ``not_found`` error."""
    kind = getattr(getattr(exc, "kind", None), "value", getattr(exc, "kind", None)) if exc is not None else None
    if control["kind"] == "negative":
        if exc is not None and kind == "not_found":
            return True, "not_found (as expected)"
        if exc is not None:
            return False, f"expected not_found for an absent key, got {kind or type(exc).__name__}: {str(exc)[:300]}"
        return False, (f"phantom: the absent key {control['key']} was answered "
                       f"({getattr(out, 'status', 'ok')}): {str(getattr(out, 'text', out))[:200]}")
    if exc is not None:
        return False, f"{kind or type(exc).__name__}: {str(exc)[:600]}"
    status = str(getattr(out, "status", "ok")) if getattr(out, "is_data_result", False) else "ok"
    text = str(getattr(out, "full_text", None) or getattr(out, "text", None) or out)
    if status not in ("ok", "partial"):
        return False, f"the sentinel came back {status}: {text[:300]}"
    missing = [f"{k}={v}" for k, v in control["key"].items() if v is not None and str(v) not in text]
    if missing:
        return False, f"the result does not carry the sentinel key ({', '.join(missing)}): {text[:300]}"
    return True, f"{status}: {text.replace(chr(10), ' ')[:140]}"


def _smoke_call(raw: Mapping[str, Any], name: str, mode: str, controls: list[dict[str, Any]]
                ) -> list[tuple[str, dict[str, Any], str]]:
    """``[(tool, args, how)]`` for one server: ``explicit`` (the server's ``smoke:`` entry),
    ``control`` (sentinel controls) or ``upstream`` (``SMOKE_CALLS``)."""
    smoke = raw.get("smoke", None)
    if smoke is False:
        return []
    if isinstance(smoke, dict) and smoke.get("tool"):
        return [(str(smoke["tool"]), dict(smoke.get("args") or {}), "explicit")]
    if mode == "gateway" and controls:
        return [(c["tool"], dict(c["args"]), "control") for c in controls]
    if name in SMOKE_CALLS and (mode == "upstream" or not (name in OPEN_TARGETS_SERVERS or
                                                            name == "functional_genomics")):
        tool, args = SMOKE_CALLS[name]
        return [(tool, dict(args), "upstream")]
    return []


async def smoke_mcp(config: dict[str, Any], *, log_dir: str | os.PathLike | None = None,
                    servers: Iterable[str] | None = None, mode: str = "gateway",
                    keep: bool = False) -> list[CheckResult]:
    """Start every enabled MCP server and make cheap calls.

    ``mode="gateway"`` (the default; with the data layer): the servers start behind the data
    gateway (with its ``data`` child) and each runs its sentinel controls (:func:`sentinel_controls`):
    a positive control must return the sentinel's key with a success, so an empty answer fails, and
    a negative control must return ``not_found``, so a phantom answer fails. Servers without a
    local-table sentinel (PubMed, ClinicalTrials.gov) make their ``SMOKE_CALLS`` call; Open Targets
    and Tahoe servers without one are reported as not smoke-tested. ``mode="upstream"`` (``vbt
    doctor --smoke=upstream``) makes the old ``SMOKE_CALLS`` call per server, which forces the
    upstream tools' multi-GB loads (through the gateway's admission control when the data layer is on).
    A server's ``smoke: {tool, args}`` entry (or ``smoke: false``) overrides both.

    A server fails when it does not start, advertises no tools, or a call fails (an error, or a
    failed control).

    The working directory (``vbt-doctor-*``: the run directory, the MCP output and, without ``log_dir``, the
    servers' logs) is removed afterwards, unless ``keep`` or a check failed without ``log_dir``: then the logs the
    hints name stay and a line says where (RR-9: every run used to leave one behind).
    """
    from .tools.base import ToolFailure
    from .tools.mcp_bridge import MCPBridge, MCPServerConfig

    if mode not in SMOKE_MODES:
        raise ValueError(f"smoke mode must be one of {SMOKE_MODES}, got {mode!r}")
    specs_raw = [s for s in _servers(config) if servers is None or s.get("name") in set(servers)]
    tmp = tempfile.mkdtemp(prefix="vbt-doctor-")
    extra = base_tool_env(config)
    extra.update({"VBT_RUN_DIR": tmp, "MCP_OUTPUT_DIR": str(Path(tmp) / "mcp")})
    results: list[CheckResult] = []
    gateway = None
    launch = list(specs_raw)
    if specs_raw and data_layer_active(config):
        try:
            from .datalayer import build_gateway
            gateway = build_gateway(config, {"dir": tmp, "run_id": "doctor", "mcp_output_dir": str(Path(tmp) / "mcp")})
            names = {s.get("name") for s in specs_raw}
            launch += [s for s in gateway.extra_servers() or [] if s.get("name") not in names]
        except Exception as exc:  # noqa: BLE001 - smoke the servers without the gateway, and say so
            gateway = None
            results.append(CheckResult("MCP smoke: data gateway", False, required=False, kind="mcp",
                                       detail=f"the gateway could not be built ({type(exc).__name__}: {exc}); "
                                              "the servers are smoke-tested without it"[:1500]))
    specs = [MCPServerConfig(**{k: v for k, v in s.items() if k in MCPServerConfig.__dataclass_fields__})
             for s in launch]
    kwargs: dict[str, Any] = {"gateway": gateway} if gateway is not None else {}
    bridge = MCPBridge(specs, extra_env=extra, log_dir=log_dir or Path(tmp) / "logs",
                       options=config.get("mcp") or {}, **kwargs)
    if mode == "gateway" and gateway is None and not any(isinstance(s.get("smoke"), dict) for s in specs_raw):
        mode = "upstream"   # no gateway: no sentinel controls
    try:
        await bridge.start()
        status = bridge.status()
        if gateway is not None and mode == "gateway":
            await gateway.wait_readiness(SMOKE_READINESS_WAIT_S)
        for raw in specs_raw:
            name = str(raw.get("name"))
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
            controls: list[dict[str, Any]] = []
            if gateway is not None and mode == "gateway":
                schemas = {t.name.split("__", 2)[-1]: dict(t.input_schema or {}) for t in bridge.tools
                           if t.name.startswith(f"mcp__{name}__")}
                controls = sentinel_controls(gateway.catalog, name, schemas)
            calls = _smoke_call(raw, name, mode, controls)
            if not calls and mode == "gateway" and gateway is not None:
                results.append(CheckResult(f"MCP {name}: sentinel controls", True, required=False, kind="mcp",
                                           detail="no tool answers a table sentinel by key alone; "
                                                  "`vbt doctor --smoke=upstream` makes a live call"))
            timeout = min(float(getattr(next((x for x in specs if x.name == name), None), "timeout_s", None)
                                or SMOKE_CALL_TIMEOUT_S), SMOKE_CALL_TIMEOUT_S)
            for i, (tool, args, how) in enumerate(calls):
                control = controls[i] if how == "control" else None
                tag = f" [{control['kind']} control]" if control else ""
                label = f"MCP {name}: {tool}({', '.join(f'{k}={v!r}' for k, v in args.items())}){tag}"
                out: Any = None
                err: BaseException | None = None
                try:
                    out = await asyncio.wait_for(bridge.call(name, tool, args), timeout + 5)
                except ToolFailure as exc:
                    err = exc
                except Exception as exc:  # noqa: BLE001
                    err = exc
                if control is not None:
                    ok, detail = judge_control(control, out, err)
                    results.append(CheckResult(label, ok, detail=detail[:1500], kind="mcp",
                                               hint="" if ok else "the sentinel control failed: check the table's "
                                                                  "readiness (`vbt ds check`) and the server log"))
                elif isinstance(err, ToolFailure):
                    results.append(CheckResult(label, False, detail=str(err)[:1500], kind="mcp",
                                               hint="the tool returned an error: check the reference data and the "
                                                    "server log"))
                elif err is not None:
                    results.append(CheckResult(label, False, detail=f"{type(err).__name__}: {err}"[:1500], kind="mcp"))
                else:
                    text = str(getattr(out, "text", out)) if getattr(out, "is_data_result", False) else \
                        (out if isinstance(out, str) else str(out))
                    results.append(CheckResult(label, True, detail=text.replace("\n", " ")[:160], kind="mcp"))
    finally:
        close = getattr(gateway, "aclose", None)
        if callable(close):
            try:
                await close()
            except Exception:  # noqa: BLE001
                pass
        await bridge.aclose()
        failed = any(not r.ok for r in results)
        if keep or (failed and not log_dir):
            results.append(CheckResult("MCP smoke: working directory kept", True, required=False, kind="mcp",
                                       detail=f"the servers' logs and outputs are in {tmp} (remove it when done)"))
        else:
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
    always raise; broken MCP commands raise unless ``allow_missing_data``.
    Reference data: with the data layer, readiness is scoped to tools and the
    session blocks only when no data tool granted to any agent is ready
    (``data.readiness.block_when: all_unready``; ``any_unready`` blocks on any
    unready granted tool, ``never`` does not block); one missing table only
    degrades the tools that read it. Without the data layer, missing reference
    data raises. ``allow_missing_data`` never blocks on data (the run is then
    degraded: the caller marks it and tells the agents which tools lack data).
    Per turn the data check reuses the session's cached results (stat-only
    signatures; only tables whose layout changed are re-checked). With
    ``start_mcp=False`` (``--no-mcp``: no MCP server is started) only the
    credentials are checked. Returns the check results.
    """
    if _provider_name(config, provider) == "mock":
        return []
    if not (config.get("orchestration") or {}).get("require_reference_data", True):
        return []
    results = [check_credentials(config, provider)]
    if start_mcp:
        results += check_reference_data(config, per_turn=True) if per_turn else check_reference_data(config)
        if not per_turn:
            results += check_mcp_commands(config)
    failed = [r for r in results if r.required and not r.ok]

    def blocks(r: CheckResult) -> bool:
        if r.kind == "credentials":
            return True
        if allow_missing_data:
            return False
        if r.kind == "data" and r.scope is not None:   # tool-scoped: only the granted-tools summary blocks
            return r.label == DATA_TOOLS_LABEL
        return True

    blocking = [r for r in failed if blocks(r)]
    if blocking:
        lines = "; ".join(f"{r.label}: {r.detail or 'failed'}" + (f" (fix: {r.hint})" if r.hint else "")
                          for r in blocking)
        raise DataReadinessError(f"Not ready: {lines}. {TURN_NOT_SENT}", results)
    return results


def degraded_servers(config: dict[str, Any], results: Iterable[CheckResult]) -> dict[str, str]:
    """Servers that lack their reference data, given ``require_ready`` results ({server: reason}).

    With the data layer, only servers whose **every** bound tool is unready (one missing table no
    longer names every Open Targets server); the unready tools are :func:`degraded_tools`."""
    results = list(results)
    names = {s.get("name") for s in _servers(config)}
    summary = next((r for r in _data_layer_results(results) if r.label == DATA_TOOLS_LABEL), None)
    if summary is not None:
        tools = degraded_tools(config, results)
        out = {}
        for server, bound in sorted((summary.scope.get("bound") or {}).items()):
            if server in names and bound and all(t in tools for t in bound):
                out[server] = f"every tool unready: {tools[bound[0]]}"[:500]
        return out
    out = {}
    for r in results:
        if r.ok or r.kind != "data":
            continue
        if "OPEN_TARGETS" in r.label:
            for n in sorted(names & OPEN_TARGETS_SERVERS):
                out[n] = f"Open Targets data unavailable: {r.detail}"
        elif "Tahoe" in r.label and "functional_genomics" in names:
            out.setdefault("functional_genomics", f"Tahoe-100M data unavailable: {r.detail}")
    return out


SERVE_HINT = ("start the inference server (`vbt local serve --profile <h100|h200|rtxpro6000|5090>` or "
              "`docker compose -f deploy/local/docker-compose.yml --profile h100 up -d`; deploy/local/README.md) "
              "and check provider.options.base_url / VBT_LLM_BASE_URL; `vbt local check` probes its capabilities")


def serve_hint(config: dict[str, Any], provider: Any = None) -> str:
    """How to start the model server of the configured provider. ``vbt local serve`` starts vLLM, so a provider that
    talks to another server (``llamacpp``: llama-server) is told how to start that one
    (``providers.openai_compat.serve_hint``, the hint the provider's own errors give)."""
    name = _provider_name(config, provider)
    if name == "llamacpp":
        from .providers.openai_compat import serve_hint as provider_hint

        hint = provider_hint(name).split("? ", 1)[-1]     # the provider's question is the check's label here
        return f"{hint[:1].lower()}{hint[1:]}; `vbt local check` probes its capabilities"
    return SERVE_HINT


def check_server_reachable(config: dict[str, Any], provider: Any, *, timeout_s: float = 3.0) -> CheckResult:
    """``vbt doctor`` without ``--smoke``: is the local model server up at all? One
    bounded ``GET /health`` per server; optional (never fails the doctor), the
    full readiness check is ``--smoke`` / ``vbt local check``."""
    label = f"{_provider_name(config, provider)} model server"
    full = "`vbt doctor --smoke` or `vbt local check` checks served models and context window"
    probe = getattr(provider, "reachable", None)
    if probe is None:
        return CheckResult(label, True, required=False, kind="model",
                           detail="not contacted (run `vbt doctor --smoke` or `vbt local check`)")

    async def run() -> list[tuple[str, str | None]]:
        try:
            return await asyncio.wait_for(probe(timeout_s=timeout_s), timeout_s * 4 + 2)
        finally:
            try:
                await provider.aclose()
            except Exception:  # noqa: BLE001
                pass

    try:
        servers = asyncio.run(run())
    except Exception as exc:  # noqa: BLE001 - a probe failure is reported, never raised
        return CheckResult(label, False, required=False, kind="model",
                           detail=f"could not be checked ({type(exc).__name__}: {exc})"[:1500],
                           hint=serve_hint(config, provider))
    down = [(u, why) for u, why in servers if why]
    if down:
        where = "; ".join(f"nothing answers at {u} ({why})" for u, why in down)
        return CheckResult(label, False, required=False, kind="model",
                           detail=f"not running: {where}"[:1500], hint=serve_hint(config, provider))
    return CheckResult(label, True, required=False, kind="model",
                       detail=f"responding at {', '.join(u for u, _ in servers)} ({full})")


async def check_model_server(config: dict[str, Any], provider: Any) -> list[CheckResult]:
    """``vbt doctor --smoke`` for providers with a readiness hook (local servers):
    ``provider.prepare()`` (/health, served models) and the server's
    ``max_model_len`` against the configured context windows."""
    if not has_prepare(provider):
        return []
    name = _provider_name(config, provider)
    try:
        await prepare_provider(provider, config)
    except ProviderNotReadyError as exc:
        return [CheckResult(f"{name} model server ready", False, detail=str(exc)[:1500],
                            hint=serve_hint(config, provider),
                            kind="model")]
    info = await provider_server_info(provider)
    served = [str(m.get("id")) for m in info.get("models") or [] if isinstance(m, dict)]
    bits = [f"serves {', '.join(served) or '(none listed)'}"]
    if info.get("version"):
        bits.append(f"version {info['version']}")
    if info.get("family"):
        bits.append(f"family {info['family']}")
    out = [CheckResult(f"{name} model server ready", True, detail="; ".join(bits), kind="model")]
    window, mml = configured_window(config), info.get("max_model_len")
    if window and mml:
        ok = int(mml) >= int(window)
        out.append(CheckResult(
            "model context window", ok, required=False, kind="model",
            detail=f"server max_model_len {int(mml):,}; configured context_window_tokens {int(window):,}",
            hint=(f"lower models.<tier>.context_window_tokens to {int(mml)} (and context.default_window_tokens), "
                  f"or serve with --max-model-len {int(window)}")))
    elif not mml:
        out.append(CheckResult("model context window", True, required=False, kind="model",
                               detail="the server does not report max_model_len; the configured window is used"))
    return out


def check_search(config: dict[str, Any], provider: Any) -> CheckResult | None:
    """Which WebSearch backend the run will use (no network, no secrets)."""
    if not (config.get("web") or {}).get("enabled", True):
        return None
    try:
        from .tools.search_backends import describe_search_backend
        d = describe_search_backend(config, provider)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("web search backend", False, required=False, detail=f"{type(exc).__name__}: {exc}",
                           kind="general")
    if d.get("problem"):
        return CheckResult("web search backend", False, required=False, detail=str(d["problem"]),
                           hint="see docs/WEB_SEARCH.md", kind="general")
    if not d.get("backend"):
        return CheckResult("web search backend", False, required=False, detail=str(d.get("reason") or "none"),
                           hint="start SearxNG (`docker compose -f deploy/local/docker-compose.yml up -d searxng`) "
                                "and set SEARXNG_URL, or set BRAVE_SEARCH_API_KEY (docs/WEB_SEARCH.md)",
                           kind="general")
    where = f" at {d['url']}" if d.get("url") and d.get("backend") == "searxng" else ""
    return CheckResult("web search backend", True, required=False, kind="general",
                       detail=f"{d['backend']}{where} ({d.get('reason') or 'configured'})")


async def smoke_search(config: dict[str, Any], provider: Any) -> CheckResult | None:
    """One live query through the WebSearch backend (``vbt doctor --smoke``)."""
    if not (config.get("web") or {}).get("enabled", True):
        return None
    try:
        from .tools.search_backends import check_search_backend
        r = await check_search_backend(config, provider)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("web search query", False, required=False, detail=f"{type(exc).__name__}: {exc}",
                           kind="general")
    if r.get("backend") is None:
        return None  # check_search already reported why
    detail = (f"{r['backend']}: {r.get('n_results', 0)} results in {r.get('elapsed_s')}s" if r.get("ok")
              else f"{r['backend']}: {r.get('error')}")
    return CheckResult("web search query", bool(r.get("ok")), required=False, detail=str(detail)[:1500],
                       hint="see docs/WEB_SEARCH.md", kind="general")


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

def tool_readiness_lines(results: Iterable[CheckResult], *, every_tool: bool = False) -> list[str]:
    """Per-server tool readiness from data-layer results: ``target: 14 of 15 tools ready`` and the
    unready (or, with ``every_tool``, every) tool with its reason."""
    results = list(results)
    summary = next((r for r in _data_layer_results(results) if r.label == DATA_TOOLS_LABEL), None)
    if summary is None:
        return []
    unready: dict[str, str] = {}
    partial: dict[str, str] = {}
    for r in _data_layer_results(results):
        target = partial if (r.scope or {}).get("partition") else unready
        for tool, why in r.tools.items():
            target.setdefault(tool, why)
    lines = ["Data tools (tool-scoped readiness; `vbt ds check --tool server.tool` explains one):"]
    for server, tools in sorted((summary.scope.get("bound") or {}).items()):
        n_ready = sum(1 for t in tools if t not in unready)
        lines.append(f"  {server}: {n_ready} of {len(tools)} tools ready"
                     + (f", {sum(1 for t in tools if t in partial)} with unavailable partitions"
                        if any(t in partial for t in tools) else ""))
        for t in tools:
            short = t.split("__", 2)[-1]
            if t in unready:
                lines.append(f"    [!!] {short}: {unready[t]}")
            elif t in partial:
                lines.append(f"    [--] {short}: {partial[t]}")
            elif every_tool:
                lines.append(f"    [ok] {short}")
    return lines


def run_doctor(config: dict[str, Any], *, smoke: bool | str = False, analysis: bool = False, data: bool = False,
               out: Callable[[str], Any] = print) -> int:
    """Print an installation report; return 0 when every required check passes.

    ``smoke``: True or ``"gateway"`` runs the sentinel controls through the data gateway (the
    upstream ``SMOKE_CALLS`` without the data layer); ``"upstream"`` the old one-call-per-server
    smoke. ``data``: list every data tool's readiness (not only the unready ones)."""
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
        provider = create_configured_provider(config)
    except Exception as exc:  # noqa: BLE001 - fall back to the env-var check
        provider = None
        if _provider_name(config) in LOCAL_PROVIDERS:
            add(CheckResult(f"{_provider_name(config)} provider options", False, detail=f"{type(exc).__name__}: {exc}",
                            hint="fix provider.options in the config (docs/PROVIDERS.md)", kind="credentials"))
    add(check_credentials(config, provider))
    if provider is not None and has_prepare(provider) and not smoke:
        add(check_server_reachable(config, provider))
    add(check_search(config, provider))
    data_results = check_reference_data(config)
    add(data_results)
    for line in tool_readiness_lines(data_results, every_tool=data):
        out(line)
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
        if provider is not None:
            async def live() -> list[CheckResult | None]:
                try:
                    return [*await check_model_server(config, provider), await smoke_search(config, provider)]
                finally:
                    try:
                        await provider.aclose()
                    except Exception:  # noqa: BLE001
                        pass
            add([r for r in asyncio.run(live()) if r is not None])
        if _servers(config):
            mode = smoke if smoke in SMOKE_MODES else "gateway"
            add(asyncio.run(smoke_mcp(config, mode=str(mode))))
        else:
            add(CheckResult("MCP smoke test", True, detail="no MCP servers enabled", required=False))
    failed = [r for r in results if r.required and not r.ok]
    out("PASS" if not failed else f"FAIL -- {len(failed)} check(s) failed; fix the items marked [!!].")
    return 0 if not failed else 1


def _doctor_handler(args: Any, config: dict[str, Any]) -> int:
    smoke = getattr(args, "smoke", False)
    return run_doctor(config, smoke=smoke if smoke in SMOKE_MODES else bool(smoke),
                      analysis=bool(getattr(args, "analysis", False)), data=bool(getattr(args, "data", False)))


def add_doctor_parser(sub: Any) -> Any:
    """Register ``vbt doctor`` on an argparse subparsers object."""
    d = sub.add_parser("doctor", help="check installation, credentials, reference data and MCP servers")
    d.add_argument("--smoke", nargs="?", const="gateway", default=False, choices=SMOKE_MODES,
                   help="also contact the model server (local providers: /health, served models, max_model_len), "
                        "run one web search, and start every MCP server: with the data layer each server runs "
                        "its sentinel positive and negative controls through the gateway (an empty positive or "
                        "a phantom negative fails); --smoke=upstream makes one live upstream call per server "
                        "instead (fails on tool errors)")
    d.add_argument("--data", action="store_true",
                   help="list the readiness of every data tool (default: only the unready ones); "
                        "`vbt ds check` gives the per-table, column and partition detail")
    d.add_argument("--analysis", action="store_true",
                   help="also check the Python/R analysis stack (scanpy, pydeseq2, rpy2, lme4, glmmTMB, ...)")
    d.set_defaults(handler=_doctor_handler)
    # `vbt validate`, the host certification (vbt.validate), is registered with the doctor it extends
    from .validate import add_validate_parser

    add_validate_parser(sub)
    return d


__all__ = [
    "CheckResult", "DataCatalogError", "DataCheckUnavailable", "DataReadiness", "DataReadinessError", "LOCAL_PROVIDERS",
    "ProviderNotReadyError", "TURN_NOT_SENT", "DATA_TOOLS_LABEL", "check_data_readiness", "data_catalog",
    "data_child_command", "data_findings", "data_layer_active", "data_readiness", "degraded_tables",
    "degraded_tools", "granted_tools", "judge_control", "last_data_check", "load_data_readiness", "run_data_check",
    "sentinel_controls", "tool_readiness_lines",
    "add_doctor_parser", "check_analysis_stack", "check_credentials", "check_mcp_commands", "check_mcp_imports",
    "check_model_server", "check_reference_data", "check_search", "check_server_reachable", "configured_models",
    "configured_window", "degraded_servers", "has_prepare", "prepare_provider", "provider_server_info", "require_ready", "run_doctor",
    "smoke_mcp", "smoke_search", "upstream_doctor",
]
