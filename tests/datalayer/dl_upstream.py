"""Run the unmodified upstream MCP servers on fixtures, through the real bridge and runtime (§19).

* :func:`start_bridge` builds ``MCPServerConfig`` entries from ``configs/mcp_servers.yaml``
  (``vbt.config.load_config(['mock'], overrides)``) with ``-B`` and ``PYTHONDONTWRITEBYTECODE=1``
  (nothing may be written under ``third_party/``), the fixture data paths and
  ``VBT_EUTILS_BASE`` in the environment, and the clinicaltrials server replaced by
  ``stubs/clinicaltrials_stubbed_server.py`` (the upstream file runs unchanged). With
  ``gateway=True`` the config gets ``data: {enabled: true, gateway: {mode: enforce}}`` and the
  bridge is built with ``vbt.datalayer.build_gateway`` (skipped while that does not exist).
* :func:`call` returns ``CallResult(is_error, text, obj, status, kind, payload)``.
* :class:`LiveBridge` runs one bridge on its own event loop thread so module-scoped fixtures
  can share started servers across synchronous, parametrized tests.
* :func:`run_with_runtime` scripts the mock provider to issue the tool calls and then
  ``record_claims`` through ``Runtime``, so ``is_error``, ``result_status`` and ``coverage``
  come from the real recording path.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import importlib.util
import json
import os
import sys
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from stubs import CLINICALTRIALS_WRAPPER

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = REPO / "third_party" / "TheVirtualBiotech"
PUBMED_SERVER = REPO / "src" / "vbt" / "mcp_servers" / "pubmed_server.py"
OVERLAY_SENTINEL = REPO / "configs" / "data" / "overlays" / "target.yaml"

FAST_START = {"start_backoff_s": 0.05, "start_backoff_factor": 1.0, "start_attempts": 2, "start_timeout_s": 120,
              "default_timeout_s": 180}
GATEWAY_MODULE = "vbt.datalayer.gateway"


# ---------------------------------------------------------------------------- availability


def upstream_root() -> Path:
    return Path(os.environ.get("VBT_UPSTREAM") or UPSTREAM)


def upstream_missing() -> str | None:
    """Why the upstream servers cannot run here (None when they can)."""
    root = upstream_root()
    if not (root / "src" / "mcp_servers" / "target_mcp" / "server.py").is_file():
        return f"upstream checkout missing at {root} (git submodule update --init)"
    for mod in ("fastmcp", "mcp", "pandas", "pyarrow", "dotenv"):
        if importlib.util.find_spec(mod) is None:
            return f"{mod} not installed"
    return None


def pubmed_hook_missing() -> str | None:
    """The PubMed server must honour ``VBT_EUTILS_BASE`` before its cases can run offline."""
    try:
        text = PUBMED_SERVER.read_text()
    except OSError:
        return f"{PUBMED_SERVER} missing"
    return None if "VBT_EUTILS_BASE" in text else "pubmed_server.py does not honour VBT_EUTILS_BASE yet"


def gateway_missing() -> str | None:
    """Enforce mode needs the gateway package and the shipped overlays."""
    try:
        found = importlib.util.find_spec(GATEWAY_MODULE) is not None
    except ModuleNotFoundError:
        found = False
    if not found:
        return f"{GATEWAY_MODULE} is not implemented yet"
    if not OVERLAY_SENTINEL.is_file():
        return f"{OVERLAY_SENTINEL.relative_to(REPO)} is not shipped yet"
    return None


# ---------------------------------------------------------------------------- parametrization

def upstream_pycache() -> set[str]:
    """``__pycache__`` directories under the upstream checkout."""
    if not UPSTREAM.is_dir():
        return set()
    return {str(p.relative_to(UPSTREAM)) for p in UPSTREAM.rglob("__pycache__") if p.is_dir()}


# Taken when the data-layer conftest first imports this module, before any test launches upstream code.
UPSTREAM_PYCACHE_AT_START = upstream_pycache()

MODES = ("off", "enforce")
HAVE_ARROW = all(importlib.util.find_spec(m) is not None for m in ("pyarrow", "pandas"))

needs_arrow = pytest.mark.skipif(not HAVE_ARROW, reason="pyarrow and pandas are needed for the fixtures")
needs_fastmcp = pytest.mark.skipif(upstream_missing() is not None, reason=upstream_missing() or "")
needs_pubmed_hook = pytest.mark.skipif(pubmed_hook_missing() is not None, reason=pubmed_hook_missing() or "")


def mode_marks(mode: str, *, xfail_off: bool) -> list[Any]:
    marks: list[Any] = [pytest.mark.correctness]
    if mode == "off" and xfail_off:
        # raises=AssertionError: a fixture or start-up error still fails the test instead of xfailing it
        marks.append(pytest.mark.xfail(strict=True, raises=AssertionError,
                                       reason="today's behaviour without the gateway (pinned separately); "
                                              "passes once the gateway enforces"))
    if mode == "enforce":
        reason = gateway_missing()
        marks.append(pytest.mark.skipif(reason is not None, reason=f"enforce mode: {reason}"))
    return marks


def cases(case_ids: Iterable[str], modes: Sequence[str] = MODES, *, xfail_off: bool = True) -> list[Any]:
    """``pytest.param(case, mode)`` for every case and mode, with the mode's marks."""
    return [pytest.param(case, mode, id=f"{case}-{mode}", marks=mode_marks(mode, xfail_off=xfail_off))
            for case in case_ids for mode in modes]


def mode_params(*, xfail_off: bool = False) -> list[Any]:
    return [pytest.param(m, id=m, marks=mode_marks(m, xfail_off=xfail_off)) for m in MODES]



# ---------------------------------------------------------------------------- readiness precondition

READ_TABLES = (
    "target", "disease", "disease_hpo", "disease_phenotype", "drug_molecule", "known_drug", "pharmacogenomics",
    "target_prioritisation", "openfda_significant_adverse_target_reactions",
    "openfda_significant_adverse_drug_reactions", "mouse_phenotype", "l2g_prediction", "interaction", "evidence",
    "association_overall_direct", "association_by_overall_indirect", "association_by_datasource_direct",
    "association_by_datasource_indirect", "colocalisation_coloc", "colocalisation_ecaviar", "study", "go",
    "biosample", "expression",
)
INDEXES = ("ensembl_gene", "ot_disease", "chembl_molecule")
READY = {"ready", "ok", "awaiting_producer", "unbound"}


def readiness_problems(snapshot: Any, tables: Iterable[str] = READ_TABLES, indexes: Iterable[str] = INDEXES
                       ) -> list[str]:
    """Not-ready entries of a readiness snapshot that name one of ``tables`` or ``indexes``.

    The snapshot is walked structurally: any mapping with a ``status`` (or ``ready``) and a name
    (``table``, ``name``, ``id_type``, ``index`` or its key in the parent) counts as one entry.
    """
    wanted = set(tables) | set(indexes)
    out: list[str] = []

    def short(name: str) -> str:
        return name.rsplit(".", 1)[-1] if "." in name else name

    def visit(node: Any, key: str | None) -> None:
        if isinstance(node, Mapping):
            name = next((str(node[k]) for k in ("table", "name", "id_type", "index") if isinstance(node.get(k), str)),
                        key)
            status = node.get("status")
            ready = node.get("ready")
            if name and short(name) in wanted and (status is not None or ready is not None):
                ok = (str(status) in READY) if status is not None else bool(ready)
                if not ok:
                    out.append(f"{name}: {status if status is not None else 'not ready'}")
            for k, v in node.items():
                visit(v, str(k))
        elif isinstance(node, list):
            for v in node:
                visit(v, key)

    visit(snapshot, None)
    return out


def ensure_fixture_ready(bridge: Any) -> None:
    if not getattr(bridge, "gateway_mode", False):
        return
    snapshot = bridge.readiness_snapshot()
    if not snapshot:
        pytest.fail("FIXTURE: the gateway returned no readiness snapshot for the fixture")
    problems = readiness_problems(snapshot)
    if problems:
        pytest.fail("FIXTURE: fixture tables or resolver indexes not ready: " + "; ".join(problems))



# ---------------------------------------------------------------------------- configuration


@dataclass
class DataEnv:
    """Where the fixture data and stubs live; ``env()`` is what each server process gets."""

    ot_root: Path | None = None
    tahoe_root: Path | None = None
    eutils_base: str | None = None
    output_dir: Path | None = None
    extra: dict[str, str] = field(default_factory=dict)

    def env(self) -> dict[str, str]:
        out = {"PYTHONDONTWRITEBYTECODE": "1", "PRELOAD_MCP_DATA": "0", "VBT_UPSTREAM": str(upstream_root())}
        if self.ot_root is not None:
            out["OPEN_TARGETS_DATA_PATH"] = str(self.ot_root)
        if self.tahoe_root is not None:
            out["TAHOE_DATA_PATH"] = str(self.tahoe_root)
        if self.output_dir is not None:
            out["MCP_OUTPUT_DIR"] = str(self.output_dir)
        if self.eutils_base:
            out["VBT_EUTILS_BASE"] = self.eutils_base
            local = "127.0.0.1,localhost"
            for key in ("NO_PROXY", "no_proxy"):
                prev = os.environ.get(key, "")
                out[key] = f"{prev},{local}" if prev else local
        out.update(self.extra)
        return out


def _merge(base: dict[str, Any], extra: Mapping[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def data_overrides(*, gateway: bool, tmp_path: Path) -> dict[str, Any]:
    if not gateway:
        return {"data": {"enabled": False}}
    return {"data": {"enabled": True, "gateway": {"mode": "enforce"}, "cache_dir": str(tmp_path / "dl-cache")}}


def harness_config(*, gateway: bool, tmp_path: Path, env: DataEnv | Mapping[str, str] | None = None,
                overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The ``mock`` profile with the real MCP server list and the data section for ``gateway``."""
    from vbt.config import load_config

    data_env = env.env() if isinstance(env, DataEnv) else dict(env or {})
    base: dict[str, Any] = {
        "paths": {"runs_dir": str(tmp_path / "runs")},
        "mcp_servers_file": "configs/mcp_servers.yaml",
        "vars": {"upstream": str(upstream_root())},
        "tool_env": {k: data_env[k] for k in ("OPEN_TARGETS_DATA_PATH", "TAHOE_DATA_PATH") if k in data_env},
    }
    base = _merge(base, data_overrides(gateway=gateway, tmp_path=tmp_path))
    return load_config(["mock"], overrides=_merge(base, overrides))


def server_specs(config: Mapping[str, Any], servers: Sequence[str], env: Mapping[str, str]) -> list[dict[str, Any]]:
    """The configured entries for ``servers``: ``-B`` first, the stubbed clinicaltrials wrapper,
    and ``env`` added to each server's environment."""
    by_name = {s["name"]: s for s in (config.get("mcp_servers") or {}).get("servers", [])}
    out = []
    for name in servers:
        if name not in by_name:
            raise KeyError(f"server {name!r} is not in configs/mcp_servers.yaml")
        spec = copy.deepcopy(by_name[name])
        args = [a for a in spec.get("args", []) if a != "-B"]
        if name == "clinicaltrials":
            args = [a for a in args if not str(a).endswith(".py")] + [str(CLINICALTRIALS_WRAPPER)]
        spec["args"] = ["-B", *args]
        spec["env"] = {**(spec.get("env") or {}), **env}
        spec["timeout_s"] = min(float(spec.get("timeout_s") or 180), 180.0)
        spec["start_timeout_s"] = 120
        spec["enabled"] = True
        out.append(spec)
    return out


def _require_gateway() -> Any:
    reason = gateway_missing()
    if reason:
        pytest.skip(reason)
    try:
        from vbt.datalayer import build_gateway
    except ImportError as exc:
        pytest.skip(str(exc))
    return build_gateway


def _bridge_configs(specs: Sequence[Mapping[str, Any]]) -> list[Any]:
    from vbt.tools.mcp_bridge import MCPServerConfig

    fields = MCPServerConfig.__dataclass_fields__
    return [MCPServerConfig(**{k: v for k, v in s.items() if k in fields}) for s in specs]


async def start_bridge(servers: Sequence[str], *, env: DataEnv | Mapping[str, str], gateway: bool,
                       tmp_path: Path, overrides: Mapping[str, Any] | None = None) -> Any:
    """Start ``servers`` through ``MCPBridge`` (with the gateway in front when ``gateway``)."""
    from vbt.tools.mcp_bridge import MCPBridge

    data_env = env.env() if isinstance(env, DataEnv) else dict(env)
    config = harness_config(gateway=gateway, tmp_path=tmp_path, env=data_env, overrides=overrides)
    specs = server_specs(config, servers, data_env)
    gw = None
    kwargs: dict[str, Any] = {}
    if gateway:
        build_gateway = _require_gateway()
        gw = build_gateway(config, None)
        specs += list(gw.extra_servers() or [])
        kwargs["gateway"] = gw
    try:
        bridge = MCPBridge(_bridge_configs(specs), log_dir=tmp_path / "mcp-logs", options=FAST_START, **kwargs)
    except TypeError as exc:   # a bridge without the gateway seam
        pytest.skip(f"MCPBridge has no gateway seam yet: {exc}")
    bridge.config = config
    if not hasattr(bridge, "gateway"):
        bridge.gateway = gw
    await bridge.start()
    if bridge.failures:
        tails = {n: bridge.stderr_tail(n, 1500) for n in bridge.failures}
        await bridge.aclose()
        raise RuntimeError(f"FIXTURE: upstream servers failed to start: {bridge.failures} {tails}")
    return bridge


# ---------------------------------------------------------------------------- results


@dataclass
class CallResult:
    """One call's outcome; unpacks as ``(is_error, text, obj, status, kind, payload)``."""

    is_error: bool
    text: str
    obj: Any
    status: str | None
    kind: str | None
    payload: dict[str, Any]
    provenance: Any = None          # the DataResult's provenance record (gateway on)

    def __iter__(self) -> Iterator[Any]:
        return iter((self.is_error, self.text, self.obj, self.status, self.kind, self.payload))

    @property
    def header(self) -> dict[str, Any]:
        """The ``_vbt`` header of a success ({} without one)."""
        return (self.obj.get("_vbt") or {}) if isinstance(self.obj, dict) else {}

    def rows(self, key: str) -> list[Any]:
        """``obj[key]`` as a list ([] when absent)."""
        if not isinstance(self.obj, dict):
            return []
        value = self.obj.get(key)
        return value if isinstance(value, list) else []


def _loads(text: str) -> Any:
    body = text
    if body.startswith("[vbt-data"):            # text results carry the header as the first line
        body = body.split("\n", 1)[1] if "\n" in body else ""
    try:
        return json.loads(body)
    except (TypeError, ValueError):
        return text


def result_of(out: Any) -> CallResult:
    """A successful bridge return value (str, content parts or a DataResult) as a CallResult."""
    if getattr(out, "is_data_result", False):
        obj = out.obj
        status = getattr(out, "status", None)
        return CallResult(False, out.text, obj, status, None, {}, getattr(out, "provenance", None))
    if isinstance(out, list):
        text = "\n".join(str(getattr(p, "text", "") or "") for p in out)
    else:
        text = str(out)
    obj = _loads(text)
    header = obj.get("_vbt") if isinstance(obj, dict) else None
    status = header.get("status") if isinstance(header, dict) else None
    return CallResult(False, text, obj, status, None, {})


def error_of(exc: BaseException) -> CallResult:
    """A tool failure as a CallResult: ``payload`` is the error envelope, ``kind`` its ``kind``."""
    text = str(exc)
    envelope = getattr(exc, "envelope", None)
    payload: Any = envelope() if callable(envelope) else None
    if not isinstance(payload, dict):
        payload = _loads(text)
    if not isinstance(payload, dict):
        payload = {"error": text}
    kind = getattr(exc, "kind", None) or payload.get("kind")
    kind = getattr(kind, "value", kind)
    return CallResult(True, text, payload, "error", kind, payload)


async def call(bridge: Any, server: str, tool: str, args: Mapping[str, Any]) -> CallResult:
    """Call ``server.tool(args)`` through the bridge; failures come back as ``is_error`` results."""
    from vbt.tools.base import ToolFailure

    try:
        out = await bridge.call(server, tool, dict(args))
    except ToolFailure as exc:
        return error_of(exc)
    return result_of(out)


# ---------------------------------------------------------------------------- shared live bridge


class LiveBridge:
    """A started bridge on a private event-loop thread, callable from synchronous tests.

    Results are cached per ``(server, tool, args)``: the data is read-only, and the pinned and
    the required assertions of a case look at the same call.
    """

    def __init__(self, servers: Sequence[str], *, env: DataEnv, gateway: bool, tmp_path: Path,
                 overrides: Mapping[str, Any] | None = None) -> None:
        self.servers = tuple(servers)
        self.env = env
        self.gateway_mode = gateway
        self.tmp_path = tmp_path
        self._cache: dict[str, CallResult] = {}
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, name="live-bridge", daemon=True)
        self._thread.start()
        try:
            self.bridge = self.run(start_bridge(servers, env=env, gateway=gateway, tmp_path=tmp_path,
                                                overrides=overrides), timeout=900)
        except BaseException:
            self._stop_loop()
            raise

    def run(self, coro: Any, timeout: float = 600) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    @property
    def gateway(self) -> Any:
        return getattr(self.bridge, "gateway", None)

    def call(self, server: str, tool: str, args: Mapping[str, Any] | None = None, *, cached: bool = True
             ) -> CallResult:
        key = json.dumps([server, tool, args or {}], sort_keys=True, default=str)
        if cached and key in self._cache:
            return self._cache[key]
        res = self.run(call(self.bridge, server, tool, args or {}))
        self._cache[key] = res
        return res

    def readiness_snapshot(self) -> dict[str, Any]:
        gw = self.gateway
        return gw.readiness_snapshot() if gw is not None else {}

    def stderr(self, server: str) -> str:
        return self.bridge.stderr_tail(server, 4000)

    def _stop_loop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=10)
        self.loop.close()

    def close(self) -> None:
        try:
            self.run(self.bridge.aclose(), timeout=120)
        finally:
            self._stop_loop()


# ---------------------------------------------------------------------------- runtime runs


@dataclass
class RuntimeOutcome:
    """What a scripted runtime run recorded: per call the ``tool_end`` trace event, per claim
    batch the ``record_claims`` output, the stored claims and the run manifest."""

    tool_use_ids: list[str]
    ends: list[dict[str, Any]]
    claim_results: list[dict[str, Any]]
    stored_claims: list[dict[str, Any]]
    manifest: dict[str, Any]
    run_dir: Path

    def end(self, i: int) -> dict[str, Any]:
        return self.ends[i]

    def claim(self, i: int) -> dict[str, Any]:
        return self.claim_results[i]


def _substitute(obj: Any, ids: Sequence[str]) -> Any:
    """Replace ``{"kind": "tool_call", "call": i}`` evidence with the i-th call's tool_use_id."""
    if isinstance(obj, dict):
        if obj.get("kind") == "tool_call" and "call" in obj and "tool_use_id" not in obj:
            out = {k: v for k, v in obj.items() if k != "call"}
            out["tool_use_id"] = ids[int(obj["call"])]
            return out
        return {k: _substitute(v, ids) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute(v, ids) for v in obj]
    return obj


@contextlib.contextmanager
def _no_bytecode() -> Iterator[None]:
    """Harness preflight imports upstream ``tools/doctor.py`` in-process; keep it from writing
    ``__pycache__`` into the upstream checkout during tests."""
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = prev


async def run_with_runtime(calls: Sequence[tuple[str, str, Mapping[str, Any]]],
                           claims: Sequence[Sequence[Mapping[str, Any]]] = (), *,
                           env: DataEnv, gateway: bool, tmp_path: Path,
                           overrides: Mapping[str, Any] | None = None, preflight: bool = False,
                           servers: Sequence[str] | None = None) -> RuntimeOutcome:
    """Run the calls and then one ``record_claims`` per claim batch through ``Runtime``.

    The mock provider issues every call in one assistant turn, then one turn per claim batch;
    claim evidence ``{"kind": "tool_call", "call": i}`` cites the i-th call. With ``preflight``
    the session runs the real readiness gate with ``allow_missing_data`` (a non-mock provider
    name so the gate is not skipped), which is what fills ``MANIFEST.degraded``.
    """
    from vbt.agents import AgentDefinition
    from vbt.orchestrator import open_session
    from vbt.providers.mock import ScriptedProvider, call as tool_call, reply, turn

    servers = list(servers or dict.fromkeys(s for s, _, _ in calls))
    data_env = env.env()
    data_env.pop("MCP_OUTPUT_DIR", None)         # the runtime points it at the run directory
    extra: dict[str, Any] = {"preflight": {"allow_missing_data": True} if preflight else {"skip": True}}
    if preflight:
        extra["orchestration"] = {"require_reference_data": True}
    config = harness_config(gateway=gateway, tmp_path=tmp_path, env=data_env, overrides=_merge(extra, overrides))
    config["mcp_servers"] = {"servers": server_specs(config, servers, data_env)}
    if gateway:
        _require_gateway()

    planned = [tool_call(f"mcp__{s}__{t}", **dict(a)) for s, t, a in calls]
    ids = [c.id for c in planned]
    batches = [tool_call("mcp__provenance__record_claims", claims=_substitute(list(b), ids)) for b in claims]
    script = [turn(*planned)] + [turn(b) for b in batches] + [reply("done")]

    def next_item(agent: str, system: Any, messages: list[Any], tools: list[Any]) -> Any:
        return script.pop(0) if script else reply("done")

    provider = ScriptedProvider(next_item)
    if preflight:
        provider.name = "scripted"            # require_ready skips the gate for the 'mock' provider
    with _no_bytecode():
        session = await open_session(config, provider=provider, start_mcp=True)
    try:
        rt = session.rt
        probe = AgentDefinition(name="data-probe", description="correctness-test probe", prompt="Run the calls.",
                                tier="scientist", tools=[f"mcp__{s}__*" for s in servers]
                                + ["mcp__provenance__record_claims"])
        await rt.run_agent(probe, "Run the scripted calls.", depth=1)
        events = rt.run.events()
        by_id = {e.get("tool_use_id"): e for e in events if e.get("type") == "tool_end"}
        ends = [by_id.get(i, {}) for i in ids]
        claim_results = []
        for b in batches:
            out = by_id.get(b.id, {}).get("output", "")
            claim_results.append(_loads(out) if isinstance(out, str) else out)
        claims_file = rt.run.dir / "evidence" / "claims.json"
        stored = []
        if claims_file.exists():
            data = json.loads(claims_file.read_text())
            stored = data.get("claims", data) if isinstance(data, dict) else data
        manifest = dict(getattr(rt.run, "manifest", {}) or {})
        return RuntimeOutcome(ids, ends, claim_results, list(stored), manifest, rt.run.dir)
    finally:
        await session.close()
