"""Running ``vbt validate``'s data work contained: the oracle under the reaper, and the servers through MCPBridge.

* :func:`run_oracle` runs :mod:`.oracle` (stdlib + pyarrow, never ``vbt``) under the reaper with a memory limit
  (the data child's by default), keeps the reaper's status and returns the answers and the peak.
* :class:`Calls`: one MCPBridge with the gateway in front of the configured servers. ``enforce`` calls go through
  the gateway (``MCPBridge.call``), ``off`` calls to the same unmodified server processes without it
  (``MCPBridge.call_raw``, what a bridge built without a gateway calls), so one set of server processes answers
  both modes. Every call is timed, and the gateway's data-child requests are timed per verb (the witness).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = ["ORACLE", "run_reaped", "run_oracle", "run_data_check_contained", "Outcome", "outcome_of",
           "failure_of", "Calls"]

ORACLE = Path(__file__).resolve().with_name("oracle.py")


def _python(config: Mapping[str, Any]) -> str:
    return str((config.get("vars") or {}).get("mcp_python") or sys.executable)


def run_reaped(config: Mapping[str, Any], argv: Sequence[str], env: Mapping[str, str] | None = None, *,
               limit_mb: int | None = None, name: str = "data", timeout: float = 3600.0
               ) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    """Run ``argv`` under the reaper (the data child's limit and containment unless ``limit_mb``) and return the
    process and the reaper's last status (``peak_rss_mb``, ``containment``, ``limit_mb``, ``exit``). Not on Linux
    the command runs as given."""
    from ..datalayer.launch import DATA_SERVER, build_launch_spec
    from ..datalayer.settings import DataSettings
    from ..tools.mcp_bridge import MCPServerConfig

    settings = DataSettings.from_config(dict(config))
    with tempfile.TemporaryDirectory(prefix="vbt-validate-status-") as tmp:
        cfg = MCPServerConfig(DATA_SERVER if limit_mb is None else name, command=str(argv[0]),
                              args=[str(a) for a in argv[1:]], mem_limit_mb=limit_mb)
        try:
            spec = build_launch_spec(cfg, settings, tmp)
        except Exception:  # noqa: BLE001 - not Linux: run as given
            spec = None
        run_env = dict(env) if env is not None else {k: v for k, v in os.environ.items()
                                                      if not k.startswith("PYTHON")}
        full = list(argv)
        if spec is not None:
            full = [spec.command, *spec.args]
            run_env.update(spec.env)
        proc = subprocess.run(full, capture_output=True, text=True, timeout=timeout, env=run_env)
        status: dict[str, Any] = {}
        if spec is not None:
            try:
                status = json.loads(Path(spec.status_path).read_text())
            except (OSError, ValueError):
                status = {}
        status["rc"] = proc.returncode
    return proc, status


def run_oracle(config: Mapping[str, Any], queries: Sequence[Mapping[str, Any]], *, limit_mb: int | None = None,
               timeout: float = 3600.0, name: str = "validate-oracle") -> tuple[dict[str, Any], dict[str, Any]]:
    """``(answers, status)``: the oracle's answers by query id and the reaper's status. The oracle runs under the
    data child's limit unless ``limit_mb``."""
    with tempfile.TemporaryDirectory(prefix="vbt-validate-") as tmp:
        qpath, apath = Path(tmp) / "queries.json", Path(tmp) / "answers.json"
        qpath.write_text(json.dumps(list(queries), default=str), encoding="utf-8")
        proc, status = run_reaped(config, [_python(config), "-E", str(ORACLE), str(qpath), str(apath)],
                                  limit_mb=limit_mb, name=name, timeout=timeout)
        try:
            answers = json.loads(apath.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            answers = {"_error": (proc.stderr or proc.stdout or "").strip()[-800:] or f"exit {proc.returncode}"}
    return answers, status


def run_data_check_contained(config: Mapping[str, Any], *, depth: str, tables: Sequence[str] = (),
                             timeout: float = 24 * 3600.0) -> tuple[dict[str, Any], dict[str, Any]]:
    """The data child's ``--check --json`` (``vbt ds check``) under the reaper, in the session check's groups (the
    large tables each in a process of their own, :func:`vbt.preflight.check_groups`): ``(CheckResponse, status)``
    with the largest group's reaper status."""
    from ..preflight import DataCheckUnavailable, _merge_checks, check_groups, data_child_command

    responses: list[dict[str, Any]] = []
    worst: dict[str, Any] = {}
    deadline = time.monotonic() + timeout
    groups = check_groups(dict(config), tables)
    for group in groups:
        args = ["--check", "--json", "--depth", depth]
        for t in group:
            args += ["--table", str(t)]
        cmd, env = data_child_command(dict(config), *args)
        proc, status = run_reaped(config, cmd, env, timeout=max(60.0, deadline - time.monotonic()))
        line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("{")), None)
        if proc.returncode != 0 or line is None:
            raise DataCheckUnavailable(f"{cmd[0]} exited {proc.returncode}: "
                                       f"{(proc.stderr or proc.stdout).strip()[-500:]}")
        responses.append(json.loads(line))
        if float(status.get("peak_rss_mb") or 0) >= float(worst.get("peak_rss_mb") or 0):
            worst = {**status, "tables": group[:5]}
    worst["groups"] = len(groups)
    return (_merge_checks(responses) if len(responses) > 1 else responses[0]), worst


@dataclass
class Outcome:
    """One call's outcome (the shape the judges read)."""

    is_error: bool
    kind: str | None
    obj: Any
    text: str
    seconds: float
    provenance: Any = None

    @property
    def header(self) -> dict[str, Any]:
        return (self.obj.get("_vbt") or {}) if isinstance(self.obj, dict) else {}

    @property
    def status(self) -> str | None:
        return "error" if self.is_error else self.header.get("status")


def _loads(text: str) -> Any:
    body = text
    if body.startswith("[vbt-data"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
    try:
        return json.loads(body)
    except (TypeError, ValueError):
        return text


def outcome_of(out: Any, seconds: float) -> Outcome:
    if getattr(out, "is_data_result", False):
        return Outcome(False, None, out.obj, str(out.text), seconds, getattr(out, "provenance", None))
    text = "\n".join(str(getattr(p, "text", "") or "") for p in out) if isinstance(out, list) else str(out)
    return Outcome(False, None, _loads(text), text, seconds)


def failure_of(exc: BaseException, seconds: float) -> Outcome:
    text = str(exc)
    envelope = getattr(exc, "envelope", None)
    payload: Any = envelope() if callable(envelope) else None
    if not isinstance(payload, dict):
        payload = _loads(text)
    if not isinstance(payload, dict):
        payload = {"error": text}
    kind = getattr(exc, "kind", None) or payload.get("kind")
    return Outcome(True, getattr(kind, "value", kind), payload, text, seconds)


@dataclass
class Calls:
    """One started bridge (gateway in front) and its timings; use as ``async with Calls(config, servers)``."""

    config: Mapping[str, Any]
    servers: Sequence[str]
    log_dir: Path
    readiness: Mapping[str, Any] | None = None       # a CheckResponse to hand the gateway (no second session check)
    timeout_s: float = 600.0
    bridge: Any = None
    gateway: Any = None
    verb_times: list[tuple[str, str, str, float]] = field(default_factory=list)   # (server, tool, verb, seconds)
    _current: tuple[str, str] = ("", "")
    started_s: float = 0.0
    failures: dict[str, str] = field(default_factory=dict)

    async def __aenter__(self) -> "Calls":
        from ..datalayer import build_gateway
        from ..preflight import base_tool_env
        from ..tools.mcp_bridge import MCPBridge, MCPServerConfig

        wanted = set(self.servers)
        raw = [dict(s) for s in (self.config.get("mcp_servers") or {}).get("servers", [])
               if s.get("name") in wanted and s.get("enabled", True)]
        tmp = self.log_dir / "run"
        tmp.mkdir(parents=True, exist_ok=True)
        self.gateway = build_gateway(dict(self.config), {"dir": str(tmp), "run_id": "validate",
                                                         "mcp_output_dir": str(tmp / "mcp")})
        names = {s.get("name") for s in raw}
        raw += [s for s in self.gateway.extra_servers() or [] if s.get("name") not in names]
        specs = [MCPServerConfig(**{k: v for k, v in s.items() if k in MCPServerConfig.__dataclass_fields__})
                 for s in raw]
        extra = base_tool_env(dict(self.config))
        extra.update({"VBT_RUN_DIR": str(tmp), "MCP_OUTPUT_DIR": str(tmp / "mcp")})
        options = {**(self.config.get("mcp") or {}), "default_timeout_s": self.timeout_s}
        self.bridge = MCPBridge(specs, extra_env=extra, log_dir=self.log_dir / "logs", options=options,
                                gateway=self.gateway)
        self._instrument()
        t0 = time.monotonic()
        await self.bridge.start()
        self.started_s = time.monotonic() - t0
        self.failures = dict(self.bridge.failures or {})
        if self.readiness:
            self.gateway.set_readiness(self.readiness)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        close = getattr(self.gateway, "aclose", None)
        if callable(close):
            try:
                await close()
            except Exception:  # noqa: BLE001
                pass
        if self.bridge is not None:
            await self.bridge.aclose()

    def _instrument(self) -> None:
        """Time the gateway's data-child requests per verb (``_witness``, ``_serve``, ...) for the current call."""
        service = getattr(self.gateway, "service", None)
        call = getattr(service, "call", None)
        if not callable(call):
            return
        calls = self

        async def timed(verb: str, request: Any) -> Any:
            t0 = time.monotonic()
            try:
                return await call(verb, request)
            finally:
                calls.verb_times.append((*calls._current, verb, time.monotonic() - t0))

        service.call = timed

    async def call(self, server: str, tool: str, args: Mapping[str, Any], *, mode: str = "enforce") -> Outcome:
        from ..tools.base import ToolFailure

        self._current = (server, tool)
        t0 = time.monotonic()
        try:
            fn = self.bridge.call if mode == "enforce" else self.bridge.call_raw
            out = await asyncio.wait_for(fn(server, tool, dict(args)), self.timeout_s + 30)
        except ToolFailure as exc:
            return failure_of(exc, time.monotonic() - t0)
        except asyncio.TimeoutError:
            return Outcome(True, "timeout", {"error": f"no answer in {self.timeout_s:.0f} s"}, "timeout",
                           time.monotonic() - t0)
        except Exception as exc:  # noqa: BLE001 - a broken call is a finding, not a crash of validate
            return Outcome(True, type(exc).__name__, {"error": str(exc)[:800]}, str(exc)[:800], time.monotonic() - t0)
        finally:
            self._current = ("", "")
        return outcome_of(out, time.monotonic() - t0)

    def tool_schemas(self, server: str) -> dict[str, dict[str, Any]]:
        """``{tool: upstream input schema}`` as the server listed it (before the gateway's rewrite)."""
        schemas = getattr(self.gateway, "_schemas", None) or {}
        out = {t: dict(s) for (srv, t), s in schemas.items() if srv == server}
        if out:
            return out
        return {t.name.split("__", 2)[-1]: dict(t.input_schema or {}) for t in self.bridge.tools
                if t.name.startswith(f"mcp__{server}__")}

    def server_status(self) -> dict[str, dict[str, Any]]:
        """Each server's reaper status (``peak_rss_mb``, ``containment``, ``limit_mb``)."""
        out: dict[str, dict[str, Any]] = {}
        for name, st in (self.bridge.status() or {}).items():
            path = st.get("status_file")
            try:
                out[name] = json.loads(Path(path).read_text()) if path else {}
            except (OSError, ValueError):
                out[name] = {}
        return out
