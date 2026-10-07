"""``data.*`` configuration with in-code defaults (§17). No pyarrow.

``DataSettings.from_config(config)`` deep-merges ``config["data"]`` over :data:`DATA_DEFAULTS`,
expands ``${VAR:-x}`` / ``${vars.x}`` with the same helpers as :mod:`vbt.config`, and resolves
the descriptor, overlay and cache directories against the project root, so hand-built
configs work before ``configs/default.yaml`` carries the ``data`` block.

The data child receives the same settings as JSON in ``VBT_DATA_SETTINGS``
(:meth:`DataSettings.to_json` / :meth:`DataSettings.from_json`).
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from .. import config as _config

__all__ = [
    "DATA_DEFAULTS", "SETTINGS_ENV", "GatewaySettings", "ServiceSettings", "ResolutionSettings", "WitnessSettings",
    "DeriveSettings", "MemorySettings", "ReadinessSettings", "ResultsSettings", "LeakageSettings",
    "SourcesSettings", "ProvenanceSettings", "PluginsSettings", "DataSettings",
]

SETTINGS_ENV = "VBT_DATA_SETTINGS"

#: §17, verbatim. Mirrored with comments in configs/default.yaml.
DATA_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "descriptors_dir": "configs/data/sources",
    "overlays_dir": "configs/data/overlays",
    "cache_dir": "${VBT_DATA_DIR:-data}/.vbt-datalayer",
    "gateway": {
        "mode": "enforce",
        "enforce_servers": "all",
        "profile": "safe",
        "when_service_down": "strict",
        "unbound_empty": "empty_unverified",
    },
    "service": {"mem_limit_mb": 3000, "max_concurrency": 4, "timeout_s": 600, "max_resident_mb": 2000},
    "resolution": {
        "max_candidates": 10,
        "allow": ["raw_member", "normalized", "label", "previous", "alias", "exact_synonym", "related_synonym",
                  "retired", "xref", "crosswalk", "parent_family"],
        "max_hops": 2,
        "max_expand": 5000,
        "min_resolved_fraction": 0.95,
        "remote_ttl_s": 3600,
    },
    "witness": {
        "enabled": True,
        "max_scan_bytes": 2_000_000_000,
        "max_inflate_rows": 5000,
        "max_inflate_bytes": 20_000_000,
        "max_key_set": 20000,
        "repair_max_bytes": 500_000_000,
        "topk": True,
    },
    "derive": {"enum_max": 64, "description_max_chars": 1200},
    "memory": {
        "default_server_mb": 12000,
        "limit_kind": "rlimit_data",
        "estimate_safety": 1.3,
        "expansion": {"flat": 1.5, "string": 3.5, "nested": 8.0, "fragmentation": 1.15},
        "object_overhead_bytes": {"string": 50, "nested_item": 120, "struct_item": 240},
        "recycle_idle_servers": True,
        "recycle_wait_s": 30,
        "max_recycles_per_10min": 4,
        "max_oom_kills": 3,
        "max_result_bytes": 2_000_000,
        "host_budget_mb": "auto",
    },
    "readiness": {
        "per_turn_depth": "shallow",
        "session_depth": "standard",
        "key_check_full_max_rows": 50_000_000,
        "vocab_budget_bytes": 500_000_000,
        "remote_ttl_s": 3600,
        "sentinels": True,
        "block_when": "all_unready",
    },
    "results": {"relation_list_max": 50},
    "leakage": {"ceiling": None},
    "sources": {"alias": {}},
    "provenance": {"row_keys_max": 10000, "dir": "logs/data_provenance"},
    "plugins": {"paths": [], "entry_points": True, "disabled": [], "override": {}, "require_conformance": False},
}


def _tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(value)
    return (value,)


_GATEWAY_CHOICES: dict[str, tuple[str, ...]] = {
    "mode": ("off", "observe", "enforce"), "profile": ("safe", "fidelity"),
    "when_service_down": ("strict", "lenient"), "unbound_empty": ("empty_unverified", "error"),
}


@dataclass(frozen=True)
class GatewaySettings:
    mode: Literal["off", "observe", "enforce"] = "enforce"
    enforce_servers: str | tuple[str, ...] = "all"
    profile: Literal["safe", "fidelity"] = "safe"
    when_service_down: Literal["strict", "lenient"] = "strict"
    unbound_empty: Literal["empty_unverified", "error"] = "empty_unverified"

    def __post_init__(self) -> None:
        # YAML 1.1 reads an unquoted `mode: off` as false (and `on` as true): map them back, and refuse any
        # value outside the declared set, so a typo never silently runs the gateway in another mode
        for name, allowed in _GATEWAY_CHOICES.items():
            value = getattr(self, name)
            if isinstance(value, bool):
                value = {False: "off", True: "on"}[value] if name == "mode" else value
                object.__setattr__(self, name, value)
            if value not in allowed:
                raise ValueError(f"data.gateway.{name} must be one of {', '.join(allowed)} (got {value!r}); "
                                 f"quote the value in YAML (\"off\")")

    def enforces(self, server: str) -> bool:
        """True when calls to ``server`` are enforced (not only observed)."""
        if self.mode != "enforce":
            return False
        return self.enforce_servers == "all" or server in _tuple(self.enforce_servers)


@dataclass(frozen=True)
class ServiceSettings:
    mem_limit_mb: int = 3000
    max_concurrency: int = 4
    timeout_s: float = 600
    max_resident_mb: int = 2000


@dataclass(frozen=True)
class ResolutionSettings:
    max_candidates: int = 10
    allow: tuple[str, ...] = tuple(DATA_DEFAULTS["resolution"]["allow"])
    max_hops: int = 2
    max_expand: int = 5000
    min_resolved_fraction: float = 0.95
    remote_ttl_s: float = 3600


@dataclass(frozen=True)
class WitnessSettings:
    enabled: bool = True
    max_scan_bytes: int = 2_000_000_000
    max_inflate_rows: int = 5000
    max_inflate_bytes: int = 20_000_000
    max_key_set: int = 20000
    repair_max_bytes: int = 500_000_000
    topk: bool = True


@dataclass(frozen=True)
class DeriveSettings:
    enum_max: int = 64
    description_max_chars: int = 1200


@dataclass(frozen=True)
class MemorySettings:
    default_server_mb: int | str = 12000            # 'auto' derives from estimates (§14.2)
    limit_kind: Literal["rlimit_data", "cgroup", "watchdog", "none"] = "rlimit_data"
    estimate_safety: float = 1.3
    expansion: Mapping[str, float] = field(default_factory=lambda: dict(DATA_DEFAULTS["memory"]["expansion"]))
    object_overhead_bytes: Mapping[str, int] = field(
        default_factory=lambda: dict(DATA_DEFAULTS["memory"]["object_overhead_bytes"]))
    recycle_idle_servers: bool = True
    recycle_wait_s: float = 30
    max_recycles_per_10min: int = 4
    max_oom_kills: int = 3
    max_result_bytes: int = 2_000_000
    host_budget_mb: int | str = "auto"


@dataclass(frozen=True)
class ReadinessSettings:
    per_turn_depth: Literal["shallow", "standard", "deep"] = "shallow"
    session_depth: Literal["shallow", "standard", "deep"] = "standard"
    key_check_full_max_rows: int = 50_000_000
    vocab_budget_bytes: int = 500_000_000
    remote_ttl_s: float = 3600
    sentinels: bool = True
    block_when: str = "all_unready"


@dataclass(frozen=True)
class ResultsSettings:
    relation_list_max: int = 50


@dataclass(frozen=True)
class LeakageSettings:
    ceiling: str | None = None


@dataclass(frozen=True)
class SourcesSettings:
    alias: Mapping[str, str] = field(default_factory=dict)   # {open_targets: zenodo_vbt}


@dataclass(frozen=True)
class ProvenanceSettings:
    row_keys_max: int = 10000
    dir: str = "logs/data_provenance"


@dataclass(frozen=True)
class PluginsSettings:
    paths: tuple[str, ...] = ()
    entry_points: bool = True
    disabled: tuple[str, ...] = ()
    override: Mapping[str, str] = field(default_factory=dict)
    require_conformance: bool = False


_SECTIONS: dict[str, type] = {
    "gateway": GatewaySettings, "service": ServiceSettings, "resolution": ResolutionSettings,
    "witness": WitnessSettings, "derive": DeriveSettings, "memory": MemorySettings,
    "readiness": ReadinessSettings, "results": ResultsSettings, "leakage": LeakageSettings,
    "sources": SourcesSettings, "provenance": ProvenanceSettings, "plugins": PluginsSettings,
}


def _section(cls: type, data: Mapping[str, Any] | None) -> Any:
    data = dict(data or {})
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        default = f.default
        if isinstance(default, tuple) or f.name in ("allow", "paths", "disabled"):
            value = _tuple(value)
        elif f.name == "enforce_servers" and not isinstance(value, str):
            value = _tuple(value)
        elif isinstance(value, Mapping):
            value = dict(value)
        kwargs[f.name] = value
    return cls(**kwargs)


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _plain(v) for k, v in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class DataSettings:
    """Typed ``data.*`` settings. ``raw`` keeps the merged mapping (unknown keys included)."""

    enabled: bool = True
    descriptors_dir: Path = Path("configs/data/sources")
    overlays_dir: Path = Path("configs/data/overlays")
    cache_dir: Path = Path("data/.vbt-datalayer")
    project_root: Path = _config.PROJECT_ROOT
    gateway: GatewaySettings = field(default_factory=GatewaySettings)
    service: ServiceSettings = field(default_factory=ServiceSettings)
    resolution: ResolutionSettings = field(default_factory=ResolutionSettings)
    witness: WitnessSettings = field(default_factory=WitnessSettings)
    derive: DeriveSettings = field(default_factory=DeriveSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    readiness: ReadinessSettings = field(default_factory=ReadinessSettings)
    results: ResultsSettings = field(default_factory=ResultsSettings)
    leakage: LeakageSettings = field(default_factory=LeakageSettings)
    sources: SourcesSettings = field(default_factory=SourcesSettings)
    provenance: ProvenanceSettings = field(default_factory=ProvenanceSettings)
    plugins: PluginsSettings = field(default_factory=PluginsSettings)
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None) -> "DataSettings":
        """Merge ``config["data"]`` over the defaults, expand variables and resolve directories."""
        config = config or {}
        variables = {str(k): "" if v is None else str(v) for k, v in (config.get("vars") or {}).items()}
        root = Path(variables.get("project_root") or _config.PROJECT_ROOT)
        return cls.from_dict(dict(config.get("data") or {}), project_root=root, variables=variables)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, project_root: Path | None = None,
                  variables: Mapping[str, str] | None = None) -> "DataSettings":
        """Build from a ``data`` mapping: missing keys take the defaults, ``${...}`` is expanded
        (``vars.*`` from ``variables``) and relative directories are taken from the project root."""
        data = _config.deep_merge(DATA_DEFAULTS, dict(data or {}))
        root = Path(project_root or data.get("project_root") or _config.PROJECT_ROOT)
        variables = dict(variables or {})
        variables.setdefault("project_root", str(root))
        data = _config._expand(data, variables)

        def _dir(value: Any) -> Path:
            p = Path(str(value)).expanduser()
            return p if p.is_absolute() else (root / p)

        return cls(
            enabled=bool(data.get("enabled", True)),
            descriptors_dir=_dir(data["descriptors_dir"]),
            overlays_dir=_dir(data["overlays_dir"]),
            cache_dir=_dir(data["cache_dir"]),
            project_root=root,
            raw=copy.deepcopy(data),
            **{name: _section(sec, data.get(name)) for name, sec in _SECTIONS.items()},
        )

    def to_dict(self) -> dict[str, Any]:
        out = {k: _plain(v) for k, v in asdict(self).items() if k not in ("raw",)}
        for name in _SECTIONS:
            out[name] = _plain(getattr(self, name))
        return out

    def to_json(self) -> str:
        """JSON for ``VBT_DATA_SETTINGS`` (directories absolute)."""
        return json.dumps(self.to_dict(), default=str, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "DataSettings":
        data = json.loads(text)
        return cls.from_dict(data, project_root=Path(data.get("project_root") or _config.PROJECT_ROOT))

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "DataSettings":
        """Settings from ``VBT_DATA_SETTINGS`` (the data child), else the defaults."""
        text = (environ if environ is not None else os.environ).get(SETTINGS_ENV)
        return cls.from_json(text) if text else cls.from_dict({})
