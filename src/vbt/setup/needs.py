"""What the enabled roster needs: the upstream tools some enabled agent may call, the tables their overlays
bind, the sources those tables belong to, and where each local source's files are expected.

Everything is derived from the configuration (``configs/agents.yaml``, ``configs/mcp_servers.yaml``) and the
data layer's descriptors and overlays; nothing here names a dataset. The data layer's own ``data`` server (the
native tools) reads whatever is present and is not a need.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

__all__ = ["Needs", "compute_needs", "root_variables"]

#: The data layer's native-tools server: generic readers over every descriptor, never a reason to fetch data.
NATIVE_SERVER = "data"

_VAR_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::?-[^{}]*(?:\{[^{}]*\}[^{}]*)*)?\}$")


@dataclass
class Needs:
    """``tools``: ``server.tool`` granted to an enabled agent on an enabled server; ``tables``: ``source.table`` ->
    ``{"servers", "tools", "full_load"}`` (``full_load``: some tool loads the whole table upstream);
    ``sources``: source -> ``{"kind", "release", "root", "root_var", "tables"}``."""

    tools: list[str] = field(default_factory=list)
    tables: dict[str, dict[str, Any]] = field(default_factory=dict)
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    servers: dict[str, list[str]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"tools": list(self.tools), "tables": self.tables, "sources": self.sources,
                "servers": self.servers, "errors": list(self.errors)}

    def local_tables(self, *, optional: bool = True) -> list[str]:
        """Tables of local sources (files on this host), sorted; ``optional=False`` leaves out the tables only a
        derived serve's optional dependency names (its tool degrades without them)."""
        return sorted(t for t, rec in self.tables.items()
                      if self.sources.get(t.split(".", 1)[0], {}).get("kind") == "local"
                      and (optional or not rec.get("optional")))

    def server_full_loads(self) -> dict[str, list[str]]:
        """``{server: [tables some tool of that server loads whole]}`` (what the server's memory must hold)."""
        out: dict[str, list[str]] = {}
        for table, rec in self.tables.items():
            for server in rec.get("full_load_servers") or []:
                out.setdefault(server, []).append(table)
        return {k: sorted(v) for k, v in sorted(out.items())}


def root_variables(raw_root: Any) -> str | None:
    """The environment variable a descriptor's ``root`` is exactly (``${NAME}`` or ``${NAME:-default}``), else
    None (a root with a suffix, like ``${DIR:-...}/sub``, is not a single directory variable)."""
    if not isinstance(raw_root, str):
        return None
    m = _VAR_RE.match(raw_root.strip())
    return m.group(1) if m else None


def _raw_roots(descriptors_dir: Path) -> dict[str, Any]:
    import yaml

    out: dict[str, Any] = {}
    for path in sorted(descriptors_dir.glob("*.y*ml")) if descriptors_dir.is_dir() else []:
        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except Exception:  # noqa: BLE001 - lint reports it
            continue
        if isinstance(raw, Mapping) and raw.get("source"):
            out[str(raw["source"])] = raw.get("root")
    return out


def _enabled_servers(config: Mapping[str, Any]) -> list[str]:
    servers = (config.get("mcp_servers") or {}).get("servers") or []
    return [str(s.get("name")) for s in servers if isinstance(s, Mapping) and s.get("enabled", True) is not False]


def compute_needs(config: dict[str, Any]) -> Needs:
    """The needs of the enabled roster (agents + enabled MCP servers) under ``config``."""
    from ..preflight import data_catalog, granted_tools

    needs = Needs()
    try:
        settings, catalog, _registry = data_catalog(config)
    except Exception as exc:  # noqa: BLE001 - reported, not raised: setup goes on without the data layer
        needs.errors.append(f"data catalog: {exc}")
        return needs
    enabled = set(_enabled_servers(config))
    candidates = [f"mcp__{s}__{t}" for s in catalog.servers() if s in enabled and s != NATIVE_SERVER
                  for t in catalog.tools(s)]
    granted = granted_tools(config, candidates)
    raw_roots = _raw_roots(Path(settings.descriptors_dir))
    for name in sorted(granted):
        server, tool = name[len("mcp__"):].split("__", 1)
        try:
            contract = catalog.contract(server, tool)
        except Exception as exc:  # noqa: BLE001
            needs.errors.append(f"{server}.{tool}: {exc}")
            continue
        if contract.binding is None:
            continue
        needs.tools.append(f"{server}.{tool}")
        needs.servers.setdefault(server, []).append(tool)
        full = {str(t) for t in contract.full_table_reads}
        # a derived serve's own dependencies too: required (a compare_with table) and optional (the GO hierarchy
        # get_go_enrichment propagates over; without it the tool answers in its degraded mode: DEP-10)
        try:
            from ..datalayer.gateway.readiness import derived_dependencies

            dep_required, dep_optional = derived_dependencies(contract, catalog)
        except Exception:  # noqa: BLE001
            dep_required, dep_optional = [], []
        refs = [(r, False) for r in contract.tables] + [(r, False) for r in dep_required] + \
            [(r, True) for r in dep_optional]
        for ref, optional in refs:
            try:
                ct = catalog.table(ref)
            except Exception as exc:  # noqa: BLE001
                needs.errors.append(f"{server}.{tool}: table {ref}: {exc}")
                continue
            physical = str(ct.physical) if ct.is_item_table else str(ref)
            rec = needs.tables.setdefault(physical, {"servers": [], "tools": [], "full_load_servers": [],
                                                     "optional": optional})
            rec["optional"] = bool(rec.get("optional")) and optional       # required by any tool: required
            if server not in rec["servers"]:
                rec["servers"].append(server)
            if f"{server}.{tool}" not in rec["tools"]:
                rec["tools"].append(f"{server}.{tool}")
            if (str(ref) in full or physical in full) and server not in rec["full_load_servers"]:
                rec["full_load_servers"].append(server)
    for table in needs.tables:
        source = table.split(".", 1)[0]
        if source in needs.sources:
            needs.sources[source]["tables"].append(table.split(".", 1)[1])
            continue
        try:
            desc = catalog.source(source)
        except Exception as exc:  # noqa: BLE001
            needs.errors.append(f"source {source}: {exc}")
            continue
        release = getattr(desc, "release", None)
        expect = getattr(release, "expect", None) if release is not None else None
        needs.sources[source] = {
            "kind": str(getattr(desc, "kind", "local") or "local"),
            "release": str(expect) if expect else None,
            "root": str(desc.root) if getattr(desc, "root", None) else None,
            "root_var": root_variables(raw_roots.get(source)),
            "tables": [table.split(".", 1)[1]],
        }
    for rec in needs.sources.values():
        rec["tables"].sort()
    needs.tools.sort()
    for server in needs.servers:
        needs.servers[server].sort()
    return needs
