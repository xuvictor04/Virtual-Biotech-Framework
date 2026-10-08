"""What to acquire for a request, and what a table unlocks: overlays -> tools -> tables. No pyarrow.

* :func:`tool_tables` is every table a reviewed tool's binding can read: its ``reads`` (any access, including the
  upstream server's own reads, whatever the arguments select), the bound and derived tables and their sections,
  ``rows_of``, result sections and coverage universes, each mapped to its physical table (an item table shares
  its parent's files).
* :func:`resolve_targets` turns ``SOURCE``, ``SOURCE.TABLE`` (or ``SOURCE.GROUP`` for an ``extra`` download
  group), ``--for-tools`` names (``server.tool``, ``mcp__server__tool``, globs) and ``--for-agents`` names (their
  tool lists in ``configs/agents.yaml``) into ``{source: [names]}`` with the reasons each was chosen.
* :func:`tools_by_table` is the reverse map ``vbt data status`` prints ("unlocks").
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["Targets", "tool_tables", "tools_by_table", "resolve_targets", "expand_tools", "agent_tools",
           "physical"]


def physical(catalog: Any, ref: str) -> str | None:
    """``source.table`` of the files ``ref`` reads (its parent for an item table); None when unknown."""
    try:
        t = catalog.table(ref)
    except Exception:  # noqa: BLE001 - an unknown or quarantined table is reported by the caller
        return None
    return str(t.physical)


def tool_tables(catalog: Any, server: str, tool: str) -> list[str]:
    """Every physical table the binding of ``server.tool`` can read (see the module docstring)."""
    try:
        contract = catalog.contract(server, tool)
    except Exception:  # noqa: BLE001
        return []
    b = getattr(contract, "binding", None)
    if b is None or contract.generic:
        return []
    refs: list[str] = list(b.reads)
    refs.append(contract.bound_table or "")
    for name in getattr(contract, "selector_args", []):
        a = contract.args[name]
        refs.extend(".".join(str(t).split(".")[:2]) for t in a.values.values() if isinstance(t, str))
        if isinstance(a.binds, dict):
            refs.extend(".".join(str(t).split(".")[:2]) for t in a.binds.values())
    if b.derived is not None:
        refs.append(b.derived.table)
        refs.extend(s.table for s in b.derived.sections.values())
    refs.append(b.result.rows_of or "")
    refs.extend(s.table for s in b.result.sections.values())
    out: list[str] = []
    for ref in refs:
        if not ref or "." not in ref:
            continue
        phys = physical(catalog, ref)
        if phys and phys not in out:
            out.append(phys)
    for ref in list(out):                              # coverage universes are read to decide absence
        try:
            spec = catalog.table(ref).spec
        except Exception:  # noqa: BLE001
            continue
        cov = getattr(spec, "coverage", None)
        if cov is not None and cov.universe is not None:
            u = cov.universe.table
            phys = physical(catalog, u if "." in u else f"{ref.split('.')[0]}.{u}")
            if phys and phys not in out:
                out.append(phys)
    return out


def tools_by_table(catalog: Any) -> dict[str, list[str]]:
    """``{source.table: [mcp__server__tool, ...]}`` over every reviewed binding."""
    out: dict[str, list[str]] = {}
    for server in catalog.servers():
        for tool in catalog.tools(server):
            for ref in tool_tables(catalog, server, tool):
                out.setdefault(ref, []).append(f"mcp__{server}__{tool}")
    return {k: sorted(set(v)) for k, v in out.items()}


def _split(name: str) -> tuple[str, str]:
    text = str(name).strip()
    if text.startswith("mcp__"):
        server, _, tool = text[5:].partition("__")
    else:
        server, _, tool = text.partition(".")
    return server, tool or "*"


def expand_tools(catalog: Any, names: Iterable[str]) -> tuple[list[tuple[str, str]], list[str]]:
    """``([(server, tool)], unmatched names)``: tool names and globs against the reviewed bindings."""
    found: list[tuple[str, str]] = []
    missing: list[str] = []
    servers = list(catalog.servers())
    for name in names:
        s_pat, t_pat = _split(name)
        hit = [(s, t) for s in servers if fnmatch.fnmatchcase(s, s_pat)
               for t in catalog.tools(s) if fnmatch.fnmatchcase(t, t_pat)]
        if not hit:
            missing.append(name)
        found.extend(h for h in hit if h not in found)
    return found, missing


def agent_tools(config: Mapping[str, Any], agents: Iterable[str]) -> tuple[dict[str, list[str]], list[str]]:
    """``({agent: tool names}, unknown agents)`` from the configuration's roster (``agents.yaml``)."""
    spec = dict(config.get("agents") or {})
    roster = {"cso": spec.get("cso") or {}, **dict(spec.get("agents") or {})}

    def flat(items: Any) -> list[str]:
        if items is None:
            return []
        if not isinstance(items, list):
            return [str(items)]
        out: list[str] = []
        for i in items:
            out.extend(flat(i) if isinstance(i, list) else [str(i)])
        return out

    found: dict[str, list[str]] = {}
    unknown: list[str] = []
    for a in agents:
        if a not in roster or not isinstance(roster[a], Mapping):
            unknown.append(a)
            continue
        found[a] = [t for t in flat(roster[a].get("tools")) if t.startswith("mcp__")]
    return found, unknown


@dataclass
class Targets:
    """``wanted``: ``{source: [tables or extra groups]}``; ``why``: ``{source.name: [reasons]}``; ``notes``: what
    could not be planned (unknown names, remote tables, tables without an acquisition entry)."""

    wanted: dict[str, list[str]] = field(default_factory=dict)
    why: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add(self, catalog: Any, ref: str, reason: str) -> None:
        source, _, name = ref.partition(".")
        try:
            desc = catalog.source(source)
        except Exception:  # noqa: BLE001
            self.errors.append(f"{ref}: unknown source {source!r}")
            return
        acq = desc.acquisition
        if desc.kind == "remote" or (acq is not None and acq.mode == "remote"):
            self.notes.append(f"{ref}: {source} is read live (nothing to acquire)")
            return
        if acq is None:
            self.notes.append(f"{ref}: {source} declares no acquisition section")
            return
        if name not in acq.tables and name not in acq.extra:
            if name in desc.tables:
                self.notes.append(f"{ref}: no acquisition entry (the table is written by a tool call or another step)")
            else:
                self.errors.append(f"{ref}: {source} has no table or download group {name!r}")
            return
        names = self.wanted.setdefault(source, [])
        if name not in names:
            names.append(name)
        reasons = self.why.setdefault(ref, [])
        if reason not in reasons:
            reasons.append(reason)


def resolve_targets(catalog: Any, config: Mapping[str, Any], names: Sequence[str] = (), *,
                    tools: Sequence[str] = (), agents: Sequence[str] = (),
                    include_optional: bool = False) -> Targets:
    """See the module docstring. A bare ``SOURCE`` means every table of its acquisition section (and the extra
    groups not marked ``optional``, or all of them with ``include_optional``)."""
    out = Targets()
    for name in names:
        source, sep, rest = str(name).partition(".")
        if not sep:
            try:
                desc = catalog.source(source)
            except Exception:  # noqa: BLE001
                out.errors.append(f"{name}: unknown source")
                continue
            acq = desc.acquisition
            if desc.kind == "remote" or (acq is not None and acq.mode == "remote"):
                out.notes.append(f"{name}: read live (nothing to acquire)")
                if acq is not None and acq.mode == "remote":
                    out.wanted.setdefault(source, [])
                continue
            if acq is None:
                out.notes.append(f"{name}: the descriptor declares no acquisition section")
                continue
            for t in acq.tables:
                out.add(catalog, f"{source}.{t}", "requested")
            for g, e in acq.extra.items():
                if include_optional or not e.optional:
                    out.add(catalog, f"{source}.{g}", "requested")
            continue
        try:
            ref = physical(catalog, name) or name
        except Exception:  # noqa: BLE001
            ref = name
        if ref not in (name,) and "." in ref:
            out.add(catalog, ref, f"requested ({name} is an item table of it)")
        else:
            out.add(catalog, name, "requested")
    tool_names = list(tools)
    for agent, ts in agent_tools(config, agents)[0].items():
        for t in ts:
            if t.startswith("mcp__data__"):
                out.notes.append(f"{agent}: {t} reads whatever table a call names (not counted)")
                continue
            tool_names.append(t)
            out.why.setdefault(f"agent:{agent}", []).append(t)
    for a in agent_tools(config, agents)[1]:
        out.errors.append(f"{a}: unknown agent (configs/agents.yaml)")
    found, missing = expand_tools(catalog, tool_names)
    for name in missing:
        s, t = _split(name)
        if s in set(catalog.servers()) or s in {"data", "provenance", "pubmed"}:
            out.notes.append(f"{name}: no reviewed binding names its tables")
        else:
            out.notes.append(f"{name}: not a data tool (no overlay binding)")
    for server, tool in found:
        for ref in tool_tables(catalog, server, tool):
            out.add(catalog, ref, f"mcp__{server}__{tool}")
    return out
