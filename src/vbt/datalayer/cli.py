"""``vbt datasource`` (alias ``vbt ds``): inspect and check the data layer (docs/DATA_LAYER.md §16).

``vbt data`` is the Zenodo fetcher, hence the longer name. Phase-1 commands::

    vbt ds list                                    sources, tables, release, readiness summary
    vbt ds describe <source>[.<table>]             grain, key, roles, coverage, vocabularies
    vbt ds lint [--strict]                         descriptors and overlays
    vbt ds check [--depth D] [--table S.T] [--column S.T.C] [--tool server.tool] [--json]
    vbt ds resolve <id_type> <value...>            resolutions with rules and candidates
    vbt ds explain <server>.<tool> | --all         binding, serve mode, reads, derived text, defects
    vbt ds fingerprint [--write]                   table fingerprints (what gets pinned)
    vbt ds index build [--id-type T] [--access-paths [--huge]]
    vbt ds estimate --table S.T | --tool server.tool
    vbt ds retro-audit <run>                       re-classify a recorded run offline

Phase 4::

    vbt ds status [<run> | --log-dir DIR]          per-server memory (reaper status files), host budget,
                                                   calibrations and measured feedback
    vbt ds calibrate --table S.T [--row-groups N]  sample-and-scale memory calibration (data child)
    vbt ds overlay init <server> [--from-json F]   scaffold an overlay for a third-party MCP server

Phase 5::

    vbt ds replay <run> <tool_use_id...> | --all   re-execute recorded data calls on current data and compare
                                                   row keys, output rows, computed values, fingerprints
    vbt ds diff-release --from A --to B [--source S] role columns, types, encodings, vocabularies and
                                                   matrix axes between two releases (data child)
    vbt ds explain --all --json [--schemas]        bindings and upstream input schemas as JSON (the CI
                                                   snapshot tests/datalayer/upstream_tool_schemas.json)
    vbt ds graduate [server...] [--run R...]       the observe -> enforce checklist per server, with
                                                   retro-audit evidence from recorded runs

Handlers follow the repo convention ``handler(args, config) -> int``. Nothing here imports pyarrow:
commands that read data (``check``, ``fingerprint``, ``index build``, ``estimate``) run the data
child's command line (``<mcp_python> -E src/vbt/datalayer/service/server.py ...``) as a subprocess.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

__all__ = ["add_datasource_parsers", "ALIASES", "COMMANDS"]

ALIASES = ("ds",)
#: Python run by ``vbt ds estimate`` in the data child's interpreter: the ``_stats`` verb over the
#: tables, printed as one JSON line (``python -E -c`` with ``src/`` put on sys.path first).
_STATS_SNIPPET = (
    "import json, sys; sys.path.insert(0, sys.argv[1]); "
    "from vbt.datalayer.service import ServiceContext; "
    "from vbt.datalayer.service.verbs import load_verbs; "
    "from vbt.datalayer.settings import DataSettings; "
    "ctx = ServiceContext(DataSettings.from_env()); "
    "print(json.dumps(load_verbs()['_stats'](ctx, {'tables': sys.argv[2:]}), sort_keys=True, default=str))"
)


#: Python run by commands that call one data-child function ``module:function(ctx, payload)`` in the data
#: child's interpreter (``python -E -c``; ``src/`` is put on sys.path first), printed as one JSON line.
_CALL_SNIPPET = (
    "import importlib, json, sys; sys.path.insert(0, sys.argv[1]); "
    "from vbt.datalayer.service import ServiceContext; "
    "from vbt.datalayer.settings import DataSettings; "
    "mod, fn = sys.argv[2].split(':'); f = getattr(importlib.import_module(mod), fn); "
    "ctx = ServiceContext(DataSettings.from_env()); "
    "print(json.dumps(f(ctx, json.loads(sys.argv[3])), sort_keys=True, default=str))"
)

#: How to bind a third-party server: the header of every ``vbt ds overlay init`` scaffold and the
#: command's help (docs/DATA_LAYER.md §8.6, §9.5, §11.9).
THIRD_PARTY_HOWTO = """\
How to add a third-party MCP server (docs/DATA_LAYER.md §8.6, §9.5, §11.9):
 1. Add the server to configs/mcp_servers.yaml (command + args for stdio, or url: for HTTP). Until an
    overlay reviews its tools, the generic guard applies: explicit not-found shapes become not_found,
    structural empties are uncitable (empty_unverified), HTTP 5xx is source_error.
 2. Run `vbt ds overlay init <server>` (it lists the tools and writes configs/data/overlays/<server>.yaml).
    Every tool is `status: unreviewed`; every argument is `role: unbound` or a guessed role (limit,
    free_text, output_path). `guess:` comments name identifier kinds whose plugin recognised the
    examples in the tool's docstring (looks_like); they are hints, not bindings.
 3. If the server reads a dataset the catalog does not describe, add a descriptor under
    configs/data/sources/ (kind: remote for live APIs; layout live_api with a count endpoint gives
    the remote witness). Opaque IDs are a local_key id_type with YAML options, not new code.
 4. Review each tool: bind arguments (`binds: <source>.<table>.<column>`, `accepts: [<id_type>]`),
    describe the result (`rows`, `total`, `not_found_when`: an HTTP 404 means not found only when it is
    declared), and set `status: reviewed`. A payload JSONPaths cannot describe gets an envelope plugin
    (kind `envelope`, entry point group vbt.datalayer.envelope) named by the result's codec.
 5. Check with `vbt ds lint`, `vbt ds explain <server>.<tool>` and
    `vbt datasource conformance --plugin <name>` for any new plugin.
"""


def _out(text: str = "") -> None:
    print(text)


def _err(text: str) -> None:
    print(text, file=sys.stderr)


def _catalog(config: dict[str, Any]) -> tuple[Any, Any, Any]:
    from ..preflight import data_catalog
    return data_catalog(config)


def _cache(config: dict[str, Any], catalog: Any, registry: Any) -> Any:
    from .gateway.readiness import ReadinessCache
    from .settings import DataSettings
    cache = ReadinessCache(DataSettings.from_config(config).cache_dir, catalog, registry)
    cache.load()
    return cache


def _split_tool(text: str) -> tuple[str, str]:
    server, sep, tool = str(text).removeprefix("mcp__").replace("__", ".", 1).partition(".")
    if not sep or not server or not tool:
        raise ValueError(f"expected <server>.<tool>, got {text!r}")
    return server, tool


def _unknown_tool(config: dict[str, Any], catalog: Any, server: str, tool: str, *, bound: bool) -> str | None:
    """Why ``server.tool`` names nothing (None when it is known): the server must be in the catalog or the
    configured MCP servers; with ``bound`` the tool must have a binding (a check or an estimate needs one)."""
    configured = {s.get("name") for s in (config.get("mcp_servers") or {}).get("servers", [])}
    if server not in set(catalog.servers()) | configured:
        return f"unknown server {server!r}"
    if bound and tool not in set(catalog.tools(server)):
        return f"{server}.{tool} has no binding in the catalog (unknown tool, or not reviewed)"
    return None


def _unknown_tables(catalog: Any, refs: Iterable[str]) -> list[str]:
    out = []
    for ref in refs:
        try:
            catalog.table(ref)
        except Exception:  # noqa: BLE001
            out.append(str(ref))
    return out


# ---------------------------------------------------------------------------- list / describe / lint


def cmd_list(args: argparse.Namespace, config: dict[str, Any]) -> int:
    _settings, catalog, registry = _catalog(config)
    cache = _cache(config, catalog, registry)
    for source, desc in sorted(catalog.sources.items()):
        release = desc.release.expect or ""
        _out(f"{source}  ({desc.title}; {desc.kind}" + (f"; release {release}" if release else "") + ")")
        for table in sorted(desc.tables):
            ref = f"{source}.{table}"
            t = catalog.table(ref)
            phys, _ = cache.physical(ref)
            m = cache.get(phys)
            status = m.status if m is not None else "unchecked"
            served = f"  served from {t.served_from}" if getattr(t, "served_from", None) else ""
            _out(f"  {table:<44} {t.spec.kind:<9} {status}{served}")
    if not cache.tables:
        _out("(no cached readiness: run `vbt ds check`)")
    return 0


def _role_line(name: str, col: Any) -> str:
    data = col.model_dump(exclude_none=True, exclude_defaults=True, by_alias=True) if hasattr(col, "model_dump") \
        else dict(col)
    role = data.pop("role", getattr(col, "role", "?"))
    extra = ", ".join(f"{k}={v}" for k, v in sorted(data.items()) if k not in ("fields", "description"))
    return f"    {name}: {role}" + (f" ({extra})" if extra else "")[:200]


def cmd_describe(args: argparse.Namespace, config: dict[str, Any]) -> int:
    _settings, catalog, _registry = _catalog(config)
    target = str(args.target)
    source, _, table = target.partition(".")
    try:
        desc = catalog.source(source)
    except Exception as exc:  # noqa: BLE001
        _err(f"error: {exc}")
        return 2
    tables = [table] if table else sorted(desc.tables)
    if not table:
        _out(f"{source}: {desc.title} ({desc.kind}); root {desc.root or '(remote)'}")
        for name, spec in sorted(desc.id_types.items()):
            _out(f"  id_type {name}: plugin {spec.plugin}")
    for name in tables:
        ref = f"{source}.{name}"
        try:
            t = catalog.table(ref)
        except Exception as exc:  # noqa: BLE001
            _err(f"error: {exc}")
            return 2
        spec = t.spec
        _out(f"{ref}  [{spec.kind}]  grain: {spec.grain}")
        _out(f"  key: {', '.join(t.key) or '(none)'}")
        if t.is_item_table:
            _out(f"  items of: {spec.items_of.table} {spec.items_of.path}")
        if spec.coverage is not None:
            _out(f"  coverage: {spec.coverage.model_dump(exclude_none=True, exclude_defaults=True)}")
        if spec.partitions:
            _out(f"  partitions: {', '.join(spec.partitions)}")
        if table:
            for col, cspec in sorted(spec.columns.items()):
                _out(_role_line(col, cspec))
                vocab = getattr(cspec, "values", None) or getattr(cspec, "vocabulary", None)
                if vocab:
                    _out(f"      vocabulary: {list(vocab)[:20]}")
        else:
            roles: dict[str, int] = {}
            for cspec in spec.columns.values():
                roles[str(getattr(cspec, "role", "?"))] = roles.get(str(getattr(cspec, "role", "?")), 0) + 1
            _out("  roles: " + ", ".join(f"{k} {v}" for k, v in sorted(roles.items())))
    return 0


def cmd_lint(args: argparse.Namespace, config: dict[str, Any]) -> int:
    try:
        _settings, catalog, registry = _catalog(config)
    except Exception as exc:  # noqa: BLE001 - a descriptor that does not load is the first lint error
        _err(f"error: {type(exc).__name__}: {exc}")
        return 1
    findings = catalog.lint(registry, strict=True if args.strict else None)
    errors = [f for f in findings if f.level == "error"]
    for f in findings:
        if f.level == "error" or not args.quiet:
            _out(str(f) + (f"  [{f.rule}]" if getattr(f, "rule", "") else ""))
    n_tools = sum(len(catalog.tools(s)) for s in catalog.servers())
    _out(f"{len(catalog.sources)} sources, {n_tools} bound tools: {len(errors)} error(s), "
         f"{len(findings) - len(errors)} warning(s)")
    return 1 if errors else 0


# ---------------------------------------------------------------------------- check


def _index_status(config: dict[str, Any], catalog: Any, response: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolver index status of every id_type with a local universe: ``ready`` when the sidecar for
    the universe table's current fingerprint exists (``vbt ds index build`` writes it)."""
    from .resolve import IndexStore
    from .settings import DataSettings

    store = IndexStore(DataSettings.from_config(config).cache_dir)
    tables = response.get("tables") or {}
    out: dict[str, dict[str, Any]] = {}
    for source, desc in sorted(catalog.sources.items()):
        for name, spec in sorted(desc.id_types.items()):
            if getattr(spec, "index", None) == "remote" or spec.universe is None:
                continue
            first = spec.universe[0] if isinstance(spec.universe, list) else spec.universe
            uref = getattr(first, "table", None) or ".".join(str(first).split(".")[:-1])
            uref = uref if uref.count(".") == 1 else f"{source}.{uref}"
            try:
                phys = str(catalog.table(uref).physical) if catalog.table(uref).is_item_table else uref
            except Exception:  # noqa: BLE001
                phys = uref
            fp = (tables.get(phys) or {}).get("fingerprint")
            q = f"{source}:{name}"
            built = store.fingerprints(source, name)
            if not fp and built:
                out[q] = {"name": name, "id_type": q, "status": "ready", "table": uref, "fingerprint": built[0],
                          "detail": f"the newest sidecar; {uref} has no checked fingerprint to compare"}
            elif not fp:
                out[q] = {"name": name, "id_type": q, "status": "unchecked", "table": uref,
                          "detail": f"no fingerprint for {uref}"}
            elif store.exists(source, fp, name):
                out[q] = {"name": name, "id_type": q, "status": "ready", "table": uref, "fingerprint": fp}
            else:
                out[q] = {"name": name, "id_type": q, "status": "missing", "table": uref, "fingerprint": fp,
                          "detail": f"build it with `vbt ds index build --id-type {q}` (or on first use)"}
    return out


def cmd_check(args: argparse.Namespace, config: dict[str, Any]) -> int:
    from ..preflight import DataCheckUnavailable, data_findings, load_data_readiness, run_data_check
    from .gateway.readiness import call_readiness

    tables = list(args.table or [])
    column = None
    if args.column:
        parts = str(args.column).split(".")
        if len(parts) < 3:
            _err("error: --column takes source.table.column")
            return 2
        tables.append(".".join(parts[:2]))
        column = ".".join(parts[2:])
    _settings, catalog, _registry = _catalog(config)
    bad = _unknown_tables(catalog, tables)
    if bad:
        _err(f"error: unknown table(s): {', '.join(bad)}")
        return 2
    tool = None
    if args.tool:
        try:
            tool = _split_tool(args.tool)
            why = _unknown_tool(config, catalog, *tool, bound=True)
            if why:
                raise ValueError(why)
            contract = catalog.contract(*tool)
        except Exception as exc:  # noqa: BLE001
            _err(f"error: {exc}")
            return 2
        tables.extend(contract.tables)
    try:
        response = run_data_check(config, tables=sorted(set(tables)), depth=args.depth)
    except DataCheckUnavailable as exc:
        _err(f"error: {exc}")
        return 2
    configured = [s.get("name") for s in (config.get("mcp_servers") or {}).get("servers", [])
                  if s.get("enabled", True)]
    dr, cache, catalog = load_data_readiness(config, response=response, servers=configured or catalog.servers())
    indexes = _index_status(config, catalog, response)
    tool_result = None
    if tool is not None:
        contract = catalog.contract(*tool)
        r = call_readiness(contract, cache, bound_table=contract.bound_table)
        tool_result = {"tool": f"mcp__{tool[0]}__{tool[1]}", "ready": r.ready, "reasons": r.reasons,
                       "sections_unavailable": [x["name"] for x in r.soft],
                       "unchecked": r.unchecked, "unavailable_partitions": r.unavailable_partitions}
    if args.json:
        body = {"tables": response.get("tables") or {}, "table_errors": response.get("table_errors") or {},
                "quarantined": response.get("quarantined") or [], "depth": response.get("depth"), "indexes": indexes,
                "tools": {"unready": dr.unready, "partial": dr.partial,
                          "bound": {s: list(t) for s, t in sorted(dr.bound.items())}}}
        if tool_result is not None:
            body["tool"] = tool_result
        _out(json.dumps(body, sort_keys=True, default=str))
        if tool_result is not None:
            return 0 if tool_result["ready"] else 1
        return _check_rc(config, dr, set(tables))
    shown = set(tables)
    for ref, m in sorted(dr.tables.items()):
        if shown and ref not in shown:
            continue
        _out(f"{ref}: {m.status}" + (f"  [{m.fingerprint}]" if m.fingerprint and args.verbose else ""))
        for c in m.checks:
            if c.ok or (column and c.column != column and not str(c.column or "").startswith(column + ".")):
                continue
            where = (f" {c.column}" if c.column else "") + (f" [{c.partition}]" if c.partition else "")
            _out(f"  [{c.level}] {c.name}{where}: {c.detail}")
        if column:
            st = m.columns.get(column) or m.containers.get(column)
            _out(f"  column {column}: {st or 'ready'}")
    for ref, err in sorted(dr.errors.items()):
        if not shown or ref in shown:
            _out(f"{ref}: check failed: {err}")
    for q in response.get("quarantined") or []:
        _out(f"quarantined {q.get('kind')} {q.get('file')}: {q.get('error')} (the tools depending on it are refused)")
    if not shown:
        _out("")
        for r in data_findings(config, dr):
            if r.scope and "granted" in r.scope or not r.scope or "table" not in r.scope:
                _out(r.line())
        _out("")
        _out("Resolver indexes:")
        for q, entry in indexes.items():
            _out(f"  {q}: {entry['status']}" + (f" ({entry['detail']})" if entry.get("detail") and
                                                 entry["status"] != "ready" else ""))
    if tool_result is not None:
        _out(f"{tool_result['tool']}: {'ready' if tool_result['ready'] else 'not ready'}")
        for x in tool_result["reasons"]:
            where = x["name"] + (f".{x['column']}" if x.get("column") else "") + \
                (f" [{x['partition']}]" if x.get("partition") else "")
            _out(f"  {where}: {x['check']}: {x['detail']}")
        for ref in tool_result["unchecked"]:
            _out(f"  {ref}: unchecked")
        return 0 if tool_result["ready"] else 1
    return _check_rc(config, dr, shown)


def _check_rc(config: dict[str, Any], dr: Any, shown: set[str]) -> int:
    """1 when a checked table failed its check or a required readiness result fails, else 0."""
    from ..preflight import data_findings

    if any(not shown or ref in shown for ref in dr.errors):
        return 1
    if shown:
        return 1 if any(dr.tables[r].status not in ("ready", "partial") for r in shown if r in dr.tables) else 0
    return 1 if any(not r.ok and r.required for r in data_findings(config, dr)) else 0


# ---------------------------------------------------------------------------- resolve


def resolver_for(config: dict[str, Any]) -> tuple[Any, Any]:
    """``(Resolver, catalog)`` resolving against the newest built sidecar of each id_type (offline:
    no data child; an id_type without a sidecar resolves with existence ``unknown``)."""
    from .resolve import IndexStore, Resolver

    settings, catalog, registry = _catalog(config)
    store = IndexStore(settings.cache_dir)

    def fingerprint(source: str, id_type: str) -> str | None:
        fps = store.fingerprints(source, id_type)
        return fps[0] if fps else None

    return Resolver(registry, catalog, store.provider(fingerprint), settings=settings), catalog


def _no_local_index(config: dict[str, Any], catalog: Any, qualified: str) -> str | None:
    """Why ``qualified`` (or the key type it labels) resolves offline only to existence ``unknown``."""
    from .resolve import IndexStore
    from .settings import DataSettings

    store = IndexStore(DataSettings.from_config(config).cache_dir)
    try:
        source, spec = catalog.id_type(qualified)
    except Exception:  # noqa: BLE001
        return None
    name = qualified.split(":", 1)[-1]
    target = getattr(spec, "label_of", None) or name
    if getattr(spec, "index", None) == "remote" or getattr(spec, "resolvable", True) is False:
        return None
    if store.fingerprints(source, target):
        return None
    return (f"no local index for {source}:{target}; values resolve by syntax only (existence unknown). "
            f"Build it with `vbt ds index build --id-type {source}:{target}`")


def cmd_resolve(args: argparse.Namespace, config: dict[str, Any]) -> int:
    resolver, catalog = resolver_for(config)
    try:
        source, _spec = catalog.id_type(args.id_type)
    except Exception as exc:  # noqa: BLE001
        _err(f"error: {exc}")
        return 2
    qualified = args.id_type if ":" in args.id_type else f"{source}:{args.id_type}"
    no_index = _no_local_index(config, catalog, qualified)
    if no_index:
        _err(f"note: {no_index}")
    rc = 0
    rows = []
    for value in args.values:
        res = resolver.resolve(value, [qualified], existence=args.existence)
        row = {"value": value, "status": res.status, "canonical": res.canonical, "rule": res.rule,
               "id_type": res.matched_id_type or res.id_type, "existence": res.existence,
               "candidates": [getattr(c, "canonical", None) or str(c) for c in (res.candidates or [])],
               "suggestions": [getattr(c, "canonical", None) or str(c) for c in (getattr(res, "suggestions", None) or [])],
               "tried": list(res.tried or [])}
        rows.append(row)
        if not res.ok:
            rc = 1
    if args.json:
        _out(json.dumps(rows, sort_keys=True, default=str))
        return rc
    for row in rows:
        line = f"{row['value']} -> {row['canonical'] or '-'}  [{row['status']}"
        line += f", rule {row['rule']}" if row["rule"] else ""
        line += f", existence {row['existence']}]" if row["existence"] else "]"
        _out(line)
        if row["candidates"]:
            _out(f"  candidates: {', '.join(map(str, row['candidates'][:10]))}")
        if row["suggestions"]:
            _out(f"  did you mean: {', '.join(map(str, row['suggestions'][:5]))}")
        if args.verbose:
            for t in row["tried"]:
                _out(f"  tried {t}")
    return rc


# ---------------------------------------------------------------------------- explain


def explain_tool(catalog: Any, registry: Any, server: str, tool: str, *, description: str = "",
                 schema: Mapping[str, Any] | None = None, settings: Any = None) -> list[str]:
    """The explain lines of one tool: binding, serve mode, reads, arguments, derived schema and text, defects."""
    from .derive import annotate_schema, describe_tool

    contract = catalog.contract(server, tool)
    b = contract.binding
    lines = [f"{server}.{tool}"]
    if contract.generic or b is None:
        lines.append("  no reviewed binding: the generic guard applies (not_found, empty_unverified)")
        return lines
    if contract.alias_of:
        lines.append(f"  binding shared with {contract.alias_of}")
    lines.append(f"  status: {b.status}; serve: {b.serve}; witness: {b.witness}; on_contradiction: {b.on_contradiction}")
    if b.block is not None:
        lines.append(f"  blocked: {b.block.model_dump(exclude_none=True)}")
    lines.append(f"  bound table: {contract.bound_table or '-'}")
    for ref, rs in sorted(b.reads.items()):
        cond = f" when {rs.when}" if rs.when else ""
        lines.append(f"  reads {ref} ({rs.access}){cond}" + (f": {', '.join(rs.columns)}" if rs.columns else ""))
    for name, a in sorted(contract.args.items()):
        cols = ", ".join(f"{t}.{c}" for t, c in contract.arg_columns(name)) or "-"
        bits = [a.role, f"op {a.op}"]
        if a.accepts:
            bits.append("accepts " + "|".join(a.accepts))
        if a.existence != "universe":
            bits.append(f"existence {a.existence}")
        lines.append(f"  arg {name}: {'; '.join(bits)} -> {cols}")
    if b.derived is not None:
        lines.append(f"  derived: {b.derived.model_dump(exclude_none=True, exclude_defaults=True)}")
    max_chars = getattr(getattr(settings, "derive", None), "description_max_chars", 1200)
    enum_max = getattr(getattr(settings, "derive", None), "enum_max", 64)
    if not schema:
        # An empty placeholder would read as "takes no arguments"; --json shows the input schema.
        lines.append("  derived schema: (upstream schema not available offline)")
    else:
        try:
            derived = annotate_schema(contract, dict(schema), catalog=catalog, registry=registry, enum_max=enum_max)
            lines.append("  derived schema: " + json.dumps(derived, sort_keys=True, default=str)[:1500])
        except Exception as exc:  # noqa: BLE001
            lines.append(f"  derived schema: unavailable ({type(exc).__name__}: {exc})")
    try:
        text = describe_tool(contract, description, catalog=catalog, max_chars=max_chars)
        lines.append("  derived text:")
        lines.extend(f"    {ln}" for ln in text.splitlines())
    except Exception as exc:  # noqa: BLE001
        lines.append(f"  derived text: unavailable ({type(exc).__name__}: {exc})")
    if contract.full_table_reads:
        lines.append(f"  full-table upstream loads: {', '.join(contract.full_table_reads)} "
                     f"(`vbt ds estimate --tool {server}.{tool}`)")
    for d in b.defects:
        lines.append(f"  defect {d.id}: {d.what}" + (f" ({d.where})" if d.where else "")
                     + (f"; detector {d.test}" if d.test else ""))
    return lines


#: Upstream parameter annotations as JSON schema types (the subset FastMCP derives for these signatures).
_SIMPLE_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "dict": "object",
                 "Dict": "object", "Any": None}


def _annotation_schema(node: Any) -> dict[str, Any]:
    import ast

    if node is None:
        return {}
    text = ast.unparse(node).replace("typing.", "")
    if text.startswith("Optional[") and text.endswith("]"):
        return _annotation_schema(ast.parse(text[len("Optional["):-1], mode="eval").body)
    if " | None" in text:
        return _annotation_schema(ast.parse(text.replace(" | None", ""), mode="eval").body)
    base = text.split("[")[0]
    if base in _SIMPLE_TYPES:
        return {"type": _SIMPLE_TYPES[base]} if _SIMPLE_TYPES[base] else {}
    if base in ("list", "List", "Sequence", "tuple", "Tuple"):
        inner = text[len(base) + 1:-1] if "[" in text else ""
        items = _annotation_schema(ast.parse(inner, mode="eval").body) if inner else {}
        return {"type": "array", "items": items} if items else {"type": "array"}
    return {}


def _function_schema(fn: Any) -> dict[str, Any]:
    import ast

    args = fn.args.args
    defaults = [None] * (len(args) - len(fn.args.defaults)) + list(fn.args.defaults)
    props: dict[str, Any] = {}
    required = []
    for a, d in zip(args, defaults):
        if a.arg in ("self", "ctx"):
            continue
        prop = _annotation_schema(a.annotation)
        if d is None:
            required.append(a.arg)
        else:
            try:
                prop["default"] = ast.literal_eval(d)
            except ValueError:
                pass
        props[a.arg] = prop
    out: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        out["required"] = required
    return out


def upstream_tool_schemas(upstream: str | Path | None, project_root: str | Path | None = None
                          ) -> dict[str, dict[str, Any]]:
    """``{"server.tool": input schema}`` of every bridged tool, parsed from the upstream ``server.py``
    ``register_tool`` calls and the harness PubMed server (AST only: nothing is imported or run)."""
    import ast

    out: dict[str, dict[str, Any]] = {}
    root = Path(upstream) if upstream else None
    servers = root / "src" / "mcp_servers" if root else None
    if servers is not None and servers.is_dir():
        for server_dir in sorted(p for p in servers.iterdir() if (p / "server.py").is_file()):
            if server_dir.name == "provenance_mcp":
                continue
            tree = ast.parse((server_dir / "server.py").read_text(encoding="utf-8"))
            imports: dict[str, tuple[str, str]] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src.mcp_servers"):
                    for alias in node.names:
                        imports[alias.asname or alias.name] = (node.module, alias.name)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "register_tool":
                    ref = getattr(node.args[1], "id", None) if len(node.args) > 1 else None
                    if ref not in imports:
                        continue
                    module, name = imports[ref]
                    path = root / (module.replace(".", "/") + ".py")  # type: ignore[operator]
                    fn = next((n for n in ast.parse(path.read_text(encoding="utf-8")).body
                               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name), None)
                    if fn is not None:
                        out[f"{server_dir.name.removesuffix('_mcp')}.{name}"] = _function_schema(fn)
    pubmed = Path(project_root or Path(__file__).resolve().parents[3]) / "src" / "vbt" / "mcp_servers" / "pubmed_server.py"
    if pubmed.is_file():
        for fn in ast.parse(pubmed.read_text(encoding="utf-8")).body:
            if isinstance(fn, ast.FunctionDef) and any(
                    isinstance(d, ast.Call) and getattr(d.func, "attr", None) == "tool" for d in fn.decorator_list):
                out[f"pubmed.{fn.name}"] = _function_schema(fn)
    return dict(sorted(out.items()))


def schema_sha256(schema: Mapping[str, Any]) -> str:
    import hashlib

    text = json.dumps(schema, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _upstream_root(config: Mapping[str, Any]) -> str | None:
    from .catalog import variables_from_config

    return variables_from_config(dict(config)).get("upstream")


def explain_json(config: Mapping[str, Any], catalog: Any, *, upstream: str | Path | None = None,
                 project_root: str | Path | None = None, schemas_only: bool = False) -> dict[str, Any]:
    """``vbt ds explain --all --json``: per tool its binding (status, serve mode, bound table, bound
    arguments, upstream parameters no argument binds) and the upstream input schema with its hash.
    ``schemas_only`` keeps the schemas (the CI drift snapshot ``upstream_tool_schemas.json``)."""
    from ..pinning import git_info

    root = upstream if upstream is not None else _upstream_root(config)
    schemas = upstream_tool_schemas(root, project_root)
    commit = None
    if root and Path(root).is_dir():
        commit = git_info(str(root)).get("upstream_commit")
    tools: dict[str, Any] = {}
    names = {f"{s}.{t}" for s in catalog.servers() for t in catalog.tools(s)} | set(schemas)
    for name in sorted(names):
        server, _, tool = name.partition(".")
        schema = schemas.get(name)
        entry: dict[str, Any] = {}
        if schema is not None:
            entry["input_schema"] = schema
            entry["input_sha256"] = schema_sha256(schema)
        if schemas_only:
            if schema is not None:
                tools[name] = entry
            continue
        try:
            contract = catalog.contract(server, tool)
        except Exception:  # noqa: BLE001 - an unknown server: no binding
            contract = None
        b = getattr(contract, "binding", None)
        bound = b is not None and not getattr(contract, "generic", False)
        entry["bound"] = bound
        if bound:
            entry.update({"status": b.status, "serve": b.serve, "bound_table": contract.bound_table,
                          "args": sorted(contract.args)})
            if schema is not None:
                entry["unbound_params"] = sorted(set(schema.get("properties") or {}) - set(contract.args))
        tools[name] = entry
    head = {"schema": "vbt.upstream_tool_schemas/1" if schemas_only else "vbt.explain/1",
            "upstream_commit": commit}
    return {**head, "tools": tools}


def cmd_explain(args: argparse.Namespace, config: dict[str, Any]) -> int:
    settings, catalog, registry = _catalog(config)
    if getattr(args, "json", False):
        if not args.all and not args.target:
            _err("error: name a tool (<server>.<tool>) or pass --all")
            return 2
        body = explain_json(config, catalog, project_root=getattr(settings, "project_root", None),
                            schemas_only=bool(getattr(args, "schemas", False)))
        if args.target and not args.all:
            try:
                server, tool = _split_tool(args.target)
            except ValueError as exc:
                _err(f"error: {exc}")
                return 2
            body["tools"] = {k: v for k, v in body["tools"].items() if k == f"{server}.{tool}"}
        _out(json.dumps(body, indent=1, sort_keys=True, default=str))
        return 0
    if args.all:
        targets = [(s, t) for s in catalog.servers() for t in catalog.tools(s)]
    elif args.target:
        try:
            targets = [_split_tool(args.target)]
            why = _unknown_tool(config, catalog, *targets[0], bound=False)
            if why:
                raise ValueError(why)
        except ValueError as exc:
            _err(f"error: {exc}")
            return 2
    else:
        _err("error: name a tool (<server>.<tool>) or pass --all")
        return 2
    try:  # the upstream input schemas, parsed offline (AST); none when the checkout is absent
        schemas = upstream_tool_schemas(_upstream_root(config), getattr(settings, "project_root", None))
    except Exception:  # noqa: BLE001
        schemas = {}
    rc = 0
    for server, tool in targets:
        try:
            for line in explain_tool(catalog, registry, server, tool, settings=settings,
                                     schema=schemas.get(f"{server}.{tool}")):
                _out(line)
        except Exception as exc:  # noqa: BLE001
            _err(f"{server}.{tool}: error: {type(exc).__name__}: {exc}")
            rc = 1
    return rc


# ---------------------------------------------------------------------------- fingerprint / index / estimate


def cmd_fingerprint(args: argparse.Namespace, config: dict[str, Any]) -> int:
    from ..preflight import DataCheckUnavailable, run_data_check
    from .settings import DataSettings

    try:
        response = run_data_check(config, tables=list(args.table or []), depth="shallow")
    except DataCheckUnavailable as exc:
        _err(f"error: {exc}")
        return 2
    fps = {ref: m.get("fingerprint") for ref, m in sorted((response.get("tables") or {}).items())}
    for ref, fp in fps.items():
        _out(f"{ref}  {fp or '-'}")
    if args.write:
        path = Path(DataSettings.from_config(config).cache_dir) / "fingerprints.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"tables": fps}, indent=1, sort_keys=True) + "\n")
        _out(f"written: {path}")
    return 0


def _run_reaped(config: Mapping[str, Any], argv: list[str], env: Mapping[str, str], *,
                timeout: float) -> subprocess.CompletedProcess[str]:
    """Run a data-child command line under the reaper with the data child's memory limit (see
    :func:`vbt.preflight.run_contained`)."""
    from ..preflight import run_contained

    return run_contained(config, argv, env, timeout=timeout)


def _run_child(config: dict[str, Any], *child_args: str, timeout: float = 3600.0) -> tuple[int, str, str]:
    from ..preflight import data_child_command

    cmd, env = data_child_command(config, *child_args)
    try:
        proc = _run_reaped(config, cmd, env, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 2, "", f"could not run {cmd[0]}: {exc}"
    return proc.returncode, proc.stdout, proc.stderr


def _call_child(config: dict[str, Any], target: str, payload: Mapping[str, Any], *,
                timeout: float = 3600.0) -> dict[str, Any]:
    """Run ``module:function(ctx, payload)`` in the data child's interpreter; returns its JSON."""
    from ..preflight import data_child_command
    from .settings import DataSettings

    settings = DataSettings.from_config(config)
    cmd, env = data_child_command(config)
    src = str(Path(settings.project_root) / "src")
    argv = [cmd[0], "-E", "-c", _CALL_SNIPPET, src, target, json.dumps(dict(payload))]
    proc = _run_reaped(config, argv, env, timeout=timeout)
    line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("{")), None)
    if proc.returncode != 0 or line is None:
        raise RuntimeError((proc.stderr or proc.stdout).strip()[-800:] or f"exit {proc.returncode}")
    return json.loads(line)


def cmd_index_build_huge(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """``index build --access-paths --huge``: resumable sidecars of huge tables, called until complete
    (each call within ``--time-budget-s`` and ``--max-bytes``) or ``--once``."""
    _settings, catalog, _registry = _catalog(config)
    tables = list(args.table or [])
    if not tables:
        tables = [f"{s}.{n}" for s, d in sorted(catalog.sources.items()) for n, t in sorted(d.tables.items())
                  if t.size_class == "huge" and any(ap.via == "sidecar_index" for ap in t.access_paths)]
    bad = _unknown_tables(catalog, tables)
    if bad:
        _err(f"error: unknown table(s): {', '.join(bad)}")
        return 2
    rc = 0
    for ref in tables:
        payload = {"table": ref, "time_budget_s": args.time_budget_s, "budget_bytes": args.max_bytes,
                   "force": bool(args.force)}
        while True:
            try:
                body = _call_child(config, "vbt.datalayer.service.verbs.index_huge:build_index_huge", payload)
            except Exception as exc:  # noqa: BLE001
                _err(f"failed {ref}: {exc}")
                rc = 1
                break
            payload["force"] = False
            if args.json:
                _out(json.dumps(body, sort_keys=True))
            else:
                _out(f"{ref}: {body.get('status')} {body.get('units_done')}/{body.get('units_total')} row groups"
                     + (f", {body['rows']} values -> {body.get('path')}" if body.get("rows") is not None else "")
                     + (f" ({body['reason']})" if body.get("reason") else ""))
            if body.get("status") == "complete" or args.once:
                break
    return rc


def cmd_index_build(args: argparse.Namespace, config: dict[str, Any]) -> int:
    from ..preflight import DataCheckUnavailable

    if getattr(args, "huge", False):
        return cmd_index_build_huge(args, config)
    child: list[str] = ["--build-index"]
    for t in args.id_type or []:
        child += ["--id-type", t]
    if args.access_paths:
        child.append("--access-paths")
        for t in args.table or []:
            child += ["--table", t]
    if not args.id_type and not args.access_paths:
        _settings, catalog, _registry = _catalog(config)
        for source, desc in sorted(catalog.sources.items()):
            for name, spec in sorted(desc.id_types.items()):
                if spec.universe is not None and getattr(spec, "index", None) != "remote":
                    child += ["--id-type", f"{source}:{name}"]
    if args.force:
        child.append("--force")
    try:
        rc, stdout, stderr = _run_child(config, *child)
    except DataCheckUnavailable as exc:
        _err(f"error: {exc}")
        return 2
    line = next((ln for ln in reversed(stdout.splitlines()) if ln.startswith("{")), None)
    if line is None:
        _err(f"error: the data child failed: {(stderr or stdout).strip()[-800:]}")
        return rc or 2
    body = json.loads(line)
    if args.json:
        _out(line)
        return rc
    for b in body.get("built") or []:
        what = b.get("id_type") or f"{b.get('table')} {','.join(b.get('access_path') or [])}"
        _out(f"built {what}: {b.get('rows')} rows -> {b.get('path')}")
    for k, err in sorted((body.get("errors") or {}).items()):
        _out(f"failed {k}: {err}")
    return rc


def _table_stats(config: dict[str, Any], tables: list[str]) -> dict[str, Any]:
    from ..preflight import data_child_command
    from .settings import DataSettings

    settings = DataSettings.from_config(config)
    cmd, env = data_child_command(config)
    src = str(Path(settings.project_root) / "src")
    argv = [cmd[0], "-E", "-c", _STATS_SNIPPET, src, *tables]
    proc = _run_reaped(config, argv, env, timeout=3600)
    line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("{")), None)
    if proc.returncode != 0 or line is None:
        raise RuntimeError((proc.stderr or proc.stdout).strip()[-800:] or f"exit {proc.returncode}")
    return json.loads(line)


def cmd_estimate(args: argparse.Namespace, config: dict[str, Any]) -> int:
    from .ipc import TableStatsModel
    from .launch import host_memory_mb, server_limit_mb
    from .memory import MemoryEstimator

    settings, catalog, _registry = _catalog(config)
    tables: list[str] = []
    full: set[str] = set()
    server = None
    if args.tool:
        try:
            server, tool = _split_tool(args.tool)
            why = _unknown_tool(config, catalog, server, tool, bound=True)
            if why:
                raise ValueError(why)
            contract = catalog.contract(server, tool)
        except Exception as exc:  # noqa: BLE001
            _err(f"error: {exc}")
            return 2
        tables = sorted({catalog.table(r).physical if catalog.table(r).is_item_table else r for r in contract.tables})
        full = set(contract.full_table_reads)
    elif args.table:
        tables = list(args.table)
        bad = _unknown_tables(catalog, tables)
        if bad:
            _err(f"error: unknown table(s): {', '.join(bad)}")
            return 2
        full = set(tables)
    else:
        _err("error: pass --table S.T or --tool server.tool")
        return 2
    try:
        stats = _table_stats(config, [str(t) for t in tables])
    except Exception as exc:  # noqa: BLE001
        _err(f"error: the data child could not compute table statistics: {exc}")
        return 2
    est = MemoryEstimator.from_settings(settings)
    host = host_memory_mb()
    limit = None
    if server is not None:
        spec = next((s for s in (config.get("mcp_servers") or {}).get("servers", []) if s.get("name") == server), {})
        try:
            from ..tools.mcp_bridge import MCPServerConfig
            cfg = MCPServerConfig(**{k: v for k, v in spec.items() if k in MCPServerConfig.__dataclass_fields__})
            limit = server_limit_mb(cfg, settings)
        except Exception:  # noqa: BLE001
            limit = None
    total_upstream = 0
    for ref, ts in sorted((stats.get("tables") or {}).items()):
        model = TableStatsModel.model_validate(ts)
        up = est.peak_upstream(model)
        scan = est.peak_arrow_scan(model)
        if ref in full or str(ref) in {str(f) for f in full}:
            total_upstream += up
        _out(f"{ref}: {model.rows if model.rows is not None else '?'} rows, {model.fragments} fragments, "
             f"{model.bytes_on_disk / 1e6:.1f} MB on disk; upstream full load ~{up / 1e6:.0f} MB, "
             f"projected scan ~{scan / 1e6:.0f} MB")
    errors = dict(stats.get("table_errors") or stats.get("errors") or {})
    for ref, err in sorted(errors.items()):
        _out(f"{ref}: error: {err}")
    missing = sorted(str(r) for r in errors if str(r) in {str(f) for f in full} or server is None)
    if server is not None:
        if missing:
            # a table the tool loads whole could not be measured: no size, so no admission verdict
            _out(f"tool {args.tool}: not estimable ({', '.join(missing)} unavailable)")
            return 1
        _out(f"tool {args.tool}: upstream peak ~{total_upstream / 1e6:.0f} MB"
             + (f"; server limit {limit} MB" if limit else "")
             + (f"; host {host} MB" if host else "")
             + ("; admissible" if not limit or total_upstream / 1e6 <= limit else "; NOT admissible (too_large)"))
    elif host:
        _out(f"host memory: {host} MB")
    return 1 if missing else 0


# ---------------------------------------------------------------------------- status / calibrate


def _status_dir(args: argparse.Namespace, config: dict[str, Any]) -> Path | None:
    if getattr(args, "log_dir", None):
        return Path(args.log_dir)
    if getattr(args, "run", None):
        from .retro_audit import resolve_run

        return resolve_run(args.run, config) / "logs" / "mcp"
    return None


def memory_status(config: dict[str, Any], status_dir: Path | None) -> dict[str, Any]:
    """Servers' reaper status files, the host budget and the stored calibrations (``vbt ds status``)."""
    from .memory.calibrate import factors_summary, load_calibrations, load_feedback
    from .memory.host import host_budget_mb, host_total_mb
    from .memory.ledger import read_status
    from .settings import DataSettings

    settings = DataSettings.from_config(config)
    servers: dict[str, Any] = {}
    if status_dir is not None and status_dir.is_dir():
        for path in sorted(status_dir.glob("*.status.json")):
            data = read_status(path)
            if data:
                servers[path.name[:-len(".status.json")]] = data
    cals = load_calibrations(settings.cache_dir)
    feedback = load_feedback(settings.cache_dir)
    resident = sum(float(d.get("rss_mb") or 0.0) for d in servers.values() if not d.get("exit"))
    return {"host": {"total_mb": host_total_mb(), "budget_mb": host_budget_mb(settings),
                     "resident_mb": round(resident, 1), "limit_kind": settings.memory.limit_kind},
            "servers": servers, "status_dir": str(status_dir) if status_dir else None,
            "calibrations": {fp: {"table": c.get("table"), "rows_sampled": c.get("rows_sampled"),
                                  "bytes_per_row": c.get("bytes_per_row"), "factors": c.get("factors"),
                                  "at": c.get("at")} for fp, c in sorted(cals.items())},
            "factors": factors_summary(cals.values()),
            "feedback": {k: {"table": v.get("table"), "factor": v.get("factor"),
                             "observations": len(v.get("observations") or [])} for k, v in sorted(feedback.items())}}


def cmd_status(args: argparse.Namespace, config: dict[str, Any]) -> int:
    try:
        status_dir = _status_dir(args, config)
    except FileNotFoundError as exc:
        _err(f"error: {exc}")
        return 2
    body = memory_status(config, status_dir)
    if args.json:
        _out(json.dumps(body, sort_keys=True, default=str))
        return 0
    host = body["host"]
    budget = host["budget_mb"]
    total = f"{host['total_mb']:,.0f}" if host["total_mb"] else "?"
    _out(f"host: {total} MB; budget "
         + (f"{budget:,.0f} MB" if budget else "off") + f"; resident {host['resident_mb']:,.0f} MB; "
         f"containment {host['limit_kind']}")
    if status_dir is None:
        _out("servers: pass a run (or --log-dir) to read the reaper status files")
    elif not body["servers"]:
        _out(f"servers: no status files under {status_dir}")
    for name, d in sorted(body["servers"].items()):
        state = "exited " + json.dumps(d["exit"], sort_keys=True) if d.get("exit") else f"rss {d.get('rss_mb')} MB"
        _out(f"  {name:<20} {state}; peak {d.get('peak_rss_mb')} MB; limit {d.get('limit_mb')} MB "
             f"({d.get('containment')}); hash seed {d.get('hash_seed')}")
    for fp, c in body["calibrations"].items():
        _out(f"calibration {c['table']} [{fp}]: {c['rows_sampled']} rows sampled, "
             f"{(c['bytes_per_row'] or 0):,.0f} B/row, factors {c['factors']}")
    if not body["calibrations"]:
        _out("calibrations: none (estimates use the seed factors; run `vbt ds calibrate --table S.T`)")
    for key, f in body["feedback"].items():
        _out(f"measured {f['table']} [{key}]: x{f['factor']} over {f['observations']} load(s)")
    return 0


def cmd_calibrate(args: argparse.Namespace, config: dict[str, Any]) -> int:
    _settings, catalog, _registry = _catalog(config)
    tables = list(args.table or [])
    if not tables:
        _err("error: pass --table S.T (repeatable)")
        return 2
    bad = _unknown_tables(catalog, tables)
    if bad:
        _err(f"error: unknown table(s): {', '.join(bad)}")
        return 2
    rc = 0
    for ref in tables:
        try:
            cal = _call_child(config, "vbt.datalayer.memory.calibrate:calibrate",
                              {"table": ref, "row_groups": int(args.row_groups)})
        except Exception as exc:  # noqa: BLE001
            _err(f"failed {ref}: {exc}")
            rc = 1
            continue
        if args.json:
            _out(json.dumps(cal, sort_keys=True, default=str))
            continue
        _out(f"{ref}: {cal.get('rows_sampled')} of {cal.get('rows')} rows sampled; "
             f"{(cal.get('bytes_per_row') or 0):,.0f} pandas bytes per row; factors vs seed {cal.get('factors')}")
        _out(f"  written: {cal.get('path')}")
    return rc


# ---------------------------------------------------------------------------- overlay init

_LIMIT_NAMES = frozenset({"limit", "max_results", "top_k", "top_n", "size", "page_size", "max_cells", "n",
                          "max_hits", "num_results", "count"})
_TEXT_NAMES = frozenset({"query", "q", "search", "term", "text", "keywords", "filter", "value_filter"})
_EXAMPLE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_:.\-]*[0-9][A-Za-z0-9_:.\-]*")


def _guess_kinds(registry: Any, texts: Iterable[str], *, top: int = 3) -> list[tuple[str, str]]:
    """``[(id_type, example)]``: identifier plugins whose ``looks_like`` recognises a token of ``texts``."""
    tokens: list[str] = []
    for text in texts:
        for tok in _EXAMPLE_RE.findall(str(text or "")):
            tok = tok.strip(".,;:()[]'\"")
            if len(tok) >= 3 and tok not in tokens:
                tokens.append(tok)
    scores: dict[str, tuple[float, str]] = {}
    for plugin in (registry.all("identifier") if registry is not None else []):
        for tok in tokens:
            try:
                score = float(plugin.looks_like(tok))
            except Exception:  # noqa: BLE001 - a plugin that cannot score a token does not guess
                continue
            if score >= 0.9 and score > scores.get(plugin.id_type, (0.0, ""))[0]:
                scores[plugin.id_type] = (score, tok)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1][0], kv[0]))
    return [(kind, ex) for kind, (_s, ex) in ranked[:top]]


def _arg_doc(doc: str, name: str) -> list[str]:
    return [ln.strip() for ln in (doc or "").splitlines() if name in ln]


def _yaml_text(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def scaffold_overlay(server: str, tools: Iterable[Mapping[str, Any]], registry: Any = None, *,
                     url: str | None = None) -> str:
    """The YAML text of an overlay scaffold: every tool ``status: unreviewed``, arguments ``role: unbound``
    (or a guessed limit, free-text or output-path role), identifier kinds guessed as comments."""
    lines = [f"# Overlay scaffold for the MCP server {server!r} (generated by `vbt ds overlay init`).", "#"]
    lines += ["# " + ln if ln else "#" for ln in THIRD_PARTY_HOWTO.rstrip().splitlines()]
    lines += ["schema: vbt.overlay/1", f"server: {_yaml_text(server)}"]
    if url:
        lines.append(f"match: {{url_prefix: {_yaml_text(url)}}}")
    entries = sorted((dict(t) for t in tools), key=lambda t: str(t.get("name")))
    if not entries:
        lines.append("tools: {}")
        return "\n".join(lines) + "\n"
    lines.append("tools:")
    for t in entries:
        name = str(t.get("name"))
        doc = str(t.get("description") or "")
        schema = t.get("inputSchema") or t.get("input_schema") or {}
        props = dict(schema.get("properties") or {})
        first = next((ln.strip() for ln in doc.splitlines() if ln.strip()), "")
        lines.append(f"  {_yaml_text(name)}:")
        if first:
            lines.append(f"    # {first[:150]}")
        lines.append("    status: unreviewed")
        if not props:
            lines.append("    args: {}")
        else:
            lines.append("    args:")
            for arg, prop in sorted(props.items()):
                prop = dict(prop or {})
                typ = prop.get("type") or ("/".join(str(x.get("type")) for x in prop.get("anyOf", [])
                                                    if isinstance(x, Mapping)) or "any")
                lower = arg.lower()
                if lower in _LIMIT_NAMES and "integer" in str(typ):
                    role = "{role: limit}"
                elif lower in _TEXT_NAMES:
                    role = "{role: free_text, interpreted_as: engine}"
                elif "output" in lower and ("path" in lower or "file" in lower):
                    role = "{role: output_path}"
                else:
                    role = "{role: unbound}"
                texts = [str(prop.get("description") or ""), *map(str, prop.get("examples") or []),
                         *([prop["default"]] if isinstance(prop.get("default"), str) else []),
                         *_arg_doc(doc, arg)]
                guesses = _guess_kinds(registry, texts) if role == "{role: unbound}" else []
                note = f"  # {typ}"
                if guesses:
                    note += "; guess: " + ", ".join(f"{k} (from {ex!r})" for k, ex in guesses)
                lines.append(f"      {_yaml_text(arg)}: {role}{note}")
        lines.append("    result: {}")
    return "\n".join(lines) + "\n"


def _list_tools_live(config: dict[str, Any], server: str, timeout: float = 120.0) -> tuple[list[dict[str, Any]],
                                                                                           str | None]:
    """``(tools, url)`` of a configured MCP server, listed through the bridge without a gateway."""
    import asyncio

    from ..tools.mcp_bridge import MCPBridge, MCPServerConfig

    spec = next((s for s in (config.get("mcp_servers") or {}).get("servers", []) if s.get("name") == server), None)
    if spec is None:
        raise ValueError(f"server {server!r} is not in configs/mcp_servers.yaml (or pass --from-json)")
    cfg = MCPServerConfig(**{k: v for k, v in spec.items() if k in MCPServerConfig.__dataclass_fields__})

    async def run() -> list[dict[str, Any]]:
        bridge = MCPBridge([cfg], options=config.get("mcp") or {})
        try:
            tools = await bridge.start(connect_timeout=timeout)
        finally:
            await bridge.aclose()
        prefix = f"mcp__{server}__"
        return [{"name": t.name[len(prefix):] if t.name.startswith(prefix) else t.name,
                 "description": t.description, "inputSchema": t.input_schema} for t in tools]

    return asyncio.run(run()), cfg.url


def lint_scaffold(text: str, config: dict[str, Any] | None = None) -> list[Any]:
    """Validate a scaffold's YAML as an overlay and lint it against the loaded catalog."""
    import yaml

    from .descriptor.lint import lint_overlay
    from .descriptor.overlay import Overlay

    overlay = Overlay.model_validate(yaml.safe_load(text))
    sources: Any = {}
    registry = None
    if config is not None:
        try:
            _settings, catalog, registry = _catalog(config)
            sources = catalog
        except Exception:  # noqa: BLE001 - no catalog: lint against nothing
            sources = {}
    return lint_overlay(overlay, sources, registry)


def cmd_overlay_init(args: argparse.Namespace, config: dict[str, Any]) -> int:
    from .settings import DataSettings

    server = str(args.server)
    try:
        if args.from_json:
            data = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
            tools = data.get("tools", data) if isinstance(data, Mapping) else data
            url = data.get("url") if isinstance(data, Mapping) else None
        else:
            tools, url = _list_tools_live(config, server)
    except Exception as exc:  # noqa: BLE001
        _err(f"error: could not list the tools of {server!r}: {exc}")
        return 2
    try:
        _settings, _catalog_obj, registry = _catalog(config)
    except Exception:  # noqa: BLE001 - guesses need identifier plugins only
        from .plugins.registry import discover
        registry = discover(entry_points=False)
    text = scaffold_overlay(server, tools, registry, url=url)
    out = Path(args.out) if args.out else Path(DataSettings.from_config(config).overlays_dir) / f"{server}.yaml"
    if args.out == "-":
        _out(text)
    else:
        if out.exists() and not args.force:
            _err(f"error: {out} exists (pass --force to overwrite)")
            return 2
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        _out(f"written: {out} ({len(list(tools))} tools, all unreviewed)")
    findings = lint_scaffold(text, config)
    errors = [f for f in findings if f.level == "error"]
    for f in findings:
        _out(str(f))
    return 1 if errors else 0


# ---------------------------------------------------------------------------- retro-audit


def cmd_retro_audit(args: argparse.Namespace, config: dict[str, Any]) -> int:
    from .retro_audit import format_report, retro_audit, resolve_run

    try:
        run_dir = resolve_run(args.run, config)
    except FileNotFoundError as exc:
        _err(f"error: {exc}")
        return 2
    report = retro_audit(run_dir, config)
    if args.json:
        _out(json.dumps(report, sort_keys=True, default=str))
    else:
        for line in format_report(report):
            _out(line)
    return 0


# ---------------------------------------------------------------------------- phase 5: replay


def cmd_replay(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Re-execute recorded data calls on current data and compare (``replay.py``). Exit 0 when every call
    matches or only live records changed (``source_updated``), 1 on a mismatch, 2 when a call cannot run."""
    import asyncio

    from .replay import data_calls, format_result, replay_run
    from .retro_audit import resolve_run

    try:
        run_dir = resolve_run(args.run, config)
    except FileNotFoundError as exc:
        _err(f"error: {exc}")
        return 2
    ids = data_calls(run_dir) if args.all else list(args.tool_use_id or [])
    if not ids:
        _err("error: name tool_use ids or pass --all" if not args.all else f"{run_dir} has no data provenance records")
        return 2
    results = asyncio.run(replay_run(run_dir, ids, config, backend=args.backend))
    if args.json:
        _out(json.dumps([r.to_dict() for r in results], sort_keys=True, default=str))
    else:
        for r in results:
            for line in format_result(r):
                _out(line)
    if any(r.status == "replay_mismatch" for r in results):
        return 1
    return 2 if any(r.status == "unavailable" for r in results) else 0


def cmd_diff_release(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Compare two releases of a source in the data child (``diff_release.py``); exit 1 when a change
    makes some tool not_ready or changes its answers."""
    from .diff_release import format_report

    payload = {"from": args.from_, "to": args.to, "sources": list(args.source or []),
               "tables": list(args.table or []), "vocab": not args.quick, "axes": not args.quick}
    try:
        body = _call_child(config, "vbt.datalayer.diff_release:diff_release_child", payload)
    except Exception as exc:  # noqa: BLE001
        _err(f"error: the data child failed: {exc}")
        return 2
    if body.get("error"):
        _err(f"error: {body['error']}")
        return 2
    if args.json:
        _out(json.dumps(body, sort_keys=True, default=str))
    else:
        for line in format_report(body):
            _out(line)
    return 1 if (body.get("summary") or {}).get("breaking_tables") else 0


# ---------------------------------------------------------------------------- phase 5: graduation

#: The checklist items of :func:`graduation_checklist`, in order.
GRADUATION_ITEMS = ("overlay", "lint", "reviewed", "blocked_alternatives", "observe_evidence")


def graduation_checklist(config: Mapping[str, Any], servers: Iterable[str] | None = None,
                         runs: Iterable[str | Path] = (), *, catalog: Any = None, registry: Any = None
                         ) -> dict[str, dict[str, Any]]:
    """The observe -> enforce checklist (§22) per server: an overlay exists and lints without errors,
    every tool is reviewed, every blocked tool names an alternative, and recorded runs give retro-audit
    evidence (calls observed, calls the gateway would now refuse or qualify, and how many of those a
    claim cited). ``graduated`` is True only when every item passed; ``observe_evidence`` is None (not
    passed) without recorded runs."""
    from .descriptor.lint import lint_overlay
    from .retro_audit import retro_audit

    if catalog is None or registry is None:
        _settings, catalog, registry = _catalog(dict(config))
    wanted = list(servers or catalog.servers())
    audits = []
    for run in runs:
        try:
            audits.append(retro_audit(Path(run), dict(config), catalog=catalog,
                                      resolver=_resolver_or_none(config, catalog)))
        except Exception as exc:  # noqa: BLE001 - one unreadable run does not hide the others
            audits.append({"run": str(run), "calls": [], "error": str(exc)})
    out: dict[str, dict[str, Any]] = {}
    for server in wanted:
        items: dict[str, dict[str, Any]] = {}
        ov = catalog.overlays.get(server)
        items["overlay"] = {"ok": ov is not None, "detail": "configs/data/overlays/" + server + ".yaml"
                            if ov is not None else "no overlay: the generic guard applies"}
        if ov is None:
            out[server] = {"graduated": False, "items": items}
            continue
        errors = [str(f) for f in lint_overlay(ov, catalog, registry) if f.level == "error"]
        items["lint"] = {"ok": not errors, "detail": f"{len(errors)} error(s)", "errors": errors[:10]}
        unreviewed = sorted(t for t, b in ov.tools.items() if b.status != "reviewed")
        items["reviewed"] = {"ok": not unreviewed, "detail": f"{len(ov.tools) - len(unreviewed)}/{len(ov.tools)} "
                             "tools reviewed", "unreviewed": unreviewed}
        bad_blocks = sorted(t for t, b in ov.tools.items() if b.serve == "block" and b.block is not None
                            and not (b.block.alternatives or b.hidden or b.block.hidden))
        items["blocked_alternatives"] = {"ok": not bad_blocks, "detail": "every blocked tool names an alternative"
                                         if not bad_blocks else f"no alternative: {', '.join(bad_blocks)}"}
        calls = [c for a in audits for c in a.get("calls") or [] if str(c.get("tool", "")).startswith(
            f"mcp__{server}__")]
        changed = [c for c in calls if c.get("changed")]
        cited = [c for c in changed if c.get("cited_by")]
        if not audits:
            items["observe_evidence"] = {"ok": None, "detail": "no recorded runs given (vbt ds graduate --run R)"}
        else:
            items["observe_evidence"] = {
                "ok": bool(calls), "calls": len(calls), "changed": len(changed), "cited_changed": len(cited),
                "detail": f"{len(calls)} recorded call(s); {len(changed)} would now be refused or qualified, "
                          f"{len(cited)} of them cited" if calls else "no recorded call of this server"}
        out[server] = {"graduated": all(i.get("ok") is True for i in items.values()), "items": items}
    return out


def _resolver_or_none(config: Mapping[str, Any], catalog: Any) -> Any:
    try:
        resolver, _catalog_ = resolver_for(dict(config))
        return resolver
    except Exception:  # noqa: BLE001 - retro-audit then keeps existence unknown
        return None


def cmd_graduate(args: argparse.Namespace, config: dict[str, Any]) -> int:
    from .retro_audit import resolve_run

    runs = []
    for r in args.run or []:
        try:
            runs.append(resolve_run(r, config))
        except FileNotFoundError as exc:
            _err(f"error: {exc}")
            return 2
    report = graduation_checklist(config, args.server or None, runs)
    if args.json:
        _out(json.dumps(report, sort_keys=True, default=str))
    else:
        for server, entry in report.items():
            _out(f"{server}: {'graduated' if entry['graduated'] else 'not graduated'}")
            for name, item in entry["items"].items():
                mark = {True: "ok", False: "FAIL", None: "--"}[item.get("ok")]
                _out(f"  {name:<22} {mark:<5} {item.get('detail', '')}")
    return 0 if all(e["graduated"] for e in report.values()) else 1


# ---------------------------------------------------------------------------- parser

COMMANDS: dict[str, Callable[[argparse.Namespace, dict[str, Any]], int]] = {
    "list": cmd_list, "describe": cmd_describe, "lint": cmd_lint, "check": cmd_check, "resolve": cmd_resolve,
    "explain": cmd_explain, "fingerprint": cmd_fingerprint, "index build": cmd_index_build,
    "estimate": cmd_estimate, "retro-audit": cmd_retro_audit, "status": cmd_status, "calibrate": cmd_calibrate,
    "overlay init": cmd_overlay_init, "replay": cmd_replay, "diff-release": cmd_diff_release,
    "graduate": cmd_graduate,
}


def _add_common(p: argparse.ArgumentParser, *flags: str) -> None:
    if "json" in flags:
        p.add_argument("--json", action="store_true", help="print JSON")
    if "verbose" in flags:
        p.add_argument("-V", "--verbose-ds", dest="verbose", action="store_true", help="more detail")


def add_datasource_parsers(sub: Any) -> Any:
    """Register ``vbt datasource`` (alias ``vbt ds``) on an argparse subparsers object."""
    d = sub.add_parser("datasource", aliases=list(ALIASES),
                       help="data layer: descriptors, overlays, readiness, resolution (docs/DATA_LAYER.md)")
    ds = d.add_subparsers(dest="ds_cmd", required=True)

    p = ds.add_parser("list", help="sources, tables, release and cached readiness")
    p.set_defaults(handler=cmd_list)

    p = ds.add_parser("describe", help="grain, key, roles, coverage and vocabularies of a source or table")
    p.add_argument("target", help="<source> or <source>.<table>")
    p.set_defaults(handler=cmd_describe)

    p = ds.add_parser("lint", help="lint descriptors and overlays (roles, keys, bindings, plugins, defects)")
    p.add_argument("--strict", action="store_true", help="lint every descriptor as strict")
    p.add_argument("-q", "--quiet", action="store_true", help="print errors only")
    p.set_defaults(handler=cmd_lint)

    p = ds.add_parser("check", help="table, column, partition, index and tool readiness (runs the data child)")
    p.add_argument("--depth", choices=("shallow", "standard", "deep"), default=None,
                   help="check depth (default data.readiness.session_depth)")
    p.add_argument("--table", action="append", help="source.table (repeatable)")
    p.add_argument("--column", help="source.table.column")
    p.add_argument("--tool", help="server.tool: the call-scoped readiness of one tool")
    _add_common(p, "json", "verbose")
    p.set_defaults(handler=cmd_check)

    p = ds.add_parser("resolve", help="resolve identifiers with rules and candidates (offline, from sidecars)")
    p.add_argument("id_type", help="[source:]id_type, e.g. ensembl_gene or open_targets:ot_disease")
    p.add_argument("values", nargs="+")
    p.add_argument("--existence", choices=("universe", "bound", "upstream", "off"), default="universe")
    _add_common(p, "json", "verbose")
    p.set_defaults(handler=cmd_resolve)

    p = ds.add_parser("explain", help="binding, serve mode, reads, derived schema and text, defects of a tool")
    p.add_argument("target", nargs="?", help="<server>.<tool>")
    p.add_argument("--all", action="store_true", help="every bound tool")
    p.add_argument("--schemas", action="store_true",
                   help="with --json: only the upstream input schemas (the CI snapshot upstream_tool_schemas.json)")
    _add_common(p, "json")
    p.set_defaults(handler=cmd_explain)

    p = ds.add_parser("fingerprint", help="table fingerprints (what gets pinned)")
    p.add_argument("--table", action="append", help="source.table (repeatable)")
    p.add_argument("--write", action="store_true", help="write <data.cache_dir>/fingerprints.json")
    p.set_defaults(handler=cmd_fingerprint)

    p = ds.add_parser("index", help="sidecar indexes")
    ix = p.add_subparsers(dest="index_cmd", required=True)
    b = ix.add_parser("build", help="prebuild resolver sidecars and row-group value indexes")
    b.add_argument("--id-type", action="append", help="[source:]id_type (repeatable; default: every local one)")
    b.add_argument("--access-paths", action="store_true", help="row-group indexes of declared access paths")
    b.add_argument("--table", action="append", help="limit --access-paths to these tables")
    b.add_argument("--force", action="store_true", help="rebuild indexes that exist")
    b.add_argument("--huge", action="store_true",
                   help="with --access-paths: resumable builds of huge tables (time and byte budgets per step)")
    b.add_argument("--time-budget-s", type=float, default=None, help="--huge: seconds per step (default 3600)")
    b.add_argument("--max-bytes", type=int, default=None, help="--huge: decoded bytes per step (default unbounded)")
    b.add_argument("--once", action="store_true", help="--huge: run one step only")
    _add_common(b, "json")
    b.set_defaults(handler=cmd_index_build)

    p = ds.add_parser("estimate", help="memory estimate and admissibility on this host")
    p.add_argument("--table", action="append", help="source.table (repeatable)")
    p.add_argument("--tool", help="server.tool")
    p.set_defaults(handler=cmd_estimate)

    p = ds.add_parser("retro-audit", help="re-classify a recorded run's data calls offline")
    p.add_argument("run", help="run id, prefix, path or 'latest'")
    _add_common(p, "json")
    p.set_defaults(handler=cmd_retro_audit)

    p = ds.add_parser("status", help="per-server memory from reaper status files, host budget, calibrations")
    p.add_argument("run", nargs="?", help="run id, prefix, path or 'latest' (reads <run>/logs/mcp)")
    p.add_argument("--log-dir", help="a directory of <server>.status.json files")
    _add_common(p, "json")
    p.set_defaults(handler=cmd_status)

    p = ds.add_parser("calibrate", help="sample-and-scale memory calibration of tables (runs the data child)")
    p.add_argument("--table", action="append", help="source.table (repeatable)")
    p.add_argument("--row-groups", type=int, default=3, help="row groups sampled (1-3; default 3)")
    _add_common(p, "json")
    p.set_defaults(handler=cmd_calibrate)

    p = ds.add_parser("overlay", help="overlays of MCP servers")
    ov = p.add_subparsers(dest="overlay_cmd", required=True)
    o = ov.add_parser("init", help="scaffold an overlay for a third-party server from its tool listing",
                      description=THIRD_PARTY_HOWTO, formatter_class=argparse.RawDescriptionHelpFormatter)
    o.add_argument("server", help="the server's name in configs/mcp_servers.yaml")
    o.add_argument("--from-json", help="a saved tool listing ([{name, description, inputSchema}]) instead of "
                   "starting the server")
    o.add_argument("--out", help="output path ('-' prints it; default configs/data/overlays/<server>.yaml)")
    o.add_argument("--force", action="store_true", help="overwrite an existing overlay")
    o.set_defaults(handler=cmd_overlay_init)

    p = ds.add_parser("replay", help="re-execute recorded data calls on current data and compare row-key hashes")
    p.add_argument("run", help="run id, prefix, path or 'latest'")
    p.add_argument("tool_use_id", nargs="*", help="tool_use ids of recorded data calls")
    p.add_argument("--all", action="store_true", help="every call with a data provenance record")
    p.add_argument("--backend", choices=("auto", "inprocess", "bridge"), default="auto",
                   help="bridge (auto): a temporary MCP bridge with the call's server and the data child, under "
                        "the reaper's memory limit; inprocess: the data child's verbs in this process, with no "
                        "memory limit (opt-in, for debugging)")
    _add_common(p, "json")
    p.set_defaults(handler=cmd_replay)

    p = ds.add_parser("diff-release", help="role columns, types, encodings, vocabularies and matrix axes between "
                                           "two releases (runs the data child)")
    p.add_argument("--from", dest="from_", required=True, help="data root or release label of the old release")
    p.add_argument("--to", required=True, help="data root or release label of the new release")
    p.add_argument("--source", action="append", help="source to compare (repeatable; default: the sources whose "
                                                     "configured release is --from)")
    p.add_argument("--table", action="append", help="source.table (repeatable)")
    p.add_argument("--quick", action="store_true", help="footers only: no vocabulary snapshots or matrix axes")
    _add_common(p, "json")
    p.set_defaults(handler=cmd_diff_release)

    p = ds.add_parser("graduate", help="observe -> enforce checklist per server with retro-audit evidence")
    p.add_argument("server", nargs="*", help="servers (default: every overlay)")
    p.add_argument("--run", action="append", help="recorded run (observe mode) to retro-audit (repeatable)")
    _add_common(p, "json")
    p.set_defaults(handler=cmd_graduate)
    return d
