"""``vbt datasource`` (alias ``vbt ds``): inspect and check the data layer (docs/DATA_LAYER.md §16).

``vbt data`` is the Zenodo fetcher, hence the longer name. Phase-1 commands::

    vbt ds list                                    sources, tables, release, readiness summary
    vbt ds describe <source>[.<table>]             grain, key, roles, coverage, vocabularies
    vbt ds lint [--strict]                         descriptors and overlays
    vbt ds check [--depth D] [--table S.T] [--column S.T.C] [--tool server.tool] [--json]
    vbt ds resolve <id_type> <value...>            resolutions with rules and candidates
    vbt ds explain <server>.<tool> | --all         binding, serve mode, reads, derived text, defects
    vbt ds fingerprint [--write]                   table fingerprints (what gets pinned)
    vbt ds index build [--id-type T] [--access-paths]
    vbt ds estimate --table S.T | --tool server.tool
    vbt ds retro-audit <run>                       re-classify a recorded run offline

Handlers follow the repo convention ``handler(args, config) -> int``. Nothing here imports pyarrow:
commands that read data (``check``, ``fingerprint``, ``index build``, ``estimate``) run the data
child's command line (``<mcp_python> -E src/vbt/datalayer/service/server.py ...``) as a subprocess.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

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
    tool = None
    if args.tool:
        try:
            tool = _split_tool(args.tool)
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
                       "unchecked": r.unchecked, "unavailable_partitions": r.unavailable_partitions}
    if args.json:
        body = {"tables": response.get("tables") or {}, "table_errors": response.get("table_errors") or {},
                "depth": response.get("depth"), "indexes": indexes,
                "tools": {"unready": dr.unready, "partial": dr.partial,
                          "bound": {s: list(t) for s, t in sorted(dr.bound.items())}}}
        if tool_result is not None:
            body["tool"] = tool_result
        _out(json.dumps(body, sort_keys=True, default=str))
        return 0 if (tool_result or {}).get("ready", True) else 1
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
    return 0


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


def cmd_resolve(args: argparse.Namespace, config: dict[str, Any]) -> int:
    resolver, catalog = resolver_for(config)
    try:
        source, _spec = catalog.id_type(args.id_type)
    except Exception as exc:  # noqa: BLE001
        _err(f"error: {exc}")
        return 2
    qualified = args.id_type if ":" in args.id_type else f"{source}:{args.id_type}"
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
    try:
        derived = annotate_schema(contract, dict(schema or {"type": "object", "properties": {}}), catalog=catalog,
                                  registry=registry, enum_max=enum_max)
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


def cmd_explain(args: argparse.Namespace, config: dict[str, Any]) -> int:
    settings, catalog, registry = _catalog(config)
    if args.all:
        targets = [(s, t) for s in catalog.servers() for t in catalog.tools(s)]
    elif args.target:
        try:
            targets = [_split_tool(args.target)]
        except ValueError as exc:
            _err(f"error: {exc}")
            return 2
    else:
        _err("error: name a tool (<server>.<tool>) or pass --all")
        return 2
    rc = 0
    for server, tool in targets:
        try:
            for line in explain_tool(catalog, registry, server, tool, settings=settings):
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


def _run_child(config: dict[str, Any], *child_args: str, timeout: float = 3600.0) -> tuple[int, str, str]:
    from ..preflight import data_child_command

    cmd, env = data_child_command(config, *child_args)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 2, "", f"could not run {cmd[0]}: {exc}"
    return proc.returncode, proc.stdout, proc.stderr


def cmd_index_build(args: argparse.Namespace, config: dict[str, Any]) -> int:
    from ..preflight import DataCheckUnavailable

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
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=3600, env=env)
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
            contract = catalog.contract(server, tool)
        except Exception as exc:  # noqa: BLE001
            _err(f"error: {exc}")
            return 2
        tables = sorted({catalog.table(r).physical if catalog.table(r).is_item_table else r for r in contract.tables})
        full = set(contract.full_table_reads)
    elif args.table:
        tables = list(args.table)
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
    for ref, err in sorted((stats.get("table_errors") or stats.get("errors") or {}).items()):
        _out(f"{ref}: error: {err}")
    if server is not None:
        _out(f"tool {args.tool}: upstream peak ~{total_upstream / 1e6:.0f} MB"
             + (f"; server limit {limit} MB" if limit else "")
             + (f"; host {host} MB" if host else "")
             + ("; admissible" if not limit or total_upstream / 1e6 <= limit else "; NOT admissible (too_large)"))
    elif host:
        _out(f"host memory: {host} MB")
    return 0


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


# ---------------------------------------------------------------------------- parser

COMMANDS: dict[str, Callable[[argparse.Namespace, dict[str, Any]], int]] = {
    "list": cmd_list, "describe": cmd_describe, "lint": cmd_lint, "check": cmd_check, "resolve": cmd_resolve,
    "explain": cmd_explain, "fingerprint": cmd_fingerprint, "index build": cmd_index_build,
    "estimate": cmd_estimate, "retro-audit": cmd_retro_audit,
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
    return d
