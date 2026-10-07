#!/usr/bin/env python3
"""The data child's entry point: a FastMCP server named ``data`` with the hidden verbs (§4, §11.8).

Launched as ``${vars.mcp_python} -E <project_root>/src/vbt/datalayer/service/server.py`` (through the
reaper). ``-E`` ignores ``PYTHONPATH`` and ``vbt`` may not be installed in that interpreter, so the
first thing this file does is put ``Path(__file__).resolve().parents[3]`` (``src/``) on ``sys.path``.

Settings come from ``VBT_DATA_SETTINGS`` (JSON written by ``DataGateway.extra_servers()``); without
it the in-code defaults apply. Every module of ``service/verbs/`` contributes ``VERBS``; each hidden
verb is registered as a tool with exactly one parameter, ``request`` (the ipc request model as a JSON
object, or JSON text). The public verbs (``PUBLIC_VERBS``, listed as ``mcp__data__<verb>``) take their
own arguments, which are the payload: the child lists the catalog-free skeleton of each verb's schema
(``derive.tools.native_input_schema``) and the gateway's listing adds the table enums and ``where``
schemas. The gateway adds the calling ``agent`` to the arguments, so the child's listed schemas
accept extra properties. At most ``data.service.max_concurrency`` verbs run at once.

Command line (preflight and ``vbt ds`` use it without MCP)::

    server.py                                   serve MCP over stdio
    server.py --check --json [--table S.T ...] [--depth shallow|standard|deep]
    server.py --build-index --id-type [SOURCE:]ID_TYPE
    server.py --build-index --access-paths [--table S.T ...]
    server.py --list-verbs
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[3])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import argparse  # noqa: E402
import json  # noqa: E402
from typing import Any, Callable, Mapping  # noqa: E402

from vbt.datalayer.ipc import VERB_CHECK  # noqa: E402
from vbt.datalayer.service import ServiceContext  # noqa: E402
from vbt.datalayer.service.verbs import load_verbs  # noqa: E402

__all__ = ["build_context", "make_server", "tool_function", "public_schema", "main"]


def build_context() -> ServiceContext:
    from vbt.datalayer.settings import DataSettings

    return ServiceContext(DataSettings.from_env())


def _payload(request: Any) -> dict[str, Any]:
    if request is None:
        return {}
    if isinstance(request, (str, bytes)):
        text = request.decode("utf-8") if isinstance(request, bytes) else request
        return dict(json.loads(text or "{}"))
    return dict(request)


def tool_function(ctx_ref: Callable[[], ServiceContext], name: str,
                  verb: Callable[[ServiceContext, Mapping[str, Any]], dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """The FastMCP tool of one verb: one ``request`` parameter, the context built on first use."""

    def tool(request: dict[str, Any] | str | None = None) -> dict[str, Any]:
        ctx = ctx_ref()
        with ctx.slots:
            return verb(ctx, _payload(request))

    tool.__name__ = name.lstrip("_") or "verb"
    tool.__doc__ = (verb.__module__.rsplit(".", 1)[-1] + ": internal data-layer verb " + name +
                    " (request: the vbt.datalayer.ipc request model as JSON)")
    return tool


def public_schema(verb: str) -> dict[str, Any]:
    """The child's listed schema of one public verb: its arguments without the catalog's enums (the
    gateway's listing derives those), open to the ``agent`` the gateway adds."""
    from types import SimpleNamespace

    from vbt.datalayer.derive.tools import native_input_schema

    schema = native_input_schema(SimpleNamespace(sources={}), verb, [])
    for prop in schema["properties"].values():
        if prop.get("enum") == []:
            del prop["enum"]
        prop.pop("x-vbt-where", None)
    schema["additionalProperties"] = True
    return schema


def _public_tool(ctx_ref: Callable[[], ServiceContext], name: str,
                 verb: Callable[[ServiceContext, Mapping[str, Any]], dict[str, Any]]) -> Any:
    """The FastMCP tool of one public verb: the call's arguments are the payload (a lone ``request``
    argument, the hidden verbs' form, is accepted too)."""
    import anyio
    from fastmcp.tools import Tool

    class PublicVerb(Tool):
        async def run(self, arguments: dict[str, Any]) -> Any:
            args = dict(arguments or {})
            payload = _payload(args["request"]) if set(args) == {"request"} else args

            def call() -> dict[str, Any]:
                ctx = ctx_ref()
                with ctx.slots:
                    return verb(ctx, payload)

            return self.convert_result(await anyio.to_thread.run_sync(call))

    from vbt.datalayer.derive.tools import VERB_DESCRIPTIONS

    return PublicVerb(name=name, description=VERB_DESCRIPTIONS.get(name, f"data-layer verb {name}"),
                      parameters=public_schema(name))


def make_server(ctx: ServiceContext | None = None) -> Any:
    from fastmcp import FastMCP

    from vbt.datalayer.service.verbs.public import PUBLIC_VERBS

    holder: dict[str, ServiceContext] = {}
    if ctx is not None:
        holder["ctx"] = ctx

    def ctx_ref() -> ServiceContext:
        if "ctx" not in holder:
            holder["ctx"] = build_context()
        return holder["ctx"]

    mcp = FastMCP("data")
    for name, verb in sorted(load_verbs().items()):
        if name in PUBLIC_VERBS:
            mcp.add_tool(_public_tool(ctx_ref, name, verb))
            continue
        fn = tool_function(ctx_ref, name, verb)
        mcp.tool(name=name, description=fn.__doc__)(fn)
    return mcp


def _check(args: argparse.Namespace) -> int:
    ctx = build_context()
    verb = load_verbs()[VERB_CHECK]
    resp = verb(ctx, {"tables": list(args.table or []), "depth": args.depth})
    if args.json:
        print(json.dumps(resp, sort_keys=True))
    else:
        for ref, t in sorted(resp["tables"].items()):
            print(f"{ref}: {t['status']}")
            for c in t["checks"]:
                if not c["ok"]:
                    print(f"  [{c['level']}] {c['name']}: {c['detail']}")
        for ref, err in sorted(resp["errors"].items()):
            print(f"{ref}: error: {err}")
    return 0


def _build_index(args: argparse.Namespace) -> int:
    from vbt.datalayer.service.verbs.index_build import build_access_path, build_resolver_index

    ctx = build_context()
    results: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    if args.id_type:
        for name in args.id_type:
            try:
                src, _ = ctx.catalog.id_type(name)
                bare = name.partition(":")[2] or name
                results.append({"id_type": f"{src}:{bare}",
                                **build_resolver_index(ctx, src, bare, force=args.force).model_dump()})
            except Exception as exc:  # noqa: BLE001 - one failed index never hides the others
                errors[name] = f"{type(exc).__name__}: {exc}"
    if args.access_paths:
        refs = list(args.table or []) or ctx.table_refs(item_tables=False)
        for ref in refs:
            t = ctx.table(ref)
            for ap in t.physical_spec.access_paths:
                if ap.via != "sidecar_index":
                    continue
                key = f"{ref}:{','.join(ap.columns)}"
                try:
                    results.append({"table": ref, "access_path": list(ap.columns),
                                    **build_access_path(ctx, ref, list(ap.columns), force=args.force).model_dump()})
                except Exception as exc:  # noqa: BLE001
                    errors[key] = f"{type(exc).__name__}: {exc}"
    print(json.dumps({"built": results, "errors": errors}, sort_keys=True))
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="server.py", description="vbt data child (hidden data-layer verbs)")
    parser.add_argument("--check", action="store_true", help="run readiness checks and exit")
    parser.add_argument("--json", action="store_true", help="print JSON (with --check)")
    parser.add_argument("--table", action="append", help="source.table (repeatable)")
    parser.add_argument("--depth", default="standard", choices=("shallow", "standard", "deep"))
    parser.add_argument("--build-index", action="store_true", help="build sidecar indexes and exit")
    parser.add_argument("--id-type", action="append", help="resolver index of [source:]id_type (repeatable)")
    parser.add_argument("--access-paths", action="store_true", help="row-group indexes of sidecar access paths")
    parser.add_argument("--force", action="store_true", help="rebuild indexes that exist")
    parser.add_argument("--list-verbs", action="store_true", help="print the verb names and exit")
    args = parser.parse_args(argv)
    if args.list_verbs:
        print(json.dumps(sorted(load_verbs())))
        return 0
    if args.check:
        return _check(args)
    if args.build_index:
        if not args.id_type and not args.access_paths:
            parser.error("--build-index needs --id-type or --access-paths")
        return _build_index(args)
    make_server().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
