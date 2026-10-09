"""Authoring tools: how the system creates project-specific utilities as needed (docs/PROJECTS.md).

The data/tooling engineer (``configs/agents.yaml``) uses them when a dataset, format or helper the analysis needs
is missing. They work only while a project is active (``--project``, or the project's ``profile.yaml``), and
write only inside that project (destinations are derived from validated names, never taken from arguments) or
inside the calling agent's own work directory (staging copies and registration receipts):

* ``ProjectInfo`` -- the active project: its registered sources, utilities, plugins, skills, pending items, review
  mode and sandbox;
* ``InspectDataset`` -- profile a CSV/TSV/Parquet file the agent may read (columns, types, nulls, distinct values,
  uniqueness, examples) and draft a descriptor for it (a starting point; the engineer edits it);
* ``RegisterDataSpec`` -- register a descriptor, overlay or acquisition spec after ``vbt ds lint`` and
  ``vbt ds check`` pass (rejected with their errors otherwise); data files can be imported into the project;
* ``RegisterPlugin`` -- register a plugin of an existing kind after its conformance suite passes in the sandbox;
* ``RegisterUtility`` -- register a Python function or script with a JSON schema, docstring and tests; once the
  tests pass in the sandbox it is the tool ``util__<name>`` for this and later sessions of the project.

Validation, review (``projects.review``), provenance and storage are :mod:`vbt.projects.authoring`; every
registration is also written to ``work/<agent>/project_registrations/`` and registered as an artifact, so it is
listed in the run's MANIFEST and audit.html.
"""

from __future__ import annotations

import functools
import json
import math
import re
from pathlib import Path
from typing import Any, Awaitable, Callable

import yaml

from .base import Tool, ToolContext, ToolFailure, schema

__all__ = ["authoring_tools", "AUTHORING_TOOLS", "inspect_dataset", "draft_descriptor"]

AUTHORING_TOOLS = ("ProjectInfo", "InspectDataset", "RegisterDataSpec", "RegisterPlugin", "RegisterUtility")
MAX_FIXTURE_BYTES = 5_000_000
REVIEW_VERDICT = re.compile(r"VERDICT:\s*(APPROVE|REJECT)", re.IGNORECASE)


# ---------------------------------------------------------------------------- shared


def _refusals(fn: Callable[[ToolContext, dict[str, Any]], Awaitable[Any]]) -> Any:
    """A project path that would leave the project, or a sandbox that cannot run (``projects.sandbox: bwrap``
    without bubblewrap), is a refusal the agent sees, never a crash."""
    @functools.wraps(fn)
    async def wrapper(ctx: ToolContext, a: dict[str, Any]) -> Any:
        from ..projects.model import ProjectError

        try:
            return await fn(ctx, a)
        except (ProjectError, RuntimeError) as exc:
            raise ToolFailure(f"refused: {exc}") from None
    return wrapper


def _project(ctx: ToolContext) -> Any:
    from ..projects.model import active_project

    config = getattr(ctx.runtime, "config", None) or {}
    project = active_project(config)
    if project is None:
        raise ToolFailure("no project is active in this session: the system stores what it creates in a project. "
                          "Ask the user to start the session in one (`vbt project init NAME`, then `vbt --profile "
                          "<projects>/NAME/profile.yaml chat` or `--project NAME`).")
    return project


def _readable(ctx: ToolContext, raw: str) -> Path:
    from .policy import PathPolicy

    pol = PathPolicy.from_ctx(ctx)
    p = pol.resolve(str(raw), for_read=True)
    why = pol.read_denial(p)
    if why:
        raise ToolFailure(why)
    return p


def _text(ctx: ToolContext, a: dict[str, Any], key: str, path_key: str) -> str:
    """Inline ``key`` text, else the file ``path_key`` names (read under the agent's read policy)."""
    if a.get(key):
        return str(a[key])
    if a.get(path_key):
        p = _readable(ctx, str(a[path_key]))
        if not p.is_file():
            raise ToolFailure(f"no such file: {p}")
        return p.read_text(encoding="utf-8", errors="replace")
    raise ToolFailure(f"give {key} (the text) or {path_key} (a file you wrote)")


def _who(ctx: ToolContext) -> dict[str, Any]:
    return {"agent": ctx.agent, "run_id": getattr(ctx.run, "run_id", None), "invocation_id": ctx.invocation_id or None,
            "tool_use_id": ctx.tool_call_id or None}


def _author(ctx: ToolContext, project: Any) -> Any:
    from ..projects.authoring import Author
    from .policy import PathPolicy

    config = dict(getattr(ctx.runtime, "config", None) or {})
    work = ctx.run.agent_dir(ctx.agent) if ctx.agent not in ("cso", "scientific-reviewer") else \
        Path(ctx.run.dir) / "work" / "_cso"
    return Author(project, config, _who(ctx), policy=PathPolicy.from_ctx(ctx), work_dir=Path(work),
                  reviewer=_reviewer(ctx))


def _reviewer(ctx: ToolContext) -> Any:
    """``projects.review: reviewer``: the scientific reviewer sees the item and its validation and answers."""
    rt = ctx.runtime
    if not callable(getattr(rt, "delegate", None)) or "scientific-reviewer" not in (getattr(rt, "agents", None) or {}):
        return None

    async def review(summary: dict[str, Any]) -> Any:
        from ..projects.authoring import Review

        prompt = ("Review an item the system is about to register in its project, before it can be used by later "
                  "analyses. Judge whether it is scientifically sound and does what it claims: the data semantics "
                  "(identifiers, measures, coverage, what an absent row means), the code's correctness and the "
                  "adequacy of its tests. Answer with a line `VERDICT: APPROVE` or `VERDICT: REJECT`, followed by "
                  "your reasons.\n\n" + json.dumps(summary, indent=1, default=str)[:60000])
        res = await rt.delegate("scientific-reviewer", prompt, description=f"review project {summary.get('kind')} "
                                f"{summary.get('name')}", parent_agent=ctx.agent,
                                parent_invocation_id=ctx.invocation_id or None, depth=ctx.depth + 1)
        text = (getattr(res, "text", "") or getattr(res, "full_text", "") or "").strip()
        m = REVIEW_VERDICT.findall(text)
        verdict = m[-1].lower() if m else "reject"
        return Review("approve" if verdict == "approve" else "reject", "scientific-reviewer",
                      text[:4000] if m else f"no VERDICT line in the review: {text[:2000]}",
                      getattr(res, "invocation_id", None))
    return review


def _receipt(ctx: ToolContext, project: Any, outcome: Any) -> str | None:
    """Write the registration record into the agent's workspace and register it as an artifact (MANIFEST,
    audit.html). Returns its run-relative path."""
    rec = outcome.record or {}
    folder = ctx.run.agent_dir(ctx.agent) / "project_registrations"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{outcome.kind}-{outcome.name}-v{rec.get('version')}.json"
    path.write_text(json.dumps({"project": project.name, "project_dir": str(project.root), **outcome.to_json(),
                                "provenance": rec}, indent=1, default=str), encoding="utf-8")
    rel = ctx.run.rel(path)
    tests = ((rec.get("validation") or {}).get("tests") or {})
    review = rec.get("review") or {}
    desc = (f"Project {project.name}: {outcome.kind} {outcome.name} v{rec.get('version')} {outcome.status} "
            f"({rec.get('source_hash')}; " + (f"{tests.get('passed')} test(s) passed in {tests.get('sandbox')}; "
                                              if tests else "") + f"review: {review.get('verdict') or 'none'})")
    register = getattr(ctx.run, "register_artifact", None)
    if callable(register):
        register(rel, desc, ctx.agent, kind=f"project_{outcome.kind}")
    return rel


async def _finish(ctx: ToolContext, project: Any, outcome: Any) -> dict[str, Any]:
    out = outcome.to_json()
    ctx.trace("project_registration", project=project.name, kind=outcome.kind, name=outcome.name,
              status=outcome.status, ok=outcome.ok, version=(outcome.record or {}).get("version"),
              source_hash=(outcome.record or {}).get("source_hash"), message=outcome.message[:500])
    if not outcome.ok:
        raise ToolFailure(json.dumps(out, indent=1, default=str)[:20000])
    out["receipt"] = _receipt(ctx, project, outcome)
    if outcome.status == "registered":
        if outcome.kind == "utility":
            _add_utility_tool(ctx, project, outcome)
            out["tool"] = (outcome.record or {}).get("tool")
        else:
            from ..projects.reload import refresh_data_layer
            note = await refresh_data_layer(ctx.runtime, project)
            out["available"] = ("now: the data tools serve it in this session" if not note else note)
    return out


def _add_utility_tool(ctx: ToolContext, project: Any, outcome: Any) -> None:
    from ..projects.utilities import utility_tool

    tool = utility_tool(project, outcome.record or {})
    changed = getattr(ctx.runtime, "_on_tools_changed", None)
    if callable(changed):
        changed([tool])
    else:
        registry = getattr(ctx.runtime, "registry", None)
        if registry is not None:
            registry.add(tool)


# ---------------------------------------------------------------------------- ProjectInfo


def _project_info(ctx: ToolContext, a: dict[str, Any]) -> Any:
    from ..projects import ledger
    from ..projects.authoring import pending_items
    from ..projects.model import ProjectSettings
    from ..projects.prompt import resources
    from ..projects.sandbox import sandbox_plan

    project = _project(ctx)
    config = dict(getattr(ctx.runtime, "config", None) or {})
    settings = ProjectSettings.from_config(config, project)
    try:
        sandbox = sandbox_plan(config, network=settings.test_network)[0]
    except RuntimeError as exc:
        sandbox = f"unavailable: {exc}"
    events = ledger.read_ledger(project)[-int(a.get("events") or 10):]
    return {"project": project.summary(), "review": settings.review, "sandbox": sandbox,
            "check_depth": settings.check_depth, "resources": resources(project), "pending": pending_items(project),
            "recent_events": [{k: e.get(k) for k in ("at", "event", "kind", "name", "message")} for e in events],
            "layout": {"descriptors": str(project.descriptors_dir), "data": str(project.data_dir),
                       "utilities": str(project.utilities_dir), "plugins": str(project.plugins_dir),
                       "skills": str(project.skills_dir)},
            "conventions": ("descriptor root: ${VBT_PROJECT_DIR}/data/<source> with the files imported by "
                            "RegisterDataSpec(files=[...]); utility files: utility.py (entry function with a "
                            "docstring) + test_utility.py (test_* functions); names: lowercase identifiers")}


# ---------------------------------------------------------------------------- InspectDataset


def _jsonable(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


#: The csv/tsv format plugins infer column types from the first block of this many bytes.
_FIRST_BLOCK = 1 << 20
_NUMERIC = ("int", "uint", "double", "float", "decimal")


def _read_text_table(path: Path, fmt: str, max_rows: int, max_bytes: int = 0) -> tuple[Any, dict[str, str], bool]:
    """``(table, column_types, cut)``: the types the csv/tsv format plugin infers (the first MiB; an all-empty
    column is text), then every column a later block does not convert to that type is read as string. Those columns
    are returned: a descriptor must declare them (``options.column_types``) or the format plugin refuses the file.
    ``cut``: reading stopped at ``max_rows`` rows or ``max_bytes`` decoded bytes."""
    import pyarrow as pa
    import pyarrow.csv as pcsv

    parse = pcsv.ParseOptions(delimiter="\t" if fmt == "tsv" else ",")
    read = pcsv.ReadOptions(block_size=_FIRST_BLOCK, use_threads=False)
    # as the format plugin reads: only an unquoted empty cell is null ("n/a", "NA" are values)
    nulls = {"null_values": [""], "strings_can_be_null": True, "quoted_strings_can_be_null": False}
    try:
        first = next(iter(pcsv.open_csv(path, parse_options=parse, read_options=read,
                                        convert_options=pcsv.ConvertOptions(**nulls))))
    except StopIteration:
        first = None
    types = {f.name: (pa.string() if pa.types.is_null(f.type) else f.type) for f in first.schema} if first else {}
    overrides: dict[str, str] = {}
    for _ in range(len(types) + 1):
        convert = pcsv.ConvertOptions(column_types=types, **nulls)
        try:
            reader = pcsv.open_csv(path, parse_options=parse, read_options=read, convert_options=convert)
            batches, n, size, cut = [], 0, 0, False
            for b in reader:
                batches.append(b)
                n += b.num_rows
                size += b.nbytes
                if n >= max_rows or (max_bytes and size >= max_bytes):
                    cut = True
                    break
            return (pa.Table.from_batches(batches, schema=reader.schema) if batches else
                    reader.schema.empty_table()), overrides, cut
        except pa.ArrowInvalid as exc:
            m = re.search(r"column #(\d+)", str(exc))
            names = list(types)
            if not m or int(m.group(1)) >= len(names) or pa.types.is_string(types[names[int(m.group(1))]]):
                raise
            name = names[int(m.group(1))]
            types[name] = pa.string()
            overrides[name] = "string"
    raise ValueError(f"{path}: the column types do not settle")


def _delimited_format(path: Path) -> str:
    """``tsv`` when the header line has more tabs than commas (whatever the suffix: ``.txt`` exports), else ``csv``."""
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:     # type: ignore[operator]
        header = f.readline()
    return "tsv" if header.count("\t") > header.count(",") else "csv"


def _key_of(table: Any, columns: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """``(key columns, the nullable ones)``: a unique null-free column (an *_id/key/accession/code name first), else
    the first unique combination of two (or, up to 500,000 rows, three) null-free text or integer columns, else one
    that needs columns with nulls (they are the nullable key parts)."""
    import itertools

    rows = table.num_rows
    if rows == 0:
        return [], []
    keyable = [c for c in columns if str(c["type"]).startswith(("string", "large_string", "int", "uint", "dictionary"))
               and _declarable(c["name"])]
    cands = [c for c in keyable if c["nulls"] == 0]
    singles = [c for c in cands if c.get("unique")]
    named = [c for c in singles if re.search(r"(^|_)(id|key|accession|code)$", c["name"].lower())]
    if singles:
        return [(named or singles)[0]["name"]], []
    sizes = (2, 3) if rows <= 500_000 else (2,)
    for pool in (cands, keyable):
        names = [c["name"] for c in sorted(pool, key=lambda c: -(c.get("distinct") or 0))][:12]
        for k in sizes:
            for combo in itertools.combinations(names, k):
                if table.select(list(combo)).group_by(list(combo)).aggregate([]).num_rows == rows:
                    ordered = sorted(combo, key=table.column_names.index)       # the file's column order
                    return ordered, [n for n in ordered if table.column(n).null_count]
    return [], []


def _declarable(name: str) -> bool:
    """A descriptor declares a column by its name, read as a path: a ``.``, a backtick or a bracket would make it a
    nested field, so such columns cannot be declared (the data layer leaves undeclared columns out)."""
    return not re.search(r"[.`\[\]/^@]", name) and bool(name.strip())


def _pattern_of(col: Any) -> str:
    """The narrowest of a few generic patterns every value of a key column matches (``local_key``'s canonical)."""
    import pyarrow.compute as pc

    text = pc.cast(col, "string")
    candidates = []
    prefixes = {m.group(0) for v in pc.unique(text).slice(0, 2000).to_pylist() if v is not None
                for m in [re.match(r"[A-Za-z]+(?=\d+$)", v)] if m}
    if len(prefixes) == 1:
        candidates.append(f"^{re.escape(prefixes.pop())}\\d+$")
    candidates += [r"^\d+$", r"^[A-Za-z0-9_.:-]+$", r"^\S(?:.*\S)?$", r"^.+$"]
    for pat in candidates:
        if pc.all(pc.match_substring_regex(text, pat)).as_py():
            return pat
    return r"^.+$"


#: InspectDataset runs in the harness process: what it profiles is bounded in rows and in decoded bytes.
INSPECT_MAX_ROWS = 2_000_000
INSPECT_MAX_BYTES = 512 << 20


def inspect_dataset(path: Path, *, sample_rows: int = 5, max_rows: int = INSPECT_MAX_ROWS,
                    max_bytes: int = INSPECT_MAX_BYTES) -> dict[str, Any]:
    """Columns, types, nulls, distinct counts and examples of a CSV/TSV/Parquet file, the column(s) that identify a
    row, and (CSV/TSV) the columns whose type the first block does not show. At most ``max_rows`` rows and
    ``max_bytes`` decoded bytes are profiled (``complete`` says whether that was the whole file); the row count of a
    Parquet file comes from its footer."""
    import pyarrow as pa
    import pyarrow.compute as pc

    suffix = "".join(path.suffixes[-2:]).lower()
    fmt = "parquet" if suffix.endswith(".parquet") or path.suffix == ".pq" else _delimited_format(path)
    total_rows: int | None = None
    column_types: dict[str, str] = {}
    if fmt == "parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        total_rows = pf.metadata.num_rows
        batches, n, size, cut = [], 0, 0, False
        for b in pf.iter_batches(batch_size=65536):
            batches.append(b)
            n += b.num_rows
            size += b.nbytes
            if n >= max_rows or (max_bytes and size >= max_bytes):
                cut = n < total_rows
                break
        table = pa.Table.from_batches(batches, schema=pf.schema_arrow) if batches else pf.schema_arrow.empty_table()
    else:
        table, column_types, cut = _read_text_table(path, fmt, max_rows, max_bytes)
    rows = table.num_rows
    columns = []
    for name in table.column_names:
        col = table.column(name)
        nulls = col.null_count
        try:
            distinct = len(pc.unique(col))
        except (pa.ArrowNotImplementedError, pa.ArrowInvalid, TypeError):
            distinct = None
        non_null = rows - nulls
        entry: dict[str, Any] = {"name": name, "type": str(col.type), "nulls": nulls, "distinct": distinct,
                                 "unique": distinct is not None and non_null > 0 and distinct - (1 if nulls else 0)
                                 == non_null}
        if pa.types.is_integer(col.type) or pa.types.is_floating(col.type):
            mm = pc.min_max(col)
            entry["min"], entry["max"] = _jsonable(mm["min"].as_py()), _jsonable(mm["max"].as_py())
        entry["examples"] = [_jsonable(v) for v in col.drop_null().slice(0, 5).to_pylist()]
        columns.append(entry)
    key, nullable = _key_of(table, columns)
    notes = []
    skipped = [c["name"] for c in columns if not _declarable(c["name"])]
    if skipped:
        notes.append(f"columns {skipped} cannot be declared (a '.' or other path character in the name reads as a "
                     "nested field); the draft leaves them out, so the data tools do not serve them")
    if not key:
        notes.append("no column or combination of up to three columns identifies a row: choose the key yourself "
                     "(the data may hold duplicate rows)")
    elif nullable:
        notes.append(f"the key needs columns with empty values {nullable}: they are declared key.nullable; check that "
                     "an empty value is part of the row's identity")
    return {"path": str(path), "format": fmt, "bytes": path.stat().st_size,
            "rows": total_rows if total_rows is not None else rows, "rows_profiled": rows,
            "complete": not cut, "columns": columns,
            "key": key, "key_nullable": nullable,
            "key_pattern": _pattern_of(table.column(key[0])) if len(key) == 1 else None,
            "column_types": column_types, "notes": notes,
            "sample": [{k: _jsonable(v) for k, v in r.items()} for r in table.slice(0, sample_rows).to_pylist()]}


def _role(col: dict[str, Any], rows: int, key: list[str]) -> dict[str, Any]:
    t = str(col["type"])
    if col["name"] in key:
        if len(key) == 1:
            return {"role": "identifier", "id_type": "__KEY__", "self": True}
        return {"role": "identifier", "resolvable": False}
    if t.startswith(_NUMERIC):
        return {"role": "measure", "statistic": "numeric", "missing": "unknown"}
    if t == "bool":
        return {"role": "flag", "missing": "unknown"}
    distinct = col.get("distinct") or 0
    if t.startswith(("string", "large_string", "dictionary")) and distinct and distinct <= max(50, rows // 20):
        return {"role": "category", "vocab": "data"}
    return {"role": "payload"}


def draft_descriptor(info: dict[str, Any], source: str, table: str, *, release: str = "local-1") -> str:
    """A descriptor draft for one file (to be edited: grain, coverage and roles are guesses from the data)."""
    path = Path(info["path"])
    key = list(info.get("key") or [])
    cols: dict[str, Any] = {}
    for c in info["columns"]:
        if not _declarable(c["name"]):
            continue
        spec = _role(c, int(info.get("rows_profiled") or 0), key)
        if spec.get("id_type") == "__KEY__":
            spec["id_type"] = f"{table}_key"
        cols[c["name"]] = spec
    fmt: Any = info["format"]
    if info.get("column_types"):
        fmt = {"plugin": info["format"], "options": {"column_types": dict(info["column_types"])}}
    doc: dict[str, Any] = {
        "schema": "vbt.datasource/1", "source": source,
        "title": f"{path.name} (drafted by InspectDataset: review the roles, grain and coverage)",
        "root": "${VBT_PROJECT_DIR}/data/" + source,
        "release": {"expect": release, "from": "literal"},
        "defaults": {"format": fmt, "layout": "single_file", "missing": "unknown"},
    }
    if len(key) == 1:
        doc["id_types"] = {f"{table}_key": {"plugin": "local_key",
                                            "options": {"canonical": info.get("key_pattern") or r"^.+$"},
                                            "universe": f"{table}.{key[0]}"}}
    doc["tables"] = {table: {
        "kind": "entity", "path": path.name,
        "grain": f"one row of {path.name}" + (f", keyed by {', '.join(key)}" if key else ""),
        "key": {"columns": key or ["<choose the column(s) that identify a row>"], "check": "full",
                **({"nullable": list(info["key_nullable"])} if info.get("key_nullable") else {})},
        "coverage": {"statement": f"The rows of {path.name} as provided ({info['rows']} rows); an absent row is "
                                  "not recorded here.", "absence_means": "unknown"},
        "columns": cols}}
    return yaml.safe_dump(doc, sort_keys=False, width=120, allow_unicode=True)


def _inspect(ctx: ToolContext, a: dict[str, Any]) -> Any:
    p = _readable(ctx, str(a.get("path") or ""))
    if not p.is_file():
        raise ToolFailure(f"no such file: {p}")
    try:
        info = inspect_dataset(p, sample_rows=int(a.get("sample_rows") or 5))
    except ImportError:
        raise ToolFailure("InspectDataset needs pyarrow (pip install 'vbt-harness[analysis]')") from None
    except Exception as exc:  # noqa: BLE001 - an unreadable file is the answer
        raise ToolFailure(f"{p} could not be read as {p.suffix or 'a table'}: {type(exc).__name__}: {exc}") from None
    source = re.sub(r"[^a-z0-9_]", "_", str(a.get("source") or p.stem).lower()).strip("_") or "dataset"
    if not source[0].isalpha():
        source = "d_" + source
    table = re.sub(r"[^a-z0-9_]", "_", str(a.get("table") or "rows").lower()).strip("_") or "rows"
    info["draft_descriptor"] = draft_descriptor(info, source, table)
    info["next"] = ("Edit the draft (roles, grain, coverage, release), write it to a file in your workspace, then "
                    f"RegisterDataSpec(kind='descriptor', path=<that file>, files=['{p}'], why=...).")
    return info


# ---------------------------------------------------------------------------- registrations


@_refusals
async def _register_data_spec(ctx: ToolContext, a: dict[str, Any]) -> Any:
    project = _project(ctx)
    text = _text(ctx, a, "content", "path")
    files = [_readable(ctx, str(f)) for f in a.get("files") or []]
    for f in files:
        if not f.is_file():
            raise ToolFailure(f"no such file to import: {f}")
    outcome = await _author(ctx, project).register_data_spec(str(a.get("kind") or "descriptor"), text,
                                                             why=str(a.get("why") or ""),
                                                             source=a.get("source"), files=files)
    return await _finish(ctx, project, outcome)


@_refusals
async def _register_plugin(ctx: ToolContext, a: dict[str, Any]) -> Any:
    project = _project(ctx)
    text = _text(ctx, a, "content", "path")
    outcome = await _author(ctx, project).register_plugin(str(a.get("kind") or ""), text, why=str(a.get("why") or ""))
    return await _finish(ctx, project, outcome)


def _fixtures(ctx: ToolContext, directory: Path | None) -> dict[str, bytes]:
    if directory is None or not (directory / "fixtures").is_dir():
        return {}
    out: dict[str, bytes] = {}
    total = 0
    for p in sorted((directory / "fixtures").rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        _readable(ctx, str(p))
        total += p.stat().st_size
        if total > MAX_FIXTURE_BYTES:
            raise ToolFailure(f"fixtures are larger than {MAX_FIXTURE_BYTES:,} bytes; keep test data small")
        out[p.relative_to(directory / "fixtures").as_posix()] = p.read_bytes()
    return out


@_refusals
async def _register_utility(ctx: ToolContext, a: dict[str, Any]) -> Any:
    project = _project(ctx)
    directory = None
    if a.get("directory"):
        directory = _readable(ctx, str(a["directory"]))
        if not directory.is_dir():
            raise ToolFailure(f"no such directory: {directory}")
        a = {**a, "code_path": a.get("code_path") or str(directory / "utility.py"),
             "tests_path": a.get("tests_path") or str(directory / "test_utility.py")}
    code = _text(ctx, a, "code", "code_path")
    tests = _text(ctx, a, "tests", "tests_path")
    input_schema = a.get("input_schema")
    if isinstance(input_schema, str):
        try:
            input_schema = json.loads(input_schema)
        except ValueError:
            raise ToolFailure("input_schema must be a JSON schema object") from None
    outcome = await _author(ctx, project).register_utility(
        str(a.get("name") or ""), description=str(a.get("description") or ""), module_text=code, tests_text=tests,
        input_schema=input_schema, why=str(a.get("why") or ""), entry=str(a.get("entry") or "run"),
        mode=str(a.get("mode") or "function"), timeout_s=a.get("timeout_s"), fixtures=_fixtures(ctx, directory))
    return await _finish(ctx, project, outcome)


# ---------------------------------------------------------------------------- registration


def authoring_tools() -> list[Tool]:
    path = {"type": "string", "description": "a file you wrote (workspace-relative or absolute)"}
    why = {"type": "string", "description": "why the project needs it: the request or gap it serves (recorded)"}
    return [
        Tool("ProjectInfo", "The active project: its registered data sources (and tables), utilities (tools), "
                            "plugins and skills, items waiting for review, the review mode, the sandbox, recent "
                            "registrations and the file conventions.",
             schema({"events": {"type": "integer", "description": "recent ledger events to show (default 10)"}}),
             _project_info, source="harness", blocking=True),
        Tool("InspectDataset", "Profile a CSV, TSV or Parquet file you can read (columns, types, nulls, distinct "
                               "values, uniqueness, examples, sample rows) and draft a project descriptor for it. "
                               "The draft is a starting point: check the roles, key, grain and coverage before "
                               "registering it with RegisterDataSpec.",
             schema({"path": path, "source": {"type": "string", "description": "source name for the draft"},
                     "table": {"type": "string", "description": "table name for the draft (default rows)"},
                     "sample_rows": {"type": "integer"}}, ["path"]),
             _inspect, source="harness", blocking=True),
        Tool("RegisterDataSpec", "Register a data source in the active project: kind descriptor (a vbt.datasource/1 "
                                 "file; `files` copies data files into the project's data/<source>/), overlay (a "
                                 "vbt.overlay/1 file for a server the core ships none for) or acquisition (the "
                                 "`acquisition:` section of an existing project descriptor `source`). It is linted "
                                 "(vbt ds lint) and checked on the data (vbt ds check) first and refused with the "
                                 "errors otherwise; once registered, the mcp__data__* tools serve its tables.",
             schema({"kind": {"type": "string", "enum": ["descriptor", "overlay", "acquisition"]},
                     "path": path, "content": {"type": "string", "description": "the YAML (instead of path)"},
                     "source": {"type": "string", "description": "acquisition: the project descriptor to extend"},
                     "files": {"type": "array", "items": {"type": "string"},
                               "description": "data files to import into <project>/data/<source>/"},
                     "why": why}, ["kind", "why"]),
             _register_data_spec, source="harness"),
        Tool("RegisterPlugin", "Register a plugin of an existing kind (format, layout, statistic, identifier, "
                               "envelope, acquisition) in the active project: a module with one @register class "
                               "and a literal `name`. Its kind's conformance suite runs in the sandbox first; it is "
                               "refused unless every case passes.",
             schema({"kind": {"type": "string"}, "path": path,
                     "content": {"type": "string", "description": "the module source (instead of path)"},
                     "why": why}, ["kind", "why"]),
             _register_plugin, source="harness"),
        Tool("RegisterUtility", "Register a project utility: a Python function (mode function: `entry`, default "
                                "run, with a docstring, called with the arguments as keywords) or script (mode "
                                "script: reads the JSON arguments on stdin, prints its result), its JSON schema "
                                "and its tests (test_utility.py: test_* functions; `import utility`). The tests run "
                                "in the sandbox first; once they pass it is the tool util__<name> in this and later "
                                "sessions of the project. Put utility.py, test_utility.py and fixtures/ in one "
                                "directory and pass `directory`, or pass the code inline.",
             schema({"name": {"type": "string", "description": "lowercase identifier; the tool is util__<name>"},
                     "description": {"type": "string", "description": "what it computes and returns (the tool's "
                                                                      "description)"},
                     "input_schema": {"type": "object", "description": "JSON schema of the arguments: {type: "
                                                                       "object, properties, required}"},
                     "directory": path, "code": {"type": "string"}, "code_path": path,
                     "tests": {"type": "string"}, "tests_path": path,
                     "entry": {"type": "string", "description": "function name (default run)"},
                     "mode": {"type": "string", "enum": ["function", "script"]},
                     "timeout_s": {"type": "number"}, "why": why}, ["name", "description", "input_schema", "why"]),
             _register_utility, source="harness"),
    ]
