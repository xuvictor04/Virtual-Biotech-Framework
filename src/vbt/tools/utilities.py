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
import os
import re
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

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
    return {"project": project.summary(), "review": settings.review, "plugin_review": settings.review_for("plugin"),
            "sandbox": sandbox,
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


def _read_text_table(paths: Path | Sequence[Path], fmt: str, max_rows: int, max_bytes: int = 0
                     ) -> tuple[Any, dict[str, str], bool, int]:
    """``(table, column_types, cut, files read)`` of one CSV/TSV file or of the files of a directory, read as the
    csv/tsv format plugin reads them: types inferred per file from its first MiB (an all-empty column is text). A
    column whose type differs between files is declared: the other files' type when it is empty in one file's first
    MiB, ``double`` for int and float, else ``string``; a column a later block does not convert is declared string.
    Those columns are returned: a descriptor must declare them (``options.column_types``) or the format plugin
    refuses the files. ``cut``: reading stopped at ``max_rows`` rows or ``max_bytes`` decoded bytes."""
    import pyarrow as pa
    import pyarrow.csv as pcsv

    files = [paths] if isinstance(paths, Path) else list(paths)
    parse = pcsv.ParseOptions(delimiter="\t" if fmt == "tsv" else ",")
    read = pcsv.ReadOptions(block_size=_FIRST_BLOCK, use_threads=False)
    # as the format plugin reads: only an unquoted empty cell is null ("n/a", "NA" are values)
    nulls = {"null_values": [""], "strings_can_be_null": True, "quoted_strings_can_be_null": False}
    cache: dict[Path, tuple[dict[str, Any], set[str]]] = {}

    def inferred(path: Path) -> tuple[dict[str, Any], set[str]]:
        """The plugin's per-file types, and the columns that are empty in the first MiB (typed as text)."""
        if path not in cache:
            try:
                first = next(iter(pcsv.open_csv(path, parse_options=parse, read_options=read,
                                                convert_options=pcsv.ConvertOptions(**nulls))))
                empty = {f.name for f in first.schema if pa.types.is_null(f.type)}
                cache[path] = ({f.name: (pa.string() if f.name in empty else f.type) for f in first.schema}, empty)
            except StopIteration:
                cache[path] = ({}, set())
        return cache[path]

    def widened(a: Any, b: Any, a_empty: bool, b_empty: bool) -> str:
        if a_empty != b_empty and str(b if a_empty else a) in _ALIASES:
            return str(b if a_empty else a)
        if all(pa.types.is_integer(x) or pa.types.is_floating(x) for x in (a, b)):
            return "double"
        return "string"

    overrides: dict[str, str] = {}
    for _ in range(64):
        batches: list[Any] = []
        schema: Any = None
        seen: dict[str, tuple[Any, bool]] = {}           # column -> (type, empty in that file's first MiB)
        header: list[str] | None = None
        n = size = read_files = 0
        cut = restart = False
        for path in files:
            own, empty = inferred(path)
            if header is None and own:
                header = list(own)
            elif own and list(own) != header:
                raise ValueError(f"{path.name} has the columns {list(own)[:8]}, {files[0].name} has "
                                 f"{(header or [])[:8]}: the files of one table share their header")
            for k, t in own.items():
                if k in overrides:
                    continue
                if k in seen and seen[k][0] != t and not (seen[k][1] and k in empty):
                    overrides[k] = widened(seen[k][0], t, seen[k][1], k in empty)
                    restart = True
                elif k not in seen or seen[k][1]:
                    seen[k] = (t, k in empty)
            if restart:
                break
            types = {k: (pa.type_for_alias(overrides[k]) if k in overrides else t) for k, t in own.items()}
            try:
                reader = pcsv.open_csv(path, parse_options=parse, read_options=read,
                                       convert_options=pcsv.ConvertOptions(column_types=types, **nulls))
                for b in reader:
                    batches.append(b)
                    n += b.num_rows
                    size += b.nbytes
                    if n >= max_rows or (max_bytes and size >= max_bytes):
                        cut = True
                        break
            except pa.ArrowInvalid as exc:
                m = re.search(r"column #(\d+)", str(exc))
                cols = list(types)
                if not m or int(m.group(1)) >= len(cols) or pa.types.is_string(types[cols[int(m.group(1))]]):
                    raise
                overrides[cols[int(m.group(1))]] = "string"
                restart = True
                break
            read_files += 1
            schema = schema if schema is not None else reader.schema
            if cut:
                break
        if restart:
            continue
        if schema is None:
            return pa.table({}), overrides, False, read_files
        return (pa.Table.from_batches(batches, schema=schema) if batches else schema.empty_table()), overrides, \
            cut, read_files
    raise ValueError(f"{files[0]}: the column types do not settle")


#: Arrow types the csv plugin's ``column_types`` accepts by name (``pyarrow.type_for_alias``).
_ALIASES = frozenset({"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float", "double",
                      "bool", "string", "large_string", "date32[day]"})


#: Data files a table directory may hold, by format (the ``sharded_dir`` layout's file pattern).
_SHARD_PATTERNS = (("parquet", "*.parquet"), ("csv", "*.csv"), ("csv", "*.csv.gz"), ("tsv", "*.tsv"),
                   ("tsv", "*.tsv.gz"), ("tsv", "*.txt"), ("tsv", "*.txt.gz"))
#: The most files a directory import or a directory profile lists (a link loop ends here, as a refusal).
MAX_DIR_FILES = 200_000


def _shards(path: Path) -> tuple[str, str, list[Path]]:
    """``(format, pattern, files)`` of a table directory: the data files the ``sharded_dir`` layout lists for the
    most common data suffix (``_``/``.``-prefixed directories, hidden and partial files and ``_SUCCESS`` skipped)."""
    from ..datalayer.plugins.layouts import is_data_name, walk_files

    rels = []
    for rel, _entry in walk_files(str(path)):
        rels.append(rel)
        if len(rels) > MAX_DIR_FILES:
            raise ValueError(f"{path} holds more than {MAX_DIR_FILES:,} files (or a link loop)")
    best: tuple[str, str, list[str]] | None = None
    for fmt, pattern in _SHARD_PATTERNS:
        hits = sorted(r for r in rels if is_data_name(os.path.basename(r), pattern))
        if hits and (best is None or len(hits) > len(best[2])):
            best = (fmt, pattern, hits)
    if best is None:
        raise ValueError(f"{path} holds no .parquet, .csv or .tsv data files")
    fmt, pattern, hits = best
    files = [path / h for h in hits]
    if pattern.startswith("*.txt"):
        fmt = _delimited_format(files[0])
    return fmt, pattern, files


def _delimited_format(path: Path) -> str:
    """``tsv`` when the header line has more tabs than commas (whatever the suffix: ``.txt`` exports), else ``csv``."""
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:     # type: ignore[operator]
        header = f.readline()
    return "tsv" if header.count("\t") > header.count(",") else "csv"


#: Rows up to which a table without a unique key is tested for exact copies (each row's content hashed).
CONTENT_CHECK_ROWS = 250_000


def _key_of(table: Any, columns: list[dict[str, Any]]) -> tuple[list[str], list[str], str]:
    """``(key columns, the nullable ones, row identity)``: a unique null-free column (an *_id/key/accession/code name
    first), else the first unique combination of two (or, up to 500,000 rows, three) null-free text or integer
    columns, else one that needs columns with nulls (they are the nullable key parts) -- row identity ``key``. Else,
    when no two rows are equal (nested columns included), the combination with the most groups and row identity
    ``content_hash`` (the data layer's identity for a grouping key whose rows differ elsewhere, e.g. Open Targets
    ``drug_mechanism_of_action``); with exact copies ``none``."""
    import itertools

    rows = table.num_rows
    if rows == 0:
        return [], [], "key"
    keyable = [c for c in columns if str(c["type"]).startswith(("string", "large_string", "int", "uint", "dictionary"))
               and _declarable(c["name"])]
    cands = [c for c in keyable if c["nulls"] == 0]
    singles = [c for c in cands if c.get("unique")]
    named = [c for c in singles if re.search(r"(^|_)(id|key|accession|code)$", c["name"].lower())]
    if singles:
        return [(named or singles)[0]["name"]], [], "key"
    sizes = (2, 3) if rows <= 500_000 else (2,)
    best: tuple[int, int, tuple[str, ...]] | None = None
    for pool in (cands, keyable):
        names = [c["name"] for c in sorted(pool, key=lambda c: -(c.get("distinct") or 0))][:12]
        for k in sizes:
            for combo in itertools.combinations(names, k):
                groups = table.select(list(combo)).group_by(list(combo)).aggregate([]).num_rows
                if groups == rows:
                    ordered = sorted(combo, key=table.column_names.index)       # the file's column order
                    return ordered, [n for n in ordered if table.column(n).null_count], "key"
                if best is None or (groups, -k) > (best[0], -best[1]):
                    best = (groups, k, combo)
    if best is None or rows > CONTENT_CHECK_ROWS:
        return [], [], "key"
    ordered = sorted(best[2], key=table.column_names.index)
    nullable = [n for n in ordered if table.column(n).null_count]
    return ordered, nullable, "content_hash" if _rows_distinct(table) else "none"


def _rows_distinct(table: Any) -> bool:
    """No two rows are equal, nested values included (a hash of each row's canonical JSON)."""
    import hashlib

    seen: set[bytes] = set()
    for batch in table.to_batches(max_chunksize=8192):
        for row in batch.to_pylist():
            digest = hashlib.sha1(json.dumps(row, sort_keys=True, default=str).encode("utf-8")).digest()
            if digest in seen:
                return False
            seen.add(digest)
    return True


def _declarable(name: str) -> bool:
    """A descriptor declares a top-level column by its literal name; a ``.`` in it is read as part of the name (HGNC's
    ``pseudogene.org``). A backtick, a bracket or a path prefix (``/``, ``^``, ``@``) cannot be declared (the data
    layer leaves undeclared columns out)."""
    return not re.search(r"[`\[\]/^@]", name) and bool(name.strip())


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


def _read_parquet(files: list[Path], max_rows: int, max_bytes: int) -> tuple[Any, int, bool, int, list[str]]:
    """``(table, rows in every footer, cut, files read, notes)`` of Parquet files read in order up to the bounds.
    Files whose schema differs from the first are named in the notes (their columns are unified where Arrow can)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    total = 0
    first = None
    differ: list[str] = []
    for f in files:
        meta = pq.read_metadata(f)
        total += meta.num_rows
        schema = meta.schema.to_arrow_schema()
        if first is None:
            first = schema
        elif not schema.equals(first):
            differ.append(f.name)
    tables, n, size, read_files = [], 0, 0, 0
    for f in files:
        pf = pq.ParquetFile(f)
        batches = []
        for b in pf.iter_batches(batch_size=65536):
            batches.append(b)
            n += b.num_rows
            size += b.nbytes
            if n >= max_rows or (max_bytes and size >= max_bytes):
                break
        tables.append(pa.Table.from_batches(batches, schema=pf.schema_arrow) if batches
                      else pf.schema_arrow.empty_table())
        read_files += 1
        if n >= max_rows or (max_bytes and size >= max_bytes):
            break
    try:
        table = pa.concat_tables(tables, promote_options="permissive") if len(tables) > 1 else tables[0]
    except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        raise ValueError(f"the files' schemas cannot be unified ({exc}); files that differ from {files[0].name}: "
                         f"{differ[:5]}") from None
    notes = [f"{len(differ)} of {len(files)} files have a schema that differs from {files[0].name}'s (e.g. "
             f"{differ[:3]}): check that they belong to one table"] if differ else []
    return table, total, n < total, read_files, notes


def inspect_dataset(path: Path, *, sample_rows: int = 5, max_rows: int = INSPECT_MAX_ROWS,
                    max_bytes: int = INSPECT_MAX_BYTES) -> dict[str, Any]:
    """Columns, types, nulls, distinct counts and examples of a CSV/TSV/Parquet file or of a directory of such files
    (one table in shards: Spark ``part-*`` output, a release directory), the column(s) that identify a row, and
    (CSV/TSV) the columns whose type the first block of a file does not show. At most ``max_rows`` rows and
    ``max_bytes`` decoded bytes are profiled, file after file (``complete`` says whether that was everything); the
    row count of Parquet files comes from their footers."""
    import pyarrow as pa
    import pyarrow.compute as pc

    pattern = None
    if path.is_dir():
        fmt, pattern, files = _shards(path)
    else:
        suffix = "".join(path.suffixes[-2:]).lower()
        fmt = "parquet" if suffix.endswith(".parquet") or path.suffix == ".pq" else _delimited_format(path)
        files = [path]
    total_rows: int | None = None
    column_types: dict[str, str] = {}
    notes: list[str] = []
    if fmt == "parquet":
        table, total_rows, cut, read_files, notes = _read_parquet(files, max_rows, max_bytes)
    else:
        table, column_types, cut, read_files = _read_text_table(files, fmt, max_rows, max_bytes)
        cut = cut or read_files < len(files)
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
    key, nullable, identity = _key_of(table, columns)
    skipped = [c["name"] for c in columns if not _declarable(c["name"])]
    if skipped:
        notes.append(f"columns {skipped} cannot be declared (a backtick, bracket or path prefix in the name reads as "
                     "path syntax); the draft leaves them out, so the data tools do not serve them")
    if not key:
        notes.append("no column or combination of up to three columns identifies a row: choose the key yourself "
                     "(the data may hold duplicate rows)")
    elif identity == "content_hash":
        notes.append(f"no column or combination of up to three columns is unique, but no two rows are equal: the draft "
                     f"groups rows by {key} (the combination with the most groups) with row_identity: content_hash "
                     "(a row is identified by its whole content); check that this grouping is the table's grain")
    elif identity == "none":
        notes.append(f"some rows are exact copies of others: the draft groups rows by {key} with row_identity: none "
                     "(copies are counted as stored); check whether the copies are an error in the delivery")
    if key and nullable:
        notes.append(f"the key needs columns with empty values {nullable}: they are declared key.nullable; check that "
                     "an empty value is part of the row's identity")
    if pattern is not None and cut:
        notes.append(f"profiled {read_files} of {len(files)} files: the key and types hold for those; the readiness "
                     "check at registration reads every file")
    return {"path": str(path), "format": fmt, "bytes": sum(f.stat().st_size for f in files),
            "layout": "sharded_dir" if pattern is not None else "single_file", "files": len(files),
            "files_profiled": read_files, **({"pattern": pattern} if pattern is not None else {}),
            "rows": total_rows if total_rows is not None else rows, "rows_profiled": rows,
            "complete": not cut, "columns": columns,
            "key": key, "key_nullable": nullable, "key_identity": identity,
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
    """A descriptor draft for one file or a directory of files (to be edited: grain, coverage and roles are guesses
    from the data)."""
    from ..datalayer.plugins.layouts import FORMAT_PATTERNS

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
    files = ""
    if info.get("layout") == "sharded_dir":
        pattern = info.get("pattern")
        doc["defaults"]["layout"] = ("sharded_dir" if pattern == FORMAT_PATTERNS.get(str(info["format"])) else
                                     {"plugin": "sharded_dir", "options": {"pattern": pattern}})
        files = f" in {info.get('files')} files"
    if len(key) == 1:
        doc["id_types"] = {f"{table}_key": {"plugin": "local_key",
                                            "options": {"canonical": info.get("key_pattern") or r"^.+$"},
                                            "universe": f"{table}.{key[0]}"}}
    doc["tables"] = {table: {
        "kind": "entity", "path": path.name,
        "grain": f"one row of {path.name}" + (f", keyed by {', '.join(key)}" if key else ""),
        "key": {"columns": key or ["<choose the column(s) that identify a row>"], "check": "full",
                **({"nullable": list(info["key_nullable"])} if info.get("key_nullable") else {}),
                **({"row_identity": info["key_identity"]} if info.get("key_identity", "key") != "key" else {})},
        "coverage": {"statement": f"The rows of {path.name} as provided ({info['rows']} rows{files}); an absent "
                                  "row is not recorded here.", "absence_means": "unknown"},
        "columns": cols}}
    return yaml.safe_dump(doc, sort_keys=False, width=120, allow_unicode=True)


async def _inspect(ctx: ToolContext, a: dict[str, Any]) -> Any:
    """Profile in the sandbox (``runner.py inspect``): the file is parsed outside the harness process, under the
    workspace memory limit, without network, and read-only under bwrap."""
    from ..projects.model import ProjectSettings
    from ..projects.sandbox import run_sandboxed, runner_argv
    from .policy import PathPolicy

    p = _readable(ctx, str(a.get("path") or ""))
    if not (p.is_file() or p.is_dir()):
        raise ToolFailure(f"no such file or directory: {p}")
    config = dict(getattr(ctx.runtime, "config", None) or {})
    try:
        res = await run_sandboxed(runner_argv(None, "inspect", str(p), str(int(a.get("sample_rows") or 5))),
                                  policy=PathPolicy.from_ctx(ctx), cwd=Path(ctx.workspace), config=config,
                                  label=f"inspect_{ctx.tool_call_id or 'call'}",
                                  timeout_s=ProjectSettings.from_config(config).test_timeout_s, network=False)
    except RuntimeError as exc:                        # projects.sandbox: bwrap without a working bwrap
        raise ToolFailure(f"refused: {exc}") from None
    result = res.result if isinstance(res.result, dict) else {}
    if not res.ok or not result.get("ok"):
        what = "a directory of data files" if p.is_dir() else (p.suffix or "a table")
        why = result.get("error") or ("timed out" if res.timed_out else f"exit code {res.exit_code}")
        notes = "".join(f" [{n}]" for n in res.notes)
        raise ToolFailure(f"{p} could not be read as {what}: {why}{notes}\n{res.output[-2000:]}".rstrip())
    info = result["info"]
    info["sandbox"] = res.sandbox
    source = re.sub(r"[^a-z0-9_]", "_", str(a.get("source") or p.stem).lower()).strip("_") or "dataset"
    if not source[0].isalpha():
        source = "d_" + source
    table = re.sub(r"[^a-z0-9_]", "_", str(a.get("table") or "rows").lower()).strip("_") or "rows"
    info["draft_descriptor"] = draft_descriptor(info, source, table)
    info["next"] = ("Edit the draft (roles, grain, coverage, release), write it to a file in your workspace, then "
                    f"RegisterDataSpec(kind='descriptor', path=<that file>, files=['{p}'], why=...).")
    return info


# ---------------------------------------------------------------------------- registrations


def _import_files(ctx: ToolContext, raws: Sequence[Any]) -> dict[str, Path]:
    """``{path under data/<source>/: file}`` of the files and directories to import: a file keeps its name, a
    directory its name and the data files under it (as the ``sharded_dir`` layout lists them: hidden, partial and
    ``_SUCCESS`` files and ``_``/``.``-prefixed directories left out). Every file must be readable by the agent,
    where its links lead included."""
    from ..datalayer.plugins.layouts import JUNK_NAMES, is_hidden, is_partial, walk_files

    out: dict[str, Path] = {}
    for raw in raws:
        p = _readable(ctx, str(raw))
        if p.is_file():
            entries = [(p.name, p)]
        elif p.is_dir():
            entries = []
            for rel, _entry in walk_files(str(p)):
                name = os.path.basename(rel)
                if is_hidden(name) or is_partial(name) or name in JUNK_NAMES:
                    continue
                full = p / rel
                try:
                    entries.append((f"{p.name}/{rel}", _readable(ctx, str(full))))
                except ToolFailure as exc:
                    real = os.path.realpath(full)
                    raise ToolFailure(f"{exc}" + (f" (a link to {real})" if real != str(full) else "")) from None
                if len(entries) > MAX_DIR_FILES:
                    raise ToolFailure(f"{p} holds more than {MAX_DIR_FILES:,} files (or a link loop): register it "
                                      "in place (a descriptor root you can read) instead of importing it")
            if not entries:
                raise ToolFailure(f"{p} holds no data files to import")
        else:
            raise ToolFailure(f"no such file or directory to import: {p}")
        for rel, full in entries:
            if rel in out:
                raise ToolFailure(f"two imports would both be data/<source>/{rel}")
            out[rel] = full
    return out


@_refusals
async def _register_data_spec(ctx: ToolContext, a: dict[str, Any]) -> Any:
    project = _project(ctx)
    text = _text(ctx, a, "content", "path")
    files = _import_files(ctx, a.get("files") or [])
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
        Tool("InspectDataset", "Profile a CSV, TSV or Parquet file you can read, or a directory of them (one "
                               "table in shards), (columns, types, nulls, distinct values, uniqueness, examples, "
                               "sample rows) and draft a project descriptor for it. "
                               "The draft is a starting point: check the roles, key, grain and coverage before "
                               "registering it with RegisterDataSpec.",
             schema({"path": path, "source": {"type": "string", "description": "source name for the draft"},
                     "table": {"type": "string", "description": "table name for the draft (default rows)"},
                     "sample_rows": {"type": "integer"}}, ["path"]),
             _inspect, source="harness"),
        Tool("RegisterDataSpec", "Register a data source in the active project: kind descriptor (a vbt.datasource/1 "
                                 "file; `files` copies data files or directories into the project's "
                                 "data/<source>/; data you can read may also stay where it is, named by the root), "
                                 "overlay (a "
                                 "vbt.overlay/1 file for a server the core ships none for) or acquisition (the "
                                 "`acquisition:` section of an existing project descriptor `source`). It is linted "
                                 "(vbt ds lint) and checked on the data (vbt ds check) first and refused with the "
                                 "errors otherwise; once registered, the mcp__data__* tools serve its tables.",
             schema({"kind": {"type": "string", "enum": ["descriptor", "overlay", "acquisition"]},
                     "path": path, "content": {"type": "string", "description": "the YAML (instead of path)"},
                     "source": {"type": "string", "description": "acquisition: the project descriptor to extend"},
                     "files": {"type": "array", "items": {"type": "string"},
                               "description": "data files or directories to import into <project>/data/<source>/ "
                                              "(a directory keeps its name; importing it again replaces it)"},
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
