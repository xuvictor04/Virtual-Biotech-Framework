"""The six correctness tests (docs/DATA_LAYER.md §19), generated from the descriptors and overlays of any source.

Nothing here names a table, a tool or an identifier: the cases come from the bindings' roles, the questions the
oracle answers come from the descriptors' roles, and the expected answers from :mod:`.oracle` (pyarrow over the
table's files), never from the code under test. For each bound tool whose table is ready and local:

* **CT-1** identifiers of the wrong form: a real identifier read from the bound column, rewritten into forms the
  column's identifier plugin normalizes back to it (case, version suffix, CURIE separator). Required: answered
  for the real identifier, never "not found".
* **CT-2** unknown identifiers: a well-formed identifier the oracle finds in no row. Required: ``not_found``.
* **CT-3** no false empty: the real identifier. Required: rows, and the header's total equal to the oracle's
  count when the tool's rows are the bound table's rows filtered by that argument alone. For a tool whose rows are
  items of a container (``result.rows_of`` an item table), the identifier is one whose row holds items, and the
  total is checked against the oracle's item count.
* **CT-4** global top-k: tools with a limit and a declared order on a column of the bound table. Required: the
  returned rows' order values equal the oracle's top k.
* **CT-5** every argument honoured or rejected: a value outside a selector's or an order argument's declared
  values, and ``limit=0`` where the binding declares a minimum. Required: ``invalid_argument``.
* **CT-6** unknown never favourable: a threshold filter on a column with nulls, at the oracle's median. Required:
  every returned row carries a known value that passes the threshold.

Each case runs twice through one set of unmodified servers: ``off`` (no gateway) records what upstream answers,
``enforce`` is judged. A typed refusal the host or the release explains (``too_large``, ``not_ready``,
``unsupported_filter``, ``quarantined``) is ``refused``, not a wrong answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = ["Case", "Plan", "table_source", "plan_cases", "oracle_queries", "complete_cases", "judge", "judge_off",
           "rows_of", "REFUSALS"]

#: Typed refusals that are correct answers on a host or release that cannot serve the call.
REFUSALS = frozenset({"too_large", "not_ready", "unsupported_filter", "unsupported_combination", "quarantined",
                      "service_unavailable"})
_SUPPORTED_LAYOUTS = {"sharded_dir", "single_file", "hive"}
_SUPPORTED_FORMATS = {"parquet", "csv", "tsv"}
_INVALID = "__vbt_validate_invalid__"


@dataclass
class Case:
    ct: str                       # CT-1 .. CT-6
    server: str
    tool: str
    args: dict[str, Any]
    expect: str                   # rows | not_found | invalid_argument | topk | threshold
    note: str = ""
    table: str | None = None
    column: str | None = None
    value: Any = None
    oracle: dict[str, Any] = field(default_factory=dict)
    queries: list[str] = field(default_factory=list)       # oracle query ids this case needs
    items: str | None = None                               # rows_of: the container path the rows are items of
    order: dict[str, Any] | None = None                   # CT-4: {column, field, direction, k}
    threshold: dict[str, Any] | None = None               # CT-6: {column, field, op, arg}
    skip: str | None = None

    @property
    def name(self) -> str:
        return f"{self.ct} {self.server}.{self.tool}({', '.join(f'{k}={v!r}' for k, v in self.args.items())})"


@dataclass
class Plan:
    cases: list[Case] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)     # server.tool -> why no case was generated
    queries: list[dict[str, Any]] = field(default_factory=list)


def table_source(catalog: Any, ref: str) -> tuple[dict[str, Any] | None, str | None]:
    """``(oracle source, None)`` for a local table the oracle can read (sharded_dir/single_file/hive layouts,
    parquet/csv/tsv), else ``(None, why)``."""
    try:
        t = catalog.table(ref)
    except Exception as exc:  # noqa: BLE001
        return None, f"unknown table ({exc})"
    if t.is_item_table:
        return None, "an item table (the oracle reads top-level columns of physical tables)"
    if t.physical_spec.matrix is not None:
        return None, "a matrix table"
    layout, fmt = str(t.layout or ""), str(t.format or "")
    if layout not in _SUPPORTED_LAYOUTS or fmt not in _SUPPORTED_FORMATS:
        return None, f"layout {layout or '-'} / format {fmt or '-'} (the oracle reads local parquet and CSV files)"
    desc = catalog.source(t.ref.source) if hasattr(t, "ref") else None
    root = getattr(desc, "root", None)
    path = t.physical_spec.path
    if not root or not path:
        return None, "no local root or path"
    from pathlib import Path

    source = Path(str(root)) / str(path)
    if not source.exists():
        return None, f"{source} is not on this host"
    out: dict[str, Any] = {"source": str(source), "format": fmt,
                           "partitioning": "hive" if layout == "hive" else None}
    opts = getattr(t.physical_spec.format, "options", None) if not isinstance(t.physical_spec.format, str) else None
    if isinstance(opts, Mapping) and opts.get("delimiter"):
        out["delimiter"] = opts["delimiter"]
    return out, None


def _top_level(column: str) -> bool:
    return bool(column) and not any(ch in column for ch in ".[]/")


def _required(schema: Mapping[str, Any]) -> set[str]:
    return set(schema.get("required") or [])


def _props(schema: Mapping[str, Any]) -> dict[str, Any]:
    return dict(schema.get("properties") or {})


def _variants(value: str, plugin: Any) -> list[str]:
    """Wrong forms of ``value`` that the identifier plugin normalizes back to it."""
    cands = [value.lower(), value.upper(), f"{value}.7", value.replace("_", ":", 1), value.replace(":", "_", 1),
             f" {value} "]
    out: list[str] = []
    want = _normalized(plugin, value)
    for v in cands:
        if v == value or v in out:
            continue
        if want is not None and _normalized(plugin, v) == want:
            out.append(v)
    return out[:2]


def _normalized(plugin: Any, raw: str) -> str | None:
    if plugin is None:
        return None
    try:
        res = plugin.normalize(raw)
    except Exception:  # noqa: BLE001
        return None
    return getattr(res, "value", None) if type(res).__name__ == "Normalized" else None


def _absent_candidates(value: str, plugin: Any) -> list[str]:
    """Well-formed identifiers near ``value`` (digits replaced by 9s from the right, a letter swapped): the oracle
    keeps the first that no row holds."""
    out: list[str] = []
    m = re.search(r"(\d+)(\D*)$", value)
    if m:
        digits = m.group(1)
        for width in (len(digits), max(1, len(digits) // 2), 1):
            cand = value[:m.start(1)] + digits[:-width] + "9" * width + m.group(2)
            if cand != value and cand not in out:
                out.append(cand)
            cand = value[:m.start(1)] + digits[:-width] + "8" * width + m.group(2)
            if cand != value and cand not in out:
                out.append(cand)
    if plugin is not None:
        out = [c for c in out if _normalized(plugin, c) is not None] or out
    return out[:4]


def _field_for(binding: Any, column: str) -> str:
    """The output field a result row carries ``column`` under (``result.fields`` maps renamed fields)."""
    for name, fm in (binding.result.fields or {}).items():
        if getattr(fm, "column", None) == column:
            return name
    return column


def plan_cases(catalog: Any, registry: Any, server: str, schemas: Mapping[str, Mapping[str, Any]],
               ready: set[str], *, max_tools: int = 12) -> Plan:
    """The cases of one server (the first-round oracle queries they need, the samples, are in ``Plan.queries``)."""
    plan = Plan()
    samples: dict[tuple[str, str], str] = {}

    def sample_of(source: Mapping[str, Any], column: str, items: str | None) -> str:
        key = (str(source["source"]), column, items or "")
        if key not in samples:
            samples[key] = f"{server}-s{len(samples) + 1}"
            q = {"id": samples[key], "kind": "sample", "column": column, "n": 2, **source}
            if items:
                q["items"] = items
            plan.queries.append(q)
        return samples[key]

    tools = 0
    for tool in catalog.tools(server):
        if tools >= max_tools:
            plan.skipped[f"{server}.{tool}"] = f"over --max-tools {max_tools}"
            continue
        key = f"{server}.{tool}"
        try:
            c = catalog.contract(server, tool)
        except Exception as exc:  # noqa: BLE001
            plan.skipped[key] = f"no contract ({exc})"
            continue
        b = c.binding
        if b is None or c.quarantined or b.serve == "block" or b.hidden:
            plan.skipped[key] = "unbound, blocked, hidden or quarantined"
            continue
        schema = schemas.get(tool)
        if schema is None:
            plan.skipped[key] = "the server does not list it"
            continue
        table = c.bound_table
        if not table or table not in ready:
            plan.skipped[key] = f"bound table {table or '-'} is not ready on this host"
            continue
        source, why = table_source(catalog, table)
        if source is None:
            plan.skipped[key] = f"{table}: {why}"
            continue
        items: str | None = None
        if b.result.rows_of:
            it = c.item_table_of()
            if it is None or str(it.physical) != str(table) or not it.items_path:
                plan.skipped[key] = f"rows are items of {b.result.rows_of}, not of a container of {table}"
                continue
            items = str(it.items_path)
        elif b.result.kind == "record":
            items = _record_rows(catalog, b, table)
        props, required = _props(schema), _required(schema)
        id_arg: str | None = None
        column: str | None = None
        for name, a in c.identifier_args.items():
            cols = [col for tbl, col in c.arg_columns(name) if tbl == table]
            if name in props and cols and _top_level(cols[0]) and a.op in (None, "eq", "in") and not a.gateway_only:
                id_arg, column = name, cols[0]
                break
        others = required - ({id_arg} if id_arg else set())
        if others:
            plan.skipped[key] = f"required arguments the roles cannot fill: {sorted(others)}"
            continue
        base = {"server": server, "tool": tool, "table": table, "column": column, "items": items}
        found: list[Case] = []
        sample = [sample_of(source, column, items)] if id_arg and column else []
        fill: dict[str, Any] = {id_arg: None} if id_arg else {}
        limit = c.limit_arg if c.limit_arg in props else None
        if id_arg is not None:
            plugin = _id_plugin(catalog, registry, table, str(column), c.args[id_arg])
            found.append(Case("CT-3", args=dict(fill), expect="rows", queries=sample, note=f"a value of {table}.{column}",
                              **base))
            found.append(Case("CT-1", args=dict(fill), expect="rows", queries=sample, oracle={"plugin": plugin},
                              note="wrong forms the identifier plugin normalizes", **base))
            found.append(Case("CT-2", args=dict(fill), expect="not_found", queries=sample, oracle={"plugin": plugin},
                              note="a well-formed identifier no row holds", **base))
            order = list(b.result.order or [])
            if limit and order and _top_level(order[0].column) and order[0].direction in ("asc", "desc") \
                    and not b.result.rows_of:
                first = order[0]
                found.append(Case("CT-4", args={**fill, limit: 3}, expect="topk", queries=sample, **base,
                                  order={"column": first.column, "field": _field_for(b, first.column),
                                         "direction": first.direction, "nulls": first.nulls, "k": 3}))
            for name, a in c.args.items():
                if name not in props or name == id_arg or a.role != "filter" or a.op not in ("ge", "gt", "le", "lt"):
                    continue
                cols = [col for tbl, col in c.arg_columns(name) if tbl == table]
                if cols and _top_level(cols[0]):
                    # the threshold alone where the identifier is optional (the oracle's median is the column's):
                    # an identifier lookup with a threshold is often a combination the binding declines
                    ct6 = {name: None} if id_arg not in required else {**fill, name: None}
                    found.append(Case("CT-6", args=ct6, expect="threshold", queries=sample, **base,
                                      threshold={"column": cols[0], "field": _field_for(b, cols[0]), "op": a.op,
                                                 "arg": name}))
                    break
        for name, a in c.args.items():
            if name not in props:
                continue
            if a.role in ("selector", "order_by") and a.values:
                found.append(Case("CT-5", args={**fill, name: _INVALID}, expect="invalid_argument", queries=sample,
                                  note=f"{name} outside its declared values", **base))
            elif a.role == "limit" and a.min is not None and a.min >= 1:
                found.append(Case("CT-5", args={**fill, name: 0}, expect="invalid_argument", queries=sample,
                                  note=f"{name}=0 under the declared minimum {a.min:g}", **base))
        if not found:
            plan.skipped[key] = "no argument the roles can generate a case for"
            continue
        plan.cases.extend(found)
        tools += 1
    return plan


def _record_rows(catalog: Any, binding: Any, table: str) -> str | None:
    """A record tool whose rows are one field of the record (``rows: $.f``) answers rows only when the column that
    field carries is not null: that column, else None (the rows are the record)."""
    paths = [p for p in binding.result.row_paths if p not in ("$", "")]
    if len(paths) != 1 or not re.fullmatch(r"\$\.\w+", paths[0]):
        return None
    name = paths[0][2:]
    mapped = (binding.result.fields or {}).get(name)
    column = getattr(mapped, "column", None) or name
    try:
        columns = catalog.table(table).columns
    except Exception:  # noqa: BLE001
        return None
    return column if _top_level(column) and column in columns else None


def _id_plugin(catalog: Any, registry: Any, table: str, column: str, arg: Any) -> Any:
    """The identifier plugin of the bound column (its ``id_type``), else of the argument's first accepted type."""
    try:
        t = catalog.table(table)
        col = t.columns.get(column)
        id_type = getattr(col, "id_type", None) or (arg.accepts[0] if arg.accepts else None)
        if not id_type or registry is None:
            return None
        _q, spec = catalog.id_type(str(id_type), t.ref.source)
        plugin = registry.find("identifier", spec.plugin)
        if plugin is not None and hasattr(plugin, "configure"):
            plugin = plugin.configure(dict(spec.options or {}), None)
        return plugin
    except Exception:  # noqa: BLE001 - no plugin: the variants are not generated
        return None


def oracle_queries(plan: Plan, answers: Mapping[str, Any], catalog: Any) -> list[dict[str, Any]]:
    """Second-round oracle queries once the samples are known: the counts (CT-3, CT-2 candidates), the top k (CT-4)
    and the null counts and median (CT-6)."""
    out: list[dict[str, Any]] = []
    n = 0
    for case in plan.cases:
        sample = (answers.get(case.queries[0]) or {}) if case.queries else {}
        values = sample.get("values") or []
        if not values or not case.table or not case.column:
            continue
        src, _ = table_source(catalog, case.table)
        if src is None:
            continue
        value = values[0]
        n += 1
        if case.ct == "CT-3":
            q = {"id": f"c{n}", "kind": "count", "column": case.column, "values": [value], **src}
            if case.items:
                q["items"] = case.items
            out.append(q)
            case.oracle["count_id"] = f"c{n}"
        elif case.ct == "CT-2":
            cands = _absent_candidates(str(value), case.oracle.get("plugin"))
            case.oracle["candidates"] = cands
            out.append({"id": f"c{n}", "kind": "count", "column": case.column, "values": cands, **src})
            case.oracle["count_id"] = f"c{n}"
        elif case.ct == "CT-4" and case.order:
            out.append({"id": f"c{n}", "kind": "topk", "filter": {"column": case.column, "value": value},
                        "order": [{"column": case.order["column"], "direction": case.order["direction"],
                                   "nulls": case.order.get("nulls") or "last"}], "k": case.order["k"], **src})
            case.oracle["topk_id"] = f"c{n}"
        elif case.ct == "CT-6" and case.threshold:
            out.append({"id": f"c{n}", "kind": "nulls", "column": case.threshold["column"], **src})
            case.oracle["nulls_id"] = f"c{n}"
    return out


def complete_cases(plan: Plan, answers: Mapping[str, Any]) -> None:
    """Fill each case's arguments and expected answer from the oracle; a case the oracle cannot back is skipped."""
    for case in plan.cases:
        sample = (answers.get(case.queries[0]) or {}) if case.queries else {}
        if sample.get("error"):
            case.skip = f"oracle: {sample['error']}"
            continue
        values = sample.get("values") or []
        if any(v is None for v in case.args.values()) and not values:
            case.skip = "the oracle found no value to ask for"
            continue
        value = values[0] if values else None
        case.value = value
        id_args = [k for k, v in case.args.items() if v is None and (case.threshold is None or k != case.threshold["arg"])]
        for k in id_args:
            case.args[k] = value
        if case.items:
            case.oracle["sample_items"] = (sample.get("items") or {}).get(str(value))
        if case.ct == "CT-3":
            got = answers.get(case.oracle.get("count_id", "")) or {}
            case.oracle["count"] = (got.get("counts") or {}).get(str(value))
            if case.items:
                case.oracle["items"] = (got.get("items") or {}).get(str(value))
        elif case.ct == "CT-1":
            variants = _variants(str(value), case.oracle.get("plugin"))
            if not variants:
                case.skip = "the identifier plugin normalizes no wrong form back to this value"
                continue
            case.args = {k: (variants[0] if k in id_args else v) for k, v in case.args.items()}
            case.oracle["canonical"] = value
            case.oracle["variants"] = variants
        elif case.ct == "CT-2":
            got = answers.get(case.oracle.get("count_id", "")) or {}
            absent = [c for c in case.oracle.get("candidates") or [] if (got.get("counts") or {}).get(c) == 0]
            if not absent:
                case.skip = "no well-formed absent identifier found"
                continue
            case.args = {k: (absent[0] if k in id_args else v) for k, v in case.args.items()}
            case.oracle["absent"] = absent[0]
        elif case.ct == "CT-4":
            got = answers.get(case.oracle.get("topk_id", "")) or {}
            if got.get("error"):
                case.skip = f"oracle: {got['error']}"
                continue
            case.oracle["top"] = got.get("top")
            case.oracle["total"] = got.get("total")
        elif case.ct == "CT-6" and case.threshold:
            got = answers.get(case.oracle.get("nulls_id", "")) or {}
            if got.get("error") or got.get("median") is None:
                case.skip = "no numeric values to set a threshold at"
                continue
            if not got.get("nulls"):
                case.skip = "the column has no nulls: an unknown value cannot be tested here"
                continue
            case.args[case.threshold["arg"]] = got["median"]
            case.oracle.update({"median": got["median"], "nulls": got["nulls"]})
        case.oracle.pop("plugin", None)


def rows_of(obj: Any, path: str | Sequence[str] | None) -> list[Any] | None:
    """The result rows at a ``$``/``$.a.b`` path (the binding's ``result.rows``), None when they are not a list."""
    paths = [path] if isinstance(path, str) or path is None else list(path)
    for p in paths:
        cur = obj
        for part in [x for x in str(p or "$").lstrip("$").split(".") if x]:
            cur = cur.get(part) if isinstance(cur, dict) else None
        if isinstance(cur, list):
            return cur
    return None


def _count_rows(case: Case, out: Any, binding_rows: Any) -> int | None:
    rows = rows_of(out.obj, binding_rows)
    return len(rows) if rows is not None else None


def judge(case: Case, out: Any, binding: Any) -> tuple[str, str]:
    """``(verdict, detail)`` of the enforce answer: correct | refused | wrong | skipped."""
    kind = out.kind if out.is_error else None
    if out.is_error and kind in REFUSALS:
        return "refused", f"{kind}: {_short(out)}"
    if case.expect == "not_found":
        if out.is_error and kind == "not_found":
            return "correct", "not_found"
        return "wrong", f"expected not_found, got {kind or out.status}: {_short(out)}"
    if case.expect == "invalid_argument":
        if out.is_error and kind == "invalid_argument":
            return "correct", "invalid_argument"
        return "wrong", f"expected invalid_argument, got {kind or out.status}: {_short(out)}"
    if out.is_error:
        return "wrong", f"{kind}: {_short(out)}"
    rows_path = binding.result.rows if binding is not None else "$"
    if case.expect == "rows":
        what = "items" if case.items else "rows"
        want = case.oracle.get("items" if case.items else "count")
        have = want if case.ct == "CT-3" else case.oracle.get("sample_items" if case.items else "count")
        if out.status == "empty":
            return "wrong", f"empty where the oracle has {what} ({have}) for {case.value!r}"
        total = out.header.get("total")
        if case.ct == "CT-3" and isinstance(total, int) and isinstance(want, int) and _plain_rows(binding, case):
            if total != want:
                return "wrong", f"total {total}, oracle {want} {what}"
            return "correct", f"total {total} = oracle {what}"
        if case.ct == "CT-1":
            canon = str(case.oracle.get("canonical"))
            if canon not in out.text and canon not in str(out.header):
                return "wrong", f"the answer does not name {canon}"
            return "correct", f"resolved to {canon}"
        return "correct", f"{out.status}" + (f", total {total}" if total is not None else "")
    if case.expect == "topk" and case.order:
        rows = rows_of(out.obj, rows_path) or []
        got = [r.get(case.order["field"]) for r in rows if isinstance(r, dict)]
        want = list(case.oracle.get("top") or [])[:len(got) or case.order["k"]]
        if _same(got, want):
            return "correct", f"top {len(got)} = oracle {want}"
        return "wrong", f"returned {got}, oracle top {want}"
    if case.expect == "threshold" and case.threshold:
        rows = rows_of(out.obj, rows_path) or []
        bad = [r.get(case.threshold["field"]) for r in rows if isinstance(r, dict)
               and not _passes(r.get(case.threshold["field"]), case.threshold["op"], case.oracle.get("median"))]
        if bad:
            return "wrong", f"{len(bad)} returned row(s) fail {case.threshold['op']} {case.oracle.get('median')}: " \
                            f"{bad[:3]}"
        return "correct", f"{len(rows)} row(s), every value known and passing"
    return "correct", str(out.status)


def judge_off(case: Case, out: Any, binding: Any) -> str:
    """What the unmodified server answered, in words (off mode is recorded, not judged)."""
    if out.is_error:
        return f"error {out.kind or ''}: {_short(out, 120)}".strip()
    n = _count_rows(case, out, binding.result.rows if binding is not None else "$")
    text = _short(out, 120)
    if case.expect == "not_found":
        return f"success ({n if n is not None else '?'} rows): {text}"
    if case.expect == "invalid_argument":
        return f"accepted ({n if n is not None else '?'} rows)"
    if case.expect == "topk" and case.order:
        rows = rows_of(out.obj, binding.result.rows if binding is not None else "$") or []
        got = [r.get(case.order["field"]) for r in rows if isinstance(r, dict)]
        want = list(case.oracle.get("top") or [])[:len(got) or case.order["k"]]
        return f"top {got}" + (" (= oracle)" if _same(got, want) else
                               " (differs from the oracle)")
    if case.expect == "threshold" and case.threshold:
        rows = rows_of(out.obj, binding.result.rows if binding is not None else "$") or []
        bad = sum(1 for r in rows if isinstance(r, dict) and not _passes(r.get(case.threshold["field"]),
                                                                         case.threshold["op"], case.oracle.get("median")))
        return f"{len(rows)} row(s), {bad} with an unknown or failing value"
    return f"{n if n is not None else '?'} rows: {text}"


def _plain_rows(binding: Any, case: Case) -> bool:
    """The tool's rows are the bound table's rows filtered by the identifier alone (so the oracle's count is its
    total): no item rows, no derived view, no other argument sent."""
    if binding is None or binding.derived is not None or binding.result.kind != "rows":
        return False
    if binding.result.rows_of and not case.items:
        return False
    return len(case.args) == 1


def _passes(value: Any, op: str, threshold: Any) -> bool:
    if value is None or threshold is None or (isinstance(value, float) and value != value):
        return False
    try:
        v, t = float(value), float(threshold)
    except (TypeError, ValueError):
        return False
    return {"ge": v >= t, "gt": v > t, "le": v <= t, "lt": v < t}.get(op, True)


def _same(got: Sequence[Any], want: Sequence[Any]) -> bool:
    if len(got) != len(want):
        return False
    for a, b in zip(got, want):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if abs(float(a) - float(b)) > 1e-9 * max(1.0, abs(float(b))):
                return False
        elif a != b:
            return False
    return True


def _short(out: Any, n: int = 200) -> str:
    text = out.text if isinstance(out.text, str) else str(out.obj)
    return " ".join(text.split())[:n]
