"""Result field maps: JSONPath extraction, descriptor-column views of rows, parent keys and
computed fields (§8.1 ``ResultSpec.fields``, §11.4 step 2). No pyarrow.

Upstream rows keep their own field names. :class:`FieldMapper` gives each extracted row a
**logical view**: the same dict (shallow copy) with the descriptor columns the overlay maps
(``{"entity_id": {column: word}}`` adds ``word``; a dotted column such as
``protocolSection.statusModule.startDateStruct.date`` is added as nested dicts), the
``parent_key`` values read from the payload and the ``key_from_args`` values of the call.
Predicates, ranks and row keys are evaluated on the logical view; :meth:`FieldMapper.to_output`
removes the added names and writes changed values back to the upstream field, so the agent
sees upstream's shape.

The JSONPath subset is ``$``, ``.name``, ``['name']``, ``[*]`` and ``[N]``. Path predicates
(``not_found_when``, ``exists_when``, ``nested_errors``, ``partial_when``) are
``<path> == <literal>``, ``<path> != <literal>``, ``<path> =~ '<regex>'`` or a bare ``<path>``
(present and not empty).
"""

from __future__ import annotations

import copy
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "WILDCARD", "parse_jsonpath", "jp_get", "jp_first", "jp_set", "jp_test", "parse_payload", "extract_rows",
    "place_rows", "FieldMapper", "cosine", "recount", "set_path", "get_path", "column_name",
]

WILDCARD = "*"

_TOKEN = re.compile(r"\.([^.\[\]]+)|\[\s*'([^']*)'\s*\]|\[\s*\"([^\"]*)\"\s*\]|\[\s*(\*|-?\d+)\s*\]")


def parse_jsonpath(path: str) -> list[str | int]:
    """``$.a[*].b`` -> ``["a", "*", "b"]``; ``$`` -> ``[]``. A leading ``$`` is optional."""
    text = str(path).strip()
    if text.startswith("$"):
        text = text[1:]
    elif text and not text.startswith((".", "[")):
        text = "." + text
    out: list[str | int] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None:
            raise ValueError(f"unsupported JSONPath {path!r} at {text[pos:]!r}")
        name, q1, q2, idx = m.groups()
        if name is not None:
            out.append(WILDCARD if name == "*" else name)
        elif q1 is not None or q2 is not None:
            out.append(q1 if q1 is not None else q2)  # type: ignore[arg-type]
        else:
            out.append(WILDCARD if idx == "*" else int(idx))
        pos = m.end()
    return out


def _step(values: list[Any], part: str | int) -> list[Any]:
    out: list[Any] = []
    for v in values:
        if part == WILDCARD:
            if isinstance(v, list):
                out.extend(v)
            elif isinstance(v, Mapping):
                out.extend(v.values())
        elif isinstance(part, int):
            if isinstance(v, list) and -len(v) <= part < len(v):
                out.append(v[part])
        elif isinstance(v, Mapping) and part in v:
            out.append(v[part])
    return out


def jp_get(obj: Any, path: str) -> list[Any]:
    """Every value at ``path`` (wildcards expand); ``[]`` when nothing is there."""
    values = [obj]
    for part in parse_jsonpath(path):
        values = _step(values, part)
        if not values:
            return []
    return values


def jp_first(obj: Any, path: str, default: Any = None) -> Any:
    found = jp_get(obj, path)
    return found[0] if found else default


def jp_set(obj: Any, path: str, value: Any) -> Any:
    """Set ``path`` (no wildcards) to ``value``, creating dicts; returns the (possibly new) root."""
    parts = parse_jsonpath(path)
    if not parts:
        return value
    cur = obj
    for i, part in enumerate(parts[:-1]):
        nxt = parts[i + 1]
        if isinstance(part, int):
            cur = cur[part]
            continue
        if part == WILDCARD:
            raise ValueError(f"cannot set a wildcard path {path!r}")
        if not isinstance(cur.get(part), (dict, list)):
            cur[part] = [] if isinstance(nxt, int) else {}
        cur = cur[part]
    last = parts[-1]
    if last == WILDCARD:
        raise ValueError(f"cannot set a wildcard path {path!r}")
    cur[last] = value  # type: ignore[index]
    return obj


_LITERAL_RE = re.compile(r"^\s*(?:'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"|(.+?))\s*$")
_EXPR_RE = re.compile(r"^\s*(\$[^=!~\s]*)\s*(==|!=|=~)\s*(.+?)\s*$")


def _literal(text: str) -> Any:
    m = _LITERAL_RE.match(text)
    if m is None:
        return text
    if m.group(1) is not None or m.group(2) is not None:
        raw = m.group(1) if m.group(1) is not None else m.group(2)
        return re.sub(r"\\(.)", r"\1", raw)
    word = (m.group(3) or "").strip()
    if word == "null":
        return None
    if word in ("true", "false"):
        return word == "true"
    try:
        return json.loads(word)
    except ValueError:
        return word


def _present(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, (str, list, dict)) and not v:
        return False
    return True


def jp_test(obj: Any, expr: str) -> bool:
    """Evaluate a path predicate (see the module docstring). A comparison holds when **any**
    value at the path satisfies it; ``!= null`` on a missing path is false."""
    m = _EXPR_RE.match(str(expr))
    if m is None:
        return any(_present(v) for v in jp_get(obj, str(expr).strip()))
    path, op, lit = m.groups()
    values = jp_get(obj, path)
    if op == "=~":
        pattern = _literal(lit)
        rx = re.compile(str(pattern))
        return any(v is not None and rx.search(v if isinstance(v, str) else json.dumps(v, default=str))
                   for v in values)
    want = _literal(lit)
    if op == "==":
        return any(_same(v, want) for v in values)
    return bool(values) and any(not _same(v, want) for v in values)


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or a == b and type(a) is type(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, str) and isinstance(b, (int, float)):
        return a == str(b)
    return a == b


def parse_payload(text: str | None, structured: Any = None) -> tuple[Any, bool]:
    """``(object, is_json)``: the structured content when present, else the text parsed as JSON
    (non-JSON text stays text)."""
    if structured is not None:
        # FastMCP wraps non-object returns as {"result": ...}
        if isinstance(structured, Mapping) and set(structured) == {"result"} and text:
            try:
                return json.loads(text), True
            except ValueError:
                pass
        return copy.deepcopy(structured), True
    if text is None:
        return None, False
    try:
        return json.loads(text), True
    except ValueError:
        return text, False


def extract_rows(obj: Any, paths: Sequence[str]) -> list[tuple[str, list[Any]]]:
    """``[(path, rows)]`` for each row path: a list at the path is the rows; a record (``$`` on a
    dict) is one row; a missing path gives no rows."""
    out: list[tuple[str, list[Any]]] = []
    for path in paths:
        found = jp_get(obj, path)
        if not found:
            out.append((path, []))
            continue
        value = found[0]
        if isinstance(value, list):
            out.append((path, value))
        elif isinstance(value, Mapping):
            out.append((path, [value]))
        else:
            out.append((path, []))
    return out


def place_rows(obj: Any, path: str, rows: list[Any], *, record: bool = False) -> Any:
    """Write ``rows`` back at ``path``; a record path gets its single row (or None)."""
    value: Any = (rows[0] if rows else None) if record else rows
    if not parse_jsonpath(path):
        return value if record else rows
    return jp_set(obj, path, value)


# ---------------------------------------------------------------------------
# Column paths inside one row
# ---------------------------------------------------------------------------

def column_name(column: str) -> str:
    """A descriptor column path without a leading ``/`` (table-level form)."""
    return column[1:] if column.startswith("/") else column


def _plain_parts(column: str) -> list[str] | None:
    """The names of a dotted path without lists, or None when it crosses a list."""
    col = column_name(column)
    if "[" in col or col.startswith("@") or col.startswith("^"):
        return None
    return [p for p in col.split(".") if p]


def get_path(row: Any, column: str) -> Any:
    parts = _plain_parts(column)
    if parts is None:
        return row.get(column) if isinstance(row, Mapping) else None
    cur = row
    for p in parts:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(p)
    return cur


def set_path(row: dict[str, Any], column: str, value: Any) -> bool:
    """Set a dotted path (creating dicts); False when the path crosses a list."""
    parts = _plain_parts(column)
    if not parts:
        return False
    cur = row
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {} if nxt is None else None
            if nxt is None:
                return False
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value
    return True


def cosine(a: Sequence[float] | None, b: Sequence[float] | None) -> float | None:
    """Cosine similarity; None for missing, mismatched or zero-norm vectors (T14)."""
    if not a or not b or len(a) != len(b):
        return None
    try:
        dot = sum(float(x) * float(y) for x, y in zip(a, b))
        na = math.sqrt(sum(float(x) ** 2 for x in a))
        nb = math.sqrt(sum(float(y) ** 2 for y in b))
    except (TypeError, ValueError):
        return None
    if na == 0 or nb == 0 or not math.isfinite(dot):
        return None
    out = dot / (na * nb)
    return out if math.isfinite(out) else None


# ---------------------------------------------------------------------------
# The field mapper
# ---------------------------------------------------------------------------

class FieldMapper:
    """Logical views of result rows (see the module docstring).

    ``fields`` is ``ResultSpec.fields`` (``{result field: FieldMap}``); ``parent_key`` maps key
    columns to payload JSONPaths; ``key_from_args`` maps key columns to argument names."""

    def __init__(self, fields: Mapping[str, Any] | None = None, *, parent_key: Mapping[str, str] | None = None,
                 key_from_args: Mapping[str, str] | None = None) -> None:
        self.fields = dict(fields or {})
        self.parent_key = dict(parent_key or {})
        self.key_from_args = dict(key_from_args or {})
        self._created: dict[int, set[str]] = {}
        self._mapped: dict[int, dict[str, str]] = {}
        self.counters: dict[str, dict[str, int]] = {"placeholders": {}, "dangling": {}, "computed": {}}
        self.unmapped: list[str] = []                  # fields whose column path cannot be placed on a row

    def column_of(self, field: str) -> str | None:
        fm = self.fields.get(field)
        if fm is None:
            return None
        return column_name(fm.column) if fm.column else field

    def field_of(self, column: str) -> str | None:
        col = column_name(column)
        for field, fm in self.fields.items():
            if (column_name(fm.column) if fm.column else field) == col:
                return field
        return None

    def to_logical(self, row: Any, *, payload: Any = None, args: Mapping[str, Any] | None = None,
                   anchor_vector: Sequence[float] | None = None) -> Any:
        """The logical view of one row (non-dict rows are returned unchanged)."""
        if not isinstance(row, Mapping):
            return row
        out = dict(row)
        created: set[str] = set()
        mapped: dict[str, str] = {}

        def add(column: str, value: Any) -> None:
            top = column_name(column).split(".")[0].split("[")[0]
            if top not in row:
                created.add(top)
            if not set_path(out, column, value) and column not in out:
                self.unmapped.append(column)

        for field, fm in self.fields.items():
            if fm.computed is not None:
                value = row.get(field)
                spec = fm.computed
                src = spec.get("from")
                if value is None and anchor_vector is not None and src:
                    value = cosine(get_path(row, str(src)), anchor_vector)
                    if value is not None:
                        self.counters["computed"][field] = self.counters["computed"].get(field, 0) + 1
                if value is not None and isinstance(value, float) and not math.isfinite(value):
                    value = None
                out[field] = value
                continue
            col = column_name(fm.column)
            value = row.get(field)
            if fm.placeholders and any(_same(value, p) for p in fm.placeholders):
                bucket = "dangling" if fm.on_placeholder == "dangling_ref" else "placeholders"
                self.counters[bucket][field] = self.counters[bucket].get(field, 0) + 1
                value = None
                if col == field:
                    out[field] = None
            if col != field:
                add(col, value)
                mapped[col] = field
            else:
                out[field] = value
        for col, path in self.parent_key.items():
            if get_path(out, col) is None:
                add(col, jp_first(payload, path))
        for col, arg in self.key_from_args.items():
            if get_path(out, col) is None and args is not None:
                add(col, args.get(arg))
        self._created[id(out)] = created
        self._mapped[id(out)] = mapped
        return out

    def to_output(self, row: Any) -> Any:
        """The upstream-shaped row: added names removed, changed mapped values written back."""
        if not isinstance(row, Mapping):
            return row
        rid = id(row)
        created = self._created.get(rid, set())
        mapped = self._mapped.get(rid, {})
        out = dict(row)
        for col, field in mapped.items():
            if field in out or field in self.fields:
                out[field] = get_path(row, col)
        for name in created:
            out.pop(name, None)
        return out

    def logical_rows(self, rows: Iterable[Any], **kw: Any) -> list[Any]:
        return [self.to_logical(r, **kw) for r in rows]

    def output_rows(self, rows: Iterable[Any]) -> list[Any]:
        return [self.to_output(r) for r in rows]


# ---------------------------------------------------------------------------
# Recounting (T11 count_fields)
# ---------------------------------------------------------------------------

_LEN_RE = re.compile(r"^\s*len\(\s*(\$[^)]*)\s*\)\s*$")


def recount(obj: Any, count_fields: Sequence[str] | Mapping[str, str], returned: int) -> dict[str, Any]:
    """Set ``count_fields`` in ``obj``: the list form gets ``returned``; the dict form maps a
    target path to ``len(<path>)``, evaluated per element when both share a ``[*]`` prefix.
    Returns ``{path: new value}`` for the fields that changed."""
    changed: dict[str, Any] = {}
    if isinstance(count_fields, Mapping):
        for target, expr in count_fields.items():
            m = _LEN_RE.match(str(expr))
            if m is None:
                continue
            source = m.group(1)
            t_parts, s_parts = parse_jsonpath(target), parse_jsonpath(source)
            if WILDCARD in t_parts:
                i = t_parts.index(WILDCARD)
                if s_parts[:i + 1] != t_parts[:i + 1]:
                    continue
                prefix = "$" + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in t_parts[:i])
                elements = jp_first(obj, prefix)
                if not isinstance(elements, list):
                    continue
                for k, el in enumerate(elements):
                    if not isinstance(el, dict):
                        continue
                    t_rest = "$" + "".join(f".{p}" for p in t_parts[i + 1:])
                    s_rest = "$" + "".join(f".{p}" for p in s_parts[i + 1:])
                    src = jp_first(el, s_rest)
                    n = len(src) if isinstance(src, (list, dict)) else 0
                    if jp_first(el, t_rest) != n:
                        jp_set(el, t_rest, n)
                        changed[f"{prefix}[{k}]{t_rest[1:]}"] = n
            else:
                src = jp_first(obj, source)
                n = len(src) if isinstance(src, (list, dict)) else 0
                if jp_get(obj, target) and jp_first(obj, target) != n:
                    jp_set(obj, target, n)
                    changed[target] = n
        return changed
    for path in count_fields:
        if jp_get(obj, path) and jp_first(obj, path) != returned:
            jp_set(obj, path, returned)
            changed[path] = returned
    return changed
