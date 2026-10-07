"""The role vocabulary (§7), Arrow type compatibility (R4) and the path grammar (§6.4). No pyarrow.

Every column, nested field and matrix axis field has exactly one :class:`Role`; facets
refine it. :data:`ROLE_FACETS` lists the role-specific facets and :data:`COMMON_FACETS`
the facets valid on every role except ``payload`` and ``ignore``;
``descriptor/columns.py`` defines one model per role with exactly these facets.

Path grammar (normative, §6.4)::

    path      ::= [ "@" axis "." ] segment { "." segment }
    segment   ::= name { "[]" | "[" cond "]" }
    cond      ::= name "=" literal                    (item predicate: dbXrefs[source=NCBI_Gene].id)
    axis      ::= "row" | "col" | "obs" | "var"       (obs = row, var = col; matrix tables only)
    name      ::= identifier | "`" any-char-but-backtick "`"
    literal   ::= quoted string | number | "true" | "false" | "null"

plus the reference-scoping prefixes ``^.`` (parent level, repeatable) and ``/`` (table
level), an empty first segment for item-relative paths (``[].label``, ``[]``), and bare
words as string literals in item predicates (``[source=NCBI_Gene]``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any

__all__ = [
    "Role", "ROLE_FACETS", "COMMON_FACETS", "EXISTENCE_FACETS", "KEYABLE_ROLES", "CONTAINER_ROLES",
    "MISSING_KINDS", "facets_for", "ARROW_COMPAT", "type_family", "arrow_compatible",
    "LIST", "ItemCond", "Segment", "Path", "PathError", "parse_path", "format_path", "format_name",
    "AXES",
]


class Role(str, Enum):
    identifier = "identifier"
    label = "label"
    synonym = "synonym"
    category = "category"
    scope = "scope"
    qualifier = "qualifier"
    measure = "measure"
    count = "count"  # type: ignore[assignment]  # the role name shadows str.count on members
    flag = "flag"
    time = "time"
    position = "position"
    hierarchy = "hierarchy"
    member = "member"
    endpoint = "endpoint"
    vector = "vector"
    text = "text"
    reference = "reference"
    nested = "nested"
    payload = "payload"
    ignore = "ignore"

    def __str__(self) -> str:
        return self.value


_CATEGORY = frozenset({"vocab", "match", "aliases", "aliases_from", "values", "hierarchy_via", "negative_values",
                       "projection_of", "lossy"})
_CONTAINER = frozenset({"fields", "item_key", "null_means", "empty_means", "null_items", "coverage", "rank",
                        "grain"})

#: Role-specific facets (§7 table). Two additions the §6.5-§6.8 examples require: ``authority``
#: on identifiers (a release-snapshot key column) and, on ``member`` containers, the nested
#: container facets plus ``ref`` (member keys checked against their universe, R9).
ROLE_FACETS: dict[Role, frozenset[str]] = {
    Role.identifier: frozenset({"id_type", "self", "ref", "maps_to", "cardinality", "form", "alternate_key",
                                "retired_into", "xref", "id_type_from", "kind_from", "resolvable", "authority"}),
    Role.label: frozenset({"of", "id_type", "unique", "authority"}),
    Role.synonym: frozenset({"of", "synonym_kind", "synonym_kind_from"}),
    Role.category: _CATEGORY,
    Role.scope: _CATEGORY | {"unit_from", "statistic"},
    Role.qualifier: frozenset({"effect", "default_filter"}),
    Role.measure: frozenset({"statistic", "fallback", "scale", "unit", "unit_from", "direction", "encoding", "cutoff",
                             "comparable_within", "significant_above", "of", "part_of", "columns", "family",
                             "undefined_when", "levels"}),
    Role.count: frozenset({"counts", "distinct_by", "length_of", "comparable_within"}),
    Role.flag: frozenset({"true_means", "encoding", "event_of", "partition_items"}),
    Role.time: frozenset({"precision", "unit", "as_of", "fallback", "partial_dates"}),
    Role.position: frozenset({"part", "build", "chrom_style", "coordinate_base"}),
    Role.hierarchy: frozenset({"relation", "of", "closure", "reflexive", "closure_of", "inverse_of", "predicate",
                               "target_id_type"}),
    Role.member: frozenset({"membership", "propagated_over", "ref"}) | _CONTAINER,
    Role.endpoint: frozenset({"side", "directed", "id_type", "ref"}),
    Role.vector: frozenset({"dim", "metric", "normalized", "norm_column", "element"}),
    Role.text: frozenset({"searchable"}),
    Role.reference: frozenset({"ref_kinds"}),
    Role.nested: _CONTAINER,
    Role.payload: frozenset(),
    Role.ignore: frozenset({"reason"}),
}

#: Common facets (rev 2), valid on every role unless the column is ``payload`` or ``ignore``.
COMMON_FACETS: frozenset[str] = frozenset({
    "path", "missing", "missing_values", "unknown_when", "placeholders", "placeholder_when", "optional", "present_in",
    "applies_when", "by_partition", "verified", "verified_by", "equals", "level", "scope", "unique_within", "side",
    "list_delimiter", "parse", "stored_as", "integrity", "remote_name", "description",
})

#: The only common facets ``payload`` and ``ignore`` columns take: release- or fragment-dependent
#: existence (R4 warns instead of failing) and the one-sentence description.
EXISTENCE_FACETS: frozenset[str] = frozenset({"optional", "present_in", "description"})

KEYABLE_ROLES: frozenset[Role] = frozenset(set(Role) - {Role.text, Role.payload, Role.vector, Role.ignore})
CONTAINER_ROLES: frozenset[Role] = frozenset({Role.nested, Role.member})
MISSING_KINDS: tuple[str, ...] = ("unknown", "zero", "absent", "not_applicable", "non_entity", "false")


def facets_for(role: Role | str) -> frozenset[str]:
    """Every facet a column of ``role`` may carry (besides ``role`` itself)."""
    r = Role(role)
    common = EXISTENCE_FACETS if r in (Role.payload, Role.ignore) else COMMON_FACETS
    return ROLE_FACETS[r] | common


# ---------------------------------------------------------------------------
# Arrow type compatibility (R4)
# ---------------------------------------------------------------------------

_INT = re.compile(r"^u?int(8|16|32|64)$")

#: Type families each role accepts natively. Strings with ``parse``/``stored_as``, floats with
#: ``stored_as`` (integral counts) and encoded flags are handled in :func:`arrow_compatible`;
#: list types are checked by their element type for non-container roles (per-element roles).
ARROW_COMPAT: dict[Role, frozenset[str]] = {
    Role.identifier: frozenset({"string", "int"}),
    Role.label: frozenset({"string"}),
    Role.synonym: frozenset({"string", "struct"}),
    Role.category: frozenset({"string", "int", "float", "bool"}),
    Role.scope: frozenset({"string", "int", "float", "bool", "temporal"}),
    Role.qualifier: frozenset({"bool", "string", "int"}),
    Role.measure: frozenset({"int", "float"}),
    Role.count: frozenset({"int"}),
    Role.flag: frozenset({"bool"}),
    Role.time: frozenset({"temporal", "int"}),
    Role.position: frozenset({"string", "int"}),
    Role.hierarchy: frozenset({"string", "list"}),
    Role.member: frozenset({"list", "string", "struct"}),
    Role.endpoint: frozenset({"string", "int"}),
    Role.vector: frozenset({"list"}),
    Role.text: frozenset({"string"}),
    Role.reference: frozenset({"string", "int"}),
    Role.nested: frozenset({"struct", "list", "map"}),
    Role.payload: frozenset({"string", "int", "float", "bool", "temporal", "duration", "binary", "list", "struct",
                             "map", "null", "other"}),
    Role.ignore: frozenset({"string", "int", "float", "bool", "temporal", "duration", "binary", "list", "struct",
                            "map", "null", "other"}),
}

_LIST_PREFIXES = ("list<", "large_list<", "fixed_size_list<", "list_view<", "large_list_view<")


def _norm_type(arrow_type: str) -> str:
    return str(arrow_type).strip().lower()


def _list_element(t: str) -> str:
    inner = t[t.index("<") + 1: t.rindex(">")]
    name, sep, rest = inner.partition(":")
    if sep and name.strip() in {"item", "element", "elem", "value"}:
        inner = rest
    inner = inner.strip()
    return inner[:-len(" not null")].strip() if inner.endswith(" not null") else inner   # non-nullable elements


def type_family(arrow_type: str) -> str:
    """``string | int | float | bool | temporal | duration | binary | list | struct | map | null | other``.
    ``large_*``, ``string_view`` and dictionary-encoded types map like their plain forms."""
    t = _norm_type(arrow_type)
    if t.startswith("dictionary<"):
        m = re.search(r"values\s*=\s*([^,>]+(?:<[^>]*>)?)", t)
        return type_family(m.group(1)) if m else "other"
    if t.startswith(_LIST_PREFIXES):
        return "list"
    if t.startswith("struct<"):
        return "struct"
    if t.startswith("map<"):
        return "map"
    if t in {"string", "large_string", "utf8", "large_utf8", "string_view"}:
        return "string"
    if t in {"bool", "boolean"}:
        return "bool"
    if _INT.match(t):
        return "int"
    if t in {"halffloat", "float", "double", "float16", "float32", "float64"} or t.startswith("decimal"):
        return "float"
    if t.startswith(("timestamp", "date32", "date64", "date", "time32", "time64")):
        return "temporal"
    if t.startswith("duration"):
        return "duration"
    if t in {"binary", "large_binary", "binary_view"} or t.startswith("fixed_size_binary"):
        return "binary"
    if t == "null":
        return "null"
    return "other"


def arrow_compatible(role: Role | str, arrow_type: str, *, parse: Any = None, stored_as: str | None = None,
                     encoding: Any = None, list_delimiter: str | None = None, element: bool = False) -> bool:
    """Is a column of ``arrow_type`` a valid store for ``role`` (R4)?

    Non-container roles on ``list<T>`` apply per element, so the element type is checked.
    Strings are accepted for numeric, time and flag roles with ``parse`` or ``stored_as``;
    floats for counts with ``stored_as`` (integrality confirmed by R6); any scalar for flags
    with an ``encoding``; strings for list-like roles with a ``list_delimiter``.
    """
    r = Role(role)
    t = _norm_type(arrow_type)
    fam = type_family(t)
    if r in (Role.payload, Role.ignore):
        return True
    if fam == "null":
        return False  # an all-empty column inferred as null is reported, never assumed typed
    if fam == "list" and r not in (Role.nested, Role.member, Role.vector, Role.hierarchy) and not element:
        return arrow_compatible(r, _list_element(t), parse=parse, stored_as=stored_as, encoding=encoding,
                                list_delimiter=list_delimiter, element=True)
    if r == Role.vector:
        return fam == "list" and type_family(_list_element(t)) in {"float", "int"}
    if fam in ARROW_COMPAT[r]:
        return True
    if fam == "string" and (parse is not None or stored_as is not None or list_delimiter is not None):
        return r not in (Role.nested,)
    if r == Role.count and fam == "float" and stored_as is not None:
        return True
    if r == Role.flag and encoding and fam in {"int", "string", "float"}:
        return True
    if r in (Role.time,) and fam in {"string", "float"} and stored_as is not None:
        return True
    if r == Role.qualifier and fam == "float" and encoding:
        return True
    return False


# ---------------------------------------------------------------------------
# Path grammar (§6.4)
# ---------------------------------------------------------------------------

LIST = "[]"
AXES = {"row": "row", "col": "col", "obs": "row", "var": "col"}
_NAME = re.compile(r"\w+", re.UNICODE)
_BARE = re.compile(r"[\w.:\-+]+", re.UNICODE)
_NUMBER = re.compile(r"^-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")


class PathError(ValueError):
    """A path that does not follow the §6.4 grammar."""


@dataclass(frozen=True)
class ItemCond:
    """An item predicate ``[field=literal]`` selecting list items."""

    field: str
    value: Any

    def matches(self, item: Any) -> bool:
        return isinstance(item, dict) and item.get(self.field) == self.value


@dataclass(frozen=True)
class Segment:
    name: str                                          # "" for an item-relative leading segment ("[].label")
    brackets: tuple[Any, ...] = ()                     # LIST or ItemCond per bracket

    @property
    def is_list(self) -> bool:
        return bool(self.brackets)


@dataclass(frozen=True)
class Path:
    segments: tuple[Segment, ...]
    axis: str | None = None                            # "row" | "col" (obs/var normalised)
    up: int = 0                                        # number of leading "^." (parent levels)
    absolute: bool = False                             # leading "/" (table level)

    @property
    def text(self) -> str:
        return format_path(self)

    def __str__(self) -> str:
        return self.text

    @property
    def head(self) -> str:
        return self.segments[0].name if self.segments else ""

    @property
    def crosses_list(self) -> bool:
        return any(s.is_list for s in self.segments)

    @property
    def is_item_relative(self) -> bool:
        return bool(self.segments) and self.segments[0].name == ""

    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.segments)

    def split_container(self) -> tuple["Path", "Path"] | None:
        """``(container, rest)`` at the first list segment: ``a.b[].c[].d`` -> (``a.b[]``, ``c[].d``).
        The container keeps its brackets; ``rest`` is item-relative (may be empty). None without lists."""
        for i, seg in enumerate(self.segments):
            if seg.is_list:
                head = Path(self.segments[: i + 1], self.axis, self.up, self.absolute)
                return head, Path(self.segments[i + 1:])
        return None

    def strip_brackets(self) -> "Path":
        """The same path without list brackets (the column/field structure only)."""
        return Path(tuple(Segment(s.name) for s in self.segments if s.name), self.axis, self.up, self.absolute)


def _read_name(text: str, i: int) -> tuple[str, int]:
    if i < len(text) and text[i] == "`":
        j = text.find("`", i + 1)
        if j < 0:
            raise PathError(f"unterminated backtick in {text!r}")
        return text[i + 1:j], j + 1
    m = _NAME.match(text, i)
    if not m:
        return "", i
    return m.group(0), m.end()


def _read_literal(text: str, i: int) -> tuple[Any, int]:
    if i >= len(text):
        raise PathError(f"missing literal in {text!r}")
    ch = text[i]
    if ch in "\"'":
        j = i + 1
        buf = []
        while j < len(text) and text[j] != ch:
            if text[j] == "\\" and j + 1 < len(text):
                j += 1
            buf.append(text[j])
            j += 1
        if j >= len(text):
            raise PathError(f"unterminated string literal in {text!r}")
        return "".join(buf), j + 1
    m = _BARE.match(text, i)
    if not m:
        raise PathError(f"bad literal at {i} in {text!r}")
    word = m.group(0)
    if word in ("true", "false"):
        return word == "true", m.end()
    if word == "null":
        return None, m.end()
    if _NUMBER.match(word):
        return (float(word) if any(c in word for c in ".eE") else int(word)), m.end()
    return word, m.end()


@lru_cache(maxsize=4096)
def parse_path(text: str) -> Path:
    """Parse a §6.4 path (unlimited depth, ``col[][]``, item predicates, axis and scope prefixes)."""
    if not isinstance(text, str):
        raise PathError(f"path must be a string, got {type(text).__name__}")
    s = text.strip()
    if not s:
        raise PathError("empty path")
    i = 0
    up = 0
    absolute = False
    if s.startswith("/"):
        absolute, i = True, 1
    while s.startswith("^.", i):
        up += 1
        i += 2
    if absolute and up:
        raise PathError(f"'/' and '^.' cannot be combined: {text!r}")
    axis = None
    if i < len(s) and s[i] == "@":
        name, j = _read_name(s, i + 1)
        if name not in AXES:
            raise PathError(f"unknown axis {name!r} in {text!r} (row, col, obs, var)")
        if j >= len(s) or s[j] != ".":
            raise PathError(f"axis prefix must be followed by '.': {text!r}")
        axis, i = AXES[name], j + 1
    segments: list[Segment] = []
    while True:
        name, i = _read_name(s, i)
        brackets: list[Any] = []
        while i < len(s) and s[i] == "[":
            if s.startswith("[]", i):
                brackets.append(LIST)
                i += 2
                continue
            field, j = _read_name(s, i + 1)
            if not field or j >= len(s) or s[j] != "=":
                raise PathError(f"bad item predicate at {i} in {text!r} (expected [field=literal])")
            value, j = _read_literal(s, j + 1)
            if j >= len(s) or s[j] != "]":
                raise PathError(f"unterminated item predicate in {text!r}")
            brackets.append(ItemCond(field, value))
            i = j + 1
        if not name and (segments or not brackets):
            raise PathError(f"empty segment at {i} in {text!r}")
        segments.append(Segment(name, tuple(brackets)))
        if i >= len(s):
            break
        if s[i] != ".":
            raise PathError(f"unexpected {s[i]!r} at {i} in {text!r}")
        i += 1
        if i >= len(s):
            raise PathError(f"path ends with '.': {text!r}")
    return Path(tuple(segments), axis, up, absolute)


def _format_name(name: str) -> str:
    """A column or field name as one §6.4 path segment (backtick-quoted unless it is an identifier)."""
    if name == "" or _NAME.fullmatch(name):
        return name
    if "`" in name:
        raise PathError(f"name cannot contain a backtick: {name!r}")
    return f"`{name}`"


format_name = _format_name


def _format_literal(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    s = str(value)
    if _BARE.fullmatch(s) and s not in ("true", "false", "null") and not _NUMBER.match(s):
        return s
    return json.dumps(s, ensure_ascii=False)


def format_path(path: Path) -> str:
    """The canonical text of a parsed path (``parse_path(format_path(p)) == p``)."""
    out = "/" if path.absolute else "^." * path.up
    if path.axis:
        out += f"@{path.axis}."
    parts = []
    for seg in path.segments:
        part = _format_name(seg.name)
        for b in seg.brackets:
            part += LIST if b == LIST else f"[{_format_name(b.field)}={_format_literal(b.value)}]"
        parts.append(part)
    return out + ".".join(parts)
