"""SOMA ``value_filter`` strings -> the predicate IR and back (§11.3 step 4, rev 2). No pyarrow.

Census tools take a free ``value_filter`` such as ``tissue_general == 'lung' and disease in
['normal', 'COVID-19']``. Upstream passes it to TileDB-SOMA verbatim, so a misspelled value
silently selects nothing and duplicate cells are counted unless the caller remembers
``is_primary_data == True``. The gateway therefore:

1. parses the filter with :func:`parse` (``==``, ``!=``, ``in [...]``, ``not in [...]``, ``<``,
   ``<=``, ``>``, ``>=``, ``and``, ``or``, ``not``, parentheses, quoted strings with backslash
   escapes, numbers, ``True``/``False``) into :mod:`vbt.datalayer.predicate` nodes; an unparsable
   filter is :class:`SomaFilterError` (``invalid_argument`` in enforce mode);
2. resolves every column and string value against the Census vocabularies
   (:func:`resolve_filter`; the gateway fetches them with the upstream ``list_metadata_values``
   tool and keeps them in a :class:`VocabCache` with a TTL): exact, then a unique casefold match;
3. ANDs ``is_primary_data == True`` unless ``include_duplicates`` (:func:`enforce_primary`);
4. recompiles the filter with correct quoting (:func:`compile_filter`).

:func:`fixes_single` tells whether a filter fixes a column to exactly one value (the donor
balanced tool is accepted only when ``dataset_id`` is fixed, because ``donor_id`` is unique only
within a dataset).
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Sequence

from ..errors import nearest
from ..predicate import And, Cmp, Eq, In, Not, Or, Predicate

__all__ = ["SomaFilterError", "Token", "tokenize", "parse", "compile_filter", "quote", "resolve_filter",
           "enforce_primary", "fixes_single", "filter_columns", "VocabCache", "PRIMARY_COLUMN", "LANGUAGE",
           "VOCAB_TOOL", "VOCAB_ARG", "VOCAB_PATH"]

#: ``ArgBinding.escape`` value of arguments that hold a SOMA value filter.
LANGUAGE = "soma"
PRIMARY_COLUMN = "is_primary_data"
#: The upstream listing tool on the same server that returns a column's vocabulary.
VOCAB_TOOL = "list_metadata_values"
VOCAB_ARG = "column_name"
VOCAB_PATH = "$.value_counts[*].value"


class SomaFilterError(ValueError):
    """An unparsable filter, or a column or value that does not resolve."""

    def __init__(self, message: str, *, column: str | None = None, value: Any = None,
                 valid_values: Sequence[Any] | None = None) -> None:
        super().__init__(message)
        self.column = column
        self.value = value
        self.valid_values = list(valid_values) if valid_values is not None else None


@dataclass(frozen=True)
class Token:
    kind: str                                          # ident | string | number | op | lbr | rbr | lpar | rpar | comma | kw
    value: Any
    pos: int


_OPS = ("==", "!=", "<=", ">=", "<", ">")
_KEYWORDS = {"and", "or", "not", "in"}
_CONSTS = {"True": True, "true": True, "False": False, "false": False}
_NUMBER = re.compile(r"-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def tokenize(text: str) -> list[Token]:
    out: list[Token] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c in "'\"":
            quote_char, j, buf = c, i + 1, []
            while j < n and text[j] != quote_char:
                if text[j] == "\\" and j + 1 < n:
                    buf.append(text[j + 1])
                    j += 2
                    continue
                buf.append(text[j])
                j += 1
            if j >= n:
                raise SomaFilterError(f"unterminated string at position {i}")
            out.append(Token("string", "".join(buf), i))
            i = j + 1
            continue
        two = text[i:i + 2]
        if two in _OPS:
            out.append(Token("op", two, i))
            i += 2
            continue
        if c in "<>":
            out.append(Token("op", c, i))
            i += 1
            continue
        if c == "=":
            raise SomaFilterError(f"'=' at position {i}: use '=='")
        simple = {"[": "lbr", "]": "rbr", "(": "lpar", ")": "rpar", ",": "comma"}
        if c in simple:
            out.append(Token(simple[c], c, i))
            i += 1
            continue
        m = _NUMBER.match(text, i)
        if m and (c.isdigit() or c in "-."):
            raw = m.group(0)
            out.append(Token("number", float(raw) if any(ch in raw for ch in ".eE") else int(raw), i))
            i = m.end()
            continue
        m = _IDENT.match(text, i)
        if m:
            word = m.group(0)
            if word in _KEYWORDS:
                out.append(Token("kw", word, i))
            elif word in _CONSTS:
                out.append(Token("const", _CONSTS[word], i))
            else:
                out.append(Token("ident", word, i))
            i = m.end()
            continue
        raise SomaFilterError(f"unexpected character {c!r} at position {i}")
    return out


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.t = tokens
        self.i = 0

    def peek(self, kind: str | None = None, value: Any = None) -> Token | None:
        if self.i >= len(self.t):
            return None
        tok = self.t[self.i]
        if kind is not None and tok.kind != kind:
            return None
        if value is not None and tok.value != value:
            return None
        return tok

    def take(self, kind: str, value: Any = None, what: str = "") -> Token:
        tok = self.peek(kind, value)
        if tok is None:
            got = self.t[self.i] if self.i < len(self.t) else None
            where = f"at position {got.pos} ({got.value!r})" if got else "at the end"
            raise SomaFilterError(f"expected {what or value or kind} {where}")
        self.i += 1
        return tok

    def expr(self) -> Predicate:
        parts = [self.and_expr()]
        while self.peek("kw", "or"):
            self.i += 1
            parts.append(self.and_expr())
        return parts[0] if len(parts) == 1 else Or(tuple(parts))

    def and_expr(self) -> Predicate:
        parts = [self.not_expr()]
        while self.peek("kw", "and"):
            self.i += 1
            parts.append(self.not_expr())
        return parts[0] if len(parts) == 1 else And(tuple(parts))

    def not_expr(self) -> Predicate:
        if self.peek("kw", "not"):
            self.i += 1
            return Not(self.not_expr())
        return self.atom()

    def atom(self) -> Predicate:
        if self.peek("lpar"):
            self.i += 1
            inner = self.expr()
            self.take("rpar", what="')'")
            return inner
        col = self.take("ident", what="a column name").value
        if self.peek("kw", "in"):
            self.i += 1
            return In(col, tuple(self.values()))
        if self.peek("kw", "not"):
            self.i += 1
            self.take("kw", "in", what="'in' after 'not'")
            return Not(In(col, tuple(self.values())))
        op = self.take("op", what="a comparison operator").value
        value = self.value()
        if op == "==":
            return Eq(col, value)
        return Cmp(col, op, value)

    def value(self) -> Any:
        tok = self.peek()
        if tok is None or tok.kind not in ("string", "number", "const"):
            where = f"at position {tok.pos} ({tok.value!r})" if tok else "at the end"
            raise SomaFilterError(f"expected a value {where}")
        self.i += 1
        return tok.value

    def values(self) -> list[Any]:
        self.take("lbr", what="'['")
        out = [self.value()]
        while self.peek("comma"):
            self.i += 1
            out.append(self.value())
        self.take("rbr", what="']'")
        return out


def parse(text: str) -> Predicate:
    """Parse a SOMA value filter (see the module docstring)."""
    if not str(text or "").strip():
        raise SomaFilterError("empty value_filter")
    p = _Parser(tokenize(str(text)))
    out = p.expr()
    if p.i != len(p.t):
        tok = p.t[p.i]
        raise SomaFilterError(f"unexpected {tok.value!r} at position {tok.pos}")
    return out


def quote(value: Any) -> str:
    """A SOMA literal: strings single-quoted with ``\\`` and ``'`` escaped; booleans ``True``/
    ``False``; numbers in their shortest repr."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SomaFilterError(f"non-finite number {value!r} in a filter")
        return repr(value)
    text = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{text}'"


def compile_filter(p: Predicate) -> str:
    """The SOMA filter text of ``p`` (inverse of :func:`parse` up to whitespace and parentheses)."""
    if isinstance(p, Eq):
        return f"{p.column} == {quote(p.value)}"
    if isinstance(p, Cmp):
        return f"{p.column} {p.op} {quote(p.value)}"
    if isinstance(p, In):
        return f"{p.column} in [{', '.join(quote(v) for v in p.values)}]"
    if isinstance(p, Not):
        if isinstance(p.pred, In):
            inner = p.pred
            return f"{inner.column} not in [{', '.join(quote(v) for v in inner.values)}]"
        return f"not ({compile_filter(p.pred)})"
    if isinstance(p, And):
        return " and ".join(_group(q) for q in p.preds)
    if isinstance(p, Or):
        return " or ".join(_group(q) for q in p.preds)
    raise SomaFilterError(f"cannot express {type(p).__name__} in a SOMA filter")


def _group(p: Predicate) -> str:
    text = compile_filter(p)
    return f"({text})" if isinstance(p, (And, Or)) else text


def filter_columns(p: Predicate) -> list[str]:
    from ..predicate import columns
    return sorted(columns(p))


def _map_values(p: Predicate, fn: Callable[[str, Any], Any]) -> Predicate:
    if isinstance(p, Eq):
        return Eq(p.column, fn(p.column, p.value))
    if isinstance(p, Cmp):
        return Cmp(p.column, p.op, fn(p.column, p.value))
    if isinstance(p, In):
        return In(p.column, tuple(fn(p.column, v) for v in p.values))
    if isinstance(p, Not):
        return Not(_map_values(p.pred, fn))
    if isinstance(p, And):
        return And(tuple(_map_values(q, fn) for q in p.preds))
    if isinstance(p, Or):
        return Or(tuple(_map_values(q, fn) for q in p.preds))
    return p


def resolve_filter(p: Predicate, vocab: Mapping[str, Sequence[Any] | None], *,
                   columns: Sequence[str] | None = None) -> tuple[Predicate, list[str]]:
    """Resolve columns and string values against ``vocab`` (``{column: values | None}``; None:
    not checkable). Returns ``(predicate, notes)``. An unknown column (when ``columns`` lists
    the valid ones) or a value with no exact or unique casefold match raises :class:`SomaFilterError`."""
    notes: list[str] = []
    known_cols = list(columns) if columns is not None else None
    for col in filter_columns(p):
        if known_cols is not None and col not in known_cols:
            raise SomaFilterError(f"unknown column {col!r} in value_filter", column=col,
                                  valid_values=nearest(col, known_cols, 10))

    def fix(col: str, value: Any) -> Any:
        values = vocab.get(col)
        if values is None or not isinstance(value, str):
            return value
        if value in values:
            return value
        folded = [v for v in values if isinstance(v, str) and v.casefold() == value.casefold()]
        if len(folded) == 1:
            notes.append(f"value_filter: {col} {value!r} -> {folded[0]!r} (casefold)")
            return folded[0]
        raise SomaFilterError(f"{col} has no value {value!r}", column=col, value=value,
                              valid_values=list(values))

    return _map_values(p, fix), notes


def _fixed_values(p: Predicate, column: str) -> list[Any] | None:
    """Values ``column`` is restricted to by top-level conjuncts (None: not restricted)."""
    if isinstance(p, Eq) and p.column == column:
        return [p.value]
    if isinstance(p, In) and p.column == column:
        return list(p.values)
    if isinstance(p, And):
        found: list[Any] | None = None
        for q in p.preds:
            vals = _fixed_values(q, column)
            if vals is not None:
                found = vals if found is None else [v for v in found if v in vals]
        return found
    return None


def fixes_single(p: Predicate | None, column: str) -> bool:
    """``p`` restricts ``column`` to exactly one value (``col == v`` or ``col in [v]`` conjunct)."""
    if p is None:
        return False
    vals = _fixed_values(p, column)
    return vals is not None and len(set(map(repr, vals))) == 1


def enforce_primary(p: Predicate | None, include_duplicates: bool = False,
                    column: str = PRIMARY_COLUMN) -> tuple[Predicate | None, bool]:
    """AND ``column == True`` unless ``include_duplicates`` or the filter already fixes it.
    Returns ``(predicate, added)``."""
    if include_duplicates:
        return p, False
    vals = _fixed_values(p, column) if p is not None else None
    if vals is not None:
        return p, False
    guard = Eq(column, True)
    if p is None:
        return guard, True
    parts = list(p.preds) if isinstance(p, And) else [p]
    return And(tuple(parts + [guard])), True


class VocabCache:
    """Census vocabularies per column with a TTL (``data.resolution.remote_ttl_s``)."""

    def __init__(self, ttl_s: float = 3600.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.ttl_s = float(ttl_s)
        self.clock = clock
        self._values: dict[str, tuple[float, list[Any]]] = {}

    def get(self, column: str) -> list[Any] | None:
        hit = self._values.get(column)
        if hit is None or self.clock() - hit[0] > self.ttl_s:
            return None
        return hit[1]

    def put(self, column: str, values: Sequence[Any]) -> None:
        self._values[column] = (self.clock(), list(values))

    async def fetch(self, columns: Sequence[str],
                    fetcher: Callable[[str], Awaitable[Sequence[Any] | None]]) -> dict[str, list[Any] | None]:
        """The vocabularies of ``columns``, fetching stale ones with ``fetcher(column)`` (a failed
        fetch gives None: that column is not checked)."""
        out: dict[str, list[Any] | None] = {}
        for col in columns:
            got = self.get(col)
            if got is None:
                try:
                    fetched = await fetcher(col)
                except Exception:  # noqa: BLE001 - an unreachable vocabulary is not checked
                    fetched = None
                if fetched is not None:
                    self.put(col, fetched)
                    got = list(fetched)
            out[col] = got
        return out
