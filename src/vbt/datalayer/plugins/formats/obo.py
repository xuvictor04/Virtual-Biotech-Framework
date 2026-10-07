"""The ``obo`` format plugin: OBO 1.2/1.4 ontologies as the published projection (§9.4; phase 2, F12).
pyarrow is imported inside methods only.

Capabilities: ``tabular``, ``nested``.

Every ``[Term]`` stanza is one row of a fixed **projection schema** (``Typedef`` and ``Instance``
stanzas are skipped):

=================  ==========================================  =========================================
column             type                                        from
=================  ==========================================  =========================================
``id``             string                                      ``id:``
``name``           string                                      ``name:``
``namespace``      string                                      ``namespace:``
``def``            string                                      ``def: "text" [xrefs]`` (the text)
``synonyms``       list<struct<text, scope>>                   ``synonym: "text" SCOPE [TYPE] [xrefs]``;
                                                               scope lower-cased (``exact``, ``related``,
                                                               ``broad``, ``narrow``: a synonym's kind)
``alt_id``         list<string>                                ``alt_id:``
``is_obsolete``    bool (absent = false)                       ``is_obsolete:``
``replaced_by``    list<string>                                ``replaced_by:``
``consider``       list<string>                                ``consider:``
``is_a``           list<string>                                ``is_a:`` (trailing ``! comment`` dropped)
``relationship``   list<struct<type, target>>                  ``relationship: part_of UBERON:0002048``
``xref``           list<string>                                ``xref:`` (the CURIE)
``subset``         list<string>                                ``subset:``
=================  ==========================================  =========================================

Absent list tags are ``[]`` and absent scalar tags null, so a term without parents is
distinguishable from an unread one. :meth:`OboFormat.metadata` returns the header tags
(``format-version``, ``data-version``, ``ontology``, ``date`` ...), which the descriptor's release
reads (``release.from: format.data-version``). :meth:`OboFormat.stats` makes no pass: ``rows`` is
``None`` and the byte size is the estimate (``method: size_only``). A zero-length file, binary
content, a file without a ``format-version`` header, a stanza without an ``id`` or a file cut
mid-line raises :class:`FormatError` naming the fragment (I14).
"""

from __future__ import annotations

import io
import re
from typing import Any, Callable, ClassVar, Iterator, Mapping, Sequence

from ...predicate import Predicate
from ..base import ColumnStats, Fragment, FragmentStats, PluginBase
from ..registry import register
from .csv import (
    Memo,
    fragment_identity,
    generic_leaf_path,
    native_rows,
    open_bytes,
    projection,
    residual_compile,
    storage_type_of,
    unreadable,
)

__all__ = ["OboFormat", "OBO_COLUMNS", "parse_obo", "write_obo", "obo_schema", "obo_projection_golden"]

#: Projection columns: name -> "scalar" | "list" | "flag" | "synonyms" | "relationship".
OBO_COLUMNS: dict[str, str] = {
    "id": "scalar", "name": "scalar", "namespace": "scalar", "def": "scalar", "synonyms": "synonyms",
    "alt_id": "list", "is_obsolete": "flag", "replaced_by": "list", "consider": "list", "is_a": "list",
    "relationship": "relationship", "xref": "list", "subset": "list",
}
_SCOPES = ("EXACT", "RELATED", "BROAD", "NARROW")
_QUOTED = re.compile(r'^"((?:[^"\\]|\\.)*)"\s*(.*)$')
_PARSED = Memo()


def obo_schema() -> Any:
    import pyarrow as pa

    s = pa.string()
    fields = []
    for name, kind in OBO_COLUMNS.items():
        if kind == "scalar":
            fields.append(pa.field(name, s))
        elif kind == "flag":
            fields.append(pa.field(name, pa.bool_()))
        elif kind == "list":
            fields.append(pa.field(name, pa.list_(s)))
        elif kind == "synonyms":
            fields.append(pa.field(name, pa.list_(pa.struct([("text", s), ("scope", s)]))))
        else:
            fields.append(pa.field(name, pa.list_(pa.struct([("type", s), ("target", s)]))))
    return pa.schema(fields)


def _unescape(text: str) -> str:
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t"}.get(m.group(1), m.group(1)), text)


def _strip_comment(value: str) -> str:
    """``value`` without a trailing ``! comment`` (outside quotes) and trailing qualifiers ``{...}``."""
    out, quoted, escaped = [], False, False
    for i, ch in enumerate(value):
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            quoted = not quoted
        elif ch == "!" and not quoted and (i == 0 or value[i - 1] in " \t"):
            break
        out.append(ch)
    text = "".join(out).strip()
    return re.sub(r"\s*\{[^{}]*\}\s*$", "", text) if not text.endswith('"') else text


def _empty_row() -> dict[str, Any]:
    return {name: ([] if kind in ("list", "synonyms", "relationship") else (False if kind == "flag" else None))
            for name, kind in OBO_COLUMNS.items()}


def parse_obo(lines: Iterator[str], where: str = "OBO file") -> tuple[dict[str, str], list[dict[str, Any]]]:
    """``(header tags, term rows)`` of an OBO text; raises ``ValueError`` on malformed content."""
    header: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    stanza: str | None = None
    row: dict[str, Any] | None = None
    n = 0

    def flush() -> None:
        if row is not None:
            if not row.get("id"):
                raise ValueError(f"{where}: a [Term] stanza ending at line {n} has no id")
            rows.append(row)

    for n, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("!"):
            continue
        if line.startswith("[") and line.endswith("]"):
            flush()
            stanza = line[1:-1]
            row = _empty_row() if stanza == "Term" else None
            continue
        tag, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"{where}: line {n} is not 'tag: value'")
        tag, value = tag.strip(), value.strip()
        if stanza is None:
            header.setdefault(tag, value)
            continue
        if row is None:
            continue
        value = _strip_comment(value)
        kind = OBO_COLUMNS.get(tag)
        if tag == "def":
            m = _QUOTED.match(value)
            row["def"] = _unescape(m.group(1)) if m else value
        elif tag == "synonym":
            m = _QUOTED.match(value)
            if m is None:
                raise ValueError(f"{where}: line {n}: synonym without a quoted text")
            rest = m.group(2).split()
            scope = rest[0].upper() if rest and rest[0].upper() in _SCOPES else "RELATED"
            row["synonyms"].append({"text": _unescape(m.group(1)), "scope": scope.lower()})
        elif tag == "relationship":
            parts = value.split()
            if len(parts) < 2:
                raise ValueError(f"{where}: line {n}: relationship needs a type and a target")
            row["relationship"].append({"type": parts[0], "target": parts[1]})
        elif tag == "intersection_of":
            continue
        elif kind == "scalar":
            row[tag] = value if row[tag] is None else row[tag]
        elif kind == "flag":
            row[tag] = value.lower() == "true"
        elif kind == "list":
            row[tag].append(value.split()[0] if tag in ("is_a", "xref", "replaced_by", "consider", "alt_id")
                            and value else value)
    flush()
    if "format-version" not in header:
        raise ValueError(f"{where}: no 'format-version' header: not an OBO file")
    return header, rows


@register
class OboFormat(PluginBase):
    kind: ClassVar[str] = "format"
    name: ClassVar[str] = "obo"
    version: ClassVar[str] = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset({"tabular", "nested"})
    requires: ClassVar[tuple[str, ...]] = ("pyarrow",)

    def _parse(self, frag: Fragment) -> tuple[dict[str, str], list[dict[str, Any]]]:
        return _PARSED.get((self.name, *fragment_identity(frag)), lambda: self._read(frag))

    def _read(self, frag: Fragment) -> tuple[dict[str, str], list[dict[str, Any]]]:
        fh = open_bytes(frag)
        with fh:
            try:
                data = fh.read()
            except (OSError, EOFError) as exc:
                raise unreadable(frag, exc, "cannot be read") from exc
        if not data:
            raise unreadable(frag, None, "is empty: not an OBO file")
        if b"\x00" in data:
            raise unreadable(frag, None, "holds binary data (NUL bytes): not an OBO file")
        if not data.endswith((b"\n", b"\r")):
            raise unreadable(frag, None, "does not end with a newline: the file was cut short")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise unreadable(frag, exc, "is not UTF-8 text") from exc
        try:
            return parse_obo(iter(io.StringIO(text)), frag.uri)
        except ValueError as exc:
            raise unreadable(frag, exc, "is not a well-formed OBO file") from exc

    def logical_schema(self, frag: Fragment) -> Any:
        self._parse(frag)
        return obo_schema()

    def leaf_path(self, path: str, schema: Any) -> str:
        return generic_leaf_path(path, schema)

    def stats(self, frag: Fragment) -> FragmentStats:
        self._parse(frag)                                   # an unreadable file is an error, never empty
        size = int(frag.size or 0)
        schema = obo_schema()
        per = size // max(1, len(schema))
        cols = {f.name: ColumnStats(uncompressed_bytes=per, null_count=None, storage_type=str(f.type),
                                    kind="nested" if OBO_COLUMNS[f.name] in ("list", "synonyms", "relationship")
                                    else "string")  # type: ignore[arg-type]
                for f in schema}
        return FragmentStats(rows=None, row_groups=1, columns=cols, method="size_only")

    def metadata(self, frag: Fragment) -> Mapping[str, str]:
        header, rows = self._parse(frag)
        return {**header, "terms": str(len(rows))}

    def storage_type_of(self, schema: Any) -> Callable[[str], str | None]:
        return storage_type_of(schema)

    def compile(self, predicate: Predicate, schema: Any) -> tuple[Any, Predicate | None]:
        return residual_compile(predicate, schema)

    def scan(self, frags: Sequence[Fragment], *, columns: list[str] | None, predicate: Predicate | None,
             partitions: Mapping[str, str], batch_rows: int = 1024,
             row_groups: Mapping[str, Sequence[int]] | None = None, memory_pool: Any = None) -> Iterator[Any]:
        return self._scan(list(frags), columns, predicate, max(1, int(batch_rows)))

    def _scan(self, frags: list[Fragment], columns: list[str] | None, predicate: Predicate | None,
              batch_rows: int) -> Iterator[Any]:
        import pyarrow as pa

        schema = obo_schema()
        for frag in frags:
            _header, rows = self._parse(frag)
            _, residual = self.compile(predicate, schema) if predicate is not None else (None, None)
            wanted = projection(columns, residual, schema.names)
            sub = pa.schema([schema.field(c) for c in wanted])
            for start in range(0, len(rows), batch_rows):
                chunk = [{c: r[c] for c in wanted} for r in rows[start:start + batch_rows]]
                yield pa.RecordBatch.from_pylist(chunk, schema=sub)

    def to_native(self, table: Any) -> list[dict[str, Any]]:
        return native_rows(table)

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import FormatCases

        return FormatCases(extension=".obo", write=write_obo, projection=obo_projection_golden)


def _quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def write_obo(table: Any, path: str, row_group_size: int | None = None, *,
              header: Mapping[str, str] | None = None) -> None:
    """Write projection rows as OBO 1.2 ``[Term]`` stanzas (the inverse of :func:`parse_obo`)."""
    head = {"format-version": "1.2", "data-version": "golden/1", "ontology": "golden"}
    head.update(header or {})
    out = [f"{k}: {v}" for k, v in head.items()]
    for row in table.to_pylist():
        out += ["", "[Term]", f"id: {row['id']}"]
        for tag in ("name", "namespace"):
            if row.get(tag) is not None:
                out.append(f"{tag}: {row[tag]}")
        if row.get("def") is not None:
            out.append(f"def: {_quote(row['def'])} []")
        for tag in ("alt_id", "subset"):
            out += [f"{tag}: {v}" for v in row.get(tag) or []]
        for syn in row.get("synonyms") or []:
            out.append(f"synonym: {_quote(syn['text'])} {syn['scope'].upper()} []")
        out += [f"xref: {v}" for v in row.get("xref") or []]
        out += [f"is_a: {v} ! parent" for v in row.get("is_a") or []]
        out += [f"relationship: {r['type']} {r['target']} ! target" for r in row.get("relationship") or []]
        if row.get("is_obsolete"):
            out.append("is_obsolete: true")
        out += [f"replaced_by: {v}" for v in row.get("replaced_by") or []]
        out += [f"consider: {v}" for v in row.get("consider") or []]
    out += ["", "[Typedef]", "id: part_of", "name: part of"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


def obo_projection_golden() -> Any:
    """A projection golden: obsolete terms with replaced_by/consider, synonyms of every scope (a quote and
    a ``!`` inside the text), multi-parent terms, relationships, xrefs, subsets and absent tags."""
    import pyarrow as pa

    def term(id_: str, **kw: Any) -> dict[str, Any]:
        row = _empty_row()
        row["id"] = id_
        row.update(kw)
        return row

    rows = [
        term("GO:0008150", name="biological_process", namespace="biological_process", subset=["goslim_generic"]),
        term("GO:0009987", name="cellular process", namespace="biological_process",
             **{"def": 'Any process that is carried out at the cellular level; "cell" level!'},
             is_a=["GO:0008150"], synonyms=[{"text": "cell physiology", "scope": "exact"},
                                            {"text": "cellular physiological process", "scope": "broad"}],
             xref=["Wikipedia:Cell_(biology)"]),
        term("GO:0005737", name="cytoplasm", namespace="cellular_component",
             synonyms=[{"text": "cytosol (broad sense)", "scope": "narrow"}],
             relationship=[{"type": "part_of", "target": "GO:0005622"}], alt_id=["GO:0000000"]),
        term("GO:0005622", name="intracellular anatomical structure", namespace="cellular_component",
             is_a=["GO:0110165", "GO:0005575"]),
        term("GO:0000005", name="obsolete ribosomal chaperone activity", namespace="molecular_function",
             is_obsolete=True, consider=["GO:0042254", "GO:0044183"]),
        term("GO:0000108", name="obsolete repairosome", namespace="cellular_component", is_obsolete=True,
             replaced_by=["GO:0000109"], synonyms=[{"text": "repairosome", "scope": "related"}]),
        term("GO:9999999"),
    ]
    return pa.Table.from_pylist(rows, schema=obo_schema())
