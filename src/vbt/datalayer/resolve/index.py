"""Resolver sidecar indexes (§11.5): derived, deletable gzip TSV files. Stdlib only (gzip + csv); no pyarrow.

The data child builds one sidecar per id_type (``_build_index``) by projecting the universe key,
the label and synonym columns (``resolve_via``), retired-ID, xref and crosswalk columns, each
table's stored spelling of the key and parent families, and writes
``<cache_dir>/<source>/<fingerprint>/index/<id_type>.tsv.gz`` with the header
``label_key, canonical, rule, label, stored_table, stored_value, family``. The harness loads it
lazily (:meth:`IndexStore.load`). ``label_key`` is the id_type's plugin ``label_key`` of the row's
lookup text, so the builder and the resolver must use the same configured plugin.

Row kinds (the ``rule`` column, grammar of ``resolve/rules.py``):

==================  =========================  =================  =============================================
rule                label_key of               canonical          other columns
==================  =========================  =================  =============================================
exact               the key                    a universe key     label: primary label; family: parent key
label_exact:<col>   the label                  its key            label as stored; stored_table: the table
                                                                  whose own label it is (native labels)
synonym:<kind>      the synonym                its key            label: the synonym
retired:<col>       the retired ID             its replacement    label: the retired ID; canonical empty when
                                                                  the term was retired without a replacement
xref:<namespace>    the cross-reference        its key            label: the xref as stored
crosswalk:<name>    the other type's key       the key it maps to label: the other type's key as stored
stored_form         the stored value           its key            stored_table, stored_value (as stored)
attr:<col>          (empty)                    a key              label: the key's value of a
                                                                  ``disambiguate_with`` column
==================  =========================  =================  =============================================

Only ``exact`` rows define the universe (:meth:`ResolverIndex.contains`); a retired ID is listed
with ``retired`` rows and is not a member.
"""

from __future__ import annotations

import csv
import gzip
import io
import os
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from .rules import RuleError, parse_entry_rule

__all__ = [
    "HEADER", "Entry", "IndexFormatError", "IndexMissing", "ResolverIndex", "IndexStore", "write_sidecar",
    "read_sidecar", "within_one_edit", "table_matches",
]

HEADER: tuple[str, ...] = ("label_key", "canonical", "rule", "label", "stored_table", "stored_value", "family")


class _TSV(csv.Dialect):
    """Tab-separated, ``\n`` line ends; fields holding tabs, newlines or quotes are quoted."""

    delimiter = "\t"
    quotechar = '"'
    doublequote = True
    skipinitialspace = False
    lineterminator = "\n"
    quoting = csv.QUOTE_MINIMAL


class IndexFormatError(ValueError):
    """A sidecar that is not a resolver index (wrong header, bad rule, short row)."""


class IndexMissing(FileNotFoundError):
    """No sidecar for this (source, fingerprint, id_type): the index must be built."""


@dataclass(frozen=True)
class Entry:
    label_key: str
    canonical: str
    rule: str
    label: str = ""
    stored_table: str = ""
    stored_value: str = ""
    family: str = ""

    @property
    def head(self) -> str:
        return self.rule.partition(":")[0]

    @property
    def arg(self) -> str | None:
        return self.rule.partition(":")[2] or None

    def as_row(self) -> tuple[str, ...]:
        return (self.label_key, self.canonical, self.rule, self.label, self.stored_table, self.stored_value,
                self.family)

    @classmethod
    def coerce(cls, row: "Entry | Mapping[str, Any] | Iterable[Any]") -> "Entry":
        if isinstance(row, Entry):
            return row
        if isinstance(row, Mapping):
            known = {f.name for f in fields(cls)}
            extra = set(row) - known
            if extra:
                raise IndexFormatError(f"unknown index columns {sorted(extra)} (columns: {list(HEADER)})")
            return cls(**{k: "" if v is None else str(v) for k, v in row.items()})
        values = ["" if v is None else str(v) for v in row]
        if not 3 <= len(values) <= len(HEADER):
            raise IndexFormatError(f"an index row has {len(HEADER)} columns, got {len(values)}: {values!r}")
        return cls(*values)


def table_matches(stored_table: str, table: str | None) -> bool:
    """``source.table`` and a bare ``table`` name the same table when their table parts agree."""
    if not table or not stored_table:
        return False
    if stored_table == table:
        return True
    a, b = stored_table.split("."), str(table).split(".")
    if len(a) == 2 and len(b) == 2:
        return False
    return a[-1] == b[-1]


def write_sidecar(path: str | Path, rows: Iterable["Entry | Mapping[str, Any] | Iterable[Any]"]) -> int:
    """Write a sidecar atomically (temporary file + ``os.replace``) in a deterministic order; returns the
    number of rows. Every row's ``rule`` must parse (``rules.parse_entry_rule``)."""
    path = Path(path)
    entries = [Entry.coerce(r) for r in rows]
    for e in entries:
        try:
            parse_entry_rule(e.rule)
        except RuleError as exc:
            raise IndexFormatError(f"index row {e.as_row()!r}: {exc}") from None
    entries.sort(key=Entry.as_row)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
            text = io.TextIOWrapper(gz, encoding="utf-8", newline="")
            writer = csv.writer(text, dialect=_TSV)
            writer.writerow(HEADER)
            for e in entries:
                writer.writerow(e.as_row())
            text.flush()
            text.detach()
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return len(entries)


def read_sidecar(path: str | Path) -> Iterator[Entry]:
    """Rows of a sidecar (header checked)."""
    path = Path(path)
    if not path.is_file():
        raise IndexMissing(f"no resolver index at {path}")
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, dialect=_TSV)
        header = next(reader, None)
        if tuple(header or ()) != HEADER:
            raise IndexFormatError(f"{path}: header {header!r} is not {list(HEADER)!r}")
        for n, row in enumerate(reader, start=2):
            if len(row) != len(HEADER):
                raise IndexFormatError(f"{path}:{n}: {len(row)} columns, expected {len(HEADER)}")
            yield Entry(*row)


def within_one_edit(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` differ by at most one insertion, deletion or substitution (O(n))."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = 0
    while i < la and a[i] == b[i]:
        i += 1
    if la == lb:
        return a[i + 1:] == b[i + 1:]
    return a[i:] == b[i + 1:]


class ResolverIndex:
    """One id_type's resolver index in memory."""

    def __init__(self, entries: Iterable["Entry | Mapping[str, Any] | Iterable[Any]"] = (), *,
                 source: str | None = None, id_type: str | None = None, fingerprint: str | None = None,
                 path: str | Path | None = None) -> None:
        self.source, self.id_type, self.fingerprint = source, id_type, fingerprint
        self.path = Path(path) if path is not None else None
        self._by_key: dict[str, list[Entry]] = {}
        self._members: dict[str, Entry] = {}
        self._children: dict[str, list[str]] = {}
        self._by_canonical: dict[str, list[Entry]] = {}
        self._attrs: dict[str, dict[str, str]] = {}
        self._lengths: dict[int, list[str]] | None = None
        self._count = 0
        for raw in entries:
            e = Entry.coerce(raw)
            self._count += 1
            if e.head == "attr":
                self._attrs.setdefault(e.canonical, {})[e.arg or ""] = e.label
                continue
            self._by_key.setdefault(e.label_key, []).append(e)
            if e.canonical:
                self._by_canonical.setdefault(e.canonical, []).append(e)
            if e.head == "exact" and e.canonical:
                self._members.setdefault(e.canonical, e)
                if e.family and e.family != e.canonical:
                    self._children.setdefault(e.family, []).append(e.canonical)
        self._canonicals = tuple(sorted(self._members))

    @classmethod
    def load(cls, path: str | Path, **meta: Any) -> "ResolverIndex":
        return cls(read_sidecar(path), path=path, **meta)

    def __len__(self) -> int:
        return self._count

    def __repr__(self) -> str:
        where = f"{self.source}:{self.id_type}" if self.id_type else "index"
        return f"<ResolverIndex {where} {self.universe_size} keys, {self._count} rows>"

    # -- universe -------------------------------------------------------------

    @property
    def universe_size(self) -> int:
        return len(self._members)

    def contains(self, canonical: Any) -> bool:
        return str(canonical) in self._members

    def canonicals(self) -> tuple[str, ...]:
        """Every universe key, sorted (the universe sample for plugin ``configure``)."""
        return self._canonicals

    # -- lookups --------------------------------------------------------------

    def lookup(self, label_key: str) -> list[Entry]:
        """Every row (but ``attr`` rows) whose ``label_key`` equals ``label_key``."""
        return list(self._by_key.get(label_key, ()))

    def rows_for(self, canonical: str, head: str | None = None) -> list[Entry]:
        """Rows whose canonical is ``canonical`` (optionally of one rule head), e.g. a reverse crosswalk."""
        rows = self._by_canonical.get(str(canonical), ())
        return [e for e in rows if head is None or e.head == head]

    def labels_for(self, canonical: str) -> list[Entry]:
        return self.rows_for(canonical, "label_exact")

    def label(self, canonical: str) -> str | None:
        """The canonical key's primary label: the ``exact`` row's label, else its first label row."""
        member = self._members.get(str(canonical))
        if member is not None and member.label:
            return member.label
        labels = self.labels_for(canonical)
        return labels[0].label if labels else None

    def native_label(self, canonical: str, table: str) -> str | None:
        """``table``'s own label of the key (``SEPT9`` in a pre-2020 cohort for ``SEPTIN9``)."""
        for e in self.labels_for(canonical):
            if table_matches(e.stored_table, table):
                return e.label
        return None

    def stored_value(self, canonical: str, table: str) -> str | None:
        """``table``'s stored spelling of the key (``'Erdafitinib '``), or None when the table stores it as is."""
        for e in self.rows_for(canonical, "stored_form"):
            if table_matches(e.stored_table, table):
                return e.stored_value
        return None

    def stored_values(self, canonical: str) -> dict[str, str]:
        return {e.stored_table: e.stored_value for e in self.rows_for(canonical, "stored_form")}

    def attributes(self, canonical: str) -> dict[str, str]:
        """The key's ``disambiguate_with`` values (``attr:<column>`` rows)."""
        return dict(self._attrs.get(str(canonical), {}))

    # -- families -------------------------------------------------------------

    def parent(self, canonical: str) -> str | None:
        member = self._members.get(str(canonical))
        if member is not None and member.family and member.family != member.canonical:
            return member.family
        return None

    def family(self, canonical: str) -> tuple[str, ...]:
        """The parent/salt family of a key: the parent first, then its other members (sorted); ``(key,)``
        for a key without a family."""
        canonical = str(canonical)
        head = self.parent(canonical) or canonical
        members = sorted(set(self._children.get(head, ())) - {head})
        if not members:
            return (canonical,)
        return (head, *members)

    # -- suggestions ----------------------------------------------------------

    def suggest(self, label_key: str, max_n: int = 5) -> list[Entry]:
        """Rows whose label key is within one edit of ``label_key`` (never the key itself), one per canonical,
        universe keys and labels first. Same index only: suggestions never cross id_types."""
        if not label_key:
            return []
        if self._lengths is None:
            self._lengths = {}
            for k in self._by_key:
                if k:
                    self._lengths.setdefault(len(k), []).append(k)
        keys: list[str] = []
        for n in (len(label_key) - 1, len(label_key), len(label_key) + 1):
            keys.extend(k for k in self._lengths.get(n, ()) if k != label_key and within_one_edit(label_key, k))
        order = {"exact": 0, "label_exact": 1, "synonym": 2}
        hits = sorted((e for k in keys for e in self._by_key[k] if e.canonical and e.head != "retired"),
                      key=lambda e: (order.get(e.head, 3), e.label_key, e.canonical))
        out: list[Entry] = []
        seen: set[str] = set()
        for e in hits:
            if e.canonical not in seen:
                seen.add(e.canonical)
                out.append(e)
            if len(out) >= max_n:
                break
        return out


class IndexStore:
    """Sidecar locations under ``data.cache_dir`` and a cache of loaded indexes (reloaded when the file changes)."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self._loaded: dict[Path, tuple[tuple[int, int], ResolverIndex]] = {}

    @staticmethod
    def _part(value: str, what: str) -> str:
        text = str(value)
        if not text or text in (".", "..") or "/" in text or "\\" in text or "\0" in text:
            raise ValueError(f"invalid {what} for a sidecar path: {value!r}")
        return text

    def path(self, source: str, fingerprint: str, id_type: str) -> Path:
        """``<cache_dir>/<source>/<fingerprint>/index/<id_type>.tsv.gz`` (``id_type`` bare or ``source:name``)."""
        bare = str(id_type).partition(":")[2] if ":" in str(id_type) else str(id_type)
        fp = str(fingerprint).replace(":", "_")
        return self.cache_dir / self._part(source, "source") / self._part(fp, "fingerprint") / "index" / \
            f"{self._part(bare, 'id_type')}.tsv.gz"

    def exists(self, source: str, fingerprint: str, id_type: str) -> bool:
        return self.path(source, fingerprint, id_type).is_file()

    def write_sidecar(self, source: str, fingerprint: str, id_type: str,
                      rows: Iterable["Entry | Mapping[str, Any] | Iterable[Any]"]) -> Path:
        path = self.path(source, fingerprint, id_type)
        write_sidecar(path, rows)
        self._loaded.pop(path, None)
        return path

    def load(self, source: str, fingerprint: str, id_type: str) -> ResolverIndex:
        """The loaded index (raises :class:`IndexMissing` when it has not been built)."""
        path = self.path(source, fingerprint, id_type)
        try:
            st = path.stat()
        except FileNotFoundError:
            raise IndexMissing(f"no resolver index for {source}:{id_type} at fingerprint {fingerprint} "
                               f"({path}); build it with `vbt ds index build --id-type {id_type}`") from None
        sig = (st.st_mtime_ns, st.st_size)
        hit = self._loaded.get(path)
        if hit is not None and hit[0] == sig:
            return hit[1]
        bare = str(id_type).partition(":")[2] if ":" in str(id_type) else str(id_type)
        index = ResolverIndex.load(path, source=source, id_type=bare, fingerprint=str(fingerprint))
        self._loaded[path] = (sig, index)
        return index

    def fingerprints(self, source: str, id_type: str) -> list[str]:
        """Fingerprints with a built sidecar for this id_type, newest first."""
        root = self.cache_dir / self._part(source, "source")
        bare = str(id_type).partition(":")[2] if ":" in str(id_type) else str(id_type)
        found = []
        if root.is_dir():
            for d in root.iterdir():
                p = d / "index" / f"{bare}.tsv.gz"
                if p.is_file():
                    found.append((p.stat().st_mtime_ns, d.name))
        return [name for _, name in sorted(found, reverse=True)]

    def provider(self, fingerprint_of: Callable[[str, str], str | None] | Mapping[str, str]
                 ) -> Callable[[str, str], ResolverIndex | None]:
        """An index provider for :class:`~vbt.datalayer.resolve.resolver.Resolver`: ``(source, id_type) ->
        index | None``. ``fingerprint_of`` maps a source (or ``(source, id_type)``) to the fingerprint the
        index must have; a missing sidecar gives None (the resolver reports existence ``unknown``)."""

        def fp(source: str, id_type: str) -> str | None:
            if callable(fingerprint_of):
                return fingerprint_of(source, id_type)
            return fingerprint_of.get(f"{source}:{id_type}") or fingerprint_of.get(source)

        def get(source: str, id_type: str) -> ResolverIndex | None:
            f = fp(source, id_type)
            if not f:
                return None
            try:
                return self.load(source, f, id_type)
            except IndexMissing:
                return None

        return get
