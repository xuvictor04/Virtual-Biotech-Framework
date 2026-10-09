"""Plugin protocols, their dataclasses and optional capabilities (§9.2). No pyarrow at import.

Five kinds: ``format``, ``layout``, ``statistic``, ``identifier`` and (phase 4, the worked example
of adding a kind, §9.5) ``envelope``: decoders of third-party result shapes that an overlay's
JSONPaths cannot describe. Each plugin class declares
``kind``, ``name``, ``version``, ``api == API_VERSION``, ``capabilities`` and ``requires``
(importable modules probed by preflight). Protocol methods that only some plugins implement
are **optional capabilities**: a plugin lists the capability in ``capabilities``, the
conformance suite runs only the cases for declared capabilities, and the core calls the
method only when the capability is present (:data:`CAPABILITY_METHODS`). Later phases add
plugins, never protocol methods.

Format plugins import pyarrow only inside the methods that read data (the Arrow types below are
plain ``Any`` aliases), so the harness can import any plugin module.

A sixth kind, ``acquisition`` (:class:`AcquisitionPlugin`), runs on the harness side only: the transports of
``vbt data acquire`` (an HTTPS server with a directory index or a checksum list, a Hugging Face dataset
repository, a public S3 or GCS bucket, a JSON index such as a Zenodo record or a figshare article, members of a
remote zip). It is discovered like the others but kept out of :data:`~vbt.datalayer.plugins.KINDS`, so the data
child's registry, its provenance ``plugins`` and the pinned kind set do not change: see
:data:`~vbt.datalayer.plugins.HARNESS_KINDS` and :data:`HARNESS_CAPABILITIES`.
"""

from __future__ import annotations

import copy
import re
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Literal, Mapping, Protocol, Self, Sequence

from ..predicate import Predicate, RankKey

if TYPE_CHECKING:  # pragma: no cover
    from ..descriptor.columns import MeasureCol as MeasureSpec
    from ..descriptor.models import ConstraintSpec, RemoteBudget

#: Arrow objects as format plugins return them. The harness never imports pyarrow (I12), so the
#: protocols name them through these aliases only.
ArrowSchema = Any
ArrowTable = Any
ArrowRecordBatch = Any

__all__ = [
    "API_VERSION", "KIND_NAMES", "Fragment", "ColumnStats", "FragmentStats", "CheckItem", "Manifest", "LayoutSpec",
    "Page", "ParsedResult", "ValueSnapshot", "ConfirmResult", "ConfirmedFacts", "AggResult", "FamilySpec", "Normalized", "Rejected",
    "Candidate", "Resolution", "ResolutionStatus", "UnsupportedFilter", "FormatError", "PluginError",
    "NORMALIZE_STEPS", "CAPABILITIES", "CAPABILITY_METHODS", "REQUIRED_METHODS", "REQUIRED_ATTRS",
    "FormatPlugin", "LayoutPlugin", "StatisticPlugin", "IdentifierPlugin", "EnvelopePlugin", "PluginBase",
    "IdentifierBase", "EnvelopeBase",
    "plugin_key", "ArrowSchema", "ArrowTable", "ArrowRecordBatch",
    "RemoteFile", "AcquisitionError", "AcquisitionPlugin", "AcquisitionBase", "HARNESS_CAPABILITIES",
    "CHECKSUM_ALGOS",
]

API_VERSION = 1
KIND_NAMES = ("format", "layout", "statistic", "identifier", "envelope")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fragment:
    uri: str
    size: int | None
    mtime_ns: int | None
    partition: Mapping[str, Any] = field(default_factory=dict)
    fragment_key: str | None = None                    # e.g. "GSE12251" from TableSpec.fragment_key
    sha256: str | None = None                          # from a manifest when known


@dataclass(frozen=True)
class ColumnStats:
    uncompressed_bytes: int
    null_count: int | None
    num_values: int | None = None                      # leaf values incl. nested items (object-overhead estimate)
    max_rep_level: int = 0
    max_def_level: int = 0
    min: Any = None
    max: Any = None
    storage_type: str | None = None                    # "float", "large_string", "int32", ...
    kind: Literal["flat", "string", "nested", "dense_matrix", "sparse_matrix"] = "flat"


@dataclass(frozen=True)
class FragmentStats:
    rows: int | None                                   # None when unknown without a scan (OBO, text)
    row_groups: int
    columns: Mapping[str, ColumnStats]
    method: Literal["footer", "scan", "size_only"] = "footer"
    shape: tuple[int, int] | None = None               # matrices: (n_rows, n_cols)


@dataclass(frozen=True)
class CheckItem:
    """One readiness finding of a layout probe or a check (R1..R10)."""

    name: str
    ok: bool
    detail: str = ""
    hint: str = ""
    level: Literal["error", "warning", "info"] = "error"
    column: str | None = None
    partition: str | None = None


@dataclass(frozen=True)
class Manifest:
    """A parsed download/preparation manifest."""

    path: str | None
    entries: Mapping[str, Mapping[str, Any]]           # relpath -> {bytes, sha256 | md5}
    data: Mapping[str, Any] = field(default_factory=dict)
    sha256: str | None = None                          # of the manifest file itself
    complete: bool | None = None


@dataclass(frozen=True)
class LayoutSpec:
    """What a layout plugin needs to know about one table."""

    table: str                                         # "source.table"
    path: str | None
    options: Mapping[str, Any] = field(default_factory=dict)
    partitions: Mapping[str, str] = field(default_factory=dict)        # name -> declared type
    partition_expect: Mapping[str, Sequence[Any] | None] = field(default_factory=dict)   # declared vocabularies
    fragment_key: Mapping[str, Any] | None = None      # {name, from, pattern}
    format: str | None = None


@dataclass(frozen=True)
class Page:
    """One page of a live request."""

    rows: Sequence[Mapping[str, Any]]
    total: int | None
    next: str | None
    as_of: str | None


@dataclass(frozen=True)
class ParsedResult:
    """What an envelope plugin read from one tool result (§9.5, F21).

    ``rows`` is None when the payload holds no row list the spec names (a record, a count, an
    unknown shape); ``found`` is False only for an explicit not-found and None when the payload does
    not say; ``unparsed`` marks a shape the plugin does not understand: the caller then makes no
    claim from it and never invents rows. ``errors`` are nested error messages a success envelope
    carried, and ``http_status`` the HTTP status the payload or its error text reports."""

    rows: Sequence[Any] | None = None
    total: int | None = None
    found: bool | None = None
    message: str | None = None
    unparsed: bool = False
    errors: tuple[str, ...] = ()
    http_status: int | None = None


@dataclass(frozen=True)
class ValueSnapshot:
    """Distinct values of a column (R6/R7): every value with its count when the scan was complete."""

    values: tuple[Any, ...]
    counts: Mapping[Any, int] | None = None
    null_count: int = 0
    nan_count: int = 0
    complete: bool = True
    fingerprint: str | None = None
    storage_type: str | None = None


@dataclass(frozen=True)
class ConfirmResult:
    confirmed: bool | None                             # None: cannot decide from this evidence
    facts: Mapping[str, Any] = field(default_factory=dict)   # confirmed facts (range, codes seen, ...)
    detail: str = ""


ConfirmedFacts = Mapping[str, Any]


@dataclass(frozen=True)
class AggResult:
    value: Any                                         # None for an empty input, never 0
    n: int
    n_excluded: int = 0
    detail: str = ""


@dataclass(frozen=True)
class FamilySpec:
    """A multiple-testing family: the columns that define one family."""

    columns: tuple[str, ...]
    method: str | None = None
    origin: str | None = None


@dataclass(frozen=True)
class Normalized:
    value: str
    steps: tuple[str, ...] = ()                        # from NORMALIZE_STEPS (I-5)


@dataclass(frozen=True)
class Rejected:
    reason: str
    looks_like: tuple[str, ...] = ()


@dataclass(frozen=True)
class Candidate:
    id: str
    label: str | None = None
    via: str | None = None                             # the rule that matched
    extra: Mapping[str, Any] = field(default_factory=dict)   # disambiguate_with values


# resolved_unverified: existence ``bound`` and the key is absent from the universe index (§11.5).
ResolutionStatus = Literal["resolved", "resolved_unverified", "ambiguous", "not_found", "rejected", "unknown"]


@dataclass(frozen=True)
class Resolution:
    """The outcome of resolving one value (§11.5)."""

    status: ResolutionStatus
    canonical: str | None = None
    rule: str | None = None                            # closed rule grammar: exact, label_exact:<col>, ...
    tried: tuple[str, ...] = ()
    candidates: tuple[Candidate, ...] = ()
    id_type: str | None = None                         # canonical (bound column's) id_type, qualified
    matched_id_type: str | None = None                 # the accepted kind that matched, qualified
    family: tuple[str, ...] = ()
    hops: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    raw: Any = None


class UnsupportedFilter(Exception):
    """A threshold that cannot be applied faithfully (I9, scale, comparable_within)."""

    def __init__(self, message: str, *, reason: str = "scale", column: str | None = None,
                 confirmed_range: Sequence[Any] | None = None, group: Sequence[str] | None = None) -> None:
        super().__init__(message)
        self.reason = reason                           # scale | unconfirmed_encoding | unbound_argument | group
        self.column = column
        self.confirmed_range = confirmed_range
        self.group = tuple(group or ())


class FormatError(Exception):
    """A fragment that is corrupt, truncated or unreadable: never an empty table (F-8, I14)."""

    def __init__(self, message: str, *, fragment: str | None = None, partition: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.fragment = fragment
        self.partition = dict(partition or {})


class PluginError(Exception):
    """A plugin that cannot be registered (collision, API mismatch, missing attributes)."""


#: Checksum algorithms a listing may report, strongest first (``git_sha1`` is the blob id of a file a git
#: repository stores without LFS, ``crc32`` the CRC of a zip member).
CHECKSUM_ALGOS: tuple[str, ...] = ("sha256", "sha1", "md5", "git_sha1", "crc32")


@dataclass(frozen=True)
class RemoteFile:
    """One file a source publishes, as an acquisition plugin lists it.

    ``path`` is a canonical relative path (where the file lands under the download directory, and what the
    descriptor's acquisition patterns match); ``url`` where it is read from; ``checksums`` the publisher's
    digests by algorithm (:data:`CHECKSUM_ALGOS`, lowercase hex); ``member`` the member name when the file is
    inside an archive at ``url``."""

    path: str
    url: str
    size: int | None = None
    checksums: Mapping[str, str] = field(default_factory=dict)
    member: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


class AcquisitionError(Exception):
    """A listing or a transfer that failed. A transport never answers a failure with an empty listing or an
    empty file (I14). ``transient`` failures (connection reset, HTTP 429/5xx) are retried by the engine."""

    def __init__(self, message: str, *, url: str | None = None, status: int | None = None,
                 transient: bool = False) -> None:
        super().__init__(message)
        self.url = url
        self.status = status
        self.transient = transient


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------

#: The I-5 whitelist of identifier normalisation steps.
NORMALIZE_STEPS: frozenset[str] = frozenset({
    "strip", "upper", "lower", "strip_version", "curie_colon_to_underscore", "curie_underscore_to_colon",
    "strip_chr", "separator_to_underscore", "strip_prefix", "canonical_prefix_case", "strip_suffix",
})

#: Capabilities a plugin of each kind may declare (§9.1, rev 2).
CAPABILITIES: dict[str, frozenset[str]] = {
    "format": frozenset({"tabular", "pushdown", "stats", "stats_scan", "nested", "leaf_projection", "matrix",
                         "vectors", "string_compile"}),
    "layout": frozenset({"scan", "count", "live", "upstream_only"}),
    "statistic": frozenset({"test", "paired", "veto_labels"}),
    "identifier": frozenset({"label", "union", "options"}),
    "envelope": frozenset({"rows", "totals", "nested_errors", "http_status"}),
}

#: Capabilities of the harness-side kinds (:data:`~vbt.datalayer.plugins.HARNESS_KINDS`), kept apart from
#: :data:`CAPABILITIES` (whose keys are the data child's kinds). ``acquisition``: ``range`` a fetch resumes from
#: an offset; ``sizes`` / ``checksums`` the listing reports them; ``members`` the files are members of an archive
#: (fetched whole, checked by CRC); ``index`` the listing reads one index document that can be kept beside the
#: data (``index_cache``), so a manifest can be rewritten offline.
HARNESS_CAPABILITIES: dict[str, frozenset[str]] = {
    "acquisition": frozenset({"range", "sizes", "checksums", "members", "index"}),
}

#: Methods a declared capability requires.
CAPABILITY_METHODS: dict[str, dict[str, tuple[str, ...]]] = {
    "format": {"leaf_projection": ("read_leaves",), "matrix": ("axis_values", "slice"), "string_compile": ("quote",)},
    "layout": {"live": ("request",), "count": ("count",)},
    "statistic": {"test": ("test",), "paired": ("aggregate_pair",), "veto_labels": ("vetoed_companions",)},
    "identifier": {},
    "envelope": {},
    "acquisition": {},
}

#: Methods every plugin of a kind implements.
REQUIRED_METHODS: dict[str, tuple[str, ...]] = {
    "format": ("logical_schema", "leaf_path", "stats", "metadata", "scan", "compile", "to_native"),
    "layout": ("fragments", "partition_columns", "signature", "fingerprint", "partition_fingerprints", "probe",
               "as_of"),
    "statistic": ("sort_key", "predicate", "bounds", "validate", "aggregate", "comparable", "describe", "family"),
    "identifier": ("configure", "normalize", "normalize_stored", "looks_like", "label_key", "describe"),
    "envelope": ("decode",),
    "acquisition": ("listing", "fetch", "describe"),
}

REQUIRED_ATTRS: dict[str, tuple[str, ...]] = {
    "format": ("kind", "name", "version", "capabilities"),
    "layout": ("kind", "name", "version", "capabilities"),
    "statistic": ("kind", "name", "version", "capabilities"),
    "identifier": ("kind", "name", "version", "id_type", "canonical", "examples"),
    "envelope": ("kind", "name", "version", "capabilities"),
    "acquisition": ("kind", "name", "version", "capabilities"),
}


def plugin_key(plugin: Any) -> str:
    """``"kind/name"`` (the key of ``versions()`` and of provenance ``plugins``)."""
    return f"{plugin.kind}/{plugin.name}"


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class FormatPlugin(Protocol):
    kind: ClassVar[str] = "format"
    name: ClassVar[str]
    version: ClassVar[str]
    api: ClassVar[int] = API_VERSION
    capabilities: ClassVar[frozenset[str]]
    requires: ClassVar[tuple[str, ...]] = ()

    def logical_schema(self, frag: Fragment) -> ArrowSchema: ...
    def leaf_path(self, path: str, schema: ArrowSchema) -> str: ...
    def stats(self, frag: Fragment) -> FragmentStats: ...
    def metadata(self, frag: Fragment) -> Mapping[str, str]: ...

    def scan(self, frags: Sequence[Fragment], *, columns: list[str], predicate: Predicate | None,
             partitions: Mapping[str, str], batch_rows: int = 1024,
             row_groups: Mapping[str, Sequence[int]] | None = None) -> Iterator[ArrowRecordBatch]: ...

    def compile(self, predicate: Predicate, schema: ArrowSchema) -> tuple[Any, Predicate | None]:
        """``(pushdown, residual)``: literals cast to the column's storage type; NaN compiles as null for
        measure/count/time/flag columns; Contains/Any/All over lists are always residual."""
        ...

    def to_native(self, table: ArrowTable) -> list[dict[str, Any]]: ...

    # capability leaf_projection
    def read_leaves(self, frag: Fragment, leaves: list[str], row_groups: Sequence[int] | None) -> ArrowTable: ...

    # capability matrix
    def axis_values(self, frag: Fragment, axis: str) -> ArrowTable: ...

    def slice(self, frag: Fragment, value: str, *, row_predicate: Predicate | None, col_keys: Sequence[Any] | None,
              budget_bytes: int) -> Iterator[ArrowRecordBatch]: ...

    # capability string_compile: quote one literal for the format's query language (essie, soma, ...)
    def quote(self, value: Any) -> str: ...

    @classmethod
    def conformance_cases(cls) -> Any: ...


class LayoutPlugin(Protocol):
    kind: ClassVar[str] = "layout"
    name: ClassVar[str]
    version: ClassVar[str]
    api: ClassVar[int] = API_VERSION
    capabilities: ClassVar[frozenset[str]]
    requires: ClassVar[tuple[str, ...]] = ()

    def fragments(self, root: str, spec: LayoutSpec) -> list[Fragment]:
        """Lists by NAME; excludes *.part, *.part.json, dotfiles and _SUCCESS; never skips a listed
        file because it is unreadable (I14)."""
        ...

    def partition_columns(self, spec: LayoutSpec) -> dict[str, str]: ...
    def signature(self, root: str, spec: LayoutSpec) -> str: ...      # stat-only, stdlib only (harness side)
    def fingerprint(self, frags: list[Fragment], manifest: Manifest | None) -> str: ...
    def partition_fingerprints(self, frags: list[Fragment], manifest: Manifest | None) -> dict[str, str]: ...
    def probe(self, root: str, spec: LayoutSpec, manifest: Manifest | None) -> list[CheckItem]: ...
    def as_of(self, root: str, spec: LayoutSpec) -> str | None: ...

    # capability live
    def request(self, spec: LayoutSpec, *, predicate: Predicate | None, projection: list[str],
                page_token: str | None, budget: "RemoteBudget") -> Page: ...

    # capability count (phase 4): an independent count request, the remote witness (§11.6); None when
    # the predicate cannot be expressed to the source
    def count(self, spec: LayoutSpec, *, predicate: Predicate | None, budget: "RemoteBudget") -> int | None: ...

    @classmethod
    def conformance_cases(cls) -> Any: ...


class StatisticPlugin(Protocol):
    kind: ClassVar[str] = "statistic"
    name: ClassVar[str]
    version: ClassVar[str]
    api: ClassVar[int] = API_VERSION
    capabilities: ClassVar[frozenset[str]] = frozenset()
    requires: ClassVar[tuple[str, ...]] = ()

    def sort_key(self, column: str, spec: "MeasureSpec", direction: str | None) -> RankKey: ...

    def predicate(self, column: str, op: str, value: Any, spec: "MeasureSpec", confirmed: ConfirmedFacts | None,
                  fixed_scope: Mapping[str, Any]) -> Predicate:
        """Raises UnsupportedFilter (I9, scale; a threshold on a comparable_within measure whose
        group is not fixed by ``fixed_scope``)."""
        ...

    def bounds(self, spec: "MeasureSpec", constraints: Sequence["ConstraintSpec"]) -> dict[str, Any]: ...
    def validate(self, stats: ColumnStats, snapshot: ValueSnapshot | None, spec: "MeasureSpec") -> ConfirmResult: ...

    def aggregate(self, values: Sequence[float | None], how: str, spec: "MeasureSpec",
                  keys: Sequence[Any] | None = None) -> AggResult:
        """Empty -> None; deduplicates on level keys."""
        ...

    def comparable(self, a: Mapping[str, Any], b: Mapping[str, Any], spec: "MeasureSpec") -> bool: ...
    def describe(self, spec: "MeasureSpec") -> str: ...
    def family(self, spec: "MeasureSpec") -> FamilySpec | None: ...

    # capability test
    def test(self, overlap: int, set_n: int, query_n: int, universe_n: int, spec: "MeasureSpec") -> float: ...

    # capability paired
    def aggregate_pair(self, times: Sequence[float | None], events: Sequence[bool | None], how: str,
                       spec: "MeasureSpec") -> AggResult: ...

    # capability veto_labels
    def vetoed_companions(self, spec: "MeasureSpec") -> tuple[str, ...]: ...

    @classmethod
    def conformance_cases(cls) -> Any: ...


class IdentifierPlugin(Protocol):
    kind: ClassVar[str] = "identifier"
    name: ClassVar[str]
    version: ClassVar[str]
    api: ClassVar[int] = API_VERSION
    capabilities: ClassVar[frozenset[str]] = frozenset()
    requires: ClassVar[tuple[str, ...]] = ()
    id_type: ClassVar[str]                             # "ensembl_gene"
    canonical: ClassVar[str]                           # regex of the canonical form (may depend on options)
    examples: ClassVar[tuple[str, ...]]
    cardinality: ClassVar[Literal["one", "many"]] = "one"
    overlaps: ClassVar[frozenset[str]] = frozenset()   # kinds that legitimately share syntax

    def configure(self, options: Mapping[str, Any], universe_sample: Sequence[str] | None) -> "IdentifierPlugin": ...
    def normalize(self, raw: str) -> Normalized | Rejected: ...           # agent input; syntactic only
    def normalize_stored(self, stored: str) -> Normalized | Rejected: ...  # stored values (index build)
    def looks_like(self, raw: str) -> float: ...                          # 0..1, for wrong-kind diagnostics
    def label_key(self, raw: str) -> str: ...                             # NFKC + casefold key for label lookup
    def describe(self) -> str: ...                                        # <= 200 chars, includes an example

    @classmethod
    def conformance_cases(cls) -> Any: ...


class EnvelopePlugin(Protocol):
    """Decodes one tool result into rows, a total and a not-found verdict (phase 4, F21).

    ``spec`` holds the overlay's codec options (JSONPaths and predicates for the builtin ``jsonpath``
    envelope). A plugin never invents rows: a shape it does not understand is ``unparsed``."""

    kind: ClassVar[str] = "envelope"
    name: ClassVar[str]
    version: ClassVar[str]
    api: ClassVar[int] = API_VERSION
    capabilities: ClassVar[frozenset[str]]
    requires: ClassVar[tuple[str, ...]] = ()

    def decode(self, raw_text: str | None, structured: Any, spec: Mapping[str, Any]) -> ParsedResult: ...

    @classmethod
    def conformance_cases(cls) -> Any: ...


class AcquisitionPlugin(Protocol):
    """A transport of ``vbt data acquire`` (harness side): lists what a source publishes and streams one file.

    ``options`` are the descriptor's ``acquisition.transport.options`` (``{release}`` substituted); ``session``
    is the engine's HTTP session (:class:`vbt.datalayer.plugins.acquisition.HttpSession`: retries, and the
    proxy and CA settings of the environment), so a plugin opens no connection of its own and tests serve it
    from a local fixture server. A plugin is generic (a protocol or a hosting platform), never one dataset:
    everything specific to a source is in its descriptor."""

    kind: ClassVar[str] = "acquisition"
    name: ClassVar[str]
    version: ClassVar[str]
    api: ClassVar[int] = API_VERSION
    capabilities: ClassVar[frozenset[str]]
    requires: ClassVar[tuple[str, ...]] = ()

    def listing(self, options: Mapping[str, Any], session: Any, *, index_cache: Any = None) -> list[RemoteFile]:
        """Every file the source publishes under ``options``, sorted by path. ``index_cache`` (capability
        ``index``) is a directory where the index document is kept and read back when it still verifies.
        Raises :class:`AcquisitionError`, never returns a partial listing."""
        ...

    def fetch(self, file: RemoteFile, session: Any, *, offset: int = 0) -> Iterator[bytes]:
        """The file's bytes from ``offset`` (capability ``range``; without it only offset 0)."""
        ...

    def describe(self, options: Mapping[str, Any]) -> str:
        """Where the files come from, in one line (at most 200 characters)."""
        ...

    @classmethod
    def conformance_cases(cls) -> Any: ...


# ---------------------------------------------------------------------------
# Convenience bases (optional; plugins only have to satisfy the protocol)
# ---------------------------------------------------------------------------

class PluginBase:
    """Defaults every plugin kind shares."""

    kind: ClassVar[str]
    name: ClassVar[str]
    version: ClassVar[str] = "1.0"
    api: ClassVar[int] = API_VERSION
    capabilities: ClassVar[frozenset[str]] = frozenset()
    requires: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def conformance_cases(cls) -> Any:
        return ()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.kind}/{self.name} {self.version}>"


class IdentifierBase(PluginBase):
    """Defaults for identifier plugins: ``configure`` returns a configured copy holding ``options``,
    ``normalize_stored`` normalises like agent input, ``label_key`` is NFKC + casefold,
    ``looks_like`` scores the canonical pattern."""

    kind: ClassVar[str] = "identifier"
    id_type: ClassVar[str]
    canonical: ClassVar[str]
    examples: ClassVar[tuple[str, ...]] = ()
    cardinality: ClassVar[Literal["one", "many"]] = "one"
    overlaps: ClassVar[frozenset[str]] = frozenset()

    def __init__(self) -> None:
        self.options: dict[str, Any] = {}

    def configure(self, options: Mapping[str, Any], universe_sample: Sequence[str] | None) -> Self:
        other = copy.copy(self)
        other.options = dict(options or {})
        return other

    @property
    def pattern(self) -> str:
        """The canonical regex in effect (an ``options.canonical`` overrides the class value)."""
        return str(self.options.get("canonical") or self.canonical)

    def normalize(self, raw: str) -> Normalized | Rejected:
        value = str(raw).strip()
        steps = ("strip",) if value != raw else ()
        if re.fullmatch(self.pattern, value):
            return Normalized(value, steps)
        return Rejected(f"not a {self.id_type} identifier (expected {self.pattern})")

    def normalize_stored(self, stored: str) -> Normalized | Rejected:
        return self.normalize(stored)

    def looks_like(self, raw: str) -> float:
        return 1.0 if re.fullmatch(self.pattern, str(raw).strip()) else 0.0

    def label_key(self, raw: str) -> str:
        return unicodedata.normalize("NFKC", str(raw)).strip().casefold()

    def describe(self) -> str:
        example = self.examples[0] if self.examples else ""
        return f"{self.id_type} identifier, e.g. {example}"[:200]


class EnvelopeBase(PluginBase):
    """Defaults for envelope plugins."""

    kind: ClassVar[str] = "envelope"


class AcquisitionBase(PluginBase):
    """Defaults for acquisition plugins: ``fetch`` streams ``file.url`` from ``offset`` with a ``Range`` request
    (the session refuses a server that ignores the range)."""

    kind: ClassVar[str] = "acquisition"

    def fetch(self, file: RemoteFile, session: Any, *, offset: int = 0) -> Iterator[bytes]:
        return session.stream_bytes(file.url, offset=offset)

    def describe(self, options: Mapping[str, Any]) -> str:
        return f"{self.name}: {dict(options)}"[:200]
