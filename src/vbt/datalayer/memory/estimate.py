"""Memory estimates from footer statistics (§10.4). No pyarrow.

Two estimators, both fed by the data child's ``_stats`` payload (:class:`vbt.datalayer.ipc.
TableStatsModel`, or its JSON form):

- **Upstream pandas load** (admission of ``full_table`` reads by upstream servers)::

      peak(t) = (sum_leaves uncompressed_bytes x expansion(kind)
                 + sum_leaves num_values x object_overhead(kind, depth)
                 + sum_struct_containers items x struct_item) x fragmentation

  pandas creates one Python object per nested item, so footer bytes alone understate the peak
  about 9x on a three-level fixture (VERIFIED: 11.6 MB of leaf bytes, +1,001 MB peak RSS against
  a 107 MB bytes-only estimate). Seed overheads (``data.memory.object_overhead_bytes``): string
  50 B per value, nested item 120 B per value and list level (``max_rep_level``), and one dict of
  240 B per struct item.
- **Data-child Arrow scan**: projected leaf bytes x a decode factor; this is what
  ``data.witness.max_scan_bytes`` is compared with.

Phase 4 (F19) replaces the seeds per table in tiers (:mod:`.calibrate`): a **sample** calibration
(``<cache>/<source>/<fingerprint>/calibration.json``, measured pandas bytes per row of one to three
row groups) is used instead of the seed model for that fingerprint, and **measured** feedback from
reaper status files (``memory_feedback.json``: peak RSS of real cold loads over their estimates)
multiplies either. :meth:`MemoryEstimator.tier` says which tier an estimate came from.

All results are bytes (``MB`` = 2**20 converts).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..roles import LIST, Path, PathError, Segment, format_path, parse_path

__all__ = ["MB", "DECODE_FACTORS", "STRING_TYPES", "TIERS", "Leaf", "MemoryEstimator", "leaves_of"]

MB = 1024 * 1024

#: Arrow bytes per footer (uncompressed) byte when the data child decodes a leaf. Seeds; a table's
#: sample calibration (phase 4, F19) replaces them for its fingerprint.
DECODE_FACTORS: dict[str, float] = {"flat": 1.1, "string": 1.5, "nested": 2.0, "dense_matrix": 1.0,
                                    "sparse_matrix": 1.5}

STRING_TYPES = frozenset({"string", "large_string", "utf8", "large_utf8", "binary", "large_binary", "str",
                          "object"})

#: Rows a bounded upstream scan holds at least once (one batch of ``scan_top_rows``).
SCAN_BATCH_ROWS = 1024

#: Where an estimate's factors came from, most trusted first.
TIERS = ("measured", "sample", "seed")


@dataclass(frozen=True)
class Leaf:
    """One leaf column of a table's ``_stats`` payload."""

    path: str
    uncompressed_bytes: int = 0
    num_values: int | None = None
    max_rep_level: int = 0
    kind: str = "flat"
    storage_type: str | None = None
    segments: tuple[Segment, ...] = field(default=(), compare=False, repr=False)

    @property
    def depth(self) -> int:
        """List levels above the leaf (``max_rep_level``; the path's ``[]`` count when larger)."""
        brackets = sum(1 for s in self.segments for b in s.brackets if b == LIST)
        return max(int(self.max_rep_level or 0), brackets)

    @property
    def is_string(self) -> bool:
        st = (self.storage_type or "").lower()
        return self.kind == "string" or st in STRING_TYPES or st.startswith(("string", "large_string", "utf8"))


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _segments(path: str) -> tuple[Segment, ...]:
    try:
        return parse_path(path).segments
    except PathError:
        return tuple(Segment(p) for p in path.split(".") if p)


def leaves_of(table_stats: Any) -> list[Leaf]:
    """The leaves of a ``TableStatsModel`` (or its JSON form), sorted by path."""
    columns = _get(table_stats, "columns", None) or {}
    out = []
    for path, col in sorted(columns.items()):
        out.append(Leaf(
            path=str(path), uncompressed_bytes=int(_get(col, "uncompressed_bytes", 0) or 0),
            num_values=_get(col, "num_values"), max_rep_level=int(_get(col, "max_rep_level", 0) or 0),
            kind=str(_get(col, "kind", "flat") or "flat"), storage_type=_get(col, "storage_type"),
            segments=_segments(str(path))))
    return out


def _selected(leaf: Leaf, wanted: Iterable[str]) -> bool:
    """True when ``leaf`` lies under one of the projected column paths (segment boundary match)."""
    for w in wanted:
        if leaf.path == w or leaf.path.startswith(w + ".") or leaf.path.startswith(w + "["):
            return True
        stripped = format_path(Path(tuple(Segment(s.name) for s in leaf.segments)))
        if stripped == w or stripped.startswith(w + "."):
            return True
    return False


class MemoryEstimator:
    """Byte estimates for admission, scan budgets and limit maxima (§10.4)."""

    def __init__(self, expansion: Mapping[str, float] | None = None, fragmentation: float | None = None,
                 object_overhead_bytes: Mapping[str, int] | None = None, safety: float = 1.3,
                 decode: Mapping[str, float] | None = None, *,
                 calibrations: Mapping[str, Mapping[str, Any]] | None = None,
                 feedback: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        exp = dict(expansion or {"flat": 1.5, "string": 3.5, "nested": 8.0})
        self.fragmentation = float(fragmentation if fragmentation is not None else exp.pop("fragmentation", 1.15))
        exp.pop("fragmentation", None)
        self.expansion = {str(k): float(v) for k, v in exp.items()}
        self.object_overhead = {"string": 50, "nested_item": 120, "struct_item": 240,
                                **{str(k): int(v) for k, v in (object_overhead_bytes or {}).items()}}
        self.safety = float(safety)
        self.decode = {**DECODE_FACTORS, **{str(k): float(v) for k, v in (decode or {}).items()}}
        self.calibrations: dict[str, dict[str, Any]] = {str(k): dict(v) for k, v in (calibrations or {}).items()}
        self.feedback: dict[str, dict[str, Any]] = {str(k): dict(v) for k, v in (feedback or {}).items()}

    @classmethod
    def from_settings(cls, settings: Any, *, load_calibrations: bool = False) -> "MemoryEstimator":
        """From ``DataSettings`` (``data.memory.*``); with ``load_calibrations`` the sample calibrations
        and measured feedback under ``data.cache_dir`` are loaded too."""
        mem = settings.memory
        exp = dict(mem.expansion)
        cals = fb = None
        if load_calibrations:
            from .calibrate import load_calibrations as _cals, load_feedback

            cache = getattr(settings, "cache_dir", None)
            cals, fb = _cals(cache), load_feedback(cache)
        return cls(expansion=exp, fragmentation=exp.get("fragmentation"),
                   object_overhead_bytes=dict(mem.object_overhead_bytes), safety=float(mem.estimate_safety),
                   calibrations=cals, feedback=fb)

    # ------------------------------------------------------------------ tiers

    def add_calibration(self, cal: Mapping[str, Any], fingerprint: str | None = None) -> None:
        """Use a sample calibration (:func:`.calibrate.fit` record) for its table fingerprint."""
        fp = fingerprint or cal.get("fingerprint")
        if not fp:
            raise ValueError("a calibration needs the fingerprint of the table it measured")
        self.calibrations[str(fp)] = dict(cal)

    def add_feedback(self, key: str, factor: float) -> None:
        """Use a measured feedback factor (peak RSS over the estimate) for a fingerprint or table name."""
        self.feedback[str(key)] = {"factor": float(factor)}

    def _key(self, table_stats: Any) -> str | None:
        fp = _get(table_stats, "fingerprint")
        return str(fp) if fp else None

    def calibration_for(self, table_stats: Any) -> dict[str, Any] | None:
        key = self._key(table_stats)
        return self.calibrations.get(key) if key else None

    def feedback_factor(self, table_stats: Any, table: str | None = None) -> float | None:
        for key in (self._key(table_stats), table):
            entry = self.feedback.get(key) if key else None
            if entry and isinstance(entry.get("factor"), (int, float)) and entry["factor"] > 0:
                return float(entry["factor"])
        return None

    def tier(self, table_stats: Any, table: str | None = None) -> str:
        """``measured`` (feedback from real loads), ``sample`` (a calibration of this fingerprint) or ``seed``."""
        if self.feedback_factor(table_stats, table) is not None:
            return "measured"
        return "sample" if self.calibration_for(table_stats) is not None else "seed"

    # ------------------------------------------------------------------ per leaf

    def expansion_of(self, leaf: Leaf) -> float:
        """pandas bytes per footer byte: nested leaves use the nested factor, matrices the flat one."""
        kind = leaf.kind
        if leaf.depth > 0:
            kind = "nested"
        elif kind in ("dense_matrix", "sparse_matrix"):
            kind = "flat"
        elif leaf.is_string:
            kind = "string"
        return self.expansion.get(kind, self.expansion.get("flat", 1.5))

    def overhead(self, leaf: Leaf) -> int:
        """Python object bytes per leaf value: a str object, plus one nested item per list level."""
        per = self.object_overhead["string"] if leaf.is_string else 0
        return per + self.object_overhead["nested_item"] * leaf.depth

    @staticmethod
    def _num_values(leaf: Leaf, rows: int | None) -> int:
        if leaf.num_values is not None:
            return int(leaf.num_values)
        if leaf.depth == 0 and rows is not None:
            return int(rows)
        return 0

    def struct_items(self, leaves: list[Leaf], rows: int | None = None) -> dict[str, int]:
        """Items per struct container (``a``, ``a[]``, ``a[].b[]``, ...): each becomes one dict.

        A container's item count is the ``num_values`` of a field directly inside it (one value
        per item, nulls included); without one, the smallest count among its deeper leaves.
        """
        direct: dict[str, int] = {}
        deeper: dict[str, int] = {}
        for leaf in leaves:
            segs = leaf.segments
            n = self._num_values(leaf, rows)
            for i in range(len(segs) - 1):
                if not segs[i].name:
                    continue
                key = format_path(Path(segs[: i + 1]))
                rest = segs[i + 1:]
                if len(rest) == 1 and not rest[0].brackets:
                    direct[key] = max(direct.get(key, 0), n)
                else:
                    deeper[key] = min(deeper.get(key, n), n)
        return {k: direct.get(k, deeper.get(k, 0)) for k in sorted(set(direct) | set(deeper))}

    # ------------------------------------------------------------------ estimators

    def bytes_term(self, table_stats: Any) -> float:
        """``sum_leaves uncompressed_bytes x expansion(kind)`` (the revision-1 estimate without
        fragmentation)."""
        return sum(leaf.uncompressed_bytes * self.expansion_of(leaf) for leaf in leaves_of(table_stats))

    def objects_term(self, table_stats: Any) -> float:
        """``sum_leaves num_values x overhead + sum_containers items x struct_item``."""
        leaves = leaves_of(table_stats)
        rows = _get(table_stats, "rows")
        per_value = sum(self._num_values(leaf, rows) * self.overhead(leaf) for leaf in leaves)
        structs = sum(self.struct_items(leaves, rows).values()) * self.object_overhead["struct_item"]
        return float(per_value + structs)

    def peak_upstream(self, table_stats: Any, table: str | None = None) -> int:
        """Peak bytes of an upstream server loading the whole table into pandas: the sample calibration's
        measured bytes per row when this fingerprint has one, else the seed model; times the measured
        feedback factor when real loads were observed."""
        if table_stats is None:
            return 0
        cal = self.calibration_for(table_stats)
        rows = _get(table_stats, "rows")
        if cal is not None and cal.get("bytes_per_row") and rows is not None:
            base = float(cal["bytes_per_row"]) * int(rows)
        else:
            base = self.bytes_term(table_stats) + self.objects_term(table_stats)
        factor = self.feedback_factor(table_stats, table)
        if factor is not None:
            base *= factor
        return int(math.ceil(base * self.fragmentation))

    def peak_arrow_scan(self, table_stats: Any, leaves: Iterable[str] | None = None) -> int:
        """Decoded Arrow bytes of the data child scanning the projected ``leaves`` (all when None),
        before any partition, row-group or index pruning."""
        if table_stats is None:
            return 0
        wanted = list(leaves) if leaves is not None else None
        cal = self.calibration_for(table_stats)
        decode = {**self.decode, **{str(k): float(v) for k, v in ((cal or {}).get("decode") or {}).items()}}
        total = 0.0
        for leaf in leaves_of(table_stats):
            if wanted is not None and not _selected(leaf, wanted):
                continue
            kind = "nested" if leaf.depth > 0 else ("string" if leaf.is_string else leaf.kind)
            total += leaf.uncompressed_bytes * decode.get(kind, decode["flat"])
        return int(math.ceil(total))

    def transient(self, table_stats: Any, predicate_selectivity: float | None = None) -> int:
        """Bytes an upstream bounded scan holds: the matching share of the pandas peak, and at
        least one scan batch. ``None`` selectivity is unknown and counts as the whole table."""
        if table_stats is None:
            return 0
        peak = self.peak_upstream(table_stats)
        share = 1.0 if predicate_selectivity is None else min(1.0, max(0.0, float(predicate_selectivity)))
        rows = _get(table_stats, "rows")
        if rows:
            share = max(share, min(1.0, SCAN_BATCH_ROWS / float(rows)))
        return int(math.ceil(peak * share))

    def row_bytes(self, grain: str | None, table_stats: Any) -> int | None:
        """Result bytes per row at ``grain`` (``row``, an item table or ``cell``): the measured
        ``row_bytes_p99`` of that grain. None when not measured: the stored-row mean is no
        substitute (one disease row is 307 KB while the mean is 1.1 KB)."""
        p99 = _get(table_stats, "row_bytes_p99", None) or {}
        value = p99.get(grain or "row")
        return int(value) if value else None

    def max_rows(self, grain: str | None, table_stats: Any, max_result_bytes: int) -> int | None:
        """``limit.maximum = floor(max_result_bytes / row_bytes_p99)`` (at least 1), or None."""
        rb = self.row_bytes(grain, table_stats)
        if not rb:
            return None
        return max(1, int(max_result_bytes) // rb)

    @staticmethod
    def densify(n_obs: int, n_vars: int, itemsize: int = 8) -> int:
        """Bytes of a dense ``n_obs x n_vars`` matrix (``X.toarray()``, ``read_h5ad`` without backing)."""
        return int(n_obs) * int(n_vars) * int(itemsize)

    def peak_all(self, stats: Iterable[Any]) -> int:
        """``sum peak_upstream`` over several tables' stats (``None`` entries count 0)."""
        return sum(self.peak_upstream(s) for s in stats if s is not None)
