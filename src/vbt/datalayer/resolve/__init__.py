"""Identifier resolution (§11.5): sidecar indexes, the closed rule grammar and the resolver. No pyarrow.

* ``index.py``: :class:`IndexStore` locates and loads the gzip TSV sidecars the data child builds
  (``<cache_dir>/<source>/<fingerprint>/index/<id_type>.tsv.gz``); :class:`ResolverIndex` answers
  lookups, membership, stored forms, native labels, families and edit-distance suggestions.
* ``rules.py``: :func:`parse_rule`/:func:`format_rule` for the closed grammar and the per-id_type
  rule lists (:func:`default_rules`, :func:`rules_for`).
* ``resolver.py``: :class:`Resolver` turns an argument value into one canonical key of the bound
  column's id_type, or a typed outcome (``ambiguous``, ``not_found``, ``rejected``, ``unknown``);
  :func:`error_for` and :func:`list_error` give the §12.1 errors.
"""

from __future__ import annotations

from .index import HEADER, Entry, IndexFormatError, IndexMissing, IndexStore, ResolverIndex, read_sidecar, write_sidecar
from .resolver import (
    EXISTENCE_MODES,
    ResolutionResult,
    Resolver,
    ResolverConfigError,
    error_for,
    list_error,
)
from .rules import Rule, RuleError, default_rules, format_rule, parse_entry_rule, parse_rule, rules_for

__all__ = [
    "HEADER", "Entry", "IndexFormatError", "IndexMissing", "IndexStore", "ResolverIndex", "read_sidecar",
    "write_sidecar", "EXISTENCE_MODES", "ResolutionResult", "Resolver", "ResolverConfigError", "error_for",
    "list_error", "Rule", "RuleError", "default_rules", "format_rule", "parse_entry_rule", "parse_rule", "rules_for",
]
