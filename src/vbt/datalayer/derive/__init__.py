"""Derivations from roles (§10): argument schemas and agent-facing text for bound tools. No pyarrow.

* :mod:`.schema` annotates a tool's upstream input schema from its binding (identifier kinds,
  vocabularies as enums, auto-derived gateway-only scope arguments, selector and order enums,
  threshold bounds, limit bounds, list sizes, engine documentation, anchors).
* :mod:`.text` builds the description: upstream's first sentence plus a generated block
  (source, grain, key, arguments, order, totals, unknowns, levels, cutoffs, coverage, evidence
  nature), capped at ``data.derive.description_max_chars``.

Phase 2 adds ``tools.py`` (native tools from roles). Generic tools are never rewritten.
"""

from __future__ import annotations

from .schema import annotate_schema, auto_scope_args
from .text import describe_tool, unavailable_text

__all__ = ["annotate_schema", "auto_scope_args", "describe_tool", "unavailable_text"]
