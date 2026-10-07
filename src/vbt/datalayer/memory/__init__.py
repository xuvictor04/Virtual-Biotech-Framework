"""Memory safety for MCP servers (§10.4, §14). No pyarrow.

- :mod:`.estimate` -- the two estimators (upstream pandas load with the ``num_values`` object
  term, data-child Arrow scan), per-grain row bytes, densification.
- :mod:`.ledger` -- tables resident per server generation; RSS from the reaper's status file.
- :mod:`.admission` -- admission before upstream calls: ``too_large``, learned refusals,
  recycle-to-evict with a thrash guard, the cold-call lock, count-first remote admission.
- :mod:`.crash` -- memory signatures, the reaper's exit marker, crash decisions.
"""

from __future__ import annotations

from .admission import Admission, AdmissionController, TableRead
from .crash import MEMORY_SIGNATURES, classify_error_text, crash_decision, is_memory_exit, parse_exit_marker
from .estimate import MB, MemoryEstimator
from .ledger import ResidencyLedger, read_status

__all__ = [
    "MB", "MemoryEstimator", "ResidencyLedger", "read_status", "TableRead", "Admission", "AdmissionController",
    "MEMORY_SIGNATURES", "classify_error_text", "parse_exit_marker", "is_memory_exit", "crash_decision",
]
