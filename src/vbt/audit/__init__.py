"""The run audit record: storage, claims, plan, provenance, rendering and re-execution.

Modules (all stdlib-only, so any machine can audit a run directory):

* ``storage``    atomic writes, the run lock, layout rules and snapshots
* ``claims``     claim-evidence validation (``validate_claims``) and ``load_claims``
* ``plan``       ``validate_plan``, ``reconcile`` and ``render_plan_md``
* ``provenance`` the provenance index built from ``logs/trace.jsonl``
* ``render``     ``[[claim:ID]]`` anchor stripping/numbering and the claims appendix
* ``rerun``      ``rerun_scripts``: re-execute agent scripts in a scratch copy

``vbt.session.Run`` writes the record; ``vbt.verify.verify_run`` checks it.
"""

from __future__ import annotations

__all__ = ["storage", "claims", "plan", "provenance", "render", "rerun"]
