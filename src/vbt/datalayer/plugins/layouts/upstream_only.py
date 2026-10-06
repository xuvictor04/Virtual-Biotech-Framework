"""``upstream_only``: remote sources served only by their upstream tools in phase 1 (§9.4).

There are no fragments to read (ClinicalTrials.gov, cBioPortal, Census until the ``soma``
layout of phase 4). The probe reports ``served by upstream tools only`` as a passing finding,
so readiness never mistakes the missing local files for an outage, and ``as_of`` is the call
time (UTC) unless the result carries its own. Signature and fingerprint are constant per table.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, ClassVar

from ..base import CheckItem, Fragment, LayoutSpec, Manifest, PluginBase
from ..registry import register


@register
class UpstreamOnlyLayout(PluginBase):
    kind: ClassVar[str] = "layout"
    name: ClassVar[str] = "upstream_only"
    version: ClassVar[str] = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset({"upstream_only"})

    def fragments(self, root: str, spec: LayoutSpec) -> list[Fragment]:
        return []

    def partition_columns(self, spec: LayoutSpec) -> dict[str, str]:
        return {}

    def signature(self, root: str, spec: LayoutSpec) -> str:
        return "sig1:upstream_only:" + hashlib.sha256(str(spec.table).encode()).hexdigest()[:16]

    def fingerprint(self, frags: list[Fragment], manifest: Manifest | None) -> str:
        return "fp1:upstream_only"

    def partition_fingerprints(self, frags: list[Fragment], manifest: Manifest | None) -> dict[str, str]:
        return {}

    def probe(self, root: str, spec: LayoutSpec, manifest: Manifest | None) -> list[CheckItem]:
        return [CheckItem("upstream_only", True, "served by upstream tools only", level="info")]

    def as_of(self, root: str, spec: LayoutSpec) -> str | None:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import LayoutCases

        return LayoutCases(tree="none", path=None)
