"""Builtin identifier plugins (§9.4, phase 1). Stdlib only; no pyarrow.

Each module of this package defines one family of identifier plugins decorated with
``@register``; discovery imports them all. The helpers here keep the families consistent:

* :class:`Trace` records the normalisation steps a plugin applies. Only steps that change the
  value are recorded, so ``ENSG00000169174.12`` reports ``strip_version`` alone and the resolver
  can name the rule ``normalized:strip_version`` (CT-1).
* :class:`KeyIdentifier` is the base of structured kinds: ``normalize`` (agent input) and
  ``normalize_stored`` (stored values, index build) share one ``_normalize(text, stored)`` hook,
  rejections carry ``looks_like`` hints from :data:`STRUCTURED`, and the I-1 cases are the
  class's ``cases``.
* :class:`TextIdentifier` is the base of free-syntax kinds (labels and opaque keys). They accept
  text, but never a value with the syntax of a structured kind in :data:`STRUCTURED` unless they
  declare that overlap: a gene symbol is never an accession, a drug name never a PMID (I-4).
  Free-syntax kinds overlap each other (:data:`TEXT_KINDS`).

Plugins that take YAML options (``prefixes``, ``canonical``) declare the ``options`` capability
and a representative configuration in ``example_options`` (and ``example_universe``), which the
conformance suite uses for I-3 and I-4.
"""

from __future__ import annotations

import copy
import re
import unicodedata
from typing import Any, ClassVar, Mapping, Self, Sequence

from ..base import IdentifierBase, Normalized, Rejected

__all__ = [
    "TEXT_KINDS", "STRUCTURED", "structured_hits", "text_overlaps", "as_text", "nfkc", "bare_pattern", "Trace",
    "KeyIdentifier", "TextIdentifier", "configured_examples", "set_attr",
]

#: Free-syntax kinds: labels and opaque keys whose syntax legitimately overlaps.
TEXT_KINDS: frozenset[str] = frozenset({
    "hgnc_symbol", "disease_name", "drug_name", "tahoe_drug", "tahoe_gene_name", "cbio_cancer_type", "cbio_study",
    "cbio_sample", "cbio_patient", "local_key",
})

_I = re.IGNORECASE
#: Structured identifier syntaxes of the builtin kinds (matched against the stripped value, case-insensitively,
#: with the input variants the plugins accept). Free-syntax kinds reject these values; rejections name them in
#: ``looks_like``.
STRUCTURED: tuple[tuple[str, re.Pattern[str]], ...] = tuple((kind, re.compile(rx, _I)) for kind, rx in (
    ("ensembl_gene", r"ENSG\d{11}(?:\.\d+)?(?:_PAR_Y)?"),
    ("ensembl_gene_mouse", r"ENSMUSG\d{11}(?:\.\d+)?"),
    ("ensembl_gene_any", r"ENS[A-Z]*G\d{11}(?:\.\d+)?"),
    ("ot_disease", r"(?:https?://\S+/)?(?:EFO|MONDO|Orphanet|HP|OTAR|DOID|OBA|GO|MP|NCIT|OBI|GSSO|PATO|OGMS|UBERON)"
                   r"[_:][A-Z0-9]+"),
    ("chembl_molecule", r"CHEMBL\d+"),
    ("inchikey", r"(?:InChIKey=)?[A-Z]{14}-[A-Z]{10}-[A-Z]"),
    ("go", r"(?:https?://\S+/)?GO[_:]\d{7}"),
    ("reactome", r"R-[A-Z]{3}-\d+(?:\.\d+)?"),
    ("so", r"SO[_:]\d{7}"),
    ("hpo", r"(?:https?://\S+/)?HP[_:]\d{7}"),
    ("pmid", r"(?:PMID:?\s*)?\d+"),
    ("pmcid", r"(?:PMCID:?\s*)?PMC\d+"),
    ("doi", r"(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?10\.\d{4,9}/\S+"),
    ("europepmc_ppr", r"PPR\d+"),
    ("nct_id", r"NCT\d+"),
    ("ot_variant", r"(?:chr)?(?:\d{1,2}|X|Y|MT?)[_:\-]\d+[_:\-][ACGTN]+[_:\-][ACGTN]+"),
    ("rsid", r"(?-i:rs)\d+|(?-i:RS)\d{3,}"),        # RS1 is a gene symbol
    ("study_locus_id", r"[0-9a-f]{32}"),
    ("gwas_study", r"GCST\d+|FINNGEN_R\d+_\w+"),
    ("depmap_cell_line", r"ACH-\d{6}"),
    ("uniprot_accession", r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}"),
    ("cellosaurus", r"CVCL[_:][A-Z0-9]{4}"),
    ("chromosome", r"(?:chr)?(?:\d{1,2}|X|Y|MT?)"),
    ("ncbi_taxon", r"(?:NCBITaxon[:_]|taxon:)?\d+"),
))


def structured_hits(value: str, exclude: Any = ()) -> tuple[str, ...]:
    """The structured kinds whose syntax ``value`` has (``exclude`` names kinds to ignore)."""
    skip = set(exclude)
    return tuple(kind for kind, rx in STRUCTURED if kind not in skip and rx.fullmatch(value))


def text_overlaps(name: str, *extra: str) -> frozenset[str]:
    """Overlaps of a free-syntax kind: every other free-syntax kind plus ``extra``."""
    return (TEXT_KINDS - {name}) | frozenset(extra)


def as_text(raw: Any) -> str:
    """Agent or stored input as text: integers render as digits (``9606`` -> ``"9606"``), None as ``""``."""
    if raw is None:
        return ""
    if isinstance(raw, bool):
        return str(raw)
    if isinstance(raw, int):
        return str(raw)
    if isinstance(raw, float) and raw.is_integer():
        return str(int(raw))
    return str(raw)


def nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def bare_pattern(rx: str) -> str:
    """``rx`` without its ``^``/``$`` anchors, for alternations."""
    return rx[1:-1] if rx.startswith("^") and rx.endswith("$") else rx


class Trace:
    """A value being normalised and the I-5 steps that changed it."""

    __slots__ = ("value", "steps")

    def __init__(self, value: str) -> None:
        self.value = value
        self.steps: list[str] = []

    def apply(self, step: str, value: str) -> "Trace":
        if value != self.value:
            self.value = value
            if step not in self.steps:
                self.steps.append(step)
        return self

    def strip(self) -> "Trace":
        return self.apply("strip", self.value.strip())

    def upper(self) -> "Trace":
        return self.apply("upper", self.value.upper())

    def lower(self) -> "Trace":
        return self.apply("lower", self.value.lower())

    def done(self) -> Normalized:
        return Normalized(self.value, tuple(self.steps))


class KeyIdentifier(IdentifierBase):
    """Base of the builtin plugins. Subclasses implement ``_normalize(text, stored)``."""

    version: ClassVar[str] = "1.0"
    description: ClassVar[str] = ""                    # describe(): this sentence plus an example
    cases: ClassVar[tuple[Mapping[str, Any], ...]] = ()   # I-1 cases (see conformance/identifier.py)
    example_options: ClassVar[Mapping[str, Any] | None] = None   # the representative configuration
    example_universe: ClassVar[tuple[str, ...] | None] = None

    @property
    def pattern(self) -> str:
        return self.canonical                          # configure() may set an instance value

    def configure(self, options: Mapping[str, Any], universe_sample: Sequence[str] | None) -> Self:
        """A configured copy holding ``options`` (subclasses extend it; the registered instance never changes)."""
        other = copy.copy(self)
        other.options = dict(options or {})
        return other

    # -- protocol -------------------------------------------------------------

    def normalize(self, raw: Any) -> Normalized | Rejected:
        text = as_text(raw)
        if not text.strip():
            return Rejected(f"empty value is not a {self.id_type} identifier")
        return self._normalize(text, False)

    def normalize_stored(self, stored: Any) -> Normalized | Rejected:
        text = as_text(stored)
        if not text.strip():
            return Rejected(f"empty value is not a {self.id_type} identifier")
        return self._normalize(text, True)

    def looks_like(self, raw: Any) -> float:
        return 1.0 if isinstance(self.normalize(raw), Normalized) else 0.0

    def describe(self) -> str:
        example = self.examples[0] if self.examples else ""
        text = self.description or f"{self.id_type} identifier"
        return f"{text}, e.g. {example}"[:200] if example else text[:200]

    @classmethod
    def conformance_cases(cls) -> tuple[Mapping[str, Any], ...]:
        return tuple(cls.cases)

    # -- helpers --------------------------------------------------------------

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        if re.fullmatch(self.canonical, t.value):
            return t.done()
        return self.reject(t.value)

    def matches(self, value: str) -> bool:
        return re.fullmatch(self.canonical, value) is not None

    def reject(self, value: str, reason: str | None = None) -> Rejected:
        hints = structured_hits(value, exclude={self.name, self.id_type})
        if reason is None:
            reason = f"not a {self.id_type} identifier (e.g. {self.examples[0]})" if self.examples else \
                f"not a {self.id_type} identifier"
        if hints and "looks like" not in reason:
            reason = f"{reason}; looks like {hints[0]}"
        return Rejected(reason, hints)


class TextIdentifier(KeyIdentifier):
    """Free-syntax kinds: any text matching ``canonical`` that has no structured kind's syntax (unless
    declared in ``overlaps``). ``looks_like`` stays low: accepting text is weak evidence of the kind."""

    #: Scored by looks_like when the value is accepted (below the resolver's 0.8 hint threshold).
    text_score: ClassVar[float] = 0.3

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        return self._check(t, stored)

    def _check(self, t: Trace, stored: bool) -> Normalized | Rejected:
        bad = self._structured(t.value, stored)
        if bad is not None:
            return bad
        if not re.fullmatch(self.canonical, t.value):
            return Rejected(self.syntax_reason(t.value))
        return t.done()

    def _structured(self, value: str, stored: bool) -> Rejected | None:
        """Agent input with a structured kind's syntax (not declared in ``overlaps``) is rejected. Stored
        values are whatever the source holds, so they are not judged by other kinds' syntax."""
        if stored:
            return None
        hits = structured_hits(value, exclude=set(self.overlaps) | {self.name, self.id_type})
        if hits:
            return Rejected(f"looks like a {hits[0]} identifier, not a {self.id_type}", hits)
        return None

    def syntax_reason(self, value: str) -> str:
        return f"not a {self.id_type} (e.g. {self.examples[0]})" if self.examples else f"not a {self.id_type}"

    def looks_like(self, raw: Any) -> float:
        return self.text_score if isinstance(self.normalize(raw), Normalized) else 0.0


def set_attr(plugin: Any, name: str, value: Any) -> None:
    """Set a per-instance value of a class-level attribute (``canonical``, ``examples``) on a configured copy."""
    setattr(plugin, name, value)


def configured_examples(options: Mapping[str, Any], defaults: Sequence[str], pattern: str) -> tuple[str, ...]:
    """Examples of a configured instance: ``options.examples``, else the defaults that still match."""
    given = options.get("examples")
    if given:
        return tuple(str(e) for e in given)
    return tuple(e for e in defaults if re.fullmatch(pattern, e))
