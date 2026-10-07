"""cBioPortal keys: ``cbio_cancer_type`` and ``cbio_study`` (lower case), ``cbio_sample`` and
``cbio_patient`` (syntax only).

Sample and patient IDs are checked for syntax only: TCGA barcodes have a strict form (a patient is
``TCGA-XX-XXXX``, a sample adds ``-NN[A]``, and each kind rejects the other's barcode); other
studies use free IDs (``P-0000004-T01-IM3``). No upstream tool lists a study's samples, so existence
is decided per returned row (``ResultSpec.exists_when``, W6), not here.
"""

from __future__ import annotations

import re

from ..base import Normalized, Rejected
from ..registry import register
from . import TextIdentifier, Trace, text_overlaps

_GENERIC = r"(?!TCGA-)[A-Za-z0-9][A-Za-z0-9._\-]*"


class _LowerKey(TextIdentifier):
    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        bad = self._structured(t.value, stored)         # structured syntax is judged before case folding
        if bad is not None:
            return bad
        t.lower()
        return self._check(t, stored)


@register
class CbioCancerType(_LowerKey):
    name = id_type = "cbio_cancer_type"
    canonical = r"^[a-z0-9][a-z0-9_\-]*$"
    examples = ("brca", "luad")
    overlaps = text_overlaps("cbio_cancer_type")
    description = "cBioPortal cancer type ID (lower case)"
    cases = (
        {"raw": "BRCA", "expected": "brca", "steps": ["lower"]},
        {"raw": "breast cancer", "rejected": True},
        {"raw": "GO_0005737", "rejected": True, "looks_like": "go"},
    )


@register
class CbioStudy(_LowerKey):
    name = id_type = "cbio_study"
    canonical = r"^[a-z0-9][a-z0-9_\-]*$"
    examples = ("brca_tcga_pan_can_atlas_2018", "msk_impact_2017")
    overlaps = text_overlaps("cbio_study")
    description = "cBioPortal study ID (lower case)"
    cases = (
        {"raw": "BRCA_TCGA_PAN_CAN_ATLAS_2018", "expected": "brca_tcga_pan_can_atlas_2018", "steps": ["lower"]},
        {"raw": "FINNGEN_R12_I9_HYPTENS", "rejected": True, "looks_like": "gwas_study"},
        {"raw": "brca tcga", "rejected": True},
    )


class _Barcode(TextIdentifier):
    tcga: str = ""                                     # the strict TCGA form of this kind
    other: tuple[str, str] = ("", "")                  # (pattern, kind) of the sibling kind's TCGA form

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        if t.value[:5].upper() == "TCGA-":
            t.upper()
            if re.fullmatch(self.tcga, t.value):
                return t.done()
            pattern, kind = self.other
            if re.fullmatch(pattern, t.value):
                return Rejected(f"{t.value} is a TCGA {kind.split('_')[1]} barcode, not a {self.id_type}", (kind,))
            return Rejected(f"{t.value} is not a TCGA {self.id_type.split('_')[1]} barcode ({self.examples[0]})")
        return self._check(t, stored)


_PATIENT = r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}"
_SAMPLE = r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2}[A-Z]?"


@register
class CbioSample(_Barcode):
    name = id_type = "cbio_sample"
    canonical = rf"^(?:{_SAMPLE}|{_GENERIC})$"
    examples = ("TCGA-A1-A0SB-01", "P-0000004-T01-IM3")
    overlaps = text_overlaps("cbio_sample")
    tcga = _SAMPLE
    other = (_PATIENT, "cbio_patient")
    description = "cBioPortal sample ID (syntax only; TCGA sample barcodes are strict)"
    cases = (
        {"raw": "tcga-a1-a0sb-01", "expected": "TCGA-A1-A0SB-01", "steps": ["upper"]},
        {"raw": "TCGA-A1-A0SB", "rejected": True, "looks_like": "cbio_patient"},
        {"raw": "P-0000004-T01-IM3", "expected": "P-0000004-T01-IM3"},
        {"raw": "ACH-000001", "rejected": True, "looks_like": "depmap_cell_line"},
    )


@register
class CbioPatient(_Barcode):
    name = id_type = "cbio_patient"
    canonical = rf"^(?:{_PATIENT}|{_GENERIC})$"
    examples = ("TCGA-A1-A0SB", "P-0000004")
    overlaps = text_overlaps("cbio_patient")
    tcga = _PATIENT
    other = (_SAMPLE, "cbio_sample")
    description = "cBioPortal patient ID (syntax only; TCGA patient barcodes are strict)"
    cases = (
        {"raw": "TCGA-A1-A0SB", "expected": "TCGA-A1-A0SB", "steps": []},
        {"raw": "TCGA-A1-A0SB-01", "rejected": True, "looks_like": "cbio_sample"},
        {"raw": "P-0000004", "expected": "P-0000004"},
    )
