"""``ot_variant``: Open Targets variant IDs ``CHROM_POS_REF_ALT`` (``19_44908822_C_T``).

``chr19:44908822:C:T`` and ``19-44908822-C-T`` are folded: the ``chr`` prefix is removed
(``strip_chr``), ``:`` and ``-`` become ``_`` (``separator_to_underscore``) and alleles are
upper-cased. The universe (millions of variants) is resolved remotely (``index: remote``).
"""

from __future__ import annotations

import re

from ..base import Normalized, Rejected
from ..registry import register
from . import KeyIdentifier, Trace

CHROMS = r"(?:[1-9]|1\d|2[0-2]|X|Y|MT)"
_LOOSE = re.compile(r"(chr)?(\w{1,2})([_:\-])(\d+)[_:\-]([ACGTNacgtn]+)[_:\-]([ACGTNacgtn]+)", re.IGNORECASE)


@register
class OtVariant(KeyIdentifier):
    name = id_type = "ot_variant"
    canonical = rf"^{CHROMS}_\d+_[ACGTN]+_[ACGTN]+$"
    examples = ("19_44908822_C_T", "1_154453788_CA_C")
    description = "Open Targets variant ID CHROM_POS_REF_ALT (GRCh38; chr and : or - separators folded)"
    cases = (
        {"raw": "19_44908822_C_T", "expected": "19_44908822_C_T", "steps": []},
        {"raw": "chr19:44908822:C:T", "expected": "19_44908822_C_T", "steps": ["strip_chr", "separator_to_underscore"]},
        {"raw": "19-44908822-c-t", "expected": "19_44908822_C_T", "steps": ["separator_to_underscore", "upper"]},
        {"raw": "chrX_1000_A_G", "expected": "X_1000_A_G", "steps": ["strip_chr"]},
        {"raw": "chr19:44908822", "rejected": True},
        {"raw": "rs7412", "rejected": True, "looks_like": "rsid"},
        {"raw": "23_100_A_C", "rejected": True},
    )

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        t = Trace(text).strip()
        m = _LOOSE.fullmatch(t.value)
        if m is None:
            return self.reject(t.value)
        chrom, pos, ref, alt = m.group(2), m.group(4), m.group(5), m.group(6)
        if m.group(1):
            t.apply("strip_chr", t.value[len(m.group(1)):])
        t.apply("separator_to_underscore", "_".join((chrom, pos, ref, alt)))
        t.apply("upper", "_".join((chrom.upper(), pos, ref.upper(), alt.upper())))
        return t.done() if self.matches(t.value) else self.reject(t.value, f"chromosome {chrom!r} is not 1-22, X, Y "
                                                                           "or MT")
