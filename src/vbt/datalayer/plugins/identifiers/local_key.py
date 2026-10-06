"""``local_key``: opaque, dataset-specific keys configured in YAML (G2: no Python per dataset).

```yaml
id_types:
  msigdb_set:     {plugin: local_key, options: {canonical: "^HALLMARK_[A-Z0-9_]+$", normalize: [strip, upper]}}
  chemical_probe: {plugin: local_key, options: {canonical: "^[A-Za-z0-9][A-Za-z0-9_.\\- ]*$"}, resolvable: false}
```

Options: ``canonical`` (required regex), ``normalize`` (ordered steps from the I-5 whitelist that
apply without further knowledge: strip, upper, lower, strip_version, strip_prefix (with
``strip_prefix: <text or list>``), strip_suffix (with ``strip_suffix: ...``),
separator_to_underscore, curie_colon_to_underscore, curie_underscore_to_colon; default
``[strip]``) and ``examples``. Unconfigured, the plugin accepts nothing. The identifier suite
treats each configured instance as its own kind accepting only its declared pattern (I-4), using
``example_options`` as the representative configuration.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar, Mapping, Self, Sequence

from ..base import Normalized, PluginError, Rejected
from ..registry import register
from . import KeyIdentifier, Trace, configured_examples, set_attr, text_overlaps

#: Steps a configured local key may apply.
LOCAL_STEPS: tuple[str, ...] = ("strip", "upper", "lower", "strip_version", "strip_prefix", "strip_suffix",
                                "separator_to_underscore", "curie_colon_to_underscore", "curie_underscore_to_colon")


def _texts(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    return (str(value),) if isinstance(value, str) else tuple(str(v) for v in value)


@register
class LocalKey(KeyIdentifier):
    name = id_type = "local_key"
    canonical = r"(?!)"                                # accepts nothing until configured
    examples = ("HALLMARK_APOPTOSIS", "HALLMARK_HYPOXIA")
    capabilities = frozenset({"options"})
    overlaps = text_overlaps("local_key")
    description = "dataset-specific key with a YAML-configured pattern (options.canonical)"
    example_options: ClassVar[Mapping[str, Any] | None] = {"canonical": "^HALLMARK_[A-Z0-9_]+$",
                                                           "normalize": ["strip", "upper"]}
    cases = (
        {"raw": "HALLMARK_APOPTOSIS", "rejected": True},
        {"raw": " hallmark_apoptosis", "expected": "HALLMARK_APOPTOSIS", "steps": ["strip", "upper"],
         "options": {"canonical": "^HALLMARK_[A-Z0-9_]+$", "normalize": ["strip", "upper"]}, "requires": "options"},
        {"raw": "KEGG_APOPTOSIS", "rejected": True,
         "options": {"canonical": "^HALLMARK_[A-Z0-9_]+$", "normalize": ["strip", "upper"]}, "requires": "options"},
        {"raw": "PA:PA166104996", "expected": "PA166104996", "steps": ["strip_prefix"],
         "options": {"canonical": "^PA\\d+$", "normalize": ["strip", "strip_prefix"], "strip_prefix": "PA:"},
         "requires": "options"},
    )

    def __init__(self) -> None:
        super().__init__()
        self.steps: tuple[str, ...] = ()
        self.configured = False

    def configure(self, options: Mapping[str, Any], universe_sample: Sequence[str] | None) -> Self:
        options = dict(options or {})
        canonical = options.get("canonical")
        if not canonical:
            raise PluginError("local_key needs options.canonical (a regex of the key)")
        try:
            re.compile(str(canonical))
        except re.error as exc:
            raise PluginError(f"local_key options.canonical is not a regex: {exc}") from exc
        steps = tuple(str(s) for s in options.get("normalize", ["strip"]))
        bad = [s for s in steps if s not in LOCAL_STEPS]
        if bad:
            raise PluginError(f"local_key options.normalize: unknown or unsupported steps {bad} "
                              f"(allowed: {list(LOCAL_STEPS)})")
        other = super().configure(options, universe_sample)
        set_attr(other, "canonical", str(canonical))
        set_attr(other, "examples", configured_examples(options, type(self).examples, str(canonical)))
        other.steps = steps
        other.configured = True
        return other

    def describe(self) -> str:
        if not self.configured:
            return super().describe()
        example = self.examples[0] if self.examples else self.canonical
        return f"key matching {self.canonical}, e.g. {example}"[:200]

    def _normalize(self, text: str, stored: bool) -> Normalized | Rejected:
        if not self.configured:
            return Rejected("local_key is not configured (options.canonical)")
        t = Trace(text)
        for step in self.steps:
            v = t.value
            if step == "strip":
                t.strip()
            elif step == "upper":
                t.upper()
            elif step == "lower":
                t.lower()
            elif step == "strip_version":
                t.apply(step, re.sub(r"\.\d+$", "", v))
            elif step == "strip_prefix":
                for p in _texts(self.options.get("strip_prefix")):
                    if v.startswith(p):
                        t.apply(step, v[len(p):])
                        break
            elif step == "strip_suffix":
                for s in _texts(self.options.get("strip_suffix")):
                    if s and v.endswith(s):
                        t.apply(step, v[: -len(s)])
                        break
            elif step == "separator_to_underscore":
                t.apply(step, re.sub(r"[:\-\s]", "_", v))
            elif step == "curie_colon_to_underscore":
                t.apply(step, v.replace(":", "_", 1))
            elif step == "curie_underscore_to_colon":
                t.apply(step, v.replace("_", ":", 1))
        if re.fullmatch(self.canonical, t.value):
            return t.done()
        return Rejected(f"{t.value!r} does not match this key's pattern {self.canonical}")
