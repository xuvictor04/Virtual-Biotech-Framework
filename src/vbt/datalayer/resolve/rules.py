"""The closed resolver rule grammar and per-id_type rule lists (§11.5, rev 2). No pyarrow.

A resolution is recorded with exactly one rule of this grammar (``Resolution.rule``)::

    exact | raw_member | normalized:<step>[+<step>...] | label_exact:<column> | label_casefold:<column>
    | synonym:<kind> | retired:<column> | xref:<namespace> | crosswalk:<name>[><name>...]
    | parent_family | stored_form

In an id_type's ``rules`` list the argument may be omitted (``normalized`` takes any steps,
``label_exact`` any label column, ``retired``/``xref``/``crosswalk`` any column, namespace or
crosswalk); with an argument the rule is restricted to it (``label_exact:approvedSymbol``,
``normalized:upper`` accepts only normalisations whose steps are among those listed). Synonym kinds
``narrow`` and ``broad`` are valid grammar but **never resolve**: a broad synonym would substitute
a broader concept (I1), so they appear only in search results.

Index rows (``resolve/index.py``) carry the same grammar in their ``rule`` column, plus ``attr:<column>``
rows holding the ``disambiguate_with`` values of a canonical key (never a resolution rule).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..plugins.base import NORMALIZE_STEPS

__all__ = [
    "RULE_HEADS", "ARG_REQUIRED", "SYNONYM_KINDS", "SEARCH_ONLY_SYNONYMS", "UNIQUE_ONLY", "MEMBERSHIP_HEADS",
    "LABEL_HEADS", "ENTRY_HEADS", "ALLOW_FAMILY", "Rule", "RuleError", "parse_rule", "parse_entry_rule", "format_rule",
    "default_rules", "rules_for", "allowed", "resolves",
]

#: Rule heads of the closed grammar.
RULE_HEADS: tuple[str, ...] = ("exact", "raw_member", "normalized", "label_exact", "label_casefold", "synonym",
                               "retired", "xref", "crosswalk", "parent_family", "stored_form")
#: Heads that never take an argument.
_NO_ARG = frozenset({"exact", "raw_member", "parent_family", "stored_form"})
#: Heads whose recorded (resolution) form needs an argument; a rules-list entry may omit it.
ARG_REQUIRED = frozenset({"normalized", "label_exact", "label_casefold", "synonym", "retired", "xref", "crosswalk"})
SYNONYM_KINDS: tuple[str, ...] = ("exact", "alias", "previous", "obsolete", "related", "broad", "narrow")
SEARCH_ONLY_SYNONYMS = frozenset({"broad", "narrow"})
#: Rules that resolve only to a unique canonical (several -> ambiguous, never the first).
UNIQUE_ONLY = frozenset({"label_casefold", "synonym"})
#: Rules decided by universe membership of the value itself.
MEMBERSHIP_HEADS = frozenset({"exact", "raw_member", "normalized"})
#: Rules that match labels (a label kind resolves through its parent type's index with these).
LABEL_HEADS = frozenset({"label_exact", "label_casefold", "synonym"})
#: Heads an index row may carry: resolution rules plus ``attr`` (disambiguation values).
ENTRY_HEADS: tuple[str, ...] = RULE_HEADS + ("attr",)

#: data.resolution.allow families (§17) per rule; ``exact`` and ``stored_form`` are always allowed.
ALLOW_FAMILY: dict[str, str | None] = {
    "exact": None, "stored_form": None, "raw_member": "raw_member", "normalized": "normalized",
    "label_exact": "label", "label_casefold": "label", "retired": "retired", "xref": "xref",
    "crosswalk": "crosswalk", "parent_family": "parent_family",
    "synonym:previous": "previous", "synonym:obsolete": "previous", "synonym:alias": "alias",
    "synonym:exact": "exact_synonym", "synonym:related": "related_synonym",
}

_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-\[\]@/^]*")


class RuleError(ValueError):
    """A rule outside the closed grammar."""


@dataclass(frozen=True)
class Rule:
    head: str
    arg: str | None = None

    @property
    def text(self) -> str:
        return format_rule(self)

    @property
    def steps(self) -> tuple[str, ...]:
        """``normalized:<a>+<b>`` -> ``(a, b)``."""
        return tuple(self.arg.split("+")) if self.head == "normalized" and self.arg else ()

    @property
    def chain(self) -> tuple[str, ...]:
        """``crosswalk:<a>><b>`` -> ``(a, b)``."""
        return tuple(self.arg.split(">")) if self.head == "crosswalk" and self.arg else ()

    @property
    def synonym_kind(self) -> str | None:
        return self.arg if self.head == "synonym" else None

    @property
    def search_only(self) -> bool:
        return self.head == "synonym" and self.arg in SEARCH_ONLY_SYNONYMS

    def __str__(self) -> str:
        return self.text


def _parse(text: str | Rule, heads: Sequence[str]) -> Rule:
    if isinstance(text, Rule):
        return text
    raw = str(text).strip()
    head, sep, arg = raw.partition(":")
    if head not in heads:
        raise RuleError(f"unknown resolver rule {raw!r} (grammar: {', '.join(heads)})")
    if not sep:
        return Rule(head)
    if head in _NO_ARG:
        raise RuleError(f"rule {head!r} takes no argument ({raw!r})")
    if not arg:
        raise RuleError(f"rule {raw!r} has an empty argument")
    if head == "normalized":
        bad = [s for s in arg.split("+") if s not in NORMALIZE_STEPS]
        if bad:
            raise RuleError(f"rule {raw!r}: steps {bad} are not normalisation steps ({sorted(NORMALIZE_STEPS)})")
    elif head == "synonym":
        if arg not in SYNONYM_KINDS:
            raise RuleError(f"rule {raw!r}: synonym kinds are {', '.join(SYNONYM_KINDS)}")
    elif head == "crosswalk":
        names = arg.split(">")
        if not all(_NAME.fullmatch(n) for n in names):
            raise RuleError(f"rule {raw!r}: crosswalk names are identifiers joined by '>'")
    elif not _NAME.fullmatch(arg):
        raise RuleError(f"rule {raw!r}: {arg!r} is not a column, namespace or name")
    return Rule(head, arg)


def parse_rule(text: str | Rule) -> Rule:
    """Parse one rule of the closed grammar (raises :class:`RuleError`)."""
    return _parse(text, RULE_HEADS)


def parse_entry_rule(text: str | Rule) -> Rule:
    """Parse an index row's ``rule`` value (the grammar plus ``attr:<column>``)."""
    rule = _parse(text, ENTRY_HEADS)
    if rule.head == "attr" and not rule.arg:
        raise RuleError("index rows with rule 'attr' name a column (attr:<column>)")
    return rule


def format_rule(rule: Rule | str, arg: str | Sequence[str] | None = None) -> str:
    """``format_rule("normalized", ["strip", "upper"]) == "normalized:strip+upper"``."""
    if isinstance(rule, Rule):
        head, a = rule.head, rule.arg
    else:
        head, a = str(rule), None
        if arg is not None:
            if isinstance(arg, str):
                a = arg
            else:
                a = (">" if head == "crosswalk" else "+").join(arg)
    return f"{head}:{a}" if a else head


#: The default rule order (§11.5).
_DEFAULT_RULES: tuple[str, ...] = ("raw_member", "exact", "normalized", "label_exact", "label_casefold",
                                   "synonym:previous", "synonym:alias", "synonym:exact", "synonym:related",
                                   "retired", "xref", "crosswalk")


def default_rules(plugin: Any = None) -> list[Rule]:
    """The default ordered rule list; a plugin may override it with a ``default_rules`` class attribute."""
    custom = getattr(plugin, "default_rules", None) if plugin is not None else None
    texts = custom if isinstance(custom, (list, tuple)) else _DEFAULT_RULES
    return [parse_rule(t) for t in texts]


def allowed(rule: Rule, allow: Iterable[str] | None) -> bool:
    """True when ``data.resolution.allow`` permits ``rule`` (None allows everything)."""
    if allow is None:
        return True
    if rule.head == "synonym":
        if rule.arg is None:
            return True                                # each matched synonym kind is checked on its own
        family = ALLOW_FAMILY.get(rule.text, rule.arg)
    else:
        family = ALLOW_FAMILY.get(rule.head, rule.head)
    return family is None or family in set(allow)


def resolves(rule: Rule) -> bool:
    """False for rules that never resolve (narrow and broad synonyms)."""
    return not rule.search_only


def rules_for(spec: Any, plugin: Any = None, allow: Iterable[str] | None = None) -> list[Rule]:
    """The ordered rules of an id_type: ``spec.rules`` or the defaults, without rules that never resolve
    or that ``allow`` excludes. A ``resolvable: false`` id_type keeps only the membership rules."""
    texts = getattr(spec, "rules", None)
    rules = [parse_rule(t) for t in texts] if texts else default_rules(plugin)
    out = [r for r in rules if resolves(r) and allowed(r, allow)]
    if getattr(spec, "resolvable", True) is False:
        out = [r for r in out if r.head in MEMBERSHIP_HEADS]
    return out
