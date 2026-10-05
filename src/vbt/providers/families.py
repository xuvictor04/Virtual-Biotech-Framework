"""Model-family dialects for OpenAI-compatible servers (vLLM, SGLang, llama.cpp).

One OpenAI Chat Completions adapter (:mod:`vbt.providers.openai_compat`) serves
many open-weight models, but each family's chat template has its own reasoning
controls, sampling recommendations and quirks. A :class:`ModelFamily` holds
those per-family facts; :func:`resolve_family` picks one from a model id.

Sources: the local-LLM research digest (2026-10-05) and its verifier
corrections: the Qwen3.8 model card and chat template, the vLLM 0.31 recipe,
the DeepSeek-V4-Flash card and vLLM's deepseek_v4 tokenizer mode.

Key facts encoded here
----------------------
* Qwen3.8 (default family ``qwen3_8``): top-level ``reasoning_effort`` accepts
  only ``xhigh``, ``medium``, ``low`` and ``none`` (vLLM turns ``none`` into
  ``enable_thinking=false``). ``high``, ``max`` and ``minimal`` make the chat
  template raise ("Unexpected reasoning effort"), which comes back as HTTP 400,
  so the harness vocabulary is mapped: low->low, medium->medium,
  high/xhigh/max->xhigh, thinking off or no effort -> none.
  ``thinking_token_budget`` is a hard reasoning cap (vLLM forces ``</think>``).
  Sampling (card): thinking T=1.0 top_p=0.95 top_k=20 min_p=0 presence=0
  repetition=1.0; non-thinking T=0.7 top_p=0.8 top_k=20 min_p=0 presence=1.5.
  ``preserve_thinking`` is left at the template default (true), so replayed
  reasoning keeps the prompt append-only (prefix-cache friendly).
* Qwen3.5 / Qwen3.6 (``qwen3_6``, e.g. Qwen3.6-35B-A3B): no reasoning-effort
  levels (the template ignores them); thinking is the ``enable_thinking``
  chat-template toggle; the template defaults ``preserve_thinking`` to FALSE,
  so the adapter sends ``preserve_thinking: true``. Sampling: thinking T=1.0
  top_p=0.95 top_k=20 presence=1.5; non-thinking T=0.7 top_p=0.8 presence=1.5.
* Qwen3 (``qwen3``, the 2025 hybrid models such as Qwen3-0.6B ... Qwen3-235B,
  handy for small CPU smoke tests): ``enable_thinking`` toggle; card sampling
  thinking T=0.6 top_p=0.95 top_k=20 min_p=0, non-thinking T=0.7 top_p=0.8.
* DeepSeek-V4-Flash (``deepseek_v4``): effort vocabulary none/low/high/max
  (vLLM maps medium->low and xhigh->high itself; the adapter sends canonical
  values), thinking toggled with ``chat_template_kwargs {"thinking": bool}``;
  sampling T=1.0 top_p=0.95 (no top_k/min_p guidance).
* ``generic``: plain OpenAI semantics; nothing family-specific is sent unless
  the provider option ``reasoning_effort_supported`` is set, in which case the
  standard low/medium/high vocabulary is used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

#: Harness effort vocabulary (``ModelSettings.effort``).
HARNESS_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")

#: ``reasoning_request_style`` values.
STYLE_REASONING_EFFORT = "reasoning_effort"          # top-level ``reasoning_effort`` field
STYLE_CHAT_TEMPLATE_KWARGS = "chat_template_kwargs"  # ``chat_template_kwargs`` toggles only
STYLE_NONE = "none"                                  # send nothing reasoning-related


@dataclass(frozen=True)
class ModelFamily:
    """Per-family request/response dialect.

    ``effort_map`` maps the harness effort vocabulary to the value sent on the
    wire; ``allowed_efforts`` is the set the server/template accepts (anything
    else is never sent). ``off_effort`` is the wire value that disables thinking
    (``'none'``), or None when thinking is toggled only via
    ``thinking_kwarg``. ``thinking_kwarg`` names the ``chat_template_kwargs``
    key that toggles thinking (``enable_thinking`` / ``thinking``); with
    ``send_thinking_kwarg`` the toggle is sent explicitly on every request.
    ``reasoning_response_fields`` are read (in order) from responses;
    ``replay_reasoning_field`` is the assistant-message field used to send
    prior reasoning back (None: never replayed).
    """

    name: str
    match: tuple[str, ...] = ()
    effort_map: dict[str, str] = field(default_factory=dict)
    allowed_efforts: frozenset[str] = frozenset()
    sampling_thinking: dict[str, Any] = field(default_factory=dict)
    sampling_plain: dict[str, Any] = field(default_factory=dict)
    reasoning_request_style: str = STYLE_NONE
    reasoning_response_fields: tuple[str, ...] = ("reasoning", "reasoning_content")
    replay_reasoning_field: str | None = "reasoning"
    supports_thinking_budget: bool = False
    default_chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    max_tool_name_len: int = 64
    off_effort: str | None = None
    thinking_kwarg: str | None = None
    send_thinking_kwarg: bool = False
    #: Wire effort used when thinking is on but the harness effort is unknown.
    default_effort: str | None = None
    #: Thinking with ``effort=None`` (True) or effort None means thinking off (False).
    thinking_without_effort: bool = False
    #: The family can think at all (False: never sends reasoning controls).
    thinking: bool = True

    # ------------------------------------------------------------------ helpers

    def map_effort(self, effort: str | None) -> str | None:
        """Wire effort for a harness effort (None when the family has no levels
        or the value cannot be mapped to an allowed one)."""
        if not self.allowed_efforts:
            return None
        if effort is None:
            return self.default_effort
        e = str(effort).strip().lower()
        if e in self.effort_map:
            return self.effort_map[e]
        if e in self.allowed_efforts:
            return e
        return self.default_effort

    def reasoning_mode(self, effort: str | None, thinking: bool,
                       override: str | None = None) -> tuple[bool, str | None]:
        """``(thinking_on, wire_effort)`` for the harness settings.

        ``override`` is an explicit wire value (``extra['reasoning_effort']``);
        it is used verbatim when the family accepts it, otherwise it is mapped
        like a harness effort, so a disallowed value is never sent.
        """
        if not self.thinking:
            return False, None
        if override is not None:
            o = str(override).strip().lower()
            if self.off_effort is not None and o == self.off_effort:
                return False, self.off_effort
            if o in ("none", "off", "false"):
                return False, self.off_effort
            wire = o if o in self.allowed_efforts else self.map_effort(o)
            return True, wire
        if not thinking:
            return False, self.off_effort
        if effort is None and not self.thinking_without_effort:
            return False, self.off_effort
        return True, self.map_effort(effort)

    def reasoning_fields(self, thinking_on: bool, wire_effort: str | None) -> dict[str, Any]:
        """Top-level request fields controlling reasoning (``reasoning_effort``
        and/or ``chat_template_kwargs``). Never contains a disallowed effort."""
        out: dict[str, Any] = {}
        kwargs: dict[str, Any] = dict(self.default_chat_template_kwargs)
        if self.reasoning_request_style == STYLE_REASONING_EFFORT:
            if wire_effort is not None and (wire_effort in self.allowed_efforts or wire_effort == self.off_effort):
                out["reasoning_effort"] = wire_effort
            if self.thinking_kwarg and (self.send_thinking_kwarg or not thinking_on):
                kwargs[self.thinking_kwarg] = bool(thinking_on)
        elif self.reasoning_request_style == STYLE_CHAT_TEMPLATE_KWARGS:
            if self.thinking_kwarg:
                kwargs[self.thinking_kwarg] = bool(thinking_on)
        if kwargs:
            out["chat_template_kwargs"] = kwargs
        return out

    def sampling(self, thinking_on: bool) -> dict[str, Any]:
        return dict(self.sampling_thinking if thinking_on else self.sampling_plain)


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------

_QWEN_THINKING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
                  "repetition_penalty": 1.0}
_QWEN_PLAIN = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
               "repetition_penalty": 1.0}

QWEN3_8 = ModelFamily(
    name="qwen3_8",
    match=(r"qwen3\.8(?!\d)", r"qwen3p8(?!\d)"),
    effort_map={"minimal": "low", "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh",
                "max": "xhigh"},
    allowed_efforts=frozenset({"xhigh", "medium", "low"}),
    off_effort="none",
    sampling_thinking=dict(_QWEN_THINKING),
    sampling_plain=dict(_QWEN_PLAIN),
    reasoning_request_style=STYLE_REASONING_EFFORT,
    reasoning_response_fields=("reasoning", "reasoning_content"),
    replay_reasoning_field="reasoning",
    supports_thinking_budget=True,
    # preserve_thinking stays at the template default (true): append-only prompts.
    default_chat_template_kwargs={},
    max_tool_name_len=64,
    # 'none' alone switches thinking off in vLLM; enable_thinking=false is also
    # sent then so servers that ignore reasoning_effort (SGLang, llama.cpp) agree.
    thinking_kwarg="enable_thinking",
    default_effort="medium",
)

QWEN3_6 = ModelFamily(
    name="qwen3_6",
    match=(r"qwen3\.[56](?!\d)",),
    effort_map={},
    allowed_efforts=frozenset(),
    sampling_thinking={"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
                       "repetition_penalty": 1.0},
    sampling_plain=dict(_QWEN_PLAIN),
    reasoning_request_style=STYLE_CHAT_TEMPLATE_KWARGS,
    replay_reasoning_field="reasoning",
    supports_thinking_budget=True,
    # The Qwen3.5/3.6 template defaults preserve_thinking to false, which drops
    # older reasoning whenever a new user message arrives (prefix-cache misses).
    default_chat_template_kwargs={"preserve_thinking": True},
    thinking_kwarg="enable_thinking",
    # No effort levels: ModelSettings.thinking alone decides.
    thinking_without_effort=True,
)

QWEN3 = ModelFamily(
    name="qwen3",
    match=(r"(?:^|/)qwen3-(?!coder)",),
    sampling_thinking={"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
    sampling_plain={"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0},
    reasoning_request_style=STYLE_CHAT_TEMPLATE_KWARGS,
    replay_reasoning_field="reasoning",
    supports_thinking_budget=True,
    thinking_kwarg="enable_thinking",
    thinking_without_effort=True,
)

DEEPSEEK_V4 = ModelFamily(
    name="deepseek_v4",
    match=(r"deepseek[-_]?v4", r"dsv4"),
    effort_map={"minimal": "low", "low": "low", "medium": "low", "high": "high", "xhigh": "high", "max": "max"},
    allowed_efforts=frozenset({"low", "high", "max"}),
    off_effort="none",
    sampling_thinking={"temperature": 1.0, "top_p": 0.95},
    sampling_plain={"temperature": 1.0, "top_p": 0.95},
    reasoning_request_style=STYLE_REASONING_EFFORT,
    reasoning_response_fields=("reasoning", "reasoning_content"),
    replay_reasoning_field="reasoning",
    supports_thinking_budget=False,
    thinking_kwarg="thinking",
    send_thinking_kwarg=True,
    default_effort="high",
)

GENERIC = ModelFamily(
    name="generic",
    match=(),
    reasoning_request_style=STYLE_NONE,
    replay_reasoning_field="reasoning_content",
    supports_thinking_budget=False,
    thinking=True,
)

#: OpenAI-standard effort levels, used by ``generic`` when the provider option
#: ``reasoning_effort_supported`` is set.
GENERIC_WITH_EFFORT = replace(
    GENERIC,
    effort_map={"minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"},
    allowed_efforts=frozenset({"low", "medium", "high"}),
    reasoning_request_style=STYLE_REASONING_EFFORT,
    default_effort="medium",
)

FAMILIES: dict[str, ModelFamily] = {f.name: f for f in (QWEN3_8, QWEN3_6, QWEN3, DEEPSEEK_V4, GENERIC)}
#: Resolution order for model-id matching (most specific first).
_ORDER = (QWEN3_8, QWEN3_6, DEEPSEEK_V4, QWEN3)
_ALIASES = {"qwen3.8": "qwen3_8", "qwen38": "qwen3_8", "qwen3.6": "qwen3_6", "qwen3.5": "qwen3_6",
            "qwen3_5": "qwen3_6", "dsv4": "deepseek_v4", "deepseek-v4": "deepseek_v4", "openai": "generic",
            "default": "generic"}


def family_names() -> list[str]:
    return sorted(FAMILIES)


def resolve_family(model_id: str | None, explicit: str | None = None) -> ModelFamily:
    """The family for ``model_id``.

    ``explicit`` (a family name, e.g. from ``provider.options.family``) wins
    unless it is None, empty or ``'auto'``; an unknown explicit name raises
    ``ValueError``. Otherwise the model id (served name or HF repo id, e.g.
    ``qwen3.8-27b`` or ``RedHatAI/Qwen3.8-27B-NVFP4``) is matched against each
    family's patterns (case-insensitive); no match gives ``generic``.
    """
    if explicit is not None and str(explicit).strip().lower() not in ("", "auto"):
        key = str(explicit).strip().lower()
        key = _ALIASES.get(key, key)
        if key not in FAMILIES:
            raise ValueError(f"unknown model family {explicit!r}; known: {family_names()} (or 'auto')")
        return FAMILIES[key]
    mid = (model_id or "").strip().lower()
    if mid:
        for fam in _ORDER:
            if any(re.search(p, mid) for p in fam.match):
                return fam
    return GENERIC


def match_family(model_id: str | None) -> ModelFamily | None:
    """The family whose patterns match ``model_id``, or None (no generic fallback)."""
    fam = resolve_family(model_id)
    return None if fam is GENERIC else fam


__all__ = [
    "DEEPSEEK_V4", "FAMILIES", "GENERIC", "GENERIC_WITH_EFFORT", "HARNESS_EFFORTS", "ModelFamily", "QWEN3",
    "QWEN3_6", "QWEN3_8", "STYLE_CHAT_TEMPLATE_KWARGS", "STYLE_NONE", "STYLE_REASONING_EFFORT", "family_names",
    "match_family", "resolve_family",
]
