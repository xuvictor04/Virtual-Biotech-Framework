"""Agent-loop-level retry for transient provider failures.

Vendor SDKs retry only the initial HTTP response, with a short budget. Long
agent runs also need to survive sustained overload (529), rate limits (429),
dropped connections and errors that arrive *after* streaming has started. Those
surface as :class:`~vbt.providers.base.RetryableProviderError`, and
:func:`complete_with_retry` re-sends the same request with jittered exponential
backoff, honouring the server's ``retry-after`` when it sent one.

Re-sending is safe because conversations are append-only and a failed response
is never appended to the history. Partial streamed text from a failed attempt
may already have reached ``on_text``; ``on_retry`` lets callers mark that.

Provider-neutral: no vendor SDK imports.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
from dataclasses import dataclass, fields
from typing import Any, Awaitable, Callable, Union

from .base import LLMProvider, ModelResponse, RetryableProviderError

log = logging.getLogger(__name__)

RetryCallback = Callable[[int, RetryableProviderError, float], Union[None, Awaitable[None]]]


@dataclass
class RetryPolicy:
    """``attempts`` is the total number of calls (first try included)."""

    attempts: int = 8
    base_delay_s: float = 2.0
    max_delay_s: float = 60.0
    jitter: float = 0.25           # +/- fraction applied to the exponential delay
    max_retry_after_s: float = 600.0  # cap on a server-requested retry-after

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "RetryPolicy":
        """Build from the ``retry:`` config section (or a full config holding one).

        Unknown keys are ignored; missing keys keep the in-code defaults.
        """
        cfg = dict(cfg or {})
        if isinstance(cfg.get("retry"), dict):
            cfg = dict(cfg["retry"])
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for k, v in cfg.items():
            if k in known and v is not None:
                kwargs[k] = int(v) if k == "attempts" else float(v)
        policy = cls(**kwargs)
        policy.attempts = max(1, policy.attempts)
        return policy

    def delay(self, retry_index: int, retry_after: float | None = None,
              rng: Callable[[], float] = random.random) -> float:
        """Delay before retry number ``retry_index`` (0-based).

        A server-provided ``retry_after`` wins (capped at ``max_retry_after_s``);
        otherwise ``base * 2**n`` capped at ``max_delay_s`` with +/- jitter.
        """
        if retry_after is not None and retry_after >= 0:
            return float(min(retry_after, self.max_retry_after_s))
        d = min(self.max_delay_s, self.base_delay_s * (2 ** max(0, retry_index)))
        if self.jitter:
            d *= 1.0 + self.jitter * (2.0 * rng() - 1.0)
        return max(0.0, min(d, self.max_delay_s * (1.0 + self.jitter)))


async def complete_with_retry(
    provider: LLMProvider,
    *,
    policy: RetryPolicy | None = None,
    on_retry: RetryCallback | None = None,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    **complete_kwargs: Any,
) -> ModelResponse:
    """``provider.complete(**complete_kwargs)`` with retries on transient errors.

    Non-retryable errors (``ProviderError``, ``ContextOverflowError``, anything
    else) propagate immediately. After the last attempt the final
    ``RetryableProviderError`` is re-raised with ``.attempts`` set. The returned
    response has ``.retries`` = number of failed attempts before it.
    """
    policy = policy or RetryPolicy()
    attempts = max(1, int(policy.attempts))
    for attempt in range(1, attempts + 1):
        try:
            resp = await provider.complete(**complete_kwargs)
        except RetryableProviderError as exc:
            if attempt >= attempts:
                exc.attempts = attempt  # type: ignore[attr-defined]
                raise
            delay = policy.delay(attempt - 1, exc.retry_after)
            log.info("provider %s transient error (attempt %d/%d): %s; retrying in %.1fs",
                     getattr(provider, "name", "?"), attempt, attempts, exc, delay)
            if on_retry is not None:
                r = on_retry(attempt, exc, delay)
                if inspect.isawaitable(r):
                    await r
            await sleep(delay)
            continue
        resp.retries = attempt - 1
        return resp
    raise AssertionError("unreachable")  # pragma: no cover


__all__ = ["RetryPolicy", "complete_with_retry"]
