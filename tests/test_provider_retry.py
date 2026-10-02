"""Agent-loop-level retry of transient provider errors (no network)."""

import pytest

from vbt.providers.base import (
    ContextOverflowError,
    Message,
    ModelSettings,
    ProviderError,
    RetryableProviderError,
)
from vbt.providers.mock import ScriptedProvider, fail, reply
from vbt.providers.retry import RetryPolicy, complete_with_retry

S = ModelSettings("mock", "m", extra={"agent_name": "a"})


class Sleeper:
    def __init__(self):
        self.calls = []

    async def __call__(self, s):
        self.calls.append(s)


def _provider(*items):
    return ScriptedProvider.from_rules({"a": list(items)})


async def _go(provider, policy=None, **kw):
    sleep = Sleeper()
    resp = await complete_with_retry(provider, policy=policy or RetryPolicy(jitter=0.0), sleep=sleep,
                                     settings=S, system="", messages=[Message.user("q")], tools=[], **kw)
    return resp, sleep.calls


async def test_retry_honours_retry_after_then_backs_off():
    p = _provider(fail(RetryableProviderError("429", retry_after=3.5, status=429)),
                  fail(RetryableProviderError("529", status=529)),
                  fail(RetryableProviderError("conn")),
                  reply("ok"))
    resp, sleeps = await _go(p, RetryPolicy(base_delay_s=2.0, max_delay_s=60.0, jitter=0.0))
    assert resp.message.text == "ok" and resp.retries == 3
    assert sleeps == [3.5, 4.0, 8.0]  # retry-after wins; then base * 2**n
    assert len(p.calls) == 4


async def test_non_retryable_errors_are_not_retried():
    for exc in (ProviderError("bad request"), ContextOverflowError("prompt is too long"), ValueError("bug")):
        p = _provider(fail(exc), reply("never"))
        with pytest.raises(type(exc)):
            await _go(p)
        assert len(p.calls) == 1


async def test_gives_up_after_attempts():
    p = _provider(*[fail(RetryableProviderError("overloaded", status=529)) for _ in range(5)])
    sleep = Sleeper()
    with pytest.raises(RetryableProviderError) as ei:
        await complete_with_retry(p, policy=RetryPolicy(attempts=3, jitter=0.0), sleep=sleep,
                                  settings=S, system="", messages=[Message.user("q")], tools=[])
    assert ei.value.attempts == 3 and len(p.calls) == 3 and len(sleep.calls) == 2


async def test_on_retry_callback_sync_and_async():
    seen = []

    async def on_retry(attempt, exc, delay):
        seen.append((attempt, exc.status, delay))

    p = _provider(fail(), reply("ok"))
    resp, _ = await _go(p, RetryPolicy(base_delay_s=1.0, jitter=0.0), on_retry=on_retry)
    assert resp.retries == 1 and seen == [(1, 529, 1.0)]
    seen2 = []
    p = _provider(fail(), reply("ok"))
    await _go(p, on_retry=lambda a, e, d: seen2.append(a))
    assert seen2 == [1]


def test_delay_caps_and_jitter():
    pol = RetryPolicy(base_delay_s=2.0, max_delay_s=60.0, jitter=0.25)
    assert pol.delay(10, rng=lambda: 0.5) == 60.0
    assert pol.delay(0, rng=lambda: 1.0) == pytest.approx(2.5)
    assert pol.delay(0, rng=lambda: 0.0) == pytest.approx(1.5)
    assert pol.delay(0, retry_after=10_000) == pol.max_retry_after_s


def test_policy_from_config():
    assert RetryPolicy.from_config(None) == RetryPolicy()
    pol = RetryPolicy.from_config({"attempts": 3, "base_delay_s": 1, "unknown": 5})
    assert pol.attempts == 3 and pol.base_delay_s == 1.0 and pol.max_delay_s == 60.0
    assert RetryPolicy.from_config({"retry": {"attempts": 2}}).attempts == 2
    assert RetryPolicy.from_config({"attempts": 0}).attempts == 1


def test_provider_neutral_modules_do_not_import_anthropic():
    import subprocess
    import sys
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src"
    code = ("import sys; sys.path.insert(0, %r); import vbt.providers, vbt.providers.base, vbt.providers.mock, "
            "vbt.providers.retry, vbt.context; print('anthropic' in sys.modules)" % str(src))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"
