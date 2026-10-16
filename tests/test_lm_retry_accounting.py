"""Offline accounting tests: per-attempt stats, additive cross-attempt totals,
billed failed attempts, and reasoning-token arithmetic (no paid API calls)."""

from pathlib import Path

import pytest
from litellm.types.utils import Choices, Message, ModelResponse, Usage

import lotus
import lotus.models.lm as lm_module
from lotus.cache import CacheConfig, CacheFactory, CacheType
from lotus.models import LM
from lotus.types import LMStats, LotusRetryExhaustedError

# ---------------------------------------------------------------------------
# work dirs: tempfile under the repo's outputs/ (pytest's tmp machinery uses
# extended-length paths that the DSH file sandbox denies on this host)
# ---------------------------------------------------------------------------


def make_work_dir(prefix: str) -> Path:
    # os.makedirs (not tempfile.mkdtemp): under this host's file sandbox,
    # mkdtemp dirs get deny-ACLs that block sqlite file creation.
    import os
    import uuid

    base = Path(__file__).resolve().parents[1] / "outputs" / "_retry_test_work"
    base.mkdir(parents=True, exist_ok=True)
    d = base / f"{prefix}_{uuid.uuid4().hex[:8]}"
    os.makedirs(d, exist_ok=True)
    return d


def sqlite_cache():
    """Fresh SQLite cache in a scratch dir for one test (no pytest tmp machinery)."""
    return CacheFactory.create_cache(
        CacheConfig(CacheType.SQLITE, 1024, cache_dir=str(make_work_dir("cache")))
    )

# ---------------------------------------------------------------------------
# shared offline builders
# ---------------------------------------------------------------------------


def make_usage(prompt=10, completion=1, reasoning=None, cost=None):
    kw = dict(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion)
    if reasoning is not None:
        kw["completion_tokens_details"] = {"reasoning_tokens": reasoning, "audio_tokens": 0}
    if cost is not None:
        kw["cost_details"] = {"upstream_inference_cost": cost}
    return Usage(**kw)


def make_response(resp_id, content, usage=None, psf=None):
    kwargs = dict(
        id=resp_id,
        created=1,
        model="openai/gpt-5-mini",
        object="chat.completion",
        system_fingerprint=None,
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(content=content, role="assistant"),
            )
        ],
    )
    if usage is not None:
        kwargs["usage"] = usage
    if psf is not None:
        kwargs["provider_specific_fields"] = psf
    return ModelResponse(**kwargs)


OPENROUTER_FAILURE_PSF = {
    "error": {
        "code": 502,
        "message": "Stream ended before a terminal response event",
        "metadata": {"error_type": "provider_unavailable"},
    },
    "native_finish_reason": "error",
}

OK_USAGE = make_usage(10, 1, cost=0.00001)
FAIL_USAGE = make_usage(120, 146, reasoning=146, cost=0.000322)


class FakeBatch:
    def __init__(self, responses_by_call):
        self.responses_by_call = list(responses_by_call)
        self.calls = []

    def __call__(self, model, messages, **kwargs):
        self.calls.append([m[0]["content"] for m in messages])
        return self.responses_by_call.pop(0)


# ---------------------------------------------------------------------------
# Test 7 â€” exact accounting across a failed attempt + resumed attempt
# ---------------------------------------------------------------------------


def test_accounting_across_attempts(monkeypatch):
    lotus.settings.configure(enable_cache=True)
    msgs = [[{"role": "user", "content": f"p{i}"}] for i in range(4)]
    fail = make_response("fail", None, usage=FAIL_USAGE, psf=OPENROUTER_FAILURE_PSF)
    cache_dir = make_work_dir("shared_cache")  # both attempts share this DB

    # ---- attempt 1 (max_retries=1): p0,p1 valid; p2 transient then retry-ok; p3 exhausts
    cache1 = CacheFactory.create_cache(CacheConfig(CacheType.SQLITE, 1024, cache_dir=str(cache_dir)))
    lm1 = LM(model="openrouter/openai/gpt-5-mini", max_batch_size=4, max_retries=1, cache=cache1)
    responses1 = [
        make_response("r0", "True", OK_USAGE),
        make_response("r1", "True", OK_USAGE),
        fail,
        fail,
    ]
    retries1 = [make_response("r2-ok", "True", OK_USAGE), fail]  # p2 ok, p3 retryable again
    fake1 = FakeBatch([responses1, retries1])
    monkeypatch.setattr(lm_module, "batch_completion", fake1)

    with pytest.raises(LotusRetryExhaustedError):
        lm1(msgs, show_progress_bar=False)

    s1 = lm1.stats
    assert s1.physical_requests == 6  # dispatch1(4) + retry dispatch(2)
    assert s1.retry_attempts == 2  # p2 and p3 each retried once
    assert s1.retryable_failures == 3  # p2 fail + p3 fail on both dispatches
    assert s1.fatal_failures == 0
    assert s1.cache_hits == 0
    assert s1.physical_usage.prompt_tokens == 10 + 10 + 120 + 10 + 120 + 120
    assert s1.physical_usage.completion_tokens == 1 + 1 + 146 + 1 + 146 + 146
    assert s1.physical_usage.total_tokens == 11 + 11 + 266 + 11 + 266 + 266
    assert s1.physical_usage.reasoning_tokens == 146 * 3  # all three billed failures
    assert s1.physical_usage.total_cost == pytest.approx(3 * 0.00001 + 3 * 0.000322)

    # ---- attempt 2 (resume): p0,p1,p2 are cache hits; only p3 reaches the provider
    cache2 = CacheFactory.create_cache(CacheConfig(CacheType.SQLITE, 1024, cache_dir=str(cache_dir)))
    lm2 = LM(model="openrouter/openai/gpt-5-mini", max_batch_size=4, max_retries=1, cache=cache2)
    fake2 = FakeBatch([[make_response("r3-ok", "True", OK_USAGE)]])
    monkeypatch.setattr(lm_module, "batch_completion", fake2)

    out = lm2(msgs, show_progress_bar=False)
    assert out.outputs == ["True"] * 4

    s2 = lm2.stats
    assert s2.physical_requests == 1
    assert s2.retry_attempts == 0
    assert s2.retryable_failures == 0
    assert s2.cache_hits == 3
    assert s2.physical_usage.total_tokens == 11
    assert fake2.calls == [["p3"]]

    # ---- cumulative physical totals = attempt1 + attempt2 (additive domain)
    cum = s1 + s2
    assert cum.physical_requests == 7
    assert cum.retry_attempts == 2
    assert cum.retryable_failures == 3
    assert cum.fatal_failures == 0
    assert cum.cache_hits == 3
    assert cum.physical_usage.total_tokens == 11 + 11 + 266 + 11 + 266 + 266 + 11
    assert cum.physical_usage.reasoning_tokens == 146 * 3
    assert cum.physical_usage.total_cost == pytest.approx(4 * 0.00001 + 3 * 0.000322)


# ---------------------------------------------------------------------------
# LMStats / TotalUsage arithmetic with the new fields
# ---------------------------------------------------------------------------


def test_total_usage_add_sub_with_reasoning():
    a = LMStats.TotalUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, reasoning_tokens=4, total_cost=0.01)
    b = LMStats.TotalUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3, reasoning_tokens=1, total_cost=0.001)
    c = a + b
    assert (c.prompt_tokens, c.completion_tokens, c.total_tokens) == (12, 6, 18)
    assert (c.reasoning_tokens, c.total_cost) == (5, 0.011)
    d = a - b
    assert (d.prompt_tokens, d.completion_tokens, d.reasoning_tokens) == (8, 4, 3)


def test_lm_stats_add_includes_retry_counters():
    a = LMStats(physical_requests=6, retry_attempts=2, retryable_failures=2, fatal_failures=1, cache_hits=1)
    b = LMStats(physical_requests=1, retry_attempts=0, retryable_failures=0, fatal_failures=0, cache_hits=3)
    c = a + b
    assert (c.physical_requests, c.retry_attempts, c.retryable_failures, c.fatal_failures, c.cache_hits) == (
        7,
        2,
        2,
        1,
        4,
    )
