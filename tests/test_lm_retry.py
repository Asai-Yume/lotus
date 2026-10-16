"""Offline tests for LM retry/validation semantics (no paid API calls).

Monkeypatches ``lotus.models.lm.batch_completion`` with scripted responses,
covering: valid batches, partial-exception batches, OpenRouter-style partial
responses (content=None + provider error metadata), permanent failures,
retry exhaustion, empty-string semantics, and the rate-limited path.
"""

from pathlib import Path

import litellm
import pytest
from litellm.types.utils import Choices, Message, ModelResponse, Usage

import lotus
import lotus.models.lm as lm_module
from lotus.cache import CacheConfig, CacheFactory, CacheType
from lotus.models import LM
from lotus.types import LotusRetryExhaustedError

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


def make_response(resp_id, content, usage=None, reasoning=None, psf=None):
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
                message=Message(
                    content=content,
                    role="assistant",
                    tool_calls=None,
                    function_call=None,
                    reasoning_content=reasoning,
                ),
            )
        ],
    )
    if usage is not None:
        kwargs["usage"] = usage
    if psf is not None:
        kwargs["provider_specific_fields"] = psf
    return ModelResponse(**kwargs)


# The observed OpenRouter failure shape from the Amphibian run.
OPENROUTER_FAILURE_PSF = {
    "error": {
        "code": 502,
        "message": "Stream ended before a terminal response event",
        "metadata": {"error_type": "provider_unavailable"},
    },
    "native_finish_reason": "error",
}


class FakeHttpxResponse:
    status_code = 502
    headers = {}
    request = object()

    @property
    def text(self):
        return "502"


def retryable_exception():
    return litellm.exceptions.ServiceUnavailableError(
        message="Stream ended before a terminal response event",
        llm_provider="openrouter",
        model="openai/gpt-5-mini",
        response=FakeHttpxResponse(),
    )


def fatal_exception():
    return litellm.exceptions.AuthenticationError(
        message="invalid api key",
        llm_provider="openrouter",
        model="openai/gpt-5-mini",
        response=FakeHttpxResponse(),
    )


class FakeBatch:
    """Scripted batch_completion: records dispatched prompt contents and pops the
    next response list per call."""

    def __init__(self, responses_by_call):
        self.responses_by_call = list(responses_by_call)
        self.calls = []

    def __call__(self, model, messages, **kwargs):
        self.calls.append([m[0]["content"] for m in messages])
        return self.responses_by_call.pop(0)


# ---------------------------------------------------------------------------
# Test 1 â€” valid batch
# ---------------------------------------------------------------------------


def test_valid_batch_all_cached_no_retries(monkeypatch):
    cache = sqlite_cache()
    lotus.settings.configure(enable_cache=True)
    lm = LM(model="openrouter/openai/gpt-5-mini", max_batch_size=4, max_retries=2, cache=cache)
    msgs = [[{"role": "user", "content": f"p{i}"}] for i in range(8)]
    responses = [make_response(f"r{i}", "True", make_usage(10, 1)) for i in range(8)]
    fake = FakeBatch([responses])
    monkeypatch.setattr(lm_module, "batch_completion", fake)

    out = lm(msgs, show_progress_bar=False)

    assert out.outputs == ["True"] * 8
    assert len(fake.calls) == 1
    assert fake.calls[0] == [f"p{i}" for i in range(8)]
    st = lm.stats
    assert st.physical_requests == 8
    assert st.retry_attempts == 0
    assert st.retryable_failures == 0
    assert st.fatal_failures == 0
    for i in range(8):
        assert cache.get(lm._hash_messages(msgs[i], lm.kwargs)) is not None


# ---------------------------------------------------------------------------
# Test 2 Ã¢â‚¬â€ partial exception batch
# ---------------------------------------------------------------------------


def test_partial_exception_batch_retries_only_failed(monkeypatch):
    cache = sqlite_cache()
    lotus.settings.configure(enable_cache=True)
    lm = LM(model="openrouter/openai/gpt-5-mini", max_batch_size=8, max_retries=2, cache=cache)
    msgs = [[{"role": "user", "content": f"p{i}"}] for i in range(8)]
    hashes = [lm._hash_messages(m, lm.kwargs) for m in msgs]

    responses = [make_response(f"r{i}", "True", make_usage(10, 1)) for i in range(8)]
    responses[5] = retryable_exception()
    retry = make_response("r5-retry", "True", make_usage(10, 1))

    class VerifyingBatch(FakeBatch):
        def __call__(self, model, messages, **kwargs):
            self.calls.append([m[0]["content"] for m in messages])
            if len(self.calls) == 2:  # retry dispatch: 7 siblings must already be persisted
                missing = [h for i, h in enumerate(hashes) if i != 5 and cache.get(h) is None]
                assert not missing, f"sibling successes not persisted before retry: {missing}"
            return self.responses_by_call.pop(0)

    fake = VerifyingBatch([responses, [retry]])
    monkeypatch.setattr(lm_module, "batch_completion", fake)

    out = lm(msgs, show_progress_bar=False)

    assert out.outputs == ["True"] * 8
    assert len(fake.calls) == 2
    assert fake.calls[0] == [f"p{i}" for i in range(8)]
    assert fake.calls[1] == ["p5"]  # only the failed logical request was dispatched again
    st = lm.stats
    assert st.physical_requests == 9
    assert st.retry_attempts == 1
    assert st.retryable_failures == 1
    assert st.fatal_failures == 0
    for i in range(8):
        assert cache.get(hashes[i]) is not None


# ---------------------------------------------------------------------------
# Test 3 Ã¢â‚¬â€ OpenRouter-style partial response (content=None, provider error)
# ---------------------------------------------------------------------------


def test_openrouter_partial_response_usage_counted_not_cached_retried(monkeypatch):
    cache = sqlite_cache()
    lotus.settings.configure(enable_cache=True)
    lm = LM(model="openrouter/openai/gpt-5-mini", max_batch_size=8, max_retries=2, cache=cache)
    msgs = [[{"role": "user", "content": f"p{i}"}] for i in range(3)]

    failed_usage = make_usage(prompt=120, completion=146, reasoning=146, cost=0.000322)
    failed = make_response("gen-bad", None, usage=failed_usage, reasoning="...", psf=OPENROUTER_FAILURE_PSF)
    good0 = make_response("r0", "True", make_usage(10, 1, cost=0.00001))
    good2 = make_response("r2", "True", make_usage(10, 1, cost=0.00001))
    retry_ok = make_response("r1-ok", "True", make_usage(10, 1, cost=0.00001))
    fake = FakeBatch([[good0, failed, good2], [retry_ok]])
    monkeypatch.setattr(lm_module, "batch_completion", fake)

    out = lm(msgs, show_progress_bar=False)

    assert out.outputs == ["True", "True", "True"]
    st = lm.stats
    assert st.physical_requests == 4
    assert st.retry_attempts == 1
    assert st.retryable_failures == 1
    # the failed response was billed: its usage counts physically
    assert st.physical_usage.prompt_tokens == 10 + 120 + 10 + 10
    assert st.physical_usage.completion_tokens == 1 + 146 + 1 + 1
    assert st.physical_usage.total_tokens == 11 + 266 + 11 + 11
    assert st.physical_usage.reasoning_tokens == 146
    assert st.physical_usage.total_cost == pytest.approx(0.000322 + 3 * 0.00001)
    # the retried success was cached under the failed prompt's hash; the failure was not
    h1 = lm._hash_messages(msgs[1], lm.kwargs)
    cached = cache.get(h1)
    assert cached is not None and cached.choices[0].message.content == "True"


def test_openrouter_partial_response_no_retries_raises_and_not_cached(monkeypatch):
    cache = sqlite_cache()
    lotus.settings.configure(enable_cache=True)
    lm = LM(model="openrouter/openai/gpt-5-mini", max_batch_size=8, max_retries=0, cache=cache)
    msgs = [[{"role": "user", "content": "p0"}]]
    failed = make_response(
        "gen-bad",
        None,
        usage=make_usage(120, 146, reasoning=146, cost=0.000322),
        psf=OPENROUTER_FAILURE_PSF,
    )
    fake = FakeBatch([[failed]])
    monkeypatch.setattr(lm_module, "batch_completion", fake)

    with pytest.raises(LotusRetryExhaustedError):
        lm(msgs, show_progress_bar=False)
    # never cached; billed usage still counted
    assert cache.get(lm._hash_messages(msgs[0], lm.kwargs)) is None
    assert lm.stats.physical_requests == 1
    assert lm.stats.physical_usage.total_tokens == 266


# ---------------------------------------------------------------------------
# Test 4 Ã¢â‚¬â€ permanent auth/config failure
# ---------------------------------------------------------------------------


def test_permanent_auth_failure_no_retry(monkeypatch):
    cache = sqlite_cache()
    lotus.settings.configure(enable_cache=True)
    lm = LM(model="openrouter/openai/gpt-5-mini", max_batch_size=8, max_retries=5, cache=cache)
    msgs = [[{"role": "user", "content": f"p{i}"}] for i in range(3)]
    responses = [make_response(f"r{i}", "True", make_usage(10, 1)) for i in range(3)]
    responses[1] = fatal_exception()
    fake = FakeBatch([responses])
    monkeypatch.setattr(lm_module, "batch_completion", fake)

    with pytest.raises(litellm.exceptions.AuthenticationError):
        lm(msgs, show_progress_bar=False)

    st = lm.stats
    assert st.physical_requests == 3
    assert st.retry_attempts == 0  # permanent: no retries
    assert st.retryable_failures == 0
    assert st.fatal_failures == 1
    # failed prompt not cached; valid siblings were persisted before the raise
    assert cache.get(lm._hash_messages(msgs[1], lm.kwargs)) is None
    assert cache.get(lm._hash_messages(msgs[0], lm.kwargs)) is not None
    assert cache.get(lm._hash_messages(msgs[2], lm.kwargs)) is not None


# ---------------------------------------------------------------------------
# Test 10 Ã¢â‚¬â€ retry exhaustion
# ---------------------------------------------------------------------------


def test_retry_exhaustion(monkeypatch):
    cache = sqlite_cache()
    lotus.settings.configure(enable_cache=True)
    lm = LM(model="openrouter/openai/gpt-5-mini", max_batch_size=8, max_retries=2, cache=cache)
    msgs = [[{"role": "user", "content": "p0"}]]
    fail = make_response(
        "bad",
        None,
        usage=make_usage(120, 146, reasoning=146, cost=0.000322),
        psf=OPENROUTER_FAILURE_PSF,
    )
    fake = FakeBatch([[fail], [fail], [fail]])
    monkeypatch.setattr(lm_module, "batch_completion", fake)

    with pytest.raises(LotusRetryExhaustedError) as ei:
        lm(msgs, show_progress_bar=False)

    st = lm.stats
    assert st.physical_requests == 3  # max_retries=2 => up to 3 physical attempts
    assert st.retry_attempts == 2
    assert st.retryable_failures == 3
    assert st.fatal_failures == 0
    assert st.physical_usage.total_tokens == 3 * 266  # every billed attempt counted
    assert cache.get(lm._hash_messages(msgs[0], lm.kwargs)) is None
    e = ei.value
    assert e.attempts == 3
    assert e.hash == lm._hash_messages(msgs[0], lm.kwargs)
    assert e.last_error is not None and "502" in e.last_error


# ---------------------------------------------------------------------------
# Empty-string semantics (native behavior preserved: only content=None is rejected)
# ---------------------------------------------------------------------------


def test_empty_string_content_preserves_native_semantics(monkeypatch):
    cache = sqlite_cache()
    lotus.settings.configure(enable_cache=True)
    lm = LM(model="openrouter/openai/gpt-5-mini", max_batch_size=8, max_retries=5, cache=cache)
    msgs = [[{"role": "user", "content": "p0"}]]
    resp = make_response("r0", "", make_usage(10, 1))
    fake = FakeBatch([[resp]])
    monkeypatch.setattr(lm_module, "batch_completion", fake)

    out = lm(msgs, show_progress_bar=False)

    assert out.outputs == [""]  # native _get_top_choice only rejects None
    assert lm.stats.retry_attempts == 0
    assert lm.stats.retryable_failures == 0
    assert cache.get(lm._hash_messages(msgs[0], lm.kwargs)) is not None


# ---------------------------------------------------------------------------
# Test 11 Ã¢â‚¬â€ rate-limited path uses the same retry/validation primitive
# ---------------------------------------------------------------------------


def test_rate_limited_path_uses_retry_primitive(monkeypatch):
    cache = sqlite_cache()
    lotus.settings.configure(enable_cache=True)
    lm = LM(
        model="openrouter/openai/gpt-5-mini",
        rate_limit=1000,
        max_batch_size=2,
        max_retries=2,
        cache=cache,
    )
    msgs = [[{"role": "user", "content": f"p{i}"}] for i in range(3)]
    responses = [make_response(f"r{i}", "True", make_usage(10, 1)) for i in range(3)]
    responses[1] = retryable_exception()
    retry = make_response("r1-retry", "True", make_usage(10, 1))
    # rate-limited path chunks into sub-batches of max_batch_size=2:
    # dispatch [p0,p1] -> retry [p1] -> then [p2]
    fake = FakeBatch([[responses[0], responses[1]], [retry], [responses[2]]])
    monkeypatch.setattr(lm_module, "batch_completion", fake)

    out = lm(msgs, show_progress_bar=False)

    assert out.outputs == ["True"] * 3
    st = lm.stats
    assert st.physical_requests == 4
    assert st.retry_attempts == 1
    assert st.retryable_failures == 1
    # rate-limited chunking: [p0,p1] -> retry [p1] -> then [p2]
    assert fake.calls == [["p0", "p1"], ["p1"], ["p2"]]
