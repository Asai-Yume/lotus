"""Offline tests for LiteLLMRM embedding retries (no paid API calls).

Monkeypatches ``lotus.models.litellm_rm.embedding`` with scripted behavior:
transient exception -> retry, malformed result count -> retry, permanent
failure -> no retry, and retry exhaustion for a malformed response.
"""

import litellm
import numpy as np
import pytest

import lotus.models.litellm_rm as rm_module
from lotus.models import LiteLLMRM


class FakeHttpxResponse:
    status_code = 502
    headers = {}
    request = object()


class FakeEmbeddingResponse:
    def __init__(self, count, dim=2):
        self.data = [{"embedding": [0.1 * (i + 1), 0.2 * (i + 1)], "index": i} for i in range(count)]


# ---------------------------------------------------------------------------
# Test 9 — embedding transient failure retried, then success
# ---------------------------------------------------------------------------


def test_embedding_transient_failure_retried_then_success(monkeypatch):
    calls = []

    def fake_embedding(model, input):
        calls.append(len(input))
        if len(calls) == 1:
            raise litellm.exceptions.ServiceUnavailableError(
                message="provider down",
                llm_provider="openrouter",
                model="openai/text-embedding-3-small",
                response=FakeHttpxResponse(),
            )
        return FakeEmbeddingResponse(len(input))

    monkeypatch.setattr(rm_module, "embedding", fake_embedding)
    rm = LiteLLMRM(model="openrouter/openai/text-embedding-3-small", max_batch_size=2, max_retries=2)

    out = rm(["a", "b", "c"])

    assert calls == [2, 2, 1]  # batch1 fails once then retries; batch2 succeeds first try
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out[1], [0.2, 0.4])  # unchanged embedding values


def test_embedding_malformed_response_retried_then_success(monkeypatch):
    calls = []

    def fake_embedding(model, input):
        calls.append(len(input))
        if len(calls) == 1:
            return FakeEmbeddingResponse(1)  # wrong result count for 2 inputs
        return FakeEmbeddingResponse(len(input))

    monkeypatch.setattr(rm_module, "embedding", fake_embedding)
    rm = LiteLLMRM(model="openrouter/openai/text-embedding-3-small", max_batch_size=2, max_retries=2)

    out = rm(["a", "b"])

    assert calls == [2, 2]
    assert out.shape == (2, 2)


def test_embedding_malformed_exhausts_after_retries(monkeypatch):
    calls = []

    def fake_embedding(model, input):
        calls.append(len(input))
        return FakeEmbeddingResponse(0)  # always malformed (empty data)

    monkeypatch.setattr(rm_module, "embedding", fake_embedding)
    rm = LiteLLMRM(model="openrouter/openai/text-embedding-3-small", max_batch_size=2, max_retries=1)

    with pytest.raises(RuntimeError, match="results for"):
        rm(["a", "b"])

    assert len(calls) == 2  # 1 initial attempt + 1 retry (max_retries=1)


def test_embedding_permanent_failure_not_retried(monkeypatch):
    calls = []

    def fake_embedding(model, input):
        calls.append(len(input))
        raise litellm.exceptions.AuthenticationError(
            message="invalid api key",
            llm_provider="openrouter",
            model="openai/text-embedding-3-small",
            response=FakeHttpxResponse(),
        )

    monkeypatch.setattr(rm_module, "embedding", fake_embedding)
    rm = LiteLLMRM(model="openrouter/openai/text-embedding-3-small", max_batch_size=2, max_retries=5)

    with pytest.raises(litellm.exceptions.AuthenticationError):
        rm(["a", "b"])

    assert len(calls) == 1  # permanent: no retries


def test_embedding_max_retries_zero_preserves_no_retry_behavior(monkeypatch):
    calls = []

    def fake_embedding(model, input):
        calls.append(len(input))
        raise litellm.exceptions.ServiceUnavailableError(
            message="provider down",
            llm_provider="openrouter",
            model="openai/text-embedding-3-small",
            response=FakeHttpxResponse(),
        )

    monkeypatch.setattr(rm_module, "embedding", fake_embedding)
    rm = LiteLLMRM(model="openrouter/openai/text-embedding-3-small", max_batch_size=2)  # default 0

    with pytest.raises(litellm.exceptions.ServiceUnavailableError):
        rm(["a", "b"])

    assert len(calls) == 1
