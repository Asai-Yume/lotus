"""Offline tests for the LM completion-budget omission flag (no paid calls).

Any-Join's ``batch_llm_verify`` sends no ``max_tokens`` /
``max_completion_tokens`` / ``max_output_tokens``. LOTUS's LM historically always
transmits its own budget (8192 for reasoning models). These tests verify, at the
mocked ``batch_completion`` boundary, that:

* the historical/default path still sends exactly what it sent before;
* the corrected Any-Join-aligned path omits the budget parameter *entirely*
  (the key is absent, not set to ``None``);
* no other request parameter (model, temperature) changes.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from litellm.types.utils import Choices, Message, ModelResponse, Usage

import lotus
import lotus.models.lm as lm_module
from lotus.models import LM

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import run_top1_llm_verify as runner_module  # noqa: E402

BUDGET_KEYS = ("max_completion_tokens", "max_tokens", "max_output_tokens")


def _fake_response(content: str) -> ModelResponse:
    return ModelResponse(
        id="r1",
        created=1,
        model="openai/gpt-5-mini",
        object="chat.completion",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(content=content, role="assistant"),
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


class BoundarySpy:
    """Records the exact kwargs handed to the provider boundary."""

    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = []

    def __call__(self, model, messages, **kwargs):
        self.calls.append(
            {
                "model": model,
                "kwargs": dict(kwargs),
                "messages": messages,
                "n_messages": len(messages),
            }
        )
        content = self.contents[min(len(self.calls) - 1, len(self.contents) - 1)]
        return [_fake_response(content) for _ in messages]


@pytest.fixture(autouse=True)
def _disable_cache():
    lotus.settings.configure(enable_cache=False)
    yield
    lotus.settings.configure(enable_cache=True)


# ---------------------------------------------------------------------------
# Core LM behaviour
# ---------------------------------------------------------------------------


def test_default_lm_sends_max_completion_tokens(monkeypatch):
    spy = BoundarySpy(["ok"])
    monkeypatch.setattr(lm_module, "batch_completion", spy)
    lm = LM(model="openrouter/openai/gpt-5-mini", temperature=0.0)
    out = lm([[{"role": "user", "content": "hi"}]], show_progress_bar=False)
    assert out.outputs == ["ok"]

    kwargs = spy.calls[0]["kwargs"]
    # Historical/default behaviour: the resolved reasoning-model budget is sent.
    assert kwargs["max_completion_tokens"] == 8192
    assert "max_tokens" not in kwargs
    assert "max_output_tokens" not in kwargs
    assert kwargs["temperature"] == 0.0
    assert kwargs["drop_params"] is True


def test_omit_flag_sends_no_completion_budget_key(monkeypatch):
    spy = BoundarySpy(["ok"])
    monkeypatch.setattr(lm_module, "batch_completion", spy)
    lm = LM(
        model="openrouter/openai/gpt-5-mini",
        temperature=0.0,
        send_max_completion_tokens=False,
    )
    out = lm([[{"role": "user", "content": "hi"}]], show_progress_bar=False)
    assert out.outputs == ["ok"]

    kwargs = spy.calls[0]["kwargs"]
    for key in BUDGET_KEYS:
        # Absent entirely -- crucially NOT present with a None value, which the
        # underlying SDK could serialize into the request body.
        assert key not in kwargs, f"{key} must be absent, got {kwargs.get(key)!r}"
    # Everything else is unchanged.
    assert kwargs["temperature"] == 0.0
    assert kwargs["drop_params"] is True


def test_omit_flag_drops_an_explicitly_passed_budget(monkeypatch):
    spy = BoundarySpy(["ok"])
    monkeypatch.setattr(lm_module, "batch_completion", spy)
    lm = LM(model="openrouter/openai/gpt-5-mini", send_max_completion_tokens=False)
    lm(
        [[{"role": "user", "content": "hi"}]],
        show_progress_bar=False,
        max_tokens=123,
        max_output_tokens=456,
    )
    kwargs = spy.calls[0]["kwargs"]
    for key in BUDGET_KEYS:
        assert key not in kwargs


def test_lm_stores_the_flag_with_default_true():
    assert LM(model="openrouter/openai/gpt-5-mini").send_max_completion_tokens is True
    assert (
        LM(model="openrouter/openai/gpt-5-mini", send_max_completion_tokens=False)
        .send_max_completion_tokens
        is False
    )


# ---------------------------------------------------------------------------
# Runner wiring
# ---------------------------------------------------------------------------


def test_omit_accessor_defaults_off():
    assert runner_module._omit_lm_completion_budget(argparse.Namespace()) is False
    assert (
        runner_module._omit_lm_completion_budget(
            argparse.Namespace(omit_lm_completion_budget=True)
        )
        is True
    )


def test_parser_exposes_omit_flag_off_by_default():
    parser = runner_module.build_parser(_REPO_ROOT)
    args = parser.parse_args(["--family", "wt", "--all", "--dry-run"])
    assert args.omit_lm_completion_budget is False
    args2 = parser.parse_args(
        ["--family", "wt", "--all", "--dry-run", "--omit-lm-completion-budget"]
    )
    assert args2.omit_lm_completion_budget is True


def test_conflicting_budget_flags_are_rejected():
    parser = runner_module.build_parser(_REPO_ROOT)
    args = parser.parse_args(
        [
            "--family", "wt", "--all", "--dry-run",
            "--lm-max-tokens", "16384",
            "--omit-lm-completion-budget",
        ]
    )
    with pytest.raises(SystemExit):
        runner_module.validate_args(parser, args)


def test_configure_models_routes_the_omit_flag(monkeypatch):
    import lotus.models.sentence_transformers_rm as st_module

    class FakeTransformer:
        def __init__(self, model, device=None):
            pass

    monkeypatch.setattr(st_module, "SentenceTransformer", FakeTransformer)

    def _args(**overrides):
        base = dict(
            lm_model="openrouter/openai/gpt-5-mini",
            llm_batch_size=8,
            max_retries=5,
            retry_backoff_seconds=2.0,
            embedding_model="BAAI/bge-base-en-v1.5",
            embedding_backend="sentence_transformers",
            embedding_batch_size=64,
            verification_mode="anyjoin_batch",
            verification_batch_size=100,
            lm_max_tokens=None,
            omit_lm_completion_budget=False,
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    lm_default, _, _ = runner_module.configure_models(_args())
    assert lm_default.send_max_completion_tokens is True

    lm_omitted, _, _ = runner_module.configure_models(
        _args(omit_lm_completion_budget=True)
    )
    assert lm_omitted.send_max_completion_tokens is False
    # temperature is still explicit either way
    assert lm_default.kwargs["temperature"] == 0.0
    assert lm_omitted.kwargs["temperature"] == 0.0


def test_corrected_verification_request_omits_budget_at_the_boundary(monkeypatch):
    """End-to-end: the corrected batched verifier's API kwargs carry no budget key."""
    spy = BoundarySpy(['[{"id": 0, "response": "Yes"}]'])
    monkeypatch.setattr(lm_module, "batch_completion", spy)

    lm = LM(
        model="openrouter/openai/gpt-5-mini",
        temperature=0.0,
        max_batch_size=8,
        max_retries=5,
        retry_backoff_seconds=2.0,
        send_max_completion_tokens=False,
    )
    candidates = pd.DataFrame(
        {
            "source_id": ["s0"],
            "target_id": ["t0"],
            "source_value": ["NY"],
            "target_value": ["New York"],
            "similarity_score": [0.9],
        }
    )
    args = argparse.Namespace(
        verification_mode="anyjoin_batch",
        verification_batch_size=100,
        omit_lm_completion_budget=True,
    )
    verdicts, totals = runner_module.verify_candidates_batched(candidates, lm, args)
    assert verdicts == [True]
    assert totals["verification_batches"] == 1

    assert len(spy.calls) == 1
    kwargs = spy.calls[0]["kwargs"]
    for key in BUDGET_KEYS:
        assert key not in kwargs, f"{key} leaked into the corrected request: {kwargs!r}"
    # Any-Join's create() call sends only model/temperature/messages, so no
    # response_format (which get_completion would inject as None) either.
    assert "response_format" not in kwargs
    assert kwargs["temperature"] == 0.0

    # the request still contains Any-Join's prompt shape (same call, no other change)
    call = spy.calls[0]
    assert call["n_messages"] == 1  # exactly one logical request / conversation
    conversation = call["messages"][0]
    assert [m["role"] for m in conversation] == ["system", "user"]
    assert '0. Value 1: "NY" | Value 2: "New York"' in conversation[1]["content"]
    assert "exact same real-world entity" in conversation[1]["content"]


def test_historical_verification_request_keeps_budget_at_the_boundary(monkeypatch):
    """The default path is unchanged: the budget key is still transmitted."""
    spy = BoundarySpy(["ok"])
    monkeypatch.setattr(lm_module, "batch_completion", spy)
    lm = LM(
        model="openrouter/openai/gpt-5-mini",
        temperature=0.0,
        max_batch_size=8,
        max_retries=5,
        retry_backoff_seconds=2.0,
    )
    assert lm.send_max_completion_tokens is True
    lm([[{"role": "user", "content": "hi"}]], show_progress_bar=False)
    kwargs = spy.calls[0]["kwargs"]
    assert kwargs["max_completion_tokens"] == 8192
    assert "max_tokens" not in kwargs
    assert "max_output_tokens" not in kwargs
