"""Offline tests for cache validation, process-equivalent resume, deterministic
request reuse, and the runner's checkpoint manifest (no paid API calls)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from litellm.types.utils import Choices, Message, ModelResponse, Usage

import lotus
import lotus.models.lm as lm_module
from lotus.cache import CacheConfig, CacheFactory, CacheType
from lotus.models import LM
from lotus.types import CascadeArgs, LotusRetryExhaustedError

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
# shared offline builders (kept local to avoid cross-module import coupling)
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


OPENROUTER_FAILURE_PSF = {
    "error": {
        "code": 502,
        "message": "Stream ended before a terminal response event",
        "metadata": {"error_type": "provider_unavailable"},
    },
    "native_finish_reason": "error",
}


class FakeBatch:
    def __init__(self, responses_by_call):
        self.responses_by_call = list(responses_by_call)
        self.calls = []

    def __call__(self, model, messages, **kwargs):
        self.calls.append([m[0]["content"] for m in messages])
        return self.responses_by_call.pop(0)


# ---------------------------------------------------------------------------
# Test 5 â€” poisoned SQLite cache entry is rejected, deleted, treated as miss
# ---------------------------------------------------------------------------


def test_poisoned_sqlite_cache_entry_rejected_and_replaced(monkeypatch):
    lotus.settings.configure(enable_cache=True)
    cache = sqlite_cache()
    lm = LM(model="openrouter/openai/gpt-5-mini", max_retries=0, cache=cache)
    msgs = [[{"role": "user", "content": "p0"}]]
    h = lm._hash_messages(msgs[0], lm.kwargs)

    poisoned = make_response("gen-bad", None, usage=make_usage(120, 146), psf=OPENROUTER_FAILURE_PSF)
    cache.insert(h, poisoned)

    good = make_response("r0", "True", make_usage(10, 1))
    fake = FakeBatch([[good]])
    monkeypatch.setattr(lm_module, "batch_completion", fake)

    out = lm(msgs, show_progress_bar=False)

    assert out.outputs == ["True"]
    assert len(fake.calls) == 1  # poisoned entry treated as a miss -> provider called
    replaced = cache.get(h)
    assert replaced is not None and replaced.choices[0].message.content == "True"
    assert replaced.id == "r0"  # valid response replaced the poisoned entry


# ---------------------------------------------------------------------------
# Test 6 â€” process-equivalent resume
# ---------------------------------------------------------------------------


def test_process_equivalent_resume(monkeypatch):
    lotus.settings.configure(enable_cache=True)
    msgs = [[{"role": "user", "content": f"p{i}"}] for i in range(4)]
    cache_dir = make_work_dir("shared_cache")  # both "processes" share this DB

    # ---- attempt 1: p0,p1,p3 succeed and persist; p2 always retryable -> exhausted
    cache1 = CacheFactory.create_cache(CacheConfig(CacheType.SQLITE, 1024, cache_dir=str(cache_dir)))
    lm1 = LM(model="openrouter/openai/gpt-5-mini", max_batch_size=4, max_retries=1, cache=cache1)
    fail2 = make_response("bad2", None, usage=make_usage(10, 10), psf=OPENROUTER_FAILURE_PSF)
    responses = [make_response(f"r{i}", "True", make_usage(10, 1)) for i in range(4)]
    responses[2] = fail2
    fake1 = FakeBatch([responses, [fail2]])
    monkeypatch.setattr(lm_module, "batch_completion", fake1)

    with pytest.raises(LotusRetryExhaustedError):
        lm1(msgs, show_progress_bar=False)

    hashes1 = [lm1._hash_messages(m, lm1.kwargs) for m in msgs]
    for i in (0, 1, 3):
        assert cache1.get(hashes1[i]) is not None, f"p{i} not persisted"
    assert cache1.get(hashes1[2]) is None, "failed prompt must not be cached"

    # ---- attempt 2 (process-equivalent): same DB; only the missing prompt reaches provider
    cache2 = CacheFactory.create_cache(CacheConfig(CacheType.SQLITE, 1024, cache_dir=str(cache_dir)))
    lm2 = LM(model="openrouter/openai/gpt-5-mini", max_batch_size=4, max_retries=1, cache=cache2)
    ok2 = make_response("r2-ok", "True", make_usage(10, 1))
    fake2 = FakeBatch([[ok2]])
    monkeypatch.setattr(lm_module, "batch_completion", fake2)

    out = lm2(msgs, show_progress_bar=False)

    assert out.outputs == ["True"] * 4
    assert [c for c in fake2.calls] == [["p2"]]  # only the missing request was executed
    assert lm2.stats.cache_hits == 3


# ---------------------------------------------------------------------------
# Test 8 â€” deterministic sampling and request hashes
# ---------------------------------------------------------------------------


def test_deterministic_sampling_and_request_hashes():
    from lotus.sem_ops.cascade_utils import importance_sampling

    scores = [0.05 + (i % 11) * 0.03 for i in range(3000)]

    def sample():
        return importance_sampling(
            scores,
            CascadeArgs(
                sampling_percentage=0.005,
                cascade_IS_random_seed=0,
                cascade_IS_max_sample_range=200,
            ),
        )

    idx1, cf1 = sample()
    idx2, cf2 = sample()
    np.testing.assert_array_equal(idx1, idx2)
    np.testing.assert_array_equal(cf1, cf2)

    lm1 = LM(model="openrouter/openai/gpt-5-mini")
    lm2 = LM(model="openrouter/openai/gpt-5-mini")
    msgs = [[{"role": "user", "content": "hello world"}]]
    assert lm1._hash_messages(msgs[0], lm1.kwargs) == lm2._hash_messages(msgs[0], lm2.kwargs)


# ---------------------------------------------------------------------------
# Test 12 â€” runner manifest: MUST-MATCH refusal before any API request
# ---------------------------------------------------------------------------


def _load_runner_module():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "run_sem_join_bench", repo_root / "scripts" / "run_sem_join.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _base_config(runner):
    return {
        "manifest_version": runner.MANIFEST_VERSION,
        "cache_schema_version": runner.CACHE_SCHEMA_VERSION,
        "family": "autofj",
        "dataset": "Amphibian",
        "source_rows": 1161,
        "target_rows": 3663,
        "source_fingerprint": "sha256:aaa",
        "target_fingerprint": "sha256:bbb",
        "source_id_col": "id",
        "source_value_col": "title",
        "target_id_col": "id",
        "target_value_col": "title",
        "join_instruction": "{source_value} and {target_value} refer to the same real-world entity.",
        "lm_model": "openrouter/openai/gpt-5-mini",
        "embedding_model": "openrouter/openai/text-embedding-3-small",
        "lm_kwargs": {"temperature": 0.0, "max_completion_tokens": 8192},
        "cascade_seed": 0,
        "sampling_percentage": 0.005,
        "recall_target": 0.8,
        "precision_target": 0.8,
        "failure_probability": 0.2,
        "min_join_cascade_size": 100,
        "lotus_commit": "abc",
        "litellm_version": "1.94.0",
    }


def test_manifest_mismatch_refuses_resume():
    runner = _load_runner_module()
    checkpoint_dir = make_work_dir("checkpoint")
    lotus_work_dir = make_work_dir("lotus_work")
    config = _base_config(runner)

    # fresh run creates the manifest
    runner.prepare_checkpoint(
        checkpoint_dir=checkpoint_dir, lotus_work_dir=lotus_work_dir, config=config, resume=False
    )
    # identical config resumes
    runner.prepare_checkpoint(
        checkpoint_dir=checkpoint_dir, lotus_work_dir=lotus_work_dir, config=dict(config), resume=True
    )
    # altered MUST-MATCH field (data fingerprint) -> refused before any model request
    bad = dict(config, source_fingerprint="sha256:CHANGED")
    with pytest.raises(ValueError, match="source_fingerprint"):
        runner.prepare_checkpoint(
            checkpoint_dir=checkpoint_dir, lotus_work_dir=lotus_work_dir, config=bad, resume=True
        )
    # informational-only change (batch size) does NOT block a valid resume
    info = dict(config, llm_batch_size=999, retry_config={"max_retries": 1})
    runner.prepare_checkpoint(
        checkpoint_dir=checkpoint_dir, lotus_work_dir=lotus_work_dir, config=info, resume=True
    )
    # missing manifest refuses a resume
    with pytest.raises(FileNotFoundError):
        runner.prepare_checkpoint(
            checkpoint_dir=make_work_dir("empty"),
            lotus_work_dir=make_work_dir("w2"),
            config=config,
            resume=True,
        )


def test_fingerprint_detects_same_row_count_edit():
    runner = _load_runner_module()
    f1 = runner.fingerprint_values([0, 1], ["apple", "banana"])
    assert runner.fingerprint_values([0, 1], ["apple", "banana"]) == f1  # deterministic
    assert runner.fingerprint_values([0, 1], ["apple", "banana!"]) != f1  # value edit
    assert runner.fingerprint_values([1, 0], ["apple", "banana"]) != f1  # row-order edit
    f_none = runner.fingerprint_values([0, 1], ["apple", None])
    f_nan = runner.fingerprint_values([0, 1], ["apple", float("nan")])
    assert f_none != f1
    assert f_nan == f_none  # missing values canonicalized deterministically
