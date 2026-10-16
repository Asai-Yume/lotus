"""Offline tests for the corrected Top-1 + Any-Join-style batch verification.

Covers the batched verifier contract only (no paid calls, model boundary mocked):

* 100 pairs -> 1 batch, 101 pairs -> 2 batches, <100 pairs -> 1 batch
* ID-based alignment when the model reorders items
* missing id -> no positional shifting
* out-of-range id -> ignored, pair rejected
* malformed response -> documented default=False policy
* "Yes" accepted / "No" rejected
* final partial batch
* prompt matches Any-Join's batch_values_prompt semantics
* dry-run performs no LLM call and no embedding load
* historical (sem_filter) verification path remains the default
"""

import argparse
import json
import os
import sys
import types
import uuid
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import run_top1_llm_verify as runner_module  # noqa: E402


def make_work_dir(prefix: str) -> Path:
    base = _REPO_ROOT / "outputs" / "_retry_test_work"
    base.mkdir(parents=True, exist_ok=True)
    d = base / f"{prefix}_{uuid.uuid4().hex[:8]}"
    os.makedirs(d, exist_ok=True)
    return d


class RecordingLM:
    """Minimal LM stand-in: records prompts, returns canned responses.

    The corrected verifier calls the LM directly with a single conversation
    (mirroring Any-Join's request shape), so this stand-in implements ``__call__``.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.calls = 0

    def __call__(self, messages, show_progress_bar=True, progress_bar_desc="", **kwargs):
        conversation = messages[0]
        system_prompt = conversation[0]["content"]
        user_prompt = conversation[1]["content"]
        self.prompts.append((system_prompt, user_prompt))
        self.calls += 1
        if not self.responses:
            raise AssertionError("RecordingLM received more calls than canned responses")
        return types.SimpleNamespace(outputs=[self.responses.pop(0)])


def _args(**overrides):
    base = dict(
        embedding_model=runner_module.DEFAULT_EMBEDDING_MODEL,
        lm_model=runner_module.DEFAULT_LM_MODEL,
        embedding_batch_size=64,
        llm_batch_size=8,
        max_retries=5,
        retry_backoff_seconds=2.0,
        join_instruction=runner_module.DEFAULT_JOIN_INSTRUCTION,
        dry_run=False,
        continue_on_error=False,
        output_root=Path("."),
        source_id_col=None,
        source_value_col=None,
        target_id_col=None,
        target_value_col=None,
        gt_source_col=None,
        gt_target_col=None,
        embedding_backend="litellm",
        verification_mode="anyjoin_batch",
        verification_batch_size=100,
        lm_max_tokens=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _candidates(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_id": [str(i) for i in range(n)],
            "target_value": [f"T{i}" for i in range(n)],
            "source_id": [str(i) for i in range(n)],
            "source_value": [f"S{i}" for i in range(n)],
            "similarity_score": [0.5] * n,
        }
    )


def _yes_no(items):
    return json.dumps([{"id": i, "response": r} for i, r in items])


# ---------------------------------------------------------------------------
# Batching boundaries
# ---------------------------------------------------------------------------


def test_exactly_100_pairs_is_one_batch():
    model = RecordingLM([_yes_no([(i, "Yes") for i in range(100)])])
    verdicts, totals = runner_module.verify_candidates_batched(_candidates(100), model, _args())
    assert model.calls == 1
    assert totals["verification_batches"] == 1
    assert all(verdicts)
    assert totals["verification_pairs_verified"] == 100
    assert totals["verification_pairs_unresolved"] == 0
    # the single prompt must contain all 100 numbered pairs
    _, user_prompt = model.prompts[0]
    assert '0. Value 1: "S0" | Value 2: "T0"' in user_prompt
    assert '99. Value 1: "S99" | Value 2: "T99"' in user_prompt


def test_101_pairs_is_two_batches():
    responses = [
        _yes_no([(i, "No") for i in range(100)]),
        _yes_no([(0, "Yes")]),
    ]
    model = RecordingLM(responses)
    verdicts, totals = runner_module.verify_candidates_batched(_candidates(101), model, _args())
    assert model.calls == 2
    assert totals["verification_batches"] == 2
    assert verdicts[:100] == [False] * 100
    assert verdicts[100] is True
    # second batch re-indexes locally from 0 (Any-Join local ids)
    _, second_prompt = model.prompts[1]
    assert '0. Value 1: "S100" | Value 2: "T100"' in second_prompt


def test_fewer_than_100_pairs_is_one_batch():
    model = RecordingLM([_yes_no([(i, "Yes") for i in range(7)])])
    verdicts, totals = runner_module.verify_candidates_batched(_candidates(7), model, _args())
    assert model.calls == 1
    assert totals["verification_batches"] == 1
    assert verdicts == [True] * 7


def test_final_partial_batch():
    # 250 pairs -> batches of 100, 100, 50
    responses = [
        _yes_no([(i, "Yes") for i in range(100)]),
        _yes_no([(i, "Yes") for i in range(100)]),
        _yes_no([(i, "Yes") for i in range(50)]),
    ]
    model = RecordingLM(responses)
    verdicts, totals = runner_module.verify_candidates_batched(_candidates(250), model, _args())
    assert model.calls == 3
    assert totals["verification_batches"] == 3
    assert len(verdicts) == 250
    assert all(verdicts)


# ---------------------------------------------------------------------------
# ID-aligned parsing (no positional shifting)
# ---------------------------------------------------------------------------


def test_reordered_ids_still_map_correctly():
    # Model returns items out of order; ids must drive alignment.
    items = [(2, "Yes"), (0, "No"), (4, "Yes"), (1, "No"), (3, "Yes")]
    model = RecordingLM([_yes_no(items)])
    verdicts, totals = runner_module.verify_candidates_batched(_candidates(5), model, _args())
    assert verdicts == [False, False, True, True, True]
    assert totals["verification_pairs_verified"] == 5


def test_missing_id_does_not_shift_later_verdicts():
    # id 1 is omitted: pair 1 must not inherit id 2's "Yes".
    items = [(0, "No"), (2, "Yes"), (3, "Yes")]
    model = RecordingLM([_yes_no(items)])
    verdicts, totals = runner_module.verify_candidates_batched(_candidates(4), model, _args())
    assert verdicts == [False, False, True, True]
    assert totals["verification_pairs_unresolved"] == 1  # pair 1 unresolved -> rejected


def test_out_of_range_id_is_ignored_and_pair_rejected():
    items = [(0, "Yes"), (1, "Yes"), (99, "Yes"), (-1, "Yes")]
    model = RecordingLM([_yes_no(items)])
    verdicts, totals = runner_module.verify_candidates_batched(_candidates(3), model, _args())
    assert verdicts == [True, True, False]  # index 2 never answered -> rejected
    assert totals["verification_out_of_range_ids"] == 2
    assert totals["verification_pairs_unresolved"] == 1


def test_duplicate_id_last_value_wins_like_anyjoin():
    items = [(0, "No"), (0, "Yes"), (1, "No")]
    model = RecordingLM([_yes_no(items)])
    verdicts, totals = runner_module.verify_candidates_batched(_candidates(2), model, _args())
    assert verdicts == [True, False]
    assert totals["verification_duplicate_ids"] == 1


def test_malformed_response_rejects_all_pairs_in_batch():
    model = RecordingLM(["this is not json at all"])
    verdicts, totals = runner_module.verify_candidates_batched(_candidates(3), model, _args())
    assert verdicts == [False, False, False]
    assert totals["verification_malformed_responses"] == 3
    assert totals["verification_pairs_unresolved"] == 3


def test_markdown_fenced_response_is_parsed():
    raw = "```json\n" + _yes_no([(0, "Yes"), (1, "No")]) + "\n```"
    model = RecordingLM([raw])
    verdicts, _ = runner_module.verify_candidates_batched(_candidates(2), model, _args())
    assert verdicts == [True, False]


def test_bare_dict_response_is_normalised_to_single_item():
    raw = json.dumps({"id": 0, "response": "Yes"})
    model = RecordingLM([raw])
    verdicts, totals = runner_module.verify_candidates_batched(_candidates(1), model, _args())
    assert verdicts == [True]
    assert totals["verification_pairs_verified"] == 1


def test_missing_id_field_is_counted():
    raw = json.dumps([{"response": "Yes"}, {"id": 1, "response": "Yes"}])
    model = RecordingLM([raw])
    verdicts, totals = runner_module.verify_candidates_batched(_candidates(2), model, _args())
    assert verdicts == [False, True]
    assert totals["verification_missing_ids"] == 1


# ---------------------------------------------------------------------------
# Yes / No semantics
# ---------------------------------------------------------------------------


def test_yes_is_accepted_and_no_is_rejected():
    model = RecordingLM([_yes_no([(0, "Yes"), (1, "No"), (2, "yes"), (3, "NO")])])
    verdicts, _ = runner_module.verify_candidates_batched(_candidates(4), model, _args())
    assert verdicts == [True, False, True, False]


def test_non_yes_strings_are_rejected():
    model = RecordingLM([_yes_no([(0, "Maybe"), (1, ""), (2, "true")])])
    verdicts, _ = runner_module.verify_candidates_batched(_candidates(3), model, _args())
    assert verdicts == [False, False, False]


def test_acceptance_semantics_are_exactly_anyjoin():
    # Any-Join: str(res.get("response", "")).lower() == "yes" -- no stripping,
    # no boolean handling, no extra permissiveness.
    model = RecordingLM(
        [_yes_no([(0, "Yes"), (1, "yes"), (2, "YES"), (3, "Yes "), (4, " Yes"), (5, "y")])]
    )
    verdicts, _ = runner_module.verify_candidates_batched(_candidates(6), model, _args())
    # exact lowercase match accepts "Yes"/"yes"/"YES"; whitespace variants are
    # rejected because Any-Join does not strip.
    assert verdicts == [True, True, True, False, False, False]


def test_boolean_true_response_is_rejected_like_anyjoin():
    # str(True).lower() == "true" != "yes" -> rejected, matching the reference.
    raw = json.dumps([{"id": 0, "response": True}])
    model = RecordingLM([raw])
    verdicts, _ = runner_module.verify_candidates_batched(_candidates(1), model, _args())
    assert verdicts == [False]


def test_boolean_false_response_is_rejected():
    raw = json.dumps([{"id": 0, "response": False}])
    model = RecordingLM([raw])
    verdicts, _ = runner_module.verify_candidates_batched(_candidates(1), model, _args())
    assert verdicts == [False]


def test_response_is_yes_helper_matches_reference_expression():
    for value in ("Yes", "yes", "YES", "yEs"):
        assert runner_module._response_is_yes(value) is True
    for value in ("No", "no", "Yes ", " Yes", "true", "True", "", "1", None, True, False):
        assert runner_module._response_is_yes(value) is False


# ---------------------------------------------------------------------------
# json_repair-based parsing (Any-Join's exact parser)
# ---------------------------------------------------------------------------


def test_uses_json_repair_loads(monkeypatch):
    """The batched parser must call json_repair.loads exactly like Any-Join."""
    assert runner_module.json_repair is not None, "json-repair must be installed"
    calls = []
    real_loads = runner_module.json_repair.loads

    def spy(raw, *args, **kwargs):
        calls.append(raw)
        return real_loads(raw, *args, **kwargs)

    monkeypatch.setattr(runner_module.json_repair, "loads", spy)
    model = RecordingLM([_yes_no([(0, "Yes")])])
    verdicts, _ = runner_module.verify_candidates_batched(_candidates(1), model, _args())
    assert verdicts == [True]
    assert len(calls) == 1
    assert '"response": "Yes"' in calls[0]


def test_json_repair_repairs_trailing_comma_and_single_quotes():
    trailing = '[{"id": 0, "response": "Yes"},]'
    single_quotes = "[{'id': 0, 'response': 'Yes'}]"
    model = RecordingLM([trailing, single_quotes])
    v1, t1 = runner_module.verify_candidates_batched(_candidates(1), model, _args())
    v2, t2 = runner_module.verify_candidates_batched(_candidates(1), model, _args())
    assert v1 == [True] and t1["verification_malformed_responses"] == 0
    assert v2 == [True] and t2["verification_malformed_responses"] == 0


def test_json_repair_recovers_array_from_prose():
    raw = 'Sure, here is the JSON: [{"id": 0, "response": "Yes"}] hope that helps'
    model = RecordingLM([raw])
    verdicts, totals = runner_module.verify_candidates_batched(_candidates(1), model, _args())
    assert verdicts == [True]
    assert totals["verification_malformed_responses"] == 0


def test_empty_candidate_frame_makes_no_call():
    model = RecordingLM([])
    verdicts, totals = runner_module.verify_candidates_batched(_candidates(0), model, _args())
    assert model.calls == 0
    assert verdicts == []
    assert totals["verification_batches"] == 0


# ---------------------------------------------------------------------------
# Prompt equivalence with Any-Join's batch_values_prompt
# ---------------------------------------------------------------------------


def test_prompt_contains_anyjoin_system_and_schema():
    system_msg, user_msg = runner_module.anyjoin_batch_values_prompt(
        [("NY", "New York")], mode="equivalence", include_explanation=False
    )
    assert system_msg.startswith("You are an expert Data Entity Resolution Specialist.")
    assert "JSON array" in system_msg
    assert "exact same real-world entity" in user_msg
    assert "Criteria for \"Yes\"" in user_msg
    assert "Criteria for \"No\"" in user_msg
    assert '0. Value 1: "NY" | Value 2: "New York"' in user_msg
    assert 'Output a JSON array: [{"id": <int>, "response": "Yes" or "No"}, ...]' in user_msg


def test_prompt_rejects_invalid_mode():
    with pytest.raises(ValueError):
        runner_module.anyjoin_batch_values_prompt([("a", "b")], mode="nonsense")


# ---------------------------------------------------------------------------
# Backward compatibility + dry run
# ---------------------------------------------------------------------------


def test_defaults_keep_historical_path():
    parser = runner_module.build_parser(_REPO_ROOT)
    args = parser.parse_args(["--family", "wt", "--all", "--dry-run"])
    assert args.embedding_backend == "litellm"
    assert args.verification_mode == "sem_filter"
    assert args.verification_batch_size == 100
    assert args.embedding_model == runner_module.DEFAULT_EMBEDDING_MODEL
    assert args.lm_model == runner_module.DEFAULT_LM_MODEL
    # no invented seed control
    assert not any("--seed" in str(action.option_strings) for action in parser._actions)


def test_corrected_flags_parse():
    parser = runner_module.build_parser(_REPO_ROOT)
    args = parser.parse_args(
        [
            "--family", "wt", "--all", "--dry-run",
            "--embedding-backend", "sentence_transformers",
            "--embedding-model", "BAAI/bge-base-en-v1.5",
            "--verification-mode", "anyjoin_batch",
            "--verification-batch-size", "100",
            "--lm-max-tokens", "16384",
        ]
    )
    assert args.embedding_backend == "sentence_transformers"
    assert args.embedding_model == "BAAI/bge-base-en-v1.5"
    assert args.verification_mode == "anyjoin_batch"
    assert args.verification_batch_size == 100
    assert args.lm_max_tokens == 16384


def test_accessor_defaults_when_attributes_absent():
    bare = argparse.Namespace()
    assert runner_module._embedding_backend(bare) == "litellm"
    assert runner_module._verification_mode(bare) == "sem_filter"
    assert runner_module._verification_batch_size(bare) == 100


def test_configure_models_builds_local_bge_rm_without_download(monkeypatch):
    """The corrected backend must construct SentenceTransformersRM for BGE.

    ``SentenceTransformer`` is monkeypatched so no model download/network access
    happens; this validates the wiring only.
    """
    import lotus.models.sentence_transformers_rm as st_module

    captured = {}

    class FakeTransformer:
        def __init__(self, model, device=None):
            captured["model"] = model
            captured["device"] = device

    monkeypatch.setattr(st_module, "SentenceTransformer", FakeTransformer)

    args = _args(
        embedding_backend="sentence_transformers",
        embedding_model="BAAI/bge-base-en-v1.5",
        embedding_batch_size=64,
        verification_mode="anyjoin_batch",
    )
    lm, rm, vs = runner_module.configure_models(args)
    assert isinstance(rm, runner_module.CountingSentenceTransformersRM)
    assert captured["model"] == "BAAI/bge-base-en-v1.5"
    assert rm.normalize_embeddings is True  # matches Any-Join's normalized embeddings
    assert rm.max_batch_size == 64
    # Any-Join registry prefixes are applied asymmetrically.
    assert rm.query_prefix == "Represent this sentence: "
    assert rm.passage_prefix == ""
    assert runner_module._embedding_query_prefix(args) == "Represent this sentence: "
    # counters exist for accounting parity with the hosted backend
    assert rm.embedding_api_requests == 0
    assert rm.embedding_batches == 0
    # temperature is supplied explicitly rather than left to the provider
    assert lm.kwargs.get("temperature") == 0.0


def test_litellm_backend_is_historical_default(monkeypatch):
    args = _args(embedding_backend="litellm")
    lm, rm, vs = runner_module.configure_models(args)
    assert isinstance(rm, runner_module.CountingLiteLLMRM)
    assert lm.kwargs.get("temperature") == 0.0
    # hosted embedding path keeps no asymmetric prefix
    assert runner_module._embedding_query_prefix(args) == ""


# ---------------------------------------------------------------------------
# Any-Join BGE query-prefix fidelity
# ---------------------------------------------------------------------------


def test_anyjoin_prefix_registry_matches_reference_entries():
    """The BGE/E5 entries must mirror Any-Join's utils.EMBEDDING_PREFIXES."""
    assert runner_module.ANYJOIN_EMBEDDING_PREFIXES["BAAI/bge-base-en-v1.5"] == (
        "Represent this sentence: ",
        "",
    )
    assert runner_module.ANYJOIN_EMBEDDING_PREFIXES["BAAI/bge-large-en-v1.5"] == (
        "Represent this sentence: ",
        "",
    )
    assert runner_module.ANYJOIN_EMBEDDING_PREFIXES["intfloat/e5-base-v2"] == (
        "query: ",
        "passage: ",
    )


def _make_prefix_recording_rm(monkeypatch, model="BAAI/bge-base-en-v1.5"):
    """Build a CountingSentenceTransformersRM with a recording fake encoder."""
    import torch
    import lotus.models.sentence_transformers_rm as st_module

    encoded: list[list[str]] = []

    class FakeTransformer:
        def __init__(self, model, device=None):
            pass

        def encode(self, batch, convert_to_tensor=False, normalize_embeddings=True, show_progress_bar=False):
            encoded.append(list(batch))
            return torch.ones((len(batch), 4))

    monkeypatch.setattr(st_module, "SentenceTransformer", FakeTransformer)
    query_prefix, passage_prefix = runner_module.ANYJOIN_EMBEDDING_PREFIXES.get(
        model, (None, None)
    )
    rm = runner_module.CountingSentenceTransformersRM(
        model=model,
        max_batch_size=64,
        normalize_embeddings=True,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
    )
    return rm, encoded


def test_query_side_gets_prefix_and_index_side_stays_raw(monkeypatch):
    rm, encoded = _make_prefix_recording_rm(monkeypatch)

    # Index/passage side (sem_index -> RM.__call__): raw values.
    rm(["Apple", "Banana"])
    # Query side (sem_sim_join -> convert_query_to_query_vector): prefixed.
    rm.convert_query_to_query_vector(["Apple", "Banana"])
    assert encoded[0] == ["Apple", "Banana"]
    assert encoded[1] == [
        "Represent this sentence: Apple",
        "Represent this sentence: Banana",
    ]


def test_query_prefix_applies_to_pandas_series_and_scalar(monkeypatch):
    import pandas as pd

    rm, encoded = _make_prefix_recording_rm(monkeypatch)
    rm.convert_query_to_query_vector(pd.Series(["Cherry"]))
    rm.convert_query_to_query_vector("Cherry")
    assert encoded[0] == ["Represent this sentence: Cherry"]
    assert encoded[1] == ["Represent this sentence: Cherry"]


def test_precomputed_query_vectors_pass_through_untouched(monkeypatch):
    import numpy as np

    rm, encoded = _make_prefix_recording_rm(monkeypatch)
    vectors = np.array([[0.1, 0.2], [0.3, 0.4]])
    out = rm.convert_query_to_query_vector(vectors)
    assert out is vectors
    assert encoded == []  # no re-encoding, therefore no prefixing


def test_unregistered_local_model_gets_no_prefix(monkeypatch):
    rm, encoded = _make_prefix_recording_rm(monkeypatch, model="sentence-transformers/all-mpnet-base-v2")
    rm.convert_query_to_query_vector(["Apple"])
    assert rm.query_prefix is None
    assert encoded[0] == ["Apple"]


def test_dry_run_makes_no_llm_or_embedding_call(monkeypatch):
    work = make_work_dir("bge_dry")
    dataset_dir = work / "wt" / "FakeDataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"src": ["a", "b"]}).to_csv(dataset_dir / "source.csv", index=False)
    pd.DataFrame({"tgt": ["x", "y"]}).to_csv(dataset_dir / "target.csv", index=False)
    pd.DataFrame({"source-src": ["a"], "target-tgt": ["x"]}).to_csv(
        dataset_dir / "ground truth.csv", index=False
    )
    (dataset_dir / "rows.txt").write_text("src:tgt\nsource\n", encoding="utf-8")

    def _explode(*args, **kwargs):
        raise AssertionError("configure_models must not run during a dry run")

    monkeypatch.setattr(runner_module, "configure_models", _explode)

    output_root = work / "out"
    result = runner_module.run_one_dataset(
        repo_root=_REPO_ROOT,
        family="wt",
        dataset_dir=dataset_dir,
        args=_args(
            dry_run=True,
            output_root=output_root,
            embedding_backend="sentence_transformers",
            embedding_model="BAAI/bge-base-en-v1.5",
            verification_mode="anyjoin_batch",
        ),
        repository_commit=None,
    )
    assert result.status == "dry_run"
    assert result.llm_physical_requests == 0
    assert result.embedding_api_requests == 0
    assert result.verification_batches == 0
    # expected logical requests are ceil(pairs/100)
    assert result.expected_logical_verification_calls == 1
    assert result.verification_batch_size == 100
