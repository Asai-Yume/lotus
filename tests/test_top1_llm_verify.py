"""Offline tests for the Top-1 + LLM Verify baseline (no paid API calls).

Covers: K=1 one candidate per source, ID/value/similarity-score preservation,
duplicate target values, verification alignment, parse-failure -> False,
accepted/rejected output behavior, retrieval-hit/recall diagnostics (incl.
multi-GT and partial-overlap), value-based families, dry-run zero model calls,
and full-path outputs/evaluation/accounting wiring.
"""

import argparse
import json
import logging
import os
import sys
import uuid
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from litellm.types.utils import Choices, Message, ModelResponse, Usage

import lotus
import lotus.models.lm as lm_module
from lotus.models import LM
from lotus.models.rm import RM
from lotus.sem_ops.sem_filter import sem_filter
from lotus.types import LMStats, SemanticFilterOutput
from lotus.vector_store import FaissVS

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from run_sem_sim_join import ResolvedSchema  # noqa: E402

import run_top1_llm_verify as runner_module  # noqa: E402


def make_work_dir(prefix: str) -> Path:
    base = _REPO_ROOT / "outputs" / "_retry_test_work"
    base.mkdir(parents=True, exist_ok=True)
    d = base / f"{prefix}_{uuid.uuid4().hex[:8]}"
    os.makedirs(d, exist_ok=True)
    return d


def _args(**overrides):
    base = dict(
        embedding_model="openrouter/openai/text-embedding-3-small",
        lm_model="openrouter/openai/gpt-5-mini",
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
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# Retrieval assembly
# ---------------------------------------------------------------------------


class FakeRM(RM):
    """Deterministic, network-free retrieval model (hash-based unit vectors)."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def _embed(self, docs):
        self.calls += 1
        rng = np.random.default_rng(2024)
        vecs = []
        for doc in docs:
            h = abs(hash(doc)) % (2**32)
            vecs.append(
                np.array(
                    [float(h % 11) / 11.0, float((h // 11) % 11) / 11.0, float((h // 121) % 11) / 11.0]
                )
            )
        arr = np.array(vecs)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1.0, norms)


def _configure_fixture(source, target):
    rm = FakeRM()
    lm = LM(model="openrouter/openai/gpt-5-mini")  # constructed, never called
    vs = FaissVS()
    lotus.settings.configure(lm=lm, rm=rm, vs=vs, enable_cache=True)
    return rm


def test_k1_one_candidate_per_source():
    source = pd.DataFrame({"source_id": ["s0", "s1", "s2"], "source_value": ["Alpha", "Beta", "Gamma"]})
    target = pd.DataFrame(
        {"target_id": ["t0", "t1", "t2", "t3", "t4"], "target_value": ["a", "b", "c", "d", "e"]}
    )
    _configure_fixture(source, target)
    index_dir = make_work_dir("idx")

    _, candidates, _, _ = runner_module.retrieve_top1(source, target, index_dir, _args())

    assert list(candidates.columns) == [
        "target_id",
        "target_value",
        "source_id",
        "source_value",
        "similarity_score",
    ]
    assert len(candidates) == len(source) == 3
    assert candidates["source_id"].nunique() == 3  # exactly one candidate per source
    assert candidates["source_id"].tolist() == ["s0", "s1", "s2"]  # order preserved


def test_ids_values_score_preserved():
    source = pd.DataFrame({"source_id": ["s0", "s1"], "source_value": ["Alpha", "Beta"]})
    target = pd.DataFrame({"target_id": ["t0", "t1", "t2"], "target_value": ["alpha", "beta", "gamma"]})
    _configure_fixture(source, target)
    index_dir = make_work_dir("idx")

    _, candidates, _, _ = runner_module.retrieve_top1(source, target, index_dir, _args())

    target_lookup = target.set_index("target_id")["target_value"].to_dict()
    for _, row in candidates.iterrows():
        assert row["target_id"] in target_lookup  # physical target ID recovered
        assert row["target_value"] == target_lookup[row["target_id"]]  # value consistent
        assert row["source_id"] in {"s0", "s1"}
        assert isinstance(row["similarity_score"], float)
        assert not pd.isna(row["similarity_score"])


def test_duplicate_target_values_physical_ids():
    source = pd.DataFrame({"source_id": ["s0", "s1", "s2"], "source_value": ["Alpha", "Beta", "Gamma"]})
    target = pd.DataFrame({"target_id": ["t0", "t1", "t2"], "target_value": ["dup", "dup", "unique"]})
    _configure_fixture(source, target)
    index_dir = make_work_dir("idx")

    _, candidates, _, _ = runner_module.retrieve_top1(source, target, index_dir, _args())

    assert len(candidates) == len(source)
    assert set(candidates["target_id"]).issubset(set(target["target_id"]))  # physical row IDs


def test_row_count_assertion():
    joined = pd.DataFrame(
        {
            "target_id": ["t0"],
            "target_value": ["a"],
            "source_id": ["s0"],
            "source_value": ["Alpha"],
            "_scores": [0.9],
        }
    )
    with pytest.raises(RuntimeError, match="expected 3 rows"):
        runner_module._assemble_candidates(joined, source_len=3)


# ---------------------------------------------------------------------------
# LLM verification
# ---------------------------------------------------------------------------


def _candidates():
    return pd.DataFrame(
        {
            "target_id": ["t0", "t1", "t2"],
            "target_value": ["Alpha Corp", "Beta Inc", "Gamma Ltd"],
            "source_id": ["s0", "s1", "s2"],
            "source_value": ["alpha corp", "Beta Incorporated", "gamma"],
            "similarity_score": [0.9, 0.8, 0.7],
        }
    )


def test_verification_alignment(monkeypatch):
    captured = {}

    class FakeOutput:
        outputs = [True, False, True]
        raw_outputs = ["Answer: True", "Answer: False", "Answer: True"]
        explanations = [None, None, None]

    def fake_sem_filter(docs, model, instruction, **kwargs):
        captured["docs"] = docs
        captured["instruction"] = instruction
        captured["default"] = kwargs.get("default")
        return FakeOutput()

    monkeypatch.setattr(runner_module, "sem_filter", fake_sem_filter)
    lm = LM(model="openrouter/openai/gpt-5-mini")

    verified = runner_module.verify_candidates(_candidates(), lm, _args())

    assert verified == [True, False, True]
    assert len(captured["docs"]) == 3  # one doc per candidate, order preserved
    assert captured["instruction"] == runner_module.DEFAULT_JOIN_INSTRUCTION
    assert captured["default"] is False


def test_parse_failure_defaults_false(monkeypatch):
    """Real sem_filter + real LM with a mocked provider returning garbage."""
    lotus.settings.configure(enable_cache=True)
    lm = LM(model="openrouter/openai/gpt-5-mini")

    def fake_batch_completion(model, messages, **kwargs):
        resp = ModelResponse(
            id="r1",
            created=1,
            model="openai/gpt-5-mini",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="this is not a boolean answer", role="assistant"),
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        return [resp]

    monkeypatch.setattr(lm_module, "batch_completion", fake_batch_completion)
    docs = [{"text": "source_value: x, target_value: y"}]
    output = sem_filter(docs, lm, runner_module.DEFAULT_JOIN_INSTRUCTION, default=False)
    assert output.outputs == [False]  # unparseable -> default False (reject)


def test_rejected_candidate_excluded_from_predictions():
    candidates = _candidates()
    candidates["llm_verified"] = [True, False, True]
    predictions = candidates.loc[candidates["llm_verified"] == True].copy()  # noqa: E712
    assert len(predictions) == 2
    assert "s1" not in predictions["source_id"].tolist()
    assert candidates["source_id"].tolist() == ["s0", "s1", "s2"]  # candidates keep all rows


def test_verify_candidates_length_mismatch_raises(monkeypatch):
    class BadOutput:
        outputs = [True]  # wrong length
        raw_outputs = ["Answer: True"]
        explanations = [None]

    monkeypatch.setattr(
        runner_module,
        "sem_filter",
        lambda *a, **k: BadOutput(),
    )
    lm = LM(model="openrouter/openai/gpt-5-mini")
    with pytest.raises(RuntimeError, match="different number of outputs"):
        runner_module.verify_candidates(_candidates(), lm, _args())


def test_semantic_filter_output_type_constructible():
    out = SemanticFilterOutput(
        raw_outputs=["Answer: True"],
        outputs=[True],
        explanations=[None],
    )
    assert out.outputs == [True]


# ---------------------------------------------------------------------------
# Retrieval/verification diagnostics
# ---------------------------------------------------------------------------


def _frames_id_based():
    source = pd.DataFrame({"source_id": ["s0", "s1"], "source_value": ["Alpha", "Beta"]})
    ground_truth = pd.DataFrame({"target_id": ["t0"], "source_id": ["s0"]})
    return source, ground_truth


def _candidates_id_based(hits_and_verified):
    rows = []
    for (target_id, source_id), verified in hits_and_verified:
        rows.append(
            {
                "target_id": target_id,
                "target_value": "v",
                "source_id": source_id,
                "source_value": "w",
                "similarity_score": 0.9,
                "llm_verified": verified,
            }
        )
    return pd.DataFrame(rows)


def test_correct_top1_id_based():
    source, gt = _frames_id_based()
    candidates = _candidates_id_based([(("t0", "s0"), True)])
    d = runner_module.top1_retrieval_diagnostics(candidates, gt, source, "autofj")
    assert d["top1_retrieval_hits"] == 1
    assert d["top1_retrieval_recall"] == 1.0
    assert d["retrieval_failures"] == 0


def test_incorrect_top1_id_based():
    source, gt = _frames_id_based()
    candidates = _candidates_id_based([(("t9", "s0"), True)])
    d = runner_module.top1_retrieval_diagnostics(candidates, gt, source, "autofj")
    assert d["top1_retrieval_hits"] == 0
    assert d["top1_retrieval_recall"] == 0.0
    assert d["retrieval_failures"] == 1  # GT-matched source without a hit


def test_multiple_valid_gt_targets():
    source = pd.DataFrame({"source_id": ["s0"], "source_value": ["Alpha"]})
    gt = pd.DataFrame({"target_id": ["t0", "t2"], "source_id": ["s0", "s0"]})
    candidates = _candidates_id_based([(("t2", "s0"), True)])
    d = runner_module.top1_retrieval_diagnostics(candidates, gt, source, "autofj")
    assert d["top1_retrieval_hits"] == 1  # any valid GT target counts as a hit
    assert d["top1_retrieval_recall"] == 1.0


def test_unmatched_source_excluded_from_denominator():
    # Partial-overlap: s1 has no GT pair. Its candidate must not shrink recall.
    source, gt = _frames_id_based()  # GT only mentions s0
    candidates = _candidates_id_based([(("t0", "s0"), True), (("t7", "s1"), False)])
    d = runner_module.top1_retrieval_diagnostics(candidates, gt, source, "autofj")
    assert d["top1_retrieval_hits"] == 1
    assert d["top1_retrieval_recall"] == 1.0  # denominator = 1 (only s0)
    assert d["retrieval_failures"] == 0


def test_verification_decomposition():
    source, gt = _frames_id_based()
    # s0: retrieval hit, rejected (verification FN). s1: no GT pair, but its
    # retrieved candidate (t7) was accepted -> verification false accept.
    candidates = _candidates_id_based([(("t0", "s0"), False), (("t7", "s1"), True)])
    d = runner_module.top1_retrieval_diagnostics(candidates, gt, source, "autofj")
    assert d["top1_retrieval_hits"] == 1
    assert d["verification_fn_given_retrieval_hit"] == 1
    assert d["verification_false_accepts"] == 1  # wrong candidate accepted by the LLM
    assert d["verification_recall_given_retrieval_hit"] == 0.0


def test_value_based_family():
    source = pd.DataFrame({"source_id": ["s0", "s1"], "source_value": ["Alpha", "Beta"]})
    gt = pd.DataFrame({"target_value": ["alpha corp"], "source_value": ["Alpha"]})
    candidates = pd.DataFrame(
        {
            "target_id": ["0"],
            "target_value": ["alpha corp"],
            "source_id": ["s0"],
            "source_value": ["Alpha"],
            "similarity_score": [0.9],
            "llm_verified": [True],
        }
    )
    d = runner_module.top1_retrieval_diagnostics(candidates, gt, source, "kbwt")
    assert d["top1_retrieval_hits"] == 1
    assert d["top1_retrieval_recall"] == 1.0


def test_duplicate_target_diagnostics():
    target = pd.DataFrame({"target_id": ["0", "1", "2"], "target_value": ["x", "x", "y"]})
    d = runner_module.duplicate_target_diagnostics(target)
    assert d["duplicate_target_value_rows"] == 2
    assert d["duplicate_target_values"] == 1


# ---------------------------------------------------------------------------
# Runner-level wiring
# ---------------------------------------------------------------------------


def _write_autofj_dataset(dataset_dir: Path):
    (dataset_dir / "right.csv").write_text("id,title\ns0,Alpha\ns1,Beta\n", encoding="utf-8")
    (dataset_dir / "left.csv").write_text("id,title\nt0,alpha\nt1,beta\n", encoding="utf-8")
    (dataset_dir / "gt.csv").write_text("id_r,id_l\ns0,t0\n", encoding="utf-8")


def test_dry_run_zero_calls(monkeypatch):
    dataset_dir = make_work_dir("ds")
    _write_autofj_dataset(dataset_dir)
    output_root = make_work_dir("out")

    def boom(*a, **k):
        raise AssertionError("dry run must not configure/use models")

    monkeypatch.setattr(runner_module, "configure_models", boom)
    monkeypatch.setattr(runner_module, "retrieve_top1", boom)
    monkeypatch.setattr(runner_module, "verify_candidates", boom)

    result = runner_module.run_one_dataset(
        repo_root=_REPO_ROOT,
        family="autofj",
        dataset_dir=dataset_dir,
        args=_args(dry_run=True, output_root=output_root),
        repository_commit=None,
    )

    assert result.status == "dry_run"
    assert result.source_rows == 2
    assert result.target_rows == 2
    assert result.cartesian_pairs == 4
    assert result.expected_top1_candidates == 2
    assert result.expected_logical_verification_calls == 2
    ds_out = output_root / "autofj" / dataset_dir.name
    assert (ds_out / "resolved_schema.json").exists()
    assert not (ds_out / "top1_candidates.csv").exists()
    assert not (ds_out / "predictions.csv").exists()


def test_full_path_outputs_evaluation_accounting(monkeypatch):
    dataset_dir = make_work_dir("ds2")
    output_root = make_work_dir("out2")

    source = pd.DataFrame({"source_id": ["s0", "s1", "s2"], "source_value": ["A", "B", "C"]})
    target = pd.DataFrame({"target_id": ["t0", "t1", "t2"], "target_value": ["a", "b", "c"]})
    ground_truth = pd.DataFrame({"target_id": ["t0", "t1"], "source_id": ["s0", "s1"]})
    schema = ResolvedSchema(
        source_file="right.csv",
        target_file="left.csv",
        ground_truth_file="gt.csv",
        source_id_col="id",
        source_value_col="title",
        target_id_col="id",
        target_value_col="title",
        gt_source_col="id_r",
        gt_target_col="id_l",
        source_id_is_index=False,
        target_id_is_index=False,
    )
    candidates = pd.DataFrame(
        {
            "target_id": ["t0", "t1", "t2"],
            "target_value": ["a", "b", "c"],
            "source_id": ["s0", "s1", "s2"],
            "source_value": ["A", "B", "C"],
            "similarity_score": [0.99, 0.98, 0.97],
        }
    )

    class FakeLM:
        stats = LMStats()

        def __init__(self):
            self.cache = type("C", (), {"reset": lambda self=None: None})()
            self.model = "openrouter/openai/gpt-5-mini"

    class FakeRM:
        embedding_rm_invocations = 2
        embedding_api_requests = 3
        embedded_rows = 4

    monkeypatch.setattr(runner_module, "load_dataset", lambda *a, **k: (source, target, ground_truth, schema))
    monkeypatch.setattr(runner_module, "configure_models", lambda args: (FakeLM(), FakeRM(), object()))
    monkeypatch.setattr(
        runner_module,
        "retrieve_top1",
        lambda source, target, index_dir, args: (target, candidates, 0.1, 0.2),
    )
    monkeypatch.setattr(runner_module, "verify_candidates", lambda candidates, model, args: [True, False, True])

    result = runner_module.run_one_dataset(
        repo_root=_REPO_ROOT,
        family="autofj",
        dataset_dir=dataset_dir,
        args=_args(output_root=output_root),
        repository_commit=None,
    )

    candidates_path = output_root / "autofj" / dataset_dir.name / "top1_candidates.csv"
    predictions_path = output_root / "autofj" / dataset_dir.name / "predictions.csv"
    metrics_path = output_root / "autofj" / dataset_dir.name / "metrics.json"

    cands = pd.read_csv(candidates_path)
    preds = pd.read_csv(predictions_path)
    assert len(cands) == 3
    assert set(cands["llm_verified"].tolist()) == {True, False}
    assert list(preds["source_id"]) == ["s0", "s2"]  # s1 rejected

    # The six historical columns keep their exact order and values; the four
    # metadata columns are appended at the output boundary.
    assert list(preds.columns) == [
        "target_id",
        "target_value",
        "source_id",
        "source_value",
        "similarity_score",
        "llm_verified",
        "dataset",
        "llm_model",
        "target_column",
        "source_column",
    ]
    assert preds["target_id"].tolist() == ["t0", "t2"]
    assert preds["target_value"].tolist() == ["a", "c"]
    assert preds["similarity_score"].tolist() == [0.99, 0.97]
    assert preds["llm_verified"].tolist() == [True, True]
    # dataset is the same identifier the run result and metrics record.
    assert preds["dataset"].tolist() == [dataset_dir.name, dataset_dir.name]
    assert result.dataset == dataset_dir.name
    # Effective provider-request model (litellm strips the "openrouter/" prefix).
    assert preds["llm_model"].tolist() == ["openai/gpt-5-mini", "openai/gpt-5-mini"]
    assert preds["target_column"].tolist() == ["title", "title"]
    assert preds["source_column"].tolist() == ["title", "title"]

    assert result.status == "success"
    assert result.retrieved_candidates == 3
    assert result.logical_verification_calls == 3
    assert result.accepted_pairs == 2
    assert result.rejected_candidates == 1
    assert result.top1_retrieval_hits == 2
    assert result.top1_retrieval_recall == 1.0  # denominator = 2 GT-matched sources
    assert result.verification_fn_given_retrieval_hit == 1  # s1: hit but rejected
    assert result.verification_false_accepts == 1  # s2: no GT but accepted
    assert result.embedding_api_requests == 3
    assert result.llm_physical_requests == 0
    assert result.llm_cache_hits == 0

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["precision"] == 0.5  # TP (s0) / 2 predictions
    assert metrics["recall"] == 0.5  # TP / 2 GT pairs


def test_effective_provider_model_routing():
    """The metadata resolver matches LiteLLM's own provider normalization."""
    assert runner_module._effective_provider_model("openrouter/openai/gpt-5-mini") == "openai/gpt-5-mini"
    assert runner_module._effective_provider_model("openai/gpt-5-mini") == "gpt-5-mini"
    assert runner_module._effective_provider_model("gpt-4o-mini") == "gpt-4o-mini"
    assert (
        runner_module._effective_provider_model("openrouter/anthropic/claude-3.5-sonnet")
        == "anthropic/claude-3.5-sonnet"
    )
    # A runtime --lm-model override flows through unchanged (provider-resolved).
    assert runner_module._effective_provider_model("openrouter/openai/gpt-4o") == "openai/gpt-4o"
    # Empty/unset model strings resolve to "metadata unavailable", no warning.
    assert runner_module._effective_provider_model("") == ""
    assert runner_module._effective_provider_model(None) == ""  # type: ignore[arg-type]


def test_effective_provider_model_resolution_failure(monkeypatch, caplog):
    """A failing resolver yields blank metadata plus a log record, never a raise.

    The report goes through ``lotus.logger`` rather than ``warnings`` so that
    warnings-as-errors cannot turn missing metadata into a failed run.
    """
    def boom(model):
        raise RuntimeError("resolver unavailable")

    monkeypatch.setattr(runner_module, "get_llm_provider", boom)

    with caplog.at_level(logging.WARNING):
        assert runner_module._effective_provider_model("openrouter/openai/gpt-5-mini") == ""
    assert any(
        "Could not resolve provider-request model metadata" in record.message
        for record in caplog.records
    )
    # No exception detail is leaked into the log record.
    assert all("resolver unavailable" not in record.getMessage() for record in caplog.records)

    # Empty input short-circuits before the resolver and stays silent.
    caplog.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with caplog.at_level(logging.WARNING):
            assert runner_module._effective_provider_model("") == ""
    assert caplog.records == []


def test_metadata_resolution_failure_still_writes_predictions(monkeypatch, caplog):
    """Forced metadata-resolution failure must not disturb prediction writing.

    ``warnings.simplefilter("error")`` is active for the whole run, proving the
    failure path cannot turn missing metadata into an aborted dataset.
    """
    dataset_dir = make_work_dir("ds_meta_fail")
    output_root = make_work_dir("out_meta_fail")

    source = pd.DataFrame({"source_id": ["s0", "s1"], "source_value": ["A", "B"]})
    target = pd.DataFrame({"target_id": ["t0", "t1"], "target_value": ["a", "b"]})
    ground_truth = pd.DataFrame({"target_id": ["t0"], "source_id": ["s0"]})
    schema = ResolvedSchema(
        source_file="right.csv",
        target_file="left.csv",
        ground_truth_file="gt.csv",
        source_id_col="id",
        source_value_col="title",
        target_id_col="id",
        target_value_col="title",
        gt_source_col="id_r",
        gt_target_col="id_l",
        source_id_is_index=False,
        target_id_is_index=False,
    )
    candidates = pd.DataFrame(
        {
            "target_id": ["t0", "t1"],
            "target_value": ["a", "b"],
            "source_id": ["s0", "s1"],
            "source_value": ["A", "B"],
            "similarity_score": [0.99, 0.97],
        }
    )

    class FakeLM:
        stats = LMStats()

        def __init__(self):
            self.cache = type("C", (), {"reset": lambda self=None: None})()
            self.model = "openrouter/openai/gpt-5-mini"

    class FakeRM:
        embedding_rm_invocations = 2
        embedding_api_requests = 3
        embedded_rows = 4

    def boom(model):
        raise RuntimeError("resolver unavailable")

    monkeypatch.setattr(runner_module, "get_llm_provider", boom)
    monkeypatch.setattr(
        runner_module, "load_dataset", lambda *a, **k: (source, target, ground_truth, schema)
    )
    monkeypatch.setattr(runner_module, "configure_models", lambda args: (FakeLM(), FakeRM(), object()))
    monkeypatch.setattr(
        runner_module, "retrieve_top1", lambda source, target, index_dir, args: (target, candidates, 0.1, 0.2)
    )
    monkeypatch.setattr(runner_module, "verify_candidates", lambda candidates, model, args: [True, True])

    # warnings-as-errors is active for the resolution failure: the report goes
    # through lotus.logger, so the dataset still completes instead of raising.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with caplog.at_level(logging.WARNING):
            result = runner_module.run_one_dataset(
                repo_root=_REPO_ROOT,
                family="autofj",
                dataset_dir=dataset_dir,
                args=_args(output_root=output_root),
                repository_commit=None,
            )
    assert any(
        "Could not resolve provider-request model metadata" in record.message
        for record in caplog.records
    )
    ds_out = output_root / "autofj" / dataset_dir.name
    preds = pd.read_csv(ds_out / "predictions.csv")

    # Predicted rows are still written, with the full ten-column schema.
    assert list(preds.columns) == [
        "target_id",
        "target_value",
        "source_id",
        "source_value",
        "similarity_score",
        "llm_verified",
        "dataset",
        "llm_model",
        "target_column",
        "source_column",
    ]
    assert len(preds) == 2
    # Every original field survives untouched.
    assert preds["target_id"].tolist() == ["t0", "t1"]
    assert preds["target_value"].tolist() == ["a", "b"]
    assert preds["source_id"].tolist() == ["s0", "s1"]
    assert preds["source_value"].tolist() == ["A", "B"]
    assert preds["similarity_score"].tolist() == [0.99, 0.97]
    assert preds["llm_verified"].tolist() == [True, True]
    # dataset is unaffected by the metadata-resolution failure.
    assert preds["dataset"].tolist() == [dataset_dir.name, dataset_dir.name]
    # Only the model metadata is unavailable; it is blank, never the raw id.
    assert preds["llm_model"].isna().all()  # blank fields read back as NaN
    assert "openrouter/openai/gpt-5-mini" not in preds["llm_model"].astype(str).tolist()
    assert preds["target_column"].tolist() == ["title", "title"]
    assert preds["source_column"].tolist() == ["title", "title"]

    # Evaluation and accounting still complete normally.
    #
    # Accounting rule: precision is true_positives / len(distinct predicted
    # pairs) and the evaluator applies NO annotation-domain restriction, so the
    # accepted-but-not-GT pair (t1, s1) counts against precision.  Both
    # candidates are accepted here, so pred_pairs = {(t0, s0), (t1, s1)} while
    # gt_pairs = {(t0, s0)}: TP=1, FP=1, FN=0, precision=0.5, recall=1.0.
    # metrics.json serialises the run-result fields (accepted_pairs == the
    # prediction-pair count); it contains no "prediction_pairs" key.
    metrics = json.loads((ds_out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["ground_truth_pairs"] == 1
    assert metrics["accepted_pairs"] == 2
    assert metrics["rejected_candidates"] == 0
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert result.status == "success"
    assert result.accepted_pairs == 2
    assert result.true_positives == 1
    assert result.false_positives == 1


def test_full_path_anyjoin_batch_outputs_schema(monkeypatch):
    """anyjoin_batch mode appends the same three metadata columns."""
    dataset_dir = make_work_dir("ds_batch")
    output_root = make_work_dir("out_batch")

    source = pd.DataFrame({"source_id": ["s0", "s1"], "source_value": ["A", "B"]})
    target = pd.DataFrame({"target_id": ["t0", "t1"], "target_value": ["a", "b"]})
    ground_truth = pd.DataFrame({"target_value": ["a"], "source_value": ["A"]})
    schema = ResolvedSchema(
        source_file="source.csv",
        target_file="target.csv",
        ground_truth_file="ground truth.csv",
        source_id_col="source_id",
        source_value_col="name",
        target_id_col="target_id",
        target_value_col="title",
        gt_source_col="source_value",
        gt_target_col="target_value",
        source_id_is_index=False,
        target_id_is_index=False,
    )
    candidates = pd.DataFrame(
        {
            "target_id": ["t0", "t1"],
            "target_value": ["a", "b"],
            "source_id": ["s0", "s1"],
            "source_value": ["A", "B"],
            "similarity_score": [0.9, 0.8],
        }
    )

    class FakeLM:
        stats = LMStats()

        def __init__(self):
            self.model = "openrouter/openai/gpt-5-mini"

    class FakeRM:
        embedding_rm_invocations = 1
        embedding_api_requests = 2
        embedded_rows = 3
        embedding_batches = 0

    monkeypatch.setattr(
        runner_module, "load_dataset", lambda *a, **k: (source, target, ground_truth, schema)
    )
    monkeypatch.setattr(runner_module, "configure_models", lambda args: (FakeLM(), FakeRM(), object()))
    monkeypatch.setattr(runner_module, "retrieve_top1", lambda s, t, i, a: (target, candidates, 0.1, 0.2))
    monkeypatch.setattr(
        runner_module,
        "verify_candidates_batched",
        lambda candidates, model, args: (
            [True, False],
            {
                "verification_batches": 1,
                "verification_pairs_verified": 2,
                "verification_pairs_unresolved": 0,
                "verification_malformed_responses": 0,
                "verification_missing_ids": 0,
                "verification_out_of_range_ids": 0,
                "verification_duplicate_ids": 0,
            },
        ),
    )

    result = runner_module.run_one_dataset(
        repo_root=_REPO_ROOT,
        family="kbwt",
        dataset_dir=dataset_dir,
        args=_args(
            output_root=output_root,
            verification_mode="anyjoin_batch",
            verification_batch_size=100,
        ),
        repository_commit=None,
    )

    ds_out = output_root / "kbwt" / dataset_dir.name
    preds = pd.read_csv(ds_out / "predictions.csv")
    assert list(preds.columns) == [
        "target_id",
        "target_value",
        "source_id",
        "source_value",
        "similarity_score",
        "llm_verified",
        "dataset",
        "llm_model",
        "target_column",
        "source_column",
    ]
    assert len(preds) == 1
    assert preds.iloc[0]["target_id"] == "t0"
    assert preds.iloc[0]["source_value"] == "A"
    assert preds.iloc[0]["dataset"] == dataset_dir.name
    assert preds.iloc[0]["llm_model"] == "openai/gpt-5-mini"
    assert preds.iloc[0]["target_column"] == "title"  # schema.target_value_col
    assert preds.iloc[0]["source_column"] == "name"  # schema.source_value_col

    # Batch provenance stays in top1_candidates.csv, never in predictions.csv.
    cands = pd.read_csv(ds_out / "top1_candidates.csv")
    assert "verification_batch_id" in cands.columns
    assert "verification_batch_position" in cands.columns
    assert "verification_batch_id" not in preds.columns
    assert "verification_batch_position" not in preds.columns

    assert result.status == "success"
    assert result.verification_mode == "anyjoin_batch"


# ---------------------------------------------------------------------------
# dataset metadata column: every family, both verification modes
# ---------------------------------------------------------------------------


def _run_mocked_dataset(monkeypatch, *, family, verification_mode, verdicts,
                        make_dataset, schema, source, target, ground_truth,
                        candidates, prefix, resolver=None, lm_model=None):
    """Run run_one_dataset with fully mocked externals and return (result, preds)."""
    dataset_dir = make_dataset()
    output_root = make_work_dir(f"out_{prefix}")

    class _FakeLM:
        stats = LMStats()

        def __init__(self):
            self.cache = type("C", (), {"reset": lambda self=None: None})()
            self.model = lm_model or "openrouter/openai/gpt-5-mini"

    class _FakeRM:
        embedding_rm_invocations = 2
        embedding_api_requests = 3
        embedded_rows = 4
        embedding_batches = 1

    monkeypatch.setattr(
        runner_module, "load_dataset",
        lambda *a, **k: (source, target, ground_truth, schema),
    )
    monkeypatch.setattr(
        runner_module, "configure_models", lambda args: (_FakeLM(), _FakeRM(), object())
    )
    monkeypatch.setattr(
        runner_module, "retrieve_top1",
        lambda s, t, i, a: (target, candidates.copy(), 0.1, 0.2),
    )
    monkeypatch.setattr(runner_module, "verify_candidates", lambda c, m, a: list(verdicts))
    monkeypatch.setattr(
        runner_module, "verify_candidates_batched",
        lambda c, m, a: (
            list(verdicts),
            {
                "verification_batches": 1,
                "verification_pairs_verified": len(verdicts),
                "verification_pairs_unresolved": 0,
                "verification_malformed_responses": 0,
                "verification_missing_ids": 0,
                "verification_out_of_range_ids": 0,
                "verification_duplicate_ids": 0,
            },
        ),
    )
    if resolver is not None:
        monkeypatch.setattr(runner_module, "get_llm_provider", resolver)

    result = runner_module.run_one_dataset(
        repo_root=_REPO_ROOT,
        family=family,
        dataset_dir=dataset_dir,
        args=_args(
            output_root=output_root,
            verification_mode=verification_mode,
            verification_batch_size=100,
            lm_model=lm_model or "openrouter/openai/gpt-5-mini",
        ),
        repository_commit=None,
    )
    preds_path = output_root / family / dataset_dir.name / "predictions.csv"
    return result, pd.read_csv(preds_path), preds_path, dataset_dir, output_root


def _family_case(family, source_value_col, target_value_col):
    """Fixtures for one family: frames, ground truth and a fresh dataset dir."""
    source = pd.DataFrame({"source_id": ["s0", "s1", "s2"], "source_value": ["A", "B", "C"]})
    target = pd.DataFrame({"target_id": ["t0", "t1", "t2"], "target_value": ["a", "b", "c"]})
    if family in ("kbwt", "ss", "wt"):
        ground_truth = pd.DataFrame({"target_value": ["a"], "source_value": ["A"]})
    else:
        ground_truth = pd.DataFrame({"target_id": ["t0"], "source_id": ["s0"]})
    candidates = pd.DataFrame(
        {
            "target_id": ["t0", "t1", "t2"],
            "target_value": ["a", "b", "c"],
            "source_id": ["s0", "s1", "s2"],
            "source_value": ["A", "B", "C"],
            "similarity_score": [0.99, 0.97, 0.95],
        }
    )
    schema = ResolvedSchema(
        source_file="source.csv",
        target_file="target.csv",
        ground_truth_file="ground truth.csv",
        source_id_col="source_id",
        source_value_col=source_value_col,
        target_id_col="target_id",
        target_value_col=target_value_col,
        gt_source_col="source_value",
        gt_target_col="target_value",
        source_id_is_index=False,
        target_id_is_index=False,
    )
    return source, target, ground_truth, candidates, schema


FAMILY_CASES = [
    ("autofj", "title", "title"),
    ("autofj_overlap", "title", "title"),
    ("kbwt", "airport code", "city"),
    ("ss", "value", "value"),
    ("wt", "fullname", "username"),
]


def _family_tmp_dir(family):
    return lambda: make_work_dir(f"ds_{family}")


@pytest.mark.parametrize("family,source_value_col,target_value_col", FAMILY_CASES)
@pytest.mark.parametrize("verification_mode", ["sem_filter", "anyjoin_batch"])
def test_dataset_column_all_families_both_modes(
    monkeypatch, family, source_value_col, target_value_col, verification_mode
):
    """dataset is populated for every family and both verification modes."""
    source, target, ground_truth, candidates, schema = _family_case(
        family, source_value_col, target_value_col
    )
    result, preds, _, dataset_dir, _ = _run_mocked_dataset(
        monkeypatch,
        family=family,
        verification_mode=verification_mode,
        verdicts=[True, False, True],
        make_dataset=_family_tmp_dir(family),
        schema=schema,
        source=source,
        target=target,
        ground_truth=ground_truth,
        candidates=candidates,
        prefix=f"{family}_{verification_mode}",
    )

    assert list(preds.columns) == [
        "target_id",
        "target_value",
        "source_id",
        "source_value",
        "similarity_score",
        "llm_verified",
        "dataset",
        "llm_model",
        "target_column",
        "source_column",
    ]
    # dataset carries the runner's own identifier, not the output path.
    assert preds["dataset"].tolist() == [dataset_dir.name, dataset_dir.name]
    assert result.dataset == dataset_dir.name
    # Existing fields and evaluation are unaffected.
    assert preds["source_id"].tolist() == ["s0", "s2"]
    assert preds["similarity_score"].tolist() == [0.99, 0.95]
    assert preds["target_column"].tolist() == [target_value_col] * 2
    assert preds["source_column"].tolist() == [source_value_col] * 2
    assert preds["llm_model"].tolist() == ["openai/gpt-5-mini"] * 2
    assert result.status == "success"
    assert result.accepted_pairs == 2
    assert result.true_positives == 1


def test_dataset_column_header_only_empty_output(monkeypatch):
    """Zero accepted predictions still yields the ten-column header."""
    source, target, ground_truth, candidates, schema = _family_case("ss", "value", "value")
    result, preds, preds_path, _, _ = _run_mocked_dataset(
        monkeypatch,
        family="ss",
        verification_mode="anyjoin_batch",
        verdicts=[False, False, False],
        make_dataset=_family_tmp_dir("ss_empty"),
        schema=schema,
        source=source,
        target=target,
        ground_truth=ground_truth,
        candidates=candidates,
        prefix="ss_empty",
    )

    assert len(preds) == 0
    assert list(preds.columns) == [
        "target_id",
        "target_value",
        "source_id",
        "source_value",
        "similarity_score",
        "llm_verified",
        "dataset",
        "llm_model",
        "target_column",
        "source_column",
    ]
    lines = preds_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert lines[0] == (
        "target_id,target_value,source_id,source_value,similarity_score,"
        "llm_verified,dataset,llm_model,target_column,source_column"
    )
    assert result.status == "success"
    assert result.accepted_pairs == 0


def test_dataset_column_survives_metadata_resolution_failure(monkeypatch, caplog):
    """dataset is still written when the model-metadata resolver fails."""
    def boom(model):
        raise RuntimeError("resolver unavailable")

    source, target, ground_truth, candidates, schema = _family_case("kbwt", "airport code", "city")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with caplog.at_level(logging.WARNING):
            result, preds, _, dataset_dir, _ = _run_mocked_dataset(
                monkeypatch,
                family="kbwt",
                verification_mode="sem_filter",
                verdicts=[True, True, True],
                make_dataset=_family_tmp_dir("kbwt_fail"),
                schema=schema,
                source=source,
                target=target,
                ground_truth=ground_truth,
                candidates=candidates,
                prefix="kbwt_fail",
                resolver=boom,
            )

    assert list(preds.columns) == [
        "target_id",
        "target_value",
        "source_id",
        "source_value",
        "similarity_score",
        "llm_verified",
        "dataset",
        "llm_model",
        "target_column",
        "source_column",
    ]
    assert preds["dataset"].tolist() == [dataset_dir.name] * 3
    assert preds["llm_model"].isna().all()
    assert preds["source_id"].tolist() == ["s0", "s1", "s2"]
    assert preds["similarity_score"].tolist() == [0.99, 0.97, 0.95]
    assert result.status == "success"
    assert any(
        "Could not resolve provider-request model metadata" in record.message
        for record in caplog.records
    )
