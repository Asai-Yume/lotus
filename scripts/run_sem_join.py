from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

import lotus
from lotus.cache import CacheConfig, CacheFactory, CacheType
from lotus.models import LM, LiteLLMRM
from lotus.types import CascadeArgs
from lotus.vector_store import FaissVS

# Reuse the exact dataset loading/orientation/evaluation logic from the existing
# LOTUS-Sim runner so the two baselines differ in the join operator, not in how
# benchmark data are interpreted.
from run_sem_sim_join import (  # type: ignore
    FAMILY_LAYOUTS,
    DatasetSchemaError,
    ResolvedSchema,
    discover_datasets,
    evaluate_predictions,
    find_repo_root,
    get_git_commit,
    load_dataset,
    load_openrouter_key,
    pair_set,
    write_summary,
)


DEFAULT_EMBEDDING_MODEL = "openrouter/openai/text-embedding-3-small"
DEFAULT_LM_MODEL = "openrouter/openai/gpt-5-mini"
DEFAULT_EMBEDDING_BATCH_SIZE = 64
DEFAULT_LM_BATCH_SIZE = 64
DEFAULT_RECALL_TARGET = 0.8
DEFAULT_PRECISION_TARGET = 0.8
DEFAULT_SAMPLING_PERCENTAGE = 0.1
DEFAULT_FAILURE_PROBABILITY = 0.2
DEFAULT_MIN_JOIN_CASCADE_SIZE = 100
DEFAULT_CASCADE_SEED = 0
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0
DEFAULT_CHECKPOINT_CACHE_SIZE = 250_000

MANIFEST_VERSION = 2
CACHE_SCHEMA_VERSION = 1

# Keys of run_config.json that gate --resume. Informational fields (versions,
# batch sizes, retry configuration, timestamps) are recorded but do not block a
# resume, since they never affect prompt text, cache keys, or index content.
MANIFEST_MUST_MATCH_KEYS = frozenset(
    {
        "manifest_version",
        "cache_schema_version",
        "family",
        "dataset",
        "source_rows",
        "target_rows",
        "source_fingerprint",
        "target_fingerprint",
        "source_id_col",
        "source_value_col",
        "target_id_col",
        "target_value_col",
        "join_instruction",
        "lm_model",
        "embedding_model",
        "lm_kwargs",
        "cascade_seed",
        "sampling_percentage",
        "recall_target",
        "precision_target",
        "failure_probability",
        "min_join_cascade_size",
        "lotus_commit",
        "litellm_version",
    }
)

DEFAULT_JOIN_INSTRUCTION = (
    "{source_value} and {target_value} refer to the same real-world entity."
)


class CountingLiteLLMRM(LiteLLMRM):
    """LiteLLMRM with behavior-preserving counters for embedding API batches."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.embedding_rm_invocations = 0
        self.embedding_api_requests = 0
        self.embedded_rows = 0

    def _embed(self, docs: list[str]):  # type: ignore[override]
        self.embedding_rm_invocations += 1
        self.embedded_rows += len(docs)
        if docs:
            self.embedding_api_requests += math.ceil(len(docs) / self.max_batch_size)
        return super()._embed(docs)


@dataclass
class RunResult:
    family: str
    dataset: str
    status: str
    execution_mode: str
    cascade_requested: bool
    cascade_used: bool
    embedding_model: str
    lm_model: str
    join_instruction: str
    default_on_parse_failure: bool
    source_rows: int
    target_rows: int
    cartesian_pairs: int
    prediction_rows: int
    prediction_pairs: int
    ground_truth_pairs: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    source_coverage: float
    gt_matched_source_rows: int
    gt_unmatched_source_rows: int
    unmatched_source_rows_with_prediction: int
    join_seconds: float
    total_seconds: float
    recall_target: float
    precision_target: float
    sampling_percentage: float
    failure_probability: float
    min_join_cascade_size: int
    cascade_seed: int
    chosen_cascade_plan: str | None
    learned_positive_threshold: float | None
    learned_negative_threshold: float | None
    join_resolved_by_helper_model: int | None
    join_helper_positive: int | None
    join_helper_negative: int | None
    join_resolved_by_large_model: int | None
    optimization_llm_calls: int | None
    total_logical_llm_calls: int
    llm_cache_hits: int
    llm_operator_cache_hits: int
    llm_virtual_prompt_tokens: int
    llm_virtual_completion_tokens: int
    llm_virtual_total_tokens: int
    llm_virtual_cost: float
    llm_physical_prompt_tokens: int
    llm_physical_completion_tokens: int
    llm_physical_total_tokens: int
    llm_physical_cost: float
    embedding_rm_invocations: int
    embedding_api_requests: int
    embedded_rows: int
    faiss_factory: str
    faiss_metric: str
    parse_failures: int | None
    parse_failures_available: bool
    repository_commit: str | None
    predictions_file: str
    metrics_file: str
    cascade_stats_file: str
    schema_file: str
    lotus_work_dir: str
    error: str | None = None
    # --- cross-attempt cumulative accounting (summed over checkpoint/attempt_*.json) ---
    attempts: int = 0
    resume_requested: bool = False
    cumulative_physical_requests: int = 0
    cumulative_retry_attempts: int = 0
    cumulative_retryable_failures: int = 0
    cumulative_llm_physical_prompt_tokens: int = 0
    cumulative_llm_physical_completion_tokens: int = 0
    cumulative_llm_physical_reasoning_tokens: int = 0
    cumulative_llm_physical_total_tokens: int = 0
    cumulative_llm_physical_cost: float = 0.0
    cumulative_llm_cache_hits: int = 0
    cumulative_embedding_retry_attempts: int = 0


def configure_lotus(
    embedding_model: str,
    lm_model: str,
    embedding_batch_size: int,
    lm_batch_size: int,
    *,
    checkpoint_cache_dir: Path,
    checkpoint_cache_size: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> tuple[LM, CountingLiteLLMRM, FaissVS]:
    """Configure LOTUS with a dataset-local persistent SQLite LM cache and bounded retries."""
    cache = CacheFactory.create_cache(
        CacheConfig(
            CacheType.SQLITE,
            checkpoint_cache_size,
            cache_dir=str(checkpoint_cache_dir),
        )
    )
    lm = LM(
        model=lm_model,
        max_batch_size=lm_batch_size,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        cache=cache,
    )
    rm = CountingLiteLLMRM(
        model=embedding_model,
        max_batch_size=embedding_batch_size,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    vs = FaissVS()

    # Fresh runs delete the checkpoint directory before configuration; --resume
    # intentionally preserves the SQLite cache, so do not reset the cache here.
    lm.reset_stats()

    lotus.settings.configure(
        lm=lm,
        rm=rm,
        vs=vs,
        enable_cache=True,
    )
    return lm, rm, vs


def fingerprint_values(ids: list[Any], values: list[Any]) -> str:
    """Deterministic fingerprint of join-relevant (id, value) pairs in row order.

    Missing values are canonicalized to null; strings are hashed exactly as the
    runner/model sees them (no casefold/strip normalization). Row order matters:
    it drives sampling and prompt order.
    """
    payload = json.dumps(
        [[i, None if pd.isna(v) else str(v)] for i, v in zip(ids, values)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def lm_request_kwargs(lm_model: str) -> dict[str, Any]:
    """The exact LM request kwargs the runner will use (offline, no API call).

    Constructing the LM only performs model-support detection (e.g. litellm
    ``supports_reasoning``), which determines ``max_completion_tokens`` — a
    component of every request cache hash.
    """
    probe = LM(model=lm_model)
    return dict(probe.kwargs)


def checkpoint_config(
    *,
    family: str,
    dataset: str,
    source: pd.DataFrame,
    target: pd.DataFrame,
    schema: ResolvedSchema,
    args: argparse.Namespace,
    repository_commit: str | None,
) -> dict[str, Any]:
    """Deterministic run manifest for checkpoint validation."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "family": family,
        "dataset": dataset,
        "source_rows": int(len(source)),
        "target_rows": int(len(target)),
        "source_fingerprint": fingerprint_values(
            source["source_id"].tolist(), source["source_value"].tolist()
        ),
        "target_fingerprint": fingerprint_values(
            target["target_id"].tolist(), target["target_value"].tolist()
        ),
        "source_id_col": schema.source_id_col,
        "source_value_col": schema.source_value_col,
        "target_id_col": schema.target_id_col,
        "target_value_col": schema.target_value_col,
        "join_instruction": args.join_instruction,
        "lm_model": args.lm_model,
        "embedding_model": args.embedding_model,
        "lm_kwargs": lm_request_kwargs(args.lm_model),
        "cascade_seed": args.cascade_seed,
        "sampling_percentage": args.sampling_percentage,
        "recall_target": args.recall_target,
        "precision_target": args.precision_target,
        "failure_probability": args.failure_probability,
        "min_join_cascade_size": args.min_join_cascade_size,
        "lotus_commit": repository_commit,
        "litellm_version": importlib.metadata.version("litellm"),
        # --- informational only (never gate a resume) ---
        "openai_version": importlib.metadata.version("openai"),
        "python_version": platform.python_version(),
        "llm_batch_size": args.llm_batch_size,
        "embedding_batch_size": args.embedding_batch_size,
        "retry_config": {
            "max_retries": args.max_retries,
            "retry_backoff_seconds": args.retry_backoff_seconds,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def prepare_checkpoint(
    *,
    checkpoint_dir: Path,
    lotus_work_dir: Path,
    config: dict[str, Any],
    resume: bool,
) -> Path:
    """Validate (resume) or create (fresh) the dataset checkpoint.

    On --resume the manifest's MUST-MATCH fields are compared against the current
    configuration BEFORE any model/API request; a mismatch refuses the resume.
    On a fresh run the old checkpoint and lotus_work state are removed.
    """
    manifest_file = checkpoint_dir / "run_config.json"

    if resume:
        if not manifest_file.is_file():
            raise FileNotFoundError(
                f"--resume was requested, but no checkpoint manifest exists at "
                f"{manifest_file}. This run must start fresh."
            )
        existing = json.loads(manifest_file.read_text(encoding="utf-8"))
        differing = sorted(
            key for key in MANIFEST_MUST_MATCH_KEYS if existing.get(key) != config.get(key)
        )
        if differing:
            details = "; ".join(
                f"{key}: checkpoint={existing.get(key)!r}, current={config.get(key)!r}"
                for key in differing
            )
            raise ValueError(
                "Checkpoint configuration does not match the requested run; refusing "
                f"--resume before any model request. Differences: {details}"
            )
    else:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        shutil.rmtree(lotus_work_dir, ignore_errors=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text(
            json.dumps(config, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return manifest_file


def next_attempt_id(checkpoint_dir: Path) -> int:
    """Next attempt number = max existing attempt_*.json id + 1 (fresh starts at 1)."""
    existing = [
        int(p.stem.split("_")[1])
        for p in checkpoint_dir.glob("attempt_*.json")
        if p.stem.startswith("attempt_")
    ]
    return (max(existing) if existing else 0) + 1


def attempt_snapshot(
    *,
    attempt: int,
    status: str,
    elapsed_seconds: float,
    lm: LM,
    rm: CountingLiteLLMRM,
    error: str | None = None,
) -> dict[str, Any]:
    """Immutable per-attempt accounting snapshot (physical usage of THIS attempt)."""
    usage = lm.stats
    return {
        "attempt": attempt,
        "status": status,
        "error": error,
        "elapsed_seconds": elapsed_seconds,
        "lm": {
            "physical_requests": int(usage.physical_requests),
            "retry_attempts": int(usage.retry_attempts),
            "retryable_failures": int(usage.retryable_failures),
            "fatal_failures": int(usage.fatal_failures),
            "cache_hits": int(usage.cache_hits),
            "operator_cache_hits": int(usage.operator_cache_hits),
            "physical_usage": {
                "prompt_tokens": int(usage.physical_usage.prompt_tokens),
                "completion_tokens": int(usage.physical_usage.completion_tokens),
                "reasoning_tokens": int(usage.physical_usage.reasoning_tokens),
                "total_tokens": int(usage.physical_usage.total_tokens),
                "total_cost": float(usage.physical_usage.total_cost),
            },
            "virtual_usage": {
                "prompt_tokens": int(usage.virtual_usage.prompt_tokens),
                "completion_tokens": int(usage.virtual_usage.completion_tokens),
                "reasoning_tokens": int(usage.virtual_usage.reasoning_tokens),
                "total_tokens": int(usage.virtual_usage.total_tokens),
                "total_cost": float(usage.virtual_usage.total_cost),
            },
        },
        "embedding": {
            "rm_invocations": int(rm.embedding_rm_invocations),
            "api_requests": int(rm.embedding_api_requests),
            "embedded_rows": int(rm.embedded_rows),
            "retry_attempts": int(getattr(rm, "embedding_retry_attempts", 0)),
        },
    }


def write_attempt_snapshot(checkpoint_dir: Path, snapshot: dict[str, Any]) -> Path:
    """Atomically write one immutable attempt snapshot (temp file + os.replace)."""
    path = checkpoint_dir / f"attempt_{snapshot['attempt']:04d}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_attempt_snapshots(checkpoint_dir: Path) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for path in sorted(checkpoint_dir.glob("attempt_*.json")):
        try:
            snapshots.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return snapshots


def cumulative_accounting(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    """Cumulative physical accounting across all attempt snapshots.

    Physical quantities are additive across attempts (each attempt's physical
    usage counts only that attempt's fresh provider requests). Virtual usage is
    deliberately NOT summed: cache-hit virtual usage would be double-counted.
    """
    lm = [s.get("lm", {}) for s in snapshots]
    phys = [s.get("physical_usage", {}) for s in lm]

    def total(key: str, default: Any = 0) -> Any:
        return sum((s.get(key, default) for s in lm), start=0)

    def total_phys(key: str, default: Any = 0) -> Any:
        return sum((s.get(key, default) for s in phys), start=0)

    return {
        "attempts": len(snapshots),
        "physical_requests": int(total("physical_requests")),
        "retry_attempts": int(total("retry_attempts")),
        "retryable_failures": int(total("retryable_failures")),
        "fatal_failures": int(total("fatal_failures")),
        "cache_hits": int(total("cache_hits")),
        "operator_cache_hits": int(total("operator_cache_hits")),
        "physical_prompt_tokens": int(total_phys("prompt_tokens")),
        "physical_completion_tokens": int(total_phys("completion_tokens")),
        "physical_reasoning_tokens": int(total_phys("reasoning_tokens")),
        "physical_total_tokens": int(total_phys("total_tokens")),
        "physical_cost": float(total_phys("total_cost")),
        "embedding_retry_attempts": int(
            sum(s.get("embedding", {}).get("retry_attempts", 0) for s in snapshots)
        ),
    }


def failure_state(
    *,
    family: str,
    dataset: str,
    attempt: int,
    exc: BaseException,
    elapsed_seconds: float,
    checkpoint_dir: Path,
    cumulative: dict[str, Any],
) -> dict[str, Any]:
    """Human-readable failure record; no secrets are included."""
    return {
        "family": family,
        "dataset": dataset,
        "status": "failed",
        "attempt": attempt,
        "error": f"{type(exc).__name__}: {str(exc)[:500]}",
        "elapsed_seconds_this_attempt": elapsed_seconds,
        "resume_available": True,
        "checkpoint_dir": str(checkpoint_dir),
        "cumulative_physical_accounting": cumulative,
        "note": (
            "Per-attempt details are in checkpoint/attempt_*.json. Cumulative physical "
            "usage spans all attempts; OpenRouter usage remains authoritative for billing."
        ),
    }


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    """Run LOTUS's relative helper-index writes inside a dataset-local folder."""
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump())
    if hasattr(value, "__dict__"):
        return {
            str(key): jsonable(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def empty_prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["target_id", "target_value", "source_id", "source_value"]
    )


def normalize_join_output(joined: pd.DataFrame) -> pd.DataFrame:
    required = ["target_id", "target_value", "source_id", "source_value"]
    missing = [column for column in required if column not in joined.columns]
    if missing:
        raise RuntimeError(
            "Unexpected sem_join output columns. "
            f"Missing {missing}; available columns: {joined.columns.tolist()}"
        )

    predictions = joined[required].copy()
    for column in required:
        predictions[column] = predictions[column].astype("string")
    return predictions


def cascade_fields(stats: dict[str, Any] | None) -> dict[str, Any]:
    if not stats:
        return {
            "chosen_cascade_plan": None,
            "learned_positive_threshold": None,
            "learned_negative_threshold": None,
            "join_resolved_by_helper_model": None,
            "join_helper_positive": None,
            "join_helper_negative": None,
            "join_resolved_by_large_model": None,
            "optimization_llm_calls": None,
        }

    cascade_args = stats.get("cascade_args")
    return {
        "chosen_cascade_plan": getattr(cascade_args, "join_cascade_strategy", None),
        "learned_positive_threshold": getattr(
            cascade_args, "join_cascade_pos_threshold", None
        ),
        "learned_negative_threshold": getattr(
            cascade_args, "join_cascade_neg_threshold", None
        ),
        "join_resolved_by_helper_model": int(
            stats.get("join_resolved_by_helper_model", 0)
        ),
        "join_helper_positive": int(stats.get("join_helper_positive", 0)),
        "join_helper_negative": int(stats.get("join_helper_negative", 0)),
        "join_resolved_by_large_model": int(
            stats.get("join_resolved_by_large_model", 0)
        ),
        "optimization_llm_calls": int(stats.get("optimized_join_cost", 0)),
    }


def run_one_dataset(
    *,
    family: str,
    dataset_dir: Path,
    args: argparse.Namespace,
    repository_commit: str | None,
) -> RunResult:
    dataset = dataset_dir.name
    output_dir = args.output_root / family / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_file = output_dir / "predictions.csv"
    metrics_file = output_dir / "metrics.json"
    cascade_stats_file = output_dir / "cascade_stats.json"
    schema_file = output_dir / "resolved_schema.json"
    lotus_work_dir = output_dir / "lotus_work"
    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_cache_dir = checkpoint_dir / "lm_cache"
    failure_state_file = output_dir / "failure_state.json"

    source, target, ground_truth, schema = load_dataset(dataset_dir, family, args)
    schema_file.write_text(
        json.dumps(asdict(schema), indent=2),
        encoding="utf-8",
    )

    cartesian_pairs = len(source) * len(target)
    would_use_cascade = cartesian_pairs >= args.min_join_cascade_size

    print(f"\n{'=' * 72}")
    print(f"Family:              {family}")
    print(f"Dataset:             {dataset}")
    print(f"Source/query rows:   {len(source)}")
    print(f"Target/reference:    {len(target)}")
    print(f"Cartesian pairs:     {cartesian_pairs}")
    print(f"Ground-truth pairs:  {len(pair_set(ground_truth, family))}")
    print(
        "Evaluation pairs:    "
        + (
            "(target_value, source_value)"
            if family == "kbwt"
            else "(target_id, source_id)"
        )
    )
    print(f"Would use cascade:   {would_use_cascade}")
    print(f"Cascade seed:        {args.cascade_seed}")
    print(f"Instruction:         {args.join_instruction}")
    print(f"{'=' * 72}")

    if args.dry_run:
        predictions = empty_prediction_frame()
        metrics = evaluate_predictions(predictions, ground_truth, source, family)
        execution_mode = (
            "cascade_would_run" if would_use_cascade else "full_join_fallback_would_run"
        )
        return RunResult(
            family=family,
            dataset=dataset,
            status="dry_run",
            execution_mode=execution_mode,
            cascade_requested=True,
            cascade_used=False,
            embedding_model=args.embedding_model,
            lm_model=args.lm_model,
            join_instruction=args.join_instruction,
            default_on_parse_failure=False,
            source_rows=len(source),
            target_rows=len(target),
            cartesian_pairs=cartesian_pairs,
            prediction_rows=0,
            prediction_pairs=0,
            ground_truth_pairs=int(metrics["ground_truth_pairs"]),
            true_positives=0,
            false_positives=0,
            false_negatives=int(metrics["false_negatives"]),
            precision=0.0,
            recall=0.0,
            f1=0.0,
            source_coverage=0.0,
            gt_matched_source_rows=int(metrics["gt_matched_source_rows"]),
            gt_unmatched_source_rows=int(metrics["gt_unmatched_source_rows"]),
            unmatched_source_rows_with_prediction=0,
            join_seconds=0.0,
            total_seconds=0.0,
            recall_target=args.recall_target,
            precision_target=args.precision_target,
            sampling_percentage=args.sampling_percentage,
            failure_probability=args.failure_probability,
            min_join_cascade_size=args.min_join_cascade_size,
            cascade_seed=args.cascade_seed,
            chosen_cascade_plan=None,
            learned_positive_threshold=None,
            learned_negative_threshold=None,
            join_resolved_by_helper_model=None,
            join_helper_positive=None,
            join_helper_negative=None,
            join_resolved_by_large_model=None,
            optimization_llm_calls=None,
            total_logical_llm_calls=0,
            llm_cache_hits=0,
            llm_operator_cache_hits=0,
            llm_virtual_prompt_tokens=0,
            llm_virtual_completion_tokens=0,
            llm_virtual_total_tokens=0,
            llm_virtual_cost=0.0,
            llm_physical_prompt_tokens=0,
            llm_physical_completion_tokens=0,
            llm_physical_total_tokens=0,
            llm_physical_cost=0.0,
            embedding_rm_invocations=0,
            embedding_api_requests=0,
            embedded_rows=0,
            faiss_factory="Flat",
            faiss_metric="METRIC_INNER_PRODUCT",
            parse_failures=None,
            parse_failures_available=False,
            repository_commit=repository_commit,
            predictions_file=str(predictions_file),
            metrics_file=str(metrics_file),
            cascade_stats_file=str(cascade_stats_file),
            schema_file=str(schema_file),
            lotus_work_dir=str(lotus_work_dir),
        )

    config = checkpoint_config(
        family=family,
        dataset=dataset,
        source=source,
        target=target,
        schema=schema,
        args=args,
        repository_commit=repository_commit,
    )
    prepare_checkpoint(
        checkpoint_dir=checkpoint_dir,
        lotus_work_dir=lotus_work_dir,
        config=config,
        resume=args.resume,
    )
    if not args.resume:
        failure_state_file.unlink(missing_ok=True)

    attempt_id = next_attempt_id(checkpoint_dir)

    lm, rm, _vs = configure_lotus(
        args.embedding_model,
        args.lm_model,
        args.embedding_batch_size,
        args.llm_batch_size,
        checkpoint_cache_dir=checkpoint_cache_dir,
        checkpoint_cache_size=args.checkpoint_cache_size,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
    )

    cascade_args = CascadeArgs(
        recall_target=args.recall_target,
        precision_target=args.precision_target,
        sampling_percentage=args.sampling_percentage,
        failure_probability=args.failure_probability,
        min_join_cascade_size=args.min_join_cascade_size,
        cascade_IS_random_seed=args.cascade_seed,
    )

    join_start = time.perf_counter()
    try:
        with working_directory(lotus_work_dir):
            sem_join_result = source.sem_join(
                target,
                args.join_instruction,
                default=False,
                cascade_args=cascade_args,
                return_stats=True,
            )
    except BaseException as exc:
        elapsed = time.perf_counter() - join_start
        snapshot = attempt_snapshot(
            attempt=attempt_id,
            status="failure",
            elapsed_seconds=elapsed,
            lm=lm,
            rm=rm,
            error=f"{type(exc).__name__}: {str(exc)[:500]}",
        )
        write_attempt_snapshot(checkpoint_dir, snapshot)
        cumulative = cumulative_accounting(load_attempt_snapshots(checkpoint_dir))
        failure_state_file.write_text(
            json.dumps(
                failure_state(
                    family=family,
                    dataset=dataset,
                    attempt=attempt_id,
                    exc=exc,
                    elapsed_seconds=elapsed,
                    checkpoint_dir=checkpoint_dir,
                    cumulative=cumulative,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"Failure state/checkpoint: {failure_state_file}",
            file=sys.stderr,
        )
        raise
    join_seconds = time.perf_counter() - join_start

    if isinstance(sem_join_result, tuple):
        joined, native_stats = sem_join_result
    else:
        joined = sem_join_result
        native_stats = None

    cascade_used = native_stats is not None
    if would_use_cascade and not cascade_used:
        raise RuntimeError(
            "sem_join was expected to use the cascade because the Cartesian "
            f"size is {cartesian_pairs} >= {args.min_join_cascade_size}, but "
            "return_stats=True did not return cascade statistics."
        )
    if (not would_use_cascade) and cascade_used:
        raise RuntimeError(
            "sem_join unexpectedly returned cascade statistics even though the "
            f"Cartesian size is {cartesian_pairs} < {args.min_join_cascade_size}."
        )

    predictions = normalize_join_output(joined)
    predictions.to_csv(predictions_file, index=False)
    metrics = evaluate_predictions(predictions, ground_truth, source, family)

    if native_stats is not None:
        cascade_stats_file.write_text(
            json.dumps(jsonable(native_stats), indent=2),
            encoding="utf-8",
        )
        total_logical_llm_calls = int(native_stats.get("total_LM_calls", 0))
    else:
        cascade_stats_file.write_text(
            json.dumps(
                {
                    "cascade_used": False,
                    "reason": (
                        "Cartesian product is smaller than min_join_cascade_size; "
                        "LOTUS used exhaustive sem_join()."
                    ),
                    "cartesian_pairs": cartesian_pairs,
                    "min_join_cascade_size": args.min_join_cascade_size,
                    "total_LM_calls": cartesian_pairs,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        total_logical_llm_calls = cartesian_pairs

    cf = cascade_fields(native_stats)
    usage = lm.stats
    total_seconds = join_seconds

    # Immutable per-attempt snapshot + cumulative physical accounting across
    # attempts (a --resume run keeps prior attempts' physical usage).
    snapshot = attempt_snapshot(
        attempt=attempt_id,
        status="success",
        elapsed_seconds=total_seconds,
        lm=lm,
        rm=rm,
    )
    write_attempt_snapshot(checkpoint_dir, snapshot)
    cumulative = cumulative_accounting(load_attempt_snapshots(checkpoint_dir))
    failure_state_file.unlink(missing_ok=True)

    result = RunResult(
        family=family,
        dataset=dataset,
        status="success",
        execution_mode="cascade" if cascade_used else "full_join_fallback",
        cascade_requested=True,
        cascade_used=cascade_used,
        embedding_model=args.embedding_model,
        lm_model=args.lm_model,
        join_instruction=args.join_instruction,
        default_on_parse_failure=False,
        source_rows=len(source),
        target_rows=len(target),
        cartesian_pairs=cartesian_pairs,
        prediction_rows=len(predictions),
        prediction_pairs=int(metrics["prediction_pairs"]),
        ground_truth_pairs=int(metrics["ground_truth_pairs"]),
        true_positives=int(metrics["true_positives"]),
        false_positives=int(metrics["false_positives"]),
        false_negatives=int(metrics["false_negatives"]),
        precision=float(metrics["precision"]),
        recall=float(metrics["recall"]),
        f1=float(metrics["f1"]),
        source_coverage=float(metrics["source_coverage"]),
        gt_matched_source_rows=int(metrics["gt_matched_source_rows"]),
        gt_unmatched_source_rows=int(metrics["gt_unmatched_source_rows"]),
        unmatched_source_rows_with_prediction=int(
            metrics["unmatched_source_rows_with_prediction"]
        ),
        join_seconds=join_seconds,
        total_seconds=total_seconds,
        recall_target=args.recall_target,
        precision_target=args.precision_target,
        sampling_percentage=args.sampling_percentage,
        failure_probability=args.failure_probability,
        min_join_cascade_size=args.min_join_cascade_size,
        cascade_seed=args.cascade_seed,
        chosen_cascade_plan=cf["chosen_cascade_plan"],
        learned_positive_threshold=cf["learned_positive_threshold"],
        learned_negative_threshold=cf["learned_negative_threshold"],
        join_resolved_by_helper_model=cf["join_resolved_by_helper_model"],
        join_helper_positive=cf["join_helper_positive"],
        join_helper_negative=cf["join_helper_negative"],
        join_resolved_by_large_model=cf["join_resolved_by_large_model"],
        optimization_llm_calls=cf["optimization_llm_calls"],
        total_logical_llm_calls=total_logical_llm_calls,
        llm_cache_hits=int(usage.cache_hits),
        llm_operator_cache_hits=int(usage.operator_cache_hits),
        llm_virtual_prompt_tokens=int(usage.virtual_usage.prompt_tokens),
        llm_virtual_completion_tokens=int(usage.virtual_usage.completion_tokens),
        llm_virtual_total_tokens=int(usage.virtual_usage.total_tokens),
        llm_virtual_cost=float(usage.virtual_usage.total_cost),
        llm_physical_prompt_tokens=int(usage.physical_usage.prompt_tokens),
        llm_physical_completion_tokens=int(usage.physical_usage.completion_tokens),
        llm_physical_total_tokens=int(usage.physical_usage.total_tokens),
        llm_physical_cost=float(usage.physical_usage.total_cost),
        embedding_rm_invocations=int(rm.embedding_rm_invocations),
        embedding_api_requests=int(rm.embedding_api_requests),
        embedded_rows=int(rm.embedded_rows),
        faiss_factory="Flat",
        faiss_metric="METRIC_INNER_PRODUCT",
        # The current cascade implementation does not expose the raw/final
        # sem_filter outputs through SemanticJoinOutput, so parse failures cannot
        # be counted accurately without changing LOTUS library code.
        parse_failures=None,
        parse_failures_available=False,
        repository_commit=repository_commit,
        predictions_file=str(predictions_file),
        metrics_file=str(metrics_file),
        cascade_stats_file=str(cascade_stats_file),
        schema_file=str(schema_file),
        lotus_work_dir=str(lotus_work_dir),
        attempts=int(cumulative["attempts"]),
        resume_requested=bool(args.resume),
        cumulative_physical_requests=int(cumulative["physical_requests"]),
        cumulative_retry_attempts=int(cumulative["retry_attempts"]),
        cumulative_retryable_failures=int(cumulative["retryable_failures"]),
        cumulative_llm_physical_prompt_tokens=int(cumulative["physical_prompt_tokens"]),
        cumulative_llm_physical_completion_tokens=int(cumulative["physical_completion_tokens"]),
        cumulative_llm_physical_reasoning_tokens=int(cumulative["physical_reasoning_tokens"]),
        cumulative_llm_physical_total_tokens=int(cumulative["physical_total_tokens"]),
        cumulative_llm_physical_cost=float(cumulative["physical_cost"]),
        cumulative_llm_cache_hits=int(cumulative["cache_hits"]),
        cumulative_embedding_retry_attempts=int(cumulative["embedding_retry_attempts"]),
    )

    metrics_file.write_text(
        json.dumps(asdict(result), indent=2),
        encoding="utf-8",
    )

    print(f"Execution mode:      {result.execution_mode}")
    print(f"Prediction rows:     {result.prediction_rows}")
    print(f"Evaluated pairs:     {result.prediction_pairs}")
    print(f"Precision:           {result.precision:.6f}")
    print(f"Recall:              {result.recall:.6f}")
    print(f"F1:                  {result.f1:.6f}")
    print(f"Source coverage:     {result.source_coverage:.6f}")
    print(f"Join time:           {result.join_seconds:.4f} s")
    print(f"Logical LLM calls:   {result.total_logical_llm_calls}")
    print(f"LLM cache hits:      {result.llm_cache_hits}")
    print(f"Physical LLM tokens: {result.llm_physical_total_tokens}")
    print(f"Physical LLM cost:   ${result.llm_physical_cost:.6f}")
    print(f"Embedding requests:  {result.embedding_api_requests}")
    print(f"Embedded rows:       {result.embedded_rows}")
    if result.attempts > 1:
        print(f"Attempts:            {result.attempts} (resumed)")
    if result.cumulative_physical_requests > 0:
        print(f"Cumulative phys req: {result.cumulative_physical_requests}")
        print(f"Cumulative retries:  {result.cumulative_retry_attempts}")
        print(f"Cumulative phys tok: {result.cumulative_llm_physical_total_tokens}")
        print(f"Cumulative phys cost:${result.cumulative_llm_physical_cost:.6f}")
    if result.cascade_used:
        print(f"Cascade plan:        {result.chosen_cascade_plan}")
        print(f"Positive threshold:  {result.learned_positive_threshold}")
        print(f"Negative threshold:  {result.learned_negative_threshold}")
        print(f"Optimization calls:  {result.optimization_llm_calls}")
        print(f"Helper resolved:     {result.join_resolved_by_helper_model}")
        print(f"Large-model resolved:{result.join_resolved_by_large_model}")
    print(f"Predictions:         {predictions_file}")
    print(f"Metrics:             {metrics_file}")

    return result


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run native LOTUS sem_join with its join cascade on VIDA AutoFJ, "
            "AutoFJ-overlap, or KBWT datasets."
        )
    )
    parser.add_argument(
        "--family",
        choices=sorted(FAMILY_LAYOUTS),
        required=True,
    )

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--dataset",
        action="append",
        help="Dataset directory name. Repeat to run multiple datasets.",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Run every dataset directory in the selected family.",
    )

    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--lm-model", default=DEFAULT_LM_MODEL)
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
    )
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=DEFAULT_LM_BATCH_SIZE,
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=(
            "Retry count for transient provider failures and empty/partial chat "
            "responses (after the first physical attempt). Defaults to 5."
        ),
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_SECONDS,
        help="Initial exponential-backoff delay between retries (capped at 30s).",
    )
    parser.add_argument(
        "--checkpoint-cache-size",
        type=int,
        default=DEFAULT_CHECKPOINT_CACHE_SIZE,
        help="Maximum entries in the per-dataset persistent SQLite LM cache.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse the dataset checkpoint (SQLite request cache, lotus_work index) "
            "from a failed earlier attempt. All MUST-MATCH manifest settings must "
            "equal the stored run_config.json or the resume is refused before any "
            "model request."
        ),
    )
    parser.add_argument("--recall-target", type=float, default=DEFAULT_RECALL_TARGET)
    parser.add_argument(
        "--precision-target", type=float, default=DEFAULT_PRECISION_TARGET
    )
    parser.add_argument(
        "--sampling-percentage", type=float, default=DEFAULT_SAMPLING_PERCENTAGE
    )
    parser.add_argument(
        "--failure-probability", type=float, default=DEFAULT_FAILURE_PROBABILITY
    )
    parser.add_argument(
        "--min-join-cascade-size",
        type=int,
        default=DEFAULT_MIN_JOIN_CASCADE_SIZE,
        help=(
            "Native LOTUS threshold for using the join cascade. Below this "
            "Cartesian size, sem_join performs exhaustive LLM matching."
        ),
    )
    parser.add_argument(
        "--cascade-seed",
        type=int,
        default=DEFAULT_CASCADE_SEED,
        help="Fixed importance-sampling seed for cascade threshold learning.",
    )
    parser.add_argument(
        "--join-instruction",
        default=DEFAULT_JOIN_INSTRUCTION,
        help=(
            "Natural-language entity-equivalence predicate. The normalized source "
            "and target column names are distinct, so explicit :left/:right suffixes "
            "are intentionally avoided for Windows-safe internal index paths."
        ),
    )
    parser.add_argument(
        "--key-file",
        type=Path,
        default=None,
        help=(
            "OpenRouter key file. Defaults to <repository-root>/openrouter.key. "
            "Ignored when OPENROUTER_API_KEY is already set."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "outputs" / "lotus_sem_join",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve files/columns and validate data without calling models.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue a family batch after a dataset fails.",
    )

    # Keep the same schema overrides as LOTUS-Sim.
    parser.add_argument("--source-id-col", default=None)
    parser.add_argument("--source-value-col", default=None)
    parser.add_argument("--target-id-col", default=None)
    parser.add_argument("--target-value-col", default=None)
    parser.add_argument("--gt-source-col", default=None)
    parser.add_argument("--gt-target-col", default=None)

    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.embedding_batch_size < 1:
        parser.error("--embedding-batch-size must be at least 1.")
    if args.llm_batch_size < 1:
        parser.error("--llm-batch-size must be at least 1.")
    if args.max_retries < 0:
        parser.error("--max-retries must be at least 0.")
    if args.retry_backoff_seconds < 0:
        parser.error("--retry-backoff-seconds must be at least 0.")
    if args.checkpoint_cache_size < 1:
        parser.error("--checkpoint-cache-size must be at least 1.")
    if not 0 < args.recall_target <= 1:
        parser.error("--recall-target must be in (0, 1].")
    if not 0 < args.precision_target <= 1:
        parser.error("--precision-target must be in (0, 1].")
    if not 0 < args.sampling_percentage <= 1:
        parser.error("--sampling-percentage must be in (0, 1].")
    if not 0 < args.failure_probability < 1:
        parser.error("--failure-probability must be in (0, 1).")
    if args.min_join_cascade_size < 1:
        parser.error("--min-join-cascade-size must be at least 1.")
    if "{source_value}" not in args.join_instruction:
        parser.error(
            "--join-instruction must reference {source_value} so LOTUS can "
            "resolve the source column unambiguously."
        )
    if "{target_value}" not in args.join_instruction:
        parser.error(
            "--join-instruction must reference {target_value} so LOTUS can "
            "resolve the target column unambiguously."
        )


def main() -> int:
    repo_root = find_repo_root(Path(__file__).parent)
    parser = build_parser(repo_root)
    args = parser.parse_args()
    validate_args(parser, args)

    args.output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else (repo_root / args.output_root)
    ).resolve()

    data_root = repo_root / "data" / args.family
    datasets = discover_datasets(data_root, args.dataset, args.all)
    repository_commit = get_git_commit(repo_root)

    if not args.dry_run:
        load_openrouter_key(repo_root, args.key_file)

    results: list[RunResult] = []
    failures: list[dict[str, str]] = []

    for dataset_dir in datasets:
        try:
            result = run_one_dataset(
                family=args.family,
                dataset_dir=dataset_dir,
                args=args,
                repository_commit=repository_commit,
            )
            results.append(result)
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            print(
                f"\nERROR [{args.family}/{dataset_dir.name}]: {error_message}",
                file=sys.stderr,
            )
            traceback.print_exc()
            failures.append(
                {
                    "family": args.family,
                    "dataset": dataset_dir.name,
                    "error": error_message,
                }
            )
            if not args.continue_on_error:
                break

    summary_name = (
        f"{args.family}_dry_run_summary.csv"
        if args.dry_run
        else f"{args.family}_summary.csv"
    )
    summary_path = args.output_root / summary_name
    if results:
        write_summary(results, summary_path)
        print(f"\nSummary: {summary_path}")

    if failures:
        failures_path = args.output_root / f"{args.family}_errors.csv"
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(failures_path, index=False)
        print(f"Errors:  {failures_path}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
