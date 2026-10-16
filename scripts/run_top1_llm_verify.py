from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import textwrap
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # Any-Join's response parser; required only by the batched verifier path.
    import json_repair
except ImportError:  # pragma: no cover - dependency guard for the historical path
    json_repair = None  # type: ignore[assignment]

import lotus
from litellm import get_llm_provider
from lotus.models import LM, LiteLLMRM, SentenceTransformersRM
from lotus.sem_ops.sem_filter import sem_filter
from lotus.templates import task_instructions
from lotus.vector_store import FaissVS

# Reuse the benchmark's dataset loading / schema resolution / evaluation logic.
from run_sem_sim_join import (  # type: ignore
    FAMILY_LAYOUTS,
    DatasetSchemaError,
    ResolvedSchema,
    discover_datasets,
    evaluate_predictions,
    evaluation_pair_columns,
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
DEFAULT_LLM_BATCH_SIZE = 8
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0

# Corrected fair baseline: BGE retrieval + Any-Join-style batched verification.
DEFAULT_BGE_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_VERIFICATION_BATCH_SIZE = 100
DEFAULT_VERIFICATION_MODE = "sem_filter"

DEFAULT_JOIN_INSTRUCTION = (
    "{source_value} and {target_value} refer to the same real-world entity."
)

# Asymmetric (query, passage) embedding prefixes, mirroring Any-Join's
# ``utils.EMBEDDING_PREFIXES`` registry. Any-Join's ``Pipeline`` auto-resolves
# these from the embedding-model name (src/pipeline.py) and passes them into the
# blocking stage, whose semantic matcher prepends the query prefix to the left
# (query) side and the passage prefix to the right (passage) side before
# encoding (src/blocking.py). That blocking stage is what generates the
# benchmark's retrieved candidates, so the corrected LOTUS baseline reproduces
# the same asymmetric encoding for BGE.
ANYJOIN_EMBEDDING_PREFIXES: dict[str, tuple[str, str]] = {
    "intfloat/e5-base-v2": ("query: ", "passage: "),
    "intfloat/e5-large-v2": ("query: ", "passage: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
    "BAAI/bge-base-en-v1.5": ("Represent this sentence: ", ""),
    "BAAI/bge-large-en-v1.5": ("Represent this sentence: ", ""),
}


class CountingSentenceTransformersRM(SentenceTransformersRM):
    """Local SentenceTransformers RM with the same counters as CountingLiteLLMRM.

    Used for the corrected baseline's ``BAAI/bge-base-en-v1.5`` retrieval model.
    Embeddings are produced locally, so ``embedding_api_requests`` stays 0 (no
    paid embedding API is called) and local encode batches are counted
    separately in ``embedding_batches``.

    Asymmetric prefixes mirror Any-Join: LOTUS indexes the target table through
    ``RM.__call__`` (the passage side) and embeds source queries through
    ``RM.convert_query_to_query_vector`` (the query side).
    """

    def __init__(
        self,
        *args: Any,
        query_prefix: str | None = None,
        passage_prefix: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.embedding_rm_invocations = 0
        self.embedding_api_requests = 0
        self.embedded_rows = 0
        self.embedding_batches = 0

    def __call__(self, docs: list[str]):  # type: ignore[override]
        """Index/passage side: apply the passage prefix (empty for BGE)."""
        if self.passage_prefix:
            docs = [self.passage_prefix + str(doc) for doc in docs]
        return super().__call__(docs)

    def convert_query_to_query_vector(self, queries):  # type: ignore[override]
        """Query side: apply Any-Join's query prefix before encoding.

        Pre-computed vectors (numpy arrays) are passed through untouched, as in
        ``RM.convert_query_to_query_vector``.
        """
        if self.query_prefix and not isinstance(queries, np.ndarray):
            if isinstance(queries, pd.Series):
                queries = queries.tolist()
            elif isinstance(queries, str):
                queries = [queries]
            queries = [self.query_prefix + str(query) for query in queries]
        return super().convert_query_to_query_vector(queries)

    def _embed(self, docs: list[str]):  # type: ignore[override]
        self.embedding_rm_invocations += 1
        self.embedded_rows += len(docs)
        if docs:
            self.embedding_batches += math.ceil(len(docs) / self.max_batch_size)
        return super()._embed(docs)


class CountingLiteLLMRM(LiteLLMRM):
    """LiteLLMRM with live embedding request/row counters (mirrors the
    run_sem_join.py pattern so no embedding accounting is estimated)."""

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
class Top1RunResult:
    family: str
    dataset: str
    status: str
    embedding_model: str
    embedding_backend: str
    embedding_query_prefix: str
    lm_model: str
    lm_temperature: float
    lm_completion_budget_omitted: bool
    verification_mode: str
    verification_batch_size: int
    join_instruction: str
    default_on_parse_failure: bool
    # --- dataset scale ---
    source_rows: int
    target_rows: int
    cartesian_pairs: int
    ground_truth_pairs: int
    gt_matched_source_rows: int
    gt_unmatched_source_rows: int
    # --- retrieval ---
    retrieved_candidates: int
    expected_top1_candidates: int
    top1_retrieval_hits: int
    top1_retrieval_recall: float
    duplicate_target_value_rows: int
    duplicate_target_values: int
    # --- verification (pairs vs logical requests) ---
    logical_verification_calls: int
    expected_logical_verification_calls: int
    verification_batches: int
    verification_pairs_verified: int
    verification_pairs_unresolved: int
    verification_malformed_responses: int
    verification_missing_ids: int
    verification_out_of_range_ids: int
    verification_duplicate_ids: int
    accepted_pairs: int
    rejected_candidates: int
    retrieval_failures: int
    verification_fn_given_retrieval_hit: int
    verification_false_accepts: int
    verification_recall_given_retrieval_hit: float
    # --- final quality ---
    precision: float
    recall: float
    f1: float
    source_coverage: float
    true_positives: int
    false_positives: int
    false_negatives: int
    # --- runtime ---
    indexing_seconds: float
    retrieval_seconds: float
    verification_seconds: float
    evaluation_seconds: float
    total_seconds: float
    # --- embedding accounting ---
    embedding_rm_invocations: int
    embedding_api_requests: int
    embedded_rows: int
    embedding_batches: int
    # --- LLM accounting (LMStats) ---
    llm_physical_requests: int
    llm_retry_attempts: int
    llm_retryable_failures: int
    llm_fatal_failures: int
    llm_cache_hits: int
    llm_physical_prompt_tokens: int
    llm_physical_completion_tokens: int
    llm_physical_reasoning_tokens: int
    llm_physical_total_tokens: int
    llm_physical_cost: float
    # --- provenance ---
    repository_commit: str | None
    predictions_file: str
    candidates_file: str
    metrics_file: str
    schema_file: str
    error: str | None = None


def _embedding_backend(args: argparse.Namespace) -> str:
    """Embedding backend, defaulting to the historical hosted (litellm) path."""
    return getattr(args, "embedding_backend", "litellm")


def _verification_mode(args: argparse.Namespace) -> str:
    """Verification procedure, defaulting to the historical sem_filter path."""
    return getattr(args, "verification_mode", DEFAULT_VERIFICATION_MODE)


def _verification_batch_size(args: argparse.Namespace) -> int:
    """Candidate pairs per logical verification request (default 100)."""
    return max(1, int(getattr(args, "verification_batch_size", DEFAULT_VERIFICATION_BATCH_SIZE)))


def _embedding_query_prefix(args: argparse.Namespace) -> str:
    """Resolve the Any-Join query prefix actually used for retrieval.

    Only the local SentenceTransformers backend applies an asymmetric prefix;
    the historical hosted backend and unregistered local models use no prefix.
    """
    if _embedding_backend(args) != "sentence_transformers":
        return ""
    query_prefix, _passage_prefix = ANYJOIN_EMBEDDING_PREFIXES.get(
        args.embedding_model, (None, None)
    )
    return query_prefix or ""


def _omit_lm_completion_budget(args: argparse.Namespace) -> bool:
    """Whether the LM must omit the completion-budget parameter entirely.

    Off by default so every historical path keeps sending LOTUS's normal
    completion budget. The corrected Any-Join-aligned launchers enable it so the
    request shape matches Any-Join's ``batch_llm_verify`` call, which sends no
    max_tokens/max_completion_tokens/max_output_tokens.
    """
    return bool(getattr(args, "omit_lm_completion_budget", False))


def _effective_provider_model(model: str) -> str:
    """Provider-request model identifier for ``LM.model``, or "" if unavailable.

    LiteLLM receives ``LM.model`` verbatim (e.g. "openrouter/openai/gpt-5-mini")
    and normalizes it at completion time via ``get_llm_provider``, which strips
    the LiteLLM routing prefix before the request is built.

    Verified scope: for the installed LiteLLM version this normalization was
    traced through to the OpenRouter request payload, where the identifier
    appears as "openai/gpt-5-mini". This is *not* a universal claim — for
    providers whose handlers rewrite the model further (AWS Bedrock, Vertex AI,
    ...) the payload identifier can differ from the value returned here.

    Metadata only: this never changes request behavior. On resolution failure it
    returns "" (metadata unavailable) and reports the problem through
    ``lotus.logger`` rather than a Python warning, so a caller running with
    warnings-as-errors cannot turn missing metadata into a failed run. The
    column therefore has exactly one meaning; an empty value means "not
    available", not "no LLM participated".
    """
    if not model:
        return ""
    try:
        return get_llm_provider(model)[0]
    except Exception:
        # Deliberately no exception detail: provider resolvers can embed
        # environment configuration in their messages.
        lotus.logger.warning(
            "Could not resolve provider-request model metadata; "
            "writing an empty llm_model value."
        )
        return ""


def configure_models(args: argparse.Namespace):
    """Configure LOTUS models with model-layer retries and an in-memory cache.

    Two backward-compatible embedding backends are supported:

    * ``litellm`` (historical default) — ``CountingLiteLLMRM`` for hosted
      embedding models such as ``openrouter/openai/text-embedding-3-small``.
    * ``sentence_transformers`` — ``CountingSentenceTransformersRM`` for local
      HuggingFace models such as ``BAAI/bge-base-en-v1.5``, matching Any-Join's
      ``SentenceTransformer(..., normalize_embeddings=True)`` usage.

    ``temperature`` is always supplied explicitly (0.0) so the verifier never
    relies on a provider default.
    """
    lm_kwargs: dict[str, Any] = {
        "model": args.lm_model,
        "max_batch_size": args.llm_batch_size,
        "max_retries": args.max_retries,
        "retry_backoff_seconds": args.retry_backoff_seconds,
        "temperature": 0.0,
        # When the Any-Join-aligned flag is set, the LM must not transmit any
        # completion-budget parameter, matching Any-Join's request shape.
        "send_max_completion_tokens": not _omit_lm_completion_budget(args),
    }
    if getattr(args, "lm_max_tokens", None) is not None:
        lm_kwargs["max_tokens"] = args.lm_max_tokens
    lm = LM(**lm_kwargs)

    if _embedding_backend(args) == "sentence_transformers":
        # Mirror Any-Join's Pipeline: auto-resolve query/passage prefixes from
        # the embedding-model name (utils.EMBEDDING_PREFIXES). For
        # BAAI/bge-base-en-v1.5 that is ("Represent this sentence: ", "").
        query_prefix, passage_prefix = ANYJOIN_EMBEDDING_PREFIXES.get(
            args.embedding_model, (None, None)
        )
        rm: Any = CountingSentenceTransformersRM(
            model=args.embedding_model,
            max_batch_size=args.embedding_batch_size,
            normalize_embeddings=True,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
        )
    else:
        rm = CountingLiteLLMRM(
            model=args.embedding_model,
            max_batch_size=args.embedding_batch_size,
            max_retries=args.max_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
        )
    vs = FaissVS()

    if lm.cache is not None:
        lm.cache.reset()

    lotus.settings.configure(
        lm=lm,
        rm=rm,
        vs=vs,
        enable_cache=True,
    )
    return lm, rm, vs


def _assemble_candidates(joined: pd.DataFrame, source_len: int) -> pd.DataFrame:
    """Canonical candidate frame from a K=1 sem_sim_join result."""
    candidates = joined[
        ["target_id", "target_value", "source_id", "source_value", "_scores"]
    ].rename(columns={"_scores": "similarity_score"})
    for column in ["target_id", "target_value", "source_id", "source_value"]:
        candidates[column] = candidates[column].astype("string")
    if len(candidates) != source_len:
        raise RuntimeError(
            f"Unexpected sem_sim_join(K=1) output size: expected {source_len} rows "
            f"for {source_len} source rows, got {len(candidates)}."
        )
    return candidates


def retrieve_top1(
    source: pd.DataFrame,
    target: pd.DataFrame,
    index_dir: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """Index the target table and retrieve exactly one candidate per source row."""
    index_start = time.perf_counter()
    indexed_target = target.sem_index("target_value", str(index_dir))
    indexing_seconds = time.perf_counter() - index_start

    retrieval_start = time.perf_counter()
    joined = source.sem_sim_join(
        indexed_target,
        left_on="source_value",
        right_on="target_value",
        K=1,
    )
    retrieval_seconds = time.perf_counter() - retrieval_start

    candidates = _assemble_candidates(joined, len(source))
    return indexed_target, candidates, indexing_seconds, retrieval_seconds


def verify_candidates(
    candidates: pd.DataFrame,
    model: LM,
    args: argparse.Namespace,
) -> list[bool]:
    """Binary LLM verification of every retrieved candidate via sem_filter.

    One logical verification per candidate; outputs are aligned 1:1 with the
    candidate rows. Parse failures resolve to the default (False -> reject).
    """
    docs = task_instructions.df2multimodal_info(
        candidates[["source_value", "target_value"]],
        ["source_value", "target_value"],
    )
    output = sem_filter(
        docs,
        model,
        args.join_instruction,
        default=False,
        show_progress_bar=True,
        progress_bar_desc="Verifying top-1 candidates",
    )
    if len(output.outputs) != len(candidates):
        raise RuntimeError(
            "sem_filter returned a different number of outputs "
            f"({len(output.outputs)}) than candidates ({len(candidates)})."
        )
    return list(output.outputs)


def anyjoin_batch_values_prompt(
    pairs: list[tuple[str, str]],
    mode: str = "equivalence",
    include_explanation: bool = False,
) -> tuple[str, str]:
    """Reproduce Any-Join's ``prompts.batch_values_prompt`` semantics exactly.

    The corrected baseline verifies up to ``verification_batch_size`` Top-1
    candidate pairs in ONE logical LLM request, using the same system message,
    numbered pair listing, Yes/No criteria, and output schema as Any-Join's
    ``batch_values_prompt`` (mode='equivalence', include_explanation=False).

    This is a faithful local reimplementation: LOTUS does not read Any-Join's
    files at runtime and does not depend on their external location.
    """
    system_message = (
        "You are an expert Data Entity Resolution Specialist. "
        "Analyze the input string pairs and return valid, parseable JSON only — "
        "a single JSON array with one object per pair, in the order given. "
        "Do not include markdown formatting."
    )

    pair_lines = "\n".join(
        f'{i}. Value 1: "{v1}" | Value 2: "{v2}"'
        for i, (v1, v2) in enumerate(pairs)
    )

    if include_explanation:
        item_schema = '{"id": <int>, "response": "Yes" or "No", "explanation": "..."}'
    else:
        item_schema = '{"id": <int>, "response": "Yes" or "No"}'

    output_instruction = f'Output a JSON array: [{item_schema}, ...]'

    if mode == "equivalence":
        user_message = textwrap.dedent(f"""
            For each pair below determine if the two values represent the exact same real-world entity.

            Criteria for "Yes": same person, organization, location, or object — including abbreviations,
            misspellings, or standard variations (e.g., "NY" vs "New York").
            Criteria for "No": distinct entities even if related (e.g., "iPhone 12" vs "iPhone 13"),
            or homonyms.

            Pairs:
            {pair_lines}

            {output_instruction}
        """)
    elif mode == "relationship":
        rel_instruction = "Ignore trivial connections. Focus on: Hierarchical, Part/Whole, Spatial/Temporal, Causal."
        user_message = textwrap.dedent(f"""
            For each pair below determine if there is a direct, meaningful semantic relationship
            between the two values.

            {rel_instruction}

            Pairs:
            {pair_lines}

            {output_instruction}
        """)
    else:
        raise ValueError(f"Invalid mode '{mode}'")

    return system_message, user_message


def parse_anyjoin_batch_response(
    raw: str,
    chunk_start: int,
    chunk_len: int,
) -> tuple[dict[int, Any], dict[str, int]]:
    """Parse one Any-Join-style batch response and align items by 0-based ``id``.

    Uses ``json_repair.loads`` exactly as Any-Join's ``batch_llm_verify`` does,
    so repairing behaviour (markdown fences, trailing commas, single quotes,
    prose-wrapped arrays) is identical to the reference implementation.

    Returns ``(responses, diagnostics)`` where ``responses`` maps an *absolute*
    candidate index to the raw response value, and diagnostics counts
    ``malformed``/``missing_ids``/``out_of_range_ids``/``duplicate_ids``.

    Alignment follows Any-Join's ``batch_llm_verify``: each item is placed using
    ``abs_idx = chunk_start + int(item["id"])`` and bounds-checked, so a missing
    id never shifts a later verdict onto an earlier pair. A duplicated id
    overwrites the earlier entry (Any-Join's plain list assignment = last wins).
    ``None``/absent ids and non-dict items are ignored. A bare dict is
    normalised into a single-item list, as Any-Join does.
    """
    if json_repair is None:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "json_repair is required by the Any-Join-style batched verifier. "
            "Install it into the LOTUS environment, e.g. "
            "`python -m pip install json-repair`."
        )

    diagnostics = {
        "malformed": 0,
        "missing_ids": 0,
        "out_of_range_ids": 0,
        "duplicate_ids": 0,
    }
    responses: dict[int, Any] = {}

    parsed: Any = json_repair.loads(raw if isinstance(raw, str) else "")

    if parsed is None or parsed == "":
        diagnostics["malformed"] = chunk_len
        return responses, diagnostics

    # Any-Join normalises a bare dict into a single-item list.
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        diagnostics["malformed"] = chunk_len
        return responses, diagnostics

    for item in parsed:
        if not isinstance(item, dict):
            continue
        local_id = item.get("id")
        if local_id is None:
            diagnostics["missing_ids"] += 1
            continue
        try:
            local_index = int(local_id)
        except (TypeError, ValueError):
            diagnostics["missing_ids"] += 1
            continue
        abs_idx = chunk_start + local_index
        if not (0 <= abs_idx < chunk_start + chunk_len):
            diagnostics["out_of_range_ids"] += 1
            continue
        if abs_idx in responses:
            diagnostics["duplicate_ids"] += 1
        # Any-Join stores the parsed item and reads item["response"]; keep the
        # raw value so acceptance semantics (_response_is_yes) see exactly what
        # the model returned.
        responses[abs_idx] = item.get("response", "")

    return responses, diagnostics


def _response_is_yes(response: Any) -> bool:
    """Any-Join's exact acceptance test: ``str(response).lower() == "yes"``.

    No extra permissive behaviour: no whitespace stripping, no boolean-``True``
    acceptance. A JSON ``true`` becomes the string ``"true"`` and is rejected,
    exactly as the reference expression would evaluate.
    """
    return str(response).lower() == "yes"


def verify_candidates_batched(
    candidates: pd.DataFrame,
    model: LM,
    args: argparse.Namespace,
) -> tuple[list[bool], dict[str, int]]:
    """Any-Join-style batched LLM verification of Top-1 candidate pairs.

    Sends up to ``verification_batch_size`` candidate pairs (default 100)
    in ONE logical chat-completion request, using Any-Join's
    ``batch_values_prompt`` semantics, then aligns each returned item by its
    0-based ``id``. Batches are dataset-local and deterministic: pairs
    ``0..99``, ``100..199``, etc., preserving source-row order.

    Failure policy (matches the benchmark's documented ``default=False``):

    * a pair with no returned verdict (missing/out-of-range id, malformed
      response, or a response that is not "Yes") is **rejected** (False);
    * an API failure that survives the model layer's retries raises, so
      ``--continue-on-error`` records it as a dataset error — it is never
      silently counted as a semantic "No".
    """
    pairs = list(
        zip(
            candidates["source_value"].astype(str).tolist(),
            candidates["target_value"].astype(str).tolist(),
        )
    )
    batch_size = _verification_batch_size(args)
    n_batches = max(1, math.ceil(len(pairs) / batch_size)) if pairs else 0

    verdicts: list[bool] = [False] * len(pairs)
    totals = {
        "verification_batches": 0,
        "verification_pairs_verified": 0,
        "verification_pairs_unresolved": 0,
        "verification_malformed_responses": 0,
        "verification_missing_ids": 0,
        "verification_out_of_range_ids": 0,
        "verification_duplicate_ids": 0,
    }

    for batch_index, chunk_start in enumerate(range(0, len(pairs), batch_size)):
        chunk = pairs[chunk_start : chunk_start + batch_size]
        system_msg, user_msg = anyjoin_batch_values_prompt(
            chunk,
            mode="equivalence",
            include_explanation=False,
        )
        # One conversation -> one logical request. This mirrors Any-Join's
        # ``client.chat.completions.create(model=..., temperature=..., messages=...)``
        # exactly: only model, temperature and messages are sent. Calling the LM
        # directly (rather than via get_completion) avoids injecting an extra
        # ``response_format=None`` parameter that Any-Join never sends; with
        # --omit-lm-completion-budget the request also carries no completion-budget
        # parameter. The model layer handles retries and token accounting.
        output = model(
            [[{"role": "system", "content": system_msg},
              {"role": "user", "content": user_msg}]],
            show_progress_bar=False,
            progress_bar_desc=(
                f"Verifying batch {batch_index + 1}/{n_batches} "
                f"({len(chunk)} pairs)"
            ),
        )
        raw = output.outputs[0]
        totals["verification_batches"] += 1

        responses, diagnostics = parse_anyjoin_batch_response(
            raw if isinstance(raw, str) else str(raw),
            chunk_start,
            len(chunk),
        )
        totals["verification_malformed_responses"] += diagnostics["malformed"]
        totals["verification_missing_ids"] += diagnostics["missing_ids"]
        totals["verification_out_of_range_ids"] += diagnostics["out_of_range_ids"]
        totals["verification_duplicate_ids"] += diagnostics["duplicate_ids"]

        for offset in range(len(chunk)):
            abs_idx = chunk_start + offset
            if abs_idx in responses:
                verdicts[abs_idx] = _response_is_yes(responses[abs_idx])
                totals["verification_pairs_verified"] += 1
            else:
                totals["verification_pairs_unresolved"] += 1

    return verdicts, totals


def top1_retrieval_diagnostics(
    candidates: pd.DataFrame,
    ground_truth: pd.DataFrame,
    source: pd.DataFrame,
    family: str,
) -> dict[str, Any]:
    """Source-row-based retrieval/verification diagnostics (GT used only here)."""
    target_col, source_col = evaluation_pair_columns(family)
    gt_pairs = {(str(t), str(s)) for t, s in pair_set(ground_truth, family)}

    hit_flags = []
    for _, row in candidates.iterrows():
        pair = (str(row[target_col]), str(row[source_col]))
        hit_flags.append(pair in gt_pairs)
    hits = sum(1 for h in hit_flags if h)

    verified = [bool(v) for v in candidates["llm_verified"].tolist()]
    accepted_hits = sum(1 for h, v in zip(hit_flags, verified) if h and v)
    fn_given_hit = sum(1 for h, v in zip(hit_flags, verified) if h and not v)
    false_accepts = sum(1 for h, v in zip(hit_flags, verified) if not h and v)

    eval_metrics = evaluate_predictions(candidates, ground_truth, source, family)
    gt_matched = int(eval_metrics["gt_matched_source_rows"])

    return {
        "top1_retrieval_hits": hits,
        "top1_retrieval_recall": (hits / gt_matched) if gt_matched else 0.0,
        "retrieval_failures": gt_matched - hits,
        "verification_fn_given_retrieval_hit": fn_given_hit,
        "verification_false_accepts": false_accepts,
        "verification_recall_given_retrieval_hit": (accepted_hits / hits) if hits else 0.0,
    }


def duplicate_target_diagnostics(target: pd.DataFrame) -> dict[str, int]:
    values = target["target_value"].astype("string")
    duplicated = values[values.duplicated(keep=False)]
    return {
        "duplicate_target_value_rows": int(len(duplicated)),
        "duplicate_target_values": int(duplicated.nunique()),
    }


def lm_stats_snapshot(lm: LM) -> dict[str, Any]:
    usage = lm.stats
    return {
        "llm_physical_requests": int(usage.physical_requests),
        "llm_retry_attempts": int(usage.retry_attempts),
        "llm_retryable_failures": int(usage.retryable_failures),
        "llm_fatal_failures": int(usage.fatal_failures),
        "llm_cache_hits": int(usage.cache_hits),
        "llm_physical_prompt_tokens": int(usage.physical_usage.prompt_tokens),
        "llm_physical_completion_tokens": int(usage.physical_usage.completion_tokens),
        "llm_physical_reasoning_tokens": int(usage.physical_usage.reasoning_tokens),
        "llm_physical_total_tokens": int(usage.physical_usage.total_tokens),
        "llm_physical_cost": float(usage.physical_usage.total_cost),
    }


def run_one_dataset(
    *,
    repo_root: Path,
    family: str,
    dataset_dir: Path,
    args: argparse.Namespace,
    repository_commit: str | None,
) -> Top1RunResult:
    dataset = dataset_dir.name
    output_dir = args.output_root / family / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_file = output_dir / "predictions.csv"
    candidates_file = output_dir / "top1_candidates.csv"
    metrics_file = output_dir / "metrics.json"
    schema_file = output_dir / "resolved_schema.json"
    index_dir = output_dir / "target_index"

    source, target, ground_truth, schema = load_dataset(dataset_dir, family, args)

    schema_file.write_text(
        json.dumps(asdict(schema), indent=2),
        encoding="utf-8",
    )

    cartesian_pairs = len(source) * len(target)
    gt_pairs = pair_set(ground_truth, family)
    expected_embedding_requests = (
        math.ceil(len(source) / args.embedding_batch_size)
        + math.ceil(len(target) / args.embedding_batch_size)
    )

    print(f"\n{'=' * 72}")
    print(f"Family:              {family}")
    print(f"Dataset:             {dataset}")
    print(f"Source/query rows:   {len(source)}")
    print(f"Target/reference:    {len(target)}")
    print(f"Cartesian pairs:     {cartesian_pairs}")
    print(f"Ground-truth pairs:  {len(gt_pairs)}")
    print(
        "Evaluation pairs:    "
        + (
            "(target_value, source_value)"
            if family in ("kbwt", "ss", "wt")
            else "(target_id, source_id)"
        )
    )
    print(f"Instruction:         {args.join_instruction}")
    print(f"{'=' * 72}")

    if args.dry_run:
        empty_candidates = pd.DataFrame(
            columns=[
                "target_id",
                "target_value",
                "source_id",
                "source_value",
                "similarity_score",
                "llm_verified",
            ]
        )
        dry_metrics = evaluate_predictions(
            empty_candidates[["target_id", "target_value", "source_id", "source_value"]],
            ground_truth,
            source,
            family,
        )
        print(f"Expected top-1 candidates:            {len(source)}")
        print(f"Verification mode:                    {_verification_mode(args)}")
        if _verification_mode(args) == "anyjoin_batch":
            expected_batches = (
                math.ceil(len(source) / max(1, int(_verification_batch_size(args))))
                if len(source)
                else 0
            )
            print(
                f"Expected logical LLM batch requests:  {expected_batches} "
                f"({_verification_batch_size(args)} pairs/request)"
            )
        else:
            expected_batches = len(source)
            print(f"Expected logical LLM verifications:   {len(source)}")
        print(f"Expected embedding requests:          {expected_embedding_requests}")
        return Top1RunResult(
            family=family,
            dataset=dataset,
            status="dry_run",
            embedding_model=args.embedding_model,
            embedding_backend=_embedding_backend(args),
        embedding_query_prefix=_embedding_query_prefix(args),
            lm_model=args.lm_model,
            lm_temperature=0.0,
        lm_completion_budget_omitted=_omit_lm_completion_budget(args),
            verification_mode=_verification_mode(args),
            verification_batch_size=int(_verification_batch_size(args)),
            join_instruction=args.join_instruction,
            default_on_parse_failure=False,
            source_rows=len(source),
            target_rows=len(target),
            cartesian_pairs=cartesian_pairs,
            ground_truth_pairs=int(dry_metrics["ground_truth_pairs"]),
            gt_matched_source_rows=int(dry_metrics["gt_matched_source_rows"]),
            gt_unmatched_source_rows=int(dry_metrics["gt_unmatched_source_rows"]),
            retrieved_candidates=0,
            expected_top1_candidates=len(source),
            top1_retrieval_hits=0,
            top1_retrieval_recall=0.0,
            duplicate_target_value_rows=0,
            duplicate_target_values=0,
            logical_verification_calls=0,
            expected_logical_verification_calls=expected_batches,
            verification_batches=0,
            verification_pairs_verified=0,
            verification_pairs_unresolved=0,
            verification_malformed_responses=0,
            verification_missing_ids=0,
            verification_out_of_range_ids=0,
            verification_duplicate_ids=0,
            accepted_pairs=0,
            rejected_candidates=0,
            retrieval_failures=0,
            verification_fn_given_retrieval_hit=0,
            verification_false_accepts=0,
            verification_recall_given_retrieval_hit=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            source_coverage=0.0,
            true_positives=0,
            false_positives=0,
            false_negatives=int(dry_metrics["false_negatives"]),
            indexing_seconds=0.0,
            retrieval_seconds=0.0,
            verification_seconds=0.0,
            evaluation_seconds=0.0,
            total_seconds=0.0,
            embedding_rm_invocations=0,
            embedding_api_requests=0,
            embedded_rows=0,
            embedding_batches=0,
            llm_physical_requests=0,
            llm_retry_attempts=0,
            llm_retryable_failures=0,
            llm_fatal_failures=0,
            llm_cache_hits=0,
            llm_physical_prompt_tokens=0,
            llm_physical_completion_tokens=0,
            llm_physical_reasoning_tokens=0,
            llm_physical_total_tokens=0,
            llm_physical_cost=0.0,
            repository_commit=repository_commit,
            predictions_file=str(predictions_file),
            candidates_file=str(candidates_file),
            metrics_file=str(metrics_file),
            schema_file=str(schema_file),
        )

    if len(target) < 1:
        raise DatasetSchemaError("The target/indexed table is empty.")

    shutil.rmtree(index_dir, ignore_errors=True)

    lm, rm, vs = configure_models(args)

    total_start = time.perf_counter()
    indexed_target, candidates, indexing_seconds, retrieval_seconds = retrieve_top1(
        source, target, index_dir, args
    )

    verification_start = time.perf_counter()
    if _verification_mode(args) == "anyjoin_batch":
        verified, verification_totals = verify_candidates_batched(candidates, lm, args)
    else:
        verified = verify_candidates(candidates, lm, args)
        verification_totals = {
            "verification_batches": 0,
            "verification_pairs_verified": len(candidates),
            "verification_pairs_unresolved": 0,
            "verification_malformed_responses": 0,
            "verification_missing_ids": 0,
            "verification_out_of_range_ids": 0,
            "verification_duplicate_ids": 0,
        }
    verification_seconds = time.perf_counter() - verification_start
    candidates["llm_verified"] = verified

    # Optional provenance: which verification batch/position produced each
    # verdict. Added only for the batched verifier; the columns are appended so
    # the historical standardized schema stays intact.
    if _verification_mode(args) == "anyjoin_batch":
        batch_size = max(1, int(_verification_batch_size(args)))
        candidates["verification_batch_id"] = [
            (i // batch_size) for i in range(len(candidates))
        ]
        candidates["verification_batch_position"] = [
            (i % batch_size) for i in range(len(candidates))
        ]

    candidates.to_csv(candidates_file, index=False)

    predictions = candidates.loc[candidates["llm_verified"] == True].copy()  # noqa: E712
    # Keep predictions.csv in the historical schema; batch provenance lives in
    # top1_candidates.csv only.
    predictions = predictions.drop(
        columns=[
            col
            for col in ("verification_batch_id", "verification_batch_position")
            if col in predictions.columns
        ]
    )
    # Metadata columns appended at the output boundary, after filtering and
    # batch-provenance removal, so the six historical columns keep their exact
    # order. ``dataset`` is the same identifier the run result and metrics use
    # (the dataset directory name); llm_model is the effective provider-request
    # model identifier that LiteLLM resolves for the configured LM; the column
    # names come from the resolved schema and preserve original header spelling.
    predictions["dataset"] = dataset
    predictions["llm_model"] = _effective_provider_model(lm.model)
    predictions["target_column"] = schema.target_value_col
    predictions["source_column"] = schema.source_value_col
    predictions.to_csv(predictions_file, index=False)

    evaluation_start = time.perf_counter()
    metrics = evaluate_predictions(
        predictions[["target_id", "target_value", "source_id", "source_value"]],
        ground_truth,
        source,
        family,
    )
    diagnostics = top1_retrieval_diagnostics(candidates, ground_truth, source, family)
    duplicates = duplicate_target_diagnostics(target)
    evaluation_seconds = time.perf_counter() - evaluation_start
    total_seconds = time.perf_counter() - total_start

    lm_stats = lm_stats_snapshot(lm)

    result = Top1RunResult(
        family=family,
        dataset=dataset,
        status="success",
        embedding_model=args.embedding_model,
        embedding_backend=_embedding_backend(args),
        embedding_query_prefix=_embedding_query_prefix(args),
        lm_model=args.lm_model,
        lm_temperature=0.0,
        lm_completion_budget_omitted=_omit_lm_completion_budget(args),
        verification_mode=_verification_mode(args),
        verification_batch_size=int(_verification_batch_size(args)),
        join_instruction=args.join_instruction,
        default_on_parse_failure=False,
        source_rows=len(source),
        target_rows=len(target),
        cartesian_pairs=cartesian_pairs,
        ground_truth_pairs=int(metrics["ground_truth_pairs"]),
        gt_matched_source_rows=int(metrics["gt_matched_source_rows"]),
        gt_unmatched_source_rows=int(metrics["gt_unmatched_source_rows"]),
        retrieved_candidates=len(candidates),
        expected_top1_candidates=len(source),
        top1_retrieval_hits=int(diagnostics["top1_retrieval_hits"]),
        top1_retrieval_recall=float(diagnostics["top1_retrieval_recall"]),
        duplicate_target_value_rows=int(duplicates["duplicate_target_value_rows"]),
        duplicate_target_values=int(duplicates["duplicate_target_values"]),
        # For the batched verifier, one logical call verifies up to
        # verification_batch_size pairs; for the historical sem_filter path each
        # pair is its own logical call. Pairs and requests are reported separately.
        logical_verification_calls=(
            int(verification_totals["verification_batches"])
            if _verification_mode(args) == "anyjoin_batch"
            else len(candidates)
        ),
        expected_logical_verification_calls=(
            math.ceil(len(candidates) / max(1, int(_verification_batch_size(args))))
            if _verification_mode(args) == "anyjoin_batch"
            else len(source)
        ),
        verification_batches=int(verification_totals["verification_batches"]),
        verification_pairs_verified=int(verification_totals["verification_pairs_verified"]),
        verification_pairs_unresolved=int(verification_totals["verification_pairs_unresolved"]),
        verification_malformed_responses=int(
            verification_totals["verification_malformed_responses"]
        ),
        verification_missing_ids=int(verification_totals["verification_missing_ids"]),
        verification_out_of_range_ids=int(
            verification_totals["verification_out_of_range_ids"]
        ),
        verification_duplicate_ids=int(verification_totals["verification_duplicate_ids"]),
        accepted_pairs=int(metrics["prediction_pairs"]),
        rejected_candidates=len(candidates) - int(metrics["prediction_pairs"]),
        retrieval_failures=int(diagnostics["retrieval_failures"]),
        verification_fn_given_retrieval_hit=int(
            diagnostics["verification_fn_given_retrieval_hit"]
        ),
        verification_false_accepts=int(diagnostics["verification_false_accepts"]),
        verification_recall_given_retrieval_hit=float(
            diagnostics["verification_recall_given_retrieval_hit"]
        ),
        precision=float(metrics["precision"]),
        recall=float(metrics["recall"]),
        f1=float(metrics["f1"]),
        source_coverage=float(metrics["source_coverage"]),
        true_positives=int(metrics["true_positives"]),
        false_positives=int(metrics["false_positives"]),
        false_negatives=int(metrics["false_negatives"]),
        indexing_seconds=indexing_seconds,
        retrieval_seconds=retrieval_seconds,
        verification_seconds=verification_seconds,
        evaluation_seconds=evaluation_seconds,
        total_seconds=total_seconds,
        embedding_rm_invocations=int(rm.embedding_rm_invocations),
        embedding_api_requests=int(rm.embedding_api_requests),
        embedded_rows=int(rm.embedded_rows),
        embedding_batches=int(getattr(rm, "embedding_batches", 0)),
        llm_physical_requests=int(lm_stats["llm_physical_requests"]),
        llm_retry_attempts=int(lm_stats["llm_retry_attempts"]),
        llm_retryable_failures=int(lm_stats["llm_retryable_failures"]),
        llm_fatal_failures=int(lm_stats["llm_fatal_failures"]),
        llm_cache_hits=int(lm_stats["llm_cache_hits"]),
        llm_physical_prompt_tokens=int(lm_stats["llm_physical_prompt_tokens"]),
        llm_physical_completion_tokens=int(lm_stats["llm_physical_completion_tokens"]),
        llm_physical_reasoning_tokens=int(lm_stats["llm_physical_reasoning_tokens"]),
        llm_physical_total_tokens=int(lm_stats["llm_physical_total_tokens"]),
        llm_physical_cost=float(lm_stats["llm_physical_cost"]),
        repository_commit=repository_commit,
        predictions_file=str(predictions_file),
        candidates_file=str(candidates_file),
        metrics_file=str(metrics_file),
        schema_file=str(schema_file),
    )

    metrics_file.write_text(
        json.dumps(asdict(result), indent=2),
        encoding="utf-8",
    )

    print(f"Retrieved candidates:   {result.retrieved_candidates}")
    print(f"Logical verifications:  {result.logical_verification_calls}")
    print(f"Accepted pairs:         {result.accepted_pairs}")
    print(f"Rejected candidates:    {result.rejected_candidates}")
    print(f"Top-1 retrieval hits:   {result.top1_retrieval_hits}")
    print(f"Top-1 retrieval recall: {result.top1_retrieval_recall:.6f}")
    print(f"Precision:              {result.precision:.6f}")
    print(f"Recall:                 {result.recall:.6f}")
    print(f"F1:                     {result.f1:.6f}")
    print(f"Source coverage:        {result.source_coverage:.6f}")
    print(f"Index time:             {result.indexing_seconds:.4f} s")
    print(f"Retrieval time:         {result.retrieval_seconds:.4f} s")
    print(f"Verification time:      {result.verification_seconds:.4f} s")
    print(f"Total time:             {result.total_seconds:.4f} s")
    print(f"Embedding requests:     {result.embedding_api_requests}")
    print(f"Physical LLM requests:  {result.llm_physical_requests}")
    print(f"LLM cache hits:         {result.llm_cache_hits}")
    print(f"Predictions:            {predictions_file}")
    print(f"Candidates:             {candidates_file}")

    return result


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "LOTUS Top-1 + LLM Verify baseline: for each source row, retrieve "
            "exactly one semantic top-1 target with sem_sim_join(K=1), then run "
            "one binary LLM verification via sem_filter. The source table is the "
            "query side and the target table is semantically indexed (never "
            "reversed). This is not native LOTUS sem_join()."
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

    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=(
            "Retrieval embedding model. Historical default is the hosted "
            f"'{DEFAULT_EMBEDDING_MODEL}'; the corrected fair baseline uses "
            f"'{DEFAULT_BGE_EMBEDDING_MODEL}' with "
            "--embedding-backend sentence_transformers."
        ),
    )
    parser.add_argument(
        "--embedding-backend",
        choices=("litellm", "sentence_transformers"),
        default="litellm",
        help=(
            "Embedding backend. 'litellm' keeps the historical hosted embedding "
            "model; 'sentence_transformers' loads a local HuggingFace model "
            "(e.g. BAAI/bge-base-en-v1.5) via LOTUS SentenceTransformersRM."
        ),
    )
    parser.add_argument("--lm-model", default=DEFAULT_LM_MODEL)
    parser.add_argument(
        "--lm-max-tokens",
        type=int,
        default=None,
        help=(
            "Optional completion-token budget passed to the LM. Defaults to the "
            "LM's own default (None). Mutually exclusive with "
            "--omit-lm-completion-budget."
        ),
    )
    parser.add_argument(
        "--omit-lm-completion-budget",
        action="store_true",
        help=(
            "Any-Join alignment: omit max_completion_tokens/max_tokens/"
            "max_output_tokens from the LLM API request entirely instead of "
            "sending LOTUS's default completion budget. Off by default so "
            "historical behaviour is unchanged."
        ),
    )
    parser.add_argument(
        "--verification-mode",
        choices=("sem_filter", "anyjoin_batch"),
        default=DEFAULT_VERIFICATION_MODE,
        help=(
            "Verification procedure. 'sem_filter' is the historical one-logical-"
            "call-per-pair path; 'anyjoin_batch' verifies up to "
            "--verification-batch-size pairs per logical request using Any-Join's "
            "batch_values_prompt semantics with id-aligned parsing."
        ),
    )
    parser.add_argument(
        "--verification-batch-size",
        type=int,
        default=DEFAULT_VERIFICATION_BATCH_SIZE,
        help=(
            "Candidate pairs per logical LLM verification request in "
            "'anyjoin_batch' mode (default 100). This controls pairs inside ONE "
            "prompt and is distinct from --llm-batch-size, which caps concurrent "
            "independent requests in the model layer."
        ),
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
    )
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=DEFAULT_LLM_BATCH_SIZE,
        help="Physical LLM batching inside LM; never changes the logical call count.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Model-layer transient retries (after the first physical attempt).",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_SECONDS,
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
        default=repo_root / "outputs" / "lotus_top1_llm_verify",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate dataset/schema/counts with zero embedding and zero LLM "
            "API calls."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue a family batch after a dataset fails.",
    )

    # Column overrides (mainly for WT, whose tables carry arbitrary schemas).
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
    if getattr(args, "verification_batch_size", 1) < 1:
        parser.error("--verification-batch-size must be at least 1.")
    if getattr(args, "lm_max_tokens", None) is not None and args.lm_max_tokens < 1:
        parser.error("--lm-max-tokens must be at least 1.")
    if _omit_lm_completion_budget(args) and getattr(args, "lm_max_tokens", None) is not None:
        parser.error(
            "--lm-max-tokens cannot be combined with --omit-lm-completion-budget: "
            "one sets an explicit completion budget, the other omits it entirely."
        )


def main() -> int:
    repo_root = find_repo_root(Path(__file__).parent)
    parser = build_parser(repo_root)
    args = parser.parse_args()
    validate_args(parser, args)

    # The entity-equivalence instruction is a fixed part of the baseline design
    # (not a user-facing flag), but the runner reads it from the namespace.
    args.join_instruction = DEFAULT_JOIN_INSTRUCTION

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

    results: list[Top1RunResult] = []
    failures: list[dict[str, str]] = []

    for dataset_dir in datasets:
        try:
            result = run_one_dataset(
                repo_root=repo_root,
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
