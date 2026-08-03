from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

import lotus
from lotus.models import LM, LiteLLMRM
from lotus.vector_store import FaissVS


DEFAULT_EMBEDDING_MODEL = "openrouter/openai/text-embedding-3-small"
DEFAULT_LM_MODEL = "openrouter/openai/gpt-5-mini"
DEFAULT_BATCH_SIZE = 64
DEFAULT_K = 1


class DatasetSchemaError(ValueError):
    """Raised when a dataset's files or columns cannot be resolved safely."""


@dataclass(frozen=True)
class FamilyLayout:
    source_filename: str
    target_filename: str
    gt_filenames: tuple[str, ...]
    source_id_candidates: tuple[str, ...]
    source_value_candidates: tuple[str, ...]
    target_id_candidates: tuple[str, ...]
    target_value_candidates: tuple[str, ...]
    gt_source_candidates: tuple[str, ...]
    gt_target_candidates: tuple[str, ...]


FAMILY_LAYOUTS: dict[str, FamilyLayout] = {
    "autofj": FamilyLayout(
        source_filename="right.csv",
        target_filename="left.csv",
        gt_filenames=("gt.csv",),
        source_id_candidates=("id", "id_r", "source_id"),
        source_value_candidates=("value", "source_value"),
        target_id_candidates=("id", "id_l", "target_id"),
        target_value_candidates=("value", "target_value"),
        gt_source_candidates=("id_r", "source_id"),
        gt_target_candidates=("id_l", "target_id"),
    ),
    "autofj_overlap": FamilyLayout(
        source_filename="right.csv",
        target_filename="left.csv",
        gt_filenames=("gt.csv",),
        source_id_candidates=("id", "id_r", "source_id"),
        source_value_candidates=("value", "source_value"),
        target_id_candidates=("id", "id_l", "target_id"),
        target_value_candidates=("value", "target_value"),
        gt_source_candidates=("id_r", "source_id"),
        gt_target_candidates=("id_l", "target_id"),
    ),
    "kbwt": FamilyLayout(
        source_filename="source.csv",
        target_filename="target.csv",
        gt_filenames=(
            "ground truth.csv",
            "ground_truth.csv",
            "ground-truth.csv",
            "gt.csv",
        ),
        source_id_candidates=("source_id", "id", "row_id", "source_row_id"),
        source_value_candidates=(
            "source_value",
            "value",
            "source",
            "name",
            "title",
            "text",
        ),
        target_id_candidates=("target_id", "id", "row_id", "target_row_id"),
        target_value_candidates=(
            "target_value",
            "value",
            "target",
            "name",
            "title",
            "text",
        ),
        gt_source_candidates=("source_id", "id_r", "source", "source_row_id"),
        gt_target_candidates=("target_id", "id_l", "target", "target_row_id"),
    ),
}


@dataclass
class ResolvedSchema:
    source_file: str
    target_file: str
    ground_truth_file: str
    source_id_col: str
    source_value_col: str
    target_id_col: str
    target_value_col: str
    gt_source_col: str
    gt_target_col: str
    source_id_is_index: bool = False
    target_id_is_index: bool = False


@dataclass
class RunResult:
    family: str
    dataset: str
    status: str
    embedding_model: str
    lm_model: str
    k: int
    score_threshold: float | None
    source_rows: int
    target_rows: int
    raw_prediction_rows: int
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
    index_seconds: float
    join_seconds: float
    total_seconds: float
    expected_target_embedding_requests: int
    expected_source_embedding_requests: int
    expected_total_embedding_requests: int
    embedded_target_rows: int
    embedded_source_rows: int
    expected_llm_calls: int
    faiss_factory: str
    faiss_metric: str
    score_semantics: str
    random_seed: str
    repository_commit: str | None
    predictions_file: str
    raw_predictions_file: str
    metrics_file: str
    schema_file: str
    error: str | None = None


def find_repo_root(start: Path) -> Path:
    """Find the LOTUS checkout without relying on a fixed parents[N] depth."""
    resolved = start.resolve()
    for directory in (resolved, *resolved.parents):
        if (directory / "pyproject.toml").is_file() and (directory / "lotus").is_dir():
            return directory
    raise FileNotFoundError(
        f"Could not locate the LOTUS repository root above {resolved}. "
        "Expected a directory containing pyproject.toml and the lotus package."
    )


def get_git_commit(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def load_openrouter_key(repo_root: Path, key_file: Path | None) -> None:
    """Load OPENROUTER_API_KEY only when it is not already set."""
    if os.getenv("OPENROUTER_API_KEY"):
        return

    path = key_file.resolve() if key_file else repo_root / "openrouter.key"
    if not path.is_file():
        raise FileNotFoundError(
            "OPENROUTER_API_KEY is not set and the key file was not found: "
            f"{path}"
        )

    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"OpenRouter key file is empty: {path}")

    os.environ["OPENROUTER_API_KEY"] = key


def configure_lotus(
    embedding_model: str,
    lm_model: str,
    batch_size: int,
) -> tuple[LiteLLMRM, FaissVS]:
    lm = LM(model=lm_model)
    rm = LiteLLMRM(
        model=embedding_model,
        max_batch_size=batch_size,
    )
    vs = FaissVS()

    # Clear results from previous runs before this dataset starts.
    if lm.cache is not None:
        lm.cache.reset()

    lotus.settings.configure(
        lm=lm,
        rm=rm,
        vs=vs,
        enable_cache=True,
    )

    return rm, vs

def find_case_insensitive_file(dataset_dir: Path, names: Iterable[str]) -> Path:
    files = {
        child.name.casefold(): child
        for child in dataset_dir.iterdir()
        if child.is_file()
    }
    for name in names:
        match = files.get(name.casefold())
        if match is not None:
            return match
    raise FileNotFoundError(
        f"None of the expected files {list(names)} were found in {dataset_dir}. "
        f"Available files: {sorted(child.name for child in dataset_dir.iterdir() if child.is_file())}"
    )


def resolve_existing_column(
    df: pd.DataFrame,
    explicit: str | None,
    candidates: Iterable[str],
    *,
    role: str,
    allow_index: bool = False,
    exclude: set[str] | None = None,
) -> tuple[str, bool]:
    exclude = exclude or set()

    if explicit:
        if explicit == "__index__":
            if not allow_index:
                raise DatasetSchemaError(f"{role} cannot use __index__.")
            return explicit, True
        if explicit not in df.columns:
            raise DatasetSchemaError(
                f"{role} column {explicit!r} was not found. "
                f"Available columns: {df.columns.tolist()}"
            )
        return explicit, False

    for candidate in candidates:
        if candidate in df.columns and candidate not in exclude:
            return candidate, False

    remaining = [column for column in df.columns if column not in exclude]
    if len(remaining) == 1:
        return remaining[0], False

    index_hint = (
        " Use the corresponding option with __index__ only when the ground "
        "truth explicitly uses zero-based CSV row positions."
        if allow_index
        else ""
    )
    raise DatasetSchemaError(
        f"Could not resolve {role}. Available columns: {df.columns.tolist()}. "
        "Provide the corresponding command-line column option explicitly."
        f"{index_hint}"
    )


def prepare_id_series(
    df: pd.DataFrame,
    column: str,
    use_index: bool,
    *,
    role: str,
) -> pd.Series:
    if use_index:
        return pd.Series(df.index.astype(str), index=df.index, dtype="string")

    series = df[column].astype("string")
    if series.isna().any():
        raise DatasetSchemaError(f"{role} contains null IDs in column {column!r}.")
    return series


def validate_text_values(series: pd.Series, role: str, column: str) -> pd.Series:
    values = series.astype("string")
    invalid = values.isna() | values.fillna("").str.strip().eq("")
    if invalid.any():
        rows = invalid[invalid].index.tolist()[:10]
        raise DatasetSchemaError(
            f"{role} column {column!r} contains {int(invalid.sum())} null or "
            f"empty values. First affected row indices: {rows}"
        )
    return values



def _strip_string_series(series: pd.Series) -> pd.Series:
    """Convert a column to trimmed pandas strings without changing CSV files."""
    return series.astype("string").str.strip()


def _casefold_column_lookup(df: pd.DataFrame, name: str) -> str | None:
    wanted = name.casefold()
    for column in df.columns:
        if str(column).casefold() == wanted:
            return str(column)
    return None


def _membership_coverage(series: pd.Series, allowed: set[str]) -> float:
    values = _strip_string_series(series)
    if len(values) == 0 or values.isna().any():
        return 0.0
    return float(values.isin(allowed).mean())


def _resolve_kbwt_value_column(
    df: pd.DataFrame,
    explicit: str | None,
    candidates: Iterable[str],
    *,
    role: str,
) -> str:
    if explicit:
        if explicit not in df.columns:
            raise DatasetSchemaError(
                f"{role} column {explicit!r} was not found. "
                f"Available columns: {df.columns.tolist()}"
            )
        return explicit

    if len(df.columns) == 1:
        return str(df.columns[0])

    for candidate in candidates:
        match = _casefold_column_lookup(df, candidate)
        if match is not None:
            return match

    raise DatasetSchemaError(
        f"Could not resolve {role}. Available columns: {df.columns.tolist()}. "
        "Provide the corresponding command-line value-column option explicitly."
    )


def _resolve_kbwt_id_column(
    df: pd.DataFrame,
    explicit: str | None,
    candidates: Iterable[str],
    *,
    value_column: str,
    role: str,
) -> tuple[str, bool]:
    if explicit:
        if explicit == "__index__":
            return explicit, True
        if explicit not in df.columns:
            raise DatasetSchemaError(
                f"{role} column {explicit!r} was not found. "
                f"Available columns: {df.columns.tolist()}"
            )
        if explicit == value_column:
            raise DatasetSchemaError(
                f"{role} and join value both resolve to {explicit!r}. "
                "Use --source-id-col __index__ or --target-id-col __index__ "
                "for value-only KBWT tables."
            )
        return explicit, False

    for candidate in candidates:
        match = _casefold_column_lookup(df, candidate)
        if match is not None and match != value_column:
            return match, False

    # Original KBWT source.csv and target.csv files are commonly value-only.
    # In that case, use stable zero-based CSV row positions as internal IDs.
    return "__index__", True


def _side_match_score(
    gt_series: pd.Series,
    ids: pd.Series,
    values: pd.Series,
) -> float:
    id_set = set(_strip_string_series(ids).dropna().tolist())
    value_set = set(_strip_string_series(values).dropna().tolist())
    return max(
        _membership_coverage(gt_series, id_set),
        _membership_coverage(gt_series, value_set),
    )


def _resolve_kbwt_gt_columns(
    gt_raw: pd.DataFrame,
    *,
    source_value_col: str,
    target_value_col: str,
    source_ids: pd.Series,
    source_values: pd.Series,
    target_ids: pd.Series,
    target_values: pd.Series,
    explicit_source: str | None,
    explicit_target: str | None,
) -> tuple[str, str]:
    columns = [str(column) for column in gt_raw.columns]

    if explicit_source:
        if explicit_source not in gt_raw.columns:
            raise DatasetSchemaError(
                f"Ground-truth source column {explicit_source!r} was not found. "
                f"Available columns: {columns}"
            )
        gt_source_col = explicit_source
    else:
        gt_source_col = ""

    if explicit_target:
        if explicit_target not in gt_raw.columns:
            raise DatasetSchemaError(
                f"Ground-truth target column {explicit_target!r} was not found. "
                f"Available columns: {columns}"
            )
        gt_target_col = explicit_target
    else:
        gt_target_col = ""

    if gt_source_col and gt_target_col:
        if gt_source_col == gt_target_col:
            raise DatasetSchemaError(
                "Ground-truth source and target columns cannot be the same."
            )
        return gt_source_col, gt_target_col

    if len(columns) != 2:
        raise DatasetSchemaError(
            "KBWT ground truth must have exactly two columns unless both "
            "--gt-source-col and --gt-target-col are supplied. "
            f"Available columns: {columns}"
        )

    if gt_source_col:
        remaining = [column for column in columns if column != gt_source_col]
        return gt_source_col, remaining[0]

    if gt_target_col:
        remaining = [column for column in columns if column != gt_target_col]
        return remaining[0], gt_target_col

    # Prefer exact header matches, which is the common KBWT format.
    source_header_match = _casefold_column_lookup(gt_raw, source_value_col)
    target_header_match = _casefold_column_lookup(gt_raw, target_value_col)
    if (
        source_header_match is not None
        and target_header_match is not None
        and source_header_match != target_header_match
    ):
        return source_header_match, target_header_match

    first, second = columns
    forward_score = _side_match_score(
        gt_raw[first], source_ids, source_values
    ) + _side_match_score(
        gt_raw[second], target_ids, target_values
    )
    reverse_score = _side_match_score(
        gt_raw[second], source_ids, source_values
    ) + _side_match_score(
        gt_raw[first], target_ids, target_values
    )

    if forward_score > reverse_score:
        return first, second
    if reverse_score > forward_score:
        return second, first

    raise DatasetSchemaError(
        "Could not determine KBWT ground-truth orientation safely. "
        f"Columns: {columns}. Provide --gt-source-col and --gt-target-col."
    )


def _map_kbwt_gt_side_to_ids(
    gt_series: pd.Series,
    *,
    side_ids: pd.Series,
    side_values: pd.Series,
    gt_column: str,
    role: str,
) -> pd.Series:
    gt_values = _strip_string_series(gt_series)
    if gt_values.isna().any() or gt_values.fillna("").eq("").any():
        raise DatasetSchemaError(
            f"Ground-truth {role} column {gt_column!r} contains null or empty values."
        )

    ids = _strip_string_series(side_ids)
    values = _strip_string_series(side_values)
    id_set = set(ids.tolist())
    value_set = set(values.tolist())

    id_coverage = float(gt_values.isin(id_set).mean())
    value_coverage = float(gt_values.isin(value_set).mean())

    if id_coverage == 1.0 and value_coverage < 1.0:
        return gt_values

    if value_coverage == 1.0 and id_coverage < 1.0:
        mode = "value"
    elif id_coverage == 1.0 and value_coverage == 1.0:
        # A column explicitly named as an ID is more likely to contain IDs.
        mode = "id" if "id" in gt_column.casefold() else "value"
        if mode == "id":
            return gt_values
    else:
        missing_as_ids = sorted(set(gt_values) - id_set)[:5]
        missing_as_values = sorted(set(gt_values) - value_set)[:5]
        raise DatasetSchemaError(
            f"Ground-truth {role} column {gt_column!r} matches neither the "
            f"{role} IDs nor values completely. Missing as IDs: {missing_as_ids}; "
            f"missing as values: {missing_as_values}"
        )

    duplicate_values = values[values.duplicated(keep=False)]
    if not duplicate_values.empty:
        examples = duplicate_values.drop_duplicates().head(10).tolist()
        raise DatasetSchemaError(
            f"Cannot map value-based KBWT ground truth to {role} row IDs because "
            f"the {role} table contains duplicate join values. Examples: {examples}"
        )

    value_to_id = dict(zip(values.tolist(), ids.tolist()))
    mapped = gt_values.map(value_to_id).astype("string")
    if mapped.isna().any():
        raise DatasetSchemaError(
            f"Failed to map some ground-truth {role} values to row IDs."
        )
    return mapped


def load_kbwt_dataset(
    *,
    source_path: Path,
    target_path: Path,
    gt_path: Path,
    layout: FamilyLayout,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ResolvedSchema]:
    source_raw = pd.read_csv(source_path, dtype="string").reset_index(drop=True)
    target_raw = pd.read_csv(target_path, dtype="string").reset_index(drop=True)
    gt_raw = pd.read_csv(gt_path, dtype="string")

    source_value_col = _resolve_kbwt_value_column(
        source_raw,
        args.source_value_col,
        layout.source_value_candidates,
        role="source join value",
    )
    target_value_col = _resolve_kbwt_value_column(
        target_raw,
        args.target_value_col,
        layout.target_value_candidates,
        role="target join value",
    )

    source_id_col, source_id_is_index = _resolve_kbwt_id_column(
        source_raw,
        args.source_id_col,
        layout.source_id_candidates,
        value_column=source_value_col,
        role="source ID",
    )
    target_id_col, target_id_is_index = _resolve_kbwt_id_column(
        target_raw,
        args.target_id_col,
        layout.target_id_candidates,
        value_column=target_value_col,
        role="target ID",
    )

    source_ids = prepare_id_series(
        source_raw, source_id_col, source_id_is_index, role="source ID"
    )
    target_ids = prepare_id_series(
        target_raw, target_id_col, target_id_is_index, role="target ID"
    )
    source_values = validate_text_values(
        source_raw[source_value_col], "source", source_value_col
    )
    target_values = validate_text_values(
        target_raw[target_value_col], "target", target_value_col
    )

    source = pd.DataFrame(
        {
            "source_id": source_ids,
            "source_value": source_values,
        }
    )
    target = pd.DataFrame(
        {
            "target_id": target_ids,
            "target_value": target_values,
        }
    )

    gt_source_col, gt_target_col = _resolve_kbwt_gt_columns(
        gt_raw,
        source_value_col=source_value_col,
        target_value_col=target_value_col,
        source_ids=source["source_id"],
        source_values=source["source_value"],
        target_ids=target["target_id"],
        target_values=target["target_value"],
        explicit_source=args.gt_source_col,
        explicit_target=args.gt_target_col,
    )

    # KBWT ground truth is value-based. Preserve value pairs directly instead
    # of forcing them into row IDs, because duplicate source or target values
    # make row-ID reconstruction ambiguous and arbitrary.
    gt_source_values = validate_text_values(
        gt_raw[gt_source_col], "ground-truth source", gt_source_col
    )
    gt_target_values = validate_text_values(
        gt_raw[gt_target_col], "ground-truth target", gt_target_col
    )

    missing_source_values = sorted(
        set(gt_source_values.astype(str)) - set(source["source_value"].astype(str))
    )
    missing_target_values = sorted(
        set(gt_target_values.astype(str)) - set(target["target_value"].astype(str))
    )
    if missing_source_values:
        raise DatasetSchemaError(
            "KBWT ground truth contains source values not found in source.csv. "
            f"Examples: {missing_source_values[:10]}"
        )
    if missing_target_values:
        raise DatasetSchemaError(
            "KBWT ground truth contains target values not found in target.csv. "
            f"Examples: {missing_target_values[:10]}"
        )

    ground_truth = pd.DataFrame(
        {
            "target_value": gt_target_values,
            "source_value": gt_source_values,
        }
    )

    if source.empty:
        raise DatasetSchemaError("The source/query table is empty.")
    if target.empty:
        raise DatasetSchemaError("The target/indexed table is empty.")
    if not source["source_id"].is_unique:
        raise DatasetSchemaError("Generated or supplied KBWT source IDs are not unique.")
    if not target["target_id"].is_unique:
        raise DatasetSchemaError("Generated or supplied KBWT target IDs are not unique.")

    schema = ResolvedSchema(
        source_file=str(source_path),
        target_file=str(target_path),
        ground_truth_file=str(gt_path),
        source_id_col=source_id_col,
        source_value_col=source_value_col,
        target_id_col=target_id_col,
        target_value_col=target_value_col,
        gt_source_col=gt_source_col,
        gt_target_col=gt_target_col,
        source_id_is_index=source_id_is_index,
        target_id_is_index=target_id_is_index,
    )
    return source, target, ground_truth, schema

def load_dataset(
    dataset_dir: Path,
    family: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ResolvedSchema]:
    layout = FAMILY_LAYOUTS[family]

    source_path = find_case_insensitive_file(dataset_dir, (layout.source_filename,))
    target_path = find_case_insensitive_file(dataset_dir, (layout.target_filename,))
    gt_path = find_case_insensitive_file(dataset_dir, layout.gt_filenames)

    if family == "kbwt":
        return load_kbwt_dataset(
            source_path=source_path,
            target_path=target_path,
            gt_path=gt_path,
            layout=layout,
            args=args,
        )

    source_raw = pd.read_csv(source_path, dtype="string")
    target_raw = pd.read_csv(target_path, dtype="string")
    gt_raw = pd.read_csv(gt_path, dtype="string")

    source_id_col, source_id_is_index = resolve_existing_column(
        source_raw,
        args.source_id_col,
        layout.source_id_candidates,
        role="source ID",
        allow_index=True,
    )
    source_value_col, _ = resolve_existing_column(
        source_raw,
        args.source_value_col,
        layout.source_value_candidates,
        role="source join value",
        exclude=set() if source_id_is_index else {source_id_col},
    )

    target_id_col, target_id_is_index = resolve_existing_column(
        target_raw,
        args.target_id_col,
        layout.target_id_candidates,
        role="target ID",
        allow_index=True,
    )
    target_value_col, _ = resolve_existing_column(
        target_raw,
        args.target_value_col,
        layout.target_value_candidates,
        role="target join value",
        exclude=set() if target_id_is_index else {target_id_col},
    )

    gt_source_col, _ = resolve_existing_column(
        gt_raw,
        args.gt_source_col,
        layout.gt_source_candidates,
        role="ground-truth source ID",
    )
    gt_target_col, _ = resolve_existing_column(
        gt_raw,
        args.gt_target_col,
        layout.gt_target_candidates,
        role="ground-truth target ID",
        exclude={gt_source_col},
    )

    # Reset before indexing so the FAISS vector positions and DataFrame index are
    # the same 0..N-1 sequence. Do not reset again after sem_index().
    source_raw = source_raw.reset_index(drop=True)
    target_raw = target_raw.reset_index(drop=True)

    source = pd.DataFrame(
        {
            "source_id": prepare_id_series(
                source_raw, source_id_col, source_id_is_index, role="source ID"
            ),
            "source_value": validate_text_values(
                source_raw[source_value_col], "source", source_value_col
            ),
        }
    )
    target = pd.DataFrame(
        {
            "target_id": prepare_id_series(
                target_raw, target_id_col, target_id_is_index, role="target ID"
            ),
            "target_value": validate_text_values(
                target_raw[target_value_col], "target", target_value_col
            ),
        }
    )

    ground_truth = pd.DataFrame(
        {
            "target_id": gt_raw[gt_target_col].astype("string"),
            "source_id": gt_raw[gt_source_col].astype("string"),
        }
    )

    if source.empty:
        raise DatasetSchemaError("The source/query table is empty.")
    if target.empty:
        raise DatasetSchemaError("The target/indexed table is empty.")
    if ground_truth[["target_id", "source_id"]].isna().any().any():
        raise DatasetSchemaError("The ground truth contains null source or target IDs.")
    if not source["source_id"].is_unique:
        duplicates = source.loc[
            source["source_id"].duplicated(keep=False), "source_id"
        ].head(10).tolist()
        raise DatasetSchemaError(
            f"source_id must be unique. Example duplicate IDs: {duplicates}"
        )
    if not target["target_id"].is_unique:
        duplicates = target.loc[
            target["target_id"].duplicated(keep=False), "target_id"
        ].head(10).tolist()
        raise DatasetSchemaError(
            f"target_id must be unique. Example duplicate IDs: {duplicates}"
        )

    schema = ResolvedSchema(
        source_file=str(source_path),
        target_file=str(target_path),
        ground_truth_file=str(gt_path),
        source_id_col=source_id_col,
        source_value_col=source_value_col,
        target_id_col=target_id_col,
        target_value_col=target_value_col,
        gt_source_col=gt_source_col,
        gt_target_col=gt_target_col,
        source_id_is_index=source_id_is_index,
        target_id_is_index=target_id_is_index,
    )
    return source, target, ground_truth, schema


def evaluation_pair_columns(family: str) -> tuple[str, str]:
    # AutoFJ ground truth has stable row IDs. KBWT ground truth consists of
    # source/target values, and duplicate values make row-ID mapping ambiguous.
    if family == "kbwt":
        return "target_value", "source_value"
    return "target_id", "source_id"


def pair_set(df: pd.DataFrame, family: str) -> set[tuple[str, str]]:
    if df.empty:
        return set()

    target_col, source_col = evaluation_pair_columns(family)
    missing = [col for col in (target_col, source_col) if col not in df.columns]
    if missing:
        raise DatasetSchemaError(
            f"Evaluation columns missing for family {family!r}: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    return set(
        map(
            tuple,
            df[[target_col, source_col]]
            .astype("string")
            .drop_duplicates()
            .to_numpy(),
        )
    )


def _column_string_set(df: pd.DataFrame, column: str) -> set[str]:
    if column not in df.columns or df.empty:
        return set()
    return set(df[column].astype("string").dropna().astype(str))


def evaluate_predictions(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    source: pd.DataFrame,
    family: str,
) -> dict[str, int | float]:
    pred_pairs = pair_set(predictions, family)
    gt_pairs = pair_set(ground_truth, family)

    true_positives = len(pred_pairs & gt_pairs)
    false_positives = len(pred_pairs - gt_pairs)
    false_negatives = len(gt_pairs - pred_pairs)

    precision = true_positives / len(pred_pairs) if pred_pairs else 0.0
    recall = true_positives / len(gt_pairs) if gt_pairs else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    source_ids = _column_string_set(source, "source_id")
    predicted_source_ids = _column_string_set(predictions, "source_id")

    if family == "kbwt":
        gt_source_values = _column_string_set(ground_truth, "source_value")
        matched_mask = source["source_value"].astype(str).isin(gt_source_values)
        gt_matched_source_ids = set(source.loc[matched_mask, "source_id"].astype(str))
    else:
        gt_matched_source_ids = _column_string_set(ground_truth, "source_id")

    gt_unmatched_source_ids = source_ids - gt_matched_source_ids

    return {
        "prediction_pairs": len(pred_pairs),
        "ground_truth_pairs": len(gt_pairs),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "source_coverage": (
            len(predicted_source_ids & source_ids) / len(source_ids)
            if source_ids
            else 0.0
        ),
        "gt_matched_source_rows": len(gt_matched_source_ids & source_ids),
        "gt_unmatched_source_rows": len(gt_unmatched_source_ids),
        "unmatched_source_rows_with_prediction": len(
            gt_unmatched_source_ids & predicted_source_ids
        ),
    }


def run_one_dataset(
    *,
    repo_root: Path,
    family: str,
    dataset_dir: Path,
    args: argparse.Namespace,
    repository_commit: str | None,
) -> RunResult:
    dataset = dataset_dir.name
    output_dir = args.output_root / family / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_file = output_dir / "predictions.csv"
    raw_predictions_file = output_dir / "predictions_raw.csv"
    metrics_file = output_dir / "metrics.json"
    schema_file = output_dir / "resolved_schema.json"
    index_dir = output_dir / "target_index"

    source, target, ground_truth, schema = load_dataset(
        dataset_dir, family, args
    )

    schema_file.write_text(
        json.dumps(asdict(schema), indent=2),
        encoding="utf-8",
    )

    print(f"\n{'=' * 72}")
    print(f"Family:  {family}")
    print(f"Dataset: {dataset}")
    print(f"Source/query rows:   {len(source)}")
    print(f"Target/indexed rows: {len(target)}")
    print(f"Ground-truth pairs:  {len(pair_set(ground_truth, family))}")
    print(
        "Evaluation pairs:    "
        + (
            "(target_value, source_value)"
            if family == "kbwt"
            else "(target_id, source_id)"
        )
    )
    print(f"Source value column: {schema.source_value_col}")
    print(f"Target value column: {schema.target_value_col}")
    print(f"{'=' * 72}")

    if args.dry_run:
        empty_predictions = pd.DataFrame(
            columns=[
                "target_id",
                "target_value",
                "source_id",
                "source_value",
                "score",
            ]
        )
        dry_metrics = evaluate_predictions(
            empty_predictions, ground_truth, source, family
        )
        return RunResult(
            family=family,
            dataset=dataset,
            status="dry_run",
            embedding_model=args.embedding_model,
            lm_model=args.lm_model,
            k=args.k,
            score_threshold=args.score_threshold,
            source_rows=len(source),
            target_rows=len(target),
            raw_prediction_rows=0,
            prediction_pairs=0,
            ground_truth_pairs=int(dry_metrics["ground_truth_pairs"]),
            true_positives=0,
            false_positives=0,
            false_negatives=int(dry_metrics["false_negatives"]),
            precision=0.0,
            recall=0.0,
            f1=0.0,
            source_coverage=0.0,
            gt_matched_source_rows=int(dry_metrics["gt_matched_source_rows"]),
            gt_unmatched_source_rows=int(dry_metrics["gt_unmatched_source_rows"]),
            unmatched_source_rows_with_prediction=int(
                dry_metrics["unmatched_source_rows_with_prediction"]
            ),
            index_seconds=0.0,
            join_seconds=0.0,
            total_seconds=0.0,
            expected_target_embedding_requests=0,
            expected_source_embedding_requests=0,
            expected_total_embedding_requests=0,
            embedded_target_rows=0,
            embedded_source_rows=0,
            expected_llm_calls=0,
            faiss_factory="Flat",
            faiss_metric="METRIC_INNER_PRODUCT",
            score_semantics="larger inner-product score is better; no native threshold",
            random_seed="not exposed",
            repository_commit=repository_commit,
            predictions_file=str(predictions_file),
            raw_predictions_file=str(raw_predictions_file),
            metrics_file=str(metrics_file),
            schema_file=str(schema_file),
        )

    if args.k < 1:
        raise ValueError("--k must be at least 1.")

    # The local FaissVS remaps FAISS -1 sentinel indices through NumPy indexing.
    # K > target size can therefore turn missing neighbors into the final target
    # row instead of filtering them. Refuse that unsafe configuration.
    if args.k > len(target):
        raise ValueError(
            f"K={args.k} exceeds the indexed target size ({len(target)}). "
            "The current local FaissVS implementation does not safely preserve "
            "FAISS -1 sentinel IDs when searching a subset."
        )

    shutil.rmtree(index_dir, ignore_errors=True)

    rm, _vs = configure_lotus(
        args.embedding_model,
        args.lm_model,
        args.batch_size,
    )

    index_start = time.perf_counter()
    indexed_target = target.sem_index("target_value", str(index_dir))
    index_seconds = time.perf_counter() - index_start

    join_start = time.perf_counter()
    joined = source.sem_sim_join(
        indexed_target,
        left_on="source_value",
        right_on="target_value",
        K=args.k,
    )
    join_seconds = time.perf_counter() - join_start

    raw_predictions = (
        joined[
            [
                "target_id",
                "target_value",
                "source_id",
                "source_value",
                "_scores",
            ]
        ]
        .rename(columns={"_scores": "score"})
        .copy()
    )
    raw_predictions["target_id"] = raw_predictions["target_id"].astype("string")
    raw_predictions["source_id"] = raw_predictions["source_id"].astype("string")
    raw_predictions["target_value"] = raw_predictions["target_value"].astype("string")
    raw_predictions["source_value"] = raw_predictions["source_value"].astype("string")

    expected_raw_rows = len(source) * args.k
    if len(raw_predictions) != expected_raw_rows:
        raise RuntimeError(
            "Unexpected sem_sim_join output size: "
            f"expected {expected_raw_rows} rows for {len(source)} queries "
            f"with K={args.k}, but received {len(raw_predictions)}."
        )

    raw_predictions.to_csv(raw_predictions_file, index=False)

    predictions = raw_predictions.copy()
    if args.score_threshold is not None:
        predictions = predictions.loc[
            predictions["score"] >= args.score_threshold
        ].copy()

    predictions.to_csv(predictions_file, index=False)

    metrics = evaluate_predictions(predictions, ground_truth, source, family)

    target_requests = math.ceil(len(target) / rm.max_batch_size)
    source_requests = math.ceil(len(source) / rm.max_batch_size)
    total_seconds = index_seconds + join_seconds

    result = RunResult(
        family=family,
        dataset=dataset,
        status="success",
        embedding_model=args.embedding_model,
        lm_model=args.lm_model,
        k=args.k,
        score_threshold=args.score_threshold,
        source_rows=len(source),
        target_rows=len(target),
        raw_prediction_rows=len(raw_predictions),
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
        index_seconds=index_seconds,
        join_seconds=join_seconds,
        total_seconds=total_seconds,
        expected_target_embedding_requests=target_requests,
        expected_source_embedding_requests=source_requests,
        expected_total_embedding_requests=target_requests + source_requests,
        embedded_target_rows=len(target),
        embedded_source_rows=len(source),
        expected_llm_calls=0,
        faiss_factory="Flat",
        faiss_metric="METRIC_INNER_PRODUCT",
        score_semantics="larger inner-product score is better; no native threshold",
        random_seed="not exposed",
        repository_commit=repository_commit,
        predictions_file=str(predictions_file),
        raw_predictions_file=str(raw_predictions_file),
        metrics_file=str(metrics_file),
        schema_file=str(schema_file),
    )

    metrics_file.write_text(
        json.dumps(asdict(result), indent=2),
        encoding="utf-8",
    )

    print(f"Operator cache:      {lotus.settings.enable_cache}")
    print(f"Raw prediction rows: {len(raw_predictions)}")
    print(f"Evaluated pairs:     {result.prediction_pairs}")
    print(f"Precision:           {result.precision:.6f}")
    print(f"Recall:              {result.recall:.6f}")
    print(f"F1:                  {result.f1:.6f}")
    print(f"Source coverage:     {result.source_coverage:.6f}")
    print(f"Index time:          {result.index_seconds:.4f} s")
    print(f"Join time:           {result.join_seconds:.4f} s")
    print(f"Total time:          {result.total_seconds:.4f} s")
    print(
        "Expected embedding requests: "
        f"{result.expected_total_embedding_requests} "
        f"({target_requests} target + {source_requests} source)"
    )
    print("Expected LLM calls:  0")
    print(f"Predictions:         {predictions_file}")

    return result


def discover_datasets(
    data_root: Path,
    requested: list[str] | None,
    run_all: bool,
) -> list[Path]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset family directory not found: {data_root}")

    available = sorted(
        [path for path in data_root.iterdir() if path.is_dir()],
        key=lambda path: path.name.casefold(),
    )

    if run_all:
        if not available:
            raise FileNotFoundError(f"No dataset directories found under {data_root}")
        return available

    assert requested
    by_name = {path.name.casefold(): path for path in available}
    resolved: list[Path] = []
    missing: list[str] = []

    for name in requested:
        path = by_name.get(name.casefold())
        if path is None:
            missing.append(name)
        else:
            resolved.append(path)

    if missing:
        raise FileNotFoundError(
            f"Datasets not found under {data_root}: {missing}. "
            f"Available datasets: {[path.name for path in available]}"
        )
    return resolved


def write_summary(results: list[RunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(result) for result in results]).to_csv(path, index=False)


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run LOTUS sem_sim_join on VIDA AutoFJ, AutoFJ-overlap, or KBWT "
            "datasets. The source table is the query side and the target table "
            "is semantically indexed."
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
        help=(
            "Dataset directory name. Repeat this option to run multiple datasets."
        ),
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Run every dataset directory in the selected family.",
    )

    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help=(
            "Optional post-retrieval minimum inner-product score. "
            "sem_sim_join itself has no native threshold."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
    )
    parser.add_argument(
        "--lm-model",
        default=DEFAULT_LM_MODEL,
        help=(
            "Configured for LOTUS compatibility. sem_index and sem_sim_join "
            "should not call this chat model."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
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
        default=repo_root / "outputs" / "lotus_sem_sim_join",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve files/columns and validate data without embedding or joining.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue a family batch after a dataset fails.",
    )

    # Column overrides are most useful for KBWT, whose original tables may have
    # multiple columns. Use __index__ for an ID derived from the CSV row index.
    parser.add_argument("--source-id-col", default=None)
    parser.add_argument("--source-value-col", default=None)
    parser.add_argument("--target-id-col", default=None)
    parser.add_argument("--target-value-col", default=None)
    parser.add_argument("--gt-source-col", default=None)
    parser.add_argument("--gt-target-col", default=None)

    return parser


def main() -> int:
    repo_root = find_repo_root(Path(__file__).parent)
    parser = build_parser(repo_root)
    args = parser.parse_args()

    args.output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else (repo_root / args.output_root)
    ).resolve()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")

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
                repo_root=repo_root,
                family=args.family,
                dataset_dir=dataset_dir,
                args=args,
                repository_commit=repository_commit,
            )
            results.append(result)
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            print(f"\nERROR [{args.family}/{dataset_dir.name}]: {error_message}", file=sys.stderr)
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
