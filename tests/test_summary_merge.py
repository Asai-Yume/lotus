"""Offline regression tests for merge-aware family summary writing.

A family may be split across multiple runner invocations (e.g. the SS batch
runs 104 datasets first and then 4 datasets with explicit GT columns). Each
invocation calls ``write_summary`` with its own rows; the writer must merge
with any existing summary file so the final family summary contains one row
per completed dataset instead of only the last invocation's rows.

Also verifies that zero-accepted-pair datasets still produce summary rows.
No network or API calls are made.
"""

import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from run_sem_sim_join import write_summary  # noqa: E402


def make_work_dir(prefix: str) -> Path:
    base = _REPO_ROOT / "outputs" / "_retry_test_work"
    base.mkdir(parents=True, exist_ok=True)
    d = base / f"{prefix}_{uuid.uuid4().hex[:8]}"
    os.makedirs(d, exist_ok=True)
    return d


def _row(family: str, dataset: str, status: str = "success", accepted_pairs: int = 1) -> dict:
    return {
        "family": family,
        "dataset": dataset,
        "status": status,
        "source_rows": 10,
        "retrieved_candidates": 10,
        "logical_verification_calls": 10,
        "accepted_pairs": accepted_pairs,
        "rejected_candidates": 10 - accepted_pairs,
        "precision": 1.0 if accepted_pairs else 0.0,
        "recall": 1.0 if accepted_pairs else 0.0,
        "f1": 1.0 if accepted_pairs else 0.0,
        "source_coverage": 1.0 if accepted_pairs else 0.0,
    }


@dataclass
class _Result:
    """Minimal stand-in for the runner result dataclasses."""

    family: str
    dataset: str
    status: str
    source_rows: int
    retrieved_candidates: int
    logical_verification_calls: int
    accepted_pairs: int
    rejected_candidates: int
    precision: float
    recall: float
    f1: float
    source_coverage: float


def _results(rows: list[dict]):
    return [_Result(**row) for row in rows]


def _load_summary(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def test_invocation_a_plus_b_combines_into_full_family_summary():
    work = make_work_dir("summary_merge")
    summary_path = work / "ss_summary.csv"

    # Invocation A: 104 normal datasets.
    inv_a = _results([_row("ss", f"FF-dataset-{i:03d}") for i in range(104)])
    write_summary(inv_a, summary_path)

    # Invocation B: 4 datasets with explicit GT overrides.
    inv_b = _results([_row("ss", name) for name in ("FF-name-combine", "FF-name-combine-long")])
    write_summary(inv_b, summary_path)

    combined = _load_summary(summary_path)
    assert len(combined) == 106
    assert combined["dataset"].nunique() == 106
    assert combined["dataset"].duplicated().sum() == 0
    assert {"FF-dataset-000", "FF-dataset-103", "FF-name-combine", "FF-name-combine-long"} <= set(combined["dataset"])


def test_rerunning_one_dataset_updates_instead_of_duplicating():
    work = make_work_dir("summary_rerun")
    summary_path = work / "ss_summary.csv"

    inv_a = _results([_row("ss", "FF-bikes"), _row("ss", "FF-bikes-long")])
    write_summary(inv_a, summary_path)

    # Deliberately rerun FF-bikes with a different outcome; it must replace
    # the old row, not add a duplicate.
    rerun = _results([_row("ss", "FF-bikes", accepted_pairs=0)])
    write_summary(rerun, summary_path)

    combined = _load_summary(summary_path)
    assert len(combined) == 2
    assert combined["dataset"].duplicated().sum() == 0
    bikes = combined[combined["dataset"] == "FF-bikes"].iloc[0]
    assert bikes["accepted_pairs"] == 0  # newest result wins
    assert bikes["precision"] == 0.0
    longs = combined[combined["dataset"] == "FF-bikes-long"].iloc[0]
    assert longs["accepted_pairs"] == 1  # untouched by the rerun


def test_zero_accepted_pairs_does_not_suppress_summary_row():
    work = make_work_dir("summary_zero")
    summary_path = work / "ss_summary.csv"

    zero = _results([_row("ss", "FF-bikes", accepted_pairs=0)])
    write_summary(zero, summary_path)

    combined = _load_summary(summary_path)
    assert len(combined) == 1
    row = combined.iloc[0]
    assert row["dataset"] == "FF-bikes"
    assert row["status"] == "success"
    assert row["accepted_pairs"] == 0
    assert row["rejected_candidates"] == 10
    assert row["precision"] == 0.0
    assert row["recall"] == 0.0
    assert row["f1"] == 0.0
    assert row["source_coverage"] == 0.0


def test_merge_is_idempotent_across_three_invocations():
    work = make_work_dir("summary_three")
    summary_path = work / "ss_summary.csv"

    write_summary(_results([_row("ss", f"D{i}") for i in range(30)]), summary_path)
    write_summary(_results([_row("ss", f"D{i}", accepted_pairs=0) for i in range(30, 60)]), summary_path)
    write_summary(_results([_row("ss", "D15", accepted_pairs=0)]), summary_path)

    combined = _load_summary(summary_path)
    assert len(combined) == 60
    assert combined["dataset"].duplicated().sum() == 0
    # D15 was rerun last: newest (zero) result wins.
    d15 = combined[combined["dataset"] == "D15"].iloc[0]
    assert d15["accepted_pairs"] == 0


def test_write_summary_with_empty_results_is_noop():
    work = make_work_dir("summary_empty")
    summary_path = work / "ss_summary.csv"

    write_summary(_results([_row("ss", "D1")]), summary_path)
    before = _load_summary(summary_path)
    write_summary([], summary_path)
    after = _load_summary(summary_path)
    pd.testing.assert_frame_equal(before, after)
