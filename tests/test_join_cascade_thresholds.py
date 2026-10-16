"""Offline tests for join-cascade threshold classification (no paid calls).

Regression coverage for the equal-threshold boundary: with
``pos_threshold == neg_threshold``, a helper score exactly at the threshold must
be classified as positive (and only positive), so that

    helper_positive + helper_negative + uncertain == total pairs

always holds. Separated thresholds must behave exactly as the historical
``score >= pos`` / ``score <= neg`` rules.
"""

import numpy as np
import pandas as pd

from lotus.sem_ops.cascade_utils import learn_cascade_thresholds
from lotus.sem_ops.sem_join import _classify_join_scores
from lotus.types import CascadeArgs


def _invariants(scores, pos, neg):
    """Check the partition + mutual-exclusivity invariants of the classifier."""
    high_mask, neg_count, low_mask = _classify_join_scores(scores, pos, neg)
    high = scores[high_mask]
    neg_vals = scores[(scores < pos) & (scores <= neg)]
    low = scores[low_mask]
    assert neg_count == len(neg_vals)
    assert set(high.index).isdisjoint(neg_vals.index)
    assert set(high.index).isdisjoint(low.index)
    assert set(neg_vals.index).isdisjoint(low.index)
    assert len(high) + len(neg_vals) + len(low) == len(scores)
    return high, neg_vals, low


def test_equal_threshold_no_double_count():
    # Scores containing 0.5 / 0.9 / 1.0 with pos == neg == 1.0.
    s = pd.Series([0.5, 0.9, 1.0, 1.0, 0.7])
    high, neg_vals, low = _invariants(s, 1.0, 1.0)
    assert len(high) == 2          # the two 1.0 pairs are positive
    assert len(neg_vals) == 3      # 0.5, 0.9, 0.7; the 1.0 pairs are NOT negative
    assert len(low) == 0
    assert len(high) + len(neg_vals) + len(low) == len(s)


def test_equal_threshold_partitions_realistic_join():
    # Reproduce the TennisTournament shape: one pair at exactly 1.0, 8747 below.
    scores = [1.0] + [0.7] * 27 + [0.5] * (8748 - 28)
    s = pd.Series(scores)
    high, neg_vals, low = _invariants(s, 1.0, 1.0)
    assert len(high) == 1
    assert len(neg_vals) == 8747
    assert len(low) == 0
    # helper_resolved must not exceed the Cartesian pair count
    assert len(high) + len(neg_vals) == len(s) == 8748


def test_separated_thresholds_preserve_native_rule():
    s = pd.Series([0.2, 0.4, 0.45, 0.6, 0.9, 1.0])
    high, neg_vals, low = _invariants(s, 0.9, 0.4)
    # Historical rules when pos > neg:
    assert len(neg_vals) == (s <= 0.4).sum()          # 0.2, 0.4
    assert len(high) == (s >= 0.9).sum()              # 0.9, 1.0
    assert len(low) == ((s > 0.4) & (s < 0.9)).sum()  # 0.45, 0.6


def test_partition_holds_across_threshold_pairs():
    s = pd.Series(np.linspace(0.1, 1.0, 19))
    for pos, neg in [(1.0, 1.0), (0.9, 0.9), (0.95, 0.3), (0.8, 0.5), (1.0, 0.0)]:
        _invariants(s, pos, neg)


def test_plan_cost_equals_uncertain_count():
    # The optimizer's predicted final LLM cost is the number of uncertain pairs.
    s = pd.Series([0.5, 0.9, 1.0, 0.6, 0.95])
    high_mask, neg_count, low_mask = _classify_join_scores(s, 0.9, 0.4)
    assert low_mask.sum() == len(s) - high_mask.sum() - neg_count


def test_learning_can_produce_equal_thresholds_on_degenerate_sample():
    # Documents native learning: an all-True sample at score 1.0 yields
    # (1.0, 1.0); the classifier must still partition the pair set.
    args = CascadeArgs(
        sampling_percentage=0.1, cascade_IS_random_seed=0, cascade_IS_max_sample_range=200
    )
    proxy = [1.0] * 47
    oracle = [True] * 47
    correction = np.full(47, 1.0 / 8748.0)
    (pos, neg), _ = learn_cascade_thresholds(proxy, oracle, correction, args)
    assert (pos, neg) == (1.0, 1.0)
    s = pd.Series([1.0, 0.5, 0.9])
    high_mask, neg_count, low_mask = _classify_join_scores(s, pos, neg)
    assert high_mask.sum() + neg_count + low_mask.sum() == len(s)
    assert high_mask.sum() == 1  # only the exact-1.0 pair is positive
