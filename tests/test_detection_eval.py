"""
tests/test_detection_eval.py
----------------------------
Tests for Route R0, the detection evaluation harness.

TESTING STANCE (CLAUDE.md §8b.5: "a test that cannot fail is not a test")
=========================================================================
The primary metric of the whole research program is average precision. If the
only check on it were "call ``pr_auc`` and see that it returns a float", a
wrong implementation would pass. So:

- :func:`_ap_reference` re-derives average precision **from the definition**
  in plain Python — no numpy, no sklearn — and every AP test asserts the two
  agree. The inputs include adversarial cases (all ties, two-valued scores,
  duplicated extremes) because ties are where AP implementations differ.
- Label counts are asserted against the three totals in
  ``docs/cost_semantics.md`` (631 / 7,093 / 1,701). These are hard-coded
  on purpose: if a change moves them, the test must fail loudly, because the
  totals are quoted in the cost-semantics document and in results tables.
- The leakage test uses a spy detector that records every cell its train mask
  offered and fails if an evaluation cell appears.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

from src.data.loader import DataLoader
from src.detection_eval.harness import (
    DetectorProtocol,
    RuleDetectorAdapter,
    evaluate_detector,
    flatten_cells,
    stratified_cell_splits,
)
from src.detection_eval.labels import cell_labels, label_stats
from src.detection_eval.leaderboard import (
    MISSING,
    load_store,
    record,
    render_markdown,
)
from src.detection_eval.metrics import (
    DetectionMetrics,
    best_f1,
    evaluate_scores,
    f1_at_threshold,
    lift_at_k,
    pr_auc,
    precision_at_k,
    recall_at_k,
    roc_auc,
)
from src.evaluation.repair_metrics import ValueComparator

DATA_ROOT = "datasets"

#: docs/cost_semantics.md, §"Total genuine corruptions per dataset".
EXPECTED_CORRUPTIONS = {
    "breast_cancer": 631,
    "smartfactory": 7093,
    "adult": 1701,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Independent reference implementations (deliberately naive and slow)
# ═══════════════════════════════════════════════════════════════════════════════


def _ap_reference(y: Sequence[int], s: Sequence[float]) -> float:
    """
    Average precision from the definition, in pure Python.

        AP = sum_n (R_n - R_{n-1}) * P_n

    taken over the distinct score values in descending order, where the
    classifier at step n is "flag iff score >= t_n". Tied scores share a
    single threshold, so ties cannot be exploited — the property that makes
    this the correct reference for a heavily-tied rule detector.

    No numpy, no sklearn: if this agrees with
    :func:`src.detection_eval.metrics.pr_auc`, two independent code paths
    agree.
    """
    n_pos = sum(1 for v in y if v)
    if n_pos == 0:
        raise ValueError("AP undefined with no positives")
    thresholds = sorted(set(float(v) for v in s), reverse=True)
    ap = 0.0
    prev_recall = 0.0
    for t in thresholds:
        tp = sum(1 for yi, si in zip(y, s) if si >= t and yi)
        flagged = sum(1 for si in s if si >= t)
        precision = tp / flagged if flagged else 0.0
        recall = tp / n_pos
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def _best_f1_reference(y: Sequence[int], s: Sequence[float]) -> float:
    """Brute-force best F1 over every distinct threshold, in pure Python."""
    n_pos = sum(1 for v in y if v)
    best = 0.0
    for t in sorted(set(float(v) for v in s)):
        tp = sum(1 for yi, si in zip(y, s) if si >= t and yi)
        flagged = sum(1 for si in s if si >= t)
        if tp == 0:
            continue
        best = max(best, 2.0 * tp / (flagged + n_pos))
    return best


def _expected_precision_at_k_reference(
    y: Sequence[int], s: Sequence[float], k: int
) -> float:
    """
    Expected precision@k under random tie-breaking, by direct enumeration of
    tie groups in pure Python.
    """
    groups: dict = {}
    for yi, si in zip(y, s):
        g = groups.setdefault(float(si), [0, 0])
        g[0] += 1
        g[1] += 1 if yi else 0
    remaining = k
    tp = 0.0
    for score in sorted(groups, reverse=True):
        size, pos = groups[score]
        if remaining <= 0:
            break
        if remaining >= size:
            tp += pos
            remaining -= size
        else:
            tp += pos * remaining / size
            remaining = 0
    return tp / k


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def bc_pair() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return DataLoader(DATA_ROOT).load("breast_cancer")


@pytest.fixture(scope="module")
def bc_labels(bc_pair) -> pd.DataFrame:
    clean_df, dirty_df = bc_pair
    return cell_labels(clean_df, dirty_df)


def _write_tiny_dataset(root: Path, name: str = "tiny") -> Path:
    """
    A 40-row, 3-column clean/dirty pair with 6 injected corruptions, written
    so ``evaluate_detector`` can load it through the real ``DataLoader``.
    """
    n = 40
    clean = pd.DataFrame(
        {
            "num": [float(i % 10) + 1.0 for i in range(n)],
            "cat": ["a" if i % 2 else "b" for i in range(n)],
            "idn": list(range(n)),
        }
    )
    dirty = clean.copy()
    dirty.loc[0, "num"] = np.nan          # MISSING
    dirty.loc[1, "num"] = 9999.0          # OUT_OF_RANGE
    dirty.loc[2, "cat"] = "zzz"           # NOVEL_CATEGORY
    dirty.loc[3, "cat"] = " a"            # WHITESPACE
    dirty.loc[4, "num"] = np.nan
    dirty.loc[5, "cat"] = "qqq"

    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    clean.to_csv(d / "clean.csv", index=False)
    dirty.to_csv(d / "dirty.csv", index=False)
    return d


# ═══════════════════════════════════════════════════════════════════════════════
# labels.py
# ═══════════════════════════════════════════════════════════════════════════════


class TestCellLabels:
    """Ground-truth corruption labels."""

    @pytest.mark.parametrize("dataset", sorted(EXPECTED_CORRUPTIONS))
    def test_corruption_count_matches_documented_total(self, dataset: str) -> None:
        """
        Counts must equal docs/cost_semantics.md exactly.

        Hard-coded deliberately. A mismatch means either the label definition
        changed or the data did; both must be reported, not absorbed.
        """
        clean_df, dirty_df = DataLoader(DATA_ROOT).load(dataset)
        labels = cell_labels(clean_df, dirty_df)
        assert int(labels.to_numpy().sum()) == EXPECTED_CORRUPTIONS[dataset]

    def test_shape_index_and_columns_preserved(self, bc_pair, bc_labels) -> None:
        _, dirty_df = bc_pair
        assert bc_labels.shape == dirty_df.shape
        assert list(bc_labels.columns) == list(dirty_df.columns)
        assert bc_labels.index.equals(dirty_df.index)
        assert bc_labels.to_numpy().dtype == np.bool_

    def test_nan_equals_nan(self) -> None:
        """A cell blank in both frames is not corrupted."""
        clean = pd.DataFrame({"a": [np.nan, 1.0, np.nan]})
        dirty = pd.DataFrame({"a": [np.nan, 1.0, 2.0]})
        labels = cell_labels(clean, dirty)
        assert labels["a"].tolist() == [False, False, True]

    def test_nan_versus_value_is_corrupted_in_both_directions(self) -> None:
        clean = pd.DataFrame({"a": [1.0, np.nan]})
        dirty = pd.DataFrame({"a": [np.nan, 1.0]})
        assert cell_labels(clean, dirty)["a"].tolist() == [True, True]

    def test_int_versus_float_of_same_value_is_equal(self) -> None:
        """
        371 vs 371.0 must be equal: NaN injection upcasts int64 to float64
        (src/data/loader.py), and a naive != would label whole columns
        corrupted.
        """
        clean = pd.DataFrame({"a": pd.Series([371, 2, 3], dtype="int64")})
        dirty = pd.DataFrame({"a": pd.Series([371.0, 2.0, 4.0], dtype="float64")})
        assert cell_labels(clean, dirty)["a"].tolist() == [False, False, True]

    def test_string_whitespace_is_corrupted_by_default_comparator(self) -> None:
        """
        ValueComparator strips before comparing, so ' a' == 'a'. That is the
        canonical definition and this test pins it, so a future change to the
        comparator is visible here rather than silently shifting every count.
        """
        clean = pd.DataFrame({"a": ["a", "b"]})
        dirty = pd.DataFrame({"a": [" a", "B"]})
        assert cell_labels(clean, dirty)["a"].tolist() == [False, True]

    def test_custom_comparator_is_honoured(self) -> None:
        clean = pd.DataFrame({"a": [1.0, 10.0]})
        dirty = pd.DataFrame({"a": [1.4, 20.0]})
        strict = cell_labels(clean, dirty)
        loose = cell_labels(clean, dirty, comparator=ValueComparator(abs_tol=0.5))
        assert strict["a"].tolist() == [True, True]
        assert loose["a"].tolist() == [False, True]

    def test_fast_path_matches_full_comparator_loop(self) -> None:
        """
        The vectorised pre-filter must be a pure speed-up.

        This frame exercises both branches: raw-equal cells (skipped),
        raw-unequal-but-comparator-equal cells (371 vs 371.0, ' a' vs 'a'),
        both-NaN cells, and genuinely different cells. The expectation is
        recomputed by looping the comparator over every cell, which is the
        semantics the module claims to implement.
        """
        clean = pd.DataFrame(
            {
                "m": [1, 2, np.nan, 4, 5, 6],
                "s": ["a", "b", "c", "d", np.nan, "f"],
            }
        )
        dirty = pd.DataFrame(
            {
                "m": [1.0, 99.0, np.nan, 4, np.nan, 6.0000000001],
                "s": [" a", "B", "c", "zzz", np.nan, "f"],
            }
        )
        comparator = ValueComparator()
        expected = pd.DataFrame(
            {
                col: [
                    not comparator.are_equal(dirty[col].to_numpy()[i], clean[col].to_numpy()[i])
                    for i in range(len(dirty))
                ]
                for col in dirty.columns
            }
        )
        got = cell_labels(clean, dirty)
        assert got["m"].tolist() == expected["m"].tolist()
        assert got["s"].tolist() == expected["s"].tolist()

    def test_row_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Row count mismatch"):
            cell_labels(pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1, 2]}))

    def test_missing_column_raises(self) -> None:
        with pytest.raises(ValueError, match="absent from clean_df"):
            cell_labels(pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]}))

    def test_empty_frame(self) -> None:
        empty = pd.DataFrame({"a": pd.Series(dtype=float)})
        labels = cell_labels(empty, empty)
        assert labels.shape == (0, 1)
        stats = label_stats(labels)
        assert stats["n_cells"] == 0
        assert stats["n_corrupted"] == 0
        assert stats["prevalence"] == 0.0


class TestLabelStats:
    def test_totals_and_prevalence(self, bc_labels) -> None:
        stats = label_stats(bc_labels)
        assert stats["n_cells"] == 699 * 12
        assert stats["n_corrupted"] == 631
        assert stats["prevalence"] == pytest.approx(631 / (699 * 12))
        assert stats["n_rows"] == 699
        assert stats["n_columns"] == 12

    def test_per_column_covers_every_column_and_sums_to_total(self, bc_labels) -> None:
        stats = label_stats(bc_labels)
        assert set(stats["per_column"]) == {str(c) for c in bc_labels.columns}
        assert sum(stats["per_column"].values()) == stats["n_corrupted"]

    def test_per_column_prevalence_is_per_row_rate(self, bc_labels) -> None:
        stats = label_stats(bc_labels)
        for col, count in stats["per_column"].items():
            assert stats["per_column_prevalence"][col] == pytest.approx(count / 699)


# ═══════════════════════════════════════════════════════════════════════════════
# metrics.py — average precision against an independent implementation
# ═══════════════════════════════════════════════════════════════════════════════


class TestAveragePrecisionAgreement:
    """pr_auc must agree with the textbook definition, including under ties."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
    def test_agrees_on_random_input(self, seed: int) -> None:
        rng = random.Random(seed)
        n = 200
        y = [1 if rng.random() < 0.2 else 0 for _ in range(n)]
        if sum(y) == 0:
            y[0] = 1
        s = [rng.random() for _ in range(n)]
        assert pr_auc(y, s) == pytest.approx(_ap_reference(y, s), abs=1e-12)

    @pytest.mark.parametrize("seed", [10, 11, 12, 13])
    def test_agrees_when_scores_are_correlated_with_labels(self, seed: int) -> None:
        rng = random.Random(seed)
        n = 300
        y = [1 if rng.random() < 0.15 else 0 for _ in range(n)]
        if sum(y) == 0:
            y[0] = 1
        s = [rng.random() + (0.6 if yi else 0.0) for yi in y]
        assert pr_auc(y, s) == pytest.approx(_ap_reference(y, s), abs=1e-12)

    @pytest.mark.parametrize("seed", [20, 21, 22])
    def test_agrees_on_heavily_tied_scores(self, seed: int) -> None:
        """
        The adversarial case, and the one that matters: the rule detector's
        scores are a handful of constants over hundreds of thousands of cells.
        """
        rng = random.Random(seed)
        n = 400
        y = [1 if rng.random() < 0.1 else 0 for _ in range(n)]
        if sum(y) == 0:
            y[0] = 1
        s = [rng.choice([0.0, 0.0, 0.0, 0.0, 0.3, 0.7, 0.9, 1.0]) for _ in range(n)]
        assert pr_auc(y, s) == pytest.approx(_ap_reference(y, s), abs=1e-12)

    def test_agrees_when_every_score_is_identical(self) -> None:
        y = [0, 1, 0, 0, 1, 0, 0, 0]
        s = [0.5] * 8
        assert pr_auc(y, s) == pytest.approx(_ap_reference(y, s), abs=1e-12)

    def test_agrees_on_two_valued_scores(self) -> None:
        y = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
        s = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert pr_auc(y, s) == pytest.approx(_ap_reference(y, s), abs=1e-12)

    def test_agrees_on_duplicated_extremes(self) -> None:
        y = [1, 0, 1, 0, 1, 0]
        s = [1.0, 1.0, 0.0, 0.0, 0.5, 0.5]
        assert pr_auc(y, s) == pytest.approx(_ap_reference(y, s), abs=1e-12)

    def test_agrees_on_a_real_detector_output(self, bc_pair, bc_labels) -> None:
        """
        End-to-end agreement on the actual rule-detector scores for
        breast_cancer — 8,388 cells, 5 distinct scores, 95 % of them tied.
        """
        clean_df, dirty_df = bc_pair
        from src.data.profiler import profile_dataset

        loader = DataLoader(DATA_ROOT)
        profile = profile_dataset(
            "breast_cancer", loader.discover("breast_cancer"), clean_df, dirty_df
        )
        scores = RuleDetectorAdapter().score_cells(dirty_df, profile)
        y = flatten_cells(bc_labels).astype(int).tolist()
        s = flatten_cells(scores).astype(float).tolist()
        assert pr_auc(y, s) == pytest.approx(_ap_reference(y, s), abs=1e-10)


class TestPrAuc:
    def test_perfect_ranking_is_one(self) -> None:
        y = [0, 0, 0, 1, 1]
        s = [0.1, 0.2, 0.3, 0.9, 1.0]
        assert pr_auc(y, s) == pytest.approx(1.0)

    def test_random_scores_approximate_prevalence(self) -> None:
        """
        A random scorer has AP ≈ prevalence. This is the fact that makes
        PR-AUC readable: every reported number must be compared against the
        prevalence beside it.
        """
        rng = np.random.default_rng(1234)
        n = 40000
        prevalence = 0.05
        y = (rng.random(n) < prevalence).astype(int)
        s = rng.random(n)
        assert pr_auc(y, s) == pytest.approx(float(y.mean()), abs=0.01)

    def test_constant_scores_equal_prevalence_exactly(self) -> None:
        """
        With one threshold the curve is a single point (precision=prevalence,
        recall=1), so AP = prevalence exactly, not approximately.
        """
        y = [1] * 7 + [0] * 93
        s = [0.42] * 100
        assert pr_auc(y, s) == pytest.approx(0.07, abs=1e-12)

    def test_all_negative_returns_none_not_zero(self) -> None:
        """
        0.0 would read as a measured score in a results table (CLAUDE.md
        §5.4). Undefined must be None.
        """
        result = pr_auc([0, 0, 0], [0.1, 0.5, 0.9])
        assert result is None
        assert result is not 0.0  # noqa: F632 - the point is it is not a number

    def test_all_positive_is_one(self) -> None:
        assert pr_auc([1, 1, 1], [0.1, 0.5, 0.9]) == pytest.approx(1.0)

    def test_empty_returns_none(self) -> None:
        assert pr_auc([], []) is None

    def test_nan_scores_raise(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            pr_auc([0, 1], [0.5, float("nan")])

    def test_inf_scores_raise(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            pr_auc([0, 1], [0.5, float("inf")])

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            pr_auc([0, 1, 1], [0.5, 0.6])

    def test_non_binary_labels_raise(self) -> None:
        with pytest.raises(ValueError, match="binary"):
            pr_auc([0, 2], [0.5, 0.6])

    def test_monotone_rescaling_does_not_change_pr_auc(self) -> None:
        """AP is rank-based: any strictly increasing transform is invisible."""
        rng = np.random.default_rng(7)
        y = (rng.random(500) < 0.2).astype(int)
        s = rng.random(500)
        assert pr_auc(y, s) == pytest.approx(pr_auc(y, 3.0 * s + 11.0), abs=1e-12)


class TestRocAuc:
    def test_perfect_ranking_is_one(self) -> None:
        assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)

    def test_constant_scores_is_half(self) -> None:
        assert roc_auc([0, 1, 0, 1], [0.5] * 4) == pytest.approx(0.5)

    def test_reversed_ranking_is_zero(self) -> None:
        assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)

    def test_single_class_returns_none(self) -> None:
        assert roc_auc([0, 0, 0], [0.1, 0.2, 0.3]) is None
        assert roc_auc([1, 1, 1], [0.1, 0.2, 0.3]) is None

    def test_empty_returns_none(self) -> None:
        assert roc_auc([], []) is None

    def test_matches_mann_whitney_definition(self) -> None:
        """
        ROC-AUC = P(score of a random positive > score of a random negative),
        ties counting a half. Re-derived here by direct enumeration.
        """
        y = [1, 0, 1, 0, 0, 1]
        s = [0.9, 0.9, 0.4, 0.1, 0.4, 0.2]
        pos = [si for yi, si in zip(y, s) if yi]
        neg = [si for yi, si in zip(y, s) if not yi]
        wins = sum(
            1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg
        )
        assert roc_auc(y, s) == pytest.approx(wins / (len(pos) * len(neg)))


class TestF1:
    def test_f1_at_threshold_matches_hand_computation(self) -> None:
        y = [1, 1, 0, 0]
        s = [0.9, 0.4, 0.8, 0.1]
        # threshold 0.5 flags {0.9, 0.8}: tp=1, fp=1, fn=1
        # precision 0.5, recall 0.5, f1 0.5
        assert f1_at_threshold(y, s, 0.5) == pytest.approx(0.5)

    def test_threshold_is_inclusive(self) -> None:
        """`score >= threshold`, not `>`. Pinned because it shifts every F1."""
        assert f1_at_threshold([1, 0], [0.5, 0.1], 0.5) == pytest.approx(1.0)
        assert f1_at_threshold([1, 0], [0.5, 0.1], 0.51) == pytest.approx(0.0)

    def test_f1_at_threshold_none_without_positives(self) -> None:
        assert f1_at_threshold([0, 0], [0.1, 0.9], 0.5) is None

    def test_f1_zero_when_nothing_flagged_is_a_real_measurement(self) -> None:
        """Flagging nothing is a measured failure (0.0), not undefined (None)."""
        assert f1_at_threshold([1, 0], [0.1, 0.1], 0.9) == 0.0

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_best_f1_matches_brute_force(self, seed: int) -> None:
        rng = random.Random(seed)
        n = 120
        y = [1 if rng.random() < 0.25 else 0 for _ in range(n)]
        if sum(y) == 0:
            y[0] = 1
        s = [rng.choice([0.0, 0.3, 0.5, 0.7, 1.0]) for _ in range(n)]
        got, thr = best_f1(y, s)
        assert got == pytest.approx(_best_f1_reference(y, s), abs=1e-12)
        assert f1_at_threshold(y, s, thr) == pytest.approx(got, abs=1e-12)

    def test_best_f1_perfect_separation(self) -> None:
        f1, thr = best_f1([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        assert f1 == pytest.approx(1.0)
        assert thr == pytest.approx(0.8)

    def test_best_f1_none_without_positives(self) -> None:
        assert best_f1([0, 0], [0.1, 0.9]) == (None, None)

    def test_best_f1_empty(self) -> None:
        assert best_f1([], []) == (None, None)

    def test_best_f1_is_at_least_flag_everything(self) -> None:
        """
        Flagging every cell is always available, so best-F1 can never fall
        below its F1. Catches a sweep that misses the lowest threshold.
        """
        rng = random.Random(99)
        y = [1 if rng.random() < 0.3 else 0 for _ in range(80)]
        y[0] = 1
        s = [rng.random() for _ in range(80)]
        n_pos = sum(y)
        flag_all_f1 = 2.0 * n_pos / (len(y) + n_pos)
        got, _ = best_f1(y, s)
        assert got >= flag_all_f1 - 1e-12


class TestAtK:
    def test_precision_at_k_without_ties(self) -> None:
        y = [1, 0, 1, 0, 0]
        s = [0.9, 0.8, 0.7, 0.2, 0.1]
        assert precision_at_k(y, s, 1) == pytest.approx(1.0)
        assert precision_at_k(y, s, 2) == pytest.approx(0.5)
        assert precision_at_k(y, s, 3) == pytest.approx(2 / 3)
        assert precision_at_k(y, s, 5) == pytest.approx(0.4)

    def test_precision_at_k_uses_expected_tie_breaking(self) -> None:
        """
        Four cells tied at 0.5, two positive, k=2 → expectation 2*(2/4)/2 = 0.5.
        Index-order tie-breaking would give 1.0 or 0.0 depending on layout,
        which is exactly the artefact this avoids.
        """
        y = [1, 1, 0, 0]
        s = [0.5, 0.5, 0.5, 0.5]
        assert precision_at_k(y, s, 2) == pytest.approx(0.5)
        assert precision_at_k(y, s, 4) == pytest.approx(0.5)

    def test_precision_at_k_is_order_invariant_under_ties(self) -> None:
        """
        Permuting the arrays must not move the number. This is the property
        that matters on real data, where cells are enumerated column-major
        and index order correlates with the label.
        """
        y = [1, 1, 1, 0, 0, 0, 0, 0]
        s = [0.5] * 8
        rng = random.Random(5)
        base = precision_at_k(y, s, 3)
        for _ in range(10):
            idx = list(range(8))
            rng.shuffle(idx)
            assert precision_at_k([y[i] for i in idx], [s[i] for i in idx], 3) == (
                pytest.approx(base)
            )

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_precision_at_k_matches_reference(self, seed: int) -> None:
        rng = random.Random(seed)
        n = 150
        y = [1 if rng.random() < 0.2 else 0 for _ in range(n)]
        s = [rng.choice([0.0, 0.25, 0.5, 1.0]) for _ in range(n)]
        for k in (1, 5, 37, 150):
            assert precision_at_k(y, s, k) == pytest.approx(
                _expected_precision_at_k_reference(y, s, k), abs=1e-12
            )

    def test_k_zero_is_none(self) -> None:
        assert precision_at_k([1, 0], [0.9, 0.1], 0) is None
        assert recall_at_k([1, 0], [0.9, 0.1], 0) is None
        assert lift_at_k([1, 0], [0.9, 0.1], 0) is None

    def test_k_larger_than_n_clamps_to_n(self) -> None:
        y = [1, 0, 0, 0]
        s = [0.9, 0.3, 0.2, 0.1]
        assert precision_at_k(y, s, 999) == pytest.approx(0.25)
        assert recall_at_k(y, s, 999) == pytest.approx(1.0)

    def test_negative_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 0"):
            precision_at_k([1, 0], [0.9, 0.1], -1)

    def test_non_int_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be an int"):
            precision_at_k([1, 0], [0.9, 0.1], 1.5)

    def test_recall_at_k_none_without_positives(self) -> None:
        assert recall_at_k([0, 0], [0.9, 0.1], 1) is None

    def test_recall_at_k_at_n_is_one(self) -> None:
        y = [1, 0, 1, 0]
        s = [0.1, 0.2, 0.3, 0.4]
        assert recall_at_k(y, s, 4) == pytest.approx(1.0)

    def test_lift_is_precision_over_prevalence(self) -> None:
        y = [1] + [0] * 99
        s = [1.0] + [0.0] * 99
        assert precision_at_k(y, s, 1) == pytest.approx(1.0)
        assert lift_at_k(y, s, 1) == pytest.approx(1.0 / 0.01)

    def test_lift_none_when_no_positives(self) -> None:
        assert lift_at_k([0, 0], [0.9, 0.1], 1) is None

    def test_precision_at_k_empty_input(self) -> None:
        """k clamps to 0 on an empty input, so the answer is undefined."""
        assert precision_at_k([], [], 3) is None


class TestEvaluateScores:
    def test_bundle_matches_individual_functions(self) -> None:
        rng = np.random.default_rng(3)
        y = (rng.random(600) < 0.1).astype(int)
        s = rng.random(600) + 0.4 * y
        m = evaluate_scores(y, s)
        assert m.n == 600
        assert m.n_pos == int(y.sum())
        assert m.prevalence == pytest.approx(float(y.mean()))
        assert m.k == int(y.sum())
        assert m.pr_auc == pytest.approx(pr_auc(y, s))
        assert m.roc_auc == pytest.approx(roc_auc(y, s))
        assert m.best_f1 == pytest.approx(best_f1(y, s)[0])
        assert m.precision_at_k == pytest.approx(precision_at_k(y, s, m.k))
        assert m.recall_at_k == pytest.approx(recall_at_k(y, s, m.k))
        assert m.lift_at_k == pytest.approx(lift_at_k(y, s, m.k))

    def test_k_defaults_to_n_pos_making_precision_equal_recall(self) -> None:
        """With k = n_pos, precision@k == recall@k identically."""
        rng = np.random.default_rng(11)
        y = (rng.random(400) < 0.2).astype(int)
        s = rng.random(400)
        m = evaluate_scores(y, s)
        assert m.precision_at_k == pytest.approx(m.recall_at_k)

    def test_explicit_k_overrides(self) -> None:
        y = [1, 0, 0, 0]
        s = [0.9, 0.3, 0.2, 0.1]
        assert evaluate_scores(y, s, k=2).k == 2

    def test_all_negative_bundle_is_all_none_not_zero(self) -> None:
        m = evaluate_scores([0, 0, 0], [0.1, 0.2, 0.3])
        assert m.n_pos == 0
        assert m.prevalence == 0.0
        assert m.pr_auc is None
        assert m.roc_auc is None
        assert m.best_f1 is None
        assert m.best_f1_threshold is None
        assert m.precision_at_k is None  # k defaults to n_pos = 0
        assert m.recall_at_k is None
        assert m.lift_at_k is None

    def test_all_positive_bundle(self) -> None:
        m = evaluate_scores([1, 1, 1], [0.1, 0.2, 0.3])
        assert m.pr_auc == pytest.approx(1.0)
        assert m.roc_auc is None
        assert m.precision_at_k == pytest.approx(1.0)

    def test_nan_scores_raise(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            evaluate_scores([0, 1], [0.1, float("nan")])

    def test_to_dict_is_json_serialisable_and_keeps_none(self) -> None:
        m = evaluate_scores([0, 0], [0.1, 0.2])
        d = m.to_dict()
        assert d["pr_auc"] is None
        assert json.loads(json.dumps(d))["pr_auc"] is None

    def test_score_fields_cover_the_reported_metrics(self) -> None:
        for field in DetectionMetrics.SCORE_FIELDS:
            assert field in evaluate_scores([1, 0], [0.9, 0.1]).to_dict()


# ═══════════════════════════════════════════════════════════════════════════════
# harness.py
# ═══════════════════════════════════════════════════════════════════════════════


class TestFlatten:
    def test_column_major_order(self) -> None:
        frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert flatten_cells(frame).tolist() == [1, 2, 3, 4]

    def test_labels_and_scores_flatten_consistently(self) -> None:
        labels = pd.DataFrame({"a": [True, False], "b": [False, True]})
        scores = pd.DataFrame({"a": [0.9, 0.1], "b": [0.2, 0.8]})
        y = flatten_cells(labels)
        s = flatten_cells(scores)
        # The two positives are the two highest scores — only true if the
        # flattening order is the same for both frames.
        assert sorted(s[y.astype(bool)].tolist()) == [0.8, 0.9]


class TestStratifiedSplits:
    def test_prevalence_preserved_in_every_split(self) -> None:
        rng = np.random.default_rng(0)
        y = (rng.random(50000) < 0.0025).astype(bool)
        splits = stratified_cell_splits(y, n_splits=5, seed=42, split_fraction=0.3)
        target = float(y.mean())
        for eval_idx in splits:
            assert float(y[eval_idx].mean()) == pytest.approx(target, abs=1e-4)

    def test_train_side_prevalence_also_preserved(self) -> None:
        rng = np.random.default_rng(1)
        y = (rng.random(20000) < 0.01).astype(bool)
        target = float(y.mean())
        for eval_idx in stratified_cell_splits(y, n_splits=3, seed=7):
            mask = np.ones(y.size, dtype=bool)
            mask[eval_idx] = False
            assert float(y[mask].mean()) == pytest.approx(target, abs=1e-3)

    def test_split_sizes_match_the_fraction(self) -> None:
        y = np.array([True] * 100 + [False] * 900)
        for eval_idx in stratified_cell_splits(y, n_splits=4, seed=42, split_fraction=0.3):
            assert int(y[eval_idx].sum()) == 30
            assert int((~y[eval_idx]).sum()) == 270

    def test_deterministic_for_a_given_seed(self) -> None:
        y = np.array([True] * 50 + [False] * 450)
        a = stratified_cell_splits(y, n_splits=3, seed=42)
        b = stratified_cell_splits(y, n_splits=3, seed=42)
        for x, z in zip(a, b):
            assert np.array_equal(x, z)

    def test_different_seeds_give_different_splits(self) -> None:
        y = np.array([True] * 50 + [False] * 450)
        a = stratified_cell_splits(y, n_splits=1, seed=42)[0]
        b = stratified_cell_splits(y, n_splits=1, seed=43)[0]
        assert not np.array_equal(a, b)

    def test_splits_differ_from_each_other(self) -> None:
        y = np.array([True] * 50 + [False] * 450)
        splits = stratified_cell_splits(y, n_splits=5, seed=42)
        assert not all(np.array_equal(splits[0], s) for s in splits[1:])

    def test_adding_splits_does_not_change_earlier_ones(self) -> None:
        """Split i depends on seed+i only, so the set is append-only."""
        y = np.array([True] * 40 + [False] * 360)
        three = stratified_cell_splits(y, n_splits=3, seed=42)
        five = stratified_cell_splits(y, n_splits=5, seed=42)
        for a, b in zip(three, five[:3]):
            assert np.array_equal(a, b)

    def test_eval_split_is_never_positive_free(self) -> None:
        """
        One positive in 10,000 cells: round(0.3 * 1) = 0, which would make
        PR-AUC undefined and silently drop the split from the CI. The clamp
        keeps one positive.
        """
        y = np.zeros(10000, dtype=bool)
        y[123] = True
        for eval_idx in stratified_cell_splits(y, n_splits=3, seed=42):
            assert int(y[eval_idx].sum()) == 1

    def test_invalid_arguments_raise(self) -> None:
        y = np.array([True, False, False, False])
        with pytest.raises(ValueError, match="n_splits"):
            stratified_cell_splits(y, n_splits=0)
        with pytest.raises(ValueError, match="seed"):
            stratified_cell_splits(y, seed=-1)
        with pytest.raises(ValueError, match="split_fraction"):
            stratified_cell_splits(y, split_fraction=0.0)
        with pytest.raises(ValueError, match="split_fraction"):
            stratified_cell_splits(y, split_fraction=1.0)


class _SpyFittedDetector:
    """
    A detector that records exactly which cells its train mask offered.

    Exists to prove the leakage guarantee of
    :func:`evaluate_detector`: if the harness ever hands a fitted detector a
    mask covering an evaluation cell, the recorded set intersects the eval set
    and the test fails.
    """

    name = "spy"

    def __init__(self) -> None:
        self.fit_calls = 0
        self.train_cells: List[set] = []
        self.train_rows: List[set] = []
        self.saw_labels = False

    def fit(self, dirty_df: pd.DataFrame, train_mask: pd.DataFrame) -> None:
        self.fit_calls += 1
        assert train_mask.shape == dirty_df.shape
        assert list(train_mask.columns) == list(dirty_df.columns)
        assert train_mask.to_numpy().dtype == np.bool_
        cells = {
            (int(r), str(c))
            for ci, c in enumerate(train_mask.columns)
            for r in np.flatnonzero(train_mask[c].to_numpy())
        }
        self.train_cells.append(cells)
        self.train_rows.append({r for r, _ in cells})

    def score_cells(
        self, dirty_df: pd.DataFrame, profile: Optional[Any] = None
    ) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            rng.random(dirty_df.shape),
            index=dirty_df.index,
            columns=dirty_df.columns,
        )


class _ConstantDetector:
    """Scores every cell the same. No ``fit``."""

    name = "constant"

    def score_cells(
        self, dirty_df: pd.DataFrame, profile: Optional[Any] = None
    ) -> pd.DataFrame:
        return pd.DataFrame(
            0.5, index=dirty_df.index, columns=dirty_df.columns, dtype=float
        )


class _BadShapeDetector:
    name = "bad_shape"

    def score_cells(
        self, dirty_df: pd.DataFrame, profile: Optional[Any] = None
    ) -> pd.DataFrame:
        return pd.DataFrame({dirty_df.columns[0]: np.zeros(len(dirty_df))})


class _NanDetector:
    name = "nan"

    def score_cells(
        self, dirty_df: pd.DataFrame, profile: Optional[Any] = None
    ) -> pd.DataFrame:
        frame = pd.DataFrame(
            0.0, index=dirty_df.index, columns=dirty_df.columns, dtype=float
        )
        frame.iloc[0, 0] = np.nan
        return frame


class _ReorderedColumnsDetector:
    name = "reordered"

    def score_cells(
        self, dirty_df: pd.DataFrame, profile: Optional[Any] = None
    ) -> pd.DataFrame:
        frame = pd.DataFrame(
            0.0, index=dirty_df.index, columns=dirty_df.columns, dtype=float
        )
        return frame[list(frame.columns)[::-1]]


class TestProtocol:
    def test_adapter_satisfies_the_protocol(self) -> None:
        assert isinstance(RuleDetectorAdapter(), DetectorProtocol)

    def test_plain_object_does_not(self) -> None:
        assert not isinstance(object(), DetectorProtocol)


class TestRuleDetectorAdapter:
    def test_score_frame_shape_and_alignment(self, bc_pair) -> None:
        from src.data.profiler import profile_dataset

        clean_df, dirty_df = bc_pair
        loader = DataLoader(DATA_ROOT)
        profile = profile_dataset(
            "breast_cancer", loader.discover("breast_cancer"), clean_df, dirty_df
        )
        scores = RuleDetectorAdapter().score_cells(dirty_df, profile)
        assert scores.shape == dirty_df.shape
        assert list(scores.columns) == list(dirty_df.columns)
        assert scores.index.equals(dirty_df.index)
        assert np.isfinite(scores.to_numpy()).all()

    def test_scores_land_on_the_detected_cells(self, bc_pair) -> None:
        """
        Every DetectionRecord must appear at its own (row, column), and no
        other cell may be non-zero. Re-derived from detect() independently of
        the adapter's own loop.
        """
        from src.data.profiler import profile_dataset
        from src.detection.detector import Detector

        clean_df, dirty_df = bc_pair
        loader = DataLoader(DATA_ROOT)
        profile = profile_dataset(
            "breast_cancer", loader.discover("breast_cancer"), clean_df, dirty_df
        )
        detector = Detector(profile)
        records = detector.detect(dirty_df)
        scores = RuleDetectorAdapter(detector=detector).score_cells(dirty_df, profile)

        expected = {(r.row_id, r.column): r.detection_score for r in records}
        for (row_id, col), score in expected.items():
            assert scores.iloc[row_id, scores.columns.get_loc(col)] == pytest.approx(score)
        assert int((scores.to_numpy() > 0.0).sum()) == len(
            [s for s in expected.values() if s > 0.0]
        )

    def test_unflagged_cells_score_zero(self, bc_pair) -> None:
        from src.data.profiler import profile_dataset

        clean_df, dirty_df = bc_pair
        loader = DataLoader(DATA_ROOT)
        profile = profile_dataset(
            "breast_cancer", loader.discover("breast_cancer"), clean_df, dirty_df
        )
        scores = RuleDetectorAdapter().score_cells(dirty_df, profile)
        assert float(scores.to_numpy().min()) == 0.0
        assert int((scores.to_numpy() == 0.0).sum()) > scores.size // 2

    def test_scores_are_heavily_tied(self, bc_pair) -> None:
        """
        The documented finding: a rule detector emits a handful of constants.
        If this ever stops holding, the claim in the adapter's docstring (and
        in the R0 report) is stale and must be rewritten.
        """
        from src.data.profiler import profile_dataset

        clean_df, dirty_df = bc_pair
        loader = DataLoader(DATA_ROOT)
        profile = profile_dataset(
            "breast_cancer", loader.discover("breast_cancer"), clean_df, dirty_df
        )
        scores = RuleDetectorAdapter().score_cells(dirty_df, profile)
        distinct = np.unique(scores.to_numpy())
        assert distinct.size <= 10
        _, counts = np.unique(scores.to_numpy(), return_counts=True)
        assert counts.max() / scores.size > 0.5

    def test_requires_profile_or_detector(self) -> None:
        with pytest.raises(ValueError, match="pre-built Detector or a"):
            RuleDetectorAdapter().score_cells(pd.DataFrame({"a": [1]}))

    def test_invalid_iqr_k_raises(self) -> None:
        with pytest.raises(ValueError, match="iqr_k"):
            RuleDetectorAdapter(iqr_k=0.0)

    def test_never_receives_clean_data(self) -> None:
        """
        No-leakage assertion (CLAUDE.md §3): ``score_cells`` has exactly two
        parameters, neither of which can be a clean DataFrame.
        """
        import inspect

        params = list(inspect.signature(RuleDetectorAdapter.score_cells).parameters)
        assert params == ["self", "dirty_df", "profile"]
        assert not any("clean" in p for p in params)


class TestEvaluateDetector:
    def test_leakage_fitted_detector_never_sees_an_eval_cell(self, tmp_path) -> None:
        """
        THE leakage test. The spy records every cell its train mask marked
        True; none of them may be an evaluation cell of that split.
        """
        root = tmp_path / "data"
        _write_tiny_dataset(root)
        spy = _SpyFittedDetector()
        out = evaluate_detector(
            spy, "tiny", data_root=str(root), n_splits=3, seed=42, split_fraction=0.3
        )
        assert out["refit_per_split"] is True
        assert spy.fit_calls == 3

        clean_df, dirty_df = DataLoader(str(root)).load("tiny")
        labels = cell_labels(clean_df, dirty_df)
        y = flatten_cells(labels).astype(bool)
        splits = stratified_cell_splits(y, n_splits=3, seed=42, split_fraction=0.3)
        n_rows = dirty_df.shape[0]
        columns = list(dirty_df.columns)

        for eval_idx, train_cells in zip(splits, spy.train_cells):
            eval_cells = {
                (int(i % n_rows), str(columns[int(i // n_rows)])) for i in eval_idx
            }
            assert eval_cells, "the split must actually hold out cells"
            assert not (eval_cells & train_cells), (
                "LEAKAGE: the fitted detector's train mask covered "
                f"{len(eval_cells & train_cells)} evaluation cell(s)."
            )
            # And the mask must be the exact complement — no cells withheld
            # from training for no reason either.
            all_cells = {(r, c) for c in columns for r in range(n_rows)}
            assert train_cells == all_cells - eval_cells

    def test_fit_signature_cannot_carry_labels(self, tmp_path) -> None:
        """
        Structural no-leakage guarantee: the harness calls ``fit`` with
        exactly ``(dirty_df, train_mask)``. There is nowhere to put labels.
        """
        root = tmp_path / "data"
        _write_tiny_dataset(root)
        captured: List[tuple] = []

        class _ArgSpy(_SpyFittedDetector):
            def fit(self, *args, **kwargs):  # type: ignore[override]
                captured.append((len(args), tuple(sorted(kwargs))))
                return super().fit(*args, **kwargs)

        evaluate_detector(_ArgSpy(), "tiny", data_root=str(root), n_splits=2)
        assert captured
        for n_args, kwarg_names in captured:
            assert n_args == 2
            assert kwarg_names == ()

    def test_full_is_flagged_in_sample_only_for_a_refit_detector(self, tmp_path) -> None:
        """
        For a refit detector, ``full`` uses the last split's model and is
        therefore partly in-sample. The flag must say so, or a reader would
        quote a contaminated whole-dataset number.
        """
        root = tmp_path / "data"
        _write_tiny_dataset(root)
        fitted = evaluate_detector(
            _SpyFittedDetector(), "tiny", data_root=str(root), n_splits=2
        )
        unfitted = evaluate_detector(
            _ConstantDetector(), "tiny", data_root=str(root), n_splits=2
        )
        assert fitted["full_is_in_sample"] is True
        assert unfitted["full_is_in_sample"] is False

    def test_unfitted_detector_is_scored_once(self, tmp_path) -> None:
        root = tmp_path / "data"
        _write_tiny_dataset(root)
        out = evaluate_detector(
            _ConstantDetector(), "tiny", data_root=str(root), n_splits=4
        )
        assert out["refit_per_split"] is False
        assert out["method"] == "constant"
        # A constant scorer's AP is the prevalence of each eval split.
        for split in out["per_split"]:
            assert split["pr_auc"] == pytest.approx(split["prevalence"], abs=1e-12)

    def test_deterministic(self, tmp_path) -> None:
        root = tmp_path / "data"
        _write_tiny_dataset(root)
        a = evaluate_detector(_ConstantDetector(), "tiny", data_root=str(root), seed=42)
        b = evaluate_detector(_ConstantDetector(), "tiny", data_root=str(root), seed=42)
        assert a["intervals"] == b["intervals"]
        assert a["per_split"] == b["per_split"]

    def test_bad_shape_rejected(self, tmp_path) -> None:
        root = tmp_path / "data"
        _write_tiny_dataset(root)
        with pytest.raises(ValueError, match="returned shape"):
            evaluate_detector(_BadShapeDetector(), "tiny", data_root=str(root))

    def test_nan_scores_rejected(self, tmp_path) -> None:
        root = tmp_path / "data"
        _write_tiny_dataset(root)
        with pytest.raises(ValueError, match="non-finite"):
            evaluate_detector(_NanDetector(), "tiny", data_root=str(root))

    def test_reordered_columns_rejected(self, tmp_path) -> None:
        root = tmp_path / "data"
        _write_tiny_dataset(root)
        with pytest.raises(ValueError, match="different order"):
            evaluate_detector(_ReorderedColumnsDetector(), "tiny", data_root=str(root))

    def test_rule_detector_on_breast_cancer(self) -> None:
        """
        The R0 baseline, end to end on real data. Asserts structure and
        internal consistency, not a hard-coded score — the score is the
        measurement, and pinning it would make the test a tautology that
        breaks on any legitimate detector change.
        """
        out = evaluate_detector(RuleDetectorAdapter(), "breast_cancer")
        assert out["label_stats"]["n_corrupted"] == 631
        assert out["full"]["n"] == 699 * 12
        assert out["full"]["pr_auc"] is not None
        # A real signal: must beat the random baseline, which is prevalence.
        assert out["full"]["pr_auc"] > out["full"]["prevalence"]
        iv = out["intervals"]["pr_auc"]
        assert iv["n"] == 5
        assert iv["lo"] <= iv["mean"] <= iv["hi"]
        assert len(out["per_split"]) == 5
        assert out["score_structure"]["n_distinct_scores"] <= 10

    def test_smartfactory_default_disables_outlier_flagging(self) -> None:
        """
        The shipped configuration (src/pipeline/runner.py) turns IQR fences
        off for smartfactory. The harness must measure the baseline the
        project actually runs, not a different one.
        """
        out = evaluate_detector(
            RuleDetectorAdapter(), "smartfactory", n_splits=1
        )
        # With fences off, only MISSING fires, and its score is exactly 1.0.
        assert set(np.round([g["score"] for g in out["score_structure"]["top_groups"]], 6)) <= {
            0.0,
            1.0,
        }

    def test_interval_empty_when_metric_undefined_everywhere(self, tmp_path) -> None:
        """
        An all-positive dataset makes ROC-AUC undefined on every split. The
        interval must come back with n = 0 (renders as '—'), not mean 0.0.
        """
        root = tmp_path / "allpos"
        d = root / "ap"
        d.mkdir(parents=True)
        clean = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
        dirty = pd.DataFrame({"a": [9.0, 8.0, 7.0, 6.0], "b": [9.0, 8.0, 7.0, 6.0]})
        clean.to_csv(d / "clean.csv", index=False)
        dirty.to_csv(d / "dirty.csv", index=False)
        out = evaluate_detector(_ConstantDetector(), "ap", data_root=str(root), n_splits=2)
        assert out["full"]["n_pos"] == 8
        assert out["intervals"]["roc_auc"]["n"] == 0
        assert out["intervals"]["pr_auc"]["n"] == 2

    def test_no_clean_data_parameter_anywhere(self) -> None:
        """No-leakage assertion: the public entry point takes no clean frame."""
        import inspect

        params = list(inspect.signature(evaluate_detector).parameters)
        assert not any("clean" in p for p in params)
        assert "dataset" in params


# ═══════════════════════════════════════════════════════════════════════════════
# leaderboard.py
# ═══════════════════════════════════════════════════════════════════════════════


def _fake_result(pr: float, dataset: str = "d1", n_pos: int = 10) -> dict:
    """A minimal evaluate_detector()-shaped dict."""
    return {
        "dataset": dataset,
        "method": "x",
        "n_splits": 5,
        "seed": 42,
        "split_fraction": 0.3,
        "label_stats": {"n_cells": 1000, "n_corrupted": n_pos, "prevalence": n_pos / 1000},
        "full": {
            "n": 1000,
            "n_pos": n_pos,
            "prevalence": n_pos / 1000,
            "k": n_pos,
            "pr_auc": pr,
            "roc_auc": 0.8,
            "best_f1": 0.5,
            "precision_at_k": 0.4,
            "recall_at_k": 0.4,
            "lift_at_k": 2.0,
        },
        "per_split": [],
        "intervals": {
            "pr_auc": {"mean": pr, "lo": pr - 0.02, "hi": pr + 0.02, "std": 0.01, "n": 5},
            "roc_auc": {"mean": 0.8, "lo": 0.78, "hi": 0.82, "std": 0.01, "n": 5},
            "best_f1": {"mean": 0.5, "lo": 0.48, "hi": 0.52, "std": 0.01, "n": 5},
            "precision_at_k": {"mean": 0.4, "lo": 0.38, "hi": 0.42, "std": 0.01, "n": 5},
            "recall_at_k": {"mean": 0.4, "lo": 0.38, "hi": 0.42, "std": 0.01, "n": 5},
            "lift_at_k": {"mean": 2.0, "lo": 1.9, "hi": 2.1, "std": 0.05, "n": 5},
        },
        "score_structure": {"n_distinct_scores": 3},
    }


class TestLeaderboard:
    def test_round_trip(self, tmp_path) -> None:
        store = tmp_path / "lb.json"
        record("m1", "d1", _fake_result(0.5), notes="hello", store_path=store)
        data = load_store(store)
        assert len(data["entries"]) == 1
        entry = data["entries"][0]
        assert entry["method"] == "m1"
        assert entry["dataset"] == "d1"
        assert entry["notes"] == "hello"
        assert entry["pr_auc"]["mean"] == pytest.approx(0.5)
        assert entry["n_pos"] == 10
        assert entry["prevalence"] == pytest.approx(0.01)

    def test_missing_store_reads_as_empty(self, tmp_path) -> None:
        assert load_store(tmp_path / "nope.json")["entries"] == []

    def test_replace_updates_rather_than_duplicating(self, tmp_path) -> None:
        store = tmp_path / "lb.json"
        record("m1", "d1", _fake_result(0.5), store_path=store)
        record("m1", "d1", _fake_result(0.7), store_path=store)
        entries = load_store(store)["entries"]
        assert len(entries) == 1
        assert entries[0]["pr_auc"]["mean"] == pytest.approx(0.7)

    def test_replace_false_keeps_both(self, tmp_path) -> None:
        store = tmp_path / "lb.json"
        record("m1", "d1", _fake_result(0.5), store_path=store)
        record("m1", "d1", _fake_result(0.7), store_path=store, replace=False)
        assert len(load_store(store)["entries"]) == 2

    def test_different_methods_coexist(self, tmp_path) -> None:
        store = tmp_path / "lb.json"
        record("a", "d1", _fake_result(0.5), store_path=store)
        record("b", "d1", _fake_result(0.6), store_path=store)
        record("a", "d2", _fake_result(0.1, dataset="d2"), store_path=store)
        assert len(load_store(store)["entries"]) == 3

    def test_accepts_a_detection_metrics_object(self, tmp_path) -> None:
        store = tmp_path / "lb.json"
        metrics = evaluate_scores([1, 0, 0, 1], [0.9, 0.2, 0.1, 0.8])
        entry = record("single", "d1", metrics, store_path=store)
        assert entry["pr_auc"]["n"] == 1  # one run has no interval
        assert entry["pr_auc"]["mean"] == pytest.approx(1.0)

    def test_undefined_metric_stored_as_n_zero(self, tmp_path) -> None:
        store = tmp_path / "lb.json"
        metrics = evaluate_scores([0, 0, 0], [0.1, 0.2, 0.3])
        entry = record("degenerate", "d1", metrics, store_path=store)
        assert entry["pr_auc"]["n"] == 0

    def test_empty_name_or_dataset_raises(self, tmp_path) -> None:
        store = tmp_path / "lb.json"
        with pytest.raises(ValueError, match="name"):
            record("", "d1", _fake_result(0.5), store_path=store)
        with pytest.raises(ValueError, match="dataset"):
            record("m", "  ", _fake_result(0.5), store_path=store)

    def test_bad_metrics_type_raises(self, tmp_path) -> None:
        with pytest.raises(TypeError, match="metrics must be"):
            record("m", "d", 0.5, store_path=tmp_path / "lb.json")

    def test_malformed_store_raises(self, tmp_path) -> None:
        store = tmp_path / "lb.json"
        store.write_text('{"nope": 1}', encoding="utf-8")
        with pytest.raises(ValueError, match="Malformed"):
            load_store(store)

    def test_wrong_version_raises(self, tmp_path) -> None:
        store = tmp_path / "lb.json"
        store.write_text('{"version": 99, "entries": []}', encoding="utf-8")
        with pytest.raises(ValueError, match="version"):
            load_store(store)

    def test_render_sorts_by_pr_auc_descending(self, tmp_path) -> None:
        store = tmp_path / "lb.json"
        md = tmp_path / "lb.md"
        record("low", "d1", _fake_result(0.10), store_path=store)
        record("high", "d1", _fake_result(0.90), store_path=store)
        record("mid", "d1", _fake_result(0.50), store_path=store)
        text = render_markdown(store_path=store, out_path=md)
        rows = [ln for ln in text.splitlines() if ln.startswith("| ") and "---" not in ln]
        order = [r.split("|")[1].strip() for r in rows if r.split("|")[1].strip() in
                 {"low", "mid", "high"}]
        assert order == ["high", "mid", "low"]

    def test_undefined_pr_auc_sorts_last_and_prints_dash(self, tmp_path) -> None:
        store = tmp_path / "lb.json"
        md = tmp_path / "lb.md"
        record("scored", "d1", _fake_result(0.0001), store_path=store)
        record("unscored", "d1", evaluate_scores([0, 0], [0.1, 0.2]), store_path=store)
        text = render_markdown(store_path=store, out_path=md)
        rows = [ln for ln in text.splitlines() if ln.startswith("| ")]
        names = [r.split("|")[1].strip() for r in rows]
        assert names.index("scored") < names.index("unscored")
        unscored_row = next(r for r in rows if r.split("|")[1].strip() == "unscored")
        # Columns 2..5 are PR-AUC / ROC-AUC / best-F1 / P@k. Every one of them
        # must print the dash, never 0.0000 — the prevalence column after them
        # legitimately reads 0.00000 because the dataset has no positives.
        metric_cells = [c.strip() for c in unscored_row.split("|")[2:6]]
        assert metric_cells == [MISSING] * 4

    def test_render_groups_by_dataset(self, tmp_path) -> None:
        store = tmp_path / "lb.json"
        md = tmp_path / "lb.md"
        record("m", "alpha", _fake_result(0.5, dataset="alpha"), store_path=store)
        record("m", "beta", _fake_result(0.5, dataset="beta"), store_path=store)
        text = render_markdown(store_path=store, out_path=md)
        assert "## alpha" in text
        assert "## beta" in text
        assert text.index("## alpha") < text.index("## beta")

    def test_render_includes_prevalence_and_ci(self, tmp_path) -> None:
        store = tmp_path / "lb.json"
        md = tmp_path / "lb.md"
        record("m", "d1", _fake_result(0.5), notes="n", store_path=store)
        text = render_markdown(store_path=store, out_path=md)
        assert "prevalence" in text
        assert "0.5000 [0.4800, 0.5200]" in text
        assert md.exists()
        assert md.read_text(encoding="utf-8") == text

    def test_render_empty_store(self, tmp_path) -> None:
        text = render_markdown(store_path=tmp_path / "none.json", out_path=tmp_path / "o.md")
        assert "No entries recorded yet" in text

    def test_render_escapes_pipes_in_notes(self, tmp_path) -> None:
        store = tmp_path / "lb.json"
        md = tmp_path / "lb.md"
        record("m", "d1", _fake_result(0.5), notes="a | b\nc", store_path=store)
        text = render_markdown(store_path=store, out_path=md)
        assert "a \\| b c" in text


# ═══════════════════════════════════════════════════════════════════════════════
# No-leakage, package level
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoLeakage:
    def test_only_labels_module_imports_clean_data_semantics(self) -> None:
        """
        ``clean_df`` may appear only in labels.py (label construction) and in
        harness.py (where it is loaded and immediately turned into labels).
        metrics.py and leaderboard.py must not mention it at all: they operate
        on scores and labels, never on values.
        """
        pkg = Path("src/detection_eval")
        for name in ("metrics.py", "leaderboard.py"):
            text = (pkg / name).read_text(encoding="utf-8")
            assert "clean_df" not in text, f"{name} references clean_df"

    def test_harness_passes_no_labels_to_a_detector(self) -> None:
        """
        Source-level check: the only calls made on a detector are ``fit`` and
        ``score_cells``, and neither call site mentions a label variable.
        """
        text = Path("src/detection_eval/harness.py").read_text(encoding="utf-8")
        assert "fit_fn(dirty_df, train_mask)" in text
        assert "detector.score_cells(dirty_df, profile)" in text
        assert "score_cells(dirty_df, profile, y" not in text

    def test_labels_are_not_reachable_from_score_cells_signature(self) -> None:
        import inspect

        sig = inspect.signature(RuleDetectorAdapter.score_cells)
        assert set(sig.parameters) == {"self", "dirty_df", "profile"}
