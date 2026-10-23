"""
tests/test_aggregate_bootstrap.py
---------------------------------
Tests for `bootstrap_metric` in `src/experiments/aggregate.py`.

The decisive test is `TestFixesTheZeroWidthDefect`. This function exists for
exactly one reason (`docs/w11_plan.md`): five of eleven datasets sit at or
below the experiments' row cap, so their "replicates" are identical runs and
the published intervals came out zero-width — which reads as perfect
stability when in fact nothing was resampled. If cell-level bootstrapping
does not produce a real interval on a single run, the fix does not work.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.detection_eval.metrics import pr_auc, roc_auc
from src.experiments.aggregate import (
    Interval,
    bootstrap_interval,
    bootstrap_metric,
)

SEED = 42


def _scored(n: int = 2000, prevalence: float = 0.05, seed: int = SEED):
    """Overlapping score distributions, so the metric is genuinely uncertain."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    s = np.where(y == 1, rng.beta(5.0, 2.0, n), rng.beta(2.0, 6.0, n))
    return y, s


# ═══════════════════════════════════════════════════════════════════════════
# THE DECISIVE TEST
# ═══════════════════════════════════════════════════════════════════════════


class TestFixesTheZeroWidthDefect:
    def test_identical_replicates_give_bootstrap_interval_zero_width(self):
        """
        Premise: the OLD approach collapses. Five identical replicate values
        bootstrap to a zero-width interval, which is the published defect.
        """
        iv = bootstrap_interval([0.8360] * 5, seed=SEED)
        assert iv.hi - iv.lo == 0.0, (
            "if this ever becomes non-zero the premise of W11 has changed")

    def test_cell_bootstrap_gives_a_real_interval_from_one_run(self):
        """The fix: a genuine interval from a single run, no replicates."""
        y, s = _scored()
        iv = bootstrap_metric(y, s, pr_auc, seed=SEED)
        assert iv.n > 0
        assert iv.hi - iv.lo > 0.01, (
            f"interval is {iv.lo:.4f}-{iv.hi:.4f}; too narrow to be a real "
            f"uncertainty estimate")

    def test_small_dataset_still_gets_an_interval(self):
        """
        The datasets that broke are small (breast_cancer has 699 rows). A
        smaller sample must give a WIDER interval, not no interval.
        """
        y_big, s_big = _scored(n=4000)
        y_sml, s_sml = _scored(n=400)
        wide = bootstrap_metric(y_sml, s_sml, pr_auc, seed=SEED)
        narrow = bootstrap_metric(y_big, s_big, pr_auc, seed=SEED)
        assert wide.n > 0 and narrow.n > 0
        assert (wide.hi - wide.lo) > (narrow.hi - narrow.lo), (
            f"400 rows gave {wide.hi-wide.lo:.4f} but 4000 gave "
            f"{narrow.hi-narrow.lo:.4f}; uncertainty must grow as n shrinks")


class TestProperties:
    def test_point_estimate_is_the_observed_value(self):
        """`mean` must be the metric on the real data, not the bootstrap mean."""
        y, s = _scored()
        iv = bootstrap_metric(y, s, pr_auc, seed=SEED)
        assert iv.mean == pytest.approx(pr_auc(y, s), abs=1e-12)

    def test_interval_brackets_the_point_estimate(self):
        for seed in (1, 7, 42):
            y, s = _scored(seed=seed)
            iv = bootstrap_metric(y, s, pr_auc, seed=SEED)
            assert iv.lo <= iv.mean <= iv.hi

    def test_deterministic(self):
        y, s = _scored()
        a = bootstrap_metric(y, s, pr_auc, seed=SEED)
        b = bootstrap_metric(y, s, pr_auc, seed=SEED)
        assert (a.mean, a.lo, a.hi, a.n) == (b.mean, b.lo, b.hi, b.n)

    def test_wider_confidence_is_wider(self):
        y, s = _scored()
        a = bootstrap_metric(y, s, pr_auc, confidence=0.80, seed=SEED)
        b = bootstrap_metric(y, s, pr_auc, confidence=0.99, seed=SEED)
        assert (b.hi - b.lo) > (a.hi - a.lo)

    def test_works_for_any_metric(self):
        y, s = _scored()
        iv = bootstrap_metric(y, s, roc_auc, seed=SEED)
        assert 0.0 <= iv.lo <= iv.hi <= 1.0

    def test_a_truly_constant_metric_reports_zero_width_honestly(self):
        """
        Perfect separation makes PR-AUC 1.0 on every resample. A zero-width
        interval is then a MEASURED fact, not an artefact -- which is the
        distinction W11 exists to restore.
        """
        y = np.array([0] * 200 + [1] * 200)
        s = np.array([0.0] * 200 + [1.0] * 200)
        iv = bootstrap_metric(y, s, pr_auc, seed=SEED)
        assert iv.mean == pytest.approx(1.0)
        assert iv.hi - iv.lo == pytest.approx(0.0, abs=1e-12)
        assert iv.n > 0, "a constant metric must still report resamples"


class TestDegenerate:
    def test_empty_input_is_an_empty_interval_not_a_score(self):
        iv = bootstrap_metric(np.array([]), np.array([]), pr_auc, seed=SEED)
        assert iv.n == 0

    def test_all_one_class_is_empty_not_zero(self):
        """pr_auc is undefined with no positive; that must not become 0.0."""
        y = np.zeros(300, dtype=int)
        s = np.random.default_rng(SEED).random(300)
        iv = bootstrap_metric(y, s, pr_auc, seed=SEED)
        assert iv.n == 0
        assert iv.mean == 0.0 and iv.lo == 0.0 and iv.hi == 0.0

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError):
            bootstrap_metric(np.zeros(5), np.zeros(4), pr_auc)

    def test_bad_parameters_rejected(self):
        y, s = _scored(n=200)
        with pytest.raises(ValueError):
            bootstrap_metric(y, s, pr_auc, confidence=0.0)
        with pytest.raises(ValueError):
            bootstrap_metric(y, s, pr_auc, confidence=1.0)
        with pytest.raises(ValueError):
            bootstrap_metric(y, s, pr_auc, n_boot=0)
