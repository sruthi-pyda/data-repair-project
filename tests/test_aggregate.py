"""
tests/test_aggregate.py
-----------------------
Phase 13 Step 1: confidence intervals and repeated-split aggregation
(src/experiments/aggregate.py).

Covers
  - Interval maths against hand-computed examples (a Student-t interval on
    [1..5] worked out by hand, not read back from the implementation)
  - bootstrap determinism: same seed → identical, different seed → different
    but close
  - bootstrap and t-interval agree within tolerance on a large normal sample
  - degenerate inputs: empty, n=1, all-equal
  - lo <= mean <= hi always
  - to_dict() keys, __str__ formatting including the n=0 "—" case
  - RepeatedSplitResult.to_dict() round-trips every nested interval
  - argument validation
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.experiments.aggregate import (
    ACTION_KEYS,
    CALIBRATION_METHODS,
    NOT_FITTED_MARKER,
    Interval,
    RepeatedSplitResult,
    bootstrap_interval,
    normal_interval,
    repeated_splits,
)
from src.risk.cost_model import CostConfig

# [1, 2, 3, 4, 5]: mean 3, s = sqrt(2.5) = 1.5811388, sem = s/sqrt(5) =
# 0.70710678, t(0.975, df=4) = 2.7764451 → half-width 1.9632433.
FIVE = [1.0, 2.0, 3.0, 4.0, 5.0]
FIVE_MEAN = 3.0
FIVE_STD = math.sqrt(2.5)
FIVE_HALF = 1.9632433


# ═══════════════════════════════════════════════════════════════════════════════
# Interval dataclass
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntervalDataclass:
    def test_to_dict_keys(self):
        iv = Interval(mean=0.5, lo=0.4, hi=0.6, std=0.1, n=7)
        assert iv.to_dict() == {
            "mean": 0.5, "lo": 0.4, "hi": 0.6, "std": 0.1, "n": 7
        }

    def test_to_dict_values_are_plain_python(self):
        iv = Interval(
            mean=np.float64(0.5), lo=np.float64(0.4), hi=np.float64(0.6),
            std=np.float64(0.1), n=np.int64(3),
        )
        d = iv.to_dict()
        assert type(d["mean"]) is float
        assert type(d["n"]) is int

    def test_str_format_is_four_decimal_places(self):
        iv = Interval(mean=0.1899, lo=0.1721, hi=0.2077, std=0.02, n=20)
        assert str(iv) == "0.1899 [0.1721, 0.2077]"

    def test_str_rounds_to_four_dp(self):
        iv = Interval(mean=0.189912345, lo=0.17209, hi=0.207749, std=0.0, n=5)
        assert str(iv) == "0.1899 [0.1721, 0.2077]"

    def test_str_of_empty_interval_is_the_marker(self):
        iv = Interval(mean=0.0, lo=0.0, hi=0.0, std=0.0, n=0)
        assert str(iv) == NOT_FITTED_MARKER
        assert str(iv) == "—"

    def test_is_empty_flag(self):
        assert Interval(0.0, 0.0, 0.0, 0.0, 0).is_empty
        assert not Interval(0.1, 0.1, 0.1, 0.0, 1).is_empty

    def test_empty_interval_str_does_not_look_like_a_score(self):
        """A 0-split method must not print 0.0000, which reads as perfect."""
        assert "0.0000" not in str(Interval(0.0, 0.0, 0.0, 0.0, 0))


# ═══════════════════════════════════════════════════════════════════════════════
# normal_interval (Student-t)
# ═══════════════════════════════════════════════════════════════════════════════


class TestNormalInterval:
    def test_hand_computed_example(self):
        iv = normal_interval(FIVE)
        assert iv.n == 5
        assert iv.mean == pytest.approx(FIVE_MEAN)
        assert iv.std == pytest.approx(FIVE_STD, abs=1e-9)
        assert iv.lo == pytest.approx(FIVE_MEAN - FIVE_HALF, abs=1e-6)
        assert iv.hi == pytest.approx(FIVE_MEAN + FIVE_HALF, abs=1e-6)

    def test_uses_t_not_normal(self):
        """At n=5 the t half-width is clearly wider than 1.96 * sem."""
        iv = normal_interval(FIVE)
        z_half = 1.959964 * FIVE_STD / math.sqrt(5)
        assert (iv.hi - iv.mean) > z_half * 1.3

    def test_wider_confidence_is_wider_interval(self):
        narrow = normal_interval(FIVE, confidence=0.80)
        wide = normal_interval(FIVE, confidence=0.99)
        assert (wide.hi - wide.lo) > (narrow.hi - narrow.lo)

    def test_empty(self):
        iv = normal_interval([])
        assert iv.n == 0 and iv.mean == 0.0 and iv.lo == 0.0 and iv.hi == 0.0
        assert str(iv) == NOT_FITTED_MARKER

    def test_single_value(self):
        iv = normal_interval([0.37])
        assert iv.n == 1
        assert iv.mean == pytest.approx(0.37)
        assert iv.lo == iv.hi == pytest.approx(0.37)
        assert iv.std == 0.0

    def test_all_equal_values(self):
        iv = normal_interval([0.25] * 8)
        assert iv.n == 8
        assert iv.std == 0.0
        assert iv.lo == iv.hi == pytest.approx(0.25)

    def test_ordering(self):
        iv = normal_interval([0.1, 0.9, 0.4, 0.3, 0.55, 0.2])
        assert iv.lo <= iv.mean <= iv.hi

    def test_rejects_bad_confidence(self):
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError):
                normal_interval(FIVE, confidence=bad)

    def test_rejects_non_finite(self):
        with pytest.raises(ValueError):
            normal_interval([0.1, float("nan")])
        with pytest.raises(ValueError):
            normal_interval([0.1, float("inf")])


# ═══════════════════════════════════════════════════════════════════════════════
# bootstrap_interval
# ═══════════════════════════════════════════════════════════════════════════════


class TestBootstrapInterval:
    def test_mean_is_the_sample_mean(self):
        iv = bootstrap_interval(FIVE, n_boot=2000, seed=42)
        assert iv.mean == pytest.approx(FIVE_MEAN)
        assert iv.std == pytest.approx(FIVE_STD, abs=1e-9)
        assert iv.n == 5

    def test_deterministic_same_seed(self):
        a = bootstrap_interval(FIVE, n_boot=5000, seed=7)
        b = bootstrap_interval(FIVE, n_boot=5000, seed=7)
        assert a == b
        assert a.to_dict() == b.to_dict()

    def test_different_seed_differs_but_stays_close(self):
        values = np.random.default_rng(11).normal(0.2, 0.05, size=30)
        a = bootstrap_interval(values, n_boot=5000, seed=1)
        b = bootstrap_interval(values, n_boot=5000, seed=2)
        assert a.mean == pytest.approx(b.mean)        # point estimate is exact
        assert (a.lo, a.hi) != (b.lo, b.hi)           # the resampling moved
        assert a.lo == pytest.approx(b.lo, abs=5e-3)  # but only a little
        assert a.hi == pytest.approx(b.hi, abs=5e-3)

    def test_agrees_with_t_interval_on_large_normal_sample(self):
        values = np.random.default_rng(99).normal(0.2, 0.05, size=400)
        boot = bootstrap_interval(values, n_boot=4000, seed=42)
        norm = normal_interval(values)
        assert boot.mean == pytest.approx(norm.mean)
        assert boot.lo == pytest.approx(norm.lo, abs=3e-3)
        assert boot.hi == pytest.approx(norm.hi, abs=3e-3)

    def test_empty(self):
        iv = bootstrap_interval([])
        assert iv.n == 0
        assert str(iv) == NOT_FITTED_MARKER

    def test_single_value(self):
        iv = bootstrap_interval([0.42], n_boot=100, seed=3)
        assert iv.n == 1
        assert iv.lo == iv.hi == pytest.approx(0.42)
        assert iv.std == 0.0

    def test_all_equal_values(self):
        iv = bootstrap_interval([0.3] * 10, n_boot=500, seed=5)
        assert iv.std == 0.0
        assert iv.lo == iv.hi
        assert iv.lo == pytest.approx(0.3)

    def test_ordering(self):
        iv = bootstrap_interval([0.1, 0.9, 0.4, 0.3, 0.55, 0.2], n_boot=3000)
        assert iv.lo <= iv.mean <= iv.hi

    def test_wider_confidence_is_wider_interval(self):
        narrow = bootstrap_interval(FIVE, confidence=0.80, n_boot=8000, seed=4)
        wide = bootstrap_interval(FIVE, confidence=0.99, n_boot=8000, seed=4)
        assert (wide.hi - wide.lo) >= (narrow.hi - narrow.lo)

    def test_rejects_bad_arguments(self):
        with pytest.raises(ValueError):
            bootstrap_interval(FIVE, n_boot=0)
        with pytest.raises(ValueError):
            bootstrap_interval(FIVE, seed=-1)
        with pytest.raises(ValueError):
            bootstrap_interval(FIVE, confidence=0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# RepeatedSplitResult
# ═══════════════════════════════════════════════════════════════════════════════


class TestRepeatedSplitResult:
    def _result(self) -> RepeatedSplitResult:
        iv = Interval(mean=0.19, lo=0.17, hi=0.21, std=0.02, n=3)
        return RepeatedSplitResult(
            dataset="breast_cancer",
            n_splits=3,
            seed=42,
            split_seeds=[1000, 1001, 1002],
            brier={"raw": iv, "isotonic": Interval(0.0, 0.0, 0.0, 0.0, 0)},
            ece={"raw": iv},
            n_fitted={"raw": 3, "isotonic": 0},
            cost={"calibrated_risk_aware": Interval(1500.0, 1400.0, 1600.0, 80.0, 3)},
            actions={"calibrated_risk_aware": {"auto": iv, "human": iv, "abstain": iv}},
            interval="bootstrap",
            n_decided=118,
        )

    def test_to_dict_round_trips(self):
        d = self._result().to_dict()
        assert d["dataset"] == "breast_cancer"
        assert d["n_splits"] == 3
        assert d["seed"] == 42
        assert d["split_seeds"] == [1000, 1001, 1002]
        assert d["interval"] == "bootstrap"
        assert d["n_decided"] == 118
        assert d["brier"]["raw"] == {
            "mean": 0.19, "lo": 0.17, "hi": 0.21, "std": 0.02, "n": 3
        }
        assert d["brier"]["isotonic"]["n"] == 0
        assert d["n_fitted"] == {"raw": 3, "isotonic": 0}
        assert d["cost"]["calibrated_risk_aware"]["mean"] == 1500.0
        assert set(d["actions"]["calibrated_risk_aware"]) == set(ACTION_KEYS)
        assert d["actions"]["calibrated_risk_aware"]["auto"]["n"] == 3

    def test_to_dict_is_json_serialisable(self):
        import json

        json.dumps(self._result().to_dict())

    def test_unfitted_method_prints_as_marker(self):
        res = self._result()
        assert str(res.brier["isotonic"]) == NOT_FITTED_MARKER
        assert res.n_fitted["isotonic"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# repeated_splits argument validation (no pipeline run)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRepeatedSplitsValidation:
    @staticmethod
    def _costs() -> CostConfig:
        return CostConfig(
            c_wrong_repair=50, c_human_review=5, c_leave_error=10,
            dataset_name="breast_cancer",
        )

    def test_n_splits_below_one(self):
        for bad in (0, -1):
            with pytest.raises(ValueError, match="n_splits"):
                repeated_splits("breast_cancer", self._costs(), n_splits=bad)

    def test_unknown_interval_method(self):
        with pytest.raises(ValueError, match="interval"):
            repeated_splits("breast_cancer", self._costs(), interval="jackknife")

    def test_negative_base_split_seed(self):
        with pytest.raises(ValueError, match="base_split_seed"):
            repeated_splits("breast_cancer", self._costs(), base_split_seed=-1)

    def test_bad_confidence(self):
        with pytest.raises(ValueError, match="confidence"):
            repeated_splits("breast_cancer", self._costs(), confidence=1.0)

    def test_unknown_dataset_propagates_from_run_pipeline(self):
        with pytest.raises(ValueError):
            repeated_splits("nope", self._costs(), n_splits=1, n_records=10)

    def test_calibration_methods_and_action_keys_are_the_expected_sets(self):
        assert CALIBRATION_METHODS == ("raw", "logistic", "isotonic", "multivariate")
        assert ACTION_KEYS == ("auto", "human", "abstain")
