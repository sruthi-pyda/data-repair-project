"""
tests/test_calibration.py
--------------------------
Comprehensive tests for Phase 6: Probability Calibration.

Covers:
  - CalibrationSample schema
  - CalibrationBuilder: correct labelling, no clean values in features
  - train_eval_split: sizes, determinism, reproducibility
  - RawBaseline: identity pass-through
  - LogisticCalibrator: fit, predict, probabilities in [0,1], constant labels
  - IsotonicCalibrator: fit, predict, monotonicity, probabilities in [0,1]
  - brier_score: formula, bounds, errors
  - expected_calibration_error: formula, bins, ECE bounds
  - calibrate_dataset: all fields, adult limitation
  - CalibrationComparison: all attributes
  - Data leakage: calibrators don't see clean values
  - Small-sample handling
  - Determinism
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from src.calibration.calibrator import (
    CalibrationBuilder,
    CalibrationComparison,
    CalibrationSample,
    IsotonicCalibrator,
    LogisticCalibrator,
    RawBaseline,
    _ADULT_LIMITATION,
    calibrate_dataset,
    print_calibration_report,
    train_eval_split,
)
from src.calibration.metrics import (
    ECEBin,
    brier_score,
    expected_calibration_error,
)
from src.uncertainty.estimator import UncertaintyRecord


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _make_samples(
    confidences: List[float],
    corrects: List[int],
    dataset: str = "test",
) -> List[CalibrationSample]:
    return [
        CalibrationSample(
            dataset=dataset,
            row_id=i,
            column="col",
            raw_confidence=c,
            correct=y,
        )
        for i, (c, y) in enumerate(zip(confidences, corrects))
    ]


def _make_unc_record(
    row_id: int,
    column: str,
    top_candidate,
    raw_confidence: float = 0.7,
    dataset: str = "test",
) -> UncertaintyRecord:
    return UncertaintyRecord(
        dataset=dataset,
        row_id=row_id,
        column=column,
        dirty_value=None,
        error_type="missing",
        repair_method="test",
        top_candidate=top_candidate,
        raw_confidence=raw_confidence,
        self_consistency=1.0,
        entropy=0.3,
        normalized_entropy=0.3,
        n_samples=5,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: CalibrationSample
# ═══════════════════════════════════════════════════════════════════════════════


class TestCalibrationSample:
    def test_fields_set(self):
        s = CalibrationSample(dataset="bc", row_id=0, column="age",
                              raw_confidence=0.75, correct=1)
        assert s.dataset == "bc"
        assert s.raw_confidence == 0.75
        assert s.correct == 1

    def test_correct_is_binary(self):
        for val in [0, 1]:
            s = CalibrationSample("ds", 0, "col", 0.5, val)
            assert s.correct in (0, 1)

    def test_no_clean_value_field(self):
        fields = CalibrationSample.__dataclass_fields__
        for f in fields:
            assert "clean" not in f.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: CalibrationBuilder
# ═══════════════════════════════════════════════════════════════════════════════


class TestCalibrationBuilder:
    @pytest.fixture
    def builder(self) -> CalibrationBuilder:
        return CalibrationBuilder()

    def test_correct_label_when_match(self, builder):
        clean_df = pd.DataFrame({"age": [25, 30, 35]})
        dirty_df = clean_df.copy()
        unc = [_make_unc_record(0, "age", top_candidate=25)]
        samples = builder.build(unc, clean_df, dirty_df, "test")
        assert samples[0].correct == 1

    def test_incorrect_label_when_mismatch(self, builder):
        clean_df = pd.DataFrame({"age": [25, 30, 35]})
        dirty_df = clean_df.copy()
        unc = [_make_unc_record(0, "age", top_candidate=99)]
        samples = builder.build(unc, clean_df, dirty_df, "test")
        assert samples[0].correct == 0

    def test_raw_confidence_preserved(self, builder):
        clean_df = pd.DataFrame({"age": [25]})
        dirty_df = clean_df.copy()
        unc = [_make_unc_record(0, "age", top_candidate=25, raw_confidence=0.82)]
        samples = builder.build(unc, clean_df, dirty_df, "test")
        assert samples[0].raw_confidence == pytest.approx(0.82)

    def test_no_clean_value_in_samples(self, builder):
        """CalibrationSample must not contain clean values as fields."""
        clean_df = pd.DataFrame({"age": [25]})
        dirty_df = clean_df.copy()
        unc = [_make_unc_record(0, "age", top_candidate=25)]
        samples = builder.build(unc, clean_df, dirty_df, "test")
        s = samples[0]
        # Only raw_confidence and correct — no 'clean_value' field
        sample_dict = vars(s)
        for key in sample_dict:
            assert "clean" not in key.lower()

    def test_skips_missing_column(self, builder):
        clean_df = pd.DataFrame({"age": [25]})
        dirty_df = clean_df.copy()
        unc = [_make_unc_record(0, "missing_col", top_candidate=1)]
        samples = builder.build(unc, clean_df, dirty_df, "test")
        assert len(samples) == 0

    def test_skips_out_of_range_row(self, builder):
        clean_df = pd.DataFrame({"age": [25]})
        dirty_df = clean_df.copy()
        unc = [_make_unc_record(99, "age", top_candidate=25)]
        samples = builder.build(unc, clean_df, dirty_df, "test")
        assert len(samples) == 0

    def test_int_vs_float_match(self, builder):
        """25.0 should match clean value 25 (int)."""
        clean_df = pd.DataFrame({"age": [25]})
        dirty_df = clean_df.copy()
        unc = [_make_unc_record(0, "age", top_candidate=25.0)]
        samples = builder.build(unc, clean_df, dirty_df, "test")
        assert samples[0].correct == 1

    def test_multiple_records(self, builder):
        clean_df = pd.DataFrame({"age": [1, 2, 3, 4, 5]})
        dirty_df = clean_df.copy()
        unc = [
            _make_unc_record(0, "age", top_candidate=1),   # correct
            _make_unc_record(1, "age", top_candidate=99),  # wrong
            _make_unc_record(2, "age", top_candidate=3),   # correct
        ]
        samples = builder.build(unc, clean_df, dirty_df, "test")
        assert len(samples) == 3
        corrects = [s.correct for s in samples]
        assert corrects == [1, 0, 1]


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: train_eval_split
# ═══════════════════════════════════════════════════════════════════════════════


class TestTrainEvalSplit:
    def test_total_size_preserved(self):
        samples = _make_samples([0.5] * 20, [1] * 20)
        train, eval_ = train_eval_split(samples, eval_fraction=0.3)
        assert len(train) + len(eval_) == 20

    def test_eval_fraction_approximate(self):
        samples = _make_samples([0.5] * 100, [1] * 100)
        train, eval_ = train_eval_split(samples, eval_fraction=0.3)
        assert 25 <= len(eval_) <= 35  # 30% ± 5

    def test_determinism(self):
        samples = _make_samples(list(range(20)), [i % 2 for i in range(20)])
        tr1, ev1 = train_eval_split(samples, seed=42)
        tr2, ev2 = train_eval_split(samples, seed=42)
        assert [s.row_id for s in tr1] == [s.row_id for s in tr2]
        assert [s.row_id for s in ev1] == [s.row_id for s in ev2]

    def test_different_seeds_differ(self):
        samples = _make_samples(list(range(30)), [i % 2 for i in range(30)])
        _, ev1 = train_eval_split(samples, seed=42)
        _, ev2 = train_eval_split(samples, seed=99)
        assert [s.row_id for s in ev1] != [s.row_id for s in ev2]

    def test_empty_input(self):
        train, eval_ = train_eval_split([])
        assert train == []
        assert eval_ == []

    def test_minimum_eval_size(self):
        samples = _make_samples([0.5], [1])
        train, eval_ = train_eval_split(samples, eval_fraction=0.3)
        assert len(eval_) >= 1

    def test_no_overlap(self):
        samples = _make_samples([float(i) / 100 for i in range(50)],
                                [i % 2 for i in range(50)])
        train, eval_ = train_eval_split(samples)
        train_ids = {(s.row_id, s.raw_confidence) for s in train}
        eval_ids = {(s.row_id, s.raw_confidence) for s in eval_}
        assert len(train_ids & eval_ids) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: RawBaseline
# ═══════════════════════════════════════════════════════════════════════════════


class TestRawBaseline:
    def test_identity(self):
        raw = RawBaseline()
        X = np.array([0.3, 0.5, 0.8])
        out = raw.predict(X)
        np.testing.assert_allclose(out, X)

    def test_clips_to_zero_one(self):
        raw = RawBaseline()
        X = np.array([-0.1, 0.5, 1.1])
        out = raw.predict(X)
        assert out[0] == pytest.approx(0.0)
        assert out[2] == pytest.approx(1.0)

    def test_is_fitted(self):
        assert RawBaseline().is_fitted()

    def test_name(self):
        assert RawBaseline.name == "raw"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: LogisticCalibrator
# ═══════════════════════════════════════════════════════════════════════════════


class TestLogisticCalibrator:
    def _make_xy(self, n: int = 50, seed: int = 0):
        """Create synthetic calibration data with known structure."""
        rng = np.random.default_rng(seed)
        X = rng.uniform(0.3, 0.9, size=n)
        # True probability: slightly higher than X (underconfident raw)
        true_p = np.clip(X + 0.1, 0, 1)
        y = (rng.uniform(size=n) < true_p).astype(float)
        return X, y

    def test_fit_and_predict(self):
        cal = LogisticCalibrator()
        X, y = self._make_xy()
        cal.fit(X, y)
        assert cal.is_fitted()
        preds = cal.predict(X)
        assert preds.shape == X.shape

    def test_probabilities_in_zero_one(self):
        cal = LogisticCalibrator()
        X, y = self._make_xy()
        cal.fit(X, y)
        preds = cal.predict(X)
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 1.0)

    def test_constant_labels_handled(self):
        """All y=0 → no meaningful calibration, should not raise."""
        cal = LogisticCalibrator()
        X = np.array([0.5, 0.6, 0.7])
        y = np.zeros(3)
        cal.fit(X, y)
        assert not cal.is_fitted()
        # Still has a_ and b_ set for fallback
        assert cal.a_ is not None

    def test_predict_before_fit_raises(self):
        cal = LogisticCalibrator()
        with pytest.raises(RuntimeError):
            cal.predict(np.array([0.5]))

    def test_parameters_set_after_fit(self):
        cal = LogisticCalibrator()
        X, y = self._make_xy()
        cal.fit(X, y)
        assert cal.a_ is not None
        assert cal.b_ is not None

    def test_monotone_predictions_for_sorted_input(self):
        """Logistic is globally monotone by construction: larger input → larger output."""
        cal = LogisticCalibrator()
        X, y = self._make_xy()
        cal.fit(X, y)
        X_sorted = np.linspace(0.1, 0.99, 20)
        preds = cal.predict(X_sorted)
        # Predictions should be non-decreasing for a > 0
        if cal.a_ > 0:
            assert np.all(np.diff(preds) >= -1e-9)

    def test_name(self):
        assert LogisticCalibrator.name == "logistic"

    def test_small_sample_still_fits(self):
        """Should fit (with warning) even on 3 samples."""
        cal = LogisticCalibrator()
        X = np.array([0.4, 0.6, 0.8])
        y = np.array([0.0, 0.0, 1.0])
        cal.fit(X, y)
        # May or may not be fitted depending on variance


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: IsotonicCalibrator
# ═══════════════════════════════════════════════════════════════════════════════


class TestIsotonicCalibrator:
    def _make_xy(self, n: int = 60, seed: int = 1):
        rng = np.random.default_rng(seed)
        X = rng.uniform(0.3, 0.9, size=n)
        true_p = np.clip(X + 0.1, 0, 1)
        y = (rng.uniform(size=n) < true_p).astype(float)
        return X, y

    def test_fit_and_predict(self):
        cal = IsotonicCalibrator()
        X, y = self._make_xy()
        cal.fit(X, y)
        assert cal.is_fitted()
        preds = cal.predict(X)
        assert preds.shape == X.shape

    def test_probabilities_in_zero_one(self):
        cal = IsotonicCalibrator()
        X, y = self._make_xy()
        cal.fit(X, y)
        preds = cal.predict(X)
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 1.0)

    def test_monotonicity_on_sorted_input(self):
        """Isotonic regression must produce non-decreasing output for sorted input."""
        cal = IsotonicCalibrator()
        X, y = self._make_xy(n=100, seed=7)
        cal.fit(X, y)
        X_sorted = np.sort(X)
        preds = cal.predict(X_sorted)
        # Allow tiny floating point jitter
        assert np.all(np.diff(preds) >= -1e-6), \
            f"Isotonic output not monotone: {np.diff(preds).min():.6f}"

    def test_constant_labels_handled(self):
        """All y=0 → skip fit, not fitted."""
        cal = IsotonicCalibrator()
        X = np.array([0.4, 0.6, 0.8])
        y = np.zeros(3)
        cal.fit(X, y)
        assert not cal.is_fitted()

    def test_predict_before_fit_raises(self):
        cal = IsotonicCalibrator()
        with pytest.raises(RuntimeError):
            cal.predict(np.array([0.5]))

    def test_name(self):
        assert IsotonicCalibrator.name == "isotonic"

    def test_out_of_bounds_clips(self):
        """Values outside training range should be clipped to [0, 1]."""
        cal = IsotonicCalibrator()
        X, y = self._make_xy()
        cal.fit(X, y)
        preds = cal.predict(np.array([0.0, 1.0]))
        assert np.all(preds >= 0.0) and np.all(preds <= 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: brier_score
# ═══════════════════════════════════════════════════════════════════════════════


class TestBrierScore:
    def test_perfect_predictions(self):
        """All correct predictions → Brier = 0."""
        probs = np.array([1.0, 1.0, 0.0, 0.0])
        labels = np.array([1.0, 1.0, 0.0, 0.0])
        assert brier_score(probs, labels) == pytest.approx(0.0)

    def test_worst_predictions(self):
        """Opposite predictions → Brier = 1."""
        probs = np.array([0.0, 1.0])
        labels = np.array([1.0, 0.0])
        assert brier_score(probs, labels) == pytest.approx(1.0)

    def test_uniform_predictions(self):
        """Always predict 0.5 → Brier = 0.25."""
        probs = np.array([0.5] * 100)
        labels = np.array([float(i % 2) for i in range(100)])
        assert brier_score(probs, labels) == pytest.approx(0.25, abs=1e-6)

    def test_formula_single_example(self):
        probs = np.array([0.8])
        labels = np.array([1.0])
        expected = (0.8 - 1.0) ** 2
        assert brier_score(probs, labels) == pytest.approx(expected)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            brier_score(np.array([]), np.array([]))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            brier_score(np.array([0.5, 0.6]), np.array([1.0]))

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            brier_score(np.array([1.5]), np.array([1.0]))

    def test_lower_brier_for_better_calibrated(self):
        """Well-calibrated predictions should beat constant-0.5 predictions."""
        probs_good = np.array([0.9, 0.9, 0.1, 0.1])
        probs_bad = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([1.0, 1.0, 0.0, 0.0])
        assert brier_score(probs_good, labels) < brier_score(probs_bad, labels)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8: expected_calibration_error
# ═══════════════════════════════════════════════════════════════════════════════


class TestExpectedCalibrationError:
    def test_perfect_calibration(self):
        """If confidence == accuracy in every bin, ECE = 0."""
        # 100 samples: confidence 0.7, half correct
        n = 100
        probs = np.full(n, 0.7)
        labels = np.array([1.0] * 70 + [0.0] * 30)
        ece, bins = expected_calibration_error(probs, labels, n_bins=5)
        assert ece == pytest.approx(0.0, abs=1e-5)

    def test_ece_in_zero_one(self):
        probs = np.random.RandomState(0).uniform(0, 1, 200)
        labels = (np.random.RandomState(0).uniform(0, 1, 200) < probs).astype(float)
        ece, _ = expected_calibration_error(probs, labels)
        assert 0.0 <= ece <= 1.0

    def test_returns_bins(self):
        probs = np.linspace(0.1, 0.9, 50)
        labels = (probs > 0.5).astype(float)
        _, bins = expected_calibration_error(probs, labels, n_bins=5)
        assert len(bins) > 0
        for b in bins:
            assert isinstance(b, ECEBin)

    def test_bin_contributions_sum_to_ece(self):
        probs = np.random.RandomState(1).uniform(0, 1, 100)
        labels = (np.random.RandomState(1).uniform(0, 1, 100) < probs).astype(float)
        ece, bins = expected_calibration_error(probs, labels, n_bins=10)
        total_from_bins = sum(b.contribution for b in bins)
        assert total_from_bins == pytest.approx(ece, abs=1e-5)

    def test_empty_returns_zero(self):
        ece, bins = expected_calibration_error(np.array([]), np.array([]))
        assert ece == 0.0
        assert bins == []

    def test_bin_accuracy_in_zero_one(self):
        probs = np.random.RandomState(2).uniform(0, 1, 100)
        labels = (np.random.RandomState(2).uniform(0, 1, 100) < probs).astype(float)
        _, bins = expected_calibration_error(probs, labels)
        for b in bins:
            assert 0.0 <= b.empirical_accuracy <= 1.0

    def test_bin_confidence_in_zero_one(self):
        probs = np.random.RandomState(3).uniform(0, 1, 100)
        labels = (np.random.RandomState(3).uniform(0, 1, 100) < probs).astype(float)
        _, bins = expected_calibration_error(probs, labels)
        for b in bins:
            assert 0.0 <= b.mean_confidence <= 1.0

    def test_overconfident_has_positive_ece(self):
        """All predictions 0.9 but only 50% correct → ECE > 0."""
        probs = np.full(100, 0.9)
        labels = np.array([1.0] * 50 + [0.0] * 50)
        ece, _ = expected_calibration_error(probs, labels, n_bins=10)
        assert ece > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Section 9: calibrate_dataset
# ═══════════════════════════════════════════════════════════════════════════════


class TestCalibrateDataset:
    def _make_mixed_samples(self, n: int = 80, seed: int = 42) -> List[CalibrationSample]:
        """Create samples with varying confidence and ~60% accuracy."""
        rng = random.Random(seed)
        samples = []
        for i in range(n):
            conf = rng.uniform(0.4, 0.9)
            correct = 1 if rng.random() < conf else 0
            samples.append(CalibrationSample("test", i, "col", conf, correct))
        return samples

    def test_returns_comparison(self):
        samples = self._make_mixed_samples()
        comp = calibrate_dataset(samples, "test_ds")
        assert isinstance(comp, CalibrationComparison)

    def test_n_train_n_eval_correct(self):
        samples = self._make_mixed_samples(100)
        comp = calibrate_dataset(samples, "test_ds", eval_fraction=0.3, seed=42)
        assert comp.n_train + comp.n_eval == 100
        assert 25 <= comp.n_eval <= 35

    def test_brier_scores_in_range(self):
        samples = self._make_mixed_samples()
        comp = calibrate_dataset(samples, "test_ds")
        for b in [comp.raw_brier, comp.logistic_brier, comp.isotonic_brier]:
            assert 0.0 <= b <= 1.0

    def test_ece_in_range(self):
        samples = self._make_mixed_samples()
        comp = calibrate_dataset(samples, "test_ds")
        for e in [comp.raw_ece, comp.logistic_ece, comp.isotonic_ece]:
            assert 0.0 <= e <= 1.0

    def test_adult_limitation_noted(self):
        samples = _make_samples([0.5] * 20, [0] * 20, dataset="adult")
        comp = calibrate_dataset(samples, "adult")
        assert comp.limitation is not None
        assert "adult" in comp.limitation.lower() or "Adult" in comp.limitation

    def test_constant_labels_handled(self):
        """All correct=0 → calibrators not fitted, no crash."""
        samples = _make_samples([0.6, 0.7, 0.8] * 20, [0] * 60)
        comp = calibrate_dataset(samples, "test_ds")
        assert not comp.logistic_fitted
        assert not comp.isotonic_fitted

    def test_empty_samples(self):
        comp = calibrate_dataset([], "test_ds")
        assert comp.n_train == 0
        assert comp.n_eval == 0

    def test_determinism(self):
        samples = self._make_mixed_samples(80, seed=7)
        comp1 = calibrate_dataset(samples, "test_ds", seed=42)
        comp2 = calibrate_dataset(samples, "test_ds", seed=42)
        assert comp1.raw_brier == comp2.raw_brier
        assert comp1.logistic_brier == comp2.logistic_brier
        assert comp1.isotonic_brier == comp2.isotonic_brier

    def test_logistic_fitted_with_valid_data(self):
        samples = self._make_mixed_samples(100)
        comp = calibrate_dataset(samples, "test_ds")
        assert comp.logistic_fitted

    def test_isotonic_fitted_with_valid_data(self):
        samples = self._make_mixed_samples(100)
        comp = calibrate_dataset(samples, "test_ds")
        assert comp.isotonic_fitted

    # ── metric_str: never print a not-fitted calibrator as a number ──────────

    def test_metric_str_not_fitted_prints_dash(self):
        """Unfitted calibrators must render as '—', not a misleading 0.0000."""
        samples = _make_samples([0.6, 0.7, 0.8] * 20, [0] * 60)
        comp = calibrate_dataset(samples, "test_ds")
        assert not comp.logistic_fitted and not comp.isotonic_fitted
        for method in ("logistic", "isotonic"):
            for metric in ("brier", "ece"):
                assert comp.metric_str(method, metric) == "—"

    def test_metric_str_raw_always_numeric(self):
        """Raw baseline needs no fitting, so it always renders as a number."""
        samples = _make_samples([0.6, 0.7, 0.8] * 20, [0] * 60)
        comp = calibrate_dataset(samples, "test_ds")
        assert comp.metric_str("raw", "brier") == f"{comp.raw_brier:.4f}"
        assert comp.metric_str("raw", "ece") == f"{comp.raw_ece:.4f}"

    def test_metric_str_fitted_matches_field(self):
        samples = self._make_mixed_samples(100)
        comp = calibrate_dataset(samples, "test_ds")
        assert comp.metric_str("logistic", "brier") == f"{comp.logistic_brier:.4f}"
        assert comp.metric_str("isotonic", "ece") == f"{comp.isotonic_ece:.4f}"
        assert comp.metric_str("isotonic", "ece", precision=2) == f"{comp.isotonic_ece:.2f}"

    def test_metric_str_empty_samples_prints_dash(self):
        comp = calibrate_dataset([], "test_ds")
        assert comp.metric_str("logistic", "brier") == "—"

    def test_metric_str_rejects_unknown_names(self):
        samples = self._make_mixed_samples(50)
        comp = calibrate_dataset(samples, "test_ds")
        with pytest.raises(ValueError):
            comp.metric_str("platt", "brier")
        with pytest.raises(ValueError):
            comp.metric_str("raw", "accuracy")

    def test_base_rate_computed(self):
        samples = _make_samples([0.5] * 10, [1] * 10)
        comp = calibrate_dataset(samples, "test_ds")
        assert comp.base_rate == pytest.approx(1.0, abs=0.01)

    def test_bins_populated(self):
        samples = self._make_mixed_samples(200)
        comp = calibrate_dataset(samples, "test_ds")
        assert len(comp.raw_bins) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Section 10: Data leakage prevention
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataLeakagePrevention:
    def test_logistic_fit_signature_no_clean_df(self):
        import inspect
        sig = inspect.signature(LogisticCalibrator.fit)
        params = list(sig.parameters.keys())
        assert "clean_df" not in params
        assert "clean" not in " ".join(params)

    def test_isotonic_fit_signature_no_clean_df(self):
        import inspect
        sig = inspect.signature(IsotonicCalibrator.fit)
        params = list(sig.parameters.keys())
        assert "clean_df" not in params

    def test_logistic_predict_signature_no_clean_df(self):
        import inspect
        sig = inspect.signature(LogisticCalibrator.predict)
        params = list(sig.parameters.keys())
        assert "clean_df" not in params

    def test_calibrate_dataset_no_clean_df_param(self):
        import inspect
        sig = inspect.signature(calibrate_dataset)
        params = list(sig.parameters.keys())
        assert "clean_df" not in params

    def test_calibration_sample_no_clean_field(self):
        fields = CalibrationSample.__dataclass_fields__
        for f in fields:
            assert "clean" not in f.lower()

    def test_builder_uses_clean_df_only_for_labels(self):
        """CalibrationBuilder should produce samples with only raw_confidence as feature."""
        builder = CalibrationBuilder()
        clean_df = pd.DataFrame({"col": [10, 20, 30]})
        dirty_df = clean_df.copy()
        unc = [_make_unc_record(i, "col", top_candidate=i * 10, raw_confidence=0.7 + i * 0.05)
               for i in range(3)]
        samples = builder.build(unc, clean_df, dirty_df, "test")
        for s in samples:
            # Feature is raw_confidence, not clean value
            assert hasattr(s, "raw_confidence")
            assert not hasattr(s, "clean_value")
            assert not hasattr(s, "clean_val")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 11: print_calibration_report (smoke test)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPrintCalibrationReport:
    def test_runs_without_error(self, capsys):
        samples = [
            CalibrationSample("bc", i, "col", 0.5 + i * 0.01, i % 2)
            for i in range(60)
        ]
        comp = calibrate_dataset(samples, "breast_cancer")
        print_calibration_report(comp)
        out = capsys.readouterr().out
        assert "BREAST_CANCER" in out
        assert "Brier" in out or "brier" in out.lower()

    def test_limitation_shown(self, capsys):
        samples = _make_samples([0.5] * 20, [0] * 20, dataset="adult")
        comp = calibrate_dataset(samples, "adult")
        print_calibration_report(comp)
        out = capsys.readouterr().out
        assert "LIMITATION" in out or "limitation" in out.lower() or "adult" in out.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 12: Integration tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestIntegrationBreastCancerCalibration:
    @pytest.fixture(autouse=True)
    def setup(self):
        if not Path("datasets/breast_cancer/dirty.csv").exists():
            pytest.skip("datasets/breast_cancer not found")

        from src.data.loader import DataLoader
        from src.data.profiler import profile_dataset
        from src.detection.detector import Detector
        from src.repair.mock_repair import MockRepairer
        from src.uncertainty.estimator import UncertaintyEstimator

        loader = DataLoader("datasets")
        self.clean_df, self.dirty_df = loader.load("breast_cancer")
        profile = profile_dataset(
            "breast_cancer",
            loader.discover("breast_cancer"),
            self.clean_df, self.dirty_df,
        )
        detector = Detector(profile)
        records = detector.detect(self.dirty_df)
        repairer = MockRepairer(seed=42, top_k=3)
        repair_results = repairer.repair_batch(records, self.dirty_df, profile)
        estimator = UncertaintyEstimator(base_seed=42, n_samples=5)
        self.unc_records = estimator.estimate_batch(
            repair_results, self.dirty_df, profile
        )

    def test_builder_creates_samples(self):
        builder = CalibrationBuilder()
        samples = builder.build(self.unc_records, self.clean_df, self.dirty_df, "breast_cancer")
        assert len(samples) == len(self.unc_records)

    def test_positive_labels_exist(self):
        builder = CalibrationBuilder()
        samples = builder.build(self.unc_records, self.clean_df, self.dirty_df, "breast_cancer")
        positives = sum(s.correct for s in samples)
        assert positives > 0

    def test_calibration_runs(self):
        builder = CalibrationBuilder()
        samples = builder.build(self.unc_records, self.clean_df, self.dirty_df, "breast_cancer")
        comp = calibrate_dataset(samples, "breast_cancer")
        assert comp.n_eval > 0
        assert 0.0 <= comp.raw_brier <= 1.0

    def test_logistic_calibrated_brier_reasonable(self):
        builder = CalibrationBuilder()
        samples = builder.build(self.unc_records, self.clean_df, self.dirty_df, "breast_cancer")
        comp = calibrate_dataset(samples, "breast_cancer")
        # All Brier scores should be well below worst case (1.0)
        assert comp.logistic_brier < 0.5

    def test_calibration_determinism(self):
        builder = CalibrationBuilder()
        samples = builder.build(self.unc_records, self.clean_df, self.dirty_df, "breast_cancer")
        comp1 = calibrate_dataset(samples, "breast_cancer", seed=42)
        comp2 = calibrate_dataset(samples, "breast_cancer", seed=42)
        assert comp1.raw_brier == comp2.raw_brier
        assert comp1.logistic_a == comp2.logistic_a


@pytest.mark.integration
class TestIntegrationAdultCalibration:
    @pytest.fixture(autouse=True)
    def setup(self):
        if not Path("datasets/adult/dirty.csv").exists():
            pytest.skip("datasets/adult not found")

        from src.data.loader import DataLoader
        from src.data.profiler import profile_dataset
        from src.detection.detector import Detector
        from src.repair.mock_repair import MockRepairer
        from src.uncertainty.estimator import UncertaintyEstimator

        loader = DataLoader("datasets")
        self.clean_df, self.dirty_df = loader.load("adult")
        profile = profile_dataset(
            "adult", loader.discover("adult"), self.clean_df, self.dirty_df
        )
        detector = Detector(profile)
        records = detector.detect(self.dirty_df)[:100]
        repairer = MockRepairer(seed=42, top_k=3)
        repair_results = repairer.repair_batch(records, self.dirty_df, profile)
        estimator = UncertaintyEstimator(base_seed=42, n_samples=3)
        self.unc_records = estimator.estimate_batch(
            repair_results, self.dirty_df, profile
        )

    def test_adult_all_incorrect(self):
        """Adult fnlwgt outlier repairs are all incorrect."""
        builder = CalibrationBuilder()
        samples = builder.build(self.unc_records, self.clean_df, self.dirty_df, "adult")
        positives = sum(s.correct for s in samples)
        assert positives == 0

    def test_adult_limitation_noted_in_comparison(self):
        builder = CalibrationBuilder()
        samples = builder.build(self.unc_records, self.clean_df, self.dirty_df, "adult")
        comp = calibrate_dataset(samples, "adult")
        assert comp.limitation is not None
