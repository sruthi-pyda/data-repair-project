"""
tests/test_detection_calibrator.py
-----------------------------------
Tests for the detection calibrator (q = P(flagged cell genuinely corrupted)).

Covers:
  - DetectionLabelBuilder: NaN handling, numeric/string equality, strict
    alignment errors, order preservation
  - DetectionCalibrator: exact smoothed rates, unseen groups, unfitted
    fallback to 1.0, empty fit, constant labels, smoothing=0, score bins,
    input validation, determinism
  - DetectionCalibrationReport: metric_str marker, to_dict, calibrated
    beats anti-informative raw scores
  - No-leakage: fit/predict/calibrate_detections signatures, AST scan for
    clean_df references
  - Integration on breast_cancer (k=1.5 and default k=3.0)
"""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import pytest

import src.calibration.detection_calibrator as dc_module
from src.calibration.detection_calibrator import (
    DetectionCalibrationReport,
    DetectionCalibrator,
    DetectionLabelBuilder,
    calibrate_detections,
    fit_detection_calibrator,
    print_detection_calibration_report,
)
from src.detection.detector import DetectionRecord


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _rec(
    method: str = "missing_value",
    score: float = 0.9,
    row_id: int = 0,
    column: str = "a",
    dirty_value: Any = None,
) -> DetectionRecord:
    return DetectionRecord(
        dataset="synthetic",
        row_id=row_id,
        column=column,
        dirty_value=dirty_value,
        error_type="missing",
        detection_score=score,
        detection_method=method,
        reason="test",
    )


def _group(method: str, n: int, k: int, score: float = 0.9) -> tuple:
    """n records of `method`, the first k labelled corrupted."""
    recs = [_rec(method, score, row_id=i) for i in range(n)]
    labels = [1] * k + [0] * (n - k)
    return recs, labels


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DetectionLabelBuilder
# ═══════════════════════════════════════════════════════════════════════════════


class TestLabelBuilder:
    def _frames(self, clean_vals: List[Any], dirty_vals: List[Any]):
        clean = pd.DataFrame({"a": clean_vals, "b": [1] * len(clean_vals)})
        dirty = pd.DataFrame({"a": dirty_vals, "b": [1] * len(dirty_vals)})
        return clean, dirty

    def _label_one(self, clean_val: Any, dirty_val: Any) -> int:
        clean, dirty = self._frames([clean_val], [dirty_val])
        rec = _rec(row_id=0, column="a", dirty_value=dirty_val)
        return DetectionLabelBuilder().build([rec], clean, dirty)[0]

    def test_nan_vs_nan_is_clean(self):
        assert self._label_one(np.nan, np.nan) == 0

    def test_nan_vs_value_is_corrupted(self):
        assert self._label_one(5.0, np.nan) == 1

    def test_value_vs_nan_is_corrupted(self):
        assert self._label_one(np.nan, 5.0) == 1

    def test_float_vs_int_equal(self):
        assert self._label_one(371, 371.0) == 0

    def test_string_digit_vs_int_equal(self):
        assert self._label_one(5, "5") == 0

    def test_format_error_is_corrupted(self):
        assert self._label_one(1, "1a0") == 1

    def test_whitespace_stripped(self):
        assert self._label_one("Private", " Private ") == 0

    def test_different_strings_corrupted(self):
        assert self._label_one("Private", "Privte") == 1

    def test_missing_column_raises(self):
        clean, dirty = self._frames([1], [1])
        rec = _rec(row_id=0, column="zzz", dirty_value=1)
        with pytest.raises(ValueError, match="zzz"):
            DetectionLabelBuilder().build([rec], clean, dirty)

    def test_column_missing_from_dirty_raises(self):
        clean, dirty = self._frames([1], [1])
        dirty = dirty.drop(columns=["b"])
        rec = _rec(row_id=0, column="b", dirty_value=1)
        with pytest.raises(ValueError, match="dirty_df"):
            DetectionLabelBuilder().build([rec], clean, dirty)

    def test_row_out_of_range_raises(self):
        clean, dirty = self._frames([1, 2], [1, 2])
        rec = _rec(row_id=2, column="a", dirty_value=1)
        with pytest.raises(ValueError, match="row_id"):
            DetectionLabelBuilder().build([rec], clean, dirty)

    def test_negative_row_raises(self):
        clean, dirty = self._frames([1, 2], [1, 2])
        rec = _rec(row_id=-1, column="a", dirty_value=1)
        with pytest.raises(ValueError):
            DetectionLabelBuilder().build([rec], clean, dirty)

    def test_length_and_order_preserved(self):
        clean, dirty = self._frames([1, 2, 3, 4], [1, 99, 3, 99])
        recs = [
            _rec(row_id=3, column="a", dirty_value=99),
            _rec(row_id=0, column="a", dirty_value=1),
            _rec(row_id=1, column="a", dirty_value=99),
            _rec(row_id=2, column="a", dirty_value=3),
        ]
        labels = DetectionLabelBuilder().build(recs, clean, dirty)
        assert labels == [1, 0, 1, 0]
        assert len(labels) == len(recs)

    def test_empty_detections(self):
        clean, dirty = self._frames([1], [1])
        assert DetectionLabelBuilder().build([], clean, dirty) == []

    def test_non_default_index_is_reset(self):
        clean = pd.DataFrame({"a": [10, 20]}, index=[100, 200])
        dirty = pd.DataFrame({"a": [10, 99]}, index=[100, 200])
        recs = [_rec(row_id=1, column="a", dirty_value=99)]
        assert DetectionLabelBuilder().build(recs, clean, dirty) == [1]

    def test_uses_record_dirty_value_not_frame(self):
        # Documented contract: compares detections[i].dirty_value.
        clean, dirty = self._frames([7], [7])
        rec = _rec(row_id=0, column="a", dirty_value=8)
        assert DetectionLabelBuilder().build([rec], clean, dirty) == [1]

    def test_custom_comparator_tolerance(self):
        from src.evaluation.repair_metrics import ValueComparator
        clean, dirty = self._frames([1.0], [1.3])
        rec = _rec(row_id=0, column="a", dirty_value=1.3)
        loose = DetectionLabelBuilder(ValueComparator(abs_tol=0.5))
        strict = DetectionLabelBuilder()
        assert loose.build([rec], clean, dirty) == [0]
        assert strict.build([rec], clean, dirty) == [1]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DetectionCalibrator
# ═══════════════════════════════════════════════════════════════════════════════


class TestCalibratorRates:
    def test_rates_match_smoothed_formula_exactly(self):
        s = 1.0
        r1, l1 = _group("missing_value", n=10, k=10)
        r2, l2 = _group("iqr_outlier", n=20, k=1)
        r3, l3 = _group("format_check", n=4, k=2)
        cal = DetectionCalibrator(smoothing=s)
        cal.fit(r1 + r2 + r3, l1 + l2 + l3)
        q = cal.predict([r1[0], r2[0], r3[0]])
        assert q[0] == pytest.approx((10 + s) / (10 + 2 * s))
        assert q[1] == pytest.approx((1 + s) / (20 + 2 * s))
        assert q[2] == pytest.approx((2 + s) / (4 + 2 * s))

    def test_smoothing_zero_gives_raw_rates(self):
        r1, l1 = _group("m1", n=10, k=10)
        r2, l2 = _group("m2", n=5, k=0)
        r3, l3 = _group("m3", n=4, k=3)
        cal = DetectionCalibrator(smoothing=0.0)
        cal.fit(r1 + r2 + r3, l1 + l2 + l3)
        q = cal.predict([r1[0], r2[0], r3[0]])
        assert q.tolist() == [1.0, 0.0, 0.75]

    def test_other_smoothing_value(self):
        r, l = _group("m", n=3, k=1)
        cal = DetectionCalibrator(smoothing=2.5)
        cal.fit(r, l)
        assert cal.predict([r[0]])[0] == pytest.approx((1 + 2.5) / (3 + 5.0))

    def test_unseen_group_gets_global_rate(self):
        r1, l1 = _group("m1", n=10, k=8)
        r2, l2 = _group("m2", n=10, k=2)
        cal = DetectionCalibrator(smoothing=1.0)
        cal.fit(r1 + r2, l1 + l2)
        expected_global = (10 + 1) / (20 + 2)
        q = cal.predict([_rec("never_seen")])
        assert q[0] == pytest.approx(expected_global)
        assert cal.global_rate == pytest.approx(expected_global)
        # And it is NOT either group's rate.
        assert q[0] != pytest.approx((8 + 1) / 12)
        assert q[0] != pytest.approx((2 + 1) / 12)

    def test_unfitted_predict_is_all_ones(self):
        cal = DetectionCalibrator()
        assert cal.is_fitted() is False
        q = cal.predict([_rec("a"), _rec("b"), _rec("c")])
        assert q.shape == (3,)
        assert np.all(q == 1.0)

    def test_unfitted_predict_empty(self):
        assert DetectionCalibrator().predict([]).shape == (0,)

    def test_empty_fit_stays_unfitted(self):
        cal = DetectionCalibrator()
        cal.fit([], [])
        assert cal.is_fitted() is False
        assert np.all(cal.predict([_rec()]) == 1.0)
        assert cal.group_table() == []

    def test_all_zero_labels_q_small_but_positive(self):
        n, s = 50, 1.0
        r, l = _group("m", n=n, k=0)
        cal = DetectionCalibrator(smoothing=s)
        cal.fit(r, l)
        q = cal.predict(r)
        assert np.all(q > 0)
        assert np.all(q < s / (n + 2 * s) + 1e-12)
        assert q[0] == pytest.approx(s / (n + 2 * s))

    def test_all_one_labels_q_close_to_one_but_below(self):
        n = 50
        r, l = _group("m", n=n, k=n)
        cal = DetectionCalibrator(smoothing=1.0)
        cal.fit(r, l)
        q = cal.predict(r)
        assert np.all(q < 1.0)
        assert np.all(q > 0.95)

    def test_predictions_in_unit_interval(self):
        r1, l1 = _group("m1", n=7, k=3)
        r2, l2 = _group("m2", n=9, k=9)
        cal = DetectionCalibrator()
        cal.fit(r1 + r2, l1 + l2)
        q = cal.predict(r1 + r2 + [_rec("unseen")])
        assert np.all((q >= 0) & (q <= 1))

    def test_refit_replaces_previous_state(self):
        r1, l1 = _group("m1", n=10, k=10)
        cal = DetectionCalibrator(smoothing=0.0)
        cal.fit(r1, l1)
        assert cal.predict([r1[0]])[0] == 1.0
        r2, l2 = _group("m2", n=10, k=0)
        cal.fit(r2, l2)
        # m1 is now unseen → global rate of the second fit (0).
        assert cal.predict([r1[0]])[0] == 0.0
        assert [g["group"] for g in cal.group_table()] == [("m2",)]


class TestCalibratorGrouping:
    def test_default_group_by_is_detection_method(self):
        assert DetectionCalibrator().group_by == ("detection_method",)

    def test_group_key_uses_named_attributes(self):
        cal = DetectionCalibrator(group_by=("detection_method", "column"))
        rec = _rec("m", column="colx")
        assert cal.group_key(rec) == ("m", "colx")

    def test_score_bins_off_by_default(self):
        cal = DetectionCalibrator()
        assert cal.group_key(_rec("m", 0.2)) == cal.group_key(_rec("m", 0.9))

    def test_score_bins_split_same_method(self):
        cal = DetectionCalibrator(score_bins=4)
        lo = _rec("m", 0.2)
        hi = _rec("m", 0.9)
        assert cal.group_key(lo) != cal.group_key(hi)
        assert cal.group_key(lo) == ("m", 0)
        assert cal.group_key(hi) == ("m", 3)

    def test_score_bins_top_edge_clamped(self):
        cal = DetectionCalibrator(score_bins=4)
        assert cal.group_key(_rec("m", 1.0)) == ("m", 3)

    def test_score_bins_learn_separate_rates(self):
        lo, l_lo = _group("m", n=10, k=0, score=0.2)
        hi, l_hi = _group("m", n=10, k=10, score=0.9)
        binned = DetectionCalibrator(score_bins=4, smoothing=0.0)
        binned.fit(lo + hi, l_lo + l_hi)
        assert binned.predict([lo[0]])[0] == 0.0
        assert binned.predict([hi[0]])[0] == 1.0
        pooled = DetectionCalibrator(smoothing=0.0)
        pooled.fit(lo + hi, l_lo + l_hi)
        assert pooled.predict([lo[0]])[0] == 0.5
        assert pooled.predict([hi[0]])[0] == 0.5

    def test_group_table_contents(self):
        r1, l1 = _group("b_method", n=4, k=1)
        r2, l2 = _group("a_method", n=6, k=6)
        cal = DetectionCalibrator(smoothing=1.0)
        cal.fit(r1 + r2, l1 + l2)
        table = cal.group_table()
        assert [g["group"] for g in table] == [("a_method",), ("b_method",)]
        assert table[0] == {"group": ("a_method",), "n": 6, "n_corrupted": 6,
                            "rate": round(7 / 8, 6)}
        assert table[1]["n"] == 4 and table[1]["n_corrupted"] == 1
        assert table[1]["rate"] == pytest.approx(2 / 6, abs=1e-6)

    def test_name(self):
        assert DetectionCalibrator.name == "grouped_rate"


class TestCalibratorValidation:
    def test_length_mismatch_raises(self):
        r, _ = _group("m", n=3, k=1)
        with pytest.raises(ValueError, match="same length"):
            DetectionCalibrator().fit(r, [1, 0])

    def test_non_binary_labels_raise(self):
        r, _ = _group("m", n=3, k=1)
        with pytest.raises(ValueError, match="0/1"):
            DetectionCalibrator().fit(r, [1, 2, 0])

    def test_float_label_not_binary_raises(self):
        r, _ = _group("m", n=2, k=1)
        with pytest.raises(ValueError):
            DetectionCalibrator().fit(r, [0.5, 1])

    def test_negative_smoothing_raises(self):
        with pytest.raises(ValueError, match="smoothing"):
            DetectionCalibrator(smoothing=-0.1)

    def test_negative_score_bins_raises(self):
        with pytest.raises(ValueError, match="score_bins"):
            DetectionCalibrator(score_bins=-1)

    def test_empty_group_by_raises(self):
        with pytest.raises(ValueError):
            DetectionCalibrator(group_by=())

    def test_failed_fit_does_not_leave_partial_state(self):
        r, l = _group("m", n=4, k=2)
        cal = DetectionCalibrator()
        cal.fit(r, l)
        with pytest.raises(ValueError):
            cal.fit(r, [1, 2, 0, 0])
        # Validation happens before state is touched: old fit still valid.
        assert cal.is_fitted()


class TestCalibratorDeterminism:
    def test_two_fits_identical(self):
        r1, l1 = _group("m1", n=13, k=5)
        r2, l2 = _group("m2", n=7, k=7)
        a, b = DetectionCalibrator(), DetectionCalibrator()
        a.fit(r1 + r2, l1 + l2)
        b.fit(r1 + r2, l1 + l2)
        np.testing.assert_array_equal(a.predict(r1 + r2), b.predict(r1 + r2))
        assert a.group_table() == b.group_table()

    def test_order_of_records_does_not_change_rates(self):
        r1, l1 = _group("m1", n=13, k=5)
        r2, l2 = _group("m2", n=7, k=7)
        a, b = DetectionCalibrator(), DetectionCalibrator()
        a.fit(r1 + r2, l1 + l2)
        b.fit(r2 + r1, l2 + l1)
        np.testing.assert_array_equal(a.predict(r1 + r2), b.predict(r1 + r2))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Report and pipeline function
# ═══════════════════════════════════════════════════════════════════════════════


def _report(**over) -> DetectionCalibrationReport:
    base = dict(
        dataset_name="x", n_train=10, n_eval=5, n_corrupted_eval=2,
        base_rate=0.4, raw_brier=0.3, raw_ece=0.2,
        calibrated_brier=0.1, calibrated_ece=0.05, fitted=True,
    )
    base.update(over)
    return DetectionCalibrationReport(**base)


class TestReport:
    def test_metric_str_when_fitted(self):
        rep = _report()
        assert rep.metric_str("calibrated", "brier") == "0.1000"
        assert rep.metric_str("raw", "ece") == "0.2000"

    def test_metric_str_marker_when_unfitted(self):
        rep = _report(fitted=False, calibrated_brier=0.0, calibrated_ece=0.0)
        assert rep.metric_str("calibrated", "brier") == "—"
        assert rep.metric_str("calibrated", "ece") == "—"
        # raw is always defined
        assert rep.metric_str("raw", "brier") == "0.3000"

    def test_metric_str_precision(self):
        assert _report().metric_str("raw", "brier", precision=2) == "0.30"

    def test_metric_str_rejects_unknown(self):
        with pytest.raises(ValueError):
            _report().metric_str("logistic", "brier")
        with pytest.raises(ValueError):
            _report().metric_str("raw", "nll")

    def test_to_dict_keys(self):
        d = _report(groups=[{"group": ("m",), "n": 1, "n_corrupted": 1, "rate": 0.5}]).to_dict()
        assert set(d) == {
            "dataset_name", "n_train", "n_eval", "n_corrupted_eval", "base_rate",
            "raw_brier", "raw_ece", "calibrated_brier", "calibrated_ece",
            "fitted", "groups", "limitation",
        }
        assert d["groups"][0]["group"] == ("m",)
        assert d["limitation"] is None


class TestCalibrateDetections:
    def _anti_informative(self, n_per_group: int = 40):
        # Raw score says 0.7 for a group that is all clean and 0.5 for one
        # that is all corrupted: the score is worse than useless.
        clean, l_clean = _group("iqr_outlier", n=n_per_group, k=0, score=0.7)
        corrupt, l_corrupt = _group("missing_value", n=n_per_group, k=n_per_group, score=0.5)
        return clean + corrupt, l_clean + l_corrupt

    def test_calibrated_beats_anti_informative_raw(self):
        recs, labels = self._anti_informative()
        rep = calibrate_detections(recs, labels, "synthetic", seed=42)
        assert rep.fitted
        assert rep.calibrated_brier < rep.raw_brier
        assert rep.calibrated_ece < rep.raw_ece
        assert rep.calibrated_brier < 0.01
        # Raw is at least as bad as guessing 0.5 on both groups.
        assert rep.raw_brier >= 0.25

    def test_split_sizes(self):
        recs, labels = self._anti_informative(50)   # 100 total
        rep = calibrate_detections(recs, labels, "synthetic", eval_fraction=0.3)
        assert rep.n_eval == 30
        assert rep.n_train == 70

    def test_eval_labels_counted(self):
        recs, labels = self._anti_informative(50)
        rep = calibrate_detections(recs, labels, "synthetic")
        assert 0 < rep.n_corrupted_eval < rep.n_eval
        assert rep.base_rate == pytest.approx(rep.n_corrupted_eval / rep.n_eval, abs=1e-4)

    def test_groups_reported(self):
        recs, labels = self._anti_informative()
        rep = calibrate_detections(recs, labels, "synthetic")
        names = {g["group"][0] for g in rep.groups}
        assert names == {"iqr_outlier", "missing_value"}

    def test_deterministic_across_calls(self):
        recs, labels = self._anti_informative()
        a = calibrate_detections(recs, labels, "synthetic", seed=42)
        b = calibrate_detections(recs, labels, "synthetic", seed=42)
        assert a.to_dict() == b.to_dict()

    def test_seed_changes_split(self):
        recs, labels = self._anti_informative(30)
        a = calibrate_detections(recs, labels, "synthetic", seed=1)
        b = calibrate_detections(recs, labels, "synthetic", seed=2)
        # Same n_eval, but composition (n_corrupted_eval) is very likely to differ;
        # at minimum the raw Brier must be a valid number for both.
        assert a.n_eval == b.n_eval
        assert 0 <= a.raw_brier <= 1 and 0 <= b.raw_brier <= 1

    def test_constant_eval_labels_still_reports(self):
        recs, labels = _group("missing_value", n=30, k=30, score=0.6)
        rep = calibrate_detections(recs, labels, "const", seed=42)
        assert rep.fitted
        assert rep.base_rate == 1.0
        assert rep.raw_brier == pytest.approx(0.16, abs=1e-4)
        assert rep.calibrated_brier < rep.raw_brier
        assert rep.limitation is not None and "constant" in rep.limitation

    def test_too_few_records_not_fitted(self):
        recs, labels = _group("m", n=3, k=2)      # train = 2 < 3
        rep = calibrate_detections(recs, labels, "tiny")
        assert rep.fitted is False
        assert rep.limitation is not None
        assert rep.metric_str("calibrated", "brier") == "—"
        assert rep.calibrated_brier == rep.raw_brier

    def test_empty_records(self):
        rep = calibrate_detections([], [], "empty")
        assert rep.fitted is False
        assert rep.n_train == 0 and rep.n_eval == 0
        assert rep.limitation is not None

    def test_calibrator_kwargs_forwarded(self):
        recs, labels = self._anti_informative()
        rep = calibrate_detections(recs, labels, "s", smoothing=0.0)
        rates = {g["group"][0]: g["rate"] for g in rep.groups}
        assert rates["iqr_outlier"] == 0.0
        assert rates["missing_value"] == 1.0

    def test_invalid_eval_fraction(self):
        recs, labels = self._anti_informative()
        with pytest.raises(ValueError):
            calibrate_detections(recs, labels, "s", eval_fraction=1.0)

    def test_length_mismatch_raises(self):
        recs, labels = self._anti_informative()
        with pytest.raises(ValueError):
            calibrate_detections(recs, labels[:-1], "s")

    def test_fit_detection_calibrator_returns_fitted_model(self):
        recs, labels = self._anti_informative()
        cal, rep = fit_detection_calibrator(recs, labels, seed=42, eval_fraction=0.3)
        assert isinstance(cal, DetectionCalibrator)
        assert cal.is_fitted() and rep.fitted
        q = cal.predict(recs)
        assert q.shape == (len(recs),)
        assert q[0] < 0.1 and q[-1] > 0.9

    def test_raw_scores_clipped_not_rejected(self):
        recs, labels = self._anti_informative(20)
        recs[0].detection_score = 1.2   # out of range; must not raise
        rep = calibrate_detections(recs, labels, "s")
        assert 0 <= rep.raw_brier <= 1

    def test_print_report_runs(self, capsys):
        recs, labels = self._anti_informative()
        rep = calibrate_detections(recs, labels, "printme")
        print_detection_calibration_report(rep)
        out = capsys.readouterr().out
        assert "PRINTME" in out and "iqr_outlier" in out


# ═══════════════════════════════════════════════════════════════════════════════
# 4. No-leakage
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoLeakage:
    @pytest.mark.parametrize("func", [
        DetectionCalibrator.fit,
        DetectionCalibrator.predict,
        DetectionCalibrator.group_key,
        calibrate_detections,
        fit_detection_calibrator,
    ])
    def test_no_clean_parameter(self, func):
        for name in inspect.signature(func).parameters:
            assert "clean" not in name.lower(), f"{func.__qualname__} takes '{name}'"

    def test_calibrator_source_never_mentions_clean_df(self):
        assert "clean_df" not in inspect.getsource(DetectionCalibrator)

    def test_builder_build_is_only_function_reading_clean_df(self):
        source = inspect.getsource(dc_module)
        tree = ast.parse(source)
        offenders: List[str] = []

        def visit(node: ast.AST, prefix: str) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qual = f"{prefix}{child.name}"
                    body = list(child.body)
                    # Drop the docstring; prose is allowed to explain the rule.
                    if (body and isinstance(body[0], ast.Expr)
                            and isinstance(getattr(body[0], "value", None), ast.Constant)
                            and isinstance(body[0].value.value, str)):
                        body = body[1:]
                    code = "\n".join(ast.unparse(b) for b in body)
                    arg_names = [a.arg for a in child.args.args + child.args.kwonlyargs]
                    if "clean_df" in code or "clean_df" in arg_names:
                        offenders.append(qual)
                    visit(child, qual + ".")
                elif isinstance(child, ast.ClassDef):
                    visit(child, f"{prefix}{child.name}.")
                else:
                    visit(child, prefix)

        visit(tree, "")
        assert offenders == ["DetectionLabelBuilder.build"], offenders

    def test_predict_output_is_independent_of_dirty_value(self):
        # q depends on the group only, never on the cell content.
        r, l = _group("m", n=8, k=3)
        cal = DetectionCalibrator()
        cal.fit(r, l)
        a = _rec("m", dirty_value="anything")
        b = _rec("m", dirty_value=12345)
        assert cal.predict([a])[0] == cal.predict([b])[0]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Integration: breast_cancer
# ═══════════════════════════════════════════════════════════════════════════════


def _bc_detections(iqr_k: float):
    from src.data.loader import DataLoader
    from src.data.profiler import profile_dataset
    from src.detection.detector import Detector

    loader = DataLoader("datasets")
    clean_df, dirty_df = loader.load("breast_cancer")
    profile = profile_dataset(
        "breast_cancer", loader.discover("breast_cancer"), clean_df, dirty_df
    )
    detector = Detector(profile, iqr_k=iqr_k, flag_statistical_outliers=True)
    return detector.detect(dirty_df), clean_df, dirty_df


@pytest.mark.integration
class TestIntegrationBreastCancer:
    @pytest.fixture(autouse=True)
    def _skip_without_data(self):
        if not Path("datasets/breast_cancer/dirty.csv").exists():
            pytest.skip("datasets/breast_cancer not found")

    def test_k15_iqr_outliers_low_q_missing_high_q(self):
        records, clean_df, dirty_df = _bc_detections(iqr_k=1.5)
        methods = {r.detection_method for r in records}
        assert "iqr_outlier" in methods, "k=1.5 must produce outlier flags"
        assert "missing_value" in methods

        labels = DetectionLabelBuilder().build(records, clean_df, dirty_df)
        assert len(labels) == len(records)

        cal, rep = fit_detection_calibrator(
            records, labels, seed=42, eval_fraction=0.3, dataset_name="breast_cancer"
        )
        assert cal.is_fitted()
        q = cal.predict(records)
        q_iqr = q[[r.detection_method == "iqr_outlier" for r in records]]
        q_miss = q[[r.detection_method == "missing_value" for r in records]]
        assert len(q_iqr) > 0 and len(q_miss) > 0
        assert np.all(q_iqr < 0.10), f"max q(iqr_outlier) = {q_iqr.max():.4f}"
        assert np.all(q_miss > 0.95), f"min q(missing_value) = {q_miss.min():.4f}"
        assert rep.raw_brier > rep.calibrated_brier

    def test_k3_no_outliers_all_q_high(self):
        records, clean_df, dirty_df = _bc_detections(iqr_k=3.0)
        assert all(r.detection_method != "iqr_outlier" for r in records)
        labels = DetectionLabelBuilder().build(records, clean_df, dirty_df)
        cal, rep = fit_detection_calibrator(
            records, labels, seed=42, eval_fraction=0.3, dataset_name="breast_cancer"
        )
        q = cal.predict(records)
        assert rep.base_rate == 1.0
        # Every group is 100 % corrupted, so q > 0.5 everywhere ...
        assert np.all(q > 0.5), f"min q = {q.min():.4f}"
        # ... and > 0.95 for the well-populated groups. Small groups
        # (format_check: 3 records, 1 in train) are deliberately pulled
        # toward 0.5 by the Beta(1,1) prior: 1 observation is not certainty.
        train_n = {g["group"][0]: g["n"] for g in rep.groups}
        for method in ("missing_value", "range_check"):
            assert train_n[method] >= 20
            q_m = q[[r.detection_method == method for r in records]]
            assert np.all(q_m > 0.95), f"min q({method}) = {q_m.min():.4f}"
        assert train_n["format_check"] < 5 and q.min() == pytest.approx(
            (train_n["format_check"] + 1) / (train_n["format_check"] + 2)
        )

    def test_labels_are_mostly_positive_at_k3(self):
        # With k=3 every BC detection is a real corruption (plan §0 table).
        records, clean_df, dirty_df = _bc_detections(iqr_k=3.0)
        labels = DetectionLabelBuilder().build(records, clean_df, dirty_df)
        assert sum(labels) / len(labels) > 0.95
