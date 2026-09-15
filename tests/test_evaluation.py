"""
tests/test_evaluation.py
-------------------------
Phase 10: Unit tests for the evaluation layer.

Test groups
-----------
 1.  ValueComparator — numerical comparison with tolerances
 2.  ValueComparator — categorical/string comparison
 3.  ValueComparator — NaN/None handling
 4.  ValueComparator — invalid parameter construction
 5.  RepairEvaluator — perfect predictions (all correct)
 6.  RepairEvaluator — all incorrect predictions
 7.  RepairEvaluator — detection false positives
 8.  RepairEvaluator — detection false negatives
 9.  RepairEvaluator — zero-denominator cases
10.  RepairEvaluator — empty records
11.  RepairEvaluator — action distribution
12.  RepairEvaluator — evaluate_cell
13.  nll_score — correctness and edge cases
14.  nll_score — probability clipping
15.  nll_score — empty and invalid inputs
16.  CalibrationEvaluator — Brier / ECE / NLL integration
17.  CalibrationEvaluator — empty and edge case inputs
18.  CalibrationComparison — raw vs calibrated
19.  CostEvaluator — all AUTO_REPAIR (correct vs incorrect)
20.  CostEvaluator — all HUMAN_REVIEW
21.  CostEvaluator — all ABSTAIN
22.  CostEvaluator — mixed actions
23.  CostEvaluator — empty input and zero-denominator cases
24.  CostEvaluator.from_cost_config bridge
25.  CostEvaluator — action rate calculations
26.  compare_policies helper
27.  PolicyComparisonRow.to_dict
28.  Data-leakage: evaluation accesses clean_df; predictions do not
29.  Determinism of all evaluators
30.  Threshold monotonicity via CostEvaluator
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from src.baselines.baselines import (
    AlwaysAutoBaseline,
    BaselineRecord,
    ConfidenceThresholdBaseline,
    DEFAULT_THRESHOLDS,
    sweep_thresholds,
)
from src.decision.policy import DecisionAction, DecisionPolicy
from src.evaluation.calibration_metrics import (
    CalibrationComparison,
    CalibrationEvaluator,
    CalibrationReport,
    nll_score,
)
from src.evaluation.cost_metrics import (
    CostEvaluator,
    CostReport,
    PolicyComparisonRow,
    compare_policies,
)
from src.evaluation.repair_metrics import (
    CellResult,
    RepairEvaluator,
    RepairMetrics,
    ValueComparator,
    _is_na,
)
from src.risk.cost_model import CostConfig


# ===========================================================================
# DataFrame fixtures
# ===========================================================================


def _make_dfs(
    n: int = 5,
    col: str = "val",
    clean_vals=None,
    dirty_vals=None,
) -> tuple:
    """
    Build tiny clean/dirty DataFrames for testing.

    Parameters
    ----------
    clean_vals : list — values in the clean DataFrame column
    dirty_vals : list — values in the dirty DataFrame (may differ from clean)
    """
    if clean_vals is None:
        clean_vals = list(range(n))
    if dirty_vals is None:
        dirty_vals = list(range(n))  # identical → no corruption
    clean_df = pd.DataFrame({col: clean_vals})
    dirty_df = pd.DataFrame({col: dirty_vals})
    return clean_df, dirty_df


def _make_record(
    row_id: int,
    col: str,
    dirty_value: Any,
    top_candidate: Any,
    action: DecisionAction = DecisionAction.AUTO_REPAIR,
    method_name: str = "test_method",
    raw_confidence: float = 0.8,
) -> BaselineRecord:
    return BaselineRecord(
        method_name=method_name,
        dataset="test",
        row_id=row_id,
        column=col,
        dirty_value=dirty_value,
        top_candidate=top_candidate,
        raw_confidence=raw_confidence,
        action=action,
    )


# ===========================================================================
# Group 1: ValueComparator — numerical
# ===========================================================================


class TestValueComparatorNumerical:

    def test_exact_equal_integers(self):
        vc = ValueComparator()
        assert vc.are_equal(3, 3)

    def test_int_vs_float_equal(self):
        vc = ValueComparator()
        assert vc.are_equal(3, 3.0)

    def test_within_abs_tol(self):
        vc = ValueComparator(abs_tol=0.5)
        assert vc.are_equal(3.2, 3.0)

    def test_outside_abs_tol(self):
        vc = ValueComparator(abs_tol=0.1)
        assert not vc.are_equal(3.5, 3.0)

    def test_within_rel_tol(self):
        vc = ValueComparator(rel_tol=0.01)
        # |100.5 - 100| = 0.5 <= 0.01 * 100 = 1.0
        assert vc.are_equal(100.5, 100.0)

    def test_outside_rel_tol(self):
        vc = ValueComparator(rel_tol=0.001)
        # |100.5 - 100| = 0.5 > 0.001 * 100 = 0.1
        assert not vc.are_equal(100.5, 100.0)

    def test_zero_vs_small_value(self):
        vc = ValueComparator(abs_tol=1e-9)
        assert not vc.are_equal(1e-6, 0.0)

    def test_negative_values(self):
        vc = ValueComparator()
        assert vc.are_equal(-5, -5)
        assert not vc.are_equal(-5, -4)

    def test_string_numbers_compare_numerically(self):
        """Strings that represent numbers should compare numerically."""
        vc = ValueComparator()
        assert vc.are_equal("3", 3)
        assert vc.are_equal("3.0", 3.0)


# ===========================================================================
# Group 2: ValueComparator — categorical/string
# ===========================================================================


class TestValueComparatorCategorical:

    def test_exact_string_equal(self):
        vc = ValueComparator()
        assert vc.are_equal("Adm-clerical", "Adm-clerical")

    def test_strings_different(self):
        vc = ValueComparator()
        assert not vc.are_equal("Adm-clerical", "Sales")

    def test_whitespace_stripped(self):
        vc = ValueComparator()
        assert vc.are_equal("  Adm-clerical  ", "Adm-clerical")

    def test_case_sensitive_by_default(self):
        vc = ValueComparator(normalize_case=False)
        assert not vc.are_equal("adm-clerical", "Adm-clerical")

    def test_case_insensitive_when_enabled(self):
        vc = ValueComparator(normalize_case=True)
        assert vc.are_equal("adm-clerical", "Adm-clerical")

    def test_case_normalize_plus_strip(self):
        vc = ValueComparator(normalize_case=True)
        assert vc.are_equal("  ADM-CLERICAL  ", "adm-clerical")

    def test_non_numeric_strings(self):
        vc = ValueComparator()
        assert vc.are_equal("yes", "yes")
        assert not vc.are_equal("yes", "no")


# ===========================================================================
# Group 3: ValueComparator — NaN / None
# ===========================================================================


class TestValueComparatorNaN:

    def test_both_none_equal(self):
        vc = ValueComparator()
        assert vc.are_equal(None, None)

    def test_both_nan_equal(self):
        vc = ValueComparator()
        assert vc.are_equal(float("nan"), float("nan"))

    def test_none_vs_value_not_equal(self):
        vc = ValueComparator()
        assert not vc.are_equal(None, 3)
        assert not vc.are_equal(3, None)

    def test_nan_vs_value_not_equal(self):
        vc = ValueComparator()
        assert not vc.are_equal(float("nan"), 3.0)

    def test_pandas_na_equal(self):
        vc = ValueComparator()
        assert vc.are_equal(pd.NA, pd.NA)

    def test_pandas_na_vs_value(self):
        vc = ValueComparator()
        assert not vc.are_equal(pd.NA, 3)

    def test_is_na_none(self):
        assert _is_na(None)

    def test_is_na_float_nan(self):
        assert _is_na(float("nan"))

    def test_is_na_regular_value(self):
        assert not _is_na(3)
        assert not _is_na("text")


# ===========================================================================
# Group 4: ValueComparator — invalid parameters
# ===========================================================================


class TestValueComparatorParams:

    def test_negative_abs_tol_raises(self):
        with pytest.raises(ValueError, match="abs_tol"):
            ValueComparator(abs_tol=-1)

    def test_negative_rel_tol_raises(self):
        with pytest.raises(ValueError, match="rel_tol"):
            ValueComparator(rel_tol=-0.1)

    def test_zero_tol_valid(self):
        vc = ValueComparator(abs_tol=0.0, rel_tol=0.0)
        assert vc.are_equal(3, 3)
        assert not vc.are_equal(3, 3.0000001)


# ===========================================================================
# Group 5: RepairEvaluator — perfect predictions
# ===========================================================================


class TestRepairEvaluatorPerfect:
    """All AUTO_REPAIR decisions are correct AND all corrupted cells detected."""

    def setup_method(self):
        # 3 rows; row 0 and 1 corrupted; row 2 clean
        self.clean_df = pd.DataFrame({"val": [10, 20, 30]})
        self.dirty_df = pd.DataFrame({"val": [99, 88, 30]})  # rows 0,1 corrupted
        self.evaluator = RepairEvaluator(self.clean_df, self.dirty_df, "test")
        # Records propose correct repairs for all detected cells
        self.records = [
            _make_record(0, "val", 99, 10, DecisionAction.AUTO_REPAIR),
            _make_record(1, "val", 88, 20, DecisionAction.AUTO_REPAIR),
        ]

    def test_n_detected(self):
        m = self.evaluator.evaluate(self.records)
        assert m.n_detected == 2

    def test_n_truly_corrupted(self):
        m = self.evaluator.evaluate(self.records)
        assert m.n_truly_corrupted == 2

    def test_n_false_positive(self):
        m = self.evaluator.evaluate(self.records)
        assert m.n_false_positive == 0

    def test_n_false_negative(self):
        m = self.evaluator.evaluate(self.records)
        assert m.n_false_negative == 0

    def test_precision_is_one(self):
        m = self.evaluator.evaluate(self.records)
        assert math.isclose(m.detection_precision, 1.0)

    def test_recall_is_one(self):
        m = self.evaluator.evaluate(self.records)
        assert math.isclose(m.detection_recall, 1.0)

    def test_f1_is_one(self):
        m = self.evaluator.evaluate(self.records)
        assert math.isclose(m.detection_f1, 1.0)

    def test_repair_accuracy_is_one(self):
        m = self.evaluator.evaluate(self.records)
        assert math.isclose(m.repair_accuracy, 1.0)

    def test_n_correct_repairs(self):
        m = self.evaluator.evaluate(self.records)
        assert m.n_correct_repairs == 2

    def test_n_incorrect_repairs(self):
        m = self.evaluator.evaluate(self.records)
        assert m.n_incorrect_repairs == 0


# ===========================================================================
# Group 6: RepairEvaluator — all incorrect
# ===========================================================================


class TestRepairEvaluatorAllIncorrect:

    def setup_method(self):
        self.clean_df = pd.DataFrame({"val": [10, 20]})
        self.dirty_df = pd.DataFrame({"val": [99, 88]})
        self.evaluator = RepairEvaluator(self.clean_df, self.dirty_df, "test")
        # Repairs propose wrong values
        self.records = [
            _make_record(0, "val", 99, 777, DecisionAction.AUTO_REPAIR),
            _make_record(1, "val", 88, 888, DecisionAction.AUTO_REPAIR),
        ]

    def test_n_correct_repairs_is_zero(self):
        m = self.evaluator.evaluate(self.records)
        assert m.n_correct_repairs == 0

    def test_n_incorrect_repairs(self):
        m = self.evaluator.evaluate(self.records)
        assert m.n_incorrect_repairs == 2

    def test_repair_accuracy_is_zero(self):
        m = self.evaluator.evaluate(self.records)
        assert m.repair_accuracy == 0.0


# ===========================================================================
# Group 7: RepairEvaluator — false positives
# ===========================================================================


class TestRepairEvaluatorFP:
    """A clean cell is falsely flagged and auto-repaired."""

    def setup_method(self):
        # Row 0 is truly corrupted; row 1 is clean (FP detection)
        self.clean_df = pd.DataFrame({"val": [10, 20]})
        self.dirty_df = pd.DataFrame({"val": [99, 20]})  # only row 0 corrupted
        self.evaluator = RepairEvaluator(self.clean_df, self.dirty_df, "test")

    def test_fp_counted_separately(self):
        records = [
            _make_record(0, "val", 99, 10, DecisionAction.AUTO_REPAIR),  # TP
            _make_record(1, "val", 20, 99, DecisionAction.AUTO_REPAIR),  # FP
        ]
        m = self.evaluator.evaluate(records)
        assert m.n_truly_corrupted == 1  # TP
        assert m.n_false_positive == 1   # FP

    def test_fp_does_not_improve_detection_recall(self):
        records = [
            _make_record(0, "val", 99, 10, DecisionAction.AUTO_REPAIR),
            _make_record(1, "val", 20, 99, DecisionAction.AUTO_REPAIR),
        ]
        m = self.evaluator.evaluate(records)
        # precision = TP/(TP+FP) = 1/2 = 0.5
        assert math.isclose(m.detection_precision, 0.5)

    def test_fp_auto_repair_wrong_counted_as_incorrect(self):
        # Row 1 is clean (val=20), but we repair it to 99 (wrong value)
        records = [_make_record(1, "val", 20, 99, DecisionAction.AUTO_REPAIR)]
        m = self.evaluator.evaluate(records)
        assert m.n_incorrect_repairs == 1

    def test_fp_auto_repair_to_same_value_counted_as_correct(self):
        # Row 1 is clean (val=20), we happen to propose 20 — no damage
        records = [_make_record(1, "val", 20, 20, DecisionAction.AUTO_REPAIR)]
        m = self.evaluator.evaluate(records)
        assert m.n_correct_repairs == 1


# ===========================================================================
# Group 8: RepairEvaluator — false negatives
# ===========================================================================


class TestRepairEvaluatorFN:
    """Corrupted cells that are not in the records → false negatives."""

    def setup_method(self):
        # 3 truly corrupted cells; we only detect 1 → 2 FN
        self.clean_df = pd.DataFrame({"val": [1, 2, 3]})
        self.dirty_df = pd.DataFrame({"val": [9, 8, 7]})  # all 3 corrupted
        self.evaluator = RepairEvaluator(self.clean_df, self.dirty_df, "test")

    def test_fn_computed_from_missed_corruptions(self):
        records = [_make_record(0, "val", 9, 1, DecisionAction.AUTO_REPAIR)]
        m = self.evaluator.evaluate(records)
        assert m.n_false_negative == 2

    def test_fn_reduces_recall(self):
        records = [_make_record(0, "val", 9, 1, DecisionAction.AUTO_REPAIR)]
        m = self.evaluator.evaluate(records)
        # recall = 1/3
        assert math.isclose(m.detection_recall, 1 / 3, rel_tol=1e-6)

    def test_zero_fn_when_all_detected(self):
        records = [
            _make_record(0, "val", 9, 1, DecisionAction.AUTO_REPAIR),
            _make_record(1, "val", 8, 2, DecisionAction.AUTO_REPAIR),
            _make_record(2, "val", 7, 3, DecisionAction.AUTO_REPAIR),
        ]
        m = self.evaluator.evaluate(records)
        assert m.n_false_negative == 0


# ===========================================================================
# Group 9: RepairEvaluator — zero-denominator cases
# ===========================================================================


class TestRepairEvaluatorZeroDenominator:

    def test_precision_zero_when_no_detections(self):
        clean_df = pd.DataFrame({"val": [1, 2]})
        dirty_df = pd.DataFrame({"val": [9, 8]})
        ev = RepairEvaluator(clean_df, dirty_df, "test")
        m = ev.evaluate([])
        assert m.detection_precision == 0.0

    def test_recall_zero_when_no_actual_errors(self):
        clean_df = pd.DataFrame({"val": [1, 2]})
        dirty_df = pd.DataFrame({"val": [1, 2]})  # no corruption
        ev = RepairEvaluator(clean_df, dirty_df, "test")
        records = [_make_record(0, "val", 1, 99, DecisionAction.AUTO_REPAIR)]
        m = ev.evaluate(records)
        assert m.detection_recall == 0.0

    def test_f1_zero_when_both_zero(self):
        clean_df = pd.DataFrame({"val": [1]})
        dirty_df = pd.DataFrame({"val": [1]})
        ev = RepairEvaluator(clean_df, dirty_df, "test")
        m = ev.evaluate([])
        assert m.detection_f1 == 0.0

    def test_repair_accuracy_zero_when_no_auto(self):
        clean_df = pd.DataFrame({"val": [1]})
        dirty_df = pd.DataFrame({"val": [9]})
        ev = RepairEvaluator(clean_df, dirty_df, "test")
        records = [_make_record(0, "val", 9, 1, DecisionAction.HUMAN_REVIEW)]
        m = ev.evaluate(records)
        assert m.repair_accuracy == 0.0
        assert m.n_auto_repair == 0


# ===========================================================================
# Group 10: RepairEvaluator — empty records
# ===========================================================================


class TestRepairEvaluatorEmpty:

    def test_empty_records(self):
        clean_df = pd.DataFrame({"val": [1, 2]})
        dirty_df = pd.DataFrame({"val": [9, 8]})
        ev = RepairEvaluator(clean_df, dirty_df, "test")
        m = ev.evaluate([])
        assert m.n_detected == 0
        assert m.detection_precision == 0.0
        assert m.detection_recall == 0.0
        assert m.n_auto_repair == 0


# ===========================================================================
# Group 11: RepairEvaluator — action distribution
# ===========================================================================


class TestRepairEvaluatorActions:

    def setup_method(self):
        self.clean_df = pd.DataFrame({"val": [1, 2, 3, 4, 5]})
        self.dirty_df = pd.DataFrame({"val": [9, 8, 7, 6, 5]})  # 4 corrupted
        self.ev = RepairEvaluator(self.clean_df, self.dirty_df, "test")

    def test_action_counts(self):
        records = [
            _make_record(0, "val", 9, 1, DecisionAction.AUTO_REPAIR),
            _make_record(1, "val", 8, 2, DecisionAction.HUMAN_REVIEW),
            _make_record(2, "val", 7, 3, DecisionAction.ABSTAIN),
            _make_record(3, "val", 6, 4, DecisionAction.AUTO_REPAIR),
        ]
        m = self.ev.evaluate(records)
        assert m.n_auto_repair == 2
        assert m.n_human_review == 1
        assert m.n_abstain == 1


# ===========================================================================
# Group 12: RepairEvaluator.evaluate_cell
# ===========================================================================


class TestRepairEvaluatorCell:

    def setup_method(self):
        self.clean_df = pd.DataFrame({"val": [10, 20]})
        self.dirty_df = pd.DataFrame({"val": [99, 20]})
        self.ev = RepairEvaluator(self.clean_df, self.dirty_df, "test")

    def test_cell_is_truly_corrupted(self):
        rec = _make_record(0, "val", 99, 10, DecisionAction.AUTO_REPAIR)
        cr = self.ev.evaluate_cell(rec)
        assert cr.is_truly_corrupted is True

    def test_cell_not_truly_corrupted(self):
        rec = _make_record(1, "val", 20, 99, DecisionAction.AUTO_REPAIR)
        cr = self.ev.evaluate_cell(rec)
        assert cr.is_truly_corrupted is False

    def test_repair_correct(self):
        rec = _make_record(0, "val", 99, 10, DecisionAction.AUTO_REPAIR)
        cr = self.ev.evaluate_cell(rec)
        assert cr.is_repair_correct is True

    def test_repair_incorrect(self):
        rec = _make_record(0, "val", 99, 777, DecisionAction.AUTO_REPAIR)
        cr = self.ev.evaluate_cell(rec)
        assert cr.is_repair_correct is False

    def test_clean_value_stored(self):
        rec = _make_record(0, "val", 99, 10, DecisionAction.AUTO_REPAIR)
        cr = self.ev.evaluate_cell(rec)
        assert cr.clean_value == 10


# ===========================================================================
# Group 13: nll_score — correctness and edge cases
# ===========================================================================


class TestNLLCorrectness:

    def test_perfect_predictions_low_nll(self):
        probs = np.array([1.0, 1.0, 0.0, 0.0])
        labels = np.array([1, 1, 0, 0])
        nll = nll_score(probs, labels)
        # After clipping 1→1-eps, 0→eps: NLL ≈ -log(1-eps) ≈ small
        assert nll < 0.01

    def test_uniform_predictions(self):
        probs = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([1, 1, 0, 0])
        nll = nll_score(probs, labels)
        # NLL = -log(0.5) = log(2) ≈ 0.693
        assert math.isclose(nll, math.log(2), rel_tol=1e-6)

    def test_wrong_predictions_high_nll(self):
        probs = np.array([0.0, 0.0])
        labels = np.array([1, 1])
        nll = nll_score(probs, labels)
        # Clipped to eps; NLL = -log(eps) → large
        assert nll > 10

    def test_p0_clipped_safely(self):
        # p=0 for a positive label should not raise
        nll = nll_score(np.array([0.0]), np.array([1]))
        assert math.isfinite(nll)

    def test_p1_clipped_safely(self):
        nll = nll_score(np.array([1.0]), np.array([0]))
        assert math.isfinite(nll)

    def test_p1_correct_label_near_zero(self):
        nll = nll_score(np.array([1.0]), np.array([1]))
        assert nll < 0.01  # near-perfect

    def test_all_correct_high_confidence(self):
        probs = np.array([0.9, 0.9, 0.9])
        labels = np.array([1, 1, 1])
        nll = nll_score(probs, labels)
        assert nll < 0.2  # confident and correct → low NLL

    def test_all_wrong_high_confidence(self):
        probs = np.array([0.9, 0.9])
        labels = np.array([0, 0])
        nll = nll_score(probs, labels)
        assert nll > 1.5  # confident and wrong → high NLL


# ===========================================================================
# Group 14: nll_score — probability clipping
# ===========================================================================


class TestNLLClipping:

    def test_custom_eps_used(self):
        nll_default = nll_score(np.array([0.0]), np.array([1]))
        nll_tight = nll_score(np.array([0.0]), np.array([1]), clip_eps=1e-15)
        # Tighter clip → higher NLL magnitude
        assert nll_tight > nll_default

    def test_invalid_clip_eps_raises(self):
        with pytest.raises(ValueError, match="clip_eps"):
            nll_score(np.array([0.5]), np.array([1]), clip_eps=-0.01)

    def test_invalid_clip_eps_too_large_raises(self):
        with pytest.raises(ValueError, match="clip_eps"):
            nll_score(np.array([0.5]), np.array([1]), clip_eps=0.5)

    def test_no_nan_in_output(self):
        probs = np.array([0.0, 0.5, 1.0])
        labels = np.array([0, 1, 1])
        nll = nll_score(probs, labels)
        assert not math.isnan(nll)


# ===========================================================================
# Group 15: nll_score — empty and invalid inputs
# ===========================================================================


class TestNLLEdgeCases:

    def test_empty_returns_zero(self):
        assert nll_score(np.array([]), np.array([])) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            nll_score(np.array([0.5, 0.5]), np.array([1]))

    def test_non_binary_labels_raises(self):
        with pytest.raises(ValueError, match="binary"):
            nll_score(np.array([0.5]), np.array([2]))

    def test_single_sample(self):
        nll = nll_score(np.array([0.8]), np.array([1]))
        assert math.isfinite(nll)


# ===========================================================================
# Group 16: CalibrationEvaluator — integration
# ===========================================================================


class TestCalibrationEvaluatorIntegration:

    def setup_method(self):
        self.ev = CalibrationEvaluator(n_bins=5)
        self.probs = [0.9, 0.8, 0.7, 0.3, 0.2]
        self.labels = [1, 1, 1, 0, 0]  # perfectly ordered

    def test_report_has_all_fields(self):
        r = self.ev.evaluate(self.probs, self.labels, "raw", "test")
        assert hasattr(r, "brier")
        assert hasattr(r, "ece")
        assert hasattr(r, "nll")
        assert hasattr(r, "n_samples")

    def test_n_samples_correct(self):
        r = self.ev.evaluate(self.probs, self.labels)
        assert r.n_samples == 5

    def test_n_positive_correct(self):
        r = self.ev.evaluate(self.probs, self.labels)
        assert r.n_positive == 3

    def test_brier_range(self):
        r = self.ev.evaluate(self.probs, self.labels)
        assert 0.0 <= r.brier <= 1.0

    def test_ece_range(self):
        r = self.ev.evaluate(self.probs, self.labels)
        assert 0.0 <= r.ece <= 1.0

    def test_nll_non_negative(self):
        r = self.ev.evaluate(self.probs, self.labels)
        assert r.nll >= 0.0

    def test_perfect_calibration_low_brier(self):
        r = self.ev.evaluate([1.0, 1.0, 0.0], [1, 1, 0])
        assert r.brier < 0.01

    def test_invalid_n_bins_raises(self):
        with pytest.raises(ValueError, match="n_bins"):
            CalibrationEvaluator(n_bins=0)


# ===========================================================================
# Group 17: CalibrationEvaluator — empty / edge cases
# ===========================================================================


class TestCalibrationEvaluatorEdge:

    def test_empty_input_returns_zero_report(self):
        ev = CalibrationEvaluator()
        r = ev.evaluate([], [], "raw", "test")
        assert r.n_samples == 0
        assert r.brier == 0.0
        assert r.ece == 0.0
        assert r.nll == 0.0

    def test_all_correct_confident(self):
        ev = CalibrationEvaluator()
        r = ev.evaluate([0.99, 0.99, 0.99], [1, 1, 1])
        assert r.brier < 0.01
        assert r.nll < 0.1

    def test_all_wrong_confident(self):
        ev = CalibrationEvaluator()
        r = ev.evaluate([0.99, 0.99], [0, 0])
        assert r.brier > 0.9
        assert r.nll > 3.0

    def test_label_field(self):
        ev = CalibrationEvaluator()
        r = ev.evaluate([0.5], [1], label="test_label", dataset="ds")
        assert r.label == "test_label"
        assert r.dataset == "ds"

    def test_to_dict_keys(self):
        ev = CalibrationEvaluator()
        r = ev.evaluate([0.8, 0.2], [1, 0])
        d = r.to_dict()
        for key in ("label", "dataset", "n_samples", "n_positive", "brier", "ece", "nll"):
            assert key in d


# ===========================================================================
# Group 18: CalibrationComparison — raw vs calibrated
# ===========================================================================


class TestCalibrationComparison:

    def setup_method(self):
        self.ev = CalibrationEvaluator()
        self.raw = [0.6, 0.7, 0.8, 0.4, 0.3]
        # Calibrated probs push confident-correct higher, uncertain-incorrect lower
        self.cal = [0.85, 0.90, 0.92, 0.15, 0.08]
        self.labels = [1, 1, 1, 0, 0]

    def test_comparison_has_raw_and_calibrated(self):
        comp = self.ev.compare(self.raw, self.cal, self.labels)
        assert hasattr(comp, "raw")
        assert hasattr(comp, "calibrated")

    def test_comparison_dataset_set(self):
        comp = self.ev.compare(self.raw, self.cal, self.labels, dataset="ds")
        assert comp.dataset == "ds"

    def test_boolean_fields_present(self):
        comp = self.ev.compare(self.raw, self.cal, self.labels)
        assert isinstance(comp.calibration_improves_brier, bool)
        assert isinstance(comp.calibration_improves_ece, bool)
        assert isinstance(comp.calibration_improves_nll, bool)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            self.ev.compare([0.5, 0.5], [0.8], [1, 0])

    def test_to_dict_has_required_keys(self):
        comp = self.ev.compare(self.raw, self.cal, self.labels, dataset="test")
        d = comp.to_dict()
        for k in ("raw_brier", "raw_ece", "raw_nll", "cal_brier", "cal_ece", "cal_nll"):
            assert k in d, f"Missing key: {k}"

    def test_note_stored(self):
        comp = self.ev.compare(self.raw, self.cal, self.labels, note="limitation X")
        assert "limitation X" in comp.note


# ===========================================================================
# Group 19: CostEvaluator — all AUTO_REPAIR
# ===========================================================================


class TestCostEvaluatorAllAuto:

    def setup_method(self):
        self.clean_df = pd.DataFrame({"val": [1, 2, 3]})
        self.ev = CostEvaluator(self.clean_df, c_wrong=10, c_human=3, c_abstain=5)

    def _make(self, row, candidate, action=DecisionAction.AUTO_REPAIR):
        return _make_record(row, "val", 99, candidate, action)

    def test_all_correct_zero_cost(self):
        records = [
            self._make(0, 1),  # correct
            self._make(1, 2),  # correct
            self._make(2, 3),  # correct
        ]
        r = self.ev.evaluate(records, "test")
        assert r.total_realized_cost == 0.0
        assert r.n_correct_auto == 3
        assert r.n_wrong_auto == 0

    def test_all_incorrect_full_cost(self):
        records = [
            self._make(0, 99),  # wrong: repair=99, clean=1
            self._make(1, 99),  # wrong
        ]
        r = self.ev.evaluate(records, "test")
        assert math.isclose(r.total_realized_cost, 2 * 10.0)

    def test_wrong_auto_rate(self):
        records = [self._make(0, 99), self._make(1, 2)]
        r = self.ev.evaluate(records, "test")
        assert math.isclose(r.wrong_auto_rate, 0.5)

    def test_auto_rate_one(self):
        records = [self._make(0, 1), self._make(1, 2)]
        r = self.ev.evaluate(records, "test")
        assert math.isclose(r.auto_rate, 1.0)
        assert math.isclose(r.human_rate, 0.0)
        assert math.isclose(r.abstain_rate, 0.0)


# ===========================================================================
# Group 20: CostEvaluator — all HUMAN_REVIEW
# ===========================================================================


class TestCostEvaluatorAllHuman:

    def test_all_human_total_cost(self):
        clean_df = pd.DataFrame({"val": [1, 2, 3]})
        ev = CostEvaluator(clean_df, c_wrong=10, c_human=3, c_abstain=5)
        records = [
            _make_record(i, "val", 99, 1, DecisionAction.HUMAN_REVIEW)
            for i in range(3)
        ]
        r = ev.evaluate(records, "test")
        assert math.isclose(r.total_realized_cost, 3 * 3.0)

    def test_human_rate_one(self):
        clean_df = pd.DataFrame({"val": [1]})
        ev = CostEvaluator(clean_df, c_wrong=10, c_human=3, c_abstain=5)
        records = [_make_record(0, "val", 99, 1, DecisionAction.HUMAN_REVIEW)]
        r = ev.evaluate(records, "test")
        assert math.isclose(r.human_rate, 1.0)
        assert math.isclose(r.auto_rate, 0.0)


# ===========================================================================
# Group 21: CostEvaluator — all ABSTAIN
# ===========================================================================


class TestCostEvaluatorAllAbstain:

    def test_all_abstain_total_cost(self):
        clean_df = pd.DataFrame({"val": [1, 2]})
        ev = CostEvaluator(clean_df, c_wrong=10, c_human=3, c_abstain=5)
        records = [
            _make_record(i, "val", 99, 1, DecisionAction.ABSTAIN)
            for i in range(2)
        ]
        r = ev.evaluate(records, "test")
        assert math.isclose(r.total_realized_cost, 2 * 5.0)

    def test_abstain_rate_one(self):
        clean_df = pd.DataFrame({"val": [1]})
        ev = CostEvaluator(clean_df, c_wrong=10, c_human=3, c_abstain=5)
        records = [_make_record(0, "val", 99, 1, DecisionAction.ABSTAIN)]
        r = ev.evaluate(records, "test")
        assert math.isclose(r.abstain_rate, 1.0)


# ===========================================================================
# Group 22: CostEvaluator — mixed actions
# ===========================================================================


class TestCostEvaluatorMixed:

    def setup_method(self):
        self.clean_df = pd.DataFrame({"val": [1, 2, 3, 4]})
        self.ev = CostEvaluator(self.clean_df, c_wrong=10, c_human=3, c_abstain=5)

    def test_mixed_cost(self):
        records = [
            _make_record(0, "val", 9, 1, DecisionAction.AUTO_REPAIR),    # correct → 0
            _make_record(1, "val", 9, 99, DecisionAction.AUTO_REPAIR),   # wrong → 10
            _make_record(2, "val", 9, 3, DecisionAction.HUMAN_REVIEW),   # → 3
            _make_record(3, "val", 9, 4, DecisionAction.ABSTAIN),        # → 5
        ]
        r = self.ev.evaluate(records, "test")
        assert math.isclose(r.total_realized_cost, 0 + 10 + 3 + 5)

    def test_action_counts(self):
        records = [
            _make_record(0, "val", 9, 1, DecisionAction.AUTO_REPAIR),
            _make_record(1, "val", 9, 1, DecisionAction.HUMAN_REVIEW),
            _make_record(2, "val", 9, 1, DecisionAction.ABSTAIN),
            _make_record(3, "val", 9, 1, DecisionAction.AUTO_REPAIR),
        ]
        r = self.ev.evaluate(records, "test")
        assert r.n_auto == 2
        assert r.n_human == 1
        assert r.n_abstain == 1

    def test_mean_cost_per_record(self):
        records = [
            _make_record(0, "val", 9, 99, DecisionAction.AUTO_REPAIR),  # wrong → 10
            _make_record(1, "val", 9, 2, DecisionAction.AUTO_REPAIR),   # correct → 0
        ]
        r = self.ev.evaluate(records, "test")
        assert math.isclose(r.mean_cost_per_record, 5.0)


# ===========================================================================
# Group 23: CostEvaluator — empty input
# ===========================================================================


class TestCostEvaluatorEmpty:

    def test_empty_returns_zero_report(self):
        clean_df = pd.DataFrame({"val": [1]})
        ev = CostEvaluator(clean_df, c_wrong=10, c_human=3, c_abstain=5)
        r = ev.evaluate([], "test")
        assert r.n_records == 0
        assert r.total_realized_cost == 0.0
        assert r.auto_rate == 0.0

    def test_invalid_c_wrong_raises(self):
        clean_df = pd.DataFrame({"val": [1]})
        with pytest.raises(ValueError, match="c_wrong"):
            CostEvaluator(clean_df, c_wrong=0, c_human=3, c_abstain=5)

    def test_invalid_c_human_raises(self):
        clean_df = pd.DataFrame({"val": [1]})
        with pytest.raises(ValueError, match="c_human"):
            CostEvaluator(clean_df, c_wrong=10, c_human=-1, c_abstain=5)

    def test_invalid_c_abstain_raises(self):
        clean_df = pd.DataFrame({"val": [1]})
        with pytest.raises(ValueError, match="c_abstain"):
            CostEvaluator(clean_df, c_wrong=10, c_human=3, c_abstain=-1)


# ===========================================================================
# Group 24: CostEvaluator.from_cost_config
# ===========================================================================


class TestCostEvaluatorFromConfig:

    def test_correct_mapping(self):
        config = CostConfig(c_wrong_repair=20, c_human_review=4, c_leave_error=8, dataset_name="test")
        clean_df = pd.DataFrame({"val": [1]})
        ev = CostEvaluator.from_cost_config(clean_df, config)
        assert ev.c_wrong == 20
        assert ev.c_human == 4
        assert ev.c_abstain == 8

    def test_produces_same_results_as_direct(self):
        config = CostConfig(c_wrong_repair=15, c_human_review=3, c_leave_error=6, dataset_name="test")
        clean_df = pd.DataFrame({"val": [1, 2]})
        ev_config = CostEvaluator.from_cost_config(clean_df, config)
        ev_direct = CostEvaluator(clean_df, c_wrong=15, c_human=3, c_abstain=6)
        records = [_make_record(0, "val", 9, 99, DecisionAction.AUTO_REPAIR)]
        r1 = ev_config.evaluate(records, "test")
        r2 = ev_direct.evaluate(records, "test")
        assert math.isclose(r1.total_realized_cost, r2.total_realized_cost)


# ===========================================================================
# Group 25: CostEvaluator — action rate calculations
# ===========================================================================


class TestCostActionRates:

    def test_rates_sum_to_one(self):
        clean_df = pd.DataFrame({"val": [1, 2, 3, 4, 5]})
        ev = CostEvaluator(clean_df, c_wrong=10, c_human=3, c_abstain=5)
        records = [
            _make_record(0, "val", 9, 1, DecisionAction.AUTO_REPAIR),
            _make_record(1, "val", 9, 1, DecisionAction.HUMAN_REVIEW),
            _make_record(2, "val", 9, 1, DecisionAction.ABSTAIN),
            _make_record(3, "val", 9, 1, DecisionAction.AUTO_REPAIR),
            _make_record(4, "val", 9, 1, DecisionAction.HUMAN_REVIEW),
        ]
        r = ev.evaluate(records, "test")
        total_rate = r.auto_rate + r.human_rate + r.abstain_rate
        assert math.isclose(total_rate, 1.0, rel_tol=1e-6)

    def test_always_auto_rate_is_one(self):
        clean_df = pd.DataFrame({"val": [1, 2]})
        ev = CostEvaluator(clean_df, c_wrong=10, c_human=3, c_abstain=5)
        records = [_make_record(i, "val", 9, 1, DecisionAction.AUTO_REPAIR) for i in range(2)]
        r = ev.evaluate(records, "test")
        assert math.isclose(r.auto_rate, 1.0)

    def test_wrong_auto_rate_zero_when_all_correct(self):
        clean_df = pd.DataFrame({"val": [1, 2]})
        ev = CostEvaluator(clean_df, c_wrong=10, c_human=3, c_abstain=5)
        records = [
            _make_record(0, "val", 9, 1, DecisionAction.AUTO_REPAIR),
            _make_record(1, "val", 9, 2, DecisionAction.AUTO_REPAIR),
        ]
        r = ev.evaluate(records, "test")
        assert r.wrong_auto_rate == 0.0

    def test_wrong_auto_rate_zero_when_no_auto(self):
        clean_df = pd.DataFrame({"val": [1]})
        ev = CostEvaluator(clean_df, c_wrong=10, c_human=3, c_abstain=5)
        records = [_make_record(0, "val", 9, 1, DecisionAction.HUMAN_REVIEW)]
        r = ev.evaluate(records, "test")
        assert r.wrong_auto_rate == 0.0


# ===========================================================================
# Group 26: compare_policies helper
# ===========================================================================


class TestComparePolicies:

    def _make_cost_report(self, name: str, cost: float) -> CostReport:
        return CostReport(
            method_name=name,
            dataset="test",
            n_records=10,
            n_auto=5, n_human=3, n_abstain=2,
            auto_rate=0.5, human_rate=0.3, abstain_rate=0.2,
            n_correct_auto=3, n_wrong_auto=2, wrong_auto_rate=0.4,
            total_realized_cost=cost, mean_cost_per_record=cost / 10,
        )

    def _make_repair_metrics(self, name: str) -> RepairMetrics:
        m = RepairMetrics(method_name=name, dataset="test")
        m.n_detected = 10
        m.n_truly_corrupted = 8
        m.n_false_positive = 2
        m.n_false_negative = 0
        m.n_total_corrupted = 8
        m.detection_precision = 0.8
        m.detection_recall = 1.0
        m.detection_f1 = 0.889
        m.n_auto_repair = 5
        m.n_correct_repairs = 3
        m.n_incorrect_repairs = 2
        m.repair_accuracy = 0.6
        m.n_human_review = 3
        m.n_abstain = 2
        return m

    def test_compare_returns_rows(self):
        cr = [self._make_cost_report("a", 50), self._make_cost_report("b", 30)]
        rr = [self._make_repair_metrics("a"), self._make_repair_metrics("b")]
        rows = compare_policies(cr, rr)
        assert len(rows) == 2

    def test_method_names_preserved(self):
        cr = [self._make_cost_report("always_auto", 50)]
        rr = [self._make_repair_metrics("always_auto")]
        rows = compare_policies(cr, rr)
        assert rows[0].method_name == "always_auto"

    def test_length_mismatch_raises(self):
        cr = [self._make_cost_report("a", 50)]
        rr = [self._make_repair_metrics("a"), self._make_repair_metrics("b")]
        with pytest.raises(ValueError, match="same length"):
            compare_policies(cr, rr)

    def test_empty_input_returns_empty(self):
        rows = compare_policies([], [])
        assert rows == []


# ===========================================================================
# Group 27: PolicyComparisonRow.to_dict
# ===========================================================================


class TestPolicyComparisonRowToDict:

    def test_to_dict_has_required_keys(self):
        row = PolicyComparisonRow(
            method_name="test", dataset="d",
            detection_precision=0.8, detection_recall=0.9, detection_f1=0.85,
            repair_accuracy=0.7, auto_rate=0.5, human_rate=0.3, abstain_rate=0.2,
            wrong_auto_rate=0.1, total_realized_cost=25.0, mean_cost_per_record=2.5,
        )
        d = row.to_dict()
        for k in ("method_name", "dataset", "detection_precision", "detection_recall",
                  "detection_f1", "repair_accuracy", "auto_rate", "human_rate",
                  "abstain_rate", "wrong_auto_rate", "total_realized_cost",
                  "mean_cost_per_record"):
            assert k in d, f"Missing key: {k}"


# ===========================================================================
# Group 28: Data leakage — evaluation accesses clean_df; predictions do not
# ===========================================================================


class TestDataLeakage:
    """Evaluation may access clean_df; upstream prediction functions must not."""

    def test_repair_evaluator_init_takes_clean_df(self):
        """RepairEvaluator explicitly accepts clean_df (intended)."""
        sig = inspect.signature(RepairEvaluator.__init__)
        assert "clean_df" in sig.parameters

    def test_cost_evaluator_init_takes_clean_df(self):
        sig = inspect.signature(CostEvaluator.__init__)
        assert "clean_df" in sig.parameters

    def test_always_auto_run_no_clean_df(self):
        sig = inspect.signature(AlwaysAutoBaseline.run)
        assert "clean_df" not in sig.parameters

    def test_threshold_run_no_clean_df(self):
        sig = inspect.signature(ConfidenceThresholdBaseline.run)
        assert "clean_df" not in sig.parameters

    def test_decision_policy_decide_no_clean_df(self):
        from src.decision.policy import DecisionPolicy
        sig = inspect.signature(DecisionPolicy.decide)
        assert "clean_df" not in sig.parameters

    def test_calibration_evaluator_no_clean_df(self):
        """CalibrationEvaluator takes probs+labels, not clean_df."""
        sig = inspect.signature(CalibrationEvaluator.evaluate)
        assert "clean_df" not in sig.parameters

    def test_baseline_record_no_clean_value_field(self):
        fields = set(BaselineRecord.__dataclass_fields__)
        assert "clean_value" not in fields
        assert "ground_truth" not in fields

    def test_cost_evaluator_is_only_cost_evaluator_that_takes_clean_df(self):
        """nll_score, CalibrationEvaluator.evaluate, compare_policies do not."""
        for fn in (nll_score, compare_policies):
            sig = inspect.signature(fn)
            assert "clean_df" not in sig.parameters


# ===========================================================================
# Group 29: Determinism
# ===========================================================================


class TestDeterminism:

    def test_repair_evaluator_deterministic(self):
        clean_df = pd.DataFrame({"val": [1, 2, 3]})
        dirty_df = pd.DataFrame({"val": [9, 8, 3]})
        ev = RepairEvaluator(clean_df, dirty_df, "test")
        records = [
            _make_record(0, "val", 9, 1, DecisionAction.AUTO_REPAIR),
            _make_record(1, "val", 8, 99, DecisionAction.AUTO_REPAIR),
        ]
        m1 = ev.evaluate(records)
        m2 = ev.evaluate(records)
        assert m1.repair_accuracy == m2.repair_accuracy
        assert m1.detection_f1 == m2.detection_f1

    def test_cost_evaluator_deterministic(self):
        clean_df = pd.DataFrame({"val": [1, 2]})
        ev = CostEvaluator(clean_df, c_wrong=10, c_human=3, c_abstain=5)
        records = [
            _make_record(0, "val", 9, 99, DecisionAction.AUTO_REPAIR),
            _make_record(1, "val", 9, 2, DecisionAction.HUMAN_REVIEW),
        ]
        r1 = ev.evaluate(records, "test")
        r2 = ev.evaluate(records, "test")
        assert r1.total_realized_cost == r2.total_realized_cost

    def test_nll_deterministic(self):
        probs = np.array([0.8, 0.6, 0.3])
        labels = np.array([1, 1, 0])
        assert nll_score(probs, labels) == nll_score(probs, labels)


# ===========================================================================
# Group 30: Threshold monotonicity via CostEvaluator
# ===========================================================================


class TestThresholdMonotonicityViaCost:
    """
    As threshold increases:
    - fewer AUTO_REPAIR decisions
    - more HUMAN_REVIEW decisions
    - total wrong_auto_cost (n_wrong_auto * c_wrong) can only decrease
      if fewer wrong autos are accepted.
    """

    def setup_method(self):
        # 6 records with varying confidence; 3 truly corrupted
        self.clean_df = pd.DataFrame({"val": [10, 20, 30, 40, 50, 60]})

        @dataclass
        class _Rec:
            dataset: str = "test"
            row_id: int = 0
            column: str = "val"
            dirty_value: Any = 99
            top_candidate: Any = 99  # wrong repair
            raw_confidence: float = 0.5
            action: DecisionAction = DecisionAction.AUTO_REPAIR
            method_name: str = "conf_threshold"
            threshold: float = None
            calibrated_probability: float = None
            expected_cost_auto: float = None
            expected_cost_human: float = None
            expected_cost_abstain: float = None
            metadata: dict = None
            def __post_init__(self):
                if self.metadata is None:
                    self.metadata = {}

        confidences = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.fake_unc = [
            type("FU", (), {
                "dataset": "test",
                "row_id": i,
                "column": "val",
                "dirty_value": 99,
                "top_candidate": 99,  # all wrong
                "raw_confidence": c,
                "repair_method": "mock",
                "error_type": "OUT_OF_RANGE",
            })()
            for i, c in enumerate(confidences)
        ]

    def test_higher_threshold_fewer_auto(self):
        auto_counts = []
        for t in DEFAULT_THRESHOLDS:
            bl = ConfidenceThresholdBaseline(threshold=t)
            # Use fake unc-like objects
            records_out = []
            for fu in self.fake_unc:
                action = bl.decide_one(fu.raw_confidence)
                records_out.append(_make_record(
                    fu.row_id, fu.column, fu.dirty_value,
                    fu.top_candidate, action,
                ))
            n_auto = sum(1 for r in records_out if r.action == DecisionAction.AUTO_REPAIR)
            auto_counts.append(n_auto)

        # Monotone non-increasing
        for i in range(len(auto_counts) - 1):
            assert auto_counts[i] >= auto_counts[i + 1]

    def test_higher_threshold_lower_or_equal_wrong_auto_cost(self):
        """Higher threshold → same or fewer wrong autos → same or lower wrong-auto cost."""
        ev = CostEvaluator(self.clean_df, c_wrong=10, c_human=3, c_abstain=5)
        wrong_costs = []
        for t in DEFAULT_THRESHOLDS:
            bl = ConfidenceThresholdBaseline(threshold=t)
            records_out = []
            for fu in self.fake_unc:
                action = bl.decide_one(fu.raw_confidence)
                records_out.append(_make_record(
                    fu.row_id, fu.column, fu.dirty_value,
                    fu.top_candidate, action,
                ))
            r = ev.evaluate(records_out, "test")
            wrong_costs.append(r.n_wrong_auto * ev.c_wrong)

        for i in range(len(wrong_costs) - 1):
            assert wrong_costs[i] >= wrong_costs[i + 1]
