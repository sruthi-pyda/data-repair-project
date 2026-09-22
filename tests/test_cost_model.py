"""
tests/test_cost_model.py
-------------------------
Phase 7: Tests for risk and cost modeling.

Covers:
  - CostConfig construction and validation
  - CostConfig.from_yaml() for each dataset and defaults
  - ExpectedCosts computation
  - Analytical thresholds
  - Sensitivity analysis
  - RiskRecord construction
  - compute_risk_batch with skip_columns
  - RiskEvaluator empirical cost computation
  - print_risk_report runs without error
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from src.risk.cost_model import (
    Action,
    AnalyticalThresholds,
    CostConfig,
    ExpectedCosts,
    RiskCalculator,
    RiskRecord,
    SensitivityPoint,
)
from src.risk.evaluation import (
    ActionCounts,
    EmpiricalCostReport,
    RiskEvaluator,
    print_risk_report,
)

COSTS_YAML = Path("configs/costs.yaml")


# ===========================================================================
# Helpers
# ===========================================================================


def make_cost_config(wrong=10.0, review=2.0, leave=5.0, name="test") -> CostConfig:
    return CostConfig(c_wrong_repair=wrong, c_human_review=review,
                      c_leave_error=leave, dataset_name=name)


@dataclass
class _FakeUncRecord:
    """Minimal stand-in for UncertaintyRecord."""
    dataset: str
    row_id: int
    column: str
    dirty_value: Any
    error_type: str
    repair_method: str
    top_candidate: Any
    raw_confidence: float
    self_consistency: float = 1.0
    entropy: float = 0.0
    normalized_entropy: float = 0.0
    n_samples: int = 5
    vote_counts: dict = None
    candidate_frequencies: dict = None
    n_candidates: int = 1

    def __post_init__(self):
        if self.vote_counts is None:
            self.vote_counts = {}
        if self.candidate_frequencies is None:
            self.candidate_frequencies = {}


def make_fake_unc(row_id=0, column="age", top_candidate=30,
                  confidence=0.9, dataset="test") -> _FakeUncRecord:
    return _FakeUncRecord(
        dataset=dataset,
        row_id=row_id,
        column=column,
        dirty_value=99,
        error_type="OUT_OF_RANGE",
        repair_method="statistical",
        top_candidate=top_candidate,
        raw_confidence=confidence,
    )


# ===========================================================================
# CostConfig tests
# ===========================================================================


class TestCostConfig:

    def test_basic_construction(self):
        c = make_cost_config()
        assert c.c_wrong_repair == 10.0
        assert c.c_human_review == 2.0
        assert c.c_leave_error == 5.0
        assert c.dataset_name == "test"

    def test_frozen(self):
        c = make_cost_config()
        with pytest.raises((AttributeError, TypeError)):
            c.c_wrong_repair = 99  # type: ignore

    def test_wrong_repair_must_be_positive(self):
        with pytest.raises(ValueError, match="c_wrong_repair"):
            CostConfig(c_wrong_repair=0, c_human_review=2, c_leave_error=5)

    def test_wrong_repair_negative_raises(self):
        with pytest.raises(ValueError):
            CostConfig(c_wrong_repair=-1, c_human_review=2, c_leave_error=5)

    def test_human_review_zero_allowed(self):
        c = CostConfig(c_wrong_repair=10, c_human_review=0, c_leave_error=5)
        assert c.c_human_review == 0.0

    def test_human_review_negative_raises(self):
        with pytest.raises(ValueError, match="c_human_review"):
            CostConfig(c_wrong_repair=10, c_human_review=-1, c_leave_error=5)

    def test_leave_error_zero_allowed(self):
        c = CostConfig(c_wrong_repair=10, c_human_review=2, c_leave_error=0)
        assert c.c_leave_error == 0.0

    def test_leave_error_negative_raises(self):
        with pytest.raises(ValueError, match="c_leave_error"):
            CostConfig(c_wrong_repair=10, c_human_review=2, c_leave_error=-1)

    def test_default_factory(self):
        c = CostConfig.default()
        assert c.c_wrong_repair == 10.0
        assert c.c_human_review == 2.0
        assert c.c_leave_error == 5.0
        assert c.dataset_name == "unknown"

    def test_default_with_name(self):
        c = CostConfig.default("breast_cancer")
        assert c.dataset_name == "breast_cancer"

    def test_equality(self):
        c1 = make_cost_config()
        c2 = make_cost_config()
        assert c1 == c2

    def test_inequality_different_costs(self):
        c1 = make_cost_config(wrong=10)
        c2 = make_cost_config(wrong=25)
        assert c1 != c2


class TestCostConfigFromYaml:

    def test_from_yaml_defaults(self):
        c = CostConfig.from_yaml(COSTS_YAML, "unknown_dataset")
        assert c.c_wrong_repair == 10.0
        assert c.c_human_review == 2.0
        assert c.c_leave_error == 5.0

    def test_from_yaml_adult(self):
        c = CostConfig.from_yaml(COSTS_YAML, "adult")
        assert c.c_wrong_repair == 10.0
        assert c.c_human_review == 2.0
        assert c.c_leave_error == 5.0
        assert c.dataset_name == "adult"

    def test_from_yaml_adult_uses_original_leave_cost(self):
        """Adult detections are false positives, but abstention on them now
        comes from the learned corruption probability q ≈ 0 (E[LEAVE] =
        q · C_leave), not from a hand-set cost. The original value 5 is
        restored; a temporary 1 (2026-09-22) was reverted."""
        c = CostConfig.from_yaml(COSTS_YAML, "adult")
        assert c.c_leave_error == 5.0
        assert c.c_leave_error > c.c_human_review

    def test_from_yaml_breast_cancer(self):
        c = CostConfig.from_yaml(COSTS_YAML, "breast_cancer")
        assert c.c_wrong_repair == 50.0
        assert c.c_human_review == 5.0
        assert c.c_leave_error == 10.0

    def test_from_yaml_smartfactory(self):
        c = CostConfig.from_yaml(COSTS_YAML, "smartfactory")
        assert c.c_wrong_repair == 25.0
        assert c.c_human_review == 2.0
        assert c.c_leave_error == 5.0

    def test_from_yaml_nonexistent_dataset_uses_defaults(self):
        c = CostConfig.from_yaml(COSTS_YAML, "nonexistent_xyz")
        assert c.c_wrong_repair == 10.0

    def test_from_yaml_dataset_name_stored(self):
        c = CostConfig.from_yaml(COSTS_YAML, "breast_cancer")
        assert c.dataset_name == "breast_cancer"

    def test_from_yaml_all_positive(self):
        for ds in ["adult", "breast_cancer", "smartfactory"]:
            c = CostConfig.from_yaml(COSTS_YAML, ds)
            assert c.c_wrong_repair > 0
            assert c.c_human_review >= 0
            assert c.c_leave_error >= 0


# ===========================================================================
# ExpectedCosts / compute_expected_costs tests
# ===========================================================================


class TestComputeExpectedCosts:

    def setup_method(self):
        self.calc = RiskCalculator(make_cost_config(wrong=10, review=2, leave=5))

    def test_p_zero_accept_is_max(self):
        ec = self.calc.compute_expected_costs(0.0)
        assert ec.accept == pytest.approx(10.0)
        assert ec.review == pytest.approx(2.0)
        assert ec.leave == pytest.approx(5.0)
        assert ec.best_action == Action.HUMAN_REVIEW

    def test_p_one_accept_is_zero(self):
        ec = self.calc.compute_expected_costs(1.0)
        assert ec.accept == pytest.approx(0.0)
        assert ec.best_action == Action.ACCEPT_REPAIR

    def test_p_at_accept_vs_review_threshold(self):
        # p* = 1 - 2/10 = 0.80 → at exactly 0.80 ACCEPT = REVIEW = 2.0
        ec = self.calc.compute_expected_costs(0.80)
        assert ec.accept == pytest.approx(2.0)
        assert ec.review == pytest.approx(2.0)

    def test_p_just_above_threshold_accept_wins(self):
        ec = self.calc.compute_expected_costs(0.85)
        assert ec.best_action == Action.ACCEPT_REPAIR

    def test_p_just_below_threshold_review_wins(self):
        ec = self.calc.compute_expected_costs(0.75)
        assert ec.best_action == Action.HUMAN_REVIEW

    def test_accept_cost_formula(self):
        for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
            ec = self.calc.compute_expected_costs(p)
            assert ec.accept == pytest.approx((1 - p) * 10.0)

    def test_review_cost_is_constant(self):
        for p in [0.0, 0.5, 1.0]:
            ec = self.calc.compute_expected_costs(p)
            assert ec.review == pytest.approx(2.0)

    def test_leave_cost_is_constant(self):
        for p in [0.0, 0.5, 1.0]:
            ec = self.calc.compute_expected_costs(p)
            assert ec.leave == pytest.approx(5.0)

    def test_min_cost_is_correct(self):
        ec = self.calc.compute_expected_costs(0.9)
        assert ec.min_cost == min(ec.accept, ec.review, ec.leave)

    def test_p_out_of_range_raises(self):
        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            self.calc.compute_expected_costs(-0.01)

    def test_p_greater_than_one_raises(self):
        with pytest.raises(ValueError):
            self.calc.compute_expected_costs(1.01)

    def test_returns_expected_costs_type(self):
        ec = self.calc.compute_expected_costs(0.5)
        assert isinstance(ec, ExpectedCosts)
        assert ec.p == 0.5

    def test_leave_wins_when_all_costs_equal_leave(self):
        # If C_leave < C_review and p is low enough ACCEPT > leave
        calc = RiskCalculator(make_cost_config(wrong=10, review=8, leave=3))
        ec = calc.compute_expected_costs(0.0)
        assert ec.best_action == Action.LEAVE_UNCHANGED

    def test_high_wrong_cost_pushes_action_to_review(self):
        # With very high wrong repair cost, even moderate p → REVIEW
        calc = RiskCalculator(make_cost_config(wrong=100, review=2, leave=5))
        ec = calc.compute_expected_costs(0.5)
        # E[accept] = 0.5*100=50, E[review]=2 → REVIEW
        assert ec.best_action == Action.HUMAN_REVIEW


# ===========================================================================
# Action enum tests
# ===========================================================================


class TestActionEnum:

    def test_three_actions(self):
        assert len(Action) == 3

    def test_values(self):
        assert Action.ACCEPT_REPAIR.value == "ACCEPT_REPAIR"
        assert Action.HUMAN_REVIEW.value == "HUMAN_REVIEW"
        assert Action.LEAVE_UNCHANGED.value == "LEAVE_UNCHANGED"

    def test_str_subclass(self):
        assert isinstance(Action.ACCEPT_REPAIR, str)

    def test_equality_with_string(self):
        assert Action.ACCEPT_REPAIR == "ACCEPT_REPAIR"


# ===========================================================================
# Analytical thresholds tests
# ===========================================================================


class TestAnalyticalThresholds:

    def test_default_costs_thresholds(self):
        # wrong=10, review=2, leave=5
        calc = RiskCalculator(make_cost_config())
        th = calc.analytical_thresholds()
        assert th.p_accept_vs_review == pytest.approx(0.80)
        assert th.p_accept_vs_leave == pytest.approx(0.50)

    def test_review_cheaper_than_leave_default(self):
        calc = RiskCalculator(make_cost_config())
        th = calc.analytical_thresholds()
        assert th.review_cheaper_than_leave is True  # 2 < 5

    def test_breast_cancer_thresholds(self):
        c = CostConfig(c_wrong_repair=50, c_human_review=5, c_leave_error=10,
                       dataset_name="breast_cancer")
        calc = RiskCalculator(c)
        th = calc.analytical_thresholds()
        assert th.p_accept_vs_review == pytest.approx(1 - 5/50)
        assert th.p_accept_vs_leave == pytest.approx(1 - 10/50)

    def test_smartfactory_thresholds(self):
        c = CostConfig(c_wrong_repair=25, c_human_review=2, c_leave_error=5,
                       dataset_name="smartfactory")
        calc = RiskCalculator(c)
        th = calc.analytical_thresholds()
        assert th.p_accept_vs_review == pytest.approx(1 - 2/25)
        assert th.p_accept_vs_leave == pytest.approx(1 - 5/25)

    def test_thresholds_clamped_to_unit_interval(self):
        # c_review >= c_wrong → threshold would be <= 0
        c = CostConfig(c_wrong_repair=5, c_human_review=10, c_leave_error=3,
                       dataset_name="test")
        calc = RiskCalculator(c)
        th = calc.analytical_thresholds()
        assert 0.0 <= th.p_accept_vs_review <= 1.0
        assert 0.0 <= th.p_accept_vs_leave <= 1.0

    def test_policy_string_nonempty(self):
        calc = RiskCalculator(make_cost_config())
        th = calc.analytical_thresholds()
        assert len(th.recommended_policy) > 10

    def test_returns_analytical_thresholds_type(self):
        calc = RiskCalculator(make_cost_config())
        th = calc.analytical_thresholds()
        assert isinstance(th, AnalyticalThresholds)

    def test_costs_stored_in_result(self):
        costs = make_cost_config()
        calc = RiskCalculator(costs)
        th = calc.analytical_thresholds()
        assert th.costs is costs

    def test_leave_cheaper_than_review(self):
        # review > leave → review_cheaper_than_leave = False
        c = CostConfig(c_wrong_repair=10, c_human_review=8, c_leave_error=3,
                       dataset_name="test")
        calc = RiskCalculator(c)
        th = calc.analytical_thresholds()
        assert th.review_cheaper_than_leave is False


# ===========================================================================
# Sensitivity analysis tests
# ===========================================================================


class TestSensitivityOverP:

    def setup_method(self):
        self.calc = RiskCalculator(make_cost_config())

    def test_default_returns_21_points(self):
        pts = self.calc.sensitivity_over_p()
        assert len(pts) == 21

    def test_first_p_is_zero(self):
        pts = self.calc.sensitivity_over_p()
        assert pts[0].p == pytest.approx(0.0)

    def test_last_p_is_one(self):
        pts = self.calc.sensitivity_over_p()
        assert pts[-1].p == pytest.approx(1.0)

    def test_p_values_are_monotone(self):
        pts = self.calc.sensitivity_over_p()
        ps = [pt.p for pt in pts]
        assert ps == sorted(ps)

    def test_all_have_best_action(self):
        pts = self.calc.sensitivity_over_p()
        for pt in pts:
            assert isinstance(pt.best_action, Action)

    def test_low_p_not_accept(self):
        pts = self.calc.sensitivity_over_p()
        assert pts[0].best_action != Action.ACCEPT_REPAIR

    def test_high_p_is_accept(self):
        pts = self.calc.sensitivity_over_p()
        assert pts[-1].best_action == Action.ACCEPT_REPAIR

    def test_cost_accept_decreasing(self):
        pts = self.calc.sensitivity_over_p()
        costs = [pt.cost_accept for pt in pts]
        for a, b in zip(costs, costs[1:]):
            assert a >= b - 1e-9

    def test_cost_review_constant(self):
        pts = self.calc.sensitivity_over_p()
        for pt in pts:
            assert pt.cost_review == pytest.approx(2.0)

    def test_cost_leave_constant(self):
        pts = self.calc.sensitivity_over_p()
        for pt in pts:
            assert pt.cost_leave == pytest.approx(5.0)

    def test_custom_n_steps(self):
        pts = self.calc.sensitivity_over_p(n_steps=11)
        assert len(pts) == 11
        assert pts[0].p == pytest.approx(0.0)
        assert pts[-1].p == pytest.approx(1.0)

    def test_n_steps_less_than_2_raises(self):
        with pytest.raises(ValueError):
            self.calc.sensitivity_over_p(n_steps=1)

    def test_action_transitions(self):
        # With default costs (wrong=10, review=2, leave=5):
        # p < 0.5  → REVIEW (cheapest)
        # 0.5 ≤ p < 0.8 → REVIEW (still cheaper than leave, and accept costs > 2)
        # p ≥ 0.8 → ACCEPT
        pts = self.calc.sensitivity_over_p(n_steps=101)
        for pt in pts:
            ec = self.calc.compute_expected_costs(pt.p)
            assert pt.best_action == ec.best_action

    def test_returns_sensitivity_point_type(self):
        pts = self.calc.sensitivity_over_p(n_steps=5)
        for pt in pts:
            assert isinstance(pt, SensitivityPoint)


# ===========================================================================
# compute_risk_record tests
# ===========================================================================


class TestComputeRiskRecord:

    def setup_method(self):
        self.costs = make_cost_config()
        self.calc = RiskCalculator(self.costs)

    def test_basic_fields(self):
        rec = make_fake_unc(row_id=5, column="age", top_candidate=30)
        rr = self.calc.compute_risk_record(rec, cal_prob=0.9)
        assert rr.dataset == "test"
        assert rr.row_id == 5
        assert rr.column == "age"
        assert rr.top_candidate == 30
        assert rr.calibrated_probability == pytest.approx(0.9)
        assert rr.probability_wrong == pytest.approx(0.1)

    def test_expected_costs_populated(self):
        rec = make_fake_unc()
        rr = self.calc.compute_risk_record(rec, cal_prob=0.9)
        assert rr.expected_cost_accept == pytest.approx(0.1 * 10.0)
        assert rr.expected_cost_review == pytest.approx(2.0)
        assert rr.expected_cost_leave == pytest.approx(5.0)

    def test_lowest_cost_action_high_p(self):
        rec = make_fake_unc(confidence=0.95)
        rr = self.calc.compute_risk_record(rec, cal_prob=0.95)
        assert rr.lowest_cost_action == Action.ACCEPT_REPAIR

    def test_lowest_cost_action_low_p(self):
        rec = make_fake_unc(confidence=0.3)
        rr = self.calc.compute_risk_record(rec, cal_prob=0.3)
        assert rr.lowest_cost_action == Action.HUMAN_REVIEW

    def test_minimum_expected_cost_matches(self):
        rec = make_fake_unc(confidence=0.7)
        rr = self.calc.compute_risk_record(rec, cal_prob=0.7)
        assert rr.minimum_expected_cost == min(
            rr.expected_cost_accept,
            rr.expected_cost_review,
            rr.expected_cost_leave,
        )

    def test_costs_stored(self):
        rec = make_fake_unc()
        rr = self.calc.compute_risk_record(rec, cal_prob=0.5)
        assert rr.costs is self.costs

    def test_to_dict_has_required_keys(self):
        rec = make_fake_unc()
        rr = self.calc.compute_risk_record(rec, cal_prob=0.7)
        d = rr.to_dict()
        for key in [
            "dataset", "row_id", "column", "top_candidate",
            "calibrated_probability", "probability_wrong",
            "expected_cost_accept", "expected_cost_review", "expected_cost_leave",
            "minimum_expected_cost", "lowest_cost_action",
            "c_wrong_repair", "c_human_review", "c_leave_error",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_action_is_string(self):
        rec = make_fake_unc()
        rr = self.calc.compute_risk_record(rec, cal_prob=0.9)
        d = rr.to_dict()
        assert isinstance(d["lowest_cost_action"], str)

    def test_invalid_prob_raises(self):
        rec = make_fake_unc()
        with pytest.raises(ValueError):
            self.calc.compute_risk_record(rec, cal_prob=1.5)


# ===========================================================================
# compute_risk_batch tests
# ===========================================================================


class TestComputeRiskBatch:

    def setup_method(self):
        self.calc = RiskCalculator(make_cost_config())

    def test_basic_batch(self):
        recs = [make_fake_unc(row_id=i, column="age") for i in range(5)]
        probs = [0.9, 0.8, 0.7, 0.6, 0.5]
        result = self.calc.compute_risk_batch(recs, probs)
        assert len(result) == 5

    def test_length_mismatch_raises(self):
        recs = [make_fake_unc(row_id=i) for i in range(3)]
        probs = [0.9, 0.8]
        with pytest.raises(ValueError, match="length"):
            self.calc.compute_risk_batch(recs, probs)

    def test_skip_columns_excludes_records(self):
        recs = [
            make_fake_unc(row_id=0, column="age"),
            make_fake_unc(row_id=1, column="Unnamed: 0"),
            make_fake_unc(row_id=2, column="age"),
        ]
        probs = [0.9, 0.9, 0.9]
        result = self.calc.compute_risk_batch(recs, probs, skip_columns=["Unnamed: 0"])
        assert len(result) == 2
        for rr in result:
            assert rr.column != "Unnamed: 0"

    def test_skip_multiple_columns(self):
        recs = [
            make_fake_unc(row_id=0, column="age"),
            make_fake_unc(row_id=1, column="Unnamed: 0"),
            make_fake_unc(row_id=2, column="Sample code number"),
            make_fake_unc(row_id=3, column="glucose"),
        ]
        probs = [0.9] * 4
        result = self.calc.compute_risk_batch(
            recs, probs,
            skip_columns=["Unnamed: 0", "Sample code number"]
        )
        assert len(result) == 2

    def test_empty_input(self):
        result = self.calc.compute_risk_batch([], [])
        assert result == []

    def test_all_skipped(self):
        recs = [make_fake_unc(row_id=0, column="Unnamed: 0")]
        result = self.calc.compute_risk_batch(recs, [0.5],
                                              skip_columns=["Unnamed: 0"])
        assert result == []

    def test_no_skip_columns(self):
        recs = [make_fake_unc(row_id=i) for i in range(4)]
        result = self.calc.compute_risk_batch(recs, [0.9]*4, skip_columns=None)
        assert len(result) == 4

    def test_row_ids_preserved(self):
        recs = [make_fake_unc(row_id=100 + i) for i in range(3)]
        result = self.calc.compute_risk_batch(recs, [0.9]*3)
        assert [rr.row_id for rr in result] == [100, 101, 102]


# ===========================================================================
# RiskEvaluator tests
# ===========================================================================


def _make_clean_dirty_dfs():
    """Small synthetic clean/dirty pair for testing."""
    clean = pd.DataFrame({
        "age": [25, 30, 35, 40],
        "income": [0, 1, 0, 1],
    })
    dirty = pd.DataFrame({
        "age": [99, 30, 35, 40],   # row 0 corrupted
        "income": [0, 1, 0, 1],
    })
    return clean, dirty


def _make_risk_record(row_id=0, column="age", top_candidate=25,
                      action=Action.ACCEPT_REPAIR, cal_prob=0.9) -> RiskRecord:
    costs = make_cost_config()
    ec_map = {
        Action.ACCEPT_REPAIR: (1 - cal_prob) * costs.c_wrong_repair,
        Action.HUMAN_REVIEW: costs.c_human_review,
        Action.LEAVE_UNCHANGED: costs.c_leave_error,
    }
    min_cost = ec_map[action]
    return RiskRecord(
        dataset="test",
        row_id=row_id,
        column=column,
        top_candidate=top_candidate,
        calibrated_probability=cal_prob,
        probability_wrong=1 - cal_prob,
        expected_cost_accept=ec_map[Action.ACCEPT_REPAIR],
        expected_cost_review=ec_map[Action.HUMAN_REVIEW],
        expected_cost_leave=ec_map[Action.LEAVE_UNCHANGED],
        minimum_expected_cost=min_cost,
        lowest_cost_action=action,
        costs=costs,
    )


class TestRiskEvaluator:

    def setup_method(self):
        self.clean, self.dirty = _make_clean_dirty_dfs()
        self.costs = make_cost_config()
        self.evaluator = RiskEvaluator(self.clean, self.dirty, "test")

    def test_empty_records_returns_report(self):
        report = self.evaluator.evaluate([], self.costs)
        assert report.n_records == 0
        assert report.empirical_cost_policy == 0.0

    def test_correct_repair_accept_costs_zero(self):
        # row_id=0, column="age", top=25 (correct clean value)
        rr = _make_risk_record(row_id=0, column="age", top_candidate=25,
                               action=Action.ACCEPT_REPAIR, cal_prob=0.9)
        report = self.evaluator.evaluate([rr], self.costs)
        assert report.n_correct_top1 == 1
        assert report.n_incorrect_top1 == 0
        assert report.empirical_cost_policy == pytest.approx(0.0)

    def test_incorrect_repair_accept_costs_wrong(self):
        # row_id=0, column="age", top=99 (wrong, clean is 25)
        rr = _make_risk_record(row_id=0, column="age", top_candidate=99,
                               action=Action.ACCEPT_REPAIR, cal_prob=0.9)
        report = self.evaluator.evaluate([rr], self.costs)
        assert report.n_incorrect_top1 == 1
        assert report.empirical_cost_policy == pytest.approx(self.costs.c_wrong_repair)

    def test_review_action_costs_c_human_review(self):
        rr = _make_risk_record(row_id=0, column="age", top_candidate=25,
                               action=Action.HUMAN_REVIEW, cal_prob=0.3)
        report = self.evaluator.evaluate([rr], self.costs)
        assert report.empirical_cost_policy == pytest.approx(self.costs.c_human_review)
        assert report.action_counts.review == 1

    def test_leave_action_costs_c_leave(self):
        rr = _make_risk_record(row_id=0, column="age", top_candidate=25,
                               action=Action.LEAVE_UNCHANGED, cal_prob=0.3)
        report = self.evaluator.evaluate([rr], self.costs)
        assert report.empirical_cost_policy == pytest.approx(self.costs.c_leave_error)
        assert report.action_counts.leave == 1

    def test_action_counts_totals(self):
        recs = [
            _make_risk_record(row_id=0, column="age", top_candidate=25,
                              action=Action.ACCEPT_REPAIR),
            _make_risk_record(row_id=1, column="age", top_candidate=25,
                              action=Action.HUMAN_REVIEW),
            _make_risk_record(row_id=2, column="age", top_candidate=35,
                              action=Action.LEAVE_UNCHANGED),
        ]
        report = self.evaluator.evaluate(recs, self.costs)
        assert report.action_counts.accept == 1
        assert report.action_counts.review == 1
        assert report.action_counts.leave == 1
        assert report.action_counts.total == 3

    def test_n_records_matches(self):
        recs = [_make_risk_record(row_id=i, column="age", top_candidate=25+i)
                for i in range(4)]
        report = self.evaluator.evaluate(recs, self.costs)
        assert report.n_records == 4

    def test_baseline_accept_all(self):
        # 1 correct, 1 incorrect → cost_accept_all = 1 * C_wrong
        recs = [
            _make_risk_record(row_id=0, column="age", top_candidate=25,  # correct
                              action=Action.ACCEPT_REPAIR),
            _make_risk_record(row_id=0, column="age", top_candidate=99,  # wrong
                              action=Action.ACCEPT_REPAIR),
        ]
        report = self.evaluator.evaluate(recs, self.costs)
        assert report.empirical_cost_accept_all == pytest.approx(
            1 * self.costs.c_wrong_repair
        )

    def test_baseline_review_all(self):
        n = 3
        recs = [_make_risk_record(row_id=i, column="age", top_candidate=25)
                for i in range(n)]
        report = self.evaluator.evaluate(recs, self.costs)
        assert report.empirical_cost_review_all == pytest.approx(
            n * self.costs.c_human_review
        )

    def test_baseline_leave_all(self):
        """Always-LEAVE charges C_leave only for genuinely corrupted cells.

        In the fixture only row 0 is corrupted, so leaving rows 0-2 alone
        costs one C_leave, not three (consistent with E[LEAVE] = q * C_leave;
        a false-positive detection left alone costs nothing)."""
        n = 3
        recs = [_make_risk_record(row_id=i, column="age", top_candidate=25)
                for i in range(n)]
        report = self.evaluator.evaluate(recs, self.costs)
        assert report.n_genuinely_corrupted == 1
        assert report.empirical_cost_leave_all == pytest.approx(
            1 * self.costs.c_leave_error
        )

    def test_baseline_leave_all_charges_every_corrupted_cell(self):
        clean = pd.DataFrame({"age": [25, 30, 35]})
        dirty = pd.DataFrame({"age": [99, 99, 99]})
        ev = RiskEvaluator(clean, dirty, "test")
        recs = [_make_risk_record(row_id=i, column="age", top_candidate=25)
                for i in range(3)]
        report = ev.evaluate(recs, self.costs)
        assert report.n_genuinely_corrupted == 3
        assert report.empirical_cost_leave_all == pytest.approx(
            3 * self.costs.c_leave_error
        )

    def test_leave_on_false_positive_costs_nothing(self):
        # row 1 is NOT corrupted in the fixture
        rr = _make_risk_record(row_id=1, column="age", top_candidate=30,
                               action=Action.LEAVE_UNCHANGED, cal_prob=0.3)
        report = self.evaluator.evaluate([rr], self.costs)
        assert report.empirical_cost_policy == pytest.approx(0.0)

    def test_mean_cost_per_record(self):
        recs = [_make_risk_record(row_id=i, column="age", top_candidate=25,
                                  action=Action.HUMAN_REVIEW)
                for i in range(4)]
        report = self.evaluator.evaluate(recs, self.costs)
        expected_mean = report.empirical_cost_policy / 4
        assert report.mean_cost_per_record_policy == pytest.approx(expected_mean)

    def test_cost_savings_vs_leave_sign_follows_genuine_corruption(self):
        """Reviewing every flagged cell only beats leaving them alone when
        most of the flags are real. With 1 corrupted of 4 (fixture),
        review_all = 4 * 2 = 8 > leave_all = 1 * 5 = 5, so savings < 0.
        With all 4 corrupted, leave_all = 20 > 8, so savings > 0."""
        n = 4
        recs = [_make_risk_record(row_id=i, column="age", top_candidate=25,
                                  action=Action.HUMAN_REVIEW)
                for i in range(n)]
        report = self.evaluator.evaluate(recs, self.costs)
        assert report.cost_savings_vs_leave_all < 0

        clean = pd.DataFrame({"age": [25, 30, 35, 40]})
        dirty = pd.DataFrame({"age": [99, 99, 99, 99]})
        report_all = RiskEvaluator(clean, dirty, "test").evaluate(recs, self.costs)
        assert report_all.cost_savings_vs_leave_all > 0

    def test_adult_limitation_string(self):
        evaluator = RiskEvaluator(self.clean, self.dirty, "adult")
        recs = [_make_risk_record()]
        report = evaluator.evaluate(recs, self.costs)
        assert "adult" in report.limitation.lower() or "fnlwgt" in report.limitation.lower()

    def test_non_adult_no_limitation(self):
        recs = [_make_risk_record()]
        report = self.evaluator.evaluate(recs, self.costs)
        assert report.limitation == ""

    def test_missing_row_id_not_correct(self):
        # row_id=999 doesn't exist in clean_df → should be counted as incorrect
        rr = _make_risk_record(row_id=999, column="age", top_candidate=25,
                               action=Action.ACCEPT_REPAIR)
        report = self.evaluator.evaluate([rr], self.costs)
        assert report.n_incorrect_top1 == 1

    def test_float_comparison_tolerance(self):
        # Clean value is int 25, candidate is float 25.0 → should match
        clean = pd.DataFrame({"val": [25]})
        dirty = pd.DataFrame({"val": [99]})
        ev = RiskEvaluator(clean, dirty, "test")
        rr = _make_risk_record(row_id=0, column="val", top_candidate=25.0,
                               action=Action.ACCEPT_REPAIR)
        report = ev.evaluate([rr], self.costs)
        assert report.n_correct_top1 == 1

    def test_report_is_empirical_cost_report_type(self):
        recs = [_make_risk_record()]
        report = self.evaluator.evaluate(recs, self.costs)
        assert isinstance(report, EmpiricalCostReport)


# ===========================================================================
# ActionCounts tests
# ===========================================================================


class TestActionCounts:

    def test_total(self):
        ac = ActionCounts(accept=3, review=2, leave=1)
        assert ac.total == 6

    def test_empty_total(self):
        ac = ActionCounts()
        assert ac.total == 0

    def test_fractions_sum_to_one(self):
        ac = ActionCounts(accept=3, review=3, leave=4)
        fracs = ac.fractions()
        assert sum(fracs.values()) == pytest.approx(1.0)

    def test_fractions_empty_no_division_error(self):
        ac = ActionCounts()
        fracs = ac.fractions()
        assert fracs["accept"] == 0.0


# ===========================================================================
# print_risk_report smoke tests
# ===========================================================================


class TestPrintRiskReport:

    def test_runs_without_error(self, capsys):
        clean, dirty = _make_clean_dirty_dfs()
        ev = RiskEvaluator(clean, dirty, "test")
        costs = make_cost_config()
        recs = [
            _make_risk_record(row_id=0, column="age", top_candidate=25,
                              action=Action.ACCEPT_REPAIR),
            _make_risk_record(row_id=0, column="age", top_candidate=99,
                              action=Action.HUMAN_REVIEW),
        ]
        report = ev.evaluate(recs, costs)
        print_risk_report(report)
        captured = capsys.readouterr()
        assert "EMPIRICAL COST REPORT" in captured.out

    def test_contains_dataset_name(self, capsys):
        clean, dirty = _make_clean_dirty_dfs()
        ev = RiskEvaluator(clean, dirty, "breast_cancer")
        costs = make_cost_config()
        recs = [_make_risk_record()]
        report = ev.evaluate(recs, costs)
        print_risk_report(report)
        captured = capsys.readouterr()
        assert "BREAST_CANCER" in captured.out

    def test_shows_action_distribution(self, capsys):
        clean, dirty = _make_clean_dirty_dfs()
        ev = RiskEvaluator(clean, dirty, "test")
        costs = make_cost_config()
        recs = [_make_risk_record(action=Action.ACCEPT_REPAIR)]
        report = ev.evaluate(recs, costs)
        print_risk_report(report)
        captured = capsys.readouterr()
        assert "ACCEPT_REPAIR" in captured.out

    def test_adult_limitation_shown(self, capsys):
        clean, dirty = _make_clean_dirty_dfs()
        ev = RiskEvaluator(clean, dirty, "adult")
        costs = make_cost_config()
        recs = [_make_risk_record()]
        report = ev.evaluate(recs, costs)
        print_risk_report(report)
        captured = capsys.readouterr()
        assert "[NOTE]" in captured.out
