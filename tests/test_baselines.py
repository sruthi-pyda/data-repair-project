"""
tests/test_baselines.py
------------------------
Phase 9: Unit tests for baseline repair-decision policies.

Test groups
-----------
 1.  AlwaysAutoBaseline — always selects AUTO_REPAIR
 2.  AlwaysAutoBaseline — candidate and confidence preservation
 3.  AlwaysAutoBaseline — determinism
 4.  AlwaysAutoBaseline — empty input
 5.  ConfidenceThresholdBaseline — threshold routing logic
 6.  ConfidenceThresholdBaseline — threshold edge cases (boundary values)
 7.  ConfidenceThresholdBaseline — method_name includes threshold
 8.  ConfidenceThresholdBaseline — invalid threshold raises ValueError
 9.  ConfidenceThresholdBaseline — multiple thresholds produce different results
10.  ConfidenceThresholdBaseline — empty input
11.  sweep_thresholds helper
12.  BaselineRecord fields and to_dict
13.  wrap_proposed_method adapter
14.  No clean_df access (signature inspection)
15.  No ground-truth leakage (field inspection)
16.  Determinism across repeated runs
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, Dict

import pytest

from src.baselines.baselines import (
    AlwaysAutoBaseline,
    BaselineRecord,
    ConfidenceThresholdBaseline,
    DEFAULT_THRESHOLDS,
    sweep_thresholds,
    wrap_proposed_method,
)
from src.decision.policy import DecisionAction, DecisionPolicy, DecisionResult


# ===========================================================================
# Helpers — minimal UncertaintyRecord stand-in
# ===========================================================================


@dataclass
class _FakeUncRecord:
    """
    Minimal stand-in for UncertaintyRecord.
    Contains NO clean values — only dirty-data information.
    """
    dataset: str = "test_dataset"
    row_id: int = 0
    column: str = "col_a"
    dirty_value: Any = "dirty_val"
    error_type: str = "NOVEL_CATEGORY"
    repair_method: str = "mock"
    top_candidate: Any = "repair_val"
    raw_confidence: float = 0.75
    self_consistency: float = 0.8
    entropy: float = 0.5
    normalized_entropy: float = 0.3
    n_samples: int = 5
    vote_counts: Dict = None
    candidate_frequencies: Dict = None
    n_candidates: int = 3

    def __post_init__(self):
        if self.vote_counts is None:
            self.vote_counts = {}
        if self.candidate_frequencies is None:
            self.candidate_frequencies = {}


def make_records(
    n: int,
    confidence: float = 0.75,
    dataset: str = "test",
) -> list:
    """Create n fake UncertaintyRecords with the given raw_confidence."""
    return [
        _FakeUncRecord(
            dataset=dataset,
            row_id=i,
            column="col",
            dirty_value=f"dirty_{i}",
            top_candidate=f"repair_{i}",
            raw_confidence=confidence,
        )
        for i in range(n)
    ]


def make_varied_records() -> list:
    """Records with varying confidence levels for threshold tests."""
    confidences = [0.4, 0.55, 0.65, 0.75, 0.82, 0.9, 0.95]
    return [
        _FakeUncRecord(
            row_id=i,
            raw_confidence=c,
            top_candidate=f"repair_{i}",
        )
        for i, c in enumerate(confidences)
    ]


# ===========================================================================
# Group 1: AlwaysAutoBaseline — always AUTO_REPAIR
# ===========================================================================


class TestAlwaysAutoAction:
    """AlwaysAutoBaseline always selects AUTO_REPAIR regardless of confidence."""

    def test_single_record_always_auto(self):
        baseline = AlwaysAutoBaseline()
        records = baseline.run(make_records(1, confidence=0.3))
        assert records[0].action == DecisionAction.AUTO_REPAIR

    def test_high_confidence_auto(self):
        baseline = AlwaysAutoBaseline()
        records = baseline.run(make_records(1, confidence=0.99))
        assert records[0].action == DecisionAction.AUTO_REPAIR

    def test_low_confidence_still_auto(self):
        baseline = AlwaysAutoBaseline()
        records = baseline.run(make_records(1, confidence=0.01))
        assert records[0].action == DecisionAction.AUTO_REPAIR

    def test_all_records_auto(self):
        baseline = AlwaysAutoBaseline()
        recs = make_varied_records()
        results = baseline.run(recs)
        assert all(r.action == DecisionAction.AUTO_REPAIR for r in results)

    def test_zero_confidence_auto(self):
        rec = _FakeUncRecord(raw_confidence=0.0)
        baseline = AlwaysAutoBaseline()
        results = baseline.run([rec])
        assert results[0].action == DecisionAction.AUTO_REPAIR

    def test_output_length_matches_input(self):
        baseline = AlwaysAutoBaseline()
        recs = make_records(7)
        results = baseline.run(recs)
        assert len(results) == 7

    def test_method_name(self):
        baseline = AlwaysAutoBaseline()
        recs = make_records(1)
        results = baseline.run(recs)
        assert results[0].method_name == "always_auto"


# ===========================================================================
# Group 2: AlwaysAutoBaseline — candidate and confidence preservation
# ===========================================================================


class TestAlwaysAutoFields:
    """BaselineRecord must faithfully copy fields from UncertaintyRecord."""

    def setup_method(self):
        rec = _FakeUncRecord(
            dataset="adult",
            row_id=42,
            column="occupation",
            dirty_value="Adm-clericl",
            top_candidate="Adm-clerical",
            raw_confidence=0.82,
        )
        self.result = AlwaysAutoBaseline(dataset_name="adult").run([rec])[0]

    def test_dataset_preserved(self):
        assert self.result.dataset == "adult"

    def test_row_id_preserved(self):
        assert self.result.row_id == 42

    def test_column_preserved(self):
        assert self.result.column == "occupation"

    def test_dirty_value_preserved(self):
        assert self.result.dirty_value == "Adm-clericl"

    def test_top_candidate_preserved(self):
        assert self.result.top_candidate == "Adm-clerical"

    def test_raw_confidence_preserved(self):
        assert math.isclose(self.result.raw_confidence, 0.82)

    def test_no_calibrated_probability(self):
        assert self.result.calibrated_probability is None

    def test_no_threshold(self):
        assert self.result.threshold is None

    def test_no_cost_fields(self):
        assert self.result.expected_cost_auto is None
        assert self.result.expected_cost_human is None
        assert self.result.expected_cost_abstain is None


# ===========================================================================
# Group 3: AlwaysAutoBaseline — determinism
# ===========================================================================


class TestAlwaysAutoDeterminism:
    """Repeated runs with the same input must produce identical output."""

    def test_repeated_run_same_actions(self):
        baseline = AlwaysAutoBaseline()
        recs = make_varied_records()
        run1 = baseline.run(recs)
        run2 = baseline.run(recs)
        actions1 = [r.action for r in run1]
        actions2 = [r.action for r in run2]
        assert actions1 == actions2

    def test_repeated_run_same_candidates(self):
        baseline = AlwaysAutoBaseline()
        recs = make_records(5, confidence=0.6)
        run1 = [r.top_candidate for r in baseline.run(recs)]
        run2 = [r.top_candidate for r in baseline.run(recs)]
        assert run1 == run2


# ===========================================================================
# Group 4: AlwaysAutoBaseline — empty input
# ===========================================================================


class TestAlwaysAutoEmpty:
    def test_empty_input_returns_empty(self):
        baseline = AlwaysAutoBaseline()
        assert baseline.run([]) == []


# ===========================================================================
# Group 5: ConfidenceThresholdBaseline — threshold routing
# ===========================================================================


class TestConfidenceThresholdRouting:
    """Correct routing: >= threshold → AUTO, < threshold → HUMAN_REVIEW."""

    def test_exactly_at_threshold_is_auto(self):
        baseline = ConfidenceThresholdBaseline(threshold=0.8)
        rec = _FakeUncRecord(raw_confidence=0.8)
        result = baseline.run([rec])[0]
        assert result.action == DecisionAction.AUTO_REPAIR

    def test_above_threshold_is_auto(self):
        baseline = ConfidenceThresholdBaseline(threshold=0.8)
        rec = _FakeUncRecord(raw_confidence=0.9)
        result = baseline.run([rec])[0]
        assert result.action == DecisionAction.AUTO_REPAIR

    def test_below_threshold_is_human(self):
        baseline = ConfidenceThresholdBaseline(threshold=0.8)
        rec = _FakeUncRecord(raw_confidence=0.79)
        result = baseline.run([rec])[0]
        assert result.action == DecisionAction.HUMAN_REVIEW

    def test_well_below_threshold_is_human(self):
        baseline = ConfidenceThresholdBaseline(threshold=0.8)
        rec = _FakeUncRecord(raw_confidence=0.1)
        result = baseline.run([rec])[0]
        assert result.action == DecisionAction.HUMAN_REVIEW

    def test_threshold_recorded_in_result(self):
        baseline = ConfidenceThresholdBaseline(threshold=0.7)
        result = baseline.run(make_records(1))[0]
        assert math.isclose(result.threshold, 0.7)

    def test_mixed_records_correct_split(self):
        baseline = ConfidenceThresholdBaseline(threshold=0.75)
        recs = [
            _FakeUncRecord(row_id=0, raw_confidence=0.74),  # HUMAN
            _FakeUncRecord(row_id=1, raw_confidence=0.75),  # AUTO
            _FakeUncRecord(row_id=2, raw_confidence=0.90),  # AUTO
            _FakeUncRecord(row_id=3, raw_confidence=0.30),  # HUMAN
        ]
        results = baseline.run(recs)
        assert results[0].action == DecisionAction.HUMAN_REVIEW
        assert results[1].action == DecisionAction.AUTO_REPAIR
        assert results[2].action == DecisionAction.AUTO_REPAIR
        assert results[3].action == DecisionAction.HUMAN_REVIEW

    def test_decide_one_at_threshold(self):
        baseline = ConfidenceThresholdBaseline(threshold=0.6)
        assert baseline.decide_one(0.6) == DecisionAction.AUTO_REPAIR

    def test_decide_one_below_threshold(self):
        baseline = ConfidenceThresholdBaseline(threshold=0.6)
        assert baseline.decide_one(0.59) == DecisionAction.HUMAN_REVIEW

    def test_no_abstain_action_produced(self):
        """Threshold baseline only produces AUTO or HUMAN, never ABSTAIN."""
        baseline = ConfidenceThresholdBaseline(threshold=0.8)
        results = baseline.run(make_varied_records())
        for r in results:
            assert r.action != DecisionAction.ABSTAIN


# ===========================================================================
# Group 6: ConfidenceThresholdBaseline — boundary edge cases
# ===========================================================================


class TestConfidenceThresholdEdgeCases:
    def test_threshold_one_all_human_except_perfect(self):
        baseline = ConfidenceThresholdBaseline(threshold=1.0)
        recs = [
            _FakeUncRecord(raw_confidence=0.99),  # HUMAN
            _FakeUncRecord(raw_confidence=1.0),   # AUTO
        ]
        results = baseline.run(recs)
        assert results[0].action == DecisionAction.HUMAN_REVIEW
        assert results[1].action == DecisionAction.AUTO_REPAIR

    def test_very_small_threshold_all_auto(self):
        baseline = ConfidenceThresholdBaseline(threshold=0.01)
        recs = make_records(5, confidence=0.05)
        results = baseline.run(recs)
        assert all(r.action == DecisionAction.AUTO_REPAIR for r in results)

    def test_confidence_one_always_auto(self):
        baseline = ConfidenceThresholdBaseline(threshold=0.8)
        rec = _FakeUncRecord(raw_confidence=1.0)
        result = baseline.run([rec])[0]
        assert result.action == DecisionAction.AUTO_REPAIR

    def test_confidence_zero_always_human(self):
        baseline = ConfidenceThresholdBaseline(threshold=0.5)
        rec = _FakeUncRecord(raw_confidence=0.0)
        result = baseline.run([rec])[0]
        assert result.action == DecisionAction.HUMAN_REVIEW

    def test_nearly_equal_confidence_and_threshold(self):
        # floating-point: 0.7999... < 0.8 → HUMAN
        baseline = ConfidenceThresholdBaseline(threshold=0.8)
        rec = _FakeUncRecord(raw_confidence=0.8 - 1e-10)
        result = baseline.run([rec])[0]
        assert result.action == DecisionAction.HUMAN_REVIEW


# ===========================================================================
# Group 7: ConfidenceThresholdBaseline — method_name
# ===========================================================================


class TestConfidenceThresholdMethodName:
    def test_method_name_includes_threshold(self):
        baseline = ConfidenceThresholdBaseline(threshold=0.8)
        assert "0.80" in baseline.method_name

    def test_method_name_in_record(self):
        baseline = ConfidenceThresholdBaseline(threshold=0.7)
        result = baseline.run(make_records(1))[0]
        assert "0.70" in result.method_name

    def test_different_thresholds_different_method_names(self):
        b1 = ConfidenceThresholdBaseline(threshold=0.6)
        b2 = ConfidenceThresholdBaseline(threshold=0.9)
        assert b1.method_name != b2.method_name

    def test_method_name_contains_prefix(self):
        baseline = ConfidenceThresholdBaseline(threshold=0.5)
        assert baseline.method_name.startswith("confidence_threshold")


# ===========================================================================
# Group 8: ConfidenceThresholdBaseline — invalid threshold raises ValueError
# ===========================================================================


class TestConfidenceThresholdInvalidInput:
    def test_threshold_zero_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            ConfidenceThresholdBaseline(threshold=0.0)

    def test_threshold_negative_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            ConfidenceThresholdBaseline(threshold=-0.1)

    def test_threshold_greater_than_one_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            ConfidenceThresholdBaseline(threshold=1.01)

    def test_threshold_two_raises(self):
        with pytest.raises(ValueError):
            ConfidenceThresholdBaseline(threshold=2.0)

    def test_threshold_one_is_valid(self):
        """threshold=1.0 is valid (only perfect confidence → AUTO)."""
        baseline = ConfidenceThresholdBaseline(threshold=1.0)
        assert baseline.threshold == 1.0


# ===========================================================================
# Group 9: Multiple thresholds produce different results
# ===========================================================================


class TestMultipleThresholds:
    """Higher threshold → fewer AUTO_REPAIR decisions."""

    def test_higher_threshold_fewer_auto(self):
        recs = make_varied_records()
        bl_low = ConfidenceThresholdBaseline(threshold=0.5)
        bl_high = ConfidenceThresholdBaseline(threshold=0.9)
        auto_low = sum(1 for r in bl_low.run(recs) if r.action == DecisionAction.AUTO_REPAIR)
        auto_high = sum(1 for r in bl_high.run(recs) if r.action == DecisionAction.AUTO_REPAIR)
        assert auto_high <= auto_low

    def test_all_thresholds_produce_distinct_outputs_on_varied_data(self):
        recs = make_varied_records()
        method_names = set()
        for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
            bl = ConfidenceThresholdBaseline(threshold=t)
            name = bl.run(recs)[0].method_name
            method_names.add(name)
        # All 5 thresholds should produce distinct method names
        assert len(method_names) == 5

    def test_threshold_050_all_auto_when_all_above(self):
        recs = make_records(4, confidence=0.6)
        bl = ConfidenceThresholdBaseline(threshold=0.5)
        results = bl.run(recs)
        assert all(r.action == DecisionAction.AUTO_REPAIR for r in results)

    def test_threshold_090_all_human_when_all_below(self):
        recs = make_records(4, confidence=0.6)
        bl = ConfidenceThresholdBaseline(threshold=0.9)
        results = bl.run(recs)
        assert all(r.action == DecisionAction.HUMAN_REVIEW for r in results)


# ===========================================================================
# Group 10: ConfidenceThresholdBaseline — empty input
# ===========================================================================


class TestConfidenceThresholdEmpty:
    def test_empty_input_returns_empty(self):
        baseline = ConfidenceThresholdBaseline(threshold=0.8)
        assert baseline.run([]) == []

    def test_output_length_matches_input(self):
        baseline = ConfidenceThresholdBaseline(threshold=0.8)
        recs = make_records(6, confidence=0.9)
        assert len(baseline.run(recs)) == 6


# ===========================================================================
# Group 11: sweep_thresholds helper
# ===========================================================================


class TestSweepThresholds:
    def test_default_thresholds_used(self):
        recs = make_records(3)
        results = sweep_thresholds(recs)
        assert set(results.keys()) == set(DEFAULT_THRESHOLDS)

    def test_custom_thresholds(self):
        recs = make_records(3)
        results = sweep_thresholds(recs, thresholds=[0.5, 0.9])
        assert set(results.keys()) == {0.5, 0.9}

    def test_each_threshold_returns_record_list(self):
        recs = make_records(4)
        results = sweep_thresholds(recs)
        for t, records in results.items():
            assert isinstance(records, list)
            assert len(records) == 4

    def test_empty_records_sweep(self):
        results = sweep_thresholds([])
        for t, records in results.items():
            assert records == []

    def test_invalid_threshold_in_sweep_raises(self):
        recs = make_records(2)
        with pytest.raises(ValueError):
            sweep_thresholds(recs, thresholds=[0.8, 0.0])

    def test_sweep_monotonic_auto_rate(self):
        """As threshold increases, auto rate cannot increase."""
        recs = make_varied_records()
        thresholds = sorted(DEFAULT_THRESHOLDS)
        auto_rates = []
        for t in thresholds:
            bl = ConfidenceThresholdBaseline(threshold=t)
            n_auto = sum(1 for r in bl.run(recs) if r.action == DecisionAction.AUTO_REPAIR)
            auto_rates.append(n_auto)
        # auto_rates must be non-increasing as threshold increases
        for i in range(len(auto_rates) - 1):
            assert auto_rates[i] >= auto_rates[i + 1], (
                f"Auto rate increased from t={thresholds[i]} to t={thresholds[i+1]}: "
                f"{auto_rates[i]} → {auto_rates[i+1]}"
            )


# ===========================================================================
# Group 12: BaselineRecord fields and to_dict
# ===========================================================================


class TestBaselineRecord:
    def make_record(self) -> BaselineRecord:
        return BaselineRecord(
            method_name="always_auto",
            dataset="adult",
            row_id=7,
            column="occupation",
            dirty_value="Adm-clericl",
            top_candidate="Adm-clerical",
            raw_confidence=0.82,
            action=DecisionAction.AUTO_REPAIR,
        )

    def test_to_dict_contains_method_name(self):
        d = self.make_record().to_dict()
        assert d["method_name"] == "always_auto"

    def test_to_dict_contains_action(self):
        d = self.make_record().to_dict()
        assert d["action"] == "AUTO_REPAIR"

    def test_to_dict_contains_row_id(self):
        d = self.make_record().to_dict()
        assert d["row_id"] == 7

    def test_to_dict_none_for_unused_fields(self):
        d = self.make_record().to_dict()
        assert d["threshold"] is None
        assert d["calibrated_probability"] is None
        assert d["expected_cost_auto"] is None

    def test_to_dict_is_flat_dict(self):
        d = self.make_record().to_dict()
        assert isinstance(d, dict)
        for v in d.values():
            assert not isinstance(v, dict), f"Nested dict found: {v}"

    def test_metadata_included_in_to_dict(self):
        rec = BaselineRecord(
            method_name="test",
            dataset="d",
            row_id=0,
            column="c",
            dirty_value="x",
            top_candidate="y",
            raw_confidence=0.5,
            action=DecisionAction.HUMAN_REVIEW,
            metadata={"repair_method": "mock", "error_type": "MISSING"},
        )
        d = rec.to_dict()
        assert d["repair_method"] == "mock"
        assert d["error_type"] == "MISSING"


# ===========================================================================
# Group 13: wrap_proposed_method adapter
# ===========================================================================


class TestWrapProposedMethod:
    """wrap_proposed_method converts Phase 8 DecisionResult → BaselineRecord."""

    def make_decision_result(self, p: float = 0.8, action=DecisionAction.AUTO_REPAIR) -> DecisionResult:
        policy = DecisionPolicy(c_wrong=10, c_human=3, c_abstain=5)
        return policy.decide(p)

    def test_correct_method_name(self):
        recs = make_records(1)
        probs = [0.8]
        drs = [self.make_decision_result()]
        result = wrap_proposed_method(recs, probs, drs)[0]
        assert result.method_name == "calibrated_risk_aware"

    def test_calibrated_probability_stored(self):
        recs = make_records(1)
        probs = [0.75]
        drs = [self.make_decision_result(0.75)]
        result = wrap_proposed_method(recs, probs, drs)[0]
        assert math.isclose(result.calibrated_probability, 0.75)

    def test_cost_fields_populated(self):
        recs = make_records(1)
        probs = [0.9]
        drs = [self.make_decision_result(0.9)]
        result = wrap_proposed_method(recs, probs, drs)[0]
        assert result.expected_cost_auto is not None
        assert result.expected_cost_human is not None
        assert result.expected_cost_abstain is not None

    def test_action_matches_decision_result(self):
        recs = make_records(1)
        dr = self.make_decision_result(0.99)  # high p → AUTO
        result = wrap_proposed_method(recs, [0.99], [dr])[0]
        assert result.action == dr.action

    def test_length_mismatch_raises(self):
        recs = make_records(2)
        probs = [0.8]
        drs = [self.make_decision_result()]
        with pytest.raises(ValueError, match="same length"):
            wrap_proposed_method(recs, probs, drs)

    def test_all_probs_mismatch_raises(self):
        recs = make_records(1)
        probs = [0.8, 0.9]
        drs = [self.make_decision_result()]
        with pytest.raises(ValueError, match="same length"):
            wrap_proposed_method(recs, probs, drs)

    def test_empty_input(self):
        result = wrap_proposed_method([], [], [])
        assert result == []

    def test_batch_length_matches(self):
        n = 5
        recs = make_records(n)
        probs = [0.7] * n
        drs = [self.make_decision_result(0.7)] * n
        results = wrap_proposed_method(recs, probs, drs)
        assert len(results) == n

    def test_dirty_value_preserved(self):
        rec = _FakeUncRecord(dirty_value="my_dirty_val", top_candidate="my_repair")
        dr = self.make_decision_result()
        result = wrap_proposed_method([rec], [0.8], [dr])[0]
        assert result.dirty_value == "my_dirty_val"

    def test_top_candidate_preserved(self):
        rec = _FakeUncRecord(top_candidate="specific_repair")
        dr = self.make_decision_result()
        result = wrap_proposed_method([rec], [0.8], [dr])[0]
        assert result.top_candidate == "specific_repair"


# ===========================================================================
# Group 14: No clean_df access — signature inspection
# ===========================================================================


class TestNoCleanDfAccess:
    """Baselines must not accept clean_df in any decision-making function."""

    def test_always_auto_run_no_clean_df(self):
        sig = inspect.signature(AlwaysAutoBaseline.run)
        assert "clean_df" not in sig.parameters

    def test_always_auto_init_no_clean_df(self):
        sig = inspect.signature(AlwaysAutoBaseline.__init__)
        assert "clean_df" not in sig.parameters

    def test_threshold_run_no_clean_df(self):
        sig = inspect.signature(ConfidenceThresholdBaseline.run)
        assert "clean_df" not in sig.parameters

    def test_threshold_init_no_clean_df(self):
        sig = inspect.signature(ConfidenceThresholdBaseline.__init__)
        assert "clean_df" not in sig.parameters

    def test_decide_one_no_clean_df(self):
        sig = inspect.signature(ConfidenceThresholdBaseline.decide_one)
        assert "clean_df" not in sig.parameters

    def test_sweep_thresholds_no_clean_df(self):
        sig = inspect.signature(sweep_thresholds)
        assert "clean_df" not in sig.parameters

    def test_wrap_proposed_method_no_clean_df(self):
        sig = inspect.signature(wrap_proposed_method)
        assert "clean_df" not in sig.parameters


# ===========================================================================
# Group 15: No ground-truth leakage — BaselineRecord field inspection
# ===========================================================================


class TestNoGroundTruthLeakage:
    """BaselineRecord must not store ground-truth or clean values."""

    def test_no_clean_value_field(self):
        fields = {f.name for f in BaselineRecord.__dataclass_fields__.values()}
        assert "clean_value" not in fields

    def test_no_ground_truth_field(self):
        fields = {f.name for f in BaselineRecord.__dataclass_fields__.values()}
        assert "ground_truth" not in fields

    def test_no_is_correct_field(self):
        fields = {f.name for f in BaselineRecord.__dataclass_fields__.values()}
        assert "is_correct" not in fields

    def test_no_clean_df_field(self):
        fields = {f.name for f in BaselineRecord.__dataclass_fields__.values()}
        assert "clean_df" not in fields

    def test_always_auto_result_has_no_ground_truth(self):
        baseline = AlwaysAutoBaseline()
        rec = _FakeUncRecord()
        result = baseline.run([rec])[0]
        assert not hasattr(result, "clean_value")
        assert not hasattr(result, "ground_truth")

    def test_threshold_result_has_no_ground_truth(self):
        baseline = ConfidenceThresholdBaseline(threshold=0.8)
        rec = _FakeUncRecord()
        result = baseline.run([rec])[0]
        assert not hasattr(result, "clean_value")
        assert not hasattr(result, "ground_truth")


# ===========================================================================
# Group 16: Determinism across repeated runs
# ===========================================================================


class TestDeterminism:
    """Both baselines must be fully deterministic."""

    def test_always_auto_deterministic(self):
        baseline = AlwaysAutoBaseline()
        recs = make_varied_records()
        run1 = [r.action for r in baseline.run(recs)]
        run2 = [r.action for r in baseline.run(recs)]
        assert run1 == run2

    def test_threshold_deterministic(self):
        baseline = ConfidenceThresholdBaseline(threshold=0.75)
        recs = make_varied_records()
        run1 = [r.action for r in baseline.run(recs)]
        run2 = [r.action for r in baseline.run(recs)]
        assert run1 == run2

    def test_sweep_deterministic(self):
        recs = make_records(5, confidence=0.7)
        sweep1 = {t: [r.action for r in rs] for t, rs in sweep_thresholds(recs).items()}
        sweep2 = {t: [r.action for r in rs] for t, rs in sweep_thresholds(recs).items()}
        assert sweep1 == sweep2

    def test_wrap_proposed_deterministic(self):
        recs = make_records(3)
        probs = [0.6, 0.8, 0.9]
        policy = DecisionPolicy(c_wrong=10, c_human=3, c_abstain=5)
        drs = [policy.decide(p) for p in probs]
        r1 = [r.action for r in wrap_proposed_method(recs, probs, drs)]
        r2 = [r.action for r in wrap_proposed_method(recs, probs, drs)]
        assert r1 == r2
