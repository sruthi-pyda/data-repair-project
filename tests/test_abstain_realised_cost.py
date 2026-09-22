"""
tests/test_abstain_realised_cost.py
------------------------------------
Step 3 of docs/three_actions_plan.md: the REALISED cost of ABSTAIN is
charged only when the flagged cell was genuinely corrupted.

Why
---
Phase 8 now computes  E[ABSTAIN] = q * C_abstain  with
q = P(cell genuinely corrupted).  For the evaluation to be the outcome
that expectation is taken over, the realised charge must be
    C_abstain  if dirty != clean   (the error persists)
    0          if dirty == clean   (false alarm; leaving it alone was right)
Charging C_abstain unconditionally would punish the policy for correctly
ignoring false alarms (the whole Adult situation).

Scope
-----
Only src/evaluation/cost_metrics.py::CostEvaluator is covered here.  The
Phase 7 demo evaluator (src/risk/evaluation.py) is intentionally NOT
changed in this step: two existing Phase 7 tests
(tests/test_cost_model.py::TestRiskEvaluator::test_baseline_leave_all and
::test_cost_savings_positive_when_policy_cheaper) build their "always
LEAVE" expectation on rows that are NOT corrupted in the shared fixture,
so the conditional rule would change their expectations.  See the Step 3
report for the options.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.baselines.baselines import BaselineRecord
from src.decision.policy import DecisionAction
from src.evaluation.cost_metrics import CostEvaluator, CostReport
from src.risk.cost_model import CostConfig


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

C_WRONG, C_HUMAN, C_ABSTAIN = 10.0, 3.0, 5.0


def _rec(
    row_id: int,
    col: str,
    dirty_value: Any,
    top_candidate: Any,
    action: DecisionAction,
) -> BaselineRecord:
    return BaselineRecord(
        method_name="test_method",
        dataset="test",
        row_id=row_id,
        column=col,
        dirty_value=dirty_value,
        top_candidate=top_candidate,
        raw_confidence=0.8,
        action=action,
    )


def _ev(clean_df: pd.DataFrame) -> CostEvaluator:
    return CostEvaluator(clean_df, c_wrong=C_WRONG, c_human=C_HUMAN,
                         c_abstain=C_ABSTAIN)


ABS = DecisionAction.ABSTAIN
HUM = DecisionAction.HUMAN_REVIEW
AUTO = DecisionAction.AUTO_REPAIR


# ---------------------------------------------------------------------------
# 1. Single-record ABSTAIN charging
# ---------------------------------------------------------------------------


class TestAbstainCharge:

    def test_abstain_on_clean_cell_costs_zero(self):
        clean = pd.DataFrame({"val": [7]})
        r = _ev(clean).evaluate([_rec(0, "val", 7, 7, ABS)], "t")
        assert r.total_realized_cost == 0.0
        assert r.n_abstain == 1
        assert r.n_abstain_on_clean == 1
        assert r.n_abstain_on_corrupted == 0

    def test_abstain_on_corrupted_cell_costs_c_abstain(self):
        clean = pd.DataFrame({"val": [7]})
        r = _ev(clean).evaluate([_rec(0, "val", 99, 7, ABS)], "t")
        assert math.isclose(r.total_realized_cost, C_ABSTAIN)
        assert r.n_abstain_on_corrupted == 1
        assert r.n_abstain_on_clean == 0

    def test_nan_dirty_vs_nan_clean_is_not_corrupted(self):
        clean = pd.DataFrame({"val": [np.nan]})
        r = _ev(clean).evaluate([_rec(0, "val", np.nan, 1, ABS)], "t")
        assert r.total_realized_cost == 0.0
        assert r.n_abstain_on_clean == 1
        assert r.n_genuinely_corrupted == 0

    def test_nan_dirty_vs_value_clean_is_corrupted(self):
        clean = pd.DataFrame({"val": [4.0]})
        r = _ev(clean).evaluate([_rec(0, "val", np.nan, 4.0, ABS)], "t")
        assert math.isclose(r.total_realized_cost, C_ABSTAIN)
        assert r.n_abstain_on_corrupted == 1
        assert r.n_genuinely_corrupted == 1

    def test_none_dirty_vs_value_clean_is_corrupted(self):
        clean = pd.DataFrame({"val": ["a"]})
        r = _ev(clean).evaluate([_rec(0, "val", None, "a", ABS)], "t")
        assert math.isclose(r.total_realized_cost, C_ABSTAIN)

    def test_numeric_tolerance_371_float_vs_int_not_corrupted(self):
        clean = pd.DataFrame({"val": [371]})
        r = _ev(clean).evaluate([_rec(0, "val", 371.0, 371, ABS)], "t")
        assert r.total_realized_cost == 0.0
        assert r.n_abstain_on_clean == 1

    def test_string_whitespace_is_not_corruption_under_comparator(self):
        # ValueComparator strips surrounding whitespace on both sides.
        clean = pd.DataFrame({"val": ["abc"]})
        r = _ev(clean).evaluate([_rec(0, "val", " abc ", "abc", ABS)], "t")
        assert r.total_realized_cost == 0.0

    def test_string_difference_is_corruption(self):
        clean = pd.DataFrame({"val": ["abc"]})
        r = _ev(clean).evaluate([_rec(0, "val", "abd", "abc", ABS)], "t")
        assert math.isclose(r.total_realized_cost, C_ABSTAIN)

    def test_missing_row_in_clean_df_is_treated_as_corrupted(self):
        # Conservative: we cannot prove it was a false alarm, so charge.
        clean = pd.DataFrame({"val": [1]})
        r = _ev(clean).evaluate([_rec(999, "val", 1, 1, ABS)], "t")
        assert math.isclose(r.total_realized_cost, C_ABSTAIN)
        assert r.n_abstain_on_corrupted == 1

    def test_missing_column_in_clean_df_is_treated_as_corrupted(self):
        clean = pd.DataFrame({"val": [1]})
        r = _ev(clean).evaluate([_rec(0, "other", 1, 1, ABS)], "t")
        assert math.isclose(r.total_realized_cost, C_ABSTAIN)
        assert r.n_abstain_on_corrupted == 1


# ---------------------------------------------------------------------------
# 2. Batches: totals, counts, other actions unchanged
# ---------------------------------------------------------------------------


class TestBatchTotals:

    def setup_method(self):
        # rows: 0 clean, 1 corrupted, 2 clean, 3 corrupted, 4 corrupted, 5 clean
        self.clean = pd.DataFrame({"val": [1, 2, 3, 4, 5, 6]})
        self.ev = _ev(self.clean)

    def test_mixed_abstain_batch_totals_exactly(self):
        recs = [
            _rec(0, "val", 1, 1, ABS),     # clean → 0
            _rec(1, "val", 99, 2, ABS),    # corrupted → 5
            _rec(2, "val", 3, 3, ABS),     # clean → 0
            _rec(3, "val", 99, 4, ABS),    # corrupted → 5
        ]
        r = self.ev.evaluate(recs, "t")
        assert math.isclose(r.total_realized_cost, 2 * C_ABSTAIN)
        assert r.n_abstain == 4
        assert r.n_abstain_on_clean == 2
        assert r.n_abstain_on_corrupted == 2
        assert r.n_abstain_on_clean + r.n_abstain_on_corrupted == r.n_abstain
        assert math.isclose(r.mean_cost_per_record, 2 * C_ABSTAIN / 4)

    def test_mixed_actions_add_up_exactly(self):
        recs = [
            _rec(0, "val", 1, 1, AUTO),     # clean, correct repair → 0
            _rec(1, "val", 99, 99, AUTO),   # corrupted, wrong repair → 10
            _rec(2, "val", 3, 3, HUM),      # clean, human → 3 (unconditional)
            _rec(3, "val", 99, 4, HUM),     # corrupted, human → 3
            _rec(4, "val", 99, 5, ABS),     # corrupted, abstain → 5
            _rec(5, "val", 6, 6, ABS),      # clean, abstain → 0
        ]
        r = self.ev.evaluate(recs, "t")
        assert math.isclose(r.total_realized_cost, 0 + 10 + 3 + 3 + 5 + 0)
        assert r.n_auto == 2 and r.n_human == 2 and r.n_abstain == 2
        assert r.n_correct_auto == 1 and r.n_wrong_auto == 1

    def test_n_genuinely_corrupted_counts_across_all_actions(self):
        recs = [
            _rec(0, "val", 1, 1, AUTO),    # clean
            _rec(1, "val", 99, 2, AUTO),   # corrupted
            _rec(2, "val", 3, 3, HUM),     # clean
            _rec(3, "val", 99, 4, HUM),    # corrupted
            _rec(4, "val", 99, 5, ABS),    # corrupted
            _rec(5, "val", 6, 6, ABS),     # clean
        ]
        r = self.ev.evaluate(recs, "t")
        assert r.n_genuinely_corrupted == 3

    def test_human_and_auto_charging_unchanged_without_abstain(self):
        # Human review is charged whether or not the cell was corrupted;
        # auto repair is charged by correctness only.  Same totals as before.
        recs = [
            _rec(0, "val", 1, 1, HUM),      # clean → 3
            _rec(1, "val", 99, 2, HUM),     # corrupted → 3
            _rec(2, "val", 3, 3, AUTO),     # clean, correct → 0
            _rec(3, "val", 99, 99, AUTO),   # corrupted, wrong → 10
            _rec(4, "val", 99, 5, AUTO),    # corrupted, correct → 0
        ]
        r = self.ev.evaluate(recs, "t")
        old_semantics = 2 * C_HUMAN + 1 * C_WRONG
        assert math.isclose(r.total_realized_cost, old_semantics)
        assert r.n_abstain == 0
        assert r.n_abstain_on_clean == 0 and r.n_abstain_on_corrupted == 0

    def test_consistency_all_corrupted_equals_old_unconditional_formula(self):
        # When every cell is genuinely corrupted (q would be 1), the new
        # rule is invisible: total == n_abstain * c_abstain.
        recs = [_rec(i, "val", 99, i + 1, ABS) for i in range(6)]
        r = self.ev.evaluate(recs, "t")
        assert r.n_genuinely_corrupted == 6
        assert math.isclose(r.total_realized_cost, r.n_abstain * C_ABSTAIN)
        assert r.n_abstain_on_corrupted == r.n_abstain


# ---------------------------------------------------------------------------
# 3. CostReport: new fields, property, to_dict, backwards compatibility
# ---------------------------------------------------------------------------


class TestCostReportFields:

    def test_abstain_precision_no_abstentions_is_zero(self):
        clean = pd.DataFrame({"val": [1]})
        r = _ev(clean).evaluate([_rec(0, "val", 1, 1, HUM)], "t")
        assert r.n_abstain == 0
        assert r.abstain_precision == 0.0

    def test_abstain_precision_all_clean_is_one(self):
        clean = pd.DataFrame({"val": [1, 2]})
        recs = [_rec(0, "val", 1, 1, ABS), _rec(1, "val", 2, 2, ABS)]
        r = _ev(clean).evaluate(recs, "t")
        assert math.isclose(r.abstain_precision, 1.0)

    def test_abstain_precision_all_corrupted_is_zero(self):
        clean = pd.DataFrame({"val": [1, 2]})
        recs = [_rec(0, "val", 9, 1, ABS), _rec(1, "val", 9, 2, ABS)]
        r = _ev(clean).evaluate(recs, "t")
        assert r.abstain_precision == 0.0

    def test_abstain_precision_mixed(self):
        clean = pd.DataFrame({"val": [1, 2, 3, 4]})
        recs = [
            _rec(0, "val", 1, 1, ABS),   # clean
            _rec(1, "val", 9, 2, ABS),   # corrupted
            _rec(2, "val", 9, 3, ABS),   # corrupted
            _rec(3, "val", 9, 4, ABS),   # corrupted
        ]
        r = _ev(clean).evaluate(recs, "t")
        assert math.isclose(r.abstain_precision, 0.25)

    def test_to_dict_has_new_keys(self):
        clean = pd.DataFrame({"val": [1, 2]})
        recs = [_rec(0, "val", 1, 1, ABS), _rec(1, "val", 9, 2, ABS)]
        d = _ev(clean).evaluate(recs, "t").to_dict()
        assert d["n_abstain_on_clean"] == 1
        assert d["n_abstain_on_corrupted"] == 1
        assert d["n_genuinely_corrupted"] == 1
        assert math.isclose(d["abstain_precision"], 0.5)

    def test_cost_report_without_new_fields_still_constructs(self):
        r = CostReport(method_name="m", dataset="d", n_records=3,
                       n_abstain=2, total_realized_cost=10.0)
        assert r.n_abstain_on_clean == 0
        assert r.n_abstain_on_corrupted == 0
        assert r.n_genuinely_corrupted == 0
        assert r.abstain_precision == 0.0
        assert "n_abstain_on_clean" in r.to_dict()

    def test_empty_records_report_has_zero_new_fields(self):
        clean = pd.DataFrame({"val": [1]})
        r = _ev(clean).evaluate([], "t")
        assert r.n_records == 0
        assert r.n_abstain_on_clean == 0
        assert r.n_genuinely_corrupted == 0
        assert r.abstain_precision == 0.0

    def test_from_cost_config_uses_conditional_rule(self):
        costs = CostConfig(c_wrong_repair=10, c_human_review=2,
                           c_leave_error=5, dataset_name="test")
        clean = pd.DataFrame({"val": [1, 2]})
        ev = CostEvaluator.from_cost_config(clean, costs)
        recs = [_rec(0, "val", 1, 1, ABS), _rec(1, "val", 9, 2, ABS)]
        r = ev.evaluate(recs, "test")
        assert math.isclose(r.total_realized_cost, 5.0)
        assert r.cost_per_abstain == 5.0


# ---------------------------------------------------------------------------
# 4. No-leakage: evaluator signature is unchanged, records carry no truth
# ---------------------------------------------------------------------------


class TestNoLeakage:

    def test_evaluate_signature_does_not_take_clean_df(self):
        import inspect
        sig = inspect.signature(CostEvaluator.evaluate)
        assert "clean_df" not in sig.parameters
        assert "dirty_df" not in sig.parameters

    def test_report_carries_no_clean_values(self):
        clean = pd.DataFrame({"val": [12345]})
        r = _ev(clean).evaluate([_rec(0, "val", 1, 1, ABS)], "t")
        assert "12345" not in str(r.to_dict())


# ---------------------------------------------------------------------------
# 5. Integration: Adult, all-abstain policy, all false positives → cost 0
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAdultAllAbstainIntegration:

    def test_adult_all_abstain_costs_zero(self):
        from src.baselines.baselines import wrap_proposed_method
        from src.decision.policy import DecisionPolicy
        from src.pipeline.runner import run_pipeline

        bundle = run_pipeline("adult", n_records=200)
        costs = CostConfig(c_wrong_repair=10, c_human_review=2,
                           c_leave_error=5, dataset_name="adult")

        # Force abstention on every record via q = 0 (E[ABSTAIN] = 0).
        policy = DecisionPolicy.from_cost_config(costs)
        decisions = [policy.decide(p, corruption_probability=0.0)
                     for p in bundle.calibrated_probs]
        assert all(d.action == ABS for d in decisions)

        records = wrap_proposed_method(
            bundle.unc_records, bundle.calibrated_probs, decisions, "adult",
        )
        report = CostEvaluator.from_cost_config(bundle.clean_df, costs) \
            .evaluate(records, "adult")

        assert report.n_records == 200
        assert report.n_abstain == 200
        # Adult detections are fnlwgt IQR false positives: none corrupted.
        assert report.n_genuinely_corrupted == 0
        assert report.n_abstain_on_clean == 200
        assert report.n_abstain_on_corrupted == 0
        assert report.total_realized_cost == 0.0
        assert math.isclose(report.abstain_precision, 1.0)

        # Under the OLD unconditional semantics this batch would have cost
        # n_abstain * c_leave_error — i.e. the policy would be punished for
        # correctly ignoring 200 false alarms.
        old_semantics_cost = report.n_abstain * costs.c_leave_error
        assert old_semantics_cost == 200 * 5
        assert report.total_realized_cost < old_semantics_cost
