"""
tests/test_calibrated_abstention.py
------------------------------------
Step 2 of docs/three_actions_plan.md: the policy (Phase 8) and the risk
model (Phase 7) accept a calibrated corruption probability q and charge

    E[ABSTAIN] = q · C_abstain          (q defaults to 1.0)

instead of the flat C_abstain.

Test groups
-----------
 1. Backward compatibility — q omitted is bit-for-bit q = 1.0, and the
    dataclasses default the new field to 1.0.
 2. Linearity — abstain cost scales with q and is independent of p.
 3. (p, q) regions — all three actions occur on one cost config and every
    grid point matches an independently computed argmin.
 4. Exact ties resolve by the documented priority AUTO > HUMAN > ABSTAIN.
 5. Invalid q is rejected (mirroring the p validation).
 6. decide_batch with / without corruption_probabilities.
 7. breakeven_auto_vs_abstain(q) formula and clamping.
 8. Phase 7 and Phase 8 agree on costs, action mapping and tie order.
 9. Serialisation: RiskRecord.to_dict and wrap_proposed_method metadata.
10. Reason string shows q × C_abstain.
11. No-leakage: new parameters do not smuggle ground truth.

Expected actions in group 3 are computed from the algebra inside the test,
never by calling the code under test.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from src.baselines.baselines import wrap_proposed_method
from src.decision.policy import DecisionAction, DecisionPolicy, DecisionResult
from src.risk.cost_model import (
    Action,
    CostConfig,
    ExpectedCosts,
    RiskCalculator,
    RiskRecord,
)

COSTS_YAML = Path(__file__).resolve().parent.parent / "configs" / "costs.yaml"
DATASETS = ["adult", "breast_cancer", "smartfactory"]

P_GRID = [round(i / 100, 2) for i in range(101)]
Q_GRID = [round(i / 20, 2) for i in range(21)]

# Phase 7 → Phase 8 action names
ACTION_MAP = {
    Action.ACCEPT_REPAIR: DecisionAction.AUTO_REPAIR,
    Action.HUMAN_REVIEW: DecisionAction.HUMAN_REVIEW,
    Action.LEAVE_UNCHANGED: DecisionAction.ABSTAIN,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bc_policy() -> DecisionPolicy:
    """Breast-cancer shipped costs: wrong=50, human=5, abstain=10."""
    return DecisionPolicy(c_wrong=50.0, c_human=5.0, c_abstain=10.0)


def bc_costs() -> CostConfig:
    return CostConfig(
        c_wrong_repair=50.0, c_human_review=5.0, c_leave_error=10.0,
        dataset_name="breast_cancer",
    )


def expected_action(p: float, q: float, cw: float, ch: float, ca: float) -> DecisionAction:
    """
    Independent re-derivation of the decision rule.

        E[auto]    = (1 - p) * cw
        E[human]   = ch
        E[abstain] = q * ca

    argmin with ties broken AUTO > HUMAN > ABSTAIN.
    """
    e_auto = (1.0 - p) * cw
    e_human = ch
    e_abstain = q * ca
    m = min(e_auto, e_human, e_abstain)
    if e_auto == m:
        return DecisionAction.AUTO_REPAIR
    if e_human == m:
        return DecisionAction.HUMAN_REVIEW
    return DecisionAction.ABSTAIN


def _same_result(a: DecisionResult, b: DecisionResult) -> bool:
    """Field-by-field equality (frozen dataclass __eq__ does this too, but
    be explicit so a failure names the field)."""
    for f in dataclasses.fields(DecisionResult):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        assert va == vb, f"field {f.name!r} differs: {va!r} vs {vb!r}"
    return True


@dataclass
class _FakeUnc:
    """Minimal stand-in for UncertaintyRecord (the fields the adapters read)."""
    dataset: str = "test"
    row_id: int = 0
    column: str = "age"
    dirty_value: Any = 999
    error_type: str = "OUT_OF_RANGE"
    repair_method: str = "statistical"
    top_candidate: Any = 30
    raw_confidence: float = 0.9


# ===========================================================================
# Group 1: Backward compatibility
# ===========================================================================


class TestBackwardCompatibility:

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_decide_without_q_equals_q_one_on_grid(self, dataset):
        costs = CostConfig.from_yaml(COSTS_YAML, dataset)
        policy = DecisionPolicy.from_cost_config(costs)
        for p in P_GRID:
            assert _same_result(policy.decide(p), policy.decide(p, 1.0))

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_decide_without_q_reports_q_one(self, dataset):
        costs = CostConfig.from_yaml(COSTS_YAML, dataset)
        policy = DecisionPolicy.from_cost_config(costs)
        for p in P_GRID:
            r = policy.decide(p)
            assert r.corruption_probability == 1.0
            assert r.expected_cost_abstain == policy.c_abstain

    def test_decision_result_default_q_is_one(self):
        r = DecisionResult(
            calibrated_probability=0.5,
            probability_wrong=0.5,
            expected_cost_auto=25.0,
            expected_cost_human=5.0,
            expected_cost_abstain=10.0,
            action=DecisionAction.HUMAN_REVIEW,
            reason="r",
            c_wrong=50.0,
            c_human=5.0,
            c_abstain=10.0,
        )
        assert r.corruption_probability == 1.0

    def test_decision_result_q_is_last_field(self):
        names = [f.name for f in dataclasses.fields(DecisionResult)]
        assert names[-1] == "corruption_probability"
        assert names[:-1] == [
            "calibrated_probability", "probability_wrong",
            "expected_cost_auto", "expected_cost_human",
            "expected_cost_abstain", "action", "reason",
            "c_wrong", "c_human", "c_abstain",
        ]

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_compute_expected_costs_without_q_equals_q_one(self, dataset):
        calc = RiskCalculator(CostConfig.from_yaml(COSTS_YAML, dataset))
        for p in P_GRID:
            a = calc.compute_expected_costs(p)
            b = calc.compute_expected_costs(p, 1.0)
            assert a == b
            assert a.corruption_probability == 1.0
            assert a.leave == calc.costs.c_leave_error

    def test_expected_costs_default_q_is_one(self):
        ec = ExpectedCosts(
            p=0.5, accept=25.0, review=5.0, leave=10.0,
            best_action=Action.HUMAN_REVIEW, min_cost=5.0,
        )
        assert ec.corruption_probability == 1.0

    def test_risk_record_default_q_is_one(self):
        rr = RiskRecord(
            dataset="d", row_id=0, column="c", top_candidate=1,
            calibrated_probability=0.5, probability_wrong=0.5,
            expected_cost_accept=25.0, expected_cost_review=5.0,
            expected_cost_leave=10.0, minimum_expected_cost=5.0,
            lowest_cost_action=Action.HUMAN_REVIEW, costs=bc_costs(),
        )
        assert rr.corruption_probability == 1.0
        assert rr.to_dict()["corruption_probability"] == 1.0

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_sensitivity_without_q_equals_q_one(self, dataset):
        calc = RiskCalculator(CostConfig.from_yaml(COSTS_YAML, dataset))
        assert calc.sensitivity_over_p(21) == calc.sensitivity_over_p(21, 1.0)
        assert calc.sensitivity_over_p(101) == calc.sensitivity_over_p(
            101, corruption_probability=1.0
        )

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_analytical_thresholds_without_q_equals_q_one(self, dataset):
        calc = RiskCalculator(CostConfig.from_yaml(COSTS_YAML, dataset))
        assert calc.analytical_thresholds() == calc.analytical_thresholds(1.0)

    def test_analytical_thresholds_q_one_reproduces_original_formula(self):
        calc = RiskCalculator(bc_costs())
        th = calc.analytical_thresholds()
        assert th.p_accept_vs_review == pytest.approx(1 - 5 / 50)
        assert th.p_accept_vs_leave == pytest.approx(1 - 10 / 50)
        assert th.review_cheaper_than_leave is True
        # q = 1 must not leak a "q×" term into the message
        assert "×" not in th.recommended_policy

    def test_compute_risk_record_without_q(self):
        calc = RiskCalculator(bc_costs())
        rr = calc.compute_risk_record(_FakeUnc(), 0.6)
        assert rr.corruption_probability == 1.0
        assert rr.expected_cost_leave == 10.0
        assert rr.lowest_cost_action == Action.HUMAN_REVIEW

    def test_compute_risk_batch_without_q(self):
        calc = RiskCalculator(bc_costs())
        recs = [_FakeUnc(row_id=i) for i in range(4)]
        ps = [0.1, 0.5, 0.9, 0.99]
        a = calc.compute_risk_batch(recs, ps)
        b = calc.compute_risk_batch(recs, ps, corruption_probs=[1.0] * 4)
        assert a == b
        assert all(r.corruption_probability == 1.0 for r in a)


# ===========================================================================
# Group 2: Linearity in q
# ===========================================================================


class TestLinearity:

    @pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 1.0])
    def test_abstain_cost_is_q_times_c_abstain(self, q):
        policy = bc_policy()
        for p in (0.0, 0.3, 0.7, 1.0):
            r = policy.decide(p, q)
            assert r.expected_cost_abstain == pytest.approx(q * 10.0, abs=1e-12)
            assert r.corruption_probability == q

    @pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 1.0])
    def test_abstain_cost_independent_of_p(self, q):
        policy = bc_policy()
        vals = {policy.decide(p, q).expected_cost_abstain for p in P_GRID}
        assert len(vals) == 1

    @pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 1.0])
    def test_auto_and_human_costs_unaffected_by_q(self, q):
        policy = bc_policy()
        for p in P_GRID:
            base = policy.decide(p)
            r = policy.decide(p, q)
            assert r.expected_cost_auto == base.expected_cost_auto
            assert r.expected_cost_human == base.expected_cost_human

    @pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 1.0])
    def test_phase7_leave_cost_is_q_times_c_leave(self, q):
        calc = RiskCalculator(bc_costs())
        for p in (0.0, 0.3, 0.7, 1.0):
            ec = calc.compute_expected_costs(p, q)
            assert ec.leave == pytest.approx(q * 10.0, abs=1e-12)
            assert ec.corruption_probability == q

    def test_q_zero_makes_abstain_free(self):
        r = bc_policy().decide(0.5, 0.0)
        assert r.expected_cost_abstain == 0.0
        assert r.action == DecisionAction.ABSTAIN


# ===========================================================================
# Group 3: (p, q) regions
# ===========================================================================


class TestRegions:
    """Breast-cancer costs: wrong=50, human=5, abstain=10."""

    CW, CH, CA = 50.0, 5.0, 10.0

    def test_all_three_actions_occur_on_grid(self):
        policy = bc_policy()
        seen = {policy.decide(p, q).action for p in P_GRID for q in Q_GRID}
        assert seen == {
            DecisionAction.AUTO_REPAIR,
            DecisionAction.HUMAN_REVIEW,
            DecisionAction.ABSTAIN,
        }

    def test_abstain_unreachable_at_q_one(self):
        """The structural problem this step fixes: with q=1, c_abstain >
        c_human so ABSTAIN never fires on this config."""
        policy = bc_policy()
        assert all(
            policy.decide(p, 1.0).action != DecisionAction.ABSTAIN
            for p in P_GRID
        )

    def test_every_grid_point_matches_algebra(self):
        policy = bc_policy()
        for p in P_GRID:
            for q in Q_GRID:
                got = policy.decide(p, q).action
                exp = expected_action(p, q, self.CW, self.CH, self.CA)
                assert got == exp, f"p={p} q={q}: got {got}, expected {exp}"

    def test_auto_region(self):
        policy = bc_policy()
        for p in P_GRID:
            for q in Q_GRID:
                is_auto = (1 - p) * self.CW <= min(self.CH, q * self.CA)
                assert (policy.decide(p, q).action == DecisionAction.AUTO_REPAIR) == is_auto

    def test_abstain_region(self):
        policy = bc_policy()
        for p in P_GRID:
            for q in Q_GRID:
                is_abstain = (q * self.CA < self.CH) and (q * self.CA < (1 - p) * self.CW)
                assert (policy.decide(p, q).action == DecisionAction.ABSTAIN) == is_abstain

    def test_human_region(self):
        policy = bc_policy()
        for p in P_GRID:
            for q in Q_GRID:
                is_human = (self.CH < (1 - p) * self.CW) and (self.CH <= q * self.CA)
                assert (policy.decide(p, q).action == DecisionAction.HUMAN_REVIEW) == is_human

    def test_worked_examples_from_docstring(self):
        policy = bc_policy()
        assert policy.decide(0.95, 1.0).action == DecisionAction.AUTO_REPAIR
        assert policy.decide(0.60, 1.0).action == DecisionAction.HUMAN_REVIEW
        assert policy.decide(0.60, 0.2).action == DecisionAction.ABSTAIN

    def test_phase7_every_grid_point_matches_algebra(self):
        calc = RiskCalculator(bc_costs())
        for p in P_GRID:
            for q in Q_GRID:
                got = ACTION_MAP[calc.compute_expected_costs(p, q).best_action]
                exp = expected_action(p, q, self.CW, self.CH, self.CA)
                assert got == exp, f"p={p} q={q}: got {got}, expected {exp}"


# ===========================================================================
# Group 4: Exact ties
# ===========================================================================


class TestTies:
    """Costs chosen so all products are exact in binary."""

    def test_auto_ties_abstain_resolves_to_auto(self):
        # (1-p)*c_wrong = 0.25*8 = 2 ; q*c_abstain = 0.5*4 = 2 ; c_human = 3
        policy = DecisionPolicy(c_wrong=8.0, c_human=3.0, c_abstain=4.0)
        r = policy.decide(0.75, 0.5)
        assert r.expected_cost_auto == r.expected_cost_abstain == 2.0
        assert r.action == DecisionAction.AUTO_REPAIR

    def test_abstain_ties_human_resolves_to_human(self):
        # q*c_abstain = 0.5*4 = 2 = c_human ; auto = 0.5*8 = 4
        policy = DecisionPolicy(c_wrong=8.0, c_human=2.0, c_abstain=4.0)
        r = policy.decide(0.5, 0.5)
        assert r.expected_cost_human == r.expected_cost_abstain == 2.0
        assert r.expected_cost_auto > 2.0
        assert r.action == DecisionAction.HUMAN_REVIEW

    def test_three_way_tie_resolves_to_auto(self):
        # auto = 0.25*8 = 2 ; human = 2 ; abstain = 0.5*4 = 2
        policy = DecisionPolicy(c_wrong=8.0, c_human=2.0, c_abstain=4.0)
        r = policy.decide(0.75, 0.5)
        assert r.expected_cost_auto == r.expected_cost_human == r.expected_cost_abstain
        assert r.action == DecisionAction.AUTO_REPAIR

    def test_phase7_ties_same_priority(self):
        costs = CostConfig(c_wrong_repair=8.0, c_human_review=2.0, c_leave_error=4.0)
        calc = RiskCalculator(costs)
        # ACCEPT vs LEAVE tie → ACCEPT
        assert calc.compute_expected_costs(0.75, 0.5).best_action == Action.ACCEPT_REPAIR
        # REVIEW vs LEAVE tie → REVIEW
        assert calc.compute_expected_costs(0.5, 0.5).best_action == Action.HUMAN_REVIEW


# ===========================================================================
# Group 5: Invalid q
# ===========================================================================


class TestInvalidQ:

    @pytest.mark.parametrize("q", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")])
    def test_decide_rejects_out_of_range(self, q):
        with pytest.raises(ValueError, match="corruption_probability"):
            bc_policy().decide(0.5, q)

    @pytest.mark.parametrize("q", [-0.1, 1.1, float("nan"), float("inf")])
    def test_phase7_rejects_out_of_range(self, q):
        calc = RiskCalculator(bc_costs())
        with pytest.raises(ValueError):
            calc.compute_expected_costs(0.5, q)
        with pytest.raises(ValueError):
            calc.compute_risk_record(_FakeUnc(), 0.5, q)
        with pytest.raises(ValueError):
            calc.sensitivity_over_p(5, q)
        with pytest.raises(ValueError):
            calc.analytical_thresholds(q)

    @pytest.mark.parametrize("q", [-0.1, 1.1, float("nan")])
    def test_breakeven_rejects_out_of_range(self, q):
        with pytest.raises(ValueError):
            bc_policy().breakeven_auto_vs_abstain(q)

    def test_q_zero_and_one_are_valid_boundaries(self):
        policy = bc_policy()
        assert policy.decide(0.5, 0.0).corruption_probability == 0.0
        assert policy.decide(0.5, 1.0).corruption_probability == 1.0

    @pytest.mark.parametrize("bad", [True, "0.5", "abc", None, [0.5]])
    def test_q_coercion_mirrors_p(self, bad):
        """
        The p check is `float(x)` then a range test.  q must behave the same
        way for the same input: raise the same exception type, or accept
        with the same coerced value.  (Today float(True)=1.0 and
        float("0.5")=0.5 are ACCEPTED for p, so they are accepted for q too;
        "abc", None and [0.5] raise.)
        """
        policy = bc_policy()
        try:
            p_out = policy.decide(bad)
            p_exc = None
        except Exception as e:  # noqa: BLE001
            p_out, p_exc = None, type(e)
        try:
            q_out = policy.decide(0.5, bad)
            q_exc = None
        except Exception as e:  # noqa: BLE001
            q_out, q_exc = None, type(e)

        assert p_exc == q_exc
        if p_exc is None:
            assert q_out.corruption_probability == p_out.calibrated_probability


# ===========================================================================
# Group 6: decide_batch
# ===========================================================================


class TestDecideBatch:

    def test_none_equals_per_item_q_one(self):
        policy = bc_policy()
        ps = [0.0, 0.2, 0.5, 0.9, 1.0]
        got = policy.decide_batch(ps)
        exp = [policy.decide(p, 1.0) for p in ps]
        assert got == exp
        assert policy.decide_batch(ps, None) == exp

    def test_explicit_list(self):
        policy = bc_policy()
        ps = [0.6, 0.6, 0.6]
        qs = [1.0, 0.4, 0.0]
        got = policy.decide_batch(ps, qs)
        assert [r.action for r in got] == [
            DecisionAction.HUMAN_REVIEW,
            DecisionAction.ABSTAIN,
            DecisionAction.ABSTAIN,
        ]
        assert [r.corruption_probability for r in got] == qs

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            bc_policy().decide_batch([0.5, 0.6], [1.0])
        with pytest.raises(ValueError, match="length"):
            bc_policy().decide_batch([0.5], [1.0, 0.5])

    def test_empty(self):
        assert bc_policy().decide_batch([], []) == []
        assert bc_policy().decide_batch([]) == []

    def test_invalid_q_in_list_raises(self):
        with pytest.raises(ValueError):
            bc_policy().decide_batch([0.5, 0.5], [0.5, 1.5])

    # Phase 7 mirror -------------------------------------------------------

    def test_phase7_batch_explicit_and_mismatch(self):
        calc = RiskCalculator(bc_costs())
        recs = [_FakeUnc(row_id=i) for i in range(3)]
        out = calc.compute_risk_batch(recs, [0.6] * 3, corruption_probs=[1.0, 0.4, 0.0])
        assert [r.lowest_cost_action for r in out] == [
            Action.HUMAN_REVIEW, Action.LEAVE_UNCHANGED, Action.LEAVE_UNCHANGED,
        ]
        with pytest.raises(ValueError, match="corruption_probs"):
            calc.compute_risk_batch(recs, [0.6] * 3, corruption_probs=[1.0])

    def test_phase7_batch_q_aligned_with_skip_columns(self):
        """q is indexed against the unfiltered input, so skipping a column
        must drop its q too rather than shifting the remaining ones."""
        calc = RiskCalculator(bc_costs())
        recs = [
            _FakeUnc(row_id=0, column="Unnamed: 0"),
            _FakeUnc(row_id=1, column="age"),
            _FakeUnc(row_id=2, column="Unnamed: 0"),
            _FakeUnc(row_id=3, column="age"),
        ]
        qs = [0.9, 0.1, 0.9, 0.3]
        out = calc.compute_risk_batch(
            recs, [0.6] * 4, skip_columns=["Unnamed: 0"], corruption_probs=qs
        )
        assert [r.row_id for r in out] == [1, 3]
        assert [r.corruption_probability for r in out] == [0.1, 0.3]


# ===========================================================================
# Group 7: breakeven_auto_vs_abstain(q)
# ===========================================================================


class TestBreakevenAutoVsAbstain:

    def test_default_matches_original(self):
        policy = bc_policy()
        assert policy.breakeven_auto_vs_abstain() == policy.breakeven_auto_vs_abstain(1.0)
        assert policy.breakeven_auto_vs_abstain() == pytest.approx(1 - 10 / 50)

    @pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_formula(self, q):
        policy = bc_policy()
        assert policy.breakeven_auto_vs_abstain(q) == pytest.approx(1 - q * 10 / 50)

    def test_q_zero_gives_threshold_one(self):
        """Abstain free → AUTO only ties at p = 1 (both cost 0).  1.0 is
        inside [0, 1] so it is returned, not clamped to None."""
        assert bc_policy().breakeven_auto_vs_abstain(0.0) == 1.0

    def test_threshold_is_where_action_flips(self):
        policy = DecisionPolicy(c_wrong=8.0, c_human=100.0, c_abstain=4.0)
        q = 0.5
        t = policy.breakeven_auto_vs_abstain(q)  # 1 - 0.5*4/8 = 0.75
        assert t == 0.75
        assert policy.decide(t, q).action == DecisionAction.AUTO_REPAIR      # tie → AUTO
        assert policy.decide(t - 0.01, q).action == DecisionAction.ABSTAIN
        assert policy.decide(t + 0.01, q).action == DecisionAction.AUTO_REPAIR

    def test_clamping_to_none_when_below_zero(self):
        # c_abstain > c_wrong: with q=1 the threshold is negative → None,
        # but with a small q it comes back into range.
        policy = DecisionPolicy(c_wrong=4.0, c_human=1.0, c_abstain=8.0)
        assert policy.breakeven_auto_vs_abstain() is None
        assert policy.breakeven_auto_vs_abstain(1.0) is None
        assert policy.breakeven_auto_vs_abstain(0.5) == pytest.approx(0.0)
        assert policy.breakeven_auto_vs_abstain(0.25) == pytest.approx(0.5)

    def test_phase7_thresholds_track_q(self):
        calc = RiskCalculator(bc_costs())
        th = calc.analytical_thresholds(0.25)
        assert th.p_accept_vs_leave == pytest.approx(1 - 0.25 * 10 / 50)
        assert th.p_accept_vs_review == pytest.approx(1 - 5 / 50)  # unaffected by q
        # 5 < 0.25*10 = 2.5 is False → leave now cheaper than review
        assert th.review_cheaper_than_leave is False
        assert "LEAVE_UNCHANGED" in th.recommended_policy
        assert "0.25" in th.recommended_policy

    def test_phase7_thresholds_clamped(self):
        calc = RiskCalculator(CostConfig(c_wrong_repair=4.0, c_human_review=1.0, c_leave_error=8.0))
        assert calc.analytical_thresholds(1.0).p_accept_vs_leave == 0.0
        assert calc.analytical_thresholds(0.25).p_accept_vs_leave == pytest.approx(0.5)


# ===========================================================================
# Group 8: Phase 7 / Phase 8 agreement
# ===========================================================================


class TestPhase7Phase8Agreement:

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_random_pairs_agree(self, dataset):
        costs = CostConfig.from_yaml(COSTS_YAML, dataset)
        calc = RiskCalculator(costs)
        policy = DecisionPolicy.from_cost_config(costs)
        rng = random.Random(42)
        for _ in range(500):
            p = rng.random()
            q = rng.random()
            ec = calc.compute_expected_costs(p, q)
            dr = policy.decide(p, q)
            assert abs(ec.accept - dr.expected_cost_auto) < 1e-9
            assert abs(ec.review - dr.expected_cost_human) < 1e-9
            assert abs(ec.leave - dr.expected_cost_abstain) < 1e-9
            assert ACTION_MAP[ec.best_action] == dr.action
            assert abs(ec.corruption_probability - dr.corruption_probability) < 1e-9

    def test_boundary_pairs_agree(self):
        costs = bc_costs()
        calc = RiskCalculator(costs)
        policy = DecisionPolicy.from_cost_config(costs)
        for p in (0.0, 1.0):
            for q in (0.0, 1.0):
                assert ACTION_MAP[calc.compute_expected_costs(p, q).best_action] \
                    == policy.decide(p, q).action

    def test_exact_ties_agree(self):
        costs = CostConfig(c_wrong_repair=8.0, c_human_review=2.0, c_leave_error=4.0)
        calc = RiskCalculator(costs)
        policy = DecisionPolicy.from_cost_config(costs)
        for p, q in [(0.75, 0.5), (0.5, 0.5), (0.75, 0.25)]:
            assert ACTION_MAP[calc.compute_expected_costs(p, q).best_action] \
                == policy.decide(p, q).action

    def test_sensitivity_sweep_agrees_with_policy(self):
        costs = bc_costs()
        calc = RiskCalculator(costs)
        policy = DecisionPolicy.from_cost_config(costs)
        for q in (0.0, 0.3, 0.6, 1.0):
            for pt in calc.sensitivity_over_p(21, q):
                dr = policy.decide(pt.p, q)
                assert ACTION_MAP[pt.best_action] == dr.action
                assert abs(pt.cost_leave - dr.expected_cost_abstain) < 1e-9


# ===========================================================================
# Group 9: Serialisation
# ===========================================================================


class TestSerialisation:

    def test_risk_record_to_dict_has_q(self):
        calc = RiskCalculator(bc_costs())
        rr = calc.compute_risk_record(_FakeUnc(), 0.6, 0.3)
        d = rr.to_dict()
        assert d["corruption_probability"] == 0.3
        assert d["expected_cost_leave"] == pytest.approx(3.0)
        assert d["lowest_cost_action"] == "LEAVE_UNCHANGED"

    def test_wrap_proposed_method_metadata_has_q(self):
        policy = bc_policy()
        uncs = [_FakeUnc(row_id=0), _FakeUnc(row_id=1)]
        ps = [0.6, 0.6]
        qs = [1.0, 0.2]
        drs = policy.decide_batch(ps, qs)
        recs = wrap_proposed_method(uncs, ps, drs, dataset_name="test")
        assert [r.metadata["corruption_probability"] for r in recs] == qs
        assert recs[1].action == DecisionAction.ABSTAIN
        assert recs[1].expected_cost_abstain == pytest.approx(2.0)

    def test_wrap_proposed_method_old_style_result_defaults_q_one(self):
        """A DecisionResult-like object without the new attribute (built by
        pre-Step-2 code) still wraps and reports q = 1."""

        class _OldDR:
            action = DecisionAction.HUMAN_REVIEW
            expected_cost_auto = 20.0
            expected_cost_human = 5.0
            expected_cost_abstain = 10.0
            c_wrong, c_human, c_abstain = 50.0, 5.0, 10.0

        recs = wrap_proposed_method([_FakeUnc()], [0.6], [_OldDR()])
        assert recs[0].metadata["corruption_probability"] == 1.0

    def test_wrap_keeps_existing_metadata_keys(self):
        drs = bc_policy().decide_batch([0.6])
        recs = wrap_proposed_method([_FakeUnc()], [0.6], drs)
        for k in ("repair_method", "error_type", "c_wrong", "c_human", "c_abstain"):
            assert k in recs[0].metadata


# ===========================================================================
# Group 10: Reason string
# ===========================================================================


class TestReason:

    def test_reason_shows_q_times_c_abstain(self):
        r = bc_policy().decide(0.6, 0.25)
        assert "E[Cost_abstain] = 0.2500 × 10.0 = 2.5000" in r.reason

    def test_reason_default_q_shows_one(self):
        r = bc_policy().decide(0.6)
        assert "E[Cost_abstain] = 1.0000 × 10.0 = 10.0000" in r.reason

    def test_reason_other_lines_unchanged(self):
        r = bc_policy().decide(0.6, 0.25)
        lines = r.reason.split("\n")
        assert len(lines) == 5
        assert lines[0] == "Calibrated P(correct) = 0.6000  →  P(wrong) = 0.4000"
        assert lines[1] == "E[Cost_auto]    = 0.4000 × 50.0 = 20.0000"
        assert lines[2] == "E[Cost_human]   = 5.0000"
        assert lines[4] == "Decision: ABSTAIN  (minimum expected cost = 2.5000)"

    def test_reason_action_matches(self):
        r = bc_policy().decide(0.6, 0.25)
        assert r.action == DecisionAction.ABSTAIN
        assert "ABSTAIN" in r.reason


# ===========================================================================
# Group 11: No leakage
# ===========================================================================


class TestNoLeakage:

    def test_new_parameter_names_do_not_reference_ground_truth(self):
        for fn in (
            DecisionPolicy.decide,
            DecisionPolicy.decide_batch,
            DecisionPolicy.breakeven_auto_vs_abstain,
            RiskCalculator.compute_expected_costs,
            RiskCalculator.compute_risk_record,
            RiskCalculator.compute_risk_batch,
            RiskCalculator.analytical_thresholds,
            RiskCalculator.sensitivity_over_p,
        ):
            for name in inspect.signature(fn).parameters:
                assert "clean" not in name
                assert "ground_truth" not in name
                assert "label" not in name

    def test_new_fields_do_not_reference_ground_truth(self):
        for cls in (DecisionResult, ExpectedCosts, RiskRecord):
            for f in dataclasses.fields(cls):
                assert "clean" not in f.name
                assert "ground_truth" not in f.name

    def test_q_is_a_plain_probability_not_a_label(self):
        """q is consumed as a number in [0, 1]; passing a 0/1 label works
        only because it is a valid probability, and the policy never
        inspects anything else about it."""
        policy = bc_policy()
        assert policy.decide(0.5, 0).expected_cost_abstain == 0.0
        assert policy.decide(0.5, 1).expected_cost_abstain == 10.0
        assert math.isclose(policy.decide(0.5, 0.5).expected_cost_abstain, 5.0)
