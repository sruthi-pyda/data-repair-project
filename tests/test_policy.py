"""
tests/test_policy.py
---------------------
Phase 8: Unit tests for the AUTO / HUMAN_REVIEW / ABSTAIN decision policy.

Test groups
-----------
 1. Spec example from the project spec (p=0.79, wrong=20, human=5, abstain=2)
 2. AUTO_REPAIR is selected when it is cheapest
 3. HUMAN_REVIEW is selected when it is cheapest
 4. ABSTAIN is selected when it is cheapest
 5. Monotonicity: increasing c_wrong never makes AUTO more attractive
 6. Monotonicity: decreasing c_human makes HUMAN more attractive
 7. Monotonicity: decreasing c_abstain makes ABSTAIN more attractive
 8. Tie-breaking is deterministic and follows documented priority
 9. Edge cases: p=0 and p=1
10. Invalid probabilities raise ValueError
11. Invalid cost parameters raise ValueError
12. DecisionResult fields are complete and correct
13. decide_batch processes lists correctly
14. DecisionPolicy.from_cost_config() bridges Phase 7 CostConfig
15. Breakeven threshold helpers
16. Policy does NOT access clean_df
17. Reason string is non-empty and contains key information
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any

import pytest
from pathlib import Path

from src.decision.policy import (
    DecisionAction,
    DecisionPolicy,
    DecisionResult,
    _TIE_PRIORITY,
)
from src.risk.cost_model import CostConfig


# ===========================================================================
# Helpers
# ===========================================================================


def make_policy(
    wrong: float = 10.0,
    human: float = 2.0,
    abstain: float = 5.0,
) -> DecisionPolicy:
    """Create a DecisionPolicy with given cost parameters."""
    return DecisionPolicy(c_wrong=wrong, c_human=human, c_abstain=abstain)


def expected_auto(p: float, c_wrong: float) -> float:
    """E[Cost_auto] = (1 - p) * C_wrong."""
    return (1.0 - p) * c_wrong


# ===========================================================================
# Group 1: Spec example
# ===========================================================================
# From the project spec:
#   p=0.79, C_wrong=20, C_human=5, C_abstain=2
#   auto = (1 - 0.79) * 20 = 0.21 * 20 = 4.2
#   human = 5
#   abstain = 2
#   → ABSTAIN


class TestSpecExample:
    """Test the exact example provided in the project specification."""

    def setup_method(self):
        self.policy = DecisionPolicy(c_wrong=20, c_human=5, c_abstain=2)
        self.result = self.policy.decide(0.79)

    def test_action_is_abstain(self):
        assert self.result.action == DecisionAction.ABSTAIN

    def test_expected_auto_cost(self):
        assert math.isclose(self.result.expected_cost_auto, 4.2, rel_tol=1e-9)

    def test_expected_human_cost(self):
        assert math.isclose(self.result.expected_cost_human, 5.0, rel_tol=1e-9)

    def test_expected_abstain_cost(self):
        assert math.isclose(self.result.expected_cost_abstain, 2.0, rel_tol=1e-9)

    def test_calibrated_probability(self):
        assert math.isclose(self.result.calibrated_probability, 0.79, rel_tol=1e-9)

    def test_probability_wrong(self):
        assert math.isclose(self.result.probability_wrong, 0.21, rel_tol=1e-9)

    def test_calibrated_plus_wrong_sum_to_one(self):
        s = self.result.calibrated_probability + self.result.probability_wrong
        assert math.isclose(s, 1.0, rel_tol=1e-9)

    def test_reason_is_non_empty(self):
        assert len(self.result.reason) > 0

    def test_reason_contains_action(self):
        assert "ABSTAIN" in self.result.reason


# ===========================================================================
# Group 2: AUTO_REPAIR selected
# ===========================================================================


class TestAutoRepairSelected:
    """AUTO_REPAIR is selected when (1-p)*C_wrong < C_human and < C_abstain."""

    def test_high_p_low_wrong_cost(self):
        # p=0.95, wrong=5, human=3, abstain=10
        # auto = 0.05*5 = 0.25 < 3 < 10
        policy = make_policy(wrong=5, human=3, abstain=10)
        result = policy.decide(0.95)
        assert result.action == DecisionAction.AUTO_REPAIR

    def test_p_one_always_auto(self):
        # p=1 → E[auto]=0, cheapest possible
        policy = make_policy(wrong=100, human=1, abstain=1)
        result = policy.decide(1.0)
        assert result.action == DecisionAction.AUTO_REPAIR

    def test_p_one_auto_cost_is_zero(self):
        policy = make_policy(wrong=100, human=1, abstain=1)
        result = policy.decide(1.0)
        assert result.expected_cost_auto == 0.0

    def test_auto_beats_human_when_p_just_above_threshold(self):
        # threshold for auto vs human: p* = 1 - human/wrong = 1 - 2/10 = 0.8
        policy = make_policy(wrong=10, human=2, abstain=100)
        result = policy.decide(0.85)  # above 0.8
        assert result.action == DecisionAction.AUTO_REPAIR

    def test_auto_with_very_cheap_wrong_cost(self):
        # Even low p can produce AUTO if c_wrong is tiny
        policy = make_policy(wrong=0.1, human=0.5, abstain=0.5)
        result = policy.decide(0.2)
        # auto = 0.8*0.1 = 0.08 < 0.5
        assert result.action == DecisionAction.AUTO_REPAIR

    def test_auto_cost_formula(self):
        policy = make_policy(wrong=20, human=5, abstain=8)
        result = policy.decide(0.9)
        expected = (1 - 0.9) * 20
        assert math.isclose(result.expected_cost_auto, expected, rel_tol=1e-9)

    def test_auto_when_both_alternatives_expensive(self):
        policy = make_policy(wrong=10, human=50, abstain=50)
        result = policy.decide(0.6)
        # auto = 0.4*10 = 4 < 50
        assert result.action == DecisionAction.AUTO_REPAIR


# ===========================================================================
# Group 3: HUMAN_REVIEW selected
# ===========================================================================


class TestHumanReviewSelected:
    """HUMAN_REVIEW is selected when it is cheaper than both AUTO and ABSTAIN."""

    def test_low_p_cheap_human_expensive_abstain(self):
        # p=0.3, wrong=20, human=3, abstain=10
        # auto=0.7*20=14, human=3, abstain=10 → HUMAN
        policy = make_policy(wrong=20, human=3, abstain=10)
        result = policy.decide(0.3)
        assert result.action == DecisionAction.HUMAN_REVIEW

    def test_human_when_abstain_more_expensive(self):
        policy = make_policy(wrong=15, human=2, abstain=8)
        result = policy.decide(0.5)
        # auto=0.5*15=7.5, human=2, abstain=8 → HUMAN
        assert result.action == DecisionAction.HUMAN_REVIEW

    def test_human_at_p_zero_low_human_cost(self):
        # p=0, auto = 1*wrong = 20, human=1, abstain=5 → HUMAN
        policy = make_policy(wrong=20, human=1, abstain=5)
        result = policy.decide(0.0)
        assert result.action == DecisionAction.HUMAN_REVIEW

    def test_human_review_cost_formula(self):
        policy = make_policy(wrong=10, human=3, abstain=8)
        result = policy.decide(0.5)
        assert math.isclose(result.expected_cost_human, 3.0, rel_tol=1e-9)


# ===========================================================================
# Group 4: ABSTAIN selected
# ===========================================================================


class TestAbstainSelected:
    """ABSTAIN is selected when it is cheaper than both AUTO and HUMAN_REVIEW."""

    def test_abstain_when_cheapest(self):
        # p=0.5, wrong=20, human=5, abstain=2 → auto=10, human=5, abstain=2
        policy = make_policy(wrong=20, human=5, abstain=2)
        result = policy.decide(0.5)
        assert result.action == DecisionAction.ABSTAIN

    def test_abstain_at_very_low_p(self):
        policy = make_policy(wrong=100, human=10, abstain=3)
        result = policy.decide(0.05)
        # auto=0.95*100=95, human=10, abstain=3 → ABSTAIN
        assert result.action == DecisionAction.ABSTAIN

    def test_abstain_cost_formula(self):
        policy = make_policy(wrong=20, human=5, abstain=2)
        result = policy.decide(0.5)
        assert math.isclose(result.expected_cost_abstain, 2.0, rel_tol=1e-9)

    def test_abstain_when_auto_and_human_both_expensive(self):
        policy = make_policy(wrong=50, human=30, abstain=1)
        result = policy.decide(0.4)
        # auto=0.6*50=30, human=30, abstain=1 → ABSTAIN
        assert result.action == DecisionAction.ABSTAIN


# ===========================================================================
# Group 5: Monotonicity — increasing c_wrong
# ===========================================================================


class TestMonotonicityWrongCost:
    """
    Increasing c_wrong must never make AUTO_REPAIR more attractive.

    As C_wrong increases:
    - E[Cost_auto] = (1-p) * C_wrong increases (for p < 1)
    - E[Cost_human] and E[Cost_abstain] are unchanged
    Therefore, AUTO can only become less attractive, never more.
    """

    def test_increasing_wrong_cost_does_not_improve_auto(self):
        p = 0.7
        human = 5.0
        abstain = 4.0
        # Increase c_wrong across a range; AUTO should never go from
        # HUMAN/ABSTAIN back to AUTO_REPAIR once it has been overtaken.
        prev_action = None
        saw_non_auto = False
        for wrong in [2, 5, 10, 20, 50, 100]:
            policy = DecisionPolicy(c_wrong=wrong, c_human=human, c_abstain=abstain)
            action = policy.decide(p).action
            if saw_non_auto:
                # Once non-AUTO, must stay non-AUTO
                assert action != DecisionAction.AUTO_REPAIR, (
                    f"AUTO_REPAIR returned after non-AUTO at c_wrong={wrong}"
                )
            if action != DecisionAction.AUTO_REPAIR:
                saw_non_auto = True
            prev_action = action

    def test_auto_cost_strictly_increases_with_wrong_cost(self):
        p = 0.6
        wrong1, wrong2 = 10.0, 20.0
        p1 = DecisionPolicy(c_wrong=wrong1, c_human=5, c_abstain=5)
        p2 = DecisionPolicy(c_wrong=wrong2, c_human=5, c_abstain=5)
        cost1 = p1.decide(p).expected_cost_auto
        cost2 = p2.decide(p).expected_cost_auto
        assert cost2 > cost1

    def test_high_wrong_cost_prevents_auto(self):
        # At p=0.8, auto = 0.2*wrong
        # For wrong=100: auto=20 > human=3 → not AUTO
        policy = DecisionPolicy(c_wrong=100, c_human=3, c_abstain=10)
        result = policy.decide(0.8)
        assert result.action != DecisionAction.AUTO_REPAIR


# ===========================================================================
# Group 6: Monotonicity — decreasing c_human
# ===========================================================================


class TestMonotonicityHumanCost:
    """
    Decreasing c_human makes HUMAN_REVIEW more attractive.

    As C_human decreases, E[Cost_human] decreases, making HUMAN_REVIEW
    more likely to be chosen.
    """

    def test_decreasing_human_cost_does_not_remove_human(self):
        p = 0.5
        wrong = 20.0
        abstain = 15.0
        # As human cost decreases, HUMAN should become attractive or stay
        prev_action = None
        saw_human = False
        for human in [15, 10, 5, 2, 0.5, 0.1]:
            policy = DecisionPolicy(c_wrong=wrong, c_human=human, c_abstain=abstain)
            action = policy.decide(p).action
            if action == DecisionAction.HUMAN_REVIEW:
                saw_human = True
        assert saw_human, "HUMAN_REVIEW never selected as c_human decreased"

    def test_cheap_human_beats_expensive_abstain(self):
        # abstain=10, human=0.5 → HUMAN wins at most p values
        policy = DecisionPolicy(c_wrong=20, c_human=0.5, c_abstain=10)
        result = policy.decide(0.5)
        assert result.action == DecisionAction.HUMAN_REVIEW

    def test_human_cost_formula_unchanged_by_p(self):
        policy = make_policy(wrong=20, human=3, abstain=5)
        for p in [0.0, 0.3, 0.7, 1.0]:
            result = policy.decide(p)
            assert math.isclose(result.expected_cost_human, 3.0)

    def test_reducing_human_cost_can_switch_action_from_abstain(self):
        p = 0.5
        wrong = 20.0
        abstain = 2.0
        # At human=5: auto=10, human=5, abstain=2 → ABSTAIN
        policy_before = DecisionPolicy(c_wrong=wrong, c_human=5, c_abstain=abstain)
        assert policy_before.decide(p).action == DecisionAction.ABSTAIN
        # At human=1: auto=10, human=1, abstain=2 → HUMAN
        policy_after = DecisionPolicy(c_wrong=wrong, c_human=1, c_abstain=abstain)
        assert policy_after.decide(p).action == DecisionAction.HUMAN_REVIEW


# ===========================================================================
# Group 7: Monotonicity — decreasing c_abstain
# ===========================================================================


class TestMonotonicityAbstainCost:
    """
    Decreasing c_abstain makes ABSTAIN more attractive.
    """

    def test_cheap_abstain_beats_expensive_human_and_auto(self):
        policy = DecisionPolicy(c_wrong=20, c_human=10, c_abstain=0.5)
        result = policy.decide(0.5)
        assert result.action == DecisionAction.ABSTAIN

    def test_reducing_abstain_cost_can_switch_from_human(self):
        p = 0.5
        wrong = 20.0
        # human=2, abstain=5 → auto=10, human=2, abstain=5 → HUMAN
        policy_before = DecisionPolicy(c_wrong=wrong, c_human=2, c_abstain=5)
        assert policy_before.decide(p).action == DecisionAction.HUMAN_REVIEW
        # human=2, abstain=1 → auto=10, human=2, abstain=1 → ABSTAIN
        policy_after = DecisionPolicy(c_wrong=wrong, c_human=2, c_abstain=1)
        assert policy_after.decide(p).action == DecisionAction.ABSTAIN

    def test_abstain_cost_formula_unchanged_by_p(self):
        policy = make_policy(wrong=20, human=5, abstain=3)
        for p in [0.0, 0.5, 0.99, 1.0]:
            result = policy.decide(p)
            assert math.isclose(result.expected_cost_abstain, 3.0)


# ===========================================================================
# Group 8: Tie-breaking
# ===========================================================================


class TestTieBreaking:
    """
    Ties must be handled deterministically.

    Documented priority: AUTO_REPAIR > HUMAN_REVIEW > ABSTAIN.
    """

    def test_auto_beats_human_on_tie(self):
        # Tie when (1-p)*wrong = human
        # p=0.8, wrong=10: auto = 0.2*10 = 2 = human
        policy = make_policy(wrong=10, human=2, abstain=100)
        result = policy.decide(0.8)
        assert result.action == DecisionAction.AUTO_REPAIR

    def test_auto_beats_abstain_on_tie(self):
        # (1-p)*wrong = abstain
        # p=0.5, wrong=10: auto=5, abstain=5, human=100
        policy = make_policy(wrong=10, human=100, abstain=5)
        result = policy.decide(0.5)
        assert result.action == DecisionAction.AUTO_REPAIR

    def test_human_beats_abstain_on_tie(self):
        # human = abstain = 3, auto is expensive
        # p=0.0, wrong=5: auto=5, human=3, abstain=3 → HUMAN over ABSTAIN
        policy = make_policy(wrong=5, human=3, abstain=3)
        result = policy.decide(0.0)
        assert result.action == DecisionAction.HUMAN_REVIEW

    def test_auto_wins_three_way_tie(self):
        # All three equal: auto = human = abstain
        # (1-p)*wrong = human = abstain
        # p=0, wrong=3, human=3, abstain=3: auto=1*3=3=human=abstain → AUTO
        policy = DecisionPolicy(c_wrong=3, c_human=3, c_abstain=3)
        result = policy.decide(0.0)
        assert result.action == DecisionAction.AUTO_REPAIR

    def test_tie_priority_dict_complete(self):
        """All three actions must be present in the priority dict."""
        assert DecisionAction.AUTO_REPAIR in _TIE_PRIORITY
        assert DecisionAction.HUMAN_REVIEW in _TIE_PRIORITY
        assert DecisionAction.ABSTAIN in _TIE_PRIORITY

    def test_auto_has_lowest_priority_value(self):
        """AUTO_REPAIR must have lower priority value than HUMAN and ABSTAIN."""
        assert _TIE_PRIORITY[DecisionAction.AUTO_REPAIR] < _TIE_PRIORITY[DecisionAction.HUMAN_REVIEW]
        assert _TIE_PRIORITY[DecisionAction.AUTO_REPAIR] < _TIE_PRIORITY[DecisionAction.ABSTAIN]

    def test_human_lower_priority_value_than_abstain(self):
        assert _TIE_PRIORITY[DecisionAction.HUMAN_REVIEW] < _TIE_PRIORITY[DecisionAction.ABSTAIN]

    def test_tie_result_is_deterministic(self):
        """Same inputs always produce same output."""
        policy = make_policy(wrong=10, human=2, abstain=100)
        results = [policy.decide(0.8).action for _ in range(20)]
        assert len(set(results)) == 1


# ===========================================================================
# Group 9: Edge cases p=0 and p=1
# ===========================================================================


class TestEdgeCases:
    """Edge cases at p=0 and p=1."""

    def test_p_zero_auto_cost_equals_c_wrong(self):
        # p=0 → P(wrong)=1 → E[auto] = 1 * C_wrong
        policy = make_policy(wrong=15, human=10, abstain=10)
        result = policy.decide(0.0)
        assert math.isclose(result.expected_cost_auto, 15.0)

    def test_p_zero_probability_wrong_is_one(self):
        policy = make_policy()
        result = policy.decide(0.0)
        assert math.isclose(result.probability_wrong, 1.0)

    def test_p_one_auto_cost_is_zero(self):
        policy = make_policy(wrong=50, human=10, abstain=5)
        result = policy.decide(1.0)
        assert result.expected_cost_auto == 0.0

    def test_p_one_always_selects_auto(self):
        # p=1 → E[auto]=0, which is always the minimum
        for wrong in [1, 5, 100]:
            for human in [0.0, 1.0, 50.0]:
                for abstain in [0.0, 1.0, 50.0]:
                    policy = DecisionPolicy(c_wrong=wrong, c_human=human, c_abstain=abstain)
                    result = policy.decide(1.0)
                    assert result.action == DecisionAction.AUTO_REPAIR, (
                        f"Expected AUTO_REPAIR at p=1 but got {result.action} "
                        f"(wrong={wrong}, human={human}, abstain={abstain})"
                    )

    def test_p_zero_action_depends_on_costs(self):
        # At p=0, auto = c_wrong; outcome depends on relative costs
        # HUMAN cheaper: wrong=10, human=3, abstain=8
        p1 = make_policy(wrong=10, human=3, abstain=8)
        assert p1.decide(0.0).action == DecisionAction.HUMAN_REVIEW

        # ABSTAIN cheaper: wrong=10, human=8, abstain=3
        p2 = make_policy(wrong=10, human=8, abstain=3)
        assert p2.decide(0.0).action == DecisionAction.ABSTAIN

        # AUTO still cheapest (won't happen at p=0 if auto = c_wrong > 0)
        # auto=wrong>0; only wins if human and abstain are also that big or bigger
        # auto=5=human=5=abstain=5 → tie, AUTO wins by priority
        p3 = DecisionPolicy(c_wrong=5, c_human=5, c_abstain=5)
        assert p3.decide(0.0).action == DecisionAction.AUTO_REPAIR


# ===========================================================================
# Group 10: Invalid probabilities
# ===========================================================================


class TestInvalidProbabilities:
    """Invalid probability values must raise ValueError."""

    def test_p_negative_raises(self):
        policy = make_policy()
        with pytest.raises(ValueError, match="calibrated_probability"):
            policy.decide(-0.01)

    def test_p_greater_than_one_raises(self):
        policy = make_policy()
        with pytest.raises(ValueError, match="calibrated_probability"):
            policy.decide(1.01)

    def test_p_nan_raises(self):
        policy = make_policy()
        with pytest.raises((ValueError, TypeError)):
            policy.decide(float("nan"))

    def test_p_minus_one_raises(self):
        policy = make_policy()
        with pytest.raises(ValueError):
            policy.decide(-1.0)

    def test_p_two_raises(self):
        policy = make_policy()
        with pytest.raises(ValueError):
            policy.decide(2.0)

    def test_p_infinity_raises(self):
        policy = make_policy()
        with pytest.raises((ValueError, OverflowError)):
            policy.decide(float("inf"))


# ===========================================================================
# Group 11: Invalid cost parameters
# ===========================================================================


class TestInvalidCosts:
    """Invalid cost parameters must raise ValueError at construction."""

    def test_c_wrong_zero_raises(self):
        with pytest.raises(ValueError, match="c_wrong"):
            DecisionPolicy(c_wrong=0.0, c_human=2.0, c_abstain=5.0)

    def test_c_wrong_negative_raises(self):
        with pytest.raises(ValueError, match="c_wrong"):
            DecisionPolicy(c_wrong=-1.0, c_human=2.0, c_abstain=5.0)

    def test_c_human_negative_raises(self):
        with pytest.raises(ValueError, match="c_human"):
            DecisionPolicy(c_wrong=10.0, c_human=-0.1, c_abstain=5.0)

    def test_c_abstain_negative_raises(self):
        with pytest.raises(ValueError, match="c_abstain"):
            DecisionPolicy(c_wrong=10.0, c_human=2.0, c_abstain=-1.0)

    def test_c_human_zero_is_valid(self):
        """c_human=0 is valid (free human review)."""
        policy = DecisionPolicy(c_wrong=10.0, c_human=0.0, c_abstain=5.0)
        result = policy.decide(0.5)
        assert result.action == DecisionAction.HUMAN_REVIEW

    def test_c_abstain_zero_is_valid(self):
        """c_abstain=0 is valid (no cost to abstain)."""
        policy = DecisionPolicy(c_wrong=10.0, c_human=5.0, c_abstain=0.0)
        result = policy.decide(0.5)
        assert result.action == DecisionAction.ABSTAIN


# ===========================================================================
# Group 12: DecisionResult fields
# ===========================================================================


class TestDecisionResultFields:
    """DecisionResult must contain all required fields with correct values."""

    def setup_method(self):
        self.policy = DecisionPolicy(c_wrong=20, c_human=5, c_abstain=2)
        self.result = self.policy.decide(0.79)

    def test_has_calibrated_probability(self):
        assert hasattr(self.result, "calibrated_probability")

    def test_has_probability_wrong(self):
        assert hasattr(self.result, "probability_wrong")

    def test_has_expected_cost_auto(self):
        assert hasattr(self.result, "expected_cost_auto")

    def test_has_expected_cost_human(self):
        assert hasattr(self.result, "expected_cost_human")

    def test_has_expected_cost_abstain(self):
        assert hasattr(self.result, "expected_cost_abstain")

    def test_has_action(self):
        assert hasattr(self.result, "action")

    def test_has_reason(self):
        assert hasattr(self.result, "reason")

    def test_has_c_wrong(self):
        assert hasattr(self.result, "c_wrong")
        assert self.result.c_wrong == 20.0

    def test_has_c_human(self):
        assert hasattr(self.result, "c_human")
        assert self.result.c_human == 5.0

    def test_has_c_abstain(self):
        assert hasattr(self.result, "c_abstain")
        assert self.result.c_abstain == 2.0

    def test_p_plus_p_wrong_equals_one(self):
        r = self.result
        assert math.isclose(
            r.calibrated_probability + r.probability_wrong, 1.0, rel_tol=1e-9
        )

    def test_action_is_decision_action(self):
        assert isinstance(self.result.action, DecisionAction)

    def test_result_is_frozen(self):
        """DecisionResult must be immutable (frozen dataclass)."""
        with pytest.raises((AttributeError, TypeError)):
            self.result.action = DecisionAction.AUTO_REPAIR  # type: ignore


# ===========================================================================
# Group 13: decide_batch
# ===========================================================================


class TestDecideBatch:
    """decide_batch processes lists correctly."""

    def test_batch_empty_list(self):
        policy = make_policy()
        results = policy.decide_batch([])
        assert results == []

    def test_batch_single_element(self):
        policy = make_policy()
        results = policy.decide_batch([0.9])
        assert len(results) == 1
        assert isinstance(results[0], DecisionResult)

    def test_batch_matches_individual(self):
        policy = make_policy(wrong=20, human=5, abstain=2)
        probs = [0.1, 0.5, 0.79, 0.9, 1.0]
        batch = policy.decide_batch(probs)
        for p, res in zip(probs, batch):
            individual = policy.decide(p)
            assert res.action == individual.action
            assert math.isclose(res.expected_cost_auto, individual.expected_cost_auto)

    def test_batch_length_matches_input(self):
        policy = make_policy()
        probs = [0.1, 0.5, 0.8, 0.95]
        results = policy.decide_batch(probs)
        assert len(results) == len(probs)

    def test_batch_invalid_prob_raises(self):
        policy = make_policy()
        with pytest.raises(ValueError):
            policy.decide_batch([0.5, -0.1, 0.8])


# ===========================================================================
# Group 14: from_cost_config (Phase 7 bridge)
# ===========================================================================


class TestFromCostConfig:
    """DecisionPolicy.from_cost_config() must correctly map CostConfig fields."""

    def make_config(self, wrong=15.0, review=3.0, leave=7.0) -> CostConfig:
        return CostConfig(
            c_wrong_repair=wrong,
            c_human_review=review,
            c_leave_error=leave,
            dataset_name="test",
        )

    def test_c_wrong_mapped(self):
        config = self.make_config(wrong=15.0)
        policy = DecisionPolicy.from_cost_config(config)
        assert policy.c_wrong == 15.0

    def test_c_human_mapped(self):
        config = self.make_config(review=3.0)
        policy = DecisionPolicy.from_cost_config(config)
        assert policy.c_human == 3.0

    def test_c_abstain_mapped_from_leave(self):
        config = self.make_config(leave=7.0)
        policy = DecisionPolicy.from_cost_config(config)
        assert policy.c_abstain == 7.0

    def test_from_config_produces_same_decision_as_manual(self):
        config = self.make_config(wrong=20.0, review=5.0, leave=2.0)
        policy_from_config = DecisionPolicy.from_cost_config(config)
        policy_manual = DecisionPolicy(c_wrong=20.0, c_human=5.0, c_abstain=2.0)
        for p in [0.1, 0.5, 0.79, 0.9]:
            r1 = policy_from_config.decide(p)
            r2 = policy_manual.decide(p)
            assert r1.action == r2.action
            assert math.isclose(r1.expected_cost_auto, r2.expected_cost_auto)

    def test_from_config_with_default_config(self):
        config = CostConfig.default(dataset_name="adult")
        policy = DecisionPolicy.from_cost_config(config)
        result = policy.decide(0.7)
        assert result.action in list(DecisionAction)


# ===========================================================================
# Group 14b: Shipped cost configs (configs/costs.yaml)
# ===========================================================================


COSTS_YAML = Path(__file__).resolve().parent.parent / "configs" / "costs.yaml"
SHIPPED_DATASETS = ["adult", "breast_cancer", "smartfactory"]


def _sweep_actions(policy: DecisionPolicy, steps: int = 1000) -> set:
    """Every action the policy selects across p in [0, 1]."""
    return {policy.decide(i / steps).action for i in range(steps + 1)}


class TestShippedCostConfigs:
    """
    Guard against a degenerate cost table. Because E[HUMAN_REVIEW] and
    E[ABSTAIN] are both constant in p, a config where every dataset has
    c_leave_error > c_human_review can never ABSTAIN, and Phase 10
    comparison tables would silently print 0% abstention on every row.
    """

    @pytest.mark.parametrize("dataset", SHIPPED_DATASETS)
    def test_every_shipped_config_loads_into_policy(self, dataset):
        config = CostConfig.from_yaml(COSTS_YAML, dataset)
        policy = DecisionPolicy.from_cost_config(config)
        assert policy.decide(0.5).action in list(DecisionAction)

    @pytest.mark.parametrize("dataset", SHIPPED_DATASETS)
    def test_auto_repair_reachable_for_every_dataset(self, dataset):
        config = CostConfig.from_yaml(COSTS_YAML, dataset)
        policy = DecisionPolicy.from_cost_config(config)
        assert DecisionAction.AUTO_REPAIR in _sweep_actions(policy)

    def test_abstain_reachable_in_at_least_one_shipped_config(self):
        reachable = {
            ds: DecisionAction.ABSTAIN
            in _sweep_actions(DecisionPolicy.from_cost_config(CostConfig.from_yaml(COSTS_YAML, ds)))
            for ds in SHIPPED_DATASETS
        }
        assert any(reachable.values()), (
            f"ABSTAIN unreachable under every shipped config: {reachable}. "
            "At least one dataset must set c_leave_error < c_human_review."
        )

    def test_adult_abstains_on_low_confidence(self):
        """Adult detections are false positives with p near 0; the rational
        action is to leave them alone, not to route them all to a human."""
        config = CostConfig.from_yaml(COSTS_YAML, "adult")
        policy = DecisionPolicy.from_cost_config(config)
        for p in [0.0, 0.05, 0.2, 0.5, 0.85]:
            assert policy.decide(p).action == DecisionAction.ABSTAIN, p

    def test_adult_still_auto_repairs_on_high_confidence(self):
        config = CostConfig.from_yaml(COSTS_YAML, "adult")
        policy = DecisionPolicy.from_cost_config(config)
        assert policy.decide(0.99).action == DecisionAction.AUTO_REPAIR

    @pytest.mark.parametrize("dataset", ["breast_cancer", "smartfactory"])
    def test_genuine_corruption_datasets_route_to_human(self, dataset):
        """Where detections are real errors, low confidence should go to a
        reviewer rather than be abstained on."""
        config = CostConfig.from_yaml(COSTS_YAML, dataset)
        policy = DecisionPolicy.from_cost_config(config)
        assert policy.decide(0.3).action == DecisionAction.HUMAN_REVIEW


# ===========================================================================
# Group 15: Breakeven threshold helpers
# ===========================================================================


class TestBreakevenThresholds:
    """Analytical breakeven threshold methods."""

    def test_breakeven_auto_vs_human_formula(self):
        # p* = 1 - c_human / c_wrong = 1 - 2/10 = 0.8
        policy = make_policy(wrong=10, human=2, abstain=100)
        t = policy.breakeven_auto_vs_human()
        assert t is not None
        assert math.isclose(t, 0.8, rel_tol=1e-9)

    def test_breakeven_auto_vs_abstain_formula(self):
        # p** = 1 - c_abstain / c_wrong = 1 - 5/10 = 0.5
        policy = make_policy(wrong=10, human=100, abstain=5)
        t = policy.breakeven_auto_vs_abstain()
        assert t is not None
        assert math.isclose(t, 0.5, rel_tol=1e-9)

    def test_at_threshold_auto_and_human_tie(self):
        # p* = 1 - 2/10 = 0.8
        policy = make_policy(wrong=10, human=2, abstain=100)
        t = policy.breakeven_auto_vs_human()
        result = policy.decide(t)
        # At threshold, tie → AUTO wins by priority
        assert result.action == DecisionAction.AUTO_REPAIR
        assert math.isclose(result.expected_cost_auto, result.expected_cost_human, rel_tol=1e-9)

    def test_below_threshold_human_wins(self):
        policy = make_policy(wrong=10, human=2, abstain=100)
        t = policy.breakeven_auto_vs_human()
        result = policy.decide(max(0.0, t - 0.1))
        assert result.action == DecisionAction.HUMAN_REVIEW

    def test_threshold_none_when_outside_range(self):
        # c_human > c_wrong → threshold would be < 0 → None
        policy = make_policy(wrong=2, human=10, abstain=100)
        t = policy.breakeven_auto_vs_human()
        # 1 - 10/2 = -4 → outside [0,1] → None
        assert t is None


# ===========================================================================
# Group 16: Policy does not access clean_df
# ===========================================================================


class TestNoCleanDfAccess:
    """The policy must not accept or use clean_df."""

    def test_decide_signature_has_no_clean_df_param(self):
        sig = inspect.signature(DecisionPolicy.decide)
        assert "clean_df" not in sig.parameters

    def test_decide_batch_signature_has_no_clean_df_param(self):
        sig = inspect.signature(DecisionPolicy.decide_batch)
        assert "clean_df" not in sig.parameters

    def test_policy_init_has_no_clean_df_param(self):
        sig = inspect.signature(DecisionPolicy.__init__)
        assert "clean_df" not in sig.parameters

    def test_decision_result_has_no_clean_df_field(self):
        """DecisionResult dataclass must not have a clean_df field."""
        fields = {f.name for f in DecisionResult.__dataclass_fields__.values()}
        assert "clean_df" not in fields
        assert "clean_value" not in fields
        assert "ground_truth" not in fields


# ===========================================================================
# Group 17: Reason string
# ===========================================================================


class TestReasonString:
    """The reason string must be informative."""

    def test_reason_is_non_empty(self):
        policy = make_policy()
        result = policy.decide(0.7)
        assert result.reason.strip() != ""

    def test_reason_contains_probability(self):
        policy = make_policy()
        result = policy.decide(0.7)
        # Reason should mention the probability
        assert "0.7" in result.reason or "P(correct)" in result.reason

    def test_reason_contains_action(self):
        policy = make_policy()
        result = policy.decide(0.7)
        assert result.action.value in result.reason

    def test_reason_contains_cost_values(self):
        policy = DecisionPolicy(c_wrong=20, c_human=5, c_abstain=2)
        result = policy.decide(0.79)
        # Should mention numerical values
        assert "5" in result.reason or "2" in result.reason

    def test_reason_is_string(self):
        policy = make_policy()
        result = policy.decide(0.5)
        assert isinstance(result.reason, str)


# ===========================================================================
# Group 18: DecisionAction enum
# ===========================================================================


class TestDecisionActionEnum:
    """DecisionAction must define the three required action names."""

    def test_auto_repair_exists(self):
        assert DecisionAction.AUTO_REPAIR is not None

    def test_human_review_exists(self):
        assert DecisionAction.HUMAN_REVIEW is not None

    def test_abstain_exists(self):
        assert DecisionAction.ABSTAIN is not None

    def test_auto_repair_value(self):
        assert DecisionAction.AUTO_REPAIR.value == "AUTO_REPAIR"

    def test_human_review_value(self):
        assert DecisionAction.HUMAN_REVIEW.value == "HUMAN_REVIEW"

    def test_abstain_value(self):
        assert DecisionAction.ABSTAIN.value == "ABSTAIN"

    def test_exactly_three_actions(self):
        actions = list(DecisionAction)
        assert len(actions) == 3
