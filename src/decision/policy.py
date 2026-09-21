"""
src/decision/policy.py
-----------------------
Phase 8: AUTO / HUMAN_REVIEW / ABSTAIN decision policy.

DECISION RULE
=============
For a proposed repair with calibrated probability of correctness p:

    P(wrong) = 1 - p

Expected costs:
    E[Cost_auto]    = (1 - p) * C_wrong    — cost if wrong repair is applied
    E[Cost_human]   = C_human              — fixed cost of routing to human
    E[Cost_abstain] = q * C_abstain        — cost of leaving the cell alone

where q = corruption_probability = P(the flagged cell is genuinely
corrupted).  Abstaining on a false alarm costs nothing; abstaining on a
real error costs C_abstain.  q defaults to 1.0, which reproduces the
original rule (every flagged cell is assumed to be a real error and
E[Cost_abstain] = C_abstain is flat in p).

The policy selects the minimum expected cost action:

    decision = argmin(E[Cost_auto], E[Cost_human], E[Cost_abstain])

Worked example (breast_cancer costs: C_wrong=50, C_human=5, C_abstain=10)

    p = 0.95, q = 1.0   →  auto 2.50, human 5.00, abstain 10.00  →  AUTO_REPAIR
    p = 0.60, q = 1.0   →  auto 20.0, human 5.00, abstain 10.00  →  HUMAN_REVIEW
    p = 0.60, q = 0.2   →  auto 20.0, human 5.00, abstain  2.00  →  ABSTAIN

With q = 1 both HUMAN and ABSTAIN are flat lines, so one of them is
unreachable on every cost config.  A calibrated q < C_human / C_abstain
is what makes ABSTAIN a live option.

SUPPORTED ACTIONS
=================
    AUTO_REPAIR   — automatically apply the top repair candidate
    HUMAN_REVIEW  — route to a human reviewer for manual correction
    ABSTAIN       — leave the cell unchanged (do not repair)

TIE-BREAKING
============
When two or more actions share the minimum expected cost, the policy
applies a deterministic priority: AUTO_REPAIR > HUMAN_REVIEW > ABSTAIN.

Rationale: if AUTO_REPAIR is equally cheap as alternatives, prefer
automation; if HUMAN_REVIEW ties with ABSTAIN, prefer routing to a
human (who can still correct the error, unlike abstaining).

This priority is documented here and applied consistently.

COST PARAMETERS
===============
All costs are experimental assumptions in abstract "penalty units".
They are NOT derived from the REIN benchmark or real monetary prices.
Configure them in configs/costs.yaml:

    c_wrong_repair  — C_wrong  (Phase 7 name → policy C_wrong)
    c_human_review  — C_human  (Phase 7 name → policy C_human)
    c_leave_error   — C_abstain (Phase 7 "leave unchanged" = abstaining)

RELATIONSHIP TO PHASE 7 RISK MODEL
====================================
Phase 7 (cost_model.py) provides CostConfig and RiskCalculator using
action names ACCEPT_REPAIR / HUMAN_REVIEW / LEAVE_UNCHANGED.

This module uses the spec-required names:
    ACCEPT_REPAIR   → AUTO_REPAIR
    HUMAN_REVIEW    → HUMAN_REVIEW  (same)
    LEAVE_UNCHANGED → ABSTAIN

The DecisionPolicy.from_cost_config() factory bridges the two layers.

DATA LEAKAGE RULE
==================
This module does NOT access clean_df at any point.
The calibrated_probability input must come from the calibration stage.
The corruption_probability input must likewise come from a calibrator
fitted on a training split (Phase 6-style); this module never derives it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from src.risk.cost_model import CostConfig


# ---------------------------------------------------------------------------
# Action enum
# ---------------------------------------------------------------------------


class DecisionAction(str, Enum):
    """
    The three possible actions for a detected and repaired cell.

    AUTO_REPAIR   — automatically apply the top repair candidate
    HUMAN_REVIEW  — route to a human reviewer for manual correction
    ABSTAIN       — leave the cell unchanged (do not repair)
    """

    AUTO_REPAIR = "AUTO_REPAIR"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    ABSTAIN = "ABSTAIN"


# ---------------------------------------------------------------------------
# DecisionResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionResult:
    """
    Full explanation of a single policy decision.

    All cost values are in abstract "penalty units" — experimental
    assumptions, NOT real monetary values or REIN-provided costs.

    Attributes
    ----------
    calibrated_probability : float
        p — calibrated probability the repair is correct, in [0, 1].
    probability_wrong : float
        1 - p — probability the repair is incorrect.
    expected_cost_auto : float
        E[Cost_auto] = (1 - p) * C_wrong
    expected_cost_human : float
        E[Cost_human] = C_human  (fixed, independent of p)
    expected_cost_abstain : float
        E[Cost_abstain] = q × C_abstain  (independent of p; q defaults to 1,
        in which case this is simply C_abstain)
    action : DecisionAction
        The selected minimum-cost action.
    reason : str
        Human-readable explanation of the decision, including all costs.
    c_wrong : float
        Wrong-repair cost parameter used.
    c_human : float
        Human-review cost parameter used.
    c_abstain : float
        Abstention cost parameter used.
    corruption_probability : float
        q — calibrated probability the flagged cell is genuinely corrupted,
        in [0, 1].  Defaults to 1.0 (every flag assumed real), which
        reproduces the pre-Step-2 behaviour exactly.
    """

    calibrated_probability: float
    probability_wrong: float
    expected_cost_auto: float
    expected_cost_human: float
    expected_cost_abstain: float
    action: DecisionAction
    reason: str
    c_wrong: float
    c_human: float
    c_abstain: float
    corruption_probability: float = 1.0


# ---------------------------------------------------------------------------
# Tie-breaking priority
# ---------------------------------------------------------------------------

# When two or more actions share the minimum expected cost, the action
# with the lowest priority value wins.  Order: AUTO_REPAIR first, then
# HUMAN_REVIEW, then ABSTAIN.
_TIE_PRIORITY: dict = {
    DecisionAction.AUTO_REPAIR: 0,
    DecisionAction.HUMAN_REVIEW: 1,
    DecisionAction.ABSTAIN: 2,
}


# ---------------------------------------------------------------------------
# DecisionPolicy
# ---------------------------------------------------------------------------


class DecisionPolicy:
    """
    Calibrated expected-cost decision policy.

    Given a calibrated probability p (that a repair is correct) and
    three configurable cost parameters, selects the action with the
    lowest expected cost:

        E[Cost_auto]    = (1 - p) * C_wrong
        E[Cost_human]   = C_human              (constant)
        E[Cost_abstain] = q * C_abstain        (constant in p; q defaults to 1)

        decision = argmin(E[Cost_auto], E[Cost_human], E[Cost_abstain])

    q = corruption_probability is supplied per decision (see decide()),
    not at construction time, because it is a property of the cell, not
    of the cost configuration.

    Parameters
    ----------
    c_wrong : float
        Cost of applying an incorrect automatic repair.
        Must be strictly positive (> 0).
    c_human : float
        Fixed cost of routing to human review.
        Must be non-negative (>= 0).
    c_abstain : float
        Cost of abstaining on a genuinely corrupted cell.
        Must be non-negative (>= 0).

    Tie-breaking
    ------------
    When multiple actions share the minimum expected cost, priority is:
        AUTO_REPAIR > HUMAN_REVIEW > ABSTAIN

    Notes
    -----
    This policy does NOT access clean_df.  Calibrated probabilities
    must come from the calibration stage (Phase 6).
    """

    def __init__(
        self,
        c_wrong: float,
        c_human: float,
        c_abstain: float,
    ) -> None:
        if c_wrong <= 0:
            raise ValueError(
                f"c_wrong must be strictly positive, got {c_wrong}"
            )
        if c_human < 0:
            raise ValueError(
                f"c_human must be non-negative, got {c_human}"
            )
        if c_abstain < 0:
            raise ValueError(
                f"c_abstain must be non-negative, got {c_abstain}"
            )
        self.c_wrong = float(c_wrong)
        self.c_human = float(c_human)
        self.c_abstain = float(c_abstain)

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_cost_config(cls, costs: CostConfig) -> "DecisionPolicy":
        """
        Create a DecisionPolicy from a Phase 7 CostConfig.

        Mapping
        -------
        CostConfig.c_wrong_repair  → DecisionPolicy.c_wrong
        CostConfig.c_human_review  → DecisionPolicy.c_human
        CostConfig.c_leave_error   → DecisionPolicy.c_abstain

        Parameters
        ----------
        costs : CostConfig
            Cost configuration loaded from configs/costs.yaml.
        """
        return cls(
            c_wrong=costs.c_wrong_repair,
            c_human=costs.c_human_review,
            c_abstain=costs.c_leave_error,
        )

    # ── Core decision ────────────────────────────────────────────────────────

    def decide(
        self,
        calibrated_probability: float,
        corruption_probability: float = 1.0,
    ) -> DecisionResult:
        """
        Compute the minimum-cost action for a given calibrated probability.

        Formula
        -------
        Let p = calibrated_probability, q = corruption_probability.

            E[Cost_auto]    = (1 - p) * C_wrong
            E[Cost_human]   = C_human
            E[Cost_abstain] = q * C_abstain

            action = argmin(E[Cost_auto], E[Cost_human], E[Cost_abstain])

        Parameters
        ----------
        calibrated_probability : float
            p — calibrated probability the repair is correct.
            Must be in [0, 1].
        corruption_probability : float, optional
            q — calibrated probability the flagged cell is genuinely
            corrupted.  Must be in [0, 1].  Defaults to 1.0, which makes
            E[Cost_abstain] = C_abstain and reproduces the original rule
            exactly.

        Returns
        -------
        DecisionResult

        Raises
        ------
        ValueError
            If calibrated_probability or corruption_probability is outside
            [0, 1] (NaN and ±inf are rejected by the same check).
        """
        p = float(calibrated_probability)
        if not (0.0 <= p <= 1.0):
            raise ValueError(
                f"calibrated_probability must be in [0, 1], got {p}"
            )
        q = float(corruption_probability)
        if not (0.0 <= q <= 1.0):
            raise ValueError(
                f"corruption_probability must be in [0, 1], got {q}"
            )

        p_wrong = 1.0 - p
        e_auto = p_wrong * self.c_wrong
        e_human = self.c_human
        e_abstain = q * self.c_abstain

        # Select minimum-cost action; on tie, apply documented priority
        candidates: dict = {
            DecisionAction.AUTO_REPAIR: e_auto,
            DecisionAction.HUMAN_REVIEW: e_human,
            DecisionAction.ABSTAIN: e_abstain,
        }
        min_cost = min(candidates.values())
        tied = [a for a, cost in candidates.items() if cost == min_cost]
        # deterministic: sort by priority, take first
        action = min(tied, key=lambda a: _TIE_PRIORITY[a])

        reason = self._build_reason(
            p, p_wrong, e_auto, e_human, e_abstain, action, min_cost, q
        )

        return DecisionResult(
            calibrated_probability=round(p, 10),
            probability_wrong=round(p_wrong, 10),
            expected_cost_auto=round(e_auto, 10),
            expected_cost_human=round(e_human, 10),
            expected_cost_abstain=round(e_abstain, 10),
            action=action,
            reason=reason,
            c_wrong=self.c_wrong,
            c_human=self.c_human,
            c_abstain=self.c_abstain,
            corruption_probability=round(q, 10),
        )

    def _build_reason(
        self,
        p: float,
        p_wrong: float,
        e_auto: float,
        e_human: float,
        e_abstain: float,
        action: DecisionAction,
        min_cost: float,
        q: float = 1.0,
    ) -> str:
        """Build a human-readable explanation of the decision."""
        lines = [
            f"Calibrated P(correct) = {p:.4f}  →  P(wrong) = {p_wrong:.4f}",
            f"E[Cost_auto]    = {p_wrong:.4f} × {self.c_wrong} = {e_auto:.4f}",
            f"E[Cost_human]   = {e_human:.4f}",
            f"E[Cost_abstain] = {q:.4f} × {self.c_abstain} = {e_abstain:.4f}",
            f"Decision: {action.value}  (minimum expected cost = {min_cost:.4f})",
        ]
        return "\n".join(lines)

    # ── Batch ────────────────────────────────────────────────────────────────

    def decide_batch(
        self,
        calibrated_probabilities: List[float],
        corruption_probabilities: Optional[List[float]] = None,
    ) -> List[DecisionResult]:
        """
        Apply the policy to a list of calibrated probabilities.

        Parameters
        ----------
        calibrated_probabilities : list[float]
            Each value must be in [0, 1].
        corruption_probabilities : list[float], optional
            One q per record, same order.  None (default) means q = 1.0
            for every record, i.e. the original behaviour.

        Returns
        -------
        list[DecisionResult]
            One result per input probability, in the same order.

        Raises
        ------
        ValueError
            If corruption_probabilities is given and its length differs
            from calibrated_probabilities.
        """
        if corruption_probabilities is None:
            return [self.decide(p) for p in calibrated_probabilities]
        if len(corruption_probabilities) != len(calibrated_probabilities):
            raise ValueError(
                f"corruption_probabilities length "
                f"({len(corruption_probabilities)}) must match "
                f"calibrated_probabilities length "
                f"({len(calibrated_probabilities)})"
            )
        return [
            self.decide(p, q)
            for p, q in zip(calibrated_probabilities, corruption_probabilities)
        ]

    # ── Breakeven thresholds ─────────────────────────────────────────────────

    def breakeven_auto_vs_human(self) -> Optional[float]:
        """
        Minimum p at which AUTO_REPAIR equals HUMAN_REVIEW in expected cost.

        Derivation
        ----------
        E[Cost_auto] = E[Cost_human]
        (1 - p) * C_wrong = C_human
        1 - p = C_human / C_wrong
        p = 1 - C_human / C_wrong

        Returns None if C_human >= C_wrong (AUTO never cheaper than HUMAN
        since even at p=1 the cost is 0 which equals C_human only when
        C_human = 0; the threshold would be >= 1).

        Returns
        -------
        float or None
            Threshold p* in [0, 1], or None if the threshold is outside [0, 1].
        """
        threshold = 1.0 - self.c_human / self.c_wrong
        if 0.0 <= threshold <= 1.0:
            return threshold
        return None

    def breakeven_auto_vs_abstain(
        self, corruption_probability: float = 1.0
    ) -> Optional[float]:
        """
        Minimum p at which AUTO_REPAIR equals ABSTAIN in expected cost.

        Derivation
        ----------
        (1 - p) * C_wrong = q * C_abstain
        p = 1 - q * C_abstain / C_wrong

        With q = 1 (default) this is the original 1 - C_abstain / C_wrong.
        As q → 0 abstaining becomes free and the threshold → 1.0: AUTO
        only ties ABSTAIN at p = 1, where both cost zero.

        Parameters
        ----------
        corruption_probability : float, optional
            q in [0, 1].  Defaults to 1.0.

        Returns None if the threshold is outside [0, 1] (the clamping rule
        is unchanged from the q = 1 version).
        """
        q = float(corruption_probability)
        if not (0.0 <= q <= 1.0):
            raise ValueError(
                f"corruption_probability must be in [0, 1], got {q}"
            )
        threshold = 1.0 - q * self.c_abstain / self.c_wrong
        if 0.0 <= threshold <= 1.0:
            return threshold
        return None
