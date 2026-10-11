"""
tests/test_decision_regions.py
-------------------------------
Item 5b.3 of docs/three_actions_plan.md: pin the cost semantics so they
cannot drift silently.

`docs/cost_semantics.md` makes four claims that the rest of the project —
the paper's framing, the dashboard's decision map, the Step 6 ablation —
relies on. None of them is enforced anywhere else:

  section 4  the (p, q) decision regions, with three boundaries
                 p* = 1 - C_human / C_wrong          (AUTO vs HUMAN)
                 q* = C_human / C_abstain            (HUMAN vs ABSTAIN)
                 p  = 1 - q * C_abstain / C_wrong    (AUTO vs ABSTAIN)
  section 4  the honest limitation: at a FIXED q, only two of the three
             actions are ever reachable as p varies
  section 2  realised cost, interpretation (a): C_abstain is charged only
             when the abstained cell was genuinely corrupted
  section 2  expectation/realisation correspondence: E[ABSTAIN] = q *
             C_abstain is an unbiased estimate of the realised charge

What each test guards against
-----------------------------
1/2  someone "simplifying" the argmin or the tie priority, which would
     move a boundary and invalidate every reported action distribution.
3    the model acquiring a p-dependent abstention arm without the docs
     being updated — if three actions ever become reachable at a fixed q,
     docs/cost_semantics.md section 4 is wrong and the paper's honest
     caveat must be rewritten.
4    a revert to unconditional abstain charging (interpretation (b)),
     which would again penalise the policy for correctly ignoring false
     alarms.  The (b) total is hand-computed here too, and asserted NOT to
     be what the evaluator returns.
5    the expectation and the realisation drifting apart, which is the
     methodological reason (a) was chosen over (b) in the first place.

Expected actions are re-derived here from the three cost formulas and the
documented tie priority AUTO > HUMAN > ABSTAIN.  The implementation is
never called to produce an expectation.

No ground truth reaches a decision: DecisionPolicy takes only (p, q) and
costs, and CostEvaluator sees clean_df only for post-hoc scoring.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from src.baselines.baselines import BaselineRecord
from src.decision.policy import DecisionAction, DecisionPolicy
from src.evaluation.cost_metrics import CostEvaluator
from src.risk.cost_model import CostConfig

AUTO = DecisionAction.AUTO_REPAIR
HUMAN = DecisionAction.HUMAN_REVIEW
ABSTAIN = DecisionAction.ABSTAIN

COSTS_YAML = Path("configs/costs.yaml")
SHIPPED_DATASETS = ("adult", "breast_cancer", "smartfactory")

# (c_wrong, c_human, c_abstain, label) — synthetic configs covering the cases
# the shipped three do not: cheap abstention, an exact HUMAN/ABSTAIN tie, an
# abstention so expensive it can never win, and zero-cost review.
SYNTHETIC_CONFIGS: List[Tuple[float, float, float, str]] = [
    (10.0, 5.0, 2.0, "c_abstain < c_human"),
    (10.0, 5.0, 5.0, "c_abstain == c_human"),
    (10.0, 2.0, 1000.0, "very large c_abstain"),
    (50.0, 0.0, 10.0, "free human review"),
    (1.0, 0.5, 0.25, "all costs below 1"),
    (100.0, 99.0, 100.0, "c_human just under c_wrong"),
]

GRID = [i / 100.0 for i in range(101)]  # 101 x 101 = 10,201 points


def _policies() -> List[Tuple[str, DecisionPolicy]]:
    """Every shipped cost config plus the synthetic ones."""
    out: List[Tuple[str, DecisionPolicy]] = []
    for name in SHIPPED_DATASETS:
        costs = CostConfig.from_yaml(COSTS_YAML, name)
        out.append((f"shipped:{name}", DecisionPolicy.from_cost_config(costs)))
    for c_wrong, c_human, c_abstain, label in SYNTHETIC_CONFIGS:
        out.append((
            f"synthetic:{label}",
            DecisionPolicy(c_wrong=c_wrong, c_human=c_human, c_abstain=c_abstain),
        ))
    return out


# ---------------------------------------------------------------------------
# Independent re-derivation of the expected action
# ---------------------------------------------------------------------------


def expected_action(
    p: float, q: float, c_wrong: float, c_human: float, c_abstain: float
) -> DecisionAction:
    """
    The minimum-cost action, derived here from the three cost formulas.

        E[AUTO]    = (1 - p) * C_wrong
        E[HUMAN]   = C_human
        E[ABSTAIN] = q * C_abstain

    Ties resolve by the documented priority AUTO > HUMAN > ABSTAIN, written
    out as an explicit chain of comparisons rather than an argmin, so this
    function shares no code path with DecisionPolicy.decide.
    """
    e_auto = (1.0 - p) * c_wrong
    e_human = c_human
    e_abstain = q * c_abstain
    if e_auto <= e_human and e_auto <= e_abstain:
        return AUTO
    if e_human <= e_abstain:
        return HUMAN
    return ABSTAIN


def region_action(
    p: float, q: float, c_wrong: float, c_human: float, c_abstain: float
) -> DecisionAction:
    """
    The same action, derived instead from the three region inequalities of
    docs/cost_semantics.md section 4:

        AUTO    <=> p >= 1 - C_h/C_w       and p >= 1 - q*C_a/C_w
        HUMAN   <=> p <  1 - C_h/C_w       and q >= C_h/C_a
        ABSTAIN <=> p <  1 - q*C_a/C_w     and q <  C_h/C_a
    """
    p_star = 1.0 - c_human / c_wrong
    q_star = c_human / c_abstain if c_abstain > 0 else math.inf
    p_slope = 1.0 - q * c_abstain / c_wrong
    if p >= p_star and p >= p_slope:
        return AUTO
    if q >= q_star:
        return HUMAN
    return ABSTAIN


def _near_tie(
    p: float, q: float, c_wrong: float, c_human: float, c_abstain: float,
    eps: float = 1e-9,
) -> bool:
    """True when two arms are within eps, where float form decides the tie."""
    costs = [(1.0 - p) * c_wrong, c_human, q * c_abstain]
    return any(
        abs(a - b) <= eps
        for i, a in enumerate(costs)
        for b in costs[i + 1:]
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. The whole (p, q) grid
# ═══════════════════════════════════════════════════════════════════════════════


class TestDecisionGrid:
    """DecisionPolicy.decide agrees with the derived regions everywhere."""

    @pytest.mark.parametrize("name,policy", _policies(), ids=lambda v: v if isinstance(v, str) else "")
    def test_grid_matches_cost_formulas(self, name: str, policy: DecisionPolicy):
        mismatches = []
        for p in GRID:
            for q in GRID:
                want = expected_action(
                    p, q, policy.c_wrong, policy.c_human, policy.c_abstain
                )
                got = policy.decide(p, corruption_probability=q).action
                if got != want:
                    mismatches.append((p, q, want.value, got.value))
        assert not mismatches, (
            f"{name}: {len(mismatches)} of {len(GRID) ** 2} grid points disagree "
            f"with the cost formulas; first five {mismatches[:5]}"
        )

    @pytest.mark.parametrize("name,policy", _policies(), ids=lambda v: v if isinstance(v, str) else "")
    def test_grid_matches_section4_region_inequalities(
        self, name: str, policy: DecisionPolicy
    ):
        """
        The section-4 inequalities are algebraically equivalent to the
        formulas, so they must agree at every point that is not sitting on a
        boundary (where only the floating-point form of the comparison
        decides, and the two forms round differently).
        """
        mismatches = []
        checked = 0
        for p in GRID:
            for q in GRID:
                if _near_tie(p, q, policy.c_wrong, policy.c_human, policy.c_abstain):
                    continue
                checked += 1
                want = region_action(
                    p, q, policy.c_wrong, policy.c_human, policy.c_abstain
                )
                got = policy.decide(p, corruption_probability=q).action
                if got != want:
                    mismatches.append((p, q, want.value, got.value))
        assert checked > 0.5 * len(GRID) ** 2, (
            f"{name}: only {checked} points were off-boundary; the test would "
            f"be vacuous"
        )
        assert not mismatches, (
            f"{name}: {len(mismatches)} off-boundary points disagree with the "
            f"section-4 regions; first five {mismatches[:5]}"
        )

    def test_grid_test_is_able_to_fail(self):
        """A policy with swapped costs must break the expectation."""
        good = DecisionPolicy(c_wrong=50, c_human=5, c_abstain=10)
        bad = DecisionPolicy(c_wrong=50, c_human=10, c_abstain=5)
        disagreements = sum(
            1
            for p in GRID
            for q in (0.0, 0.3, 0.6, 1.0)
            if bad.decide(p, q).action
            != expected_action(p, q, good.c_wrong, good.c_human, good.c_abstain)
        )
        assert disagreements > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. The three boundary formulas, crossed one at a time
# ═══════════════════════════════════════════════════════════════════════════════


EPS = 1e-3


class TestBoundaryFormulas:
    """Each documented boundary actually flips the action when crossed."""

    def test_p_star_flips_auto_and_human(self):
        """p* = 1 - C_h/C_w, with q above q* so HUMAN is the live alternative."""
        policy = DecisionPolicy(c_wrong=50, c_human=5, c_abstain=10)
        p_star = 1.0 - policy.c_human / policy.c_wrong
        assert math.isclose(p_star, 0.90)
        q = 1.0                       # q* = 0.5, so q > q*: ABSTAIN is dead
        assert policy.decide(p_star - EPS, q).action is HUMAN
        assert policy.decide(p_star + EPS, q).action is AUTO

    def test_q_star_flips_human_and_abstain(self):
        """q* = C_h/C_abstain, with p low enough that AUTO never wins."""
        policy = DecisionPolicy(c_wrong=50, c_human=5, c_abstain=10)
        q_star = policy.c_human / policy.c_abstain
        assert math.isclose(q_star, 0.50)
        p = 0.0
        assert policy.decide(p, q_star - EPS).action is ABSTAIN
        assert policy.decide(p, q_star + EPS).action is HUMAN

    def test_q_star_exact_tie_goes_to_human_by_priority(self):
        """At q == q* the two arms tie; priority puts HUMAN above ABSTAIN."""
        policy = DecisionPolicy(c_wrong=50, c_human=5, c_abstain=10)
        result = policy.decide(0.0, policy.c_human / policy.c_abstain)
        assert math.isclose(result.expected_cost_abstain, result.expected_cost_human)
        assert result.action is HUMAN

    def test_sloped_line_flips_auto_and_abstain(self):
        """p = 1 - q*C_abstain/C_w, with q below q* so HUMAN is the dead arm."""
        policy = DecisionPolicy(c_wrong=50, c_human=5, c_abstain=10)
        q = 0.2                       # q < q* = 0.5, and q*C_a = 2 < C_h = 5
        p_line = 1.0 - q * policy.c_abstain / policy.c_wrong
        assert math.isclose(p_line, 0.96)
        assert policy.decide(p_line - EPS, q).action is ABSTAIN
        assert policy.decide(p_line + EPS, q).action is AUTO

    def test_sloped_line_moves_with_q(self):
        """
        The AUTO/ABSTAIN boundary is sloped, not vertical: p = 1 - q*C_a/C_w
        falls as q rises, so a single p can sit on either side of it
        depending on q.  A higher q makes abstention more expensive and
        therefore makes AUTO the cheaper arm at the same confidence.
        """
        policy = DecisionPolicy(c_wrong=50, c_human=5, c_abstain=10)
        lines = {q: 1.0 - q * policy.c_abstain / policy.c_wrong
                 for q in (0.1, 0.4)}
        assert math.isclose(lines[0.1], 0.98)
        assert math.isclose(lines[0.4], 0.92)
        assert lines[0.1] > lines[0.4]

        p = 0.95                      # below the q=0.1 line, above the q=0.4 one
        assert policy.decide(p, 0.1).action is ABSTAIN
        assert policy.decide(p, 0.4).action is AUTO

    @pytest.mark.parametrize("dataset,p_star,q_star", [
        ("breast_cancer", 0.900, 0.500),
        ("smartfactory", 0.920, 0.400),
        ("adult", 0.800, 0.400),
    ])
    def test_shipped_boundaries_match_the_documented_table(
        self, dataset: str, p_star: float, q_star: float
    ):
        """docs/cost_semantics.md section 4's table of boundaries."""
        costs = CostConfig.from_yaml(COSTS_YAML, dataset)
        policy = DecisionPolicy.from_cost_config(costs)
        assert math.isclose(1.0 - policy.c_human / policy.c_wrong, p_star, abs_tol=1e-9)
        assert math.isclose(policy.c_human / policy.c_abstain, q_star, abs_tol=1e-9)
        assert math.isclose(policy.breakeven_auto_vs_human(), p_star, abs_tol=1e-9)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. The honest limitation: at most two actions at a fixed q
# ═══════════════════════════════════════════════════════════════════════════════


class TestFixedQReachability:
    """
    docs/cost_semantics.md section 4: both HUMAN and ABSTAIN are constant in
    p, so they never cross.  At a fixed q, sweeping p therefore yields at
    most two distinct actions, and which two is decided by q vs q*.

    If this ever yields three, the cost model has changed and the paper's
    caveat ("per cell only two arms are live, population-wide all three")
    is no longer true.
    """

    @pytest.mark.parametrize("name,policy", _policies(), ids=lambda v: v if isinstance(v, str) else "")
    def test_at_most_two_actions_per_fixed_q(self, name: str, policy: DecisionPolicy):
        for q in GRID:
            actions = {policy.decide(p, q).action for p in GRID}
            assert len(actions) <= 2, (
                f"{name}: q={q} yields {sorted(a.value for a in actions)} — three "
                f"actions at a fixed q contradicts docs/cost_semantics.md section 4"
            )

    def test_which_two_depends_on_q_versus_q_star(self):
        policy = DecisionPolicy(c_wrong=50, c_human=5, c_abstain=10)
        q_star = policy.c_human / policy.c_abstain          # 0.5

        below = {policy.decide(p, q_star - 0.1).action for p in GRID}
        above = {policy.decide(p, q_star + 0.1).action for p in GRID}

        assert below == {AUTO, ABSTAIN}, f"below q*: {below}"
        assert above == {AUTO, HUMAN}, f"above q*: {above}"
        assert HUMAN not in below      # review unreachable below q*
        assert ABSTAIN not in above    # abstention unreachable above q*

    def test_all_three_appear_across_a_population_of_mixed_q(self):
        """
        The population-level claim the paper makes: all three actions occur
        once q varies per cell.  Mirrors breast_cancer k=1.5 (high-p real
        errors, low-p real errors, and near-certain false alarms).
        """
        policy = DecisionPolicy(c_wrong=50, c_human=5, c_abstain=10)
        population = [(0.97, 0.99), (0.60, 0.99), (0.60, 0.04)]
        actions = {policy.decide(p, q).action for p, q in population}
        assert actions == {AUTO, HUMAN, ABSTAIN}

    def test_sweeping_p_at_q_equal_one_can_never_abstain(self):
        """The pre-branch degeneracy (CLAUDE.md section 5.2), still true at q=1."""
        for dataset in SHIPPED_DATASETS:
            costs = CostConfig.from_yaml(COSTS_YAML, dataset)
            policy = DecisionPolicy.from_cost_config(costs)
            assert costs.c_leave_error > costs.c_human_review, dataset
            actions = {policy.decide(p, 1.0).action for p in GRID}
            assert ABSTAIN not in actions, dataset


# ═══════════════════════════════════════════════════════════════════════════════
# 4. The realised-cost equation
# ═══════════════════════════════════════════════════════════════════════════════

C_WRONG, C_HUMAN, C_ABSTAIN = 10.0, 3.0, 5.0

# (action, genuinely corrupted, repair correct) — the full 3 x 2 x 2 cross.
COMBINATIONS: List[Tuple[DecisionAction, bool, bool]] = [
    (action, corrupted, correct)
    for action in (AUTO, HUMAN, ABSTAIN)
    for corrupted in (True, False)
    for correct in (True, False)
]


def _build_population() -> Tuple[pd.DataFrame, List[BaselineRecord]]:
    """
    One record per combination, on its own row of a one-column clean frame.

      clean value      = 10 * row_id
      dirty value      = clean           if not corrupted
                         clean + 1       if corrupted
      top candidate    = clean           if the repair is correct
                         clean + 7       if the repair is wrong
    """
    clean_values = [10 * i for i in range(len(COMBINATIONS))]
    clean_df = pd.DataFrame({"val": clean_values})
    records: List[BaselineRecord] = []
    for row_id, (action, corrupted, correct) in enumerate(COMBINATIONS):
        clean = clean_values[row_id]
        records.append(BaselineRecord(
            method_name="semantics_probe",
            dataset="synthetic",
            row_id=row_id,
            column="val",
            dirty_value=clean + 1 if corrupted else clean,
            top_candidate=clean if correct else clean + 7,
            raw_confidence=0.5,
            action=action,
        ))
    return clean_df, records


def _hand_total(conditional_abstain: bool) -> float:
    """
    The realised-cost equation of docs/cost_semantics.md section 2, written
    out term by term rather than looped over the evaluator's logic.

      cost_i = C_wrong   * [AUTO and repair wrong]
             + C_human   * [HUMAN]
             + C_abstain * [ABSTAIN] * corrupt_i      <- (a), conditional
             + C_abstain * [ABSTAIN]                  <- (b), unconditional
    """
    total = 0.0
    for action, corrupted, correct in COMBINATIONS:
        if action is AUTO and not correct:
            total += C_WRONG
        elif action is HUMAN:
            total += C_HUMAN
        elif action is ABSTAIN and (corrupted or not conditional_abstain):
            total += C_ABSTAIN
    return total


class TestRealisedCostEquation:
    """CostEvaluator implements interpretation (a), not (b)."""

    def test_evaluator_matches_interpretation_a(self):
        clean_df, records = _build_population()
        report = CostEvaluator(
            clean_df, c_wrong=C_WRONG, c_human=C_HUMAN, c_abstain=C_ABSTAIN
        ).evaluate(records, dataset="synthetic")

        # 4 AUTO (2 wrong -> 2*10), 4 HUMAN (4*3), 4 ABSTAIN (2 corrupted -> 2*5)
        assert math.isclose(_hand_total(conditional_abstain=True), 42.0)
        assert math.isclose(report.total_realized_cost, _hand_total(True))

    def test_evaluator_is_not_interpretation_b(self):
        """
        This is the test that fails if unconditional charging comes back.
        (b) differs from (a) by exactly C_abstain per abstention on a clean
        cell, and this population contains two of them.
        """
        clean_df, records = _build_population()
        report = CostEvaluator(
            clean_df, c_wrong=C_WRONG, c_human=C_HUMAN, c_abstain=C_ABSTAIN
        ).evaluate(records, dataset="synthetic")

        b_total = _hand_total(conditional_abstain=False)
        assert math.isclose(b_total, 52.0)
        assert report.n_abstain_on_clean == 2
        assert not math.isclose(report.total_realized_cost, b_total), (
            "CostEvaluator is charging C_abstain unconditionally — "
            "docs/cost_semantics.md section 2 chose interpretation (a)"
        )
        assert math.isclose(
            b_total - report.total_realized_cost,
            report.n_abstain_on_clean * C_ABSTAIN,
        )

    def test_two_interpretations_coincide_when_every_flag_is_real(self):
        """Why previously reported BC/SF k=3 numbers did not move."""
        clean_df = pd.DataFrame({"val": [0, 10, 20]})
        records = [
            BaselineRecord(
                method_name="all_corrupt", dataset="synthetic", row_id=i,
                column="val", dirty_value=10 * i + 1, top_candidate=10 * i,
                raw_confidence=0.5, action=ABSTAIN,
            )
            for i in range(3)
        ]
        report = CostEvaluator(
            clean_df, c_wrong=C_WRONG, c_human=C_HUMAN, c_abstain=C_ABSTAIN
        ).evaluate(records, dataset="synthetic")
        assert report.n_abstain_on_clean == 0
        assert math.isclose(report.total_realized_cost, 3 * C_ABSTAIN)

    def test_per_record_breakdown_counts(self):
        clean_df, records = _build_population()
        report = CostEvaluator(
            clean_df, c_wrong=C_WRONG, c_human=C_HUMAN, c_abstain=C_ABSTAIN
        ).evaluate(records, dataset="synthetic")
        assert (report.n_auto, report.n_human, report.n_abstain) == (4, 4, 4)
        assert report.n_wrong_auto == 2
        assert report.n_correct_auto == 2
        assert report.n_abstain_on_corrupted == 2
        assert report.n_abstain_on_clean == 2
        assert report.n_genuinely_corrupted == 6
        assert math.isclose(report.mean_cost_per_record, 42.0 / 12)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Expectation / realisation correspondence
# ═══════════════════════════════════════════════════════════════════════════════


class TestExpectationMatchesRealisation:
    """
    docs/cost_semantics.md section 2: (a) is the only pairing in which the
    expected cost the policy minimises is an unbiased estimate of the cost
    the evaluator charges.  With q set to the population's true corruption
    fraction f, the mean realised abstain cost must approach q * C_abstain.
    """

    N_CELLS = 20_000
    FRACTION = 0.30
    TOLERANCE = 0.02      # 2% relative

    def _population(self, f: float) -> Tuple[pd.DataFrame, List[BaselineRecord], int]:
        rng = np.random.default_rng(42)
        corrupted = rng.random(self.N_CELLS) < f
        clean_df = pd.DataFrame({"val": np.arange(self.N_CELLS, dtype=float)})
        records = [
            BaselineRecord(
                method_name="mc_probe", dataset="synthetic", row_id=i,
                column="val",
                dirty_value=float(i) + (1.0 if corrupted[i] else 0.0),
                top_candidate=float(i),
                raw_confidence=0.5, action=ABSTAIN,
            )
            for i in range(self.N_CELLS)
        ]
        return clean_df, records, int(corrupted.sum())

    def test_mean_realised_abstain_cost_approximates_q_times_c_abstain(self):
        clean_df, records, n_corrupt = self._population(self.FRACTION)
        q = self.FRACTION

        expected = DecisionPolicy(
            c_wrong=C_WRONG, c_human=C_HUMAN, c_abstain=C_ABSTAIN
        ).decide(0.0, corruption_probability=q).expected_cost_abstain
        assert math.isclose(expected, q * C_ABSTAIN)

        report = CostEvaluator(
            clean_df, c_wrong=C_WRONG, c_human=C_HUMAN, c_abstain=C_ABSTAIN
        ).evaluate(records, dataset="synthetic")
        assert report.n_abstain == self.N_CELLS
        assert report.n_abstain_on_corrupted == n_corrupt

        realised_mean = report.total_realized_cost / report.n_abstain
        assert math.isclose(realised_mean, expected, rel_tol=self.TOLERANCE), (
            f"mean realised abstain cost {realised_mean:.4f} vs "
            f"E[ABSTAIN] = q*C_abstain = {expected:.4f} "
            f"({n_corrupt}/{self.N_CELLS} cells corrupted)"
        )

    def test_unconditional_charge_would_be_biased_by_the_same_population(self):
        """
        The counterfactual: under (b) every abstention costs C_abstain, so
        the realised mean would be C_abstain regardless of q — more than
        three times E[ABSTAIN] here.  That bias is why (a) was chosen.
        """
        _, _, n_corrupt = self._population(self.FRACTION)
        expected = self.FRACTION * C_ABSTAIN
        unconditional_mean = C_ABSTAIN
        assert not math.isclose(unconditional_mean, expected, rel_tol=self.TOLERANCE)
        assert unconditional_mean > 3 * expected
        realised_a = n_corrupt * C_ABSTAIN / self.N_CELLS
        assert abs(realised_a - expected) < abs(unconditional_mean - expected)

    @pytest.mark.parametrize("f", [0.0, 0.1, 0.5, 1.0])
    def test_correspondence_holds_across_corruption_fractions(self, f: float):
        clean_df, records, n_corrupt = self._population(f)
        report = CostEvaluator(
            clean_df, c_wrong=C_WRONG, c_human=C_HUMAN, c_abstain=C_ABSTAIN
        ).evaluate(records, dataset="synthetic")
        realised_mean = report.total_realized_cost / self.N_CELLS
        expected = f * C_ABSTAIN
        assert abs(realised_mean - expected) <= max(0.02 * expected, 0.02)
