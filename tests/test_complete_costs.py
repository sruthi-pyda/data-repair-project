"""
tests/test_complete_costs.py
----------------------------
Route W5.1a (docs/research_program.md §11, reviewer gap 4a): the
complete-cost mode of :class:`src.evaluation.cost_metrics.CostEvaluator`.

WHAT IS ACTUALLY BEING TESTED
=============================
Every realised cost the project has published prices UNDETECTED
corruptions at zero.  ``evaluate(include_false_negatives=True)`` adds

    surcharge = C_abstain x |{corrupted cells} - {flagged cells}|

Five properties matter, and each one is written so it CAN fail:

  1. the default mode is bit-identical to the historical behaviour —
     checked against a total re-derived by hand in the test, not against
     a second call of the code under test;
  2. the surcharge is EXACTLY C_abstain x n_unflagged, with n_unflagged
     hand-counted from frames small enough to count by eye;
  3. perfect detection recall => zero surcharge;
  4. zero detection recall => surcharge = every corruption x C_abstain
     (including the degenerate "flagged nothing at all" case, where the
     flagged-only cost is 0 and the real harm is maximal);
  5. **the surcharge is policy-invariant.**  It is a function of the
     detector's flag set and of ground truth, never of the action chosen,
     so it adds the same constant to every policy and cannot reorder
     them.  This is the key property: it is what makes the complete-cost
     number honest to report and useless for ranking.

Every expected number below is derived in a comment from the fixture
frames.  No test calls ``evaluate`` to produce its own expectation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.baselines.baselines import BaselineRecord
from src.decision.policy import DecisionAction
from src.evaluation.cost_metrics import CostEvaluator, CostReport, corrupted_cells
from src.risk.cost_model import CostConfig

AUTO = DecisionAction.AUTO_REPAIR
HUMAN = DecisionAction.HUMAN_REVIEW
ABSTAIN = DecisionAction.ABSTAIN

C_WRONG, C_HUMAN, C_ABSTAIN = 50.0, 5.0, 10.0


# ═══════════════════════════════════════════════════════════════════════════════
# Fixture: five rows, two columns, THREE corrupted cells (countable by eye)
# ═══════════════════════════════════════════════════════════════════════════════
#
#   row | clean a | clean b | dirty a | dirty b | corrupted?
#   ----+---------+---------+---------+---------+---------------------------
#    0  |    1    |   "x"   |    1    |   "x"   | no
#    1  |    2    |   "y"   |   99    |   "y"   | YES  (1, "a")
#    2  |    3    |   "z"   |    3    |  "ZZ"   | YES  (2, "b")
#    3  |    4    |   "w"   |    4    |   "w"   | no
#    4  |    5    |   "v"   |  NaN    |   "v"   | YES  (4, "a")
#
# TOTAL CORRUPTED CELLS = 3


def _clean() -> pd.DataFrame:
    return pd.DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "w", "v"]}
    )


def _dirty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 99.0, 3.0, 4.0, np.nan],
            "b": ["x", "y", "ZZ", "w", "v"],
        }
    )


ALL_CORRUPTED = {(1, "a"), (2, "b"), (4, "a")}


def _rec(
    row: int,
    column: str,
    dirty_value,
    top_candidate,
    action: DecisionAction,
    method: str = "m",
) -> BaselineRecord:
    return BaselineRecord(
        method_name=method,
        dataset="fixture",
        row_id=row,
        column=column,
        dirty_value=dirty_value,
        top_candidate=top_candidate,
        raw_confidence=0.5,
        action=action,
    )


def _ev(dirty: pd.DataFrame | None = None) -> CostEvaluator:
    return CostEvaluator(
        clean_df=_clean(),
        c_wrong=C_WRONG,
        c_human=C_HUMAN,
        c_abstain=C_ABSTAIN,
        dirty_df=dirty,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. corrupted_cells — the census itself
# ═══════════════════════════════════════════════════════════════════════════════


class TestCorruptionCensus:

    def test_finds_exactly_the_three_hand_counted_cells(self):
        assert corrupted_cells(_clean(), _dirty()) == ALL_CORRUPTED

    def test_numeric_dtype_promotion_is_not_corruption(self):
        """371 (int) vs 371.0 (float) is the same value, not an error."""
        clean = pd.DataFrame({"a": [371, 2]})
        dirty = pd.DataFrame({"a": [371.0, 2.0]})
        assert corrupted_cells(clean, dirty) == set()

    def test_both_missing_is_not_corruption(self):
        clean = pd.DataFrame({"a": [np.nan, 1.0]})
        dirty = pd.DataFrame({"a": [np.nan, 1.0]})
        assert corrupted_cells(clean, dirty) == set()

    def test_exactly_one_missing_is_corruption_either_way(self):
        clean = pd.DataFrame({"a": [np.nan, 1.0]})
        dirty = pd.DataFrame({"a": [1.0, np.nan]})
        assert corrupted_cells(clean, dirty) == {(0, "a"), (1, "a")}

    def test_whitespace_only_difference_is_not_corruption(self):
        clean = pd.DataFrame({"b": ["Private", "State-gov"]})
        dirty = pd.DataFrame({"b": [" Private ", "State-gov"]})
        assert corrupted_cells(clean, dirty) == set()

    def test_skip_columns_excludes_that_column_entirely(self):
        assert corrupted_cells(_clean(), _dirty(), skip_columns=["a"]) == {(2, "b")}
        assert corrupted_cells(_clean(), _dirty(), skip_columns=["a", "b"]) == set()

    def test_non_range_index_still_yields_positional_keys(self):
        """row_id is a POSITION, so a shifted index must not shift the keys."""
        clean = _clean()
        dirty = _dirty()
        clean.index = [100, 101, 102, 103, 104]
        dirty.index = [7, 8, 9, 10, 11]
        assert corrupted_cells(clean, dirty) == ALL_CORRUPTED

    def test_column_absent_from_dirty_is_ignored(self):
        clean = pd.DataFrame({"a": [1, 2], "ghost": [1, 2]})
        dirty = pd.DataFrame({"a": [1, 2]})
        assert corrupted_cells(clean, dirty) == set()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Default mode is bit-identical to the historical behaviour
# ═══════════════════════════════════════════════════════════════════════════════


class TestDefaultModeUnchanged:
    """
    Hand-built batch, cost re-derived here from the realised-cost equation
    of docs/cost_semantics.md §2 rather than from the code under test.

    Records:
      (1,"a") AUTO,    top_candidate 2   == clean 2      -> correct   -> 0
      (2,"b") AUTO,    top_candidate "Q" != clean "z"    -> wrong     -> 50
      (3,"a") HUMAN                                                   ->  5
      (4,"a") ABSTAIN, dirty NaN != clean 5 -> corrupted              -> 10
      (0,"a") ABSTAIN, dirty 1   == clean 1 -> false alarm            ->  0
    EXPECTED flagged-only total = 0 + 50 + 5 + 10 + 0 = 65
    """

    EXPECTED_TOTAL = 65.0

    def _batch(self):
        return [
            _rec(1, "a", 99.0, 2, AUTO),
            _rec(2, "b", "ZZ", "Q", AUTO),
            _rec(3, "a", 4.0, 4, HUMAN),
            _rec(4, "a", np.nan, 5, ABSTAIN),
            _rec(0, "a", 1.0, 1, ABSTAIN),
        ]

    def test_no_argument_matches_hand_computed_total(self):
        r = _ev().evaluate(self._batch(), dataset="fixture")
        assert math.isclose(r.total_realized_cost, self.EXPECTED_TOTAL)

    def test_explicit_false_matches_hand_computed_total(self):
        r = _ev(_dirty()).evaluate(
            self._batch(), dataset="fixture", include_false_negatives=False
        )
        assert math.isclose(r.total_realized_cost, self.EXPECTED_TOTAL)

    def test_default_and_explicit_false_agree_field_for_field(self):
        a = _ev().evaluate(self._batch(), dataset="fixture").to_dict()
        b = _ev(_dirty()).evaluate(
            self._batch(), dataset="fixture", include_false_negatives=False
        ).to_dict()
        assert a == b

    def test_default_mode_reports_no_surcharge_and_no_census(self):
        r = _ev(_dirty()).evaluate(self._batch(), dataset="fixture")
        assert r.include_false_negatives is False
        assert r.false_negative_surcharge == 0.0
        assert r.n_total_corrupted_cells == 0
        assert r.n_unflagged_corrupted == 0
        # detection_recall must be None, not 0.0: 0.0 reads as a measurement.
        assert r.detection_recall is None
        assert math.isclose(r.flagged_only_cost, self.EXPECTED_TOTAL)

    def test_default_mode_needs_no_dirty_df(self):
        """The historical constructor took no dirty_df and must keep working."""
        r = CostEvaluator(
            clean_df=_clean(), c_wrong=C_WRONG, c_human=C_HUMAN, c_abstain=C_ABSTAIN
        ).evaluate(self._batch(), dataset="fixture")
        assert math.isclose(r.total_realized_cost, self.EXPECTED_TOTAL)

    def test_from_cost_config_without_dirty_df_still_works(self):
        costs = CostConfig(
            c_wrong_repair=C_WRONG, c_human_review=C_HUMAN,
            c_leave_error=C_ABSTAIN, dataset_name="fixture",
        )
        r = CostEvaluator.from_cost_config(_clean(), costs).evaluate(self._batch())
        assert math.isclose(r.total_realized_cost, self.EXPECTED_TOTAL)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. The surcharge is exactly C_abstain x (corrupted and never flagged)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSurchargeMagnitude:
    """
    Flag set = {(1,"a"), (3,"a")}.
      (1,"a") is corrupted -> flagged, so NOT a false negative
      (3,"a") is clean     -> a false alarm, irrelevant to the surcharge
    Corrupted cells = {(1,"a"), (2,"b"), (4,"a")}
    UNFLAGGED corrupted = {(2,"b"), (4,"a")}  -> 2 cells
    EXPECTED surcharge = 2 x C_abstain = 2 x 10 = 20
    """

    EXPECTED_UNFLAGGED = 2
    EXPECTED_SURCHARGE = 20.0

    def _batch(self):
        # (1,"a") AUTO with a correct repair -> 0; (3,"a") HUMAN -> 5
        return [
            _rec(1, "a", 99.0, 2, AUTO),
            _rec(3, "a", 4.0, 4, HUMAN),
        ]

    FLAGGED_ONLY = 5.0  # 0 + C_HUMAN

    def test_counts_are_the_hand_counted_ones(self):
        r = _ev(_dirty()).evaluate(
            self._batch(), dataset="fixture", include_false_negatives=True
        )
        assert r.n_total_corrupted_cells == 3
        assert r.n_genuinely_corrupted == 1          # flagged AND corrupted
        assert r.n_unflagged_corrupted == self.EXPECTED_UNFLAGGED

    def test_surcharge_is_exactly_c_abstain_times_unflagged(self):
        r = _ev(_dirty()).evaluate(
            self._batch(), dataset="fixture", include_false_negatives=True
        )
        assert math.isclose(r.false_negative_surcharge, self.EXPECTED_SURCHARGE)

    def test_total_is_flagged_only_plus_surcharge(self):
        r = _ev(_dirty()).evaluate(
            self._batch(), dataset="fixture", include_false_negatives=True
        )
        assert math.isclose(r.flagged_only_cost, self.FLAGGED_ONLY)
        assert math.isclose(
            r.total_realized_cost, self.FLAGGED_ONLY + self.EXPECTED_SURCHARGE
        )

    def test_difference_between_the_two_modes_is_exactly_the_surcharge(self):
        off = _ev(_dirty()).evaluate(self._batch(), dataset="fixture")
        on = _ev(_dirty()).evaluate(
            self._batch(), dataset="fixture", include_false_negatives=True
        )
        assert math.isclose(
            on.total_realized_cost - off.total_realized_cost,
            self.EXPECTED_SURCHARGE,
        )

    def test_detection_recall_is_flagged_corruptions_over_all_corruptions(self):
        r = _ev(_dirty()).evaluate(
            self._batch(), dataset="fixture", include_false_negatives=True
        )
        assert math.isclose(r.detection_recall, 1.0 / 3.0)

    def test_mode_flag_is_recorded_on_the_report(self):
        r = _ev(_dirty()).evaluate(
            self._batch(), dataset="fixture", include_false_negatives=True
        )
        assert r.include_false_negatives is True
        assert r.to_dict()["include_false_negatives"] is True

    def test_dirty_df_passed_to_evaluate_instead_of_constructor(self):
        r = _ev().evaluate(
            self._batch(), dataset="fixture",
            include_false_negatives=True, dirty_df=_dirty(),
        )
        assert math.isclose(r.false_negative_surcharge, self.EXPECTED_SURCHARGE)

    def test_missing_dirty_df_raises(self):
        with pytest.raises(ValueError, match="dirty"):
            _ev().evaluate(
                self._batch(), dataset="fixture", include_false_negatives=True
            )

    def test_skip_columns_removes_those_false_negatives(self):
        """
        Skipping column "b" removes (2,"b") from the census, leaving
        corrupted = {(1,"a"), (4,"a")} and unflagged = {(4,"a")} -> 1 x 10.
        """
        r = _ev(_dirty()).evaluate(
            self._batch(), dataset="fixture",
            include_false_negatives=True, skip_columns=["b"],
        )
        assert r.n_total_corrupted_cells == 2
        assert r.n_unflagged_corrupted == 1
        assert math.isclose(r.false_negative_surcharge, C_ABSTAIN)

    def test_memo_is_not_reused_for_a_different_dirty_frame(self):
        """A cached census must never be served to a different dataset."""
        ev = _ev()
        first = ev.evaluate(
            self._batch(), include_false_negatives=True, dirty_df=_dirty()
        )
        # A frame with ONE corruption only: (0,"a").
        other = pd.DataFrame({"a": [9, 2, 3, 4, 5], "b": ["x", "y", "z", "w", "v"]})
        second = ev.evaluate(
            self._batch(), include_false_negatives=True, dirty_df=other
        )
        assert first.n_total_corrupted_cells == 3
        assert second.n_total_corrupted_cells == 1
        # (0,"a") is not in the flag set, so it is the one false negative.
        assert second.n_unflagged_corrupted == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 4. The two extremes of detection recall
# ═══════════════════════════════════════════════════════════════════════════════


class TestRecallExtremes:

    def test_perfect_recall_has_zero_surcharge(self):
        """Flag all three corrupted cells -> nothing is undetected."""
        batch = [
            _rec(1, "a", 99.0, 2, HUMAN),
            _rec(2, "b", "ZZ", "z", HUMAN),
            _rec(4, "a", np.nan, 5, HUMAN),
        ]
        r = _ev(_dirty()).evaluate(
            batch, dataset="fixture", include_false_negatives=True
        )
        assert r.n_total_corrupted_cells == 3
        assert r.n_unflagged_corrupted == 0
        assert r.false_negative_surcharge == 0.0
        assert math.isclose(r.detection_recall, 1.0)
        # 3 x C_HUMAN, and the complete-cost total adds nothing.
        assert math.isclose(r.total_realized_cost, 3 * C_HUMAN)
        assert math.isclose(r.flagged_only_cost, r.total_realized_cost)

    def test_perfect_recall_plus_false_alarms_still_has_zero_surcharge(self):
        """Precision is irrelevant to the surcharge; only recall matters."""
        batch = [
            _rec(1, "a", 99.0, 2, ABSTAIN),
            _rec(2, "b", "ZZ", "z", ABSTAIN),
            _rec(4, "a", np.nan, 5, ABSTAIN),
            _rec(0, "a", 1.0, 1, ABSTAIN),   # false alarm
            _rec(3, "a", 4.0, 4, ABSTAIN),   # false alarm
        ]
        r = _ev(_dirty()).evaluate(
            batch, dataset="fixture", include_false_negatives=True
        )
        assert r.n_unflagged_corrupted == 0
        assert r.false_negative_surcharge == 0.0

    def test_zero_recall_surcharge_is_every_corruption(self):
        """Flag only clean cells: flagged-only cost is tiny, real harm is not."""
        batch = [
            _rec(0, "a", 1.0, 1, ABSTAIN),
            _rec(3, "a", 4.0, 4, ABSTAIN),
        ]
        r = _ev(_dirty()).evaluate(
            batch, dataset="fixture", include_false_negatives=True
        )
        assert r.n_genuinely_corrupted == 0
        assert r.n_unflagged_corrupted == 3
        assert math.isclose(r.flagged_only_cost, 0.0)
        assert math.isclose(r.false_negative_surcharge, 3 * C_ABSTAIN)
        assert math.isclose(r.total_realized_cost, 3 * C_ABSTAIN)
        assert math.isclose(r.detection_recall, 0.0)

    def test_flagging_nothing_at_all_charges_the_whole_dataset(self):
        """
        The degenerate case the flagged-only metric hides completely: an
        empty record list costs 0 today and 3 x C_abstain in truth.
        """
        off = _ev(_dirty()).evaluate([], dataset="fixture")
        on = _ev(_dirty()).evaluate(
            [], dataset="fixture", include_false_negatives=True
        )
        assert off.total_realized_cost == 0.0
        assert on.n_records == 0
        assert on.n_unflagged_corrupted == 3
        assert math.isclose(on.false_negative_surcharge, 3 * C_ABSTAIN)
        assert math.isclose(on.total_realized_cost, 3 * C_ABSTAIN)

    def test_clean_dataset_has_no_surcharge_and_undefined_recall(self):
        ev = CostEvaluator(
            clean_df=_clean(), c_wrong=C_WRONG, c_human=C_HUMAN,
            c_abstain=C_ABSTAIN, dirty_df=_clean(),
        )
        r = ev.evaluate(
            [_rec(0, "a", 1, 1, ABSTAIN)], include_false_negatives=True
        )
        assert r.n_total_corrupted_cells == 0
        assert r.false_negative_surcharge == 0.0
        # No corruption at all -> recall is undefined, not 0.0 and not 1.0.
        assert r.detection_recall is None


# ═══════════════════════════════════════════════════════════════════════════════
# 5. THE KEY PROPERTY: the surcharge is policy-invariant
# ═══════════════════════════════════════════════════════════════════════════════


class TestSurchargeIsPolicyInvariant:
    """
    The surcharge depends on the detector's flag SET and on ground truth.
    It does not depend on which action the policy chose, so for a fixed
    detector it adds the same constant to every policy and cannot reorder
    them.  Stated as a property and checked over every action assignment
    the policies could possibly make.
    """

    FLAGS = [(1, "a", 99.0, 2), (3, "a", 4.0, 4), (0, "a", 1.0, 1)]
    # Corrupted = {(1,"a"), (2,"b"), (4,"a")}; flagged = {(1,"a"), (3,"a"), (0,"a")}
    # UNFLAGGED corrupted = {(2,"b"), (4,"a")} -> surcharge 2 x 10 = 20
    EXPECTED_SURCHARGE = 20.0

    def _policy(self, action_per_flag, name):
        return [
            _rec(row, col, dirty, cand, act, method=name)
            for (row, col, dirty, cand), act in zip(self.FLAGS, action_per_flag)
        ]

    def test_same_surcharge_for_every_uniform_policy(self):
        ev = _ev(_dirty())
        surcharges = {}
        for action, name in ((AUTO, "always-auto"), (HUMAN, "always-human"),
                             (ABSTAIN, "always-abstain")):
            r = ev.evaluate(
                self._policy([action] * 3, name), include_false_negatives=True
            )
            surcharges[name] = r.false_negative_surcharge
        assert set(surcharges.values()) == {self.EXPECTED_SURCHARGE}

    def test_same_surcharge_for_all_27_action_assignments(self):
        """Exhaustive over the action space, so no policy can be a special case."""
        ev = _ev(_dirty())
        seen = set()
        for a in (AUTO, HUMAN, ABSTAIN):
            for b in (AUTO, HUMAN, ABSTAIN):
                for c in (AUTO, HUMAN, ABSTAIN):
                    r = ev.evaluate(
                        self._policy([a, b, c], "mixed"),
                        include_false_negatives=True,
                    )
                    seen.add(round(r.false_negative_surcharge, 9))
                    assert r.n_unflagged_corrupted == 2
        assert seen == {self.EXPECTED_SURCHARGE}

    def test_ranking_is_preserved_because_the_shift_is_constant(self):
        """
        The consequence that matters for the paper: pricing false negatives
        changes magnitudes and never the order between policies.
        """
        ev = _ev(_dirty())
        policies = {
            "always-auto": [AUTO] * 3,
            "always-human": [HUMAN] * 3,
            "always-abstain": [ABSTAIN] * 3,
            "mixed": [AUTO, HUMAN, ABSTAIN],
        }
        off, on = {}, {}
        for name, actions in policies.items():
            recs = self._policy(actions, name)
            off[name] = ev.evaluate(recs).total_realized_cost
            on[name] = ev.evaluate(
                recs, include_false_negatives=True
            ).total_realized_cost
        order_off = sorted(off, key=lambda k: (off[k], k))
        order_on = sorted(on, key=lambda k: (on[k], k))
        assert order_off == order_on
        for name in policies:
            assert math.isclose(on[name] - off[name], self.EXPECTED_SURCHARGE)

    def test_surcharge_depends_on_the_flag_set_not_the_actions(self):
        """Change the flag set, and only then does the surcharge move."""
        ev = _ev(_dirty())
        narrow = ev.evaluate(
            [_rec(1, "a", 99.0, 2, AUTO)], include_false_negatives=True
        )
        wide = ev.evaluate(
            [
                _rec(1, "a", 99.0, 2, AUTO),
                _rec(2, "b", "ZZ", "z", AUTO),
            ],
            include_false_negatives=True,
        )
        assert math.isclose(narrow.false_negative_surcharge, 2 * C_ABSTAIN)
        assert math.isclose(wide.false_negative_surcharge, 1 * C_ABSTAIN)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Backwards compatibility of the report object
# ═══════════════════════════════════════════════════════════════════════════════


class TestReportBackwardsCompatibility:

    def test_old_constructor_still_valid(self):
        r = CostReport(
            method_name="m", dataset="d", n_records=3, total_realized_cost=10.0
        )
        assert r.include_false_negatives is False
        assert r.n_total_corrupted_cells == 0
        assert r.n_unflagged_corrupted == 0
        assert r.false_negative_surcharge == 0.0
        assert math.isclose(r.flagged_only_cost, 10.0)
        assert r.detection_recall is None

    def test_to_dict_carries_the_new_keys(self):
        d = CostReport(method_name="m", dataset="d", n_records=0).to_dict()
        for key in (
            "include_false_negatives",
            "n_total_corrupted_cells",
            "n_unflagged_corrupted",
            "false_negative_surcharge",
            "flagged_only_cost",
            "detection_recall",
        ):
            assert key in d

    def test_negative_abstain_cost_still_rejected(self):
        with pytest.raises(ValueError):
            CostEvaluator(
                clean_df=_clean(), c_wrong=1.0, c_human=1.0, c_abstain=-1.0
            )
