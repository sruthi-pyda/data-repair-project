"""
tests/test_plots.py
-------------------
Phase 11: figure-returning helpers in src/visualization/plots.py.

Every helper must return a matplotlib Figure, survive empty input, label
cost axes in experimental units, and never touch clean data. The expected
cost curve is additionally checked for its arithmetic, since it is the
dashboard's centrepiece.
"""

from __future__ import annotations

import inspect

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from src.calibration.calibrator import CalibrationSample, calibrate_dataset
from src.decision.policy import DecisionAction
from src.evaluation.cost_metrics import CostReport, PolicyComparisonRow
from src.visualization import plots as P


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _row(name: str, cost: float, auto=0.3, human=0.6, abstain=0.1) -> PolicyComparisonRow:
    return PolicyComparisonRow(
        method_name=name, dataset="test",
        detection_precision=0.9, detection_recall=0.8, detection_f1=0.85,
        repair_accuracy=0.7, auto_rate=auto, human_rate=human, abstain_rate=abstain,
        wrong_auto_rate=0.1, total_realized_cost=cost, mean_cost_per_record=cost / 100,
    )


def _cost_report(name: str, cost: float) -> CostReport:
    return CostReport(method_name=name, dataset="test", n_records=100,
                      total_realized_cost=cost, mean_cost_per_record=cost / 100)


ROWS = [_row("calibrated_risk_aware", 900), _row("always_auto", 2500, 1, 0, 0),
        _row("confidence_threshold_0.80", 1100, 0.1, 0.9, 0)]


class TestCandidateBar:
    def test_returns_figure(self):
        fig = P.candidate_bar([("5", 0.7), ("4", 0.2), ("6", 0.1)])
        assert isinstance(fig, Figure)
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert labels == ["5", "4", "6"]

    def test_empty(self):
        assert isinstance(P.candidate_bar([]), Figure)


class TestExpectedCostCurves:
    def test_lines_encode_formulas(self):
        fig = P.expected_cost_curves(c_wrong=50, c_human=5, c_abstain=10, p=0.6)
        ax = fig.axes[0]
        auto, human, abstain = ax.get_lines()[:3]
        xs, ys = auto.get_xdata(), auto.get_ydata()
        assert np.allclose(ys, (1 - xs) * 50)
        assert np.allclose(human.get_ydata(), 5)
        assert np.allclose(abstain.get_ydata(), 10)

    def test_axes_labelled_in_experimental_units(self):
        fig = P.expected_cost_curves(10, 2, 5)
        assert P.COST_UNITS_LABEL in fig.axes[0].get_ylabel()
        assert "money" not in fig.axes[0].get_ylabel().lower()

    def test_marker_drawn_when_p_given(self):
        with_p = P.expected_cost_curves(10, 2, 5, p=0.4)
        without = P.expected_cost_curves(10, 2, 5)
        assert len(with_p.axes[0].get_lines()) > len(without.axes[0].get_lines())

    @pytest.mark.parametrize("p", [0.0, 1.0, -0.5, 1.5])
    def test_edge_probabilities_do_not_crash(self, p):
        assert isinstance(P.expected_cost_curves(10, 2, 5, p=p), Figure)

    def test_zero_or_negative_c_wrong_rejected(self):
        with pytest.raises(ValueError):
            P.expected_cost_curves(0, 2, 5)
        with pytest.raises(ValueError):
            P.expected_cost_curves(-1, 2, 5)

    def test_equal_costs_tie_handled(self):
        assert isinstance(P.expected_cost_curves(10, 5, 5, p=0.5), Figure)


class TestDecisionScatter:
    def test_points_coloured_by_action(self):
        probs = [0.1, 0.5, 0.95]
        acts = [DecisionAction.ABSTAIN, DecisionAction.HUMAN_REVIEW, DecisionAction.AUTO_REPAIR]
        fig = P.decision_scatter(probs, acts, 10, 2, 1)
        assert len(fig.axes[0].collections) >= 3

    def test_empty(self):
        assert isinstance(P.decision_scatter([], [], 10, 2, 1), Figure)


class TestConfidenceHistogram:
    def test_two_series(self):
        fig = P.confidence_histogram([0.2, 0.5, 0.8], [0.3, 0.4, 0.9], "isotonic")
        assert isinstance(fig, Figure)
        assert len(fig.axes[0].patches) > 0

    def test_empty(self):
        assert isinstance(P.confidence_histogram([], []), Figure)


class TestReliabilityFigure:
    def test_fitted_methods_drawn(self):
        rng = np.random.default_rng(0)
        samples = []
        for i in range(150):
            c = float(rng.uniform(0.3, 0.9))
            samples.append(CalibrationSample("t", i, "c", c, int(rng.random() < c),
                                             float(rng.uniform(0.5, 1)), float(rng.uniform(0, 1))))
        comp = calibrate_dataset(samples, "t")
        fig = P.reliability_figure(comp)
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert any(l.startswith("raw") for l in labels)
        assert any(l.startswith("multivariate") for l in labels)

    def test_unfitted_methods_omitted(self):
        samples = [CalibrationSample("t", i, "c", 0.6, 0) for i in range(30)]
        comp = calibrate_dataset(samples, "t")
        fig = P.reliability_figure(comp)
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert not any(l.startswith(("logistic", "isotonic", "multivariate")) for l in labels)


class TestPolicyFigures:
    def test_action_distribution(self):
        fig = P.action_distribution(ROWS)
        ticks = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert len(ticks) == 3
        assert any("Proposed" in t for t in ticks)
        assert any("≥ 0.80" in t for t in ticks)

    def test_cost_comparison_units(self):
        fig = P.cost_comparison(ROWS)
        assert P.COST_UNITS_LABEL in fig.axes[0].get_ylabel()
        assert len(fig.axes[0].patches) == 3

    def test_threshold_sweep(self):
        sweep = {t: _cost_report(f"confidence_threshold_{t:.2f}", 2000 - 1000 * t)
                 for t in (0.5, 0.7, 0.9)}
        fig = P.threshold_sweep(sweep, proposed_cost=900, current_threshold=0.7)
        assert isinstance(fig, Figure)
        assert P.COST_UNITS_LABEL in fig.axes[0].get_ylabel()

    @pytest.mark.parametrize("fn", [P.action_distribution, P.cost_comparison])
    def test_empty_rows(self, fn):
        assert isinstance(fn([]), Figure)

    def test_empty_sweep(self):
        assert isinstance(P.threshold_sweep({}), Figure)


class TestNoLeakage:
    def test_no_plot_function_takes_clean_data(self):
        for name, fn in inspect.getmembers(P, inspect.isfunction):
            if fn.__module__ != P.__name__ or name.startswith("_"):
                continue
            params = list(inspect.signature(fn).parameters)
            assert not any("clean" in p.lower() for p in params), name

    def test_module_body_never_mentions_clean_data(self):
        """The module docstring states the rule; the code must not need it."""
        src = inspect.getsource(P)
        body = src.split('"""', 2)[2]
        assert "clean" not in body.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: calibrated abstention — q in the cost curves, and the (p, q) map
# ═══════════════════════════════════════════════════════════════════════════════

BC_COSTS = (50.0, 5.0, 10.0)  # breast_cancer: C_wrong, C_human, C_abstain


def _independent_optimal(p: float, q: float, c_wrong: float, c_human: float,
                         c_abstain: float) -> str:
    """
    Re-derivation of the optimal action written from scratch for this test:
    the plain argmin of the three expectations with the documented tie
    priority AUTO_REPAIR > HUMAN_REVIEW > ABSTAIN (``list.index(min(...))``
    returns the first minimum). It deliberately uses nothing from
    ``src.visualization.plots`` and does not reuse the region inequalities of
    docs/cost_semantics.md section 4 — it checks them.
    """
    expectations = [(1.0 - p) * c_wrong, c_human, q * c_abstain]
    names = ("AUTO_REPAIR", "HUMAN_REVIEW", "ABSTAIN")
    return names[expectations.index(min(expectations))]


def _drawn_action(fig: Figure, p_index: int, q_index: int) -> str:
    """Action the figure actually shaded at a grid cell (row = q, col = p)."""
    grid = np.asarray(fig.axes[0].images[0].get_array())
    return P.DECISION_MAP_ACTIONS[int(grid[q_index, p_index])].value


def _region_values(fig: Figure) -> set:
    return set(np.asarray(fig.axes[0].images[0].get_array()).ravel().tolist())


class TestExpectedCostCurvesWithQ:
    def test_abstain_line_sits_at_q_times_c_abstain(self):
        fig = P.expected_cost_curves(50, 5, 10, p=0.6, corruption_probability=0.3)
        abstain = fig.axes[0].get_lines()[2]
        assert np.allclose(abstain.get_ydata(), 0.3 * 10)

    def test_q_one_is_identical_to_not_passing_q(self):
        a = P.expected_cost_curves(50, 5, 10, p=0.6)
        b = P.expected_cost_curves(50, 5, 10, p=0.6, corruption_probability=1.0)
        for la, lb in zip(a.axes[0].get_lines(), b.axes[0].get_lines()):
            assert np.array_equal(la.get_ydata(), lb.get_ydata())
            assert np.array_equal(la.get_xdata(), lb.get_xdata())
        assert a.axes[0].get_ylim() == b.axes[0].get_ylim()

    def test_q_zero_makes_abstention_free(self):
        fig = P.expected_cost_curves(50, 5, 10, corruption_probability=0.0)
        assert np.allclose(fig.axes[0].get_lines()[2].get_ydata(), 0.0)

    def test_label_shows_the_formula(self):
        fig = P.expected_cost_curves(50, 5, 10, corruption_probability=0.25)
        labels = [line.get_label() for line in fig.axes[0].get_lines()[:3]]
        assert any("q × C_abstain" in lbl for lbl in labels)
        assert any("2.5" in lbl for lbl in labels)

    def test_abstain_breakeven_uses_q_times_c_abstain(self):
        """q=0.3 -> E[ABSTAIN]=3 -> the abstain break-even moves to 1-3/50."""
        fig = P.expected_cost_curves(50, 5, 10, corruption_probability=0.3)
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert any("p*=0.94 vs abstain" in t for t in texts), texts
        plain = P.expected_cost_curves(50, 5, 10)
        assert any("p*=0.80 vs abstain" in t.get_text() for t in plain.axes[0].texts)

    @pytest.mark.parametrize("q", [0.0, 1.0, -0.5, 1.5])
    def test_edge_q_does_not_crash(self, q):
        assert isinstance(
            P.expected_cost_curves(10, 2, 5, p=0.5, corruption_probability=q), Figure
        )


class TestReviewAbstainThreshold:
    def test_normal(self):
        assert P.review_abstain_threshold(5, 10) == pytest.approx(0.5)
        assert P.review_abstain_threshold(2, 5) == pytest.approx(0.4)

    def test_zero_abstain_cost_makes_abstention_always_cheaper(self):
        assert P.review_abstain_threshold(5, 0) == float("inf")

    def test_both_zero_resolves_to_review_by_tie_priority(self):
        assert P.review_abstain_threshold(0, 0) == 0.0


class TestDecisionMap:
    def test_returns_figure_with_axes_labelled_p_and_q(self):
        fig = P.decision_map(*BC_COSTS)
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert ax.get_xlabel().endswith("p")
        assert ax.get_ylabel().endswith("q")
        assert ax.get_xlim() == (0, 1) and ax.get_ylim() == (0, 1)
        assert P.COST_UNITS_LABEL in ax.get_title()

    def test_all_three_regions_present_for_breast_cancer_costs(self):
        fig = P.decision_map(*BC_COSTS)
        assert _region_values(fig) == {0, 1, 2}
        labels = [t.get_text() for t in fig.axes[0].texts]
        for action in P.DECISION_MAP_ACTIONS:
            assert action.value in labels

    def test_region_membership_matches_independent_derivation(self):
        """
        Sweep the drawn grid and compare every sampled cell against
        ``_independent_optimal``. Cells lying on a boundary (two expectations
        within 1e-9) are skipped: which side of an exact tie a float lands on
        is not what this test is about.
        """
        fig = P.decision_map(*BC_COSTS)
        n = P.DECISION_MAP_GRID
        checked = 0
        for pi in range(0, n, 3):
            for qi in range(0, n, 5):
                p, q = pi / (n - 1), qi / (n - 1)
                exp = sorted([(1 - p) * 50.0, 5.0, q * 10.0])
                if exp[1] - exp[0] < 1e-9:
                    continue
                expected = _independent_optimal(p, q, *BC_COSTS)
                assert _drawn_action(fig, pi, qi) == expected, (p, q, expected)
                checked += 1
        assert checked > 1000

    @pytest.mark.parametrize("costs", [(50, 5, 10), (25, 2, 5), (10, 2, 5)])
    def test_membership_matches_derivation_for_every_shipped_config(self, costs):
        fig = P.decision_map(*costs)
        n = P.DECISION_MAP_GRID
        for pi in range(0, n, 17):
            for qi in range(0, n, 23):
                p, q = pi / (n - 1), qi / (n - 1)
                exp = sorted([(1 - p) * costs[0], float(costs[1]), q * costs[2]])
                if exp[1] - exp[0] < 1e-9:
                    continue
                assert _drawn_action(fig, pi, qi) == _independent_optimal(p, q, *costs)

    def test_boundary_lines_are_labelled_with_their_formulas(self):
        fig = P.decision_map(*BC_COSTS)
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert any("p* = 1 − C_human/C_wrong = 0.900" in lbl for lbl in labels), labels
        assert any("q* = C_human/C_abstain = 0.500" in lbl for lbl in labels), labels
        assert any("p = 1 − q·C_abstain/C_wrong" in lbl for lbl in labels), labels


class TestDecisionMapDegenerate:
    def test_tiny_abstain_cost_removes_the_review_region(self):
        """q* = 5/0.001 = 5000 >= 1: review is optimal nowhere on the square."""
        fig = P.decision_map(50, 5, 0.001)
        vals = _region_values(fig)
        assert isinstance(fig, Figure)
        assert 1 not in vals and 2 in vals

    def test_huge_abstain_cost_shrinks_abstention_to_q_near_zero(self):
        """q* = 5/1e6 ~ 0: abstention survives only on the q = 0 edge."""
        fig = P.decision_map(50, 5, 1e6)
        grid = np.asarray(fig.axes[0].images[0].get_array())
        assert 1 in _region_values(fig)
        assert (grid == 2).sum() <= grid.shape[1]  # at most the bottom row

    @pytest.mark.parametrize("costs", [
        (50, 0, 10),      # p* = 1, q* = 0
        (50, 60, 10),     # p* < 0
        (50, 5, 0),       # q* = inf, abstention free
        (50, 0, 0),       # exact tie everywhere -> review by priority
        (1, 5, 5),        # every boundary off the square
    ])
    def test_degenerate_configs_do_not_crash(self, costs):
        fig = P.decision_map(*costs)
        assert isinstance(fig, Figure)
        assert _region_values(fig) <= {0, 1, 2}

    def test_degenerate_config_is_annotated_rather_than_hidden(self):
        """q* >= 1 removes HUMAN_REVIEW; q* <= 0 removes ABSTAIN. The figure
        must say which arm vanished instead of silently dropping a colour."""
        fig = P.decision_map(50, 5, 0.001)          # q* = 5000
        notes = " ".join(t.get_text() for t in fig.axes[0].texts)
        assert "HUMAN_REVIEW is optimal nowhere" in notes
        fig = P.decision_map(50, 0, 10)             # q* = 0
        notes = " ".join(t.get_text() for t in fig.axes[0].texts)
        assert "ABSTAIN is optimal nowhere" in notes

    def test_non_positive_c_wrong_rejected(self):
        with pytest.raises(ValueError):
            P.decision_map(0, 5, 10)
        with pytest.raises(ValueError):
            P.decision_map(-1, 5, 10)

    def test_tiny_grid_rejected(self):
        with pytest.raises(ValueError):
            P.decision_map(50, 5, 10, n_grid=1)


class TestDecisionMapMarkers:
    def test_point_renders_and_is_clipped(self):
        plain = P.decision_map(*BC_COSTS)
        with_point = P.decision_map(*BC_COSTS, point=(0.62, 0.99))
        assert len(with_point.axes[0].get_lines()) > len(plain.axes[0].get_lines())
        labels = [t.get_text() for t in with_point.axes[0].get_legend().get_texts()]
        assert any("this cell  p=0.620, q=0.990" in lbl for lbl in labels)
        out_of_range = P.decision_map(*BC_COSTS, point=(1.5, -0.2))
        labels = [t.get_text() for t in out_of_range.axes[0].get_legend().get_texts()]
        assert any("p=1.000, q=0.000" in lbl for lbl in labels)

    def test_population_scatter_groups_by_action(self):
        pts = [
            (0.95, 0.99, DecisionAction.AUTO_REPAIR),
            (0.30, 0.99, DecisionAction.HUMAN_REVIEW),
            (0.30, 0.03, "ABSTAIN"),            # string form accepted too
            (0.40, 0.03, DecisionAction.ABSTAIN),
        ]
        fig = P.decision_map(*BC_COSTS, points=pts)
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert any("AUTO_REPAIR cells (1)" in lbl for lbl in labels)
        assert any("ABSTAIN cells (2)" in lbl for lbl in labels)
        assert len(fig.axes[0].collections) >= 3

    def test_point_and_points_combine(self):
        fig = P.decision_map(*BC_COSTS, point=(0.5, 0.5),
                             points=[(0.1, 0.1, DecisionAction.ABSTAIN)])
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert any("this cell" in lbl for lbl in labels)
        assert any("ABSTAIN cells (1)" in lbl for lbl in labels)

    def test_empty_points_is_fine(self):
        assert isinstance(P.decision_map(*BC_COSTS, points=[]), Figure)
