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
