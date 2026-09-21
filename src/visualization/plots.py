"""
src/visualization/plots.py
--------------------------
Phase 11: figure-returning plot helpers for the Streamlit dashboard and the
Phase 13 experiment runner.

Every public function returns a ``matplotlib.figure.Figure`` and never
writes to disk, so callers decide whether to ``st.pyplot(fig)`` or
``fig.savefig(...)``. Callers own the figure and should ``plt.close(fig)``
when done (Streamlit does this itself).

The Phase 6 reliability/Brier plots in ``src/calibration/plots.py`` save
straight to PNG for the demo scripts; ``reliability_figure`` here reuses
their bin-drawing helper so the two stay visually consistent.

LEAKAGE RULE
============
Nothing here reads ``clean_df``. Functions that draw evaluation results
(``action_distribution``, ``cost_comparison``, ``threshold_sweep``) take
already-scored rows produced by ``src/evaluation``; the ground truth was
consumed upstream, after decisions were made.

COST UNITS
==========
Every cost axis is labelled "experimental cost units". They are abstract
penalty weights from ``configs/costs.yaml``, not money.

Worked example
==============
    >>> fig = expected_cost_curves(c_wrong=50, c_human=5, c_abstain=10, p=0.7)
    >>> fig.axes[0].get_xlabel()
    'Calibrated probability of correctness  p'
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless; Streamlit renders the figure itself

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src.calibration.calibrator import CalibrationComparison  # noqa: E402
from src.calibration.plots import _COLOURS as _CAL_COLOURS  # noqa: E402
from src.calibration.plots import _plot_bins  # noqa: E402
from src.decision.policy import DecisionAction  # noqa: E402
from src.evaluation.cost_metrics import CostReport, PolicyComparisonRow  # noqa: E402

COST_UNITS_LABEL = "experimental cost units"

ACTION_COLOURS: Dict[str, str] = {
    DecisionAction.AUTO_REPAIR.value: "#27ae60",   # green
    DecisionAction.HUMAN_REVIEW.value: "#f39c12",  # amber
    DecisionAction.ABSTAIN.value: "#7f8c8d",       # grey
}

METHOD_LABELS: Dict[str, str] = {
    "calibrated_risk_aware": "Proposed: calibrated + cost-aware",
    "always_auto": "Baseline: always auto-repair",
}


def _method_label(name: str) -> str:
    if name in METHOD_LABELS:
        return METHOD_LABELS[name]
    if name.startswith("confidence_threshold_"):
        return f"Baseline: raw confidence ≥ {name.rsplit('_', 1)[-1]}"
    return name


def _new_fig(width: float = 7.0, height: float = 4.0) -> Tuple[Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height))
    ax.grid(True, alpha=0.3)
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: candidate table
# ═══════════════════════════════════════════════════════════════════════════════


def candidate_bar(
    candidates: Sequence[Tuple[Any, float]],
    title: str = "Repair candidates",
) -> Figure:
    """Horizontal bar of candidate probabilities, top candidate first."""
    fig, ax = _new_fig(6.5, 0.6 * max(3, len(candidates)) + 1.0)
    if not candidates:
        ax.text(0.5, 0.5, "No candidates", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    labels = [str(v) for v, _ in candidates]
    probs = [float(p) for _, p in candidates]
    y = np.arange(len(candidates))[::-1]
    colours = ["#2980b9"] + ["#95a5a6"] * (len(candidates) - 1)
    ax.barh(y, probs, color=colours, alpha=0.85)
    for yi, p in zip(y, probs):
        ax.text(p + 0.01, yi, f"{p:.3f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Probability assigned by the repairer")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: the interactive risk demonstration
# ═══════════════════════════════════════════════════════════════════════════════


def expected_cost_curves(
    c_wrong: float,
    c_human: float,
    c_abstain: float,
    p: Optional[float] = None,
    n_points: int = 201,
) -> Figure:
    """
    Expected cost of each action as a function of calibrated probability.

        E[AUTO_REPAIR]  = (1 - p) · c_wrong      (falls with p)
        E[HUMAN_REVIEW] = c_human                (flat)
        E[ABSTAIN]      = c_abstain              (flat)

    The lower envelope is shaded by the winning action so that moving a cost
    slider visibly moves the break-even points. A vertical marker shows the
    current record's ``p`` when given.
    """
    if c_wrong <= 0:
        raise ValueError(f"c_wrong must be positive, got {c_wrong}")
    ps = np.linspace(0.0, 1.0, n_points)
    e_auto = (1.0 - ps) * c_wrong
    e_human = np.full_like(ps, c_human)
    e_abstain = np.full_like(ps, c_abstain)
    stacked = np.vstack([e_auto, e_human, e_abstain])
    winner = stacked.argmin(axis=0)  # ties → lowest index → AUTO > HUMAN > ABSTAIN
    envelope = stacked.min(axis=0)
    actions = [DecisionAction.AUTO_REPAIR, DecisionAction.HUMAN_REVIEW, DecisionAction.ABSTAIN]

    fig, ax = _new_fig(8.0, 4.5)
    ax.plot(ps, e_auto, color=ACTION_COLOURS["AUTO_REPAIR"], lw=2,
            label=f"E[AUTO_REPAIR] = (1−p)·{c_wrong:g}")
    ax.plot(ps, e_human, color=ACTION_COLOURS["HUMAN_REVIEW"], lw=2,
            label=f"E[HUMAN_REVIEW] = {c_human:g}")
    ax.plot(ps, e_abstain, color=ACTION_COLOURS["ABSTAIN"], lw=2, ls="--",
            label=f"E[ABSTAIN] = {c_abstain:g}")

    # Shade the winning region under the envelope.
    for k, action in enumerate(actions):
        mask = winner == k
        if mask.any():
            ax.fill_between(ps, 0, envelope, where=mask,
                            color=ACTION_COLOURS[action.value], alpha=0.12,
                            interpolate=False)

    y_max = max(c_wrong, c_human, c_abstain) * 1.15

    # Break-even points (only those inside (0, 1)), labelled vertically at
    # mid-height so they never collide with the legend in the upper right.
    for other, name, frac in ((c_human, "review", 0.62), (c_abstain, "abstain", 0.50)):
        bp = 1.0 - other / c_wrong
        if 0.0 < bp < 1.0:
            ax.axvline(bp, color="#34495e", lw=0.8, ls=":", alpha=0.7)
            ax.text(bp, y_max * frac, f" p*={bp:.2f} vs {name}", fontsize=8,
                    color="#34495e", va="center", ha="left", rotation=90)

    if p is not None:
        p = float(np.clip(p, 0.0, 1.0))
        ax.axvline(p, color="#c0392b", lw=2, alpha=0.9)
        ax.text(p, y_max * 0.97, f" this cell: p={p:.3f}", color="#c0392b",
                fontsize=9, va="top", ha="left" if p < 0.7 else "right")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Calibrated probability of correctness  p")
    ax.set_ylabel(f"Expected cost ({COST_UNITS_LABEL})")
    ax.set_title("Which action is cheapest at each confidence level?")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
              fontsize=9, frameon=False)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4/9: uncertainty and calibration
# ═══════════════════════════════════════════════════════════════════════════════


def confidence_histogram(
    raw: Sequence[float],
    calibrated: Sequence[float],
    calibration_method: str = "calibrated",
    bins: int = 20,
) -> Figure:
    """Overlaid histograms of raw vs calibrated confidence."""
    fig, ax = _new_fig(7.0, 3.8)
    edges = np.linspace(0.0, 1.0, bins + 1)
    if len(raw):
        ax.hist(raw, bins=edges, alpha=0.55, color=_CAL_COLOURS["raw"], label="raw confidence")
    if len(calibrated):
        ax.hist(calibrated, bins=edges, alpha=0.55, color=_CAL_COLOURS["logistic"],
                label=f"calibrated ({calibration_method})")
    if not len(raw) and not len(calibrated):
        ax.text(0.5, 0.5, "No records", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability assigned to top candidate")
    ax.set_ylabel("Number of cells")
    ax.set_title("Confidence distribution: raw vs calibrated")
    if len(raw) or len(calibrated):
        ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def reliability_figure(comparison: CalibrationComparison) -> Figure:
    """
    Reliability diagram for one dataset; mirrors the Phase 6 PNG but returns
    the figure. Not-fitted methods are omitted rather than drawn from copied
    raw metrics.
    """
    fig, ax = _new_fig(5.5, 5.5)
    diag = np.linspace(0, 1, 50)
    ax.plot(diag, diag, "--", color="#7f8c8d", lw=1.2, label="perfect calibration")
    _plot_bins(ax, comparison.raw_bins, "raw",
               f"raw  Brier={comparison.raw_brier:.3f} ECE={comparison.raw_ece:.3f}")
    for method, bins_attr in (
        ("logistic", "logistic_bins"),
        ("isotonic", "isotonic_bins"),
        ("multivariate", "multivariate_bins"),
    ):
        if comparison.fitted(method):
            _plot_bins(
                ax, getattr(comparison, bins_attr), method,
                f"{method}  Brier={comparison.metric_str(method, 'brier', 3)} "
                f"ECE={comparison.metric_str(method, 'ece', 3)}",
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"Reliability — {comparison.dataset_name}  (n_eval={comparison.n_eval})")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8/9: policy comparison (already-scored rows)
# ═══════════════════════════════════════════════════════════════════════════════


def action_distribution(rows: Sequence[PolicyComparisonRow]) -> Figure:
    """Stacked horizontal bars: share of AUTO / HUMAN / ABSTAIN per method."""
    fig, ax = _new_fig(8.0, 0.7 * max(2, len(rows)) + 1.2)
    if not rows:
        ax.text(0.5, 0.5, "No methods evaluated", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for key, action in (
        ("auto_rate", DecisionAction.AUTO_REPAIR),
        ("human_rate", DecisionAction.HUMAN_REVIEW),
        ("abstain_rate", DecisionAction.ABSTAIN),
    ):
        vals = np.array([getattr(r, key) for r in rows], dtype=float)
        ax.barh(y, vals, left=left, color=ACTION_COLOURS[action.value],
                alpha=0.85, label=action.value)
        for yi, l, v in zip(y, left, vals):
            if v >= 0.06:
                ax.text(l + v / 2, yi, f"{v:.0%}", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([_method_label(r.method_name) for r in rows])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of detected cells")
    ax.set_title("What each policy does with the same cells")
    ax.legend(loc="lower right", fontsize=8, ncol=3)
    fig.tight_layout()
    return fig


def cost_comparison(rows: Sequence[PolicyComparisonRow]) -> Figure:
    """Total realised cost per method; the proposed method is highlighted."""
    fig, ax = _new_fig(8.0, 4.0)
    if not rows:
        ax.text(0.5, 0.5, "No methods evaluated", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    names = [_method_label(r.method_name) for r in rows]
    costs = [r.total_realized_cost for r in rows]
    colours = ["#8e44ad" if r.method_name == "calibrated_risk_aware" else "#95a5a6" for r in rows]
    x = np.arange(len(rows))
    bars = ax.bar(x, costs, color=colours, alpha=0.9)
    for b, c, r in zip(bars, costs, rows):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c:,.0f}\n(mean {r.mean_cost_per_record:.2f})",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=10)
    ax.set_ylabel(f"Total realised cost ({COST_UNITS_LABEL})")
    ax.set_ylim(0, max(costs) * 1.25 if max(costs) > 0 else 1)
    ax.set_title("Realised cost against ground truth (lower is better)")
    fig.tight_layout()
    return fig


def threshold_sweep(
    sweep: Dict[float, CostReport],
    proposed_cost: Optional[float] = None,
    current_threshold: Optional[float] = None,
) -> Figure:
    """
    Realised cost of the raw-confidence baseline across thresholds, with the
    proposed method's cost as a horizontal reference line.
    """
    fig, ax = _new_fig(7.0, 3.8)
    if not sweep:
        ax.text(0.5, 0.5, "No sweep results", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    ts = sorted(sweep)
    cs = [sweep[t].total_realized_cost for t in ts]
    ax.plot(ts, cs, "o-", color="#95a5a6", lw=2, label="raw-confidence threshold baseline")
    if current_threshold is not None and current_threshold in sweep:
        ax.plot([current_threshold], [sweep[current_threshold].total_realized_cost], "o",
                color="#e67e22", markersize=11, label=f"current threshold {current_threshold:.2f}")
    if proposed_cost is not None:
        ax.axhline(proposed_cost, color="#8e44ad", lw=2, label="proposed method")
    ax.set_xlabel("Confidence threshold for auto-repair")
    ax.set_ylabel(f"Total realised cost ({COST_UNITS_LABEL})")
    ax.set_title("Can a tuned threshold match the calibrated policy?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def decision_scatter(
    calibrated_probs: Sequence[float],
    actions: Sequence[DecisionAction],
    c_wrong: float,
    c_human: float,
    c_abstain: float,
) -> Figure:
    """
    Every decided cell as a point on the expected-cost curves, coloured by
    the action taken, so the split of the population is visible at once.
    """
    fig = expected_cost_curves(c_wrong, c_human, c_abstain, p=None)
    ax = fig.axes[0]
    probs = np.asarray(calibrated_probs, dtype=float)
    if len(probs):
        rng = np.random.default_rng(0)
        jitter = rng.uniform(-0.02, 0.02, size=len(probs)) * max(c_wrong, c_human, c_abstain)
        for action in DecisionAction:
            mask = np.array([a == action for a in actions], dtype=bool)
            if mask.any():
                ax.scatter(probs[mask], (1 - probs[mask]) * c_wrong + jitter[mask],
                           s=14, color=ACTION_COLOURS[action.value], alpha=0.6,
                           edgecolor="none", label=f"{action.value} ({int(mask.sum())})")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
                  fontsize=8, frameon=False)
    ax.set_title("Every detected cell placed on the cost curve")
    return fig
