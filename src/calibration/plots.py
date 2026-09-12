"""
src/calibration/plots.py
-------------------------
Reliability diagram generation for calibration evaluation.

HOW TO INTERPRET A RELIABILITY DIAGRAM
========================================
A reliability diagram (also called a calibration curve) plots:

  x-axis: Mean predicted confidence in each bin
  y-axis: Empirical accuracy (fraction of correct predictions) in that bin

A perfectly calibrated system lies on the diagonal y = x:
  - If the model says "70% confident", ~70% should be correct.

Points ABOVE the diagonal:
  - Model is UNDERCONFIDENT (actual accuracy exceeds stated confidence)

Points BELOW the diagonal:
  - Model is OVERCONFIDENT (actual accuracy is lower than stated confidence)

The shaded gap between the curve and the diagonal represents the calibration
error — smaller gap = better calibrated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.calibration.calibrator import CalibrationComparison
from src.calibration.metrics import ECEBin


# Colour scheme for the three methods
_COLOURS = {
    "raw": "#e74c3c",       # red
    "logistic": "#2980b9",  # blue
    "isotonic": "#27ae60",  # green
}

_DIAGONAL_COLOUR = "#7f8c8d"  # grey


def plot_reliability_diagram(
    comparison: CalibrationComparison,
    output_dir: Path,
    n_diagonal_points: int = 100,
) -> Path:
    """
    Create and save a reliability diagram for one dataset.

    Plots raw, logistic, and isotonic calibration curves against the
    perfect-calibration diagonal.

    Parameters
    ----------
    comparison : CalibrationComparison
        Output of calibrate_dataset().
    output_dir : Path
        Directory where the figure is saved.
    n_diagonal_points : int
        Number of points for the diagonal reference line.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Perfect calibration diagonal
    diag = np.linspace(0, 1, n_diagonal_points)
    ax.plot(diag, diag, linestyle="--", color=_DIAGONAL_COLOUR,
            linewidth=1.5, label="Perfect calibration", zorder=1)
    ax.fill_between(diag, diag, alpha=0.05, color=_DIAGONAL_COLOUR)

    # Plot each method's reliability curve
    _plot_bins(ax, comparison.raw_bins, "raw", "Raw confidence")
    if comparison.logistic_fitted:
        _plot_bins(ax, comparison.logistic_bins, "logistic", "Logistic calibration")
    if comparison.isotonic_fitted:
        _plot_bins(ax, comparison.isotonic_bins, "isotonic", "Isotonic calibration")

    # Annotations
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence", fontsize=12)
    ax.set_ylabel("Empirical accuracy", fontsize=12)
    ax.set_title(
        f"Reliability Diagram — {comparison.dataset_name.replace('_', ' ').title()}\n"
        f"n_eval={comparison.n_eval}  base_rate={comparison.base_rate:.3f}",
        fontsize=12,
    )

    # Metric table in legend
    legend_lines = [
        mpatches.Patch(color=_DIAGONAL_COLOUR, alpha=0.5, label="Perfect calibration"),
        mpatches.Patch(
            color=_COLOURS["raw"],
            label=f"Raw  Brier={comparison.raw_brier:.4f}  ECE={comparison.raw_ece:.4f}",
        ),
    ]
    if comparison.logistic_fitted:
        legend_lines.append(mpatches.Patch(
            color=_COLOURS["logistic"],
            label=f"Logistic  Brier={comparison.logistic_brier:.4f}  "
                  f"ECE={comparison.logistic_ece:.4f}",
        ))
    if comparison.isotonic_fitted:
        legend_lines.append(mpatches.Patch(
            color=_COLOURS["isotonic"],
            label=f"Isotonic  Brier={comparison.isotonic_brier:.4f}  "
                  f"ECE={comparison.isotonic_ece:.4f}",
        ))

    ax.legend(handles=legend_lines, loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    out_path = output_dir / f"reliability_{comparison.dataset_name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return out_path


def plot_brier_comparison(
    comparisons: List[CalibrationComparison],
    output_dir: Path,
) -> Path:
    """
    Bar chart comparing Brier scores across datasets and methods.

    Parameters
    ----------
    comparisons : list[CalibrationComparison]
    output_dir : Path

    Returns
    -------
    Path to saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid = [c for c in comparisons if c.n_eval > 0]
    if not valid:
        return output_dir / "brier_comparison.png"

    datasets = [c.dataset_name for c in valid]
    n_ds = len(datasets)
    x = np.arange(n_ds)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, 3 * n_ds), 5))

    raw_scores = [c.raw_brier for c in valid]
    log_scores = [c.logistic_brier for c in valid]
    iso_scores = [c.isotonic_brier for c in valid]

    bars_raw = ax.bar(x - width, raw_scores, width, label="Raw",
                      color=_COLOURS["raw"], alpha=0.8)
    bars_log = ax.bar(x, log_scores, width, label="Logistic",
                      color=_COLOURS["logistic"], alpha=0.8)
    bars_iso = ax.bar(x + width, iso_scores, width, label="Isotonic",
                      color=_COLOURS["isotonic"], alpha=0.8)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Brier Score (lower = better)")
    ax.set_title("Brier Score Comparison by Dataset and Calibration Method")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in datasets])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(raw_scores), max(log_scores), max(iso_scores)) * 1.2 + 0.01)

    # Value labels on bars
    for bars in [bars_raw, bars_log, bars_iso]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.003,
                f"{h:.4f}",
                ha="center", va="bottom", fontsize=7,
            )

    out_path = output_dir / "brier_comparison.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return out_path


# ─── Internal helpers ──────────────────────────────────────────────────────────


def _plot_bins(
    ax: plt.Axes,
    bins: List[ECEBin],
    method: str,
    label: str,
) -> None:
    """Plot reliability curve from ECEBin list on the given axis."""
    if not bins:
        return
    xs = [b.mean_confidence for b in bins]
    ys = [b.empirical_accuracy for b in bins]
    sizes = [max(20, b.n_samples) for b in bins]

    colour = _COLOURS.get(method, "#333333")
    ax.plot(xs, ys, "o-", color=colour, linewidth=2, markersize=6,
            label=label, zorder=3)
    # Shade calibration gap
    ax.fill_between(xs, xs, ys, alpha=0.08, color=colour)
