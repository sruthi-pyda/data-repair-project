"""
src/calibration/metrics.py
---------------------------
Calibration quality metrics: Brier score and Expected Calibration Error (ECE).

WHAT CALIBRATION METRICS MEASURE
==================================

A well-calibrated system should satisfy:
    P(correct | predicted_prob ≈ p) ≈ p

Meaning: among all predictions where the system says "70% confident",
roughly 70% should actually be correct.

Brier Score
-----------
Measures overall prediction quality (accuracy + calibration combined):

    B = mean((p_i - y_i)²)

where p_i ∈ [0,1] and y_i ∈ {0,1}.

- Range: [0, 1]. Lower is better.
- B = 0: perfect predictions.
- B = 0.25: equivalent to always predicting 0.5 (the worst "uninformed" score).
- Sensitive to both accuracy (how often correct) and calibration (how confident).

Expected Calibration Error (ECE)
---------------------------------
Measures calibration specifically — does predicted confidence match empirical accuracy?

Algorithm:
1. Bin predictions by confidence into n_bins equal-width intervals.
2. For each non-empty bin:
       acc_bin = mean(y_i) for i in bin
       conf_bin = mean(p_i) for i in bin
3. ECE = Σ_bins (n_bin / n_total) × |acc_bin − conf_bin|

- Range: [0, 1]. Lower is better.
- ECE = 0: perfectly calibrated.
- ECE > 0.05 is generally considered poor calibration.

Interpreting a reliability diagram
------------------------------------
Plot predicted confidence (x-axis) against empirical accuracy (y-axis).
A perfectly calibrated system lies on the diagonal y = x.
- Points above diagonal: underconfident (model says 0.4 but accuracy is 0.6)
- Points below diagonal: overconfident (model says 0.8 but accuracy is 0.5)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ECEBin:
    """Statistics for one confidence bin in the ECE computation."""

    bin_idx: int
    bin_lower: float
    bin_upper: float
    n_samples: int
    mean_confidence: float
    empirical_accuracy: float
    calibration_gap: float       # |acc - conf|
    contribution: float          # (n / total) × |acc - conf|

    def __repr__(self) -> str:
        return (
            f"ECEBin([{self.bin_lower:.2f}, {self.bin_upper:.2f}]  "
            f"n={self.n_samples}  conf={self.mean_confidence:.3f}  "
            f"acc={self.empirical_accuracy:.3f}  gap={self.calibration_gap:.3f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Brier score
# ═══════════════════════════════════════════════════════════════════════════════


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the Brier score.

    B = mean((p_i - y_i)²)

    Parameters
    ----------
    probs : np.ndarray, shape (n,)
        Predicted probabilities in [0, 1].
    labels : np.ndarray, shape (n,)
        Binary ground-truth labels in {0, 1}.

    Returns
    -------
    float
        Brier score in [0, 1]. Lower is better.

    Raises
    ------
    ValueError
        If inputs are empty or have different lengths.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)

    if len(probs) == 0:
        raise ValueError("probs is empty.")
    if len(probs) != len(labels):
        raise ValueError(
            f"probs and labels must have the same length. "
            f"Got {len(probs)} and {len(labels)}."
        )
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError("All probs must be in [0, 1].")

    return float(np.mean((probs - labels) ** 2))


# ═══════════════════════════════════════════════════════════════════════════════
# Expected Calibration Error (ECE)
# ═══════════════════════════════════════════════════════════════════════════════


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, List[ECEBin]]:
    """
    Compute Expected Calibration Error and per-bin statistics.

    Uses equal-width bins over [0, 1].

    Parameters
    ----------
    probs : np.ndarray, shape (n,)
        Predicted probabilities in [0, 1].
    labels : np.ndarray, shape (n,)
        Binary ground-truth labels in {0, 1}.
    n_bins : int
        Number of equal-width bins. Default 10 (each bin spans 0.1).

    Returns
    -------
    (ece, bins)
        ece : float — ECE in [0, 1]. Lower is better.
        bins : list[ECEBin] — per-bin statistics (only non-empty bins).
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)

    if len(probs) == 0:
        return 0.0, []
    if len(probs) != len(labels):
        raise ValueError("probs and labels must have equal length.")

    n_total = len(probs)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins_out: List[ECEBin] = []
    ece = 0.0

    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]

        # Include the right edge in the last bin
        if i < n_bins - 1:
            mask = (probs >= lo) & (probs < hi)
        else:
            mask = (probs >= lo) & (probs <= hi)

        n_bin = int(mask.sum())
        if n_bin == 0:
            continue

        conf_bin = float(probs[mask].mean())
        acc_bin = float(labels[mask].mean())
        gap = abs(acc_bin - conf_bin)
        contribution = (n_bin / n_total) * gap
        ece += contribution

        bins_out.append(ECEBin(
            bin_idx=i,
            bin_lower=lo,
            bin_upper=hi,
            n_samples=n_bin,
            mean_confidence=round(conf_bin, 4),
            empirical_accuracy=round(acc_bin, 4),
            calibration_gap=round(gap, 4),
            contribution=round(contribution, 6),
        ))

    return round(ece, 6), bins_out
