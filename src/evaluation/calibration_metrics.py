"""
src/evaluation/calibration_metrics.py
---------------------------------------
Phase 10: Calibration quality evaluation.

WHAT IS EVALUATED
==================
Two confidence streams are compared:
  1. Raw confidence     — top candidate probability assigned by the repairer
  2. Calibrated prob.   — output of Phase 6 calibration (logistic / isotonic)

For each stream we compute:
  - Brier Score
  - Expected Calibration Error (ECE)
  - Negative Log-Likelihood (NLL)

Lower values are better for all three metrics.

METRIC FORMULAE
================

Brier Score
-----------
  B = mean((p_i - y_i)²)

where p_i is the predicted probability and y_i ∈ {0, 1} is the actual
correctness label.  Range [0, 1].

ECE (Expected Calibration Error)
---------------------------------
Bins predictions into n_bins equal-width intervals over [0, 1].
For each non-empty bin b:
  ECE_b = (n_b / N) × |acc_b − conf_b|

ECE = Σ_b ECE_b

Range [0, 1].  See src/calibration/metrics.py for the implementation.

NLL (Negative Log-Likelihood)
------------------------------
  NLL = −mean( y × log(p) + (1−y) × log(1−p) )

Probabilities are clipped to [ε, 1−ε] (default ε = 1e-7) before
taking logarithms to prevent log(0).

Lower NLL means predictions are both confident and correct.

HOW TO INTERPRET RESULTS
==========================
  - B = 0 and NLL = 0: perfect predictions (all p_i = y_i).
  - B = 0.25 and NLL = log(2) ≈ 0.693: equivalent to always predicting 0.5.
  - ECE = 0: predictions are perfectly calibrated.
  - Calibrated confidence should have lower ECE than raw confidence
    IF the calibration was effective.  Do NOT assume this; report honestly.

DATA-LEAKAGE BOUNDARY
======================
CalibrationEvaluator NEVER accesses clean_df directly.  It works only
with (probs, correctness_labels) arrays, where the correctness labels
are already derived from clean_df by an upstream evaluation step.

Passing correctness_labels is NOT passing clean_df.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# Reuse Phase 6 implementations to avoid code duplication
from src.calibration.metrics import (
    ECEBin,
    brier_score,
    expected_calibration_error,
)

logger = logging.getLogger(__name__)

# Clipping epsilon used before log() in NLL to prevent log(0)
_NLL_CLIP_EPS: float = 1e-7


# ═══════════════════════════════════════════════════════════════════════════════
# NLL
# ═══════════════════════════════════════════════════════════════════════════════


def nll_score(
    probs: np.ndarray,
    labels: np.ndarray,
    clip_eps: float = _NLL_CLIP_EPS,
) -> float:
    """
    Compute Negative Log-Likelihood for binary correctness predictions.

    Formula
    -------
      NLL = −mean( y × log(p) + (1−y) × log(1−p) )

    Probabilities are clipped to [clip_eps, 1−clip_eps] before log().
    Default clip_eps = 1e-7 ensures log(0) never occurs.

    Parameters
    ----------
    probs : array-like, shape (n,)
        Predicted probabilities of correctness, in [0, 1].
    labels : array-like, shape (n,)
        Binary correctness labels in {0, 1}.
    clip_eps : float
        Clipping epsilon.  Default 1e-7.

    Returns
    -------
    float
        NLL ≥ 0.  Lower is better.
        Returns 0.0 for an empty input.

    Raises
    ------
    ValueError
        If probs and labels have different lengths, or if labels
        contain values outside {0, 1}.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)

    if len(probs) == 0:
        return 0.0
    if len(probs) != len(labels):
        raise ValueError(
            f"probs and labels must have the same length. "
            f"Got {len(probs)} and {len(labels)}."
        )
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must be binary (0 or 1).")

    if clip_eps < 0 or clip_eps >= 0.5:
        raise ValueError(
            f"clip_eps must be in [0, 0.5), got {clip_eps}"
        )

    p_clipped = np.clip(probs, clip_eps, 1.0 - clip_eps)
    nll = -float(np.mean(
        labels * np.log(p_clipped) + (1.0 - labels) * np.log(1.0 - p_clipped)
    ))
    return nll


# ═══════════════════════════════════════════════════════════════════════════════
# CalibrationReport
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CalibrationReport:
    """
    Calibration quality metrics for one confidence stream.

    Attributes
    ----------
    label : str
        Human-readable identifier, e.g. "raw" or "calibrated".
    dataset : str
    n_samples : int
    n_positive : int
        Count of correct labels (y=1).
    brier : float
    ece : float
    nll : float
    ece_bins : list[ECEBin]
        Per-bin statistics (from Phase 6 ECE implementation).
    """

    label: str
    dataset: str
    n_samples: int
    n_positive: int
    brier: float
    ece: float
    nll: float
    ece_bins: List[ECEBin] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "dataset": self.dataset,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "brier": self.brier,
            "ece": self.ece,
            "nll": self.nll,
        }


@dataclass
class CalibrationComparison:
    """
    Side-by-side comparison of raw vs calibrated confidence streams.

    Attributes
    ----------
    dataset : str
    raw : CalibrationReport
    calibrated : CalibrationReport
    calibration_improves_brier : bool
        True iff calibrated.brier < raw.brier.
    calibration_improves_ece : bool
        True iff calibrated.ece < raw.ece.
    calibration_improves_nll : bool
        True iff calibrated.nll < raw.nll.
    note : str
        Optional limitation / dataset-specific caveat.
    """

    dataset: str
    raw: CalibrationReport
    calibrated: CalibrationReport
    calibration_improves_brier: bool
    calibration_improves_ece: bool
    calibration_improves_nll: bool
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            **{f"raw_{k}": v for k, v in self.raw.to_dict().items()
               if k not in ("label", "dataset")},
            **{f"cal_{k}": v for k, v in self.calibrated.to_dict().items()
               if k not in ("label", "dataset")},
            "calibration_improves_brier": self.calibration_improves_brier,
            "calibration_improves_ece": self.calibration_improves_ece,
            "calibration_improves_nll": self.calibration_improves_nll,
            "note": self.note,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CalibrationEvaluator
# ═══════════════════════════════════════════════════════════════════════════════


class CalibrationEvaluator:
    """
    Compute calibration metrics for a set of predictions.

    Does NOT access clean_df.  Takes (probs, labels) arrays directly.

    The correctness labels must have been derived from clean_df by an
    upstream evaluation step (e.g., CalibrationBuilder from Phase 6).

    Parameters
    ----------
    n_bins : int
        Number of ECE bins.  Default 10.
    nll_clip_eps : float
        Clipping epsilon for NLL.  Default 1e-7.
    """

    def __init__(
        self,
        n_bins: int = 10,
        nll_clip_eps: float = _NLL_CLIP_EPS,
    ) -> None:
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
        self.n_bins = n_bins
        self.nll_clip_eps = nll_clip_eps

    def evaluate(
        self,
        probs: List[float],
        labels: List[int],
        label: str = "predictions",
        dataset: str = "unknown",
    ) -> CalibrationReport:
        """
        Compute Brier, ECE, and NLL for one confidence stream.

        Parameters
        ----------
        probs : list[float]
            Predicted probabilities of correctness in [0, 1].
        labels : list[int]
            Binary correctness labels in {0, 1}.
        label : str
            Identifier for this stream, e.g. "raw" or "calibrated".
        dataset : str
            Dataset name.

        Returns
        -------
        CalibrationReport

        Notes
        -----
        Returns a zero-filled report for empty inputs rather than raising.
        """
        probs_arr = np.asarray(probs, dtype=float)
        labels_arr = np.asarray(labels, dtype=float)

        if len(probs_arr) == 0:
            return CalibrationReport(
                label=label, dataset=dataset,
                n_samples=0, n_positive=0,
                brier=0.0, ece=0.0, nll=0.0,
            )

        n_pos = int(labels_arr.sum())
        b = brier_score(probs_arr, labels_arr)
        ece, bins = expected_calibration_error(probs_arr, labels_arr, self.n_bins)
        nll = nll_score(probs_arr, labels_arr, self.nll_clip_eps)

        return CalibrationReport(
            label=label,
            dataset=dataset,
            n_samples=len(probs_arr),
            n_positive=n_pos,
            brier=round(b, 6),
            ece=round(ece, 6),
            nll=round(nll, 6),
            ece_bins=bins,
        )

    def compare(
        self,
        raw_probs: List[float],
        cal_probs: List[float],
        labels: List[int],
        dataset: str = "unknown",
        note: str = "",
    ) -> CalibrationComparison:
        """
        Compare raw vs calibrated confidence streams.

        Parameters
        ----------
        raw_probs : list[float]
        cal_probs : list[float]
        labels : list[int]
        dataset : str
        note : str
            Optional dataset-level caveat (e.g. limitation note).

        Returns
        -------
        CalibrationComparison

        Raises
        ------
        ValueError
            If raw_probs, cal_probs, and labels have different lengths.
        """
        if not (len(raw_probs) == len(cal_probs) == len(labels)):
            raise ValueError(
                f"raw_probs, cal_probs, labels must all have the same length. "
                f"Got {len(raw_probs)}, {len(cal_probs)}, {len(labels)}."
            )
        raw_report = self.evaluate(raw_probs, labels, label="raw", dataset=dataset)
        cal_report = self.evaluate(cal_probs, labels, label="calibrated", dataset=dataset)

        return CalibrationComparison(
            dataset=dataset,
            raw=raw_report,
            calibrated=cal_report,
            calibration_improves_brier=cal_report.brier < raw_report.brier,
            calibration_improves_ece=cal_report.ece < raw_report.ece,
            calibration_improves_nll=cal_report.nll < raw_report.nll,
            note=note,
        )
