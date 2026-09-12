"""
src/calibration/calibrator.py
------------------------------
Phase 6: Probability calibration for repair predictions.

WHAT IS CALIBRATION?
=====================
Raw confidence from a repair system is not necessarily a true probability.
A model that says "0.90 confident" may only be correct 60% of the time.

Calibration learns a mapping:
    raw_confidence  →  calibrated_probability

such that calibrated_probability ≈ P(prediction is correct).

After calibration:
    calibrated_probability ≈ 0.9  →  ~90% of such predictions are correct

WHY THIS MATTERS
=================
The decision policy (Phase 8 — AUTO / HUMAN_REVIEW / ABSTAIN) needs
reliable probabilities to make cost-sensitive decisions. If the system says
"90% confident" but only gets it right 60% of the time, the policy will
make too many automatic repairs and incur unnecessary errors.

CALIBRATION METHODS
====================

Method A: Logistic calibration (Platt scaling)
------------------------------------------------
Learns a sigmoid transformation:
    p_cal = sigmoid(a × p_raw + b) = 1 / (1 + exp(-(a × p_raw + b)))

Assumes the relationship between raw confidence and true probability is
smooth and monotone. Fit by minimising binary cross-entropy over the
calibration set. Two free parameters: scale (a) and bias (b).

Method B: Isotonic regression
------------------------------
Non-parametric monotone mapping. Finds the best step-function
f: [0,1] → [0,1] such that f is non-decreasing and minimises MSE.
No assumed functional form — can capture arbitrary monotone shapes.
Needs more data than logistic calibration to be reliable.

METHOD COMPARISON
==================
| Method     | Assumptions          | Data needed | Risk of over-fit |
|------------|----------------------|-------------|-----------------|
| Raw        | None (baseline)      | None        | None            |
| Logistic   | Smooth sigmoid shape | Low (~20+)  | Low             |
| Isotonic   | Only monotonicity    | Moderate    | Moderate        |

DATA LEAKAGE PREVENTION
========================
clean_df is used ONLY to create 0/1 correctness labels in CalibrationBuilder.
The calibrator's input feature is ONLY raw_confidence — NOT the actual
clean value. The calibrator never learns "what the correct value is".

Leakage rule: CalibrationBuilder.build(unc_records, clean_df, ...) is the
ONLY function that reads clean_df in the calibration phase. The fitted
calibrators (LogisticCalibrator, IsotonicCalibrator) see only (X=conf, y=0/1).

ADULT DATASET LIMITATION
=========================
Phase 3 detected 5,829 cells in Adult, but all are statistical outliers in
the `fnlwgt` column — NOT the true income label flips. The correctness
labels will all be 0 (median imputation never corrects a valid fnlwgt value).
With constant y=0, calibration provides no signal. Adult is documented as
a limitation and excluded from calibration training.

Usage
======
    from src.calibration.calibrator import CalibrationBuilder, calibrate_dataset

    builder = CalibrationBuilder(numeric_tolerance=1e-9)
    samples = builder.build(unc_records, clean_df, dirty_df, "breast_cancer")
    comparison = calibrate_dataset(samples, dataset_name="breast_cancer")
    print_calibration_report(comparison)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid function
from sklearn.isotonic import IsotonicRegression

from src.calibration.metrics import ECEBin, brier_score, expected_calibration_error
from src.uncertainty.estimator import UncertaintyRecord

logger = logging.getLogger(__name__)

# Minimum calibration training samples to attempt fitting
_MIN_TRAIN_SAMPLES: int = 10

# Minimum label variance required for meaningful calibration
_MIN_LABEL_VARIANCE: float = 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# CalibrationSample
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CalibrationSample:
    """
    One (raw_confidence, correctness) pair for calibration.

    Attributes
    ----------
    dataset : str
    row_id : int
    column : str
    raw_confidence : float
        Input feature to the calibrator. No clean values inside.
    correct : int
        Ground-truth label: 1 if top_candidate == clean value, else 0.
        Derived from clean_df via CalibrationBuilder ONLY.
    """

    dataset: str
    row_id: int
    column: str
    raw_confidence: float
    correct: int  # 0 or 1


# ═══════════════════════════════════════════════════════════════════════════════
# CalibrationBuilder
# ═══════════════════════════════════════════════════════════════════════════════


class CalibrationBuilder:
    """
    Converts UncertaintyRecords into CalibrationSamples.

    This is the ONLY class in the calibration phase that reads clean_df.
    It uses clean_df exclusively to create 0/1 correctness labels.

    Parameters
    ----------
    numeric_tolerance : float
        Tolerance for numeric comparison (int vs float64 mismatches).
    """

    def __init__(self, numeric_tolerance: float = 1e-9) -> None:
        self.numeric_tolerance = numeric_tolerance

    def build(
        self,
        unc_records: List[UncertaintyRecord],
        clean_df: pd.DataFrame,
        dirty_df: pd.DataFrame,
        dataset_name: str,
    ) -> List[CalibrationSample]:
        """
        Build calibration samples from uncertainty records.

        clean_df is used ONLY for 0/1 correctness labels.
        The calibrator feature (raw_confidence) comes from unc_records
        which contains NO clean values.
        """
        clean_df_idx = clean_df.reset_index(drop=True)
        samples: List[CalibrationSample] = []

        for rec in unc_records:
            if rec.column not in clean_df_idx.columns:
                continue
            if rec.row_id >= len(clean_df_idx):
                continue
            clean_val = clean_df_idx.at[rec.row_id, rec.column]
            correct = int(self._is_correct(rec.top_candidate, clean_val))
            samples.append(CalibrationSample(
                dataset=dataset_name,
                row_id=rec.row_id,
                column=rec.column,
                raw_confidence=rec.raw_confidence,
                correct=correct,
            ))

        n_pos = sum(s.correct for s in samples)
        logger.debug(
            "CalibrationBuilder '%s': %d samples, %d positive (%.1f%%).",
            dataset_name, len(samples), n_pos,
            100 * n_pos / len(samples) if samples else 0,
        )
        return samples

    def _is_correct(self, candidate: Any, clean_val: Any) -> bool:
        if clean_val is None or (isinstance(clean_val, float) and pd.isna(clean_val)):
            return candidate is None or (
                isinstance(candidate, float) and pd.isna(candidate)
            )
        if candidate is None:
            return False
        try:
            c_num = float(clean_val)
            r_num = float(candidate)
            if abs(c_num - r_num) <= self.numeric_tolerance * max(1.0, abs(c_num)):
                return True
        except (TypeError, ValueError):
            pass
        return str(candidate).strip() == str(clean_val).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Train/eval split
# ═══════════════════════════════════════════════════════════════════════════════


def train_eval_split(
    samples: List[CalibrationSample],
    eval_fraction: float = 0.30,
    seed: int = 42,
) -> Tuple[List[CalibrationSample], List[CalibrationSample]]:
    """
    Split calibration samples into training (70%) and evaluation (30%) sets.

    Deterministic: same seed → same split every run.

    Returns
    -------
    (train_samples, eval_samples)
    """
    if not samples:
        return [], []
    shuffled = samples.copy()
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * eval_fraction))
    return shuffled[n_eval:], shuffled[:n_eval]


# ═══════════════════════════════════════════════════════════════════════════════
# Calibrators
# ═══════════════════════════════════════════════════════════════════════════════


class RawBaseline:
    """
    Baseline: returns raw_confidence unchanged.
    Used to compare whether calibration improves over uncalibrated predictions.
    """

    name: str = "raw"

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(X, dtype=float), 0.0, 1.0)

    def is_fitted(self) -> bool:
        return True


class LogisticCalibrator:
    """
    Platt-style logistic calibration.

    Learns:
        p_calibrated = sigmoid(a × p_raw + b)
                     = 1 / (1 + exp(-(a × p_raw + b)))

    Fitted by minimising binary cross-entropy (negative log-likelihood):
        L(a, b) = −mean[ y × log(sigmoid(ax+b)) + (1−y) × log(1−sigmoid(ax+b)) ]

    a=1, b=0 → identity in sigmoid space (no calibration effect).
    a < 1   → compresses predictions toward 0.5 (less confident).
    a > 1   → expands predictions away from 0.5 (more confident).
    b < 0   → shifts all predictions downward.
    """

    name: str = "logistic"

    def __init__(self, init_a: float = 1.0, init_b: float = 0.0) -> None:
        self.init_a = init_a
        self.init_b = init_b
        self.a_: Optional[float] = None
        self.b_: Optional[float] = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if y.var() < _MIN_LABEL_VARIANCE:
            logger.warning(
                "LogisticCalibrator: constant labels (all %s). Setting a=1, b=0.",
                int(y[0]) if len(y) else "?",
            )
            self.a_ = 1.0
            self.b_ = float(np.log(y.mean() + 1e-9) - np.log(1 - y.mean() + 1e-9))
            self._fitted = False
            return

        eps = 1e-9

        def _nll(params: np.ndarray) -> float:
            a, b = params
            p = expit(a * X + b)
            return -float(np.mean(
                y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)
            ))

        result = minimize(
            _nll,
            x0=[self.init_a, self.init_b],
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        self.a_ = float(result.x[0])
        self.b_ = float(result.x[1])
        self._fitted = True
        logger.debug("LogisticCalibrator: a=%.4f b=%.4f converged=%s",
                     self.a_, self.b_, result.success)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.a_ is None:
            raise RuntimeError("LogisticCalibrator not fitted.")
        return expit(self.a_ * np.asarray(X, dtype=float) + self.b_)

    def is_fitted(self) -> bool:
        return self._fitted


class IsotonicCalibrator:
    """
    Isotonic regression calibration.

    Non-parametric monotone step-function that minimises MSE
    subject to the mapping being non-decreasing.

    Unlike logistic calibration:
    - No assumed shape — can fit any monotone curve.
    - More data needed (≥ 50 training samples recommended).
    - Values outside the training range are clipped.
    """

    name: str = "isotonic"

    def __init__(self) -> None:
        self._reg: Optional[IsotonicRegression] = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if y.var() < _MIN_LABEL_VARIANCE:
            logger.warning("IsotonicCalibrator: constant labels. Skipping fit.")
            self._fitted = False
            return

        self._reg = IsotonicRegression(y_min=0.0, y_max=1.0,
                                       out_of_bounds="clip", increasing=True)
        self._reg.fit(X, y)
        self._fitted = True
        logger.debug("IsotonicCalibrator fitted on %d samples.", len(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or self._reg is None:
            raise RuntimeError("IsotonicCalibrator not fitted.")
        return np.clip(self._reg.predict(np.asarray(X, dtype=float)), 0.0, 1.0)

    def is_fitted(self) -> bool:
        return self._fitted


# ═══════════════════════════════════════════════════════════════════════════════
# CalibrationComparison
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CalibrationComparison:
    """
    Calibration results for one dataset (eval split only).

    Attributes
    ----------
    dataset_name : str
    n_train / n_eval : int
        Split sizes.
    n_positive_eval : int
        Correct labels in eval split.
    base_rate : float
        Empirical accuracy = n_positive_eval / n_eval.
    raw/logistic/isotonic_brier : float
        Brier scores (lower = better).
    raw/logistic/isotonic_ece : float
        Expected Calibration Error (lower = better).
    logistic/isotonic_fitted : bool
        Whether each calibrator was successfully fitted.
    raw/logistic/isotonic_bins : list[ECEBin]
        Per-bin stats for reliability diagrams.
    logistic_a / logistic_b : float
        Fitted sigmoid parameters.
    limitation : str or None
        Human-readable dataset limitation note.
    """

    dataset_name: str
    n_train: int
    n_eval: int
    n_positive_eval: int
    base_rate: float

    raw_brier: float
    logistic_brier: float
    isotonic_brier: float

    raw_ece: float
    logistic_ece: float
    isotonic_ece: float

    logistic_fitted: bool
    isotonic_fitted: bool

    raw_bins: List[ECEBin] = field(default_factory=list)
    logistic_bins: List[ECEBin] = field(default_factory=list)
    isotonic_bins: List[ECEBin] = field(default_factory=list)

    logistic_a: Optional[float] = None
    logistic_b: Optional[float] = None

    limitation: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline function
# ═══════════════════════════════════════════════════════════════════════════════


_ADULT_LIMITATION = (
    "Adult: Phase 3 detected only statistical outliers in 'fnlwgt' "
    "(IQR false positives). These are not genuine errors — all correctness "
    "labels are 0. No meaningful calibration signal exists. "
    "True errors (income label flips) are undetectable from dirty data alone."
)


def calibrate_dataset(
    samples: List[CalibrationSample],
    dataset_name: str,
    eval_fraction: float = 0.30,
    seed: int = 42,
    n_ece_bins: int = 10,
) -> CalibrationComparison:
    """
    Run the full calibration pipeline for one dataset.

    Steps:
    1. Split into train (70%) / eval (30%) sets.
    2. Check label variance — skip fitting if constant labels.
    3. Fit LogisticCalibrator and IsotonicCalibrator on train.
    4. Evaluate raw, logistic, isotonic on eval set.
    5. Return CalibrationComparison with all metrics.
    """
    limitation: Optional[str] = None
    if dataset_name == "adult":
        limitation = _ADULT_LIMITATION
        logger.warning("calibrate_dataset: Adult limitation noted.")

    if not samples:
        return CalibrationComparison(
            dataset_name=dataset_name,
            n_train=0, n_eval=0, n_positive_eval=0, base_rate=0.0,
            raw_brier=0.25, logistic_brier=0.25, isotonic_brier=0.25,
            raw_ece=0.0, logistic_ece=0.0, isotonic_ece=0.0,
            logistic_fitted=False, isotonic_fitted=False,
            limitation=limitation,
        )

    train_samples, eval_samples = train_eval_split(
        samples, eval_fraction=eval_fraction, seed=seed
    )

    X_train = np.array([s.raw_confidence for s in train_samples])
    y_train = np.array([s.correct for s in train_samples], dtype=float)
    X_eval = np.array([s.raw_confidence for s in eval_samples])
    y_eval = np.array([s.correct for s in eval_samples], dtype=float)

    n_pos_eval = int(y_eval.sum())
    base_rate = float(y_eval.mean()) if len(y_eval) > 0 else 0.0

    if y_train.size > 0 and y_train.var() < _MIN_LABEL_VARIANCE and limitation is None:
        limitation = (
            f"'{dataset_name}': all training labels are constant "
            f"({'1' if y_train.mean() > 0.5 else '0'}). "
            "Calibration provides no signal."
        )

    log_cal = LogisticCalibrator()
    iso_cal = IsotonicCalibrator()
    raw_base = RawBaseline()

    if len(X_train) >= 1:
        log_cal.fit(X_train, y_train)
        iso_cal.fit(X_train, y_train)

    raw_preds = raw_base.predict(X_eval)
    log_preds = log_cal.predict(X_eval) if log_cal.a_ is not None else raw_preds.copy()
    iso_preds = iso_cal.predict(X_eval) if iso_cal.is_fitted() else raw_preds.copy()

    raw_b = brier_score(raw_preds, y_eval)
    log_b = brier_score(log_preds, y_eval)
    iso_b = brier_score(iso_preds, y_eval)

    raw_ece, raw_bins = expected_calibration_error(raw_preds, y_eval, n_ece_bins)
    log_ece, log_bins = expected_calibration_error(log_preds, y_eval, n_ece_bins)
    iso_ece, iso_bins = expected_calibration_error(iso_preds, y_eval, n_ece_bins)

    logger.info(
        "Calibration '%s': train=%d eval=%d base=%.3f | "
        "Brier raw=%.4f log=%.4f iso=%.4f | ECE raw=%.4f log=%.4f iso=%.4f",
        dataset_name, len(train_samples), len(eval_samples), base_rate,
        raw_b, log_b, iso_b, raw_ece, log_ece, iso_ece,
    )

    return CalibrationComparison(
        dataset_name=dataset_name,
        n_train=len(train_samples),
        n_eval=len(eval_samples),
        n_positive_eval=n_pos_eval,
        base_rate=round(base_rate, 4),
        raw_brier=round(raw_b, 4),
        logistic_brier=round(log_b, 4),
        isotonic_brier=round(iso_b, 4),
        raw_ece=round(raw_ece, 4),
        logistic_ece=round(log_ece, 4),
        isotonic_ece=round(iso_ece, 4),
        logistic_fitted=log_cal.is_fitted(),
        isotonic_fitted=iso_cal.is_fitted(),
        raw_bins=raw_bins,
        logistic_bins=log_bins,
        isotonic_bins=iso_bins,
        logistic_a=log_cal.a_,
        logistic_b=log_cal.b_,
        limitation=limitation,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Report printer
# ═══════════════════════════════════════════════════════════════════════════════


def print_calibration_report(comp: CalibrationComparison) -> None:
    """Pretty-print a CalibrationComparison to stdout."""
    w = 68
    print(f"\n{'═' * w}")
    print(f"  CALIBRATION REPORT — {comp.dataset_name.upper()}")
    print(f"{'─' * w}")
    print(f"  Train samples  : {comp.n_train:,}")
    print(f"  Eval samples   : {comp.n_eval:,}")
    print(f"  Positive labels: {comp.n_positive_eval:,}  "
          f"(base rate = {comp.base_rate:.3f})")

    if comp.limitation:
        print(f"\n  LIMITATION: {comp.limitation}")

    print(f"\n  {'Method':<12} {'Brier':>9} {'ECE':>9} {'Status':>10}")
    print(f"  {'─' * 12} {'─' * 9} {'─' * 9} {'─' * 10}")
    print(f"  {'raw':<12} {comp.raw_brier:>9.4f} {comp.raw_ece:>9.4f} {'(baseline)':>10}")
    log_status = "fitted" if comp.logistic_fitted else "skipped"
    iso_status = "fitted" if comp.isotonic_fitted else "skipped"
    print(f"  {'logistic':<12} {comp.logistic_brier:>9.4f} {comp.logistic_ece:>9.4f} {log_status:>10}")
    print(f"  {'isotonic':<12} {comp.isotonic_brier:>9.4f} {comp.isotonic_ece:>9.4f} {iso_status:>10}")

    if comp.logistic_a is not None and comp.logistic_fitted:
        print(f"\n  Logistic params: sigmoid({comp.logistic_a:.4f} × p + {comp.logistic_b:.4f})")

    if comp.logistic_fitted or comp.isotonic_fitted:
        briers = {
            "raw": comp.raw_brier,
            "logistic": comp.logistic_brier,
            "isotonic": comp.isotonic_brier,
        }
        best = min(briers, key=briers.get)
        print(f"  Best method (Brier): {best.upper()}")

    print(f"{'═' * w}\n")
