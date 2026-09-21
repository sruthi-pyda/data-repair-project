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

Method C: Multivariate logistic calibration
--------------------------------------------
Phase 5 computes three uncertainty signals per repair, but Methods A and B
use only the first:

    raw_confidence      probability the repairer assigned to its top candidate
    self_consistency    fraction of resampled repair calls that agreed with it
    normalized_entropy  spread of the candidate distribution, in [0, 1]

Method C fits a logistic regression over all three:

    p_cal = sigmoid(w0 + w1·raw_conf + w2·self_consistency + w3·norm_entropy)

with a small L2 penalty for numerical stability. It is the ablation for
CLAUDE.md §5.3: if Brier/ECE do not improve over Method A, the extra
signals carry no information beyond raw confidence; if they do, that is a
cheap, real contribution. Samples that lack the extra signals (built from
older records) skip Method C rather than imputing.

METHOD COMPARISON
==================
| Method       | Features                  | Assumptions          | Data needed | Risk of over-fit |
|--------------|---------------------------|----------------------|-------------|-----------------|
| Raw          | raw_confidence            | None (baseline)      | None        | None            |
| Logistic     | raw_confidence            | Smooth sigmoid shape | Low (~20+)  | Low             |
| Isotonic     | raw_confidence            | Only monotonicity    | Moderate    | Moderate        |
| Multivariate | conf + consistency + ent. | Linear in logit      | Low (~30+)  | Low (L2)        |

DATA LEAKAGE PREVENTION
========================
clean_df is used ONLY to create 0/1 correctness labels in CalibrationBuilder.
The calibrator's input features are ONLY the Phase 5 uncertainty signals
(raw_confidence, self_consistency, normalized_entropy) — NOT the actual
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

# Feature names used by MultivariateLogisticCalibrator, in column order.
MULTIVARIATE_FEATURES: Tuple[str, str, str] = (
    "raw_confidence", "self_consistency", "normalized_entropy",
)

# L2 penalty on the multivariate weights (not the bias). Small: it only
# prevents divergence when a feature separates the labels perfectly.
_MULTIVARIATE_L2: float = 1e-3


# ═══════════════════════════════════════════════════════════════════════════════
# CalibrationSample
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CalibrationSample:
    """
    One (uncertainty signals, correctness) pair for calibration.

    Attributes
    ----------
    dataset : str
    row_id : int
    column : str
    raw_confidence : float
        Primary input feature to the calibrator. No clean values inside.
    correct : int
        Ground-truth label: 1 if top_candidate == clean value, else 0.
        Derived from clean_df via CalibrationBuilder ONLY.
    self_consistency : float or None
        Phase 5 agreement fraction. Optional so that samples built before
        this field existed (or by hand in tests) still construct.
    normalized_entropy : float or None
        Phase 5 candidate-distribution entropy scaled to [0, 1]. Optional
        for the same reason.

    Both optional fields are additional calibrator *inputs* (Method C). They
    come from UncertaintyRecord, which never sees clean values.
    """

    dataset: str
    row_id: int
    column: str
    raw_confidence: float
    correct: int  # 0 or 1
    self_consistency: Optional[float] = None
    normalized_entropy: Optional[float] = None

    def has_multivariate_features(self) -> bool:
        """True when every Method C feature is present."""
        return self.self_consistency is not None and self.normalized_entropy is not None

    def feature_vector(self) -> List[float]:
        """Features in MULTIVARIATE_FEATURES order. Raises if any is missing."""
        if not self.has_multivariate_features():
            raise ValueError(
                f"Sample ({self.dataset}, row {self.row_id}, {self.column}) "
                "lacks self_consistency / normalized_entropy."
            )
        return [
            float(self.raw_confidence),
            float(self.self_consistency),  # type: ignore[arg-type]
            float(self.normalized_entropy),  # type: ignore[arg-type]
        ]


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
                self_consistency=rec.self_consistency,
                normalized_entropy=rec.normalized_entropy,
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


class MultivariateLogisticCalibrator:
    """
    Logistic regression over all three Phase 5 uncertainty signals.

    Learns:
        p_calibrated = sigmoid(w · x + b)
        x = [raw_confidence, self_consistency, normalized_entropy]

    Fitted by minimising binary cross-entropy plus a small L2 penalty on w
    (not on b). With L2 → 0 and w2 = w3 = 0 this reduces exactly to
    LogisticCalibrator, so any Brier/ECE gain over Method A is attributable
    to the extra signals.

    Interpretation of fitted weights (sign, not magnitude):
        w1 > 0   higher raw confidence   → more likely correct
        w2 > 0   more self-consistent    → more likely correct
        w3 < 0   higher entropy (spread) → less likely correct

    Same interface as LogisticCalibrator / IsotonicCalibrator, but ``X`` is
    2-D with shape (n_samples, 3). A 1-D ``X`` is rejected rather than
    silently broadcast.
    """

    name: str = "multivariate"
    feature_names: Tuple[str, ...] = MULTIVARIATE_FEATURES

    def __init__(self, l2: float = _MULTIVARIATE_L2) -> None:
        if l2 < 0:
            raise ValueError(f"l2 must be >= 0, got {l2}")
        self.l2 = l2
        self.weights_: Optional[np.ndarray] = None
        self.bias_: Optional[float] = None
        self._fitted = False

    @staticmethod
    def _as_matrix(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(MULTIVARIATE_FEATURES):
            raise ValueError(
                f"Expected X of shape (n, {len(MULTIVARIATE_FEATURES)}), got {X.shape}"
            )
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = self._as_matrix(X)
        y = np.asarray(y, dtype=float)
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}")

        if len(y) == 0 or y.var() < _MIN_LABEL_VARIANCE:
            logger.warning(
                "MultivariateLogisticCalibrator: constant labels. Skipping fit."
            )
            self.weights_ = None
            self.bias_ = None
            self._fitted = False
            return

        eps = 1e-9
        n_feat = X.shape[1]

        def _objective(params: np.ndarray) -> float:
            w, b = params[:n_feat], params[n_feat]
            p = expit(X @ w + b)
            nll = -float(np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
            return nll + self.l2 * float(np.dot(w, w))

        # Start from Method A's identity: weight 1 on raw_confidence, 0 elsewhere.
        x0 = np.zeros(n_feat + 1)
        x0[0] = 1.0
        result = minimize(
            _objective, x0=x0, method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        self.weights_ = np.asarray(result.x[:n_feat], dtype=float)
        self.bias_ = float(result.x[n_feat])
        self._fitted = True
        logger.debug(
            "MultivariateLogisticCalibrator: w=%s b=%.4f converged=%s",
            np.round(self.weights_, 4), self.bias_, result.success,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or self.weights_ is None or self.bias_ is None:
            raise RuntimeError("MultivariateLogisticCalibrator not fitted.")
        X = self._as_matrix(X)
        return np.clip(expit(X @ self.weights_ + self.bias_), 0.0, 1.0)

    def is_fitted(self) -> bool:
        return self._fitted


def samples_to_feature_matrix(samples: List[CalibrationSample]) -> np.ndarray:
    """Stack Method C feature vectors into an (n, 3) array."""
    if not samples:
        return np.empty((0, len(MULTIVARIATE_FEATURES)))
    return np.array([s.feature_vector() for s in samples], dtype=float)


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
    multivariate_brier / multivariate_ece : float
        Method C metrics. Copies of the raw metrics when not fitted.
    multivariate_fitted : bool
        False when labels are constant OR the samples lack the extra
        Phase 5 signals.
    multivariate_bins : list[ECEBin]
    multivariate_weights : list[float] or None
        Fitted [w_raw_conf, w_self_consistency, w_norm_entropy].
    multivariate_bias : float or None
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

    # Method C (multivariate). Defaulted so older call sites still construct.
    multivariate_brier: float = 0.25
    multivariate_ece: float = 0.0
    multivariate_fitted: bool = False
    multivariate_bins: List[ECEBin] = field(default_factory=list)
    multivariate_weights: Optional[List[float]] = None
    multivariate_bias: Optional[float] = None

    _NOT_FITTED_MARKER = "—"
    _METHODS = ("raw", "logistic", "isotonic", "multivariate")

    def fitted(self, method: str) -> bool:
        """Whether ``method`` produced genuine calibrated predictions."""
        if method not in self._METHODS:
            raise ValueError(f"Unknown method '{method}'")
        if method == "raw":
            return True
        return bool(getattr(self, f"{method}_fitted"))

    def metric_str(self, method: str, metric: str, precision: int = 4) -> str:
        """
        Format a Brier/ECE value for display, or ``"—"`` when the calibrator
        behind it was never fitted.

        When a calibrator refuses to fit (e.g. Adult, where every label is 0)
        the stored ``logistic_*`` / ``isotonic_*`` values are copies of the raw
        metrics, not genuine calibrated scores. Printing them as numbers reads
        like a perfect ``0.0000`` in a results table. Use this instead of
        formatting the field directly.

        Parameters
        ----------
        method : {"raw", "logistic", "isotonic", "multivariate"}
        metric : {"brier", "ece"}
        """
        if metric not in ("brier", "ece"):
            raise ValueError(f"Unknown metric '{metric}'")
        if not self.fitted(method):
            return self._NOT_FITTED_MARKER
        return f"{getattr(self, f'{method}_{metric}'):.{precision}f}"


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
    3. Fit LogisticCalibrator and IsotonicCalibrator on train, plus
       MultivariateLogisticCalibrator when every sample carries the extra
       Phase 5 signals.
    4. Evaluate raw, logistic, isotonic, multivariate on eval set.
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
    multi_cal = MultivariateLogisticCalibrator()
    raw_base = RawBaseline()

    if len(X_train) >= 1:
        log_cal.fit(X_train, y_train)
        iso_cal.fit(X_train, y_train)

    # Method C needs the extra Phase 5 signals on every sample; skip (not
    # impute) when any are missing so the ablation stays honest.
    has_multi = all(s.has_multivariate_features() for s in samples)
    if has_multi and len(X_train) >= 1:
        multi_cal.fit(samples_to_feature_matrix(train_samples), y_train)
    elif not has_multi:
        logger.info(
            "Calibration '%s': samples lack self_consistency/normalized_entropy; "
            "multivariate calibrator skipped.", dataset_name,
        )

    raw_preds = raw_base.predict(X_eval)
    log_preds = log_cal.predict(X_eval) if log_cal.a_ is not None else raw_preds.copy()
    iso_preds = iso_cal.predict(X_eval) if iso_cal.is_fitted() else raw_preds.copy()
    multi_preds = (
        multi_cal.predict(samples_to_feature_matrix(eval_samples))
        if multi_cal.is_fitted() else raw_preds.copy()
    )

    raw_b = brier_score(raw_preds, y_eval)
    log_b = brier_score(log_preds, y_eval)
    iso_b = brier_score(iso_preds, y_eval)
    multi_b = brier_score(multi_preds, y_eval)

    raw_ece, raw_bins = expected_calibration_error(raw_preds, y_eval, n_ece_bins)
    log_ece, log_bins = expected_calibration_error(log_preds, y_eval, n_ece_bins)
    iso_ece, iso_bins = expected_calibration_error(iso_preds, y_eval, n_ece_bins)
    multi_ece, multi_bins = expected_calibration_error(multi_preds, y_eval, n_ece_bins)

    logger.info(
        "Calibration '%s': train=%d eval=%d base=%.3f | "
        "Brier raw=%.4f log=%.4f iso=%.4f multi=%.4f | "
        "ECE raw=%.4f log=%.4f iso=%.4f multi=%.4f",
        dataset_name, len(train_samples), len(eval_samples), base_rate,
        raw_b, log_b, iso_b, multi_b, raw_ece, log_ece, iso_ece, multi_ece,
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
        multivariate_brier=round(multi_b, 4),
        multivariate_ece=round(multi_ece, 4),
        multivariate_fitted=multi_cal.is_fitted(),
        multivariate_bins=multi_bins,
        multivariate_weights=(
            [round(float(w), 4) for w in multi_cal.weights_]
            if multi_cal.is_fitted() and multi_cal.weights_ is not None else None
        ),
        multivariate_bias=(
            round(multi_cal.bias_, 4)
            if multi_cal.is_fitted() and multi_cal.bias_ is not None else None
        ),
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
    for method in ("logistic", "isotonic", "multivariate"):
        status = "fitted" if comp.fitted(method) else "skipped"
        print(f"  {method:<12} {comp.metric_str(method, 'brier'):>9} "
              f"{comp.metric_str(method, 'ece'):>9} {status:>10}")

    if comp.logistic_a is not None and comp.logistic_fitted:
        print(f"\n  Logistic params: sigmoid({comp.logistic_a:.4f} × p + {comp.logistic_b:.4f})")
    if comp.multivariate_fitted and comp.multivariate_weights is not None:
        terms = " + ".join(
            f"{w:.4f}·{name}" for w, name in zip(comp.multivariate_weights, MULTIVARIATE_FEATURES)
        )
        print(f"  Multivariate  : sigmoid({terms} + {comp.multivariate_bias:.4f})")

    fitted_methods = [m for m in ("logistic", "isotonic", "multivariate") if comp.fitted(m)]
    if fitted_methods:
        briers = {"raw": comp.raw_brier}
        briers.update({m: getattr(comp, f"{m}_brier") for m in fitted_methods})
        best = min(briers, key=briers.get)
        print(f"  Best method (Brier): {best.upper()}")

    print(f"{'═' * w}\n")
