"""
src/calibration/detection_calibrator.py
----------------------------------------
Detection calibration: P(flagged cell is genuinely corrupted).

WHY A SECOND CALIBRATOR?
=========================
Phase 6 calibrates ``p`` = P(the proposed repair is correct). The decision
policy also needs a second quantity that the pipeline has so far assumed
silently:

    q = P(the flagged cell is genuinely corrupted)

Expected abstention cost is not a constant. Leaving a cell alone costs
nothing if the detector raised a false alarm, and ``C_abstain`` if the cell
really was corrupted:

    E[ABSTAIN] = q · C_abstain

With ``q ≡ 1`` (today's implicit assumption) ``E[ABSTAIN]`` is a flat line,
and two flat lines (HUMAN_REVIEW is also flat in ``p``) never cross, so one
of the two actions is unreachable on every dataset. A real ``q`` makes all
three decisions live. See docs/three_actions_plan.md §0.

WHY NOT USE detection_score?
=============================
``DetectionRecord.detection_score`` is a suspicion score, not a probability.
Measured on the full datasets (2026-09-22):

    missing_value / range_check / format_check / vocab_check   100 % corrupted
    iqr_outlier (k=3), adult + smartfactory                      0 % of 41,468
    iqr_outlier (k=1.5), breast_cancer                           3 % (5/195)

Adult outliers score ~0.6 and none are corrupted, so treating the score as
a probability is badly miscalibrated. The detection *method* carries almost
all the signal; the score bins within a method carry essentially none.

METHOD: grouped empirical rate with a Beta prior
================================================
``q`` is learned from 0/1 corruption labels on a training split, grouped by
``detection_method`` (default). For a group with ``n`` records of which
``k`` are corrupted:

    q_group = (k + s) / (n + 2s)          s = smoothing (Beta(s, s) prior)

With ``s = 1`` (Laplace) a group of 100 all-clean records gives
``q = 1/102 ≈ 0.0098`` rather than exactly 0, and 100 all-corrupted gives
``q ≈ 0.990`` rather than exactly 1 — the calibrator never claims certainty
from finite evidence. Unseen groups fall back to the global rate computed
with the same smoothing. ``score_bins > 0`` appends a quantised
``detection_score`` to the group key; it is the ablation lever and is
expected to add nothing.

Not fitted → ``q = 1.0`` for every record. This reproduces the pipeline's
historical behaviour (every detection assumed to be a real error) so that
callers without labels degrade to the old decisions, not to an exception.

LEAKAGE RULE
=============
``clean_df`` is read in exactly ONE place in this module:
``DetectionLabelBuilder.build``, which turns each detection into a 0/1 label
(1 = dirty value differs from clean value). This is the same permitted use
as Phase 6's ``CalibrationBuilder`` (CLAUDE.md §3).

``DetectionCalibrator.fit`` / ``predict`` and ``calibrate_detections`` see
only ``DetectionRecord`` fields (method, score) plus the 0/1 labels. They
never receive clean data and never learn "what the correct value is". The
test suite asserts this by signature inspection and by AST-walking this
module for ``clean_df`` references.

WORKED EXAMPLE
==============
Train split (breast_cancer, Tukey fence k=1.5, smoothing=1):

    missing_value   n=270  k=270  →  q = 271/272 = 0.996
    format_check    n= 12  k= 12  →  q =  13/14  = 0.929
    iqr_outlier     n=136  k=  4  →  q =   5/138 = 0.036

A ``missing_value`` cell with BC costs (wrong=50, human=5, abstain=10):

    E[ABSTAIN] = 0.996 × 10 = 9.96   > E[HUMAN] = 5     → never abstain

An ``iqr_outlier`` cell:

    E[ABSTAIN] = 0.036 × 10 = 0.36   < E[HUMAN] = 5     → ABSTAIN unless
                                                           p is high enough
                                                           for AUTO

Usage
======
    from src.calibration.detection_calibrator import (
        DetectionLabelBuilder, fit_detection_calibrator,
    )

    labels = DetectionLabelBuilder().build(detections, clean_df, dirty_df)
    calibrator, report = fit_detection_calibrator(detections, labels, seed=42)
    q = calibrator.predict(detections)           # one q per detection
    print(report.metric_str("calibrated", "brier"))
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.calibration.metrics import brier_score, expected_calibration_error
from src.detection.detector import DetectionRecord
from src.evaluation.repair_metrics import ValueComparator

logger = logging.getLogger(__name__)

# Fewer training records than this → refuse to fit (q falls back to 1.0).
_MIN_TRAIN_RECORDS: int = 3

# Printed in place of a metric when the calibrator was never fitted.
_NOT_FITTED_MARKER: str = "—"

GroupKey = Tuple[Any, ...]


# ═══════════════════════════════════════════════════════════════════════════════
# DetectionLabelBuilder — the ONLY reader of clean_df in this module
# ═══════════════════════════════════════════════════════════════════════════════


class DetectionLabelBuilder:
    """
    Turn detections into 0/1 corruption labels.

    label = 1  if the dirty cell differs from the clean cell at
               (row_id, column), else 0.

    This is the ONLY class in this module that reads ``clean_df``. It uses
    it exclusively to produce the label; nothing about the clean value is
    stored or returned.

    Alignment is strict: a detection whose column is absent from
    ``clean_df`` or whose ``row_id`` is out of range raises ``ValueError``
    rather than being dropped. Downstream code (the pipeline runner) relies
    on ``labels[i]`` belonging to ``detections[i]``, so silent drops would
    misalign every later record. This differs deliberately from
    ``CalibrationBuilder``, which skips such records.

    Which dirty value is compared: ``detections[i].dirty_value``, the value
    the detector actually saw. ``dirty_df`` is used only for validation
    (row/column existence) so that a detection list from a different frame
    is rejected loudly.

    Parameters
    ----------
    comparator : ValueComparator, optional
        Phase 10 comparator (NaN == NaN, numeric tolerance, string strip).
        Defaults to ``ValueComparator()`` with 1e-9 tolerances.
    """

    def __init__(self, comparator: Optional[ValueComparator] = None) -> None:
        self.comparator = comparator if comparator is not None else ValueComparator()

    def build(
        self,
        detections: List[DetectionRecord],
        clean_df: pd.DataFrame,
        dirty_df: pd.DataFrame,
    ) -> List[int]:
        """
        Return one 0/1 label per detection, in the same order.

        Raises
        ------
        ValueError
            If any detection's column is missing from ``clean_df`` or
            ``dirty_df``, or its ``row_id`` is outside either frame.
        """
        clean_idx = clean_df.reset_index(drop=True)
        dirty_idx = dirty_df.reset_index(drop=True)
        n_clean, n_dirty = len(clean_idx), len(dirty_idx)

        labels: List[int] = []
        for i, rec in enumerate(detections):
            if rec.column not in clean_idx.columns:
                raise ValueError(
                    f"Detection {i} (row {rec.row_id}): column '{rec.column}' "
                    "not in clean_df."
                )
            if rec.column not in dirty_idx.columns:
                raise ValueError(
                    f"Detection {i} (row {rec.row_id}): column '{rec.column}' "
                    "not in dirty_df."
                )
            if not (0 <= rec.row_id < n_clean):
                raise ValueError(
                    f"Detection {i}: row_id {rec.row_id} outside clean_df "
                    f"(len {n_clean})."
                )
            if not (0 <= rec.row_id < n_dirty):
                raise ValueError(
                    f"Detection {i}: row_id {rec.row_id} outside dirty_df "
                    f"(len {n_dirty})."
                )
            clean_val = clean_idx.at[rec.row_id, rec.column]
            corrupted = not self.comparator.are_equal(rec.dirty_value, clean_val)
            labels.append(int(corrupted))

        n_pos = sum(labels)
        logger.debug(
            "DetectionLabelBuilder: %d detections, %d corrupted (%.1f%%).",
            len(labels), n_pos, 100 * n_pos / len(labels) if labels else 0.0,
        )
        return labels


# ═══════════════════════════════════════════════════════════════════════════════
# DetectionCalibrator — never sees clean data
# ═══════════════════════════════════════════════════════════════════════════════


def _validate_labels(labels: Sequence[int], n_records: int) -> np.ndarray:
    y = np.asarray(list(labels))
    if len(y) != n_records:
        raise ValueError(
            f"records and labels must have the same length; got "
            f"{n_records} records and {len(y)} labels."
        )
    if len(y) == 0:
        return y.astype(int)
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("labels must be 0/1.")
    return y.astype(int)


class DetectionCalibrator:
    """
    Per-group smoothed corruption rate. See module docstring.

    Parameters
    ----------
    group_by : tuple of str
        ``DetectionRecord`` attribute names forming the group key.
        Default ``("detection_method",)``.
    smoothing : float
        Beta(s, s) prior pseudo-counts. ``0`` → raw empirical rate.
    score_bins : int
        If > 0, append ``min(score_bins - 1, int(detection_score * score_bins))``
        to the group key. Ablation lever; default 0 (off).

    Inputs to ``fit`` / ``predict`` are detection records and 0/1 labels
    only. No clean values enter this class.
    """

    name: str = "grouped_rate"

    def __init__(
        self,
        group_by: Tuple[str, ...] = ("detection_method",),
        smoothing: float = 1.0,
        score_bins: int = 0,
    ) -> None:
        if smoothing < 0:
            raise ValueError(f"smoothing must be >= 0, got {smoothing}")
        if score_bins < 0:
            raise ValueError(f"score_bins must be >= 0, got {score_bins}")
        if not group_by:
            raise ValueError("group_by must name at least one attribute.")
        self.group_by: Tuple[str, ...] = tuple(group_by)
        self.smoothing = float(smoothing)
        self.score_bins = int(score_bins)
        self._counts: Dict[GroupKey, Tuple[int, int]] = {}   # key -> (n, k)
        self._rates: Dict[GroupKey, float] = {}
        self._global_rate: float = 1.0
        self._global_counts: Tuple[int, int] = (0, 0)
        self._fitted = False

    # ── grouping ────────────────────────────────────────────────────────────

    def group_key(self, record: DetectionRecord) -> GroupKey:
        """Group key for one record (attributes in ``group_by`` + optional bin)."""
        key: List[Any] = [getattr(record, f) for f in self.group_by]
        if self.score_bins > 0:
            score = float(record.detection_score)
            key.append(min(self.score_bins - 1, int(score * self.score_bins)))
        return tuple(key)

    # ── fitting ─────────────────────────────────────────────────────────────

    def _smoothed(self, k: int, n: int) -> float:
        s = self.smoothing
        denom = n + 2 * s
        if denom <= 0:
            # Only when smoothing == 0 and n == 0; caller guards against it.
            return 1.0
        return float((k + s) / denom)

    def fit(self, records: List[DetectionRecord], labels: Sequence[int]) -> None:
        """
        Learn per-group rates from ``(records, labels)``.

        Empty input leaves the calibrator unfitted (``predict`` → 1.0).
        Length mismatch, non-0/1 labels → ``ValueError``.
        """
        y = _validate_labels(labels, len(records))
        self._counts = {}
        self._rates = {}
        self._fitted = False
        self._global_rate = 1.0
        self._global_counts = (0, 0)

        if len(records) == 0:
            logger.warning("DetectionCalibrator.fit: no records; staying unfitted.")
            return

        counts: Dict[GroupKey, List[int]] = {}
        for rec, lab in zip(records, y):
            key = self.group_key(rec)
            c = counts.setdefault(key, [0, 0])
            c[0] += 1
            c[1] += int(lab)

        for key, (n, k) in counts.items():
            self._counts[key] = (n, k)
            self._rates[key] = self._smoothed(k, n)

        n_all, k_all = len(records), int(y.sum())
        self._global_counts = (n_all, k_all)
        self._global_rate = self._smoothed(k_all, n_all)
        self._fitted = True
        logger.debug(
            "DetectionCalibrator fitted: %d records, %d groups, global rate %.4f.",
            n_all, len(self._rates), self._global_rate,
        )

    # ── prediction ──────────────────────────────────────────────────────────

    def predict(self, records: List[DetectionRecord]) -> np.ndarray:
        """
        q for each record, in [0, 1].

        Unseen group → global rate. Not fitted → all 1.0 (historical
        assumption that every detection is a genuine error).
        """
        if not self._fitted:
            return np.ones(len(records), dtype=float)
        out = np.empty(len(records), dtype=float)
        for i, rec in enumerate(records):
            out[i] = self._rates.get(self.group_key(rec), self._global_rate)
        return np.clip(out, 0.0, 1.0)

    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def global_rate(self) -> float:
        return self._global_rate

    def group_table(self) -> List[Dict[str, Any]]:
        """Per-group (key, n, n_corrupted, rate) rows for reporting; sorted by key."""
        rows: List[Dict[str, Any]] = []
        for key in sorted(self._counts, key=lambda k: tuple(str(x) for x in k)):
            n, k = self._counts[key]
            rows.append({
                "group": tuple(key),
                "n": int(n),
                "n_corrupted": int(k),
                "rate": round(self._rates[key], 6),
            })
        return rows


# ═══════════════════════════════════════════════════════════════════════════════
# DetectionCalibrationReport
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DetectionCalibrationReport:
    """
    Hold-out evaluation of q versus the raw ``detection_score``.

    "raw" treats ``detection_score`` as if it were P(corrupted).
    "calibrated" is the grouped-rate q. Both scored on the eval split.

    Attributes
    ----------
    dataset_name : str
    n_train / n_eval : int
    n_corrupted_eval : int
    base_rate : float
        n_corrupted_eval / n_eval.
    raw_brier, raw_ece, calibrated_brier, calibrated_ece : float
        Lower is better. Calibrated values are copies of raw when not fitted.
    fitted : bool
    groups : list of dict
        ``DetectionCalibrator.group_table()`` of the fitted calibrator.
    limitation : str or None
    """

    dataset_name: str
    n_train: int
    n_eval: int
    n_corrupted_eval: int
    base_rate: float
    raw_brier: float
    raw_ece: float
    calibrated_brier: float
    calibrated_ece: float
    fitted: bool
    groups: List[Dict[str, Any]] = field(default_factory=list)
    limitation: Optional[str] = None

    _METHODS = ("raw", "calibrated")

    def metric_str(self, method: str, metric: str, precision: int = 4) -> str:
        """Format a metric, or ``"—"`` for calibrated metrics when not fitted."""
        if method not in self._METHODS:
            raise ValueError(f"Unknown method '{method}'")
        if metric not in ("brier", "ece"):
            raise ValueError(f"Unknown metric '{metric}'")
        if method == "calibrated" and not self.fitted:
            return _NOT_FITTED_MARKER
        return f"{getattr(self, f'{method}_{metric}'):.{precision}f}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "n_train": self.n_train,
            "n_eval": self.n_eval,
            "n_corrupted_eval": self.n_corrupted_eval,
            "base_rate": self.base_rate,
            "raw_brier": self.raw_brier,
            "raw_ece": self.raw_ece,
            "calibrated_brier": self.calibrated_brier,
            "calibrated_ece": self.calibrated_ece,
            "fitted": self.fitted,
            "groups": [dict(g) for g in self.groups],
            "limitation": self.limitation,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Split + pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def _split_indices(
    n: int, eval_fraction: float, seed: int
) -> Tuple[List[int], List[int]]:
    """Deterministic shuffle-split on indices, mirroring calibrator.train_eval_split."""
    if n == 0:
        return [], []
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_eval = max(1, int(n * eval_fraction))
    return idx[n_eval:], idx[:n_eval]


def fit_detection_calibrator(
    records: List[DetectionRecord],
    labels: Sequence[int],
    seed: int = 42,
    eval_fraction: float = 0.30,
    dataset_name: str = "",
    n_ece_bins: int = 10,
    **calibrator_kwargs: Any,
) -> Tuple[DetectionCalibrator, DetectionCalibrationReport]:
    """
    Fit on the train split, evaluate on the eval split.

    Returns the fitted calibrator (so the caller can ``predict`` q for
    every record) and the hold-out report. No clean data enters here.
    """
    y = _validate_labels(labels, len(records))
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError(f"eval_fraction must be in (0, 1), got {eval_fraction}")
    calibrator = DetectionCalibrator(**calibrator_kwargs)

    train_idx, eval_idx = _split_indices(len(records), eval_fraction, seed)
    train_recs = [records[i] for i in train_idx]
    eval_recs = [records[i] for i in eval_idx]
    y_train = y[train_idx] if train_idx else np.array([], dtype=int)
    y_eval = y[eval_idx] if eval_idx else np.array([], dtype=int)

    limitation: Optional[str] = None
    if len(train_recs) < _MIN_TRAIN_RECORDS:
        limitation = (
            f"'{dataset_name}': only {len(train_recs)} training detections "
            f"(< {_MIN_TRAIN_RECORDS}); detection calibrator not fitted, q = 1.0."
        )
        logger.warning(limitation)
    else:
        calibrator.fit(train_recs, y_train)
        if y_train.var() == 0:
            limitation = (
                f"'{dataset_name}': all training labels are constant "
                f"({int(y_train[0])}); q is a smoothed constant per group."
            )

    if len(eval_recs) == 0:
        report = DetectionCalibrationReport(
            dataset_name=dataset_name, n_train=len(train_recs), n_eval=0,
            n_corrupted_eval=0, base_rate=0.0,
            raw_brier=0.25, raw_ece=0.0, calibrated_brier=0.25, calibrated_ece=0.0,
            fitted=calibrator.is_fitted(), groups=calibrator.group_table(),
            limitation=limitation or f"'{dataset_name}': no detections.",
        )
        return calibrator, report

    raw_preds = np.clip(
        np.array([float(r.detection_score) for r in eval_recs]), 0.0, 1.0
    )
    cal_preds = calibrator.predict(eval_recs) if calibrator.is_fitted() else raw_preds.copy()
    y_eval_f = y_eval.astype(float)

    raw_b = brier_score(raw_preds, y_eval_f)
    cal_b = brier_score(cal_preds, y_eval_f)
    raw_ece, _ = expected_calibration_error(raw_preds, y_eval_f, n_ece_bins)
    cal_ece, _ = expected_calibration_error(cal_preds, y_eval_f, n_ece_bins)

    logger.info(
        "Detection calibration '%s': train=%d eval=%d base=%.3f | "
        "Brier raw=%.4f cal=%.4f | ECE raw=%.4f cal=%.4f",
        dataset_name, len(train_recs), len(eval_recs), float(y_eval_f.mean()),
        raw_b, cal_b, raw_ece, cal_ece,
    )

    report = DetectionCalibrationReport(
        dataset_name=dataset_name,
        n_train=len(train_recs),
        n_eval=len(eval_recs),
        n_corrupted_eval=int(y_eval.sum()),
        base_rate=round(float(y_eval_f.mean()), 4),
        raw_brier=round(raw_b, 4),
        raw_ece=round(raw_ece, 4),
        calibrated_brier=round(cal_b, 4),
        calibrated_ece=round(cal_ece, 4),
        fitted=calibrator.is_fitted(),
        groups=calibrator.group_table(),
        limitation=limitation,
    )
    return calibrator, report


def calibrate_detections(
    records: List[DetectionRecord],
    labels: Sequence[int],
    dataset_name: str,
    eval_fraction: float = 0.30,
    seed: int = 42,
    n_ece_bins: int = 10,
    **calibrator_kwargs: Any,
) -> DetectionCalibrationReport:
    """Report-only convenience wrapper around ``fit_detection_calibrator``."""
    _, report = fit_detection_calibrator(
        records, labels, seed=seed, eval_fraction=eval_fraction,
        dataset_name=dataset_name, n_ece_bins=n_ece_bins, **calibrator_kwargs,
    )
    return report


def print_detection_calibration_report(report: DetectionCalibrationReport) -> None:
    """Pretty-print a DetectionCalibrationReport to stdout."""
    w = 68
    print(f"\n{'═' * w}")
    print(f"  DETECTION CALIBRATION — {report.dataset_name.upper()}")
    print(f"{'─' * w}")
    print(f"  Train detections : {report.n_train:,}")
    print(f"  Eval detections  : {report.n_eval:,}")
    print(f"  Corrupted (eval) : {report.n_corrupted_eval:,}  "
          f"(base rate = {report.base_rate:.3f})")
    if report.limitation:
        print(f"\n  LIMITATION: {report.limitation}")
    print(f"\n  {'Method':<12} {'Brier':>9} {'ECE':>9}")
    print(f"  {'─' * 12} {'─' * 9} {'─' * 9}")
    for m in ("raw", "calibrated"):
        print(f"  {m:<12} {report.metric_str(m, 'brier'):>9} "
              f"{report.metric_str(m, 'ece'):>9}")
    if report.groups:
        print(f"\n  {'Group':<36} {'n':>6} {'corrupt':>8} {'q':>8}")
        for g in report.groups:
            print(f"  {str(g['group']):<36} {g['n']:>6} {g['n_corrupted']:>8} "
                  f"{g['rate']:>8.4f}")
    print(f"{'═' * w}\n")
