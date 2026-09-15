"""
src/evaluation/repair_metrics.py
---------------------------------
Phase 10: Repair quality evaluation against ground truth.

DATA-LEAKAGE BOUNDARY
======================
This module is STRICTLY for post-hoc evaluation.

  PREDICTION (upstream)  : dirty_df  →  decisions  (NO clean_df)
  EVALUATION (this file) : decisions + clean_df  →  accuracy metrics

clean_df is used ONLY inside RepairEvaluator and is NEVER passed upstream
to any detection, repair, uncertainty, calibration, or decision function.

IMPORTANT CONCEPT SEPARATION
==============================
Three distinct concepts are evaluated separately:

  A. Detection correctness
     Was this cell actually corrupted?
     Measured by comparing clean_df[row, col] with dirty_df[row, col].

  B. Repair correctness
     Does the proposed repair match the clean value?
     Evaluated only when action == AUTO_REPAIR.

  C. Decision appropriateness
     Given the ground truth, was the routing decision the right one?
     (e.g., HUMAN_REVIEW on a truly corrupted cell is appropriate.)

A false-positive detection (clean == dirty but cell was flagged) is
recorded as FP and never counts as a "correct repair" in the detection
sense, even if the top candidate happens to equal the clean/dirty value.

METRIC DEFINITIONS
==================

Let:
  TP = cells detected AND truly corrupted (clean ≠ dirty)
  FP = cells detected but NOT truly corrupted (clean == dirty)
  FN = truly corrupted cells that were NOT detected

  Precision = TP / (TP + FP)
    "Of all the cells we flagged, what fraction were actually errors?"
    Returns 0.0 when no cells were detected (denominator = 0).

  Recall = TP / (TP + FN)
    "Of all the actual errors, what fraction did we find?"
    Returns 0.0 when there are no actual errors.

  F1 = 2 * Precision * Recall / (Precision + Recall)
    Returns 0.0 when both Precision and Recall are 0.

For REPAIR ACCURACY (among AUTO_REPAIR decisions only):
  Repair accuracy = correct_repairs / n_auto_repair
  A repair is correct iff top_candidate matches clean_value within tolerance.

VALUE COMPARISON
================
See ValueComparator for configurable numerical/categorical comparison.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from src.baselines.baselines import BaselineRecord
from src.decision.policy import DecisionAction

logger = logging.getLogger(__name__)

# Default tolerances for numerical comparison.
# abs_tol=1e-9 and rel_tol=1e-9 gives effectively exact integer comparison
# while tolerating float64 rounding errors.
_DEFAULT_ABS_TOL: float = 1e-9
_DEFAULT_REL_TOL: float = 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# ValueComparator
# ═══════════════════════════════════════════════════════════════════════════════


class ValueComparator:
    """
    Compare a predicted repair value against a ground-truth clean value.

    NUMERICAL COMPARISON
    --------------------
    Uses the formula:
        abs(predicted - actual) <= max(abs_tol, rel_tol * abs(actual))

    Default tolerances are 1e-9, giving effectively exact comparison for
    integers while tolerating float64 rounding differences from dtype
    promotion (e.g., int 3 vs float 3.0).

    Make tolerances larger (e.g., abs_tol=0.5) for approximate numerical
    comparisons such as imputed medians.

    CATEGORICAL COMPARISON
    ----------------------
    Strips surrounding whitespace from both sides before comparing.
    Case normalization is optional (default: case-sensitive).

    Setting normalize_case=True means "Adm-clerical" == "adm-clerical".
    This should only be enabled if the domain is genuinely case-insensitive.
    The default is False to avoid silently hiding real differences.

    Parameters
    ----------
    abs_tol : float
        Absolute tolerance for numerical comparison. Default 1e-9.
    rel_tol : float
        Relative tolerance for numerical comparison. Default 1e-9.
    normalize_case : bool
        Strip and lowercase strings before comparison. Default False.
    """

    def __init__(
        self,
        abs_tol: float = _DEFAULT_ABS_TOL,
        rel_tol: float = _DEFAULT_REL_TOL,
        normalize_case: bool = False,
    ) -> None:
        if abs_tol < 0:
            raise ValueError(f"abs_tol must be >= 0, got {abs_tol}")
        if rel_tol < 0:
            raise ValueError(f"rel_tol must be >= 0, got {rel_tol}")
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.normalize_case = normalize_case

    def are_equal(self, predicted: Any, actual: Any) -> bool:
        """
        Return True if predicted matches actual within configured tolerances.

        Algorithm
        ---------
        1. If both are NA (None, NaN, pd.NA) → True.
        2. If exactly one is NA → False.
        3. Try numeric comparison: abs(p - a) <= max(abs_tol, rel_tol * |a|).
        4. Fall back to string comparison (strip whitespace; optionally fold case).
        """
        pred_na = _is_na(predicted)
        act_na = _is_na(actual)
        if pred_na and act_na:
            return True
        if pred_na or act_na:
            return False

        # Attempt numeric comparison
        try:
            p_num = float(predicted)
            a_num = float(actual)
            # Propagate NaN inequality explicitly
            if math.isnan(p_num) or math.isnan(a_num):
                return False
            tol = max(self.abs_tol, self.rel_tol * abs(a_num))
            return abs(p_num - a_num) <= tol
        except (TypeError, ValueError):
            pass

        # String comparison
        p_str = str(predicted).strip()
        a_str = str(actual).strip()
        if self.normalize_case:
            p_str = p_str.lower()
            a_str = a_str.lower()
        return p_str == a_str


def _is_na(value: Any) -> bool:
    """Return True if value is None, float NaN, or pd.NA."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Per-cell result
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CellResult:
    """
    Ground-truth evaluation for a single detected-and-decided cell.

    This object is produced by RepairEvaluator and is used to build
    aggregate RepairMetrics.  clean_value is stored here so the
    caller can audit individual cells.

    Attributes
    ----------
    method_name : str
    row_id : int
    column : str
    dirty_value : Any
    top_candidate : Any
        The proposed repair value.
    clean_value : Any
        The ground-truth clean value (from clean_df).
    action : DecisionAction
    is_truly_corrupted : bool
        True iff clean_df[row_id, column] ≠ dirty_df[row_id, column].
        (True Positive in detection terms.)
    is_repair_correct : bool
        True iff top_candidate matches clean_value within tolerance.
        Evaluated regardless of whether cell was truly corrupted.
    """

    method_name: str
    row_id: int
    column: str
    dirty_value: Any
    top_candidate: Any
    clean_value: Any
    action: DecisionAction
    is_truly_corrupted: bool
    is_repair_correct: bool


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregate repair metrics
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RepairMetrics:
    """
    Aggregate repair and detection quality metrics for one method + dataset.

    DETECTION METRICS (require knowing ALL truly corrupted cells)
    -------------------------------------------------------------
    n_detected          : all records evaluated (all were flagged as suspicious)
    n_truly_corrupted   : TP — detected AND truly corrupted
    n_false_positive    : FP — detected but NOT truly corrupted
    n_false_negative    : FN — truly corrupted but NOT detected
    n_total_corrupted   : TP + FN (total actual errors in dataset)

    detection_precision : TP / (TP + FP)  [0.0 if no detections]
    detection_recall    : TP / (TP + FN)  [0.0 if no actual errors]
    detection_f1        : harmonic mean of precision and recall

    REPAIR METRICS (among AUTO_REPAIR decisions only)
    -------------------------------------------------
    n_auto_repair       : count of AUTO_REPAIR decisions
    n_correct_repairs   : AUTO_REPAIR decisions where top_candidate == clean
    n_incorrect_repairs : AUTO_REPAIR decisions where top_candidate != clean
    repair_accuracy     : correct_repairs / n_auto_repair  [0.0 if no auto repairs]

    NOTE: repair_accuracy is computed over ALL auto-repaired cells, including
    false-positive detections.  A FP cell that gets auto-repaired with a value
    equal to the (clean==dirty) value is counted as correct — we did not
    change anything meaningful.  A FP cell auto-repaired to a different value
    is counted as an incorrect repair (we introduced a new error).

    ACTION DISTRIBUTION
    -------------------
    n_human_review / n_abstain
    """

    method_name: str
    dataset: str

    # Detection
    n_detected: int = 0
    n_truly_corrupted: int = 0
    n_false_positive: int = 0
    n_false_negative: int = 0
    n_total_corrupted: int = 0

    detection_precision: float = 0.0
    detection_recall: float = 0.0
    detection_f1: float = 0.0

    # Repair (AUTO_REPAIR decisions only)
    n_auto_repair: int = 0
    n_correct_repairs: int = 0
    n_incorrect_repairs: int = 0
    repair_accuracy: float = 0.0

    # Action distribution
    n_human_review: int = 0
    n_abstain: int = 0

    def to_dict(self) -> Dict:
        return {
            "method_name": self.method_name,
            "dataset": self.dataset,
            "n_detected": self.n_detected,
            "n_truly_corrupted": self.n_truly_corrupted,
            "n_false_positive": self.n_false_positive,
            "n_false_negative": self.n_false_negative,
            "n_total_corrupted": self.n_total_corrupted,
            "detection_precision": self.detection_precision,
            "detection_recall": self.detection_recall,
            "detection_f1": self.detection_f1,
            "n_auto_repair": self.n_auto_repair,
            "n_correct_repairs": self.n_correct_repairs,
            "n_incorrect_repairs": self.n_incorrect_repairs,
            "repair_accuracy": self.repair_accuracy,
            "n_human_review": self.n_human_review,
            "n_abstain": self.n_abstain,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# RepairEvaluator
# ═══════════════════════════════════════════════════════════════════════════════


class RepairEvaluator:
    """
    Evaluate repair quality of BaselineRecords against ground truth.

    This is the ONLY class in this module that reads clean_df.
    clean_df is used ONLY to build the corruption ground-truth set and
    to look up clean values for individual cells.

    Parameters
    ----------
    clean_df : pd.DataFrame
        Ground-truth clean data.  Read-only; never modified.
    dirty_df : pd.DataFrame
        The dirty data used for detection/repair.  Read-only.
        Required to compute false negatives (corrupted cells not detected).
    dataset_name : str
    comparator : ValueComparator, optional
        Custom value comparison strategy.
    """

    def __init__(
        self,
        clean_df: pd.DataFrame,
        dirty_df: pd.DataFrame,
        dataset_name: str,
        comparator: Optional[ValueComparator] = None,
    ) -> None:
        self.clean_df = clean_df.reset_index(drop=True)
        self.dirty_df = dirty_df.reset_index(drop=True)
        self.dataset_name = dataset_name
        self.comparator = comparator or ValueComparator()
        self._gt_corrupted: Set[Tuple[int, str]] = self._build_gt_corrupted()
        logger.debug(
            "RepairEvaluator '%s': %d ground-truth corrupted cells.",
            dataset_name, len(self._gt_corrupted),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self, records: List[BaselineRecord]) -> RepairMetrics:
        """
        Evaluate a list of BaselineRecords against ground truth.

        Parameters
        ----------
        records : list[BaselineRecord]
            Output of a baseline or proposed method.  All records must
            belong to the same dataset as this evaluator.

        Returns
        -------
        RepairMetrics
        """
        method_name = records[0].method_name if records else "unknown"
        n_total_corrupted = len(self._gt_corrupted)

        detected_set: Set[Tuple[int, str]] = {
            (r.row_id, r.column) for r in records
        }

        n_detected = len(detected_set)
        n_tp = len(detected_set & self._gt_corrupted)
        n_fp = len(detected_set - self._gt_corrupted)
        n_fn = n_total_corrupted - n_tp

        precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
        recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        n_auto = n_correct = n_incorrect = 0
        n_human = n_abstain = 0

        for rec in records:
            action = rec.action
            if action == DecisionAction.AUTO_REPAIR:
                n_auto += 1
                clean_val = self._get_clean_value(rec.row_id, rec.column)
                if self.comparator.are_equal(rec.top_candidate, clean_val):
                    n_correct += 1
                else:
                    n_incorrect += 1
            elif action == DecisionAction.HUMAN_REVIEW:
                n_human += 1
            else:  # ABSTAIN
                n_abstain += 1

        repair_accuracy = n_correct / n_auto if n_auto > 0 else 0.0

        return RepairMetrics(
            method_name=method_name,
            dataset=self.dataset_name,
            n_detected=n_detected,
            n_truly_corrupted=n_tp,
            n_false_positive=n_fp,
            n_false_negative=n_fn,
            n_total_corrupted=n_total_corrupted,
            detection_precision=round(precision, 6),
            detection_recall=round(recall, 6),
            detection_f1=round(f1, 6),
            n_auto_repair=n_auto,
            n_correct_repairs=n_correct,
            n_incorrect_repairs=n_incorrect,
            repair_accuracy=round(repair_accuracy, 6),
            n_human_review=n_human,
            n_abstain=n_abstain,
        )

    def evaluate_cell(self, rec: BaselineRecord) -> CellResult:
        """
        Evaluate a single BaselineRecord against ground truth.

        Parameters
        ----------
        rec : BaselineRecord

        Returns
        -------
        CellResult
        """
        clean_val = self._get_clean_value(rec.row_id, rec.column)
        is_corrupted = (rec.row_id, rec.column) in self._gt_corrupted
        is_correct = self.comparator.are_equal(rec.top_candidate, clean_val)
        return CellResult(
            method_name=rec.method_name,
            row_id=rec.row_id,
            column=rec.column,
            dirty_value=rec.dirty_value,
            top_candidate=rec.top_candidate,
            clean_value=clean_val,
            action=rec.action,
            is_truly_corrupted=is_corrupted,
            is_repair_correct=is_correct,
        )

    # ── Internals ──────────────────────────────────────────────────────────────

    def _get_clean_value(self, row_id: int, column: str) -> Any:
        try:
            return self.clean_df.at[row_id, column]
        except (KeyError, IndexError):
            return None

    def _build_gt_corrupted(self) -> Set[Tuple[int, str]]:
        """
        Build the set of (row_id, column) pairs where clean ≠ dirty.

        Uses the same vectorized algorithm as DetectorEvaluator so that
        Phase 10 and Phase 3 ground-truth sets are consistent.
        NaN introduced by corruption is detected separately from value changes.
        Numeric dtype promotion (int→float64) is handled by numeric comparison.
        """
        corrupted: Set[Tuple[int, str]] = set()
        n_rows = min(len(self.clean_df), len(self.dirty_df))
        common_cols = [
            c for c in self.clean_df.columns
            if c in self.dirty_df.columns
        ]

        for col in common_cols:
            c_col = self.clean_df[col].iloc[:n_rows]
            d_col = self.dirty_df[col].iloc[:n_rows]

            clean_null = c_col.isna()
            dirty_null = d_col.isna()

            # NaN introduced in dirty
            introduced = (~clean_null) & dirty_null
            for row_id in c_col.index[introduced]:
                corrupted.add((int(row_id), col))

            # Value changed among non-null pairs
            both_present = ~clean_null & ~dirty_null
            c_num = pd.to_numeric(c_col[both_present], errors="coerce")
            d_num = pd.to_numeric(d_col[both_present], errors="coerce")
            both_numeric = c_num.notna() & d_num.notna()

            for row_id in c_num.index[both_numeric & (c_num != d_num)]:
                corrupted.add((int(row_id), col))

            str_mask = ~both_numeric & (
                c_col[both_present].astype(str) != d_col[both_present].astype(str)
            )
            for row_id in c_col[both_present].index[str_mask]:
                corrupted.add((int(row_id), col))

        return corrupted
