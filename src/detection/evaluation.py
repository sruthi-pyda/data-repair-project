"""
src/detection/evaluation.py
----------------------------
EVALUATION-ONLY utility for comparing detector output against clean ground truth.

PURPOSE AND DATA LEAKAGE BOUNDARY
===================================
This module is STRICTLY for measuring detector performance AFTER detection.

  DETECTION  (detector.py)  : dirty_df  →  suspicious cells
  EVALUATION (this file)    : suspicious cells + clean_df  →  accuracy metrics

clean_df is accessed HERE and only HERE.
It must never be imported into or passed to detector.py.

Why this separation matters:
  If clean values were available during detection, the detector could trivially
  compare dirty_df[row, col] against clean_df[row, col] and achieve 100%
  recall. That would defeat the purpose of the pipeline and produce misleading
  research results. The evaluation utility is only run AFTER detection to score
  how well the detector performed.

Definitions used
==================
  True Positive (TP)  — detector flagged a cell AND clean ≠ dirty (genuine error)
  False Positive (FP) — detector flagged a cell but clean == dirty (no real error)
  False Negative (FN) — a real error (clean ≠ dirty) that the detector missed

  Precision = TP / (TP + FP)
  Recall    = TP / (TP + FN)
  F1        = 2 * Precision * Recall / (Precision + Recall)

Usage
======
    from src.detection.evaluation import DetectorEvaluator

    evaluator = DetectorEvaluator(clean_df, dirty_df, dataset_name="adult")
    results = evaluator.evaluate(detection_records)
    evaluator.print_report(results)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from src.detection.detector import DetectionRecord

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """
    Summary of detector performance against ground truth.

    Attributes
    ----------
    dataset_name : str
    n_total_cells : int
        Total cells examined (n_rows × n_cols).
    n_actual_errors : int
        Number of cells where clean ≠ dirty (ground-truth corruptions).
    n_detected : int
        Number of cells flagged by the detector.
    tp : int
        True positives.
    fp : int
        False positives.
    fn : int
        False negatives (missed errors).
    precision : float
    recall : float
    f1 : float
    by_error_type : dict
        TP/FP breakdown by DetectionRecord.error_type.
    by_column : dict
        TP/FP/FN breakdown per column.
    """

    dataset_name: str
    n_total_cells: int
    n_actual_errors: int
    n_detected: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    by_error_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    by_column: Dict[str, Dict[str, int]] = field(default_factory=dict)


class DetectorEvaluator:
    """
    Compares detector output against clean ground truth.

    Parameters
    ----------
    clean_df : pd.DataFrame
        The ground-truth clean DataFrame. Used ONLY here.
    dirty_df : pd.DataFrame
        The dirty DataFrame (same that was passed to the detector).
    dataset_name : str
        Name of the dataset.
    numeric_tolerance : float
        Relative tolerance for comparing numerical values (to handle
        int vs float64 dtype differences caused by NaN injection).
        Default 1e-9 (effectively exact comparison for integers).
    """

    def __init__(
        self,
        clean_df: pd.DataFrame,
        dirty_df: pd.DataFrame,
        dataset_name: str,
        numeric_tolerance: float = 1e-9,
    ) -> None:
        self.clean_df = clean_df.reset_index(drop=True)
        self.dirty_df = dirty_df.reset_index(drop=True)
        self.dataset_name = dataset_name
        self.numeric_tolerance = numeric_tolerance

        # Pre-compute the ground-truth set of corrupted (row_id, column) pairs
        self._gt_corrupted: Set[Tuple[int, str]] = self._compute_ground_truth()
        logger.info(
            "DetectorEvaluator initialised for '%s': %d ground-truth corrupted cells.",
            dataset_name,
            len(self._gt_corrupted),
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def evaluate(self, records: List[DetectionRecord]) -> EvaluationResults:
        """
        Compare detection records against ground truth.

        Parameters
        ----------
        records : list[DetectionRecord]
            Output of Detector.detect().

        Returns
        -------
        EvaluationResults
        """
        detected_set: Set[Tuple[int, str]] = {
            (r.row_id, r.column) for r in records
        }

        tp_set = detected_set & self._gt_corrupted
        fp_set = detected_set - self._gt_corrupted
        fn_set = self._gt_corrupted - detected_set

        tp = len(tp_set)
        fp = len(fp_set)
        fn = len(fn_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # ── Breakdown by error type ──────────────────────────────────────
        by_error_type: Dict[str, Dict[str, int]] = {}
        for rec in records:
            et = rec.error_type
            if et not in by_error_type:
                by_error_type[et] = {"tp": 0, "fp": 0}
            pair = (rec.row_id, rec.column)
            if pair in self._gt_corrupted:
                by_error_type[et]["tp"] += 1
            else:
                by_error_type[et]["fp"] += 1

        # ── Breakdown by column ──────────────────────────────────────────
        all_cols = list(self.clean_df.columns)
        by_column: Dict[str, Dict[str, int]] = {}
        for col in all_cols:
            if col not in self.dirty_df.columns:
                continue
            col_detected = {r.row_id for r in records if r.column == col}
            col_gt = {row_id for (row_id, c) in self._gt_corrupted if c == col}
            col_tp = len(col_detected & col_gt)
            col_fp = len(col_detected - col_gt)
            col_fn = len(col_gt - col_detected)
            by_column[col] = {"tp": col_tp, "fp": col_fp, "fn": col_fn}

        total_cells = self.clean_df.shape[0] * self.clean_df.shape[1]

        return EvaluationResults(
            dataset_name=self.dataset_name,
            n_total_cells=total_cells,
            n_actual_errors=len(self._gt_corrupted),
            n_detected=len(detected_set),
            tp=tp,
            fp=fp,
            fn=fn,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            by_error_type=by_error_type,
            by_column=by_column,
        )

    @staticmethod
    def print_report(results: EvaluationResults) -> None:
        """Pretty-print evaluation results to stdout."""
        w = 60
        print(f"\n{'═' * w}")
        print(f"  DETECTION EVALUATION — {results.dataset_name.upper()}")
        print(f"{'─' * w}")
        print(f"  Total cells         : {results.n_total_cells:,}")
        print(f"  Ground-truth errors : {results.n_actual_errors:,}")
        print(f"  Detected (flagged)  : {results.n_detected:,}")
        print(f"{'─' * w}")
        print(f"  True Positives  (TP): {results.tp:,}")
        print(f"  False Positives (FP): {results.fp:,}")
        print(f"  False Negatives (FN): {results.fn:,}")
        print(f"{'─' * w}")
        print(f"  Precision : {results.precision:.4f}")
        print(f"  Recall    : {results.recall:.4f}")
        print(f"  F1 Score  : {results.f1:.4f}")
        if results.by_error_type:
            print(f"\n  By error type:")
            for et, counts in sorted(results.by_error_type.items()):
                print(f"    {et:<25} TP={counts['tp']:4d}  FP={counts['fp']:4d}")
        if results.by_column:
            print(f"\n  By column (non-zero):")
            for col, counts in sorted(results.by_column.items()):
                if any(v > 0 for v in counts.values()):
                    print(
                        f"    {col:<35} "
                        f"TP={counts['tp']:4d}  FP={counts['fp']:4d}  FN={counts['fn']:4d}"
                    )
        print(f"{'═' * w}\n")

    # ------------------------------------------------------------------ #
    # Internal: compute ground-truth corrupted cells                      #
    # ------------------------------------------------------------------ #

    def _compute_ground_truth(self) -> Set[Tuple[int, str]]:
        """
        Identify all (row_id, column) pairs where clean ≠ dirty.

        Uses numeric comparison where possible to avoid false positives from
        dtype changes (int 3 in clean vs float 3.0 in dirty after NaN injection).
        """
        corrupted: Set[Tuple[int, str]] = set()
        n_rows = min(len(self.clean_df), len(self.dirty_df))
        common_cols = [c for c in self.clean_df.columns if c in self.dirty_df.columns]

        for col in common_cols:
            c_col = self.clean_df[col].iloc[:n_rows]
            d_col = self.dirty_df[col].iloc[:n_rows]

            clean_null = c_col.isna()
            dirty_null = d_col.isna()

            # NaN introduced
            introduced = (~clean_null) & dirty_null
            for row_id in c_col.index[introduced]:
                corrupted.add((int(row_id), col))

            # Value changed — use numeric comparison when possible
            both_present = ~clean_null & ~dirty_null
            c_num = pd.to_numeric(c_col[both_present], errors="coerce")
            d_num = pd.to_numeric(d_col[both_present], errors="coerce")
            both_numeric = c_num.notna() & d_num.notna()

            num_diff_idx = c_num.index[both_numeric & (c_num != d_num)]
            for row_id in num_diff_idx:
                corrupted.add((int(row_id), col))

            str_diff_mask = ~both_numeric & (
                c_col[both_present].astype(str) != d_col[both_present].astype(str)
            )
            for row_id in c_col[both_present].index[str_diff_mask]:
                corrupted.add((int(row_id), col))

        return corrupted
