"""
src/repair/evaluation.py
-------------------------
EVALUATION-ONLY utility for measuring repair quality against clean ground truth.

PURPOSE AND DATA LEAKAGE BOUNDARY
===================================
This module is the ONLY place where clean_df is compared against repair output.

  REPAIR    (mock_repair.py) : dirty_df  →  RepairResult candidates
  EVALUATION (this file)     : RepairResult + clean_df  →  accuracy metrics

clean_df is accessed HERE and only HERE (in the repair phase).

Definitions
===========
  Top-1 accuracy : fraction of repaired cells where candidates[0].value == clean value
  Top-k accuracy : fraction where ANY of the top-k candidates == clean value
  Weighted accuracy : sum of candidate.probability for the correct value (where found)

Usage
======
    from src.repair.evaluation import RepairEvaluator

    evaluator = RepairEvaluator(clean_df, dirty_df, dataset_name="breast_cancer")
    results = evaluator.evaluate(repair_results)
    evaluator.print_report(results)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.repair.mock_repair import RepairResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Results dataclass
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RepairEvaluationResults:
    """
    Summary of repair accuracy against ground truth.

    Attributes
    ----------
    dataset_name : str
    n_repaired : int
        Total cells for which repair was attempted.
    top1_correct : int
        Cells where the top candidate matched the clean value.
    topk_correct : int
        Cells where ANY candidate matched the clean value.
    top1_accuracy : float
        top1_correct / n_repaired
    topk_accuracy : float
        topk_correct / n_repaired
    mean_correct_prob : float
        Mean probability assigned to the correct value across all repaired cells
        (0.0 for cells where correct value was not in candidates).
    by_method : dict
        top-1 accuracy breakdown per repair_method.
    by_error_type : dict
        top-1 accuracy breakdown per error_type.
    by_column : dict
        top-1 accuracy breakdown per column.
    """

    dataset_name: str
    n_repaired: int
    top1_correct: int
    topk_correct: int
    top1_accuracy: float
    topk_accuracy: float
    mean_correct_prob: float
    by_method: Dict[str, Dict[str, int]] = field(default_factory=dict)
    by_error_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    by_column: Dict[str, Dict[str, int]] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluator
# ═══════════════════════════════════════════════════════════════════════════════


class RepairEvaluator:
    """
    Compares repair output against clean ground truth.

    Parameters
    ----------
    clean_df : pd.DataFrame
        The ground-truth clean DataFrame. Used ONLY here.
    dirty_df : pd.DataFrame
        The dirty DataFrame (same that was passed to the repairer).
    dataset_name : str
    numeric_tolerance : float
        Tolerance for numerical comparison (to handle int vs float64 dtype).
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

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def evaluate(self, results: List[RepairResult]) -> RepairEvaluationResults:
        """
        Evaluate repair candidates against clean ground truth.

        Parameters
        ----------
        results : list[RepairResult]
            Output of MockRepairer.repair_batch().

        Returns
        -------
        RepairEvaluationResults
        """
        if not results:
            return RepairEvaluationResults(
                dataset_name=self.dataset_name,
                n_repaired=0,
                top1_correct=0,
                topk_correct=0,
                top1_accuracy=0.0,
                topk_accuracy=0.0,
                mean_correct_prob=0.0,
            )

        top1_correct = 0
        topk_correct = 0
        correct_probs: List[float] = []

        by_method: Dict[str, Dict[str, int]] = {}
        by_error_type: Dict[str, Dict[str, int]] = {}
        by_column: Dict[str, Dict[str, int]] = {}

        for res in results:
            clean_val = self._get_clean_value(res.row_id, res.column)

            # Top-1 and top-k correctness
            t1 = self._is_correct(res.candidates[0].value, clean_val) if res.candidates else False
            tk = any(self._is_correct(c.value, clean_val) for c in res.candidates)

            if t1:
                top1_correct += 1
            if tk:
                topk_correct += 1

            # Probability assigned to the correct value
            prob_correct = 0.0
            for c in res.candidates:
                if self._is_correct(c.value, clean_val):
                    prob_correct = c.probability
                    break
            correct_probs.append(prob_correct)

            # Breakdown by repair method
            m = res.repair_method
            if m not in by_method:
                by_method[m] = {"top1_correct": 0, "total": 0}
            by_method[m]["total"] += 1
            if t1:
                by_method[m]["top1_correct"] += 1

            # Breakdown by error type
            et = res.error_type
            if et not in by_error_type:
                by_error_type[et] = {"top1_correct": 0, "topk_correct": 0, "total": 0}
            by_error_type[et]["total"] += 1
            if t1:
                by_error_type[et]["top1_correct"] += 1
            if tk:
                by_error_type[et]["topk_correct"] += 1

            # Breakdown by column
            col = res.column
            if col not in by_column:
                by_column[col] = {"top1_correct": 0, "topk_correct": 0, "total": 0}
            by_column[col]["total"] += 1
            if t1:
                by_column[col]["top1_correct"] += 1
            if tk:
                by_column[col]["topk_correct"] += 1

        n = len(results)
        top1_acc = round(top1_correct / n, 4) if n > 0 else 0.0
        topk_acc = round(topk_correct / n, 4) if n > 0 else 0.0
        mean_prob = round(sum(correct_probs) / len(correct_probs), 4) if correct_probs else 0.0

        logger.info(
            "RepairEvaluator '%s': n=%d  top-1=%.3f  top-k=%.3f",
            self.dataset_name, n, top1_acc, topk_acc,
        )

        return RepairEvaluationResults(
            dataset_name=self.dataset_name,
            n_repaired=n,
            top1_correct=top1_correct,
            topk_correct=topk_correct,
            top1_accuracy=top1_acc,
            topk_accuracy=topk_acc,
            mean_correct_prob=mean_prob,
            by_method=by_method,
            by_error_type=by_error_type,
            by_column=by_column,
        )

    @staticmethod
    def print_report(results: RepairEvaluationResults) -> None:
        """Pretty-print repair evaluation results to stdout."""
        w = 64
        print(f"\n{'═' * w}")
        print(f"  REPAIR EVALUATION — {results.dataset_name.upper()}")
        print(f"{'─' * w}")
        print(f"  Cells repaired          : {results.n_repaired:,}")
        print(f"{'─' * w}")
        print(f"  Top-1 correct           : {results.top1_correct:,}  ({results.top1_accuracy:.4f})")
        print(f"  Top-k correct           : {results.topk_correct:,}  ({results.topk_accuracy:.4f})")
        print(f"  Mean prob(correct value): {results.mean_correct_prob:.4f}")

        if results.by_error_type:
            print(f"\n  By error type:")
            for et, counts in sorted(results.by_error_type.items()):
                t = counts["total"]
                t1 = counts["top1_correct"]
                tk = counts["topk_correct"]
                acc1 = t1 / t if t > 0 else 0.0
                acck = tk / t if t > 0 else 0.0
                print(
                    f"    {et:<25} total={t:4d}  top-1={acc1:.3f}  top-k={acck:.3f}"
                )

        if results.by_method:
            print(f"\n  By repair method:")
            for m, counts in sorted(results.by_method.items()):
                t = counts["total"]
                t1 = counts["top1_correct"]
                acc1 = t1 / t if t > 0 else 0.0
                print(f"    {m:<35} total={t:4d}  top-1={acc1:.3f}")

        if results.by_column:
            print(f"\n  By column (non-zero activity):")
            for col, counts in sorted(results.by_column.items()):
                if counts["total"] > 0:
                    t = counts["total"]
                    acc1 = counts["top1_correct"] / t
                    acck = counts["topk_correct"] / t
                    print(
                        f"    {col:<35} total={t:4d}  top-1={acc1:.3f}  top-k={acck:.3f}"
                    )
        print(f"{'═' * w}\n")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_clean_value(self, row_id: int, column: str) -> Any:
        """Retrieve the clean value for a given cell."""
        if column not in self.clean_df.columns:
            return None
        if row_id >= len(self.clean_df):
            return None
        return self.clean_df.at[row_id, column]

    def _is_correct(self, candidate_val: Any, clean_val: Any) -> bool:
        """
        Check whether a candidate matches the clean value.

        Uses numeric comparison when both parse as numbers to avoid
        int-vs-float false negatives.
        """
        if clean_val is None or (isinstance(clean_val, float) and pd.isna(clean_val)):
            return candidate_val is None or (
                isinstance(candidate_val, float) and pd.isna(candidate_val)
            )

        if candidate_val is None:
            return False

        # Try numeric comparison
        try:
            c_num = float(clean_val)
            r_num = float(candidate_val)
            if abs(c_num - r_num) <= self.numeric_tolerance * max(1.0, abs(c_num)):
                return True
        except (TypeError, ValueError):
            pass

        # Fall back to string comparison
        return str(candidate_val).strip() == str(clean_val).strip()
