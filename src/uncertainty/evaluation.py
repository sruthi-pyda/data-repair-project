"""
src/uncertainty/evaluation.py
------------------------------
EVALUATION-ONLY analysis of uncertainty signal quality.

PURPOSE AND DATA LEAKAGE BOUNDARY
===================================
This module is the ONLY place where clean_df is used in the uncertainty phase.

  ESTIMATION  (estimator.py) : repair results → UncertaintyRecord (no clean_df)
  EVALUATION  (this file)    : UncertaintyRecord + clean_df → correlation analysis

The key question answered here:
  "Do cells with higher raw confidence / higher self-consistency / lower entropy
   tend to be repaired correctly?"

If uncertainty signals correlate with correctness, then:
  - A calibration model (Phase 6) can use them to estimate P(correct)
  - The decision policy (Phase 8) can use calibrated probabilities to decide
    AUTO / HUMAN_REVIEW / ABSTAIN

Usage
======
    from src.uncertainty.evaluation import UncertaintyCorrelationAnalysis

    analysis = UncertaintyCorrelationAnalysis(clean_df, dirty_df, "breast_cancer")
    report = analysis.analyse(uncertainty_records)
    analysis.print_report(report)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.uncertainty.estimator import UncertaintyRecord

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Output types
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BinStats:
    """Accuracy statistics for one bin of an uncertainty signal."""

    bin_label: str
    n_cells: int
    n_correct: int
    accuracy: float


@dataclass
class CorrelationReport:
    """
    Post-hoc analysis of how uncertainty signals relate to repair correctness.

    All correctness labels come from clean_df — ground truth only used here.

    Attributes
    ----------
    dataset_name : str
    n_total : int
        Total records analysed.
    n_correct_top1 : int
        Records where top_candidate matched the clean value.
    overall_top1_accuracy : float
    by_confidence_bin : list[BinStats]
        Top-1 accuracy in quartile bins of raw_confidence.
    by_consistency_bin : list[BinStats]
        Top-1 accuracy in quartile bins of self_consistency.
    by_entropy_bin : list[BinStats]
        Top-1 accuracy in quartile bins of normalized_entropy.
    pearson_conf_accuracy : float or None
        Pearson correlation between raw_confidence and correctness (0/1).
    pearson_cons_accuracy : float or None
        Pearson correlation between self_consistency and correctness.
    pearson_entropy_accuracy : float or None
        Pearson correlation between normalized_entropy and correctness.
        (Expected to be negative: higher entropy → less likely correct.)
    """

    dataset_name: str
    n_total: int
    n_correct_top1: int
    overall_top1_accuracy: float
    by_confidence_bin: List[BinStats] = field(default_factory=list)
    by_consistency_bin: List[BinStats] = field(default_factory=list)
    by_entropy_bin: List[BinStats] = field(default_factory=list)
    pearson_conf_accuracy: Optional[float] = None
    pearson_cons_accuracy: Optional[float] = None
    pearson_entropy_accuracy: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis class
# ═══════════════════════════════════════════════════════════════════════════════


class UncertaintyCorrelationAnalysis:
    """
    Evaluates whether uncertainty signals predict repair correctness.

    Parameters
    ----------
    clean_df : pd.DataFrame
        Ground-truth clean data. Used ONLY here.
    dirty_df : pd.DataFrame
        The dirty DataFrame.
    dataset_name : str
    numeric_tolerance : float
        Tolerance for numeric comparison (handles int vs float64 mismatches).
    n_bins : int
        Number of equal-frequency bins for each signal (default 4 = quartiles).
    """

    def __init__(
        self,
        clean_df: pd.DataFrame,
        dirty_df: pd.DataFrame,
        dataset_name: str,
        numeric_tolerance: float = 1e-9,
        n_bins: int = 4,
    ) -> None:
        self.clean_df = clean_df.reset_index(drop=True)
        self.dirty_df = dirty_df.reset_index(drop=True)
        self.dataset_name = dataset_name
        self.numeric_tolerance = numeric_tolerance
        self.n_bins = n_bins

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def analyse(self, records: List[UncertaintyRecord]) -> CorrelationReport:
        """
        Analyse the relationship between uncertainty signals and correctness.

        Parameters
        ----------
        records : list[UncertaintyRecord]
            Output of UncertaintyEstimator.estimate_batch(). NO clean values inside.

        Returns
        -------
        CorrelationReport
        """
        if not records:
            return CorrelationReport(
                dataset_name=self.dataset_name,
                n_total=0,
                n_correct_top1=0,
                overall_top1_accuracy=0.0,
            )

        # Label each record as correct (1) or not (0) using clean_df
        correctness: List[int] = []
        for rec in records:
            c_val = self._get_clean_value(rec.row_id, rec.column)
            correct = int(self._is_correct(rec.top_candidate, c_val))
            correctness.append(correct)

        n_total = len(records)
        n_correct = sum(correctness)
        overall_acc = round(n_correct / n_total, 4) if n_total > 0 else 0.0

        # ── Bin analysis ─────────────────────────────────────────────────
        confidences = [r.raw_confidence for r in records]
        consistencies = [r.self_consistency for r in records]
        entropies = [r.normalized_entropy for r in records]

        conf_bins = _bin_analysis(confidences, correctness, self.n_bins, "confidence")
        cons_bins = _bin_analysis(consistencies, correctness, self.n_bins, "consistency")
        entr_bins = _bin_analysis(entropies, correctness, self.n_bins, "norm_entropy")

        # ── Pearson correlations ──────────────────────────────────────────
        p_conf = _pearson(confidences, correctness)
        p_cons = _pearson(consistencies, correctness)
        p_entr = _pearson(entropies, correctness)

        logger.info(
            "CorrelationAnalysis '%s': n=%d acc=%.3f "
            "r(conf)=%.3f r(cons)=%.3f r(entr)=%.3f",
            self.dataset_name, n_total, overall_acc,
            p_conf or 0, p_cons or 0, p_entr or 0,
        )

        return CorrelationReport(
            dataset_name=self.dataset_name,
            n_total=n_total,
            n_correct_top1=n_correct,
            overall_top1_accuracy=overall_acc,
            by_confidence_bin=conf_bins,
            by_consistency_bin=cons_bins,
            by_entropy_bin=entr_bins,
            pearson_conf_accuracy=p_conf,
            pearson_cons_accuracy=p_cons,
            pearson_entropy_accuracy=p_entr,
        )

    @staticmethod
    def print_report(report: CorrelationReport) -> None:
        """Pretty-print correlation analysis to stdout."""
        w = 68
        print(f"\n{'═' * w}")
        print(f"  UNCERTAINTY CORRELATION ANALYSIS — {report.dataset_name.upper()}")
        print(f"{'─' * w}")
        print(f"  Total records          : {report.n_total:,}")
        print(f"  Top-1 correct          : {report.n_correct_top1:,}")
        print(f"  Overall top-1 accuracy : {report.overall_top1_accuracy:.4f}")
        print(f"{'─' * w}")

        def _print_bins(bins: List[BinStats], label: str) -> None:
            if not bins:
                return
            print(f"\n  {label} bins (accuracy per quartile):")
            for b in bins:
                bar = "█" * int(b.accuracy * 20)
                print(
                    f"    {b.bin_label:<20} n={b.n_cells:4d}  "
                    f"acc={b.accuracy:.3f}  {bar}"
                )

        _print_bins(report.by_confidence_bin, "Raw confidence")
        _print_bins(report.by_consistency_bin, "Self-consistency")
        _print_bins(report.by_entropy_bin, "Norm. entropy (lower → more certain)")

        print(f"\n  Pearson r (signal ↔ correctness):")
        print(f"    raw_confidence    : {_fmt_r(report.pearson_conf_accuracy)}")
        print(f"    self_consistency  : {_fmt_r(report.pearson_cons_accuracy)}")
        print(f"    normalized_entropy: {_fmt_r(report.pearson_entropy_accuracy)}  "
              f"(expect negative)")
        print(f"{'═' * w}\n")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_clean_value(self, row_id: int, column: str) -> Any:
        if column not in self.clean_df.columns:
            return None
        if row_id >= len(self.clean_df):
            return None
        return self.clean_df.at[row_id, column]

    def _is_correct(self, candidate_val: Any, clean_val: Any) -> bool:
        if clean_val is None or (isinstance(clean_val, float) and pd.isna(clean_val)):
            return candidate_val is None or (
                isinstance(candidate_val, float) and pd.isna(candidate_val)
            )
        if candidate_val is None:
            return False
        try:
            c_num = float(clean_val)
            r_num = float(candidate_val)
            if abs(c_num - r_num) <= self.numeric_tolerance * max(1.0, abs(c_num)):
                return True
        except (TypeError, ValueError):
            pass
        return str(candidate_val).strip() == str(clean_val).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _bin_analysis(
    signal: List[float],
    correctness: List[int],
    n_bins: int,
    label: str,
) -> List[BinStats]:
    """
    Divide signal into n_bins equal-frequency bins, compute accuracy per bin.
    """
    if not signal:
        return []

    n = len(signal)
    indexed = sorted(enumerate(signal), key=lambda x: x[1])
    bin_size = max(1, n // n_bins)
    bins_out: List[BinStats] = []

    for b in range(n_bins):
        start = b * bin_size
        end = start + bin_size if b < n_bins - 1 else n
        slice_idxs = [idx for idx, _ in indexed[start:end]]

        if not slice_idxs:
            continue

        vals_in_bin = [signal[i] for i in slice_idxs]
        n_cells = len(slice_idxs)
        n_correct = sum(correctness[i] for i in slice_idxs)
        acc = round(n_correct / n_cells, 4) if n_cells > 0 else 0.0
        bin_min = min(vals_in_bin)
        bin_max = max(vals_in_bin)
        bin_label = f"[{bin_min:.2f}, {bin_max:.2f}]"

        bins_out.append(BinStats(
            bin_label=bin_label,
            n_cells=n_cells,
            n_correct=n_correct,
            accuracy=acc,
        ))

    return bins_out


def _pearson(x: List[float], y: List[int]) -> Optional[float]:
    """Pearson correlation coefficient between x (float) and y (0/1 int)."""
    n = len(x)
    if n < 2:
        return None
    x_arr = x
    y_arr = [float(v) for v in y]
    mx = sum(x_arr) / n
    my = sum(y_arr) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x_arr, y_arr))
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x_arr))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y_arr))
    if sx < 1e-12 or sy < 1e-12:
        return None
    return round(num / (sx * sy), 4)


def _fmt_r(r: Optional[float]) -> str:
    if r is None:
        return "N/A"
    return f"{r:+.4f}"
