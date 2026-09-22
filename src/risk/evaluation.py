"""
src/risk/evaluation.py
-----------------------
Phase 7: Empirical cost evaluation for risk records.

This is the ONLY place in the risk phase where clean_df is accessed.
Empirical cost = average penalty actually incurred across all risk decisions,
comparing to ground truth from clean_df.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.risk.cost_model import Action, CostConfig, RiskRecord


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ActionCounts:
    """Count of decisions by action type."""
    accept: int = 0
    review: int = 0
    leave: int = 0

    @property
    def total(self) -> int:
        return self.accept + self.review + self.leave

    def fractions(self) -> Dict[str, float]:
        n = max(self.total, 1)
        return {
            "accept": self.accept / n,
            "review": self.review / n,
            "leave": self.leave / n,
        }


@dataclass
class EmpiricalCostReport:
    """
    Empirical cost evaluation results for one dataset.

    Attributes
    ----------
    dataset_name : str
    n_records : int
        Number of risk records evaluated.
    action_counts : ActionCounts
    n_correct_top1 : int
        Records where the top-1 repair candidate matches clean value.
    n_incorrect_top1 : int
        Records where top-1 repair is wrong.
    empirical_cost_accept_all : float
        Total cost if ACCEPT_REPAIR applied to all records.
    empirical_cost_review_all : float
        Total cost if HUMAN_REVIEW applied to all records.
    empirical_cost_leave_all : float
        Total cost if LEAVE_UNCHANGED applied to all records.
    empirical_cost_policy : float
        Total cost under the risk model's recommended actions.
    mean_cost_per_record_policy : float
        Empirical cost / n_records under policy.
    mean_cost_per_record_accept_all : float
    mean_cost_per_record_review_all : float
    mean_cost_per_record_leave_all : float
    cost_savings_vs_review_all : float
        empirical_cost_review_all - empirical_cost_policy
        (positive = policy cheaper than always-review)
    cost_savings_vs_leave_all : float
    costs : CostConfig
    limitation : str
        Dataset-specific note (e.g. Adult FP limitation).
    n_genuinely_corrupted : int
        Records whose dirty value differs from the clean value. Only these
        incur C_leave when left unchanged; a false-positive detection left
        alone costs nothing. Keeps this evaluator consistent with
        E[LEAVE_UNCHANGED] = q · C_leave in RiskCalculator and with the
        Phase 10 CostEvaluator.
    """
    dataset_name: str
    n_records: int
    action_counts: ActionCounts
    n_correct_top1: int
    n_incorrect_top1: int
    empirical_cost_accept_all: float
    empirical_cost_review_all: float
    empirical_cost_leave_all: float
    empirical_cost_policy: float
    mean_cost_per_record_policy: float
    mean_cost_per_record_accept_all: float
    mean_cost_per_record_review_all: float
    mean_cost_per_record_leave_all: float
    cost_savings_vs_review_all: float
    cost_savings_vs_leave_all: float
    costs: CostConfig
    limitation: str = ""
    n_genuinely_corrupted: int = 0


# ---------------------------------------------------------------------------
# RiskEvaluator
# ---------------------------------------------------------------------------

_ADULT_LIMITATION = (
    "Adult detections are statistical false positives (fnlwgt IQR outliers). "
    "True corruption (income label flips) is undetectable from dirty data. "
    "All records are incorrect repairs; empirical cost for ACCEPT_ALL = "
    "n * C_wrong_repair."
)

_SKIP_COLUMNS_BC = {"Unnamed: 0", "Sample code number"}


class RiskEvaluator:
    """
    Evaluates the empirical cost of risk-model decisions against ground truth.

    Only this class accesses clean_df in Phase 7.

    Parameters
    ----------
    clean_df : pd.DataFrame
        Ground-truth data for the dataset.
    dirty_df : pd.DataFrame
        Dirty data used during repair.
    dataset_name : str
    """

    def __init__(
        self,
        clean_df: pd.DataFrame,
        dirty_df: pd.DataFrame,
        dataset_name: str,
    ):
        self.clean_df = clean_df
        self.dirty_df = dirty_df
        self.dataset_name = dataset_name

    def evaluate(
        self,
        risk_records: List[RiskRecord],
        costs: CostConfig,
    ) -> EmpiricalCostReport:
        """
        Compute empirical costs for a list of risk records.

        For each record, ground truth is looked up from clean_df.
        Correctness: top_candidate matches clean value (numeric tolerance 1e-9
        for float comparisons, string equality otherwise).

        Parameters
        ----------
        risk_records : list[RiskRecord]
        costs : CostConfig
            Used to compute per-action penalty values.

        Returns
        -------
        EmpiricalCostReport
        """
        n = len(risk_records)
        if n == 0:
            return self._empty_report(costs)

        action_counts = ActionCounts()
        n_correct = 0
        n_incorrect = 0
        n_genuinely_corrupted = 0
        total_policy_cost = 0.0

        for rr in risk_records:
            # Ground truth lookup
            correct = self._is_correct(rr)
            corrupted = self._is_genuinely_corrupted(rr)
            if corrupted:
                n_genuinely_corrupted += 1
            if correct:
                n_correct += 1
            else:
                n_incorrect += 1

            # Tally action
            action = rr.lowest_cost_action
            if action == Action.ACCEPT_REPAIR:
                action_counts.accept += 1
                # Correct repair costs 0; incorrect costs C_wrong_repair
                total_policy_cost += 0.0 if correct else costs.c_wrong_repair
            elif action == Action.HUMAN_REVIEW:
                action_counts.review += 1
                total_policy_cost += costs.c_human_review
            else:  # LEAVE_UNCHANGED
                action_counts.leave += 1
                # Leaving a genuinely corrupted cell costs C_leave; leaving a
                # false-positive detection alone is the right outcome and
                # costs nothing (mirrors E[LEAVE] = q * C_leave).
                if corrupted:
                    total_policy_cost += costs.c_leave_error

        # Baselines: cost if we applied one action to every record
        # ACCEPT_ALL: incorrect records cost C_wrong, correct cost 0
        cost_accept_all = n_incorrect * costs.c_wrong_repair
        # REVIEW_ALL: every record costs C_human_review
        cost_review_all = n * costs.c_human_review
        # LEAVE_ALL: only genuinely corrupted records cost C_leave
        cost_leave_all = n_genuinely_corrupted * costs.c_leave_error

        limitation = _ADULT_LIMITATION if self.dataset_name == "adult" else ""

        return EmpiricalCostReport(
            dataset_name=self.dataset_name,
            n_records=n,
            action_counts=action_counts,
            n_correct_top1=n_correct,
            n_incorrect_top1=n_incorrect,
            empirical_cost_accept_all=cost_accept_all,
            empirical_cost_review_all=cost_review_all,
            empirical_cost_leave_all=cost_leave_all,
            empirical_cost_policy=total_policy_cost,
            mean_cost_per_record_policy=total_policy_cost / n,
            mean_cost_per_record_accept_all=cost_accept_all / n,
            mean_cost_per_record_review_all=cost_review_all / n,
            mean_cost_per_record_leave_all=cost_leave_all / n,
            cost_savings_vs_review_all=cost_review_all - total_policy_cost,
            cost_savings_vs_leave_all=cost_leave_all - total_policy_cost,
            costs=costs,
            limitation=limitation,
            n_genuinely_corrupted=n_genuinely_corrupted,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _is_genuinely_corrupted(self, rr: RiskRecord) -> bool:
        """
        True if the dirty cell differs from the clean cell (NaN == NaN).

        Uses dirty_df rather than any value on the record so the check is
        independent of what the repairer proposed.
        """
        try:
            clean_val = self.clean_df.at[rr.row_id, rr.column]
            dirty_val = self.dirty_df.at[rr.row_id, rr.column]
        except (KeyError, IndexError):
            return False
        c_na = isinstance(clean_val, float) and clean_val != clean_val
        d_na = isinstance(dirty_val, float) and dirty_val != dirty_val
        if c_na and d_na:
            return False
        if c_na or d_na:
            return True
        try:
            return abs(float(clean_val) - float(dirty_val)) >= 1e-9
        except (TypeError, ValueError):
            return str(clean_val).strip() != str(dirty_val).strip()

    def _is_correct(self, rr: RiskRecord) -> bool:
        """Check whether top_candidate matches the clean value for this cell."""
        try:
            clean_val = self.clean_df.at[rr.row_id, rr.column]
        except (KeyError, IndexError):
            return False

        candidate = rr.top_candidate
        if candidate is None:
            return False

        # Numeric comparison
        try:
            c_float = float(clean_val)
            p_float = float(candidate)
            return abs(c_float - p_float) < 1e-9
        except (TypeError, ValueError):
            pass

        # String comparison
        return str(candidate).strip() == str(clean_val).strip()

    def _empty_report(self, costs: CostConfig) -> EmpiricalCostReport:
        return EmpiricalCostReport(
            dataset_name=self.dataset_name,
            n_records=0,
            action_counts=ActionCounts(),
            n_correct_top1=0,
            n_incorrect_top1=0,
            empirical_cost_accept_all=0.0,
            empirical_cost_review_all=0.0,
            empirical_cost_leave_all=0.0,
            empirical_cost_policy=0.0,
            mean_cost_per_record_policy=0.0,
            mean_cost_per_record_accept_all=0.0,
            mean_cost_per_record_review_all=0.0,
            mean_cost_per_record_leave_all=0.0,
            cost_savings_vs_review_all=0.0,
            cost_savings_vs_leave_all=0.0,
            costs=costs,
        )


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def print_risk_report(report: EmpiricalCostReport) -> None:
    """Print a human-readable empirical cost report."""
    w = 72
    ds = report.dataset_name.upper()
    print(f"\n{'─' * w}")
    print(f"  EMPIRICAL COST REPORT — {ds}")
    print(f"{'─' * w}")
    print(f"  Records evaluated : {report.n_records}")
    print(f"  Top-1 correct     : {report.n_correct_top1}  "
          f"({100*report.n_correct_top1/max(report.n_records,1):.1f}%)")
    print(f"  Top-1 incorrect   : {report.n_incorrect_top1}  "
          f"({100*report.n_incorrect_top1/max(report.n_records,1):.1f}%)")

    fracs = report.action_counts.fractions()
    print(f"\n  Action distribution:")
    print(f"    ACCEPT_REPAIR  : {report.action_counts.accept:>5}  "
          f"({100*fracs['accept']:5.1f}%)")
    print(f"    HUMAN_REVIEW   : {report.action_counts.review:>5}  "
          f"({100*fracs['review']:5.1f}%)")
    print(f"    LEAVE_UNCHANGED: {report.action_counts.leave:>5}  "
          f"({100*fracs['leave']:5.1f}%)")
    print(f"  Genuinely corrupted (dirty != clean): {report.n_genuinely_corrupted} "
          f"of {report.n_records}  (only these cost C_leave when left)")

    print(f"\n  Cost parameters: wrong={report.costs.c_wrong_repair}, "
          f"review={report.costs.c_human_review}, "
          f"leave={report.costs.c_leave_error}")

    print(f"\n  {'Strategy':<20} {'Total cost':>12} {'Mean/record':>12}")
    print(f"  {'─'*20} {'─'*12} {'─'*12}")
    print(f"  {'Policy (risk model)':<20} "
          f"{report.empirical_cost_policy:>12.2f} "
          f"{report.mean_cost_per_record_policy:>12.4f}")
    print(f"  {'Always ACCEPT':<20} "
          f"{report.empirical_cost_accept_all:>12.2f} "
          f"{report.mean_cost_per_record_accept_all:>12.4f}")
    print(f"  {'Always REVIEW':<20} "
          f"{report.empirical_cost_review_all:>12.2f} "
          f"{report.mean_cost_per_record_review_all:>12.4f}")
    print(f"  {'Always LEAVE':<20} "
          f"{report.empirical_cost_leave_all:>12.2f} "
          f"{report.mean_cost_per_record_leave_all:>12.4f}")

    savings_r = report.cost_savings_vs_review_all
    savings_l = report.cost_savings_vs_leave_all
    sign_r = "saves" if savings_r >= 0 else "costs extra"
    sign_l = "saves" if savings_l >= 0 else "costs extra"
    print(f"\n  Policy vs always-REVIEW : {abs(savings_r):.2f} units {sign_r}")
    print(f"  Policy vs always-LEAVE  : {abs(savings_l):.2f} units {sign_l}")

    if report.limitation:
        print(f"\n  [NOTE] {report.limitation}")
    print(f"{'─' * w}")
