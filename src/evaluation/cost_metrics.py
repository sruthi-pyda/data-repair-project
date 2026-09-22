"""
src/evaluation/cost_metrics.py
--------------------------------
Phase 10: Realized cost evaluation and policy comparison.

EXPECTED COST vs REALIZED COST
================================

EXPECTED COST (pre-decision)
-----------------------------
Computed by the policy BEFORE knowing whether the repair is actually correct.
  E[Cost_auto]    = (1 − p) × C_wrong
  E[Cost_human]   = C_human
  E[Cost_abstain] = q × C_abstain      q = P(cell is genuinely corrupted)
                                       (Phase 8 default q = 1.0)

Stored in BaselineRecord.expected_cost_{auto,human,abstain} for the
proposed method.  None for baselines that do not use cost-aware routing.

REALIZED COST (post-evaluation)
--------------------------------
Computed AFTER comparing decisions against ground truth (clean_df).

  AUTO_REPAIR + repair correct   →  0
  AUTO_REPAIR + repair incorrect →  C_wrong
  HUMAN_REVIEW                   →  C_human   (regardless of correctness;
                                               the reviewer still worked)
  ABSTAIN + cell truly corrupted →  C_abstain (the error persists)
  ABSTAIN + cell was a false alarm → 0        (leaving a correct cell alone
                                               is the right outcome)

Total realized cost = Σ realized_cost(action_i)

This is the number that matters for comparing policies.

WHY ABSTAIN IS CHARGED CONDITIONALLY
-------------------------------------
The policy's expected abstain cost is q × C_abstain, where q is the
probability that the flagged cell is genuinely corrupted (learned by
src/calibration/detection_calibrator.py).  The realised cost must be the
outcome that expectation is *over*: C_abstain when the cell really was
corrupted, 0 when the detector raised a false alarm.  Charging C_abstain
unconditionally would penalise the policy for correctly ignoring false
alarms — exactly the behaviour a risk-aware policy is supposed to exhibit
on datasets such as Adult, where every detection is a statistical
false positive.  When every flagged cell is truly corrupted (q = 1) the
two rules coincide, so previously reported numbers on such batches are
unchanged.

"Genuinely corrupted" is decided with the same ValueComparator used for
repair correctness: dirty_value ≠ clean value (NaN == NaN, numeric
tolerance).  A record whose column/row is absent from clean_df is treated
as corrupted (conservative: the abstention is charged).

DATA-LEAKAGE BOUNDARY
======================
CostEvaluator accesses clean_df ONLY to determine (a) repair correctness
for AUTO_REPAIR cells and (b) whether a cell was genuinely corrupted, for
the ABSTAIN charge and the n_genuinely_corrupted count.  Both are
post-decision scoring; clean_df is never passed upstream.

COST PARAMETERS
===============
Accept a CostConfig from Phase 7, or explicit float values.
Mapping:
  CostConfig.c_wrong_repair  → C_wrong
  CostConfig.c_human_review  → C_human
  CostConfig.c_leave_error   → C_abstain
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from src.baselines.baselines import BaselineRecord, DEFAULT_THRESHOLDS
from src.decision.policy import DecisionAction
from src.evaluation.repair_metrics import ValueComparator
from src.risk.cost_model import CostConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CostReport
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CostReport:
    """
    Realized cost evaluation for one method on one dataset.

    EXPECTED vs REALIZED (see module docstring for definitions).

    Attributes
    ----------
    method_name : str
    dataset : str
    n_records : int

    Action counts
    -------------
    n_auto : int
    n_human : int
    n_abstain : int
    auto_rate : float
    human_rate : float
    abstain_rate : float

    Repair outcome
    --------------
    n_correct_auto : int
        AUTO_REPAIR decisions with correct repair.
    n_wrong_auto : int
        AUTO_REPAIR decisions with incorrect repair.
    wrong_auto_rate : float
        n_wrong_auto / n_auto  (0.0 if n_auto == 0)

    Realized cost
    -------------
    total_realized_cost : float
    mean_cost_per_record : float
    cost_per_wrong_auto : float
        C_wrong parameter used.
    cost_per_human : float
        C_human parameter used.
    cost_per_abstain : float
        C_abstain parameter used.

    Abstention outcome (added with calibrated abstention; default 0 so
    older constructors keep working)
    -----------------------------------------------------------------
    n_abstain_on_corrupted : int
        ABSTAIN decisions on cells that were genuinely corrupted
        (charged C_abstain each).
    n_abstain_on_clean : int
        ABSTAIN decisions on false alarms — dirty == clean (charged 0).
    n_genuinely_corrupted : int
        Records (any action) whose dirty value differs from the clean
        value.  Useful for the "always leave" reference cost
        n_genuinely_corrupted × C_abstain.
    abstain_precision : float  (property)
        n_abstain_on_clean / n_abstain — the fraction of abstentions that
        were the right call.  0.0 when n_abstain == 0.
    """

    method_name: str
    dataset: str
    n_records: int

    # Action distribution
    n_auto: int = 0
    n_human: int = 0
    n_abstain: int = 0
    auto_rate: float = 0.0
    human_rate: float = 0.0
    abstain_rate: float = 0.0

    # Repair correctness (among AUTO actions)
    n_correct_auto: int = 0
    n_wrong_auto: int = 0
    wrong_auto_rate: float = 0.0

    # Realized cost
    total_realized_cost: float = 0.0
    mean_cost_per_record: float = 0.0

    # Cost parameters used
    cost_per_wrong_auto: float = 0.0
    cost_per_human: float = 0.0
    cost_per_abstain: float = 0.0

    # Abstention outcome (appended; defaults keep old constructors valid)
    n_abstain_on_corrupted: int = 0
    n_abstain_on_clean: int = 0
    n_genuinely_corrupted: int = 0

    @property
    def abstain_precision(self) -> float:
        """Fraction of ABSTAIN decisions that landed on a false alarm."""
        if self.n_abstain <= 0:
            return 0.0
        return self.n_abstain_on_clean / self.n_abstain

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "dataset": self.dataset,
            "n_records": self.n_records,
            "n_auto": self.n_auto,
            "n_human": self.n_human,
            "n_abstain": self.n_abstain,
            "auto_rate": self.auto_rate,
            "human_rate": self.human_rate,
            "abstain_rate": self.abstain_rate,
            "n_correct_auto": self.n_correct_auto,
            "n_wrong_auto": self.n_wrong_auto,
            "wrong_auto_rate": self.wrong_auto_rate,
            "total_realized_cost": self.total_realized_cost,
            "mean_cost_per_record": self.mean_cost_per_record,
            "cost_per_wrong_auto": self.cost_per_wrong_auto,
            "cost_per_human": self.cost_per_human,
            "cost_per_abstain": self.cost_per_abstain,
            "n_abstain_on_corrupted": self.n_abstain_on_corrupted,
            "n_abstain_on_clean": self.n_abstain_on_clean,
            "n_genuinely_corrupted": self.n_genuinely_corrupted,
            "abstain_precision": self.abstain_precision,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PolicyComparisonRow
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PolicyComparisonRow:
    """
    One row of a policy comparison summary table.

    Combines repair quality and cost metrics for one method.
    """

    method_name: str
    dataset: str
    detection_precision: float
    detection_recall: float
    detection_f1: float
    repair_accuracy: float
    auto_rate: float
    human_rate: float
    abstain_rate: float
    wrong_auto_rate: float
    total_realized_cost: float
    mean_cost_per_record: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "dataset": self.dataset,
            "detection_precision": self.detection_precision,
            "detection_recall": self.detection_recall,
            "detection_f1": self.detection_f1,
            "repair_accuracy": self.repair_accuracy,
            "auto_rate": self.auto_rate,
            "human_rate": self.human_rate,
            "abstain_rate": self.abstain_rate,
            "wrong_auto_rate": self.wrong_auto_rate,
            "total_realized_cost": self.total_realized_cost,
            "mean_cost_per_record": self.mean_cost_per_record,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CostEvaluator
# ═══════════════════════════════════════════════════════════════════════════════


class CostEvaluator:
    """
    Compute realized costs for BaselineRecord lists against ground truth.

    This is the ONLY class in this module that accesses clean_df.
    It uses clean_df ONLY to check (a) whether AUTO_REPAIR decisions are
    correct and (b) whether each flagged cell was genuinely corrupted,
    which decides if an ABSTAIN is charged (see module docstring).

    Parameters
    ----------
    clean_df : pd.DataFrame
        Ground-truth clean data.  Read-only.
    c_wrong : float
        Realized cost of applying an incorrect repair.
    c_human : float
        Realized cost of routing to human review.
    c_abstain : float
        Realized cost of abstaining on a genuinely corrupted cell.
        Abstaining on a false alarm costs 0.
    comparator : ValueComparator, optional
        Custom comparison strategy.  Defaults to exact comparison.
    """

    def __init__(
        self,
        clean_df: pd.DataFrame,
        c_wrong: float,
        c_human: float,
        c_abstain: float,
        comparator: Optional[ValueComparator] = None,
    ) -> None:
        if c_wrong <= 0:
            raise ValueError(f"c_wrong must be positive, got {c_wrong}")
        if c_human < 0:
            raise ValueError(f"c_human must be non-negative, got {c_human}")
        if c_abstain < 0:
            raise ValueError(f"c_abstain must be non-negative, got {c_abstain}")
        self.clean_df = clean_df.reset_index(drop=True)
        self.c_wrong = float(c_wrong)
        self.c_human = float(c_human)
        self.c_abstain = float(c_abstain)
        self.comparator = comparator or ValueComparator()

    @classmethod
    def from_cost_config(
        cls,
        clean_df: pd.DataFrame,
        costs: CostConfig,
        comparator: Optional[ValueComparator] = None,
    ) -> "CostEvaluator":
        """
        Create CostEvaluator from a Phase 7 CostConfig.

        Mapping:
            c_wrong_repair  → c_wrong
            c_human_review  → c_human
            c_leave_error   → c_abstain
        """
        return cls(
            clean_df=clean_df,
            c_wrong=costs.c_wrong_repair,
            c_human=costs.c_human_review,
            c_abstain=costs.c_leave_error,
            comparator=comparator,
        )

    # ── Core evaluation ───────────────────────────────────────────────────────

    def evaluate(
        self,
        records: List[BaselineRecord],
        dataset: str = "unknown",
    ) -> CostReport:
        """
        Compute realized cost metrics for a list of BaselineRecords.

        Parameters
        ----------
        records : list[BaselineRecord]
        dataset : str

        Returns
        -------
        CostReport
        """
        n = len(records)
        method_name = records[0].method_name if records else "unknown"

        if n == 0:
            return self._empty_report(method_name, dataset)

        n_auto = n_human = n_abstain = 0
        n_correct_auto = n_wrong_auto = 0
        n_abstain_on_corrupted = n_abstain_on_clean = 0
        n_genuinely_corrupted = 0
        total_cost = 0.0

        for rec in records:
            action = rec.action
            genuinely_corrupted = self._is_genuinely_corrupted(rec)
            if genuinely_corrupted:
                n_genuinely_corrupted += 1

            if action == DecisionAction.AUTO_REPAIR:
                n_auto += 1
                correct = self._is_repair_correct(rec)
                if correct:
                    n_correct_auto += 1
                    # Correct repair: no cost incurred
                else:
                    n_wrong_auto += 1
                    total_cost += self.c_wrong
            elif action == DecisionAction.HUMAN_REVIEW:
                n_human += 1
                total_cost += self.c_human
            else:  # ABSTAIN
                n_abstain += 1
                # Realised counterpart of E[abstain] = q * C_abstain:
                # charged only when the error actually persists.
                if genuinely_corrupted:
                    n_abstain_on_corrupted += 1
                    total_cost += self.c_abstain
                else:
                    n_abstain_on_clean += 1

        denom = max(n, 1)
        auto_rate = n_auto / denom
        human_rate = n_human / denom
        abstain_rate = n_abstain / denom
        wrong_auto_rate = n_wrong_auto / max(n_auto, 1) if n_auto > 0 else 0.0

        return CostReport(
            method_name=method_name,
            dataset=dataset,
            n_records=n,
            n_auto=n_auto,
            n_human=n_human,
            n_abstain=n_abstain,
            auto_rate=round(auto_rate, 6),
            human_rate=round(human_rate, 6),
            abstain_rate=round(abstain_rate, 6),
            n_correct_auto=n_correct_auto,
            n_wrong_auto=n_wrong_auto,
            wrong_auto_rate=round(wrong_auto_rate, 6),
            total_realized_cost=round(total_cost, 6),
            mean_cost_per_record=round(total_cost / denom, 6),
            cost_per_wrong_auto=self.c_wrong,
            cost_per_human=self.c_human,
            cost_per_abstain=self.c_abstain,
            n_abstain_on_corrupted=n_abstain_on_corrupted,
            n_abstain_on_clean=n_abstain_on_clean,
            n_genuinely_corrupted=n_genuinely_corrupted,
        )

    # ── Internals ──────────────────────────────────────────────────────────────

    def _is_repair_correct(self, rec: BaselineRecord) -> bool:
        """Check whether top_candidate matches the clean value."""
        try:
            clean_val = self.clean_df.at[rec.row_id, rec.column]
        except (KeyError, IndexError):
            return False
        return self.comparator.are_equal(rec.top_candidate, clean_val)

    def _is_genuinely_corrupted(self, rec: BaselineRecord) -> bool:
        """
        True if the flagged cell's dirty value differs from the clean value.

        Mirrors the label rule of DetectionLabelBuilder (Step 1): the same
        ValueComparator, so NaN == NaN and 371.0 == 371.  A cell that cannot
        be looked up in clean_df is treated as corrupted, matching the
        conservative choice in _is_repair_correct (unknown -> not correct).
        """
        try:
            clean_val = self.clean_df.at[rec.row_id, rec.column]
        except (KeyError, IndexError):
            return True
        return not self.comparator.are_equal(rec.dirty_value, clean_val)

    def _empty_report(self, method_name: str, dataset: str) -> CostReport:
        return CostReport(
            method_name=method_name, dataset=dataset, n_records=0,
            cost_per_wrong_auto=self.c_wrong,
            cost_per_human=self.c_human,
            cost_per_abstain=self.c_abstain,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Policy comparison helpers
# ═══════════════════════════════════════════════════════════════════════════════


def compare_policies(
    cost_reports: List[CostReport],
    repair_reports: list,  # list[RepairMetrics] from repair_metrics.py
) -> List[PolicyComparisonRow]:
    """
    Build a comparison table by joining CostReport and RepairMetrics.

    Parameters
    ----------
    cost_reports : list[CostReport]
    repair_reports : list[RepairMetrics]
        Must be in the same order as cost_reports (same method + dataset).

    Returns
    -------
    list[PolicyComparisonRow]

    Raises
    ------
    ValueError
        If the two lists have different lengths.
    """
    if len(cost_reports) != len(repair_reports):
        raise ValueError(
            f"cost_reports and repair_reports must have the same length. "
            f"Got {len(cost_reports)} and {len(repair_reports)}."
        )
    rows = []
    for cr, rr in zip(cost_reports, repair_reports):
        rows.append(PolicyComparisonRow(
            method_name=cr.method_name,
            dataset=cr.dataset,
            detection_precision=rr.detection_precision,
            detection_recall=rr.detection_recall,
            detection_f1=rr.detection_f1,
            repair_accuracy=rr.repair_accuracy,
            auto_rate=cr.auto_rate,
            human_rate=cr.human_rate,
            abstain_rate=cr.abstain_rate,
            wrong_auto_rate=cr.wrong_auto_rate,
            total_realized_cost=cr.total_realized_cost,
            mean_cost_per_record=cr.mean_cost_per_record,
        ))
    return rows


def print_comparison_table(rows: List[PolicyComparisonRow]) -> None:
    """Pretty-print a policy comparison table to stdout."""
    if not rows:
        print("(no rows to display)")
        return
    w = 120
    print(f"\n{'─' * w}")
    print(f"  POLICY COMPARISON TABLE")
    print(f"{'─' * w}")
    hdr = (
        f"  {'Method':<35} {'Dataset':<15} "
        f"{'Prec':>6} {'Rec':>6} {'F1':>6} "
        f"{'RepAcc':>7} {'Auto%':>6} {'Human%':>7} {'Abs%':>5} "
        f"{'WrongAuto%':>11} {'TotalCost':>10} {'AvgCost':>8}"
    )
    print(hdr)
    print(f"{'─' * w}")
    for r in rows:
        print(
            f"  {r.method_name:<35} {r.dataset:<15} "
            f"{r.detection_precision:>6.3f} {r.detection_recall:>6.3f} {r.detection_f1:>6.3f} "
            f"{r.repair_accuracy:>7.3f} {100*r.auto_rate:>6.1f} "
            f"{100*r.human_rate:>7.1f} {100*r.abstain_rate:>5.1f} "
            f"{100*r.wrong_auto_rate:>11.1f} {r.total_realized_cost:>10.2f} "
            f"{r.mean_cost_per_record:>8.4f}"
        )
    print(f"{'─' * w}")
