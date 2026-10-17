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

COMPLETE COSTS — THE SECOND COSTING MODE  (W5.1a, gap 4a)
==========================================================
Everything above prices only the cells the detector FLAGGED.  A genuinely
corrupted cell that was never flagged costs zero, because no action was
ever taken on it.  That is the right number for "how good is this
policy" — a policy cannot be blamed for a cell it never saw — and it is
the WRONG number for "how much data-quality harm remains", which is what
a practitioner cares about.  On Adult the gap is total: all 1,701
injected corruptions are undetected, so the flagged-only cost of 0 is
the cost of a problem the pipeline never touched.

``evaluate(..., include_false_negatives=True)`` adds

    surcharge = C_abstain × |{genuinely corrupted cells} − {flagged cells}|

An unrepaired error persists whether or not anyone noticed it, so the
charge is exactly ``C_abstain`` — the corruption-persistence cost of
docs/cost_semantics.md §1, applied to the cells nobody abstained *on*
because nobody saw them.  Three consequences, all reported rather than
buried:

1. **Default is False**, so every number published so far is unchanged,
   bit-for-bit.  ``tests/test_complete_costs.py`` asserts that.
2. **The surcharge is policy-invariant.** It depends on the detector's
   flag SET and on ground truth, not on which action the policy chose.
   Given one detector it therefore adds the same constant to every
   policy and CANNOT change the ranking between policies.  What it
   changes is the magnitude and the interpretation: a "zero cost"
   headline becomes "zero on the 0 % of the problem we can see".
3. **The addressable fraction of total harm is the detection recall**
   (``CostReport.detection_recall``).  It is the ceiling on what any
   decision policy, however good, can do.

``flagged_only_cost`` and ``false_negative_surcharge`` are always both
available on the report, so a table can print the pair side by side.

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
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from src.baselines.baselines import BaselineRecord, DEFAULT_THRESHOLDS
from src.decision.policy import DecisionAction
from src.evaluation.repair_metrics import ValueComparator
from src.risk.cost_model import CostConfig

logger = logging.getLogger(__name__)

CellKey = Tuple[int, str]


# ═══════════════════════════════════════════════════════════════════════════════
# Whole-dataset corruption census (W5.1a)
# ═══════════════════════════════════════════════════════════════════════════════


def corrupted_cells(
    clean_df: pd.DataFrame,
    dirty_df: pd.DataFrame,
    comparator: Optional[ValueComparator] = None,
    skip_columns: Iterable[str] = (),
) -> Set[CellKey]:
    """
    Every genuinely corrupted cell of the WHOLE dataset, as ``(row, column)``.

    This is the denominator the flagged-only cost never had.  The rule is
    the same one :meth:`CostEvaluator._is_genuinely_corrupted` applies to a
    single flagged cell, and the same one
    :class:`~src.calibration.detection_calibrator.DetectionLabelBuilder`
    applies to produce ``q``'s labels, so "corrupted" means one thing
    across the project.

    ``row`` is a zero-based POSITION, matching ``DetectionRecord.row_id`` and
    ``BaselineRecord.row_id``; both frames are positionally re-indexed here
    so the keys line up for any input index.

    Vectorised pre-filter, then the comparator only where it can matter:
    exactly one of the two values NA is a corruption, both NA is not, and
    among the rest only those whose stripped string forms differ are
    candidates (so ``371`` == ``371.0`` survives the comparator, as it must).

    Parameters
    ----------
    clean_df, dirty_df : pd.DataFrame
        Ground truth and input.  Columns present in only one of them are
        skipped (nothing can be said about them).
    comparator : ValueComparator, optional
        Defaults to the canonical one.
    skip_columns : iterable of str
        Columns excluded from the census — used for ID-like columns whose
        "repairs" are meaningless and which the pipeline never decides on
        (``PipelineBundle.skip_columns``).  Counting their corruptions as
        false negatives would charge the policy for cells it is
        deliberately not allowed to touch.

    Returns
    -------
    set[(int, str)]
    """
    cmp_ = comparator or ValueComparator()
    skip = set(skip_columns)
    clean = clean_df.reset_index(drop=True)
    dirty = dirty_df.reset_index(drop=True)

    out: Set[CellKey] = set()
    for col in clean.columns:
        if col in skip or col not in dirty.columns:
            continue
        c = clean[col]
        d = dirty[col]
        na_c, na_d = c.isna(), d.isna()
        for i in c.index[na_c ^ na_d]:
            out.add((int(i), col))
        both_present = (~na_c) & (~na_d)
        str_differs = c.astype(str).str.strip() != d.astype(str).str.strip()
        for i in c.index[both_present & str_differs]:
            if not cmp_.are_equal(d.at[i], c.at[i]):
                out.add((int(i), col))
    return out


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

    Complete costs (W5.1a; all zero/False unless
    ``evaluate(include_false_negatives=True)``)
    ------------------------------------------------------------------
    include_false_negatives : bool
        Which costing mode produced this report.  A table MUST print it:
        the two totals are not comparable.
    n_total_corrupted_cells : int
        Genuinely corrupted cells in the whole dataset (minus any
        ``skip_columns``) — the denominator of total harm.
    n_unflagged_corrupted : int
        Corrupted cells the detector never flagged.  Each one is charged
        ``C_abstain``.
    false_negative_surcharge : float
        ``C_abstain × n_unflagged_corrupted``.  Included in
        ``total_realized_cost`` when the mode is on, and zero otherwise.
    flagged_only_cost : float  (property)
        ``total_realized_cost − false_negative_surcharge`` — the number
        every previously published table reports.
    detection_recall : float or None  (property)
        ``n_genuinely_corrupted / n_total_corrupted_cells``: the fraction
        of total harm ANY policy on this detector could even address.
        ``None`` (not 0.0) when the census was not computed or the dataset
        has no corruption — a 0.0 in a results table reads as a measured
        score.
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

    # Complete costs (W5.1a; appended, defaults keep old constructors valid)
    include_false_negatives: bool = False
    n_total_corrupted_cells: int = 0
    n_unflagged_corrupted: int = 0
    false_negative_surcharge: float = 0.0

    @property
    def abstain_precision(self) -> float:
        """Fraction of ABSTAIN decisions that landed on a false alarm."""
        if self.n_abstain <= 0:
            return 0.0
        return self.n_abstain_on_clean / self.n_abstain

    @property
    def flagged_only_cost(self) -> float:
        """Realised cost on the flagged cells alone (the historical number)."""
        return round(self.total_realized_cost - self.false_negative_surcharge, 6)

    @property
    def detection_recall(self) -> Optional[float]:
        """
        Fraction of the dataset's total corruption this detector flagged.

        The ceiling on what any decision policy can address.  ``None`` when
        the whole-dataset census was not computed (costing mode off) or the
        dataset has no corruption at all.
        """
        if self.n_total_corrupted_cells <= 0:
            return None
        return self.n_genuinely_corrupted / self.n_total_corrupted_cells

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
            "include_false_negatives": self.include_false_negatives,
            "n_total_corrupted_cells": self.n_total_corrupted_cells,
            "n_unflagged_corrupted": self.n_unflagged_corrupted,
            "false_negative_surcharge": self.false_negative_surcharge,
            "flagged_only_cost": self.flagged_only_cost,
            "detection_recall": self.detection_recall,
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
    dirty_df : pd.DataFrame, optional
        The input data.  Needed ONLY by the complete-cost mode
        (``evaluate(include_false_negatives=True)``), which must know which
        cells were corrupted but never flagged; it can also be passed to
        ``evaluate`` directly.  Leaving it ``None`` is the historical
        behaviour and nothing reads it.
    """

    def __init__(
        self,
        clean_df: pd.DataFrame,
        c_wrong: float,
        c_human: float,
        c_abstain: float,
        comparator: Optional[ValueComparator] = None,
        dirty_df: Optional[pd.DataFrame] = None,
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
        self.dirty_df = dirty_df
        # One-entry memo for the whole-dataset corruption census, which is
        # the same for every policy scored against the same (clean, dirty,
        # skip_columns) triple.  Recomputing it per policy would scan Smart
        # Factory's 42,732 x 18 cells eight times for an identical answer.
        self._census_key: Optional[Tuple[int, Tuple[str, ...]]] = None
        self._census: Optional[Set[CellKey]] = None
        self._census_src: Optional[pd.DataFrame] = None

    @classmethod
    def from_cost_config(
        cls,
        clean_df: pd.DataFrame,
        costs: CostConfig,
        comparator: Optional[ValueComparator] = None,
        dirty_df: Optional[pd.DataFrame] = None,
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
            dirty_df=dirty_df,
        )

    # ── Core evaluation ───────────────────────────────────────────────────────

    def evaluate(
        self,
        records: List[BaselineRecord],
        dataset: str = "unknown",
        include_false_negatives: bool = False,
        dirty_df: Optional[pd.DataFrame] = None,
        skip_columns: Optional[Iterable[str]] = None,
    ) -> CostReport:
        """
        Compute realized cost metrics for a list of BaselineRecords.

        Parameters
        ----------
        records : list[BaselineRecord]
        dataset : str
        include_false_negatives : bool
            ``False`` (default) prices only the flagged cells — the
            historical behaviour, bit-identical.  ``True`` additionally
            charges ``C_abstain`` for every genuinely corrupted cell that
            does not appear in ``records``, i.e. that the detector never
            flagged (W5.1a; see the module docstring).  The surcharge is
            policy-invariant for a fixed detector, so it changes totals and
            never rankings.
        dirty_df : pd.DataFrame, optional
            Required by ``include_false_negatives=True`` unless one was
            given to the constructor.  Ignored otherwise.
        skip_columns : iterable of str, optional
            Columns excluded from the whole-dataset corruption census —
            pass ``PipelineBundle.skip_columns`` so ID-like columns the
            pipeline never decides on are not charged as false negatives.

        Returns
        -------
        CostReport

        Raises
        ------
        ValueError
            ``include_false_negatives=True`` with no ``dirty_df`` available.
        """
        n = len(records)
        method_name = records[0].method_name if records else "unknown"

        # The surcharge is computed FIRST, because it is also the whole of the
        # cost when records is empty (a detector that flags nothing still
        # leaves every corruption in place).
        surcharge = 0.0
        n_total_corrupted = 0
        n_unflagged = 0
        if include_false_negatives:
            census = self._corruption_census(dirty_df, skip_columns)
            flagged = {(int(r.row_id), r.column) for r in records}
            n_total_corrupted = len(census)
            n_unflagged = len(census - flagged)
            surcharge = self.c_abstain * n_unflagged

        if n == 0:
            return self._empty_report(
                method_name, dataset,
                include_false_negatives=include_false_negatives,
                n_total_corrupted_cells=n_total_corrupted,
                n_unflagged_corrupted=n_unflagged,
                surcharge=surcharge,
            )

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

        # Undetected corruption is charged last, so `total_cost` above is
        # exactly the historical flagged-only number and the two are
        # separable on the report.
        total_cost += surcharge

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
            include_false_negatives=bool(include_false_negatives),
            n_total_corrupted_cells=n_total_corrupted,
            n_unflagged_corrupted=n_unflagged,
            false_negative_surcharge=round(surcharge, 6),
        )

    # ── Internals ──────────────────────────────────────────────────────────────

    def _corruption_census(
        self,
        dirty_df: Optional[pd.DataFrame],
        skip_columns: Optional[Iterable[str]],
    ) -> Set[CellKey]:
        """
        Memoised whole-dataset corruption census for the complete-cost mode.

        The memo is keyed on the dirty frame's identity AND verified by
        ``is``, so a different frame can never be served a stale answer.
        """
        ddf = dirty_df if dirty_df is not None else self.dirty_df
        if ddf is None:
            raise ValueError(
                "include_false_negatives=True needs the dirty data: pass "
                "dirty_df to evaluate() or to the CostEvaluator constructor. "
                "Undetected corruptions cannot be counted from clean_df alone."
            )
        skip = tuple(sorted(set(skip_columns or ())))
        key = (id(ddf), skip)
        if self._census is not None and self._census_key == key and ddf is self._census_src:
            return self._census
        census = corrupted_cells(
            self.clean_df, ddf, comparator=self.comparator, skip_columns=skip
        )
        self._census = census
        self._census_key = key
        self._census_src = ddf
        return census

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

    def _empty_report(
        self,
        method_name: str,
        dataset: str,
        include_false_negatives: bool = False,
        n_total_corrupted_cells: int = 0,
        n_unflagged_corrupted: int = 0,
        surcharge: float = 0.0,
    ) -> CostReport:
        """
        Report for an empty record list.

        With the complete-cost mode on this is NOT a zero-cost report: a
        detector that flagged nothing left every corruption in place, so the
        surcharge is the whole of the harm.  That is the limiting case the
        flagged-only metric hides most completely.
        """
        return CostReport(
            method_name=method_name, dataset=dataset, n_records=0,
            total_realized_cost=round(surcharge, 6),
            cost_per_wrong_auto=self.c_wrong,
            cost_per_human=self.c_human,
            cost_per_abstain=self.c_abstain,
            include_false_negatives=bool(include_false_negatives),
            n_total_corrupted_cells=int(n_total_corrupted_cells),
            n_unflagged_corrupted=int(n_unflagged_corrupted),
            false_negative_surcharge=round(surcharge, 6),
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
