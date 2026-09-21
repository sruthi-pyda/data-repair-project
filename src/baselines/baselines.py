"""
src/baselines/baselines.py
---------------------------
Phase 9: Baseline repair-decision policies for comparison with the
calibrated risk-aware method (Phase 8).

OVERVIEW
========
Two baselines are implemented here:

  Baseline 1 — AlwaysAutoBaseline
  --------------------------------
  Always selects AUTO_REPAIR for every detected error.
  Represents the naive strategy: "trust the LLM unconditionally".

  Baseline 2 — ConfidenceThresholdBaseline
  -----------------------------------------
  Applies a fixed raw-confidence threshold:
      if raw_confidence >= threshold  →  AUTO_REPAIR
      else                            →  HUMAN_REVIEW
  The threshold is configurable.  Multiple thresholds can be swept
  by instantiating multiple baselines.

PROPOSED METHOD
===============
The calibrated risk-aware policy (src/decision/policy.DecisionPolicy)
is the proposed method and is NOT reimplemented here.  All three methods
share the BaselineRecord output type so that Phase 10 can evaluate them
on equal terms.

COMMON INTERFACE
================
Every baseline exposes:

    baseline.run(unc_records) -> list[BaselineRecord]

BaselineRecord carries enough information to:
  - reconstruct which method / threshold was used
  - evaluate repair quality vs ground truth (in Phase 10)
  - compute action distribution and cost metrics

DATA-LEAKAGE RULES
==================
  - No baseline accesses clean_df.
  - No baseline accepts a clean_df argument at decision time.
  - No baseline inspects ground-truth labels during decision-making.
  - All inputs come from UncertaintyRecord (which was built entirely
    from dirty data + detection output).
  - Ground truth is used ONLY by the evaluation layer (Phase 10),
    after all decisions have been produced.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.decision.policy import DecisionAction
from src.uncertainty.estimator import UncertaintyRecord

logger = logging.getLogger(__name__)

# Canonical set of thresholds for a full confidence-threshold sweep
DEFAULT_THRESHOLDS: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9]


# ---------------------------------------------------------------------------
# BaselineRecord — shared output type
# ---------------------------------------------------------------------------


@dataclass
class BaselineRecord:
    """
    Decision record produced by any baseline or the proposed method.

    This common output type lets Phase 10 evaluate all methods on the
    same footing without code duplication.

    Attributes
    ----------
    method_name : str
        Human-readable identifier, e.g. "always_auto",
        "confidence_threshold_0.80", or "calibrated_risk_aware".
    dataset : str
    row_id : int
    column : str
    dirty_value : Any
        The original corrupted value (from dirty data).
    top_candidate : Any
        The repair candidate selected (highest raw probability).
    raw_confidence : float
        Raw (uncalibrated) probability assigned to the top candidate.
    action : DecisionAction
        The decision made: AUTO_REPAIR, HUMAN_REVIEW, or ABSTAIN.
    threshold : float or None
        Confidence threshold used (only set for ConfidenceThresholdBaseline).
    calibrated_probability : float or None
        Calibrated probability of correctness.
        None for baselines that do not use calibration.
    expected_cost_auto : float or None
        E[Cost_auto] from the risk-aware policy.  None for baselines.
    expected_cost_human : float or None
        E[Cost_human] from the risk-aware policy.  None for baselines.
    expected_cost_abstain : float or None
        E[Cost_abstain] from the risk-aware policy.  None for baselines.
    metadata : dict
        Additional key–value pairs specific to the method.

    Notes
    -----
    clean_df and ground-truth labels are NOT stored here.
    They belong to the evaluation layer.
    """

    method_name: str
    dataset: str
    row_id: int
    column: str
    dirty_value: Any
    top_candidate: Any
    raw_confidence: float
    action: DecisionAction

    # Optional fields — only populated by methods that support them
    threshold: Optional[float] = None
    calibrated_probability: Optional[float] = None
    expected_cost_auto: Optional[float] = None
    expected_cost_human: Optional[float] = None
    expected_cost_abstain: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a flat dictionary for export to CSV / DataFrame."""
        return {
            "method_name": self.method_name,
            "dataset": self.dataset,
            "row_id": self.row_id,
            "column": self.column,
            "dirty_value": self.dirty_value,
            "top_candidate": self.top_candidate,
            "raw_confidence": self.raw_confidence,
            "action": self.action.value,
            "threshold": self.threshold,
            "calibrated_probability": self.calibrated_probability,
            "expected_cost_auto": self.expected_cost_auto,
            "expected_cost_human": self.expected_cost_human,
            "expected_cost_abstain": self.expected_cost_abstain,
            **self.metadata,
        }


# ---------------------------------------------------------------------------
# Baseline 1: Always Auto-Repair
# ---------------------------------------------------------------------------


class AlwaysAutoBaseline:
    """
    Baseline 1: Always select AUTO_REPAIR.

    Regardless of confidence, entropy, or cost, this policy applies the
    top repair candidate automatically to every detected error.

    Research question
    -----------------
    "What happens if we blindly trust the LLM every time?"

    This measures the ceiling accuracy achievable if repair is always
    applied, and also exposes the total cost incurred when wrong repairs
    are accepted without any filter.

    Data-leakage guarantee
    ----------------------
    Decisions are made from UncertaintyRecord alone.
    clean_df is never accessed.

    Parameters
    ----------
    dataset_name : str
        Used to label output records.  Does not affect decisions.
    """

    METHOD_NAME = "always_auto"

    def __init__(self, dataset_name: str = "unknown") -> None:
        self.dataset_name = dataset_name

    def run(self, unc_records: List[UncertaintyRecord]) -> List[BaselineRecord]:
        """
        Apply always-auto policy to a list of UncertaintyRecords.

        Parameters
        ----------
        unc_records : list[UncertaintyRecord]
            Uncertainty records from Phase 5.  Must not contain clean values.

        Returns
        -------
        list[BaselineRecord]
            One BaselineRecord per input record, all with action=AUTO_REPAIR.
        """
        results: List[BaselineRecord] = []
        for rec in unc_records:
            results.append(
                BaselineRecord(
                    method_name=self.METHOD_NAME,
                    dataset=rec.dataset,
                    row_id=rec.row_id,
                    column=rec.column,
                    dirty_value=rec.dirty_value,
                    top_candidate=rec.top_candidate,
                    raw_confidence=rec.raw_confidence,
                    action=DecisionAction.AUTO_REPAIR,
                    metadata={
                        "repair_method": rec.repair_method,
                        "error_type": rec.error_type,
                    },
                )
            )
        logger.debug(
            "AlwaysAutoBaseline '%s': %d records → all AUTO_REPAIR.",
            self.dataset_name, len(results),
        )
        return results


# ---------------------------------------------------------------------------
# Baseline 2: Confidence-Only Threshold
# ---------------------------------------------------------------------------


class ConfidenceThresholdBaseline:
    """
    Baseline 2: Route decisions using raw top-candidate confidence.

    Rule
    ----
        if raw_confidence >= threshold  →  AUTO_REPAIR
        else                            →  HUMAN_REVIEW

    The threshold is configurable; 0.8 is a common default but is NOT
    assumed to be universally optimal.  Use sweep_thresholds() to
    evaluate multiple thresholds in a single call.

    Research question
    -----------------
    "Is raw confidence alone sufficient to make good routing decisions?"
    Comparison with the calibrated risk-aware policy reveals whether
    calibration and explicit cost modeling add value beyond a simple
    confidence cutoff.

    Data-leakage guarantee
    ----------------------
    Decisions are made from UncertaintyRecord.raw_confidence alone.
    clean_df is never accessed.

    Parameters
    ----------
    threshold : float
        Confidence cutoff in (0, 1].
        raw_confidence >= threshold  →  AUTO_REPAIR
        raw_confidence <  threshold  →  HUMAN_REVIEW
    dataset_name : str
        Used to label output records.  Does not affect decisions.

    Raises
    ------
    ValueError
        If threshold is not in (0, 1].
    """

    def __init__(
        self,
        threshold: float,
        dataset_name: str = "unknown",
    ) -> None:
        if not (0.0 < threshold <= 1.0):
            raise ValueError(
                f"threshold must be in (0, 1], got {threshold!r}"
            )
        self.threshold = float(threshold)
        self.dataset_name = dataset_name

    @property
    def method_name(self) -> str:
        """Unique identifier including the threshold value."""
        return f"confidence_threshold_{self.threshold:.2f}"

    def decide_one(self, raw_confidence: float) -> DecisionAction:
        """
        Apply the threshold rule to a single confidence value.

        Parameters
        ----------
        raw_confidence : float
            Raw probability of the top repair candidate.

        Returns
        -------
        DecisionAction.AUTO_REPAIR  if raw_confidence >= threshold
        DecisionAction.HUMAN_REVIEW otherwise
        """
        if raw_confidence >= self.threshold:
            return DecisionAction.AUTO_REPAIR
        return DecisionAction.HUMAN_REVIEW

    def run(self, unc_records: List[UncertaintyRecord]) -> List[BaselineRecord]:
        """
        Apply the confidence-threshold policy to a list of UncertaintyRecords.

        Parameters
        ----------
        unc_records : list[UncertaintyRecord]
            Uncertainty records from Phase 5.  Must not contain clean values.

        Returns
        -------
        list[BaselineRecord]
            One BaselineRecord per input record.
        """
        results: List[BaselineRecord] = []
        n_auto = 0
        for rec in unc_records:
            action = self.decide_one(rec.raw_confidence)
            if action == DecisionAction.AUTO_REPAIR:
                n_auto += 1
            results.append(
                BaselineRecord(
                    method_name=self.method_name,
                    dataset=rec.dataset,
                    row_id=rec.row_id,
                    column=rec.column,
                    dirty_value=rec.dirty_value,
                    top_candidate=rec.top_candidate,
                    raw_confidence=rec.raw_confidence,
                    action=action,
                    threshold=self.threshold,
                    metadata={
                        "repair_method": rec.repair_method,
                        "error_type": rec.error_type,
                    },
                )
            )
        logger.debug(
            "ConfidenceThresholdBaseline '%s' (t=%.2f): %d records → "
            "%d AUTO_REPAIR, %d HUMAN_REVIEW.",
            self.dataset_name, self.threshold, len(results),
            n_auto, len(results) - n_auto,
        )
        return results


# ---------------------------------------------------------------------------
# Sweep helper
# ---------------------------------------------------------------------------


def sweep_thresholds(
    unc_records: List[UncertaintyRecord],
    thresholds: Optional[List[float]] = None,
    dataset_name: str = "unknown",
) -> Dict[float, List[BaselineRecord]]:
    """
    Run ConfidenceThresholdBaseline for multiple thresholds in one call.

    Parameters
    ----------
    unc_records : list[UncertaintyRecord]
    thresholds : list[float], optional
        Threshold values to sweep.  Defaults to DEFAULT_THRESHOLDS
        [0.5, 0.6, 0.7, 0.8, 0.9].
    dataset_name : str

    Returns
    -------
    dict[float, list[BaselineRecord]]
        Maps each threshold to the list of BaselineRecords produced at
        that threshold.

    Raises
    ------
    ValueError
        If any threshold is not in (0, 1].
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    results: Dict[float, List[BaselineRecord]] = {}
    for t in thresholds:
        baseline = ConfidenceThresholdBaseline(threshold=t, dataset_name=dataset_name)
        results[t] = baseline.run(unc_records)
    return results


# ---------------------------------------------------------------------------
# Adapter: proposed method → BaselineRecord
# ---------------------------------------------------------------------------


def wrap_proposed_method(
    unc_records: List[UncertaintyRecord],
    calibrated_probs: List[float],
    decision_results: list,  # list[DecisionResult] from Phase 8
    dataset_name: str = "unknown",
) -> List[BaselineRecord]:
    """
    Convert Phase 8 DecisionResult objects into BaselineRecord format.

    This adapter lets Phase 10 evaluate the proposed calibrated risk-aware
    policy on the same footing as the two baselines without duplicating
    evaluation logic.

    Parameters
    ----------
    unc_records : list[UncertaintyRecord]
        Must be in the same order as decision_results.
    calibrated_probs : list[float]
        Calibrated probability for each record (same order).
    decision_results : list[DecisionResult]
        Outputs of DecisionPolicy.decide() from Phase 8 (same order).
    dataset_name : str

    Returns
    -------
    list[BaselineRecord]

    Raises
    ------
    ValueError
        If the three input lists have different lengths.
    """
    if not (len(unc_records) == len(calibrated_probs) == len(decision_results)):
        raise ValueError(
            f"All three inputs must have the same length. Got "
            f"unc_records={len(unc_records)}, "
            f"calibrated_probs={len(calibrated_probs)}, "
            f"decision_results={len(decision_results)}."
        )

    records: List[BaselineRecord] = []
    for unc, cal_p, dr in zip(unc_records, calibrated_probs, decision_results):
        records.append(
            BaselineRecord(
                method_name="calibrated_risk_aware",
                dataset=unc.dataset,
                row_id=unc.row_id,
                column=unc.column,
                dirty_value=unc.dirty_value,
                top_candidate=unc.top_candidate,
                raw_confidence=unc.raw_confidence,
                action=dr.action,
                calibrated_probability=cal_p,
                expected_cost_auto=dr.expected_cost_auto,
                expected_cost_human=dr.expected_cost_human,
                expected_cost_abstain=dr.expected_cost_abstain,
                metadata={
                    "repair_method": unc.repair_method,
                    "error_type": unc.error_type,
                    "c_wrong": dr.c_wrong,
                    "c_human": dr.c_human,
                    "c_abstain": dr.c_abstain,
                    # q used for E[Cost_abstain] = q × C_abstain.  getattr so
                    # a DecisionResult built before this field existed
                    # still wraps (it implied q = 1).
                    "corruption_probability": getattr(
                        dr, "corruption_probability", 1.0
                    ),
                },
            )
        )
    return records
