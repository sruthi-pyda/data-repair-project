"""
src/risk/cost_model.py
-----------------------
Phase 7: Risk and cost modeling for repair decisions.

For a proposed repair with calibrated probability p:
  E[ACCEPT_REPAIR]   = (1 - p) * C_wrong_repair
  E[HUMAN_REVIEW]    = C_human_review  (constant)
  E[LEAVE_UNCHANGED] = C_leave_error   (constant)

The lowest-cost action is taken as the recommended decision.

Analytical thresholds (where actions become equal):
  ACCEPT vs REVIEW:  p* = 1 - C_human_review / C_wrong_repair
  ACCEPT vs LEAVE:   p** = 1 - C_leave_error / C_wrong_repair
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Action enum
# ---------------------------------------------------------------------------


class Action(str, Enum):
    """Possible actions for a detected and repaired cell."""
    ACCEPT_REPAIR = "ACCEPT_REPAIR"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    LEAVE_UNCHANGED = "LEAVE_UNCHANGED"


# ---------------------------------------------------------------------------
# Cost configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostConfig:
    """
    Cost parameters for a single dataset context.

    All values are in abstract "penalty units" — lower total cost = better policy.

    Parameters
    ----------
    c_wrong_repair : float
        Penalty for automatically applying an incorrect repair.
    c_human_review : float
        Penalty for routing a cell to human review.
    c_leave_error : float
        Penalty for leaving a genuine error unrepaired.
    dataset_name : str
        The dataset this config applies to (for reporting).
    """
    c_wrong_repair: float
    c_human_review: float
    c_leave_error: float
    dataset_name: str = "unknown"

    def __post_init__(self):
        if self.c_wrong_repair <= 0:
            raise ValueError(
                f"c_wrong_repair must be positive, got {self.c_wrong_repair}"
            )
        if self.c_human_review < 0:
            raise ValueError(
                f"c_human_review must be non-negative, got {self.c_human_review}"
            )
        if self.c_leave_error < 0:
            raise ValueError(
                f"c_leave_error must be non-negative, got {self.c_leave_error}"
            )

    @classmethod
    def from_yaml(cls, path: Path, dataset_name: str) -> "CostConfig":
        """
        Load cost config from configs/costs.yaml.

        Checks for a dataset-specific override block first; falls back to the
        'defaults' block if the dataset is not listed.

        Parameters
        ----------
        path : Path
            Path to the YAML cost config file.
        dataset_name : str
            Dataset identifier (e.g. "breast_cancer", "smartfactory", "adult").
        """
        path = Path(path)
        with path.open() as fh:
            cfg = yaml.safe_load(fh)

        defaults = cfg.get("defaults") or {}
        ds_block = (cfg.get("datasets") or {}).get(dataset_name)
        block = ds_block if ds_block is not None else defaults

        def _get(key: str, fallback: float) -> float:
            val = block.get(key, defaults.get(key, fallback))
            return float(val)

        return cls(
            c_wrong_repair=_get("c_wrong_repair", 10.0),
            c_human_review=_get("c_human_review", 2.0),
            c_leave_error=_get("c_leave_error", 5.0),
            dataset_name=dataset_name,
        )

    @classmethod
    def default(cls, dataset_name: str = "unknown") -> "CostConfig":
        """Return the default cost configuration (wrong=10, review=2, leave=5)."""
        return cls(
            c_wrong_repair=10.0,
            c_human_review=2.0,
            c_leave_error=5.0,
            dataset_name=dataset_name,
        )


# ---------------------------------------------------------------------------
# Expected-cost dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectedCosts:
    """
    Expected costs for each action at a given calibrated probability p.

    Attributes
    ----------
    p : float
        Calibrated probability the repair is correct, in [0, 1].
    accept : float
        E[ACCEPT_REPAIR] = (1 - p) * C_wrong_repair
    review : float
        E[HUMAN_REVIEW] = C_human_review
    leave : float
        E[LEAVE_UNCHANGED] = C_leave_error
    best_action : Action
        Action with the lowest expected cost.
    min_cost : float
        Expected cost of the best action.
    """
    p: float
    accept: float
    review: float
    leave: float
    best_action: Action
    min_cost: float


# ---------------------------------------------------------------------------
# Risk record
# ---------------------------------------------------------------------------


@dataclass
class RiskRecord:
    """
    Full risk assessment for a single detected cell.

    Attributes
    ----------
    dataset : str
    row_id : int
    column : str
    top_candidate : object
        The proposed repair value.
    calibrated_probability : float
        p — calibrated probability the repair is correct.
    probability_wrong : float
        1 - p
    expected_cost_accept : float
    expected_cost_review : float
    expected_cost_leave : float
    minimum_expected_cost : float
    lowest_cost_action : Action
    costs : CostConfig
        The cost parameters used.
    """
    dataset: str
    row_id: int
    column: str
    top_candidate: object
    calibrated_probability: float
    probability_wrong: float
    expected_cost_accept: float
    expected_cost_review: float
    expected_cost_leave: float
    minimum_expected_cost: float
    lowest_cost_action: Action
    costs: CostConfig

    def to_dict(self) -> Dict:
        return {
            "dataset": self.dataset,
            "row_id": self.row_id,
            "column": self.column,
            "top_candidate": self.top_candidate,
            "calibrated_probability": self.calibrated_probability,
            "probability_wrong": self.probability_wrong,
            "expected_cost_accept": self.expected_cost_accept,
            "expected_cost_review": self.expected_cost_review,
            "expected_cost_leave": self.expected_cost_leave,
            "minimum_expected_cost": self.minimum_expected_cost,
            "lowest_cost_action": self.lowest_cost_action.value,
            "c_wrong_repair": self.costs.c_wrong_repair,
            "c_human_review": self.costs.c_human_review,
            "c_leave_error": self.costs.c_leave_error,
        }


# ---------------------------------------------------------------------------
# Analytical threshold result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalyticalThresholds:
    """
    Analytical breakeven thresholds derived from cost parameters.

    Attributes
    ----------
    p_accept_vs_review : float
        Minimum p for ACCEPT to beat REVIEW.
        p* = 1 - C_human_review / C_wrong_repair
    p_accept_vs_leave : float
        Minimum p for ACCEPT to beat LEAVE.
        p** = 1 - C_leave_error / C_wrong_repair
    review_cheaper_than_leave : bool
        Whether C_human_review < C_leave_error (affects policy for low-p region).
    recommended_policy : str
        Human-readable summary of the implied threshold policy.
    costs : CostConfig
    """
    p_accept_vs_review: float
    p_accept_vs_leave: float
    review_cheaper_than_leave: bool
    recommended_policy: str
    costs: CostConfig


# ---------------------------------------------------------------------------
# Sensitivity-analysis point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensitivityPoint:
    """One row of the sensitivity sweep table."""
    p: float
    cost_accept: float
    cost_review: float
    cost_leave: float
    best_action: Action


# ---------------------------------------------------------------------------
# RiskCalculator
# ---------------------------------------------------------------------------


class RiskCalculator:
    """
    Computes expected costs and recommended actions for a given cost config.

    Parameters
    ----------
    costs : CostConfig
        Cost parameters for this dataset.
    """

    def __init__(self, costs: CostConfig):
        self.costs = costs

    # ── Core computation ──────────────────────────────────────────────────────

    def compute_expected_costs(self, p: float) -> ExpectedCosts:
        """
        Compute expected cost for each action at calibrated probability p.

        Parameters
        ----------
        p : float
            Calibrated probability the repair is correct, in [0, 1].

        Returns
        -------
        ExpectedCosts
        """
        if not 0.0 <= p <= 1.0:
            raise ValueError(
                f"Calibrated probability p must be in [0, 1], got {p}"
            )

        c = self.costs
        e_accept = (1.0 - p) * c.c_wrong_repair
        e_review = c.c_human_review
        e_leave = c.c_leave_error

        costs_by_action = {
            Action.ACCEPT_REPAIR: e_accept,
            Action.HUMAN_REVIEW: e_review,
            Action.LEAVE_UNCHANGED: e_leave,
        }
        best = min(costs_by_action, key=costs_by_action.__getitem__)

        return ExpectedCosts(
            p=p,
            accept=e_accept,
            review=e_review,
            leave=e_leave,
            best_action=best,
            min_cost=costs_by_action[best],
        )

    def compute_risk_record(
        self,
        unc_record,
        cal_prob: float,
    ) -> RiskRecord:
        """
        Build a RiskRecord from an UncertaintyRecord and calibrated probability.

        Parameters
        ----------
        unc_record : UncertaintyRecord
            From src.uncertainty.estimator.
        cal_prob : float
            Calibrated probability (output of LogisticCalibrator or IsotonicCalibrator).

        Returns
        -------
        RiskRecord
        """
        ec = self.compute_expected_costs(cal_prob)
        top_val = getattr(unc_record, "top_candidate", None)

        return RiskRecord(
            dataset=unc_record.dataset,
            row_id=unc_record.row_id,
            column=unc_record.column,
            top_candidate=top_val,
            calibrated_probability=cal_prob,
            probability_wrong=1.0 - cal_prob,
            expected_cost_accept=ec.accept,
            expected_cost_review=ec.review,
            expected_cost_leave=ec.leave,
            minimum_expected_cost=ec.min_cost,
            lowest_cost_action=ec.best_action,
            costs=self.costs,
        )

    def compute_risk_batch(
        self,
        unc_records: list,
        cal_probs: List[float],
        skip_columns: Optional[List[str]] = None,
    ) -> List[RiskRecord]:
        """
        Compute risk records for a list of UncertaintyRecords.

        Parameters
        ----------
        unc_records : list[UncertaintyRecord]
        cal_probs : list[float]
            One calibrated probability per record (same order).
        skip_columns : list[str], optional
            Columns to skip (e.g. ID columns such as "Unnamed: 0").
            Records in these columns are omitted from the output.

        Returns
        -------
        list[RiskRecord]
        """
        if len(unc_records) != len(cal_probs):
            raise ValueError(
                f"unc_records length ({len(unc_records)}) must match "
                f"cal_probs length ({len(cal_probs)})"
            )
        skip = set(skip_columns or [])
        results = []
        for rec, p in zip(unc_records, cal_probs):
            if rec.column in skip:
                continue
            results.append(self.compute_risk_record(rec, p))
        return results

    # ── Analytical thresholds ─────────────────────────────────────────────────

    def analytical_thresholds(self) -> AnalyticalThresholds:
        """
        Derive breakeven probability thresholds analytically.

        ACCEPT beats REVIEW when:  p > 1 - C_human_review / C_wrong_repair
        ACCEPT beats LEAVE when:   p > 1 - C_leave_error / C_wrong_repair

        Returns
        -------
        AnalyticalThresholds
        """
        c = self.costs
        p_star = 1.0 - c.c_human_review / c.c_wrong_repair
        p_dstar = 1.0 - c.c_leave_error / c.c_wrong_repair

        # Clamp to valid probability range
        p_star = max(0.0, min(1.0, p_star))
        p_dstar = max(0.0, min(1.0, p_dstar))

        review_cheaper = c.c_human_review < c.c_leave_error

        if review_cheaper:
            policy = (
                f"ACCEPT if p > {p_star:.4f} "
                f"(= 1 - {c.c_human_review}/{c.c_wrong_repair}); "
                f"otherwise HUMAN_REVIEW "
                f"(review cheaper than leave at threshold {p_dstar:.4f})"
            )
        else:
            policy = (
                f"ACCEPT if p > {p_dstar:.4f} "
                f"(= 1 - {c.c_leave_error}/{c.c_wrong_repair}); "
                f"otherwise LEAVE_UNCHANGED "
                f"(leave cheaper than review at low p)"
            )

        return AnalyticalThresholds(
            p_accept_vs_review=p_star,
            p_accept_vs_leave=p_dstar,
            review_cheaper_than_leave=review_cheaper,
            recommended_policy=policy,
            costs=c,
        )

    # ── Sensitivity analysis ──────────────────────────────────────────────────

    def sensitivity_over_p(self, n_steps: int = 21) -> List[SensitivityPoint]:
        """
        Tabulate expected costs and best action over a grid of p values in [0, 1].

        Parameters
        ----------
        n_steps : int
            Number of evenly-spaced p values (default 21 → 0.00, 0.05, …, 1.00).

        Returns
        -------
        list[SensitivityPoint]
        """
        if n_steps < 2:
            raise ValueError("n_steps must be >= 2")

        points = []
        for i in range(n_steps):
            p = round(i / (n_steps - 1), 8)
            ec = self.compute_expected_costs(p)
            points.append(SensitivityPoint(
                p=p,
                cost_accept=ec.accept,
                cost_review=ec.review,
                cost_leave=ec.leave,
                best_action=ec.best_action,
            ))
        return points
