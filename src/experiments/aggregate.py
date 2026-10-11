"""
src/experiments/aggregate.py
----------------------------
Repeated-split experiments and confidence intervals.

WHY THIS MODULE EXISTS
======================
Every calibration number this project has reported so far came from ONE
70/30 split. Breast Cancer has ~392 calibration samples, so a single split
holds out ~118 of them, and we have been quoting differences of 0.005 Brier
between methods off the back of that. A difference of 0.005 with no error
bar is not a result; it is a coin flip with four decimal places.

This module runs the pipeline over many calibration splits and reports
``mean [lo, hi]`` instead of a bare number, so a reader can see whether
"isotonic beats logistic" survives resampling or not.

WHAT VARIES, AND WHAT MUST NOT
==============================
The mock repairer is seeded from ``(dataset, column, row_id)``, so the
proposed repairs are a function of ``run_pipeline``'s ``seed``. The
calibration train/eval split is a *separate* source of randomness, driven
by ``split_seed``. A "confidence interval" that moved both at once would
mix two different quantities and mean nothing.

So :func:`repeated_splits` holds ``seed`` FIXED and varies ``split_seed``
over ``base_split_seed + i``. Across those runs:

    identical   detections, repair candidates, raw_confidence,
                self_consistency, entropy  (the predictions)
    varying     calibrated p, calibrated q, and therefore the decisions,
                the realised cost and the action shares

``tests/test_split_seed.py`` asserts that isolation property directly,
including the negative half of it: at least one of p/q must actually move,
otherwise the split is not doing anything and the interval is fake.

A sweep over ``seed`` (sensitivity to mock-repair noise) is a DIFFERENT
experiment and must be reported separately — see docs/phase13_plan.md §0.

HONEST REPORTING: n_fitted
==========================
A calibrator can fail to fit on a given split (constant labels — the
documented Adult negative result). Averaging the fallback value into the
mean would quietly turn "never fitted" into a plausible-looking score.
:class:`RepeatedSplitResult` therefore carries ``n_fitted[method]``
alongside the interval, and only genuinely fitted splits contribute. A
method that fitted in 0 splits still gets an :class:`Interval`, with
``n = 0``, whose ``__str__`` renders ``"—"`` — so a table can print it
without special-casing, and a reader cannot mistake it for a score.

LEAKAGE NOTE (CLAUDE.md §3)
===========================
This module reads no columns and compares no values. It calls
``run_pipeline`` / ``decide_all`` / ``evaluate_policies``, which already
confine ``clean_df`` to the two label builders (0/1 labels for p and q) and
to the after-the-fact evaluators. ``split_seed`` only re-partitions those
already-permitted labels; it is a decision-side knob and gives the pipeline
no new sight of ground truth. In particular ``split_seed`` is NOT selected
by looking at which split scores best — the seeds are a fixed arithmetic
sequence chosen before the run, and all of them are reported.

INTERVALS
=========
``bootstrap`` (default) is a percentile bootstrap of the mean: resample the
per-split values with replacement ``n_boot`` times and take the empirical
2.5th / 97.5th percentiles. It assumes nothing about the shape of the
distribution, which matters because Brier scores over splits are bounded
and often skewed. ``normal`` is the Student-t interval
``mean ± t(1-α/2, n-1) · s/√n``, reported when a parametric interval is
wanted for comparison. On a large, well-behaved sample the two agree; where
they disagree, the bootstrap is the one to trust here.

Both are degenerate-input safe: ``n = 0`` → an all-zero interval flagged by
``n``; ``n = 1`` → ``lo == hi == mean`` and ``std == 0`` (one observation
carries no information about spread, and pretending otherwise would be the
exact dishonesty this module exists to remove).

WORKED EXAMPLE
==============
    >>> from src.risk.cost_model import CostConfig
    >>> costs = CostConfig(c_wrong_repair=50, c_human_review=5,
    ...                    c_leave_error=10, dataset_name="breast_cancer")
    >>> res = repeated_splits("breast_cancer", costs, n_splits=20)
    >>> res.split_seeds[:3]
    [1000, 1001, 1002]
    >>> print(res.brier["isotonic"])            # mean [lo, hi] over splits
    0.1899 [0.1721, 0.2077]
    >>> res.n_fitted["isotonic"]
    20
    >>> print(res.cost["calibrated_risk_aware"])
    1495.0000 [1420.0000, 1570.0000]
    >>> print(res.actions["calibrated_risk_aware"]["abstain"])
    0.3521 [0.3380, 0.3662]

An unfitted method prints without blowing up a table:

    >>> print(repeated_splits("adult", costs, n_splits=3).brier["isotonic"])
    —
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from src.pipeline.runner import (
    DEFAULT_IQR_K,
    decide_all,
    evaluate_policies,
    run_pipeline,
)
from src.risk.cost_model import CostConfig

logger = logging.getLogger(__name__)

#: Calibration methods reported by ``CalibrationComparison``.
CALIBRATION_METHODS: Tuple[str, ...] = ("raw", "logistic", "isotonic", "multivariate")

#: Action-share keys, matching ``CostReport.{auto,human,abstain}_rate``.
ACTION_KEYS: Tuple[str, ...] = ("auto", "human", "abstain")

#: Printed in place of an interval that has no observations behind it.
NOT_FITTED_MARKER: str = "—"

DEFAULT_N_SPLITS: int = 20
DEFAULT_BASE_SPLIT_SEED: int = 1000
DEFAULT_N_BOOT: int = 10_000
INTERVAL_METHODS: Tuple[str, ...] = ("bootstrap", "normal")


# ═══════════════════════════════════════════════════════════════════════════════
# Interval
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Interval:
    """
    A point estimate with a confidence interval and the sample behind it.

    Attributes
    ----------
    mean : float
        Sample mean of the observed values.
    lo, hi : float
        Lower and upper confidence bound. ``lo <= mean <= hi`` always holds
        for the intervals produced by this module.
    std : float
        Sample standard deviation (ddof=1). ``0.0`` when ``n <= 1``.
    n : int
        Number of observations. ``0`` means "no observations at all" — the
        interval is a placeholder so tables can print it, NOT a score. See
        :meth:`is_empty` and :meth:`__str__`.
    """

    mean: float
    lo: float
    hi: float
    std: float
    n: int

    @property
    def is_empty(self) -> bool:
        """True when no observation backs this interval (``n == 0``)."""
        return self.n == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": float(self.mean),
            "lo": float(self.lo),
            "hi": float(self.hi),
            "std": float(self.std),
            "n": int(self.n),
        }

    def __str__(self) -> str:
        """``"0.1899 [0.1721, 0.2077]"``, or ``"—"`` when ``n == 0``."""
        if self.is_empty:
            return NOT_FITTED_MARKER
        return f"{self.mean:.4f} [{self.lo:.4f}, {self.hi:.4f}]"


def _empty_interval() -> Interval:
    """The ``n = 0`` placeholder. Renders as ``"—"``."""
    return Interval(mean=0.0, lo=0.0, hi=0.0, std=0.0, n=0)


def _prepare(values: Sequence[float], confidence: float) -> np.ndarray:
    """Validate ``confidence`` and coerce ``values`` to a 1-D float array."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    arr = np.asarray(list(values), dtype=float).ravel()
    if arr.size and not np.all(np.isfinite(arr)):
        raise ValueError("values must all be finite (no NaN / inf).")
    return arr


def _sample_std(arr: np.ndarray) -> float:
    """Sample std (ddof=1); 0.0 for fewer than two observations."""
    return float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def bootstrap_interval(
    values: Sequence[float],
    confidence: float = 0.95,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 42,
) -> Interval:
    """
    Percentile bootstrap interval for the mean of ``values``.

    Resamples ``values`` with replacement ``n_boot`` times, takes the mean
    of each resample, and reports the empirical ``(1-confidence)/2`` and
    ``1-(1-confidence)/2`` percentiles. Makes no distributional assumption.

    Deterministic: the resampling uses ``numpy.random.default_rng(seed)``,
    so the same ``(values, confidence, n_boot, seed)`` always yields the
    same interval.

    Parameters
    ----------
    values : sequence of float
        Per-split observations. Empty → ``Interval`` with ``n = 0``.
    confidence : float
        In (0, 1). Default 0.95.
    n_boot : int
        Resamples. Must be >= 1.
    seed : int
        Bootstrap RNG seed. Must be >= 0.

    Notes
    -----
    A single observation gives ``lo == hi == mean``: every resample is that
    same value. That is the honest answer, not a bug.
    """
    arr = _prepare(values, confidence)
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    if arr.size == 0:
        return _empty_interval()

    mean = float(arr.mean())
    if arr.size == 1:
        return Interval(mean=mean, lo=mean, hi=mean, std=0.0, n=1)
    if np.all(arr == arr[0]):
        # Every resample is the same number. Short-circuit so the bounds are
        # exactly the mean rather than a float-rounding hair away from it.
        return Interval(mean=mean, lo=mean, hi=mean, std=0.0, n=int(arr.size))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    alpha = 1.0 - confidence
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # The point estimate is the sample mean, so clamp the bounds around it
    # (they can only cross it in pathological ties).
    return Interval(
        mean=mean,
        lo=float(min(lo, mean)),
        hi=float(max(hi, mean)),
        std=_sample_std(arr),
        n=int(arr.size),
    )


def normal_interval(values: Sequence[float], confidence: float = 0.95) -> Interval:
    """
    Student-t confidence interval for the mean of ``values``.

    ``mean ± t(1 - α/2, n - 1) · s / √n`` with ``s`` the sample standard
    deviation (ddof=1). The t distribution (not the normal) is used because
    ``n`` is the number of splits — 20, not 2,000.

    Empty → ``Interval`` with ``n = 0``. A single observation, or
    all-identical observations, give ``lo == hi == mean`` and ``std == 0``.
    """
    arr = _prepare(values, confidence)
    if arr.size == 0:
        return _empty_interval()

    mean = float(arr.mean())
    if arr.size == 1:
        return Interval(mean=mean, lo=mean, hi=mean, std=0.0, n=1)

    if np.all(arr == arr[0]):
        return Interval(mean=mean, lo=mean, hi=mean, std=0.0, n=int(arr.size))

    std = _sample_std(arr)
    sem = std / np.sqrt(arr.size)
    half = float(stats.t.ppf(1.0 - (1.0 - confidence) / 2.0, df=arr.size - 1) * sem)
    return Interval(
        mean=mean, lo=mean - half, hi=mean + half, std=std, n=int(arr.size)
    )


def _make_interval(
    values: Sequence[float],
    interval: str,
    confidence: float,
    n_boot: int,
    seed: int,
) -> Interval:
    """Dispatch to :func:`bootstrap_interval` or :func:`normal_interval`."""
    if interval == "bootstrap":
        return bootstrap_interval(values, confidence=confidence, n_boot=n_boot, seed=seed)
    if interval == "normal":
        return normal_interval(values, confidence=confidence)
    raise ValueError(
        f"interval must be one of {INTERVAL_METHODS}, got {interval!r}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RepeatedSplitResult
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RepeatedSplitResult:
    """
    Aggregated result of running one dataset over many calibration splits.

    Attributes
    ----------
    dataset : str
    n_splits : int
        Number of split seeds actually run.
    seed : int
        The pipeline seed, held FIXED across every split (predictions do
        not move; only the calibrators do).
    split_seeds : list of int
        The seeds used, in order: ``base_split_seed + i``.
    brier, ece : dict
        Calibration method → :class:`Interval` over the splits in which
        that method genuinely fitted.
    n_fitted : dict
        Calibration method → how many splits it fitted in. ``raw`` is
        always ``n_splits``; a 0 here means the matching interval is the
        ``n = 0`` placeholder and prints as ``"—"``.
    cost : dict
        Policy name → interval of REALISED cost (total penalty units, not a
        mean per record; ``CostReport.total_realized_cost``).
    actions : dict
        Policy name → action (``auto``/``human``/``abstain``) → interval of
        that action's share of decided records.
    interval : str
        Which interval method produced the numbers (``"bootstrap"`` or
        ``"normal"``), so a table can say so.
    n_decided : int
        Decided records per split (identical across splits — the split seed
        does not change what is detected).

    Costs are abstract experimental penalty units, never money
    (CLAUDE.md §2).
    """

    dataset: str
    n_splits: int
    seed: int
    split_seeds: List[int]
    brier: Dict[str, Interval] = field(default_factory=dict)
    ece: Dict[str, Interval] = field(default_factory=dict)
    n_fitted: Dict[str, int] = field(default_factory=dict)
    cost: Dict[str, Interval] = field(default_factory=dict)
    actions: Dict[str, Dict[str, Interval]] = field(default_factory=dict)
    interval: str = "bootstrap"
    n_decided: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "n_splits": int(self.n_splits),
            "seed": int(self.seed),
            "split_seeds": [int(s) for s in self.split_seeds],
            "brier": {m: iv.to_dict() for m, iv in self.brier.items()},
            "ece": {m: iv.to_dict() for m, iv in self.ece.items()},
            "n_fitted": {m: int(n) for m, n in self.n_fitted.items()},
            "cost": {p: iv.to_dict() for p, iv in self.cost.items()},
            "actions": {
                p: {a: iv.to_dict() for a, iv in acts.items()}
                for p, acts in self.actions.items()
            },
            "interval": self.interval,
            "n_decided": int(self.n_decided),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# repeated_splits
# ═══════════════════════════════════════════════════════════════════════════════


def repeated_splits(
    dataset: str,
    costs: CostConfig,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = 42,
    base_split_seed: int = DEFAULT_BASE_SPLIT_SEED,
    iqr_k: float = DEFAULT_IQR_K,
    flag_outliers: Optional[bool] = None,
    n_records: int = 0,
    data_root: str | Path = "datasets",
    interval: str = "bootstrap",
    confidence: float = 0.95,
    n_boot: int = DEFAULT_N_BOOT,
    threshold: float = 0.8,
) -> RepeatedSplitResult:
    """
    Run ``dataset`` over ``n_splits`` calibration splits and aggregate.

    ``seed`` is held fixed and ``split_seed`` takes the values
    ``base_split_seed + i`` for ``i in range(n_splits)``. For each split the
    full pipeline runs once (it is the expensive part — one call per split
    seed, never one per metric), then ``decide_all`` and
    ``evaluate_policies`` score it against the given costs.

    Collected per split: Brier and ECE for each calibration method that
    genuinely fitted, realised cost per policy, and action shares per
    policy. Each series is turned into an :class:`Interval`.

    Parameters
    ----------
    dataset : str
        One of ``src.pipeline.runner.DATASETS``.
    costs : CostConfig
        Experimental penalty units (never money).
    n_splits : int
        Must be >= 1.
    seed : int
        Pipeline seed, fixed across splits. Also seeds the bootstrap, so
        two identical calls give byte-identical intervals.
    base_split_seed : int
        First split seed. Must be >= 0.
    iqr_k, flag_outliers, n_records, data_root :
        Passed straight to :func:`run_pipeline`. ``n_records=0`` (the
        default here, unlike the demo default) means the FULL dataset.
    interval : {"bootstrap", "normal"}
    confidence : float
        In (0, 1).
    n_boot : int
        Bootstrap resamples (ignored when ``interval="normal"``).
    threshold : float
        Confidence threshold for the Phase 9 baseline inside
        ``evaluate_policies``.

    Raises
    ------
    ValueError
        ``n_splits < 1``, a bad ``interval``, or a negative
        ``base_split_seed``.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    if interval not in INTERVAL_METHODS:
        raise ValueError(
            f"interval must be one of {INTERVAL_METHODS}, got {interval!r}"
        )
    if base_split_seed < 0:
        raise ValueError(f"base_split_seed must be >= 0, got {base_split_seed}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    split_seeds = [base_split_seed + i for i in range(n_splits)]

    brier_vals: Dict[str, List[float]] = {m: [] for m in CALIBRATION_METHODS}
    ece_vals: Dict[str, List[float]] = {m: [] for m in CALIBRATION_METHODS}
    n_fitted: Dict[str, int] = {m: 0 for m in CALIBRATION_METHODS}
    cost_vals: Dict[str, List[float]] = {}
    action_vals: Dict[str, Dict[str, List[float]]] = {}
    n_decided = 0

    for split_seed in split_seeds:
        # ONE pipeline run per split seed. Detection and repair are
        # unaffected by split_seed; the calibrators (and hence p, q) are.
        bundle = run_pipeline(
            dataset,
            n_records=n_records,
            seed=seed,
            data_root=data_root,
            iqr_k=iqr_k,
            flag_outliers=flag_outliers,
            split_seed=split_seed,
        )
        decisions = decide_all(bundle, costs)
        if len(decisions) != bundle.n_records:
            raise RuntimeError(
                f"decide_all returned {len(decisions)} decisions for "
                f"{bundle.n_records} records (split_seed={split_seed})."
            )
        ev = evaluate_policies(bundle, costs, threshold=threshold)
        n_decided = len(bundle.decision_indices())

        comparison = bundle.comparison
        if comparison is not None:
            for method in CALIBRATION_METHODS:
                if not comparison.fitted(method):
                    continue
                n_fitted[method] += 1
                brier_vals[method].append(
                    float(getattr(comparison, f"{method}_brier"))
                )
                ece_vals[method].append(float(getattr(comparison, f"{method}_ece")))

        for name, report in ev.cost_reports.items():
            cost_vals.setdefault(name, []).append(
                float(report.total_realized_cost)
            )
            shares = action_vals.setdefault(name, {a: [] for a in ACTION_KEYS})
            for action in ACTION_KEYS:
                shares[action].append(float(getattr(report, f"{action}_rate")))

    def iv(values: Sequence[float]) -> Interval:
        return _make_interval(values, interval, confidence, n_boot, seed)

    result = RepeatedSplitResult(
        dataset=dataset,
        n_splits=n_splits,
        seed=seed,
        split_seeds=split_seeds,
        brier={m: iv(brier_vals[m]) for m in CALIBRATION_METHODS},
        ece={m: iv(ece_vals[m]) for m in CALIBRATION_METHODS},
        n_fitted=dict(n_fitted),
        cost={name: iv(vals) for name, vals in cost_vals.items()},
        actions={
            name: {a: iv(shares[a]) for a in ACTION_KEYS}
            for name, shares in action_vals.items()
        },
        interval=interval,
        n_decided=n_decided,
    )
    logger.info(
        "repeated_splits '%s': %d splits (seed=%d, split_seeds %d..%d), "
        "%d decided records; fitted counts %s",
        dataset, n_splits, seed, split_seeds[0], split_seeds[-1],
        n_decided, n_fitted,
    )
    return result
