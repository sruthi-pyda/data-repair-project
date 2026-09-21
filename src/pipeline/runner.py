"""
src/pipeline/runner.py
----------------------
Shared end-to-end runner for the dashboard (Phase 11) and the experiment
script (Phase 13). It wires Phases 1–10 together in exactly the order the
demo scripts use, and splits the work into three stages so a UI can re-run
the cheap stages interactively:

    run_pipeline()       load → profile → detect → repair → uncertainty →
                         calibrate                              (seconds)
    decide_all()         calibrated p + cost config → decision per cell
                                                                (milliseconds)
    evaluate_policies()  proposed method vs. baselines, scored against
                         ground truth                           (milliseconds)

DATA LEAKAGE RULES (CLAUDE.md §3)
==================================
``clean_df`` is ground truth and may only be read after predictions exist.

    Stage                       Reads clean_df?   Why
    -------------------------   ---------------   ------------------------------
    profile_dataset()           yes (aggregate)   documented soft-leakage caveat:
                                                  column kinds, quartiles, vocab
    Detector.detect()           no
    MockRepairer.repair_batch() no
    UncertaintyEstimator        no
    CalibrationBuilder.build()  yes (labels)      the ONE legitimate entry point
    calibrator.fit/predict      no                sees (signals, 0/1) only
    decide_all()                no                takes p and costs, nothing else
    evaluate_policies()         yes               scoring after the fact

``decide_all`` deliberately has no ``clean_df`` parameter and ``PipelineBundle``
exposes the decision inputs (``unc_records``, ``calibrated_probs``) separately
from the evaluation inputs (``clean_df``), so a caller cannot accidentally
route ground truth into a decision. ``tests/test_pipeline_runner.py`` asserts
both properties.

CALIBRATOR SELECTION
====================
Logistic, isotonic and multivariate calibrators are fitted on a 70/30 split
and the one with the lowest evaluation Brier score is used for every record.
When no calibrator can fit (Adult: every correctness label is 0) the
logistic fallback is used, which collapses to the empirical base rate. The
bundle records this in ``calibration_method`` and ``calibration_note`` so the
UI can say so instead of showing a silently degenerate number.

Worked example
==============
    >>> bundle = run_pipeline("breast_cancer", n_records=200)
    >>> costs = CostConfig(c_wrong_repair=50, c_human_review=5,
    ...                    c_leave_error=10, dataset_name="breast_cancer")
    >>> decisions = decide_all(bundle, costs)
    >>> decisions[0].action
    <DecisionAction.HUMAN_REVIEW: 'HUMAN_REVIEW'>
    >>> ev = evaluate_policies(bundle, costs, threshold=0.8)
    >>> [r.method_name for r in ev.rows]
    ['calibrated_risk_aware', 'always_auto', 'confidence_threshold_0.80']
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.baselines.baselines import (
    AlwaysAutoBaseline,
    BaselineRecord,
    ConfidenceThresholdBaseline,
    wrap_proposed_method,
)
from src.calibration.calibrator import (
    CalibrationBuilder,
    CalibrationComparison,
    CalibrationSample,
    IsotonicCalibrator,
    LogisticCalibrator,
    MultivariateLogisticCalibrator,
    calibrate_dataset,
    samples_to_feature_matrix,
    train_eval_split,
)
from src.calibration.metrics import brier_score
from src.data.loader import DataLoader
from src.data.profiler import profile_dataset
from src.data.schema import DatasetProfile
from src.decision.policy import DecisionPolicy, DecisionResult
from src.detection.detector import DetectionRecord, Detector
from src.evaluation.cost_metrics import (
    CostEvaluator,
    CostReport,
    PolicyComparisonRow,
    compare_policies,
)
from src.evaluation.repair_metrics import RepairEvaluator, RepairMetrics
from src.repair.mock_repair import MockRepairer, RepairResult
from src.risk.cost_model import CostConfig
from src.uncertainty.estimator import UncertaintyEstimator, UncertaintyRecord

logger = logging.getLogger(__name__)

DATASETS: Tuple[str, ...] = ("breast_cancer", "smartfactory", "adult")

# ID-like columns whose "repairs" are meaningless; mirrors demo_risk.py.
SKIP_COLUMNS: Dict[str, List[str]] = {
    "breast_cancer": ["Unnamed: 0", "Sample code number"],
    "adult": [],
    "smartfactory": [],
}

# Datasets whose non-NaN values are known valid, so IQR fences only produce
# false positives (configs/default.yaml: dataset_overrides).
_NO_OUTLIER_FLAGGING = frozenset({"smartfactory"})

DEFAULT_EVAL_FRACTION = 0.30


# ═══════════════════════════════════════════════════════════════════════════════
# Bundle
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PipelineBundle:
    """
    Everything produced by ``run_pipeline`` for one dataset.

    Decision inputs (no ground truth inside):
        profile, dirty_df, detections, repair_results, unc_records,
        calibrated_probs, calibration_method, calibration_note
    Evaluation inputs (ground truth):
        clean_df, cal_samples, comparison

    ``unc_records``, ``cal_samples`` and ``calibrated_probs`` are aligned by
    index. ``cal_samples`` can be shorter than ``unc_records`` only if a
    record's column is absent from clean_df, which never happens for the
    shipped datasets; ``run_pipeline`` raises if they diverge.
    """

    dataset: str
    n_requested: int
    seed: int
    n_samples: int
    top_k: int

    profile: DatasetProfile
    clean_df: pd.DataFrame
    dirty_df: pd.DataFrame

    detections: List[DetectionRecord]
    repair_results: List[RepairResult]
    unc_records: List[UncertaintyRecord]

    cal_samples: List[CalibrationSample]
    comparison: Optional[CalibrationComparison]
    calibration_method: str
    calibration_note: Optional[str]
    calibrated_probs: List[float]

    n_detected_total: int = 0

    @property
    def n_records(self) -> int:
        return len(self.unc_records)

    @property
    def raw_confidences(self) -> List[float]:
        return [r.raw_confidence for r in self.unc_records]

    @property
    def skip_columns(self) -> List[str]:
        return SKIP_COLUMNS.get(self.dataset, [])

    def decision_indices(self) -> List[int]:
        """Indices of records that are not in an ID-like skipped column."""
        skip = set(self.skip_columns)
        return [i for i, r in enumerate(self.unc_records) if r.column not in skip]


@dataclass
class PolicyEvaluation:
    """Ground-truth scored comparison of the proposed method and baselines."""

    dataset: str
    costs: CostConfig
    threshold: float
    records: Dict[str, List[BaselineRecord]] = field(default_factory=dict)
    cost_reports: Dict[str, CostReport] = field(default_factory=dict)
    repair_reports: Dict[str, RepairMetrics] = field(default_factory=dict)
    rows: List[PolicyComparisonRow] = field(default_factory=list)
    threshold_sweep: Dict[float, CostReport] = field(default_factory=dict)

    @property
    def proposed_name(self) -> str:
        return "calibrated_risk_aware"


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1: run_pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def run_pipeline(
    dataset: str,
    n_records: int = 500,
    seed: int = 42,
    n_samples: int = 5,
    top_k: int = 3,
    data_root: str | Path = "datasets",
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
) -> PipelineBundle:
    """
    Run Phases 1–6 for one dataset and return a :class:`PipelineBundle`.

    Parameters
    ----------
    dataset : str
        One of :data:`DATASETS`.
    n_records : int
        Cap on detection records carried forward (demo speed). ``<= 0`` means
        no cap.
    seed, n_samples, top_k :
        Passed to MockRepairer / UncertaintyEstimator.
    data_root : path
        Folder containing ``<dataset>/{clean,dirty}.csv``.
    eval_fraction : float
        Calibration hold-out fraction.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {DATASETS}.")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    loader = DataLoader(data_root)
    clean_df, dirty_df = loader.load(dataset)
    profile = profile_dataset(dataset, loader.discover(dataset), clean_df, dirty_df)

    # ── Phase 3: detection (dirty data + population-level profile only) ──
    detector = Detector(
        profile, flag_statistical_outliers=dataset not in _NO_OUTLIER_FLAGGING
    )
    all_detections = detector.detect(dirty_df)
    detections = all_detections[:n_records] if n_records > 0 else all_detections

    # ── Phases 4–5: repair + uncertainty (no clean_df) ────────────────────
    repairer = MockRepairer(seed=seed, top_k=top_k)
    repair_results = repairer.repair_batch(detections, dirty_df, profile)
    estimator = UncertaintyEstimator(base_seed=seed, n_samples=n_samples)
    unc_records = estimator.estimate_batch(repair_results, dirty_df, profile)

    # ── Phase 6: calibration (clean_df enters here, for 0/1 labels only) ──
    cal_samples = CalibrationBuilder().build(unc_records, clean_df, dirty_df, dataset)
    if len(cal_samples) != len(unc_records):
        raise RuntimeError(
            f"CalibrationBuilder dropped records ({len(cal_samples)} of "
            f"{len(unc_records)}); alignment with unc_records is lost."
        )

    comparison: Optional[CalibrationComparison] = None
    method, note, probs = "none", None, []
    if cal_samples:
        comparison = calibrate_dataset(
            cal_samples, dataset, eval_fraction=eval_fraction, seed=seed
        )
        method, note, probs = _calibrate_all(cal_samples, seed, eval_fraction)

    logger.info(
        "run_pipeline '%s': %d/%d detections, calibration=%s",
        dataset, len(detections), len(all_detections), method,
    )
    return PipelineBundle(
        dataset=dataset,
        n_requested=n_records,
        seed=seed,
        n_samples=n_samples,
        top_k=top_k,
        profile=profile,
        clean_df=clean_df,
        dirty_df=dirty_df,
        detections=detections,
        repair_results=repair_results,
        unc_records=unc_records,
        cal_samples=cal_samples,
        comparison=comparison,
        calibration_method=method,
        calibration_note=note,
        calibrated_probs=probs,
        n_detected_total=len(all_detections),
    )


def _calibrate_all(
    samples: List[CalibrationSample], seed: int, eval_fraction: float
) -> Tuple[str, Optional[str], List[float]]:
    """
    Fit the three calibrators on a split, pick the lowest eval Brier, and
    return calibrated probabilities for *every* sample in input order.

    Returns (method_name, note, probs). ``note`` is set when the result is a
    fallback rather than a fitted calibrator.
    """
    train, evalset = train_eval_split(samples, eval_fraction=eval_fraction, seed=seed)
    X_all = np.array([s.raw_confidence for s in samples], dtype=float)

    if len(train) < 3 or len(evalset) < 1:
        return (
            "raw (too few samples to calibrate)",
            f"Only {len(samples)} calibration samples; raw confidence used as-is.",
            [float(v) for v in np.clip(X_all, 0.0, 1.0)],
        )

    X_tr = np.array([s.raw_confidence for s in train], dtype=float)
    y_tr = np.array([s.correct for s in train], dtype=float)
    X_ev = np.array([s.raw_confidence for s in evalset], dtype=float)
    y_ev = np.array([s.correct for s in evalset], dtype=float)

    log_cal, iso_cal, multi_cal = (
        LogisticCalibrator(), IsotonicCalibrator(), MultivariateLogisticCalibrator()
    )
    log_cal.fit(X_tr, y_tr)
    iso_cal.fit(X_tr, y_tr)
    has_multi = all(s.has_multivariate_features() for s in samples)
    if has_multi:
        multi_cal.fit(samples_to_feature_matrix(train), y_tr)

    candidates: List[Tuple[str, float, np.ndarray]] = []
    if log_cal.is_fitted():
        candidates.append(("logistic", brier_score(log_cal.predict(X_ev), y_ev),
                           log_cal.predict(X_all)))
    if iso_cal.is_fitted():
        candidates.append(("isotonic", brier_score(iso_cal.predict(X_ev), y_ev),
                           iso_cal.predict(X_all)))
    if multi_cal.is_fitted():
        candidates.append((
            "multivariate",
            brier_score(multi_cal.predict(samples_to_feature_matrix(evalset)), y_ev),
            multi_cal.predict(samples_to_feature_matrix(samples)),
        ))

    if candidates:
        name, _, preds = min(candidates, key=lambda c: c[1])
        return name, None, [float(v) for v in preds]

    # Nothing fitted: constant labels. LogisticCalibrator's fallback encodes
    # the empirical base rate, which is the honest probability here.
    base_rate = float(y_tr.mean()) if len(y_tr) else 0.0
    preds = log_cal.predict(X_all)
    return (
        "base-rate fallback (no calibrator fitted)",
        (
            f"All {len(train)} training labels are "
            f"{'correct' if base_rate > 0.5 else 'incorrect'}; no calibrator "
            f"can fit. Calibrated probability collapses to the base rate "
            f"({base_rate:.2f}). This is the documented Adult negative result."
        ),
        [float(v) for v in np.clip(preds, 0.0, 1.0)],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2: decide_all  (NO clean_df — by construction)
# ═══════════════════════════════════════════════════════════════════════════════


def decide_all(bundle: PipelineBundle, costs: CostConfig) -> List[DecisionResult]:
    """
    Apply the Phase 8 policy to every record's calibrated probability.

    Takes only the bundle's calibrated probabilities and the cost config.
    Returns one DecisionResult per ``unc_record`` (same order), including
    skipped ID-like columns so indices stay aligned; use
    ``bundle.decision_indices()`` to filter those out for reporting.
    """
    policy = DecisionPolicy.from_cost_config(costs)
    return [policy.decide(p) for p in bundle.calibrated_probs]


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3: evaluate_policies  (ground truth used here, after decisions exist)
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_policies(
    bundle: PipelineBundle,
    costs: CostConfig,
    threshold: float = 0.8,
    sweep: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
) -> PolicyEvaluation:
    """
    Score the proposed method against the two Phase 9 baselines.

    Records in ``bundle.skip_columns`` are excluded from every method so the
    comparison is like-for-like (mirrors demo_risk.py).
    """
    idx = bundle.decision_indices()
    unc = [bundle.unc_records[i] for i in idx]
    probs = [bundle.calibrated_probs[i] for i in idx]
    decisions = [decide_all(bundle, costs)[i] for i in idx]

    ev = PolicyEvaluation(dataset=bundle.dataset, costs=costs, threshold=threshold)
    if not unc:
        return ev

    ev.records[ev.proposed_name] = wrap_proposed_method(
        unc, probs, decisions, dataset_name=bundle.dataset
    )
    ev.records["always_auto"] = AlwaysAutoBaseline(bundle.dataset).run(unc)
    thr_baseline = ConfidenceThresholdBaseline(threshold=threshold, dataset_name=bundle.dataset)
    thr_records = thr_baseline.run(unc)
    ev.records[thr_records[0].method_name] = thr_records

    cost_eval = CostEvaluator.from_cost_config(bundle.clean_df, costs)
    repair_eval = RepairEvaluator(bundle.clean_df, bundle.dirty_df, bundle.dataset)

    cost_reports, repair_reports = [], []
    for name, recs in ev.records.items():
        cr = cost_eval.evaluate(recs, dataset=bundle.dataset)
        rr = repair_eval.evaluate(recs)
        ev.cost_reports[name] = cr
        ev.repair_reports[name] = rr
        cost_reports.append(cr)
        repair_reports.append(rr)
    ev.rows = compare_policies(cost_reports, repair_reports)

    for t in sweep:
        recs = ConfidenceThresholdBaseline(threshold=t, dataset_name=bundle.dataset).run(unc)
        ev.threshold_sweep[t] = cost_eval.evaluate(recs, dataset=bundle.dataset)

    return ev
