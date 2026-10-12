#!/usr/bin/env python
"""
run_experiment.py
-----------------
Phase 13, Step 2: the one command that regenerates **every** reported
number and figure for this project.

    python run_experiment.py --dataset breast_cancer
    python run_experiment.py --all
    python run_experiment.py --quick --all          # CI / smoke, NOT paper-grade
    python run_experiment.py --all --mode real --llm-live   # Phase 12 repairs

REPAIR MODE (Phase 12)
======================
``--mode mock`` (the default) is the historical path. ``--mode real`` runs
the pipeline with ``src/repair/llm_repair.py`` and needs two things: a usable
key (``LLM_API_KEY``, or ``LLM_API_KEYS`` for the rotating pool, in a
gitignored ``.env``) **and** ``--llm-live``. Without a key it exits 2 naming
the variable and the file; without ``--llm-live`` it exits 2 rather than
spending roughly 13,314 billed calls because a flag was mistyped. No key is
ever printed, logged or written to an artifact.

Every table states the mode, the model id and the per-dataset fallback count
and rate, because a real-mode run in which most cells fell back to the mock
repairer is a mock run with a label on it. ``--no-llm-cache`` forces fresh
calls; the cache is on by default, which makes a re-run free and exactly
reproducible.

**One real limit, stated everywhere it matters:** ``repeated_splits``
(``src/experiments/aggregate.py``) builds its own ``MockRepairer`` and takes
no repair mode, so T2 and T4's cost interval describe the *mock* pipeline's
calibration even in real mode. T1, T3 and the predictions CSV use the real
repairs.

Before this script existed, every number in the project came from a demo
script or an ad-hoc ablation, each with its own record cap, its own seed
handling and its own cost arithmetic (docs/phase13_plan.md §0). Nothing
here is hand-written into a table: each cell is produced by code that ran
in this process, and a quantity that cannot be computed is printed as
``—`` with the reason in that table's notes.

WHAT IT WRITES
==============
    <out>/tables/t1_repair_quality.md        + .csv
    <out>/tables/t2_calibration.md           + .csv
    <out>/tables/t3_risk_aware.md            + .csv
    <out>/tables/t4_policy_comparison.md     + .csv
    <out>/figures/<dataset>_reliability.png
    <out>/figures/<dataset>_confidence.png
    <out>/figures/<dataset>_cost_comparison.png
    <out>/figures/<dataset>_action_distribution.png
    <out>/figures/<dataset>_threshold_sweep.png
    <out>/figures/<dataset>_decision_map.png
    <out>/predictions/<dataset>.csv
    <out>/run_metadata.json

TWO SOURCES OF NUMBERS, DELIBERATELY NOT MIXED
==============================================
``seed`` drives mock-repair / uncertainty noise; ``split_seed`` drives the
calibration train/eval split only (docs/phase13_plan.md §0). So:

* **T1** (repair quality) and **T3** (risk-aware performance) come from one
  *representative* run at ``seed = split_seed = --seed``. Point estimates,
  stated as such in the table notes.
* **T2** (calibration) and the cost column of **T4** come from
  ``src.experiments.aggregate.repeated_splits``, which holds ``seed`` fixed
  and varies ``split_seed``, so the interval is an interval over *splits*
  and nothing else. Where a T3 quantity also has an interval available it
  is printed beside the point estimate.

LEAKAGE RULES (CLAUDE.md §3)
============================
This script never passes ``clean_df`` into a decision. It calls
``run_pipeline`` → ``decide_all`` → ``evaluate_policies`` in that order;
ground truth enters only inside ``CalibrationBuilder`` /
``DetectionLabelBuilder`` (0/1 labels) and inside ``src/evaluation`` (after
every decision exists). The predictions CSV carries two ground-truth
columns; they are **grouped last**, prefixed ``eval_``, and the file's
header comment names them as evaluation-only.

HONEST-REPORTING RULES (docs/phase13_plan.md §0, docs/cost_semantics.md)
========================================================================
* All costs are abstract **experimental penalty units**, never money and
  never supplied by the REIN benchmark.
* Adult's realised cost of 0 is printed as *zero residual-corruption cost
  on the flagged cells*, always beside its undetected-corruption count.
* Every row produced under a calibrator fallback (Phase 6 base-rate, or
  q = 1) is marked.
* False negatives are reported per dataset; the cost model prices them at
  zero, which is a scope limit on every cost number here.
* If the proposed method loses on a dataset, the closing summary says so.

Worked example
==============
    $ python run_experiment.py --quick --dataset breast_cancer
    ... four tables printed ...
    [written] results/tables/t1_repair_quality.md
    ...
    Honest reading
    - breast_cancer: proposed 1620.00 vs best baseline 1775.00 (always_human) — proposed wins.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless: figures are saved, never shown
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.baselines.baselines import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    BaselineRecord,
)
from src.console import enable_unicode_stdout  # noqa: E402
from src.decision.policy import DecisionAction, DecisionResult  # noqa: E402
from src.evaluation.cost_metrics import CostEvaluator, CostReport  # noqa: E402
from src.evaluation.repair_metrics import RepairEvaluator, RepairMetrics  # noqa: E402
from src.experiments.aggregate import Interval, RepeatedSplitResult, repeated_splits  # noqa: E402
from src.pipeline import (  # noqa: E402
    DATASETS,
    PipelineBundle,
    PolicyEvaluation,
    decide_all,
    evaluate_policies,
    run_pipeline,
)
from src.repair.llm_repair import LLMConfig  # noqa: E402
from src.risk.cost_model import CostConfig  # noqa: E402
from src.visualization import plots  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_CHOICES: Tuple[str, ...] = tuple(sorted(DATASETS))
COSTS_YAML = Path("configs/costs.yaml")

NOT_COMPUTED = "—"
COST_UNITS = "experimental penalty units"

# --quick: small enough for CI, explicitly not paper-grade.
QUICK_N_RECORDS = 400
QUICK_SPLITS = 3

# Confidence threshold for the headline raw-confidence baseline. The sweep
# (DEFAULT_THRESHOLDS) reports every threshold, so this only picks which one
# is called "the" threshold baseline in T3/T4.
BASELINE_THRESHOLD = 0.8

PROPOSED = "calibrated_risk_aware"

# A real-mode run at or above this fallback rate is flagged on every table:
# at that point most "LLM repairs" are mock repairs wearing a label.
HIGH_FALLBACK_RATE = 0.20
ALWAYS_HUMAN = "always_human"
ALWAYS_ABSTAIN = "always_abstain"

FIGURE_KINDS: Tuple[str, ...] = (
    "reliability",
    "confidence",
    "cost_comparison",
    "action_distribution",
    "threshold_sweep",
    "decision_map",
)

TABLE_STEMS: Tuple[str, ...] = (
    "t1_repair_quality",
    "t2_calibration",
    "t3_risk_aware",
    "t4_policy_comparison",
)

CALIBRATION_METHODS: Tuple[str, ...] = ("raw", "logistic", "isotonic", "multivariate")

QUICK_BANNER = (
    "> ⚠ **`--quick` run — NOT paper-grade.** Detections are capped at "
    f"{QUICK_N_RECORDS} records per dataset and calibration uses only "
    f"{QUICK_SPLITS} splits. This mode exists for tests and CI. Paper numbers "
    "require `python run_experiment.py --all` (no cap, 20 splits)."
)

# Columns of <out>/predictions/<dataset>.csv. Decision inputs first, then the
# two ground-truth columns, grouped last and prefixed `eval_`.
PREDICTION_DECISION_COLUMNS: Tuple[str, ...] = (
    "dataset",
    "row_id",
    "column",
    "detection_method",
    "error_type",
    "dirty_value",
    # Per-cell repair provenance: 'mock', or in real mode 'llm' /
    # 'llm->mock (<reason>)'. A fallback is visible row by row rather than
    # only in the aggregate fallback rate.
    "repair_method",
    "top_candidate",
    "raw_confidence",
    "self_consistency",
    "normalized_entropy",
    "p_repair_correct",
    "q_cell_corrupted",
    "expected_cost_auto",
    "expected_cost_human",
    "expected_cost_abstain",
    "action",
)
PREDICTION_EVAL_ONLY_COLUMNS: Tuple[str, ...] = (
    "eval_repair_correct",
    "eval_cell_genuinely_corrupted",
)
PREDICTION_COLUMNS = PREDICTION_DECISION_COLUMNS + PREDICTION_EVAL_ONLY_COLUMNS


# ═══════════════════════════════════════════════════════════════════════════════
# Small formatting helpers (every table cell goes through one of these)
# ═══════════════════════════════════════════════════════════════════════════════


def _f(value: Optional[float], precision: int = 4) -> str:
    """Fixed-precision float, or ``—`` when there is no number."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return NOT_COMPUTED
    return f"{value:.{precision}f}"


def _i(value: Optional[int]) -> str:
    return NOT_COMPUTED if value is None else f"{value:,}"


def _nonempty(iv: Optional[Interval]) -> Optional[Interval]:
    """
    Drop the ``n = 0`` placeholder interval.

    ``aggregate`` returns an empty :class:`Interval` (``n == 0``) where a
    series had no observations — a placeholder, explicitly "NOT a score".
    Printing its zeros would read as a perfect result, so it becomes ``—``.
    """
    if iv is None or getattr(iv, "is_empty", False) or getattr(iv, "n", 1) == 0:
        return None
    return iv


def _interval_str(iv: Optional[Interval], precision: int = 4) -> str:
    """``mean [lo, hi]`` via the contract's ``__str__``, or ``—``."""
    iv = _nonempty(iv)
    if iv is None:
        return NOT_COMPUTED
    if precision == 4:
        return str(iv)  # the contract's own rendering
    return f"{iv.mean:,.{precision}f} [{iv.lo:,.{precision}f}, {iv.hi:,.{precision}f}]"


def _lookup(mapping: Optional[Dict[str, Any]], *names: str) -> Optional[Any]:
    """First present key out of ``names`` (tolerant of naming variations)."""
    if not mapping:
        return None
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _action_interval(
    actions: Optional[Dict[str, Dict[str, Interval]]],
    policy: str,
    action: DecisionAction,
) -> Optional[Interval]:
    per_policy = _lookup(actions, policy, PROPOSED, "proposed")
    if not isinstance(per_policy, dict):
        return None
    return _nonempty(
        _lookup(
            per_policy,
            action.value,
            action.value.lower(),
            action.name,
            action.value.split("_")[0].lower(),
        )
    )


def _csv_interval_fields(iv: Optional[Interval]) -> Tuple[str, str, str, str, str]:
    """(mean, lo, hi, std, n) as CSV strings; blanks when unavailable."""
    iv = _nonempty(iv)
    if iv is None:
        return ("", "", "", "", "")
    return (
        _f(getattr(iv, "mean", None), 6),
        _f(getattr(iv, "lo", None), 6),
        _f(getattr(iv, "hi", None), 6),
        _f(getattr(iv, "std", None), 6),
        str(getattr(iv, "n", "")),
    )


def _md_table(header: Sequence[str], aligns: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(aligns) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def _scalar(value: Any) -> Any:
    """Make a cell value CSV-safe and deterministic (NaN → empty string)."""
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return value


# ═══════════════════════════════════════════════════════════════════════════════
# Per-dataset measurement
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DatasetRun:
    """Everything measured for one dataset, point estimates plus intervals."""

    dataset: str
    costs: CostConfig
    bundle: PipelineBundle
    evaluation: PolicyEvaluation
    decisions: List[DecisionResult]
    decision_indices: List[int]
    cost_reports: Dict[str, CostReport]
    repair_metrics: Dict[str, RepairMetrics]
    top1_correct: int
    top1_total: int
    repeated: Optional[RepeatedSplitResult] = None
    repeated_error: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    # ── derived ───────────────────────────────────────────────────────────────

    @property
    def proposed_cost(self) -> CostReport:
        return self.cost_reports[PROPOSED]

    @property
    def proposed_repair(self) -> RepairMetrics:
        return self.repair_metrics[PROPOSED]

    @property
    def n_false_negatives(self) -> int:
        return self.proposed_repair.n_false_negative

    @property
    def top1_accuracy(self) -> Optional[float]:
        if self.top1_total == 0:
            return None
        return self.top1_correct / self.top1_total

    @property
    def p_fallback(self) -> Optional[str]:
        return self.bundle.calibration_note

    @property
    def q_fallback(self) -> Optional[str]:
        report = self.bundle.detection_calibration
        if report is not None and not report.fitted:
            return (
                "detection calibrator could not fit; q = 1.0 for every record, "
                "which reproduces the pre-calibration policy. An assumption, "
                "not evidence (docs/cost_semantics.md §6)."
            )
        return None

    def fallback_marks(self) -> List[str]:
        marks = []
        if self.p_fallback:
            marks.append(f"**{self.dataset}** p-calibration fallback: {self.p_fallback}")
        if self.q_fallback:
            marks.append(f"**{self.dataset}** q-calibration fallback: {self.q_fallback}")
        return marks

    def baseline_costs(self) -> Dict[str, CostReport]:
        return {k: v for k, v in self.cost_reports.items() if k != PROPOSED}


def _constant_action_records(
    bundle: PipelineBundle, idx: Sequence[int], action: DecisionAction, name: str
) -> List[BaselineRecord]:
    """
    A degenerate baseline that takes the same action on every flagged cell.

    Phase 9 ships always-auto and the confidence threshold; "always human"
    and "always abstain" are the two remaining corners of the action space
    and T4 needs them as cost references. They see only
    ``UncertaintyRecord`` fields — no ground truth, no calibrated p.
    """
    return [
        BaselineRecord(
            method_name=name,
            dataset=bundle.unc_records[i].dataset,
            row_id=bundle.unc_records[i].row_id,
            column=bundle.unc_records[i].column,
            dirty_value=bundle.unc_records[i].dirty_value,
            top_candidate=bundle.unc_records[i].top_candidate,
            raw_confidence=bundle.unc_records[i].raw_confidence,
            action=action,
            metadata={
                "repair_method": bundle.unc_records[i].repair_method,
                "error_type": bundle.unc_records[i].error_type,
            },
        )
        for i in idx
    ]


def measure_dataset(
    dataset: str,
    *,
    n_records: int,
    seed: int,
    splits: int,
    iqr_k: float,
    flag_outliers: Optional[bool],
    base_split_seed: int,
    data_root: str,
    mode: str = "mock",
    llm_config: Optional[LLMConfig] = None,
    llm_client: Optional[object] = None,
) -> DatasetRun:
    """
    Run the pipeline once (the representative run) and the repeated-split
    aggregation, then score every policy against ground truth.

    ``mode`` selects the repairer (Phase 12). It applies to the
    **representative run only**: ``repeated_splits`` always builds a mock
    repairer, because ``src/experiments/aggregate.py`` does not thread a
    repair mode. T2 and the cost interval of T4 therefore describe the mock
    pipeline's calibration even in real mode, and
    :func:`_repair_mode_notes` says so on every table.
    """
    costs = CostConfig.from_yaml(COSTS_YAML, dataset)
    bundle = run_pipeline(
        dataset,
        n_records=n_records,
        seed=seed,
        split_seed=seed,
        iqr_k=iqr_k,
        flag_outliers=flag_outliers,
        data_root=data_root,
        mode=mode,
        llm_config=llm_config,
        llm_client=llm_client,
    )
    idx = bundle.decision_indices()
    all_decisions = decide_all(bundle, costs)
    decisions = [all_decisions[i] for i in idx]

    evaluation = evaluate_policies(
        bundle, costs, threshold=BASELINE_THRESHOLD, sweep=DEFAULT_THRESHOLDS
    )

    # Score the three Phase 9 / proposed methods plus the two degenerate
    # corners on exactly the same cells, with the same evaluators.
    records: Dict[str, List[BaselineRecord]] = dict(evaluation.records)
    records[ALWAYS_HUMAN] = _constant_action_records(
        bundle, idx, DecisionAction.HUMAN_REVIEW, ALWAYS_HUMAN
    )
    records[ALWAYS_ABSTAIN] = _constant_action_records(
        bundle, idx, DecisionAction.ABSTAIN, ALWAYS_ABSTAIN
    )

    cost_eval = CostEvaluator.from_cost_config(bundle.clean_df, costs)
    repair_eval = RepairEvaluator(bundle.clean_df, bundle.dirty_df, dataset)
    cost_reports: Dict[str, CostReport] = {}
    repair_metrics: Dict[str, RepairMetrics] = {}
    for name, recs in records.items():
        cost_reports[name] = cost_eval.evaluate(recs, dataset=dataset)
        repair_metrics[name] = repair_eval.evaluate(recs)

    # Top-1 repair accuracy is a property of the REPAIRER, not of any policy:
    # the share of decided cells whose top candidate equals the clean value,
    # regardless of the action taken. cal_samples[i].correct is that 0/1 label.
    top1_correct = int(sum(bundle.cal_samples[i].correct for i in idx)) if bundle.cal_samples else 0

    repeated: Optional[RepeatedSplitResult] = None
    repeated_error: Optional[str] = None
    try:
        repeated = repeated_splits(
            dataset,
            costs,
            n_splits=splits,
            seed=seed,
            base_split_seed=base_split_seed,
            iqr_k=iqr_k,
            flag_outliers=flag_outliers,
            n_records=n_records,
            data_root=data_root,
        )
    except Exception as exc:  # reported as "—", never silently dropped
        repeated_error = f"{type(exc).__name__}: {exc}"

    return DatasetRun(
        dataset=dataset,
        costs=costs,
        bundle=bundle,
        evaluation=evaluation,
        decisions=decisions,
        decision_indices=idx,
        cost_reports=cost_reports,
        repair_metrics=repair_metrics,
        top1_correct=top1_correct,
        top1_total=len(idx),
        repeated=repeated,
        repeated_error=repeated_error,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Shared note blocks
# ═══════════════════════════════════════════════════════════════════════════════


def _cost_units_note() -> str:
    return (
        f"All costs are abstract **{COST_UNITS}** from `configs/costs.yaml`. "
        "They are not money, not clinical harm, and not supplied by the REIN "
        "benchmark (docs/cost_semantics.md §1)."
    )


def _repair_mode(runs: Sequence[DatasetRun]) -> str:
    """The repair mode actually used, read back off the bundles."""
    modes = {getattr(r.bundle, "repair_mode", "mock") for r in runs}
    if len(modes) == 1:
        return modes.pop()
    return "mixed (" + ", ".join(sorted(modes)) + ")"


def _repair_mode_notes(runs: Sequence[DatasetRun]) -> List[str]:
    """
    The provenance block that goes on **every** table.

    In mock mode it is the sentence the tables have always carried. In real
    mode it names the model and prints the fallback count and rate per
    dataset, because a "real" run where most cells fell back to the mock
    repairer is a mock run with a label on it. That is a result, not a
    footnote, so it is stated wherever the numbers are.
    """
    mode = _repair_mode(runs)
    if mode == "mock":
        return [
            "**Repair mode: `mock`.** Repairs are produced by "
            "`src/repair/mock_repair.py` (difflib similarity + frequency "
            "weighting + median/quartile imputation + seeded noise). "
            "**No LLM was called.**",
        ]

    notes = ["**Repair mode: `real` (Phase 12, `src/repair/llm_repair.py`).**"]
    high = []
    for run in runs:
        stats = getattr(run.bundle, "llm_stats", None) or {}
        rate = float(stats.get("fallback_rate", 0.0) or 0.0)
        cells = int(stats.get("cells", 0) or 0)
        fallbacks = int(stats.get("fallbacks", 0) or 0)
        reasons = stats.get("fallback_reasons") or {}
        reason_str = (
            "; ".join(f"{k} {v}" for k, v in sorted(reasons.items())) if reasons else "none"
        )
        notes.append(
            f"`{run.dataset}`: model `{stats.get('model', NOT_COMPUTED)}` · "
            f"{cells:,} cells · {int(stats.get('successes', 0) or 0):,} answered by "
            f"the model · **{fallbacks:,} fell back to the mock repairer "
            f"({rate:.1%})** · reasons: {reason_str} · "
            f"{int(stats.get('cache_hits', 0) or 0):,} cache hits · "
            f"{int(stats.get('rate_limited', 0) or 0):,} rate limits · "
            f"{int(stats.get('key_rotations', 0) or 0):,} key rotations."
        )
        if rate >= HIGH_FALLBACK_RATE:
            high.append(f"{run.dataset} {rate:.1%}")
    if high:
        notes.append(
            f"⚠ **High fallback rate ({'; '.join(high)}).** Those cells were "
            f"repaired by `src/repair/mock_repair.py`, not by the model, so "
            f"this table is a blend and must not be read as an LLM result. "
            f"Per-cell provenance is in the `repair_method` column of "
            f"`predictions/<dataset>.csv` (`llm` vs `llm->mock (<reason>)`)."
        )
    notes.append(
        "**The intervals in T2 and the cost interval in T4 come from the "
        "mock repairer even in real mode**: `repeated_splits` "
        "(`src/experiments/aggregate.py`) builds its own `MockRepairer` and "
        "takes no repair mode, so those intervals describe the calibration "
        "of the mock pipeline. T1, T3 and the predictions CSV use the real "
        "repairs. Mixing them would be the dishonest reading."
    )
    return notes


def _predictions_mode_comment(bundle: PipelineBundle) -> str:
    """The repair-mode clause of a predictions file's header comment."""
    stats = getattr(bundle, "llm_stats", None) or {}
    if getattr(bundle, "repair_mode", "mock") != "real":
        return "mode=mock (src/repair/mock_repair.py — no LLM call)."
    return (
        f"mode=real (src/repair/llm_repair.py), model={stats.get('model', '?')}, "
        f"{int(stats.get('successes', 0) or 0):,} cells answered by the model and "
        f"{int(stats.get('fallbacks', 0) or 0):,} fell back to the mock repairer "
        f"({float(stats.get('fallback_rate', 0.0) or 0.0):.1%}); the "
        f"repair_method column says which, per cell."
    )


def _with_repair_mode(
    table: Tuple[List[str], List[List[Any]]], runs: Sequence[DatasetRun]
) -> Tuple[List[str], List[List[Any]]]:
    """
    Append the provenance columns to a table CSV.

    Every table CSV's first column is the dataset name (the md tables carry
    the same information as a note), so the three columns can be bolted on
    without touching each writer: ``repair_mode``, ``llm_model`` and
    ``llm_fallback_rate``. They are blank in mock mode, which keeps a mock
    run's CSV diffable against the historical ones apart from the literal
    ``mock`` label.
    """
    header, rows = table
    info: Dict[str, Tuple[Any, Any, Any]] = {}
    for run in runs:
        stats = getattr(run.bundle, "llm_stats", None) or {}
        mode = getattr(run.bundle, "repair_mode", "mock")
        info[run.dataset] = (
            mode,
            stats.get("model", "") if mode == "real" else "",
            _f(float(stats.get("fallback_rate", 0.0) or 0.0), 6) if mode == "real" else "",
        )
    out_header = list(header) + ["repair_mode", "llm_model", "llm_fallback_rate"]
    out_rows = [
        list(row) + list(info.get(str(row[0]), ("mock", "", ""))) for row in rows
    ]
    return out_header, out_rows


def _false_negative_note(runs: Sequence[DatasetRun]) -> str:
    parts = [f"{r.dataset}: {r.n_false_negatives:,}" for r in runs]
    return (
        "**False negatives are priced at zero.** Genuine corruptions the "
        "detector never flagged (" + "; ".join(parts) + ") enter no arm of the "
        "objective, so every cost number in these tables is a statement about "
        "the flagged cells only (docs/cost_semantics.md §3)."
    )


def _adult_framing_note(runs: Sequence[DatasetRun]) -> Optional[str]:
    """The mandatory Adult framing, with its measured cost stated beside it."""
    adult = next((r for r in runs if r.dataset == "adult"), None)
    if adult is None:
        return None
    total = adult.proposed_cost.total_realized_cost
    return (
        f"**Adult framing (docs/cost_semantics.md §3).** Adult's realised cost "
        f"measures only residual corruption on the cells the detector flagged; "
        f"the measured value here is {total:,.2f}. A value of 0 must be read as "
        f"*zero residual-corruption cost on the flagged cells*, never as "
        f"\"zero cost\" or \"the data is now clean\": "
        f"{adult.n_false_negatives:,} genuinely corrupted cells (income-label "
        f"flips) were never detected, no action was ever taken on them, and "
        f"they enter no arm of the objective. The defensible claim is that the "
        f"calibrated policy declines to act on a detector whose flags are "
        f"false positives, not that Adult was repaired."
    )


def _fallback_notes(runs: Sequence[DatasetRun]) -> List[str]:
    marks: List[str] = []
    for run in runs:
        marks.extend(run.fallback_marks())
    return marks


def _protocol_single_run(seed: int, n_records: int) -> str:
    cap = "no cap (full dataset)" if n_records <= 0 else f"capped at {n_records:,} detections"
    return (
        f"**Protocol:** one representative pipeline run, `seed = {seed}`, "
        f"`split_seed = {seed}`, detections {cap}. These are point estimates "
        f"from a single calibration split, not interval estimates."
    )


def _protocol_repeated(runs: Sequence[DatasetRun], splits: int, seed: int) -> str:
    seeds = ""
    for run in runs:
        if run.repeated is not None and getattr(run.repeated, "split_seeds", None):
            ss = list(run.repeated.split_seeds)
            seeds = f" split seeds {ss[0]}–{ss[-1]}" if len(ss) > 1 else f" split seed {ss[0]}"
            break
    return (
        f"**Protocol:** `src.experiments.aggregate.repeated_splits`, "
        f"{splits} repeated calibration splits with `seed = {seed}` held fixed "
        f"and only `split_seed` varying,{seeds}. Intervals are 95 % bootstrap "
        f"intervals over splits, written `mean [lo, hi]`. Varying `split_seed` "
        f"changes the calibrators (and therefore p and q) but not the "
        f"detections or the mock repairs, so the interval is over calibration "
        f"splits and nothing else (docs/phase13_plan.md §0)."
    )


def _notes_block(notes: Sequence[str]) -> List[str]:
    lines = ["**Notes**", ""]
    lines += [f"- {n}" for n in notes if n]
    lines.append("")
    return lines


def _table_document(
    title: str, what: str, protocol: str, body: Sequence[str], notes: Sequence[str], quick: bool
) -> str:
    lines: List[str] = [f"# {title}", ""]
    if quick:
        lines += [QUICK_BANNER, ""]
    lines += [what, "", protocol, ""]
    lines += list(body)
    lines.append("")
    lines += _notes_block(notes)
    return "\n".join(lines).rstrip() + "\n"


# ═══════════════════════════════════════════════════════════════════════════════
# T1 — repair quality
# ═══════════════════════════════════════════════════════════════════════════════

_T1_HEADER = (
    "dataset", "detections (all)", "decided cells", "precision", "recall", "F1",
    "top-1 repair accuracy", "genuine corruptions (full dataset)",
    "detected genuine (TP)", "false negatives",
)
_T1_ALIGN = ("---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:")


def t1_rows(runs: Sequence[DatasetRun]) -> List[List[str]]:
    rows = []
    for run in runs:
        m = run.proposed_repair
        rows.append([
            run.dataset,
            _i(run.bundle.n_detected_total),
            _i(len(run.decision_indices)),
            _f(m.detection_precision),
            _f(m.detection_recall),
            _f(m.detection_f1),
            _f(run.top1_accuracy),
            _i(m.n_total_corrupted),
            _i(m.n_truly_corrupted),
            _i(m.n_false_negative),
        ])
    return rows


def t1_document(runs: Sequence[DatasetRun], seed: int, n_records: int, quick: bool) -> str:
    notes = [
        "`detections (all)` is every cell the Phase 3 detector flagged; "
        "`decided cells` is what the policy actually acted on (ID-like "
        "columns in `SKIP_COLUMNS` are excluded, as in every other table).",
        "Precision / recall / F1 are **detection** metrics over the decided "
        "cells against ground-truth corruption. Recall is low wherever the "
        "corruption is undetectable from dirty data alone.",
        "`top-1 repair accuracy` is a property of the repairer, not of a "
        "policy: the share of decided cells whose top candidate equals the "
        "clean value, counted regardless of the action taken.",
        "`genuine corruptions (full dataset)` counts every cell where dirty ≠ "
        "clean anywhere in the dataset, including ID-like columns, so "
        "`false negatives = genuine corruptions − detected genuine`.",
        _false_negative_note(runs),
    ]
    notes += _repair_mode_notes(runs)
    notes += _fallback_notes(runs)
    return _table_document(
        "T1 — Repair quality",
        "Detection quality and raw repair quality per dataset, before any "
        "cost-sensitive decision is taken.",
        _protocol_single_run(seed, n_records),
        _md_table(_T1_HEADER, _T1_ALIGN, t1_rows(runs)),
        notes,
        quick,
    )


def t1_csv(runs: Sequence[DatasetRun]) -> Tuple[List[str], List[List[Any]]]:
    header = [
        "dataset", "n_detections_total", "n_decided_cells", "detection_precision",
        "detection_recall", "detection_f1", "top1_repair_accuracy",
        "n_genuine_corruptions_full_dataset", "n_detected_genuine", "n_false_negatives",
    ]
    rows = []
    for run in runs:
        m = run.proposed_repair
        acc = run.top1_accuracy
        rows.append([
            run.dataset, run.bundle.n_detected_total, len(run.decision_indices),
            _f(m.detection_precision, 6), _f(m.detection_recall, 6), _f(m.detection_f1, 6),
            "" if acc is None else _f(acc, 6),
            m.n_total_corrupted, m.n_truly_corrupted, m.n_false_negative,
        ])
    return header, rows


# ═══════════════════════════════════════════════════════════════════════════════
# T2 — calibration
# ═══════════════════════════════════════════════════════════════════════════════

_T2_HEADER = ("dataset", "method", "Brier  mean [95 % CI]", "ECE  mean [95 % CI]",
              "splits fitted", "status")
_T2_ALIGN = ("---", "---", "---:", "---:", "---:", "---")


def _t2_status(run: DatasetRun, method: str, n_fitted: Optional[int], splits: int) -> str:
    if run.repeated is None:
        return f"not computed — repeated_splits failed ({run.repeated_error})"
    if method == "raw":
        return "uncalibrated reference"
    if n_fitted is None:
        return "not reported by repeated_splits"
    if n_fitted == 0:
        return "⚠ never fitted — fallback in every split"
    if n_fitted < splits:
        return f"⚠ fitted in {n_fitted} of {splits} splits; the rest fell back"
    return "fitted in every split"


def t2_rows(runs: Sequence[DatasetRun], splits: int) -> List[List[str]]:
    rows = []
    for run in runs:
        rs = run.repeated
        for method in CALIBRATION_METHODS:
            brier = _lookup(getattr(rs, "brier", None), method) if rs else None
            ece = _lookup(getattr(rs, "ece", None), method) if rs else None
            n_fitted = _lookup(getattr(rs, "n_fitted", None), method) if rs else None
            if method != "raw" and n_fitted == 0:
                brier = ece = None  # a fallback's metrics are not that method's
            rows.append([
                run.dataset, method,
                _interval_str(brier), _interval_str(ece),
                NOT_COMPUTED if n_fitted is None else f"{n_fitted}/{splits}",
                _t2_status(run, method, n_fitted, splits),
            ])
    return rows


def t2_document(runs: Sequence[DatasetRun], splits: int, seed: int, quick: bool) -> str:
    notes = [
        "Brier and ECE are computed on each split's held-out evaluation set; "
        "**lower is better** for both. The interval is over splits, so it "
        "answers \"would a different split have changed the ranking?\" — not "
        "\"how uncertain is the metric on this split?\".",
        "`splits fitted` is the contract's `n_fitted`: how many splits that "
        "calibrator actually fitted. A method that never fitted shows `—` for "
        "its metrics rather than the raw metrics it copies internally, because "
        "printing those reads as a perfect score (CLAUDE.md §5.4).",
        "Two differences whose intervals overlap are **not** separated by this "
        "experiment, however far apart the means are.",
    ]
    for run in runs:
        if run.repeated is None:
            notes.append(
                f"⚠ **{run.dataset}**: every cell is `—` because "
                f"`repeated_splits` raised — {run.repeated_error}."
            )
    notes += _repair_mode_notes(runs)
    notes += _fallback_notes(runs)
    return _table_document(
        "T2 — Calibration: raw vs logistic vs isotonic vs multivariate",
        "How trustworthy the probability of repair correctness (p) is, with "
        "confidence intervals over repeated calibration splits.",
        _protocol_repeated(runs, splits, seed),
        _md_table(_T2_HEADER, _T2_ALIGN, t2_rows(runs, splits)),
        notes,
        quick,
    )


def t2_csv(runs: Sequence[DatasetRun], splits: int) -> Tuple[List[str], List[List[Any]]]:
    header = [
        "dataset", "method",
        "brier_mean", "brier_lo", "brier_hi", "brier_std", "brier_n",
        "ece_mean", "ece_lo", "ece_hi", "ece_std", "ece_n",
        "n_fitted", "n_splits",
    ]
    rows = []
    for run in runs:
        rs = run.repeated
        for method in CALIBRATION_METHODS:
            brier = _lookup(getattr(rs, "brier", None), method) if rs else None
            ece = _lookup(getattr(rs, "ece", None), method) if rs else None
            n_fitted = _lookup(getattr(rs, "n_fitted", None), method) if rs else None
            if method != "raw" and n_fitted == 0:
                brier = ece = None
            rows.append(
                [run.dataset, method]
                + list(_csv_interval_fields(brier))
                + list(_csv_interval_fields(ece))
                + ["" if n_fitted is None else n_fitted, splits]
            )
    return header, rows


# ═══════════════════════════════════════════════════════════════════════════════
# T3 — risk-aware performance
# ═══════════════════════════════════════════════════════════════════════════════

_T3_HEADER = (
    "dataset", "AUTO share", "HUMAN share", "ABSTAIN share", "realised cost",
    "mean cost / cell", "wrong-auto rate", "abstain on corrupted",
    "abstain on clean", "p\\*", "q\\*",
)
_T3_ALIGN = ("---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:")


def _share_cell(point: float, iv: Optional[Interval]) -> str:
    """Point estimate, plus the split interval when one is available."""
    if iv is None:
        return _f(point)
    return f"{_f(point)}  ({_interval_str(iv)})"


def t3_rows(runs: Sequence[DatasetRun]) -> List[List[str]]:
    rows = []
    for run in runs:
        cr = run.proposed_cost
        rs = run.repeated
        actions = getattr(rs, "actions", None) if rs else None
        cost_iv = _lookup(getattr(rs, "cost", None), PROPOSED, "proposed") if rs else None
        c = run.costs
        rows.append([
            run.dataset,
            _share_cell(cr.auto_rate, _action_interval(actions, PROPOSED, DecisionAction.AUTO_REPAIR)),
            _share_cell(cr.human_rate, _action_interval(actions, PROPOSED, DecisionAction.HUMAN_REVIEW)),
            _share_cell(cr.abstain_rate, _action_interval(actions, PROPOSED, DecisionAction.ABSTAIN)),
            f"{cr.total_realized_cost:,.2f}"
            + ("" if cost_iv is None else f"  ({_interval_str(cost_iv, 2)})"),
            _f(cr.mean_cost_per_record),
            _f(cr.wrong_auto_rate),
            f"{cr.n_abstain_on_corrupted:,} / {cr.n_abstain:,}",
            f"{cr.n_abstain_on_clean:,} / {cr.n_abstain:,}",
            _f(1.0 - c.c_human_review / c.c_wrong_repair, 3),
            _f(c.c_human_review / c.c_leave_error, 3) if c.c_leave_error else NOT_COMPUTED,
        ])
    return rows


def t3_document(runs: Sequence[DatasetRun], seed: int, n_records: int, quick: bool) -> str:
    notes = [
        _cost_units_note(),
        "Counts and rates are from the single representative run; where the "
        "same quantity also has an interval over repeated splits it is printed "
        "in parentheses as `mean [lo, hi]`.",
        "`realised cost` charges `C_wrong` for an applied repair that is "
        "wrong, `C_human` per cell routed to a reviewer (**including false "
        "alarms**), and `C_abstain` only when an abstained cell was genuinely "
        "corrupted — interpretation (a) in docs/cost_semantics.md §2, the one "
        "under which `E[ABSTAIN] = q·C_abstain` is an unbiased estimate of "
        "what the evaluator charges.",
        "`abstain on corrupted` is the abstention mistake the cost model "
        "charges for; `abstain on clean` is the false alarm it correctly "
        "charges nothing for.",
        "`p* = 1 − C_human/C_wrong` and `q* = C_human/C_abstain` are the "
        "derived decision boundaries (docs/cost_semantics.md §4). `q*` is "
        "horizontal, so for a single cell at fixed q only two of the three "
        "arms are ever live; all three occur across a population that mixes "
        "real errors with false alarms.",
        _false_negative_note(runs),
    ]
    adult = _adult_framing_note(runs)
    if adult:
        notes.append(adult)
    notes += _repair_mode_notes(runs)
    notes += _fallback_notes(runs)
    return _table_document(
        "T3 — Risk-aware performance of the proposed policy",
        "What the calibrated, cost-sensitive policy actually does with the "
        "flagged cells, and what that costs against ground truth.",
        _protocol_single_run(seed, n_records),
        _md_table(_T3_HEADER, _T3_ALIGN, t3_rows(runs)),
        notes,
        quick,
    )


def t3_csv(runs: Sequence[DatasetRun]) -> Tuple[List[str], List[List[Any]]]:
    header = [
        "dataset", "n_decided", "n_auto", "n_human", "n_abstain",
        "auto_rate", "human_rate", "abstain_rate",
        "auto_rate_mean", "auto_rate_lo", "auto_rate_hi",
        "human_rate_mean", "human_rate_lo", "human_rate_hi",
        "abstain_rate_mean", "abstain_rate_lo", "abstain_rate_hi",
        "total_realized_cost", "mean_cost_per_record",
        "cost_mean", "cost_lo", "cost_hi", "cost_std", "cost_n",
        "n_wrong_auto", "wrong_auto_rate",
        "n_abstain_on_corrupted", "n_abstain_on_clean",
        "c_wrong", "c_human", "c_abstain", "p_star", "q_star",
    ]
    rows = []
    for run in runs:
        cr = run.proposed_cost
        rs = run.repeated
        actions = getattr(rs, "actions", None) if rs else None
        cost_iv = _lookup(getattr(rs, "cost", None), PROPOSED, "proposed") if rs else None
        c = run.costs
        auto_iv = _action_interval(actions, PROPOSED, DecisionAction.AUTO_REPAIR)
        human_iv = _action_interval(actions, PROPOSED, DecisionAction.HUMAN_REVIEW)
        abstain_iv = _action_interval(actions, PROPOSED, DecisionAction.ABSTAIN)
        rows.append([
            run.dataset, cr.n_records, cr.n_auto, cr.n_human, cr.n_abstain,
            _f(cr.auto_rate, 6), _f(cr.human_rate, 6), _f(cr.abstain_rate, 6),
            *_csv_interval_fields(auto_iv)[:3],
            *_csv_interval_fields(human_iv)[:3],
            *_csv_interval_fields(abstain_iv)[:3],
            _f(cr.total_realized_cost, 6), _f(cr.mean_cost_per_record, 6),
            *_csv_interval_fields(cost_iv),
            cr.n_wrong_auto, _f(cr.wrong_auto_rate, 6),
            cr.n_abstain_on_corrupted, cr.n_abstain_on_clean,
            _f(c.c_wrong_repair, 4), _f(c.c_human_review, 4), _f(c.c_leave_error, 4),
            _f(1.0 - c.c_human_review / c.c_wrong_repair, 6),
            _f(c.c_human_review / c.c_leave_error, 6) if c.c_leave_error else "",
        ])
    return header, rows


# ═══════════════════════════════════════════════════════════════════════════════
# T4 — policy comparison
# ═══════════════════════════════════════════════════════════════════════════════

_T4_HEADER = (
    "dataset", "policy", "realised cost", "mean / cell",
    "cost over splits  mean [95 % CI]", "AUTO", "HUMAN", "ABSTAIN", "wrong-auto rate",
)
_T4_ALIGN = ("---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:")

_POLICY_ORDER = (PROPOSED, "always_auto", ALWAYS_HUMAN, ALWAYS_ABSTAIN)


def _t4_policy_reports(run: DatasetRun) -> List[Tuple[str, CostReport]]:
    """Policies in a fixed, deterministic order: corners, then the sweep."""
    out: List[Tuple[str, CostReport]] = []
    for name in _POLICY_ORDER:
        if name in run.cost_reports:
            out.append((name, run.cost_reports[name]))
    for t in sorted(run.evaluation.threshold_sweep):
        out.append((f"confidence_threshold_{t:.2f}", run.evaluation.threshold_sweep[t]))
    return out


def t4_rows(runs: Sequence[DatasetRun]) -> List[List[str]]:
    rows = []
    for run in runs:
        rs = run.repeated
        for name, cr in _t4_policy_reports(run):
            iv = _lookup(getattr(rs, "cost", None), name) if rs else None
            if name == PROPOSED and iv is None and rs is not None:
                iv = _lookup(getattr(rs, "cost", None), "proposed")
            rows.append([
                run.dataset, name,
                f"{cr.total_realized_cost:,.2f}", _f(cr.mean_cost_per_record),
                _interval_str(iv, 2), _i(cr.n_auto), _i(cr.n_human), _i(cr.n_abstain),
                _f(cr.wrong_auto_rate),
            ])
    return rows


def _t4_winner_notes(runs: Sequence[DatasetRun]) -> List[str]:
    notes = []
    for run in runs:
        proposed = run.proposed_cost.total_realized_cost
        others = {
            name: cr.total_realized_cost
            for name, cr in _t4_policy_reports(run)
            if name != PROPOSED
        }
        if not others:
            continue
        best_name = min(others, key=lambda k: (others[k], k))
        best = others[best_name]
        if proposed < best:
            verdict = f"proposed wins by {best - proposed:,.2f}"
        elif proposed > best:
            verdict = (
                f"⚠ **proposed LOSES** to `{best_name}` by {proposed - best:,.2f}"
            )
        else:
            verdict = f"tie with `{best_name}`"
        notes.append(
            f"**{run.dataset}**: proposed {proposed:,.2f} vs best alternative "
            f"`{best_name}` {best:,.2f} — {verdict}."
        )
    return notes


def t4_document(runs: Sequence[DatasetRun], splits: int, seed: int, n_records: int, quick: bool) -> str:
    notes = [
        _cost_units_note(),
        "`realised cost`, `mean / cell` and the action counts are from the "
        "single representative run described above; the interval column is the "
        f"same policy's cost over {splits} repeated calibration splits, from "
        "`repeated_splits`, and is blank (`—`) for any policy that aggregation "
        "does not report.",
        "`always_human` and `always_abstain` are the two remaining corners of "
        "the action space, added here as cost references. They are not Phase 9 "
        "baselines and they never abstain *selectively*, so they are context "
        "for the cost scale rather than competitive methods.",
        "The threshold baseline acts on **raw** confidence, which is what the "
        "calibration in T2 is meant to improve on; comparing it against the "
        "proposed row is the whole claim of the project.",
        "A lower realised cost here does **not** mean better repairs — T1 "
        "holds the repairs fixed across every row. Only the actions differ.",
        _false_negative_note(runs),
    ]
    notes += _t4_winner_notes(runs)
    adult = _adult_framing_note(runs)
    if adult:
        notes.append(adult)
    notes += _repair_mode_notes(runs)
    notes += _fallback_notes(runs)
    return _table_document(
        "T4 — Policy comparison",
        "The proposed calibrated risk-aware policy against always-auto, "
        "always-human, always-abstain and the raw-confidence threshold sweep, "
        "all scored on the same cells with the same cost config.",
        _protocol_single_run(seed, n_records)
        + "  \n"
        + _protocol_repeated(runs, splits, seed),
        _md_table(_T4_HEADER, _T4_ALIGN, t4_rows(runs)),
        notes,
        quick,
    )


def t4_csv(runs: Sequence[DatasetRun], splits: int) -> Tuple[List[str], List[List[Any]]]:
    header = [
        "dataset", "policy", "total_realized_cost", "mean_cost_per_record",
        "cost_mean", "cost_lo", "cost_hi", "cost_std", "cost_n", "n_splits",
        "n_auto", "n_human", "n_abstain", "n_wrong_auto", "wrong_auto_rate",
    ]
    rows = []
    for run in runs:
        rs = run.repeated
        for name, cr in _t4_policy_reports(run):
            iv = _lookup(getattr(rs, "cost", None), name) if rs else None
            if name == PROPOSED and iv is None and rs is not None:
                iv = _lookup(getattr(rs, "cost", None), "proposed")
            rows.append([
                run.dataset, name, _f(cr.total_realized_cost, 6),
                _f(cr.mean_cost_per_record, 6),
                *_csv_interval_fields(iv), splits,
                cr.n_auto, cr.n_human, cr.n_abstain, cr.n_wrong_auto,
                _f(cr.wrong_auto_rate, 6),
            ])
    return header, rows


# ═══════════════════════════════════════════════════════════════════════════════
# Writers
# ═══════════════════════════════════════════════════════════════════════════════


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]],
               comments: Sequence[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        for line in comments:
            fh.write(f"# {line}\n")
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(list(header))
        for row in rows:
            writer.writerow([_scalar(c) for c in row])


def write_predictions(run: DatasetRun, path: Path) -> int:
    """
    One row per decided cell. Decision inputs first; the two ground-truth
    columns last, prefixed ``eval_`` and named in the header comment.
    """
    bundle = run.bundle
    rows: List[List[Any]] = []
    for position, i in enumerate(run.decision_indices):
        rec = bundle.unc_records[i]
        det = bundle.detections[i]
        dec = run.decisions[position]
        rows.append([
            bundle.dataset,
            rec.row_id,
            rec.column,
            det.detection_method,
            rec.error_type,
            rec.dirty_value,
            rec.repair_method,
            rec.top_candidate,
            _f(rec.raw_confidence, 6),
            _f(rec.self_consistency, 6),
            _f(rec.normalized_entropy, 6),
            _f(bundle.calibrated_probs[i], 6),
            _f(bundle.corruption_probs[i], 6) if bundle.corruption_probs else _f(1.0, 6),
            _f(dec.expected_cost_auto, 6),
            _f(dec.expected_cost_human, 6),
            _f(dec.expected_cost_abstain, 6),
            dec.action.value,
            int(bundle.cal_samples[i].correct) if bundle.cal_samples else "",
            int(bundle.detection_labels[i]) if bundle.detection_labels else "",
        ])
    comments = [
        f"{bundle.dataset}: one row per DECIDED cell "
        f"({len(rows)} rows; ID-like columns in SKIP_COLUMNS excluded).",
        f"Generated by run_experiment.py (Phase 13) with seed={bundle.seed}, "
        f"split_seed={getattr(bundle, 'split_seed', bundle.seed)}, "
        f"iqr_k={bundle.iqr_k:g}, flag_outliers={bundle.flag_outliers}, "
        f"{_predictions_mode_comment(bundle)}",
        "EVALUATION-ONLY COLUMNS (ground truth, read only AFTER every decision "
        "existed, never an input to detection, repair, uncertainty or the "
        "policy), grouped at the end of the header: "
        + ", ".join(PREDICTION_EVAL_ONLY_COLUMNS)
        + ".",
        "Every other column is a decision input or a decision output. "
        "p_repair_correct = P(top candidate is correct); q_cell_corrupted = "
        "P(cell is genuinely corrupted); expected costs are in abstract "
        "experimental penalty units.",
    ]
    _write_csv(path, PREDICTION_COLUMNS, rows, comments)
    return len(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════════


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_figures(run: DatasetRun, figures_dir: Path) -> Tuple[List[str], List[str]]:
    """
    One set of figures per dataset, named ``<dataset>_<kind>.png``.

    Returns (written filenames, skipped notes).
    """
    written: List[str] = []
    skipped: List[str] = []
    ds = run.dataset
    bundle = run.bundle
    c = run.costs
    rows = [r for r in run.evaluation.rows]

    def _name(kind: str) -> Path:
        return figures_dir / f"{ds}_{kind}.png"

    if bundle.comparison is not None:
        _save(plots.reliability_figure(bundle.comparison), _name("reliability"))
        written.append(_name("reliability").name)
    else:
        skipped.append(f"{ds}_reliability.png: no calibration comparison (no samples)")

    _save(
        plots.confidence_histogram(
            bundle.raw_confidences, bundle.calibrated_probs, bundle.calibration_method
        ),
        _name("confidence"),
    )
    written.append(_name("confidence").name)

    if rows:
        _save(plots.cost_comparison(rows), _name("cost_comparison"))
        written.append(_name("cost_comparison").name)
        _save(plots.action_distribution(rows), _name("action_distribution"))
        written.append(_name("action_distribution").name)
    else:
        skipped.append(f"{ds}_cost_comparison.png / {ds}_action_distribution.png: "
                       "no evaluated policies")

    _save(
        plots.threshold_sweep(
            run.evaluation.threshold_sweep,
            proposed_cost=run.proposed_cost.total_realized_cost,
            current_threshold=BASELINE_THRESHOLD,
        ),
        _name("threshold_sweep"),
    )
    written.append(_name("threshold_sweep").name)

    points = [
        (
            bundle.calibrated_probs[i],
            bundle.corruption_probs[i] if bundle.corruption_probs else 1.0,
            run.decisions[position].action,
        )
        for position, i in enumerate(run.decision_indices)
    ]
    _save(
        plots.decision_map(
            c_wrong=c.c_wrong_repair,
            c_human=c.c_human_review,
            c_abstain=c.c_leave_error,
            points=points,
        ),
        _name("decision_map"),
    )
    written.append(_name("decision_map").name)
    return written, skipped


# ═══════════════════════════════════════════════════════════════════════════════
# Metadata
# ═══════════════════════════════════════════════════════════════════════════════


def git_commit() -> Optional[str]:
    """``git rev-parse HEAD``, or ``None`` when git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.stdout.strip()
    return commit if out.returncode == 0 and commit else None


def _metadata_llm_model(runs: Sequence[DatasetRun]) -> Optional[str]:
    """The model id that produced the repairs, or None in mock mode."""
    models = {
        str((getattr(r.bundle, "llm_stats", None) or {}).get("model"))
        for r in runs
        if getattr(r.bundle, "repair_mode", "mock") == "real"
    }
    if not models:
        return None
    return models.pop() if len(models) == 1 else ", ".join(sorted(models))


def _metadata_repairer(runs: Sequence[DatasetRun]) -> str:
    """One line naming the repairer that produced the representative run."""
    if _repair_mode(runs) == "mock":
        return "src/repair/mock_repair.py (mock; no LLM call)"
    return (
        f"src/repair/llm_repair.py (real; model={_metadata_llm_model(runs)}; "
        f"per-cell fallback to src/repair/mock_repair.py)"
    )


def _package_version(name: str) -> Optional[str]:
    try:
        module = __import__(name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "")) or None


def build_metadata(
    runs: Sequence[DatasetRun],
    args: argparse.Namespace,
    *,
    n_records: int,
    splits: int,
    predictions: Dict[str, int],
    figures: Dict[str, List[str]],
    figure_skips: List[str],
    wall_seconds: float,
) -> Dict[str, Any]:
    """
    Two sub-objects on purpose. Everything that is a function of the flags
    lives under ``deterministic`` so two runs of the same command can be
    diffed byte for byte; the clock, the elapsed time and the invocation
    paths live under ``non_deterministic``.
    """
    per_dataset: Dict[str, Any] = {}
    for run in runs:
        b = run.bundle
        m = run.proposed_repair
        rs = run.repeated
        per_dataset[run.dataset] = {
            "n_rows_dirty": int(len(b.dirty_df)),
            "n_columns_dirty": int(b.dirty_df.shape[1]),
            "n_detections_total": int(b.n_detected_total),
            "n_records_carried": int(b.n_records),
            "n_decided_cells": int(len(run.decision_indices)),
            "n_prediction_rows": int(predictions.get(run.dataset, 0)),
            "n_genuine_corruptions_full_dataset": int(m.n_total_corrupted),
            "n_detected_genuine": int(m.n_truly_corrupted),
            "n_false_negatives": int(m.n_false_negative),
            "top1_repair_correct": int(run.top1_correct),
            "iqr_k": float(b.iqr_k),
            "flag_outliers_resolved": bool(b.flag_outliers),
            "repair_mode": getattr(b, "repair_mode", "mock"),
            "llm_stats": getattr(b, "llm_stats", None),
            "p_calibration_method": b.calibration_method,
            "p_calibration_fallback": bool(b.calibration_note),
            "q_calibrator_fitted": bool(
                b.detection_calibration.fitted if b.detection_calibration else False
            ),
            "costs": {
                "c_wrong_repair": float(run.costs.c_wrong_repair),
                "c_human_review": float(run.costs.c_human_review),
                "c_leave_error": float(run.costs.c_leave_error),
                "units": COST_UNITS,
            },
            "repeated_splits": {
                "available": rs is not None,
                "error": run.repeated_error,
                "n_splits": int(getattr(rs, "n_splits", 0)) if rs else 0,
                "split_seeds": [int(s) for s in getattr(rs, "split_seeds", [])] if rs else [],
                "n_fitted": {
                    k: int(v) for k, v in (getattr(rs, "n_fitted", {}) or {}).items()
                } if rs else {},
            },
            "figures": figures.get(run.dataset, []),
        }

    return {
        "deterministic": {
            "script": "run_experiment.py",
            "phase": 13,
            "mode": args.mode,
            # Read back off the bundles rather than off the flags: in real mode
            # the two can only agree, but the bundle is what actually ran.
            "repair_mode": _repair_mode(runs),
            "llm_model": _metadata_llm_model(runs),
            "llm_cache": bool(getattr(args, "llm_cache", True)),
            "llm_stats": {
                r.dataset: (getattr(r.bundle, "llm_stats", None)) for r in runs
            },
            "datasets": [r.dataset for r in runs],
            "seed": int(args.seed),
            "split_seed_of_representative_run": int(args.seed),
            "base_split_seed": int(args.base_split_seed),
            "n_splits": int(splits),
            "n_records": int(n_records),
            "iqr_k": float(args.iqr_k),
            "flag_outliers": args.flag_outliers,
            "quick": bool(args.quick),
            "paper_grade": not bool(args.quick),
            "figures_written": not bool(args.no_figures),
            "figure_skips": figure_skips,
            "baseline_threshold": BASELINE_THRESHOLD,
            "threshold_sweep": [float(t) for t in DEFAULT_THRESHOLDS],
            "costs_config": str(COSTS_YAML).replace("\\", "/"),
            "data_root": str(args.data_root).replace("\\", "/"),
            "repairer": _metadata_repairer(runs),
            "interval_repairer": (
                "src/repair/mock_repair.py (repeated_splits always builds a "
                "MockRepairer; T2 and T4's cost interval are mock-based even "
                "in real mode)"
            ),
            "git_commit": git_commit(),
            "versions": {
                "python": platform.python_version(),
                "pandas": _package_version("pandas"),
                "numpy": _package_version("numpy"),
                "matplotlib": _package_version("matplotlib"),
                "sklearn": _package_version("sklearn"),
                "platform": platform.platform(),
            },
            "per_dataset": per_dataset,
        },
        "non_deterministic": {
            "_why": (
                "The clock, the elapsed time and the invocation paths differ "
                "between two runs of the same command by design. Everything "
                "under 'deterministic' must be byte-identical."
            ),
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
            "wall_time_seconds": round(wall_seconds, 3),
            "output_dir": str(Path(args.output_dir).as_posix()),
            "argv": sys.argv[1:],
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# stdout summary
# ═══════════════════════════════════════════════════════════════════════════════


def _rule(char: str = "═", width: int = 78) -> str:
    return char * width


def honest_reading(runs: Sequence[DatasetRun], splits: int, quick: bool) -> List[str]:
    """
    The closing read, written from the measured numbers only. Nothing here
    is pre-judged: each bullet is a comparison of values computed above.
    """
    lines = ["Honest reading", _rule("─")]

    for run in runs:
        proposed = run.proposed_cost.total_realized_cost
        others = {
            name: cr.total_realized_cost
            for name, cr in _t4_policy_reports(run)
            if name != PROPOSED
        }
        if others:
            best_name = min(others, key=lambda k: (others[k], k))
            best = others[best_name]
            if proposed < best:
                verdict = f"proposed wins by {best - proposed:,.2f}"
            elif proposed > best:
                verdict = f"PROPOSED LOSES by {proposed - best:,.2f}"
            else:
                verdict = "tie"
            lines.append(
                f"- {run.dataset}: proposed {proposed:,.2f} vs best alternative "
                f"{best_name} {best:,.2f} — {verdict} ({COST_UNITS})."
            )

    # Calibration: does any fitted method beat raw, and do the intervals separate?
    for run in runs:
        rs = run.repeated
        raw = _lookup(getattr(rs, "brier", None), "raw") if rs else None
        if raw is None:
            lines.append(
                f"- {run.dataset}: no calibration interval available "
                f"({run.repeated_error or 'raw Brier not reported'}); printed as '—'."
            )
            continue
        best_name, best_iv = None, None
        for method in CALIBRATION_METHODS:
            if method == "raw":
                continue
            if (_lookup(getattr(rs, "n_fitted", None), method) or 0) == 0:
                continue
            iv = _lookup(getattr(rs, "brier", None), method)
            if iv is None:
                continue
            if best_iv is None or iv.mean < best_iv.mean:
                best_name, best_iv = method, iv
        if best_iv is None:
            lines.append(
                f"- {run.dataset}: no calibrator fitted in any of {splits} splits, "
                f"so calibration cannot be claimed to help here; raw Brier "
                f"{_interval_str(raw)}."
            )
            continue
        separated = best_iv.hi < raw.lo or raw.hi < best_iv.lo
        direction = "below" if best_iv.mean < raw.mean else "above"
        lines.append(
            f"- {run.dataset}: best calibrator {best_name} Brier "
            f"{_interval_str(best_iv)} vs raw {_interval_str(raw)} — mean is "
            f"{direction} raw, and the 95 % intervals "
            f"{'do NOT overlap (separated)' if separated else 'OVERLAP, so this run does not separate them'}."
        )

    fn = "; ".join(f"{r.dataset} {r.n_false_negatives:,}" for r in runs)
    lines.append(
        f"- Scope limit: undetected genuine corruptions ({fn}) are priced at "
        f"zero by the cost model, so every cost above describes the flagged "
        f"cells only (docs/cost_semantics.md §3)."
    )
    adult = next((r for r in runs if r.dataset == "adult"), None)
    if adult is not None:
        lines.append(
            f"- Adult's realised cost of {adult.proposed_cost.total_realized_cost:,.2f} "
            f"is zero residual-corruption cost on the flagged cells, not a clean "
            f"dataset: {adult.n_false_negatives:,} corruptions were never detected."
        )
    for mark in _fallback_notes(runs):
        lines.append(f"- Fallback: {mark}")
    if _repair_mode(runs) == "mock":
        lines.append(
            "- Every repair above came from the mock repairer, not an LLM "
            "(--mode mock), so these numbers bound the decision layer, not "
            "an LLM's repair quality."
        )
    else:
        for run in runs:
            stats = getattr(run.bundle, "llm_stats", None) or {}
            rate = float(stats.get("fallback_rate", 0.0) or 0.0)
            verdict = (
                "most cells are real LLM repairs"
                if rate < HIGH_FALLBACK_RATE
                else "a large share of these 'LLM' repairs are mock repairs"
            )
            lines.append(
                f"- {run.dataset}: real mode, model "
                f"{stats.get('model', NOT_COMPUTED)}, "
                f"{int(stats.get('fallbacks', 0) or 0):,} of "
                f"{int(stats.get('cells', 0) or 0):,} cells fell back to the "
                f"mock repairer ({rate:.1%}) — {verdict}."
            )
        lines.append(
            "- T2 and T4's cost interval are still mock-based: "
            "repeated_splits takes no repair mode. Only T1, T3 and the "
            "predictions CSV reflect the real model."
        )
    if quick:
        lines.append(
            f"- This was a --quick run ({QUICK_N_RECORDS}-record cap, "
            f"{QUICK_SPLITS} splits). Not paper-grade."
        )
    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_experiment.py",
        description=(
            "Phase 13 experiment runner: regenerate every reported table, "
            "figure and prediction file for the risk-aware data-repair "
            "pipeline. Deterministic for a given set of flags."
        ),
        epilog=(
            "Costs are abstract experimental penalty units, never money. "
            "Ground truth is read only after predictions exist (CLAUDE.md "
            "section 3). See docs/phase13_plan.md for the protocol."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--dataset", choices=DATASET_CHOICES,
        help="run one dataset",
    )
    target.add_argument(
        "--all", action="store_true",
        help="run every dataset (%s)" % ", ".join(DATASET_CHOICES),
    )
    parser.add_argument(
        "--mode", choices=("mock", "real"), default="mock",
        help=(
            "'mock' uses src/repair/mock_repair.py; 'real' calls the Phase 12 "
            "LLM repairer and needs a key plus --llm-live"
        ),
    )
    parser.add_argument(
        "--llm-live", action="store_true",
        help=(
            "confirm that --mode real may send billed API calls. Without it "
            "--mode real refuses rather than spending quota by accident"
        ),
    )
    parser.add_argument(
        "--llm-cache", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "reuse cached model responses (--no-llm-cache forces fresh calls). "
            "The cache makes a real-mode re-run free and reproducible"
        ),
    )
    parser.add_argument(
        "--n-records", type=int, default=0,
        help="cap on detections carried per dataset; 0 means no cap",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="pipeline seed (mock-repair and uncertainty noise)")
    parser.add_argument("--splits", type=int, default=20,
                        help="repeated calibration splits for the T2/T4 intervals")
    parser.add_argument("--base-split-seed", type=int, default=1000,
                        help="first split_seed; repeated splits use base..base+splits-1")
    parser.add_argument("--iqr-k", type=float, default=3.0,
                        help="Tukey fence multiplier for the IQR outlier check")
    parser.add_argument(
        "--flag-outliers", choices=("auto", "on", "off"), default="auto",
        help="'auto' uses the per-dataset default (off for smartfactory)",
    )
    parser.add_argument("--output-dir", default="results",
                        help="root for tables/, figures/, predictions/ and run_metadata.json")
    parser.add_argument("--data-root", default="datasets",
                        help="folder containing <dataset>/{clean,dirty}.csv")
    parser.add_argument(
        "--quick", action="store_true",
        help=f"CI mode: n_records={QUICK_N_RECORDS}, splits={QUICK_SPLITS}. NOT paper-grade",
    )
    parser.add_argument("--no-figures", action="store_true",
                        help="write tables and predictions but no PNGs")
    return parser


def resolve_flag_outliers(choice: str) -> Optional[bool]:
    return {"auto": None, "on": True, "off": False}[choice]


# The one honest caveat that belongs in every real-mode refusal: the
# interval machinery has no repair mode, so T2 and T4's cost interval stay
# mock-based however the representative run is produced.
_INTERVAL_CAVEAT = (
    "Note: repeated-split intervals (T2, and T4's cost interval) are NOT "
    "implemented for real mode — src/experiments/aggregate.py always builds "
    "a MockRepairer and takes no repair mode. T1, T3 and the predictions CSV "
    "would use the real repairs; those intervals would not."
)

# No usable key. Names the file and the variable; never a key value.
REAL_MODE_NO_KEY_MESSAGE = (
    "run_experiment.py: --mode real needs an API key and none is configured.\n"
    "Phase 12 real-LLM repair (src/repair/llm_repair.py) is implemented, but "
    "no usable key was found, so every cell would fall back to the mock "
    "repairer and the run would be a mock run with a misleading label.\n"
    "Fix: put LLM_API_KEY=<your key> in .env in the repository root (or set "
    "LLM_API_KEYS=<key1,key2,...> for the rotating pool, and LLM_BASE_URL / "
    "LLM_MODEL for a non-OpenAI provider). .env is gitignored; this script "
    "never prints, logs or stores a key.\n"
    f"{_INTERVAL_CAVEAT}\n"
    "Re-run with --mode mock (the default), which uses "
    "src/repair/mock_repair.py."
)

# Key present but the run was not confirmed. A full real run is ~13,314
# billed calls (392 + 7,093 + 5,829 cells), so it is never implicit.
REAL_MODE_NEEDS_CONFIRMATION_MESSAGE = (
    "run_experiment.py: --mode real also needs --llm-live.\n"
    "Phase 12 real-LLM repair (src/repair/llm_repair.py) is implemented and a "
    "usable key is configured, but a full real-mode run is about 13,314 billed "
    "model calls (breast_cancer 392 + smartfactory 7,093 + adult 5,829), so "
    "this runner never sends them on --mode real alone.\n"
    "Re-run with --mode real --llm-live to confirm, optionally with --quick "
    "or --n-records to bound the spend, and --no-llm-cache to ignore cached "
    "responses.\n"
    f"{_INTERVAL_CAVEAT}\n"
    "Re-run with --mode mock (the default) for the mock pipeline."
)


def resolve_llm_config(dotenv_path: str = ".env") -> LLMConfig:
    """
    Read the LLM configuration from `.env` plus the process environment.

    A function, not an inline call, so a test can replace it with an explicit
    :class:`LLMConfig` and never read the user's real `.env`. The returned
    object redacts its key in ``repr``, ``str`` and ``to_dict``.
    """
    return LLMConfig.from_env(dotenv_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    enable_unicode_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)

    llm_config: Optional[LLMConfig] = None
    if args.mode == "real":
        # Refuse before anything is written, so a refused run leaves no
        # half-built artifact set behind.
        llm_config = resolve_llm_config()
        if not llm_config.is_configured:
            print(REAL_MODE_NO_KEY_MESSAGE, file=sys.stderr)
            return 2
        if not args.llm_live:
            print(REAL_MODE_NEEDS_CONFIRMATION_MESSAGE, file=sys.stderr)
            return 2
        if not args.llm_cache:
            llm_config = replace(llm_config, use_cache=False)

    datasets = list(DATASET_CHOICES) if args.all else [args.dataset]
    n_records = QUICK_N_RECORDS if args.quick else args.n_records
    splits = QUICK_SPLITS if args.quick else args.splits
    if splits < 1:
        parser.error(f"--splits must be >= 1, got {splits}")

    out = Path(args.output_dir)
    tables_dir, figures_dir, predictions_dir = (
        out / "tables", out / "figures", out / "predictions"
    )

    started = time.perf_counter()
    print(_rule())
    print("Phase 13 experiment runner" + ("  [--quick: NOT paper-grade]" if args.quick else ""))
    print(_rule())
    print(
        f"datasets={', '.join(datasets)}  mode={args.mode}  seed={args.seed}  "
        f"splits={splits}  n_records={n_records or 'all'}  "
        f"iqr_k={args.iqr_k:g}  flag_outliers={args.flag_outliers}"
    )
    if llm_config is not None:
        # Never the key: only the model, the pool size and the cache setting.
        print(
            f"repairer  real (src/repair/llm_repair.py)  "
            f"model={llm_config.model}  base_url={llm_config.base_url}  "
            f"keys={llm_config.n_api_keys}  cache={llm_config.use_cache}"
        )
    print(f"output  {out.as_posix()}")
    print()

    runs: List[DatasetRun] = []
    for dataset in datasets:
        print(f"[run] {dataset} ...", flush=True)
        run = measure_dataset(
            dataset,
            n_records=n_records,
            seed=args.seed,
            splits=splits,
            iqr_k=args.iqr_k,
            flag_outliers=resolve_flag_outliers(args.flag_outliers),
            base_split_seed=args.base_split_seed,
            data_root=args.data_root,
            mode=args.mode,
            llm_config=llm_config,
        )
        if run.repeated_error:
            print(f"      repeated_splits unavailable: {run.repeated_error}", file=sys.stderr)
        runs.append(run)

    # ── predictions ───────────────────────────────────────────────────────────
    prediction_counts: Dict[str, int] = {}
    for run in runs:
        path = predictions_dir / f"{run.dataset}.csv"
        prediction_counts[run.dataset] = write_predictions(run, path)

    # ── figures ───────────────────────────────────────────────────────────────
    figures: Dict[str, List[str]] = {}
    figure_skips: List[str] = []
    if args.no_figures:
        figure_skips.append("--no-figures: no PNG written")
    else:
        for run in runs:
            written, skipped = write_figures(run, figures_dir)
            figures[run.dataset] = written
            figure_skips.extend(skipped)

    # ── tables ────────────────────────────────────────────────────────────────
    documents = {
        "t1_repair_quality": t1_document(runs, args.seed, n_records, args.quick),
        "t2_calibration": t2_document(runs, splits, args.seed, args.quick),
        "t3_risk_aware": t3_document(runs, args.seed, n_records, args.quick),
        "t4_policy_comparison": t4_document(runs, splits, args.seed, n_records, args.quick),
    }
    csvs = {
        "t1_repair_quality": _with_repair_mode(t1_csv(runs), runs),
        "t2_calibration": _with_repair_mode(t2_csv(runs, splits), runs),
        "t3_risk_aware": _with_repair_mode(t3_csv(runs), runs),
        "t4_policy_comparison": _with_repair_mode(t4_csv(runs, splits), runs),
    }
    for stem in TABLE_STEMS:
        _write_text(tables_dir / f"{stem}.md", documents[stem])
        header, rows = csvs[stem]
        _write_csv(tables_dir / f"{stem}.csv", header, rows)

    # ── metadata ──────────────────────────────────────────────────────────────
    metadata = build_metadata(
        runs, args,
        n_records=n_records, splits=splits,
        predictions=prediction_counts, figures=figures,
        figure_skips=figure_skips,
        wall_seconds=time.perf_counter() - started,
    )
    _write_text(
        out / "run_metadata.json",
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )

    # ── stdout report ─────────────────────────────────────────────────────────
    for stem in TABLE_STEMS:
        print(documents[stem])
    print(_rule())
    for line in honest_reading(runs, splits, args.quick):
        print(line)
    print(_rule())
    for stem in TABLE_STEMS:
        print(f"[written] {(tables_dir / (stem + '.md')).as_posix()}")
        print(f"[written] {(tables_dir / (stem + '.csv')).as_posix()}")
    for run in runs:
        print(f"[written] {(predictions_dir / (run.dataset + '.csv')).as_posix()} "
              f"({prediction_counts[run.dataset]:,} decided cells)")
    for name in sorted(n for names in figures.values() for n in names):
        print(f"[written] {(figures_dir / name).as_posix()}")
    for skip in figure_skips:
        print(f"[skipped] {skip}")
    print(f"[written] {(out / 'run_metadata.json').as_posix()}")
    print(f"[done] {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
