"""
experiments/three_actions_ablation.py
-------------------------------------
Step 6 of docs/three_actions_plan.md: the honest main results table.

The branch `feat/calibrated-abstention` made two separate changes:

    (1) the EVALUATOR now charges C_abstain only when the abstained cell
        was genuinely corrupted      (Step 3, src/evaluation/cost_metrics.py)
    (2) the POLICY now uses E[ABSTAIN] = q * C_abstain with a LEARNED
        q = P(cell genuinely corrupted)
                                     (Steps 1-4, detection_calibrator + policy)

Either one alone moves the realised cost, so a single before/after number
cannot say which mechanism is responsible. This script separates them:

    V1  original          q = 1,            abstain charged unconditionally
                          (= Sruthi's pre-branch semantics, HEAD 5d8cc22)
    V2  eval-only         q = 1,            abstain charged only if corrupted
                          (isolates change (1))
    V3  proposed          learned q (s=1),  conditional charge
                          (V2 + change (2) — the shipped branch)
    V4  q from raw score   q = detection_score, conditional charge
                          (the naive q that was rejected in plan section 0)
    V5  no smoothing      learned q (s=0),  conditional charge
                          (V3 without the Beta(s,s) prior)

V1 -> V2 is the gain attributable to the evaluator change alone.
V2 -> V3 is the gain attributable to the learned q.
V4 and V5 check that V3's gain is not an artefact of *how* q was obtained.

COST SEMANTICS
==============
All costs are abstract experimental penalty units (configs/costs.yaml), not
money and not supplied by REIN. C_abstain = `c_leave_error` = the cost of
leaving a GENUINE error unrepaired, so the realised charge is

    cost_i = C_wrong   * [AUTO and repair wrong]
           + C_human   * [HUMAN]
           + C_abstain * [ABSTAIN] * corrupt_i        (interpretation (a))

V1 drops the `* corrupt_i` factor (interpretation (b)). See
docs/cost_semantics.md sections 1-2 for the derivation and section 3 for
why a realised cost of 0 on Adult means "zero residual-corruption cost on
the flagged cells" and must always be printed beside the count of
undetected genuine corruptions.

BASELINES
=========
Always-auto and the confidence-threshold baseline are taken from the
Phase 9 classes directly (`AlwaysAutoBaseline`, `ConfidenceThresholdBaseline`)
and scored with the Phase 10 `CostEvaluator`, NOT via
`src.pipeline.evaluate_policies` — that helper re-runs `decide_all` once per
record (O(n^2)) and is unusable at Smart Factory's 42,732 detections.
Neither baseline ever emits ABSTAIN, so their realised cost is identical
under interpretations (a) and (b) and is reported once per scenario.

LEAKAGE
=======
`clean_df` is read only after every decision exists: for the 0/1 corruption
labels that train q (the permitted use, CLAUDE.md section 3, symmetric with
Phase 6), and for scoring. No variant passes ground truth into a decision.

Run:
    PYTHONIOENCODING=utf-8 python experiments/three_actions_ablation.py

Writes results/tables/three_actions_ablation.md and prints the same tables.
Deterministic: seed 42 throughout, identical output on re-run.
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.baselines import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    AlwaysAutoBaseline,
    ConfidenceThresholdBaseline,
    wrap_proposed_method,
)
from src.calibration.detection_calibrator import fit_detection_calibrator  # noqa: E402
from src.decision.policy import DecisionAction, DecisionPolicy, DecisionResult  # noqa: E402
from src.evaluation.cost_metrics import CostEvaluator  # noqa: E402
from src.evaluation.repair_metrics import ValueComparator  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402
from src.risk.cost_model import CostConfig  # noqa: E402

COSTS_YAML = Path("configs/costs.yaml")
OUT_PATH = Path("results/tables/three_actions_ablation.md")
SEED = 42

# (dataset, iqr_k, flag_outliers, cap)   cap 0 = full data
SCENARIOS: List[Tuple[str, float, Optional[bool], int]] = [
    ("breast_cancer", 3.0, None, 0),
    ("breast_cancer", 1.5, None, 0),
    ("smartfactory", 3.0, False, 0),
    ("smartfactory", 3.0, True, 0),
    ("adult", 3.0, None, 0),
]

# Genuine-corruption counts quoted in docs/cost_semantics.md section 3, used
# as a cross-check on the independently measured numbers.
DOC_CORRUPTION_COUNTS = {"breast_cancer": 630, "smartfactory": 7093, "adult": 1701}

VARIANT_DOC = {
    "V1 original": "q=1, abstain charged unconditionally (pre-branch)",
    "V2 eval-only": "q=1, abstain charged only when cell corrupted",
    "V3 proposed": "learned q (smoothing=1), conditional charge",
    "V4 q=raw score": "q=detection_score (rejected idea), conditional charge",
    "V5 no smoothing": "learned q (smoothing=0), conditional charge",
}

AUTO, HUMAN, ABSTAIN = (
    DecisionAction.AUTO_REPAIR,
    DecisionAction.HUMAN_REVIEW,
    DecisionAction.ABSTAIN,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Ground-truth helpers (all read clean_df AFTER decisions exist)
# ═══════════════════════════════════════════════════════════════════════════════


def _is_corrupted(bundle, i: int) -> bool:
    """True if carried record *i*'s dirty cell really differs from clean.

    Same rule as DetectionLabelBuilder / CostEvaluator (NaN == NaN, numeric
    tolerance), written out here so the ablation does not depend on either.
    """
    r = bundle.unc_records[i]
    clean = bundle.clean_df.reset_index(drop=True).at[r.row_id, r.column]
    dirty = bundle.dirty_df.reset_index(drop=True).at[r.row_id, r.column]
    if pd.isna(clean) and pd.isna(dirty):
        return False
    if pd.isna(clean) or pd.isna(dirty):
        return True
    try:
        return not np.isclose(float(clean), float(dirty))
    except (TypeError, ValueError):
        return str(clean).strip() != str(dirty).strip()


def full_dataset_corruptions(
    clean_df: pd.DataFrame, dirty_df: pd.DataFrame, skip_columns: Sequence[str]
) -> Tuple[int, int]:
    """
    Count genuinely corrupted cells in the WHOLE dataset (not just flags).

    Returns (total, total_excluding_skip_columns). Vectorised pre-filter —
    exactly one of the two values NA is a corruption, both NA is not, and
    the remaining candidates (string forms differ) go through
    ValueComparator so that 371 == 371.0.
    """
    comparator = ValueComparator()
    skip = set(skip_columns)
    total = 0
    total_ex = 0
    for col in clean_df.columns:
        if col not in dirty_df.columns:
            continue
        c = clean_df[col].reset_index(drop=True)
        d = dirty_df[col].reset_index(drop=True)
        na_c, na_d = c.isna(), d.isna()
        n_col = int((na_c ^ na_d).sum())
        both_present = (~na_c) & (~na_d)
        str_differs = c.astype(str).str.strip() != d.astype(str).str.strip()
        for i in c.index[both_present & str_differs]:
            if not comparator.are_equal(d.at[i], c.at[i]):
                n_col += 1
        total += n_col
        if col not in skip:
            total_ex += n_col
    return total, total_ex


# ═══════════════════════════════════════════════════════════════════════════════
# One ablation variant
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class VariantStats:
    """Measured outcome of one (scenario, variant) cell of the table."""

    name: str
    n_decided: int
    n_auto: int
    n_human: int
    n_abstain: int
    total_cost: float
    mean_cost: float
    n_wrong_auto: int
    n_abstain_on_clean: int
    n_abstain_on_corrupted: int
    q_summary: str


def _realised(
    bundle,
    idx: Sequence[int],
    actions: Dict[int, DecisionAction],
    costs: CostConfig,
    corrupted: Dict[int, bool],
    conditional_abstain: bool,
) -> Tuple[float, int, int, int]:
    """
    Realised cost under interpretation (a) (``conditional_abstain=True``) or
    (b) (``False``), plus (n_wrong_auto, n_abstain_on_clean,
    n_abstain_on_corrupted).

    C_wrong   charged when an applied repair is wrong
    C_human   charged per cell routed to a reviewer
    C_abstain charged when a genuinely corrupted cell is left unrepaired
              (interpretation (a)); unconditionally under (b)
    """
    total = 0.0
    n_wrong = n_abs_clean = n_abs_corrupt = 0
    for i in idx:
        action = actions[i]
        if action == AUTO:
            if bundle.cal_samples[i].correct == 1:
                continue
            n_wrong += 1
            total += costs.c_wrong_repair
        elif action == HUMAN:
            total += costs.c_human_review
        else:
            if corrupted[i]:
                n_abs_corrupt += 1
                total += costs.c_leave_error
            else:
                n_abs_clean += 1
                if not conditional_abstain:
                    total += costs.c_leave_error
    return total, n_wrong, n_abs_clean, n_abs_corrupt


def run_variant(
    bundle,
    policy: DecisionPolicy,
    costs: CostConfig,
    idx: Sequence[int],
    corrupted: Dict[int, bool],
    name: str,
    qs: Sequence[float],
    conditional_abstain: bool,
) -> Tuple[VariantStats, List[DecisionResult]]:
    """Decide every carried record with this variant's q, then score it."""
    decisions = {
        i: policy.decide(bundle.calibrated_probs[i], corruption_probability=float(qs[i]))
        for i in idx
    }
    actions = {i: d.action for i, d in decisions.items()}
    counts = Counter(actions.values())
    total, n_wrong, n_abs_clean, n_abs_corrupt = _realised(
        bundle, idx, actions, costs, corrupted, conditional_abstain
    )
    q_vals = np.array([qs[i] for i in idx], dtype=float)
    if q_vals.size and np.allclose(q_vals, q_vals[0]):
        # 4 dp: a smoothed all-corrupted group sits at 0.9999, not 1.0000, and
        # the difference matters when reading "is this the q = 1 fallback?".
        q_summary = f"all {q_vals[0]:.4f}"
    elif q_vals.size:
        q_summary = f"{q_vals.min():.4f}–{q_vals.max():.4f} (mean {q_vals.mean():.4f})"
    else:
        q_summary = "—"
    stats = VariantStats(
        name=name,
        n_decided=len(idx),
        n_auto=counts[AUTO],
        n_human=counts[HUMAN],
        n_abstain=counts[ABSTAIN],
        total_cost=total,
        mean_cost=total / max(len(idx), 1),
        n_wrong_auto=n_wrong,
        n_abstain_on_clean=n_abs_clean,
        n_abstain_on_corrupted=n_abs_corrupt,
        q_summary=q_summary,
    )
    return stats, [decisions[i] for i in idx]


# ═══════════════════════════════════════════════════════════════════════════════
# One scenario
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ScenarioResult:
    """Everything measured for one (dataset, detector setting) row group."""

    dataset: str
    iqr_k: float
    flag_outliers: bool
    cap: int
    costs: CostConfig
    n_decided: int
    n_corrupted_decided: int
    n_detected_total: int
    full_corruptions: int
    full_corruptions_excl_id: int
    n_detected_genuine: int
    raw_brier: str
    q_brier: str
    det_fitted: bool
    det_note: Optional[str]
    p_method: str
    p_note: Optional[str]
    variants: List[VariantStats] = field(default_factory=list)
    baselines: List[Tuple[str, float, float, int, int, int]] = field(default_factory=list)
    crosscheck: str = ""

    @property
    def label(self) -> str:
        return f"{self.dataset} k={self.iqr_k:g} outliers={'on' if self.flag_outliers else 'off'}"

    def variant(self, name: str) -> VariantStats:
        return next(v for v in self.variants if v.name == name)


def _baseline_rows(bundle, costs: CostConfig, idx: Sequence[int]):
    """Always-auto and the best (lowest-cost) confidence-threshold baseline."""
    unc = [bundle.unc_records[i] for i in idx]
    evaluator = CostEvaluator.from_cost_config(bundle.clean_df, costs)
    rows = []

    auto_report = evaluator.evaluate(
        AlwaysAutoBaseline(bundle.dataset).run(unc), dataset=bundle.dataset
    )
    rows.append((
        "always-auto",
        auto_report.total_realized_cost,
        auto_report.mean_cost_per_record,
        auto_report.n_auto,
        auto_report.n_human,
        auto_report.n_wrong_auto,
    ))

    best = None
    for t in DEFAULT_THRESHOLDS:
        baseline = ConfidenceThresholdBaseline(threshold=t, dataset_name=bundle.dataset)
        report = evaluator.evaluate(baseline.run(unc), dataset=bundle.dataset)
        if best is None or report.total_realized_cost < best[1]:
            best = (
                f"conf-threshold {t:.2f} (best of {len(DEFAULT_THRESHOLDS)})",
                report.total_realized_cost,
                report.mean_cost_per_record,
                report.n_auto,
                report.n_human,
                report.n_wrong_auto,
            )
    if best is not None:
        rows.append(best)
    return rows


def run_scenario(dataset: str, iqr_k: float, flag_outliers, cap: int) -> ScenarioResult:
    bundle = run_pipeline(
        dataset, n_records=cap, seed=SEED, iqr_k=iqr_k, flag_outliers=flag_outliers
    )
    costs = CostConfig.from_yaml(COSTS_YAML, dataset)
    policy = DecisionPolicy.from_cost_config(costs)
    idx = bundle.decision_indices()
    corrupted = {i: _is_corrupted(bundle, i) for i in idx}

    full_total, full_ex = full_dataset_corruptions(
        bundle.clean_df, bundle.dirty_df, bundle.skip_columns
    )
    report = bundle.detection_calibration

    ones = [1.0] * len(bundle.unc_records)
    scores = [float(np.clip(d.detection_score, 0.0, 1.0)) for d in bundle.detections]
    cal0, _ = fit_detection_calibrator(
        bundle.detections, bundle.detection_labels, seed=bundle.seed,
        eval_fraction=0.30, dataset_name=dataset, smoothing=0.0,
    )
    q_s0 = [float(v) for v in cal0.predict(bundle.detections)]

    result = ScenarioResult(
        dataset=dataset,
        iqr_k=float(iqr_k),
        flag_outliers=bool(bundle.flag_outliers),
        cap=cap,
        costs=costs,
        n_decided=len(idx),
        n_corrupted_decided=sum(corrupted.values()),
        n_detected_total=bundle.n_detected_total,
        full_corruptions=full_total,
        full_corruptions_excl_id=full_ex,
        n_detected_genuine=int(sum(bundle.detection_labels)),
        raw_brier=report.metric_str("raw", "brier") if report else "—",
        q_brier=report.metric_str("calibrated", "brier") if report else "—",
        det_fitted=bool(report.fitted) if report else False,
        det_note=report.limitation if report else None,
        p_method=bundle.calibration_method,
        p_note=bundle.calibration_note,
    )

    spec = [
        ("V1 original", ones, False),
        ("V2 eval-only", ones, True),
        ("V3 proposed", bundle.corruption_probs, True),
        ("V4 q=raw score", scores, True),
        ("V5 no smoothing", q_s0, True),
    ]
    v3_decisions: List[DecisionResult] = []
    for name, qs, conditional in spec:
        stats, decisions = run_variant(
            bundle, policy, costs, idx, corrupted, name, qs, conditional
        )
        result.variants.append(stats)
        if name == "V3 proposed":
            v3_decisions = decisions

    # Independent cross-check of V3's hand-computed total against the Phase 10
    # evaluator, which decides corruption and repair correctness via
    # ValueComparator on clean_df rather than via cal_samples.
    recs = wrap_proposed_method(
        [bundle.unc_records[i] for i in idx],
        [bundle.calibrated_probs[i] for i in idx],
        v3_decisions,
        dataset_name=bundle.dataset,
    )
    cost_report = CostEvaluator.from_cost_config(bundle.clean_df, costs).evaluate(
        recs, dataset=bundle.dataset
    )
    hand = result.variant("V3 proposed").total_cost
    agree = abs(cost_report.total_realized_cost - hand) < 1e-6
    result.crosscheck = (
        f"CostEvaluator {cost_report.total_realized_cost:.1f} vs hand-computed "
        f"{hand:.1f} — {'agree' if agree else 'DISAGREE'}"
    )

    result.baselines = _baseline_rows(bundle, costs, idx)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Rendering
# ═══════════════════════════════════════════════════════════════════════════════

_VARIANT_COLS = (
    "| variant | n | AUTO | HUMAN | ABSTAIN | total cost | mean/record | "
    "wrong-auto | abstain on clean | abstain on corrupted | q |"
)
_VARIANT_SEP = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"


def _variant_row(v: VariantStats) -> str:
    return (
        f"| {v.name} | {v.n_decided} | {v.n_auto} | {v.n_human} | {v.n_abstain} | "
        f"{v.total_cost:.1f} | {v.mean_cost:.4f} | {v.n_wrong_auto} | "
        f"{v.n_abstain_on_clean} | {v.n_abstain_on_corrupted} | {v.q_summary} |"
    )


def _fallback_marks(r: ScenarioResult) -> List[str]:
    marks = []
    if not r.det_fitted:
        marks.append("detection calibrator FELL BACK to q = 1 (an assumption, not evidence)")
    elif r.det_note:
        marks.append(f"detection calibrator note: {r.det_note}")
    if r.p_note:
        marks.append(f"Phase 6 fallback: {r.p_note}")
    return marks


def scenario_block(r: ScenarioResult) -> List[str]:
    """Markdown block for one scenario (also printed to stdout verbatim)."""
    lines: List[str] = []
    lines.append(f"### {r.label}")
    lines.append("")
    lines.append(
        f"Costs (penalty units): C_wrong={r.costs.c_wrong_repair:g}, "
        f"C_human={r.costs.c_human_review:g}, C_abstain={r.costs.c_leave_error:g}. "
        f"Boundaries: p\\*={1 - r.costs.c_human_review / r.costs.c_wrong_repair:.3f}, "
        f"q\\*={r.costs.c_human_review / r.costs.c_leave_error:.3f}."
    )
    lines.append("")
    lines.append(
        f"- decided cells: **{r.n_decided}** "
        f"(of {r.n_detected_total} detections; ID-like columns excluded)"
    )
    lines.append(
        f"- genuinely corrupted among decided cells: **{r.n_corrupted_decided}** "
        f"({100 * r.n_corrupted_decided / max(r.n_decided, 1):.1f}%)"
    )
    doc = DOC_CORRUPTION_COUNTS.get(r.dataset)
    cross = f"; docs/cost_semantics.md section 3 quotes {doc}" if doc is not None else ""
    lines.append(
        f"- genuine corruptions in the FULL dataset: **{r.full_corruptions}** "
        f"({r.full_corruptions_excl_id} outside ID-like columns){cross}"
    )
    lines.append(
        f"- of those, detected at all: **{r.n_detected_genuine}** → "
        f"**{r.full_corruptions - r.n_detected_genuine} false negatives**, "
        f"invisible to every cost number below (no action was ever taken on them)"
    )
    lines.append(
        f"- detection calibration Brier: raw detection_score {r.raw_brier} → "
        f"learned q {r.q_brier} (lower is better)"
    )
    lines.append(f"- Phase 6 calibration method used: **{r.p_method}**")
    for mark in _fallback_marks(r):
        lines.append(f"- ⚠ {mark}")
    lines.append("")
    lines.append(_VARIANT_COLS)
    lines.append(_VARIANT_SEP)
    for v in r.variants:
        lines.append(_variant_row(v))
    lines.append("")
    lines.append("| baseline (no ABSTAIN, so charge rule is irrelevant) | total cost | mean/record | AUTO | HUMAN | wrong-auto |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, total, mean, n_auto, n_human, n_wrong in r.baselines:
        lines.append(
            f"| {name} | {total:.1f} | {mean:.4f} | {n_auto} | {n_human} | {n_wrong} |"
        )
    lines.append("")
    lines.append(f"Cross-check: {r.crosscheck}.")
    if r.dataset == "adult":
        lines.append("")
        lines.append(
            f"**Framing (docs/cost_semantics.md section 3):** a realised cost of 0 here is "
            f"*zero residual-corruption cost on the flagged cells*, not a clean "
            f"dataset. {r.full_corruptions - r.n_detected_genuine} genuine "
            f"corruptions were never detected and enter no arm of the objective."
        )
    lines.append("")
    return lines


def attribution_rows(results: List[ScenarioResult]) -> List[Tuple[str, float, float, float, str]]:
    """Per scenario: total V1→V3 gain, split into evaluator-only and learned-q."""
    rows = []
    for r in results:
        v1 = r.variant("V1 original").total_cost
        v2 = r.variant("V2 eval-only").total_cost
        v3 = r.variant("V3 proposed").total_cost
        total, eval_only, q_gain = v1 - v3, v1 - v2, v2 - v3
        if abs(total) < 1e-9:
            verdict = "no change — nothing to attribute"
        elif abs(eval_only) < 1e-9:
            verdict = "entirely the learned q"
        elif abs(q_gain) < 1e-9:
            verdict = "entirely the evaluator change"
        else:
            verdict = (
                f"evaluator {100 * eval_only / total:.0f}% / "
                f"learned q {100 * q_gain / total:.0f}%"
            )
        rows.append((r.label, total, eval_only, q_gain, verdict))
    return rows


def summary_block(results: List[ScenarioResult]) -> List[str]:
    lines = ["## Summary: realised cost by variant", ""]
    header = "| scenario | " + " | ".join(VARIANT_DOC) + " | always-auto | best threshold |"
    lines.append(header)
    lines.append("|---" * (len(VARIANT_DOC) + 3) + "|")
    for r in results:
        cells = [f"{r.variant(n).total_cost:.1f}" for n in VARIANT_DOC]
        auto = next((t for n, t, *_ in r.baselines if n == "always-auto"), float("nan"))
        thr = next((t for n, t, *_ in r.baselines if n.startswith("conf-threshold")), float("nan"))
        lines.append(
            f"| {r.label} | " + " | ".join(cells) + f" | {auto:.1f} | {thr:.1f} |"
        )
    lines.append("")
    lines.append("## Attribution: which change did the work?")
    lines.append("")
    lines.append("| scenario | V1→V3 cost reduction | from the evaluator change (V1→V2) | from the learned q (V2→V3) | verdict |")
    lines.append("|---|---:|---:|---:|---|")
    for label, total, eval_only, q_gain, verdict in attribution_rows(results):
        lines.append(
            f"| {label} | {total:.1f} | {eval_only:.1f} | {q_gain:.1f} | {verdict} |"
        )
    lines.append("")
    lines.append("## Action distribution by variant")
    lines.append("")
    lines.append("| scenario | variant | AUTO | HUMAN | ABSTAIN |")
    lines.append("|---|---|---:|---:|---:|")
    for r in results:
        for v in r.variants:
            lines.append(
                f"| {r.label} | {v.name} | {v.n_auto} | {v.n_human} | {v.n_abstain} |"
            )
    lines.append("")
    return lines


def _q_variant_note(results: List[ScenarioResult]) -> List[str]:
    """What V4 and V5 measured, stated from the numbers."""
    notes = []
    for r in results:
        v3, v4, v5 = (r.variant(n) for n in ("V3 proposed", "V4 q=raw score", "V5 no smoothing"))
        same4 = (v4.n_auto, v4.n_human, v4.n_abstain) == (v3.n_auto, v3.n_human, v3.n_abstain)
        same5 = (v5.n_auto, v5.n_human, v5.n_abstain) == (v3.n_auto, v3.n_human, v3.n_abstain)
        notes.append(
            f"- **{r.label}**: raw detection_score as q "
            f"{'reproduces' if same4 else 'does NOT reproduce'} the learned-q actions "
            f"(cost {v4.total_cost:.1f} vs {v3.total_cost:.1f}); dropping the smoothing "
            f"prior {'changes nothing' if same5 else 'changes the actions'} "
            f"(cost {v5.total_cost:.1f})."
        )
    return notes


def reading_block(results: List[ScenarioResult]) -> List[str]:
    """Honest reading, with every claim keyed to a measured number above."""
    lines = ["## What this shows", ""]
    attrib = attribution_rows(results)

    only_q = [l for l, t, e, q, _ in attrib if abs(t) > 1e-9 and abs(e) < 1e-9]
    only_eval = [l for l, t, e, q, _ in attrib if abs(t) > 1e-9 and abs(q) < 1e-9]
    mixed = [l for l, t, e, q, _ in attrib if abs(t) > 1e-9 and abs(e) > 1e-9 and abs(q) > 1e-9]
    unchanged = [l for l, t, *_ in attrib if abs(t) < 1e-9]

    lines.append(
        "1. **Where the gain comes from.** The ablation separates the evaluator "
        "change (V1→V2) from the learned corruption probability (V2→V3), because "
        "either alone would move the realised cost."
    )
    if only_q:
        lines.append(
            f"   - Entirely attributable to the learned q: {', '.join(only_q)}. "
            "V2 is identical to V1 there, because with q = 1 the policy never "
            "abstains, so a conditional abstain charge has nothing to act on."
        )
    if only_eval:
        lines.append(f"   - Entirely attributable to the evaluator change: {', '.join(only_eval)}.")
    if mixed:
        lines.append(f"   - Split between the two: {', '.join(mixed)}.")
    if unchanged:
        lines.append(
            f"   - Unchanged by the branch (regression pin): {', '.join(unchanged)}."
        )
    lines.append("")
    lines.append(
        "2. **The two changes are not independent in principle, only in these "
        "runs.** V2 can only differ from V1 on cells where the old policy already "
        "abstained. Under every shipped cost config with q = 1, `C_abstain > "
        "C_human`, so ABSTAIN is unreachable (CLAUDE.md section 5.2) and V1 = V2 by "
        "construction. The evaluator change is therefore a *precondition* for the "
        "learned q to pay off, not an independent source of gain: without it, "
        "abstaining on a false alarm would still be charged `C_abstain` and the "
        "learned q would lose money instead of saving it."
    )
    lines.append("")
    lines.append("3. **How q was obtained matters:**")
    lines.extend(f"   {n}" for n in _q_variant_note(results))
    lines.append("")
    lines.append("## What this does NOT show")
    lines.append("")
    lines.append(
        "- **Not a repair-quality result.** Every variant shares the same "
        "detections, the same mock repairs and the same calibrated p. Only the "
        "abstention arm moves. No number here says a cell was repaired better."
    )
    fn = [
        (r.label, r.full_corruptions - r.n_detected_genuine)
        for r in results
        if r.full_corruptions - r.n_detected_genuine > 0
    ]
    if fn:
        lines.append(
            "- **False negatives dominate and are priced at zero.** "
            + "; ".join(f"{l}: {n} undetected genuine corruptions" for l, n in fn)
            + ". No policy is charged for them, so a low total cost is a statement "
            "about the flagged cells only (docs/cost_semantics.md section 3)."
        )
    lines.append(
        "- **Not a per-cell three-way choice.** `q*` is horizontal, so at a fixed q "
        "only two of the three arms are ever live for a single cell; all three "
        "appear across a population that mixes real errors with false alarms "
        "(docs/cost_semantics.md section 4)."
    )
    lines.append(
        "- **Mock repairs, single split, one seed.** p comes from "
        "`src/repair/mock_repair.py`, not an LLM; calibration uses one 70/30 split "
        "at seed 42 with no confidence interval (CLAUDE.md section 5.4)."
    )
    lines.append(
        "- **The baselines never abstain,** so they are unaffected by the "
        "semantics question this ablation is about; they are context for the cost "
        "scale, not a test of it."
    )
    lines.append(
        "- **Soft leakage is unchanged.** `DatasetProfile` is still built from "
        "`clean_df` (CLAUDE.md section 3), so detector fences and imputation "
        "statistics carry population-level clean knowledge in every variant alike."
    )
    lines.append("")
    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

_HEADER = [
    "# Three-actions ablation — did the learned q improve the decision policy?",
    "",
    "Generated by `experiments/three_actions_ablation.py` (Step 6 of "
    "`docs/three_actions_plan.md`). Seed 42, mock repair mode, full datasets "
    "(no record cap). Every number below was produced by that script; re-running "
    "it reproduces this file exactly.",
    "",
    "Costs are **abstract experimental penalty units** from `configs/costs.yaml` — "
    "not money, not clinical harm, not supplied by the REIN benchmark.",
    "",
    "## Variants",
    "",
    "| variant | q used | realised abstain charge |",
    "|---|---|---|",
    "| V1 original | q = 1 for every cell | unconditional (interpretation (b)) |",
    "| V2 eval-only | q = 1 for every cell | only when the cell was genuinely corrupted (a) |",
    "| V3 proposed | learned per `detection_method`, Beta(1,1) smoothing | conditional (a) |",
    "| V4 q=raw score | raw `detection_score` (rejected in plan section 0) | conditional (a) |",
    "| V5 no smoothing | learned per `detection_method`, no prior | conditional (a) |",
    "",
    "V1 is the pre-branch semantics at HEAD `5d8cc22`; V3 is the shipped branch. "
    "V1→V2 isolates the evaluator change, V2→V3 the learned q.",
    "",
]


def main() -> None:
    results = [run_scenario(*s) for s in SCENARIOS]

    lines: List[str] = list(_HEADER)
    lines += summary_block(results)
    lines.append("## Per-scenario detail")
    lines.append("")
    for r in results:
        lines += scenario_block(r)
    lines += reading_block(results)

    text = "\n".join(lines).rstrip() + "\n"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text, encoding="utf-8")

    print(text)
    print(f"[written] {OUT_PATH}")


if __name__ == "__main__":
    main()
