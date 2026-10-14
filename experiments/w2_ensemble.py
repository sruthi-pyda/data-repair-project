"""
experiments/w2_ensemble.py
--------------------------
Wave-2 route W2.1: does combining cheap detection signals — with a tiny,
explicit label budget — beat BOTH single detectors, and how many labels does
it take?

Run
---
    python experiments/w2_ensemble.py
    python experiments/w2_ensemble.py --datasets breast_cancer --splits 3
    python experiments/w2_ensemble.py --budgets 5 20 --no-leaderboard

Writes ``results/w2_ensemble.md``, updates the shared detection leaderboard
(``results/detection_leaderboard.{json,md}``) through
:func:`src.detection_eval.leaderboard.record`, and prints every table it
writes. Nothing is hand-entered.

WHAT IS MEASURED, AND AGAINST WHAT
==================================
Wave 1 measured (``docs/research_program.md`` §5) that neither detector family
dominates: rules win on breast_cancer (PR-AUC 0.6712 vs 0.6404), the
conditional model wins on adult (0.0120 vs 0.0025, and 0.70 vs 0.13 on the
`income` column where adult's corruption actually lives). This experiment asks
whether their *combination* beats both, and at what label cost.

Every row below is produced by the same object through the same harness, so
the comparison isolates the combination rule:

    rule_only        r_score alone — reproduces the wave-1 rule baseline
                     bit-for-bit (asserted in tests)
    cond_only        the conditional model's `full` score alone
    rank_mean        unsupervised: mean normalised rank of the two
    noisy_or         unsupervised: 1 - (1-rule)(1-cond)
    ensemble_b{k}    meta-classifier on k labelled TRAIN cells, k = 5..100
    ensemble_bfull   every train label — the supervised upper bound
    ensemble_bfull_gbm   the same, with a shallow GBM instead of logistic

The primary metric is PR-AUC with a bootstrap CI over splits
(``docs/research_program.md`` §3). ROC-AUC and oracle best-F1 are printed for
comparability with published systems, flagged as what they are.

PROTOCOL
========
Fixed by ``src.detection_eval.harness.evaluate_detector``: cell-level
corruption labels, stratified seeded cell splits, the detector refitted on
each split's train mask, metrics on that split's eval cells only, bootstrap
CI over splits. Identical to the protocol behind every wave-1 leaderboard row.

Because the meta-classifier is refitted per split on a *fresh* label sample,
the CI includes **label-sample variance**, which is the dominant uncertainty
at k = 5 and precisely what a label-efficiency curve has to show.

LEAKAGE (CLAUDE.md §3)
======================
``clean_df`` is read here for exactly two things:
:func:`src.detection_eval.labels.cell_labels` (evaluation labels + split
stratification) and the :class:`~src.detectors.ensemble.LabelOracle` the
detector may query *at most* ``budget`` times per fit, only at cells inside
that split's train mask. The oracle is constructed with a hard cap of
``budget * n_splits`` and this script **asserts** the realised query count
equals it, so an overspend fails the run rather than inflating a row.

Soft-leakage caveat, inherited unchanged from the wave-1 rule baseline: the
``DatasetProfile`` the rule block needs is built from ``clean_df``, so rule
fences and vocabularies encode clean aggregate statistics. Stated here and in
every leaderboard note.

WHAT THIS EXPERIMENT DOES NOT DO
================================
- It does not refit the unsupervised signal layer per evaluation split. The
  conditional features are instead scored **out of fold over rows** (5 folds),
  so no cell's conditional score comes from a model that saw its row. That
  matters: a single in-sample fit measures PR-AUC 0.5014 on breast_cancer
  where 5-fold out-of-fold measures 0.6408, and wave 1's row-split protocol
  measured 0.6404 — so the out-of-fold control is the *fair* single-detector
  baseline and the in-sample one would be a handicapped straw man.
- It sweeps one cost-free design choice per family (two combiners, two
  learners). It is not a hyper-parameter search, and a tuned ensemble is not
  what the label-efficiency question asks about.
- Smart Factory remains uninformative: its corruption is exactly its NaN
  cells, which the trivial missing-token rule already scores 1.0. A PR-AUC of
  1.0000 there measures the dataset, not the method.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.console import enable_unicode_stdout  # noqa: E402
from src.data.loader import DataLoader  # noqa: E402
from src.data.profiler import profile_dataset  # noqa: E402
from src.detection_eval import leaderboard  # noqa: E402
from src.detection_eval.harness import (  # noqa: E402
    evaluate_detector,
    flatten_cells,
    stratified_cell_splits,
)
from src.detection_eval.labels import cell_labels, label_stats  # noqa: E402
from src.detection_eval.metrics import evaluate_scores  # noqa: E402
from src.detectors.ensemble import (  # noqa: E402
    EnsembleDetector,
    FeatureCache,
    LabelOracle,
)

DATASETS: Tuple[str, ...] = ("adult", "breast_cancer", "smartfactory")
BUDGETS: Tuple[int, ...] = (5, 10, 20, 50, 100)
SEED = 42
N_SPLITS = 5
OUTPUT = REPO_ROOT / "results" / "w2_ensemble.md"

MISSING = "—"

#: Rows every dataset gets before the budget sweep, in report order.
#: ``(label, budget, combiner, learner, sampler)``
ROW_PLAN: Tuple[Tuple[str, Any, str, str, str], ...] = (
    ("rule_only", 0, "rule_only", "logreg", "per_column"),
    ("cond_only", 0, "cond_only", "logreg", "per_column"),
    ("ensemble_b0_rank_mean", 0, "rank_mean", "logreg", "per_column"),
    ("ensemble_b0_noisy_or", 0, "noisy_or", "logreg", "per_column"),
)


# ═══════════════════════════════════════════════════════════════════════════════
# Formatting helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _fmt(interval: Optional[Dict[str, Any]], with_ci: bool = True) -> str:
    """Render an interval dict; ``n = 0`` (undefined) prints ``—``, never 0.0."""
    if not interval or int(interval.get("n", 0)) == 0:
        return MISSING
    mean = float(interval["mean"])
    if not with_ci or int(interval["n"]) < 2:
        return f"{mean:.4f}"
    return f"{mean:.4f} [{float(interval['lo']):.4f}, {float(interval['hi']):.4f}]"


def _mean(interval: Optional[Dict[str, Any]]) -> Optional[float]:
    if not interval or int(interval.get("n", 0)) == 0:
        return None
    return float(interval["mean"])


def _budget_label(budget: Any) -> str:
    return "full" if budget == "full" else str(budget)


# ═══════════════════════════════════════════════════════════════════════════════
# One dataset
# ═══════════════════════════════════════════════════════════════════════════════


class RowResult:
    """One measured row: what was run, what it scored, and what degraded."""

    def __init__(
        self,
        label: str,
        budget: Any,
        combiner: str,
        learner: str,
        sampler: str,
        metrics: Dict[str, Any],
        reports: List[Dict[str, Any]],
        oracle_queries: int,
        seconds: float,
    ) -> None:
        self.label = label
        self.budget = budget
        self.combiner = combiner
        self.learner = learner
        self.sampler = sampler
        self.metrics = metrics
        self.reports = reports
        self.oracle_queries = oracle_queries
        self.seconds = seconds

    @property
    def pr_auc(self) -> Optional[Dict[str, Any]]:
        return self.metrics["intervals"].get("pr_auc")

    @property
    def n_labelled(self) -> int:
        return sum(int(r["n_labelled"]) for r in self.reports)

    @property
    def n_positive_labels(self) -> int:
        return sum(int(r["n_positive_labels"]) for r in self.reports)

    @property
    def n_supervised_splits(self) -> int:
        return sum(1 for r in self.reports if r["used_supervision"])

    @property
    def fallbacks(self) -> List[str]:
        return [r["fallback_reason"] for r in self.reports if r["fallback_reason"]]

    def supervision_cell(self) -> str:
        if self.budget == 0:
            return "none (label-free)"
        n = len(self.reports)
        if self.n_supervised_splits == n:
            return f"{self.n_supervised_splits}/{n} splits"
        return f"**{self.n_supervised_splits}/{n} splits** (rest fell back)"


def run_dataset(
    dataset: str,
    data_root: Path,
    budgets: Sequence[int],
    n_splits: int,
    seed: int,
    with_gbm: bool,
    conditional_max_iter: int,
) -> Dict[str, Any]:
    """Measure every row of :data:`ROW_PLAN` plus the budget sweep."""
    loader = DataLoader(str(data_root))
    clean_df, dirty_df = loader.load(dataset)
    profile = profile_dataset(dataset, loader.discover(dataset), clean_df, dirty_df)

    # clean_df's only uses: evaluation labels, split stratification, and the
    # metered oracle. Nothing else below sees it.
    labels = cell_labels(clean_df, dirty_df)
    stats = label_stats(labels)

    # Reproduce the shipped per-dataset rule configuration, so the rule block
    # here equals the wave-1 baseline's (fences off for smartfactory).
    flag_outliers = dataset != "smartfactory"
    cache = FeatureCache()
    cond_kwargs = {"max_iter": conditional_max_iter}

    plan: List[Tuple[str, Any, str, str, str]] = list(ROW_PLAN)
    # The headline curve uses the default per-column sampler...
    for k in budgets:
        plan.append((f"ensemble_b{k}", int(k), "noisy_or", "logreg", "per_column"))
    # ...and the same curve is repeated with the two global samplers, which is
    # the ablation showing why the informed share is allocated per column. Both
    # are obvious things to do and both collapse on adult.
    for sampler in ("diverse", "top_combiner"):
        for k in budgets:
            plan.append(
                (f"ensemble_b{k}_{sampler}", int(k), "noisy_or", "logreg", sampler)
            )
    plan.append(("ensemble_bfull", "full", "noisy_or", "logreg", "per_column"))
    if with_gbm:
        plan.append(("ensemble_bfull_gbm", "full", "noisy_or", "gbm", "per_column"))

    rows: List[RowResult] = []
    for label, budget, combiner, learner, sampler in plan:
        started = time.time()
        cap = None if budget == "full" else int(budget) * n_splits
        oracle = LabelOracle(labels, max_queries=cap)
        reports: List[Dict[str, Any]] = []

        detector = EnsembleDetector(
            budget=budget,
            label_oracle=oracle,
            profile=profile,
            combiner=combiner,
            learner=learner,
            sampler=sampler,
            seed=seed,
            flag_statistical_outliers=flag_outliers,
            conditional_kwargs=cond_kwargs,
            feature_cache=cache,
            name=label,
        )
        # Capture one report per split: the harness refits per split and the
        # fallbacks are per split, so a single final report would hide them.
        original_fit = detector.fit

        def fit_and_report(*args, _det=detector, _orig=original_fit, _sink=reports, **kwargs):
            out = _orig(*args, **kwargs)
            _sink.append(_det.report())
            return out

        detector.fit = fit_and_report  # type: ignore[method-assign]

        metrics = evaluate_detector(
            detector,
            dataset,
            data_root=str(data_root),
            n_splits=n_splits,
            seed=seed,
        )
        seconds = time.time() - started

        if cap is not None:
            # Hard check: the budget is per fit, n_splits fits happened.
            assert oracle.n_queries == cap, (
                f"{label}: oracle spent {oracle.n_queries} labels, expected {cap}"
            )
        rows.append(
            RowResult(
                label=label,
                budget=budget,
                combiner=combiner,
                learner=learner,
                sampler=sampler,
                metrics=metrics,
                reports=reports,
                oracle_queries=oracle.n_queries,
                seconds=seconds,
            )
        )
        print(
            f"  {dataset:14s} {label:22s} PR-AUC {_fmt(rows[-1].pr_auc):28s} "
            f"({seconds:.1f}s)"
        )

    per_column = _per_column_probe(
        dataset=dataset,
        dirty_df=dirty_df,
        profile=profile,
        labels=labels,
        budgets=budgets,
        seed=seed,
        n_splits=n_splits,
        flag_outliers=flag_outliers,
        cond_kwargs=cond_kwargs,
        cache=cache,
    )

    return {
        "dataset": dataset,
        "label_stats": stats,
        "rows": rows,
        "per_column": per_column,
        "feature_notes": rows[0].reports[0]["feature_notes"] if rows[0].reports else [],
        "coefficients": _coefficients_for(rows),
    }


def _coefficients_for(rows: Sequence[RowResult]) -> Dict[str, Dict[str, float]]:
    """Split-0 logistic weights per supervised row, for the write-up.

    Reported because *which* signal the learner leans on is the actual finding:
    if the weight sits entirely on one family, the ensemble is not combining
    anything.
    """
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        if row.learner != "logreg" or not row.reports:
            continue
        coefficients = row.reports[0].get("coefficients")
        if coefficients:
            out[row.label] = coefficients
    return out


def _per_column_probe(
    dataset: str,
    dirty_df: pd.DataFrame,
    profile: Any,
    labels: pd.DataFrame,
    budgets: Sequence[int],
    seed: int,
    n_splits: int,
    flag_outliers: bool,
    cond_kwargs: Dict[str, Any],
    cache: FeatureCache,
) -> Dict[str, Any]:
    """
    PR-AUC on the most-corrupted column alone, on split 0 only.

    Why: wave 1's sharpest result is per-column — the conditional model scores
    0.70 PR-AUC on adult's `income` while the whole-table number is 0.0120,
    because 14 of 15 columns are guaranteed clean and dilute precision. A
    whole-table comparison alone would hide whether the ensemble *keeps* that
    per-column edge. Split 0 only, so this is a probe with no CI, and it is
    labelled as such.
    """
    per_column_counts = label_stats(labels)["per_column"]
    target = max(per_column_counts, key=lambda c: per_column_counts[c])
    if per_column_counts[target] == 0:
        return {"column": None, "reason": "no corrupted cells in any column"}

    y = flatten_cells(labels).astype(bool)
    eval_idx = stratified_cell_splits(y, n_splits=n_splits, seed=seed)[0]
    train_flat = np.ones(y.size, dtype=bool)
    train_flat[eval_idx] = False
    n_rows = len(dirty_df)
    train_mask = pd.DataFrame(
        {
            col: train_flat[i * n_rows : (i + 1) * n_rows]
            for i, col in enumerate(dirty_df.columns)
        },
        index=dirty_df.index,
    )

    col_pos = list(dirty_df.columns).index(target)
    col_slice = np.zeros(y.size, dtype=bool)
    col_slice[col_pos * n_rows : (col_pos + 1) * n_rows] = True
    keep = np.flatnonzero(col_slice[eval_idx])
    if keep.size == 0 or not y[eval_idx][keep].any():
        return {
            "column": target,
            "reason": "split 0 holds no corrupted cell of this column",
        }

    variants: List[Tuple[str, Any, str]] = [
        ("rule_only", 0, "rule_only"),
        ("cond_only", 0, "cond_only"),
        ("ensemble_b0_noisy_or", 0, "noisy_or"),
    ] + [(f"ensemble_b{k}", int(k), "noisy_or") for k in budgets]

    scores: Dict[str, Optional[float]] = {}
    for label, budget, combiner in variants:
        detector = EnsembleDetector(
            budget=budget,
            label_oracle=LabelOracle(labels),
            profile=profile,
            combiner=combiner,
            sampler="per_column",
            seed=seed,
            flag_statistical_outliers=flag_outliers,
            conditional_kwargs=cond_kwargs,
            feature_cache=cache,
        )
        detector.fit(dirty_df, train_mask)
        flat = flatten_cells(detector.score_cells(dirty_df))
        metrics = evaluate_scores(y[eval_idx][keep], flat[eval_idx][keep])
        scores[label] = metrics.pr_auc

    return {
        "column": target,
        "n_corrupted_in_column": int(per_column_counts[target]),
        "n_eval_cells": int(keep.size),
        "n_eval_positives": int(y[eval_idx][keep].sum()),
        "pr_auc": scores,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════


def _verdict(result: Dict[str, Any]) -> str:
    """
    The one sentence the route has to answer: at what budget, if any, does the
    ensemble beat BOTH single detectors?

    Two bars, both reported: the mean beating both means (weak), and the
    ensemble's CI lower bound clearing the better single's upper bound
    (strong — non-overlapping intervals).
    """
    rows = {r.label: r for r in result["rows"]}
    singles = {
        name: rows[name].pr_auc for name in ("rule_only", "cond_only") if name in rows
    }
    single_means = {k: _mean(v) for k, v in singles.items()}
    if any(v is None for v in single_means.values()):
        return "both single detectors are undefined here, so there is nothing to beat."
    best_name = max(single_means, key=lambda k: single_means[k])  # type: ignore[arg-type]
    best_mean = float(single_means[best_name])  # type: ignore[arg-type]
    best_hi = float(singles[best_name]["hi"])  # type: ignore[index]

    weak: List[str] = []
    strong: List[str] = []
    tied: List[str] = []
    for row in result["rows"]:
        if row.label in ("rule_only", "cond_only"):
            continue
        mean = _mean(row.pr_auc)
        if mean is None:
            continue
        if mean > best_mean + 1e-9:
            weak.append(row.label)
            if float(row.pr_auc["lo"]) > best_hi:  # type: ignore[index]
                strong.append(row.label)
        elif abs(mean - best_mean) <= 1e-9:
            tied.append(row.label)

    best_single = f"best single detector = {best_name} at {best_mean:.4f}"
    if not weak:
        # A tie is a tie. Reporting a tie at a ceiling as a "loss" would be
        # exactly the kind of spin CLAUDE.md §8b rule 8 forbids in the other
        # direction too.
        if tied:
            ceiling = " at the ceiling" if best_mean >= 1.0 - 1e-9 else ""
            return (
                f"**tie{ceiling}** ({best_single}); {len(tied)} configuration(s) "
                f"match it exactly and none beats it. Nothing here can improve on "
                "a detector that is already perfect, so this row measures the "
                "dataset, not the method."
            )
        return (
            f"**no configuration beat both single detectors** ({best_single}); "
            "this dataset is a loss for the route."
        )
    text = f"beats both on the mean ({best_single}): {', '.join(weak)}."
    if strong:
        text += f" Non-overlapping CI (strong): {', '.join(strong)}."
    else:
        text += (
            " No configuration clears the better single's CI upper bound, so the"
            " margin is within split noise."
        )
    return text


def build_report(
    results: Sequence[Dict[str, Any]],
    n_splits: int,
    seed: int,
    budgets: Sequence[int],
    total_seconds: float,
) -> str:
    out: List[str] = [
        "# W2.1 — signal ensembling with an explicit label budget",
        "",
        "Generated by `experiments/w2_ensemble.py`. **Do not edit by hand.**",
        "Every metric comes from `src.detection_eval` (the wave-1 harness), so",
        "these rows are directly comparable with `results/detection_leaderboard.md`.",
        "",
        f"- splits: {n_splits} stratified cell splits, seed {seed}, 30 % eval each",
        f"- budgets swept: {', '.join(str(b) for b in budgets)}, plus 0 and full",
        "- primary metric: **PR-AUC** (average precision) with a 95 % bootstrap CI",
        "  over splits; read it against the prevalence in each heading, since a",
        "  random scorer scores ≈ prevalence",
        "- `best-F1` is an **oracle** threshold (an upper bound, not an operating",
        "  point); `P@k` is R-precision with *k = n_pos*",
        "- `—` means undefined / not measured, never 0.0",
        f"- total runtime {total_seconds:.1f}s",
        "",
        "## What each row is",
        "",
        "| row | labels | combination |",
        "|---|---|---|",
        "| `rule_only` | none | the six Phase-3 rules' own `detection_score`"
        " (reproduces the wave-1 rule baseline bit-for-bit) |",
        "| `cond_only` | none | the conditional model's `full` score alone |",
        "| `ensemble_b0_rank_mean` | none | mean normalised rank of the two signals |",
        "| `ensemble_b0_noisy_or` | none | `1 - (1-rule)(1-cond)` |",
        "| `ensemble_b{k}` | k per split | logistic meta-classifier over 19 features |",
        "| `ensemble_bfull` | all train cells | the supervised upper bound |",
        "| `ensemble_bfull_gbm` | all train cells | same, shallow GBM |",
        "| `ensemble_b{k}_diverse` | k per split | same, but the informed labels"
        " come from the two signals' *global* tops |",
        "| `ensemble_b{k}_top_combiner` | k per split | same, but the informed"
        " labels come from the combiner's *global* top |",
        "",
    ]

    for result in results:
        stats = result["label_stats"]
        out += [
            f"## {result['dataset']}",
            "",
            f"{stats['n_cells']:,} cells, {stats['n_corrupted']:,} corrupted, "
            f"prevalence {stats['prevalence']:.5f} "
            f"({100 * stats['prevalence']:.3f} %)",
            "",
            "| row | budget | PR-AUC [95% CI] | ROC-AUC | best-F1 (oracle) | P@k |"
            " labels spent | positives seen | supervision |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for row in result["rows"]:
            intervals = row.metrics["intervals"]
            out.append(
                f"| `{row.label}` | {_budget_label(row.budget)} "
                f"| {_fmt(row.pr_auc)} "
                f"| {_fmt(intervals.get('roc_auc'), with_ci=False)} "
                f"| {_fmt(intervals.get('best_f1'), with_ci=False)} "
                f"| {_fmt(intervals.get('precision_at_k'), with_ci=False)} "
                f"| {row.n_labelled if row.budget != 0 else 0} "
                f"| {row.n_positive_labels if row.budget != 0 else 0} "
                f"| {row.supervision_cell()} |"
            )
        out += ["", f"**Verdict:** {_verdict(result)}", ""]

        # Label-efficiency curve, flat, so it can be read at a glance.
        curve = [
            (row.label, _mean(row.pr_auc))
            for row in result["rows"]
            if row.label.startswith("ensemble_b") and row.sampler == "per_column"
        ]
        out += [
            "Label-efficiency curve (PR-AUC mean, default `per_column` sampler):",
            "",
        ]
        out += [
            "| " + " | ".join(label for label, _ in curve) + " |",
            "|" + "---|" * len(curve),
            "| "
            + " | ".join(
                (MISSING if value is None else f"{value:.4f}") for _, value in curve
            )
            + " |",
            "",
        ]

        fallbacks = {
            row.label: row.fallbacks for row in result["rows"] if row.fallbacks
        }
        if fallbacks:
            out += ["**Fallbacks (a row that fell back is not a measurement of",
                    "supervision):**", ""]
            for label, reasons in fallbacks.items():
                out.append(f"- `{label}`: {len(reasons)} split(s) — {reasons[0]}")
            out.append("")

        per_column = result["per_column"]
        if per_column.get("pr_auc"):
            out += [
                f"### Per-column probe — `{per_column['column']}` "
                f"({per_column['n_corrupted_in_column']:,} corrupted cells)",
                "",
                "Split 0 only, so **no CI**. Wave 1's sharpest number was"
                " per-column (the conditional model scores far higher on the one",
                "column that actually carries adult's corruption than on the whole",
                "table), so this checks whether the ensemble keeps that edge.",
                "",
                f"{per_column['n_eval_cells']:,} eval cells of this column, "
                f"{per_column['n_eval_positives']:,} corrupted.",
                "",
                "| row | PR-AUC |",
                "|---|---|",
            ]
            for label, value in per_column["pr_auc"].items():
                shown = MISSING if value is None else f"{value:.4f}"
                out.append(f"| `{label}` | {shown} |")
            out.append("")
        elif per_column.get("reason"):
            out += [
                f"### Per-column probe — skipped ({per_column['reason']})",
                "",
            ]

        coefficients = result["coefficients"]
        if coefficients:
            out += [
                "### Where the learner puts its weight (split 0, logistic)",
                "",
                "The finding is *which* family the weight lands on. If one family"
                " takes all of it, the ensemble is not combining anything.",
                "",
            ]
            for label, weights in coefficients.items():
                top = sorted(weights.items(), key=lambda kv: -abs(kv[1]))[:5]
                rendered = ", ".join(f"`{n}` {w:+.3f}" for n, w in top)
                out.append(f"- `{label}`: {rendered}")
            out.append("")

        notes = result["feature_notes"]
        if notes:
            out += ["Feature-build notes: " + "; ".join(notes), ""]

    out += [
        "## Honest limitations",
        "",
        "- The unsupervised signal layer (rules, conditional model) sees no labels",
        "  and is not refitted per evaluation split. The conditional features are",
        "  scored **out of fold over rows** (5 seeded folds), so no cell's",
        "  conditional score comes from a model that saw its row — measured",
        "  PR-AUC on breast_cancer: 0.5014 for a single in-sample fit, 0.6408",
        "  out of fold, against wave 1's row-split 0.6404. Only the",
        "  meta-classifier is refitted per split.",
        "- The per-column probes spend their own label budget (same k, split 0)",
        "  on top of the main table's; they are a diagnostic, not part of the",
        "  headline budget accounting.",
        "- The rule block needs a `DatasetProfile`, which the project builds from",
        "  `clean_df`. Rule fences and vocabularies therefore encode clean",
        "  aggregate statistics — the documented soft-leakage caveat of CLAUDE.md",
        "  §3, inherited unchanged from the wave-1 rule baseline.",
        "- The k labels are spent *informedly*: half allocated round-robin across",
        "  columns (each column's best-ranked, not-trivially-missing cell), half",
        "  uniformly at random. The two global alternatives are reported in full",
        "  as the `_diverse` and `_top_combiner` rows, and **no sampler wins on",
        "  both datasets**. On adult `_top_combiner` draws zero positives at",
        "  every budget up to 100, and `_diverse` draws one in one split of five",
        "  at k=50, because the top of a global ranking there is the rules'",
        "  `fnlwgt` false positives and the blank `workclass`/`occupation` cells,",
        "  all guaranteed clean. On breast_cancer the trade reverses:",
        "  `_top_combiner` is better for k <= 20 (0.7318 against 0.6326 at k=5),",
        "  since at 7.5 % prevalence the global top is already positive-rich.",
        "  Per-column is the default on worst-case regret, and the breast_cancer",
        "  loss is reported rather than hidden. Uniform-only",
        "  sampling at adult's 0.251 % prevalence would return no positive in ~95 %",
        "  of runs, so it would measure the dice, not the learner. Choosing which",
        "  cells to label costs no labels, but it does mean the curve describes",
        "  *active* labelling, which is what Raha and ED2 also do.",
        "- Smart Factory's corruption is exactly its missing cells, so the trivial",
        "  missing-token rule already scores 1.0000 and nothing here can improve",
        "  on it. That row measures the dataset.",
        "- Per-column probes are split 0 only and carry no CI.",
        "",
    ]
    return "\n".join(out) + "\n"


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════


def main(argv: Optional[List[str]] = None) -> int:
    enable_unicode_stdout()
    parser = argparse.ArgumentParser(
        description="W2.1 — signal ensembling with an explicit label budget"
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--data-root", default=str(REPO_ROOT / "datasets"))
    parser.add_argument("--budgets", nargs="+", type=int, default=list(BUDGETS))
    parser.add_argument("--splits", type=int, default=N_SPLITS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--conditional-max-iter",
        type=int,
        default=100,
        help="boosting iterations per column model (matches wave 1)",
    )
    parser.add_argument("--no-gbm", action="store_true", help="skip the GBM row")
    parser.add_argument(
        "--no-leaderboard",
        action="store_true",
        help="do not write results/detection_leaderboard.*",
    )
    parser.add_argument("--out", default=str(OUTPUT))
    args = parser.parse_args(argv)

    started = time.time()
    results: List[Dict[str, Any]] = []
    for dataset in args.datasets:
        print(f"\n=== {dataset}")
        results.append(
            run_dataset(
                dataset=dataset,
                data_root=Path(args.data_root),
                budgets=args.budgets,
                n_splits=args.splits,
                seed=args.seed,
                with_gbm=not args.no_gbm,
                conditional_max_iter=args.conditional_max_iter,
            )
        )
    total = time.time() - started

    report = build_report(results, args.splits, args.seed, args.budgets, total)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    if not args.no_leaderboard:
        for result in results:
            for row in result["rows"]:
                leaderboard.record(
                    name=row.label,
                    dataset=result["dataset"],
                    metrics=row.metrics,
                    notes=_leaderboard_note(row),
                )
        leaderboard.render_markdown()
        print(f"\nupdated {leaderboard.DEFAULT_MD_PATH}")

    print("\n" + report)
    print(f"wrote {out_path}  ({total:.1f}s total)")
    return 0


def _leaderboard_note(row: RowResult) -> str:
    """What the row is and what it does not do — kept on the leaderboard row."""
    base = (
        f"W2.1 ensemble (experiments/w2_ensemble.py), combiner={row.combiner}, "
        f"budget={_budget_label(row.budget)}"
    )
    if row.budget == 0:
        base += ", label-free"
    else:
        base += (
            f", learner={row.learner}, sampler={row.sampler}, "
            f"{row.n_labelled} labels total over "
            f"{len(row.reports)} splits ({row.n_positive_labels} positive), "
            "sampled from train cells only via a metered LabelOracle"
        )
    base += (
        ". 19 per-cell features: six rule indicators + rule score, conditional "
        "full/marginal/ratio, leave-one-out row aggregates, column cardinality/"
        "missing rate, column-kind one-hot. The signal layer sees no labels and "
        "is not refitted per evaluation split; the conditional features are "
        "scored out of fold over rows (5 seeded folds), so no cell's conditional "
        "score comes from a model that saw its row (in-sample instead: 0.5014 vs "
        "0.6408 PR-AUC on breast_cancer). Only the meta-classifier is refitted "
        "per split. Rule features inherit the documented DatasetProfile "
        "soft-leakage caveat (CLAUDE.md §3)."
    )
    if row.fallbacks:
        base += (
            f" WARNING: {len(row.fallbacks)} of {len(row.reports)} splits fell "
            f"back to the unsupervised combiner — {row.fallbacks[0]}"
        )
    return base


if __name__ == "__main__":
    raise SystemExit(main())
