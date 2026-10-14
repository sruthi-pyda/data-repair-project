"""
experiments/r2_conditional.py
-----------------------------
Route R2 experiment: does conditional modelling detect errors that no
syntactic rule can see, and is *conditioning* the factor that carries the edge?

Run
---
    python experiments/r2_conditional.py
    python experiments/r2_conditional.py --datasets adult --bootstrap 0

Writes ``results/r2_conditional.md`` and prints the same tables.

WHAT IS MEASURED
================
Three scoring modes of ``ConditionalDetector`` on all three datasets:

    full      score = 1 - P(value | rest of row)     the method
    marginal  score = 1 - P(value)                   the control
    ratio     sigmoid(log P(value) - log P(value|row))

"marginal" is the control that decides the headline question. If "full" does
not beat "marginal", then conditioning on the row is NOT what helps and that
is the finding to report.

Primary metric is PR-AUC (average precision), per docs/research_program.md
Section 3: at Adult's 0.25% cell-level prevalence, ROC-AUC and accuracy are
vanity metrics. ROC-AUC is printed anyway, flagged as such, because published
systems quote it.

PROTOCOL
========
- Labels: cell is corrupted iff dirty value != clean value under
  ``ValueComparator`` (tolerance-based, so int64 -> float64 promotion in the
  dirty CSV is not counted as corruption; a naive string compare inflates
  Breast Cancer from 631 to 3,854 "corruptions" and Smart Factory from 7,093
  to 425,610).
- Split: seeded row split, stratified on "row contains at least one corrupted
  cell". 70% fit / 30% evaluate. The detector is fitted on train rows only and
  every reported number comes from eval rows only.
- Bootstrap: percentile 95% CI for PR-AUC, resampling eval *cells*.
- One fit per dataset, scored in all three modes. The three modes then share
  byte-identical fitted models, which makes the ablation a strictly controlled
  comparison instead of two independent runs. This is exact for "marginal":
  its statistics (per-value frequencies, robust medians and scales) are
  computed from the same train slice whether or not an estimator is also
  fitted.

LEAKAGE (CLAUDE.md Section 3)
=============================
``clean_df`` is used in exactly two places, both after prediction:
``_build_labels`` (to produce evaluation labels) and the split stratifier
(derived from those labels). The detector receives ``dirty_df`` and a row mask
and nothing else; ``profile=None`` is passed so that even the clean-derived
``DatasetProfile`` never enters.

ADDITIONAL SLICES
=================
- "non-missing" restricts to eval cells whose dirty value is present. Smart
  Factory's corruption is *exactly* its NaN cells, so its headline score is
  produced entirely by the detector's explicit missing-cell policy
  (``nan_score=1.0``), not by conditional modelling. The non-missing slice is
  where the conditional contribution stands alone.
- Per-column PR-AUC, because the two score families (a categorical point
  probability and a numeric tail probability) share a range but not a measure,
  so a whole-table PR-AUC mixes scales. Adult's ``income`` column is the one
  that matters: it is where the rule detector scores F1 = 0.000.

No number in the output file is written by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import DataLoader  # noqa: E402
from src.detectors.conditional import MODES, ConditionalDetector  # noqa: E402
from src.evaluation.repair_metrics import ValueComparator  # noqa: E402

DATASETS: Tuple[str, ...] = ("adult", "breast_cancer", "smartfactory")
SEED = 42
EVAL_FRACTION = 0.30
OUTPUT = REPO_ROOT / "results" / "r2_conditional.md"


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation backend: prefer the R0 harness, fall back to sklearn
# ═══════════════════════════════════════════════════════════════════════════════


def _resolve_backend() -> Tuple[Optional[object], str]:
    """
    Use route R0's harness (``src.detection_eval``) when it is importable and
    exposes the expected entry points, otherwise compute every metric here
    with sklearn.

    R0 was built concurrently with this route, so this probes rather than
    assumes, and the chosen backend is recorded in the output file. When the
    harness IS used, every PR-AUC is still recomputed with sklearn and the two
    are required to agree (see ``compute_metrics``) -- two independent
    implementations agreeing is worth more than either alone, and it means a
    later change inside the harness cannot silently move these numbers.
    """
    try:
        import src.detection_eval as backend  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on R0 landing
        return None, f"sklearn only (src.detection_eval not importable: {exc})"

    required = ("evaluate_scores", "cell_labels")
    missing = [n for n in required if not callable(getattr(backend, n, None))]
    if missing:
        return None, (
            "sklearn only (src.detection_eval present but lacks "
            f"{', '.join(missing)})"
        )
    return backend, "src.detection_eval.evaluate_scores (cross-checked vs sklearn)"


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Metrics:
    """Detection metrics for one (dataset, mode, slice)."""

    n_cells: int
    n_positive: int
    prevalence: float
    pr_auc: float
    pr_auc_ci: Tuple[float, float]
    best_f1: float
    best_f1_threshold: float
    precision_at_k: float
    recall_at_k: float
    roc_auc: float
    lift: float  # pr_auc / prevalence
    # |harness PR-AUC - sklearn PR-AUC|; NaN when only one backend ran.
    backend_pr_auc_delta: float = float("nan")

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_cells": self.n_cells,
            "n_positive": self.n_positive,
            "prevalence": self.prevalence,
            "pr_auc": self.pr_auc,
            "pr_auc_ci_low": self.pr_auc_ci[0],
            "pr_auc_ci_high": self.pr_auc_ci[1],
            "best_f1": self.best_f1,
            "best_f1_threshold": self.best_f1_threshold,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "roc_auc": self.roc_auc,
            "lift_over_prevalence": self.lift,
            "backend_pr_auc_delta": self.backend_pr_auc_delta,
        }


def _best_f1(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Maximum F1 over every threshold on the precision-recall curve, and the
    threshold achieving it. Re-derived from precision/recall rather than taken
    from any single operating point.
    """
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denom = precision + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    # precision_recall_curve returns one more precision/recall point than
    # thresholds (the trailing (1, 0) point has no threshold).
    best = int(np.nanargmax(f1[:-1])) if len(f1) > 1 else 0
    threshold = float(thresholds[best]) if len(thresholds) else float("nan")
    return float(f1[best]), threshold


def _precision_at_k(
    labels: np.ndarray, scores: np.ndarray, k: int
) -> Tuple[float, float]:
    """
    Precision and recall when flagging the top-k scored cells, with
    k = the true number of corrupted cells. At that budget precision and
    recall coincide numerically, which is the point: it is the single number a
    practitioner with an exact review budget cares about.
    """
    if k <= 0 or k > len(scores):
        return float("nan"), float("nan")
    order = np.argsort(-scores, kind="stable")[:k]
    hits = int(labels[order].sum())
    n_pos = int(labels.sum())
    return hits / k, (hits / n_pos if n_pos else float("nan"))


def _bootstrap_pr_auc(
    labels: np.ndarray, scores: np.ndarray, n_resamples: int, seed: int
) -> Tuple[float, float]:
    """Percentile 95% CI for PR-AUC, resampling eval cells with replacement."""
    if n_resamples <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(labels)
    values: List[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        y = labels[idx]
        if y.min() == y.max():  # a degenerate resample carries no information
            continue
        values.append(float(average_precision_score(y, scores[idx])))
    if not values:
        return float("nan"), float("nan")
    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int,
    seed: int,
    backend: Optional[object] = None,
) -> Metrics:
    """
    Detection metrics for one (labels, scores) pair.

    When ``backend`` is route R0's harness, its ``evaluate_scores`` produces
    the reported numbers and the sklearn path is run as well; the two PR-AUCs
    must agree to 1e-9 or the run aborts. Without a backend the sklearn path
    is authoritative.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(labels.sum())
    prevalence = n_pos / len(labels) if len(labels) else float("nan")

    if n_pos == 0 or n_pos == len(labels):
        nan = float("nan")
        return Metrics(
            n_cells=len(labels),
            n_positive=n_pos,
            prevalence=prevalence,
            pr_auc=nan,
            pr_auc_ci=(nan, nan),
            best_f1=nan,
            best_f1_threshold=nan,
            precision_at_k=nan,
            recall_at_k=nan,
            roc_auc=nan,
            lift=nan,
        )

    # sklearn path -- always computed, so it can act as the cross-check.
    sk_pr_auc = float(average_precision_score(labels, scores))
    sk_f1, sk_threshold = _best_f1(labels, scores)
    sk_p_at_k, sk_r_at_k = _precision_at_k(labels, scores, n_pos)
    sk_roc = float(roc_auc_score(labels, scores))

    pr_auc, f1, threshold = sk_pr_auc, sk_f1, sk_threshold
    p_at_k, r_at_k, roc = sk_p_at_k, sk_r_at_k, sk_roc
    delta = float("nan")

    if backend is not None:
        harness = backend.evaluate_scores(labels, scores)  # type: ignore[attr-defined]
        pr_auc = float(harness.pr_auc)
        f1 = float(harness.best_f1)
        threshold = float(harness.best_f1_threshold)
        p_at_k = float(harness.precision_at_k)
        r_at_k = float(harness.recall_at_k)
        roc = float(harness.roc_auc)
        delta = abs(pr_auc - sk_pr_auc)
        if delta > 1e-9:
            raise AssertionError(
                "src.detection_eval and sklearn disagree on PR-AUC: "
                f"{pr_auc!r} vs {sk_pr_auc!r} (delta {delta:.3e}). "
                "Refusing to report a number two implementations cannot agree on."
            )

    return Metrics(
        n_cells=len(labels),
        n_positive=n_pos,
        prevalence=prevalence,
        pr_auc=pr_auc,
        pr_auc_ci=_bootstrap_pr_auc(labels, scores, n_bootstrap, seed),
        best_f1=f1,
        best_f1_threshold=threshold,
        precision_at_k=p_at_k,
        recall_at_k=r_at_k,
        roc_auc=roc,
        lift=pr_auc / prevalence if prevalence > 0 else float("nan"),
        backend_pr_auc_delta=delta,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Labels and splits (the only places clean data is touched)
# ═══════════════════════════════════════════════════════════════════════════════


def build_labels(
    clean_df: pd.DataFrame,
    dirty_df: pd.DataFrame,
    backend: Optional[object] = None,
) -> pd.DataFrame:
    """
    Per-cell corruption labels: True iff the dirty value differs from the
    clean value under ``ValueComparator``.

    Delegates to ``src.detection_eval.cell_labels`` when the R0 harness is
    available so that every route labels cells identically; the local
    implementation below is the fallback and is deliberately equivalent.

    POST-PREDICTION USE OF CLEAN DATA ONLY. The returned frame is handed to
    the metric functions, never to the detector.
    """
    if backend is not None:
        return backend.cell_labels(clean_df, dirty_df)  # type: ignore[attr-defined]

    comparator = ValueComparator()
    labels = pd.DataFrame(False, index=dirty_df.index, columns=dirty_df.columns)
    for col in dirty_df.columns:
        clean_values = clean_df[col].to_numpy()
        dirty_values = dirty_df[col].to_numpy()
        labels[col] = [
            not comparator.are_equal(d, c)
            for d, c in zip(dirty_values, clean_values)
        ]
    return labels


def stratified_row_split(
    labels: pd.DataFrame, eval_fraction: float, seed: int
) -> np.ndarray:
    """
    Seeded row split stratified on "row contains at least one corrupted cell".

    Returns a boolean train mask. Stratification keeps the corrupted-row rate
    comparable across fit and eval, which matters most on Adult where only
    3.8% of rows carry a corruption at all.
    """
    row_has_error = labels.any(axis=1).to_numpy()
    indices = np.arange(len(labels))
    stratify = row_has_error if 0 < row_has_error.sum() < len(indices) else None
    train_idx, _ = train_test_split(
        indices,
        test_size=eval_fraction,
        random_state=seed,
        stratify=stratify,
    )
    mask = np.zeros(len(labels), dtype=bool)
    mask[train_idx] = True
    return mask


# ═══════════════════════════════════════════════════════════════════════════════
# One dataset
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DatasetResult:
    dataset: str
    n_rows: int
    n_columns: int
    n_eval_rows: int
    skipped_columns: Dict[str, str]
    fit_seconds: float
    overall: Dict[str, Metrics] = field(default_factory=dict)
    non_missing: Dict[str, Metrics] = field(default_factory=dict)
    per_column: Dict[str, Dict[str, Metrics]] = field(default_factory=dict)
    tail_ablation: Dict[str, Metrics] = field(default_factory=dict)


def run_dataset(
    dataset: str,
    data_root: Path,
    n_bootstrap: int,
    seed: int,
    max_iter: int,
    backend: Optional[object] = None,
) -> DatasetResult:
    loader = DataLoader(data_root=data_root)
    clean_df, dirty_df = loader.load(dataset)
    dirty_df = dirty_df[clean_df.columns]

    labels = build_labels(clean_df, dirty_df, backend)
    train_mask = stratified_row_split(labels, EVAL_FRACTION, seed)
    eval_mask = ~train_mask

    print(f"\n[{dataset}] {dirty_df.shape[0]} rows x {dirty_df.shape[1]} cols; "
          f"fit on {int(train_mask.sum())}, evaluate on {int(eval_mask.sum())} rows")

    # ── Fit once on train rows, with profile=None (fully self-supervised) ───
    t0 = time.time()
    detector = ConditionalDetector(mode="full", seed=seed, max_iter=max_iter)
    detector.fit(dirty_df, profile=None, train_mask=train_mask)
    fit_seconds = time.time() - t0
    print(f"[{dataset}] fitted in {fit_seconds:.1f}s; "
          f"skipped {sorted(detector.skipped_columns_)}")

    result = DatasetResult(
        dataset=dataset,
        n_rows=len(dirty_df),
        n_columns=dirty_df.shape[1],
        n_eval_rows=int(eval_mask.sum()),
        skipped_columns=dict(detector.skipped_columns_),
        fit_seconds=fit_seconds,
    )

    eval_labels = labels.loc[eval_mask]
    eval_dirty = dirty_df.loc[eval_mask]
    present = (~eval_dirty.isna()).to_numpy().ravel()
    flat_labels = eval_labels.to_numpy().ravel()

    for mode in MODES:
        t0 = time.time()
        scores = detector.score_cells(dirty_df, mode=mode).loc[eval_mask]
        flat_scores = scores.to_numpy().ravel()
        print(f"[{dataset}] mode={mode:<9} scored in {time.time() - t0:.1f}s")

        result.overall[mode] = compute_metrics(
            flat_labels, flat_scores, n_bootstrap, seed, backend
        )
        result.non_missing[mode] = compute_metrics(
            flat_labels[present], flat_scores[present], n_bootstrap, seed, backend
        )

        per_col: Dict[str, Metrics] = {}
        for col in dirty_df.columns:
            col_labels = eval_labels[col].to_numpy()
            if col_labels.sum() == 0:
                continue
            per_col[col] = compute_metrics(
                col_labels, scores[col].to_numpy(), 0, seed, backend
            )
        result.per_column[mode] = per_col

    # ── Residual-tail ablation (mode="full" only) ──────────────────────────
    # residual_tail affects only the scoring arithmetic: the fitted estimators
    # and the robust residual scales are untouched, so mutating it and
    # re-scoring is exactly equivalent to constructing a second detector with
    # the other tail -- at zero extra fitting cost.
    for tail in ("cauchy", "gaussian"):
        detector.residual_tail = tail
        flat = detector.score_cells(dirty_df, mode="full").loc[eval_mask]
        result.tail_ablation[tail] = compute_metrics(
            flat_labels, flat.to_numpy().ravel(), 0, seed, backend
        )
    detector.residual_tail = "cauchy"

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════════


def _fmt(value: float, places: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{places}f}"


def _overall_table(results: List[DatasetResult], attr: str) -> List[str]:
    lines = [
        "| dataset | mode | cells | positives | prevalence | PR-AUC | 95% CI | "
        "best-F1 | P@k | ROC-AUC | lift |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for res in results:
        table = getattr(res, attr)
        for mode in MODES:
            m = table.get(mode)
            if m is None:
                continue
            ci = (
                f"[{_fmt(m.pr_auc_ci[0], 4)}, {_fmt(m.pr_auc_ci[1], 4)}]"
                if np.isfinite(m.pr_auc_ci[0])
                else "—"
            )
            lines.append(
                f"| {res.dataset} | {mode} | {m.n_cells} | {m.n_positive} | "
                f"{_fmt(m.prevalence, 5)} | **{_fmt(m.pr_auc)}** | {ci} | "
                f"{_fmt(m.best_f1)} | {_fmt(m.precision_at_k)} | "
                f"{_fmt(m.roc_auc)} | {_fmt(m.lift, 2)} |"
            )
    return lines


def _per_column_table(res: DatasetResult) -> List[str]:
    cols = sorted(
        {c for mode in MODES for c in res.per_column.get(mode, {})}
    )
    lines = [
        "| column | positives | prevalence | PR-AUC full | PR-AUC marginal | "
        "PR-AUC ratio | best-F1 full |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for col in cols:
        ref = res.per_column["full"].get(col) or res.per_column["marginal"].get(col)
        if ref is None:
            continue
        cells = []
        for mode in MODES:
            m = res.per_column.get(mode, {}).get(col)
            cells.append(_fmt(m.pr_auc) if m else "—")
        f1_full = res.per_column.get("full", {}).get(col)
        lines.append(
            f"| `{col}` | {ref.n_positive} | {_fmt(ref.prevalence, 4)} | "
            f"{cells[0]} | {cells[1]} | {cells[2]} | "
            f"{_fmt(f1_full.best_f1) if f1_full else '—'} |"
        )
    return lines


def build_report(
    results: List[DatasetResult],
    backend_label: str,
    n_bootstrap: int,
    seed: int,
    max_iter: int,
    total_seconds: float,
) -> str:
    out: List[str] = []
    out.append("# R2 — Conditional-model error detection: measured results")
    out.append("")
    out.append(
        "_Generated by `experiments/r2_conditional.py`. Every number below was "
        "produced by that script; none is written by hand._"
    )
    out.append("")
    out.append("## Setup")
    out.append("")
    out.append(f"- seed: `{seed}`")
    out.append(f"- split: seeded row split, stratified on \"row has any corrupted "
               f"cell\", {int((1 - EVAL_FRACTION) * 100)}% fit / "
               f"{int(EVAL_FRACTION * 100)}% evaluate; **all numbers are eval-only**")
    out.append(f"- detector: `ConditionalDetector`, `max_iter={max_iter}`, "
               f"`profile=None` (fully self-supervised; nothing clean-derived "
               f"enters the detector)")
    out.append(f"- labels: `ValueComparator` (tolerance-based) on clean vs dirty, "
               f"post-prediction only")
    out.append(f"- metric backend: `{backend_label}`")
    out.append(f"- bootstrap resamples for the PR-AUC CI: {n_bootstrap}")
    out.append(f"- total runtime: {total_seconds:.1f}s")
    out.append("")
    out.append("Modes:")
    out.append("")
    out.append("| mode | score | role |")
    out.append("|---|---|---|")
    out.append("| `full` | `1 - P(value \\| rest of row)` | the method |")
    out.append("| `marginal` | `1 - P(value)` | **the control** — if `full` does "
               "not beat this, conditioning is not what helps |")
    out.append("| `ratio` | `sigmoid(log P(value) - log P(value \\| row))` | "
               "isolates \"marginally common, conditionally rare\" |")
    out.append("")
    out.append("One fit per dataset, scored in all three modes, so the modes "
               "share byte-identical fitted models.")
    out.append("")

    out.append("## 1. Overall (all eval cells)")
    out.append("")
    out.append("PR-AUC is the primary metric (`docs/research_program.md` §3). "
               "`lift` is PR-AUC / prevalence. ROC-AUC is a vanity metric at "
               "this prevalence and is shown only for comparability with "
               "published systems.")
    out.append("")
    out.extend(_overall_table(results, "overall"))
    out.append("")

    out.append("## 2. Non-missing eval cells only")
    out.append("")
    out.append(
        "Restricted to cells whose dirty value is present. The detector gives "
        "every missing cell `nan_score=1.0` — an explicit missing-token rule, "
        "not conditional modelling. Smart Factory's corruption is *exactly* "
        "its NaN cells, so this slice removes all of its positives and its row "
        "is undefined; that is the honest reading of its headline score. This "
        "slice is where conditional modelling stands on its own."
    )
    out.append("")
    out.extend(_overall_table(results, "non_missing"))
    out.append("")

    out.append("## 3. Per column")
    out.append("")
    out.append(
        "A whole-table PR-AUC mixes two score families that share a range but "
        "not a measure (a categorical point probability vs a numeric tail "
        "probability), so per-column numbers are reported too. Columns with no "
        "eval positives are omitted."
    )
    for res in results:
        out.append("")
        out.append(f"### {res.dataset}")
        out.append("")
        out.extend(_per_column_table(res))
    out.append("")

    out.append("## 4. Residual-tail ablation (`mode=\"full\"`, all eval cells)")
    out.append("")
    out.append(
        "The numeric branch turns a standardised residual into a plausibility "
        "through a tail function. `gaussian` is the textbook choice but its "
        "score `1 - tail` rounds to exactly 1.0 once |z| exceeds about 8.3, "
        "collapsing every strong numeric outlier onto one score; PR-AUC is a "
        "pure ranking metric, so that loses real resolution. `cauchy` decays "
        "like 1/|z| and never saturates. The default is `cauchy` **because of "
        "the numbers below**, not by preference. Same fitted models in both "
        "rows -- only the scoring arithmetic differs."
    )
    out.append("")
    out.append("| dataset | residual tail | PR-AUC | best-F1 | P@k |")
    out.append("|---|---|---:|---:|---:|")
    for res in results:
        for tail, m in res.tail_ablation.items():
            out.append(
                f"| {res.dataset} | {tail} | {_fmt(m.pr_auc)} | "
                f"{_fmt(m.best_f1)} | {_fmt(m.precision_at_k)} |"
            )
    out.append("")

    out.append("## 5. What each run fitted")
    out.append("")
    out.append("| dataset | rows | cols | eval rows | fit time | skipped columns |")
    out.append("|---|---:|---:|---:|---:|---|")
    for res in results:
        skipped = (
            ", ".join(f"`{c}` ({r})" for c, r in sorted(res.skipped_columns.items()))
            or "none"
        )
        out.append(
            f"| {res.dataset} | {res.n_rows} | {res.n_columns} | "
            f"{res.n_eval_rows} | {res.fit_seconds:.1f}s | {skipped} |"
        )
    out.append("")
    out.append(
        "Skipped columns score 0.0 by design: an ID-like or constant column has "
        "no conditional structure to model. On Breast Cancer this forfeits the "
        "corruptions living in the two identifier columns — a stated cost."
    )
    out.append("")

    out.append("## 6. Machine-readable")
    out.append("")
    out.append("```json")

    def _json_safe(value):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        raise TypeError(value)

    out.append(
        json.dumps(
            {
                res.dataset: {
                    "overall": {m: v.to_dict() for m, v in res.overall.items()},
                    "non_missing": {
                        m: v.to_dict() for m, v in res.non_missing.items()
                    },
                    "per_column": {
                        mode: {c: v.to_dict() for c, v in cols.items()}
                        for mode, cols in res.per_column.items()
                    },
                    "tail_ablation": {
                        t: v.to_dict() for t, v in res.tail_ablation.items()
                    },
                    "skipped_columns": res.skipped_columns,
                }
                for res in results
            },
            indent=2,
            allow_nan=False,
            default=_json_safe,
        )
    )
    out.append("```")
    out.append("")
    return "\n".join(out)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--data-root", default=str(REPO_ROOT / "datasets"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap", type=int, default=100,
                        help="PR-AUC bootstrap resamples (0 disables)")
    parser.add_argument("--max-iter", type=int, default=100,
                        help="boosting iterations per column model")
    parser.add_argument("--out", default=str(OUTPUT))
    args = parser.parse_args(argv)

    backend, backend_label = _resolve_backend()
    print(f"metric backend: {backend_label}")

    started = time.time()
    results: List[DatasetResult] = []
    for dataset in args.datasets:
        results.append(
            run_dataset(
                dataset=dataset,
                data_root=Path(args.data_root),
                n_bootstrap=args.bootstrap,
                seed=args.seed,
                max_iter=args.max_iter,
                backend=backend,
            )
        )
    total = time.time() - started

    report = build_report(
        results, backend_label, args.bootstrap, args.seed, args.max_iter, total
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print("\n" + "=" * 78)
    print("OVERALL (all eval cells)")
    print("=" * 78)
    for line in _overall_table(results, "overall"):
        print(line)
    print("\n" + "=" * 78)
    print("NON-MISSING EVAL CELLS ONLY")
    print("=" * 78)
    for line in _overall_table(results, "non_missing"):
        print(line)
    for res in results:
        print(f"\nPER COLUMN — {res.dataset}")
        for line in _per_column_table(res):
            print(line)
    print(f"\nwrote {out_path}  ({total:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
