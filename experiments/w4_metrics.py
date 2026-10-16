"""
experiments/w4_metrics.py
-------------------------
Route W4.2 — metric transparency. Answers reviewer gap **G1**
(``docs/research_program.md`` §9):

    "The report describes the Adult detector as going from 0.0025 to 0.0120
    overall, and to 0.70 on the tampered column. These numbers are not
    self-explanatory. Is the metric precision, recall, F1, AUPRC, or something
    else? ... What I'd want: exact metric definitions, precision-recall
    curves, per-column recall and precision, and a clear accounting of false
    positives."

Run
---
    python experiments/w4_metrics.py
    python experiments/w4_metrics.py --datasets adult --no-verify

Writes ``results/w4_metrics.md`` and ``results/figures/w4/pr_<dataset>.png``.
The definitions those tables implement are in ``docs/metrics_definitions.md``.

WHAT IS MEASURED
================
Four detectors × three datasets, all under the one fixed protocol
(``src.detection_eval.harness``: 5 stratified cell splits, seed 42, 30 % eval,
bootstrap CIs):

    rule                  the Phase-3 rule detector (``RuleDetectorAdapter``)
    conditional_full      1 - P(value | rest of row)      (out-of-fold)
    conditional_marginal  1 - P(value)                    the rarity control
    ensemble_noisy_or_b0  1 - (1-rule)(1-conditional), zero labels

NOTHING HERE INVENTS A METRIC
=============================
Every metric comes from ``src.detection_eval.metrics`` and every interval from
``src.experiments.aggregate.bootstrap_interval``, both reused unchanged. What
this script adds is *disaggregation*: the same numbers broken out per column,
per operating point and per false-positive source, which is what G1 asks for.
Two quantities are computed locally because the library deliberately does not
offer them:

- **accuracy**, because it is a vanity metric at 0.25 % prevalence and exists
  in this report only to be dismissed (``docs/metrics_definitions.md`` §7);
- **concrete flag sets** at a threshold or a top-k budget, needed to count
  false positives per column. ``precision_at_k`` returns the *expectation*
  over tie-breaking, which is the right scalar but is not a set, so the
  top-k accounting breaks ties with a seeded permutation and says so.

THREE OPERATING POINTS, BECAUSE ONE WOULD MISLEAD
=================================================
The false-positive accounting is reported at three points, and they disagree
sharply — which is itself the answer to G1:

``s > 0``       the detector's own flag set. The only meaningful point for a
                rule detector, which is a *set*, not a ranking. This is where
                "adult: 5,829 flags, 0 true positives" lives.
``top-k``       k = number of corrupted cells (equal review budget, the
                R-precision point). Comparable across detectors; this is where
                §8's *displacement* finding shows up per column.
``best-F1``     the oracle threshold. Printed with its threshold value,
                because on adult the rule detector's F1-maximising threshold
                is its *minimum* score — i.e. the oracle's best use of that
                detector is to ignore it and flag every cell. A single
                best-F1 number hides that completely.

LEAKAGE (CLAUDE.md §3)
======================
``clean_df`` is read for exactly two things: 0/1 cell labels
(``src.detection_eval.labels.cell_labels``) and split stratification. No
detector here receives labels: ``rule`` is a fixed rule set, the conditional
scores are self-supervised on the dirty table alone, and the ensemble row runs
at ``budget=0`` (no oracle, no labels). The oracle thresholds and top-k cuts
are *evaluation* constructs applied after scoring, and are labelled oracle
wherever they appear.

SELF-CHECKS (a table nobody checked is a table nobody should believe)
=====================================================================
Three assertions run by default; each can fail.

1. **AP re-derived from the plotted curve.** For every dataset × detector,
   ``AP = Σ(Rₙ − Rₙ₋₁)·Pₙ`` is recomputed from the precision-recall curve that
   is drawn, and compared to ``src.detection_eval.metrics.pr_auc``. The
   figure and the scalar are then provably the same object.
2. **Flag-nothing accuracy == 1 − prevalence**, computed from the confusion
   matrix rather than from the formula.
3. **Adapter identity.** ``conditional_full`` and ``ensemble_noisy_or_b0``
   must be bit-identical to ``EnsembleDetector(combiner="cond_only")`` and
   ``EnsembleDetector(combiner="noisy_or")``, which are the objects
   ``results/w2_ensemble.md`` reported. Verified on breast_cancer, where an
   independent second feature build is affordable (~2 min); the code path is
   identical on the other datasets. ``--no-verify`` skips it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from src.console import enable_unicode_stdout  # noqa: E402
from src.data.loader import DataLoader  # noqa: E402
from src.data.profiler import profile_dataset  # noqa: E402
from src.detection_eval.harness import (  # noqa: E402
    RuleDetectorAdapter,
    evaluate_detector,
    flatten_cells,
)
from src.detection_eval.labels import cell_labels, label_stats  # noqa: E402
from src.detection_eval.metrics import best_f1, evaluate_scores, pr_auc  # noqa: E402
from src.detectors.ensemble import (  # noqa: E402
    EnsembleDetector,
    FeatureCache,
    build_features,
)

DATASETS: Tuple[str, ...] = ("adult", "breast_cancer", "smartfactory")
SEED = 42
N_SPLITS = 5
CONDITIONAL_MAX_ITER = 100
CONDITIONAL_FOLDS = 5
MISSING = "—"
MAX_CURVE_POINTS = 50_000
TOP_FP_COLUMNS = 5

OUTPUT = REPO_ROOT / "results" / "w4_metrics.md"
FIGURE_DIR = REPO_ROOT / "results" / "figures" / "w4"

#: Detector display order. Also the legend order on every figure.
DETECTORS: Tuple[str, ...] = (
    "rule",
    "conditional_full",
    "conditional_marginal",
    "ensemble_noisy_or_b0",
)

_COLOURS: Dict[str, str] = {
    "rule": "#B3123A",
    "conditional_full": "#1F4E9C",
    "conditional_marginal": "#5BA3D0",
    "ensemble_noisy_or_b0": "#117733",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Detector adapters
# ═══════════════════════════════════════════════════════════════════════════════


class PrecomputedDetector:
    """
    ``DetectorProtocol`` over an already-computed score vector.

    Why this exists: the conditional block is five gradient-boosted models per
    column (out-of-fold over rows), so recomputing it once per detector and
    once per split would multiply a two-minute build by twelve for a
    bit-identical result. The vector is computed once per dataset by
    :func:`build_features` and sliced here.

    It deliberately has **no** ``fit``. That is not a shortcut: none of the
    four detectors measured in this script fits on labels, and the conditional
    scores are already produced out of fold over rows, so there is nothing a
    per-split refit would isolate. The harness consequently reports
    ``full_is_in_sample = False`` and the whole-dataset tables below are a
    clean measurement rather than an in-sample one.

    Parameters
    ----------
    scores : np.ndarray
        One score per cell, in :func:`flatten_cells` (column-major) order.
    index, columns : pd.Index
        From the frame the scores were built on. ``score_cells`` validates the
        frame it is handed against these, so a mis-aligned call raises instead
        of producing a plausible-looking wrong number.
    name : str
        Leaderboard label.
    """

    def __init__(
        self,
        scores: np.ndarray,
        index: pd.Index,
        columns: pd.Index,
        name: str,
    ) -> None:
        self.scores = np.asarray(scores, dtype=float)
        self.index = index
        self.columns = columns
        self.name = name
        if self.scores.size != len(index) * len(columns):
            raise ValueError(
                f"{name}: {self.scores.size} scores for "
                f"{len(index)}x{len(columns)} cells."
            )

    def score_cells(
        self, dirty_df: pd.DataFrame, profile: Optional[Any] = None
    ) -> pd.DataFrame:
        if not dirty_df.index.equals(self.index) or list(dirty_df.columns) != list(
            self.columns
        ):
            raise ValueError(
                f"{self.name}: score_cells called on a frame that is not the one "
                "the scores were built from; cell alignment would be wrong."
            )
        n_rows = len(self.index)
        data = {
            col: self.scores[i * n_rows : (i + 1) * n_rows]
            for i, col in enumerate(self.columns)
        }
        return pd.DataFrame(data, index=self.index, columns=self.columns, dtype=float)


def build_detectors(
    dataset: str, dirty_df: pd.DataFrame, profile: Any, seed: int
) -> Tuple[Dict[str, Any], List[str]]:
    """
    The four detectors, sharing one feature build.

    Returns
    -------
    (detectors, notes)
        ``detectors`` maps name → object satisfying ``DetectorProtocol``;
        ``notes`` carries anything that degraded during the feature build
        (reported, never swallowed).
    """
    # The shipped per-dataset rule configuration: IQR fences are off for
    # smartfactory, where they produce only false positives. `evaluate_detector`
    # applies the same default, so the rule row here IS the shipped baseline.
    flag_outliers = dataset != "smartfactory"

    features = build_features(
        dirty_df,
        profile=profile,
        seed=seed,
        iqr_k=3.0,
        flag_statistical_outliers=flag_outliers,
        conditional_kwargs={"max_iter": CONDITIONAL_MAX_ITER},
        conditional_folds=CONDITIONAL_FOLDS,
    )

    c_full = features.column("c_full")
    c_marginal = features.column("c_marginal")
    r_score = features.column("r_score")
    noisy_or = 1.0 - (1.0 - np.clip(r_score, 0.0, 1.0)) * (
        1.0 - np.clip(c_full, 0.0, 1.0)
    )

    detectors: Dict[str, Any] = {
        "rule": RuleDetectorAdapter(
            iqr_k=3.0, flag_statistical_outliers=flag_outliers, name="rule"
        ),
        "conditional_full": PrecomputedDetector(
            c_full, dirty_df.index, dirty_df.columns, "conditional_full"
        ),
        "conditional_marginal": PrecomputedDetector(
            c_marginal, dirty_df.index, dirty_df.columns, "conditional_marginal"
        ),
        "ensemble_noisy_or_b0": PrecomputedDetector(
            noisy_or, dirty_df.index, dirty_df.columns, "ensemble_noisy_or_b0"
        ),
    }
    return detectors, list(features.notes)


def verify_adapter_identity(
    dataset: str, dirty_df: pd.DataFrame, profile: Any, detectors: Dict[str, Any], seed: int
) -> List[str]:
    """
    Assert that two of the adapters reproduce ``EnsembleDetector`` exactly.

    The comparison objects are built with their **own** ``FeatureCache``, so
    the conditional block is computed a second time from scratch; a silent
    divergence in settings (seed, folds, ``max_iter``, outlier flag) would
    therefore show up as a mismatch rather than be hidden by a shared cache.
    """
    flag_outliers = dataset != "smartfactory"
    cache = FeatureCache()
    checks: List[str] = []
    for name, combiner in (
        ("conditional_full", "cond_only"),
        ("ensemble_noisy_or_b0", "noisy_or"),
    ):
        reference = EnsembleDetector(
            budget=0,
            profile=profile,
            combiner=combiner,
            seed=seed,
            flag_statistical_outliers=flag_outliers,
            conditional_kwargs={"max_iter": CONDITIONAL_MAX_ITER},
            conditional_folds=CONDITIONAL_FOLDS,
            feature_cache=cache,
        )
        theirs = flatten_cells(reference.fit_score(dirty_df))
        mine = flatten_cells(detectors[name].score_cells(dirty_df, profile))
        max_abs = float(np.max(np.abs(mine - theirs))) if mine.size else 0.0
        if max_abs > 0.0:
            raise AssertionError(
                f"{dataset}/{name} differs from EnsembleDetector(combiner="
                f"{combiner!r}) by up to {max_abs:.3e}; this row would not be "
                "the row results/w2_ensemble.md reported."
            )
        checks.append(
            f"`{name}` == `EnsembleDetector(combiner={combiner!r})` on "
            f"{dataset} (max |Δ| = 0, {mine.size:,} cells, independent feature build)"
        )
    return checks


# ═══════════════════════════════════════════════════════════════════════════════
# Confusion-matrix primitives (local: accuracy and concrete flag sets only)
# ═══════════════════════════════════════════════════════════════════════════════


def confusion(y: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    """
    Counts and the four derived rates for one concrete flag set.

    ``precision`` is ``None`` (renders ``—``) when nothing is flagged, and
    ``recall`` is ``None`` when there is no positive: undefined is not zero
    (``docs/metrics_definitions.md`` §9).
    """
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    n = int(y.size)
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    tn = n - tp - fp - fn
    n_flagged = tp + fp
    n_pos = tp + fn
    precision = (tp / n_flagged) if n_flagged else None
    recall = (tp / n_pos) if n_pos else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision and recall
        else (0.0 if n_pos else None)
    )
    return {
        "n": n,
        "n_flagged": n_flagged,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": ((tp + tn) / n) if n else None,
    }


def threshold_sweep(y: np.ndarray, s: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Every *realisable* operating point of ``flag iff s >= t``.

    Only prefix lengths at the end of a tie group are realisable — ``s >= t``
    cannot split a group of equal scores — which is the same restriction
    :func:`src.detection_eval.metrics.best_f1` applies. Returns arrays over
    those points, in decreasing threshold order.
    """
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    n = int(y.size)
    n_pos = int(y.sum())
    order = np.argsort(-s, kind="stable")
    y_sorted = y[order]
    s_sorted = s[order]

    tp_cum = np.cumsum(y_sorted.astype(np.int64))
    group_end = np.ones(n, dtype=bool)
    group_end[:-1] = s_sorted[:-1] != s_sorted[1:]
    idx = np.flatnonzero(group_end)

    tp = tp_cum[idx].astype(float)
    n_flagged = (idx + 1).astype(float)
    fp = n_flagged - tp
    tn = (n - n_pos) - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = tp / n_flagged
        recall = tp / n_pos if n_pos else np.full(tp.shape, np.nan)
        f1 = 2.0 * tp / (n_flagged + n_pos) if n_pos else np.full(tp.shape, np.nan)
    return {
        "threshold": s_sorted[idx],
        "n_flagged": n_flagged,
        "tp": tp,
        "fp": fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / n,
    }


def topk_mask(s: np.ndarray, k: int, seed: int) -> np.ndarray:
    """
    A concrete top-``k`` flag set, ties broken by a **seeded permutation**.

    Never by array order: cells are enumerated column-major, so array order
    correlates with the column and hence, on adult, with the label — ordering
    ties by index would hand `income` either all of the budget or none of it
    purely by where it sits in the frame. The scalar
    :func:`src.detection_eval.metrics.precision_at_k` uses the exact
    expectation over tie-breaking instead; this set is its sampled companion,
    needed only because "which columns did the flags come from" is a question
    about a set.
    """
    s = np.asarray(s, dtype=float)
    k = int(max(0, min(k, s.size)))
    mask = np.zeros(s.size, dtype=bool)
    if k == 0:
        return mask
    shuffle = np.random.default_rng(seed).permutation(s.size)
    ranked = shuffle[np.argsort(-s[shuffle], kind="stable")]
    mask[ranked[:k]] = True
    return mask


def column_slices(columns: Sequence[Any], n_rows: int) -> Dict[str, slice]:
    """Flat-array slice per column, exploiting :func:`flatten_cells`'s column-major order."""
    return {
        str(col): slice(i * n_rows, (i + 1) * n_rows)
        for i, col in enumerate(columns)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Per-detector measurement
# ═══════════════════════════════════════════════════════════════════════════════


def measure_detector(
    name: str,
    detector: Any,
    dataset: str,
    data_root: Path,
    dirty_df: pd.DataFrame,
    profile: Any,
    y: np.ndarray,
    n_splits: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Everything reported for one (dataset, detector): split metrics, the three
    operating points, the per-column breakdown and the false-positive sources.
    """
    started = time.time()
    harness = evaluate_detector(
        detector,
        dataset,
        data_root=str(data_root),
        n_splits=n_splits,
        seed=seed,
        flag_statistical_outliers=(dataset != "smartfactory"),
    )
    s = flatten_cells(detector.score_cells(dirty_df, profile)).astype(float)
    if s.size != y.size:
        raise ValueError(f"{name}: {s.size} scores for {y.size} labels")

    n_pos = int(y.sum())
    sweep = threshold_sweep(y, s)
    f1_star, t_star = best_f1(y, s)

    # ── the three operating points, as concrete flag sets ──────────────────
    points: Dict[str, Dict[str, Any]] = {}

    positive_scores = s[s > 0.0]
    t_flagged = float(positive_scores.min()) if positive_scores.size else None
    points["s > 0"] = {
        "threshold": t_flagged,
        "mask": s > 0.0,
        "note": "the detector's own flag set",
    }
    points[f"top-k (k = {n_pos:,})"] = {
        "threshold": None,
        "mask": topk_mask(s, n_pos, seed),
        "note": "equal review budget; ties broken by a seeded permutation",
    }
    if t_star is not None:
        points["best-F1 (oracle)"] = {
            "threshold": float(t_star),
            "mask": s >= float(t_star),
            "note": "threshold chosen knowing the labels — an upper bound",
        }
    for key in points:
        points[key]["confusion"] = confusion(y, points[key]["mask"])

    nothing = confusion(y, np.zeros(y.size, dtype=bool))

    # ── per-column breakdown at the oracle and the equal-budget points ─────
    slices = column_slices(list(dirty_df.columns), dirty_df.shape[0])
    oracle_mask = (
        points["best-F1 (oracle)"]["mask"]
        if "best-F1 (oracle)" in points
        else np.zeros(y.size, dtype=bool)
    )
    budget_mask = points[f"top-k (k = {n_pos:,})"]["mask"]
    per_column: List[Dict[str, Any]] = []
    for col, sl in slices.items():
        yc = y[sl]
        per_column.append(
            {
                "column": col,
                "n_cells": int(yc.size),
                "n_corrupted": int(yc.sum()),
                "pr_auc_in_column": pr_auc(yc, s[sl]),
                "oracle": confusion(yc, oracle_mask[sl]),
                "budget": confusion(yc, budget_mask[sl]),
            }
        )

    # ── where the false positives come from, per operating point ───────────
    for key, point in points.items():
        fp_mask = point["mask"] & ~y
        counts = {
            col: int(fp_mask[sl].sum()) for col, sl in slices.items()
        }
        point["fp_by_column"] = sorted(
            ((c, n) for c, n in counts.items() if n > 0),
            key=lambda kv: (-kv[1], kv[0]),
        )

    return {
        "name": name,
        "harness": harness,
        "scores": s,
        "sweep": sweep,
        "best_f1": f1_star,
        "best_f1_threshold": t_star,
        "points": points,
        "flag_nothing": nothing,
        "per_column": per_column,
        "seconds": time.time() - started,
    }


def select_operating_points(
    sweep: Dict[str, np.ndarray], n_pos: int, t_star: Optional[float]
) -> List[int]:
    """
    Up to ~8 informative rows from the full threshold sweep.

    Chosen by *flag budget* (fractions and multiples of the number of
    corrupted cells) rather than by threshold value, because a threshold means
    nothing to a reader while "we flagged 2x as many cells as there are errors"
    does. The oracle row and the extreme rows are always included, so the
    table cannot be made to look better by a lucky choice of ladder.
    """
    n_flagged = sweep["n_flagged"]
    if n_flagged.size == 0:
        return []
    targets = [
        max(1, int(round(frac * n_pos))) for frac in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
    ]
    targets.append(int(n_flagged[-1]))  # flag everything the sweep can flag
    chosen: List[int] = [0]  # the single most suspicious tie group
    for target in targets:
        pos = int(np.searchsorted(n_flagged, target, side="left"))
        chosen.append(min(pos, n_flagged.size - 1))
    if t_star is not None:
        chosen.append(int(np.argmax(sweep["f1"])))
    return sorted(set(chosen))


# ═══════════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════════


def _subsample(x: np.ndarray, y_: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Stride-subsample a curve for drawing only, keeping both endpoints."""
    if x.size <= MAX_CURVE_POINTS:
        return x, y_
    stride = int(np.ceil(x.size / MAX_CURVE_POINTS))
    keep = np.arange(0, x.size, stride)
    if keep[-1] != x.size - 1:
        keep = np.append(keep, x.size - 1)
    return x[keep], y_[keep]


def pr_curve_figure(
    dataset: str,
    prevalence: float,
    results: Dict[str, Dict[str, Any]],
    y: np.ndarray,
    out_path: Path,
) -> Tuple[Path, List[str]]:
    """
    Precision-recall curves for every detector, with the no-skill line.

    Two panels share the data: linear precision (honest about absolute
    magnitude) and log precision (readable at adult's 0.25 % prevalence, where
    every curve otherwise lies on the x-axis). The dashed horizontal line is
    ``precision = prevalence``, the score a random or constant detector gets;
    curve area below it is worse than chance.

    Returns the path written and the AP cross-check lines (assertion 1 of the
    module docstring).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    checks: List[str] = []

    for name in DETECTORS:
        if name not in results:
            continue
        s = results[name]["scores"]
        if not y.any():
            continue
        precision, recall, _ = precision_recall_curve(y.astype(int), s)
        # AP re-derived from THIS curve: AP = sum (R_n - R_{n-1}) * P_n, with
        # the arrays reversed into increasing-recall order.
        r_inc = recall[::-1]
        p_inc = precision[::-1]
        ap_curve = float(np.sum(np.diff(r_inc) * p_inc[1:]))
        ap_lib = pr_auc(y, s)
        if ap_lib is None:
            continue
        delta = abs(ap_curve - float(ap_lib))
        if delta > 1e-9:
            raise AssertionError(
                f"{dataset}/{name}: AP re-derived from the plotted curve is "
                f"{ap_curve:.10f} but src.detection_eval.metrics.pr_auc says "
                f"{float(ap_lib):.10f} (Δ={delta:.2e}). The figure and the "
                "table would not be the same object."
            )
        checks.append(
            f"{dataset}/`{name}`: AP from the plotted curve "
            f"{ap_curve:.6f} == `pr_auc` {float(ap_lib):.6f} (|Δ| = {delta:.1e})"
        )

        rx, px = _subsample(recall, precision)
        for ax in axes:
            ax.plot(
                rx,
                px,
                label=f"{name} (AP {float(ap_lib):.4f})",
                color=_COLOURS.get(name, "#444444"),
                linewidth=1.6,
            )

    for ax, scale in zip(axes, ("linear", "log")):
        ax.axhline(
            prevalence,
            linestyle="--",
            color="#666666",
            linewidth=1.2,
            label=f"no skill = prevalence ({prevalence:.4f})",
        )
        ax.set_xlabel("recall  =  TP / (corrupted cells)")
        ax.set_ylabel("precision  =  TP / (flagged cells)")
        ax.set_xlim(0.0, 1.0)
        ax.set_yscale(scale)
        if scale == "linear":
            ax.set_ylim(0.0, 1.02)
            ax.set_title("linear precision")
        else:
            ax.set_ylim(max(prevalence / 20.0, 1e-5), 1.5)
            ax.set_title("log precision (readable at low prevalence)")
        ax.grid(alpha=0.25, linewidth=0.5)

    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle(
        f"{dataset} — precision-recall, all cells "
        f"(prevalence {prevalence:.6f})",
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path, checks


# ═══════════════════════════════════════════════════════════════════════════════
# One dataset
# ═══════════════════════════════════════════════════════════════════════════════


def run_dataset(
    dataset: str,
    data_root: Path,
    n_splits: int,
    seed: int,
    verify: bool,
    figure_dir: Path,
) -> Dict[str, Any]:
    loader = DataLoader(str(data_root))
    clean_df, dirty_df = loader.load(dataset)
    profile = profile_dataset(dataset, loader.discover(dataset), clean_df, dirty_df)

    # clean_df's only uses: labels + split stratification (CLAUDE.md §3).
    labels = cell_labels(clean_df, dirty_df)
    stats = label_stats(labels)
    y = flatten_cells(labels).astype(bool)

    print(f"  building features ({dirty_df.shape[0]:,}x{dirty_df.shape[1]}) ...")
    t0 = time.time()
    detectors, notes = build_detectors(dataset, dirty_df, profile, seed)
    print(f"  features built in {time.time() - t0:.1f}s")

    checks: List[str] = []
    if verify:
        t0 = time.time()
        checks.extend(verify_adapter_identity(dataset, dirty_df, profile, detectors, seed))
        print(f"  adapter identity verified in {time.time() - t0:.1f}s")

    results: Dict[str, Dict[str, Any]] = {}
    for name in DETECTORS:
        out = measure_detector(
            name,
            detectors[name],
            dataset,
            data_root,
            dirty_df,
            profile,
            y,
            n_splits,
            seed,
        )
        results[name] = out
        iv = out["harness"]["intervals"].get("pr_auc")
        print(
            f"  {dataset:14s} {name:22s} PR-AUC {fmt_interval(iv):30s} "
            f"({out['seconds']:.1f}s)"
        )

    # Assertion 2: flag-nothing accuracy, from counts, equals 1 - prevalence.
    prevalence = float(stats["prevalence"])
    nothing = results[DETECTORS[0]]["flag_nothing"]
    expected = 1.0 - prevalence
    if abs(float(nothing["accuracy"]) - expected) > 1e-12:
        raise AssertionError(
            f"{dataset}: flag-nothing accuracy {nothing['accuracy']} != "
            f"1 - prevalence {expected}"
        )
    checks.append(
        f"{dataset}: flag-nothing accuracy from the confusion matrix "
        f"{float(nothing['accuracy']):.6f} == 1 - prevalence "
        f"{expected:.6f} (and finds 0 of {stats['n_corrupted']:,} corruptions)"
    )

    figure, ap_checks = pr_curve_figure(
        dataset, prevalence, results, y, figure_dir / f"w4_pr_{dataset}.png"
    )
    checks.extend(ap_checks)

    return {
        "dataset": dataset,
        "label_stats": stats,
        "results": results,
        "notes": notes,
        "checks": checks,
        "figure": figure,
        "columns": [str(c) for c in dirty_df.columns],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Formatting
# ═══════════════════════════════════════════════════════════════════════════════


def fmt(value: Optional[float], places: int = 4) -> str:
    """A float, or ``—`` when the quantity is undefined. Never 0.0 for undefined."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return MISSING
    return f"{float(value):.{places}f}"


def fmt_interval(interval: Optional[Dict[str, Any]], places: int = 4) -> str:
    if not interval or int(interval.get("n", 0)) == 0:
        return MISSING
    mean = float(interval["mean"])
    if int(interval["n"]) < 2:
        return f"{mean:.{places}f}"
    return (
        f"{mean:.{places}f} [{float(interval['lo']):.{places}f}, "
        f"{float(interval['hi']):.{places}f}]"
    )


def fmt_int(value: Optional[int]) -> str:
    return MISSING if value is None else f"{int(value):,}"


def _headline_table(result: Dict[str, Any]) -> List[str]:
    stats = result["label_stats"]
    prevalence = float(stats["prevalence"])
    lines = [
        "| detector | prevalence | PR-AUC [95% CI] | PR-AUC / prevalence | "
        "ROC-AUC | best-F1 (oracle) | its threshold | P@k | R@k | "
        "accuracy @best-F1 | accuracy if flagging nothing | distinct scores |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in DETECTORS:
        out = result["results"][name]
        h = out["harness"]
        full = h["full"]
        iv = h["intervals"]
        ap = iv.get("pr_auc")
        mean_ap = float(ap["mean"]) if ap and int(ap.get("n", 0)) else None
        lift = (mean_ap / prevalence) if (mean_ap is not None and prevalence) else None
        oracle = out["points"].get("best-F1 (oracle)")
        acc_oracle = oracle["confusion"]["accuracy"] if oracle else None
        lines.append(
            "| `{name}` | {prev:.6f} | {ap} | {lift} | {roc} | {bf1} | {thr} | "
            "{pk} | {rk} | {acc} | {acc0} | {ndist} |".format(
                name=name,
                prev=prevalence,
                ap=fmt_interval(ap),
                lift=(f"{lift:.1f}x" if lift is not None else MISSING),
                roc=fmt_interval(iv.get("roc_auc")),
                bf1=fmt(full["best_f1"]),
                thr=fmt(full["best_f1_threshold"], 6),
                pk=fmt(full["precision_at_k"]),
                rk=fmt(full["recall_at_k"]),
                acc=fmt(acc_oracle, 6),
                acc0=fmt(out["flag_nothing"]["accuracy"], 6),
                ndist=fmt_int(h["score_structure"]["n_distinct_scores"]),
            )
        )
    return lines


def _per_column_tables(result: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    n_pos = int(result["label_stats"]["n_corrupted"])
    for name in DETECTORS:
        out = result["results"][name]
        oracle = out["points"].get("best-F1 (oracle)")
        thr = fmt(out["best_f1_threshold"], 6)
        n_flag_oracle = fmt_int(oracle["confusion"]["n_flagged"]) if oracle else MISSING
        lines += [
            f"#### `{name}`",
            "",
            f"Oracle threshold {thr} flags {n_flag_oracle} cells in total; the "
            f"equal-budget point flags the top {n_pos:,}.",
            "",
            "| column | cells | corrupted | PR-AUC within column | "
            "flags @oracle | TP @oracle | precision @oracle | recall @oracle | "
            "flags @budget | TP @budget | precision @budget | recall @budget |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        rows = sorted(
            out["per_column"],
            key=lambda r: (-r["n_corrupted"], -r["budget"]["n_flagged"], r["column"]),
        )
        for row in rows:
            o = row["oracle"]
            b = row["budget"]
            lines.append(
                "| `{col}` | {cells} | {corr} | {ap} | {of} | {otp} | {op} | {orr} | "
                "{bf} | {btp} | {bp} | {br} |".format(
                    col=row["column"],
                    cells=fmt_int(row["n_cells"]),
                    corr=fmt_int(row["n_corrupted"]),
                    ap=fmt(row["pr_auc_in_column"]),
                    of=fmt_int(o["n_flagged"]),
                    otp=fmt_int(o["tp"]),
                    op=fmt(o["precision"]),
                    orr=fmt(o["recall"]),
                    bf=fmt_int(b["n_flagged"]),
                    btp=fmt_int(b["tp"]),
                    bp=fmt(b["precision"]),
                    br=fmt(b["recall"]),
                )
            )
        lines.append("")
    return lines


def _fp_tables(result: Dict[str, Any]) -> List[str]:
    lines = [
        "| detector | operating point | threshold | cells flagged | true positives | "
        "**false positives** | FP share of flags | recall | top false-positive columns |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in DETECTORS:
        out = result["results"][name]
        for key, point in out["points"].items():
            c = point["confusion"]
            share = (c["fp"] / c["n_flagged"]) if c["n_flagged"] else None
            top = point["fp_by_column"][:TOP_FP_COLUMNS]
            top_text = (
                ", ".join(f"`{col}` {n:,}" for col, n in top) if top else "none"
            )
            lines.append(
                "| `{name}` | {key} | {thr} | {nf} | {tp} | **{fp}** | {share} | "
                "{rec} | {top} |".format(
                    name=name,
                    key=key,
                    thr=fmt(point["threshold"], 6),
                    nf=fmt_int(c["n_flagged"]),
                    tp=fmt_int(c["tp"]),
                    fp=fmt_int(c["fp"]),
                    share=(f"{100.0 * share:.2f} %" if share is not None else MISSING),
                    rec=fmt(c["recall"]),
                    top=top_text,
                )
            )
    return lines


def _operating_tables(result: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    n_pos = int(result["label_stats"]["n_corrupted"])
    for name in DETECTORS:
        out = result["results"][name]
        sweep = out["sweep"]
        picks = select_operating_points(sweep, n_pos, out["best_f1_threshold"])
        best_i = int(np.argmax(sweep["f1"])) if sweep["f1"].size else -1
        lines += [
            f"#### `{name}`",
            "",
            "| threshold | cells flagged | flags / corrupted cells | precision | "
            "recall | F1 | accuracy | |",
            "|---:|---:|---:|---:|---:|---:|---:|---|",
            "| (flag nothing) | 0 | 0.0x | {p} | {r} | {f} | {a} | |".format(
                p=MISSING,
                r=fmt(out["flag_nothing"]["recall"]),
                f=fmt(out["flag_nothing"]["f1"]),
                a=fmt(out["flag_nothing"]["accuracy"], 6),
            ),
        ]
        for i in picks:
            mark = "**oracle best-F1**" if i == best_i else ""
            lines.append(
                "| {t} | {nf} | {ratio:.2f}x | {p} | {r} | {f} | {a} | {mark} |".format(
                    t=fmt(float(sweep["threshold"][i]), 6),
                    nf=fmt_int(int(sweep["n_flagged"][i])),
                    ratio=float(sweep["n_flagged"][i]) / n_pos if n_pos else float("nan"),
                    p=fmt(float(sweep["precision"][i])),
                    r=fmt(float(sweep["recall"][i])),
                    f=fmt(float(sweep["f1"][i])),
                    a=fmt(float(sweep["accuracy"][i]), 6),
                    mark=mark,
                )
            )
        lines.append("")
    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════


PREAMBLE = """# W4.2 — what the detection numbers actually mean

Generated by `experiments/w4_metrics.py`. **Do not edit by hand.**
Formal definitions of every metric below: **`docs/metrics_definitions.md`**.

This file answers reviewer gap **G1** (`docs/research_program.md` §9): the
report quoted `0.0025 → 0.0120 → 0.70` without saying what the metric was, and
at 0.25 % prevalence a detector can look fine while missing almost everything.

## How to read these numbers

1. **The metric is PR-AUC = average precision**, `AP = Σ(Rₙ − Rₙ₋₁)·Pₙ`, over
   **cells**. Not accuracy, not F1, not AUROC.
2. **A random or constant scorer scores PR-AUC = prevalence.** So prevalence is
   printed in every table, the `PR-AUC / prevalence` column is the lift over
   chance, and the dashed line on every figure is the no-skill baseline. Adult's
   **0.0025 is the floor, not a score.**
3. **PR-AUC is not comparable across datasets.** 0.0120 on adult (prevalence
   0.0025) is 4.8× chance; 0.67 on breast_cancer (prevalence 0.0752) is 8.9×.
   The raw numbers order these the other way round.
4. **best-F1 is an ORACLE threshold** — chosen knowing the labels. It is an
   upper bound, not an operating point, which is why the threshold it uses is
   printed beside it and a full operating-point ladder is given in §5. On adult
   the rule detector's F1-maximising threshold is its *minimum* score: the best
   possible use of that detector is to ignore it and flag every cell.
5. **Accuracy is in these tables only to be dismissed.** Compare any
   `accuracy @best-F1` with the `accuracy if flagging nothing` column beside
   it. On adult a detector that flags nothing is 99.749 % accurate and finds
   0 of 1,701 corruptions.
6. **`—` means undefined, never 0.0** (precision with no flags, ROC-AUC with
   one class, PR-AUC with no positives).
7. **Three operating points** are reported, because one would mislead: the
   detector's own flag set (`s > 0`), an equal review budget (top-k with
   k = number of corrupted cells), and the oracle best-F1 threshold.

## Protocol

- **Unit:** one cell. A cell is corrupted iff its dirty value differs from the
  clean value under `ValueComparator` (`src/detection_eval/labels.py`).
- **Splits:** 5 stratified cell splits, seed 42, 30 % eval each; 95 % percentile
  bootstrap CI over splits via `src.experiments.aggregate.bootstrap_interval`.
  The CI measures split variance only.
- **Whole-dataset tables** (per-column, false positives, operating points, PR
  curves) are computed on **all** cells at the full-set threshold, so the
  counts add up to the dataset totals. This is legitimate here because none of
  the four detectors fits on labels and the conditional scores are produced
  out of fold over rows (5 folds), so the harness reports
  `full_is_in_sample = False` for all of them.
- **Leakage:** `clean_df` is read only for labels and stratification. No
  detector receives a label. Oracle thresholds are evaluation constructs
  applied after scoring and are labelled oracle everywhere.
- Seed 42, deterministic.

## The four detectors

| detector | labels used | score |
|---|---|---|
| `rule` | none | the six Phase-3 rules' `detection_score` (the shipped baseline; IQR fences off on smartfactory, as shipped) |
| `conditional_full` | none | `1 − P(value │ rest of row)`, per-column gradient-boosted models fitted on the dirty table, scored out of fold over rows |
| `conditional_marginal` | none | `1 − P(value)` — the rarity control, which isolates whether *conditioning* is what helps |
| `ensemble_noisy_or_b0` | none | `1 − (1 − rule)(1 − conditional)` at zero labels |
"""

NOT_SHOWN = """## 7. What these numbers do not show

Stated because every one of them is a way this report could be misread.

1. **Nothing here is an operating point a user could pick.** Both of the
   interesting thresholds are label-dependent: best-F1 is an oracle, and the
   top-k budget needs k. Choosing a threshold from unlabelled data is a
   separate problem; it is what `src/calibration/detection_calibrator.py`
   (Phase 6b) and the cost model exist for, and it is not measured here.
2. **PR-AUC and per-column PR-AUC answer different questions, and the
   per-column one is partly oracle.** Restricting to `income` presumes
   knowledge that adult's corruption lives there. The honest claim remains
   `docs/research_program.md` §8: the conditional model ranks corruption well
   *within* a column and cannot tell which column to examine.
3. **The CIs cover split variance only.** For these four detectors there is no
   model refitting per split, so the interval is pure evaluation-sample noise.
   It does not cover seed variance in the conditional fits, corruption-injection
   variance, or dataset choice. Three datasets is three datasets.
4. **The per-column tables are not per-column *detectors*.** Every score comes
   from one whole-table ranking; the tables disaggregate where the flags landed,
   they do not measure a detector that was told which column to look at.
5. **Smartfactory's perfect scores are a property of its corruption, not of the
   detector.** All of its corruption is missing values, which the trivial
   missing-token rule finds exactly; a 1.0000 there is the dataset being easy.
6. **`conditional_marginal`'s role is as a control, not a candidate.** It is in
   every table so that `conditional_full`'s margin over it is visible.
7. **Costs, decisions and repair quality are out of scope.** This file is only
   about detection metrics. The decision layer's numbers are in
   `results/RESULTS.md`, and the cost defensibility question is gap G2
   (route W4.3).
8. **A false positive here is a cell whose dirty value equals its clean value.**
   On adult the conditional model's top flags are rare-but-clean
   `native_country` values. The model is *right* that they are surprising; they
   are simply not corrupted. "Surprising" and "corrupted" are different
   targets, and only the second one is the task.
"""


def write_report(
    results: List[Dict[str, Any]], out_path: Path, seconds: float, n_splits: int, seed: int
) -> None:
    lines: List[str] = [PREAMBLE, ""]
    lines += [
        f"- splits: {n_splits} stratified cell splits, seed {seed}, 30 % eval each",
        f"- total runtime {seconds:.1f}s",
        "",
    ]

    lines += ["## 1. Headline table", ""]
    for result in results:
        stats = result["label_stats"]
        lines += [
            f"### {result['dataset']}",
            "",
            f"{stats['n_cells']:,} cells, {stats['n_corrupted']:,} corrupted, "
            f"prevalence {float(stats['prevalence']):.6f} "
            f"({100.0 * float(stats['prevalence']):.3f} %), "
            f"{stats['n_rows']:,} rows x {stats['n_columns']} columns",
            "",
        ]
        lines += _headline_table(result)
        lines.append("")
        if result["notes"]:
            lines += ["Feature-build notes:", ""]
            lines += [f"- {n}" for n in result["notes"]]
            lines.append("")
    lines += [
        "`P@k` and `R@k` use k = number of corrupted cells, so they are equal by "
        "construction (both are TP/k with k = P); both are printed because both "
        "names appear in the literature. They are the R-precision.",
        "",
    ]

    lines += ["## 2. Precision-recall curves", ""]
    for result in results:
        rel = Path("figures") / "w4" / result["figure"].name
        stats = result["label_stats"]
        lines += [
            f"### {result['dataset']}",
            "",
            f"![PR curves for {result['dataset']}]({rel.as_posix()})",
            "",
            f"All {stats['n_cells']:,} cells, all four detectors overlaid. The "
            f"dashed line is precision = prevalence = "
            f"{float(stats['prevalence']):.6f}, the no-skill baseline; a curve "
            "on that line carries no information and area below it is worse than "
            "chance. The right panel is the same data with precision on a log "
            "axis, because at adult's prevalence every curve otherwise lies on "
            "the x-axis. Curves are drawn with at most "
            f"{MAX_CURVE_POINTS:,} points (stride-subsampled for file size); "
            "every AP in the legend is computed on the full curve and is "
            "asserted equal to the library's `pr_auc` (§6).",
            "",
        ]

    lines += [
        "## 3. Per-column precision and recall",
        "",
        "The reviewer's specific ask. `@oracle` = at the detector's own best-F1 "
        "threshold (label-chosen, an upper bound). `@budget` = flagging the top "
        "k cells of the whole-table ranking, k = number of corrupted cells, "
        "ties broken by a seeded permutation. `PR-AUC within column` ranks only "
        "that column's cells and is `—` where the column has no corruption.",
        "",
    ]
    for result in results:
        lines += [f"### {result['dataset']}", ""]
        lines += _per_column_tables(result)

    lines += [
        "## 4. False-positive accounting",
        "",
        "A false positive is a flagged cell whose dirty value **equals** its "
        "clean value. Counts are over all cells.",
        "",
    ]
    for result in results:
        lines += [f"### {result['dataset']}", ""]
        lines += _fp_tables(result)
        lines.append("")

    lines += [
        "## 5. Operating points",
        "",
        "The trade-off a single oracle number hides. Rows are chosen by flag "
        "budget (fractions and multiples of the number of corrupted cells), "
        "plus the extremes and the oracle row, so the ladder cannot be picked "
        "to flatter a detector. `flag iff score >= threshold`. Watch the "
        "`accuracy` column barely move.",
        "",
    ]
    for result in results:
        lines += [f"### {result['dataset']}", ""]
        lines += _operating_tables(result)

    lines += ["## 6. Self-checks that ran", ""]
    for result in results:
        for check in result["checks"]:
            lines.append(f"- {check}")
    lines += [
        "",
        "Each of these is an assertion in `experiments/w4_metrics.py`; the run "
        "aborts rather than publishing a table that failed one.",
        "",
    ]

    lines += [NOT_SHOWN]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main(argv: Optional[Sequence[str]] = None) -> int:
    enable_unicode_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--data-root", default=str(REPO_ROOT / "datasets"))
    parser.add_argument("--splits", type=int, default=N_SPLITS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=str(OUTPUT))
    parser.add_argument("--figure-dir", default=str(FIGURE_DIR))
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the adapter-identity check (saves one feature build)",
    )
    args = parser.parse_args(argv)

    started = time.time()
    results: List[Dict[str, Any]] = []
    for dataset in args.datasets:
        print(f"\n=== {dataset}")
        results.append(
            run_dataset(
                dataset,
                Path(args.data_root),
                n_splits=args.splits,
                seed=args.seed,
                # The identity check needs a second feature build; it is only
                # affordable on breast_cancer, and the code path is identical
                # for the other datasets.
                verify=(not args.no_verify) and dataset == "breast_cancer",
                figure_dir=Path(args.figure_dir),
            )
        )

    seconds = time.time() - started
    write_report(results, Path(args.out), seconds, args.splits, args.seed)
    print(f"\nWrote {args.out} ({seconds:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
