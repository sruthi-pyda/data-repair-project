"""
src/detection_eval/metrics.py
-----------------------------
Detection metrics done correctly for extreme class imbalance.

WHY NOT ACCURACY, WHY NOT ROC-AUC
=================================
Adult's corruption prevalence is 1,701 / 678,330 = 0.251 %. A detector that
flags nothing is 99.749 % accurate. ROC-AUC is nearly as bad: its x-axis is
the false-positive *rate*, and with 676,629 negatives you can raise 6,000
false alarms and move FPR by less than one percentage point. ROC-AUC therefore
looks respectable for a detector whose flags are 97 % wrong.

PR-AUC (average precision) uses precision, whose denominator is the number of
*flags*, so it is sensitive to exactly the thing that matters operationally:
of the cells you send to a human, how many were really broken.

    A random scorer has PR-AUC ≈ prevalence and ROC-AUC = 0.5.

So PR-AUC must always be read against the prevalence printed beside it. 0.08
is a strong score on adult (≈ 32× random) and a terrible one on
breast_cancer (prevalence 0.0752).

UNDEFINED IS NOT ZERO
=====================
Every function here returns ``None`` when the quantity is mathematically
undefined, never ``0.0``. ``0.0`` in a results table reads as a measured
score — CLAUDE.md §5.4 records that exact mistake being made once already
(a not-fitted calibrator printing ``Brier 0.0000`` and looking perfect).
Undefined cases:

    no positives          → PR-AUC, recall@k, best-F1, lift@k undefined
    single class          → ROC-AUC undefined
    k = 0                 → precision@k undefined (no flags to be right about)
    prevalence = 0        → lift@k undefined (division by zero)

TWO IMPLEMENTATIONS, NOT ONE
============================
Average precision is computed here with scikit-learn's
``average_precision_score``. ``tests/test_detection_eval.py`` re-derives it
from the definition independently and asserts agreement on random and
adversarial inputs (all-tied scores, two-value scores, duplicated extremes).
A single implementation of the primary metric of the whole research program
is not something to trust.

TIES ARE THE HARD PART HERE
===========================
The existing rule detector emits a handful of distinct scores (1.0, 0.9,
0.85, 0.7, 0.35, 0.3) plus 0.0 for every unflagged cell, so hundreds of
thousands of cells are tied. Tie handling is then not a detail, it is most of
the number:

- ``pr_auc`` / ``roc_auc``: sklearn groups tied scores into a single
  threshold, which is the correct (and pessimistic-in-the-right-way)
  treatment.
- ``precision_at_k`` / ``recall_at_k``: if the k-th cell lands inside a tie
  group, the answer depends on how the tie is broken. Breaking by array
  order would be catastrophic here, because cells are enumerated
  column-major — index order correlates with the label. So these functions
  return the **expectation under uniformly random tie-breaking**, which is
  order-independent, deterministic, and exactly the number you would
  converge to by shuffling. See :func:`precision_at_k`.

Worked example
==============
    >>> y = [0, 0, 1, 1]
    >>> s = [0.1, 0.2, 0.3, 0.4]
    >>> round(pr_auc(y, s), 6)
    1.0
    >>> m = evaluate_scores(y, s)
    >>> m.n_pos, m.prevalence, m.precision_at_k
    (2, 0.5, 1.0)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)

__all__ = [
    "DetectionMetrics",
    "best_f1",
    "evaluate_scores",
    "f1_at_threshold",
    "lift_at_k",
    "pr_auc",
    "precision_at_k",
    "recall_at_k",
    "roc_auc",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Input handling
# ═══════════════════════════════════════════════════════════════════════════════


def _prepare(y_true: Sequence[Any], scores: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coerce to ``(bool array, float array)`` and reject unusable input.

    Raises
    ------
    ValueError
        Lengths differ; ``scores`` contains NaN or inf; ``y_true`` holds a
        value other than 0/1/False/True.

    Notes
    -----
    NaN scores raise rather than being imputed. A NaN is a bug in the
    detector — silently treating it as "least suspicious" (the effect of
    ``nan_to_num``) would quietly lower the score of a broken method and the
    comparison would be meaningless.
    """
    y = np.asarray(list(y_true))
    s = np.asarray(list(scores), dtype=float)

    if y.shape[0] != s.shape[0]:
        raise ValueError(f"Length mismatch: y_true={y.shape[0]}, scores={s.shape[0]}")
    if y.ndim != 1 or s.ndim != 1:
        raise ValueError("y_true and scores must both be 1-D.")
    if s.size and not np.all(np.isfinite(s)):
        n_bad = int((~np.isfinite(s)).sum())
        raise ValueError(
            f"scores contain {n_bad} non-finite value(s) (NaN or inf). "
            "A detector must emit a finite score for every cell; NaN is not "
            "imputed here because that would hide the defect."
        )

    if y.size == 0:
        return np.zeros(0, dtype=bool), s

    y_num = np.asarray(y, dtype=float)
    if not np.all(np.isin(y_num, (0.0, 1.0))):
        bad = sorted({float(v) for v in y_num if v not in (0.0, 1.0)})
        raise ValueError(f"y_true must be binary 0/1; found {bad[:5]}")
    return y_num.astype(bool), s


# ═══════════════════════════════════════════════════════════════════════════════
# Threshold-free ranking metrics
# ═══════════════════════════════════════════════════════════════════════════════


def pr_auc(y_true: Sequence[Any], scores: Sequence[Any]) -> Optional[float]:
    """
    Average precision — the primary metric of the research program.

    Returns
    -------
    float or None
        ``None`` when there is no positive (average precision is a mean of
        precisions taken at each positive; with none, there is nothing to
        average) or when the input is empty. Never ``0.0`` for "undefined".

    Notes
    -----
    All-positive input returns ``1.0``: precision is 1 at every threshold,
    which is the honest answer, not a degenerate one.
    """
    y, s = _prepare(y_true, scores)
    if y.size == 0 or not y.any():
        return None
    return float(average_precision_score(y.astype(int), s))


def roc_auc(y_true: Sequence[Any], scores: Sequence[Any]) -> Optional[float]:
    """
    ROC-AUC, reported for comparability with published systems only.

    Returns
    -------
    float or None
        ``None`` when ``y_true`` has a single class (no ROC curve exists) or
        the input is empty.

    Notes
    -----
    Read §"WHY NOT ACCURACY" above before quoting this number. At 0.25 %
    prevalence ROC-AUC is not a measure of usefulness.
    """
    y, s = _prepare(y_true, scores)
    if y.size == 0 or y.all() or not y.any():
        return None
    return float(roc_auc_score(y.astype(int), s))


# ═══════════════════════════════════════════════════════════════════════════════
# Threshold metrics
# ═══════════════════════════════════════════════════════════════════════════════


def f1_at_threshold(
    y_true: Sequence[Any], scores: Sequence[Any], threshold: float
) -> Optional[float]:
    """
    F1 of the rule "flag the cell iff ``score >= threshold``".

    Returns
    -------
    float or None
        ``None`` when there is no positive label (recall undefined).
        ``0.0`` is returned — legitimately — when positives exist but the
        rule flags none of them, or flags nothing at all: that is a measured
        failure, not an undefined quantity.
    """
    y, s = _prepare(y_true, scores)
    if y.size == 0 or not y.any():
        return None
    pred = s >= float(threshold)
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return float(2.0 * precision * recall / (precision + recall))


def best_f1(
    y_true: Sequence[Any], scores: Sequence[Any]
) -> Tuple[Optional[float], Optional[float]]:
    """
    Best achievable F1 over every threshold, and the threshold achieving it.

    Candidate thresholds are the distinct score values: ``score >= t`` can
    only change as ``t`` crosses one of them, so sweeping the distinct values
    is exhaustive, not a sample.

    Returns
    -------
    (float, float) or (None, None)
        ``(None, None)`` when there is no positive label or the input is
        empty. Ties in F1 resolve to the **highest** threshold, i.e. the most
        conservative detector that attains the maximum.

    Notes
    -----
    This is an oracle number: the threshold is chosen knowing the labels, so
    it is an upper bound on what a deployed detector could reach, and it is
    comparable across methods only because every method is given the same
    advantage. Never present best-F1 as operating-point performance.
    """
    y, s = _prepare(y_true, scores)
    if y.size == 0 or not y.any():
        return None, None

    n_pos = int(y.sum())
    order = np.argsort(-s, kind="stable")
    y_sorted = y[order]
    s_sorted = s[order]

    tp_cum = np.cumsum(y_sorted.astype(np.int64))
    n_cum = np.arange(1, y.size + 1, dtype=np.int64)

    # Only prefix lengths that sit at the end of a tie group are realisable:
    # `score >= t` cannot split a group of equal scores.
    is_group_end = np.ones(y.size, dtype=bool)
    is_group_end[:-1] = s_sorted[:-1] != s_sorted[1:]
    idx = np.flatnonzero(is_group_end)

    tp = tp_cum[idx]
    flagged = n_cum[idx]
    # f1 = 2TP / (flagged + n_pos); identical to 2PR/(P+R), one division.
    f1 = 2.0 * tp / (flagged + n_pos)

    best = int(np.argmax(f1))
    return float(f1[best]), float(s_sorted[idx[best]])


# ═══════════════════════════════════════════════════════════════════════════════
# Top-k metrics (the operational view: "we can review k cells")
# ═══════════════════════════════════════════════════════════════════════════════


def _expected_tp_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    """
    Expected true positives in the top ``k`` under random tie-breaking.

    Whole tie groups above the cut contribute all their positives. The group
    straddling the cut contributes its positives in proportion to how much of
    it fits — which is exactly the expectation when the group is shuffled.
    """
    order = np.argsort(-s, kind="stable")
    y_sorted = y[order]
    s_sorted = s[order]

    # Group boundaries over equal scores.
    starts = np.flatnonzero(
        np.concatenate(([True], s_sorted[1:] != s_sorted[:-1]))
    )
    ends = np.concatenate((starts[1:], [y.size]))

    expected = 0.0
    remaining = k
    for start, end in zip(starts, ends):
        size = int(end - start)
        pos = int(y_sorted[start:end].sum())
        if remaining >= size:
            expected += pos
            remaining -= size
            if remaining == 0:
                break
        else:
            expected += pos * (remaining / size)
            remaining = 0
            break
    return float(expected)


def _check_k(k: int, n: int) -> int:
    """Validate ``k`` and clamp it to ``n`` (you cannot flag more cells than exist)."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError(f"k must be an int, got {k!r} ({type(k).__name__})")
    k = int(k)
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if k > n:
        logger.warning("k=%d exceeds n=%d; clamping to %d.", k, n, n)
        return n
    return k


def precision_at_k(
    y_true: Sequence[Any], scores: Sequence[Any], k: int
) -> Optional[float]:
    """
    Fraction of the top-``k`` scored cells that are genuinely corrupted.

    The operational metric: "we have budget to review k cells, what share of
    that budget is well spent?" With ``k = n_pos`` it is the R-precision, and
    is directly comparable across methods on one dataset.

    Tie handling is the **expectation under uniformly random tie-breaking**
    (see the module docstring): if the cut falls inside a group of equal
    scores, that group contributes its positives pro rata. For the rule
    detector, whose unflagged cells all score exactly 0.0, this is the
    difference between a meaningful number and an artefact of column order.

    Parameters
    ----------
    k : int
        ``0`` → ``None`` (no flags, so no precision). ``k > n`` is clamped to
        ``n`` with a warning: you cannot flag cells that do not exist, and
        silently returning a smaller denominator would flatter the method.

    Returns
    -------
    float or None
    """
    y, s = _prepare(y_true, scores)
    k = _check_k(k, y.size)
    if k == 0:
        return None
    return float(_expected_tp_at_k(y, s, k) / k)


def recall_at_k(
    y_true: Sequence[Any], scores: Sequence[Any], k: int
) -> Optional[float]:
    """
    Fraction of all corrupted cells that appear in the top-``k``.

    Same tie treatment as :func:`precision_at_k`.

    Returns
    -------
    float or None
        ``None`` when there is no positive label (nothing to recall) or
        ``k == 0``.
    """
    y, s = _prepare(y_true, scores)
    k = _check_k(k, y.size)
    n_pos = int(y.sum())
    if n_pos == 0 or k == 0:
        return None
    return float(_expected_tp_at_k(y, s, k) / n_pos)


def lift_at_k(
    y_true: Sequence[Any], scores: Sequence[Any], k: int
) -> Optional[float]:
    """
    precision@k divided by prevalence — how many times better than random.

    The scale-free companion to precision@k: a precision@k of 0.08 means
    nothing until you know whether prevalence is 0.0025 (lift 32) or 0.075
    (lift 1.07, i.e. no signal).

    Returns
    -------
    float or None
        ``None`` when precision@k is undefined or prevalence is 0.
    """
    y, s = _prepare(y_true, scores)
    k = _check_k(k, y.size)
    if y.size == 0 or k == 0:
        return None
    prevalence = float(y.mean())
    if prevalence == 0.0:
        return None
    p_at_k = _expected_tp_at_k(y, s, k) / k
    return float(p_at_k / prevalence)


# ═══════════════════════════════════════════════════════════════════════════════
# Bundle
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DetectionMetrics:
    """
    Every detection metric for one (method, dataset, split), plus the context
    needed to read them.

    ``prevalence`` and ``n_pos`` are not decoration: PR-AUC is uninterpretable
    without the prevalence it must be compared against, and ``k`` records the
    review budget that precision@k assumes.

    Attributes
    ----------
    n : int
        Cells evaluated.
    n_pos : int
        Genuinely corrupted cells among them.
    prevalence : float
        ``n_pos / n``; ``0.0`` for an empty input.
    pr_auc, roc_auc : float or None
        See :func:`pr_auc`, :func:`roc_auc`. ``None`` = undefined.
    best_f1, best_f1_threshold : float or None
        Oracle-thresholded F1 and the threshold attaining it.
    k : int
        Review budget used for the @k metrics. Defaults to ``n_pos``.
    precision_at_k, recall_at_k, lift_at_k : float or None
    """

    n: int
    n_pos: int
    prevalence: float
    pr_auc: Optional[float]
    roc_auc: Optional[float]
    best_f1: Optional[float]
    best_f1_threshold: Optional[float]
    k: int
    precision_at_k: Optional[float]
    recall_at_k: Optional[float]
    lift_at_k: Optional[float]

    # Metric fields a leaderboard aggregates over splits, in display order.
    SCORE_FIELDS = (
        "pr_auc",
        "roc_auc",
        "best_f1",
        "precision_at_k",
        "recall_at_k",
        "lift_at_k",
    )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable view; ``None`` stays ``None`` (serialises to null)."""
        return {
            "n": int(self.n),
            "n_pos": int(self.n_pos),
            "prevalence": float(self.prevalence),
            "pr_auc": _opt(self.pr_auc),
            "roc_auc": _opt(self.roc_auc),
            "best_f1": _opt(self.best_f1),
            "best_f1_threshold": _opt(self.best_f1_threshold),
            "k": int(self.k),
            "precision_at_k": _opt(self.precision_at_k),
            "recall_at_k": _opt(self.recall_at_k),
            "lift_at_k": _opt(self.lift_at_k),
        }


def _opt(value: Optional[float]) -> Optional[float]:
    return None if value is None else float(value)


def evaluate_scores(
    y_true: Sequence[Any],
    scores: Sequence[Any],
    k: Optional[int] = None,
) -> DetectionMetrics:
    """
    Compute every metric in one pass over one (labels, scores) pair.

    Parameters
    ----------
    y_true : sequence of 0/1
        Cell corruption labels — from :func:`src.detection_eval.labels.cell_labels`,
        flattened in the same order as ``scores``.
    scores : sequence of float
        Detector scores, higher = more likely corrupted. Must be finite.
    k : int, optional
        Review budget for the @k metrics. Default ``n_pos``, which makes
        precision@k the R-precision and the most comparable single operating
        point across methods.

    Returns
    -------
    DetectionMetrics

    Raises
    ------
    ValueError
        Non-finite scores, length mismatch, non-binary labels, negative ``k``.
    """
    y, s = _prepare(y_true, scores)
    n = int(y.size)
    n_pos = int(y.sum())
    k_eff = _check_k(n_pos if k is None else k, n)
    bf1, bthr = best_f1(y, s)

    return DetectionMetrics(
        n=n,
        n_pos=n_pos,
        prevalence=(n_pos / n) if n else 0.0,
        pr_auc=pr_auc(y, s),
        roc_auc=roc_auc(y, s),
        best_f1=bf1,
        best_f1_threshold=bthr,
        k=k_eff,
        precision_at_k=precision_at_k(y, s, k_eff),
        recall_at_k=recall_at_k(y, s, k_eff),
        lift_at_k=lift_at_k(y, s, k_eff),
    )
