"""
src/decision/threshold.py
-------------------------
Choosing an operating point from a **tiny label budget**.

WHY THIS EXISTS
===============
`results/w9c_raha_rematch.md` decomposed our three losses to Raha and found
one of them is not a detection failure at all:

    dataset    ours @20 labels    our ORACLE bound    Raha @20 labels
    hospital        0.3849             0.5054              0.4000

On hospital the detector is **already good enough to beat Raha** — our oracle
F1 exceeds theirs. We simply cannot *find* the threshold with 20 labels.

The reason is where the budget is spent, not its size. At hospital's
prevalence of 0.0255, twenty **uniformly** sampled cells contain 0.5
positives in expectation, so most draws hold zero or one positive and a
threshold fitted on them is noise. Roughly nineteen of the twenty labels go
to confirming that low-scoring cells are clean, which we already believed.

THE FIX
=======
Build **score strata** from the scores alone (no labels consumed), then
allocate the budget across strata and estimate precision inside each:

    top 0.1% | 0.1-1% | 1-5% | 5-20% | 20-100%

    precision_i = (positives found + 0.5) / (sampled + 1)   Laplace-smoothed
    positives_i = n_i * precision_i

A candidate threshold is a stratum boundary, and for the boundary after
stratum `j`:

    flagged = sum n_i (i <= j)
    TP_est  = sum positives_i (i <= j)
    FN_est  = total_estimated_positives - TP_est

giving an estimated F1, or an estimated cost. Two rules are provided because
they optimise different things:

- ``stratified_f1``   — maximise estimated F1. Comparable with Raha, which
  reports F1.
- ``stratified_cost`` — minimise ``flagged*C_review + FN_est*C_miss``. This is
  the project's own objective and the honest one for a system whose purpose
  is decisions rather than rankings.

A precision estimate per stratum is a handful of low-variance scalars, and
the strata where positives actually live get sampled deliberately rather than
by luck.

BUDGET HONESTY
==============
Every selector reveals **at most** ``budget`` labels, returns the exact count
in :class:`ThresholdChoice.n_revealed`, and takes labels only through the
``reveal`` callback so the accounting cannot be bypassed. Strata are built
from scores alone.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Stratum edges as cumulative fractions of the score-sorted population,
#: descending. Geometric rather than uniform because corruption concentrates
#: at the top of the ranking, so the interesting boundaries are all there.
DEFAULT_STRATA: Tuple[float, ...] = (0.001, 0.01, 0.05, 0.20, 1.00)

#: Laplace smoothing for a per-stratum precision estimate. 0.5 / 1.0 is the
#: Jeffreys prior, so a stratum with zero sampled positives gets a small
#: positive estimate instead of exactly zero.
SMOOTH_NUM = 0.5
SMOOTH_DEN = 1.0

METHODS = ("uniform_f1", "stratified_f1", "stratified_cost", "oracle_f1")


@dataclasses.dataclass
class ThresholdChoice:
    """What a selector decided, and what it spent to decide it."""

    method: str
    threshold: Optional[float]
    n_revealed: int
    budget: int
    note: str = ""
    #: Per-stratum diagnostics, for the report. Empty for non-stratified
    #: methods.
    strata: Tuple[Dict[str, float], ...] = ()

    def to_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


def _f1(tp: float, flagged: float, total_pos: float) -> float:
    """F1 from estimated counts; 0.0 when undefined rather than NaN."""
    if flagged <= 0 or total_pos <= 0:
        return 0.0
    p = tp / flagged
    r = tp / total_pos
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def _stratum_bounds(
    n: int, strata: Sequence[float]
) -> List[Tuple[int, int]]:
    """Half-open index ranges over the score-sorted array, top first."""
    out: List[Tuple[int, int]] = []
    start = 0
    for frac in strata:
        end = min(n, max(start + 1, int(round(frac * n))))
        if end <= start:
            continue
        out.append((start, end))
        start = end
        if start >= n:
            break
    if start < n:
        out.append((start, n))
    return out


def _allocate(budget: int, k: int) -> List[int]:
    """Spread `budget` as evenly as possible over `k` strata."""
    if k <= 0:
        return []
    base, extra = divmod(budget, k)
    return [base + (1 if i < extra else 0) for i in range(k)]


def select_threshold(
    scores: np.ndarray,
    reveal: Callable[[np.ndarray], np.ndarray],
    method: str = "stratified_f1",
    budget: int = 20,
    seed: int = 42,
    c_review: float = 1.0,
    c_miss: float = 5.0,
    strata: Sequence[float] = DEFAULT_STRATA,
) -> ThresholdChoice:
    """
    Pick a score threshold using at most ``budget`` revealed labels.

    Parameters
    ----------
    scores : np.ndarray
        Per-cell detector scores. Higher = more suspicious.
    reveal : callable
        ``reveal(indices) -> 0/1 labels``. The **only** way labels enter, so
        the budget accounting cannot be bypassed.
    method : str
        One of :data:`METHODS`. ``oracle_f1`` reveals *everything* and is an
        upper bound, never an operating point.
    budget : int
        Labels the deployable methods may reveal.
    c_review, c_miss : float
        Cost of a flagged cell and of a missed corruption, for
        ``stratified_cost``.

    Returns
    -------
    ThresholdChoice
        ``threshold`` is ``None`` when the budget could not support a choice
        (e.g. no positive was found anywhere) -- reported, never silently
        patched into a guess.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or s.size == 0:
        raise ValueError(f"scores must be a non-empty 1-D array, got {s.shape}")

    order = np.argsort(-s, kind="mergesort")  # descending, ties stable

    # ── oracle: reveal everything. UPPER BOUND, not deployable ─────────
    if method == "oracle_f1":
        y = np.asarray(reveal(np.arange(s.size)), dtype=int)
        n_pos = int(y.sum())
        if n_pos == 0:
            return ThresholdChoice(method, None, int(s.size), budget,
                                   "no positive cell exists")
        ys = y[order]
        tp = np.cumsum(ys)
        k = np.arange(1, s.size + 1)
        f1 = np.where((k > 0) & (n_pos > 0),
                      2.0 * tp / (k + n_pos), 0.0)
        best = int(np.argmax(f1))
        return ThresholdChoice(method, float(s[order[best]]), int(s.size),
                               budget, "oracle upper bound")

    # ── uniform: the failing baseline, kept for comparison ─────────────
    if method == "uniform_f1":
        rng = np.random.default_rng(seed)
        take = min(budget, s.size)
        idx = rng.choice(s.size, size=take, replace=False)
        y = np.asarray(reveal(idx), dtype=int)
        if y.sum() == 0:
            return ThresholdChoice(
                method, None, int(take), budget,
                "no positive in the uniform sample; cannot fit a threshold")
        best_t, best_f = None, -1.0
        n_pos_s = int(y.sum())
        for t in np.unique(s[idx]):
            flagged = s[idx] >= t
            f = _f1(float(np.sum(flagged & (y == 1))),
                    float(flagged.sum()), float(n_pos_s))
            if f > best_f:
                best_f, best_t = f, float(t)
        return ThresholdChoice(method, best_t, int(take), budget)

    # ── stratified: strata from SCORES ONLY, then spend inside them ────
    bounds = _stratum_bounds(s.size, strata)
    alloc = _allocate(budget, len(bounds))
    rng = np.random.default_rng(seed)

    revealed = 0
    info: List[Dict[str, float]] = []
    for (lo, hi), b in zip(bounds, alloc):
        size = hi - lo
        take = int(min(b, size))
        if take <= 0:
            info.append({"lo": lo, "hi": hi, "size": size, "sampled": 0,
                         "found": 0, "precision": 0.0})
            continue
        pick = order[lo + rng.choice(size, size=take, replace=False)]
        y = np.asarray(reveal(pick), dtype=int)
        revealed += take
        found = int(y.sum())
        prec = (found + SMOOTH_NUM) / (take + SMOOTH_DEN)
        info.append({"lo": lo, "hi": hi, "size": size, "sampled": take,
                     "found": found, "precision": prec})

    est_pos = [d["size"] * d["precision"] for d in info]
    total_pos = float(sum(est_pos))
    if total_pos <= 0:
        return ThresholdChoice(method, None, revealed, budget,
                               "no estimated positives", tuple(info))

    best_j, best_val = None, None
    for j in range(len(info)):
        flagged = float(sum(d["size"] for d in info[: j + 1]))
        tp = float(sum(est_pos[: j + 1]))
        if method == "stratified_f1":
            val = _f1(tp, flagged, total_pos)
            better = best_val is None or val > best_val
        else:  # stratified_cost
            val = flagged * c_review + max(total_pos - tp, 0.0) * c_miss
            better = best_val is None or val < best_val
        if better:
            best_val, best_j = val, j

    if best_j is None:
        return ThresholdChoice(method, None, revealed, budget,
                               "no boundary chosen", tuple(info))
    # Threshold = the score at the last included position.
    cut = int(info[best_j]["hi"]) - 1
    return ThresholdChoice(method, float(s[order[cut]]), revealed, budget,
                           "", tuple(info))


def apply_threshold(
    scores: np.ndarray, choice: ThresholdChoice
) -> np.ndarray:
    """
    Boolean flags for a choice. An unmade choice flags **nothing**, which is
    the conservative reading and keeps a failed selection from looking like a
    detector that flags everything.
    """
    s = np.asarray(scores, dtype=float)
    if choice.threshold is None:
        return np.zeros(s.size, dtype=bool)
    return s >= choice.threshold
