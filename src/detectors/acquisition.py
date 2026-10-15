"""
src/detectors/acquisition.py
----------------------------
Label-acquisition strategies for error detection at extreme class imbalance
(wave-3 route W3.1 of ``docs/research_program.md``).

WHY THIS EXISTS — THE MEASURED PROBLEM, NOT A HUNCH
===================================================
Wave 2 (``results/w2_ensemble.md``, research-program §6) measured the
label-efficiency curve of the signal ensemble and found one number that does
not fit the literature::

    adult, PR-AUC    cond_only 0.0110 | b20 0.0106 | b100 0.2656 | full 0.7024

Three facts follow from that row, all measured:

1. **Labels are what wins on adult.** Label-free fusion of the two detector
   families is *worse* than the conditional model alone (0.0106 vs 0.0110).
   The 0.0110 → 0.7024 move is bought entirely with supervision.
2. **Acquiring those labels is broken at 0.251 % prevalence.** At k <= 20 the
   sampler draws **zero positives in 5 of 5 splits**, so the budget buys
   literally nothing and the detector falls back to the unsupervised
   combiner. Raha's published 2 → 20 label jump (0.40 → 0.85 F1) is measured
   at 5–40 % error rates and does **not** transfer here; that negative result
   is already ours (research-program §6, finding 3).
3. **The bottleneck is the sampler, not the learner.** b100 scores 0.2656
   against full's 0.7024 — a 2.6x gap at 100 labels, with the *same* feature
   matrix and the *same* logistic learner on both sides. Only the set of
   labelled cells differs.

And the obvious strategy is the one that fails hardest: ``top_combiner``
draws zero positives at **every** budget up to 100 on adult, because the top
of the global ranking is the rule detector's ``fnlwgt`` IQR false positives
plus the blank ``workclass`` / ``occupation`` cells — every one of which is
guaranteed clean, since adult's corruption is confined to ``income``. The
rules are not merely uninformative there, they are *actively poisoning* the
ranking the sampler trusts: measured ROC-AUC 0.4957 for the rule score
(anti-correlated with truth) against 0.8875 for the conditional score.

So the constructive question this module exists to answer is:

    Given a budget of k labels and nothing but the detector's own scores,
    which cells should we pay to label — and how much of the b100 → full gap
    can a better choice close?

The deliverable is **"labels needed to reach X % of full-supervision
PR-AUC"**, per strategy, plus the mechanism-level number that explains any
win: **positives found per query**.

WHAT A STRATEGY MAY LOOK AT (CLAUDE.md §3, research-program §3 rule 1)
======================================================================
A strategy may read:

- the per-cell feature matrix (:func:`src.detectors.ensemble.build_features`),
  which is built from the dirty table alone;
- the detector's own current scores — the unsupervised combiner, or the
  meta-classifier fitted on labels **already paid for**;
- the labels it has already queried.

A strategy may **never** read a label it has not queried. That is enforced
structurally, not by convention:

- :func:`select_cells` and every strategy function take **no label array of
  any kind**. There is nowhere to put one. ``tests/test_acquisition.py``
  scans the signatures.
- Every label arrives through :class:`src.detectors.ensemble.LabelOracle`,
  which counts and records each query and refuses to exceed its cap, so the
  budget is auditable rather than promised.
- :class:`AcquisitionDetector` samples only from cells inside the harness's
  ``train_mask``; eval-split labels are never requested, so they cannot be
  received. The test suite spies on the oracle and asserts it, and the
  counterfactual test inverts every unqueried train label and asserts the
  resulting scores are **bitwise identical**.

THE STRATEGIES (:data:`STRATEGIES`, all spending the same k)
============================================================
Every strategy below spends **100 % of k by its own rule**. Wave 2's sampler
spent half the budget informedly and half uniformly at random
(``top_fraction=0.5``); here the mixture is removed on purpose, so that the
comparison is of *selection rules* and not of mixtures. One consequence must
be stated whenever these numbers sit beside wave 2's: ``per_column_quota`` at
k here is not bit-identical to wave 2's ``per_column`` at the same k, because
wave 2 gave only ceil(k/2) cells to the informed rule. ``random`` is the
control that covers the uniform arm.

``random``
    Uniform without replacement over train cells. **The control.** Its
    expected positives-per-query is the prevalence, which is the number every
    other strategy has to beat to have done anything at all.
``global_top``
    Top-k by the unsupervised combiner (noisy-OR of rule and conditional
    score). Wave 2 measured this drawing zero positives on adult at every
    k <= 100; it is included so the failure is in the table rather than in a
    footnote.
``per_column_quota``
    Wave 2's per-column sampler and the current default: the budget is
    allocated round-robin across columns, each column contributing its
    next-best eligible cell by within-column combiner rank. Comparing scores
    *within* a column removes the cross-column scale problem that makes a
    global ranking unreliable. Cells both signals call "missing" are excluded
    from the pool, inherited from wave 2 — see
    :meth:`AcquisitionDetector._eligible_pool`.
``uncertainty``
    Classic active learning: query the cells whose current model probability
    is nearest 0.5. **Cold-start caveat, stated because it governs the
    result:** at round 1 no model exists, so there is no probability to be
    uncertain about. The implementation then falls back to the unsupervised
    combiner read as a pseudo-probability, which means *one-shot*
    ``uncertainty`` measures the combiner's mid-range and not uncertainty
    sampling at all. Uncertainty sampling can only earn its keep in
    ``n_rounds > 1``, which is exactly why this module implements sequential
    acquisition.
``conditional_top``
    Top-k by the conditional model's score alone, ignoring the rules
    entirely. The rationale is our own measurement, not an intuition: on
    adult the conditional signal is the only one correlated with truth
    (ROC-AUC 0.8875 against the rules' 0.4957), so including the rule score
    in the ranking the sampler trusts is strictly harmful there.

    **It still fails on adult, and the reason is instructive.** Measured
    here: 0 positives in 500 queries. ``c_full`` is
    ``1 - P(value | rest of row)``, and a *blank* cell is not modelled at
    all — it is assigned ``ConditionalDetector.nan_score``, which is 1.0 by
    policy. Adult has ~6,500 blank ``workclass`` / ``occupation`` cells, all
    of them clean, so the top of a ``c_full`` ranking is a block of
    guaranteed negatives before any modelled cell is reached. The strategy's
    own rationale is about *modelled* cells; a policy constant of 1.0 is not
    evidence of corruption.
``cond_top_live``
    ``conditional_top`` restricted to cells the signals do **not** both call
    missing. One factor changed against the row above, so the pair isolates
    the cost of that NaN policy constant on acquisition. The exclusion reads
    no ground truth — only ``r_missing`` and the conditional model's own NaN
    score — so it is free.
``stratified_by_score``
    Bin cells by combiner score and sample uniformly within bins, round-robin
    across bins, so the budget covers the score *range* instead of piling on
    one end. Bins are **equal-width** over the observed range; quantile bins
    collapse at this skew (on adult ~99 % of cells share the score 0.0, so
    the 0.1…0.9 quantiles are all 0.0 and nine of ten edges coincide).
    Empty bins are skipped.
``batch_diverse``
    Top-scoring but de-duplicated: walk the global ranking and accept a cell
    only while its column and its row still have quota (at most
    ``ceil(k / n_columns)`` per column, at most one per row), so a batch
    cannot spend itself inside one column. This isolates exactly one factor
    against ``global_top`` — de-duplication — because it applies no other
    exclusion.
``cond_per_column``
    **This module's own strategy**, and the reasoning is a composition of two
    measurements rather than a new idea. Wave 2 established (a) that
    per-column allocation is the only sampler that learns anything on adult,
    and (b) that the conditional score is the only signal correlated with
    truth there. ``per_column_quota`` ranks within a column by the combiner,
    which still mixes in the anti-correlated rule score;
    ``conditional_top`` uses the right signal but on a global ranking. This
    strategy takes the per-column allocation and ranks within each column by
    ``c_full`` alone. If the composition is right it should dominate both
    parents on adult; if it does not, that is reported.

ONE-SHOT VS SEQUENTIAL
======================
``n_rounds=1`` spends the whole budget before any label is seen.
``n_rounds=R`` splits it into R rounds of ``k/R`` (remainder to the earlier
rounds), **refitting the meta-classifier between rounds** so later queries
are guided by a model trained on the labels already bought. Already-labelled
cells are removed from the pool, so no label is ever paid for twice.

Which strategies can actually change between rounds is a design decision, and
it is stated rather than hidden (:data:`USES_MODEL`):

- ``global_top``, ``per_column_quota``, ``uncertainty``,
  ``stratified_by_score`` and ``batch_diverse`` re-rank on the **current
  model's probability** once one exists. For them sequential acquisition is a
  genuinely different sampler.
- ``conditional_top`` and ``cond_per_column`` are defined on ``c_full``,
  which no amount of labelling changes, and ``random`` ignores every score.
  For these three, sequential selects **exactly** the same cells as one-shot
  (modulo the per-round RNG stream for ``random``). That is not a bug: they
  are the control that shows any sequential-vs-one-shot difference elsewhere
  comes from the refit and not from the round structure.

FAILURE IS REPORTED, NEVER SILENT
=================================
Inherited from :class:`~src.detectors.ensemble.EnsembleDetector`: if the
accumulated label sample holds only one class there is nothing to learn, so
the detector falls back to the unsupervised combiner and says so through
``used_supervision_`` / ``fallback_reason_`` / :meth:`report`. At 0.251 %
prevalence this is the *expected* outcome for a weak strategy at small k, and
a row that fell back is not a measurement of supervision.

WHAT IS REUSED, AND WHY NOTHING IS COPIED
=========================================
CLAUDE.md §8: "Phase 8 correctly wraps Phase 7 rather than duplicating it —
follow that pattern." :class:`AcquisitionDetector` therefore **subclasses**
:class:`~src.detectors.ensemble.EnsembleDetector` and reuses its feature
layer (:func:`~src.detectors.ensemble.build_features`,
:class:`~src.detectors.ensemble.FeatureCache`, the out-of-fold conditional
scoring), its unsupervised combiners, its oracle, its meta-classifier fit and
its fallback semantics. The only thing overridden is *which cells get
labelled*, which is the one thing this route is about. ``src/detectors/
ensemble.py`` is not modified.

WORKED EXAMPLE
==============
    >>> import numpy as np, pandas as pd
    >>> n = 400
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"a": rng.integers(0, 5, n).astype(float)})
    >>> df["b"] = df["a"] * 2.0
    >>> guide = np.zeros(df.size); guide[:10] = 1.0
    >>> picks, notes = select_cells("global_top", 5, guide=guide,
    ...                            train_idx=np.arange(df.size), n_rows=n, seed=0)
    >>> picks.size, bool(np.all(picks < 10)), notes
    (5, True, [])
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.detection_eval.harness import flatten_cells
from src.detectors.ensemble import (
    CellFeatures,
    EnsembleDetector,
    FeatureCache,
    LabelOracle,
)

logger = logging.getLogger(__name__)

__all__ = [
    "STRATEGIES",
    "USES_MODEL",
    "DEFAULT_N_BINS",
    "AcquisitionDetector",
    "RoundRecord",
    "select_cells",
]

#: Every acquisition strategy, in report order. ``random`` first because it is
#: the control every other row is read against.
STRATEGIES: Tuple[str, ...] = (
    "random",
    "global_top",
    "per_column_quota",
    "uncertainty",
    "conditional_top",
    "stratified_by_score",
    "batch_diverse",
    "cond_top_live",
    "cond_per_column",
)

#: Does this strategy re-rank on the refitted model's probability between
#: rounds? ``False`` means sequential picks the same cells as one-shot, which
#: makes those strategies the control for the sequential comparison.
USES_MODEL: Dict[str, bool] = {
    "random": False,
    "global_top": True,
    "per_column_quota": True,
    "uncertainty": True,
    "conditional_top": False,
    "stratified_by_score": True,
    "batch_diverse": True,
    "cond_top_live": False,
    "cond_per_column": False,
}

#: Strategies that rank by the conditional score instead of the guide, by
#: definition (see the module docstring: on adult the rule score is
#: anti-correlated with truth, so a strategy may deliberately refuse it).
_CONDITIONAL_RANKED: Tuple[str, ...] = (
    "conditional_top",
    "cond_top_live",
    "cond_per_column",
)

#: Strategies that exclude cells both signals call "missing" from the pool,
#: inherited from wave 2's ``per_column`` sampler. The membership of this
#: tuple is itself an ablation: ``conditional_top`` is deliberately **not**
#: in it, and ``cond_top_live`` is the same strategy that is, so the two rows
#: isolate the exclusion from everything else.
_EXCLUDES_TRIVIAL_MISSING: Tuple[str, ...] = (
    "per_column_quota",
    "cond_top_live",
    "cond_per_column",
)

DEFAULT_N_BINS = 10

#: ``batch_diverse`` walks the ranking in chunks, stopping as soon as k cells
#: have been accepted. A first chunk this large covers every case where the
#: caps reject little; a dominated ranking (adult's ``fnlwgt`` column) simply
#: takes further chunks rather than silently returning short.
_DIVERSE_CHUNK = 4096


# ═══════════════════════════════════════════════════════════════════════════════
# Ranking primitives
# ═══════════════════════════════════════════════════════════════════════════════


def _seeded_order(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Positions of ``values`` sorted descending, ties broken by a fresh random
    key.

    Tie-breaking is not cosmetic here. Cells are enumerated **column-major**
    (:func:`src.detection_eval.harness.flatten_cells`), so array order
    correlates with the column — and on adult the column *determines* the
    label, because every corrupted cell lives in ``income``. A stable sort
    would therefore leak column order into the sampler's choices and make a
    strategy look better or worse than it is.
    """
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        return np.empty(0, dtype=int)
    return np.lexsort((rng.random(values.size), -values))


def _top_k(values: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Positions of the k largest ``values``, ties broken randomly."""
    if k <= 0:
        return np.empty(0, dtype=int)
    return _seeded_order(values, rng)[: int(k)]


def _within_group_rank(groups: np.ndarray, order: np.ndarray) -> np.ndarray:
    """
    0-based rank of each element inside its own group, given ``order`` already
    sorted by ``(group, descending score)``.

    Vectorised on purpose: the Python loop this replaces ran over every train
    cell (474,831 of them on adult) once per round, which dominated the sweep.
    """
    sorted_groups = groups[order]
    n = sorted_groups.size
    if n == 0:
        return np.empty(0, dtype=int)
    starts = np.flatnonzero(
        np.concatenate(([True], sorted_groups[1:] != sorted_groups[:-1]))
    )
    sizes = np.diff(np.concatenate((starts, [n])))
    return np.arange(n) - np.repeat(starts, sizes)


def _round_robin_by_group(
    groups: np.ndarray, values: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Round-robin allocation across groups: take every group's best cell, then
    every group's second-best, and so on until k cells are chosen.

    Returns positions into ``groups`` / ``values``. Equivalent to the Python
    round-robin in
    :meth:`src.detectors.ensemble.EnsembleDetector._per_column_informed`,
    which ``tests/test_acquisition.py`` asserts on a small frame rather than
    taking on trust.
    """
    if k <= 0 or groups.size == 0:
        return np.empty(0, dtype=int)
    order = np.lexsort((rng.random(values.size), -values, groups))
    rank = _within_group_rank(groups, order)
    # Within one rank level, visit groups in frame order: deterministic, and
    # it reproduces wave 2's `sorted(queues)` pass order.
    pick = np.lexsort((groups[order], rank))
    return order[pick[: int(k)]]


# ═══════════════════════════════════════════════════════════════════════════════
# The strategies
# ═══════════════════════════════════════════════════════════════════════════════


def _strategy_random(
    pool: np.ndarray, guide: np.ndarray, n_rows: int, k: int, rng, **_: Any
) -> np.ndarray:
    size = min(int(k), pool.size)
    if size <= 0:
        return np.empty(0, dtype=int)
    return rng.choice(pool, size=size, replace=False)


def _strategy_global_top(
    pool: np.ndarray, guide: np.ndarray, n_rows: int, k: int, rng, **_: Any
) -> np.ndarray:
    return pool[_top_k(guide[pool], k, rng)]


def _strategy_per_column(
    pool: np.ndarray, guide: np.ndarray, n_rows: int, k: int, rng, **_: Any
) -> np.ndarray:
    columns = pool // max(int(n_rows), 1)
    return pool[_round_robin_by_group(columns, guide[pool], k, rng)]


def _strategy_uncertainty(
    pool: np.ndarray, guide: np.ndarray, n_rows: int, k: int, rng, **_: Any
) -> np.ndarray:
    """Nearest to 0.5. ``guide`` is a probability (model) or a [0, 1] combiner
    score read as a pseudo-probability during cold start."""
    return pool[_top_k(-np.abs(guide[pool] - 0.5), k, rng)]


def _strategy_stratified(
    pool: np.ndarray,
    guide: np.ndarray,
    n_rows: int,
    k: int,
    rng,
    n_bins: int = DEFAULT_N_BINS,
    **_: Any,
) -> np.ndarray:
    """
    Equal-width bins over the observed guide range, round-robin across the
    non-empty bins, uniform at random within a bin.

    Equal-width rather than quantile: at adult's skew ~99 % of cells share
    score 0.0, so nine of ten quantile edges coincide and quantile binning
    degenerates into "the zero mass plus one other bin". Equal-width keeps the
    budget spread over the *range*, which is what the strategy is for.
    """
    values = guide[pool]
    lo, hi = float(values.min()), float(values.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # A constant guide carries no range to stratify; this is uniform
        # sampling, and saying so is better than pretending to stratify.
        return _strategy_random(pool, guide, n_rows, k, rng)
    edges = np.linspace(lo, hi, int(max(2, n_bins)) + 1)
    bin_of = np.clip(np.digitize(values, edges[1:-1], right=False), 0, len(edges) - 2)
    # Uniform within a bin = rank by a random key, then round-robin by bin.
    return pool[_round_robin_by_group(bin_of, rng.random(values.size), k, rng)]


def _strategy_batch_diverse(
    pool: np.ndarray, guide: np.ndarray, n_rows: int, k: int, rng, **_: Any
) -> np.ndarray:
    """
    Global top, de-duplicated across columns and rows.

    Caps: at most ``ceil(k / n_columns_in_pool)`` cells per column and at most
    one per row. The walk goes down the **whole** ranking if it has to — a
    ranking dominated by one column (adult's ``fnlwgt`` false positives) is
    exactly the case this strategy exists for, so stopping at a fixed prefix
    there would have turned the de-duplication into a random top-up and
    measured the wrong thing. It is chunked only so the common case touches a
    few thousand cells rather than all 678,330.
    """
    k = int(k)
    if k <= 0 or pool.size == 0:
        return np.empty(0, dtype=int)
    n_rows = max(int(n_rows), 1)
    ranked = pool[_seeded_order(guide[pool], rng)]
    n_columns = max(1, int(np.unique(pool // n_rows).size))
    per_column_cap = max(1, int(math.ceil(k / n_columns)))

    chosen: List[int] = []
    used_rows: set = set()
    column_count: Dict[int, int] = {}
    for start in range(0, ranked.size, _DIVERSE_CHUNK):
        for cell in ranked[start : start + _DIVERSE_CHUNK].tolist():
            col, row = cell // n_rows, cell % n_rows
            if column_count.get(col, 0) >= per_column_cap or row in used_rows:
                continue
            chosen.append(int(cell))
            used_rows.add(row)
            column_count[col] = column_count.get(col, 0) + 1
            if len(chosen) >= k:
                return np.asarray(chosen, dtype=int)
    return np.asarray(chosen, dtype=int)


_STRATEGY_FUNCTIONS = {
    "random": _strategy_random,
    "global_top": _strategy_global_top,
    "per_column_quota": _strategy_per_column,
    "uncertainty": _strategy_uncertainty,
    "conditional_top": _strategy_global_top,
    "stratified_by_score": _strategy_stratified,
    "batch_diverse": _strategy_batch_diverse,
    "cond_top_live": _strategy_global_top,
    "cond_per_column": _strategy_per_column,
}


def select_cells(
    strategy: str,
    k: int,
    guide: np.ndarray,
    train_idx: np.ndarray,
    n_rows: int,
    seed: int = 42,
    conditional: Optional[np.ndarray] = None,
    eligible: Optional[np.ndarray] = None,
    already_labelled: Optional[Sequence[int]] = None,
    n_bins: int = DEFAULT_N_BINS,
) -> Tuple[np.ndarray, List[str]]:
    """
    Choose up to ``k`` flat cell indices to pay for. **Takes no labels.**

    This is the whole of an acquisition strategy, as a pure function, so it
    can be measured on a synthetic set without constructing a detector — and
    so that the no-leakage claim is checkable by reading one signature.

    Parameters
    ----------
    strategy : str
        One of :data:`STRATEGIES`.
    k : int
        Labels to spend now. ``<= 0`` returns an empty selection.
    guide : np.ndarray
        The strategy's ranking signal over **all** cells, column-major: the
        current model's P(corrupted) when one has been fitted on paid labels,
        otherwise the unsupervised combiner. Values outside [0, 1] are fine
        for the ranking strategies but make ``uncertainty``'s "nearest 0.5"
        meaningless, so the caller is responsible for passing a probability.
    train_idx : np.ndarray
        Flat indices the method is permitted to label (the harness's train
        cells). Nothing outside this array can be selected.
    n_rows : int
        Rows of the source frame, which turns a flat index into
        ``(row, column)``: ``column = idx // n_rows``.
    seed : int
        Seeds the RNG. The selection is deterministic given
        ``(strategy, k, guide, train_idx, seed)``.
    conditional : np.ndarray, optional
        ``c_full`` over all cells. Required by ``conditional_top`` and
        ``cond_per_column``; ignored otherwise.
    eligible : np.ndarray, optional
        Boolean over all cells. ``per_column_quota`` and ``cond_per_column``
        restrict their pool to eligible cells (wave 2 excluded cells both
        signals call "missing"); if that leaves nothing, the restriction is
        dropped and a note says so.
    already_labelled : sequence of int, optional
        Cells paid for in earlier rounds, excluded from the pool so no label
        is bought twice.
    n_bins : int
        Bins for ``stratified_by_score``.

    Returns
    -------
    (chosen, notes)
        ``chosen`` holds at most ``k`` distinct indices, all drawn from
        ``train_idx``. ``notes`` records anything that degraded — an exhausted
        pool, a dropped eligibility filter, a random top-up.

    Raises
    ------
    ValueError
        Unknown strategy, or a conditional-ranked strategy without
        ``conditional``.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    notes: List[str] = []
    k = int(k)
    train_idx = np.asarray(train_idx, dtype=int).ravel()
    guide = np.asarray(guide, dtype=float).ravel()

    if k <= 0 or train_idx.size == 0:
        return np.empty(0, dtype=int), notes

    pool = train_idx
    if already_labelled is not None:
        paid = np.asarray(list(already_labelled), dtype=int)
        if paid.size:
            pool = np.setdiff1d(pool, paid, assume_unique=False)
    if pool.size == 0:
        notes.append("every train cell was already labelled; nothing left to query")
        return np.empty(0, dtype=int), notes

    ranking = guide
    if strategy in _CONDITIONAL_RANKED:
        if conditional is None:
            raise ValueError(
                f"strategy {strategy!r} ranks by the conditional score and "
                "therefore requires `conditional`."
            )
        ranking = np.asarray(conditional, dtype=float).ravel()

    restricted = pool
    if strategy in _EXCLUDES_TRIVIAL_MISSING and eligible is not None:
        keep = np.asarray(eligible, dtype=bool).ravel()[pool]
        if keep.any():
            restricted = pool[keep]
        else:
            notes.append(
                "no eligible (non-missing) train cell remained; the "
                "missing-cell exclusion was dropped for this round"
            )

    rng = np.random.default_rng(int(seed))
    chosen = np.asarray(
        _STRATEGY_FUNCTIONS[strategy](
            restricted, ranking, int(n_rows), k, rng, n_bins=n_bins
        ),
        dtype=int,
    ).ravel()
    chosen = pd.unique(chosen) if chosen.size else chosen

    # Top-up. A strategy can come back short (caps exhausted, an eligibility
    # filter smaller than k). The budget is the comparison's fixed axis, so it
    # is spent in full — uniformly at random, which is the control's rule, and
    # recorded so a "win" cannot hide behind a random tail.
    if chosen.size < k:
        remaining = np.setdiff1d(pool, chosen, assume_unique=False)
        n_extra = min(k - int(chosen.size), int(remaining.size))
        if n_extra > 0:
            extra = rng.choice(remaining, size=n_extra, replace=False)
            if strategy != "random":
                notes.append(
                    f"{strategy} returned {chosen.size} of {k} cells; "
                    f"{n_extra} topped up uniformly at random"
                )
            chosen = np.concatenate((chosen, np.asarray(extra, dtype=int)))
        elif remaining.size == 0:
            notes.append(
                f"pool exhausted: {chosen.size} of {k} requested labels available"
            )

    if chosen.size:
        assert np.all(np.isin(chosen, train_idx)), (
            "acquisition selected a cell outside train_idx, which would be a "
            "leakage bug"
        )
    return np.sort(chosen.astype(int)), notes


# ═══════════════════════════════════════════════════════════════════════════════
# The detector
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RoundRecord:
    """
    One acquisition round: what it spent, what it found, what it was guided by.

    ``n_positive`` is the mechanism-level number the whole route turns on. A
    strategy that improves PR-AUC without improving positives-per-query has
    not solved the sampling problem this route exists for, and the write-up
    must be able to tell those two cases apart.
    """

    index: int
    n_queried: int
    n_positive: int
    guided_by: str
    model_fitted: bool
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round": self.index,
            "n_queried": self.n_queried,
            "n_positive": self.n_positive,
            "guided_by": self.guided_by,
            "model_fitted": self.model_fitted,
            "notes": list(self.notes),
        }


class AcquisitionDetector(EnsembleDetector):
    """
    :class:`~src.detectors.ensemble.EnsembleDetector` with a choice of label
    acquisition strategy and optional sequential (multi-round) acquisition.

    Everything except *which cells get labelled* is inherited: the 19-feature
    matrix, the out-of-fold conditional scoring, the shared
    :class:`~src.detectors.ensemble.FeatureCache`, the unsupervised
    combiners, the logistic/GBM meta-classifier, the single-class fallback and
    the harness contract (``fit(dirty_df, train_mask)`` / ``score_cells``).

    Parameters
    ----------
    strategy : str
        One of :data:`STRATEGIES`. Default ``"random"`` — the control, chosen
        as the default on purpose so that an unconfigured run measures the
        baseline rather than the method.
    n_rounds : int
        ``1`` (default) spends the whole budget before seeing a label.
        ``R > 1`` splits it into R rounds and refits between them; see
        :data:`USES_MODEL` for which strategies that can actually change.
        Clamped to the budget, since a round cannot buy less than one label.
    n_bins : int
        Bins for ``stratified_by_score``.
    budget, label_oracle, profile, combiner, learner, seed, …
        As :class:`~src.detectors.ensemble.EnsembleDetector`. ``sampler`` is
        accepted but unused — acquisition replaces it — and is forced to the
        inherited default so a stale value cannot silently take effect.
    precomputed_features : CellFeatures, optional
        Skip :func:`~src.detectors.ensemble.build_features` and use this
        matrix. A test seam: it lets the strategy mechanism be measured on a
        hand-built feature matrix with a known positive region, without
        fitting a conditional model. Never used by the experiment.

    Attributes set by :meth:`fit` (in addition to the inherited ones)
    ----------------------------------------------------------------
    rounds_ : list[RoundRecord]
    positives_per_query_ : float or None
        ``n_positive_labels_ / n_labelled_``. ``None`` when nothing was
        queried — never 0.0, which would read as "we looked and found none".
    """

    def __init__(
        self,
        strategy: str = "random",
        n_rounds: int = 1,
        n_bins: int = DEFAULT_N_BINS,
        budget: Union[int, str] = 0,
        label_oracle: Optional[LabelOracle] = None,
        profile: Any = None,
        combiner: str = "noisy_or",
        learner: str = "logreg",
        seed: int = 42,
        iqr_k: float = 3.0,
        flag_statistical_outliers: bool = True,
        conditional_kwargs: Optional[Dict[str, Any]] = None,
        conditional_folds: int = 5,
        feature_cache: Optional[FeatureCache] = None,
        precomputed_features: Optional[CellFeatures] = None,
        name: Optional[str] = None,
    ) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
        if int(n_rounds) < 1:
            raise ValueError(f"n_rounds must be >= 1, got {n_rounds}")
        if int(n_bins) < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")

        super().__init__(
            budget=budget,
            label_oracle=label_oracle,
            profile=profile,
            combiner=combiner,
            learner=learner,
            seed=seed,
            iqr_k=iqr_k,
            flag_statistical_outliers=flag_statistical_outliers,
            conditional_kwargs=conditional_kwargs,
            conditional_folds=conditional_folds,
            feature_cache=feature_cache,
            name=name,
        )
        self.strategy = str(strategy)
        self.n_rounds = int(n_rounds)
        self.n_bins = int(n_bins)
        self.precomputed_features = precomputed_features
        if name is None:
            mode = "seq" if self.n_rounds > 1 else "oneshot"
            self.name = f"acq_{self.strategy}_b{self.budget}_{mode}"

        self.rounds_: List[RoundRecord] = []
        self.positives_per_query_: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Feature seam                                                        #
    # ------------------------------------------------------------------ #

    def _features_for(self, dirty_df: pd.DataFrame, profile: Any) -> CellFeatures:
        """Inherited cached builder, unless a matrix was injected for a test."""
        if self.precomputed_features is not None:
            features = self.precomputed_features
            expected = dirty_df.shape[0] * dirty_df.shape[1]
            if features.x.shape[0] != expected:
                raise ValueError(
                    f"precomputed_features has {features.x.shape[0]} cells, "
                    f"dirty_df has {expected}."
                )
            return features
        return super()._features_for(dirty_df, profile)

    def _eligible_pool(self, features: CellFeatures) -> np.ndarray:
        """
        Boolean over all cells: ``True`` = worth paying to label.

        Excludes cells both signals call "missing", reusing the inherited
        :meth:`~src.detectors.ensemble.EnsembleDetector._trivially_missing`
        heuristic. The measured reason (wave 2): ~6,500 adult cells are blank
        and **none** of them is corrupted, so a sampler that labels the
        top-scoring cells spends its whole budget on guaranteed negatives.
        Excluding them costs no labels and reads no ground truth — only the
        two signals' own output.
        """
        return ~self._trivially_missing(features)

    # ------------------------------------------------------------------ #
    # Acquisition                                                          #
    # ------------------------------------------------------------------ #

    def _round_sizes(self, k_total: int) -> List[int]:
        """
        Split ``k_total`` into rounds, remainder to the earlier rounds.

        The *total* is identical to the one-shot budget by construction, which
        is the only way a one-shot-vs-sequential comparison means anything.
        """
        rounds = max(1, min(int(self.n_rounds), int(k_total)))
        base, extra = divmod(int(k_total), rounds)
        return [base + (1 if i < extra else 0) for i in range(rounds)]

    def fit(
        self,
        dirty_df: pd.DataFrame,
        train_mask: Optional[pd.DataFrame] = None,
        profile: Any = None,
    ) -> "AcquisitionDetector":
        """
        Build the features, then spend the budget through the strategy.

        Called by :func:`src.detection_eval.harness.evaluate_detector` as
        ``fit(dirty_df, train_mask)`` — positionally, with **no labels**. The
        signature has nowhere to put a label array; every label comes from
        :attr:`label_oracle`, for cells inside ``train_mask`` only.

        ``budget=0`` and ``budget="full"`` are delegated to the inherited
        implementation unchanged: there is no acquisition decision to make
        when you buy nothing, or everything.
        """
        self.rounds_ = []
        self.positives_per_query_ = None

        if self.budget == 0 or self.budget == "full":
            super().fit(dirty_df, train_mask, profile)
            if self.n_labelled_:
                self.positives_per_query_ = (
                    self.n_positive_labels_ / self.n_labelled_
                )
            return self

        resolved_profile = self.profile if self.profile is not None else profile
        self.features_ = self._features_for(dirty_df, resolved_profile)
        features = self.features_
        n_cells = features.x.shape[0]

        train_flat = (
            np.ones(n_cells, dtype=bool)
            if train_mask is None
            else flatten_cells(pd.DataFrame(train_mask)).astype(bool)
        )
        if train_flat.size != n_cells:
            raise ValueError(
                f"train_mask has {train_flat.size} cells, dirty_df has {n_cells}."
            )

        self.labelled_indices_ = np.empty(0, dtype=int)
        self.n_labelled_ = 0
        self.n_positive_labels_ = 0
        self.used_supervision_ = False
        self.fallback_reason_ = None
        self.coefficients_ = None
        self._model = None

        if self.label_oracle is None:
            raise ValueError(
                f"budget={self.budget!r} requires a label_oracle; it is the only "
                "permitted route for ground truth into this detector."
            )

        train_idx = np.flatnonzero(train_flat)
        if train_idx.size == 0:
            self.fallback_reason_ = "train_mask selected zero cells; no labels queryable"
            logger.warning("AcquisitionDetector: %s", self.fallback_reason_)
            self.is_fitted_ = True
            return self

        unsupervised = self._unsupervised_scores(features)
        conditional = features.column("c_full")
        eligible = self._eligible_pool(features)

        k_total = min(int(self.budget), int(train_idx.size))
        paid_idx: List[int] = []
        paid_y: List[int] = []

        for r, k_r in enumerate(self._round_sizes(k_total)):
            model_available = self._model is not None and USES_MODEL[self.strategy]
            if model_available:
                guide = self._score_vector(features)
                guided_by = "model"
            else:
                guide = unsupervised
                guided_by = (
                    "conditional" if self.strategy in _CONDITIONAL_RANKED else "combiner"
                )
            chosen, notes = select_cells(
                strategy=self.strategy,
                k=k_r,
                guide=guide,
                train_idx=train_idx,
                n_rows=features.shape[0],
                seed=self.seed + r,
                conditional=conditional,
                eligible=eligible,
                already_labelled=paid_idx,
                n_bins=self.n_bins,
            )
            if chosen.size == 0:
                self.rounds_.append(
                    RoundRecord(r, 0, 0, guided_by, self._model is not None, notes)
                )
                continue

            y = self.label_oracle.query(chosen).astype(int)
            paid_idx.extend(int(i) for i in chosen)
            paid_y.extend(int(v) for v in y)

            fitted = self._refit(features.x, train_idx, paid_idx, paid_y)
            self.rounds_.append(
                RoundRecord(
                    index=r,
                    n_queried=int(chosen.size),
                    n_positive=int(y.sum()),
                    guided_by=guided_by,
                    model_fitted=fitted,
                    notes=notes,
                )
            )

        self.labelled_indices_ = np.asarray(sorted(paid_idx), dtype=int)
        self.n_labelled_ = len(paid_idx)
        self.n_positive_labels_ = int(sum(paid_y))
        self.positives_per_query_ = (
            self.n_positive_labels_ / self.n_labelled_ if self.n_labelled_ else None
        )
        self.used_supervision_ = self._model is not None

        if not self.used_supervision_:
            only = "positives" if self.n_positive_labels_ else "negatives"
            self.fallback_reason_ = (
                f"label sample of {self.n_labelled_} cell(s) contained only {only} "
                f"({self.n_positive_labels_} positive); a classifier has nothing to "
                f"learn, so the unsupervised '{self.combiner}' combiner is used"
            )
            logger.warning("AcquisitionDetector: %s", self.fallback_reason_)

        self.is_fitted_ = True
        return self

    def _refit(
        self,
        x: np.ndarray,
        train_idx: np.ndarray,
        paid_idx: Sequence[int],
        paid_y: Sequence[int],
    ) -> bool:
        """
        Refit the meta-classifier on every label paid for so far.

        Returns ``True`` if a model is now in place. A single-class sample has
        nothing to learn, so the model is left as it was (``None`` on the
        first rounds) and the inherited unsupervised fallback stands — the
        same semantics as :class:`~src.detectors.ensemble.EnsembleDetector`,
        which is what makes wave-2 and wave-3 rows comparable.
        """
        y = np.asarray(paid_y, dtype=int)
        if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
            return self._model is not None
        try:
            self._fit_meta(x, train_idx, np.asarray(paid_idx, dtype=int), y)
            return True
        except Exception as exc:  # pragma: no cover - sklearn edge cases
            self._model = None
            self.fallback_reason_ = (
                f"meta-classifier failed to fit ({type(exc).__name__}: {exc}); "
                f"falling back to the unsupervised '{self.combiner}' combiner"
            )
            logger.warning("AcquisitionDetector: %s", self.fallback_reason_)
            return False

    # ------------------------------------------------------------------ #
    # Reporting                                                            #
    # ------------------------------------------------------------------ #

    def report(self) -> Dict[str, Any]:
        """
        The inherited report plus the acquisition-specific fields.

        ``positives_per_query`` is the one the write-up turns on: it separates
        "the strategy found corruption" from "the learner got lucky".
        """
        out = super().report()
        out.update(
            {
                "strategy": self.strategy,
                "n_rounds": self.n_rounds,
                "n_rounds_run": len(self.rounds_),
                "positives_per_query": self.positives_per_query_,
                "rounds": [r.to_dict() for r in self.rounds_],
                "round_notes": [n for r in self.rounds_ for n in r.notes],
                "uses_model_between_rounds": USES_MODEL[self.strategy],
            }
        )
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"AcquisitionDetector(strategy={self.strategy!r}, "
            f"budget={self.budget!r}, n_rounds={self.n_rounds}, seed={self.seed})"
        )
