"""
src/detection_eval/harness.py
-----------------------------
The comparable-evaluation protocol for detection routes (R0).

THE POINT
=========
Every route in ``docs/research_program.md`` (rules, conditional modelling,
signal ensembling, …) must produce one number that can be put next to every
other route's number. That requires all of them to be measured on the same
labels, the same splits, the same seeds and the same metrics. This module is
that fixture. A route's job is to supply ``score_cells``; everything else is
fixed here so it cannot be tuned per route.

THE CONTRACT
============
A detector is anything satisfying :class:`DetectorProtocol`::

    score_cells(dirty_df, profile=None) -> pd.DataFrame   # float, higher = worse

The frame must have the same shape, index and columns as ``dirty_df``: one
score per cell, including cells the method is confident about. "I did not flag
it" must be expressed as a low score, not an absent row, because PR-AUC is a
ranking metric over *all* cells and a method that only scores its own flags
would be scored on a different population than its competitors.

Optionally a detector may implement::

    fit(dirty_df, train_mask) -> None     # train_mask: bool frame, True = train cell

LEAKAGE: HOW THE PROTOCOL ENFORCES IT
=====================================
Two separate things must not leak, and they leak in different ways.

1. **Labels.** ``cell_labels`` reads ``clean_df``; the labels are used for
   metrics and for stratifying splits, and are *never* passed to a detector.
   ``fit`` takes ``(dirty_df, train_mask)`` — there is no label parameter in
   the signature, so a detector cannot receive one even by accident. This is
   the same structural guarantee ``decide_all()`` has (CLAUDE.md §3).

2. **Dirty values at evaluation cells.** A fitted detector could memorise the
   cells it will be scored on. So :func:`evaluate_detector`, for each split:

       a. builds ``train_mask`` = True exactly on the split's TRAIN cells;
       b. calls ``detector.fit(dirty_df, train_mask)`` — the detector is
          contractually permitted to read ``dirty_df`` only where the mask is
          True, and the mask is the only thing telling it which those are;
       c. calls ``score_cells`` on the full frame (a detector must score every
          cell, including train cells — scoring is not where leakage happens);
       d. computes metrics on the EVAL cells only.

   A detector with no ``fit`` is scored once and reused across splits: there
   is nothing to leak, and re-scoring would waste time without changing a
   number. Splits still differ, because they evaluate different cell subsets.

   ``tests/test_detection_eval.py`` asserts (b) with a spy detector that
   records every cell its mask offered and fails if any eval cell appears.

   Honest limitation: the split is over *cells*, so a row can have some cells
   in train and others in eval (this is what the research program asks for —
   stratification over cells is the only way to control an 0.25 % positive
   rate). A conditional model that predicts column A from columns B…Z of the
   same row therefore sees eval-row *context*, just not the eval cell's own
   value. That is the right semantics for a self-supervised cell-level
   detector (at inference it genuinely has the rest of the row) but it is not
   row-level isolation, and any route that fits a row-level model must say so.

WHY BOOTSTRAP CIs OVER SPLITS
=============================
Research program §3.2: never report one run. Five splits give five values per
metric; :func:`src.experiments.aggregate.bootstrap_interval` (reused as-is —
its ``(values, confidence, n_boot, seed)`` signature fits directly) turns them
into a percentile interval. With n=5 the interval is wide and honest. It
measures *split variance only*: for a detector with no ``fit`` it is pure
evaluation-sample noise, not model variance.

Worked example
==============
    >>> from src.detection_eval.harness import RuleDetectorAdapter, evaluate_detector
    >>> out = evaluate_detector(RuleDetectorAdapter(), "breast_cancer")
    >>> out["intervals"]["pr_auc"]["mean"]        # doctest: +SKIP
    0.74...
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from src.data.loader import DataLoader
from src.data.profiler import profile_dataset
from src.detection.detector import Detector
from src.detection_eval.labels import cell_labels, label_stats
from src.detection_eval.metrics import DetectionMetrics, evaluate_scores
from src.experiments.aggregate import Interval, bootstrap_interval

logger = logging.getLogger(__name__)

__all__ = [
    "DetectorProtocol",
    "RuleDetectorAdapter",
    "evaluate_detector",
    "flatten_cells",
    "stratified_cell_splits",
]

DEFAULT_N_SPLITS = 5
DEFAULT_SEED = 42
DEFAULT_SPLIT_FRACTION = 0.30


# ═══════════════════════════════════════════════════════════════════════════════
# The detector contract
# ═══════════════════════════════════════════════════════════════════════════════


@runtime_checkable
class DetectorProtocol(Protocol):
    """
    What a detection route must provide to be measurable by this harness.

    ``score_cells(dirty_df, profile=None) -> pd.DataFrame``
        Floats, higher = more likely corrupted. Same shape, index and columns
        as ``dirty_df``. The scale is irrelevant (every metric here is
        rank-based or oracle-thresholded), but the values must be finite.

    ``name`` (optional attribute)
        Leaderboard label. Defaults to the class name.

    ``fit(dirty_df, train_mask)`` (optional method)
        Called once per split before scoring, if present. May read
        ``dirty_df`` only where ``train_mask`` is True. Receives no labels —
        the signature has nowhere to put them.
    """

    def score_cells(
        self, dirty_df: pd.DataFrame, profile: Optional[Any] = None
    ) -> pd.DataFrame:  # pragma: no cover - structural protocol
        ...


class RuleDetectorAdapter:
    """
    Wraps the existing Phase 3 :class:`src.detection.detector.Detector` so it
    satisfies :class:`DetectorProtocol`. This is the baseline every route must
    beat (research program §1).

    Mapping: ``detect()`` returns one :class:`DetectionRecord` per suspicious
    cell; each record's ``detection_score`` is written at its
    ``(row_id, column)``. Every other cell gets ``0.0``.

    THE SCORE IS COARSE AND HEAVILY TIED — THIS IS A FINDING, NOT A DEFECT
    ======================================================================
    The rule detector emits one of six fixed constants::

        1.00  MISSING
        0.90  OUT_OF_RANGE
        0.85  FORMAT_ERROR (numeric column, unparseable)
        0.70  binary column, value outside the known two-value set
        0.35  NOVEL_CATEGORY / casing variant
        0.30  WHITESPACE

    plus a computed-but-narrow band for statistical outliers, plus ``0.0``
    everywhere else. On adult that is ~672,000 cells sharing the single value
    0.0 and ~5,800 sharing a handful of others. Consequences, which must be
    stated whenever this baseline's numbers are quoted:

    - Inside a tie group the ranking carries no information, so PR-AUC gets
      the group's average precision — it cannot reward the detector for
      ordering within a rule. A rule detector is a *set*, and PR-AUC measures
      a *ranking*; the baseline is therefore being measured on an axis it does
      not attempt. That is exactly why PR-AUC is where a probabilistic
      detector can win.
    - The achievable PR-AUC is bounded by the step function the ties define:
      with g distinct scores the PR curve has at most g points.
    - precision@k is only meaningful via expected tie-breaking
      (:func:`src.detection_eval.metrics.precision_at_k` does this), otherwise
      it reads out column order.

    Parameters
    ----------
    detector : Detector, optional
        A pre-built detector. If omitted, one is built per call from the
        ``profile`` passed to :meth:`score_cells`, using ``iqr_k`` and
        ``flag_statistical_outliers``.
    iqr_k : float
        IQR fence multiplier, default 3.0 (the shipped default).
    flag_statistical_outliers : bool
        Default True. ``src/pipeline/runner.py`` disables this for
        smartfactory, where fences produce only false positives;
        :func:`evaluate_detector` reproduces that per-dataset default so the
        baseline measured here is the baseline the project actually ships.
    name : str
        Leaderboard label.
    """

    def __init__(
        self,
        detector: Optional[Detector] = None,
        iqr_k: float = 3.0,
        flag_statistical_outliers: bool = True,
        name: str = "rule_detector",
    ) -> None:
        if iqr_k <= 0:
            raise ValueError(f"iqr_k must be > 0, got {iqr_k}")
        self.detector = detector
        self.iqr_k = float(iqr_k)
        self.flag_statistical_outliers = bool(flag_statistical_outliers)
        self.name = name

    def score_cells(
        self, dirty_df: pd.DataFrame, profile: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        Score every cell of ``dirty_df``.

        Returns
        -------
        pd.DataFrame
            float64, same shape/index/columns as ``dirty_df``.

        Raises
        ------
        ValueError
            Neither a pre-built detector nor a ``profile`` was supplied.
        """
        detector = self.detector
        if detector is None:
            if profile is None:
                raise ValueError(
                    "RuleDetectorAdapter needs either a pre-built Detector or a "
                    "DatasetProfile passed as `profile` (the Phase 3 detector is "
                    "constructed from a profile)."
                )
            detector = Detector(
                profile,
                iqr_k=self.iqr_k,
                flag_statistical_outliers=self.flag_statistical_outliers,
            )

        # Positional assignment: DetectionRecord.row_id is a zero-based
        # *position* in dirty_df, which only coincides with the index label
        # for a RangeIndex. Writing into a numpy array by position is correct
        # for any index and avoids a silent KeyError-free mis-write.
        arr = np.zeros(dirty_df.shape, dtype=float)
        col_pos = {c: i for i, c in enumerate(dirty_df.columns)}
        n_rows = dirty_df.shape[0]
        records = detector.detect(dirty_df)
        n_skipped = 0
        for rec in records:
            j = col_pos.get(rec.column)
            i = int(rec.row_id)
            if j is None or not (0 <= i < n_rows):
                n_skipped += 1
                continue
            arr[i, j] = float(rec.detection_score)
        scores = pd.DataFrame(
            arr, index=dirty_df.index, columns=dirty_df.columns, dtype=float
        )
        if n_skipped:
            logger.warning(
                "%d detection record(s) did not map onto a cell of dirty_df.",
                n_skipped,
            )
        logger.info(
            "RuleDetectorAdapter: %d of %d cells scored above 0.0 (%d distinct scores).",
            int((scores.to_numpy() > 0.0).sum()),
            scores.size,
            len(np.unique(scores.to_numpy())),
        )
        return scores


# ═══════════════════════════════════════════════════════════════════════════════
# Flattening and splitting
# ═══════════════════════════════════════════════════════════════════════════════


def flatten_cells(frame: pd.DataFrame) -> np.ndarray:
    """
    Flatten a cell frame to 1-D in a fixed, documented order.

    Order is **column-major**: all of column 0's rows, then column 1's, … It
    is an arbitrary choice, but it must be the *same* arbitrary choice for
    labels and for every detector's scores, or the two arrays would not line
    up. :func:`evaluate_detector` checks column equality before flattening, so
    a detector that reorders columns is rejected rather than silently
    mis-scored.
    """
    return np.concatenate([frame[c].to_numpy().ravel() for c in frame.columns])


def stratified_cell_splits(
    y: np.ndarray,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
    split_fraction: float = DEFAULT_SPLIT_FRACTION,
) -> List[np.ndarray]:
    """
    ``n_splits`` eval-index arrays, each holding ``split_fraction`` of the
    positives *and* ``split_fraction`` of the negatives.

    Stratification is not optional here. At 0.25 % prevalence an unstratified
    30 % split of adult would contain 510 positives ± ~22 by chance, so
    per-split PR-AUC would partly measure how many positives the dice handed
    out. Stratifying fixes the per-split prevalence to the population value,
    leaving only *which* positives varying — which is the sampling noise we
    actually want the CI to describe.

    Determinism: split ``i`` uses ``default_rng(seed + i)``, so the whole set
    is reproducible from ``seed`` alone and adding a split never changes the
    earlier ones.

    Parameters
    ----------
    y : np.ndarray
        Boolean/0-1 labels, one per cell, in :func:`flatten_cells` order.
    n_splits : int
        >= 1.
    seed : int
        >= 0.
    split_fraction : float
        In (0, 1). Fraction held out for evaluation.

    Returns
    -------
    list of np.ndarray
        Sorted eval indices per split. The train side is the complement.

    Notes
    -----
    Rounding: ``round(fraction * n_pos)`` positives and
    ``round(fraction * n_neg)`` negatives, each clamped to at least 1 when the
    class is non-empty, so an eval split is never silently positive-free (a
    positive-free eval split makes PR-AUC undefined and would drop the split
    from the CI without saying so).
    """
    y = np.asarray(y).astype(bool).ravel()
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    if not 0.0 < split_fraction < 1.0:
        raise ValueError(f"split_fraction must be in (0, 1), got {split_fraction}")

    pos_idx = np.flatnonzero(y)
    neg_idx = np.flatnonzero(~y)

    def take(n_total: int) -> int:
        if n_total == 0:
            return 0
        return max(1, min(n_total, int(round(split_fraction * n_total))))

    n_pos_eval = take(pos_idx.size)
    n_neg_eval = take(neg_idx.size)

    splits: List[np.ndarray] = []
    for i in range(n_splits):
        rng = np.random.default_rng(seed + i)
        chosen_pos = rng.permutation(pos_idx)[:n_pos_eval]
        chosen_neg = rng.permutation(neg_idx)[:n_neg_eval]
        splits.append(np.sort(np.concatenate((chosen_pos, chosen_neg))))
    return splits


# ═══════════════════════════════════════════════════════════════════════════════
# The evaluation entry point
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_detector(
    detector: DetectorProtocol,
    dataset: str,
    data_root: str = "datasets",
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
    split_fraction: float = DEFAULT_SPLIT_FRACTION,
    flag_statistical_outliers: Optional[bool] = None,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """
    Measure ``detector`` on ``dataset`` under the fixed R0 protocol.

    Parameters
    ----------
    detector : DetectorProtocol
        Must implement ``score_cells``. If it also implements ``fit``, it is
        refitted per split on train cells only (see module docstring).
    dataset : str
        Dataset directory name under ``data_root``.
    data_root : str
        Default ``"datasets"``.
    n_splits : int
        Evaluation splits, default 5.
    seed : int
        Split seed, default 42 (project-wide).
    split_fraction : float
        Eval fraction per split, default 0.30.
    flag_statistical_outliers : bool, optional
        Only consulted when ``detector`` is a :class:`RuleDetectorAdapter`
        that was constructed without an explicit detector; ``None`` applies
        the per-dataset shipped default (off for smartfactory, on otherwise),
        so this harness measures the baseline the project actually runs.
    confidence : float
        CI level, default 0.95.

    Returns
    -------
    dict
        ``dataset``, ``method``, ``n_splits``, ``seed``, ``split_fraction``,
        ``refit_per_split``, ``full_is_in_sample`` (True exactly when the
        detector was refitted, in which case ``full`` is partly in-sample and
        the honest numbers are ``intervals`` / ``per_split``),
        ``label_stats`` (full-dataset), ``full`` (metrics on every cell, as a
        dict), ``per_split`` (list of metric dicts), ``intervals``
        (``{metric: Interval.to_dict()}``, mean/lo/hi/std/n) and
        ``score_structure`` (distinct-score count and the tie profile — the
        tie structure is itself a reported finding).

    Notes
    -----
    ``intervals`` drops splits where a metric was undefined (``None``) rather
    than coercing them to 0.0. A metric undefined on every split yields an
    ``Interval`` with ``n = 0``, which renders as ``—``.
    """
    loader = DataLoader(data_root)
    clean_df, dirty_df = loader.load(dataset)
    profile = profile_dataset(dataset, loader.discover(dataset), clean_df, dirty_df)

    # clean_df's only use: 0/1 labels. Nothing below hands them to `detector`.
    labels = cell_labels(clean_df, dirty_df)
    stats = label_stats(labels)
    y = flatten_cells(labels).astype(bool)

    if isinstance(detector, RuleDetectorAdapter) and detector.detector is None:
        resolved = flag_statistical_outliers
        if resolved is None:
            resolved = dataset != "smartfactory"
        detector = RuleDetectorAdapter(
            iqr_k=detector.iqr_k,
            flag_statistical_outliers=bool(resolved),
            name=detector.name,
        )

    method = str(getattr(detector, "name", type(detector).__name__))
    fit_fn = getattr(detector, "fit", None)
    needs_fit = callable(fit_fn)

    splits = stratified_cell_splits(
        y, n_splits=n_splits, seed=seed, split_fraction=split_fraction
    )

    cached_scores: Optional[np.ndarray] = None
    if not needs_fit:
        # Nothing to leak, so one scoring pass serves every split.
        cached_scores = _score(detector, dirty_df, profile)

    per_split: List[DetectionMetrics] = []
    structure: Optional[Dict[str, Any]] = None
    for eval_idx in splits:
        if needs_fit:
            flat_mask = np.ones(y.size, dtype=bool)
            flat_mask[eval_idx] = False
            train_mask = _unflatten(flat_mask, dirty_df).astype(bool)
            fit_fn(dirty_df, train_mask)
            s = _score(detector, dirty_df, profile)
        else:
            assert cached_scores is not None
            s = cached_scores
        if structure is None:
            structure = _score_structure(s)
        per_split.append(evaluate_scores(y[eval_idx], s[eval_idx]))

    # For a detector with no `fit`, `full` is a clean whole-dataset
    # measurement. For a refit detector it is NOT: the surviving `s` comes
    # from the last split's model, which was fitted on cells that other
    # splits hold out, so `full` is partly in-sample. The flag
    # `full_is_in_sample` says so, and such a method must be read off
    # `intervals` / `per_split`, never off `full`.
    full_scores = cached_scores if cached_scores is not None else s
    full = evaluate_scores(y, full_scores)
    if structure is None:  # pragma: no cover - n_splits >= 1 guarantees a pass
        structure = _score_structure(full_scores)

    intervals = {
        field: _interval_for(per_split, field, confidence=confidence, seed=seed)
        for field in DetectionMetrics.SCORE_FIELDS
    }

    return {
        "dataset": dataset,
        "method": method,
        "n_splits": int(n_splits),
        "seed": int(seed),
        "split_fraction": float(split_fraction),
        "refit_per_split": bool(needs_fit),
        "full_is_in_sample": bool(needs_fit),
        "label_stats": stats,
        "full": full.to_dict(),
        "per_split": [m.to_dict() for m in per_split],
        "intervals": {k: v.to_dict() for k, v in intervals.items()},
        "score_structure": structure,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Internal
# ═══════════════════════════════════════════════════════════════════════════════


def _score(
    detector: DetectorProtocol, dirty_df: pd.DataFrame, profile: Any
) -> np.ndarray:
    """
    Call ``score_cells`` and validate the frame before trusting a number of it.

    A detector that returns the wrong shape, reordered columns, a different
    index or non-finite scores is rejected loudly. Silently reindexing would
    produce a plausible-looking leaderboard row from mis-aligned data.
    """
    frame = detector.score_cells(dirty_df, profile)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"{type(detector).__name__}.score_cells must return a DataFrame, "
            f"got {type(frame).__name__}."
        )
    if frame.shape != dirty_df.shape:
        raise ValueError(
            f"score_cells returned shape {frame.shape}, expected {dirty_df.shape}. "
            "A detector must score every cell, including the ones it does not flag."
        )
    if list(frame.columns) != list(dirty_df.columns):
        raise ValueError(
            "score_cells returned columns in a different order/name set than "
            "dirty_df; cell alignment would be wrong."
        )
    if not frame.index.equals(dirty_df.index):
        raise ValueError("score_cells returned a different index than dirty_df.")
    flat = flatten_cells(frame).astype(float)
    if flat.size and not np.all(np.isfinite(flat)):
        raise ValueError(
            f"score_cells returned {int((~np.isfinite(flat)).sum())} non-finite "
            "score(s). Scores must be finite."
        )
    return flat


def _unflatten(flat: np.ndarray, like: pd.DataFrame) -> pd.DataFrame:
    """Inverse of :func:`flatten_cells` (column-major)."""
    n_rows = like.shape[0]
    data = {
        col: flat[i * n_rows : (i + 1) * n_rows] for i, col in enumerate(like.columns)
    }
    return pd.DataFrame(data, index=like.index, columns=like.columns)


def _score_structure(scores: np.ndarray) -> Dict[str, Any]:
    """
    Describe the tie structure of a score vector.

    Reported because it bounds what PR-AUC can be: with ``n_distinct``
    distinct values the PR curve has at most ``n_distinct`` points, and the
    largest tie group's size says how much of the ranking is a coin flip.
    """
    values, counts = np.unique(scores, return_counts=True)
    order = np.argsort(-counts)
    top = [
        {"score": float(values[i]), "n_cells": int(counts[i])} for i in order[:8]
    ]
    return {
        "n_cells": int(scores.size),
        "n_distinct_scores": int(values.size),
        "largest_tie_group": int(counts.max()) if counts.size else 0,
        "largest_tie_fraction": (
            float(counts.max() / scores.size) if scores.size else 0.0
        ),
        "n_nonzero": int(np.count_nonzero(scores)),
        "top_groups": top,
    }


def _interval_for(
    per_split: Sequence[DetectionMetrics],
    field: str,
    confidence: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """
    Bootstrap interval over splits for one metric, skipping undefined splits.

    Reuses :func:`src.experiments.aggregate.bootstrap_interval` unchanged.
    """
    values = [
        float(getattr(m, field))
        for m in per_split
        if getattr(m, field) is not None
    ]
    return bootstrap_interval(values, confidence=confidence, seed=seed)
