"""
src/detectors/ensemble.py
-------------------------
Signal-ensembling error detection with an explicit, tiny label budget
(wave-2 route W2.1 of ``docs/research_program.md``; route R3 + factor 2 of
``docs/literature.md``).

WHY THIS EXISTS — MEASURED, NOT ASSUMED
=======================================
Wave 1 measured two detector families under the same harness and found that
neither dominates (``docs/research_program.md`` §5)::

    dataset        rule detector      conditional model
    adult          PR-AUC 0.0025      PR-AUC 0.0120   (0.70 vs 0.13 on `income`)
    breast_cancer  PR-AUC 0.6712      PR-AUC 0.6404   (rules WIN here)

Rules see *syntactic* corruption (a missing token, an illegal value, a bad
format). A conditional model sees *semantic* corruption (a legal value in an
improbable row). Adult's corruption is 100 % valid→valid label flips, which no
rule can see; Breast Cancer's is mostly format errors and missing values,
which the conditional model partly wastes its capacity on. The two signals are
therefore **complementary, not competing** — and nothing in the project had
ever combined them.

The second measured fact comes from the literature (``docs/literature.md``,
factor 2): the largest documented jump in this field is the *first ~20 labels*
(Raha: 2 labels ≈ 0.40 F1 → 20 labels 0.85). This project uses **zero** labels
for detection. So the question this module exists to answer is not "is an
ensemble nicer" but:

    How many labelled cells does it take for a combination of cheap signals
    to beat BOTH single detectors — and does it saturate?

The deliverable is a **label-efficiency curve**, PR-AUC vs k, directly
comparable in shape (not in absolute value) to Raha's 2→20 result.

THE METHOD
==========
1. Build one per-cell feature vector out of signals that each cost little
   (:data:`FEATURE_NAMES`, 19 features, documented below).
2. Combine them:

   ``budget=0``       unsupervised. No labels at all. Rank-average (default)
                      or noisy-OR of the two detector signals. This is the
                      honest default and the fallback for every failure.
   ``budget=k``       query a metered :class:`LabelOracle` for exactly *k*
                      cells drawn from the TRAIN split, fit a small
                      :class:`~sklearn.linear_model.LogisticRegression` (or a
                      shallow GBM) on those *k* rows, predict every cell.
   ``budget="full"``  all train labels. The supervised upper bound.

3. Score every cell. Higher = more likely corrupted.

THE FEATURES (fixed order, :data:`FEATURE_NAMES`, shape ``(n_cells, 19)``)
==========================================================================
Cells are enumerated **column-major**, exactly as
:func:`src.detection_eval.harness.flatten_cells` does, so row *i* of the
feature matrix is cell *i* of the harness's flattened label vector. This is
not a convenience; a different order would silently mis-train the learner.

    rule block (7)      r_missing, r_out_of_range, r_statistical_outlier,
                        r_format_error, r_novel_category, r_whitespace
                        — binary indicators, one per
                        :class:`src.detection.detector.ErrorType`, plus
                        r_score, the rule's own ``detection_score``
                        (max over the records landing on that cell).
    conditional (3)     c_full      = 1 - P(value | rest of row)
                        c_marginal  = 1 - P(value)
                        c_ratio     = sigmoid(log P(value) - log P(value|row))
                        — all three read off ONE fitted
                        :class:`~src.detectors.conditional.ConditionalDetector`
                        (``mode="full"`` holds the marginal statistics too, so
                        the three modes cost one fit between them).
    row block (2)       row_n_other_flagged  = rule-flagged cells elsewhere in
                                               this row (leave-one-out)
                        row_mean_cond_other  = mean c_full over the row's
                                               OTHER cells (leave-one-out)
                        — leave-one-out on purpose: a self-inclusive row mean
                        is partly the cell's own feature again, which lets a
                        2-feature learner double-count it.
    column block (2)    col_cardinality_ratio = distinct non-null / n_rows
                        col_missing_rate      = fraction missing, in dirty
    column kind (5)     col_kind_numerical, col_kind_binary,
                        col_kind_categorical, col_kind_mixed, col_kind_id_like
                        — one-hot over :class:`src.data.schema.ColumnKind`.
                        Fixed width on purpose: the feature matrix must have
                        the same 19 columns on every dataset or the curves
                        are not comparable.

LEAKAGE (CLAUDE.md §3 and research-program §3 rule 1 — non-negotiable)
======================================================================
**The label budget is the only place ground truth enters, and it enters
through one metered object.**

- :class:`LabelOracle` is the single door. It wraps a 0/1 cell-corruption
  label frame (built by the permitted
  :func:`src.detection_eval.labels.cell_labels`), counts every query, records
  every queried cell index, and refuses to answer more than ``max_queries``.
- :meth:`EnsembleDetector.fit` samples its *k* cells **from the train mask
  only** and queries the oracle for exactly those. Eval-split labels are
  never requested, so they cannot be received.
  ``tests/test_ensemble_detector.py`` spies on the oracle and asserts both:
  every queried index is a train cell, and the query count never exceeds the
  budget.
- No parameter of this module is named after, or derived from, clean data.
  :func:`build_features` takes ``dirty_df`` and a profile; there is nowhere to
  put a clean frame. The test suite scans every public signature for
  ``"clean"``.
- ``budget=0`` never constructs a query at all; the test asserts the oracle
  is untouched.

WHAT THIS MODULE FITS, EXACTLY (stated, as §3 rule 1 requires)
==============================================================
Two different things are fitted, on two different data regimes, and conflating
them would make the numbers unreadable:

**The signal layer** (rule detector, conditional model) sees the dirty table
and **no labels**, and it is not refitted per evaluation split. The
conditional features are nevertheless scored **out of fold over rows** (5
folds by default), so no cell's conditional score comes from a model that saw
that cell's row. Three things follow:

  * Fitting is label-free either way, so the fold structure is a modelling
    choice, not a leakage question.
  * It is not cosmetic. A single in-sample fit partly fits the corruption and
    measures **0.5014** PR-AUC on breast_cancer where 5-fold out-of-fold
    measures **0.6408** — and wave 1's row-split protocol measured 0.6404 for
    the same detector. Scoring in-sample would hand the ensemble a
    handicapped single-detector baseline to beat. See
    :func:`_conditional_scores` for the full table.
  * The result is cacheable across the whole budget sweep
    (:class:`FeatureCache`) — ``folds`` conditional fits per dataset instead
    of one per (budget × split), which is the difference between minutes and
    hours.

**The meta-classifier** is supervised and is refitted per split on a fresh
label sample drawn from that split's train mask. The per-split CI therefore
includes *label-sample variance*, which is the dominant uncertainty at k = 5
and exactly what a label-efficiency curve must show.

**Soft-leakage caveat, inherited and unchanged** (CLAUDE.md §3): the rule
block needs a :class:`~src.data.schema.DatasetProfile`, which the project
builds from ``clean_df``, so rule fences and vocabularies encode clean
*aggregate* statistics. This module reads one field of it — ``columns[c].kind``
— and nothing else; it never touches ``profile.n_corrupted_cells`` or
``corrupted_cells_by_column``, which are ground truth. The caveat is the same
one the wave-1 rule baseline carries, which is why the comparison is fair.

HOW THE k LABELS ARE SPENT (and why not uniformly)
==================================================
Uniform sampling is useless at this prevalence: 20 cells drawn uniformly from
adult (0.251 % positive) contain at least one positive only ~5 % of the time,
so 19 of 20 runs would train on a single class. Published label-efficient
detectors (Raha, ED2) all spend their budget *informedly*.

:meth:`EnsembleDetector.fit` therefore splits the budget:

    ceil(k * top_fraction)   chosen informedly from the signals' own output
                             (costs no labels to choose — that is the point)
    the remainder            uniformly at random from the rest of the train
                             cells, which supplies negatives and keeps the
                             sample from collapsing to one class

*How* the informed share is chosen is itself a measured decision, and no
choice wins everywhere — the same shape of result as the two detectors.
Measured, 5 splits, seed 42, PR-AUC:

    adult            k=5    k=10   k=20   k=50   k=100
    per_column       .0106  .0106  .0106  .1018  .2656
    diverse          .0106  .0106  .0106  .1151  .0106
    top_combiner     .0106  .0106  .0106  .0106  .0106   (never learns)

    breast_cancer    k=5    k=10   k=20   k=50   k=100
    per_column       .6326  .5725  .6763  .7281  .7617
    top_combiner     .7318  .7086  .7354  .7319  .7324

(.0106 on adult is the label-free fallback: the sample held no positive, so no
classifier was fitted at all.)

On adult the top of a *global* ranking is the rule detector's `fnlwgt` false
positives and the blank cells of `workclass` / `occupation` — all guaranteed
clean, because adult's corruption lives only in `income`. ``top_combiner``
therefore draws **zero positives at every budget up to 100**, and ``diverse``
draws one, in one split of five, at k=50. ``per_column`` allocates round-robin
across columns (what Raha's per-column clustering does) and is the only one
that learns anything there.

On breast_cancer the trade reverses: at 7.5 % prevalence spread over several
columns the global top is already positive-rich, so ``top_combiner`` is better
for k <= 20, while per-column allocation spends labels on columns holding
little corruption. ``per_column`` is the default on **worst-case regret** —
its worst case is ~0.10 PR-AUC on breast_cancer at k=5, against
``top_combiner`` learning nothing at all on adult — and the loss is reported,
not hidden.

``top_fraction`` defaults to 0.5. Ties are broken by a seeded permutation,
never by array order — cells are enumerated column-major, so array order
correlates with the column and hence, on adult, with the label. The sampler is
deterministic given ``(seed, train_mask, features)``.

FAILURE IS REPORTED, NEVER SILENT
=================================
If the sampled labels contain only one class (very possible at k = 5), there
is nothing a classifier can learn. The detector then **falls back to the
unsupervised combiner** and says so: ``used_supervision_`` is False,
``fallback_reason_`` carries the explanation, :meth:`report` returns it, and
the experiment prints it in the results table. It does not crash, and it does
not return a garbage ranking that looks like a measurement.

WORKED EXAMPLE
==============
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> df = pd.DataFrame({"a": rng.integers(0, 5, n).astype(float)})
    >>> df["b"] = df["a"] * 2.0
    >>> df.loc[0, "b"] = 99.0                     # syntactic: out of range
    >>> df.loc[1, "b"] = 0.0                      # semantic: legal, wrong row
    >>> det = EnsembleDetector(budget=0, seed=42).fit(df)
    >>> s = det.score_cells(df)
    >>> bool(s.loc[0, "b"] > s["b"].median())
    True
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.data.schema import ColumnKind
from src.detection.detector import Detector, ErrorType
from src.detection_eval.harness import flatten_cells
from src.detectors.conditional import ConditionalDetector

logger = logging.getLogger(__name__)

__all__ = [
    "COMBINERS",
    "SAMPLERS",
    "FEATURE_NAMES",
    "FeatureCache",
    "CellFeatures",
    "EnsembleDetector",
    "LabelOracle",
    "normalised_ranks",
]

# ─────────────────────────────────────────────────────────────────────────────
# The feature contract. Order and width are part of the module's API: the
# experiment reports coefficients against these names and the tests assert the
# shape, so a silent reordering would corrupt every reported number.
# ─────────────────────────────────────────────────────────────────────────────

RULE_FEATURES: Tuple[str, ...] = (
    "r_missing",
    "r_out_of_range",
    "r_statistical_outlier",
    "r_format_error",
    "r_novel_category",
    "r_whitespace",
    "r_score",
)

CONDITIONAL_FEATURES: Tuple[str, ...] = ("c_full", "c_marginal", "c_ratio")

ROW_FEATURES: Tuple[str, ...] = ("row_n_other_flagged", "row_mean_cond_other")

COLUMN_FEATURES: Tuple[str, ...] = ("col_cardinality_ratio", "col_missing_rate")

KIND_FEATURES: Tuple[str, ...] = (
    "col_kind_numerical",
    "col_kind_binary",
    "col_kind_categorical",
    "col_kind_mixed",
    "col_kind_id_like",
)

FEATURE_NAMES: Tuple[str, ...] = (
    RULE_FEATURES + CONDITIONAL_FEATURES + ROW_FEATURES + COLUMN_FEATURES + KIND_FEATURES
)

#: Maps an :class:`src.detection.detector.ErrorType` onto its indicator feature.
_ERROR_TYPE_TO_FEATURE: Dict[str, str] = {
    ErrorType.MISSING.value: "r_missing",
    ErrorType.OUT_OF_RANGE.value: "r_out_of_range",
    ErrorType.STATISTICAL_OUTLIER.value: "r_statistical_outlier",
    ErrorType.FORMAT_ERROR.value: "r_format_error",
    ErrorType.NOVEL_CATEGORY.value: "r_novel_category",
    ErrorType.WHITESPACE.value: "r_whitespace",
}

def _enum_value(value: Any) -> str:
    """
    The *value* of a ``(str, Enum)`` member, never its repr.

    ``str(ColumnKind.NUMERICAL)`` is ``"ColumnKind.NUMERICAL"``, not
    ``"numerical"`` — a mixin enum does not inherit ``__str__`` from ``str``.
    Looking the member up by ``str()`` silently misses every time, which is
    exactly how the column-kind features were first written and why they came
    out all-zero. Accepts a plain string too, since
    :class:`src.detection.detector.DetectionRecord` stores ``error_type`` as
    one already.
    """
    return str(getattr(value, "value", value))


_KIND_TO_FEATURE: Dict[str, str] = {
    ColumnKind.NUMERICAL.value: "col_kind_numerical",
    ColumnKind.BINARY.value: "col_kind_binary",
    ColumnKind.CATEGORICAL.value: "col_kind_categorical",
    ColumnKind.MIXED.value: "col_kind_mixed",
    ColumnKind.ID_LIKE.value: "col_kind_id_like",
}

#: Unsupervised combiners, and the two single-signal controls.
#:
#: ``rank_mean``  mean of the normalised ranks of ``r_score`` and ``c_full``.
#:               Rank-averaging is scale-free, which matters because the rule
#:               score is six fixed constants while the conditional score is
#:               continuous on [0, 1]; averaging the raw values would let the
#:               rule's coarse steps dominate.
#: ``noisy_or``  1 - (1 - r_score)(1 - c_full): a product of experts, reading
#:               each signal as an independent probability of corruption.
#: ``rule_only`` / ``cond_only``  the single-detector controls, run through the
#:               identical feature/harness plumbing so that "the ensemble beats
#:               both" is a comparison of combination rules and nothing else.
#:               ``rule_only`` reproduces
#:               :class:`src.detection_eval.harness.RuleDetectorAdapter`
#:               bit-for-bit, which the test suite asserts.
COMBINERS: Tuple[str, ...] = ("rank_mean", "noisy_or", "rule_only", "cond_only")

LEARNERS: Tuple[str, ...] = ("logreg", "gbm")

#: How the label budget is spent. See :meth:`EnsembleDetector._sample_budget`.
#:
#: ``per_column``    informed share allocated round-robin across columns
#:                   (what Raha's per-column clustering actually does)
#: ``diverse``       informed share split round-robin across the signals
#:                   separately, over cells that are not trivially missing
#: ``top_combiner``  informed share to the top of the unsupervised combiner
SAMPLERS: Tuple[str, ...] = ("per_column", "diverse", "top_combiner")

_SIGNAL_FEATURES: Tuple[str, str] = ("r_score", "c_full")


# ═══════════════════════════════════════════════════════════════════════════════
# The one door ground truth may come through
# ═══════════════════════════════════════════════════════════════════════════════


class LabelOracle:
    """
    A metered human annotator: the single, counted entry point for labels.

    Wraps a boolean cell-label frame (normally
    :func:`src.detection_eval.labels.cell_labels`, the permitted ``clean_df``
    consumer) and exposes exactly one operation: "tell me the label of these
    cells". Every query is counted and every queried cell index is recorded,
    so a test can prove what the detector asked for, and ``max_queries``
    makes the budget a hard constraint rather than a promise.

    Parameters
    ----------
    labels : pd.DataFrame
        Boolean/0-1, one cell per entry. Flattened **column-major** to match
        :func:`src.detection_eval.harness.flatten_cells`, so index *i* means
        the same cell here as in the harness and in :func:`build_features`.
    max_queries : int, optional
        Hard cap on the total number of labels this oracle will ever return.
        ``None`` = uncapped (used for ``budget="full"``, the supervised upper
        bound). Exceeding it raises :class:`RuntimeError` — a budget that can
        be quietly overspent is not a budget.

    Attributes
    ----------
    n_queries : int
        Labels returned so far.
    queried_indices : list[int]
        Every flat cell index ever asked for, in ask order.

    Examples
    --------
    >>> labels = pd.DataFrame({"a": [True, False], "b": [False, False]})
    >>> oracle = LabelOracle(labels, max_queries=2)
    >>> oracle.query([0, 1]).tolist()
    [True, False]
    >>> oracle.n_queries
    2
    >>> oracle.query([2])                      # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    RuntimeError: label budget exhausted ...
    """

    def __init__(self, labels: pd.DataFrame, max_queries: Optional[int] = None) -> None:
        if not isinstance(labels, pd.DataFrame):
            raise TypeError(f"labels must be a DataFrame, got {type(labels).__name__}")
        if max_queries is not None and max_queries < 0:
            raise ValueError(f"max_queries must be >= 0 or None, got {max_queries}")
        self._flat = flatten_cells(labels).astype(bool)
        self.max_queries = max_queries
        self.n_queries = 0
        self.queried_indices: List[int] = []

    @property
    def n_cells(self) -> int:
        return int(self._flat.size)

    def query(self, flat_indices: Sequence[int]) -> np.ndarray:
        """
        Return the labels of ``flat_indices`` (column-major cell indices).

        Raises
        ------
        RuntimeError
            The query would push the total past ``max_queries``.
        IndexError
            An index is outside the label frame.
        """
        idx = np.asarray(list(flat_indices), dtype=int)
        if idx.size and (idx.min() < 0 or idx.max() >= self._flat.size):
            raise IndexError(
                f"flat index out of range for a {self._flat.size}-cell label frame"
            )
        if self.max_queries is not None and self.n_queries + idx.size > self.max_queries:
            raise RuntimeError(
                f"label budget exhausted: {self.n_queries} already spent, "
                f"{idx.size} more requested, cap {self.max_queries}."
            )
        self.n_queries += int(idx.size)
        self.queried_indices.extend(int(i) for i in idx)
        return self._flat[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# Features
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CellFeatures:
    """
    The per-cell feature matrix plus everything needed to read it.

    Attributes
    ----------
    x : np.ndarray
        ``(n_cells, len(FEATURE_NAMES))`` float64. Row *i* is cell *i* in
        column-major order.
    names : tuple[str, ...]
        ``FEATURE_NAMES``, carried along so a consumer never has to assume it.
    shape : tuple[int, int]
        ``(n_rows, n_cols)`` of the source frame, for unflattening.
    index, columns : pd.Index
        The source frame's index and columns, so a score vector can be turned
        back into a frame of exactly the right shape.
    notes : list[str]
        Anything that degraded: a conditional model that would not fit, a
        profile that was not supplied. Reported, never swallowed.
    """

    x: np.ndarray
    names: Tuple[str, ...]
    shape: Tuple[int, int]
    index: pd.Index
    columns: pd.Index
    notes: List[str] = field(default_factory=list)

    def column(self, name: str) -> np.ndarray:
        """One feature as a 1-D vector, by name."""
        return self.x[:, self.names.index(name)]

    def as_frame(self, values: np.ndarray) -> pd.DataFrame:
        """Unflatten a per-cell vector back to the source frame's shape."""
        n_rows = self.shape[0]
        data = {
            col: np.asarray(values[i * n_rows : (i + 1) * n_rows], dtype=float)
            for i, col in enumerate(self.columns)
        }
        return pd.DataFrame(data, index=self.index, columns=self.columns, dtype=float)


class FeatureCache:
    """
    Memoises :func:`build_features` across detectors.

    The expensive part of the feature matrix is the conditional model (one
    gradient-boosted model per column). The label-efficiency sweep builds a
    *new* detector for every budget and the harness refits it on every split,
    so without a shared cache the same conditional model would be refitted
    ``n_budgets * n_splits`` times for an identical result.

    The key is a content fingerprint of the frame (shape, column names and
    :func:`pandas.util.hash_pandas_object`) plus the feature-affecting
    settings, so two different frames — or the same frame with a different
    conditional seed — can never collide onto one entry.

    Pass one instance to every detector in a sweep; the default is a private
    per-detector cache, so nothing is shared by accident.
    """

    def __init__(self) -> None:
        self._store: Dict[Tuple[Any, ...], CellFeatures] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def fingerprint(frame: pd.DataFrame) -> Tuple[Any, ...]:
        """Content fingerprint of a frame: shape, columns, and a value hash."""
        try:
            digest = int(pd.util.hash_pandas_object(frame, index=True).sum())
        except Exception:  # pragma: no cover - exotic dtypes
            digest = int(pd.util.hash_pandas_object(frame.astype(str), index=True).sum())
        return (frame.shape, tuple(map(str, frame.columns)), digest)

    def get_or_build(self, key: Tuple[Any, ...], builder) -> CellFeatures:
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        value = builder()
        self._store[key] = value
        return value

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:  # pragma: no cover - cosmetic
        return len(self._store)


def normalised_ranks(values: np.ndarray) -> np.ndarray:
    """
    Average ranks of ``values``, scaled to [0, 1]; ties share their mean rank.

    Tie handling is the whole point. The rule detector emits six distinct
    scores over hundreds of thousands of cells, so "the rank of a cell" is
    only well defined as the *average* rank of its tie group. Breaking ties by
    array order would be actively harmful here, because cells are enumerated
    column-major and array order therefore correlates with the column — and on
    adult the column determines the label.

    >>> normalised_ranks(np.array([5.0, 1.0, 5.0, 3.0])).round(4).tolist()
    [0.8333, 0.0, 0.8333, 0.3333]
    """
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return arr
    if arr.size == 1:
        return np.zeros(1, dtype=float)
    _, inverse, counts = np.unique(arr, return_inverse=True, return_counts=True)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    average_rank = starts + (counts - 1) / 2.0
    return average_rank[inverse] / float(arr.size - 1)


DEFAULT_CONDITIONAL_FOLDS = 5


def _conditional_scores(
    dirty_df: pd.DataFrame,
    seed: int,
    folds: int,
    conditional_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    The three conditional features, scored **out of fold** by default.

    WHY OUT OF FOLD — MEASURED, NOT STYLISTIC
    =========================================
    A conditional model fitted on the whole dirty table trains on the very
    cells it will score. At breast_cancer's 7.5 % corruption prevalence that
    is not a small effect: the model partly *fits the corruption*, so a
    corrupted cell it trained on gets a high ``P(observed)`` and therefore a
    LOW suspicion score, and the detection is lost. Measured under the R0
    harness (3 splits, seed 42, ``max_iter=100``)::

        breast_cancer  single full-table fit   PR-AUC 0.5014
                       3-fold out-of-fold      PR-AUC 0.6197
                       5-fold out-of-fold      PR-AUC 0.6408

    Wave 1's row-split protocol reported **0.6404** for the same detector, and
    that is the same regime: its eval rows were never fitted. 5-fold
    out-of-fold reproduces it to 0.0004. So the single-fit number is an
    artefact of scoring in-sample, not a property of the method — and using
    the single fit would hand the ensemble a handicapped baseline to beat,
    which is why the default is 5 folds and not 1.

    The fix is standard stacked generalisation: partition the ROWS into
    ``folds`` seeded folds, fit one model per fold on the other folds, and
    keep only its scores for its own held-out rows. Every cell's conditional
    score then comes from a model that never saw that cell's row.

    It costs no labels — the folds are over rows of the dirty table, chosen by
    a seeded RNG — so this remains fully self-supervised. It costs ``folds``
    conditional fits instead of one, which is the reason the result is cached
    (:class:`FeatureCache`) rather than recomputed per budget and per split.

    ``folds=1`` restores the single in-sample fit, kept so the handicap above
    can be reproduced rather than taken on trust.

    Returns
    -------
    (blocks, notes)
        ``blocks`` maps ``c_full`` / ``c_marginal`` / ``c_ratio`` to
        ``(n_rows, n_cols)`` arrays; ``notes`` records any degradation.
    """
    n_rows, n_cols = dirty_df.shape
    names = (("c_full", "full"), ("c_marginal", "marginal"), ("c_ratio", "ratio"))
    blocks = {name: np.zeros((n_rows, n_cols), dtype=float) for name, _ in names}
    notes: List[str] = []

    kwargs = dict(conditional_kwargs or {})
    kwargs.setdefault("seed", seed)
    kwargs["mode"] = "full"  # a full-mode fit can score all three modes

    folds = max(1, int(folds))

    def single_fit() -> bool:
        try:
            model = ConditionalDetector(**kwargs).fit(dirty_df, profile=None)
            for name, mode in names:
                blocks[name] = model.score_cells(dirty_df, mode=mode).to_numpy(
                    dtype=float
                )
            return True
        except Exception as exc:
            notes.append(
                f"conditional model did not fit ({type(exc).__name__}: {exc}); "
                "c_full / c_marginal / c_ratio are all 0.0"
            )
            return False

    if folds == 1:
        single_fit()
        return blocks, notes

    assignment = np.random.default_rng(seed).integers(0, folds, n_rows)
    try:
        for fold in range(folds):
            held_out = assignment == fold
            if not held_out.any():
                continue
            model = ConditionalDetector(**kwargs).fit(
                dirty_df, profile=None, train_mask=~held_out
            )
            for name, mode in names:
                scored = model.score_cells(dirty_df, mode=mode).to_numpy(dtype=float)
                blocks[name][held_out, :] = scored[held_out, :]
        notes.append(f"conditional features scored out of fold ({folds} folds)")
    except Exception as exc:
        # A fold too small to fit is a real possibility on a tiny frame. Fall
        # back to the single in-sample fit and SAY SO, because that fallback
        # costs measurable PR-AUC (see the numbers above).
        for name, _ in names:
            blocks[name] = np.zeros((n_rows, n_cols), dtype=float)
        notes.append(
            f"out-of-fold conditional scoring failed ({type(exc).__name__}: {exc}); "
            "fell back to a single in-sample fit, which is measurably weaker"
        )
        single_fit()

    return blocks, notes


def build_features(
    dirty_df: pd.DataFrame,
    profile: Any = None,
    seed: int = 42,
    iqr_k: float = 3.0,
    flag_statistical_outliers: bool = True,
    conditional_kwargs: Optional[Dict[str, Any]] = None,
    rule_detector: Optional[Detector] = None,
    conditional_folds: int = DEFAULT_CONDITIONAL_FOLDS,
) -> CellFeatures:
    """
    Assemble the per-cell feature matrix from the dirty table alone.

    **No labels, no clean frame.** There is no parameter that could carry one.
    ``profile`` is read for exactly one thing, ``columns[c].kind``; see the
    module docstring's soft-leakage note.

    Parameters
    ----------
    dirty_df : pd.DataFrame
        The table to featurise. The only data source.
    profile : DatasetProfile, optional
        Needed by the Phase 3 rule detector. ``None`` leaves the seven rule
        features and the five kind features at 0.0 and records a note — the
        conditional and row/column blocks still work, so the ensemble
        degrades rather than failing.
    seed : int
        Conditional-model seed. Scores are reproducible from it.
    iqr_k, flag_statistical_outliers
        Rule-detector settings. Defaults match the shipped Phase 3 detector;
        ``src/pipeline/runner.py`` turns fences off for smartfactory, and the
        experiment reproduces that per-dataset default so the rule block here
        equals the wave-1 baseline's.
    conditional_kwargs : dict, optional
        Extra keyword arguments for
        :class:`~src.detectors.conditional.ConditionalDetector`.
    rule_detector : Detector, optional
        A pre-built rule detector, used instead of constructing one.
    conditional_folds : int
        Row folds for out-of-fold conditional scoring, default 5. ``1`` fits
        once on the whole table, which scores the conditional features
        in-sample and costs real PR-AUC — see :func:`_conditional_scores` for
        the measured difference.

    Returns
    -------
    CellFeatures
    """
    if not isinstance(dirty_df, pd.DataFrame):
        raise TypeError(f"dirty_df must be a DataFrame, got {type(dirty_df).__name__}")
    if dirty_df.shape[1] == 0:
        raise ValueError("dirty_df has no columns")

    n_rows, n_cols = dirty_df.shape
    notes: List[str] = []
    blocks: Dict[str, np.ndarray] = {
        name: np.zeros((n_rows, n_cols), dtype=float) for name in FEATURE_NAMES
    }

    # ── rule block ────────────────────────────────────────────────────────
    detector = rule_detector
    if detector is None and profile is not None:
        detector = Detector(
            profile,
            iqr_k=iqr_k,
            flag_statistical_outliers=flag_statistical_outliers,
        )
    if detector is None:
        notes.append(
            "no DatasetProfile supplied: rule and column-kind features are all 0.0"
        )
    else:
        col_pos = {c: i for i, c in enumerate(dirty_df.columns)}
        n_skipped = 0
        for rec in detector.detect(dirty_df):
            j = col_pos.get(rec.column)
            i = int(rec.row_id)
            if j is None or not (0 <= i < n_rows):
                # row_id is a zero-based *position* (the harness's adapter
                # documents this); anything out of range is dropped loudly
                # rather than silently written to the wrong cell.
                n_skipped += 1
                continue
            feature = _ERROR_TYPE_TO_FEATURE.get(_enum_value(rec.error_type))
            if feature is not None:
                blocks[feature][i, j] = 1.0
            # A cell can be flagged by more than one rule (whitespace *and* a
            # novel category, say). The score keeps the strongest claim.
            blocks["r_score"][i, j] = max(
                blocks["r_score"][i, j], float(rec.detection_score)
            )
        if n_skipped:
            notes.append(f"{n_skipped} detection record(s) did not map onto a cell")

    # ── conditional block ─────────────────────────────────────────────────
    conditional_blocks, conditional_notes = _conditional_scores(
        dirty_df,
        seed=seed,
        folds=conditional_folds,
        conditional_kwargs=conditional_kwargs,
    )
    blocks.update(conditional_blocks)
    notes.extend(conditional_notes)

    # ── row block (leave-one-out, so a cell is not its own context) ───────
    flagged = (blocks["r_score"] > 0.0).astype(float)
    blocks["row_n_other_flagged"] = flagged.sum(axis=1, keepdims=True) - flagged
    cond = blocks["c_full"]
    if n_cols > 1:
        blocks["row_mean_cond_other"] = (
            cond.sum(axis=1, keepdims=True) - cond
        ) / float(n_cols - 1)
    else:
        notes.append("single-column frame: row_mean_cond_other is 0.0 by definition")

    # ── column block ──────────────────────────────────────────────────────
    for j, col in enumerate(dirty_df.columns):
        series = dirty_df[col]
        non_null = series.dropna()
        ratio = (non_null.nunique() / n_rows) if n_rows else 0.0
        blocks["col_cardinality_ratio"][:, j] = float(ratio)
        blocks["col_missing_rate"][:, j] = (
            float(series.isna().mean()) if n_rows else 0.0
        )
        kind_feature = None
        if profile is not None:
            col_profile = getattr(profile, "columns", {}).get(col)
            if col_profile is not None:
                kind_feature = _KIND_TO_FEATURE.get(
                    _enum_value(getattr(col_profile, "kind", ""))
                )
        if kind_feature is not None:
            blocks[kind_feature][:, j] = 1.0

    x = np.column_stack(
        [
            np.concatenate([blocks[name][:, j] for j in range(n_cols)])
            for name in FEATURE_NAMES
        ]
    ).astype(float)
    x[~np.isfinite(x)] = 0.0

    if x.shape != (n_rows * n_cols, len(FEATURE_NAMES)):  # pragma: no cover
        raise AssertionError(f"feature matrix has shape {x.shape}, which is a bug")

    return CellFeatures(
        x=x,
        names=FEATURE_NAMES,
        shape=(n_rows, n_cols),
        index=dirty_df.index.copy(),
        columns=dirty_df.columns.copy(),
        notes=notes,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# The detector
# ═══════════════════════════════════════════════════════════════════════════════


class EnsembleDetector:
    """
    Per-cell error detector combining cheap signals, optionally with a tiny
    explicit label budget. Satisfies
    :class:`src.detection_eval.harness.DetectorProtocol`.

    Parameters
    ----------
    budget : int or ``"full"``
        ``0`` = unsupervised (no labels at all, no oracle required).
        ``k > 0`` = query the oracle for exactly *k* train cells.
        ``"full"`` = every train cell; the supervised upper bound.
    label_oracle : LabelOracle, optional
        Required when ``budget != 0``. The only source of ground truth.
    profile : DatasetProfile, optional
        Needed for the rule block. Supplied at construction because
        :meth:`fit` is called by the harness as ``fit(dirty_df, train_mask)``
        and so has nowhere to receive one; :meth:`score_cells` also accepts
        one, and the constructor's wins when both are present.
    combiner : str
        Unsupervised combination rule, used at ``budget=0`` and as the
        fallback for every supervised failure. One of :data:`COMBINERS`;
        ``rule_only`` / ``cond_only`` are the single-signal controls.
    learner : str
        ``"logreg"`` (default) or ``"gbm"``. With k = 5..20 labels a linear
        model is the only defensible choice; the GBM is offered for
        ``budget="full"`` and is reported separately.
    sampler : str
        How the informed share of the budget is spent. ``"per_column"``
        (default) allocates it round-robin across columns, which is what
        Raha's per-column clustering does; ``"diverse"`` splits it across the
        two signals separately; ``"top_combiner"`` sends it all to the top of
        the unsupervised combiner. The last two are global rankings; neither
        learns anything on adult up to k = 100, while ``top_combiner`` is the
        better choice on breast_cancer for k <= 20. :meth:`_sample_budget`
        carries the measured table.
    top_fraction : float
        Share of the budget spent informedly; the rest is uniform random.
        Default 0.5. See the module docstring for why uniform-only sampling
        is useless here.
    seed : int
        Seeds the label sample, the tie-breaking permutation, the conditional
        model and the learner. Scores are reproducible from it.
    iqr_k, flag_statistical_outliers
        Rule-detector settings, passed to :func:`build_features`.
    conditional_kwargs : dict, optional
        Passed to :class:`~src.detectors.conditional.ConditionalDetector`.
    conditional_folds : int
        Row folds for out-of-fold conditional scoring (default 5), so no
        cell's conditional score comes from a model that saw its row. ``1``
        reverts to a single in-sample fit; :func:`_conditional_scores` has the
        measured cost of doing that.
    feature_cache : FeatureCache, optional
        Share one across a sweep so the conditional model is fitted once.
    name : str, optional
        Leaderboard label. Default ``"ensemble_b0"`` / ``"ensemble_b20"`` /
        ``"ensemble_bfull"``.

    Attributes set by :meth:`fit`
    -----------------------------
    features_ : CellFeatures
    labelled_indices_ : np.ndarray      the cells actually labelled
    n_labelled_, n_positive_labels_ : int
    used_supervision_ : bool            False = the unsupervised fallback ran
    fallback_reason_ : str or None       why, if it did
    coefficients_ : dict or None        feature → weight (logreg only)
    is_fitted_ : bool
    """

    def __init__(
        self,
        budget: Union[int, str] = 0,
        label_oracle: Optional[LabelOracle] = None,
        profile: Any = None,
        combiner: str = "rank_mean",
        learner: str = "logreg",
        sampler: str = "per_column",
        top_fraction: float = 0.5,
        seed: int = 42,
        iqr_k: float = 3.0,
        flag_statistical_outliers: bool = True,
        conditional_kwargs: Optional[Dict[str, Any]] = None,
        conditional_folds: int = DEFAULT_CONDITIONAL_FOLDS,
        feature_cache: Optional[FeatureCache] = None,
        name: Optional[str] = None,
    ) -> None:
        self.budget = _validate_budget(budget)
        if combiner not in COMBINERS:
            raise ValueError(f"combiner must be one of {COMBINERS}, got {combiner!r}")
        if learner not in LEARNERS:
            raise ValueError(f"learner must be one of {LEARNERS}, got {learner!r}")
        if sampler not in SAMPLERS:
            raise ValueError(f"sampler must be one of {SAMPLERS}, got {sampler!r}")
        if not 0.0 <= top_fraction <= 1.0:
            raise ValueError(f"top_fraction must be in [0, 1], got {top_fraction}")
        if seed < 0:
            raise ValueError(f"seed must be >= 0, got {seed}")

        self.label_oracle = label_oracle
        self.profile = profile
        self.combiner = combiner
        self.learner = learner
        self.sampler = sampler
        self.top_fraction = float(top_fraction)
        self.seed = int(seed)
        self.iqr_k = float(iqr_k)
        self.flag_statistical_outliers = bool(flag_statistical_outliers)
        self.conditional_kwargs = dict(conditional_kwargs or {})
        if int(conditional_folds) < 1:
            raise ValueError(
                f"conditional_folds must be >= 1, got {conditional_folds}"
            )
        self.conditional_folds = int(conditional_folds)
        self.feature_cache = feature_cache if feature_cache is not None else FeatureCache()
        self.name = name or f"ensemble_b{self.budget}"

        # Fitted state
        self.features_: Optional[CellFeatures] = None
        self.labelled_indices_: np.ndarray = np.empty(0, dtype=int)
        self.n_labelled_: int = 0
        self.n_positive_labels_: int = 0
        self.used_supervision_: bool = False
        self.fallback_reason_: Optional[str] = None
        self.coefficients_: Optional[Dict[str, float]] = None
        self.is_fitted_: bool = False
        self._model: Any = None
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_scale: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Harness interface                                                    #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        dirty_df: pd.DataFrame,
        train_mask: Optional[pd.DataFrame] = None,
        profile: Any = None,
    ) -> "EnsembleDetector":
        """
        Build the features and, if a budget was given, spend it.

        Called by :func:`src.detection_eval.harness.evaluate_detector` as
        ``fit(dirty_df, train_mask)`` — positionally, with no profile and
        **no labels**: the signature has nowhere to put a label array, which
        is the structural half of the leakage guarantee. Labels can only
        arrive through :attr:`label_oracle`, and only for cells inside
        ``train_mask``.

        Parameters
        ----------
        dirty_df : pd.DataFrame
            The dirty table. The only data source for every feature.
        train_mask : pd.DataFrame, optional
            Boolean frame, same shape as ``dirty_df``; ``True`` marks a cell
            the method may label. ``None`` = every cell is labellable (used
            outside the harness, e.g. in doctests).
        profile : DatasetProfile, optional
            Fallback when the constructor was not given one.

        Returns
        -------
        self
        """
        resolved_profile = self.profile if self.profile is not None else profile
        self.features_ = self._features_for(dirty_df, resolved_profile)

        n_cells = self.features_.x.shape[0]
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

        if self.budget == 0:
            self.fallback_reason_ = None
            self.is_fitted_ = True
            return self

        if self.label_oracle is None:
            raise ValueError(
                f"budget={self.budget!r} requires a label_oracle; it is the only "
                "permitted route for ground truth into this detector."
            )

        train_idx = np.flatnonzero(train_flat)
        if train_idx.size == 0:
            self.fallback_reason_ = "train_mask selected zero cells; no labels queryable"
            logger.warning("EnsembleDetector: %s", self.fallback_reason_)
            self.is_fitted_ = True
            return self

        chosen = self._sample_budget(train_idx)
        y = self.label_oracle.query(chosen).astype(int)
        self.labelled_indices_ = chosen
        self.n_labelled_ = int(chosen.size)
        self.n_positive_labels_ = int(y.sum())

        if self.n_positive_labels_ == 0 or self.n_positive_labels_ == y.size:
            only = "positives" if self.n_positive_labels_ else "negatives"
            self.fallback_reason_ = (
                f"label sample of {y.size} cell(s) contained only {only} "
                f"({self.n_positive_labels_} positive); a classifier has nothing to "
                f"learn, so the unsupervised '{self.combiner}' combiner is used"
            )
            logger.warning("EnsembleDetector: %s", self.fallback_reason_)
            self.is_fitted_ = True
            return self

        try:
            self._fit_meta(self.features_.x, train_idx, chosen, y)
            self.used_supervision_ = True
        except Exception as exc:
            self._model = None
            self.fallback_reason_ = (
                f"meta-classifier failed to fit ({type(exc).__name__}: {exc}); "
                f"falling back to the unsupervised '{self.combiner}' combiner"
            )
            logger.warning("EnsembleDetector: %s", self.fallback_reason_)

        self.is_fitted_ = True
        return self

    def score_cells(
        self, dirty_df: pd.DataFrame, profile: Any = None
    ) -> pd.DataFrame:
        """
        Score every cell of ``dirty_df``. Higher = more likely corrupted.

        Returns a float frame with exactly ``dirty_df``'s shape, index and
        column order, as the harness requires: "I did not flag it" is a low
        score, never an absent row.

        Calling this without :meth:`fit` fits the unsupervised path on the
        spot (there is nothing to learn at ``budget=0``) and raises when a
        budget was requested, because spending a budget is an explicit act.
        """
        if not self.is_fitted_:
            if self.budget != 0:
                raise RuntimeError(
                    "score_cells() called before fit(); a label budget must be "
                    "spent explicitly in fit()."
                )
            self.fit(dirty_df, None, profile)

        resolved_profile = self.profile if self.profile is not None else profile
        features = self._features_for(dirty_df, resolved_profile)
        scores = self._score_vector(features)
        if not np.all(np.isfinite(scores)):  # pragma: no cover - defensive
            raise ValueError("ensemble produced non-finite scores")
        return features.as_frame(scores)

    def fit_score(
        self,
        dirty_df: pd.DataFrame,
        train_mask: Optional[pd.DataFrame] = None,
        profile: Any = None,
    ) -> pd.DataFrame:
        """Convenience: :meth:`fit` then :meth:`score_cells`."""
        return self.fit(dirty_df, train_mask, profile).score_cells(dirty_df, profile)

    def report(self) -> Dict[str, Any]:
        """
        What was fitted and what degraded — for the experiment write-up.

        ``fallback_reason`` is the field that must be printed beside any
        number this detector produced: a supervised row that silently fell
        back to the unsupervised combiner is not a measurement of supervision.
        """
        if not self.is_fitted_:
            raise RuntimeError("report() called before fit()")
        return {
            "name": self.name,
            "budget": self.budget,
            "combiner": self.combiner,
            "learner": self.learner,
            "sampler": self.sampler,
            "conditional_folds": self.conditional_folds,
            "seed": self.seed,
            "top_fraction": self.top_fraction,
            "n_labelled": self.n_labelled_,
            "n_positive_labels": self.n_positive_labels_,
            "used_supervision": self.used_supervision_,
            "fallback_reason": self.fallback_reason_,
            "coefficients": self.coefficients_,
            "feature_names": list(FEATURE_NAMES),
            "oracle_queries": (
                None if self.label_oracle is None else self.label_oracle.n_queries
            ),
            "feature_notes": list(self.features_.notes) if self.features_ else [],
        }

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _features_for(self, dirty_df: pd.DataFrame, profile: Any) -> CellFeatures:
        """Cached :func:`build_features`. The cache key includes every setting
        that can change a feature value, so a cache hit is always sound."""
        key = FeatureCache.fingerprint(dirty_df) + (
            self.seed,
            self.iqr_k,
            self.flag_statistical_outliers,
            tuple(sorted((str(k), str(v)) for k, v in self.conditional_kwargs.items())),
            self.conditional_folds,
            profile is not None,
            str(getattr(profile, "name", "")),
        )
        return self.feature_cache.get_or_build(
            key,
            lambda: build_features(
                dirty_df,
                profile=profile,
                seed=self.seed,
                iqr_k=self.iqr_k,
                flag_statistical_outliers=self.flag_statistical_outliers,
                conditional_kwargs=self.conditional_kwargs,
                conditional_folds=self.conditional_folds,
            ),
        )

    def _unsupervised_scores(self, features: CellFeatures) -> np.ndarray:
        """The label-free combination. Also the fallback and the sampler's guide."""
        rule = features.column("r_score")
        cond = features.column("c_full")
        if self.combiner == "rule_only":
            return rule.copy()
        if self.combiner == "cond_only":
            return cond.copy()
        if self.combiner == "noisy_or":
            r = np.clip(rule, 0.0, 1.0)
            c = np.clip(cond, 0.0, 1.0)
            return 1.0 - (1.0 - r) * (1.0 - c)
        return 0.5 * (normalised_ranks(rule) + normalised_ranks(cond))

    def _score_vector(self, features: CellFeatures) -> np.ndarray:
        if self._model is None:
            return self._unsupervised_scores(features)
        x = self._standardise(features.x)
        proba = self._model.predict_proba(x)
        # A classifier fitted on 5 labels can still end up single-class if the
        # sample was degenerate; guard rather than index blindly.
        classes = list(getattr(self._model, "classes_", [0, 1]))
        if len(classes) < 2:  # pragma: no cover - guarded earlier in fit()
            return self._unsupervised_scores(features)
        return np.asarray(proba[:, classes.index(1)], dtype=float)

    def _ranked_local(
        self, scores: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Local positions sorted by ``scores`` descending, ties broken by a
        seeded permutation.

        Never by array order: cells are enumerated column-major, so array order
        correlates with the column and hence, on adult, with the label.
        """
        shuffle = rng.permutation(scores.size)
        return shuffle[np.argsort(-scores[shuffle], kind="stable")]

    def _trivially_missing(self, features: CellFeatures) -> np.ndarray:
        """
        Cells both signals call "missing", which a label would not teach
        anything about.

        A missing cell is scored 1.0 by the rule block (``r_missing``) and by
        the conditional block (``ConditionalDetector.nan_score``), so its
        label is already implied by a feature the learner can read for free.
        On adult that distinction is decisive: ~6.5k cells are blank and none
        of them is corrupted (adult's corruption is income label flips), so a
        sampler that labels the top-scoring cells spends its whole budget on
        guaranteed negatives. Excluding them from the *labelling* pool uses no
        labels and no ground truth — only the two signals' own output.

        Heuristic, and stated as one: ``r_missing == 1`` is exact, while
        ``c_full == nan_score`` is an equality test against the explicit NaN
        policy (the Cauchy residual tail never actually reaches 1.0, so a
        modelled cell does not collide with it).
        """
        nan_score = float(self.conditional_kwargs.get("nan_score", 1.0))
        return (features.column("r_missing") > 0.0) | (
            features.column("c_full") >= nan_score - 1e-12
        )

    def _per_column_informed(
        self, train_idx: np.ndarray, n_top: int, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Informed picks allocated **round-robin across columns**: each pass
        takes every column's next-best unlabelled, not-trivially-missing cell.

        WHY PER COLUMN — THE MEASURED REASON
        ====================================
        A global ranking is dominated by whichever column has the widest score
        distribution, which need not be a column that contains corruption. On
        adult, both a combiner-led and a signal-diversified global sampler draw
        **zero positives even at k = 100** (measured): the top of the global
        ranking is the rule detector's `fnlwgt` IQR false positives and the
        blank cells of `workclass` / `occupation`, every one of which is
        genuinely clean, because adult's corruption is confined to `income`.

        Per-column allocation is also the mechanism behind the literature
        result this route is built on. Raha's 2 → 20 label jump
        (``docs/literature.md`` factor 2) comes from clustering **per
        column** and spending the budget across those clusters, not from
        labelling a global top-k.

        It is not free. On breast_cancer, where corruption is spread over
        several columns at 7.5 % prevalence, the global top is already
        positive-rich and per-column allocation is *worse* for k <= 20
        (0.6326 against 0.7318 at k=5). It is the default on worst-case
        regret only; :meth:`_sample_budget` carries the full measured table.

        Ordering within a column is by the unsupervised combiner. Comparing
        scores *within* one column removes the cross-column scale problem that
        makes the global ranking unreliable in the first place.
        """
        assert self.features_ is not None
        n_rows = self.features_.shape[0]
        column_of = train_idx // max(n_rows, 1)

        eligible = ~self._trivially_missing(self.features_)[train_idx]
        if not eligible.any():
            eligible = np.ones(train_idx.size, dtype=bool)

        scores = self._unsupervised_scores(self.features_)[train_idx]
        ranked = self._ranked_local(scores, rng)  # global order, seeded ties

        # Per column, its eligible cells best-first.
        queues: Dict[int, List[int]] = {}
        for local in ranked:
            if not eligible[local]:
                continue
            queues.setdefault(int(column_of[local]), []).append(int(local))

        order = sorted(queues)  # frame column order: deterministic
        chosen: List[int] = []
        cursor = {col: 0 for col in order}
        while len(chosen) < n_top:
            progressed = False
            for col in order:
                if len(chosen) >= n_top:
                    break
                position = cursor[col]
                if position < len(queues[col]):
                    chosen.append(queues[col][position])
                    cursor[col] = position + 1
                    progressed = True
            if not progressed:  # every column exhausted
                break
        return np.asarray(chosen, dtype=int)

    def _sample_budget(self, train_idx: np.ndarray) -> np.ndarray:
        """
        Pick which train cells to label. Deterministic given
        ``(seed, train_idx, features)``.

        ``sampler="per_column"`` (default)
            The informed share is allocated round-robin across columns; see
            :meth:`_per_column_informed`, which has the measured reason.
        ``sampler="diverse"``
            The informed share is split round-robin across the signals
            *separately* (``r_score``, then ``c_full``), over cells that are
            not trivially missing; the rest is uniform random. The motivation
            was the route's own premise — the two signals disagree about
            which cells matter — but it is still a *global* ranking per
            signal, and it does not rescue adult.
        ``sampler="top_combiner"``
            The informed share goes to the top of the unsupervised combiner,
            with no exclusions. The obvious thing to do; it never learns
            anything on adult, and it is the best choice on breast_cancer for
            k <= 20. An ablation, not a straw man.
        """
        assert self.features_ is not None
        if self.budget == "full":
            return train_idx
        k = min(int(self.budget), int(train_idx.size))
        rng = np.random.default_rng(self.seed)
        n_top = min(k, int(math.ceil(k * self.top_fraction)))
        chosen: List[int] = []

        if self.sampler == "top_combiner":
            scores = self._unsupervised_scores(self.features_)[train_idx]
            chosen = list(self._ranked_local(scores, rng)[:n_top])
        elif self.sampler == "per_column":
            chosen = list(self._per_column_informed(train_idx, n_top, rng))
        else:
            eligible = ~self._trivially_missing(self.features_)[train_idx]
            if not eligible.any():
                # Everything is missing; there is no better pool to draw from.
                eligible = np.ones(train_idx.size, dtype=bool)
            eligible_local = np.flatnonzero(eligible)
            n_signals = len(_SIGNAL_FEATURES)
            # Even split, remainder to the earlier signals.
            quotas = [
                n_top // n_signals + (1 if i < n_top % n_signals else 0)
                for i in range(n_signals)
            ]
            taken: set = set()
            for feature, quota in zip(_SIGNAL_FEATURES, quotas):
                if quota <= 0:
                    continue
                signal = self.features_.column(feature)[train_idx][eligible_local]
                for local in eligible_local[self._ranked_local(signal, rng)]:
                    if len(taken) >= sum(quotas):
                        break
                    if int(local) not in taken:
                        taken.add(int(local))
                        quota -= 1
                        if quota <= 0:
                            break
            chosen = sorted(taken)

        remaining = np.setdiff1d(
            np.arange(train_idx.size), np.asarray(chosen, dtype=int)
        )
        n_rand = min(k - len(chosen), remaining.size)
        rand = (
            rng.choice(remaining, size=n_rand, replace=False)
            if n_rand > 0
            else np.empty(0, int)
        )

        chosen_local = np.concatenate((np.asarray(chosen, dtype=int), rand)).astype(int)
        return np.sort(train_idx[chosen_local])

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        if self._scaler_mean is None or self._scaler_scale is None:
            return x
        return (x - self._scaler_mean) / self._scaler_scale

    def _fit_meta(
        self,
        x: np.ndarray,
        train_idx: np.ndarray,
        chosen: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Fit the meta-classifier on the ``len(y)`` labelled rows.

        Standardisation statistics come from the **train cells' features**, not
        from the labelled sample (5 rows give a meaningless scale) and not
        from the whole table (that would read eval-cell features at fit time).
        Feature values are unsupervised throughout — no label is involved in
        the scaling.
        """
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression

        train_x = x[train_idx]
        self._scaler_mean = train_x.mean(axis=0)
        scale = train_x.std(axis=0)
        scale[scale < 1e-12] = 1.0
        self._scaler_scale = scale

        x_fit = self._standardise(x[chosen])
        if self.learner == "gbm":
            model: Any = HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=3,
                learning_rate=0.1,
                min_samples_leaf=5,
                early_stopping=False,
                random_state=self.seed,
            )
        else:
            model = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=self.seed,
            )
        model.fit(x_fit, y)
        self._model = model

        if self.learner == "logreg":
            coef = np.asarray(model.coef_, dtype=float).ravel()
            self.coefficients_ = {
                name: float(w) for name, w in zip(FEATURE_NAMES, coef)
            }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"EnsembleDetector(budget={self.budget!r}, combiner={self.combiner!r}, "
            f"learner={self.learner!r}, seed={self.seed})"
        )


def _validate_budget(budget: Union[int, str]) -> Union[int, str]:
    if isinstance(budget, str):
        if budget != "full":
            raise ValueError(f"budget must be an int >= 0 or 'full', got {budget!r}")
        return "full"
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)):
        raise TypeError(f"budget must be an int >= 0 or 'full', got {budget!r}")
    if budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget}")
    return int(budget)
