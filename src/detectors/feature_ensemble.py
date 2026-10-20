"""
src/detectors/feature_ensemble.py
---------------------------------
W8-C — **Raha-style per-cell feature-vector ensemble** (`docs/w8_plan.md` §4).

WHY THIS EXISTS
===============
Raha (SIGMOD 2019) is the strongest published system on these benchmarks, and
its core architecture is *not* a clever single detector. It runs many
configurations of many base detectors, treats their outputs as a **feature
vector per cell**, and trains a classifier on that vector. ED2 and HoloDetect
add representation at three *scopes*: attribute-level, tuple-level and
dataset-level features.

`docs/w8_plan.md` §2 records those as factors **F3** (many weak detectors as a
feature vector, then classify) and **F4** (three representation scopes), and
records that this project has never done either. Wave 2 did noisy-OR fusion
over two detectors, which is strictly weaker: a fixed combination rule cannot
learn that (say) a high conditional score matters only in a low-cardinality
column, which is exactly the kind of interaction a tree ensemble picks up.

So: assemble every signal this repo already produces into one per-cell vector,
and classify.

THE FEATURE VECTOR
==================
26 features in three scopes. The *cell* scope is the per-cell evidence; the
*attribute* and *tuple* scopes are the F4 context.

cell (16)
    ``cond_full`` / ``cond_marginal`` / ``cond_ratio``
        The conditional detector's three modes. One model fitted in ``full``
        mode scores all three without refitting (see
        ``ConditionalDetector.score_cells``), so this costs one fit.
    ``q_full`` / ``q_marginal`` / ``q_ratio``
        The **within-column percentile rank** of each of the above, in
        [0, 1]. These are the cross-column-comparable versions; see the
        COMPARABILITY section, which is not optional advice in this project.
    ``z_full``
        Within-column standardised ``cond_full``, clipped to +-10. A second,
        scale-sensitive view of the same quantity: the percentile rank knows
        the *order* of a column's scores but not whether the top score is far
        from the rest, and "far from the rest" is the outlier signal.
    ``rule_score``
        The Phase 3 rule detector's ``detection_score`` at this cell, 0.0 if
        no rule fired. Identical by construction to
        ``RuleDetectorAdapter.score_cells`` (asserted in the tests).
    ``rule_<method>`` x 7
        One binary per rule check -- ``missing_value``, ``format_check``,
        ``range_check``, ``iqr_outlier``, ``vocab_check``,
        ``whitespace_check``, ``format_check_mixed``. The adapter collapses
        the six/seven checks into one scalar whose magnitude is an arbitrary
        severity constant; separating them lets the classifier learn that
        (for example) ``vocab_check`` is informative on hospital and pure
        noise on adult, which a single scalar cannot express.
    ``is_missing``
        1.0 where the cell is null. Kept explicit and separable so a trivial
        missing-token rule cannot quietly take credit for the method
        (the same discipline as ``ConditionalDetector.nan_score``).

attribute (6)
    ``col_card_ratio``   distinct non-null values / rows -- dimensionless
    ``col_rare_frac``    fraction of cells whose value occurs in < 1 % of rows
    ``col_missing_rate`` fraction of null cells
    ``col_is_numeric``   1.0 if >= 80 % of non-null values parse as floats
    ``col_mean_full``    the column's mean ``cond_full``
    ``col_std_full``     the column's standard deviation of ``cond_full``

tuple (4)
    ``row_frac_high``    fraction of the row's cells with ``q_full`` > 0.9
    ``row_mean_q_full``  the row's mean ``q_full``
    ``row_max_q_full``   the row's max ``q_full``
    ``row_missing_frac`` fraction of the row's cells that are null

Every tuple-level feature is a *fraction*, not a count, so a 20-column table
and a 7-column table put them on the same scale.

COMPARABILITY -- READ THIS BEFORE ADDING A FEATURE
==================================================
Two earlier mechanisms in this project died on cross-column confounds (W5.3 on
cardinality, W6.1 on coverage). Raw detector scores carry column-specific
units: a categorical column's ``cond_full`` is one minus a point probability
mass, a numeric column's is one minus a tail mass, and the two are
commensurable in *range and direction only*, never in measure (see
``src/detectors/conditional.py``). Pooling raw scores across columns therefore
ranks columns by their score *scale* as much as by their contamination.

The mitigation here is to carry both views: the raw score, and the
within-column percentile rank, which is by construction uniform on [0, 1] for
*every* column regardless of cardinality or dtype. ``tests/test_feature_ensemble.py``
asserts the rank features' distributions match across a 2-value column and a
2000-value column, and that assertion fails if someone replaces a rank with a
raw score.

``col_card_ratio`` and ``col_rare_frac`` are deliberately *in* the vector --
Raha and ED2 both use attribute-level features, and withholding them would be
testing a weaker architecture than the published one. The risk that they
become the whole signal is handled by measurement, not by omission: the
experiment reports permutation importances and a ``cell_only`` ablation that
drops both scopes.

TRAINING LABELS -- SYNTHETIC CORRUPTION
=======================================
There are no real labels available to the detector (that is the whole point of
the research program). So the classifier is trained on corruption we inject
ourselves, with ``src/datagen/corruptor.py``:

    aug_df = dirty_df + injected corruption of several types
    y = 1 at injected positions, 0 elsewhere
    train on a row-disjoint 70 % of aug_df
    score the REAL dirty_df, evaluate against REAL labels

The two frames are different, and the two label sources are different, so
nothing is trained on the thing it scores.

Three honest caveats, which the report must carry:

1. **The negatives are not clean.** ``dirty_df`` already contains real
   corruption (adult 0.25 % of cells, breast_cancer 7.5 %), and those cells
   are labelled 0 in training. The classifier is therefore trained to call
   real corruption *negative*. This biases it against exactly what we want.
2. **The features shift.** Features on ``aug_df`` come from a conditional
   model fitted on ``aug_df``; features on ``dirty_df`` come from a model
   fitted on ``dirty_df``. The *procedure* is identical, the fitted models are
   not.
3. **The injected policy is designed, not learned.** HoloDetect's ablation
   (0.953 F1 for learned augmentation vs 0.166 for random transformations) is
   the pre-registered risk, recorded in `docs/w8_plan.md` §2 F2. In particular
   adult's real corruption is entirely one-directional (1,701 flips of
   ``income`` from the 75 % majority class to the 24 % minority), while
   ``swap_valid`` is frequency-weighted. A transfer gap there is expected and
   is one of the more informative things this experiment can measure.

LEAKAGE (CLAUDE.md §3 -- non-negotiable)
========================================
Nothing in this module may see clean data.

- The conditional detector is fitted with ``profile=None``.
- The rule detector *needs* a profile, so the caller supplies a
  ``profile_factory`` which this module calls with a **dirty-derived frame in
  both the clean and the dirty position** -- use :func:`dirty_only_profile`.
  Building the profile from clean data imports clean distributional knowledge
  and is worth up to 0.93 F1 on some datasets
  (``results/w5_raha_leakage_check.md``), so it is not a technicality.
- Synthetic corruption is injected into the dirty frame, never the clean one.
- No method of this class accepts, or is derived from, clean data. The tests
  assert that by signature scan and by a spy that fails if any frame reaching
  the classifier differs from a dirty-derived one.

Clean data is read **only afterwards**, by the experiment, via
``src/detection_eval/labels.py::cell_labels``, to score what already exists.

CELL ORDER
==========
Feature rows are in **column-major** order -- all of column 0's rows, then
column 1's -- byte-identical to
``src/detection_eval/harness.py::flatten_cells``, so a score vector produced
here lines up with a flattened label frame without any further bookkeeping.

DETERMINISM
===========
Given the same ``dirty_df`` and ``seed``, the feature matrix and the scores are
reproducible. The injection is seeded per cell by the corruptor, the
conditional models are seeded, the classifier is seeded, and the row split is
drawn from a seeded generator.

Worked example
--------------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": list(range(40)), "b": ["x", "y"] * 20})
    >>> ens = FeatureEnsembleDetector(seed=42, injection_rate=0.05)
    >>> _ = ens.fit(df)                              # doctest: +SKIP
    >>> scores = ens.score_cells(df)                 # doctest: +SKIP
    >>> scores.shape                                 # doctest: +SKIP
    (40, 2)
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.schema import DatasetPaths
from src.detection.detector import Detector

logger = logging.getLogger(__name__)

__all__ = [
    "RULE_METHODS",
    "FEATURE_GROUPS",
    "FeatureMatrix",
    "dirty_only_profile",
    "extract_features",
    "FeatureEnsembleDetector",
]


#: Every ``detection_method`` string ``src/detection/detector.py`` can emit.
#: Hard-coded rather than discovered so that a new check in the rule detector
#: shows up as a test failure (an unknown method name) instead of silently
#: vanishing from the feature vector.
RULE_METHODS: Tuple[str, ...] = (
    "missing_value",
    "format_check",
    "range_check",
    "iqr_outlier",
    "vocab_check",
    "whitespace_check",
    "format_check_mixed",
)

#: Fraction of rows below which a value counts as "rare" for
#: ``col_rare_frac``. Matches the profiler's own rare-category threshold.
RARE_THRESHOLD = 0.01

#: ``q_full`` above this counts as "high" for ``row_frac_high``.
HIGH_QUANTILE = 0.9

#: Fraction of non-null values that must parse as finite floats for
#: ``col_is_numeric``. Matches the profiler's MIXED threshold.
NUMERIC_PARSE_THRESHOLD = 0.8

#: Default injection mix. Several types in one augmented frame so the
#: classifier sees heterogeneous corruption from a single conditional fit.
DEFAULT_INJECTION_TYPES: Tuple[str, ...] = (
    "missing",
    "format",
    "out_of_range",
    "swap_valid",
    "typo",
)

_CELL_FEATURES: Tuple[str, ...] = (
    "cond_full",
    "cond_marginal",
    "cond_ratio",
    "q_full",
    "q_marginal",
    "q_ratio",
    "z_full",
    "rule_score",
) + tuple(f"rule_{m}" for m in RULE_METHODS) + ("is_missing",)

_ATTRIBUTE_FEATURES: Tuple[str, ...] = (
    "col_card_ratio",
    "col_rare_frac",
    "col_missing_rate",
    "col_is_numeric",
    "col_mean_full",
    "col_std_full",
)

_TUPLE_FEATURES: Tuple[str, ...] = (
    "row_frac_high",
    "row_mean_q_full",
    "row_max_q_full",
    "row_missing_frac",
)

#: scope -> feature names. The experiment's ablations are subsets of this.
FEATURE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "cell": _CELL_FEATURES,
    "attribute": _ATTRIBUTE_FEATURES,
    "tuple": _TUPLE_FEATURES,
}

FEATURE_NAMES: Tuple[str, ...] = (
    _CELL_FEATURES + _ATTRIBUTE_FEATURES + _TUPLE_FEATURES
)

#: feature name -> its scope, for importance tables and ablations.
FEATURE_SCOPE: Dict[str, str] = {
    name: scope for scope, names in FEATURE_GROUPS.items() for name in names
}


# ═══════════════════════════════════════════════════════════════════════════
# Feature matrix container
# ═══════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass(frozen=True)
class FeatureMatrix:
    """
    A per-cell feature matrix in column-major cell order.

    Attributes
    ----------
    x : np.ndarray, shape (n_cells, n_features), float64
        Guaranteed finite -- :func:`extract_features` asserts it.
    names : tuple[str, ...]
        Feature names, aligned with ``x``'s columns.
    cell_columns : np.ndarray, shape (n_cells,), object
        Which table column each row of ``x`` came from.
    cell_rows : np.ndarray, shape (n_cells,), int
        Which zero-based table row position each row of ``x`` came from.
    n_rows, n_columns : int
        Shape of the frame the features were extracted from.
    """

    x: np.ndarray
    names: Tuple[str, ...]
    cell_columns: np.ndarray
    cell_rows: np.ndarray
    n_rows: int
    n_columns: int

    def subset(self, names: Sequence[str]) -> "FeatureMatrix":
        """A copy keeping only ``names``, in the order given."""
        idx = [self.names.index(n) for n in names]
        return dataclasses.replace(
            self, x=self.x[:, idx], names=tuple(names)
        )

    def as_frame(self) -> pd.DataFrame:
        """``x`` as a labelled frame -- for inspection, not for the model."""
        return pd.DataFrame(self.x, columns=list(self.names))

    def to_cell_frame(self, values: np.ndarray, like: pd.DataFrame) -> pd.DataFrame:
        """
        Fold a per-cell vector (in this matrix's order) back into a frame
        shaped like ``like``.
        """
        values = np.asarray(values, dtype=float).ravel()
        if values.size != self.x.shape[0]:
            raise ValueError(
                f"expected {self.x.shape[0]} values, got {values.size}"
            )
        out = pd.DataFrame(
            0.0, index=like.index.copy(), columns=like.columns, dtype=float
        )
        n = self.n_rows
        for j, col in enumerate(like.columns):
            out[col] = values[j * n : (j + 1) * n]
        return out


# ═══════════════════════════════════════════════════════════════════════════
# Small helpers
# ═══════════════════════════════════════════════════════════════════════════


def dirty_only_profile(name: str, dirty_df: pd.DataFrame, paths: Any = None):
    """
    Build a :class:`DatasetProfile` from the **dirty** frame alone.

    ``profile_dataset`` takes a (clean, dirty) pair; this passes ``dirty_df``
    in *both* positions, which is the whole point. A profile built from clean
    data imports clean quartiles, valid ranges and vocabularies, and this
    project has measured that advantage at up to 0.93 F1
    (``results/w5_raha_leakage_check.md``). Using it here would make every
    number in W8-C incomparable with the leakage-controlled figures.
    """
    from src.data.profiler import profile_dataset

    if paths is None:
        paths = DatasetPaths(
            name=name,
            clean_path=__import__("pathlib").Path(f"<dirty-only:{name}>"),
            dirty_path=__import__("pathlib").Path(f"<dirty-only:{name}>"),
        )
    return profile_dataset(name, paths, dirty_df, dirty_df)


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    """
    Within-column percentile rank in (0, 1], ties averaged.

    Uniform on (0, 1] for every column by construction, which is what makes
    it cross-column comparable. An all-constant column maps to all 1.0 --
    every value is tied at the top, which is the correct answer for "where in
    this column's distribution does this value sit" when the distribution is
    a point mass, and it carries no discriminative information either way.
    """
    s = pd.Series(np.asarray(values, dtype=float))
    return (s.rank(method="average", pct=True)).to_numpy(dtype=float)


def _numeric_fraction(series: pd.Series) -> float:
    """Fraction of non-null values that parse as finite floats."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0
    ok = 0
    for v in non_null.to_numpy():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            ok += 1
    return ok / len(non_null)


def _rare_fraction(series: pd.Series, threshold: float = RARE_THRESHOLD) -> float:
    """
    Fraction of *cells* whose value occurs in fewer than ``threshold`` of rows.

    Not the fraction of rare *categories* -- the fraction of cells sitting on
    one. A column of 2,000 distinct values each appearing once has rare
    fraction 1.0; a 2-value column has 0.0. This is the quantity falsification
    condition 4 asks us to correlate against.
    """
    n = len(series)
    if n == 0:
        return 0.0
    counts = series.value_counts(dropna=True)
    if counts.empty:
        return 0.0
    rare_values = set(counts.index[counts < threshold * n])
    if not rare_values:
        return 0.0
    return float(series.isin(rare_values).mean())


def _rule_arrays(
    dirty_df: pd.DataFrame, detector: Optional[Detector]
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    """
    Run the Phase 3 rule detector once and return

        (score_matrix, {method: binary_matrix}, unknown_methods)

    all shaped ``dirty_df.shape``. ``score_matrix`` is built with exactly the
    rule ``RuleDetectorAdapter`` uses -- write ``detection_score`` at
    ``(row_id, column)``, 0.0 elsewhere, last record wins -- so the two agree
    cell for cell; ``tests/test_feature_ensemble.py`` asserts the equality
    rather than taking this comment's word for it.

    ``unknown_methods`` names any ``detection_method`` not in
    :data:`RULE_METHODS`; the caller surfaces it instead of dropping it.
    """
    shape = dirty_df.shape
    score = np.zeros(shape, dtype=float)
    flags = {m: np.zeros(shape, dtype=float) for m in RULE_METHODS}
    unknown: List[str] = []
    if detector is None:
        return score, flags, unknown

    col_pos = {c: i for i, c in enumerate(dirty_df.columns)}
    n_rows = shape[0]
    for rec in detector.detect(dirty_df):
        j = col_pos.get(rec.column)
        i = int(rec.row_id)
        if j is None or not (0 <= i < n_rows):
            continue
        score[i, j] = float(rec.detection_score)
        method = str(rec.detection_method)
        if method in flags:
            flags[method][i, j] = 1.0
        elif method not in unknown:
            unknown.append(method)
    return score, flags, unknown


# ═══════════════════════════════════════════════════════════════════════════
# Feature extraction
# ═══════════════════════════════════════════════════════════════════════════


def extract_features(
    dirty_df: pd.DataFrame,
    cond_detector: Any,
    rule_detector: Optional[Detector] = None,
) -> Tuple[FeatureMatrix, Dict[str, Any]]:
    """
    Build the per-cell feature matrix for ``dirty_df``.

    Parameters
    ----------
    dirty_df : pd.DataFrame
        The frame to describe. The ONLY data source. No clean data.
    cond_detector : ConditionalDetector
        Already fitted, in ``mode="full"`` so that all three modes can be
        scored from the one fit.
    rule_detector : Detector | None
        The Phase 3 rule detector, built from a **dirty-only** profile. When
        ``None`` the eight rule features are all zero and the returned info
        dict says so.

    Returns
    -------
    (FeatureMatrix, info)
        ``info`` carries the three mode score frames (for the baselines the
        falsification conditions require) and any unknown rule method names.

    Raises
    ------
    ValueError
        A feature came out non-finite. This is a hard error, not a fill:
        a silent NaN in a tree ensemble's input changes the split structure
        without changing the output shape, so it must never pass quietly.
    """
    n_rows, n_cols = dirty_df.shape
    if n_rows == 0 or n_cols == 0:
        raise ValueError(f"dirty_df is empty: shape {dirty_df.shape}")

    mode_frames = {
        mode: cond_detector.score_cells(dirty_df, mode=mode)
        for mode in ("full", "marginal", "ratio")
    }
    for mode, frame in mode_frames.items():
        if list(frame.columns) != list(dirty_df.columns):
            raise ValueError(f"mode {mode!r} score frame columns do not match")

    rule_score, rule_flags, unknown_methods = _rule_arrays(dirty_df, rule_detector)
    if unknown_methods:
        logger.warning(
            "rule detector emitted method(s) absent from RULE_METHODS: %s",
            unknown_methods,
        )

    # ── per-cell, per-column blocks ───────────────────────────────────────
    blocks: Dict[str, List[np.ndarray]] = {name: [] for name in FEATURE_NAMES}
    missing_matrix = np.zeros((n_rows, n_cols), dtype=float)
    q_full_matrix = np.zeros((n_rows, n_cols), dtype=float)

    for j, col in enumerate(dirty_df.columns):
        series = dirty_df[col]
        raw = {m: mode_frames[m][col].to_numpy(dtype=float) for m in mode_frames}
        raw = {m: np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0)
               for m, v in raw.items()}

        q = {m: _percentile_rank(v) for m, v in raw.items()}
        q_full_matrix[:, j] = q["full"]

        mean_full = float(np.mean(raw["full"]))
        std_full = float(np.std(raw["full"]))
        if std_full > 0:
            z = np.clip((raw["full"] - mean_full) / std_full, -10.0, 10.0)
        else:
            z = np.zeros(n_rows, dtype=float)

        is_missing = series.isna().to_numpy().astype(float)
        missing_matrix[:, j] = is_missing

        blocks["cond_full"].append(raw["full"])
        blocks["cond_marginal"].append(raw["marginal"])
        blocks["cond_ratio"].append(raw["ratio"])
        blocks["q_full"].append(q["full"])
        blocks["q_marginal"].append(q["marginal"])
        blocks["q_ratio"].append(q["ratio"])
        blocks["z_full"].append(z)
        blocks["rule_score"].append(rule_score[:, j])
        for m in RULE_METHODS:
            blocks[f"rule_{m}"].append(rule_flags[m][:, j])
        blocks["is_missing"].append(is_missing)

        ones = np.ones(n_rows, dtype=float)
        n_distinct = int(series.nunique(dropna=True))
        blocks["col_card_ratio"].append(ones * (n_distinct / max(n_rows, 1)))
        blocks["col_rare_frac"].append(ones * _rare_fraction(series))
        blocks["col_missing_rate"].append(ones * float(is_missing.mean()))
        blocks["col_is_numeric"].append(
            ones * (1.0 if _numeric_fraction(series) >= NUMERIC_PARSE_THRESHOLD
                    else 0.0)
        )
        blocks["col_mean_full"].append(ones * mean_full)
        blocks["col_std_full"].append(ones * std_full)

    # ── tuple scope: computed across the whole row, then broadcast ────────
    row_frac_high = (q_full_matrix > HIGH_QUANTILE).mean(axis=1)
    row_mean_q = q_full_matrix.mean(axis=1)
    row_max_q = q_full_matrix.max(axis=1)
    row_missing = missing_matrix.mean(axis=1)
    for _ in range(n_cols):
        blocks["row_frac_high"].append(row_frac_high)
        blocks["row_mean_q_full"].append(row_mean_q)
        blocks["row_max_q_full"].append(row_max_q)
        blocks["row_missing_frac"].append(row_missing)

    x = np.column_stack(
        [np.concatenate(blocks[name]) for name in FEATURE_NAMES]
    ).astype(float)

    if not np.all(np.isfinite(x)):
        bad = [
            FEATURE_NAMES[j]
            for j in range(x.shape[1])
            if not np.all(np.isfinite(x[:, j]))
        ]
        raise ValueError(f"non-finite values in feature(s): {bad}")

    cell_columns = np.concatenate(
        [np.full(n_rows, col, dtype=object) for col in dirty_df.columns]
    )
    cell_rows = np.concatenate([np.arange(n_rows) for _ in dirty_df.columns])

    matrix = FeatureMatrix(
        x=x,
        names=FEATURE_NAMES,
        cell_columns=cell_columns,
        cell_rows=cell_rows,
        n_rows=n_rows,
        n_columns=n_cols,
    )
    info = {
        "mode_frames": mode_frames,
        "unknown_rule_methods": tuple(unknown_methods),
        "rule_detector_used": rule_detector is not None,
    }
    return matrix, info


# ═══════════════════════════════════════════════════════════════════════════
# The detector
# ═══════════════════════════════════════════════════════════════════════════


class FeatureEnsembleDetector:
    """
    Per-cell feature-vector ensemble, trained on synthetic corruption.

    See the module docstring for the architecture, the feature list, the
    comparability discipline, the training-label caveats and the leakage
    guarantee.

    Parameters
    ----------
    seed : int
        Seeds the injection, the conditional models, the row split and the
        classifier. Scores are reproducible given ``(dirty_df, seed)``.
    injection_types : sequence of str
        Corruption types injected in sequence into one augmented frame. Each
        must be one of ``src.datagen.corruptor.CORRUPTION_TYPES``.
    injection_rate : float
        Per type, as a fraction of **all** cells (the corruptor's own unit).
        The default 0.01 over five types gives a training positive rate of at
        most 5 %, comfortably above the real prevalences we must detect.
    train_fraction : float
        Fraction of the augmented frame's **rows** used for fitting the
        classifier. The rest is held out for the sanity check. The split is
        by row, not by cell, so the tuple-level features of a training cell
        never describe a held-out row.
    feature_names : sequence of str | None
        Feature subset to train on. ``None`` uses all of
        :data:`FEATURE_NAMES`. This is how the experiment runs its scope
        ablations without recomputing features.
    max_iter, learning_rate, max_leaf_nodes, min_samples_leaf, l2_regularization
        Passed to ``sklearn.ensemble.HistGradientBoostingClassifier``.
    profile_factory : callable | None
        ``df -> DatasetProfile``, used to build the rule detector for a frame.
        MUST build from the dirty frame only -- pass
        ``lambda df: dirty_only_profile(name, df)``. ``None`` disables the
        rule features (they are then all zero and ``report()`` says so).
    flag_statistical_outliers, iqr_k : bool, float
        Rule-detector settings; mirror ``src/pipeline/runner.py``'s
        per-dataset defaults so the rule features are the rule detector the
        project actually ships.
    conditional_kwargs : dict | None
        Extra keyword arguments for :class:`ConditionalDetector`.

    Attributes (set by ``fit``)
    ---------------------------
    classifier_ : HistGradientBoostingClassifier
    train_features_, train_labels_ : FeatureMatrix, np.ndarray
        The augmented frame's features and injected labels (all rows).
    train_row_mask_ : np.ndarray of bool, shape (n_rows,)
        Which rows of the augmented frame the classifier was fitted on.
    injection_reports_ : list[InjectionReport]
    holdout_scores_, holdout_labels_ : np.ndarray
        The sanity check's material: classifier output and injected labels on
        the held-out rows.
    is_fitted_ : bool
    """

    def __init__(
        self,
        seed: int = 42,
        injection_types: Sequence[str] = DEFAULT_INJECTION_TYPES,
        injection_rate: float = 0.01,
        train_fraction: float = 0.7,
        feature_names: Optional[Sequence[str]] = None,
        max_iter: int = 150,
        learning_rate: float = 0.1,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        profile_factory: Optional[Callable[[pd.DataFrame], Any]] = None,
        flag_statistical_outliers: bool = True,
        iqr_k: float = 3.0,
        conditional_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not injection_types:
            raise ValueError("injection_types must not be empty")
        if not 0.0 < injection_rate < 1.0:
            raise ValueError(
                f"injection_rate must be in (0, 1), got {injection_rate}"
            )
        if not 0.0 < train_fraction < 1.0:
            raise ValueError(
                f"train_fraction must be in (0, 1), got {train_fraction}"
            )
        names = tuple(FEATURE_NAMES if feature_names is None else feature_names)
        unknown = [n for n in names if n not in FEATURE_NAMES]
        if unknown:
            raise ValueError(f"unknown feature name(s): {unknown}")
        if not names:
            raise ValueError("feature_names must not be empty")

        self.seed = int(seed)
        self.injection_types = tuple(injection_types)
        self.injection_rate = float(injection_rate)
        self.train_fraction = float(train_fraction)
        self.feature_names = names
        self.max_iter = int(max_iter)
        self.learning_rate = float(learning_rate)
        self.max_leaf_nodes = int(max_leaf_nodes)
        self.min_samples_leaf = int(min_samples_leaf)
        self.l2_regularization = float(l2_regularization)
        self.profile_factory = profile_factory
        self.flag_statistical_outliers = bool(flag_statistical_outliers)
        self.iqr_k = float(iqr_k)
        self.conditional_kwargs = dict(conditional_kwargs or {})

        self.classifier_: Any = None
        self.train_features_: Optional[FeatureMatrix] = None
        self.train_labels_: Optional[np.ndarray] = None
        self.train_row_mask_: Optional[np.ndarray] = None
        self.injection_reports_: List[Any] = []
        self.holdout_scores_: Optional[np.ndarray] = None
        self.holdout_labels_: Optional[np.ndarray] = None
        self.score_detector_: Any = None
        self.score_info_: Dict[str, Any] = {}
        self.augmented_frame_: Optional[pd.DataFrame] = None
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _build_rule_detector(self, df: pd.DataFrame) -> Optional[Detector]:
        """Rule detector for ``df``, from a profile built from ``df`` alone."""
        if self.profile_factory is None:
            return None
        profile = self.profile_factory(df)
        if profile is None:
            return None
        return Detector(
            profile,
            iqr_k=self.iqr_k,
            flag_statistical_outliers=self.flag_statistical_outliers,
        )

    def _fit_conditional(self, df: pd.DataFrame) -> Any:
        """A ``ConditionalDetector`` fitted on ``df`` with ``profile=None``."""
        from src.detectors.conditional import ConditionalDetector

        det = ConditionalDetector(
            mode="full", seed=self.seed, **self.conditional_kwargs
        )
        # profile=None is the leakage guarantee: the detector then identifies
        # ID-like columns from dirty cardinality alone.
        det.fit(df, profile=None)
        return det

    def compute_features(
        self, df: pd.DataFrame
    ) -> Tuple[FeatureMatrix, Dict[str, Any]]:
        """
        Fit a conditional model on ``df``, build the rule detector from a
        dirty-only profile of ``df``, and return ``(FeatureMatrix, info)``.

        Public because feature extraction is the expensive half (one
        conditional fit plus one rule pass) and the scope ablations in
        ``experiments/w8c_ensemble.py`` must reuse one extraction rather than
        repeat it three times — otherwise the variants would be compared on
        separately-fitted features instead of byte-identical ones.
        """
        cond = self._fit_conditional(df)
        rules = self._build_rule_detector(df)
        matrix, info = extract_features(df, cond, rules)
        info["conditional_detector"] = cond
        info["skipped_columns"] = dict(getattr(cond, "skipped_columns_", {}) or {})
        return matrix, info

    #: Back-compatible private alias.
    _features_for = compute_features

    def train_matrix_split(self) -> Tuple[FeatureMatrix, np.ndarray, np.ndarray]:
        """
        ``(train_features_, train_labels_, cell_in_train_mask)``.

        The cell mask is the row mask expanded to cells, which is what a
        classifier refit on a feature subset needs.
        """
        if not self.is_fitted_:
            raise RuntimeError("train_matrix_split() called before fit()")
        assert self.train_features_ is not None
        assert self.train_labels_ is not None
        assert self.train_row_mask_ is not None
        return (
            self.train_features_,
            self.train_labels_,
            self.train_row_mask_[self.train_features_.cell_rows],
        )

    def make_classifier(self):
        """An unfitted classifier with exactly this instance's settings."""
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=False,
            random_state=self.seed,
        )

    def _augment(self, dirty_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Inject each type in turn into ``dirty_df``; return
        ``(augmented_frame, label_matrix)``.

        Labels are the **union** of the injected positions. Chaining means a
        later type can land on a cell an earlier one already changed; the
        union is still exactly "this cell was altered by us", which is the
        label we train on.
        """
        from src.datagen.corruptor import inject

        current = dirty_df.reset_index(drop=True)
        labels = np.zeros(current.shape, dtype=bool)
        col_pos = {c: i for i, c in enumerate(current.columns)}
        self.injection_reports_ = []

        for offset, ctype in enumerate(self.injection_types):
            # A distinct seed per type, so two types do not draw the same
            # position permutation and pile onto the same cells.
            corrupted, report = inject(
                current,
                ctype,
                self.injection_rate,
                seed=self.seed + 1000 * (offset + 1),
            )
            for row, col in report.positions:
                labels[int(row), col_pos[col]] = True
            self.injection_reports_.append(report)
            current = corrupted

        return current, labels

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit(
        self, dirty_df: pd.DataFrame, profile: Any = None
    ) -> "FeatureEnsembleDetector":
        """
        Train the ensemble on synthetic corruption injected into ``dirty_df``.

        Parameters
        ----------
        dirty_df : pd.DataFrame
            The dirty table. The ONLY data source.
        profile : ignored
            Accepted for ``DetectorProtocol`` parity. Deliberately unused: a
            ``DatasetProfile`` is built from clean data, and letting one in
            here would break the leakage guarantee. Use ``profile_factory``,
            which this class calls with a dirty frame.

        Returns
        -------
        self
        """
        if not isinstance(dirty_df, pd.DataFrame):
            raise TypeError(f"dirty_df must be a DataFrame, got {type(dirty_df)!r}")
        if dirty_df.shape[0] == 0 or dirty_df.shape[1] == 0:
            raise ValueError(f"dirty_df is empty: shape {dirty_df.shape}")

        aug_df, label_matrix = self._augment(dirty_df)
        self.augmented_frame_ = aug_df
        n_pos = int(label_matrix.sum())
        if n_pos == 0:
            raise RuntimeError(
                "no synthetic corruption could be injected into this frame, so "
                "there are no training positives; the ensemble cannot be fitted."
            )

        matrix, _ = self.compute_features(aug_df)
        y = np.concatenate(
            [label_matrix[:, j] for j in range(aug_df.shape[1])]
        ).astype(int)
        self.train_features_ = matrix
        self.train_labels_ = y

        # Row-disjoint split. Cells of a training row never appear in the
        # holdout, so the tuple-level features cannot carry a row across.
        rng = np.random.default_rng(self.seed)
        n_rows = aug_df.shape[0]
        row_mask = np.zeros(n_rows, dtype=bool)
        n_train_rows = max(1, int(round(self.train_fraction * n_rows)))
        row_mask[rng.permutation(n_rows)[:n_train_rows]] = True
        self.train_row_mask_ = row_mask

        cell_in_train = row_mask[matrix.cell_rows]
        sub = matrix.subset(self.feature_names)
        x_train, y_train = sub.x[cell_in_train], y[cell_in_train]
        if len(np.unique(y_train)) < 2:
            raise RuntimeError(
                "the training rows contain only one class "
                f"({int(y_train.sum())} positives of {y_train.size}); "
                "raise injection_rate or train_fraction."
            )

        self.classifier_ = self.make_classifier().fit(x_train, y_train)

        holdout = ~cell_in_train
        self.holdout_scores_ = self.classifier_.predict_proba(sub.x[holdout])[:, 1]
        self.holdout_labels_ = y[holdout]

        self.is_fitted_ = True
        logger.info(
            "FeatureEnsembleDetector fitted: %d injected positives of %d cells "
            "(%d features, %d train rows of %d)",
            n_pos, y.size, len(self.feature_names), n_train_rows, n_rows,
        )
        return self

    def score_cells(
        self, dirty_df: pd.DataFrame, profile: Any = None  # noqa: ARG002
    ) -> pd.DataFrame:
        """
        Score every cell of ``dirty_df``. Higher = more likely corrupted.

        Features are recomputed on ``dirty_df`` with a conditional model
        fitted on ``dirty_df`` -- the same *procedure* as in training, not the
        same fitted model (module docstring, caveat 2).

        Returns
        -------
        pd.DataFrame
            float64 in [0, 1], same shape/index/columns as ``dirty_df``.
        """
        if not self.is_fitted_:
            raise RuntimeError("score_cells() called before fit()")

        matrix, info = self.compute_features(dirty_df)
        self.score_info_ = info
        self.score_detector_ = info["conditional_detector"]
        sub = matrix.subset(self.feature_names)
        proba = self.classifier_.predict_proba(sub.x)[:, 1]
        self.score_features_ = matrix
        return matrix.to_cell_frame(proba, dirty_df)

    def fit_score(self, dirty_df: pd.DataFrame, profile: Any = None) -> pd.DataFrame:
        """``fit(...)`` then ``score_cells(...)`` on the same dirty frame."""
        return self.fit(dirty_df, profile=profile).score_cells(dirty_df)

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    def permutation_importance(
        self,
        matrix: Optional[FeatureMatrix] = None,
        y: Optional[np.ndarray] = None,
        n_repeats: int = 3,
        sample: int = 50_000,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Permutation importance as the **drop in ROC-AUC**, per feature.

        Measured on the held-out rows of the augmented frame by default, so a
        feature the classifier merely memorised scores nothing. ROC-AUC
        rather than accuracy because the positive rate is a few percent and
        accuracy is nearly constant there.

        Returns a name -> mean AUC drop mapping; larger means more important,
        and a negative value means permuting the feature *helped*, which is
        noise and is reported as measured.
        """
        if not self.is_fitted_:
            raise RuntimeError("permutation_importance() called before fit()")
        from src.detection_eval.metrics import roc_auc

        if matrix is None or y is None:
            assert self.train_features_ is not None
            assert self.train_labels_ is not None
            assert self.train_row_mask_ is not None
            keep = ~self.train_row_mask_[self.train_features_.cell_rows]
            matrix = dataclasses.replace(
                self.train_features_,
                x=self.train_features_.x[keep],
                cell_columns=self.train_features_.cell_columns[keep],
                cell_rows=self.train_features_.cell_rows[keep],
            )
            y = self.train_labels_[keep]

        sub = matrix.subset(self.feature_names)
        x = sub.x
        y = np.asarray(y, dtype=int)
        rng = np.random.default_rng(self.seed if seed is None else seed)
        if 0 < sample < x.shape[0]:
            idx = rng.choice(x.shape[0], size=sample, replace=False)
            x, y = x[idx], y[idx]
        if len(np.unique(y)) < 2:
            return {name: float("nan") for name in self.feature_names}

        base = roc_auc(y, self.classifier_.predict_proba(x)[:, 1])
        if base is None:
            return {name: float("nan") for name in self.feature_names}

        out: Dict[str, float] = {}
        for j, name in enumerate(self.feature_names):
            drops = []
            for _ in range(n_repeats):
                xp = x.copy()
                xp[:, j] = xp[rng.permutation(xp.shape[0]), j]
                a = roc_auc(y, self.classifier_.predict_proba(xp)[:, 1])
                drops.append(float(base) - (0.5 if a is None else float(a)))
            out[name] = float(np.mean(drops))
        return out

    def report(self) -> Dict[str, Any]:
        """What was fitted -- for experiment write-ups."""
        if not self.is_fitted_:
            raise RuntimeError("report() called before fit()")
        assert self.train_labels_ is not None
        return {
            "seed": self.seed,
            "injection_types": list(self.injection_types),
            "injection_rate": self.injection_rate,
            "n_features": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "n_train_cells": int(self.train_labels_.size),
            "n_injected_positives": int(self.train_labels_.sum()),
            "injected_rate": float(self.train_labels_.mean()),
            "rule_features_active": self.profile_factory is not None,
            "injection": [r.to_dict() for r in self.injection_reports_],
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "fitted" if self.is_fitted_ else "unfitted"
        return (
            f"FeatureEnsembleDetector(seed={self.seed}, "
            f"n_features={len(self.feature_names)}, {state})"
        )
