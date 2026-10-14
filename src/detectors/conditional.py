"""
src/detectors/conditional.py
----------------------------
Conditional-model error detection (research route R2 of docs/research_program.md).

WHY THIS EXISTS
===============
The project's hand-written six-rule detector scores F1 = 0.000 on Adult. The
reason is structural, not a bug: Adult's corruption replaces a *valid* income
label with the *other valid* income label. Every syntactic check the rule
detector owns -- missing token, out-of-range, IQR fence, format, novel
category, whitespace -- asks "is this value legal?". A label flip is legal.

The only way to see such a corruption is to stop asking whether the value is
legal *marginally* and start asking whether it is probable *conditionally*:

    P(cell value | all the other values in the row)

An income of ">50K" on a row whose education, occupation and hours-per-week
all point at "<=50K" is perfectly valid marginally and highly improbable
conditionally. That gap is the detection signal.

THE METHOD
==========
For every column C of the dirty table, fit a model that predicts C from all
the other columns, using the dirty table alone. Then score each cell of C by
how implausible its *observed* value is under that model.

    categorical / low-cardinality C:  plausibility = P(observed value | row)
    numerical C:                      plausibility = P(|residual| >= |observed residual|)

    suspicion score = 1 - plausibility      (higher = more likely corrupted)

Both plausibilities live in [0, 1] and both answer "how unsurprising is what
we actually saw", which is what makes the two column types comparable on one
scale. The asymmetry is real and is stated plainly: the categorical number is
a *point* probability mass, the numerical number is a *tail* mass. They are
commensurable in range and direction, not in measure. Per-column score
distributions therefore still differ, and a global (whole-table) PR-AUC mixes
them; experiments/r2_conditional.py reports per-column PR-AUC alongside the
global number so this is visible rather than hidden.

For numerics the tail uses a Gaussian residual model with a robust scale:

    scale = 1.4826 * MAD(train residuals)        (MAD -> sigma for a Gaussian)
    z     = (observed - predicted) / scale
    plausibility = erfc(|z| / sqrt(2))           (two-sided Gaussian tail)
    score        = 1 - plausibility = erf(|z| / sqrt(2))

THREE MODES -- THE ABLATION
===========================
The headline claim is that *conditioning on the row* is what buys the edge.
That claim is only meaningful against a control, so this one class implements
three scoring modes and the experiment runs all of them:

    mode="full"      score = 1 - P(value | rest of row)          [the method]
    mode="marginal"  score = 1 - P(value)                        [the control]
                     categorical: 1 - relative frequency
                     numerical:   robust z-score, same erf squash
    mode="ratio"     score = sigmoid( log P(value) - log P(value | row) )

"marginal" is the honest control. If "full" does not beat "marginal" then
conditioning is *not* the factor carrying the edge, and we need to know that.

"ratio" exists because the fundamental statement of the idea is "valid
marginally, improbable conditionally" -- that is a *difference* of two
log-probabilities, not a single one. A value that is rare both ways (a rare
but genuine country) scores low under the ratio, while a common value in the
wrong row scores high. Dividing the marginal rarity out also makes the score
more comparable across columns, because each column's own baseline rarity
cancels. It is one extra line of code over the two required modes and it
isolates the factor directly, so it is reported too.

WHAT IT DOES NOT DO
===================
- It cannot see corruption in a column that nothing else in the row predicts.
  A pure-noise column is undetectable by construction, and the test suite
  asserts the detector correctly *fails* there (PR-AUC near prevalence)
  instead of flagging everything.
- It skips ID-like columns (cardinality ~ row count) and constant columns,
  scoring them 0.0. Corruption inside an identifier is a different problem
  (no conditional structure exists to model), and including such a column as
  a feature only injects noise. On Breast Cancer this forfeits the 107
  corruptions that live in 'Unnamed: 0' and 'Sample code number'. That is a
  stated cost, not an oversight.
- It trains on dirty data, so the corruptions themselves are in the training
  signal. This is fine while corruption is rare (Adult: 0.25% of cells) and
  progressively self-defeating as prevalence rises (Breast Cancer: 7.5%).
- A missing cell has no observed value to assess, so it is handled by a
  separate explicit policy (``nan_score``, default 1.0) rather than by the
  model. This is the cheap syntactic missing-token rule, NOT conditional
  modelling, and conflating the two would let a trivial rule take credit for
  the method. Smart Factory's corruption is *exactly* its NaN cells, so its
  score is entirely this policy; the experiment therefore also reports a
  "non-missing cells only" slice, where the conditional contribution stands
  alone.

LEAKAGE (CLAUDE.md Section 3 -- non-negotiable)
===============================================
SELF-SUPERVISED ONLY. ``fit`` and ``score_cells`` receive the dirty frame and
nothing else. No parameter of this class is named or derived from clean data,
no clean value is ever compared against, and the fitted estimators' training
input is a deterministic function of ``dirty_df`` and ``train_mask`` alone.
``tests/test_conditional_detector.py`` asserts both halves of that: a
signature scan for any parameter containing "clean", and a sentinel spy that
proves eval rows never reach an estimator's ``fit``.

The optional ``profile`` argument is accepted for interface compatibility with
``src/detection/detector.py`` and is used for *one* thing only: honouring
``ColumnKind.ID_LIKE`` when deciding which columns to skip. Because
``DatasetProfile`` is built from ``clean_df`` (the documented soft-leakage
caveat in CLAUDE.md Section 3), passing it inherits that caveat. The default
is ``None`` and experiments/r2_conditional.py passes ``None``: ID-like columns
are then identified from dirty cardinality alone, so the reported numbers are
fully self-supervised.

DETERMINISM
===========
Given the same ``dirty_df``, ``train_mask`` and ``seed``, scores are bitwise
reproducible. ``early_stopping=False`` is set deliberately: sklearn's
automatic early stopping (which switches on above 10k samples) carves a random
validation split out of the training data, which would both introduce a second
source of randomness and silently shrink the training set.

WORKED EXAMPLE
==============
    >>> import pandas as pd
    >>> from src.detectors.conditional import ConditionalDetector
    >>> df = pd.DataFrame({"a": [1, 2, 3] * 40, "b": [2, 4, 6] * 40})
    >>> df.loc[0, "b"] = 6                       # valid b, impossible for a=1
    >>> det = ConditionalDetector(mode="full", seed=42).fit(df)
    >>> scores = det.score_cells(df)
    >>> bool(scores.loc[0, "b"] > scores.loc[3, "b"])
    True
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

logger = logging.getLogger(__name__)

# Scoring modes. See the module docstring for what each one isolates.
MODES: Tuple[str, ...] = ("full", "marginal", "ratio")

# Residual tail families for the numeric branch. See _residual_tail.
RESIDUAL_TAILS: Tuple[str, ...] = ("cauchy", "gaussian")

# Role assigned to each column after fit().
ROLE_CLASSIFIER = "classifier"
ROLE_REGRESSOR = "regressor"
ROLE_SKIPPED = "skipped"

# 1 / Phi^{-1}(3/4): converts a median-absolute-deviation into a Gaussian sigma.
_MAD_TO_SIGMA = 1.4826

# sklearn's HistGradientBoosting* native categorical support requires codes in
# [0, max_bins), and max_bins defaults to 255.
_SKLEARN_MAX_BINS = 255


# ═══════════════════════════════════════════════════════════════════════════════
# Per-column fitted state
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ColumnModel:
    """
    Everything learned about one column of the dirty table.

    One of these exists per column after ``fit``, including for skipped
    columns (role == ``"skipped"``), so that ``report()`` can state exactly
    what happened to every column rather than silently omitting some.

    Attributes
    ----------
    column : str
        Column name.
    role : str
        ``"classifier"``, ``"regressor"`` or ``"skipped"``.
    reason : str
        Human-readable justification for the role. For skipped columns this is
        the skip reason ("id_like", "constant", ...).
    n_train : int
        Rows actually handed to the estimator (train rows with a non-null
        target), 0 when skipped.
    is_categorical_target : bool
        True when the target was treated as a discrete label set.
    numeric_target : bool
        True when the raw column dtype was numeric. Recorded at fit time so
        that scoring canonicalises observed values exactly the way fitting
        did, even if a later frame reports a different-but-equivalent dtype.
    classes : list | None
        Observed training values, in sorted order, for a classifier target.
    residual_scale : float | None
        Robust Gaussian sigma of training residuals, for a regressor target.
    marginal_freq : dict | None
        Relative frequency of each training value, for a categorical target.
        Used by ``mode="marginal"`` and ``mode="ratio"``.
    marginal_center, marginal_scale : float | None
        Robust median / sigma of the raw training values, for a numeric target.
    """

    column: str
    role: str
    reason: str
    n_train: int = 0
    is_categorical_target: bool = False
    numeric_target: bool = False
    classes: Optional[List[Any]] = None
    residual_scale: Optional[float] = None
    marginal_freq: Optional[Dict[Any, float]] = None
    marginal_center: Optional[float] = None
    marginal_scale: Optional[float] = None

    # Fitted sklearn estimator and the feature columns it was given.
    # Excluded from to_dict() -- not serialisable and not a reported result.
    estimator: Any = field(default=None, repr=False, compare=False)
    feature_columns: Optional[List[str]] = field(default=None, repr=False, compare=False)

    @property
    def is_skipped(self) -> bool:
        return self.role == ROLE_SKIPPED

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict view suitable for JSON / logging (no estimator)."""
        return {
            "column": self.column,
            "role": self.role,
            "reason": self.reason,
            "n_train": self.n_train,
            "is_categorical_target": self.is_categorical_target,
            "numeric_target": self.numeric_target,
            "n_classes": len(self.classes) if self.classes is not None else None,
            "residual_scale": self.residual_scale,
            "marginal_center": self.marginal_center,
            "marginal_scale": self.marginal_scale,
            "n_features": len(self.feature_columns) if self.feature_columns else 0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _is_na(value: Any) -> bool:
    """True for None / NaN / pd.NA. Mirrors ValueComparator's NA handling."""
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, bool):
        return result
    return bool(np.all(result))


def _canonical_key(value: Any, numeric_column: bool) -> Any:
    """
    Hashable, type-stable key for one observed value.

    Numeric columns keep their float value; everything else becomes ``str``
    *without* stripping, because surrounding whitespace is itself one of the
    corruption kinds this project detects and stripping would erase it.
    """
    if numeric_column:
        return float(value)
    return str(value)


def _robust_scale(values: np.ndarray, eps: float) -> float:
    """
    Gaussian-equivalent sigma from the median absolute deviation.

    Falls back to the standard deviation when the MAD is degenerate (more than
    half the values identical), and to ``eps`` when both are zero, so that a
    constant residual stream yields z = 0 rather than a division by zero.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return eps
    mad = float(np.median(np.abs(finite - np.median(finite))))
    scale = _MAD_TO_SIGMA * mad
    if scale <= eps:
        scale = float(np.std(finite))
    if not np.isfinite(scale) or scale <= eps:
        scale = eps
    return scale


def _residual_tail(z: np.ndarray, tail: str) -> np.ndarray:
    """
    Two-sided tail mass P(|Z| >= |z|) -- the numeric analogue of
    "probability of the observed value": the plausibility of a residual at
    least this extreme. ``1 - tail`` is the suspicion score.

    tail="cauchy" (default)
        1 - (2/pi) * arctan(|z|). Decays like 1/|z|, so it never saturates in
        float64 and keeps a usable ranking out to |z| ~ 1e15.
    tail="gaussian"
        erfc(|z| / sqrt(2)). The textbook choice, but it falls below 1e-17
        once |z| exceeds about 8.3, at which point the reported score
        ``1 - tail`` rounds to exactly 1.0 in float64. Every strong outlier
        then collapses onto the identical score and ranking resolution inside
        the numeric columns is gone -- fatal for PR-AUC, which is purely a
        ranking metric. The residual scale here comes from *in-sample*
        residuals and so is biased small, which pushes |z| into that saturated
        region routinely.

    Both are monotone in |z| and bounded in [0, 1]. The default is "cauchy"
    for the resolution reason above; the choice is measured, not assumed, and
    the measurement is reported in results/r2_conditional.md.
    """
    z_arr = np.asarray(z, dtype=float)
    if z_arr.size == 0:
        return z_arr.astype(float)
    abs_z = np.abs(z_arr)
    with np.errstate(invalid="ignore"):
        if tail == "cauchy":
            out = 1.0 - (2.0 / math.pi) * np.arctan(abs_z)
        elif tail == "gaussian":
            out = np.vectorize(math.erfc, otypes=[float])(abs_z / math.sqrt(2.0))
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(f"unknown residual_tail {tail!r}")
    out = np.asarray(out, dtype=float)
    out[~np.isfinite(z_arr)] = np.nan
    return out


def _is_integer_like(non_null: pd.Series) -> bool:
    """
    Could this column be an identifier?

    Near-unique cardinality alone is not enough: a continuous measurement is
    also near-unique (600 draws from a normal distribution give 600 distinct
    values), and skipping such a column as "ID-like" would throw away a
    perfectly modellable numeric column. Identifiers are integer-valued or
    non-numeric; a column carrying genuine fractional values is a measurement.

    Non-numeric columns return True because a string identifier (Breast
    Cancer's ``Sample code number``) must still be caught.

    Known limitation: an integer measurement that happens to be near-unique
    would be skipped. Adult's ``fnlwgt`` is the closest case in this project
    and sits at 26,741/45,222 = 0.59, well under the 0.9 ratio, so it is
    modelled.
    """
    if not pd.api.types.is_numeric_dtype(non_null):
        return True
    values = pd.to_numeric(non_null, errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return True
    return bool(np.all(finite == np.round(finite)))


def _numeric_parse_fraction(series: pd.Series) -> float:
    """Fraction of non-null values that parse as a finite float."""
    non_null = series[~series.isna()]
    if len(non_null) == 0:
        return 0.0
    coerced = pd.to_numeric(non_null, errors="coerce")
    return float(coerced.notna().mean())


# ═══════════════════════════════════════════════════════════════════════════════
# The detector
# ═══════════════════════════════════════════════════════════════════════════════


class ConditionalDetector:
    """
    Self-supervised per-cell error detector based on conditional modelling.

    Fits one gradient-boosted model per column predicting that column from the
    others, using the dirty table alone, then scores each cell by how
    implausible its observed value is. See the module docstring for the method,
    the three-mode ablation, the leakage guarantee and the known limitations.

    Parameters
    ----------
    mode : str
        ``"full"`` (conditional, the method), ``"marginal"`` (rarity only, the
        control) or ``"ratio"`` (marginal minus conditional log-plausibility).
        ``"marginal"`` fits no estimators, so it is also the cheap fallback.
    seed : int
        Random state for the estimators. Scores are reproducible given
        (dirty_df, train_mask, seed).
    max_classes : int
        A column with at most this many distinct non-null training values is
        modelled as a classifier; otherwise, if it parses as numeric, as a
        regressor. Default 50. On Adult this routes ``income`` (2 values) and
        ``education`` (16) to classifiers and ``age`` (74) / ``fnlwgt``
        (26,741) to regressors.
    max_iter, learning_rate, max_depth, min_samples_leaf, l2_regularization
        Passed straight to the sklearn estimators.
    nan_score : float
        Score given to a cell whose observed value is missing, in every mode.
        Default 1.0. A missing cell has no value to assess conditionally; this
        is the trivial missing-token rule and is kept explicit and separable
        (see the module docstring) rather than being attributed to the model.
    residual_tail : str
        Residual tail family for numeric columns: ``"cauchy"`` (default,
        non-saturating) or ``"gaussian"``. See ``_residual_tail`` -- the
        Gaussian tail underflows to 0 past |z| ~ 8.3 and collapses all strong
        numeric outliers onto the same score, which costs ranking resolution.
    id_unique_ratio, id_min_unique : float, int
        A column is treated as ID-like, and skipped, when it has more than
        ``id_min_unique`` distinct non-null values AND
        distinct / non-null >= ``id_unique_ratio`` AND its values are
        integer-valued or non-numeric (a column with genuine fractional
        values is a measurement, not an identifier). Defaults 0.9 / 50, which
        catch Breast Cancer's ``Unnamed: 0`` (658/680) and ``Sample code
        number`` (623/681) while leaving Adult's ``fnlwgt`` (26,741/45,222 =
        0.59) modelled.
    numeric_parse_threshold : float
        Fraction of non-null values that must parse as finite floats before a
        high-cardinality column is modelled as a regressor. Default 0.8,
        matching the profiler's MIXED threshold.
    min_train_rows : int
        Minimum usable training rows for a column to be modelled at all.
    eps : float
        Probability floor, used to keep ``log`` finite in ``mode="ratio"`` and
        to guard division by a zero scale.

    Attributes (set by ``fit``)
    ---------------------------
    column_models_ : dict[str, ColumnModel]
    skipped_columns_ : dict[str, str]      column -> skip reason
    feature_columns_ : list[str]           columns usable as predictors
    n_train_rows_ : int
    is_fitted_ : bool
    """

    def __init__(
        self,
        mode: str = "full",
        seed: int = 42,
        max_classes: int = 50,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        nan_score: float = 1.0,
        residual_tail: str = "cauchy",
        id_unique_ratio: float = 0.9,
        id_min_unique: int = 50,
        numeric_parse_threshold: float = 0.8,
        min_train_rows: int = 20,
        eps: float = 1e-6,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if not 0.0 <= nan_score <= 1.0:
            raise ValueError(f"nan_score must be in [0, 1], got {nan_score}")
        if residual_tail not in RESIDUAL_TAILS:
            raise ValueError(
                f"residual_tail must be one of {RESIDUAL_TAILS}, got {residual_tail!r}"
            )
        if max_classes < 2:
            raise ValueError(f"max_classes must be >= 2, got {max_classes}")
        if not 0.0 < eps < 1.0:
            raise ValueError(f"eps must be in (0, 1), got {eps}")

        self.mode = mode
        self.seed = seed
        self.max_classes = max_classes
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.nan_score = nan_score
        self.residual_tail = residual_tail
        self.id_unique_ratio = id_unique_ratio
        self.id_min_unique = id_min_unique
        self.numeric_parse_threshold = numeric_parse_threshold
        self.min_train_rows = min_train_rows
        self.eps = eps

        # Fitted state
        self.column_models_: Dict[str, ColumnModel] = {}
        self.skipped_columns_: Dict[str, str] = {}
        self.feature_columns_: List[str] = []
        self.n_train_rows_: int = 0
        self.is_fitted_: bool = False

        # Encoding spec, learned on train rows only:
        #   column -> (treat_as_categorical, categories | None)
        self._encoding: Dict[str, Tuple[bool, Optional[List[Any]]]] = {}
        self._columns: List[str] = []

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        dirty_df: pd.DataFrame,
        profile: Any = None,
        train_mask: Optional[Sequence[bool]] = None,
    ) -> "ConditionalDetector":
        """
        Learn one per-column model from the dirty table.

        Parameters
        ----------
        dirty_df : pd.DataFrame
            The dirty table. This is the ONLY data source. No clean data,
            no labels.
        profile : DatasetProfile | None
            Optional, and used for one thing only: honouring
            ``ColumnKind.ID_LIKE`` when choosing columns to skip. Since the
            profile is built from clean data, passing it inherits the
            documented soft-leakage caveat (CLAUDE.md Section 3). Default
            ``None`` keeps the detector fully self-supervised.
        train_mask : sequence of bool | None
            Row mask selecting the fitting rows, positionally aligned with
            ``dirty_df``. ``None`` means all rows. Rows outside the mask are
            never passed to any estimator and never contribute to any
            statistic (categories, frequencies, medians, residual scales).

        Returns
        -------
        self
        """
        if not isinstance(dirty_df, pd.DataFrame):
            raise TypeError(f"dirty_df must be a DataFrame, got {type(dirty_df)!r}")
        if dirty_df.shape[1] == 0:
            raise ValueError("dirty_df has no columns")

        mask = self._resolve_mask(dirty_df, train_mask)
        train_df = dirty_df.loc[mask]
        self.n_train_rows_ = int(len(train_df))
        self._columns = list(dirty_df.columns)

        if self.n_train_rows_ == 0:
            raise ValueError("train_mask selects zero rows")

        # ── 1. Decide the role of every column, from the TRAIN slice only ──
        id_like_hint = self._id_like_from_profile(profile)
        roles: Dict[str, Tuple[str, str]] = {}
        for col in self._columns:
            roles[col] = self._decide_role(train_df[col], col, id_like_hint)

        self.skipped_columns_ = {
            c: reason for c, (role, reason) in roles.items() if role == ROLE_SKIPPED
        }

        # ── 2. Build the encoding spec and the shared feature matrix ───────
        self.feature_columns_ = self._choose_feature_columns(train_df, roles)
        self._encoding = {
            col: self._encoding_for(train_df[col]) for col in self.feature_columns_
        }
        x_all, cat_mask = self._encode(dirty_df)
        x_train = x_all[mask.to_numpy(), :]

        # ── 3. Fit one model per non-skipped column ────────────────────────
        for col in self._columns:
            role, reason = roles[col]
            if role == ROLE_SKIPPED:
                self.column_models_[col] = ColumnModel(
                    column=col, role=role, reason=reason
                )
                continue
            self.column_models_[col] = self._fit_column(
                col=col,
                role=role,
                reason=reason,
                target_train=train_df[col],
                x_train=x_train,
                cat_mask=cat_mask,
            )

        # Some columns can only be found unfittable once fitting is attempted
        # (a classifier whose train slice collapsed to one class, a column with
        # no other column to condition on). Refresh the skip list so report()
        # and skipped_columns_ describe what actually happened.
        self.skipped_columns_ = {
            c: m.reason for c, m in self.column_models_.items() if m.is_skipped
        }

        self.is_fitted_ = True
        logger.info(
            "ConditionalDetector(mode=%s) fitted on %d rows: %d modelled, %d skipped",
            self.mode,
            self.n_train_rows_,
            sum(1 for m in self.column_models_.values() if not m.is_skipped),
            len(self.skipped_columns_),
        )
        return self

    def score_cells(
        self,
        dirty_df: pd.DataFrame,
        profile: Any = None,  # noqa: ARG002 - interface parity with fit()
        mode: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Score every cell of ``dirty_df``. Higher = more likely corrupted.

        Parameters
        ----------
        dirty_df : pd.DataFrame
            Frame to score. Must have the same columns as the fitting frame.
        profile : ignored
            Accepted for interface parity with ``fit``.
        mode : str | None
            Scoring mode override. ``None`` uses the mode the detector was
            constructed with. A detector fitted in ``"full"`` mode holds the
            conditional estimators *and* the marginal statistics, so it can
            score all three modes; this lets the ablation compare modes
            against byte-identical fitted models instead of refitting, which
            is both cheaper and a tighter comparison. A detector fitted in
            ``"marginal"`` mode has no estimators and can only score
            ``"marginal"``.

        Returns
        -------
        pd.DataFrame
            Float frame with exactly the index and columns of ``dirty_df``.
            Values lie in [0, 1]. Skipped columns are all-zero.
        """
        if not self.is_fitted_:
            raise RuntimeError("score_cells() called before fit()")
        if list(dirty_df.columns) != self._columns:
            raise ValueError(
                "score_cells() received different columns than fit(): "
                f"expected {self._columns}, got {list(dirty_df.columns)}"
            )
        active_mode = self.mode if mode is None else mode
        if active_mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {active_mode!r}")
        if active_mode != "marginal" and self.mode == "marginal":
            raise RuntimeError(
                f"cannot score in mode {active_mode!r}: this detector was fitted "
                "with mode='marginal' and holds no conditional estimators"
            )

        needs_model = active_mode in ("full", "ratio")
        x_all, _ = self._encode(dirty_df) if needs_model else (None, None)

        scores = pd.DataFrame(
            0.0, index=dirty_df.index.copy(), columns=dirty_df.columns, dtype=float
        )
        for col in self._columns:
            model = self.column_models_[col]
            if model.is_skipped:
                continue
            scores[col] = self._score_column(
                model, dirty_df[col], x_all, active_mode
            )
        return scores

    def fit_score(
        self,
        dirty_df: pd.DataFrame,
        profile: Any = None,
        train_mask: Optional[Sequence[bool]] = None,
    ) -> pd.DataFrame:
        """Convenience: ``fit(...)`` then ``score_cells(dirty_df)``."""
        return self.fit(dirty_df, profile=profile, train_mask=train_mask).score_cells(
            dirty_df
        )

    def report(self) -> Dict[str, Any]:
        """Summary of what was fitted -- for logging and experiment write-ups."""
        if not self.is_fitted_:
            raise RuntimeError("report() called before fit()")
        return {
            "mode": self.mode,
            "seed": self.seed,
            "residual_tail": self.residual_tail,
            "n_train_rows": self.n_train_rows_,
            "n_columns": len(self._columns),
            "n_modelled": sum(
                1 for m in self.column_models_.values() if not m.is_skipped
            ),
            "skipped_columns": dict(self.skipped_columns_),
            "feature_columns": list(self.feature_columns_),
            "columns": {c: m.to_dict() for c, m in self.column_models_.items()},
        }

    # ------------------------------------------------------------------ #
    # Role decisions                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_mask(
        dirty_df: pd.DataFrame, train_mask: Optional[Sequence[bool]]
    ) -> pd.Series:
        """Normalise ``train_mask`` into a boolean Series on dirty_df's index."""
        if train_mask is None:
            return pd.Series(True, index=dirty_df.index)
        arr = np.asarray(train_mask)
        if arr.dtype != bool:
            raise TypeError(f"train_mask must be boolean, got dtype {arr.dtype}")
        if arr.shape != (len(dirty_df),):
            raise ValueError(
                f"train_mask must have length {len(dirty_df)}, got {arr.shape}"
            )
        return pd.Series(arr, index=dirty_df.index)

    @staticmethod
    def _id_like_from_profile(profile: Any) -> set:
        """ID-like column names taken from a DatasetProfile, if one was given."""
        if profile is None:
            return set()
        names = getattr(profile, "id_columns", None)
        return set(names) if names else set()

    def _decide_role(
        self, train_series: pd.Series, col: str, id_like_hint: set
    ) -> Tuple[str, str]:
        """
        Decide whether a column is modelled as a classifier, a regressor, or
        skipped -- using the training slice of the dirty data only.
        """
        non_null = train_series[~train_series.isna()]
        n_non_null = len(non_null)
        n_unique = int(non_null.nunique())

        if n_non_null == 0:
            return ROLE_SKIPPED, "all_missing"
        if n_unique < 2:
            return ROLE_SKIPPED, "constant"
        if col in id_like_hint:
            return ROLE_SKIPPED, "id_like_profile"
        if (
            n_unique > self.id_min_unique
            and n_unique / n_non_null >= self.id_unique_ratio
            and _is_integer_like(non_null)
        ):
            return (
                ROLE_SKIPPED,
                f"id_like_cardinality({n_unique}/{n_non_null})",
            )
        if n_non_null < self.min_train_rows:
            return ROLE_SKIPPED, f"too_few_train_rows({n_non_null})"

        if n_unique <= self.max_classes:
            return ROLE_CLASSIFIER, f"n_unique={n_unique}<=max_classes"

        parse_frac = _numeric_parse_fraction(train_series)
        if parse_frac >= self.numeric_parse_threshold:
            return ROLE_REGRESSOR, f"numeric_parse_fraction={parse_frac:.3f}"

        return (
            ROLE_SKIPPED,
            f"high_cardinality_non_numeric(n_unique={n_unique}, "
            f"parse={parse_frac:.3f})",
        )

    def _choose_feature_columns(
        self, train_df: pd.DataFrame, roles: Dict[str, Tuple[str, str]]
    ) -> List[str]:
        """
        Columns usable as predictors.

        Dropped: skipped columns (ID-like columns are high-entropy noise and
        would blow past sklearn's 255-bin categorical limit; constant columns
        carry no information), and any non-numeric column with more than 255
        distinct training values, for the same bin limit.
        """
        features: List[str] = []
        for col in self._columns:
            if roles[col][0] == ROLE_SKIPPED:
                continue
            series = train_df[col]
            if pd.api.types.is_numeric_dtype(series):
                features.append(col)
                continue
            n_unique = int(series[~series.isna()].nunique())
            if n_unique > _SKLEARN_MAX_BINS:
                logger.info(
                    "Column %r dropped from features: %d categories > %d bins",
                    col,
                    n_unique,
                    _SKLEARN_MAX_BINS,
                )
                continue
            features.append(col)
        return features

    # ------------------------------------------------------------------ #
    # Encoding                                                             #
    # ------------------------------------------------------------------ #

    def _encoding_for(
        self, train_series: pd.Series
    ) -> Tuple[bool, Optional[List[Any]]]:
        """
        Encoding spec for one feature column, learned on train rows.

        Numeric dtypes pass through as floats. Everything else is **ordinal
        encoded** against the sorted training categories and declared to
        sklearn as a native categorical feature. Ordinal codes impose an
        arbitrary order, which would be wrong for a linear model; it is
        harmless here because HistGradientBoosting's native categorical
        support splits on *subsets* of category values rather than on
        thresholds. Keeping the raw tokens as categories (instead of coercing
        to numbers) also preserves format-error strings such as Breast
        Cancer's ``'1a0'`` as their own category, which is exactly the kind of
        value we want visible.

        Unseen categories and missing values both encode to NaN, which
        HistGradientBoosting handles natively.
        """
        if pd.api.types.is_numeric_dtype(train_series):
            return False, None
        cats = sorted(
            {str(v) for v in train_series[~train_series.isna()].to_numpy()}
        )
        return True, cats

    def _encode(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Build the shared float feature matrix and its categorical mask."""
        n_rows = len(df)
        n_feat = len(self.feature_columns_)
        x = np.full((n_rows, n_feat), np.nan, dtype=float)
        cat_mask = np.zeros(n_feat, dtype=bool)

        for j, col in enumerate(self.feature_columns_):
            is_cat, cats = self._encoding[col]
            series = df[col]
            if not is_cat:
                x[:, j] = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
                continue
            cat_mask[j] = True
            # Vectorised ordinal encoding. pd.Categorical gives -1 for values
            # the training slice never contained; -1 and genuine NaN both
            # become NaN, which HistGradientBoosting treats as missing.
            codes = pd.Categorical(
                series.astype(str), categories=cats
            ).codes.astype(float)
            codes[codes < 0] = np.nan
            codes[series.isna().to_numpy()] = np.nan
            x[:, j] = codes
        return x, cat_mask

    def _feature_subset(
        self, col: str, x: np.ndarray, cat_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Feature matrix with the target column removed (no self-prediction)."""
        keep = [j for j, name in enumerate(self.feature_columns_) if name != col]
        names = [self.feature_columns_[j] for j in keep]
        return x[:, keep], cat_mask[keep], names

    @staticmethod
    def _canonical_series(series: pd.Series, numeric_target: bool) -> pd.Series:
        """
        Vectorised counterpart of ``_canonical_key`` for a whole column.

        Must agree with ``_canonical_key`` exactly, because fit builds the
        class keys with that function and scoring looks them up with this one.
        """
        if numeric_target:
            return pd.to_numeric(series, errors="coerce").astype(float)
        return series.astype(str)

    def _feature_subset_x(self, col: str, x: np.ndarray) -> np.ndarray:
        """Just the feature-matrix slice (scoring path needs no categorical mask)."""
        keep = [j for j, name in enumerate(self.feature_columns_) if name != col]
        return x[:, keep]

    # ------------------------------------------------------------------ #
    # Fitting one column                                                   #
    # ------------------------------------------------------------------ #

    def _fit_column(
        self,
        col: str,
        role: str,
        reason: str,
        target_train: pd.Series,
        x_train: np.ndarray,
        cat_mask: np.ndarray,
    ) -> ColumnModel:
        """Fit the conditional model and the marginal statistics for one column."""
        numeric_target = pd.api.types.is_numeric_dtype(target_train)
        model = ColumnModel(
            column=col,
            role=role,
            reason=reason,
            is_categorical_target=(role == ROLE_CLASSIFIER),
            numeric_target=numeric_target,
        )

        x_sub, cat_sub, feat_names = self._feature_subset(col, x_train, cat_mask)
        model.feature_columns = feat_names

        # ── Marginal statistics (needed by every mode) ─────────────────────
        non_null_mask = (~target_train.isna()).to_numpy()
        if role == ROLE_CLASSIFIER:
            keys = [
                _canonical_key(v, numeric_target)
                for v in target_train.to_numpy()[non_null_mask]
            ]
            classes = sorted(set(keys))
            model.classes = classes
            total = float(len(keys))
            counts: Dict[Any, int] = {}
            for k in keys:
                counts[k] = counts.get(k, 0) + 1
            model.marginal_freq = {k: counts[k] / total for k in classes}
        else:
            raw_numeric = pd.to_numeric(target_train, errors="coerce").to_numpy(
                dtype=float
            )
            usable = raw_numeric[np.isfinite(raw_numeric)]
            model.marginal_center = (
                float(np.median(usable)) if usable.size else 0.0
            )
            model.marginal_scale = _robust_scale(usable, self.eps)

        # ── mode="marginal" fits no estimator at all ───────────────────────
        if self.mode == "marginal":
            model.n_train = int(non_null_mask.sum())
            return model

        if x_sub.shape[1] == 0:
            model.role = ROLE_SKIPPED
            model.reason = "no_other_columns_to_condition_on"
            return model

        # ── Conditional estimator ──────────────────────────────────────────
        if role == ROLE_CLASSIFIER:
            key_to_index = {k: i for i, k in enumerate(model.classes or [])}
            y = np.array(
                [
                    key_to_index[_canonical_key(v, numeric_target)]
                    for v in target_train.to_numpy()[non_null_mask]
                ],
                dtype=int,
            )
            x_fit = x_sub[non_null_mask, :]
            if len(np.unique(y)) < 2 or x_fit.shape[0] < self.min_train_rows:
                model.role = ROLE_SKIPPED
                model.reason = f"unfittable_classifier(n={x_fit.shape[0]})"
                return model
            est = HistGradientBoostingClassifier(
                max_iter=self.max_iter,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                l2_regularization=self.l2_regularization,
                categorical_features=cat_sub if cat_sub.any() else None,
                early_stopping=False,
                random_state=self.seed,
            )
            est.fit(x_fit, y)
            model.estimator = est
            model.n_train = int(x_fit.shape[0])
            return model

        # Regressor
        y_raw = pd.to_numeric(target_train, errors="coerce").to_numpy(dtype=float)
        usable_mask = np.isfinite(y_raw)
        x_fit = x_sub[usable_mask, :]
        y_fit = y_raw[usable_mask]
        if x_fit.shape[0] < self.min_train_rows:
            model.role = ROLE_SKIPPED
            model.reason = f"unfittable_regressor(n={x_fit.shape[0]})"
            return model
        est = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            categorical_features=cat_sub if cat_sub.any() else None,
            early_stopping=False,
            random_state=self.seed,
        )
        est.fit(x_fit, y_fit)
        model.estimator = est
        model.n_train = int(x_fit.shape[0])

        # Robust scale of TRAIN residuals -> the Gaussian sigma used for z.
        # In-sample residuals slightly understate the scale, which makes the
        # tail a little over-confident; stated, not hidden.
        residuals = y_fit - est.predict(x_fit)
        model.residual_scale = _robust_scale(residuals, self.eps)
        return model

    # ------------------------------------------------------------------ #
    # Scoring one column                                                   #
    # ------------------------------------------------------------------ #

    def _score_column(
        self,
        model: ColumnModel,
        series: pd.Series,
        x_all: Optional[np.ndarray],
        active_mode: str,
    ) -> np.ndarray:
        """
        Per-cell score for one column, in [0, 1].

        Missing cells take ``nan_score`` in every mode (see module docstring).
        """
        n = len(series)
        observed_na = series.isna().to_numpy()

        p_marginal = self._marginal_plausibility(model, series)
        if active_mode == "marginal":
            scores = 1.0 - p_marginal
        else:
            assert x_all is not None  # guaranteed by score_cells
            p_cond = self._conditional_plausibility(model, series, x_all)
            if active_mode == "full":
                scores = 1.0 - p_cond
            else:  # ratio
                lo, hi = self.eps, 1.0
                pm = np.clip(p_marginal, lo, hi)
                pc = np.clip(p_cond, lo, hi)
                raw = np.log(pm) - np.log(pc)
                scores = 1.0 / (1.0 + np.exp(-raw))

        scores = np.asarray(scores, dtype=float).reshape(n)
        scores[~np.isfinite(scores)] = 0.0
        scores[observed_na] = self.nan_score
        return np.clip(scores, 0.0, 1.0)

    def _marginal_plausibility(
        self, model: ColumnModel, series: pd.Series
    ) -> np.ndarray:
        """
        P(observed value), ignoring the rest of the row.

        Categorical: the training relative frequency (unseen value -> 0).
        Numerical: the two-sided Gaussian tail of the robust z-score about the
        training median, i.e. "how usual is a value at least this far out".
        """
        n = len(series)
        if model.is_categorical_target:
            freq = model.marginal_freq or {}
            keys = self._canonical_series(series, model.numeric_target)
            out = keys.map(freq).to_numpy(dtype=float)
            out[~np.isfinite(out)] = 0.0
            out[series.isna().to_numpy()] = np.nan
            return out

        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        scale = model.marginal_scale or self.eps
        center = model.marginal_center or 0.0
        z = (values - center) / max(scale, self.eps)
        tail = _residual_tail(z, self.residual_tail)
        # A value that does not parse as a number at all is maximally
        # implausible under a numeric marginal, not "unknown".
        tail[~np.isfinite(values) & ~series.isna().to_numpy()] = 0.0
        return tail

    def _conditional_plausibility(
        self, model: ColumnModel, series: pd.Series, x_all: np.ndarray
    ) -> np.ndarray:
        """
        P(observed value | rest of row).

        Categorical: the predicted class probability of the observed value; a
        value the training slice never contained gets 0.0 (never-seen is
        maximally surprising, which is the correct reading for a novel
        category or a format error).
        Numerical: the two-sided Gaussian tail of the standardised residual.
        """
        n = len(series)
        x_sub = self._feature_subset_x(model.column, x_all)

        if model.is_categorical_target:
            proba = model.estimator.predict_proba(x_sub)
            classes = model.classes or []
            key_to_index = {k: i for i, k in enumerate(classes)}
            # Column position of each class index inside predict_proba's
            # output. -1 marks a class the estimator never saw (possible when
            # a class exists in the marginal counts but not in the fit slice).
            class_pos = np.full(len(classes), -1, dtype=int)
            for pos, cls in enumerate(model.estimator.classes_):
                class_pos[int(cls)] = pos

            idx = (
                self._canonical_series(series, model.numeric_target)
                .map(key_to_index)
                .to_numpy(dtype=float)
            )
            out = np.zeros(n, dtype=float)
            known = np.isfinite(idx)
            if known.any():
                positions = class_pos[idx[known].astype(int)]
                usable = positions >= 0
                rows = np.flatnonzero(known)[usable]
                out[rows] = proba[rows, positions[usable]]
            out[series.isna().to_numpy()] = np.nan
            return out

        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        predicted = model.estimator.predict(x_sub)
        scale = model.residual_scale or self.eps
        z = (values - predicted) / max(scale, self.eps)
        tail = _residual_tail(z, self.residual_tail)
        tail[~np.isfinite(values) & ~series.isna().to_numpy()] = 0.0
        return tail

    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "fitted" if self.is_fitted_ else "unfitted"
        return (
            f"ConditionalDetector(mode={self.mode!r}, seed={self.seed}, {state}, "
            f"n_modelled={sum(1 for m in self.column_models_.values() if not m.is_skipped)})"
        )
