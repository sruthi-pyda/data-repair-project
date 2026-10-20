"""
src/detectors/cacm.py
---------------------
W8-A — **CACM**, the contamination-aware conditional mixture.
Specified in `docs/w8_plan.md` §3 and §4; falsification conditions in §5.

WHY THIS EXISTS
===============
`docs/w8_plan.md` §1 settled the diagnosis. *Within* a corrupted column our
conditional detector already captures 77.9 % of the Bayes-optimal signal
(0.5626 of a 0.7222 ceiling on adult `income`). *Across* columns it is at
chance: whole-table PR-AUC 0.0025 on adult, which equals the prevalence to
four decimals. The entire remaining loss is **choosing the column**, and four
mechanisms have died trying (M1 likelihood ratio, self-supervised injection,
M2 contamination fraction, detector-confidence abstention).

They all died the same way. Each built a per-column summary of a per-cell
*score*, and a score carries column-specific units, so the ranking ended up
measuring rarity, or cardinality, or merely whether the detector scored the
column at all.

THE IDENTIFYING RESTRICTION
===========================
Write the dirty table's generative process for one column `c`. Corruption is
applied to rows chosen at random, **independently of what else is in the
row**. The truth is not: a clean value is predictable from the rest of its
row. So the observed value distribution is a two-component mixture with an
asymmetry that identifies it:

    P_obs(v | x)  =  (1 - r_c) * f(v | x)      <- clean:      DEPENDS on x
                   +      r_c  * g(v)          <- corruption: does NOT

`x`-independence of `g` is the whole idea. It is the one structural fact that
tells corruption apart from legitimate rarity, and none of the four failed
mechanisms used it. It is also exactly why M2 failed: M2 fitted a mixture to
the *scores*, a 1-D summary in which the x-dependence/x-independence
asymmetry has already been destroyed. The mixture has to be fitted in
`(v given x)` space, which is where the restriction lives.

The per-cell output is then a posterior responsibility — a probability, not a
score:

    P(corrupt | v, x)  =         r_c * g(v)
                          -----------------------------------
                          (1-r_c) * f(v|x)  +  r_c * g(v)

Why that shape is right for both of our failures:

  rare-but-valid     a rare *correct* value has small `g(v)` (corruption does
                     not concentrate there) and relatively large `f(v|x)` (the
                     row implies it)                      -> posterior LOW
  the real adult case a flipped `income` has large `g(v)` (all 1,701 flips are
                     `1 -> 0`, so the corruption mass sits on `0`) and small
                     `f(v|x)` (the row implies `1`)        -> posterior HIGH

And because the output is a posterior probability by construction, it needs
no per-column normalisation. Manufacturing cross-column comparability is
precisely what every failed mechanism was trying and failing to do.

Estimating `g` rather than assuming it is factor **F2** of `docs/w8_plan.md`
§2 — a corruption policy learned from the dirty table. HoloDetect's ablation
is the reason that matters: learned augmentation 0.953 F1 against 0.166 for
random transformations.

PRIOR ART — STATE IT, DO NOT OVERCLAIM
======================================
Kubica & Moore, *Probabilistic Noise Identification and Data Cleaning* (CMU,
2002/2003) is the closest ancestor: a clean model, a noise model and a
corruption matrix fitted by EM-style alternating optimisation. That is the
honest citation and it belongs anywhere this module is described. What is
different here is narrow and should be claimed narrowly: a **per-column
conditional-plus-marginal** mixture with **x-independence as the explicit
identifying restriction**, yielding a **cross-column-comparable calibrated
posterior** that feeds a cost-sensitive three-way decision. Also related:
noisy-label identifiability via multinomial mixtures (arXiv 2301.01405).

The claim is *not* "a new way to model noise". It is "x-independence
identifies the contamination, which makes per-cell error probabilities
comparable across columns, which is what column triage needs".

THE ALGORITHM
=============
One EM run per column of the dirty table.

    discretise   every column becomes categorical, so there is exactly ONE
                 implementation and `g` is always a histogram (see below)
    initialise   r_c = 0.05, g = the column's empirical marginal
    E-step       z_i = posterior responsibility of cell i, formula above
    M-step       f  <- refit with sample_weight = (1 - z_i)
                 g  <- the z-weighted value histogram (additively smoothed)
                 r_c <- mean(z)
    stop         when max_i |z_i - z_i^prev| < tol, or after `em_iters`

The first E-step needs an `f`, so iteration 0 fits `f` with uniform weights
(equivalent to z = 0 everywhere) and then takes the first E-step with
r_c = `r_init` and g = the marginal.

INITIALISATION — A MEASURED DEPARTURE FROM THE PLAN
===================================================
`docs/w8_plan.md` §4 says to initialise `r_c` small, e.g. 0.05. **The default
here is 0.3, because 0.05 was measured and found to converge from the wrong
side far too slowly.** On synthetic data generated with a known
`r = 0.20` and a known `g` concentrated on one value (n = 2000, 20 EM
iterations, the generator of `tests/test_cacm.py::make_contaminated`):

    r_init   folds   r_hat    g(corrupt value)   cell ROC-AUC
    0.05       2     0.0684        0.952            0.9888
    0.30       2     0.1617        0.979            0.9889
    0.45       2     0.1801        0.981            0.9888
    0.05       1     0.0001        0.147            0.8831

Two things to read off that table.

First, the **ranking is essentially invariant** to the initialisation
(0.9888 / 0.9889 / 0.9888) while the **estimated rate is not**. A flexible
`f` fitted on contaminated data can explain the corruption as part of the
conditional, so "r small and f explains everything" is a near-fixed point;
EM started at 0.05 crawls away from it (and had not converged after 20
iterations, final step size 0.051), whereas started above the truth it
descends onto it. Starting high and letting EM come down is the standard
remedy for exactly this degenerate attractor, and it is the one supported by
the measurement here.

Second, the last row is why the cross-fitting below is not optional: with
`n_folds=1` the whole thing collapses to the floor and `g` never leaves its
marginal initialisation.

The obvious objection to starting high is that a **clean** column would then
be left stranded at a high rate, which would wreck column triage. Measured,
on one table carrying a clean column and two contaminated ones side by side
(n = 1500, `r_init = 0.3`):

    column   true r    r_hat @ 10 iters   r_hat @ 20 iters
    clean    0.000     0.0001 (at floor)  0.0001 (at floor)
    light    0.030     0.0241             0.0241
    heavy    0.293     0.1578             0.1937

A clean column goes straight to the floor, a lightly contaminated one is
estimated closely, and a heavily contaminated one is **underestimated but
correctly ordered**. Triage needs the ordering, so the default budget is
`em_iters=10`; the underestimation at high contamination is a stated
limitation, not a tuned-away one.

DISCRETISATION
==============
Every column is mapped to at most `n_bins` categories (default **10**, and
that choice is reported in `results/w8a_cacm.md`):

  numeric, cardinality > n_bins   `n_bins` quantile bins
  anything else                   the `n_bins - 1` most frequent values keep
                                  their own category; the remaining tail
                                  collapses into one `__OTHER__` bucket
  missing                         its own `__NA__` category

Three consequences worth being explicit about, because they are limitations
and not conveniences:

1. One implementation covers numeric and categorical columns, and `g` is a
   histogram over a finite support in both cases. That is the point.
2. NaN gets its own category deliberately. Smart Factory's corruption *is*
   its NaN cells, so `g` must be allowed to place mass on "missing". This is
   the mixture doing the work, not a syntactic missing-token rule.
3. A high-cardinality text column collapses almost entirely into
   `__OTHER__`, so its discretised target is near-constant, `f` is
   uninformative, and the posterior comes out nearly constant. That column is
   then **unscorable** and the experiment excludes it under falsification
   condition 5 rather than letting a constant vector enter an AUC. The
   collapse is visible in `report()`, not hidden.

CROSS-FITTING
=============
`f` is fitted by K-fold cross-fitting (default K = 2) and every `f(v|x)` used
in an E-step is an **out-of-fold** prediction. This is not an optimisation,
it is load-bearing. Fitted in-sample, a gradient-boosted `f` partly memorises
the corrupted cells it is supposed to down-weight, inflates `f(v|x)` there,
and drives the responsibilities — and therefore `r_c` — towards zero. Two
folds cost about the same total compute as one full-data fit, so the honest
version is also the cheap one.

DEGENERATE COLLAPSE
===================
`r_c` is clamped into `[r_min, r_max]` (default `[1e-4, 0.5]`). Hitting
either bound is recorded in `ColumnMixture.collapsed` and surfaced by
`report()`; a column at the bound has not estimated a contamination rate, it
has run out of the interval, and reporting it as an estimate would be a
fabricated number. The upper bound is 0.5 because beyond it the labelling of
the two components swaps and "corruption" becomes the majority process, at
which point the identifying restriction no longer picks out the thing we
mean.

LEAKAGE (CLAUDE.md §3 — non-negotiable)
=======================================
SELF-SUPERVISED ONLY. `fit` and `score_cells` take the dirty frame and
nothing else. No parameter is named after, or derived from, clean data. The
`profile` argument exists only for interface parity with
`src/detectors/conditional.py`, is **ignored**, and defaults to `None`;
nothing in this module reads it, so passing one cannot leak the documented
soft-leakage caveat either. Clean data is read only afterwards, by
`experiments/w8a_cacm.py`, through
`src/detection_eval/labels.py::cell_labels`, purely to score results that
already exist. `tests/test_cacm.py` asserts both halves: a signature scan for
any parameter mentioning clean/label/truth, and a spy on the estimator that
proves every target value reaching `fit` is a value of the dirty frame.

DETERMINISM
===========
Given the same `dirty_df` and `seed`, scores are bitwise reproducible. Fold
assignment comes from a `default_rng(seed)` permutation and
`early_stopping=False` keeps sklearn from carving its own random validation
split out of the training data.

WORKED EXAMPLE
==============
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> x = rng.integers(0, 4, size=600)
    >>> y = x.copy()                      # y is a deterministic function of x
    >>> hit = rng.random(600) < 0.2       # corrupt 20 % of rows ...
    >>> y[hit] = 0                        # ... always to the value 0
    >>> df = pd.DataFrame({"x": x, "y": y})
    >>> from src.detectors.cacm import CACMDetector
    >>> det = CACMDetector(seed=42).fit(df)
    >>> 0.1 < det.column_mixtures_["y"].r < 0.35
    True
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

logger = logging.getLogger(__name__)

__all__ = [
    "CACMDetector",
    "ColumnMixture",
    "posterior_corrupt",
    "NA_TOKEN",
    "OTHER_TOKEN",
    "ROLE_MIXTURE",
    "ROLE_SKIPPED",
]

#: Category used for a missing cell. Corruption is allowed to place mass here
#: (Smart Factory's corruption is exactly its NaN cells).
NA_TOKEN = "__NA__"

#: Bucket absorbing the rare tail of a high-cardinality discretised column.
OTHER_TOKEN = "__OTHER__"

ROLE_MIXTURE = "mixture"
ROLE_SKIPPED = "skipped"

#: sklearn's native categorical support needs codes in [0, max_bins).
_SKLEARN_MAX_BINS = 255


# ═══════════════════════════════════════════════════════════════════════════════
# The posterior — a free function so it can be tested against hand arithmetic
# ═══════════════════════════════════════════════════════════════════════════════


def posterior_corrupt(
    r: float,
    f: np.ndarray,
    g: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    E-step: ``P(corrupt | v, x) = r*g / ((1-r)*f + r*g)``, elementwise.

    Parameters
    ----------
    r : float
        Contamination rate, in (0, 1).
    f : np.ndarray
        ``f(v_i | x_i)`` — the clean conditional probability of the value
        actually observed in cell *i*. Out-of-fold in the fitted detector.
    g : np.ndarray
        ``g(v_i)`` — the corruption model's probability of that same value.
        No dependence on ``x``: that is the identifying restriction.
    eps : float
        Denominator floor. When both components vanish the cell carries no
        information either way and the responsibility is set to ``r`` (the
        prior), which is the only honest answer.

    Returns
    -------
    np.ndarray
        Responsibilities in [0, 1], same shape as ``f``.
    """
    if not 0.0 < r < 1.0:
        raise ValueError(f"r must be in (0, 1), got {r}")
    f_arr = np.asarray(f, dtype=float)
    g_arr = np.asarray(g, dtype=float)
    if f_arr.shape != g_arr.shape:
        raise ValueError(f"f and g must have the same shape, {f_arr.shape} vs {g_arr.shape}")

    clean_part = (1.0 - r) * np.clip(f_arr, 0.0, None)
    dirty_part = r * np.clip(g_arr, 0.0, None)
    denom = clean_part + dirty_part
    out = np.full(f_arr.shape, float(r), dtype=float)
    ok = denom > eps
    out[ok] = dirty_part[ok] / denom[ok]
    out[~np.isfinite(out)] = float(r)
    return np.clip(out, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-column fitted state
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ColumnMixture:
    """
    Everything the EM run learned about one column.

    One of these exists for every column after ``fit``, including skipped
    ones (``role == "skipped"``), so ``report()`` can account for every
    column instead of silently omitting some.

    Attributes
    ----------
    column : str
    role : str
        ``"mixture"`` or ``"skipped"``.
    reason : str
        Why that role. For skipped columns, the skip reason.
    r : float | None
        Fitted contamination rate ``r_c``. ``None`` when skipped.
    categories : list[str] | None
        The discretised support, in fitted order.
    g : dict[str, float] | None
        The fitted corruption distribution over ``categories``.
    g_init : dict[str, float] | None
        The initial value of ``g`` (the column marginal), kept so the report
        can show how far EM moved it.
    n_iter : int
        EM iterations actually run.
    delta : float | None
        Final ``max_i |z_i - z_i^prev|``. Small means converged.
    converged : bool
        ``delta < tol``.
    collapsed : str
        ``""``, ``"r_at_min"`` or ``"r_at_max"`` — ``r`` hit a clamp bound, so
        it is a bound, not an estimate.
    mean_posterior : float | None
        Mean responsibility over the column. This is the column-level triage
        statistic.
    n_categories : int
        ``len(categories)``.
    other_fraction : float
        Share of cells that landed in ``__OTHER__``. High means the
        discretisation collapsed the column and the posterior will be close
        to constant.
    """

    column: str
    role: str
    reason: str
    r: Optional[float] = None
    categories: Optional[List[str]] = None
    g: Optional[Dict[str, float]] = None
    g_init: Optional[Dict[str, float]] = None
    n_iter: int = 0
    delta: Optional[float] = None
    converged: bool = False
    collapsed: str = ""
    mean_posterior: Optional[float] = None
    n_categories: int = 0
    other_fraction: float = 0.0

    #: Final out-of-fold f(v|x) and g(v) per cell. Diagnostics, not results;
    #: excluded from to_dict().
    f_obs: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    g_obs: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    @property
    def is_skipped(self) -> bool:
        return self.role == ROLE_SKIPPED

    def top_g(self, k: int = 5) -> List[Tuple[str, float]]:
        """The ``k`` categories carrying the most corruption mass."""
        if not self.g:
            return []
        return sorted(self.g.items(), key=lambda kv: -kv[1])[:k]

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict view for JSON / logging."""
        return {
            "column": self.column,
            "role": self.role,
            "reason": self.reason,
            "r": self.r,
            "n_categories": self.n_categories,
            "other_fraction": self.other_fraction,
            "n_iter": self.n_iter,
            "delta": self.delta,
            "converged": self.converged,
            "collapsed": self.collapsed,
            "mean_posterior": self.mean_posterior,
            "top_g": self.top_g(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# The detector
# ═══════════════════════════════════════════════════════════════════════════════


class CACMDetector:
    """
    Contamination-aware conditional mixture (W8-A).

    Fits, per column of the dirty table, the two-component mixture

        P_obs(v | x) = (1 - r_c) * f(v | x) + r_c * g(v)

    by EM, and scores each cell with the posterior responsibility that it is
    corrupt. See the module docstring for the identifying restriction, the
    discretisation, the cross-fitting and the leakage guarantee.

    Parameters
    ----------
    n_bins : int
        Maximum number of discretised categories per column, *excluding* the
        ``__NA__`` category. Default 10. Numeric columns with more distinct
        values than this get quantile bins; other columns keep their
        ``n_bins - 1`` most frequent values and collapse the tail into
        ``__OTHER__``.
    em_iters : int
        Maximum EM iterations. Default 10. See the module docstring for the
        measurement behind that number.
    n_folds : int
        Cross-fitting folds for ``f``. Default 2. ``1`` disables
        cross-fitting, which biases ``r_c`` towards 0 (see the module
        docstring) and is offered only for the ablation.
    r_init : float
        Initial contamination rate. Default **0.3**, a measured departure
        from the plan's suggested 0.05 -- see the module docstring for the
        table that justifies it. 0.05 leaves EM crawling away from the
        degenerate "f explains everything" fixed point.
    r_min, r_max : float
        Clamp for ``r_c``. Defaults 1e-4 and 0.5.
    tol : float
        EM stops when ``max_i |z_i - z_i^prev| < tol``. Default 1e-3.
    g_smoothing : float
        Additive (Laplace) smoothing on the ``z``-weighted histogram for
        ``g``, so no category has exactly zero corruption probability and the
        posterior is always defined. Default 1.0.
    max_iter, learning_rate, max_depth, min_samples_leaf, l2_regularization
        Passed through to ``HistGradientBoostingClassifier``.
    id_unique_ratio, id_min_unique : float, int
        A column with more than ``id_min_unique`` distinct non-null values
        and a distinct/non-null ratio at or above ``id_unique_ratio`` is
        treated as ID-like and skipped. Same rule, and same defaults
        (0.9 / 50), as ``src/detectors/conditional.py``, so the two
        detectors' eligible column sets are comparable.
    min_rows : int
        Minimum usable rows for a column to be modelled.
    seed : int
        Determinism. Governs fold assignment and the estimators.

    Attributes (set by ``fit``)
    ---------------------------
    column_mixtures_ : dict[str, ColumnMixture]
    skipped_columns_ : dict[str, str]
    feature_columns_ : list[str]
    n_rows_ : int
    is_fitted_ : bool
    """

    def __init__(
        self,
        n_bins: int = 10,
        em_iters: int = 10,
        n_folds: int = 2,
        r_init: float = 0.3,
        r_min: float = 1e-4,
        r_max: float = 0.5,
        tol: float = 1e-3,
        g_smoothing: float = 1.0,
        max_iter: int = 30,
        learning_rate: float = 0.1,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        id_unique_ratio: float = 0.9,
        id_min_unique: int = 50,
        min_rows: int = 30,
        seed: int = 42,
    ) -> None:
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        if em_iters < 1:
            raise ValueError(f"em_iters must be >= 1, got {em_iters}")
        if n_folds < 1:
            raise ValueError(f"n_folds must be >= 1, got {n_folds}")
        if not 0.0 < r_min < r_max < 1.0:
            raise ValueError(f"need 0 < r_min < r_max < 1, got {r_min}, {r_max}")
        if not r_min <= r_init <= r_max:
            raise ValueError(f"r_init {r_init} must lie in [{r_min}, {r_max}]")
        if g_smoothing < 0.0:
            raise ValueError(f"g_smoothing must be >= 0, got {g_smoothing}")
        if tol <= 0.0:
            raise ValueError(f"tol must be > 0, got {tol}")

        self.n_bins = n_bins
        self.em_iters = em_iters
        self.n_folds = n_folds
        self.r_init = r_init
        self.r_min = r_min
        self.r_max = r_max
        self.tol = tol
        self.g_smoothing = g_smoothing
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.id_unique_ratio = id_unique_ratio
        self.id_min_unique = id_min_unique
        self.min_rows = min_rows
        self.seed = seed

        self.column_mixtures_: Dict[str, ColumnMixture] = {}
        self.skipped_columns_: Dict[str, str] = {}
        self.feature_columns_: List[str] = []
        self.n_rows_: int = 0
        self.is_fitted_: bool = False

        self._columns: List[str] = []
        self._codes: Dict[str, np.ndarray] = {}       # column -> discrete code per row
        self._categories: Dict[str, List[str]] = {}   # column -> category labels
        self._encoding: Dict[str, Tuple[bool, Optional[List[str]]]] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        dirty_df: pd.DataFrame,
        profile: Any = None,  # noqa: ARG002 - ignored; interface parity only
    ) -> "CACMDetector":
        """
        Run one EM mixture fit per column of the dirty table.

        Parameters
        ----------
        dirty_df : pd.DataFrame
            The dirty table. **The only data source.** No clean data, no
            labels, no profile-derived statistics.
        profile : ignored
            Accepted so this class is drop-in against
            ``src/detectors/conditional.py``. Nothing reads it, so passing a
            clean-derived profile cannot leak anything.

        Returns
        -------
        self
        """
        if not isinstance(dirty_df, pd.DataFrame):
            raise TypeError(f"dirty_df must be a DataFrame, got {type(dirty_df)!r}")
        if dirty_df.shape[1] == 0:
            raise ValueError("dirty_df has no columns")
        if len(dirty_df) == 0:
            raise ValueError("dirty_df has no rows")

        self._columns = list(dirty_df.columns)
        self.n_rows_ = int(len(dirty_df))
        self.column_mixtures_ = {}
        self._codes = {}
        self._categories = {}

        # ── 1. Role of every column, and its discretisation ────────────────
        roles: Dict[str, Tuple[str, str]] = {}
        for col in self._columns:
            role, reason = self._decide_role(dirty_df[col])
            roles[col] = (role, reason)
            if role == ROLE_SKIPPED:
                continue
            cats, codes = self._discretise(dirty_df[col])
            if len(cats) < 2:
                roles[col] = (ROLE_SKIPPED, f"constant_after_discretisation({len(cats)})")
                continue
            self._categories[col] = cats
            self._codes[col] = codes

        # ── 2. Shared feature matrix (raw values, NOT the discretisation) ──
        # f(v|x) conditions on the row as it actually is; binning the features
        # too would throw away conditioning power for no benefit.
        self.feature_columns_ = self._choose_feature_columns(dirty_df, roles)
        self._encoding = {
            col: self._encoding_for(dirty_df[col]) for col in self.feature_columns_
        }
        x_all, cat_mask = self._encode(dirty_df)

        # ── 3. One EM run per modelled column ──────────────────────────────
        folds = self._fold_assignment(self.n_rows_)
        for col in self._columns:
            role, reason = roles[col]
            if role == ROLE_SKIPPED:
                self.column_mixtures_[col] = ColumnMixture(
                    column=col, role=role, reason=reason
                )
                continue
            self.column_mixtures_[col] = self._fit_column(
                col=col, reason=reason, x_all=x_all, cat_mask=cat_mask, folds=folds
            )

        self.skipped_columns_ = {
            c: m.reason for c, m in self.column_mixtures_.items() if m.is_skipped
        }
        self.is_fitted_ = True
        logger.info(
            "CACMDetector fitted on %d rows: %d columns modelled, %d skipped",
            self.n_rows_,
            sum(1 for m in self.column_mixtures_.values() if not m.is_skipped),
            len(self.skipped_columns_),
        )
        return self

    def score_cells(
        self,
        dirty_df: pd.DataFrame,
        profile: Any = None,  # noqa: ARG002 - ignored; interface parity only
    ) -> pd.DataFrame:
        """
        Posterior ``P(corrupt | v, x)`` for every cell. Higher = more likely
        corrupted.

        Only the frame ``fit`` was given can be scored: the responsibilities
        come from out-of-fold predictions tied to that frame's rows, and
        re-deriving them for unseen rows would silently use in-sample ``f``,
        which is the one thing the cross-fitting exists to prevent.
        Falsification is about this table, so this restriction costs nothing
        and makes the alternative impossible to do by accident.

        Returns
        -------
        pd.DataFrame
            Float frame with exactly ``dirty_df``'s index and columns, values
            in [0, 1]. Skipped columns are all-zero.
        """
        if not self.is_fitted_:
            raise RuntimeError("score_cells() called before fit()")
        if list(dirty_df.columns) != self._columns:
            raise ValueError(
                "score_cells() received different columns than fit(): "
                f"expected {self._columns}, got {list(dirty_df.columns)}"
            )
        if len(dirty_df) != self.n_rows_:
            raise ValueError(
                "score_cells() must be called on the frame fit() was given "
                f"({self.n_rows_} rows), got {len(dirty_df)}"
            )

        out = pd.DataFrame(
            0.0, index=dirty_df.index.copy(), columns=dirty_df.columns, dtype=float
        )
        for col in self._columns:
            mix = self.column_mixtures_[col]
            if mix.is_skipped or mix.f_obs is None or mix.g_obs is None or mix.r is None:
                continue
            out[col] = posterior_corrupt(mix.r, mix.f_obs, mix.g_obs)
        return out

    def fit_score(self, dirty_df: pd.DataFrame, profile: Any = None) -> pd.DataFrame:
        """Convenience: ``fit(...)`` then ``score_cells(...)``."""
        return self.fit(dirty_df, profile=profile).score_cells(dirty_df)

    def column_statistic(self, statistic: str = "mean_posterior") -> Dict[str, Optional[float]]:
        """
        The per-column triage quantity, for every column.

        ``"mean_posterior"`` is the mean responsibility; ``"r"`` is the fitted
        contamination rate. Skipped columns map to ``None`` — never to 0.0,
        because "not scored" is not "scored zero", and collapsing the two is
        the bug that made W6.1's baseline look identical to its mechanism.
        """
        if statistic not in ("mean_posterior", "r"):
            raise ValueError(f"unknown statistic {statistic!r}")
        if not self.is_fitted_:
            raise RuntimeError("column_statistic() called before fit()")
        return {
            c: (None if m.is_skipped else getattr(m, statistic))
            for c, m in self.column_mixtures_.items()
        }

    def report(self) -> Dict[str, Any]:
        """Summary of what was fitted, for experiment write-ups."""
        if not self.is_fitted_:
            raise RuntimeError("report() called before fit()")
        return {
            "n_bins": self.n_bins,
            "em_iters": self.em_iters,
            "n_folds": self.n_folds,
            "r_init": self.r_init,
            "g_smoothing": self.g_smoothing,
            "seed": self.seed,
            "n_rows": self.n_rows_,
            "n_columns": len(self._columns),
            "n_modelled": sum(
                1 for m in self.column_mixtures_.values() if not m.is_skipped
            ),
            "skipped_columns": dict(self.skipped_columns_),
            "collapsed_columns": {
                c: m.collapsed
                for c, m in self.column_mixtures_.items()
                if m.collapsed
            },
            "feature_columns": list(self.feature_columns_),
            "columns": {c: m.to_dict() for c, m in self.column_mixtures_.items()},
        }

    # ------------------------------------------------------------------ #
    # Roles and discretisation                                             #
    # ------------------------------------------------------------------ #

    def _decide_role(self, series: pd.Series) -> Tuple[str, str]:
        """Model this column, or skip it? Dirty data only."""
        non_null = series[~series.isna()]
        n_non_null = len(non_null)
        if n_non_null == 0:
            return ROLE_SKIPPED, "all_missing"
        n_unique = int(non_null.nunique())
        if n_unique < 2 and n_non_null == len(series):
            return ROLE_SKIPPED, "constant"
        if len(series) < self.min_rows:
            return ROLE_SKIPPED, f"too_few_rows({len(series)})"
        if (
            n_unique > self.id_min_unique
            and n_unique / n_non_null >= self.id_unique_ratio
            and _is_identifier_like(non_null)
        ):
            return ROLE_SKIPPED, f"id_like_cardinality({n_unique}/{n_non_null})"
        return ROLE_MIXTURE, f"n_unique={n_unique}"

    def _discretise(self, series: pd.Series) -> Tuple[List[str], np.ndarray]:
        """
        Map a column onto at most ``n_bins`` categories plus ``__NA__``.

        Returns ``(categories, codes)`` where ``codes[i]`` indexes
        ``categories``. See the module docstring for the rule and its three
        stated consequences.
        """
        na = series.isna().to_numpy()
        labels = np.full(len(series), NA_TOKEN, dtype=object)
        present = ~na

        numeric = pd.api.types.is_numeric_dtype(series)
        n_unique = int(series[present].nunique()) if present.any() else 0

        if numeric and n_unique > self.n_bins:
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values) & present
            qs = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
            edges = np.unique(np.quantile(values[finite], qs)) if finite.any() else np.array([])
            idx = np.searchsorted(edges, values[finite], side="right")
            labels[finite] = np.array([f"q{int(i)}" for i in idx], dtype=object)
            # A numeric cell that is present but not finite-parseable keeps a
            # category of its own: it is a value, not a missing cell.
            odd = present & ~finite
            labels[odd] = "__UNPARSED__"
        else:
            as_str = series[present].astype(str)
            counts = as_str.value_counts()
            keep = list(counts.index[: max(self.n_bins - 1, 1)])
            if len(counts) <= self.n_bins:
                keep = list(counts.index)
            keep_set = set(keep)
            arr = as_str.to_numpy()
            labels[present] = np.array(
                [v if v in keep_set else OTHER_TOKEN for v in arr], dtype=object
            )

        cats = sorted({str(v) for v in labels})
        code_of = {c: i for i, c in enumerate(cats)}
        codes = np.array([code_of[str(v)] for v in labels], dtype=int)
        return cats, codes

    def _choose_feature_columns(
        self, df: pd.DataFrame, roles: Dict[str, Tuple[str, str]]
    ) -> List[str]:
        """
        Columns usable as predictors: everything not skipped, minus
        non-numeric columns with more than 255 distinct values (sklearn's
        categorical bin limit).
        """
        features: List[str] = []
        for col in self._columns:
            if roles[col][0] == ROLE_SKIPPED:
                continue
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                features.append(col)
                continue
            if int(series[~series.isna()].nunique()) > _SKLEARN_MAX_BINS:
                continue
            features.append(col)
        return features

    # ------------------------------------------------------------------ #
    # Feature encoding (shared with conditional.py's approach)             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _encoding_for(series: pd.Series) -> Tuple[bool, Optional[List[str]]]:
        """Numeric passes through; everything else is ordinal-coded as native
        categorical, which HistGradientBoosting splits on value subsets."""
        if pd.api.types.is_numeric_dtype(series):
            return False, None
        return True, sorted({str(v) for v in series[~series.isna()].to_numpy()})

    def _encode(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        n_rows, n_feat = len(df), len(self.feature_columns_)
        x = np.full((n_rows, n_feat), np.nan, dtype=float)
        cat_mask = np.zeros(n_feat, dtype=bool)
        for j, col in enumerate(self.feature_columns_):
            is_cat, cats = self._encoding[col]
            series = df[col]
            if not is_cat:
                x[:, j] = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
                continue
            cat_mask[j] = True
            codes = pd.Categorical(series.astype(str), categories=cats).codes.astype(float)
            codes[codes < 0] = np.nan
            codes[series.isna().to_numpy()] = np.nan
            x[:, j] = codes
        return x, cat_mask

    def _feature_subset(
        self, col: str, x: np.ndarray, cat_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Features with the target column removed (no self-prediction)."""
        keep = [j for j, name in enumerate(self.feature_columns_) if name != col]
        return x[:, keep], cat_mask[keep]

    def _fold_assignment(self, n: int) -> np.ndarray:
        """Deterministic fold index per row."""
        if self.n_folds <= 1:
            return np.zeros(n, dtype=int)
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        folds = np.empty(n, dtype=int)
        folds[perm] = np.arange(n) % self.n_folds
        return folds

    # ------------------------------------------------------------------ #
    # EM for one column                                                    #
    # ------------------------------------------------------------------ #

    def _fit_column(
        self,
        col: str,
        reason: str,
        x_all: np.ndarray,
        cat_mask: np.ndarray,
        folds: np.ndarray,
    ) -> ColumnMixture:
        cats = self._categories[col]
        codes = self._codes[col]
        k = len(cats)
        n = len(codes)

        mix = ColumnMixture(
            column=col,
            role=ROLE_MIXTURE,
            reason=reason,
            categories=list(cats),
            n_categories=k,
            other_fraction=float(
                np.mean([cats[c] == OTHER_TOKEN for c in codes])
            ),
        )

        x_sub, cat_sub = self._feature_subset(col, x_all, cat_mask)
        if x_sub.shape[1] == 0:
            mix.role = ROLE_SKIPPED
            mix.reason = "no_other_columns_to_condition_on"
            return mix
        if n < self.min_rows:
            mix.role = ROLE_SKIPPED
            mix.reason = f"too_few_rows({n})"
            return mix

        # ── initialisation: r small, g = the column marginal ───────────────
        r = float(self.r_init)
        counts = np.bincount(codes, minlength=k).astype(float)
        g_vec = counts / counts.sum()
        mix.g_init = {cats[i]: float(g_vec[i]) for i in range(k)}

        z = np.zeros(n, dtype=float)
        f_obs = np.full(n, np.nan, dtype=float)
        delta: Optional[float] = None

        for it in range(self.em_iters):
            # ── M-step for f: refit with sample_weight = (1 - z), and take
            #    OUT-OF-FOLD predictions so f never scores its own rows.
            weights = np.clip(1.0 - z, 1e-6, None)
            f_obs = self._out_of_fold_f(
                x_sub, cat_sub, codes, weights, folds, k
            )

            # ── E-step ─────────────────────────────────────────────────────
            z_new = posterior_corrupt(r, f_obs, g_vec[codes])
            delta = float(np.max(np.abs(z_new - z))) if n else 0.0
            z = z_new
            mix.n_iter = it + 1

            # ── M-step for g and r ─────────────────────────────────────────
            z_sum = float(z.sum())
            weighted = np.bincount(codes, weights=z, minlength=k).astype(float)
            g_vec = (weighted + self.g_smoothing) / (
                z_sum + self.g_smoothing * k
            )
            r_raw = float(z.mean())
            r = float(np.clip(r_raw, self.r_min, self.r_max))
            if r_raw <= self.r_min:
                mix.collapsed = "r_at_min"
            elif r_raw >= self.r_max:
                mix.collapsed = "r_at_max"
            else:
                mix.collapsed = ""

            if delta is not None and delta < self.tol:
                break

        # Final E-step with the last (g, r) so the reported posterior is the
        # one implied by the parameters we report, not one iteration stale.
        g_obs = g_vec[codes]
        z = posterior_corrupt(r, f_obs, g_obs)

        mix.r = r
        mix.g = {cats[i]: float(g_vec[i]) for i in range(k)}
        mix.delta = delta
        mix.converged = bool(delta is not None and delta < self.tol)
        mix.mean_posterior = float(np.mean(z))
        mix.f_obs = f_obs
        mix.g_obs = g_obs
        return mix

    def _out_of_fold_f(
        self,
        x: np.ndarray,
        cat_mask: np.ndarray,
        codes: np.ndarray,
        weights: np.ndarray,
        folds: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """
        ``f(v_i | x_i)`` for every row, from a model that did not see row *i*.

        A category absent from a fold's training rows gets probability 0 for
        the held-out rows showing it: under that fold's clean model the value
        is impossible, which is the correct reading and pushes the
        responsibility towards 1.
        """
        n = len(codes)
        out = np.zeros(n, dtype=float)
        n_folds = max(self.n_folds, 1)

        for fold in range(n_folds):
            if n_folds == 1:
                train_idx = np.arange(n)
                test_idx = np.arange(n)
            else:
                test_idx = np.flatnonzero(folds == fold)
                train_idx = np.flatnonzero(folds != fold)
            if test_idx.size == 0:
                continue
            y_train = codes[train_idx]
            present = np.unique(y_train)
            if present.size < 2 or train_idx.size < self.min_rows:
                # Nothing to condition on in this fold: fall back to the
                # weighted marginal, which is the honest degenerate f.
                w = weights[train_idx]
                tot = float(w.sum())
                marg = (
                    np.bincount(y_train, weights=w, minlength=k) / tot
                    if tot > 0
                    else np.full(k, 1.0 / k)
                )
                out[test_idx] = marg[codes[test_idx]]
                continue

            est = HistGradientBoostingClassifier(
                max_iter=self.max_iter,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                l2_regularization=self.l2_regularization,
                categorical_features=cat_mask if cat_mask.any() else None,
                early_stopping=False,
                random_state=self.seed,
            )
            est.fit(x[train_idx, :], y_train, sample_weight=weights[train_idx])
            proba = est.predict_proba(x[test_idx, :])

            # Map class label -> column of predict_proba's output.
            pos = np.full(k, -1, dtype=int)
            for j, cls in enumerate(est.classes_):
                pos[int(cls)] = j
            want = pos[codes[test_idx]]
            ok = want >= 0
            vals = np.zeros(test_idx.size, dtype=float)
            vals[ok] = proba[np.flatnonzero(ok), want[ok]]
            out[test_idx] = vals

        out[~np.isfinite(out)] = 0.0
        return np.clip(out, 0.0, 1.0)

    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "fitted" if self.is_fitted_ else "unfitted"
        return (
            f"CACMDetector(n_bins={self.n_bins}, em_iters={self.em_iters}, "
            f"n_folds={self.n_folds}, seed={self.seed}, {state})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _is_identifier_like(non_null: pd.Series) -> bool:
    """
    Could this near-unique column be an identifier rather than a measurement?

    Same reasoning as ``src/detectors/conditional.py::_is_integer_like``: a
    continuous measurement is also near-unique, so near-uniqueness alone is
    not enough. Identifiers are integer-valued or non-numeric; a column with
    genuine fractional values is a measurement.
    """
    if not pd.api.types.is_numeric_dtype(non_null):
        return True
    values = pd.to_numeric(non_null, errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return True
    return bool(np.all(finite == np.round(finite)))


def leakage_signature_report() -> Dict[str, List[str]]:
    """
    Parameter names of this module's public entry points.

    Exists so ``tests/test_cacm.py`` can assert mechanically that no entry
    point accepts clean data, labels or ground truth, rather than relying on
    a human reading the signatures.
    """
    out: Dict[str, List[str]] = {}
    for name in ("fit", "score_cells", "fit_score"):
        out[name] = list(inspect.signature(getattr(CACMDetector, name)).parameters)
    return out
