"""
src/detectors/excess_surprise.py
--------------------------------
**Excess Surprise** — a per-column corruption signal that cancels the one
nuisance parameter which defeated every previous mechanism.

WHY THIS EXISTS
===============
`docs/w8_plan.md` §8 records the root cause of four separate failures (M1 the
likelihood ratio, W5.3 self-detectability, M2 the contamination fraction, and
W8-A's CACM):

    The per-column MODEL ERROR RATE is a nuisance parameter, and every
    per-column corruption statistic we built absorbed it instead of the
    corruption.

CACM made it unmistakable: on adult's `income` it fitted a contamination rate
of 0.1852 against a true rate of 0.0400, because 0.1852 is roughly the
conditional model's own error rate on that column. To an 85 %-accurate model,
its own mistakes also look independent of the row.

It also explains the asymmetry we measured. A column's model error rate is
**constant across that column's cells**, so within a column it is an offset
that does not change the ranking — and indeed we reach 77.9 % of the Bayes
ceiling there. Across columns it **differs**, and those differences dominate
— hence chance.

THE IDEA
========
Do not ask how surprised the model is. Ask **how much more surprised than it
expected to be**. A probabilistic model forecasts its own error for free:

    observed_error_c  =  mean over rows of  (1 - f(v_observed | x))
    expected_error_c  =  mean over rows of  (1 - sum_v f(v | x)^2)
    excess_c          =  observed_error_c - expected_error_c

`expected_error` is what the model's *own* predictive distribution implies its
error should be, if the observed values were genuinely drawn from it. So:

- **clean column** (easy or hard): if the model is calibrated, observed is
  about equal to expected, and `excess ~= 0`. **The difficulty cancels,
  because both terms carry it in the same units.**
- **corrupted column**: a fraction `r` of cells were replaced by draws the
  model could not anticipate, so it is wrong *more often than it forecast*
  and `excess > 0`, growing with `r`.

This is a per-column **calibration gap**. `docs/literature.md` records that no
published paper evaluates the calibration quality of an error detector, which
is where this project's contribution already sits.

TWO CONSISTENT PAIRINGS, BOTH REPORTED
======================================
A gap is only meaningful if its two terms measure the same thing. There are
two valid pairings and mixing them across pairs is a bug:

- ``self`` (**primary**) — probabilistic.
  observed `1 - f(v_obs|x)` against expected `1 - sum_v f(v|x)^2`.
  Under a correct model these have the same expectation, so the gap is
  exactly a calibration residual.
- ``top1`` — hard 0/1.
  observed `argmax != v_obs` against expected `1 - max_v f(v|x)`.
  Also comparable, but discards the probability mass and is noisier.

CROSS-FITTING IS NOT OPTIONAL
=============================
An in-sample fit memorises the corrupted cells too, which deflates
`observed_error` and destroys the very signal being measured. All predictions
here are **out-of-fold**.

LEAKAGE
=======
Fitted on the dirty frame only. No clean data, no `DatasetProfile`, no
labels. Ground truth is used only to score the result afterwards, which is
the permitted evaluation use (`CLAUDE.md` §3).
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

#: Quantile bins for numeric columns, so one implementation covers every
#: column type. 10 matches the W8-A experiment's default.
DEFAULT_BINS = 10

#: A column needs at least this many distinct values to be modelled at all.
MIN_CLASSES = 2

#: Classes with fewer members than the fold count break a stratified split.
DEFAULT_FOLDS = 2

PAIRINGS = ("self", "top1")


@dataclasses.dataclass
class ColumnSurprise:
    """Per-column result. `excess_*` are the statistics of interest."""

    column: str
    n_classes: int
    observed_error: float
    expected_error_self: float
    expected_error_top1: float
    observed_error_01: float
    excess_self: float
    excess_top1: float
    skipped: str = ""

    @property
    def scored(self) -> bool:
        return not self.skipped

    def excess(self, pairing: str = "self") -> Optional[float]:
        if self.skipped:
            return None
        return self.excess_self if pairing == "self" else self.excess_top1

    def expected(self, pairing: str = "self") -> Optional[float]:
        if self.skipped:
            return None
        return (self.expected_error_self if pairing == "self"
                else self.expected_error_top1)

    def observed(self, pairing: str = "self") -> Optional[float]:
        if self.skipped:
            return None
        return (self.observed_error if pairing == "self"
                else self.observed_error_01)

    def to_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


def _codes(frame: pd.DataFrame, bins: int) -> Tuple[np.ndarray, List[int]]:
    """
    Encode every column as integer codes; numeric columns via quantile bins.

    Returns ``(codes, n_classes_per_column)``. Missing values get their own
    code, because "missing" is itself a corruption type here and must not be
    silently dropped.
    """
    cols: List[np.ndarray] = []
    sizes: List[int] = []
    for name in frame.columns:
        s = frame[name]
        if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > bins:
            num = pd.to_numeric(s, errors="coerce")
            try:
                binned = pd.qcut(num, q=bins, duplicates="drop", labels=False)
            except (ValueError, IndexError):
                binned = pd.Series(np.zeros(len(num)), index=num.index)
            code = pd.Series(binned).fillna(-1).to_numpy()
        else:
            code = pd.factorize(s.astype("object"), use_na_sentinel=False)[0]
        code = np.asarray(code, dtype=np.int64)
        cols.append(code)
        sizes.append(int(np.unique(code).size))
    return np.column_stack(cols) if cols else np.zeros((len(frame), 0)), sizes


def _out_of_fold_proba(
    X: np.ndarray, y: np.ndarray, seed: int, folds: int, max_iter: int
) -> Optional[np.ndarray]:
    """
    Out-of-fold ``predict_proba`` aligned to the global class ordering.

    Returns ``None`` when the column cannot be modelled. In-sample fitting is
    deliberately impossible here: it would deflate `observed_error` and
    manufacture the result.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold

    classes = np.unique(y)
    if classes.size < MIN_CLASSES:
        return None
    counts = pd.Series(y).value_counts()
    keep = set(counts[counts >= folds].index)
    usable = np.array([v in keep for v in y], dtype=bool)
    if usable.sum() < folds * 2 or np.unique(y[usable]).size < MIN_CLASSES:
        return None

    index = {c: i for i, c in enumerate(classes)}
    proba = np.full((len(y), classes.size), np.nan, dtype=float)
    idx = np.arange(len(y))[usable]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            skf = StratifiedKFold(n_splits=folds, shuffle=True,
                                  random_state=seed)
            for tr, te in skf.split(X[usable], y[usable]):
                model = HistGradientBoostingClassifier(
                    random_state=seed, max_iter=max_iter)
                model.fit(X[usable][tr], y[usable][tr])
                p = model.predict_proba(X[usable][te])
                for j, cls in enumerate(model.classes_):
                    proba[idx[te], index[int(cls)]] = p[:, j]
        except Exception:
            return None

    # Rows the folds could not cover fall back to the training marginal, which
    # is the most conservative choice available.
    bad = np.isnan(proba).any(axis=1)
    if bad.all():
        return None
    if bad.any():
        proba[bad] = np.nanmean(proba[~bad], axis=0)
    proba = np.clip(np.nan_to_num(proba, nan=0.0), 1e-12, 1.0)
    proba /= proba.sum(axis=1, keepdims=True)
    return proba


class ExcessSurprise:
    """
    Per-column calibration-gap detector.

    Parameters
    ----------
    bins : int
        Quantile bins for numeric columns.
    seed : int
        Random seed for the cross-fitting split and the estimators.
    folds : int
        Cross-fitting folds. 2 is the cheap default; more is slower and not
        obviously better for this statistic.
    max_iter : int
        Boosting iterations per fold.
    """

    def __init__(
        self,
        bins: int = DEFAULT_BINS,
        seed: int = 42,
        folds: int = DEFAULT_FOLDS,
        max_iter: int = 50,
    ) -> None:
        if bins < 2:
            raise ValueError(f"bins must be >= 2, got {bins}")
        if folds < 2:
            raise ValueError(f"folds must be >= 2, got {folds}")
        self.bins = int(bins)
        self.seed = int(seed)
        self.folds = int(folds)
        self.max_iter = int(max_iter)
        self.columns_: Dict[str, ColumnSurprise] = {}
        self.is_fitted_ = False

    # ------------------------------------------------------------------ #

    def fit(self, dirty_df: pd.DataFrame, profile: object = None
            ) -> "ExcessSurprise":
        """
        Compute the per-column calibration gap from the DIRTY frame alone.

        ``profile`` is accepted for interface parity with the project's other
        detectors and is **ignored**, so no clean-derived statistic can enter
        through it.
        """
        if not isinstance(dirty_df, pd.DataFrame):
            raise TypeError(f"dirty_df must be a DataFrame, got {type(dirty_df)!r}")
        if dirty_df.shape[0] == 0 or dirty_df.shape[1] == 0:
            raise ValueError(f"dirty_df is empty: shape {dirty_df.shape}")

        codes, sizes = _codes(dirty_df, self.bins)
        names = list(dirty_df.columns)
        self.columns_ = {}

        for j, name in enumerate(names):
            y = codes[:, j]
            n_cls = sizes[j]
            if n_cls < MIN_CLASSES:
                self.columns_[name] = ColumnSurprise(
                    name, n_cls, *(float("nan"),) * 6,
                    skipped=f"constant_after_discretisation({n_cls})")
                continue
            feat = np.delete(codes, j, axis=1).astype(float)
            if feat.shape[1] == 0:
                self.columns_[name] = ColumnSurprise(
                    name, n_cls, *(float("nan"),) * 6,
                    skipped="no_feature_columns")
                continue

            proba = _out_of_fold_proba(
                feat, y, self.seed, self.folds, self.max_iter)
            if proba is None:
                self.columns_[name] = ColumnSurprise(
                    name, n_cls, *(float("nan"),) * 6,
                    skipped="not_modellable")
                continue

            classes = np.unique(y)
            pos = {c: i for i, c in enumerate(classes)}
            obs_col = np.array([pos[v] for v in y], dtype=int)
            p_obs = proba[np.arange(len(y)), obs_col]

            # --- pairing "self": probabilistic, the primary statistic ----
            observed_error = float(np.mean(1.0 - p_obs))
            expected_self = float(np.mean(1.0 - np.sum(proba ** 2, axis=1)))

            # --- pairing "top1": hard 0/1 ---------------------------------
            pred = np.argmax(proba, axis=1)
            observed_01 = float(np.mean(pred != obs_col))
            expected_top1 = float(np.mean(1.0 - np.max(proba, axis=1)))

            self.columns_[name] = ColumnSurprise(
                column=name,
                n_classes=n_cls,
                observed_error=observed_error,
                expected_error_self=expected_self,
                expected_error_top1=expected_top1,
                observed_error_01=observed_01,
                excess_self=observed_error - expected_self,
                excess_top1=observed_01 - expected_top1,
            )

        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------ #

    def column_scores(self, pairing: str = "self") -> Dict[str, Optional[float]]:
        """``{column: excess}``, ``None`` for columns that were skipped."""
        self._check(pairing)
        return {k: v.excess(pairing) for k, v in self.columns_.items()}

    def ingredient_scores(
        self, which: str, pairing: str = "self"
    ) -> Dict[str, Optional[float]]:
        """
        ``{column: observed}`` or ``{column: expected}``.

        Exists so falsification condition 9 can be evaluated: the gap has to
        beat **both** of its own ingredients, or the subtraction earns
        nothing.
        """
        self._check(pairing)
        if which not in ("observed", "expected"):
            raise ValueError(f"which must be observed|expected, got {which!r}")
        return {
            k: (v.observed(pairing) if which == "observed" else v.expected(pairing))
            for k, v in self.columns_.items()
        }

    @property
    def skipped_columns_(self) -> Dict[str, str]:
        return {k: v.skipped for k, v in self.columns_.items() if v.skipped}

    def _check(self, pairing: str) -> None:
        if not self.is_fitted_:
            raise RuntimeError("called before fit()")
        if pairing not in PAIRINGS:
            raise ValueError(f"pairing must be one of {PAIRINGS}, got {pairing!r}")
