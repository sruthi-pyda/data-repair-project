"""
src/detectors/synth_supervised.py
---------------------------------
W8-B — **learned-policy synthetic supervision**: train a supervised per-cell
error classifier on corruption whose *policy is estimated from the data*
rather than designed.

Planned in ``docs/w8_plan.md`` §4 (W8-B), committed before this file existed.

WHY THIS MECHANISM
==================
``docs/w8_plan.md`` §2 decomposes the published edge into seven factors and
names **F2 — learn the corruption model from the dirty data, then train a
supervised detector on synthesised errors — as the single biggest factor this
project does not have**. HoloDetect (SIGMOD 2019, arXiv 1904.02285) is the
evidence: on Soccer at 10 % training data a *learned* augmentation policy
reaches F1 **0.953** while *random* transformations reach **0.166**. The same
ablation is the warning: a designed or random policy does not work, so the
whole mechanism turns on whether the policy can be estimated.

THE CENTRAL DIFFICULTY, STATED PLAINLY
======================================
Estimating the corruption policy requires knowing which cells are corrupt,
which is the problem being solved. The circularity is real and this module
does not hide it — it exposes it as the three-arm design the experiment runs:

================  ==========================================  ==============
arm               policy source                               deployable?
================  ==========================================  ==============
**0 designed**    ``src/datagen/corruptor.py``'s six designed  yes, zero
                  types. HoloDetect's "random transformations"  labels
                  arm.
**1 learned**     estimated from a **budget of labelled        yes, at the
                  cells** (default 20, the Raha/ED2/SAGED      stated budget
                  benchmarking budget). The budget is the
                  only clean data the arm sees, and the
                  budget size is reported with every number.
**2 oracle**      estimated from **all** true labels.          **NO** — upper
                  Measures how much of the gap is policy       bound only
                  quality rather than classifier capacity.
================  ==========================================  ==============

Arm 2 is **not a method**. Every function and report string that touches it
says "oracle".

WHAT A POLICY IS HERE
=====================
Per column ``c``, a :class:`ColumnPolicy` records:

* ``weight`` — the share of all corruptions that land in this column. This is
  what carries **column triage**: a policy estimated from adult's labels puts
  weight 1.0 on ``income``, a designed policy spreads weight over every
  eligible column.
* ``kind_probs`` — the distribution over *transition kinds*: ``missing``
  (value blanked), ``parse_break`` (a numeric became non-parseable),
  ``out_of_range`` (numeric left the column's observed range),
  ``swap`` (replaced by another value the column already holds), and
  ``numeric_shift`` (numeric moved inside range).
* ``transitions`` — for ``swap``, the empirical ``clean value -> dirty value``
  distribution. This is the part that matters on adult, whose real corruption
  is **entirely one-directional** (all 1,701 flips are ``income`` ``1 -> 0``,
  majority class to minority). A frequency-weighted designed ``swap_valid``
  does not reproduce that, which is exactly why arm 0 is expected to fail.

Estimating a transition distribution needs ``(clean, dirty)`` pairs, so the
budgeted label used here is the HoloDetect-style label ("this cell is wrong,
and the right value is v"), **stronger than Raha's binary "is this cell an
error"**. That is stated wherever arm 1 is reported; it is not smuggled in.

FEATURES — CROSS-COLUMN COMPARABILITY IS THE CONSTRAINT
=======================================================
Two earlier mechanisms in this project died on this exact point (W5.3's
cardinality confound, W6.1's per-column score scales). Raw conditional
scores carry column-specific units, so every **cell-scope** feature here is
either

* a **per-column empirical quantile** of a score (uniform on ``[0, 1]`` in
  every column by construction, whatever the column's cardinality or kind), or
* a **dimensionless indicator** (is-missing, does-not-parse).

Per-column quantiles alone, however, cannot rank *columns* — they are
uniform in every column, so a cell at quantile 0.99 of a clean column looks
exactly like a cell at quantile 0.99 of a corrupted one. Column triage
therefore needs **column-scope** features too, and those are restricted to
quantities that are dimensionless *and* comparable: the fraction of the
column's conditional scores above a fixed high cut, the column's mean
``ratio``-mode score (which divides out marginal rarity by construction, see
``src/detectors/conditional.py``), its missing fraction and its numeric
fraction.

**Cardinality and the rare-value fraction are deliberately NOT features.**
They are the two known confounds (W5.3, W6.1), and the experiment reports the
correlation of the mechanism's column statistic with both. A feature set that
contained them would make those correlations uninterpretable.

LEAKAGE (CLAUDE.md §3 — non-negotiable)
=======================================
* :class:`SynthSupervisedDetector` has **no** parameter that accepts clean
  data. It is fitted from ``(dirty_df, policy)`` and nothing else, and the
  conditional detector it uses internally is fitted with ``profile=None``.
* :func:`designed_policy` reads only ``dirty_df``. Arm 0 therefore reads no
  clean data at all, and ``tests/test_synth_supervised.py`` asserts it by
  handing the arm a sentinel frame that raises on any access.
* ``clean_df`` enters this module in exactly two clearly named places:
  :func:`label_cells` (the *budget*, arm 1 — a caller-chosen, counted set of
  cells) and :func:`oracle_labelled_cells` (arm 2 — the oracle). Both return
  labels; neither returns or stores a frame.
* Scoring against ground truth happens outside, in
  ``src/detection_eval/labels.py::cell_labels``, after predictions exist.

DETERMINISM
===========
Everything is seeded. The synthesiser draws from ``default_rng`` seeded by a
BLAKE2b digest of ``(seed, policy source, column)``, so the cells chosen in
one column do not depend on how many were chosen in another. Given the same
``(dirty_df, policy, seed)`` the synthesised frame and the fitted classifier
are reproducible, and the tests assert it.

WORKED EXAMPLE
==============
    >>> import pandas as pd
    >>> from src.detectors.synth_supervised import (
    ...     LabelledCell, estimate_policy)
    >>> dirty = pd.DataFrame({"a": [1, 1, 0, 1] * 30, "b": list("wxyz") * 30})
    >>> labels = [LabelledCell(0, "a", 0, 1, True),
    ...           LabelledCell(2, "a", 0, 1, True)]
    >>> pol = estimate_policy(dirty, labels, source="learned")
    >>> sorted(pol.columns)
    ['a']
    >>> pol.columns["a"].transitions
    {1: {0: 1.0}}
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.datagen.corruptor import inject
from src.detectors.conditional import ConditionalDetector

logger = logging.getLogger(__name__)

__all__ = [
    "CELL_FEATURES",
    "COLUMN_FEATURES",
    "FEATURE_NAMES",
    "ColumnPolicy",
    "CorruptionPolicy",
    "DESIGNED_TYPES",
    "LabelledCell",
    "SynthSupervisedDetector",
    "TRANSITION_KINDS",
    "choose_budget_cells",
    "column_quantile",
    "designed_policy",
    "estimate_policy",
    "extract_features",
    "label_cells",
    "oracle_balanced_budget",
    "oracle_labelled_cells",
    "synthesise",
]

#: Transition kinds a policy can hold. ``swap`` is the only kind that carries
#: a value-level transition distribution; the others are structural.
TRANSITION_KINDS: Tuple[str, ...] = (
    "missing",
    "parse_break",
    "out_of_range",
    "swap",
    "numeric_shift",
)

#: Designed corruption types for arm 0, in reporting order. These are
#: ``src/datagen/corruptor.py``'s types minus ``swap_conditional``, which is
#: excluded only because it needs the *clean* table's conditional structure
#: to place a value; every other type is applied to the dirty table directly.
DESIGNED_TYPES: Tuple[str, ...] = (
    "missing",
    "format",
    "out_of_range",
    "swap_valid",
    "typo",
)

#: Cell-scope features: per-column quantiles and dimensionless indicators.
CELL_FEATURES: Tuple[str, ...] = (
    "cond_q",
    "marg_q",
    "ratio_q",
    "freq_q",
    "len_dev_q",
    "is_missing",
    "is_parse_fail",
    "row_mean_cond_q",
    "row_max_cond_q",
)

#: Column-scope features. Dimensionless, and deliberately excluding
#: cardinality and the rare-value fraction (the two known confounds).
COLUMN_FEATURES: Tuple[str, ...] = (
    "col_frac_cond_high",
    "col_mean_ratio",
    "col_missing_frac",
    "col_numeric_frac",
)

FEATURE_NAMES: Tuple[str, ...] = CELL_FEATURES + COLUMN_FEATURES

#: Conditional-score cut used by ``col_frac_cond_high``. Fixed, not tuned:
#: the conditional score is ``1 - plausibility``, so 0.9 means "the observed
#: value had under 10 % plausibility".
COND_HIGH_CUT: float = 0.9

_EPS = 1e-12


# ═══════════════════════════════════════════════════════════════════════════
# Seeding
# ═══════════════════════════════════════════════════════════════════════════


def _stream_seed(seed: int, *parts: Any) -> int:
    """
    Reproducible 64-bit seed for one named stream.

    BLAKE2b of the parts rather than ``hash()``: Python salts string hashing
    per process, so ``hash()`` would not reproduce across runs. Same device
    as ``src/datagen/corruptor.py``.
    """
    payload = "|".join([str(int(seed))] + [str(p) for p in parts]).encode("utf-8")
    return int(struct.unpack("<Q", hashlib.blake2b(payload, digest_size=8).digest())[0])


def _rng(seed: int, *parts: Any) -> np.random.Generator:
    return np.random.default_rng(_stream_seed(seed, *parts))


# ═══════════════════════════════════════════════════════════════════════════
# Labels (the budget, and the oracle)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class LabelledCell:
    """
    One labelled cell.

    ``clean_value`` is populated only when ``is_corrupt`` is True, and it is
    what makes the HoloDetect-style transition distribution estimable. This
    is a **stronger label than Raha's binary flag** and must be described as
    such wherever an arm using it is reported.
    """

    row: int
    column: str
    dirty_value: Any
    clean_value: Any
    is_corrupt: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row": int(self.row),
            "column": str(self.column),
            "dirty_value": self.dirty_value,
            "clean_value": self.clean_value,
            "is_corrupt": bool(self.is_corrupt),
        }


def choose_budget_cells(
    dirty_df: pd.DataFrame,
    n: int,
    strategy: str = "random",
    scores: Optional[pd.DataFrame] = None,
    seed: int = 42,
) -> List[Tuple[int, str]]:
    """
    Pick which cells to spend the label budget on. **Reads no clean data.**

    Parameters
    ----------
    strategy : {'random', 'top_score'}
        ``random``
            uniform over all cells — the unbiased budget, and the one whose
            expected yield at adult's 0.25 % prevalence is 0.05 positives in
            20 labels. That arithmetic is the point, not a flaw to hide.
        ``top_score``
            the ``n`` highest-scoring cells under ``scores`` (an unsupervised
            detector's output). This is the Raha/ED2 habit of spending labels
            where a cheap detector is already suspicious, and it uses no
            ground truth to choose.

    Returns
    -------
    list[(row_position, column)]
        Deterministic for a given ``(dirty_df, n, strategy, scores, seed)``.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    cells = [(i, c) for c in dirty_df.columns for i in range(len(dirty_df))]
    if n == 0 or not cells:
        return []
    if strategy == "random":
        rng = _rng(seed, "budget", "random", n)
        idx = rng.choice(len(cells), size=min(n, len(cells)), replace=False)
        return [cells[int(i)] for i in np.sort(idx)]
    if strategy == "top_score":
        if scores is None:
            raise ValueError("strategy='top_score' needs `scores`")
        flat = [
            (float(scores[c].iat[i]) if np.isfinite(_as_float(scores[c].iat[i])) else 0.0,
             i, c)
            for (i, c) in cells
        ]
        # Deterministic tie-break on (column, row) so equal scores do not
        # depend on sort stability of a float comparison.
        flat.sort(key=lambda t: (-t[0], str(t[2]), t[1]))
        return [(i, c) for _, i, c in flat[:n]]
    raise ValueError(f"unknown strategy {strategy!r}")


def _as_float(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def label_cells(
    clean_df: pd.DataFrame,
    dirty_df: pd.DataFrame,
    cells: Sequence[Tuple[int, str]],
) -> List[LabelledCell]:
    """
    **The budget oracle.** Reveal the labels for exactly ``cells``.

    This is one of the two places ``clean_df`` enters this module. It returns
    labels for the cells the caller already chose (without clean data) and
    never returns or retains a frame, so the information released is bounded
    by ``len(cells)`` — which is the budget, and is reported.
    """
    out: List[LabelledCell] = []
    for row, col in cells:
        dv = dirty_df[col].iat[row]
        cv = clean_df[col].iat[row]
        corrupt = not _values_equal(dv, cv)
        out.append(
            LabelledCell(
                row=int(row),
                column=str(col),
                dirty_value=dv,
                clean_value=cv if corrupt else dv,
                is_corrupt=bool(corrupt),
            )
        )
    return out


def oracle_labelled_cells(
    clean_df: pd.DataFrame,
    dirty_df: pd.DataFrame,
) -> List[LabelledCell]:
    """
    **ORACLE — not deployable.** Every truly corrupted cell, labelled.

    Arm 2 only. It exists to separate "the policy is wrong" from "the
    classifier cannot learn", and every number derived from it is labelled
    ``oracle`` in the report.
    """
    out: List[LabelledCell] = []
    for col in dirty_df.columns:
        dv = dirty_df[col].to_numpy()
        cv = clean_df[col].to_numpy()
        for i in range(len(dirty_df)):
            if not _values_equal(dv[i], cv[i]):
                out.append(
                    LabelledCell(
                        row=i, column=str(col), dirty_value=dv[i],
                        clean_value=cv[i], is_corrupt=True,
                    )
                )
    return out


def oracle_balanced_budget(
    clean_df: pd.DataFrame,
    dirty_df: pd.DataFrame,
    n: int,
    seed: int = 42,
) -> List[LabelledCell]:
    """
    **SEMI-ORACLE — not deployable.** ``n`` labels, half of them forced onto
    genuinely corrupted cells.

    Choosing a balanced sample requires knowing where the corruptions are, so
    this is *not* a budget anyone could spend in practice. It is reported as
    arm 1b for one diagnostic purpose: it separates **budget placement** (can
    20 random labels even find a positive?) from **policy estimation** (given
    a handful of positives, is the estimated policy good enough?). Without it,
    an arm-1 failure is uninterpretable.
    """
    allc = oracle_labelled_cells(clean_df, dirty_df)
    rng = _rng(seed, "oracle_balanced", n)
    n_pos = min(len(allc), max(1, n // 2))
    pos_idx = rng.choice(len(allc), size=n_pos, replace=False) if allc else []
    pos = [allc[int(i)] for i in np.sort(pos_idx)] if len(pos_idx) else []
    pos_keys = {(p.row, p.column) for p in pos}
    neg_needed = n - len(pos)
    negs: List[LabelledCell] = []
    cells = [(i, c) for c in dirty_df.columns for i in range(len(dirty_df))]
    order = rng.permutation(len(cells))
    corrupt_keys = {(c.row, c.column) for c in allc}
    for j in order:
        if len(negs) >= neg_needed:
            break
        row, col = cells[int(j)]
        if (row, col) in pos_keys or (row, col) in corrupt_keys:
            continue
        dv = dirty_df[col].iat[row]
        negs.append(LabelledCell(row, str(col), dv, dv, False))
    return pos + negs


def _values_equal(a: Any, b: Any) -> bool:
    """
    Loose equality matching ``src/detection_eval/labels.py``'s intent:
    ``NaN == NaN`` and ``371 == 371.0``.
    """
    a_na, b_na = _is_na(a), _is_na(b)
    if a_na or b_na:
        return a_na and b_na
    fa, fb = _as_float(a), _as_float(b)
    if math.isfinite(fa) and math.isfinite(fb):
        return abs(fa - fb) <= 1e-9
    return str(a) == str(b)


def _is_na(v: Any) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


# ═══════════════════════════════════════════════════════════════════════════
# The policy
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ColumnPolicy:
    """What corruption looks like in one column. See the module docstring."""

    column: str
    weight: float
    kind_probs: Dict[str, float]
    transitions: Dict[Any, Dict[Any, float]] = field(default_factory=dict)
    value_dist: Dict[Any, float] = field(default_factory=dict)
    n_observed: int = 0

    @property
    def is_one_directional(self) -> bool:
        """
        True when every observed transition writes the **same** dirty value.

        This is the structural property adult's real corruption has (all
        flips ``1 -> 0``), and the experiment reports per arm whether the
        estimated policy recovers it.
        """
        targets = {t for d in self.transitions.values() for t in d}
        return len(targets) == 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "weight": float(self.weight),
            "kind_probs": {k: float(v) for k, v in self.kind_probs.items()},
            "transitions": {
                str(k): {str(t): float(p) for t, p in d.items()}
                for k, d in self.transitions.items()
            },
            "n_observed": int(self.n_observed),
            "one_directional": bool(self.is_one_directional),
        }


@dataclass(frozen=True)
class CorruptionPolicy:
    """
    A corruption policy, and where it came from.

    ``source`` is one of ``designed`` (arm 0), ``learned`` (arm 1) or
    ``oracle`` (arm 2) and is carried into every report string so an oracle
    number can never be printed without its label.
    """

    source: str
    rate: float
    columns: Dict[str, ColumnPolicy]
    n_labels: int = 0
    n_positive_labels: int = 0
    notes: Tuple[str, ...] = ()

    @property
    def is_oracle(self) -> bool:
        return self.source == "oracle"

    @property
    def is_empty(self) -> bool:
        """
        True when nothing can be synthesised from this policy.

        A ``designed`` policy legitimately carries no per-column entries —
        the corruptor decides eligibility itself — so only a non-positive
        rate makes it empty. An estimated policy with no columns means the
        budget found no corruption, which is the empty case that matters.
        """
        if self.rate <= 0.0:
            return True
        return self.source != "designed" and not self.columns

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "rate": float(self.rate),
            "n_labels": int(self.n_labels),
            "n_positive_labels": int(self.n_positive_labels),
            "columns": {c: p.to_dict() for c, p in self.columns.items()},
            "notes": list(self.notes),
        }


def designed_policy(dirty_df: pd.DataFrame, rate: float = 0.02) -> CorruptionPolicy:
    """
    Arm 0's policy marker. **Reads no clean data.**

    The designed arm does not estimate anything, so this object carries only
    the rate; :func:`synthesise` routes it to ``src/datagen/corruptor.py``,
    which applies :data:`DESIGNED_TYPES` to the dirty table. The per-column
    weights recorded here are the corruptor's own eligibility spread, filled
    in by :func:`synthesise` after the fact for reporting.
    """
    if dirty_df.shape[0] == 0 or dirty_df.shape[1] == 0:
        raise ValueError(f"dirty_df is empty: shape {dirty_df.shape}")
    return CorruptionPolicy(
        source="designed",
        rate=float(rate),
        columns={},
        notes=("designed types: " + ", ".join(DESIGNED_TYPES),),
    )


def estimate_policy(
    dirty_df: pd.DataFrame,
    labelled: Sequence[LabelledCell],
    source: str = "learned",
    min_rate: float = 1e-4,
) -> CorruptionPolicy:
    """
    Estimate a corruption policy from labelled ``(clean, dirty)`` pairs.

    The only clean information used is what is already inside ``labelled``,
    which the caller obtained through :func:`label_cells` (budgeted) or
    :func:`oracle_labelled_cells` (oracle). ``dirty_df`` supplies the column
    statistics needed to classify a transition kind (observed range, value
    vocabulary) and contains no labels.

    Returns an **empty policy** when no labelled cell is corrupt. That is the
    honest outcome of a budget that found nothing, and
    :class:`SynthSupervisedDetector` refuses to fit on it rather than
    silently substituting a designed policy.
    """
    positives = [c for c in labelled if c.is_corrupt]
    n_labels = len(labelled)
    if not positives:
        return CorruptionPolicy(
            source=source,
            rate=0.0,
            columns={},
            n_labels=n_labels,
            n_positive_labels=0,
            notes=(
                f"{n_labels} labelled cells contained 0 corrupted cells; no "
                f"policy is estimable. This is a measurement, not a failure "
                f"to compute.",
            ),
        )

    rate = max(float(len(positives)) / max(n_labels, 1), min_rate)
    by_col: Dict[str, List[LabelledCell]] = {}
    for c in positives:
        by_col.setdefault(c.column, []).append(c)

    stats = {col: _column_stats(dirty_df[col]) for col in by_col}
    columns: Dict[str, ColumnPolicy] = {}
    total = float(len(positives))
    for col, cells in by_col.items():
        kinds: Dict[str, int] = {}
        trans: Dict[Any, Dict[Any, int]] = {}
        values: Dict[Any, int] = {}
        for cell in cells:
            kind = _transition_kind(cell, stats[col])
            kinds[kind] = kinds.get(kind, 0) + 1
            if kind in ("swap", "numeric_shift", "out_of_range"):
                key = _canon(cell.clean_value)
                tgt = _canon(cell.dirty_value)
                trans.setdefault(key, {})
                trans[key][tgt] = trans[key].get(tgt, 0) + 1
                values[tgt] = values.get(tgt, 0) + 1
        columns[col] = ColumnPolicy(
            column=col,
            weight=len(cells) / total,
            kind_probs=_normalise(kinds),
            transitions={k: _normalise(v) for k, v in trans.items()},
            value_dist=_normalise(values),
            n_observed=len(cells),
        )

    return CorruptionPolicy(
        source=source,
        rate=rate,
        columns=columns,
        n_labels=n_labels,
        n_positive_labels=len(positives),
        notes=(
            f"estimated from {len(positives)} corrupted cells among "
            f"{n_labels} labels",
        ),
    )


def _normalise(counts: Mapping[Any, int]) -> Dict[Any, float]:
    total = float(sum(counts.values()))
    if total <= 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def _canon(v: Any) -> Any:
    """Hashable canonical form: ``371.0`` and ``371`` become the same key."""
    if _is_na(v):
        return None
    f = _as_float(v)
    if math.isfinite(f):
        if abs(f - round(f)) <= 1e-9:
            return int(round(f))
        return float(f)
    return str(v)


@dataclass(frozen=True)
class _ColStats:
    """
    Column statistics used to classify a transition. Computed from the
    **dirty** column, which is the only column available deployably — and
    which therefore already contains the corruption. That is why the range
    test below uses a Tukey fence rather than the observed min/max (the
    corrupted value *is* the min or max, so min/max could never flag it) and
    why the vocabulary test requires a value to occur at least twice (a
    one-off value is evidence of itself, not of a vocabulary).
    """

    numeric_frac: float
    lo: float
    hi: float
    counts: Dict[Any, int]


#: Tukey multiplier for the ``out_of_range`` test, matching the shipped rule
#: detector's ``k = 3`` fence so "out of range" means one thing project-wide.
FENCE_K: float = 3.0

#: A value must occur at least this many times in the dirty column before a
#: transition into it counts as a ``swap`` rather than a novel value.
MIN_VOCAB_COUNT: int = 2


def _column_stats(series: pd.Series) -> _ColStats:
    non_null = series.dropna()
    if non_null.empty:
        return _ColStats(0.0, float("nan"), float("nan"), {})
    arr = non_null.to_numpy()
    vals = [_as_float(v) for v in arr]
    finite = np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
    frac = finite.size / len(vals)
    if finite.size:
        q1, q3 = float(np.quantile(finite, 0.25)), float(np.quantile(finite, 0.75))
        iqr = q3 - q1
        lo, hi = q1 - FENCE_K * iqr, q3 + FENCE_K * iqr
    else:
        lo = hi = float("nan")
    counts: Dict[Any, int] = {}
    for v in arr:
        k = _canon(v)
        counts[k] = counts.get(k, 0) + 1
    return _ColStats(frac, lo, hi, counts)


def _transition_kind(cell: LabelledCell, stats: _ColStats) -> str:
    """Classify one ``(clean -> dirty)`` pair into a :data:`TRANSITION_KINDS`."""
    if _is_na(cell.dirty_value):
        return "missing"
    numeric_column = stats.numeric_frac >= 0.8
    f_dirty = _as_float(cell.dirty_value)
    if numeric_column and not math.isfinite(f_dirty):
        return "parse_break"
    if numeric_column and math.isfinite(stats.lo) and math.isfinite(stats.hi):
        if f_dirty < stats.lo or f_dirty > stats.hi:
            return "out_of_range"
    if stats.counts.get(_canon(cell.dirty_value), 0) >= MIN_VOCAB_COUNT:
        return "swap"
    return "numeric_shift" if numeric_column else "swap"


# ═══════════════════════════════════════════════════════════════════════════
# Synthesis
# ═══════════════════════════════════════════════════════════════════════════


def synthesise(
    dirty_df: pd.DataFrame,
    policy: CorruptionPolicy,
    seed: int = 42,
    target_positives: int = 2000,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Apply ``policy`` to ``dirty_df``, producing a training table.

    Returns ``(synth_df, injected_mask, info)``. ``injected_mask`` is a
    boolean frame, True exactly where this function wrote a value.

    Note what the training labels mean: the synthetic positives are cells
    **this function corrupted**, while the negatives are every other cell,
    which still contains the table's own real (unknown) corruptions at the
    dataset's prevalence. That is label noise of known magnitude, it is
    unavoidable without the labels we are trying not to use, and it is
    reported rather than assumed away.

    ``target_positives`` bounds the number of injected cells so training
    cost does not scale with table size.
    """
    if policy.is_empty:
        raise ValueError(
            f"policy from {policy.source!r} is empty ({policy.notes}); "
            f"nothing can be synthesised"
        )
    n_rows, n_cols = dirty_df.shape
    n_cells = n_rows * n_cols
    n_target = int(min(target_positives, max(1, round(policy.rate * n_cells))))

    if policy.source == "designed":
        return _synthesise_designed(dirty_df, n_target, seed)
    return _synthesise_learned(dirty_df, policy, n_target, seed)


def _blank_mask(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        np.zeros(df.shape, dtype=bool), index=df.index, columns=df.columns
    )


def _synthesise_designed(
    dirty_df: pd.DataFrame, n_target: int, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Arm 0: the designed types from ``src/datagen/corruptor.py``, applied to
    the **dirty** table (never the clean one), split evenly across types.
    """
    n_cells = dirty_df.shape[0] * dirty_df.shape[1]
    per_type_rate = (n_target / len(DESIGNED_TYPES)) / n_cells
    out = dirty_df.copy()
    mask = _blank_mask(dirty_df)
    per_type: Dict[str, int] = {}
    for ctype in DESIGNED_TYPES:
        try:
            out, report = inject(out, ctype, per_type_rate, seed=seed)
        except ValueError as exc:  # pragma: no cover - defensive
            logger.warning("designed type %s failed: %s", ctype, exc)
            per_type[ctype] = 0
            continue
        for row, col in report.positions:
            mask.iat[row, mask.columns.get_loc(col)] = True
        per_type[ctype] = report.n_injected
    info = {
        "source": "designed",
        "n_injected": int(mask.to_numpy().sum()),
        "per_type": per_type,
        "per_column": {
            c: int(mask[c].sum()) for c in mask.columns if mask[c].any()
        },
    }
    return out, mask, info


def _synthesise_learned(
    dirty_df: pd.DataFrame,
    policy: CorruptionPolicy,
    n_target: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Arms 1 and 2: apply the estimated per-column policies."""
    out = dirty_df.copy()
    mask = _blank_mask(dirty_df)
    n_rows = len(dirty_df)
    per_column: Dict[str, int] = {}

    for col, cp in sorted(policy.columns.items()):
        if col not in out.columns:
            continue
        n_here = int(round(cp.weight * n_target))
        if n_here <= 0:
            continue
        rng = _rng(seed, policy.source, "synth", col)
        rows = rng.permutation(n_rows)
        series = out[col].astype(object)
        stats = _column_stats(dirty_df[col])
        kinds = list(cp.kind_probs)
        kprob = np.asarray([cp.kind_probs[k] for k in kinds], dtype=float)
        kprob = kprob / kprob.sum() if kprob.sum() > 0 else None
        written = 0
        for r in rows:
            if written >= n_here:
                break
            r = int(r)
            current = series.iat[r]
            if _is_na(current):
                continue
            kind = kinds[int(rng.choice(len(kinds), p=kprob))] if kinds else "swap"
            new = _apply_kind(kind, current, cp, stats, rng)
            if new is None or _values_equal(new, current):
                continue
            series.iat[r] = new
            mask.iat[r, mask.columns.get_loc(col)] = True
            written += 1
        out[col] = _restore_dtype(series, dirty_df[col])
        per_column[col] = written

    info = {
        "source": policy.source,
        "n_injected": int(mask.to_numpy().sum()),
        "per_column": per_column,
        "policy_columns": sorted(policy.columns),
    }
    return out, mask, info


def _apply_kind(
    kind: str,
    current: Any,
    cp: ColumnPolicy,
    stats: _ColStats,
    rng: np.random.Generator,
) -> Any:
    """One corruption, drawn from the estimated policy."""
    if kind == "missing":
        return np.nan
    if kind == "parse_break":
        text = str(current)
        return text + "x" if text else "x"
    if kind == "out_of_range":
        if not math.isfinite(stats.hi) or not math.isfinite(stats.lo):
            return None
        span = max(stats.hi - stats.lo, 1.0)
        return stats.hi + span * float(rng.choice([5.0, 10.0, 20.0]))
    # swap / numeric_shift: use the estimated transition distribution
    key = _canon(current)
    table = cp.transitions.get(key)
    if not table:
        table = cp.value_dist
    if not table:
        return None
    keys = list(table)
    probs = np.asarray([table[k] for k in keys], dtype=float)
    if probs.sum() <= 0:
        return None
    probs = probs / probs.sum()
    return keys[int(rng.choice(len(keys), p=probs))]


def _restore_dtype(series: pd.Series, original: pd.Series) -> pd.Series:
    new = series.infer_objects()
    if new.dtype != original.dtype and not new.isna().any():
        try:
            cast = new.astype(original.dtype)
        except (TypeError, ValueError, OverflowError):
            return new
        if bool((cast == new).all()):
            return cast
    return new


# ═══════════════════════════════════════════════════════════════════════════
# Features
# ═══════════════════════════════════════════════════════════════════════════


def column_quantile(values: np.ndarray) -> np.ndarray:
    """
    Per-column empirical quantile in ``[0, 1]``, ties averaged.

    This is the cross-column comparability device: whatever a column's scale,
    cardinality or kind, the output is (up to ties) uniform on ``[0, 1]``.
    NaNs map to 0.5, the uninformative middle, so a missing score never
    reads as extreme.

        >>> column_quantile(np.array([3.0, 1.0, 2.0]))
        array([1. , 0. , 0.5])
    """
    x = np.asarray(values, dtype=float)
    out = np.full(x.shape, 0.5, dtype=float)
    ok = np.isfinite(x)
    n = int(ok.sum())
    if n == 0:
        return out
    if n == 1:
        out[ok] = 0.5
        return out
    v = x[ok]
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    # average ties, so a constant column maps to 0.5 everywhere
    uniq, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.zeros(uniq.size, dtype=float)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    out[ok] = ranks / (n - 1)
    return out


def extract_features(
    df: pd.DataFrame,
    detector: Optional[ConditionalDetector] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, List[Tuple[int, str]], ConditionalDetector]:
    """
    Per-cell feature matrix for ``df``.

    Returns ``(X, cells, detector)`` where ``cells[k]`` is the
    ``(row_position, column)`` of row ``k`` of ``X`` and the column order of
    ``X`` is :data:`FEATURE_NAMES`.

    A :class:`~src.detectors.conditional.ConditionalDetector` is fitted on
    ``df`` if one is not supplied (``profile=None``, dirty data only). The
    feature values are then *properties of this table*: ranks are taken
    within this table's columns, so train and test tables are each
    self-normalised, which is what makes a classifier trained on a
    synthesised table applicable to the real one.
    """
    if detector is None:
        detector = ConditionalDetector(mode="full", seed=seed).fit(df)
    s_cond = detector.score_cells(df)
    s_marg = detector.score_cells(df, mode="marginal")
    s_ratio = detector.score_cells(df, mode="ratio")

    n_rows, n_cols = df.shape
    cond_q = np.empty((n_rows, n_cols), dtype=float)
    per_col: Dict[str, Dict[str, np.ndarray]] = {}
    col_level: Dict[str, Dict[str, float]] = {}

    for j, col in enumerate(df.columns):
        raw_cond = s_cond[col].to_numpy(dtype=float)
        raw_marg = s_marg[col].to_numpy(dtype=float)
        raw_ratio = s_ratio[col].to_numpy(dtype=float)
        series = df[col]
        values = series.to_numpy()

        missing = np.asarray([_is_na(v) for v in values], dtype=float)
        floats = np.asarray([_as_float(v) for v in values], dtype=float)
        numeric_frac = float(
            np.isfinite(floats[missing == 0]).mean()
        ) if (missing == 0).any() else 0.0
        parse_fail = np.asarray(
            [
                1.0 if (numeric_frac >= 0.8 and m == 0 and not np.isfinite(f)) else 0.0
                for m, f in zip(missing, floats)
            ],
            dtype=float,
        )

        counts = series.astype(object).where(series.notna(), None).value_counts(
            dropna=False
        )
        freq = np.asarray(
            [float(counts.get(v if not _is_na(v) else None, 0)) for v in values],
            dtype=float,
        )
        lengths = np.asarray(
            [float(len(str(v))) if not _is_na(v) else np.nan for v in values],
            dtype=float,
        )
        med_len = float(np.nanmedian(lengths)) if np.isfinite(lengths).any() else 0.0
        len_dev = np.abs(lengths - med_len)

        cq = column_quantile(raw_cond)
        cond_q[:, j] = cq
        per_col[col] = {
            "cond_q": cq,
            "marg_q": column_quantile(raw_marg),
            "ratio_q": column_quantile(raw_ratio),
            # rare values should score HIGH, so invert the frequency quantile
            "freq_q": 1.0 - column_quantile(freq),
            "len_dev_q": column_quantile(len_dev),
            "is_missing": missing,
            "is_parse_fail": parse_fail,
        }
        finite_cond = raw_cond[np.isfinite(raw_cond)]
        finite_ratio = raw_ratio[np.isfinite(raw_ratio)]
        col_level[col] = {
            "col_frac_cond_high": float(
                (finite_cond > COND_HIGH_CUT).mean()
            ) if finite_cond.size else 0.0,
            "col_mean_ratio": float(finite_ratio.mean()) if finite_ratio.size else 0.0,
            "col_missing_frac": float(missing.mean()),
            "col_numeric_frac": numeric_frac,
        }

    row_mean = cond_q.mean(axis=1)
    row_max = cond_q.max(axis=1)

    cells: List[Tuple[int, str]] = []
    blocks: List[np.ndarray] = []
    for col in df.columns:
        pc = per_col[col]
        cl = col_level[col]
        block = np.column_stack(
            [pc["cond_q"], pc["marg_q"], pc["ratio_q"], pc["freq_q"],
             pc["len_dev_q"], pc["is_missing"], pc["is_parse_fail"],
             row_mean, row_max]
            + [np.full(n_rows, cl[name], dtype=float) for name in COLUMN_FEATURES]
        )
        blocks.append(block)
        cells.extend((i, col) for i in range(n_rows))

    x = np.vstack(blocks)
    assert x.shape == (n_rows * n_cols, len(FEATURE_NAMES))
    return x, cells, detector


# ═══════════════════════════════════════════════════════════════════════════
# The detector
# ═══════════════════════════════════════════════════════════════════════════


class SynthSupervisedDetector:
    """
    Train a per-cell error classifier on policy-synthesised corruption.

    Parameters
    ----------
    policy : CorruptionPolicy
        Where the synthetic corruption comes from. Its ``source`` is carried
        into :attr:`report` so an oracle-derived score can never be reported
        unlabelled.
    seed : int
    target_positives : int
        Upper bound on synthesised positives (training cost control).
    negatives_per_positive : int
        Negatives sampled per synthetic positive. The full negative set is
        hundreds of thousands of cells, almost all uninformative; sampling
        bounds the fit and the ratio is recorded.

    Leakage
    -------
    ``fit`` and ``score_cells`` take the dirty frame and nothing else. There
    is no parameter named or derived from clean data; the policy object holds
    estimated distributions, not cells of the clean table.
    """

    def __init__(
        self,
        policy: CorruptionPolicy,
        seed: int = 42,
        target_positives: int = 2000,
        negatives_per_positive: int = 10,
        max_iter: int = 100,
    ) -> None:
        if policy.is_empty:
            raise ValueError(
                f"policy from {policy.source!r} is empty: {policy.notes}. "
                f"Refusing to fit — a budget that found no corruption cannot "
                f"supply a policy, and substituting a designed one would "
                f"silently turn arm 1 into arm 0."
            )
        if negatives_per_positive < 1:
            raise ValueError("negatives_per_positive must be >= 1")
        self.policy = policy
        self.seed = int(seed)
        self.target_positives = int(target_positives)
        self.negatives_per_positive = int(negatives_per_positive)
        self.max_iter = int(max_iter)

        self.model_ = None
        self.synth_info_: Dict[str, Any] = {}
        self.n_train_: int = 0
        self.n_train_positive_: int = 0
        self.is_fitted_: bool = False
        self._columns: List[str] = []

    def fit(self, dirty_df: pd.DataFrame) -> "SynthSupervisedDetector":
        """Synthesise, extract features, train. Dirty data only."""
        from sklearn.ensemble import HistGradientBoostingClassifier

        self._columns = list(dirty_df.columns)
        synth_df, mask, info = synthesise(
            dirty_df, self.policy, seed=self.seed,
            target_positives=self.target_positives,
        )
        self.synth_info_ = info
        if int(mask.to_numpy().sum()) == 0:
            raise ValueError(
                f"policy {self.policy.source!r} injected 0 cells; nothing to "
                f"train on. info={info}"
            )

        x, cells, _ = extract_features(synth_df, seed=self.seed)
        # `cells` is column-major (all of column 0's rows, then column 1's),
        # which is exactly ``mask.to_numpy().T.ravel()``. Asserted, not
        # assumed, because a silent mis-alignment here would train the
        # classifier on shuffled labels and still "work".
        y = mask.to_numpy().T.ravel().astype(int)
        assert y.size == len(cells)
        assert cells[len(mask)] == (0, mask.columns[1]) or mask.shape[1] == 1

        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        rng = _rng(self.seed, "negatives", self.policy.source)
        n_neg = min(neg.size, pos.size * self.negatives_per_positive)
        keep_neg = np.sort(rng.choice(neg.size, size=n_neg, replace=False))
        idx = np.sort(np.concatenate([pos, neg[keep_neg]]))

        x_fit, y_fit = x[idx], y[idx]
        self.n_train_ = int(x_fit.shape[0])
        self.n_train_positive_ = int(y_fit.sum())
        if len(np.unique(y_fit)) < 2:
            raise ValueError("training set has a single class")

        self.model_ = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            random_state=self.seed,
            early_stopping=False,
        ).fit(x_fit, y_fit)
        self.is_fitted_ = True
        logger.info(
            "SynthSupervisedDetector(%s): %d train cells, %d positive",
            self.policy.source, self.n_train_, self.n_train_positive_,
        )
        return self

    def score_cells(
        self,
        dirty_df: pd.DataFrame,
        detector: Optional[ConditionalDetector] = None,
    ) -> pd.DataFrame:
        """
        ``P(cell is corrupted)`` for every cell, as a frame shaped like
        ``dirty_df``.

        ``detector`` lets a caller reuse one conditional fit across arms;
        it is fitted on ``dirty_df`` and carries no clean data either way.
        """
        if not self.is_fitted_:
            raise RuntimeError("score_cells called before fit")
        if list(dirty_df.columns) != self._columns:
            raise ValueError(
                f"column mismatch: fitted on {self._columns}, "
                f"got {list(dirty_df.columns)}"
            )
        x, cells, _ = extract_features(dirty_df, detector=detector, seed=self.seed)
        p = self.model_.predict_proba(x)[:, 1]
        n_rows, n_cols = dirty_df.shape
        # Same column-major ordering as `cells`; checked rather than trusted.
        assert cells[0] == (0, dirty_df.columns[0])
        assert cells[1] == (1, dirty_df.columns[0]) or n_rows == 1
        return pd.DataFrame(
            p.reshape(n_cols, n_rows).T,
            index=dirty_df.index, columns=dirty_df.columns,
        )

    def report(self) -> Dict[str, Any]:
        return {
            "policy_source": self.policy.source,
            "is_oracle": self.policy.is_oracle,
            "policy": self.policy.to_dict(),
            "synth": dict(self.synth_info_),
            "n_train": self.n_train_,
            "n_train_positive": self.n_train_positive_,
            "features": list(FEATURE_NAMES),
            "is_fitted": self.is_fitted_,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "fitted" if self.is_fitted_ else "unfitted"
        return (
            f"SynthSupervisedDetector(policy={self.policy.source!r}, "
            f"seed={self.seed}, {state})"
        )
