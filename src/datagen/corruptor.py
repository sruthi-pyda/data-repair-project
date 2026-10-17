"""
src/datagen/corruptor.py
------------------------
Controlled, seeded corruption injection into a clean table.

THE SIX CORRUPTION TYPES, AND WHAT EACH ONE TESTS
=================================================
Each type is placed deliberately on the recoverable/destroyed axis that
``docs/PROJECT_REPORT.md`` §2 established as the governing constraint, and on
the syntactic/semantic axis that ``docs/research_program.md`` §5 established
as the rule-vs-conditional detector boundary.

====================  ===========  ==========  =====================================
type                  recoverable  detectable  what it is
====================  ===========  ==========  =====================================
``missing``           no           trivially   the cell is blanked (NaN)
``format``            yes          syntactic   a numeric is mangled into a
                                               non-parseable string (``10`` -> ``1O``)
``out_of_range``      yes          syntactic   a numeric is replaced by a value far
                                               outside the observed range
``swap_valid``        no           **not**     replaced by another value drawn from
                                               the SAME column's marginal
``swap_conditional``  no           semantic    replaced by a value that is valid
                                               globally but improbable given the row
``typo``              partly       syntactic   character-level perturbation of a
                                               categorical string
====================  ===========  ==========  =====================================

``swap_valid`` is the honesty anchor: the written value already occurs in that
column, so no vocabulary, range, format or missing-token rule can see it. A
rule detector *must* score at chance there.

``swap_conditional`` is the case the whole research program turns on. The
value is chosen to be inside the column's global 1.5x IQR fence (numeric) or
to be a globally *common* category (categorical) while being absent or
near-absent from the rows that share the target row's conditioning bin. So:
globally unremarkable, conditionally improbable. A rule cannot see it; a model
of ``P(cell | rest of row)`` can.

WHAT "RATE" MEANS
=================
``rate`` is the fraction of **all** cells in the frame
(``n_rows * n_columns``) that are corrupted, so rates are comparable across
types and across datasets. Not every column is eligible for every type (you
cannot mangle the format of a free-text category, and there is no sensible
typo for an integer), so the eligible pool can be smaller than the requested
count. When that happens the whole pool is corrupted and
:attr:`InjectionReport.achieved_rate` comes out below
:attr:`InjectionReport.requested_rate`. The shortfall is **reported, never
silently rounded away** — a results table that quotes a requested rate the
injector could not reach is a fabricated number.

COLUMNS THAT ARE NEVER CORRUPTED
================================
ID-like columns are excluded from every type. A column counts as ID-like when
it has more than ``id_min_unique`` distinct non-null values AND
``distinct / non-null >= id_unique_ratio``, and its values are integer-valued
or non-numeric. These thresholds are copied verbatim from
:class:`src.detectors.conditional.ConditionalDetector` so that the columns the
injector skips are exactly the columns the conditional detector skips. On
Breast Cancer that is ``Unnamed: 0`` (699/699) and ``Sample code number``
(645/699); Adult's ``fnlwgt`` (26,741/45,222 = 0.59) stays eligible.

Corrupting an identifier is meaningless: a swapped ID is undetectable by
anything and a mangled one is detectable by everything, so either way the
number says more about the column than about the method.

DETERMINISM
===========
Two independent seeded streams:

* **positions** — one permutation of the eligible cell pool, drawn from
  ``default_rng`` seeded by ``(seed, corruption_type, rate)``. The pool is
  built in a fixed (column order, row order) sequence before shuffling, so it
  does not depend on dict iteration order.
* **values** — one generator per cell, seeded by a BLAKE2b digest of
  ``(seed, corruption_type, column, row)``. ``hashlib`` is used rather than
  the built-in ``hash`` because string hashing is salted per process and
  would not reproduce. Per-cell seeding means the value written at one cell
  does not depend on how many cells before it happened to be skipped.

Consequence, asserted by the tests: the same ``(clean_df, type, rate, seed)``
produces a bit-identical frame and an identical position list, on every run
and every platform.

DTYPES
======
The corrupted frame always has the clean frame's shape, column order and
index. Dtypes are preserved **where the corruption allows it**:

* ``swap_valid`` / ``swap_conditional`` write values taken from the column
  itself, so an integer column stays ``int64``.
* ``missing`` writes NaN, which upcasts an integer column to ``float64``.
  This is exactly what the shipped ``dirty.csv`` files do
  (``src/data/loader.py`` documents the same upcast).
* ``format`` and ``typo`` write strings, so the column becomes ``object``
  with mixed types — again matching the shipped data, where Breast Cancer's
  ``Bare Nuclei`` is ``object`` because of its format errors.
* ``out_of_range`` writes a value of the column's own kind and keeps the
  dtype unless the column was already ``object``.

Worked example
==============
    >>> import pandas as pd
    >>> from src.datagen.corruptor import Corruptor
    >>> clean = pd.DataFrame({"a": [1, 2, 3, 4], "b": list("wxyz")})
    >>> corrupted, report = Corruptor("missing", rate=0.25, seed=42).inject(clean)
    >>> report.n_injected
    2
    >>> corrupted.shape == clean.shape
    True
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "CORRUPTION_TYPES",
    "ColumnRole",
    "Corruptor",
    "InjectionReport",
    "classify_columns",
    "eligible_columns",
    "inject",
]

#: The corruption types this module can inject, in reporting order.
CORRUPTION_TYPES: Tuple[str, ...] = (
    "missing",
    "format",
    "out_of_range",
    "swap_valid",
    "swap_conditional",
    "typo",
)

#: Fraction of non-null values that must parse as a finite float before a
#: column is treated as numeric. 0.8 matches the profiler's MIXED threshold
#: (``src/data/profiler.py``) and the conditional detector's
#: ``numeric_parse_threshold``, so "numeric" means one thing project-wide.
NUMERIC_PARSE_THRESHOLD: float = 0.8

#: ID-like thresholds, copied from ConditionalDetector.
ID_UNIQUE_RATIO: float = 0.9
ID_MIN_UNIQUE: int = 50

#: A column with more distinct values than this is not treated as a
#: categorical vocabulary, so it is not eligible for ``typo``.
MAX_CATEGORIES: int = 200

#: ``out_of_range`` offsets, as multiples of the column's full range span.
#: The smallest is 5x the span, which for every column in every dataset here
#: is far beyond ``Q3 + 3*IQR``. Deliberately NOT tuned to the fence: see the
#: package docstring's anti-circularity note.
OUT_OF_RANGE_MULTIPLIERS: Tuple[float, ...] = (5.0, 10.0, 20.0, 50.0)

#: Fence used to *exclude* candidate values for ``swap_conditional``. 1.5 is
#: the tightest Tukey fence anyone uses and is tighter than the detector's
#: shipped k = 3, so a value inside it is invisible to the fence at either
#: setting.
CONDITIONAL_FENCE_K: float = 1.5

#: Digit -> lookalike letter substitutions for ``format``. These are the
#: OCR/typing confusions that produce real-world format errors, and they are
#: the same family as Breast Cancer's shipped ``1a0`` / ``a10``.
_DIGIT_LOOKALIKES: Dict[str, str] = {
    "0": "O", "1": "I", "2": "Z", "3": "E", "4": "A",
    "5": "S", "6": "G", "7": "T", "8": "B", "9": "q",
}

#: Keyboard neighbours for ``typo`` substitutions (QWERTY, lowercase).
_KEYBOARD_NEIGHBOURS: Dict[str, str] = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}

#: Characters the injector never writes, because the detector's
#: ``_NULL_STRINGS`` / missing-token rule would read them as MISSING and the
#: corruption type would stop being disjoint from ``missing``.
_FORBIDDEN_CHARS = frozenset("?")


class ColumnRole(str, Enum):
    """
    What the injector may do with a column.

    ``ID``
        Near-unique identifier. Never corrupted (see the module docstring).
    ``NUMERIC``
        At least :data:`NUMERIC_PARSE_THRESHOLD` of its non-null values parse
        as finite floats. Eligible for every type except ``typo``.
    ``TEXT``
        Not numeric, with a usable vocabulary (2..:data:`MAX_CATEGORIES`
        distinct values). Eligible for everything except ``format`` and
        ``out_of_range``, which need a number to deform.
    ``UNUSABLE``
        Constant, all-null, or a free-text column with too many distinct
        values to treat as a vocabulary. Never corrupted.
    """

    ID = "id"
    NUMERIC = "numeric"
    TEXT = "text"
    UNUSABLE = "unusable"


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class InjectionReport:
    """
    Exactly what was injected, where, and what could not be.

    Attributes
    ----------
    corruption_type : str
        One of :data:`CORRUPTION_TYPES`.
    requested_rate, achieved_rate : float
        Fractions of **all** cells. ``achieved_rate`` is
        ``n_injected / n_cells``, which is below ``requested_rate`` whenever
        the eligible pool was too small or candidate values ran out.
    seed : int
    n_rows, n_columns, n_cells : int
        Shape of the clean frame; ``n_cells = n_rows * n_columns``.
    n_eligible_cells : int
        Size of the pool positions were drawn from.
    n_requested, n_injected, n_attempted : int
        Target count, cells actually changed, and cells tried (which exceeds
        ``n_injected`` when some candidate values could not be constructed).
    positions : tuple[tuple[int, str], ...]
        ``(row_position, column)`` pairs, sorted by column order then row.
        Row positions are zero-based positional indices into the frame, the
        same convention as :class:`src.detection.detector.DetectionRecord`.
    per_column : dict[str, int]
        Injected count per column.
    eligible_columns : tuple[str, ...]
    skipped_columns : dict[str, str]
        column -> reason it was not eligible for this type.
    conditioning_columns : dict[str, str]
        ``swap_conditional`` only: target column -> the column its value was
        made improbable with respect to.
    notes : tuple[str, ...]
        Anything a results table must disclose, e.g. an unreachable rate.
    """

    corruption_type: str
    requested_rate: float
    achieved_rate: float
    seed: int
    n_rows: int
    n_columns: int
    n_cells: int
    n_eligible_cells: int
    n_requested: int
    n_injected: int
    n_attempted: int
    positions: Tuple[Tuple[int, str], ...]
    per_column: Dict[str, int] = field(default_factory=dict)
    eligible_columns: Tuple[str, ...] = ()
    skipped_columns: Dict[str, str] = field(default_factory=dict)
    conditioning_columns: Dict[str, str] = field(default_factory=dict)
    notes: Tuple[str, ...] = ()

    @property
    def rate_shortfall(self) -> float:
        """``requested_rate - achieved_rate``; 0.0 when the rate was met."""
        return float(self.requested_rate - self.achieved_rate)

    @property
    def is_empty(self) -> bool:
        """True when nothing could be injected at all."""
        return self.n_injected == 0

    def position_frame(self) -> pd.DataFrame:
        """The injected positions as a two-column frame (for CSV dumps)."""
        return pd.DataFrame(
            list(self.positions), columns=["row_id", "column"]
        ).astype({"row_id": int})

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe summary. ``positions`` is omitted — use the count."""
        return {
            "corruption_type": self.corruption_type,
            "requested_rate": float(self.requested_rate),
            "achieved_rate": float(self.achieved_rate),
            "rate_shortfall": float(self.rate_shortfall),
            "seed": int(self.seed),
            "n_rows": int(self.n_rows),
            "n_columns": int(self.n_columns),
            "n_cells": int(self.n_cells),
            "n_eligible_cells": int(self.n_eligible_cells),
            "n_requested": int(self.n_requested),
            "n_injected": int(self.n_injected),
            "n_attempted": int(self.n_attempted),
            "per_column": {str(k): int(v) for k, v in self.per_column.items()},
            "eligible_columns": list(self.eligible_columns),
            "skipped_columns": {str(k): str(v) for k, v in self.skipped_columns.items()},
            "conditioning_columns": {
                str(k): str(v) for k, v in self.conditioning_columns.items()
            },
            "notes": list(self.notes),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Column classification
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_float(value: Any) -> Optional[float]:
    """Return ``value`` as a finite float, or ``None`` if it is not one."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _numeric_parse_fraction(series: pd.Series) -> float:
    """Fraction of non-null values that parse as a finite float."""
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    ok = sum(1 for v in non_null.to_numpy() if _parse_float(v) is not None)
    return ok / len(non_null)


def _is_integer_like(series: pd.Series) -> bool:
    """True when every parseable non-null value is integral."""
    for v in series.dropna().to_numpy():
        f = _parse_float(v)
        if f is None:
            return False
        if abs(f - round(f)) > 1e-9:
            return False
    return True


def classify_columns(
    clean_df: pd.DataFrame,
    id_unique_ratio: float = ID_UNIQUE_RATIO,
    id_min_unique: int = ID_MIN_UNIQUE,
    max_categories: int = MAX_CATEGORIES,
    numeric_parse_threshold: float = NUMERIC_PARSE_THRESHOLD,
) -> Dict[str, ColumnRole]:
    """
    Assign a :class:`ColumnRole` to every column of ``clean_df``.

    The ID rule is :class:`ConditionalDetector`'s verbatim: more than
    ``id_min_unique`` distinct non-null values, a distinct/non-null ratio at
    or above ``id_unique_ratio``, and values that are integer-valued or
    non-numeric (a column of genuine fractions is a measurement, not a key).

    Returns
    -------
    dict[str, ColumnRole]
        One entry per column, in column order.
    """
    roles: Dict[str, ColumnRole] = {}
    for col in clean_df.columns:
        s = clean_df[col]
        non_null = s.dropna()
        n_non_null = len(non_null)
        n_unique = int(non_null.nunique())
        if n_non_null == 0 or n_unique < 2:
            roles[col] = ColumnRole.UNUSABLE
            continue

        parse_frac = _numeric_parse_fraction(s)
        numeric = parse_frac >= numeric_parse_threshold

        ratio = n_unique / n_non_null
        id_shape = n_unique > id_min_unique and ratio >= id_unique_ratio
        id_valued = (not numeric) or _is_integer_like(s)
        if id_shape and id_valued:
            roles[col] = ColumnRole.ID
            continue

        if numeric:
            roles[col] = ColumnRole.NUMERIC
        elif n_unique <= max_categories:
            roles[col] = ColumnRole.TEXT
        else:
            roles[col] = ColumnRole.UNUSABLE
        continue
    return roles


def eligible_columns(
    clean_df: pd.DataFrame,
    corruption_type: str,
    roles: Optional[Dict[str, ColumnRole]] = None,
    protected_columns: Sequence[str] = (),
) -> Tuple[Tuple[str, ...], Dict[str, str]]:
    """
    Which columns this corruption type can be injected into.

    Returns
    -------
    (eligible, skipped)
        ``eligible`` is a tuple of column names in frame order. ``skipped``
        maps every other column to a one-line reason, which goes straight
        into :attr:`InjectionReport.skipped_columns` so a zero in a results
        table is always explained.
    """
    _check_type(corruption_type)
    if roles is None:
        roles = classify_columns(clean_df)
    protected = set(protected_columns)

    eligible: List[str] = []
    skipped: Dict[str, str] = {}
    for col in clean_df.columns:
        role = roles.get(col, ColumnRole.UNUSABLE)
        if col in protected:
            skipped[col] = "protected by caller"
            continue
        if role is ColumnRole.ID:
            skipped[col] = "id-like (never corrupted)"
            continue
        if role is ColumnRole.UNUSABLE:
            skipped[col] = "constant, all-null, or free text (no vocabulary)"
            continue

        if corruption_type in ("format", "out_of_range"):
            if role is not ColumnRole.NUMERIC:
                skipped[col] = "not numeric: nothing to deform"
                continue
            if corruption_type == "out_of_range":
                num = pd.to_numeric(clean_df[col], errors="coerce").dropna()
                if num.empty or not np.isfinite(num.to_numpy()).any():
                    skipped[col] = "no finite numeric values"
                    continue
        elif corruption_type == "typo":
            if role is not ColumnRole.TEXT:
                skipped[col] = (
                    "numeric column: a character-level typo there is a format "
                    "error, which is a separate type"
                )
                continue
        eligible.append(col)

    return tuple(eligible), skipped


def _check_type(corruption_type: str) -> None:
    if corruption_type not in CORRUPTION_TYPES:
        raise ValueError(
            f"corruption_type must be one of {CORRUPTION_TYPES}, "
            f"got {corruption_type!r}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Seeding
# ═══════════════════════════════════════════════════════════════════════════════


def _cell_seed(seed: int, corruption_type: str, column: str, row: int) -> int:
    """
    Reproducible 64-bit seed for one cell.

    BLAKE2b of the tuple, not ``hash()``: Python salts string hashing per
    process, so ``hash()`` would give a different frame on every run.
    """
    payload = f"{int(seed)}|{corruption_type}|{column}|{int(row)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int(struct.unpack("<Q", digest)[0])


def _cell_rng(seed: int, corruption_type: str, column: str, row: int):
    return np.random.default_rng(_cell_seed(seed, corruption_type, column, row))


# ═══════════════════════════════════════════════════════════════════════════════
# Value generators
# ═══════════════════════════════════════════════════════════════════════════════


def _mangle_number(value: Any, rng) -> Optional[str]:
    """
    Turn a numeric value into a string that is **not** parseable as a number.

    Two operations, chosen per cell: substitute a digit with its lookalike
    letter (``10`` -> ``1O``), or insert a letter at a random position
    (``10`` -> ``1a0``). Both reproduce the shape of Breast Cancer's shipped
    format errors. The result is verified non-parseable before it is
    returned; ``None`` means no mangling worked and the cell must be skipped.
    """
    text = _numeric_text(value)
    if text is None or not text:
        return None
    letters = "abcdefghjkmnpqrstuvwxyz"  # no 'i'/'l'/'o' ambiguity in output
    for _ in range(12):
        op = rng.integers(0, 2)
        chars = list(text)
        if op == 0:
            digit_positions = [i for i, c in enumerate(chars) if c in _DIGIT_LOOKALIKES]
            if not digit_positions:
                continue
            i = int(rng.choice(digit_positions))
            chars[i] = _DIGIT_LOOKALIKES[chars[i]]
        else:
            i = int(rng.integers(0, len(chars) + 1))
            chars.insert(i, str(rng.choice(list(letters))))
        out = "".join(chars)
        if set(out) & _FORBIDDEN_CHARS:
            continue
        if out == text:
            continue
        if _parse_float(out) is None:
            return out
    return None


def _numeric_text(value: Any) -> Optional[str]:
    """Canonical string form of a numeric value: ``10`` not ``10.0``."""
    f = _parse_float(value)
    if f is None:
        return None
    if abs(f - round(f)) <= 1e-9:
        return str(int(round(f)))
    return repr(f)


def _typo(text: str, forbidden: frozenset, rng) -> Optional[str]:
    """
    Character-level perturbation of a categorical string.

    Four operations: transpose two adjacent characters, delete one, duplicate
    one, or substitute one with a QWERTY neighbour. The result is rejected if
    it equals the original or if it is itself a **valid value of the column**
    — a "typo" that lands on another legitimate category is a ``swap_valid``,
    not a typo, and letting it through would blur the two types.
    ``None`` means no perturbation worked.
    """
    if not text:
        return None
    for _ in range(16):
        chars = list(text)
        op = int(rng.integers(0, 4))
        if op == 0 and len(chars) >= 2:
            i = int(rng.integers(0, len(chars) - 1))
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif op == 1 and len(chars) >= 2:
            del chars[int(rng.integers(0, len(chars)))]
        elif op == 2:
            i = int(rng.integers(0, len(chars)))
            chars.insert(i, chars[i])
        else:
            i = int(rng.integers(0, len(chars)))
            lower = chars[i].lower()
            neighbours = _KEYBOARD_NEIGHBOURS.get(lower)
            if not neighbours:
                continue
            repl = str(rng.choice(list(neighbours)))
            chars[i] = repl.upper() if chars[i].isupper() else repl
        out = "".join(chars)
        if not out or out == text or out in forbidden:
            continue
        if set(out) & _FORBIDDEN_CHARS:
            continue
        return out
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Conditioning (for swap_conditional)
# ═══════════════════════════════════════════════════════════════════════════════


def _discretise(series: pd.Series, role: ColumnRole, n_bins: int = 4) -> np.ndarray:
    """
    Integer bin codes for one column, for the association search and the
    conditional grouping. Numeric columns are cut into quantile bins;
    everything else uses its own category codes. ``-1`` marks a null.
    """
    if role is ColumnRole.NUMERIC:
        num = pd.to_numeric(series, errors="coerce")
        try:
            codes = pd.qcut(num, n_bins, labels=False, duplicates="drop")
        except (ValueError, IndexError):
            codes = pd.Series(np.zeros(len(num)), index=num.index)
        arr = pd.Series(codes).to_numpy(dtype=float)
        arr = np.where(np.isnan(arr), -1.0, arr)
        return arr.astype(int)
    codes = pd.Categorical(series.astype("object").where(series.notna(), None)).codes
    return np.asarray(codes, dtype=int)


def _normalised_mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    """
    Normalised mutual information between two integer code arrays.

    Written out rather than imported so the association measure is visible
    and testable: ``2 * I(a;b) / (H(a) + H(b))``, 0.0 when either side is
    constant. Only rows where both codes are valid (``>= 0``) contribute.
    """
    mask = (a >= 0) & (b >= 0)
    if mask.sum() < 2:
        return 0.0
    x, y = a[mask], b[mask]
    n = x.size
    xs, xi = np.unique(x, return_inverse=True)
    ys, yi = np.unique(y, return_inverse=True)
    if xs.size < 2 or ys.size < 2:
        return 0.0
    joint = np.zeros((xs.size, ys.size), dtype=float)
    np.add.at(joint, (xi, yi), 1.0)
    joint /= n
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    def _h(p: np.ndarray) -> float:
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    hx, hy = _h(px), _h(py)
    if hx <= 0.0 or hy <= 0.0:
        return 0.0
    nz = joint > 0
    mi = float((joint[nz] * np.log(joint[nz] / np.outer(px, py)[nz])).sum())
    return max(0.0, 2.0 * mi / (hx + hy))


# ═══════════════════════════════════════════════════════════════════════════════
# The injector
# ═══════════════════════════════════════════════════════════════════════════════


class Corruptor:
    """
    Inject one corruption type at one rate into a clean frame.

    Parameters
    ----------
    corruption_type : str
        One of :data:`CORRUPTION_TYPES`.
    rate : float
        Fraction of **all** cells to corrupt, in ``[0, 1]``. ``0.0`` returns
        the frame unchanged (and the tests assert that).
    seed : int
        Base seed. See the module docstring for the two seeded streams.
    protected_columns : sequence of str
        Columns never to corrupt, on top of the automatic ID exclusion.
    id_unique_ratio, id_min_unique, max_categories, numeric_parse_threshold
        Column-classification thresholds; defaults match
        :class:`src.detectors.conditional.ConditionalDetector` and the
        profiler.
    conditional_fence_k : float
        Tukey multiplier used to *exclude* candidate values for
        ``swap_conditional``, so the injected value is inside even the
        tightest fence. Default 1.5.

    Raises
    ------
    ValueError
        Unknown ``corruption_type``, ``rate`` outside ``[0, 1]``, negative
        ``seed``.
    """

    def __init__(
        self,
        corruption_type: str,
        rate: float,
        seed: int = 42,
        protected_columns: Sequence[str] = (),
        id_unique_ratio: float = ID_UNIQUE_RATIO,
        id_min_unique: int = ID_MIN_UNIQUE,
        max_categories: int = MAX_CATEGORIES,
        numeric_parse_threshold: float = NUMERIC_PARSE_THRESHOLD,
        conditional_fence_k: float = CONDITIONAL_FENCE_K,
    ) -> None:
        _check_type(corruption_type)
        if not 0.0 <= float(rate) <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {rate}")
        if int(seed) < 0:
            raise ValueError(f"seed must be >= 0, got {seed}")
        if conditional_fence_k <= 0:
            raise ValueError(
                f"conditional_fence_k must be > 0, got {conditional_fence_k}"
            )
        self.corruption_type = corruption_type
        self.rate = float(rate)
        self.seed = int(seed)
        self.protected_columns = tuple(protected_columns)
        self.id_unique_ratio = float(id_unique_ratio)
        self.id_min_unique = int(id_min_unique)
        self.max_categories = int(max_categories)
        self.numeric_parse_threshold = float(numeric_parse_threshold)
        self.conditional_fence_k = float(conditional_fence_k)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def inject(
        self, clean_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, InjectionReport]:
        """
        Return ``(corrupted_df, report)``.

        ``corrupted_df`` is a new frame with ``clean_df``'s shape, index and
        column order. ``clean_df`` is never modified.

        Raises
        ------
        ValueError
            ``clean_df`` is empty, or has duplicate column names (positions
            would be ambiguous).
        """
        if clean_df.shape[0] == 0 or clean_df.shape[1] == 0:
            raise ValueError(f"clean_df is empty: shape {clean_df.shape}")
        if clean_df.columns.duplicated().any():
            raise ValueError(
                "clean_df has duplicate column names; injected positions "
                "would be ambiguous."
            )

        n_rows, n_cols = clean_df.shape
        n_cells = n_rows * n_cols
        roles = classify_columns(
            clean_df,
            id_unique_ratio=self.id_unique_ratio,
            id_min_unique=self.id_min_unique,
            max_categories=self.max_categories,
            numeric_parse_threshold=self.numeric_parse_threshold,
        )
        cols, skipped = eligible_columns(
            clean_df,
            self.corruption_type,
            roles=roles,
            protected_columns=self.protected_columns,
        )

        # Pool, in a fixed order: all of column 0's rows, then column 1's, ...
        # (the same column-major order detection_eval.flatten_cells uses).
        pool: List[Tuple[int, str]] = []
        for col in cols:
            s = clean_df[col]
            for i in range(n_rows):
                if self._cell_is_eligible(s.iat[i]):
                    pool.append((i, col))

        n_requested = int(round(self.rate * n_cells))
        notes: List[str] = []
        if n_requested > len(pool):
            notes.append(
                f"requested {n_requested} cells but only {len(pool)} were "
                f"eligible for '{self.corruption_type}'; the whole eligible "
                f"pool was corrupted and the achieved rate is below the "
                f"requested rate."
            )

        if n_requested == 0 or not pool:
            if not pool and n_requested > 0:
                notes.append(
                    f"no column is eligible for '{self.corruption_type}' in "
                    f"this frame; nothing was injected."
                )
            return clean_df.copy(), InjectionReport(
                corruption_type=self.corruption_type,
                requested_rate=self.rate,
                achieved_rate=0.0,
                seed=self.seed,
                n_rows=n_rows,
                n_columns=n_cols,
                n_cells=n_cells,
                n_eligible_cells=len(pool),
                n_requested=n_requested,
                n_injected=0,
                n_attempted=0,
                positions=(),
                eligible_columns=cols,
                skipped_columns=skipped,
                notes=tuple(notes),
            )

        # Position stream. One permutation of the whole pool, walked in order
        # until the target count is reached. Walking a permutation (rather
        # than drawing exactly n_requested up front) is what lets the
        # injector make up for cells whose value could not be constructed
        # while staying deterministic.
        pos_rng = np.random.default_rng(
            abs(_cell_seed(self.seed, self.corruption_type, "__positions__", 0))
            ^ int(round(self.rate * 1_000_000))
        )
        order = pos_rng.permutation(len(pool))

        prepared = self._prepare(clean_df, cols, roles)
        changes: Dict[str, Dict[int, Any]] = {c: {} for c in cols}
        n_injected = 0
        n_attempted = 0
        for idx in order:
            if n_injected >= n_requested:
                break
            row, col = pool[int(idx)]
            n_attempted += 1
            new_value = self._value_for(clean_df, col, row, prepared)
            if new_value is _SKIP:
                continue
            changes[col][row] = new_value
            n_injected += 1

        corrupted = self._apply(clean_df, changes)
        positions = tuple(
            (row, col) for col in cols for row in sorted(changes[col])
        )
        per_column = {col: len(changes[col]) for col in cols if changes[col]}
        if n_injected < n_requested and len(pool) >= n_requested:
            notes.append(
                f"{n_requested - n_injected} of {n_requested} target cells "
                f"could not be corrupted (no valid replacement value existed "
                f"for them); the achieved rate is below the requested rate."
            )

        report = InjectionReport(
            corruption_type=self.corruption_type,
            requested_rate=self.rate,
            achieved_rate=n_injected / n_cells,
            seed=self.seed,
            n_rows=n_rows,
            n_columns=n_cols,
            n_cells=n_cells,
            n_eligible_cells=len(pool),
            n_requested=n_requested,
            n_injected=n_injected,
            n_attempted=n_attempted,
            positions=positions,
            per_column=per_column,
            eligible_columns=cols,
            skipped_columns=skipped,
            conditioning_columns=dict(prepared.get("conditioning", {})),
            notes=tuple(notes),
        )
        logger.info(
            "Corruptor('%s', rate=%.4f, seed=%d): %d/%d cells injected "
            "(%d eligible) across %d columns.",
            self.corruption_type, self.rate, self.seed, n_injected,
            n_requested, len(pool), len(per_column),
        )
        return corrupted, report

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _cell_is_eligible(self, value: Any) -> bool:
        """
        A null cell is eligible for nothing: there is no value to corrupt,
        and blanking an already-blank cell would be a no-op that the
        round-trip check would then mis-attribute.
        """
        try:
            if pd.isna(value):
                return False
        except (TypeError, ValueError):
            pass
        if self.corruption_type in ("format", "out_of_range"):
            return _parse_float(value) is not None
        return True

    def _prepare(
        self,
        clean_df: pd.DataFrame,
        cols: Sequence[str],
        roles: Dict[str, ColumnRole],
    ) -> Dict[str, Any]:
        """
        Pre-compute everything that is a property of the *column* rather than
        of the cell: marginals, ranges, fences, conditioning columns.
        """
        prepared: Dict[str, Any] = {"roles": roles}
        ctype = self.corruption_type

        if ctype == "out_of_range":
            bounds = {}
            for col in cols:
                num = pd.to_numeric(clean_df[col], errors="coerce").dropna()
                lo, hi = float(num.min()), float(num.max())
                span = hi - lo
                if span <= 0:
                    span = max(abs(hi), 1.0)
                bounds[col] = (lo, hi, span, _is_integer_like(clean_df[col]))
            prepared["bounds"] = bounds

        if ctype in ("swap_valid", "swap_conditional", "typo"):
            marginals = {}
            for col in cols:
                vc = clean_df[col].dropna().value_counts()
                marginals[col] = vc
            prepared["marginals"] = marginals

        if ctype == "swap_conditional":
            codes = {
                col: _discretise(clean_df[col], roles.get(col, ColumnRole.UNUSABLE))
                for col in clean_df.columns
                if roles.get(col) in (ColumnRole.NUMERIC, ColumnRole.TEXT)
            }
            conditioning: Dict[str, str] = {}
            for target in cols:
                best, best_score = None, -1.0
                for other, other_codes in codes.items():
                    if other == target:
                        continue
                    score = _normalised_mutual_information(
                        codes[target], other_codes
                    )
                    # Deterministic tie-break on column name.
                    if score > best_score + 1e-12 or (
                        abs(score - best_score) <= 1e-12
                        and best is not None
                        and str(other) < str(best)
                    ):
                        best, best_score = other, score
                if best is not None:
                    conditioning[target] = best
            prepared["conditioning"] = conditioning
            prepared["codes"] = codes
            prepared["cond_cache"] = {}
            fences = {}
            for col in cols:
                if roles.get(col) is not ColumnRole.NUMERIC:
                    continue
                num = pd.to_numeric(clean_df[col], errors="coerce").dropna()
                q1, q3 = float(num.quantile(0.25)), float(num.quantile(0.75))
                iqr = q3 - q1
                fences[col] = (
                    q1 - self.conditional_fence_k * iqr,
                    q3 + self.conditional_fence_k * iqr,
                )
            prepared["fences"] = fences
        return prepared

    def _value_for(
        self,
        clean_df: pd.DataFrame,
        col: str,
        row: int,
        prepared: Dict[str, Any],
    ) -> Any:
        """Replacement value for one cell, or :data:`_SKIP`."""
        ctype = self.corruption_type
        original = clean_df[col].iat[row]
        rng = _cell_rng(self.seed, ctype, str(col), row)

        if ctype == "missing":
            return np.nan

        if ctype == "format":
            out = _mangle_number(original, rng)
            return _SKIP if out is None else out

        if ctype == "out_of_range":
            lo, hi, span, integral = prepared["bounds"][col]
            mult = float(rng.choice(np.asarray(OUT_OF_RANGE_MULTIPLIERS)))
            above = bool(rng.integers(0, 2))
            value = hi + mult * span if above else lo - mult * span
            if integral:
                value = int(round(value))
            if isinstance(original, str):
                return str(value)
            return value

        if ctype == "swap_valid":
            vc: pd.Series = prepared["marginals"][col]
            candidates = vc.drop(labels=[original], errors="ignore")
            if candidates.empty:
                return _SKIP
            weights = candidates.to_numpy(dtype=float)
            weights = weights / weights.sum()
            pick = int(rng.choice(len(candidates), p=weights))
            return candidates.index[pick]

        if ctype == "swap_conditional":
            return self._conditional_value(clean_df, col, row, prepared, original)

        if ctype == "typo":
            vc: pd.Series = prepared["marginals"][col]
            forbidden = frozenset(str(v) for v in vc.index)
            out = _typo(str(original), forbidden, rng)
            return _SKIP if out is None else out

        raise AssertionError(f"unhandled corruption type {ctype!r}")  # pragma: no cover

    def _conditional_value(
        self,
        clean_df: pd.DataFrame,
        col: str,
        row: int,
        prepared: Dict[str, Any],
        original: Any,
    ) -> Any:
        """
        A value that is valid globally but improbable given this row.

        The row's *conditioning bin* is the bin of the most strongly
        associated other column (normalised mutual information over
        discretised columns). Then:

        * **categorical target** — among the column's global values, prefer
          one that never co-occurs with this conditioning bin, and among
          those the globally *most common* one. Globally common plus
          conditionally impossible is the hardest case for a detector that
          only knows marginals, which is exactly the control the research
          program needs.
        * **numeric target** — among the column's global values that lie
          inside the 1.5x IQR fence (so no fence can flag them), take the one
          furthest from the median of the rows sharing this conditioning bin.
        """
        cond_col = prepared["conditioning"].get(col)
        if cond_col is None:
            return _SKIP
        codes = prepared["codes"]
        bin_id = int(codes[cond_col][row])
        cache_key = (col, cond_col, bin_id)
        cache = prepared["cond_cache"]
        role = prepared["roles"].get(col)

        if cache_key not in cache:
            mask = codes[cond_col] == bin_id
            group = clean_df[col][mask].dropna()
            if role is ColumnRole.NUMERIC:
                num = pd.to_numeric(group, errors="coerce").dropna()
                cache[cache_key] = (
                    float(num.median()) if not num.empty else None
                )
            else:
                cache[cache_key] = group.value_counts()
        ref = cache[cache_key]

        vc: pd.Series = prepared["marginals"][col]
        if role is ColumnRole.NUMERIC:
            if ref is None:
                return _SKIP
            lo, hi = prepared["fences"][col]
            best, best_dist = _SKIP, -1.0
            for value in vc.index:
                f = _parse_float(value)
                if f is None or not (lo <= f <= hi):
                    continue
                if _values_equal(value, original):
                    continue
                dist = abs(f - ref)
                if dist > best_dist + 1e-12:
                    best, best_dist = value, dist
            return best

        counts: pd.Series = ref
        zero, fallback, fallback_count = None, None, None
        for value in vc.index:  # vc is sorted by descending global frequency
            if _values_equal(value, original):
                continue
            c = int(counts.get(value, 0))
            if c == 0:
                zero = value
                break
            if fallback_count is None or c < fallback_count:
                fallback, fallback_count = value, c
        if zero is not None:
            return zero
        return _SKIP if fallback is None else fallback

    def _apply(
        self, clean_df: pd.DataFrame, changes: Dict[str, Dict[int, Any]]
    ) -> pd.DataFrame:
        """
        Build the corrupted frame, restoring the clean dtype where the written
        values allow it (see the module docstring's DTYPES section).
        """
        out = clean_df.copy()
        for col, cells in changes.items():
            if not cells:
                continue
            series = out[col].astype(object)
            for row, value in cells.items():
                series.iat[row] = value
            new = series.infer_objects()
            original_dtype = clean_df[col].dtype
            if new.dtype != original_dtype and not new.isna().any():
                try:
                    cast = new.astype(original_dtype)
                except (TypeError, ValueError, OverflowError):
                    cast = None
                if cast is not None and bool((cast == new).all()):
                    new = cast
            out[col] = new
        return out


class _Skip:
    """Sentinel: this cell could not be corrupted."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "<SKIP>"


_SKIP = _Skip()


def _values_equal(a: Any, b: Any) -> bool:
    """Loose equality used only to avoid writing a cell's own value back."""
    fa, fb = _parse_float(a), _parse_float(b)
    if fa is not None and fb is not None:
        return abs(fa - fb) <= 1e-9
    return str(a) == str(b)


def inject(
    clean_df: pd.DataFrame,
    corruption_type: str,
    rate: float,
    seed: int = 42,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, InjectionReport]:
    """
    One-call wrapper around :class:`Corruptor`.

        >>> import pandas as pd
        >>> clean = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        >>> corrupted, report = inject(clean, "out_of_range", 0.4, seed=7)
        >>> report.n_injected
        2
    """
    return Corruptor(corruption_type, rate, seed=seed, **kwargs).inject(clean_df)
