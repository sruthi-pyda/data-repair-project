"""
src/detectors/text_fallback.py
------------------------------
A detector for the columns `ConditionalDetector` **cannot model at all**.

THE COVERAGE PROBLEM THIS EXISTS TO FIX
=======================================
`results/w7_coverage.md` measured how many columns our conditional detector
can form any opinion about:

    flights        0 of  7
    movies         2 of 17
    beers          3 of 11
    rayyan         3 of 11
    restaurants    6 of 16
    hospital       8 of 20

It skips `high_cardinality_non_numeric` columns because it models a column as
a **category**, and a column holding hundreds of distinct strings has no
usable class structure. Real dirty-data benchmarks are mostly text, so this is
a scope limit on the entire Wave 1 contribution — and the first thing a
reviewer hits by running our code on `flights`.

THE IDEA: MODEL THE STRING, NOT THE CATEGORY
============================================
A high-cardinality text column still has structure, just not categorical
structure. Three signals, each cheap, each catching a different corruption
type, each computed from the **dirty column alone**:

1. ``pattern`` — **shape rarity.** Map a value to its signature
   (``"ABC-123"`` -> ``"AAA-999"``) and score by how rare that shape is in the
   column. Catches format corruption and parse breaks. This is the signal
   `dBoost` and `FAHES` exploit.

2. ``ngram`` — **character n-gram implausibility.** Fit a character trigram
   model on the column's own values and score a value by its mean negative
   per-character log-likelihood. Catches typos and garbage strings.

3. ``neighbour`` — **closeness to a frequent value**, and this one inverts the
   usual logic deliberately:

       a TYPO is CLOSE to a valid value without equalling it
       a LEGITIMATELY RARE value is usually FAR from everything

   So for an infrequent value, *high* similarity to some frequent value is
   **suspicious**, and low similarity is not. Every other mechanism in this
   project ranked rare things first and failed; this signal is the one that
   can rank a rare-but-well-separated value *low*, which is exactly the
   rare-versus-wrong distinction we have never managed to make.

Scores are combined by per-column **rank averaging**, so no signal's raw
scale can dominate another's — the cross-column comparability failure that
has bitten this project twice.

LEAKAGE
=======
Fitted on the dirty column only. No clean data, no `DatasetProfile`, no
labels. ``fit`` accepts a ``profile`` argument for interface parity with the
project's other detectors and **ignores it**, so nothing clean-derived can
enter that way.
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SIGNALS = ("pattern", "ngram", "neighbour")

#: A value seen in at least this fraction of rows counts as "frequent" and so
#: is a plausible typo source for the ``neighbour`` signal.
FREQUENT_THRESHOLD = 0.01

#: Frequent values compared against, most common first. Bounds the cost of
#: the ``neighbour`` signal at O(n_rare * TOP_K).
TOP_K_FREQUENT = 40

#: Character n-gram order. 3 is the usual choice for short field values.
NGRAM_N = 3

#: The ``neighbour`` signal needs at least this many frequent values to act
#: as plausible typo sources. Below it the signal is uninformative, so the
#: column is reported **unsupported** rather than scored from the remaining
#: signals -- honest coverage beats inflated coverage
#: (`docs/w9_plan.md` §W9.2b, abstention).
MIN_FREQUENT_FOR_NEIGHBOUR = 2

#: A score this high means "missing", which the rule detector's
#: missing-token check already owns; kept consistent with
#: ``ConditionalDetector(nan_score=1.0)``.
NAN_SCORE = 1.0


def signature(value: object) -> str:
    """
    Shape of a value: letters -> ``A``, digits -> ``9``, everything else kept.

    Runs of the same class collapse, so ``"Boston"`` and ``"Cambridge"`` share
    the signature ``"A"`` while ``"1a0"`` becomes ``"9A9"``. Collapsing is what
    makes the signature a *shape* rather than a length-sensitive fingerprint.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "<NA>"
    out: List[str] = []
    for ch in str(value):
        cls = "A" if ch.isalpha() else ("9" if ch.isdigit() else ch)
        if not out or out[-1] != cls:
            out.append(cls)
    return "".join(out)


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks scaled to [0, 1]; constant input maps to all zeros."""
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    out = np.zeros(v.size, dtype=float)
    if finite.sum() == 0 or np.unique(v[finite]).size < 2:
        return out
    x = v[finite]
    order = np.argsort(x, kind="mergesort")
    r = np.empty(x.size, dtype=float)
    r[order] = np.arange(1, x.size + 1, dtype=float)
    for val in np.unique(x):
        m = x == val
        if m.sum() > 1:
            r[m] = r[m].mean()
        out[finite] = (r - 1.0) / max(x.size - 1, 1)
    return out


@dataclasses.dataclass
class ColumnReport:
    column: str
    n_values: int
    n_distinct: int
    n_frequent: int
    scored: bool
    reason: str = ""


class TextFallbackDetector:
    """
    Per-column string-structure detector for high-cardinality text.

    Parameters
    ----------
    frequent_threshold : float
        Minimum frequency for a value to be a plausible typo source.
    top_k : int
        How many frequent values the ``neighbour`` signal compares against.
    ngram_n : int
        Character n-gram order.
    signals : sequence of str
        Which signals to combine. Exposed so the experiment can report each
        one alone, as `docs/w9_plan.md` §W9.2 condition 4 requires.
    """

    def __init__(
        self,
        frequent_threshold: float = FREQUENT_THRESHOLD,
        top_k: int = TOP_K_FREQUENT,
        ngram_n: int = NGRAM_N,
        signals: Sequence[str] = SIGNALS,
        min_frequent: int = MIN_FREQUENT_FOR_NEIGHBOUR,
    ) -> None:
        bad = [s for s in signals if s not in SIGNALS]
        if bad:
            raise ValueError(f"unknown signals {bad}; valid are {SIGNALS}")
        if not 0.0 < frequent_threshold < 1.0:
            raise ValueError(
                f"frequent_threshold must be in (0,1), got {frequent_threshold}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if ngram_n < 2:
            raise ValueError(f"ngram_n must be >= 2, got {ngram_n}")
        self.frequent_threshold = float(frequent_threshold)
        self.top_k = int(top_k)
        self.ngram_n = int(ngram_n)
        if min_frequent < 0:
            raise ValueError(f"min_frequent must be >= 0, got {min_frequent}")
        self.signals = tuple(signals)
        self.min_frequent = int(min_frequent)
        self.reports_: Dict[str, ColumnReport] = {}

    # ------------------------------------------------------------------ #
    # Signals
    # ------------------------------------------------------------------ #

    def _pattern_scores(self, values: Sequence[object]) -> np.ndarray:
        sigs = [signature(v) for v in values]
        counts = Counter(sigs)
        n = max(len(sigs), 1)
        # Rarer shape => higher score.
        return np.array([1.0 - counts[s] / n for s in sigs], dtype=float)

    def _ngram_scores(self, values: Sequence[object]) -> np.ndarray:
        n = self.ngram_n
        texts = ["" if (v is None or (isinstance(v, float) and pd.isna(v)))
                 else str(v) for v in values]
        counts: Counter = Counter()
        context: Counter = Counter()
        for t in texts:
            pad = "^" * (n - 1) + t + "$"
            for i in range(len(pad) - n + 1):
                gram = pad[i:i + n]
                counts[gram] += 1
                context[gram[:-1]] += 1
        vocab = max(len(set("".join(texts))), 1)

        out = np.zeros(len(texts), dtype=float)
        for k, t in enumerate(texts):
            pad = "^" * (n - 1) + t + "$"
            total, logp = 0, 0.0
            for i in range(len(pad) - n + 1):
                gram = pad[i:i + n]
                # add-one smoothing, so an unseen gram is improbable but
                # never impossible
                num = counts[gram] + 1.0
                den = context[gram[:-1]] + vocab
                logp += math.log(num / den)
                total += 1
            out[k] = -logp / total if total else 0.0
        return out

    def _neighbour_scores(self, values: Sequence[object]) -> np.ndarray:
        """
        Similarity to the nearest FREQUENT value, for infrequent values only.

        Deliberately inverted relative to every other signal in this project:
        an infrequent value that is *very similar* to a frequent one looks
        like a typo of it, while an infrequent value far from everything looks
        like a legitimately rare entry. Frequent values themselves score 0.
        """
        s = pd.Series(list(values), dtype="object")
        freq = s.value_counts(normalize=True, dropna=True)
        frequent = [v for v, f in freq.items()
                    if f >= self.frequent_threshold][: self.top_k]
        frequent_set = set(frequent)
        freq_str = [str(v) for v in frequent]

        cache: Dict[str, float] = {}
        out = np.zeros(len(s), dtype=float)
        if not freq_str:
            return out

        for i, v in enumerate(s):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            if v in frequent_set:
                continue  # a common value is not a typo of anything
            key = str(v)
            if key in cache:
                out[i] = cache[key]
                continue
            best = 0.0
            for cand in freq_str:
                if cand == key:
                    continue
                r = SequenceMatcher(None, key, cand).ratio()
                if r > best:
                    best = r
                    if best > 0.99:
                        break
            cache[key] = best
            out[i] = best
        return out

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def score_column(
        self, series: pd.Series, per_signal: bool = False
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Score one column. Returns ``(combined, {signal: scores})``.

        Signals are combined by **rank averaging** so no raw scale dominates.
        Missing values get :data:`NAN_SCORE`.
        """
        values = list(series.astype("object").to_numpy())
        raw: Dict[str, np.ndarray] = {}
        if "pattern" in self.signals:
            raw["pattern"] = self._pattern_scores(values)
        if "ngram" in self.signals:
            raw["ngram"] = self._ngram_scores(values)
        if "neighbour" in self.signals:
            raw["neighbour"] = self._neighbour_scores(values)

        if not raw:
            return np.zeros(len(values), dtype=float), {}
        stacked = np.vstack([_ranks(raw[s]) for s in self.signals if s in raw])
        combined = stacked.mean(axis=0)

        is_na = np.array(
            [v is None or (isinstance(v, float) and pd.isna(v)) for v in values],
            dtype=bool)
        combined = np.where(is_na, NAN_SCORE, combined)
        return combined, (raw if per_signal else {})

    def score_cells(
        self,
        dirty_df: pd.DataFrame,
        columns: Optional[Sequence[str]] = None,
        profile: object = None,  # noqa: ARG002 - parity; deliberately ignored
    ) -> pd.DataFrame:
        """
        Score ``columns`` (default: all) of the DIRTY frame.

        Columns not scored come back all-zero, matching how
        ``ConditionalDetector`` reports a column it skips, so the two can be
        combined without a special case.
        """
        if not isinstance(dirty_df, pd.DataFrame):
            raise TypeError(f"dirty_df must be a DataFrame, got {type(dirty_df)!r}")
        if dirty_df.shape[0] == 0 or dirty_df.shape[1] == 0:
            raise ValueError(f"dirty_df is empty: shape {dirty_df.shape}")

        targets = list(dirty_df.columns if columns is None else columns)
        unknown = [c for c in targets if c not in dirty_df.columns]
        if unknown:
            raise ValueError(f"columns not in frame: {unknown}")

        out = pd.DataFrame(
            0.0, index=dirty_df.index, columns=dirty_df.columns, dtype=float)
        self.reports_ = {}
        for col in dirty_df.columns:
            s = dirty_df[col]
            n_distinct = int(s.nunique(dropna=True))
            vc = s.value_counts(normalize=True, dropna=True)
            n_freq = int((vc >= self.frequent_threshold).sum())
            if col not in targets:
                self.reports_[col] = ColumnReport(
                    col, len(s), n_distinct, n_freq, False, "not requested")
                continue
            if n_distinct < 2:
                self.reports_[col] = ColumnReport(
                    col, len(s), n_distinct, n_freq, False, "constant")
                continue
            if "neighbour" in self.signals and n_freq < self.min_frequent:
                # Abstain rather than guess: with too few frequent values
                # there are no plausible typo sources, so the signal that
                # distinguishes rare-but-valid from corrupted cannot fire.
                self.reports_[col] = ColumnReport(
                    col, len(s), n_distinct, n_freq, False,
                    f"unsupported: only {n_freq} frequent value(s), "
                    f"neighbour signal uninformative")
                continue
            scores, _ = self.score_column(s)
            out[col] = scores
            self.reports_[col] = ColumnReport(
                col, len(s), n_distinct, n_freq, True)
        return out

    def fit(self, dirty_df: pd.DataFrame, profile: object = None
            ) -> "TextFallbackDetector":
        """
        Present for interface parity. This detector is stateless per column —
        every statistic is computed from the column being scored — so there is
        nothing to fit and ``profile`` is ignored.
        """
        if not isinstance(dirty_df, pd.DataFrame):
            raise TypeError(f"dirty_df must be a DataFrame, got {type(dirty_df)!r}")
        return self
