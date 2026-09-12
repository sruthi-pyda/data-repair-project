"""
src/repair/mock_repair.py
--------------------------
Deterministic mock LLM repair-candidate generator.

PURPOSE
=======
This module simulates what a real LLM would do when asked to repair a dirty cell.
It exists so the complete pipeline (detection → repair → uncertainty → calibration →
decision) can run fully offline without an API key.

IMPORTANT — DATA LEAKAGE PREVENTION
=====================================
This module MUST NOT access clean.csv or any per-cell clean values.

Inputs allowed:
  - DetectionRecord     — from Phase 3 detector (dirty data only)
  - dirty_df            — the dirty DataFrame (for vocabulary extraction and context)
  - DatasetProfile      — aggregate schema facts (ranges, vocabularies from clean data
                          are acceptable as population-level schema knowledge)

Not allowed:
  - clean_df passed to any repair function
  - comparing dirty_value against clean ground-truth during generation

Clean values are used ONLY by RepairEvaluator (in src/repair/evaluation.py)
to measure accuracy AFTER generation.

HOW THE MOCK WORKS
==================
The mock is not an LLM. It uses deterministic heuristics + seeded randomness:

  CATEGORICAL / NOVEL_CATEGORY:
    Compute string-similarity scores (difflib SequenceMatcher) between the dirty
    value and every known vocabulary item. Weight by frequency. Add seeded noise.
    Normalise to produce a probability distribution.

  CATEGORICAL / MISSING:
    Sample the known vocabulary proportional to clean-data frequency.
    Higher-frequency categories get higher probability.

  NUMERICAL / OUT_OF_RANGE:
    Apply digit-normalisation heuristics (e.g. 33→3, 88→8 for digit-repeat
    corruption). Fall back to column median and adjacent quartiles.

  NUMERICAL / MISSING:
    Use column median (highest prob), Q25, Q75 as three candidates.

  NUMERICAL / FORMAT_ERROR:
    Strip non-numeric characters and try to recover a parseable number.
    Fall back to median.

  MIXED / FORMAT_ERROR or NOVEL_CATEGORY:
    Combine string-similarity (on the numeric-like part) with known vocab.

REALISM
========
The mock is intentionally imperfect:
  - String similarity can select the wrong candidate.
  - Frequency-based repair picks the most common value, not the correct one.
  - Seeded noise perturbs probabilities so the system is not trivially confident.
  - Digit-normalisation can fail for non-digit-repeat corruptions.

This imperfection is required so that Phase 5 calibration has real errors to work with.

DETERMINISM
===========
Every cell gets a seed derived from (dataset, column, row_id) so repeated runs
produce identical results.

Usage
======
    from src.data.loader import DataLoader
    from src.data.profiler import profile_dataset
    from src.detection.detector import Detector
    from src.repair.mock_repair import MockRepairer, build_repair_context

    loader = DataLoader("datasets")
    clean_df, dirty_df = loader.load("breast_cancer")
    profile = profile_dataset("breast_cancer", loader.discover("breast_cancer"), clean_df, dirty_df)

    detector = Detector(profile)
    records = detector.detect(dirty_df)          # no clean_df here

    repairer = MockRepairer(seed=42)
    for rec in records[:5]:
        context = build_repair_context(rec, dirty_df, profile)
        result = repairer.repair(context)
        print(result)
"""

from __future__ import annotations

import difflib
import hashlib
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.schema import ColumnKind, DatasetProfile
from src.detection.detector import DetectionRecord, ErrorType

logger = logging.getLogger(__name__)

# Maximum candidates to generate per cell
_DEFAULT_TOP_K: int = 3

# Noise scale added to similarity/frequency scores before normalisation.
# Keeps the mock realistic (not always 100% confident in the right answer).
_NOISE_SCALE: float = 0.08


# ═══════════════════════════════════════════════════════════════════════════════
# Output types
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RepairCandidate:
    """A single candidate repair value with its probability."""

    value: Any
    probability: float

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "probability": round(self.probability, 4)}


@dataclass
class RepairContext:
    """
    Compact context object describing what the repairer knows about a cell.

    Built from dirty_df and DatasetProfile. Contains NO clean values.
    Passed to prompts.py for LLM prompt construction and to MockRepairer
    for candidate generation.
    """

    dataset: str
    column: str
    kind: ColumnKind
    dtype: str                           # clean dtype string ('int64', 'object', …)
    dirty_value: Any                     # value in dirty DataFrame
    error_type: str                      # from DetectionRecord
    row_id: int
    row_context: Dict[str, Any]          # other column values in the same row
    vocabulary: List[str]                # known valid values (from dirty data)
    column_stats: Optional[Dict[str, float]] = None  # numerical stats from profile


@dataclass
class RepairResult:
    """
    Full output of the mock repairer for one suspicious cell.

    top_candidate is always candidates[0] (highest probability).
    """

    dataset: str
    row_id: int
    column: str
    dirty_value: Any
    error_type: str
    candidates: List[RepairCandidate]
    repair_method: str
    context_used: Dict[str, Any] = field(default_factory=dict)

    @property
    def top_candidate(self) -> RepairCandidate:
        return self.candidates[0] if self.candidates else RepairCandidate(None, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "row_id": self.row_id,
            "column": self.column,
            "original_value": self.dirty_value,
            "candidates": [c.to_dict() for c in self.candidates],
            "repair_method": self.repair_method,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Context builder  (no clean_df)
# ═══════════════════════════════════════════════════════════════════════════════


def build_repair_context(
    record: DetectionRecord,
    dirty_df: pd.DataFrame,
    profile: DatasetProfile,
    n_context_cols: int = 8,
) -> RepairContext:
    """
    Build a RepairContext from a DetectionRecord and dirty data.

    Parameters
    ----------
    record : DetectionRecord
        The suspicious cell to repair.
    dirty_df : pd.DataFrame
        The dirty DataFrame. Used to extract vocabulary and row context.
        Must NOT be the clean DataFrame.
    profile : DatasetProfile
        Schema information (column kinds, numerical stats).
    n_context_cols : int
        Maximum number of other columns to include in row_context.

    Returns
    -------
    RepairContext
    """
    col = record.column
    col_profile = profile.columns[col]
    row = dirty_df.iloc[record.row_id]

    # Row context: neighbouring columns (skip the dirty cell itself and ID columns)
    context_cols = [
        c for c in profile.repairable_columns
        if c != col and c in dirty_df.columns
    ][:n_context_cols]
    row_context = {c: _safe_val(row[c]) for c in context_cols}

    # Vocabulary from dirty data (not from clean profile)
    vocabulary = _extract_vocabulary_from_dirty(dirty_df[col], col_profile)

    # Numerical stats come from the profile (aggregate clean stats → acceptable)
    col_stats: Optional[Dict[str, float]] = None
    if col_profile.numerical_stats is not None:
        s = col_profile.numerical_stats
        col_stats = {
            "min": s.min, "max": s.max,
            "median": s.q50, "q25": s.q25, "q75": s.q75, "iqr": s.iqr,
        }

    return RepairContext(
        dataset=record.dataset,
        column=col,
        kind=col_profile.kind,
        dtype=col_profile.dtype_clean,
        dirty_value=record.dirty_value,
        error_type=record.error_type,
        row_id=record.row_id,
        row_context=row_context,
        vocabulary=vocabulary,
        column_stats=col_stats,
    )


def _extract_vocabulary_from_dirty(
    series: pd.Series,
    col_profile,
    max_vocab: int = 50,
) -> List[str]:
    """
    Extract a ranked vocabulary list from the dirty column.

    For CATEGORICAL/BINARY/MIXED: all non-null distinct values ranked by frequency.
    For NUMERICAL: empty list (vocabulary not meaningful for continuous values).
    Only returns values that are not themselves malformed (parseable for the
    expected type where appropriate).
    """
    if col_profile.kind == ColumnKind.NUMERICAL:
        return []

    non_null = series.dropna().astype(str)

    # For NUMERICAL-like kinds, exclude values that can't be parsed as numbers
    if col_profile.kind == ColumnKind.MIXED:
        parseable = non_null[pd.to_numeric(non_null, errors="coerce").notna()]
        # Also keep known non-numeric vocabulary tokens like '?'
        known_non_numeric = non_null[~non_null.isin(parseable)]
        # Only keep non-numeric tokens that are frequent (likely legitimate)
        freq = non_null.value_counts()
        threshold = max(1, int(0.005 * len(series)))
        legit_non_numeric = [v for v in known_non_numeric.unique()
                             if freq.get(v, 0) >= threshold]
        vocab_series = pd.concat([parseable, pd.Series(legit_non_numeric)])
    else:
        vocab_series = non_null

    counts = vocab_series.value_counts()
    return [str(v) for v in counts.index[:max_vocab]]


def _safe_val(val: Any) -> Any:
    """Convert numpy scalars to Python primitives; leave NaN as None."""
    if pd.isna(val):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


# ═══════════════════════════════════════════════════════════════════════════════
# MockRepairer
# ═══════════════════════════════════════════════════════════════════════════════


class MockRepairer:
    """
    Deterministic mock LLM repair-candidate generator.

    Parameters
    ----------
    seed : int
        Base random seed. Each cell gets a derived seed for determinism.
    top_k : int
        Maximum number of candidates to return.
    """

    def __init__(self, seed: int = 42, top_k: int = _DEFAULT_TOP_K) -> None:
        self.seed = seed
        self.top_k = top_k

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def repair(self, context: RepairContext) -> RepairResult:
        """
        Generate repair candidates for one suspicious cell.

        Parameters
        ----------
        context : RepairContext
            Built by build_repair_context() — contains NO clean values.

        Returns
        -------
        RepairResult
        """
        rng = self._make_rng(context)

        error_type = context.error_type
        kind = context.kind

        # ── Strategy dispatch ────────────────────────────────────────────
        if error_type == ErrorType.MISSING.value:
            if kind in (ColumnKind.CATEGORICAL, ColumnKind.BINARY, ColumnKind.MIXED):
                raw, method = self._frequency_repair(context, rng)
            else:
                raw, method = self._statistical_repair(context, rng)

        elif error_type == ErrorType.NOVEL_CATEGORY.value:
            raw, method = self._similarity_repair(context, rng)

        elif error_type == ErrorType.OUT_OF_RANGE.value:
            raw, method = self._range_normalisation_repair(context, rng)

        elif error_type == ErrorType.FORMAT_ERROR.value:
            if kind in (ColumnKind.CATEGORICAL, ColumnKind.MIXED):
                raw, method = self._format_categorical_repair(context, rng)
            else:
                raw, method = self._format_numerical_repair(context, rng)

        elif error_type == ErrorType.STATISTICAL_OUTLIER.value:
            raw, method = self._statistical_repair(context, rng)

        elif error_type == ErrorType.WHITESPACE.value:
            raw, method = self._whitespace_repair(context, rng)

        else:
            # Unknown error type: fall back to frequency/statistical
            if kind in (ColumnKind.CATEGORICAL, ColumnKind.BINARY, ColumnKind.MIXED):
                raw, method = self._frequency_repair(context, rng)
            else:
                raw, method = self._statistical_repair(context, rng)

        # ── Validate and normalise ───────────────────────────────────────
        candidates = _validate_and_normalise(raw, self.top_k)

        if not candidates:
            # Ultimate fallback: single candidate = dirty value, low confidence
            candidates = [RepairCandidate(value=context.dirty_value, probability=0.2)]
            method = "fallback_identity"

        logger.debug(
            "Repaired [%s/%s row=%d] via %s: top=%s (p=%.3f)",
            context.column, context.error_type, context.row_id,
            method, candidates[0].value, candidates[0].probability,
        )

        return RepairResult(
            dataset=context.dataset,
            row_id=context.row_id,
            column=context.column,
            dirty_value=context.dirty_value,
            error_type=context.error_type,
            candidates=candidates,
            repair_method=method,
            context_used={
                "vocabulary_size": len(context.vocabulary),
                "has_stats": context.column_stats is not None,
                "kind": context.kind.value,
            },
        )

    def repair_batch(
        self,
        records: List[DetectionRecord],
        dirty_df: pd.DataFrame,
        profile: DatasetProfile,
    ) -> List[RepairResult]:
        """
        Repair all detection records in a batch.

        Parameters
        ----------
        records : list[DetectionRecord]
        dirty_df : pd.DataFrame
            Dirty data (no clean values).
        profile : DatasetProfile

        Returns
        -------
        list[RepairResult]
        """
        results = []
        for rec in records:
            ctx = build_repair_context(rec, dirty_df, profile)
            results.append(self.repair(ctx))
        logger.info(
            "Batch repair complete for '%s': %d cells repaired.",
            profile.name, len(results),
        )
        return results

    # ------------------------------------------------------------------ #
    # Repair strategies                                                    #
    # ------------------------------------------------------------------ #

    def _frequency_repair(
        self,
        context: RepairContext,
        rng: np.random.Generator,
    ) -> Tuple[List[Tuple[Any, float]], str]:
        """
        MISSING categorical/binary: sample vocabulary proportional to frequency.

        The most common category in the dirty data gets the highest probability.
        This is deliberately imperfect — the most common value may not be correct
        for this specific row.
        """
        vocab = context.vocabulary
        if not vocab:
            return [], "frequency_repair_empty_vocab"

        # Vocab is already sorted by frequency (build_repair_context does this)
        n = min(self.top_k, len(vocab))
        candidates_vals = vocab[:n]

        # Assign exponentially decaying initial scores
        raw_scores = np.array([math.exp(-0.8 * i) for i in range(n)], dtype=float)

        # Add seeded noise
        noise = rng.uniform(-_NOISE_SCALE, _NOISE_SCALE, size=n)
        raw_scores = np.clip(raw_scores + noise, 0.01, None)

        return list(zip(candidates_vals, raw_scores.tolist())), "frequency_repair"

    def _similarity_repair(
        self,
        context: RepairContext,
        rng: np.random.Generator,
    ) -> Tuple[List[Tuple[Any, float]], str]:
        """
        NOVEL_CATEGORY: rank vocabulary by string similarity to the dirty value.

        Uses difflib.SequenceMatcher for edit-distance-like similarity.
        High similarity → high probability. Seeded noise makes it occasionally
        pick the wrong candidate.
        """
        dirty_str = str(context.dirty_value).strip()
        vocab = context.vocabulary
        if not vocab:
            return [(dirty_str, 0.3)], "similarity_repair_empty_vocab"

        scores: List[Tuple[str, float]] = []
        for v in vocab:
            sim = difflib.SequenceMatcher(None, dirty_str.lower(), v.lower()).ratio()
            scores.append((v, sim))

        # Sort by similarity descending
        scores.sort(key=lambda x: x[1], reverse=True)
        top_n = scores[: self.top_k]

        # Apply seeded noise
        noise = rng.uniform(-_NOISE_SCALE, _NOISE_SCALE, size=len(top_n))
        noisy = [(v, max(0.01, s + noise[i])) for i, (v, s) in enumerate(top_n)]

        return noisy, "similarity_repair"

    def _statistical_repair(
        self,
        context: RepairContext,
        rng: np.random.Generator,
    ) -> Tuple[List[Tuple[Any, float]], str]:
        """
        MISSING / STATISTICAL_OUTLIER numerical: use column statistics.

        Candidates (in decreasing probability):
          1. Median        (best single-value guess)
          2. Q25           (lower quartile)
          3. Q75           (upper quartile)

        All values are cast to the clean dtype where possible.
        """
        stats = context.column_stats
        if stats is None:
            return [], "statistical_repair_no_stats"

        median = stats["median"]
        q25 = stats["q25"]
        q75 = stats["q75"]

        # Cast to integer when clean dtype is integer
        if "int" in context.dtype:
            median = int(round(median))
            q25 = int(round(q25))
            q75 = int(round(q75))

        # De-duplicate
        raw_vals: List[Any] = []
        for v in [median, q25, q75]:
            if v not in raw_vals:
                raw_vals.append(v)

        raw_scores = np.array(
            [math.exp(-0.6 * i) for i in range(len(raw_vals))], dtype=float
        )
        noise = rng.uniform(-_NOISE_SCALE, _NOISE_SCALE, size=len(raw_vals))
        raw_scores = np.clip(raw_scores + noise, 0.01, None)

        return list(zip(raw_vals, raw_scores.tolist())), "statistical_repair"

    def _range_normalisation_repair(
        self,
        context: RepairContext,
        rng: np.random.Generator,
    ) -> Tuple[List[Tuple[Any, float]], str]:
        """
        OUT_OF_RANGE numerical: attempt to normalise the value back into range.

        Heuristics applied in order (highest to lowest confidence):
          1. Digit-repeat squashing: 33→3, 88→8, 11→1 (divide by 11 then 10)
          2. Column median
          3. Q25 / Q75

        Digit-repeat squashing detects the Breast Cancer pattern where values like
        '33', '88' are corrupted forms of '3', '8'. The heuristic checks:
          - dirty / 11 ∈ [min, max] and is an integer → strong candidate
          - dirty / 10 ∈ [min, max] (approximate) → weaker candidate

        These are heuristics, not guaranteed to be correct.
        """
        stats = context.column_stats
        dirty_val = context.dirty_value

        try:
            num = float(dirty_val)
        except (TypeError, ValueError):
            # Can't parse — fall back to statistical repair
            return self._statistical_repair(context, rng)

        candidates: List[Tuple[Any, float]] = []

        if stats is not None:
            vmin, vmax = stats["min"], stats["max"]
            median = stats["median"]
            use_int = "int" in context.dtype

            # Heuristic 1: divide by 11 (digit-repeat pattern: 33=3×11, 88=8×11)
            div11 = num / 11.0
            if vmin <= div11 <= vmax and abs(div11 - round(div11)) < 0.01:
                v = int(round(div11)) if use_int else div11
                candidates.append((v, 0.75))

            # Heuristic 2: divide by 10
            div10 = num / 10.0
            if vmin <= div10 <= vmax:
                v = int(round(div10)) if use_int else div10
                if v not in [c[0] for c in candidates]:
                    candidates.append((v, 0.55))

            # Heuristic 3: clamp to nearest boundary
            clamped = max(vmin, min(vmax, num))
            if use_int:
                clamped = int(round(clamped))
            if clamped not in [c[0] for c in candidates]:
                candidates.append((clamped, 0.35))

            # Heuristic 4: column median
            med_v = int(round(median)) if use_int else median
            if med_v not in [c[0] for c in candidates]:
                candidates.append((med_v, 0.25))

        if not candidates:
            return self._statistical_repair(context, rng)

        # Add seeded noise and limit to top_k
        candidates = candidates[: self.top_k]
        noise = rng.uniform(-_NOISE_SCALE * 0.5, _NOISE_SCALE * 0.5, size=len(candidates))
        noisy = [(v, max(0.01, s + noise[i])) for i, (v, s) in enumerate(candidates)]

        return noisy, "range_normalisation_repair"

    def _format_numerical_repair(
        self,
        context: RepairContext,
        rng: np.random.Generator,
    ) -> Tuple[List[Tuple[Any, float]], str]:
        """
        FORMAT_ERROR in a numerical column: extract numeric characters and try to
        recover a valid number.

        Example: '1a0' → extract digits → '10' → check range → candidate
        """
        dirty_str = str(context.dirty_value)
        candidates: List[Tuple[Any, float]] = []
        stats = context.column_stats
        use_int = "int" in context.dtype

        # Strategy 1: strip all non-digit/non-dot characters
        numeric_chars = re.sub(r"[^0-9.\-]", "", dirty_str)
        if numeric_chars:
            try:
                recovered = float(numeric_chars)
                if stats is not None and stats["min"] <= recovered <= stats["max"]:
                    v = int(round(recovered)) if use_int else recovered
                    candidates.append((v, 0.70))
                elif numeric_chars.isdigit():
                    # Try digit-squash (e.g. '10' from '1a0' — might be digit repeat)
                    v = int(numeric_chars) if use_int else float(numeric_chars)
                    candidates.append((v, 0.50))
            except ValueError:
                pass

        # Strategy 2: keep only leading digit sequence
        leading = re.match(r"^\d+", dirty_str.strip())
        if leading:
            try:
                leading_val = int(leading.group()) if use_int else float(leading.group())
                if leading_val not in [c[0] for c in candidates]:
                    score = 0.55 if (stats and stats["min"] <= leading_val <= stats["max"]) else 0.30
                    candidates.append((leading_val, score))
            except ValueError:
                pass

        # Fallback: column median
        if stats is not None:
            med = int(round(stats["median"])) if use_int else stats["median"]
            if med not in [c[0] for c in candidates]:
                candidates.append((med, 0.25))

        if not candidates:
            return self._statistical_repair(context, rng)

        candidates = candidates[: self.top_k]
        noise = rng.uniform(-_NOISE_SCALE * 0.5, _NOISE_SCALE * 0.5, size=len(candidates))
        noisy = [(v, max(0.01, s + noise[i])) for i, (v, s) in enumerate(candidates)]
        return noisy, "format_numerical_repair"

    def _format_categorical_repair(
        self,
        context: RepairContext,
        rng: np.random.Generator,
    ) -> Tuple[List[Tuple[Any, float]], str]:
        """
        FORMAT_ERROR in CATEGORICAL/MIXED: use similarity repair as first strategy,
        with additional attempt to extract meaningful substring.
        """
        dirty_str = str(context.dirty_value).strip()

        # First try similarity
        sim_candidates, _ = self._similarity_repair(context, rng)

        # Also try extracting numeric part (for MIXED columns)
        if context.kind == ColumnKind.MIXED:
            digits_only = re.sub(r"[^0-9]", "", dirty_str)
            if digits_only and digits_only not in [str(v) for v, _ in sim_candidates]:
                sim_candidates.append((digits_only, 0.4))

        return sim_candidates[: self.top_k], "format_categorical_repair"

    def _whitespace_repair(
        self,
        context: RepairContext,
        rng: np.random.Generator,
    ) -> Tuple[List[Tuple[Any, float]], str]:
        """
        WHITESPACE: strip whitespace and apply case normalisation.

        The stripped value is the primary candidate with high confidence.
        Add a few alternatives from vocabulary as fallbacks.
        """
        stripped = str(context.dirty_value).strip()
        candidates: List[Tuple[Any, float]] = [(stripped, 0.85)]

        # Add vocabulary alternatives as lower-probability candidates
        vocab = context.vocabulary
        for v in vocab[: self.top_k - 1]:
            if v != stripped and v not in [c[0] for c in candidates]:
                candidates.append((v, 0.1))

        noise = rng.uniform(-_NOISE_SCALE * 0.3, _NOISE_SCALE * 0.3, size=len(candidates))
        noisy = [(v, max(0.01, s + noise[i])) for i, (v, s) in enumerate(candidates)]
        return noisy[: self.top_k], "whitespace_repair"

    # ------------------------------------------------------------------ #
    # Seeded RNG                                                           #
    # ------------------------------------------------------------------ #

    def _make_rng(self, context: RepairContext) -> np.random.Generator:
        """
        Create a deterministic RNG for this specific cell.

        Seed = base_seed XOR hash(dataset:column:row_id).
        This guarantees: same cell → same candidates on every run.
        """
        cell_key = f"{context.dataset}:{context.column}:{context.row_id}"
        hash_int = int(hashlib.md5(cell_key.encode()).hexdigest(), 16) % (2**31)
        combined_seed = (self.seed ^ hash_int) % (2**31)
        return np.random.default_rng(combined_seed)


# ═══════════════════════════════════════════════════════════════════════════════
# Probability validation and normalisation
# ═══════════════════════════════════════════════════════════════════════════════


def _validate_and_normalise(
    raw: List[Tuple[Any, float]],
    top_k: int,
) -> List[RepairCandidate]:
    """
    Validate raw (value, score) pairs and normalise probabilities to sum to 1.

    Rules:
    - Filter out entries with non-numeric or negative scores.
    - Keep at most top_k candidates.
    - Probabilities must each be in (0, 1] after normalisation.
    - Total must sum to 1.0 (within floating-point tolerance).
    """
    valid: List[Tuple[Any, float]] = []
    for val, score in raw:
        if not isinstance(score, (int, float)):
            continue
        if math.isnan(score) or math.isinf(score):
            continue
        if score <= 0:
            continue
        valid.append((val, float(score)))

    if not valid:
        return []

    # Limit to top_k
    valid = sorted(valid, key=lambda x: x[1], reverse=True)[:top_k]

    # Normalise
    total = sum(s for _, s in valid)
    if total <= 0:
        return []

    return [
        RepairCandidate(value=val, probability=round(score / total, 6))
        for val, score in valid
    ]
