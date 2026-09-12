"""
src/detection/detector.py
--------------------------
Rule-based error detection for dirty DataFrames.

IMPORTANT — DATA LEAKAGE PREVENTION
=====================================
This module MUST NOT access clean.csv or any per-cell clean values.

What IS allowed:
  - dirty_df  — the actual input to the pipeline
  - DatasetProfile — aggregate schema statistics derived from the clean dataset
    (population ranges, known vocabularies, IQR values). These are
    schema-level facts, not per-cell ground truth.

What is NOT allowed:
  - Comparing dirty_df.iloc[row, col]  with  clean_df.iloc[row, col]
  - Importing or accepting a clean DataFrame as parameter in detect()

Why profile stats are acceptable:
  A column range like [1, 10] or a vocabulary like {Private, Self-emp, …}
  represents schema knowledge, equivalent to domain expertise a human data
  engineer would apply. It does not reveal which specific cells are wrong —
  it tells us what valid values look like. Downstream detection then flags
  values that violate those population-level rules.

Detection output concept separation
=====================================
  DETECTION SCORE  — how suspicious this cell appears
  (NOT the probability that a specific repair is correct — that is Phase 5)

Detection methods implemented
===============================
  1. MISSING           — NaN, empty string, null-like tokens
  2. OUT_OF_RANGE       — outside clean-column [min, max]
  3. STATISTICAL_OUTLIER— beyond IQR fence or z-score threshold
  4. FORMAT_ERROR       — dirty value cannot be parsed as expected type
  5. NOVEL_CATEGORY     — value absent from the known vocabulary
  6. WHITESPACE         — leading/trailing whitespace or inconsistent casing

Each suspicious cell produces one DetectionRecord.
One cell can produce at most one record (highest-scoring issue wins).

Usage
=====
    from src.data.loader import DataLoader
    from src.data.profiler import profile_dataset
    from src.detection.detector import Detector

    loader = DataLoader("datasets")
    clean_df, dirty_df = loader.load("adult")
    profile = profile_dataset("adult", loader.discover("adult"), clean_df, dirty_df)

    # NOTE: detector receives ONLY dirty_df — no clean_df
    detector = Detector(profile)
    records = detector.detect(dirty_df)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from src.data.schema import ColumnKind, DatasetProfile, NumericalStats

logger = logging.getLogger(__name__)

# ── Null-like string tokens treated as MISSING ────────────────────────────────
_NULL_STRINGS: Set[str] = {
    "", "null", "none", "na", "n/a", "nan", "nil", "missing",
    "undefined", "unknown", "#n/a", "n.a.", "n.a",
}

# ── Default IQR multiplier (cells beyond Q1 - k*IQR or Q3 + k*IQR are flagged)
_DEFAULT_IQR_K: float = 3.0   # conservative: reduces false positives

# ── Default z-score threshold
_DEFAULT_ZSCORE_THRESHOLD: float = 4.0   # very conservative


class ErrorType(str, Enum):
    """Structured error type labels produced by the detector."""

    MISSING = "MISSING"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    STATISTICAL_OUTLIER = "STATISTICAL_OUTLIER"
    FORMAT_ERROR = "FORMAT_ERROR"
    NOVEL_CATEGORY = "NOVEL_CATEGORY"
    WHITESPACE = "WHITESPACE"


@dataclass
class DetectionRecord:
    """
    One suspicious cell identified by the detector.

    Fields
    ------
    dataset : str
        Name of the dataset (e.g. "adult").
    row_id : int
        Zero-based row index in the dirty DataFrame.
    column : str
        Column name.
    dirty_value : Any
        The raw value in the dirty DataFrame (may be NaN).
    error_type : str
        One of the ErrorType enum values.
    detection_score : float
        In [0, 1]. Higher = more suspicious.
        This is a suspicion score, NOT a repair confidence.
    detection_method : str
        Short identifier of the detection rule used.
    reason : str
        Human-readable explanation for why this cell is suspicious.
    """

    dataset: str
    row_id: int
    column: str
    dirty_value: Any
    error_type: str
    detection_score: float
    detection_method: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "row_id": self.row_id,
            "column": self.column,
            "dirty_value": self.dirty_value,
            "error_type": self.error_type,
            "detection_score": self.detection_score,
            "detection_method": self.detection_method,
            "reason": self.reason,
        }


class Detector:
    """
    Schema-aware, rule-based error detector.

    Detects suspicious cells in a dirty DataFrame using the DatasetProfile
    (which encodes population-level schema knowledge from the clean data).

    The detector never accesses per-cell clean values. It is safe to call
    detect() passing only the dirty DataFrame.

    Parameters
    ----------
    profile : DatasetProfile
        Profile built during Phase 2. Provides column kinds, valid ranges,
        known vocabularies, and IQR values.
    iqr_k : float
        IQR fence multiplier. Default 3.0 (conservative).
    zscore_threshold : float
        Z-score threshold for outlier detection. Default 4.0 (conservative).
    null_strings : set[str] | None
        Additional null-like string tokens to treat as MISSING.
        Merged with the built-in set.
    flag_statistical_outliers : bool
        If False, skip IQR/z-score checks. Useful for datasets like
        Smart Factory where non-NaN values are known-correct sensor readings.
        Default True.
    """

    def __init__(
        self,
        profile: DatasetProfile,
        iqr_k: float = _DEFAULT_IQR_K,
        zscore_threshold: float = _DEFAULT_ZSCORE_THRESHOLD,
        null_strings: Optional[Set[str]] = None,
        flag_statistical_outliers: bool = True,
    ) -> None:
        self.profile = profile
        self.iqr_k = iqr_k
        self.zscore_threshold = zscore_threshold
        self.flag_statistical_outliers = flag_statistical_outliers
        self._null_strings = _NULL_STRINGS | (null_strings or set())

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def detect(self, dirty_df: pd.DataFrame) -> List[DetectionRecord]:
        """
        Detect suspicious cells in *dirty_df*.

        Parameters
        ----------
        dirty_df : pd.DataFrame
            The dirty input DataFrame. Must NOT be the clean DataFrame.
            Column order must match profile.column_names.

        Returns
        -------
        list[DetectionRecord]
            One record per suspicious cell (highest-scoring issue per cell).
            Sorted by (column, row_id).
        """
        # Map (row_id, column) → best record seen so far
        best: Dict[tuple, DetectionRecord] = {}

        for col in self.profile.column_names:
            if col not in dirty_df.columns:
                logger.warning(
                    "Column '%s' in profile but not in dirty_df — skipping.", col
                )
                continue

            col_profile = self.profile.columns[col]
            col_series = dirty_df[col]

            records = self._detect_column(col, col_series, col_profile)
            for rec in records:
                key = (rec.row_id, rec.column)
                if key not in best or rec.detection_score > best[key].detection_score:
                    best[key] = rec

        result = sorted(best.values(), key=lambda r: (r.column, r.row_id))
        logger.info(
            "Detection complete for '%s': %d suspicious cells found across %d columns.",
            self.profile.name,
            len(result),
            len({r.column for r in result}),
        )
        return result

    def detection_summary(self, records: List[DetectionRecord]) -> Dict[str, Any]:
        """Return a summary dict of detection results (counts by error type, column)."""
        by_type: Dict[str, int] = {}
        by_col: Dict[str, int] = {}
        for r in records:
            by_type[r.error_type] = by_type.get(r.error_type, 0) + 1
            by_col[r.column] = by_col.get(r.column, 0) + 1
        return {
            "dataset": self.profile.name,
            "total_suspicious_cells": len(records),
            "by_error_type": by_type,
            "by_column": by_col,
        }

    # ------------------------------------------------------------------ #
    # Column dispatch                                                      #
    # ------------------------------------------------------------------ #

    def _detect_column(
        self,
        col: str,
        series: pd.Series,
        col_profile,
    ) -> List[DetectionRecord]:
        """Run all applicable checks for one column and return all flagged records."""
        all_records: List[DetectionRecord] = []

        # 1. Missing-value detection — applies to every column kind
        all_records.extend(self._detect_missing(col, series))

        # Only check non-null values for subsequent rules
        non_null_mask = series.notna()
        if col_profile.kind == ColumnKind.ID_LIKE:
            # ID columns: only report missing; don't flag value anomalies
            return all_records

        if col_profile.kind == ColumnKind.NUMERICAL:
            all_records.extend(
                self._detect_numerical(col, series, col_profile, non_null_mask)
            )

        elif col_profile.kind == ColumnKind.BINARY:
            # Binary columns: check that non-null values are in the known two-value set
            all_records.extend(
                self._detect_binary(col, series, col_profile, non_null_mask)
            )

        elif col_profile.kind in (ColumnKind.CATEGORICAL, ColumnKind.MIXED):
            all_records.extend(
                self._detect_categorical(col, series, col_profile, non_null_mask)
            )

        return all_records

    # ------------------------------------------------------------------ #
    # 1. Missing-value detection                                          #
    # ------------------------------------------------------------------ #

    def _detect_missing(self, col: str, series: pd.Series) -> List[DetectionRecord]:
        """
        Flag cells that are NaN or contain null-like string tokens.

        Detection score is 1.0 — a missing value is unambiguously suspicious
        (even if the clean data also has missing in the same position, which
        we cannot know without ground truth).
        """
        records: List[DetectionRecord] = []

        for row_id, val in series.items():
            is_missing = False
            reason = ""

            if pd.isna(val):
                is_missing = True
                reason = "Cell is NaN"
            elif isinstance(val, str) and val.strip().lower() in self._null_strings:
                is_missing = True
                reason = f"Null-like string token: '{val}'"

            if is_missing:
                records.append(
                    DetectionRecord(
                        dataset=self.profile.name,
                        row_id=int(row_id),
                        column=col,
                        dirty_value=val,
                        error_type=ErrorType.MISSING.value,
                        detection_score=1.0,
                        detection_method="missing_value",
                        reason=reason,
                    )
                )

        return records

    # ------------------------------------------------------------------ #
    # 2. Numerical anomaly detection                                      #
    # ------------------------------------------------------------------ #

    def _detect_numerical(
        self,
        col: str,
        series: pd.Series,
        col_profile,
        non_null_mask: pd.Series,
    ) -> List[DetectionRecord]:
        """
        Flag non-null numerical values that are out-of-range, statistically
        anomalous, or unparseable as numbers.

        Checks (applied in priority order per cell):
        a. Format error — value cannot be parsed as a number (highest priority)
        b. Out-of-range — outside [profile.min, profile.max]
        c. Statistical outlier — beyond IQR fence (if enabled)
        """
        records: List[DetectionRecord] = []
        stats = col_profile.numerical_stats  # from clean data (population stats)

        non_null_vals = series[non_null_mask]

        # Try parsing all non-null values as numbers
        numeric_vals = pd.to_numeric(non_null_vals, errors="coerce")

        for row_id in non_null_vals.index:
            raw_val = non_null_vals[row_id]
            num_val = numeric_vals[row_id]
            rec = None

            # a. Format error
            if pd.isna(num_val):
                rec = DetectionRecord(
                    dataset=self.profile.name,
                    row_id=int(row_id),
                    column=col,
                    dirty_value=raw_val,
                    error_type=ErrorType.FORMAT_ERROR.value,
                    detection_score=0.85,
                    detection_method="format_check",
                    reason=(
                        f"Column '{col}' is numerical (dtype {col_profile.dtype_clean}) "
                        f"but dirty value '{raw_val}' cannot be parsed as a number."
                    ),
                )

            # b. Out-of-range (uses profile range derived from clean data)
            elif stats is not None and (
                num_val < stats.min or num_val > stats.max
            ):
                rec = DetectionRecord(
                    dataset=self.profile.name,
                    row_id=int(row_id),
                    column=col,
                    dirty_value=raw_val,
                    error_type=ErrorType.OUT_OF_RANGE.value,
                    detection_score=0.9,
                    detection_method="range_check",
                    reason=(
                        f"Value {num_val} is outside the observed clean range "
                        f"[{stats.min}, {stats.max}] for column '{col}'."
                    ),
                )

            # c. Statistical outlier (IQR fence)
            elif self.flag_statistical_outliers and stats is not None and stats.iqr > 0:
                lower_fence = stats.q25 - self.iqr_k * stats.iqr
                upper_fence = stats.q75 + self.iqr_k * stats.iqr
                if num_val < lower_fence or num_val > upper_fence:
                    # Score scales with how far beyond the fence the value is
                    distance = max(
                        max(0, lower_fence - num_val),
                        max(0, num_val - upper_fence),
                    )
                    score = min(0.75, 0.45 + 0.1 * (distance / (stats.iqr + 1e-9)))
                    rec = DetectionRecord(
                        dataset=self.profile.name,
                        row_id=int(row_id),
                        column=col,
                        dirty_value=raw_val,
                        error_type=ErrorType.STATISTICAL_OUTLIER.value,
                        detection_score=round(score, 4),
                        detection_method="iqr_outlier",
                        reason=(
                            f"Value {num_val} is beyond the IQR fence "
                            f"[{lower_fence:.2f}, {upper_fence:.2f}] "
                            f"(Q1={stats.q25}, Q3={stats.q75}, IQR={stats.iqr}, k={self.iqr_k})."
                        ),
                    )

            if rec is not None:
                records.append(rec)

        return records

    # ------------------------------------------------------------------ #
    # 3. Binary column detection                                          #
    # ------------------------------------------------------------------ #

    def _detect_binary(
        self,
        col: str,
        series: pd.Series,
        col_profile,
        non_null_mask: pd.Series,
    ) -> List[DetectionRecord]:
        """
        For binary columns, flag non-null values that are not in the known
        two-value vocabulary from the clean data.

        NOTE: Label flips (e.g. Adult income 0↔1) are NOT detectable here
        because both 0 and 1 are valid vocabulary values. This is an inherent
        limitation of rule-based detection on binary columns without context.
        """
        if col_profile.categorical_stats is None:
            return []

        known_vocab: Set[str] = set(col_profile.categorical_stats.value_counts.keys())
        records: List[DetectionRecord] = []
        non_null_vals = series[non_null_mask]

        for row_id, val in non_null_vals.items():
            val_str = str(val).strip()
            if val_str not in known_vocab:
                records.append(
                    DetectionRecord(
                        dataset=self.profile.name,
                        row_id=int(row_id),
                        column=col,
                        dirty_value=val,
                        error_type=ErrorType.NOVEL_CATEGORY.value,
                        detection_score=0.7,
                        detection_method="vocab_check",
                        reason=(
                            f"Value '{val}' is not in the known binary vocabulary "
                            f"{known_vocab} for column '{col}'."
                        ),
                    )
                )

        return records

    # ------------------------------------------------------------------ #
    # 4. Categorical / mixed anomaly detection                            #
    # ------------------------------------------------------------------ #

    def _detect_categorical(
        self,
        col: str,
        series: pd.Series,
        col_profile,
        non_null_mask: pd.Series,
    ) -> List[DetectionRecord]:
        """
        Detect anomalies in categorical and mixed columns:

        a. Novel category  — value not in the clean vocabulary
        b. Format error    — MIXED column value cannot be parsed as a number
        c. Whitespace      — leading/trailing whitespace or mixed case
        """
        records: List[DetectionRecord] = []
        non_null_vals = series[non_null_mask]

        known_vocab: Set[str] = set()
        if col_profile.categorical_stats is not None:
            known_vocab = set(col_profile.categorical_stats.value_counts.keys())

        for row_id, val in non_null_vals.items():
            val_str = str(val)
            stripped = val_str.strip()
            rec = None

            # a. Novel category (not in known vocabulary)
            if known_vocab and val_str not in known_vocab:
                # Check if the issue is purely whitespace/casing
                if stripped.lower() in {v.lower() for v in known_vocab}:
                    rec = DetectionRecord(
                        dataset=self.profile.name,
                        row_id=int(row_id),
                        column=col,
                        dirty_value=val,
                        error_type=ErrorType.WHITESPACE.value,
                        detection_score=0.35,
                        detection_method="whitespace_check",
                        reason=(
                            f"Value '{val}' matches a known category case-insensitively "
                            f"but has different casing or whitespace in column '{col}'."
                        ),
                    )
                elif stripped != val_str:
                    rec = DetectionRecord(
                        dataset=self.profile.name,
                        row_id=int(row_id),
                        column=col,
                        dirty_value=val,
                        error_type=ErrorType.WHITESPACE.value,
                        detection_score=0.3,
                        detection_method="whitespace_check",
                        reason=(
                            f"Value '{val}' has leading/trailing whitespace in column '{col}'."
                        ),
                    )
                else:
                    rec = DetectionRecord(
                        dataset=self.profile.name,
                        row_id=int(row_id),
                        column=col,
                        dirty_value=val,
                        error_type=ErrorType.NOVEL_CATEGORY.value,
                        detection_score=0.7,
                        detection_method="vocab_check",
                        reason=(
                            f"Value '{val}' is not in the known vocabulary "
                            f"({len(known_vocab)} values) for column '{col}'."
                        ),
                    )

            # b. Format error for MIXED columns: value not parseable as number AND
            #    not already in the known vocabulary.
            #    If a non-numeric value like '?' IS in the known vocabulary, it is a
            #    domain-accepted token (e.g. Breast Cancer 'Bare Nuclei' uses '?' to
            #    denote legitimately missing-but-not-NaN values). Do not re-flag it.
            elif col_profile.kind == ColumnKind.MIXED and val_str not in known_vocab:
                num = pd.to_numeric(val_str, errors="coerce")
                if pd.isna(num) and val_str.lower() not in self._null_strings:
                    if rec is None:
                        rec = DetectionRecord(
                            dataset=self.profile.name,
                            row_id=int(row_id),
                            column=col,
                            dirty_value=val,
                            error_type=ErrorType.FORMAT_ERROR.value,
                            detection_score=0.8,
                            detection_method="format_check_mixed",
                            reason=(
                                f"MIXED column '{col}' expects mostly-numeric values; "
                                f"'{val}' cannot be parsed as a number and is not in "
                                f"the known vocabulary."
                            ),
                        )

            if rec is not None:
                records.append(rec)

        return records
