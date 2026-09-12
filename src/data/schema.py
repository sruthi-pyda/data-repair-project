"""
src/data/schema.py
------------------
All data-classes and enumerations that describe a loaded dataset, its profile,
and the compatibility between a clean/dirty pair.

Design principles:
- Plain Python dataclasses — no external dependencies.
- Safe to import from any module without circular imports.
- Every field is documented so later phases (detection, repair, evaluation) can
  rely on these structures without re-reading the source.
- The schema captures *facts about the data*; it does not make repair decisions.

Corruption ground-truth (clean vs dirty diffs) is stored here for *reporting
and evaluation* only. It must never be used as input to the repair system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Enumerations
# ═══════════════════════════════════════════════════════════════════════════════


class ColumnKind(str, Enum):
    """
    Semantic classification of a column based on its clean-data distribution.

    Used by detection and repair modules to choose appropriate strategies.

    NUMERICAL   – continuous or discrete numeric values (int/float dtype)
    BINARY      – exactly two distinct non-null values (e.g. 0/1, 2/4)
    CATEGORICAL  – string/object column with a manageable vocabulary
    MIXED       – object column where most values parse as numbers but some
                  do not (example: Breast Cancer 'Bare Nuclei' which contains '?')
    ID_LIKE     – very high cardinality column that is likely a row identifier
                  (e.g. 'Sample code number'); typically not repaired
    """

    NUMERICAL = "numerical"
    BINARY = "binary"
    CATEGORICAL = "categorical"
    MIXED = "mixed"
    ID_LIKE = "id_like"


class CorruptionKind(str, Enum):
    """
    Types of corruption observed when comparing clean vs dirty columns.

    MISSING_INTRODUCED – clean has a value, dirty has NaN
    VALUE_CHANGED      – both non-NaN but string representations differ
    OUT_OF_RANGE       – dirty numeric value falls outside clean column's
                         [min, max] range (numerical columns only)
    FORMAT_ERROR       – dirty value cannot be parsed as the same type as
                         the clean column's dtype (e.g. '1a0' in an int col)
    LABEL_FLIP         – binary column where the value is the opposite class
                         (observed in Adult 'income': 1701 rows flipped 0↔1)
    """

    MISSING_INTRODUCED = "missing_introduced"
    VALUE_CHANGED = "value_changed"
    OUT_OF_RANGE = "out_of_range"
    FORMAT_ERROR = "format_error"
    LABEL_FLIP = "label_flip"


# ═══════════════════════════════════════════════════════════════════════════════
# File paths
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DatasetPaths:
    """Resolved file-system paths for a clean/dirty dataset pair."""

    name: str
    clean_path: Path
    dirty_path: Path

    def validate(self) -> None:
        """Raise FileNotFoundError if either file is missing."""
        for p in (self.clean_path, self.dirty_path):
            if not p.exists():
                raise FileNotFoundError(f"Dataset file not found: {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# Per-column statistics (computed from clean data only)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class NumericalStats:
    """
    Descriptive statistics for a numerical column, computed from *clean* data.

    These are used by the detection module (e.g. IQR-based outlier detection)
    and by the Streamlit dashboard. All values are floats for JSON-serialisability.
    """

    mean: float
    std: float
    min: float
    q25: float       # 25th percentile
    q50: float       # median
    q75: float       # 75th percentile
    max: float
    iqr: float       # q75 - q25; used for IQR outlier rule


@dataclass
class CategoricalStats:
    """
    Distribution summary for a categorical or binary column, from *clean* data.

    value_counts  – dict mapping category → count (all categories, sorted descending)
    top_values    – the most frequent values (up to top_n)
    n_rare        – number of distinct values appearing in < rare_threshold of rows
    rare_threshold – fraction used to define "rare" (from config, default 0.01)
    """

    value_counts: Dict[str, int]
    top_values: List[str]
    n_rare: int
    rare_threshold: float = 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# Ground-truth corruption summary (clean vs dirty — for reporting only)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CorruptionSummary:
    """
    Column-level characterisation of how clean and dirty differ.

    IMPORTANT: This is computed from the clean/dirty comparison for *profiling
    and evaluation reporting only*. It must NOT be passed to the detection or
    repair modules as input — that would constitute data leakage.

    Fields
    ------
    n_missing_introduced : int
        Cells where clean has a value but dirty has NaN.
    n_value_changed : int
        Cells where both are non-NaN but string representations differ.
    n_out_of_range : int
        Subset of value_changed where the dirty numeric value is outside
        [clean_min, clean_max]. Only meaningful for NUMERICAL columns.
    n_format_error : int
        Subset of value_changed where the dirty value cannot be parsed as
        the expected dtype. E.g. '1a0' in an integer column.
    n_label_flip : int
        For BINARY columns: cells where the dirty value is the opposite class.
        (Identified in Adult 'income': 1→0 or 0→1.)
    corruption_kinds : list[CorruptionKind]
        Which corruption kinds are present in this column (non-zero counts).
    total : int
        Total corrupted cells = missing_introduced + value_changed.
    """

    n_missing_introduced: int = 0
    n_value_changed: int = 0
    n_out_of_range: int = 0
    n_format_error: int = 0
    n_label_flip: int = 0
    corruption_kinds: List[CorruptionKind] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.n_missing_introduced + self.n_value_changed


# ═══════════════════════════════════════════════════════════════════════════════
# Per-column profile
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ColumnProfile:
    """
    Complete profile for one column across both the clean and dirty DataFrames.

    The primary stats (kind, numerical_stats, categorical_stats) are derived
    from *clean* data. Dirty-side counts (missing_dirty, n_unique_dirty,
    dtype_dirty) are included for comparison.

    corruption is the ground-truth diff summary — for reporting/evaluation only.
    """

    name: str
    kind: ColumnKind

    # Dtype strings from pandas (e.g. 'int64', 'object', 'float64')
    dtype_clean: str
    dtype_dirty: str

    # Missing value counts
    missing_clean: int
    missing_dirty: int

    # Unique value counts (excludes NaN)
    n_unique_clean: int
    n_unique_dirty: int

    # Per-kind statistics (one will be None depending on kind)
    numerical_stats: Optional[NumericalStats] = None   # set for NUMERICAL / BINARY
    categorical_stats: Optional[CategoricalStats] = None  # set for CATEGORICAL / BINARY / MIXED

    # Ground-truth corruption characterisation (reporting only)
    corruption: Optional[CorruptionSummary] = None

    # Backward-compat alias so Phase 1 tests keep working
    @property
    def dtype(self) -> str:
        """Alias for dtype_clean (used by Phase 1 tests)."""
        return self.dtype_clean


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset-level validation (clean vs dirty compatibility)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ColumnCompatibility:
    """Dtype comparison for one column between clean and dirty DataFrames."""

    column: str
    dtype_clean: str
    dtype_dirty: str

    @property
    def dtype_changed(self) -> bool:
        """True when dtype differs — typically int64 → float64 due to NaN injection."""
        return self.dtype_clean != self.dtype_dirty


@dataclass
class DatasetValidation:
    """
    Result of validating a clean/dirty CSV pair for compatibility.

    is_compatible is True when the pair is usable: same number of rows and
    the same set of columns (order may differ — the loader handles that).

    Warnings capture non-fatal issues (e.g. dtype changes caused by NaN injection).
    """

    dataset_name: str
    row_count_match: bool
    n_rows_clean: int
    n_rows_dirty: int
    column_set_match: bool
    column_order_match: bool
    only_in_clean: List[str]
    only_in_dirty: List[str]
    column_dtype_changes: List[ColumnCompatibility]
    warnings: List[str] = field(default_factory=list)

    @property
    def is_compatible(self) -> bool:
        """True when row counts and column sets match (order is acceptable to differ)."""
        return self.row_count_match and self.column_set_match

    def raise_if_incompatible(self) -> None:
        """Raise ValueError if the pair cannot be used together."""
        if not self.is_compatible:
            issues = []
            if not self.row_count_match:
                issues.append(
                    f"Row count mismatch: clean={self.n_rows_clean}, "
                    f"dirty={self.n_rows_dirty}"
                )
            if not self.column_set_match:
                issues.append(
                    f"Column set mismatch. "
                    f"Only in clean: {self.only_in_clean}. "
                    f"Only in dirty: {self.only_in_dirty}."
                )
            raise ValueError(
                f"Dataset '{self.dataset_name}' clean/dirty pair is incompatible:\n"
                + "\n".join(f"  • {i}" for i in issues)
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset-level profile
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DatasetProfile:
    """
    Complete profile of a dataset pair (clean + dirty).

    Populated by src.data.profiler.profile_dataset().
    Consumed downstream by:
    - detection (column kinds, numerical ranges, categorical vocabularies)
    - calibration (split counts)
    - evaluation (ground-truth corruption counts)
    - Streamlit dashboard (overview panel)
    """

    name: str
    clean_path: Path
    dirty_path: Path

    # Shape (from clean DataFrame)
    n_rows: int
    n_cols: int
    column_names: List[str]   # in clean column order

    # Per-column profiles
    columns: Dict[str, ColumnProfile] = field(default_factory=dict)

    # Aggregate missing counts
    total_missing_clean: int = 0
    total_missing_dirty: int = 0

    # Ground-truth diff totals (for reporting/evaluation — NOT for repair input)
    n_corrupted_cells: Optional[int] = None
    corrupted_cells_by_column: Dict[str, int] = field(default_factory=dict)

    # Dataset-level validation result
    validation: Optional[DatasetValidation] = None

    # ── Convenience accessors ───────────────────────────────────────────────

    @property
    def numerical_columns(self) -> List[str]:
        """Column names classified as NUMERICAL or BINARY."""
        return [
            name for name, cp in self.columns.items()
            if cp.kind in (ColumnKind.NUMERICAL, ColumnKind.BINARY)
        ]

    @property
    def categorical_columns(self) -> List[str]:
        """Column names classified as CATEGORICAL, MIXED, or BINARY."""
        return [
            name for name, cp in self.columns.items()
            if cp.kind in (ColumnKind.CATEGORICAL, ColumnKind.MIXED, ColumnKind.BINARY)
        ]

    @property
    def id_columns(self) -> List[str]:
        """Column names classified as ID_LIKE (usually excluded from repair)."""
        return [
            name for name, cp in self.columns.items()
            if cp.kind == ColumnKind.ID_LIKE
        ]

    @property
    def repairable_columns(self) -> List[str]:
        """Columns that are candidates for repair (excludes ID_LIKE columns)."""
        return [
            name for name, cp in self.columns.items()
            if cp.kind != ColumnKind.ID_LIKE
        ]

    # ── Serialisation ───────────────────────────────────────────────────────

    def summary_dict(self) -> Dict[str, Any]:
        """Return a plain dict suitable for JSON serialisation or logging."""
        return {
            "name": self.name,
            "clean_path": str(self.clean_path),
            "dirty_path": str(self.dirty_path),
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "column_names": self.column_names,
            "total_missing_clean": self.total_missing_clean,
            "total_missing_dirty": self.total_missing_dirty,
            "n_corrupted_cells": self.n_corrupted_cells,
            "corrupted_cells_by_column": self.corrupted_cells_by_column,
            "numerical_columns": self.numerical_columns,
            "categorical_columns": self.categorical_columns,
            "id_columns": self.id_columns,
        }
