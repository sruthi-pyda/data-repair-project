"""
src/data/profiler.py
--------------------
Full dataset profiling: column classification, numerical statistics,
categorical summaries, and ground-truth corruption characterisation.

Design constraints:
- Primary statistics (kind, numerical_stats, categorical_stats) come from
  the *clean* DataFrame only.  The clean data is the reference distribution.
- The dirty DataFrame is used only to compute:
    a) dirty-side counts (missing_dirty, n_unique_dirty, dtype_dirty) — metadata
    b) corruption summary (for evaluation reporting) — ground-truth comparison
- Neither the per-column stats nor the corruption characterisation must be fed
  into the detection or repair modules. They are for profiling, evaluation, and
  the Streamlit dashboard.

Column kind classification rules (derived from clean data):
  ID_LIKE    – cardinality > 80 % of rows (likely a row identifier)
  BINARY     – exactly 2 distinct non-null values in clean column
  NUMERICAL  – numeric dtype (int / float) and not ID_LIKE / BINARY
  MIXED      – object dtype where ≥ 80 % of non-null values parse as numbers
  CATEGORICAL – everything else (object dtype with a manageable vocabulary)

Usage:
    from src.data.loader import DataLoader
    from src.data.profiler import profile_dataset, print_profile

    loader = DataLoader("datasets")
    clean_df, dirty_df = loader.load("adult")
    paths = loader.discover("adult")
    profile = profile_dataset("adult", paths, clean_df, dirty_df)
    print_profile(profile)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.schema import (
    CategoricalStats,
    ColumnCompatibility,
    ColumnKind,
    ColumnProfile,
    CorruptionKind,
    CorruptionSummary,
    DatasetPaths,
    DatasetProfile,
    DatasetValidation,
    NumericalStats,
)

logger = logging.getLogger(__name__)

# ── Tunable constants ────────────────────────────────────────────────────────

# Fraction of rows below which a category is considered "rare"
_RARE_THRESHOLD: float = 0.01

# Cardinality fraction above which a column is treated as ID_LIKE
_ID_CARDINALITY_THRESHOLD: float = 0.80

# Fraction of non-null values that must parse as numeric for MIXED classification
_MIXED_NUMERIC_FRACTION: float = 0.80

# Maximum number of top values stored in CategoricalStats
_TOP_N: int = 20


# ═══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════════


def profile_dataset(
    name: str,
    paths: DatasetPaths,
    clean_df: pd.DataFrame,
    dirty_df: pd.DataFrame,
    rare_threshold: float = _RARE_THRESHOLD,
) -> DatasetProfile:
    """
    Build a full DatasetProfile from a clean/dirty DataFrame pair.

    Parameters
    ----------
    name : str
        Dataset name (e.g. "adult").
    paths : DatasetPaths
        Resolved file paths (stored as metadata).
    clean_df : pd.DataFrame
        The clean reference DataFrame.
    dirty_df : pd.DataFrame
        The dirty input DataFrame (must already be column-aligned to clean_df).
    rare_threshold : float
        Fraction of rows below which a category value is considered rare.

    Returns
    -------
    DatasetProfile
    """
    n_rows = clean_df.shape[0]
    common_cols = [c for c in clean_df.columns if c in dirty_df.columns]

    column_profiles: Dict[str, ColumnProfile] = {}
    for col in common_cols:
        cp = _profile_column(
            col,
            clean_series=clean_df[col],
            dirty_series=dirty_df[col],
            n_rows=n_rows,
            rare_threshold=rare_threshold,
        )
        column_profiles[col] = cp

    total_missing_clean = int(clean_df[common_cols].isna().values.sum())
    total_missing_dirty = int(dirty_df[common_cols].isna().values.sum())

    # ── Ground-truth cell diff ───────────────────────────────────────────────
    n_corrupted, corrupted_by_col = _count_corrupted_cells(
        clean_df[common_cols], dirty_df[common_cols]
    )

    logger.info(
        "Profiled '%s': %d rows × %d cols | "
        "missing_dirty=%d | corrupted_cells=%d",
        name,
        n_rows,
        len(common_cols),
        total_missing_dirty,
        n_corrupted,
    )

    return DatasetProfile(
        name=name,
        clean_path=paths.clean_path,
        dirty_path=paths.dirty_path,
        n_rows=n_rows,
        n_cols=len(common_cols),
        column_names=common_cols,
        columns=column_profiles,
        total_missing_clean=total_missing_clean,
        total_missing_dirty=total_missing_dirty,
        n_corrupted_cells=n_corrupted,
        corrupted_cells_by_column=corrupted_by_col,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Column profiling
# ═══════════════════════════════════════════════════════════════════════════════


def _profile_column(
    name: str,
    clean_series: pd.Series,
    dirty_series: pd.Series,
    n_rows: int,
    rare_threshold: float,
) -> ColumnProfile:
    """Build a ColumnProfile for one column."""
    kind = _classify_column_kind(clean_series, n_rows)
    dtype_clean = str(clean_series.dtype)
    dtype_dirty = str(dirty_series.dtype)

    numerical_stats: Optional[NumericalStats] = None
    categorical_stats: Optional[CategoricalStats] = None

    # Compute NumericalStats only when the clean series is actually numeric.
    # BINARY columns with object dtype (e.g. gender: "Male"/"Female") have no
    # meaningful numerical stats, so we skip them to avoid NaN-filled results.
    needs_numerical_stats = (
        kind == ColumnKind.NUMERICAL
        or (kind == ColumnKind.BINARY and pd.api.types.is_numeric_dtype(clean_series))
    )
    if needs_numerical_stats:
        numerical_stats = _compute_numerical_stats(clean_series)

    if kind in (ColumnKind.CATEGORICAL, ColumnKind.BINARY, ColumnKind.MIXED, ColumnKind.ID_LIKE):
        categorical_stats = _compute_categorical_stats(
            clean_series, n_rows, rare_threshold
        )

    corruption = _characterize_corruption(
        name, kind, clean_series, dirty_series, numerical_stats
    )

    return ColumnProfile(
        name=name,
        kind=kind,
        dtype_clean=dtype_clean,
        dtype_dirty=dtype_dirty,
        missing_clean=int(clean_series.isna().sum()),
        missing_dirty=int(dirty_series.isna().sum()),
        n_unique_clean=int(clean_series.nunique(dropna=True)),
        n_unique_dirty=int(dirty_series.nunique(dropna=True)),
        numerical_stats=numerical_stats,
        categorical_stats=categorical_stats,
        corruption=corruption,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Column kind classification
# ═══════════════════════════════════════════════════════════════════════════════


def _classify_column_kind(series: pd.Series, n_rows: int) -> ColumnKind:
    """
    Classify a clean column into one of the ColumnKind categories.

    Rules (applied in order):
    1. If cardinality > _ID_CARDINALITY_THRESHOLD fraction of rows → ID_LIKE
    2. If exactly 2 distinct non-null values → BINARY
    3. If numeric dtype (int / float) → NUMERICAL
    4. If object dtype and ≥ _MIXED_NUMERIC_FRACTION of non-null values parse
       as numbers → MIXED
    5. Otherwise → CATEGORICAL
    """
    non_null = series.dropna()
    n_unique = non_null.nunique()

    # 1. High cardinality → ID_LIKE
    if n_rows > 0 and (n_unique / n_rows) > _ID_CARDINALITY_THRESHOLD:
        return ColumnKind.ID_LIKE

    # 2. Binary (exactly 2 distinct values)
    if n_unique == 2:
        return ColumnKind.BINARY

    # 3. Numeric dtype
    if pd.api.types.is_numeric_dtype(series):
        return ColumnKind.NUMERICAL

    # 4. Object dtype: check if mostly parseable as numbers
    if series.dtype == object:
        numeric_parseable = pd.to_numeric(non_null, errors="coerce").notna().sum()
        if len(non_null) > 0 and (numeric_parseable / len(non_null)) >= _MIXED_NUMERIC_FRACTION:
            return ColumnKind.MIXED

    # 5. Default
    return ColumnKind.CATEGORICAL


# ═══════════════════════════════════════════════════════════════════════════════
# Numerical statistics (from clean data)
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_numerical_stats(series: pd.Series) -> NumericalStats:
    """
    Compute descriptive statistics for a numerical column from clean data.

    The IQR is stored explicitly because it drives the IQR-based outlier
    detector in Phase 3.

    Parameters
    ----------
    series : pd.Series
        A column from the *clean* DataFrame. May be int or float dtype.
    """
    # For BINARY or MIXED columns that landed here, coerce to numeric
    numeric = pd.to_numeric(series, errors="coerce").dropna()

    q25 = float(numeric.quantile(0.25))
    q75 = float(numeric.quantile(0.75))

    return NumericalStats(
        mean=float(numeric.mean()),
        std=float(numeric.std()),
        min=float(numeric.min()),
        q25=q25,
        q50=float(numeric.median()),
        q75=q75,
        max=float(numeric.max()),
        iqr=q75 - q25,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Categorical statistics (from clean data)
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_categorical_stats(
    series: pd.Series,
    n_rows: int,
    rare_threshold: float,
) -> CategoricalStats:
    """
    Compute value-distribution summary for a categorical/binary/mixed column
    using *clean* data.

    Parameters
    ----------
    series : pd.Series
        A column from the *clean* DataFrame.
    n_rows : int
        Total row count of the DataFrame (used to compute "rare" threshold).
    rare_threshold : float
        A category is "rare" if it appears in fewer than this fraction of rows.

    Returns
    -------
    CategoricalStats
        value_counts  – all distinct values and their counts (desc order)
        top_values    – up to _TOP_N most frequent values (as strings)
        n_rare        – number of distinct values below the rare threshold
    """
    counts = series.dropna().astype(str).value_counts()
    value_counts_dict = {str(k): int(v) for k, v in counts.items()}

    rare_min_count = max(1, int(rare_threshold * n_rows))
    n_rare = int((counts < rare_min_count).sum())

    top_values = [str(v) for v in counts.index[:_TOP_N].tolist()]

    return CategoricalStats(
        value_counts=value_counts_dict,
        top_values=top_values,
        n_rare=n_rare,
        rare_threshold=rare_threshold,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Corruption characterisation (clean vs dirty — for reporting only)
# ═══════════════════════════════════════════════════════════════════════════════


def _characterize_corruption(
    col_name: str,
    kind: ColumnKind,
    clean_series: pd.Series,
    dirty_series: pd.Series,
    numerical_stats: Optional[NumericalStats],
) -> CorruptionSummary:
    """
    Characterise what types of corruption exist in one column.

    IMPORTANT: This result is for profiling/evaluation reporting only.
    It must NOT be used as input to the detection or repair modules.

    The function compares the clean and dirty Series cell-by-cell to count:
    - how many NaN values were introduced
    - how many values changed
    - for numerical columns: how many changed values are out of clean range
    - for object columns: how many changed values cannot be parsed as the
      expected dtype
    - for binary columns: how many are flipped to the opposite value

    Parameters
    ----------
    col_name : str
        Column name (for logging).
    kind : ColumnKind
        The classified kind of this column.
    clean_series, dirty_series : pd.Series
        Aligned series from clean and dirty DataFrames.
    numerical_stats : NumericalStats | None
        Pre-computed clean stats (needed for out-of-range check).
    """
    n_rows = min(len(clean_series), len(dirty_series))
    c = clean_series.iloc[:n_rows].reset_index(drop=True)
    d = dirty_series.iloc[:n_rows].reset_index(drop=True)

    clean_null = c.isna()
    dirty_null = d.isna()

    # Missing introduced: clean has value, dirty has NaN
    n_missing_introduced = int((~clean_null & dirty_null).sum())

    # Value changed: both non-null but values differ.
    # Use numeric comparison first to avoid false positives from dtype changes
    # caused by NaN injection (e.g. int 0 in clean → float 0.0 in dirty).
    both_present = ~clean_null & ~dirty_null
    c_present = c[both_present]
    d_present = d[both_present]
    c_num = pd.to_numeric(c_present, errors="coerce")
    d_num = pd.to_numeric(d_present, errors="coerce")
    both_numeric = c_num.notna() & d_num.notna()
    numeric_diff = both_numeric & (c_num != d_num)
    string_diff = ~both_numeric & (c_present.astype(str) != d_present.astype(str))
    value_changed_mask = numeric_diff | string_diff
    n_value_changed = int(value_changed_mask.sum())

    # Out-of-range (numerical only): dirty value outside [clean_min, clean_max]
    n_out_of_range = 0
    if kind == ColumnKind.NUMERICAL and numerical_stats is not None:
        # Look only at cells that actually changed value (not just dtype-float)
        changed_dirty = d_present[value_changed_mask]
        dirty_numeric = pd.to_numeric(changed_dirty, errors="coerce")
        n_out_of_range = int(
            (
                dirty_numeric.notna()
                & (
                    (dirty_numeric < numerical_stats.min)
                    | (dirty_numeric > numerical_stats.max)
                )
            ).sum()
        )

    # Format error: changed value that cannot be parsed as the clean dtype
    n_format_error = 0
    if kind in (ColumnKind.NUMERICAL, ColumnKind.MIXED):
        changed_dirty = d_present[value_changed_mask]
        dirty_numeric = pd.to_numeric(changed_dirty, errors="coerce")
        n_format_error = int(dirty_numeric.isna().sum())

    # Label flip (binary columns: all value changes are class flips by definition)
    n_label_flip = 0
    if kind == ColumnKind.BINARY:
        n_label_flip = n_value_changed

    # Collect which kinds are present
    kinds_present: List[CorruptionKind] = []
    if n_missing_introduced > 0:
        kinds_present.append(CorruptionKind.MISSING_INTRODUCED)
    if n_value_changed > 0:
        if kind == ColumnKind.BINARY:
            kinds_present.append(CorruptionKind.LABEL_FLIP)
        else:
            kinds_present.append(CorruptionKind.VALUE_CHANGED)
    if n_out_of_range > 0:
        kinds_present.append(CorruptionKind.OUT_OF_RANGE)
    if n_format_error > 0:
        kinds_present.append(CorruptionKind.FORMAT_ERROR)

    return CorruptionSummary(
        n_missing_introduced=n_missing_introduced,
        n_value_changed=n_value_changed,
        n_out_of_range=n_out_of_range,
        n_format_error=n_format_error,
        n_label_flip=n_label_flip,
        corruption_kinds=kinds_present,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Ground-truth cell diff count
# ═══════════════════════════════════════════════════════════════════════════════


def _count_corrupted_cells(
    clean_df: pd.DataFrame,
    dirty_df: pd.DataFrame,
) -> Tuple[int, Dict[str, int]]:
    """
    Count total and per-column corrupted cells (for ground-truth reporting).

    A cell is counted as corrupted if:
    - the clean value is non-null AND the dirty value is NaN, OR
    - both are non-null but their string representations differ.

    Note: this uses string comparison to handle dtype changes caused by NaN
    injection (e.g. the integer 3 in clean vs. 3.0 in dirty after pandas
    upcasts to float64). String '3' == '3.0' would be a false positive, so we
    also try numeric comparison when both parse as numbers.
    """
    n_rows = min(len(clean_df), len(dirty_df))
    c = clean_df.iloc[:n_rows].reset_index(drop=True)
    d = dirty_df.iloc[:n_rows].reset_index(drop=True)

    clean_nan = c.isna()
    dirty_nan = d.isna()

    introduced_nan = (~clean_nan) & dirty_nan

    # For value changes, try numeric comparison first to avoid '3' vs '3.0' mismatches
    val_diff = pd.DataFrame(False, index=c.index, columns=c.columns)
    for col in c.columns:
        both_present = ~clean_nan[col] & ~dirty_nan[col]
        # Try numeric comparison
        c_num = pd.to_numeric(c.loc[both_present, col], errors="coerce")
        d_num = pd.to_numeric(d.loc[both_present, col], errors="coerce")
        both_numeric = c_num.notna() & d_num.notna()
        # Where both are numeric: compare as numbers
        num_diff = both_numeric & (c_num != d_num)
        # Where not both numeric: compare as strings
        str_diff = ~both_numeric & (
            c.loc[both_present, col].astype(str) != d.loc[both_present, col].astype(str)
        )
        val_diff.loc[both_present, col] = num_diff | str_diff

    diff_mask = introduced_nan | val_diff

    n_total = int(diff_mask.values.sum())
    by_col = {
        col: int(diff_mask[col].sum())
        for col in c.columns
        if diff_mask[col].sum() > 0
    }
    return n_total, by_col


# ═══════════════════════════════════════════════════════════════════════════════
# Display helper
# ═══════════════════════════════════════════════════════════════════════════════


def print_profile(profile: DatasetProfile) -> None:
    """Pretty-print a DatasetProfile to stdout (useful in notebooks and CLI)."""
    w = 70
    print(f"\n{'═' * w}")
    print(f"  Dataset : {profile.name}")
    print(f"  Clean   : {profile.clean_path}")
    print(f"  Dirty   : {profile.dirty_path}")
    print(f"  Shape   : {profile.n_rows:,} rows × {profile.n_cols} columns")
    print(f"  Missing (clean) : {profile.total_missing_clean:,}")
    print(f"  Missing (dirty) : {profile.total_missing_dirty:,}")
    print(f"  Corrupted cells (ground truth) : {profile.n_corrupted_cells:,}")
    print(f"  Numerical cols  : {profile.numerical_columns}")
    print(f"  Categorical cols: {profile.categorical_columns}")
    print(f"  ID-like cols    : {profile.id_columns}")
    print(f"\n  {'Column':<35} {'Kind':<12} {'Dtype_C':<10} {'Miss_C':>6} {'Miss_D':>6} {'Corrupt':>8}")
    print(f"  {'─' * 80}")
    for col in profile.column_names:
        cp = profile.columns[col]
        corrupted = profile.corrupted_cells_by_column.get(col, 0)
        print(
            f"  {col:<35} {cp.kind.value:<12} {cp.dtype_clean:<10} "
            f"{cp.missing_clean:>6} {cp.missing_dirty:>6} {corrupted:>8}"
        )
        if cp.corruption and cp.corruption.corruption_kinds:
            kinds_str = ", ".join(k.value for k in cp.corruption.corruption_kinds)
            print(f"    {'↳ corruption: ' + kinds_str}")
    print(f"{'═' * w}\n")
