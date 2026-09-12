"""
src/data/loader.py
------------------
Dataset discovery, loading, and clean/dirty pair validation.

Responsibilities:
- Locate dataset files on disk given a root directory and dataset name.
- Load clean and dirty CSVs into pandas DataFrames without modifying them.
- Automatically handle column-order differences between clean and dirty
  (observed in Smart Factory: 'labels' is first in clean, last in dirty).
- Validate that the pair is compatible (same row count, same column set).
- Detect dtype changes caused by NaN injection (e.g. int64 → float64).

Usage:
    from src.data.loader import DataLoader

    loader = DataLoader(data_root="datasets")

    # Basic load
    clean_df, dirty_df = loader.load("adult")

    # With validation result
    clean_df, dirty_df, validation = loader.load_with_validation("breast_cancer")

    # Discovery only (no loading)
    paths = loader.discover("smartfactory")

    # Validation of already-loaded DataFrames
    validation = loader.validate("adult", clean_df, dirty_df)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src.data.schema import (
    ColumnCompatibility,
    DatasetPaths,
    DatasetValidation,
)

logger = logging.getLogger(__name__)

# Filenames probed when auto-discovering the clean/dirty pair.
# First match wins, so the common lowercase names take priority.
_CLEAN_CANDIDATES: List[str] = ["clean.csv", "Clean.csv", "CLEAN.csv"]
_DIRTY_CANDIDATES: List[str] = ["dirty.csv", "Dirty.csv", "DIRTY.csv"]


class DataLoader:
    """
    Loads REIN-style clean/dirty CSV pairs from a local dataset directory.

    Parameters
    ----------
    data_root : str | Path
        Root directory containing one sub-folder per dataset.
        Default ``"datasets"`` → ``datasets/adult/``, ``datasets/breast_cancer/``, …
    """

    def __init__(self, data_root: str | Path = "datasets") -> None:
        self.data_root = Path(data_root).resolve()
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Data root directory does not exist: {self.data_root}"
            )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def discover(self, dataset_name: str) -> DatasetPaths:
        """
        Locate the clean and dirty CSV files for *dataset_name*.

        Parameters
        ----------
        dataset_name : str
            Subdirectory name under ``data_root``, e.g. ``"adult"``.

        Returns
        -------
        DatasetPaths
            Validated paths to both files.

        Raises
        ------
        FileNotFoundError
            If the dataset directory or either CSV file is missing.
        """
        dataset_dir = self.data_root / dataset_name
        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {dataset_dir}\n"
                f"Available directories: {self.list_datasets()}"
            )

        clean_path = self._find_file(dataset_dir, _CLEAN_CANDIDATES, "clean")
        dirty_path = self._find_file(dataset_dir, _DIRTY_CANDIDATES, "dirty")

        paths = DatasetPaths(
            name=dataset_name,
            clean_path=clean_path,
            dirty_path=dirty_path,
        )
        paths.validate()

        logger.info(
            "Discovered dataset '%s': clean=%s  dirty=%s",
            dataset_name,
            clean_path.name,
            dirty_path.name,
        )
        return paths

    def load(
        self,
        dataset_name: str,
        align_columns: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and return (clean_df, dirty_df) for *dataset_name*.

        DataFrames are returned as-is from pandas — original dtypes are
        preserved. Note that NaN injection in dirty.csv will cause pandas to
        upcast integer columns to float64; this is expected and documented in
        the DatasetValidation returned by ``load_with_validation``.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset subdirectory.
        align_columns : bool
            If True, reorder dirty_df columns to match clean_df column order.
            Required for Smart Factory where 'labels' is at the end of dirty.csv.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            ``(clean_df, dirty_df)`` — raw DataFrames, never modified.
        """
        paths = self.discover(dataset_name)
        clean_df = pd.read_csv(paths.clean_path)
        dirty_df = pd.read_csv(paths.dirty_path)

        logger.info(
            "Loaded '%s':  clean=%s  dirty=%s",
            dataset_name,
            clean_df.shape,
            dirty_df.shape,
        )

        if align_columns and list(clean_df.columns) != list(dirty_df.columns):
            dirty_df = self._align_columns(clean_df, dirty_df, dataset_name)

        return clean_df, dirty_df

    def load_with_validation(
        self,
        dataset_name: str,
        align_columns: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, DatasetValidation]:
        """
        Load the dataset pair and run compatibility validation.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset subdirectory.
        align_columns : bool
            Whether to auto-align dirty column order to clean.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, DatasetValidation]
            ``(clean_df, dirty_df, validation)``

        Raises
        ------
        ValueError
            If the pair is fundamentally incompatible (different row counts or
            non-overlapping column sets).
        """
        clean_df, dirty_df = self.load(dataset_name, align_columns=align_columns)
        validation = self.validate(dataset_name, clean_df, dirty_df)
        validation.raise_if_incompatible()
        return clean_df, dirty_df, validation

    def validate(
        self,
        dataset_name: str,
        clean_df: pd.DataFrame,
        dirty_df: pd.DataFrame,
    ) -> DatasetValidation:
        """
        Validate a clean/dirty DataFrame pair for compatibility.

        Checks:
        1. Row count equality.
        2. Column set equality (same names, ignoring order).
        3. Column order equality.
        4. Dtype changes per column (e.g. int64 → float64 from NaN injection).

        Parameters
        ----------
        dataset_name : str
            Used only for logging and error messages.
        clean_df, dirty_df : pd.DataFrame
            Already-loaded DataFrames (dirty should already be column-aligned
            if load() was called with align_columns=True).

        Returns
        -------
        DatasetValidation
        """
        warnings: list[str] = []

        # ── Row count ───────────────────────────────────────────────────────
        row_count_match = len(clean_df) == len(dirty_df)
        if not row_count_match:
            warnings.append(
                f"Row count mismatch: clean={len(clean_df)}, dirty={len(dirty_df)}"
            )

        # ── Column sets ─────────────────────────────────────────────────────
        clean_col_set = set(clean_df.columns)
        dirty_col_set = set(dirty_df.columns)
        only_in_clean = sorted(clean_col_set - dirty_col_set)
        only_in_dirty = sorted(dirty_col_set - clean_col_set)
        column_set_match = not only_in_clean and not only_in_dirty

        if not column_set_match:
            warnings.append(
                f"Column set mismatch — only_in_clean={only_in_clean}, "
                f"only_in_dirty={only_in_dirty}"
            )

        column_order_match = list(clean_df.columns) == list(dirty_df.columns)
        if not column_order_match and column_set_match:
            warnings.append(
                "Column order differs between clean and dirty "
                "(loader has reordered dirty to match clean)."
            )

        # ── Dtype changes ────────────────────────────────────────────────────
        # NaN injection forces pandas to upcast integer columns to float64.
        # This is expected behaviour and is documented here, not fixed.
        dtype_changes: list[ColumnCompatibility] = []
        common_cols = [c for c in clean_df.columns if c in dirty_col_set]
        for col in common_cols:
            c_dtype = str(clean_df[col].dtype)
            d_dtype = str(dirty_df[col].dtype)
            compat = ColumnCompatibility(column=col, dtype_clean=c_dtype, dtype_dirty=d_dtype)
            dtype_changes.append(compat)
            if compat.dtype_changed:
                warnings.append(
                    f"Column '{col}': dtype changed {c_dtype} → {d_dtype} "
                    "(likely caused by NaN injection in dirty.csv)."
                )

        result = DatasetValidation(
            dataset_name=dataset_name,
            row_count_match=row_count_match,
            n_rows_clean=len(clean_df),
            n_rows_dirty=len(dirty_df),
            column_set_match=column_set_match,
            column_order_match=column_order_match,
            only_in_clean=only_in_clean,
            only_in_dirty=only_in_dirty,
            column_dtype_changes=dtype_changes,
            warnings=warnings,
        )

        if warnings:
            for w in warnings:
                logger.warning("[%s] %s", dataset_name, w)
        else:
            logger.info("Validation passed for '%s': no issues found.", dataset_name)

        return result

    def list_datasets(self) -> List[str]:
        """Return the names of all dataset subdirectories found under data_root."""
        return self._list_dataset_dirs()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _find_file(
        self,
        directory: Path,
        candidates: List[str],
        label: str,
    ) -> Path:
        """Return the first candidate filename that exists in *directory*."""
        for name in candidates:
            p = directory / name
            if p.exists():
                return p
        present = [f.name for f in directory.iterdir() if f.is_file()]
        raise FileNotFoundError(
            f"Could not find {label} CSV in {directory}. "
            f"Tried: {candidates}. "
            f"Files present: {present}"
        )

    def _align_columns(
        self,
        clean_df: pd.DataFrame,
        dirty_df: pd.DataFrame,
        dataset_name: str,
    ) -> pd.DataFrame:
        """
        Reorder dirty_df columns to match clean_df column order.

        Only reorders — does not add or drop columns. If the column sets differ,
        only common columns are kept and a warning is logged.
        """
        clean_cols = set(clean_df.columns)
        dirty_cols = set(dirty_df.columns)

        only_in_clean = clean_cols - dirty_cols
        only_in_dirty = dirty_cols - clean_cols

        if only_in_clean or only_in_dirty:
            logger.warning(
                "Column mismatch in '%s'. "
                "Only in clean: %s. Only in dirty: %s. "
                "Proceeding with common columns only.",
                dataset_name,
                only_in_clean,
                only_in_dirty,
            )
            common = [c for c in clean_df.columns if c in dirty_cols]
            return dirty_df[common]

        logger.info(
            "Reordered dirty.csv columns for '%s' to match clean.csv.",
            dataset_name,
        )
        return dirty_df[clean_df.columns]

    def _list_dataset_dirs(self) -> List[str]:
        return [
            d.name
            for d in sorted(self.data_root.iterdir())
            if d.is_dir() and not d.name.startswith(".")
        ]
