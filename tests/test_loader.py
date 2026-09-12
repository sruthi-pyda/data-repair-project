"""
tests/test_loader.py
--------------------
Unit tests for src.data.loader and src.data.profiler.

Tests cover:
- DataLoader.discover() finds correct paths
- DataLoader.load() returns correct shapes
- Column alignment (Smart Factory dirty column order)
- Missing dataset raises FileNotFoundError
- profile_dataset() computes correct stats
- Ground-truth diff counts match hand-verified numbers
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.loader import DataLoader
from src.data.profiler import profile_dataset
from src.data.schema import DatasetProfile, DatasetPaths

# ── Constants ────────────────────────────────────────────────────────────────

DATA_ROOT = "datasets"

ADULT_ROWS = 45222
ADULT_COLS = 15

BC_ROWS = 699
BC_COLS = 12

SF_ROWS = 23645
SF_COLS = 19


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def loader() -> DataLoader:
    return DataLoader(DATA_ROOT)


@pytest.fixture(scope="module")
def adult_data(loader):
    return loader.load("adult")


@pytest.fixture(scope="module")
def bc_data(loader):
    return loader.load("breast_cancer")


@pytest.fixture(scope="module")
def sf_data(loader):
    return loader.load("smartfactory")


# ── DataLoader.discover() ─────────────────────────────────────────────────────


class TestDiscover:
    def test_adult_paths_exist(self, loader):
        paths = loader.discover("adult")
        assert paths.clean_path.exists()
        assert paths.dirty_path.exists()

    def test_breast_cancer_paths_exist(self, loader):
        paths = loader.discover("breast_cancer")
        assert paths.clean_path.exists()
        assert paths.dirty_path.exists()

    def test_smartfactory_paths_exist(self, loader):
        paths = loader.discover("smartfactory")
        assert paths.clean_path.exists()
        assert paths.dirty_path.exists()

    def test_missing_dataset_raises(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.discover("nonexistent_dataset_xyz")

    def test_list_datasets_contains_all_three(self, loader):
        names = loader.list_datasets()
        assert "adult" in names
        assert "breast_cancer" in names
        assert "smartfactory" in names


# ── DataLoader.load() shape tests ─────────────────────────────────────────────


class TestLoad:
    def test_adult_clean_shape(self, adult_data):
        clean_df, _ = adult_data
        assert clean_df.shape == (ADULT_ROWS, ADULT_COLS)

    def test_adult_dirty_shape(self, adult_data):
        _, dirty_df = adult_data
        assert dirty_df.shape == (ADULT_ROWS, ADULT_COLS)

    def test_adult_column_names(self, adult_data):
        clean_df, dirty_df = adult_data
        expected = [
            "age", "workclass", "fnlwgt", "education", "educational_num",
            "marital_status", "occupation", "relationship", "race", "gender",
            "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
        ]
        assert list(clean_df.columns) == expected
        assert list(dirty_df.columns) == expected

    def test_breast_cancer_clean_shape(self, bc_data):
        clean_df, _ = bc_data
        assert clean_df.shape == (BC_ROWS, BC_COLS)

    def test_breast_cancer_dirty_shape(self, bc_data):
        _, dirty_df = bc_data
        assert dirty_df.shape == (BC_ROWS, BC_COLS)

    def test_smartfactory_clean_shape(self, sf_data):
        clean_df, _ = sf_data
        assert clean_df.shape == (SF_ROWS, SF_COLS)

    def test_smartfactory_dirty_shape(self, sf_data):
        _, dirty_df = sf_data
        assert dirty_df.shape == (SF_ROWS, SF_COLS)

    def test_smartfactory_column_order_aligned(self, sf_data):
        """Dirty SF has 'labels' at the end; loader must reorder to match clean."""
        clean_df, dirty_df = sf_data
        assert list(clean_df.columns) == list(dirty_df.columns)
        assert clean_df.columns[0] == "labels"

    def test_clean_and_dirty_same_columns(self, adult_data, bc_data, sf_data):
        """After loading, clean and dirty must have identical column lists."""
        for clean_df, dirty_df in [adult_data, bc_data, sf_data]:
            assert list(clean_df.columns) == list(dirty_df.columns)


# ── DataLoader: raw data is not modified ─────────────────────────────────────


class TestDataIntegrity:
    def test_adult_clean_has_no_missing(self, adult_data):
        clean_df, _ = adult_data
        assert clean_df.isna().sum().sum() == 0

    def test_adult_dirty_has_no_missing(self, adult_data):
        """Adult dirty only has income label corruption, no NaN introduced."""
        _, dirty_df = adult_data
        assert dirty_df.isna().sum().sum() == 0

    def test_breast_cancer_dirty_has_missing(self, bc_data):
        """Breast cancer dirty introduces NaN values in multiple columns."""
        _, dirty_df = bc_data
        assert dirty_df.isna().sum().sum() > 0

    def test_smartfactory_dirty_has_missing(self, sf_data):
        """Smart factory dirty introduces NaN values via sensor dropout."""
        _, dirty_df = sf_data
        assert dirty_df.isna().sum().sum() > 0

    def test_adult_only_income_differs(self, adult_data):
        """The REIN adult dataset corrupts only the 'income' label column."""
        clean_df, dirty_df = adult_data
        for col in clean_df.columns:
            if col == "income":
                continue
            assert (clean_df[col].astype(str) == dirty_df[col].astype(str)).all(), (
                f"Unexpected difference in column '{col}'"
            )

    def test_adult_income_corruption_count(self, adult_data):
        """Exactly 1701 income labels differ between clean and dirty."""
        clean_df, dirty_df = adult_data
        diff = (clean_df["income"] != dirty_df["income"]).sum()
        assert diff == 1701


# ── profile_dataset() ─────────────────────────────────────────────────────────


class TestProfiler:
    @pytest.fixture(scope="class")
    def adult_profile(self, loader):
        clean_df, dirty_df = loader.load("adult")
        paths = loader.discover("adult")
        return profile_dataset("adult", paths, clean_df, dirty_df)

    @pytest.fixture(scope="class")
    def bc_profile(self, loader):
        clean_df, dirty_df = loader.load("breast_cancer")
        paths = loader.discover("breast_cancer")
        return profile_dataset("breast_cancer", paths, clean_df, dirty_df)

    @pytest.fixture(scope="class")
    def sf_profile(self, loader):
        clean_df, dirty_df = loader.load("smartfactory")
        paths = loader.discover("smartfactory")
        return profile_dataset("smartfactory", paths, clean_df, dirty_df)

    def test_adult_profile_shape(self, adult_profile):
        assert adult_profile.n_rows == ADULT_ROWS
        assert adult_profile.n_cols == ADULT_COLS

    def test_adult_profile_no_missing_clean(self, adult_profile):
        assert adult_profile.total_missing_clean == 0

    def test_adult_profile_no_missing_dirty(self, adult_profile):
        assert adult_profile.total_missing_dirty == 0

    def test_adult_profile_corrupted_cells(self, adult_profile):
        """Only income is corrupted in adult: 1701 cells."""
        assert adult_profile.n_corrupted_cells == 1701
        assert "income" in adult_profile.corrupted_cells_by_column
        assert adult_profile.corrupted_cells_by_column["income"] == 1701

    def test_bc_profile_shape(self, bc_profile):
        assert bc_profile.n_rows == BC_ROWS
        assert bc_profile.n_cols == BC_COLS

    def test_bc_profile_has_dirty_missing(self, bc_profile):
        assert bc_profile.total_missing_dirty > 0

    def test_bc_profile_corrupted_cells_positive(self, bc_profile):
        assert bc_profile.n_corrupted_cells > 0

    def test_sf_profile_shape(self, sf_profile):
        assert sf_profile.n_rows == SF_ROWS
        assert sf_profile.n_cols == SF_COLS

    def test_sf_profile_has_dirty_missing(self, sf_profile):
        assert sf_profile.total_missing_dirty > 0

    def test_sf_profile_clean_has_no_missing(self, sf_profile):
        assert sf_profile.total_missing_clean == 0

    def test_sf_profile_corrupted_cells_are_all_nan(self, sf_profile):
        """In smartfactory, all corruption is NaN injection (no value changes)."""
        # All corrupted cells must be in columns that have NaN in dirty
        for col, count in sf_profile.corrupted_cells_by_column.items():
            cp = sf_profile.columns[col]
            assert cp.missing_dirty > 0, (
                f"Column '{col}' reported {count} corrupted cells "
                "but has no NaN in dirty — unexpected value change detected."
            )

    def test_profile_summary_dict_keys(self, adult_profile):
        d = adult_profile.summary_dict()
        required_keys = [
            "name", "clean_path", "dirty_path",
            "n_rows", "n_cols", "column_names",
            "total_missing_clean", "total_missing_dirty",
            "n_corrupted_cells", "corrupted_cells_by_column",
        ]
        for key in required_keys:
            assert key in d, f"Missing key in summary_dict: {key}"

    def test_profile_column_names_match(self, adult_profile):
        assert set(adult_profile.columns.keys()) == set(adult_profile.column_names)
