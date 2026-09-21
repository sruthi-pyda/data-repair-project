"""
tests/test_profiler.py
----------------------
Phase 2 tests for src.data.profiler, src.data.schema, and the loader's
validation functionality.

Coverage:
- DatasetValidation: compatibility, dtype changes, warnings
- ColumnKind classification for each dataset
- NumericalStats: correctness, ordering invariants, IQR
- CategoricalStats: value_counts, top_values, rare count
- CorruptionSummary: dataset-specific corruption types
- DatasetProfile convenience properties (repairable_columns, id_columns, …)
- load_with_validation() integration

All expected values are derived from actual dataset inspection in Phase 1/2
and must not be fabricated.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.loader import DataLoader
from src.data.profiler import _classify_column_kind, profile_dataset
from src.data.schema import (
    ColumnKind,
    CorruptionKind,
    DatasetProfile,
    DatasetValidation,
)

DATA_ROOT = "datasets"


# ── Module-level fixtures ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def loader() -> DataLoader:
    return DataLoader(DATA_ROOT)


@pytest.fixture(scope="module")
def adult_profile(loader) -> DatasetProfile:
    clean_df, dirty_df = loader.load("adult")
    return profile_dataset("adult", loader.discover("adult"), clean_df, dirty_df)


@pytest.fixture(scope="module")
def bc_profile(loader) -> DatasetProfile:
    clean_df, dirty_df = loader.load("breast_cancer")
    return profile_dataset("breast_cancer", loader.discover("breast_cancer"), clean_df, dirty_df)


@pytest.fixture(scope="module")
def sf_profile(loader) -> DatasetProfile:
    clean_df, dirty_df = loader.load("smartfactory")
    return profile_dataset("smartfactory", loader.discover("smartfactory"), clean_df, dirty_df)


@pytest.fixture(scope="module")
def adult_validation(loader) -> DatasetValidation:
    clean_df, dirty_df, val = loader.load_with_validation("adult")
    return val


@pytest.fixture(scope="module")
def bc_validation(loader) -> DatasetValidation:
    clean_df, dirty_df, val = loader.load_with_validation("breast_cancer")
    return val


@pytest.fixture(scope="module")
def sf_validation(loader) -> DatasetValidation:
    clean_df, dirty_df, val = loader.load_with_validation("smartfactory")
    return val


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DatasetValidation
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidation:
    """Validation: row count, column set, dtype changes."""

    def test_all_datasets_compatible(self, adult_validation, bc_validation, sf_validation):
        for val in (adult_validation, bc_validation, sf_validation):
            assert val.is_compatible, (
                f"Dataset '{val.dataset_name}' reported incompatible: {val.warnings}"
            )

    def test_adult_no_dtype_changes(self, adult_validation):
        """Adult corruption is label-flip only — NaN is never introduced, so
        no int→float upcasting happens."""
        changed = [c for c in adult_validation.column_dtype_changes if c.dtype_changed]
        assert changed == [], f"Unexpected dtype changes in adult: {changed}"

    def test_bc_has_dtype_changes(self, bc_validation):
        """Breast cancer NaN injection causes int64 → float64 or object changes."""
        changed = [c for c in bc_validation.column_dtype_changes if c.dtype_changed]
        assert len(changed) > 0

    def test_sf_sensor_dtype_changes_are_int_to_float(self, sf_validation):
        """Smart factory NaN injection upcasts all integer sensor columns to float64."""
        for compat in sf_validation.column_dtype_changes:
            if compat.dtype_changed:
                assert compat.dtype_clean == "int64", (
                    f"Expected int64 clean dtype for '{compat.column}', "
                    f"got {compat.dtype_clean}"
                )
                assert compat.dtype_dirty == "float64", (
                    f"Expected float64 dirty dtype for '{compat.column}', "
                    f"got {compat.dtype_dirty}"
                )

    def test_sf_labels_dtype_unchanged(self, sf_validation):
        """'labels' column has no NaN in dirty — dtype must be unchanged."""
        labels_compat = next(
            c for c in sf_validation.column_dtype_changes if c.column == "labels"
        )
        assert not labels_compat.dtype_changed

    def test_column_order_match_after_alignment(
        self, adult_validation, bc_validation, sf_validation
    ):
        """After load() column alignment, clean and dirty columns must be in same order."""
        for val in (adult_validation, bc_validation, sf_validation):
            assert val.column_order_match, (
                f"Column order mismatch for '{val.dataset_name}'"
            )

    def test_load_with_validation_returns_triple(self, loader):
        clean_df, dirty_df, val = loader.load_with_validation("adult")
        assert clean_df is not None
        assert dirty_df is not None
        assert isinstance(val, DatasetValidation)

    def test_incompatible_pair_raises(self, loader):
        """A mocked incompatible pair (different row counts) must raise ValueError."""
        import pandas as pd
        clean_df, dirty_df = loader.load("adult")
        # Truncate dirty to simulate a mismatch
        truncated = dirty_df.iloc[:100].copy()
        val = loader.validate("adult", clean_df, truncated)
        assert not val.is_compatible
        with pytest.raises(ValueError):
            val.raise_if_incompatible()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Column kind classification
# ═══════════════════════════════════════════════════════════════════════════════


class TestColumnKind:
    """Verify that columns are classified into the expected kinds."""

    # ── Adult ──────────────────────────────────────────────────────────────────

    def test_adult_income_is_binary(self, adult_profile):
        assert adult_profile.columns["income"].kind == ColumnKind.BINARY

    def test_adult_gender_is_binary(self, adult_profile):
        """Gender has exactly 2 unique values (Male / Female)."""
        assert adult_profile.columns["gender"].kind == ColumnKind.BINARY

    def test_adult_age_is_numerical(self, adult_profile):
        assert adult_profile.columns["age"].kind == ColumnKind.NUMERICAL

    def test_adult_workclass_is_categorical(self, adult_profile):
        assert adult_profile.columns["workclass"].kind == ColumnKind.CATEGORICAL

    def test_adult_occupation_is_categorical(self, adult_profile):
        assert adult_profile.columns["occupation"].kind == ColumnKind.CATEGORICAL

    def test_adult_no_id_columns(self, adult_profile):
        assert adult_profile.id_columns == []

    # ── Breast Cancer ──────────────────────────────────────────────────────────

    def test_bc_unnamed_is_id_like(self, bc_profile):
        """'Unnamed: 0' has 699 unique values across 699 rows → ID_LIKE."""
        assert bc_profile.columns["Unnamed: 0"].kind == ColumnKind.ID_LIKE

    def test_bc_sample_code_is_id_like(self, bc_profile):
        """'Sample code number' has ~92% unique rows → ID_LIKE."""
        assert bc_profile.columns["Sample code number"].kind == ColumnKind.ID_LIKE

    def test_bc_bare_nuclei_is_mixed(self, bc_profile):
        """'Bare Nuclei' is object dtype with mostly numeric strings plus '?'."""
        assert bc_profile.columns["Bare Nuclei"].kind == ColumnKind.MIXED

    def test_bc_class_is_binary(self, bc_profile):
        """'class' has exactly 2 distinct values (2 and 4)."""
        assert bc_profile.columns["class"].kind == ColumnKind.BINARY

    def test_bc_clump_thickness_is_numerical(self, bc_profile):
        assert bc_profile.columns["Clump Thickness"].kind == ColumnKind.NUMERICAL

    def test_bc_has_two_id_columns(self, bc_profile):
        assert len(bc_profile.id_columns) == 2
        assert "Unnamed: 0" in bc_profile.id_columns
        assert "Sample code number" in bc_profile.id_columns

    # ── Smart Factory ──────────────────────────────────────────────────────────

    def test_sf_labels_is_binary(self, sf_profile):
        """'labels' is 0/1 → BINARY."""
        assert sf_profile.columns["labels"].kind == ColumnKind.BINARY

    def test_sf_sensor_columns_are_numerical(self, sf_profile):
        """All 18 sensor columns must be classified as NUMERICAL."""
        sensor_cols = [c for c in sf_profile.column_names if c != "labels"]
        for col in sensor_cols:
            assert sf_profile.columns[col].kind == ColumnKind.NUMERICAL, (
                f"Expected NUMERICAL for '{col}', got {sf_profile.columns[col].kind}"
            )

    def test_sf_no_id_columns(self, sf_profile):
        assert sf_profile.id_columns == []

    # ── Regression: pandas 3.x string dtype ────────────────────────────────────
    # pandas 3.0 (or 2.x with ``future.infer_string``) reads text columns as the
    # dedicated ``str`` dtype instead of ``object``. The MIXED check used to be
    # gated on ``dtype == object`` and silently fell through to CATEGORICAL.

    def test_mixed_classification_is_dtype_agnostic(self):
        """Same values must classify as MIXED under both object and str dtype."""
        values = [str(i % 10 + 1) for i in range(40)] + ["1a0", "?"]

        as_object = pd.Series(values, dtype=object)
        assert _classify_column_kind(as_object, len(values)) == ColumnKind.MIXED

        with pd.option_context("future.infer_string", True):
            as_str = pd.Series(values)
            assert as_str.dtype != object, "fixture must exercise the non-object path"
            assert _classify_column_kind(as_str, len(values)) == ColumnKind.MIXED

    def test_categorical_stays_categorical_under_str_dtype(self):
        """Non-numeric text must still be CATEGORICAL, not accidentally MIXED."""
        values = ["red", "green", "blue", "green", "red"] * 4
        with pd.option_context("future.infer_string", True):
            s = pd.Series(values)
            assert _classify_column_kind(s, len(values)) == ColumnKind.CATEGORICAL

    def test_bc_bare_nuclei_is_mixed_under_str_dtype(self, loader):
        """Full-dataset check: Bare Nuclei must be MIXED with format errors
        characterised even when pandas infers the ``str`` dtype."""
        with pd.option_context("future.infer_string", True):
            clean_df, dirty_df = loader.load("breast_cancer")
            profile = profile_dataset(
                "breast_cancer", loader.discover("breast_cancer"), clean_df, dirty_df
            )
        col = profile.columns["Bare Nuclei"]
        assert col.kind == ColumnKind.MIXED
        assert CorruptionKind.FORMAT_ERROR in col.corruption.corruption_kinds


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Numerical statistics
# ═══════════════════════════════════════════════════════════════════════════════


class TestNumericalStats:
    """Verify NumericalStats correctness using known clean-data facts."""

    def test_ordering_invariant(self, adult_profile):
        """For every column with NumericalStats: min ≤ q25 ≤ q50 ≤ q75 ≤ max."""
        for name, cp in adult_profile.columns.items():
            s = cp.numerical_stats
            if s is None:
                continue
            assert s.min <= s.q25 <= s.q50 <= s.q75 <= s.max, (
                f"Ordering violated for '{name}': {s}"
            )

    def test_iqr_equals_q75_minus_q25(self, adult_profile, bc_profile, sf_profile):
        """IQR must equal q75 - q25 for every column that has NumericalStats."""
        for profile in (adult_profile, bc_profile, sf_profile):
            for name, cp in profile.columns.items():
                s = cp.numerical_stats
                if s is None:
                    continue
                assert abs(s.iqr - (s.q75 - s.q25)) < 1e-9, (
                    f"IQR mismatch for '{profile.name}/{name}': "
                    f"iqr={s.iqr}, q75-q25={s.q75 - s.q25}"
                )

    def test_adult_age_range(self, adult_profile):
        """Adult age in clean data must be in [17, 90]."""
        s = adult_profile.columns["age"].numerical_stats
        assert s is not None
        assert s.min == 17.0
        assert s.max == 90.0

    def test_adult_age_mean_approx(self, adult_profile):
        s = adult_profile.columns["age"].numerical_stats
        assert 38.0 <= s.mean <= 39.0  # known: ~38.55

    def test_bc_clump_thickness_range(self, bc_profile):
        """Clump Thickness valid range in clean is [1, 10]."""
        s = bc_profile.columns["Clump Thickness"].numerical_stats
        assert s is not None
        assert s.min >= 1
        assert s.max <= 10

    def test_sf_sensor_has_stats(self, sf_profile):
        """All sensor columns should have NumericalStats."""
        for col in [c for c in sf_profile.column_names if c != "labels"]:
            assert sf_profile.columns[col].numerical_stats is not None, (
                f"Missing NumericalStats for sensor '{col}'"
            )

    def test_std_nonnegative(self, adult_profile, bc_profile, sf_profile):
        """Std must be ≥ 0 for every column that has NumericalStats."""
        for profile in (adult_profile, bc_profile, sf_profile):
            for name, cp in profile.columns.items():
                if cp.numerical_stats is not None:
                    assert cp.numerical_stats.std >= 0, (
                        f"Negative std for {profile.name}/{name}"
                    )

    def test_numerical_stats_none_for_categorical(self, adult_profile):
        """Purely categorical columns should not have NumericalStats."""
        assert adult_profile.columns["workclass"].numerical_stats is None
        assert adult_profile.columns["occupation"].numerical_stats is None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Categorical statistics
# ═══════════════════════════════════════════════════════════════════════════════


class TestCategoricalStats:
    """Verify CategoricalStats contents using known clean-data facts."""

    def test_adult_workclass_has_value_counts(self, adult_profile):
        s = adult_profile.columns["workclass"].categorical_stats
        assert s is not None
        assert "Private" in s.value_counts
        assert s.value_counts["Private"] == 33307

    def test_adult_workclass_top_values_is_list(self, adult_profile):
        s = adult_profile.columns["workclass"].categorical_stats
        assert isinstance(s.top_values, list)
        assert len(s.top_values) > 0
        assert s.top_values[0] == "Private"  # most frequent

    def test_adult_income_value_counts_has_two_keys(self, adult_profile):
        """Income is BINARY with values 0 and 1."""
        s = adult_profile.columns["income"].categorical_stats
        assert s is not None
        assert set(s.value_counts.keys()) == {"0", "1"}

    def test_bc_bare_nuclei_clean_vocabulary(self, bc_profile):
        """Clean Bare Nuclei contains numeric strings 1–10 plus '?'."""
        s = bc_profile.columns["Bare Nuclei"].categorical_stats
        assert s is not None
        clean_vocab = set(s.value_counts.keys())
        assert "?" in clean_vocab  # '?' exists in clean data
        # All numeric strings 1-10 should be present
        for n in map(str, range(1, 11)):
            assert n in clean_vocab, f"Missing '{n}' from clean Bare Nuclei vocabulary"

    def test_bc_class_has_exactly_two_values(self, bc_profile):
        """class column has values 2 and 4 only."""
        s = bc_profile.columns["class"].categorical_stats
        assert s is not None
        assert set(s.value_counts.keys()) == {"2", "4"}

    def test_sf_labels_has_exactly_two_values(self, sf_profile):
        s = sf_profile.columns["labels"].categorical_stats
        assert s is not None
        assert set(s.value_counts.keys()) == {"0", "1"}

    def test_categorical_stats_none_for_pure_numerical(self, sf_profile):
        """Purely NUMERICAL columns should not have CategoricalStats."""
        for col in [c for c in sf_profile.column_names if c != "labels"]:
            assert sf_profile.columns[col].categorical_stats is None, (
                f"Unexpected CategoricalStats for numerical sensor '{col}'"
            )

    def test_n_rare_is_nonnegative(self, adult_profile, bc_profile):
        for profile in (adult_profile, bc_profile):
            for name, cp in profile.columns.items():
                if cp.categorical_stats:
                    assert cp.categorical_stats.n_rare >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Corruption characterisation (ground-truth reporting)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCorruptionCharacterisation:
    """
    Verify that corruption kinds match the actual dataset-specific corruption
    types discovered in Phase 1.

    IMPORTANT: These tests verify the *profiler's characterisation of known
    ground truth*. This information is for reporting only — it is NOT used by
    the detection or repair modules.
    """

    # ── Adult ──────────────────────────────────────────────────────────────────

    def test_adult_income_is_label_flip(self, adult_profile):
        c = adult_profile.columns["income"].corruption
        assert CorruptionKind.LABEL_FLIP in c.corruption_kinds
        assert c.n_label_flip == 1701

    def test_adult_income_no_missing(self, adult_profile):
        """Adult dirty has no NaN — income corruption is pure value change."""
        c = adult_profile.columns["income"].corruption
        assert c.n_missing_introduced == 0

    def test_adult_non_income_columns_have_no_corruption(self, adult_profile):
        """Only 'income' is corrupted in the adult dataset."""
        for name, cp in adult_profile.columns.items():
            if name == "income":
                continue
            assert cp.corruption.total == 0, (
                f"Unexpected corruption in adult column '{name}': {cp.corruption}"
            )

    # ── Smart Factory ──────────────────────────────────────────────────────────

    def test_sf_sensors_only_have_missing_introduced(self, sf_profile):
        """Smart factory corruption is purely NaN injection — no value changes."""
        sensor_cols = [c for c in sf_profile.column_names if c != "labels"]
        for col in sensor_cols:
            c = sf_profile.columns[col].corruption
            assert c.n_value_changed == 0, (
                f"Unexpected value_changed in SF sensor '{col}': {c.n_value_changed}"
            )
            assert CorruptionKind.MISSING_INTRODUCED in c.corruption_kinds
            assert CorruptionKind.VALUE_CHANGED not in c.corruption_kinds

    def test_sf_labels_has_no_corruption(self, sf_profile):
        """SF 'labels' column is never corrupted."""
        c = sf_profile.columns["labels"].corruption
        assert c.total == 0

    def test_sf_missing_count_equals_corrupted_count(self, sf_profile):
        """In SF, every corrupted cell is a missing value (and vice versa)."""
        assert sf_profile.n_corrupted_cells == sf_profile.total_missing_dirty

    # ── Breast Cancer ──────────────────────────────────────────────────────────

    def test_bc_class_is_label_flip(self, bc_profile):
        c = bc_profile.columns["class"].corruption
        assert CorruptionKind.LABEL_FLIP in c.corruption_kinds
        assert c.n_label_flip == 15

    def test_bc_numerical_cols_have_out_of_range(self, bc_profile):
        """At least some BC numerical columns should have out_of_range corruption
        (e.g. digit duplication: 3 → 33, outside the 1-10 valid range)."""
        oor_cols = [
            name for name, cp in bc_profile.columns.items()
            if cp.corruption and CorruptionKind.OUT_OF_RANGE in cp.corruption.corruption_kinds
        ]
        assert len(oor_cols) > 0, "Expected at least one BC column with OUT_OF_RANGE corruption"

    def test_bc_has_missing_introduced(self, bc_profile):
        """BC dirty introduces NaN in multiple columns."""
        missing_cols = [
            name for name, cp in bc_profile.columns.items()
            if cp.corruption and CorruptionKind.MISSING_INTRODUCED in cp.corruption.corruption_kinds
        ]
        assert len(missing_cols) > 5  # many columns affected

    def test_bc_has_format_errors(self, bc_profile):
        """BC has format errors like '1a0', 'a10' in some columns."""
        format_err_cols = [
            name for name, cp in bc_profile.columns.items()
            if cp.corruption and CorruptionKind.FORMAT_ERROR in cp.corruption.corruption_kinds
        ]
        assert len(format_err_cols) > 0

    def test_bc_bare_nuclei_has_format_errors(self, bc_profile):
        c = bc_profile.columns["Bare Nuclei"].corruption
        assert CorruptionKind.FORMAT_ERROR in c.corruption_kinds

    def test_bc_bare_nuclei_has_missing(self, bc_profile):
        c = bc_profile.columns["Bare Nuclei"].corruption
        assert c.n_missing_introduced == 23

    def test_bc_total_corrupted_matches_expected(self, bc_profile):
        assert bc_profile.n_corrupted_cells == 631

    def test_bc_corrupted_by_column_class(self, bc_profile):
        assert bc_profile.corrupted_cells_by_column["class"] == 15

    def test_bc_corrupted_by_column_clump_thickness(self, bc_profile):
        assert bc_profile.corrupted_cells_by_column["Clump Thickness"] == 73

    def test_bc_corrupted_by_column_normal_nucleoli(self, bc_profile):
        assert bc_profile.corrupted_cells_by_column["Normal Nucleoli"] == 63


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DatasetProfile convenience properties
# ═══════════════════════════════════════════════════════════════════════════════


class TestProfileProperties:
    """Verify the derived lists of column names from DatasetProfile."""

    def test_adult_all_repairable(self, adult_profile):
        """No ID columns → all 15 adult columns are repairable."""
        assert len(adult_profile.repairable_columns) == 15

    def test_bc_repairable_excludes_id_cols(self, bc_profile):
        """BC has 2 ID columns → 10 repairable columns."""
        assert len(bc_profile.repairable_columns) == 10
        assert "Unnamed: 0" not in bc_profile.repairable_columns
        assert "Sample code number" not in bc_profile.repairable_columns

    def test_sf_all_repairable(self, sf_profile):
        """No ID columns in SF → all 19 columns repairable."""
        assert len(sf_profile.repairable_columns) == 19

    def test_binary_columns_are_in_numerical_list(self, adult_profile):
        """BINARY columns appear in numerical_columns regardless of dtype."""
        assert "income" in adult_profile.numerical_columns
        assert "gender" in adult_profile.numerical_columns

    def test_numeric_binary_has_numerical_stats(self, adult_profile):
        """Numeric BINARY (income: 0/1) gets NumericalStats."""
        assert adult_profile.columns["income"].numerical_stats is not None

    def test_string_binary_has_no_numerical_stats(self, adult_profile):
        """Object-dtype BINARY (gender: Male/Female) has no NumericalStats."""
        assert adult_profile.columns["gender"].numerical_stats is None

    def test_binary_columns_are_also_in_categorical_list(self, adult_profile):
        """BINARY columns also appear in categorical_columns (they have CategoricalStats)."""
        assert "income" in adult_profile.categorical_columns
        assert "gender" in adult_profile.categorical_columns

    def test_sf_labels_in_both_lists(self, sf_profile):
        """SF 'labels' is BINARY → appears in both numerical and categorical lists."""
        assert "labels" in sf_profile.numerical_columns
        assert "labels" in sf_profile.categorical_columns

    def test_id_like_not_in_repairable(self, bc_profile):
        for col in bc_profile.id_columns:
            assert col not in bc_profile.repairable_columns


# ═══════════════════════════════════════════════════════════════════════════════
# 7. summary_dict completeness
# ═══════════════════════════════════════════════════════════════════════════════


class TestSummaryDict:
    def test_all_required_keys_present(self, adult_profile, bc_profile, sf_profile):
        required = [
            "name", "clean_path", "dirty_path",
            "n_rows", "n_cols", "column_names",
            "total_missing_clean", "total_missing_dirty",
            "n_corrupted_cells", "corrupted_cells_by_column",
            "numerical_columns", "categorical_columns", "id_columns",
        ]
        for profile in (adult_profile, bc_profile, sf_profile):
            d = profile.summary_dict()
            for key in required:
                assert key in d, f"Missing key '{key}' in {profile.name}.summary_dict()"

    def test_summary_dict_json_serialisable(self, adult_profile):
        import json
        d = adult_profile.summary_dict()
        # Paths are strings, all other values are plain Python types
        try:
            json.dumps(d)
        except TypeError as e:
            pytest.fail(f"summary_dict() is not JSON-serialisable: {e}")
