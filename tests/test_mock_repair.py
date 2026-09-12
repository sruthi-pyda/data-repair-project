"""
tests/test_mock_repair.py
--------------------------
Comprehensive tests for Phase 4: Mock LLM Repair system.

Covers:
  - RepairCandidate / RepairContext / RepairResult dataclasses
  - build_repair_context (no clean_df leakage)
  - MockRepairer strategies (frequency, similarity, statistical, range, format, whitespace)
  - _validate_and_normalise: probabilities sum to 1, top_k enforced
  - Determinism: same cell → same result on every call
  - repair_batch: all records processed, results aligned
  - RepairEvaluator: top-1 accuracy, top-k accuracy, mean correct prob
  - Prompts: build_repair_prompt, parse_llm_response
  - Data leakage prevention: clean_df never passed to MockRepairer
"""

from __future__ import annotations

import math
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.data.schema import (
    ColumnKind,
    ColumnProfile,
    DatasetProfile,
    NumericalStats,
    CategoricalStats,
)
from src.detection.detector import DetectionRecord, ErrorType
from src.repair.mock_repair import (
    MockRepairer,
    RepairCandidate,
    RepairContext,
    RepairResult,
    _validate_and_normalise,
    build_repair_context,
)
from src.repair.evaluation import RepairEvaluator, RepairEvaluationResults
from src.repair.prompts import (
    build_repair_prompt,
    parse_llm_response,
    SYSTEM_PROMPT,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def num_stats() -> NumericalStats:
    return NumericalStats(
        mean=5.0, std=2.0, min=1.0, q25=3.0, q50=5.0, q75=7.0, max=10.0, iqr=4.0
    )


@pytest.fixture
def cat_stats() -> CategoricalStats:
    return CategoricalStats(
        value_counts={"Male": 30, "Female": 20},
        top_values=["Male", "Female"],
        n_rare=0,
    )


def _make_profile(
    name: str,
    columns: Dict[str, ColumnProfile],
    n_rows: int = 100,
) -> DatasetProfile:
    """Build a minimal DatasetProfile for testing."""
    from pathlib import Path
    profile = DatasetProfile(
        name=name,
        clean_path=Path(f"datasets/{name}/clean.csv"),
        dirty_path=Path(f"datasets/{name}/dirty.csv"),
        n_rows=n_rows,
        n_cols=len(columns),
        column_names=list(columns.keys()),
        columns=columns,
    )
    return profile


def _numerical_col_profile(
    name: str,
    stats: NumericalStats,
    dtype: str = "int64",
) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        kind=ColumnKind.NUMERICAL,
        dtype_clean=dtype,
        dtype_dirty=dtype,
        missing_clean=0,
        missing_dirty=0,
        n_unique_clean=10,
        n_unique_dirty=10,
        numerical_stats=stats,
    )


def _categorical_col_profile(
    name: str,
    cat_stats: CategoricalStats,
    dtype: str = "object",
) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        kind=ColumnKind.CATEGORICAL,
        dtype_clean=dtype,
        dtype_dirty=dtype,
        missing_clean=0,
        missing_dirty=0,
        n_unique_clean=2,
        n_unique_dirty=2,
        categorical_stats=cat_stats,
    )


def _binary_col_profile(name: str) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        kind=ColumnKind.BINARY,
        dtype_clean="object",
        dtype_dirty="object",
        missing_clean=0,
        missing_dirty=0,
        n_unique_clean=2,
        n_unique_dirty=2,
        categorical_stats=CategoricalStats(
            value_counts={"Male": 60, "Female": 40},
            top_values=["Male", "Female"],
            n_rare=0,
        ),
    )


@pytest.fixture
def simple_profile(num_stats, cat_stats) -> DatasetProfile:
    columns = {
        "age": _numerical_col_profile("age", num_stats),
        "gender": _binary_col_profile("gender"),
        "occupation": _categorical_col_profile("occupation", cat_stats),
    }
    return _make_profile("test_ds", columns)


@pytest.fixture
def simple_dirty_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25, None, 33, 4, 5],
        "gender": ["Male", None, "Female", "Male", "Female"],
        "occupation": ["Engineer", None, "Doctor", "Engineer", "Teacher"],
    })


@pytest.fixture
def simple_clean_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25, 5, 3, 4, 5],
        "gender": ["Male", "Female", "Female", "Male", "Female"],
        "occupation": ["Engineer", "Doctor", "Doctor", "Engineer", "Teacher"],
    })


def _make_detection_record(
    dataset: str = "test_ds",
    row_id: int = 1,
    column: str = "age",
    dirty_value: Any = None,
    error_type: str = ErrorType.MISSING.value,
    score: float = 1.0,
) -> DetectionRecord:
    return DetectionRecord(
        dataset=dataset,
        row_id=row_id,
        column=column,
        dirty_value=dirty_value,
        error_type=error_type,
        detection_score=score,
        detection_method="test",
        reason="test",
    )


@pytest.fixture
def repairer() -> MockRepairer:
    return MockRepairer(seed=42, top_k=3)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Dataclass contracts
# ═══════════════════════════════════════════════════════════════════════════════


class TestRepairCandidate:
    def test_to_dict_keys(self):
        rc = RepairCandidate(value="Male", probability=0.75)
        d = rc.to_dict()
        assert "value" in d and "probability" in d

    def test_to_dict_values(self):
        rc = RepairCandidate(value=5, probability=0.333333)
        d = rc.to_dict()
        assert d["value"] == 5
        assert d["probability"] == pytest.approx(0.3333, abs=1e-3)

    def test_none_value_allowed(self):
        rc = RepairCandidate(value=None, probability=0.5)
        assert rc.to_dict()["value"] is None


class TestRepairResult:
    def test_top_candidate_first(self):
        candidates = [
            RepairCandidate(value="A", probability=0.7),
            RepairCandidate(value="B", probability=0.3),
        ]
        res = RepairResult(
            dataset="ds", row_id=0, column="col", dirty_value="X",
            error_type="missing", candidates=candidates, repair_method="test",
        )
        assert res.top_candidate.value == "A"

    def test_top_candidate_empty(self):
        res = RepairResult(
            dataset="ds", row_id=0, column="col", dirty_value="X",
            error_type="missing", candidates=[], repair_method="test",
        )
        assert res.top_candidate.value is None
        assert res.top_candidate.probability == 0.0

    def test_to_dict_structure(self):
        res = RepairResult(
            dataset="ds", row_id=0, column="col", dirty_value="X",
            error_type="missing",
            candidates=[RepairCandidate("A", 1.0)],
            repair_method="freq",
        )
        d = res.to_dict()
        assert d["dataset"] == "ds"
        assert d["row_id"] == 0
        assert len(d["candidates"]) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: validate_and_normalise
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateAndNormalise:
    def test_probabilities_sum_to_one(self):
        raw = [("A", 0.6), ("B", 0.3), ("C", 0.1)]
        candidates = _validate_and_normalise(raw, top_k=3)
        total = sum(c.probability for c in candidates)
        assert math.isclose(total, 1.0, abs_tol=1e-5)

    def test_top_k_limit(self):
        raw = [("A", 0.5), ("B", 0.4), ("C", 0.3), ("D", 0.2)]
        candidates = _validate_and_normalise(raw, top_k=2)
        assert len(candidates) <= 2

    def test_top_k_sorted_descending(self):
        raw = [("A", 0.2), ("B", 0.8), ("C", 0.5)]
        candidates = _validate_and_normalise(raw, top_k=3)
        probs = [c.probability for c in candidates]
        assert probs == sorted(probs, reverse=True)

    def test_filters_negative_scores(self):
        raw = [("A", 0.5), ("B", -0.1), ("C", 0.3)]
        candidates = _validate_and_normalise(raw, top_k=3)
        assert all(c.probability > 0 for c in candidates)
        assert len(candidates) == 2

    def test_filters_zero_scores(self):
        raw = [("A", 0.0), ("B", 0.5)]
        candidates = _validate_and_normalise(raw, top_k=3)
        assert len(candidates) == 1

    def test_filters_nan_scores(self):
        raw = [("A", float("nan")), ("B", 0.5)]
        candidates = _validate_and_normalise(raw, top_k=3)
        assert len(candidates) == 1

    def test_empty_input(self):
        assert _validate_and_normalise([], top_k=3) == []

    def test_all_invalid(self):
        raw = [("A", -1.0), ("B", float("inf"))]
        # inf is filtered too
        result = _validate_and_normalise(raw, top_k=3)
        # inf > 0 so it won't be filtered by score <= 0, but isnan/isinf check catches it
        assert result == [] or all(math.isfinite(c.probability) for c in result)

    def test_single_candidate_prob_is_one(self):
        raw = [("A", 0.7)]
        candidates = _validate_and_normalise(raw, top_k=3)
        assert len(candidates) == 1
        assert math.isclose(candidates[0].probability, 1.0, abs_tol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: build_repair_context
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildRepairContext:
    def test_returns_repair_context(self, simple_profile, simple_dirty_df):
        rec = _make_detection_record(row_id=1, column="age", dirty_value=None)
        ctx = build_repair_context(rec, simple_dirty_df, simple_profile)
        assert isinstance(ctx, RepairContext)

    def test_no_clean_df_required(self, simple_profile, simple_dirty_df):
        """build_repair_context should work without clean_df."""
        rec = _make_detection_record(row_id=1, column="age", dirty_value=None)
        # Should not raise even though clean_df is not passed
        ctx = build_repair_context(rec, simple_dirty_df, simple_profile)
        assert ctx is not None

    def test_dirty_value_preserved(self, simple_profile, simple_dirty_df):
        rec = _make_detection_record(row_id=0, column="age", dirty_value=25)
        ctx = build_repair_context(rec, simple_dirty_df, simple_profile)
        assert ctx.dirty_value == 25

    def test_row_context_excludes_target_column(self, simple_profile, simple_dirty_df):
        rec = _make_detection_record(row_id=0, column="age", dirty_value=25)
        ctx = build_repair_context(rec, simple_dirty_df, simple_profile)
        assert "age" not in ctx.row_context

    def test_numerical_stats_populated(self, simple_profile, simple_dirty_df):
        rec = _make_detection_record(row_id=0, column="age", dirty_value=25)
        ctx = build_repair_context(rec, simple_dirty_df, simple_profile)
        assert ctx.column_stats is not None
        assert "median" in ctx.column_stats

    def test_categorical_vocabulary_not_empty(self, simple_profile, simple_dirty_df):
        rec = _make_detection_record(
            row_id=1, column="gender", dirty_value=None,
            error_type=ErrorType.MISSING.value
        )
        ctx = build_repair_context(rec, simple_dirty_df, simple_profile)
        assert len(ctx.vocabulary) > 0

    def test_numerical_vocabulary_empty(self, simple_profile, simple_dirty_df):
        rec = _make_detection_record(row_id=1, column="age", dirty_value=None)
        ctx = build_repair_context(rec, simple_dirty_df, simple_profile)
        assert ctx.vocabulary == []

    def test_kind_matches_profile(self, simple_profile, simple_dirty_df):
        rec = _make_detection_record(row_id=0, column="gender", dirty_value="Male")
        ctx = build_repair_context(rec, simple_dirty_df, simple_profile)
        assert ctx.kind == ColumnKind.BINARY

    def test_dataset_name_preserved(self, simple_profile, simple_dirty_df):
        rec = _make_detection_record(dataset="test_ds", row_id=0, column="age")
        ctx = build_repair_context(rec, simple_dirty_df, simple_profile)
        assert ctx.dataset == "test_ds"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: MockRepairer strategies
# ═══════════════════════════════════════════════════════════════════════════════


class TestFrequencyRepair:
    def test_returns_candidates(self, repairer, simple_profile, simple_dirty_df):
        rec = _make_detection_record(
            row_id=1, column="gender", dirty_value=None,
            error_type=ErrorType.MISSING.value
        )
        ctx = build_repair_context(rec, simple_dirty_df, simple_profile)
        result = repairer.repair(ctx)
        assert len(result.candidates) > 0

    def test_candidates_in_vocabulary(self, repairer, simple_profile, simple_dirty_df):
        rec = _make_detection_record(
            row_id=1, column="gender", dirty_value=None,
            error_type=ErrorType.MISSING.value
        )
        ctx = build_repair_context(rec, simple_dirty_df, simple_profile)
        result = repairer.repair(ctx)
        vocab = set(ctx.vocabulary)
        for c in result.candidates:
            assert c.value in vocab

    def test_probabilities_sum_to_one(self, repairer, simple_profile, simple_dirty_df):
        rec = _make_detection_record(
            row_id=1, column="gender", dirty_value=None,
            error_type=ErrorType.MISSING.value
        )
        ctx = build_repair_context(rec, simple_dirty_df, simple_profile)
        result = repairer.repair(ctx)
        total = sum(c.probability for c in result.candidates)
        assert math.isclose(total, 1.0, abs_tol=1e-4)


class TestStatisticalRepair:
    def test_numerical_missing_returns_candidates(self, repairer, simple_profile, simple_dirty_df):
        rec = _make_detection_record(
            row_id=1, column="age", dirty_value=None,
            error_type=ErrorType.MISSING.value
        )
        ctx = build_repair_context(rec, simple_dirty_df, simple_profile)
        result = repairer.repair(ctx)
        assert len(result.candidates) > 0

    def test_median_in_candidates(self, repairer, simple_profile, simple_dirty_df, num_stats):
        rec = _make_detection_record(
            row_id=1, column="age", dirty_value=None,
            error_type=ErrorType.MISSING.value
        )
        ctx = build_repair_context(rec, simple_dirty_df, simple_profile)
        result = repairer.repair(ctx)
        # Median is 5 (int), should be in candidates
        candidate_values = [c.value for c in result.candidates]
        # The median (int-rounded) should appear
        assert int(round(num_stats.q50)) in candidate_values

    def test_probabilities_positive(self, repairer, simple_profile, simple_dirty_df):
        rec = _make_detection_record(
            row_id=1, column="age", dirty_value=None,
            error_type=ErrorType.MISSING.value
        )
        ctx = build_repair_context(rec, simple_dirty_df, simple_profile)
        result = repairer.repair(ctx)
        assert all(c.probability > 0 for c in result.candidates)


class TestSimilarityRepair:
    def test_novel_category_returns_candidates(self, repairer, simple_profile, simple_dirty_df):
        rec = _make_detection_record(
            row_id=0, column="gender", dirty_value="Mael",
            error_type=ErrorType.NOVEL_CATEGORY.value
        )
        ctx = build_repair_context(rec, simple_dirty_df, simple_profile)
        result = repairer.repair(ctx)
        assert len(result.candidates) > 0

    def test_top_candidate_is_similar(self, repairer, simple_profile, simple_dirty_df):
        rec = _make_detection_record(
            row_id=0, column="gender", dirty_value="Mael",
            error_type=ErrorType.NOVEL_CATEGORY.value
        )
        ctx = build_repair_context(rec, simple_dirty_df, simple_profile)
        result = repairer.repair(ctx)
        # "Mael" is closest to "Male"
        assert result.top_candidate.value == "Male"

    def test_probabilities_sum_to_one(self, repairer, simple_profile, simple_dirty_df):
        rec = _make_detection_record(
            row_id=0, column="gender", dirty_value="Fmale",
            error_type=ErrorType.NOVEL_CATEGORY.value
        )
        ctx = build_repair_context(rec, simple_dirty_df, simple_profile)
        result = repairer.repair(ctx)
        total = sum(c.probability for c in result.candidates)
        assert math.isclose(total, 1.0, abs_tol=1e-4)


class TestRangeNormalisationRepair:
    def test_digit_repeat_corrected(self, repairer, num_stats):
        """33 should be corrected to 3 (33/11=3) via digit-repeat heuristic."""
        col_prof = _numerical_col_profile("val", num_stats)
        profile = _make_profile("ds", {"val": col_prof})
        dirty_df = pd.DataFrame({"val": [33]})
        rec = _make_detection_record(
            row_id=0, column="val", dirty_value=33,
            error_type=ErrorType.OUT_OF_RANGE.value
        )
        ctx = build_repair_context(rec, dirty_df, profile)
        result = repairer.repair(ctx)
        candidate_values = [c.value for c in result.candidates]
        # 33/11 = 3, which is in [1, 10]
        assert 3 in candidate_values

    def test_out_of_range_returns_candidates(self, repairer, num_stats):
        col_prof = _numerical_col_profile("val", num_stats)
        profile = _make_profile("ds", {"val": col_prof})
        dirty_df = pd.DataFrame({"val": [99]})
        rec = _make_detection_record(
            row_id=0, column="val", dirty_value=99,
            error_type=ErrorType.OUT_OF_RANGE.value
        )
        ctx = build_repair_context(rec, dirty_df, profile)
        result = repairer.repair(ctx)
        assert len(result.candidates) > 0

    def test_probabilities_sum_to_one(self, repairer, num_stats):
        col_prof = _numerical_col_profile("val", num_stats)
        profile = _make_profile("ds", {"val": col_prof})
        dirty_df = pd.DataFrame({"val": [88]})
        rec = _make_detection_record(
            row_id=0, column="val", dirty_value=88,
            error_type=ErrorType.OUT_OF_RANGE.value
        )
        ctx = build_repair_context(rec, dirty_df, profile)
        result = repairer.repair(ctx)
        total = sum(c.probability for c in result.candidates)
        assert math.isclose(total, 1.0, abs_tol=1e-4)


class TestFormatRepair:
    def test_format_numerical_extracts_digits(self, repairer, num_stats):
        """'1a0' → extract '10' which is in range [1, 10]. Should be candidate."""
        col_prof = _numerical_col_profile("val", num_stats)
        profile = _make_profile("ds", {"val": col_prof})
        dirty_df = pd.DataFrame({"val": ["1a0"]})
        rec = _make_detection_record(
            row_id=0, column="val", dirty_value="1a0",
            error_type=ErrorType.FORMAT_ERROR.value
        )
        ctx = build_repair_context(rec, dirty_df, profile)
        result = repairer.repair(ctx)
        assert len(result.candidates) > 0

    def test_format_categorical_returns_candidates(self, repairer, cat_stats):
        col_prof = _categorical_col_profile("occ", cat_stats)
        profile = _make_profile("ds", {"occ": col_prof})
        dirty_df = pd.DataFrame({"occ": ["Engineerr", "Doctor"]})
        rec = _make_detection_record(
            row_id=0, column="occ", dirty_value="Engineerr",
            error_type=ErrorType.FORMAT_ERROR.value
        )
        ctx = build_repair_context(rec, dirty_df, profile)
        result = repairer.repair(ctx)
        assert len(result.candidates) > 0

    def test_format_probabilities_sum_to_one(self, repairer, num_stats):
        col_prof = _numerical_col_profile("val", num_stats)
        profile = _make_profile("ds", {"val": col_prof})
        dirty_df = pd.DataFrame({"val": ["a10"]})
        rec = _make_detection_record(
            row_id=0, column="val", dirty_value="a10",
            error_type=ErrorType.FORMAT_ERROR.value
        )
        ctx = build_repair_context(rec, dirty_df, profile)
        result = repairer.repair(ctx)
        total = sum(c.probability for c in result.candidates)
        assert math.isclose(total, 1.0, abs_tol=1e-4)


class TestWhitespaceRepair:
    def test_stripped_value_is_top_candidate(self, repairer, cat_stats):
        col_prof = _categorical_col_profile("occ", cat_stats)
        profile = _make_profile("ds", {"occ": col_prof})
        dirty_df = pd.DataFrame({"occ": ["  Male  ", "Female"]})
        rec = _make_detection_record(
            row_id=0, column="occ", dirty_value="  Male  ",
            error_type=ErrorType.WHITESPACE.value
        )
        ctx = build_repair_context(rec, dirty_df, profile)
        result = repairer.repair(ctx)
        assert result.top_candidate.value == "Male"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Determinism
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    def test_same_cell_same_result(self, simple_profile, simple_dirty_df):
        rec = _make_detection_record(row_id=1, column="age", dirty_value=None)
        r1 = MockRepairer(seed=42)
        r2 = MockRepairer(seed=42)
        ctx = build_repair_context(rec, simple_dirty_df, simple_profile)
        res1 = r1.repair(ctx)
        res2 = r2.repair(ctx)
        assert res1.candidates[0].value == res2.candidates[0].value
        assert res1.candidates[0].probability == res2.candidates[0].probability

    def test_different_seeds_may_differ(self, simple_profile, simple_dirty_df):
        rec = _make_detection_record(row_id=1, column="gender", dirty_value=None,
                                     error_type=ErrorType.MISSING.value)
        ctx = build_repair_context(rec, simple_dirty_df, simple_profile)
        r1 = MockRepairer(seed=42)
        r2 = MockRepairer(seed=999)
        res1 = r1.repair(ctx)
        res2 = r2.repair(ctx)
        # Probabilities will differ (noise is seeded differently)
        prob1 = [c.probability for c in res1.candidates]
        prob2 = [c.probability for c in res2.candidates]
        assert prob1 != prob2

    def test_different_cells_different_seeds(self, simple_profile, simple_dirty_df):
        """Two different cells should get different effective seeds."""
        rec1 = _make_detection_record(row_id=0, column="age", dirty_value=25)
        rec2 = _make_detection_record(row_id=1, column="age", dirty_value=None)
        r = MockRepairer(seed=42)
        ctx1 = build_repair_context(rec1, simple_dirty_df, simple_profile)
        ctx2 = build_repair_context(rec2, simple_dirty_df, simple_profile)
        rng1 = r._make_rng(ctx1)
        rng2 = r._make_rng(ctx2)
        # The two RNGs should produce different first values
        v1 = rng1.random()
        v2 = rng2.random()
        assert v1 != v2

    def test_repeated_batch_identical(self, simple_profile, simple_dirty_df):
        records = [
            _make_detection_record(row_id=1, column="age", dirty_value=None),
            _make_detection_record(row_id=1, column="gender", dirty_value=None,
                                   error_type=ErrorType.MISSING.value),
        ]
        r1 = MockRepairer(seed=42)
        r2 = MockRepairer(seed=42)
        results1 = r1.repair_batch(records, simple_dirty_df, simple_profile)
        results2 = r2.repair_batch(records, simple_dirty_df, simple_profile)
        for a, b in zip(results1, results2):
            assert a.top_candidate.value == b.top_candidate.value
            assert a.top_candidate.probability == b.top_candidate.probability


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: repair_batch
# ═══════════════════════════════════════════════════════════════════════════════


class TestRepairBatch:
    def test_length_matches_records(self, repairer, simple_profile, simple_dirty_df):
        records = [
            _make_detection_record(row_id=1, column="age", dirty_value=None),
            _make_detection_record(row_id=1, column="gender", dirty_value=None,
                                   error_type=ErrorType.MISSING.value),
        ]
        results = repairer.repair_batch(records, simple_dirty_df, simple_profile)
        assert len(results) == 2

    def test_each_result_has_candidates(self, repairer, simple_profile, simple_dirty_df):
        records = [
            _make_detection_record(row_id=1, column="age", dirty_value=None),
        ]
        results = repairer.repair_batch(records, simple_dirty_df, simple_profile)
        assert len(results[0].candidates) > 0

    def test_empty_records(self, repairer, simple_profile, simple_dirty_df):
        results = repairer.repair_batch([], simple_dirty_df, simple_profile)
        assert results == []

    def test_result_fields_populated(self, repairer, simple_profile, simple_dirty_df):
        records = [
            _make_detection_record(
                dataset="test_ds", row_id=1, column="age", dirty_value=None
            ),
        ]
        results = repairer.repair_batch(records, simple_dirty_df, simple_profile)
        res = results[0]
        assert res.dataset == "test_ds"
        assert res.row_id == 1
        assert res.column == "age"
        assert res.repair_method != ""


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: top_k enforcement
# ═══════════════════════════════════════════════════════════════════════════════


class TestTopKEnforcement:
    def test_top_1_returns_at_most_one(self, simple_profile, simple_dirty_df):
        repairer = MockRepairer(seed=42, top_k=1)
        rec = _make_detection_record(row_id=1, column="gender", dirty_value=None,
                                     error_type=ErrorType.MISSING.value)
        ctx = build_repair_context(rec, simple_dirty_df, simple_profile)
        result = repairer.repair(ctx)
        assert len(result.candidates) <= 1

    def test_top_3_returns_at_most_three(self, simple_profile, simple_dirty_df):
        repairer = MockRepairer(seed=42, top_k=3)
        rec = _make_detection_record(row_id=1, column="gender", dirty_value=None,
                                     error_type=ErrorType.MISSING.value)
        ctx = build_repair_context(rec, simple_dirty_df, simple_profile)
        result = repairer.repair(ctx)
        assert len(result.candidates) <= 3


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8: Data leakage prevention
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataLeakagePrevention:
    def test_repair_does_not_accept_clean_df(self, repairer, simple_profile, simple_dirty_df):
        """MockRepairer.repair takes RepairContext, not clean_df."""
        rec = _make_detection_record(row_id=1, column="age", dirty_value=None)
        ctx = build_repair_context(rec, simple_dirty_df, simple_profile)
        # repair() signature: (context: RepairContext) -> RepairResult
        import inspect
        sig = inspect.signature(repairer.repair)
        params = list(sig.parameters.keys())
        assert "clean_df" not in params
        assert "clean" not in " ".join(params)

    def test_repair_batch_does_not_accept_clean_df(self, repairer):
        import inspect
        sig = inspect.signature(repairer.repair_batch)
        params = list(sig.parameters.keys())
        assert "clean_df" not in params
        assert "clean" not in " ".join(params)

    def test_build_repair_context_does_not_accept_clean_df(self):
        import inspect
        sig = inspect.signature(build_repair_context)
        params = list(sig.parameters.keys())
        assert "clean_df" not in params
        assert "clean" not in " ".join(params)

    def test_repair_context_has_no_clean_attribute(self):
        """RepairContext should not have a clean_value or clean_df field."""
        ctx_fields = [f.name for f in RepairContext.__dataclass_fields__.values()]
        for f in ctx_fields:
            assert "clean" not in f.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 9: RepairEvaluator
# ═══════════════════════════════════════════════════════════════════════════════


class TestRepairEvaluator:
    def _make_results(self, column: str, dirty_value: Any, candidates_vals: List[Any]) -> List[RepairResult]:
        n = len(candidates_vals)
        if n == 0:
            candidates = []
        else:
            prob_each = 1.0 / n
            candidates = [RepairCandidate(v, round(prob_each, 6)) for v in candidates_vals]
        return [
            RepairResult(
                dataset="test", row_id=0, column=column,
                dirty_value=dirty_value, error_type="missing",
                candidates=candidates, repair_method="test",
            )
        ]

    def test_perfect_top1(self, simple_clean_df, simple_dirty_df):
        """When top candidate matches clean, top-1 accuracy = 1.0."""
        evaluator = RepairEvaluator(simple_clean_df, simple_dirty_df, "test")
        # Row 0, age: clean=25, dirty=25 (same)
        results = self._make_results("age", 25, [25, 5, 3])
        res = evaluator.evaluate(results)
        assert res.top1_accuracy == 1.0

    def test_topk_correct_not_top1(self, simple_clean_df, simple_dirty_df):
        """Correct value in position 2 → top-k correct, top-1 wrong."""
        evaluator = RepairEvaluator(simple_clean_df, simple_dirty_df, "test")
        # Row 0, age: clean=25, top candidate is 99 (wrong), but 25 is in candidates
        results = self._make_results("age", None, [99, 25])
        res = evaluator.evaluate(results)
        assert res.top1_accuracy == 0.0
        assert res.topk_accuracy == 1.0

    def test_all_wrong(self, simple_clean_df, simple_dirty_df):
        evaluator = RepairEvaluator(simple_clean_df, simple_dirty_df, "test")
        results = self._make_results("age", None, [999, 888])
        res = evaluator.evaluate(results)
        assert res.top1_accuracy == 0.0
        assert res.topk_accuracy == 0.0

    def test_empty_results(self, simple_clean_df, simple_dirty_df):
        evaluator = RepairEvaluator(simple_clean_df, simple_dirty_df, "test")
        res = evaluator.evaluate([])
        assert res.n_repaired == 0
        assert res.top1_accuracy == 0.0

    def test_n_repaired_count(self, simple_clean_df, simple_dirty_df):
        evaluator = RepairEvaluator(simple_clean_df, simple_dirty_df, "test")
        results = (
            self._make_results("age", None, [25])
            + self._make_results("age", None, [5])
        )
        res = evaluator.evaluate(results)
        assert res.n_repaired == 2

    def test_mean_correct_prob_correct_found(self, simple_clean_df, simple_dirty_df):
        """When correct value is the only candidate, mean_correct_prob should be 1.0."""
        evaluator = RepairEvaluator(simple_clean_df, simple_dirty_df, "test")
        results = self._make_results("age", None, [25])  # row 0, clean age=25
        res = evaluator.evaluate(results)
        assert math.isclose(res.mean_correct_prob, 1.0, abs_tol=1e-4)

    def test_by_error_type_populated(self, simple_clean_df, simple_dirty_df):
        evaluator = RepairEvaluator(simple_clean_df, simple_dirty_df, "test")
        res_list = [
            RepairResult(
                dataset="test", row_id=0, column="age",
                dirty_value=None, error_type="missing",
                candidates=[RepairCandidate(25, 1.0)],
                repair_method="statistical_repair",
            )
        ]
        res = evaluator.evaluate(res_list)
        assert "missing" in res.by_error_type

    def test_by_column_populated(self, simple_clean_df, simple_dirty_df):
        evaluator = RepairEvaluator(simple_clean_df, simple_dirty_df, "test")
        res_list = [
            RepairResult(
                dataset="test", row_id=0, column="age",
                dirty_value=None, error_type="missing",
                candidates=[RepairCandidate(25, 1.0)],
                repair_method="statistical_repair",
            )
        ]
        res = evaluator.evaluate(res_list)
        assert "age" in res.by_column

    def test_numeric_comparison_int_vs_float(self, simple_clean_df, simple_dirty_df):
        """Candidate 25.0 should match clean value 25 (int)."""
        evaluator = RepairEvaluator(simple_clean_df, simple_dirty_df, "test")
        results = self._make_results("age", None, [25.0])
        res = evaluator.evaluate(results)
        assert res.top1_accuracy == 1.0

    def test_print_report_runs_without_error(self, simple_clean_df, simple_dirty_df, capsys):
        evaluator = RepairEvaluator(simple_clean_df, simple_dirty_df, "test_ds")
        res_list = [
            RepairResult(
                dataset="test_ds", row_id=0, column="age",
                dirty_value=None, error_type="missing",
                candidates=[RepairCandidate(25, 1.0)],
                repair_method="statistical_repair",
            )
        ]
        res = evaluator.evaluate(res_list)
        RepairEvaluator.print_report(res)
        captured = capsys.readouterr()
        assert "REPAIR EVALUATION" in captured.out
        assert "TEST_DS" in captured.out


# ═══════════════════════════════════════════════════════════════════════════════
# Section 10: Prompts
# ═══════════════════════════════════════════════════════════════════════════════


class TestPrompts:
    def _make_context(self) -> RepairContext:
        return RepairContext(
            dataset="breast_cancer",
            column="Bare Nuclei",
            kind=ColumnKind.MIXED,
            dtype="object",
            dirty_value="33",
            error_type=ErrorType.OUT_OF_RANGE.value,
            row_id=10,
            row_context={"Clump Thickness": 5, "Cell Size": 3},
            vocabulary=["1", "2", "3", "4", "5", "10", "?"],
            column_stats={"min": 1.0, "max": 10.0, "median": 3.0, "q25": 1.0, "q75": 5.0, "iqr": 4.0},
        )

    def test_build_repair_prompt_returns_string(self):
        ctx = self._make_context()
        prompt = build_repair_prompt(ctx)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_prompt_contains_dirty_value(self):
        ctx = self._make_context()
        prompt = build_repair_prompt(ctx)
        assert "33" in prompt

    def test_prompt_contains_column_name(self):
        ctx = self._make_context()
        prompt = build_repair_prompt(ctx)
        assert "Bare Nuclei" in prompt

    def test_prompt_contains_error_type(self):
        ctx = self._make_context()
        prompt = build_repair_prompt(ctx)
        assert "OUT_OF_RANGE" in prompt or "out_of_range" in prompt

    def test_prompt_contains_vocabulary(self):
        ctx = self._make_context()
        prompt = build_repair_prompt(ctx)
        assert "1" in prompt  # part of vocab

    def test_prompt_contains_stats(self):
        ctx = self._make_context()
        prompt = build_repair_prompt(ctx)
        assert "median" in prompt

    def test_prompt_contains_row_context(self):
        ctx = self._make_context()
        prompt = build_repair_prompt(ctx)
        assert "Clump Thickness" in prompt

    def test_system_prompt_is_string(self):
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 100

    def test_parse_valid_json_response(self):
        ctx = self._make_context()
        response = '{"candidates": [{"value": 3, "probability": 0.8}, {"value": 5, "probability": 0.2}], "reasoning": "digit repeat"}'
        candidates = parse_llm_response(response, ctx)
        assert len(candidates) == 2
        total = sum(c.probability for c in candidates)
        assert math.isclose(total, 1.0, abs_tol=1e-4)

    def test_parse_json_in_code_block(self):
        ctx = self._make_context()
        response = '```json\n{"candidates": [{"value": 3, "probability": 1.0}], "reasoning": "test"}\n```'
        candidates = parse_llm_response(response, ctx)
        assert len(candidates) == 1

    def test_parse_invalid_json_returns_empty(self):
        ctx = self._make_context()
        candidates = parse_llm_response("not json at all", ctx)
        assert candidates == []

    def test_parse_normalises_probabilities(self):
        ctx = self._make_context()
        # Probabilities 0.5 and 0.5 → already sum to 1
        response = '{"candidates": [{"value": 3, "probability": 0.5}, {"value": 5, "probability": 0.5}]}'
        candidates = parse_llm_response(response, ctx)
        total = sum(c.probability for c in candidates)
        assert math.isclose(total, 1.0, abs_tol=1e-5)

    def test_parse_filters_invalid_probs(self):
        ctx = self._make_context()
        response = '{"candidates": [{"value": 3, "probability": -0.1}, {"value": 5, "probability": 0.9}]}'
        candidates = parse_llm_response(response, ctx)
        # Negative prob filtered, only 1 remains
        assert len(candidates) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Section 11: Integration — real datasets (smoke tests)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestIntegrationBreastCancer:
    """Run mock repair on a few real BC records (requires datasets/ directory)."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        from pathlib import Path
        if not Path("datasets/breast_cancer/dirty.csv").exists():
            pytest.skip("datasets/breast_cancer not found")
        from src.data.loader import DataLoader
        from src.data.profiler import profile_dataset
        from src.detection.detector import Detector

        loader = DataLoader("datasets")
        self.clean_df, self.dirty_df = loader.load("breast_cancer")
        self.profile = profile_dataset(
            "breast_cancer",
            loader.discover("breast_cancer"),
            self.clean_df,
            self.dirty_df,
        )
        detector = Detector(self.profile)
        self.records = detector.detect(self.dirty_df)
        self.repairer = MockRepairer(seed=42, top_k=3)

    def test_batch_runs_without_error(self):
        results = self.repairer.repair_batch(
            self.records[:20], self.dirty_df, self.profile
        )
        assert len(results) == min(20, len(self.records))

    def test_all_results_have_candidates(self):
        results = self.repairer.repair_batch(
            self.records[:20], self.dirty_df, self.profile
        )
        for res in results:
            assert len(res.candidates) > 0, f"No candidates for {res.column} row={res.row_id}"

    def test_probabilities_sum_to_one(self):
        results = self.repairer.repair_batch(
            self.records[:20], self.dirty_df, self.profile
        )
        for res in results:
            total = sum(c.probability for c in res.candidates)
            assert math.isclose(total, 1.0, abs_tol=1e-4), \
                f"Probs don't sum to 1 for {res.column} row={res.row_id}: {total}"

    def test_determinism_batch(self):
        results1 = self.repairer.repair_batch(
            self.records[:10], self.dirty_df, self.profile
        )
        results2 = MockRepairer(seed=42, top_k=3).repair_batch(
            self.records[:10], self.dirty_df, self.profile
        )
        for a, b in zip(results1, results2):
            assert a.top_candidate.value == b.top_candidate.value
            assert a.top_candidate.probability == b.top_candidate.probability

    def test_evaluator_runs(self):
        results = self.repairer.repair_batch(
            self.records, self.dirty_df, self.profile
        )
        evaluator = RepairEvaluator(self.clean_df, self.dirty_df, "breast_cancer")
        eval_res = evaluator.evaluate(results)
        assert eval_res.n_repaired == len(results)
        assert 0.0 <= eval_res.top1_accuracy <= 1.0
        assert eval_res.topk_accuracy >= eval_res.top1_accuracy

    def test_evaluator_top_k_ge_top1(self):
        results = self.repairer.repair_batch(
            self.records, self.dirty_df, self.profile
        )
        evaluator = RepairEvaluator(self.clean_df, self.dirty_df, "breast_cancer")
        eval_res = evaluator.evaluate(results)
        assert eval_res.topk_accuracy >= eval_res.top1_accuracy


@pytest.mark.integration
class TestIntegrationSmartFactory:
    @pytest.fixture(autouse=True)
    def load_data(self):
        from pathlib import Path
        if not Path("datasets/smartfactory/dirty.csv").exists():
            pytest.skip("datasets/smartfactory not found")
        from src.data.loader import DataLoader
        from src.data.profiler import profile_dataset
        from src.detection.detector import Detector

        loader = DataLoader("datasets")
        self.clean_df, self.dirty_df = loader.load("smartfactory")
        self.profile = profile_dataset(
            "smartfactory",
            loader.discover("smartfactory"),
            self.clean_df,
            self.dirty_df,
        )
        detector = Detector(self.profile, flag_statistical_outliers=False)
        self.records = detector.detect(self.dirty_df)
        self.repairer = MockRepairer(seed=42, top_k=3)

    def test_all_sf_records_are_missing(self):
        """SF corruption is missing-only; all repair methods should be statistical."""
        results = self.repairer.repair_batch(
            self.records[:20], self.dirty_df, self.profile
        )
        for res in results:
            assert res.error_type.upper() == "MISSING", f"Expected MISSING, got {res.error_type}"

    def test_all_results_have_candidates(self):
        results = self.repairer.repair_batch(
            self.records[:20], self.dirty_df, self.profile
        )
        for res in results:
            assert len(res.candidates) > 0
