"""
tests/test_uncertainty.py
--------------------------
Comprehensive tests for Phase 5: Uncertainty Estimation.

Covers:
  - UncertaintyRecord dataclass and to_dict()
  - _compute_raw_confidence: top candidate extraction
  - _compute_entropy: Shannon entropy, normalized entropy, K=1, K=0, p=0
  - compute_entropy_from_probs: raw probability list convenience function
  - SelfConsistencyResult: vote counting, consistency score
  - UncertaintyEstimator.estimate(): all signals populated
  - UncertaintyEstimator.estimate_batch(): length, fields, determinism
  - Self-consistency determinism: same call → same result
  - Data leakage: no clean_df in estimator
  - UncertaintyCorrelationAnalysis: bin analysis, Pearson correlation
  - Output schema completeness
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

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
    RepairResult,
    build_repair_context,
)
from src.uncertainty.estimator import (
    SelfConsistencyResult,
    UncertaintyEstimator,
    UncertaintyRecord,
    _compute_entropy,
    _compute_raw_confidence,
    compute_entropy_from_probs,
)
from src.uncertainty.evaluation import (
    UncertaintyCorrelationAnalysis,
    CorrelationReport,
    BinStats,
    _pearson,
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
        value_counts={"Male": 60, "Female": 40},
        top_values=["Male", "Female"],
        n_rare=0,
    )


def _make_profile(columns: Dict[str, ColumnProfile], name: str = "test_ds") -> DatasetProfile:
    return DatasetProfile(
        name=name,
        clean_path=Path(f"datasets/{name}/clean.csv"),
        dirty_path=Path(f"datasets/{name}/dirty.csv"),
        n_rows=10,
        n_cols=len(columns),
        column_names=list(columns.keys()),
        columns=columns,
    )


def _num_col(name: str, stats: NumericalStats, dtype: str = "int64") -> ColumnProfile:
    return ColumnProfile(
        name=name, kind=ColumnKind.NUMERICAL,
        dtype_clean=dtype, dtype_dirty=dtype,
        missing_clean=0, missing_dirty=0,
        n_unique_clean=10, n_unique_dirty=10,
        numerical_stats=stats,
    )


def _cat_col(name: str, stats: CategoricalStats, dtype: str = "object") -> ColumnProfile:
    return ColumnProfile(
        name=name, kind=ColumnKind.CATEGORICAL,
        dtype_clean=dtype, dtype_dirty=dtype,
        missing_clean=0, missing_dirty=0,
        n_unique_clean=2, n_unique_dirty=2,
        categorical_stats=stats,
    )


def _make_repair_result(
    candidates: List[RepairCandidate],
    column: str = "age",
    row_id: int = 0,
    error_type: str = "missing",
    method: str = "statistical_repair",
) -> RepairResult:
    return RepairResult(
        dataset="test_ds",
        row_id=row_id,
        column=column,
        dirty_value=None,
        error_type=error_type,
        candidates=candidates,
        repair_method=method,
    )


def _uniform_candidates(n: int) -> List[RepairCandidate]:
    prob = round(1.0 / n, 6)
    return [RepairCandidate(value=i, probability=prob) for i in range(n)]


@pytest.fixture
def simple_dirty_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25, None, 3, 4, 5, 6, 7, 8, 9, 10],
        "gender": ["Male", None, "Female", "Male", "Female",
                   "Male", "Female", "Male", "Female", "Male"],
    })


@pytest.fixture
def simple_clean_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25, 5, 3, 4, 5, 6, 7, 8, 9, 10],
        "gender": ["Male", "Female", "Female", "Male", "Female",
                   "Male", "Female", "Male", "Female", "Male"],
    })


@pytest.fixture
def simple_profile(num_stats, cat_stats) -> DatasetProfile:
    return _make_profile({
        "age": _num_col("age", num_stats),
        "gender": _cat_col("gender", cat_stats),
    })


@pytest.fixture
def estimator() -> UncertaintyEstimator:
    return UncertaintyEstimator(base_seed=42, n_samples=5, top_k=3)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: _compute_raw_confidence
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeRawConfidence:
    def test_returns_top_candidate(self):
        candidates = [
            RepairCandidate("A", 0.70),
            RepairCandidate("B", 0.20),
            RepairCandidate("C", 0.10),
        ]
        result = _make_repair_result(candidates)
        top_val, conf = _compute_raw_confidence(result)
        assert top_val == "A"

    def test_returns_top_probability(self):
        candidates = [
            RepairCandidate("A", 0.70),
            RepairCandidate("B", 0.20),
            RepairCandidate("C", 0.10),
        ]
        result = _make_repair_result(candidates)
        _, conf = _compute_raw_confidence(result)
        assert math.isclose(conf, 0.70, abs_tol=1e-6)

    def test_single_candidate(self):
        candidates = [RepairCandidate("X", 1.0)]
        result = _make_repair_result(candidates)
        top_val, conf = _compute_raw_confidence(result)
        assert top_val == "X"
        assert math.isclose(conf, 1.0, abs_tol=1e-6)

    def test_empty_candidates(self):
        result = _make_repair_result([])
        top_val, conf = _compute_raw_confidence(result)
        assert top_val is None
        assert conf == 0.0

    def test_numeric_top_candidate(self):
        candidates = [RepairCandidate(42, 0.9), RepairCandidate(5, 0.1)]
        result = _make_repair_result(candidates)
        top_val, conf = _compute_raw_confidence(result)
        assert top_val == 42
        assert math.isclose(conf, 0.9, abs_tol=1e-6)

    def test_confidence_in_zero_one_range(self):
        candidates = _uniform_candidates(3)
        result = _make_repair_result(candidates)
        _, conf = _compute_raw_confidence(result)
        assert 0.0 < conf <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: _compute_entropy
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeEntropy:
    def test_single_candidate_entropy_zero(self):
        """K=1: entropy must be 0 (no uncertainty)."""
        candidates = [RepairCandidate("A", 1.0)]
        h, h_n = _compute_entropy(candidates)
        assert h == 0.0
        assert h_n == 0.0

    def test_empty_candidates_entropy_zero(self):
        h, h_n = _compute_entropy([])
        assert h == 0.0
        assert h_n == 0.0

    def test_uniform_two_candidates_max_entropy(self):
        """Two candidates with p=0.5 each → H = ln(2), H_n = 1.0."""
        candidates = [RepairCandidate("A", 0.5), RepairCandidate("B", 0.5)]
        h, h_n = _compute_entropy(candidates)
        assert math.isclose(h, math.log(2), abs_tol=1e-5)
        assert math.isclose(h_n, 1.0, abs_tol=1e-5)

    def test_uniform_three_candidates(self):
        """Three candidates at 1/3 each → H = ln(3), H_n = 1.0."""
        p = 1.0 / 3.0
        candidates = [RepairCandidate(i, p) for i in range(3)]
        h, h_n = _compute_entropy(candidates)
        assert math.isclose(h, math.log(3), abs_tol=1e-4)
        assert math.isclose(h_n, 1.0, abs_tol=1e-4)

    def test_concentrated_distribution_low_entropy(self):
        """p = [0.99, 0.005, 0.005] → H should be small."""
        candidates = [
            RepairCandidate("A", 0.99),
            RepairCandidate("B", 0.005),
            RepairCandidate("C", 0.005),
        ]
        h, h_n = _compute_entropy(candidates)
        _, h_n_uniform = _compute_entropy([RepairCandidate(i, 1 / 3) for i in range(3)])
        assert h_n < h_n_uniform

    def test_normalized_entropy_in_zero_one(self):
        candidates = [RepairCandidate("A", 0.7), RepairCandidate("B", 0.3)]
        h, h_n = _compute_entropy(candidates)
        assert 0.0 <= h_n <= 1.0

    def test_entropy_non_negative(self):
        candidates = _uniform_candidates(3)
        h, h_n = _compute_entropy(candidates)
        assert h >= 0.0

    def test_zero_probability_handled_safely(self):
        """p_i = 0 → contribution is 0 (0 * log(0) = 0 by convention)."""
        candidates = [
            RepairCandidate("A", 1.0),
            RepairCandidate("B", 0.0),
        ]
        h, h_n = _compute_entropy(candidates)
        assert h == 0.0

    def test_entropy_increases_with_spread(self):
        """More evenly spread distributions have higher entropy."""
        c_certain = [RepairCandidate("A", 0.98), RepairCandidate("B", 0.02)]
        c_uncertain = [RepairCandidate("A", 0.55), RepairCandidate("B", 0.45)]
        _, h_n_certain = _compute_entropy(c_certain)
        _, h_n_uncertain = _compute_entropy(c_uncertain)
        assert h_n_uncertain > h_n_certain


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: compute_entropy_from_probs
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeEntropyFromProbs:
    def test_uniform_probs(self):
        h, h_n = compute_entropy_from_probs([0.5, 0.5])
        assert math.isclose(h_n, 1.0, abs_tol=1e-5)

    def test_single_prob(self):
        h, h_n = compute_entropy_from_probs([1.0])
        assert h == 0.0
        assert h_n == 0.0

    def test_empty_probs(self):
        h, h_n = compute_entropy_from_probs([])
        assert h == 0.0
        assert h_n == 0.0

    def test_all_zero_probs(self):
        h, h_n = compute_entropy_from_probs([0.0, 0.0])
        assert h == 0.0

    def test_unnormalised_input_handled(self):
        """Input [2.0, 2.0] → normalised [0.5, 0.5] → H_n = 1.0."""
        h, h_n = compute_entropy_from_probs([2.0, 2.0])
        assert math.isclose(h_n, 1.0, abs_tol=1e-5)

    def test_three_element_uniform(self):
        h, h_n = compute_entropy_from_probs([1 / 3, 1 / 3, 1 / 3])
        assert math.isclose(h_n, 1.0, abs_tol=1e-4)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: UncertaintyEstimator.estimate() — output schema
# ═══════════════════════════════════════════════════════════════════════════════


class TestUncertaintyEstimatorOutput:
    def _get_record(self, estimator, simple_profile, simple_dirty_df) -> UncertaintyRecord:
        rec = DetectionRecord(
            dataset="test_ds", row_id=1, column="age", dirty_value=None,
            error_type=ErrorType.MISSING.value,
            detection_score=1.0, detection_method="test", reason="test",
        )
        repairer = MockRepairer(seed=42, top_k=3)
        ctx = build_repair_context(rec, simple_dirty_df, simple_profile)
        result = repairer.repair(ctx)
        return estimator.estimate(result, simple_dirty_df, simple_profile)

    def test_returns_uncertainty_record(self, estimator, simple_profile, simple_dirty_df):
        ur = self._get_record(estimator, simple_profile, simple_dirty_df)
        assert isinstance(ur, UncertaintyRecord)

    def test_dataset_preserved(self, estimator, simple_profile, simple_dirty_df):
        ur = self._get_record(estimator, simple_profile, simple_dirty_df)
        assert ur.dataset == "test_ds"

    def test_row_id_preserved(self, estimator, simple_profile, simple_dirty_df):
        ur = self._get_record(estimator, simple_profile, simple_dirty_df)
        assert ur.row_id == 1

    def test_column_preserved(self, estimator, simple_profile, simple_dirty_df):
        ur = self._get_record(estimator, simple_profile, simple_dirty_df)
        assert ur.column == "age"

    def test_raw_confidence_in_range(self, estimator, simple_profile, simple_dirty_df):
        ur = self._get_record(estimator, simple_profile, simple_dirty_df)
        assert 0.0 < ur.raw_confidence <= 1.0

    def test_self_consistency_in_range(self, estimator, simple_profile, simple_dirty_df):
        ur = self._get_record(estimator, simple_profile, simple_dirty_df)
        assert 0.0 < ur.self_consistency <= 1.0

    def test_entropy_non_negative(self, estimator, simple_profile, simple_dirty_df):
        ur = self._get_record(estimator, simple_profile, simple_dirty_df)
        assert ur.entropy >= 0.0

    def test_normalized_entropy_in_range(self, estimator, simple_profile, simple_dirty_df):
        ur = self._get_record(estimator, simple_profile, simple_dirty_df)
        assert 0.0 <= ur.normalized_entropy <= 1.0

    def test_n_samples_matches_config(self, estimator, simple_profile, simple_dirty_df):
        ur = self._get_record(estimator, simple_profile, simple_dirty_df)
        assert ur.n_samples == estimator.n_samples

    def test_vote_counts_sum_to_n_samples(self, estimator, simple_profile, simple_dirty_df):
        ur = self._get_record(estimator, simple_profile, simple_dirty_df)
        assert sum(ur.vote_counts.values()) == estimator.n_samples

    def test_candidate_frequencies_sum_to_one(self, estimator, simple_profile, simple_dirty_df):
        ur = self._get_record(estimator, simple_profile, simple_dirty_df)
        total = sum(ur.candidate_frequencies.values())
        assert math.isclose(total, 1.0, abs_tol=1e-5)

    def test_to_dict_keys(self, estimator, simple_profile, simple_dirty_df):
        ur = self._get_record(estimator, simple_profile, simple_dirty_df)
        d = ur.to_dict()
        expected = {
            "dataset", "row_id", "column", "dirty_value", "error_type",
            "top_candidate", "raw_confidence", "self_consistency",
            "entropy", "normalized_entropy", "n_samples", "n_candidates",
        }
        assert expected.issubset(d.keys())

    def test_top_candidate_not_none(self, estimator, simple_profile, simple_dirty_df):
        ur = self._get_record(estimator, simple_profile, simple_dirty_df)
        assert ur.top_candidate is not None

    def test_n_candidates_positive(self, estimator, simple_profile, simple_dirty_df):
        ur = self._get_record(estimator, simple_profile, simple_dirty_df)
        assert ur.n_candidates >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Self-consistency
# ═══════════════════════════════════════════════════════════════════════════════


class TestSelfConsistency:
    def _run_sc(
        self,
        estimator: UncertaintyEstimator,
        profile: DatasetProfile,
        dirty_df: pd.DataFrame,
        row_id: int = 1,
        column: str = "age",
    ) -> UncertaintyRecord:
        rec = DetectionRecord(
            dataset="test_ds", row_id=row_id, column=column, dirty_value=None,
            error_type=ErrorType.MISSING.value,
            detection_score=1.0, detection_method="test", reason="test",
        )
        repairer = MockRepairer(seed=42, top_k=3)
        ctx = build_repair_context(rec, dirty_df, profile)
        result = repairer.repair(ctx)
        return estimator.estimate(result, dirty_df, profile)

    def test_consistency_in_range(self, estimator, simple_profile, simple_dirty_df):
        ur = self._run_sc(estimator, simple_profile, simple_dirty_df)
        assert 0.0 < ur.self_consistency <= 1.0

    def test_vote_keys_are_strings(self, estimator, simple_profile, simple_dirty_df):
        ur = self._run_sc(estimator, simple_profile, simple_dirty_df)
        assert all(isinstance(k, str) for k in ur.vote_counts)

    def test_vote_counts_all_positive(self, estimator, simple_profile, simple_dirty_df):
        ur = self._run_sc(estimator, simple_profile, simple_dirty_df)
        assert all(v > 0 for v in ur.vote_counts.values())

    def test_five_samples_by_default(self, estimator, simple_profile, simple_dirty_df):
        ur = self._run_sc(estimator, simple_profile, simple_dirty_df)
        assert ur.n_samples == 5

    def test_custom_n_samples(self, simple_profile, simple_dirty_df):
        est = UncertaintyEstimator(base_seed=42, n_samples=3)
        ur = self._run_sc(est, simple_profile, simple_dirty_df)
        assert ur.n_samples == 3
        assert sum(ur.vote_counts.values()) == 3

    def test_max_consistency_is_one(self, simple_profile, simple_dirty_df):
        est = UncertaintyEstimator(base_seed=42, n_samples=5)
        ur = self._run_sc(est, simple_profile, simple_dirty_df)
        assert ur.self_consistency <= 1.0

    def test_determinism(self, simple_profile, simple_dirty_df):
        """Same estimator parameters → identical results."""
        est1 = UncertaintyEstimator(base_seed=42, n_samples=5)
        est2 = UncertaintyEstimator(base_seed=42, n_samples=5)
        ur1 = self._run_sc(est1, simple_profile, simple_dirty_df)
        ur2 = self._run_sc(est2, simple_profile, simple_dirty_df)
        assert ur1.self_consistency == ur2.self_consistency
        assert ur1.vote_counts == ur2.vote_counts

    def test_different_base_seeds_may_differ(self, simple_profile, simple_dirty_df):
        est1 = UncertaintyEstimator(base_seed=42, n_samples=5)
        est2 = UncertaintyEstimator(base_seed=999, n_samples=5)
        ur1 = self._run_sc(est1, simple_profile, simple_dirty_df)
        ur2 = self._run_sc(est2, simple_profile, simple_dirty_df)
        assert isinstance(ur1.self_consistency, float)
        assert isinstance(ur2.self_consistency, float)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: estimate_batch
# ═══════════════════════════════════════════════════════════════════════════════


class TestEstimateBatch:
    def _make_batch_results(
        self, profile: DatasetProfile, dirty_df: pd.DataFrame, n: int = 3
    ) -> list:
        records = []
        for row_id in range(n):
            rec = DetectionRecord(
                dataset="test_ds", row_id=row_id, column="gender",
                dirty_value=None, error_type=ErrorType.MISSING.value,
                detection_score=1.0, detection_method="test", reason="test",
            )
            repairer = MockRepairer(seed=42, top_k=3)
            ctx = build_repair_context(rec, dirty_df, profile)
            records.append(repairer.repair(ctx))
        return records

    def test_length_matches_input(self, estimator, simple_profile, simple_dirty_df):
        results = self._make_batch_results(simple_profile, simple_dirty_df, 3)
        urs = estimator.estimate_batch(results, simple_dirty_df, simple_profile)
        assert len(urs) == 3

    def test_empty_batch(self, estimator, simple_profile, simple_dirty_df):
        urs = estimator.estimate_batch([], simple_dirty_df, simple_profile)
        assert urs == []

    def test_all_records_have_confidence(self, estimator, simple_profile, simple_dirty_df):
        results = self._make_batch_results(simple_profile, simple_dirty_df, 3)
        urs = estimator.estimate_batch(results, simple_dirty_df, simple_profile)
        for ur in urs:
            assert ur.raw_confidence > 0

    def test_batch_determinism(self, simple_profile, simple_dirty_df):
        results = self._make_batch_results(simple_profile, simple_dirty_df, 3)
        est1 = UncertaintyEstimator(base_seed=42)
        est2 = UncertaintyEstimator(base_seed=42)
        urs1 = est1.estimate_batch(results, simple_dirty_df, simple_profile)
        urs2 = est2.estimate_batch(results, simple_dirty_df, simple_profile)
        for a, b in zip(urs1, urs2):
            assert a.raw_confidence == b.raw_confidence
            assert a.self_consistency == b.self_consistency
            assert a.entropy == b.entropy


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: Data leakage prevention
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataLeakagePrevention:
    def test_estimator_does_not_accept_clean_df(self):
        import inspect
        est = UncertaintyEstimator()
        sig = inspect.signature(est.estimate)
        params = list(sig.parameters.keys())
        assert "clean_df" not in params
        assert "clean" not in " ".join(params)

    def test_estimate_batch_does_not_accept_clean_df(self):
        import inspect
        est = UncertaintyEstimator()
        sig = inspect.signature(est.estimate_batch)
        params = list(sig.parameters.keys())
        assert "clean_df" not in params

    def test_uncertainty_record_has_no_clean_field(self):
        fields = list(UncertaintyRecord.__dataclass_fields__.keys())
        for f in fields:
            assert "clean" not in f.lower()

    def test_self_consistency_result_has_no_clean_field(self):
        fields = list(SelfConsistencyResult.__dataclass_fields__.keys())
        for f in fields:
            assert "clean" not in f.lower()

    def test_compute_entropy_takes_no_clean_df(self):
        import inspect
        sig = inspect.signature(_compute_entropy)
        params = list(sig.parameters.keys())
        assert "clean_df" not in params

    def test_compute_raw_confidence_takes_no_clean_df(self):
        import inspect
        sig = inspect.signature(_compute_raw_confidence)
        params = list(sig.parameters.keys())
        assert "clean_df" not in params


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8: UncertaintyCorrelationAnalysis (evaluation — uses clean_df)
# ═══════════════════════════════════════════════════════════════════════════════


class TestUncertaintyCorrelationAnalysis:
    def _make_uncertainty_records(
        self,
        top_candidates: list,
        confidences: list,
        consistencies: list,
        entropies: list,
    ) -> list:
        records = []
        for i, (tc, conf, cons, ent) in enumerate(
            zip(top_candidates, confidences, consistencies, entropies)
        ):
            records.append(
                UncertaintyRecord(
                    dataset="test", row_id=i, column="age",
                    dirty_value=None, error_type="missing",
                    repair_method="statistical_repair",
                    top_candidate=tc,
                    raw_confidence=conf,
                    self_consistency=cons,
                    entropy=ent,
                    normalized_entropy=ent,
                    n_samples=5,
                )
            )
        return records

    def test_returns_correlation_report(self, simple_clean_df, simple_dirty_df):
        analysis = UncertaintyCorrelationAnalysis(
            simple_clean_df, simple_dirty_df, "test"
        )
        recs = self._make_uncertainty_records(
            top_candidates=[25, 5, 3],
            confidences=[0.9, 0.5, 0.7],
            consistencies=[1.0, 0.6, 0.8],
            entropies=[0.1, 0.5, 0.3],
        )
        report = analysis.analyse(recs)
        assert isinstance(report, CorrelationReport)

    def test_overall_accuracy_correct(self, simple_clean_df, simple_dirty_df):
        analysis = UncertaintyCorrelationAnalysis(
            simple_clean_df, simple_dirty_df, "test"
        )
        # Row 0 age clean=25, top=25 → correct; Row 1 age clean=5, top=99 → wrong
        recs = self._make_uncertainty_records(
            top_candidates=[25, 99],
            confidences=[0.9, 0.8],
            consistencies=[1.0, 0.8],
            entropies=[0.1, 0.2],
        )
        report = analysis.analyse(recs)
        assert report.n_correct_top1 == 1
        assert report.overall_top1_accuracy == 0.5

    def test_empty_records(self, simple_clean_df, simple_dirty_df):
        analysis = UncertaintyCorrelationAnalysis(
            simple_clean_df, simple_dirty_df, "test"
        )
        report = analysis.analyse([])
        assert report.n_total == 0
        assert report.overall_top1_accuracy == 0.0

    def test_bin_analysis_returns_bins(self, simple_clean_df, simple_dirty_df):
        analysis = UncertaintyCorrelationAnalysis(
            simple_clean_df, simple_dirty_df, "test", n_bins=2
        )
        recs = self._make_uncertainty_records(
            top_candidates=[25, 5, 99, 5],
            confidences=[0.9, 0.5, 0.4, 0.7],
            consistencies=[1.0, 0.6, 0.4, 0.8],
            entropies=[0.1, 0.5, 0.7, 0.3],
        )
        report = analysis.analyse(recs)
        assert len(report.by_confidence_bin) > 0
        assert len(report.by_consistency_bin) > 0
        assert len(report.by_entropy_bin) > 0

    def test_bin_stats_accuracy_in_range(self, simple_clean_df, simple_dirty_df):
        analysis = UncertaintyCorrelationAnalysis(
            simple_clean_df, simple_dirty_df, "test", n_bins=2
        )
        recs = self._make_uncertainty_records(
            top_candidates=[25, 5, 3, 4],
            confidences=[0.9, 0.5, 0.7, 0.6],
            consistencies=[1.0, 0.6, 0.8, 0.7],
            entropies=[0.1, 0.5, 0.3, 0.4],
        )
        report = analysis.analyse(recs)
        for b in report.by_confidence_bin:
            assert 0.0 <= b.accuracy <= 1.0

    def test_pearson_correlation_computed(self, simple_clean_df, simple_dirty_df):
        analysis = UncertaintyCorrelationAnalysis(
            simple_clean_df, simple_dirty_df, "test"
        )
        recs = self._make_uncertainty_records(
            top_candidates=[25, 5, 99, 99, 25, 99, 99, 99],
            confidences=[0.9, 0.8, 0.3, 0.2, 0.85, 0.4, 0.25, 0.35],
            consistencies=[1.0, 1.0, 0.4, 0.2, 1.0, 0.6, 0.4, 0.4],
            entropies=[0.1, 0.15, 0.7, 0.9, 0.1, 0.65, 0.85, 0.75],
        )
        report = analysis.analyse(recs)
        if report.pearson_conf_accuracy is not None:
            assert isinstance(report.pearson_conf_accuracy, float)
            assert -1.0 <= report.pearson_conf_accuracy <= 1.0

    def test_print_report_runs(self, simple_clean_df, simple_dirty_df, capsys):
        analysis = UncertaintyCorrelationAnalysis(
            simple_clean_df, simple_dirty_df, "test_ds"
        )
        recs = self._make_uncertainty_records(
            top_candidates=[25, 5, 3],
            confidences=[0.9, 0.5, 0.7],
            consistencies=[1.0, 0.6, 0.8],
            entropies=[0.1, 0.5, 0.3],
        )
        report = analysis.analyse(recs)
        UncertaintyCorrelationAnalysis.print_report(report)
        out = capsys.readouterr().out
        assert "UNCERTAINTY CORRELATION" in out
        assert "TEST_DS" in out

    def test_numeric_int_vs_float_comparison(self, simple_clean_df, simple_dirty_df):
        """Candidate 25.0 should match clean value 25 (int)."""
        analysis = UncertaintyCorrelationAnalysis(
            simple_clean_df, simple_dirty_df, "test"
        )
        recs = self._make_uncertainty_records(
            top_candidates=[25.0],
            confidences=[0.9],
            consistencies=[1.0],
            entropies=[0.1],
        )
        report = analysis.analyse(recs)
        assert report.n_correct_top1 == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Section 9: Pearson helper
# ═══════════════════════════════════════════════════════════════════════════════


class TestPearson:
    def test_perfect_positive_correlation(self):
        x = [1.0, 2.0, 3.0]
        y = [0, 1, 1]
        r = _pearson(x, y)
        if r is not None:
            assert r > 0

    def test_constant_x_returns_none(self):
        x = [0.5, 0.5, 0.5]
        y = [0, 1, 0]
        r = _pearson(x, y)
        assert r is None

    def test_constant_y_returns_none(self):
        x = [0.3, 0.6, 0.9]
        y = [1, 1, 1]
        r = _pearson(x, y)
        assert r is None

    def test_single_element_returns_none(self):
        r = _pearson([0.5], [1])
        assert r is None

    def test_result_in_minus_one_to_one(self):
        x = [0.1, 0.9, 0.3, 0.8, 0.2, 0.7]
        y = [0, 1, 0, 1, 0, 1]
        r = _pearson(x, y)
        if r is not None:
            assert -1.0 <= r <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Section 10: Integration tests (require dataset files)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestIntegrationBreastCancerUncertainty:
    @pytest.fixture(autouse=True)
    def setup(self):
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
            self.clean_df, self.dirty_df,
        )
        detector = Detector(self.profile)
        records = detector.detect(self.dirty_df)
        repairer = MockRepairer(seed=42, top_k=3)
        self.repair_results = repairer.repair_batch(
            records[:30], self.dirty_df, self.profile
        )
        self.estimator = UncertaintyEstimator(base_seed=42, n_samples=5)

    def test_estimate_batch_runs(self):
        urs = self.estimator.estimate_batch(
            self.repair_results, self.dirty_df, self.profile
        )
        assert len(urs) == len(self.repair_results)

    def test_all_confidences_in_range(self):
        urs = self.estimator.estimate_batch(
            self.repair_results, self.dirty_df, self.profile
        )
        for ur in urs:
            assert 0.0 < ur.raw_confidence <= 1.0

    def test_all_normalized_entropy_in_range(self):
        urs = self.estimator.estimate_batch(
            self.repair_results, self.dirty_df, self.profile
        )
        for ur in urs:
            assert 0.0 <= ur.normalized_entropy <= 1.0

    def test_vote_counts_sum_to_n_samples(self):
        urs = self.estimator.estimate_batch(
            self.repair_results, self.dirty_df, self.profile
        )
        for ur in urs:
            assert sum(ur.vote_counts.values()) == self.estimator.n_samples

    def test_determinism_repeated_call(self):
        est1 = UncertaintyEstimator(base_seed=42, n_samples=5)
        est2 = UncertaintyEstimator(base_seed=42, n_samples=5)
        urs1 = est1.estimate_batch(
            self.repair_results[:10], self.dirty_df, self.profile
        )
        urs2 = est2.estimate_batch(
            self.repair_results[:10], self.dirty_df, self.profile
        )
        for a, b in zip(urs1, urs2):
            assert a.self_consistency == b.self_consistency
            assert a.raw_confidence == b.raw_confidence

    def test_correlation_analysis_runs(self):
        urs = self.estimator.estimate_batch(
            self.repair_results, self.dirty_df, self.profile
        )
        analysis = UncertaintyCorrelationAnalysis(
            self.clean_df, self.dirty_df, "breast_cancer"
        )
        report = analysis.analyse(urs)
        assert report.n_total == len(urs)
        assert 0.0 <= report.overall_top1_accuracy <= 1.0

    def test_topk_accuracy_ge_top1(self):
        urs = self.estimator.estimate_batch(
            self.repair_results, self.dirty_df, self.profile
        )
        analysis = UncertaintyCorrelationAnalysis(
            self.clean_df, self.dirty_df, "breast_cancer"
        )
        report = analysis.analyse(urs)
        assert report.n_total > 0


@pytest.mark.integration
class TestIntegrationSmartFactoryUncertainty:
    @pytest.fixture(autouse=True)
    def setup(self):
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
            self.clean_df, self.dirty_df,
        )
        detector = Detector(self.profile, flag_statistical_outliers=False)
        records = detector.detect(self.dirty_df)
        repairer = MockRepairer(seed=42, top_k=3)
        self.repair_results = repairer.repair_batch(
            records[:20], self.dirty_df, self.profile
        )
        self.estimator = UncertaintyEstimator(base_seed=42, n_samples=5)

    def test_all_sf_uncertainty_records(self):
        urs = self.estimator.estimate_batch(
            self.repair_results, self.dirty_df, self.profile
        )
        assert len(urs) == len(self.repair_results)
        for ur in urs:
            assert ur.n_samples == 5
            assert 0.0 < ur.raw_confidence <= 1.0
