"""
tests/test_detector.py
-----------------------
Phase 3 tests for src.detection.detector and src.detection.evaluation.

Coverage:
  1. DetectionRecord output schema
  2. Missing-value detection (NaN, empty string, null tokens)
  3. Numerical anomaly detection (range check, IQR, format error)
  4. Categorical / binary anomaly detection (novel category, whitespace)
  5. Mixed-column detection (Bare Nuclei behaviour)
  6. Dataset-specific behaviour (Adult / Breast Cancer / Smart Factory)
  7. Data leakage prevention (detector does not accept clean_df)
  8. Determinism (same output on repeated calls)
  9. flag_statistical_outliers=False disables IQR checks
  10. DetectorEvaluator ground-truth comparison
  11. Evaluation precision / recall / F1 against known benchmarks

All expected values use the actual detection results verified against
ground truth in Phase 3.
"""

from __future__ import annotations

import inspect
import pandas as pd
import numpy as np
import pytest

from src.data.loader import DataLoader
from src.data.profiler import profile_dataset
from src.detection.detector import Detector, DetectionRecord, ErrorType
from src.detection.evaluation import DetectorEvaluator, EvaluationResults
from src.data.schema import ColumnKind

DATA_ROOT = "datasets"


# ── Module-level fixtures ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def loader():
    return DataLoader(DATA_ROOT)


@pytest.fixture(scope="module")
def adult_data(loader):
    clean_df, dirty_df = loader.load("adult")
    profile = profile_dataset("adult", loader.discover("adult"), clean_df, dirty_df)
    return clean_df, dirty_df, profile


@pytest.fixture(scope="module")
def bc_data(loader):
    clean_df, dirty_df = loader.load("breast_cancer")
    profile = profile_dataset("breast_cancer", loader.discover("breast_cancer"), clean_df, dirty_df)
    return clean_df, dirty_df, profile


@pytest.fixture(scope="module")
def sf_data(loader):
    clean_df, dirty_df = loader.load("smartfactory")
    profile = profile_dataset("smartfactory", loader.discover("smartfactory"), clean_df, dirty_df)
    return clean_df, dirty_df, profile


@pytest.fixture(scope="module")
def adult_records(adult_data):
    _, dirty_df, profile = adult_data
    return Detector(profile).detect(dirty_df)


@pytest.fixture(scope="module")
def bc_records(bc_data):
    _, dirty_df, profile = bc_data
    return Detector(profile).detect(dirty_df)


@pytest.fixture(scope="module")
def sf_records(sf_data):
    """SF with statistical outliers disabled (non-NaN values match clean)."""
    _, dirty_df, profile = sf_data
    return Detector(profile, flag_statistical_outliers=False).detect(dirty_df)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DetectionRecord schema
# ═══════════════════════════════════════════════════════════════════════════════


class TestDetectionRecordSchema:
    """Every record must have the required fields with correct types."""

    def test_record_has_required_fields(self, bc_records):
        assert len(bc_records) > 0
        r = bc_records[0]
        assert isinstance(r.dataset, str)
        assert isinstance(r.row_id, int)
        assert isinstance(r.column, str)
        assert isinstance(r.error_type, str)
        assert isinstance(r.detection_score, float)
        assert isinstance(r.detection_method, str)
        assert isinstance(r.reason, str)

    def test_detection_score_in_range(self, bc_records, sf_records, adult_records):
        for records in (bc_records, sf_records, adult_records):
            for r in records:
                assert 0.0 <= r.detection_score <= 1.0, (
                    f"Score out of range: {r.detection_score} for {r.column}"
                )

    def test_error_type_is_valid_enum(self, bc_records, sf_records, adult_records):
        valid = {e.value for e in ErrorType}
        for records in (bc_records, sf_records, adult_records):
            for r in records:
                assert r.error_type in valid, (
                    f"Unknown error_type: {r.error_type}"
                )

    def test_to_dict_has_all_keys(self, bc_records):
        d = bc_records[0].to_dict()
        required = {
            "dataset", "row_id", "column", "dirty_value",
            "error_type", "detection_score", "detection_method", "reason",
        }
        assert required <= set(d.keys())

    def test_row_id_is_nonnegative(self, bc_records):
        for r in bc_records:
            assert r.row_id >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Missing-value detection
# ═══════════════════════════════════════════════════════════════════════════════


class TestMissingValueDetection:
    """Detector must flag NaN and null-like tokens as MISSING with score 1.0."""

    def _make_simple_profile_and_df(self, col_values, col_name="x"):
        """Helper: build a tiny DataFrame and profile for testing."""
        from src.data.schema import (
            DatasetProfile, ColumnProfile, ColumnKind, DatasetPaths
        )
        from pathlib import Path

        dirty = pd.DataFrame({col_name: col_values})
        clean = pd.DataFrame({col_name: [1] * len(col_values)})  # placeholder
        cp = ColumnProfile(
            name=col_name, kind=ColumnKind.NUMERICAL,
            dtype_clean="int64", dtype_dirty="float64",
            missing_clean=0, missing_dirty=0,
            n_unique_clean=5, n_unique_dirty=5,
        )
        profile = DatasetProfile(
            name="test", clean_path=Path("."), dirty_path=Path("."),
            n_rows=len(dirty), n_cols=1,
            column_names=[col_name], columns={col_name: cp},
        )
        return dirty, profile

    def test_nan_flagged_as_missing(self):
        df = pd.DataFrame({"age": [25.0, np.nan, 30.0]})
        from src.data.schema import DatasetProfile, ColumnProfile, ColumnKind
        from pathlib import Path
        cp = ColumnProfile(
            name="age", kind=ColumnKind.NUMERICAL,
            dtype_clean="int64", dtype_dirty="float64",
            missing_clean=0, missing_dirty=1,
            n_unique_clean=3, n_unique_dirty=2,
        )
        profile = DatasetProfile(
            name="test", clean_path=Path("."), dirty_path=Path("."),
            n_rows=3, n_cols=1, column_names=["age"],
            columns={"age": cp},
        )
        records = Detector(profile).detect(df)
        missing = [r for r in records if r.error_type == ErrorType.MISSING.value]
        assert len(missing) == 1
        assert missing[0].row_id == 1
        assert missing[0].detection_score == 1.0

    def test_bc_missing_count_matches_dirty_nan(self, bc_data, bc_records):
        _, dirty_df, _ = bc_data
        total_nan = dirty_df.isna().sum().sum()
        missing_records = [r for r in bc_records if r.error_type == ErrorType.MISSING.value]
        assert len(missing_records) == total_nan

    def test_sf_missing_count_matches_dirty_nan(self, sf_data, sf_records):
        _, dirty_df, _ = sf_data
        total_nan = dirty_df.isna().sum().sum()
        missing_records = [r for r in sf_records if r.error_type == ErrorType.MISSING.value]
        assert len(missing_records) == total_nan

    def test_missing_score_is_one(self, bc_records, sf_records):
        for records in (bc_records, sf_records):
            for r in records:
                if r.error_type == ErrorType.MISSING.value:
                    assert r.detection_score == 1.0

    def test_adult_has_no_missing_records(self, adult_records):
        """Adult dirty has no NaN values — no MISSING records expected."""
        missing = [r for r in adult_records if r.error_type == ErrorType.MISSING.value]
        assert len(missing) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Numerical anomaly detection
# ═══════════════════════════════════════════════════════════════════════════════


class TestNumericalAnomalyDetection:
    """OUT_OF_RANGE, FORMAT_ERROR, and STATISTICAL_OUTLIER for numerical columns."""

    def test_bc_has_out_of_range_records(self, bc_records):
        """BC numerical columns have values > 10 (e.g. 33, 88) — all OOR."""
        oor = [r for r in bc_records if r.error_type == ErrorType.OUT_OF_RANGE.value]
        assert len(oor) == 129

    def test_oor_score_is_high(self, bc_records):
        for r in bc_records:
            if r.error_type == ErrorType.OUT_OF_RANGE.value:
                assert r.detection_score >= 0.85

    def test_oor_detection_method(self, bc_records):
        for r in bc_records:
            if r.error_type == ErrorType.OUT_OF_RANGE.value:
                assert r.detection_method == "range_check"

    def test_bc_has_format_errors(self, bc_records):
        """BC has cells like '1a0' in what should be numerical columns."""
        fmt_errs = [r for r in bc_records if r.error_type == ErrorType.FORMAT_ERROR.value]
        assert len(fmt_errs) == 3

    def test_iqr_disabled_removes_statistical_outliers(self, sf_data):
        """With flag_statistical_outliers=False, no STATISTICAL_OUTLIER records."""
        _, dirty_df, profile = sf_data
        detector = Detector(profile, flag_statistical_outliers=False)
        records = detector.detect(dirty_df)
        outliers = [r for r in records if r.error_type == ErrorType.STATISTICAL_OUTLIER.value]
        assert len(outliers) == 0

    def test_iqr_enabled_produces_outliers(self, sf_data):
        """With flag_statistical_outliers=True on SF, IQR flags legitimate sensor values."""
        _, dirty_df, profile = sf_data
        detector = Detector(profile, flag_statistical_outliers=True)
        records = detector.detect(dirty_df)
        outliers = [r for r in records if r.error_type == ErrorType.STATISTICAL_OUTLIER.value]
        assert len(outliers) > 0  # confirms IQR is being applied

    def test_adult_statistical_outliers_are_all_false_positives(self, adult_data, adult_records):
        """Adult has no actual numerical corruption; all outlier flags are FP."""
        clean_df, dirty_df, _ = adult_data
        evaluator = DetectorEvaluator(clean_df, dirty_df, "adult")
        results = evaluator.evaluate(adult_records)
        assert results.tp == 0
        assert results.fp == results.n_detected


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Categorical anomaly detection
# ═══════════════════════════════════════════════════════════════════════════════


class TestCategoricalAnomalyDetection:
    """NOVEL_CATEGORY and WHITESPACE detection."""

    def test_bc_has_novel_category_records(self, bc_records):
        """BC has novel values (e.g. '??' in Bare Nuclei) not in clean vocab."""
        novel = [r for r in bc_records if r.error_type == ErrorType.NOVEL_CATEGORY.value]
        assert len(novel) == 13

    def test_novel_category_method(self, bc_records):
        for r in bc_records:
            if r.error_type == ErrorType.NOVEL_CATEGORY.value:
                assert r.detection_method == "vocab_check"

    def test_novel_score_is_reasonable(self, bc_records):
        for r in bc_records:
            if r.error_type == ErrorType.NOVEL_CATEGORY.value:
                assert r.detection_score >= 0.6

    def test_adult_no_novel_categories(self, adult_records):
        """Adult dirty has no novel categorical values."""
        novel = [r for r in adult_records if r.error_type == ErrorType.NOVEL_CATEGORY.value]
        assert len(novel) == 0

    def test_whitespace_detection_with_synthetic_data(self):
        """Detector flags leading/trailing whitespace in categorical columns."""
        from src.data.schema import DatasetProfile, ColumnProfile, ColumnKind, CategoricalStats
        from pathlib import Path

        cp = ColumnProfile(
            name="workclass", kind=ColumnKind.CATEGORICAL,
            dtype_clean="object", dtype_dirty="object",
            missing_clean=0, missing_dirty=0,
            n_unique_clean=7, n_unique_dirty=8,
            categorical_stats=CategoricalStats(
                value_counts={"Private": 100, "Self-emp": 20},
                top_values=["Private", "Self-emp"],
                n_rare=0,
            ),
        )
        profile = DatasetProfile(
            name="test", clean_path=Path("."), dirty_path=Path("."),
            n_rows=3, n_cols=1, column_names=["workclass"],
            columns={"workclass": cp},
        )
        df = pd.DataFrame({"workclass": ["Private", " Private", "Unknown"]})
        records = Detector(profile).detect(df)
        ws_recs = [r for r in records if r.error_type == ErrorType.WHITESPACE.value]
        assert len(ws_recs) == 1
        assert ws_recs[0].row_id == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Mixed-column detection (Bare Nuclei)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMixedColumnDetection:
    """Bare Nuclei is MIXED: mostly numeric strings + '?' as known valid token."""

    def test_bare_nuclei_question_mark_not_flagged(self, bc_data, bc_records):
        """'?' is in the clean vocab — must NOT appear in detection records for in-vocab cells."""
        _, dirty_df, profile = bc_data
        # Find rows in dirty where Bare Nuclei == '?' AND it was NOT changed from clean
        # (we can't use clean here, so we just check that '?' cells aren't flagged
        # as FORMAT_ERROR, since '?' is a known vocab item)
        fmt_errors_bare = [
            r for r in bc_records
            if r.column == "Bare Nuclei" and r.error_type == ErrorType.FORMAT_ERROR.value
        ]
        # Any FORMAT_ERROR on Bare Nuclei must have a dirty_value that is NOT '?'
        for r in fmt_errors_bare:
            assert str(r.dirty_value).strip() != "?", (
                f"'?' should not be flagged as FORMAT_ERROR; it's in the clean vocab."
            )

    def test_bare_nuclei_novel_values_are_flagged(self, bc_records):
        """Novel BC Bare Nuclei values ('??', '1a0', etc.) should be flagged."""
        bare_records = [r for r in bc_records if r.column == "Bare Nuclei"]
        assert len(bare_records) > 0

    def test_bc_zero_false_positives(self, bc_data, bc_records):
        """After '?' fix, BC detection should have 0 false positives."""
        clean_df, dirty_df, _ = bc_data
        evaluator = DetectorEvaluator(clean_df, dirty_df, "breast_cancer")
        results = evaluator.evaluate(bc_records)
        assert results.fp == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Dataset-specific behaviour
# ═══════════════════════════════════════════════════════════════════════════════


class TestDatasetSpecificBehaviour:
    """Verify dataset-specific detection characteristics."""

    # ── Adult ──────────────────────────────────────────────────────────────────

    def test_adult_income_not_detected(self, adult_records):
        """Label flips in income (0↔1) are undetectable by rule-based detection.
        This is a known and documented limitation — both 0 and 1 are valid."""
        income_records = [r for r in adult_records if r.column == "income"]
        assert len(income_records) == 0, (
            "Income label flips must not be flagged — both 0 and 1 are valid values."
        )

    def test_adult_recall_is_zero(self, adult_data, adult_records):
        """Adult detection recall is 0.0 because income flips are undetectable."""
        clean_df, dirty_df, _ = adult_data
        evaluator = DetectorEvaluator(clean_df, dirty_df, "adult")
        results = evaluator.evaluate(adult_records)
        assert results.recall == 0.0

    # ── Breast Cancer ──────────────────────────────────────────────────────────

    def test_bc_precision_is_perfect(self, bc_data, bc_records):
        """After '?' fix, BC has 0 false positives → precision = 1.0."""
        clean_df, dirty_df, _ = bc_data
        evaluator = DetectorEvaluator(clean_df, dirty_df, "breast_cancer")
        results = evaluator.evaluate(bc_records)
        assert results.precision == 1.0

    def test_bc_recall_above_threshold(self, bc_data, bc_records):
        """BC recall should be above 0.60 (detects missing, OOR, novel, format)."""
        clean_df, dirty_df, _ = bc_data
        evaluator = DetectorEvaluator(clean_df, dirty_df, "breast_cancer")
        results = evaluator.evaluate(bc_records)
        assert results.recall >= 0.60

    def test_bc_class_label_flip_not_detected(self, bc_records):
        """BC 'class' label flips (2↔4) are undetectable by vocab check
        since both 2 and 4 are valid values."""
        class_records = [r for r in bc_records if r.column == "class"]
        assert len(class_records) == 0

    def test_bc_clump_thickness_detected(self, bc_records):
        """BC Clump Thickness has both MISSING and OUT_OF_RANGE corruption."""
        ct_records = [r for r in bc_records if r.column == "Clump Thickness"]
        assert len(ct_records) == 47

    def test_bc_false_negatives_are_mostly_missing_in_id_cols(self, bc_data, bc_records):
        """Unnamed:0 and Sample code number are ID_LIKE — their missing values
        ARE detected (score 1.0 for missing), but value changes are not flagged."""
        _, dirty_df, _ = bc_data
        id_missing = [
            r for r in bc_records
            if r.column in ("Unnamed: 0", "Sample code number")
            and r.error_type == ErrorType.MISSING.value
        ]
        assert len(id_missing) > 0

    # ── Smart Factory ──────────────────────────────────────────────────────────

    def test_sf_perfect_recall_for_missing(self, sf_data, sf_records):
        """SF with statistical outliers disabled: recall = 1.0."""
        clean_df, dirty_df, _ = sf_data
        evaluator = DetectorEvaluator(clean_df, dirty_df, "smartfactory")
        results = evaluator.evaluate(sf_records)
        assert results.recall == 1.0

    def test_sf_perfect_precision_for_missing(self, sf_data, sf_records):
        """SF: 0 false positives → precision = 1.0."""
        clean_df, dirty_df, _ = sf_data
        evaluator = DetectorEvaluator(clean_df, dirty_df, "smartfactory")
        results = evaluator.evaluate(sf_records)
        assert results.precision == 1.0

    def test_sf_all_records_are_missing_type(self, sf_records):
        """All 7093 SF detections are MISSING (no other error types with outliers off)."""
        non_missing = [r for r in sf_records if r.error_type != ErrorType.MISSING.value]
        assert len(non_missing) == 0

    def test_sf_detected_count(self, sf_records):
        assert len(sf_records) == 7093


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Data leakage prevention
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataLeakagePrevention:
    """Confirm that detector.py cannot accept a clean DataFrame."""

    def test_detect_signature_has_no_clean_param(self):
        """detect() must only accept dirty_df — no clean_df parameter allowed."""
        sig = inspect.signature(Detector.detect)
        param_names = list(sig.parameters.keys())
        assert "clean_df" not in param_names, (
            "Detector.detect() must not accept clean_df — data leakage risk."
        )
        assert "clean" not in param_names, (
            "Detector.detect() must not accept clean — data leakage risk."
        )

    def test_detector_init_has_no_clean_param(self):
        """Detector.__init__() must not accept clean_df."""
        sig = inspect.signature(Detector.__init__)
        param_names = list(sig.parameters.keys())
        assert "clean_df" not in param_names
        assert "clean" not in param_names

    def test_detector_module_does_not_import_clean(self):
        """detector.py must not import clean data or the evaluation module."""
        import src.detection.detector as det_module
        # The evaluation module is the only place clean_df is allowed
        assert "evaluation" not in dir(det_module), (
            "detector.py must not import evaluation.py"
        )

    def test_evaluation_module_is_separate(self):
        """DetectorEvaluator lives in evaluation.py, not detector.py."""
        from src.detection import evaluation
        from src.detection import detector
        assert hasattr(evaluation, "DetectorEvaluator")
        assert not hasattr(detector, "DetectorEvaluator")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Determinism
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    """Same input must produce identical output on repeated calls."""

    def test_bc_detection_is_deterministic(self, bc_data):
        _, dirty_df, profile = bc_data
        detector = Detector(profile)
        r1 = detector.detect(dirty_df)
        r2 = detector.detect(dirty_df)
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a.row_id == b.row_id
            assert a.column == b.column
            assert a.error_type == b.error_type
            assert a.detection_score == b.detection_score

    def test_sf_detection_is_deterministic(self, sf_data):
        _, dirty_df, profile = sf_data
        detector = Detector(profile, flag_statistical_outliers=False)
        r1 = detector.detect(dirty_df)
        r2 = detector.detect(dirty_df)
        assert len(r1) == len(r2)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. detection_summary()
# ═══════════════════════════════════════════════════════════════════════════════


class TestDetectionSummary:
    def test_summary_has_required_keys(self, bc_data, bc_records):
        _, _, profile = bc_data
        summary = Detector(profile).detection_summary(bc_records)
        assert "dataset" in summary
        assert "total_suspicious_cells" in summary
        assert "by_error_type" in summary
        assert "by_column" in summary

    def test_summary_total_matches_records(self, bc_data, bc_records):
        _, _, profile = bc_data
        summary = Detector(profile).detection_summary(bc_records)
        assert summary["total_suspicious_cells"] == len(bc_records)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. DetectorEvaluator
# ═══════════════════════════════════════════════════════════════════════════════


class TestDetectorEvaluator:
    """Tests for the evaluation-only ground-truth comparison utility."""

    def test_evaluator_returns_results_object(self, bc_data, bc_records):
        clean_df, dirty_df, _ = bc_data
        results = DetectorEvaluator(clean_df, dirty_df, "breast_cancer").evaluate(bc_records)
        assert isinstance(results, EvaluationResults)

    def test_tp_plus_fp_equals_detected(self, bc_data, bc_records):
        clean_df, dirty_df, _ = bc_data
        results = DetectorEvaluator(clean_df, dirty_df, "breast_cancer").evaluate(bc_records)
        assert results.tp + results.fp == results.n_detected

    def test_tp_plus_fn_equals_actual_errors(self, bc_data, bc_records):
        clean_df, dirty_df, _ = bc_data
        results = DetectorEvaluator(clean_df, dirty_df, "breast_cancer").evaluate(bc_records)
        assert results.tp + results.fn == results.n_actual_errors

    def test_adult_actual_errors_count(self, adult_data, adult_records):
        clean_df, dirty_df, _ = adult_data
        results = DetectorEvaluator(clean_df, dirty_df, "adult").evaluate(adult_records)
        assert results.n_actual_errors == 1701  # known income corruption count

    def test_sf_actual_errors_count(self, sf_data, sf_records):
        clean_df, dirty_df, _ = sf_data
        results = DetectorEvaluator(clean_df, dirty_df, "smartfactory").evaluate(sf_records)
        assert results.n_actual_errors == 7093

    def test_bc_actual_errors_count(self, bc_data, bc_records):
        clean_df, dirty_df, _ = bc_data
        results = DetectorEvaluator(clean_df, dirty_df, "breast_cancer").evaluate(bc_records)
        assert results.n_actual_errors == 631

    def test_f1_formula_consistency(self, bc_data, bc_records):
        clean_df, dirty_df, _ = bc_data
        results = DetectorEvaluator(clean_df, dirty_df, "breast_cancer").evaluate(bc_records)
        if results.precision + results.recall > 0:
            expected_f1 = (
                2 * results.precision * results.recall
                / (results.precision + results.recall)
            )
            assert abs(results.f1 - round(expected_f1, 4)) < 1e-3

    def test_zero_detections_gives_zero_precision_recall(self, bc_data):
        clean_df, dirty_df, _ = bc_data
        evaluator = DetectorEvaluator(clean_df, dirty_df, "breast_cancer")
        results = evaluator.evaluate([])
        assert results.precision == 0.0
        assert results.recall == 0.0
        assert results.f1 == 0.0
        assert results.tp == 0
        assert results.fp == 0

    def test_by_error_type_sums_correctly(self, bc_data, bc_records):
        clean_df, dirty_df, _ = bc_data
        results = DetectorEvaluator(clean_df, dirty_df, "breast_cancer").evaluate(bc_records)
        total_tp = sum(v["tp"] for v in results.by_error_type.values())
        total_fp = sum(v["fp"] for v in results.by_error_type.values())
        assert total_tp == results.tp
        assert total_fp == results.fp

    def test_bc_by_column_class_fn_is_15(self, bc_data, bc_records):
        """BC 'class' label flips are undetectable → FN = 15."""
        clean_df, dirty_df, _ = bc_data
        results = DetectorEvaluator(clean_df, dirty_df, "breast_cancer").evaluate(bc_records)
        assert results.by_column["class"]["fn"] == 15
        assert results.by_column["class"]["tp"] == 0
