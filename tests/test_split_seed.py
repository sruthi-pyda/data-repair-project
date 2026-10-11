"""
tests/test_split_seed.py
------------------------
Phase 13 Step 1: the ISOLATION PROPERTY that repeated-split confidence
intervals rest on (docs/phase13_plan.md §0).

``run_pipeline`` has two seeds and they must do different jobs:

    seed        → MockRepairer + UncertaintyEstimator (the predictions)
    split_seed  → the calibration train/eval split ONLY (p and q)

If ``split_seed`` leaked into the repairer, a "confidence interval over
splits" would really be an interval over *different repairs*, which is a
different experiment and a dishonest error bar. If ``split_seed`` changed
nothing at all, the interval would be a constant dressed up as a range.
Both failure modes are asserted against here, the second one explicitly.

Covers
  - split_seed=None is bit-identical to passing split_seed == seed
    (the default path cannot have moved)
  - varying split_seed: detections / repairs / raw_confidence /
    self_consistency / normalized_entropy IDENTICAL; p and/or q DIFFER
  - varying seed: raw confidences move (the repair noise really is on seed)
  - determinism: same (seed, split_seed) twice → identical bundles
  - validation: split_seed must be an int >= 0
  - a small repeated_splits run: n==3 intervals where fitted, bounds
    ordered, deterministic across two calls
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from src.experiments.aggregate import repeated_splits
from src.pipeline import PipelineBundle, run_pipeline
from src.risk.cost_model import CostConfig

DATA_ROOT = "datasets"
COSTS_YAML = Path("configs/costs.yaml")
DATASET = "breast_cancer"
N = 120


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison helpers — field by field, because a bundle holds DataFrames
# ═══════════════════════════════════════════════════════════════════════════════


def detection_fingerprint(bundle: PipelineBundle) -> List[tuple]:
    return [
        (d.row_id, d.column, d.detection_method, repr(d.dirty_value),
         round(float(d.detection_score), 12))
        for d in bundle.detections
    ]


def top_candidate_fingerprint(bundle: PipelineBundle) -> List[tuple]:
    out = []
    for r in bundle.repair_results:
        if r.candidates:
            c = r.candidates[0]
            out.append((repr(c.value), round(float(c.probability), 12)))
        else:
            out.append((None, None))
    return out


def all_candidate_fingerprint(bundle: PipelineBundle) -> List[tuple]:
    return [
        tuple((repr(c.value), round(float(c.probability), 12)) for c in r.candidates)
        for r in bundle.repair_results
    ]


def uncertainty_fingerprint(bundle: PipelineBundle) -> List[tuple]:
    return [
        (round(u.raw_confidence, 12), round(u.self_consistency, 12),
         round(u.entropy, 12), round(u.normalized_entropy, 12))
        for u in bundle.unc_records
    ]


def assert_predictions_identical(a: PipelineBundle, b: PipelineBundle) -> None:
    """Everything produced BEFORE calibration must match exactly."""
    assert detection_fingerprint(a) == detection_fingerprint(b)
    assert all_candidate_fingerprint(a) == all_candidate_fingerprint(b)
    assert uncertainty_fingerprint(a) == uncertainty_fingerprint(b)
    assert a.raw_confidences == b.raw_confidences
    assert a.detection_labels == b.detection_labels
    assert a.n_detected_total == b.n_detected_total


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures (module-scoped: run_pipeline is the expensive part)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def default_bundle() -> PipelineBundle:
    """Today's call: one seed, split_seed left at None."""
    return run_pipeline(DATASET, n_records=N, seed=42, data_root=DATA_ROOT)


@pytest.fixture(scope="module")
def explicit_bundle() -> PipelineBundle:
    """The same thing said out loud: split_seed == seed."""
    return run_pipeline(DATASET, n_records=N, seed=42, split_seed=42,
                        data_root=DATA_ROOT)


@pytest.fixture(scope="module")
def split_1000() -> PipelineBundle:
    return run_pipeline(DATASET, n_records=N, seed=42, split_seed=1000,
                        data_root=DATA_ROOT)


@pytest.fixture(scope="module")
def split_2000() -> PipelineBundle:
    return run_pipeline(DATASET, n_records=N, seed=42, split_seed=2000,
                        data_root=DATA_ROOT)


@pytest.fixture(scope="module")
def bc_costs() -> CostConfig:
    return CostConfig.from_yaml(COSTS_YAML, DATASET)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. The default path cannot have moved
# ═══════════════════════════════════════════════════════════════════════════════


class TestDefaultPathUnchanged:
    def test_split_seed_none_equals_split_seed_equal_to_seed(
        self, default_bundle, explicit_bundle
    ):
        assert_predictions_identical(default_bundle, explicit_bundle)
        assert default_bundle.calibrated_probs == explicit_bundle.calibrated_probs
        assert default_bundle.corruption_probs == explicit_bundle.corruption_probs
        assert default_bundle.calibration_method == explicit_bundle.calibration_method
        assert default_bundle.calibration_note == explicit_bundle.calibration_note
        assert default_bundle.detection_note == explicit_bundle.detection_note

    def test_reported_comparison_is_also_unchanged(
        self, default_bundle, explicit_bundle
    ):
        a, b = default_bundle.comparison, explicit_bundle.comparison
        assert (a is None) == (b is None)
        assert a is not None
        for attr in (
            "n_train", "n_eval", "n_positive_eval", "base_rate",
            "raw_brier", "logistic_brier", "isotonic_brier", "multivariate_brier",
            "raw_ece", "logistic_ece", "isotonic_ece", "multivariate_ece",
            "logistic_fitted", "isotonic_fitted", "multivariate_fitted",
            "logistic_a", "logistic_b", "multivariate_weights",
            "multivariate_bias", "limitation",
        ):
            assert getattr(a, attr) == getattr(b, attr), attr

    def test_detection_calibration_report_unchanged(
        self, default_bundle, explicit_bundle
    ):
        assert (
            default_bundle.detection_calibration.to_dict()
            == explicit_bundle.detection_calibration.to_dict()
        )

    def test_bundle_records_the_effective_split_seed(self, default_bundle):
        assert default_bundle.split_seed == default_bundle.seed == 42

    def test_bundle_records_an_explicit_split_seed(self, split_1000):
        assert split_1000.seed == 42
        assert split_1000.split_seed == 1000

    @pytest.mark.parametrize("dataset", ["breast_cancer", "adult", "smartfactory"])
    def test_default_path_unchanged_on_every_dataset(self, dataset):
        a = run_pipeline(dataset, n_records=40, seed=42, data_root=DATA_ROOT)
        b = run_pipeline(dataset, n_records=40, seed=42, split_seed=42,
                         data_root=DATA_ROOT)
        assert_predictions_identical(a, b)
        assert a.calibrated_probs == b.calibrated_probs
        assert a.corruption_probs == b.corruption_probs


# ═══════════════════════════════════════════════════════════════════════════════
# 2. THE ISOLATION PROPERTY
# ═══════════════════════════════════════════════════════════════════════════════


class TestSplitSeedIsolation:
    def test_detections_identical(self, split_1000, split_2000):
        assert detection_fingerprint(split_1000) == detection_fingerprint(split_2000)

    def test_top_repair_candidates_identical(self, split_1000, split_2000):
        assert (
            top_candidate_fingerprint(split_1000)
            == top_candidate_fingerprint(split_2000)
        )

    def test_all_repair_candidates_identical(self, split_1000, split_2000):
        assert (
            all_candidate_fingerprint(split_1000)
            == all_candidate_fingerprint(split_2000)
        )

    def test_raw_confidences_identical(self, split_1000, split_2000):
        assert split_1000.raw_confidences == split_2000.raw_confidences

    def test_self_consistency_identical(self, split_1000, split_2000):
        assert (
            [u.self_consistency for u in split_1000.unc_records]
            == [u.self_consistency for u in split_2000.unc_records]
        )

    def test_normalized_entropy_identical(self, split_1000, split_2000):
        assert (
            [u.normalized_entropy for u in split_1000.unc_records]
            == [u.normalized_entropy for u in split_2000.unc_records]
        )

    def test_ground_truth_labels_identical(self, split_1000, split_2000):
        """The labels themselves do not move; only which of them are trained on."""
        assert split_1000.detection_labels == split_2000.detection_labels
        assert (
            [s.correct for s in split_1000.cal_samples]
            == [s.correct for s in split_2000.cal_samples]
        )

    def test_the_split_actually_does_something(self, split_1000, split_2000):
        """
        The negative half of the property. If neither p nor q moves when the
        calibration split moves, the split is inert and a repeated-split
        "confidence interval" would be a fabricated range. Fail loudly.
        """
        p_differs = split_1000.calibrated_probs != split_2000.calibrated_probs
        q_differs = split_1000.corruption_probs != split_2000.corruption_probs
        assert p_differs or q_differs, (
            "split_seed changed neither calibrated_probs nor corruption_probs; "
            "the calibration split is not being re-drawn, so repeated-split "
            "intervals would be meaningless."
        )

    def test_reported_brier_moves_across_splits(self, split_1000, split_2000):
        """The eval-split metrics are what the intervals are built from."""
        a, b = split_1000.comparison, split_2000.comparison
        assert a is not None and b is not None
        assert a.raw_brier != b.raw_brier


# ═══════════════════════════════════════════════════════════════════════════════
# 3. seed still owns the repair noise
# ═══════════════════════════════════════════════════════════════════════════════


class TestSeedStillDrivesPredictions:
    def test_varying_seed_moves_raw_confidences(self):
        a = run_pipeline(DATASET, n_records=60, seed=1, split_seed=1000,
                         data_root=DATA_ROOT)
        b = run_pipeline(DATASET, n_records=60, seed=2, split_seed=1000,
                         data_root=DATA_ROOT)
        assert a.raw_confidences != b.raw_confidences

    def test_varying_seed_leaves_detections_alone(self):
        """Detection is seed-free; only repair/uncertainty are seeded."""
        a = run_pipeline(DATASET, n_records=60, seed=1, split_seed=1000,
                         data_root=DATA_ROOT)
        b = run_pipeline(DATASET, n_records=60, seed=2, split_seed=1000,
                         data_root=DATA_ROOT)
        assert detection_fingerprint(a) == detection_fingerprint(b)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Determinism
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    def test_same_seed_pair_twice_is_identical(self):
        a = run_pipeline(DATASET, n_records=80, seed=7, split_seed=1234,
                         data_root=DATA_ROOT)
        b = run_pipeline(DATASET, n_records=80, seed=7, split_seed=1234,
                         data_root=DATA_ROOT)
        assert_predictions_identical(a, b)
        assert a.calibrated_probs == b.calibrated_probs
        assert a.corruption_probs == b.corruption_probs
        assert a.calibration_method == b.calibration_method
        assert a.split_seed == b.split_seed == 1234


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestSplitSeedValidation:
    @pytest.mark.parametrize("bad", [-1, -42])
    def test_negative_rejected(self, bad):
        with pytest.raises(ValueError, match="split_seed"):
            run_pipeline(DATASET, n_records=20, split_seed=bad, data_root=DATA_ROOT)

    @pytest.mark.parametrize("bad", ["x", 1.5, 0.0, [1], {"a": 1}])
    def test_non_int_rejected(self, bad):
        with pytest.raises(ValueError, match="split_seed"):
            run_pipeline(DATASET, n_records=20, split_seed=bad, data_root=DATA_ROOT)

    def test_zero_is_allowed(self):
        bundle = run_pipeline(DATASET, n_records=20, split_seed=0,
                              data_root=DATA_ROOT)
        assert bundle.split_seed == 0

    def test_numpy_int_is_allowed(self):
        import numpy as np

        bundle = run_pipeline(DATASET, n_records=20, split_seed=np.int64(5),
                              data_root=DATA_ROOT)
        assert bundle.split_seed == 5
        assert isinstance(bundle.split_seed, int)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. A small repeated_splits run end to end
# ═══════════════════════════════════════════════════════════════════════════════


class TestRepeatedSplitsSmoke:
    @pytest.fixture(scope="class")
    def result(self, bc_costs):
        return repeated_splits(
            DATASET, bc_costs, n_splits=3, n_records=N, data_root=DATA_ROOT,
            n_boot=500,
        )

    def test_split_seeds_are_the_arithmetic_sequence(self, result):
        assert result.split_seeds == [1000, 1001, 1002]
        assert result.n_splits == 3
        assert result.seed == 42
        assert result.dataset == DATASET

    def test_raw_fitted_in_every_split(self, result):
        assert result.n_fitted["raw"] == 3
        assert result.brier["raw"].n == 3
        assert result.ece["raw"].n == 3

    def test_intervals_have_n_equal_to_fitted_count(self, result):
        for method, n in result.n_fitted.items():
            assert result.brier[method].n == n
            assert result.ece[method].n == n

    def test_every_method_has_an_interval_even_unfitted(self, result):
        for method in ("raw", "logistic", "isotonic", "multivariate"):
            assert method in result.brier
            assert method in result.ece
            assert isinstance(str(result.brier[method]), str)

    def test_bounds_ordered(self, result):
        for iv in list(result.brier.values()) + list(result.ece.values()):
            assert iv.lo <= iv.mean <= iv.hi
        for iv in result.cost.values():
            assert iv.lo <= iv.mean <= iv.hi

    def test_policies_present(self, result):
        assert "calibrated_risk_aware" in result.cost
        assert "always_auto" in result.cost
        for policy, actions in result.actions.items():
            assert set(actions) == {"auto", "human", "abstain"}
            total = sum(a.mean for a in actions.values())
            assert total == pytest.approx(1.0, abs=1e-6)

    def test_cost_is_positive(self, result):
        assert result.cost["always_auto"].mean > 0

    def test_n_decided_recorded(self, result):
        assert result.n_decided > 0

    def test_deterministic_across_two_calls(self, bc_costs, result):
        again = repeated_splits(
            DATASET, bc_costs, n_splits=3, n_records=N, data_root=DATA_ROOT,
            n_boot=500,
        )
        assert again.to_dict() == result.to_dict()

    def test_normal_interval_variant_runs(self, bc_costs):
        res = repeated_splits(
            DATASET, bc_costs, n_splits=3, n_records=N, data_root=DATA_ROOT,
            interval="normal",
        )
        assert res.interval == "normal"
        assert res.brier["raw"].n == 3
        assert res.brier["raw"].lo <= res.brier["raw"].mean <= res.brier["raw"].hi
