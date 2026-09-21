"""
tests/test_pipeline_runner.py
------------------------------
Phase 11 support: the shared end-to-end runner in src/pipeline/runner.py.

Covers
  - run_pipeline produces aligned, well-formed outputs on every dataset
  - decide_all is deterministic, cost-sensitive and ground-truth-free
  - evaluate_policies produces the three expected methods and a sweep
  - degenerate cases: Adult fallback, tiny n, invalid arguments
  - NO-LEAKAGE: decisions cannot depend on clean values by construction
    and, empirically, do not change when clean cell values are scrambled
    in a way that preserves the population-level statistics the profiler
    is allowed to use.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.decision.policy import DecisionAction, DecisionResult
from src.pipeline import (
    DATASETS,
    SKIP_COLUMNS,
    PipelineBundle,
    PolicyEvaluation,
    decide_all,
    evaluate_policies,
    run_pipeline,
)
from src.pipeline.runner import _calibrate_all
from src.calibration.calibrator import CalibrationSample
from src.risk.cost_model import CostConfig

DATA_ROOT = "datasets"
COSTS_YAML = Path("configs/costs.yaml")


@pytest.fixture(scope="module")
def bc_bundle() -> PipelineBundle:
    return run_pipeline("breast_cancer", n_records=150, seed=42, data_root=DATA_ROOT)


@pytest.fixture(scope="module")
def adult_bundle() -> PipelineBundle:
    return run_pipeline("adult", n_records=100, seed=42, data_root=DATA_ROOT)


@pytest.fixture(scope="module")
def bc_costs() -> CostConfig:
    return CostConfig.from_yaml(COSTS_YAML, "breast_cancer")


# ═══════════════════════════════════════════════════════════════════════════════
# run_pipeline
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunPipeline:
    def test_outputs_are_aligned(self, bc_bundle):
        b = bc_bundle
        n = b.n_records
        assert n == 150
        assert len(b.detections) == n
        assert len(b.repair_results) == n
        assert len(b.unc_records) == n
        assert len(b.cal_samples) == n
        assert len(b.calibrated_probs) == n
        for det, rr, unc, cs in zip(b.detections, b.repair_results, b.unc_records, b.cal_samples):
            assert det.row_id == rr.row_id == unc.row_id == cs.row_id
            assert det.column == rr.column == unc.column == cs.column

    def test_cap_respected_and_total_reported(self, bc_bundle):
        assert bc_bundle.n_detected_total >= bc_bundle.n_records
        assert bc_bundle.n_requested == 150

    def test_no_cap_when_zero(self):
        b = run_pipeline("breast_cancer", n_records=0, seed=42, data_root=DATA_ROOT)
        assert b.n_records == b.n_detected_total

    def test_probabilities_in_unit_interval(self, bc_bundle):
        assert all(0.0 <= p <= 1.0 for p in bc_bundle.calibrated_probs)
        assert all(0.0 <= p <= 1.0 for p in bc_bundle.raw_confidences)

    def test_breast_cancer_uses_a_fitted_calibrator(self, bc_bundle):
        assert bc_bundle.calibration_method in ("logistic", "isotonic", "multivariate")
        assert bc_bundle.calibration_note is None
        assert bc_bundle.comparison is not None
        assert bc_bundle.comparison.n_eval > 0

    def test_adult_falls_back_and_says_so(self, adult_bundle):
        b = adult_bundle
        assert "fallback" in b.calibration_method
        assert b.calibration_note is not None
        assert "Adult" in b.calibration_note or "base rate" in b.calibration_note
        # Every label is 0, so the base-rate probability is ~0 for all records.
        assert max(b.calibrated_probs) < 0.05

    def test_deterministic_across_runs(self):
        a = run_pipeline("breast_cancer", n_records=60, seed=7, data_root=DATA_ROOT)
        b = run_pipeline("breast_cancer", n_records=60, seed=7, data_root=DATA_ROOT)
        assert a.calibrated_probs == b.calibrated_probs
        assert [r.top_candidate for r in a.unc_records] == [r.top_candidate for r in b.unc_records]

    def test_seed_changes_mock_noise(self):
        a = run_pipeline("breast_cancer", n_records=60, seed=1, data_root=DATA_ROOT)
        b = run_pipeline("breast_cancer", n_records=60, seed=2, data_root=DATA_ROOT)
        assert a.raw_confidences != b.raw_confidences

    @pytest.mark.parametrize("dataset", DATASETS)
    def test_every_dataset_runs(self, dataset):
        b = run_pipeline(dataset, n_records=40, seed=42, data_root=DATA_ROOT)
        assert b.dataset == dataset
        assert b.n_records == 40
        assert len(b.calibrated_probs) == 40

    def test_unknown_dataset_rejected(self):
        with pytest.raises(ValueError):
            run_pipeline("not_a_dataset", data_root=DATA_ROOT)

    def test_invalid_hyperparameters_rejected(self):
        with pytest.raises(ValueError):
            run_pipeline("breast_cancer", n_samples=0, data_root=DATA_ROOT)
        with pytest.raises(ValueError):
            run_pipeline("breast_cancer", top_k=0, data_root=DATA_ROOT)

    def test_decision_indices_exclude_skip_columns(self, bc_bundle):
        idx = bc_bundle.decision_indices()
        skipped = set(SKIP_COLUMNS["breast_cancer"])
        assert all(bc_bundle.unc_records[i].column not in skipped for i in idx)
        assert len(idx) <= bc_bundle.n_records


class TestCalibrateAll:
    def test_too_few_samples_returns_raw(self):
        samples = [CalibrationSample("t", i, "c", 0.3 + 0.1 * i, i % 2) for i in range(2)]
        method, note, probs = _calibrate_all(samples, seed=0, eval_fraction=0.3)
        assert method.startswith("raw")
        assert note is not None
        assert probs == pytest.approx([0.3, 0.4])

    def test_constant_labels_fallback(self):
        samples = [CalibrationSample("t", i, "c", 0.6, 0, 0.9, 0.2) for i in range(30)]
        method, note, probs = _calibrate_all(samples, seed=0, eval_fraction=0.3)
        assert "fallback" in method
        assert note is not None
        assert len(probs) == 30
        assert max(probs) < 0.01

    def test_returns_prob_per_sample_in_order(self):
        rng = np.random.default_rng(3)
        samples = []
        for i in range(120):
            c = float(rng.uniform(0.3, 0.9))
            samples.append(CalibrationSample("t", i, "c", c, int(rng.random() < c),
                                             float(rng.uniform(0.5, 1)), float(rng.uniform(0, 1))))
        method, note, probs = _calibrate_all(samples, seed=0, eval_fraction=0.3)
        assert method in ("logistic", "isotonic", "multivariate")
        assert note is None
        assert len(probs) == 120
        assert all(0.0 <= p <= 1.0 for p in probs)


# ═══════════════════════════════════════════════════════════════════════════════
# decide_all
# ═══════════════════════════════════════════════════════════════════════════════


class TestDecideAll:
    def test_one_decision_per_record(self, bc_bundle, bc_costs):
        d = decide_all(bc_bundle, bc_costs)
        assert len(d) == bc_bundle.n_records
        assert all(isinstance(x, DecisionResult) for x in d)

    def test_decisions_follow_calibrated_probability(self, bc_bundle, bc_costs):
        for p, d in zip(bc_bundle.calibrated_probs, decide_all(bc_bundle, bc_costs)):
            assert d.calibrated_probability == pytest.approx(p)

    def test_costs_change_decisions_same_probabilities(self, bc_bundle):
        cheap_wrong = CostConfig(c_wrong_repair=1, c_human_review=5, c_leave_error=10,
                                 dataset_name="breast_cancer")
        cheap_abstain = CostConfig(c_wrong_repair=50, c_human_review=5, c_leave_error=0,
                                   dataset_name="breast_cancer")
        a = {d.action for d in decide_all(bc_bundle, cheap_wrong)}
        b = {d.action for d in decide_all(bc_bundle, cheap_abstain)}
        assert a == {DecisionAction.AUTO_REPAIR}
        assert DecisionAction.ABSTAIN in b
        assert DecisionAction.HUMAN_REVIEW not in b

    def test_adult_all_abstain_under_shipped_config(self, adult_bundle):
        costs = CostConfig.from_yaml(COSTS_YAML, "adult")
        actions = {d.action for d in decide_all(adult_bundle, costs)}
        assert actions == {DecisionAction.ABSTAIN}

    def test_signature_has_no_ground_truth(self):
        params = list(inspect.signature(decide_all).parameters)
        assert params == ["bundle", "costs"]
        assert not any("clean" in p.lower() for p in params)


# ═══════════════════════════════════════════════════════════════════════════════
# evaluate_policies
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluatePolicies:
    def test_three_methods_and_sweep(self, bc_bundle, bc_costs):
        ev = evaluate_policies(bc_bundle, bc_costs, threshold=0.8)
        assert isinstance(ev, PolicyEvaluation)
        names = [r.method_name for r in ev.rows]
        assert names[0] == "calibrated_risk_aware"
        assert "always_auto" in names
        assert any(n.startswith("confidence_threshold_") for n in names)
        assert set(ev.threshold_sweep) == {0.5, 0.6, 0.7, 0.8, 0.9}
        assert set(ev.cost_reports) == set(ev.repair_reports) == set(ev.records)

    def test_rates_sum_to_one(self, bc_bundle, bc_costs):
        for r in evaluate_policies(bc_bundle, bc_costs).rows:
            assert r.auto_rate + r.human_rate + r.abstain_rate == pytest.approx(1.0, abs=1e-6)

    def test_always_auto_is_all_auto(self, bc_bundle, bc_costs):
        ev = evaluate_policies(bc_bundle, bc_costs)
        assert ev.cost_reports["always_auto"].auto_rate == pytest.approx(1.0)

    def test_skip_columns_excluded_from_every_method(self, bc_bundle, bc_costs):
        ev = evaluate_policies(bc_bundle, bc_costs)
        skipped = set(SKIP_COLUMNS["breast_cancer"])
        for recs in ev.records.values():
            assert all(r.column not in skipped for r in recs)
            assert len(recs) == len(bc_bundle.decision_indices())

    def test_custom_sweep(self, bc_bundle, bc_costs):
        ev = evaluate_policies(bc_bundle, bc_costs, threshold=0.65, sweep=[0.65, 0.75])
        assert set(ev.threshold_sweep) == {0.65, 0.75}

    def test_proposed_beats_always_auto_on_breast_cancer(self, bc_bundle, bc_costs):
        """Reproduces the demo_risk finding at demo scale; not a tuned number."""
        ev = evaluate_policies(bc_bundle, bc_costs)
        assert (ev.cost_reports["calibrated_risk_aware"].total_realized_cost
                < ev.cost_reports["always_auto"].total_realized_cost)


# ═══════════════════════════════════════════════════════════════════════════════
# No-leakage
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoLeakage:
    def test_bundle_separates_decision_inputs_from_ground_truth(self, bc_bundle):
        """Decision-side records must not carry clean values."""
        for rec in bc_bundle.unc_records:
            assert not any("clean" in f.lower() for f in rec.__dataclass_fields__)
        for det in bc_bundle.detections:
            assert not any("clean" in f.lower() for f in det.__dataclass_fields__)

    def test_decisions_invariant_to_scrambled_ground_truth(self, tmp_path):
        """
        Scramble clean values within each column (a row permutation per
        column). Column-level statistics — the only clean-derived
        information the profiler may use — are unchanged, so detection,
        repair and uncertainty must be identical. Only the 0/1 labels (and
        hence calibration) may differ.
        """
        src = Path(DATA_ROOT) / "breast_cancer"
        dst = tmp_path / "breast_cancer"
        dst.mkdir()
        clean = pd.read_csv(src / "clean.csv")
        rng = np.random.default_rng(0)
        scrambled = clean.copy()
        for col in scrambled.columns:
            scrambled[col] = scrambled[col].to_numpy()[rng.permutation(len(scrambled))]
        scrambled.to_csv(dst / "clean.csv", index=False)
        (dst / "dirty.csv").write_bytes((src / "dirty.csv").read_bytes())

        real = run_pipeline("breast_cancer", n_records=120, seed=42, data_root=DATA_ROOT)
        fake = run_pipeline("breast_cancer", n_records=120, seed=42, data_root=tmp_path)

        assert [(d.row_id, d.column, d.error_type) for d in real.detections] == \
               [(d.row_id, d.column, d.error_type) for d in fake.detections]
        assert [(str(r.top_candidate), r.raw_confidence, r.self_consistency, r.normalized_entropy)
                for r in real.unc_records] == \
               [(str(r.top_candidate), r.raw_confidence, r.self_consistency, r.normalized_entropy)
                for r in fake.unc_records]
        # Labels are allowed to differ — that is the one legitimate use.
        assert [s.correct for s in real.cal_samples] != [s.correct for s in fake.cal_samples]

    def test_run_pipeline_never_passes_clean_df_to_decision_stages(self):
        """Static check on the runner source: the only calls that receive
        clean_df are the profiler, the calibration builder and evaluation."""
        import src.pipeline.runner as runner
        source = inspect.getsource(runner)
        allowed = {
            "loader.load(dataset)",  # produces clean_df; does not consume it
            "profile_dataset(dataset, loader.discover(dataset), clean_df, dirty_df)",
            "CalibrationBuilder().build(unc_records, clean_df, dirty_df, dataset)",
            "CostEvaluator.from_cost_config(bundle.clean_df, costs)",
            "RepairEvaluator(bundle.clean_df, bundle.dirty_df, bundle.dataset)",
        }
        for line in source.splitlines():
            stripped = line.strip()
            if "clean_df" in stripped and "(" in stripped and "=" in stripped.split("(")[0]:
                call = stripped.split("=", 1)[1].strip()
                assert any(a in call for a in allowed), f"unexpected clean_df use: {stripped}"
