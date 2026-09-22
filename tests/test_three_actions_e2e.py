"""
tests/test_three_actions_e2e.py
-------------------------------
End-to-end claims behind docs/three_actions_plan.md: with a learned
corruption probability q, all three decisions are used, for the right
cells, and nothing changes where the detector raises no false alarms.

These run the real pipeline on the shipped datasets. The numeric pins
come from a measured run (2026-09-22, seed 42) and are meant to FAIL if
the behaviour drifts — do not loosen them to make them pass; investigate.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pytest

from src.decision.policy import DecisionAction, DecisionPolicy
from src.pipeline import decide_all, evaluate_policies, run_pipeline
from src.risk.cost_model import CostConfig

COSTS_YAML = Path("configs/costs.yaml")
DATA_ROOT = "datasets"


def _by_method(bundle, decisions):
    out = defaultdict(Counter)
    for i in bundle.decision_indices():
        out[bundle.detections[i].detection_method][decisions[i].action] += 1
    return out


def _share(counter: Counter, action: DecisionAction) -> float:
    n = sum(counter.values())
    return counter[action] / n if n else 0.0


@pytest.fixture(scope="module")
def bc_costs():
    return CostConfig.from_yaml(COSTS_YAML, "breast_cancer")


@pytest.fixture(scope="module")
def bc_k15():
    return run_pipeline("breast_cancer", n_records=0, iqr_k=1.5, data_root=DATA_ROOT)


@pytest.fixture(scope="module")
def bc_k3():
    return run_pipeline("breast_cancer", n_records=0, data_root=DATA_ROOT)


@pytest.mark.integration
class TestBreastCancerTukeyFence:
    """k = 1.5: 195 near-certain false alarms next to 392 real errors."""

    def test_all_three_actions_used(self, bc_k15, bc_costs):
        d = decide_all(bc_k15, bc_costs)
        total = Counter(d[i].action for i in bc_k15.decision_indices())
        assert set(total) == set(DecisionAction), dict(total)
        assert total[DecisionAction.AUTO_REPAIR] >= 20
        assert total[DecisionAction.ABSTAIN] >= 150

    def test_outliers_abstained_missing_never(self, bc_k15, bc_costs):
        by = _by_method(bc_k15, decide_all(bc_k15, bc_costs))
        assert _share(by["iqr_outlier"], DecisionAction.ABSTAIN) >= 0.95, dict(by["iqr_outlier"])
        assert _share(by["missing_value"], DecisionAction.ABSTAIN) == 0.0, dict(by["missing_value"])
        assert _share(by["range_check"], DecisionAction.ABSTAIN) == 0.0

    def test_learned_q_separates_check_types(self, bc_k15):
        qs = defaultdict(list)
        for i, m in enumerate(bc_k15.detection_methods):
            qs[m].append(bc_k15.corruption_probs[i])
        assert max(qs["iqr_outlier"]) < 0.10
        assert min(qs["missing_value"]) > 0.95
        assert bc_k15.detection_calibration is not None
        assert bc_k15.detection_calibration.calibrated_brier < bc_k15.detection_calibration.raw_brier

    def test_policy_beats_both_baselines_and_abstains_only_on_clean_cells(self, bc_k15, bc_costs):
        ev = evaluate_policies(bc_k15, bc_costs, threshold=0.8)
        prop = ev.cost_reports["calibrated_risk_aware"]
        others = [r for k, r in ev.cost_reports.items() if k != "calibrated_risk_aware"]
        assert all(prop.total_realized_cost < o.total_realized_cost for o in others)
        # 195 abstentions; only the ~5 genuinely corrupted outliers are charged
        assert prop.n_abstain == 195
        assert prop.n_abstain_on_clean >= 185
        assert prop.n_abstain_on_corrupted <= 10


@pytest.mark.integration
class TestBreastCancerDefaultFenceUnchanged:
    """k = 3 (default): no false alarms, so learned q must change nothing."""

    def test_detection_count_pinned(self, bc_k3):
        assert bc_k3.n_records == 392
        assert "iqr_outlier" not in set(bc_k3.detection_methods)

    def test_decisions_identical_to_q_equals_one(self, bc_k3, bc_costs):
        pol = DecisionPolicy.from_cost_config(bc_costs)
        with_q = decide_all(bc_k3, bc_costs)
        without = [pol.decide(p, 1.0) for p in bc_k3.calibrated_probs]
        changed = [
            (i, bc_k3.detection_methods[i], round(bc_k3.corruption_probs[i], 3),
             without[i].action.value, with_q[i].action.value)
            for i in bc_k3.decision_indices() if with_q[i].action != without[i].action
        ]
        assert changed == [], f"learned q flipped {len(changed)} decisions: {changed[:5]}"

    def test_action_distribution_pinned(self, bc_k3, bc_costs):
        d = decide_all(bc_k3, bc_costs)
        total = Counter(d[i].action for i in bc_k3.decision_indices())
        assert total == {DecisionAction.AUTO_REPAIR: 52, DecisionAction.HUMAN_REVIEW: 303}, dict(total)

    def test_realised_cost_pinned(self, bc_k3, bc_costs):
        ev = evaluate_policies(bc_k3, bc_costs, threshold=0.8)
        assert ev.cost_reports["calibrated_risk_aware"].total_realized_cost == pytest.approx(1615.0)
        assert ev.cost_reports["always_auto"].total_realized_cost == pytest.approx(5800.0)


@pytest.mark.integration
class TestAdultOriginalConfig:
    """All Adult flags are false alarms; learned q ≈ 0 makes abstention win
    with Sruthi's original costs (10 / 2 / 5), no hand-set override."""

    @pytest.fixture(scope="class")
    def adult(self):
        return run_pipeline("adult", n_records=2000, data_root=DATA_ROOT)

    def test_config_is_original(self):
        c = CostConfig.from_yaml(COSTS_YAML, "adult")
        assert (c.c_wrong_repair, c.c_human_review, c.c_leave_error) == (10.0, 2.0, 5.0)

    def test_everything_abstained_and_free(self, adult):
        costs = CostConfig.from_yaml(COSTS_YAML, "adult")
        d = decide_all(adult, costs)
        assert {x.action for x in d} == {DecisionAction.ABSTAIN}
        assert max(adult.corruption_probs) < 0.05
        ev = evaluate_policies(adult, costs, threshold=0.8)
        prop = ev.cost_reports["calibrated_risk_aware"]
        assert prop.total_realized_cost == 0.0
        assert prop.n_abstain_on_clean == 2000
        assert ev.cost_reports["always_auto"].total_realized_cost == pytest.approx(20000.0)


@pytest.mark.integration
class TestSmartFactoryOutliersOn:
    """Outlier flags on: 35,639 false alarms next to 7,093 real missing
    values. Best repair p = 0.58, so AUTO must never fire."""

    @pytest.fixture(scope="class")
    def sf(self):
        return run_pipeline("smartfactory", n_records=0, flag_outliers=True, data_root=DATA_ROOT)

    def test_outliers_abstained_missing_reviewed_no_auto(self, sf):
        costs = CostConfig.from_yaml(COSTS_YAML, "smartfactory")
        by = _by_method(sf, decide_all(sf, costs))
        assert _share(by["iqr_outlier"], DecisionAction.ABSTAIN) >= 0.95
        assert _share(by["missing_value"], DecisionAction.HUMAN_REVIEW) == 1.0
        assert sum(c[DecisionAction.AUTO_REPAIR] for c in by.values()) == 0
