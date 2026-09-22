"""
experiments/smoothing_ablation.py
---------------------------------
Is the calibrated-abstention result an artefact of the smoothing prior?

The detection calibrator estimates, per detector check type,

    q_group = (k + s) / (n + 2s)        s = smoothing strength

A Beta(s, s) prior centred on 0.5. s is a free parameter, so the
conclusions must be shown to be robust to it. This script sweeps

    s ∈ {0 (raw rates), 0.5 (Jeffreys), 1 (default, Laplace), 5, 20}

and reports, per dataset/detector setting: the learned q per check type,
the action distribution, the realised cost, and the detection-calibration
Brier score. A conclusion that survives all five is not a smoothing
artefact.

Also reports the q = 1 baseline (no detection calibration = the original
pipeline's implicit assumption) in the same table, so the effect of the
mechanism is separable from the effect of its hyper-parameter.

Run:
    PYTHONIOENCODING=utf-8 python experiments/smoothing_ablation.py
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.decision.policy import DecisionAction, DecisionPolicy  # noqa: E402
from src.calibration.detection_calibrator import (  # noqa: E402
    DetectionLabelBuilder,
    fit_detection_calibrator,
)
from src.pipeline import run_pipeline  # noqa: E402
from src.risk.cost_model import CostConfig  # noqa: E402

COSTS_YAML = Path("configs/costs.yaml")
SMOOTHINGS = [0.0, 0.5, 1.0, 5.0, 20.0]
SCENARIOS = [
    ("breast_cancer", 3.0, None, 0),
    ("breast_cancer", 1.5, None, 0),
    ("smartfactory", 3.0, True, 0),
    ("adult", 3.0, None, 2000),
]


def _is_corrupted(bundle, i: int) -> bool:
    r = bundle.unc_records[i]
    clean = bundle.clean_df.reset_index(drop=True).at[r.row_id, r.column]
    dirty = bundle.dirty_df.reset_index(drop=True).at[r.row_id, r.column]
    if pd.isna(clean) and pd.isna(dirty):
        return False
    if pd.isna(clean) or pd.isna(dirty):
        return True
    try:
        return not np.isclose(float(clean), float(dirty))
    except (TypeError, ValueError):
        return str(clean).strip() != str(dirty).strip()


def realised_cost(bundle, idx, actions, costs, corrupted) -> float:
    """Realised cost under the paper's definition of each penalty.

    C_wrong   charged when an applied repair is wrong
    C_human   charged per cell routed to a reviewer
    C_leave   charged when a GENUINELY corrupted cell is left unrepaired
    """
    total = 0.0
    for i in idx:
        a = actions[i]
        if a == DecisionAction.AUTO_REPAIR:
            total += 0.0 if bundle.cal_samples[i].correct == 1 else costs.c_wrong_repair
        elif a == DecisionAction.HUMAN_REVIEW:
            total += costs.c_human_review
        else:
            total += costs.c_leave_error if corrupted[i] else 0.0
    return total


def run_scenario(dataset: str, iqr_k: float, flag_outliers, cap: int) -> None:
    bundle = run_pipeline(dataset, n_records=cap, iqr_k=iqr_k, flag_outliers=flag_outliers)
    costs = CostConfig.from_yaml(COSTS_YAML, dataset)
    policy = DecisionPolicy.from_cost_config(costs)
    idx = bundle.decision_indices()
    corrupted = {i: _is_corrupted(bundle, i) for i in idx}
    labels = DetectionLabelBuilder().build(bundle.detections, bundle.clean_df, bundle.dirty_df)
    methods = bundle.detection_methods
    groups = sorted(set(methods))

    print(f"\n{'=' * 100}")
    print(f"  {dataset}  iqr_k={iqr_k}  flag_outliers={bundle.flag_outliers}  "
          f"n_decided={len(idx)}  costs {costs.c_wrong_repair:g}/"
          f"{costs.c_human_review:g}/{costs.c_leave_error:g}")
    print(f"  genuinely corrupted among decided cells: "
          f"{sum(corrupted.values())} / {len(idx)}")
    print(f"{'-' * 100}")
    header = (f"  {'smoothing':<12} {'AUTO':>6} {'HUMAN':>7} {'ABSTAIN':>8} "
              f"{'cost':>10} {'det.Brier':>10}   q per check type")
    print(header)
    print(f"  {'-' * 11} {'-' * 6} {'-' * 7} {'-' * 8} {'-' * 10} {'-' * 10}   {'-' * 30}")

    # Baseline: no detection calibration at all (q = 1 everywhere).
    actions = {i: policy.decide(bundle.calibrated_probs[i], 1.0).action for i in idx}
    c = Counter(actions.values())
    print(f"  {'q=1 (orig)':<12} {c[DecisionAction.AUTO_REPAIR]:>6} "
          f"{c[DecisionAction.HUMAN_REVIEW]:>7} {c[DecisionAction.ABSTAIN]:>8} "
          f"{realised_cost(bundle, idx, actions, costs, corrupted):>10.1f} "
          f"{'—':>10}   all 1.000")

    for s in SMOOTHINGS:
        cal, report = fit_detection_calibrator(
            bundle.detections, labels, seed=bundle.seed,
            eval_fraction=0.30, dataset_name=dataset, smoothing=s,
        )
        q = cal.predict(bundle.detections)
        actions = {i: policy.decide(bundle.calibrated_probs[i], float(q[i])).action for i in idx}
        c = Counter(actions.values())
        per_group = {g: float(np.mean([q[i] for i in range(len(methods)) if methods[i] == g]))
                     for g in groups}
        qs = "  ".join(f"{g.replace('_check', '').replace('_value', '')}={v:.3f}"
                       for g, v in per_group.items())
        print(f"  {('s=' + format(s, 'g')):<12} {c[DecisionAction.AUTO_REPAIR]:>6} "
              f"{c[DecisionAction.HUMAN_REVIEW]:>7} {c[DecisionAction.ABSTAIN]:>8} "
              f"{realised_cost(bundle, idx, actions, costs, corrupted):>10.1f} "
              f"{report.calibrated_brier:>10.4f}   {qs}")


def main() -> None:
    print("SMOOTHING ABLATION — is calibrated abstention robust to the prior?")
    print("Costs are experimental penalty units. C_leave is charged only when a")
    print("genuinely corrupted cell is left unrepaired (see docs/cost_semantics.md).")
    for dataset, k, flags, cap in SCENARIOS:
        run_scenario(dataset, k, flags, cap)
    print()


if __name__ == "__main__":
    main()
