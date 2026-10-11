"""
demo_risk.py
------------
Phase 7 CLI demo: Risk and cost modeling for repair decisions.

Run:
    python3 demo_risk.py

For each dataset (Breast Cancer, SmartFactory, Adult):
  1. Detect errors, repair, estimate uncertainty (no clean_df)
  2. Calibrate probabilities (logistic or isotonic)
  3. Load cost config from configs/costs.yaml
  4. Compute expected costs and recommended action per cell
  5. Print sensitivity analysis and analytical thresholds
  6. Evaluate empirical costs against ground truth (clean_df only here)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.console import enable_unicode_stdout
from src.calibration.calibrator import (
    CalibrationBuilder,
    LogisticCalibrator,
    IsotonicCalibrator,
    train_eval_split,
)
from src.data.loader import DataLoader
from src.data.profiler import profile_dataset
from src.detection.detector import Detector
from src.repair.mock_repair import MockRepairer
from src.risk.cost_model import (
    Action,
    CostConfig,
    RiskCalculator,
)
from src.risk.evaluation import RiskEvaluator, print_risk_report
from src.uncertainty.estimator import UncertaintyEstimator

SEED = 42
N_SAMPLES = 5
COSTS_YAML = Path("configs/costs.yaml")
DATASETS = ["breast_cancer", "smartfactory", "adult"]

# Per-dataset columns to skip for risk decisions (ID-like columns)
SKIP_COLUMNS = {
    "breast_cancer": ["Unnamed: 0", "Sample code number"],
    "adult": [],
    "smartfactory": [],
}

# Cap for demo speed
MAX_DETECT = 500


def _best_calibrator(train_set, eval_set):
    """Fit logistic + isotonic and return the better one (lower Brier) on eval."""
    from sklearn.metrics import brier_score_loss

    X_train = np.array([s.raw_confidence for s in train_set])
    y_train = np.array([s.correct for s in train_set], dtype=float)
    X_eval = np.array([s.raw_confidence for s in eval_set])
    y_eval = np.array([s.correct for s in eval_set], dtype=float)

    log_cal = LogisticCalibrator()
    log_cal.fit(X_train, y_train)

    iso_cal = IsotonicCalibrator()
    iso_cal.fit(X_train, y_train)

    log_preds = log_cal.predict(X_eval)
    iso_preds = iso_cal.predict(X_eval) if iso_cal.is_fitted() else log_preds

    log_brier = brier_score_loss(y_eval, log_preds)
    iso_brier = brier_score_loss(y_eval, iso_preds)

    if iso_cal.is_fitted() and iso_brier <= log_brier:
        return iso_cal, "isotonic", iso_brier
    return log_cal, "logistic", log_brier


def run_dataset(name: str, loader: DataLoader):
    w = 72
    print(f"\n{'═' * w}")
    print(f"  DATASET: {name.upper()}  —  Phase 7 Risk & Cost Modeling")
    print(f"{'═' * w}")

    # ── Load and profile ──────────────────────────────────────────────────
    clean_df, dirty_df = loader.load(name)
    profile = profile_dataset(name, loader.discover(name), clean_df, dirty_df)

    # ── Detect (no clean_df) ──────────────────────────────────────────────
    flag_outliers = name != "smartfactory"
    detector = Detector(profile, flag_statistical_outliers=flag_outliers)
    records = detector.detect(dirty_df)[:MAX_DETECT]
    print(f"  Detections: {len(records)}")

    if not records:
        print("  No errors detected — skipping.")
        return

    # ── Repair + uncertainty (no clean_df) ───────────────────────────────
    repairer = MockRepairer(seed=SEED, top_k=3)
    repair_results = repairer.repair_batch(records, dirty_df, profile)
    estimator = UncertaintyEstimator(base_seed=SEED, n_samples=N_SAMPLES)
    unc_records = estimator.estimate_batch(repair_results, dirty_df, profile)

    # ── Build calibration samples (labels from clean_df ONLY) ─────────────
    builder = CalibrationBuilder()
    cal_samples = builder.build(unc_records, clean_df, dirty_df, name)

    if not cal_samples:
        print("  No calibration samples — skipping.")
        return

    # ── Train calibrator ──────────────────────────────────────────────────
    train_set, eval_set = train_eval_split(cal_samples, eval_fraction=0.3, seed=SEED)

    if len(train_set) < 3 or len(eval_set) < 1:
        print("  Not enough data for calibration — skipping.")
        return

    calibrator, cal_method, cal_brier = _best_calibrator(train_set, eval_set)
    print(f"  Calibration method: {cal_method}  (eval Brier={cal_brier:.4f})")

    # ── Load cost config ──────────────────────────────────────────────────
    costs = CostConfig.from_yaml(COSTS_YAML, name)
    print(f"  Cost config: wrong={costs.c_wrong_repair}, "
          f"review={costs.c_human_review}, leave={costs.c_leave_error}")

    # ── Analytical thresholds ─────────────────────────────────────────────
    calc = RiskCalculator(costs)
    thresholds = calc.analytical_thresholds()
    print(f"\n  Analytical thresholds:")
    print(f"    ACCEPT > REVIEW when p > {thresholds.p_accept_vs_review:.4f}")
    print(f"    ACCEPT > LEAVE  when p > {thresholds.p_accept_vs_leave:.4f}")
    print(f"    Review cheaper than Leave: {thresholds.review_cheaper_than_leave}")
    print(f"    Policy: {thresholds.recommended_policy}")

    # ── Sensitivity table (abbreviated) ───────────────────────────────────
    print(f"\n  Sensitivity analysis (p → action):")
    print(f"  {'p':>6}  {'E[accept]':>10} {'E[review]':>10} {'E[leave]':>10}  {'best_action'}")
    print(f"  {'─'*6}  {'─'*10} {'─'*10} {'─'*10}  {'─'*14}")
    pts = calc.sensitivity_over_p(n_steps=11)
    for pt in pts:
        print(f"  {pt.p:>6.2f}  {pt.cost_accept:>10.4f} {pt.cost_review:>10.4f} "
              f"{pt.cost_leave:>10.4f}  {pt.best_action.value}")

    # ── Calibrate all uncertainty records ─────────────────────────────────
    all_confidences = np.array([s.raw_confidence for s in cal_samples])
    cal_probs = calibrator.predict(all_confidences)

    # ── Compute risk records ───────────────────────────────────────────────
    # Map cal_samples back to unc_records order (both are in same order)
    skip = SKIP_COLUMNS.get(name, [])
    risk_records = calc.compute_risk_batch(unc_records, list(cal_probs), skip_columns=skip)

    if not risk_records:
        print("  No risk records after filtering — skipping empirical evaluation.")
        return

    # ── Show sample risk decisions ─────────────────────────────────────────
    print(f"\n  Sample risk decisions (first 10 risk records):")
    print(f"  {'Row':<5} {'Column':<22} {'Cal p':>7} {'E[acc]':>7} "
          f"{'E[rev]':>7} {'E[lv]':>7}  Action")
    print(f"  {'─'*5} {'─'*22} {'─'*7} {'─'*7} {'─'*7} {'─'*7}  {'─'*14}")
    for rr in risk_records[:10]:
        print(
            f"  {rr.row_id:<5} {rr.column:<22} {rr.calibrated_probability:>7.4f} "
            f"{rr.expected_cost_accept:>7.4f} {rr.expected_cost_review:>7.4f} "
            f"{rr.expected_cost_leave:>7.4f}  {rr.lowest_cost_action.value}"
        )

    # ── Action distribution ────────────────────────────────────────────────
    action_dist = {a: 0 for a in Action}
    for rr in risk_records:
        action_dist[rr.lowest_cost_action] += 1
    total = len(risk_records)
    print(f"\n  Action distribution ({total} records):")
    for action, count in action_dist.items():
        pct = 100 * count / max(total, 1)
        print(f"    {action.value:<18}: {count:>5}  ({pct:5.1f}%)")

    # ── Empirical cost evaluation (clean_df used ONLY here) ───────────────
    evaluator = RiskEvaluator(clean_df, dirty_df, name)
    report = evaluator.evaluate(risk_records, costs)
    print_risk_report(report)


def main() -> None:
    enable_unicode_stdout()
    loader = DataLoader("datasets")

    for name in DATASETS:
        try:
            run_dataset(name, loader)
        except Exception as exc:
            print(f"\n[{name}] error: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\n  Phase 7 demo complete.\n")


if __name__ == "__main__":
    main()
