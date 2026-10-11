"""
demo_calibration.py
--------------------
Phase 6 CLI demo: Probability calibration for repair predictions.

Run:
    python3 demo_calibration.py

For each dataset (Breast Cancer, SmartFactory, Adult):
  1. Detect errors, repair, estimate uncertainty (no clean_df)
  2. Build calibration samples (labels from clean_df ONLY)
  3. Fit logistic + isotonic calibrators on training split
  4. Evaluate all three methods on held-out eval split
  5. Show before/after examples (raw → calibrated)
  6. Save reliability diagrams to results/figures/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.console import enable_unicode_stdout
from src.calibration.calibrator import (
    CalibrationBuilder,
    IsotonicCalibrator,
    LogisticCalibrator,
    RawBaseline,
    calibrate_dataset,
    print_calibration_report,
    train_eval_split,
)
from src.calibration.plots import plot_brier_comparison, plot_reliability_diagram
from src.data.loader import DataLoader
from src.data.profiler import profile_dataset
from src.detection.detector import Detector
from src.repair.mock_repair import MockRepairer
from src.uncertainty.estimator import UncertaintyEstimator

SEED = 42
N_SAMPLES = 5
FIGURE_DIR = Path("results/figures")
DATASETS = ["breast_cancer", "smartfactory", "adult"]

# Cap for demo speed
MAX_DETECT = 500


def run_dataset(name: str, loader: DataLoader):
    w = 72
    print(f"\n{'═' * w}")
    print(f"  DATASET: {name.upper()}  —  Phase 6 Calibration")
    print(f"{'═' * w}")

    # ── Load and profile ──────────────────────────────────────────────────
    clean_df, dirty_df = loader.load(name)
    profile = profile_dataset(name, loader.discover(name), clean_df, dirty_df)

    # ── Detect (no clean_df) ──────────────────────────────────────────────
    flag_outliers = name != "smartfactory"
    detector = Detector(profile, flag_statistical_outliers=flag_outliers)
    records = detector.detect(dirty_df)[:MAX_DETECT]
    print(f"  Using {len(records)} detection records for calibration")

    if not records:
        print("  No errors detected — skipping.")
        return None

    # ── Repair + uncertainty (no clean_df) ───────────────────────────────
    repairer = MockRepairer(seed=SEED, top_k=3)
    repair_results = repairer.repair_batch(records, dirty_df, profile)
    estimator = UncertaintyEstimator(base_seed=SEED, n_samples=N_SAMPLES)
    unc_records = estimator.estimate_batch(repair_results, dirty_df, profile)

    # ── Build calibration samples (labels from clean_df ONLY) ─────────────
    builder = CalibrationBuilder()
    cal_samples = builder.build(unc_records, clean_df, dirty_df, name)
    n_pos = sum(s.correct for s in cal_samples)
    print(f"  Calibration samples: {len(cal_samples)}  |  "
          f"positive: {n_pos} ({100*n_pos/len(cal_samples):.1f}%)")

    # ── Calibrate ─────────────────────────────────────────────────────────
    comparison = calibrate_dataset(cal_samples, name, eval_fraction=0.3, seed=SEED)

    # ── Print report ──────────────────────────────────────────────────────
    print_calibration_report(comparison)

    # ── Show before/after examples (EVALUATION ONLY) ─────────────────────
    if comparison.logistic_fitted and comparison.n_eval > 0:
        train_set, eval_set = train_eval_split(cal_samples, eval_fraction=0.3, seed=SEED)
        X_eval = np.array([s.raw_confidence for s in eval_set])
        y_eval = np.array([s.correct for s in eval_set])

        log_cal = LogisticCalibrator()
        iso_cal = IsotonicCalibrator()
        X_train = np.array([s.raw_confidence for s in train_set])
        y_train = np.array([s.correct for s in train_set], dtype=float)
        log_cal.fit(X_train, y_train)
        iso_cal.fit(X_train, y_train)

        print(f"  Sample predictions (first 8 eval examples):")
        print(f"  {'Row':<5} {'Column':<22} {'Raw conf':>9} "
              f"{'Logistic':>9} {'Isotonic':>9} {'Correct'}")
        print(f"  {'─'*5} {'─'*22} {'─'*9} {'─'*9} {'─'*9} {'─'*7}")

        sample_limit = min(8, len(eval_set))
        log_preds = log_cal.predict(X_eval[:sample_limit])
        iso_preds = (
            iso_cal.predict(X_eval[:sample_limit]) if iso_cal.is_fitted()
            else log_preds
        )

        for i in range(sample_limit):
            s = eval_set[i]
            correct_str = "CORRECT" if s.correct == 1 else "wrong  "
            print(
                f"  {s.row_id:<5} {s.column:<22} {s.raw_confidence:>9.4f} "
                f"{log_preds[i]:>9.4f} {iso_preds[i]:>9.4f} {correct_str}"
            )
        print(f"\n  [Ground truth shown for EVALUATION ONLY]")

    # ── Save reliability diagram ──────────────────────────────────────────
    fig_path = plot_reliability_diagram(comparison, FIGURE_DIR)
    print(f"\n  Reliability diagram saved: {fig_path}")

    return comparison


def main() -> None:
    enable_unicode_stdout()
    loader = DataLoader("datasets")
    comparisons = []

    for name in DATASETS:
        try:
            comp = run_dataset(name, loader)
            if comp is not None:
                comparisons.append(comp)
        except Exception as exc:
            print(f"\n[{name}] error: {exc}")
            import traceback
            traceback.print_exc()

    # ── Summary table ─────────────────────────────────────────────────────
    if comparisons:
        w = 72
        print(f"\n{'═' * w}")
        print(f"  CALIBRATION SUMMARY TABLE")
        print(f"{'─' * w}")
        print(f"  {'Dataset':<18} {'Method':<12} {'Brier':>9} {'ECE':>9} {'Fitted':>8}")
        print(f"  {'─'*18} {'─'*12} {'─'*9} {'─'*9} {'─'*8}")
        for comp in comparisons:
            ds = comp.dataset_name
            print(f"  {ds:<18} {'raw':<12} {comp.raw_brier:>9.4f} {comp.raw_ece:>9.4f} {'(baseline)':>8}")
            for method in ("logistic", "isotonic", "multivariate"):
                fitted_s = "yes" if comp.fitted(method) else "no"
                print(f"  {'':<18} {method:<12} {comp.metric_str(method, 'brier'):>9} {comp.metric_str(method, 'ece'):>9} {fitted_s:>8}")

        # Brier comparison plot
        valid = [c for c in comparisons if c.logistic_fitted]
        if len(valid) >= 2:
            fig_path = plot_brier_comparison(comparisons, FIGURE_DIR)
            print(f"\n  Brier comparison chart saved: {fig_path}")

        print(f"{'═' * w}")

    print(f"\n  Phase 6 demo complete. Figures saved to {FIGURE_DIR}/\n")


if __name__ == "__main__":
    main()
