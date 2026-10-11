"""
demo_uncertainty.py
--------------------
Phase 5 CLI demo: Uncertainty signals for detected errors in each dataset.

Run:
    python3 demo_uncertainty.py

Shows, for each dataset:
  - A sample of individual uncertainty records (with ground-truth for display)
  - Correlation analysis of signals vs. correctness
  - Distribution summary of each signal

No clean ground truth is passed to the uncertainty estimator.
Ground truth appears ONLY in the evaluation section.
"""

from __future__ import annotations

from src.console import enable_unicode_stdout
from src.data.loader import DataLoader
from src.data.profiler import profile_dataset
from src.detection.detector import Detector
from src.repair.mock_repair import MockRepairer
from src.uncertainty.estimator import UncertaintyEstimator
from src.uncertainty.evaluation import UncertaintyCorrelationAnalysis

DATASETS = ["breast_cancer", "smartfactory", "adult"]
SEED = 42
N_SAMPLES = 5
SAMPLE_DISPLAY = 8
MAX_RECORDS = 200   # cap for demo speed


def run_dataset(name: str, loader: DataLoader) -> None:
    w = 72
    print(f"\n{'═' * w}")
    print(f"  DATASET: {name.upper()}  —  Phase 5 Uncertainty Demo")
    print(f"{'═' * w}")

    # ── Load and profile ──────────────────────────────────────────────────
    clean_df, dirty_df = loader.load(name)
    profile = profile_dataset(name, loader.discover(name), clean_df, dirty_df)

    # ── Detect ────────────────────────────────────────────────────────────
    flag_outliers = name != "smartfactory"
    detector = Detector(profile, flag_statistical_outliers=flag_outliers)
    records = detector.detect(dirty_df)

    if not records:
        print("  No errors detected — skipping.")
        return

    capped = records[:MAX_RECORDS]
    print(f"  Detected {len(records)} errors (using first {len(capped)} for demo)")

    # ── Repair (no clean_df) ──────────────────────────────────────────────
    repairer = MockRepairer(seed=SEED, top_k=3)
    repair_results = repairer.repair_batch(capped, dirty_df, profile)

    # ── Uncertainty estimation (no clean_df) ─────────────────────────────
    estimator = UncertaintyEstimator(base_seed=SEED, n_samples=N_SAMPLES)
    unc_records = estimator.estimate_batch(repair_results, dirty_df, profile)

    # ── Sample display ────────────────────────────────────────────────────
    print(f"\n  Sample uncertainty records (first {SAMPLE_DISPLAY}):")
    hdr = f"  {'Col':<20} {'Error':<18} {'Dirty':<10} {'Top-cand':<12} {'Conf':>6} {'Cons':>6} {'H_n':>6}"
    print(hdr)
    print(f"  {'─' * 20} {'─' * 18} {'─' * 10} {'─' * 12} {'─' * 6} {'─' * 6} {'─' * 6}")

    for ur in unc_records[:SAMPLE_DISPLAY]:
        dirty_str = str(ur.dirty_value)[:8]
        cand_str = str(ur.top_candidate)[:10]
        print(
            f"  {ur.column:<20} {ur.error_type:<18} {dirty_str:<10} "
            f"{cand_str:<12} {ur.raw_confidence:>6.3f} "
            f"{ur.self_consistency:>6.3f} {ur.normalized_entropy:>6.3f}"
        )

    # ── Signal distribution summary ───────────────────────────────────────
    confs = [u.raw_confidence for u in unc_records]
    cons = [u.self_consistency for u in unc_records]
    entrs = [u.normalized_entropy for u in unc_records]

    print(f"\n  Signal distribution across {len(unc_records)} cells:")
    print(f"  {'Signal':<22} {'Min':>6} {'Mean':>6} {'Max':>6} {'Std':>6}")
    print(f"  {'─' * 22} {'─' * 6} {'─' * 6} {'─' * 6} {'─' * 6}")
    for label, vals in [("raw_confidence", confs), ("self_consistency", cons), ("norm_entropy", entrs)]:
        n = len(vals)
        mn = min(vals)
        mx = max(vals)
        mean = sum(vals) / n
        std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
        print(f"  {label:<22} {mn:>6.3f} {mean:>6.3f} {mx:>6.3f} {std:>6.3f}")

    # ── Evaluation (uses clean_df — ONLY here) ────────────────────────────
    print(f"\n  Correlation with correctness (ground truth used here only):")
    analysis = UncertaintyCorrelationAnalysis(
        clean_df, dirty_df, name, n_bins=4
    )
    report = analysis.analyse(unc_records)

    print(f"  Overall top-1 accuracy : {report.overall_top1_accuracy:.3f}")

    def _fmt(r):
        return f"{r:+.3f}" if r is not None else "N/A"

    print(f"  Pearson r(conf↔correct): {_fmt(report.pearson_conf_accuracy)}")
    print(f"  Pearson r(cons↔correct): {_fmt(report.pearson_cons_accuracy)}")
    print(f"  Pearson r(entr↔correct): {_fmt(report.pearson_entropy_accuracy)}  (expect ≤ 0)")

    if report.by_confidence_bin:
        print(f"\n  Accuracy by confidence quartile:")
        for b in report.by_confidence_bin:
            bar = "█" * int(b.accuracy * 20)
            print(f"    conf {b.bin_label:<18} n={b.n_cells:4d}  acc={b.accuracy:.3f}  {bar}")


def main() -> None:
    enable_unicode_stdout()
    loader = DataLoader("datasets")
    for name in DATASETS:
        try:
            run_dataset(name, loader)
        except Exception as exc:
            print(f"\n[{name}] skipped: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\n{'═' * 72}")
    print("  Phase 5 demo complete.")
    print(f"{'═' * 72}\n")


if __name__ == "__main__":
    main()
