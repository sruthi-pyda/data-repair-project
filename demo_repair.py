"""
demo_repair.py
---------------
Phase 4 CLI demo: Mock LLM repair candidates for each dataset.

Run:
    python3 demo_repair.py

Shows top-3 repair candidates for a sample of detected errors per dataset,
then reports top-1 and top-k accuracy from the repair evaluator.
"""

from __future__ import annotations

import sys

from src.data.loader import DataLoader
from src.data.profiler import profile_dataset
from src.detection.detector import Detector
from src.detection.evaluation import DetectorEvaluator
from src.repair.evaluation import RepairEvaluator
from src.repair.mock_repair import MockRepairer, build_repair_context

DATASETS = ["breast_cancer", "smartfactory", "adult"]
SAMPLE_DISPLAY = 6   # cells to display per dataset
SEED = 42


def run_dataset(name: str, loader: DataLoader) -> None:
    w = 72
    print(f"\n{'═' * w}")
    print(f"  DATASET: {name.upper()}")
    print(f"{'═' * w}")

    clean_df, dirty_df = loader.load(name)
    profile = profile_dataset(
        name, loader.discover(name), clean_df, dirty_df
    )

    # Detection
    flag_outliers = name != "smartfactory"
    detector = Detector(profile, flag_statistical_outliers=flag_outliers)
    records = detector.detect(dirty_df)

    det_evaluator = DetectorEvaluator(clean_df, dirty_df, name)
    det_results = det_evaluator.evaluate(records)
    print(
        f"  Detection: {len(records)} flagged | "
        f"P={det_results.precision:.3f} R={det_results.recall:.3f} F1={det_results.f1:.3f}"
    )

    if not records:
        print("  No errors detected — skipping repair.")
        return

    # Repair
    repairer = MockRepairer(seed=SEED, top_k=3)
    results = repairer.repair_batch(records, dirty_df, profile)

    # Evaluation (uses clean_df here — only place it's allowed)
    rep_evaluator = RepairEvaluator(clean_df, dirty_df, name)
    rep_results = rep_evaluator.evaluate(results)
    print(
        f"  Repair  : {rep_results.n_repaired} cells | "
        f"Top-1={rep_results.top1_accuracy:.3f} Top-k={rep_results.topk_accuracy:.3f} "
        f"mean_p(correct)={rep_results.mean_correct_prob:.3f}"
    )

    # Display sample
    print(f"\n  Sample repair candidates (first {SAMPLE_DISPLAY}):")
    print(f"  {'Column':<22} {'Error':<22} {'Dirty':<12} {'Candidates'}")
    print(f"  {'─' * 22} {'─' * 22} {'─' * 12} {'─' * 30}")

    for res in results[:SAMPLE_DISPLAY]:
        dirty_str = str(res.dirty_value)[:10]
        cands = "  |  ".join(
            f"{c.value!r}({c.probability:.2f})"
            for c in res.candidates
        )
        print(f"  {res.column:<22} {res.error_type:<22} {dirty_str:<12} {cands}")


def main() -> None:
    loader = DataLoader("datasets")

    discovered = {name for name, _ in loader.discover.__self__._root.iterdir()
                  if (loader.discover.__self__._root / name).is_dir()} \
        if False else None  # avoid introspection; just try each

    for name in DATASETS:
        try:
            run_dataset(name, loader)
        except Exception as exc:
            print(f"\n[{name}] skipped: {exc}")

    print(f"\n{'═' * 72}")
    print("  Phase 4 demo complete.")
    print(f"{'═' * 72}\n")


if __name__ == "__main__":
    main()
