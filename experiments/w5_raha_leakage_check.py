"""
experiments/w5_raha_leakage_check.py
------------------------------------
Settle the caveat that `results/w5_raha_comparison.md` records as the one
thing blocking its own conclusion.

That comparison beats Raha on three of four datasets while using **zero
labels** against Raha's twenty — but our rule detector's valid ranges,
vocabularies and IQR fences come from `DatasetProfile`, which is built from
`clean_df` (CLAUDE.md §3). So "zero labels" is not the same as "no clean
knowledge", and the comparison stated plainly that it was **not settled**
until the leakage ablation ran.

W5.1b (`results/w5_leakage_ablation.md`) settled it for breast_cancer and
found the caveat **material** there: PR-AUC -0.1488 with disjoint CIs. It
did not cover hospital, rayyan or beers, which is three quarters of the
head-to-head.

This script closes that. For each of the four Raha datasets it recomputes
our rule detector's **natural operating point** under both profiles:

    clean profile  — the shipped path (what the comparison reported)
    dirty profile  — profile_dataset(name, paths, dirty, dirty), so every
                     fence, range and vocabulary is dirty-derived

The dirty arm uses **no clean data of any kind**, so its P/R/F1 is a
genuinely zero-knowledge number and is the one that can be set against
Raha's 20-label result without an asterisk.

Cheap by design: the rule detector needs only a profile, not the
conditional model, so this is seconds per dataset rather than the hours
W5.1b needed.

Run:  python experiments/w5_raha_leakage_check.py
Writes results/w5_raha_leakage_check.md. Seed 42.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.console import enable_unicode_stdout  # noqa: E402
from src.data.loader import DataLoader  # noqa: E402
from src.data.profiler import profile_dataset  # noqa: E402
from src.detection.detector import Detector  # noqa: E402
from src.detection_eval.harness import RuleDetectorAdapter, flatten_cells  # noqa: E402
from src.detection_eval.labels import cell_labels  # noqa: E402
from src.pipeline.runner import default_flag_outliers  # noqa: E402

enable_unicode_stdout()

SEED = 42
IQR_K = 3.0
DATASETS = ("breast_cancer", "hospital", "rayyan", "beers")
RAHA_JSON = ROOT / "results" / "w5_raha_raw.json"
OUT_MD = ROOT / "results" / "w5_raha_leakage_check.md"


def set_metrics(y: np.ndarray, flagged: np.ndarray) -> Dict[str, float]:
    """
    Precision / recall / F1 of a flag SET, re-derived here from raw counts.

    Deliberately not routed through a metrics helper: this is the number the
    external comparison turns on, so it is computed from tp/fp/fn directly
    where it can be checked by eye.
    """
    tp = int(np.sum(flagged & y))
    fp = int(np.sum(flagged & ~y))
    fn = int(np.sum(~flagged & y))
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "flagged": int(flagged.sum())}


def measure(dataset: str, arm: str) -> Optional[Dict[str, float]]:
    loader = DataLoader("datasets")
    clean, dirty = loader.load(dataset)
    paths = loader.discover(dataset)

    if arm == "clean":
        profile = profile_dataset(dataset, paths, clean, dirty)
    else:
        profile = profile_dataset(dataset, paths, dirty, dirty)
        # Proof the wrapper took effect: these two fields are computed by
        # comparing the two frames, so passing the same frame twice must
        # zero them. A silently failed patch would show a zero gap.
        if profile.n_corrupted_cells != 0:
            raise RuntimeError(
                f"{dataset}: dirty profiling did not take effect "
                f"(n_corrupted_cells={profile.n_corrupted_cells})"
            )

    detector = Detector(
        profile, iqr_k=IQR_K,
        flag_statistical_outliers=default_flag_outliers(dataset),
    )
    adapter = RuleDetectorAdapter(detector=detector, name=f"rule_{arm}")
    y = flatten_cells(cell_labels(clean, dirty)).astype(bool)
    scores = flatten_cells(adapter.score_cells(dirty, profile))
    out = set_metrics(y, scores > 0.0)
    out["n_corrupted"] = int(y.sum())
    out["n_cells"] = int(y.size)
    return out


def main() -> int:
    raha: Dict[str, Dict] = {}
    if RAHA_JSON.exists():
        raha = json.loads(RAHA_JSON.read_text(encoding="utf-8")).get(
            "results", {})

    rows: Dict[str, Dict[str, Dict[str, float]]] = {}
    print(f"{'dataset':<15}{'arm':<7}{'P':>8}{'R':>8}{'F1':>8}"
          f"{'flagged':>9}{'TP':>8}")
    print("-" * 63)
    for ds in DATASETS:
        rows[ds] = {}
        for arm in ("clean", "dirty"):
            t0 = time.time()
            try:
                m = measure(ds, arm)
            except Exception as exc:
                print(f"{ds:<15}{arm:<7}  FAILED  {type(exc).__name__}: {exc}")
                continue
            rows[ds][arm] = m
            print(f"{ds:<15}{arm:<7}{m['precision']:>8.4f}{m['recall']:>8.4f}"
                  f"{m['f1']:>8.4f}{m['flagged']:>9,}{m['tp']:>8,}"
                  f"   [{time.time() - t0:.0f}s]")

    L: List[str] = []
    a = L.append
    a("# W5.2b — does the Raha comparison survive dirty profiling?")
    a("")
    a("`results/w5_raha_comparison.md` beats Raha on three of four datasets")
    a("using **zero labels** against Raha's twenty, but recorded that the")
    a("comparison was **not settled**: our rule detector's fences, ranges and")
    a("vocabularies come from `DatasetProfile`, built from `clean_df`, so")
    a("\"zero labels\" is not \"no clean knowledge\".")
    a("")
    a("W5.1b settled that for breast_cancer and found the caveat **material**")
    a("(PR-AUC -0.1488, disjoint CIs). It did not cover hospital, rayyan or")
    a("beers — three quarters of the head-to-head. This closes that gap by")
    a("recomputing our natural operating point with **every profile statistic")
    a("derived from dirty data**. The dirty arm uses no clean data of any")
    a("kind, so it is the number that can face Raha's without an asterisk.")
    a("")
    a("Generated by `experiments/w5_raha_leakage_check.py`. Seed 42, full")
    a("datasets, `score > 0` operating point (a rule detector is a *set*, so")
    a("no threshold is tuned and no oracle is used).")
    a("")
    a("## Head-to-head, now including the zero-knowledge arm")
    a("")
    a("| dataset | ours F1 (clean profile) | **ours F1 (dirty profile)** | Raha F1 (20 labels) | winner on the dirty arm |")
    a("|---|---:|---:|---:|---|")
    for ds in DATASETS:
        c = rows.get(ds, {}).get("clean")
        d = rows.get(ds, {}).get("dirty")
        rf = raha.get(ds, {}).get("f1")
        if c is None or d is None:
            a(f"| {ds} | — | — | "
              f"{'—' if rf is None else f'{rf:.4f}'} | not measured |")
            continue
        if rf is None:
            win = "no Raha result"
        elif d["f1"] > rf:
            win = "**ours**"
        elif d["f1"] < rf:
            win = "**Raha**"
        else:
            win = "tie"
        a(f"| {ds} | {c['f1']:.4f} | **{d['f1']:.4f}** | "
          f"{'—' if rf is None else f'{rf:.4f}'} | {win} |")
    a("")
    a("## Full detail")
    a("")
    a("| dataset | arm | precision | recall | F1 | flagged | TP | corrupted cells |")
    a("|---|---|---:|---:|---:|---:|---:|---:|")
    for ds in DATASETS:
        for arm in ("clean", "dirty"):
            m = rows.get(ds, {}).get(arm)
            if m is None:
                continue
            a(f"| {ds} | {arm} | {m['precision']:.4f} | {m['recall']:.4f} | "
              f"{m['f1']:.4f} | {m['flagged']:,} | {m['tp']:,} | "
              f"{m['n_corrupted']:,} |")
    a("")
    a("## Cost of the clean profile, per dataset")
    a("")
    a("| dataset | F1 delta (dirty - clean) | recall delta | precision delta |")
    a("|---|---:|---:|---:|")
    for ds in DATASETS:
        c = rows.get(ds, {}).get("clean")
        d = rows.get(ds, {}).get("dirty")
        if c is None or d is None:
            continue
        a(f"| {ds} | {d['f1'] - c['f1']:+.4f} | "
          f"{d['recall'] - c['recall']:+.4f} | "
          f"{d['precision'] - c['precision']:+.4f} |")
    a("")
    a("## What this settles, and what it does not")
    a("")
    a("Settled: whether our wins over Raha depend on clean-derived profile")
    a("statistics, on all four datasets rather than one.")
    a("")
    a("Not settled: Raha still ran at a single 20-label budget and one seed")
    a("per dataset, and the four larger datasets exhausted memory with")
    a("OpenBLAS allocation failures. Raha's breast_cancer result remains")
    a("anomalous (13 flags out of 8,388, recall 0.021) and is still not")
    a("quoted as evidence of anything. And this compares **detection only** —")
    a("Raha makes no repair/review/abstain decision at all, which is where")
    a("this project's actual contribution lies.")
    a("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
