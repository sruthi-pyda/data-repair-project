"""
experiments/w9c_raha_rematch.py
-------------------------------
W9.3 — **are we still worse than Raha?**

`results/w5_raha_comparison.md` recorded the honest answer at the time: once
the clean-derived `DatasetProfile` was removed, our rule detector collapsed
and Raha won 3 of 4 datasets.

    dataset        ours (clean prof.)  ours (ZERO knowledge)  Raha (20 labels)
    breast_cancer        0.7664              0.6327                0.0404
    hospital             0.9271              0.0000                0.4000
    rayyan               0.3702              0.0558                0.6457
    beers                0.8622              0.0558                0.7988

Since then the text fallback detector (W9.2) covers columns nothing else
could — `hospital` went from 2 of 20 columns scored to 11, `flights` from 0
of 7 to 6 — so the question is open again and must be re-measured rather
than argued.

COMPARING FAIRLY WITH A FLAG SET
================================
Raha emits a flag **set** at a 20-label budget, so its F1 is a single
operating point. Our combined detector emits a continuous score, and the
choice of threshold decides the F1 entirely. Three points are reported, and
only the second is a like-for-like comparison:

- **zero-label** — the threshold implied by flagging every cell any detector
  scores above zero. Uses no labels at all, so it is *harder* than Raha's
  setting, and is reported for that reason.
- **20-label (the fair comparison)** — spend exactly Raha's budget: sample 20
  cells, reveal their labels, and pick the threshold maximising F1 on those
  20. Same information, same output type.
- **oracle best-F1** — the threshold chosen with full knowledge of the
  labels. An **upper bound**, never an operating point, and labelled as such
  wherever it appears.

Every arm is **zero clean-knowledge**: the conditional detector is fitted
with `profile=None`, and the rule detector's profile is built from the dirty
frame in both positions, because building it from clean data is exactly the
leak `results/w5_raha_leakage_check.md` measured.

The 20-label threshold is repeated over `--splits` label samples and reported
as mean [min, max], since a single draw of 20 cells at these prevalences is
nearly meaningless on its own.

Run:  python experiments/w9c_raha_rematch.py
Writes results/w9c_raha_rematch.md. Seeds 42..42+splits-1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.console import enable_unicode_stdout  # noqa: E402
from src.data.loader import DataLoader  # noqa: E402
from src.data.profiler import profile_dataset  # noqa: E402
from src.detection.detector import Detector  # noqa: E402
from src.detection_eval.harness import RuleDetectorAdapter  # noqa: E402
from src.detection_eval.labels import cell_labels  # noqa: E402
from src.detection_eval.metrics import best_f1, pr_auc  # noqa: E402
from src.detectors.conditional import ConditionalDetector  # noqa: E402
from src.detectors.text_fallback import TextFallbackDetector  # noqa: E402
from src.pipeline.runner import default_flag_outliers  # noqa: E402

enable_unicode_stdout()

BASE_SEED = 42
IQR_K = 3.0
BUDGET = 20
DATASETS = ("breast_cancer", "hospital", "rayyan", "beers")
RAHA_JSON = ROOT / "results" / "w5_raha_raw.json"
OUT_MD = ROOT / "results" / "w9c_raha_rematch.md"


def _f(v: Optional[float], nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}"


def _mr(vals: Sequence[Optional[float]], nd: int = 4) -> str:
    xs = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    if not xs:
        return "—"
    if len(xs) == 1:
        return f"{xs[0]:.{nd}f}"
    return f"{np.mean(xs):.{nd}f} [{min(xs):.{nd}f}, {max(xs):.{nd}f}]"


def prf(y: np.ndarray, flagged: np.ndarray) -> Tuple[float, float, float]:
    """Precision / recall / F1 from raw counts, checkable by eye."""
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    fn = int(np.sum(~flagged & (y == 1)))
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) > 0 else 0.0)


def threshold_from_budget(
    y: np.ndarray, scores: np.ndarray, budget: int, seed: int
) -> Optional[float]:
    """
    Pick a threshold using exactly `budget` revealed labels.

    This is the only place labels enter a deployable arm, and it is Raha's
    own protocol: a fixed label budget simulated from ground truth, standard
    Raha / ED2 / SAGED practice.
    """
    rng = np.random.default_rng(seed)
    if y.size <= budget:
        return None
    idx = rng.choice(y.size, size=budget, replace=False)
    ys, ss = y[idx], scores[idx]
    if ys.sum() == 0 or ys.sum() == ys.size:
        # No positive (or no negative) in the sample: nothing to fit a
        # threshold on. At these prevalences this is the common case and is
        # itself the finding -- reported, not silently patched.
        return None
    best_t, best_f = None, -1.0
    for t in np.unique(ss):
        f = prf(ys, ss >= t)[2]
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t


def measure(dataset: str, splits: int) -> Dict[str, object]:
    loader = DataLoader("datasets")
    clean, dirty = loader.load(dataset)
    clean = clean.reset_index(drop=True)
    dirty = dirty.reset_index(drop=True)
    cols = list(dirty.columns)
    labels = cell_labels(clean, dirty)
    y = labels[cols].to_numpy().astype(int).ravel(order="F")

    # ── conditional, zero knowledge ────────────────────────────────────
    cond = ConditionalDetector(mode="full", seed=BASE_SEED).fit(dirty)
    cond_s = cond.score_cells(dirty)

    # ── rule detector, DIRTY-profiled so it carries no clean knowledge ─
    paths = loader.discover(dataset)
    profile = profile_dataset(dataset, paths, dirty, dirty)
    rule = Detector(profile, iqr_k=IQR_K,
                    flag_statistical_outliers=default_flag_outliers(dataset))
    rule_s = RuleDetectorAdapter(detector=rule, name="rule").score_cells(
        dirty, profile)

    # ── text fallback, on columns the conditional detector misses ──────
    fb = TextFallbackDetector()
    fb_s = fb.score_cells(dirty)
    cond_cols = {c for c in cols
                 if np.unique(cond_s[c].to_numpy(dtype=float)).size > 1}
    new_cols = [c for c in cols
                if c not in cond_cols
                and (fb.reports_.get(c) is not None
                     and fb.reports_[c].scored)]

    # Combined: conditional where it covers, fallback where it does not,
    # and the rule detector's flags unioned in as a floor. Each component is
    # zero-knowledge.
    combined = cond_s.copy()
    for c in new_cols:
        combined[c] = fb_s[c]
    comb = combined[cols].to_numpy(dtype=float).ravel(order="F")
    rule_flat = rule_s[cols].to_numpy(dtype=float).ravel(order="F")
    union = np.maximum(comb, rule_flat)

    out: Dict[str, object] = {
        "dataset": dataset,
        "n_cells": int(y.size),
        "n_corrupt": int(y.sum()),
        "prevalence": float(y.mean()),
        "cond_cols": len(cond_cols),
        "new_cols": len(new_cols),
        "n_cols": len(cols),
        "pr_auc": pr_auc(y, union),
        "oracle_f1": best_f1(y, union)[0],
    }

    # zero-label operating point: flag anything scored above zero
    out["zero_label"] = prf(y, union > 0.0)

    # 20-label operating point, repeated over label draws
    f1s, ps, rs, n_fitted = [], [], [], 0
    for k in range(splits):
        t = threshold_from_budget(y, union, BUDGET, BASE_SEED + k)
        if t is None:
            continue
        n_fitted += 1
        p, r, f = prf(y, union >= t)
        ps.append(p)
        rs.append(r)
        f1s.append(f)
    out["budget_p"], out["budget_r"], out["budget_f1"] = ps, rs, f1s
    out["n_fitted"] = n_fitted
    out["n_budget_draws"] = splits
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="W9.3 Raha rematch")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args(argv)

    raha: Dict[str, Dict] = {}
    if RAHA_JSON.exists():
        raha = json.loads(RAHA_JSON.read_text(encoding="utf-8")).get(
            "results", {})

    rows: List[Dict[str, object]] = []
    print("W9.3 — Raha rematch, zero clean knowledge")
    print(f"budget {BUDGET}  label draws {args.splits}\n")
    for ds in args.datasets:
        t0 = time.time()
        try:
            r = measure(ds, args.splits)
        except Exception as exc:
            print(f"  {ds:<14} FAILED {type(exc).__name__}: {exc}")
            continue
        rows.append(r)
        rf = raha.get(ds, {}).get("f1")
        print(f"  {ds:<14} cols {r['cond_cols']}+{r['new_cols']}/{r['n_cols']}"
              f"  PR-AUC {_f(r['pr_auc'])}"
              f"  zero-label F1 {_f(r['zero_label'][2])}"
              f"  {BUDGET}-label F1 {_mr(r['budget_f1'])}"
              f"  oracle {_f(r['oracle_f1'])}"
              f"  Raha {_f(rf)}   [{time.time()-t0:.0f}s]")

    L: List[str] = []
    a = L.append
    a("# W9.3 — are we still worse than Raha?")
    a("")
    a("Generated by `experiments/w9c_raha_rematch.py`. Full datasets, no row")
    a(f"cap. Label budget {BUDGET}, repeated over {args.splits} draws.")
    a("")
    a("`results/w5_raha_comparison.md` recorded that once the clean-derived")
    a("`DatasetProfile` was removed, our rule detector collapsed and **Raha won")
    a("3 of 4**. The text fallback (W9.2) now covers columns nothing else")
    a("could, so this re-measures rather than argues.")
    a("")
    a("## How the comparison is made fair")
    a("")
    a("Raha emits a flag **set** at a 20-label budget, so its F1 is one")
    a("operating point. Ours emits a continuous score, where the threshold")
    a("decides the F1 entirely. Three points, and **only the middle one is")
    a("like-for-like**:")
    a("")
    a("- **zero-label** — flag every cell any detector scores above zero. No")
    a("  labels at all, so a *harder* setting than Raha's.")
    a(f"- **{BUDGET}-label** — Raha's own budget and protocol: sample "
      f"{BUDGET} cells, reveal them, pick the best threshold. **The fair")
    a("  comparison.**")
    a("- **oracle best-F1** — threshold chosen knowing every label. An")
    a("  **upper bound, never an operating point.**")
    a("")
    a("Every arm is zero clean-knowledge: the conditional detector is fitted")
    a("with `profile=None` and the rule detector's profile is built from the")
    a("dirty frame in both positions.")
    a("")
    a("## Results")
    a("")
    a(f"| dataset | prevalence | cols covered | PR-AUC | zero-label F1 | **{BUDGET}-label F1** | **Raha F1** | oracle F1 (bound) |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        rf = raha.get(str(r["dataset"]), {}).get("f1")
        a(f"| {r['dataset']} | {_f(r['prevalence'], 5)} | "
          f"{r['cond_cols']}+{r['new_cols']}/{r['n_cols']} | "
          f"{_f(r['pr_auc'])} | {_f(r['zero_label'][2])} | "
          f"**{_mr(r['budget_f1'])}** | **{_f(rf)}** | "
          f"{_f(r['oracle_f1'])} |")
    a("")
    a("`cols covered` is conditional + fallback, out of total.")
    a("")
    a("### Head to head on the fair comparison")
    a("")
    a(f"| dataset | ours ({BUDGET} labels) | Raha ({BUDGET} labels) | winner |")
    a("|---|---:|---:|---|")
    wins = losses = ties = 0
    for r in rows:
        rf = raha.get(str(r["dataset"]), {}).get("f1")
        ours = (float(np.mean(r["budget_f1"]))  # type: ignore[arg-type]
                if r["budget_f1"] else None)
        if ours is None or rf is None:
            w = "—"
        elif ours > rf:
            w, _ = "**ours**", (wins := wins + 1)
        elif ours < rf:
            w, _ = "**Raha**", (losses := losses + 1)
        else:
            w, _ = "tie", (ties := ties + 1)
        a(f"| {r['dataset']} | {_f(ours)} | {_f(rf)} | {w} |")
    a("")
    a(f"**{wins} win(s), {losses} loss(es), {ties} tie(s)** on the fair")
    a("comparison.")
    a("")
    a("### How often 20 labels could even fit a threshold")
    a("")
    a("| dataset | draws that contained a positive |")
    a("|---|---|")
    for r in rows:
        a(f"| {r['dataset']} | {r['n_fitted']}/{r['n_budget_draws']} |")
    a("")
    a("A draw with no positive cell cannot fit a threshold at all. At these")
    a("prevalences that is a frequent outcome and is itself a finding — the")
    a("same one Wave 2 recorded, where 20 random labels found zero errors on")
    a("adult in 5 of 5 splits.")
    a("")
    a("## Caveats that still stand")
    a("")
    a("- Raha ran at **one** budget and **one** seed per dataset, and the four")
    a("  larger datasets exhausted memory with OpenBLAS allocation failures.")
    a("- Raha's `breast_cancer` result remains anomalous — 13 flags out of")
    a("  8,388, recall 0.021 — far below its published behaviour, and it is")
    a("  still not quoted as evidence of anything.")
    a("- This compares **detection only**. Raha makes no repair, review or")
    a("  abstain decision, which is where this project's contribution lies.")
    a("")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
