"""
experiments/w12_policy_ablation.py
----------------------------------
W12 — the two experiments the paper needs to defend its main claim.

Both are cheap because the decision rule is arithmetic over quantities the
pipeline already produced, so no detector is refitted.

1. COST SENSITIVITY (reviewer point A). The zero-regret result was measured
   at ONE cost configuration per dataset. If it only holds there it is an
   artefact of the configuration. This sweeps the two ratios that matter —
   `C_wrong / C_human` and `C_abstain / C_human` — and reports how often the
   proposed policy is still best-or-tied, over TOTAL HARM, which prices
   undetected corruption rather than hiding it.

2. CALIBRATION ABLATION (reviewer point B). The paper argues that calibrated
   probabilities produce better decisions than raw confidence. That has never
   been tested directly. Here the identical cost rule is driven by raw
   confidence instead of the calibrated `p`, everything else held fixed, so
   any difference is attributable to calibration alone.

TOTAL HARM
==========
Every policy is scored on

    harm = (wrong auto-repairs) * C_wrong
         + (cells reviewed)     * C_human
         + (corrupted cells left unrepaired, flagged or not) * C_abstain

The last term includes corruption the detector never flagged, which is what
stops a policy that detects nothing from looking good. `do-nothing` is
included as a reference line for exactly that reason.

Run:  python experiments/w12_policy_ablation.py
Writes results/w12_policy_ablation.md. Seed 42.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.console import enable_unicode_stdout  # noqa: E402
from src.evaluation.cost_metrics import corrupted_cells  # noqa: E402
from src.pipeline.runner import run_pipeline  # noqa: E402

enable_unicode_stdout()

SEED = 42
DATASETS = ("breast_cancer", "smartfactory", "adult")
OUT_MD = ROOT / "results" / "w12_policy_ablation.md"

#: Cost grid, expressed as ratios to C_human (which is fixed at 1). These
#: span "a wrong repair is only twice as bad as a review" to "fifty times",
#: and the same for leaving an error in place.
WRONG_RATIOS = (2.0, 5.0, 10.0, 25.0, 50.0)
ABSTAIN_RATIOS = (0.5, 1.0, 2.0, 5.0, 10.0)
C_HUMAN = 1.0


def decide(p: np.ndarray, q: np.ndarray, cw: float, ch: float, ca: float
           ) -> np.ndarray:
    """0 = repair, 1 = review, 2 = abstain. Ties: repair > review > abstain."""
    costs = np.stack([(1.0 - p) * cw, np.full_like(p, ch), q * ca], axis=1)
    return np.argmin(costs, axis=1)


def harm(action: np.ndarray, correct: np.ndarray, is_corrupt: np.ndarray,
         n_missed: int, cw: float, ch: float, ca: float) -> float:
    """Total harm, including corruption the detector never flagged."""
    wrong_repairs = int(np.sum((action == 0) & (correct == 0)))
    reviewed = int(np.sum(action == 1))
    left_corrupt = int(np.sum((action == 2) & (is_corrupt == 1)))
    return (wrong_repairs * cw + reviewed * ch
            + (left_corrupt + n_missed) * ca)


def policies(p: np.ndarray, raw: np.ndarray, q: np.ndarray,
             cw: float, ch: float, ca: float) -> Dict[str, np.ndarray]:
    n = p.shape[0]
    return {
        "proposed (calibrated)": decide(p, q, cw, ch, ca),
        "uncalibrated (raw conf.)": decide(raw, q, cw, ch, ca),
        "always repair": np.zeros(n, dtype=int),
        "always review": np.ones(n, dtype=int),
        "always abstain": np.full(n, 2, dtype=int),
        "conf >= 0.70": np.where(raw >= 0.70, 0, 1),
        "conf >= 0.90": np.where(raw >= 0.90, 0, 1),
    }


def load(dataset: str) -> Optional[Dict[str, object]]:
    b = run_pipeline(dataset, n_records=0, seed=SEED)
    idx = b.decision_indices()
    if not idx:
        return None
    p = np.array([b.calibrated_probs[i] for i in idx], dtype=float)
    raw = np.array([b.unc_records[i].raw_confidence for i in idx], dtype=float)
    q = (np.array([b.corruption_probs[i] for i in idx], dtype=float)
         if b.corruption_probs else np.ones(len(idx)))
    correct = np.array([b.cal_samples[i].correct for i in idx], dtype=int)
    is_corrupt = np.array([b.detection_labels[i] for i in idx], dtype=int)

    census = corrupted_cells(b.clean_df, b.dirty_df, skip_columns=b.skip_columns)
    flagged = {(int(b.unc_records[i].row_id), b.unc_records[i].column)
               for i in idx}
    n_missed = len(census - flagged)
    return {"p": p, "raw": raw, "q": q, "correct": correct,
            "is_corrupt": is_corrupt, "n_missed": n_missed,
            "n": len(idx), "n_corrupt_total": len(census)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = ap.parse_args(argv)

    data: Dict[str, Dict[str, object]] = {}
    print("W12 — cost sensitivity and the calibration ablation\n")
    for ds in args.datasets:
        t0 = time.time()
        try:
            d = load(ds)
        except Exception as exc:
            print(f"  {ds:<14} FAILED {type(exc).__name__}: {exc}")
            continue
        if d is None:
            print(f"  {ds:<14} no decided cells")
            continue
        data[ds] = d
        print(f"  {ds:<14} {d['n']:>6,} decided, {d['n_missed']:>6,} missed "
              f"of {d['n_corrupt_total']:,} corruptions   [{time.time()-t0:.0f}s]")

    names = list(policies(np.zeros(1), np.zeros(1), np.zeros(1), 1, 1, 1))
    # best-or-tied counts over the grid, per dataset and pooled
    wins: Dict[str, Dict[str, int]] = {n: {ds: 0 for ds in data} for n in names}
    total_cfgs = len(WRONG_RATIOS) * len(ABSTAIN_RATIOS)
    # head-to-head: proposed vs uncalibrated
    h2h = {"proposed": 0, "uncalibrated": 0, "tie": 0}
    deltas: List[float] = []

    for ds, d in data.items():
        for cw_r in WRONG_RATIOS:
            for ca_r in ABSTAIN_RATIOS:
                cw, ch, ca = cw_r * C_HUMAN, C_HUMAN, ca_r * C_HUMAN
                acts = policies(d["p"], d["raw"], d["q"], cw, ch, ca)
                h = {n: harm(a, d["correct"], d["is_corrupt"],
                             int(d["n_missed"]), cw, ch, ca)
                     for n, a in acts.items()}
                h["do nothing"] = (int(d["n_corrupt_total"])) * ca
                best = min(h.values())
                for n in names:
                    if h[n] <= best + 1e-9:
                        wins[n][ds] += 1
                pr, un = h["proposed (calibrated)"], h["uncalibrated (raw conf.)"]
                deltas.append(un - pr)
                if pr < un - 1e-9:
                    h2h["proposed"] += 1
                elif un < pr - 1e-9:
                    h2h["uncalibrated"] += 1
                else:
                    h2h["tie"] += 1

    L: List[str] = []
    a = L.append
    a("# W12 — cost sensitivity and the calibration ablation")
    a("")
    a("Generated by `experiments/w12_policy_ablation.py`. Seed 42, full")
    a("datasets. Both experiments are arithmetic over quantities the pipeline")
    a("already produced, so no detector is refitted and nothing here depends")
    a("on a new model.")
    a("")
    a("All policies are scored on **total harm**, which charges wrong")
    a("auto-repairs, human reviews, and every corrupted cell left unrepaired")
    a("**including corruption the detector never flagged**. That last term is")
    a("what stops a policy that detects nothing from looking good, and")
    a("`do nothing` is carried as a reference line for the same reason.")
    a("")
    a("## 1. Setup")
    a("")
    a("| dataset | decided cells | corruptions missed entirely | total corruptions |")
    a("|---|---:|---:|---:|")
    for ds, d in data.items():
        a(f"| {ds} | {d['n']:,} | {d['n_missed']:,} | {d['n_corrupt_total']:,} |")
    a("")
    a(f"The cost grid fixes `C_human = 1` and sweeps "
      f"`C_wrong` over {list(WRONG_RATIOS)} and `C_abstain` over "
      f"{list(ABSTAIN_RATIOS)}, giving **{total_cfgs} configurations per "
      f"dataset** against the single configuration used previously.")
    a("")
    a("## 2. Cost sensitivity — is zero regret an artefact of one configuration?")
    a("")
    a("How often each policy is best or tied-best on total harm, out of")
    a(f"{total_cfgs} cost configurations per dataset.")
    a("")
    hdr = "| policy | " + " | ".join(data) + " | all |"
    a(hdr)
    a("|---" * (len(data) + 2) + "|")
    for n in names:
        tot = sum(wins[n].values())
        cells = " | ".join(f"{wins[n][ds]}/{total_cfgs}" for ds in data)
        bold = "**" if n.startswith("proposed") else ""
        a(f"| {bold}{n}{bold} | {cells} | {bold}{tot}/"
          f"{total_cfgs*len(data)}{bold} |")
    a("")
    a("A policy counted here is best or tied on that configuration; several")
    a("can tie at once, so columns need not sum to the number of")
    a("configurations.")
    a("")
    a("## 3. Calibration ablation — does calibrating `p` change decisions?")
    a("")
    a("The identical cost rule, driven by **raw confidence** instead of the")
    a("calibrated `p`. Everything else is held fixed, so any difference is")
    a("attributable to calibration alone.")
    a("")
    n_cfg = sum(h2h.values())
    a("| outcome | configurations |")
    a("|---|---:|")
    a(f"| calibrated strictly better | {h2h['proposed']}/{n_cfg} |")
    a(f"| uncalibrated strictly better | {h2h['uncalibrated']}/{n_cfg} |")
    a(f"| identical | {h2h['tie']}/{n_cfg} |")
    a("")
    if deltas:
        arr = np.asarray(deltas, dtype=float)
        a(f"Mean harm saved by calibrating: **{arr.mean():,.1f}** units "
          f"(median {np.median(arr):,.1f}, range {arr.min():,.1f} to "
          f"{arr.max():,.1f}).")
        a("")
        if h2h["proposed"] == 0 and h2h["uncalibrated"] == 0:
            a("**Calibration changes no decision anywhere on this grid.** That")
            a("is a negative result for the calibration argument and is")
            a("reported as one: on these datasets the cost comparison is")
            a("decided by quantities that the calibration map preserves.")
    a("")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_MD.relative_to(ROOT)}")
    for n in names:
        print(f"  {n:<26} best-or-tied {sum(wins[n].values())}/"
              f"{total_cfgs*len(data)}")
    print(f"  calibration h2h: {h2h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
