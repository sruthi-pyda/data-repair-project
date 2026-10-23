"""
experiments/w10_threshold.py
----------------------------
W10 — does stratified threshold selection convert the hospital loss?

Planned in `docs/w10_plan.md`, committed before this file existed.

`results/w9c_raha_rematch.md` split our three losses to Raha in two:

    dataset    ours @20 labels    our ORACLE bound    Raha @20 labels
    hospital        0.3849             0.5054              0.4000
    beers           0.5430             0.6491              0.7988
    rayyan          0.1950             0.3035              0.6457

**hospital's oracle bound exceeds Raha's F1**, so the detector is already
good enough and the threshold is what loses. beers and rayyan lose even at
their oracles, so they are genuine detection deficits and nothing here can
fix them — they are included only to check this change does not make them
worse.

The detector is identical to W9.3's: conditional detector (`profile=None`),
plus the gated text fallback on columns it cannot model, unioned with the
rule detector built from the **dirty** frame in both positions. Zero clean
knowledge in every arm. Only the operating point changes.

Each selector is run over `--draws` label draws and the **realised** F1 is
measured against the full labels afterwards. Everything is reported as
mean [min, max]; no single draw is quoted.

Run:  python experiments/w10_threshold.py
Writes results/w10_threshold.md. Seeds 42..42+draws-1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.console import enable_unicode_stdout  # noqa: E402
from src.data.loader import DataLoader  # noqa: E402
from src.data.profiler import profile_dataset  # noqa: E402
from src.decision.threshold import (  # noqa: E402
    apply_threshold, select_threshold,
)
from src.detection.detector import Detector  # noqa: E402
from src.detection_eval.harness import RuleDetectorAdapter  # noqa: E402
from src.detection_eval.labels import cell_labels  # noqa: E402
from src.detection_eval.metrics import best_f1  # noqa: E402
from src.detectors.conditional import ConditionalDetector  # noqa: E402
from src.detectors.text_fallback import TextFallbackDetector  # noqa: E402
from src.pipeline.runner import default_flag_outliers  # noqa: E402

enable_unicode_stdout()

BASE_SEED = 42
IQR_K = 3.0
BUDGET = 20
DATASETS = ("breast_cancer", "hospital", "rayyan", "beers")
METHODS = ("uniform_f1", "stratified_f1", "stratified_cost")
RAHA_JSON = ROOT / "results" / "w5_raha_raw.json"
OUT_MD = ROOT / "results" / "w10_threshold.md"


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


def prf(y: np.ndarray, flagged: np.ndarray):
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    fn = int(np.sum(~flagged & (y == 1)))
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) > 0 else 0.0), tp, fp, fn


def build_scores(dataset: str):
    """The W9.3 detector, unchanged. Zero clean knowledge."""
    loader = DataLoader("datasets")
    clean, dirty = loader.load(dataset)
    clean = clean.reset_index(drop=True)
    dirty = dirty.reset_index(drop=True)
    cols = list(dirty.columns)

    cond = ConditionalDetector(mode="full", seed=BASE_SEED).fit(dirty)
    cond_s = cond.score_cells(dirty)

    paths = loader.discover(dataset)
    profile = profile_dataset(dataset, paths, dirty, dirty)
    rule = Detector(profile, iqr_k=IQR_K,
                    flag_statistical_outliers=default_flag_outliers(dataset))
    rule_s = RuleDetectorAdapter(detector=rule, name="rule").score_cells(
        dirty, profile)

    fb = TextFallbackDetector()
    fb_s = fb.score_cells(dirty)
    cond_cols = {c for c in cols
                 if np.unique(cond_s[c].to_numpy(dtype=float)).size > 1}
    combined = cond_s.copy()
    for c in cols:
        rep = fb.reports_.get(c)
        if c not in cond_cols and rep is not None and rep.scored:
            combined[c] = fb_s[c]

    scores = np.maximum(
        combined[cols].to_numpy(dtype=float).ravel(order="F"),
        rule_s[cols].to_numpy(dtype=float).ravel(order="F"))
    y = cell_labels(clean, dirty)[cols].to_numpy().astype(int).ravel(order="F")
    return scores, y


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="W10 threshold selection")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--budget", type=int, default=BUDGET)
    args = ap.parse_args(argv)

    raha = {}
    if RAHA_JSON.exists():
        raha = json.loads(RAHA_JSON.read_text(encoding="utf-8")).get(
            "results", {})

    rows: List[Dict[str, object]] = []
    print("W10 — threshold selection from a tiny label budget")
    print(f"budget {args.budget}  draws {args.draws}\n")

    for ds in args.datasets:
        t0 = time.time()
        try:
            scores, y = build_scores(ds)
        except Exception as exc:
            print(f"  {ds:<14} FAILED {type(exc).__name__}: {exc}")
            continue

        def reveal(idx: np.ndarray) -> np.ndarray:
            return y[np.asarray(idx, dtype=int)]

        rec: Dict[str, object] = {
            "dataset": ds, "n_cells": int(y.size), "n_pos": int(y.sum()),
            "prevalence": float(y.mean()),
            "oracle_f1": best_f1(y, scores)[0],
            "raha_f1": raha.get(ds, {}).get("f1"),
            "methods": {},
        }
        for m in METHODS:
            f1s, ps, rs, fitted, spent = [], [], [], 0, []
            for k in range(args.draws):
                ch = select_threshold(scores, reveal, method=m,
                                      budget=args.budget, seed=BASE_SEED + k)
                spent.append(ch.n_revealed)
                if ch.threshold is None:
                    f1s.append(0.0)
                    ps.append(0.0)
                    rs.append(0.0)
                    continue
                fitted += 1
                p, r, f, *_ = prf(y, apply_threshold(scores, ch))
                ps.append(p)
                rs.append(r)
                f1s.append(f)
            rec["methods"][m] = {  # type: ignore[index]
                "f1": f1s, "p": ps, "r": rs, "fitted": fitted,
                "max_spent": max(spent) if spent else 0,
            }
        rows.append(rec)
        best = {m: np.mean(rec["methods"][m]["f1"]) for m in METHODS}  # type: ignore[index]
        print(f"  {ds:<14} prev {rec['prevalence']:.4f}  "
              + "  ".join(f"{m.split('_')[0][:5]}={best[m]:.4f}"
                          for m in METHODS)
              + f"  oracle={_f(rec['oracle_f1'])}"
              f"  Raha={_f(rec['raha_f1'])}   [{time.time()-t0:.0f}s]")

    # ── report ─────────────────────────────────────────────────────────
    L: List[str] = []
    a = L.append
    a("# W10 — does stratified threshold selection convert the hospital loss?")
    a("")
    a("Generated by `experiments/w10_threshold.py`. Planned in")
    a("`docs/w10_plan.md`, committed before the code. Full datasets, budget")
    a(f"{args.budget}, {args.draws} label draws, seeds "
      f"{BASE_SEED}..{BASE_SEED + args.draws - 1}. Every figure is")
    a("**mean [min, max]** over draws.")
    a("")
    a("## 0. What is and is not being changed")
    a("")
    a("The detector is **identical** to W9.3's and zero clean knowledge:")
    a("conditional detector with `profile=None`, plus the gated text fallback")
    a("on columns it cannot model, unioned with the rule detector built from")
    a("the dirty frame in both positions. **Only the operating point moves.**")
    a("No detection number changes.")
    a("")
    a("`results/w9c_raha_rematch.md` split our three losses in two:")
    a("**hospital's oracle bound exceeds Raha's F1**, so its detector is")
    a("already good enough and the threshold is what loses. beers and rayyan")
    a("lose even at their oracles, so they are genuine detection deficits —")
    a("included here only to check this change does not make them worse.")
    a("")
    a("## 1. Realised F1 by selection rule")
    a("")
    a("| dataset | prevalence | uniform (current) | **stratified F1** | stratified cost | oracle (bound) | **Raha** |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        m = r["methods"]  # type: ignore[index]
        a(f"| {r['dataset']} | {_f(r['prevalence'], 5)} | "
          f"{_mr(m['uniform_f1']['f1'])} | "
          f"**{_mr(m['stratified_f1']['f1'])}** | "
          f"{_mr(m['stratified_cost']['f1'])} | "
          f"{_f(r['oracle_f1'])} | **{_f(r['raha_f1'])}** |")
    a("")
    a("## 2. How often each rule could fit a threshold at all")
    a("")
    a("A draw containing no positive cannot fit a threshold; it is scored 0,")
    a("not dropped, because a selector that fails to choose is a selector that")
    a("fails.")
    a("")
    a("| dataset | uniform | stratified F1 | stratified cost |")
    a("|---|---|---|---|")
    for r in rows:
        m = r["methods"]  # type: ignore[index]
        a(f"| {r['dataset']} | {m['uniform_f1']['fitted']}/{args.draws} | "
          f"{m['stratified_f1']['fitted']}/{args.draws} | "
          f"{m['stratified_cost']['fitted']}/{args.draws} |")
    a("")
    a("## 3. Budget honesty")
    a("")
    a("| dataset | max labels revealed, any method | budget |")
    a("|---|---:|---:|")
    for r in rows:
        m = r["methods"]  # type: ignore[index]
        a(f"| {r['dataset']} | "
          f"{max(m[k]['max_spent'] for k in METHODS)} | {args.budget} |")
    a("")
    a("## 4. Verdict against the pre-registered conditions")
    a("")
    a("| condition | result |")
    a("|---|---|")
    hosp = next((r for r in rows if r["dataset"] == "hospital"), None)
    if hosp:
        m = hosp["methods"]  # type: ignore[index]
        u = float(np.mean(m["uniform_f1"]["f1"]))
        s = float(np.mean(m["stratified_f1"]["f1"]))
        c = float(np.mean(m["stratified_cost"]["f1"]))
        rf = hosp["raha_f1"]
        a(f"| 1 beats uniform on hospital | "
          f"{'**PASS**' if s > u else '**FAIL**'} — "
          f"{s:.4f} vs {u:.4f} |")
        best_s = max(s, c)
        a(f"| 2 **converts the loss** (hospital > Raha {_f(rf)}) | "
          f"{'**PASS**' if (rf is not None and best_s > rf) else '**FAIL**'}"
          f" — best rule {best_s:.4f} |")
    others = [r for r in rows if r["dataset"] != "hospital"]
    worse = []
    for r in others:
        m = r["methods"]  # type: ignore[index]
        u = float(np.mean(m["uniform_f1"]["f1"]))
        s = max(float(np.mean(m["stratified_f1"]["f1"])),
                float(np.mean(m["stratified_cost"]["f1"])))
        if s < u - 1e-9:
            worse.append((r["dataset"], s - u))
    a(f"| 3 does not break the others | "
      f"{'**PASS**' if not worse else '**FAIL**'} — "
      + ("none degraded" if not worse
         else ", ".join(f"`{d}` {v:+.4f}" for d, v in worse)) + " |")
    over = []
    for r in rows:
        ob = r["oracle_f1"]
        if ob is None:
            continue
        m = r["methods"]  # type: ignore[index]
        for k in METHODS:
            if float(np.mean(m[k]["f1"])) > float(ob) + 1e-9:
                over.append((r["dataset"], k))
    a(f"| 4 nothing beats its own oracle | "
      f"{'**PASS**' if not over else '**FAIL**'} — "
      + ("no violation" if not over
         else ", ".join(f"`{d}`/{k}" for d, k in over)) + " |")
    ok_budget = all(
        max(r["methods"][k]["max_spent"] for k in METHODS) <= args.budget  # type: ignore[index]
        for r in rows)
    a(f"| 5 budget honesty | "
      f"{'**PASS**' if ok_budget else '**FAIL**'} — no method exceeded "
      f"{args.budget} labels |")
    a(f"| 6 reproducibility | **PASS** — mean [min, max] over "
      f"{args.draws} draws |")
    a("")
    a("## 5. Caveats")
    a("")
    a("- This is **threshold selection, not detection**. beers and rayyan lose")
    a("  to Raha at their oracle bounds, so no operating point can convert")
    a("  them; that needs a better detector.")
    a("- Raha ran at one budget and one seed per dataset, and its")
    a("  `breast_cancer` result remains anomalous (13 flags of 8,388) and is")
    a("  still not quoted as evidence of anything.")
    a("- Detection only. Raha makes no repair, review or abstain decision.")
    a("")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
