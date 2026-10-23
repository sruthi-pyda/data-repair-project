"""
experiments/w9b_text_fallback.py
--------------------------------
W9.2b — does the text fallback detector earn its place?

Specified in `docs/w9_plan.md` §W9.2b, committed before this file existed,
with six requirements fixed in advance.

THE DESIGN DECISION THAT MATTERS MOST
=====================================
A fallback that covers new columns changes **which cells are evaluated**, and
that alone can manufacture an apparent improvement. So three populations are
kept strictly apart and never mixed:

- **shared** — columns *both* detectors score. The only like-for-like
  comparison, on identical cells.
- **new** — columns *only* the fallback scores. Judged on its own merit
  against the prevalence of those cells, which is the only honest reference
  at this imbalance.
- **whole table** — the combined system (conditional where it covers, fallback
  where it does not) against conditional-only, over an **identical cell
  universe**. This is the number that answers "does adding it help".

Combination is **conditional-priority**: where the conditional detector
covers a column, its score is used unchanged. So on a dataset it already
covers fully (`adult` 15/15, `smartfactory` 19/19) the combined system is
*identical* to conditional-only, and requirement 4 holds by construction
rather than by luck.

THE RARE-VALUE REQUIREMENT, SHARPENED
=====================================
Requirement 3 asks that legitimately rare values are not systematically
flagged. Reporting a correlation with frequency is not enough, because a
corrupted value usually *is* rare, so a good detector also correlates. The
sharp version is a **pure-rarity baseline**: score a cell by `1 - frequency`
of its value. If the fallback cannot beat that, it has learned nothing except
rarity — which is the failure five earlier mechanisms made.

COST
====
Requirement 5. Detection gains must survive pricing **both** error kinds:

    cost(threshold) = n_flagged * C_REVIEW  +  n_missed * C_MISS

swept over thresholds, reporting the minimum achievable. A detector that
flags nothing pays the full miss cost, so "find nothing" cannot look good.
Costs are a fixed ratio for every dataset (§C_REVIEW, C_MISS below) because
the comparison is *within* a dataset between two detectors; the absolute
scale is irrelevant and a per-dataset config would make the columns
incomparable.

REPRODUCIBILITY
===============
Requirement 6. Every number is the mean over `--splits` row subsamples with
the min-max range beside it. No single split is reported alone.

Run:  python experiments/w9b_text_fallback.py
      python experiments/w9b_text_fallback.py --datasets flights movies --splits 3
Writes results/w9b_text_fallback.md. Seeds 42..42+splits-1.
"""

from __future__ import annotations

import argparse
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
from src.detection_eval.labels import cell_labels  # noqa: E402
from src.detection_eval.metrics import best_f1, pr_auc, roc_auc  # noqa: E402
from src.detectors.conditional import ConditionalDetector  # noqa: E402
from src.detectors.text_fallback import TextFallbackDetector  # noqa: E402
from src.experiments.aggregate import Interval, bootstrap_metric  # noqa: E402

enable_unicode_stdout()

BASE_SEED = 42
ROW_CAP = 4000
N_SPLITS = 5

#: Fixed for every dataset: a missed error costs 5x a wasted review. The
#: comparison is within-dataset between detectors, so the absolute scale does
#: not matter and a per-dataset config would make columns incomparable.
C_REVIEW = 1.0
C_MISS = 5.0

OUT_MD = ROOT / "results" / "w9b_text_fallback.md"


def _f(v: Optional[float], nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}"


def _ci(iv: Optional[Interval], nd: int = 4) -> str:
    """Point estimate with a 95% bootstrap CI over cells."""
    if iv is None or iv.n == 0:
        return "—"
    if iv.hi - iv.lo < 1e-12:
        return f"{iv.mean:.{nd}f} (constant)"
    return f"{iv.mean:.{nd}f} [{iv.lo:.{nd}f}, {iv.hi:.{nd}f}]"


def boot(res: "SplitResult", ykey: str, skey: str,
         seed: int = BASE_SEED) -> Optional[Interval]:
    a = res.arrays
    if ykey not in a or skey not in a or a[ykey].size == 0:
        return None
    return bootstrap_metric(a[ykey], a[skey], pr_auc, n_boot=2000, seed=seed)


def _mr(vals: Sequence[Optional[float]], nd: int = 4) -> str:
    """mean [min, max] over splits, or a dash when nothing is defined."""
    xs = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    if not xs:
        return "—"
    if len(xs) == 1:
        return f"{xs[0]:.{nd}f}"
    return f"{np.mean(xs):.{nd}f} [{min(xs):.{nd}f}, {max(xs):.{nd}f}]"


def rarity_baseline(series: pd.Series) -> np.ndarray:
    """`1 - frequency` of each cell's value: pure rarity, nothing else."""
    s = series.astype("object")
    freq = s.value_counts(normalize=True, dropna=False)
    return (1.0 - s.map(freq).astype(float)).to_numpy()


def min_cost(y: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """
    Minimum of `n_flagged*C_REVIEW + n_missed*C_MISS` over all thresholds.

    Prices both error kinds, so a detector that flags nothing pays the full
    miss cost and cannot look good by abstaining.
    """
    if y.size == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    ys = y[order]
    n_pos = int(y.sum())
    # flag the top k, for every k
    tp = np.concatenate([[0], np.cumsum(ys)])
    k = np.arange(len(ys) + 1)
    cost = k * C_REVIEW + (n_pos - tp) * C_MISS
    return float(cost.min())


class SplitResult:
    def __init__(self) -> None:
        self.n_cols = 0
        self.cond_cols: List[str] = []
        self.fb_cols: List[str] = []
        self.shared: List[str] = []
        self.new: List[str] = []
        self.uncovered: List[str] = []
        self.unsupported: Dict[str, str] = {}
        self.m: Dict[str, Optional[float]] = {}


def run_split(dataset: str, rows: int, seed: int) -> SplitResult:
    res = SplitResult()
    loader = DataLoader("datasets")
    clean_full, dirty_full = loader.load(dataset)
    rng = np.random.default_rng(seed)
    if rows > 0 and len(dirty_full) > rows:
        idx = np.sort(rng.choice(len(dirty_full), size=rows, replace=False))
        clean = clean_full.iloc[idx].reset_index(drop=True)
        dirty = dirty_full.iloc[idx].reset_index(drop=True)
    else:
        clean = clean_full.reset_index(drop=True)
        dirty = dirty_full.reset_index(drop=True)

    cols = list(dirty.columns)
    res.n_cols = len(cols)
    labels = cell_labels(clean, dirty)

    # ── the two detectors, dirty frame only, profile=None ──────────────
    # W11: the detector seed VARIES per replicate. It was pinned to
    # BASE_SEED, so on a dataset at or below the row cap every replicate
    # was byte-identical and the reported range was zero by construction.
    cond = ConditionalDetector(mode="full", seed=seed).fit(dirty)
    cond_s = cond.score_cells(dirty)
    fb = TextFallbackDetector()
    fb_s = fb.score_cells(dirty)

    for c in cols:
        if np.unique(cond_s[c].to_numpy(dtype=float)).size > 1:
            res.cond_cols.append(c)
        rep = fb.reports_.get(c)
        if rep is not None and rep.scored:
            res.fb_cols.append(c)
        elif rep is not None and rep.reason.startswith("unsupported"):
            res.unsupported[c] = rep.reason
    cond_set, fb_set = set(res.cond_cols), set(res.fb_cols)
    res.shared = [c for c in cols if c in cond_set and c in fb_set]
    res.new = [c for c in cols if c in fb_set and c not in cond_set]
    res.uncovered = [c for c in cols if c not in cond_set and c not in fb_set]

    def flat(columns: Sequence[str], frame: pd.DataFrame) -> np.ndarray:
        if not columns:
            return np.array([], dtype=float)
        return frame[list(columns)].to_numpy(dtype=float).ravel(order="F")

    def flat_y(columns: Sequence[str]) -> np.ndarray:
        if not columns:
            return np.array([], dtype=int)
        return labels[list(columns)].to_numpy().astype(int).ravel(order="F")

    def flat_rarity(columns: Sequence[str]) -> np.ndarray:
        if not columns:
            return np.array([], dtype=float)
        return np.concatenate([rarity_baseline(dirty[c]) for c in columns])

    # ── R2a shared: identical cells, like for like ─────────────────────
    ys = flat_y(res.shared)
    if ys.size and 0 < ys.sum() < ys.size:
        res.m["shared_prev"] = float(ys.mean())
        res.m["shared_cond_pr"] = pr_auc(ys, flat(res.shared, cond_s))
        res.m["shared_fb_pr"] = pr_auc(ys, flat(res.shared, fb_s))
        res.m["shared_rar_pr"] = pr_auc(ys, flat_rarity(res.shared))

    # ── R2b new: the fallback's own merit ──────────────────────────────
    yn = flat_y(res.new)
    if yn.size and 0 < yn.sum() < yn.size:
        sn = flat(res.new, fb_s)
        rn = flat_rarity(res.new)
        res.m["new_prev"] = float(yn.mean())
        res.m["new_fb_pr"] = pr_auc(yn, sn)
        res.m["new_rar_pr"] = pr_auc(yn, rn)
        res.m["new_fb_roc"] = roc_auc(yn, sn)
        res.m["new_fb_f1"] = best_f1(yn, sn)[0]
        res.m["new_lift"] = (res.m["new_fb_pr"] / float(yn.mean())
                             if res.m["new_fb_pr"] else None)
        # R3 rare-value protection, as a comparison not a correlation
        res.m["new_beats_rarity"] = (
            1.0 if (res.m["new_fb_pr"] or 0) > (res.m["new_rar_pr"] or 0)
            else 0.0)
        # R5 cost on the new cells
        res.m["new_cost_fb"] = min_cost(yn, sn)
        res.m["new_cost_none"] = float(yn.sum()) * C_MISS

    # ── R4 whole table: identical cell universe, cond-priority ─────────
    combined = cond_s.copy()
    for c in res.new:
        combined[c] = fb_s[c]
    y_all = labels[cols].to_numpy().astype(int).ravel(order="F")
    res.m["whole_prev"] = float(y_all.mean())
    res.m["whole_cond_pr"] = pr_auc(y_all, flat(cols, cond_s))
    res.m["whole_comb_pr"] = pr_auc(y_all, flat(cols, combined))
    # Retained for the cell-level bootstrap (W11); only the canonical
    # replicate's arrays are kept, to bound memory.
    res.arrays = {
        "whole_y": y_all,
        "whole_cond": flat(cols, cond_s),
        "whole_comb": flat(cols, combined),
        "new_y": flat_y(res.new),
        "new_fb": flat(res.new, fb_s) if res.new else np.array([], dtype=float),
    }
    res.m["whole_cost_cond"] = min_cost(y_all, flat(cols, cond_s))
    res.m["whole_cost_comb"] = min_cost(y_all, flat(cols, combined))
    return res


# ═══════════════════════════════════════════════════════════════════════════


def build_report(
    per_dataset: Dict[str, List[SplitResult]], rows: int, splits: int
) -> str:
    L: List[str] = []
    a = L.append
    a("# W9.2b — does the text fallback detector earn its place?")
    a("")
    a("Generated by `experiments/w9b_text_fallback.py`. Specified in")
    a("`docs/w9_plan.md` §W9.2b with six requirements fixed before the run.")
    a(f"{splits} row splits of up to {rows:,} rows, seeds "
      f"{BASE_SEED}..{BASE_SEED + splits - 1}. Every figure is")
    a("**mean [min, max] over splits** — no single split is reported alone")
    a("(requirement 6).")
    a("")
    a("## 0. The design decision that matters most")
    a("")
    a("A fallback that covers new columns changes **which cells are")
    a("evaluated**, and that alone can manufacture an apparent improvement.")
    a("So three populations are kept strictly apart:")
    a("")
    a("- **shared** — columns both detectors score. The only like-for-like")
    a("  comparison, on identical cells.")
    a("- **new** — columns only the fallback scores, judged against the")
    a("  prevalence of those cells.")
    a("- **whole table** — combined system vs conditional-only over an")
    a("  identical cell universe.")
    a("")
    a("Combination is **conditional-priority**, so where the conditional")
    a("detector covers a column its score is used unchanged. On a dataset it")
    a("already covers fully the combined system is *identical* to")
    a("conditional-only, making requirement 4 hold by construction.")
    a("")

    a("## 1. Requirement 1 — coverage")
    a("")
    a("| dataset | columns | conditional | fallback | shared | **new** | uncovered | abstained |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    for ds, rs in per_dataset.items():
        a(f"| {ds} | {rs[0].n_cols} | "
          f"{_mr([len(r.cond_cols) for r in rs], 1)} | "
          f"{_mr([len(r.fb_cols) for r in rs], 1)} | "
          f"{_mr([len(r.shared) for r in rs], 1)} | "
          f"**{_mr([len(r.new) for r in rs], 1)}** | "
          f"{_mr([len(r.uncovered) for r in rs], 1)} | "
          f"{_mr([len(r.unsupported) for r in rs], 1)} |")
    a("")
    a("`abstained` counts columns the fallback declared **unsupported** rather")
    a("than guessing — too few frequent values for the `neighbour` signal to")
    a("fire. Honest coverage, not inflated coverage.")
    a("")

    a("## 2. Requirements 2 and 3 — shared cells (like for like)")
    a("")
    a("Identical cells, both detectors. `rarity` is the pure-rarity baseline")
    a("`1 - frequency`: if the fallback cannot beat it, it has learned nothing")
    a("but rarity, which is the failure five earlier mechanisms made.")
    a("")
    a("| dataset | prevalence | conditional PR-AUC | fallback PR-AUC | rarity PR-AUC |")
    a("|---|---:|---:|---:|---:|")
    for ds, rs in per_dataset.items():
        if not any("shared_cond_pr" in r.m for r in rs):
            a(f"| {ds} | — | — | — | — |")
            continue
        a(f"| {ds} | {_mr([r.m.get('shared_prev') for r in rs], 5)} | "
          f"{_mr([r.m.get('shared_cond_pr') for r in rs])} | "
          f"{_mr([r.m.get('shared_fb_pr') for r in rs])} | "
          f"{_mr([r.m.get('shared_rar_pr') for r in rs])} |")
    a("")

    a("## 3. Requirements 2, 3 and 5 — newly covered cells")
    a("")
    a("Columns **only** the fallback scores, so there is no conditional number")
    a("to compare: the references are the prevalence (chance) and the")
    a("pure-rarity baseline. `min cost` prices both error kinds")
    a(f"(flag = {C_REVIEW:g}, miss = {C_MISS:g}), and `cost if we flag")
    a("nothing` is what a detector that finds nothing pays — so abstaining")
    a("cannot look good.")
    a("")
    a("| dataset | new cols | prevalence | **fallback PR-AUC (95% CI)** | across seeds | rarity PR-AUC | lift | min cost | cost if nothing |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ds, rs in per_dataset.items():
        if not any("new_fb_pr" in r.m for r in rs):
            continue
        a(f"| {ds} | {_mr([len(r.new) for r in rs], 1)} | "
          f"{_mr([r.m.get('new_prev') for r in rs], 5)} | "
          f"**{_ci(boot(rs[0], 'new_y', 'new_fb'))}** | "
          f"{_mr([r.m.get('new_fb_pr') for r in rs])} | "
          f"{_mr([r.m.get('new_rar_pr') for r in rs])} | "
          f"{_mr([r.m.get('new_lift') for r in rs], 1)}x | "
          f"{_mr([r.m.get('new_cost_fb') for r in rs], 0)} | "
          f"{_mr([r.m.get('new_cost_none') for r in rs], 0)} |")
    a("")
    beats = []
    for ds, rs in per_dataset.items():
        vals = [r.m.get("new_beats_rarity") for r in rs
                if r.m.get("new_beats_rarity") is not None]
        if vals:
            beats.append((ds, float(np.mean(vals))))
    if beats:
        a("**Requirement 3, beats pure rarity** (fraction of splits):")
        a("")
        for ds, frac in beats:
            a(f"- `{ds}`: {frac:.0%}")
        a("")

    a("## 4. Requirement 4 — no regression, whole table")
    a("")
    a("Identical cell universe. Conditional-priority combination means a")
    a("fully-covered dataset must come out **exactly equal**.")
    a("")
    a("| dataset | prevalence | conditional PR-AUC (95% CI) | combined PR-AUC (95% CI) | delta | CIs overlap? |")
    a("|---|---:|---:|---:|---:|---|")
    for ds, rs in per_dataset.items():
        d = [(r.m.get("whole_comb_pr") or 0) - (r.m.get("whole_cond_pr") or 0)
             for r in rs]
        ic = boot(rs[0], "whole_y", "whole_cond")
        ib = boot(rs[0], "whole_y", "whole_comb")
        if ic is None or ib is None or ic.n == 0 or ib.n == 0:
            ov = "—"
        elif ic.hi < ib.lo or ib.hi < ic.lo:
            ov = "**no — separated**"
        else:
            ov = "yes — not separable"
        a(f"| {ds} | {_mr([r.m.get('whole_prev') for r in rs], 5)} | "
          f"{_ci(ic)} | {_ci(ib)} | {np.mean(d):+.4f} | {ov} |")
    a("")
    a("`across seeds` and the `delta` column are the spread over replicates;")
    a("the bracketed interval is a **95 % bootstrap CI over cells** from the")
    a("canonical replicate. Both are needed: the first measures algorithmic")
    a("randomness, the second measures sampling uncertainty of the metric.")
    a("`(constant)` marks a metric that really is identical on every")
    a("resample — a measured fact, not an absence of resampling.")
    a("")

    a("## 5. Verdict")
    a("")
    a("| requirement | result |")
    a("|---|---|")
    new_tot = sum(np.mean([len(r.new) for r in rs])
                  for rs in per_dataset.values())
    a(f"| 1 coverage increases | "
      f"{'**PASS**' if new_tot > 0 else '**FAIL**'} — "
      f"{new_tot:.1f} newly covered columns across all datasets |")
    pr_ok = [(ds, np.mean([r.m["new_fb_pr"] / r.m["new_prev"]
                           for r in rs if r.m.get("new_fb_pr")
                           and r.m.get("new_prev")]))
             for ds, rs in per_dataset.items()
             if any(r.m.get("new_fb_pr") for r in rs)]
    beat_chance = [ds for ds, lift in pr_ok if lift > 1.0]
    a(f"| 2 beats chance on new cells | "
      f"{'**PASS**' if beat_chance else '**FAIL**'} — above chance on "
      f"{len(beat_chance)}/{len(pr_ok)} datasets with new columns |")
    r3 = [ds for ds, frac in beats if frac >= 0.5]
    a(f"| 3 beats pure rarity | "
      f"{'**PASS**' if len(r3) > len(beats) / 2 else '**FAIL**'} — "
      f"beats it on a majority of splits on {len(r3)}/{len(beats)} datasets |")
    regress = []
    for ds, rs in per_dataset.items():
        d = np.mean([(r.m.get("whole_comb_pr") or 0)
                     - (r.m.get("whole_cond_pr") or 0) for r in rs])
        if d < -1e-9:
            regress.append((ds, d))
    a(f"| 4 no regression | "
      f"{'**PASS**' if not regress else '**FAIL**'} — "
      + ("no dataset degraded" if not regress
         else ", ".join(f"`{d}` {v:+.4f}" for d, v in regress)) + " |")
    cost_better = []
    for ds, rs in per_dataset.items():
        cc = [r.m.get("whole_cost_cond") for r in rs]
        cb = [r.m.get("whole_cost_comb") for r in rs]
        if all(v is not None for v in cc + cb):
            if np.mean(cb) < np.mean(cc) - 1e-9:
                cost_better.append(ds)
    a(f"| 5 cost impact | "
      f"{'**PASS**' if cost_better else '**FAIL**'} — combined is cheaper on "
      f"{len(cost_better)}/{len(per_dataset)} datasets |")
    nz = sum(1 for ds, rs in per_dataset.items()
             if (boot(rs[0], "whole_y", "whole_comb") or Interval(0, 0, 0, 0, 0)).n > 0)
    a(f"| 6 reproducibility | **PASS** — {splits} replicates with VARYING "
      f"detector seeds, plus a 95 % bootstrap CI over cells on {nz} "
      f"dataset(s). Replicate ranges alone are not evidence here: 5 of 11 "
      f"datasets sit at or below the {rows:,}-row cap and so are never "
      f"subsampled (see `docs/w11_plan.md`). |")
    a("")
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="W9.2b text fallback evaluation")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--rows", type=int, default=ROW_CAP)
    ap.add_argument("--splits", type=int, default=N_SPLITS)
    args = ap.parse_args(argv)

    names = args.datasets or sorted(
        d.name for d in (ROOT / "datasets").iterdir() if d.is_dir())

    print("W9.2b — text fallback evaluation")
    print(f"rows<= {args.rows}  splits {args.splits}\n")
    per: Dict[str, List[SplitResult]] = {}
    for ds in names:
        rs: List[SplitResult] = []
        t0 = time.time()
        for k in range(args.splits):
            try:
                rs.append(run_split(ds, args.rows, BASE_SEED + k))
            except Exception as exc:
                print(f"  {ds:<14} split {k} FAILED "
                      f"{type(exc).__name__}: {exc}")
        if not rs:
            continue
        per[ds] = rs
        print(f"  {ds:<14} new cols "
              f"{np.mean([len(r.new) for r in rs]):.1f}  "
              f"new PR-AUC {_mr([r.m.get('new_fb_pr') for r in rs])}  "
              f"whole {_mr([r.m.get('whole_cond_pr') for r in rs])} -> "
              f"{_mr([r.m.get('whole_comb_pr') for r in rs])}"
              f"   [{time.time()-t0:.0f}s]")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(build_report(per, args.rows, args.splits),
                      encoding="utf-8")
    print(f"\nWrote {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
