"""
experiments/w7_real_ceiling.py
------------------------------
W7.1e — the Bayes-optimal ceiling for the corruption that is **actually in
the shipped datasets**, rather than for corruption we injected.

WHY THIS SUPERSEDES THE SYNTHETIC GRID FOR THE HEADLINE NUMBER
=============================================================
`results/w7_ceiling.md` measured a ceiling of 0.2723 PR-AUC for `swap_valid`
on a low-tail column at adult's prevalence, and concluded a 6.4x improvement
was available. That number is for **frequency-weighted swap corruption**, and
inspecting the real data shows adult's corruption is nothing of the kind:

    real adult income corruption: 1,701 of 45,222 cells
    clean -> dirty transitions:   1 -> 0   x1701
                                  0 -> 1   x0

It is **entirely one-directional**: the majority class (1, at 75.22 % of the
column) is flipped to the minority class (0, at 24.78 %). So a corrupted cell
is indistinguishable *marginally* from an ordinary minority-class cell — among
all cells showing 0 there are roughly 11,206 genuine ones against 1,701
corrupted, so marginal evidence alone caps the posterior near 13 %.

`swap_valid` spreads replacements across the column's values by frequency,
which is an easier problem. The synthetic ceiling therefore **overstates**
what is achievable on the real table, and the honest headline needs the
ceiling for the real process.

THE COMPUTATION
===============
Same posterior, with `P_inject` read off the real corruption instead of an
injector's report:

    P(corrupt | v, x) = r * P_inject(v) / [ r*P_inject(v) + (1-r)*P_clean(v|x) ]

- `r` = the column's true corruption rate
- `P_inject(v)` = the empirical distribution of the values corrupted cells
  were turned INTO (for adult: all mass on `0`)
- `P_clean(v | x)` = cross-fitted oracle on the clean frame

All three read ground truth. This is an **oracle ceiling, non-deployable by
construction**, exactly like the `(ii)` AUC in W5.3 — it answers "how much
signal exists", never "what should we flag".

A cell whose value the corruption never produces gets `P_inject = 0` and so
posterior 0. For adult that means every cell showing `1` is provably clean,
which is correct and is itself part of why the problem is hard: all the
uncertainty is concentrated in the `0` cells.

Run:  python experiments/w7_real_ceiling.py
Writes results/w7_real_ceiling.md. Seed 42.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.console import enable_unicode_stdout  # noqa: E402
from src.data.loader import DataLoader  # noqa: E402
from src.detection_eval.labels import cell_labels  # noqa: E402
from src.detection_eval.metrics import pr_auc, roc_auc  # noqa: E402
from src.detectors.conditional import ConditionalDetector  # noqa: E402

# Reuse the oracle and the posterior from the synthetic study so the two
# ceilings are computed by byte-identical code.
from experiments.w7_ceiling import (  # noqa: E402
    bayes_posterior, oracle_conditional, rare_fraction,
)

enable_unicode_stdout()

SEED = 42
ROW_CAP = 8000
OUT_MD = ROOT / "results" / "w7_real_ceiling.md"


def analyse(dataset: str, rows: int) -> Optional[Dict[str, object]]:
    loader = DataLoader("datasets")
    clean_full, dirty_full = loader.load(dataset)
    rng = np.random.default_rng(SEED)
    if rows > 0 and len(dirty_full) > rows:
        idx = np.sort(rng.choice(len(dirty_full), size=rows, replace=False))
        clean = clean_full.iloc[idx].reset_index(drop=True)
        dirty = dirty_full.iloc[idx].reset_index(drop=True)
    else:
        clean = clean_full.reset_index(drop=True)
        dirty = dirty_full.reset_index(drop=True)

    # Labels come from the project's own tested builder, NOT an ad-hoc
    # comparison. A naive `astype(object)` compare makes 1 != 1.0 wherever
    # clean and dirty differ in dtype, which made breast_cancer's index
    # column look 100 % corrupted and got it selected as the target.
    labels = cell_labels(clean, dirty)
    diffs = {c: int(labels[c].to_numpy().sum()) for c in dirty.columns}
    column = max(diffs, key=lambda c: diffs[c])
    n_corr = diffs[column]
    if n_corr == 0:
        return None

    y = labels[column].to_numpy().astype(int)
    r = float(y.mean())

    # P_inject: what corrupted cells were turned INTO (reads ground truth).
    became = dirty[column].astype("object").to_numpy()[y == 1]
    keys = [np.nan if (isinstance(v, float) and pd.isna(v)) else v
            for v in became]
    vc = pd.Series(keys, dtype="object").value_counts(normalize=True,
                                                      dropna=False)
    p_inject = {k: float(v) for k, v in vc.items()}
    transitions = pd.Series([
        f"{a} -> {b}" for a, b in zip(
            clean[column].astype("object").to_numpy()[y == 1], became)
    ]).value_counts()

    # Oracle P_clean(v|x), cross-fitted on the clean frame.
    class_index, p_clean = oracle_conditional(clean, column, seed=SEED)
    post = bayes_posterior(dirty[column], class_index, p_clean, p_inject, r)

    # Our zero-knowledge detector on the real dirty frame.
    det = ConditionalDetector(mode="full", seed=SEED).fit(dirty)
    ours = det.score_cells(dirty)[column].to_numpy(dtype=float)

    # How much does marginal evidence alone allow? Among cells whose value
    # the corruption can produce, the share that is actually corrupted.
    in_support = np.array([
        (np.nan if (isinstance(v, float) and pd.isna(v)) else v) in p_inject
        for v in dirty[column].astype("object").to_numpy()])
    marginal_cap = (float(y[in_support].mean())
                    if in_support.any() else float("nan"))

    return {
        "dataset": dataset, "column": column, "n_rows": len(dirty),
        "n_corrupted": int(y.sum()), "rate": r,
        "transitions": transitions, "p_inject": p_inject,
        "n_in_support": int(in_support.sum()), "marginal_cap": marginal_cap,
        "pr_ceiling": pr_auc(y, post), "pr_ours": pr_auc(y, ours),
        "roc_ceiling": roc_auc(y, post), "roc_ours": roc_auc(y, ours),
        "rare_frac": rare_fraction(dirty[column]),
        "one_directional": bool(len(p_inject) == 1),
    }


def _f(v: Optional[float], nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*",
                    default=["adult", "breast_cancer", "smartfactory"])
    ap.add_argument("--rows", type=int, default=ROW_CAP)
    args = ap.parse_args(argv)

    out: List[Dict[str, object]] = []
    for ds in args.datasets:
        t0 = time.time()
        try:
            r = analyse(ds, args.rows)
        except Exception as exc:
            print(f"{ds}: FAILED {type(exc).__name__}: {exc}")
            continue
        if r is None:
            print(f"{ds}: no corruption found")
            continue
        out.append(r)
        cap = (r["pr_ours"] / r["pr_ceiling"]
               if r["pr_ours"] and r["pr_ceiling"] else None)
        print(f"{ds:<14} col=`{r['column']}` n_corr={r['n_corrupted']:,} "
              f"rate={r['rate']:.4f}  ceiling={_f(r['pr_ceiling'])} "
              f"ours={_f(r['pr_ours'])} captured={_f(cap, 3)} "
              f"[{time.time()-t0:.0f}s]")

    L: List[str] = []
    a = L.append
    a("# W7.1e — the ceiling for the corruption that is actually in the data")
    a("")
    a("Generated by `experiments/w7_real_ceiling.py`. Seed 42.")
    a("")
    a("`results/w7_ceiling.md` measured a ceiling of **0.2723** for")
    a("`swap_valid` at adult's prevalence and concluded a 6.4x improvement was")
    a("available. **That number is for frequency-weighted swap corruption, and")
    a("the real data is not that.** This file computes the ceiling for the")
    a("corruption actually present, using byte-identical oracle and posterior")
    a("code.")
    a("")
    a("## 1. What the real corruption actually is")
    a("")
    for r in out:
        a(f"### {r['dataset']} — `{r['column']}`")
        a("")
        a(f"{r['n_corrupted']:,} corrupted cells of {r['n_rows']:,} "
          f"(rate {r['rate']:.4f}).")
        a("")
        a("| clean -> dirty transition | count |")
        a("|---|---:|")
        for k, v in list(r["transitions"].items())[:8]:  # type: ignore[union-attr]
            a(f"| `{k}` | {v:,} |")
        if len(r["transitions"]) > 8:  # type: ignore[arg-type]
            a(f"| … {len(r['transitions']) - 8} more | |")
        a("")
        if r["one_directional"]:
            a("**Entirely one-directional.** Every corrupted cell was turned")
            a("into the same value, so a corrupted cell is indistinguishable")
            a("*marginally* from an ordinary cell holding that value.")
        a("")
        a(f"Cells whose value the corruption could have produced: "
          f"**{r['n_in_support']:,}**, of which "
          f"**{r['marginal_cap']:.4f}** are genuinely corrupted. That share is")
        a("the hard cap on what marginal evidence alone can tell you — every")
        a("cell outside the support is provably clean, so all the uncertainty")
        a("is concentrated in that subset.")
        a("")

    a("## 2. Ceiling versus what we achieve")
    a("")
    a("| dataset | column | rate | **ceiling PR-AUC** | ours | **captured** | ceiling vs chance |")
    a("|---|---|---:|---:|---:|---:|---:|")
    for r in out:
        cap = (r["pr_ours"] / r["pr_ceiling"]
               if r["pr_ours"] and r["pr_ceiling"] else None)
        lift = (r["pr_ceiling"] / r["rate"]
                if r["pr_ceiling"] and r["rate"] else None)
        a(f"| {r['dataset']} | `{r['column']}` | {r['rate']:.4f} | "
          f"**{_f(r['pr_ceiling'])}** | {_f(r['pr_ours'])} | "
          f"**{_f(cap, 3)}** | {_f(lift, 1)}x |")
    a("")
    a("Chance PR-AUC equals the prevalence, so `ceiling vs chance` is how much")
    a("signal exists at all. ROC-AUC for reference:")
    a("")
    a("| dataset | ceiling ROC | ours ROC |")
    a("|---|---:|---:|")
    for r in out:
        a(f"| {r['dataset']} | {_f(r['roc_ceiling'])} | {_f(r['roc_ours'])} |")
    a("")
    a("## 3. Why this is the number to quote")
    a("")
    a("The synthetic grid answers \"how much signal would exist if corruption")
    a("were frequency-weighted\". This answers \"how much exists in the table we")
    a("actually ship\". Where the two disagree, **this one governs**, and the")
    a("earlier 6.4x figure must not be quoted for real adult.")
    a("")
    a("Standing caveats, unchanged from `docs/w7_plan.md` §3.3: the ceiling is")
    a("an **estimate**, it depends on the oracle's `P_clean` (a weak oracle")
    a("understates it, so a captured fraction above 1.0 means the oracle lost")
    a("to the method rather than the method beating optimality), and the oracle")
    a("reads ground truth so it is **non-deployable by construction**.")
    a("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
