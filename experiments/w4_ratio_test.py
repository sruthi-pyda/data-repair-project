"""
experiments/w4_ratio_test.py
----------------------------
One decisive test, from W4.1's central finding.

Ren et al., *Likelihood Ratios for Out-of-Distribution Detection* (NeurIPS
2019), show that a **density** such as `P(x | context)` is confounded by
population-level background statistics, and that the correct statistic is a
**likelihood ratio** against a background model.

That is exactly our failure mode. `docs/research_program.md` §8 measured, on
adult, that of the top 5,000 cells scored by `1 − P(value | row)`, **zero**
are in the tampered `income` column — the top is `native_country`, where a
rare country is genuinely improbable and completely clean.

For a value-swap corruption the ratio reduces to

    log [ P(value) / P(value | row) ]

- rare-but-clean: low density under **both** models → ratio ≈ 1 → not flagged
- an income flip: marginally **common**, conditionally **improbable**
  → ratio large → flagged

`src/detectors/conditional.py` already implements this as `mode="ratio"`.
Wave 1 only ever judged it on overall PR-AUC, where it looked slightly worse.
It was never asked the question that matters:

    **does the ratio put income cells at the TOP of the ranking?**

PR-AUC over 678,330 cells is dominated by the long tail. The top of the
ranking is what an analyst or a label budget actually sees, so that is what
this measures.

Falsification (stated before running, per W4.1): the mechanism is refuted if
the top 1,701 ratio-scored cells still contain **zero** income cells.

Run:  python experiments/w4_ratio_test.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.console import enable_unicode_stdout  # noqa: E402
from src.data.loader import DataLoader  # noqa: E402
from src.data.profiler import profile_dataset  # noqa: E402
from src.detection_eval.labels import cell_labels  # noqa: E402
from src.detection_eval.metrics import pr_auc, roc_auc  # noqa: E402
from src.detectors.conditional import ConditionalDetector  # noqa: E402

DATASET = "adult"
TARGET_COLUMN = "income"
MODES = ("full", "marginal", "ratio")


def main() -> int:
    enable_unicode_stdout()
    t0 = time.time()
    loader = DataLoader("datasets")
    clean, dirty = loader.load(DATASET)
    profile = profile_dataset(DATASET, loader.discover(DATASET), clean, dirty)
    labels = cell_labels(clean, dirty)

    y = labels.to_numpy().ravel(order="F").astype(int)
    cols = np.repeat(np.array(dirty.columns, dtype=object), len(dirty))
    n_pos = int(y.sum())
    in_target = cols == TARGET_COLUMN

    print("=" * 86)
    print("Does a LIKELIHOOD RATIO put the tampered column at the top,")
    print("where a raw DENSITY does not?   (Ren et al., NeurIPS 2019)")
    print("=" * 86)
    print(f"{DATASET}: {len(y):,} cells, {n_pos:,} corrupted "
          f"(prevalence {y.mean():.5f}), all in `{TARGET_COLUMN}`")
    print(f"Refuted if the top {n_pos:,} ratio-scored cells contain 0 "
          f"`{TARGET_COLUMN}` cells.\n")

    header = (f"{'mode':<10} {'PR-AUC all':>11} {'PR-AUC col':>11} "
              f"{'ROC all':>9} {'in top-N':>9} {'recall@N':>9}  top column")
    print(header)
    print("-" * len(header))

    rows = []
    for mode in MODES:
        det = ConditionalDetector(mode=mode, seed=42).fit(dirty, profile=profile)
        scores = det.score_cells(dirty, profile=profile)
        s = scores.to_numpy().ravel(order="F").astype(float)

        top = np.argsort(-s)[:n_pos]
        n_in_target = int(in_target[top].sum())
        tp = int(y[top].sum())
        dom_col, dom_n = Counter(cols[top]).most_common(1)[0]

        rows.append((mode, n_in_target, tp))
        print(f"{mode:<10} {pr_auc(y, s):>11.4f} "
              f"{pr_auc(y[in_target], s[in_target]):>11.4f} "
              f"{roc_auc(y, s):>9.4f} {n_in_target:>9,} "
              f"{tp / n_pos:>9.4f}  {dom_col} ({dom_n:,})")

    print()
    best = max(rows, key=lambda r: r[1])
    if best[1] == 0:
        print("VERDICT: REFUTED. No mode places a single tampered cell in the")
        print("top-N. The likelihood ratio does not fix column localisation on")
        print("this dataset, and the honest claim stands: per-cell surprise is")
        print("solved; locating a valid-to-valid corruption from the dirty")
        print("table alone is not.")
    else:
        print(f"VERDICT: NOT REFUTED. `{best[0]}` places {best[1]:,} "
              f"`{TARGET_COLUMN}` cells in the top {n_pos:,} "
              f"({best[2]:,} true positives).")
        print("The ratio moves the top of the ranking where the density could")
        print("not. Worth building on.")
    print(f"\n[done] {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
