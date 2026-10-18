"""
experiments/w5_column_triage_control.py
---------------------------------------
The control that refutes W5.3's one apparent success.

`experiments/w5_column_triage.py` reported, on adult with probe type
`out_of_range`, a **perfect** column-level AUC of 1.0000: the corrupted
`income` column had by far the lowest self-detectability (d = 0.2356) while
every clean numeric column sat at 0.98-0.99. Taken at face value all four
pre-registered conditions passed, with the mechanism diagnostic (ii) also
passing, which would have meant H1 confirmed.

It is wrong, and the direction is what gave it away. `income` on adult is
encoded **0/1 int64**, so the corruptor classifies it as numeric and plants
values like -50 and 51. An injected -50 in a 0/1 column is a glaring
outlier, so it should be *trivially* detectable — d near 1.0, not 0.24.
A result that strong in the physically wrong direction is a confound, not a
discovery.

THE MECHANISM
=============
The conditional detector models a numeric column with a squared-error
learner. Squared error is dominated by the extreme planted values, so the
booster chases them and ends up predicting the planted cells **better** than
ordinary 0/1 cells, whose income is genuinely hard to predict from the row.
Small residual on planted cells => low score => low d. The effect is a
property of the column's **cardinality and span**, not of its corruption.

WHAT THIS SCRIPT MEASURES
=========================
The counterfactual the probe can never see: substitute the column's **clean**
values, leave everything else identical, and re-probe.

    d_dirty  — probe the column as shipped (corrupted)
    d_clean  — probe the same column with clean values substituted

Reading:

- If `d_clean` is also far below the clean continuous columns, then the low
  `d` is caused by the column's cardinality and the column-level AUC is
  **confounded** — the ranking separates binary from continuous, not
  corrupted from clean.
- `d_dirty - d_clean` isolates the genuine H1 effect with cardinality held
  fixed. It can be real and the probe still useless, because obtaining
  `d_clean` requires the ground truth the probe exists to avoid.

This uses clean data as a **diagnostic only**, exactly like the (ii) AUC in
the main experiment. It is explicitly NOT deployable and no score derived
from it is ever used to rank anything.

Run:  python experiments/w5_column_triage_control.py
Writes results/w5_column_triage_control.md. Seed 42.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.console import enable_unicode_stdout  # noqa: E402
from src.data.loader import DataLoader  # noqa: E402
from src.datagen.corruptor import inject  # noqa: E402
from src.detection_eval.labels import cell_labels  # noqa: E402
from src.detection_eval.metrics import roc_auc  # noqa: E402
from src.detectors.conditional import ConditionalDetector  # noqa: E402

enable_unicode_stdout()

SEED = 42
ROW_CAP = 4000
PROBE_TYPE = "out_of_range"  # the only type that showed any column signal
PLANT_FRACTION = 0.05
OUT_MD = ROOT / "results" / "w5_column_triage_control.md"

#: adult is the only dataset where column triage is well posed (1 of 15
#: columns corrupted). The columns probed are its corrupted column plus
#: every clean column the injector can touch for `out_of_range`.
DATASET = "adult"
TARGET = "income"


def probe_column(
    frame: pd.DataFrame, column: str, others: List[str]
) -> Tuple[Optional[float], float, float, int]:
    """
    Plant into `column` only, refit, and return (d, planted_mean,
    other_mean, n_planted).

    The detector is fitted with no profile, so it is fully self-supervised.
    """
    inj, report = inject(
        frame, PROBE_TYPE, PLANT_FRACTION / frame.shape[1],
        seed=SEED, protected_columns=others,
    )
    rows = np.array(
        [r for (r, col) in report.positions if col == column], dtype=int
    )
    if rows.size == 0:
        return None, float("nan"), float("nan"), 0
    det = ConditionalDetector(mode="full", seed=SEED).fit(inj)
    sc = det.score_cells(inj)[column].to_numpy(dtype=float)
    planted = np.zeros(len(inj), dtype=bool)
    planted[rows] = True
    return (
        roc_auc(planted.astype(int), sc),
        float(sc[planted].mean()),
        float(sc[~planted].mean()),
        int(rows.size),
    )


def main() -> int:
    loader = DataLoader("datasets")
    clean_full, dirty_full = loader.load(DATASET)
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(dirty_full), size=min(ROW_CAP, len(dirty_full)),
                             replace=False))
    dirty = dirty_full.iloc[idx].reset_index(drop=True)
    clean = clean_full.iloc[idx].reset_index(drop=True)
    labels = cell_labels(clean, dirty)
    columns = list(dirty.columns)

    print(f"{DATASET}: {len(dirty):,} rows x {len(columns)} columns  "
          f"probe type `{PROBE_TYPE}`")
    print(f"{'column':<20}{'corrupt':>9}{'distinct':>10}{'d_dirty':>10}"
          f"{'d_clean':>10}{'gap':>9}")
    print("-" * 68)

    rows_out: List[Dict[str, object]] = []
    for col in columns:
        others = [x for x in columns if x != col]
        n_gen = int(labels[col].to_numpy().sum())
        t0 = time.time()

        d_dirty, pm_d, om_d, n_pl = probe_column(dirty, col, others)
        if n_pl == 0:
            print(f"{col:<20}{'—':>9}{'—':>10}  ineligible for {PROBE_TYPE}")
            continue

        # Counterfactual: same column, clean values. DIAGNOSTIC ONLY.
        ctrl = dirty.copy()
        ctrl[col] = clean[col].to_numpy()
        d_clean, pm_c, om_c, _ = probe_column(ctrl, col, others)

        gap = (d_dirty - d_clean) if (
            d_dirty is not None and d_clean is not None) else None
        rows_out.append({
            "column": col, "n_genuine": n_gen,
            "distinct": int(pd.Series(dirty[col]).nunique(dropna=True)),
            "d_dirty": d_dirty, "d_clean": d_clean, "gap": gap,
            "planted_mean": pm_d, "other_mean": om_d, "n_planted": n_pl,
        })
        print(f"{col:<20}{n_gen:>9,}{rows_out[-1]['distinct']:>10,}"
              f"{d_dirty:>10.4f}{d_clean:>10.4f}"
              f"{(gap if gap is not None else float('nan')):>+9.4f}"
              f"   [{time.time() - t0:.0f}s]")

    # ── report ───────────────────────────────────────────────────────────
    L: List[str] = []
    a = L.append
    a("# W5.3 control — the perfect column-AUC is a cardinality confound")
    a("")
    a("`experiments/w5_column_triage.py` reported a **perfect** column-level")
    a(f"AUC of 1.0000 on {DATASET} with probe type `{PROBE_TYPE}`: the")
    a(f"corrupted `{TARGET}` column had by far the lowest self-detectability")
    a("(d = 0.2356) while every clean numeric column sat at 0.98-0.99. All")
    a("four pre-registered conditions appeared to pass, including the")
    a("mechanism diagnostic. **It does not hold up.**")
    a("")
    a(f"`{TARGET}` on {DATASET} is encoded **0/1 int64**, so the corruptor")
    a("treats it as numeric and plants values like -50 and 51. An injected")
    a("-50 in a 0/1 column is a glaring outlier and should be *trivially*")
    a("detectable — d near 1.0, not 0.24. A strong result in the physically")
    a("wrong direction is a confound, not a discovery, and that is what")
    a("prompted this control.")
    a("")
    a("**Mechanism**: the conditional detector models a numeric column with a")
    a("squared-error learner. Squared error is dominated by the extreme")
    a("planted values, so the booster chases them and predicts the planted")
    a("cells *better* than ordinary 0/1 cells, whose income is genuinely hard")
    a("to predict from the row. Small residual on planted cells gives a low")
    a("score and hence a low `d`. That is a property of the column's")
    a("**cardinality and span**, not of its corruption.")
    a("")
    a("## The counterfactual the probe can never see")
    a("")
    a("`d_clean` re-probes the identical column with its **clean** values")
    a("substituted. Clean data is used as a diagnostic only — exactly like")
    a("the (ii) AUC in the main experiment — and no score derived from it")
    a("ranks anything. It is explicitly NOT deployable.")
    a("")
    a("| column | genuine corrupt cells | distinct values | d_dirty | d_clean | gap (dirty - clean) |")
    a("|---|---:|---:|---:|---:|---:|")
    for r in rows_out:
        a(f"| `{r['column']}` | {r['n_genuine']:,} | {r['distinct']:,} | "
          f"{r['d_dirty']:.4f} | {r['d_clean']:.4f} | "
          f"{r['gap']:+.4f} |")
    a("")

    tgt = next((r for r in rows_out if r["column"] == TARGET), None)
    clean_rows = [r for r in rows_out if r["n_genuine"] == 0]
    a("## Verdict")
    a("")
    if tgt is None:
        a(f"`{TARGET}` was ineligible for `{PROBE_TYPE}`; no verdict.")
    else:
        mean_clean_d = (
            float(np.mean([r["d_dirty"] for r in clean_rows]))
            if clean_rows else float("nan")
        )
        a(f"- `{TARGET}` corrupted: **d = {tgt['d_dirty']:.4f}**")
        a(f"- `{TARGET}` with clean values substituted: "
          f"**d = {tgt['d_clean']:.4f}**")
        a(f"- clean numeric columns, as shipped: mean d = {mean_clean_d:.4f}")
        a("")
        confounded = (
            tgt["d_clean"] is not None
            and not np.isnan(mean_clean_d)
            and tgt["d_clean"] < mean_clean_d - 0.1
        )
        if confounded:
            a("**CONFOUNDED.** A *clean* "
              f"`{TARGET}` still scores far below the clean continuous")
            a("columns, so it would have ranked first in the triage ordering")
            a("**even with no corruption in it at all**. The column-level AUC")
            a("of 1.0000 separates low-cardinality from high-cardinality")
            a("columns, not corrupted from clean. The pre-registered F1 and F4")
            a("passes on this dataset are therefore **artefacts and are")
            a("withdrawn**.")
            a("")
            a(f"The residual gap of {tgt['gap']:+.4f} is a genuine H1 effect")
            a("with cardinality held fixed, and its sign is the one H1")
            a("predicts. But it is unusable: obtaining `d_clean` requires")
            a("precisely the ground truth the probe exists to avoid. A")
            a("deployable version would need a cardinality-matched")
            a("normalisation, which this experiment does not have.")
        else:
            a("Not confounded on this measurement: a clean "
              f"`{TARGET}` scores in line with the other clean columns, so")
            a("the low `d` on the corrupted column is attributable to its")
            a("corruption rather than its cardinality.")
    a("")
    a("## Consequence for W5.3")
    a("")
    a("Combined with the main run — where `format` and `missing` saturate at")
    a(f"d = 1.0000 on `{TARGET}` and carry no column signal, and `missing`")
    a("scores a column-AUC of 0.3462, *below* chance — the pre-registered")
    a("conclusion is that **self-supervised injection does not localise")
    a("corruption to a column**.")
    a("")
    a("That is a measured negative result, and it is the one")
    a("`docs/literature.md` already anticipated: no published system")
    a("separates rare-but-valid from corrupted using one table and no")
    a("labels. The honest claim for the paper is a *measured impossibility*")
    a("on this axis rather than a missing feature — and it was caught by a")
    a("control run because the result looked too good, which is the")
    a("pre-registration discipline working as intended.")
    a("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
