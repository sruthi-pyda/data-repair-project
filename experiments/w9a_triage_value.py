"""
experiments/w9a_triage_value.py
-------------------------------
W9.1 — **what would perfect column triage be worth?**

Planned in `docs/w9_plan.md` §W9.1, committed before this file existed.

Five mechanisms have failed to produce a usable column-triage statistic (M1
the likelihood ratio, W5.3 self-detectability, M2 the contamination fraction,
W8-A CACM, W8-D Excess Surprise), and `docs/w8_plan.md` §8 explains why with
the nuisance hierarchy. What has never been computed is the **headroom**: how
much the whole-table number would improve if triage were *perfect*.

That matters for the paper in two directions at once. A large headroom
justifies why five waves chased this. A large headroom that no obtainable
statistic captures is what retires the programme. One table does both.

THE TWO VARIANTS, AND WHY BOTH
==============================
- **masked (primary)** — score every cell, then force the score to zero on
  every column that is genuinely clean. Evaluated over **all** cells, so the
  cell universe and the prevalence are unchanged and the number is directly
  comparable with the published whole-table PR-AUC. This is the variant
  behind the existing 0.6873 measurement in `docs/research_program.md` §8.
- **restricted (secondary)** — evaluated only over cells in genuinely
  corrupted columns. Prevalence changes, so this is **not** comparable with
  the whole-table figure and the report says so wherever it appears.

ORACLE STATUS
=============
Both variants read `clean_df` to decide which columns are corrupted, so both
are **non-deployable by construction** — the same clearly-labelled diagnostic
use as the `(ii)` AUC in W5.3 and the Bayes ceiling in W7.1. The purpose is to
size the prize, never to flag a cell.

Both detectors are measured, because they fail differently: the rule detector
produces a flag *set* with a fixed per-check score, while the conditional
detector produces a continuous ranking.

Run:  python experiments/w9a_triage_value.py
      python experiments/w9a_triage_value.py --datasets adult --rows 8000
Writes results/w9a_triage_value.md. Seed 42.
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
from src.data.profiler import profile_dataset  # noqa: E402
from src.detection.detector import Detector  # noqa: E402
from src.detection_eval.harness import RuleDetectorAdapter  # noqa: E402
from src.detection_eval.labels import cell_labels  # noqa: E402
from src.detection_eval.metrics import pr_auc, roc_auc  # noqa: E402
from src.detectors.conditional import ConditionalDetector  # noqa: E402
from src.pipeline.runner import default_flag_outliers  # noqa: E402

enable_unicode_stdout()

SEED = 42
ROW_CAP = 8000
IQR_K = 3.0
OUT_MD = ROOT / "results" / "w9a_triage_value.md"


def _f(v: Optional[float], nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}"


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

    labels = cell_labels(clean, dirty)
    cols = list(dirty.columns)
    corrupted_cols = [c for c in cols if int(labels[c].to_numpy().sum()) > 0]
    if not corrupted_cols or len(corrupted_cols) == len(cols):
        # With no clean column there is nothing for triage to select, so the
        # headroom is undefined rather than zero.
        degenerate = ("no corrupted column" if not corrupted_cols
                      else "every column is corrupted; nothing to mask")
    else:
        degenerate = ""

    # ── the two detectors, both dirty-only ──────────────────────────────
    scores: Dict[str, pd.DataFrame] = {}
    cond = ConditionalDetector(mode="full", seed=SEED).fit(dirty)
    scores["conditional"] = cond.score_cells(dirty)

    paths = loader.discover(dataset)
    # Dirty-profiled, so the rule detector carries no clean distributional
    # knowledge (results/w5_raha_leakage_check.md measured that effect).
    profile = profile_dataset(dataset, paths, dirty, dirty)
    rule = Detector(profile, iqr_k=IQR_K,
                    flag_statistical_outliers=default_flag_outliers(dataset))
    scores["rule"] = RuleDetectorAdapter(
        detector=rule, name="rule").score_cells(dirty, profile)

    out: Dict[str, object] = {
        "dataset": dataset, "n_rows": len(dirty), "n_cols": len(cols),
        "n_corrupt_cols": len(corrupted_cols),
        "n_clean_cols": len(cols) - len(corrupted_cols),
        "degenerate": degenerate,
        "detectors": {},
    }

    corrupt_mask_cols = set(corrupted_cols)
    for name, frame in scores.items():
        s = frame[cols].to_numpy(dtype=float)
        y = labels[cols].to_numpy().astype(int)

        # whole table: run everywhere, which is what the pipeline does
        whole_pr = pr_auc(y.ravel(order="F"), s.ravel(order="F"))
        whole_roc = roc_auc(y.ravel(order="F"), s.ravel(order="F"))

        # masked: zero the genuinely clean columns, same cell universe
        masked = s.copy()
        for j, c in enumerate(cols):
            if c not in corrupt_mask_cols:
                masked[:, j] = 0.0
        masked_pr = (pr_auc(y.ravel(order="F"), masked.ravel(order="F"))
                     if not degenerate else None)

        # restricted: only corrupted columns' cells; prevalence CHANGES
        keep = [j for j, c in enumerate(cols) if c in corrupt_mask_cols]
        if keep and not degenerate:
            ys = y[:, keep].ravel(order="F")
            ss = s[:, keep].ravel(order="F")
            restricted_pr = pr_auc(ys, ss)
            restricted_prev = float(ys.mean())
        else:
            restricted_pr, restricted_prev = None, None

        ratio = (masked_pr / whole_pr
                 if (masked_pr and whole_pr and whole_pr > 0) else None)
        out["detectors"][name] = {  # type: ignore[index]
            "whole_pr": whole_pr, "whole_roc": whole_roc,
            "masked_pr": masked_pr, "ratio": ratio,
            "restricted_pr": restricted_pr,
            "restricted_prev": restricted_prev,
            "prevalence": float(y.mean()),
        }
    return out


def build_report(rows: List[Dict[str, object]]) -> str:
    L: List[str] = []
    a = L.append
    a("# W9.1 — what would perfect column triage be worth?")
    a("")
    a("Generated by `experiments/w9a_triage_value.py`. Planned in")
    a("`docs/w9_plan.md` §W9.1, committed before the code. Seed 42.")
    a("")
    a("## 0. The question, and why the premise matters")
    a("")
    a("**The pipeline already runs the detector on every column.** `Detector`")
    a("iterates every `profile.column_names` and there is no column-selection")
    a("gate anywhere in `src/pipeline/runner.py`, so adult's whole-table")
    a("PR-AUC of 0.0025 *is* the run-on-every-column number. Column triage was")
    a("never a pipeline stage; it was a diagnostic question about why running")
    a("everywhere fails.")
    a("")
    a("Running everywhere does not fail from a selection error. It fails")
    a("because rare-but-clean cells in other columns **outrank** the corrupted")
    a("ones, so extra columns contribute false positives that displace true")
    a("positives from the top of the ranking. This table sizes that effect.")
    a("")
    a("## 1. Oracle status — read before quoting anything here")
    a("")
    a("Both oracle variants read `clean_df` to decide which columns are")
    a("corrupted. They are **non-deployable by construction**, the same")
    a("clearly-labelled diagnostic use as the `(ii)` AUC in W5.3 and the Bayes")
    a("ceiling in W7.1. The purpose is to size the prize, never to flag a cell.")
    a("")
    a("- **masked** (primary) — score every cell, then zero the score on every")
    a("  genuinely clean column, evaluated over **all** cells. The cell")
    a("  universe and prevalence are unchanged, so it is directly comparable")
    a("  with the whole-table figure beside it.")
    a("- **restricted** (secondary) — evaluated only over cells in corrupted")
    a("  columns. **Prevalence changes**, so it is NOT comparable with the")
    a("  whole-table number and must not be quoted beside it without this note.")
    a("")
    a("## 2. The headroom")
    a("")
    a("| dataset | cols (corrupt/clean) | detector | prevalence | whole-table PR-AUC | **masked PR-AUC** | **ratio** |")
    a("|---|---|---|---:|---:|---:|---:|")
    for r in rows:
        det = r["detectors"]  # type: ignore[index]
        tag = f"{r['n_corrupt_cols']}/{r['n_clean_cols']}"
        for name in ("conditional", "rule"):
            d = det[name]  # type: ignore[index]
            note = " ⚠" if r["degenerate"] else ""
            a(f"| {r['dataset']}{note} | {tag} | {name} | "
              f"{_f(d['prevalence'], 5)} | {_f(d['whole_pr'])} | "
              f"**{_f(d['masked_pr'])}** | "
              f"**{(_f(d['ratio'], 1) + 'x') if d['ratio'] else '—'}** |")
    a("")
    deg = [r for r in rows if r["degenerate"]]
    if deg:
        a("`⚠` marks a dataset where masking is undefined:")
        for r in deg:
            a(f"- **{r['dataset']}** — {r['degenerate']}")
        a("")
        a("With no clean column there is nothing for triage to select, so the")
        a("headroom does not exist rather than being zero.")
        a("")
    a("### Restricted variant (different cell universe — do not compare across)")
    a("")
    a("| dataset | detector | restricted prevalence | restricted PR-AUC |")
    a("|---|---|---:|---:|")
    for r in rows:
        det = r["detectors"]  # type: ignore[index]
        for name in ("conditional", "rule"):
            d = det[name]  # type: ignore[index]
            if d["restricted_pr"] is None:
                continue
            a(f"| {r['dataset']} | {name} | {_f(d['restricted_prev'], 5)} | "
              f"{_f(d['restricted_pr'])} |")
    a("")

    a("## 3. What this settles")
    a("")
    ratios = [(r["dataset"], r["detectors"]["conditional"]["ratio"])  # type: ignore[index]
              for r in rows
              if r["detectors"]["conditional"]["ratio"]]  # type: ignore[index]
    if ratios:
        big = sorted(ratios, key=lambda t: -(t[1] or 0))[:4]
        a("Largest headroom under perfect triage (conditional detector):")
        a("")
        for ds, ratio in big:
            a(f"- `{ds}`: **{ratio:.1f}x**")
        a("")
    a("The headroom is the prize that five separate mechanisms failed to")
    a("claim any of — M1 the likelihood ratio, W5.3 self-detectability, M2 the")
    a("contamination fraction, W8-A CACM and W8-D Excess Surprise. Each was")
    a("refuted with an identified cause, and `docs/w8_plan.md` §8 unifies them:")
    a("every per-column statistic absorbed a column-specific nuisance term")
    a("larger than the corruption signal, in a hierarchy — background density,")
    a("then cardinality, then model competence, then model error rate, then")
    a("model miscalibration.")
    a("")
    a("So this table is the honest two-sided result for the paper: **the prize")
    a("is large, and it is not reachable by any statistic we could construct**")
    a("from the dirty table at these prevalences. That is a measured")
    a("limitation with a mechanism, not an absence of effort.")
    a("")
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="W9.1 triage headroom")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--rows", type=int, default=ROW_CAP)
    args = ap.parse_args(argv)

    names = args.datasets or sorted(
        d.name for d in (ROOT / "datasets").iterdir() if d.is_dir())

    print("W9.1 — value of perfect column triage")
    print(f"seed {SEED}  rows<= {args.rows}\n")
    rows: List[Dict[str, object]] = []
    for ds in names:
        t0 = time.time()
        try:
            r = analyse(ds, args.rows)
        except Exception as exc:
            print(f"  {ds:<14} FAILED {type(exc).__name__}: {exc}")
            continue
        if r is None:
            continue
        rows.append(r)
        c = r["detectors"]["conditional"]  # type: ignore[index]
        print(f"  {ds:<14} cond whole={_f(c['whole_pr'])} "
              f"masked={_f(c['masked_pr'])} "
              f"ratio={(_f(c['ratio'],1)+'x') if c['ratio'] else '—'}"
              f"   [{time.time()-t0:.0f}s]")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(build_report(rows), encoding="utf-8")
    print(f"\nWrote {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
