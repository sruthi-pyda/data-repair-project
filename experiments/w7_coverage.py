"""
experiments/w7_coverage.py
--------------------------
W7.0a — per-dataset **coverage**: which columns can each detector even form
an opinion about?

`results/w5_cross_domain.md` reports a lift figure for nine datasets and says
the method "generalises to six of nine". It reports no coverage at all, so a
reader cannot tell that on `flights` the conditional detector scores **0 of 7
columns**. The lift numbers are not wrong — cross-domain evaluates the
**rule** detector, and the 0/7 is the **conditional** detector — but
presenting effectiveness without applicability invites exactly the
misreading a reviewer will make.

Two different notions of coverage, reported separately because they are not
the same thing:

**Conditional detector** (`src/detectors/conditional.py`, the Wave 1
contribution). It *explicitly skips* columns it cannot model and records why:
`id_like_cardinality`, `high_cardinality_non_numeric`, `constant`. A skipped
column gets an all-zero score vector. This is the applicability limit of the
Wave 1 work, and it is severe on text-heavy tables.

**Rule detector** (`src/detection/detector.py`, the shipped Phase 3). It has
no skip list — it simply never fires on some columns, because none of its six
checks apply. A column it never flags is one it has no opinion about, which
for ranking purposes is the same practical limitation.

Both are reported as `scored / total`, with the genuinely-corrupted columns
each detector cannot see marked, because those are corruptions that are
invisible **by construction** rather than merely missed.

Leakage: clean data is read only to label which columns are corrupted, after
coverage has been determined. Coverage itself is a property of the dirty
frame and the detector.

Run:  python experiments/w7_coverage.py
Writes results/w7_coverage.md. Seed 42.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.console import enable_unicode_stdout  # noqa: E402
from src.data.loader import DataLoader  # noqa: E402
from src.data.profiler import profile_dataset  # noqa: E402
from src.detection.detector import Detector  # noqa: E402
from src.detection_eval.harness import RuleDetectorAdapter  # noqa: E402
from src.detection_eval.labels import cell_labels  # noqa: E402
from src.detectors.conditional import ConditionalDetector  # noqa: E402
from src.pipeline.runner import default_flag_outliers  # noqa: E402

enable_unicode_stdout()

SEED = 42
ROW_CAP = 8000
IQR_K = 3.0
OUT_MD = ROOT / "results" / "w7_coverage.md"


def measure(dataset: str) -> Dict[str, object]:
    loader = DataLoader("datasets")
    clean_full, dirty_full = loader.load(dataset)
    rng = np.random.default_rng(SEED)
    if len(dirty_full) > ROW_CAP:
        idx = np.sort(rng.choice(len(dirty_full), size=ROW_CAP, replace=False))
        dirty = dirty_full.iloc[idx].reset_index(drop=True)
        clean = clean_full.iloc[idx].reset_index(drop=True)
    else:
        dirty = dirty_full.reset_index(drop=True)
        clean = clean_full.reset_index(drop=True)

    cols = list(dirty.columns)
    labels = cell_labels(clean, dirty)
    corrupted = {c for c in cols if int(labels[c].to_numpy().sum()) > 0}

    # ── conditional detector: explicit skip list ────────────────────────
    cond = ConditionalDetector(mode="full", seed=SEED).fit(dirty)
    cond_scores = cond.score_cells(dirty)
    cond_skipped: Dict[str, str] = dict(
        getattr(cond, "skipped_columns_", {}) or {})
    # A column with a constant score vector carries no information even if it
    # is not on the skip list, so treat it the same way.
    for c in cols:
        if c in cond_skipped:
            continue
        if np.unique(cond_scores[c].to_numpy(dtype=float)).size <= 1:
            cond_skipped[c] = "constant score"
    cond_scored = [c for c in cols if c not in cond_skipped]

    # ── rule detector: columns it never flags ───────────────────────────
    paths = loader.discover(dataset)
    profile = profile_dataset(dataset, paths, clean, dirty)
    rule = Detector(profile, iqr_k=IQR_K,
                    flag_statistical_outliers=default_flag_outliers(dataset))
    rule_scores = RuleDetectorAdapter(detector=rule, name="rule").score_cells(
        dirty, profile)
    rule_scored = [c for c in cols
                   if float(np.nansum(rule_scores[c].to_numpy(dtype=float))) > 0]
    rule_silent = [c for c in cols if c not in rule_scored]

    return {
        "dataset": dataset,
        "n_cols": len(cols),
        "n_corrupted_cols": len(corrupted),
        "cond_scored": cond_scored,
        "cond_skipped": cond_skipped,
        "cond_invisible": sorted(c for c in cond_skipped if c in corrupted),
        "rule_scored": rule_scored,
        "rule_silent": rule_silent,
        "rule_invisible": sorted(c for c in rule_silent if c in corrupted),
        "corrupted": sorted(corrupted),
    }


def main() -> int:
    root = ROOT / "datasets"
    names = sorted(d.name for d in root.iterdir() if d.is_dir())
    rows: List[Dict[str, object]] = []

    hdr = f"{'dataset':<14}{'cols':>5}{'cond':>9}{'rule':>9}  invisible corruptions"
    print(hdr)
    print("-" * len(hdr))
    for ds in names:
        t0 = time.time()
        try:
            r = measure(ds)
        except Exception as exc:
            print(f"{ds:<14}  FAILED {type(exc).__name__}: {str(exc)[:40]}")
            continue
        rows.append(r)
        print(f"{ds:<14}{r['n_cols']:>5}"
              f"{len(r['cond_scored']):>6}/{r['n_cols']:<2}"
              f"{len(r['rule_scored']):>6}/{r['n_cols']:<2}"
              f"  cond {len(r['cond_invisible'])}, rule "
              f"{len(r['rule_invisible'])}   [{time.time()-t0:.0f}s]")

    L: List[str] = []
    a = L.append
    a("# W7.0a — detector coverage: which columns can we form an opinion about?")
    a("")
    a("Generated by `experiments/w7_coverage.py`. Seed 42, row cap "
      f"{ROW_CAP:,}.")
    a("")
    a("`results/w5_cross_domain.md` reports a lift figure per dataset and")
    a("concludes the method \"generalises to six of nine\". It reports **no**")
    a("coverage, so a reader cannot tell that the conditional detector scores")
    a("**0 of 7** of `flights`' columns. The lift numbers are not wrong —")
    a("cross-domain evaluates the **rule** detector while the 0/7 is the")
    a("**conditional** detector — but reporting effectiveness without")
    a("applicability invites the misreading a reviewer will make.")
    a("")
    a("Two notions of coverage, kept separate because they differ:")
    a("")
    a("- **conditional** — columns it does not *explicitly skip*")
    a("  (`id_like_cardinality`, `high_cardinality_non_numeric`, `constant`),")
    a("  plus any whose score vector is constant anyway. This is the")
    a("  applicability limit of the Wave 1 contribution.")
    a("- **rule** — columns it ever flags. It has no skip list; it simply")
    a("  never fires where none of its six checks apply, which for ranking")
    a("  purposes is the same practical limit.")
    a("")
    a("## Coverage")
    a("")
    a("| dataset | columns | conditional scored | rule scored | corrupted columns | corrupted & invisible to conditional | corrupted & invisible to rule |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        a(f"| {r['dataset']} | {r['n_cols']} | "
          f"{len(r['cond_scored'])} | {len(r['rule_scored'])} | "
          f"{r['n_corrupted_cols']} | {len(r['cond_invisible'])} | "
          f"{len(r['rule_invisible'])} |")
    a("")
    a("The last two columns are corruptions that are invisible **by")
    a("construction** rather than merely missed: the detector forms no")
    a("opinion about that column at all, so no threshold, calibration or")
    a("cost model can recover them.")
    a("")
    a("## Why each column is skipped (conditional detector)")
    a("")
    for r in rows:
        if not r["cond_skipped"]:
            a(f"- **{r['dataset']}**: nothing skipped "
              f"({r['n_cols']}/{r['n_cols']} scored).")
            continue
        reasons: Dict[str, List[str]] = {}
        for col, why in sorted(r["cond_skipped"].items()):  # type: ignore[union-attr]
            key = str(why).split("(")[0]
            reasons.setdefault(key, []).append(col)
        parts = "; ".join(
            f"`{k}` x{len(v)} ({', '.join(v[:4])}"
            + (", …" if len(v) > 4 else "") + ")"
            for k, v in sorted(reasons.items()))
        a(f"- **{r['dataset']}** — {len(r['cond_skipped'])} of "
          f"{r['n_cols']} skipped: {parts}")
    a("")
    a("## What this means")
    a("")
    worst = sorted(rows, key=lambda r: len(r["cond_scored"]) / r["n_cols"])[:4]
    a("The conditional detector's applicability is **severe on text-heavy")
    a("tables**, which is what most real dirty-data benchmarks are:")
    a("")
    for r in worst:
        a(f"- `{r['dataset']}`: {len(r['cond_scored'])}/{r['n_cols']} columns "
          f"scored")
    a("")
    a("So Wave 1's conditional modelling — the contribution that took adult")
    a("from 0.0025 to 0.0120 PR-AUC — **cannot run on most columns of most")
    a("of these datasets**. That is a scope limitation and belongs in the")
    a("paper as one, not as a footnote.")
    a("")
    a("It also explains the cross-domain pattern without a separate story:")
    a("where the rule detector works it is because the corruption is")
    a("syntactic, and syntactic corruption is exactly what rules see without")
    a("needing a model of the column.")
    a("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
