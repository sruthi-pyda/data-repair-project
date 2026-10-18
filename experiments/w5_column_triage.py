"""
experiments/w5_column_triage.py
-------------------------------
Route W5.3 — reviewer **gap 2**, the hardest open problem in the project.

Pre-registered in `docs/w5_3_column_triage_plan.md`, committed before this
file existed. Read that document first: it fixes the hypothesis, the
headline metric, the negative-set confound and four falsification
conditions **in advance**, so the result cannot be reinterpreted after the
fact. The same discipline correctly killed the likelihood-ratio idea in
W4.1, and this experiment is set up so it can kill this idea too.

THE PROBLEM
===========
Our detector finds the odd cell *inside* a column and has no idea *which*
column to look at. On adult, of the 5,000 cells it scores highest,
**zero** are in the tampered `income` column — the top is `native_country`,
where a rare country is genuinely improbable and completely clean.

THE PROBE
=========
For each column `c`: plant corruption of type `T` in `c` **only**, refit the
detector, and measure how well it recovers the planted cells. Because we
planted them, their positions are free labels that leak nothing.

    d_c = ROC-AUC(planted vs non-planted cells of column c)
    t_c = -d_c

**H1**: a column that is *already* corrupted has its model fitted on
polluted targets, so planted corruption is **harder** to find there. Rank
columns by ascending `d_c`.

The detector is fitted with ``profile=None``, which keeps it fully
self-supervised — no clean data of any kind enters the probe. Ground truth
is used only to score the ranking after it exists, and for the clearly
marked non-deployable diagnostic below.

TWO AUCs, BOTH REPORTED (plan §5b)
==================================
Planting into an already-corrupted column leaves **genuine** corruptions in
the negative set, which score high and drag the AUC down for a reason that
has nothing to do with H1. So:

  (i)  deployable  — negatives are all non-planted cells. ZERO labels.
       **The falsification conditions are evaluated on this one**, because
       it is the only version that could run in deployment.
  (ii) mechanism   — negatives restricted to genuinely-clean cells. Uses
       labels, is NOT deployable, and exists only to say which mechanism is
       driving the result.

(i) pass + (ii) pass -> H1 as stated. (i) pass + (ii) fail -> the probe
works via contamination; usable, but H1 must not be claimed.

ROW CAP — the one approximation, and why it is sound
====================================================
A full-adult probe is 136 s to fit + 52 s to score; 15 columns x 4 types is
~3 hours for one dataset. Rows are therefore subsampled (seeded) to
``--rows``. Row subsampling preserves each column's **corruption fraction**
in expectation, and that fraction is exactly what H1's mechanism depends
on, so the probe's logic is untouched. Detector hyperparameters are left at
their shipped defaults; the cap is the only approximation.

Run:
    python experiments/w5_column_triage.py
    python experiments/w5_column_triage.py --rows 4000 --datasets adult

Writes results/w5_column_triage.md. Deterministic: seed 42.
"""

from __future__ import annotations

import argparse
import dataclasses
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
from src.datagen.corruptor import inject  # noqa: E402
from src.detection_eval.labels import cell_labels  # noqa: E402
from src.detection_eval.metrics import roc_auc  # noqa: E402
from src.detectors.conditional import ConditionalDetector  # noqa: E402

enable_unicode_stdout()

SEED = 42
DATASETS = ("adult", "breast_cancer", "smartfactory")

#: Probe types. `swap_valid` matches adult's true corruption and is the
#: circularity control (plan §5); the other three are structurally
#: different.
PROBE_TYPES = ("out_of_range", "format", "missing", "swap_valid")
MATCHING_TYPE = "swap_valid"

#: Fraction of the TARGET COLUMN's cells to plant. `Corruptor.rate` is a
#: fraction of ALL cells, so the rate passed is this divided by the column
#: count.
PLANT_FRACTION = 0.05

OUT_MD = ROOT / "results" / "w5_column_triage.md"


# ═══════════════════════════════════════════════════════════════════════════
# Records
# ═══════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class ColumnProbe:
    column: str
    is_corrupted: bool
    n_genuine: int
    n_planted: int
    d_deployable: Optional[float]
    d_mechanism: Optional[float]
    note: str = ""

    @property
    def eligible(self) -> bool:
        return self.n_planted > 0 and self.d_deployable is not None


@dataclasses.dataclass
class TypeResult:
    dataset: str
    probe_type: str
    probes: List[ColumnProbe]
    col_auc_deployable: Optional[float]
    col_auc_mechanism: Optional[float]
    col_auc_baseline: Optional[float]
    seconds: float

    def ranked(self) -> List[ColumnProbe]:
        """Eligible columns, most-suspicious first (ascending d_c)."""
        ok = [p for p in self.probes if p.eligible]
        return sorted(ok, key=lambda p: (p.d_deployable, p.column))


# ═══════════════════════════════════════════════════════════════════════════
# Core
# ═══════════════════════════════════════════════════════════════════════════


def subsample(
    clean: pd.DataFrame, dirty: pd.DataFrame, cap: int, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Seeded row subsample, both frames kept aligned.

    Row sampling preserves each column's corruption FRACTION in
    expectation, which is the quantity H1 depends on.
    """
    if cap <= 0 or len(dirty) <= cap:
        return clean.reset_index(drop=True), dirty.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(dirty), size=cap, replace=False))
    return (
        clean.iloc[idx].reset_index(drop=True),
        dirty.iloc[idx].reset_index(drop=True),
    )


def column_level_auc(
    probes: Sequence[ColumnProbe], key: str
) -> Optional[float]:
    """
    How well does `t_c = -d_c` separate corrupted columns from clean ones?

    Chance is 0.5. Computed over **eligible** columns only: a column the
    injector cannot touch produces no score, and inventing one would be
    fabricating a number.
    """
    rows = [
        (p.is_corrupted, getattr(p, key))
        for p in probes
        if p.eligible and getattr(p, key) is not None
    ]
    if not rows:
        return None
    y = np.array([r[0] for r in rows], dtype=int)
    # t_c = -d_c : lower detectability => more suspicious
    t = np.array([-float(r[1]) for r in rows], dtype=float)
    return roc_auc(y, t)


def probe_dataset(
    dataset: str, cap: int, verbose: bool = True
) -> Tuple[List[TypeResult], Dict[str, object]]:
    loader = DataLoader("datasets")
    clean_full, dirty_full = loader.load(dataset)
    clean, dirty = subsample(clean_full, dirty_full, cap, SEED)
    labels = cell_labels(clean, dirty)
    columns = list(dirty.columns)
    n_cols = len(columns)

    genuine = {c: int(labels[c].to_numpy().sum()) for c in columns}
    corrupted_cols = {c for c in columns if genuine[c] > 0}

    meta: Dict[str, object] = {
        "n_rows": len(dirty),
        "n_rows_full": len(dirty_full),
        "n_columns": n_cols,
        "corrupted_columns": sorted(corrupted_cols),
        "genuine": genuine,
    }

    if verbose:
        print(f"\n{'=' * 78}")
        print(
            f"{dataset}: {len(dirty):,} rows (of {len(dirty_full):,}) x "
            f"{n_cols} columns  ·  {len(corrupted_cols)} corrupted column(s): "
            f"{', '.join(sorted(corrupted_cols)) or 'none'}"
        )
        print("=" * 78)

    # ── F4 baseline: mean raw score per column, one fit, no injection ────
    t0 = time.time()
    det0 = ConditionalDetector(mode="full", seed=SEED).fit(dirty)
    base = det0.score_cells(dirty).mean()
    if verbose:
        print(f"  baseline fit+score: {time.time() - t0:.0f}s")
    base_probes = [
        ColumnProbe(
            column=c,
            is_corrupted=c in corrupted_cols,
            n_genuine=genuine[c],
            n_planted=1,  # marks it eligible for the baseline ranking
            # Baseline suspicion is the mean score, and t = -d, so negate
            # to keep "higher mean score => more suspicious".
            d_deployable=-float(base[c]),
            d_mechanism=None,
        )
        for c in columns
    ]
    baseline_auc = column_level_auc(base_probes, "d_deployable")
    if verbose:
        top_base = sorted(columns, key=lambda c: -float(base[c]))[:3]
        print(
            f"  BASELINE (mean raw score) column-AUC = "
            f"{baseline_auc if baseline_auc is None else round(baseline_auc, 4)}"
            f"   top-3: {', '.join(top_base)}"
        )

    results: List[TypeResult] = []
    rate = PLANT_FRACTION / n_cols

    for ptype in PROBE_TYPES:
        t_type = time.time()
        probes: List[ColumnProbe] = []
        if verbose:
            print(f"\n  --- probe type: {ptype} (rate {rate:.5f}) ---")

        for c in columns:
            protected = [x for x in columns if x != c]
            try:
                injected, report = inject(
                    dirty, ptype, rate, seed=SEED, protected_columns=protected
                )
            except Exception as exc:  # documented, never invented
                probes.append(
                    ColumnProbe(c, c in corrupted_cols, genuine[c], 0,
                                None, None, f"inject failed: {exc}")
                )
                continue

            planted_rows = [r for (r, col) in report.positions if col == c]
            if not planted_rows:
                probes.append(
                    ColumnProbe(c, c in corrupted_cols, genuine[c], 0,
                                None, None, "not eligible for this type")
                )
                if verbose:
                    print(f"    {c:<22} ineligible")
                continue

            det = ConditionalDetector(mode="full", seed=SEED).fit(injected)
            sc = det.score_cells(injected)[c].to_numpy(dtype=float)

            # A column the detector SKIPS (ID-like) scores all-zero, so its
            # AUC is exactly 0.5 — and since t = -d, that 0.5 would rank it
            # as the MOST suspicious column in the table. It is an absence
            # of measurement, not a measurement, so it is excluded, the same
            # way this project reports `—` rather than 0.0 for undefined.
            # Observed on breast_cancer: `Unnamed: 0` and `Sample code
            # number` both sat at exactly d = 0.5000 and took the top two
            # triage ranks for every probe type.
            n_distinct = int(np.unique(sc).size)
            if n_distinct <= 1:
                probes.append(
                    ColumnProbe(c, c in corrupted_cols, genuine[c],
                                len(planted_rows), None, None,
                                "detector skips this column "
                                f"(constant score {sc[0]:.3g})")
                )
                if verbose:
                    print(f"    {c:<22} skipped by detector "
                          f"(constant score)")
                continue

            planted = np.zeros(len(injected), dtype=bool)
            planted[np.asarray(planted_rows, dtype=int)] = True
            gen = labels[c].to_numpy().astype(bool)

            d_i = roc_auc(planted.astype(int), sc)
            # (ii) negatives restricted to genuinely-clean cells. NOT
            # deployable: uses ground truth. Diagnostic only.
            keep = planted | (~planted & ~gen)
            d_ii = roc_auc(planted[keep].astype(int), sc[keep])

            probes.append(
                ColumnProbe(c, c in corrupted_cols, genuine[c],
                            len(planted_rows), d_i, d_ii)
            )
            if verbose:
                mark = "CORRUPT" if c in corrupted_cols else "clean  "
                print(
                    f"    {c:<22} {mark}  planted {len(planted_rows):>5}  "
                    f"d(i)={_f(d_i)}  d(ii)={_f(d_ii)}"
                )

        res = TypeResult(
            dataset=dataset,
            probe_type=ptype,
            probes=probes,
            col_auc_deployable=column_level_auc(probes, "d_deployable"),
            col_auc_mechanism=column_level_auc(probes, "d_mechanism"),
            col_auc_baseline=baseline_auc,
            seconds=time.time() - t_type,
        )
        results.append(res)
        if verbose:
            order = [p.column for p in res.ranked()[:3]]
            print(
                f"  => column-AUC(i)={_f(res.col_auc_deployable)}  "
                f"(ii)={_f(res.col_auc_mechanism)}  "
                f"baseline={_f(baseline_auc)}  "
                f"[{res.seconds:.0f}s]  top-3: {', '.join(order)}"
            )

    return results, meta


def _f(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.4f}"


# ═══════════════════════════════════════════════════════════════════════════
# Verdict
# ═══════════════════════════════════════════════════════════════════════════


def verdicts(
    all_res: Dict[str, List[TypeResult]], meta: Dict[str, Dict[str, object]]
) -> List[str]:
    """Evaluate F1–F4 exactly as the plan fixed them, on the (i) AUC."""
    L: List[str] = []
    a = L.append

    a("## 5. Pre-registered falsification conditions")
    a("")
    a("Evaluated on the **deployable** AUC (i), as plan §5b requires.")
    a("")

    # ── F1 ───────────────────────────────────────────────────────────────
    a("### F1 — column-level AUC must beat 0.5")
    a("")
    a("| dataset | probe type | column-AUC (i) | verdict |")
    a("|---|---|---:|---|")
    f1_pass: Dict[Tuple[str, str], bool] = {}
    for ds, results in all_res.items():
        for r in results:
            v = r.col_auc_deployable
            ok = v is not None and v > 0.5
            f1_pass[(ds, r.probe_type)] = ok
            a(f"| {ds} | {r.probe_type} | {_f(v)} | "
              f"{'pass' if ok else '**REFUTED**'} |")
    a("")

    # ── F2 ───────────────────────────────────────────────────────────────
    a("### F2 — adult: `income` must be in the top 5 of 15")
    a("")
    a("| probe type | rank of `income` | of | one-sided p | verdict |")
    a("|---|---:|---:|---:|---|")
    for r in all_res.get("adult", []):
        ranked = r.ranked()
        names = [p.column for p in ranked]
        if "income" not in names:
            a(f"| {r.probe_type} | — | {len(names)} | — | "
              f"**n/a (ineligible)** |")
            continue
        rank = names.index("income") + 1
        p_val = rank / len(names)
        ok = rank <= 5
        a(f"| {r.probe_type} | {rank} | {len(names)} | {p_val:.3f} | "
          f"{'pass' if ok else '**REFUTED**'} |")
    a("")

    # ── F3 ───────────────────────────────────────────────────────────────
    a("### F3 — circularity")
    a("")
    a("Declared circular if the probe passes F1 **only** for "
      f"`{MATCHING_TYPE}` (which matches adult's true corruption) and fails "
      "for every structurally-different type.")
    a("")
    for ds in all_res:
        match = f1_pass.get((ds, MATCHING_TYPE), False)
        others = [
            t for t in PROBE_TYPES
            if t != MATCHING_TYPE and f1_pass.get((ds, t), False)
        ]
        if match and not others:
            a(f"- **{ds}**: **CIRCULAR** — passes only for "
              f"`{MATCHING_TYPE}`. In deployment the true corruption type is "
              "unknown, so a probe that needs to match it is worthless.")
        elif others:
            a(f"- **{ds}**: not circular — also passes for "
              f"{', '.join('`' + t + '`' for t in others)}.")
        else:
            a(f"- **{ds}**: n/a — does not pass F1 for `{MATCHING_TYPE}`.")
    a("")

    # ── F4 ───────────────────────────────────────────────────────────────
    a("### F4 — must beat the mean-raw-score baseline")
    a("")
    a("The baseline is the thing already known to fail "
      "(`native_country` first on adult), so not beating it means the probe "
      "adds nothing.")
    a("")
    a("| dataset | probe type | probe (i) | baseline | delta | verdict |")
    a("|---|---|---:|---:|---:|---|")
    for ds, results in all_res.items():
        for r in results:
            v, b = r.col_auc_deployable, r.col_auc_baseline
            if v is None or b is None:
                a(f"| {ds} | {r.probe_type} | {_f(v)} | {_f(b)} | — | n/a |")
                continue
            ok = v > b
            a(f"| {ds} | {r.probe_type} | {v:.4f} | {b:.4f} | "
              f"{v - b:+.4f} | {'pass' if ok else '**REFUTED**'} |")
    a("")

    # ── mechanism ────────────────────────────────────────────────────────
    a("### Which mechanism? (i) vs (ii)")
    a("")
    a("| dataset | probe type | (i) deployable | (ii) mechanism | reading |")
    a("|---|---|---:|---:|---|")
    for ds, results in all_res.items():
        for r in results:
            i_v, ii_v = r.col_auc_deployable, r.col_auc_mechanism
            if i_v is None:
                reading = "no result"
            elif i_v <= 0.5:
                reading = "probe does not work"
            elif ii_v is not None and ii_v > 0.5:
                reading = "H1 as stated"
            else:
                reading = "**works, but via negative-set contamination — H1 unsupported**"
            a(f"| {ds} | {r.probe_type} | {_f(i_v)} | {_f(ii_v)} | "
              f"{reading} |")
    a("")

    overall = any(f1_pass.values())
    a("### Overall")
    a("")
    if not overall:
        a("**The probe is REFUTED on every dataset and every probe type.**")
        a("")
        a("Self-detectability does not localise corruption to a column. This")
        a("is a measured negative result, pre-registered, and it strengthens")
        a("rather than weakens the position `docs/literature.md` already")
        a("records: no published system separates rare-but-valid from")
        a("corrupted using one table and no labels, and neither do we. The")
        a("honest claim for the paper is a *measured impossibility* on this")
        a("axis, not a missing feature.")
    else:
        passing = [k for k, v in f1_pass.items() if v]
        a(f"F1 passes for {len(passing)} (dataset, type) pair(s): "
          + ", ".join(f"{d}/{t}" for d, t in passing) + ".")
        a("")
        a("Read this against F3 and F4 before claiming anything: a pass that")
        a("is circular (F3) or that fails to beat the mean-score baseline")
        a("(F4) licenses no claim at all.")
    a("")
    return L


def write_report(
    all_res: Dict[str, List[TypeResult]],
    meta: Dict[str, Dict[str, object]],
    cap: int,
) -> str:
    L: List[str] = []
    a = L.append

    a("# W5.3 — column triage by self-supervised injection")
    a("")
    a("Route W5.3, reviewer **gap 2**. Generated by")
    a("`experiments/w5_column_triage.py`. Pre-registered in")
    a("`docs/w5_3_column_triage_plan.md`, committed **before** this")
    a("experiment was written. Seed 42.")
    a("")
    a("## 0. The question")
    a("")
    a("Our detector finds the odd cell *inside* a column but has no idea")
    a("*which* column to look at: on adult, **zero** of the 5,000")
    a("highest-scored cells are in the tampered `income` column.")
    a("")
    a("The probe plants corruption in one column at a time, refits, and")
    a("measures how well it recovers what it planted:")
    a("")
    a("    d_c = ROC-AUC(planted vs non-planted cells of column c)")
    a("    t_c = -d_c")
    a("")
    a("**H1**: an already-corrupted column has its model fitted on polluted")
    a("targets, so planted corruption is *harder* to find there — lower")
    a("`d_c`. The detector is fitted with `profile=None`, so **no clean data")
    a("of any kind enters the probe**.")
    a("")
    a("## 1. Setup")
    a("")
    a("| dataset | rows used | of | columns | corrupted columns |")
    a("|---|---:|---:|---:|---|")
    for ds, m in meta.items():
        cc = m["corrupted_columns"]
        a(f"| {ds} | {m['n_rows']:,} | {m['n_rows_full']:,} | "
          f"{m['n_columns']} | {len(cc)}: "
          f"{', '.join('`' + c + '`' for c in cc) or 'none'} |")
    a("")
    a(f"Rows are subsampled (seeded) to {cap:,} where larger. Row sampling")
    a("preserves each column's corruption **fraction**, which is the")
    a("quantity H1 depends on; detector hyperparameters are shipped")
    a("defaults. A full-adult probe is 136 s to fit plus 52 s to score, so")
    a("15 columns x 4 types would be ~3 hours for one dataset.")
    a("")
    a(f"Plant fraction: {PLANT_FRACTION:.0%} of the target column's cells.")
    a("A column the injector cannot touch for a given type is reported")
    a("**ineligible** and excluded, rather than given an invented score.")
    a("")

    a("## 2. Per-column results")
    a("")
    for ds, results in all_res.items():
        a(f"### {ds}")
        a("")
        for r in results:
            a(f"**probe type `{r.probe_type}`** — column-AUC (i) "
              f"{_f(r.col_auc_deployable)}, (ii) {_f(r.col_auc_mechanism)}, "
              f"baseline {_f(r.col_auc_baseline)} ({r.seconds:.0f}s)")
            a("")
            a("| rank | column | truly corrupted? | genuine cells | planted | d(i) | d(ii) |")
            a("|---:|---|---|---:|---:|---:|---:|")
            for i, p in enumerate(r.ranked(), start=1):
                a(f"| {i} | `{p.column}` | "
                  f"{'**yes**' if p.is_corrupted else 'no'} | "
                  f"{p.n_genuine:,} | {p.n_planted:,} | "
                  f"{_f(p.d_deployable)} | {_f(p.d_mechanism)} |")
            skipped = [p for p in r.probes if not p.eligible]
            if skipped:
                a("")
                a("Ineligible: " + ", ".join(
                    f"`{p.column}` ({p.note})" for p in skipped))
            a("")

    a("## 3. Reading the ranking")
    a("")
    a("`t_c = -d_c`, so rank 1 is the column where planted corruption was")
    a("**hardest** to recover — the most suspicious column under H1. A")
    a("probe type whose `d_c` is ~1.0 for every column carries no column")
    a("signal at all: the planted corruption is trivially detectable")
    a("everywhere, so the statistic cannot discriminate. That is a real")
    a("limit of the method and is visible directly in the tables above.")
    a("")

    L.extend(verdicts(all_res, meta))
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=8000,
                    help="row cap per dataset (0 = no cap)")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = ap.parse_args(argv)

    t0 = time.time()
    all_res: Dict[str, List[TypeResult]] = {}
    meta: Dict[str, Dict[str, object]] = {}
    for ds in args.datasets:
        res, m = probe_dataset(ds, args.rows)
        all_res[ds] = res
        meta[ds] = m

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(write_report(all_res, meta, args.rows), encoding="utf-8")
    print(f"\nWrote {OUT_MD.relative_to(ROOT)}  [{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
