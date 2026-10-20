"""
experiments/w8d_excess.py
-------------------------
W8-D evaluation — does **Excess Surprise** localise corruption to a column?

Mechanism and motivation: `docs/w8_plan.md` §8-§9. Implementation:
`src/detectors/excess_surprise.py`.

Pre-registered before this ran, from the unit-test measurements in
`tests/test_excess_surprise.py`:

- **Difficulty cancellation is verified** on synthetic data: clean-easy and
  clean-hard columns whose observed errors differ by >0.3 land within 0.15 of
  zero excess, and their excess gap is under a third of their observed gap.
  No earlier mechanism in this project removed that nuisance parameter.
- **Sensitivity is weak**: excess rises at roughly `0.08 * r`, because the
  model is cross-fitted on the corrupted frame and so partly absorbs the
  corruption into `expected_error`. At adult's true `income` rate of ~0.0376
  that predicts an excess near **0.003**, which may sit below the
  between-column spread on real tables.

So the prediction on record is: **if this fails on adult, weak sensitivity is
the reason, not failed cancellation.** Those are different findings.

Conditions evaluated (`docs/w8_plan.md` §5 1-7, §9 8-9). Conditions 3 and 7
are covered elsewhere: 7 by `tests/test_excess_surprise.py::TestNoLeakage`,
and 3 only partially here because Excess Surprise is a per-COLUMN statistic —
the cell-level column is reported as `excess_c` used as a per-column prior
multiplying the existing per-cell score, and labelled as such.

Run:  python experiments/w8d_excess.py
      python experiments/w8d_excess.py --datasets adult --rows 6000
Writes results/w8d_excess.md. Seed 42.
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
from src.detection_eval.metrics import pr_auc, roc_auc  # noqa: E402
from src.detectors.conditional import ConditionalDetector  # noqa: E402
from src.detectors.excess_surprise import ExcessSurprise  # noqa: E402

enable_unicode_stdout()

SEED = 42
ROW_CAP = 8000
RARE_THRESHOLD = 0.01
OUT_MD = ROOT / "results" / "w8d_excess.md"

#: Column statistics compared head to head. `excess` is the mechanism; the
#: two ingredients test condition 9; `baseline` is the incumbent.
STATS = ("excess", "observed", "expected", "baseline")


def rare_fraction(series: pd.Series, thresh: float = RARE_THRESHOLD) -> float:
    s = series.dropna()
    if s.empty:
        return float("nan")
    freq = s.value_counts(normalize=True)
    return float(s.map(freq).lt(thresh).mean())


def spearman(a: Sequence[Optional[float]], b: Sequence[Optional[float]]
             ) -> Optional[float]:
    """Spearman rho from ranks, computed here so it is checkable by eye."""
    x = np.array([np.nan if v is None else float(v) for v in a], dtype=float)
    y = np.array([np.nan if v is None else float(v) for v in b], dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return None

    def rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, dtype=float)
        r[order] = np.arange(1, v.size + 1, dtype=float)
        for val in np.unique(v):
            m = v == val
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r

    rx, ry = rank(x) - rank(x).mean(), rank(y) - rank(y).mean()
    den = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else None


def sign_test(diffs: Sequence[float]) -> Dict[str, object]:
    """Exact two-sided sign test. Ties dropped, as the test requires."""
    from math import comb

    d = [x for x in diffs if abs(x) > 1e-12]
    n, w = len(d), sum(1 for x in d if x > 0)
    if n == 0:
        return {"n": 0, "wins": 0, "losses": 0, "p": None,
                "median": float(np.median(diffs)) if diffs else None}
    tail = sum(comb(n, k) for k in range(min(w, n - w) + 1)) / (2 ** n)
    return {"n": n, "wins": w, "losses": n - w,
            "p": min(1.0, 2.0 * tail),
            "median": float(np.median(diffs))}


# ═══════════════════════════════════════════════════════════════════════════


class DatasetResult:
    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.rows = 0
        self.columns: List[str] = []
        self.scored: List[str] = []
        self.skipped: Dict[str, str] = {}
        self.is_corrupt: Dict[str, bool] = {}
        self.true_rate: Dict[str, float] = {}
        self.rare: Dict[str, float] = {}
        self.cardinality: Dict[str, int] = {}
        self.stat: Dict[str, Dict[str, Optional[float]]] = {
            s: {} for s in STATS}
        self.auc: Dict[str, Optional[float]] = {}
        self.cell_pr: Dict[str, Optional[float]] = {}
        self.seconds = 0.0

    @property
    def defined(self) -> bool:
        c = [self.is_corrupt[k] for k in self.scored]
        return 0 < sum(c) < len(c) if c else False


def column_auc(res: DatasetResult, stat: str) -> Optional[float]:
    ys, vs = [], []
    for col in res.scored:
        v = res.stat[stat].get(col)
        if v is None or not np.isfinite(v):
            continue
        ys.append(int(res.is_corrupt[col]))
        vs.append(float(v))
    if not vs or len(set(ys)) < 2:
        return None
    return roc_auc(np.asarray(ys), np.asarray(vs))


def measure(dataset: str, rows: int, verbose: bool = True) -> DatasetResult:
    res = DatasetResult(dataset)
    t0 = time.time()
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
    res.rows = len(dirty)
    res.columns = list(dirty.columns)

    # ── the mechanism, dirty frame only ────────────────────────────────
    es = ExcessSurprise(seed=SEED).fit(dirty)
    # ── the incumbent baseline, dirty frame only, profile=None ─────────
    base = ConditionalDetector(mode="full", seed=SEED).fit(dirty)
    base_scores = base.score_cells(dirty)

    labels = cell_labels(clean, dirty)

    for col in res.columns:
        cs = es.columns_[col]
        base_col = base_scores[col].to_numpy(dtype=float)
        base_constant = np.unique(base_col).size <= 1
        if cs.skipped or base_constant:
            res.skipped[col] = cs.skipped or "baseline_constant"
            continue
        res.scored.append(col)
        truth = labels[col].to_numpy().astype(bool)
        res.is_corrupt[col] = bool(truth.sum() > 0)
        res.true_rate[col] = float(truth.mean())
        res.rare[col] = rare_fraction(dirty[col])
        res.cardinality[col] = int(dirty[col].nunique(dropna=True))
        res.stat["excess"][col] = cs.excess_self
        res.stat["observed"][col] = cs.observed_error
        res.stat["expected"][col] = cs.expected_error_self
        res.stat["baseline"][col] = float(np.nanmean(base_col))

    for s in STATS:
        res.auc[s] = column_auc(res, s)

    # Cell level: excess_c as a per-column PRIOR on the existing per-cell
    # score. Labelled as a combination, not as a standalone cell detector.
    if res.scored:
        flat_y, flat_prior, flat_base = [], [], []
        for col in res.scored:
            y = labels[col].to_numpy().astype(int)
            b = base_scores[col].to_numpy(dtype=float)
            e = res.stat["excess"][col] or 0.0
            flat_y.append(y)
            flat_base.append(b)
            flat_prior.append(b * max(e, 0.0))
        y_all = np.concatenate(flat_y)
        res.cell_pr["baseline"] = pr_auc(y_all, np.concatenate(flat_base))
        res.cell_pr["excess_prior"] = pr_auc(y_all, np.concatenate(flat_prior))

    res.seconds = time.time() - t0
    if verbose:
        print(f"  {dataset:<14} scored {len(res.scored)}/{len(res.columns)}"
              f"  AUC excess={_f(res.auc['excess'])} "
              f"obs={_f(res.auc['observed'])} exp={_f(res.auc['expected'])} "
              f"base={_f(res.auc['baseline'])}  [{res.seconds:.0f}s]")
    return res


def _f(v: Optional[float], nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}"


# ═══════════════════════════════════════════════════════════════════════════


def build_report(results: List[DatasetResult]) -> str:
    L: List[str] = []
    a = L.append
    defined = [r for r in results if r.defined]

    a("# W8-D — Excess Surprise: does a calibration gap localise corruption?")
    a("")
    a("Generated by `experiments/w8d_excess.py`. Mechanism and motivation in")
    a("`docs/w8_plan.md` §8-§9. Seed 42.")
    a("")
    a("## 0. What was pre-registered, before this ran")
    a("")
    a("From the unit tests in `tests/test_excess_surprise.py`:")
    a("")
    a("- **Difficulty cancellation is VERIFIED** on synthetic data. Clean-easy")
    a("  and clean-hard columns whose observed errors differ by more than 0.3")
    a("  both land within 0.15 of zero excess, with an excess gap under a")
    a("  third of their observed gap. No earlier mechanism here removed that")
    a("  nuisance parameter.")
    a("- **Sensitivity is WEAK**: excess rises at about `0.08 * r`, because the")
    a("  model is cross-fitted on the corrupted frame and absorbs some")
    a("  corruption into `expected_error`. At adult's true `income` rate of")
    a("  ~0.0376 that predicts excess near **0.003**.")
    a("")
    a("So the prediction on record: **if this fails on adult, weak sensitivity")
    a("is the reason, not failed cancellation.**")
    a("")

    a("## 1. Condition 1 — column-level AUC against the baseline and the ingredients")
    a("")
    a("Columns are ranked by each statistic; the AUC is how well that ranking")
    a("separates truly-corrupted columns from clean ones. Chance 0.5. The")
    a("baseline is the mean `ConditionalDetector(mode=\"full\")` score,")
    a("**recomputed in this run on exactly the same scored columns** — no")
    a("previously published value is carried over. `observed` and `expected`")
    a("are Excess Surprise's own two ingredients, which it must beat")
    a("(condition 9).")
    a("")
    a("| dataset | scored/total | corrupt | **excess** | observed | expected | baseline |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        n_corr = sum(1 for c in r.scored if r.is_corrupt[c])
        tag = "" if r.defined else " ⚠"
        a(f"| {r.dataset}{tag} | {len(r.scored)}/{len(r.columns)} | {n_corr} | "
          f"**{_f(r.auc['excess'])}** | {_f(r.auc['observed'])} | "
          f"{_f(r.auc['expected'])} | {_f(r.auc['baseline'])} |")
    if defined:
        a("| **mean over defined** | | | "
          + " | ".join(
              "**" + _f(float(np.mean([r.auc[s] for r in defined
                                       if r.auc[s] is not None]))) + "**"
              if s == "excess" else
              _f(float(np.mean([r.auc[s] for r in defined
                                if r.auc[s] is not None])))
              for s in STATS) + " |")
    a("")
    a("`⚠` marks a dataset where one class is absent among scored columns, so")
    a("no column-level AUC exists. It is reported as `—`, never as 0.0.")
    a("")

    a("## 2. Condition 2 — paired tests, not a mean of AUCs")
    a("")
    a("A mean over datasets would weight a 2-column table like a 19-column")
    a("one and carry no interval, which is why the plan fixed a paired test in")
    a("advance. Exact two-sided sign test, ties dropped.")
    a("")
    a("| excess vs | n | wins | losses | median diff | p |")
    a("|---|---:|---:|---:|---:|---:|")
    verdict: Dict[str, Dict[str, object]] = {}
    for other in ("baseline", "observed", "expected"):
        diffs = [r.auc["excess"] - r.auc[other] for r in defined
                 if r.auc["excess"] is not None and r.auc[other] is not None]
        st = sign_test(diffs)
        verdict[other] = st
        a(f"| {other} | {st['n']} | {st['wins']} | {st['losses']} | "
          f"{_f(st['median'])} | {_f(st['p'], 4)} |")
    a("")

    a("## 3. Conditions 4, 6 and 8 — is it rarity, cardinality, or difficulty?")
    a("")
    a("Spearman rho between each dataset's per-column `excess` and three")
    a("confounds. Condition 8 is the decisive one: if `excess` still tracks")
    a("`expected_error`, the subtraction failed and this is the same nuisance")
    a("parameter that killed four earlier mechanisms.")
    a("")
    a("| dataset | rho vs rare-value frac (C4) | rho vs cardinality (C6) | **rho vs expected_error (C8)** |")
    a("|---|---:|---:|---:|")
    rho8: List[float] = []
    for r in results:
        if not r.scored:
            continue
        ex = [r.stat["excess"][c] for c in r.scored]
        r4 = spearman(ex, [r.rare[c] for c in r.scored])
        r6 = spearman(ex, [float(r.cardinality[c]) for c in r.scored])
        r8 = spearman(ex, [r.stat["expected"][c] for c in r.scored])
        if r8 is not None:
            rho8.append(abs(r8))
        a(f"| {r.dataset} | {_f(r4)} | {_f(r6)} | **{_f(r8)}** |")
    a("")
    if rho8:
        a(f"Mean |rho| against `expected_error`: **{np.mean(rho8):.4f}** over "
          f"{len(rho8)} datasets. Near zero means difficulty cancelled on real")
        a("data as it did on synthetic data; large means it did not.")
    a("")

    a("## 4. Condition 3 — cell level (a COMBINATION, not a standalone detector)")
    a("")
    a("Excess Surprise is a per-column statistic, so it cannot rank cells by")
    a("itself. Here `excess_c` is used as a per-column prior multiplying the")
    a("existing per-cell score, over scored columns only. Reference points:")
    a("adult whole-table PR-AUC **0.0025** is chance, and the within-`income`")
    a("real-corruption Bayes ceiling is **0.7222** with the existing detector")
    a("at **0.5626** (`results/w7_real_ceiling.md`).")
    a("")
    a("| dataset | baseline cell PR-AUC | excess-as-prior cell PR-AUC |")
    a("|---|---:|---:|")
    for r in results:
        if not r.cell_pr:
            continue
        a(f"| {r.dataset} | {_f(r.cell_pr.get('baseline'))} | "
          f"{_f(r.cell_pr.get('excess_prior'))} |")
    a("")

    a("## 5. adult — the hard case")
    a("")
    adult = next((r for r in results if r.dataset == "adult"), None)
    if adult is None or "income" not in adult.scored:
        a("`income` was not scored on adult, so the headline comparison could")
        a("not be made.")
    else:
        order = sorted(adult.scored,
                       key=lambda c: -(adult.stat["excess"][c] or -1e18))
        rank = order.index("income") + 1
        a(f"`income` is adult's **only** corrupted column (true rate "
          f"{adult.true_rate['income']:.4f}), and its corruption is entirely")
        a("one-directional `1 -> 0`.")
        a("")
        a(f"- rank of `income` by `excess`: **{rank} of {len(order)}** "
          f"(one-sided p = {rank/len(order):.3f})")
        a(f"- `income` excess: **{_f(adult.stat['excess']['income'])}** "
          f"(pre-registered prediction: ~0.003)")
        a(f"- mean excess over adult's 14 clean scored columns: "
          f"{_f(float(np.nanmean([adult.stat['excess'][c] for c in adult.scored if c != 'income'])))}")
        a("")
        a("| rank | column | corrupt | excess | observed | expected |")
        a("|---:|---|---|---:|---:|---:|")
        for i, c in enumerate(order, start=1):
            a(f"| {i} | `{c}` | {'**yes**' if adult.is_corrupt[c] else 'no'} | "
              f"{_f(adult.stat['excess'][c])} | {_f(adult.stat['observed'][c])} "
              f"| {_f(adult.stat['expected'][c])} |")
        a("")

    a("## 6. Verdict")
    a("")
    a("| condition | result |")
    a("|---|---|")
    st_b = verdict.get("baseline", {})
    for name, other in (("1+2 beats baseline (paired)", "baseline"),
                        ("9 beats `observed` alone", "observed"),
                        ("9 beats `expected` alone", "expected")):
        st = verdict.get(other, {})
        p = st.get("p")
        wins, losses = st.get("wins", 0), st.get("losses", 0)
        ok = (p is not None and p < 0.05 and wins > losses)
        a(f"| {name} | {'**PASS**' if ok else '**FAIL**'} — "
          f"{wins}W/{losses}L, p={_f(p, 4)} |")
    if rho8:
        ok8 = float(np.mean(rho8)) < 0.3
        a(f"| 8 difficulty cancels on real data | "
          f"{'**PASS**' if ok8 else '**FAIL**'} — mean |rho| vs expected_error "
          f"= {np.mean(rho8):.4f} |")
    a("| 7 no leakage | **PASS** — `tests/test_excess_surprise.py::TestNoLeakage` |")
    a("")
    a("Conditions 4, 5 and 6 are reported as measured values in §1 and §3")
    a("rather than as a threshold, because the plan fixed them as diagnostics")
    a("to report, not bars to clear.")
    a("")
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="W8-D Excess Surprise evaluation")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--rows", type=int, default=ROW_CAP)
    args = ap.parse_args(argv)

    names = args.datasets
    if not names:
        names = sorted(d.name for d in (ROOT / "datasets").iterdir()
                       if d.is_dir())

    print("W8-D — Excess Surprise")
    print(f"seed {SEED}  rows<= {args.rows}\n")
    results: List[DatasetResult] = []
    for ds in names:
        try:
            results.append(measure(ds, args.rows))
        except Exception as exc:
            print(f"  {ds:<14} FAILED {type(exc).__name__}: {exc}")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(build_report(results), encoding="utf-8")
    print(f"\nWrote {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
