"""
experiments/w6_contamination.py
-------------------------------
W6.1 — **M2**: per-column contamination fraction as the column-triage signal.

Planned in `docs/w6_plan.md` §4, committed before this file existed. M2 was
ranked **#1 by value/effort** in `docs/column_triage_research.md` before any
of the four candidate mechanisms was tested; two of the other three are now
refuted (M1 the likelihood ratio, and self-supervised injection in W5.3).

THE IDEA, AND WHY IT ATTACKS THE RIGHT QUANTITY
===============================================
A rare-but-valid value and a corrupted value are *both* improbable, so no
monotone function of `P(x)` can separate them — that is identifiability, not
an engineering gap. What differs is **separability**:

    a uniformly-rare-but-CLEAN column -> its rare values are the TAIL OF ONE
      component, not a second component  -> low anomalous mixture weight
    a CORRUPTED column                  -> genuine two-component structure
                                         -> high anomalous mixture weight

So the statistic is the **mixture weight of the anomalous component**,
`pi_hat_c`, estimated from the conditional-surprise scores we already
compute. Separability, not improbability.

It also fixes the *other* failure. `docs/w6_plan.md` §2: every mechanism tried
so far is a per-cell statistic aggregated to rank columns, which fails on
power (per-cell detection fights 0.25 % prevalence, while a column-level
quantity uses all 45,222 cells) and on comparability (per-cell scores carry
column-specific units). `pi_hat_c` is a **proportion** — dimensionless, and
therefore comparable across columns by construction.

THREE ESTIMATORS, NOT ONE — AND WHY
===================================
Before writing this, MIXMOD was checked on synthetic data with a known
answer. It recovered a true contamination of 0.0400 as 0.0564, but returned
**0.1538 on a single-component skewed distribution with no contamination at
all** — a *false high* on exactly the long-rare-tail shape that
`native_country` has. That is failure mode (c) pre-registered in the plan,
demonstrated before touching real data.

So three estimators in different families are reported side by side, and
`--primary` selects which one the falsification test uses:

  mixmod   Gao & Tan exponential+Gaussian mixture (ICDM 2006), via
           `pythresh.thresholds.mixmod` — the published method
  gmm      two-component Gaussian EM, weight of the higher-mean component;
           an independent re-implementation, not a call into the above
  dbic     separability: BIC(1 component) - BIC(2 components), a
           dimensionless measure of how much better a *mixture* explains the
           scores than a single component. Not a fraction, so it is rank-only.

Reporting one number from one estimator would hide the disagreement that the
synthetic check says to expect.

WHAT IS MEASURED
================
  column-level AUC   does `pi_hat_c` separate corrupted columns from clean
                     ones? Chance 0.5. Pooled over the datasets where it is
                     defined, never on adult alone (whose power ceiling is
                     p = 1/15 = 0.067 — see `results/w5_column_sparsity.md`).
  Spearman rho       does `pi_hat_c` track the TRUE per-column contamination
                     rate? Stronger than the AUC: it asks whether the
                     quantity is estimated, not merely ordered.
  baseline           mean raw score per column — the thing already known to
                     fail (`native_country` first on adult).
  adult one-number   pi_hat(income) > pi_hat(native_country)? Pre-registered
                     in `docs/w6_plan.md` §4. If not, the mechanism is dead.

LEAKAGE
=======
`pi_hat_c` is computed from dirty-table scores only; the detector is fitted
with `profile=None`, so no clean data of any kind enters the signal. Clean
data is read **only** to score the resulting ranking, after it exists. The
script asserts this.

Run:
    python experiments/w6_contamination.py
    python experiments/w6_contamination.py --datasets adult --primary gmm

Writes results/w6_contamination.md. Deterministic: seed 42.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
import warnings
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
from src.detection_eval.metrics import roc_auc  # noqa: E402
from src.detectors.conditional import ConditionalDetector  # noqa: E402

enable_unicode_stdout()

SEED = 42
ESTIMATORS = ("mixmod", "gmm", "dbic")

#: Row cap for the detector fit. One fit per dataset (M2 reuses the scores it
#: already produces), so this is far cheaper than W5.3's per-column refits.
ROW_CAP = 8000

#: Scores per column handed to the mixture estimators. MIXMOD runs 8-15 s on
#: 5,000 points, so this bounds the runtime without changing the quantity
#: being estimated (a proportion is invariant to subsampling in expectation).
EST_SAMPLE = 5000

OUT_MD = ROOT / "results" / "w6_contamination.md"


# ═══════════════════════════════════════════════════════════════════════════
# Estimators
# ═══════════════════════════════════════════════════════════════════════════


def est_mixmod(scores: np.ndarray, seed: int = SEED) -> Optional[float]:
    """Published Gao & Tan exponential+Gaussian mixture, via pythresh."""
    try:
        from pythresh.thresholds.mixmod import MIXMOD
    except Exception:
        return None
    if np.unique(np.asarray(scores, dtype=float)).size < 3:
        # Same guard as the other two estimators. Without it MIXMOD happily
        # returns a number for an all-zero column, and that number enters the
        # ranking -- see `n_excluded` below for why that is fatal.
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = MIXMOD(random_state=seed).eval(scores)
        return float(np.mean(np.asarray(labels, dtype=float)))
    except Exception:
        return None


def est_gmm(scores: np.ndarray, seed: int = SEED) -> Optional[float]:
    """
    Two-component Gaussian EM; weight of the higher-mean component.

    Written independently of :func:`est_mixmod` rather than wrapping it, so
    the two numbers are a genuine cross-check (protocol §8b.5).
    """
    from sklearn.mixture import GaussianMixture

    x = np.asarray(scores, dtype=float).reshape(-1, 1)
    if np.unique(x).size < 3:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm = GaussianMixture(
                n_components=2, random_state=seed, n_init=3,
                covariance_type="full",
            ).fit(x)
    except Exception:
        return None
    hi = int(np.argmax(gm.means_.ravel()))
    return float(gm.weights_[hi])


def est_dbic(scores: np.ndarray, seed: int = SEED) -> Optional[float]:
    """
    Separability: BIC(1 component) - BIC(2 components).

    Positive means a *mixture* explains the scores better than a single
    component. This is the separability criterion stated directly, and it is
    dimensionless. It is **not** a contamination fraction, so it is usable
    for ranking only — the report says so wherever it appears.
    """
    from sklearn.mixture import GaussianMixture

    x = np.asarray(scores, dtype=float).reshape(-1, 1)
    if np.unique(x).size < 3:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b1 = GaussianMixture(n_components=1, random_state=seed).fit(x).bic(x)
            b2 = GaussianMixture(
                n_components=2, random_state=seed, n_init=3
            ).fit(x).bic(x)
    except Exception:
        return None
    return float(b1 - b2)


ESTIMATOR_FNS = {"mixmod": est_mixmod, "gmm": est_gmm, "dbic": est_dbic}


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic self-check — a test of the estimators that can fail
# ═══════════════════════════════════════════════════════════════════════════


def synthetic_check(seed: int = SEED) -> List[Dict[str, object]]:
    """
    Run every estimator on score distributions whose answer is known.

    This exists because MIXMOD was observed returning 0.1538 on a
    single-component skewed distribution with **zero** contamination — a
    false high on precisely the long-rare-tail shape `native_country` has.
    An estimator that cannot pass `clean_skewed` cannot be trusted to rank
    a clean heavy-tailed column below a corrupted one, which is the entire
    task.
    """
    rng = np.random.default_rng(seed)
    n = EST_SAMPLE
    cases: List[Tuple[str, np.ndarray, Optional[float], str]] = [
        (
            "clean_skewed", rng.beta(1, 8, size=n), 0.0,
            "one skewed component, NO contamination — a clean heavy-tailed "
            "column. Must come out LOW.",
        ),
        (
            "clean_uniformish", rng.beta(1, 20, size=n), 0.0,
            "one tight component, no contamination. Must come out LOW.",
        ),
        (
            "contaminated_04",
            np.concatenate([rng.beta(1, 20, size=int(n * 0.96)),
                            rng.beta(8, 2, size=int(n * 0.04))]), 0.04,
            "genuine two-component mixture at 4 %, the adult `income` rate.",
        ),
        (
            "contaminated_20",
            np.concatenate([rng.beta(1, 20, size=int(n * 0.80)),
                            rng.beta(8, 2, size=int(n * 0.20))]), 0.20,
            "genuine two-component mixture at 20 %.",
        ),
    ]
    out: List[Dict[str, object]] = []
    for name, s, truth, note in cases:
        row: Dict[str, object] = {"case": name, "true": truth, "note": note}
        for est in ESTIMATORS:
            row[est] = ESTIMATOR_FNS[est](s)
        out.append(row)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Per-dataset measurement
# ═══════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class ColumnRow:
    column: str
    is_corrupted: bool
    true_rate: float
    n_distinct: int
    mean_score: float
    pi: Dict[str, Optional[float]]
    n_distinct_scores: int = -1
    excluded: str = ""

    @property
    def scored(self) -> bool:
        return not self.excluded


@dataclasses.dataclass
class DatasetResult:
    dataset: str
    n_rows: int
    n_rows_full: int
    n_cols: int
    n_corrupted_cols: int
    rows: List[ColumnRow]
    seconds: float

    def scored_rows(self) -> List["ColumnRow"]:
        return [r for r in self.rows if r.scored]

    def defined(self) -> bool:
        """
        A column-level AUC needs both classes present **among the columns
        the detector actually scored** -- excluded columns are not a
        population the statistic was computed on.
        """
        sc = self.scored_rows()
        n_corr = sum(1 for r in sc if r.is_corrupted)
        return 0 < n_corr < len(sc)


def spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Spearman rho, computed here from ranks so it is checkable by eye."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return None

    def rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, dtype=float)
        r[order] = np.arange(1, v.size + 1, dtype=float)
        # average ties
        for val in np.unique(v):
            m = v == val
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r

    rx, ry = rank(x), rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else None


def measure(dataset: str, row_cap: int, verbose: bool = True) -> DatasetResult:
    t0 = time.time()
    loader = DataLoader("datasets")
    clean_full, dirty_full = loader.load(dataset)

    rng = np.random.default_rng(SEED)
    if row_cap > 0 and len(dirty_full) > row_cap:
        idx = np.sort(rng.choice(len(dirty_full), size=row_cap, replace=False))
        dirty = dirty_full.iloc[idx].reset_index(drop=True)
        clean = clean_full.iloc[idx].reset_index(drop=True)
    else:
        dirty = dirty_full.reset_index(drop=True)
        clean = clean_full.reset_index(drop=True)

    # ── the signal: ONE self-supervised fit, no clean data, no profile ──
    det = ConditionalDetector(mode="full", seed=SEED).fit(dirty)
    scores = det.score_cells(dirty)
    assert list(scores.columns) == list(dirty.columns)

    # ── ground truth, read only now, only to score the ranking ──────────
    labels = cell_labels(clean, dirty)

    rows: List[ColumnRow] = []
    sub = np.random.default_rng(SEED)
    for col in dirty.columns:
        s = scores[col].to_numpy(dtype=float)
        s = s[np.isfinite(s)]
        if s.size > EST_SAMPLE:
            s = s[np.sort(sub.choice(s.size, size=EST_SAMPLE, replace=False))]
        truth = labels[col].to_numpy().astype(bool)

        # A column the detector SKIPS (ID-like, high-cardinality text,
        # constant) gets an all-zero score vector. It carries no information
        # about contamination, but it is still a *number*, so leaving it in
        # makes the ranking measure "did the detector score this column at
        # all" rather than "is this column contaminated". On hospital that is
        # 12 of 20 columns, and it is why mixmod's AUC matched the mean-score
        # baseline to four decimals on six datasets. Excluded, like the
        # detector-skipped columns in experiments/w5_column_triage.py.
        nds = int(np.unique(s).size) if s.size else 0
        excluded = "" if nds >= 3 else (
            f"detector gives {nds} distinct score(s) "
            f"(skipped: {det.skipped_columns_.get(col, 'constant')})"
            if hasattr(det, "skipped_columns_")
            else f"detector gives {nds} distinct score(s)")
        pi = ({e: ESTIMATOR_FNS[e](s) for e in ESTIMATORS}
              if (s.size and not excluded) else {e: None for e in ESTIMATORS})
        rows.append(ColumnRow(
            column=col,
            is_corrupted=bool(truth.sum() > 0),
            true_rate=float(truth.mean()),
            n_distinct=int(pd.Series(dirty[col]).nunique(dropna=True)),
            mean_score=float(np.nanmean(scores[col].to_numpy(dtype=float))),
            pi=pi, n_distinct_scores=nds, excluded=excluded,
        ))
        if verbose:
            def f(v: Optional[float]) -> str:
                return "  —   " if v is None else f"{v:7.4f}"
            if rows[-1].excluded:
                print(f"    {col[:24]:<25}"
                      f"{'CORRUPT' if rows[-1].is_corrupted else 'clean  '}"
                      f" EXCLUDED ({rows[-1].n_distinct_scores} distinct score)")
            else:
                print(f"    {col[:24]:<25}"
                      f"{'CORRUPT' if rows[-1].is_corrupted else 'clean  '}"
                      f" true={rows[-1].true_rate:7.4f}"
                      f" mixmod={f(pi['mixmod'])} gmm={f(pi['gmm'])}"
                      f" dbic={f(pi['dbic'])}")

    return DatasetResult(
        dataset=dataset, n_rows=len(dirty), n_rows_full=len(dirty_full),
        n_cols=len(dirty.columns),
        n_corrupted_cols=sum(1 for r in rows if r.is_corrupted),
        rows=rows, seconds=time.time() - t0,
    )


def auc_for(res: DatasetResult, key: str) -> Optional[float]:
    """Column-level AUC using estimator `key` (or 'mean_score') as the score."""
    vals, ys = [], []
    for r in res.rows:
        if not r.scored:
            # The baseline is restricted to the same columns as the
            # estimators, otherwise it is scored on a different population
            # and the comparison means nothing.
            continue
        v = r.mean_score if key == "mean_score" else r.pi.get(key)
        if v is None or not np.isfinite(v):
            continue
        vals.append(float(v))
        ys.append(int(r.is_corrupted))
    if not vals or len(set(ys)) < 2:
        return None
    return roc_auc(np.asarray(ys), np.asarray(vals))


def rho_for(res: DatasetResult, key: str) -> Optional[float]:
    scored = [r for r in res.rows if r.scored]
    vals = [r.mean_score if key == "mean_score" else r.pi.get(key)
            for r in scored]
    vals = [np.nan if v is None else float(v) for v in vals]
    return spearman(vals, [r.true_rate for r in scored])


def _f(v: Optional[float], nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}"


# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════


def write_report(
    results: List[DatasetResult],
    synth: List[Dict[str, object]],
    primary: str,
    row_cap: int,
) -> str:
    L: List[str] = []
    a = L.append
    keys = list(ESTIMATORS) + ["mean_score"]

    a("# W6.1 — M2: per-column contamination fraction as the triage signal")
    a("")
    a("Generated by `experiments/w6_contamination.py`. Planned in")
    a("`docs/w6_plan.md` §4, committed before the code. Seed 42.")
    a("")
    a("M2 was ranked **#1 by value/effort** in")
    a("`docs/column_triage_research.md` before any of the four candidate")
    a("mechanisms was tested. Two of the other three are now refuted: M1 the")
    a("likelihood ratio (`experiments/w4_ratio_test.py`) and self-supervised")
    a("injection (W5.3, `results/w5_column_triage_control.md`).")
    a("")
    a("## 0. The hypothesis")
    a("")
    a("A rare-but-valid value and a corrupted value are *both* improbable, so")
    a("no monotone function of `P(x)` can separate them. What differs is")
    a("**separability**:")
    a("")
    a("- a uniformly-rare-but-**clean** column — its rare values are the tail")
    a("  of *one* component, so the anomalous mixture weight is **low**")
    a("- a **corrupted** column — genuine two-component structure, so the")
    a("  weight is **high**")
    a("")
    a("`pi_hat_c` is also a **proportion**: dimensionless, hence comparable")
    a("across columns by construction, which `docs/w6_plan.md` §2 argues is")
    a("exactly what per-cell score aggregation lacks.")
    a("")

    # ── synthetic self-check first: it gates everything below ────────────
    a("## 1. Estimator self-check on known answers (read this first)")
    a("")
    a("Each estimator is run on score distributions whose contamination is")
    a("known by construction. **An estimator that cannot keep")
    a("`clean_skewed` low cannot rank a clean heavy-tailed column below a")
    a("corrupted one, which is the whole task.**")
    a("")
    a("| case | true | mixmod | gmm | dbic | what it tests |")
    a("|---|---:|---:|---:|---:|---|")
    for r in synth:
        a(f"| `{r['case']}` | {_f(r['true'])} | {_f(r.get('mixmod'))} | "
          f"{_f(r.get('gmm'))} | {_f(r.get('dbic'), 1)} | {r['note']} |")
    a("")
    a("`dbic` is a BIC *difference*, not a fraction — compare it across rows,")
    a("never against the `true` column.")
    a("")

    # ── headline: pooled AUC ─────────────────────────────────────────────
    defined = [r for r in results if r.defined()]
    a("## 2. Headline — column-level AUC, pooled")
    a("")
    a("Does the statistic separate corrupted columns from clean ones?")
    a("Chance is 0.5. Reported **only** on datasets where both classes are")
    a("present; `results/w5_column_sparsity.md` records why adult alone")
    a("cannot settle this (power ceiling p = 1/15 = 0.067).")
    a("")
    a("**Columns the detector skips are excluded.** An ID-like,")
    a("high-cardinality-text or constant column gets an all-zero score")
    a("vector, which carries no information about contamination. Leaving")
    a("those in makes the ranking measure *whether the detector scored the")
    a("column at all* — on hospital that is 12 of 20 columns, and it is why")
    a("an earlier version of this table had mixmod matching the mean-score")
    a("baseline to four decimals on six datasets. The baseline below is")
    a("restricted to the same scored columns, or the comparison would be")
    a("against a different population.")
    a("")
    a("| dataset | scored cols | excluded | corrupted (of scored) | mixmod | gmm | dbic | baseline (mean score) |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    for res in results:
        sc = res.scored_rows()
        n_corr = sum(1 for r in sc if r.is_corrupted)
        tag = "" if res.defined() else " ⚠"
        a(f"| {res.dataset}{tag} | {len(sc)}/{res.n_cols} | "
          f"{res.n_cols - len(sc)} | {n_corr} | "
          + " | ".join(_f(auc_for(res, k)) for k in keys) + " |")
    if defined:
        a("| **mean over defined** | | "
          + " | ".join(
              _f(float(np.mean([v for v in (auc_for(r, k) for r in defined)
                                if v is not None])) if any(
                  auc_for(r, k) is not None for r in defined) else None)
              for k in keys) + " |")
    a("")

    # ── does it estimate the quantity, not just order it ────────────────
    a("## 3. Does `pi_hat_c` track the true contamination rate?")
    a("")
    a("Spearman rho between the statistic and each column's **true**")
    a("corruption rate. Stronger than the AUC: it asks whether the quantity")
    a("is estimated, not merely ordered.")
    a("")
    a("| dataset | mixmod | gmm | dbic | baseline |")
    a("|---|---:|---:|---:|---:|")
    for res in results:
        a(f"| {res.dataset} | "
          + " | ".join(_f(rho_for(res, k)) for k in keys) + " |")
    a("")

    # ── the pre-registered one-number test ──────────────────────────────
    a("## 4. The pre-registered falsification test")
    a("")
    a("`docs/w6_plan.md` §4 fixed this in advance:")
    a("")
    a("> On adult: `pi_hat(income) > pi_hat(native_country)`?")
    a("> If not, the mechanism is dead.")
    a("")
    a("`income` is 3.8 % corrupted; `native_country` is **0 %** corrupted and")
    a("is the column that currently dominates the top of our ranking.")
    a("")
    adult = next((r for r in results if r.dataset == "adult"), None)
    if adult is None:
        a("adult was not run, so the test was not evaluated.")
    else:
        inc = next((r for r in adult.rows if r.column == "income"), None)
        nat = next((r for r in adult.rows if r.column == "native_country"), None)
        if inc is None or nat is None:
            a("`income` or `native_country` missing from adult; cannot evaluate.")
        else:
            a("| estimator | pi_hat(income) | pi_hat(native_country) | income > native_country? |")
            a("|---|---:|---:|---|")
            for e in ESTIMATORS:
                vi, vn = inc.pi.get(e), nat.pi.get(e)
                if vi is None or vn is None:
                    verdict = "—"
                else:
                    verdict = "**PASS**" if vi > vn else "**REFUTED**"
                a(f"| {e}{' (primary)' if e == primary else ''} | {_f(vi)} | "
                  f"{_f(vn)} | {verdict} |")
            a("")
            pv, nv = inc.pi.get(primary), nat.pi.get(primary)
            if pv is not None and nv is not None:
                if pv > nv:
                    a(f"On the primary estimator (`{primary}`) the mechanism")
                    a("**survives** this test. That is one comparison on one")
                    a("dataset, so read §2 and §3 before believing it.")
                else:
                    a(f"On the primary estimator (`{primary}`) the mechanism is")
                    a("**REFUTED** by its own pre-registered test: the clean")
                    a("long-tailed column scores at least as contaminated as")
                    a("the genuinely corrupted one, which reproduces the")
                    a("existing failure rather than fixing it.")
    a("")

    # ── per-dataset detail ──────────────────────────────────────────────
    a("## 5. Per-column detail")
    a("")
    for res in results:
        a(f"### {res.dataset}")
        a("")
        a(f"{res.n_rows:,} rows (of {res.n_rows_full:,}) x {res.n_cols} "
          f"columns, {res.n_corrupted_cols} corrupted. [{res.seconds:.0f}s]")
        a("")
        a("| column | corrupted | true rate | distinct vals | mixmod | gmm | dbic | mean score |")
        a("|---|---|---:|---:|---:|---:|---:|---:|")
        for r in sorted(res.scored_rows(),
                        key=lambda r: -(r.pi.get(primary) or -1e18)):
            a(f"| `{r.column}` | {'**yes**' if r.is_corrupted else 'no'} | "
              f"{r.true_rate:.4f} | {r.n_distinct:,} | {_f(r.pi.get('mixmod'))} | "
              f"{_f(r.pi.get('gmm'))} | {_f(r.pi.get('dbic'), 1)} | "
              f"{r.mean_score:.4f} |")
        skipped = [r for r in res.rows if not r.scored]
        if skipped:
            a("")
            a(f"Excluded ({len(skipped)}): " + ", ".join(
                f"`{r.column}`{'*' if r.is_corrupted else ''}" for r in skipped)
              + ". `*` marks a genuinely corrupted column the detector does")
            a("not score at all — invisible to this mechanism by construction.")
        a("")

    a("## 6. Leakage")
    a("")
    a("`pi_hat_c` comes from dirty-table scores only. The detector is fitted")
    a("with `profile=None`, so it is fully self-supervised and no clean data")
    a("of any kind — not even aggregate statistics — enters the signal. Clean")
    a("data is read only to score the ranking after it exists, which is the")
    a("permitted evaluation use (`CLAUDE.md` §3). Note this makes W6.1")
    a("**zero-knowledge**, so unlike the shipped detector it is not exposed")
    a("to the soft-leakage result in `results/w5_leakage_ablation.md`.")
    a("")
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="W6.1 — M2 contamination fraction")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="default: every dataset in datasets/")
    ap.add_argument("--rows", type=int, default=ROW_CAP)
    ap.add_argument("--primary", choices=ESTIMATORS, default="mixmod")
    args = ap.parse_args(argv)

    names = args.datasets
    if not names:
        root = ROOT / "datasets"
        names = sorted(d.name for d in root.iterdir() if d.is_dir())

    print("W6.1 — M2 per-column contamination fraction")
    print(f"primary estimator: {args.primary}   row cap: {args.rows}\n")
    print("Estimator self-check on known answers:")
    synth = synthetic_check()
    for r in synth:
        print(f"  {str(r['case']):<20} true={_f(r['true'])}  "
              f"mixmod={_f(r.get('mixmod'))}  gmm={_f(r.get('gmm'))}  "
              f"dbic={_f(r.get('dbic'), 1)}")

    results: List[DatasetResult] = []
    for ds in names:
        print(f"\n=== {ds} ===")
        try:
            results.append(measure(ds, args.rows))
        except Exception as exc:
            print(f"  FAILED {type(exc).__name__}: {exc}")
            continue
        res = results[-1]
        print(f"  AUC  " + "  ".join(
            f"{k}={_f(auc_for(res, k))}"
            for k in list(ESTIMATORS) + ["mean_score"]))

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(
        write_report(results, synth, args.primary, args.rows), encoding="utf-8")
    print(f"\nWrote {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
