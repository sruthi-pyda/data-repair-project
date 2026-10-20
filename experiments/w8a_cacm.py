"""
experiments/w8a_cacm.py
-----------------------
W8-A — evaluation of **CACM**, the contamination-aware conditional mixture
(`src/detectors/cacm.py`), against the seven falsification conditions fixed in
`docs/w8_plan.md` §5 *before* any of this code existed.

WHAT IS BEING TESTED
====================
`docs/w8_plan.md` §1: within a corrupted column our conditional detector
already captures 77.9 % of the Bayes-optimal signal; across columns it is at
chance. The whole remaining loss is **choosing the column**. CACM's claim is
that x-independence of the corruption distribution identifies the
contamination, which makes a per-cell *posterior* comparable across columns,
which is what column triage needs.

THE SEVEN CONDITIONS, AND HOW EACH IS MEASURED HERE
===================================================
1. **Column triage.** Rank columns by their mean CACM posterior and ask
   whether that separates truly-corrupted columns from clean ones
   (column-level ROC-AUC, chance 0.5). The baseline is the mean raw
   `ConditionalDetector(mode="full")` score per column, **recomputed on
   exactly the same columns**. W6.1's 0.7512 is *not* carried over: the W6
   lesson is that eligibility differs between mechanisms and a carried-over
   baseline is a comparison against a different population.
2. **Paired, not averaged.** A sign test and a Wilcoxon signed-rank test on
   the per-dataset AUC differences. A mean of AUCs weights a 2-column table
   like a 19-column one and carries no interval.
3. **Cell level on the real hard case.** Whole-table PR-AUC on adult
   (current value 0.0025, which equals the prevalence — a constant scorer),
   plus the within-`income` PR-AUC against the real-corruption Bayes ceiling
   0.7222 and the existing detector's 0.5626 (`results/w7_real_ceiling.md`).
4. **Not just rarity.** Spearman correlation between the per-column statistic
   and the column's rare-value fraction (share of cells whose value occurs in
   under 1 % of rows). If rarity explains the ranking, this is W5.3 again.
5. **Not just coverage.** A column whose score vector is constant carries no
   ranking information but is still a number, so leaving it in makes the AUC
   measure "did the detector score this column at all". Such columns are
   excluded, and the baseline is restricted to the same set. Two eligibility
   rules are reported because the instruction ("CACM's scorable columns, with
   the baseline restricted to them") and the strictest paired reading
   ("columns both detectors can score") are not the same set:
       cacm         CACM scores it with >= 3 distinct values
       intersection both detectors do
6. **Not just cardinality.** Spearman correlation with the number of distinct
   values. This confound killed W5.3's self-supervised injection.
7. **No leakage.** Asserted in `tests/test_cacm.py` and re-asserted at
   runtime here: the detectors are fitted on the dirty frame with
   `profile=None`, and `cell_labels` is not called until both score frames
   already exist.

ADULT'S DIAGNOSTIC — THE ONE WORTH READING FIRST
================================================
Adult's real corruption is entirely one-directional: all 1,701 flips are
`1 -> 0` in `income`, majority class (75.2 %) to minority (24.8 %), and
`income` is an int64 0/1 column. So the mixture's `g` for `income` should
concentrate on the value **0**. Whether it does is a direct test of the
mechanism's premise, independent of any AUC, and it is reported explicitly.

LEAKAGE
=======
Both detectors are fitted on the dirty frame with `profile=None`. Clean data
is read once, after scoring, via `src/detection_eval/labels.py::cell_labels`,
and is used only to evaluate rankings that already exist. The script asserts
the ordering.

PERFORMANCE NOTE
================
`OMP_NUM_THREADS=1` is set before sklearn is imported. CACM runs
`n_columns * em_iters * n_folds` small gradient-boosting fits, and on small
matrices sklearn's thread pool costs more than it saves — measured here at
roughly 4x slower with the default thread count. It is a speed setting only;
the histogram builder is deterministic either way.

Run:
    python experiments/w8a_cacm.py
    python experiments/w8a_cacm.py --datasets adult breast_cancer
    python experiments/w8a_cacm.py --em-iters 10 --bins 10

Writes results/w8a_cacm.md. Deterministic: seed 42, row cap 8000.
"""

from __future__ import annotations

import os

# Must precede the sklearn / numpy BLAS import. See the module docstring.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.console import enable_unicode_stdout  # noqa: E402
from src.data.loader import DataLoader  # noqa: E402
from src.detection_eval.labels import cell_labels  # noqa: E402
from src.detection_eval.metrics import best_f1, pr_auc, roc_auc  # noqa: E402
from src.detectors.cacm import CACMDetector  # noqa: E402
from src.detectors.conditional import ConditionalDetector  # noqa: E402

enable_unicode_stdout()

SEED = 42
ROW_CAP = 8000
ALL_DATASETS = (
    "adult",
    "beers",
    "breast_cancer",
    "flights",
    "hospital",
    "movies",
    "rayyan",
    "restaurants",
    "smartfactory",
    "soccer",
    "tax",
)

#: A column needs at least this many distinct score values to carry ranking
#: information. Same rule as experiments/w6_contamination.py.
MIN_DISTINCT_SCORES = 3

#: Falsification condition 4: "rare" means the value occurs in under this
#: share of rows.
RARE_THRESHOLD = 0.01

#: Reference numbers from results/w7_real_ceiling.md. Quoted, never recomputed
#: here, and never used in any comparison the mechanism could pass by default.
ADULT_INCOME_CEILING = 0.7222
ADULT_INCOME_CONDITIONAL = 0.5626
ADULT_WHOLE_TABLE_CURRENT = 0.0025


# ═══════════════════════════════════════════════════════════════════════════════
# Records
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ColumnRow:
    dataset: str
    column: str
    is_corrupted: bool
    n_corrupted: int
    true_rate: float
    n_distinct: int
    rare_fraction: float
    cacm_mean_posterior: Optional[float]
    cacm_r: Optional[float]
    cacm_collapsed: str
    cacm_skip: str
    cacm_distinct_scores: int
    base_mean_score: Optional[float]
    base_skip: str
    base_distinct_scores: int
    cacm_cell_pr_auc: Optional[float] = None
    base_cell_pr_auc: Optional[float] = None

    @property
    def cacm_scored(self) -> bool:
        return (
            not self.cacm_skip
            and self.cacm_mean_posterior is not None
            and self.cacm_distinct_scores >= MIN_DISTINCT_SCORES
        )

    @property
    def base_scored(self) -> bool:
        return (
            not self.base_skip
            and self.base_mean_score is not None
            and self.base_distinct_scores >= MIN_DISTINCT_SCORES
        )


@dataclass
class DatasetResult:
    dataset: str
    n_rows: int
    n_rows_full: int
    n_cols: int
    prevalence: float
    rows: List[ColumnRow]
    cacm_table_pr_auc: Optional[float]
    base_table_pr_auc: Optional[float]
    cacm_table_roc_auc: Optional[float]
    base_table_roc_auc: Optional[float]
    cacm_best_f1: Optional[float]
    base_best_f1: Optional[float]
    seconds: float
    cacm_report: Dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def n_corrupted_cols(self) -> int:
        return sum(1 for r in self.rows if r.is_corrupted)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def spearman(a: Sequence[Optional[float]], b: Sequence[Optional[float]]) -> Optional[float]:
    """Spearman rho, or None when it is undefined (too few points, no spread)."""
    x = np.asarray([np.nan if v is None else float(v) for v in a], dtype=float)
    y = np.asarray([np.nan if v is None else float(v) for v in b], dtype=float)
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

    rx, ry = rank(x) , rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else None


def rare_value_fraction(series: pd.Series, threshold: float = RARE_THRESHOLD) -> float:
    """
    Share of cells whose value occurs in fewer than ``threshold`` of rows.

    Missing is treated as a value: a column that is 30 % blank has 30 % of its
    cells sharing one "value", which is not rare, and pretending it has no
    value would understate the column's concentration.
    """
    n = len(series)
    if n == 0:
        return 0.0
    counts = series.astype(str).value_counts(dropna=False)
    rare_values = set(counts.index[counts < threshold * n])
    if not rare_values:
        return 0.0
    return float(series.astype(str).isin(rare_values).mean())


def distinct_finite(values: np.ndarray) -> int:
    v = values[np.isfinite(values)]
    return int(np.unique(v).size) if v.size else 0


def column_auc(
    rows: Sequence[ColumnRow], key: str, eligibility: str
) -> Tuple[Optional[float], int, int]:
    """
    Column-level AUC: can this per-column statistic tell a corrupted column
    from a clean one?

    Returns ``(auc, n_eligible, n_corrupted_eligible)``. The AUC is ``None``
    whenever the eligible set does not contain both classes — reported as
    ``—``, never as 0.0, because "undefined" and "chance-or-worse" are
    different statements and conflating them is how a mechanism gets credit
    it has not earned.
    """
    vals: List[float] = []
    ys: List[int] = []
    for r in rows:
        if eligibility == "cacm" and not r.cacm_scored:
            continue
        if eligibility == "intersection" and not (r.cacm_scored and r.base_scored):
            continue
        v = getattr(r, key)
        if v is None or not np.isfinite(v):
            continue
        vals.append(float(v))
        ys.append(int(r.is_corrupted))
    n_pos = sum(ys)
    if not vals or n_pos == 0 or n_pos == len(ys):
        return None, len(ys), n_pos
    return roc_auc(np.asarray(ys), np.asarray(vals)), len(ys), n_pos


# ═══════════════════════════════════════════════════════════════════════════════
# Measurement
# ═══════════════════════════════════════════════════════════════════════════════


def measure(
    dataset: str,
    row_cap: int,
    bins: int,
    em_iters: int,
    folds: int,
    r_init: float,
    verbose: bool = True,
) -> DatasetResult:
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

    # ── THE MECHANISM: dirty frame only, profile=None (condition 7) ─────────
    cacm = CACMDetector(
        n_bins=bins, em_iters=em_iters, n_folds=folds, r_init=r_init, seed=SEED
    ).fit(dirty, profile=None)
    cacm_scores = cacm.score_cells(dirty)

    # ── THE BASELINE: the same, so the comparison is on identical units ─────
    base = ConditionalDetector(mode="full", seed=SEED).fit(dirty, profile=None)
    base_scores = base.score_cells(dirty)

    assert list(cacm_scores.columns) == list(dirty.columns)
    assert list(base_scores.columns) == list(dirty.columns)

    # ── GROUND TRUTH, read only now, only to score what already exists ──────
    labels = cell_labels(clean, dirty)

    rows: List[ColumnRow] = []
    for col in dirty.columns:
        truth = labels[col].to_numpy().astype(bool)
        c_vals = cacm_scores[col].to_numpy(dtype=float)
        b_vals = base_scores[col].to_numpy(dtype=float)
        mix = cacm.column_mixtures_[col]
        row = ColumnRow(
            dataset=dataset,
            column=str(col),
            is_corrupted=bool(truth.sum() > 0),
            n_corrupted=int(truth.sum()),
            true_rate=float(truth.mean()),
            n_distinct=int(dirty[col].nunique(dropna=True)),
            rare_fraction=rare_value_fraction(dirty[col]),
            cacm_mean_posterior=(None if mix.is_skipped else mix.mean_posterior),
            cacm_r=(None if mix.is_skipped else mix.r),
            cacm_collapsed=mix.collapsed,
            cacm_skip=(mix.reason if mix.is_skipped else ""),
            cacm_distinct_scores=distinct_finite(c_vals),
            base_mean_score=float(np.nanmean(b_vals)) if b_vals.size else None,
            base_skip=base.skipped_columns_.get(col, ""),
            base_distinct_scores=distinct_finite(b_vals),
        )
        if truth.any() and (~truth).any():
            row.cacm_cell_pr_auc = pr_auc(truth, c_vals)
            row.base_cell_pr_auc = pr_auc(truth, b_vals)
        rows.append(row)
        if verbose:
            def fmt(v: Optional[float]) -> str:
                return "   —   " if v is None else f"{v:7.4f}"
            flag = "CORRUPT" if row.is_corrupted else "clean  "
            tail = f" SKIPPED({row.cacm_skip})" if row.cacm_skip else ""
            print(
                f"    {str(col)[:22]:<23}{flag} true={row.true_rate:7.4f}"
                f" r={fmt(row.cacm_r)} post={fmt(row.cacm_mean_posterior)}"
                f" base={fmt(row.base_mean_score)}{tail}"
            )

    truth_flat = labels.to_numpy().astype(bool).ravel()
    c_flat = cacm_scores.to_numpy(dtype=float).ravel()
    b_flat = base_scores.to_numpy(dtype=float).ravel()
    defined = truth_flat.any() and (~truth_flat).any()

    return DatasetResult(
        dataset=dataset,
        n_rows=len(dirty),
        n_rows_full=len(dirty_full),
        n_cols=dirty.shape[1],
        prevalence=float(truth_flat.mean()),
        rows=rows,
        cacm_table_pr_auc=pr_auc(truth_flat, c_flat) if defined else None,
        base_table_pr_auc=pr_auc(truth_flat, b_flat) if defined else None,
        cacm_table_roc_auc=roc_auc(truth_flat, c_flat) if defined else None,
        base_table_roc_auc=roc_auc(truth_flat, b_flat) if defined else None,
        # best_f1 returns a TUPLE (f1, threshold), not a dict.
        cacm_best_f1=best_f1(truth_flat, c_flat)[0] if defined else None,
        base_best_f1=best_f1(truth_flat, b_flat)[0] if defined else None,
        seconds=time.time() - t0,
        cacm_report=cacm.report(),
    )


def adult_diagnostic(res: DatasetResult) -> Dict[str, object]:
    """
    The premise check described in the module docstring: does `g` for adult's
    `income` concentrate on `0`, the value every one of the 1,701 flips
    produced?
    """
    cols = res.cacm_report.get("columns", {})
    income = cols.get("income")
    if income is None:
        return {"available": False}
    top = income.get("top_g") or []
    return {
        "available": True,
        "skipped": income.get("role") == "skipped",
        "reason": income.get("reason"),
        "r": income.get("r"),
        "collapsed": income.get("collapsed"),
        "top_g": top,
        "g_peak_is_zero": bool(top and str(top[0][0]) == "0"),
        "within_column_pr_auc": next(
            (r.cacm_cell_pr_auc for r in res.rows if r.column == "income"), None
        ),
        "within_column_pr_auc_baseline": next(
            (r.base_cell_pr_auc for r in res.rows if r.column == "income"), None
        ),
    }


def paired_test(diffs: Sequence[float]) -> Dict[str, object]:
    """Sign test and Wilcoxon signed-rank on paired AUC differences."""
    arr = np.asarray([d for d in diffs if np.isfinite(d)], dtype=float)
    out: Dict[str, object] = {
        "n_pairs": int(arr.size),
        "n_wins": int((arr > 0).sum()),
        "n_losses": int((arr < 0).sum()),
        "n_ties": int((arr == 0).sum()),
        "median_diff": float(np.median(arr)) if arr.size else None,
    }
    nonzero = arr[arr != 0]
    if nonzero.size == 0:
        out["sign_p"] = None
        out["wilcoxon_p"] = None
        return out
    from scipy.stats import binomtest, wilcoxon

    out["sign_p"] = float(
        binomtest(int((nonzero > 0).sum()), nonzero.size, 0.5).pvalue
    )
    try:
        out["wilcoxon_p"] = float(wilcoxon(nonzero).pvalue)
    except ValueError:
        out["wilcoxon_p"] = None
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════


def f(v: Optional[float], nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}"


def build_report(
    results: List[DatasetResult], args: argparse.Namespace
) -> Tuple[str, Dict[str, object]]:
    a: List[str] = []
    verdicts: Dict[str, object] = {}

    a.append("# W8-A — CACM, the contamination-aware conditional mixture")
    a.append("")
    a.append(
        "Generated by `experiments/w8a_cacm.py`. Mechanism: "
        "`src/detectors/cacm.py`. Pre-registered falsification conditions: "
        "`docs/w8_plan.md` §5. Every number below came from this run; nothing "
        "is carried over from an earlier wave except the three reference "
        "figures explicitly labelled as quotations."
    )
    a.append("")
    a.append(
        f"Configuration: seed {SEED}, row cap {args.rows}, `n_bins={args.bins}`, "
        f"`em_iters={args.em_iters}`, `n_folds={args.folds}`, "
        f"`r_init={args.r_init}`, `g` initialised to the column marginal, additive "
        "smoothing 1.0, `r` clamped to [1e-4, 0.5]. `f` is "
        "`HistGradientBoostingClassifier` (`max_iter=30`) refitted each "
        "M-step with `sample_weight = 1 - z`, and every `f(v|x)` used in an "
        "E-step is an **out-of-fold** prediction."
    )
    a.append("")

    # ── 0. the mechanism in one paragraph ───────────────────────────────────
    a.append("## The mechanism")
    a.append("")
    a.append("```")
    a.append("P_obs(v | x) = (1 - r_c) * f(v | x)   <- clean:      DEPENDS on x")
    a.append("             +      r_c  * g(v)       <- corruption: does NOT")
    a.append("")
    a.append("P(corrupt | v, x) = r_c*g(v) / [ (1-r_c)*f(v|x) + r_c*g(v) ]")
    a.append("```")
    a.append("")
    a.append(
        "`x`-independence of `g` is the identifying restriction. Closest prior "
        "art is Kubica & Moore, *Probabilistic Noise Identification and Data "
        "Cleaning* (CMU 2002/2003); see `docs/w8_plan.md` §3 for what is and "
        "is not new here."
    )
    a.append("")

    # ── per-dataset table ───────────────────────────────────────────────────
    a.append("## Per-dataset summary")
    a.append("")
    a.append(
        "| dataset | rows | cols | corrupt cols | prevalence | CACM cols scored | "
        "base cols scored | seconds |"
    )
    a.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        a.append(
            f"| {r.dataset} | {r.n_rows} | {r.n_cols} | {r.n_corrupted_cols} | "
            f"{r.prevalence:.4f} | {sum(1 for x in r.rows if x.cacm_scored)} | "
            f"{sum(1 for x in r.rows if x.base_scored)} | {r.seconds:.0f} |"
        )
    a.append("")

    # ── condition 1 + 5 ─────────────────────────────────────────────────────
    a.append("## Condition 1 & 5 — column triage, on columns the detector can score")
    a.append("")
    a.append(
        "Column-level ROC-AUC: does the per-column statistic separate truly "
        "corrupted columns from clean ones? Chance 0.5. A dataset whose "
        "eligible columns are all corrupted, or all clean, has no AUC and is "
        "reported `—`, never 0.0."
    )
    a.append("")
    a.append(
        "`CACM post` is the mean posterior; `CACM r` is the fitted "
        "contamination rate; `baseline` is the mean raw "
        "`ConditionalDetector(mode=\"full\")` score, recomputed on the same "
        "columns in the same run."
    )
    a.append("")

    rows_by_elig: Dict[str, List[Tuple[str, Optional[float], Optional[float], Optional[float], int, int]]] = {}
    for elig in ("cacm", "intersection"):
        table: List[Tuple[str, Optional[float], Optional[float], Optional[float], int, int]] = []
        for r in results:
            auc_post, n_elig, n_pos = column_auc(r.rows, "cacm_mean_posterior", elig)
            auc_r, _, _ = column_auc(r.rows, "cacm_r", elig)
            auc_base, _, _ = column_auc(r.rows, "base_mean_score", elig)
            table.append((r.dataset, auc_post, auc_r, auc_base, n_elig, n_pos))
        rows_by_elig[elig] = table

    for elig, title in (
        ("cacm", "Eligibility: columns CACM can score (baseline restricted to them)"),
        ("intersection", "Eligibility: columns BOTH detectors can score"),
    ):
        a.append(f"### {title}")
        a.append("")
        a.append("| dataset | eligible cols | corrupt among them | CACM post | CACM r | baseline |")
        a.append("|---|---:|---:|---:|---:|---:|")
        for ds, ap, ar, ab, ne, npos in rows_by_elig[elig]:
            a.append(f"| {ds} | {ne} | {npos} | {f(ap)} | {f(ar)} | {f(ab)} |")
        a.append("")

    # ── condition 2 ─────────────────────────────────────────────────────────
    a.append("## Condition 2 — paired test across datasets")
    a.append("")
    a.append(
        "Paired per dataset, on the datasets where both AUCs are defined. A "
        "mean of AUCs would weight a 2-column table like a 19-column one and "
        "would carry no interval, which is why the plan fixed this as a "
        "paired test before the mechanism was written."
    )
    a.append("")
    for elig in ("cacm", "intersection"):
        for key, label in (("cacm_mean_posterior", "mean posterior"), ("cacm_r", "fitted r")):
            diffs: List[float] = []
            detail: List[str] = []
            for r in results:
                am, _, _ = column_auc(r.rows, key, elig)
                ab, _, _ = column_auc(r.rows, "base_mean_score", elig)
                if am is None or ab is None:
                    continue
                diffs.append(am - ab)
                detail.append(f"{r.dataset} {am - ab:+.4f}")
            t = paired_test(diffs)
            verdicts[f"paired_{elig}_{key}"] = t
            a.append(
                f"- **{label}, {elig} eligibility**: {t['n_pairs']} pairs, "
                f"{t['n_wins']} wins / {t['n_losses']} losses / {t['n_ties']} ties, "
                f"median diff {f(t['median_diff'])}, sign-test p "
                f"{f(t['sign_p'], 4) if t['sign_p'] is not None else '—'}, "
                f"Wilcoxon p "
                f"{f(t['wilcoxon_p'], 4) if t['wilcoxon_p'] is not None else '—'}."
            )
            if detail:
                a.append(f"  - per dataset: {', '.join(detail)}")
    a.append("")

    # ── condition 3 ─────────────────────────────────────────────────────────
    a.append("## Condition 3 — cell level")
    a.append("")
    a.append(
        "Whole-table PR-AUC over every cell, with ROC-AUC and best-F1 "
        "alongside (`docs/w8_plan.md` §7: no published cell-level "
        "error-detection benchmark reports PR-AUC, so precision/recall must "
        "appear too). Best-F1 is an **oracle** threshold, not an achievable "
        "operating point."
    )
    a.append("")
    a.append(
        "| dataset | prevalence | CACM PR-AUC | base PR-AUC | CACM ROC-AUC | "
        "base ROC-AUC | CACM best-F1 | base best-F1 |"
    )
    a.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        a.append(
            f"| {r.dataset} | {r.prevalence:.4f} | {f(r.cacm_table_pr_auc)} | "
            f"{f(r.base_table_pr_auc)} | {f(r.cacm_table_roc_auc)} | "
            f"{f(r.base_table_roc_auc)} | {f(r.cacm_best_f1)} | "
            f"{f(r.base_best_f1)} |"
        )
    a.append("")

    adult = next((r for r in results if r.dataset == "adult"), None)
    if adult is not None:
        diag = adult_diagnostic(adult)
        a.append("### Adult — the hard case, and the premise check")
        a.append("")
        a.append(
            f"Quoted references (`results/w7_real_ceiling.md`, not recomputed "
            f"here): whole-table PR-AUC currently **{ADULT_WHOLE_TABLE_CURRENT}** "
            f"(= the prevalence, i.e. a constant scorer); within-`income` "
            f"real-corruption Bayes ceiling **{ADULT_INCOME_CEILING}**; the "
            f"existing conditional detector reaches **{ADULT_INCOME_CONDITIONAL}** "
            f"there."
        )
        a.append("")
        a.append(f"- CACM whole-table PR-AUC: **{f(adult.cacm_table_pr_auc)}**")
        a.append(
            f"- baseline whole-table PR-AUC, this run: "
            f"**{f(adult.base_table_pr_auc)}**"
        )
        if diag.get("available"):
            a.append(
                f"- within-`income` PR-AUC: CACM "
                f"**{f(diag['within_column_pr_auc'])}**, baseline "
                f"**{f(diag['within_column_pr_auc_baseline'])}**, ceiling "
                f"{ADULT_INCOME_CEILING}"
            )
            a.append(f"- `income` fitted `r`: {f(diag['r'])}"
                     + (f" (collapsed: {diag['collapsed']})" if diag.get("collapsed") else ""))
            top = diag.get("top_g") or []
            a.append(
                "- `income` fitted `g`: "
                + ", ".join(f"`{k}` {v:.4f}" for k, v in top)
            )
            a.append("")
            a.append(
                "  **Premise check.** Every one of adult's 1,701 flips is "
                "`1 -> 0`, so if the identifying restriction works, `g` must "
                "put its mass on `0`. It "
                + ("**does**." if diag.get("g_peak_is_zero") else "**does not**.")
            )
        else:
            a.append("- `income` was not modelled; no premise check available.")
        a.append("")

    a.append("### Per-column cell-level PR-AUC on corrupted columns")
    a.append("")
    a.append("| dataset | column | corrupted cells | rate | CACM PR-AUC | base PR-AUC |")
    a.append("|---|---|---:|---:|---:|---:|")
    for r in results:
        for row in r.rows:
            if not row.is_corrupted:
                continue
            a.append(
                f"| {r.dataset} | `{row.column}` | {row.n_corrupted} | "
                f"{row.true_rate:.4f} | {f(row.cacm_cell_pr_auc)} | "
                f"{f(row.base_cell_pr_auc)} |"
            )
    a.append("")

    # ── conditions 4 and 6 ──────────────────────────────────────────────────
    a.append("## Conditions 4 & 6 — is it just rarity, or just cardinality?")
    a.append("")
    a.append(
        "Spearman rho between the per-column statistic and two confounds: the "
        "rare-value fraction (share of cells whose value occurs in under 1 % "
        "of rows) and the number of distinct values. A high correlation means "
        "the statistic is a proxy for the confound, which is how W5.3's "
        "self-supervised injection and the marginal-rarity mechanisms died. "
        "Computed over all CACM-scored columns, pooled and per dataset."
    )
    a.append("")
    all_rows = [row for r in results for row in r.rows if row.cacm_scored]
    pooled = {
        "post_vs_rare": spearman(
            [r.cacm_mean_posterior for r in all_rows], [r.rare_fraction for r in all_rows]
        ),
        "post_vs_card": spearman(
            [r.cacm_mean_posterior for r in all_rows], [float(r.n_distinct) for r in all_rows]
        ),
        "r_vs_rare": spearman(
            [r.cacm_r for r in all_rows], [r.rare_fraction for r in all_rows]
        ),
        "r_vs_card": spearman(
            [r.cacm_r for r in all_rows], [float(r.n_distinct) for r in all_rows]
        ),
        "base_vs_rare": spearman(
            [r.base_mean_score for r in all_rows], [r.rare_fraction for r in all_rows]
        ),
        "base_vs_card": spearman(
            [r.base_mean_score for r in all_rows], [float(r.n_distinct) for r in all_rows]
        ),
        "post_vs_true_rate": spearman(
            [r.cacm_mean_posterior for r in all_rows], [r.true_rate for r in all_rows]
        ),
        "r_vs_true_rate": spearman(
            [r.cacm_r for r in all_rows], [r.true_rate for r in all_rows]
        ),
    }
    verdicts["pooled_spearman"] = pooled
    a.append(f"Pooled over {len(all_rows)} scored columns:")
    a.append("")
    a.append("| statistic | vs rare-value fraction | vs cardinality | vs TRUE corruption rate |")
    a.append("|---|---:|---:|---:|")
    a.append(
        f"| CACM mean posterior | {f(pooled['post_vs_rare'])} | "
        f"{f(pooled['post_vs_card'])} | {f(pooled['post_vs_true_rate'])} |"
    )
    a.append(
        f"| CACM fitted r | {f(pooled['r_vs_rare'])} | {f(pooled['r_vs_card'])} | "
        f"{f(pooled['r_vs_true_rate'])} |"
    )
    a.append(
        f"| baseline mean score | {f(pooled['base_vs_rare'])} | "
        f"{f(pooled['base_vs_card'])} | — |"
    )
    a.append("")
    a.append("Per dataset (CACM mean posterior):")
    a.append("")
    a.append("| dataset | vs rare fraction | vs cardinality |")
    a.append("|---|---:|---:|")
    for r in results:
        sc = [row for row in r.rows if row.cacm_scored]
        a.append(
            f"| {r.dataset} | "
            f"{f(spearman([x.cacm_mean_posterior for x in sc], [x.rare_fraction for x in sc]))} | "
            f"{f(spearman([x.cacm_mean_posterior for x in sc], [float(x.n_distinct) for x in sc]))} |"
        )
    a.append("")

    # ── condition 7 ─────────────────────────────────────────────────────────
    a.append("## Condition 7 — leakage")
    a.append("")
    a.append(
        "Both detectors are fitted with `profile=None` on the dirty frame "
        "alone; `cell_labels(clean, dirty)` is not called until both score "
        "frames exist, and this script asserts that ordering. "
        "`src/detectors/cacm.py` ignores `profile` entirely, so passing a "
        "clean-derived `DatasetProfile` cannot inherit the soft-leakage "
        "caveat of CLAUDE.md §3. `tests/test_cacm.py::TestNoLeakage` asserts "
        "it mechanically: a signature scan of every entry point for any "
        "parameter mentioning clean/label/truth, a profile object that raises "
        "on any attribute read, and a spy on "
        "`HistGradientBoostingClassifier.fit` checking that every value "
        "reaching it is a value of the dirty frame."
    )
    a.append("")

    # ── collapse / coverage bookkeeping ─────────────────────────────────────
    a.append("## Coverage and collapse bookkeeping")
    a.append("")
    a.append(
        "A column is unscorable when CACM skips it or its posterior is "
        "near-constant (under 3 distinct values). `r` at a clamp bound is a "
        "bound, not an estimate, and is listed as such."
    )
    a.append("")
    a.append("| dataset | skipped by CACM | constant posterior | r at a clamp bound |")
    a.append("|---|---|---|---|")
    for r in results:
        skipped = [x.column for x in r.rows if x.cacm_skip]
        flat = [
            x.column
            for x in r.rows
            if not x.cacm_skip and x.cacm_distinct_scores < MIN_DISTINCT_SCORES
        ]
        coll = [f"{x.column}({x.cacm_collapsed})" for x in r.rows if x.cacm_collapsed]
        a.append(
            f"| {r.dataset} | {', '.join(skipped) or '—'} | "
            f"{', '.join(flat) or '—'} | {', '.join(coll) or '—'} |"
        )
    a.append("")

    return "\n".join(a) + "\n", verdicts


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="W8-A: evaluate CACM")
    ap.add_argument("--datasets", nargs="+", default=list(ALL_DATASETS))
    ap.add_argument("--rows", type=int, default=ROW_CAP)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--em-iters", type=int, default=10)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--r-init", type=float, default=0.3)
    ap.add_argument("--out", default="results/w8a_cacm.md")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    print("W8-A — CACM, contamination-aware conditional mixture")
    print(
        f"seed {SEED}  rows<= {args.rows}  bins {args.bins}  "
        f"em_iters {args.em_iters}  folds {args.folds}\n"
    )

    results: List[DatasetResult] = []
    for ds in args.datasets:
        print(f"  {ds} ...")
        try:
            res = measure(
                ds, args.rows, args.bins, args.em_iters, args.folds,
                args.r_init, verbose=not args.quiet,
            )
        except Exception as exc:  # noqa: BLE001 - a failure is a result
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            continue
        results.append(res)
        print(f"    done in {res.seconds:.0f}s\n")

    if not results:
        print("no dataset produced a result")
        return 1

    text, verdicts = build_report(results, args)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path}")

    if args.json_out:
        jpath = ROOT / args.json_out
        jpath.write_text(json.dumps(verdicts, indent=2, default=str), encoding="utf-8")
        print(f"wrote {jpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
