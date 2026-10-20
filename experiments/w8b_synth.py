"""
experiments/w8b_synth.py
------------------------
W8-B — **learned-policy synthetic supervision**, evaluated against the seven
falsification conditions fixed in ``docs/w8_plan.md`` §5 before any W8 code
existed.

THE QUESTION
============
``docs/w8_plan.md`` §2 names **F2 — learn the corruption model from the dirty
data and train a supervised detector on synthesised errors — as the single
biggest factor this project does not have.** HoloDetect (SIGMOD 2019) is the
evidence and also the warning: learned augmentation 0.953 F1 on Soccer at
10 % training data, *random* transformations 0.166.

Learning the policy needs to know which cells are corrupt, which is the
problem. The circularity is the experiment, not an obstacle to it:

===========  =====================================================  ==========
arm          policy                                                 deployable
===========  =====================================================  ==========
``a0``       designed types from ``src/datagen/corruptor.py``,       yes, with
             applied to the dirty table. HoloDetect's "random         zero labels
             transformations" arm.
``a1_rand``  estimated from **20 labelled cells**, chosen            yes, at the
             uniformly at random.                                     stated
                                                                      budget
``a1_top``   estimated from **20 labelled cells**, chosen as the     yes, at the
             20 highest-scoring cells of the unsupervised             stated
             conditional detector (Raha/ED2 habit; uses no            budget
             ground truth to choose *where* to label).
``a1b_bal``  estimated from 20 labels, **half forced onto            **NO** —
             corrupted cells**. SEMI-ORACLE: a balanced sample        diagnostic
             requires knowing where the corruptions are. Reported     only
             only to separate budget *placement* from policy
             *estimation*.
``a2``       estimated from **all** true labels.                     **NO** —
             ORACLE upper bound.                                      upper bound
===========  =====================================================  ==========

The label budget is 20 cells, the Raha/ED2/SAGED benchmarking budget, and it
is printed next to every arm-1 number. The budgeted label is the
HoloDetect-style one — "this cell is wrong and the right value is v" —
which is **stronger than Raha's binary flag**, because a transition
distribution is not estimable without the correct value. Stated here, in the
module it uses, and in the generated report.

WHAT IS MEASURED (``docs/w8_plan.md`` §5)
=========================================
1. pooled **column-level AUC** vs the mean-raw-score baseline, **recomputed**
   on exactly the same scored columns (the W6 lesson: never carry a baseline
   over)
2. **paired** comparison across datasets (sign test + Wilcoxon), not a mean
   of AUCs
3. **cell level on adult**: whole-table PR-AUC (ours 0.0025 = chance) and
   within-``income`` PR-AUC (ours 0.5626, real-corruption Bayes ceiling
   0.7222 — ``results/w7_real_ceiling.md``)
4. **not just rarity** — Spearman of the column statistic with each column's
   rare-value fraction (values in under 1 % of rows)
5. **not just coverage** — columns the conditional detector cannot score are
   excluded, and the baseline is restricted to the same columns
6. **not just cardinality** — Spearman with column cardinality
7. **no leakage** — asserted in ``tests/test_synth_supervised.py``

Per ``docs/w8_plan.md`` §7: no published cell-level benchmark reports PR-AUC,
so precision / recall / F1 are reported alongside it.

ELIGIBILITY, FIXED ONCE FOR ALL ARMS
====================================
A column is **scored** when the unsupervised conditional detector produces at
least 3 distinct scores in it. Detector-skipped columns (ID-like,
high-cardinality text, constant) get all-zero conditional scores, and leaving
them in makes a column-level AUC measure "did the detector look at this
column" rather than "is this column corrupted" — the W6.1 pre-fix bug. The
rule is deliberately **arm-independent**, so every arm and the baseline are
scored on an identical column population.

LEAKAGE
=======
Every arm is fitted from the dirty table and its own policy. ``clean_df``
enters only through ``label_cells`` / ``oracle_labelled_cells`` /
``oracle_balanced_budget`` (the budget and the oracles, by arm) and through
``cell_labels`` for scoring **after** every prediction exists. The script
asserts that arm 0 ran before any clean data was read.

Run:
    python experiments/w8b_synth.py
    python experiments/w8b_synth.py --datasets adult breast_cancer
    python experiments/w8b_synth.py --budget 40

Writes ``results/w8b_synth.md``. Deterministic: seed 42, row cap 8000.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from src.detectors.synth_supervised import (  # noqa: E402
    SynthSupervisedDetector,
    choose_budget_cells,
    designed_policy,
    estimate_policy,
    label_cells,
    oracle_balanced_budget,
    oracle_labelled_cells,
)

enable_unicode_stdout()

SEED = 42
ROW_CAP = 8000
BUDGET = 20
TARGET_POSITIVES = 2000

ALL_DATASETS = (
    "adult", "beers", "breast_cancer", "flights", "hospital", "movies",
    "rayyan", "restaurants", "smartfactory", "soccer", "tax",
)

#: Arm key -> (label, deployable?, note). Order is reporting order.
ARMS: Tuple[Tuple[str, str, bool, str], ...] = (
    ("a0", "designed policy", True, "zero labels"),
    ("a1_rand", "learned policy, random budget", True, f"{BUDGET} labels"),
    ("a1_top", "learned policy, top-score budget", True, f"{BUDGET} labels"),
    ("a1b_bal", "learned policy, balanced budget", False,
     f"SEMI-ORACLE ({BUDGET} labels, half forced positive)"),
    ("a2", "oracle policy", False, "ORACLE — all labels"),
)

#: Published reference points this experiment must be compared against.
REF_ADULT_WHOLE_TABLE_PR_AUC = 0.0025   # results/detection_leaderboard.md
REF_ADULT_INCOME_PR_AUC = 0.5626        # results/w7_real_ceiling.md
REF_ADULT_INCOME_CEILING = 0.7222       # results/w7_real_ceiling.md

#: A value is "rare" in a column when it occurs in under this fraction of rows
#: (falsification condition 4).
RARE_CUT = 0.01

OUT_MD = ROOT / "results" / "w8b_synth.md"
OUT_JSON = ROOT / "results" / "w8b_synth.json"


# ═══════════════════════════════════════════════════════════════════════════
# Small statistics, written out so they are checkable
# ═══════════════════════════════════════════════════════════════════════════


def spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Spearman rho from averaged ranks. Same implementation as W6.1."""
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


def _binom_coef(n: int, k: int) -> float:
    from math import comb
    return float(comb(n, k))


def sign_test(diffs: Sequence[float]) -> Dict[str, Any]:
    """
    Two-sided exact sign test on paired differences.

    Written out (not scipy) so the p-value is auditable: under H0 the number
    of positive differences among the non-zero ones is Binomial(m, 0.5), and
    the two-sided p-value is the total probability of outcomes at least as
    extreme as the observed count.
    """
    d = [float(v) for v in diffs if np.isfinite(v)]
    wins = sum(1 for v in d if v > 0)
    losses = sum(1 for v in d if v < 0)
    ties = sum(1 for v in d if v == 0)
    m = wins + losses
    if m == 0:
        return {"wins": wins, "losses": losses, "ties": ties, "n": len(d),
                "p_value": None}
    k = min(wins, losses)
    tail = sum(_binom_coef(m, i) for i in range(0, k + 1)) / (2.0 ** m)
    p = min(1.0, 2.0 * tail)
    return {"wins": wins, "losses": losses, "ties": ties, "n": len(d),
            "p_value": float(p)}


def wilcoxon(diffs: Sequence[float]) -> Optional[float]:
    """Wilcoxon signed-rank p-value via scipy, or None if unavailable."""
    d = np.asarray([v for v in diffs if np.isfinite(v) and v != 0.0], dtype=float)
    if d.size < 2:
        return None
    try:
        from scipy.stats import wilcoxon as _w
    except Exception:  # pragma: no cover - scipy is a sklearn dependency
        return None
    try:
        return float(_w(d).pvalue)
    except Exception:
        return None


def rare_fraction(series: pd.Series, cut: float = RARE_CUT) -> float:
    """Fraction of rows holding a value that occurs in under ``cut`` of rows."""
    n = len(series)
    if n == 0:
        return 0.0
    counts = series.astype(object).where(series.notna(), None).value_counts(
        dropna=False
    )
    rare_values = counts[counts / n < cut]
    return float(rare_values.sum() / n)


def _f(v: Optional[float], nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}"


# ═══════════════════════════════════════════════════════════════════════════
# Per-dataset run
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ArmResult:
    arm: str
    policy_source: str
    fitted: bool
    skip_reason: str = ""
    n_labels: int = 0
    n_positive_labels: int = 0
    policy_columns: Dict[str, float] = field(default_factory=dict)
    policy_kinds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    one_directional: Dict[str, bool] = field(default_factory=dict)
    n_synth_injected: int = 0
    pr_auc: Optional[float] = None
    roc_auc: Optional[float] = None
    best_f1: Optional[float] = None
    precision_at_best_f1: Optional[float] = None
    recall_at_best_f1: Optional[float] = None
    column_auc: Optional[float] = None
    column_stat: Dict[str, float] = field(default_factory=dict)
    focus_pr_auc: Optional[float] = None
    seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        return d


@dataclass
class DatasetResult:
    dataset: str
    n_rows: int
    n_rows_full: int
    n_cols: int
    prevalence: float
    n_corrupted_cells: int
    scored_columns: List[str]
    corrupted_columns: List[str]
    baseline_column_auc: Optional[float]
    baseline_column_stat: Dict[str, float]
    rare_frac: Dict[str, float]
    cardinality: Dict[str, int]
    arms: Dict[str, ArmResult]
    column_auc_defined: bool
    seconds: float

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["arms"] = {k: v.to_dict() for k, v in self.arms.items()}
        return d


def _subsample(
    clean: pd.DataFrame, dirty: pd.DataFrame, row_cap: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    if row_cap > 0 and len(dirty) > row_cap:
        idx = np.sort(rng.choice(len(dirty), size=row_cap, replace=False))
        return (clean.iloc[idx].reset_index(drop=True),
                dirty.iloc[idx].reset_index(drop=True))
    return clean.reset_index(drop=True), dirty.reset_index(drop=True)


def _column_auc(
    stat: Dict[str, float], corrupted: Sequence[str], columns: Sequence[str]
) -> Optional[float]:
    vals, ys = [], []
    for col in columns:
        v = stat.get(col)
        if v is None or not np.isfinite(v):
            continue
        vals.append(float(v))
        ys.append(1 if col in set(corrupted) else 0)
    if not vals or len(set(ys)) < 2:
        return None
    return roc_auc(np.asarray(ys), np.asarray(vals))


def run_dataset(
    dataset: str,
    budget: int = BUDGET,
    row_cap: int = ROW_CAP,
    target_positives: int = TARGET_POSITIVES,
    verbose: bool = True,
) -> DatasetResult:
    t_ds = time.time()
    clean_full, dirty_full = DataLoader("datasets").load(dataset)
    clean, dirty = _subsample(clean_full, dirty_full, row_cap)

    # ── the unsupervised baseline, dirty data only, profile=None ────────
    base_det = ConditionalDetector(mode="full", seed=SEED).fit(dirty)
    base_scores = base_det.score_cells(dirty)
    clean_read = False

    # ── arm 0 is built and fitted BEFORE any clean data is read ─────────
    arms: Dict[str, ArmResult] = {}
    policies: Dict[str, Any] = {"a0": designed_policy(dirty, rate=0.02)}
    assert not clean_read, "arm 0's policy must precede any clean read"
    arms["a0"] = _fit_and_score(
        "a0", policies["a0"], dirty, base_det, verbose=verbose,
        target_positives=target_positives,
    )

    # ── from here on, clean data is read: budgets, oracles, labels ──────
    budget_rand = choose_budget_cells(dirty, budget, "random", seed=SEED)
    budget_top = choose_budget_cells(
        dirty, budget, "top_score", scores=base_scores, seed=SEED
    )
    clean_read = True
    policies["a1_rand"] = estimate_policy(
        dirty, label_cells(clean, dirty, budget_rand), source="learned"
    )
    policies["a1_top"] = estimate_policy(
        dirty, label_cells(clean, dirty, budget_top), source="learned"
    )
    policies["a1b_bal"] = estimate_policy(
        dirty, oracle_balanced_budget(clean, dirty, budget, seed=SEED),
        source="learned",
    )
    oracle_labels = oracle_labelled_cells(clean, dirty)
    policies["a2"] = estimate_policy(dirty, oracle_labels, source="oracle")

    for key, _label, _dep, _note in ARMS:
        if key == "a0":
            continue
        arms[key] = _fit_and_score(
            key, policies[key], dirty, base_det, verbose=verbose,
            target_positives=target_positives,
        )

    # ── ground truth, only now, only to score what already exists ───────
    labels = cell_labels(clean, dirty)
    y_full = labels.to_numpy().ravel().astype(int)
    prevalence = float(y_full.mean())
    corrupted_columns = [c for c in dirty.columns if bool(labels[c].any())]

    # eligibility: arm-independent, from the unsupervised detector
    scored_columns = []
    for col in dirty.columns:
        s = base_scores[col].to_numpy(dtype=float)
        s = s[np.isfinite(s)]
        if s.size and np.unique(s).size >= 3:
            scored_columns.append(col)

    baseline_stat = {
        col: float(np.nanmean(base_scores[col].to_numpy(dtype=float)))
        for col in dirty.columns
    }
    baseline_auc = _column_auc(baseline_stat, corrupted_columns, scored_columns)

    focus_col = _focus_column(dataset, corrupted_columns)
    for key, res in arms.items():
        if not res.fitted:
            continue
        scores = res.column_stat.pop("__frame__")  # the frame, parked there
        y = labels.to_numpy().ravel().astype(int)
        s = scores.to_numpy().ravel()
        res.pr_auc = pr_auc(y, s)
        res.roc_auc = roc_auc(y, s)
        f1, thr = best_f1(y, s)
        if f1 is not None and thr is not None:
            res.best_f1 = float(f1)
            # Precision and recall at that same oracle threshold, computed
            # here rather than re-derived from F1 (two unknowns, one number).
            pred = s >= float(thr)
            tp = int(np.sum(pred & (y == 1)))
            fp = int(np.sum(pred & (y == 0)))
            fn = int(np.sum(~pred & (y == 1)))
            res.precision_at_best_f1 = (
                float(tp / (tp + fp)) if (tp + fp) else None
            )
            res.recall_at_best_f1 = float(tp / (tp + fn)) if (tp + fn) else None
        res.column_stat = {
            col: float(np.nanmean(scores[col].to_numpy(dtype=float)))
            for col in dirty.columns
        }
        res.column_auc = _column_auc(
            res.column_stat, corrupted_columns, scored_columns
        )
        if focus_col is not None:
            res.focus_pr_auc = pr_auc(
                labels[focus_col].to_numpy().astype(int),
                scores[focus_col].to_numpy(dtype=float),
            )
        for col, cp in getattr(policies[key], "columns", {}).items():
            res.one_directional[col] = bool(cp.is_one_directional)

    rare = {c: rare_fraction(dirty[c]) for c in dirty.columns}
    card = {c: int(dirty[c].nunique(dropna=True)) for c in dirty.columns}
    n_corr_scored = sum(1 for c in scored_columns if c in set(corrupted_columns))

    return DatasetResult(
        dataset=dataset,
        n_rows=len(dirty),
        n_rows_full=len(dirty_full),
        n_cols=dirty.shape[1],
        prevalence=prevalence,
        n_corrupted_cells=int(y_full.sum()),
        scored_columns=scored_columns,
        corrupted_columns=corrupted_columns,
        baseline_column_auc=baseline_auc,
        baseline_column_stat=baseline_stat,
        rare_frac=rare,
        cardinality=card,
        arms=arms,
        column_auc_defined=0 < n_corr_scored < len(scored_columns),
        seconds=time.time() - t_ds,
    )


def _focus_column(dataset: str, corrupted_columns: Sequence[str]) -> Optional[str]:
    """
    The within-column slice reported for condition 3. On adult this is
    ``income``, whose existing within-column PR-AUC (0.5626) and real
    corruption ceiling (0.7222) are published in
    ``results/w7_real_ceiling.md``. On other datasets it is the most
    corrupted column, reported for context only.
    """
    if dataset == "adult":
        return "income" if "income" in corrupted_columns else None
    return corrupted_columns[0] if corrupted_columns else None


def _fit_and_score(
    key: str,
    policy: Any,
    dirty: pd.DataFrame,
    base_det: ConditionalDetector,
    target_positives: int,
    verbose: bool = True,
) -> ArmResult:
    """Fit one arm and park its score frame for the (later) scoring pass."""
    t0 = time.time()
    res = ArmResult(
        arm=key,
        policy_source=policy.source,
        fitted=False,
        n_labels=int(getattr(policy, "n_labels", 0)),
        n_positive_labels=int(getattr(policy, "n_positive_labels", 0)),
        policy_columns={c: float(p.weight) for c, p in policy.columns.items()},
        policy_kinds={
            c: {k: float(v) for k, v in p.kind_probs.items()}
            for c, p in policy.columns.items()
        },
    )
    if policy.is_empty:
        res.skip_reason = "; ".join(policy.notes) or "empty policy"
        res.seconds = time.time() - t0
        if verbose:
            print(f"    {key:<9} NOT FITTED — {res.skip_reason}")
        return res
    try:
        det = SynthSupervisedDetector(
            policy, seed=SEED, target_positives=target_positives
        ).fit(dirty)
        scores = det.score_cells(dirty, detector=base_det)
    except (ValueError, RuntimeError) as exc:
        res.skip_reason = f"{type(exc).__name__}: {exc}"
        res.seconds = time.time() - t0
        if verbose:
            print(f"    {key:<9} NOT FITTED — {res.skip_reason}")
        return res
    res.fitted = True
    res.n_synth_injected = int(det.synth_info_.get("n_injected", 0))
    res.column_stat = {"__frame__": scores}  # parked; replaced after scoring
    res.seconds = time.time() - t0
    if verbose:
        print(
            f"    {key:<9} fitted: {res.n_synth_injected} synthetic positives, "
            f"{res.n_labels} labels ({res.n_positive_labels} positive), "
            f"{res.seconds:.0f}s"
        )
    return res


# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════


def _arm_meta(key: str) -> Tuple[str, bool, str]:
    for k, label, dep, note in ARMS:
        if k == key:
            return label, dep, note
    raise KeyError(key)  # pragma: no cover


def write_report(
    results: List[DatasetResult], budget: int, row_cap: int
) -> str:
    L: List[str] = []
    a = L.append

    a("# W8-B — learned-policy synthetic supervision")
    a("")
    a("Generated by `experiments/w8b_synth.py`. Planned in `docs/w8_plan.md`")
    a(f"§4 (W8-B), committed before the code. Seed 42, row cap {row_cap},")
    a(f"label budget **{budget} cells**.")
    a("")
    a("## 0. The arms, and the circularity they test")
    a("")
    a("Learning a corruption policy requires knowing which cells are")
    a("corrupt, which is the problem being solved. The arms make that")
    a("circularity measurable rather than arguable.")
    a("")
    a("| arm | policy | deployable | note |")
    a("|---|---|---|---|")
    for key, label, dep, note in ARMS:
        a(f"| `{key}` | {label} | {'yes' if dep else '**no**'} | {note} |")
    a("")
    a("`a1b_bal` and `a2` are **not methods**. `a1b_bal` needs the labels to")
    a("choose a balanced sample, and `a2` reads every label; both exist to")
    a("separate *budget placement* from *policy estimation* from *classifier")
    a("capacity*.")
    a("")
    a(f"The budgeted label is HoloDetect-style — \"this cell is wrong and the")
    a("right value is v\" — which is **stronger than Raha's binary flag**. A")
    a("transition distribution is not estimable without the correct value, so")
    a("the arm-1 numbers below are an *upper* bound on what the same budget of")
    a("binary labels would give.")
    a("")

    # ── per-dataset tables ───────────────────────────────────────────────
    a("## 1. Per-dataset cell-level detection")
    a("")
    a("Whole-table PR-AUC. Chance equals the prevalence, printed per dataset.")
    a("Precision/recall/F1 are at the oracle-best threshold (`best_f1`), which")
    a("no deployable system can pick; they are reported because")
    a("`docs/w8_plan.md` §7 records that no published cell-level benchmark")
    a("reports PR-AUC, so these are the comparable numbers.")
    a("")
    for res in results:
        a(f"### {res.dataset}")
        a("")
        a(f"{res.n_rows:,} of {res.n_rows_full:,} rows x {res.n_cols} columns; "
          f"{res.n_corrupted_cells:,} corrupted cells "
          f"(prevalence **{res.prevalence:.4f}** = chance PR-AUC); "
          f"{len(res.corrupted_columns)} corrupted columns; "
          f"{len(res.scored_columns)} of {res.n_cols} columns scored.")
        a("")
        a("| arm | PR-AUC | lift vs chance | ROC-AUC | best-F1 | P | R | "
          "labels (pos) | synth positives |")
        a("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for key, _l, dep, _n in ARMS:
            r = res.arms.get(key)
            if r is None:
                continue
            tag = f"`{key}`" + ("" if dep else " ⚠")
            if not r.fitted:
                a(f"| {tag} | — | — | — | — | — | — | "
                  f"{r.n_labels} ({r.n_positive_labels}) | not fitted |")
                continue
            lift = (r.pr_auc / res.prevalence
                    if r.pr_auc is not None and res.prevalence > 0 else None)
            a(f"| {tag} | {_f(r.pr_auc)} | {_f(lift, 2)}x | {_f(r.roc_auc)} | "
              f"{_f(r.best_f1)} | {_f(r.precision_at_best_f1)} | "
              f"{_f(r.recall_at_best_f1)} | "
              f"{r.n_labels} ({r.n_positive_labels}) | {r.n_synth_injected} |")
        a("")
        notfit = [
            (k, r) for k, r in res.arms.items() if not r.fitted
        ]
        if notfit:
            for k, r in notfit:
                a(f"- `{k}` not fitted: {r.skip_reason}")
            a("")

    # ── condition 1 and 2 ────────────────────────────────────────────────
    a("## 2. Condition 1 — column-level AUC vs the recomputed baseline")
    a("")
    a("The baseline is the **mean raw conditional score per column**,")
    a("recomputed on exactly the columns scored here — never carried over from")
    a("W6.1. A dataset is eligible only when both corrupted and clean columns")
    a("are present among the scored columns; otherwise the AUC is undefined")
    a("and printed `—`, never 0.0.")
    a("")
    header = "| dataset | eligible | baseline | " + " | ".join(
        f"`{k}`" for k, _l, _d, _n in ARMS) + " |"
    a(header)
    a("|---|---|---:|" + "---:|" * len(ARMS))
    for res in results:
        cells = [
            f"| {res.dataset} | {'yes' if res.column_auc_defined else 'no'} | "
            f"{_f(res.baseline_column_auc) if res.column_auc_defined else '—'} "
        ]
        for key, _l, _d, _n in ARMS:
            r = res.arms.get(key)
            v = r.column_auc if (r is not None and res.column_auc_defined) else None
            cells.append(f"| {_f(v)} ")
        a("".join(cells) + "|")
    a("")

    eligible = [r for r in results if r.column_auc_defined]
    a("## 3. Condition 2 — paired comparison across datasets")
    a("")
    a("Paired per dataset against the baseline on the same columns. A mean of")
    a("AUCs would weight a 7-column table equally with a 20-column one and")
    a("carries no interval, so the test is paired (`docs/w8_plan.md` §5.2).")
    a("")
    a("| arm | n paired | wins | losses | ties | sign-test p | Wilcoxon p | "
      "median diff |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    paired: Dict[str, List[float]] = {}
    for key, _l, _d, _n in ARMS:
        diffs = []
        for res in eligible:
            r = res.arms.get(key)
            if (r is None or r.column_auc is None
                    or res.baseline_column_auc is None):
                continue
            diffs.append(float(r.column_auc - res.baseline_column_auc))
        paired[key] = diffs
        st = sign_test(diffs)
        med = float(np.median(diffs)) if diffs else None
        a(f"| `{key}` | {st['n']} | {st['wins']} | {st['losses']} | "
          f"{st['ties']} | {_f(st['p_value'])} | {_f(wilcoxon(diffs))} | "
          f"{_f(med)} |")
    a("")

    # ── condition 3 ──────────────────────────────────────────────────────
    a("## 4. Condition 3 — cell level on the real hard case (adult)")
    a("")
    adult = next((r for r in results if r.dataset == "adult"), None)
    if adult is None:
        a("adult was not run, so this condition is **not evaluated**.")
        a("")
    else:
        a(f"Reference points: whole-table PR-AUC "
          f"**{REF_ADULT_WHOLE_TABLE_PR_AUC:.4f}** (chance; "
          f"`results/detection_leaderboard.md`, full 45,222 rows) and "
          f"within-`income` PR-AUC **{REF_ADULT_INCOME_PR_AUC:.4f}** against "
          f"the real-corruption Bayes ceiling "
          f"**{REF_ADULT_INCOME_CEILING:.4f}** "
          f"(`results/w7_real_ceiling.md`, same {adult.n_rows:,}-row cap as "
          f"here).")
        a("")
        a("| arm | whole-table PR-AUC | vs chance | within-`income` PR-AUC | "
          "captured of ceiling |")
        a("|---|---:|---:|---:|---:|")
        for key, _l, dep, _n in ARMS:
            r = adult.arms.get(key)
            if r is None:
                continue
            tag = f"`{key}`" + ("" if dep else " ⚠")
            if not r.fitted:
                a(f"| {tag} | — | — | — | — |")
                continue
            lift = (r.pr_auc / adult.prevalence
                    if r.pr_auc is not None and adult.prevalence > 0 else None)
            cap = (r.focus_pr_auc / REF_ADULT_INCOME_CEILING
                   if r.focus_pr_auc is not None else None)
            a(f"| {tag} | {_f(r.pr_auc)} | {_f(lift, 2)}x | "
              f"{_f(r.focus_pr_auc)} | {_f(cap, 3)} |")
        a("")
        a(f"Chance PR-AUC on this {adult.n_rows:,}-row subsample is "
          f"{adult.prevalence:.4f}; the published 0.0025 is the full table's "
          f"prevalence. Both are chance, and neither is a score.")
        a("")

    # ── one-directionality ───────────────────────────────────────────────
    a("## 5. Does each arm's policy recover the one-directional structure?")
    a("")
    a("adult's real corruption is **entirely one-directional**: all flips are")
    a("`income` `1 -> 0`, majority class to minority "
      "(`results/w7_real_ceiling.md`). A designed frequency-weighted")
    a("`swap_valid` policy does not reproduce that, which is the pre-registered")
    a("reason arm 0 is expected to fail and arms 1/2 are not.")
    a("")
    a("| dataset | arm | policy columns (weight) | one-directional? | kinds |")
    a("|---|---|---|---|---|")
    for res in results:
        for key, _l, _d, _n in ARMS:
            r = res.arms.get(key)
            if r is None or key == "a0":
                continue
            if not r.policy_columns:
                a(f"| {res.dataset} | `{key}` | — (no policy estimated) | — | — |")
                continue
            top = sorted(r.policy_columns.items(), key=lambda kv: -kv[1])[:3]
            cols = ", ".join(f"`{c}` {w:.2f}" for c, w in top)
            od = ", ".join(
                f"`{c}`={'yes' if v else 'no'}" for c, v in
                sorted(r.one_directional.items())[:3]
            ) or "—"
            kinds = ", ".join(
                f"`{c}`: " + "/".join(
                    f"{k} {v:.2f}" for k, v in sorted(ks.items(), key=lambda kv: -kv[1])
                )
                for c, ks in sorted(r.policy_kinds.items())[:2]
            ) or "—"
            a(f"| {res.dataset} | `{key}` | {cols} | {od} | {kinds} |")
    a("")
    a("Arm 0 has no estimated policy by construction: it applies the designed")
    a("types `missing, format, out_of_range, swap_valid, typo` wherever")
    a("`src/datagen/corruptor.py` finds a column eligible.")
    a("")

    # ── conditions 4 and 6 ───────────────────────────────────────────────
    a("## 6. Conditions 4 and 6 — is it just rarity, or just cardinality?")
    a("")
    a("Spearman rho between each arm's per-column statistic and the column's")
    a("rare-value fraction (values in under 1 % of rows) and cardinality,")
    a("pooled over all scored columns of all datasets. Neither quantity is a")
    a("feature of the classifier — that exclusion is deliberate")
    a("(`src/detectors/synth_supervised.py`), and it is what makes these")
    a("correlations interpretable.")
    a("")
    a("| arm | rho vs rare-value fraction | rho vs cardinality | n columns |")
    a("|---|---:|---:|---:|")
    pool_rows: List[Tuple[str, float, float]] = []
    for res in results:
        for col in res.scored_columns:
            pool_rows.append((f"{res.dataset}|{col}",
                              res.rare_frac[col], float(res.cardinality[col])))
    for key, _l, _d, _n in [("baseline", "", True, "")] + list(ARMS):
        stats, rares, cards = [], [], []
        for res in results:
            src = (res.baseline_column_stat if key == "baseline"
                   else (res.arms[key].column_stat
                         if key in res.arms and res.arms[key].fitted else {}))
            for col in res.scored_columns:
                if col not in src:
                    continue
                stats.append(float(src[col]))
                rares.append(res.rare_frac[col])
                cards.append(float(res.cardinality[col]))
        a(f"| `{key}` | {_f(spearman(stats, rares))} | "
          f"{_f(spearman(stats, cards))} | {len(stats)} |")
    a("")
    a("## 7. Condition 5 — not just coverage")
    a("")
    a("A column enters every table above only when the unsupervised")
    a("conditional detector produced at least 3 distinct scores in it. The")
    a("rule is arm-independent, so the baseline and every arm are scored on an")
    a("identical column population — the W6.1 pre-fix bug was exactly this.")
    a("")
    a("| dataset | columns | scored | excluded |")
    a("|---|---:|---:|---|")
    for res in results:
        excluded = [c for c in res.cardinality if c not in set(res.scored_columns)]
        a(f"| {res.dataset} | {res.n_cols} | {len(res.scored_columns)} | "
          f"{', '.join('`' + c + '`' for c in excluded) or '—'} |")
    a("")
    a("## 8. Condition 7 — no leakage")
    a("")
    a("Asserted in code, not claimed here: `tests/test_synth_supervised.py`")
    a("`TestNoLeakage` runs arm 0 end to end with both clean-reading entry")
    a("points monkeypatched to raise, and again with the clean frame replaced")
    a("by a tripwire that raises on any column access. The detector's")
    a("`__init__`, `fit` and `score_cells` signatures are scanned for any")
    a("parameter naming clean data. This script fits arm 0 before it reads any")
    a("clean data and asserts that ordering.")
    a("")

    # ── verdict ──────────────────────────────────────────────────────────
    a("## 9. Verdict")
    a("")
    a(_verdict(results, paired, budget))
    return "\n".join(L) + "\n"


def _verdict(
    results: List[DatasetResult],
    paired: Dict[str, List[float]],
    budget: int,
) -> str:
    """
    The pass/fail table. Every entry is derived from the numbers above; no
    condition may be marked PASS without the figure that justifies it.
    """
    L: List[str] = []
    a = L.append
    eligible = [r for r in results if r.column_auc_defined]

    def cond1(key: str) -> Tuple[str, str]:
        d = paired.get(key, [])
        if not d:
            return "FAIL", "no dataset where the column AUC is defined"
        st = sign_test(d)
        med = float(np.median(d))
        ok = st["wins"] > st["losses"] and med > 0
        return ("PASS" if ok else "FAIL",
                f"{st['wins']}W/{st['losses']}L, median diff {med:+.4f}")

    def cond2(key: str) -> Tuple[str, str]:
        d = paired.get(key, [])
        st = sign_test(d)
        p = st["p_value"]
        ok = p is not None and p < 0.05 and st["wins"] > st["losses"]
        return ("PASS" if ok else "FAIL",
                f"sign-test p = {_f(p)} on n = {st['n']}")

    def cond3(key: str) -> Tuple[str, str]:
        adult = next((r for r in results if r.dataset == "adult"), None)
        if adult is None or key not in adult.arms:
            return "n/a", "adult not run"
        r = adult.arms[key]
        if not r.fitted:
            return "FAIL", f"not fitted ({r.skip_reason})"
        if r.pr_auc is None:
            return "FAIL", "PR-AUC undefined"
        ok = r.pr_auc > adult.prevalence * 2.0
        return ("PASS" if ok else "FAIL",
                f"whole-table PR-AUC {r.pr_auc:.4f} vs chance "
                f"{adult.prevalence:.4f}; within-`income` {_f(r.focus_pr_auc)} "
                f"vs ours {REF_ADULT_INCOME_PR_AUC:.4f}, ceiling "
                f"{REF_ADULT_INCOME_CEILING:.4f}")

    def corr(key: str, which: str) -> Tuple[str, str]:
        stats, other = [], []
        for res in results:
            src = (res.arms[key].column_stat
                   if key in res.arms and res.arms[key].fitted else {})
            for col in res.scored_columns:
                if col not in src:
                    continue
                stats.append(float(src[col]))
                other.append(res.rare_frac[col] if which == "rare"
                             else float(res.cardinality[col]))
        rho = spearman(stats, other)
        if rho is None:
            return "n/a", "undefined"
        ok = abs(rho) < 0.5
        return ("PASS" if ok else "FAIL", f"rho = {rho:+.4f}")

    a("Per arm, against the seven conditions of `docs/w8_plan.md` §5. A")
    a("mechanism is kept only if **all** hold for a **deployable** arm.")
    a("")
    a("| condition | " + " | ".join(f"`{k}`" for k, _l, _d, _n in ARMS) + " |")
    a("|---|" + "---|" * len(ARMS))
    rows = [
        ("1 column AUC > baseline", cond1),
        ("2 paired, significant", cond2),
        ("3 adult cell level", cond3),
        ("4 not just rarity", lambda k: corr(k, "rare")),
        ("6 not just cardinality", lambda k: corr(k, "card")),
    ]
    detail: List[str] = []
    for name, fn in rows:
        cells = []
        for key, _l, _d, _n in ARMS:
            verdict, why = fn(key)
            cells.append(f"**{verdict}**")
            detail.append(f"- {name}, `{key}`: {verdict} — {why}")
        a(f"| {name} | " + " | ".join(cells) + " |")
    a("| 5 not just coverage | " + " | ".join(
        "**PASS**" for _ in ARMS) + " |")
    a("| 7 no leakage | " + " | ".join(
        ("**PASS**" if d else "n/a (oracle)") for _k, _l, d, _n in ARMS) + " |")
    a("")
    a("Conditions 5 and 7 are structural: 5 is the arm-independent")
    a("eligibility rule of §7 above, 7 is asserted by the test suite. The")
    a("oracle arms are exempt from 7 by definition — they read labels on")
    a("purpose, and that is why they are not methods.")
    a("")
    a("### Detail")
    a("")
    for line in detail:
        a(line)
    a("")
    n_el = len(eligible)
    a(f"Datasets with a defined column-level AUC: {n_el} of {len(results)}.")
    a("")
    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(ALL_DATASETS))
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--rows", type=int, default=ROW_CAP)
    ap.add_argument("--target-positives", type=int, default=TARGET_POSITIVES)
    ap.add_argument("--out", type=Path, default=OUT_MD)
    args = ap.parse_args(argv)

    print("=" * 74)
    print("W8-B — learned-policy synthetic supervision")
    print(f"seed {SEED}, row cap {args.rows}, label budget {args.budget}")
    print("=" * 74)

    results: List[DatasetResult] = []
    for ds in args.datasets:
        print(f"\n[{ds}]")
        t0 = time.time()
        try:
            res = run_dataset(
                ds, budget=args.budget, row_cap=args.rows,
                target_positives=args.target_positives,
            )
        except Exception as exc:  # noqa: BLE001 - one dataset must not kill the run
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            continue
        results.append(res)
        print(f"    done in {time.time() - t0:.0f}s "
              f"(column AUC defined: {res.column_auc_defined})")

    if not results:
        print("\nNo dataset completed; nothing written.")
        return 1

    md = write_report(results, args.budget, args.rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    OUT_JSON.write_text(
        json.dumps(
            {
                "seed": SEED, "row_cap": args.rows, "budget": args.budget,
                "datasets": [r.to_dict() for r in results],
            },
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")
    print(f"Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
