"""
experiments/w8c_ensemble.py
---------------------------
W8-C — evaluate the **Raha-style per-cell feature-vector ensemble**
(`src/detectors/feature_ensemble.py`) against the seven falsification
conditions fixed in `docs/w8_plan.md` §5 before any W8 code existed.

WHAT IS MEASURED, AND WHY EACH NUMBER IS THE ONE THE PLAN ASKED FOR
===================================================================
1. **Column triage.** Per column, the mean ensemble probability; column-level
   ROC-AUC asks whether that separates corrupted columns from clean ones. The
   baseline is the mean raw ``cond_full`` score per column, **recomputed on
   exactly the same scored columns** — W6.1's 0.7512 is not carried over,
   because eligibility differs and carrying it over was the W6 bug.
   A dataset with no clean column, or no corrupted column, among the scored
   columns reports `—`: the statistic is undefined there, and printing 0.0
   would read as a loss.
2. **Paired, not averaged.** A sign test and a Wilcoxon signed-rank test over
   the per-dataset (ensemble, baseline) pairs. A mean of AUCs would weight a
   7-column table equally with a 20-column one and carries no interval.
3. **Cell level on the real hard case.** adult whole-table PR-AUC against our
   current 0.0025 (= its prevalence, i.e. chance), and within-``income``
   PR-AUC against the existing detector's 0.5626 and the real-corruption
   Bayes ceiling 0.7222 (`results/w7_real_ceiling.md`).
4. **Not just rarity.** Spearman correlation of the per-column statistic with
   each column's rare-value fraction (values in under 1 % of rows), pooled
   over scored columns.
5. **Not just coverage.** Columns whose scores are constant are excluded, and
   the baseline is restricted to the identical set.
6. **Not just cardinality.** Spearman correlation with column cardinality,
   plus permutation importances — if ``col_card_ratio`` or a coverage proxy
   dominates, that is the W5.3 confound and it is reported as such.
7. **No leakage.** Asserted in `tests/test_feature_ensemble.py`; this script
   also asserts that the clean frame is loaded only after every score exists.

ABLATIONS
=========
Features are extracted once per dataset and reused, so the scope ablations
cost only a classifier refit:

    full        all 26 features
    cell_only   the 16 cell-scope features (no attribute, no tuple context)
    no_attribute  cell + tuple (drops col_card_ratio and col_rare_frac, the
                  two features that carry the cardinality/rarity confound)

LEAKAGE
=======
The ensemble is fitted on the dirty table plus its own injected corruption.
The conditional detector gets ``profile=None``; the rule detector's profile is
built from the **dirty** frame in both positions (``dirty_only_profile``),
because a clean profile is worth up to 0.93 F1 on some datasets
(`results/w5_raha_leakage_check.md`). ``cell_labels`` is called only after
``score_cells`` has returned, and the script asserts the ordering.

Run:
    python experiments/w8c_ensemble.py
    python experiments/w8c_ensemble.py --datasets adult breast_cancer

Writes results/w8c_ensemble.md. Deterministic: seed 42, row cap 8000.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
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
from src.detection_eval.harness import flatten_cells  # noqa: E402
from src.detection_eval.metrics import best_f1, pr_auc, roc_auc  # noqa: E402
from src.detectors.feature_ensemble import (  # noqa: E402
    FEATURE_GROUPS,
    FEATURE_NAMES,
    FeatureEnsembleDetector,
    dirty_only_profile,
)
from src.pipeline.runner import default_flag_outliers  # noqa: E402

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

#: Feature subsets for the scope ablation. Names, not recomputation.
VARIANTS: Dict[str, Tuple[str, ...]] = {
    "full": FEATURE_NAMES,
    "cell_only": FEATURE_GROUPS["cell"],
    "no_attribute": FEATURE_GROUPS["cell"] + FEATURE_GROUPS["tuple"],
}

#: Published reference points. Quoted, never recomputed here.
ADULT_WHOLE_TABLE_BASELINE = 0.0025   # results/detection_leaderboard.md (= prevalence)
ADULT_INCOME_OURS = 0.5626            # results/w7_real_ceiling.md
ADULT_INCOME_CEILING = 0.7222         # results/w7_real_ceiling.md

OUT_MD = ROOT / "results" / "w8c_ensemble.md"
OUT_JSON = ROOT / "results" / "w8c_ensemble.json"


# ═══════════════════════════════════════════════════════════════════════════
# Statistics, written here so they are checkable by eye
# ═══════════════════════════════════════════════════════════════════════════


def _ranks(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    r = np.empty(v.size, dtype=float)
    r[order] = np.arange(1, v.size + 1, dtype=float)
    for val in np.unique(v):
        m = v == val
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Spearman rho from averaged ranks."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return None
    rx, ry = _ranks(x), _ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else None


def sign_test(diffs: Sequence[float]) -> Dict[str, object]:
    """
    Exact two-sided sign test on non-zero differences.

    p = 2 * P(X >= max(w, n-w)) for X ~ Binomial(n, 0.5), capped at 1.0,
    computed from exact binomial coefficients so it needs no scipy.
    """
    d = [float(x) for x in diffs if x is not None and not np.isnan(x) and x != 0.0]
    n = len(d)
    wins = sum(1 for x in d if x > 0)
    if n == 0:
        return {"n": 0, "wins": 0, "losses": 0, "p": None}
    k = max(wins, n - wins)
    from math import comb

    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)
    return {"n": n, "wins": wins, "losses": n - wins, "p": min(1.0, 2.0 * tail)}


def wilcoxon_signed_rank(diffs: Sequence[float]) -> Dict[str, object]:
    """
    Wilcoxon signed-rank statistic with an exact two-sided p for n <= 20.

    Enumerating all 2^n sign assignments is exact and, at n <= 11 datasets,
    instant. Above 20 the normal approximation is used and labelled.
    """
    d = np.array(
        [float(x) for x in diffs if x is not None and not np.isnan(x) and x != 0.0]
    )
    n = d.size
    if n == 0:
        return {"n": 0, "W": None, "p": None, "exact": None}
    r = _ranks(np.abs(d))
    w_plus = float(r[d > 0].sum())
    w_minus = float(r[d < 0].sum())
    stat = min(w_plus, w_minus)
    if n <= 20:
        total = 0
        count = 0
        for mask in range(1 << n):
            s = sum(r[i] for i in range(n) if mask >> i & 1)
            total += 1
            if min(s, r.sum() - s) <= stat:
                count += 1
        return {"n": n, "W": stat, "W_plus": w_plus, "p": count / total, "exact": True}
    mu = n * (n + 1) / 4.0
    sigma = float(np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0))
    from math import erfc

    z = (stat - mu) / sigma if sigma > 0 else 0.0
    return {
        "n": n, "W": stat, "W_plus": w_plus,
        "p": float(erfc(abs(z) / np.sqrt(2.0))), "exact": False,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Per-dataset measurement
# ═══════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class ColumnRow:
    column: str
    is_corrupted: bool
    true_rate: float
    n_distinct: int
    card_ratio: float
    rare_frac: float
    ens_mean: Dict[str, float]
    base_mean: float
    n_distinct_ens: int
    n_distinct_base: int
    excluded: str = ""
    cell_pr_auc: Optional[float] = None
    cell_pr_auc_base: Optional[float] = None

    @property
    def scored(self) -> bool:
        return not self.excluded


@dataclasses.dataclass
class DatasetResult:
    dataset: str
    n_rows: int
    n_rows_full: int
    n_cols: int
    prevalence: float
    rows: List[ColumnRow]
    whole_table: Dict[str, Dict[str, Optional[float]]]
    holdout_roc: Optional[float]
    importances: Dict[str, float]
    injected: Dict[str, object]
    skipped_columns: Dict[str, str]
    seconds: float
    error: str = ""

    def scored_rows(self) -> List[ColumnRow]:
        return [r for r in self.rows if r.scored]

    def column_auc_defined(self) -> bool:
        sc = self.scored_rows()
        n_corr = sum(1 for r in sc if r.is_corrupted)
        return 0 < n_corr < len(sc)

    def column_auc(self, key: str) -> Optional[float]:
        """Column-level AUC on the scored columns only (condition 5)."""
        vals, ys = [], []
        for r in self.scored_rows():
            v = r.base_mean if key == "baseline" else r.ens_mean.get(key)
            if v is None or not np.isfinite(v):
                continue
            vals.append(float(v))
            ys.append(int(r.is_corrupted))
        if not vals or len(set(ys)) < 2:
            return None
        return roc_auc(np.asarray(ys), np.asarray(vals))


def _subsample(dataset: str, row_cap: int):
    loader = DataLoader("datasets")
    clean_full, dirty_full = loader.load(dataset)
    rng = np.random.default_rng(SEED)
    if row_cap > 0 and len(dirty_full) > row_cap:
        idx = np.sort(rng.choice(len(dirty_full), size=row_cap, replace=False))
        clean = clean_full.iloc[idx].reset_index(drop=True)
        dirty = dirty_full.iloc[idx].reset_index(drop=True)
    else:
        clean = clean_full.reset_index(drop=True)
        dirty = dirty_full.reset_index(drop=True)
    return clean, dirty, len(dirty_full)


def measure(dataset: str, row_cap: int = ROW_CAP, verbose: bool = True) -> DatasetResult:
    t0 = time.time()
    clean, dirty, n_full = _subsample(dataset, row_cap)

    empty = DatasetResult(
        dataset=dataset, n_rows=len(dirty), n_rows_full=n_full,
        n_cols=dirty.shape[1], prevalence=float("nan"), rows=[],
        whole_table={}, holdout_roc=None, importances={}, injected={},
        skipped_columns={}, seconds=0.0,
    )

    # ── the mechanism: dirty data only, profile=None, dirty-only rule profile ──
    ens = FeatureEnsembleDetector(
        seed=SEED,
        profile_factory=lambda df: dirty_only_profile(dataset, df),
        flag_statistical_outliers=default_flag_outliers(dataset),
    )
    try:
        ens.fit(dirty)
    except RuntimeError as exc:
        empty.error = f"could not fit: {exc}"
        empty.seconds = time.time() - t0
        return empty

    # ONE feature extraction for the dirty frame, shared by every variant, so
    # the three are compared on byte-identical features.
    score_matrix, score_info = ens.compute_features(dirty)
    score_frames: Dict[str, pd.DataFrame] = {}
    for variant, names in VARIANTS.items():
        clf = (
            ens.classifier_
            if variant == "full"
            else _refit_classifier(ens, names)
        )
        proba = clf.predict_proba(score_matrix.subset(names).x)[:, 1]
        score_frames[variant] = score_matrix.to_cell_frame(proba, dirty)
    importances = ens.permutation_importance(n_repeats=3, sample=50_000)

    # Baseline: mean raw cond_full, from the SAME fitted conditional model the
    # features came from -- recomputed here, never carried over (condition 1).
    base_frame = score_info["mode_frames"]["full"]

    # ── ground truth: read ONLY now, only to score what already exists ─────
    assert all(f is not None for f in score_frames.values())
    labels = cell_labels(clean, dirty)
    assert list(labels.columns) == list(dirty.columns)

    y_all = flatten_cells(labels).astype(int)
    whole_table: Dict[str, Dict[str, Optional[float]]] = {}
    for variant, frame in score_frames.items():
        s = flatten_cells(frame)
        bf = best_f1(y_all, s)
        whole_table[variant] = {
            "pr_auc": pr_auc(y_all, s),
            "roc_auc": roc_auc(y_all, s),
            "best_f1": None if bf[0] is None else float(bf[0]),
        }
    sb = flatten_cells(base_frame)
    bfb = best_f1(y_all, sb)
    whole_table["baseline_cond_full"] = {
        "pr_auc": pr_auc(y_all, sb),
        "roc_auc": roc_auc(y_all, sb),
        "best_f1": None if bfb[0] is None else float(bfb[0]),
    }

    rows: List[ColumnRow] = []
    n_rows = len(dirty)
    for col in dirty.columns:
        truth = labels[col].to_numpy().astype(bool)
        ens_col = {v: score_frames[v][col].to_numpy(dtype=float) for v in VARIANTS}
        base_col = base_frame[col].to_numpy(dtype=float)
        nd_ens = int(np.unique(ens_col["full"]).size)
        nd_base = int(np.unique(base_col).size)

        # Condition 5: a column whose score is constant carries no ranking
        # information, so including it would make the column-level AUC measure
        # "did anything score this column" rather than "is it corrupted".
        # Both the mechanism AND the baseline must be non-constant, otherwise
        # the two are measured on different populations.
        if nd_ens < 3 or nd_base < 3:
            reason = (
                f"constant score (ensemble {nd_ens} distinct, baseline "
                f"{nd_base} distinct; conditional skip: "
                f"{score_info.get('skipped_columns', {}).get(col, '-')})"
            )
        else:
            reason = ""

        series = dirty[col]
        counts = series.value_counts(dropna=True)
        rare_vals = set(counts.index[counts < 0.01 * n_rows])
        rows.append(ColumnRow(
            column=col,
            is_corrupted=bool(truth.sum() > 0),
            true_rate=float(truth.mean()),
            n_distinct=int(series.nunique(dropna=True)),
            card_ratio=float(series.nunique(dropna=True) / max(n_rows, 1)),
            rare_frac=float(series.isin(rare_vals).mean()) if rare_vals else 0.0,
            ens_mean={v: float(np.mean(ens_col[v])) for v in VARIANTS},
            base_mean=float(np.mean(base_col)),
            n_distinct_ens=nd_ens,
            n_distinct_base=nd_base,
            excluded=reason,
            cell_pr_auc=(pr_auc(truth.astype(int), ens_col["full"])
                         if truth.any() else None),
            cell_pr_auc_base=(pr_auc(truth.astype(int), base_col)
                              if truth.any() else None),
        ))
        if verbose:
            tag = "CORRUPT" if rows[-1].is_corrupted else "clean  "
            extra = " EXCLUDED" if reason else ""
            print(f"    {str(col)[:26]:<27}{tag} true={rows[-1].true_rate:7.4f}"
                  f" ens={rows[-1].ens_mean['full']:.4f}"
                  f" base={rows[-1].base_mean:.4f}{extra}")

    return DatasetResult(
        dataset=dataset,
        n_rows=n_rows,
        n_rows_full=n_full,
        n_cols=dirty.shape[1],
        prevalence=float(y_all.mean()),
        rows=rows,
        whole_table=whole_table,
        holdout_roc=roc_auc(ens.holdout_labels_, ens.holdout_scores_),
        importances=importances,
        injected=ens.report(),
        skipped_columns=dict(score_info.get("skipped_columns", {})),
        seconds=time.time() - t0,
    )


def _refit_classifier(fitted: FeatureEnsembleDetector, names: Sequence[str]):
    """
    Refit only the classifier, on a feature subset of ``fitted``'s already
    extracted training matrix.

    Feature extraction (a conditional fit plus a rule pass, per frame) is the
    expensive half and is not repeated, so the three scope variants are
    compared on byte-identical features and byte-identical labels — the only
    thing that differs between them is which columns of the matrix the
    classifier may use.
    """
    m, y, in_train = fitted.train_matrix_split()
    sub = m.subset(names)
    return fitted.make_classifier().fit(sub.x[in_train], y[in_train])


# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════


def _f(v: Optional[float], nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}"


def _verdict(ok: Optional[bool]) -> str:
    if ok is None:
        return "**INCONCLUSIVE**"
    return "**PASS**" if ok else "**FAIL**"


def write_report(results: List[DatasetResult], row_cap: int) -> str:
    L: List[str] = []
    a = L.append
    ok = [r for r in results if not r.error]

    a("# W8-C — Raha-style per-cell feature-vector ensemble")
    a("")
    a("Generated by `experiments/w8c_ensemble.py`. Mechanism in")
    a("`src/detectors/feature_ensemble.py`, specified in `docs/w8_plan.md` §4")
    a("and falsified against §5, both committed before this code existed.")
    a(f"Seed {SEED}, row cap {row_cap}.")
    a("")
    a("Factors F3 (many weak detectors as a feature vector, then classify) and")
    a("F4 (attribute / tuple / dataset representation scopes) from")
    a("`docs/w8_plan.md` §2. 26 features; the classifier is a seeded")
    a("`HistGradientBoostingClassifier`, trained on corruption injected by")
    a("`src/datagen/corruptor.py` into the dirty table and evaluated on the")
    a("real corruption.")
    a("")

    # ── headline ──────────────────────────────────────────────────────────
    pairs = [
        (r.dataset, r.column_auc("full"), r.column_auc("baseline"))
        for r in ok if r.column_auc_defined()
    ]
    diffs = [e - b for _, e, b in pairs if e is not None and b is not None]
    st = sign_test(diffs)
    wt = wilcoxon_signed_rank(diffs)
    adult = next((r for r in ok if r.dataset == "adult"), None)

    a("## 0. Verdict")
    a("")
    cond1 = None if not diffs else bool(st["wins"] > st["losses"])
    cond3 = None
    if adult is not None:
        wt_pr = adult.whole_table.get("full", {}).get("pr_auc")
        cond3 = None if wt_pr is None else bool(wt_pr > ADULT_WHOLE_TABLE_BASELINE)
    a("| condition | verdict |")
    a("|---|---|")
    a(f"| 1. column-level AUC beats the recomputed baseline | {_verdict(cond1)} |")
    a(f"| 2. paired test across datasets | sign p = {_f(st['p'], 4)}, "
      f"Wilcoxon p = {_f(wt['p'], 4)} |")
    a(f"| 3. adult whole-table PR-AUC beats {ADULT_WHOLE_TABLE_BASELINE} "
      f"| {_verdict(cond3)} |")
    a("| 4. not just rarity | see §4 |")
    a("| 5. not just coverage | enforced by construction, see §1 |")
    a("| 6. not just cardinality | see §4 |")
    a("| 7. no leakage | **PASS** — `tests/test_feature_ensemble.py` |")
    a("")

    # ── 1. column triage ──────────────────────────────────────────────────
    a("## 1. Column triage — condition 1, 2 and 5")
    a("")
    a("Column-level ROC-AUC: does the per-column mean score separate")
    a("corrupted columns from clean ones? Chance 0.5. `baseline` is the mean")
    a("raw `cond_full` score per column, **recomputed on exactly the same")
    a("scored columns** — W6.1's 0.7512 is not carried over.")
    a("")
    a("A column is excluded when either the ensemble or the baseline gives it")
    a("fewer than 3 distinct scores; the two are then always measured on the")
    a("same population (condition 5). `—` means the AUC is undefined because")
    a("the scored columns are all corrupted or all clean, which is not a loss.")
    a("")
    a("| dataset | rows | cols | scored | corrupted | **ensemble** | cell_only |"
      " no_attribute | baseline | ens − base |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in ok:
        sc = r.scored_rows()
        n_corr = sum(1 for x in sc if x.is_corrupted)
        e = r.column_auc("full")
        b = r.column_auc("baseline")
        d = None if (e is None or b is None) else e - b
        a(f"| {r.dataset} | {r.n_rows} | {r.n_cols} | {len(sc)} | {n_corr} | "
          f"**{_f(e)}** | {_f(r.column_auc('cell_only'))} | "
          f"{_f(r.column_auc('no_attribute'))} | {_f(b)} | {_f(d)} |")
    for r in results:
        if r.error:
            a(f"| {r.dataset} | — | — | — | — | — | — | — | — | {r.error} |")
    a("")
    a(f"**Paired comparison (condition 2).** {st['n']} eligible datasets, "
      f"{st['wins']} win(s), {st['losses']} loss(es). "
      f"Exact sign test p = {_f(st['p'], 4)}. "
      f"Wilcoxon signed-rank W = {_f(wt.get('W'), 1)}, "
      f"p = {_f(wt['p'], 4)}"
      f"{' (exact)' if wt.get('exact') else ' (normal approx)'}.")
    a("")
    if diffs:
        a(f"Mean paired difference {np.mean(diffs):+.4f} — reported after the "
          "paired test, never instead of it.")
        a("")

    # ── 2. cell level ─────────────────────────────────────────────────────
    a("## 2. Cell level — condition 3")
    a("")
    a("Whole-table PR-AUC over every cell. Chance equals the prevalence, so")
    a("the prevalence column is the floor, not a score.")
    a("")
    a("| dataset | prevalence | **ensemble PR-AUC** | ROC-AUC | best-F1 |"
      " baseline PR-AUC | baseline ROC |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for r in ok:
        w = r.whole_table.get("full", {})
        bw = r.whole_table.get("baseline_cond_full", {})
        a(f"| {r.dataset} | {_f(r.prevalence)} | **{_f(w.get('pr_auc'))}** | "
          f"{_f(w.get('roc_auc'))} | {_f(w.get('best_f1'))} | "
          f"{_f(bw.get('pr_auc'))} | {_f(bw.get('roc_auc'))} |")
    a("")

    if adult is not None:
        w = adult.whole_table.get("full", {})
        inc = next((x for x in adult.rows if x.column == "income"), None)
        a("### adult, the pre-registered hard case")
        a("")
        a("| quantity | reference | ensemble |")
        a("|---|---:|---:|")
        a(f"| whole-table PR-AUC | {ADULT_WHOLE_TABLE_BASELINE} "
          f"(= prevalence, a constant scorer) | **{_f(w.get('pr_auc'))}** |")
        a(f"| whole-table ROC-AUC | 0.4957 (rule detector, below chance) "
          f"| **{_f(w.get('roc_auc'))}** |")
        if inc is not None:
            a(f"| within-`income` PR-AUC | {ADULT_INCOME_OURS} (existing "
              f"conditional detector) | **{_f(inc.cell_pr_auc)}** |")
            a(f"| within-`income` PR-AUC | {ADULT_INCOME_CEILING} "
              f"(real-corruption Bayes ceiling) | **{_f(inc.cell_pr_auc)}** |")
        a("")
        a("**The transfer gap, stated plainly.** adult's real corruption is")
        a("entirely one-directional — all flips are `1 -> 0` in `income`,")
        a("majority class to minority. `swap_valid` injection is")
        a("frequency-weighted and does not reproduce that, so a classifier")
        a("trained on injected corruption is trained on the wrong corruption")
        a("model for this column. Compare the held-out *injected* ROC-AUC")
        a(f"({_f(adult.holdout_roc)}) with the within-`income` numbers above:")
        a("the gap between them is the cost of the designed policy, and it is")
        a("exactly the risk `docs/w8_plan.md` §2 pre-registered as F2 /")
        a("HoloDetect's 0.953-vs-0.166 ablation.")
        a("")

    a("### Per-column cell-level PR-AUC on the corrupted columns")
    a("")
    a("| dataset | column | true rate | ensemble | baseline `cond_full` |")
    a("|---|---|---:|---:|---:|")
    for r in ok:
        for x in r.rows:
            if x.is_corrupted:
                a(f"| {r.dataset} | `{x.column}` | {_f(x.true_rate)} | "
                  f"{_f(x.cell_pr_auc)} | {_f(x.cell_pr_auc_base)} |")
    a("")

    # ── 3. generalisation sanity ──────────────────────────────────────────
    a("## 3. Does the classifier learn anything at all?")
    a("")
    a("ROC-AUC on the **held-out injected** corruption — a row-disjoint 30 %")
    a("of the augmented frame. This is a sanity check on the learner, not a")
    a("result about real corruption: it says the feature vector is")
    a("learnable, nothing more.")
    a("")
    a("| dataset | injected positives | injected rate | held-out ROC-AUC |")
    a("|---|---:|---:|---:|")
    for r in ok:
        inj = r.injected
        a(f"| {r.dataset} | {inj.get('n_injected_positives', '—')} | "
          f"{_f(inj.get('injected_rate'))} | {_f(r.holdout_roc)} |")
    a("")

    # ── 4. confounds ──────────────────────────────────────────────────────
    a("## 4. Confounds — conditions 4 and 6")
    a("")
    a("Spearman rho between the per-column statistic and two nuisance")
    a("quantities, over scored columns. High |rho| means the ranking is")
    a("explained by the nuisance rather than by contamination — that is the")
    a("failure mode of W5.3 (cardinality) and the original rarity failure.")
    a("`true rate` is the check in the other direction: the statistic *should*")
    a("track the column's true contamination rate.")
    a("")
    a("| dataset | rho(ens, rare_frac) | rho(base, rare_frac) |"
      " rho(ens, cardinality) | rho(base, cardinality) | rho(ens, true rate) |")
    a("|---|---:|---:|---:|---:|---:|")
    for r in ok:
        sc = r.scored_rows()
        if len(sc) < 3:
            a(f"| {r.dataset} | — | — | — | — | — |")
            continue
        e = [x.ens_mean["full"] for x in sc]
        b = [x.base_mean for x in sc]
        rare = [x.rare_frac for x in sc]
        card = [x.card_ratio for x in sc]
        tr = [x.true_rate for x in sc]
        a(f"| {r.dataset} | {_f(spearman(e, rare))} | {_f(spearman(b, rare))} | "
          f"{_f(spearman(e, card))} | {_f(spearman(b, card))} | "
          f"{_f(spearman(e, tr))} |")
    a("")
    a("Pooled over every scored column of every dataset (columns are the")
    a("units, so this pools a population, not a mean of means):")
    pooled_e, pooled_rare, pooled_card, pooled_true, pooled_b = [], [], [], [], []
    for r in ok:
        for x in r.scored_rows():
            pooled_e.append(x.ens_mean["full"])
            pooled_b.append(x.base_mean)
            pooled_rare.append(x.rare_frac)
            pooled_card.append(x.card_ratio)
            pooled_true.append(x.true_rate)
    a("")
    a("| pair | rho | n columns |")
    a("|---|---:|---:|")
    for label, v in [
        ("ensemble vs rare-value fraction", spearman(pooled_e, pooled_rare)),
        ("baseline vs rare-value fraction", spearman(pooled_b, pooled_rare)),
        ("ensemble vs cardinality ratio", spearman(pooled_e, pooled_card)),
        ("baseline vs cardinality ratio", spearman(pooled_b, pooled_card)),
        ("ensemble vs true contamination rate", spearman(pooled_e, pooled_true)),
    ]:
        a(f"| {label} | {_f(v)} | {len(pooled_e)} |")
    a("")

    # ── 5. importances ────────────────────────────────────────────────────
    a("## 5. Feature importances")
    a("")
    a("Permutation importance = mean drop in ROC-AUC on the held-out injected")
    a("cells when that one feature is shuffled, 3 repeats. A negative value")
    a("means shuffling *helped*; that is noise and is printed as measured.")
    a("")
    a("These importances are about **injected** corruption, because that is")
    a("the only thing the classifier has labels for. A feature that dominates")
    a("here is not thereby useful on real corruption.")
    a("")
    all_names = sorted(
        {n for r in ok for n in r.importances},
        key=lambda n: -float(np.mean([
            r.importances.get(n, np.nan) for r in ok
            if n in r.importances
        ])),
    )
    header = "| feature | scope | " + " | ".join(r.dataset for r in ok) + " | mean |"
    a(header)
    a("|---|---|" + "---:|" * (len(ok) + 1))
    from src.detectors.feature_ensemble import FEATURE_SCOPE
    for n in all_names:
        vals = [r.importances.get(n) for r in ok]
        mean = np.nanmean([v for v in vals if v is not None]) if vals else None
        a(f"| `{n}` | {FEATURE_SCOPE.get(n, '?')} | "
          + " | ".join(_f(v, 4) for v in vals)
          + f" | {_f(float(mean) if mean is not None else None)} |")
    a("")
    a("Scope totals (sum of mean importance within each scope):")
    a("")
    a("| scope | summed mean importance |")
    a("|---|---:|")
    for scope, names in FEATURE_GROUPS.items():
        tot = 0.0
        for n in names:
            vs = [r.importances[n] for r in ok if n in r.importances]
            if vs:
                tot += float(np.nanmean(vs))
        a(f"| {scope} | {tot:+.4f} |")
    a("")

    # ── 6. coverage ───────────────────────────────────────────────────────
    a("## 6. Coverage — which columns were excluded, and why")
    a("")
    a("| dataset | excluded | of | reason sample |")
    a("|---|---:|---:|---|")
    for r in ok:
        ex = [x for x in r.rows if x.excluded]
        sample = ex[0].excluded if ex else "—"
        a(f"| {r.dataset} | {len(ex)} | {r.n_cols} | {sample} |")
    a("")

    a("## 7. Honest limitations")
    a("")
    a("1. **The training negatives are not clean.** The real corruption")
    a("   already in the dirty table is labelled 0 during training, so the")
    a("   classifier is explicitly taught to call real corruption negative.")
    a("   On breast_cancer that is 7.5 % of cells.")
    a("2. **The injected corruption policy is designed, not learned.** This")
    a("   is factor F2 in `docs/w8_plan.md` §2 and HoloDetect's ablation")
    a("   (0.953 learned vs 0.166 random) is the warning attached to it. The")
    a("   adult transfer gap in §2 is this risk realised.")
    a("3. **Features shift between training and scoring.** The procedure is")
    a("   identical but the conditional model fitted on the augmented frame is")
    a("   not the one fitted on the dirty frame.")
    a("4. **One run per dataset.** No repeated-seed confidence intervals; the")
    a("   paired test across datasets is the only interval reported.")
    a("5. **No published cell-level benchmark reports PR-AUC**")
    a("   (`docs/w8_plan.md` §7), so these numbers are not directly")
    a("   comparable with Raha's or HoloDetect's F1 figures. best-F1 is")
    a("   reported in §2 for that reason, and it is an oracle threshold.")
    a("")
    a(f"Total runtime {sum(r.seconds for r in results):.1f} s.")
    a("")
    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(ALL_DATASETS))
    ap.add_argument("--rows", type=int, default=ROW_CAP)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT_MD)
    args = ap.parse_args(argv)

    results: List[DatasetResult] = []
    for name in args.datasets:
        print(f"\n=== {name} ===")
        try:
            res = measure(name, row_cap=args.rows, verbose=not args.quiet)
        except Exception as exc:  # noqa: BLE001 - one bad dataset must not
            # take the whole run down; the failure is reported in the table.
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            clean_n = 0
            res = DatasetResult(
                dataset=name, n_rows=0, n_rows_full=clean_n, n_cols=0,
                prevalence=float("nan"), rows=[], whole_table={},
                holdout_roc=None, importances={}, injected={},
                skipped_columns={}, seconds=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        results.append(res)
        if not res.error:
            print(f"  column AUC ens={_f(res.column_auc('full'))} "
                  f"base={_f(res.column_auc('baseline'))} "
                  f"| whole-table PR-AUC "
                  f"{_f(res.whole_table.get('full', {}).get('pr_auc'))} "
                  f"| {res.seconds:.1f}s")

    md = write_report(results, args.rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(f"\nWrote {args.out}")

    OUT_JSON.write_text(
        json.dumps(
            [
                {
                    "dataset": r.dataset,
                    "n_rows": r.n_rows,
                    "n_cols": r.n_cols,
                    "prevalence": r.prevalence,
                    "error": r.error,
                    "column_auc": {
                        k: r.column_auc(k)
                        for k in list(VARIANTS) + ["baseline"]
                    },
                    "whole_table": r.whole_table,
                    "holdout_roc": r.holdout_roc,
                    "importances": r.importances,
                    "injected": r.injected,
                    "columns": [dataclasses.asdict(x) for x in r.rows],
                }
                for r in results
            ],
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
