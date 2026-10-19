"""
experiments/w7_ceiling.py
-------------------------
W7.1 — the controlled corruption study with a **Bayes-optimal ceiling**.

Planned in `docs/w7_plan.md` §3, committed before this file existed.

WHY A CEILING INSTEAD OF A FIFTH MECHANISM
==========================================
Four mechanisms have been refuted, and each died from a confound found
*after* running on real data: cardinality (W5.3), model competence (W6.1),
detector coverage (W6.1 pre-fix). A fifth tested the same way will probably
die the same way. So this measures **how much signal exists to capture**
under conditions we control, so a mechanism can be judged against the
ceiling rather than against our guesses.

Because we inject the corruption ourselves we know the rate `r`, the
injection distribution `P_inject`, and the exact cell positions. That makes
the optimal posterior computable:

    P(corrupt | v, x)  =            r * P_inject(v)
                          ---------------------------------------
                          r * P_inject(v) + (1-r) * P_clean(v | x)

`r` and `P_inject` are known **by construction** (measured from the injector's
own report). `P_clean(v | x)` is estimated from the **clean** frame by an
**oracle** model, cross-fitted so it is not scored on its own training rows.

Ranking cells by that posterior gives the PR-AUC no method could exceed, so
the headline becomes

    captured fraction  =  PR-AUC(method) / PR-AUC(ceiling)

turning four failures from "we could not do it" into "here is the information
budget, and here is the share each method captures".

THIS IS AN ESTIMATED CEILING, NOT A THEOREM
===========================================
- `P_clean(v|x)` carries estimation error. Where the oracle is poor the
  ceiling is **understated**, so a captured fraction above 1.0 is possible
  and means the oracle was weaker than the method, not that the method beat
  optimality.
- It is conditional on the injection model. A different `P_inject` gives a
  different ceiling.
- The oracle reads clean data and is **non-deployable by construction** — the
  same clearly-labelled diagnostic use as the `(ii)` AUC in W5.3. No
  deployable statistic in this script touches it.

INTERNAL VALIDITY CHECK — a test that can fail
==============================================
For `missing`, the injected value is NaN, which never occurs in the clean
column, so `P_clean(NaN|x) = 0` and the optimal posterior is exactly 1 on
injected cells and 0 elsewhere. **The ceiling must therefore come out at
PR-AUC = 1.0 for `missing`.** If it does not, the ceiling computation is
wrong and no other number in the run can be trusted. The report states the
check and its outcome before any headline figure.

THE RARE-BUT-VALID CONTROL (plan §3.2)
======================================
A value that is rare *but correct* cannot be made by editing a cell —
editing makes it wrong. So rarity has to come from the column's natural
distribution, and the control comes from choosing **which column to
corrupt**:

    config A (the hard, real case):  corrupt a LOW-TAIL column,
                                     leave a HEAVY-TAILED column clean
    config B (the reverse):          corrupt the HEAVY-TAILED column,
                                     leave the low-tail one clean

Config A is the adult `income`-versus-`native_country` failure built
deliberately and repeatably. A statistic that tracks rarity rather than
corruption ranks the clean heavy-tailed column first and fails by design.

Run:
    python experiments/w7_ceiling.py
    python experiments/w7_ceiling.py --datasets adult --rows 6000

Writes results/w7_ceiling.md. Deterministic: seed 42.
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
from src.datagen.corruptor import inject  # noqa: E402
from src.detection_eval.metrics import pr_auc, roc_auc  # noqa: E402
from src.detectors.conditional import ConditionalDetector  # noqa: E402

enable_unicode_stdout()

SEED = 42
ROW_CAP = 8000
RATES = (0.0025, 0.01, 0.05)
TYPES = ("missing", "swap_valid")
OUT_MD = ROOT / "results" / "w7_ceiling.md"

#: (dataset, low-tail column, heavy-tail column). Chosen from the clean
#: frame's rare-value fractions, printed in the report so the choice is
#: auditable rather than asserted.
TARGETS: Dict[str, Tuple[str, str]] = {
    "adult": ("income", "native_country"),
    "breast_cancer": ("class", "Clump Thickness"),
}


def rare_fraction(series: pd.Series, thresh: float = 0.01) -> float:
    """Fraction of non-null cells whose value occurs in < `thresh` of rows."""
    s = series.dropna()
    if s.empty:
        return float("nan")
    freq = s.value_counts(normalize=True)
    return float(s.map(freq).lt(thresh).mean())


# ═══════════════════════════════════════════════════════════════════════════
# Oracle P_clean(v | x) — reads clean data, NON-DEPLOYABLE, diagnostic only
# ═══════════════════════════════════════════════════════════════════════════


def oracle_conditional(
    clean: pd.DataFrame, column: str, seed: int = SEED, n_splits: int = 3
) -> Tuple[Dict[object, int], np.ndarray]:
    """
    Cross-fitted ``P_clean(value | rest of row)`` for one column.

    Returns ``(class_index, proba)`` where ``proba[i, class_index[v]]`` is
    P(column = v | row i) under a model trained on the CLEAN frame, using
    out-of-fold predictions so no row is scored by a model that saw it.

    Cross-fitting matters: an in-sample fit would make the oracle look
    sharper than it is and so **overstate** the ceiling.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold

    y_raw = clean[column].astype("object").to_numpy()
    classes = [c for c in pd.unique(y_raw) if pd.notna(c)]
    class_index = {c: i for i, c in enumerate(sorted(classes, key=repr))}
    y = np.array([class_index.get(v, -1) for v in y_raw], dtype=int)

    feats = [c for c in clean.columns if c != column]
    X = np.column_stack([
        pd.factorize(clean[c].astype("object"), use_na_sentinel=True)[0]
        if not pd.api.types.is_numeric_dtype(clean[c])
        else pd.to_numeric(clean[c], errors="coerce").to_numpy(dtype=float)
        for c in feats
    ]) if feats else np.zeros((len(clean), 1))

    proba = np.full((len(clean), len(class_index)), np.nan, dtype=float)
    ok = y >= 0
    # Classes with too few members for the fold count break StratifiedKFold.
    counts = pd.Series(y[ok]).value_counts()
    usable = ok & pd.Series(y).isin(
        counts[counts >= n_splits].index).to_numpy()
    if usable.sum() < n_splits * 2 or pd.Series(y[usable]).nunique() < 2:
        # Fall back to the marginal; recorded in the report as a degenerate
        # oracle rather than silently producing a number.
        marg = np.zeros(len(class_index), dtype=float)
        for v, i in class_index.items():
            marg[i] = float(np.mean(y_raw == v))
        proba[:] = marg
        return class_index, proba

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx_all = np.arange(len(clean))[usable]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, te in skf.split(X[usable], y[usable]):
            model = HistGradientBoostingClassifier(
                random_state=seed, max_iter=100)
            model.fit(X[usable][tr], y[usable][tr])
            p = model.predict_proba(X[usable][te])
            for j, cls in enumerate(model.classes_):
                proba[idx_all[te], int(cls)] = p[:, j]
    # Rows the oracle could not cover get the marginal.
    missing = np.isnan(proba).any(axis=1)
    if missing.any():
        marg = np.nan_to_num(np.nanmean(proba[~missing], axis=0)) \
            if (~missing).any() else np.full(len(class_index),
                                             1.0 / max(len(class_index), 1))
        proba[missing] = marg
    return class_index, np.nan_to_num(proba, nan=0.0)


def bayes_posterior(
    observed: pd.Series,
    class_index: Dict[object, int],
    p_clean: np.ndarray,
    p_inject: Dict[object, float],
    rate: float,
) -> np.ndarray:
    """
    The optimal posterior, elementwise.

        r * P_inject(v)  /  [ r * P_inject(v) + (1-r) * P_clean(v | x) ]

    A value outside the injector's support gets ``P_inject = 0`` and hence
    posterior 0. A value the clean model never emits gets ``P_clean = 0``, so
    the posterior is 1 whenever the injector could have produced it — which
    is what makes `missing` an exact ceiling of 1.0.
    """
    vals = observed.astype("object").to_numpy()
    out = np.zeros(len(vals), dtype=float)
    for i, v in enumerate(vals):
        key = np.nan if (isinstance(v, float) and np.isnan(v)) else v
        pi = p_inject.get(key, 0.0)
        if pi <= 0.0:
            continue
        j = class_index.get(key, -1)
        pc = float(p_clean[i, j]) if j >= 0 else 0.0
        num = rate * pi
        den = num + (1.0 - rate) * pc
        out[i] = num / den if den > 0 else 0.0
    return out


# ═══════════════════════════════════════════════════════════════════════════
# One condition
# ═══════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class Condition:
    dataset: str
    config: str
    target: str
    control: str
    ctype: str
    rate_requested: float
    rate_achieved: float
    n_injected: int
    pr_ours: Optional[float]
    pr_ceiling: Optional[float]
    roc_ours: Optional[float]
    roc_ceiling: Optional[float]
    stat_target: Optional[float]
    stat_control: Optional[float]
    target_rare_frac: float
    control_rare_frac: float
    oracle_degenerate: bool
    seconds: float

    @property
    def captured(self) -> Optional[float]:
        if not self.pr_ours or not self.pr_ceiling or self.pr_ceiling <= 0:
            return None
        return self.pr_ours / self.pr_ceiling

    @property
    def ranks_target_first(self) -> Optional[bool]:
        if self.stat_target is None or self.stat_control is None:
            return None
        return self.stat_target > self.stat_control


def run_condition(
    dataset: str, clean: pd.DataFrame, target: str, control: str,
    config: str, ctype: str, rate: float,
) -> Condition:
    t0 = time.time()
    cols = list(clean.columns)
    others = [c for c in cols if c != target]

    # Inject into the TARGET column only. The base frame is CLEAN, so every
    # positive is one we created and every other column is untouched.
    # `Corruptor.rate` is a fraction of ALL cells, so dividing by the column
    # count makes `rate` the fraction of the TARGET column's cells.
    corrupted, report = inject(
        clean, ctype, rate / len(cols), seed=SEED, protected_columns=others,
    )
    rows = np.array([r for (r, c) in report.positions if c == target], dtype=int)
    y = np.zeros(len(clean), dtype=int)
    y[rows] = 1

    if rows.size == 0:
        return Condition(
            dataset, config, target, control, ctype, rate, 0.0, 0,
            None, None, None, None, None, None,
            rare_fraction(clean[target]), rare_fraction(clean[control]),
            False, time.time() - t0)

    # ── our zero-knowledge method: no clean data, no profile ────────────
    det = ConditionalDetector(mode="full", seed=SEED).fit(corrupted)
    scores = det.score_cells(corrupted)
    s_target = scores[target].to_numpy(dtype=float)

    # ── the oracle ceiling: clean data, NON-DEPLOYABLE ─────────────────
    class_index, p_clean = oracle_conditional(clean, target)
    degenerate = bool(np.allclose(p_clean, p_clean[0], atol=1e-12))
    injected_vals = corrupted[target].astype("object").to_numpy()[rows]
    keys = [np.nan if (isinstance(v, float) and np.isnan(v)) else v
            for v in injected_vals]
    vc = pd.Series(keys, dtype="object").value_counts(normalize=True,
                                                     dropna=False)
    p_inject = {k: float(v) for k, v in vc.items()}
    r_ach = float(rows.size) / len(clean)
    post = bayes_posterior(corrupted[target], class_index, p_clean,
                           p_inject, r_ach)

    # ── column-level statistic for the rare-but-valid comparison ───────
    stat_t = float(np.nanmean(s_target))
    stat_c = float(np.nanmean(scores[control].to_numpy(dtype=float)))

    return Condition(
        dataset=dataset, config=config, target=target, control=control,
        ctype=ctype, rate_requested=rate, rate_achieved=r_ach,
        n_injected=int(rows.size),
        pr_ours=pr_auc(y, s_target), pr_ceiling=pr_auc(y, post),
        roc_ours=roc_auc(y, s_target), roc_ceiling=roc_auc(y, post),
        stat_target=stat_t, stat_control=stat_c,
        target_rare_frac=rare_fraction(clean[target]),
        control_rare_frac=rare_fraction(clean[control]),
        oracle_degenerate=degenerate, seconds=time.time() - t0,
    )


def _f(v: Optional[float], nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}"


# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════


def write_report(conds: List[Condition], rows_used: Dict[str, int]) -> str:
    L: List[str] = []
    a = L.append
    a("# W7.1 — the Bayes-optimal ceiling for corruption detection")
    a("")
    a("Generated by `experiments/w7_ceiling.py`. Planned in")
    a("`docs/w7_plan.md` §3, committed before the code. Seed 42.")
    a("")
    a("Corruption is **injected by us into a clean table**, so the rate `r`,")
    a("the injection distribution `P_inject` and the exact positions are known")
    a("by construction. That makes the optimal posterior computable:")
    a("")
    a("    P(corrupt | v,x) = r*P_inject(v) / [ r*P_inject(v) + (1-r)*P_clean(v|x) ]")
    a("")
    a("Ranking by it gives the **PR-AUC no method could exceed**. The headline")
    a("is then `captured = PR-AUC(ours) / PR-AUC(ceiling)`.")
    a("")

    # ── validity check first: it gates everything ───────────────────────
    a("## 1. Internal validity check (read first)")
    a("")
    a("For `missing` the injected value is NaN, which never occurs in the")
    a("clean column, so `P_clean(NaN|x) = 0` and the optimal posterior is")
    a("exactly 1 on injected cells and 0 elsewhere. **The ceiling must come")
    a("out at PR-AUC = 1.0.** If it does not, the ceiling computation is")
    a("wrong and nothing else here can be trusted.")
    a("")
    miss = [c for c in conds if c.ctype == "missing" and c.pr_ceiling is not None]
    a("| dataset | config | rate | ceiling PR-AUC | = 1.0? |")
    a("|---|---|---:|---:|---|")
    for c in miss:
        ok = c.pr_ceiling is not None and abs(c.pr_ceiling - 1.0) < 1e-9
        a(f"| {c.dataset} | {c.config} | {c.rate_requested:.4f} | "
          f"{_f(c.pr_ceiling)} | {'yes' if ok else '**NO**'} |")
    a("")
    if miss and all(abs(c.pr_ceiling - 1.0) < 1e-9 for c in miss):
        a("**Check passes.** The ceiling computation reproduces the exact")
        a("analytic answer where one exists.")
    elif miss:
        a("**CHECK FAILS.** Treat every other number in this file as")
        a("unverified until the ceiling computation is corrected.")
    else:
        a("No `missing` condition produced a ceiling; the check did not run.")
    a("")

    # ── headline ────────────────────────────────────────────────────────
    a("## 2. How much signal exists, and how much we capture")
    a("")
    a("| dataset | config | type | rate | n injected | ours PR-AUC | **ceiling** | **captured** |")
    a("|---|---|---|---:|---:|---:|---:|---:|")
    for c in conds:
        a(f"| {c.dataset} | {c.config} | `{c.ctype}` | "
          f"{c.rate_requested:.4f} | {c.n_injected:,} | {_f(c.pr_ours)} | "
          f"**{_f(c.pr_ceiling)}** | **{_f(c.captured, 3)}** |")
    a("")
    a("`captured` above 1.0 does **not** mean a method beat optimality — it")
    a("means the oracle `P_clean` was weaker than the method, so the ceiling")
    a("is understated there. The ceiling is an *estimate*, conditional on the")
    a("injection model, and the oracle is non-deployable by construction.")
    a("")

    # ── rare-but-valid control ──────────────────────────────────────────
    a("## 3. The rare-but-valid control")
    a("")
    a("A rare *but correct* value cannot be made by editing a cell, so rarity")
    a("comes from the column's natural distribution and the control comes from")
    a("choosing which column to corrupt. `rare frac` is the fraction of cells")
    a("whose value occurs in under 1 % of rows.")
    a("")
    a("| dataset | config | type | rate | corrupted column (rare frac) | clean control (rare frac) | mean score: target | control | ranks target first? |")
    a("|---|---|---|---:|---|---|---:|---:|---|")
    for c in conds:
        rtf = c.ranks_target_first
        a(f"| {c.dataset} | {c.config} | `{c.ctype}` | {c.rate_requested:.4f} | "
          f"`{c.target}` ({c.target_rare_frac:.3f}) | "
          f"`{c.control}` ({c.control_rare_frac:.3f}) | "
          f"{_f(c.stat_target)} | {_f(c.stat_control)} | "
          f"{'—' if rtf is None else ('**yes**' if rtf else '**NO**')} |")
    a("")
    hard = [c for c in conds
            if c.config == "A" and c.ctype == "swap_valid"
            and c.ranks_target_first is not None]
    if hard:
        n_ok = sum(1 for c in hard if c.ranks_target_first)
        a(f"On the hard case — config A, `swap_valid`, a corrupted low-tail")
        a(f"column against a clean heavy-tailed one — the mean-score statistic")
        a(f"ranks the corrupted column first in **{n_ok} of {len(hard)}**")
        a("conditions.")
        if n_ok == 0:
            a("")
            a("Zero. The statistic tracks **rarity, not corruption**, which is")
            a("the original adult failure reproduced deliberately and")
            a("repeatably instead of observed once.")
    a("")

    a("## 4. Setup")
    a("")
    a("| dataset | rows used | low-tail column | heavy-tailed column |")
    a("|---|---:|---|---|")
    seen = set()
    for c in conds:
        if c.dataset in seen:
            continue
        seen.add(c.dataset)
        lo, hi = (c.target, c.control) if c.config == "A" else (c.control, c.target)
        a(f"| {c.dataset} | {rows_used.get(c.dataset, 0):,} | `{lo}` | `{hi}` |")
    a("")
    a("Config **A** corrupts the low-tail column and leaves the heavy-tailed")
    a("one clean — the hard, real case. Config **B** is the reverse.")
    a("")
    degen = [c for c in conds if c.oracle_degenerate]
    if degen:
        a(f"**Oracle degenerate in {len(degen)} condition(s)** — it fell back")
        a("to the marginal, so the ceiling there reflects marginal rarity")
        a("only and understates what a conditional oracle could do.")
        a("")
    a("Base frames are the **clean** tables, so every positive is one we")
    a("injected and no pre-existing corruption confounds the measurement.")
    a("")
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="W7.1 Bayes ceiling")
    ap.add_argument("--datasets", nargs="*", default=list(TARGETS))
    ap.add_argument("--rows", type=int, default=ROW_CAP)
    ap.add_argument("--rates", type=float, nargs="*", default=list(RATES))
    ap.add_argument("--types", nargs="*", default=list(TYPES))
    args = ap.parse_args(argv)

    conds: List[Condition] = []
    rows_used: Dict[str, int] = {}
    loader = DataLoader("datasets")

    for ds in args.datasets:
        if ds not in TARGETS:
            print(f"  skip {ds}: no target/control pair defined")
            continue
        low, heavy = TARGETS[ds]
        clean_full, _ = loader.load(ds)
        rng = np.random.default_rng(SEED)
        if args.rows > 0 and len(clean_full) > args.rows:
            idx = np.sort(rng.choice(len(clean_full), size=args.rows,
                                     replace=False))
            clean = clean_full.iloc[idx].reset_index(drop=True)
        else:
            clean = clean_full.reset_index(drop=True)
        rows_used[ds] = len(clean)

        if low not in clean.columns or heavy not in clean.columns:
            print(f"  skip {ds}: columns {low}/{heavy} not present")
            continue

        print(f"\n=== {ds} ({len(clean):,} rows) ===")
        print(f"  low-tail `{low}` rare_frac={rare_fraction(clean[low]):.4f}   "
              f"heavy `{heavy}` rare_frac={rare_fraction(clean[heavy]):.4f}")
        for config, tgt, ctl in (("A", low, heavy), ("B", heavy, low)):
            for ctype in args.types:
                for rate in args.rates:
                    c = run_condition(ds, clean, tgt, ctl, config, ctype, rate)
                    conds.append(c)
                    print(f"  {config} {ctype:<11} r={rate:<7.4f} "
                          f"n={c.n_injected:<5} ours={_f(c.pr_ours)} "
                          f"ceiling={_f(c.pr_ceiling)} "
                          f"captured={_f(c.captured, 3)} [{c.seconds:.0f}s]")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(write_report(conds, rows_used), encoding="utf-8")
    print(f"\nWrote {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
