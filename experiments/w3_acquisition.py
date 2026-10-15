"""
experiments/w3_acquisition.py
-----------------------------
Wave 3: **where should a tiny label budget be spent?**

THE MEASURED PROBLEM (wave 2, `results/w2_ensemble.md`), PR-AUC on adult:

    cond_only 0.0110 | b20 0.0106 | b100 0.2656 | FULL 0.7024

Labels win on adult. Acquiring them is what is broken, at 0.251 % prevalence:

- k <= 20 draws **zero** positives in 5/5 splits, so it buys nothing;
- ``global_top`` draws zero positives at every k <= 100, because the global
  top is the rules' ``fnlwgt`` false positives plus blank
  ``workclass``/``occupation`` cells — all guaranteed clean. The rules
  actively *poison* the acquisition;
- the b100 -> full gap (0.2656 -> 0.7024, 2.6x) is therefore a **sampling**
  problem, not a learner problem.

Published label-efficiency results (Raha: 2 labels ~0.40 F1 -> 20 labels 0.85)
are measured at 5-40 % error rates and do not transfer here; that negative
result is already ours. This script asks the constructive question: **can a
smarter acquisition strategy close the gap with far fewer labels?**

THE DELIVERABLE is not a PR-AUC table. It is, per strategy:

    how many labels to reach 50 % and 80 % of full-supervision PR-AUC

plus **positives-found-per-query**, which is the mechanism-level number that
separates "the strategy found corruption" from "the learner got lucky".

PROTOCOL
========
``src.detection_eval.harness.evaluate_detector``: cell-level stratified
splits, refit per split, bootstrap CIs over splits. Every label goes through
``ensemble.LabelOracle``, whose budget is asserted exactly against the number
of fits, so an over-spending strategy fails loudly rather than quietly
winning. Fits see train cells only; the harness owns the split.

One-shot (``n_rounds=1``) is compared with sequential (``n_rounds>1``) at the
**same total k**, which is where active learning earns its keep or does not.

Run (deterministic, seed 42):

    python experiments/w3_acquisition.py
    python experiments/w3_acquisition.py --quick     # smaller sweep
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.console import enable_unicode_stdout  # noqa: E402
from src.data.loader import DataLoader  # noqa: E402
from src.data.profiler import profile_dataset  # noqa: E402
from src.detection_eval.harness import evaluate_detector  # noqa: E402
from src.detection_eval.labels import cell_labels, label_stats  # noqa: E402
from src.detectors.acquisition import STRATEGIES, AcquisitionDetector  # noqa: E402
from src.detectors.ensemble import FeatureCache, LabelOracle  # noqa: E402

DATASETS = ("breast_cancer", "adult", "smartfactory")
BUDGETS = (10, 20, 50, 100, 200, 500)
QUICK_BUDGETS = (20, 100)
ROUNDS = (1, 5)
TARGETS = (0.50, 0.80)
SEED = 42
N_SPLITS = 5


@dataclass
class Row:
    dataset: str
    strategy: str
    budget: int
    n_rounds: int
    pr_auc: Optional[float]
    pr_lo: Optional[float]
    pr_hi: Optional[float]
    best_f1: Optional[float]
    positives_per_query: Optional[float]
    n_positive_found: Optional[float]
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset, "strategy": self.strategy,
            "budget": self.budget, "n_rounds": self.n_rounds,
            "pr_auc": self.pr_auc, "pr_lo": self.pr_lo, "pr_hi": self.pr_hi,
            "best_f1": self.best_f1,
            "positives_per_query": self.positives_per_query,
            "n_positive_found": self.n_positive_found,
            "seconds": round(self.seconds, 1), "notes": self.notes,
        }


def _json_safe(v: Any) -> Any:
    """Non-finite -> None, before encoding. An undefined metric is null,
    never 0.0, which would read as a real score."""
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, float):
        return None if not math.isfinite(v) else v
    return v


def _iv(metrics: Dict[str, Any], name: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    iv = (metrics.get("intervals") or {}).get(name) or {}
    mean, lo, hi = iv.get("mean"), iv.get("lo"), iv.get("hi")
    if mean is None or (isinstance(mean, float) and not math.isfinite(mean)):
        return None, None, None
    return mean, lo, hi


def _fmt(v: Optional[float], nd: int = 4) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def _ci(row: Row) -> str:
    if row.pr_auc is None:
        return "—"
    if row.pr_lo is None:
        return f"{row.pr_auc:.4f}"
    return f"{row.pr_auc:.4f} [{row.pr_lo:.4f}, {row.pr_hi:.4f}]"


def run_one(
    dataset: str, strategy: str, budget: int, n_rounds: int,
    labels, profile, cache: FeatureCache, data_root: Path,
    n_splits: int, seed: int,
) -> Row:
    started = time.time()
    cap = int(budget) * n_splits          # the harness refits once per split
    oracle = LabelOracle(labels, max_queries=cap)
    reports: List[Dict[str, Any]] = []

    det = AcquisitionDetector(
        strategy=strategy, n_rounds=n_rounds, budget=int(budget),
        label_oracle=oracle, profile=profile, seed=seed,
        feature_cache=cache,
        flag_statistical_outliers=(dataset != "smartfactory"),
        name=f"{strategy}_b{budget}_r{n_rounds}",
    )
    original_fit = det.fit

    def fit_and_report(*a, _d=det, _o=original_fit, _s=reports, **kw):
        out = _o(*a, **kw)
        _s.append(_d.report())
        return out

    det.fit = fit_and_report  # type: ignore[method-assign]

    metrics = evaluate_detector(
        det, dataset, data_root=str(data_root), n_splits=n_splits, seed=seed
    )
    pr, lo, hi = _iv(metrics, "pr_auc")
    f1, _, _ = _iv(metrics, "best_f1")

    ppq = [r.get("positives_per_query") for r in reports
           if r.get("positives_per_query") is not None]
    found = [sum(rd.get("n_positive", 0) for rd in r.get("rounds", []))
             for r in reports]
    notes = sorted({n for r in reports for n in r.get("round_notes", [])})

    # Hard budget check: an over-spending strategy must fail, not win quietly.
    assert oracle.n_queries <= cap, (
        f"{strategy} b{budget} r{n_rounds}: spent {oracle.n_queries} > {cap}"
    )
    return Row(
        dataset=dataset, strategy=strategy, budget=int(budget), n_rounds=n_rounds,
        pr_auc=pr, pr_lo=lo, pr_hi=hi, best_f1=f1,
        positives_per_query=(sum(ppq) / len(ppq)) if ppq else None,
        n_positive_found=(sum(found) / len(found)) if found else None,
        seconds=time.time() - started, notes=notes[:3],
    )


def labels_to_reach(rows: Sequence[Row], full_pr: Optional[float],
                    frac: float) -> Optional[int]:
    """Smallest budget whose PR-AUC reaches ``frac`` of full supervision."""
    if full_pr is None or full_pr <= 0:
        return None
    target = frac * full_pr
    hits = [r.budget for r in rows if r.pr_auc is not None and r.pr_auc >= target]
    return min(hits) if hits else None


def run_dataset(dataset: str, budgets: Sequence[int], strategies: Sequence[str],
                rounds: Sequence[int], data_root: Path, n_splits: int,
                seed: int) -> Tuple[List[Row], Dict[str, Any]]:
    loader = DataLoader(data_root)
    clean_df, dirty_df = loader.load(dataset)
    profile = profile_dataset(dataset, loader.discover(dataset), clean_df, dirty_df)
    labels = cell_labels(clean_df, dirty_df)
    stats = label_stats(labels)
    cache = FeatureCache()

    print(f"\n[{dataset}] {stats['n_cells']:,} cells, "
          f"{stats['n_corrupted']:,} corrupted (prevalence {stats['prevalence']:.5f})")

    rows: List[Row] = []
    # Full supervision is the ceiling every strategy is measured against.
    full_row = run_one(dataset, "random", 0, 1, labels, profile, cache,
                       data_root, n_splits, seed)
    full_det = AcquisitionDetector(
        strategy="random", budget="full", label_oracle=LabelOracle(labels),
        profile=profile, seed=seed, feature_cache=cache,
        flag_statistical_outliers=(dataset != "smartfactory"), name="full",
    )
    full_metrics = evaluate_detector(full_det, dataset, data_root=str(data_root),
                                     n_splits=n_splits, seed=seed)
    full_pr, _, _ = _iv(full_metrics, "pr_auc")
    print(f"[{dataset}] label-free b0 {_fmt(full_row.pr_auc)}  "
          f"full-supervision ceiling {_fmt(full_pr)}")

    for strategy in strategies:
        for n_rounds in rounds:
            if n_rounds > 1 and not _sequential_meaningful(strategy):
                continue
            for k in budgets:
                row = run_one(dataset, strategy, k, n_rounds, labels, profile,
                              cache, data_root, n_splits, seed)
                rows.append(row)
                print(f"   {strategy:<20} k={k:<4} r={n_rounds}  "
                      f"PR-AUC {_fmt(row.pr_auc)}  "
                      f"pos/query {_fmt(row.positives_per_query, 3)}  "
                      f"({row.seconds:.0f}s)")
    meta = {"stats": stats, "b0_pr_auc": full_row.pr_auc, "full_pr_auc": full_pr}
    return rows, meta


def _sequential_meaningful(strategy: str) -> bool:
    """Sequential only differs for a strategy that consults the model."""
    from src.detectors.acquisition import USES_MODEL
    return bool(USES_MODEL.get(strategy, False))


def build_report(all_rows: Dict[str, List[Row]], meta: Dict[str, Dict[str, Any]],
                 budgets: Sequence[int], quick: bool) -> str:
    out: List[str] = []
    out.append("# Wave 3 — where should a tiny label budget be spent?")
    out.append("")
    if quick:
        out.append("> **QUICK RUN — not paper-grade.** Reduced budgets/strategies.")
        out.append("")
    out.append("Generated by `python experiments/w3_acquisition.py`. "
               f"Seed {SEED}, {N_SPLITS} stratified cell splits, bootstrap 95 % CIs. "
               "PR-AUC is primary (prevalence is far below 1 % on adult).")
    out.append("")
    out.append("## 1. The headline: labels needed to reach a fraction of full supervision")
    out.append("")
    out.append("Full supervision is the ceiling. A strategy that never reaches the "
               "target at any tested budget is `—`; that is a result, not a gap.")
    out.append("")
    for ds, rows in all_rows.items():
        m = meta[ds]
        out.append(f"### {ds}  (prevalence {m['stats']['prevalence']:.5f}, "
                   f"full-supervision PR-AUC {_fmt(m['full_pr_auc'])})")
        out.append("")
        hdr = ["strategy", "rounds"] + [f"{int(f*100)} % of full" for f in TARGETS] + \
              ["best PR-AUC", "pos/query @ max k"]
        out.append("| " + " | ".join(hdr) + " |")
        out.append("|" + "---|" * len(hdr))
        combos = sorted({(r.strategy, r.n_rounds) for r in rows})
        for strat, nr in combos:
            sub = [r for r in rows if r.strategy == strat and r.n_rounds == nr]
            if not sub:
                continue
            reach = [labels_to_reach(sub, m["full_pr_auc"], f) for f in TARGETS]
            best = max((r for r in sub if r.pr_auc is not None),
                       key=lambda r: r.pr_auc, default=None)
            top = max(sub, key=lambda r: r.budget)
            out.append("| `{}` | {} | {} | {} | {} | {} |".format(
                strat, nr,
                *[str(x) if x is not None else "—" for x in reach],
                _ci(best) if best else "—",
                _fmt(top.positives_per_query, 3)))
        out.append("")

    out.append("## 2. Full label-efficiency curves")
    out.append("")
    for ds, rows in all_rows.items():
        out.append(f"### {ds}")
        out.append("")
        out.append("| strategy | rounds | " + " | ".join(f"k={k}" for k in budgets) + " |")
        out.append("|" + "---|" * (len(budgets) + 2))
        for strat, nr in sorted({(r.strategy, r.n_rounds) for r in rows}):
            cells = []
            for k in budgets:
                hit = next((r for r in rows if r.strategy == strat
                            and r.n_rounds == nr and r.budget == k), None)
                cells.append(_fmt(hit.pr_auc) if hit else "—")
            out.append(f"| `{strat}` | {nr} | " + " | ".join(cells) + " |")
        out.append("")
        out.append("Positives found per query (the mechanism):")
        out.append("")
        out.append("| strategy | rounds | " + " | ".join(f"k={k}" for k in budgets) + " |")
        out.append("|" + "---|" * (len(budgets) + 2))
        for strat, nr in sorted({(r.strategy, r.n_rounds) for r in rows}):
            cells = []
            for k in budgets:
                hit = next((r for r in rows if r.strategy == strat
                            and r.n_rounds == nr and r.budget == k), None)
                cells.append(_fmt(hit.positives_per_query, 3) if hit else "—")
            out.append(f"| `{strat}` | {nr} | " + " | ".join(cells) + " |")
        out.append("")

    out.append("## 3. Machine-readable")
    out.append("")
    out.append("```json")
    out.append(json.dumps(_json_safe({
        ds: {"meta": meta[ds], "rows": [r.to_dict() for r in rows]}
        for ds, rows in all_rows.items()
    }), indent=2, allow_nan=False))
    out.append("```")
    out.append("")
    out.append("## 4. What this does NOT show")
    out.append("")
    out.append("- **One cost/hyper-parameter setting per dataset.** No sweep.")
    out.append("- **Smart Factory is uninformative**: its corruption is exactly the "
               "NaN cells, so every method ties at the ceiling and the label curve "
               "is flat by construction.")
    out.append("- **The labels are free here.** A real deployment pays a human per "
               "query; this measures only how informative each query is.")
    out.append("- **Rule features still inherit the `DatasetProfile` soft-leakage "
               "caveat** (CLAUDE.md §3), identically across strategies.")
    return "\n".join(out)


def main() -> int:
    enable_unicode_stdout()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--splits", type=int, default=N_SPLITS)
    ap.add_argument("--data-root", default="datasets")
    ap.add_argument("--out", default="results/w3_acquisition.md")
    args = ap.parse_args()

    budgets = QUICK_BUDGETS if args.quick else BUDGETS
    rounds = (1,) if args.quick else ROUNDS
    data_root = Path(args.data_root)

    print("=" * 78)
    print("Wave 3 — acquisition strategies at extreme class imbalance")
    print("=" * 78)
    print(f"datasets={args.datasets}  strategies={len(args.strategies)}  "
          f"budgets={list(budgets)}  rounds={list(rounds)}  seed={args.seed}")

    all_rows: Dict[str, List[Row]] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    t0 = time.time()
    for ds in args.datasets:
        rows, m = run_dataset(ds, budgets, args.strategies, rounds,
                              data_root, args.splits, args.seed)
        all_rows[ds], meta[ds] = rows, m

    report = build_report(all_rows, meta, budgets, args.quick)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\n[written] {out_path}")
    print(f"[done] {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
