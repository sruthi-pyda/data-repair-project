"""
experiments/w5_leakage_ablation.py
----------------------------------
Route W5.1b of docs/research_program.md §11 — reviewer **gap 4b**, and the
ablation CLAUDE.md §3 has named since day one and nobody had run.

THE CAVEAT, VERBATIM FROM CLAUDE.md §3
======================================
    "Known soft-leakage caveat (document, don't silently 'fix'):
    `DatasetProfile` is built from `clean_df`, so detector IQR fences /
    valid ranges / vocabularies, and the repairer's imputation medians and
    quartiles, are derived from clean aggregate statistics. The code
    documents this as acceptable population-level schema knowledge. A
    reviewer will challenge it. Preferred resolution is an ablation
    computing those stats from dirty data and reporting the gap — not a
    quiet code change that invalidates already-reported numbers."

This script is that ablation. It changes no shipped code path: the clean
arm reproduces every published number, and the dirty arm is a second,
clearly labelled run.

HOW THE DIRTY ARM IS BUILT — AND WHY THIS WAY
=============================================
Three options were available:

  (a) add a `source="clean"|"dirty"` switch to `profile_dataset`;
  (b) add a switch to `run_pipeline`;
  (c) call `profile_dataset(name, paths, dirty_df, dirty_df)` — the dirty
      frame in BOTH positions — and install that wrapper for the duration
      of the dirty arm.

**(c) is used**, because it touches no shipped file at all. Everything the
detector and repairer read off a profile is computed from the frame passed
in the `clean_df` position (`src/data/profiler.py`: `_classify_column_kind`,
`_compute_numerical_stats`, `_compute_categorical_stats` all take
`clean_series`), so passing the dirty frame there makes **every**
detection- and repair-relevant statistic dirty-derived:

    column kind            <- dirty
    min / max / quartiles / IQR / median   <- dirty
    categorical vocabulary and rare-value counts  <- dirty

The only casualty is `_characterize_corruption` / `_count_corrupted_cells`,
which compare the two frames and therefore report zero corruption in the
dirty arm. Nothing in Phase 3 or Phase 4 reads either field — they are
reporting-only — and the experiment **asserts** they came out zero, which
is the proof that the wrapper actually took effect. If the patch silently
failed the two arms would be identical and the gap would look reassuringly
small; that failure mode is what the assertion exists to catch.

Ground truth still enters the dirty arm exactly where it enters the clean
one (0/1 labels for p and q, and scoring) — this ablation removes
*aggregate* clean knowledge, not the permitted label use.

WHAT IS MEASURED, IN FOUR FAMILIES
==================================
  1. DETECTION      PR-AUC / ROC-AUC / oracle best-F1 via
                    `src.detection_eval` (5 stratified cell splits, seed
                    42, bootstrap CIs), plus precision/recall/F1 at the
                    detector's own operating point (`score > 0`) — a rule
                    detector is a SET, so its operating point is the
                    number comparable with published work.
  2. REPAIR         top-1 accuracy over decided records.
  3. CALIBRATION    Brier / ECE per calibrator over 5 calibration splits
                    (`src.experiments.aggregate.repeated_splits`).
  4. COST           realised policy cost, flagged-only AND total harm
                    (W5.1a mode — the only cost that is comparable across
                    the two arms, because the two arms flag different
                    cells but face the same total corruption), plus
                    whether the zero-regret claim survives.

Run:
    python experiments/w5_leakage_ablation.py

Writes results/w5_leakage_ablation.md. Deterministic: seed 42, full
datasets, mock repair mode, shipped detector and cost defaults.
"""

from __future__ import annotations

import dataclasses
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.pipeline.runner as runner_mod  # noqa: E402

from src.console import enable_unicode_stdout  # noqa: E402
from src.data.loader import DataLoader  # noqa: E402
from src.data.profiler import profile_dataset  # noqa: E402
from src.detection.detector import Detector  # noqa: E402
from src.detection_eval.harness import (  # noqa: E402
    RuleDetectorAdapter,
    evaluate_detector,
    flatten_cells,
    stratified_cell_splits,
)
from src.detection_eval.labels import cell_labels  # noqa: E402
from src.experiments.aggregate import (  # noqa: E402
    Interval,
    bootstrap_interval,
    repeated_splits,
)
from src.pipeline.runner import default_flag_outliers, run_pipeline  # noqa: E402
from src.risk.cost_model import CostConfig  # noqa: E402

enable_unicode_stdout()

SEED = 42
N_SPLITS = 5
DATASETS = ["adult", "breast_cancer", "smartfactory"]
COSTS_YAML = ROOT / "configs" / "costs.yaml"
OUT_MD = ROOT / "results" / "w5_leakage_ablation.md"
ARMS = ("clean", "dirty")
IQR_K = 3.0

# Published detection baselines (results/detection_leaderboard.md) — the
# clean arm must reproduce these or nothing else here is trustworthy.
PUBLISHED_PR_AUC = {
    "smartfactory": 1.0000,
    "breast_cancer": 0.6712,
    "adult": 0.0025,
}


# ═══════════════════════════════════════════════════════════════════════════════
# The dirty-profiling wrapper (option (c) — see the module docstring)
# ═══════════════════════════════════════════════════════════════════════════════


def profile_from_dirty(name, paths, clean_df, dirty_df, **kwargs):
    """
    ``profile_dataset`` with the dirty frame in BOTH positions.

    Every statistic Phase 3 and Phase 4 read is computed from the frame
    given in the ``clean_df`` position, so this makes all of them
    dirty-derived. ``clean_df`` is accepted and deliberately discarded.
    """
    return profile_dataset(name, paths, dirty_df, dirty_df, **kwargs)


@contextmanager
def dirty_profiling() -> Iterator[None]:
    """
    Make ``run_pipeline`` (and anything it calls) profile from dirty data.

    Rebinding the name in ``src.pipeline.runner`` is the least invasive
    mechanism available: no shipped file is edited, and the clean arm is
    the untouched code path. ``repeated_splits`` calls ``run_pipeline``,
    so it is covered too.
    """
    original = runner_mod.profile_dataset
    runner_mod.profile_dataset = profile_from_dirty
    try:
        yield
    finally:
        runner_mod.profile_dataset = original


def build_profile(dataset: str, arm: str):
    """Standalone profile for the detection arm (no pipeline involved)."""
    loader = DataLoader("datasets")
    clean_df, dirty_df = loader.load(dataset)
    paths = loader.discover(dataset)
    if arm == "clean":
        return profile_dataset(dataset, paths, clean_df, dirty_df)
    return profile_from_dirty(dataset, paths, clean_df, dirty_df)


def assert_patch_bit(dataset: str, profile, arm: str) -> None:
    """
    Prove the dirty arm is genuinely dirty-profiled.

    If the wrapper silently failed, both arms would be identical and the
    measured gap would be a comforting zero. These two fields are computed
    by comparing the two frames passed in, so in the dirty arm (same frame
    twice) they MUST be zero / equal. They are reporting-only fields that
    no detector or repairer reads, so forcing them is harmless.
    """
    if arm != "dirty":
        return
    if profile.n_corrupted_cells != 0:
        raise RuntimeError(
            f"{dataset}: dirty-profiling wrapper did not take effect — "
            f"profile.n_corrupted_cells = {profile.n_corrupted_cells}, "
            f"expected 0. Refusing to report a fake zero gap."
        )
    if profile.total_missing_clean != profile.total_missing_dirty:
        raise RuntimeError(
            f"{dataset}: dirty-profiling wrapper did not take effect — "
            f"missing counts differ ({profile.total_missing_clean} vs "
            f"{profile.total_missing_dirty})."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Family 1 — detection
# ═══════════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class DetectionArm:
    pr_auc: Interval
    roc_auc: Interval
    best_f1: Interval
    precision: Interval
    recall: Interval
    f1: Interval
    n_flagged: int
    n_tp: int
    n_total_corrupted: int
    distinct_scores: int


def _set_metrics(y: np.ndarray, flagged: np.ndarray) -> Tuple[float, float, float]:
    """Precision / recall / F1 of a flag SET, re-derived from counts here."""
    tp = int(np.sum(flagged & y))
    fp = int(np.sum(flagged & ~y))
    fn = int(np.sum(~flagged & y))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return precision, recall, f1


def measure_detection(dataset: str, arm: str) -> DetectionArm:
    profile = build_profile(dataset, arm)
    assert_patch_bit(dataset, profile, arm)

    detector = Detector(
        profile,
        iqr_k=IQR_K,
        flag_statistical_outliers=default_flag_outliers(dataset),
    )
    adapter = RuleDetectorAdapter(detector=detector, name=f"rule_{arm}_profile")

    out = evaluate_detector(
        adapter, dataset, n_splits=N_SPLITS, seed=SEED, confidence=0.95
    )
    iv = {k: Interval(**{kk: v[kk] for kk in ("mean", "lo", "hi", "std", "n")})
          for k, v in out["intervals"].items()}

    # Operating-point P/R/F1 on the same stratified splits, so the CI is
    # computed the same way the harness computes its own.
    loader = DataLoader("datasets")
    clean_df, dirty_df = loader.load(dataset)
    labels = cell_labels(clean_df, dirty_df)
    y = flatten_cells(labels).astype(bool)
    scores = flatten_cells(adapter.score_cells(dirty_df, profile))
    flagged_all = scores > 0.0
    splits = stratified_cell_splits(y, n_splits=N_SPLITS, seed=SEED)
    p_vals, r_vals, f_vals = [], [], []
    for eval_idx in splits:
        p, r, f = _set_metrics(y[eval_idx], flagged_all[eval_idx])
        p_vals.append(p)
        r_vals.append(r)
        f_vals.append(f)

    return DetectionArm(
        pr_auc=iv["pr_auc"],
        roc_auc=iv["roc_auc"],
        best_f1=iv["best_f1"],
        precision=bootstrap_interval(p_vals, seed=SEED),
        recall=bootstrap_interval(r_vals, seed=SEED),
        f1=bootstrap_interval(f_vals, seed=SEED),
        n_flagged=int(flagged_all.sum()),
        n_tp=int(np.sum(flagged_all & y)),
        n_total_corrupted=int(y.sum()),
        distinct_scores=int(out["score_structure"]["n_distinct"])
        if "n_distinct" in out["score_structure"] else -1,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Families 2-4 — repair, calibration, cost
# ═══════════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class PipelineArm:
    n_decided: int
    n_detected_total: int
    top1_correct: int
    top1_accuracy: float
    calibration_method: str
    brier: Dict[str, Interval]
    ece: Dict[str, Interval]
    n_fitted: Dict[str, int]
    cost: Dict[str, Interval]
    fixed_cost: Dict[str, float]
    n_corrupted_decided: int
    total_harm: Dict[str, float]
    n_unflagged_corrupted: int
    surcharge: float


def measure_pipeline(dataset: str, arm: str) -> PipelineArm:
    costs = CostConfig.from_yaml(COSTS_YAML, dataset)

    ctx = dirty_profiling() if arm == "dirty" else _noop()
    with ctx:
        bundle = run_pipeline(dataset, n_records=0, seed=SEED)
        assert_patch_bit(dataset, bundle.profile, arm)
        rs = repeated_splits(
            dataset, costs, n_splits=N_SPLITS, seed=SEED,
            base_split_seed=SEED, n_records=0,
        )

    idx = bundle.decision_indices()
    n_decided = len(idx)
    top1 = sum(1 for i in idx if bundle.cal_samples[i].correct == 1)
    n_corrupt_decided = sum(1 for i in idx if bundle.detection_labels[i] == 1)

    # Fixed policies: their realised cost does not depend on the calibration
    # split, so they are computed once, from counts, by hand.
    #   always-human   : every decided cell reviewed
    #   always-abstain : every decided cell left alone -> charged only when
    #                    genuinely corrupted
    fixed = {
        "always-human": n_decided * costs.c_human_review,
        "always-abstain": n_corrupt_decided * costs.c_leave_error,
    }

    # Total harm (W5.1a): flagged-only + C_abstain x undetected corruptions.
    # This is the ONLY cost comparable across the two arms, because the arms
    # flag different cells but face the same total corruption.
    from src.evaluation.cost_metrics import corrupted_cells

    census = corrupted_cells(
        bundle.clean_df, bundle.dirty_df, skip_columns=bundle.skip_columns
    )
    flagged_keys = {
        (int(bundle.unc_records[i].row_id), bundle.unc_records[i].column)
        for i in idx
    }
    n_unflagged = len(census - flagged_keys)
    surcharge = costs.c_leave_error * n_unflagged

    total_harm = {
        name: iv.mean + surcharge for name, iv in rs.cost.items()
    }
    for name, value in fixed.items():
        total_harm[name] = value + surcharge
    total_harm["do-nothing (no detector)"] = len(census) * costs.c_leave_error

    return PipelineArm(
        n_decided=n_decided,
        n_detected_total=bundle.n_detected_total,
        top1_correct=top1,
        top1_accuracy=top1 / n_decided if n_decided else float("nan"),
        calibration_method=bundle.calibration_method,
        brier=rs.brier,
        ece=rs.ece,
        n_fitted=rs.n_fitted,
        cost=rs.cost,
        fixed_cost=fixed,
        n_corrupted_decided=n_corrupt_decided,
        total_harm=total_harm,
        n_unflagged_corrupted=n_unflagged,
        surcharge=surcharge,
    )


@contextmanager
def _noop() -> Iterator[None]:
    yield


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════


def _iv(v: Optional[Interval]) -> str:
    if v is None or v.n == 0:
        return "—"
    return f"{v.mean:.4f} [{v.lo:.4f}, {v.hi:.4f}]"


def _delta(a: Optional[Interval], b: Optional[Interval]) -> str:
    """dirty minus clean, with a note when the CIs do not overlap."""
    if a is None or b is None or a.n == 0 or b.n == 0:
        return "—"
    d = b.mean - a.mean
    disjoint = (b.hi < a.lo) or (a.hi < b.lo)
    return f"{d:+.4f}" + (" **(CIs disjoint)**" if disjoint else "")


def _num(x: float) -> str:
    return f"{x:,.0f}" if abs(x - round(x)) < 1e-9 else f"{x:,.2f}"


def write_report(
    det: Dict[str, Dict[str, DetectionArm]],
    pipe: Dict[str, Dict[str, PipelineArm]],
) -> str:
    L: List[str] = []
    a = L.append

    a("# W5.1b — the soft-leakage ablation: profiling from dirty data")
    a("")
    a("Route W5.1b of `docs/research_program.md` §11, reviewer **gap 4b**, and")
    a("the ablation `CLAUDE.md` §3 has named since day one. Generated by")
    a("`experiments/w5_leakage_ablation.py`. Seed 42, full datasets, mock repair")
    a("mode, shipped detector/cost defaults, 5 splits with bootstrap 95 % CIs.")
    a("")
    a("## 0. What the two arms are")
    a("")
    a("| arm | `DatasetProfile` built from | what that means |")
    a("|---|---|---|")
    a("| **clean** | `clean_df` | the shipped code path; reproduces every published number |")
    a("| **dirty** | `dirty_df` only | IQR fences, valid ranges, vocabularies, imputation medians/quartiles and column kinds all derived from corrupted data |")
    a("")
    a("Mechanism: `profile_dataset(name, paths, dirty_df, dirty_df)` — the dirty")
    a("frame in both positions — installed for the dirty arm by rebinding the")
    a("name inside `src.pipeline.runner`. **No shipped file is edited.** Every")
    a("statistic Phase 3 and Phase 4 read comes from the frame passed in the")
    a("`clean_df` position, so all of them become dirty-derived.")
    a("")
    a("The wrapper is verified, not trusted: `profile.n_corrupted_cells` and the")
    a("two missing-value counts are computed by comparing the two frames, so in")
    a("the dirty arm they must be 0 and equal. The script **aborts** if they are")
    a("not — a silently failed patch would show a reassuring zero gap.")
    a("")
    a("Ground truth still enters the dirty arm exactly where it enters the clean")
    a("one (0/1 labels for `p` and `q`, and scoring). This ablation removes")
    a("*aggregate* clean knowledge, not the permitted label use.")
    a("")

    # ── 1. detection ─────────────────────────────────────────────────────
    a("## 1. Detection")
    a("")
    a("PR-AUC is primary at these prevalences; the operating-point P/R/F1 is")
    a("what a rule detector actually delivers (it is a *set*, not a ranking).")
    a("")
    a("| dataset | metric | clean profile | dirty profile | delta |")
    a("|---|---|---|---|---|")
    for ds in DATASETS:
        c, d = det[ds]["clean"], det[ds]["dirty"]
        for label, ca, da in (
            ("PR-AUC", c.pr_auc, d.pr_auc),
            ("ROC-AUC", c.roc_auc, d.roc_auc),
            ("best-F1 (oracle)", c.best_f1, d.best_f1),
            ("precision @ score>0", c.precision, d.precision),
            ("recall @ score>0", c.recall, d.recall),
            ("F1 @ score>0", c.f1, d.f1),
        ):
            a(f"| {ds} | {label} | {_iv(ca)} | {_iv(da)} | {_delta(ca, da)} |")
    a("")
    a("Flag counts on the whole dataset (no split), which explain the above:")
    a("")
    a("| dataset | corrupted cells | clean: flagged / TP | dirty: flagged / TP |")
    a("|---|---:|---:|---:|")
    for ds in DATASETS:
        c, d = det[ds]["clean"], det[ds]["dirty"]
        a(
            f"| {ds} | {c.n_total_corrupted:,} | "
            f"{c.n_flagged:,} / {c.n_tp:,} | {d.n_flagged:,} / {d.n_tp:,} |"
        )
    a("")
    a("### Published-baseline check (clean arm)")
    a("")
    a("| dataset | published PR-AUC | clean arm here | agrees? |")
    a("|---|---:|---:|---|")
    for ds in DATASETS:
        pub = PUBLISHED_PR_AUC[ds]
        got = det[ds]["clean"].pr_auc.mean
        ok = abs(got - pub) < 5e-4
        a(f"| {ds} | {pub:.4f} | {got:.4f} | {'yes' if ok else '**NO**'} |")
    a("")

    # ── 2. repair ────────────────────────────────────────────────────────
    a("## 2. Repair top-1 accuracy")
    a("")
    a("Fraction of decided records whose top candidate equals the clean value.")
    a("The two arms flag different cells, so the denominators differ — both are")
    a("printed, because an accuracy on a different population is not a")
    a("like-for-like comparison and must not be read as one.")
    a("")
    a("| dataset | clean: correct / decided | clean acc | dirty: correct / decided | dirty acc | delta |")
    a("|---|---:|---:|---:|---:|---:|")
    for ds in DATASETS:
        c, d = pipe[ds]["clean"], pipe[ds]["dirty"]
        a(
            f"| {ds} | {c.top1_correct:,} / {c.n_decided:,} | {c.top1_accuracy:.4f} | "
            f"{d.top1_correct:,} / {d.n_decided:,} | {d.top1_accuracy:.4f} | "
            f"{d.top1_accuracy - c.top1_accuracy:+.4f} |"
        )
    a("")

    # ── 3. calibration ───────────────────────────────────────────────────
    a("## 3. Calibration — Brier and ECE")
    a("")
    a("5 calibration splits (split seeds 42–46), pipeline seed fixed at 42.")
    a("`n_fitted` is how many of the 5 splits that calibrator could fit in;")
    a("`0/5` prints as `—` and is a refusal to fit, never a score of 0.")
    a("")
    for ds in DATASETS:
        c, d = pipe[ds]["clean"], pipe[ds]["dirty"]
        a(f"### {ds}")
        a("")
        a("| calibrator | fitted (clean/dirty) | Brier clean | Brier dirty | delta | ECE clean | ECE dirty | delta |")
        a("|---|---|---|---|---|---|---|---|")
        for m in sorted(set(c.brier) | set(d.brier)):
            cb, db = c.brier.get(m), d.brier.get(m)
            ce, de = c.ece.get(m), d.ece.get(m)
            a(
                f"| {m} | {c.n_fitted.get(m, 0)}/{N_SPLITS} , "
                f"{d.n_fitted.get(m, 0)}/{N_SPLITS} | "
                f"{_iv(cb)} | {_iv(db)} | {_delta(cb, db)} | "
                f"{_iv(ce)} | {_iv(de)} | {_delta(ce, de)} |"
            )
        a("")

    # ── 4. cost ──────────────────────────────────────────────────────────
    a("## 4. Realised cost and the zero-regret claim")
    a("")
    a("Two costings, both needed:")
    a("")
    a("- **flagged-only** — the published metric. NOT comparable across arms:")
    a("  the arms flag different numbers of cells, so a lower flagged-only cost")
    a("  can mean a better policy or merely a smaller flag set.")
    a("- **total harm** (W5.1a mode) — flagged-only plus")
    a("  `C_abstain x undetected corruptions`. The denominator is the dataset's")
    a("  whole corruption, which is the same for both arms, so **this is the")
    a("  comparison that is actually like-for-like.**")
    a("")
    for ds in DATASETS:
        c, d = pipe[ds]["clean"], pipe[ds]["dirty"]
        a(f"### {ds}")
        a("")
        a(
            f"Flags: clean {c.n_decided:,} decided ({c.n_corrupted_decided:,} "
            f"genuinely corrupted, {c.n_unflagged_corrupted:,} missed, surcharge "
            f"{_num(c.surcharge)}); dirty {d.n_decided:,} decided "
            f"({d.n_corrupted_decided:,} genuinely corrupted, "
            f"{d.n_unflagged_corrupted:,} missed, surcharge {_num(d.surcharge)})."
        )
        a("")
        a("| policy | flagged-only clean | flagged-only dirty | total harm clean | total harm dirty | total-harm delta |")
        a("|---|---|---|---:|---:|---:|")
        names = sorted(
            set(c.cost) | set(d.cost) | set(c.fixed_cost) | set(d.fixed_cost)
        )
        for name in names:
            cf = _iv(c.cost[name]) if name in c.cost else _num(c.fixed_cost[name])
            df = _iv(d.cost[name]) if name in d.cost else _num(d.fixed_cost[name])
            ct = c.total_harm.get(name)
            dt = d.total_harm.get(name)
            delta = f"{dt - ct:+,.1f}" if (ct is not None and dt is not None) else "—"
            a(
                f"| {name} | {cf} | {df} | "
                f"{_num(ct) if ct is not None else '—'} | "
                f"{_num(dt) if dt is not None else '—'} | {delta} |"
            )
        dn = c.total_harm["do-nothing (no detector)"]
        a(f"| _do-nothing (no detector)_ | — | — | {_num(dn)} | {_num(dn)} | 0 |")
        a("")

    # ── 5. zero regret ───────────────────────────────────────────────────
    a("## 5. Does the zero-regret claim survive dirty profiling?")
    a("")
    a("`results/RESULTS.md` claims the proposed policy is the only one with")
    a("**zero worst-case regret**: best or tied-best on every dataset. Regret is")
    a("re-computed here per arm, on total harm (the comparable costing), against")
    a("every baseline measured.")
    a("")
    a("| dataset | arm | proposed | best baseline | regret | best-or-tied? |")
    a("|---|---|---:|---:|---:|---|")
    survives = {arm: True for arm in ARMS}
    for ds in DATASETS:
        for arm in ARMS:
            p = pipe[ds][arm]
            proposed = p.total_harm["calibrated_risk_aware"]
            baselines = {
                k: v for k, v in p.total_harm.items()
                if k != "calibrated_risk_aware"
            }
            best_name = min(baselines, key=lambda k: baselines[k])
            regret = proposed - baselines[best_name]
            ok = regret <= 1e-6
            survives[arm] = survives[arm] and ok
            a(
                f"| {ds} | {arm} | {_num(proposed)} | "
                f"{best_name} {_num(baselines[best_name])} | {regret:+,.1f} | "
                f"{'yes' if ok else '**NO**'} |"
            )
    a("")
    for arm in ARMS:
        a(f"- **{arm} profile**: zero worst-case regret "
          f"{'HOLDS' if survives[arm] else 'FAILS'}.")
    a("")

    # ── 6. verdict ───────────────────────────────────────────────────────
    a("## 6. Verdict — is the caveat material?")
    a("")
    a("Filled in from the measurements above; see the summary table.")
    a("")
    a("| dataset | PR-AUC delta | recall delta | repair-acc delta | total-harm delta (proposed) |")
    a("|---|---:|---:|---:|---:|")
    for ds in DATASETS:
        cd, dd = det[ds]["clean"], det[ds]["dirty"]
        cp, pp = pipe[ds]["clean"], pipe[ds]["dirty"]
        ch = cp.total_harm["calibrated_risk_aware"]
        dh = pp.total_harm["calibrated_risk_aware"]
        a(
            f"| {ds} | {_delta(cd.pr_auc, dd.pr_auc)} | "
            f"{_delta(cd.recall, dd.recall)} | "
            f"{pp.top1_accuracy - cp.top1_accuracy:+.4f} | {dh - ch:+,.1f} |"
        )
    a("")
    return "\n".join(L) + "\n"


def main() -> int:
    det: Dict[str, Dict[str, DetectionArm]] = {}
    pipe: Dict[str, Dict[str, PipelineArm]] = {}
    for ds in DATASETS:
        det[ds] = {}
        pipe[ds] = {}
        for arm in ARMS:
            print(f"\n=== {ds} / {arm} profile ===")
            det[ds][arm] = measure_detection(ds, arm)
            d = det[ds][arm]
            print(
                f"  detection  PR-AUC {_iv(d.pr_auc)}  "
                f"P {d.precision.mean:.4f} R {d.recall.mean:.4f} "
                f"F1 {d.f1.mean:.4f}  flagged {d.n_flagged:,} TP {d.n_tp:,}"
            )
            pipe[ds][arm] = measure_pipeline(ds, arm)
            p = pipe[ds][arm]
            print(
                f"  pipeline   decided {p.n_decided:,}  top-1 "
                f"{p.top1_correct:,} ({p.top1_accuracy:.4f})  "
                f"proposed flagged-only {_iv(p.cost.get('calibrated_risk_aware'))}  "
                f"total harm {_num(p.total_harm['calibrated_risk_aware'])}"
            )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(write_report(det, pipe), encoding="utf-8")
    print(f"\nWrote {OUT_MD.relative_to(ROOT)}")

    bad = [
        ds for ds in DATASETS
        if abs(det[ds]["clean"].pr_auc.mean - PUBLISHED_PR_AUC[ds]) >= 5e-4
    ]
    if bad:
        print(
            "\nSELF-VERIFICATION FAILED: the clean arm does not reproduce the "
            f"published PR-AUC for {bad}. The report records the mismatch."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
