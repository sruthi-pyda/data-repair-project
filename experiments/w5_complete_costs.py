"""
experiments/w5_complete_costs.py
--------------------------------
Route W5.1a of docs/research_program.md §11 — reviewer **gap 4a**:
*"undetected corruptions are priced at zero."*

THE PROBLEM, STATED PRECISELY
=============================
Every realised cost this project has published is computed over the cells
the detector FLAGGED.  A genuinely corrupted cell that was never flagged
enters no arm of the objective, so it costs nothing.  On Adult that is
**all 1,701** injected corruptions; the much-quoted "realised cost 0" is
therefore the cost of a problem the pipeline never touched.

docs/cost_semantics.md §3 already says this in words.  This script
measures it.

WHAT IS COMPUTED
================
For every dataset and every policy, three numbers side by side:

    flagged-only cost     what every published table reports
    FN surcharge          C_abstain x (corrupted cells never flagged)
    total harm            the sum

plus two framing quantities:

    detection recall      the FRACTION of total harm any policy on this
                          detector could even address — the ceiling
    do-nothing reference  no detector, no policy: every corruption
                          persists, i.e. n_corrupted x C_abstain.  The
                          only baseline that prices the whole problem,
                          and the one the pipeline has to beat to have
                          done anything at all.

WHY C_abstain IS THE RIGHT CHARGE
=================================
`C_abstain` (config key `c_leave_error`) is corruption-PERSISTENCE cost:
charged when a genuinely corrupted cell is left unrepaired
(docs/cost_semantics.md §1, interpretation (a)).  An undetected
corruption is left unrepaired.  The only difference from an abstention is
that nobody noticed, and the data does not care whether anybody noticed.
So the charge is identical, and no new cost parameter is invented.

Per-flag operational cost is NOT added here: it is decision-invariant and
must be reported separately as `n x C_flag` (docs/cost_semantics.md §2).

THE RANKING QUESTION — ANSWERED IN ADVANCE, THEN MEASURED
=========================================================
The surcharge is a function of the detector's flag SET and of ground
truth.  It does not depend on which action a policy chose, so for a fixed
detector it adds the SAME CONSTANT to every policy and cannot reorder
them.  That is proved as a property in tests/test_complete_costs.py
(exhaustively over the action space) and is re-measured here per dataset
rather than assumed.  The finding is therefore not "the ranking changes"
but:

    the ranking is untouched; the MAGNITUDES and the headline are not.

POLICIES
========
    proposed            the shipped calibrated risk-aware policy
    always-auto         every flagged cell AUTO
    always-human        every flagged cell HUMAN
    always-abstain      every flagged cell ABSTAIN
    conf>=0.50 .. 0.90  AUTO if raw_confidence >= t else HUMAN
    do-nothing          no detector at all (empty record list)

`always-human` and `always-abstain` have no shipped baseline class, so
they are built by replacing the action on the always-auto records —
exactly as experiments/w4_costs.py does.

LEAKAGE (CLAUDE.md §3)
======================
Ground truth enters only after every decision exists: via the Phase 6b
label builder inside `run_pipeline` (the permitted use, for q), and here
for scoring.  The whole-dataset corruption census is pure scoring — it is
computed by `src.evaluation.cost_metrics.corrupted_cells` inside the
evaluator and is never visible to a detector, a repairer or a policy.

Run:
    python experiments/w5_complete_costs.py

Writes results/w5_complete_costs.md.  Deterministic: seed 42, full
datasets, mock repair mode, shipped detector and cost defaults.
"""

from __future__ import annotations

import dataclasses
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.baselines import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    AlwaysAutoBaseline,
    BaselineRecord,
    ConfidenceThresholdBaseline,
    wrap_proposed_method,
)
from src.console import enable_unicode_stdout  # noqa: E402
from src.decision.policy import DecisionAction  # noqa: E402
from src.evaluation.cost_metrics import (  # noqa: E402
    CostEvaluator,
    CostReport,
    corrupted_cells,
)
from src.pipeline.runner import decide_all, run_pipeline  # noqa: E402
from src.risk.cost_model import CostConfig  # noqa: E402

enable_unicode_stdout()

SEED = 42
DATASETS = ["adult", "breast_cancer", "smartfactory"]
COSTS_YAML = ROOT / "configs" / "costs.yaml"
OUT_MD = ROOT / "results" / "w5_complete_costs.md"

DO_NOTHING = "do-nothing (no detector)"
PROPOSED = "proposed (calibrated risk-aware)"

# Flagged-only totals published in results/RESULTS.md §1 and
# results/tables/three_actions_ablation.md.  Hard-coded ONLY as an
# assertion target: if this script cannot reproduce them, the complete-cost
# numbers it adds are not trustworthy either.
PUBLISHED_FLAGGED_ONLY = {
    "adult": {
        PROPOSED: 0.0,
        "always-auto": 58290.0,
        "always-human": 11658.0,
        "always-abstain": 0.0,
        "conf>=0.70": 11658.0,
    },
    "breast_cancer": {
        PROPOSED: 1615.0,
        "always-auto": 5800.0,
        "always-human": 1775.0,
        "always-abstain": 3550.0,
        "conf>=0.70": 1720.0,
    },
    "smartfactory": {
        PROPOSED: 14186.0,
        "always-auto": 126275.0,
        "always-human": 14186.0,
        "always-abstain": 35465.0,
        "conf>=0.70": 14186.0,
    },
}

# Genuine-corruption counts documented in docs/cost_semantics.md §3 — a
# cross-check on the independently computed census, not its source.
DOC_CORRUPTION_COUNTS = {"breast_cancer": 631, "smartfactory": 7093, "adult": 1701}


# ═══════════════════════════════════════════════════════════════════════════════
# Policy construction
# ═══════════════════════════════════════════════════════════════════════════════


def _force_action(
    records: List[BaselineRecord], action: DecisionAction, name: str
) -> List[BaselineRecord]:
    """
    Fixed policy: the same flags, one action for all of them.

    Built by replacing the action on an existing record list, so the flag
    set — which is what the false-negative surcharge depends on — is
    identical to every other policy's by construction.
    """
    return [
        dataclasses.replace(r, method_name=name, action=action) for r in records
    ]


def build_policies(bundle, costs: CostConfig) -> Dict[str, List[BaselineRecord]]:
    """Every policy scored in this experiment, all over the same flag set."""
    idx = bundle.decision_indices()
    unc = [bundle.unc_records[i] for i in idx]
    probs = [bundle.calibrated_probs[i] for i in idx]
    all_decisions = decide_all(bundle, costs)
    decisions = [all_decisions[i] for i in idx]

    auto_records = AlwaysAutoBaseline(bundle.dataset).run(unc)

    policies: Dict[str, List[BaselineRecord]] = {
        PROPOSED: wrap_proposed_method(
            unc, probs, decisions, dataset_name=bundle.dataset
        ),
        "always-auto": auto_records,
        "always-human": _force_action(
            auto_records, DecisionAction.HUMAN_REVIEW, "always-human"
        ),
        "always-abstain": _force_action(
            auto_records, DecisionAction.ABSTAIN, "always-abstain"
        ),
    }
    for t in DEFAULT_THRESHOLDS:
        policies[f"conf>={t:.2f}"] = ConfidenceThresholdBaseline(
            threshold=t, dataset_name=bundle.dataset
        ).run(unc)
    # The reference point with no pipeline at all: nothing flagged, so every
    # corruption in the dataset persists.
    policies[DO_NOTHING] = []
    return policies


# ═══════════════════════════════════════════════════════════════════════════════
# One dataset
# ═══════════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class DatasetResult:
    dataset: str
    costs: CostConfig
    n_decided: int
    n_detected_total: int
    reports_off: Dict[str, CostReport]
    reports_on: Dict[str, CostReport]
    n_total_corrupted: int
    n_flagged_corrupted: int
    n_unflagged_corrupted: int
    surcharge: float
    detection_recall: float
    skip_columns: List[str]
    verification: List[Tuple[str, float, float, bool]]
    # Second census with NO column exclusions: ID-like columns the pipeline is
    # structurally forbidden from deciding on are still corrupted data, and a
    # practitioner's harm includes them.  Reported beside the primary number.
    n_total_corrupted_all: int = 0
    n_unflagged_all: int = 0
    surcharge_all: float = 0.0
    unflagged_by_column: List[Tuple[str, int]] = dataclasses.field(
        default_factory=list
    )

    @property
    def do_nothing_cost(self) -> float:
        return self.reports_on[DO_NOTHING].total_realized_cost

    def ranking(self, mode: str) -> List[str]:
        """Policies ordered by cost (ties broken by name, so it is a function)."""
        reports = self.reports_off if mode == "off" else self.reports_on
        names = [n for n in reports if n != DO_NOTHING]
        return sorted(names, key=lambda n: (reports[n].total_realized_cost, n))


def run_dataset(dataset: str) -> DatasetResult:
    print(f"\n=== {dataset} ===")
    bundle = run_pipeline(dataset, n_records=0, seed=SEED)
    costs = CostConfig.from_yaml(COSTS_YAML, dataset)
    policies = build_policies(bundle, costs)

    evaluator = CostEvaluator.from_cost_config(
        bundle.clean_df, costs, dirty_df=bundle.dirty_df
    )
    skip = list(bundle.skip_columns)

    reports_off: Dict[str, CostReport] = {}
    reports_on: Dict[str, CostReport] = {}
    for name, recs in policies.items():
        reports_off[name] = evaluator.evaluate(recs, dataset=dataset)
        reports_on[name] = evaluator.evaluate(
            recs, dataset=dataset,
            include_false_negatives=True, skip_columns=skip,
        )

    ref = reports_on[PROPOSED]
    n_total = ref.n_total_corrupted_cells
    n_flagged = ref.n_genuinely_corrupted
    n_unflagged = ref.n_unflagged_corrupted
    recall = n_flagged / n_total if n_total else float("nan")

    # Second census, no column exclusions, plus where the misses live.
    census_all = corrupted_cells(bundle.clean_df, bundle.dirty_df)
    flagged_set = {(int(r.row_id), r.column) for r in policies[PROPOSED]}
    unflagged_all = census_all - flagged_set
    by_col = Counter(col for _, col in unflagged_all)

    # ── Self-verification: reproduce the published flagged-only numbers ──
    verification: List[Tuple[str, float, float, bool]] = []
    for name, expected in PUBLISHED_FLAGGED_ONLY.get(dataset, {}).items():
        got = reports_off[name].total_realized_cost
        verification.append((name, expected, got, abs(got - expected) < 1e-6))

    print(
        f"  flags {len(policies[PROPOSED])} decided / {bundle.n_detected_total} "
        f"detected | corrupted cells {n_total} "
        f"(flagged {n_flagged}, unflagged {n_unflagged}) | recall {recall:.4f}"
    )
    print(f"  surcharge = {costs.c_leave_error:g} x {n_unflagged} = {ref.false_negative_surcharge:g}")
    for name, expected, got, ok in verification:
        print(f"  check {name:<32} published {expected:>10.1f} measured {got:>10.1f} "
              f"{'OK' if ok else 'MISMATCH'}")

    return DatasetResult(
        dataset=dataset,
        costs=costs,
        n_decided=len(policies[PROPOSED]),
        n_detected_total=bundle.n_detected_total,
        reports_off=reports_off,
        reports_on=reports_on,
        n_total_corrupted=n_total,
        n_flagged_corrupted=n_flagged,
        n_unflagged_corrupted=n_unflagged,
        surcharge=ref.false_negative_surcharge,
        detection_recall=recall,
        skip_columns=skip,
        verification=verification,
        n_total_corrupted_all=len(census_all),
        n_unflagged_all=len(unflagged_all),
        surcharge_all=costs.c_leave_error * len(unflagged_all),
        unflagged_by_column=by_col.most_common(),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════


def _fmt(x: float) -> str:
    return f"{x:,.0f}" if abs(x - round(x)) < 1e-9 else f"{x:,.2f}"


def write_report(results: List[DatasetResult]) -> str:
    L: List[str] = []
    a = L.append

    a("# W5.1a — complete costs: what undetected corruption actually costs")
    a("")
    a("Route W5.1a of `docs/research_program.md` §11, reviewer **gap 4a**.")
    a("Generated by `experiments/w5_complete_costs.py`. Seed 42, full datasets,")
    a("mock repair mode, shipped detector defaults, costs from")
    a("`configs/costs.yaml` (abstract experimental penalty units — never money).")
    a("")
    a("## 0. What changed and what did not")
    a("")
    a("`CostEvaluator.evaluate(..., include_false_negatives=True)` adds")
    a("")
    a("```")
    a("surcharge = C_abstain x |{genuinely corrupted cells} - {flagged cells}|")
    a("```")
    a("")
    a("`C_abstain` (`c_leave_error`) is corruption-persistence cost")
    a("(`docs/cost_semantics.md` §1). An undetected corruption is an unrepaired")
    a("error; the only difference from an abstention is that nobody noticed, and")
    a("the data does not care. So no new cost parameter is invented.")
    a("")
    a("The mode defaults to **off**, so every previously published number is")
    a("unchanged bit-for-bit (asserted in `tests/test_complete_costs.py`, and")
    a("re-verified against `results/RESULTS.md` §1 in §4 below).")
    a("")

    # ── 1. The census ────────────────────────────────────────────────────
    a("## 1. How much of the problem the detector can even see")
    a("")
    a("| dataset | corrupted cells | flagged | flagged & corrupted | **detection recall** | unflagged corrupted | C_abstain | **surcharge** |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        a(
            f"| {r.dataset} | {r.n_total_corrupted:,} | {r.n_decided:,} | "
            f"{r.n_flagged_corrupted:,} | **{r.detection_recall:.4f}** | "
            f"{r.n_unflagged_corrupted:,} | {r.costs.c_leave_error:g} | "
            f"**{_fmt(r.surcharge)}** |"
        )
    a("")
    a("Detection recall is the **ceiling on what any decision policy can")
    a("address**, however well calibrated. The surcharge is the part of total")
    a("harm that no policy in this project touches.")
    a("")
    for r in results:
        doc = DOC_CORRUPTION_COUNTS.get(r.dataset)
        note = ""
        if doc is not None:
            note = (
                f" (docs/cost_semantics.md §3 quotes {doc:,}"
                + ("; agrees" if doc == r.n_total_corrupted
                   else f"; measured here {r.n_total_corrupted:,}")
                + ")"
            )
        skip = f", excluding ID-like columns {r.skip_columns}" if r.skip_columns else ""
        a(f"- **{r.dataset}**: {r.n_total_corrupted:,} corrupted cells{skip}{note}.")
    a("")
    a("### 1b. The same census with NO column exclusions")
    a("")
    a("The primary census above excludes the ID-like columns the pipeline is")
    a("structurally forbidden from deciding on (`PipelineBundle.skip_columns`):")
    a("charging the policy for a cell it may not touch would be unfair. A")
    a("practitioner's data-quality harm nevertheless includes them, so both are")
    a("reported. Only breast_cancer has any such columns.")
    a("")
    a("| dataset | corrupted (excl. ID) | surcharge | corrupted (all columns) | surcharge |")
    a("|---|---:|---:|---:|---:|")
    for r in results:
        a(
            f"| {r.dataset} | {r.n_total_corrupted:,} | {_fmt(r.surcharge)} | "
            f"{r.n_total_corrupted_all:,} | {_fmt(r.surcharge_all)} |"
        )
    a("")
    for r in results:
        if r.n_total_corrupted_all != r.n_total_corrupted:
            extra = r.n_total_corrupted_all - r.n_total_corrupted
            a(
                f"- **{r.dataset}**: {extra} of the {r.n_total_corrupted_all:,} "
                f"corruptions sit in {r.skip_columns}. Including them raises the "
                f"surcharge from {_fmt(r.surcharge)} to {_fmt(r.surcharge_all)} "
                f"({r.n_unflagged_all:,} unflagged cells)."
            )
    a("")
    a("### 1c. Where the undetected corruption lives")
    a("")
    a("All columns, so the ID-like ones are visible too.")
    a("")
    for r in results:
        if not r.unflagged_by_column:
            a(f"- **{r.dataset}**: no undetected corruption.")
            continue
        top = ", ".join(f"`{c}` {n}" for c, n in r.unflagged_by_column[:6])
        more = (
            f" (+{len(r.unflagged_by_column) - 6} more columns)"
            if len(r.unflagged_by_column) > 6 else ""
        )
        a(f"- **{r.dataset}** ({r.n_unflagged_all:,} missed): {top}{more}.")
    a("")

    # ── 2. The three-column table per dataset ────────────────────────────
    a("## 2. Flagged-only cost / surcharge / total harm, per policy")
    a("")
    a("All values in experimental penalty units. The surcharge column is")
    a("**constant down each table** — see §3.")
    a("")
    for r in results:
        c = r.costs
        a(f"### {r.dataset} (C_wrong={c.c_wrong_repair:g}, "
          f"C_human={c.c_human_review:g}, C_abstain={c.c_leave_error:g})")
        a("")
        a("| policy | auto | human | abstain | flagged-only | + FN surcharge | **total harm** | vs do-nothing |")
        a("|---|---:|---:|---:|---:|---:|---:|---:|")
        order = r.ranking("on")
        dn = r.do_nothing_cost
        for name in order:
            off = r.reports_off[name]
            on = r.reports_on[name]
            delta = on.total_realized_cost - dn
            sign = "+" if delta > 0 else ""
            a(
                f"| {name} | {off.n_auto:,} | {off.n_human:,} | {off.n_abstain:,} | "
                f"{_fmt(off.total_realized_cost)} | {_fmt(on.false_negative_surcharge)} | "
                f"**{_fmt(on.total_realized_cost)}** | {sign}{_fmt(delta)} |"
            )
        dn_rep = r.reports_on[DO_NOTHING]
        a(
            f"| _{DO_NOTHING}_ | 0 | 0 | 0 | 0 | {_fmt(dn_rep.false_negative_surcharge)} | "
            f"**{_fmt(dn)}** | 0 |"
        )
        a("")

    # ── 3. Ranking ───────────────────────────────────────────────────────
    a("## 3. Does the policy ranking change? **No — and it cannot.**")
    a("")
    a("The surcharge depends on the detector's flag *set* and on ground truth,")
    a("not on which action the policy chose. For a fixed detector it therefore")
    a("adds the same constant to every policy, which cannot reorder them. This")
    a("is proved exhaustively over the action space in")
    a("`tests/test_complete_costs.py::TestSurchargeIsPolicyInvariant` and")
    a("re-measured per dataset here:")
    a("")
    a("| dataset | surcharge identical across all policies? | ranking identical? | top policy (both modes) |")
    a("|---|---|---|---|")
    for r in results:
        charges = {
            round(rep.false_negative_surcharge, 6)
            for name, rep in r.reports_on.items() if name != DO_NOTHING
        }
        same_charge = len(charges) == 1
        off_order = r.ranking("off")
        on_order = r.ranking("on")
        best = r.reports_on[on_order[0]].total_realized_cost
        tied = [
            n for n in on_order
            if abs(r.reports_on[n].total_realized_cost - best) < 1e-6
        ]
        top = ", ".join(f"`{n}`" for n in tied)
        if len(tied) > 1:
            top += f" — **tied** at {_fmt(best)}"
        a(
            f"| {r.dataset} | {'yes' if same_charge else 'NO'} "
            f"({', '.join(_fmt(c) for c in sorted(charges))}) | "
            f"{'yes' if off_order == on_order else 'NO'} | "
            f"{top} |"
        )
    a("")
    a("So the honest statement is **not** \"pricing false negatives changes which")
    a("policy wins\". It is:")
    a("")
    a("> The zero-regret ranking survives intact, because the omission flattered")
    a("> every policy by exactly the same amount. What does not survive is the")
    a("> *magnitude* of the claim.")
    a("")
    for r in results:
        prop_off = r.reports_off[PROPOSED].total_realized_cost
        prop_on = r.reports_on[PROPOSED].total_realized_cost
        dn = r.do_nothing_cost
        if dn > 0:
            red = 100.0 * (1.0 - prop_on / dn)
            a(
                f"- **{r.dataset}**: the proposed policy's cost goes "
                f"{_fmt(prop_off)} → **{_fmt(prop_on)}**. Against the only "
                f"baseline that prices the whole problem (do-nothing, "
                f"{_fmt(dn)}) that is a **{red:.1f}% reduction in total harm**."
            )
        else:
            a(f"- **{r.dataset}**: do-nothing costs 0, so no reduction is defined.")
    a("")

    # ── 4. Verification ──────────────────────────────────────────────────
    a("## 4. Self-verification — the flagged-only numbers are unchanged")
    a("")
    a("Every flagged-only total below is re-measured by this script and")
    a("compared with the published value in `results/RESULTS.md` §1.")
    a("")
    a("| dataset | policy | published | measured | |")
    a("|---|---|---:|---:|---|")
    all_ok = True
    for r in results:
        for name, expected, got, ok in r.verification:
            all_ok = all_ok and ok
            a(f"| {r.dataset} | {name} | {_fmt(expected)} | {_fmt(got)} | "
              f"{'OK' if ok else '**MISMATCH**'} |")
    a("")
    a(f"**All checks pass: {all_ok}.**")
    a("")
    a("### 4b. A derived identity the two modes must satisfy")
    a("")
    a("`always-abstain` leaves every *flagged* corruption in place, and the")
    a("surcharge leaves every *unflagged* one in place. Its total harm must")
    a("therefore equal `n_corrupted x C_abstain` exactly — which is also the")
    a("do-nothing cost. Nothing in the implementation enforces this; it is an")
    a("independent re-derivation of the same quantity by two routes.")
    a("")
    a("| dataset | always-abstain total | n_corrupted x C_abstain | do-nothing | identical? |")
    a("|---|---:|---:|---:|---|")
    identity_ok = True
    for r in results:
        abst = r.reports_on["always-abstain"].total_realized_cost
        derived = r.n_total_corrupted * r.costs.c_leave_error
        dn = r.do_nothing_cost
        ok = abs(abst - derived) < 1e-6 and abs(dn - derived) < 1e-6
        identity_ok = identity_ok and ok
        a(
            f"| {r.dataset} | {_fmt(abst)} | {_fmt(derived)} | {_fmt(dn)} | "
            f"{'yes' if ok else '**NO**'} |"
        )
    a("")
    a(f"**Identity holds on every dataset: {identity_ok}.**")
    a("")

    # ── 5. What must go in the paper ─────────────────────────────────────
    a("## 5. What the paper must now say")
    a("")
    adult = next((r for r in results if r.dataset == "adult"), None)
    if adult is not None:
        a(f"1. **Adult's \"0\" is {_fmt(adult.do_nothing_cost)}.** The flagged-only")
        a(f"   cost is genuinely 0, and that result stands: all "
          f"{adult.n_decided:,} flags are false alarms "
          f"({adult.n_flagged_corrupted} of them genuinely corrupted), and the")
        a("   policy correctly declines to act on every one of them.")
        a(f"   But {adult.n_unflagged_corrupted:,} genuine corruptions were never")
        a(f"   detected, so total harm is {adult.n_unflagged_corrupted:,} x "
          f"{adult.costs.c_leave_error:g} = **{_fmt(adult.do_nothing_cost)}**, which is")
        a("   *exactly* the do-nothing cost. On Adult the pipeline reduces total")
        a("   data-quality harm by **0%**. The flagged-only win is a win about")
        a("   not wasting review effort on false alarms, and nothing more.")
    a("2. **Both numbers must always appear together.** Flagged-only answers")
    a("   \"was the decision right?\"; total harm answers \"is the data better?\".")
    a("   Reporting only the first is what gap 4a objects to.")
    a("3. **Detection recall is the ceiling**, and it is the quantity the")
    a("   research program's wave 1–3 work attacks. The decision layer cannot")
    a("   improve it by any amount of calibration.")
    a("4. **The ranking result is unaffected**, and saying so is stronger than")
    a("   hiding the surcharge: the omission was uniform, not selective.")
    a("")
    return "\n".join(L) + "\n"


def main() -> int:
    results = [run_dataset(d) for d in DATASETS]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    text = write_report(results)
    OUT_MD.write_text(text, encoding="utf-8")
    print(f"\nWrote {OUT_MD.relative_to(ROOT)}")

    print("\n" + "=" * 78)
    print("  FLAGGED-ONLY / SURCHARGE / TOTAL HARM")
    print("=" * 78)
    print(f"  {'dataset':<15}{'policy':<34}{'flagged':>10}{'surch.':>10}{'total':>11}")
    for r in results:
        for name in r.ranking("on"):
            off = r.reports_off[name]
            on = r.reports_on[name]
            print(
                f"  {r.dataset:<15}{name:<34}"
                f"{off.total_realized_cost:>10,.0f}"
                f"{on.false_negative_surcharge:>10,.0f}"
                f"{on.total_realized_cost:>11,.0f}"
            )
        print(
            f"  {r.dataset:<15}{DO_NOTHING:<34}{0:>10,.0f}"
            f"{r.do_nothing_cost:>10,.0f}{r.do_nothing_cost:>11,.0f}"
        )
    print("=" * 78)

    ok = all(v[3] for r in results for v in r.verification)
    if not ok:
        print("\nSELF-VERIFICATION FAILED: a flagged-only total does not match "
              "results/RESULTS.md. The report records the mismatch.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
