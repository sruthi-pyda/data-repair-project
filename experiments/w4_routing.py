"""
experiments/w4_routing.py
--------------------------
Wave 4, route W4.4: **difficulty-aware repair routing** (mechanism M1).

THE HYPOTHESIS, AND IT COMES FROM OUR OWN MEASUREMENT
=====================================================
`docs/PROJECT_REPORT.md` §3.5, controlled comparison on 80 real Breast Cancer
cells:

    corruption                        rule-based repairer      LLM
    recoverable (out-of-range/format)       100 %             92.5 %
    destroyed (missing)                      45 %             50 %

The model is **worse where the answer is determined** and marginally better
where it is a guess. So stop sending everything to one repairer: route
deterministic cells to the cheap rule and ambiguous ones to the model. If the
difficulty signal were perfect the ceiling is ~100 % on recoverable + ~50 % on
destroyed, which beats *both* pure strategies.

WHAT THIS SCRIPT MEASURES
=========================
Every estimator in `src/repair/routing.py`, on the **same cells**, with the
same rule repairer and the same model repairer:

  1. top-1 repair accuracy, overall and **split by corruption type** (the
     split is the whole hypothesis; an overall number alone cannot falsify it);
  2. model invocations — routing should cut the API bill even if it does not
     move accuracy, and a mechanism that halves the bill for free is a result;
  3. the realised **policy cost** when that repairer feeds the full pipeline,
     because this is a decision project and accuracy is not the objective.

Included by requirement: the two pure controls (`always_rule`, `always_llm`)
and the `oracle` ceiling. The oracle reads ground truth, is an **upper bound
only**, and is labelled as such in every table it appears in.

DATASET
=======
Breast Cancer only. It is the one dataset where repair meaningfully works:
Smart Factory's corruption is 71 % destroyed and Adult's true corruption is
label flips the detector never finds (`docs/PROJECT_REPORT.md` §2, §3.3). A
routing table on either would measure the dataset, not the mechanism.

LEAKAGE (CLAUDE.md §3)
======================
The router and every estimator except `oracle` see dirty data only. Clean
values enter this script in exactly two places, both **after** predictions
exist: the accuracy evaluation, and the `oracle` ceiling's ground-truth map.
`tests/test_routing.py` asserts routes are bit-identical when the clean values
are scrambled.

The realised-cost column is produced by handing the routed repairer to
`src/pipeline/runner.run_pipeline` through its own `build_repairer` seam. The
seam is rebound for the duration of one call and restored immediately; no file
is modified and `run_pipeline`'s own leakage boundary is untouched.

MODES
=====
    python experiments/w4_routing.py --fake     # canned responses, no network
    python experiments/w4_routing.py --live     # real calls (cache-backed)

`--fake` is the CI/default path: a deterministic canned transport, so the run
is free and reproducible. Its accuracy numbers are a **mechanism smoke test,
not a measurement**, and the generated report says so on every table.

`--live` uses `LLMConfig.from_env()` and the on-disk response cache, so cells
already answered by the earlier 392-call Breast Cancer run cost nothing. The
cache is shared across estimators within one run, which means the *second*
estimator to visit a cell never pays for it — so `llm_invocations` (what a
deployment would spend) is reported separately from the transport's own
`llm_calls` (what this run spent).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.console import enable_unicode_stdout  # noqa: E402
from src.data.loader import DataLoader  # noqa: E402
from src.evaluation.repair_metrics import ValueComparator  # noqa: E402
from src.pipeline import runner  # noqa: E402
from src.repair.llm_repair import DiskCache, LLMConfig, LLMRepairer  # noqa: E402
from src.repair.mock_repair import MockRepairer  # noqa: E402
from src.repair.routing import (  # noqa: E402
    AMBIGUOUS_ERROR_TYPES,
    ESTIMATOR_NAMES,
    RECOVERABLE_ERROR_TYPES,
    ROUTE_RULE,
    LearnedErrorTypeEstimator,
    OracleEstimator,
    RoutingRepairer,
    fit_error_type_routes,
    make_estimator,
)
from src.risk.cost_model import CostConfig  # noqa: E402

DATASET = "breast_cancer"
DATA_ROOT = "datasets"
COSTS_YAML = "configs/costs.yaml"
OUT_MD = Path("results/w4_routing.md")
SEED = 42
TOP_K = 3
THRESHOLD = 0.8

#: Reported estimator order: pure controls first, then the real candidates,
#: then the ceiling.
ORDER: Tuple[str, ...] = (
    "always_rule",
    "always_llm",
    "by_error_type",
    "by_learned_error_type",
    "by_candidate_agreement",
    "by_candidate_margin",
    "by_context_availability",
    "oracle",
)

#: Hold-out fraction for fitting the learned route table. Matches
#: ``runner.DEFAULT_EVAL_FRACTION``, so the learned router is split the same
#: way the project's calibrators are.
EVAL_FRACTION = 0.30


# ═══════════════════════════════════════════════════════════════════════════════
# Canned transport for --fake  (no network, deterministic)
# ═══════════════════════════════════════════════════════════════════════════════


class CannedTransport:
    """
    A deterministic stand-in for a provider. **Never used by `--live`.**

    It reads the dirty value, the error type and the column statistics back out
    of the prompt text and answers with a plausible candidate list. It is not a
    model and it is not calibrated against one: its only job is to let the
    whole table be produced in CI without a key and without a socket, so the
    plumbing can be verified. Accuracy produced this way is labelled
    "mechanism smoke test" in the report and must never be quoted as a result.
    """

    name = "canned"

    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def _field(text: str, label: str) -> Optional[str]:
        match = re.search(rf"^{re.escape(label)}\s*:\s*(.*)$", text, re.MULTILINE)
        return match.group(1).strip() if match else None

    @staticmethod
    def _stat(text: str, name: str) -> Optional[float]:
        match = re.search(rf"{name}=([-\d.]+)", text)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _vocabulary(text: str) -> List[str]:
        """The prompt's 'Known valid values' list, or empty."""
        match = re.search(r"^Known valid values[^:]*:\s*\[(.*)\]\s*$", text, re.MULTILINE)
        if not match:
            return []
        return [part.strip() for part in match.group(1).split(",") if part.strip()]

    def chat_completion(self, payload: Dict[str, Any], timeout_s: float) -> str:
        self.calls += 1
        text = payload["messages"][-1]["content"]
        error_type = self._field(text, "Error type") or ""
        dirty = self._field(text, "Dirty value") or ""
        median = self._stat(text, "median")
        q25 = self._stat(text, "Q25")
        q75 = self._stat(text, "Q75")
        vmax = self._stat(text, "max")

        digits = re.sub(r"[^0-9]", "", dirty)
        values: List[Any] = []

        if error_type == "OUT_OF_RANGE" and digits:
            raw = float(digits)
            squashed = raw / 11.0
            if vmax is not None and raw > vmax and abs(squashed - round(squashed)) < 1e-9:
                values.append(int(round(squashed)))
            elif vmax is not None:
                values.append(int(round(min(raw, vmax))))
        elif error_type == "FORMAT_ERROR" and digits:
            values.append(int(digits))

        for stat in (median, q25, q75):
            if stat is not None:
                candidate = int(round(stat))
                if candidate not in values:
                    values.append(candidate)

        # Columns with no numerical statistics (categorical, or an ID-like
        # column the pipeline later skips) fall back to the dirty column's own
        # vocabulary, which is what the prompt offers and all a model would
        # have. Without this the canned answer would be the unparseable dirty
        # value itself and every such cell would show as a fallback.
        if not values:
            values = [v for v in self._vocabulary(text)[:3]]
        if not values:
            values = [digits] if digits else ["unknown"]

        probs = [0.6, 0.25, 0.15][: len(values)]
        total = sum(probs)
        return json.dumps(
            {
                "candidates": [
                    {"value": v, "probability": p / total}
                    for v, p in zip(values[: len(probs)], probs)
                ],
                "reasoning": "canned deterministic response (no model was called)",
            }
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Rows
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class EstimatorRow:
    """One estimator's measured outcome on the shared cell set."""

    name: str
    n_cells: int                                   # decided cells (policy set)
    n_cells_flagged: int                           # every flagged cell repaired
    accuracy: float
    accuracy_by_type: Dict[str, Tuple[int, int]]   # type -> (hits, n)
    accuracy_recoverable: Optional[float]
    accuracy_destroyed: Optional[float]
    n_rule: int
    n_llm: int
    llm_invocations: int                           # on the decided cells
    llm_invocations_all_cells: int
    llm_failures: int
    transport_calls: Optional[int]
    cache_hits: Optional[int]
    policy_cost: Optional[float]
    always_auto_cost: Optional[float]
    n_auto: int = 0
    n_human: int = 0
    n_abstain: int = 0
    uses_ground_truth: bool = False
    reasons: Dict[str, int] = field(default_factory=dict)
    per_cell: List[Scored] = field(default_factory=list)
    description: str = ""
    eval_accuracy: Optional[float] = None
    seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimator": self.name,
            "n_cells_decided": self.n_cells,
            "n_cells_flagged": self.n_cells_flagged,
            "accuracy": self.accuracy,
            "accuracy_by_type": {
                k: {"hits": h, "n": n, "accuracy": (h / n) if n else None}
                for k, (h, n) in sorted(self.accuracy_by_type.items())
            },
            "accuracy_recoverable": self.accuracy_recoverable,
            "accuracy_destroyed": self.accuracy_destroyed,
            "n_rule": self.n_rule,
            "n_llm": self.n_llm,
            "llm_invocations": self.llm_invocations,
            "llm_invocations_all_cells": self.llm_invocations_all_cells,
            "llm_failures": self.llm_failures,
            "transport_calls": self.transport_calls,
            "cache_hits": self.cache_hits,
            "policy_cost": self.policy_cost,
            "always_auto_cost": self.always_auto_cost,
            "n_auto": self.n_auto,
            "n_human": self.n_human,
            "n_abstain": self.n_abstain,
            "uses_ground_truth": self.uses_ground_truth,
            "reasons": dict(self.reasons),
            "description": self.description,
            "eval_accuracy": self.eval_accuracy,
            "seconds": round(self.seconds, 1),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Ground truth (evaluation side only)
# ═══════════════════════════════════════════════════════════════════════════════


def build_ground_truth(dataset: str, data_root: str) -> Dict[Tuple[int, str], Any]:
    """
    ``(row_id, column) -> clean value`` for the whole dataset.

    LEAKAGE: this is evaluation-side data. It is used to score predictions
    after they exist, and to construct the `oracle` ceiling — which is reported
    as an upper bound, never as a method. Nothing here reaches a detector, a
    repairer, the uncertainty estimator or a decision.
    """
    clean_df, _ = DataLoader(data_root).load(dataset)
    truth: Dict[Tuple[int, str], Any] = {}
    for column in clean_df.columns:
        series = clean_df[column]
        for row_id, value in enumerate(series.to_numpy()):
            truth[(row_id, str(column))] = value
    return truth


def classify(error_type: str) -> str:
    """'recoverable' / 'destroyed' / 'other' — the §2 partition."""
    if error_type in RECOVERABLE_ERROR_TYPES:
        return "recoverable"
    if error_type in AMBIGUOUS_ERROR_TYPES:
        return "destroyed"
    return "other"


#: One scored cell: ((row_id, column), error_type, top-1 correct).
Scored = Tuple[Tuple[int, str], str, bool]


def score_results(
    repair_results: Sequence[Any],
    truth: Dict[Tuple[int, str], Any],
    comparator: ValueComparator,
) -> Tuple[float, Dict[str, Tuple[int, int]], Dict[str, Tuple[int, int]], List[Scored]]:
    """
    Top-1 accuracy overall, per error type, per corruption class, and per cell.

    A cell with no clean value available is skipped entirely rather than
    counted as a miss, so a missing label can never flatter or punish a
    repairer. The per-cell list is what lets the report recompute accuracy on a
    hold-out subset without re-running anything.
    """
    per_type: Dict[str, List[int]] = {}
    per_class: Dict[str, List[int]] = {}
    per_cell: List[Scored] = []
    hits = total = 0
    for result in repair_results:
        key = (int(result.row_id), str(result.column))
        if key not in truth:
            continue
        target = truth[key]
        ok = bool(
            result.candidates
            and comparator.are_equal(result.top_candidate.value, target)
        )
        total += 1
        hits += int(ok)
        per_cell.append((key, str(result.error_type), ok))
        bucket = per_type.setdefault(str(result.error_type), [0, 0])
        bucket[0] += int(ok)
        bucket[1] += 1
        cls = per_class.setdefault(classify(str(result.error_type)), [0, 0])
        cls[0] += int(ok)
        cls[1] += 1
    accuracy = (hits / total) if total else 0.0
    return (
        accuracy,
        {k: (v[0], v[1]) for k, v in per_type.items()},
        {k: (v[0], v[1]) for k, v in per_class.items()},
        per_cell,
    )


def split_cells(
    keys: Sequence[Tuple[int, str]],
    eval_fraction: float = EVAL_FRACTION,
    seed: int = SEED,
) -> Tuple[List[int], List[int]]:
    """
    Deterministic train/eval index split over the scored cells.

    The learned route table is fitted on the train indices and reported on the
    eval indices, so no cell is routed using its own label. Same hold-out
    fraction and seed discipline as the Phase 6 calibrators.
    """
    import numpy as np

    order = np.random.default_rng(seed).permutation(len(keys))
    n_eval = int(round(eval_fraction * len(keys)))
    eval_idx = sorted(int(i) for i in order[:n_eval])
    eval_set = set(eval_idx)
    train_idx = [i for i in range(len(keys)) if i not in eval_set]
    return train_idx, eval_idx


def subset_accuracy(
    per_cell: Sequence[Scored], subset: Sequence[Tuple[int, str]]
) -> Optional[float]:
    """Top-1 accuracy restricted to `subset`; None when the subset is empty."""
    wanted = set(subset)
    scored = [ok for key, _, ok in per_cell if key in wanted]
    if not scored:
        return None
    return sum(scored) / len(scored)


# ═══════════════════════════════════════════════════════════════════════════════
# One estimator, end to end
# ═══════════════════════════════════════════════════════════════════════════════


def make_llm_repairer(
    mode: str,
    cache: DiskCache,
    config: Optional[LLMConfig],
    canned: Optional[CannedTransport],
) -> LLMRepairer:
    """
    Build a fresh model repairer sharing the run's cache and transport.

    A fresh repairer per estimator keeps the counters per estimator; the shared
    cache means a cell answered for one estimator is free for the next, which
    is why the report distinguishes ``llm_invocations`` (deployment cost) from
    ``transport_calls`` (this run's cost).
    """
    if mode == "fake":
        assert canned is not None
        return LLMRepairer(
            config=config,
            client=canned,
            fallback=MockRepairer(seed=SEED, top_k=TOP_K),
            top_k=TOP_K,
            seed=SEED,
            cache=cache,
        )
    return LLMRepairer(
        config=config,
        fallback=MockRepairer(seed=SEED, top_k=TOP_K),
        top_k=TOP_K,
        seed=SEED,
        cache=cache,
    )


def run_estimator(
    estimator: Any,
    mode: str,
    cache: DiskCache,
    config: Optional[LLMConfig],
    canned: Optional[CannedTransport],
    truth: Dict[Tuple[int, str], Any],
    costs: CostConfig,
    comparator: ValueComparator,
    name: Optional[str] = None,
) -> EstimatorRow:
    """
    Run the full pipeline with this estimator's router and score it.

    The router is injected through ``runner.build_repairer``, the seam the
    pipeline already uses to choose a Phase 4 repairer. It is rebound for the
    duration of one ``run_pipeline`` call and restored in a ``finally`` block,
    so a failure cannot leave the module patched for the next estimator.
    """
    started = time.time()
    holder: Dict[str, RoutingRepairer] = {}
    name = name or estimator.name

    needs_llm = bool(estimator.can_escalate or estimator.needs_both_results)

    def _factory(mode_arg="mock", seed=SEED, top_k=TOP_K, llm_config=None, llm_client=None):
        llm = (
            make_llm_repairer(mode, cache, config, canned) if needs_llm else None
        )
        router = RoutingRepairer(
            estimator=estimator,
            rule=MockRepairer(seed=seed, top_k=top_k),
            llm=llm,
            seed=seed,
            top_k=top_k,
        )
        holder["router"] = router
        return router

    original = runner.build_repairer
    runner.build_repairer = _factory
    try:
        bundle = runner.run_pipeline(
            DATASET,
            n_records=0,
            seed=SEED,
            top_k=TOP_K,
            data_root=DATA_ROOT,
            mode="mock",
        )
    finally:
        runner.build_repairer = original

    router = holder["router"]

    # Score the cells the policy actually acts on: ``decision_indices()`` drops
    # the two ID-like Breast Cancer columns ("Unnamed: 0", "Sample code
    # number"), which no repairer can do anything with and which the pipeline
    # already excludes from every decision and every cost. Reporting accuracy
    # on the same subset as the cost keeps the two columns of the table about
    # the same cells, and makes the accuracy directly comparable with the
    # 0.6732 published in docs/PROJECT_REPORT.md §3.5.
    decided = [bundle.repair_results[i] for i in bundle.decision_indices()]
    accuracy, by_type, by_class, per_cell = score_results(decided, truth, comparator)

    n_rule = sum(1 for r in decided if r.context_used.get("route") == "rule")
    n_llm = sum(1 for r in decided if r.context_used.get("route") == "llm")
    n_failed = sum(
        1 for r in decided
        if str(r.context_used.get("route_reason", "")).endswith("+llm_failed")
    )
    # What a deployment would spend on this subset. The oracle needs both
    # answers before it can choose, so it pays for every cell.
    if estimator.needs_both_results:
        invocations = len(decided)
    else:
        invocations = n_llm + n_failed

    evaluation = runner.evaluate_policies(bundle, costs, threshold=THRESHOLD)
    proposed = evaluation.cost_reports.get(evaluation.proposed_name)
    always_auto = evaluation.cost_reports.get("always_auto")

    llm = router.llm
    transport_calls = cache_hits = None
    if isinstance(llm, LLMRepairer):
        transport_calls = llm.stats.llm_calls
        cache_hits = llm.stats.cache_hits

    def _fraction(bucket: Optional[Tuple[int, int]]) -> Optional[float]:
        if not bucket or bucket[1] == 0:
            return None
        return bucket[0] / bucket[1]

    return EstimatorRow(
        name=name,
        n_cells=len(decided),
        n_cells_flagged=router.stats.cells,
        accuracy=accuracy,
        accuracy_by_type=by_type,
        accuracy_recoverable=_fraction(by_class.get("recoverable")),
        accuracy_destroyed=_fraction(by_class.get("destroyed")),
        n_rule=n_rule,
        n_llm=n_llm,
        llm_invocations=invocations,
        llm_invocations_all_cells=router.stats.llm_invocations,
        llm_failures=n_failed,
        transport_calls=transport_calls,
        cache_hits=cache_hits,
        policy_cost=proposed.total_realized_cost if proposed else None,
        always_auto_cost=always_auto.total_realized_cost if always_auto else None,
        n_auto=proposed.n_auto if proposed else 0,
        n_human=proposed.n_human if proposed else 0,
        n_abstain=proposed.n_abstain if proposed else 0,
        uses_ground_truth=bool(estimator.uses_ground_truth),
        reasons=dict(router.stats.reasons),
        per_cell=per_cell,
        description=estimator.describe(),
        seconds=time.time() - started,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════


def _pct(value: Optional[float]) -> str:
    """Percentage, or an em dash. Never 0.0 for an undefined quantity."""
    if value is None or not math.isfinite(value):
        return "—"
    return f"{100.0 * value:.1f}%"


def _num(value: Optional[float], places: int = 0) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:,.{places}f}"


def _label(row: EstimatorRow) -> str:
    if row.uses_ground_truth:
        return f"**{row.name}** ‡"
    if row.name in ("always_rule", "always_llm"):
        return f"{row.name} (control)"
    return row.name


def build_report(
    rows: List[EstimatorRow],
    mode: str,
    costs: CostConfig,
    model: str,
    corruption_mix: Dict[str, int],
    live_note: str,
    route_table: Optional[Dict[str, str]] = None,
    learned_note: str = "",
    n_eval: int = 0,
) -> str:
    by_name = {r.name: r for r in rows}
    pure_rule = by_name.get("always_rule")
    pure_llm = by_name.get("always_llm")
    candidates = [r for r in rows if not r.uses_ground_truth
                  and r.name not in ("always_rule", "always_llm")]

    def _held(row: Optional[EstimatorRow]) -> Optional[float]:
        return row.eval_accuracy if row is not None else None

    scored = [r for r in candidates if _held(r) is not None]
    best = max(scored, key=lambda r: r.eval_accuracy) if scored else (
        max(candidates, key=lambda r: r.accuracy) if candidates else None
    )
    best_held = _held(best)
    rule_held, llm_held = _held(pure_rule), _held(pure_llm)
    beat_both = bool(
        best_held is not None and rule_held is not None and llm_held is not None
        and best_held > rule_held and best_held > llm_held
    )
    cheapest = min(
        (r for r in rows if r.policy_cost is not None and not r.uses_ground_truth),
        key=lambda r: r.policy_cost,
        default=None,
    )

    lines: List[str] = []
    lines.append("# W4.4 — difficulty-aware repair routing (mechanism M1)")
    lines.append("")
    n_decided = rows[0].n_cells if rows else 0
    n_flagged = rows[0].n_cells_flagged if rows else 0
    lines.append(
        f"Generated by `python experiments/w4_routing.py --{mode}`. "
        f"Dataset **{DATASET}**, full data, seed {SEED}, top-k {TOP_K}. "
        f"Costs from `{COSTS_YAML}`: wrong={costs.c_wrong_repair:g}, "
        f"human={costs.c_human_review:g}, abstain={costs.c_leave_error:g} "
        f"(abstract penalty units, never monetary)."
    )
    lines.append("")
    lines.append(
        f"The detector flags **{n_flagged}** cells; every table below is over "
        f"the **{n_decided}** the policy acts on. The difference is the two "
        f"ID-like columns (`Unnamed: 0`, `Sample code number`) that "
        f"`src/pipeline/runner.py` excludes from every decision and every "
        f"cost — so accuracy and cost here describe the same cells, and the "
        f"accuracy is directly comparable with the 0.6732 in "
        f"`docs/PROJECT_REPORT.md` §3.5."
    )
    lines.append("")
    if mode == "fake":
        lines.append(
            "> **This run used `--fake`: a canned deterministic transport, not a "
            "model.** The accuracy columns below are a *mechanism smoke test* "
            "— they prove the routing plumbing works end to end without a "
            "network. They are **not** a measurement of any LLM and must not "
            "be quoted as one. The real table comes from `--live`."
        )
    else:
        lines.append(f"> Model: `{model}`. {live_note}")
    lines.append("")

    lines.append("## Corruption mix of the flagged cells")
    lines.append("")
    lines.append("| error type | class | cells |")
    lines.append("|---|---|---:|")
    for error_type, n in sorted(corruption_mix.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {error_type} | {classify(error_type)} | {n} |")
    lines.append("")

    lines.append("## Results — all estimators on the same cells")
    lines.append("")
    lines.append(
        "| estimator | top-1 acc | recoverable | destroyed | rule / llm cells | "
        "model calls | policy cost |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in ORDER:
        row = by_name.get(name)
        if row is None:
            continue
        lines.append(
            f"| {_label(row)} | {_pct(row.accuracy)} | "
            f"{_pct(row.accuracy_recoverable)} | {_pct(row.accuracy_destroyed)} | "
            f"{row.n_rule} / {row.n_llm} | {row.llm_invocations} | "
            f"{_num(row.policy_cost)} |"
        )
    lines.append("")
    lines.append(
        "‡ `oracle` reads ground truth to pick the repairer that is actually "
        "right. It is an **upper bound only**, not a method, and it calls the "
        "model on every cell (it needs both answers before it can choose), so "
        "it is also the most expensive row."
    )
    lines.append("")

    lines.append("## Accuracy by error type")
    lines.append("")
    all_types = sorted({t for r in rows for t in r.accuracy_by_type})
    lines.append("| estimator | " + " | ".join(all_types) + " |")
    lines.append("|---" * (len(all_types) + 1) + "|")
    for name in ORDER:
        row = by_name.get(name)
        if row is None:
            continue
        cells = []
        for error_type in all_types:
            hits, n = row.accuracy_by_type.get(error_type, (0, 0))
            cells.append(f"{_pct(hits / n if n else None)} ({hits}/{n})" if n else "—")
        lines.append(f"| {_label(row)} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Model-call accounting")
    lines.append("")
    lines.append(
        "| estimator | model invocations | saved vs always_llm | "
        "transport calls this run | cache hits | model failures |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    baseline_calls = pure_llm.llm_invocations if pure_llm else None
    for name in ORDER:
        row = by_name.get(name)
        if row is None:
            continue
        saved = (
            f"{baseline_calls - row.llm_invocations:,} "
            f"({_pct((baseline_calls - row.llm_invocations) / baseline_calls)})"
            if baseline_calls else "—"
        )
        lines.append(
            f"| {_label(row)} | {row.llm_invocations} | {saved} | "
            f"{_num(row.transport_calls)} | {_num(row.cache_hits)} | "
            f"{row.llm_failures} |"
        )
    lines.append("")
    lines.append(
        "`model invocations` is what a deployment would pay. `transport calls "
        "this run` is lower because every estimator shares one on-disk "
        "response cache, so the second estimator to visit a cell gets it free."
    )
    lines.append("")

    lines.append("## Held-out comparison (the number that counts)")
    lines.append("")
    lines.append(
        f"`by_learned_error_type` has a fitted parameter — its per-error-type "
        f"route table — so its accuracy on the cells it was fitted on means "
        f"nothing. This table reports every estimator on the **same "
        f"{n_eval} held-out cells** ({100 * EVAL_FRACTION:.0f}% hold-out, seed "
        f"{SEED}, the same discipline as the Phase 6 calibrators), so the "
        f"comparison is out of sample for all of them."
    )
    if route_table:
        lines.append("")
        lines.append(f"Learned route table ({learned_note}):")
        lines.append("")
        lines.append("| error type | learned route |")
        lines.append("|---|---|")
        for error_type, route in sorted(route_table.items()):
            lines.append(f"| {error_type} | {route} |")
    lines.append("")
    lines.append("| estimator | held-out top-1 acc | all-cells top-1 acc | model calls |")
    lines.append("|---|---:|---:|---:|")
    for name in ORDER:
        row = by_name.get(name)
        if row is None:
            continue
        lines.append(
            f"| {_label(row)} | {_pct(row.eval_accuracy)} | {_pct(row.accuracy)} | "
            f"{row.llm_invocations} |"
        )
    lines.append("")
    lines.append(
        f"The held-out set is {n_eval} cells, so differences of one or two "
        f"cells are noise. No confidence intervals are computed here: a single "
        f"split is enough to show whether the mechanism exists, and a "
        f"repeated-split version is the obvious follow-up."
    )
    lines.append("")

    lines.append("## Decision mix under the proposed policy")
    lines.append("")
    lines.append("| estimator | AUTO | HUMAN | ABSTAIN | policy cost | always-auto cost |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name in ORDER:
        row = by_name.get(name)
        if row is None:
            continue
        lines.append(
            f"| {_label(row)} | {row.n_auto} | {row.n_human} | {row.n_abstain} | "
            f"{_num(row.policy_cost)} | {_num(row.always_auto_cost)} |"
        )
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    if best is None or pure_rule is None or pure_llm is None:
        lines.append("Not enough estimators ran to reach a verdict.")
    else:
        lines.append(
            f"- Best non-oracle router: **{best.name}** at {_pct(best.accuracy)} "
            f"top-1 accuracy."
        )
        lines.append(
            f"- Pure controls: `always_rule` {_pct(pure_rule.accuracy)}, "
            f"`always_llm` {_pct(pure_llm.accuracy)}."
        )
        if beat_both:
            lines.append(
                f"- **Routing beat BOTH pure strategies** "
                f"(+{100 * (best.accuracy - max(pure_rule.accuracy, pure_llm.accuracy)):.1f} "
                f"points over the better control)."
            )
        else:
            better = max(pure_rule, pure_llm, key=lambda r: r.accuracy)
            lines.append(
                f"- **Routing did NOT beat both pure strategies.** The best "
                f"control, `{better.name}`, is at {_pct(better.accuracy)} vs "
                f"{_pct(best.accuracy)} for the best router — a difference of "
                f"{100 * (best.accuracy - better.accuracy):+.1f} points. "
                f"Reported as a loss, per CLAUDE.md §8b.8."
            )
        oracle = by_name.get("oracle")
        if oracle is not None:
            lines.append(
                f"- Ceiling (`oracle`, upper bound only): "
                f"{_pct(oracle.accuracy)}; headroom above the best router is "
                f"{100 * (oracle.accuracy - best.accuracy):.1f} points."
            )
        if baseline_calls and best.llm_invocations < baseline_calls:
            lines.append(
                f"- API calls: the best router makes {best.llm_invocations} "
                f"model calls vs {baseline_calls} for `always_llm` — "
                f"{baseline_calls - best.llm_invocations} saved "
                f"({_pct((baseline_calls - best.llm_invocations) / baseline_calls)})."
            )
        if cheapest is not None:
            lines.append(
                f"- Cheapest realised policy cost among deployable routers: "
                f"`{cheapest.name}` at {_num(cheapest.policy_cost)} "
                f"experimental units."
            )
    lines.append("")
    lines.append(
        "Reproduce: `python experiments/w4_routing.py --live` (or `--fake` for "
        "the offline smoke test). Plan of record: `docs/research_program.md` §9, "
        "route W4.4."
    )
    lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--fake", action="store_true",
        help="canned deterministic responses; no network, no key (default)",
    )
    group.add_argument(
        "--live", action="store_true",
        help="real provider calls via LLMConfig.from_env() (cache-backed)",
    )
    parser.add_argument(
        "--no-llm-cache", action="store_true",
        help="bypass the on-disk response cache (forces fresh calls in --live)",
    )
    parser.add_argument(
        "--estimators", default="",
        help="comma-separated subset of estimators to run (default: all)",
    )
    parser.add_argument("--out", default=str(OUT_MD), help="output markdown path")
    parser.add_argument("--json", action="store_true", help="also print JSON rows")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    enable_unicode_stdout()
    args = parse_args(argv)
    mode = "live" if args.live else "fake"

    canned = CannedTransport() if mode == "fake" else None
    if mode == "fake":
        config = LLMConfig(
            api_key="canned-offline-no-network",
            base_url="https://canned.invalid/v1",
            model="canned-offline",
            temperature=0.0,
            max_tokens=TOP_K * 64,
            cache_dir=".cache/llm-w4-fake",
            use_cache=not args.no_llm_cache,
        )
        live_note = ""
    else:
        config = LLMConfig.from_env()
        if not config.is_configured:
            print(
                "--live requires a configured LLM_API_KEY / LLM_API_KEYS. "
                "Every cell would fall back to the mock repairer, which would "
                "make the 'llm' column a mislabelled copy of the rule column.",
                file=sys.stderr,
            )
            return 2
        live_note = (
            "Cells already answered by the earlier Breast Cancer run are served "
            "from the on-disk cache, so this run is reproducible and nearly free."
        )
    cache = DiskCache(config.cache_dir)

    costs = CostConfig.from_yaml(COSTS_YAML, DATASET)
    comparator = ValueComparator()
    truth = build_ground_truth(DATASET, DATA_ROOT)

    names = (
        [n.strip() for n in args.estimators.split(",") if n.strip()]
        or list(ORDER)
    )
    allowed = set(ESTIMATOR_NAMES) | {
        OracleEstimator.name, LearnedErrorTypeEstimator.name
    }
    unknown = [n for n in names if n not in allowed]
    if unknown:
        print(f"Unknown estimator(s): {unknown}", file=sys.stderr)
        return 2

    # Run in ORDER so the two pure controls exist before the learned router is
    # fitted from their 0/1 correctness labels.
    names = [n for n in ORDER if n in names]

    rows: List[EstimatorRow] = []
    by_name: Dict[str, EstimatorRow] = {}
    corruption_mix: Dict[str, int] = {}
    route_table: Dict[str, str] = {}
    learned_note = ""

    for name in names:
        if name == OracleEstimator.name:
            estimator: Any = OracleEstimator(ground_truth=truth, comparator=comparator)
        elif name == LearnedErrorTypeEstimator.name:
            needed = ("always_rule", "always_llm")
            if not all(n in by_name for n in needed):
                print(
                    f"Skipping {name}: it is fitted from the {needed} labels, "
                    f"so both controls must run first.",
                    file=sys.stderr,
                )
                continue
            rule_cells = by_name["always_rule"].per_cell
            llm_cells = by_name["always_llm"].per_cell
            keys = [key for key, _, _ in rule_cells]
            assert keys == [key for key, _, _ in llm_cells], "control cell sets diverged"
            train_idx, eval_idx = split_cells(keys)
            route_table = fit_error_type_routes(
                [rule_cells[i][1] for i in train_idx],
                [int(rule_cells[i][2]) for i in train_idx],
                [int(llm_cells[i][2]) for i in train_idx],
                default=ROUTE_RULE,
            )
            learned_note = (
                f"fitted on {len(train_idx)} train cells "
                f"({100 * (1 - EVAL_FRACTION):.0f}%), reported on "
                f"{len(eval_idx)} held-out cells"
            )
            estimator = LearnedErrorTypeEstimator(route_table=route_table)
            print(f"[w4_routing] learned route table: {route_table}", flush=True)
        else:
            estimator = make_estimator(name)

        print(f"[w4_routing] {name} ...", flush=True)
        row = run_estimator(
            estimator, mode, cache, config, canned, truth, costs, comparator, name=name
        )
        rows.append(row)
        by_name[name] = row
        if not corruption_mix:
            corruption_mix = dict(
                Counter(
                    error_type
                    for error_type, (_, n) in row.accuracy_by_type.items()
                    for _ in range(n)
                )
            )
        print(
            f"    acc={row.accuracy:.4f} rule={row.n_rule} llm={row.n_llm} "
            f"calls={row.llm_invocations} cost={row.policy_cost} "
            f"({row.seconds:.1f}s)",
            flush=True,
        )

    # Held-out accuracy for every estimator on the same eval cells, so the
    # learned router is compared out of sample against everything else.
    eval_keys: List[Tuple[int, str]] = []
    if "always_rule" in by_name:
        keys = [key for key, _, _ in by_name["always_rule"].per_cell]
        _, eval_idx = split_cells(keys)
        eval_keys = [keys[i] for i in eval_idx]
        for row in rows:
            row.eval_accuracy = subset_accuracy(row.per_cell, eval_keys)

    report = build_report(
        rows, mode, costs, config.model, corruption_mix,
        live_note=live_note if mode == "live" else "",
        route_table=route_table,
        learned_note=learned_note,
        n_eval=len(eval_keys),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nWrote {out_path}")

    if args.json:
        print(json.dumps([r.to_dict() for r in rows], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
