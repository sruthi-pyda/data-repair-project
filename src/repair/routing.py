"""
src/repair/routing.py
----------------------
Difficulty-aware repair routing (Wave 4, mechanism M1).

WHY THIS EXISTS — it comes from our own measurement
===================================================
`docs/PROJECT_REPORT.md` §3.5, controlled comparison on 80 real Breast Cancer
cells:

    corruption                        rule-based repairer      LLM
    recoverable (out-of-range/format)       100 %             92.5 %
    destroyed (missing)                      45 %             50 %

The LLM is **worse where the answer is determined by context** (it reasons
itself out of deterministic cases a rule gets right every time) and marginally
better where the answer is genuinely a guess. Sending every cell to one
repairer therefore pays the loser's price on one half of the data.

So: **route by difficulty.** Deterministic repairs go to the cheap rule,
genuinely ambiguous ones go to the model. If routing were perfect the ceiling
is ~100 % on recoverable + ~50 % on destroyed, beating *both* pure strategies.
It is also this project's own thesis applied one level up: decide what to
trust — now about *which repairer* to trust, not which value.

Routing is a hypothesis, not a result. `experiments/w4_routing.py` measures it
and `results/w4_routing.md` reports the outcome, including a loss.

WHAT IS IN HERE
===============
``RoutingRepairer`` is a **drop-in for ``MockRepairer`` / ``LLMRepairer``**:
identical ``repair(context)`` and ``repair_batch(records, dirty_df, profile)``
signatures, identical ``RepairResult`` output, no ground-truth parameter. It
can therefore be handed to anything that accepts a Phase 4 repairer, including
``src/pipeline/runner.py``'s ``build_repairer`` seam.

A ``DifficultyEstimator`` decides, per cell, which repairer answers:

    always_rule               pure control — the rule repairer, every cell
    always_llm                pure control — the model, every cell
    by_error_type             the obvious reading of the table above:
                              OUT_OF_RANGE / FORMAT_ERROR / NOVEL_CATEGORY /
                              WHITESPACE -> rule;  MISSING -> LLM
    by_learned_error_type     the same idea, but the per-error-type table is
                              LEARNED from 0/1 correctness labels on a train
                              split (`fit_error_type_routes`) instead of being
                              written by hand from one measurement. It exists
                              because the hand-written partition turned out to
                              be wrong about NOVEL_CATEGORY — see
                              `results/w4_routing.md`.
    by_candidate_agreement    run the cheap rule first; keep its answer when
                              its top candidate's probability clears a
                              threshold, escalate otherwise. SELF-SUPERVISED:
                              no error-type table, so it generalises past our
                              three datasets, which `by_error_type` does not.
    by_candidate_margin       same idea on the top-1 minus top-2 gap, i.e.
                              "do the candidates disagree?" rather than
                              "is the top one confident?"
    by_context_availability   escalate only when the row actually carries
                              enough non-null context for a model to say
                              anything useful.
    oracle                    routes each cell to whichever repairer is
                              actually right. **UPPER BOUND ONLY** — it reads
                              ground truth and is not a deployable method.
                              Every table that shows it must label it so.

WHICH REPAIRER ANSWERED IS VISIBLE IN THE OUTPUT
================================================
``RepairResult.repair_method`` becomes ``"routed:rule"`` or ``"routed:llm"``,
so routing shows up in the predictions CSV rather than being invisible.
``context_used`` keeps the full story: ``route``, ``route_reason`` (why the
estimator decided) and ``inner_method`` (the underlying repairer's own method
name, e.g. ``statistical_repair`` or ``llm->mock (timeout)``).

``RoutingRepairer.stats`` carries per-route counts, per-reason counts and
``llm_invocations`` — the number of cells actually handed to the model. That
last one is the practical selling point: routing should cut API calls as well
as (perhaps) improving accuracy, and a mechanism that improves nothing but
halves the bill is still a result.

DATA LEAKAGE PREVENTION (CLAUDE.md §3 — non-negotiable)
=======================================================
Every estimator except ``oracle`` decides from the **dirty** cell alone:

  * ``RepairContext`` — dirty value, error type, column kind/dtype, the same
    row's other dirty values, a vocabulary mined from the dirty column, and
    aggregate profile statistics (the documented population-level caveat);
  * the *structure* of the rule repairer's candidate list — probabilities and
    values it produced, never a comparison against a clean value.

No function in this module takes a ``clean_df``, and no estimator with
``uses_ground_truth == False`` can see one: ``tests/test_routing.py`` asserts
that scrambling the clean values (a within-column permutation, so column
statistics are untouched) leaves every route bit-identical, and that a fake
transport's captured request bodies contain no clean value.

``OracleEstimator`` is the single, explicit exception. It takes a
``ground_truth`` mapping, is flagged ``uses_ground_truth = True``, and
``RoutingRepairer.uses_ground_truth`` re-exports that flag so a caller cannot
accidentally present a ceiling as a method.

FAILURE HANDLING
================
A cell escalated to the model that then fails falls back to the rule repairer
for that cell, is counted in ``stats.llm_failures``, and is labelled
``route_reason = "<reason>+llm_failed"``. ``LLMRepairer`` already has its own
per-cell mock fallback, so this layer only catches a repairer that *raises*.

DETERMINISM
===========
Routing adds no randomness. Given a seed, the rule repairer is deterministic
by construction, the LLM repairer is deterministic at ``temperature=0`` through
its on-disk cache, and every estimator is a pure function of its inputs.

WORKED EXAMPLE
==============
    from src.repair.mock_repair import MockRepairer
    from src.repair.llm_repair import LLMConfig, LLMRepairer
    from src.repair.routing import RoutingRepairer, make_estimator

    rule = MockRepairer(seed=42)
    llm  = LLMRepairer(config=LLMConfig.from_env(), fallback=rule)

    router = RoutingRepairer(
        estimator=make_estimator("by_error_type"), rule=rule, llm=llm,
    )
    results = router.repair_batch(records, dirty_df, profile)   # no clean_df

    print(results[0].repair_method)      # 'routed:rule'
    print(router.stats.to_dict())
    # {'cells': 392, 'n_rule': 145, 'n_llm': 247, 'llm_invocations': 247, ...}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from src.data.schema import DatasetProfile
from src.detection.detector import DetectionRecord, ErrorType
from src.repair.mock_repair import (
    MockRepairer,
    RepairContext,
    RepairResult,
    build_repair_context,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Route names and error-type partition
# ═══════════════════════════════════════════════════════════════════════════════

ROUTE_RULE: str = "rule"
ROUTE_LLM: str = "llm"
ROUTES: Tuple[str, str] = (ROUTE_RULE, ROUTE_LLM)

#: Error types whose correct value is *implied by what is still on the page* —
#: the digits of an out-of-range value, the characters of a format error, the
#: near-miss spelling of a novel category, the payload of a whitespace error.
#: Measured at 98–100 % for the rule repairer (report §2), and 92.5 % for the
#: LLM on the out-of-range subset (§3.5). These go to the rule.
RECOVERABLE_ERROR_TYPES: frozenset = frozenset(
    {
        ErrorType.OUT_OF_RANGE.value,
        ErrorType.FORMAT_ERROR.value,
        ErrorType.NOVEL_CATEGORY.value,
        ErrorType.WHITESPACE.value,
    }
)

#: Error types where the original value is gone and any answer is a guess
#: (29–50 % for either repairer). These are the only ones worth a model call.
#:
#: STATISTICAL_OUTLIER sits here on an argument, not a measurement: the cell
#: still holds a value, but nothing in the row says what it *should* have been,
#: so it is closer to a guess than to a decoding. The shipped Breast Cancer
#: detector produces none of them, so this choice changes no reported number;
#: it is flagged here so a later dataset does not inherit it silently.
AMBIGUOUS_ERROR_TYPES: frozenset = frozenset(
    {
        ErrorType.MISSING.value,
        ErrorType.STATISTICAL_OUTLIER.value,
    }
)


# ═══════════════════════════════════════════════════════════════════════════════
# Route decision and statistics
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RouteDecision:
    """
    One estimator's verdict for one cell.

    Fields
    ------
    route : str
        ``"rule"`` or ``"llm"``.
    reason : str
        Short machine-readable reason, e.g. ``"recoverable_error_type"`` or
        ``"low_top_probability"``. Lands in
        ``RepairResult.context_used["route_reason"]`` so a routing decision is
        explainable per cell, and is tallied in ``RoutingStats.reasons``.
    """

    route: str
    reason: str

    def __post_init__(self) -> None:
        if self.route not in ROUTES:
            raise ValueError(f"route must be one of {ROUTES}, got {self.route!r}")


@dataclass
class RoutingStats:
    """
    Counters for one ``RoutingRepairer``.

    Fields
    ------
    cells : int
        Cells passed to ``repair()``.
    n_rule, n_llm : int
        Cells whose *returned* result came from each repairer. A cell escalated
        to a model that then failed counts under ``n_rule``.
    llm_invocations : int
        Times the model repairer was actually called. This is the API-cost
        number, and it is **not** always ``n_llm``: ``oracle`` calls both
        repairers on every cell and then discards one answer, so for the
        ceiling ``llm_invocations == cells`` while ``n_llm`` is smaller.
    llm_failures : int
        Escalated cells where the model repairer raised and the rule answered
        instead.
    rule_invocations : int
        Times the rule repairer was called (it is also called *before* the
        decision by the agreement/margin/oracle estimators).
    reasons : dict[str, int]
        ``RouteDecision.reason`` counts.
    routes_by_error_type : dict[str, dict[str, int]]
        ``{error_type: {"rule": n, "llm": n}}`` — which cells each route took,
        the breakdown the M1 hypothesis is actually about.
    """

    cells: int = 0
    n_rule: int = 0
    n_llm: int = 0
    llm_invocations: int = 0
    llm_failures: int = 0
    rule_invocations: int = 0
    reasons: Dict[str, int] = field(default_factory=dict)
    routes_by_error_type: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def llm_fraction(self) -> float:
        """Fraction of cells answered by the model; 0.0 when no cells seen."""
        return (self.n_llm / self.cells) if self.cells else 0.0

    def record(self, error_type: str, route: str, reason: str) -> None:
        """Tally one finalised routing decision."""
        if route == ROUTE_LLM:
            self.n_llm += 1
        else:
            self.n_rule += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        bucket = self.routes_by_error_type.setdefault(
            str(error_type), {ROUTE_RULE: 0, ROUTE_LLM: 0}
        )
        bucket[route] = bucket.get(route, 0) + 1

    def reset(self) -> None:
        """Zero every counter (used between datasets in one process)."""
        self.cells = 0
        self.n_rule = 0
        self.n_llm = 0
        self.llm_invocations = 0
        self.llm_failures = 0
        self.rule_invocations = 0
        self.reasons = {}
        self.routes_by_error_type = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cells": self.cells,
            "n_rule": self.n_rule,
            "n_llm": self.n_llm,
            "llm_invocations": self.llm_invocations,
            "llm_failures": self.llm_failures,
            "rule_invocations": self.rule_invocations,
            "llm_fraction": round(self.llm_fraction, 4),
            "reasons": dict(self.reasons),
            "routes_by_error_type": {
                k: dict(v) for k, v in self.routes_by_error_type.items()
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Difficulty estimators
# ═══════════════════════════════════════════════════════════════════════════════


class DifficultyEstimator:
    """
    Base class: decide which repairer answers one cell.

    Subclass contract
    -----------------
    name : str
        Registry key, printed in every table row.
    can_escalate : bool
        False only for ``always_rule``. ``RoutingRepairer`` refuses to be built
        without a model repairer when this is True, so a misconfiguration fails
        at construction rather than silently degrading mid-run.
    needs_rule_result : bool
        True when ``route()`` inspects the rule repairer's candidates. The
        router then runs the rule *first* and reuses that result, so the cheap
        repairer is never run twice.
    needs_both_results : bool
        True when ``route()`` needs both answers (``oracle`` only). The router
        then calls both repairers on every cell, which is why the ceiling's
        ``llm_invocations`` equals its cell count.
    uses_ground_truth : bool
        True only for ``oracle``. Anything True here is an upper bound, not a
        method, and must be labelled as such wherever it is reported.

    LEAKAGE: ``route()`` receives a ``RepairContext`` (dirty data + aggregate
    profile statistics) and ``RepairResult`` objects. Neither carries a clean
    value. The one estimator that needs ground truth declares it.
    """

    name: str = "base"
    can_escalate: bool = True
    needs_rule_result: bool = False
    needs_both_results: bool = False
    uses_ground_truth: bool = False

    def route(
        self,
        context: RepairContext,
        rule_result: Optional[RepairResult] = None,
        llm_result: Optional[RepairResult] = None,
    ) -> RouteDecision:
        raise NotImplementedError  # pragma: no cover - abstract

    def describe(self) -> str:
        """One-line description for a results table."""
        return self.name


class AlwaysRuleEstimator(DifficultyEstimator):
    """Pure control: the rule repairer answers every cell.

    With this estimator ``RoutingRepairer`` reproduces the rule repairer's
    candidates exactly and never constructs or calls a model repairer.
    """

    name = "always_rule"
    can_escalate = False

    def route(self, context, rule_result=None, llm_result=None) -> RouteDecision:
        return RouteDecision(ROUTE_RULE, "always_rule")


class AlwaysLLMEstimator(DifficultyEstimator):
    """Pure control: the model answers every cell (the Phase 12 behaviour)."""

    name = "always_llm"

    def route(self, context, rule_result=None, llm_result=None) -> RouteDecision:
        return RouteDecision(ROUTE_LLM, "always_llm")


class ByErrorTypeEstimator(DifficultyEstimator):
    """
    Route on the detector's error type — the direct reading of §3.5.

    Recoverable error types (the original is implied by what is left in the
    cell) go to the rule; ambiguous ones (the original is destroyed) go to the
    model. Anything the partition does not mention is treated as ambiguous and
    escalated, which is the conservative choice: an unknown error type is not
    evidence that a rule can decode it.

    Honest limitation: this estimator is **fitted to our three datasets by
    hand**. It encodes a measurement, not a learned rule, so it does not
    transfer to a corruption process we have not characterised. That is exactly
    why ``by_candidate_agreement`` exists.
    """

    name = "by_error_type"

    def __init__(
        self,
        rule_error_types: frozenset = RECOVERABLE_ERROR_TYPES,
        escalate_unknown: bool = True,
    ) -> None:
        self.rule_error_types = frozenset(rule_error_types)
        self.escalate_unknown = bool(escalate_unknown)

    def route(self, context, rule_result=None, llm_result=None) -> RouteDecision:
        error_type = str(context.error_type)
        if error_type in self.rule_error_types:
            return RouteDecision(ROUTE_RULE, "recoverable_error_type")
        if error_type in AMBIGUOUS_ERROR_TYPES:
            return RouteDecision(ROUTE_LLM, "ambiguous_error_type")
        if self.escalate_unknown:
            return RouteDecision(ROUTE_LLM, "unknown_error_type")
        return RouteDecision(ROUTE_RULE, "unknown_error_type")

    def describe(self) -> str:
        return f"{self.name} (rule on {sorted(self.rule_error_types)})"


def fit_error_type_routes(
    error_types: List[str],
    rule_correct: List[int],
    llm_correct: List[int],
    default: str = ROUTE_RULE,
    min_support: int = 1,
) -> Dict[str, str]:
    """
    Learn, per error type, which repairer is more accurate.

    This is the **fitting** half of :class:`LearnedErrorTypeEstimator`, and it
    is the one function in this module that consumes labels. It takes three
    aligned lists — the error type of each cell and two 0/1 correctness labels
    — and returns ``{error_type: route}``.

    LEAKAGE (CLAUDE.md §3): the labels are 0/1 *repair-correctness* labels, the
    same permitted shape ``CalibrationBuilder.build()`` produces for ``p``, and
    they must come from a **train split only**. No clean value is passed here,
    and the returned table is a learned parameter, exactly like a calibrator's
    coefficients: the estimator that consumes it sees nothing but the dirty
    cell. The experiment that fits it is responsible for the split; this
    function cannot tell a train label from a test one and does not pretend to.

    Ties go to the rule — it is the cheaper repairer, so a type where the two
    are indistinguishable should not buy API calls. An error type with fewer
    than ``min_support`` labelled cells is left out of the table entirely and
    falls through to ``default`` at route time, which is honest about a type
    nobody has measured.

    Raises
    ------
    ValueError
        If the three lists are not the same length.
    """
    if not (len(error_types) == len(rule_correct) == len(llm_correct)):
        raise ValueError(
            f"misaligned inputs: {len(error_types)} error types, "
            f"{len(rule_correct)} rule labels, {len(llm_correct)} llm labels"
        )
    if default not in ROUTES:
        raise ValueError(f"default must be one of {ROUTES}, got {default!r}")

    tally: Dict[str, List[int]] = {}
    for error_type, rule_ok, llm_ok in zip(error_types, rule_correct, llm_correct):
        bucket = tally.setdefault(str(error_type), [0, 0, 0])
        bucket[0] += int(bool(rule_ok))
        bucket[1] += int(bool(llm_ok))
        bucket[2] += 1

    table: Dict[str, str] = {}
    for error_type, (rule_hits, llm_hits, n) in tally.items():
        if n < max(1, int(min_support)):
            continue
        table[error_type] = ROUTE_LLM if llm_hits > rule_hits else ROUTE_RULE
    return table


class LearnedErrorTypeEstimator(DifficultyEstimator):
    """
    Route on a per-error-type table **learned from labels**, not hand-written.

    ``by_error_type`` encodes the §3.5 measurement by hand, and that
    measurement covered only out-of-range, format and missing cells. This
    estimator asks the data instead: for each error type, which repairer was
    actually more accurate on a *training* split? The table is a learned
    parameter, fitted by :func:`fit_error_type_routes` the way Phase 6 fits a
    calibrator, and at route time this estimator sees nothing but the dirty
    cell's error type.

    It is therefore a method, not a ceiling: ``uses_ground_truth`` is False
    because no clean value and no label reaches ``route()``. The honesty
    requirement lands on the *experiment*, which must fit on one split and
    report on another — ``experiments/w4_routing.py`` does, and says so.

    ``default`` is used for an error type absent from the table (unseen in
    training, or below ``min_support``). It defaults to the rule: an unmeasured
    type is not a reason to spend an API call.
    """

    name = "by_learned_error_type"

    def __init__(self, route_table: Mapping[str, str], default: str = ROUTE_RULE) -> None:
        if default not in ROUTES:
            raise ValueError(f"default must be one of {ROUTES}, got {default!r}")
        bad = {k: v for k, v in dict(route_table).items() if v not in ROUTES}
        if bad:
            raise ValueError(f"route_table has invalid routes: {bad}")
        self.route_table: Dict[str, str] = {str(k): str(v) for k, v in route_table.items()}
        self.default = default

    @property
    def can_escalate(self) -> bool:  # type: ignore[override]
        """True unless the learned table never routes to the model."""
        return self.default == ROUTE_LLM or ROUTE_LLM in self.route_table.values()

    def route(self, context, rule_result=None, llm_result=None) -> RouteDecision:
        error_type = str(context.error_type)
        if error_type in self.route_table:
            route = self.route_table[error_type]
            return RouteDecision(route, f"learned_{route}")
        return RouteDecision(self.default, "learned_default")

    def describe(self) -> str:
        pairs = ", ".join(f"{k}->{v}" for k, v in sorted(self.route_table.items()))
        return f"{self.name} ({pairs or 'empty table'}; default {self.default})"


class ByCandidateAgreementEstimator(DifficultyEstimator):
    """
    Self-supervised routing from the cheap repairer's own candidate list.

    Run the rule repairer first, then keep its answer when it looks decided and
    escalate when it does not. Two readings of "decided", selected by ``mode``:

    ``"top_probability"``
        The top candidate's probability, i.e. *how confident* the rule is.
    ``"margin"``
        Top-1 minus top-2, i.e. *whether the candidates disagree*. A single
        candidate has probability 1.0 after normalisation and therefore the
        maximum margin.

    Escalate when the score is **below** ``threshold``. An empty candidate list
    always escalates — there is nothing to keep.

    Unlike ``by_error_type`` this needs no table of error types, so it is the
    variant that could generalise. Whether it actually predicts "determined vs
    guess" is a measurement, and in mock mode it is a poor one: the mock's
    probabilities come from fixed per-strategy constants rather than from
    evidence about the cell. See `results/w4_routing.md`.
    """

    name = "by_candidate_agreement"
    needs_rule_result = True

    def __init__(self, threshold: float = 0.5, mode: str = "top_probability") -> None:
        if not (0.0 <= float(threshold) <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        if mode not in ("top_probability", "margin"):
            raise ValueError(
                f"mode must be 'top_probability' or 'margin', got {mode!r}"
            )
        self.threshold = float(threshold)
        self.mode = mode

    def score(self, rule_result: Optional[RepairResult]) -> Optional[float]:
        """
        The agreement score for this cell, or ``None`` when undefined.

        ``None`` means "no candidates at all", which always escalates. It is
        deliberately not 0.0: an undefined score and a confidently-zero score
        are different things, and the project prints ``—`` for the former.
        """
        if rule_result is None or not rule_result.candidates:
            return None
        probs = [float(c.probability) for c in rule_result.candidates]
        if self.mode == "top_probability":
            return probs[0]
        top2 = probs[1] if len(probs) > 1 else 0.0
        return probs[0] - top2

    def route(self, context, rule_result=None, llm_result=None) -> RouteDecision:
        value = self.score(rule_result)
        if value is None:
            return RouteDecision(ROUTE_LLM, "no_rule_candidates")
        if value < self.threshold:
            reason = (
                "low_top_probability" if self.mode == "top_probability"
                else "low_candidate_margin"
            )
            return RouteDecision(ROUTE_LLM, reason)
        reason = (
            "confident_top_probability" if self.mode == "top_probability"
            else "wide_candidate_margin"
        )
        return RouteDecision(ROUTE_RULE, reason)

    def describe(self) -> str:
        return f"{self.name} ({self.mode} < {self.threshold:g} -> llm)"


class ByContextAvailabilityEstimator(DifficultyEstimator):
    """
    Escalate only when the row carries enough context to reason from.

    A model's one advantage over a column statistic is the rest of the row. If
    most of the row is missing there is nothing for it to condition on, so the
    call buys nothing and the cheap imputation is as good as anything.

    ``min_known_fraction`` is the fraction of non-null context values required
    before escalating. A row with no context columns at all scores 0.0 and
    stays with the rule.
    """

    name = "by_context_availability"

    def __init__(self, min_known_fraction: float = 0.5) -> None:
        if not (0.0 <= float(min_known_fraction) <= 1.0):
            raise ValueError(
                f"min_known_fraction must be in [0, 1], got {min_known_fraction}"
            )
        self.min_known_fraction = float(min_known_fraction)

    @staticmethod
    def known_fraction(context: RepairContext) -> float:
        """Fraction of the row's context values that are present."""
        values = list((context.row_context or {}).values())
        if not values:
            return 0.0
        known = sum(1 for v in values if v is not None and not _is_missing(v))
        return known / len(values)

    def route(self, context, rule_result=None, llm_result=None) -> RouteDecision:
        fraction = self.known_fraction(context)
        if fraction >= self.min_known_fraction:
            return RouteDecision(ROUTE_LLM, "context_available")
        return RouteDecision(ROUTE_RULE, "context_too_sparse")

    def describe(self) -> str:
        return f"{self.name} (>= {self.min_known_fraction:g} known -> llm)"


class OracleEstimator(DifficultyEstimator):
    """
    **UPPER BOUND ONLY — reads ground truth. Not a method.**

    Runs both repairers and keeps whichever is actually correct, so its
    accuracy is by construction at least that of any other estimator on the
    same cells and the same two repairers. It answers "how much is routing
    worth if the difficulty signal were perfect?" and nothing else.

    Ties go to the rule (the cheap repairer), and when neither is correct the
    rule answers too — neither choice can change accuracy, and both keep the
    ceiling's cost realistic rather than flattering the model.

    ``ground_truth`` maps ``(row_id, column) -> the correct value``. It is
    supplied by the *experiment*, after predictions exist, exactly as the
    evaluators in ``src/evaluation/`` are. ``uses_ground_truth`` is True so no
    caller can mistake this for something deployable.
    """

    name = "oracle"
    needs_rule_result = True
    needs_both_results = True
    uses_ground_truth = True

    def __init__(
        self,
        ground_truth: Mapping[Tuple[int, str], Any],
        comparator: Optional[Any] = None,
    ) -> None:
        self.ground_truth = dict(ground_truth)
        if comparator is None:
            from src.evaluation.repair_metrics import ValueComparator

            comparator = ValueComparator()
        self.comparator = comparator

    def _is_correct(self, result: Optional[RepairResult], target: Any) -> bool:
        if result is None or not result.candidates:
            return False
        return bool(self.comparator.are_equal(result.top_candidate.value, target))

    def route(self, context, rule_result=None, llm_result=None) -> RouteDecision:
        key = (int(context.row_id), str(context.column))
        if key not in self.ground_truth:
            # No label for this cell: fall back to the cheap repairer rather
            # than guessing. Counted under its own reason so a results table
            # can show how much of the ceiling was actually informed.
            return RouteDecision(ROUTE_RULE, "no_ground_truth")
        target = self.ground_truth[key]
        if self._is_correct(rule_result, target):
            return RouteDecision(ROUTE_RULE, "rule_correct")
        if self._is_correct(llm_result, target):
            return RouteDecision(ROUTE_LLM, "llm_correct")
        return RouteDecision(ROUTE_RULE, "neither_correct")

    def describe(self) -> str:
        return f"{self.name} (UPPER BOUND — uses ground truth)"


def _is_missing(value: Any) -> bool:
    """True for None / NaN / pd.NA / an empty-or-whitespace string."""
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and not value.strip():
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Estimator registry
# ═══════════════════════════════════════════════════════════════════════════════

#: Name -> zero-argument-constructible factory. ``oracle`` is absent on
#: purpose: it cannot be built without ground truth, so it must be constructed
#: explicitly by an experiment and can never be selected by name from a CLI
#: flag that a deployment might inherit.
ESTIMATORS: Dict[str, Any] = {
    AlwaysRuleEstimator.name: AlwaysRuleEstimator,
    AlwaysLLMEstimator.name: AlwaysLLMEstimator,
    ByErrorTypeEstimator.name: ByErrorTypeEstimator,
    ByCandidateAgreementEstimator.name: ByCandidateAgreementEstimator,
    "by_candidate_margin": lambda **kw: ByCandidateAgreementEstimator(
        threshold=kw.pop("threshold", 0.2), mode=kw.pop("mode", "margin"), **kw
    ),
    ByContextAvailabilityEstimator.name: ByContextAvailabilityEstimator,
}

#: Declaration order, so every table lists estimators the same way.
ESTIMATOR_NAMES: Tuple[str, ...] = tuple(ESTIMATORS)


def make_estimator(name: str, **kwargs: Any) -> DifficultyEstimator:
    """
    Build a registered estimator by name.

    Raises
    ------
    ValueError
        For an unknown name, including ``"oracle"`` — the ceiling needs ground
        truth and must be constructed directly.
    """
    if name == OracleEstimator.name:
        raise ValueError(
            "The 'oracle' estimator reads ground truth and is an upper bound "
            "only; construct OracleEstimator(ground_truth=...) explicitly."
        )
    if name == LearnedErrorTypeEstimator.name:
        raise ValueError(
            "The 'by_learned_error_type' estimator needs a route table fitted "
            "on a train split; construct "
            "LearnedErrorTypeEstimator(route_table=fit_error_type_routes(...)) "
            "explicitly so the split is visible at the call site."
        )
    if name not in ESTIMATORS:
        raise ValueError(
            f"Unknown estimator '{name}'. Choose from {sorted(ESTIMATORS)}."
        )
    factory = ESTIMATORS[name]
    estimator = factory(**kwargs)
    if name == "by_candidate_margin":
        # The registry entry is a lambda over one class; keep the reported name
        # faithful to the flag that selected it.
        estimator.name = "by_candidate_margin"
    return estimator


# ═══════════════════════════════════════════════════════════════════════════════
# RoutingRepairer
# ═══════════════════════════════════════════════════════════════════════════════


class RoutingRepairer:
    """
    Difficulty-aware repairer — a drop-in for ``MockRepairer`` / ``LLMRepairer``.

    Parameters
    ----------
    estimator : DifficultyEstimator
        Decides the route per cell.
    rule : repairer or None
        The cheap repairer. ``None`` builds ``MockRepairer(seed, top_k)`` — the
        exact object every reported mock number came from.
    llm : repairer or None
        The model repairer. Any object with ``repair(context) -> RepairResult``
        is accepted, so a test injects a stub and never touches the network.
        It is **never constructed implicitly**: passing ``None`` with an
        estimator that can escalate raises, because a router that quietly
        becomes ``always_rule`` would report a fake win.
    seed, top_k : int
        Used only to build the default rule repairer.

    Notes
    -----
    ``repair()`` never raises for a model-side failure: the cell falls back to
    the rule repairer and is counted in ``stats.llm_failures``.
    """

    def __init__(
        self,
        estimator: DifficultyEstimator,
        rule: Optional[Any] = None,
        llm: Optional[Any] = None,
        seed: int = 42,
        top_k: int = 3,
    ) -> None:
        if estimator is None:
            raise ValueError("estimator is required")
        needs_llm = bool(estimator.can_escalate or estimator.needs_both_results)
        if needs_llm and llm is None:
            raise ValueError(
                f"Estimator '{estimator.name}' can route to the model, so a "
                f"'llm' repairer is required. Pass one explicitly (nothing is "
                f"constructed implicitly: a router that silently became "
                f"always_rule would report a result it did not earn)."
            )
        self.estimator = estimator
        self.seed = seed
        self.top_k = top_k
        self.rule = rule if rule is not None else MockRepairer(seed=seed, top_k=top_k)
        self.llm = llm
        self._stats = RoutingStats()

    # ------------------------------------------------------------------ #
    # Public API (identical to MockRepairer / LLMRepairer)                #
    # ------------------------------------------------------------------ #

    @property
    def stats(self) -> RoutingStats:
        """Live counters: routes taken, reasons, model invocations, failures."""
        return self._stats

    @property
    def uses_ground_truth(self) -> bool:
        """True when the estimator is a ground-truth ceiling, not a method."""
        return bool(self.estimator.uses_ground_truth)

    def reset_stats(self) -> None:
        """Zero the counters (used between datasets in one process)."""
        self._stats.reset()

    def repair(self, context: RepairContext) -> RepairResult:
        """
        Route one cell and return the chosen repairer's result.

        Parameters
        ----------
        context : RepairContext
            Built by ``build_repair_context()`` — contains NO clean values.

        Returns
        -------
        RepairResult
            ``repair_method`` is ``"routed:rule"`` or ``"routed:llm"``;
            ``context_used`` carries ``route``, ``route_reason`` and
            ``inner_method``.
        """
        self._stats.cells += 1

        rule_result: Optional[RepairResult] = None
        llm_result: Optional[RepairResult] = None

        if self.estimator.needs_both_results:
            rule_result = self._run_rule(context)
            llm_result = self._run_llm(context)
        elif self.estimator.needs_rule_result:
            rule_result = self._run_rule(context)

        decision = self.estimator.route(context, rule_result, llm_result)
        route, reason = decision.route, decision.reason

        if route == ROUTE_LLM:
            if llm_result is None:
                llm_result = self._run_llm(context)
            if llm_result is None:
                # The model repairer raised. Fall back to the cheap repairer
                # for this one cell and say so in the reason.
                route = ROUTE_RULE
                reason = f"{reason}+llm_failed"
                chosen = rule_result if rule_result is not None else self._run_rule(context)
            else:
                chosen = llm_result
        else:
            chosen = rule_result if rule_result is not None else self._run_rule(context)

        self._stats.record(context.error_type, route, reason)
        return self._annotate(chosen, route, reason)

    def repair_batch(
        self,
        records: List[DetectionRecord],
        dirty_df: pd.DataFrame,
        profile: DatasetProfile,
    ) -> List[RepairResult]:
        """
        Route and repair every detection record, in input order.

        Mirrors ``MockRepairer.repair_batch`` exactly: one ``RepairResult`` per
        input record, same order, built from ``build_repair_context()``. No
        ``clean_df`` parameter exists, by design.
        """
        results: List[RepairResult] = []
        for rec in records:
            ctx = build_repair_context(rec, dirty_df, profile)
            results.append(self.repair(ctx))
        logger.info(
            "Routed repair complete for '%s' (%s): %d cells, %d rule, %d llm, "
            "%d model invocations, %d model failures.",
            profile.name, self.estimator.name, len(results),
            self._stats.n_rule, self._stats.n_llm,
            self._stats.llm_invocations, self._stats.llm_failures,
        )
        return results

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _run_rule(self, context: RepairContext) -> RepairResult:
        self._stats.rule_invocations += 1
        return self.rule.repair(context)

    def _run_llm(self, context: RepairContext) -> Optional[RepairResult]:
        """
        Call the model repairer; ``None`` on any exception.

        ``LLMRepairer`` already handles provider failures per cell with its own
        mock fallback, so this catches only a repairer that *raises* — a stub,
        a programming error, or a transport that escaped its own handling.
        """
        if self.llm is None:  # pragma: no cover - guarded at construction
            return None
        self._stats.llm_invocations += 1
        try:
            return self.llm.repair(context)
        except Exception as exc:  # noqa: BLE001 - routing must never be fatal
            self._stats.llm_failures += 1
            logger.warning(
                "Model repairer failed for [%s row=%d]; falling back to the "
                "rule repairer for this cell: %s",
                context.column, context.row_id, exc,
            )
            return None

    @staticmethod
    def _annotate(result: RepairResult, route: str, reason: str) -> RepairResult:
        """
        Stamp the route onto the result without touching its candidates.

        Candidates, dirty value, row id, column and error type are passed
        through untouched, so ``always_rule`` / ``always_llm`` reproduce their
        pure repairer's candidate list bit-for-bit; only the provenance fields
        change.
        """
        context_used = dict(result.context_used)
        context_used["route"] = route
        context_used["route_reason"] = reason
        context_used["inner_method"] = result.repair_method
        return replace(
            result,
            repair_method=f"routed:{route}",
            context_used=context_used,
        )


def build_routing_repairer(
    estimator: str = "by_error_type",
    rule: Optional[Any] = None,
    llm: Optional[Any] = None,
    seed: int = 42,
    top_k: int = 3,
    **estimator_kwargs: Any,
) -> RoutingRepairer:
    """
    Convenience factory: build a ``RoutingRepairer`` from an estimator *name*.

    Mirrors ``src/pipeline/runner.build_repairer``'s role for the routed
    repairer. ``llm`` is still never constructed implicitly — see
    ``RoutingRepairer``.
    """
    return RoutingRepairer(
        estimator=make_estimator(estimator, **estimator_kwargs),
        rule=rule,
        llm=llm,
        seed=seed,
        top_k=top_k,
    )
