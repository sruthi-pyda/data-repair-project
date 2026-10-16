"""
tests/test_routing.py
----------------------
Behavioural tests for Wave 4 mechanism M1: src/repair/routing.py.

NO TEST IN THIS FILE TOUCHES THE NETWORK.
=========================================
Every test that involves the model repairer either injects a `FakeClient` into
`LLMRepairer(client=...)` or uses a local stub repairer. An autouse fixture
additionally replaces `llm_repair._build_default_transport` and
`urllib.request.urlopen` with functions that raise, so an accidental real
transport construction or HTTP call fails the test loudly rather than reaching
a provider. `TestNoNetwork` asserts that guard actually holds.

NO REAL API KEY APPEARS HERE. `FAKE_KEY` is a fabricated string.

TESTS THAT CAN FAIL
===================
The central one is `TestRoutingBeatsBothPureStrategies`: on a synthetic set
where the rule repairer is perfect on one error type and the model is perfect
on the other, `by_error_type` must beat BOTH pure strategies. If that fails the
M1 hypothesis is wrong at the mechanism level, not just unlucky on our data.
Expected accuracies are re-derived in the test from the construction of the
synthetic set, never read back from the code under test.

LEAKAGE (CLAUDE.md §3)
======================
`TestLeakage` asserts three separate things:
  * no public parameter or dataclass field of the router (or of any estimator
    that does not declare `uses_ground_truth`) is named after clean data;
  * routes are bit-identical when the clean values are scrambled by a
    within-column permutation (which leaves column statistics, the only
    clean-derived information the profiler may use, untouched);
  * the captured request bodies of a fake transport contain no clean value.
"""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import pytest

from src.data.loader import DataLoader
from src.data.profiler import profile_dataset
from src.data.schema import (
    CategoricalStats,
    ColumnKind,
    ColumnProfile,
    DatasetProfile,
    NumericalStats,
)
from src.detection.detector import DetectionRecord, Detector, ErrorType
from src.evaluation.repair_metrics import ValueComparator
from src.repair import llm_repair
from src.repair.llm_repair import LLMConfig, LLMRepairer
from src.repair.mock_repair import (
    MockRepairer,
    RepairCandidate,
    RepairContext,
    RepairResult,
    build_repair_context,
)
from src.repair.routing import (
    AMBIGUOUS_ERROR_TYPES,
    ESTIMATOR_NAMES,
    ESTIMATORS,
    RECOVERABLE_ERROR_TYPES,
    ROUTE_LLM,
    ROUTE_RULE,
    AlwaysLLMEstimator,
    AlwaysRuleEstimator,
    ByCandidateAgreementEstimator,
    ByContextAvailabilityEstimator,
    ByErrorTypeEstimator,
    DifficultyEstimator,
    OracleEstimator,
    RouteDecision,
    RoutingRepairer,
    RoutingStats,
    build_routing_repairer,
    make_estimator,
)

FAKE_KEY = "unit-test-key-0000-not-real"
DATA_ROOT = "datasets"


# ═══════════════════════════════════════════════════════════════════════════════
# Fake transport + network guard  (same shape as tests/test_llm_repair.py)
# ═══════════════════════════════════════════════════════════════════════════════


class FakeClient:
    """
    Stand-in for `LLMTransport`.

    Records every request body (deep-copied, so later mutation cannot hide a
    leak) and serves queued responses; the last queued item is reused, which
    keeps batch tests short.
    """

    name = "fake"

    def __init__(self, responses: Optional[List[Any]] = None) -> None:
        self.responses: List[Any] = list(responses or [])
        self.payloads: List[Dict[str, Any]] = []
        self.calls = 0

    def chat_completion(self, payload: Dict[str, Any], timeout_s: float) -> str:
        self.calls += 1
        self.payloads.append(copy.deepcopy(payload))
        if not self.responses:
            raise AssertionError("FakeClient ran out of queued responses")
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, BaseException):
            raise item
        return str(item)

    @property
    def payload_text(self) -> str:
        """Every captured request body as one JSON string, for leak scanning."""
        return json.dumps(self.payloads)


class _NetworkGuard:
    def __init__(self) -> None:
        self.transport_builds: List[str] = []
        self.http_attempts: List[str] = []


@pytest.fixture(autouse=True)
def network_guard(monkeypatch) -> _NetworkGuard:
    """Make any real transport construction or HTTP call fail loudly."""
    guard = _NetworkGuard()

    def _boom_transport(config):
        guard.transport_builds.append(repr(config))
        raise AssertionError("a real LLM transport was constructed during a test")

    def _boom_urlopen(*args, **kwargs):
        guard.http_attempts.append("urlopen")
        raise AssertionError("a real HTTP request was attempted during a test")

    monkeypatch.setattr(llm_repair, "_build_default_transport", _boom_transport)
    monkeypatch.setattr("urllib.request.urlopen", _boom_urlopen)
    monkeypatch.setattr(llm_repair.time, "sleep", lambda s: None)
    return guard


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers: contexts, stub repairers, accuracy
# ═══════════════════════════════════════════════════════════════════════════════


def _num_stats() -> NumericalStats:
    return NumericalStats(
        mean=5.0, std=2.0, min=1.0, q25=3.0, q50=5.0, q75=7.0, max=10.0, iqr=4.0
    )


def _ctx(
    row_id: int,
    error_type: str,
    dirty_value: Any = 33,
    column: str = "Bare Nuclei",
    row_context: Optional[Dict[str, Any]] = None,
    kind: ColumnKind = ColumnKind.NUMERICAL,
) -> RepairContext:
    """A RepairContext for an int64 numerical column."""
    return RepairContext(
        dataset="test_ds",
        column=column,
        kind=kind,
        dtype="int64",
        dirty_value=dirty_value,
        error_type=error_type,
        row_id=row_id,
        row_context={"Clump Thickness": 5, "Cell Size": 3}
        if row_context is None
        else row_context,
        vocabulary=[],
        column_stats={
            "min": 1.0, "max": 10.0, "median": 3.0,
            "q25": 1.0, "q75": 5.0, "iqr": 4.0,
        },
    )


class StubRepairer:
    """
    A repairer with a known, fixed accuracy profile.

    `correct_on` is the set of error types this repairer answers correctly; on
    anything else it returns `wrong_value`. `truth` maps (row_id, column) to
    the correct value. `top_probability` controls the candidate list's shape so
    the agreement estimators can be exercised independently of accuracy.
    """

    def __init__(
        self,
        name: str,
        correct_on: Sequence[str],
        truth: Dict[Tuple[int, str], Any],
        wrong_value: Any = -999,
        top_probability: float = 1.0,
        probability_by_row: Optional[Dict[int, float]] = None,
    ) -> None:
        self.name = name
        self.correct_on = set(correct_on)
        self.truth = dict(truth)
        self.wrong_value = wrong_value
        self.top_probability = float(top_probability)
        self.probability_by_row = dict(probability_by_row or {})
        self.seen: List[Tuple[int, str]] = []

    def repair(self, context: RepairContext) -> RepairResult:
        key = (int(context.row_id), str(context.column))
        self.seen.append(key)
        if context.error_type in self.correct_on and key in self.truth:
            value = self.truth[key]
        else:
            value = self.wrong_value
        p = self.probability_by_row.get(int(context.row_id), self.top_probability)
        candidates = [RepairCandidate(value=value, probability=p)]
        if p < 1.0:
            candidates.append(RepairCandidate(value=self.wrong_value, probability=1.0 - p))
        return RepairResult(
            dataset=context.dataset,
            row_id=context.row_id,
            column=context.column,
            dirty_value=context.dirty_value,
            error_type=context.error_type,
            candidates=candidates,
            repair_method=f"stub_{self.name}",
            context_used={"stub": self.name},
        )


class RaisingRepairer:
    """A repairer that always raises — used for the fallback test."""

    def __init__(self) -> None:
        self.calls = 0

    def repair(self, context: RepairContext) -> RepairResult:
        self.calls += 1
        raise RuntimeError("provider exploded")


def _accuracy(
    results: Sequence[RepairResult],
    truth: Dict[Tuple[int, str], Any],
    comparator: Optional[ValueComparator] = None,
) -> float:
    """Top-1 accuracy, computed here rather than by the code under test."""
    cmp_ = comparator or ValueComparator()
    if not results:
        return 0.0
    hits = 0
    for r in results:
        target = truth.get((int(r.row_id), str(r.column)))
        if r.candidates and cmp_.are_equal(r.top_candidate.value, target):
            hits += 1
    return hits / len(results)


def _run(
    estimator: DifficultyEstimator,
    contexts: Sequence[RepairContext],
    rule: Any,
    llm: Any,
) -> Tuple[RoutingRepairer, List[RepairResult]]:
    router = RoutingRepairer(estimator=estimator, rule=rule, llm=llm)
    return router, [router.repair(c) for c in contexts]


def _config(tmp_path: Optional[Path] = None, **overrides: Any) -> LLMConfig:
    kwargs: Dict[str, Any] = dict(
        api_key=FAKE_KEY,
        base_url="https://fake.invalid/v1",
        model="test-model",
        temperature=0.0,
        max_tokens=256,
        timeout_s=5.0,
        max_retries=0,
        retry_backoff_s=0.0,
        cache_dir=str(tmp_path / "llmcache") if tmp_path else ".cache/llm-tests",
    )
    kwargs.update(overrides)
    return LLMConfig(**kwargs)


def _resp(*pairs: Any, reasoning: str = "because") -> str:
    return json.dumps(
        {
            "candidates": [{"value": v, "probability": p} for v, p in pairs],
            "reasoning": reasoning,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# The synthetic routing world
# ─────────────────────────────────────────────────────────────────────────────
#
# 6 OUT_OF_RANGE cells (recoverable) and 4 MISSING cells (destroyed).
# The rule repairer is perfect on OUT_OF_RANGE and wrong on MISSING;
# the model is perfect on MISSING and wrong on OUT_OF_RANGE.
#
# Re-derived expectations, independent of the implementation:
#   always_rule     = 6/10 = 0.6
#   always_llm      = 4/10 = 0.4
#   by_error_type   =   10/10 = 1.0
#
N_OOR = 6
N_MISSING = 4


def _split_world() -> Tuple[List[RepairContext], Dict[Tuple[int, str], Any]]:
    contexts: List[RepairContext] = []
    truth: Dict[Tuple[int, str], Any] = {}
    for i in range(N_OOR):
        contexts.append(_ctx(i, ErrorType.OUT_OF_RANGE.value, dirty_value=33))
        truth[(i, "Bare Nuclei")] = 3
    for j in range(N_MISSING):
        row = 100 + j
        contexts.append(_ctx(row, ErrorType.MISSING.value, dirty_value=None))
        truth[(row, "Bare Nuclei")] = 7
    return contexts, truth


def _split_repairers(truth: Dict[Tuple[int, str], Any]) -> Tuple[StubRepairer, StubRepairer]:
    rule = StubRepairer("rule", [ErrorType.OUT_OF_RANGE.value], truth, wrong_value=-1)
    llm = StubRepairer("llm", [ErrorType.MISSING.value], truth, wrong_value=-2)
    return rule, llm


# ═══════════════════════════════════════════════════════════════════════════════
# The central hypothesis test
# ═══════════════════════════════════════════════════════════════════════════════


class TestRoutingBeatsBothPureStrategies:
    """
    If routing cannot win in a world built to its advantage, M1 is wrong.

    This is the mechanism-level test: it does not claim anything about real
    data (that is `experiments/w4_routing.py`), only that the router actually
    composes two complementary repairers instead of degenerating to one.
    """

    def test_by_error_type_beats_both_pure_strategies(self):
        contexts, truth = _split_world()
        n = len(contexts)

        rule, llm = _split_repairers(truth)
        _, routed = _run(ByErrorTypeEstimator(), contexts, rule, llm)
        _, pure_rule = _run(AlwaysRuleEstimator(), contexts, *_split_repairers(truth))
        _, pure_llm = _run(AlwaysLLMEstimator(), contexts, *_split_repairers(truth))

        acc_routed = _accuracy(routed, truth)
        acc_rule = _accuracy(pure_rule, truth)
        acc_llm = _accuracy(pure_llm, truth)

        # Independently re-derived, not read back from the implementation.
        assert acc_rule == pytest.approx(N_OOR / n)
        assert acc_llm == pytest.approx(N_MISSING / n)
        assert acc_routed == pytest.approx(1.0)

        assert acc_routed > acc_rule
        assert acc_routed > acc_llm

    def test_the_routes_are_the_ones_the_measurement_implies(self):
        contexts, truth = _split_world()
        rule, llm = _split_repairers(truth)

        router, _ = _run(ByErrorTypeEstimator(), contexts, rule, llm)

        assert router.stats.n_rule == N_OOR
        assert router.stats.n_llm == N_MISSING
        by_type = router.stats.routes_by_error_type
        assert by_type[ErrorType.OUT_OF_RANGE.value] == {ROUTE_RULE: N_OOR, ROUTE_LLM: 0}
        assert by_type[ErrorType.MISSING.value] == {ROUTE_RULE: 0, ROUTE_LLM: N_MISSING}

    def test_routing_cuts_model_calls_against_always_llm(self):
        contexts, truth = _split_world()

        routed, _ = _run(ByErrorTypeEstimator(), contexts, *_split_repairers(truth))
        pure, _ = _run(AlwaysLLMEstimator(), contexts, *_split_repairers(truth))

        assert pure.stats.llm_invocations == len(contexts)
        assert routed.stats.llm_invocations == N_MISSING
        assert routed.stats.llm_invocations < pure.stats.llm_invocations

    def test_always_rule_makes_no_model_call_at_all(self):
        contexts, truth = _split_world()
        rule, llm = _split_repairers(truth)

        router, _ = _run(AlwaysRuleEstimator(), contexts, rule, llm)

        assert router.stats.llm_invocations == 0
        assert llm.seen == []


# ═══════════════════════════════════════════════════════════════════════════════
# by_error_type partition
# ═══════════════════════════════════════════════════════════════════════════════


class TestByErrorType:
    @pytest.mark.parametrize("error_type", sorted(RECOVERABLE_ERROR_TYPES))
    def test_recoverable_types_stay_with_the_rule(self, error_type):
        decision = ByErrorTypeEstimator().route(_ctx(0, error_type))
        assert decision.route == ROUTE_RULE
        assert decision.reason == "recoverable_error_type"

    @pytest.mark.parametrize("error_type", sorted(AMBIGUOUS_ERROR_TYPES))
    def test_ambiguous_types_escalate(self, error_type):
        decision = ByErrorTypeEstimator().route(_ctx(0, error_type))
        assert decision.route == ROUTE_LLM
        assert decision.reason == "ambiguous_error_type"

    def test_the_two_sets_partition_the_detector_error_types(self):
        """Every ErrorType the detector can emit must be classified."""
        all_types = {e.value for e in ErrorType}
        assert RECOVERABLE_ERROR_TYPES | AMBIGUOUS_ERROR_TYPES == all_types
        assert not (RECOVERABLE_ERROR_TYPES & AMBIGUOUS_ERROR_TYPES)

    def test_an_unknown_error_type_escalates_conservatively(self):
        decision = ByErrorTypeEstimator().route(_ctx(0, "SOMETHING_NEW"))
        assert decision.route == ROUTE_LLM
        assert decision.reason == "unknown_error_type"

    def test_escalate_unknown_false_keeps_it_with_the_rule(self):
        decision = ByErrorTypeEstimator(escalate_unknown=False).route(
            _ctx(0, "SOMETHING_NEW")
        )
        assert decision.route == ROUTE_RULE


# ═══════════════════════════════════════════════════════════════════════════════
# by_candidate_agreement / by_candidate_margin
# ═══════════════════════════════════════════════════════════════════════════════


class TestByCandidateAgreement:
    def _mixed_confidence_world(self):
        """
        Six cells whose rule-side top probability is fixed per row.

        Threshold 0.5 ⇒ rows 1, 3 and 5 (0.20, 0.40, 0.49) must escalate and
        rows 0, 2 and 4 (0.90, 0.50, 0.75) must not. 0.50 is on the boundary
        and must stay (the rule is `score < threshold`).
        """
        probs = {0: 0.90, 1: 0.20, 2: 0.50, 3: 0.40, 4: 0.75, 5: 0.49}
        contexts = [_ctx(i, ErrorType.MISSING.value, dirty_value=None) for i in probs]
        truth = {(i, "Bare Nuclei"): 7 for i in probs}
        rule = StubRepairer(
            "rule", [ErrorType.MISSING.value], truth, wrong_value=-1,
            probability_by_row=probs,
        )
        llm = StubRepairer("llm", [ErrorType.MISSING.value], truth, wrong_value=-2)
        expected_escalated = {1, 3, 5}
        return contexts, rule, llm, expected_escalated

    def test_escalates_exactly_the_low_confidence_cells(self):
        contexts, rule, llm, expected = self._mixed_confidence_world()

        router, results = _run(
            ByCandidateAgreementEstimator(threshold=0.5), contexts, rule, llm
        )

        escalated: Set[int] = {
            int(r.row_id) for r in results if r.context_used["route"] == ROUTE_LLM
        }
        # The specific cells, not just the count.
        assert escalated == expected
        assert {k[0] for k in llm.seen} == expected
        assert router.stats.n_llm == len(expected)

    def test_a_cell_with_no_candidates_escalates(self):
        class Empty:
            def repair(self, context):
                return RepairResult(
                    dataset=context.dataset, row_id=context.row_id,
                    column=context.column, dirty_value=context.dirty_value,
                    error_type=context.error_type, candidates=[],
                    repair_method="empty",
                )

        truth = {(0, "Bare Nuclei"): 7}
        llm = StubRepairer("llm", [ErrorType.MISSING.value], truth)

        router, results = _run(
            ByCandidateAgreementEstimator(), [_ctx(0, ErrorType.MISSING.value)],
            Empty(), llm,
        )

        assert results[0].context_used["route_reason"] == "no_rule_candidates"
        assert router.stats.n_llm == 1

    def test_margin_mode_uses_the_top_two_gap_not_the_top_probability(self):
        """
        A candidate list of 0.45 / 0.40 / 0.15 is *unconfident* (top 0.45) but
        has a narrow margin (0.05). Threshold 0.5 on top-probability escalates
        it; threshold 0.2 on margin also escalates it. A list of 0.45 / 0.05
        has the same top probability and a wide margin, so the two modes must
        disagree on it — which is the whole point of having both.
        """
        narrow = RepairResult(
            dataset="d", row_id=0, column="c", dirty_value=1, error_type="MISSING",
            candidates=[
                RepairCandidate(9, 0.45), RepairCandidate(8, 0.40), RepairCandidate(7, 0.15),
            ],
            repair_method="x",
        )
        wide = RepairResult(
            dataset="d", row_id=0, column="c", dirty_value=1, error_type="MISSING",
            candidates=[RepairCandidate(9, 0.45), RepairCandidate(8, 0.05)],
            repair_method="x",
        )
        by_prob = ByCandidateAgreementEstimator(threshold=0.5, mode="top_probability")
        by_margin = ByCandidateAgreementEstimator(threshold=0.2, mode="margin")
        ctx = _ctx(0, ErrorType.MISSING.value)

        assert by_prob.score(narrow) == pytest.approx(0.45)
        assert by_margin.score(narrow) == pytest.approx(0.05)
        assert by_margin.score(wide) == pytest.approx(0.40)

        assert by_prob.route(ctx, narrow).route == ROUTE_LLM
        assert by_margin.route(ctx, narrow).route == ROUTE_LLM
        assert by_prob.route(ctx, wide).route == ROUTE_LLM      # still unconfident
        assert by_margin.route(ctx, wide).route == ROUTE_RULE   # but decided

    def test_single_candidate_has_the_maximum_margin(self):
        single = RepairResult(
            dataset="d", row_id=0, column="c", dirty_value=1, error_type="MISSING",
            candidates=[RepairCandidate(9, 1.0)], repair_method="x",
        )
        est = ByCandidateAgreementEstimator(threshold=0.2, mode="margin")
        assert est.score(single) == pytest.approx(1.0)
        assert est.route(_ctx(0, ErrorType.MISSING.value), single).route == ROUTE_RULE

    def test_score_is_none_without_a_rule_result(self):
        assert ByCandidateAgreementEstimator().score(None) is None

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_invalid_threshold_rejected(self, bad):
        with pytest.raises(ValueError, match="threshold"):
            ByCandidateAgreementEstimator(threshold=bad)

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="mode"):
            ByCandidateAgreementEstimator(mode="vibes")

    def test_the_rule_repairer_is_not_run_twice_for_a_kept_cell(self):
        truth = {(0, "Bare Nuclei"): 7}
        rule = StubRepairer("rule", [ErrorType.MISSING.value], truth, top_probability=0.9)
        llm = StubRepairer("llm", [ErrorType.MISSING.value], truth)

        _run(ByCandidateAgreementEstimator(threshold=0.5),
             [_ctx(0, ErrorType.MISSING.value)], rule, llm)

        assert len(rule.seen) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# by_context_availability
# ═══════════════════════════════════════════════════════════════════════════════


class TestByContextAvailability:
    def test_escalates_when_the_row_has_context(self):
        ctx = _ctx(0, ErrorType.MISSING.value, row_context={"a": 1, "b": 2})
        decision = ByContextAvailabilityEstimator(min_known_fraction=0.5).route(ctx)
        assert decision.route == ROUTE_LLM
        assert decision.reason == "context_available"

    def test_keeps_a_sparse_row_with_the_rule(self):
        ctx = _ctx(0, ErrorType.MISSING.value, row_context={"a": None, "b": np.nan, "c": 1})
        est = ByContextAvailabilityEstimator(min_known_fraction=0.5)
        assert est.known_fraction(ctx) == pytest.approx(1 / 3)
        assert est.route(ctx).route == ROUTE_RULE

    def test_a_row_with_no_context_columns_scores_zero(self):
        ctx = _ctx(0, ErrorType.MISSING.value, row_context={})
        est = ByContextAvailabilityEstimator(min_known_fraction=0.01)
        assert est.known_fraction(ctx) == 0.0
        assert est.route(ctx).route == ROUTE_RULE

    def test_blank_strings_count_as_missing_context(self):
        ctx = _ctx(0, ErrorType.MISSING.value, row_context={"a": "   ", "b": "x"})
        assert ByContextAvailabilityEstimator().known_fraction(ctx) == pytest.approx(0.5)

    @pytest.mark.parametrize("bad", [-0.5, 2.0])
    def test_invalid_fraction_rejected(self, bad):
        with pytest.raises(ValueError, match="min_known_fraction"):
            ByContextAvailabilityEstimator(min_known_fraction=bad)


# ═══════════════════════════════════════════════════════════════════════════════
# oracle — the ceiling
# ═══════════════════════════════════════════════════════════════════════════════


class TestOracleIsAnUpperBound:
    def _world(self):
        """
        Ten cells. The rule is right on 6, the model on 4, and the sets are
        disjoint, so a perfect router scores 10/10 while each pure strategy
        scores its own share.
        """
        return _split_world()

    def test_oracle_matches_every_other_estimator_or_beats_it(self):
        contexts, truth = self._world()
        oracle_acc = None
        accuracies: Dict[str, float] = {}

        for name in ESTIMATOR_NAMES:
            rule, llm = _split_repairers(truth)
            _, results = _run(make_estimator(name), contexts, rule, llm)
            accuracies[name] = _accuracy(results, truth)

        rule, llm = _split_repairers(truth)
        _, results = _run(OracleEstimator(ground_truth=truth), contexts, rule, llm)
        oracle_acc = _accuracy(results, truth)

        assert oracle_acc == pytest.approx(1.0)
        for name, acc in accuracies.items():
            assert oracle_acc >= acc - 1e-12, f"{name} ({acc}) beat the ceiling"

    def test_oracle_prefers_the_cheap_repairer_on_a_tie(self):
        """Both correct ⇒ the rule answers, so the ceiling's cost is honest."""
        truth = {(0, "Bare Nuclei"): 3}
        rule = StubRepairer("rule", [ErrorType.OUT_OF_RANGE.value], truth)
        llm = StubRepairer("llm", [ErrorType.OUT_OF_RANGE.value], truth)

        router, results = _run(
            OracleEstimator(ground_truth=truth),
            [_ctx(0, ErrorType.OUT_OF_RANGE.value)], rule, llm,
        )

        assert results[0].context_used["route"] == ROUTE_RULE
        assert results[0].context_used["route_reason"] == "rule_correct"
        assert router.stats.n_llm == 0

    def test_oracle_routes_to_the_rule_when_neither_is_correct(self):
        truth = {(0, "Bare Nuclei"): 3}
        rule = StubRepairer("rule", [], truth, wrong_value=-1)
        llm = StubRepairer("llm", [], truth, wrong_value=-2)

        _, results = _run(
            OracleEstimator(ground_truth=truth),
            [_ctx(0, ErrorType.OUT_OF_RANGE.value)], rule, llm,
        )

        assert results[0].context_used["route_reason"] == "neither_correct"
        assert results[0].top_candidate.value == -1

    def test_oracle_without_a_label_for_the_cell_keeps_the_rule(self):
        rule = StubRepairer("rule", [], {}, wrong_value=-1)
        llm = StubRepairer("llm", [], {}, wrong_value=-2)

        _, results = _run(
            OracleEstimator(ground_truth={}),
            [_ctx(0, ErrorType.OUT_OF_RANGE.value)], rule, llm,
        )

        assert results[0].context_used["route_reason"] == "no_ground_truth"

    def test_oracle_calls_the_model_on_every_cell_so_its_cost_is_not_a_saving(self):
        """
        The ceiling needs both answers before it can choose, so it is also the
        most expensive option. A table must not present it as cheap.
        """
        contexts, truth = self._world()
        rule, llm = _split_repairers(truth)

        router, _ = _run(OracleEstimator(ground_truth=truth), contexts, rule, llm)

        assert router.stats.llm_invocations == len(contexts)
        assert router.stats.n_llm < router.stats.llm_invocations

    def test_oracle_is_flagged_as_using_ground_truth(self):
        truth = {(0, "Bare Nuclei"): 3}
        rule, llm = _split_repairers(truth)
        router = RoutingRepairer(
            estimator=OracleEstimator(ground_truth=truth), rule=rule, llm=llm
        )

        assert router.uses_ground_truth is True
        assert "UPPER BOUND" in OracleEstimator(ground_truth=truth).describe()

    def test_oracle_cannot_be_built_from_the_registry_by_name(self):
        assert "oracle" not in ESTIMATORS
        with pytest.raises(ValueError, match="upper bound"):
            make_estimator("oracle")


# ═══════════════════════════════════════════════════════════════════════════════
# Pure controls reproduce the pure repairers
# ═══════════════════════════════════════════════════════════════════════════════


class TestPureControlsReproduceThePureRepairers:
    """
    `always_rule` / `always_llm` must be the unmodified repairers plus a label.

    Candidate values and probabilities are compared exactly; `repair_method`
    is expected to differ (that is the whole point of the provenance field)
    and the original value is asserted to survive in
    `context_used["inner_method"]`.
    """

    def _contexts(self) -> List[RepairContext]:
        return [
            _ctx(0, ErrorType.OUT_OF_RANGE.value, dirty_value=33),
            _ctx(1, ErrorType.MISSING.value, dirty_value=None),
            _ctx(2, ErrorType.FORMAT_ERROR.value, dirty_value="1a0"),
        ]

    @staticmethod
    def _identical(a: RepairResult, b: RepairResult) -> bool:
        return (
            a.dataset == b.dataset
            and a.row_id == b.row_id
            and a.column == b.column
            and str(a.dirty_value) == str(b.dirty_value)
            and a.error_type == b.error_type
            and [(c.value, c.probability) for c in a.candidates]
            == [(c.value, c.probability) for c in b.candidates]
        )

    def test_always_rule_reproduces_the_mock_repairer(self):
        contexts = self._contexts()
        pure = MockRepairer(seed=42)
        expected = [MockRepairer(seed=42).repair(c) for c in contexts]

        router = RoutingRepairer(estimator=AlwaysRuleEstimator(), rule=pure)
        actual = [router.repair(c) for c in contexts]

        for got, want in zip(actual, expected):
            assert self._identical(got, want)
            assert got.repair_method == "routed:rule"
            assert got.context_used["inner_method"] == want.repair_method

    def test_always_llm_reproduces_the_llm_repairer(self, tmp_path):
        contexts = self._contexts()
        responses = [_resp((3, 0.7), (4, 0.3))]

        pure_client = FakeClient(list(responses))
        pure = LLMRepairer(config=_config(tmp_path), client=pure_client, use_cache=False)
        expected = [pure.repair(c) for c in contexts]

        routed_client = FakeClient(list(responses))
        routed_llm = LLMRepairer(
            config=_config(tmp_path), client=routed_client, use_cache=False
        )
        router = RoutingRepairer(
            estimator=AlwaysLLMEstimator(), rule=MockRepairer(seed=42), llm=routed_llm
        )
        actual = [router.repair(c) for c in contexts]

        assert routed_client.calls == pure_client.calls == len(contexts)
        for got, want in zip(actual, expected):
            assert self._identical(got, want)
            assert got.repair_method == "routed:llm"
            assert got.context_used["inner_method"] == "llm"

    def test_always_rule_needs_no_model_repairer_at_all(self):
        router = RoutingRepairer(estimator=AlwaysRuleEstimator(), rule=MockRepairer())
        assert router.llm is None
        result = router.repair(_ctx(0, ErrorType.OUT_OF_RANGE.value))
        assert result.repair_method == "routed:rule"

    def test_an_escalating_estimator_without_a_model_repairer_is_refused(self):
        for estimator in (
            AlwaysLLMEstimator(),
            ByErrorTypeEstimator(),
            ByCandidateAgreementEstimator(),
            ByContextAvailabilityEstimator(),
        ):
            with pytest.raises(ValueError, match="llm"):
                RoutingRepairer(estimator=estimator, rule=MockRepairer())

    def test_estimator_is_required(self):
        with pytest.raises(ValueError, match="estimator"):
            RoutingRepairer(estimator=None, rule=MockRepairer())


# ═══════════════════════════════════════════════════════════════════════════════
# Interface compatibility: a drop-in for the other repairers
# ═══════════════════════════════════════════════════════════════════════════════


class TestDropInInterface:
    def test_repair_signature_matches_the_mock_repairer(self):
        assert (
            list(inspect.signature(RoutingRepairer.repair).parameters)
            == list(inspect.signature(MockRepairer.repair).parameters)
        )

    def test_repair_batch_signature_matches_the_mock_repairer(self):
        assert (
            list(inspect.signature(RoutingRepairer.repair_batch).parameters)
            == list(inspect.signature(MockRepairer.repair_batch).parameters)
        )

    def test_repair_batch_returns_one_result_per_record_in_order(self):
        loader = DataLoader(DATA_ROOT)
        clean_df, dirty_df = loader.load("breast_cancer")
        profile = profile_dataset(
            "breast_cancer", loader.discover("breast_cancer"), clean_df, dirty_df
        )
        records = Detector(profile).detect(dirty_df)[:25]
        truth = {}
        rule = MockRepairer(seed=42)
        llm = StubRepairer("llm", [], truth, wrong_value=0)

        router = RoutingRepairer(estimator=ByErrorTypeEstimator(), rule=rule, llm=llm)
        results = router.repair_batch(records, dirty_df, profile)

        assert len(results) == len(records)
        assert [(r.row_id, r.column) for r in results] == [
            (d.row_id, d.column) for d in records
        ]
        assert router.stats.cells == len(records)
        assert router.stats.n_rule + router.stats.n_llm == len(records)

    def test_empty_input_returns_an_empty_list(self):
        loader = DataLoader(DATA_ROOT)
        clean_df, dirty_df = loader.load("breast_cancer")
        profile = profile_dataset(
            "breast_cancer", loader.discover("breast_cancer"), clean_df, dirty_df
        )
        router = RoutingRepairer(
            estimator=ByErrorTypeEstimator(),
            rule=MockRepairer(seed=42),
            llm=StubRepairer("llm", [], {}),
        )

        assert router.repair_batch([], dirty_df, profile) == []
        assert router.stats.cells == 0
        assert router.stats.llm_fraction == 0.0

    def test_build_routing_repairer_builds_from_a_name(self):
        router = build_routing_repairer(
            "by_error_type", rule=MockRepairer(seed=1), llm=StubRepairer("llm", [], {})
        )
        assert router.estimator.name == "by_error_type"

    def test_unknown_estimator_name_rejected(self):
        with pytest.raises(ValueError, match="Unknown estimator"):
            make_estimator("telepathy")

    def test_every_registered_name_builds_with_no_arguments(self):
        for name in ESTIMATOR_NAMES:
            estimator = make_estimator(name)
            assert estimator.name == name
            assert estimator.uses_ground_truth is False

    def test_by_candidate_margin_is_the_margin_mode(self):
        estimator = make_estimator("by_candidate_margin")
        assert estimator.mode == "margin"
        assert estimator.name == "by_candidate_margin"


# ═══════════════════════════════════════════════════════════════════════════════
# Determinism, failure handling, provenance
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    def test_two_runs_with_the_same_seed_are_identical(self):
        contexts, truth = _split_world()

        def once():
            rule = MockRepairer(seed=42)
            llm = StubRepairer("llm", [ErrorType.MISSING.value], truth)
            router = RoutingRepairer(
                estimator=ByErrorTypeEstimator(), rule=rule, llm=llm
            )
            results = [router.repair(c) for c in contexts]
            return [
                (r.repair_method, [(str(c.value), c.probability) for c in r.candidates])
                for r in results
            ], router.stats.to_dict()

        first, first_stats = once()
        second, second_stats = once()

        assert first == second
        assert first_stats == second_stats

    def test_a_different_seed_changes_the_rule_side_candidates(self):
        """Negative control: the determinism test above is not vacuous."""
        ctx = _ctx(0, ErrorType.MISSING.value, dirty_value=None)

        a = RoutingRepairer(
            estimator=AlwaysRuleEstimator(), rule=MockRepairer(seed=1)
        ).repair(ctx)
        b = RoutingRepairer(
            estimator=AlwaysRuleEstimator(), rule=MockRepairer(seed=9999)
        ).repair(ctx)

        assert [c.probability for c in a.candidates] != [c.probability for c in b.candidates]


class TestModelFailureFallsBackToTheRule:
    def test_a_raising_model_repairer_falls_back_for_that_cell(self):
        ctx = _ctx(0, ErrorType.MISSING.value, dirty_value=None)
        rule = MockRepairer(seed=42)
        expected = MockRepairer(seed=42).repair(ctx)
        raiser = RaisingRepairer()

        router = RoutingRepairer(
            estimator=ByErrorTypeEstimator(), rule=rule, llm=raiser
        )
        result = router.repair(ctx)

        assert raiser.calls == 1
        assert router.stats.llm_failures == 1
        assert router.stats.n_rule == 1
        assert router.stats.n_llm == 0
        assert result.repair_method == "routed:rule"
        assert result.context_used["route_reason"].endswith("+llm_failed")
        assert [(c.value, c.probability) for c in result.candidates] == [
            (c.value, c.probability) for c in expected.candidates
        ]

    def test_the_batch_completes_despite_a_failing_model(self):
        contexts, truth = _split_world()
        router = RoutingRepairer(
            estimator=AlwaysLLMEstimator(),
            rule=MockRepairer(seed=42),
            llm=RaisingRepairer(),
        )

        results = [router.repair(c) for c in contexts]

        assert len(results) == len(contexts)
        assert router.stats.llm_failures == len(contexts)
        assert all(r.repair_method == "routed:rule" for r in results)

    def test_an_llm_repairers_own_fallback_is_still_visible(self, tmp_path):
        """An LLMRepairer that fell back internally keeps saying so."""
        client = FakeClient(["not json at all"])
        llm = LLMRepairer(config=_config(tmp_path), client=client, use_cache=False)
        router = RoutingRepairer(
            estimator=AlwaysLLMEstimator(), rule=MockRepairer(seed=42), llm=llm
        )

        result = router.repair(_ctx(0, ErrorType.MISSING.value, dirty_value=None))

        assert result.repair_method == "routed:llm"
        assert result.context_used["inner_method"].startswith("llm->mock")
        assert router.stats.llm_failures == 0  # it did not raise


class TestProvenanceAndStats:
    def test_repair_method_names_the_route(self):
        contexts, truth = _split_world()
        rule, llm = _split_repairers(truth)

        _, results = _run(ByErrorTypeEstimator(), contexts, rule, llm)

        methods = {r.repair_method for r in results}
        assert methods == {"routed:rule", "routed:llm"}

    def test_context_used_carries_route_reason_and_inner_method(self):
        truth = {(0, "Bare Nuclei"): 3}
        rule, llm = _split_repairers(truth)

        _, results = _run(
            ByErrorTypeEstimator(), [_ctx(0, ErrorType.OUT_OF_RANGE.value)], rule, llm
        )

        used = results[0].context_used
        assert used["route"] == ROUTE_RULE
        assert used["route_reason"] == "recoverable_error_type"
        assert used["inner_method"] == "stub_rule"
        assert used["stub"] == "rule"  # the inner repairer's own context survives

    def test_stats_to_dict_is_serialisable_and_complete(self):
        contexts, truth = _split_world()
        rule, llm = _split_repairers(truth)
        router, _ = _run(ByErrorTypeEstimator(), contexts, rule, llm)

        payload = router.stats.to_dict()
        json.dumps(payload)  # must not raise

        assert payload["cells"] == len(contexts)
        assert payload["n_rule"] + payload["n_llm"] == payload["cells"]
        assert payload["llm_fraction"] == pytest.approx(N_MISSING / len(contexts))
        assert sum(payload["reasons"].values()) == payload["cells"]

    def test_reset_stats_zeroes_the_counters(self):
        contexts, truth = _split_world()
        rule, llm = _split_repairers(truth)
        router, _ = _run(ByErrorTypeEstimator(), contexts, rule, llm)

        router.reset_stats()

        assert router.stats.to_dict()["cells"] == 0
        assert router.stats.reasons == {}

    def test_route_decision_rejects_an_unknown_route(self):
        with pytest.raises(ValueError, match="route must be one of"):
            RouteDecision("telepathy", "why not")

    def test_stats_llm_fraction_is_zero_with_no_cells(self):
        assert RoutingStats().llm_fraction == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Leakage (CLAUDE.md §3)
# ═══════════════════════════════════════════════════════════════════════════════


def _leak_dirty_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25.0, 999.0, 31.0],
        "job": ["Weldrr", "Welder", "Baker"],
        "note": ["note_row0_ok", "note_row1_ok", "ROW2_NOTE_SENTINEL"],
    })


def _leak_clean_df() -> pd.DataFrame:
    """Ground truth whose every cell is deliberately disjoint from the dirty
    cell, so "no clean value appears" is an exact assertion rather than a
    vacuous one."""
    return pd.DataFrame({
        "age": [404401, 404402, 404403],
        "job": ["CLEANJOB_ALPHA", "CLEANJOB_BETA", "CLEANJOB_GAMMA"],
        "note": ["CLEANNOTE_ALPHA", "CLEANNOTE_BETA", "CLEANNOTE_GAMMA"],
    })


def _leak_profile() -> DatasetProfile:
    columns = {
        "age": ColumnProfile(
            name="age", kind=ColumnKind.NUMERICAL, dtype_clean="int64",
            dtype_dirty="float64", missing_clean=0, missing_dirty=0,
            n_unique_clean=3, n_unique_dirty=3, numerical_stats=_num_stats(),
        ),
        "job": ColumnProfile(
            name="job", kind=ColumnKind.CATEGORICAL, dtype_clean="object",
            dtype_dirty="object", missing_clean=0, missing_dirty=0,
            n_unique_clean=3, n_unique_dirty=3,
            categorical_stats=CategoricalStats(
                value_counts={"Welder": 1, "Baker": 1}, top_values=["Welder"], n_rare=0,
            ),
        ),
        "note": ColumnProfile(
            name="note", kind=ColumnKind.CATEGORICAL, dtype_clean="object",
            dtype_dirty="object", missing_clean=0, missing_dirty=0,
            n_unique_clean=3, n_unique_dirty=3,
            categorical_stats=CategoricalStats(
                value_counts={"note_row0_ok": 1}, top_values=["note_row0_ok"], n_rare=0,
            ),
        ),
    }
    return DatasetProfile(
        name="leak_ds",
        clean_path=Path("datasets/leak_ds/clean.csv"),
        dirty_path=Path("datasets/leak_ds/dirty.csv"),
        n_rows=3, n_cols=3, column_names=list(columns), columns=columns,
    )


class TestLeakage:
    def test_no_router_parameter_is_named_after_clean_data(self):
        for func in (
            RoutingRepairer.__init__,
            RoutingRepairer.repair,
            RoutingRepairer.repair_batch,
            build_routing_repairer,
            make_estimator,
        ):
            for param in inspect.signature(func).parameters:
                assert "clean" not in param.lower(), f"{func.__name__}({param})"

    def test_no_non_oracle_estimator_takes_clean_data(self):
        for name in ESTIMATOR_NAMES:
            estimator = make_estimator(name)
            assert estimator.uses_ground_truth is False
            for func in (type(estimator).__init__, type(estimator).route):
                for param in inspect.signature(func).parameters:
                    assert "clean" not in param.lower(), f"{name}({param})"

    def test_the_oracle_is_the_only_ground_truth_consumer(self):
        truth = {(0, "Bare Nuclei"): 3}
        assert OracleEstimator(ground_truth=truth).uses_ground_truth is True
        assert all(
            make_estimator(n).uses_ground_truth is False for n in ESTIMATOR_NAMES
        )

    def test_routes_are_identical_when_clean_values_are_scrambled(self, tmp_path):
        """
        Scramble the clean values with a within-column permutation: column
        statistics (the only clean-derived information the profiler may use)
        are unchanged, so every routing decision must be bit-identical. If the
        router ever consulted a per-cell clean value this fails.
        """
        src = Path(DATA_ROOT) / "breast_cancer"
        dst = tmp_path / "breast_cancer"
        dst.mkdir()
        clean = pd.read_csv(src / "clean.csv")
        rng = np.random.default_rng(0)
        scrambled = clean.copy()
        for col in scrambled.columns:
            scrambled[col] = scrambled[col].to_numpy()[rng.permutation(len(scrambled))]
        scrambled.to_csv(dst / "clean.csv", index=False)
        (dst / "dirty.csv").write_bytes((src / "dirty.csv").read_bytes())

        def routes(data_root) -> Dict[str, List[Tuple[int, str, str, str]]]:
            loader = DataLoader(data_root)
            clean_df, dirty_df = loader.load("breast_cancer")
            profile = profile_dataset(
                "breast_cancer", loader.discover("breast_cancer"), clean_df, dirty_df
            )
            records = Detector(profile).detect(dirty_df)[:80]
            out: Dict[str, List[Tuple[int, str, str, str]]] = {}
            for name in ESTIMATOR_NAMES:
                router = RoutingRepairer(
                    estimator=make_estimator(name),
                    rule=MockRepairer(seed=42),
                    llm=StubRepairer("llm", [], {}, wrong_value=0),
                )
                results = router.repair_batch(records, dirty_df, profile)
                out[name] = [
                    (
                        r.row_id, r.column,
                        r.context_used["route"], r.context_used["route_reason"],
                    )
                    for r in results
                ]
            return out

        real = routes(DATA_ROOT)
        fake = routes(tmp_path)

        assert real == fake
        # Positive control: at least one estimator did split the cells, so the
        # equality above is not between two constant lists.
        assert any(
            len({r[2] for r in rows}) > 1 for rows in real.values()
        ), "no estimator produced a mixed route; the invariance test is vacuous"

    def test_no_clean_value_reaches_the_model_request_body(self, tmp_path):
        client = FakeClient([_resp(("Welder", 1.0))])
        llm = LLMRepairer(config=_config(tmp_path), client=client, use_cache=False)
        router = RoutingRepairer(
            estimator=AlwaysLLMEstimator(), rule=MockRepairer(seed=42), llm=llm
        )
        record = DetectionRecord(
            dataset="leak_ds", row_id=0, column="job", dirty_value="Weldrr",
            error_type=ErrorType.NOVEL_CATEGORY.value, detection_score=0.9,
            detection_method="novel_category", reason="not in vocabulary",
        )

        router.repair_batch([record], _leak_dirty_df(), _leak_profile())

        assert client.calls == 1
        body = client.payload_text
        clean_df = _leak_clean_df()
        for column in clean_df.columns:
            for value in clean_df[column]:
                assert str(value) not in body, f"clean value {value!r} leaked"
        # Positive control: the permitted dirty context IS present, so the
        # assertions above cannot pass on an empty payload.
        assert "Weldrr" in body
        assert "note_row0_ok" in body
        assert "ROW2_NOTE_SENTINEL" not in body

    def test_repair_results_carry_no_clean_field(self):
        truth = {(0, "Bare Nuclei"): 3}
        rule, llm = _split_repairers(truth)
        _, results = _run(
            ByErrorTypeEstimator(), [_ctx(0, ErrorType.OUT_OF_RANGE.value)], rule, llm
        )

        assert not any(
            "clean" in f.lower() for f in results[0].__dataclass_fields__
        )
        assert not any("clean" in str(k).lower() for k in results[0].context_used)


# ═══════════════════════════════════════════════════════════════════════════════
# No network
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoNetwork:
    def test_the_guard_blocks_a_real_transport(self, network_guard):
        with pytest.raises(AssertionError, match="real LLM transport"):
            llm_repair._build_default_transport(_config())
        assert network_guard.transport_builds

    def test_the_guard_blocks_urlopen(self, network_guard):
        import urllib.request

        with pytest.raises(AssertionError, match="real HTTP request"):
            urllib.request.urlopen("https://example.invalid")
        assert network_guard.http_attempts

    def test_routing_a_full_batch_opens_no_socket(self, tmp_path):
        client = FakeClient([_resp((3, 1.0))])
        llm = LLMRepairer(config=_config(tmp_path), client=client, use_cache=False)
        contexts, truth = _split_world()
        router = RoutingRepairer(
            estimator=ByErrorTypeEstimator(), rule=MockRepairer(seed=42), llm=llm
        )

        results = [router.repair(c) for c in contexts]

        assert len(results) == len(contexts)
        assert client.calls == N_MISSING
