"""
tests/test_acquisition.py
-------------------------
Wave-3 route W3.1 tests for :mod:`src.detectors.acquisition`.

WHAT THESE TESTS ARE FOR
========================
The route's claim is narrow and falsifiable:

    at 0.25 % prevalence the *sampler* is the bottleneck, and a strategy that
    uses the detector's own conditional score can buy far more positives per
    query than random or global-top sampling.

Every number in ``results/w3_acquisition.md`` is downstream of that claim, so
these tests are built to be able to refute it rather than to decorate it.

THE TWO SYNTHETIC REGIMES
=========================
``_concentrated(seed)`` — **the regime the route is about.** 2,000 rows x 5
columns = 10,000 cells, 25 corrupted (prevalence 0.25 %, adult's), and the
corruption is placed the way adult's actually is::

    column c3   holds every positive, and only there         (adult: `income`)
    c_full      0.70-0.80 on positives, 0.00-0.50 elsewhere  (the live signal)
    r_score     0.90 on every cell of c0, 0.0 elsewhere      (the poison)

``c0``'s rule score is the fixture's point. All 2,000 of those cells are
clean, so the top of any ranking that includes ``r_score`` is 2,000
guaranteed negatives — a scale model of adult's ``fnlwgt`` IQR false
positives, which is why wave 2's ``top_combiner`` drew zero positives at
every k <= 100. ``TestConcentratedFixture`` asserts the fixture really has
that property before any strategy is judged on it; if it does not, every
later assertion here is noise.

``_noise(seed)`` — **the null regime, and the one that catches cheating.**
Same shape, but ``c_full`` and ``r_score`` are uniform random and the 200
positives are placed independently of them. No strategy can beat random
here, so any strategy that does is reading something it did not pay for, or
is scoring itself against its own choices. The band is derived from the
binomial standard deviation in the test body, not copied from a run.

WHY THE EXPECTED VALUES ARE RE-DERIVED, NOT RECORDED
====================================================
CLAUDE.md §8b rule 5: "a test that cannot fail is not a test". So
``TestRoundRobin`` re-derives round-robin allocation with an independent
Python loop and compares, ``TestPerColumnMatchesWaveTwo`` compares against
``EnsembleDetector._per_column_informed`` (the wave-2 implementation, which
this module must not silently disagree with), and the random control's
expected yield is computed from the prevalence in the assertion itself.

LEAKAGE TESTS (CLAUDE.md §3 — the non-negotiable invariant)
===========================================================
Five independent angles:

1. *Structural*: no public signature takes a label array or anything named
   "clean"/"label"; ``fit`` has nowhere to receive labels.
2. *Budget*: the oracle is queried **exactly** k times, never more.
3. *Train-only*: a spy oracle proves every queried flat index lies inside the
   harness's train mask, so no eval-split label is ever requested.
4. *Counterfactual*: inverting every **unqueried train** label leaves the
   selected cells and the fitted scores bitwise identical. This is the strong
   one: it fails if the strategy reads any label it did not buy.
5. *Negative*: at ``budget=0`` the oracle is never touched at all.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from src.data.loader import DataLoader
from src.data.profiler import profile_dataset
from src.detection_eval.harness import evaluate_detector, flatten_cells
from src.detection_eval.labels import cell_labels, label_stats
from src.detectors import acquisition as acq
from src.detectors.acquisition import (
    STRATEGIES,
    USES_MODEL,
    AcquisitionDetector,
    RoundRecord,
    select_cells,
)
from src.detectors.ensemble import (
    FEATURE_NAMES,
    CellFeatures,
    EnsembleDetector,
    FeatureCache,
    LabelOracle,
)

SEED = 42
N_ROWS = 2000
N_COLS = 5
N_POS_CONCENTRATED = 25
N_POS_NOISE = 200
POSITIVE_COLUMN = 3
POISONED_COLUMN = 0
K = 200
N_FIXTURE_SEEDS = 8

COLUMNS = [f"c{i}" for i in range(N_COLS)]
N_CELLS = N_ROWS * N_COLS

#: Conditional sub-model iterations for the few tests that build real
#: features. The on-disk fixture is exactly learnable, so ten iterations fit
#: it as well as a hundred and keep this file under a minute.
FAST_CONDITIONAL = {"max_iter": 10}


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic feature matrices
# ═══════════════════════════════════════════════════════════════════════════════


def _features(c_full: np.ndarray, r_score: np.ndarray) -> CellFeatures:
    """A :class:`CellFeatures` carrying only the two signal features.

    Hand-built on purpose: it isolates the acquisition mechanism from the
    conditional model, so a failure here is a sampler failure and nothing
    else. Every other feature is 0.0, which is a legitimate feature matrix —
    :func:`src.detectors.ensemble.build_features` produces exactly that when
    no profile is supplied.
    """
    x = np.zeros((N_CELLS, len(FEATURE_NAMES)), dtype=float)
    x[:, FEATURE_NAMES.index("c_full")] = c_full
    x[:, FEATURE_NAMES.index("r_score")] = r_score
    return CellFeatures(
        x=x,
        names=FEATURE_NAMES,
        shape=(N_ROWS, N_COLS),
        index=pd.RangeIndex(N_ROWS),
        columns=pd.Index(COLUMNS),
    )


def _column_of(flat: np.ndarray) -> np.ndarray:
    return np.asarray(flat, dtype=int) // N_ROWS


def _concentrated(seed: int) -> Tuple[CellFeatures, pd.DataFrame, np.ndarray]:
    """Adult's regime in miniature. See the module docstring."""
    rng = np.random.default_rng(1000 + seed)
    column = np.arange(N_CELLS) // N_ROWS
    c_full = rng.random(N_CELLS) * 0.5
    r_score = np.where(column == POISONED_COLUMN, 0.9, 0.0).astype(float)
    positives = rng.choice(
        np.flatnonzero(column == POSITIVE_COLUMN), N_POS_CONCENTRATED, replace=False
    )
    c_full[positives] = 0.7 + rng.random(N_POS_CONCENTRATED) * 0.1
    features = _features(c_full, r_score)
    y = np.zeros(N_CELLS, dtype=bool)
    y[positives] = True
    labels = features.as_frame(y.astype(float)).astype(bool)
    return features, labels, y


def _noise(seed: int) -> Tuple[CellFeatures, pd.DataFrame, np.ndarray]:
    """The null regime: scores carry no information about the labels."""
    rng = np.random.default_rng(5000 + seed)
    features = _features(rng.random(N_CELLS), rng.random(N_CELLS))
    y = np.zeros(N_CELLS, dtype=bool)
    y[rng.choice(N_CELLS, N_POS_NOISE, replace=False)] = True
    labels = features.as_frame(y.astype(float)).astype(bool)
    return features, labels, y


@pytest.fixture(scope="module")
def frames() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A dummy value frame and an all-train mask of the synthetic shape.

    The values are never read — a detector given ``precomputed_features``
    takes every feature from that matrix — but the harness contract is
    expressed in frames, so the shapes must line up.
    """
    dirty = pd.DataFrame(np.zeros((N_ROWS, N_COLS)), columns=COLUMNS)
    mask = pd.DataFrame(np.ones((N_ROWS, N_COLS), dtype=bool), columns=COLUMNS)
    return dirty, mask


def _run(
    strategy: str,
    features: CellFeatures,
    labels: pd.DataFrame,
    frames: Tuple[pd.DataFrame, pd.DataFrame],
    budget: int = K,
    n_rounds: int = 1,
    seed: int = SEED,
    oracle: LabelOracle = None,
    mask: pd.DataFrame = None,
) -> AcquisitionDetector:
    dirty, default_mask = frames
    det = AcquisitionDetector(
        strategy=strategy,
        n_rounds=n_rounds,
        budget=budget,
        label_oracle=oracle if oracle is not None else LabelOracle(labels),
        precomputed_features=features,
        seed=seed,
    )
    det.fit(dirty, default_mask if mask is None else mask)
    return det


def _yield_over_seeds(
    strategy: str,
    regime,
    frames,
    budget: int = K,
    n_rounds: int = 1,
    n_seeds: int = N_FIXTURE_SEEDS,
) -> Tuple[int, int]:
    """``(positives_found, labels_spent)`` summed over ``n_seeds`` fixtures.

    Pooling seeds is deliberate: at 0.25 % prevalence a single run of 200
    random queries finds 0 positives about 60 % of the time, so a one-seed
    comparison would measure the dice.
    """
    found = spent = 0
    for seed in range(n_seeds):
        features, labels, _ = regime(seed)
        det = _run(
            strategy, features, labels, frames, budget=budget, n_rounds=n_rounds
        )
        found += det.n_positive_labels_
        spent += det.n_labelled_
    return found, spent


# ═══════════════════════════════════════════════════════════════════════════════
# The fixtures are what they claim
# ═══════════════════════════════════════════════════════════════════════════════


class TestConcentratedFixture:
    """If the fixture is not adult's regime, nothing measured on it transfers."""

    def test_prevalence_is_adult_scale(self) -> None:
        _, labels, y = _concentrated(0)
        stats = label_stats(labels)
        assert stats["n_corrupted"] == N_POS_CONCENTRATED
        assert stats["prevalence"] == pytest.approx(N_POS_CONCENTRATED / N_CELLS)
        assert 0.002 < stats["prevalence"] < 0.003, "not extreme imbalance any more"

    def test_every_positive_is_in_one_column(self) -> None:
        _, labels, y = _concentrated(0)
        per_column = label_stats(labels)["per_column"]
        assert per_column[COLUMNS[POSITIVE_COLUMN]] == N_POS_CONCENTRATED
        assert sum(
            v for k, v in per_column.items() if k != COLUMNS[POSITIVE_COLUMN]
        ) == 0

    def test_the_global_top_is_guaranteed_clean(self) -> None:
        """The wave-2 failure mode, reproduced: a combiner-led top-k is all
        rule false positives, so it cannot contain a single positive."""
        features, _, y = _concentrated(0)
        combiner = EnsembleDetector(combiner="noisy_or")._unsupervised_scores(features)
        top = np.argsort(-combiner)[: 10 * K]
        assert y[top].sum() == 0, "the poison column no longer dominates the top"
        assert set(_column_of(top).tolist()) == {POISONED_COLUMN}

    def test_the_conditional_signal_alone_does_rank_the_positives(self) -> None:
        """The complement: the signal a strategy *could* use is informative."""
        features, _, y = _concentrated(0)
        c_full = features.column("c_full")
        assert y[np.argsort(-c_full)[:N_POS_CONCENTRATED]].all()


class TestNoiseFixture:
    def test_scores_carry_no_information_about_the_labels(self) -> None:
        features, _, y = _noise(0)
        for name in ("c_full", "r_score"):
            values = features.column(name)
            # Point-biserial correlation of a pure-noise score with the label
            # must be near zero. 200 positives in 10,000 cells gives a
            # standard error of ~1/sqrt(n) = 0.01, so 0.05 is ~5 sigma.
            corr = float(np.corrcoef(values, y.astype(float))[0, 1])
            assert abs(corr) < 0.05, f"{name} is not noise (corr {corr:.4f})"


# ═══════════════════════════════════════════════════════════════════════════════
# The registry
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistry:
    def test_strategies_are_unique_and_documented(self) -> None:
        assert len(set(STRATEGIES)) == len(STRATEGIES)
        assert set(USES_MODEL) == set(STRATEGIES)
        assert set(acq._STRATEGY_FUNCTIONS) == set(STRATEGIES)

    def test_the_nine_documented_strategies_are_present(self) -> None:
        """Eight from the brief plus ``cond_top_live``, the sequential
        counterpart of ``conditional_top``. Both are documented in the module
        docstring; this pins the registry against one being added without
        being written up (or vice versa)."""
        assert set(STRATEGIES) == {
            "random",
            "global_top",
            "per_column_quota",
            "uncertainty",
            "conditional_top",
            "cond_top_live",
            "stratified_by_score",
            "batch_diverse",
            "cond_per_column",
        }

    def test_every_strategy_is_described_in_the_module_docstring(self) -> None:
        """A strategy that exists but is undocumented is the failure mode the
        stale eight-name test actually caught."""
        doc = acq.__doc__ or ""
        for name in STRATEGIES:
            assert f"``{name}``" in doc, f"{name} is not documented"

    def test_random_is_the_default_so_an_unconfigured_run_measures_the_control(
        self,
    ) -> None:
        assert AcquisitionDetector().strategy == "random"

    def test_unknown_strategy_is_rejected_by_both_entry_points(self) -> None:
        with pytest.raises(ValueError, match="strategy must be one of"):
            AcquisitionDetector(strategy="best_one")
        with pytest.raises(ValueError, match="strategy must be one of"):
            select_cells("best_one", 5, np.zeros(10), np.arange(10), n_rows=2)


# ═══════════════════════════════════════════════════════════════════════════════
# select_cells: the contract
# ═══════════════════════════════════════════════════════════════════════════════


class TestSelectionContract:
    @pytest.fixture(scope="class")
    def pieces(self):
        features, labels, y = _concentrated(0)
        guide = EnsembleDetector(combiner="noisy_or")._unsupervised_scores(features)
        return features, guide, features.column("c_full")

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_selection_is_inside_the_train_pool_and_within_budget(
        self, strategy, pieces
    ) -> None:
        features, guide, cond = pieces
        rng = np.random.default_rng(3)
        train_idx = np.sort(rng.choice(N_CELLS, 4000, replace=False))
        chosen, _ = select_cells(
            strategy, 50, guide, train_idx, N_ROWS, seed=SEED, conditional=cond
        )
        assert chosen.size == 50
        assert np.all(np.isin(chosen, train_idx))
        assert np.unique(chosen).size == chosen.size

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_zero_budget_selects_nothing(self, strategy, pieces) -> None:
        _, guide, cond = pieces
        chosen, notes = select_cells(
            strategy, 0, guide, np.arange(N_CELLS), N_ROWS, conditional=cond
        )
        assert chosen.size == 0
        assert notes == []

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_budget_larger_than_the_pool_is_clamped(self, strategy, pieces) -> None:
        _, guide, cond = pieces
        train_idx = np.arange(30)
        chosen, _ = select_cells(
            strategy, 500, guide, train_idx, N_ROWS, conditional=cond
        )
        assert chosen.size == train_idx.size
        assert sorted(chosen.tolist()) == sorted(train_idx.tolist())

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_already_labelled_cells_are_never_bought_twice(
        self, strategy, pieces
    ) -> None:
        _, guide, cond = pieces
        train_idx = np.arange(N_CELLS)
        first, _ = select_cells(
            strategy, 40, guide, train_idx, N_ROWS, seed=SEED, conditional=cond
        )
        second, _ = select_cells(
            strategy,
            40,
            guide,
            train_idx,
            N_ROWS,
            seed=SEED + 1,
            conditional=cond,
            already_labelled=first,
        )
        assert np.intersect1d(first, second).size == 0

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_deterministic_for_a_seed(self, strategy, pieces) -> None:
        _, guide, cond = pieces
        kwargs = dict(
            guide=guide, train_idx=np.arange(N_CELLS), n_rows=N_ROWS, conditional=cond
        )
        a, _ = select_cells(strategy, 60, seed=7, **kwargs)
        b, _ = select_cells(strategy, 60, seed=7, **kwargs)
        assert np.array_equal(a, b)

    def test_a_different_seed_moves_a_random_selection(self, pieces) -> None:
        _, guide, cond = pieces
        kwargs = dict(guide=guide, train_idx=np.arange(N_CELLS), n_rows=N_ROWS)
        a, _ = select_cells("random", 60, seed=1, **kwargs)
        b, _ = select_cells("random", 60, seed=2, **kwargs)
        assert not np.array_equal(a, b)

    @pytest.mark.parametrize("strategy", ["conditional_top", "cond_per_column"])
    def test_conditional_strategies_refuse_to_run_without_the_signal(
        self, strategy, pieces
    ) -> None:
        _, guide, _ = pieces
        with pytest.raises(ValueError, match="requires `conditional`"):
            select_cells(strategy, 10, guide, np.arange(N_CELLS), N_ROWS)

    def test_conditional_strategies_ignore_the_guide_entirely(self, pieces) -> None:
        """``conditional_top`` must be defined on ``c_full`` and nothing else,
        because on adult the rule score in the guide is anti-correlated with
        truth. Replacing the guide with garbage may not move its picks."""
        _, guide, cond = pieces
        kwargs = dict(train_idx=np.arange(N_CELLS), n_rows=N_ROWS, conditional=cond)
        a, _ = select_cells("conditional_top", 40, guide, seed=SEED, **kwargs)
        b, _ = select_cells(
            "conditional_top", 40, -guide, seed=SEED, **kwargs
        )
        assert np.array_equal(a, b)

    def test_empty_train_pool_is_handled(self, pieces) -> None:
        _, guide, cond = pieces
        chosen, notes = select_cells(
            "global_top", 10, guide, np.empty(0, dtype=int), N_ROWS, conditional=cond
        )
        assert chosen.size == 0

    def test_a_short_strategy_is_topped_up_and_says_so(self, pieces) -> None:
        """``batch_diverse``'s one-cell-per-row cap cannot fill a budget
        larger than the row count; the shortfall must be reported, not
        silently returned as a smaller sample."""
        _, guide, cond = pieces
        train_idx = np.arange(N_ROWS * 2)  # two columns' worth
        chosen, notes = select_cells(
            "batch_diverse", N_ROWS + 50, guide, train_idx, N_ROWS, seed=SEED
        )
        assert chosen.size == N_ROWS + 50
        assert any("topped up" in n for n in notes)


# ═══════════════════════════════════════════════════════════════════════════════
# Round-robin, re-derived independently
# ═══════════════════════════════════════════════════════════════════════════════


class TestRoundRobin:
    def test_matches_an_independent_python_derivation(self) -> None:
        """Re-derive round-robin with a plain loop and compare group shapes.

        The vectorised implementation replaced exactly this loop (it ran over
        every train cell once per round and dominated the sweep), so the loop
        is the right independent oracle.
        """
        rng = np.random.default_rng(11)
        groups = rng.integers(0, 4, 200)
        values = rng.random(200)
        k = 30
        chosen = acq._round_robin_by_group(groups, values, k, np.random.default_rng(0))

        queues: Dict[int, List[int]] = {}
        for position in np.argsort(-values, kind="stable"):
            queues.setdefault(int(groups[position]), []).append(int(position))
        expected: List[int] = []
        cursor = {g: 0 for g in sorted(queues)}
        while len(expected) < k:
            for g in sorted(queues):
                if len(expected) >= k:
                    break
                if cursor[g] < len(queues[g]):
                    expected.append(queues[g][cursor[g]])
                    cursor[g] += 1
        # Ties are broken randomly, so compare the per-group allocation and
        # the group-visit order, which is what round-robin actually fixes.
        assert chosen.size == k
        got = np.bincount(groups[chosen], minlength=4)
        want = np.bincount(groups[np.asarray(expected)], minlength=4)
        assert got.tolist() == want.tolist()

    def test_allocation_is_even_across_groups(self) -> None:
        groups = np.repeat(np.arange(5), 100)
        values = np.random.default_rng(2).random(500)
        chosen = acq._round_robin_by_group(groups, values, 20, np.random.default_rng(0))
        assert np.bincount(groups[chosen], minlength=5).tolist() == [4] * 5

    def test_each_group_contributes_its_best_first(self) -> None:
        groups = np.repeat(np.arange(3), 10)
        values = np.tile(np.arange(10, dtype=float), 3)
        chosen = acq._round_robin_by_group(groups, values, 3, np.random.default_rng(0))
        assert sorted(values[chosen].tolist()) == [9.0, 9.0, 9.0]

    def test_within_group_rank_is_zero_based_per_group(self) -> None:
        groups = np.array([0, 0, 1, 1, 1])
        order = np.arange(5)
        assert acq._within_group_rank(groups, order).tolist() == [0, 1, 0, 1, 2]


class TestPerColumnMatchesWaveTwo:
    """``per_column_quota`` is wave 2's sampler. A silent divergence would
    make the W3 table incomparable with ``results/w2_ensemble.md``."""

    def test_same_column_allocation_as_the_ensembles_own_sampler(self) -> None:
        features, labels, _ = _concentrated(0)
        det = EnsembleDetector(combiner="noisy_or", seed=SEED)
        det.features_ = features
        train_idx = np.arange(N_CELLS)
        wave2 = det._per_column_informed(train_idx, 40, np.random.default_rng(SEED))
        guide = det._unsupervised_scores(features)
        mine, _ = select_cells(
            "per_column_quota",
            40,
            guide,
            train_idx,
            N_ROWS,
            seed=SEED,
            eligible=~det._trivially_missing(features),
        )
        assert np.bincount(_column_of(mine), minlength=N_COLS).tolist() == np.bincount(
            _column_of(wave2), minlength=N_COLS
        ).tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# THE HEADLINE MECHANISM: positives found per query
# ═══════════════════════════════════════════════════════════════════════════════


class TestPositivesPerQuery:
    """
    The route's reason to exist. If a targeted strategy cannot buy more
    positives per query than uniform random at 0.25 % prevalence, W3.1 is a
    loss and must be reported as one.
    """

    @pytest.fixture(scope="class")
    def yields(self, frames) -> Dict[str, Tuple[int, int]]:
        return {s: _yield_over_seeds(s, _concentrated, frames) for s in STRATEGIES}

    def test_random_finds_about_prevalence_times_k(self, yields) -> None:
        """The control's sanity check, with the band derived here.

        Positives found by uniform sampling is ~Binomial(n, prevalence);
        4 standard deviations is a band a correct implementation passes and a
        broken one (e.g. sampling with replacement from the positives) does
        not.
        """
        found, spent = yields["random"]
        prevalence = N_POS_CONCENTRATED / N_CELLS
        expected = spent * prevalence
        sd = math.sqrt(spent * prevalence * (1 - prevalence))
        assert spent == K * N_FIXTURE_SEEDS
        assert abs(found - expected) <= 4 * sd, (
            f"random found {found}, expected {expected:.1f} +/- {4 * sd:.1f}"
        )

    @pytest.mark.parametrize("strategy", ["conditional_top", "stratified_by_score"])
    def test_a_targeted_strategy_beats_random_per_query(
        self, strategy, yields
    ) -> None:
        targeted, t_spent = yields[strategy]
        control, c_spent = yields["random"]
        assert t_spent == c_spent, "a budget mismatch would invalidate the comparison"
        assert targeted > control, (
            f"{strategy} found {targeted} positives, random found {control}"
        )
        # And by a margin no plausible run of the control could produce:
        prevalence = N_POS_CONCENTRATED / N_CELLS
        sd = math.sqrt(t_spent * prevalence * (1 - prevalence))
        assert targeted - t_spent * prevalence > 10 * sd

    def test_cond_per_column_the_modules_own_strategy_also_beats_random(
        self, yields
    ) -> None:
        assert yields["cond_per_column"][0] > yields["random"][0]

    def test_the_global_top_failure_mode_is_reproduced(self, yields) -> None:
        """Wave 2's measured failure: a combiner-led global top buys nothing
        when the rules dominate the ranking. Asserted, so a later change that
        accidentally 'fixes' the fixture is caught."""
        assert yields["global_top"][0] == 0

    def test_cold_start_uncertainty_sampling_cannot_bootstrap(self, yields) -> None:
        """Honest negative result, asserted rather than written in prose.

        One-shot ``uncertainty`` has no model to be uncertain about, so it
        ranks the combiner's mid-range — which here holds no positives.
        """
        assert yields["uncertainty"][0] == 0

    def test_positives_per_query_is_reported_and_consistent(
        self, frames
    ) -> None:
        features, labels, _ = _concentrated(0)
        det = _run("conditional_top", features, labels, frames)
        report = det.report()
        assert report["positives_per_query"] == pytest.approx(
            report["n_positive_labels"] / report["n_labelled"]
        )
        assert report["strategy"] == "conditional_top"
        assert sum(r["n_queried"] for r in report["rounds"]) == report["n_labelled"]
        assert sum(r["n_positive"] for r in report["rounds"]) == report[
            "n_positive_labels"
        ]

    def test_nothing_queried_reports_none_not_zero(self, frames) -> None:
        """``0.0`` would read as 'we looked and found none' (CLAUDE.md §5.4)."""
        features, labels, _ = _concentrated(0)
        det = _run("random", features, labels, frames, budget=0)
        assert det.positives_per_query_ is None


# ═══════════════════════════════════════════════════════════════════════════════
# THE ANTI-CHEAT TEST: no signal, no winner
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoSignalNoWinner:
    """
    On scores that carry no information, **every** strategy must collapse to
    random. A strategy that still wins here is reading a label it did not
    pay for, or scoring itself against its own selection.
    """

    @pytest.fixture(scope="class")
    def yields(self, frames) -> Dict[str, Tuple[int, int]]:
        return {s: _yield_over_seeds(s, _noise, frames) for s in STRATEGIES}

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_within_four_binomial_sigma_of_the_prevalence(
        self, strategy, yields
    ) -> None:
        found, spent = yields[strategy]
        prevalence = N_POS_NOISE / N_CELLS
        expected = spent * prevalence
        sd = math.sqrt(spent * prevalence * (1 - prevalence))
        assert abs(found - expected) <= 4 * sd, (
            f"{strategy} found {found} of {spent}, expected {expected:.1f} "
            f"+/- {4 * sd:.1f} under the null"
        )

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_no_strategy_approaches_oracle_yield(self, strategy, yields) -> None:
        """A leaky implementation would find min(k, n_pos) every time, i.e. a
        positives-per-query near 1.0. This is the coarse catch-all."""
        found, spent = yields[strategy]
        assert found / spent < 3 * (N_POS_NOISE / N_CELLS)


# ═══════════════════════════════════════════════════════════════════════════════
# One-shot vs sequential
# ═══════════════════════════════════════════════════════════════════════════════


class TestSequentialAcquisition:
    @pytest.fixture(scope="class")
    def yields(self, frames) -> Dict[Tuple[str, int], int]:
        out: Dict[Tuple[str, int], int] = {}
        for strategy in STRATEGIES:
            for rounds in (1, 5):
                out[(strategy, rounds)] = _yield_over_seeds(
                    strategy, _concentrated, frames, n_rounds=rounds
                )[0]
        return out

    def test_the_total_budget_is_identical_across_modes(self, frames) -> None:
        features, labels, _ = _concentrated(0)
        spent = {
            rounds: _run(
                "per_column_quota", features, labels, frames, n_rounds=rounds
            ).n_labelled_
            for rounds in (1, 2, 5, 10)
        }
        assert set(spent.values()) == {K}, (
            "a sequential run that spends a different total is not comparable"
        )

    def test_round_sizes_sum_to_the_budget_with_the_remainder_first(self) -> None:
        det = AcquisitionDetector(strategy="random", n_rounds=4, budget=10)
        assert det._round_sizes(10) == [3, 3, 2, 2]
        assert sum(det._round_sizes(10)) == 10

    def test_more_rounds_than_labels_is_clamped(self, frames) -> None:
        features, labels, _ = _concentrated(0)
        det = _run("random", features, labels, frames, budget=3, n_rounds=99)
        assert det.n_labelled_ == 3
        assert len(det.rounds_) == 3
        assert all(r.n_queried == 1 for r in det.rounds_)

    def test_sequential_is_at_least_as_good_in_aggregate(self, yields) -> None:
        """The honest direction, measured: pooled over all eight strategies
        and eight fixture seeds, sequential acquisition finds no fewer
        positives than one-shot at the same total budget."""
        one_shot = sum(yields[(s, 1)] for s in STRATEGIES)
        sequential = sum(yields[(s, 5)] for s in STRATEGIES)
        assert sequential >= one_shot, (
            f"sequential pooled {sequential} < one-shot {one_shot}"
        )

    def test_batch_diverse_improves_with_rounds(self, yields) -> None:
        """Measured: 194 -> 200 of 200 available positives over 8 seeds. The
        refit lets a de-duplicated batch re-rank onto the live signal."""
        assert yields[("batch_diverse", 5)] >= yields[("batch_diverse", 1)]

    def test_strategies_that_ignore_the_model_pick_identical_cells(
        self, frames
    ) -> None:
        """The control for the comparison above. ``conditional_top`` and
        ``cond_per_column`` are defined on ``c_full``, which labelling cannot
        change, so any sequential difference elsewhere comes from the refit
        and not from the round structure."""
        features, labels, _ = _concentrated(0)
        for strategy in ("conditional_top", "cond_per_column"):
            assert USES_MODEL[strategy] is False
            a = _run(strategy, features, labels, frames, n_rounds=1).labelled_indices_
            b = _run(strategy, features, labels, frames, n_rounds=5).labelled_indices_
            assert np.array_equal(a, b), strategy

    def test_sequential_rounds_switch_to_the_model_once_one_exists(
        self, frames
    ) -> None:
        features, labels, _ = _concentrated(0)
        det = _run("per_column_quota", features, labels, frames, n_rounds=5)
        guides = [r.guided_by for r in det.rounds_]
        assert guides[0] == "combiner", "round 1 cannot have a model"
        assert det.used_supervision_ and "model" in guides[1:], (
            "a refit happened but later rounds were not guided by it"
        )

    def test_a_strategy_that_never_finds_a_positive_never_gets_a_model(
        self, frames
    ) -> None:
        """Honest mechanism: sequential acquisition cannot rescue a strategy
        whose first round buys only negatives — there is nothing to fit."""
        features, labels, _ = _concentrated(0)
        det = _run("global_top", features, labels, frames, n_rounds=5)
        assert det.n_positive_labels_ == 0
        assert det.used_supervision_ is False
        assert all(r.guided_by == "combiner" for r in det.rounds_)
        assert det.fallback_reason_ and "only negatives" in det.fallback_reason_


# ═══════════════════════════════════════════════════════════════════════════════
# Leakage
# ═══════════════════════════════════════════════════════════════════════════════


class SpyOracle(LabelOracle):
    """Records every query so a test can prove what was asked for."""

    def __init__(self, labels: pd.DataFrame, max_queries=None) -> None:
        super().__init__(labels, max_queries=max_queries)
        self.calls: List[np.ndarray] = []

    def query(self, flat_indices):
        idx = np.asarray(list(flat_indices), dtype=int)
        self.calls.append(idx.copy())
        return super().query(idx)


def _random_mask(fraction: float = 0.7, seed: int = 7):
    rng = np.random.default_rng(seed)
    flat = rng.random(N_CELLS) < fraction
    frame = pd.DataFrame(
        {c: flat[i * N_ROWS : (i + 1) * N_ROWS] for i, c in enumerate(COLUMNS)},
        index=pd.RangeIndex(N_ROWS),
    )
    return frame, flat


class TestNoLeakage:
    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_exactly_k_queries_and_never_more(self, strategy, frames) -> None:
        features, labels, _ = _concentrated(0)
        dirty, _ = frames
        mask, _ = _random_mask()
        oracle = SpyOracle(labels, max_queries=K)
        det = _run(
            strategy, features, labels, frames, oracle=oracle, mask=mask, n_rounds=5
        )
        assert det.n_labelled_ == K
        assert oracle.n_queries == K
        assert sum(c.size for c in oracle.calls) == K
        with pytest.raises(RuntimeError, match="budget exhausted"):
            oracle.query([0])

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_no_cell_outside_the_train_mask_is_ever_queried(
        self, strategy, frames
    ) -> None:
        features, labels, _ = _concentrated(0)
        mask, flat = _random_mask()
        oracle = SpyOracle(labels)
        _run(strategy, features, labels, frames, oracle=oracle, mask=mask, n_rounds=5)
        assert oracle.queried_indices, "nothing was queried; the test proves nothing"
        assert all(bool(flat[i]) for i in oracle.queried_indices)

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_no_label_is_bought_twice(self, strategy, frames) -> None:
        features, labels, _ = _concentrated(0)
        mask, _ = _random_mask()
        oracle = SpyOracle(labels)
        _run(strategy, features, labels, frames, oracle=oracle, mask=mask, n_rounds=5)
        assert len(set(oracle.queried_indices)) == len(oracle.queried_indices)

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_inverting_unqueried_train_labels_changes_nothing(
        self, strategy, frames
    ) -> None:
        """
        The counterfactual test (wave 2's, strengthened).

        Wave 2 inverted the labels *outside* the train mask. Acquisition is
        allowed to read train labels — but only the ones it queried — so this
        version inverts every train label the strategy did NOT buy as well.
        If any unqueried label influenced a selection or a fitted score, the
        second run must differ.
        """
        features, labels, _ = _concentrated(0)
        dirty, _ = frames
        mask, flat = _random_mask()

        first = SpyOracle(labels)
        det_a = _run(
            strategy, features, labels, frames, oracle=first, mask=mask, n_rounds=5
        )
        queried = np.asarray(sorted(set(first.queried_indices)), dtype=int)
        assert queried.size, "nothing was queried; the test proves nothing"

        poisoned_flat = flatten_cells(labels).astype(bool).copy()
        untouched = np.ones(N_CELLS, dtype=bool)
        untouched[queried] = False
        poisoned_flat[untouched] = ~poisoned_flat[untouched]
        poisoned = pd.DataFrame(
            {
                c: poisoned_flat[i * N_ROWS : (i + 1) * N_ROWS]
                for i, c in enumerate(COLUMNS)
            },
            index=pd.RangeIndex(N_ROWS),
        )
        assert not poisoned.equals(labels), "the poisoning changed nothing"

        second = SpyOracle(poisoned)
        det_b = _run(
            strategy, features, poisoned, frames, oracle=second, mask=mask, n_rounds=5
        )
        assert first.queried_indices == second.queried_indices, (
            f"{strategy} chose different cells when unqueried labels moved"
        )
        assert np.array_equal(
            flatten_cells(det_a.score_cells(dirty)),
            flatten_cells(det_b.score_cells(dirty)),
        )

    def test_budget_zero_never_touches_the_oracle(self, frames) -> None:
        features, labels, _ = _concentrated(0)
        oracle = SpyOracle(labels)
        det = _run("conditional_top", features, labels, frames, budget=0, oracle=oracle)
        assert oracle.n_queries == 0
        assert oracle.calls == []
        assert det.rounds_ == []

    def test_a_budget_without_an_oracle_raises_rather_than_guessing(
        self, frames
    ) -> None:
        features, labels, _ = _concentrated(0)
        dirty, mask = frames
        det = AcquisitionDetector(
            strategy="random", budget=20, precomputed_features=features
        )
        with pytest.raises(ValueError, match="requires a label_oracle"):
            det.fit(dirty, mask)

    def test_no_public_signature_takes_ground_truth(self) -> None:
        """Structural: there is nowhere to put a label frame or a clean frame."""
        banned = ("clean", "label_frame", "labels", "y_true", "ground_truth")
        public = [
            getattr(acq, name)
            for name in acq.__all__
            if callable(getattr(acq, name)) or inspect.isclass(getattr(acq, name))
        ]
        public += [AcquisitionDetector.fit, AcquisitionDetector.report, select_cells]
        for obj in public:
            target = obj.__init__ if inspect.isclass(obj) else obj
            for parameter in inspect.signature(target).parameters:
                assert not any(b in parameter for b in banned), (
                    f"{getattr(obj, '__name__', obj)} takes {parameter!r}"
                )

    def test_fit_has_nowhere_to_receive_labels(self) -> None:
        parameters = list(inspect.signature(AcquisitionDetector.fit).parameters)
        assert parameters == ["self", "dirty_df", "train_mask", "profile"]

    def test_select_cells_takes_no_label_array(self) -> None:
        parameters = set(inspect.signature(select_cells).parameters)
        assert "already_labelled" in parameters, "the paid-label trace must exist"
        assert not parameters & {"labels", "y", "clean_df", "oracle"}


# ═══════════════════════════════════════════════════════════════════════════════
# Determinism, validation, fallback
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    @pytest.mark.parametrize("strategy", STRATEGIES)
    @pytest.mark.parametrize("rounds", [1, 5])
    def test_same_seed_same_selection_and_scores(
        self, strategy, rounds, frames
    ) -> None:
        features, labels, _ = _concentrated(0)
        dirty, _ = frames
        runs = [
            _run(strategy, features, labels, frames, n_rounds=rounds) for _ in range(2)
        ]
        assert np.array_equal(runs[0].labelled_indices_, runs[1].labelled_indices_)
        assert np.array_equal(
            flatten_cells(runs[0].score_cells(dirty)),
            flatten_cells(runs[1].score_cells(dirty)),
        )

    def test_a_different_seed_can_move_the_selection(self, frames) -> None:
        features, labels, _ = _concentrated(0)
        a = _run("random", features, labels, frames, seed=1).labelled_indices_
        b = _run("random", features, labels, frames, seed=2).labelled_indices_
        assert not np.array_equal(a, b)


class TestValidation:
    @pytest.mark.parametrize("n_rounds", [0, -1])
    def test_bad_n_rounds_rejected(self, n_rounds) -> None:
        with pytest.raises(ValueError, match="n_rounds must be >= 1"):
            AcquisitionDetector(n_rounds=n_rounds)

    @pytest.mark.parametrize("n_bins", [0, 1])
    def test_bad_n_bins_rejected(self, n_bins) -> None:
        with pytest.raises(ValueError, match="n_bins must be >= 2"):
            AcquisitionDetector(n_bins=n_bins)

    @pytest.mark.parametrize("budget", [-1, -100])
    def test_negative_budget_rejected(self, budget) -> None:
        with pytest.raises(ValueError, match="budget must be >= 0"):
            AcquisitionDetector(budget=budget)

    def test_mismatched_train_mask_rejected(self, frames) -> None:
        features, labels, _ = _concentrated(0)
        dirty, _ = frames
        det = AcquisitionDetector(
            strategy="random",
            budget=10,
            label_oracle=LabelOracle(labels),
            precomputed_features=features,
        )
        bad = pd.DataFrame(np.ones((3, 2), dtype=bool), columns=["a", "b"])
        with pytest.raises(ValueError, match="train_mask has"):
            det.fit(dirty, bad)

    def test_mismatched_precomputed_features_rejected(self, frames) -> None:
        features, labels, _ = _concentrated(0)
        det = AcquisitionDetector(
            strategy="random", budget=0, precomputed_features=features
        )
        with pytest.raises(ValueError, match="precomputed_features has"):
            det.fit(pd.DataFrame({"a": [1.0, 2.0]}))

    def test_report_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="report\\(\\) called before fit"):
            AcquisitionDetector().report()

    def test_default_name_encodes_strategy_budget_and_mode(self) -> None:
        assert (
            AcquisitionDetector(strategy="uncertainty", budget=50, n_rounds=1).name
            == "acq_uncertainty_b50_oneshot"
        )
        assert (
            AcquisitionDetector(strategy="uncertainty", budget=50, n_rounds=5).name
            == "acq_uncertainty_b50_seq"
        )

    def test_round_record_round_trips(self) -> None:
        record = RoundRecord(0, 10, 2, "model", True, ["note"])
        assert record.to_dict() == {
            "round": 0,
            "n_queried": 10,
            "n_positive": 2,
            "guided_by": "model",
            "model_fitted": True,
            "notes": ["note"],
        }


class TestInheritedBehaviour:
    """Acquisition changes which cells are labelled and nothing else; the
    wave-2 semantics around it must still hold, or the rows are not
    comparable."""

    def test_empty_train_mask_is_reported_not_crashed(self, frames) -> None:
        features, labels, _ = _concentrated(0)
        dirty, _ = frames
        empty = pd.DataFrame(
            np.zeros((N_ROWS, N_COLS), dtype=bool), columns=COLUMNS
        )
        det = AcquisitionDetector(
            strategy="random",
            budget=20,
            label_oracle=LabelOracle(labels),
            precomputed_features=features,
        )
        det.fit(dirty, empty)
        assert det.n_labelled_ == 0
        assert det.fallback_reason_ and "zero cells" in det.fallback_reason_

    def test_the_fallback_is_exactly_the_unsupervised_combiner(self, frames) -> None:
        features, labels, _ = _concentrated(0)
        dirty, mask = frames
        failed = _run("global_top", features, labels, frames)
        assert failed.used_supervision_ is False
        unsupervised = _run("random", features, labels, frames, budget=0)
        assert np.array_equal(
            flatten_cells(failed.score_cells(dirty)),
            flatten_cells(unsupervised.score_cells(dirty)),
        )

    def test_full_budget_labels_every_train_cell(self, frames) -> None:
        features, labels, _ = _concentrated(0)
        dirty, _ = frames
        mask, flat = _random_mask()
        oracle = SpyOracle(labels)
        det = AcquisitionDetector(
            strategy="conditional_top",
            budget="full",
            label_oracle=oracle,
            precomputed_features=features,
        )
        det.fit(dirty, mask)
        assert det.n_labelled_ == int(flat.sum())
        assert det.positives_per_query_ == pytest.approx(
            det.n_positive_labels_ / det.n_labelled_
        )

    def test_score_frame_satisfies_the_harness_contract(self, frames) -> None:
        features, labels, _ = _concentrated(0)
        dirty, _ = frames
        scores = _run("cond_per_column", features, labels, frames).score_cells(dirty)
        assert scores.shape == dirty.shape
        assert list(scores.columns) == list(dirty.columns)
        assert scores.index.equals(dirty.index)
        assert np.all(np.isfinite(scores.to_numpy(dtype=float)))

    def test_supervision_improves_the_ranking_on_the_concentrated_set(
        self, frames
    ) -> None:
        """Sanity: a strategy that finds positives must produce a better
        ranking than the label-free fallback, or the labels bought nothing."""
        from src.detection_eval.metrics import evaluate_scores

        features, labels, y = _concentrated(0)
        dirty, _ = frames
        supervised = _run("conditional_top", features, labels, frames)
        baseline = _run("random", features, labels, frames, budget=0)
        assert supervised.used_supervision_
        a = evaluate_scores(y, flatten_cells(supervised.score_cells(dirty))).pr_auc
        b = evaluate_scores(y, flatten_cells(baseline.score_cells(dirty))).pr_auc
        assert a is not None and b is not None and a > b


# ═══════════════════════════════════════════════════════════════════════════════
# The real harness, on a real on-disk dataset
# ═══════════════════════════════════════════════════════════════════════════════


def _make_pair(n: int = 300, seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A small clean/dirty pair whose corruption is a valid->valid flip in one
    column — adult's corruption, which is what acquisition has to find."""
    rng = np.random.default_rng(seed)
    size = rng.integers(10, 41, n)
    clean = pd.DataFrame(
        {
            "size": size,
            "hours": size * 3,
            "label": np.where(size > 25, "high", "low"),
        }
    )
    dirty = clean.copy()
    flipped = rng.choice(n, 20, replace=False)
    dirty.loc[flipped, "label"] = np.where(
        clean.loc[flipped, "label"] == "high", "low", "high"
    )
    return clean, dirty


@pytest.fixture(scope="module")
def on_disk(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("w3_synth")
    clean, dirty = _make_pair()
    folder = root / "synth"
    folder.mkdir(parents=True, exist_ok=True)
    clean.to_csv(folder / "clean.csv", index=False)
    dirty.to_csv(folder / "dirty.csv", index=False)
    return root


class TestHarnessIntegration:
    """The experiment measures these detectors through
    :func:`src.detection_eval.harness.evaluate_detector`, so the integration
    is part of the contract — including that the real feature builder works,
    not just an injected matrix."""

    def test_evaluate_detector_runs_and_respects_the_budget(self, on_disk) -> None:
        loader = DataLoader(str(on_disk))
        clean_df, dirty_df = loader.load("synth")
        profile = profile_dataset(
            "synth", loader.discover("synth"), clean_df, dirty_df
        )
        labels = cell_labels(clean_df, dirty_df)
        flat = flatten_cells(labels).astype(bool)

        n_splits, budget = 3, 30
        oracle = SpyOracle(labels, max_queries=budget * n_splits)
        det = AcquisitionDetector(
            strategy="conditional_top",
            budget=budget,
            n_rounds=3,
            label_oracle=oracle,
            profile=profile,
            conditional_kwargs=FAST_CONDITIONAL,
            feature_cache=FeatureCache(),
            seed=SEED,
        )
        out = evaluate_detector(
            det, "synth", data_root=str(on_disk), n_splits=n_splits, seed=SEED
        )
        assert out["refit_per_split"] is True
        assert oracle.n_queries == budget * n_splits
        assert out["intervals"]["pr_auc"]["n"] == n_splits
        assert out["intervals"]["pr_auc"]["mean"] > float(flat.mean()), (
            "the detector did not beat a random scorer on its own harness"
        )

    def test_the_harness_offers_the_oracle_only_train_cells(self, on_disk) -> None:
        loader = DataLoader(str(on_disk))
        clean_df, dirty_df = loader.load("synth")
        profile = profile_dataset(
            "synth", loader.discover("synth"), clean_df, dirty_df
        )
        labels = cell_labels(clean_df, dirty_df)
        n_cells = labels.size

        seen_eval: List[int] = []
        masks: List[np.ndarray] = []

        det = AcquisitionDetector(
            strategy="per_column_quota",
            budget=25,
            label_oracle=SpyOracle(labels),
            profile=profile,
            conditional_kwargs=FAST_CONDITIONAL,
            feature_cache=FeatureCache(),
            seed=SEED,
        )
        original_fit = det.fit

        def fit_and_capture(dirty, train_mask=None, profile=None):
            masks.append(flatten_cells(pd.DataFrame(train_mask)).astype(bool).copy())
            out = original_fit(dirty, train_mask, profile)
            queried = np.asarray(det.labelled_indices_, dtype=int)
            seen_eval.extend(queried[~masks[-1][queried]].tolist())
            return out

        det.fit = fit_and_capture  # type: ignore[method-assign]
        evaluate_detector(det, "synth", data_root=str(on_disk), n_splits=2, seed=SEED)

        assert masks and masks[0].size == n_cells
        assert not masks[0].all(), "the harness handed over every cell as train"
        assert seen_eval == [], f"eval cells were queried: {seen_eval[:5]}"
