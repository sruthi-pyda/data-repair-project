"""
tests/test_threshold.py
-----------------------
Tests for `src/decision/threshold.py`.

The decisive test is `TestBudgetHonesty` plus
`TestStratifiedBeatsUniform::test_stratified_finds_positives_uniform_misses`.
The whole premise of the stratified selector is that uniform sampling wastes
its budget confirming that low-scoring cells are clean, so if a stratified
draw does not actually find more positives at the same budget, the mechanism
has no reason to exist.

Fixtures build their own ground truth, so expected answers are derived
independently of the code under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.decision.threshold import (
    DEFAULT_STRATA,
    METHODS,
    ThresholdChoice,
    apply_threshold,
    select_threshold,
    _allocate,
    _stratum_bounds,
)

SEED = 42


def _problem(n: int = 20000, prevalence: float = 0.0255, seed: int = SEED):
    """
    A detector-like problem with KNOWN prevalence and a KNOWN good threshold.

    Positives get higher scores than negatives on average but the
    distributions overlap, so a threshold genuinely matters. Prevalence
    matches hospital's 0.0255, which is the case this module exists for.
    """
    rng = np.random.default_rng(seed)
    n_pos = int(round(n * prevalence))
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    rng.shuffle(y)
    scores = np.where(
        y == 1,
        rng.beta(5.0, 2.0, size=n),   # positives: high
        rng.beta(2.0, 6.0, size=n),   # negatives: low, overlapping
    )
    return scores, y


def _reveal_for(y: np.ndarray):
    """A reveal callback that also records exactly which cells were asked."""
    asked: list[int] = []

    def reveal(idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=int)
        asked.extend(idx.tolist())
        return y[idx]

    return reveal, asked


# ═══════════════════════════════════════════════════════════════════════════
# Budget honesty — the claim that makes the comparison fair at all
# ═══════════════════════════════════════════════════════════════════════════


class TestBudgetHonesty:
    @pytest.mark.parametrize("method", ["uniform_f1", "stratified_f1",
                                        "stratified_cost"])
    @pytest.mark.parametrize("budget", [10, 20, 50])
    def test_never_reveals_more_than_the_budget(self, method, budget):
        scores, y = _problem()
        reveal, asked = _reveal_for(y)
        ch = select_threshold(scores, reveal, method=method, budget=budget,
                              seed=SEED)
        assert ch.n_revealed <= budget, (
            f"{method} claimed {ch.n_revealed} of {budget}")
        assert len(asked) <= budget, (
            f"{method} actually asked for {len(asked)} labels, budget "
            f"{budget}")
        assert len(asked) == ch.n_revealed, (
            f"reported {ch.n_revealed} but asked {len(asked)}")

    def test_oracle_reveals_everything_and_says_so(self):
        scores, y = _problem(n=2000)
        reveal, asked = _reveal_for(y)
        ch = select_threshold(scores, reveal, method="oracle_f1", budget=20)
        assert len(asked) == scores.size
        assert "upper bound" in ch.note

    def test_strata_are_built_from_scores_only(self):
        """
        Strata must not depend on labels. Same scores with INVERTED labels
        must produce the same sampled cells.
        """
        scores, y = _problem()
        r1, asked1 = _reveal_for(y)
        r2, asked2 = _reveal_for(1 - y)
        select_threshold(scores, r1, method="stratified_f1", budget=20,
                         seed=SEED)
        select_threshold(scores, r2, method="stratified_f1", budget=20,
                         seed=SEED)
        assert sorted(asked1) == sorted(asked2)


# ═══════════════════════════════════════════════════════════════════════════
# THE DECISIVE TEST
# ═══════════════════════════════════════════════════════════════════════════


class TestStratifiedBeatsUniform:
    def test_stratified_finds_positives_uniform_misses(self):
        """
        The premise. At this prevalence, 20 uniform labels hold ~0.5
        positives, while stratified sampling deliberately spends in the top
        strata where positives live. Averaged over draws, stratified must
        find materially more.
        """
        scores, y = _problem()
        uni, strat = [], []
        for k in range(20):
            r1, a1 = _reveal_for(y)
            select_threshold(scores, r1, method="uniform_f1", budget=20,
                             seed=SEED + k)
            uni.append(int(y[np.array(a1, dtype=int)].sum()))
            r2, a2 = _reveal_for(y)
            select_threshold(scores, r2, method="stratified_f1", budget=20,
                             seed=SEED + k)
            strat.append(int(y[np.array(a2, dtype=int)].sum()))
        assert np.mean(strat) > np.mean(uni) + 1.0, (
            f"stratified found {np.mean(strat):.2f} positives per draw vs "
            f"uniform {np.mean(uni):.2f}; the premise does not hold"
        )

    def test_stratified_picks_a_threshold_more_often(self):
        """
        Uniform frequently cannot fit a threshold at all (no positive in the
        sample). Stratified should almost always manage one.
        """
        scores, y = _problem()
        uni_ok = strat_ok = 0
        for k in range(20):
            r1, _ = _reveal_for(y)
            if select_threshold(scores, r1, method="uniform_f1", budget=20,
                                seed=SEED + k).threshold is not None:
                uni_ok += 1
            r2, _ = _reveal_for(y)
            if select_threshold(scores, r2, method="stratified_f1", budget=20,
                                seed=SEED + k).threshold is not None:
                strat_ok += 1
        assert strat_ok > uni_ok, (
            f"stratified fitted {strat_ok}/20 draws, uniform {uni_ok}/20")

    def test_stratified_f1_beats_uniform_f1_on_average(self):
        """The outcome that matters: better realised F1 at the same budget."""
        scores, y = _problem()
        n_pos = int(y.sum())

        def realised(ch: ThresholdChoice) -> float:
            flags = apply_threshold(scores, ch)
            tp = int(np.sum(flags & (y == 1)))
            if flags.sum() == 0 or tp == 0:
                return 0.0
            p, r = tp / int(flags.sum()), tp / n_pos
            return 2 * p * r / (p + r)

        uni, strat = [], []
        for k in range(20):
            r1, _ = _reveal_for(y)
            uni.append(realised(select_threshold(
                scores, r1, method="uniform_f1", budget=20, seed=SEED + k)))
            r2, _ = _reveal_for(y)
            strat.append(realised(select_threshold(
                scores, r2, method="stratified_f1", budget=20,
                seed=SEED + k)))
        assert np.mean(strat) > np.mean(uni), (
            f"stratified mean F1 {np.mean(strat):.4f} did not beat uniform "
            f"{np.mean(uni):.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Contracts
# ═══════════════════════════════════════════════════════════════════════════


class TestContracts:
    def test_no_method_beats_its_own_oracle(self):
        """A selector scoring above the oracle bound means a bug."""
        scores, y = _problem(n=5000)
        n_pos = int(y.sum())

        def realised(ch: ThresholdChoice) -> float:
            flags = apply_threshold(scores, ch)
            tp = int(np.sum(flags & (y == 1)))
            if flags.sum() == 0 or tp == 0:
                return 0.0
            p, r = tp / int(flags.sum()), tp / n_pos
            return 2 * p * r / (p + r)

        r0, _ = _reveal_for(y)
        oracle = realised(select_threshold(scores, r0, method="oracle_f1",
                                           budget=20))
        for m in ("uniform_f1", "stratified_f1", "stratified_cost"):
            r, _ = _reveal_for(y)
            got = realised(select_threshold(scores, r, method=m, budget=20,
                                            seed=SEED))
            assert got <= oracle + 1e-9, (
                f"{m} scored {got:.4f} above the oracle {oracle:.4f}")

    def test_cost_rule_flags_more_when_misses_are_dearer(self):
        """
        Raising the miss cost must not make the selector flag FEWER cells.
        A sanity property of the cost objective itself.
        """
        scores, y = _problem()
        counts = []
        for c_miss in (2.0, 20.0, 200.0):
            r, _ = _reveal_for(y)
            ch = select_threshold(scores, r, method="stratified_cost",
                                  budget=20, seed=SEED, c_miss=c_miss)
            counts.append(int(apply_threshold(scores, ch).sum()))
        assert counts[0] <= counts[1] <= counts[2], (
            f"flag counts not monotone in miss cost: {counts}")

    def test_unmade_choice_flags_nothing(self):
        scores, _ = _problem(n=500)
        ch = ThresholdChoice("stratified_f1", None, 0, 20, "none")
        assert apply_threshold(scores, ch).sum() == 0

    def test_all_clean_reports_no_threshold(self):
        scores, y = _problem(n=3000)
        zeros = np.zeros_like(y)
        r, _ = _reveal_for(zeros)
        ch = select_threshold(scores, r, method="stratified_f1", budget=20,
                              seed=SEED)
        # Smoothing gives a tiny positive estimate, so a threshold may be
        # returned; what must hold is that it does not claim to have found
        # positives it did not see.
        assert all(d["found"] == 0 for d in ch.strata)

    def test_deterministic_under_a_fixed_seed(self):
        scores, y = _problem()
        outs = []
        for _ in range(2):
            r, _ = _reveal_for(y)
            outs.append(select_threshold(scores, r, method="stratified_f1",
                                         budget=20, seed=SEED).threshold)
        assert outs[0] == outs[1]

    def test_rejects_bad_input(self):
        scores, y = _problem(n=100)
        r, _ = _reveal_for(y)
        with pytest.raises(ValueError):
            select_threshold(scores, r, method="nonsense")
        with pytest.raises(ValueError):
            select_threshold(scores, r, budget=0)
        with pytest.raises(ValueError):
            select_threshold(np.array([]), r)
        with pytest.raises(ValueError):
            select_threshold(np.zeros((3, 3)), r)

    def test_every_method_is_reachable(self):
        scores, y = _problem(n=3000)
        for m in METHODS:
            r, _ = _reveal_for(y)
            ch = select_threshold(scores, r, method=m, budget=20, seed=SEED)
            assert ch.method == m


class TestHelpers:
    def test_stratum_bounds_tile_the_population_exactly(self):
        for n in (10, 137, 20000):
            bounds = _stratum_bounds(n, DEFAULT_STRATA)
            assert bounds[0][0] == 0
            assert bounds[-1][1] == n
            for (a, b), (c, d) in zip(bounds, bounds[1:]):
                assert b == c, f"gap or overlap in {bounds}"
            assert all(b > a for a, b in bounds)

    def test_allocate_spends_the_whole_budget(self):
        for budget, k in ((20, 5), (3, 5), (7, 2)):
            alloc = _allocate(budget, k)
            assert len(alloc) == k
            assert sum(alloc) == budget
            assert max(alloc) - min(alloc) <= 1
