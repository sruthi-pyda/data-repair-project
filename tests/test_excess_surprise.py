"""
tests/test_excess_surprise.py
-----------------------------
Tests for `src/detectors/excess_surprise.py`.

The decisive test is `TestDifficultyCancellation`. Excess Surprise exists for
exactly one reason — to cancel the per-column model error rate, which
`docs/w8_plan.md` §8 identifies as the nuisance parameter that defeated four
previous mechanisms. If a clean-but-hard column does not come out near a
clean-but-easy one, the subtraction has failed and the mechanism is worthless
regardless of how it scores on real data. That test is written to be able to
fail and is checked first.

All synthetic fixtures build their own ground truth, so the expected answers
are derived independently of the code under test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.detectors.excess_surprise import (
    DEFAULT_BINS,
    ExcessSurprise,
    _codes,
)

SEED = 42
N_ROWS = 3000


def _frame(
    n: int = N_ROWS,
    corrupt_rate: float = 0.0,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    A frame with three columns of KNOWN character:

      `easy_clean`   a deterministic function of the features -> low model
                     error, clean
      `hard_clean`   nearly independent of the features -> HIGH model error,
                     but still clean
      `target`       the same deterministic function as `easy_clean`, with
                     `corrupt_rate` of its cells replaced by an unrelated
                     draw (x-independent, as real corruption is)

    So `easy_clean` and `hard_clean` differ hugely in difficulty while both
    being clean, which is precisely the contrast the mechanism must cancel.
    """
    rng = np.random.default_rng(seed)
    f1 = rng.integers(0, 5, size=n)
    f2 = rng.integers(0, 5, size=n)

    easy = (f1 * 2) % 5
    hard = rng.integers(0, 5, size=n)
    target = (f1 * 2) % 5

    if corrupt_rate > 0:
        k = int(round(corrupt_rate * n))
        idx = rng.choice(n, size=k, replace=False)
        # x-independent replacement, and always a DIFFERENT value so every
        # chosen cell is genuinely corrupted.
        target = target.copy()
        for i in idx:
            choices = [v for v in range(5) if v != target[i]]
            target[i] = choices[int(rng.integers(0, len(choices)))]

    return pd.DataFrame({
        "f1": f1, "f2": f2,
        "easy_clean": easy, "hard_clean": hard, "target": target,
    })


@pytest.fixture(scope="module")
def clean_fit() -> ExcessSurprise:
    return ExcessSurprise(seed=SEED, max_iter=40).fit(_frame(corrupt_rate=0.0))


# ═══════════════════════════════════════════════════════════════════════════
# THE DECISIVE TEST
# ═══════════════════════════════════════════════════════════════════════════


class TestDifficultyCancellation:
    """The mechanism's whole justification. If this fails, nothing else matters."""

    def test_hard_column_really_is_harder(self, clean_fit: ExcessSurprise):
        """Premise check: the fixture must actually differ in difficulty."""
        easy = clean_fit.columns_["easy_clean"]
        hard = clean_fit.columns_["hard_clean"]
        assert easy.scored and hard.scored
        # If this fails the fixture is broken, not the mechanism.
        assert hard.observed_error > easy.observed_error + 0.3, (
            f"fixture does not create a difficulty gap: easy "
            f"{easy.observed_error:.4f} vs hard {hard.observed_error:.4f}"
        )

    def test_excess_cancels_the_difficulty_gap(self, clean_fit: ExcessSurprise):
        """
        Both columns are CLEAN, so both excesses must be near zero and near
        each other -- even though their observed errors differ by >0.3.
        """
        easy = clean_fit.columns_["easy_clean"]
        hard = clean_fit.columns_["hard_clean"]
        observed_gap = abs(hard.observed_error - easy.observed_error)
        excess_gap = abs(hard.excess_self - easy.excess_self)
        assert excess_gap < observed_gap / 3.0, (
            f"subtraction did not cancel difficulty: observed gap "
            f"{observed_gap:.4f} but excess gap {excess_gap:.4f} "
            f"(easy {easy.excess_self:.4f}, hard {hard.excess_self:.4f})"
        )

    def test_clean_columns_have_small_excess(self, clean_fit: ExcessSurprise):
        for name in ("easy_clean", "hard_clean"):
            e = clean_fit.columns_[name].excess_self
            assert abs(e) < 0.15, f"{name} clean but excess_self={e:.4f}"


# ═══════════════════════════════════════════════════════════════════════════
# Corruption sensitivity
# ═══════════════════════════════════════════════════════════════════════════


class TestCorruptionSensitivity:
    @pytest.mark.parametrize("rate", [0.05, 0.15, 0.30])
    def test_corrupted_target_exceeds_clean_columns(self, rate: float):
        det = ExcessSurprise(seed=SEED, max_iter=40).fit(
            _frame(corrupt_rate=rate))
        target = det.columns_["target"].excess_self
        easy = det.columns_["easy_clean"].excess_self
        assert target > easy, (
            f"at r={rate} corrupted target excess {target:.4f} did not exceed "
            f"clean easy_clean {easy:.4f}"
        )

    def test_excess_increases_with_rate(self):
        """
        Monotone in r, re-derived across four rates rather than asserted once.

        MEASURED SENSITIVITY, and it is a weakness worth stating rather than
        hiding. On this fixture:

            r = 0.00 -> excess = -0.0004
            r = 0.10 -> excess = +0.0092
            r = 0.25 -> excess = +0.0208
            r = 0.40 -> excess = +0.0334

        so the slope is only about **0.08 * r**, not ~1 * r. The reason is
        absorption: the model is cross-fitted on the CORRUPTED frame, so its
        own predictive distribution widens to accommodate the corruption and
        `expected_error` rises alongside `observed_error`. The gap survives,
        but heavily attenuated.

        Consequence, recorded before the real-data run: at adult's true
        `income` rate of ~0.0376 this predicts an excess near 0.003, which
        may well sit below the between-column spread of `excess` on real
        tables. If Excess Surprise fails on adult, weak sensitivity -- not
        failed cancellation -- is the expected reason.
        """
        rates = [0.0, 0.10, 0.25, 0.40]
        got = []
        for r in rates:
            det = ExcessSurprise(seed=SEED, max_iter=40).fit(
                _frame(corrupt_rate=r))
            got.append(det.columns_["target"].excess_self)
        for a, b in zip(got, got[1:]):
            assert b > a - 0.02, f"not monotone in r: {list(zip(rates, got))}"
        # A real rise, but calibrated to the measured ~0.08*r slope rather
        # than to an assumed one. Still fails if the signal disappears.
        assert got[-1] > got[0] + 0.02, (
            f"excess did not rise materially from r=0 to r=0.40: {got}"
        )
        assert got[-1] < got[0] + 0.40, (
            f"excess tracks r far more strongly than measured; the fixture or "
            f"the estimator changed and the documented slope is stale: {got}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Mechanics, leakage, edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestMechanics:
    def test_deterministic(self):
        a = ExcessSurprise(seed=SEED, max_iter=40).fit(_frame(corrupt_rate=0.2))
        b = ExcessSurprise(seed=SEED, max_iter=40).fit(_frame(corrupt_rate=0.2))
        for k in a.columns_:
            assert a.columns_[k].excess_self == b.columns_[k].excess_self

    def test_both_pairings_available(self, clean_fit: ExcessSurprise):
        for pairing in ("self", "top1"):
            scores = clean_fit.column_scores(pairing)
            assert "target" in scores
            assert scores["target"] is not None

    def test_ingredients_exposed_for_condition_9(self, clean_fit: ExcessSurprise):
        obs = clean_fit.ingredient_scores("observed")
        exp = clean_fit.ingredient_scores("expected")
        col = "hard_clean"
        # excess must literally equal observed - expected
        assert np.isclose(
            clean_fit.columns_[col].excess_self, obs[col] - exp[col])

    def test_constant_column_is_skipped_not_scored(self):
        df = _frame(corrupt_rate=0.0)
        df["constant"] = 7
        det = ExcessSurprise(seed=SEED, max_iter=40).fit(df)
        assert "constant" in det.skipped_columns_
        assert det.columns_["constant"].excess("self") is None

    def test_rejects_empty_and_non_frame(self):
        with pytest.raises(ValueError):
            ExcessSurprise().fit(pd.DataFrame())
        with pytest.raises(TypeError):
            ExcessSurprise().fit([1, 2, 3])  # type: ignore[arg-type]

    def test_rejects_bad_params(self):
        with pytest.raises(ValueError):
            ExcessSurprise(bins=1)
        with pytest.raises(ValueError):
            ExcessSurprise(folds=1)

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            ExcessSurprise().column_scores()

    def test_missing_values_get_their_own_code(self):
        """`missing` is a corruption type, so NaN must not be dropped."""
        df = _frame(corrupt_rate=0.0)
        df.loc[: len(df) // 4, "target"] = np.nan
        codes, sizes = _codes(df, DEFAULT_BINS)
        assert codes.shape == (len(df), df.shape[1])
        assert sizes[list(df.columns).index("target")] >= 2


class TestNoLeakage:
    def test_profile_argument_is_ignored(self):
        """
        A profile is accepted for interface parity and must change nothing,
        so no clean-derived statistic can enter through it.
        """
        df = _frame(corrupt_rate=0.2)
        a = ExcessSurprise(seed=SEED, max_iter=40).fit(df)
        b = ExcessSurprise(seed=SEED, max_iter=40).fit(
            df, profile={"anything": "at all"})
        for k in a.columns_:
            assert a.columns_[k].excess_self == b.columns_[k].excess_self

    def test_fit_signature_takes_no_labels_or_clean_frame(self):
        import inspect
        params = set(inspect.signature(ExcessSurprise.fit).parameters)
        assert params == {"self", "dirty_df", "profile"}, (
            f"fit() must not accept clean data or labels; got {params}"
        )
