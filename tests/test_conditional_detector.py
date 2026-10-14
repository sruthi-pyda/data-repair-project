"""
tests/test_conditional_detector.py
----------------------------------
Route R2 tests for src.detectors.conditional.ConditionalDetector.

The suite is built around one falsifiable core claim and one falsifiable
anti-claim, because that is the only honest way to test a detection idea:

  CORE CLAIM (test_synthetic_dependent_column_is_detected)
      On a table where column B is a deterministic function of column A, and
      5% of B is replaced by *other valid B values*, conditional modelling
      recovers the corruptions (PR-AUC > 0.8) while marginal rarity cannot
      (PR-AUC near prevalence). This is exactly Adult's label-flip case, in
      miniature, with the answer known by construction. If this test fails,
      the idea does not work and route R2 is dead.

  ANTI-CLAIM (test_pure_noise_column_is_not_confidently_flagged)
      On a column that nothing in the row predicts, the detector must NOT
      separate corrupted from clean cells (PR-AUC near prevalence). A detector
      that "wins" here would simply be flagging everything, and the core claim
      above would be meaningless.

Every expected value is either known by construction (the synthetic tables)
or re-derived independently inside the test (PR-AUC via sklearn, frequencies
via pandas) rather than by calling the code under test.

Coverage:
  1. Core claim: synthetic deterministic dependency
  2. Anti-claim: pure-noise negative control
  3. Ablation is real: mode="marginal" differs from mode="full"
  4. mode="ratio" also recovers the dependency
  5. Leakage: no parameter named *clean* anywhere in the public API
  6. Leakage: train_mask rows only -- sentinel spy on estimator.fit
  7. Output contract: shape / index / columns / dtype / [0, 1] range
  8. Determinism under a fixed seed
  9. Degenerate columns: all-NaN, constant, ID-like
 10. Error handling: score before fit, bad mode, bad mask, column mismatch
 11. Missing-cell policy (nan_score) is explicit and configurable
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import average_precision_score

from src.detectors.conditional import (
    MODES,
    RESIDUAL_TAILS,
    ColumnModel,
    ConditionalDetector,
    _residual_tail,
)

SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic data with a known answer
# ═══════════════════════════════════════════════════════════════════════════════


def make_synthetic(
    n: int = 2000,
    corruption_rate: float = 0.05,
    seed: int = SEED,
):
    """
    Build a table where the ground truth is known by construction.

    Columns
    -------
    a      20 uniform integer levels -- the predictor
    b      b = 2 * a -- DETERMINISTIC in a, so P(b | a) is a point mass
    noise  10 uniform integer levels, independent of everything

    ``corruption_rate`` of b's cells, and the same fraction of noise's cells,
    are replaced by a *different value drawn from that column's own valid
    value set*. Nothing becomes illegal, nothing goes out of range, nothing
    becomes NaN -- marginally the table is untouched. The only trace of the
    corruption in b is that it breaks b's relationship with a; in noise there
    is no relationship to break, which is what makes it the control.

    Returns
    -------
    (df, corrupt_b, corrupt_noise) : DataFrame, bool array, bool array
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 20, n)
    b = a * 2
    noise = rng.integers(0, 10, n)
    df = pd.DataFrame({"a": a, "b": b, "noise": noise})

    def corrupt(col: str, values: np.ndarray) -> np.ndarray:
        flags = rng.random(n) < corruption_rate
        valid = np.unique(values)
        new = df[col].to_numpy().copy()
        for i in np.flatnonzero(flags):
            alternatives = valid[valid != new[i]]
            new[i] = rng.choice(alternatives)
        df[col] = new
        return flags

    corrupt_b = corrupt("b", b)
    corrupt_noise = corrupt("noise", noise)
    return df, corrupt_b, corrupt_noise


def pr_auc(labels: np.ndarray, scores: pd.Series | np.ndarray) -> float:
    """PR-AUC, computed independently of the code under test."""
    return float(average_precision_score(np.asarray(labels), np.asarray(scores)))


@pytest.fixture(scope="module")
def synthetic():
    return make_synthetic()


@pytest.fixture(scope="module")
def synthetic_scores(synthetic):
    """One fit, scored in all three modes, so modes compare on equal models."""
    df, _, _ = synthetic
    det = ConditionalDetector(mode="full", seed=SEED).fit(df)
    return {m: det.score_cells(df, mode=m) for m in MODES}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. THE CORE CLAIM
# ═══════════════════════════════════════════════════════════════════════════════


class TestCoreClaim:
    def test_synthetic_dependent_column_is_detected(self, synthetic, synthetic_scores):
        """
        Conditional modelling recovers valid-value-for-valid-value corruption
        that marginal rarity provably cannot see.
        """
        _, corrupt_b, _ = synthetic
        prevalence = float(corrupt_b.mean())

        ap_full = pr_auc(corrupt_b, synthetic_scores["full"]["b"])
        ap_marginal = pr_auc(corrupt_b, synthetic_scores["marginal"]["b"])

        assert ap_full > 0.8, (
            f"conditional PR-AUC on b is {ap_full:.4f}, expected > 0.8 "
            f"(prevalence {prevalence:.4f}) -- the R2 idea does not work"
        )
        # The control must sit at chance level: by construction every observed
        # value of b is a legitimate, commonly-occurring value of b.
        assert ap_marginal < 3 * prevalence, (
            f"marginal PR-AUC on b is {ap_marginal:.4f}, expected near "
            f"prevalence {prevalence:.4f} -- the control is not a control"
        )
        assert ap_full > 5 * ap_marginal

    def test_corruption_is_marginally_invisible_by_construction(self, synthetic):
        """
        Independent check that the synthetic corruption really is of the hard
        kind: every post-corruption value of b is a value b legitimately
        takes, so no frequency- or range-based rule could see it.
        """
        df, corrupt_b, _ = synthetic
        legal_values = set(range(0, 40, 2))
        assert set(df["b"].unique()) <= legal_values
        corrupted_values = set(df.loc[corrupt_b, "b"].unique())
        clean_values = set(df.loc[~corrupt_b, "b"].unique())
        assert corrupted_values <= clean_values, (
            "a corrupted value appeared that never occurs in clean rows -- "
            "the synthetic corruption is detectable marginally and the test "
            "would be measuring the wrong thing"
        )

    def test_ratio_mode_also_recovers_the_dependency(self, synthetic, synthetic_scores):
        """mode='ratio' isolates 'marginally common, conditionally rare'."""
        _, corrupt_b, _ = synthetic
        assert pr_auc(corrupt_b, synthetic_scores["ratio"]["b"]) > 0.8


# ═══════════════════════════════════════════════════════════════════════════════
# 2. THE ANTI-CLAIM (negative control)
# ═══════════════════════════════════════════════════════════════════════════════


class TestNegativeControl:
    def test_pure_noise_column_is_not_confidently_flagged(
        self, synthetic, synthetic_scores
    ):
        """
        A column nothing predicts must come out at chance. A detector that
        scored well here would just be flagging everything.
        """
        _, _, corrupt_noise = synthetic
        prevalence = float(corrupt_noise.mean())
        ap = pr_auc(corrupt_noise, synthetic_scores["full"]["noise"])
        assert ap < 4 * prevalence, (
            f"PR-AUC on the pure-noise column is {ap:.4f} at prevalence "
            f"{prevalence:.4f}; the detector is not discriminating, it is "
            "flagging indiscriminately"
        )

    def test_noise_column_scores_are_not_all_extreme(self, synthetic, synthetic_scores):
        """
        Sanity on the same point from the score distribution rather than the
        metric: an unpredictable 10-level column should get middling
        1 - P(value) scores (around 1 - 1/10), not a wall of 1.0.
        """
        scores = synthetic_scores["full"]["noise"].to_numpy()
        assert scores.mean() < 0.99
        assert np.percentile(scores, 50) < 0.99


# ═══════════════════════════════════════════════════════════════════════════════
# 3. The ablation is real, not a no-op
# ═══════════════════════════════════════════════════════════════════════════════


class TestAblationIsReal:
    def test_marginal_differs_from_full(self, synthetic):
        """Separately-constructed detectors, as the public API is used."""
        df, _, _ = synthetic
        full = ConditionalDetector(mode="full", seed=SEED).fit_score(df)
        marginal = ConditionalDetector(mode="marginal", seed=SEED).fit_score(df)
        assert not np.allclose(full["b"].to_numpy(), marginal["b"].to_numpy()), (
            "mode='marginal' produced the same scores as mode='full'; the "
            "ablation is a no-op and proves nothing"
        )

    def test_marginal_mode_fits_no_estimator(self, synthetic):
        """The control must genuinely not look at the rest of the row."""
        df, _, _ = synthetic
        det = ConditionalDetector(mode="marginal", seed=SEED).fit(df)
        assert all(m.estimator is None for m in det.column_models_.values())

    def test_marginal_detector_refuses_conditional_scoring(self, synthetic):
        df, _, _ = synthetic
        det = ConditionalDetector(mode="marginal", seed=SEED).fit(df)
        with pytest.raises(RuntimeError, match="no conditional estimators"):
            det.score_cells(df, mode="full")

    def test_marginal_scores_match_an_independent_frequency_computation(self):
        """
        Re-derive mode='marginal' from pandas value_counts instead of trusting
        the implementation.
        """
        df = pd.DataFrame({"x": list("aaaaaaaabc") * 10, "y": list("abcdefghij") * 10})
        det = ConditionalDetector(mode="marginal", seed=SEED).fit(df)
        scores = det.score_cells(df)
        expected_freq = df["x"].value_counts(normalize=True)
        for value, freq in expected_freq.items():
            rows = df.index[df["x"] == value]
            np.testing.assert_allclose(
                scores.loc[rows, "x"].to_numpy(),
                1.0 - freq,
                rtol=1e-9,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LEAKAGE (CLAUDE.md Section 3 -- non-negotiable)
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoLeakage:
    def test_no_public_parameter_mentions_clean(self):
        """
        No entry point may accept clean data, by any spelling. Scans __init__
        and every public method of the detector plus the ColumnModel fields.
        """
        targets = [ConditionalDetector.__init__] + [
            getattr(ConditionalDetector, name)
            for name in dir(ConditionalDetector)
            if not name.startswith("_") and callable(getattr(ConditionalDetector, name))
        ]
        offenders = []
        for func in targets:
            for param in inspect.signature(func).parameters:
                if "clean" in param.lower():
                    offenders.append(f"{func.__qualname__}({param})")
        assert offenders == [], f"clean-data parameters found: {offenders}"

        field_offenders = [
            f for f in ColumnModel.__dataclass_fields__ if "clean" in f.lower()
        ]
        assert field_offenders == [], field_offenders

    def test_source_never_reads_a_clean_file_or_frame(self):
        """
        Belt-and-braces textual check: the module must not reference clean.csv
        or a clean_df identifier outside prose.
        """
        import src.detectors.conditional as module

        with open(module.__file__, encoding="utf-8") as fh:
            lines = fh.readlines()
        code_lines = [
            line
            for line in lines
            if "clean" in line.lower()
            and not line.lstrip().startswith("#")
            and "clean_df" in line.replace("``clean_df``", "")
        ]
        assert code_lines == [], f"clean_df referenced in code: {code_lines}"

    def test_fit_signature_is_dirty_only(self):
        params = list(inspect.signature(ConditionalDetector.fit).parameters)
        assert params == ["self", "dirty_df", "profile", "train_mask"]

    def test_train_mask_rows_only_reach_the_estimator(self, monkeypatch):
        """
        Sentinel spy. Eval rows carry values that occur nowhere in the train
        rows; if any of them reached an estimator's fit(), the sentinel would
        show up in the recorded X or y.

        This is the strong form of the leakage assertion: it inspects what the
        underlying sklearn estimators were actually handed, not what the
        detector claims.
        """
        n = 600
        rng = np.random.default_rng(SEED)
        df = pd.DataFrame(
            {
                "cat": [f"c{v}" for v in rng.integers(0, 10, n)],
                "b": rng.integers(0, 10, n),
                "num": rng.normal(0.0, 1.0, n),
            }
        )
        train_mask = np.zeros(n, dtype=bool)
        train_mask[: n // 2] = True

        # Sentinels placed only on eval rows: a category string that occurs
        # nowhere in the train rows, and a numeric value far outside them.
        cat_sentinel = "SENTINEL"
        num_sentinel = 123456.0
        df.loc[~train_mask, "cat"] = cat_sentinel
        df.loc[~train_mask, "num"] = num_sentinel

        seen: list[tuple[np.ndarray, np.ndarray]] = []

        def make_spy(original):
            def spy(self, X, y, *args, **kwargs):
                seen.append((np.asarray(X, dtype=float), np.asarray(y, dtype=float)))
                return original(self, X, y, *args, **kwargs)

            return spy

        monkeypatch.setattr(
            HistGradientBoostingClassifier,
            "fit",
            make_spy(HistGradientBoostingClassifier.fit),
        )
        monkeypatch.setattr(
            HistGradientBoostingRegressor,
            "fit",
            make_spy(HistGradientBoostingRegressor.fit),
        )

        det = ConditionalDetector(mode="full", seed=SEED).fit(df, train_mask=train_mask)

        assert seen, "no estimator was fitted -- the spy proves nothing"
        n_train = int(train_mask.sum())
        for X, y in seen:
            assert X.shape[0] <= n_train, (
                f"estimator received {X.shape[0]} rows but only {n_train} "
                "train rows exist"
            )
            assert not np.isclose(X, num_sentinel).any(), (
                "an eval-row numeric sentinel reached estimator.fit()"
            )
            assert not np.isclose(y, num_sentinel).any()

        # "SENTINEL" was never seen during fitting, so it must not even be a
        # known category: the encoder maps it to NaN rather than to a code.
        is_cat, cats = det._encoding["cat"]
        assert is_cat
        assert cat_sentinel not in cats, (
            "an eval-row category leaked into the fitted encoding"
        )
        assert det.column_models_["cat"].classes is not None
        assert cat_sentinel not in det.column_models_["cat"].classes, (
            "an eval-row value leaked into the fitted class list"
        )

    def test_train_mask_changes_the_learned_statistics(self):
        """
        A mask that is honoured must change what is learned. If scores came
        out identical with and without the mask, the mask would be ignored.
        """
        df, _, _ = make_synthetic(n=800, seed=7)
        n = len(df)
        mask = np.zeros(n, dtype=bool)
        mask[: n // 2] = True
        all_rows = ConditionalDetector(mode="full", seed=SEED).fit_score(df)
        half = ConditionalDetector(mode="full", seed=SEED).fit_score(
            df, train_mask=mask
        )
        assert not np.allclose(all_rows.to_numpy(), half.to_numpy())

    def test_profile_argument_is_optional_and_only_used_for_id_columns(self):
        """
        profile is the one place a clean-derived object may enter; it must be
        optional and must not change anything beyond the skip list.
        """

        class FakeProfile:
            id_columns = ["a"]

        df, _, _ = make_synthetic(n=400, seed=11)
        without = ConditionalDetector(mode="full", seed=SEED).fit(df)
        with_profile = ConditionalDetector(mode="full", seed=SEED).fit(
            df, profile=FakeProfile()
        )
        assert "a" not in without.skipped_columns_
        assert with_profile.skipped_columns_["a"] == "id_like_profile"
        assert (
            with_profile.score_cells(df)["a"].to_numpy() == 0.0
        ).all(), "a profile-skipped column must score 0.0"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Output contract
# ═══════════════════════════════════════════════════════════════════════════════


class TestOutputContract:
    @pytest.mark.parametrize("mode", MODES)
    def test_shape_index_and_columns_match_input(self, mode):
        df, _, _ = make_synthetic(n=300, seed=3)
        df.index = pd.RangeIndex(start=1000, stop=1300)  # non-default index
        scores = ConditionalDetector(mode=mode, seed=SEED).fit_score(df)
        assert scores.shape == df.shape
        assert list(scores.columns) == list(df.columns)
        pd.testing.assert_index_equal(scores.index, df.index)

    @pytest.mark.parametrize("mode", MODES)
    def test_scores_are_floats_in_unit_interval(self, mode):
        df, _, _ = make_synthetic(n=300, seed=4)
        scores = ConditionalDetector(mode=mode, seed=SEED).fit_score(df)
        assert all(str(dt) == "float64" for dt in scores.dtypes)
        values = scores.to_numpy()
        assert np.isfinite(values).all()
        assert values.min() >= 0.0
        assert values.max() <= 1.0

    def test_fit_on_train_then_score_a_different_frame(self):
        """Fit and score need not be the same frame, only the same columns."""
        df, _, _ = make_synthetic(n=600, seed=5)
        train = df.iloc[:400]
        evaluate = df.iloc[400:]
        det = ConditionalDetector(mode="full", seed=SEED).fit(train)
        scores = det.score_cells(evaluate)
        assert scores.shape == evaluate.shape
        pd.testing.assert_index_equal(scores.index, evaluate.index)

    def test_report_lists_every_column(self, synthetic):
        df, _, _ = synthetic
        report = ConditionalDetector(mode="full", seed=SEED).fit(df).report()
        assert set(report["columns"]) == set(df.columns)
        assert report["mode"] == "full"
        assert report["n_train_rows"] == len(df)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Determinism
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    @pytest.mark.parametrize("mode", MODES)
    def test_same_seed_gives_identical_scores(self, mode):
        df, _, _ = make_synthetic(n=500, seed=6)
        first = ConditionalDetector(mode=mode, seed=SEED).fit_score(df)
        second = ConditionalDetector(mode=mode, seed=SEED).fit_score(df)
        pd.testing.assert_frame_equal(first, second)

    def test_repeated_scoring_of_one_fitted_detector_is_stable(self, synthetic):
        df, _, _ = synthetic
        det = ConditionalDetector(mode="full", seed=SEED).fit(df)
        pd.testing.assert_frame_equal(det.score_cells(df), det.score_cells(df))


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Degenerate columns
# ═══════════════════════════════════════════════════════════════════════════════


class TestDegenerateColumns:
    def _frame(self, n: int = 200) -> pd.DataFrame:
        rng = np.random.default_rng(SEED)
        base = rng.integers(0, 5, n)
        return pd.DataFrame(
            {
                "signal": base,
                "dependent": base * 3,
                "all_nan": np.full(n, np.nan),
                "constant": np.ones(n),
                "id_like": np.arange(n),
            }
        )

    @pytest.mark.parametrize("mode", MODES)
    def test_degenerate_columns_score_zero_and_do_not_crash(self, mode):
        df = self._frame()
        det = ConditionalDetector(mode=mode, seed=SEED).fit(df)
        scores = det.score_cells(df)
        for col in ("all_nan", "constant", "id_like"):
            assert (scores[col].to_numpy() == 0.0).all(), col

    def test_skip_reasons_are_reported(self):
        df = self._frame()
        det = ConditionalDetector(mode="full", seed=SEED).fit(df)
        assert det.skipped_columns_["all_nan"] == "all_missing"
        assert det.skipped_columns_["constant"] == "constant"
        assert det.skipped_columns_["id_like"].startswith("id_like_cardinality")
        assert "signal" not in det.skipped_columns_

    def test_skipped_columns_are_excluded_from_features(self):
        df = self._frame()
        det = ConditionalDetector(mode="full", seed=SEED).fit(df)
        assert set(det.feature_columns_).isdisjoint(
            {"all_nan", "constant", "id_like"}
        )

    def test_single_informative_column_pair_still_works(self):
        """Two columns only: the target has exactly one other column to use."""
        rng = np.random.default_rng(SEED)
        a = rng.integers(0, 6, 300)
        df = pd.DataFrame({"a": a, "b": a * 5})
        det = ConditionalDetector(mode="full", seed=SEED).fit(df)
        assert det.column_models_["b"].role == "classifier"
        assert det.score_cells(df).shape == df.shape

    def test_one_column_frame_cannot_condition_and_is_skipped(self):
        df = pd.DataFrame({"only": np.random.default_rng(SEED).integers(0, 5, 100)})
        det = ConditionalDetector(mode="full", seed=SEED).fit(df)
        assert det.column_models_["only"].reason == "no_other_columns_to_condition_on"
        assert (det.score_cells(df)["only"].to_numpy() == 0.0).all()

    def test_high_cardinality_numeric_column_uses_a_regressor(self):
        rng = np.random.default_rng(SEED)
        x = rng.normal(0.0, 1.0, 400)
        df = pd.DataFrame({"x": x, "y": 2.0 * x + rng.normal(0.0, 0.01, 400)})
        det = ConditionalDetector(mode="full", seed=SEED).fit(df)
        assert det.column_models_["y"].role == "regressor"
        assert det.column_models_["y"].residual_scale is not None

    def test_numeric_residual_outlier_scores_higher_than_a_typical_cell(self):
        """
        Independent check of the numeric branch: a cell whose value violates a
        near-exact linear relation must outrank a conforming cell.
        """
        rng = np.random.default_rng(SEED)
        n = 600
        x = rng.normal(0.0, 1.0, n)
        y = 3.0 * x + rng.normal(0.0, 0.01, n)
        df = pd.DataFrame({"x": x, "y": y})
        df.loc[0, "y"] = float(y.max())  # a legal y value, wrong for this x
        det = ConditionalDetector(mode="full", seed=SEED).fit(df)
        scores = det.score_cells(df)["y"]
        assert scores.iloc[0] > np.percentile(scores.to_numpy(), 99)


class TestResidualTail:
    """
    The numeric branch's tail family is a measured choice, not a stylistic
    one, so it gets its own assertions.
    """

    def test_gaussian_tail_saturates_and_cauchy_does_not(self):
        """
        The reason "cauchy" is the default. Past |z| ~ 8.3 the Gaussian tail
        is below 1e-17, so the reported score 1 - tail rounds to exactly 1.0
        in float64: every strong outlier collapses onto the identical score
        and PR-AUC -- a pure ranking metric -- loses all resolution there.
        The Cauchy tail keeps distinct scores out to |z| = 1e6.
        """
        z = np.array([9.0, 20.0, 500.0, 1e6])
        gaussian_score = 1.0 - _residual_tail(z, "gaussian")
        cauchy_score = 1.0 - _residual_tail(z, "cauchy")

        assert (gaussian_score == 1.0).all(), (
            "the gaussian score did not saturate; the stated reason for the "
            "cauchy default no longer holds and the default should be re-measured"
        )
        assert len(set(gaussian_score.tolist())) == 1

        assert (cauchy_score < 1.0).all()
        assert list(cauchy_score) == sorted(cauchy_score)
        assert len(set(cauchy_score.tolist())) == len(z)

    def test_both_tails_are_monotone_and_bounded(self):
        z = np.linspace(0.0, 30.0, 200)
        for tail in RESIDUAL_TAILS:
            values = _residual_tail(z, tail)
            assert values.min() >= 0.0
            assert values.max() <= 1.0
            assert np.all(np.diff(values) <= 1e-12), tail
            assert values[0] == pytest.approx(1.0)

    def test_invalid_residual_tail_raises(self):
        with pytest.raises(ValueError, match="residual_tail"):
            ConditionalDetector(residual_tail="student-t")

    def test_tail_choice_changes_numeric_scores(self):
        rng = np.random.default_rng(SEED)
        n = 400
        x = rng.normal(0.0, 1.0, n)
        df = pd.DataFrame({"x": x, "y": 2.0 * x + rng.normal(0.0, 0.05, n)})
        cauchy = ConditionalDetector(
            mode="full", seed=SEED, residual_tail="cauchy"
        ).fit_score(df)
        gaussian = ConditionalDetector(
            mode="full", seed=SEED, residual_tail="gaussian"
        ).fit_score(df)
        assert not np.allclose(cauchy["y"].to_numpy(), gaussian["y"].to_numpy())


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Missing-cell policy
# ═══════════════════════════════════════════════════════════════════════════════


class TestMissingCellPolicy:
    def _frame(self):
        rng = np.random.default_rng(SEED)
        a = rng.integers(0, 6, 300)
        df = pd.DataFrame({"a": a, "b": (a * 4).astype(float)})
        df.loc[[0, 1, 2], "b"] = np.nan
        return df

    @pytest.mark.parametrize("mode", MODES)
    def test_default_nan_score_is_one(self, mode):
        df = self._frame()
        scores = ConditionalDetector(mode=mode, seed=SEED).fit_score(df)
        np.testing.assert_allclose(scores.loc[[0, 1, 2], "b"].to_numpy(), 1.0)

    def test_nan_score_is_configurable(self):
        df = self._frame()
        scores = ConditionalDetector(mode="full", seed=SEED, nan_score=0.0).fit_score(df)
        np.testing.assert_allclose(scores.loc[[0, 1, 2], "b"].to_numpy(), 0.0)

    def test_nan_score_is_validated(self):
        with pytest.raises(ValueError, match="nan_score"):
            ConditionalDetector(nan_score=1.5)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Error handling
# ═══════════════════════════════════════════════════════════════════════════════


class TestErrorHandling:
    def test_score_before_fit_raises(self):
        df, _, _ = make_synthetic(n=100, seed=9)
        with pytest.raises(RuntimeError, match="before fit"):
            ConditionalDetector().score_cells(df)

    def test_report_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="before fit"):
            ConditionalDetector().report()

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            ConditionalDetector(mode="conditional")

    def test_non_boolean_train_mask_raises(self):
        df, _, _ = make_synthetic(n=100, seed=10)
        with pytest.raises(TypeError, match="boolean"):
            ConditionalDetector().fit(df, train_mask=np.zeros(len(df), dtype=int))

    def test_wrong_length_train_mask_raises(self):
        df, _, _ = make_synthetic(n=100, seed=10)
        with pytest.raises(ValueError, match="length"):
            ConditionalDetector().fit(df, train_mask=np.ones(7, dtype=bool))

    def test_empty_train_mask_raises(self):
        df, _, _ = make_synthetic(n=100, seed=10)
        with pytest.raises(ValueError, match="zero rows"):
            ConditionalDetector().fit(df, train_mask=np.zeros(len(df), dtype=bool))

    def test_column_mismatch_at_score_time_raises(self):
        df, _, _ = make_synthetic(n=200, seed=12)
        det = ConditionalDetector(mode="full", seed=SEED).fit(df)
        with pytest.raises(ValueError, match="different columns"):
            det.score_cells(df.drop(columns=["noise"]))

    def test_no_columns_raises(self):
        with pytest.raises(ValueError, match="no columns"):
            ConditionalDetector().fit(pd.DataFrame(index=range(5)))

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError, match="DataFrame"):
            ConditionalDetector().fit([[1, 2], [3, 4]])


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Mixed-type columns (Breast Cancer's format-error shape)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMixedTypeColumns:
    def test_format_error_token_is_an_unseen_category_and_scores_high(self):
        """
        A column of numeric-looking strings with an injected '1a0' token: the
        token never appears in the fitting rows, so P(observed | row) is 0 and
        the score is maximal. This mirrors Breast Cancer's 'Bare Nuclei'.
        """
        rng = np.random.default_rng(SEED)
        n = 400
        a = rng.integers(1, 6, n)
        df = pd.DataFrame({"a": a, "b": [str(v * 2) for v in a]})
        train_mask = np.zeros(n, dtype=bool)
        train_mask[: n // 2] = True
        df.loc[n - 1, "b"] = "1a0"  # eval row, unseen token

        det = ConditionalDetector(mode="full", seed=SEED).fit(
            df, train_mask=train_mask
        )
        scores = det.score_cells(df)
        assert scores.loc[n - 1, "b"] == pytest.approx(1.0)

    def test_object_column_is_encoded_as_a_categorical_feature(self):
        rng = np.random.default_rng(SEED)
        n = 300
        letters = rng.choice(list("abcde"), n)
        df = pd.DataFrame({"letter": letters, "num": [ord(c) for c in letters]})
        det = ConditionalDetector(mode="full", seed=SEED).fit(df)
        is_cat, cats = det._encoding["letter"]
        assert is_cat
        assert cats == sorted(set(letters))
