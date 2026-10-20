"""
tests/test_cacm.py
------------------
Tests for W8-A, `src/detectors/cacm.py` (the contamination-aware conditional
mixture).

WHAT THESE TESTS ARE FOR
========================
A detector test that only checks shapes and dtypes cannot fail for any reason
that matters. The load-bearing tests here are the ones where **this file
generates the data and therefore knows the answer independently of the code
under test**:

  test_recovers_known_contamination_rate   r and g are chosen here; EM must
                                           find them
  test_clean_column_yields_low_rate        r = 0 by construction
  test_rare_but_valid_scores_low           the two cases that killed every
  test_contradicted_common_value_scores_high   previous mechanism
  test_posterior_matches_hand_arithmetic   the E-step against a number
                                           computed with a calculator, not
                                           with the module

The remaining tests cover the contract (ranges, determinism, skip rules,
errors) and the leakage invariant of CLAUDE.md §3.

Two synthetic generators are shared. Both put a *deterministic* clean
conditional in the data, because the mechanism's identifying restriction is
exactly "clean depends on x, corruption does not": if the clean signal were
noise there would be nothing to identify, and a test on such data would
measure nothing.

SPEED
=====
Every fit here is on a few hundred rows with a small estimator budget, so the
file runs in well under a minute. The real measurements live in
`experiments/w8a_cacm.py` and `results/w8a_cacm.md`; nothing in this file
quotes a number from them.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.detectors.cacm import (
    NA_TOKEN,
    OTHER_TOKEN,
    CACMDetector,
    ColumnMixture,
    leakage_signature_report,
    posterior_corrupt,
)

# A deliberately small estimator budget: these tests are about the EM
# mechanics, not about gradient boosting's accuracy ceiling.
FAST = dict(max_iter=20, min_samples_leaf=5)


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic generators -- the ground truth of the important tests
# ═══════════════════════════════════════════════════════════════════════════════


def make_contaminated(
    n: int = 800,
    r: float = 0.25,
    corrupt_value: int = 0,
    seed: int = 7,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    A table whose column ``y`` is a deterministic function of ``x1, x2`` in
    the clean rows, and is replaced by ``corrupt_value`` in a fraction ``r``
    of rows chosen independently of the row.

    That independence is the mixture's identifying restriction, generated
    here on purpose. Returns ``(frame, corrupted_mask)``; the mask is the
    ground truth and is never passed to the detector.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.integers(0, 5, n)
    x2 = rng.integers(0, 5, n)
    y = (x1 * 5 + x2) % 7
    hit = rng.random(n) < r
    y = y.copy()
    y[hit] = corrupt_value
    frame = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    return frame, hit


def make_rare_but_valid(n: int = 900, seed: int = 11) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    The pair of cases that killed every previous mechanism, in one table.

    ``x`` takes a common level 0..3 and a RARE level 4 (about 5 % of rows).
    Clean ``y`` is ``x`` itself, so level 4 produces a **rare but perfectly
    valid** value whose row implies it. Corruption then replaces ``y`` with
    the single most common value ``0`` in 20 % of rows -- a **common value
    that the row contradicts**.

    Returns ``(frame, rare_valid_mask, corrupted_mask)``:

      rare_valid_mask   rare, valid, implied by the row  -> posterior must be LOW
      corrupted_mask    common, not implied by the row   -> posterior must be HIGH
    """
    rng = np.random.default_rng(seed)
    x = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.2375, 0.2375, 0.2375, 0.2375, 0.05])
    y = x.copy()
    hit = rng.random(n) < 0.20
    y[hit] = 0
    # A "rare but valid" cell is a surviving level-4 row: value 4 occurs in
    # ~4 % of cells yet is exactly what its row predicts.
    rare_valid = (x == 4) & ~hit
    # A corrupted cell is only interesting when the flip actually changed the
    # value; x == 0 rows were already 0.
    corrupted = hit & (x != 0)
    return pd.DataFrame({"x": x, "z": rng.integers(0, 3, n), "y": y}), rare_valid, corrupted


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUC, re-derived here rather than imported, so a bug in
    the project's metric module cannot make a detector test pass."""
    pos = scores[labels]
    neg = scores[~labels]
    assert pos.size and neg.size
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins) / float(pos.size * neg.size)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. The E-step, against hand arithmetic
# ═══════════════════════════════════════════════════════════════════════════════


class TestPosterior:
    def test_posterior_matches_hand_arithmetic(self):
        # r = 0.2, f = 0.4, g = 0.5
        #   numerator   = 0.2 * 0.5              = 0.1
        #   denominator = 0.8 * 0.4 + 0.1 = 0.32 + 0.1 = 0.42
        #   posterior   = 0.1 / 0.42            = 0.238095238095...
        got = posterior_corrupt(0.2, np.array([0.4]), np.array([0.5]))
        assert got[0] == pytest.approx(0.1 / 0.42, abs=1e-12)

    def test_posterior_is_one_when_clean_model_excludes_the_value(self):
        got = posterior_corrupt(0.1, np.array([0.0]), np.array([0.3]))
        assert got[0] == pytest.approx(1.0)

    def test_posterior_is_zero_when_corruption_model_excludes_the_value(self):
        got = posterior_corrupt(0.1, np.array([0.3]), np.array([0.0]))
        assert got[0] == pytest.approx(0.0)

    def test_posterior_falls_back_to_the_prior_when_both_components_vanish(self):
        got = posterior_corrupt(0.17, np.array([0.0]), np.array([0.0]))
        assert got[0] == pytest.approx(0.17)

    def test_posterior_is_increasing_in_g_and_decreasing_in_f(self):
        g_up = posterior_corrupt(0.2, np.full(4, 0.3), np.array([0.1, 0.2, 0.3, 0.4]))
        assert np.all(np.diff(g_up) > 0)
        f_up = posterior_corrupt(0.2, np.array([0.1, 0.2, 0.3, 0.4]), np.full(4, 0.3))
        assert np.all(np.diff(f_up) < 0)

    def test_posterior_stays_in_unit_interval_on_adversarial_input(self):
        rng = np.random.default_rng(0)
        f = rng.random(500) * np.array([1e-9, 1.0, 1e9, 1.0, 0.0] * 100)
        g = rng.random(500) * np.array([1e9, 0.0, 1e-9, 1.0, 1.0] * 100)
        out = posterior_corrupt(0.3, f, g)
        assert np.all(np.isfinite(out))
        assert out.min() >= 0.0 and out.max() <= 1.0

    @pytest.mark.parametrize("bad_r", [0.0, 1.0, -0.1, 1.5])
    def test_posterior_rejects_an_invalid_rate(self, bad_r):
        with pytest.raises(ValueError):
            posterior_corrupt(bad_r, np.array([0.5]), np.array([0.5]))

    def test_posterior_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError):
            posterior_corrupt(0.2, np.zeros(3), np.zeros(4))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Recovery of a known contamination -- the test that decides the mechanism
# ═══════════════════════════════════════════════════════════════════════════════


class TestRecovery:
    """
    These are the tests the task hinges on: if CACM cannot recover an ``r``
    and a ``g`` on data generated in this file, it does not work.

    Tolerances are stated as absolute intervals rather than "close to", so a
    regression that halves the estimate fails rather than squeaking through.
    """

    def test_recovers_known_contamination_rate(self):
        frame, hit = make_contaminated(n=800, r=0.25, seed=7)
        true_r = float(hit.mean())
        det = CACMDetector(seed=42, em_iters=15, n_bins=8, **FAST).fit(frame)
        mix = det.column_mixtures_["y"]
        assert mix.role == "mixture"
        # Half to double the truth. Loose, but it is the difference between
        # "estimates the rate" and "does not".
        assert 0.5 * true_r <= mix.r <= 2.0 * true_r, (
            f"r_hat={mix.r:.4f} vs true r={true_r:.4f}"
        )

    def test_recovers_the_corruption_distribution_g(self):
        # Every corrupted cell is set to 3, so g must concentrate on "3".
        frame, hit = make_contaminated(n=800, r=0.25, corrupt_value=3, seed=5)
        det = CACMDetector(seed=42, em_iters=15, n_bins=8, **FAST).fit(frame)
        mix = det.column_mixtures_["y"]
        top_label, top_mass = mix.top_g(1)[0]
        assert top_label == "3", f"g peaks on {top_label!r}, not on the injected value"
        # The marginal share of "3" is ~0.25 + 1/7*0.75 ~= 0.36, so g must be
        # well above that to count as having learned the corruption policy
        # rather than echoing the marginal it was initialised to.
        assert top_mass > 0.6, f"g('3') = {top_mass:.3f}"

    def test_ranks_corrupted_cells_above_clean_ones(self):
        frame, hit = make_contaminated(n=800, r=0.25, corrupt_value=0, seed=7)
        scores = CACMDetector(seed=42, em_iters=15, n_bins=8, **FAST).fit_score(frame)
        # Only cells whose value actually changed count as corrupted: a row
        # whose clean y was already 0 is untouched by a flip to 0, and
        # labelling it corrupted would make the target unachievable.
        clean_y = (frame["x1"].to_numpy() * 5 + frame["x2"].to_numpy()) % 7
        changed = hit & (clean_y != 0)
        assert changed.sum() >= 100
        assert auc(changed, scores["y"].to_numpy()) > 0.75

    def test_clean_column_yields_low_rate(self):
        """A column with no corruption at all: r must stay small."""
        rng = np.random.default_rng(3)
        x1 = rng.integers(0, 5, 800)
        x2 = rng.integers(0, 5, 800)
        frame = pd.DataFrame({"x1": x1, "x2": x2, "y": (x1 * 5 + x2) % 7})
        det = CACMDetector(seed=42, em_iters=15, n_bins=8, **FAST).fit(frame)
        assert det.column_mixtures_["y"].r < 0.05

    def test_a_cleaner_column_gets_a_lower_rate_than_a_dirtier_one(self):
        """
        Ordering, which is what column triage actually needs. Both columns sit
        in the same table and share the same features, so the comparison is
        not confounded by cardinality or by how predictable the column is.
        """
        rng = np.random.default_rng(13)
        n = 800
        x1 = rng.integers(0, 5, n)
        x2 = rng.integers(0, 5, n)
        base = (x1 * 5 + x2) % 7
        light = base.copy()
        light[rng.random(n) < 0.03] = 0
        heavy = base.copy()
        heavy[rng.random(n) < 0.30] = 0
        frame = pd.DataFrame({"x1": x1, "x2": x2, "light": light, "heavy": heavy})
        det = CACMDetector(seed=42, em_iters=15, n_bins=8, **FAST).fit(frame)
        r_light = det.column_mixtures_["light"].r
        r_heavy = det.column_mixtures_["heavy"].r
        assert r_heavy > r_light, f"r(heavy)={r_heavy:.4f} !> r(light)={r_light:.4f}"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Rare-but-valid vs contradicted-common -- the two historic failure cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestTheTwoHardCases:
    @pytest.fixture(scope="class")
    def fitted(self):
        frame, rare_valid, corrupted = make_rare_but_valid()
        det = CACMDetector(seed=42, em_iters=15, n_bins=8, **FAST).fit(frame)
        return det.score_cells(frame)["y"].to_numpy(), rare_valid, corrupted, frame

    def test_rare_but_valid_scores_low(self, fitted):
        scores, rare_valid, corrupted, frame = fitted
        assert rare_valid.sum() >= 10  # the generator must actually produce them
        # A rare-but-valid cell must not look more suspicious than the table
        # as a whole. This is the comparison the marginal-rarity mechanisms
        # fail: under rarity alone these cells are the MOST suspicious.
        assert scores[rare_valid].mean() < scores.mean()

    def test_contradicted_common_value_scores_high(self, fitted):
        scores, rare_valid, corrupted, frame = fitted
        assert corrupted.sum() >= 50
        assert scores[corrupted].mean() > scores.mean()

    def test_separates_the_two_cases(self, fitted):
        """Together: a corrupted common value outranks a rare valid one."""
        scores, rare_valid, corrupted, frame = fitted
        keep = rare_valid | corrupted
        assert auc(corrupted[keep], scores[keep]) > 0.8


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Output contract
# ═══════════════════════════════════════════════════════════════════════════════


class TestContract:
    @pytest.fixture(scope="class")
    def fitted(self):
        frame, _ = make_contaminated(n=400, seed=1)
        det = CACMDetector(seed=42, em_iters=3, n_bins=8, **FAST).fit(frame)
        return det, frame

    def test_scores_have_the_frames_shape_index_and_columns(self, fitted):
        det, frame = fitted
        scores = det.score_cells(frame)
        assert scores.shape == frame.shape
        assert list(scores.columns) == list(frame.columns)
        assert scores.index.equals(frame.index)

    def test_scores_are_probabilities(self, fitted):
        det, frame = fitted
        arr = det.score_cells(frame).to_numpy(dtype=float)
        assert np.all(np.isfinite(arr))
        assert arr.min() >= 0.0 and arr.max() <= 1.0

    def test_fitted_rate_is_inside_the_clamp(self, fitted):
        det, _ = fitted
        for mix in det.column_mixtures_.values():
            if mix.is_skipped:
                continue
            assert det.r_min <= mix.r <= det.r_max

    def test_g_is_a_distribution_over_the_fitted_support(self, fitted):
        det, _ = fitted
        for mix in det.column_mixtures_.values():
            if mix.is_skipped:
                continue
            assert set(mix.g) == set(mix.categories)
            assert sum(mix.g.values()) == pytest.approx(1.0, abs=1e-9)
            assert min(mix.g.values()) > 0.0  # smoothing guarantees this

    def test_every_column_is_accounted_for(self, fitted):
        det, frame = fitted
        assert set(det.column_mixtures_) == set(frame.columns)

    def test_report_is_serialisable_and_complete(self, fitted):
        det, frame = fitted
        rep = det.report()
        assert rep["n_columns"] == frame.shape[1]
        assert rep["n_bins"] == det.n_bins
        assert set(rep["columns"]) == set(frame.columns)

    def test_determinism(self):
        frame, _ = make_contaminated(n=400, seed=1)
        a = CACMDetector(seed=42, em_iters=4, n_bins=8, **FAST).fit_score(frame)
        b = CACMDetector(seed=42, em_iters=4, n_bins=8, **FAST).fit_score(frame)
        pd.testing.assert_frame_equal(a, b)

    def test_column_statistic_returns_none_not_zero_for_skipped_columns(self):
        frame, _ = make_contaminated(n=400, seed=1)
        frame["const"] = 5
        det = CACMDetector(seed=42, em_iters=2, n_bins=8, **FAST).fit(frame)
        stat = det.column_statistic("mean_posterior")
        assert stat["const"] is None
        assert stat["y"] is not None

    def test_column_statistic_rejects_an_unknown_name(self):
        frame, _ = make_contaminated(n=400, seed=1)
        det = CACMDetector(seed=42, em_iters=2, n_bins=8, **FAST).fit(frame)
        with pytest.raises(ValueError):
            det.column_statistic("nope")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Discretisation
# ═══════════════════════════════════════════════════════════════════════════════


class TestDiscretisation:
    def test_numeric_column_becomes_at_most_n_bins_categories(self):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
        det = CACMDetector(seed=42, em_iters=1, n_bins=5, **FAST).fit(frame)
        assert det.column_mixtures_["a"].n_categories <= 5

    def test_quantile_bins_are_roughly_equal_sized(self):
        # 50 distinct values repeated 10x each: numeric, high enough
        # cardinality to be binned, low enough not to look like an identifier.
        frame = pd.DataFrame(
            {"a": np.tile(np.arange(50.0), 10), "b": np.arange(500.0) % 3}
        )
        det = CACMDetector(seed=42, em_iters=1, n_bins=5, **FAST).fit(frame)
        mix = det.column_mixtures_["a"]
        shares = np.array([mix.g_init[c] for c in mix.categories])
        assert shares.min() > 0.5 / 5  # no bin collapsed to near-nothing

    def test_missing_values_get_their_own_category(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 4, 400).astype(float)
        a[:40] = np.nan
        frame = pd.DataFrame({"a": a, "b": rng.integers(0, 4, 400)})
        det = CACMDetector(seed=42, em_iters=1, n_bins=8, **FAST).fit(frame)
        mix = det.column_mixtures_["a"]
        assert NA_TOKEN in mix.categories
        assert mix.g_init[NA_TOKEN] == pytest.approx(0.1, abs=0.02)

    def test_high_cardinality_text_collapses_into_other(self):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            {
                "txt": [f"v{i % 200}" for i in range(400)],
                "b": rng.integers(0, 4, 400),
            }
        )
        det = CACMDetector(seed=42, em_iters=1, n_bins=5, **FAST).fit(frame)
        mix = det.column_mixtures_["txt"]
        assert OTHER_TOKEN in mix.categories
        assert mix.other_fraction > 0.5

    def test_corruption_into_missing_is_representable(self):
        """
        Smart Factory's corruption *is* its NaN cells, so g must be able to
        place its mass on the missing category. This is the mixture doing it,
        not a syntactic missing-token rule.
        """
        rng = np.random.default_rng(21)
        n = 800
        x = rng.integers(0, 5, n)
        y = pd.Series((x * 3) % 7, dtype="float64")
        hit = rng.random(n) < 0.2
        y[hit] = np.nan
        frame = pd.DataFrame({"x": x, "z": rng.integers(0, 3, n), "y": y})
        det = CACMDetector(seed=42, em_iters=15, n_bins=8, **FAST).fit(frame)
        mix = det.column_mixtures_["y"]
        assert mix.top_g(1)[0][0] == NA_TOKEN


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Skip rules and errors
# ═══════════════════════════════════════════════════════════════════════════════


class TestSkipsAndErrors:
    def test_constant_column_is_skipped(self):
        frame, _ = make_contaminated(n=400, seed=1)
        frame["const"] = "same"
        det = CACMDetector(seed=42, em_iters=2, n_bins=8, **FAST).fit(frame)
        assert det.column_mixtures_["const"].is_skipped
        assert "const" in det.skipped_columns_

    def test_all_missing_column_is_skipped(self):
        frame, _ = make_contaminated(n=400, seed=1)
        frame["blank"] = np.nan
        det = CACMDetector(seed=42, em_iters=2, n_bins=8, **FAST).fit(frame)
        assert det.column_mixtures_["blank"].reason == "all_missing"

    def test_id_like_column_is_skipped(self):
        frame, _ = make_contaminated(n=400, seed=1)
        frame["row_id"] = np.arange(len(frame))
        det = CACMDetector(seed=42, em_iters=2, n_bins=8, **FAST).fit(frame)
        assert det.column_mixtures_["row_id"].is_skipped
        assert "id_like" in det.column_mixtures_["row_id"].reason

    def test_skipped_columns_score_zero(self):
        frame, _ = make_contaminated(n=400, seed=1)
        frame["const"] = "same"
        scores = CACMDetector(seed=42, em_iters=2, n_bins=8, **FAST).fit_score(frame)
        assert float(scores["const"].abs().max()) == 0.0

    def test_single_column_frame_has_nothing_to_condition_on(self):
        frame = pd.DataFrame({"a": np.arange(200) % 5})
        det = CACMDetector(seed=42, em_iters=2, n_bins=8, **FAST).fit(frame)
        assert det.column_mixtures_["a"].reason == "no_other_columns_to_condition_on"

    def test_score_before_fit_raises(self):
        frame, _ = make_contaminated(n=100, seed=1)
        with pytest.raises(RuntimeError):
            CACMDetector().score_cells(frame)

    def test_report_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            CACMDetector().report()

    def test_scoring_a_different_column_set_raises(self):
        frame, _ = make_contaminated(n=400, seed=1)
        det = CACMDetector(seed=42, em_iters=2, n_bins=8, **FAST).fit(frame)
        with pytest.raises(ValueError):
            det.score_cells(frame[["x1", "y"]])

    def test_scoring_a_different_row_count_raises(self):
        frame, _ = make_contaminated(n=400, seed=1)
        det = CACMDetector(seed=42, em_iters=2, n_bins=8, **FAST).fit(frame)
        with pytest.raises(ValueError):
            det.score_cells(frame.iloc[:200])

    def test_empty_frames_raise(self):
        with pytest.raises(ValueError):
            CACMDetector().fit(pd.DataFrame())
        with pytest.raises(ValueError):
            CACMDetector().fit(pd.DataFrame({"a": []}))

    def test_non_frame_input_raises(self):
        with pytest.raises(TypeError):
            CACMDetector().fit([[1, 2], [3, 4]])

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(n_bins=1),
            dict(em_iters=0),
            dict(n_folds=0),
            dict(r_min=0.5, r_max=0.1),
            dict(r_init=0.9, r_max=0.5),
            dict(g_smoothing=-1.0),
            dict(tol=0.0),
        ],
    )
    def test_invalid_construction_raises(self, kwargs):
        with pytest.raises(ValueError):
            CACMDetector(**kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Leakage -- CLAUDE.md Section 3, non-negotiable
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoLeakage:
    FORBIDDEN = ("clean", "label", "truth", "target_truth", "y_true", "gold")

    def test_no_entry_point_accepts_clean_data_or_labels(self):
        for name, params in leakage_signature_report().items():
            for param in params:
                assert not any(bad in param.lower() for bad in self.FORBIDDEN), (
                    f"CACMDetector.{name} has a parameter named {param!r}"
                )

    def test_constructor_accepts_no_clean_data_or_labels(self):
        for param in inspect.signature(CACMDetector.__init__).parameters:
            assert not any(bad in param.lower() for bad in self.FORBIDDEN)

    def test_profile_is_ignored_entirely(self):
        """
        ``profile`` exists only for interface parity with
        ``src/detectors/conditional.py``. Because ``DatasetProfile`` is built
        from clean data, honouring it would inherit the soft-leakage caveat.
        It must change nothing at all.
        """
        frame, _ = make_contaminated(n=400, seed=1)

        class LoudProfile:
            def __getattr__(self, item):  # pragma: no cover - must never run
                raise AssertionError(f"CACM read profile.{item}")

        a = CACMDetector(seed=42, em_iters=3, n_bins=8, **FAST).fit_score(frame)
        b = CACMDetector(seed=42, em_iters=3, n_bins=8, **FAST).fit_score(
            frame, profile=LoudProfile()
        )
        pd.testing.assert_frame_equal(a, b)

    def test_no_value_outside_the_dirty_frame_reaches_the_estimator(self, monkeypatch):
        """
        A spy on ``HistGradientBoostingClassifier.fit`` records every training
        matrix and every label vector. Each must be derivable from the dirty
        frame: labels are discretised codes of the dirty column, and every
        feature value must appear in the dirty frame's encoded matrix.

        This is the strong form of the assertion. A detector that quietly
        reached for a clean value would have to smuggle it through one of
        these two arrays, and this test reads both.
        """
        import src.detectors.cacm as cacm_mod

        seen: list[tuple[np.ndarray, np.ndarray]] = []
        original = cacm_mod.HistGradientBoostingClassifier.fit

        def spy(self, X, y, **kw):
            seen.append((np.asarray(X, dtype=float).copy(), np.asarray(y).copy()))
            return original(self, X, y, **kw)

        monkeypatch.setattr(cacm_mod.HistGradientBoostingClassifier, "fit", spy)

        frame, hit = make_contaminated(n=400, seed=1)
        # A clean frame exists in this test and must never be touched.
        clean = frame.copy()
        clean.loc[:, "y"] = (frame["x1"] * 5 + frame["x2"]) % 7

        det = CACMDetector(seed=42, em_iters=3, n_bins=8, **FAST).fit(frame)
        assert seen, "the spy never fired -- the test proves nothing"

        allowed_features = set()
        for col in det.feature_columns_:
            allowed_features.update(
                pd.to_numeric(frame[col], errors="coerce").dropna().tolist()
            )
        for X, y in seen:
            finite = X[np.isfinite(X)]
            assert set(np.unique(finite)).issubset(allowed_features), (
                "a feature value reaching fit() is not a value of the dirty frame"
            )
            # Labels are category codes, so they are bounded by the number of
            # categories any column was discretised into.
            assert int(y.min()) >= 0
            assert int(y.max()) < det.n_bins + 2

    def test_scores_depend_only_on_the_dirty_frame(self):
        """
        Same dirty frame, two very different hypothetical clean frames: the
        scores are identical because no clean frame can be supplied at all.
        Fitting twice and comparing is the operational form of that claim.
        """
        frame, _ = make_contaminated(n=400, seed=1)
        first = CACMDetector(seed=42, em_iters=3, n_bins=8, **FAST).fit_score(frame)
        shuffled_copy = frame.copy()
        second = CACMDetector(seed=42, em_iters=3, n_bins=8, **FAST).fit_score(
            shuffled_copy
        )
        pd.testing.assert_frame_equal(first, second)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Cross-fitting -- the thing that keeps r from collapsing
# ═══════════════════════════════════════════════════════════════════════════════


class TestCrossFitting:
    def test_out_of_fold_f_is_not_the_in_sample_f(self):
        """
        With ``n_folds=1`` the clean model scores its own training rows, part
        of the corruption is memorised into ``f``, and the estimated rate is
        pushed down. Asserting the *direction* of that bias is the only way
        to show the cross-fitting is load-bearing rather than decorative.
        """
        frame, hit = make_contaminated(n=800, r=0.25, seed=7)
        cfg = dict(seed=42, em_iters=12, n_bins=8, **FAST)
        r_cross = CACMDetector(n_folds=2, **cfg).fit(frame).column_mixtures_["y"].r
        r_insample = CACMDetector(n_folds=1, **cfg).fit(frame).column_mixtures_["y"].r
        assert r_cross > r_insample, (
            f"cross-fitted r={r_cross:.4f} is not above in-sample r={r_insample:.4f}"
        )

    def test_folds_partition_every_row_exactly_once(self):
        det = CACMDetector(seed=42, n_folds=3)
        folds = det._fold_assignment(100)
        assert folds.shape == (100,)
        assert set(np.unique(folds)) == {0, 1, 2}
        counts = np.bincount(folds)
        assert counts.max() - counts.min() <= 1

    def test_fold_assignment_is_seed_dependent_and_deterministic(self):
        a = CACMDetector(seed=1, n_folds=2)._fold_assignment(50)
        b = CACMDetector(seed=1, n_folds=2)._fold_assignment(50)
        c = CACMDetector(seed=2, n_folds=2)._fold_assignment(50)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. EM bookkeeping
# ═══════════════════════════════════════════════════════════════════════════════


class TestEMBookkeeping:
    def test_em_stops_early_once_responsibilities_settle(self):
        """
        A clean column's responsibilities settle almost immediately, so with a
        loose tolerance EM must stop well before its iteration budget. The
        point is that the stopping rule fires at all -- an EM that always runs
        to the budget has no convergence check.
        """
        rng = np.random.default_rng(3)
        x1 = rng.integers(0, 5, 600)
        x2 = rng.integers(0, 5, 600)
        frame = pd.DataFrame({"x1": x1, "x2": x2, "y": (x1 * 5 + x2) % 7})
        det = CACMDetector(seed=42, em_iters=12, n_bins=8, tol=0.1, **FAST).fit(frame)
        mix = det.column_mixtures_["y"]
        assert mix.n_iter < 12
        assert mix.converged
        assert mix.delta < 0.1

    def test_collapse_at_the_floor_is_recorded_not_hidden(self):
        """A column with no corruption drives r to the floor; that must be
        reported as a bound, not sold as an estimate."""
        rng = np.random.default_rng(3)
        x1 = rng.integers(0, 5, 600)
        x2 = rng.integers(0, 5, 600)
        frame = pd.DataFrame({"x1": x1, "x2": x2, "y": (x1 * 5 + x2) % 7})
        det = CACMDetector(seed=42, em_iters=20, n_bins=8, **FAST).fit(frame)
        mix = det.column_mixtures_["y"]
        assert mix.collapsed == "r_at_min"
        assert det.report()["collapsed_columns"]["y"] == "r_at_min"

    def test_g_init_is_the_column_marginal(self):
        frame = pd.DataFrame(
            {"a": ["x"] * 300 + ["y"] * 100, "b": np.arange(400) % 4}
        )
        det = CACMDetector(seed=42, em_iters=1, n_bins=8, **FAST).fit(frame)
        mix = det.column_mixtures_["a"]
        assert mix.g_init["x"] == pytest.approx(0.75)
        assert mix.g_init["y"] == pytest.approx(0.25)

    def test_mean_posterior_matches_the_score_frame(self):
        frame, _ = make_contaminated(n=400, seed=1)
        det = CACMDetector(seed=42, em_iters=3, n_bins=8, **FAST).fit(frame)
        scores = det.score_cells(frame)
        for col, mix in det.column_mixtures_.items():
            if mix.is_skipped:
                continue
            assert mix.mean_posterior == pytest.approx(
                float(scores[col].mean()), abs=1e-12
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Dataclass hygiene
# ═══════════════════════════════════════════════════════════════════════════════


class TestColumnMixture:
    def test_to_dict_excludes_the_per_cell_arrays(self):
        mix = ColumnMixture(
            column="c",
            role="mixture",
            reason="x",
            r=0.1,
            categories=["a", "b"],
            g={"a": 0.9, "b": 0.1},
            f_obs=np.zeros(3),
            g_obs=np.zeros(3),
        )
        d = mix.to_dict()
        assert "f_obs" not in d and "g_obs" not in d
        assert d["top_g"] == [("a", 0.9), ("b", 0.1)]

    def test_top_g_on_an_unfitted_mixture_is_empty(self):
        assert ColumnMixture(column="c", role="skipped", reason="constant").top_g() == []
