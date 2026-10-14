"""
tests/test_ensemble_detector.py
-------------------------------
Wave-2 route W2.1 tests for :mod:`src.detectors.ensemble`.

WHAT THESE TESTS ARE FOR
========================
The route's claim is specific and falsifiable:

    rules see syntactic corruption, conditional models see semantic
    corruption, and a combination of the two — with at most a few dozen
    labels — beats BOTH.

So the central test builds a dataset containing **one corruption of each
kind** and asserts the ensemble beats both single detectors on PR-AUC. If
that test fails, the complementarity thesis is wrong and the route should be
reported as a loss, not patched until it passes.

THE SYNTHETIC DATASET, AND WHY IT IS BUILT THIS WAY
===================================================
400 rows, four columns, 40 corrupted cells (prevalence 2.5 %)::

    size   int 10..40                      never corrupted
    hours  == size * 3, exactly            never corrupted
    code   60 distinct string categories   20 cells -> "zzz"
    label  "high" iff size > 25            20 cells flipped to the other
                                           valid value

The two corruptions are chosen so that each detector family is **blind to the
other's**, which is what makes the comparison informative:

- ``code`` has 60 distinct non-numeric values, above
  ``ConditionalDetector.max_classes`` (50), so the conditional model *skips
  the column entirely* and scores every cell of it 0.0. It is structurally
  unable to see the novel category. The rule detector sees it immediately
  (vocabulary check). This mirrors the real forfeit documented in
  ``src/detectors/conditional.py``: the conditional model gives up Breast
  Cancer's 107 ID-column corruptions.
- ``label`` is flipped from one *valid* value to the other, so every
  syntactic rule passes it — exactly adult's corruption. It is highly
  improbable given ``size``, so the conditional model sees it.
- ``size`` and ``hours`` determine each other exactly, so the conditional
  model predicts them perfectly and generates no false positives there. That
  is deliberate: a noisy predictor column would add conditional false
  positives and turn the test into a measurement of that noise rather than of
  complementarity. The real datasets in ``experiments/w2_ensemble.py`` supply
  the hard case.

Every metric and every split comes from ``src.detection_eval`` — the same
harness wave 1 used — so a number here means the same thing as a number on
the leaderboard.

LEAKAGE TESTS
=============
Four independent angles, because this is the one invariant the project calls
non-negotiable (CLAUDE.md §3):

1. *Structural*: no public signature has a parameter containing "clean", and
   ``fit`` has nowhere to put a label array.
2. *Behavioural*: every cell the oracle is asked about is inside the train
   mask, and the query count never exceeds the budget.
3. *Counterfactual*: inverting every label **outside** the train mask leaves
   the fitted scores bitwise identical. If an eval label had any influence,
   this test fails.
4. *Negative*: at ``budget=0`` the oracle is never touched at all.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from src.data.loader import DataLoader
from src.data.profiler import profile_dataset
from src.detection.detector import Detector
from src.detection_eval.harness import (
    RuleDetectorAdapter,
    evaluate_detector,
    flatten_cells,
)
from src.detection_eval.labels import cell_labels, label_stats
from src.detection_eval.metrics import evaluate_scores
from src.detectors.ensemble import (
    COMBINERS,
    FEATURE_NAMES,
    CellFeatures,
    EnsembleDetector,
    FeatureCache,
    LabelOracle,
    build_features,
    normalised_ranks,
)

SEED = 42
N_ROWS = 400
N_SYNTACTIC = 20
N_SEMANTIC = 20

#: Boosting iterations for the conditional sub-model inside these tests.
#:
#: The synthetic structure is exactly learnable (``hours == size * 3``,
#: ``label == (size > 25)``), so ten boosting iterations fit it as well as a
#: hundred — and with five out-of-fold folds per feature build, the default
#: 100 turns this file into a 13-minute run without changing one assertion.
#: ``experiments/w2_ensemble.py`` uses the default on the real datasets, which
#: is what wave 1 used.
FAST_CONDITIONAL = {"max_iter": 10}


def _ens(**kwargs) -> EnsembleDetector:
    """``EnsembleDetector`` with the cheap conditional config as its default."""
    kwargs.setdefault("conditional_kwargs", FAST_CONDITIONAL)
    return EnsembleDetector(**kwargs)


def _feats(*args, **kwargs):
    """:func:`build_features` with the cheap conditional config as its default."""
    kwargs.setdefault("conditional_kwargs", FAST_CONDITIONAL)
    return build_features(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


def _make_pair(n: int = N_ROWS, seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The clean/dirty pair described in the module docstring."""
    rng = np.random.default_rng(seed)
    size = rng.integers(10, 41, n)
    hours = size * 3
    code = [f"c{int(rng.integers(0, 60)):02d}" for _ in range(n)]
    label = np.where(size > 25, "high", "low")
    clean = pd.DataFrame({"size": size, "hours": hours, "code": code, "label": label})

    dirty = clean.copy()
    syntactic_rows = rng.choice(n, N_SYNTACTIC, replace=False)
    dirty.loc[syntactic_rows, "code"] = "zzz"
    semantic_rows = rng.choice(
        np.setdiff1d(np.arange(n), syntactic_rows), N_SEMANTIC, replace=False
    )
    dirty.loc[semantic_rows, "label"] = np.where(
        clean.loc[semantic_rows, "label"] == "high", "low", "high"
    )
    return clean, dirty


@pytest.fixture(scope="module")
def synth_root(tmp_path_factory) -> Path:
    """The pair on disk, so the real ``DataLoader`` and harness can read it."""
    root = tmp_path_factory.mktemp("w2_synth")
    clean, dirty = _make_pair()
    folder = root / "synth"
    folder.mkdir(parents=True, exist_ok=True)
    clean.to_csv(folder / "clean.csv", index=False)
    dirty.to_csv(folder / "dirty.csv", index=False)
    return root


@pytest.fixture(scope="module")
def synth(synth_root: Path):
    """``(clean_df, dirty_df, profile, labels, y_flat)`` for the synthetic set."""
    loader = DataLoader(str(synth_root))
    clean_df, dirty_df = loader.load("synth")
    profile = profile_dataset(
        "synth", loader.discover("synth"), clean_df, dirty_df
    )
    labels = cell_labels(clean_df, dirty_df)
    return clean_df, dirty_df, profile, labels, flatten_cells(labels).astype(bool)


@pytest.fixture(scope="module")
def cache() -> FeatureCache:
    """One shared cache: the conditional model is then fitted once per frame."""
    return FeatureCache()


@pytest.fixture(scope="module")
def features(synth, cache) -> CellFeatures:
    _, dirty_df, profile, _, _ = synth
    det = _ens(budget=0, profile=profile, seed=SEED, feature_cache=cache)
    det.fit(dirty_df)
    return det.features_


def _train_mask(dirty_df: pd.DataFrame, fraction: float = 0.7, seed: int = 7):
    """A random cell-level train mask plus its flat boolean vector."""
    rng = np.random.default_rng(seed)
    n_rows = len(dirty_df)
    flat = rng.random(n_rows * dirty_df.shape[1]) < fraction
    frame = pd.DataFrame(
        {
            col: flat[i * n_rows : (i + 1) * n_rows]
            for i, col in enumerate(dirty_df.columns)
        },
        index=dirty_df.index,
    )
    return frame, flat


def _pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    value = evaluate_scores(y, scores).pr_auc
    assert value is not None
    return float(value)


def _score_flat(det: EnsembleDetector, dirty_df: pd.DataFrame) -> np.ndarray:
    return flatten_cells(det.score_cells(dirty_df))


# ═══════════════════════════════════════════════════════════════════════════════
# The synthetic dataset is what the docstring claims
# ═══════════════════════════════════════════════════════════════════════════════


class TestSyntheticFixture:
    """If the fixture is not what it claims, every later assertion is noise."""

    def test_corruption_counts_and_locations(self, synth) -> None:
        _, _, _, labels, _ = synth
        stats = label_stats(labels)
        assert stats["n_corrupted"] == N_SYNTACTIC + N_SEMANTIC
        assert stats["per_column"] == {
            "size": 0,
            "hours": 0,
            "code": N_SYNTACTIC,
            "label": N_SEMANTIC,
        }
        assert stats["prevalence"] == pytest.approx(40 / (N_ROWS * 4))

    def test_semantic_corruption_is_a_valid_value(self, synth) -> None:
        """A label flip must be invisible to any vocabulary or range rule."""
        clean_df, dirty_df, _, labels, _ = synth
        flipped = labels["label"].to_numpy()
        assert set(dirty_df.loc[flipped, "label"]) <= {"high", "low"}
        assert set(dirty_df["label"].unique()) == set(clean_df["label"].unique())

    def test_rule_detector_is_blind_to_the_semantic_corruption(self, synth) -> None:
        clean_df, dirty_df, profile, labels, _ = synth
        records = Detector(profile).detect(dirty_df)
        flagged_label_cells = [r for r in records if r.column == "label"]
        assert flagged_label_cells == []

    def test_conditional_model_is_blind_to_the_syntactic_corruption(
        self, features
    ) -> None:
        """``code`` is skipped by the conditional model, so c_full is 0 there."""
        cond = features.as_frame(features.column("c_full"))
        assert float(cond["code"].abs().max()) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# normalised_ranks — re-derived independently
# ═══════════════════════════════════════════════════════════════════════════════


class TestNormalisedRanks:
    def test_matches_an_independent_average_rank_derivation(self) -> None:
        rng = np.random.default_rng(3)
        values = rng.integers(0, 5, 50).astype(float)  # many ties on purpose
        got = normalised_ranks(values)

        # Independent derivation: a value's average rank is the number of
        # strictly smaller values plus half of (its tie-group size - 1).
        expected = np.empty_like(values)
        for i, v in enumerate(values):
            smaller = int((values < v).sum())
            tied = int((values == v).sum())
            expected[i] = (smaller + (tied - 1) / 2.0) / (values.size - 1)
        assert np.allclose(got, expected)

    def test_all_tied_maps_to_one_constant(self) -> None:
        got = normalised_ranks(np.full(10, 0.3))
        assert np.allclose(got, got[0])

    def test_monotone_and_bounded(self) -> None:
        got = normalised_ranks(np.array([0.0, 0.5, 9.0]))
        assert got.tolist() == [0.0, 0.5, 1.0]

    @pytest.mark.parametrize("size", [0, 1])
    def test_degenerate_sizes(self, size: int) -> None:
        got = normalised_ranks(np.zeros(size))
        assert got.shape == (size,)


# ═══════════════════════════════════════════════════════════════════════════════
# LabelOracle — the single metered door for ground truth
# ═══════════════════════════════════════════════════════════════════════════════


class TestLabelOracle:
    def test_flattening_is_column_major_like_the_harness(self) -> None:
        labels = pd.DataFrame({"a": [True, False], "b": [False, True]})
        oracle = LabelOracle(labels)
        assert oracle.query([0, 1, 2, 3]).tolist() == [True, False, False, True]

    def test_counts_and_records_every_query(self) -> None:
        oracle = LabelOracle(pd.DataFrame({"a": [True, False, True]}))
        oracle.query([2, 0])
        oracle.query([1])
        assert oracle.n_queries == 3
        assert oracle.queried_indices == [2, 0, 1]

    def test_cap_is_a_hard_constraint(self) -> None:
        oracle = LabelOracle(pd.DataFrame({"a": [True, False, True]}), max_queries=2)
        oracle.query([0, 1])
        with pytest.raises(RuntimeError, match="budget exhausted"):
            oracle.query([2])

    def test_out_of_range_index_raises(self) -> None:
        oracle = LabelOracle(pd.DataFrame({"a": [True]}))
        with pytest.raises(IndexError):
            oracle.query([5])

    def test_rejects_non_frame_and_negative_cap(self) -> None:
        with pytest.raises(TypeError):
            LabelOracle([1, 0])  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            LabelOracle(pd.DataFrame({"a": [True]}), max_queries=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# The feature matrix
# ═══════════════════════════════════════════════════════════════════════════════


class TestFeatureMatrix:
    def test_shape_and_names_are_the_documented_contract(self, synth, features) -> None:
        _, dirty_df, _, _, _ = synth
        assert len(FEATURE_NAMES) == 19
        assert features.names == FEATURE_NAMES
        assert features.x.shape == (dirty_df.size, 19)
        assert features.x.dtype == np.float64
        assert np.all(np.isfinite(features.x))

    def test_feature_names_are_unique(self) -> None:
        assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)

    def test_row_order_is_column_major_and_aligned_with_the_harness(
        self, synth, features
    ) -> None:
        """
        Independent check of cell alignment: build a NaN mask by hand, flatten
        it with the harness's own function, and require the ``r_missing``
        feature to agree cell for cell. Misalignment here would mis-train the
        learner invisibly.
        """
        _, dirty_df, profile, _, _ = synth
        injected = dirty_df.copy()
        injected.loc[injected.index[3], "size"] = np.nan
        injected.loc[injected.index[7], "code"] = np.nan
        built = _feats(injected, profile=profile, seed=SEED)
        expected = flatten_cells(injected.isna()).astype(float)
        assert np.array_equal(built.column("r_missing"), expected)

    def test_rule_indicators_fire_on_the_right_error_types(self, synth) -> None:
        _, dirty_df, profile, _, _ = synth
        built = _feats(dirty_df, profile=profile, seed=SEED)
        novel = built.as_frame(built.column("r_novel_category"))
        assert float(novel["code"].sum()) == float(N_SYNTACTIC)
        assert float(novel["label"].sum()) == 0.0
        # No missing values were injected, so that indicator must stay silent.
        assert float(built.column("r_missing").sum()) == 0.0

    def test_r_score_equals_the_rule_detectors_own_scores(self, synth) -> None:
        """
        Cross-check against wave 1: the ``r_score`` feature must be exactly
        what :class:`RuleDetectorAdapter` produced for the leaderboard, or the
        ensemble is not being compared against the same baseline.
        """
        _, dirty_df, profile, _, _ = synth
        built = _feats(dirty_df, profile=profile, seed=SEED)
        baseline = flatten_cells(RuleDetectorAdapter().score_cells(dirty_df, profile))
        assert np.array_equal(built.column("r_score"), baseline)

    def test_row_features_are_leave_one_out(self, synth, features) -> None:
        """Re-derive both row aggregates from the per-cell features."""
        _, dirty_df, _, _, _ = synth
        n_cols = dirty_df.shape[1]
        flagged = features.as_frame(
            (features.column("r_score") > 0).astype(float)
        ).to_numpy()
        cond = features.as_frame(features.column("c_full")).to_numpy()

        expected_flags = flagged.sum(axis=1, keepdims=True) - flagged
        expected_cond = (cond.sum(axis=1, keepdims=True) - cond) / (n_cols - 1)

        got_flags = features.as_frame(features.column("row_n_other_flagged")).to_numpy()
        got_cond = features.as_frame(features.column("row_mean_cond_other")).to_numpy()
        assert np.allclose(got_flags, expected_flags)
        assert np.allclose(got_cond, expected_cond)

    def test_column_features_are_constant_within_a_column_and_correct(
        self, synth, features
    ) -> None:
        _, dirty_df, _, _, _ = synth
        card = features.as_frame(features.column("col_cardinality_ratio"))
        miss = features.as_frame(features.column("col_missing_rate"))
        for col in dirty_df.columns:
            assert card[col].nunique() == 1
            assert miss[col].nunique() == 1
            expected = dirty_df[col].dropna().nunique() / len(dirty_df)
            assert float(card[col].iloc[0]) == pytest.approx(expected)
            assert float(miss[col].iloc[0]) == pytest.approx(
                float(dirty_df[col].isna().mean())
            )

    def test_column_kind_is_one_hot_and_matches_the_profile(
        self, synth, features
    ) -> None:
        """
        Regression test for a real bug: the kind features were originally
        looked up with ``str(ColumnKind.X)``, which yields
        ``"ColumnKind.NUMERICAL"`` rather than ``"numerical"``, so every kind
        feature silently came out 0.0.
        """
        _, dirty_df, profile, _, _ = synth
        kind_names = [n for n in FEATURE_NAMES if n.startswith("col_kind_")]
        frames = {n: features.as_frame(features.column(n)) for n in kind_names}
        for col in dirty_df.columns:
            hot = [n for n in kind_names if float(frames[n][col].iloc[0]) == 1.0]
            assert len(hot) == 1, f"{col} has kinds {hot}"
            expected = f"col_kind_{profile.columns[col].kind.value}"
            assert hot[0] == expected

    def test_without_a_profile_rule_and_kind_features_are_zero_and_noted(
        self, synth
    ) -> None:
        _, dirty_df, _, _, _ = synth
        built = _feats(dirty_df, profile=None, seed=SEED)
        for name in FEATURE_NAMES:
            if name.startswith("r_") or name.startswith("col_kind_"):
                assert float(np.abs(built.column(name)).max()) == 0.0, name
        assert any("no DatasetProfile" in note for note in built.notes)
        # The conditional block still works, so the detector degrades
        # gracefully rather than failing.
        assert float(built.column("c_full").max()) > 0.0

    def test_rejects_bad_input(self) -> None:
        with pytest.raises(TypeError):
            _feats([1, 2, 3])  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            _feats(pd.DataFrame(index=[0, 1]))

    def test_as_frame_round_trips(self, synth, features) -> None:
        _, dirty_df, _, _, _ = synth
        frame = features.as_frame(features.column("c_full"))
        assert frame.shape == dirty_df.shape
        assert list(frame.columns) == list(dirty_df.columns)
        assert frame.index.equals(dirty_df.index)


class TestOutOfFoldConditionalScoring:
    """
    The conditional features are scored out of fold by default, because a
    model fitted on the cells it scores partly fits the corruption and loses
    detections. These tests pin the mechanism, not the magnitude.
    """

    def test_out_of_fold_is_the_default_and_is_recorded(self, synth, features) -> None:
        assert any("out of fold" in note for note in features.notes)

    def test_out_of_fold_scores_differ_from_an_in_sample_fit(self, synth) -> None:
        _, dirty_df, profile, _, _ = synth
        oof = _feats(dirty_df, profile=profile, seed=SEED, conditional_folds=5)
        in_sample = _feats(
            dirty_df, profile=profile, seed=SEED, conditional_folds=1
        )
        assert not np.array_equal(
            oof.column("c_full"), in_sample.column("c_full")
        )
        # The rule block does not depend on folds, so it must be identical —
        # otherwise the two runs differ for some other reason.
        assert np.array_equal(oof.column("r_score"), in_sample.column("r_score"))

    def test_fold_assignment_is_deterministic(self, synth) -> None:
        _, dirty_df, profile, _, _ = synth
        first = _feats(dirty_df, profile=profile, seed=SEED, conditional_folds=5)
        second = _feats(
            dirty_df, profile=profile, seed=SEED, conditional_folds=5
        )
        assert np.array_equal(first.column("c_full"), second.column("c_full"))

    def test_zero_or_negative_folds_rejected(self) -> None:
        with pytest.raises(ValueError, match="conditional_folds"):
            _ens(conditional_folds=0)

    def test_folds_are_part_of_the_cache_key(self, synth) -> None:
        _, dirty_df, profile, _, _ = synth
        shared = FeatureCache()
        _ens(
            profile=profile, seed=SEED, conditional_folds=1, feature_cache=shared
        ).fit(dirty_df)
        _ens(
            profile=profile, seed=SEED, conditional_folds=5, feature_cache=shared
        ).fit(dirty_df)
        assert shared.misses == 2


class TestFeatureCache:
    def test_identical_frames_hit_the_cache_once(self, synth) -> None:
        _, dirty_df, profile, _, _ = synth
        shared = FeatureCache()
        for budget in (0, 0, 0):
            _ens(
                budget=budget, profile=profile, seed=SEED, feature_cache=shared
            ).fit(dirty_df)
        assert shared.misses == 1
        assert shared.hits == 2

    def test_different_content_does_not_collide(self, synth) -> None:
        _, dirty_df, profile, _, _ = synth
        shared = FeatureCache()
        other = dirty_df.copy()
        other.loc[other.index[0], "size"] = 999
        _ens(profile=profile, seed=SEED, feature_cache=shared).fit(dirty_df)
        _ens(profile=profile, seed=SEED, feature_cache=shared).fit(other)
        assert shared.misses == 2

    def test_a_different_seed_does_not_reuse_features(self, synth) -> None:
        _, dirty_df, profile, _, _ = synth
        shared = FeatureCache()
        _ens(profile=profile, seed=1, feature_cache=shared).fit(dirty_df)
        _ens(profile=profile, seed=2, feature_cache=shared).fit(dirty_df)
        assert shared.misses == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Unsupervised combiners (budget = 0)
# ═══════════════════════════════════════════════════════════════════════════════


class TestUnsupervisedCombiners:
    def test_rule_only_reproduces_the_wave_one_baseline_exactly(
        self, synth, cache
    ) -> None:
        """
        The strongest available cross-check: routing the rule signal through
        this module's plumbing must give bit-identical scores to the wave-1
        adapter. If it does not, "the ensemble beats the rule detector" is
        comparing two different rule detectors.
        """
        _, dirty_df, profile, _, _ = synth
        det = _ens(
            budget=0,
            combiner="rule_only",
            profile=profile,
            seed=SEED,
            feature_cache=cache,
        )
        got = _score_flat(det, dirty_df)
        expected = flatten_cells(RuleDetectorAdapter().score_cells(dirty_df, profile))
        assert np.array_equal(got, expected)

    def test_cond_only_is_exactly_the_conditional_full_score(
        self, synth, cache, features
    ) -> None:
        _, dirty_df, profile, _, _ = synth
        det = _ens(
            budget=0,
            combiner="cond_only",
            profile=profile,
            seed=SEED,
            feature_cache=cache,
        )
        assert np.array_equal(
            _score_flat(det, dirty_df), features.column("c_full")
        )

    def test_noisy_or_matches_its_formula(self, synth, cache, features) -> None:
        _, dirty_df, profile, _, _ = synth
        det = _ens(
            budget=0,
            combiner="noisy_or",
            profile=profile,
            seed=SEED,
            feature_cache=cache,
        )
        r = np.clip(features.column("r_score"), 0, 1)
        c = np.clip(features.column("c_full"), 0, 1)
        assert np.allclose(_score_flat(det, dirty_df), 1.0 - (1.0 - r) * (1.0 - c))

    def test_rank_mean_matches_its_formula(self, synth, cache, features) -> None:
        _, dirty_df, profile, _, _ = synth
        det = _ens(
            budget=0,
            combiner="rank_mean",
            profile=profile,
            seed=SEED,
            feature_cache=cache,
        )
        expected = 0.5 * (
            normalised_ranks(features.column("r_score"))
            + normalised_ranks(features.column("c_full"))
        )
        assert np.allclose(_score_flat(det, dirty_df), expected)

    @pytest.mark.parametrize("combiner", ["rank_mean", "noisy_or"])
    def test_budget_zero_produces_a_valid_ranking(
        self, synth, cache, combiner: str
    ) -> None:
        """A ranking that beats prevalence is the minimum bar: a random scorer
        scores exactly prevalence."""
        _, dirty_df, profile, labels, y = synth
        prevalence = label_stats(labels)["prevalence"]
        det = _ens(
            budget=0,
            combiner=combiner,
            profile=profile,
            seed=SEED,
            feature_cache=cache,
        )
        score = _pr_auc(y, _score_flat(det, dirty_df))
        assert score > prevalence, f"{combiner}: PR-AUC {score} <= prevalence {prevalence}"

    def test_budget_zero_never_touches_the_oracle(self, synth, cache) -> None:
        _, dirty_df, profile, labels, _ = synth
        oracle = LabelOracle(labels, max_queries=0)
        det = _ens(
            budget=0,
            label_oracle=oracle,
            profile=profile,
            seed=SEED,
            feature_cache=cache,
        )
        det.fit_score(dirty_df)
        assert oracle.n_queries == 0
        assert oracle.queried_indices == []
        assert det.report()["used_supervision"] is False

    def test_score_frame_satisfies_the_harness_contract(self, synth, cache) -> None:
        _, dirty_df, profile, _, _ = synth
        det = _ens(
            budget=0, profile=profile, seed=SEED, feature_cache=cache
        )
        frame = det.fit_score(dirty_df)
        assert frame.shape == dirty_df.shape
        assert list(frame.columns) == list(dirty_df.columns)
        assert frame.index.equals(dirty_df.index)
        assert np.all(np.isfinite(frame.to_numpy(dtype=float)))


# ═══════════════════════════════════════════════════════════════════════════════
# THE CORE CLAIM
# ═══════════════════════════════════════════════════════════════════════════════


class TestComplementarityClaim:
    """
    The route's reason to exist. These assertions are able to fail: the
    corruption types are disjoint by construction and each single detector is
    structurally blind to one of them, so a combination that does *not* beat
    both would mean the combination rule destroys information.
    """

    @staticmethod
    def _single_detector_scores(dirty_df, profile, cache, y) -> dict:
        out = {}
        for combiner in ("rule_only", "cond_only"):
            det = _ens(
                budget=0,
                combiner=combiner,
                profile=profile,
                seed=SEED,
                feature_cache=cache,
            )
            out[combiner] = _pr_auc(y, _score_flat(det, dirty_df))
        return out

    def test_each_single_detector_only_finds_its_own_half(
        self, synth, cache
    ) -> None:
        """Establishes the premise: neither single detector can exceed ~0.5,
        because each can only reach 20 of the 40 positives."""
        _, dirty_df, profile, _, y = synth
        singles = self._single_detector_scores(dirty_df, profile, cache, y)
        for name, value in singles.items():
            assert 0.3 < value < 0.75, f"{name} scored {value}, premise broken"

    def test_unsupervised_ensemble_beats_both_single_detectors(
        self, synth, cache
    ) -> None:
        _, dirty_df, profile, _, y = synth
        singles = self._single_detector_scores(dirty_df, profile, cache, y)
        det = _ens(
            budget=0,
            combiner="noisy_or",
            profile=profile,
            seed=SEED,
            feature_cache=cache,
        )
        ensemble = _pr_auc(y, _score_flat(det, dirty_df))
        assert ensemble > max(singles.values()) + 0.10, (
            f"noisy_or ensemble {ensemble:.4f} did not beat both singles "
            f"{singles} by 0.10 — the complementarity thesis fails here"
        )

    def test_supervised_ensemble_beats_both_single_detectors(
        self, synth, cache
    ) -> None:
        _, dirty_df, profile, labels, y = synth
        singles = self._single_detector_scores(dirty_df, profile, cache, y)
        mask, flat = _train_mask(dirty_df)
        eval_idx = np.flatnonzero(~flat)

        det = _ens(
            budget=50,
            label_oracle=LabelOracle(labels, max_queries=50),
            profile=profile,
            seed=SEED,
            feature_cache=cache,
        )
        det.fit(dirty_df, mask)
        assert det.used_supervision_, det.fallback_reason_
        scores = _score_flat(det, dirty_df)
        ensemble = _pr_auc(y[eval_idx], scores[eval_idx])
        assert ensemble > max(singles.values()) + 0.10, (
            f"supervised ensemble (budget=50) {ensemble:.4f} did not beat both "
            f"singles {singles} by 0.10"
        )

    def test_both_corruption_types_are_ranked_above_clean_cells(
        self, synth, cache
    ) -> None:
        """
        The mechanism behind the claim, stated directly: a combination must
        lift *both* families of positives, not trade one for the other.
        """
        _, dirty_df, profile, labels, _ = synth
        det = _ens(
            budget=0,
            combiner="noisy_or",
            profile=profile,
            seed=SEED,
            feature_cache=cache,
        )
        scores = det.fit_score(dirty_df)
        clean_cells = scores.to_numpy()[~labels.to_numpy()]
        syntactic = scores["code"].to_numpy()[labels["code"].to_numpy()]
        semantic = scores["label"].to_numpy()[labels["label"].to_numpy()]
        assert syntactic.min() > np.median(clean_cells)
        assert semantic.min() > np.median(clean_cells)


# ═══════════════════════════════════════════════════════════════════════════════
# Label efficiency
# ═══════════════════════════════════════════════════════════════════════════════


class TestLabelEfficiency:
    @pytest.fixture(scope="class")
    def curve(self, synth, cache) -> dict:
        _, dirty_df, profile, labels, y = synth
        mask, flat = _train_mask(dirty_df)
        eval_idx = np.flatnonzero(~flat)
        out = {}
        for budget in (5, 10, 20, 50, 100, "full"):
            det = _ens(
                budget=budget,
                label_oracle=LabelOracle(labels),
                profile=profile,
                seed=SEED,
                feature_cache=cache,
            )
            det.fit(dirty_df, mask)
            scores = _score_flat(det, dirty_df)
            out[budget] = (_pr_auc(y[eval_idx], scores[eval_idx]), det.report())
        return out

    def test_more_labels_do_not_make_it_worse(self, curve) -> None:
        """
        Monotone-ish, with a tolerance for genuine sampling noise. A broken
        learner (features shuffled, labels mis-aligned) shows up as a *drop*
        well beyond the tolerance.
        """
        assert curve[50][0] >= curve[5][0] - 0.05, {k: v[0] for k, v in curve.items()}
        assert curve["full"][0] >= curve[5][0] - 0.05

    def test_every_budget_spends_exactly_what_it_was_given(self, curve) -> None:
        for budget in (5, 10, 20, 50, 100):
            report = curve[budget][1]
            assert report["n_labelled"] == budget
            assert report["oracle_queries"] == budget

    def test_the_informed_sampler_finds_positives_at_tiny_budgets(
        self, curve
    ) -> None:
        """
        Stratification is the point: uniform sampling of 10 cells at 2.5 %
        prevalence has only a ~22 % chance of containing a positive, so a
        label-efficiency curve built on uniform sampling would measure the
        dice. The informed half of the budget must actually land positives.
        """
        for budget in (5, 10, 20):
            assert curve[budget][1]["n_positive_labels"] >= 1, curve[budget][1]

    def test_supervision_was_really_used(self, curve) -> None:
        for budget, (_, report) in curve.items():
            assert report["used_supervision"] is True, (budget, report)
            assert report["fallback_reason"] is None

    def test_full_budget_labels_every_train_cell(self, synth, cache) -> None:
        _, dirty_df, profile, labels, _ = synth
        mask, flat = _train_mask(dirty_df)
        det = _ens(
            budget="full",
            label_oracle=LabelOracle(labels),
            profile=profile,
            seed=SEED,
            feature_cache=cache,
        )
        det.fit(dirty_df, mask)
        assert det.n_labelled_ == int(flat.sum())


class TestBudgetSampling:
    def test_half_the_budget_goes_to_the_top_ranked_train_cells(
        self, synth, cache
    ) -> None:
        """Re-derive the informed half independently from the unsupervised
        ranking and require the ``top_combiner`` sampler to have chosen those
        cells."""
        _, dirty_df, profile, labels, _ = synth
        mask, flat = _train_mask(dirty_df)
        budget = 20
        det = _ens(
            budget=budget,
            label_oracle=LabelOracle(labels),
            combiner="noisy_or",
            sampler="top_combiner",
            profile=profile,
            seed=SEED,
            feature_cache=cache,
        )
        det.fit(dirty_df, mask)

        train_idx = np.flatnonzero(flat)
        scores = det._unsupervised_scores(det.features_)[train_idx]
        n_top = int(math.ceil(budget * det.top_fraction))
        # The score at the n_top-th position is the cut; anything the sampler
        # picked as "top" must be at least that high.
        cut = np.sort(scores)[::-1][n_top - 1]
        chosen = set(det.labelled_indices_.tolist())
        at_or_above = {
            int(i) for i in train_idx[scores >= cut]
        }
        assert len(chosen & at_or_above) >= n_top

    def test_per_column_sampler_spreads_the_informed_labels_over_columns(
        self, synth, cache
    ) -> None:
        """
        The default sampler's defining property, and the reason it exists: a
        *global* top-k can put every label in one column, which on adult means
        every label lands on a guaranteed-clean column. Round-robin allocation
        must touch every column once the informed share covers them.
        """
        _, dirty_df, profile, labels, _ = synth
        mask, _ = _train_mask(dirty_df)
        n_cols = dirty_df.shape[1]
        n_rows = len(dirty_df)
        budget = 4 * n_cols  # informed share = 2 * n_cols, so every column twice
        det = _ens(
            budget=budget,
            label_oracle=LabelOracle(labels),
            combiner="noisy_or",
            sampler="per_column",
            profile=profile,
            seed=SEED,
            feature_cache=cache,
        )
        det.fit(dirty_df, mask)
        informed = det.labelled_indices_
        columns_touched = {int(i) // n_rows for i in informed}
        assert columns_touched == set(range(n_cols)), columns_touched

    def test_per_column_and_global_samplers_choose_different_cells(
        self, synth, cache
    ) -> None:
        _, dirty_df, profile, labels, _ = synth
        mask, _ = _train_mask(dirty_df)
        picks = {}
        for sampler in ("per_column", "diverse", "top_combiner"):
            det = _ens(
                budget=20,
                label_oracle=LabelOracle(labels),
                combiner="noisy_or",
                sampler=sampler,
                profile=profile,
                seed=SEED,
                feature_cache=cache,
            )
            det.fit(dirty_df, mask)
            picks[sampler] = set(det.labelled_indices_.tolist())
        assert picks["per_column"] != picks["top_combiner"]

    @pytest.mark.parametrize("sampler", ["per_column", "diverse", "top_combiner"])
    def test_every_sampler_respects_the_budget_and_the_train_mask(
        self, synth, cache, sampler: str
    ) -> None:
        _, dirty_df, profile, labels, _ = synth
        mask, flat = _train_mask(dirty_df)
        oracle = LabelOracle(labels, max_queries=20)
        det = _ens(
            budget=20,
            label_oracle=oracle,
            sampler=sampler,
            profile=profile,
            seed=SEED,
            feature_cache=cache,
        )
        det.fit(dirty_df, mask)
        assert oracle.n_queries == 20
        assert all(bool(flat[i]) for i in oracle.queried_indices)

    def test_diverse_and_per_column_skip_trivially_missing_cells(
        self, synth_root, cache
    ) -> None:
        """
        A blank cell is already scored 1.0 by the rule block *and* by the
        conditional block, so a label spent on one teaches the learner nothing
        it cannot read off ``r_missing``. On adult that distinction decides
        whether any positive is found at all.
        """
        loader = DataLoader(str(synth_root))
        clean_df, dirty_df = loader.load("synth")
        holed = dirty_df.copy()
        holed.loc[holed.index[:150], "size"] = np.nan
        profile = profile_dataset(
            "synth", loader.discover("synth"), clean_df, holed
        )
        labels = cell_labels(clean_df, holed)
        n_rows = len(holed)
        blank_flat = {
            i * n_rows + r
            for i, col in enumerate(holed.columns)
            for r in range(n_rows)
            if pd.isna(holed[col].iloc[r])
        }
        assert blank_flat, "the fixture injected no blanks"

        for sampler in ("per_column", "diverse"):
            det = _ens(
                budget=20,
                label_oracle=LabelOracle(labels),
                sampler=sampler,
                profile=profile,
                seed=SEED,
                feature_cache=FeatureCache(),
            )
            det.fit(holed)
            informed = set(det.labelled_indices_.tolist())
            n_top = int(math.ceil(20 * det.top_fraction))
            # The random half may still hit a blank by chance; the informed
            # half must not, and the informed half is the first n_top picks.
            assert len(informed & blank_flat) <= 20 - n_top, sampler

    def test_sampler_is_deterministic_for_a_seed(self, synth, cache) -> None:
        _, dirty_df, profile, labels, _ = synth
        mask, _ = _train_mask(dirty_df)
        picks = []
        for _ in range(2):
            det = _ens(
                budget=20,
                label_oracle=LabelOracle(labels),
                profile=profile,
                seed=SEED,
                feature_cache=cache,
            )
            det.fit(dirty_df, mask)
            picks.append(det.labelled_indices_.tolist())
        assert picks[0] == picks[1]

    def test_budget_larger_than_the_train_split_is_clamped(
        self, synth, cache
    ) -> None:
        _, dirty_df, profile, labels, _ = synth
        mask, flat = _train_mask(dirty_df, fraction=0.01, seed=11)
        det = _ens(
            budget=10_000,
            label_oracle=LabelOracle(labels),
            profile=profile,
            seed=SEED,
            feature_cache=cache,
        )
        det.fit(dirty_df, mask)
        assert det.n_labelled_ <= int(flat.sum())


# ═══════════════════════════════════════════════════════════════════════════════
# Failure is reported, not hidden
# ═══════════════════════════════════════════════════════════════════════════════


class TestFallback:
    def _clean_only_mask(self, dirty_df: pd.DataFrame, labels: pd.DataFrame):
        """A train mask over cells that are all genuinely clean, so any label
        sample drawn from it is single-class by construction."""
        mask = (~labels).copy()
        mask[["code", "label"]] = False  # drop the corrupted columns entirely
        return mask

    def test_zero_positive_sample_falls_back_and_says_so(
        self, synth, cache
    ) -> None:
        _, dirty_df, profile, labels, _ = synth
        mask = self._clean_only_mask(dirty_df, labels)
        det = _ens(
            budget=20,
            label_oracle=LabelOracle(labels, max_queries=20),
            combiner="noisy_or",
            profile=profile,
            seed=SEED,
            feature_cache=cache,
        )
        det.fit(dirty_df, mask)

        assert det.n_labelled_ == 20
        assert det.n_positive_labels_ == 0
        assert det.used_supervision_ is False
        assert det.fallback_reason_ is not None
        assert "only negatives" in det.fallback_reason_
        assert det.report()["fallback_reason"] == det.fallback_reason_

    def test_the_fallback_is_exactly_the_unsupervised_combiner(
        self, synth, cache
    ) -> None:
        """Not "something reasonable" — bitwise the documented fallback."""
        _, dirty_df, profile, labels, _ = synth
        mask = self._clean_only_mask(dirty_df, labels)
        fell_back = _ens(
            budget=20,
            label_oracle=LabelOracle(labels),
            combiner="noisy_or",
            profile=profile,
            seed=SEED,
            feature_cache=cache,
        )
        fell_back.fit(dirty_df, mask)
        unsupervised = _ens(
            budget=0,
            combiner="noisy_or",
            profile=profile,
            seed=SEED,
            feature_cache=cache,
        )
        assert np.array_equal(
            _score_flat(fell_back, dirty_df), _score_flat(unsupervised, dirty_df)
        )

    def test_all_positive_sample_also_falls_back(self, synth, cache) -> None:
        _, dirty_df, profile, labels, _ = synth
        det = _ens(
            budget=5,
            label_oracle=LabelOracle(labels),
            profile=profile,
            seed=SEED,
            feature_cache=cache,
        )
        det.fit(dirty_df, labels)  # train mask = exactly the corrupted cells
        assert det.used_supervision_ is False
        assert "only positives" in (det.fallback_reason_ or "")

    def test_empty_train_mask_is_reported_not_crashed(self, synth, cache) -> None:
        _, dirty_df, profile, labels, _ = synth
        empty = pd.DataFrame(
            False, index=dirty_df.index, columns=dirty_df.columns
        )
        det = _ens(
            budget=20,
            label_oracle=LabelOracle(labels),
            profile=profile,
            seed=SEED,
            feature_cache=cache,
        )
        det.fit(dirty_df, empty)
        assert det.n_labelled_ == 0
        assert "zero cells" in (det.fallback_reason_ or "")
        assert np.all(np.isfinite(_score_flat(det, dirty_df)))

    def test_a_budget_without_an_oracle_raises_rather_than_guessing(
        self, synth, cache
    ) -> None:
        _, dirty_df, profile, _, _ = synth
        det = _ens(
            budget=20, profile=profile, seed=SEED, feature_cache=cache
        )
        with pytest.raises(ValueError, match="label_oracle"):
            det.fit(dirty_df)

    def test_scoring_before_spending_a_budget_raises(self, synth, cache) -> None:
        _, dirty_df, profile, labels, _ = synth
        det = _ens(
            budget=20,
            label_oracle=LabelOracle(labels),
            profile=profile,
            seed=SEED,
            feature_cache=cache,
        )
        with pytest.raises(RuntimeError, match="before fit"):
            det.score_cells(dirty_df)


# ═══════════════════════════════════════════════════════════════════════════════
# LEAKAGE — CLAUDE.md §3
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoLeakage:
    def test_no_public_signature_mentions_clean(self) -> None:
        targets = [build_features, normalised_ranks, EnsembleDetector.__init__]
        targets += [
            getattr(EnsembleDetector, name)
            for name in dir(EnsembleDetector)
            if not name.startswith("_") and callable(getattr(EnsembleDetector, name))
        ]
        targets += [LabelOracle.__init__, LabelOracle.query]
        for target in targets:
            for param in inspect.signature(target).parameters:
                assert "clean" not in param.lower(), f"{target} has {param}"

    def test_fit_has_nowhere_to_receive_labels(self) -> None:
        params = list(inspect.signature(EnsembleDetector.fit).parameters)
        assert params == ["self", "dirty_df", "train_mask", "profile"]
        for param in params:
            assert "label" not in param.lower()
            assert param not in ("y", "y_true", "targets")

    def test_the_feature_builder_takes_no_ground_truth(self) -> None:
        params = set(inspect.signature(build_features).parameters)
        assert not {"labels", "y", "clean_df", "oracle"} & params

    def test_every_queried_cell_is_a_train_cell(self, synth, cache) -> None:
        _, dirty_df, profile, labels, _ = synth
        mask, flat = _train_mask(dirty_df)
        oracle = LabelOracle(labels)
        det = _ens(
            budget=50,
            label_oracle=oracle,
            profile=profile,
            seed=SEED,
            feature_cache=cache,
        )
        det.fit(dirty_df, mask)
        assert oracle.queried_indices, "nothing was queried; the test proves nothing"
        assert all(bool(flat[i]) for i in oracle.queried_indices)

    def test_eval_labels_cannot_change_the_fitted_scores(
        self, synth, cache
    ) -> None:
        """
        The counterfactual test. Invert every label *outside* the train mask
        and refit: if any eval label reached the meta-classifier, the scores
        would move. They must be bitwise identical.
        """
        _, dirty_df, profile, labels, _ = synth
        mask, flat = _train_mask(dirty_df)

        poisoned_flat = flatten_cells(labels).astype(bool).copy()
        poisoned_flat[~flat] = ~poisoned_flat[~flat]
        n_rows = len(dirty_df)
        poisoned = pd.DataFrame(
            {
                col: poisoned_flat[i * n_rows : (i + 1) * n_rows]
                for i, col in enumerate(dirty_df.columns)
            },
            index=dirty_df.index,
        )
        assert not poisoned.equals(labels), "the poisoning changed nothing"

        scores = []
        for label_frame in (labels, poisoned):
            det = _ens(
                budget=50,
                label_oracle=LabelOracle(label_frame),
                profile=profile,
                seed=SEED,
                feature_cache=cache,
            )
            det.fit(dirty_df, mask)
            assert det.used_supervision_
            scores.append(_score_flat(det, dirty_df))
        assert np.array_equal(scores[0], scores[1])

    def test_the_oracle_cap_is_enforced_per_fit(self, synth, cache) -> None:
        _, dirty_df, profile, labels, _ = synth
        mask, _ = _train_mask(dirty_df)
        oracle = LabelOracle(labels, max_queries=20)
        det = _ens(
            budget=20,
            label_oracle=oracle,
            profile=profile,
            seed=SEED,
            feature_cache=cache,
        )
        det.fit(dirty_df, mask)
        assert oracle.n_queries == 20
        with pytest.raises(RuntimeError, match="budget exhausted"):
            det.fit(dirty_df, mask)


# ═══════════════════════════════════════════════════════════════════════════════
# Determinism, validation, harness integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    @pytest.mark.parametrize("budget", [0, 20])
    def test_same_seed_same_scores(self, synth, budget) -> None:
        _, dirty_df, profile, labels, _ = synth
        mask, _ = _train_mask(dirty_df)
        runs = []
        for _ in range(2):
            det = _ens(
                budget=budget,
                label_oracle=LabelOracle(labels),
                profile=profile,
                seed=SEED,
                feature_cache=FeatureCache(),  # a cold cache each time
            )
            det.fit(dirty_df, mask)
            runs.append(_score_flat(det, dirty_df))
        assert np.array_equal(runs[0], runs[1])

    def test_the_cache_does_not_change_the_answer(self, synth, cache) -> None:
        _, dirty_df, profile, _, _ = synth
        warm = _ens(
            budget=0, profile=profile, seed=SEED, feature_cache=cache
        )
        cold = _ens(
            budget=0, profile=profile, seed=SEED, feature_cache=FeatureCache()
        )
        assert np.array_equal(
            _score_flat(warm, dirty_df), _score_flat(cold, dirty_df)
        )


class TestValidation:
    @pytest.mark.parametrize("budget", [-1, -100])
    def test_negative_budget_rejected(self, budget) -> None:
        with pytest.raises(ValueError):
            _ens(budget=budget)

    @pytest.mark.parametrize("budget", ["twenty", 2.5, None, True])
    def test_non_integer_budget_rejected(self, budget) -> None:
        with pytest.raises((ValueError, TypeError)):
            _ens(budget=budget)

    def test_unknown_combiner_and_learner_rejected(self) -> None:
        with pytest.raises(ValueError, match="combiner"):
            _ens(combiner="magic")
        with pytest.raises(ValueError, match="learner"):
            _ens(learner="oracle")

    @pytest.mark.parametrize("top_fraction", [-0.1, 1.5])
    def test_bad_top_fraction_rejected(self, top_fraction) -> None:
        with pytest.raises(ValueError, match="top_fraction"):
            _ens(top_fraction=top_fraction)

    def test_negative_seed_rejected(self) -> None:
        with pytest.raises(ValueError, match="seed"):
            _ens(seed=-1)

    def test_mismatched_train_mask_rejected(self, synth, cache) -> None:
        _, dirty_df, profile, _, _ = synth
        with pytest.raises(ValueError, match="cells"):
            _ens(
                budget=0, profile=profile, seed=SEED, feature_cache=cache
            ).fit(dirty_df, pd.DataFrame({"a": [True, False]}))

    def test_default_name_encodes_the_budget(self) -> None:
        assert _ens(budget=0).name == "ensemble_b0"
        assert _ens(budget=20).name == "ensemble_b20"
        assert _ens(budget="full").name == "ensemble_bfull"

    def test_combiners_are_the_documented_set(self) -> None:
        assert COMBINERS == ("rank_mean", "noisy_or", "rule_only", "cond_only")

    def test_report_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="before fit"):
            _ens(budget=0).report()


class TestHarnessIntegration:
    """The ensemble must be measurable by the wave-1 harness, unmodified."""

    def test_evaluate_detector_runs_and_refits_per_split(
        self, synth_root, synth, cache
    ) -> None:
        _, _, profile, labels, _ = synth
        oracle = LabelOracle(labels, max_queries=20 * 3)
        det = _ens(
            budget=20,
            label_oracle=oracle,
            profile=profile,
            seed=SEED,
            feature_cache=cache,
            name="ensemble_b20",
        )
        out = evaluate_detector(
            det, "synth", data_root=str(synth_root), n_splits=3, seed=SEED
        )
        assert out["method"] == "ensemble_b20"
        assert out["refit_per_split"] is True
        assert out["full_is_in_sample"] is True
        assert out["intervals"]["pr_auc"]["n"] == 3
        assert out["intervals"]["pr_auc"]["mean"] > out["label_stats"]["prevalence"]
        # Three splits, 20 labels each: the budget is per fit, enforced.
        assert oracle.n_queries == 60

    def test_harness_splits_give_the_oracle_only_train_cells(
        self, synth_root, synth
    ) -> None:
        """
        End-to-end leakage check through the harness itself: the harness builds
        the train mask, so this proves the real protocol, not a hand-made mask.
        """
        _, dirty_df, profile, labels, _ = synth

        seen: list = []

        class SpyOracle(LabelOracle):
            def query(self, flat_indices):  # type: ignore[override]
                seen.append(np.asarray(list(flat_indices), dtype=int))
                return super().query(flat_indices)

        det = _ens(
            budget=20,
            label_oracle=SpyOracle(labels),
            profile=profile,
            seed=SEED,
            feature_cache=FeatureCache(),
        )
        out = evaluate_detector(
            det, "synth", data_root=str(synth_root), n_splits=2, seed=SEED
        )
        assert len(seen) == 2
        # Reconstruct each split's eval set exactly as the harness does and
        # require no overlap with what was labelled.
        from src.detection_eval.harness import stratified_cell_splits

        y = flatten_cells(labels).astype(bool)
        splits = stratified_cell_splits(y, n_splits=2, seed=SEED)
        for queried, eval_idx in zip(seen, splits):
            assert not set(queried.tolist()) & set(eval_idx.tolist())
        assert out["per_split"]
