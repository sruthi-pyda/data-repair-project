"""
tests/test_synth_supervised.py
------------------------------
Behavioural tests for W8-B, learned-policy synthetic supervision
(``src/detectors/synth_supervised.py``).

The tests that matter here, and why each one can fail:

``TestColumnQuantile``
    The comparability device. The expected values are written out by hand
    (``[3, 1, 2] -> [1.0, 0.0, 0.5]``) rather than recomputed with the code
    under test.

``TestCrossColumnComparability``
    The constraint that killed W5.3 and W6.1. A 2-value column and a
    500-distinct-value column must produce rank features with the *same*
    distribution. The tolerance is tight enough that returning a raw score
    instead of a rank fails it.

``TestPolicyEstimation``
    The estimator must recover a transition distribution that the test
    itself constructed — one-directional ``1 -> 0``, and a two-way
    distribution with known 0.75/0.25 weights.

``TestClassifierSanity``
    A classifier trained on corruption synthesised at seed 1 must detect the
    *same policy's* corruption injected independently at seed 2, far above
    prevalence. This is the "can it learn anything at all" gate; without it a
    null implementation would pass the rest.

``TestNoLeakage``
    Arm 0 (designed policy) must read no clean data at all. Enforced three
    ways: a signature scan, a monkeypatch that makes both clean-reading
    functions explode, and a tripwire frame that raises on any access.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.detection_eval.metrics import pr_auc
from src.detectors import synth_supervised as ss
from src.detectors.synth_supervised import (
    CELL_FEATURES,
    FEATURE_NAMES,
    CorruptionPolicy,
    LabelledCell,
    SynthSupervisedDetector,
    choose_budget_cells,
    column_quantile,
    designed_policy,
    estimate_policy,
    extract_features,
    label_cells,
    oracle_balanced_budget,
    oracle_labelled_cells,
    synthesise,
)

SEED = 42


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures — small frames with corruption whose structure the test knows
# ═══════════════════════════════════════════════════════════════════════════


def _base_frame(n: int = 300) -> pd.DataFrame:
    """
    A frame with real conditional structure, so the conditional detector has
    something to model: ``target`` is a deterministic function of ``driver``.
    """
    rng = np.random.default_rng(SEED)
    driver = rng.integers(0, 6, size=n)
    return pd.DataFrame(
        {
            "driver": driver,
            "target": (driver >= 3).astype(int),
            "noise": rng.integers(0, 4, size=n),
            "text": [["aa", "bb", "cc"][int(d) % 3] for d in driver],
        }
    )


def _flip_target(df: pd.DataFrame, rows, frm: int = 1, to: int = 0) -> pd.DataFrame:
    """One-directional corruption of ``target``: ``frm -> to`` only."""
    out = df.copy()
    for r in rows:
        if int(out["target"].iat[r]) == frm:
            out.iloc[r, out.columns.get_loc("target")] = to
    return out


class _Tripwire(pd.DataFrame):
    """A frame that raises on any column access. Never legitimately touched."""

    @property
    def _constructor(self):  # pragma: no cover - pandas internals
        return _Tripwire

    def __getitem__(self, key: Any):  # pragma: no cover - must never run
        raise AssertionError(f"clean data was accessed: {key!r}")


# ═══════════════════════════════════════════════════════════════════════════
# column_quantile
# ═══════════════════════════════════════════════════════════════════════════


class TestColumnQuantile:
    def test_hand_computed(self):
        got = column_quantile(np.array([3.0, 1.0, 2.0]))
        assert np.allclose(got, [1.0, 0.0, 0.5])

    def test_monotone_and_bounded(self):
        x = np.array([-5.0, 0.0, 0.1, 7.0, 1000.0])
        q = column_quantile(x)
        assert q.min() == pytest.approx(0.0)
        assert q.max() == pytest.approx(1.0)
        assert np.all(np.diff(q) > 0)

    def test_constant_column_is_mid(self):
        q = column_quantile(np.full(10, 0.3))
        assert np.allclose(q, 0.5)

    def test_ties_are_averaged(self):
        q = column_quantile(np.array([1.0, 1.0, 2.0]))
        # ranks 0,0,2 -> tied pair averages to 0.5 of (0+1)/2 = 0.25
        assert np.allclose(q, [0.25, 0.25, 1.0])

    def test_nan_is_mid_not_extreme(self):
        q = column_quantile(np.array([1.0, np.nan, 3.0]))
        assert q[1] == pytest.approx(0.5)
        assert q[0] == pytest.approx(0.0)
        assert q[2] == pytest.approx(1.0)

    def test_all_nan(self):
        assert np.allclose(column_quantile(np.array([np.nan, np.nan])), 0.5)

    def test_single_value(self):
        assert np.allclose(column_quantile(np.array([4.0])), 0.5)

    def test_empty(self):
        assert column_quantile(np.array([])).size == 0

    def test_invariant_to_monotone_rescaling(self):
        x = np.array([1.0, 2.0, 3.0, 10.0])
        assert np.allclose(column_quantile(x), column_quantile(x * 1e6 + 7.0))


# ═══════════════════════════════════════════════════════════════════════════
# Cross-column comparability — the constraint two mechanisms died on
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossColumnComparability:
    @pytest.fixture(scope="class")
    def extracted(self):
        rng = np.random.default_rng(SEED)
        n = 400
        df = pd.DataFrame(
            {
                "binary": rng.integers(0, 2, size=n),
                "tiny": rng.integers(0, 3, size=n),
                "wide": rng.integers(0, 500, size=n),
                "huge_scale": rng.normal(1e6, 1e5, size=n),
            }
        )
        x, cells, _ = extract_features(df, seed=SEED)
        return df, x, cells

    def test_rank_feature_means_match_across_cardinalities(self, extracted):
        """
        Every per-column rank feature must average ~0.5 in *every* column,
        however many distinct values that column has. A raw score would not.
        """
        df, x, cells = extracted
        rank_feats = ["cond_q", "marg_q", "ratio_q", "freq_q", "len_dev_q"]
        cols = list(df.columns)
        for feat in rank_feats:
            j = FEATURE_NAMES.index(feat)
            means = []
            for col in cols:
                sel = np.array([c == col for (_, c) in cells])
                means.append(float(x[sel, j].mean()))
            spread = max(means) - min(means)
            assert spread < 0.08, f"{feat}: per-column means {means} differ by {spread}"

    def test_rank_feature_quartiles_match_across_cardinalities(self, extracted):
        """Stronger than means: the whole distribution must line up."""
        df, x, cells = extracted
        j = FEATURE_NAMES.index("cond_q")
        for q in (0.25, 0.5, 0.75):
            vals = []
            for col in df.columns:
                sel = np.array([c == col for (_, c) in cells])
                vals.append(float(np.quantile(x[sel, j], q)))
            assert max(vals) - min(vals) < 0.12, f"quantile {q}: {vals}"

    def test_all_cell_features_bounded(self, extracted):
        _, x, _ = extracted
        for feat in CELL_FEATURES:
            j = FEATURE_NAMES.index(feat)
            assert x[:, j].min() >= -1e-9, feat
            assert x[:, j].max() <= 1.0 + 1e-9, feat

    def test_cardinality_is_not_a_feature(self):
        """The W5.3 confound must not be in the feature set by any name."""
        joined = " ".join(FEATURE_NAMES)
        assert "cardinal" not in joined
        assert "n_distinct" not in joined
        assert "rare" not in joined

    def test_feature_matrix_shape_and_order(self, extracted):
        df, x, cells = extracted
        assert x.shape == (df.shape[0] * df.shape[1], len(FEATURE_NAMES))
        assert len(cells) == x.shape[0]
        assert cells[0] == (0, df.columns[0])


# ═══════════════════════════════════════════════════════════════════════════
# Budget selection and labelling
# ═══════════════════════════════════════════════════════════════════════════


class TestBudget:
    def test_random_budget_size_and_determinism(self):
        df = _base_frame(50)
        a = choose_budget_cells(df, 20, "random", seed=SEED)
        b = choose_budget_cells(df, 20, "random", seed=SEED)
        assert len(a) == 20 and a == b
        assert len(set(a)) == 20

    def test_different_seed_gives_different_cells(self):
        df = _base_frame(200)
        a = choose_budget_cells(df, 20, "random", seed=1)
        b = choose_budget_cells(df, 20, "random", seed=2)
        assert a != b

    def test_zero_budget(self):
        assert choose_budget_cells(_base_frame(10), 0) == []

    def test_budget_larger_than_table(self):
        df = pd.DataFrame({"a": [1, 2]})
        assert len(choose_budget_cells(df, 99, "random", seed=SEED)) == 2

    def test_negative_budget_rejected(self):
        with pytest.raises(ValueError):
            choose_budget_cells(_base_frame(10), -1)

    def test_unknown_strategy_rejected(self):
        with pytest.raises(ValueError):
            choose_budget_cells(_base_frame(10), 5, "wishful")

    def test_top_score_needs_scores(self):
        with pytest.raises(ValueError):
            choose_budget_cells(_base_frame(10), 5, "top_score")

    def test_top_score_picks_the_highest(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        scores = pd.DataFrame({"a": [0.1, 0.9, 0.2], "b": [0.0, 0.0, 0.95]})
        got = choose_budget_cells(df, 2, "top_score", scores=scores)
        assert got == [(2, "b"), (1, "a")]

    def test_label_cells_detects_corruption(self):
        clean = _base_frame(40)
        dirty = _flip_target(clean, [3, 7])
        cells = [(3, "target"), (0, "driver")]
        labels = label_cells(clean, dirty, cells)
        by = {(l.row, l.column): l for l in labels}
        # row 3 is corrupt only if its clean target was 1
        expected = int(clean["target"].iat[3]) == 1
        assert by[(3, "target")].is_corrupt is expected
        assert by[(0, "driver")].is_corrupt is False

    def test_label_cells_nan_equals_nan(self):
        clean = pd.DataFrame({"a": [np.nan, 1.0]})
        dirty = pd.DataFrame({"a": [np.nan, 1.0]})
        labels = label_cells(clean, dirty, [(0, "a"), (1, "a")])
        assert [l.is_corrupt for l in labels] == [False, False]

    def test_label_cells_int_float_equal(self):
        clean = pd.DataFrame({"a": [371]})
        dirty = pd.DataFrame({"a": [371.0]})
        assert label_cells(clean, dirty, [(0, "a")])[0].is_corrupt is False

    def test_label_cells_nan_vs_value_is_corrupt(self):
        clean = pd.DataFrame({"a": [5.0]})
        dirty = pd.DataFrame({"a": [np.nan]})
        assert label_cells(clean, dirty, [(0, "a")])[0].is_corrupt is True

    def test_oracle_labels_find_every_corruption(self):
        clean = _base_frame(60)
        rows = [1, 4, 9, 12, 20]
        dirty = _flip_target(clean, rows)
        changed = [
            r for r in rows if clean["target"].iat[r] != dirty["target"].iat[r]
        ]
        got = oracle_labelled_cells(clean, dirty)
        assert len(got) == len(changed)
        assert {g.column for g in got} == {"target"}
        assert {g.row for g in got} == set(changed)

    def test_oracle_balanced_budget_has_positives(self):
        clean = _base_frame(100)
        dirty = _flip_target(clean, list(range(0, 40)))
        budget = oracle_balanced_budget(clean, dirty, 20, seed=SEED)
        assert len(budget) == 20
        assert sum(1 for b in budget if b.is_corrupt) == 10


# ═══════════════════════════════════════════════════════════════════════════
# Policy estimation — must recover a distribution the test constructed
# ═══════════════════════════════════════════════════════════════════════════


class TestPolicyEstimation:
    def test_recovers_one_directional_transition(self):
        dirty = pd.DataFrame({"a": [1, 0, 1, 0, 1] * 20})
        labelled = [
            LabelledCell(0, "a", 0, 1, True),
            LabelledCell(2, "a", 0, 1, True),
            LabelledCell(1, "a", 0, 0, False),
        ]
        pol = estimate_policy(dirty, labelled)
        assert pol.source == "learned"
        assert list(pol.columns) == ["a"]
        cp = pol.columns["a"]
        assert cp.weight == pytest.approx(1.0)
        assert cp.transitions == {1: {0: 1.0}}
        assert cp.is_one_directional is True
        assert cp.kind_probs == {"swap": 1.0}
        assert pol.n_labels == 3 and pol.n_positive_labels == 2

    def test_recovers_known_two_way_distribution(self):
        """Three ``1 -> 0`` and one ``0 -> 1``: weights must be 0.75 / 0.25."""
        dirty = pd.DataFrame({"a": [1, 0] * 50})
        labelled = (
            [LabelledCell(i, "a", 0, 1, True) for i in range(3)]
            + [LabelledCell(10, "a", 1, 0, True)]
        )
        cp = estimate_policy(dirty, labelled).columns["a"]
        assert cp.transitions[1] == {0: 1.0}
        assert cp.transitions[0] == {1: 1.0}
        assert cp.value_dist[0] == pytest.approx(0.75)
        assert cp.value_dist[1] == pytest.approx(0.25)
        assert cp.is_one_directional is False

    def test_column_weights_follow_label_counts(self):
        dirty = pd.DataFrame({"a": [1, 0] * 30, "b": [5, 6] * 30})
        labelled = [
            LabelledCell(0, "a", 0, 1, True),
            LabelledCell(2, "a", 0, 1, True),
            LabelledCell(4, "a", 0, 1, True),
            LabelledCell(1, "b", 6, 5, True),
        ]
        pol = estimate_policy(dirty, labelled)
        assert pol.columns["a"].weight == pytest.approx(0.75)
        assert pol.columns["b"].weight == pytest.approx(0.25)

    def test_rate_is_positive_fraction_of_labels(self):
        dirty = pd.DataFrame({"a": [1, 0] * 50})
        labelled = [LabelledCell(i, "a", 0, 1, True) for i in range(5)] + [
            LabelledCell(50 + i, "a", 1, 1, False) for i in range(15)
        ]
        assert estimate_policy(dirty, labelled).rate == pytest.approx(0.25)

    def test_no_positives_gives_empty_policy_with_a_note(self):
        dirty = pd.DataFrame({"a": [1, 0] * 50})
        labelled = [LabelledCell(i, "a", 1, 1, False) for i in range(20)]
        pol = estimate_policy(dirty, labelled)
        assert pol.is_empty
        assert pol.n_labels == 20 and pol.n_positive_labels == 0
        assert any("0 corrupted cells" in n for n in pol.notes)

    def test_empty_policy_refuses_to_fit_rather_than_falling_back(self):
        pol = CorruptionPolicy(source="learned", rate=0.0, columns={})
        with pytest.raises(ValueError, match="empty"):
            SynthSupervisedDetector(pol)

    def test_missing_kind_detected(self):
        dirty = pd.DataFrame({"a": [1.0, 2.0, np.nan] * 20})
        cp = estimate_policy(
            dirty, [LabelledCell(2, "a", np.nan, 2.0, True)]
        ).columns["a"]
        assert cp.kind_probs == {"missing": 1.0}

    def test_parse_break_kind_detected(self):
        dirty = pd.DataFrame({"a": [1.0, 2.0, 3.0] * 20 + ["2O"]})
        cp = estimate_policy(
            dirty, [LabelledCell(60, "a", "2O", 20.0, True)]
        ).columns["a"]
        assert cp.kind_probs == {"parse_break": 1.0}

    def test_out_of_range_kind_detected(self):
        dirty = pd.DataFrame({"a": list(range(50)) + [99999]})
        cp = estimate_policy(
            dirty, [LabelledCell(50, "a", 99999, 7, True)]
        ).columns["a"]
        assert cp.kind_probs == {"out_of_range": 1.0}

    def test_oracle_source_is_flagged(self):
        dirty = pd.DataFrame({"a": [1, 0] * 50})
        pol = estimate_policy(
            dirty, [LabelledCell(0, "a", 0, 1, True)], source="oracle"
        )
        assert pol.is_oracle is True
        assert pol.to_dict()["source"] == "oracle"

    def test_designed_policy_rejects_empty_frame(self):
        with pytest.raises(ValueError):
            designed_policy(pd.DataFrame())


# ═══════════════════════════════════════════════════════════════════════════
# Synthesis
# ═══════════════════════════════════════════════════════════════════════════


class TestSynthesis:
    def test_learned_policy_corrupts_only_its_columns(self):
        df = _base_frame(200)
        pol = estimate_policy(df, [LabelledCell(0, "target", 0, 1, True)])
        synth, mask, info = synthesise(df, pol, seed=SEED, target_positives=50)
        assert mask["target"].sum() > 0
        for col in df.columns:
            if col != "target":
                assert mask[col].sum() == 0
        assert synth.shape == df.shape

    def test_mask_marks_exactly_the_changed_cells(self):
        df = _base_frame(200)
        pol = estimate_policy(df, [LabelledCell(0, "target", 0, 1, True)])
        synth, mask, _ = synthesise(df, pol, seed=SEED, target_positives=40)
        for col in df.columns:
            for i in range(len(df)):
                changed = df[col].iat[i] != synth[col].iat[i]
                assert bool(mask[col].iat[i]) == bool(changed), (col, i)

    def test_one_directional_policy_writes_only_that_value(self):
        df = _base_frame(200)
        pol = estimate_policy(df, [LabelledCell(0, "target", 0, 1, True)])
        synth, mask, _ = synthesise(df, pol, seed=SEED, target_positives=60)
        written = synth["target"][mask["target"]].unique()
        assert list(written) == [0]

    def test_designed_policy_spreads_over_many_columns(self):
        df = _base_frame(300)
        synth, mask, info = synthesise(
            df, designed_policy(df, rate=0.05), seed=SEED, target_positives=200
        )
        assert mask.to_numpy().sum() > 0
        assert len(info["per_column"]) >= 2

    def test_synthesis_is_deterministic(self):
        df = _base_frame(150)
        pol = estimate_policy(df, [LabelledCell(0, "target", 0, 1, True)])
        a, ma, _ = synthesise(df, pol, seed=SEED, target_positives=30)
        b, mb, _ = synthesise(df, pol, seed=SEED, target_positives=30)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(ma, mb)

    def test_different_seed_changes_positions(self):
        df = _base_frame(300)
        pol = estimate_policy(df, [LabelledCell(0, "target", 0, 1, True)])
        _, ma, _ = synthesise(df, pol, seed=1, target_positives=30)
        _, mb, _ = synthesise(df, pol, seed=2, target_positives=30)
        assert not ma.equals(mb)

    def test_empty_policy_cannot_synthesise(self):
        df = _base_frame(50)
        pol = CorruptionPolicy(source="learned", rate=0.0, columns={})
        with pytest.raises(ValueError, match="empty"):
            synthesise(df, pol, seed=SEED)

    def test_missing_kind_writes_nan(self):
        df = pd.DataFrame({"a": np.arange(100.0), "b": np.arange(100.0)})
        pol = estimate_policy(df, [LabelledCell(0, "a", np.nan, 1.0, True)])
        synth, mask, _ = synthesise(df, pol, seed=SEED, target_positives=10)
        assert synth["a"][mask["a"]].isna().all()


# ═══════════════════════════════════════════════════════════════════════════
# Classifier sanity — the gate a null implementation fails
# ═══════════════════════════════════════════════════════════════════════════


class TestClassifierSanity:
    def test_detects_held_out_corruption_of_the_same_policy(self):
        """
        Train on corruption synthesised at seed 1; evaluate on an
        *independently* synthesised table at seed 2 under the same policy.
        PR-AUC must be far above prevalence, or the mechanism cannot learn
        even its own corruption and nothing else it reports means anything.
        """
        df = _base_frame(600)
        pol = estimate_policy(df, [LabelledCell(0, "target", 0, 1, True)])
        det = SynthSupervisedDetector(
            pol, seed=1, target_positives=150, negatives_per_positive=10
        ).fit(df)

        held, held_mask, _ = synthesise(df, pol, seed=2, target_positives=150)
        scores = det.score_cells(held)
        y = held_mask.to_numpy().ravel().astype(int)
        s = scores.to_numpy().ravel()
        prevalence = float(y.mean())
        ap = pr_auc(y, s)
        assert ap is not None
        assert ap > 5.0 * prevalence, f"PR-AUC {ap} vs prevalence {prevalence}"

    def test_designed_arm_also_learns_its_own_corruption(self):
        df = _base_frame(500)
        pol = designed_policy(df, rate=0.03)
        det = SynthSupervisedDetector(pol, seed=1, target_positives=120).fit(df)
        held, held_mask, _ = synthesise(df, pol, seed=2, target_positives=120)
        y = held_mask.to_numpy().ravel().astype(int)
        s = det.score_cells(held).to_numpy().ravel()
        ap = pr_auc(y, s)
        assert ap is not None and ap > 2.0 * float(y.mean())

    def test_scores_are_probabilities_shaped_like_the_frame(self):
        df = _base_frame(250)
        pol = estimate_policy(df, [LabelledCell(0, "target", 0, 1, True)])
        det = SynthSupervisedDetector(pol, seed=SEED, target_positives=60).fit(df)
        s = det.score_cells(df)
        assert s.shape == df.shape
        assert list(s.columns) == list(df.columns)
        arr = s.to_numpy()
        assert arr.min() >= 0.0 and arr.max() <= 1.0

    def test_determinism_of_scores(self):
        df = _base_frame(250)
        pol = estimate_policy(df, [LabelledCell(0, "target", 0, 1, True)])
        a = SynthSupervisedDetector(pol, seed=SEED, target_positives=60).fit(
            df
        ).score_cells(df)
        b = SynthSupervisedDetector(pol, seed=SEED, target_positives=60).fit(
            df
        ).score_cells(df)
        pd.testing.assert_frame_equal(a, b)

    def test_score_before_fit_raises(self):
        df = _base_frame(60)
        pol = estimate_policy(df, [LabelledCell(0, "target", 0, 1, True)])
        with pytest.raises(RuntimeError):
            SynthSupervisedDetector(pol, seed=SEED).score_cells(df)

    def test_column_mismatch_raises(self):
        df = _base_frame(200)
        pol = estimate_policy(df, [LabelledCell(0, "target", 0, 1, True)])
        det = SynthSupervisedDetector(pol, seed=SEED, target_positives=40).fit(df)
        with pytest.raises(ValueError, match="column mismatch"):
            det.score_cells(df.drop(columns=["noise"]))

    def test_bad_negative_ratio_rejected(self):
        df = _base_frame(60)
        pol = estimate_policy(df, [LabelledCell(0, "target", 0, 1, True)])
        with pytest.raises(ValueError):
            SynthSupervisedDetector(pol, negatives_per_positive=0)

    def test_report_names_the_policy_source(self):
        df = _base_frame(200)
        pol = estimate_policy(
            df, [LabelledCell(0, "target", 0, 1, True)], source="oracle"
        )
        det = SynthSupervisedDetector(pol, seed=SEED, target_positives=40).fit(df)
        rep = det.report()
        assert rep["policy_source"] == "oracle"
        assert rep["is_oracle"] is True
        assert rep["features"] == list(FEATURE_NAMES)


# ═══════════════════════════════════════════════════════════════════════════
# No leakage (CLAUDE.md §3)
# ═══════════════════════════════════════════════════════════════════════════


class TestNoLeakage:
    def test_detector_signatures_mention_no_clean_data(self):
        for obj in (
            SynthSupervisedDetector.__init__,
            SynthSupervisedDetector.fit,
            SynthSupervisedDetector.score_cells,
            designed_policy,
            synthesise,
            extract_features,
            choose_budget_cells,
        ):
            for name in inspect.signature(obj).parameters:
                assert "clean" not in name.lower(), (obj, name)

    def test_arm_zero_never_calls_a_clean_reading_function(self, monkeypatch):
        """
        Arm 0 end to end with both clean-reading entry points booby-trapped.
        If the designed arm touched clean data by any route, this explodes.
        """
        def boom(*a, **k):  # pragma: no cover - must never run
            raise AssertionError("arm 0 read clean data")

        monkeypatch.setattr(ss, "label_cells", boom)
        monkeypatch.setattr(ss, "oracle_labelled_cells", boom)
        monkeypatch.setattr(ss, "oracle_balanced_budget", boom)

        df = _base_frame(300)
        pol = designed_policy(df, rate=0.03)
        det = SynthSupervisedDetector(pol, seed=SEED, target_positives=80).fit(df)
        scores = det.score_cells(df)
        assert scores.shape == df.shape

    def test_arm_zero_pipeline_ignores_a_tripwire_clean_frame(self):
        """
        The clean frame exists in the caller's scope; arm 0 must not be able
        to reach it. Accessing any column of ``_Tripwire`` raises.
        """
        clean = _Tripwire(_base_frame(250))
        dirty = _base_frame(250)
        pol = designed_policy(dirty, rate=0.03)
        det = SynthSupervisedDetector(pol, seed=SEED, target_positives=60).fit(dirty)
        s = det.score_cells(dirty)
        assert s.shape == dirty.shape
        assert isinstance(clean, pd.DataFrame)  # still untouched

    def test_label_cells_returns_labels_not_frames(self):
        clean = _base_frame(40)
        dirty = _flip_target(clean, [3])
        labels = label_cells(clean, dirty, [(3, "target")])
        assert all(isinstance(l, LabelledCell) for l in labels)
        assert not any(isinstance(getattr(l, f), pd.DataFrame)
                       for l in labels for f in ("dirty_value", "clean_value"))

    def test_budget_choice_uses_no_clean_data(self):
        """``choose_budget_cells`` takes no clean argument, by signature."""
        params = list(inspect.signature(choose_budget_cells).parameters)
        assert params == ["dirty_df", "n", "strategy", "scores", "seed"]

    def test_policy_object_carries_no_frame(self):
        clean = _base_frame(60)
        dirty = _flip_target(clean, [1, 2, 3, 4])
        labels = oracle_labelled_cells(clean, dirty)
        pol = estimate_policy(dirty, labels, source="oracle")
        payload = pol.to_dict()
        assert isinstance(payload, dict)
        for cp in pol.columns.values():
            assert not isinstance(cp.transitions, pd.DataFrame)
            assert not isinstance(cp.value_dist, pd.DataFrame)
