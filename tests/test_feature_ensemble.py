"""
tests/test_feature_ensemble.py
------------------------------
Tests for W8-C, the Raha-style per-cell feature-vector ensemble
(`src/detectors/feature_ensemble.py`, planned in `docs/w8_plan.md` §4).

The four that matter, because they can each actually fail:

- **Comparability across cardinality.** Two earlier mechanisms in this project
  died on a cross-column confound (W5.3 cardinality, W6.1 coverage). The rank
  features are asserted to have matching distributions on a 2-value column and
  a 500-value column, and the same assertion is run on the *raw* scores to
  show it discriminates — if someone swaps a rank for a raw score the first
  assertion fails.
- **Beats chance on held-out injected corruption.** A row-disjoint holdout,
  ROC-AUC re-derived from the metrics module rather than from the classifier's
  own ``score``.
- **No non-finite feature.** Checked on a frame built to be hostile: nulls, a
  constant column, an all-null column, mixed types, a near-unique ID.
- **No leakage.** A signature scan, plus a spy that records every frame
  reaching the profile factory and every ``profile`` argument reaching the
  conditional detector, and fails if a clean frame or a non-``None`` profile
  ever appears.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.detection_eval.harness import RuleDetectorAdapter, flatten_cells
from src.detection_eval.metrics import roc_auc
from src.detectors import feature_ensemble as fe
from src.detectors.feature_ensemble import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    FEATURE_SCOPE,
    RULE_METHODS,
    FeatureEnsembleDetector,
    dirty_only_profile,
    extract_features,
)

SEED = 42


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


def _frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """A benign mixed frame: two categoricals, two numerics."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "cat2": rng.choice(["a", "b"], n),
            "cat8": rng.choice(list("abcdefgh"), n),
            "num": rng.normal(10, 2, n).round(3),
            "count": rng.integers(0, 50, n),
        }
    )


def _hostile_frame(n: int = 200, seed: int = 1) -> pd.DataFrame:
    """Every shape known to produce a NaN somewhere if handled carelessly."""
    rng = np.random.default_rng(seed)
    num = rng.normal(0, 1, n)
    num[:20] = np.nan
    return pd.DataFrame(
        {
            "const": ["same"] * n,
            "all_null": [np.nan] * n,
            "num_with_nan": num,
            "mixed": [str(i) if i % 3 else f"x{i}" for i in range(n)],
            "idlike": np.arange(n),
            "cat": rng.choice(["p", "q", "r"], n),
        }
    )


def _factory(name: str = "test"):
    return lambda df: dirty_only_profile(name, df)


#: sklearn's HistGradientBoosting has a fixed per-model cost that dominates on
#: a few-hundred-row test frame, and the conditional detector fits one model
#: per column. 20 iterations keeps the tests in seconds without changing any
#: behaviour under test -- nothing here asserts a score *value*, only
#: structure, finiteness, determinism, ordering and better-than-chance.
FAST = {"max_iter": 20}


def _fitted(df: pd.DataFrame, **kwargs) -> FeatureEnsembleDetector:
    params = dict(
        seed=SEED,
        injection_rate=0.02,
        profile_factory=_factory(),
        max_iter=40,
        conditional_kwargs=FAST,
        **kwargs,
    )
    return FeatureEnsembleDetector(**params).fit(df)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _frame()


@pytest.fixture(scope="module")
def fitted(frame) -> FeatureEnsembleDetector:
    return _fitted(frame)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Feature vector structure
# ═══════════════════════════════════════════════════════════════════════════


class TestFeatureSpec:
    def test_names_are_unique(self):
        assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)

    def test_groups_partition_the_names(self):
        flat = [n for names in FEATURE_GROUPS.values() for n in names]
        assert sorted(flat) == sorted(FEATURE_NAMES)

    def test_three_scopes_present(self):
        # F4 in docs/w8_plan.md: attribute, tuple, dataset-level context.
        assert set(FEATURE_GROUPS) == {"cell", "attribute", "tuple"}
        for scope in FEATURE_GROUPS:
            assert any(FEATURE_SCOPE[n] == scope for n in FEATURE_NAMES)

    def test_every_rule_method_has_a_feature(self):
        for m in RULE_METHODS:
            assert f"rule_{m}" in FEATURE_NAMES

    def test_rule_methods_match_the_detector(self, frame):
        """
        If src/detection/detector.py grows a seventh check, RULE_METHODS must
        grow with it, or that check silently stops being a feature.
        """
        det = fe.Detector(dirty_only_profile("t", frame))
        emitted = {r.detection_method for r in det.detect(frame)}
        assert emitted <= set(RULE_METHODS), (
            f"detector emitted unknown method(s): {emitted - set(RULE_METHODS)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Extraction: finiteness, determinism, ordering
# ═══════════════════════════════════════════════════════════════════════════


#: Extraction is deterministic given the frame, and a conditional fit is the
#: expensive part, so identical frames are extracted once. Keyed by a content
#: hash, NOT by object identity, so two separately-built but equal frames
#: share a result -- which is also a (weak) determinism check in itself.
_EXTRACT_CACHE: dict = {}


def _extract(df: pd.DataFrame, cache: bool = True):
    from src.detectors.conditional import ConditionalDetector

    key = (
        df.shape,
        tuple(map(str, df.columns)),
        int(pd.util.hash_pandas_object(df.astype(str), index=True).sum()),
    )
    if cache and key in _EXTRACT_CACHE:
        return _EXTRACT_CACHE[key]
    cond = ConditionalDetector(mode="full", seed=SEED, **FAST).fit(df, profile=None)
    rules = fe.Detector(dirty_only_profile("t", df))
    out = extract_features(df, cond, rules)
    if cache:
        _EXTRACT_CACHE[key] = out
    return out


class TestExtraction:
    def test_shape_and_names(self, frame):
        m, _ = _extract(frame)
        assert m.x.shape == (frame.size, len(FEATURE_NAMES))
        assert m.names == FEATURE_NAMES

    def test_no_nan_or_inf_on_a_benign_frame(self, frame):
        m, _ = _extract(frame)
        assert np.all(np.isfinite(m.x))

    def test_no_nan_or_inf_on_a_hostile_frame(self):
        """Nulls, an all-null column, a constant, mixed types, an ID."""
        m, _ = _extract(_hostile_frame())
        bad = [
            FEATURE_NAMES[j]
            for j in range(m.x.shape[1])
            if not np.all(np.isfinite(m.x[:, j]))
        ]
        assert bad == [], f"non-finite feature(s): {bad}"

    def test_extraction_is_deterministic(self, frame):
        # cache=False on both, or this would assert nothing at all.
        a, _ = _extract(frame, cache=False)
        b, _ = _extract(frame, cache=False)
        np.testing.assert_array_equal(a.x, b.x)

    def test_cell_order_is_column_major(self, frame):
        """Must line up with detection_eval.flatten_cells, byte for byte."""
        m, info = _extract(frame)
        expected_cols = flatten_cells(
            pd.DataFrame(
                {c: [c] * len(frame) for c in frame.columns}, columns=frame.columns
            )
        )
        assert list(m.cell_columns) == list(expected_cols)
        # and the cond_full column of x equals the flattened score frame
        j = FEATURE_NAMES.index("cond_full")
        np.testing.assert_allclose(
            m.x[:, j], flatten_cells(info["mode_frames"]["full"])
        )

    def test_to_cell_frame_round_trip(self, frame):
        m, info = _extract(frame)
        j = FEATURE_NAMES.index("cond_marginal")
        back = m.to_cell_frame(m.x[:, j], frame)
        pd.testing.assert_frame_equal(
            back, info["mode_frames"]["marginal"], check_dtype=False
        )

    def test_subset_selects_and_orders(self, frame):
        m, _ = _extract(frame)
        s = m.subset(["is_missing", "cond_full"])
        assert s.names == ("is_missing", "cond_full")
        np.testing.assert_array_equal(
            s.x[:, 1], m.x[:, FEATURE_NAMES.index("cond_full")]
        )

    def test_rule_features_zero_without_a_detector(self, frame):
        from src.detectors.conditional import ConditionalDetector

        cond = ConditionalDetector(mode="full", seed=SEED, **FAST).fit(
            frame, profile=None
        )
        m, info = extract_features(frame, cond, None)
        assert info["rule_detector_used"] is False
        for name in ["rule_score"] + [f"rule_{m_}" for m_ in RULE_METHODS]:
            assert np.all(m.x[:, FEATURE_NAMES.index(name)] == 0.0)

    def test_empty_frame_rejected(self):
        from src.detectors.conditional import ConditionalDetector

        df = _frame(50)
        cond = ConditionalDetector(mode="full", seed=SEED, **FAST).fit(
            df, profile=None
        )
        with pytest.raises(ValueError, match="empty"):
            extract_features(df.iloc[:0], cond, None)

    def test_non_finite_feature_is_an_error_not_a_fill(self, frame, monkeypatch):
        """
        A silent NaN in a tree ensemble's input changes the split structure
        without changing the output shape. Assert it raises.
        """
        monkeypatch.setattr(
            fe, "_rare_fraction", lambda s, threshold=0.01: float("nan")
        )
        with pytest.raises(ValueError, match="non-finite"):
            _extract(frame, cache=False)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Cross-column comparability — the confound that killed W5.3 and W6.1
# ═══════════════════════════════════════════════════════════════════════════


class TestComparability:
    """
    A feature that is comparable across columns must have (close to) the same
    distribution in a 2-value column and in a 500-value column, because a rank
    is uniform on (0, 1] by construction regardless of the column's content.
    """

    @staticmethod
    def _wide_cardinality_frame(n: int = 500) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        return pd.DataFrame(
            {
                "binary": rng.choice(["yes", "no"], n),
                "unique_text": [f"v{i}" for i in rng.permutation(n)],
                "many_num": rng.normal(0, 1, n).round(6),
                "few_num": rng.integers(0, 3, n),
            }
        )

    def _block(self, m, name: str, col: str, df: pd.DataFrame) -> np.ndarray:
        j = FEATURE_NAMES.index(name)
        k = list(df.columns).index(col)
        n = len(df)
        return m.x[k * n : (k + 1) * n, j]

    @pytest.mark.parametrize("feat", ["q_full", "q_marginal", "q_ratio"])
    def test_rank_features_have_matching_distributions(self, feat):
        df = self._wide_cardinality_frame()
        m, _ = _extract(df)
        qs = [0.25, 0.5, 0.75]
        per_col = {
            c: np.quantile(self._block(m, feat, c, df), qs) for c in df.columns
        }
        spread = np.max(np.array(list(per_col.values())), axis=0) - np.min(
            np.array(list(per_col.values())), axis=0
        )
        assert np.all(spread < 0.30), (
            f"{feat} quartiles differ by {spread} across columns of "
            f"cardinality {[df[c].nunique() for c in df.columns]}; "
            "the feature is not cross-column comparable"
        )

    def test_the_same_check_fails_for_raw_scores(self):
        """
        The guard above is only meaningful if it can fail. Raw ``cond_full``
        is exactly the non-comparable quantity the ranks replace, so running
        the identical check on it must NOT pass.
        """
        df = self._wide_cardinality_frame()
        m, _ = _extract(df)
        qs = [0.25, 0.5, 0.75]
        vals = np.array(
            [np.quantile(self._block(m, "cond_full", c, df), qs) for c in df.columns]
        )
        spread = vals.max(axis=0) - vals.min(axis=0)
        assert np.any(spread >= 0.30), (
            "raw cond_full happened to be comparable on this frame, so the "
            "comparability test above proves nothing here"
        )

    def test_rank_features_are_bounded(self):
        df = self._wide_cardinality_frame()
        m, _ = _extract(df)
        for feat in ("q_full", "q_marginal", "q_ratio"):
            v = m.x[:, FEATURE_NAMES.index(feat)]
            assert v.min() > 0.0 and v.max() <= 1.0

    def test_tuple_features_are_fractions_not_counts(self):
        """A 4-column and a 12-column table must put them on one scale."""
        rng = np.random.default_rng(3)
        narrow = pd.DataFrame({f"c{i}": rng.normal(size=200) for i in range(4)})
        wide = pd.DataFrame({f"c{i}": rng.normal(size=200) for i in range(12)})
        for name in ("row_frac_high", "row_mean_q_full", "row_missing_frac"):
            a = _extract(narrow)[0].x[:, FEATURE_NAMES.index(name)]
            b = _extract(wide)[0].x[:, FEATURE_NAMES.index(name)]
            assert a.max() <= 1.0 and b.max() <= 1.0
            assert abs(a.mean() - b.mean()) < 0.35, name


# ═══════════════════════════════════════════════════════════════════════════
# 4. Rule features agree with the shipped adapter
# ═══════════════════════════════════════════════════════════════════════════


class TestRuleParity:
    def test_rule_score_matches_rule_detector_adapter(self, frame):
        """
        ``rule_score`` is built here rather than by calling the adapter (to
        avoid a second detect() pass). Assert the two are identical instead of
        trusting the comment that says so.
        """
        m, _ = _extract(frame)
        mine = m.to_cell_frame(m.x[:, FEATURE_NAMES.index("rule_score")], frame)
        theirs = RuleDetectorAdapter().score_cells(
            frame, profile=dirty_only_profile("t", frame)
        )
        pd.testing.assert_frame_equal(mine, theirs, check_dtype=False)

    def test_rule_binaries_are_consistent_with_the_score(self, frame):
        m, _ = _extract(frame)
        score = m.x[:, FEATURE_NAMES.index("rule_score")]
        any_flag = np.zeros_like(score)
        for meth in RULE_METHODS:
            any_flag = np.maximum(any_flag, m.x[:, FEATURE_NAMES.index(f"rule_{meth}")])
        # every flagged cell has a non-zero score and vice versa
        assert np.array_equal(any_flag > 0, score > 0)


# ═══════════════════════════════════════════════════════════════════════════
# 5. The classifier actually learns something
# ═══════════════════════════════════════════════════════════════════════════


class TestLearning:
    def test_beats_chance_on_held_out_injected_corruption(self, fitted):
        auc = roc_auc(fitted.holdout_labels_, fitted.holdout_scores_)
        assert auc is not None
        assert auc > 0.70, f"held-out ROC-AUC {auc:.4f} is not above chance"

    def test_holdout_is_row_disjoint_from_training(self, fitted):
        m = fitted.train_features_
        in_train = fitted.train_row_mask_[m.cell_rows]
        train_rows = set(m.cell_rows[in_train].tolist())
        holdout_rows = set(m.cell_rows[~in_train].tolist())
        assert train_rows.isdisjoint(holdout_rows)

    def test_holdout_has_both_classes(self, fitted):
        assert 0 < int(fitted.holdout_labels_.sum()) < fitted.holdout_labels_.size

    def test_scores_are_probabilities(self, fitted, frame):
        s = fitted.score_cells(frame)
        assert s.shape == frame.shape
        assert list(s.columns) == list(frame.columns)
        v = s.to_numpy()
        assert np.all(np.isfinite(v)) and v.min() >= 0.0 and v.max() <= 1.0

    def test_scoring_is_deterministic(self, frame):
        a = _fitted(frame).score_cells(frame)
        b = _fitted(frame).score_cells(frame)
        pd.testing.assert_frame_equal(a, b)

    def test_seed_changes_the_injection(self, frame):
        a = _fitted(frame, injection_types=("missing",))
        b = FeatureEnsembleDetector(
            seed=SEED + 1,
            injection_rate=0.02,
            injection_types=("missing",),
            profile_factory=_factory(),
        ).fit(frame)
        assert not np.array_equal(a.train_labels_, b.train_labels_)

    def test_permutation_importance_covers_every_feature(self, fitted):
        imp = fitted.permutation_importance(n_repeats=1, sample=5000)
        assert set(imp) == set(fitted.feature_names)
        assert all(np.isfinite(v) for v in imp.values())

    def test_feature_subset_is_honoured(self, frame):
        names = ["q_full", "cond_full", "is_missing"]
        ens = _fitted(frame, feature_names=names)
        assert ens.feature_names == tuple(names)
        assert ens.classifier_.n_features_in_ == 3
        assert ens.report()["n_features"] == 3

    def test_report_counts_match_the_labels(self, fitted):
        r = fitted.report()
        assert r["n_injected_positives"] == int(fitted.train_labels_.sum())
        assert r["n_train_cells"] == int(fitted.train_labels_.size)
        assert r["rule_features_active"] is True


# ═══════════════════════════════════════════════════════════════════════════
# 6. Leakage — CLAUDE.md §3
# ═══════════════════════════════════════════════════════════════════════════


class TestNoLeakage:
    def test_no_public_parameter_mentions_clean(self):
        for name, obj in [
            ("extract_features", extract_features),
            ("FeatureEnsembleDetector.__init__", FeatureEnsembleDetector.__init__),
            ("FeatureEnsembleDetector.fit", FeatureEnsembleDetector.fit),
            ("FeatureEnsembleDetector.score_cells", FeatureEnsembleDetector.score_cells),
        ]:
            params = list(inspect.signature(obj).parameters)
            assert not any("clean" in p.lower() for p in params), (
                f"{name} has a clean-data parameter: {params}"
            )

    def test_profile_factory_only_ever_sees_dirty_derived_frames(self, frame):
        """
        A spy factory records every frame it is handed. The clean frame is
        never one of them — nothing in this module has access to clean data
        at all, and if someone wires one in, this fails.
        """
        seen = []

        def spy(df):
            seen.append(df.copy())
            return dirty_only_profile("t", df)

        clean = frame.copy()
        clean["num"] = 0.0  # a deliberately different "clean" frame
        ens = FeatureEnsembleDetector(
            seed=SEED, injection_rate=0.02, profile_factory=spy
        )
        ens.fit_score(frame)
        assert seen, "profile_factory was never called"
        for df in seen:
            assert not df.equals(clean)
            assert list(df.columns) == list(frame.columns)

    def test_conditional_detector_always_gets_profile_none(self, frame, monkeypatch):
        """
        A DatasetProfile is built from clean data, so passing one into the
        conditional detector would import the documented soft leakage.
        """
        from src.detectors import conditional as cond_mod

        seen = []
        real_fit = cond_mod.ConditionalDetector.fit

        def spy_fit(self, dirty_df, profile=None, train_mask=None):
            seen.append(profile)
            return real_fit(self, dirty_df, profile=profile, train_mask=train_mask)

        monkeypatch.setattr(cond_mod.ConditionalDetector, "fit", spy_fit)
        _fitted(frame).score_cells(frame)
        assert seen, "the conditional detector was never fitted"
        assert all(p is None for p in seen), f"profile leaked in: {seen}"

    def test_fit_ignores_a_profile_argument(self, frame):
        """
        ``fit`` accepts ``profile`` only for DetectorProtocol parity. Passing
        a profile must change nothing at all.
        """
        prof = dirty_only_profile("t", frame)
        a = _fitted(frame).score_cells(frame)
        b = FeatureEnsembleDetector(
            seed=SEED, injection_rate=0.02, profile_factory=_factory()
        ).fit(frame, profile=prof).score_cells(frame)
        pd.testing.assert_frame_equal(a, b)

    def test_dirty_only_profile_passes_the_same_frame_twice(self, frame, monkeypatch):
        import src.data.profiler as prof_mod

        seen = {}
        real = prof_mod.profile_dataset

        def spy(name, paths, clean_df, dirty_df, **kw):
            seen["same"] = clean_df is dirty_df
            return real(name, paths, clean_df, dirty_df, **kw)

        monkeypatch.setattr(prof_mod, "profile_dataset", spy)
        dirty_only_profile("t", frame)
        assert seen["same"] is True


# ═══════════════════════════════════════════════════════════════════════════
# 7. Validation and edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestValidation:
    @pytest.mark.parametrize("rate", [0.0, 1.0, -0.1, 2.0])
    def test_bad_injection_rate(self, rate):
        with pytest.raises(ValueError, match="injection_rate"):
            FeatureEnsembleDetector(injection_rate=rate)

    @pytest.mark.parametrize("frac", [0.0, 1.0, -0.5])
    def test_bad_train_fraction(self, frac):
        with pytest.raises(ValueError, match="train_fraction"):
            FeatureEnsembleDetector(train_fraction=frac)

    def test_empty_injection_types(self):
        with pytest.raises(ValueError, match="injection_types"):
            FeatureEnsembleDetector(injection_types=())

    def test_unknown_feature_name(self):
        with pytest.raises(ValueError, match="unknown feature"):
            FeatureEnsembleDetector(feature_names=["cond_full", "nope"])

    def test_empty_feature_names(self):
        with pytest.raises(ValueError, match="feature_names"):
            FeatureEnsembleDetector(feature_names=[])

    def test_score_before_fit(self, frame):
        with pytest.raises(RuntimeError, match="before fit"):
            FeatureEnsembleDetector().score_cells(frame)

    def test_report_before_fit(self):
        with pytest.raises(RuntimeError, match="before fit"):
            FeatureEnsembleDetector().report()

    def test_importance_before_fit(self):
        with pytest.raises(RuntimeError, match="before fit"):
            FeatureEnsembleDetector().permutation_importance()

    def test_fit_rejects_non_dataframe(self):
        with pytest.raises(TypeError):
            FeatureEnsembleDetector().fit([1, 2, 3])

    def test_fit_rejects_empty_frame(self):
        with pytest.raises(ValueError, match="empty"):
            FeatureEnsembleDetector().fit(pd.DataFrame())

    def test_uninjectable_frame_raises_rather_than_training_on_nothing(self):
        """
        A frame where nothing can be corrupted must fail loudly: a classifier
        with zero positives would otherwise return a constant score that a
        results table could mistake for a real detector.
        """
        df = pd.DataFrame({"a": ["same"] * 50, "b": ["same"] * 50})
        with pytest.raises(RuntimeError, match="no training positives|no synthetic"):
            FeatureEnsembleDetector(seed=SEED, injection_rate=0.1).fit(df)

    def test_to_cell_frame_rejects_wrong_length(self, frame):
        m, _ = _extract(frame)
        with pytest.raises(ValueError, match="expected"):
            m.to_cell_frame(np.zeros(3), frame)
