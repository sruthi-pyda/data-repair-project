"""
tests/test_text_fallback.py
---------------------------
Tests for `src/detectors/text_fallback.py`.

The decisive test is `TestRareButValid`. Five mechanisms in this project have
failed by ranking *rare* things first instead of *wrong* things, so a text
detector that scores a legitimately rare value above a typo has reproduced
the same failure in a new place. That test builds both cases explicitly and
is written to be able to fail.

Fixtures construct their own ground truth, so expected answers are derived
independently of the code under test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.detectors.text_fallback import (
    NAN_SCORE,
    SIGNALS,
    TextFallbackDetector,
    signature,
)


def _city_column(n_each: int = 60) -> pd.DataFrame:
    """
    A realistic high-cardinality text column with three KNOWN cases:

      index -3  `Bostonn`      a TYPO of a frequent value   -> should score HIGH
      index -2  `Ouagadougou`  legitimately rare, far from
                               everything                   -> should score LOW
      index -1  `B0st0n!!`     a format/parse break         -> should score HIGH
    """
    frequent = ["Boston", "Cambridge", "Springfield", "Worcester"]
    values: list[str] = []
    for name in frequent:
        values.extend([name] * n_each)
    values.extend(["Bostonn", "Ouagadougou", "B0st0n!!"])
    return pd.DataFrame({"city": values})


@pytest.fixture(scope="module")
def scored() -> tuple[np.ndarray, pd.DataFrame]:
    df = _city_column()
    det = TextFallbackDetector()
    out = det.score_cells(df)
    return out["city"].to_numpy(dtype=float), df


# ═══════════════════════════════════════════════════════════════════════════
# THE DECISIVE TEST
# ═══════════════════════════════════════════════════════════════════════════


class TestRareButValid:
    """
    The distinction every previous mechanism failed to make.

    `Ouagadougou` and `Bostonn` are BOTH rare -- each appears once. One is
    correct, the other is a typo. A detector that only measures rarity cannot
    separate them, so these tests are the whole point.
    """

    def test_both_candidates_are_equally_rare(self, scored):
        """Premise check: if they differ in frequency the test proves nothing."""
        _, df = scored
        vc = df["city"].value_counts()
        assert vc["Bostonn"] == vc["Ouagadougou"] == 1

    def test_typo_outscores_the_legitimately_rare_value(self, scored):
        s, _ = scored
        typo, rare = s[-3], s[-2]
        assert typo > rare, (
            f"typo 'Bostonn' scored {typo:.4f} but legitimately rare "
            f"'Ouagadougou' scored {rare:.4f}; the detector is ranking "
            f"rarity, not wrongness"
        )

    def test_format_break_outscores_the_legitimately_rare_value(self, scored):
        s, _ = scored
        assert s[-1] > s[-2], (
            f"format break 'B0st0n!!' scored {s[-1]:.4f} but legitimately "
            f"rare 'Ouagadougou' scored {s[-2]:.4f}"
        )

    def test_frequent_values_score_low(self, scored):
        s, _ = scored
        assert float(np.max(s[:240])) < min(s[-3], s[-1]), (
            "a frequent value outscored a corrupted one"
        )


# ═══════════════════════════════════════════════════════════════════════════
# The neighbour signal, which is the inverted one
# ═══════════════════════════════════════════════════════════════════════════


class TestNeighbourSignal:
    def test_typo_is_close_to_a_frequent_value_and_rare_value_is_not(self):
        df = _city_column()
        det = TextFallbackDetector(signals=("neighbour",))
        raw = det._neighbour_scores(list(df["city"].astype("object")))
        # Similarity to the nearest frequent value: high for the typo, low
        # for the genuinely distant rare value. This is the mechanism.
        assert raw[-3] > 0.8, f"typo similarity only {raw[-3]:.4f}"
        assert raw[-2] < 0.6, (
            f"'Ouagadougou' similarity {raw[-2]:.4f} is too high; it should be "
            f"far from every frequent value"
        )

    def test_frequent_values_are_not_flagged_as_typos(self):
        df = _city_column()
        det = TextFallbackDetector(signals=("neighbour",))
        raw = det._neighbour_scores(list(df["city"].astype("object")))
        assert float(np.max(raw[:240])) == 0.0


class TestSignature:
    @pytest.mark.parametrize("value,expected", [
        ("Boston", "A"),
        ("1a0", "9A9"),
        ("ABC-123", "A-9"),
        ("", ""),
    ])
    def test_shape_collapses_runs(self, value, expected):
        assert signature(value) == expected

    def test_missing_has_its_own_signature(self):
        assert signature(np.nan) == "<NA>"

    def test_distinct_shapes_are_distinguished(self):
        assert signature("12/05/2020") != signature("2020-05-12")


# ═══════════════════════════════════════════════════════════════════════════
# Mechanics and contracts
# ═══════════════════════════════════════════════════════════════════════════


class TestMechanics:
    def test_scores_bounded_and_finite(self, scored):
        s, _ = scored
        assert np.all(np.isfinite(s))
        assert float(s.min()) >= 0.0 and float(s.max()) <= 1.0

    def test_missing_gets_nan_score(self):
        df = _city_column()
        df.loc[0, "city"] = np.nan
        out = TextFallbackDetector().score_cells(df)
        assert out["city"].iloc[0] == NAN_SCORE

    def test_deterministic(self):
        df = _city_column()
        a = TextFallbackDetector().score_cells(df)["city"].to_numpy()
        b = TextFallbackDetector().score_cells(df)["city"].to_numpy()
        assert np.array_equal(a, b)

    def test_each_signal_available_alone(self):
        df = _city_column()
        for sig in SIGNALS:
            out = TextFallbackDetector(signals=(sig,)).score_cells(df)
            assert np.all(np.isfinite(out["city"].to_numpy()))

    def test_unscored_columns_are_all_zero(self):
        df = _city_column()
        df["other"] = "x"
        out = TextFallbackDetector().score_cells(df, columns=["city"])
        assert float(out["other"].abs().max()) == 0.0

    def test_constant_column_is_reported_not_scored(self):
        df = _city_column()
        df["flat"] = "same"
        det = TextFallbackDetector()
        det.score_cells(df)
        assert det.reports_["flat"].scored is False
        assert det.reports_["flat"].reason == "constant"

    def test_output_shape_and_labels_preserved(self, scored):
        s, df = scored
        out = TextFallbackDetector().score_cells(df)
        assert list(out.columns) == list(df.columns)
        assert out.index.equals(df.index)

    def test_rejects_bad_input(self):
        with pytest.raises(TypeError):
            TextFallbackDetector().score_cells(["not", "a", "frame"])
        with pytest.raises(ValueError):
            TextFallbackDetector().score_cells(pd.DataFrame())
        with pytest.raises(ValueError):
            TextFallbackDetector().score_cells(_city_column(), columns=["nope"])

    def test_rejects_bad_params(self):
        with pytest.raises(ValueError):
            TextFallbackDetector(signals=("nonsense",))
        with pytest.raises(ValueError):
            TextFallbackDetector(frequent_threshold=0.0)
        with pytest.raises(ValueError):
            TextFallbackDetector(top_k=0)
        with pytest.raises(ValueError):
            TextFallbackDetector(ngram_n=1)


class TestAbstention:
    """
    The detector may mark a column unsupported rather than guess
    (`docs/w9_plan.md` §W9.2b). The `neighbour` signal needs frequent values
    to act as typo sources; with too few, the signal that distinguishes
    rare-but-valid from corrupted cannot fire at all.
    """

    def test_all_unique_column_is_unsupported_not_scored(self):
        """An all-unique column has NO frequent values, so no typo sources."""
        df = pd.DataFrame({"uid": [f"id-{i:05d}" for i in range(400)]})
        det = TextFallbackDetector()
        out = det.score_cells(df)
        rep = det.reports_["uid"]
        assert rep.scored is False
        assert "unsupported" in rep.reason
        assert rep.n_frequent < 2
        # An abstained column must be all-zero, matching how
        # ConditionalDetector reports a column it skips.
        assert float(out["uid"].abs().max()) == 0.0

    def test_column_with_frequent_values_is_still_scored(self):
        det = TextFallbackDetector()
        det.score_cells(_city_column())
        assert det.reports_["city"].scored is True

    def test_abstention_can_be_disabled(self):
        """min_frequent=0 opts out, so the choice is explicit not hidden."""
        df = pd.DataFrame({"uid": [f"id-{i:05d}" for i in range(400)]})
        det = TextFallbackDetector(min_frequent=0)
        det.score_cells(df)
        assert det.reports_["uid"].scored is True

    def test_abstention_does_not_apply_without_the_neighbour_signal(self):
        """Pattern and ngram do not need typo sources, so they still run."""
        df = pd.DataFrame({"uid": [f"id-{i:05d}" for i in range(400)]})
        det = TextFallbackDetector(signals=("pattern", "ngram"))
        det.score_cells(df)
        assert det.reports_["uid"].scored is True

    def test_rejects_negative_min_frequent(self):
        with pytest.raises(ValueError):
            TextFallbackDetector(min_frequent=-1)


class TestNoLeakage:
    def test_profile_is_ignored(self):
        df = _city_column()
        a = TextFallbackDetector().score_cells(df)["city"].to_numpy()
        b = TextFallbackDetector().score_cells(
            df, profile={"clean": "statistics"})["city"].to_numpy()
        assert np.array_equal(a, b)

    def test_score_cells_signature_takes_no_clean_frame(self):
        import inspect
        params = set(inspect.signature(
            TextFallbackDetector.score_cells).parameters)
        assert params == {"self", "dirty_df", "columns", "profile"}, (
            f"must not accept clean data or labels; got {params}"
        )

    def test_scores_depend_only_on_the_column_passed(self):
        """Adding an unrelated column must not change a column's scores."""
        df = _city_column()
        a = TextFallbackDetector().score_cells(df)["city"].to_numpy()
        df2 = df.copy()
        df2["unrelated"] = np.arange(len(df2))
        b = TextFallbackDetector().score_cells(df2)["city"].to_numpy()
        assert np.array_equal(a, b)
