"""
tests/test_corruptor.py
-----------------------
Tests for :mod:`src.datagen.corruptor`, the controlled corruption injector
behind route W5.4b.

TESTING STANCE (CLAUDE.md §8b.5: "a test that cannot fail is not a test")
=========================================================================
The whole corruption study is only as trustworthy as the injector. If the
injector quietly corrupted cells it did not report, or reported cells it did
not corrupt, every PR-AUC in ``results/w5_corruption_study.md`` would be
measured against the wrong labels and nothing would reveal it. So the tests
here do not check that the code runs; they re-derive what it claims:

- **Round trip.** ``cell_labels(clean, corrupted)`` — the project's canonical
  "is this cell corrupted" test, used by every detection metric — is computed
  independently and asserted to equal the reported positions *exactly*, in
  both directions, on every type and on all three real datasets. A cell
  changed but not reported, or reported but not changed, fails.
- **The honesty anchor is proved, not asserted in prose.** Every value
  ``swap_valid`` writes is checked against the column's own value set. That
  property is the reason a rule detector *cannot* detect ``swap_valid``, and
  the study's credibility rests on it.
- **Format output is parsed, not eyeballed.** ``float()`` must raise on every
  injected value.
- **NMI is checked against analytically known values** (identical columns ->
  1.0, independent columns -> ~0.0), not against itself.
- **Determinism is checked across fresh processes' worth of state** by
  rebuilding the injector from scratch and comparing frames with
  ``DataFrame.equals``, and seed sensitivity is checked so that a
  constant-output bug cannot masquerade as determinism.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import pytest

from src.datagen.corruptor import (
    CORRUPTION_TYPES,
    OUT_OF_RANGE_MULTIPLIERS,
    ColumnRole,
    Corruptor,
    InjectionReport,
    _normalised_mutual_information,
    classify_columns,
    eligible_columns,
    inject,
)
from src.detection_eval.labels import cell_labels

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "datasets"

#: Row caps keep the suite fast; each is large enough that every corruption
#: type has thousands of eligible cells.
DATASET_ROWS: Dict[str, int] = {
    "breast_cancer": 699,
    "adult": 1200,
    "smartfactory": 1200,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures and helpers
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def real_frames() -> Dict[str, pd.DataFrame]:
    """The three clean tables, row-capped, in deterministic head order."""
    out: Dict[str, pd.DataFrame] = {}
    for name, n in DATASET_ROWS.items():
        path = DATA_ROOT / name / "clean.csv"
        if not path.exists():  # pragma: no cover - datasets ship with the repo
            pytest.skip(f"missing dataset {path}")
        out[name] = pd.read_csv(path).head(n).reset_index(drop=True)
    return out


@pytest.fixture
def toy() -> pd.DataFrame:
    """
    A small frame with one of each column shape.

    ``num`` numeric, ``cat`` textual categorical, ``ident`` an ID-like
    column (60 unique of 60, above the 50/0.9 thresholds), ``const``
    constant.
    """
    n = 60
    return pd.DataFrame(
        {
            "num": [1 + (i % 10) for i in range(n)],
            "cat": [["alpha", "beta", "gamma"][i % 3] for i in range(n)],
            "ident": list(range(1000, 1000 + n)),
            "const": ["same"] * n,
        }
    )


def positions_from_labels(labels: pd.DataFrame) -> Set[Tuple[int, str]]:
    """
    Independently derive ``(row_position, column)`` pairs from a boolean
    label frame. Deliberately written with plain loops rather than the
    injector's own bookkeeping.
    """
    out: Set[Tuple[int, str]] = set()
    for col in labels.columns:
        values = labels[col].to_numpy()
        for i in range(values.shape[0]):
            if bool(values[i]):
                out.add((i, col))
    return out


def parses_as_number(value: Any) -> bool:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(out)


def column_value_set(series: pd.Series) -> Set[str]:
    """Every non-null value of a column, as canonical strings."""
    out: Set[str] = set()
    for v in series.dropna().to_numpy():
        if parses_as_number(v):
            f = float(v)
            out.add(str(int(round(f))) if abs(f - round(f)) <= 1e-9 else repr(f))
        else:
            out.add(str(v))
    return out


def canonical(value: Any) -> str:
    if parses_as_number(value):
        f = float(value)
        return str(int(round(f))) if abs(f - round(f)) <= 1e-9 else repr(f)
    return str(value)


ALL_CASES = [
    (dataset, ctype)
    for dataset in DATASET_ROWS
    for ctype in CORRUPTION_TYPES
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Round trip: the reported positions ARE the changed cells
# ═══════════════════════════════════════════════════════════════════════════════


class TestRoundTrip:
    """
    ``cell_labels(clean, corrupted)`` must recover exactly the injected
    positions — no more, no fewer. This is the single most important property
    in the module: every detection metric in the study is computed against
    these labels.
    """

    @pytest.mark.parametrize("dataset,ctype", ALL_CASES)
    def test_cell_labels_recovers_exactly_the_injected_positions(
        self, real_frames, dataset, ctype
    ):
        clean = real_frames[dataset]
        corrupted, report = inject(clean, ctype, rate=0.01, seed=42)

        labels = cell_labels(clean, corrupted)
        derived = positions_from_labels(labels)
        reported = set(report.positions)

        assert derived == reported, (
            f"{dataset}/{ctype}: changed-but-unreported="
            f"{sorted(derived - reported)[:5]}, "
            f"reported-but-unchanged={sorted(reported - derived)[:5]}"
        )
        assert len(reported) == report.n_injected

    @pytest.mark.parametrize("dataset,ctype", ALL_CASES)
    def test_no_cell_outside_an_eligible_column_is_touched(
        self, real_frames, dataset, ctype
    ):
        clean = real_frames[dataset]
        corrupted, report = inject(clean, ctype, rate=0.05, seed=42)
        labels = cell_labels(clean, corrupted)
        for col in clean.columns:
            if col in report.eligible_columns:
                continue
            assert not labels[col].any(), (
                f"{dataset}/{ctype}: column {col!r} was skipped "
                f"({report.skipped_columns.get(col)!r}) but changed"
            )

    def test_an_unreported_change_would_fail_this_test(self, toy):
        """
        Guard on the guard: tamper with the corrupted frame and confirm the
        round-trip assertion actually detects it. Without this, a vacuous
        round-trip check could pass forever.
        """
        corrupted, report = inject(toy, "swap_valid", rate=0.05, seed=42)
        untouched_row = next(
            i for i in range(len(toy)) if (i, "cat") not in report.positions
        )
        corrupted = corrupted.copy()
        corrupted.loc[untouched_row, "cat"] = "SNEAKY"
        derived = positions_from_labels(cell_labels(toy, corrupted))
        assert derived != set(report.positions)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Rate
# ═══════════════════════════════════════════════════════════════════════════════


class TestRate:
    @pytest.mark.parametrize("rate", [0.001, 0.005, 0.01, 0.05, 0.10])
    @pytest.mark.parametrize("dataset", list(DATASET_ROWS))
    def test_requested_rate_is_achieved_when_the_pool_allows_it(
        self, real_frames, dataset, rate
    ):
        """
        With a pool larger than the request, the injected count must equal
        ``round(rate * n_cells)`` exactly — not approximately. The achieved
        rate then matches the requested rate to within one cell.
        """
        clean = real_frames[dataset]
        corrupted, report = inject(clean, "missing", rate=rate, seed=42)
        expected = int(round(rate * clean.size))
        assert report.n_eligible_cells >= expected
        assert report.n_injected == expected
        tolerance = 1.0 / clean.size
        assert abs(report.achieved_rate - rate) <= tolerance
        assert report.rate_shortfall <= tolerance

    @pytest.mark.parametrize("ctype", list(CORRUPTION_TYPES))
    def test_rate_zero_returns_the_frame_unchanged(self, real_frames, ctype):
        clean = real_frames["adult"]
        corrupted, report = inject(clean, ctype, rate=0.0, seed=42)
        assert corrupted.equals(clean)
        assert report.n_injected == 0
        assert report.positions == ()
        assert report.achieved_rate == 0.0
        assert report.is_empty
        assert not cell_labels(clean, corrupted).to_numpy().any()

    def test_unreachable_rate_is_reported_not_silently_rounded(self, toy):
        """
        ``format`` is only eligible for the one numeric column, so a 90 %
        rate cannot be met. The report must say so and the achieved rate must
        be the honest smaller number.
        """
        corrupted, report = inject(toy, "format", rate=0.9, seed=42)
        assert report.n_requested > report.n_eligible_cells
        assert report.n_injected == report.n_eligible_cells
        assert report.achieved_rate < report.requested_rate
        assert report.rate_shortfall > 0
        assert report.notes, "an unreachable rate must leave a note"

    def test_no_eligible_column_yields_zero_and_a_note(self, real_frames):
        """
        Smart Factory has no textual categorical column, so ``typo`` has
        nothing to act on. That must surface as 0 injected plus a stated
        reason, never as a silent success.
        """
        clean = real_frames["smartfactory"]
        corrupted, report = inject(clean, "typo", rate=0.01, seed=42)
        assert report.eligible_columns == ()
        assert report.n_injected == 0
        assert report.notes
        assert corrupted.equals(clean)
        assert set(report.skipped_columns) == set(clean.columns)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Per-type value guarantees
# ═══════════════════════════════════════════════════════════════════════════════


class TestSwapValidIsUndetectableByConstruction:
    """
    The honesty anchor. ``swap_valid`` must only ever write a value that
    already occurs in that same column, which is precisely why no
    vocabulary, range, format or missing-token rule can see it.
    """

    @pytest.mark.parametrize("dataset", list(DATASET_ROWS))
    def test_every_written_value_already_exists_in_its_column(
        self, real_frames, dataset
    ):
        clean = real_frames[dataset]
        corrupted, report = inject(clean, "swap_valid", rate=0.05, seed=42)
        assert report.n_injected > 0
        for col in set(c for _, c in report.positions):
            allowed = column_value_set(clean[col])
            for row, c in report.positions:
                if c != col:
                    continue
                written = corrupted[col].iat[row]
                assert canonical(written) in allowed, (
                    f"{dataset}: swap_valid wrote {written!r} into {col!r}, "
                    f"which is not an existing value of that column"
                )

    @pytest.mark.parametrize("dataset", list(DATASET_ROWS))
    def test_the_written_value_is_never_the_original(self, real_frames, dataset):
        clean = real_frames[dataset]
        corrupted, report = inject(clean, "swap_valid", rate=0.05, seed=42)
        for row, col in report.positions:
            assert canonical(corrupted[col].iat[row]) != canonical(
                clean[col].iat[row]
            )

    def test_swap_valid_introduces_no_new_value_anywhere(self, real_frames):
        """
        Stronger, column-set form: the corrupted frame's value set per column
        must be a subset of the clean frame's.
        """
        clean = real_frames["adult"]
        corrupted, _ = inject(clean, "swap_valid", rate=0.10, seed=42)
        for col in clean.columns:
            assert column_value_set(corrupted[col]) <= column_value_set(clean[col])

    def test_swap_valid_adds_no_missing_values(self, real_frames):
        clean = real_frames["adult"]
        corrupted, _ = inject(clean, "swap_valid", rate=0.10, seed=42)
        assert int(corrupted.isna().sum().sum()) == int(clean.isna().sum().sum())


class TestFormat:
    @pytest.mark.parametrize("dataset", list(DATASET_ROWS))
    def test_every_injected_value_is_unparseable_as_a_number(
        self, real_frames, dataset
    ):
        clean = real_frames[dataset]
        corrupted, report = inject(clean, "format", rate=0.05, seed=42)
        assert report.n_injected > 0
        for row, col in report.positions:
            value = corrupted[col].iat[row]
            assert isinstance(value, str)
            assert not parses_as_number(value), (
                f"{dataset}: format injected {value!r}, which still parses "
                f"as a number"
            )

    def test_injected_values_are_not_null_tokens(self, real_frames):
        """
        A format error that reads as MISSING would make the two types
        non-disjoint and the per-type tables meaningless.
        """
        from src.detection.detector import _NULL_STRINGS

        clean = real_frames["adult"]
        corrupted, report = inject(clean, "format", rate=0.05, seed=42)
        for row, col in report.positions:
            value = str(corrupted[col].iat[row]).strip().lower()
            assert value not in _NULL_STRINGS
            assert "?" not in value

    def test_format_only_targets_numeric_columns(self, real_frames):
        clean = real_frames["adult"]
        _, report = inject(clean, "format", rate=0.05, seed=42)
        roles = classify_columns(clean)
        for col in report.eligible_columns:
            assert roles[col] is ColumnRole.NUMERIC


class TestMissing:
    @pytest.mark.parametrize("dataset", list(DATASET_ROWS))
    def test_every_injected_cell_is_null(self, real_frames, dataset):
        clean = real_frames[dataset]
        corrupted, report = inject(clean, "missing", rate=0.05, seed=42)
        assert report.n_injected > 0
        for row, col in report.positions:
            assert pd.isna(corrupted[col].iat[row])
        n_new = int(corrupted.isna().sum().sum()) - int(clean.isna().sum().sum())
        assert n_new == report.n_injected


class TestOutOfRange:
    @pytest.mark.parametrize("dataset", list(DATASET_ROWS))
    def test_injected_values_are_far_outside_the_observed_range(
        self, real_frames, dataset
    ):
        """
        Re-derives the guarantee from the clean column's own min/max: every
        injected value must sit at least ``min(OUT_OF_RANGE_MULTIPLIERS)``
        range-spans outside ``[min, max]``. This is what keeps the injector
        from being tuned to the detector's ``Q3 + 3*IQR`` fence.
        """
        clean = real_frames[dataset]
        corrupted, report = inject(clean, "out_of_range", rate=0.05, seed=42)
        assert report.n_injected > 0
        smallest = min(OUT_OF_RANGE_MULTIPLIERS)
        for row, col in report.positions:
            num = pd.to_numeric(clean[col], errors="coerce").dropna()
            lo, hi = float(num.min()), float(num.max())
            span = hi - lo if hi > lo else max(abs(hi), 1.0)
            value = float(corrupted[col].iat[row])
            distance = max(lo - value, value - hi)
            assert distance >= smallest * span - 1e-6, (
                f"{dataset}/{col}: injected {value} is only {distance} "
                f"outside [{lo}, {hi}] (span {span})"
            )

    def test_injected_values_are_also_outside_the_detector_iqr_fence(
        self, real_frames
    ):
        """
        Independent check with the detector's own fence at its shipped
        k = 3: the injected value must be far beyond it, so the fence is not
        being handed a value placed just past it.
        """
        clean = real_frames["adult"]
        corrupted, report = inject(clean, "out_of_range", rate=0.05, seed=42)
        for row, col in report.positions:
            num = pd.to_numeric(clean[col], errors="coerce").dropna()
            q1, q3 = float(num.quantile(0.25)), float(num.quantile(0.75))
            iqr = q3 - q1
            value = float(corrupted[col].iat[row])
            assert value < q1 - 3.0 * iqr or value > q3 + 3.0 * iqr


class TestSwapConditional:
    @pytest.mark.parametrize("dataset", list(DATASET_ROWS))
    def test_written_values_are_globally_valid(self, real_frames, dataset):
        """
        ``swap_conditional`` must be as globally innocuous as
        ``swap_valid``: the value exists in the column. Its improbability is
        purely conditional.
        """
        clean = real_frames[dataset]
        corrupted, report = inject(clean, "swap_conditional", rate=0.05, seed=42)
        assert report.n_injected > 0
        for row, col in report.positions:
            allowed = column_value_set(clean[col])
            assert canonical(corrupted[col].iat[row]) in allowed

    def test_numeric_values_stay_inside_the_tightest_tukey_fence(
        self, real_frames
    ):
        """
        The design claim is that no fence can see ``swap_conditional``. Check
        it against the strictest fence anyone uses (k = 1.5), which is
        tighter than the detector's shipped k = 3.
        """
        clean = real_frames["smartfactory"]
        corrupted, report = inject(clean, "swap_conditional", rate=0.05, seed=42)
        assert report.n_injected > 0
        for row, col in report.positions:
            num = pd.to_numeric(clean[col], errors="coerce").dropna()
            q1, q3 = float(num.quantile(0.25)), float(num.quantile(0.75))
            iqr = q3 - q1
            value = float(corrupted[col].iat[row])
            assert q1 - 1.5 * iqr <= value <= q3 + 1.5 * iqr

    def test_a_conditioning_column_is_recorded_for_every_target(
        self, real_frames
    ):
        clean = real_frames["adult"]
        _, report = inject(clean, "swap_conditional", rate=0.01, seed=42)
        for col in set(c for _, c in report.positions):
            assert col in report.conditioning_columns
            assert report.conditioning_columns[col] != col

    def test_conditional_injection_is_genuinely_conditional(self, real_frames):
        """
        Non-sycophantic: re-derive the improbability. For a categorical
        target, count how often each injected value co-occurs with the
        target row's conditioning bin in the CLEAN table. The injector claims
        to prefer zero-co-occurrence values, so the vast majority of injected
        categorical cells must have count 0 — and crucially, the same cells'
        *global* frequency must not be unusually low (that is what separates
        this type from marginal rarity).
        """
        from src.datagen.corruptor import _discretise

        clean = real_frames["adult"]
        corrupted, report = inject(clean, "swap_conditional", rate=0.02, seed=42)
        roles = classify_columns(clean)
        zero_cond = 0
        total = 0
        global_ranks: List[float] = []
        for row, col in report.positions:
            if roles[col] is not ColumnRole.TEXT:
                continue
            cond_col = report.conditioning_columns[col]
            codes_t = _discretise(clean[col], roles[col])
            codes_c = _discretise(clean[cond_col], roles[cond_col])
            group = clean[col][codes_c == codes_c[row]].dropna()
            written = corrupted[col].iat[row]
            total += 1
            if int((group == written).sum()) == 0:
                zero_cond += 1
            vc = clean[col].value_counts(normalize=True)
            global_ranks.append(float(vc.get(written, 0.0)))
        assert total > 0
        assert zero_cond / total >= 0.5, (
            f"only {zero_cond}/{total} categorical swap_conditional cells "
            f"were conditionally impossible"
        )
        # The injected values are globally ordinary, not globally rare: the
        # median injected value's marginal share beats a 1-in-1000 cell.
        assert float(np.median(global_ranks)) > 0.001


class TestTypo:
    def test_injected_values_are_not_valid_categories(self, real_frames):
        """
        A character-level perturbation that lands on another legitimate
        category is a ``swap_valid``, not a typo. If that leaked through, the
        two types would not be disjoint.
        """
        clean = real_frames["adult"]
        corrupted, report = inject(clean, "typo", rate=0.05, seed=42)
        assert report.n_injected > 0
        for row, col in report.positions:
            allowed = column_value_set(clean[col])
            written = corrupted[col].iat[row]
            assert canonical(written) not in allowed
            assert isinstance(written, str)
            assert written != str(clean[col].iat[row])

    def test_typo_only_targets_textual_columns(self, real_frames):
        clean = real_frames["adult"]
        _, report = inject(clean, "typo", rate=0.01, seed=42)
        roles = classify_columns(clean)
        assert report.eligible_columns
        for col in report.eligible_columns:
            assert roles[col] is ColumnRole.TEXT

    def test_typos_stay_close_to_the_original(self, real_frames):
        """A one-character edit, not a random string."""
        clean = real_frames["adult"]
        corrupted, report = inject(clean, "typo", rate=0.02, seed=42)
        for row, col in report.positions:
            before = str(clean[col].iat[row])
            after = str(corrupted[col].iat[row])
            assert abs(len(after) - len(before)) <= 1


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Determinism and frame structure
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    @pytest.mark.parametrize("dataset,ctype", ALL_CASES)
    def test_same_seed_gives_a_bit_identical_frame(
        self, real_frames, dataset, ctype
    ):
        clean = real_frames[dataset]
        a, ra = inject(clean, ctype, rate=0.02, seed=42)
        b, rb = inject(clean, ctype, rate=0.02, seed=42)
        assert a.equals(b)
        assert ra.positions == rb.positions
        assert ra.to_dict() == rb.to_dict()

    @pytest.mark.parametrize("ctype", list(CORRUPTION_TYPES))
    def test_a_different_seed_gives_different_positions(
        self, real_frames, ctype
    ):
        """
        Without this, a constant-output bug would look like determinism.
        Smart Factory / Breast Cancer have no typo-eligible column, so the
        adult frame is used, where every type is live.
        """
        clean = real_frames["adult"]
        _, ra = inject(clean, ctype, rate=0.02, seed=42)
        _, rb = inject(clean, ctype, rate=0.02, seed=43)
        assert set(ra.positions) != set(rb.positions)
        assert ra.n_injected == rb.n_injected

    def test_positions_do_not_depend_on_the_rate_of_other_runs(
        self, real_frames
    ):
        """
        Per-cell value seeding means a cell's injected value is a function of
        (seed, type, column, row) alone. So a cell injected at both 1 % and
        5 % must receive the *same* value in both runs.
        """
        clean = real_frames["adult"]
        small, rs = inject(clean, "out_of_range", rate=0.01, seed=42)
        large, rl = inject(clean, "out_of_range", rate=0.05, seed=42)
        shared = set(rs.positions) & set(rl.positions)
        assert shared, "the two rates should overlap on some cells"
        for row, col in shared:
            assert small[col].iat[row] == large[col].iat[row]

    def test_clean_frame_is_never_mutated(self, real_frames):
        clean = real_frames["adult"]
        before = clean.copy(deep=True)
        for ctype in CORRUPTION_TYPES:
            inject(clean, ctype, rate=0.05, seed=42)
        assert clean.equals(before)


class TestFrameStructure:
    @pytest.mark.parametrize("dataset,ctype", ALL_CASES)
    def test_shape_columns_and_index_are_preserved(
        self, real_frames, dataset, ctype
    ):
        clean = real_frames[dataset]
        corrupted, _ = inject(clean, ctype, rate=0.05, seed=42)
        assert corrupted.shape == clean.shape
        assert list(corrupted.columns) == list(clean.columns)
        assert corrupted.index.equals(clean.index)

    @pytest.mark.parametrize("ctype", ["swap_valid", "swap_conditional"])
    @pytest.mark.parametrize("dataset", list(DATASET_ROWS))
    def test_in_domain_types_preserve_every_dtype(
        self, real_frames, dataset, ctype
    ):
        """
        ``swap_valid`` and ``swap_conditional`` write values taken from the
        column itself, so there is no excuse for a dtype change. The other
        types change dtypes unavoidably (NaN upcasts, strings force object),
        which the module docstring states.
        """
        clean = real_frames[dataset]
        corrupted, _ = inject(clean, ctype, rate=0.05, seed=42)
        changed = {
            c: (str(clean[c].dtype), str(corrupted[c].dtype))
            for c in clean.columns
            if clean[c].dtype != corrupted[c].dtype
        }
        assert changed == {}, f"{dataset}/{ctype} changed dtypes: {changed}"

    def test_missing_upcasts_integers_exactly_as_the_loader_documents(
        self, real_frames
    ):
        clean = real_frames["smartfactory"]
        corrupted, report = inject(clean, "missing", rate=0.05, seed=42)
        touched = set(report.per_column)
        for col in touched:
            assert str(clean[col].dtype) == "int64"
            assert str(corrupted[col].dtype) == "float64"
        for col in set(clean.columns) - touched:
            assert clean[col].dtype == corrupted[col].dtype


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Column classification
# ═══════════════════════════════════════════════════════════════════════════════


class TestColumnClassification:
    def test_toy_roles(self, toy):
        roles = classify_columns(toy)
        assert roles["num"] is ColumnRole.NUMERIC
        assert roles["cat"] is ColumnRole.TEXT
        assert roles["ident"] is ColumnRole.ID
        assert roles["const"] is ColumnRole.UNUSABLE

    def test_id_like_columns_are_never_corrupted(self, toy):
        for ctype in CORRUPTION_TYPES:
            corrupted, report = inject(toy, ctype, rate=0.2, seed=42)
            assert "ident" not in report.eligible_columns
            assert report.skipped_columns["ident"].startswith("id-like")
            assert corrupted["ident"].equals(toy["ident"])

    def test_breast_cancer_identifier_columns_are_skipped(self, real_frames):
        """
        Concrete, hard-coded expectation: the two Breast Cancer key columns
        must be excluded, and ``Bare Nuclei`` (97.7 % numeric-parseable
        despite being ``object``) must be treated as numeric.
        """
        clean = real_frames["breast_cancer"]
        roles = classify_columns(clean)
        assert roles["Unnamed: 0"] is ColumnRole.ID
        assert roles["Sample code number"] is ColumnRole.ID
        assert roles["Bare Nuclei"] is ColumnRole.NUMERIC

    def test_adult_text_columns_are_text(self, real_frames):
        clean = real_frames["adult"]
        roles = classify_columns(clean)
        for col in (
            "workclass", "education", "marital_status", "occupation",
            "relationship", "race", "gender", "native_country",
        ):
            assert roles[col] is ColumnRole.TEXT
        for col in ("age", "educational_num", "hours_per_week", "income"):
            assert roles[col] is ColumnRole.NUMERIC

    def test_protected_columns_are_honoured(self, real_frames):
        clean = real_frames["adult"]
        corrupted, report = inject(
            clean, "missing", rate=0.05, seed=42,
            protected_columns=("age", "income"),
        )
        assert "age" not in report.eligible_columns
        assert "income" not in report.eligible_columns
        assert report.skipped_columns["age"] == "protected by caller"
        assert corrupted["age"].equals(clean["age"])
        assert corrupted["income"].equals(clean["income"])

    def test_every_column_is_either_eligible_or_has_a_reason(self, real_frames):
        for dataset, clean in real_frames.items():
            for ctype in CORRUPTION_TYPES:
                eligible, skipped = eligible_columns(clean, ctype)
                assert set(eligible) | set(skipped) == set(clean.columns)
                assert not (set(eligible) & set(skipped))
                assert all(bool(r) for r in skipped.values())


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Helper: normalised mutual information
# ═══════════════════════════════════════════════════════════════════════════════


class TestNormalisedMutualInformation:
    """Checked against analytically known values, not against itself."""

    def test_identical_columns_give_one(self):
        a = np.array([0, 0, 1, 1, 2, 2])
        assert _normalised_mutual_information(a, a.copy()) == pytest.approx(1.0)

    def test_a_relabelling_is_still_one(self):
        a = np.array([0, 0, 1, 1, 2, 2])
        b = np.array([5, 5, 7, 7, 9, 9])
        assert _normalised_mutual_information(a, b) == pytest.approx(1.0)

    def test_independent_columns_give_approximately_zero(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 4, size=20000)
        b = rng.integers(0, 4, size=20000)
        assert _normalised_mutual_information(a, b) < 0.01

    def test_a_constant_column_gives_zero(self):
        a = np.array([1, 1, 1, 1])
        b = np.array([0, 1, 0, 1])
        assert _normalised_mutual_information(a, b) == 0.0

    def test_partial_dependence_is_between(self):
        a = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        b = np.array([0, 0, 0, 1, 1, 1, 1, 0])
        score = _normalised_mutual_information(a, b)
        assert 0.0 < score < 1.0

    def test_nulls_are_excluded(self):
        a = np.array([-1, 0, 0, 1, 1])
        b = np.array([7, 0, 0, 1, 1])
        assert _normalised_mutual_information(a, b) == pytest.approx(1.0)

    def test_it_is_symmetric(self):
        rng = np.random.default_rng(3)
        a = rng.integers(0, 5, size=500)
        b = (a + rng.integers(0, 2, size=500)) % 5
        assert _normalised_mutual_information(a, b) == pytest.approx(
            _normalised_mutual_information(b, a)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Validation and reporting
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidation:
    def test_unknown_corruption_type_raises(self, toy):
        with pytest.raises(ValueError, match="corruption_type"):
            Corruptor("scramble", 0.1)
        with pytest.raises(ValueError, match="corruption_type"):
            eligible_columns(toy, "scramble")

    @pytest.mark.parametrize("rate", [-0.01, 1.01, 2.0])
    def test_rate_outside_the_unit_interval_raises(self, rate):
        with pytest.raises(ValueError, match="rate"):
            Corruptor("missing", rate)

    def test_negative_seed_raises(self):
        with pytest.raises(ValueError, match="seed"):
            Corruptor("missing", 0.1, seed=-1)

    def test_non_positive_fence_raises(self):
        with pytest.raises(ValueError, match="conditional_fence_k"):
            Corruptor("missing", 0.1, conditional_fence_k=0.0)

    def test_empty_frame_raises(self):
        with pytest.raises(ValueError, match="empty"):
            Corruptor("missing", 0.1).inject(pd.DataFrame())
        with pytest.raises(ValueError, match="empty"):
            Corruptor("missing", 0.1).inject(pd.DataFrame({"a": []}))

    def test_duplicate_column_names_raise(self):
        frame = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])
        with pytest.raises(ValueError, match="duplicate"):
            Corruptor("missing", 0.1).inject(frame)

    def test_already_null_cells_are_not_eligible(self):
        """
        Blanking an already-blank cell is a no-op, and the round-trip check
        would then disagree with the report. Null cells must be excluded from
        the pool for every type.
        """
        frame = pd.DataFrame(
            {"a": [1.0, np.nan, 3.0, np.nan, 5.0, 6.0],
             "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
        )
        corrupted, report = inject(frame, "missing", rate=1.0, seed=42)
        assert report.n_eligible_cells == 10
        assert report.n_injected == 10
        assert set(report.positions) == {
            (i, "a") for i in [0, 2, 4, 5]
        } | {(i, "b") for i in range(6)}


class TestReport:
    def test_to_dict_is_json_safe(self, real_frames):
        import json

        clean = real_frames["breast_cancer"]
        _, report = inject(clean, "out_of_range", rate=0.02, seed=42)
        payload = report.to_dict()
        assert json.loads(json.dumps(payload)) == payload
        assert payload["n_injected"] == report.n_injected
        assert "positions" not in payload

    def test_per_column_counts_sum_to_the_total(self, real_frames):
        clean = real_frames["adult"]
        _, report = inject(clean, "swap_valid", rate=0.03, seed=42)
        assert sum(report.per_column.values()) == report.n_injected
        assert set(report.per_column) <= set(report.eligible_columns)

    def test_positions_are_sorted_and_unique(self, real_frames):
        clean = real_frames["adult"]
        _, report = inject(clean, "typo", rate=0.03, seed=42)
        assert len(set(report.positions)) == len(report.positions)
        order = {c: i for i, c in enumerate(report.eligible_columns)}
        keys = [(order[c], r) for r, c in report.positions]
        assert keys == sorted(keys)

    def test_position_frame_round_trips(self, real_frames):
        clean = real_frames["adult"]
        _, report = inject(clean, "missing", rate=0.01, seed=42)
        frame = report.position_frame()
        assert list(frame.columns) == ["row_id", "column"]
        assert len(frame) == report.n_injected
        assert set(zip(frame["row_id"], frame["column"])) == set(report.positions)

    def test_report_is_immutable(self, real_frames):
        clean = real_frames["breast_cancer"]
        _, report = inject(clean, "missing", rate=0.01, seed=42)
        with pytest.raises(Exception):
            report.n_injected = 0  # type: ignore[misc]
        assert isinstance(report, InjectionReport)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Leakage (CLAUDE.md §3)
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoLeakage:
    def test_the_module_imports_no_detector_repairer_or_evaluator(self):
        """
        Structural guarantee: the injector cannot leak ground truth into a
        detector because it has no way to reach one. Checked by reading the
        module source, so adding such an import fails the suite.
        """
        import src.datagen.corruptor as mod

        source = Path(mod.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "src.detection", "src.detectors", "src.repair",
            "src.evaluation", "src.decision", "src.calibration",
            "src.detection_eval", "src.pipeline",
        ):
            assert f"import {forbidden}" not in source
            assert f"from {forbidden}" not in source

    def test_inject_takes_no_dirty_frame_and_no_labels(self):
        """
        The injector's signature has nowhere to put a dirty frame or a label
        array, which is the structural half of the guarantee that injected
        corruption cannot coincide with real corruption.
        """
        import inspect

        params = set(inspect.signature(Corruptor.inject).parameters)
        assert params == {"self", "clean_df"}
        for bad in ("dirty_df", "labels", "y", "detector"):
            assert bad not in params
