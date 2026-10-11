"""
tests/test_run_experiment.py
----------------------------
Phase 13, Step 2: behavioural tests for ``run_experiment.py``.

WHAT THESE TESTS ARE FOR
========================
The experiment runner is the only thing standing between "a number in the
paper" and "a number someone typed". So the tests check the properties that
make that guarantee real, not the prose around them:

1. **The artifact set is complete.** Every table (md + csv), every figure,
   every predictions file and the metadata file are enumerated explicitly,
   so adding a table without writing it fails here.
2. **Re-running changes nothing.** Two runs with the same flags produce
   byte-identical tables and predictions. ``run_metadata.json`` is compared
   after dropping its ``non_deterministic`` sub-object, which exists
   precisely so the rest of the file can be diffed (the UTC timestamp, the
   wall time and the output path obviously differ).
3. **The CLI refuses what is not implemented.** ``--mode real`` exits 2 and
   names Phase 12; an unknown dataset is an argparse error.
4. **Predictions are one row per decided cell**, with the row count
   recomputed in the test from ``run_pipeline`` + ``decision_indices()``
   rather than read back from the artifact being tested.
5. **The leakage contract is visible in the file.** The only ground-truth
   columns are the two documented ``eval_`` columns, they are grouped last,
   and the header comment says so.
6. **The honest-reporting framing cannot be deleted silently.** Adult's
   cost row must still be accompanied by its undetected-corruption count
   and the exact phrase "zero residual-corruption cost"
   (docs/cost_semantics.md §3).

SPEED
=====
Every invocation uses ``--quick`` (400-record cap, 3 splits). The two
multi-dataset runs are module-scoped fixtures shared by several tests and
marked ``integration``; the cheap CLI-contract tests run standalone.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import run_experiment as rx  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402

SCRIPT = REPO_ROOT / "run_experiment.py"

EXPECTED_TABLE_STEMS = (
    "t1_repair_quality",
    "t2_calibration",
    "t3_risk_aware",
    "t4_policy_comparison",
)
EXPECTED_FIGURE_KINDS = (
    "reliability",
    "confidence",
    "cost_comparison",
    "action_distribution",
    "threshold_sweep",
    "decision_map",
)
ALL_DATASETS = ("adult", "breast_cancer", "smartfactory")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Invoke run_experiment.py as a subprocess, exactly as a user would."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=1800,
    )


def read_csv_rows(path: Path) -> Tuple[List[str], List[List[str]]]:
    """Header and data rows of a CSV, skipping leading ``#`` comment lines."""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        lines = [line for line in fh.read().splitlines() if not line.startswith("#")]
    reader = csv.reader(lines)
    rows = list(reader)
    return rows[0], rows[1:]


def comment_lines(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh if line.startswith("#")]


def expected_artifacts(datasets: Sequence[str], figures: bool) -> List[str]:
    """Every relative path the runner must write, enumerated explicitly."""
    paths = ["run_metadata.json"]
    for stem in EXPECTED_TABLE_STEMS:
        paths.append(f"tables/{stem}.md")
        paths.append(f"tables/{stem}.csv")
    for dataset in datasets:
        paths.append(f"predictions/{dataset}.csv")
        if figures:
            for kind in EXPECTED_FIGURE_KINDS:
                paths.append(f"figures/{dataset}_{kind}.png")
    return paths


def drop_nondeterministic(metadata: Dict) -> Dict:
    out = dict(metadata)
    out.pop("non_deterministic", None)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures: two shared runs (the only expensive things in this file)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def quick_all(tmp_path_factory) -> Tuple[Path, subprocess.CompletedProcess]:
    """``--quick --all`` into a tmp dir, with figures."""
    out = tmp_path_factory.mktemp("quick_all")
    proc = run_cli("--quick", "--all", "--output-dir", str(out))
    return out, proc


@pytest.fixture(scope="module")
def twin_runs(tmp_path_factory) -> Tuple[Path, Path]:
    """
    The same command twice into two directories, for the determinism check.

    One dataset and ``--no-figures`` keeps it cheap; determinism is a
    property of the writers, not of how many datasets they are fed.
    """
    first = tmp_path_factory.mktemp("twin_a")
    second = tmp_path_factory.mktemp("twin_b")
    for out in (first, second):
        proc = run_cli(
            "--quick", "--dataset", "breast_cancer", "--no-figures",
            "--output-dir", str(out),
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
    return first, second


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Artifact set
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestArtifacts:
    def test_exit_code_zero(self, quick_all):
        out, proc = quick_all
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_every_expected_artifact_exists(self, quick_all):
        out, _ = quick_all
        missing = [p for p in expected_artifacts(ALL_DATASETS, figures=True)
                   if not (out / p).exists()]
        assert missing == []

    def test_no_artifact_is_empty(self, quick_all):
        out, _ = quick_all
        empty = [p for p in expected_artifacts(ALL_DATASETS, figures=True)
                 if (out / p).stat().st_size == 0]
        assert empty == []

    def test_no_unexpected_files(self, quick_all):
        """The runner writes exactly the documented set, nothing extra."""
        out, _ = quick_all
        found = sorted(
            p.relative_to(out).as_posix()
            for p in out.rglob("*") if p.is_file()
        )
        assert found == sorted(expected_artifacts(ALL_DATASETS, figures=True))

    def test_table_csvs_have_one_row_per_dataset_or_more(self, quick_all):
        out, _ = quick_all
        for stem in EXPECTED_TABLE_STEMS:
            header, rows = read_csv_rows(out / "tables" / f"{stem}.csv")
            assert header[0] == "dataset", stem
            assert len(rows) >= len(ALL_DATASETS), stem
            assert {r[0] for r in rows} == set(ALL_DATASETS), stem

    def test_stdout_prints_the_tables_and_an_honest_reading(self, quick_all):
        _, proc = quick_all
        assert "T1 — Repair quality" in proc.stdout
        assert "T4 — Policy comparison" in proc.stdout
        assert "Honest reading" in proc.stdout

    def test_quick_banner_is_in_every_table(self, quick_all):
        """A --quick run must announce that it is not paper-grade."""
        out, _ = quick_all
        for stem in EXPECTED_TABLE_STEMS:
            text = (out / "tables" / f"{stem}.md").read_text(encoding="utf-8")
            assert "NOT paper-grade" in text, stem

    def test_cost_units_are_never_called_money(self, quick_all):
        out, _ = quick_all
        for stem in EXPECTED_TABLE_STEMS:
            text = (out / "tables" / f"{stem}.md").read_text(encoding="utf-8")
            assert "$" not in text, stem
            assert "dollar" not in text.lower(), stem


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Determinism
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestDeterminism:
    def test_tables_are_byte_identical(self, twin_runs):
        a, b = twin_runs
        for stem in EXPECTED_TABLE_STEMS:
            for suffix in (".md", ".csv"):
                name = f"tables/{stem}{suffix}"
                assert (a / name).read_bytes() == (b / name).read_bytes(), name

    def test_predictions_are_byte_identical(self, twin_runs):
        a, b = twin_runs
        name = "predictions/breast_cancer.csv"
        assert (a / name).read_bytes() == (b / name).read_bytes()

    def test_metadata_matches_apart_from_the_nondeterministic_block(self, twin_runs):
        a, b = twin_runs
        meta_a = json.loads((a / "run_metadata.json").read_text(encoding="utf-8"))
        meta_b = json.loads((b / "run_metadata.json").read_text(encoding="utf-8"))
        assert drop_nondeterministic(meta_a) == drop_nondeterministic(meta_b)

    def test_the_nondeterministic_block_is_separated_and_nonempty(self, twin_runs):
        a, _ = twin_runs
        meta = json.loads((a / "run_metadata.json").read_text(encoding="utf-8"))
        assert set(meta) == {"deterministic", "non_deterministic"}
        nd = meta["non_deterministic"]
        assert "utc_timestamp" in nd and "wall_time_seconds" in nd


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CLI contract (cheap — no pipeline work)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCli:
    def test_mode_real_exits_2_and_names_phase_12(self, tmp_path):
        proc = run_cli("--mode", "real", "--dataset", "adult",
                       "--output-dir", str(tmp_path))
        assert proc.returncode == 2
        combined = proc.stdout + proc.stderr
        assert "Phase 12" in combined
        assert "llm_repair.py" in combined
        assert "not implemented" in combined.lower()
        # Nothing may be written before the refusal.
        assert list(tmp_path.rglob("*")) == []

    def test_unknown_dataset_exits_nonzero(self, tmp_path):
        proc = run_cli("--dataset", "titanic", "--output-dir", str(tmp_path))
        assert proc.returncode != 0
        assert proc.returncode == 2  # argparse
        assert "invalid choice" in (proc.stdout + proc.stderr)

    def test_dataset_or_all_is_required(self, tmp_path):
        proc = run_cli("--output-dir", str(tmp_path))
        assert proc.returncode == 2

    def test_help_lists_every_documented_flag(self):
        proc = run_cli("--help")
        assert proc.returncode == 0
        for flag in ("--dataset", "--all", "--mode", "--n-records", "--seed",
                     "--splits", "--iqr-k", "--flag-outliers", "--output-dir",
                     "--quick", "--no-figures"):
            assert flag in proc.stdout, flag

    def test_flag_outliers_maps_to_none_true_false(self):
        assert rx.resolve_flag_outliers("auto") is None
        assert rx.resolve_flag_outliers("on") is True
        assert rx.resolve_flag_outliers("off") is False

    def test_quick_constants_are_the_documented_ones(self):
        assert rx.QUICK_N_RECORDS == 400
        assert rx.QUICK_SPLITS == 3


# ═══════════════════════════════════════════════════════════════════════════════
# 4. --no-figures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestNoFigures:
    def test_no_pngs_written(self, twin_runs):
        a, _ = twin_runs
        assert list(a.rglob("*.png")) == []

    def test_tables_and_predictions_still_written(self, twin_runs):
        a, _ = twin_runs
        for path in expected_artifacts(["breast_cancer"], figures=False):
            assert (a / path).exists(), path

    def test_metadata_records_the_skip(self, twin_runs):
        a, _ = twin_runs
        meta = json.loads((a / "run_metadata.json").read_text(encoding="utf-8"))
        det = meta["deterministic"]
        assert det["figures_written"] is False
        assert any("no-figures" in s for s in det["figure_skips"])


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Predictions CSV: one row per decided cell, ground truth grouped last
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestPredictions:
    @pytest.mark.parametrize("dataset", ALL_DATASETS)
    def test_one_row_per_decided_cell(self, quick_all, dataset):
        """
        Recompute the expected row count from the pipeline itself rather than
        trusting anything the runner wrote.
        """
        out, _ = quick_all
        bundle = run_pipeline(
            dataset, n_records=rx.QUICK_N_RECORDS, seed=42, split_seed=42
        )
        expected = len(bundle.decision_indices())
        _, rows = read_csv_rows(out / "predictions" / f"{dataset}.csv")
        assert len(rows) == expected
        assert expected > 0

    def test_required_columns_present_and_ordered(self, quick_all):
        out, _ = quick_all
        header, _ = read_csv_rows(out / "predictions" / "breast_cancer.csv")
        assert header == list(rx.PREDICTION_COLUMNS)
        for required in ("row_id", "column", "detection_method", "dirty_value",
                         "top_candidate", "raw_confidence", "p_repair_correct",
                         "q_cell_corrupted", "expected_cost_auto",
                         "expected_cost_human", "expected_cost_abstain", "action"):
            assert required in header, required

    def test_ground_truth_columns_are_exactly_the_documented_set(self, quick_all):
        out, _ = quick_all
        for dataset in ALL_DATASETS:
            header, _ = read_csv_rows(out / "predictions" / f"{dataset}.csv")
            eval_cols = [c for c in header if c.startswith("eval_")]
            assert eval_cols == list(rx.PREDICTION_EVAL_ONLY_COLUMNS), dataset
            # grouped at the end
            assert header[-len(eval_cols):] == eval_cols, dataset
            # and no other ground-truth column sneaked in
            for banned in ("clean_value", "clean", "is_truly_corrupted"):
                assert banned not in header, (dataset, banned)

    def test_header_comment_names_the_evaluation_only_columns(self, quick_all):
        out, _ = quick_all
        comments = "\n".join(comment_lines(out / "predictions" / "breast_cancer.csv"))
        assert "EVALUATION-ONLY" in comments
        for col in rx.PREDICTION_EVAL_ONLY_COLUMNS:
            assert col in comments, col

    def test_actions_are_valid_and_probabilities_in_range(self, quick_all):
        out, _ = quick_all
        header, rows = read_csv_rows(out / "predictions" / "breast_cancer.csv")
        i_action = header.index("action")
        i_p = header.index("p_repair_correct")
        i_q = header.index("q_cell_corrupted")
        assert {r[i_action] for r in rows} <= {"AUTO_REPAIR", "HUMAN_REVIEW", "ABSTAIN"}
        for r in rows:
            assert 0.0 <= float(r[i_p]) <= 1.0
            assert 0.0 <= float(r[i_q]) <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Metadata contents
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestMetadata:
    def test_git_commit_is_a_sha_or_null(self, quick_all):
        out, _ = quick_all
        det = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))["deterministic"]
        assert "git_commit" in det
        commit = det["git_commit"]
        assert commit is None or (len(commit) == 40 and all(
            c in "0123456789abcdef" for c in commit
        ))

    def test_seeds_and_splits_recorded(self, quick_all):
        out, _ = quick_all
        det = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))["deterministic"]
        assert det["seed"] == 42
        assert det["split_seed_of_representative_run"] == 42
        assert det["n_splits"] == rx.QUICK_SPLITS
        assert det["n_records"] == rx.QUICK_N_RECORDS
        assert det["mode"] == "mock"
        assert det["quick"] is True
        assert det["paper_grade"] is False

    def test_per_dataset_counts_match_the_prediction_files(self, quick_all):
        out, _ = quick_all
        det = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))["deterministic"]
        for dataset in ALL_DATASETS:
            entry = det["per_dataset"][dataset]
            _, rows = read_csv_rows(out / "predictions" / f"{dataset}.csv")
            assert entry["n_prediction_rows"] == len(rows), dataset
            assert entry["n_decided_cells"] == len(rows), dataset
            assert entry["n_detections_total"] > 0, dataset
            assert entry["n_false_negatives"] >= 0, dataset
            assert entry["costs"]["units"] == "experimental penalty units"

    def test_repeated_splits_seeds_recorded(self, quick_all):
        out, _ = quick_all
        det = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))["deterministic"]
        for dataset in ALL_DATASETS:
            rs = det["per_dataset"][dataset]["repeated_splits"]
            assert rs["available"] is True, dataset
            assert rs["n_splits"] == rx.QUICK_SPLITS, dataset
            assert len(rs["split_seeds"]) == rx.QUICK_SPLITS, dataset


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Honest-reporting framing (guards the rules against silent removal)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestHonestReporting:
    def test_adult_cost_row_carries_its_false_negative_count(self, quick_all):
        """
        docs/cost_semantics.md §3: Adult's realised cost must always be
        printed beside the 1,701 corruptions the detector never saw.
        """
        out, _ = quick_all
        for stem in ("t3_risk_aware", "t4_policy_comparison"):
            text = (out / "tables" / f"{stem}.md").read_text(encoding="utf-8")
            assert "adult" in text, stem
            assert "1,701" in text, stem

    def test_the_zero_residual_corruption_phrase_is_present(self, quick_all):
        out, _ = quick_all
        for stem in ("t3_risk_aware", "t4_policy_comparison"):
            text = (out / "tables" / f"{stem}.md").read_text(encoding="utf-8")
            assert "zero residual-corruption cost" in text, stem

    def test_false_negatives_reported_for_every_dataset(self, quick_all):
        out, _ = quick_all
        header, rows = read_csv_rows(out / "tables" / "t1_repair_quality.csv")
        i_fn = header.index("n_false_negatives")
        by_dataset = {r[0]: int(r[i_fn]) for r in rows}
        assert by_dataset["adult"] == 1701
        assert all(v >= 0 for v in by_dataset.values())

    def test_adult_calibration_row_prints_a_dash_not_a_perfect_score(self, quick_all):
        """
        CLAUDE.md §5.4: a calibrator that never fitted must not print
        0.0000, which reads as a perfect score in a results table.
        """
        out, _ = quick_all
        header, rows = read_csv_rows(out / "tables" / "t2_calibration.csv")
        i_method, i_fitted, i_brier = (
            header.index("method"), header.index("n_fitted"), header.index("brier_mean")
        )
        adult_rows = [r for r in rows if r[0] == "adult"]
        assert adult_rows
        for row in adult_rows:
            if row[i_method] != "raw" and row[i_fitted] in ("0", ""):
                assert row[i_brier] == "", row
        md = (out / "tables" / "t2_calibration.md").read_text(encoding="utf-8")
        assert "—" in md

    def test_fallback_rows_are_marked(self, quick_all):
        """Adult cannot fit a p-calibrator; the tables must say so."""
        out, _ = quick_all
        text = (out / "tables" / "t3_risk_aware.md").read_text(encoding="utf-8")
        assert "fallback" in text.lower()

    def test_mock_mode_is_stated(self, quick_all):
        out, _ = quick_all
        text = (out / "tables" / "t1_repair_quality.md").read_text(encoding="utf-8")
        assert "mock_repair.py" in text
        assert "No LLM was called" in text


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Leakage: the runner must not route ground truth into a decision
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoLeakage:
    def test_source_never_passes_clean_df_to_a_decision(self):
        """
        A grep-level guard, in the spirit of the other phases' leakage tests:
        ``clean_df`` may only reach the evaluators.
        """
        source = SCRIPT.read_text(encoding="utf-8")
        for banned in (
            "decide_all(bundle.clean_df",
            "DecisionPolicy(bundle.clean_df",
            "clean_df=bundle.clean_df, costs",
        ):
            assert banned not in source, banned
        # clean_df appears only where an evaluator is constructed.
        for line in source.splitlines():
            if "clean_df" in line and not line.lstrip().startswith("#"):
                assert (
                    "CostEvaluator" in line
                    or "RepairEvaluator" in line
                    or "clean_df``" in line
                    or "`clean_df`" in line
                ), line

    def test_prediction_columns_declare_the_only_ground_truth(self):
        decision = set(rx.PREDICTION_DECISION_COLUMNS)
        evaluation = set(rx.PREDICTION_EVAL_ONLY_COLUMNS)
        assert decision & evaluation == set()
        assert all(c.startswith("eval_") for c in evaluation)
        assert not any(c.startswith("eval_") for c in decision)
