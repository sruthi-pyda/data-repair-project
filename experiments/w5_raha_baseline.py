"""
experiments/w5_raha_baseline.py
-------------------------------
Route **W5.2** of ``docs/research_program.md`` §11 (Gap 1): run a *published*
error-detection system on our data and report where it beats us.

WHY THIS EXISTS
===============
Every detection number this project has ever reported was produced by code we
wrote. ``docs/research_program.md`` §11 names that the biggest remaining gap
("no external benchmark"), and ``docs/column_triage_research.md`` §4 names the
cheapest honest fix: **Raha** (Mahdavi et al., SIGMOD 2019; ACM SIGMOD *Most
Reproducible Paper* Award), whose benchmarking API takes a
``{name, path, clean_path}`` dictionary — our ``datasets/<name>/{dirty,clean}.csv``
layout verbatim.

Raha pins ``pandas==2.1.0`` in its repo requirements while this project runs on
pandas 2.3, so it lives in its own virtualenv (``.venv-raha/``). That is why
this script has **two stages that run under two different interpreters**.

    # stage 1 — under the raha venv (no project imports)
    .venv-raha/Scripts/python.exe experiments/w5_raha_baseline.py --stage raha

    # stage 2 — under the project venv (our metrics)
    python experiments/w5_raha_baseline.py --stage score

Stage 1 writes a JSON artifact (flag set + Raha's own P/R/F1 + wall time);
stage 2 reads it, re-scores it under *our* metric definitions, and writes
``results/w5_raha_comparison.md``.

WHAT "20 LABELS" MEANS — STATE THIS WHEREVER THE NUMBERS ARE QUOTED
===================================================================
``raha.Detection.LABELING_BUDGET`` defaults to 20. Those labels are **simulated
from clean.csv**: Raha only enters its labelling loop when
``dataset.has_ground_truth`` is true, and the simulated user has
``USER_LABELING_ACCURACY = 1.0``. This is the accepted Raha/ED2/SAGED
benchmarking protocol (``docs/literature.md`` R-3) and it is *not* a leak on
our side — but it does mean the comparison is **20 oracle labels (Raha) vs
zero labels (our label-free detectors)**, and Raha's labels are drawn from the
whole table rather than from a train split, so its numbers are in-sample with
respect to the label budget in a way ours are not. Both facts go in the
write-up.

Our own leakage invariant (CLAUDE.md §3) is untouched: ``clean_df`` enters this
file only through :func:`src.detection_eval.labels.cell_labels`, and nothing
here hands a label to one of *our* detectors.

CELL ALIGNMENT — THE ONE THING THAT COULD SILENTLY FAKE A RESULT
================================================================
Raha returns ``{(row_index, column_index): "JUST A DUMMY VALUE"}`` against a
frame *it* read with ``pandas.read_csv(..., dtype=str, keep_default_na=False)``.
If its row indexing were offset by one (header counted as a row), or its column
order differed, the flag set would land on the wrong cells and still produce a
plausible-looking table.

So stage 1 dumps, for a deterministic sample of flagged cells, **the literal
string Raha saw in that cell**, and stage 2 asserts it matches
``str(dirty_df.iat[row, col])`` after the same normalisation. A mis-alignment
fails the run instead of printing a number. Stage 1 also dumps Raha's frame
shape and column list, which stage 2 checks against ours.

Stage 1 copies ``dirty.csv``/``clean.csv`` into a scratch working directory
before pointing Raha at them, because ``raha.Detection`` writes a
``raha-baran-results-<name>/`` folder next to the dataset path and we do not
want that inside ``datasets/``. The copies are byte-identical, so positional
alignment is preserved.

A FLAG SET IS NOT A RANKING
===========================
Raha emits a *set*: two distinct scores (1.0 flagged, 0.0 unflagged). PR-AUC
scores a *ranking*, so Raha's PR-AUC is measured on an axis it does not
attempt — exactly the caveat ``docs/metrics_definitions.md`` §3 applies to our
own six-constant rule detector. It is reported for completeness, with the
distinct-score count printed beside it, and the operating-point P/R/F1 is the
number to read for Raha.

THE COMPARISON IS FOUR DETECTORS ON ONE CELL UNIVERSE
=====================================================
``raha`` (20 oracle labels) against three of ours (zero labels):

    rule                  the shipped six-rule Phase 3 detector
    conditional_full      1 - P(value | rest of row), out of fold over rows
    ensemble_noisy_or_b0  1 - (1 - rule)(1 - conditional), zero labels

Those three are built by importing ``experiments.w4_metrics.build_detectors``
**unchanged**, so the rows here are the same objects W4.2 measured and no
number is transcribed by hand from another document.

All four go through :func:`src.detection_eval.harness.evaluate_detector` with
the project defaults (5 stratified cell splits, seed 42, 30 % eval,
95 % bootstrap CI), and the operating-point table is computed on **all** cells,
as ``docs/metrics_definitions.md`` §8 permits for detectors that do not fit on
labels. Raha *does* consume 20 labels, so its full-dataset row is flagged
in-sample in the output.

Three operating points per detector, matching ``results/w4_metrics.md``:

    own flag set      s > 0                       (Raha: its literal flags)
    equal budget      top-k, k = Raha's flag count  (same review cost)
    oracle best-F1    the F1-maximising threshold   (upper bound, labelled)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

DATASETS = ("breast_cancer", "smartfactory", "adult")
DEFAULT_BUDGET = 20
SEED = 42

# Deterministic sample of flagged cells whose literal value is round-tripped
# through the JSON so stage 2 can prove the two frames are the same frame.
N_ALIGNMENT_PROBES = 25


TEMPSHIM_ENV = "W5_RAHA_TEMPSHIM"


def _install_temp_shim() -> None:
    """
    Tolerate ``PermissionError`` from ``os.remove`` on a temp file — a Windows
    portability bug in dBoost that otherwise kills every Raha run.

    MEASURED CAUSE, not a guess. ``raha.detection._strategy_runner_process``
    runs each ``OD`` (dBoost) strategy by writing the table to
    ``%TEMP%/<name>-<hash>.csv``, calling
    ``raha.tools.dBoost.dboost.imported_dboost.run``, reading the output and
    then ``os.remove``-ing both files. dBoost's ``cli.py`` takes its input via
    ``argparse.FileType('r')`` and **never closes the handle**. On POSIX,
    unlinking an open file is legal; on Windows it raises::

        PermissionError: [Errno 13] The process cannot access the file because
        it is being used by another process:
        '...\\Temp\\breast_cancer-2183489894....csv'

    The exception propagates out of ``pool.map`` and aborts the whole run.
    Observed on all three datasets before this shim.

    WHAT THE SHIM CHANGES: nothing that is measured. It only lets a temp-file
    *deletion* fail silently, after dBoost's output has already been read back
    into ``outputted_cells``. No detection decision, feature, label or score
    depends on whether the scratch file is unlinked. Leftover files stay in
    ``%TEMP%``; the count of skipped deletions is reported.

    It is installed at module import when ``W5_RAHA_TEMPSHIM=1``, because on
    Windows ``multiprocessing`` spawns fresh interpreters that re-import this
    module as ``__mp_main__`` — a patch applied only inside ``main()`` would
    not reach the pool workers where the failure actually happens.
    """
    import errno

    real_remove = os.remove
    if getattr(real_remove, "_w5_shimmed", False):
        return

    def tolerant_remove(path, *a, **kw):  # type: ignore[no-untyped-def]
        try:
            return real_remove(path, *a, **kw)
        except PermissionError as exc:
            if getattr(exc, "errno", None) not in (errno.EACCES, errno.EPERM):
                raise
            sys.stderr.write(
                f"[w5-tempshim] could not unlink {path!r} "
                "(dBoost leaves its input handle open on Windows); continuing.\n"
            )
            return None

    tolerant_remove._w5_shimmed = True  # type: ignore[attr-defined]
    os.remove = tolerant_remove  # type: ignore[assignment]


if os.environ.get(TEMPSHIM_ENV) == "1":
    _install_temp_shim()


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_json_path() -> str:
    """
    Where the stage-1 artifact lands.

    Default is a scratch directory, not ``results/``: the only file this route
    is allowed to add under ``results/`` is the markdown write-up. Override
    with ``--flags-json`` if you want to keep the raw artifact.
    """
    env = os.environ.get("W5_RAHA_JSON")
    if env:
        return env
    return os.path.join(_repo_root(), ".venv-raha", "w5_raha_flags.json")


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 — run Raha (under .venv-raha; imports raha, nothing of ours)
# ═══════════════════════════════════════════════════════════════════════════════


def run_raha_one(
    dataset: str,
    repo_root: str,
    work_root: str,
    budget: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run ``raha.Detection`` on one dataset and return a JSON-serialisable record.

    Returns a record with ``status == "ok"`` and the flag set, or
    ``status == "failed"`` with the exact exception text. A documented failure
    is a result; a silently dropped dataset is not (CLAUDE.md §8b.8).
    """
    import traceback

    import pandas  # raha's own dependency, present in its venv
    import raha

    src_dir = os.path.join(repo_root, "datasets", dataset)
    work_dir = os.path.join(work_root, dataset)
    os.makedirs(work_dir, exist_ok=True)
    # Byte-identical copies, so Raha's positional indexing matches ours.
    dirty_path = os.path.join(work_dir, "dirty.csv")
    clean_path = os.path.join(work_dir, "clean.csv")
    shutil.copyfile(os.path.join(src_dir, "dirty.csv"), dirty_path)
    shutil.copyfile(os.path.join(src_dir, "clean.csv"), clean_path)

    dd = {"name": dataset, "path": dirty_path, "clean_path": clean_path}

    record: Dict[str, Any] = {
        "dataset": dataset,
        "labeling_budget": int(budget),
        "raha_version": getattr(raha, "__version__", "unknown"),
        "pandas_version": pandas.__version__,
        "python_version": sys.version.split()[0],
        "windows_tempfile_shim": os.environ.get(TEMPSHIM_ENV) == "1",
    }

    try:
        app = raha.Detection()
        app.LABELING_BUDGET = int(budget)
        app.VERBOSE = bool(verbose)
        # Keep Raha's intermediate artifacts inside the scratch dir, not the repo.
        if hasattr(app, "SAVE_RESULTS"):
            app.SAVE_RESULTS = False
        record["hyperparameters"] = {
            k: _jsonable(getattr(app, k))
            for k in (
                "LABELING_BUDGET",
                "USER_LABELING_ACCURACY",
                "CLUSTERING_BASED_SAMPLING",
                "CLASSIFICATION_MODEL",
                "LABEL_PROPAGATION_METHOD",
                "ERROR_DETECTION_ALGORITHMS",
                "STRATEGY_FILTERING",
                "HISTORICAL_DATASETS",
                "SAVE_RESULTS",
            )
            if hasattr(app, k)
        }

        t0 = time.perf_counter()
        detection_dictionary = app.run(dd)
        wall = time.perf_counter() - t0

        data = raha.Dataset(dd)
        evaluation = data.get_data_cleaning_evaluation(detection_dictionary)
        p, r, f1 = (float(x) for x in evaluation[:3])

        frame = data.dataframe
        flags = sorted((int(i), int(j)) for (i, j) in detection_dictionary)

        # Alignment probes: the literal cell contents Raha saw, at a
        # deterministic spread of its own flagged positions.
        probes: List[Dict[str, Any]] = []
        if flags:
            step = max(1, len(flags) // N_ALIGNMENT_PROBES)
            for (i, j) in flags[::step][:N_ALIGNMENT_PROBES]:
                probes.append(
                    {
                        "row": i,
                        "col": j,
                        "column_name": str(frame.columns[j]),
                        "value": str(frame.iat[i, j]),
                    }
                )

        record.update(
            {
                "status": "ok",
                "wall_seconds": float(wall),
                "raha_precision": p,
                "raha_recall": r,
                "raha_f1": f1,
                "n_flagged": len(flags),
                "flags": flags,
                "raha_shape": [int(frame.shape[0]), int(frame.shape[1])],
                "raha_columns": [str(c) for c in frame.columns],
                # Raha's own notion of which cells are wrong, so stage 2 can
                # report how far it is from our ValueComparator labels.
                "raha_n_actual_errors": int(len(data.get_actual_errors_dictionary())),
                "alignment_probes": probes,
            }
        )
    except BaseException as exc:  # noqa: BLE001 - the failure IS the result
        record.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc()[-4000:],
            }
        )
    return record


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


def stage_raha(args: argparse.Namespace) -> int:
    # Install the Windows temp-file shim here AND export the flag, so that the
    # multiprocessing children (which re-import this module) install it too.
    os.environ[TEMPSHIM_ENV] = "1"
    _install_temp_shim()

    repo_root = _repo_root()
    work_root = args.work_dir or os.path.join(repo_root, ".venv-raha", "w5_work")
    os.makedirs(work_root, exist_ok=True)

    out: Dict[str, Any] = {
        "route": "W5.2",
        "stage": "raha",
        "labeling_budget": int(args.budget),
        "seed": SEED,
        "label_simulation": (
            "Raha's LABELING_BUDGET labels are simulated from clean.csv "
            "(dataset.has_ground_truth); USER_LABELING_ACCURACY = 1.0. "
            "Standard Raha/ED2/SAGED benchmarking practice."
        ),
        "records": [],
    }
    for dataset in args.datasets:
        print(f"\n=== raha: {dataset} (budget={args.budget}) ===", flush=True)
        rec = run_raha_one(dataset, repo_root, work_root, args.budget)
        if rec["status"] == "ok":
            print(
                f"  P={rec['raha_precision']:.4f} R={rec['raha_recall']:.4f} "
                f"F1={rec['raha_f1']:.4f}  flagged={rec['n_flagged']}  "
                f"{rec['wall_seconds']:.1f}s",
                flush=True,
            )
        else:
            print(f"  FAILED: {rec['error_type']}: {rec['error']}", flush=True)
        out["records"].append(rec)
        # Write after every dataset so a later crash cannot lose earlier work.
        _write_json(args.flags_json, out)

    print(f"\nwrote {args.flags_json}")
    return 0


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 — re-score under OUR metrics (under the project venv)
# ═══════════════════════════════════════════════════════════════════════════════


def stage_score(args: argparse.Namespace) -> int:
    import numpy as np
    import pandas as pd

    sys.path.insert(0, _repo_root())

    from experiments.w4_metrics import PrecomputedDetector, build_detectors
    from src.data.loader import DataLoader
    from src.data.profiler import profile_dataset
    from src.detection_eval.harness import evaluate_detector, flatten_cells
    from src.detection_eval.labels import cell_labels, label_stats
    from src.detection_eval.metrics import best_f1, evaluate_scores

    with open(args.flags_json, encoding="utf-8") as fh:
        payload = json.load(fh)
    records = {r["dataset"]: r for r in payload["records"]}

    loader = DataLoader("datasets")
    results: List[Dict[str, Any]] = []

    for dataset in args.datasets:
        rec = records.get(dataset)
        if rec is None:
            results.append(
                {
                    "dataset": dataset,
                    "status": "not_run",
                    "error": "no stage-1 record for this dataset",
                }
            )
            continue

        clean_df, dirty_df = loader.load(dataset)
        profile = profile_dataset(
            dataset, loader.discover(dataset), clean_df, dirty_df
        )
        labels = cell_labels(clean_df, dirty_df)  # clean_df's only use
        stats = label_stats(labels)
        y = flatten_cells(labels).astype(bool)
        n_pos = int(y.sum())

        entry: Dict[str, Any] = {
            "dataset": dataset,
            "status": rec["status"],
            "label_stats": {
                k: stats[k] for k in ("n_cells", "n_corrupted", "prevalence")
            },
            "raha": {k: rec.get(k) for k in
                     ("raha_precision", "raha_recall", "raha_f1", "wall_seconds",
                      "n_flagged", "raha_n_actual_errors", "labeling_budget",
                      "error_type", "error")},
            "alignment": None,
            "detectors": {},
            "operating_points": [],
        }

        detectors, notes = build_detectors(dataset, dirty_df, profile, SEED)
        entry["feature_notes"] = list(notes)
        wanted = ["rule", "conditional_full", "ensemble_noisy_or_b0"]
        ours = {k: detectors[k] for k in wanted}

        raha_scores: Optional[np.ndarray] = None
        if rec["status"] == "ok":
            raha_scores, align = _raha_score_vector(rec, dirty_df, np, pd)
            entry["alignment"] = align
            ours["raha"] = PrecomputedDetector(
                raha_scores, dirty_df.index, dirty_df.columns, "raha"
            )

        # k for the equal-budget operating point: Raha's flag count when it
        # ran, else the number of positives (the project default).
        k_budget = int(rec.get("n_flagged") or 0) or n_pos

        for name, det in ours.items():
            print(f"  evaluating {dataset}/{name} ...", flush=True)
            out = evaluate_detector(det, dataset, n_splits=args.splits, seed=SEED)
            entry["detectors"][name] = {
                "pr_auc": out["intervals"]["pr_auc"],
                "roc_auc": out["intervals"]["roc_auc"],
                "best_f1": out["intervals"]["best_f1"],
                "full": out["full"],
                "n_distinct_scores": out["score_structure"]["n_distinct_scores"],
                "labels_used": 20 if name == "raha" else 0,
            }
            s = flatten_cells(det.score_cells(dirty_df, profile)).astype(float)
            entry["operating_points"].extend(
                _operating_points(name, y, s, k_budget, np, best_f1)
            )

        results.append(entry)

    report = {
        "route": "W5.2",
        "budget": payload.get("labeling_budget"),
        "label_simulation": payload.get("label_simulation"),
        "stage1_meta": {
            d: {k: records[d].get(k) for k in
                ("raha_version", "pandas_version", "python_version",
                 "hyperparameters")}
            for d in records
        },
        "splits": args.splits,
        "seed": SEED,
        "results": results,
    }
    md = render_markdown(report)
    out_path = args.out or os.path.join(_repo_root(), "results", "w5_raha_comparison.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"\nwrote {out_path}")
    return 0


def _raha_score_vector(rec, dirty_df, np, pd) -> Tuple[Any, Dict[str, Any]]:
    """
    Turn Raha's ``{(row, col)}`` flag set into a score vector over OUR cells,
    proving the two frames are the same frame on the way.

    flagged = 1.0, unflagged = 0.0. Two distinct scores by construction; the
    caveat is in the module docstring and reproduced in the write-up.

    Raises
    ------
    AssertionError
        Shape, column order, index range or an alignment probe disagrees. A
        mis-aligned flag set would produce a plausible table from wrong cells,
        so this is fatal rather than a warning.
    """
    shape = tuple(rec["raha_shape"])
    assert shape == tuple(dirty_df.shape), (
        f"raha read shape {shape}, we have {tuple(dirty_df.shape)}"
    )
    assert rec["raha_columns"] == [str(c) for c in dirty_df.columns], (
        "raha's column order differs from ours; cell alignment would be wrong"
    )

    n_rows, n_cols = shape
    arr = np.zeros(shape, dtype=float)
    for i, j in rec["flags"]:
        assert 0 <= i < n_rows and 0 <= j < n_cols, f"flag ({i},{j}) out of range"
        arr[i, j] = 1.0

    # The probe check: does the value Raha saw at (i, j) equal the value we
    # have at (i, j)? Compared as normalised strings, because raha reads
    # dtype=str/keep_default_na=False while our loader types columns.
    mismatches = []
    for probe in rec["alignment_probes"]:
        i, j = int(probe["row"]), int(probe["col"])
        assert str(dirty_df.columns[j]) == probe["column_name"], (
            f"column {j} is {dirty_df.columns[j]!r} for us, "
            f"{probe['column_name']!r} for raha"
        )
        ours = _norm_cell(dirty_df.iat[i, j], pd)
        theirs = _norm_cell(probe["value"], pd)
        if ours != theirs:
            mismatches.append(
                {"row": i, "col": j, "ours": ours, "raha": theirs}
            )
    assert not mismatches, (
        f"{len(mismatches)} alignment probe(s) disagree, e.g. {mismatches[:3]}. "
        "Raha's flag set does not land on our cells."
    )

    scores = np.concatenate([arr[:, j] for j in range(n_cols)])  # column-major
    return scores, {
        "n_probes": len(rec["alignment_probes"]),
        "n_mismatches": 0,
        "shape_checked": list(shape),
        "columns_checked": True,
    }


def _norm_cell(value: Any, pd) -> str:
    """
    Canonical string for an alignment probe.

    Raha reads every cell as a string with ``keep_default_na=False``, so a
    blank is ``""``; our loader turns the same cell into ``NaN``. A numeric
    column that Raha sees as ``"371"`` our loader may hold as ``371.0``. Both
    differences are representational, not disagreements about the data, so
    they are normalised away here — and *only* here, where the question is
    "is this the same cell", never where a number is computed.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "n/a", "na", "null", "?", "-"}:
        return "" if s == "" else s.lower()
    try:
        f = float(s)
    except (TypeError, ValueError):
        return s
    return repr(int(f)) if f == int(f) else repr(f)


def _operating_points(name, y, s, k_budget, np, best_f1_fn) -> List[Dict[str, Any]]:
    """
    Three operating points for one detector on the full cell universe.

    ``own flag set`` is ``s > 0``; ``equal budget`` is the top ``k_budget``
    cells; ``oracle best-F1`` uses the F1-maximising threshold and is labelled
    oracle everywhere it appears.
    """
    rows: List[Dict[str, Any]] = []

    def pr(flagged_mask) -> Dict[str, Any]:
        tp = int((flagged_mask & y).sum())
        n_flag = int(flagged_mask.sum())
        p = (tp / n_flag) if n_flag else None
        r = tp / int(y.sum()) if y.sum() else None
        f1 = (2 * p * r / (p + r)) if (p and r) else (0.0 if n_flag else None)
        return {"n_flagged": n_flag, "tp": tp, "fp": n_flag - tp,
                "precision": p, "recall": r, "f1": f1}

    rows.append({"detector": name, "point": "own flag set (s > 0)",
                 "threshold": None, **pr(s > 0.0)})

    k = min(int(k_budget), s.size)
    # Deterministic top-k: break score ties by index, which is only used for
    # the equal-budget row and is reported as such.
    order = np.lexsort((np.arange(s.size), -s))
    mask = np.zeros(s.size, dtype=bool)
    mask[order[:k]] = True
    rows.append({"detector": name, "point": f"equal budget (top-k, k={k})",
                 "threshold": None, **pr(mask)})

    bf1, thr = best_f1_fn(y, s)
    if bf1 is not None and thr is not None:
        rows.append({"detector": name, "point": "oracle best-F1",
                     "threshold": float(thr), **pr(s >= thr)})
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Markdown
# ═══════════════════════════════════════════════════════════════════════════════


def _f(value: Optional[float], nd: int = 4) -> str:
    """Render a float, or ``—`` for undefined. Never ``0.0000`` for missing."""
    if value is None:
        return "—"
    return f"{value:.{nd}f}"


def _ci(interval: Optional[Dict[str, Any]], nd: int = 4) -> str:
    if not interval or interval.get("n", 0) == 0 or interval.get("mean") is None:
        return "—"
    return (
        f"{interval['mean']:.{nd}f} "
        f"[{interval['lo']:.{nd}f}, {interval['hi']:.{nd}f}]"
    )


LABEL = {
    "raha": "**raha** (SIGMOD 2019)",
    "rule": "`rule` (ours)",
    "conditional_full": "`conditional_full` (ours)",
    "ensemble_noisy_or_b0": "`ensemble_noisy_or_b0` (ours)",
}
ORDER = ("raha", "rule", "conditional_full", "ensemble_noisy_or_b0")


def render_markdown(report: Dict[str, Any]) -> str:
    L: List[str] = []
    A = L.append

    A("# W5.2 — Raha vs ours: the first external benchmark on our data")
    A("")
    A("Generated by `experiments/w5_raha_baseline.py`. **Do not edit by hand** —")
    A("`docs/research_program.md` §5: no number enters by hand.")
    A("")
    A("Route **W5.2**, closing Gap 1 of `docs/research_program.md` §11: *\"no")
    A("external benchmark — every detection number we report was produced by code")
    A("we wrote.\"* Raha (Mahdavi et al., SIGMOD 2019; ACM SIGMOD *Most")
    A("Reproducible Paper* Award) is the first published system run on our")
    A("datasets, our cells and our labels.")
    A("")

    A("## Protocol")
    A("")
    A(f"- **Raha label budget:** {report.get('budget')} "
      "(`raha.Detection.LABELING_BUDGET`).")
    A(f"- **Label simulation:** {report.get('label_simulation')}")
    A("  **Our three detectors use zero labels.** Read every comparison with")
    A("  that asymmetry in front of you.")
    A("- Raha's 20 labels are drawn from the **whole table**, not from a train")
    A("  split, so its numbers are in-sample with respect to its label budget in")
    A("  a way ours are not. There is no way to run Raha otherwise: without")
    A("  ground truth its labelling loop does not start.")
    A("- **Our metrics** (`docs/metrics_definitions.md`, implemented in")
    A("  `src/detection_eval/`): one cell is one unit; a cell is positive iff its")
    A("  dirty value differs from clean under `ValueComparator`; PR-AUC is")
    A("  uninterpolated average precision; `—` means undefined, never 0.0.")
    A(f"- **Splits:** {report.get('splits')} stratified cell splits, seed "
      f"{report.get('seed')}, 30 % eval each, 95 % percentile bootstrap CI over")
    A("  splits (split variance only).")
    A("- **Operating-point tables** are whole-dataset, as")
    A("  `docs/metrics_definitions.md` §8 permits for detectors that do not fit")
    A("  on labels. Raha *does* consume 20 labels, so its whole-dataset row is")
    A("  in-sample with respect to those 20 cells.")
    A("- **Our three detectors** are imported unchanged from")
    A("  `experiments/w4_metrics.py` (`build_detectors`), so these rows are the")
    A("  same objects W4.2 measured, not transcriptions.")
    A("- **Cell alignment is asserted, not assumed.** Raha indexes a frame it")
    A("  read itself. Stage 1 dumps the literal string Raha saw at a")
    A("  deterministic sample of its flagged positions; stage 2 fails the run")
    A("  unless every probe matches our cell, and unless Raha's shape and column")
    A("  order match ours exactly.")
    A("")

    meta = report.get("stage1_meta") or {}
    first = next((v for v in meta.values() if v), None)
    if first:
        A("### Environment (stage 1, `.venv-raha/`)")
        A("")
        A(f"- `raha` {first.get('raha_version')}, pandas "
          f"{first.get('pandas_version')}, Python {first.get('python_version')}")
        hp = first.get("hyperparameters") or {}
        if hp:
            A("- Raha hyperparameters as run: "
              + ", ".join(f"`{k}={v}`" for k, v in hp.items()))
        A("")

    # ── Raha's own numbers ────────────────────────────────────────────────────
    A("## 1. Raha's own reported numbers")
    A("")
    A("Precision / recall / F1 exactly as `raha.Dataset.get_data_cleaning_evaluation`")
    A("returns them — Raha scoring itself against its own reading of `clean.csv`,")
    A("with no involvement from our code. This is the row a Raha paper would print.")
    A("")
    A("| dataset | P | R | F1 | cells flagged | wall time | status |")
    A("|---|---:|---:|---:|---:|---:|---|")
    for r in report["results"]:
        raha = r.get("raha") or {}
        if r["status"] != "ok":
            A(f"| {r['dataset']} | — | — | — | — | — | **{r['status']}**: "
              f"`{raha.get('error_type')}` |")
            continue
        A(f"| {r['dataset']} | {_f(raha.get('raha_precision'))} | "
          f"{_f(raha.get('raha_recall'))} | {_f(raha.get('raha_f1'))} | "
          f"{raha.get('n_flagged'):,} | {raha.get('wall_seconds', 0):.1f} s | ok |")
    A("")

    A("### Raha's error count vs ours")
    A("")
    A("Raha builds its own positive set by string-comparing its `dirty`/`clean`")
    A("readings; we use `ValueComparator` (`NaN == NaN`, `371 == 371.0`). Where")
    A("the two counts differ, Raha's self-reported P/R/F1 is computed against a")
    A("slightly different denominator than our table below, which is one reason")
    A("§3 re-scores its flags rather than quoting §1.")
    A("")
    A("| dataset | our corrupted cells | Raha's actual-error count | difference |")
    A("|---|---:|---:|---:|")
    for r in report["results"]:
        if r["status"] != "ok":
            A(f"| {r['dataset']} | {r['label_stats']['n_corrupted']:,} | — | — |")
            continue
        ours = int(r["label_stats"]["n_corrupted"])
        theirs = r["raha"].get("raha_n_actual_errors")
        diff = (theirs - ours) if theirs is not None else None
        A(f"| {r['dataset']} | {ours:,} | "
          f"{theirs:,} | {diff:+,} |" if theirs is not None
          else f"| {r['dataset']} | {ours:,} | — | — |")
    A("")

    # ── Re-scored under our metrics ───────────────────────────────────────────
    A("## 2. Raha re-scored under our metrics, and the side-by-side")
    A("")
    A("**A flag set has two distinct scores.** PR-AUC scores a *ranking*; Raha")
    A("emits a *set*, so its PR-AUC is measured on an axis it does not attempt —")
    A("the identical caveat `docs/metrics_definitions.md` §3 applies to our own")
    A("six-constant rule detector. The `distinct scores` column is printed so")
    A("this is visible per row. For a flag set, read the **operating-point**")
    A("table in §3, not the PR-AUC here.")
    A("")
    for r in report["results"]:
        A(f"### {r['dataset']}")
        A("")
        ls = r["label_stats"]
        A(f"{ls['n_cells']:,} cells, {ls['n_corrupted']:,} corrupted, "
          f"prevalence {ls['prevalence']:.6f} "
          f"({100 * ls['prevalence']:.3f} %)")
        A("")
        if r["status"] != "ok":
            A(f"Raha **{r['status']}** on this dataset "
              f"(`{(r.get('raha') or {}).get('error_type')}`); see §5. Our rows")
            A("are still printed, so the dataset is not silently dropped.")
            A("")
        A("| detector | labels | PR-AUC [95 % CI] | PR-AUC / prevalence | "
          "ROC-AUC | best-F1 (oracle) | P@k | distinct scores |")
        A("|---|---:|---|---:|---:|---|---:|---:|")
        pi = ls["prevalence"]
        for name in ORDER:
            d = (r.get("detectors") or {}).get(name)
            if not d:
                continue
            mean = (d["pr_auc"] or {}).get("mean")
            lift = (mean / pi) if (mean is not None and pi) else None
            A(f"| {LABEL[name]} | {d['labels_used']} | {_ci(d['pr_auc'])} | "
              f"{_f(lift, 2)} | {_f((d['roc_auc'] or {}).get('mean'))} | "
              f"{_ci(d['best_f1'])} | "
              f"{_f(d['full'].get('precision_at_k'))} | "
              f"{d['n_distinct_scores']} |")
        A("")
        al = r.get("alignment")
        if al:
            A(f"*Alignment: {al['n_probes']} probed flagged cells, "
              f"{al['n_mismatches']} mismatches; shape "
              f"{al['shape_checked']} and column order asserted equal.*")
            A("")

    # ── Operating points ──────────────────────────────────────────────────────
    A("## 3. Operating points — the comparable P/R/F1")
    A("")
    A("Three points per detector, on **all** cells. `own flag set` is each")
    A("detector's deployed behaviour (`s > 0`; for Raha, its literal flags).")
    A("`equal budget` gives every detector Raha's flag count, so the review cost")
    A("is identical and only the choice of cells differs — **this is the fair")
    A("head-to-head row.** `oracle best-F1` is an upper bound chosen with")
    A("knowledge of the labels and is never an operating point.")
    A("")
    for r in report["results"]:
        if not r.get("operating_points"):
            continue
        A(f"### {r['dataset']}")
        A("")
        A("| detector | operating point | cells flagged | TP | FP | precision | "
          "recall | F1 |")
        A("|---|---|---:|---:|---:|---:|---:|---:|")
        by_det = {}
        for row in r["operating_points"]:
            by_det.setdefault(row["detector"], []).append(row)
        for name in ORDER:
            for row in by_det.get(name, []):
                A(f"| {LABEL[name]} | {row['point']} | {row['n_flagged']:,} | "
                  f"{row['tp']:,} | {row['fp']:,} | {_f(row['precision'])} | "
                  f"{_f(row['recall'])} | {_f(row['f1'])} |")
        A("")

    # ── Where Raha wins ───────────────────────────────────────────────────────
    A("## 4. Where Raha beats us")
    A("")
    A(_where_raha_wins(report))
    A("")

    # ── Failures ──────────────────────────────────────────────────────────────
    failed = [r for r in report["results"] if r["status"] != "ok"]
    A("## 5. Failures")
    A("")
    if not failed:
        A("None. Raha ran to completion on all three datasets at the stated budget.")
    else:
        for r in failed:
            raha = r.get("raha") or {}
            A(f"### {r['dataset']} — {r['status']}")
            A("")
            A(f"- **Exception:** `{raha.get('error_type')}`")
            A(f"- **Message:** `{raha.get('error')}`")
            A("- The dataset keeps its rows in §2/§3 for our detectors, so it is")
            A("  documented rather than dropped.")
            A("")
    A("")

    # ── What this does not show ───────────────────────────────────────────────
    A("## 6. What this does not show")
    A("")
    A("1. **A different corruption protocol.** Published Raha numbers are on")
    A("   Flights, Beers, Hospital, Movies, Rayyan, Tax — typos, formatting and")
    A("   rule violations spread over many columns. Ours are our injector's:")
    A("   adult is 100 % valid→valid `income` label flips, smartfactory is")
    A("   erasure, breast_cancer is mixed. **Nothing here is comparable to a")
    A("   published Raha number**, only to ours on these cells.")
    A("2. **One budget.** 20 labels, because that is Raha's default and SAGED's")
    A("   protocol. Raha's own curve is steep in this region (R1: ~0.40 F1 at 2")
    A("   labels, 0.85 at 20), so a different budget would move the row.")
    A("3. **One seed, one run per dataset.** No confidence interval on *Raha's*")
    A("   own P/R/F1: the CIs in §2 are over our evaluation splits of a single")
    A("   fixed flag set, so they describe evaluation-sample noise, not Raha's")
    A("   run-to-run variance in which 20 cells it chose to label. SAGED's")
    A("   protocol asks for 10 seeds; this is 1.")
    A("4. **Raha is a detector, not a repairer.** Nothing here touches the")
    A("   calibration or cost layers that carry this project's claim; it is an")
    A("   anchor for the detection stage only.")
    A("5. **Our rule and ensemble rows inherit the `DatasetProfile`")
    A("   soft-leakage caveat** (CLAUDE.md §3): fences, ranges and vocabularies")
    A("   are built from `clean_df` aggregate statistics. Raha does not use")
    A("   them. That favours **us**, not Raha, and W5.1's ablation is the fix.")
    A("")
    return "\n".join(L) + "\n"


def _where_raha_wins(report: Dict[str, Any]) -> str:
    """
    State, per dataset, whether Raha beat our best label-free detector — from
    the measured numbers, with no adjectives the numbers do not support.
    """
    lines: List[str] = []
    for r in report["results"]:
        name = r["dataset"]
        if r["status"] != "ok":
            lines.append(
                f"- **{name}** — Raha did not run (`{(r.get('raha') or {}).get('error_type')}`), "
                "so there is no comparison to make. See §5."
            )
            continue

        pts = {(row["detector"], row["point"].split(" (")[0]): row
               for row in r["operating_points"]}
        raha_own = pts.get(("raha", "own flag set"))
        if raha_own is None:
            continue
        rf1 = raha_own["f1"] or 0.0

        # Our best at the SAME review budget as Raha's flag count.
        ours_eq = [
            (n, pts[(n, "equal budget")])
            for n in ("rule", "conditional_full", "ensemble_noisy_or_b0")
            if (n, "equal budget") in pts
        ]
        if not ours_eq:
            continue
        best_name, best_row = max(ours_eq, key=lambda kv: kv[1]["f1"] or 0.0)
        bf1 = best_row["f1"] or 0.0

        verdict = (
            "**Raha wins**" if rf1 > bf1 + 1e-12
            else ("**tie**" if abs(rf1 - bf1) <= 1e-12 else "**we win**")
        )
        lines.append(
            f"- **{name}** — {verdict}. Raha's own flag set: P "
            f"{_f(raha_own['precision'])} / R {_f(raha_own['recall'])} / F1 "
            f"{_f(rf1)} on {raha_own['n_flagged']:,} flags, with 20 oracle "
            f"labels. Our best label-free detector at the *same* flag budget is "
            f"`{best_name}`: P {_f(best_row['precision'])} / R "
            f"{_f(best_row['recall'])} / F1 {_f(bf1)}. "
            f"Difference in F1: {rf1 - bf1:+.4f}."
        )
    if not lines:
        return "No dataset produced a comparable pair; see §5."
    lines.append("")
    lines.append(
        "No spin on these rows. Where Raha is ahead it is ahead with 20 oracle "
        "labels against our zero, and that is the honest statement of the gap: "
        "the project's label-free detectors are being measured against a "
        "weakly-supervised published system, and the point of W5.2 is the "
        "anchor, not a win."
    )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[2])
    ap.add_argument("--stage", choices=("raha", "score"), required=True)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--flags-json", default=_default_json_path())
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.stage == "raha":
        return stage_raha(args)
    return stage_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
