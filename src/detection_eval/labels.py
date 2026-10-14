"""
src/detection_eval/labels.py
----------------------------
Cell-level ground-truth corruption labels.

WHAT A LABEL MEANS
==================
``True`` at ``(row, column)`` means **the dirty value is not the clean
value** — the cell is genuinely corrupted. ``False`` means the dirty value
survived injection unchanged.

This is the *detection* target. It is NOT the repair-correctness target
(``p``) that ``src/calibration/`` learns, and it is NOT "the detector flagged
it". It is the thing a detector is trying to predict.

HOW EQUALITY IS DECIDED
=======================
Via :class:`src.evaluation.repair_metrics.ValueComparator`, the canonical
"is this cell corrupted" test already used by Phase 10. Reusing it is the
whole point: if detection and repair evaluation disagreed about which cells
are corrupted, no cross-phase number would be comparable.

The comparator gives us, for free, the two properties this task needs:

- ``NaN == NaN`` → equal. A cell that was blank in clean and is blank in
  dirty is not corrupted.
- ``371 == 371.0`` → equal. NaN injection upcasts int64 columns to float64
  (``src/data/loader.py`` documents this), so a naive ``!=`` would label
  every cell of every affected column as corrupted.

LEAKAGE RULE (CLAUDE.md §3)
===========================
This is the single permitted entry point for ``clean_df`` in
``src/detection_eval/`` — the symmetric case to ``CalibrationBuilder.build``
and ``DetectionLabelBuilder.build``. The returned frame is a label matrix for
*evaluation and split stratification only*. It must never be passed to a
detector, and nothing in this module touches a detector.

KNOWN TOTALS (docs/cost_semantics.md, reproduced by this module)
================================================================
    breast_cancer    631 corrupted cells of   8,388   (7.52  %)
    smartfactory   7,093 corrupted cells of 449,255   (1.58  %)
    adult          1,701 corrupted cells of 678,330   (0.251 %)

``tests/test_detection_eval.py`` asserts these three numbers exactly. If a
future change moves them, the test fails and the *expectation is reported*,
not quietly adjusted — the totals are quoted in the cost-semantics document
and in every results table.

Worked example
==============
    >>> from src.data.loader import DataLoader
    >>> from src.detection_eval.labels import cell_labels, label_stats
    >>> clean_df, dirty_df = DataLoader("datasets").load("breast_cancer")
    >>> labels = cell_labels(clean_df, dirty_df)
    >>> label_stats(labels)["n_corrupted"]
    631
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.evaluation.repair_metrics import ValueComparator

logger = logging.getLogger(__name__)

__all__ = ["cell_labels", "label_stats"]


def cell_labels(
    clean_df: pd.DataFrame,
    dirty_df: pd.DataFrame,
    comparator: Optional[ValueComparator] = None,
) -> pd.DataFrame:
    """
    Return a boolean frame, ``True`` where the dirty cell differs from clean.

    The result has **exactly** ``dirty_df``'s shape, index and column order,
    so ``labels[col].to_numpy()`` lines up with ``dirty_df[col].to_numpy()``
    and a detector's score frame can be compared cell for cell.

    Parameters
    ----------
    clean_df : pd.DataFrame
        Ground truth. Must contain every column of ``dirty_df`` and have the
        same number of rows.
    dirty_df : pd.DataFrame
        The pipeline input. Defines the shape of the output.
    comparator : ValueComparator, optional
        Equality test. Default ``ValueComparator()`` — exact for integers,
        1e-9 tolerant for float round-off, case-sensitive for strings.
        Pass a configured comparator to loosen either.

    Returns
    -------
    pd.DataFrame
        dtype ``bool``, same shape/index/columns as ``dirty_df``.

    Raises
    ------
    ValueError
        Row counts differ, or a ``dirty_df`` column is absent from
        ``clean_df``.

    Notes
    -----
    Implementation: a vectorised pre-filter marks cells that are *raw-equal*
    (``both NaN``, or ``a == b`` elementwise). Raw equality implies comparator
    equality in every branch of the comparator — identical numbers differ by
    0 ≤ tol, identical strings strip to identical strings — so those cells can
    be skipped safely. The per-cell comparator then runs only on the
    (few thousand) remaining candidates. The result is bit-identical to
    calling the comparator on every cell; the pre-filter only buys speed, and
    ``tests/test_detection_eval.py`` asserts the equivalence on a frame
    engineered to exercise both paths.
    """
    if comparator is None:
        comparator = ValueComparator()

    if len(clean_df) != len(dirty_df):
        raise ValueError(
            f"Row count mismatch: clean={len(clean_df)}, dirty={len(dirty_df)}. "
            "Use DataLoader.load(), which aligns the pair."
        )
    missing = [c for c in dirty_df.columns if c not in clean_df.columns]
    if missing:
        raise ValueError(
            f"Columns present in dirty_df but absent from clean_df: {missing}"
        )

    out: Dict[Any, np.ndarray] = {}
    for col in dirty_df.columns:
        dirty_vals = dirty_df[col].to_numpy()
        clean_vals = clean_df[col].to_numpy()

        corrupted = np.ones(len(dirty_vals), dtype=bool)
        fast_equal = _raw_equal(dirty_vals, clean_vals)
        corrupted[fast_equal] = False

        # Only the cells the cheap test could not clear reach the comparator.
        for i in np.flatnonzero(~fast_equal):
            if comparator.are_equal(dirty_vals[i], clean_vals[i]):
                corrupted[i] = False

        out[col] = corrupted

    labels = pd.DataFrame(out, index=dirty_df.index, columns=dirty_df.columns)
    logger.info(
        "Built cell labels: %d corrupted of %d cells (%.4f%%).",
        int(labels.to_numpy().sum()),
        labels.size,
        100.0 * float(labels.to_numpy().mean()) if labels.size else 0.0,
    )
    return labels


def label_stats(labels: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarise a label frame.

    Parameters
    ----------
    labels : pd.DataFrame
        Boolean frame from :func:`cell_labels`.

    Returns
    -------
    dict
        ``n_cells``, ``n_corrupted``, ``prevalence`` (0.0 for an empty frame —
        there is no ratio to report, and this field is a rate, not a score),
        ``n_rows``, ``n_columns``, ``per_column`` (``{column: count}``, every
        column present including zeros) and ``per_column_prevalence``.
    """
    arr = labels.to_numpy(dtype=bool, copy=False)
    n_cells = int(arr.size)
    n_corrupted = int(arr.sum())
    n_rows = int(labels.shape[0])

    per_column = {str(c): int(labels[c].to_numpy(dtype=bool).sum()) for c in labels.columns}
    per_column_prevalence = {
        c: (v / n_rows if n_rows else 0.0) for c, v in per_column.items()
    }

    return {
        "n_cells": n_cells,
        "n_corrupted": n_corrupted,
        "prevalence": (n_corrupted / n_cells) if n_cells else 0.0,
        "n_rows": n_rows,
        "n_columns": int(labels.shape[1]),
        "per_column": per_column,
        "per_column_prevalence": per_column_prevalence,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Internal
# ═══════════════════════════════════════════════════════════════════════════════


def _raw_equal(dirty_vals: np.ndarray, clean_vals: np.ndarray) -> np.ndarray:
    """
    Conservative vectorised "definitely equal" mask.

    ``True`` only where equality is certain under the comparator: both values
    missing, or elementwise ``==``. ``False`` means "unknown, ask the
    comparator" — never "unequal". Any elementwise comparison numpy refuses
    (ragged object arrays, incomparable types) degrades to all-``False``,
    which is slow but still correct.
    """
    both_na = pd.isna(dirty_vals) & pd.isna(clean_vals)
    try:
        eq = np.asarray(dirty_vals == clean_vals, dtype=bool)
        if eq.shape != dirty_vals.shape:  # broadcast surprise → do not trust it
            eq = np.zeros(dirty_vals.shape, dtype=bool)
    except Exception:  # pragma: no cover - defensive
        eq = np.zeros(dirty_vals.shape, dtype=bool)
    return np.asarray(both_na, dtype=bool) | eq
