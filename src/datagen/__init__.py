"""
src/datagen/
------------
Controlled corruption injection (route W5.4b of ``docs/research_program.md``
§11, gap 3).

WHY THIS PACKAGE EXISTS
=======================
Every number in this project rests on three datasets whose corruption we did
not choose. Two of them are degenerate: Smart Factory's corruption is
literally blank cells, and Adult's is 100 % income-label flips confined to one
column. So we can say how the pipeline behaves on *those three* corruption
mixes and nothing more.

This package takes the **clean** table as ground truth and injects corruption
whose type, rate, position and value are all ours. That turns "how good is the
detector?" into a function of two controlled variables — corruption type and
corruption rate — and lets us name the exact cells of that grid where the
method works and where it fails.

WHAT IT ESTABLISHES, AND WHAT IT DOES NOT
=========================================
It establishes robustness **across corruption**, on the same three domains.
It says **nothing** about cross-domain generalisation; the domains are
unchanged. Obtaining further REIN tables is a separate route. Any write-up
built on this package must repeat that limit.

THE CIRCULARITY PROBLEM, AND THE MITIGATIONS
============================================
``src/detection/detector.py`` was written for particular corruption shapes.
Injecting exactly those shapes and then detecting them proves nothing. Four
mitigations, all enforced here rather than left to the caller:

1. **Injection only ever reads the clean frame.** The shipped ``dirty.csv`` is
   never touched, so no injected corruption can coincide with a real one.
2. **No injection parameter mirrors a detector threshold.**
   :class:`Corruptor` places ``out_of_range`` values at least ``5x`` the full
   column *range span* beyond the observed min/max — orders of magnitude past
   the detector's ``Q3 + 3*IQR`` fence — rather than just outside it, which
   would make the fence's success a measurement of the injector.
   ``swap_conditional`` does the opposite: it restricts itself to values
   *inside* the global 1.5x IQR fence, so no fence can see them by
   construction.
3. **Results are reported per type, never aggregated.** A type the detector
   was designed for cannot inflate a headline.
4. **``swap_valid`` is the honesty anchor.** It only ever writes a value that
   already occurs in that same column, so no vocabulary, range, format or
   missing-token rule can detect it *by construction*. If a rule detector
   scores above chance on ``swap_valid``, the injector is broken, not the
   detector good. :mod:`tests.test_corruptor` asserts the "value already
   exists in the column" property directly.

LEAKAGE RULE (CLAUDE.md §3)
===========================
This package reads ``clean_df`` and writes a corrupted copy. That is the whole
point of it, and it is the same permitted class of use as
``CalibrationBuilder.build``: clean data produces *labels and inputs for an
experiment*, never a feature handed to a detector. Nothing here imports a
detector, a repairer, or any evaluation code, and the returned
:class:`~src.datagen.corruptor.InjectionReport` carries the injected positions
so an experiment never has to re-derive them from the clean frame at
inference time.

A corrupted frame produced here is handed to the pipeline exactly as a shipped
``dirty.csv`` is: the detector sees only the corrupted frame plus the
``DatasetProfile``. The profile is still built from the clean frame, which is
CLAUDE.md §3's known soft-leakage caveat, unchanged and separately ablated by
route W5.1.
"""

from __future__ import annotations

from src.datagen.corruptor import (
    CORRUPTION_TYPES,
    ColumnRole,
    Corruptor,
    InjectionReport,
    classify_columns,
    eligible_columns,
    inject,
)

__all__ = [
    "CORRUPTION_TYPES",
    "ColumnRole",
    "Corruptor",
    "InjectionReport",
    "classify_columns",
    "eligible_columns",
    "inject",
]
