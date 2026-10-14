"""
src/detectors/
--------------
Learned, self-supervised cell-level error detectors (research route R2+).

These detectors are distinct from ``src/detection/detector.py``, which is the
original hand-written six-rule detector. The modules here *learn* from the
dirty table itself and emit a continuous per-cell suspicion score rather than
a boolean flag, so they can be scored with PR-AUC (the primary detection
metric in ``docs/research_program.md`` Section 0).

Leakage rule (CLAUDE.md Section 3): nothing in this package ever receives
``clean_df``. Clean data exists only to build evaluation labels, outside the
detector.
"""

from src.detectors.conditional import ColumnModel, ConditionalDetector

__all__ = ["ConditionalDetector", "ColumnModel"]
