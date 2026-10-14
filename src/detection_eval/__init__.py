"""
src/detection_eval/
-------------------
Route R0 of ``docs/research_program.md``: the evaluation harness that every
other detection route is measured by.

WHY THIS PACKAGE EXISTS
=======================
The project previously optimised only ``p`` = P(proposed repair is correct)
and used hand-written rules for detection. §0 of the research program argues
that detection — P(cell is corrupted) — is where the headroom is, because a
destroyed value cannot be recovered no matter how good the repairer is.

Detection on this data is extremely imbalanced (adult: 1,701 corrupted cells
out of 678,330, a prevalence of 0.25 %). At that prevalence accuracy and
ROC-AUC are vanity metrics: a detector that flags nothing scores 99.75 %
accuracy. **PR-AUC (average precision) is the primary metric here.**

MODULES
=======
``labels.py``
    Cell-level ground-truth corruption labels from the clean/dirty pair,
    using the canonical :class:`~src.evaluation.repair_metrics.ValueComparator`
    so "is this cell corrupted" means exactly the same thing here as it does
    in Phase 10's repair evaluation.

``metrics.py``
    PR-AUC, ROC-AUC, best-F1, precision/recall/lift@k, bundled in
    :class:`~src.detection_eval.metrics.DetectionMetrics`. Undefined
    quantities return ``None``, never ``0.0`` — a ``0.0`` in a results table
    reads as a measured score.

``harness.py``
    The comparable-evaluation protocol: a detector protocol any route can
    satisfy, an adapter for the existing rule detector, stratified seeded
    cell-level splits, and bootstrap CIs over splits.

``leaderboard.py``
    The shared JSON store and ``results/detection_leaderboard.md``. No number
    enters the leaderboard by hand.

LEAKAGE RULE (CLAUDE.md §3)
===========================
``clean_df`` enters this package in exactly one place: ``labels.cell_labels``,
which turns it into 0/1 *cell-corrupted* labels. Those labels are used for
evaluation, and for stratifying the splits. **No label is ever handed to a
detector.** A detector that needs fitting is fitted on dirty values at train
cells only — see :func:`~src.detection_eval.harness.evaluate_detector` for how
the protocol enforces that.
"""

from __future__ import annotations

from src.detection_eval.labels import cell_labels, label_stats
from src.detection_eval.metrics import (
    DetectionMetrics,
    best_f1,
    evaluate_scores,
    f1_at_threshold,
    lift_at_k,
    pr_auc,
    precision_at_k,
    recall_at_k,
    roc_auc,
)

__all__ = [
    "cell_labels",
    "label_stats",
    "DetectionMetrics",
    "best_f1",
    "evaluate_scores",
    "f1_at_threshold",
    "lift_at_k",
    "pr_auc",
    "precision_at_k",
    "recall_at_k",
    "roc_auc",
]
