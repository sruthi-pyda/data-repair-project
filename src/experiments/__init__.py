"""
src/experiments
---------------
Experiment-level machinery that sits *above* the pipeline: repeated runs,
aggregation of per-run numbers into intervals, and the honest-reporting
bookkeeping that goes with them (how many runs a calibrator actually
fitted in, rather than a mean that silently averages fallbacks).

Nothing in this package touches ``clean_df`` directly. It calls
``src.pipeline.runner``, which keeps the Phase 3/4/5 stages ground-truth
free and confines clean data to the label builders and the evaluators
(CLAUDE.md §3). Varying ``split_seed`` re-splits the *already permitted*
calibration labels; it does not grant any new access to ground truth.

See ``aggregate.py`` and docs/phase13_plan.md §1.
"""

from src.experiments.aggregate import (
    Interval,
    RepeatedSplitResult,
    bootstrap_interval,
    normal_interval,
    repeated_splits,
)

__all__ = [
    "Interval",
    "RepeatedSplitResult",
    "bootstrap_interval",
    "normal_interval",
    "repeated_splits",
]
