"""
src/pipeline
------------
End-to-end orchestration of Phases 1–10 for the dashboard (Phase 11) and
the experiment runner (Phase 13). See ``runner.py`` for the leakage rules.
"""

from src.pipeline.runner import (
    DATASETS,
    SKIP_COLUMNS,
    PipelineBundle,
    PolicyEvaluation,
    decide_all,
    evaluate_policies,
    run_pipeline,
)

__all__ = [
    "DATASETS",
    "SKIP_COLUMNS",
    "PipelineBundle",
    "PolicyEvaluation",
    "decide_all",
    "evaluate_policies",
    "run_pipeline",
]
