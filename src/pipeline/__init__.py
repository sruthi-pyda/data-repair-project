"""
src/pipeline
------------
End-to-end orchestration of Phases 1–10 for the dashboard (Phase 11) and
the experiment runner (Phase 13). See ``runner.py`` for the leakage rules.

``run_pipeline(..., mode="mock"|"real")`` selects the Phase 4 repairer
(Phase 12); ``build_repairer`` is the same selection on its own, for a
caller that wants the repairer without the pipeline.
"""

from src.pipeline.runner import (
    DATASETS,
    DEFAULT_IQR_K,
    DEFAULT_REPAIR_MODE,
    REPAIR_MODES,
    SKIP_COLUMNS,
    PipelineBundle,
    PolicyEvaluation,
    build_repairer,
    decide_all,
    default_flag_outliers,
    evaluate_policies,
    run_pipeline,
)

__all__ = [
    "DATASETS",
    "DEFAULT_IQR_K",
    "DEFAULT_REPAIR_MODE",
    "REPAIR_MODES",
    "SKIP_COLUMNS",
    "PipelineBundle",
    "PolicyEvaluation",
    "build_repairer",
    "decide_all",
    "default_flag_outliers",
    "evaluate_policies",
    "run_pipeline",
]
