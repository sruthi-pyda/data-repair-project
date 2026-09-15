"""src/baselines — Phase 9: baseline repair-decision policies."""

from src.baselines.baselines import (
    AlwaysAutoBaseline,
    BaselineRecord,
    ConfidenceThresholdBaseline,
    DEFAULT_THRESHOLDS,
    sweep_thresholds,
    wrap_proposed_method,
)

__all__ = [
    "AlwaysAutoBaseline",
    "BaselineRecord",
    "ConfidenceThresholdBaseline",
    "DEFAULT_THRESHOLDS",
    "sweep_thresholds",
    "wrap_proposed_method",
]
