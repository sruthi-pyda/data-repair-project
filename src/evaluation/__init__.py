"""src/evaluation — Phase 10: post-hoc evaluation of repair quality, calibration, and cost."""

from src.evaluation.repair_metrics import (
    CellResult,
    RepairEvaluator,
    RepairMetrics,
    ValueComparator,
)
from src.evaluation.calibration_metrics import (
    CalibrationComparison,
    CalibrationEvaluator,
    CalibrationReport,
    nll_score,
)
from src.evaluation.cost_metrics import (
    CostEvaluator,
    CostReport,
    PolicyComparisonRow,
    compare_policies,
    print_comparison_table,
)

__all__ = [
    # repair_metrics
    "CellResult",
    "RepairEvaluator",
    "RepairMetrics",
    "ValueComparator",
    # calibration_metrics
    "CalibrationComparison",
    "CalibrationEvaluator",
    "CalibrationReport",
    "nll_score",
    # cost_metrics
    "CostEvaluator",
    "CostReport",
    "PolicyComparisonRow",
    "compare_policies",
    "print_comparison_table",
]
