# T4 — Policy comparison

The proposed calibrated risk-aware policy against always-auto, always-human, always-abstain and the raw-confidence threshold sweep, all scored on the same cells with the same cost config.

**Protocol:** one representative pipeline run, `seed = 42`, `split_seed = 42`, detections no cap (full dataset). These are point estimates from a single calibration split, not interval estimates.  
**Protocol:** `src.experiments.aggregate.repeated_splits`, 5 repeated calibration splits with `seed = 42` held fixed and only `split_seed` varying, split seeds 1000–1004. Intervals are 95 % bootstrap intervals over splits, written `mean [lo, hi]`. Varying `split_seed` changes the calibrators (and therefore p and q) but not the detections or the mock repairs, so the interval is over calibration splits and nothing else (docs/phase13_plan.md §0).

| dataset | policy | realised cost | mean / cell | cost over splits  mean [95 % CI] | AUTO | HUMAN | ABSTAIN | wrong-auto rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| breast_cancer | calibrated_risk_aware | 1,670.00 | 4.7042 | 1,621.00 [1,617.00, 1,626.00] | 41 | 314 | 0 | 0.0488 |
| breast_cancer | always_auto | 5,150.00 | 14.5070 | 5,800.00 [5,800.00, 5,800.00] | 355 | 0 | 0 | 0.2901 |
| breast_cancer | always_human | 1,775.00 | 5.0000 | — | 0 | 355 | 0 | 0.0000 |
| breast_cancer | always_abstain | 3,550.00 | 10.0000 | — | 0 | 0 | 355 | 0.0000 |
| breast_cancer | confidence_threshold_0.50 | 4,675.00 | 13.1690 | — | 340 | 15 | 0 | 0.2706 |
| breast_cancer | confidence_threshold_0.60 | 2,825.00 | 7.9577 | — | 260 | 95 | 0 | 0.1808 |
| breast_cancer | confidence_threshold_0.70 | 1,700.00 | 4.7887 | — | 175 | 180 | 0 | 0.0914 |
| breast_cancer | confidence_threshold_0.80 | 1,620.00 | 4.5634 | 1,770.00 [1,770.00, 1,770.00] | 51 | 304 | 0 | 0.0392 |
| breast_cancer | confidence_threshold_0.90 | 1,775.00 | 5.0000 | — | 0 | 355 | 0 | 0.0000 |

**Notes**

- All costs are abstract **experimental penalty units** from `configs/costs.yaml`. They are not money, not clinical harm, and not supplied by the REIN benchmark (docs/cost_semantics.md §1).
- `realised cost`, `mean / cell` and the action counts are from the single representative run described above; the interval column is the same policy's cost over 5 repeated calibration splits, from `repeated_splits`, and is blank (`—`) for any policy that aggregation does not report.
- `always_human` and `always_abstain` are the two remaining corners of the action space, added here as cost references. They are not Phase 9 baselines and they never abstain *selectively*, so they are context for the cost scale rather than competitive methods.
- The threshold baseline acts on **raw** confidence, which is what the calibration in T2 is meant to improve on; comparing it against the proposed row is the whole claim of the project.
- A lower realised cost here does **not** mean better repairs — T1 holds the repairs fixed across every row. Only the actions differ.
- **False negatives are priced at zero.** Genuine corruptions the detector never flagged (breast_cancer: 276) enter no arm of the objective, so every cost number in these tables is a statement about the flagged cells only (docs/cost_semantics.md §3).
- **breast_cancer**: proposed 1,670.00 vs best alternative `confidence_threshold_0.80` 1,620.00 — ⚠ **proposed LOSES** to `confidence_threshold_0.80` by 50.00.
- **Repair mode: `real` (Phase 12, `src/repair/llm_repair.py`).**
- `breast_cancer`: model `openai/gpt-oss-120b` · 392 cells · 390 answered by the model · **2 fell back to the mock repairer (0.5%)** · reasons: malformed_response 2 · 84 cache hits · 1 rate limits · 1 key rotations.
- **The intervals in T2 and the cost interval in T4 come from the mock repairer even in real mode**: `repeated_splits` (`src/experiments/aggregate.py`) builds its own `MockRepairer` and takes no repair mode, so those intervals describe the calibration of the mock pipeline. T1, T3 and the predictions CSV use the real repairs. Mixing them would be the dishonest reading.
