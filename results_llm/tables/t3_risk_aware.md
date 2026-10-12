# T3 — Risk-aware performance of the proposed policy

What the calibrated, cost-sensitive policy actually does with the flagged cells, and what that costs against ground truth.

**Protocol:** one representative pipeline run, `seed = 42`, `split_seed = 42`, detections no cap (full dataset). These are point estimates from a single calibration split, not interval estimates.

| dataset | AUTO share | HUMAN share | ABSTAIN share | realised cost | mean cost / cell | wrong-auto rate | abstain on corrupted | abstain on clean | p\* | q\* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| breast_cancer | 0.1155  (0.1656 [0.1442, 0.1876]) | 0.8845  (0.8344 [0.8124, 0.8558]) | 0.0000  (0.0000 [0.0000, 0.0000]) | 1,670.00  (1,621.00 [1,617.00, 1,626.00]) | 4.7042 | 0.0488 | 0 / 0 | 0 / 0 | 0.900 | 0.500 |

**Notes**

- All costs are abstract **experimental penalty units** from `configs/costs.yaml`. They are not money, not clinical harm, and not supplied by the REIN benchmark (docs/cost_semantics.md §1).
- Counts and rates are from the single representative run; where the same quantity also has an interval over repeated splits it is printed in parentheses as `mean [lo, hi]`.
- `realised cost` charges `C_wrong` for an applied repair that is wrong, `C_human` per cell routed to a reviewer (**including false alarms**), and `C_abstain` only when an abstained cell was genuinely corrupted — interpretation (a) in docs/cost_semantics.md §2, the one under which `E[ABSTAIN] = q·C_abstain` is an unbiased estimate of what the evaluator charges.
- `abstain on corrupted` is the abstention mistake the cost model charges for; `abstain on clean` is the false alarm it correctly charges nothing for.
- `p* = 1 − C_human/C_wrong` and `q* = C_human/C_abstain` are the derived decision boundaries (docs/cost_semantics.md §4). `q*` is horizontal, so for a single cell at fixed q only two of the three arms are ever live; all three occur across a population that mixes real errors with false alarms.
- **False negatives are priced at zero.** Genuine corruptions the detector never flagged (breast_cancer: 276) enter no arm of the objective, so every cost number in these tables is a statement about the flagged cells only (docs/cost_semantics.md §3).
- **Repair mode: `real` (Phase 12, `src/repair/llm_repair.py`).**
- `breast_cancer`: model `openai/gpt-oss-120b` · 392 cells · 390 answered by the model · **2 fell back to the mock repairer (0.5%)** · reasons: malformed_response 2 · 84 cache hits · 1 rate limits · 1 key rotations.
- **The intervals in T2 and the cost interval in T4 come from the mock repairer even in real mode**: `repeated_splits` (`src/experiments/aggregate.py`) builds its own `MockRepairer` and takes no repair mode, so those intervals describe the calibration of the mock pipeline. T1, T3 and the predictions CSV use the real repairs. Mixing them would be the dishonest reading.
