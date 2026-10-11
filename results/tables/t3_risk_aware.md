# T3 — Risk-aware performance of the proposed policy

What the calibrated, cost-sensitive policy actually does with the flagged cells, and what that costs against ground truth.

**Protocol:** one representative pipeline run, `seed = 42`, `split_seed = 42`, detections no cap (full dataset). These are point estimates from a single calibration split, not interval estimates.

| dataset | AUTO share | HUMAN share | ABSTAIN share | realised cost | mean cost / cell | wrong-auto rate | abstain on corrupted | abstain on clean | p\* | q\* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adult | 0.0000  (0.0000 [0.0000, 0.0000]) | 0.0000  (0.0000 [0.0000, 0.0000]) | 1.0000  (1.0000 [1.0000, 1.0000]) | 0.00  (0.00 [0.00, 0.00]) | 0.0000 | 0.0000 | 0 / 5,829 | 5,829 / 5,829 | 0.800 | 0.400 |
| breast_cancer | 0.1465  (0.1654 [0.1428, 0.1830]) | 0.8535  (0.8346 [0.8170, 0.8572]) | 0.0000  (0.0000 [0.0000, 0.0000]) | 1,615.00  (1,631.50 [1,620.75, 1,649.00]) | 4.5493 | 0.0385 | 0 / 0 | 0 / 0 | 0.900 | 0.500 |
| smartfactory | 0.0000  (0.0000 [0.0000, 0.0000]) | 1.0000  (1.0000 [1.0000, 1.0000]) | 0.0000  (0.0000 [0.0000, 0.0000]) | 14,186.00  (14,186.00 [14,186.00, 14,186.00]) | 2.0000 | 0.0000 | 0 / 0 | 0 / 0 | 0.920 | 0.400 |

**Notes**

- All costs are abstract **experimental penalty units** from `configs/costs.yaml`. They are not money, not clinical harm, and not supplied by the REIN benchmark (docs/cost_semantics.md §1).
- Counts and rates are from the single representative run; where the same quantity also has an interval over repeated splits it is printed in parentheses as `mean [lo, hi]`.
- `realised cost` charges `C_wrong` for an applied repair that is wrong, `C_human` per cell routed to a reviewer (**including false alarms**), and `C_abstain` only when an abstained cell was genuinely corrupted — interpretation (a) in docs/cost_semantics.md §2, the one under which `E[ABSTAIN] = q·C_abstain` is an unbiased estimate of what the evaluator charges.
- `abstain on corrupted` is the abstention mistake the cost model charges for; `abstain on clean` is the false alarm it correctly charges nothing for.
- `p* = 1 − C_human/C_wrong` and `q* = C_human/C_abstain` are the derived decision boundaries (docs/cost_semantics.md §4). `q*` is horizontal, so for a single cell at fixed q only two of the three arms are ever live; all three occur across a population that mixes real errors with false alarms.
- **False negatives are priced at zero.** Genuine corruptions the detector never flagged (adult: 1,701; breast_cancer: 276; smartfactory: 0) enter no arm of the objective, so every cost number in these tables is a statement about the flagged cells only (docs/cost_semantics.md §3).
- **Adult framing (docs/cost_semantics.md §3).** Adult's realised cost measures only residual corruption on the cells the detector flagged; the measured value here is 0.00. A value of 0 must be read as *zero residual-corruption cost on the flagged cells*, never as "zero cost" or "the data is now clean": 1,701 genuinely corrupted cells (income-label flips) were never detected, no action was ever taken on them, and they enter no arm of the objective. The defensible claim is that the calibrated policy declines to act on a detector whose flags are false positives, not that Adult was repaired.
- **adult** p-calibration fallback: All 4081 training labels are incorrect; no calibrator can fit. Calibrated probability collapses to the base rate (0.00). This is the documented Adult negative result.
