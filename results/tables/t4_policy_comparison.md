# T4 — Policy comparison

The proposed calibrated risk-aware policy against always-auto, always-human, always-abstain and the raw-confidence threshold sweep, all scored on the same cells with the same cost config.

**Protocol:** one representative pipeline run, `seed = 42`, `split_seed = 42`, detections no cap (full dataset). These are point estimates from a single calibration split, not interval estimates.  
**Protocol:** `src.experiments.aggregate.repeated_splits`, 20 repeated calibration splits with `seed = 42` held fixed and only `split_seed` varying, split seeds 1000–1019. Intervals are 95 % bootstrap intervals over splits, written `mean [lo, hi]`. Varying `split_seed` changes the calibrators (and therefore p and q) but not the detections or the mock repairs, so the interval is over calibration splits and nothing else (docs/phase13_plan.md §0).

| dataset | policy | realised cost | mean / cell | cost over splits  mean [95 % CI] | AUTO | HUMAN | ABSTAIN | wrong-auto rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| adult | calibrated_risk_aware | 0.00 | 0.0000 | 0.00 [0.00, 0.00] | 0 | 0 | 5,829 | 0.0000 |
| adult | always_auto | 58,290.00 | 10.0000 | 58,290.00 [58,290.00, 58,290.00] | 5,829 | 0 | 0 | 1.0000 |
| adult | always_human | 11,658.00 | 2.0000 | — | 0 | 5,829 | 0 | 0.0000 |
| adult | always_abstain | 0.00 | 0.0000 | — | 0 | 0 | 5,829 | 0.0000 |
| adult | confidence_threshold_0.50 | 58,218.00 | 9.9876 | — | 5,820 | 9 | 0 | 1.0000 |
| adult | confidence_threshold_0.60 | 56,418.00 | 9.6788 | — | 5,595 | 234 | 0 | 1.0000 |
| adult | confidence_threshold_0.70 | 11,658.00 | 2.0000 | — | 0 | 5,829 | 0 | 0.0000 |
| adult | confidence_threshold_0.80 | 11,658.00 | 2.0000 | 11,658.00 [11,658.00, 11,658.00] | 0 | 5,829 | 0 | 0.0000 |
| adult | confidence_threshold_0.90 | 11,658.00 | 2.0000 | — | 0 | 5,829 | 0 | 0.0000 |
| breast_cancer | calibrated_risk_aware | 1,615.00 | 4.5493 | 1,631.50 [1,620.75, 1,649.00] | 52 | 303 | 0 | 0.0385 |
| breast_cancer | always_auto | 5,800.00 | 16.3380 | 5,800.00 [5,800.00, 5,800.00] | 355 | 0 | 0 | 0.3268 |
| breast_cancer | always_human | 1,775.00 | 5.0000 | — | 0 | 355 | 0 | 0.0000 |
| breast_cancer | always_abstain | 3,550.00 | 10.0000 | — | 0 | 0 | 355 | 0.0000 |
| breast_cancer | confidence_threshold_0.50 | 5,615.00 | 15.8169 | — | 312 | 43 | 0 | 0.3462 |
| breast_cancer | confidence_threshold_0.60 | 3,780.00 | 10.6479 | — | 229 | 126 | 0 | 0.2751 |
| breast_cancer | confidence_threshold_0.70 | 1,720.00 | 4.8451 | — | 31 | 324 | 0 | 0.0645 |
| breast_cancer | confidence_threshold_0.80 | 1,770.00 | 4.9859 | 1,770.00 [1,770.00, 1,770.00] | 21 | 334 | 0 | 0.0952 |
| breast_cancer | confidence_threshold_0.90 | 1,770.00 | 4.9859 | — | 21 | 334 | 0 | 0.0952 |
| smartfactory | calibrated_risk_aware | 14,186.00 | 2.0000 | 14,186.00 [14,186.00, 14,186.00] | 0 | 7,093 | 0 | 0.0000 |
| smartfactory | always_auto | 126,275.00 | 17.8028 | 126,275.00 [126,275.00, 126,275.00] | 7,093 | 0 | 0 | 0.7121 |
| smartfactory | always_human | 14,186.00 | 2.0000 | — | 0 | 7,093 | 0 | 0.0000 |
| smartfactory | always_abstain | 35,465.00 | 5.0000 | — | 0 | 0 | 7,093 | 0.0000 |
| smartfactory | confidence_threshold_0.50 | 124,143.00 | 17.5022 | — | 6,984 | 109 | 0 | 0.7098 |
| smartfactory | confidence_threshold_0.60 | 39,785.00 | 5.6091 | — | 2,738 | 4,355 | 0 | 0.4540 |
| smartfactory | confidence_threshold_0.70 | 14,186.00 | 2.0000 | — | 0 | 7,093 | 0 | 0.0000 |
| smartfactory | confidence_threshold_0.80 | 14,186.00 | 2.0000 | 14,186.00 [14,186.00, 14,186.00] | 0 | 7,093 | 0 | 0.0000 |
| smartfactory | confidence_threshold_0.90 | 14,186.00 | 2.0000 | — | 0 | 7,093 | 0 | 0.0000 |

**Notes**

- All costs are abstract **experimental penalty units** from `configs/costs.yaml`. They are not money, not clinical harm, and not supplied by the REIN benchmark (docs/cost_semantics.md §1).
- `realised cost`, `mean / cell` and the action counts are from the single representative run described above; the interval column is the same policy's cost over 20 repeated calibration splits, from `repeated_splits`, and is blank (`—`) for any policy that aggregation does not report.
- `always_human` and `always_abstain` are the two remaining corners of the action space, added here as cost references. They are not Phase 9 baselines and they never abstain *selectively*, so they are context for the cost scale rather than competitive methods.
- The threshold baseline acts on **raw** confidence, which is what the calibration in T2 is meant to improve on; comparing it against the proposed row is the whole claim of the project.
- A lower realised cost here does **not** mean better repairs — T1 holds the repairs fixed across every row. Only the actions differ.
- **False negatives are priced at zero.** Genuine corruptions the detector never flagged (adult: 1,701; breast_cancer: 276; smartfactory: 0) enter no arm of the objective, so every cost number in these tables is a statement about the flagged cells only (docs/cost_semantics.md §3).
- **adult**: proposed 0.00 vs best alternative `always_abstain` 0.00 — tie with `always_abstain`.
- **breast_cancer**: proposed 1,615.00 vs best alternative `confidence_threshold_0.70` 1,720.00 — proposed wins by 105.00.
- **smartfactory**: proposed 14,186.00 vs best alternative `always_human` 14,186.00 — tie with `always_human`.
- **Adult framing (docs/cost_semantics.md §3).** Adult's realised cost measures only residual corruption on the cells the detector flagged; the measured value here is 0.00. A value of 0 must be read as *zero residual-corruption cost on the flagged cells*, never as "zero cost" or "the data is now clean": 1,701 genuinely corrupted cells (income-label flips) were never detected, no action was ever taken on them, and they enter no arm of the objective. The defensible claim is that the calibrated policy declines to act on a detector whose flags are false positives, not that Adult was repaired.
- **adult** p-calibration fallback: All 4081 training labels are incorrect; no calibrator can fit. Calibrated probability collapses to the base rate (0.00). This is the documented Adult negative result.
