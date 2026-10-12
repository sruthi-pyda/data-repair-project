# T2 — Calibration: raw vs logistic vs isotonic vs multivariate

How trustworthy the probability of repair correctness (p) is, with confidence intervals over repeated calibration splits.

**Protocol:** `src.experiments.aggregate.repeated_splits`, 5 repeated calibration splits with `seed = 42` held fixed and only `split_seed` varying, split seeds 1000–1004. Intervals are 95 % bootstrap intervals over splits, written `mean [lo, hi]`. Varying `split_seed` changes the calibrators (and therefore p and q) but not the detections or the mock repairs, so the interval is over calibration splits and nothing else (docs/phase13_plan.md §0).

| dataset | method | Brier  mean [95 % CI] | ECE  mean [95 % CI] | splits fitted | status |
|---|---|---:|---:|---:|---|
| breast_cancer | raw | 0.1947 [0.1873, 0.2008] | 0.1195 [0.1015, 0.1354] | 5/5 | uncalibrated reference |
| breast_cancer | logistic | 0.1955 [0.1850, 0.2057] | 0.1402 [0.1189, 0.1583] | 5/5 | fitted in every split |
| breast_cancer | isotonic | 0.1808 [0.1709, 0.1934] | 0.0696 [0.0393, 0.1052] | 5/5 | fitted in every split |
| breast_cancer | multivariate | 0.1946 [0.1835, 0.2100] | 0.0910 [0.0764, 0.1046] | 5/5 | fitted in every split |

**Notes**

- Brier and ECE are computed on each split's held-out evaluation set; **lower is better** for both. The interval is over splits, so it answers "would a different split have changed the ranking?" — not "how uncertain is the metric on this split?".
- `splits fitted` is the contract's `n_fitted`: how many splits that calibrator actually fitted. A method that never fitted shows `—` for its metrics rather than the raw metrics it copies internally, because printing those reads as a perfect score (CLAUDE.md §5.4).
- Two differences whose intervals overlap are **not** separated by this experiment, however far apart the means are.
- **Repair mode: `real` (Phase 12, `src/repair/llm_repair.py`).**
- `breast_cancer`: model `openai/gpt-oss-120b` · 392 cells · 390 answered by the model · **2 fell back to the mock repairer (0.5%)** · reasons: malformed_response 2 · 84 cache hits · 1 rate limits · 1 key rotations.
- **The intervals in T2 and the cost interval in T4 come from the mock repairer even in real mode**: `repeated_splits` (`src/experiments/aggregate.py`) builds its own `MockRepairer` and takes no repair mode, so those intervals describe the calibration of the mock pipeline. T1, T3 and the predictions CSV use the real repairs. Mixing them would be the dishonest reading.
