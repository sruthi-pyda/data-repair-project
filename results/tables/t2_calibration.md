# T2 — Calibration: raw vs logistic vs isotonic vs multivariate

How trustworthy the probability of repair correctness (p) is, with confidence intervals over repeated calibration splits.

**Protocol:** `src.experiments.aggregate.repeated_splits`, 20 repeated calibration splits with `seed = 42` held fixed and only `split_seed` varying, split seeds 1000–1019. Intervals are 95 % bootstrap intervals over splits, written `mean [lo, hi]`. Varying `split_seed` changes the calibrators (and therefore p and q) but not the detections or the mock repairs, so the interval is over calibration splits and nothing else (docs/phase13_plan.md §0).

| dataset | method | Brier  mean [95 % CI] | ECE  mean [95 % CI] | splits fitted | status |
|---|---|---:|---:|---:|---|
| adult | raw | 0.4141 [0.4139, 0.4144] | 0.6429 [0.6427, 0.6431] | 20/20 | uncalibrated reference |
| adult | logistic | — | — | 0/20 | ⚠ never fitted — fallback in every split |
| adult | isotonic | — | — | 0/20 | ⚠ never fitted — fallback in every split |
| adult | multivariate | — | — | 0/20 | ⚠ never fitted — fallback in every split |
| breast_cancer | raw | 0.1982 [0.1936, 0.2029] | 0.1074 [0.0973, 0.1177] | 20/20 | uncalibrated reference |
| breast_cancer | logistic | 0.1979 [0.1920, 0.2040] | 0.1299 [0.1177, 0.1414] | 20/20 | fitted in every split |
| breast_cancer | isotonic | 0.1842 [0.1787, 0.1900] | 0.0652 [0.0527, 0.0787] | 20/20 | fitted in every split |
| breast_cancer | multivariate | 0.1968 [0.1910, 0.2029] | 0.0846 [0.0733, 0.0964] | 20/20 | fitted in every split |
| smartfactory | raw | 0.2732 [0.2727, 0.2737] | 0.2925 [0.2901, 0.2949] | 20/20 | uncalibrated reference |
| smartfactory | logistic | 0.1691 [0.1674, 0.1708] | 0.0577 [0.0556, 0.0601] | 20/20 | fitted in every split |
| smartfactory | isotonic | 0.1632 [0.1615, 0.1649] | 0.0127 [0.0096, 0.0158] | 20/20 | fitted in every split |
| smartfactory | multivariate | 0.1883 [0.1872, 0.1894] | 0.1556 [0.1519, 0.1595] | 20/20 | fitted in every split |

**Notes**

- Brier and ECE are computed on each split's held-out evaluation set; **lower is better** for both. The interval is over splits, so it answers "would a different split have changed the ranking?" — not "how uncertain is the metric on this split?".
- `splits fitted` is the contract's `n_fitted`: how many splits that calibrator actually fitted. A method that never fitted shows `—` for its metrics rather than the raw metrics it copies internally, because printing those reads as a perfect score (CLAUDE.md §5.4).
- Two differences whose intervals overlap are **not** separated by this experiment, however far apart the means are.
- **adult** p-calibration fallback: All 4081 training labels are incorrect; no calibrator can fit. Calibrated probability collapses to the base rate (0.00). This is the documented Adult negative result.
