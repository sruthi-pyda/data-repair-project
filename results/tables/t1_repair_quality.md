# T1 — Repair quality

Detection quality and raw repair quality per dataset, before any cost-sensitive decision is taken.

**Protocol:** one representative pipeline run, `seed = 42`, `split_seed = 42`, detections no cap (full dataset). These are point estimates from a single calibration split, not interval estimates.

| dataset | detections (all) | decided cells | precision | recall | F1 | top-1 repair accuracy | genuine corruptions (full dataset) | detected genuine (TP) | false negatives |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adult | 5,829 | 5,829 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1,701 | 0 | 1,701 |
| breast_cancer | 392 | 355 | 1.0000 | 0.5626 | 0.7201 | 0.6732 | 631 | 355 | 276 |
| smartfactory | 7,093 | 7,093 | 1.0000 | 1.0000 | 1.0000 | 0.2879 | 7,093 | 7,093 | 0 |

**Notes**

- `detections (all)` is every cell the Phase 3 detector flagged; `decided cells` is what the policy actually acted on (ID-like columns in `SKIP_COLUMNS` are excluded, as in every other table).
- Precision / recall / F1 are **detection** metrics over the decided cells against ground-truth corruption. Recall is low wherever the corruption is undetectable from dirty data alone.
- `top-1 repair accuracy` is a property of the repairer, not of a policy: the share of decided cells whose top candidate equals the clean value, counted regardless of the action taken.
- `genuine corruptions (full dataset)` counts every cell where dirty ≠ clean anywhere in the dataset, including ID-like columns, so `false negatives = genuine corruptions − detected genuine`.
- **False negatives are priced at zero.** Genuine corruptions the detector never flagged (adult: 1,701; breast_cancer: 276; smartfactory: 0) enter no arm of the objective, so every cost number in these tables is a statement about the flagged cells only (docs/cost_semantics.md §3).
- Repairs are produced by `src/repair/mock_repair.py` (difflib similarity + frequency weighting + median/quartile imputation + seeded noise). **No LLM was called.** Phase 12 is not implemented.
- **adult** p-calibration fallback: All 4081 training labels are incorrect; no calibrator can fit. Calibrated probability collapses to the base rate (0.00). This is the documented Adult negative result.
