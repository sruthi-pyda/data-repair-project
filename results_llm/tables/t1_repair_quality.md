# T1 — Repair quality

Detection quality and raw repair quality per dataset, before any cost-sensitive decision is taken.

**Protocol:** one representative pipeline run, `seed = 42`, `split_seed = 42`, detections no cap (full dataset). These are point estimates from a single calibration split, not interval estimates.

| dataset | detections (all) | decided cells | precision | recall | F1 | top-1 repair accuracy | genuine corruptions (full dataset) | detected genuine (TP) | false negatives |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| breast_cancer | 392 | 355 | 1.0000 | 0.5626 | 0.7201 | 0.7099 | 631 | 355 | 276 |

**Notes**

- `detections (all)` is every cell the Phase 3 detector flagged; `decided cells` is what the policy actually acted on (ID-like columns in `SKIP_COLUMNS` are excluded, as in every other table).
- Precision / recall / F1 are **detection** metrics over the decided cells against ground-truth corruption. Recall is low wherever the corruption is undetectable from dirty data alone.
- `top-1 repair accuracy` is a property of the repairer, not of a policy: the share of decided cells whose top candidate equals the clean value, counted regardless of the action taken.
- `genuine corruptions (full dataset)` counts every cell where dirty ≠ clean anywhere in the dataset, including ID-like columns, so `false negatives = genuine corruptions − detected genuine`.
- **False negatives are priced at zero.** Genuine corruptions the detector never flagged (breast_cancer: 276) enter no arm of the objective, so every cost number in these tables is a statement about the flagged cells only (docs/cost_semantics.md §3).
- **Repair mode: `real` (Phase 12, `src/repair/llm_repair.py`).**
- `breast_cancer`: model `openai/gpt-oss-120b` · 392 cells · 390 answered by the model · **2 fell back to the mock repairer (0.5%)** · reasons: malformed_response 2 · 84 cache hits · 1 rate limits · 1 key rotations.
- **The intervals in T2 and the cost interval in T4 come from the mock repairer even in real mode**: `repeated_splits` (`src/experiments/aggregate.py`) builds its own `MockRepairer` and takes no repair mode, so those intervals describe the calibration of the mock pipeline. T1, T3 and the predictions CSV use the real repairs. Mixing them would be the dishonest reading.
