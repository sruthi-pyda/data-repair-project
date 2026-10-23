# Wave 11 plan — fix the variance reporting

Committed before any W11 code. This corrects results already published, so
it comes before any new experiment is built on top of them.

## 1. The defect

`results/w9b_text_fallback.md` reports every figure as "mean [min, max] over
5 splits" and ticks requirement 6, reproducibility. **38 of its 76 intervals
are zero-width.** Not because the method is stable — because the replicates
are not replicates.

Measured row counts against the 4,000-row cap:

| dataset | rows | do the 5 "splits" differ? |
|---|---:|---|
| breast_cancer | 699 | **no** |
| hospital | 1,000 | **no** |
| rayyan | 1,000 | **no** |
| flights | 2,376 | **no** |
| beers | 2,410 | **no** |
| movies | 7,390 | yes |
| smartfactory | 23,645 | yes |
| restaurants | 28,787 | yes |
| adult | 45,222 | yes |
| soccer, tax | 200,000 | yes |

Five of eleven datasets sit at or below the cap, so `run_split` subsamples
nothing and every replicate is the **same rows**. The detector seed is fixed
at `BASE_SEED` besides, so no model randomness varies either. Those five
datasets are five identical runs.

**Consequences, stated plainly:**

1. **`flights` 0.8360 — the headline detector result — has no uncertainty
   estimate at all.** Neither does `hospital` 0.4974.
2. A zero-width interval *reads* as perfect stability. This is a misleading
   presentation, and it is mine.
3. Requirement 6 was ticked on a protocol that does not deliver what it
   claims.

The same pattern affects `results/w8a_cacm.md` and `results/w8d_excess.md`,
which use an 8,000-row cap — there `breast_cancer`, `hospital`, `rayyan`,
`flights` and `beers` again do not resample. `w9c` and `w10` vary their
**label draws**, which is genuine variation, so they are not affected the
same way.

## 2. The fix — two sources of variation, because they measure different things

**A. Vary the detector seed per replicate.** `seed = BASE_SEED + k` for the
conditional detector as well as the row subsample, so model randomness
contributes. Cheap, and it is the variation a reader assumes is already
there.

**B. Cell-level bootstrap on the metric.** Resample the evaluated
`(label, score)` pairs with replacement, recompute the metric, and take
percentile bounds. This gives a genuine uncertainty estimate **even when the
rows cannot be resampled**, which is exactly the case that broke.

Both are needed. (A) captures algorithmic randomness; (B) captures sampling
uncertainty of the metric itself. Reporting only (A) on a small dataset
reproduces the current defect.

The existing `bootstrap_interval` in `src/experiments/aggregate.py`
resamples **per-split values**, so with five identical values it still
returns zero width. A new `bootstrap_metric(y, scores, fn, …)` is needed and
is additive — no existing function changes.

## 3. What gets reported afterwards

    PR-AUC  0.8360  95% CI [l, u]   (bootstrap over cells, B resamples)
                    seeds: mean ± range over R detector seeds

Never a bare `[min, max]` over replicates that may be identical. Where an
interval is genuinely degenerate (a metric that really is 1.0 on every
resample), say so in words rather than printing a zero-width range.

## 4. Acceptance conditions — fixed now

1. **No zero-width interval survives** unless the metric is provably
   constant, and then it is labelled rather than printed as a range.
2. **`flights` and `hospital` get real CIs.** These are the headline numbers
   and currently have none.
3. **The conclusions either hold or are restated.** If a CI now overlaps the
   baseline where the point estimate did not, the requirement-2 and
   requirement-4 verdicts in `w9b` change accordingly, and the change is
   recorded rather than quietly absorbed.
4. **Affected files carry a correction banner** pointing at the restated
   numbers, in the same way `w9c` already does for the W10 bias fix.
5. The bootstrap is deterministic under a fixed seed, and tested.

## 5. Scope

This changes **how uncertainty is reported**, not the detector, the
mechanism or any point estimate. A point estimate that moves can only move
because the detector seed now varies, and any such move is reported as a
finding.

Files: `src/experiments/aggregate.py` (one added function),
`tests/test_aggregate_bootstrap.py`, `experiments/w9b_text_fallback.py`
(reporting), `results/w9b_text_fallback.md` (regenerated).

## 6. Tasks

- [ ] Commit this plan
- [ ] `bootstrap_metric` + tests
- [ ] Vary the detector seed per replicate in `w9b`
- [ ] Re-run and restate; banner the affected files
- [ ] Record whether any verdict changed
