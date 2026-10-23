# Wave 10 plan — fix threshold selection, the one loss we can convert

Committed before any W10 code.

## 1. The problem, precisely

`results/w9c_raha_rematch.md` decomposed our three losses to Raha, and one of
them is not a detector deficit:

| dataset | ours @20 labels | **our oracle bound** | Raha @20 labels |
|---|---:|---:|---:|
| hospital | 0.3849 | **0.5054** | 0.4000 |
| beers | 0.5430 | 0.6491 | 0.7988 |
| rayyan | 0.1950 | 0.3035 | 0.6457 |

On **hospital our oracle F1 (0.5054) exceeds Raha's 0.4000.** The detector is
already good enough to win; we cannot *find* the threshold with 20 labels.
beers and rayyan lose even at their oracle, so those are genuine detection
deficits and out of scope here.

**So this is a threshold-selection problem, and it sits squarely in the
decision layer — the part of this project that already works.**

## 2. Why the current method fails

The W9.3 arm samples 20 cells **uniformly at random** and picks the threshold
maximising F1 on them. At hospital's prevalence of 0.0255, twenty uniform
cells contain **0.5 positives in expectation**. Most draws therefore contain
zero or one positive, and a threshold fitted on that is noise. The budget is
not too small — it is **spent in the wrong place**.

This is also why Raha does better with the same budget: it samples
*informatively*, not uniformly.

## 3. The fix — spend the budget where the positives are

Partition cells into **score strata** (defined by the scores alone, so no
labels are consumed to build them), then allocate the budget across strata
and estimate precision within each:

    strata by score rank:  top 0.1% | 0.1-1% | 1-5% | 5-20% | 20-100%

For each stratum `i` with size `n_i`, sample `b_i` cells, reveal them, and
estimate

    precision_i  = (positives found + 0.5) / (sampled + 1)      Laplace-smoothed
    positives_i  = n_i * precision_i

A candidate threshold is a stratum boundary. For the boundary after stratum
`j`:

    flagged  = sum of n_i for i <= j
    TP_est   = sum of positives_i for i <= j
    FN_est   = (total estimated positives) - TP_est

from which an estimated F1 — or an estimated **cost** — follows. Two
selection rules are compared, because they optimise different things:

- `stratified_f1` — pick the boundary maximising estimated F1. Comparable
  with Raha, which reports F1.
- `stratified_cost` — pick the boundary minimising
  `flagged * C_review + FN_est * C_miss`. This is the project's own
  objective, and it is the honest one for a system whose purpose is
  decisions rather than rankings.

**Why this should beat uniform sampling**: a precision estimate per stratum is
a handful of scalars with low variance, and the strata where positives
actually live get sampled deliberately instead of by luck. Uniform sampling
spends ~19 of its 20 labels confirming that low-scoring cells are clean,
which we already believed.

## 4. Falsification — fixed now

1. **Beats uniform** at the same budget, on hospital, by a paired test over
   label draws. Not a single draw.
2. **Converts the loss**: hospital F1 must exceed **Raha's 0.4000**. This is
   the whole point and it is pass/fail.
3. **Does not break the others.** breast_cancer must stay a win; beers and
   rayyan must not get *worse* than their uniform numbers (they will still
   lose, since even their oracles do).
4. **Does not beat its own oracle.** A selection rule scoring above the
   oracle best-F1 means a bug, not a discovery.
5. **Budget honesty.** Exactly `budget` labels revealed, counted and
   asserted. Strata are built from scores only.
6. **Reproducibility.** Mean [min, max] over several label draws, never one.

## 5. Scope

This changes *which threshold we pick*, not the detector. No detection number
moves; only the operating point does. If it works, the claim is narrow and
accurate: **given a detector and a tiny label budget, spending the budget by
score stratum converts a loss into a win on one dataset.**

Files: `src/decision/threshold.py`, `tests/test_threshold.py`,
`experiments/w10_threshold.py`, `results/w10_threshold.md`.

## 6. Tasks

- [ ] Commit this plan before any W10 code
- [ ] `src/decision/threshold.py` + tests
- [ ] `experiments/w10_threshold.py` over the four Raha datasets
- [ ] Record the verdict against §4, pass or fail
