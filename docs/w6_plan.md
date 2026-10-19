# Wave 6 plan — rare-vs-wrong and column localisation

Written after Wave 5 closed. Wave 5's two headline results were negative
(`results/w5_raha_leakage_check.md`, `results/w5_column_triage_control.md`),
so this document is about what is actually left to try, and why — not a fresh
brainstorm. `docs/column_triage_research.md` already surveyed the field and
ranked four candidate mechanisms **M1–M4** before any of them was tested. Two
are now dead. This plan is mostly bookkeeping on that ranking plus one
unifying observation that changes which family to pick.

---

## 1. The two problems, stated as quantities

**Problem A — rare vs wrong.** A rare-but-valid value and a corrupted value
are *both* low-probability under the table's own distribution. So **any
statistic that is a monotone function of `P(x)` alone cannot separate them.**
That is an identifiability statement, not an engineering shortfall. Measured
instance: on adult, `native_country` dominates the top of our ranking because
a rare country genuinely *is* improbable — and completely clean.

**Problem B — column localisation.** Measured, and the three numbers settle
the diagnosis:

| what is scored | PR-AUC |
|---|---:|
| `income` alone | 0.6873 |
| all 15 columns | 0.0123 |
| all 15, other 14 forced to zero | **0.6873** |

The first and third being **identical** proves this is **displacement, not
dilution**. Inside `income` our ranking is already good. The failure is that
`native_country`'s rare-value scores *outrank* `income`'s corrupted-value
scores — i.e. **per-cell scores from different columns are not on a comparable
scale.**

So Problem B is not "we need a better detector". It is a cross-column
comparability problem.

---

## 2. The unifying observation: aggregate first, then localise

Everything we have tried is a **per-cell** statistic that we then aggregate to
rank columns. That has two fatal properties:

1. **Power.** Per-cell detection on adult fights a prevalence of 0.25 %.
   A *column-level* quantity for the same column is estimated from **45,222
   cells**. The statistical power difference is enormous and it is free.
2. **Comparability.** A per-cell score has column-specific units. A
   **proportion** — "what fraction of this column is anomalous" — is
   dimensionless and therefore comparable across columns *by construction*,
   which is exactly what §1 Problem B says we lack.

This is why score-aggregation failed and why the remaining candidates are all
column-level aggregate statistics. It also explains W5.3's failure mode
precisely: `d_c` is a per-column scalar, but it is not dimensionless — it
scales with the column's cardinality, which is what the control measured.

**Rule for Wave 6: prefer a dimensionless, population-level, per-column
quantity over any aggregate of per-cell scores.**

---

## 3. Scoreboard of the four pre-ranked mechanisms

| | mechanism | status |
|---|---|---|
| **M1** | Likelihood ratio `P(x) / P(x\|row)` instead of density | **REFUTED** — `experiments/w4_ratio_test.py`. Zero income cells in the top-N on adult. |
| **§2.3** | Self-supervised corruption injection, per column | **REFUTED** — W5.3. Its one perfect score was a cardinality confound; `swap_valid`, the type matching adult's real corruption, scored 0.3077, *below chance*, in the wrong direction. |
| **M2** | **Per-column contamination fraction `pi_hat_c`** | **UNTESTED — ranked #1 before any of this began.** |
| **M3** | Spend a label budget on columns (vs ED2's Column Selector) | Untested, and **near-certain to fail on adult by its own derivation**: 20 random labels yield 0.05 expected positives, and ED2's selector needs ≥2 known errors *per column* to start. |
| **M4** | Column-level two-sample test: observed column vs its model-implied conditional | Untested, ranked #4. |

M2 and M4 are both exactly the dimensionless population-level family §2
argues for. M2 is the one to run.

---

## 4. W6.1 — M2, per-column contamination fraction (the recommended route)

**Mechanism.** For each column `c`, take the conditional-surprise scores we
*already compute* for all its cells and estimate the **mixture weight of the
anomalous component**, `pi_hat_c` — Gao & Tan's exponential+Gaussian mixture
(ICDM 2006, available as `pythresh.thresholds.mixmod`) or BBE (Garg et al.,
NeurIPS 2021). Rank columns by `pi_hat_c`.

**Why it addresses Problem A directly, where a density cannot.** A
uniformly-rare-but-clean column has a **low** anomalous mixture weight,
because its rare values are not *separable* from its own distribution — they
are the tail of one component, not a second component. A corrupted column has
genuine two-component structure. Separability, not improbability, is the
discriminator. That is the whole point.

**Why it addresses Problem B.** `pi_hat_c` is a proportion. Dimensionless,
comparable across columns, no normalisation needed.

**Why it is cheap.** The scores already exist. `pythresh` is one `pip
install`. And it drops straight into machinery we already shipped:
`detection_calibrator.py` learns `q` per detection method with a default of
1.0; `pi_hat_c` gives it a **per-column prior** instead. So a win here
improves the decision layer, which is the part of the project that survived
Wave 5.

**Pre-registered falsification — one number.** On adult:

    pi_hat(income) > pi_hat(native_country) ?

If not, the mechanism is dead and we say so. True rates for calibration:
`income` 1,701/45,222 = 3.8 % of that column's cells; **every** other column
0 %.

**Three ways it can fail, all pre-registered:**

1. **Identifiability at extreme prevalence.** On smartfactory the anomalous
   component *is* the majority, so a two-component fit may be label-swapped or
   unidentifiable.
2. **Non-separability.** BBE's guarantee needs a model that cleanly separates
   some positives. Our conditional model tops out near 85 % accuracy on
   `income` and the flipped cells are **not** cleanly separable, so the
   estimate may collapse toward zero or toward the model's error rate.
3. **The failure mode we already have, reproduced.** `native_country`'s long
   rare-value tail may itself fit a two-component mixture and yield a *false
   high* `pi_hat` — which would reproduce the current failure exactly. This is
   the most likely of the three and the falsification test above is aimed
   straight at it.

---

## 5. W6.0 — the powered re-run, and it must come first

`results/w5_column_sparsity.md` records a scoping error worth not repeating:
W5.3 was run on **adult and breast_cancer, the worst two datasets available**.
Adult's best-possible one-sided p is 1/15 = 0.067, so no adult-only result can
reach p < 0.05, and breast_cancer has no clean column at all.

Column triage is well posed on **eight** of the eleven shipped datasets:

| dataset | corrupted / total columns | cell prevalence |
|---|---|---:|
| `tax` | 9 / 15 | 0.00099 |
| `restaurants` | 10 / 16 | 0.00121 |
| `soccer` | 4 / 10 | 0.01565 |
| `hospital` | 17 / 20 | 0.02545 |
| `movies` | 11 / 17 | 0.05577 |
| `rayyan` | 8 / 11 | 0.08618 |
| `beers` | 4 / 11 | 0.12663 |
| `flights` | 4 / 7 | 0.29582 |

`tax`, `restaurants` and `soccer` are the regime that actually matters —
sparse column corruption at low prevalence — which is what adult was chosen
to represent with a sample size of one.

**So every W6 mechanism is evaluated across these eight, pooled, not on
adult.** Any mechanism judged on adult alone is judged at a power ceiling of
p = 0.067.

---

## 6. Honest novelty position — read before claiming anything

`docs/literature.md` and §2.1b of the triage doc already establish:

- **Conformal Data Cleaning** (Jäger & Bießmann, AISTATS 2024) pairs
  per-column conditional models with calibrated uncertainty. Per-column
  calibration is **prior art**, so "we calibrate per column" is not a
  contribution.
- **ED2's Column Selector** is prior art for spending labels on columns (M3).
- Calibrating a detector is not novel either (HoloDetect, 2019).

What Wave 5 did *not* damage is the decision layer: zero worst-case regret
holds under **both** profile arms. So the defensible position is **not** to
compete on detection F1, where a published method already beats us under
comparable ignorance. It is:

> Column triage reframed as **budget allocation under an explicit cost
> model** — not "which column is corrupted" but "where does the next unit of
> review budget reduce expected cost most". Published detectors output a flag
> *set* and make no decision at all, so this is a question they do not answer.

M2 fits this framing exactly, because `pi_hat_c` enters as the per-column
prior on `q`, and `q` is what the three-way decision rule consumes.

---

## 7. Tasks

- [ ] Commit this plan before any W6 code (protocol §8b.2)
- [ ] **W6.0** re-run the W5.3 probe across the eight testable datasets,
      pooled, so the refutation is properly powered rather than adult-only
- [ ] **W6.1** M2: `pi_hat_c` per column, with the one-number falsification
      test on adult and the pooled evaluation on the eight
- [ ] **W6.2** if M2 survives: feed `pi_hat_c` as the per-column prior for
      `q` and measure the *decision-layer* effect (realised cost), which is
      the only metric Wave 5 left standing
- [ ] **W6.3** only if M2 dies: M4, the column-level two-sample test
- [ ] Decide explicitly whether M3 is worth running given its own derivation
      predicts failure on adult
- [ ] The controlled corruption study remains unbuilt — `src/datagen/`
      (169 tests) has never been used in an experiment. It is the way to vary
      column sparsity and prevalence *deliberately* rather than taking
      whatever the eight datasets happen to have.
