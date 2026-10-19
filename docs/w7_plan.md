# Wave 7 plan — measure the ceiling before testing a fifth mechanism

Committed before any W7 code. Supersedes the "M4-prime next" recommendation
in `docs/w6_plan.md` §10, which remains on record.

---

## 1. Why not just build M4-prime

Four mechanisms have now been refuted, and the pattern in *how* they died is
the important part:

| mechanism | killed by |
|---|---|
| M1 likelihood ratio | no income cell in the top-N on adult |
| §2.3 self-supervised injection (W5.3) | **cardinality confound** — a clean 0/1 column scores like a corrupted one |
| M2 contamination fraction (W6.1) | **model-competence confound** — clean-but-hard columns are two-component too; loses to the mean-score baseline |
| (W6.1, pre-fix) | **coverage confound** — the ranking measured whether the detector scored the column at all |

Every one died from a confound discovered **after** running on real data,
because real tables vary in many ways simultaneously. A fifth mechanism
tested the same way will probably die the same way.

So Wave 7 does not propose a mechanism. It measures **how much signal exists
to be captured**, under conditions we control. Then a mechanism can be judged
against the ceiling rather than against our guesses.

---

## 2. W7.0 — integrity fixes first (cheap, independent, do these immediately)

Three defects in how existing results are *presented*, not in the numbers.

1. **`results/w5_cross_domain.md` reports no coverage.** It lists `flights`
   under "where it works" with a lift figure, while
   `results/w6_contamination.md` shows the *conditional* detector scores
   **0 of 7** of its columns. Those are two different detectors, so the
   cross-domain lift is not wrong — but a reader cannot tell which datasets
   our Wave 1 contribution can even run on. Add per-dataset coverage:
   columns scored, columns skipped, and why.
2. **Detection numbers should lead with the leakage-controlled figure.**
   Every detection number currently quoted is clean-profile, and
   `results/w5_leakage_ablation.md` shows the gap is large wherever the rules
   are vocabulary- or range-comparisons (hospital F1 0.9271 → **0.0000**).
   The decision-layer numbers need no restatement — they are identical in
   both arms — so this touches detection reporting only.
   *Note:* `CLAUDE.md` §3 says the preferred resolution is an ablation, "not
   a quiet code change that invalidates already-reported numbers". Promoting
   the detection figure is consistent with that (the ablation is the source,
   no code changes) and the §3 consequence list already requires the
   dirty-profile number beside any clean-profile one. Nothing is deleted.
3. **Adult's "vs do-nothing = 0" belongs up front.**
   `results/w5_complete_costs.md` already computes it: on adult our policy
   achieves *exactly* what having no detector achieves. That is a harsher and
   more important statement than any cost table, and it is currently a cell
   in a row.

---

## 3. W7.1 — the controlled corruption study with a Bayes ceiling

`src/datagen/corruptor.py` (169 tests) has never been used in an experiment.
This is what it was built for.

### 3.1 Why controlled injection, specifically

Taking a *clean* table and corrupting it ourselves means we know the
corruption rate `r`, the injection distribution `P_inject`, the exact cell
positions, and which columns are untouched. That buys three things no real
dirty dataset can:

- **one factor varied at a time**, so a confound cannot hide in the
  difference between two datasets
- **a rare-but-valid control by construction** (§3.2)
- **a computable optimum** (§3.3)

### 3.2 The rare-but-valid control — the decisive design

A value that is *rare but correct* cannot be created by editing a cell:
editing it makes it wrong. So rarity has to come from the column's natural
distribution, and the control is built by **choosing which column to
corrupt**:

- **H** — a *heavy-tailed* column (large fraction of values below 1 %
  frequency), left **completely clean**
- **T** — a *low-tail* column, **corrupted** at rate `r` with type `t`

Then the single decisive comparison is:

    statistic(T)  >  statistic(H) ?

This is exactly the adult `income`-versus-`native_country` failure, but
constructed deliberately and repeatably instead of found once. A statistic
that tracks rarity rather than corruption fails here by design.

Reported alongside: the correlation of each statistic with (a) is-corrupted
and (b) the column's rare-value fraction. If it tracks (b), we have
reproduced the original failure and should say so.

### 3.3 The Bayes-optimal ceiling

Because we generated the corruption, the optimal posterior is available:

    P(corrupt | v, x)  =            r · P_inject(v)
                          ───────────────────────────────────────
                          r · P_inject(v) + (1 − r) · P_clean(v | x)

- `r` and `P_inject` are **known by construction** from the injector.
- `P_clean(v | x)` is estimated from the **clean** frame as an **oracle**.

Ranking cells by this posterior gives the **PR-AUC no method could exceed**
under those conditions. The headline quantity is then:

    captured fraction  =  PR-AUC(our method)  /  PR-AUC(ceiling)

which turns four failures from "we could not do it" into "here is the
information budget, and here is the share each method captures".

**This is an estimated ceiling, not a theorem**, and the report must say so
every time it appears:

- `P_clean(v | x)` carries estimation error; where the oracle model is poor
  the ceiling is *understated*.
- It is conditional on the injection model. A different `P_inject` gives a
  different ceiling.
- Clean data is used as an **oracle** here, non-deployable by construction —
  the same clearly-labelled diagnostic use as the `(ii)` AUC in W5.3. No
  deployable statistic may read it.

### 3.4 Conditions

| factor | levels |
|---|---|
| corruption type | `swap_valid` (the hard valid→valid case), `out_of_range`, `missing` |
| rate `r` | 0.0025 (adult-like), 0.01, 0.05 |
| corrupted columns | 1, and ~half |
| base dataset | adult, breast_cancer, and one text-heavy REIN table |

`swap_valid` at `r = 0.0025` on one column reproduces adult exactly, which
is the regression check that the setup is faithful.

### 3.5 Pre-registered predictions — fixed before running

1. The ceiling is **high** for `missing` and `out_of_range` (the corrupted
   value is off-distribution) and **low** for `swap_valid` at low `r`.
2. Our zero-knowledge conditional detector captures a **large** fraction of
   the ceiling for the first two and a **small** fraction for `swap_valid`.
3. On the §3.2 comparison, every statistic tested so far ranks **H above T**
   for `swap_valid` — i.e. reproduces the failure.

If (1) holds and the `swap_valid` ceiling is near chance, the impossibility
result is established and **M4-prime is not worth building**. If the ceiling
is well above what we achieve, there is signal on the table and M4-prime gets
a target.

---

## 4. W7.2 — M4-prime, conditional

Build only if §3.5(2) shows unclaimed signal. Design and falsification are in
`docs/w6_plan.md` §10, with two methodological corrections adopted from
review:

- the baseline threshold (0.7512) must be **recomputed on whatever evaluation
  units M4-prime is eligible for**, never carried over
- comparison against the baseline must be a **paired per-dataset test**
  (Wilcoxon / sign), not a difference of means-of-AUCs — a mean over six
  datasets weights a 2-column table equally with a 19-column one and carries
  no interval

---

## 5. The paper's claim — pre-registered now, not positioned afterwards

Recorded before W7 results exist, so the framing cannot be fitted to them.

**Claim.** A calibrated probability of correctness combined with explicit
costs produces better *decisions* than acting on raw confidence; the same
confidence yields different actions when consequences change. Verified, and
it survives the leakage ablation in both arms.

**Not claimed.** A competitive detector. Under comparable ignorance a
published method beats ours (`results/w5_raha_comparison.md`), per-column
calibration is prior art (Conformal Data Cleaning, AISTATS 2024), and
calibrating a detector is prior art (HoloDetect 2019).

**Reported as measured limitations**, each with a number:

- rare-but-valid versus corrupted is not separable from one table with no
  labels, at low prevalence (W7.1 quantifies the ceiling)
- the conditional detector does not model most columns of text-heavy tables
  (`flights` 0/7)
- on adult the system equals do-nothing

**If W7.1 shows a near-chance ceiling**, the impossibility result becomes the
paper's spine and the decision layer its constructive contribution. That is a
coherent paper, and it is decided here rather than after the fact.

---

## 6. Tasks

- [ ] Commit this plan before any W7 code
- [ ] **W7.0a** coverage columns in `results/w5_cross_domain.md`
- [ ] **W7.0b** detection reporting leads with the dirty-profile figure
- [ ] **W7.0c** adult's "vs do-nothing = 0" surfaced in `RESULTS.md`
- [ ] **W7.1a** injection harness over the §3.4 grid, on clean bases
- [ ] **W7.1b** the heavy-tail-clean vs corrupted-low-tail comparison (§3.2)
- [ ] **W7.1c** the Bayes ceiling and the captured fraction (§3.3)
- [ ] **W7.1d** evaluate every statistic we already have against the ceiling
- [ ] **W7.2** M4-prime, only if §3.5(2) leaves signal unclaimed
