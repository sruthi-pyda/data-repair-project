# W5.3 — Column triage by self-supervised injection (PRE-REGISTERED)

Route W5.3 of `docs/research_program.md` §11 — reviewer **gap 2**, the
hardest open problem in the project and the one the measurements keep
pointing at:

> Our detector is good at finding the odd cell **inside** a column.
> It has no idea **which column** is worth looking at.

`docs/research_program.md` §8 measured the sharp version of this on adult:
of the 5,000 cells our detector is most suspicious about, **zero** are in
the tampered `income` column. The top is `native_country`, where a rare
country is genuinely improbable and completely clean. The failure is
**displacement, not dilution** — scoring all 15 columns gives 0.0123 while
scoring `income` alone gives 0.6873, and zeroing the other 14 gives
*exactly* 0.6873, which proves the others are shouting over the top rather
than watering anything down.

**This document is written and committed before the experiment runs.** Its
falsification conditions are fixed in advance so the result cannot be
reinterpreted after the fact — the same discipline that correctly killed the
likelihood-ratio idea in W4.1.

---

## 1. What we need, stated as a quantity

A per-column score `t_c`, computed from the **dirty table alone with zero
labels**, that ranks columns by whether they contain corruption.

`t_c` must be computable without ground truth. Ground truth is used *only*
to score the resulting ranking, after it exists — the §3 invariant.

## 2. Why the obvious scores cannot work

Already measured, all failed: column-mean anomaly score, rarity,
cardinality, predictability. The reason is structural, from
`docs/literature.md`:

> **No published system separates rare-but-valid from corrupted using one
> table and no labels.** Every deployed method imports a reference — a
> column corpus, a table corpus, labels, or a cost model.

So a new heuristic over the same marginal statistics is not going to work,
and we should stop proposing them. What is needed is a **reference**. The
one reference that is self-contained is corruption we inject ourselves,
because we know where it is.

## 3. The probe

For each column `c`:

1. Take `dirty_df`. Inject corruption of type `T` at rate `r` into **column
   `c` only** (`protected_columns` = every other column). The injector
   returns the exact positions, so the planted cells are known labels that
   cost nothing and leak nothing.
2. Fit the detector on the injected frame. It is unsupervised; it never sees
   the planted positions.
3. Score the cells of column `c`. Compute ROC-AUC of separating the
   **planted** cells from the non-planted cells of that column.
   Call it `d_c(T, r)` — the column's **self-detectability**.
4. `t_c = -d_c`.

## 4. The mechanism, and why the sign is what it is

`d_c` asks: *how well can I find corruption I planted in this column?*

The detector's model of column `c` is fitted on `c`'s own dirty values. If
`c` is **already corrupted**, that model is fitted on polluted targets, so
it is a worse model of the clean process, so freshly planted corruption is
harder to separate from the spread the column already has. Noise in the fit
target degrades detectability.

**Hypothesis H1: already-corrupted columns have LOWER self-detectability.**
Rank columns by ascending `d_c`; the corrupted ones should come first.

This is a claim about the *sign* of an effect, and the sign is an
assumption, not a theorem — exactly the status W4.1 flagged. The honest
competing possibility is not an opposite sign but **no measurable effect**:
`income` holds 1,701 corrupted cells out of 45,222, so its model is only
3.8 % polluted, and a 3.8 % pollution may simply not move an AUC. If so the
probe is dead and we say so.

## 5. Anti-circularity control — the thing that would invalidate this

If the probe type `T` equals the corruption type actually in the dataset,
the probe could succeed for a trivial reason: we would be planting the same
corruption that is already there and detecting *that*, which tells us
nothing we could use on data whose corruption type we do not know.

Control: adult's true corruption is **valid-to-valid label swap**. The probe
is therefore run with types that are structurally different
(`out_of_range`, `format`, `missing`) **and** with `swap_valid`, and every
type is reported separately.

**If the probe works only when `T` matches the dataset's true corruption
type, it is declared circular and reported as a negative result.** In
deployment the true type is unknown, so a probe that needs to match it is
worthless.

## 5b. The negative-set confound — pre-registered, and why both are reported

The probe plants cells in a column and asks how well they separate from the
column's **other** cells. If the column is already corrupted, those other
cells contain **genuine** corruptions, which the detector scores high. That
drags the AUC down for a reason that has nothing to do with H1: the negative
class is contaminated, not the model degraded.

This matters enough to fix the interpretation in advance, because it is a
second mechanism that would make H1 appear to pass:

- **(i) deployable AUC** — negatives are *all* non-planted cells in the
  column. Uses **zero labels**. This is the only version that could ever run
  in deployment, so **this is the version the falsification conditions in §6
  are evaluated on.**
- **(ii) mechanism AUC** — negatives restricted to cells known to be
  genuinely clean. **Uses labels, is therefore NOT deployable**, and exists
  only to say which mechanism is driving the result.

Reading the pair:

| (i) | (ii) | conclusion |
|---|---|---|
| passes | passes | H1 as stated: the column's model really is degraded |
| passes | fails | the probe **works but for a different reason** — negative-set contamination. Still a usable triage signal; H1 itself is unsupported and must not be claimed |
| fails | — | the probe does not work |

A deployable probe that works via contamination is still a deployable probe,
so (i) governs utility. But the paper may only claim the *mechanism* H1
describes if (ii) also holds. Conflating the two would be the easiest way to
overclaim here, which is why the distinction is written down before the run.

## 6. Falsification conditions — FIXED BEFORE RUNNING

Ground-truth corrupted columns per dataset are derived from `cell_labels`
(evaluation-only use, permitted by §3). Chance for a single corrupted column
among `n` columns is a uniform rank, expectation `(n+1)/2`.

Headline metric is the **column-level AUC**: rank all columns by `t` and
measure how well that separates truly-corrupted columns from clean ones.
Chance is 0.5. This handles datasets with several corrupted columns, which a
single rank does not. Evaluated on the **deployable** AUC (i) of §5b.

- **F1** — refuted on a dataset if the column-level AUC is **<= 0.5**,
  i.e. no better than ordering the columns at random.
- **F2** — refuted on adult if `income` is **not in the top 5** of 15
  columns. (Chance of top-5 is 1/3, so clearing F2 is weak evidence on its
  own; it is a necessary condition, not a sufficient one.)
- **F3** — declared **circular** if F1/F2 pass for `T = swap_valid` but fail
  for every structurally-different `T`.
- **F4** — refuted as *useless* if the probe does not beat the
  **column-mean-score baseline**, i.e. ranking columns by their mean raw
  detector score. That baseline is the thing already known to fail
  (`native_country` first on adult), so failing to beat it means the probe
  adds nothing.

A single corrupted column gives an exact one-sided p-value `rank / n`, which
is reported rather than an asterisk.

## 7. What a pass would and would not license

A pass licenses exactly one claim: *corruption can be localised to a column
by measuring how hard it is to detect corruption we plant there ourselves.*

It would **not** license "we solved detection". The per-cell ranking inside
the column is a separate problem that Wave 1 already addressed. And with
three datasets, two of which have degenerate corruption structure, a pass is
suggestive, not established.

## 8. Tasks

- [ ] Commit this plan before writing any experiment code
- [ ] `experiments/w5_column_triage.py` — the probe, all types x rates
- [ ] Ground-truth corrupted-column census per dataset
- [ ] Column-mean-score baseline (F4)
- [ ] Evaluate F1-F4, write `results/w5_column_triage.md`
- [ ] Record the verdict in `docs/research_program.md`, pass or fail
