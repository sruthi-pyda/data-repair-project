# Wave 9 plan — close the triage story, then fix the coverage collapse

Committed before any W9 code. Two tasks, independent, in this order.

Prompted by a question worth recording because its premise was wrong in an
instructive way: *"if we cannot identify the column, why not just run the
detector on every column?"* We already do. `Detector` iterates every
`profile.column_names` and there is no column-selection gate anywhere in
`src/pipeline/runner.py`. **Adult's whole-table PR-AUC of 0.0025 *is* the
run-on-every-column number.** Column triage was never a pipeline stage — it
was a diagnostic question about why running everywhere fails.

The measured answer to that question:

| what is scored | PR-AUC |
|---|---:|
| `income` alone | 0.6873 |
| all 15 columns | 0.0123 |
| all 15, other 14 forced to zero | **0.6873** |

First and third identical. Running everywhere does not fail from a selection
error; it fails because `native_country`'s rare-but-clean cells **outrank**
`income`'s corrupted ones. Extra columns contribute false positives that
displace the true positives from the top of the ranking.

---

## W9.1 — What would perfect column triage be worth?

Five mechanisms have failed to produce a usable triage statistic (M1, W5.3,
M2, CACM, Excess Surprise), and `docs/w8_plan.md` §8 explains why with the
nuisance hierarchy. What has never been computed is the **headroom**: the
value of triage if it were perfect.

**Definition.** Two variants, because they answer different questions and
only one keeps the cell universe fixed:

- **(a) masked, primary** — score every cell, then force the score to zero on
  every column that is genuinely clean. Evaluate over **all** cells.
  The cell universe and prevalence are unchanged, so this is directly
  comparable with the published whole-table number. This is the variant the
  0.6873 measurement above used.
- **(b) restricted, secondary** — evaluate only over cells in genuinely
  corrupted columns. Prevalence changes, so it is **not** comparable with the
  whole-table figure and must never be quoted beside it without that note.

**Oracle status.** Both read `clean_df` to decide which columns are corrupted,
so both are **non-deployable by construction** — the same clearly-labelled
diagnostic use as the `(ii)` AUC in W5.3 and the ceiling in W7.1. The point is
to size the prize, never to flag cells.

**What it settles.** One table that simultaneously justifies why five waves
chased triage and retires the programme: the headroom is real and large, and
no obtainable statistic captures any of it.

Report per dataset: whole-table PR-AUC, masked PR-AUC, the ratio, and the
count of corrupted versus clean columns. Both detectors (rule and
conditional), since they fail differently.

Files: `experiments/w9a_triage_value.py`, `results/w9a_triage_value.md`.

---

## W9.2 — A fallback detector for the columns we cannot model

The method's most defensible weakness, and the first thing a reviewer hits by
running our code. From `results/w7_coverage.md`:

| dataset | columns the conditional detector can model |
|---|---|
| `flights` | **0 of 7** |
| `movies` | 2 of 17 |
| `beers` | 3 of 11 |
| `rayyan` | 3 of 11 |
| `restaurants` | 6 of 16 |
| `hospital` | 8 of 20 |

`ConditionalDetector` skips `high_cardinality_non_numeric` columns because it
models a column as a **category** and a column with hundreds of distinct
strings has no usable class structure. Real dirty-data benchmarks are mostly
text, so this is a scope limit on the whole Wave 1 contribution.

### The idea: model the string, not the category

A high-cardinality text column still has structure — just not categorical
structure. Three signals, each cheap and each catching a different corruption
type:

1. **Pattern signature.** Map each value to a shape (`"ABC-123"` → `"AAA-999"`)
   and score by how rare its shape is within the column. Catches format
   corruption and parse breaks, which is what `dBoost` and `FAHES` exploit.
2. **Character n-gram likelihood.** Fit a character-level model on the
   column's own values; score a value by its mean per-character log
   likelihood. Catches typos and garbage strings.
3. **Distance to the nearest frequent value.** A typo is *close to* a valid
   value without equalling it; a legitimately rare value is usually far from
   everything. Catches typos specifically, and is the signal most likely to
   separate rare-but-valid from corrupted in text.

All three are computed from the **dirty column alone** — no profile, no clean
data, no labels.

### Falsification — fixed now

1. **Coverage must actually increase**, reported as columns scored before and
   after. Necessary but trivial, so it licenses nothing on its own.
2. **It must beat chance on the newly-covered columns.** PR-AUC against the
   prevalence of those columns, which is the only honest reference at this
   imbalance.
3. **It must not degrade the whole-table number** on datasets where the
   conditional detector already has good coverage (`adult` 15/15,
   `smartfactory` 19/19). A fallback that helps text columns by wrecking
   numeric ones is not a gain.
4. **Report each signal separately as well as combined.** If one of the three
   carries everything, the other two are decoration and should be dropped.
5. **Rare-but-valid check.** On a newly-covered column, report the correlation
   between the score and the value's frequency. A detector that simply ranks
   rare strings first has reproduced the original failure in a new place.
6. **No leakage.** Dirty column only; assert it in a test.

Files: `src/detectors/text_fallback.py`, `tests/test_text_fallback.py`,
`experiments/w9b_text_fallback.py`, `results/w9b_text_fallback.md`.

---

## Tasks

- [ ] Commit this plan before any W9 code
- [ ] **W9.1** oracle triage value table
- [ ] **W9.2** text fallback detector + tests
- [ ] **W9.2b** coverage and detection evaluation against the six conditions
- [ ] Record both verdicts, pass or fail
