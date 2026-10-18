# External baseline: Raha vs our detector

The first time this project has run a **published** method on its own data.
Until now every baseline we beat was one we wrote ourselves, which established
that our method beats our controls — not that it is competitive with the
literature.

**Raha**: BigDaMa/raha, SIGMOD Most Reproducible Paper Award. Installed from
PyPI into a dedicated venv (it pins `pandas==2.1.0`; the project runs 2.3).
**Labelling budget 20**, simulated from `clean.csv` — standard Raha / ED2 /
SAGED benchmarking practice.

## Head-to-head, both at their natural operating points

| dataset | ours P | ours R | **ours F1** | Raha P | Raha R | **Raha F1** | winner |
|---|---:|---:|---:|---:|---:|---:|---|
| breast_cancer | 1.000 | 0.621 | **0.766** | 1.000 | 0.021 | **0.040** | ours |
| hospital | 0.864 | 1.000 | **0.927** | 0.671 | 0.285 | **0.400** | ours |
| beers | 0.758 | 1.000 | **0.862** | 0.974 | 0.677 | **0.799** | ours |
| rayyan | 0.263 | 0.627 | **0.370** | 0.676 | 0.618 | **0.646** | **Raha** |

Our rule detector produces a flag *set*, so its natural operating point is a
single (P, R) pair — directly comparable to Raha's. **No oracle threshold is
used in this table**; quoting our best-F1 here would have compared our best
possible threshold against Raha's actual one.

## SETTLED, AND IT REVERSES THE CONCLUSION

**Read this before the table above.** The caveat this document originally
flagged as unsettled has now been measured
(`results/w5_raha_leakage_check.md`, W5.2b), and it does not survive.

Our detector's valid ranges, vocabularies and IQR fences come from
`DatasetProfile`, built from `clean_df`. Recomputing the identical
operating point with **every profile statistic derived from dirty data** —
no clean data of any kind — gives:

| dataset | ours F1 (clean profile) | **ours F1 (zero knowledge)** | Raha F1 (20 labels) | winner |
|---|---:|---:|---:|---|
| breast_cancer | 0.7664 | **0.6327** | 0.0404 | ours |
| hospital | 0.9271 | **0.0000** | 0.4000 | **Raha** |
| rayyan | 0.3702 | **0.0558** | 0.6457 | **Raha** |
| beers | 0.8622 | **0.0558** | 0.7988 | **Raha** |

**We go from winning three of four to losing three of four.** On hospital
the detector flags **zero cells**. The mechanism is not mysterious: the
novel-category and valid-range checks are *definitionally* comparisons
against a known-good vocabulary or range. Build that vocabulary from dirty
data and the corrupted values are inside it, so the check can never fire.

### What the honest claim is

The ordering is monotone in how much information each configuration gets:

| configuration | labels | clean aggregate knowledge | F1 on beers |
|---|---|---|---:|
| ours, dirty profile | 0 | none | 0.0558 |
| Raha | 20 | none | 0.7988 |
| ours, clean profile | 0 | full distributional | 0.8622 |

So the correct statement is **not** "we beat Raha". It is: *our detector is
not better than Raha, it is differently informed.* Its zero-label claim was
true but misleading — zero labels is not zero knowledge, and the clean profile
is plausibly a larger advantage than twenty labelled cells. Under
comparable ignorance our rule detector is substantially worse than a
published method.

This is the single most consequential negative result in Wave 5 and it must
go in the paper as such. It does **not** touch the decision-layer claim:
`results/w5_leakage_ablation.md` shows zero worst-case regret holds under
both profiles. What it removes is any claim to a competitive *detector*.

## Reading the original numbers


**We win three of four, and Raha wins rayyan** — the dataset our cross-domain
run already flagged as an undiagnosed failure (lift 2.1x at 8.62 %
prevalence). Raha reaches F1 0.646 there against our 0.370. Whatever our
detector is missing on rayyan, a published method finds it.

**We use zero labels; Raha uses twenty.** On the face of it that favours us.

**But we are not truly label-free — and the section above now shows this is decisive, not a caveat.**
Our detector's valid ranges, vocabularies and IQR fences come from
`DatasetProfile`, which is built from `clean_df` (CLAUDE.md §3). That is
population-level knowledge of the clean data — a different and arguably
larger advantage than Raha's 20 labelled cells. It has now been run, and the
superiority claim does not survive it — see the top of this document.

**Raha's breast_cancer result looks anomalous.** It flagged only 13 cells out
of 8,388 (recall 0.021) on a dataset with 7.5 % prevalence. That is far below
its published behaviour and is more likely a configuration or label-simulation
artefact than a true measurement. It is reported as measured, but should not
be quoted as evidence that we beat Raha on this dataset.

## What this does not show

- One labelling budget (20) and one seed per dataset; Raha is known to improve
  with larger budgets.
- Four datasets; the larger ones (movies, flights, soccer, tax) exhausted
  memory with repeated OpenBLAS allocation failures and are documented as
  failures rather than omitted silently.
- Raha's own corruption assumptions and ours differ; both were run on the
  identical shipped clean/dirty pairs, which is the fairest available setup
  but not a controlled one.
- This compares **detection only**. Our contribution is the decision layer;
  Raha does not make repair/review/abstain decisions at all.
