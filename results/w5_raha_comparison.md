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

## Reading this honestly

**We win three of four, and Raha wins rayyan** — the dataset our cross-domain
run already flagged as an undiagnosed failure (lift 2.1x at 8.62 %
prevalence). Raha reaches F1 0.646 there against our 0.370. Whatever our
detector is missing on rayyan, a published method finds it.

**We use zero labels; Raha uses twenty.** On the face of it that favours us.

**But we are not truly label-free, and this is the caveat that matters most.**
Our detector's valid ranges, vocabularies and IQR fences come from
`DatasetProfile`, which is built from `clean_df` (CLAUDE.md §3). That is
population-level knowledge of the clean data — a different and arguably
larger advantage than Raha's 20 labelled cells. **Until the leakage ablation
(W5.1b) is run, this comparison is not settled**, and no claim of superiority
should be made from this table alone.

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
