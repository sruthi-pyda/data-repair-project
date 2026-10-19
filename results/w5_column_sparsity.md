# Column-corruption sparsity across every shipped dataset

**This table corrects a scoping error.** `docs/w5_3_column_triage_plan.md`
§5c censused only the three original datasets and concluded that column
triage "can only really be evaluated on adult". That conclusion was drawn
from 3 of the 11 datasets in the repo. Across all 11 it is wrong:
**column triage is well posed on eight of them.**

Column triage needs a column-level AUC, which needs both corrupted *and*
clean columns. `corr` and `clean` below count columns, not cells.

| dataset | columns | corrupted | clean | cell prevalence | corrupted cells | triage testable? |
|---|---:|---:|---:|---:|---:|---|
| tax | 15 | 9 | 6 | 0.00099 | 2,966 | **yes** |
| restaurants | 16 | 10 | 6 | 0.00121 | 556 | **yes** |
| soccer | 10 | 4 | 6 | 0.01565 | 31,296 | **yes** |
| hospital | 20 | 17 | 3 | 0.02545 | 509 | **yes** |
| movies | 17 | 11 | 6 | 0.05577 | 7,006 | **yes** |
| rayyan | 11 | 8 | 3 | 0.08618 | 948 | **yes** |
| beers | 11 | 4 | 7 | 0.12663 | 3,357 | **yes** |
| flights | 7 | 4 | 3 | 0.29582 | 4,920 | **yes** |
| adult | 15 | 1 | 14 | 0.00251 | 1,701 | weak |
| smartfactory | 19 | 18 | 1 | 0.01579 | 7,093 | weak |
| breast_cancer | 12 | 12 | 0 | 0.07523 | 631 | **no** |

## Consequence

- **Testable on 8 datasets**: `beers`, `flights`, `hospital`, `movies`, `rayyan`, `restaurants`, `soccer`, `tax`.
- **Degenerate on 2**: `adult` (1 corrupted column of 15) and
  `smartfactory` (18 of 19) each have one class almost empty, so their AUC
  exists but carries almost no information.
- **Impossible on 1**: `breast_cancer`, where all 12 columns are corrupted.

The W5.3 probe was run on adult and breast_cancer — the *worst two*
choices available. Adult gives a best-possible one-sided p of 1/15 = 0.067,
so **no result on adult alone can reach p < 0.05**, and breast_cancer
cannot produce a number at all. The refutation in
`results/w5_column_triage_control.md` still stands on its own terms (the
cardinality confound is real and was directly measured), but the *power* of
the test was far lower than it needed to be.

Best candidates for a properly powered re-run, sparse corruption and low
prevalence — the regime that actually matters:

| dataset | corrupted / total columns | prevalence |
|---|---|---:|
| `tax` | 9 / 15 | 0.00099 |
| `restaurants` | 10 / 16 | 0.00121 |
| `soccer` | 4 / 10 | 0.01565 |
| `hospital` | 17 / 20 | 0.02545 |
| `movies` | 11 / 17 | 0.05577 |

