# W5.3 control — the perfect column-AUC is a cardinality confound

`experiments/w5_column_triage.py` reported a **perfect** column-level
AUC of 1.0000 on adult with probe type `out_of_range`: the
corrupted `income` column had by far the lowest self-detectability
(d = 0.2356) while every clean numeric column sat at 0.98-0.99. All
four pre-registered conditions appeared to pass, including the
mechanism diagnostic. **It does not hold up.**

`income` on adult is encoded **0/1 int64**, so the corruptor
treats it as numeric and plants values like -50 and 51. An injected
-50 in a 0/1 column is a glaring outlier and should be *trivially*
detectable — d near 1.0, not 0.24. A strong result in the physically
wrong direction is a confound, not a discovery, and that is what
prompted this control.

**Mechanism**: the conditional detector models a numeric column with a
squared-error learner. Squared error is dominated by the extreme
planted values, so the booster chases them and predicts the planted
cells *better* than ordinary 0/1 cells, whose income is genuinely hard
to predict from the row. Small residual on planted cells gives a low
score and hence a low `d`. That is a property of the column's
**cardinality and span**, not of its corruption.

## The counterfactual the probe can never see

`d_clean` re-probes the identical column with its **clean** values
substituted. Clean data is used as a diagnostic only — exactly like
the (ii) AUC in the main experiment — and no score derived from it
ranks anything. It is explicitly NOT deployable.

| column | genuine corrupt cells | distinct values | d_dirty | d_clean | gap (dirty - clean) |
|---|---:|---:|---:|---:|---:|
| `age` | 0 | 69 | 0.9867 | 0.9867 | +0.0000 |
| `educational_num` | 0 | 16 | 0.9821 | 0.9821 | +0.0000 |
| `capital_gain` | 0 | 72 | 0.9893 | 0.9893 | +0.0000 |
| `capital_loss` | 0 | 49 | 0.9923 | 0.9923 | +0.0000 |
| `hours_per_week` | 0 | 76 | 0.9869 | 0.9869 | +0.0000 |
| `income` | 152 | 2 | 0.2356 | 0.4722 | -0.2367 |

## Verdict

- `income` corrupted: **d = 0.2356**
- `income` with clean values substituted: **d = 0.4722**
- clean numeric columns, as shipped: mean d = 0.9874

**CONFOUNDED.** A *clean* `income` still scores far below the clean continuous
columns, so it would have ranked first in the triage ordering
**even with no corruption in it at all**. The column-level AUC
of 1.0000 separates low-cardinality from high-cardinality
columns, not corrupted from clean. The pre-registered F1 and F4
passes on this dataset are therefore **artefacts and are
withdrawn**.

The residual gap of -0.2367 is a genuine H1 effect
with cardinality held fixed, and its sign is the one H1
predicts. But it is unusable: obtaining `d_clean` requires
precisely the ground truth the probe exists to avoid. A
deployable version would need a cardinality-matched
normalisation, which this experiment does not have.

## Consequence for W5.3

Combined with the main run — where `format` and `missing` saturate at
d = 1.0000 on `income` and carry no column signal, and `missing`
scores a column-AUC of 0.3462, *below* chance — the pre-registered
conclusion is that **self-supervised injection does not localise
corruption to a column**.

That is a measured negative result, and it is the one
`docs/literature.md` already anticipated: no published system
separates rare-but-valid from corrupted using one table and no
labels. The honest claim for the paper is a *measured impossibility*
on this axis rather than a missing feature — and it was caught by a
control run because the result looked too good, which is the
pre-registration discipline working as intended.

