# What each cost means, and what "realised cost" measures

Written in response to a methodological challenge (2026-09-22): *before
calling calibrated abstention "solved", define precisely what `C_abstain`
represents, show the realised-cost equation under both interpretations,
justify the chosen one, and show the conclusions survive the smoothing
prior.* This document answers all four and records the derivations. It is
the reference for how the paper must describe cost.

---

## 1. The three penalties

All values are **abstract experimental penalty units** from
`configs/costs.yaml`. They are not money, not clinical harm, and not
supplied by the REIN benchmark.

| Symbol | Config key | Meaning (as defined in the original code) |
|---|---|---|
| `C_wrong` | `c_wrong_repair` | "cost of automatically applying an incorrect repair" |
| `C_human` | `c_human_review` | "cost of sending one cell to human review" |
| `C_abstain` | `c_leave_error` | **"cost of leaving a genuine error unrepaired"** |

The third row is the whole argument. The quantity is named
`c_leave_error` — *leave **error*** — and both the YAML comment and the
`CostConfig` docstring at HEAD `5d8cc22` (before any change on this
branch) define it as the penalty for leaving **a genuine error**
unrepaired. That is the project's own definition, authored with Phases
1–10, not a redefinition introduced here.

### The two candidate interpretations

**(a) Corruption-persistence cost.** `C_abstain` is charged when a cell
that really is corrupted is left corrupted. A false alarm left alone
costs nothing, because nothing bad persists.

**(b) Flag-handling cost.** `C_abstain` is charged whenever the system
declines to resolve a flag, regardless of whether the flag was real
(paying for triage effort, workflow overhead, erosion of trust).

The codebase defines (a). This branch implements (a). The previous
implementation *stated* (a) and *computed* something else — see §2.

---

## 2. The inconsistency this branch fixed

At HEAD `5d8cc22`, `src/risk/evaluation.py` charged `C_leave` for every
abstention, justified by this comment:

> `# Note: all detected records are (alleged) errors, so cost = C_leave`

That is an assumption, and it is measurably false:

| Dataset / detector | flags that are genuinely corrupted |
|---|---|
| breast_cancer, k=3 (default) | 355 / 355 — **100 %** |
| breast_cancer, k=1.5 | 360 / 550 — 65 % |
| smartfactory, outliers on | 7,093 / 42,732 — 17 % |
| adult (default) | 0 / 2,000 — **0 %** |

Where the assumption holds (breast_cancer k=3, and smartfactory with the
shipped outliers-off default) interpretation (a) and the old computation
agree exactly, and every number is unchanged. Where it fails, the old
evaluator charged a persistence penalty for corruption that did not
persist because it never existed.

So this branch did not change the definition of cost. It changed the
implementation to match the definition that was already written, and
removed an assumption the data contradicts.

### Realised-cost equations, both interpretations

Let `A_i` be the action on cell `i` and `corrupt_i = [dirty_i ≠ clean_i]`.

```
Interpretation (a) — implemented

  cost_i = C_wrong · [A_i = AUTO   and repair wrong]
         + C_human · [A_i = HUMAN]
         + C_abstain · [A_i = ABSTAIN] · corrupt_i

Interpretation (b) — the alternative

  cost_i = C_wrong · [A_i = AUTO   and repair wrong]
         + C_human · [A_i = HUMAN]
         + C_abstain · [A_i = ABSTAIN]            (no corrupt_i factor)
```

Under (b), expected and realised cost are consistent only if
`E[ABSTAIN] = C_abstain` — i.e. the original flat formula — which is
exactly the degenerate case this work set out to remove. Under (a),
`E[ABSTAIN] = q · C_abstain` is the expectation of the realised cost,
because `q = P(corrupt_i)`. **(a) is the only pairing in which the
expected cost the policy minimises is an unbiased estimate of the cost
the evaluator charges.** That is the methodological reason to choose it,
independent of which produces nicer numbers.

### Where flag-handling cost actually belongs

Interpretation (b)'s concerns (triage time, compute, trust) are real, but
they are incurred **because a flag exists**, not because of which action
follows. A per-flag cost `C_flag` therefore adds `C_flag` to all three
arms:

```
E[AUTO] + C_flag,   E[HUMAN] + C_flag,   E[ABSTAIN] + C_flag
```

A constant added to every arm cannot change an argmin. It shifts every
policy's total by exactly `n · C_flag` and changes no decision and no
ranking between policies. So it is reportable as a separate additive term
(`n × C_flag`) and is deliberately not folded into `C_abstain`, where it
would corrupt the expectation/realisation correspondence above.

Human review is the one action that *does* consume operator time, and
that cost is already `C_human`, charged unconditionally — including on
false alarms. False alarms are therefore not free in this model: they are
free only if abstained on, which is the correct incentive.

---

## 3. What "cost 0" on Adult does and does not mean

Adult's realised cost of 0 must be reported as:

> zero **residual-corruption** cost on the flagged cells

and never as "zero cost" or "the data is now clean". Two separate
qualifications are mandatory:

1. **Scope.** The metric covers only the cells the detector flagged.
   Adult contains **1,701 genuinely corrupted cells** (income-label
   flips). Phase 3 detected **none** of them. Those false negatives are
   invisible to the cost model entirely — no action was ever taken on
   them, so they enter no arm of the objective. A policy cannot be
   credited for handling errors it never saw.
2. **Detection cost.** Flagging 2,000 cells consumed detection effort
   that the model does not price (see `C_flag` above).

The defensible claim is: *given a detector whose flags are entirely false
positives, the calibrated policy correctly declines to act on all of
them, incurring no persistence penalty and no review effort, whereas the
uncalibrated policy pays 2,000 × `C_human` to review cells that were
never broken.* The finding is about **not wasting effort on false
alarms**, not about repairing Adult.

Total genuine corruptions per dataset, for context in any results table:
breast_cancer 631, smartfactory 7,093, adult 1,701.

---

## 4. The decision regions, derived (not assumed)

With `E[AUTO] = (1−p)C_w`, `E[HUMAN] = C_h`, `E[ABSTAIN] = q·C_a` and tie
priority `AUTO > HUMAN > ABSTAIN`:

```
AUTO    optimal ⟺ p ≥ 1 − C_h/C_w   and   p ≥ 1 − q·C_a/C_w
HUMAN   optimal ⟺ p < 1 − C_h/C_w   and   q ≥ C_h/C_a
ABSTAIN optimal ⟺ p < 1 − q·C_a/C_w and   q < C_h/C_a
```

Three boundaries: a **vertical** line `p* = 1 − C_h/C_w`, a
**horizontal** line `q* = C_h/C_a`, and a **sloped** line
`p = 1 − q·C_a/C_w`. Verified against the implementation on a 201×201
grid for all three shipped configs: **0 mismatches out of 40,401 points
each**.

| Dataset | `p*` (auto vs human) | `q*` (human vs abstain) | auto vs abstain |
|---|---|---|---|
| breast_cancer | 0.900 | 0.500 | p = 1 − 0.200q |
| smartfactory | 0.920 | 0.400 | p = 1 − 0.200q |
| adult | 0.800 | 0.400 | p = 1 − 0.500q |

### The honest consequence

`q*` is a horizontal line, so **for a single cell with a fixed `q`, one of
HUMAN and ABSTAIN is still unreachable**: above `q*` abstention can never
win; below it, review can never win. Both remain constant in `p`, so they
still never cross. That part of the original structural fact is
unchanged and cannot be removed while both arms are `p`-independent.

What the mechanism changes is **which** of the two is live: previously it
was fixed for a whole dataset by the config, and was always HUMAN.
Now it is determined per cell by evidence about that cell. Across a
population containing both real errors and false alarms, all three
actions therefore occur — which is what the thesis claims and what the
breast_cancer k=1.5 run demonstrates (AUTO 35 / HUMAN 320 / ABSTAIN 195).

The paper must state it this way. "All three actions are now reachable"
is true of the population; "all three are reachable for every cell" is
false, and a reviewer checking the algebra will find that in a minute.

---

## 5. Robustness to the smoothing prior

`q_group = (k + s)/(n + 2s)` — a Beta(s, s) prior centred on 0.5. `s` is
a free parameter, so `experiments/smoothing_ablation.py` sweeps
`s ∈ {0, 0.5, 1, 5, 20}` (0 = raw empirical rates, 0.5 = Jeffreys,
1 = Laplace/default) and reports q, actions and cost per scenario.

**Result: the action distribution and the realised cost are identical for
every `s`, in every scenario.** Only the q values move. Comparison with
the `q = 1` baseline (no detection calibration at all) isolates the effect
of the mechanism from the effect of its hyper-parameter:

| Scenario | q = 1 (original) | any s ∈ {0 … 20} |
|---|---|---|
| breast_cancer k=3 | 52 / 303 / 0 — cost 1615 | identical, cost **1615** |
| breast_cancer k=1.5 | 35 / 515 / 0 — cost 2675 | 35 / 320 / 195 — cost **1750** |
| smartfactory outliers on | 0 / 42732 / 0 — cost 85464 | 0 / 7093 / 35639 — cost **14186** |
| adult | 0 / 2000 / 0 — cost 4000 | 0 / 0 / 2000 — cost **0** |

Why it is stable: the abstain/review boundary is `q* = C_h/C_a`
(0.4–0.5 here), while the learned rates sit far from it — IQR-outlier
groups at 0.000–0.140 and genuine-error groups at 0.512–1.000 across the
whole sweep. Smoothing moves q but not across the boundary.

**The one tight margin, reported rather than hidden.** At `s = 20`,
breast_cancer's `format_check` group (3 records, all genuinely corrupted)
is shrunk to q = 0.512 against `q* = 0.500`. It stays on the correct side
by 0.012 in q, i.e. 0.12 penalty units. A stronger prior than any tested
would push a genuinely corrupted group into abstention, which would be
wrong. Mitigation if this ever matters: shrink toward the dataset's
global corruption rate rather than toward 0.5, i.e. `(k + s·g)/(n + s)`.
Not implemented, because at every `s` tested it changes no decision.

---

## 6. The `q = 1` fallback is an assumption, not evidence

When the detection calibrator cannot fit (no records, or a train split
with no label variance), `predict` returns **1.0 for every record**. This
reproduces the pre-calibration policy exactly, which is why the whole
existing suite passes unedited.

It must be described as: *"calibration failure defaults to `q = 1`,
preserving the pre-calibration policy."* It is a conservative
backward-compatibility choice — assume every flag is a real error and
therefore never abstain — and it is **not** evidence that `q = 1` is
correct for any dataset. Any results row produced under the fallback
should be marked as such, exactly as the Phase 6 base-rate fallback is.

---

## 7. Checklist for the paper

- [ ] Define `C_abstain` as corruption-persistence cost, quoting the
      config key `c_leave_error`; give both realised-cost equations from
      §2 and the expectation/realisation argument for choosing (a).
- [ ] Report any flag-handling cost separately as `n × C_flag`, noting it
      is decision-invariant.
- [ ] Report Adult as "zero residual-corruption cost on flagged cells",
      always beside the 1,701 undetected corruptions.
- [ ] Give the derived region boundaries (§4) and state plainly that
      per-cell only two arms are ever live, population-wide all three.
- [ ] Include the smoothing ablation table, including the `s = 20`
      format_check margin.
- [ ] Mark any row where a calibrator fell back (`q = 1`, or the Phase 6
      base-rate fallback).
