# Research program — rebuild from fundamentals, maximise metrics and novelty

Nothing is fixed: not the title, not the pipeline, not the decision formula.
Branch: `research/from-fundamentals`. Plan first, commit per change.

---

## 0. What the problem actually is

Strip the project to its primitives. We have a dirty table and a clean table.
Every cell is in one of three states:

```
(a) correct and left alone        — the overwhelming majority
(b) corrupted, and the original value is RECOVERABLE from context
(c) corrupted, and the original value is DESTROYED
```

Measured on our data (this is the governing constraint, not an opinion):

| corruption | repair accuracy | what it is |
|---|---|---|
| out-of-range | 98.4 % | (b) recoverable |
| format error | 100 % | (b) recoverable |
| missing value | 29–50 % | (c) destroyed |

**You cannot recover a destroyed value.** Any "repair accuracy" headline is
therefore mostly a statement about the (b)/(c) mix of the dataset, not about
the method. Chasing repair accuracy is chasing the dataset.

So the task worth optimising is not repair. It is:

> **Given a cell, how confident should we be that it is wrong, and what should
> we do about it?**

That decomposes into two genuine prediction problems, both with real metrics:

- **D (detection):** P(cell is corrupted). Binary, extremely imbalanced
  (adult 1,701 / 678,330 cells ≈ 0.25 %). **PR-AUC is the primary metric**;
  ROC-AUC is misleading under this imbalance. F1 for comparability with
  published systems.
- **R (repair-trust):** P(proposed repair is correct | a repair is proposed).
  This is the existing `p`. Brier/ECE.

The decision layer then consumes calibrated D and R. The current project only
ever optimised R, and used *hand-written rules* for D. D is where the headroom
is, and D is what published systems benchmark.

### The specific hard case, and why it matters

Adult's corruption is income-label flips: a **valid value replaced by another
valid value**. No syntactic rule can see it. Our detector scores F1 = 0.000 —
it finds none of the 1,701 and raises 5,829 false alarms.

Detecting (b)/(c) corruption that is *syntactically valid but semantically
improbable* requires modelling `P(cell | rest of row)`. That is the fundamental
move, and it is the one the current pipeline never makes.

---

## 1. Baseline to beat (measured, ours)

| dataset | precision | recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| adult | 0.000 | 0.000 | **0.000** | to measure |
| breast_cancer | 1.000 | 0.563 | **0.720** | to measure |
| smartfactory | 1.000 | 1.000 | **1.000** (trivial: NaN) | to measure |

Smart Factory is uninformative — the corruption is literally blank cells.
Real progress means Adult and Breast Cancer recall.

**External baseline is unknown and must be established by research, not
assumed.** We have never compared against published work (§6 of the previous
assessment). Any novelty claim is void until we know the published numbers on
these datasets.

---

## 2. Routes (run in parallel, independent, each must be falsifiable)

Each route produces: a method, PR-AUC/F1 per dataset on a held-out split, and
an honest statement of what it does *not* do. A route that loses is reported.

- **R0 — Evaluation harness + honest baselines.** Per-cell detection labels,
  stratified splits, PR-AUC/ROC-AUC/F1/precision@k, bootstrap CIs, and a
  leaderboard every other route writes into. *Nothing else can be trusted
  until this exists.* Also measures the current rule detector properly.
- **R1 — Literature decomposition.** What architectures achieve the best
  detection numbers on these benchmarks, what are the actual published
  figures, and — decomposed — which *factors* carry the edge (features?
  labels? ensembling? calibration? augmentation?). No implementation.
- **R2 — Conditional modelling.** Predict each column from the others
  (gradient boosting per column); flag cells whose observed value is
  improbable under the conditional distribution. This is the move that can
  see Adult's label flips. Fully self-supervised on dirty data.
- **R3 — Signal ensembling.** Treat every cheap signal (the six existing
  rules, conditional surprise, frequency, row-level outlierness,
  reconstruction error) as a feature; learn a meta-classifier. Tests whether
  the edge is in any single signal or in their combination.
- **R4 — Decision layer beyond the fixed formula.** The current argmin over
  three hand-set costs is one choice among many. Alternatives: learned policy,
  cost curves swept rather than fixed, selective-prediction framing
  (risk–coverage curves, AURC), conformal guarantees on abstention.

Wave 1: R0, R1, R2. Wave 2 is chosen from their results.

---

## 3. Rules for every route

1. **No ground-truth leakage into a detector.** Clean data produces labels for
   training/evaluation only, on a split the method never sees at inference.
   Self-supervised routes must state exactly what they fit on.
2. **Stratified, seeded splits.** Report mean ± bootstrap CI, never one run.
3. **PR-AUC primary.** Accuracy and ROC-AUC are vanity metrics at 0.25 %
   prevalence.
4. **A route that does not beat the baseline is a result.** Report it.
5. **Novelty is claimed only against measured published numbers**, never
   against intuition.

---

## 4. Todos

- [~] 0.1 Protocol added to CLAUDE.md; branch created; plan committed.
- [ ] R0.1 Evaluation harness (`src/detection_eval/`), PR-AUC/F1/CIs.
- [ ] R0.2 Honest baseline numbers for the existing rule detector.
- [ ] R1.1 Literature decomposition → `docs/literature.md`.
- [ ] R2.1 Conditional-model detector → leaderboard entry.
- [ ] W2 Choose wave 2 from R0–R2 results.
- [ ] Z.1 Final write-up: what beat what, by how much, with CIs, and what is
      genuinely novel against published numbers.

## 5. Results

_(leaderboard filled in by R0's harness; no number enters by hand)_
