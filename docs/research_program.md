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

## 4b. What R1 (literature) established — binding decisions

Full detail in `docs/literature.md`. These are now constraints, not options.

**Our F1 = 0.000 on Adult is the published behaviour of our method class.**
HoloDetect (SIGMOD 2019, Table 2) reports *outlier detection* on Adult at
**F1 = 0.002**, while its learned detector reaches 0.991 on the same table.
Rule/outlier detectors genuinely cannot see this corruption. Our baseline is
not a bug to be embarrassed by; it is the control.

**No published Adult number is comparable to ours.** Published Adult
corruptions are typos/outliers/rule violations spread over many columns; ours
is 100 % label flips confined to `income` (1,701 / 678,330 = 0.251 %). We
must never quote our Adult F1 beside theirs as a head-to-head.

**Breast Cancer has one real anchor:** SAGED (EDBT 2024, Table 2), 700×12 at
40 % error rate with a 20-label budget — SAGED 0.95, dBoost 0.70, KATARA 0.48,
Raha 0.18, ED2 0.18, HoloClean 0.14. Our 7.5 % error rate differs, so this is
an anchor for protocol, not a leaderboard we can join uncritically.

**Honest ceiling on Adult, derived not measured:** every flip is in the target
column, so detecting them *is* the Adult income-classification task (~85–87 %
accuracy). Derived ceiling ≈ P 0.2–0.3, R 0.6–0.8, **F1 ≈ 0.3–0.4**. A large
move from 0.000, but we do not promise 0.9. R0's harness must measure it.

**Metric practice:** 7/7 surveyed papers report P/R/F1; none reports PR-AUC.
Our PR-AUC-primary rule stays (it is correct at 0.25 % prevalence) but we
**report both**, or our numbers are unquotable.

**Protocol to match (SAGED):** 20 labels for every label-consuming method,
P/R/F1 plus runtime, 10 seeds.

### Novelty position — corrected

| claim | status |
|---|---|
| Platt/isotonic calibration of a detector | **not novel** (HoloDetect 2019) |
| *Measuring* calibration quality (ECE/Brier/reliability) of a cleaning model | **novel** — 0 of 7 papers do it |
| Expected-cost argmin over {repair, review, abstain} | novel *in data cleaning*; **not novel as a framework** (Chow's rule, SelectiveNet, cost-aware conformal deferral) |
| Three-way action space per cell | **novel in cleaning** — REIN explicitly lists "only a binary decision per cell" as a shortcoming |

**Framing:** *selective prediction and cost-sensitive deferral, transferred to
per-cell data cleaning, with calibration actually measured.* Claiming we
invented calibrated cost-sensitive decisions gets the paper rejected.

**Nearest competitor, must be positioned against:** Conformal Data Cleaning
(AISTATS 2024) already pairs per-column conditional models with calibrated
uncertainty. R2 is not novel on its own; the novelty must live in the
decision layer and in measured calibration.

### Factors that carry the edge (with cited ablations)

1. **Conditional / row-context modelling** — largest single ablation drop on
   Adult (HoloDetect 0.971 → 0.884); ZeroED −0.14 F1 average. The only factor
   with direct evidence on valid→valid corruption. → R2.
2. **The first ~20 labels** — Raha 2 labels ≈ 0.40 F1 → 20 labels **0.85**.
   We currently use **zero** labels for detection. **Largest documented jump
   in the field.** → wave 2.
3. **Synthetic-corruption augmentation** — HoloDetect +20 F1 points and 3×
   fewer labels. Must use a *different* corruption policy from our injector or
   the result is circular. → wave 2.

### Wave 2 (provisional, pending R0/R2)

- [ ] W2.1 20-label meta-classifier over {six rule outputs, conditional
      surprise, marginal rarity, row outlierness} — factor 2, biggest lever.
- [ ] W2.2 Synthetic-corruption augmentation with a non-circular policy.
- [ ] W2.3 Reframe the decision layer as selective prediction: risk–coverage
      curves and AURC, costs swept rather than fixed.
- [ ] W2.4 Explicit comparison against Conformal Data Cleaning's framing.

**Not recommended by R1:** Picket-style transformers (crashed on Adult and
Smart Factory in REIN); more hand-written rules (HoloClean 0.51 → 0.12 as
rules were cut); chasing 0.991 on Adult (different corruption entirely).

---

## 5. Results

_(leaderboard filled in by R0's harness; no number enters by hand)_
