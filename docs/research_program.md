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

### R0 baseline (rule detector) vs R2 (conditional model)

PR-AUC, 95 % CI. A constant scorer's PR-AUC equals prevalence regardless of
split protocol, so the adult baseline comparison is valid across the two
harnesses.

| dataset | rule detector (R0) | conditional `full` (R2) | marginal control (R2) |
|---|---:|---:|---:|
| adult | 0.0025 (= prevalence, ROC 0.4957) | **0.0120** [0.0108, 0.0131] (ROC **0.8875**) | 0.0085 [0.0077, 0.0091] |
| breast_cancer | **0.6712** [0.6486, 0.6917] | 0.6404 [0.5729, 0.7034] | 0.6325 [0.5729, 0.6909] |
| smartfactory | 1.0000 | 1.0000 | 1.0000 (trivial) |

**On the `income` column alone — where adult's corruption actually lives:**

| mode | PR-AUC | best-F1 (oracle) |
|---|---:|---:|
| conditional `full` | **0.6997** | 0.6758 |
| `ratio` | 0.5295 | — |
| marginal control | 0.1321 | — |

### What this establishes

1. **Conditioning on the row is the factor, confirmed by the control.** On
   adult, `full` beats `marginal` with **non-overlapping** CIs
   ([0.0108, 0.0131] vs [0.0077, 0.0091]), and 0.70 vs 0.13 PR-AUC on the
   income column. The ablation was built to be able to refute this and did
   not. ROC-AUC moves 0.4957 → 0.8875: from anti-correlated to a genuinely
   good ranking.
2. **It beats R1's derived ceiling.** R1 predicted F1 ≈ 0.3–0.4 from the
   ~85–87 % accuracy of Adult classification. We measure oracle F1 0.6758.
   The derivation assumed *hard* classification; soft probabilities separate
   better, because a flipped cell gets very low `P(observed)` while a merely
   misclassified clean cell gets a moderate one. **Caveat: 0.6758 is an
   oracle threshold (R0's upper bound), not an operating point, and is not
   comparable to published F1 at chosen thresholds.**
3. **Overall adult PR-AUC stays small (0.0120) and that is honest.** The
   corruption is confined to 1 of 15 columns, so scoring all 203,505 cells
   dilutes precision with 14 guaranteed-clean columns. Per-column PR-AUC is
   the meaningful number; report both.
4. **On breast_cancer the conditional model LOSES to the rule detector**
   (0.6404 vs 0.6712), and `full` ≈ `marginal` there, so conditioning adds
   nothing. Reported as a loss.
5. **Smart Factory remains uninformative.** Its non-missing view has **zero**
   positives — all corruption is missing values — correctly rendered `—`
   rather than 0.0.

### The synthesis that defines wave 2

The two detector families are **complementary, not competing**:

```
rules        → syntactic corruption (breast_cancer: 0.67 vs 0.64)
conditional  → semantic corruption  (adult: 0.0120 vs 0.0025, 5x on income)
neither wins everywhere
```

That is exactly R1's factor 3 (ensembling) and factor 2 (the first ~20
labels, Raha 0.40 → 0.85), and it is now motivated by our own measurements
rather than by the literature alone.


---

## 6. Wave 2 (W2.1) — ensemble + label efficiency

PR-AUC, 5 splits, seed 42, 95 % CI. Full curve in `results/w2_ensemble.md`.

| dataset | rule_only | cond_only | **b0 noisy-OR** | b20 | b100 | full |
|---|---:|---:|---:|---:|---:|---:|
| breast_cancer | 0.6712 | 0.6361 | **0.7209** [0.6977, 0.7371] | 0.6763 | 0.7617 | 0.7898 (GBM 0.8082) |
| adult | 0.0025 | 0.0110 | 0.0106 | 0.0106 | **0.2656** [0.2155, 0.3124] | **0.7024** [0.6879, 0.7169] |
| smartfactory | 1.0000 | 1.0000 | 1.0000 | — | — | tie at ceiling |

### Findings

1. **Complementarity is real and FREE on breast_cancer.** Noisy-OR at *zero
   labels* beats both single detectors with non-overlapping CIs
   (0.7209 vs 0.6712 / 0.6361). Rank-averaging is clearly worse (0.6954) —
   it inflates near-zero scores; noisy-OR is the right fusion.
2. **On adult the complementarity thesis does NOT win.** Label-free fusion is
   slightly *worse* than the conditional model alone (0.0106 vs 0.0110).
   What wins is labels.
3. **Raha's "20 labels" does not transfer to extreme imbalance.** At 0.251 %
   prevalence, k ≤ 20 buys *literally nothing* — the sampler draws zero
   positives in 5/5 splits. k=100 buys 24×. The field's label-efficiency
   results are measured at 5–40 % error rates; **this is a measured negative
   result at 0.25 % that nobody appears to have reported.**
4. **The sampler, not the learner, is the bottleneck.** b100 0.2656 vs full
   0.7024 — a 2.6× gap that more labels of the same kind would close only
   slowly. Global samplers fail outright on adult: `top_combiner` draws zero
   positives at every k ≤ 100, because the global top is the rules' `fnlwgt`
   false positives plus blank `workclass`/`occupation` — all guaranteed clean.
5. **Whole-table and per-column disagree about which method is better.** On
   adult's `income` column, supervised b100 scores 0.3456 vs cond_only's
   0.6554: the learner buys whole-table gain by learning "corruption lives in
   the binary column" and pays in within-column resolution. Both must be
   reported.

### Measurement catch worth keeping

Conditional features are scored **out-of-fold over rows** (5 folds). A single
in-sample fit scores 0.5014 on breast_cancer vs 0.6408 out-of-fold — in-sample
would have handed us a handicapped baseline to beat and a fake improvement.
5-fold reproduces wave 1's row-split number to 0.0004.

### Wave 3 — what the bottleneck points at

- [ ] W3.1 **Uncertainty-targeted label acquisition.** The gap b100 → full is
      a *sampling* problem: at 0.25 % prevalence, random and global-top
      sampling both miss positives. Use the conditional model's own scores to
      target queries where positives are likely, and measure the label curve
      against random/global-top. This is the novel direction the measurements
      actually point at.
- [ ] W3.2 Calibrate the ensemble's output and **measure** calibration
      (ECE/Brier/reliability) — per `docs/literature.md`, 0 of 7 surveyed
      papers do this, and it connects detection to the existing Phase 6b/7/8
      decision layer.
- [ ] W3.3 Write-up positioned against Conformal Data Cleaning (AISTATS 2024).


---

## 7. Wave 3 (W3.1) — where should a tiny label budget be spent?

Full curves in `results/w3_acquisition.md`. 5 splits, seed 42, bootstrap CIs.

### Adult is the only dataset where the question is meaningful

On breast_cancer the **label-free** combiner (0.7209) already exceeds 80 % of
full supervision (0.80 x 0.7898 = 0.6318), so "labels needed to reach 80 %"
is satisfied at k=10 by every strategy and the metric is **degenerate**. It is
reported, and must not be quoted. Adult is the real test: b0 = 0.0106,
full = 0.7024.

### Adult: labels needed to reach a fraction of full supervision

| strategy | rounds | 50 % of full | 80 % | best PR-AUC | pos/query |
|---|---|---|---|---|---|
| `batch_diverse` | **5** | **100** | — | 0.5408 [0.4786, 0.5888] | 0.048 |
| `per_column_quota` | 5 | 100 | — | 0.5021 [0.4777, 0.5216] | 0.041 |
| `batch_diverse` | 1 | 500 | — | 0.4748 | 0.056 |
| `cond_per_column` | 1 | 500 | — | 0.4929 | 0.055 |
| `per_column_quota` | 1 | 500 | — | 0.4922 | 0.055 |
| `random` | 1 | **—** | — | 0.2529 | 0.002 |
| `global_top` / `conditional_top` / `cond_top_live` / `uncertainty` | 1 | **—** | — | **0.0106** | **0.000** |

### Findings

1. **Sequential acquisition is worth a 5x label reduction.** `batch_diverse`
   reaches 50 % of full supervision at **k=100 sequentially vs k=500 one-shot**.
   Refitting between rounds is where active learning earns its keep here.
2. **Nothing reaches 80 % of full at any budget up to 500.** The b100 -> full
   gap identified in wave 2 narrows but does not close. Honest ceiling.
3. **The top of a well-ranked list is not where the positives are.** The
   conditional model ranks adult well overall (wave 1: ROC-AUC 0.8875), yet
   `conditional_top` queries **zero positives at every budget up to 500** and
   never moves off the label-free score. At 0.25 % prevalence the highest-
   surprise cells are genuinely unusual *clean* values in other columns. Good
   ranking and good acquisition are different problems.
4. **Diversity, not confidence, is what works.** The only strategies that
   learn anything on adult spread the budget — `batch_diverse` (de-duplicated
   across columns and rows) and `per_column_quota`. Every
   concentrate-on-the-top strategy returns exactly the label-free score.
5. **A paradox at the other extreme, measured on breast_cancer:**
   `conditional_top` achieves **1.000 positives per query** — every queried
   cell genuinely corrupted, vs random's 0.078 — and its PR-AUC does not beat
   the label-free combiner, because a sample with no negatives gives a
   discriminative classifier nothing to learn. **Acquisition precision and
   learner value pull in opposite directions.** A strategy can be too good at
   finding positives.
6. **Random is not a weak control.** On breast_cancer random attains the best
   PR-AUC of any strategy at k=500 (0.7688). It only collapses on adult, where
   it never reaches 50 % of full. Report both.

### What this does not show

- Labels are free here; this measures how *informative* a query is, not its
  human cost.
- One hyper-parameter setting per dataset; no sweep.
- Smart Factory excluded: its corruption is exactly the NaN cells, so every
  method ties at the ceiling and the curve is flat by construction.


---

## 8. Why adult's overall PR-AUC stays low — measured, not inferred

Earlier wording said the 14 untouched columns "drown out" the one corrupted
column. That was wrong as a mechanism. Measured directly:

| scored set | PR-AUC |
|---|---:|
| all 15 columns | **0.0123** |
| `income` only | **0.6873** |
| all 15, other 14 forced to score 0 | **0.6873** |

The last two are identical, so **the entire loss is caused by the untouched
columns and disappears if the detector is silent about them.** Extra clean
columns do not dilute PR-AUC by existing; a low-scoring column costs nothing.

### The real mechanism: displacement, not dilution

**Of the top 5,000 scored cells, ZERO are in `income`.** The top 1,701 slots
(the number of true corruptions) are held by:

| column | cells in top 1,701 | true positives |
|---|---:|---:|
| `native_country` | 1,348 | 0 |
| `occupation` | 343 | 0 |
| `marital_status` | 8 | 0 |
| `workclass` | 2 | 0 |
| **`income`** | **0** | — |

`native_country` is overwhelmingly "United-States", so a rare value is
genuinely improbable given the row. The conditional model is *correct* that
these cells are surprising. They are also completely clean.

### The consequence for what we may claim

The detector answers "which cells are surprising?" well. We wanted "which
cells are corrupted". On adult those are different questions, and we only
have the first.

**The 0.6873 figure is partly oracle**: it requires knowing in advance that
corruption is confined to `income`, which is ground-truth knowledge. It must
never be quoted as an unaided detector result. The defensible claim is:

> the conditional model ranks corruption well *within* a column, but cannot
> tell which column is worth looking at.

This also explains wave 3 exactly: `conditional_top` queried zero positives at
every budget because the top of its ranking is all `native_country`, and the
only strategies that learned anything were the ones that spread the budget
*across* columns (`batch_diverse`, `per_column_quota`).

**Implication for any next route:** the open problem is not better per-cell
surprise. It is **column-level triage** — deciding where corruption plausibly
lives before ranking cells inside it.

---

## 9. Wave 4 — reviewer gaps and two new mechanisms

Triggered by external review (three gaps) plus two user proposals. Each route
is falsifiable and reports losses.

### The reviewer's gaps

**G1 — the detection metric is not self-explanatory (high).** We quote 0.0025
→ 0.0120 → 0.70 without stating the metric, and at 0.25 % prevalence a
detector can look fine while missing almost everything. Needed: exact metric
definitions, precision–recall curves, **per-column precision and recall**, and
an explicit false-positive accounting.

**G2 — the costs need a defensible interpretation (med-high).** The zero-regret
result depends on one cost triple per dataset. If the advantage evaporates
under modest changes the contribution is weak. Needed: cost-sensitivity
sweeps, and a clear statement of whether costs are fixed, learned, or
(critically) **selected using test outcomes** — they are fixed and come from
`configs/costs.yaml`, never tuned on results, and that must be provable.

**G3 — novelty is not established (high).** Every baseline we beat is one we
wrote. `docs/literature.md` found published numbers but ran nothing. Needed:
direct comparison against a published method on our data.

### The two proposed mechanisms

**M1 — difficulty-aware repair routing.** Measured in §3.5 of the report:

| corruption | rule-based repairer | LLM |
|---|---:|---:|
| recoverable (out-of-range, format) | **100 %** | 92.5 % |
| destroyed (missing) | 45 % | 50 % |

The LLM is *worse where the answer is determined* and marginally better where
it is a guess. So **route by difficulty**: deterministic repairs to the rule,
genuinely ambiguous ones to the model. Upper bound if routing were perfect:
~100 % on recoverable + ~50 % on destroyed, beating both pure strategies.
This is the project's own thesis — decide what to trust — applied one level
up, to *which repairer* to trust. Cheap to build, cheap to falsify.

**M2 — column-level triage.** §8 showed the real failure: of the top 5,000
scored cells on adult, **zero** are in the tampered column; the top is
`native_country`, where rare-but-clean values are genuinely improbable. The
model ranks well *within* a column and cannot tell *which column* to examine.
Per-cell surprise is solved; column selection is not.

### Routes

- [ ] **W4.1 (research)** — how is column-level / attribute-level error
      localisation solved in the literature? What signal separates "this
      column contains unusual values" from "this column was tampered with"?
      Also: which published system can we actually *run* on our data for G3?
- [ ] **W4.2** — metric transparency: definitions, PR curves, per-column
      precision/recall, false-positive accounting. Answers G1.
- [ ] **W4.3** — cost-sensitivity sweep + proof that costs are fixed and
      never selected on test outcomes. Answers G2.
- [ ] **W4.4** — difficulty-aware repair routing. Implements M1.
- [ ] **W4.5** — column triage implementation, informed by W4.1. Implements M2.

W4.1–W4.4 run in parallel; W4.5 waits on W4.1.


---

## 10. The likelihood-ratio fix was REFUTED (pre-registered test)

W4.1's central finding was that our statistic `1 − P(x | row)` is a *density*,
and Ren et al. (NeurIPS 2019) show the correct statistic is a **likelihood
ratio** against a background model. We pre-registered the refutation condition
in `experiments/w4_ratio_test.py` before running it: *refuted if the top 1,701
ratio-scored cells contain zero `income` cells.*

| mode | PR-AUC all | PR-AUC on `income` | ROC all | `income` in top-1,701 | dominant column |
|---|---:|---:|---:|---:|---|
| `full` (density) | 0.0123 | **0.6873** | 0.8869 | **0** | `native_country` (1,348) |
| `marginal` | 0.0085 | 0.1318 | 0.7162 | **0** | `native_country` (1,490) |
| `ratio` | 0.0081 | 0.4895 | 0.7890 | **0** | `occupation` (949) |

**Refuted.** No mode places a single tampered cell in the top 1,701.

### What the ratio *did* do, and why it still failed

The theory predicted correctly as far as it went: the ratio **de-emphasised
`native_country`**, the column whose background distribution is most extreme
(dominated by "United-States"). That is exactly the confound Ren et al.
describe, and the ratio removed it.

It then handed the top of the ranking to `occupation` instead.

**Diagnosis — the ratio is biased by cardinality.** `income` is **binary**:
with two categories the marginal is ≈ 0.25/0.75, so even a confidently wrong
conditional prediction yields a *bounded* ratio. `occupation` has ~14
categories, so moderate marginals combine with small conditionals to produce
much larger ratios. A binary column **cannot** compete on this statistic no
matter how badly corrupted it is. The ratio is not comparable across columns
of different cardinality.

The ratio is also worse *within* the target column (0.4895 vs the density's
0.6873), so it is not a free swap even where it is applicable.

### What this implies

1. **The honest claim stands and is now backed by a pre-registered test:**
   per-cell surprise is solved (0.6873 inside the right column); locating a
   valid-to-valid corruption from the dirty table alone is not. This agrees
   with W4.1's hard constraint — no published system does it without
   importing outside information (a column corpus, a table corpus, labels,
   or a cost model).
2. **The cheap next test is per-column score normalisation**, not a better
   background model. If each column's scores are standardised to a common
   scale (rank or z-score within column), the top-k would draw across columns
   instead of being captured by whichever column has the highest cardinality.
   That does not *identify* the corrupted column — it only stops high-
   cardinality columns from monopolising the ranking — but it is one cheap
   run and it is falsifiable the same way.
3. **Do not report `ratio` as an improvement anywhere.** It is worse on every
   metric we measured except its intended de-confounding effect, which it
   achieved without benefit.

---

## 11. Wave 5 — the four remaining gaps

Second external review. Assessment by (value / effort) and what each needs.

### Gap 4 — incomplete costs + unablated clean-data knowledge. **Cheapest, do first**

Two separate problems, both one experiment each:

**4a. Undetected corruptions are priced at zero.** Adult excludes all 1,701
injected errors; Breast Cancer excludes 276. Every cost figure therefore
describes the *flagged* cells, not total data-quality harm. **Fix:** add a
second costing mode that charges `C_abstain` for every genuinely corrupted
cell the detector never flagged, and report both. This does not replace the
current number — a policy cannot be blamed for a cell it never saw — but
total harm is the number a practitioner cares about, and omitting it flatters
every method equally while hiding how little of the problem we touch.

**4b. `DatasetProfile` is built from `clean_df`.** Detector fences, valid
ranges, vocabularies and imputation statistics all carry population-level
clean knowledge. CLAUDE.md §3 has flagged this since day one and named the
resolution: *an ablation computing those statistics from dirty data and
reporting the gap.* It has never been run. **Fix:** run it. If the gap is
small the caveat is closed; if it is large, that is a finding we need before
the paper, not after.

### Gap 1 — no external benchmark. **Biggest gap, clear path**

W4.1 established the install path and that Raha's dataset dictionary is
`{name, path, clean_path}` — our layout verbatim. Needs a separate venv
(`raha` pins `pandas==2.1.0`; we are on 2.3). **Fix:** run Raha at a 20-label
budget on all three datasets, report P/R/F1 beside ours *on the same cells
and the same corruption*, and state plainly where it beats us.

### Gap 2 — column triage. **Hardest; needs a hypothesis, not a heuristic**

The reviewer is right that predictability, rarity and cardinality each failed,
and that the answer is not another untested heuristic. The principled position
comes from W4.1's hard constraint: **no published system separates
rare-but-valid from corrupted using one table and no labels.** Every deployed
method imports a reference — a column corpus, a table corpus, labels, or a
cost model.

So there are exactly two honest moves:

1. **Test the one self-contained reference available: synthetic corruption.**
   Inject known corruption into each column and measure how the detector's
   behaviour changes. A column that is *already* corrupted should respond
   *less*, because its baseline is already polluted. W4.1 flagged that the
   sign of this effect is an assumption, not a theorem — so it is
   pre-registered and falsifiable exactly as the likelihood-ratio test was.
   Circularity is the risk and must be controlled: the injected corruption
   policy must differ from the one that produced the dataset.
2. **Accept labels and spend them on columns.** Wave 3 already showed
   per-column allocation is the only sampler that learns on adult. ED2's
   Column Selector is prior art and must be benchmarked against, not
   reinvented.

If (1) fails, the paper's claim becomes a measured impossibility result rather
than a missing feature — which is publishable and honest.

### Gap 3 — one useful dataset. **Needs a decision from the user**

Smart Factory tests trivially-detectable missingness; Adult's whole-table
detector is weak; Breast Cancer is the only dataset exercising the full
pipeline across corruption types. Options, in order of strength:

- **More REIN datasets** (Beers, Flights, Hospital, Movies, Rayyan, Tax are
  standard) — strongest, needs the user to supply or point at them.
- **Controlled corruption study**: take clean data and inject corruption at
  varied types and rates, measuring how performance moves with each.
  Self-contained, but risks circularity and tests generalisation across
  *corruption*, not across *domains*.

### Order

- [x] **W5.1a** complete costing incl. undetected errors — `results/w5_complete_costs.md`
- [x] **W5.1b** soft-leakage ablation — `results/w5_leakage_ablation.md`.
      **Material**: breast_cancer PR-AUC −0.1488 (disjoint CIs), repair top-1
      −0.167; adult and smartfactory exactly 0. Zero worst-case regret holds
      in **both** arms, so the decision-layer claim is untouched.
- [x] **W5.2** Gap 1 — Raha run on our data, `results/w5_raha_comparison.md`
- [x] **W5.2b** leakage check on the Raha datasets —
      `results/w5_raha_leakage_check.md`. **Reverses W5.2**: zero-knowledge
      F1 gives hospital 0.0000, rayyan 0.0558, beers 0.0558 against Raha's
      0.4000 / 0.6457 / 0.7988. We win 1 of 4, not 3 of 4. The detector is
      not better than Raha, it is differently informed.
- [x] **W5.4** Gap 3 — nine-dataset cross-domain run, `results/w5_cross_domain.md`
- [ ] **W5.3** Gap 2 — pre-registered injection test for column triage.
      Plan: `docs/w5_3_column_triage_plan.md` (committed before the code).
      Census first: triage is only well posed on **adult** (1 of 15 columns
      corrupted); breast_cancer has 12 of 12 so the column-level AUC does not
      exist, and smartfactory is 18 of 19.
