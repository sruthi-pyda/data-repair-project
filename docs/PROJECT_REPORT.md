# Risk-Aware Data Repair — complete project report

Everything built so far: the original pipeline (Phases 1–13) and the three
research waves that followed. Every figure here came from code that ran;
nothing is estimated. Where a method lost, the loss is reported.

**Reading order.** §1–§2 are the problem and the governing constraint. §3 is
the original method and its results. §4 explains why we reframed. §5–§7 are
waves 1–3. §8 is the open problem. §9 is how to reproduce any number.

---

## 1. The problem

Given a table with corrupted cells, decide for each suspicious cell whether to

- **repair it automatically**,
- **route it to a human**, or
- **leave it alone**.

Three benchmark datasets (from REIN, which supplies the *data* only — no costs
and no baselines come from it):

| dataset | rows × cols | corrupted cells | prevalence | corruption type |
|---|---|---:|---:|---|
| Breast Cancer | 699 × 12 | 631 | 7.52 % | missing, format errors, out-of-range |
| Smart Factory | 23,645 × 19 | 7,093 | 1.58 % | injected missing values |
| Adult | 45,222 × 15 | 1,701 | 0.25 % | income-label flips (valid → valid) |

**The research claim is not "we repair data well."** It is that a *calibrated*
probability of correctness, combined with *explicit costs*, produces better
decisions than acting on raw confidence.

---

## 2. The governing constraint, measured first

Before optimising anything we measured what is physically recoverable. Repair
accuracy, split by corruption type:

| corruption | repair accuracy | nature |
|---|---:|---|
| Out-of-range value | **98.4 %** | recoverable — the original is implied by context |
| Format error (`1a0`) | **100 %** | recoverable |
| Missing value (Breast Cancer) | 50.0 % | **destroyed** — nothing left to recover |
| Missing value (Smart Factory) | 28.8 % | **destroyed** |

**You cannot recover a destroyed value.** Repair accuracy therefore mostly
measures a dataset's mix of recoverable vs destroyed corruption, not the
quality of the repairer. Chasing it chases the dataset.

This single table drives every later decision, including why an LLM could not
help much (§3.5) and why the contribution has to live in the decision layer.

---

## 3. The original method (Phases 1–13)

### 3.1 Architecture

```
dirty.csv
 ├─ 1–2  load + profile        → column kinds, quartiles, vocabularies
 ├─ 3    detect                → 6 rule checks (missing, range, IQR, format,
 │                                novel-category, whitespace)
 ├─ 4    repair                → top-k candidates with probabilities
 ├─ 5    uncertainty           → raw confidence, self-consistency, entropy
 ├─ 6    calibrate p           → P(repair is correct)
 ├─ 6b   calibrate q           → P(cell is genuinely corrupted)
 ├─ 7–8  cost model + policy   → argmin over three expected costs
 ├─ 9–10 baselines + evaluation
 ├─ 11   Streamlit dashboard
 ├─ 12   real LLM repair (Groq)
 └─ 13   reproducible experiment runner
```

### 3.2 The decision rule

```
E[AUTO_REPAIR]  = (1 − p) · C_wrong
E[HUMAN_REVIEW] = C_human
E[ABSTAIN]      = q · C_abstain
action = argmin
```

Costs are **abstract experimental penalty units**, never monetary or clinical.
Derived boundaries, verified against the implementation on a 201×201 grid per
dataset with **0 mismatches out of 40,401 each**:

```
AUTO vs HUMAN     p* = 1 − C_human / C_wrong          (vertical)
HUMAN vs ABSTAIN  q* = C_human / C_abstain            (horizontal)
AUTO vs ABSTAIN   p  = 1 − q·C_abstain / C_wrong      (sloped)
```

**Stated limitation:** `q*` is horizontal, so for a single cell with fixed `q`
one of HUMAN/ABSTAIN is unreachable — both are constant in `p` and never
cross. The mechanism changes *which* of the two is live, per cell, on
evidence. "All three actions reachable" is a claim about the population.

### 3.3 Headline result — zero worst-case regret

Realised cost (experimental units), full datasets, seed 42:

| policy | Adult | Breast Cancer | Smart Factory | **worst-case regret** |
|---|---:|---:|---:|---:|
| **Proposed** | **0** ‡ | **1,615** | **14,186** | **0** |
| Always human review | 11,658 | 1,775 | 14,186 | 11,658 |
| Raw confidence ≥ 0.70 | 11,658 | 1,720 | 14,186 | 11,658 |
| Always abstain | 0 | 3,550 | 35,465 | 21,279 |
| Always auto-repair | 58,290 | 5,800 | 126,275 | 112,089 |

The proposed policy attains the best cost on **every** dataset, so its
worst-case regret is zero. No fixed rule manages that.

**Per dataset, without spin:** Breast Cancer is an outright win (1,615 vs
1,720). Adult **ties** always-abstain at 0 while beating everything else.
Smart Factory is a **flat tie** with always-human — no improvement.

‡ Adult's 0 is *zero residual-corruption cost on the flagged cells only*.
Adult's 1,701 genuine corruptions were never detected and are priced at zero
for every method. Always report both.

### 3.4 Calibration, with confidence intervals (20 splits, bootstrap)

| dataset | method | Brier | splits fitted |
|---|---|---|---|
| Breast Cancer | raw | 0.1982 [0.1936, 0.2029] | — |
| Breast Cancer | logistic | 0.1979 [0.1920, 0.2040] | 20/20 |
| Breast Cancer | **isotonic** | **0.1842 [0.1787, 0.1900]** | 20/20 |
| Breast Cancer | multivariate | 0.1968 [0.1910, 0.2029] | 20/20 |
| Smart Factory | raw | 0.2732 [0.2727, 0.2737] | — |
| Smart Factory | **isotonic** | **0.1632 [0.1615, 0.1649]** | 20/20 |
| Smart Factory | multivariate | 0.1883 [0.1872, 0.1894] | 20/20 |
| Adult | all | **—** | **0/20** |

**What survives:** isotonic beats raw on both fittable datasets, intervals
non-overlapping.

**What did not survive, including our own idea:**

- **Logistic vs raw on Breast Cancer is noise** (0.1979 vs 0.1982). We had
  been quoting this as an improvement from a single split. It did not survive.
- **The multivariate calibrator is not an improvement.** Indistinguishable
  from raw on Breast Cancer Brier, and *worse* than isotonic on Smart Factory
  (0.1883 vs 0.1632; ECE 0.1556 vs 0.0127). It belongs in the paper as a
  **failed ablation**, not a contribution.
- **Adult cannot be calibrated at all** — 0 of 20 splits fitted. Printed `—`,
  never a number.

### 3.5 Phase 12 — the real LLM

392 live calls on Breast Cancer, GPT-OSS-120B via Groq, 0.5 % fallback.

| measure | mock repairer | real LLM |
|---|---:|---:|
| Top-1 repair accuracy | 0.6732 | **0.7099** |
| Proposed policy cost | **1,615** | 1,670 |
| Best simple baseline | 1,720 | **1,620** |
| verdict | proposed wins | **proposed loses by 50** |

**The LLM repairs better and the policy does worse.** Controlled comparison on
80 cells by corruption type explains why the upside is capped:

| corruption | mock | LLM |
|---|---:|---:|
| Destroyed (missing) | 45 % | 50 % — noise, both guessing |
| Recoverable (out-of-range) | **100 %** | **92.5 % — the LLM is worse** |

The LLM reasons itself out of deterministic cases a rule gets right every
time. On destroyed values nothing can help (§2).

### 3.6 Engineering

~1,850 tests at the Phase 12 branch tip. One command regenerates every table
and figure deterministically, with git commit, seeds and library versions
recorded. Undefined metrics print `—`, never `0.0`. Tests re-derive expected
answers independently and were mutation-checked by deliberately breaking the
code to confirm the right tests fail.

A hard invariant throughout: **no ground-truth leakage.** Clean data produces
0/1 labels only, on a training split, only for the two calibrators. Verified
empirically — scrambling the clean values leaves detections, repairs and
uncertainty signals byte-identical.

---

## 4. Why we reframed

Three facts from §2–§3 pointed the same way:

1. Repair accuracy is mostly a property of the dataset, not the method.
2. The decision layer only improves things where detection works.
3. **Detection was the broken link.** On Adult the rule detector scored
   PR-AUC 0.0025 — *exactly its prevalence*, i.e. a constant scorer — with
   ROC-AUC **0.4957**, slightly *anti*-correlated with corruption. It found
   0 of 1,701 real errors and raised 5,829 false alarms.

So the waves optimise **detection**: `P(cell is corrupted)`. It is the genuine
prediction problem, it is severely imbalanced (PR-AUC is the right metric),
and it is what published systems benchmark.

### Measured detection baseline (5 stratified cell splits, seed 42)

| dataset | prevalence | PR-AUC [95 % CI] | ROC-AUC | best-F1 (oracle) |
|---|---:|---:|---:|---:|
| Smart Factory | 0.0158 | 1.0000 | 1.0000 | 1.0000 |
| Breast Cancer | 0.0752 | 0.6712 [0.6486, 0.6917] | 0.8222 | 0.7835 |
| Adult | 0.0025 | **0.0025** | **0.4957** | 0.0050 |

Smart Factory's perfect score confirms the *dataset* is uninformative (the
corruption is literally blank cells), not that the detector is good.

**Structural insight:** a rule detector is a **set**, not a **ranking** —
Breast Cancer has only 5 distinct scores, 95.3 % of cells sharing one. PR-AUC
measures it on an axis it does not attempt. That gap is the headroom.

### What the literature says (`docs/literature.md`)

- **Our Adult failure is the published behaviour of this method class.**
  HoloDetect (SIGMOD 2019) reports outlier detection on Adult at **F1 0.002**,
  while its learned detector reaches 0.991 on the same table.
- **No published Adult number is comparable to ours** — theirs is typos and
  rule violations across many columns; ours is label flips in one column.
- **Factors that carry the edge:** conditional/row-context modelling
  (HoloDetect ablation 0.971 → 0.884), the first ~20 labels (Raha: 0.40 →
  0.85 F1), synthetic augmentation (+20 F1).
- **Novelty correction:** calibrating a detector is *not* novel (HoloDetect
  2019), and Conformal Data Cleaning (AISTATS 2024) already pairs per-column
  conditional models with calibrated uncertainty. What is defensible:
  *measuring* calibration quality (0 of 7 surveyed papers do) and the
  **three-way action space** — REIN itself lists "only a binary decision per
  cell" as a field shortcoming.

---

## 5. Wave 1 — conditional modelling

**Idea.** A rule asks "is this value malformed?" Adult's corruption is a valid
value swapped for another valid value, so no rule can see it. Instead ask
**"does this value make sense given the rest of its row?"** — model
`P(cell | rest of row)` and flag improbable values.

Implemented self-supervised (fits on dirty data only), with a **marginal
rarity control** so the ablation could refute the idea.

| dataset | rule detector | **conditional** | marginal control |
|---|---:|---:|---:|
| Adult | 0.0025 (ROC 0.4957) | **0.0120** (ROC **0.8875**) | 0.0085 |
| Breast Cancer | **0.6712** | 0.6404 | 0.6325 |
| Smart Factory | 1.0000 | 1.0000 | 1.0000 |

On Adult's `income` column alone: conditional **0.6997** vs marginal **0.1321**.

**Findings.**

1. **Conditioning is the factor**, confirmed by the control — non-overlapping
   CIs, 5× on the income column. ROC-AUC 0.4957 → 0.8875: from
   anti-correlated to a genuinely good ranking.
2. **It beats the literature's derived ceiling** (F1 ≈ 0.3–0.4, predicted from
   Adult classification accuracy). We measure oracle F1 0.6758, because soft
   probabilities separate better than hard classification. *Caveat: that is an
   oracle threshold, not an operating point.*
3. **On Breast Cancer it LOSES to the rule detector** (0.6404 vs 0.6712) and
   `full ≈ marginal`, so conditioning adds nothing where corruption is
   syntactic. Reported as a loss.

**Synthesis:** rules catch syntactic corruption; conditional models catch
semantic corruption; **neither wins everywhere.**

---

## 6. Wave 2 — combine them, and spend a label budget

| dataset | rules | conditional | **noisy-OR, 0 labels** | 100 labels | full supervision |
|---|---:|---:|---:|---:|---:|
| Breast Cancer | 0.6712 | 0.6361 | **0.7209 [0.6977, 0.7371]** | 0.7617 | 0.7898 |
| Adult | 0.0025 | 0.0110 | 0.0106 | **0.2656** | **0.7024** |

**Findings.**

1. **Complementarity is real and free on Breast Cancer.** Noisy-OR at *zero
   labels* beats both single detectors with non-overlapping CIs.
   Rank-averaging is worse (0.6954) — it inflates near-zero scores.
2. **On Adult the fusion does NOT win** — slightly worse than the conditional
   model alone label-free. **Labels** win there.
3. **NOVEL NEGATIVE: Raha's 2→20 label result does not transfer to extreme
   imbalance.** At 0.251 % prevalence, k ≤ 20 buys *literally nothing* — the
   sampler draws **zero positives in 5/5 splits**. The field measures label
   efficiency at 5–40 % error rates; nobody appears to have measured 0.25 %.
4. **The sampler, not the learner, is the bottleneck** (b100 0.2656 vs full
   0.7024).
5. **Whole-table and per-column metrics disagree** about which method is
   better. Both must be reported.

**Measurement catch that would have faked a result:** conditional features are
scored out-of-fold. A single in-sample fit scores 0.5014 on Breast Cancer vs
0.6408 out-of-fold — in-sample would have handed us a handicapped baseline and
a fabricated improvement.

---

## 7. Wave 3 — where should a tiny label budget be spent?

Nine acquisition strategies, budgets 10–500, one-shot vs sequential.

**Breast Cancer is degenerate for this question**: the label-free combiner
(0.7209) already exceeds 80 % of full supervision (0.632), so every strategy
"reaches" the target at k=10. Adult is the real test (b0 0.0106, full 0.7024).

### Adult — labels needed to reach 50 % of full supervision

| strategy | rounds | labels | best PR-AUC | positives/query |
|---|---|---:|---:|---:|
| **`batch_diverse`** | **5** | **100** | 0.5408 | 0.048 |
| `per_column_quota` | 5 | 100 | 0.5021 | 0.041 |
| `batch_diverse` | 1 | 500 | 0.4748 | 0.056 |
| `random` | 1 | **never** | 0.2529 | 0.002 |
| `global_top` / `conditional_top` / `uncertainty` | 1 | **never** | **0.0106** | **0.000** |

**Findings.**

1. **Sequential acquisition is worth a 5× label reduction** (k=100 vs k=500).
2. **Nothing reaches 80 % of full at any budget ≤ 500.** Honest ceiling.
3. **The top of a well-ranked list is not where the positives are.**
   `conditional_top` queries **zero positives at every budget** despite the
   model's good overall ranking.
4. **Diversity, not confidence, is what works.** Every
   concentrate-on-the-top strategy returns exactly the label-free score.
5. **A strategy can be too good at finding positives.** On Breast Cancer
   `conditional_top` achieves **1.000 positives per query** (vs random's
   0.078) and still fails to beat the label-free combiner — a sample with no
   negatives gives a discriminative classifier nothing to learn.
   Acquisition precision and learner value pull in opposite directions.
6. **Random is not a weak control** — it attains the best Breast Cancer
   PR-AUC at k=500 (0.7688). It collapses only at extreme imbalance.

---

## 8. The open problem: column-level triage

Adult's overall PR-AUC stays at 0.0123 while the income column alone reaches
0.6873. An earlier draft called this "dilution". **That was wrong**, and
measuring it properly changed the conclusion:

| scored set | PR-AUC |
|---|---:|
| all 15 columns | 0.0123 |
| `income` only | 0.6873 |
| all 15, other 14 forced to score 0 | **0.6873** |

The last two being identical proves the entire loss comes from the untouched
columns and **vanishes if the detector is silent about them**. Clean columns
do not dilute PR-AUC by existing.

**The real mechanism is displacement.** Of the top 5,000 scored cells, **zero**
are in `income`. The top 1,701 slots are:

| column | cells | true positives |
|---|---:|---:|
| `native_country` | 1,348 | 0 |
| `occupation` | 343 | 0 |
| `marital_status` | 8 | 0 |
| `workclass` | 2 | 0 |
| **`income`** | **0** | — |

`native_country` is overwhelmingly "United-States", so a rare value is
genuinely improbable given the row. The model is **correct** that those cells
are surprising — they are also completely clean.

**Consequences.**

- The 0.6873 figure is **partly oracle** — it requires knowing corruption is
  confined to `income`. Never quote it as an unaided result.
- The defensible claim is: *the conditional model ranks corruption well
  within a column, but cannot tell which column is worth looking at.*
- This explains wave 3 exactly, and it explains why budget-spreading
  strategies won — they were doing crude column triage by accident.
- **The open problem is column-level triage**, not better per-cell surprise.

---

## 8b. Wave 4 — answering external review

An external reviewer raised three gaps; two further mechanisms were proposed.

### G2 — are the costs defensible? **Yes, and the headline is robust**

Realised cost is homogeneous of degree 1 in the costs, so the decision
boundaries depend only on two *ratios*. Sweeping those on a grid:

| dataset | best or tied | **strictly best** | worst regret |
|---|---:|---:|---:|
| Adult | 100.0 % | **0.0 %** | 0.00 % |
| Breast Cancer | 96.5 % | **57.9 %** | 4.68 % |
| Smart Factory | 98.5 % | **2.9 %** | 0.65 % |

The zero-regret claim is **not knife-edge** — where it loses it loses by
single-digit percentages in narrow slivers. **But read the second column:**
Adult's 100 % is 100 % *ties* with always-abstain, and Smart Factory is mostly
tied with always-human. Robustness and superiority are different claims.

Provenance is proved four ways: the costs are frozen dataclasses read from
`configs/costs.yaml`; no optimiser touches them; and the one historical
exception is stated openly — Adult's `c_leave_error` was hand-set to 1 and
**reverted to 5 one commit later**. Validation: 0 mismatches against
`DecisionPolicy.decide` over every record × 13 cost configs.

### M1 — difficulty-aware repair routing. **Works, but not as expected**

The LLM is worse where answers are determined (92.5 % vs the rule's 100 %) and
marginally better where they are guesses. So route by difficulty:

| approach | accuracy | model calls | policy cost |
|---|---:|---:|---:|
| Rule only | 67.3 % | 0 | **1,615** |
| LLM only | 71.0 % | 355 | 1,670 |
| **Routed by error type** | **71.0 %** | **210** | 1,680 |
| Oracle (upper bound) | 77.7 % | 355 | 1,775 |

**Routing matches the LLM's accuracy with 41 % fewer model calls** — the
practical win. It does not beat the LLM on accuracy or the rule on cost.

**The sharpest finding in the project:** *better repair accuracy produces worse
policy cost.* The least accurate method is cheapest; the oracle is most
accurate and most expensive. Better repairs raise `p`, more cells auto-repair,
and a wrong auto-repair costs 50 against human review's 5. The LLM did not
fail — it succeeded at the wrong objective.

Also exposed a concrete bug: `by_error_type` sends `NOVEL_CATEGORY` to the
rule as "recoverable", scoring 30.8 % where the LLM scores 84.6 %.

### M2 — column triage. **Research reframed it; the proposed fix was refuted**

Three findings from the literature route:

1. **The problem is named** — *error localisation* (Fellegi–Holt, JASA 1976,
   still deployed by national statistical agencies), *cellwise outlier
   detection* (Technometrics 2018), *error attribution* (SIGMOD 2015).
2. **Our statistic is formally wrong.** `1 − P(x | row)` is a **density**, and
   Ren et al. (*Likelihood Ratios for OOD Detection*, NeurIPS 2019) show
   density is confounded by background statistics; the correct statistic is a
   **likelihood ratio**. That is exactly our `native_country` failure, named
   and solved in another field.
3. **Novelty correction:** ED2 (CIKM 2019) already contains a component called
   the **Column Selector**, whose *Max Error* criterion is the
   predictability-shift idea we were about to claim — and ED2 reports it gets
   stuck in local optima. We would have claimed known prior art *and* walked
   into a documented failure mode.

**The ratio fix was then tested and REFUTED**, against a condition registered
before the run (`experiments/w4_ratio_test.py`):

| mode | PR-AUC all | PR-AUC on `income` | `income` in top 1,701 | dominant column |
|---|---:|---:|---:|---|
| density (current) | 0.0123 | **0.6873** | **0** | `native_country` |
| marginal | 0.0085 | 0.1318 | **0** | `native_country` |
| likelihood ratio | 0.0081 | 0.4895 | **0** | `occupation` |

The theory was right as far as it went — the ratio **did** de-emphasise
`native_country`, the most background-skewed column. It then handed the top to
`occupation`.

**Diagnosis: the ratio is biased by cardinality.** `income` is binary, so even
a confidently wrong prediction gives a *bounded* ratio; a 14-category column
produces far larger ones. **A binary column cannot compete on this statistic
however badly corrupted it is.** The density favours the most skewed column;
the ratio favours the highest-cardinality one. Neither is asking about
corruption.

### G3 — novelty. **Now closable, not yet closed**

`Raha` (SIGMOD Most Reproducible Paper Award, last commit 2025-06-05) installs
with `pip install raha`, and its dataset dictionary is `{name, path,
clean_path}` — **our `datasets/<name>/{dirty,clean}.csv` layout verbatim**.
REIN ships a runnable harness wrapping eleven detectors. **Not yet run**: this
remains the project's largest open gap.

### G1 — metric transparency

`docs/metrics_definitions.md` gives the exact definition of every metric we
report, states that **PR-AUC is primary and a constant scorer achieves
PR-AUC = prevalence** (so Adult's 0.0025 is the *floor*, not a score), and
shows why accuracy misleads at 0.25 % prevalence — a detector flagging nothing
scores 99.75 % accuracy on Adult. Per-column precision/recall and
false-positive accounting tables are generated by
`experiments/w4_metrics.py`.

---

## 9. Reproducing any number

```bash
pip install -r requirements.txt
python -m pytest -q                              # full suite

python run_experiment.py --all --splits 20       # §3.3, §3.4 (~11 min)
python run_experiment.py --dataset breast_cancer --mode real --llm-live   # §3.5

python experiments/three_actions_ablation.py     # decision-layer ablation
python experiments/smoothing_ablation.py         # prior robustness
python experiments/r2_conditional.py             # §5
python experiments/w2_ensemble.py                # §6
python experiments/w3_acquisition.py             # §7

streamlit run app.py                             # interactive dashboard
```

Seed 42 throughout; re-runs are byte-identical. Outputs land in `results/`.

### Companion documents

| file | contents |
|---|---|
| `docs/research_program.md` | wave-by-wave record with full tables |
| `docs/literature.md` | published numbers, factor decomposition, novelty position |
| `docs/cost_semantics.md` | what each cost means; read before writing results |
| `docs/three_actions_plan.md` | the calibrated-abstention design record |
| `results/RESULTS.md` | Phase 13 results with confidence intervals |

---

## 10. Standing limitations

- **Mock repairs dominate the results.** Only one dataset has been run through
  the real LLM. The title's "LLM-Assisted" is only partly earned.
- **One usable dataset.** Adult's corruption is invisible to detection;
  Smart Factory's is entirely destroyed values. Breast Cancer is the only
  dataset where the whole loop closes.
- **No external benchmark comparison.** Every baseline we beat is one we
  wrote. Published systems have not been run on our corruption protocol.
- **False negatives are priced at zero** — 1,701 on Adult, 276 on Breast
  Cancer — so every cost figure describes the flagged cells only.
- **One cost configuration per dataset.** No sensitivity sweep yet.
- **Soft leakage:** `DatasetProfile` is built from clean data, so detector
  fences and imputation statistics carry population-level clean knowledge.
  Documented, identical across all methods, not yet ablated.
