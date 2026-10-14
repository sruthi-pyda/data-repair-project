# R1 — Literature decomposition

Route R1 of `docs/research_program.md`. **Research only; no implementation.**
Every number below is traced to a source. Anything I could not verify is marked
`not found` or flagged **[UNVERIFIED]**. No figure in this document is estimated
or inferred unless it is explicitly labelled *derived*.

Compiled 2026-10-14.

---

## Executive summary

1. **Our Adult F1 = 0.000 is the published behaviour of our method class, not a bug.**
   HoloDetect reports outlier detection on Adult at **F1 = 0.002** (P 0.999 / R 0.001),
   while its learned detector reaches **0.991** on the same data. The headroom is real
   and the mechanism that closes it is known.
2. **No published Adult number is comparable to ours.** Every Adult in the literature
   is corrupted with typos/outliers/rule violations spread over many columns; ours is
   **100 % label flips confined to `income`** (1,701 / 678,330 cells = 0.2508 %, measured).
3. **Breast Cancer has real published anchors.** SAGED (EDBT 2024) reports per-cell
   detection on it: SAGED **0.95**, dBoost **0.70**, Min-k 0.27, Raha 0.18, ED2 0.18,
   HoloClean 0.14, IQR 0.05. **Our 0.720 would sit mid-table** — but at 7.52 % error
   rate vs their 40 %, so it is an anchor, not a win.
4. **The edge comes from three factors**, in this order: conditional (row-context)
   modelling; the first ~20 human labels (Raha 2 labels ≈ 0.40 F1 → 20 labels 0.85);
   and synthetic-corruption augmentation (HoloDetect: +20 F1 points).
5. **Nobody reports PR-AUC.** All seven systems papers I read in full use precision /
   recall / F1; Conformal Data Cleaning uses TPR/FPR. Our PR-AUC-primary rule is
   methodologically right but makes us incomparable — we must report **both**.
6. **Detector-score calibration is NOT novel** — HoloDetect already applies Platt
   scaling (§4.2). But **no paper evaluates calibration quality** (no ECE, Brier, or
   reliability diagram anywhere), and **none** turns a calibrated score into a
   cost-weighted three-way action. That gap is where our contribution survives.
7. **Biggest risk to our plan:** Conformal Data Cleaning (AISTATS 2024) already pairs
   per-column conditional models with calibrated uncertainty for detection. R2 must be
   positioned against it, not presented as new.

---

## 1. Systems table

Supervision column: **none** = no labels and no user config; **config** = user supplies
rules/KB/hyperparameters; **few labels** = ~20 labelled tuples; **full** = labelled
training split; **interactive** = active learning loop.

| System | Venue / year | Core mechanism (one sentence) | Supervision |
|---|---|---|---|
| **NADEEF** | SIGMOD 2013 ([ACM](https://dl.acm.org/doi/10.1145/2463676.2465327)) | Holistic rule engine: user implements denial constraints and user-defined functions, cells violating them are flagged. | **config** (FD rules, patterns) |
| **HoloClean** | PVLDB 10(11) 2017 | Combines denial constraints with statistical correlations in a probabilistic graphical model to infer both errors and repairs. | **config** (denial constraints) |
| **dBoost** | 2016; as used in [REIN](https://openproceedings.org/2023/conf/edbt/3-paper-49.pdf) | Ensemble of histogram / Gaussian / multivariate-Gaussian-mixture outlier models, hyperparameters found by random search. | **config** (hyperparams; auto-searched) |
| **KATARA** | 2015; as used in REIN | Aligns the table against crowdsourced knowledge bases and flags cells violating semantic patterns. | **config** (knowledge base) |
| **Uni-Detect** | SIGMOD 2019, pp. 811–828 ([ACM](https://dl.acm.org/doi/10.1145/3299869.3319855)) | "What-if" local perturbation plus classical hypothesis tests calibrated on a large corpus of tables, detecting FD violations, outliers and spelling errors uniformly. | **none** (corpus-driven, no per-table config) |
| **Raha** | SIGMOD 2019, pp. 865–882 (DOI [10.1145/3299869.3324956](https://doi.org/10.1145/3299869.3324956)) | Runs a limited set of configurations of many base detection algorithms to build a feature vector per cell, then a sampling + classification scheme learns from very few labels. | **few labels** ("no more than 20 labeled tuples") |
| **Baran** (*repair*, not detection) | PVLDB 13(12):1948–1961, 2020 ([PDF](http://www.vldb.org/pvldb/vol13/p1948-mahdavi.pdf)) | Trains incrementally-updatable value-based, vicinity-based and domain-based corrector models and ensembles their candidate corrections. | **few labels** (θ_Labels = 20) |
| **HoloDetect** | SIGMOD 2019 ([PDF](http://www.holoclean.io/files/misc/detection.pdf), [arXiv 1904.02285](https://arxiv.org/abs/1904.02285)) | Learns a representation of attribute / tuple / dataset context, then uses weakly-supervised learned augmentation policies to synthesise errors and train a classifier; output calibrated with Platt scaling. | **full** split (5–10 %) + weak supervision |
| **ED2** | as used in REIN / SAGED | Extracts attribute-, tuple- and dataset-level features and uses active learning to request labels for cells the classifier is uncertain about. | **interactive** (active learning) |
| **Picket** | VLDB J 31(5):927–955, 2022 ([arXiv 2006.04730](https://arxiv.org/abs/2006.04730)) | Self-supervised multi-head-attention transformer (PicketNet) over mixed-type tabular data; flags cells by reconstruction loss, at both training and inference time. | **none** (self-supervised) |
| **AimNet** | MLSys 2020 ([PDF](https://cs.uwaterloo.ca/~ilyas/papers/WuMLSys2020.pdf)) | Dot-product attention over the *other* attributes of the row (attention weights depend on attributes, not target values) acting as a hybrid denoising autoencoder for imputation. | **none** (self-supervised) |
| **Min-k / Max-Entropy** | 2016; as used in REIN | Ensembles non-learning detectors: Min-k flags a cell found by ≥ k methods; Max-Entropy uses entropy-based sampling to order the detectors. | **config** (hyperparams) |
| **Metadata-driven** | 2016; as used in REIN | One binary feature per non-learning detector plus a metadata profiler, fed to a classifier. | **few labels** |
| **SAGED** | EDBT 2024 ([PDF](https://openproceedings.org/2024/conf/edbt/paper-95.pdf)) | Meta-learning: an ensemble of models pre-trained on *historical* datasets is matched to the new table by similarity, and their predictions form the feature vector for a meta-classifier. | **few labels** (20, fixed) |
| **ZeroED** | [arXiv 2504.05345](https://arxiv.org/abs/2504.05345) | LLM reasons out error-detection guidelines and criteria, labels ~5 % of tuples, and those LLM labels train a classifier over cell + correlated-attribute features. | **none human** (LLM labels ~5 %) |
| **ForestED** | [arXiv 2512.07246](https://arxiv.org/abs/2512.07246) | Ensembles LLM-induced decision trees over data partitions, with EM to weight partition reliability. | ~100 labelled records |
| **Conformal Data Cleaning (CDC)** | AISTATS 2024, PMLR v238 ([PDF](https://proceedings.mlr.press/v238/jager24a/jager24a.pdf)) | One AutoGluon model per column predicts that column from the others; conformal prediction calibrates each model's uncertainty, and a cell is flagged when its observed value falls outside the prediction set. | **none** (needs a clean training split; no labels for errors) |
| **REIN** (*benchmark*) | EDBT 2023, pp. 499–511 ([PDF](https://openproceedings.org/2023/conf/edbt/3-paper-49.pdf)) | Benchmark harness: 19 detectors × 19 repair methods × 14 datasets, scored both standalone and by downstream ML performance. | n/a |

---

## 2. Published numbers

### 2.1 The critical result — Adult

**HoloDetect, SIGMOD 2019, Table 2.** Adult = 97,684 rows × 11 attributes,
**1,062 erroneous cells** (0.099 % of cells, *derived*), errors injected with BART:
**70 % typos + 30 % value swaps**, spread across attributes. 5 % of the data used as
the training split; active learning at k = 100 (≤ 50 cells per iteration).

| Method | Precision | Recall | **F1** |
|---|---:|---:|---:|
| **AUG** (HoloDetect) | 0.994 | 0.987 | **0.991** |
| ActiveL (active learning, k=100) | 0.994 | 0.982 | **0.988** |
| CV (constraint violations) | 0.497 | 0.998 | 0.664 |
| HC (HoloClean) | 0.893 | 0.392 | 0.545 |
| SuperL (supervised, 5 % labels) | 0.999 | 0.350 | 0.519 |
| FBI (forbidden itemsets) | 0.990 | 0.254 | 0.405 |
| LR (logistic reg. on engineered features) | 0.051 | 0.072 | 0.059 |
| **OD (outlier detection)** | 0.999 | **0.001** | **0.002** |
| SemiL (semi-supervised) | — | — | n/a (did not terminate in 2 days) |

> **This is the headline.** `OD` is the family our rule detector belongs to, and it
> scores **F1 = 0.002** on Adult — recall 0.001. Our measured 0.000 is the same
> phenomenon. A learned, row-context detector on the *same* table reaches 0.991.

**REIN, EDBT 2023, §6.2 (Figure 2f).** Adult = 45,223 × 15, error rate **0.58**,
errors = **rule violations + outliers**, 11 detectors:

| Method | **F1** | Note (quoted from REIN) |
|---|---:|---|
| RAHA | **0.80** | "both of RAHA and ED2 outperform all other methods" |
| ED2 | **0.78** | required ~35 min on average |
| HoloClean / NADEEF / Min-k | not found (numeric) | "captured most of the rule violations only" |
| dBoost | not found (numeric) | "captured most of the outliers while failed to identify the rule violations" |
| Picket | **failed** | "terminated since it caused memory faults" on Adult and Smart Factory |

Also from REIN's discussion (§7): HoloClean's Adult F1 **drops from 0.51 to 0.12 when
the supplied rules are reduced from 17 to 7** — rule-based detection is only as good as
the hand-written rule set.

**SAGED, EDBT 2024.** Adult (45,223 × 15, error rate **0.09**, RV + OT) was held in the
**historical inventory**, so no target F1 is reported for it. `not found`.

### 2.2 Breast Cancer

**SAGED, EDBT 2024, Table 2.** Breast Cancer = 700 × 12, error rate **0.40**,
error types **MV + typos + outliers**. Fixed budget of **20 user labels** for SAGED and
all ML-based tools.

| Method | Precision | Recall | **F1** | Runtime (s) |
|---|---:|---:|---:|---:|
| **SAGED** | 0.99 | 0.92 | **0.95** | 3.73 |
| dBoost | 0.54 | 0.99 | **0.70** | 69.24 |
| KATARA | 0.35 | 0.78 | 0.48 | 103.35 |
| Min-k | 0.86 | 0.20 | 0.27 | 263.62 |
| RAHA | 0.84 | 0.10 | 0.18 | 1.41 |
| ED2 | 0.84 | 0.10 | 0.18 | 86.10 |
| HoloClean | 1.00 | 0.07 | 0.14 | 0.71 |
| FAHES | 0.11 | 0.04 | 0.06 | 0.57 |
| IQR | 0.34 | 0.03 | 0.05 | 0.35 |
| SD | 1.00 | 0.01 | 0.03 | 0.34 |
| IF (isolation forest) | 0.93 | 0.01 | 0.01 | 1.16 |

**REIN, EDBT 2023.** Breast Cancer = 700 × 12, error rate **0.08**, MV + typos +
outliers. REIN reports **no detection F1 for Breast Cancer** in the text — only repair
results (Figure 4c). `not found`.
One usable fragment: KATARA on Breast Cancer produced **6,843 false positives and 86
false negatives** (all numerical). *Derived* from that: with 8,400 cells at 8 % error
(≈ 672 errors) → P ≈ 0.079, R ≈ 0.872, F1 ≈ 0.145. **Flagged as derived, not published.**

### 2.3 Smart Factory

| Source | Error rate | Best detector | **F1** |
|---|---:|---|---:|
| REIN §6.2 (Figure 2h) | 0.153 (MV + outliers) | Min-k | **0.75** |
| REIN | — | KATARA | "generated many false positives" (no number) |
| SAGED | 0.83 (MV + OT) | — | not found (not in Table 2) |

### 2.4 Other datasets — for factor evidence only

**Raha's own reported numbers, at 20 labelled tuples**, as tabulated by BiRNN
(EDBT 2022, Table 2, [PDF](https://openproceedings.org/2022/conf/edbt/paper-40.pdf)):

| Dataset | Beers | Flights | Hospital | Movies | Rayyan | Tax | **avg** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Raha (20 labels)** | 0.99 | 0.81 | 0.72 | 0.86 | 0.79 | 0.91 | **0.85** |
| Rotom | 0.99 | n/a | 1.00 | 0.68 | 0.86 | 0.97 | 0.90 |
| ETSB-RNN | 0.98 | — | — | — | — | — | 0.91 |

**The same Raha, reproduced by others at 2 labelled tuples** (ZeroED Table III;
ForestED Table 3 — both state "2 labeled tuples per dataset for methods needing
manual labels"):

| Dataset | Hospital | Flights | Beers | Rayyan | Movies | Billionaire | Tax |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raha (ZeroED repro, 2 labels) | 0.125 | 0.591 | 0.685 | 0.422 | 0.373 | 0.174 | — |
| Raha (ForestED repro, 2 labels) | 0.140 | 0.430 | 0.680 | 0.570 | 0.250 | 0.130 | 0.487 |
| **ZeroED** (LLM labels ~5 %) | 0.811 | 0.722 | 0.774 | 0.732 | 0.765 | 0.767 | 0.505 |
| **ForestED** (~100 labels) | 0.756 | 0.720 | 0.947 | 0.704 | 0.735 | 0.566 | 0.838 |
| dBoost (ForestED repro) | 0.112 | 0.446 | 0.210 | 0.352 | 0.112 | 0.220 | 0.371 |

### 2.5 Protocol comparability — **read this before citing any number above**

Our own corruption, **measured** from `datasets/*/{clean,dirty}.csv`:

| our dataset | rows × cols | cells | errors | cell error rate | columns affected |
|---|---|---:|---:|---:|---|
| adult | 45,222 × 15 | 678,330 | 1,701 | **0.2508 %** | **`income` only** |
| breast_cancer | 699 × 12 | 8,388 | 631 | **7.5227 %** | all 12 |
| smartfactory | 23,645 × 19 | 449,255 | 388,997 | **86.5871 %** | all 19 |

| Dataset | Our protocol | Published protocol | Comparable? |
|---|---|---|---|
| **Adult** | 0.25 % cells, **100 % label flips in one column** (valid→valid) | HoloDetect: 0.099 %, 70 % typos + 30 % swaps, many columns · REIN: 58 %, rule violations + outliers · SAGED: 9 %, RV + OT | **NO.** No published Adult shares our corruption. Our F1 cannot be compared to any of them. The *mechanism* comparison (OD 0.002 vs AUG 0.991) still transfers. |
| **Breast Cancer** | 7.52 %, all columns, MV + format + out-of-range | REIN: **8 %**, MV + typos + outliers → closest match, but **no detection F1 published** · SAGED: **40 %**, MV + TP + OT | **Partially.** Error *types* match both; error *rate* matches REIN (8 % vs our 7.5 %) but SAGED's numbers are at 5× our error rate. Treat SAGED's table as an anchor, not a head-to-head. |
| **Smart Factory** | 86.6 %, all columns, blank cells | SAGED: **83 %** (very close) · REIN: 15.3 % | **Rate matches SAGED.** But our corruption is literally blank cells, so our F1 = 1.000 is trivial and uninformative either way. |

Two further warnings, both verified:
- **The same research group publishes different error rates for the same dataset.**
  Abdelaal et al. give Adult 0.58 in REIN (2023) and 0.09 in SAGED (2024); Breast Cancer
  0.08 then 0.40; Smart Factory 0.153 then 0.83. Cross-paper comparison is fragile even
  within one lab.
- **Label budget changes Raha's F1 by ~0.45** (§2.4). Any citation of "Raha F1" without
  its label budget is meaningless.

---

## 3. Factor decomposition — what actually carries the edge

Ranked by strength of the published evidence.

### 3.1 Conditional (row-context) modelling — **the dominant factor**

Predicting a cell from the rest of its row, rather than scoring it against its own
column's marginal distribution.

| Evidence | Source |
|---|---|
| On Adult, removing the **Neighborhood** representation causes the largest ablation drop of any component (full 0.971; variants span 0.971 → **0.884**, i.e. up to ~9 F1 points). "for Adult the highest drop is achieved when the Neighborhood representation is removed" | HoloDetect §6.3, Figure 3 |
| Marginal/statistical detection collapses on Adult: OD **F1 0.002**, FBI 0.405, vs learned 0.991 | HoloDetect Table 2 |
| "Excluding the vicinity-based error corrector models **significantly drops** the F1 score on datasets with high inter-column dependencies, such as Hospital and Flights." | Baran §6.3, Table 5 |
| Removing **correlated-attribute** computation is the single largest ablation loss in ZeroED: ≈ **−0.14 F1** on average, **−0.25** on Movies | ZeroED Table IV |
| AUG "models the actual data distribution and **not the side-effects of errors**" — explains why it holds up when errors are not constraint violations or outliers | HoloDetect §6.2 |
| CDC detects errors purely by per-column conditional models + conformal sets, and its corruption suite includes **value swaps between columns** and **categorical distribution shifts** — i.e. valid→valid errors | CDC §4.4, §5.1 |

**Read-across to us:** this is the only factor with direct evidence on
valid-value-to-valid-value corruption. It is R2, and it is correctly prioritised.

### 3.2 The first ~20 human labels — **large and cheap**

| Evidence | Source |
|---|---|
| **Raha: 2 labels ≈ 0.40 avg F1 → 20 labels 0.85 avg F1.** ~0.45 F1 for 18 extra labelled tuples. | 20-label figures: BiRNN Table 2. 2-label figures: ZeroED Table III + ForestED Table 3 |
| "Raha outperforms the state-of-the-art error detection techniques with **no more than 20 labeled tuples** on each dataset" | Raha abstract (SIGMOD 2019) |
| "The F1 score of Baran **quickly converges with only a few labeled tuples**" | Baran §6.2, Figure 3 |
| Wikipedia pretraining lets Baran reach its 20-label F1 with only **13 labels** (Hospital), **18** (Rayyan), **19** (Tax) | Baran §6.2 |
| Raha needs **> 20 labelled tuples** to match zero-human-label ZeroED, except on Flights and Beers | ZeroED Figure 6 |
| SAGED at a 20-label budget beats ED2 by 19.5 % and Raha by 28 % on Beers; ED2 only becomes comparable "when the labeling budget is relatively large" | SAGED §5.2 |
| REIN flags label dependence as a core *shortcoming*: ML detectors "require users intervention to label data" | REIN §7 |

**Read-across to us:** the cheapest large win available, and we currently use **zero**
labels for detection. 20 labelled cells is a trivial cost.

### 3.3 Data augmentation / synthetic corruption — **large, and it beats active learning**

| Evidence | Source |
|---|---|
| Augmentation gives "an average improvement of **20 F1 points** ... while it requires access to **3× fewer labeled** [examples]" | HoloDetect abstract / §1 |
| "Data Augmentation is **more effective than Active Learning** at capturing the heterogeneity of errors in each dataset, and hence, achieves superior recall to Active Learning **in all cases**" | HoloDetect §6.2 |
| Without augmentation, recall is the bottleneck: SuperL on Adult has P 0.999 but **R 0.350** | HoloDetect Table 2 |
| Augmentation policies are learned **weakly-supervised from the noisy input data** — no clean labels needed to learn them | HoloDetect §5 |

**Read-across to us:** we have a clean/dirty pair and a known corruption generator, so
synthesising label flips to train a supervised detector is low-cost. Must be done on a
split the detector never sees at inference (rule 1 of §3, research program).

### 3.4 Ensembling many weak signals — **real but second-order**

| Evidence | Source |
|---|---|
| "ML-based and ensemble methods, in most cases, have a **higher detection accuracy** than the other non-learning methods" | REIN §7 |
| Pure ensembles of weak rules win outright on some datasets: Min-k best on Smart Factory (**0.75**); Max-Entropy + dBoost best on Nasa (**0.85**) | REIN §6.2 |
| Metadata-driven turns each non-learning detector into **one binary feature** — the cheapest version of this idea | REIN §3.1 |
| SAGED's meta-classifier over pre-trained base models: Breast Cancer **0.95** vs best single baseline 0.70 | SAGED Table 2 |
| ForestED (ensemble of LLM-induced trees) beats single-tree ZeroED on 6 of 7 datasets | ForestED Table 3 |
| But ensembles inherit their members' blind spots: on Adult, HoloClean/NADEEF/Min-k all "captured most of the rule violations **only**" and dBoost "**failed to identify the rule violations**" | REIN §6.2 |

**Read-across to us:** R3 is worth running, but the last row is the warning — ensembling
six rules that are all blind to label flips cannot see label flips. **Ensembling only
helps once a conditional signal is in the pool.**

### 3.5 Representation learning vs feature engineering — **clear, in favour of learning**

| Evidence | Source |
|---|---|
| LR over co-occurrence + violation features gets **0.059** on Adult vs AUG **0.991**. "combining co-occurrence features and violations features in a linear way ... is **not enough** to capture the complex statistics of the dataset. This validates our choice of using representation learning and not engineered features." | HoloDetect §6.2, Table 2 |
| Removing any single representation model costs **up to 9 F1 points**; different models dominate on different datasets | HoloDetect §6.3 |
| Picket's self-supervised transformer needs **no labels at all** — but "is only suitable for small datasets" and **crashed with memory faults on Adult and Smart Factory** | REIN §7 |
| AimNet: attention over the other attributes beats ensemble baselines on 14 datasets, with **> 54 % lower runtime**, +3 % accuracy (MAR) and **+43 % (systematic missingness)** | AimNet, MLSys 2020 |

**Read-across to us:** gradient boosting per column (R2) is the cheap 80 % of this
factor. A transformer is not required, and Picket's scalability failure on our exact
datasets is a direct warning against that route.

### 3.6 Calibration of the detector score — **done, but never evaluated**

| Evidence | Source |
|---|---|
| HoloDetect **already calibrates** its classifier with **Platt scaling** on a holdout subset of the training data (§4.2), learning `q̂ᵢ = σ(a·zᵢ + b)`. | HoloDetect §4.2 |
| CDC uses **conformal prediction** — explicitly "a model-agnostic and distribution-free method to **quantify and calibrate** the uncertainty of ML models" — as its detection rule. | CDC abstract, §3 |
| ForestED produces per-cell posteriors (e.g. "a final error probability of 99.99 %") but **never checks whether they are calibrated**. | ForestED, case study |
| **No paper I read reports ECE, Brier score, or a reliability diagram for a detector.** | REIN, SAGED, HoloDetect, Baran, ZeroED, ForestED, CDC — all checked |

**Read-across to us:** "we calibrate the detector" is **not** a novelty claim —
HoloDetect did it in 2019. "We *measure* calibration quality, and then *act* on the
calibrated probability" is still open. See §4.

---

## 4. Gaps — what is underexplored

### 4.1 Calibration is applied but never measured — **open**
Platt scaling (HoloDetect) and conformal calibration (CDC) are both used as a *means* to
a binary decision. Neither paper reports a calibration metric. Across all seven systems
papers read in full, **ECE / Brier / reliability diagrams appear zero times**. Reporting
them for a detector is a genuine, if modest, contribution.

### 4.2 Cost-sensitive action selection — **open in data cleaning**
No data-cleaning paper selects among actions by expected cost. The closest findings:
- REIN shows the cost asymmetry *exists and flips with context*: "the detection
  **precision** has a relatively higher impact on the repair quality than the detection
  recall", **but** "**false negatives** in the detection phase become **more harmful**
  than false positives, in the presence of highly-effective repair methods" (§7).
  That is precisely a statement that the right operating point depends on costs — and
  REIN stops there, without a decision rule.
- **DMCO** (budget-aware co-optimisation of cleaning and AutoML) prioritises *which
  samples to clean* using model gradients and a multi-armed bandit. **[UNVERIFIED]** —
  the figure I found (">82 % of an exhaustive baseline's performance at a fraction of
  the budget") comes from a third-party blog summary, not the paper. Must be checked
  before citing.
- **SafeClean / privacy-aware cleaning-as-a-service** ([arXiv 2008.00368](https://arxiv.org/pdf/2008.00368))
  allocates a repair budget proportionally across iterations and picks repairs greedily
  (exact minimal-cost repair is NP-complete for a fixed FD set). This is budgeted
  *repair ordering*, not a per-cell action choice.
- A HILDA 2019 position paper ([PDF](https://hilda.io/2019/proceedings/HILDA2019_paper_10.pdf))
  treats human time as a limited resource and argues for identifying which factors
  deserve human feedback — a motivation for our layer, not an implementation of it.

**I found no paper that computes a per-cell expected cost over {repair, review, leave}
and selects the argmin.**

### 4.3 Abstention / selective prediction on data cleaning — **open**
Selective prediction is a **mature** field *outside* data cleaning, so our decision
layer is a **transfer**, not an invention:
- Chow's optimal reject rule is the classical origin; coverage and selective risk are
  standard quantities (**SelectiveNet**, Geifman & El-Yaniv, ICML 2019,
  [PMLR v97](http://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf)).
- **Optimal Strategies for Reject Option Classifiers**, JMLR 24
  ([PDF](https://jmlr.org/papers/volume24/21-0048/21-0048.pdf)) — learns an uncertainty
  score on top of a pre-trained classifier so the risk–coverage trade-off is set by
  tuning a threshold, no retraining.
- **Cost-aware conformal deferral already exists in another domain**: conformal selective
  prediction with cost-aware deferral for clinical triage (Nature Scientific Reports,
  2026, [PMC13022086](https://pmc.ncbi.nlm.nih.gov/articles/PMC13022086/)) chooses its
  deferral threshold **by minimising expected clinical cost**, and reports error on
  retained cases down **49.6 % (in-distribution)** and **46.7 % (under temporal shift)**
  at 80 % coverage. This is the closest published analogue to our decision layer —
  same machinery, different domain.

**No source applies risk–coverage, AURC, or a reject option to per-cell error detection
or repair.** Verified by two independent searches.

### 4.4 Conformal prediction for data repair — **partly taken (this is our main novelty risk)**
- **Conformal Data Cleaning** (AISTATS 2024) is the direct precedent: per-column
  AutoGluon models + conformal calibration, cell flagged if outside the prediction set.
  16 OpenML datasets, 4 error types (**column swaps**, random scaling, Gaussian noise,
  categorical distribution shift) at 1/5/10/30/50 % error fractions, 1,000 calibration
  points, 3 seeds. It reports **TPR and FPR only** — no precision, no F1, no PR-AUC.
  Results: CDC beats the ML baseline on TPR in ~43 % of experiments and on FPR in ~80 %;
  overall "in about 61 % of our experiments, CDC outperforms the baselines".
  **Its single knob is a confidence level, not a cost.** Its stated future work is
  iterative/multiple imputation and performance monitoring — **no mention of costs,
  abstention, or human routing.**
- **Conformal Prediction with Missing Values** (Zaffran et al., ICML 2023,
  [PMLR v202](https://proceedings.mlr.press/v202/zaffran23a/zaffran23a.pdf)): marginal
  coverage survives imputation for almost any imputation function, but coverage varies
  by missingness pattern, so conditional coverage can fail.
- **Conformal Prediction with Cellwise Outliers** ([arXiv 2505.04986](https://arxiv.org/pdf/2505.04986)):
  when test features themselves contain errors, exchangeability breaks; JDI-CP recovers
  only a weaker **1 − 2α** finite-sample guarantee. **Directly relevant** — this is our
  setting, and it says naive conformal guarantees do not hold on dirty inputs.

### 4.5 Verdict on our novelty — honest assessment

| Our claimed contribution | Status |
|---|---|
| Calibrating a probability with Platt/isotonic | **Not novel.** HoloDetect §4.2 (Platt), CDC (conformal). |
| Calibrating *repair-trust* P(repair correct) specifically | **Plausibly novel** — I found no paper that calibrates and evaluates a *repair-correctness* probability. Baran ensembles correction candidates with labels but reports no calibrated probability or calibration metric. |
| *Measuring* calibration quality (ECE/Brier) of a cleaning model | **Novel as far as I can verify** — zero occurrences across seven papers. |
| Expected-cost argmin over {auto-repair, human review, abstain} | **Novel within data cleaning**; **not novel as a framework** (Chow's rule, learning-to-defer, and cost-aware conformal deferral in clinical triage). Position as a *domain transfer*, not a new decision theory. |
| The same confidence yielding different actions as costs change | **Novel as a demonstrated result in data cleaning.** REIN's own precision-vs-recall asymmetry (§4.2) is the published motivation we can cite for why it matters. |
| Three-way action space including abstention | **Novel within data cleaning.** Every detector surveyed emits a binary per-cell decision; REIN explicitly lists "they only provide a **binary decision** for each cell" as a shortcoming of ML detectors (§7). |

**Bottom line:** the decision layer survives as a contribution, but only if framed as
*"selective prediction and cost-sensitive deferral, transferred to per-cell data
cleaning, where no prior system does anything but emit a binary flag"* — with Chow,
SelectiveNet, and the clinical-triage conformal-deferral paper cited as the machinery we
are importing. Framed as "we invented calibrated cost-sensitive decisions", it will be
rejected.

---

## 5. Metric practice

**What the field reports.** Verified by reading each paper's evaluation section:

| Paper | Metrics for detection | PR-AUC? |
|---|---|---|
| REIN (EDBT 2023) | precision, recall, F1, **IoU**, runtime | **No** |
| SAGED (EDBT 2024) | precision, recall, F1, runtime | **No** |
| HoloDetect (SIGMOD 2019) | precision, recall, F1 | **No** |
| Baran (PVLDB 2020) | precision, recall, F1, # labelled tuples, runtime | **No** |
| ZeroED (2025) | precision, recall, F1 + paired t-tests (p < 0.05) | **No** |
| ForestED (2025) | precision, recall, F1 ("widely adopted by existing error detection studies") | **No** |
| CDC (AISTATS 2024) | **TPR and FPR only** | **No** |

**PR-AUC appears in none of them.** A targeted search for PR-AUC / average precision in
tabular error detection returned only general class-imbalance methodology papers, no
error-detection work.

**Consequences for us — three concrete rules:**

1. **Report F1 at a stated operating point, always.** It is the only metric that makes
   our numbers legible to this literature. Our research program's "PR-AUC primary" rule
   is methodologically correct at 0.25 % prevalence but would make our tables
   incomparable to every published system. **Report both; lead with PR-AUC for our own
   claims, and always give F1/P/R alongside.**
2. **State the error rate and error types in every table caption.** The field does not,
   and that is exactly why Raha appears to score 0.85 and 0.40 for the same system
   (§2.4), and why Abdelaal et al. publish Adult at both 0.58 and 0.09 (§2.5). Our
   measured rates are in §2.5 and should appear in every results table.
3. **State the label budget.** HoloDetect uses a 5 % split; Raha/Baran/SAGED use 20
   labelled tuples; ZeroED/ForestED reproduce baselines at 2. A number without a budget
   is not a result.

**Protocol to match, if we want comparability.** The nearest thing to a standard is
SAGED's: *14 datasets, fixed budget of 20 user labels for every label-consuming method,
precision/recall/F1 + runtime, 10 repetitions, means reported.* REIN's is similar
(10 repetitions, different random seeds controlling the train/test split). Matching
**20 labels + 10 seeds + P/R/F1 + runtime** would make our detector directly quotable
next to SAGED Table 2 — including its Breast Cancer column, the one real anchor we have.
CDC's protocol (TPR/FPR, 3 seeds) is the one to *not* copy.

---

## 6. Recommendations — ranked by (expected gain / implementation cost)

### R-1. Per-column conditional model → calibrated surprise score  *(highest ratio)*
**Mechanism.** For each column *c*, train a gradient-boosted model predicting *c* from
the other columns, fit on dirty data only (self-supervised; no clean labels). Score each
cell by the model's surprise at the observed value: `1 − P̂(observed | rest of row)` for
categoricals, normalised residual for numerics.

**Evidence.** The only factor with direct evidence on valid→valid corruption: HoloDetect's
largest Adult ablation drop is the Neighborhood (row-context) representation (§3.1);
Baran's vicinity-based models are what fix inter-column violations; CDC detects exactly
this corruption class with precisely this architecture.

**Cost.** Low. LightGBM/sklearn, ~15 columns per dataset, no labels, no new interfaces.

**⚠️ Expected gain on Adult, stated honestly.** All 1,701 flips are in `income`, so
"predict the cell from the row" *is* the Adult classification task, whose published
ceiling is ≈ 85–87 % accuracy. A detector that flags every disagreement between
predicted and observed `income` will fire on ~13 % of 45,222 rows ≈ 5,900 cells to find
1,701 errors. **Rough derived ceiling: precision ≈ 0.2–0.3, recall ≈ 0.6–0.8,
F1 ≈ 0.3–0.4.** That is a large move from 0.000 and a genuine result — but it will
**not** reach HoloDetect's 0.991, because HoloDetect's Adult errors are 70 % typos
(syntactically visible) in attribute columns, not flips of the target. **Do not promise
0.9 on Adult.** Report it as PR-AUC and precision@k, where a calibrated conditional
model should look much better than its thresholded F1 suggests.

### R-2. Rank, don't threshold: PR-AUC + precision@k + calibrated margin  *(very high ratio)*
**Mechanism.** Do not convert the surprise score to a binary flag with a fixed cutoff.
Rank cells by calibrated P(corrupted) and report PR-AUC, precision@k, and a
risk–coverage curve. Use Platt or isotonic on a held-out split, exactly as HoloDetect
calibrates with Platt on a training holdout.

**Evidence.** At 0.25 % prevalence a thresholded F1 is dominated by the threshold, not
the signal (our own research program §3). HoloDetect's own scores are calibrated before
use (§4.2). The JMLR reject-option paper shows the risk–coverage trade-off can be tuned
by threshold on an uncertainty score without retraining.

**Cost.** Very low — this is evaluation code plus a calibrator we already have
(Phase 6 `calibrator.py` already implements Platt and isotonic).

**Gain.** Turns R-1's mediocre F1 into a defensible result, and it is the natural bridge
to the decision layer. This is also the honest answer to "why is our F1 still low."

### R-3. Add a 20-label budget + meta-classifier over all signals  *(high ratio)*
**Mechanism.** Feature-vector per cell = the six existing rule outputs (one binary
feature each, as Metadata-driven does) + conditional surprise from R-1 + column
frequency + row-level outlierness. Label 20 tuples. Train a classifier.

**Evidence.** The single largest documented jump in this literature: **Raha goes from
≈ 0.40 F1 at 2 labels to 0.85 at 20** (§3.2). Baran converges "with only a few labeled
tuples". SAGED beats Raha by 28 % at the same 20-label budget. REIN: ensemble and
ML-based methods beat non-learning ones in most cases.

**Cost.** Low-moderate. Needs a labelling interface (or simulated oracle from `clean.csv`
on a split the detector never sees at inference — and this must be done carefully, see
below).

**⚠️ Leakage discipline.** Simulating the oracle from `clean.csv` is how Raha, ED2 and
SAGED are all benchmarked, so it is accepted practice — but it must be 20 labels on a
*training* split, never on the evaluation split, and the §3 rule-1 statement of exactly
what was fitted on what is mandatory. REIN notes this simulation breaks entirely when
dirty and clean versions differ in length.

### R-4. Synthetic corruption augmentation  *(high gain, moderate cost)*
**Mechanism.** Generate synthetic label flips (and typos, swaps) on the presumed-clean
majority of the dirty data, then train a supervised detector on the synthetic labels —
HoloDetect's weak-supervision recipe, minus the learned policy.

**Evidence.** **+20 F1 points average, with 3× fewer labels**, and augmentation beats
active learning on recall **in all cases** (§3.3). Directly addresses the recall
bottleneck we have (our Adult recall is 0.000; HoloDetect's SuperL recall is 0.350
without augmentation).

**Cost.** Moderate — needs a corruption generator and care that it does not simply
memorise our own injector. If we synthesise the same flips our injector made, the result
is circular. Use a *different* flip policy than the one that made `dirty.csv`, and say so.

### R-5. Reframe the decision layer as selective prediction, and measure it  *(low cost, high novelty yield)*
**Mechanism.** Keep the three-way argmin, but add: risk–coverage curves, AURC, and cost
*sweeps* rather than three fixed cost constants. Report ECE/Brier for both D (detection)
and R (repair-trust). Cite Chow / SelectiveNet / JMLR reject-option for the machinery and
the clinical-triage conformal-deferral paper as the closest cross-domain analogue.

**Evidence.** §4.5: this is where our surviving novelty actually is. REIN's own
"precision matters more / FNs matter more with a strong repairer" asymmetry (§4.2) is the
published justification that the operating point should be cost-dependent. And no
detector in the survey emits anything but a binary flag — REIN lists that as a
shortcoming.

**Cost.** Low. Mostly evaluation and framing; Phases 7–8 already compute the argmin.

**Gain.** This is the difference between "we built a pipeline" and "we have a
contribution". It also fixes the §5.2 problem in CLAUDE.md (ABSTAIN unreachable):
sweeping costs instead of fixing them makes all three arms reachable by construction,
without the p-dependent-abstention change that would invalidate reported numbers.

### Not recommended
- **Transformer / Picket-style self-supervision.** REIN reports Picket **crashed with
  memory faults on Adult and Smart Factory** — our two non-trivial datasets — and is
  "only suitable for small datasets". Poor ratio.
- **More hand-written rules.** HoloClean's Adult F1 falls 0.51 → 0.12 when rules drop
  from 17 to 7; the ceiling is the rule author, and no rule set sees valid→valid flips.
- **Chasing HoloDetect's 0.991 on Adult.** Different corruption (§2.5). Not our number
  to reach.

---

## 7. Sources

Read in full (text extracted and quoted above):
- Abdelaal, Hammacher, Schöning. **REIN: A Comprehensive Benchmark Framework for Data Cleaning Methods in ML Pipelines.** EDBT 2023, pp. 499–511. <https://openproceedings.org/2023/conf/edbt/3-paper-49.pdf>
- Abdelaal, Ktitarev, Städtler, Schöning. **SAGED: Few-Shot Meta Learning for Tabular Data Error Detection.** EDBT 2024. <https://openproceedings.org/2024/conf/edbt/paper-95.pdf>
- Heidari, McGrath, Ilyas, Rekatsinas. **HoloDetect: Few-Shot Learning for Error Detection.** SIGMOD 2019. <http://www.holoclean.io/files/misc/detection.pdf> · <https://arxiv.org/abs/1904.02285>
- Mahdavi, Abedjan. **Baran: Effective Error Correction via a Unified Context Representation and Transfer Learning.** PVLDB 13(12):1948–1961, 2020. <http://www.vldb.org/pvldb/vol13/p1948-mahdavi.pdf>
- Jäger, Bießmann. **From Data Imputation to Data Cleaning — Automated Cleaning of Tabular Data Improves Downstream Predictive Performance.** AISTATS 2024, PMLR v238. <https://proceedings.mlr.press/v238/jager24a/jager24a.pdf>
- **Detecting Errors in Databases with Bidirectional Recurrent Neural Networks.** EDBT 2022. <https://openproceedings.org/2022/conf/edbt/paper-40.pdf>

Read via structured extraction (tables and sections quoted):
- **ZeroED: Hybrid Zero-shot Error Detection through Large Language Model Reasoning.** <https://arxiv.org/abs/2504.05345>
- **Ensembling LLM-Induced Decision Trees for Explainable and Robust Error Detection** (ForestED). <https://arxiv.org/abs/2512.07246>

Cited at abstract / secondary level (mechanism and supervision only, no numbers claimed):
- Mahdavi et al. **Raha: A Configuration-Free Error Detection System.** SIGMOD 2019, pp. 865–882. DOI [10.1145/3299869.3324956](https://doi.org/10.1145/3299869.3324956) — ACM and Semantic Scholar both returned HTTP 403; abstract text confirmed via search, clustering/label-propagation detail cited to REIN §3.1 rather than to Raha itself.
- Wang, He. **Uni-Detect: A Unified Approach to Automated Error Detection in Tables.** SIGMOD 2019, pp. 811–828. DOI [10.1145/3299869.3319855](https://doi.org/10.1145/3299869.3319855)
- Liu, Zhou, Rekatsinas. **Picket: guarding against corrupted data in tabular data during learning and inference.** VLDB J 31(5):927–955, 2022. <https://arxiv.org/abs/2006.04730>
- Wu et al. **Attention-Based Learning for Missing Data Imputation in HoloClean** (AimNet). MLSys 2020. <https://cs.uwaterloo.ca/~ilyas/papers/WuMLSys2020.pdf>
- Rekatsinas et al. **HoloClean: Holistic Data Repairs with Probabilistic Inference.** PVLDB 10(11), 2017.
- Dallachiesa et al. **NADEEF: a commodity data cleaning system.** SIGMOD 2013. DOI [10.1145/2463676.2465327](https://doi.org/10.1145/2463676.2465327)

Selective prediction / conformal (the machinery our decision layer imports):
- Geifman, El-Yaniv. **SelectiveNet: A Deep Neural Network with an Integrated Reject Option.** ICML 2019, PMLR v97. <http://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf>
- **Optimal Strategies for Reject Option Classifiers.** JMLR 24. <https://jmlr.org/papers/volume24/21-0048/21-0048.pdf>
- **Conformal selective prediction with cost-aware deferral for safe clinical triage under distribution shift.** Nature Scientific Reports, 2026. <https://pmc.ncbi.nlm.nih.gov/articles/PMC13022086/>
- Zaffran et al. **Conformal Prediction with Missing Values.** ICML 2023, PMLR v202. <https://proceedings.mlr.press/v202/zaffran23a/zaffran23a.pdf>
- **Conformal Prediction with Cellwise Outliers: A Detect-then-Impute Approach.** <https://arxiv.org/pdf/2505.04986>
- **Privacy-Aware Data Cleaning-as-a-Service.** <https://arxiv.org/pdf/2008.00368>
- **Towards an End-to-End Human-Centric Data Cleaning Framework.** HILDA 2019. <https://hilda.io/2019/proceedings/HILDA2019_paper_10.pdf>

### Flagged as unverified
- **DMCO** (budget-aware co-optimisation of data cleaning and AutoML): the ">82 % of an
  exhaustive baseline at a fraction of the budget" figure comes from a third-party blog
  summary, not the paper. **Do not cite until the primary source is read.**
- **Baran's "+23 average F1 points over Raha and HoloClean"**: seen on a ResearchGate
  summary page only; **not found** in the paper text I extracted. Do not cite.
- **Breast Cancer KATARA P/R/F1 in REIN** (≈ 0.079 / 0.872 / 0.145): *derived* by me from
  REIN's stated 6,843 FP / 86 FN and the 8 % error rate. Not a published figure.
- **Our own Adult conditional-detector ceiling (F1 ≈ 0.3–0.4, R-1)**: *derived* from the
  published ~85–87 % Adult classification accuracy and our measured 1,701 flips. It is an
  order-of-magnitude argument, not a measurement. **R0's harness must measure it.**
- Raha's clustering / label-propagation mechanism is described as REIN characterises it;
  the Raha paper itself was unreachable (403).
